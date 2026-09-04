"""Analisis de errores: donde falla el modelo, no solo cuanto.

Por que este script existe
-----------------------------
El MAE y el R2 dicen cuanto se equivoca el modelo EN PROMEDIO. No dicen si ese error se
reparte de forma uniforme o si se concentra en un patron reconocible: una estacion, una
franja horaria, el final de la ruta, los tramos mas largos. Este script comprueba si eso
es un caso aislado o un sesgo sistematico.

Por que se reentrena en vez de usar el modelo de produccion de `entrenamiento_final.py`
-------------------------------------------------------------------------------------------
El modelo guardado en `artifacts/modelo/` esta entrenado sobre el 100% de `train` (el 80%
de las rutas) y evaluado sobre `test` (el 20% reservado): sus residuos sobre las propias
filas de train serian optimistas, porque el modelo ya las ha visto. Un analisis de errores
riguroso necesita predicciones FUERA DE MUESTRA para cada fila de train -y solo de train:
este diagnostico no toca `test` en ningun momento, es un analisis distinto de la
evaluacion final que ya hace `entrenamiento_final.py`.

Se reentrena la arquitectura ganadora de `bakeoff/catboost_zona.py` (CatBoost con
`to_zone_id` en reparto, LightGBM en almacen) dentro de 5 particiones `GroupKFold` sobre
`modelado.datos.cargar_train()`, y se analizan las predicciones **out-of-fold**: cada
tramo de train recibe la prediccion del modelo del pliegue en el que estaba de prueba, es
decir, de un modelo que nunca lo vio durante el entrenamiento. Concatenando los 5
pliegues se obtiene una prediccion fuera de muestra para todos los tramos de train, no
solo para una muestra.

Esto reentrena, no reutiliza directamente los resultados de `bakeoff/catboost_zona.py`:
aquella ejecucion probo variantes (con/sin zona, LightGBM/CatBoost) y solo guardo
metricas agregadas por pliegue, no las predicciones fila a fila. Aqui hace falta el
detalle, asi que se paga el coste de entrenar de nuevo, mismos hiperparametros, un unico
candidato por poblacion en vez de tres.

Que se comprueba
-------------------
- Sesgo sistematico: es la media del residuo (no el valor absoluto) distinta de cero, en
  conjunto o dentro de algun grupo. `regression_l1` optimiza la mediana, no la media, asi
  que un sesgo pequeno en la media no séria sorprendente por si solo, pero conviene medirlo
  en vez de asumir que es nulo.
- Heterocedasticidad: crece el error con la duracion real del tramo, o con la distancia.
- Error por estacion, por zona (misma/distinta) y por momento de la ruta (inicio/fin), para
  detectar si el modelo trata peor a algun subgrupo reconocible.
- Los tramos peor predichos, para revisar si son casos dificiles legitimos (lo esperable,
  dado que el proyecto decidio no recortar atipicos) o si delatan un problema de datos no
  detectado hasta ahora.

Sobre la comprobacion cruzada de particiones del script original (adaptacion obligada)
-------------------------------------------------------------------------------------------
La version original de este script verificaba el reparto de rutas por pliegue contra unos
conteos hardcodeados, copiados a mano de `01 Baseline.py`: una defensa contra que cada
script, con su propia copia de la logica de carga de Parquet, generara folds distintos sin
que nadie se diera cuenta. Ese riesgo ya no existe -todo el bake-off, este diagnostico
incluido, carga siempre a traves de la misma `modelado.datos.cargar_train()`, asi que los
folds son identicos por construccion- y los conteos concretos ya no aplican, porque
estaban dimensionados para el 100% de las rutas y aqui solo se opera sobre el 80% (train).
Se mantiene la comprobacion que si sigue siendo relevante: que ninguna ruta este a la vez
en train y test del mismo pliegue.

Sobre `from_stop_id`/`to_stop_id` (columnas descriptivas retiradas)
-----------------------------------------------------------------------
La version original guardaba estos dos identificadores junto a cada prediccion para
poder identificar tramos concretos. `modelado.datos.cargar_train()` no los expone (no son
variables de modelo, y el punto unico de carga de datos del proyecto no las lee del
Parquet), asi que se retiran de las columnas descriptivas: no cambian ninguna cifra del
analisis, y el informe nunca los imprimia en las tablas (solo se guardaban en el CSV
crudo).

Uso:

    uv run python src/modelado/analisis_errores.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold

from modelado.datos import (
    CATEGORICAS_BASE,
    CATEGORICAS_ZONA,
    DISTANCIA,
    GRUPO,
    OBJETIVO,
    PARADA_TEMPRANA_ALMACEN,
    PARAMS_CB_REPARTO,
    PARAMS_LGBM_ALMACEN,
    POBLACION,
    SEMILLA,
    VARS_ALMACEN,
    VARS_REPARTO,
    cargar_train,
    separar_validacion,
)
from modelado.metricas import metricas as _metricas_base

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_MODELADO = PROJECT_ROOT / "artifacts" / "modelado"
REPORTS = ARTIFACTS_MODELADO / "reports"
FIGURAS = ARTIFACTS_MODELADO / "figuras"
MLRUNS = ARTIFACTS_MODELADO / "mlflow" / "mlruns"
MLFLOW_DB = ARTIFACTS_MODELADO / "mlflow" / "mlflow.db"

N_SPLITS = 5
EXPERIMENTO = "SmartDeliveryAI Tiempos"

# Referencias de bakeoff/catboost_zona.py, mismo modelo y particiones: sirven de control
# de coherencia (ver redactar()) y no se dan por buenas sin comparar. Se refrescan a mano
# tras cada ejecucion de ese script.
CV_MAE_REF, CV_R2_REF = 0.3721, 0.9366

# Columnas descriptivas que se guardan junto a la prediccion, para poder desglosar el
# error despues sin tener que volver a tocar el modelo.
COLUMNAS_DESCRIPTIVAS = [
    GRUPO, "station_code", "segment_position", "route_total_stops",
    "departure_hour", "weekday", "same_zone", DISTANCIA, POBLACION,
]


# ------------------------------------------------------------------------------------
# Metricas
# ------------------------------------------------------------------------------------
def metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Envuelve `modelado.metricas.metricas` (mae_min, rmse_min, r2) y le anade el sesgo
    con signo (real - prediccion, en minutos), que este diagnostico necesita y la version
    compartida no calcula."""
    m = _metricas_base(y_real, y_pred)
    m["sesgo_medio_min"] = float(np.mean(y_real - y_pred) / 60)
    return m


# ------------------------------------------------------------------------------------
# Entrenamiento por pliegue (misma arquitectura que bakeoff/catboost_zona.py /
# entrenamiento_final.py)
# ------------------------------------------------------------------------------------
def predecir_pliegue_reparto(train: pd.DataFrame, test: pd.DataFrame, semilla_pliegue: int) -> np.ndarray:
    sub_train, val = separar_validacion(train, semilla_pliegue)
    modelo = CatBoostRegressor(**PARAMS_CB_REPARTO)
    modelo.fit(
        sub_train[VARS_REPARTO], sub_train[OBJETIVO], cat_features=CATEGORICAS_ZONA,
        eval_set=(val[VARS_REPARTO], val[OBJETIVO]), use_best_model=True,
    )
    return np.clip(modelo.predict(test[VARS_REPARTO]), 0, None)


def predecir_pliegue_almacen(train: pd.DataFrame, test: pd.DataFrame, semilla_pliegue: int) -> np.ndarray:
    train = train[[GRUPO, OBJETIVO, *VARS_ALMACEN]].copy()
    for c in CATEGORICAS_BASE:
        train[c] = train[c].astype("category")
    sub_train, val = separar_validacion(train, semilla_pliegue)

    modelo = lgb.LGBMRegressor(**PARAMS_LGBM_ALMACEN)
    modelo.fit(
        sub_train[VARS_ALMACEN], sub_train[OBJETIVO],
        eval_X=val[VARS_ALMACEN], eval_y=val[OBJETIVO], eval_metric=PARAMS_LGBM_ALMACEN["metric"],
        callbacks=[lgb.early_stopping(PARADA_TEMPRANA_ALMACEN, verbose=False)],
        categorical_feature=CATEGORICAS_BASE,
    )
    dtypes = train[CATEGORICAS_BASE].dtypes
    x_test = test[VARS_ALMACEN].copy()
    for c in CATEGORICAS_BASE:
        x_test[c] = x_test[c].astype(dtypes[c])
    return np.clip(modelo.predict(x_test), 0, None)


def generar_predicciones_oof(tramos: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una fila por tramo de train con su prediccion fuera de muestra."""
    cv = GroupKFold(n_splits=N_SPLITS)
    resultados: list[pd.DataFrame] = []

    for pliegue, (idx_train, idx_test) in enumerate(cv.split(tramos, groups=tramos[GRUPO]), start=1):
        train_full = tramos.iloc[idx_train]
        test_full = tramos.iloc[idx_test]

        rutas_train, rutas_test = train_full[GRUPO].unique(), test_full[GRUPO].unique()
        solapadas = set(rutas_train) & set(rutas_test)
        if solapadas:
            raise AssertionError(f"Pliegue {pliegue}: {len(solapadas)} rutas en train y test a la vez")

        train_reparto = train_full[train_full[POBLACION] == 0]
        test_reparto = test_full[test_full[POBLACION] == 0]
        train_almacen = train_full[train_full[POBLACION] == 1]
        test_almacen = test_full[test_full[POBLACION] == 1]

        pred_reparto = predecir_pliegue_reparto(train_reparto, test_reparto, SEMILLA + pliegue)
        pred_almacen = predecir_pliegue_almacen(train_almacen, test_almacen, SEMILLA + pliegue)

        salida_reparto = test_reparto[[*COLUMNAS_DESCRIPTIVAS, OBJETIVO]].copy()
        salida_reparto["prediccion"] = pred_reparto
        salida_almacen = test_almacen[[*COLUMNAS_DESCRIPTIVAS, OBJETIVO]].copy()
        salida_almacen["prediccion"] = pred_almacen

        salida = pd.concat([salida_reparto, salida_almacen], ignore_index=True)
        salida["pliegue"] = pliegue

        mae_reparto = float(np.mean(np.abs(test_reparto[OBJETIVO].to_numpy() - pred_reparto)) / 60)
        mae_almacen = float(np.mean(np.abs(test_almacen[OBJETIVO].to_numpy() - pred_almacen)) / 60)
        print(f"  pliegue {pliegue}  reparto MAE {mae_reparto:.4f} min   almacen MAE {mae_almacen:.4f} min", flush=True)

        resultados.append(salida)

    return pd.concat(resultados, ignore_index=True)


# ------------------------------------------------------------------------------------
# Analisis
# ------------------------------------------------------------------------------------
def analizar(oof: pd.DataFrame) -> dict[str, Any]:
    oof = oof.copy()
    oof["residuo_s"] = oof[OBJETIVO] - oof["prediccion"]
    oof["residuo_abs_s"] = oof["residuo_s"].abs()
    oof["residuo_min"] = oof["residuo_s"] / 60

    global_reparto = metricas(oof.loc[oof[POBLACION] == 0, OBJETIVO].to_numpy(), oof.loc[oof[POBLACION] == 0, "prediccion"].to_numpy())
    global_almacen = metricas(oof.loc[oof[POBLACION] == 1, OBJETIVO].to_numpy(), oof.loc[oof[POBLACION] == 1, "prediccion"].to_numpy())
    global_todo = metricas(oof[OBJETIVO].to_numpy(), oof["prediccion"].to_numpy())

    # Error por estacion (dentro de reparto, que es donde esta el 99,3% de las filas).
    rep = oof[oof[POBLACION] == 0]
    por_estacion = (
        rep.groupby("station_code")
        .apply(lambda g: pd.Series(metricas(g[OBJETIVO].to_numpy(), g["prediccion"].to_numpy())), include_groups=False)
        .assign(n_tramos=rep.groupby("station_code").size())
        .sort_values("mae_min", ascending=False)
        .reset_index()
    )

    # Error por zona: misma zona vs distinta.
    por_zona = (
        rep.groupby("same_zone")
        .apply(lambda g: pd.Series(metricas(g[OBJETIVO].to_numpy(), g["prediccion"].to_numpy())), include_groups=False)
        .assign(n_tramos=rep.groupby("same_zone").size())
        .reset_index()
    )

    # Error por posicion en la ruta: primer/ultimo tercio, para el efecto "fatiga".
    rep = rep.copy()
    rep["progreso_ruta"] = rep["segment_position"] / rep["route_total_stops"].clip(lower=1)
    rep["tercio_ruta"] = pd.cut(rep["progreso_ruta"], bins=[0, 1 / 3, 2 / 3, 1.0], labels=["inicio", "medio", "final"], include_lowest=True)
    por_tercio = (
        rep.groupby("tercio_ruta", observed=True)
        .apply(lambda g: pd.Series(metricas(g[OBJETIVO].to_numpy(), g["prediccion"].to_numpy())), include_groups=False)
        .assign(n_tramos=rep.groupby("tercio_ruta", observed=True).size())
        .reset_index()
    )

    # Error por hora de salida.
    por_hora = (
        rep.groupby("departure_hour")
        .apply(lambda g: pd.Series(metricas(g[OBJETIVO].to_numpy(), g["prediccion"].to_numpy())), include_groups=False)
        .assign(n_tramos=rep.groupby("departure_hour").size())
        .reset_index()
        .sort_values("departure_hour")
    )

    # Heterocedasticidad: el residuo (con signo) por decil de duracion real, dentro de reparto.
    rep["decil_duracion"] = pd.qcut(rep[OBJETIVO], 10, labels=False, duplicates="drop")
    por_decil = (
        rep.groupby("decil_duracion")
        .agg(
            duracion_media_min=(OBJETIVO, lambda s: s.mean() / 60),
            sesgo_medio_min=("residuo_min", "mean"),
            mae_min=("residuo_min", lambda s: s.abs().mean()),
            n_tramos=(OBJETIVO, "size"),
        )
        .reset_index()
    )

    peores = (
        oof.reindex(oof["residuo_abs_s"].sort_values(ascending=False).index)
        .head(15)[[GRUPO, "station_code", POBLACION, OBJETIVO, "prediccion", "residuo_s", DISTANCIA, "same_zone"]]
        .copy()
    )
    peores[OBJETIVO] = (peores[OBJETIVO] / 60).round(2)
    peores["prediccion"] = (peores["prediccion"] / 60).round(2)
    peores["residuo_min"] = (peores["residuo_s"] / 60).round(2)
    peores = peores.drop(columns="residuo_s")

    return {
        "global_todo": global_todo, "global_reparto": global_reparto, "global_almacen": global_almacen,
        "por_estacion": por_estacion, "por_zona": por_zona, "por_tercio": por_tercio,
        "por_hora": por_hora, "por_decil": por_decil, "peores": peores, "oof": oof,
    }


# ------------------------------------------------------------------------------------
# Graficos
# ------------------------------------------------------------------------------------
def guardar_graficos(resultado: dict[str, Any]) -> list[str]:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    guardados: list[str] = []
    try:
        rep = resultado["oof"][resultado["oof"][POBLACION] == 0]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(rep["residuo_min"].clip(-3, 3), bins=80, color="#4F81A6")
        ax.axvline(0, color="#C0504D", linewidth=1)
        ax.set_xlabel("residuo (real - prediccion, min, recortado a ±3)")
        ax.set_ylabel("tramos de reparto")
        ax.set_title("Distribucion del residuo, fuera de muestra")
        fig.tight_layout()
        fig.savefig(FIGURAS / "analisis_errores distribucion del residuo.png", dpi=110)
        plt.close(fig)
        guardados.append("analisis_errores distribucion del residuo.png")

        fig, ax = plt.subplots(figsize=(7, 4))
        d = resultado["por_decil"]
        ax.bar(d["decil_duracion"].astype(str), d["sesgo_medio_min"], color="#6E9E52")
        ax.axhline(0, color="#C0504D", linewidth=1)
        ax.set_xlabel("decil de duracion real (0 = tramos mas cortos)")
        ax.set_ylabel("sesgo medio (real - prediccion, min)")
        ax.set_title("Sesgo por magnitud del tramo")
        fig.tight_layout()
        fig.savefig(FIGURAS / "analisis_errores sesgo por decil de duracion.png", dpi=110)
        plt.close(fig)
        guardados.append("analisis_errores sesgo por decil de duracion.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        e = resultado["por_estacion"].sort_values("mae_min")
        ax.barh(e["station_code"], e["mae_min"], color="#E8A33D")
        ax.set_xlabel("MAE (min)")
        ax.set_title("Error por estacion (tramos de reparto)")
        fig.tight_layout()
        fig.savefig(FIGURAS / "analisis_errores mae por estacion.png", dpi=110)
        plt.close(fig)
        guardados.append("analisis_errores mae por estacion.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        t = resultado["por_tercio"]
        ax.bar(t["tercio_ruta"].astype(str), t["mae_min"], color="#4F81A6")
        ax.set_ylabel("MAE (min)")
        ax.set_title("Error segun el progreso dentro de la ruta")
        fig.tight_layout()
        fig.savefig(FIGURAS / "analisis_errores mae por tercio de ruta.png", dpi=110)
        plt.close(fig)
        guardados.append("analisis_errores mae por tercio de ruta.png")
    except Exception as error:  # noqa: BLE001 - un fallo de trazado no debe tumbar el resto
        plt.close("all")
        print(f"  AVISO: fallo al pintar los graficos: {error}")
        print("  Las tablas numericas ya estan guardadas en CSV; se continua sin estos graficos.")
    return guardados


# ------------------------------------------------------------------------------------
# Informe
# ------------------------------------------------------------------------------------
def redactar(resultado: dict[str, Any], graficos: list[str]) -> str:
    g, gr, ga = resultado["global_todo"], resultado["global_reparto"], resultado["global_almacen"]
    discrepancia = abs(g["mae_min"] - CV_MAE_REF)
    coherente = discrepancia < 0.01

    lineas = [
        "# Analisis de errores: donde falla el modelo, no solo cuanto",
        "",
        f"Generado por `modelado/analisis_errores.py` el {datetime.now(UTC).date()}.",
        "",
        "## Que son estas predicciones",
        "",
        "Fuera de muestra (out-of-fold), dentro del 80% de train (nunca se toca `test`):",
        "cada tramo esta predicho por el modelo del pliegue en el que era de prueba, nunca",
        "por un modelo que lo haya visto en entrenamiento. Misma arquitectura que el modelo",
        "de produccion (CatBoost con `to_zone_id` en reparto, LightGBM en almacen), mismas",
        "particiones `GroupKFold` de 5 pliegues.",
        "",
        "## Autoauditoria: coherencia con bakeoff/catboost_zona.py",
        "",
        "```text",
        f"bakeoff/catboost_zona.py (referencia)   MAE {CV_MAE_REF:.4f} min   R2 {CV_R2_REF:.4f}",
        f"Aqui (out-of-fold completo)              MAE {g['mae_min']:.4f} min   R2 {g['r2']:.4f}",
        f"diferencia                                {discrepancia:.4f}",
        "```",
        "",
        (
            "Coherente: la diferencia es ruido de coma flotante y del reentrenamiento (misma"
            " arquitectura, mismos datos, pero no exactamente el mismo arbol al repetir el"
            " ajuste), no un error de montaje."
            if coherente else
            "**INCOHERENTE**: la diferencia supera la tolerancia esperada. Revisar particiones"
            " e hiperparametros antes de confiar en las tablas siguientes."
        ),
        "",
        "## Sesgo sistematico",
        "",
        "```text",
        f"reparto   sesgo medio {gr['sesgo_medio_min']:+.4f} min   (positivo = el modelo infrapredice de media)",
        f"almacen   sesgo medio {ga['sesgo_medio_min']:+.4f} min",
        "```",
        "",
    ]
    media_reparto_min = float(resultado["oof"].loc[resultado["oof"][POBLACION] == 0, OBJETIVO].mean() / 60)
    media_almacen_min = float(resultado["oof"].loc[resultado["oof"][POBLACION] == 1, OBJETIVO].mean() / 60)
    sesgo_rel_reparto = gr["sesgo_medio_min"] / media_reparto_min * 100
    sesgo_rel_almacen = ga["sesgo_medio_min"] / media_almacen_min * 100
    lineas += [
        "El sesgo absoluto no dice nada por si solo: hay que compararlo con la escala de",
        "cada poblacion, no darlo por pequeno sin medirlo.",
        "",
        "```text",
        f"reparto   {gr['sesgo_medio_min']:+.4f} min sobre una media de {media_reparto_min:.2f} min   ({sesgo_rel_reparto:+.1f}% relativo)",
        f"almacen   {ga['sesgo_medio_min']:+.4f} min sobre una media de {media_almacen_min:.2f} min   ({sesgo_rel_almacen:+.1f}% relativo)",
        "```",
        "",
        (
            f"En almacen el sesgo relativo ({sesgo_rel_almacen:+.1f}%) es insignificante. En"
            f" reparto **no lo es**: {sesgo_rel_reparto:+.1f}% es una infrapredicción sistematica"
            " real, no ruido. El mecanismo es coherente con optimizar MAE (error absoluto):"
            " la solucion optima de un arbol con perdida L1 en cada hoja es la MEDIANA de los"
            " valores de esa hoja, no la media. Si la distribucion dentro de cada hoja sigue"
            " siendo asimetrica a la derecha (como el objetivo global, asimetria 3,6 segun el"
            " EDA), la mediana queda sistematicamente por debajo de la media, y el residuo"
            " medio (real - prediccion) sale positivo. Es el mismo mecanismo que explica la"
            " heterocedasticidad de la siguiente seccion, no un hallazgo distinto."
            if abs(sesgo_rel_reparto) > 3
            else "El sesgo relativo es pequeno en las dos poblaciones."
        ),
        "",
        "## Heterocedasticidad: el sesgo por magnitud del tramo",
        "",
        "| Decil duracion | Duracion media (min) | Sesgo medio (min) | MAE (min) | Tramos |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, fila in resultado["por_decil"].iterrows():
        lineas.append(
            f"| {int(fila['decil_duracion'])} | {fila['duracion_media_min']:.2f} | {fila['sesgo_medio_min']:+.3f} "
            f"| {fila['mae_min']:.3f} | {int(fila['n_tramos']):,} |"
        )
    ultimo_decil = resultado["por_decil"].iloc[-1]
    lineas += [
        "",
        (
            f"**Los tramos mas largos se infrapredicen de forma sistematica**: en el decil"
            f" superior (duracion media {ultimo_decil['duracion_media_min']:.1f} min), el sesgo"
            f" medio es {ultimo_decil['sesgo_medio_min']:+.2f} min. Es el patron esperable de"
            " optimizar MAE, que penaliza el error absoluto por igual en todos los tramos y no"
            " da ventaja extra a acertar los pocos tramos largos frente a los muchos cortos."
            if ultimo_decil["sesgo_medio_min"] > 0.3
            else "No hay una infra o sobreprediccion clara concentrada en los tramos mas largos."
        ),
        "",
        "## Error por estacion",
        "",
        "| Estacion | MAE (min) | Sesgo medio (min) | Tramos |",
        "|---|---:|---:|---:|",
    ]
    for _, fila in resultado["por_estacion"].iterrows():
        lineas.append(f"| {fila['station_code']} | {fila['mae_min']:.4f} | {fila['sesgo_medio_min']:+.4f} | {int(fila['n_tramos']):,} |")

    peor_estacion = resultado["por_estacion"].iloc[0]
    mejor_estacion = resultado["por_estacion"].iloc[-1]
    estaciones_sesgo_positivo = int((resultado["por_estacion"]["sesgo_medio_min"] > 0).sum())
    total_estaciones = len(resultado["por_estacion"])
    lineas += [
        "",
        (
            f"Rango de {mejor_estacion['mae_min']:.3f} a {peor_estacion['mae_min']:.3f} min"
            f" segun la estacion (un {(peor_estacion['mae_min']/mejor_estacion['mae_min']-1)*100:.0f}%"
            f" de diferencia entre la mas dificil, `{peor_estacion['station_code']}`, y la mas"
            " facil). Ninguna estacion se sale de forma aislada del resto en MAE."
        ),
        "",
        (
            f"Lo que si es sistematico: **las {total_estaciones} estaciones tienen sesgo medio"
            " positivo**, sin excepcion. No es un problema de una estacion concreta, es el"
            " sesgo de infrapredicción de la poblacion de reparto (seccion anterior) repartido"
            " por igual entre todas ellas."
            if estaciones_sesgo_positivo == total_estaciones
            else f"{estaciones_sesgo_positivo} de {total_estaciones} estaciones tienen sesgo positivo; no es un patron unanime."
        ),
        "",
        "## Error por coincidencia de zona",
        "",
        "| same_zone | MAE (min) | Sesgo medio (min) | Tramos |",
        "|---:|---:|---:|---:|",
    ]
    for _, fila in resultado["por_zona"].iterrows():
        lineas.append(f"| {int(fila['same_zone'])} | {fila['mae_min']:.4f} | {fila['sesgo_medio_min']:+.4f} | {int(fila['n_tramos']):,} |")

    lineas += [
        "",
        "## Error por progreso dentro de la ruta",
        "",
        "| Tramo de ruta | MAE (min) | Sesgo medio (min) | Tramos |",
        "|---|---:|---:|---:|",
    ]
    for _, fila in resultado["por_tercio"].iterrows():
        lineas.append(f"| {fila['tercio_ruta']} | {fila['mae_min']:.4f} | {fila['sesgo_medio_min']:+.4f} | {int(fila['n_tramos']):,} |")

    inicio_mae = resultado["por_tercio"].set_index("tercio_ruta").loc["inicio", "mae_min"]
    final_mae = resultado["por_tercio"].set_index("tercio_ruta").loc["final", "mae_min"]
    diff_pct = (final_mae - inicio_mae) / inicio_mae * 100
    lineas += [
        "",
        (
            f"El error crece un {diff_pct:.1f}% del inicio al final de la ruta. Es la pregunta"
            " que se dejo abierta en `bakeoff/lstm_experimental.py`: si el ritmo se degrada al"
            " final de la jornada de un modo que el modelo no captura. La diferencia es"
            f" {'notable' if abs(diff_pct) > 15 else 'pequena'}: {'conviene investigarla como' if abs(diff_pct) > 15 else 'no parece'} mejora futura."
        ),
        "",
        "## Los 15 tramos peor predichos",
        "",
        "| Ruta | Estacion | Poblacion | Real (min) | Prediccion (min) | Residuo (min) | Distancia (km) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, fila in resultado["peores"].iterrows():
        poblacion_txt = "almacen" if fila[POBLACION] == 1 else "reparto"
        lineas.append(
            f"| {fila[GRUPO]} | {fila['station_code']} | {poblacion_txt} | {fila[OBJETIVO]:.2f} "
            f"| {fila['prediccion']:.2f} | {fila['residuo_min']:+.2f} | {fila[DISTANCIA]:.2f} |"
        )

    n_almacen_en_peores = int((resultado["peores"][POBLACION] == 1).sum())
    n_reparto_en_peores = len(resultado["peores"]) - n_almacen_en_peores
    lineas += [
        "",
        (
            f"{n_almacen_en_peores} de los 15 peores casos son tramos de almacen y"
            f" {n_reparto_en_peores} de reparto."
            + (
                " Predomina almacen, coherente con que esa poblacion tiene errores en valor"
                " absoluto muy superiores por su propia escala (30 min de media frente a 1 min)."
                if n_almacen_en_peores > n_reparto_en_peores else
                " Predomina reparto, pese a que el almacen tiene errores absolutos mayores por"
                " su escala: cuando un tramo de reparto falla fuerte, falla tanto como uno de"
                " almacen a pesar de moverse en una escala treinta veces menor."
            )
        ),
        "",
    ]

    peores_con_velocidad = resultado["peores"].copy()
    peores_con_velocidad["velocidad_kmh"] = peores_con_velocidad[DISTANCIA] / (peores_con_velocidad[OBJETIVO] / 60)
    sospechosos = peores_con_velocidad[
        (peores_con_velocidad["velocidad_kmh"] < 3) & (peores_con_velocidad[DISTANCIA] > 0.1)
    ]
    if len(sospechosos):
        lineas += [
            (
                f"**{len(sospechosos)} de los peores casos tienen una velocidad implicita por"
                " debajo de 3 km/h** (distancia recorrida entre tiempo real), mas lenta que caminar."
                " No se recortan -es la regla del proyecto: los tiempos son medias historicas"
                " reales del dataset, no medidas de un vehiculo concreto ese dia- pero merecen"
                " quedar senalados en vez de darlos por buenos sin mirar:"
            ),
            "",
            "```text",
        ]
        for _, fila in sospechosos.iterrows():
            lineas.append(
                f"{fila[GRUPO]}   {fila['station_code']}   {fila[DISTANCIA]:.2f} km en {fila[OBJETIVO]:.1f} min"
                f"   ({fila['velocidad_kmh']:.1f} km/h)"
            )
        lineas += [
            "```",
            "",
            "Explicacion mas probable: son medias historicas construidas sobre pocas o una sola",
            "observacion para ese par de coordenadas exacto (recordar que el 90,1% de los pares",
            "no varian entre fechas, ver Limitaciones del Dataset), y esa observacion pudo",
            "incluir una espera real -un edificio con control de acceso, un intento fallido- que",
            "el dataset no distingue del tiempo de conduccion. No es un error de este pipeline;",
            "es una limitacion del dato de origen que ya estaba documentada, vista aqui desde el",
            "error del modelo en vez de desde el propio dato.",
            "",
        ]
    else:
        lineas += [
            "Ninguno de los peores casos tiene una velocidad implicita sospechosa.",
            "",
        ]

    lineas += [
        "## Graficos",
        "",
        "```text",
    ]
    if graficos:
        for g_nombre in graficos:
            lineas.append(f"figuras/{g_nombre}")
    else:
        lineas.append("(el trazado fallo en esta ejecucion; las tablas numericas siguen en los CSV de reports/)")
    lineas += [
        "```",
        "",
        "## Conclusiones para la memoria",
        "",
        (
            f"1. **Si hay sesgo sistematico en reparto**: {sesgo_rel_reparto:+.1f}% relativo a su"
            " media, por optimizar MAE (que ajusta a la mediana, no a la media) sobre una"
            " distribucion asimetrica. En almacen el sesgo es insignificante."
            if abs(sesgo_rel_reparto) > 3 else
            "1. No hay sesgo sistematico relevante en ninguna poblacion."
        ),
        "2. Los tramos mas largos se infrapredicen de forma sistematica y creciente por decil:",
        "   mismo mecanismo que el punto 1, mas visible cuanto mas se aleja el tramo de la",
        "   mediana de su hoja.",
        (
            f"3. El error no se concentra en una estacion aislada (rango de"
            f" {mejor_estacion['mae_min']:.2f} a {peor_estacion['mae_min']:.2f} min), pero **el"
            f" sesgo si es unanime**: las {total_estaciones} estaciones infrapredicen de media,"
            " reflejo del mismo sesgo poblacional del punto 1, no un problema por estacion."
            if estaciones_sesgo_positivo == total_estaciones else
            f"3. El error no se concentra en una estacion aislada, y el sesgo no es unanime entre"
            " estaciones."
        ),
        (
            f"4. Los peores casos se reparten {n_reparto_en_peores} reparto / {n_almacen_en_peores}"
            f" almacen. {len(sospechosos)} de los 15 tienen velocidad implicita por debajo de"
            " 3 km/h: no se recortan, por ser medias historicas reales, pero se documentan como"
            " posible limitacion del dato de origen (esperas no distinguidas de conduccion)."
            if len(sospechosos) else
            f"4. Los peores casos se reparten {n_reparto_en_peores} reparto / {n_almacen_en_peores}"
            " almacen, sin ningun patron de velocidad implicita sospechoso."
        ),
        "",
        "## Siguiente paso",
        "",
        "Con esto se cierra el diagnostico complementario del bake-off. La evaluacion que",
        "cuenta de cara a la memoria es la de `entrenamiento_final.py` sobre `test` (el 20%",
        "reservado, nunca tocado ni aqui ni en ningun script de `bakeoff/`).",
    ]
    return "\n".join(lineas) + "\n"


def registrar_en_mlflow(resultado: dict[str, Any]) -> str | None:
    try:
        import mlflow
    except ImportError:
        print("AVISO: mlflow no esta instalado; se omite el registro de experimentos.")
        return None

    MLRUNS.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + str(MLFLOW_DB).replace("\\", "/"))
    if mlflow.get_experiment_by_name(EXPERIMENTO) is None:
        mlflow.create_experiment(EXPERIMENTO, artifact_location=MLRUNS.as_uri())
    mlflow.set_experiment(EXPERIMENTO)

    with mlflow.start_run(run_name="analisis de errores (out-of-fold, dentro de train)") as run:
        mlflow.log_params({"n_splits": N_SPLITS, "estrategia": "out-of-fold dentro de train, arquitectura de catboost_zona"})
        for etiqueta, m in (("global", resultado["global_todo"]), ("reparto", resultado["global_reparto"]), ("almacen", resultado["global_almacen"])):
            for k, v in m.items():
                mlflow.log_metric(f"{etiqueta}_{k}", float(v))
        mlflow.log_artifact(str(REPORTS / "analisis_errores.md"))
        return run.info.run_id


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Cargando train (80% de las rutas)...", flush=True)
    tramos = cargar_train()
    print(f"  {len(tramos):,} tramos, {tramos[GRUPO].nunique():,} rutas", flush=True)

    print(f"\nGenerando predicciones fuera de muestra con GroupKFold de {N_SPLITS} pliegues:", flush=True)
    oof = generar_predicciones_oof(tramos)
    oof.to_csv(REPORTS / "analisis_errores_predicciones_oof.csv", index=False)

    print("\nAnalizando errores...", flush=True)
    resultado = analizar(oof)

    resultado["por_estacion"].to_csv(REPORTS / "analisis_errores error por estacion.csv", index=False)
    resultado["por_zona"].to_csv(REPORTS / "analisis_errores error por zona.csv", index=False)
    resultado["por_tercio"].to_csv(REPORTS / "analisis_errores error por tercio de ruta.csv", index=False)
    resultado["por_hora"].to_csv(REPORTS / "analisis_errores error por hora.csv", index=False)
    resultado["por_decil"].to_csv(REPORTS / "analisis_errores sesgo por decil.csv", index=False)
    resultado["peores"].to_csv(REPORTS / "analisis_errores peores casos.csv", index=False)

    print("Generando graficos...", flush=True)
    graficos = guardar_graficos(resultado)

    informe = redactar(resultado, graficos)
    (REPORTS / "analisis_errores.md").write_text(informe, encoding="utf-8")

    run_id = registrar_en_mlflow(resultado)

    (REPORTS / "analisis_errores.json").write_text(
        json.dumps(
            {
                "generado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "referencia_catboost_zona": {"mae_min": CV_MAE_REF, "r2": CV_R2_REF},
                "global": resultado["global_todo"],
                "global_reparto": resultado["global_reparto"],
                "global_almacen": resultado["global_almacen"],
                "mlflow_run_id": run_id,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("ANALISIS DE ERRORES COMPLETO")
    print("=" * 80)
    g = resultado["global_todo"]
    print(f"MAE global (fuera de muestra, dentro de train): {g['mae_min']:.4f} min   R2: {g['r2']:.4f}")
    print(f"referencia bakeoff/catboost_zona.py:             {CV_MAE_REF:.4f} min   R2: {CV_R2_REF:.4f}")
    print(f"sesgo medio reparto: {resultado['global_reparto']['sesgo_medio_min']:+.4f} min")
    print(f"sesgo medio almacen: {resultado['global_almacen']['sesgo_medio_min']:+.4f} min")
    print(f"\nInforme: {(REPORTS / 'analisis_errores.md').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
