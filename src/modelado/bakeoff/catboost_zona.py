"""CatBoost con `to_zone_id` nativa: cuanto aporta la identidad de la zona.

Por que existe este script
---------------------------
`bakeoff/lightgbm_unico.py` y `bakeoff/modelo_hibrido.py` dejaron `to_zone_id` fuera a
proposito: son 8.962 zonas de destino, demasiadas para one-hot, y la codificacion por
historico manual es la pieza mas facil de contaminar con fuga de datos. CatBoost resuelve
esto con *ordered target encoding*: calcula una estadistica de la categoria usando solo
las filas anteriores en un orden aleatorio interno, evitando por construccion que una
fila influya en su propia codificacion.

Correccion sobre el plan anterior (autoauditoria antes de escribir codigo)
---------------------------------------------------------------------------
Se dijo que CatBoost se probaria "sobre el especialista de reparto, que es donde esa
variable podria aportar". Se ha comprobado antes de escribir el modelo, y es inexacto:

    poblacion    from_zone_id nulo   to_zone_id nulo   to_zone_id categorias
    almacen      100,0% (6.112/6.112)      0,75% (46/6.112)         3.596
    reparto        0,72% (6.417/892.303)   0,72% (6.469/892.303)    8.956

`from_zone_id` es nulo el 100% de las veces en almacen porque el origen es el almacen y
no pertenece a zona alguna (la misma razon por la que `same_zone` se excluyo en
`bakeoff/modelo_hibrido.py`). Pero `to_zone_id`, la zona de LLEGADA, esta presente casi
siempre en las dos poblaciones: en que zona empieza a repartir una ruta es informacion
valida y conocida antes de salir. Por eso aqui se prueba en ambos especialistas, no solo
en el de reparto.

Diseno
------
Para cada poblacion se comparan tres candidatos con la MISMA arquitectura de division
(reparto / almacen) que `bakeoff/modelo_hibrido.py`, y con la MISMA metrica y particiones:

    1. LightGBM     el especialista ya elegido en el script anterior (referencia)
    2. CatBoost sin zona    mismas variables que (1), para aislar el efecto del algoritmo
    3. CatBoost con zona    (2) + to_zone_id, para aislar el efecto de la zona

La diferencia entre (2) y (3) es el numero que responde a la pregunta del encabezado.
La diferencia entre (1) y (2) es ruido de algoritmo y sirve de control: si fuera grande,
cualquier mejora de (3) habria que atribuirla en parte al modelo y no solo a la zona.

Los 46 tramos de almacen y los 6.469 de reparto sin `to_zone_id` no se descartan: se
imputan con la categoria explicita 'SIN_ZONA', igual de valida para un modelo de arboles
que cualquier otra categoria, y evita perder filas por un dato ausente minoritario.

**Esta es la arquitectura ganadora del bake-off** (CatBoost con zona en reparto, LightGBM
en almacen), pero este script SOLO compara candidatos por CV dentro de train: no
reentrena ni guarda ningun modelo de produccion. Eso lo hace en solitario
`modelado/entrenamiento_final.py`, que entrena la combinacion ganadora ya conocida sobre
el 100% de train y evalua una vez sobre test.

Uso:

    uv run python src/modelado/bakeoff/catboost_zona.py
"""

from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
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
    PARADA_TEMPRANA_ALMACEN as PARADA_LGBM_ALMACEN,
    PARAMS_CB_REPARTO,
    PARAMS_LGBM_ALMACEN,
    POBLACION,
    SEMILLA,
    SIN_ZONA,
    VARS_ALMACEN as VARS_ALMACEN_BASE,
    VARS_REPARTO as VARS_REPARTO_ZONA,
    ZONA,
    cargar_train,
    separar_validacion,
)
from modelado.metricas import metricas

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_MODELADO = PROJECT_ROOT / "artifacts" / "modelado"
REPORTS = ARTIFACTS_MODELADO / "reports"
MLRUNS = ARTIFACTS_MODELADO / "mlflow" / "mlruns"
MLFLOW_DB = ARTIFACTS_MODELADO / "mlflow" / "mlflow.db"

N_SPLITS = 5
EXPERIMENTO = "SmartDeliveryAI Tiempos"

# Referencias de los scripts anteriores. Se validan por reproduccion en este mismo script
# (los candidatos 'lightgbm' de aqui deben coincidir, particiones y semilla son las mismas)
# en vez de darse por buenas sin mas. Se refrescan a mano tras cada ejecucion previa.
BASELINE_MAE, BASELINE_R2 = 0.4473, 0.9192
HIBRIDO_MAE, HIBRIDO_R2 = 0.3800, 0.9349
HIBRIDO_MAE_REPARTO, HIBRIDO_MAE_ALMACEN = 0.3631, 2.8540

# ------------------------------------------------------------------------------------
# Variables. Base = las de bakeoff/modelo_hibrido.py (sin zona). Con zona = base +
# to_zone_id. VARS_REPARTO_ZONA y VARS_ALMACEN_BASE vienen de modelado.datos: son
# exactamente la arquitectura ganadora (reparto con zona, almacen sin zona) que despues
# usa entrenamiento_final.py. Las variantes "opuestas" (reparto sin zona para el
# candidato LightGBM, almacen con zona para el candidato CatBoost) son propias de este
# bake-off y no se comparten con datos.py, porque no son la arquitectura final.
# ------------------------------------------------------------------------------------
VARS_REPARTO_BASE = [
    DISTANCIA,
    "same_zone",
    "segment_position",
    "cumulative_distance_km",
    "packages_at_destination",
    "service_time_at_destination_seconds",
    "volume_at_destination_cm3",
    "packages_with_window_at_destination",
    "departure_hour",
    "route_total_stops",
    "route_package_count",
    "weekday",
    *CATEGORICAS_BASE,
]
VARS_ALMACEN_ZONA = VARS_ALMACEN_BASE + [ZONA]

# Mismos parametros que bakeoff/modelo_hibrido.py, para que el candidato "lightgbm"
# reproduzca exactamente esos resultados y sirva de control de coherencia entre scripts.
PARAMS_LGBM_REPARTO = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 2000,
    "verbose": -1,
    "seed": SEMILLA,
    "num_threads": 0,
}
PARADA_LGBM_REPARTO = 50

# CatBoost para almacen (6.112 filas): mas conservador, mismo espiritu que su par LightGBM.
# No viene de modelado.datos porque no es la arquitectura ganadora en almacen (esa es
# PARAMS_LGBM_ALMACEN); es solo un candidato de este bake-off.
PARAMS_CB_ALMACEN = {
    "loss_function": "MAE",
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 4,
    "l2_leaf_reg": 5.0,
    "random_seed": SEMILLA,
    "verbose": False,
    "early_stopping_rounds": 30,
    "thread_count": -1,
}


# ------------------------------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------------------------------
def entrenar_lgbm(train: pd.DataFrame, variables: list[str], categoricas: list[str], params: dict, parada: int):
    """Fija el universo de categorias UNA sola vez, sobre todo el fold de entrenamiento,
    antes de separar la validacion interna.

    Si se convirtiera `sub_train` y `val` a 'category' por separado, cada uno tendria su
    propio conjunto de categorias (el de los valores que le tocaron), y el modelo se
    entrenaria con un universo mas pequeno que el que luego se usa para predecir en test.
    Serializando el dtype una vez sobre `train` completo, sub_train/val/test comparten
    siempre los mismos codigos para la misma categoria.
    """
    train = train[[GRUPO, OBJETIVO, *variables]].copy()
    for c in categoricas:
        train[c] = train[c].astype("category")
    dtypes = train[categoricas].dtypes

    sub_train, val = separar_validacion(train, SEMILLA)
    modelo = lgb.LGBMRegressor(**params)
    modelo.fit(
        sub_train[variables],
        sub_train[OBJETIVO],
        eval_X=val[variables],
        eval_y=val[OBJETIVO],
        eval_metric=params["metric"],
        callbacks=[lgb.early_stopping(parada, verbose=False)],
        categorical_feature=categoricas,
    )
    return modelo, dtypes


def predecir_lgbm(modelo, test: pd.DataFrame, variables: list[str], categoricas: list[str], dtypes) -> np.ndarray:
    x = test[variables].copy()
    for c in categoricas:
        # Categoria de test no vista en entrenamiento -> NaN, que LightGBM trata como
        # ausente. Es el comportamiento correcto: mejor que reventar o inventar un valor.
        x[c] = x[c].astype(dtypes[c])
    return np.clip(modelo.predict(x), 0, None)


def entrenar_catboost(train: pd.DataFrame, variables: list[str], categoricas: list[str], params: dict):
    sub_train, val = separar_validacion(train, SEMILLA)
    modelo = CatBoostRegressor(**params)
    modelo.fit(
        sub_train[variables],
        sub_train[OBJETIVO],
        cat_features=categoricas,
        eval_set=(val[variables], val[OBJETIVO]),
        use_best_model=True,
    )
    return modelo


def predecir_catboost(modelo, test: pd.DataFrame, variables: list[str]) -> np.ndarray:
    return np.clip(modelo.predict(test[variables]), 0, None)


# ------------------------------------------------------------------------------------
# Validacion cruzada
# ------------------------------------------------------------------------------------
def evaluar(tramos: pd.DataFrame) -> dict[str, Any]:
    cv = GroupKFold(n_splits=N_SPLITS)
    candidatos_reparto: list[dict[str, Any]] = []
    candidatos_almacen: list[dict[str, Any]] = []
    hibridos: list[dict[str, Any]] = []
    importancias: dict[str, list[pd.Series]] = {
        "reparto catboost sin zona": [],
        "reparto catboost con zona": [],
        "almacen catboost sin zona": [],
        "almacen catboost con zona": [],
    }

    for pliegue, (idx_train, idx_test) in enumerate(cv.split(tramos, groups=tramos[GRUPO]), start=1):
        train_full = tramos.iloc[idx_train]
        test_full = tramos.iloc[idx_test]
        solapadas = set(train_full[GRUPO]) & set(test_full[GRUPO])
        if solapadas:
            raise AssertionError(f"Pliegue {pliegue}: {len(solapadas)} rutas en train y test a la vez")

        train_reparto = train_full[train_full[POBLACION] == 0]
        test_reparto = test_full[test_full[POBLACION] == 0]
        train_almacen = train_full[train_full[POBLACION] == 1]
        test_almacen = test_full[test_full[POBLACION] == 1]
        y_reparto = test_reparto[OBJETIVO].to_numpy()
        y_almacen = test_almacen[OBJETIVO].to_numpy()

        # --- especialista de reparto: 3 candidatos ---
        modelo_lgbm_r, dtypes_r = entrenar_lgbm(
            train_reparto, VARS_REPARTO_BASE, CATEGORICAS_BASE, PARAMS_LGBM_REPARTO, PARADA_LGBM_REPARTO
        )
        pred_lgbm_r = predecir_lgbm(modelo_lgbm_r, test_reparto, VARS_REPARTO_BASE, CATEGORICAS_BASE, dtypes_r)

        modelo_cbsz_r = entrenar_catboost(train_reparto, VARS_REPARTO_BASE, CATEGORICAS_BASE, PARAMS_CB_REPARTO)
        pred_cbsz_r = predecir_catboost(modelo_cbsz_r, test_reparto, VARS_REPARTO_BASE)
        importancias["reparto catboost sin zona"].append(
            pd.Series(modelo_cbsz_r.get_feature_importance(), index=VARS_REPARTO_BASE)
        )

        modelo_cbcz_r = entrenar_catboost(train_reparto, VARS_REPARTO_ZONA, CATEGORICAS_ZONA, PARAMS_CB_REPARTO)
        pred_cbcz_r = predecir_catboost(modelo_cbcz_r, test_reparto, VARS_REPARTO_ZONA)
        importancias["reparto catboost con zona"].append(
            pd.Series(modelo_cbcz_r.get_feature_importance(), index=VARS_REPARTO_ZONA)
        )

        reparto_preds = {"lightgbm": pred_lgbm_r, "catboost sin zona": pred_cbsz_r, "catboost con zona": pred_cbcz_r}
        for nombre, pred in reparto_preds.items():
            candidatos_reparto.append({"candidato": nombre, "pliegue": pliegue, **metricas(y_reparto, pred)})

        # --- especialista de almacen: 3 candidatos ---
        modelo_lgbm_a, dtypes_a = entrenar_lgbm(
            train_almacen, VARS_ALMACEN_BASE, CATEGORICAS_BASE, PARAMS_LGBM_ALMACEN, PARADA_LGBM_ALMACEN
        )
        pred_lgbm_a = predecir_lgbm(modelo_lgbm_a, test_almacen, VARS_ALMACEN_BASE, CATEGORICAS_BASE, dtypes_a)

        modelo_cbsz_a = entrenar_catboost(train_almacen, VARS_ALMACEN_BASE, CATEGORICAS_BASE, PARAMS_CB_ALMACEN)
        pred_cbsz_a = predecir_catboost(modelo_cbsz_a, test_almacen, VARS_ALMACEN_BASE)
        importancias["almacen catboost sin zona"].append(
            pd.Series(modelo_cbsz_a.get_feature_importance(), index=VARS_ALMACEN_BASE)
        )

        modelo_cbcz_a = entrenar_catboost(train_almacen, VARS_ALMACEN_ZONA, CATEGORICAS_ZONA, PARAMS_CB_ALMACEN)
        pred_cbcz_a = predecir_catboost(modelo_cbcz_a, test_almacen, VARS_ALMACEN_ZONA)
        importancias["almacen catboost con zona"].append(
            pd.Series(modelo_cbcz_a.get_feature_importance(), index=VARS_ALMACEN_ZONA)
        )

        almacen_preds = {"lightgbm": pred_lgbm_a, "catboost sin zona": pred_cbsz_a, "catboost con zona": pred_cbcz_a}
        for nombre, pred in almacen_preds.items():
            candidatos_almacen.append({"candidato": nombre, "pliegue": pliegue, **metricas(y_almacen, pred)})

        # --- las 9 combinaciones hibridas, para no ocultar ninguna evidencia ---
        for (nombre_r, pred_r), (nombre_a, pred_a) in itertools.product(reparto_preds.items(), almacen_preds.items()):
            y_pred_full = pd.Series(index=test_full.index, dtype=float)
            y_pred_full.loc[test_reparto.index] = pred_r
            y_pred_full.loc[test_almacen.index] = pred_a
            resultado = metricas(test_full[OBJETIVO].to_numpy(), y_pred_full.to_numpy())
            hibridos.append(
                {
                    "reparto": nombre_r,
                    "almacen": nombre_a,
                    "hibrido": f"{nombre_r} + {nombre_a}",
                    "pliegue": pliegue,
                    **resultado,
                }
            )

        print(
            f"  pliegue {pliegue}  reparto: lgbm {metricas(y_reparto, pred_lgbm_r)['mae_min']:.4f} "
            f"| cb sin zona {metricas(y_reparto, pred_cbsz_r)['mae_min']:.4f} "
            f"| cb con zona {metricas(y_reparto, pred_cbcz_r)['mae_min']:.4f}   "
            f"almacen: lgbm {metricas(y_almacen, pred_lgbm_a)['mae_min']:.4f} "
            f"| cb sin zona {metricas(y_almacen, pred_cbsz_a)['mae_min']:.4f} "
            f"| cb con zona {metricas(y_almacen, pred_cbcz_a)['mae_min']:.4f}"
        )

    imp_resueltas = {}
    for nombre, lista in importancias.items():
        serie = pd.concat(lista, axis=1).mean(axis=1).sort_values(ascending=False)
        imp_resueltas[nombre] = (serie / serie.sum() * 100).round(2).rename_axis("variable").reset_index(name="ganancia_pct")

    return {
        "candidatos_reparto": pd.DataFrame(candidatos_reparto),
        "candidatos_almacen": pd.DataFrame(candidatos_almacen),
        "hibridos": pd.DataFrame(hibridos),
        "importancias": imp_resueltas,
    }


def resumir(detalle: pd.DataFrame, clave: str | list[str]) -> pd.DataFrame:
    columnas = [c for c in detalle.columns if c not in ((clave if isinstance(clave, list) else [clave]) + ["pliegue"])]
    resumen = detalle.groupby(clave)[columnas].agg(["mean", "std"]).round(4)
    resumen.columns = [f"{m}_{e}" for m, e in resumen.columns]
    return resumen.reset_index()


# ------------------------------------------------------------------------------------
# Salidas
# ------------------------------------------------------------------------------------
def redactar(
    resumen_reparto: pd.DataFrame, resumen_almacen: pd.DataFrame, resumen_hibridos: pd.DataFrame,
    importancias: dict[str, pd.DataFrame], ganador: str,
) -> str:
    fila_ganadora = resumen_hibridos[resumen_hibridos["hibrido"] == ganador].iloc[0]
    lgbm_r = resumen_reparto[resumen_reparto["candidato"] == "lightgbm"].iloc[0]
    lgbm_a = resumen_almacen[resumen_almacen["candidato"] == "lightgbm"].iloc[0]

    # Control de coherencia: el candidato 'lightgbm' de este script deberia reproducir
    # los resultados de bakeoff/modelo_hibrido.py, porque son el mismo modelo, mismos
    # datos, misma semilla. Se mide la discrepancia en vez de asumir que coincide.
    discrepancia_r = abs(lgbm_r["mae_min_mean"] - HIBRIDO_MAE_REPARTO)
    discrepancia_a = abs(lgbm_a["mae_min_mean"] - HIBRIDO_MAE_ALMACEN)
    coherente = discrepancia_r < 0.01 and discrepancia_a < 0.01

    lineas = [
        "# CatBoost: cuanto aporta la identidad de la zona",
        "",
        f"Generado por `bakeoff/catboost_zona.py` el {datetime.now(UTC).date()}.",
        "",
        "## Autoauditoria: coherencia con el modelo hibrido anterior",
        "",
        "El candidato 'lightgbm' de este script usa el mismo modelo, variables, particiones",
        "y semilla que el especialista ganador de `bakeoff/modelo_hibrido.py`, asi que debe",
        "reproducir sus cifras. Se mide la diferencia en vez de asumirla:",
        "",
        "```text",
        f"reparto   modelo_hibrido.py: {HIBRIDO_MAE_REPARTO:.4f}   aqui: {lgbm_r['mae_min_mean']:.4f}   diferencia: {discrepancia_r:.4f}",
        f"almacen   modelo_hibrido.py: {HIBRIDO_MAE_ALMACEN:.4f}   aqui: {lgbm_a['mae_min_mean']:.4f}   diferencia: {discrepancia_a:.4f}",
        "```",
        "",
        ("Coherente: la diferencia es ruido de coma flotante, no un error de montaje."
         if coherente else
         "**INCOHERENTE**: la diferencia supera la tolerancia esperada. Antes de usar "
         "cualquier resultado de este informe, revisar que las particiones y parametros "
         "coinciden de verdad con `bakeoff/modelo_hibrido.py`."),
        "",
        "## Correccion sobre el plan inicial",
        "",
        "Se planteo probar `to_zone_id` solo en el especialista de reparto. Antes de",
        "escribir el modelo se comprobo la nulidad de la variable por poblacion y la",
        "premisa era parcialmente incorrecta: `from_zone_id` si es nulo el 100% de las",
        "veces en almacen (el origen es el almacen, sin zona), pero `to_zone_id` -la zona",
        "de LLEGADA- esta presente casi siempre en las dos poblaciones. Por eso se prueba",
        "en ambos especialistas.",
        "",
        "## Resultado",
        "",
        "```text",
        f"combinacion ganadora   {ganador}",
        f"MAE                    {fila_ganadora['mae_min_mean']:.4f} min   (hibrido anterior {HIBRIDO_MAE:.4f}, baseline {BASELINE_MAE:.4f})",
        f"R2                     {fila_ganadora['r2_mean']:.4f}       (hibrido anterior {HIBRIDO_R2:.4f}, baseline {BASELINE_R2:.4f})",
        "```",
        "",
        "Convencion de las cifras de \"efecto\" que siguen: **negativo es mejora** (el MAE",
        "baja), positivo es empeora. Se aplica igual a las dos filas de cada bloque.",
        "",
        "## Bake-off del especialista de reparto",
        "",
        "| Candidato | MAE (min) | RMSE (min) | R2 |",
        "|---|---:|---:|---:|",
    ]
    for _, fila in resumen_reparto.iterrows():
        lineas.append(
            f"| {fila['candidato']} | {fila['mae_min_mean']:.4f} ± {fila['mae_min_std']:.4f} "
            f"| {fila['rmse_min_mean']:.4f} ± {fila['rmse_min_std']:.4f} | {fila['r2_mean']:.4f} ± {fila['r2_std']:.4f} |"
        )

    ganador_reparto = resumen_reparto.loc[resumen_reparto["mae_min_mean"].idxmin(), "candidato"]
    cbsz_r = resumen_reparto[resumen_reparto["candidato"] == "catboost sin zona"].iloc[0]
    cbcz_r = resumen_reparto[resumen_reparto["candidato"] == "catboost con zona"].iloc[0]
    # Convencion unica para las dos lineas: "nuevo - viejo". Negativo = el cambio mejora
    # el MAE (baja el error), positivo = lo empeora. Sin negaciones sueltas al imprimir,
    # que es justo lo que produjo el error de signo detectado en la primera version de
    # este informe: el efecto del algoritmo salia con el signo invertido.
    efecto_zona_r = cbcz_r["mae_min_mean"] - cbsz_r["mae_min_mean"]
    efecto_algoritmo_r = cbsz_r["mae_min_mean"] - lgbm_r["mae_min_mean"]

    lineas += [
        "",
        "```text",
        f"efecto del algoritmo (lgbm -> catboost, mismas variables)   {efecto_algoritmo_r:+.4f} min",
        f"efecto de anadir to_zone_id (catboost, con zona - sin zona) {efecto_zona_r:+.4f} min",
        "```",
        "",
        "## Bake-off del especialista de almacen",
        "",
        "| Candidato | MAE (min) | RMSE (min) | R2 |",
        "|---|---:|---:|---:|",
    ]
    for _, fila in resumen_almacen.iterrows():
        lineas.append(
            f"| {fila['candidato']} | {fila['mae_min_mean']:.4f} ± {fila['mae_min_std']:.4f} "
            f"| {fila['rmse_min_mean']:.4f} ± {fila['rmse_min_std']:.4f} | {fila['r2_mean']:.4f} ± {fila['r2_std']:.4f} |"
        )

    ganador_almacen = resumen_almacen.loc[resumen_almacen["mae_min_mean"].idxmin(), "candidato"]
    cbsz_a = resumen_almacen[resumen_almacen["candidato"] == "catboost sin zona"].iloc[0]
    cbcz_a = resumen_almacen[resumen_almacen["candidato"] == "catboost con zona"].iloc[0]
    efecto_zona_a = cbcz_a["mae_min_mean"] - cbsz_a["mae_min_mean"]
    efecto_algoritmo_a = cbsz_a["mae_min_mean"] - lgbm_a["mae_min_mean"]

    lineas += [
        "",
        "```text",
        f"efecto del algoritmo (lgbm -> catboost, mismas variables)   {efecto_algoritmo_a:+.4f} min",
        f"efecto de anadir to_zone_id (catboost, con zona - sin zona) {efecto_zona_a:+.4f} min",
        "```",
        "",
        f"Ganador por poblacion: reparto -> **{ganador_reparto}**, almacen -> **{ganador_almacen}**.",
        "",
        "## Las nueve combinaciones (3 candidatos de reparto x 3 de almacen)",
        "",
        "Se muestran todas, no solo la ganadora, para no ocultar evidencia:",
        "",
        "| Reparto | Almacen | MAE (min) | R2 |",
        "|---|---|---:|---:|",
    ]
    for _, fila in resumen_hibridos.sort_values("mae_min_mean").iterrows():
        marca = " **<- ganador**" if fila["hibrido"] == ganador else ""
        lineas.append(f"| {fila['reparto']} | {fila['almacen']} | {fila['mae_min_mean']:.4f}{marca} | {fila['r2_mean']:.4f} |")

    lineas += [
        "",
        "## Importancia de variables, candidatos con zona",
        "",
        "### Reparto (CatBoost con zona)",
        "",
        "| Variable | Importancia % |",
        "|---|---:|",
    ]
    for _, fila in importancias["reparto catboost con zona"].head(10).iterrows():
        lineas.append(f"| `{fila['variable']}` | {fila['ganancia_pct']:.2f} |")
    lineas += [
        "",
        "### Almacen (CatBoost con zona)",
        "",
        "| Variable | Importancia % |",
        "|---|---:|",
    ]
    for _, fila in importancias["almacen catboost con zona"].head(10).iterrows():
        lineas.append(f"| `{fila['variable']}` | {fila['ganancia_pct']:.2f} |")

    peso_zona_r = float(
        importancias["reparto catboost con zona"].set_index("variable").loc[ZONA, "ganancia_pct"]
    ) if ZONA in importancias["reparto catboost con zona"]["variable"].values else 0.0
    peso_zona_a = float(
        importancias["almacen catboost con zona"].set_index("variable").loc[ZONA, "ganancia_pct"]
    ) if ZONA in importancias["almacen catboost con zona"]["variable"].values else 0.0

    lineas += [
        "",
        "## Veredicto sobre la identidad de la zona",
        "",
        "```text",
        f"peso de to_zone_id en el especialista de reparto   {peso_zona_r:.2f} %",
        f"peso de to_zone_id en el especialista de almacen   {peso_zona_a:.2f} %",
        "```",
        "",
        "## Siguiente paso",
        "",
        "LSTM como experimento comparativo: una ruta es una secuencia, y las redes",
        "recurrentes podrian aprender contexto que los arboles no ven (por ejemplo que el",
        "ritmo se degrada al final de la jornada). No se espera que supere al modelo",
        "hibrido de arboles, pero es la comparacion que la memoria necesita para justificar",
        "por que el modelo final es de arboles. Despues, `entrenamiento_final.py` entrena",
        "esta combinacion ganadora sobre el 100% de train y evalua sobre test.",
    ]
    return "\n".join(lineas) + "\n"


def registrar_en_mlflow(
    resumen_reparto: pd.DataFrame, resumen_almacen: pd.DataFrame, resumen_hibridos: pd.DataFrame, ganador: str
) -> str | None:
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

    run_id = None
    for poblacion, resumen in (("reparto", resumen_reparto), ("almacen", resumen_almacen)):
        for _, fila in resumen.iterrows():
            with mlflow.start_run(run_name=f"catboost {poblacion} {fila['candidato']}"):
                mlflow.log_params({"poblacion": poblacion, "candidato": fila["candidato"], "modelo": "CatBoost/LightGBM"})
                mlflow.log_metric("mae_min", float(fila["mae_min_mean"]))
                mlflow.log_metric("r2", float(fila["r2_mean"]))

    with mlflow.start_run(run_name=f"hibrido v3 {ganador}") as run:
        fila = resumen_hibridos[resumen_hibridos["hibrido"] == ganador].iloc[0]
        mlflow.log_params(
            {
                "modelo": "hibrido v3 (CatBoost + LightGBM segun bake-off)",
                "combinacion": ganador,
                "n_splits": N_SPLITS,
            }
        )
        mlflow.log_metric("mae_min", float(fila["mae_min_mean"]))
        mlflow.log_metric("r2", float(fila["r2_mean"]))
        mlflow.log_metric("baseline_mae_min", BASELINE_MAE)
        mlflow.log_metric("hibrido_anterior_mae_min", HIBRIDO_MAE)
        mlflow.log_artifact(str(REPORTS / "catboost_zona.md"))
        run_id = run.info.run_id
    return run_id


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Cargando train (80% de las rutas)...")
    tramos = cargar_train()
    print(f"  {len(tramos):,} tramos, {tramos[GRUPO].nunique():,} rutas")

    print(f"\nEvaluando con GroupKFold de {N_SPLITS} pliegues (lightgbm / catboost sin zona / catboost con zona):")
    resultado = evaluar(tramos)

    resumen_reparto = resumir(resultado["candidatos_reparto"], "candidato")
    resumen_almacen = resumir(resultado["candidatos_almacen"], "candidato")
    resumen_hibridos = resumir(resultado["hibridos"], ["reparto", "almacen", "hibrido"])

    ganador = resumen_hibridos.loc[resumen_hibridos["mae_min_mean"].idxmin(), "hibrido"]

    resultado["candidatos_reparto"].to_csv(REPORTS / "catboost_zona candidatos reparto.csv", index=False)
    resultado["candidatos_almacen"].to_csv(REPORTS / "catboost_zona candidatos almacen.csv", index=False)
    resumen_hibridos.to_csv(REPORTS / "catboost_zona hibridos resumen.csv", index=False)
    for nombre, imp in resultado["importancias"].items():
        imp.to_csv(REPORTS / f"catboost_zona importancia {nombre}.csv", index=False)

    informe = redactar(resumen_reparto, resumen_almacen, resumen_hibridos, resultado["importancias"], ganador)
    (REPORTS / "catboost_zona.md").write_text(informe, encoding="utf-8")

    run_id = registrar_en_mlflow(resumen_reparto, resumen_almacen, resumen_hibridos, ganador)

    (REPORTS / "catboost_zona.json").write_text(
        json.dumps(
            {
                "generado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "n_splits": N_SPLITS,
                "ganador": ganador,
                "baseline": {"mae_min": BASELINE_MAE, "r2": BASELINE_R2},
                "hibrido_anterior": {"mae_min": HIBRIDO_MAE, "r2": HIBRIDO_R2},
                "resumen_reparto": resumen_reparto.to_dict(orient="records"),
                "resumen_almacen": resumen_almacen.to_dict(orient="records"),
                "resumen_hibridos": resumen_hibridos.to_dict(orient="records"),
                "mlflow_run_id": run_id,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    fila_ganadora = resumen_hibridos[resumen_hibridos["hibrido"] == ganador].iloc[0]
    print("\n" + "=" * 90)
    print("BAKE-OFF REPARTO")
    print("=" * 90)
    for _, fila in resumen_reparto.iterrows():
        print(f"  {fila['candidato']:<20} MAE {fila['mae_min_mean']:.4f} min   R2 {fila['r2_mean']:.4f}")
    print("\nBAKE-OFF ALMACEN")
    for _, fila in resumen_almacen.iterrows():
        print(f"  {fila['candidato']:<20} MAE {fila['mae_min_mean']:.4f} min   R2 {fila['r2_mean']:.4f}")
    print("\n" + "=" * 90)
    print(f"HIBRIDO ANTERIOR            MAE {HIBRIDO_MAE:.4f}   R2 {HIBRIDO_R2:.4f}")
    print(f"GANADOR (catboost_zona)     MAE {fila_ganadora['mae_min_mean']:.4f}   R2 {fila_ganadora['r2_mean']:.4f}   -> {ganador}")
    print("=" * 90)
    print(f"\nInforme: {(REPORTS / 'catboost_zona.md').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
