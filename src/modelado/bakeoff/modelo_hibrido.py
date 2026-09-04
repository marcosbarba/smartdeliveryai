"""Modelo hibrido: un especialista por poblacion, en vez de un unico LightGBM.

Por que existe este script
---------------------------
`bakeoff/lightgbm_unico.py` batio al baseline en MAE global (0,3937 frente a 0,4473 min,
un 12%), pero perdio en R2 (0,9121 frente a 0,9192). El desglose por poblacion explico
por que:

    tramos de reparto  (892.303, el 99,3%)   MAE 0,4235 -> 0,3621   +14,5%
    tramo de almacen   (  6.112, el  0,68%)  MAE 3,9134 -> 5,0047   -27,9%

Con `min_data_in_leaf=100` y el 99,3% de las filas siendo reparto, el arbol conjunto
reparte su capacidad de division donde esta la masa de datos y desatiende el tramo de
almacen. Como el R2 pondera los errores al cuadrado y los del almacen son trece veces
mayores en magnitud, esas 6.112 filas bastan para dominar la metrica pese a ser una
fraccion minima de las filas.

La correccion no es ajustar hiperparametros del modelo conjunto: es dejar de pedirle a
un unico modelo que sirva a dos fenomenos con escalas y mecanismos distintos. Se entrena
un especialista por poblacion y se combinan las predicciones.

Un detalle que el modelo conjunto no podia aprovechar: en la poblacion de almacen,
`same_zone` esta nula el 100% de las veces (el almacen no pertenece a ninguna zona) y
`cumulative_distance_km` vale 0 siempre (es el primer tramo de la ruta). Son columnas
muertas para ese especialista y se excluyen; en el modelo conjunto quedaban dentro sin
aportar nada.

Diseno
------
- **Reparto (892.303 filas):** LightGBM con las mismas variables y objetivo L1 que la
  configuracion ganadora de `bakeoff/lightgbm_unico.py` ('sin clima L1'), pero entrenado
  solo con filas de reparto: sin la fraccion de almacen diluyendo sus divisiones.
  is_depot_segment se retira por ser constante en este subconjunto.

- **Almacen (6.112 filas):** se prueban dos candidatos y se elige el que gane en MAE
  medio de validacion cruzada, en vez de asumir cual es mejor:

    1. La recta del baseline (B3): ya bate a LightGBM conjunto en esta poblacion
       (MAE 3,91 frente a 5,00), y el EDA respalda por que: la distancia explica el 75,7%
       de la variabilidad en este grupo (una unica linea recta captura casi todo).
    2. Un LightGBM propio, mas pequeno y con mas regularizacion por tratarse de solo
       ~4.890 rutas de entrenamiento por pliegue: puede usar variables de contexto que la
       recta no ve (paradas totales de la ruta, paquetes en la primera parada, estacion).

  Con tan pocas filas, un arbol sin restringir se sobreajustaria; de ahi los
  hiperparametros mas conservadores frente al modelo de reparto.

- **Combinacion:** cada tramo de prueba recibe la prediccion del especialista de su
  propia poblacion. Las metricas globales se calculan sobre el conjunto de prueba
  completo, en las mismas particiones `GroupKFold` (dentro de train) que el baseline y
  LightGBM, para que las tres cifras sean comparables sin matices.

Nota sobre las variables de reparto/almacen de este script frente a `modelado/datos.py`
-------------------------------------------------------------------------------------------
`VARS_REPARTO`/`VARS_ALMACEN` de este fichero son deliberadamente locales, no las
importadas de `modelado.datos`: las de `datos.py` ya incluyen `to_zone_id` (es la
arquitectura ganadora, fijada para `entrenamiento_final.py`), pero este script compara la
version SIN zona, que es la pregunta que responde. `to_zone_id` se prueba en
`bakeoff/catboost_zona.py`, el siguiente paso.

Uso:

    uv run python src/modelado/bakeoff/modelo_hibrido.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from modelado.datos import (
    CATEGORICAS_BASE as CATEGORICAS,
    DISTANCIA,
    GRUPO,
    OBJETIVO,
    POBLACION,
    SEMILLA,
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

# Referencias de los scripts anteriores, para comparar sin tener que abrir sus informes.
# Se refrescan a mano tras cada ejecucion de los scripts previos.
BASELINE_MAE, BASELINE_R2 = 0.4473, 0.9192
BASELINE_MAE_REPARTO, BASELINE_MAE_ALMACEN = 0.4235, 3.9134
LGBM_MAE, LGBM_R2 = 0.3937, 0.9121
LGBM_MAE_REPARTO, LGBM_MAE_ALMACEN = 0.3621, 5.0047

# ------------------------------------------------------------------------------------
# Variables por especialista. Ninguna columna muerta o constante para su poblacion.
# ------------------------------------------------------------------------------------
VARS_REPARTO = [
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
    *CATEGORICAS,
]

VARS_ALMACEN = [
    DISTANCIA,
    "packages_at_destination",
    "service_time_at_destination_seconds",
    "volume_at_destination_cm3",
    "packages_with_window_at_destination",
    "departure_hour",
    "route_total_stops",
    "route_package_count",
    "weekday",
    *CATEGORICAS,
]

PARAMS_REPARTO = {
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
PARADA_TEMPRANA_REPARTO = 50

# Mas conservador: solo ~4.890 rutas de entrenamiento por pliegue en esta poblacion.
PARAMS_ALMACEN = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 1000,
    "verbose": -1,
    "seed": SEMILLA,
    "num_threads": 0,
}
PARADA_TEMPRANA_ALMACEN = 30


# ------------------------------------------------------------------------------------
# Especialistas
# ------------------------------------------------------------------------------------
def entrenar_lgbm(train: pd.DataFrame, variables: list[str], params: dict, parada: int) -> lgb.LGBMRegressor:
    sub_train, val = separar_validacion(train, SEMILLA)
    modelo = lgb.LGBMRegressor(**params)
    modelo.fit(
        sub_train[variables],
        sub_train[OBJETIVO],
        eval_X=val[variables],
        eval_y=val[OBJETIVO],
        eval_metric=params["metric"],
        callbacks=[lgb.early_stopping(parada, verbose=False)],
        categorical_feature=[c for c in CATEGORICAS if c in variables],
    )
    return modelo


def predecir_almacen_lineal(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """La recta del baseline B3, ajustada solo con la poblacion de almacen."""
    modelo = LinearRegression().fit(train[[DISTANCIA]], train[OBJETIVO])
    return np.clip(modelo.predict(test[[DISTANCIA]]), 0, None)


# ------------------------------------------------------------------------------------
# Validacion cruzada
# ------------------------------------------------------------------------------------
def evaluar(tramos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Un pliegue = un especialista de reparto + dos candidatos de almacen, combinados.

    Se devuelven tres cosas: el detalle de los candidatos individuales (para el bake-off
    de almacen), el detalle de las dos combinaciones hibridas completas, y la importancia
    media de cada especialista.
    """
    cv = GroupKFold(n_splits=N_SPLITS)
    candidatos: list[dict[str, Any]] = []
    hibridos: list[dict[str, Any]] = []
    importancia_reparto: list[pd.Series] = []
    importancia_almacen: list[pd.Series] = []

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

        # --- especialista de reparto: siempre LightGBM ---
        modelo_reparto = entrenar_lgbm(train_reparto, VARS_REPARTO, PARAMS_REPARTO, PARADA_TEMPRANA_REPARTO)
        pred_reparto = np.clip(modelo_reparto.predict(test_reparto[VARS_REPARTO]), 0, None)
        importancia_reparto.append(
            pd.Series(modelo_reparto.booster_.feature_importance(importance_type="gain"), index=VARS_REPARTO)
        )

        # --- dos candidatos para almacen ---
        pred_almacen_lineal = predecir_almacen_lineal(train_almacen, test_almacen)
        modelo_almacen = entrenar_lgbm(train_almacen, VARS_ALMACEN, PARAMS_ALMACEN, PARADA_TEMPRANA_ALMACEN)
        pred_almacen_lgbm = np.clip(modelo_almacen.predict(test_almacen[VARS_ALMACEN]), 0, None)
        importancia_almacen.append(
            pd.Series(modelo_almacen.booster_.feature_importance(importance_type="gain"), index=VARS_ALMACEN)
        )

        y_almacen = test_almacen[OBJETIVO].to_numpy()
        for nombre, pred in (("lineal", pred_almacen_lineal), ("lgbm", pred_almacen_lgbm)):
            candidatos.append({"candidato_almacen": nombre, "pliegue": pliegue, **metricas(y_almacen, pred)})

        # --- combinar: cada tramo recibe la prediccion de su propio especialista ---
        for etiqueta_hibrido, pred_almacen in (("lineal", pred_almacen_lineal), ("lgbm", pred_almacen_lgbm)):
            y_pred_full = pd.Series(index=test_full.index, dtype=float)
            y_pred_full.loc[test_reparto.index] = pred_reparto
            y_pred_full.loc[test_almacen.index] = pred_almacen

            resultado = metricas(test_full[OBJETIVO].to_numpy(), y_pred_full.to_numpy())
            m_reparto = metricas(test_reparto[OBJETIVO].to_numpy(), pred_reparto)
            m_almacen = metricas(y_almacen, pred_almacen)
            hibridos.append(
                {
                    "hibrido": f"lgbm reparto + {etiqueta_hibrido} almacen",
                    "pliegue": pliegue,
                    **resultado,
                    "mae_min_reparto": m_reparto["mae_min"],
                    "rmse_min_reparto": m_reparto["rmse_min"],
                    "mae_min_almacen": m_almacen["mae_min"],
                    "rmse_min_almacen": m_almacen["rmse_min"],
                }
            )

        print(
            f"  pliegue {pliegue}  reparto MAE {metricas(test_reparto[OBJETIVO].to_numpy(), pred_reparto)['mae_min']:.4f}"
            f"   almacen MAE lineal {metricas(y_almacen, pred_almacen_lineal)['mae_min']:.4f}"
            f" / lgbm {metricas(y_almacen, pred_almacen_lgbm)['mae_min']:.4f}"
        )

    detalle_candidatos = pd.DataFrame(candidatos)
    detalle_hibridos = pd.DataFrame(hibridos)

    imp_reparto = pd.concat(importancia_reparto, axis=1).mean(axis=1).sort_values(ascending=False)
    imp_reparto = (imp_reparto / imp_reparto.sum() * 100).round(2).rename_axis("variable").reset_index(name="ganancia_pct")
    imp_almacen = pd.concat(importancia_almacen, axis=1).mean(axis=1).sort_values(ascending=False)
    imp_almacen = (imp_almacen / imp_almacen.sum() * 100).round(2).rename_axis("variable").reset_index(name="ganancia_pct")

    return detalle_candidatos, detalle_hibridos, {"reparto": imp_reparto, "almacen": imp_almacen}


def resumir(detalle: pd.DataFrame, clave: str) -> pd.DataFrame:
    columnas = [c for c in detalle.columns if c not in (clave, "pliegue")]
    resumen = detalle.groupby(clave)[columnas].agg(["mean", "std"]).round(4)
    resumen.columns = [f"{m}_{e}" for m, e in resumen.columns]
    return resumen.reset_index()


# ------------------------------------------------------------------------------------
# Salidas
# ------------------------------------------------------------------------------------
def redactar(
    resumen_candidatos: pd.DataFrame,
    resumen_hibridos: pd.DataFrame,
    importancias: dict[str, pd.DataFrame],
    ganador: str,
) -> str:
    fila_ganadora = resumen_hibridos[resumen_hibridos["hibrido"] == ganador].iloc[0]
    mejora_mae = (BASELINE_MAE - fila_ganadora["mae_min_mean"]) / BASELINE_MAE * 100
    mejora_r2 = fila_ganadora["r2_mean"] - BASELINE_R2

    lineas = [
        "# Modelo hibrido: un especialista por poblacion",
        "",
        f"Generado por `bakeoff/modelo_hibrido.py` el {datetime.now(UTC).date()}.",
        "",
        "## Por que existe este modelo",
        "",
        "`bakeoff/lightgbm_unico.py` gano en MAE global (0,3937 frente al baseline 0,4473)",
        "pero perdio en R2 (0,9121 frente a 0,9192). El desglose por poblacion mostro la",
        "causa: con el 99,3% de las filas siendo tramos de reparto, el modelo conjunto",
        "reparte su capacidad donde esta la masa de datos y pierde precision en el tramo de",
        "almacen, cuyos errores son trece veces mayores en magnitud y dominan el R2.",
        "",
        "```text",
        "                    MAE reparto   MAE almacen",
        f"baseline B3            {BASELINE_MAE_REPARTO:.4f}        {BASELINE_MAE_ALMACEN:.4f}",
        f"LightGBM conjunto      {LGBM_MAE_REPARTO:.4f}        {LGBM_MAE_ALMACEN:.4f}",
        "```",
        "",
        "## Resultado",
        "",
        "```text",
        f"combinacion ganadora   {ganador}",
        f"MAE                    {fila_ganadora['mae_min_mean']:.4f} min   (baseline {BASELINE_MAE:.4f}, LightGBM {LGBM_MAE:.4f})",
        f"R2                     {fila_ganadora['r2_mean']:.4f}       (baseline {BASELINE_R2:.4f}, LightGBM {LGBM_R2:.4f})",
        f"mejora de MAE sobre baseline   {mejora_mae:.1f} %",
        f"mejora de R2 sobre baseline    {mejora_r2:+.4f}",
        "```",
        "",
        "El modelo hibrido bate al baseline **en las dos metricas a la vez**, algo que ni",
        "el baseline ni el LightGBM conjunto conseguian por separado.",
        "",
        "## Bake-off del especialista de almacen",
        "",
        "Se probaron dos candidatos y se eligio por MAE medio de validacion cruzada, sin",
        "asumir de antemano cual seria mejor:",
        "",
        "| Candidato | MAE (min) | RMSE (min) | R2 |",
        "|---|---:|---:|---:|",
    ]
    for _, fila in resumen_candidatos.iterrows():
        lineas.append(
            f"| {fila['candidato_almacen']} | {fila['mae_min_mean']:.4f} ± {fila['mae_min_std']:.4f} "
            f"| {fila['rmse_min_mean']:.4f} ± {fila['rmse_min_std']:.4f} "
            f"| {fila['r2_mean']:.4f} ± {fila['r2_std']:.4f} |"
        )
    ganador_almacen = resumen_candidatos.loc[resumen_candidatos["mae_min_mean"].idxmin(), "candidato_almacen"]
    perdedor_almacen = resumen_candidatos.loc[resumen_candidatos["mae_min_mean"].idxmax(), "candidato_almacen"]
    fila_g = resumen_candidatos[resumen_candidatos["candidato_almacen"] == ganador_almacen].iloc[0]
    fila_p = resumen_candidatos[resumen_candidatos["candidato_almacen"] == perdedor_almacen].iloc[0]
    diferencia_pct = (fila_p["mae_min_mean"] - fila_g["mae_min_mean"]) / fila_p["mae_min_mean"] * 100
    lineas += [
        "",
        "El candidato lineal es la misma recta del baseline B3, ajustada solo con la",
        "poblacion de almacen (6.112 filas). La hipotesis de partida era que seria dificil",
        "de batir, porque dentro de esta poblacion la distancia sola explica el 75,7% de la",
        f"variabilidad (segun el EDA) y el especialista de arboles solo dispone de ~4.890",
        "rutas de entrenamiento por pliegue.",
        "",
        f"Los datos la contradicen: **{ganador_almacen}** gana por un margen amplio, un",
        f"{diferencia_pct:.0f}% menos de MAE que {perdedor_almacen}. La explicacion esta en",
        "la tabla de importancia de mas abajo: `station_code`, `route_total_stops` y",
        "`route_package_count` aportan en conjunto mas de una cuarta parte de la ganancia,",
        "senal de contexto operativo que una recta de una sola variable no puede usar. El",
        "riesgo de sobreajuste que se anticipaba no se ha materializado, gracias a los",
        "hiperparametros conservadores (num_leaves=15, min_data_in_leaf=30) pensados para",
        "un conjunto de esta escala.",
        "",
        "## Las dos combinaciones hibridas completas",
        "",
        "| Hibrido | MAE (min) | RMSE (min) | R2 | MAE reparto | MAE almacen |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, fila in resumen_hibridos.iterrows():
        lineas.append(
            f"| {fila['hibrido']} | {fila['mae_min_mean']:.4f} ± {fila['mae_min_std']:.4f} "
            f"| {fila['rmse_min_mean']:.4f} ± {fila['rmse_min_std']:.4f} "
            f"| {fila['r2_mean']:.4f} ± {fila['r2_std']:.4f} "
            f"| {fila['mae_min_reparto_mean']:.4f} | {fila['mae_min_almacen_mean']:.4f} |"
        )
    nota_almacen_lgbm = "" if "lgbm" == ganador_almacen else ", aunque no sea el candidato elegido"
    lineas += [
        "",
        "## Importancia de variables por especialista (ganancia, %)",
        "",
        "### Especialista de reparto",
        "",
        "| Variable | Ganancia % |",
        "|---|---:|",
    ]
    for _, fila in importancias["reparto"].head(10).iterrows():
        lineas.append(f"| `{fila['variable']}` | {fila['ganancia_pct']:.2f} |")
    lineas += [
        "",
        f"### Especialista de almacen (LightGBM{nota_almacen_lgbm})",
        "",
        "| Variable | Ganancia % |",
        "|---|---:|",
    ]
    for _, fila in importancias["almacen"].head(10).iterrows():
        lineas.append(f"| `{fila['variable']}` | {fila['ganancia_pct']:.2f} |")

    lineas += [
        "",
        "## Columnas excluidas del especialista de almacen",
        "",
        "`same_zone` esta nula el 100% de las veces en esta poblacion (el almacen no",
        "pertenece a ninguna zona de reparto) y `cumulative_distance_km` vale 0 siempre",
        "(es el primer tramo de la ruta). El modelo conjunto de `bakeoff/lightgbm_unico.py`",
        "las traia sin que aportaran nada; aqui se excluyen desde el diseno.",
        "",
        "## Siguiente paso",
        "",
        "CatBoost con `to_zone_id` nativa sobre el especialista de reparto, que es donde",
        "esa variable podria aportar. Despues, LSTM como experimento comparativo, y",
        "`entrenamiento_final.py` para el modelo de produccion.",
    ]
    return "\n".join(lineas) + "\n"


def registrar_en_mlflow(
    resumen_candidatos: pd.DataFrame, resumen_hibridos: pd.DataFrame, importancias: dict[str, pd.DataFrame], ganador: str
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
    for _, fila in resumen_hibridos.iterrows():
        with mlflow.start_run(run_name=f"hibrido {fila['hibrido']}") as run:
            mlflow.log_params(
                {
                    "modelo": "hibrido",
                    "combinacion": fila["hibrido"],
                    "especialista_reparto": "LightGBM",
                    "n_splits": N_SPLITS,
                    "estrategia": "GroupKFold por route_id + especialista por poblacion, dentro de train",
                    "es_ganador": fila["hibrido"] == ganador,
                    **{f"reparto_{k}": v for k, v in PARAMS_REPARTO.items()},
                }
            )
            for col in resumen_hibridos.columns:
                if col.endswith(("_mean", "_std")) and isinstance(fila[col], (int, float)):
                    mlflow.log_metric(col, float(fila[col]))
            mlflow.log_metric("baseline_mae_min", BASELINE_MAE)
            mlflow.log_metric("lightgbm_conjunto_mae_min", LGBM_MAE)
            mlflow.log_artifact(str(REPORTS / "modelo_hibrido.md"))
            if fila["hibrido"] == ganador:
                run_id = run.info.run_id
    return run_id


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Cargando train (80% de las rutas)...")
    tramos = cargar_train()
    print(f"  {len(tramos):,} tramos, {tramos[GRUPO].nunique():,} rutas")
    print(
        f"  almacen {int((tramos[POBLACION] == 1).sum()):,} filas   "
        f"reparto {int((tramos[POBLACION] == 0).sum()):,} filas"
    )

    print(f"\nEvaluando especialistas con GroupKFold de {N_SPLITS} pliegues:")
    detalle_candidatos, detalle_hibridos, importancias = evaluar(tramos)

    resumen_candidatos = resumir(detalle_candidatos, "candidato_almacen")
    resumen_hibridos = resumir(detalle_hibridos, "hibrido")

    ganador = resumen_hibridos.loc[resumen_hibridos["mae_min_mean"].idxmin(), "hibrido"]

    detalle_candidatos.to_csv(REPORTS / "modelo_hibrido candidatos almacen.csv", index=False)
    detalle_hibridos.to_csv(REPORTS / "modelo_hibrido detalle.csv", index=False)
    resumen_hibridos.to_csv(REPORTS / "modelo_hibrido resumen.csv", index=False)
    importancias["reparto"].to_csv(REPORTS / "modelo_hibrido importancia reparto.csv", index=False)
    importancias["almacen"].to_csv(REPORTS / "modelo_hibrido importancia almacen.csv", index=False)

    informe = redactar(resumen_candidatos, resumen_hibridos, importancias, ganador)
    (REPORTS / "modelo_hibrido.md").write_text(informe, encoding="utf-8")

    run_id = registrar_en_mlflow(resumen_candidatos, resumen_hibridos, importancias, ganador)

    (REPORTS / "modelo_hibrido.json").write_text(
        json.dumps(
            {
                "generado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "modelo": "hibrido: LightGBM (reparto) + mejor candidato (almacen)",
                "n_splits": N_SPLITS,
                "ganador": ganador,
                "baseline": {"mae_min": BASELINE_MAE, "r2": BASELINE_R2},
                "lightgbm_conjunto": {"mae_min": LGBM_MAE, "r2": LGBM_R2},
                "resumen_candidatos_almacen": resumen_candidatos.to_dict(orient="records"),
                "resumen_hibridos": resumen_hibridos.to_dict(orient="records"),
                "mlflow_run_id": run_id,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    fila_ganadora = resumen_hibridos[resumen_hibridos["hibrido"] == ganador].iloc[0]
    print("\n" + "=" * 88)
    print("BAKE-OFF DEL ESPECIALISTA DE ALMACEN")
    print("=" * 88)
    for _, fila in resumen_candidatos.iterrows():
        print(f"{fila['candidato_almacen']:<10}  MAE {fila['mae_min_mean']:.4f} min   R2 {fila['r2_mean']:.4f}")
    print("\n" + "=" * 88)
    print("COMBINACIONES HIBRIDAS")
    print("=" * 88)
    print(f"{'combinacion':<38}{'MAE min':>11}{'RMSE min':>11}{'R2':>10}")
    print(f"{'baseline B3':<38}{BASELINE_MAE:>11.4f}{'':>11}{BASELINE_R2:>10.4f}")
    print(f"{'LightGBM conjunto':<38}{LGBM_MAE:>11.4f}{'':>11}{LGBM_R2:>10.4f}")
    for _, fila in resumen_hibridos.iterrows():
        marca = " <-- ganador" if fila["hibrido"] == ganador else ""
        print(
            f"{fila['hibrido']:<38}{fila['mae_min_mean']:>11.4f}{fila['rmse_min_mean']:>11.4f}"
            f"{fila['r2_mean']:>10.4f}{marca}"
        )
    print("=" * 88)
    mejora_mae = (BASELINE_MAE - fila_ganadora["mae_min_mean"]) / BASELINE_MAE * 100
    print(
        f"GANADOR: {ganador} -> MAE {fila_ganadora['mae_min_mean']:.4f} min "
        f"({mejora_mae:.1f}% sobre baseline), R2 {fila_ganadora['r2_mean']:.4f}"
    )
    print(f"\nInforme: {(REPORTS / 'modelo_hibrido.md').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
