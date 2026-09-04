"""LightGBM sobre la tabla de tramos: el primer modelo serio, contra el baseline.

Que hace y por que
------------------
Predice `travel_time_seconds` con gradient boosting sobre arboles, usando **exactamente
las mismas particiones y la misma metrica** que `bakeoff/baseline.py`, para que la
comparacion sea directa y no haya que discutir si las cifras son medibles entre si.

Frente a la recta del baseline, un arbol puede aprender interacciones por su cuenta: por
ejemplo que un kilometro dentro de la misma zona cuesta menos minutos que un kilometro
entre zonas distintas. Ese 42% de variabilidad que la distancia no explica es lo que se
intenta capturar aqui.

Decisiones tomadas y su motivo
------------------------------
- **Un solo modelo con `is_depot_segment`, no dos modelos separados.** Los tramos de
  almacen son solo 6.112 filas: un modelo aparte tendria riesgo real de sobreajuste y
  perderia lo que si es comun a ambas poblaciones. El arbol parte por esa variable en la
  raiz de todos modos, con lo que consigue la especializacion sin pagar ese precio.
- **Metrica desglosada por poblacion, ademas de la global.** El 99,3% de las filas son
  tramos de reparto, asi que el MAE global mide practicamente solo el reparto: el tramo
  de almacen podria predecirse muy mal sin que la cifra global se moviera.
- **`to_zone_id` y `from_zone_id` quedan fuera.** 8.962 categorias no admiten one-hot y
  la codificacion por historico es la pieza mas delicada de la fase. Se abordan en
  `bakeoff/catboost_zona.py`, que las trata de forma nativa y sin fuga por construccion.
  La diferencia entre ambos scripts mide cuanto aporta la identidad de la zona.
- **Coordenadas fuera.** El objetivo es una media historica por par de coordenadas, asi
  que darle lat/lng permitiria memorizar pares concretos en vez de aprender el fenomeno.
- **Parada temprana contra un subconjunto de rutas del propio entrenamiento**, nunca
  contra el pliegue de prueba: usar la prueba para decidir cuando parar contaminaria la
  metrica que despues se reporta.
- **Sin ajuste de hiperparametros todavia.** Esta ejecucion establece el punto de partida
  con valores razonables. Optimizar viene despues, y sobre estas mismas particiones.

Sobre la comparacion "con clima" del script original (adaptacion obligada)
---------------------------------------------------------------------------
La version original de este script (`02 LightGBM.py`) entrenaba tres configuraciones:
"sin clima L1", "con clima L1" y "con clima L2", para medir si la meteorologia aportaba
algo que la correlacion bivariante del EDA no viera. Esa comparacion ya se hizo una vez
sobre el 100% de los datos y concluyo que el clima no aporta (peso de ganancia ~1%, MAE
ligeramente peor al incluirlo) -conclusion que quedo fijada en el diseno compartido de
`modelado/datos.py`: `VARS_REPARTO`/`VARS_ALMACEN` no incluyen columnas de clima, y
`cargar_train()` ni siquiera las lee del Parquet. Como este script (y todo el bake-off)
tiene prohibido reimplementar su propia carga de datos, las configuraciones "con clima"
no se pueden reproducir aqui sin romper esa regla; se sustituyen por una comparacion
"sin clima L1" frente a "sin clima L2" para seguir respondiendo la otra pregunta de este
script (que funcion de perdida conviene), y se documenta la conclusion sobre el clima como
un hecho ya establecido en vez de remedirlo.

Uso:

    uv run python src/modelado/bakeoff/lightgbm_unico.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
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

# Referencia de bakeoff/baseline.py, para no tener que abrir el otro informe. Se refresca
# a mano tras cada ejecucion de baseline.py: valores de partida heredados del ultimo run
# conocido, no recalculados automaticamente aqui.
BASELINE_MAE = 0.4473
BASELINE_R2 = 0.9192

# ------------------------------------------------------------------------------------
# Variables. Todas se conocen antes de que la furgoneta salga del almacen.
# ------------------------------------------------------------------------------------
VARS_TRAMO = [
    DISTANCIA,
    POBLACION,
    "same_zone",
    "segment_position",
    "cumulative_distance_km",
]
# `segment_ratio` se omite a proposito: es segment_position / route_total_stops, y ambas
# ya estan. El EDA la marco con VIF 30 junto a segment_position.

VARS_DESTINO = [
    "packages_at_destination",
    "service_time_at_destination_seconds",
    "volume_at_destination_cm3",
    "packages_with_window_at_destination",
]

VARS_RUTA = [
    "departure_hour",
    "route_total_stops",
    "route_package_count",
    "weekday",
]
# `is_weekend` se omite: correlaciona 0,756 con weekday y es derivable de ella.

_BASE = VARS_TRAMO + VARS_DESTINO + VARS_RUTA + CATEGORICAS

# Dos configuraciones que responden a la pregunta que sigue abierta tras fijar el clima
# fuera del diseno compartido: que objetivo conviene?
#
# Optimizando L1 (error absoluto) el MAE mejora claramente sobre el baseline, pero el R2
# empeora. No es un fallo, es el compromiso esperable: L1 acepta fallar mas en los tramos
# raros y largos a cambio de acertar mas en el caso tipico, y el R2 penaliza justo esos
# fallos grandes. Como la metrica principal declarada del proyecto es el MAE, conviene
# medir ambos y decidir con datos en vez de elegir el objetivo por defecto.
CONFIGURACIONES: dict[str, dict[str, Any]] = {
    "sin clima L1": {"variables": _BASE, "objective": "regression_l1", "metric": "l1"},
    "sin clima L2": {"variables": _BASE, "objective": "regression", "metric": "l2"},
}

PARAMS = {
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
PARADA_TEMPRANA = 50


# ------------------------------------------------------------------------------------
# Metricas
# ------------------------------------------------------------------------------------
def metricas_por_poblacion(test: pd.DataFrame, y_pred: np.ndarray) -> dict[str, float]:
    """Desglosa por tipo de tramo.

    Imprescindible: el 99,3% de las filas son reparto, asi que la cifra global no dice
    casi nada sobre como se predice el trayecto al barrio.
    """
    salida: dict[str, float] = {}
    for grupo, etiqueta in ((0, "reparto"), (1, "almacen")):
        mask = (test[POBLACION] == grupo).to_numpy()
        if not mask.any():
            continue
        m = metricas(test.loc[mask, OBJETIVO].to_numpy(), y_pred[mask])
        salida[f"mae_min_{etiqueta}"] = m["mae_min"]
        salida[f"rmse_min_{etiqueta}"] = m["rmse_min"]
    return salida


# ------------------------------------------------------------------------------------
# Entrenamiento
# ------------------------------------------------------------------------------------
def evaluar(tramos: pd.DataFrame, config: dict[str, Any], etiqueta: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GroupKFold por ruta, dentro de train. Devuelve metricas por pliegue e importancia
    media."""
    variables = config["variables"]
    params = {**PARAMS, "objective": config["objective"], "metric": config["metric"]}
    cv = GroupKFold(n_splits=N_SPLITS)
    filas: list[dict[str, Any]] = []
    importancias: list[pd.Series] = []

    for pliegue, (idx_train, idx_test) in enumerate(cv.split(tramos, groups=tramos[GRUPO]), start=1):
        train_completo = tramos.iloc[idx_train]
        test = tramos.iloc[idx_test]

        solapadas = set(train_completo[GRUPO]) & set(test[GRUPO])
        if solapadas:
            raise AssertionError(f"Pliegue {pliegue}: {len(solapadas)} rutas en train y test a la vez")

        train, val = separar_validacion(train_completo, SEMILLA)
        modelo = lgb.LGBMRegressor(**params)
        modelo.fit(
            train[variables],
            train[OBJETIVO],
            eval_X=val[variables],
            eval_y=val[OBJETIVO],
            eval_metric=config["metric"],
            callbacks=[lgb.early_stopping(PARADA_TEMPRANA, verbose=False)],
            categorical_feature=[c for c in CATEGORICAS if c in variables],
        )

        y_pred = np.clip(modelo.predict(test[variables]), 0, None)
        resultado = metricas(test[OBJETIVO].to_numpy(), y_pred)
        filas.append(
            {
                "configuracion": etiqueta,
                "pliegue": pliegue,
                **resultado,
                **metricas_por_poblacion(test, y_pred),
                "arboles": int(modelo.best_iteration_ or params["n_estimators"]),
            }
        )
        importancias.append(
            pd.Series(modelo.booster_.feature_importance(importance_type="gain"), index=variables)
        )
        print(
            f"  pliegue {pliegue}  MAE {resultado['mae_min']:.4f} min   "
            f"R2 {resultado['r2']:.4f}   arboles {filas[-1]['arboles']}"
        )

    detalle = pd.DataFrame(filas)
    importancia = (
        pd.concat(importancias, axis=1).mean(axis=1).sort_values(ascending=False).rename("ganancia")
    )
    importancia = (importancia / importancia.sum() * 100).round(2).rename("ganancia_pct")
    return detalle, importancia.rename_axis("variable").reset_index()


# ------------------------------------------------------------------------------------
# Salidas
# ------------------------------------------------------------------------------------
def resumir(detalle: pd.DataFrame) -> pd.DataFrame:
    columnas = [c for c in detalle.columns if c not in ("configuracion", "pliegue")]
    resumen = detalle.groupby("configuracion")[columnas].agg(["mean", "std"]).round(4)
    resumen.columns = [f"{m}_{e}" for m, e in resumen.columns]
    return resumen.reset_index()


def redactar(resumen: pd.DataFrame, importancias: dict[str, pd.DataFrame]) -> str:
    mejor = resumen.loc[resumen["mae_min_mean"].idxmin()]
    mejora = (BASELINE_MAE - mejor["mae_min_mean"]) / BASELINE_MAE * 100

    lineas = [
        "# LightGBM sobre la tabla de tramos",
        "",
        f"Generado por `bakeoff/lightgbm_unico.py` el {datetime.now(UTC).date()}.",
        "",
        "## Resultado",
        "",
        "```text",
        f"mejor configuracion   {mejor['configuracion']}",
        f"MAE                   {mejor['mae_min_mean']:.4f} min   (baseline {BASELINE_MAE:.4f})",
        f"R2                    {mejor['r2_mean']:.4f}       (baseline {BASELINE_R2:.4f})",
        f"mejora sobre baseline {mejora:.1f} %",
        "```",
        "",
        "Mismas particiones (`GroupKFold` de 5 pliegues por `route_id`, dentro de train) y",
        "misma metrica que el baseline, de modo que las cifras son directamente comparables.",
        "",
        "## Comparativa de configuraciones",
        "",
        "| Configuracion | MAE (min) | RMSE (min) | R2 | MAE reparto | MAE almacen |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, fila in resumen.iterrows():
        lineas.append(
            f"| {fila['configuracion']} "
            f"| {fila['mae_min_mean']:.4f} ± {fila['mae_min_std']:.4f} "
            f"| {fila['rmse_min_mean']:.4f} ± {fila['rmse_min_std']:.4f} "
            f"| {fila['r2_mean']:.4f} ± {fila['r2_std']:.4f} "
            f"| {fila['mae_min_reparto_mean']:.4f} "
            f"| {fila['mae_min_almacen_mean']:.4f} |"
        )

    lineas += [
        "",
        "El desglose por poblacion importa: el 99,3% de los tramos son de reparto, asi que",
        "la cifra global mide practicamente solo esa poblacion. El tramo al almacen se",
        "reporta aparte para que no quede escondido.",
        "",
        "## Importancia de variables (ganancia, %)",
        "",
    ]
    for etiqueta, imp in importancias.items():
        lineas += [f"### {etiqueta}", "", "| Variable | Ganancia % |", "|---|---:|"]
        for _, fila in imp.head(12).iterrows():
            lineas.append(f"| `{fila['variable']}` | {fila['ganancia_pct']:.2f} |")
        lineas.append("")

    fila_l1 = resumen[resumen["configuracion"] == "sin clima L1"].iloc[0]
    fila_l2 = resumen[resumen["configuracion"] == "sin clima L2"].iloc[0]
    lineas += [
        "## El objetivo importa: L1 frente a L2",
        "",
        "```text",
        f"                MAE min      R2",
        f"baseline B3      {BASELINE_MAE:.4f}   {BASELINE_R2:.4f}",
        f"L1 (error abs)   {fila_l1['mae_min_mean']:.4f}   {fila_l1['r2_mean']:.4f}",
        f"L2 (error cuad)  {fila_l2['mae_min_mean']:.4f}   {fila_l2['r2_mean']:.4f}",
        "```",
        "",
        "Optimizar el error absoluto mejora el MAE pero puede empeorar el R2, y al reves.",
        "No es un fallo: el R2 penaliza los errores grandes al cuadrado, de modo que L1",
        "acepta fallar mas en los tramos raros y largos a cambio de acertar mas en el caso",
        "tipico. Como la metrica principal declarada del proyecto es el MAE en minutos, por",
        "ser la que se explica sola ante un cliente, la eleccion natural es L1; pero conviene",
        "presentar las dos y justificar la decision en vez de heredarla del valor por defecto.",
        "",
        "## Nota sobre la meteorologia (no remedida aqui)",
        "",
        "La version original de este script comparaba 'con clima' frente a 'sin clima' y",
        "concluyo que el clima no aporta (peso de ganancia ~1%, MAE ligeramente peor al",
        "incluirlo). Esa conclusion ya esta fijada en el diseno compartido de",
        "`modelado/datos.py` (`VARS_REPARTO`/`VARS_ALMACEN` no incluyen columnas de clima, y",
        "`cargar_train()` no las lee del Parquet), asi que no se puede volver a probar aqui",
        "sin reimplementar una carga de datos propia, que es justo lo que este bake-off tiene",
        "prohibido. Se documenta como un hecho ya establecido, no como una pregunta abierta.",
        "",
    ]

    lineas += [
        "## Siguiente paso",
        "",
        "CatBoost con `to_zone_id` nativa. La diferencia frente a este resultado mide cuanto",
        "aporta la identidad de la zona, que aqui se ha dejado fuera a proposito.",
    ]
    return "\n".join(lineas) + "\n"


def registrar_en_mlflow(resumen: pd.DataFrame, importancias: dict[str, pd.DataFrame]) -> str | None:
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

    ultimo = None
    for _, fila in resumen.iterrows():
        etiqueta = str(fila["configuracion"])
        with mlflow.start_run(run_name=f"lightgbm {etiqueta}") as run:
            mlflow.log_params(
                {
                    **PARAMS,
                    "objective": CONFIGURACIONES[etiqueta]["objective"],
                    "modelo": "LightGBM",
                    "configuracion": etiqueta,
                    "variables": len(CONFIGURACIONES[etiqueta]["variables"]),
                    "n_splits": N_SPLITS,
                    "estrategia": "GroupKFold por route_id, dentro de train (80%)",
                    "to_zone_id": "excluida",
                    "clima": "excluida (fijado en modelado/datos.py)",
                }
            )
            for col in resumen.columns:
                if col.endswith(("_mean", "_std")) and isinstance(fila[col], (int, float)):
                    mlflow.log_metric(col, float(fila[col]))
            mlflow.log_metric("baseline_mae_min", BASELINE_MAE)
            mlflow.log_artifact(str(REPORTS / "lightgbm_unico.md"))
            importancias[etiqueta].to_csv(REPORTS / f"lightgbm_unico importancia {etiqueta}.csv", index=False)
            mlflow.log_artifact(str(REPORTS / f"lightgbm_unico importancia {etiqueta}.csv"))
            ultimo = run.info.run_id
    return ultimo


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Cargando train (80% de las rutas)...")
    tramos = cargar_train()
    print(f"  {len(tramos):,} tramos, {tramos[GRUPO].nunique():,} rutas")

    detalles, importancias = [], {}
    for etiqueta, config in CONFIGURACIONES.items():
        print(f"\nLightGBM '{etiqueta}' ({len(config['variables'])} variables), {N_SPLITS} pliegues:")
        detalle, importancia = evaluar(tramos, config, etiqueta)
        detalles.append(detalle)
        importancias[etiqueta] = importancia

    detalle = pd.concat(detalles, ignore_index=True)
    resumen = resumir(detalle)

    detalle.to_csv(REPORTS / "lightgbm_unico detalle.csv", index=False)
    resumen.to_csv(REPORTS / "lightgbm_unico resumen.csv", index=False)
    (REPORTS / "lightgbm_unico.md").write_text(redactar(resumen, importancias), encoding="utf-8")
    run_id = registrar_en_mlflow(resumen, importancias)

    (REPORTS / "lightgbm_unico.json").write_text(
        json.dumps(
            {
                "generado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "modelo": "LightGBM",
                "n_splits": N_SPLITS,
                "params": PARAMS,
                "baseline": {"mae_min": BASELINE_MAE, "r2": BASELINE_R2},
                "resumen": resumen.to_dict(orient="records"),
                "importancia": {k: v.to_dict(orient="records") for k, v in importancias.items()},
                "mlflow_run_id": run_id,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    mejor = resumen.loc[resumen["mae_min_mean"].idxmin()]
    print("\n" + "=" * 84)
    print("RESULTADOS")
    print("=" * 84)
    print(f"{'configuracion':<18}{'MAE min':>11}{'RMSE min':>11}{'R2':>10}{'MAE reparto':>14}{'MAE almacen':>14}")
    print(f"{'baseline B3':<18}{BASELINE_MAE:>11.4f}{'':>11}{BASELINE_R2:>10.4f}{'':>14}{'':>14}")
    for _, fila in resumen.iterrows():
        print(
            f"{fila['configuracion']:<18}{fila['mae_min_mean']:>11.4f}{fila['rmse_min_mean']:>11.4f}"
            f"{fila['r2_mean']:>10.4f}{fila['mae_min_reparto_mean']:>14.4f}{fila['mae_min_almacen_mean']:>14.4f}"
        )
    print("=" * 84)
    mejora = (BASELINE_MAE - mejor["mae_min_mean"]) / BASELINE_MAE * 100
    print(f"Mejor: '{mejor['configuracion']}' -> MAE {mejor['mae_min_mean']:.4f} min, {mejora:.1f}% sobre el baseline")
    print(f"\nInforme: {(REPORTS / 'lightgbm_unico.md').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
