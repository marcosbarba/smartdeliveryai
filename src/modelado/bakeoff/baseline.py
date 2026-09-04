"""Baseline del modelo de tiempos de entrega: la referencia contra la que se compara todo.

Por que este script existe y por que va primero
-----------------------------------------------
El analisis exploratorio ya midio tres baselines y concluyo que la referencia honesta es
ajustar una recta por poblacion: R2 0,919 y MAE 0,45 minutos. Pero esas cifras se
calcularon **sobre todos los datos y con fines descriptivos**, asi que no sirven como
termino de comparacion: estan medidas sobre las mismas filas con las que se ajusto la
recta. Este script las vuelve a calcular como es debido, ajustando solo con el conjunto
de entrenamiento y midiendo en el de prueba.

Hasta que ese numero no existe, ningun resultado de LightGBM se puede interpretar: no se
sabria si mejora o si se esta comparando contra una cifra inflada.

Decisiones que no hay que reabrir
---------------------------------
- **Particion GroupKFold por route_id.** Los tramos de una misma ruta no son
  independientes. Una particion aleatoria por filas dejaria tramos de la misma ruta en
  entrenamiento y en prueba, y el resultado seria enganosamente bueno.
- **Dos poblaciones, no una.** El tramo que sale del almacen dura 30 minutos de media y
  los saltos entre entregas 1 minuto. El baseline oficial ajusta una recta a cada uno.
- **Metrica principal MAE en minutos**, porque se explica sola: "nos equivocamos de media
  en X minutos". RMSE al lado, que penaliza mas los fallos grandes.
- **No se recortan atipicos ni los tramos de cero segundos.** Son casos reales.

De donde vienen los datos
--------------------------
Este bake-off nunca lee Gold directamente: parte de `modelado.datos.cargar_train()`, el
80% de las rutas reservado para comparar arquitecturas (split fijo, hecho una sola vez,
antes de tocar ningun modelo). El 20% de test que separa esa funcion no se toca aqui ni
en ningun otro script de `bakeoff/`: es la garantia estructural de que la eleccion de
arquitectura no se contamina con filas reservadas para la evaluacion final.

Uso:

    uv run python src/modelado/bakeoff/baseline.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from modelado.datos import DISTANCIA, GRUPO, OBJETIVO, POBLACION, cargar_train
from modelado.metricas import metricas

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_MODELADO = PROJECT_ROOT / "artifacts" / "modelado"
REPORTS = ARTIFACTS_MODELADO / "reports"
MLRUNS = ARTIFACTS_MODELADO / "mlflow" / "mlruns"
# MLflow 3 dejo en mantenimiento el backend de ficheros ('./mlruns') y exige base de
# datos. SQLite es la opcion local y gratuita, y equivale al servidor de tracking que
# usaria una empresa: solo cambia la cadena de conexion.
MLFLOW_DB = ARTIFACTS_MODELADO / "mlflow" / "mlflow.db"

N_SPLITS = 5
EXPERIMENTO = "SmartDeliveryAI Tiempos"


# ------------------------------------------------------------------------------------
# Los cuatro baselines. Todos se ajustan SOLO con entrenamiento.
# ------------------------------------------------------------------------------------
def metricas_por_poblacion(test: pd.DataFrame, y_pred: np.ndarray) -> dict[str, float]:
    """Desglosa por tipo de tramo.

    Imprescindible para comparar contra los modelos posteriores: el 99,3% de las filas
    son de reparto, asi que la cifra global no dice casi nada sobre como se predice el
    trayecto al barrio, que es donde estan los errores grandes en valor absoluto.
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


def predecir_media(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """B0: predecir siempre la media. Es el suelo: un R2 de 0 por definicion."""
    return np.full(len(test), train[OBJETIVO].mean())


def predecir_velocidad_media(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """B1: distancia dividida entre una velocidad media unica.

    Es el baseline intuitivo, y el EDA comprobo que da R2 negativo. Se mantiene
    precisamente por eso: documenta que lo obvio aqui no funciona, porque una sola
    velocidad no puede describir dos poblaciones que circulan a 35 y a 11 km/h.

    Estimar una velocidad agregada del conjunto de entrenamiento NO es fuga de datos: es
    un parametro, igual que la pendiente de una regresion. Lo prohibido seria dar al
    modelo la velocidad de cada tramo, que reconstruiria el objetivo por division.
    """
    segundos = train[OBJETIVO].sum()
    km = train[DISTANCIA].sum()
    segundos_por_km = segundos / km if km > 0 else 0.0
    return (test[DISTANCIA] * segundos_por_km).to_numpy()


def predecir_recta_unica(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """B2: una unica recta tiempo ~ distancia para los dos tipos de tramo."""
    modelo = LinearRegression().fit(train[[DISTANCIA]], train[OBJETIVO])
    return np.clip(modelo.predict(test[[DISTANCIA]]), 0, None)


def predecir_recta_por_poblacion(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """B3: una recta para el tramo al barrio y otra para los saltos de reparto.

    Este es el baseline oficial del proyecto: el que hay que batir.
    """
    pred = np.empty(len(test), dtype=float)
    for grupo in (0, 1):
        mask_train = train[POBLACION] == grupo
        mask_test = (test[POBLACION] == grupo).to_numpy()
        if not mask_test.any():
            continue
        modelo = LinearRegression().fit(train.loc[mask_train, [DISTANCIA]], train.loc[mask_train, OBJETIVO])
        pred[mask_test] = modelo.predict(test.loc[mask_test, [DISTANCIA]])
    # El tiempo no puede ser negativo. Recortar en cero es fisicamente correcto y ademas
    # es la opcion conservadora: mejora el baseline, con lo que es mas dificil de batir.
    return np.clip(pred, 0, None)


BASELINES = {
    "B0 media global": predecir_media,
    "B1 distancia / velocidad media": predecir_velocidad_media,
    "B2 una recta para todo": predecir_recta_unica,
    "B3 una recta por poblacion": predecir_recta_por_poblacion,
}
BASELINE_OFICIAL = "B3 una recta por poblacion"


# ------------------------------------------------------------------------------------
# Validacion cruzada
# ------------------------------------------------------------------------------------
def evaluar(tramos: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """GroupKFold por ruta, dentro de train (80%). Devuelve el detalle por pliegue y el
    resumen agregado."""
    cv = GroupKFold(n_splits=N_SPLITS)
    filas: list[dict[str, Any]] = []
    reparto: list[dict[str, Any]] = []

    for pliegue, (idx_train, idx_test) in enumerate(cv.split(tramos, groups=tramos[GRUPO]), start=1):
        train = tramos.iloc[idx_train]
        test = tramos.iloc[idx_test]

        reparto.append(
            {
                "pliegue": pliegue,
                "rutas_train": train[GRUPO].nunique(),
                "rutas_test": test[GRUPO].nunique(),
                "tramos_train": len(train),
                "tramos_test": len(test),
            }
        )
        # Comprobacion explicita de que ninguna ruta se cuela en ambos lados.
        solapadas = set(train[GRUPO]) & set(test[GRUPO])
        if solapadas:
            raise AssertionError(f"Pliegue {pliegue}: {len(solapadas)} rutas en train y test a la vez")

        for nombre, funcion in BASELINES.items():
            y_pred = funcion(train, test)
            resultado = metricas(test[OBJETIVO].to_numpy(), y_pred)
            filas.append(
                {
                    "baseline": nombre,
                    "pliegue": pliegue,
                    **resultado,
                    **metricas_por_poblacion(test, y_pred),
                }
            )

    detalle = pd.DataFrame(filas)
    columnas = [c for c in detalle.columns if c not in ("baseline", "pliegue")]
    resumen = detalle.groupby("baseline")[columnas].agg(["mean", "std"]).round(4)
    resumen.columns = [f"{m}_{e}" for m, e in resumen.columns]
    return detalle, {"resumen": resumen.reset_index(), "reparto": pd.DataFrame(reparto)}


# ------------------------------------------------------------------------------------
# Salidas
# ------------------------------------------------------------------------------------
def redactar(resumen: pd.DataFrame, reparto: pd.DataFrame, tramos: pd.DataFrame) -> str:
    oficial = resumen.set_index("baseline").loc[BASELINE_OFICIAL]
    lineas = [
        "# Baseline del modelo de tiempos",
        "",
        f"Generado por `bakeoff/baseline.py` el {datetime.now(UTC).date()}.",
        "",
        "## Que es esto",
        "",
        "La referencia contra la que se comparan todos los modelos posteriores, ajustada",
        "**solo con el 80% de train** (`modelado.datos.cargar_train()`) y medida dentro de",
        f"el con `GroupKFold` de {N_SPLITS} pliegues agrupando por `route_id`. El 20% de test",
        "nunca participa en el bake-off de arquitecturas.",
        "",
        "## Cifra a batir",
        "",
        "```text",
        f"MAE   {oficial['mae_min_mean']:.4f} min   (desviacion entre pliegues {oficial['mae_min_std']:.4f})",
        f"RMSE  {oficial['rmse_min_mean']:.4f} min   (desviacion entre pliegues {oficial['rmse_min_std']:.4f})",
        f"R2    {oficial['r2_mean']:.4f}       (desviacion entre pliegues {oficial['r2_std']:.4f})",
        "```",
        "",
        "## Comparativa de baselines",
        "",
        "| Baseline | MAE (min) | RMSE (min) | R2 | MAE reparto | MAE almacen |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, fila in resumen.iterrows():
        lineas.append(
            f"| {fila['baseline']} | {fila['mae_min_mean']:.4f} ± {fila['mae_min_std']:.4f} "
            f"| {fila['rmse_min_mean']:.4f} ± {fila['rmse_min_std']:.4f} "
            f"| {fila['r2_mean']:.4f} ± {fila['r2_std']:.4f} "
            f"| {fila['mae_min_reparto_mean']:.4f} | {fila['mae_min_almacen_mean']:.4f} |"
        )
    lineas += [
        "",
        "El desglose por poblacion no es decorativo: el 99,3% de los tramos son de reparto,",
        "de modo que la cifra global mide casi solo esa poblacion. Los errores del tramo al",
        "almacen son treinta veces mayores en valor absoluto y apenas mueven el MAE global,",
        "pero si dominan el R2, que penaliza los errores al cuadrado.",
    ]
    lineas += [
        "",
        "## Reparto de los pliegues (dentro de train)",
        "",
        "| Pliegue | Rutas train | Rutas test | Tramos train | Tramos test |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, fila in reparto.iterrows():
        lineas.append(
            f"| {fila['pliegue']} | {fila['rutas_train']:,} | {fila['rutas_test']:,} "
            f"| {fila['tramos_train']:,} | {fila['tramos_test']:,} |"
        )
    lineas += [
        "",
        "Ninguna ruta aparece en entrenamiento y prueba a la vez: se comprueba en cada pliegue.",
        "",
        "## Datos de partida (80% de train)",
        "",
        "```text",
        f"tramos   {len(tramos):,}",
        f"rutas    {tramos[GRUPO].nunique():,}",
        f"al barrio {int((tramos[POBLACION] == 1).sum()):,}   media "
        f"{tramos.loc[tramos[POBLACION] == 1, OBJETIVO].mean() / 60:.2f} min",
        f"reparto  {int((tramos[POBLACION] == 0).sum()):,}   media "
        f"{tramos.loc[tramos[POBLACION] == 0, OBJETIVO].mean() / 60:.2f} min",
        "```",
        "",
        "## Como leer esto",
        "",
        "El baseline intuitivo (B1) da **R2 negativo**: es peor que predecir siempre la media.",
        "Una sola velocidad no puede describir dos poblaciones que circulan a velocidades muy",
        "distintas. Por eso la referencia es B3 y no B1, y por eso un R2 alto por si solo no",
        "demostraria nada en este problema.",
        "",
        "El siguiente paso es LightGBM con **estas mismas particiones (dentro de train) y esta",
        "misma metrica**.",
    ]
    return "\n".join(lineas) + "\n"


def registrar_en_mlflow(detalle: pd.DataFrame, resumen: pd.DataFrame, tramos: pd.DataFrame) -> str | None:
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

    with mlflow.start_run(run_name="baseline") as run:
        mlflow.log_params(
            {
                "estrategia": "GroupKFold por route_id, dentro de train (80%)",
                "n_splits": N_SPLITS,
                "tramos": len(tramos),
                "rutas": tramos[GRUPO].nunique(),
                "objetivo": OBJETIVO,
                "recorte_atipicos": "ninguno",
            }
        )
        for _, fila in resumen.iterrows():
            etiqueta = fila["baseline"].split(" ")[0]  # B0, B1, B2, B3
            for metrica in ("mae_min", "rmse_min", "r2"):
                mlflow.log_metric(f"{etiqueta}_{metrica}", float(fila[f"{metrica}_mean"]))
                mlflow.log_metric(f"{etiqueta}_{metrica}_std", float(fila[f"{metrica}_std"]))
        mlflow.log_artifact(str(REPORTS / "baseline.md"))
        mlflow.log_artifact(str(REPORTS / "baseline detalle.csv"))
        return run.info.run_id


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Cargando train (80% de las rutas)...")
    tramos = cargar_train()
    print(f"  {len(tramos):,} tramos, {tramos[GRUPO].nunique():,} rutas")

    print(f"Evaluando {len(BASELINES)} baselines con GroupKFold de {N_SPLITS} pliegues...")
    detalle, agregado = evaluar(tramos)
    resumen, reparto = agregado["resumen"], agregado["reparto"]

    detalle.to_csv(REPORTS / "baseline detalle.csv", index=False)
    resumen.to_csv(REPORTS / "baseline resumen.csv", index=False)
    (REPORTS / "baseline.md").write_text(redactar(resumen, reparto, tramos), encoding="utf-8")

    run_id = registrar_en_mlflow(detalle, resumen, tramos)

    (REPORTS / "baseline.json").write_text(
        json.dumps(
            {
                "generado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "n_splits": N_SPLITS,
                "tramos": len(tramos),
                "rutas": int(tramos[GRUPO].nunique()),
                "baseline_oficial": BASELINE_OFICIAL,
                "resumen": resumen.to_dict(orient="records"),
                "mlflow_run_id": run_id,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print("RESULTADOS (media +- desviacion entre pliegues)")
    print("=" * 78)
    print(f"{'baseline':<34}{'MAE min':>12}{'RMSE min':>12}{'R2':>12}")
    for _, fila in resumen.iterrows():
        print(
            f"{fila['baseline']:<34}{fila['mae_min_mean']:>12.4f}"
            f"{fila['rmse_min_mean']:>12.4f}{fila['r2_mean']:>12.4f}"
        )
    oficial = resumen.set_index("baseline").loc[BASELINE_OFICIAL]
    print("=" * 78)
    print(f"CIFRA A BATIR -> MAE {oficial['mae_min_mean']:.4f} min | R2 {oficial['r2_mean']:.4f}")
    print("=" * 78)
    print(f"\nInforme:  {(REPORTS / 'baseline.md').relative_to(PROJECT_ROOT).as_posix()}")
    if run_id:
        print(f"MLflow:   experimento '{EXPERIMENTO}', run {run_id}")


if __name__ == "__main__":
    main()
