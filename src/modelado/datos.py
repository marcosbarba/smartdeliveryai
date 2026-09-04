"""Carga de Gold, split train/test fijo, y los artefactos derivados que dependen de el
(banco de produccion simulada, distribuciones historicas por estacion, coordenadas de
almacen). Punto unico de estas responsabilidades: ni el bake-off de arquitecturas
(bakeoff/*.py) ni entrenamiento_final.py ni analisis_errores.py las reimplementan.

Por que el split se hace aqui, antes de cualquier arquitectura
-------------------------------------------------------------------
El diseno anterior (dos scripts distintos: uno que reentrenaba sobre el 100% de las rutas
sin test propio, y otro que reservaba un 10% aparte solo para la demo del agente) tenia dos
problemas: el modelo "oficial" nunca se evaluaba contra un held-out fijo, y el bake-off de
arquitecturas se decidia mirando el 100% de los datos, así que la eleccion de arquitectura
podia estar sesgada por filas que en cualquiera de los dos disenos acababan "reservadas"
para otra cosa. Aqui hay un unico split, hecho una sola vez con semilla fija
(`separar_train_test`), y CUALQUIER script que entrene o compare arquitecturas debe pasar
por `cargar_train()`, nunca por `cargar_gold()` a pelo: es la garantia estructural de que el
test set no se toca hasta la evaluacion final.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_TRAMOS = PROJECT_ROOT / "data" / "gold" / "Dataset Tramos Ruta" / "Dataset Tramos Ruta.parquet"
DATOS_PRODUCCION_DIR = PROJECT_ROOT / "data" / "produccion_simulada"

OBJETIVO = "travel_time_seconds"
GRUPO = "route_id"
POBLACION = "is_depot_segment"
DISTANCIA = "segment_distance_km"
ZONA = "to_zone_id"
SIN_ZONA = "SIN_ZONA"

# Split train/test: el clasico 80/20, hecho una sola vez, antes de comparar arquitecturas.
FRACCION_TEST = 0.20
SEMILLA_SPLIT = 7

# Split de validacion DENTRO de train, solo para decidir cuantos arboles usar (parada
# temprana) al entrenar el modelo final. Eje de aleatoriedad distinto de SEMILLA_SPLIT
# a proposito: son dos preguntas distintas (que rutas son test / que rutas paran el
# entrenamiento), no debe ser la misma particion para las dos cosas.
FRACCION_VALIDACION = 0.15
SEMILLA = 42

CATEGORICAS_BASE = ["station_code"]
CATEGORICAS_ZONA = ["station_code", ZONA]

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
    *CATEGORICAS_ZONA,
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
    *CATEGORICAS_BASE,
]

PARAMS_CB_REPARTO = {
    "loss_function": "MAE",
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "random_seed": SEMILLA,
    "verbose": False,
    "early_stopping_rounds": 50,
    "thread_count": -1,
    "train_dir": str(PROJECT_ROOT / "artifacts" / "modelado" / "catboost_info"),
}
PARAMS_LGBM_ALMACEN = {
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
# Carga de Gold (columnas de modelo unicamente) y split train/test
# ------------------------------------------------------------------------------------
def _columnas_modelo() -> list[str]:
    return sorted({GRUPO, "segment_position", OBJETIVO, POBLACION, *VARS_REPARTO, *VARS_ALMACEN})


def cargar_gold() -> pd.DataFrame:
    """Todas las filas de Gold (Dataset Tramos Ruta), solo las columnas que usa algun
    modelo. NUNCA se entrena ni se compara arquitecturas directamente sobre esto: pasa
    siempre por separar_train_test/cargar_train/cargar_test."""
    ficheros = sorted(glob.glob(str(GOLD_TRAMOS / "*.parquet")))
    if not ficheros:
        raise FileNotFoundError(f"No hay Parquet en {GOLD_TRAMOS}. Ejecuta antes 'sdai-pipeline'.")
    tramos = pd.concat([pd.read_parquet(f, columns=_columnas_modelo()) for f in ficheros], ignore_index=True)
    tramos = tramos.sort_values([GRUPO, "segment_position"], ignore_index=True)
    tramos["same_zone"] = tramos["same_zone"].fillna(0)
    tramos["station_code"] = tramos["station_code"].astype(str)
    tramos[ZONA] = tramos[ZONA].astype(object).where(tramos[ZONA].notna(), SIN_ZONA).astype(str)
    return tramos


def _rutas_test(rutas: np.ndarray, fraccion_test: float, semilla: int) -> set:
    generador = np.random.default_rng(semilla)
    n_test = max(1, int(len(rutas) * fraccion_test))
    return set(generador.choice(rutas, size=n_test, replace=False))


def separar_train_test(
    tramos: pd.DataFrame, fraccion_test: float = FRACCION_TEST, semilla: int = SEMILLA_SPLIT
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split fijo y deterministico por route_id. Llamar siempre con los valores por
    defecto salvo que se este documentando explicitamente un experimento con otro ratio:
    el ratio y la semilla son parte del contrato de que train/test signifiquen lo mismo
    en todo el proyecto (bake-off, entrenamiento final, banco de produccion simulada)."""
    rutas = tramos[GRUPO].drop_duplicates().sort_values().to_numpy()
    rutas_test = _rutas_test(rutas, fraccion_test, semilla)
    mask_test = tramos[GRUPO].isin(rutas_test).to_numpy()
    train = tramos.loc[~mask_test].reset_index(drop=True)
    test = tramos.loc[mask_test].reset_index(drop=True)
    return train, test


def cargar_train() -> pd.DataFrame:
    """El 80% de las rutas. Unica fuente de datos valida para elegir o entrenar
    arquitecturas: bakeoff/*.py y entrenamiento_final.py parten de aqui, nunca de
    cargar_gold() ni de cargar_test()."""
    train, _ = separar_train_test(cargar_gold())
    return train


def cargar_test() -> pd.DataFrame:
    """El 20% de rutas reservado, nunca visto en ningun ajuste ni seleccion de
    arquitectura. Es el mismo dato que se guarda como banco de produccion simulada."""
    _, test = separar_train_test(cargar_gold())
    return test


def separar_validacion(train: pd.DataFrame, semilla: int = SEMILLA) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DENTRO de train, solo para decidir cuantos arboles usar (parada temprana)
    al entrenar el modelo final sobre el 100% de train. No confundir con
    separar_train_test: ese fija el test set global, este es interno a cada
    entrenamiento."""
    rutas = train[GRUPO].drop_duplicates().sort_values().to_numpy()
    generador = np.random.default_rng(semilla)
    n_val = max(1, int(len(rutas) * FRACCION_VALIDACION))
    rutas_val = set(generador.choice(rutas, size=n_val, replace=False))
    mask_val = train[GRUPO].isin(rutas_val).to_numpy()
    return train.loc[~mask_val], train.loc[mask_val]


# ------------------------------------------------------------------------------------
# Banco de produccion simulada: literalmente el test set, con travel_time_seconds intacto
# ------------------------------------------------------------------------------------
def guardar_banco_produccion() -> int:
    """Guarda el test set completo (todas las columnas de Gold, no solo las de modelo)
    como banco de produccion simulada: mismo dato que evalua el modelo final, consumido
    tambien por src/agente/servidor_mcp/simulador_datos.py para la demo. No hay un
    segundo split para el agente: es el mismo test, literal."""
    _, rutas_test = _split_rutas_completo()
    ficheros = sorted(glob.glob(str(GOLD_TRAMOS / "*.parquet")))
    trozos = [pd.read_parquet(f) for f in ficheros]
    completo = pd.concat(trozos, ignore_index=True)
    banco = completo[completo[GRUPO].isin(rutas_test)].sort_values([GRUPO, "segment_position"], ignore_index=True)

    DATOS_PRODUCCION_DIR.mkdir(parents=True, exist_ok=True)
    banco.to_parquet(DATOS_PRODUCCION_DIR / "datos_produccion.parquet", index=False)
    return len(banco)


def _split_rutas_completo() -> tuple[set, set]:
    """Las rutas de train/test como sets, calculadas sobre TODAS las route_id de Gold
    (no solo las que traen columnas de modelo) — para que guardar_banco_produccion y
    calcular_coordenadas_almacen (que necesitan columnas que cargar_gold() no carga,
    como from_lat/from_lng) usen exactamente la misma particion sin tener que leer Gold
    dos veces con las columnas de modelo."""
    ficheros = sorted(glob.glob(str(GOLD_TRAMOS / "*.parquet")))
    rutas = pd.concat(
        [pd.read_parquet(f, columns=[GRUPO]) for f in ficheros], ignore_index=True
    )[GRUPO].drop_duplicates().sort_values().to_numpy()
    rutas_test = _rutas_test(rutas, FRACCION_TEST, SEMILLA_SPLIT)
    return set(rutas) - rutas_test, rutas_test


# ------------------------------------------------------------------------------------
# Distribuciones historicas por estacion (autofill de rutas manuales del agente)
# ------------------------------------------------------------------------------------
def calcular_distribuciones_estacion(tramos_entrenamiento: pd.DataFrame) -> dict:
    """Percentiles por station_code de las variables que un alta manual no puede
    conocer. Se calculan SOLO sobre train (nunca sobre test/produccion simulada), para
    que el autofill no se apoye, ni indirectamente, en datos reservados para simular
    casos nuevos."""
    columnas = [
        "packages_at_destination", "service_time_at_destination_seconds",
        "volume_at_destination_cm3", "packages_with_window_at_destination",
    ]
    percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    distribuciones: dict = {}
    for estacion, grupo in tramos_entrenamiento.groupby("station_code"):
        entrada = {}
        for columna in columnas:
            valores = grupo[columna].dropna()
            entrada[columna] = {str(p): float(valores.quantile(p)) for p in percentiles} if len(valores) else {}
        zonas_vistas = grupo["to_zone_id"].value_counts().index.tolist()
        entrada["to_zone_id_vistas"] = zonas_vistas[:50]
        entrada["n_tramos"] = int(len(grupo))
        distribuciones[str(estacion)] = entrada
    return distribuciones


def calcular_coordenadas_almacen() -> dict[str, tuple[float, float]]:
    """Coordenada aproximada del almacen de cada estacion (media de from_lat/from_lng de
    sus tramos de almacen), calculada solo con rutas de train. Hace falta aparte porque
    from_lat/from_lng no son variables de modelo: cargar_gold() no las trae."""
    rutas_train, _ = _split_rutas_completo()
    ficheros = sorted(glob.glob(str(GOLD_TRAMOS / "*.parquet")))
    columnas = ["route_id", "station_code", "is_depot_segment", "from_lat", "from_lng"]
    depot = pd.concat([pd.read_parquet(f, columns=columnas) for f in ficheros], ignore_index=True)
    depot = depot[(depot["is_depot_segment"] == 1) & (depot["route_id"].isin(rutas_train))]
    coordenadas = depot.groupby("station_code")[["from_lat", "from_lng"]].mean()
    return {estacion: (float(fila["from_lat"]), float(fila["from_lng"])) for estacion, fila in coordenadas.iterrows()}
