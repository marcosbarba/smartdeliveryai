"""Carga y sirve el modelo hibrido del agente (CatBoost reparto + LightGBM almacen).

Replica el contrato de docs/modelado/instrucciones_de_servicio.md contra artifacts/modelo/
(el unico modelo del proyecto, entrenado sobre el 100% de train y evaluado una vez sobre
test — ver src/modelado/entrenamiento_final.py).

Los modelos se cargan una sola vez por proceso (funcion con cache), nunca por peticion: el
de reparto pesa varios cientos de MB.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# src/agente/servidor_mcp/modelos.py -> servidor_mcp -> agente -> src -> smartdeliveryai
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "artifacts" / "modelo"

SIN_ZONA = "SIN_ZONA"


class ModeloNoDisponible(RuntimeError):
    """El fichero del modelo no esta en artifacts/modelo/. Hay que ejecutar
    'uv run sdai-train' antes de poder predecir tramos de reparto."""


@lru_cache(maxsize=1)
def cargar_manifiesto() -> dict:
    ruta = MODELS_DIR / "manifiesto.json"
    if not ruta.exists():
        raise ModeloNoDisponible(
            f"No existe {ruta}. Ejecuta antes 'uv run sdai-train'."
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def cargar_modelo_reparto() -> CatBoostRegressor | None:
    ruta = MODELS_DIR / "reparto_catboost.cbm"
    if not ruta.exists():
        return None
    modelo = CatBoostRegressor()
    modelo.load_model(str(ruta))
    return modelo


@lru_cache(maxsize=1)
def cargar_modelo_almacen() -> lgb.Booster | None:
    ruta = MODELS_DIR / "almacen_lightgbm.txt"
    if not ruta.exists():
        return None
    return lgb.Booster(model_file=str(ruta))


@lru_cache(maxsize=1)
def categorias_station_code_almacen() -> list[str]:
    return cargar_manifiesto()["almacen"]["categorias_station_code"]


def imputar_comunes(filas: pd.DataFrame) -> pd.DataFrame:
    filas = filas.copy()
    if "same_zone" in filas.columns:
        filas["same_zone"] = filas["same_zone"].fillna(0)
    if "to_zone_id" in filas.columns:
        filas["to_zone_id"] = filas["to_zone_id"].astype(object).where(filas["to_zone_id"].notna(), SIN_ZONA).astype(str)
    if "station_code" in filas.columns:
        filas["station_code"] = filas["station_code"].astype(str)
    return filas


def predecir_reparto(filas: pd.DataFrame) -> np.ndarray:
    """filas debe traer, como minimo, las columnas de manifiesto['reparto']['variables']."""
    modelo = cargar_modelo_reparto()
    if modelo is None:
        raise ModeloNoDisponible(
            "reparto_catboost.cbm no esta en artifacts/modelo/. Este modelo cubre el 99% de "
            "los tramos: sin el no se pueden predecir rutas de reparto. Ejecuta "
            "'uv run sdai-train'."
        )
    variables = cargar_manifiesto()["reparto"]["variables"]
    x = imputar_comunes(filas)[variables]
    pred = modelo.predict(x)
    return np.clip(pred, 0, None)


def predecir_almacen(filas: pd.DataFrame) -> np.ndarray:
    """filas debe traer, como minimo, las columnas de manifiesto['almacen']['variables']."""
    modelo = cargar_modelo_almacen()
    if modelo is None:
        raise ModeloNoDisponible(
            "almacen_lightgbm.txt no esta en artifacts/modelo/. Ejecuta 'uv run sdai-train'."
        )
    variables = cargar_manifiesto()["almacen"]["variables"]
    x = imputar_comunes(filas)[variables].copy()
    # LightGBM nativo predice sobre el CODIGO de la columna category, no sobre el texto: hay
    # que fijar las categorias exactas del entrenamiento (mismo orden), o una prediccion con
    # pocas filas recodificaria las estaciones presentes empezando en 0 y el modelo leeria
    # una estacion distinta a la real.
    categorias = categorias_station_code_almacen()
    x["station_code"] = pd.Categorical(x["station_code"], categories=categorias)
    pred = modelo.predict(x)
    return np.clip(pred, 0, None)


def predecir_mixto(filas: pd.DataFrame) -> np.ndarray:
    """Predice un conjunto de tramos que puede mezclar el tramo de almacen
    (is_depot_segment=1) con tramos de reparto (is_depot_segment=0), enrutando cada fila al
    especialista correcto. Unico punto de la logica "que modelo segun is_depot_segment": no
    duplicarla en otros ficheros (asi se coló el bug de optimizador.py usando siempre
    predecir_reparto, incluso para el tramo de almacen)."""
    es_almacen = (filas["is_depot_segment"] == 1).to_numpy()
    tiempos = np.zeros(len(filas))
    if es_almacen.any():
        tiempos[es_almacen] = predecir_almacen(filas.loc[es_almacen])
    if (~es_almacen).any():
        tiempos[~es_almacen] = predecir_reparto(filas.loc[~es_almacen])
    return tiempos


def modelos_disponibles() -> dict:
    return {
        "reparto": cargar_modelo_reparto() is not None,
        "almacen": cargar_modelo_almacen() is not None,
    }
