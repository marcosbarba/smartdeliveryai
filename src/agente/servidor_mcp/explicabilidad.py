"""Explicacion local de una prediccion: TreeSHAP exacto, mismo patron que
src/modelado/entrenamiento_final.py (API nativa de cada libreria, no shap.TreeExplainer).
"""

from __future__ import annotations

import pandas as pd
from catboost import Pool

from .modelos import (
    ModeloNoDisponible,
    imputar_comunes,
    cargar_manifiesto,
    cargar_modelo_almacen,
    cargar_modelo_reparto,
    categorias_station_code_almacen,
)


def _shap_reparto(x: pd.DataFrame, categoricas: list[str]) -> tuple[list[float], float]:
    modelo = cargar_modelo_reparto()
    if modelo is None:
        raise ModeloNoDisponible("reparto_catboost.cbm no esta disponible para explicar.")
    pool = Pool(x, cat_features=categoricas)
    bruto = modelo.get_feature_importance(pool, type="ShapValues")
    return bruto[0, :-1].tolist(), float(bruto[0, -1])


def _shap_almacen(x: pd.DataFrame) -> tuple[list[float], float]:
    modelo = cargar_modelo_almacen()
    if modelo is None:
        raise ModeloNoDisponible("almacen_lightgbm.txt no esta disponible para explicar.")
    x = x.copy()
    x["station_code"] = pd.Categorical(x["station_code"], categories=categorias_station_code_almacen())
    bruto = modelo.predict(x, pred_contrib=True)
    return bruto[0, :-1].tolist(), float(bruto[0, -1])


def explicar_tramo(fila: dict) -> dict:
    """Contribucion de cada variable (en segundos) a la prediccion de un tramo concreto,
    ordenadas por magnitud. valor_base es lo que predeciria el modelo sin ninguna
    informacion extra (la media de su poblacion de entrenamiento)."""
    manifiesto = cargar_manifiesto()
    es_almacen = int(fila.get("is_depot_segment", 0)) == 1
    bloque = manifiesto["almacen"] if es_almacen else manifiesto["reparto"]
    variables = bloque["variables"]
    categoricas = bloque["categoricas"]

    x = imputar_comunes(pd.DataFrame([fila]))[variables]
    if es_almacen:
        valores_shap, valor_base = _shap_almacen(x)
    else:
        valores_shap, valor_base = _shap_reparto(x, categoricas)

    prediccion = valor_base + sum(valores_shap)
    contribuciones = sorted(
        (
            {"variable": var, "valor_entrada": x.iloc[0][var], "contribucion_segundos": round(val, 1)}
            for var, val in zip(variables, valores_shap)
        ),
        key=lambda c: -abs(c["contribucion_segundos"]),
    )
    return {
        "prediccion_segundos": round(max(prediccion, 0), 1),
        "valor_base_segundos": round(valor_base, 1),
        "especialista": "almacen" if es_almacen else "reparto",
        "contribuciones": contribuciones,
    }
