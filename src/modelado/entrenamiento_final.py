"""Entrena el modelo final (arquitectura ganadora del bake-off: CatBoost con to_zone_id
en reparto, LightGBM en almacen) sobre el 100% de `train` (80% de las rutas), lo evalua UNA
VEZ sobre `test` (20%, nunca visto en ningun paso anterior), guarda el modelo en
artifacts/modelo/, y calcula TreeSHAP para poder explicar cada prediccion.

Este script fusiona lo que antes eran dos scripts distintos y duplicados: "06 SHAP.py"
(entrenaba el modelo "oficial" sobre el 100% de las rutas, sin test propio) y
"entrenar_modelo_agente.py" (reentrenaba aparte con un 90/10 solo para el agente,
reimportando el primero por ruta de fichero). Aqui hay un unico modelo, un unico split
(ver datos.py), y el mismo test set sirve a la vez de metrica reportada y de banco de
produccion simulada del agente — no hay razon para que sean cosas distintas.

Cofuncion nativa, no la libreria externa 'shap', para el CALCULO
--------------------------------------------------------------------
CatBoost y LightGBM traen su propio TreeSHAP exacto (`get_feature_importance(type=
"ShapValues")` y `predict(..., pred_contrib=True)`), que trata sus categoricas nativas sin
ambiguedad. Los valores se calculan con estas APIs; la libreria `shap` solo se usa despues,
para pintar, construyendo un `shap.Explanation` a partir de valores ya calculados.

Uso:

    uv run sdai-train
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor, Pool

from .datos import (
    CATEGORICAS_BASE,
    CATEGORICAS_ZONA,
    GRUPO,
    OBJETIVO,
    PARADA_TEMPRANA_ALMACEN,
    PARAMS_CB_REPARTO,
    PARAMS_LGBM_ALMACEN,
    POBLACION,
    PROJECT_ROOT,
    SEMILLA,
    SIN_ZONA,
    VARS_ALMACEN,
    VARS_REPARTO,
    calcular_coordenadas_almacen,
    calcular_distribuciones_estacion,
    cargar_train,
    guardar_banco_produccion,
    separar_validacion,
)
from .metricas import metricas

ARTIFACTS_MODELO = PROJECT_ROOT / "artifacts" / "modelo"
ARTIFACTS_MODELADO = PROJECT_ROOT / "artifacts" / "modelado"
DOCS_MODELADO = PROJECT_ROOT / "docs" / "modelado"
MLRUNS = ARTIFACTS_MODELADO / "mlflow" / "mlruns"
MLFLOW_DB = ARTIFACTS_MODELADO / "mlflow" / "mlflow.db"

EXPERIMENTO = "SmartDeliveryAI Tiempos"
N_MUESTRA_SHAP_REPARTO = 8000  # cientos de miles de filas son demasiadas para explicar una a una.


# ------------------------------------------------------------------------------------
# Entrenamiento del modelo final: seleccionar arboles con un 15% de train, reentrenar
# con el 100% de train, evaluar sobre test.
# ------------------------------------------------------------------------------------
def entrenar_final_catboost(train: pd.DataFrame, variables: list[str], categoricas: list[str], params: dict):
    sub_train, val = separar_validacion(train, SEMILLA)

    modelo_val = CatBoostRegressor(**params)
    modelo_val.fit(
        sub_train[variables], sub_train[OBJETIVO], cat_features=categoricas,
        eval_set=(val[variables], val[OBJETIVO]), use_best_model=True,
    )
    n_arboles = modelo_val.tree_count_

    params_finales = {**params, "iterations": n_arboles, "early_stopping_rounds": None, "use_best_model": False}
    modelo_final = CatBoostRegressor(**params_finales)
    modelo_final.fit(train[variables], train[OBJETIVO], cat_features=categoricas)
    return modelo_final, n_arboles


def entrenar_final_lgbm(train: pd.DataFrame, variables: list[str], categoricas: list[str], params: dict, parada: int):
    train = train[[GRUPO, OBJETIVO, *variables]].copy()
    for c in categoricas:
        train[c] = train[c].astype("category")
    sub_train, val = separar_validacion(train, SEMILLA)

    modelo_val = lgb.LGBMRegressor(**params)
    modelo_val.fit(
        sub_train[variables], sub_train[OBJETIVO],
        eval_X=val[variables], eval_y=val[OBJETIVO], eval_metric=params["metric"],
        callbacks=[lgb.early_stopping(parada, verbose=False)], categorical_feature=categoricas,
    )
    n_arboles = modelo_val.best_iteration_ or params["n_estimators"]

    params_finales = {**params, "n_estimators": n_arboles}
    modelo_final = lgb.LGBMRegressor(**params_finales)
    modelo_final.fit(train[variables], train[OBJETIVO], categorical_feature=categoricas)
    return modelo_final, n_arboles, train[categoricas].dtypes


def evaluar_en_test(
    modelo_reparto: CatBoostRegressor, modelo_almacen: lgb.LGBMRegressor, dtypes_almacen,
    test: pd.DataFrame,
) -> dict:
    """Unica evaluacion sobre el 20% reservado: nunca visto en el entrenamiento ni en la
    seleccion de arboles de ninguno de los dos especialistas."""
    test_reparto = test[test[POBLACION] == 0].reset_index(drop=True)
    test_almacen = test[test[POBLACION] == 1].reset_index(drop=True)

    pred_reparto = np.clip(modelo_reparto.predict(test_reparto[VARS_REPARTO]), 0, None)
    x_almacen = test_almacen[VARS_ALMACEN].copy()
    for c in CATEGORICAS_BASE:
        x_almacen[c] = x_almacen[c].astype(dtypes_almacen[c])
    pred_almacen = np.clip(modelo_almacen.predict(x_almacen), 0, None)

    y_real_total = np.concatenate([test_reparto[OBJETIVO].to_numpy(), test_almacen[OBJETIVO].to_numpy()])
    pred_total = np.concatenate([pred_reparto, pred_almacen])

    return {
        "reparto": metricas(test_reparto[OBJETIVO].to_numpy(), pred_reparto),
        "almacen": metricas(test_almacen[OBJETIVO].to_numpy(), pred_almacen),
        "global": metricas(y_real_total, pred_total),
        "n_test_reparto": len(test_reparto),
        "n_test_almacen": len(test_almacen),
    }


# ------------------------------------------------------------------------------------
# SHAP: se calcula con las APIs nativas de cada libreria, no con shap.TreeExplainer.
# ------------------------------------------------------------------------------------
def shap_catboost(modelo: CatBoostRegressor, x: pd.DataFrame, categoricas: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pool = Pool(x, cat_features=categoricas)
    bruto = modelo.get_feature_importance(pool, type="ShapValues")
    return bruto[:, :-1], bruto[:, -1]


def shap_lgbm(modelo: lgb.LGBMRegressor, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    bruto = modelo.booster_.predict(x, pred_contrib=True)
    return bruto[:, :-1], bruto[:, -1]


def _datos_para_grafico(x: pd.DataFrame, categoricas: list[str]) -> pd.DataFrame:
    x_num = x.copy()
    for c in categoricas:
        x_num[c] = pd.Categorical(x_num[c]).codes
    return x_num


def _guardar_graficos(figuras_dir: Path, nombre: str, shap_values: np.ndarray, x_display: pd.DataFrame, variables: list[str]) -> list[str]:
    figuras_dir.mkdir(parents=True, exist_ok=True)
    guardados: list[str] = []
    try:
        explicacion = shap.Explanation(values=shap_values, data=x_display[variables].to_numpy(), feature_names=variables)

        plt.figure()
        shap.plots.bar(explicacion, max_display=12, show=False)
        ruta = figuras_dir / f"{nombre} importancia SHAP.png"
        plt.tight_layout()
        plt.savefig(ruta, dpi=110)
        plt.close()
        guardados.append(ruta.name)

        plt.figure()
        shap.plots.beeswarm(explicacion, max_display=12, show=False)
        ruta = figuras_dir / f"{nombre} distribucion SHAP.png"
        plt.tight_layout()
        plt.savefig(ruta, dpi=110)
        plt.close()
        guardados.append(ruta.name)
    except Exception as error:  # noqa: BLE001 - fallo de trazado, no debe tumbar el resto
        plt.close("all")
        print(f"  AVISO: fallo al pintar los graficos de '{nombre}': {error}")
    return guardados


def _importancia(shap_values: np.ndarray, variables: list[str]) -> pd.DataFrame:
    medio = np.abs(shap_values).mean(axis=0) / 60
    df = pd.DataFrame({"variable": variables, "shap_medio_min": medio})
    df["importancia_pct"] = df["shap_medio_min"] / df["shap_medio_min"].sum() * 100
    return df.sort_values("shap_medio_min", ascending=False).reset_index(drop=True)


def _ejemplo_local(nombre: str, idx: int, x: pd.DataFrame, y_real: pd.Series, shap_values: np.ndarray, base_value: float, variables: list[str]) -> str:
    pred = base_value + shap_values[idx].sum()
    contribuciones = sorted(zip(variables, shap_values[idx]), key=lambda p: -abs(p[1]))[:6]
    lineas = [
        f"### Ejemplo: {nombre}", "", "```text",
        f"tiempo real:        {y_real.iloc[idx] / 60:.2f} min",
        f"prediccion:         {pred / 60:.2f} min",
        f"valor base (media): {base_value / 60:.2f} min",
        "", "mayores contribuciones (segundos):",
    ]
    for var, val in contribuciones:
        lineas.append(f"  {var:<38} {val:+8.1f} s   (valor: {x.iloc[idx][var]})")
    lineas += ["```", ""]
    return "\n".join(lineas)


# ------------------------------------------------------------------------------------
# Persistencia
# ------------------------------------------------------------------------------------
def guardar_modelos(modelo_reparto: CatBoostRegressor, modelo_almacen: lgb.LGBMRegressor, dtypes_almacen, test_metricas: dict) -> None:
    ARTIFACTS_MODELO.mkdir(parents=True, exist_ok=True)
    modelo_reparto.save_model(str(ARTIFACTS_MODELO / "reparto_catboost.cbm"))
    modelo_almacen.booster_.save_model(str(ARTIFACTS_MODELO / "almacen_lightgbm.txt"))

    manifiesto = {
        "descripcion": "Modelo hibrido: CatBoost en tramos de reparto, LightGBM en el tramo de almacen. Entrenado sobre el 80% de las rutas (train); el 20% restante (test) nunca participo en el ajuste y es tambien el banco de produccion simulada del agente.",
        "generado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "objetivo": OBJETIVO,
        "unidad_prediccion": "segundos (dividir por 60 para minutos)",
        "poblacion": f"'{POBLACION}' = 1 selecciona el especialista de almacen, 0 el de reparto",
        "reparto": {
            "fichero": "reparto_catboost.cbm",
            "formato": "CatBoost nativo (.cbm)",
            "variables": VARS_REPARTO,
            "categoricas": CATEGORICAS_ZONA,
            "categoria_desconocida": f"to_zone_id ausente -> imputar '{SIN_ZONA}'",
            "test_mae_min": test_metricas["reparto"]["mae_min"],
            "test_r2": test_metricas["reparto"]["r2"],
        },
        "almacen": {
            "fichero": "almacen_lightgbm.txt",
            "formato": "LightGBM nativo (.txt)",
            "variables": VARS_ALMACEN,
            "categoricas": CATEGORICAS_BASE,
            "dtype_categoricas": {c: str(dtypes_almacen[c]) for c in CATEGORICAS_BASE},
            # LightGBM nativo predice sobre los CODIGOS de la columna category, no sobre
            # el texto: hay que reconstruir el dtype con exactamente estas categorias
            # (mismo orden) al servir. Ver src/agente/servidor_mcp/modelos.py.
            "categorias_station_code": dtypes_almacen["station_code"].categories.tolist(),
            "test_mae_min": test_metricas["almacen"]["mae_min"],
            "test_r2": test_metricas["almacen"]["r2"],
        },
        "test_global": test_metricas["global"],
        "avisos": [
            "No calcular velocidades a partir de los tiempos: fuga de datos.",
            "same_zone se imputa a 0 cuando falta (siempre en almacen, y en un porcentaje pequeno de reparto).",
            "to_zone_id ausente se imputa como 'SIN_ZONA', tratada como categoria propia.",
            "Las predicciones se recortan a un minimo de 0 segundos.",
            "El 20% de rutas de test es el mismo dato que data/produccion_simulada/datos_produccion.parquet: no reentrenar nunca con el.",
        ],
    }
    (ARTIFACTS_MODELO / "manifiesto.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")


def redactar_model_card(n_arboles_reparto: int, n_arboles_almacen: int, test_metricas: dict, peso_zona_reparto: float, ejemplos_md: list[str], n_rutas_train: int, n_rutas_test: int) -> str:
    return "\n".join(
        [
            "# Ficha del modelo: tiempo de tramo de reparto", "",
            f"Generada automáticamente por `entrenamiento_final.py` el {datetime.now(UTC).date()}",
            "(se escribe a la vez en `artifacts/modelo/model_card.md` y aquí: mismo contenido,",
            "sin mantenimiento manual duplicado). Arquitectura seleccionada en el bake-off — ver",
            "[bakeoff_resumen.md](bakeoff_resumen.md); contrato de servicio para el agente en",
            "[instrucciones_de_servicio.md](instrucciones_de_servicio.md).", "",
            "## Qué predice", "",
            "`travel_time_seconds`: segundos entre dos paradas consecutivas de una ruta de",
            "reparto. Sumando los tramos de una ruta se obtiene la hora estimada de llegada a",
            "cualquiera de sus paradas.", "",
            "## Arquitectura", "",
            "Híbrido de dos especialistas, seleccionados por `is_depot_segment`:", "",
            "| Especialista | Modelo | Árboles | Categóricas nativas |",
            "|---|---|---:|---|",
            f"| Reparto | CatBoost | {n_arboles_reparto} | `station_code`, `to_zone_id` |",
            f"| Almacén | LightGBM | {n_arboles_almacen} | `station_code` |",
            "",
            "La división existe porque un único árbol conjunto reparte su capacidad donde",
            "está la masa de datos y desatiende el tramo de almacén (menos del 1% de los",
            "tramos totales).", "",
            "## Datos de entrenamiento y evaluación", "",
            "Split fijo por `route_id`, hecho una sola vez antes del bake-off. El test set es",
            "el mismo dato que sirve de banco de producción simulada del agente",
            "(`data/produccion_simulada/datos_produccion.parquet`).", "",
            "| Conjunto | Rutas | % |",
            "|---|---:|---:|",
            f"| `train` | {n_rutas_train:,} | 80% |",
            f"| `test` (held-out, nunca visto en el ajuste) | {n_rutas_test:,} | 20% |",
            "",
            "## Rendimiento (evaluado UNA VEZ sobre el 20% de test reservado)", "",
            "| Población | MAE (min) | R² | n |",
            "|---|---:|---:|---:|",
            f"| Global | {test_metricas['global']['mae_min']:.4f} | {test_metricas['global']['r2']:.4f} | — |",
            f"| Reparto | {test_metricas['reparto']['mae_min']:.4f} | {test_metricas['reparto']['r2']:.4f} | {test_metricas['n_test_reparto']:,} |",
            f"| Almacén | {test_metricas['almacen']['mae_min']:.4f} | {test_metricas['almacen']['r2']:.4f} | {test_metricas['n_test_almacen']:,} |",
            "",
            "## Variables más influyentes (TreeSHAP)", "",
            "`segment_distance_km` domina en las dos poblaciones. `to_zone_id` aporta",
            f"{peso_zona_reparto:.1f}% de la magnitud media de SHAP en reparto; es la razón de",
            "usar CatBoost allí en vez de LightGBM (única de las dos librerías que trata esa",
            "variable de forma nativa sin codificación manual). Detalle completo en",
            "`artifacts/modelo/informe_shap.md`.", "",
            "## Limitaciones conocidas", "",
            "- El objetivo son medias históricas por par de coordenadas, no tiempos medidos",
            "  ese día concreto: el modelo predice la duración típica de una ruta.",
            "- La meteorología no se incluye (medida aparte, no mejora el error).",
            "- Sin datos de tráfico en tiempo real ni festivos en el periodo cubierto.",
            "- Cinco semanas de verano de 2018: no se puede evaluar estacionalidad.",
            "- **`reparto_catboost.cbm` pesa varios cientos de MB.** Cargarlo una sola vez",
            "  por proceso, nunca por petición.", "",
            "## Ejemplos de explicación local", "",
            *ejemplos_md,
            "## Cómo usarlo", "",
            "1. Cargar `reparto_catboost.cbm` y `almacen_lightgbm.txt` (ver `manifiesto.json`).",
            "2. Construir las variables de `manifiesto.json` para el tramo a predecir.",
            "3. Si `is_depot_segment == 1`, usar el modelo de almacén; si no, el de reparto.",
            "4. Recortar la predicción a un mínimo de 0 segundos.", "",
            "## Reentrenamiento", "",
            "Ejecutar de nuevo `uv run sdai-train` tras regenerar Gold (`uv run sdai-pipeline`).",
            "No editar los ficheros de `artifacts/modelo/` a mano.",
        ]
    ) + "\n"


def redactar_informe(n_arboles_reparto: int, n_arboles_almacen: int, test_metricas: dict, importancia_reparto: pd.DataFrame, importancia_almacen: pd.DataFrame, graficos_reparto: list[str], graficos_almacen: list[str], ejemplos_md: list[str]) -> str:
    peso_zona = float(importancia_reparto.set_index("variable").loc["to_zone_id", "importancia_pct"]) if "to_zone_id" in importancia_reparto["variable"].values else 0.0
    lineas = [
        "# Interpretabilidad del modelo final (TreeSHAP)", "",
        f"Generado por `entrenamiento_final.py` el {datetime.now(UTC).date()}.", "",
        "Modelo entrenado sobre el 100% de `train` (80% de las rutas), evaluado UNA VEZ",
        "sobre `test` (20%, nunca visto antes), y explicado con TreeSHAP nativo de cada",
        "libreria (no `shap.TreeExplainer`, solo se usa `shap` para el trazado).", "",
        "## Rendimiento en test", "",
        "```text",
        f"global    MAE {test_metricas['global']['mae_min']:.4f} min   R2 {test_metricas['global']['r2']:.4f}",
        f"reparto   MAE {test_metricas['reparto']['mae_min']:.4f} min   R2 {test_metricas['reparto']['r2']:.4f}   ({n_arboles_reparto} arboles)",
        f"almacen   MAE {test_metricas['almacen']['mae_min']:.4f} min   R2 {test_metricas['almacen']['r2']:.4f}   ({n_arboles_almacen} arboles)",
        "```", "",
        "## Importancia de variables por SHAP (media de |valor SHAP|, minutos)", "",
        "### Reparto", "", "| Variable | \\|SHAP\\| medio (min) | % del total |", "|---|---:|---:|",
    ]
    for _, fila in importancia_reparto.head(12).iterrows():
        lineas.append(f"| `{fila['variable']}` | {fila['shap_medio_min']:.4f} | {fila['importancia_pct']:.2f}% |")
    lineas += ["", "### Almacen", "", "| Variable | \\|SHAP\\| medio (min) | % del total |", "|---|---:|---:|"]
    for _, fila in importancia_almacen.head(12).iterrows():
        lineas.append(f"| `{fila['variable']}` | {fila['shap_medio_min']:.4f} | {fila['importancia_pct']:.2f}% |")
    lineas += [
        "", f"`to_zone_id` aporta el {peso_zona:.2f}% de la magnitud media de SHAP en reparto.", "",
        "## Graficos", "", "```text",
    ]
    todos_graficos = graficos_reparto + graficos_almacen
    lineas += [f"figuras/{g}" for g in todos_graficos] if todos_graficos else ["(el trazado fallo en esta ejecucion; la importancia numerica sigue en los CSV)"]
    lineas += ["```", "", "## Ejemplos de explicacion local", "", *ejemplos_md]
    return "\n".join(lineas) + "\n"


def registrar_en_mlflow(n_arboles_reparto: int, n_arboles_almacen: int, test_metricas: dict) -> str | None:
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

    with mlflow.start_run(run_name="modelo final (catboost+lgbm, 80/20)") as run:
        mlflow.log_params({
            "modelo": "hibrido final",
            "reparto_arboles": n_arboles_reparto,
            "almacen_arboles": n_arboles_almacen,
            "entrenado_sobre": "80% de las rutas (train)",
            "evaluado_sobre": "20% de las rutas (test, held-out real)",
        })
        mlflow.log_metric("test_global_mae_min", test_metricas["global"]["mae_min"])
        mlflow.log_metric("test_global_r2", test_metricas["global"]["r2"])
        mlflow.log_metric("test_reparto_mae_min", test_metricas["reparto"]["mae_min"])
        mlflow.log_metric("test_almacen_mae_min", test_metricas["almacen"]["mae_min"])
        return run.info.run_id


def main() -> None:
    print("Cargando train (80% de las rutas) y calculando el split de test...")
    train = cargar_train()
    reparto_train = train[train[POBLACION] == 0].reset_index(drop=True)
    almacen_train = train[train[POBLACION] == 1].reset_index(drop=True)
    print(f"  train: reparto {len(reparto_train):,} filas   almacen {len(almacen_train):,} filas")

    print("\nEntrenando el especialista de reparto (CatBoost, con to_zone_id)...")
    modelo_reparto, n_arboles_reparto = entrenar_final_catboost(reparto_train, VARS_REPARTO, CATEGORICAS_ZONA, PARAMS_CB_REPARTO)
    print(f"  {n_arboles_reparto} arboles")

    print("\nEntrenando el especialista de almacen (LightGBM)...")
    modelo_almacen, n_arboles_almacen, dtypes_almacen = entrenar_final_lgbm(almacen_train, VARS_ALMACEN, CATEGORICAS_BASE, PARAMS_LGBM_ALMACEN, PARADA_TEMPRANA_ALMACEN)
    print(f"  {n_arboles_almacen} arboles")

    print("\nGuardando el banco de produccion simulada (= test set, nunca usado hasta ahora)...")
    n_produccion = guardar_banco_produccion()
    print(f"  {n_produccion:,} filas en data/produccion_simulada/datos_produccion.parquet")

    print("\nEvaluando UNA VEZ sobre el test set (20%, nunca visto en el ajuste)...")
    from .datos import cargar_gold, separar_train_test  # noqa: PLC0415 - evita recargar Gold dos veces si no hace falta
    _, test = separar_train_test(cargar_gold())
    test_metricas = evaluar_en_test(modelo_reparto, modelo_almacen, dtypes_almacen, test)
    print(f"  global    MAE {test_metricas['global']['mae_min']:.4f} min   R2 {test_metricas['global']['r2']:.4f}")
    print(f"  reparto   MAE {test_metricas['reparto']['mae_min']:.4f} min   R2 {test_metricas['reparto']['r2']:.4f}")
    print(f"  almacen   MAE {test_metricas['almacen']['mae_min']:.4f} min   R2 {test_metricas['almacen']['r2']:.4f}")

    print("\nGuardando el modelo en artifacts/modelo/...")
    guardar_modelos(modelo_reparto, modelo_almacen, dtypes_almacen, test_metricas)

    print(f"\nCalculando SHAP en reparto (muestra de {N_MUESTRA_SHAP_REPARTO:,} tramos de train)...")
    muestra_reparto = reparto_train.sample(n=min(N_MUESTRA_SHAP_REPARTO, len(reparto_train)), random_state=SEMILLA)
    shap_reparto, base_reparto = shap_catboost(modelo_reparto, muestra_reparto[VARS_REPARTO], CATEGORICAS_ZONA)

    print("Calculando SHAP en almacen (conjunto completo de train)...")
    x_almacen = almacen_train[VARS_ALMACEN].copy()
    for c in CATEGORICAS_BASE:
        x_almacen[c] = x_almacen[c].astype(dtypes_almacen[c])
    shap_almacen, base_almacen = shap_lgbm(modelo_almacen, x_almacen)

    importancia_reparto = _importancia(shap_reparto, VARS_REPARTO)
    importancia_almacen = _importancia(shap_almacen, VARS_ALMACEN)
    importancia_reparto.to_csv(ARTIFACTS_MODELO / "shap_importancia_reparto.csv", index=False)
    importancia_almacen.to_csv(ARTIFACTS_MODELO / "shap_importancia_almacen.csv", index=False)

    print("\nGenerando graficos...")
    figuras_dir = ARTIFACTS_MODELO / "figuras"
    graficos_reparto = _guardar_graficos(figuras_dir, "Reparto", shap_reparto, _datos_para_grafico(muestra_reparto[VARS_REPARTO], CATEGORICAS_ZONA), VARS_REPARTO)
    graficos_almacen = _guardar_graficos(figuras_dir, "Almacen", shap_almacen, _datos_para_grafico(x_almacen, CATEGORICAS_BASE), VARS_ALMACEN)

    print("Construyendo ejemplos de explicacion local...")
    orden = muestra_reparto[OBJETIVO].to_numpy().argsort()
    idx_corto, idx_largo = orden[len(orden) // 2], orden[-5]
    ejemplos_md = [
        _ejemplo_local("tramo de reparto tipico", idx_corto, muestra_reparto[VARS_REPARTO], muestra_reparto[OBJETIVO], shap_reparto, float(base_reparto[idx_corto]), VARS_REPARTO),
        _ejemplo_local("tramo de reparto largo (atipico)", idx_largo, muestra_reparto[VARS_REPARTO], muestra_reparto[OBJETIVO], shap_reparto, float(base_reparto[idx_largo]), VARS_REPARTO),
        _ejemplo_local("tramo de almacen (mediana)", int(np.argsort(almacen_train[OBJETIVO].to_numpy())[len(almacen_train) // 2]), x_almacen, almacen_train[OBJETIVO], shap_almacen, float(np.median(base_almacen)), VARS_ALMACEN),
    ]

    peso_zona = float(importancia_reparto.set_index("variable").loc["to_zone_id", "importancia_pct"]) if "to_zone_id" in importancia_reparto["variable"].values else 0.0
    informe = redactar_informe(n_arboles_reparto, n_arboles_almacen, test_metricas, importancia_reparto, importancia_almacen, graficos_reparto, graficos_almacen, ejemplos_md)
    (ARTIFACTS_MODELO / "informe_shap.md").write_text(informe, encoding="utf-8")

    n_rutas_train = train[GRUPO].nunique()
    n_rutas_test = test[GRUPO].nunique()
    model_card = redactar_model_card(n_arboles_reparto, n_arboles_almacen, test_metricas, peso_zona, ejemplos_md, n_rutas_train, n_rutas_test)
    (ARTIFACTS_MODELO / "model_card.md").write_text(model_card, encoding="utf-8")
    DOCS_MODELADO.mkdir(parents=True, exist_ok=True)
    (DOCS_MODELADO / "model_card.md").write_text(model_card, encoding="utf-8")

    run_id = registrar_en_mlflow(n_arboles_reparto, n_arboles_almacen, test_metricas)

    (ARTIFACTS_MODELO / "resultado.json").write_text(
        json.dumps(
            {
                "generado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "n_arboles_reparto": n_arboles_reparto,
                "n_arboles_almacen": n_arboles_almacen,
                "test": test_metricas,
                "importancia_reparto": importancia_reparto.to_dict(orient="records"),
                "importancia_almacen": importancia_almacen.to_dict(orient="records"),
                "peso_to_zone_id_pct": peso_zona,
                "mlflow_run_id": run_id,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nCalculando distribuciones historicas por estacion (solo con train)...")
    distribuciones = calcular_distribuciones_estacion(train)
    coordenadas_almacen = calcular_coordenadas_almacen()
    n_con_coordenadas = 0
    for estacion, (lat, lng) in coordenadas_almacen.items():
        if estacion in distribuciones:
            distribuciones[estacion]["depot_lat"] = lat
            distribuciones[estacion]["depot_lng"] = lng
            n_con_coordenadas += 1
    (ARTIFACTS_MODELO / "distribuciones_estacion.json").write_text(
        json.dumps(distribuciones, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  {len(distribuciones)} estaciones ({n_con_coordenadas} con coordenadas de almacen)")

    print("\n" + "=" * 80)
    print("MODELO GUARDADO")
    print("=" * 80)
    print(f"artifacts/modelo/reparto_catboost.cbm   ({n_arboles_reparto} arboles)")
    print(f"artifacts/modelo/almacen_lightgbm.txt   ({n_arboles_almacen} arboles)")
    print("artifacts/modelo/model_card.md   (y su copia en docs/modelado/model_card.md)")
    print(f"data/produccion_simulada/datos_produccion.parquet   ({n_produccion:,} filas)")
    print(f"\ntest (20%, held-out real): MAE {test_metricas['global']['mae_min']:.4f} min, R2 {test_metricas['global']['r2']:.4f}")


if __name__ == "__main__":
    main()
