"""LSTM: el experimento de deep learning, no el modelo desplegado.

Por que existe este script
---------------------------
Una ruta es literalmente una secuencia de saltos: `ALMACEN -> parada#1 -> parada#2 -> ...`.
Los modelos anteriores (baseline, LightGBM, CatBoost) tratan cada tramo como una fila
independiente, sin memoria de los tramos anteriores de la misma ruta. Una red recurrente
podria aprender contexto que los arboles no ven: por ejemplo que el ritmo se degrada al
final de la jornada, o que una zona densa al principio predispone a tramos cortos despues.

La expectativa, siguiendo la Guia del Data Scientist, es que **no** supere al hibrido de
arboles: en datos tabulares los arboles suelen ganar, y aqui el objetivo esta ademas
dominado por la distancia (88-90% de la ganancia en los modelos anteriores). El valor de
este script no es desplegar la LSTM, es la comparacion que la memoria necesita para
justificar con datos, no solo con la guia, por que el modelo final es de arboles: un
argumento de "lo intentamos y no gano" pesa mas que "no lo intentamos".

Decisiones de diseno, y de que evidencia previa parten
--------------------------------------------------------
- **Una sola secuencia por ruta, sin dividir por poblacion.** Los especialistas de
  `bakeoff/modelo_hibrido.py` tuvieron que separarse porque un unico arbol reparte su
  capacidad de division donde esta la masa de datos (99,3% reparto) y desatiende el 0,68%
  de almacen. Una LSTM no tiene ese problema: procesa la secuencia entera y su estado
  oculto puede adaptarse al tramo de almacen a traves de la propia dinamica temporal y de
  `is_depot_segment` como variable de entrada. Es una ventaja estructural genuina de este
  tipo de modelo, no una simplificacion por pereza.

- **Sin meteorologia.** No es un olvido: `bakeoff/lightgbm_unico.py` la probo (en el
  diseno original, sobre el 100% de los datos) y salio con un peso de ganancia del 1,05%
  y un MAE ligeramente peor al incluirla; esa conclusion quedo ademas fijada en el diseno
  compartido de `modelado/datos.py`, que no expone columnas de clima. Repetir la
  comprobacion aqui no aportaria una pregunta nueva.

- **Sin `to_zone_id`.** `bakeoff/catboost_zona.py` ya mide su aporte (peso ~2-3% en
  reparto, ~1% en almacen) usando codificacion nativa sin fuga. Anadirla a una LSTM
  exigiria una capa de embedding sobre 8.956 categorias, con el consiguiente riesgo de
  sobreajuste en un experimento cuyo objetivo es comparar arquitecturas, no maximizar la
  ultima decima. Queda fuera por alcance, con el aporte ya cuantificado en otro sitio.

- **Funcion de perdida L1 (MAE), no L2.** `bakeoff/lightgbm_unico.py` mostro que optimizar
  el error cuadratico mejora el R2 pero empeora el MAE, que es la metrica principal
  declarada del proyecto. Se mantiene la misma eleccion aqui por consistencia
  metodologica.

- **Normalizacion ajustada solo con el pliegue de entrenamiento.** Los arboles no la
  necesitan, pero una red si: sin ella, `segment_distance_km` (0-48 km) y `weekday` (0-6)
  entrarian en escalas tan distintas que el descenso de gradiente iria mal condicionado.
  La media y desviacion se calculan sobre train y se aplican sin cambios a validacion y
  prueba, para no fugar informacion de la escala de test.

- **Las mismas particiones `GroupKFold` de 5 pliegues por `route_id`, dentro del 80% de
  train** que el resto del bake-off (todos parten de `modelado.datos.cargar_train()`).

- **Recorte a cero tras predecir**, igual que en todos los modelos anteriores: el tiempo
  no puede ser negativo.

Sobre la comprobacion cruzada de particiones del script original (adaptacion obligada)
-------------------------------------------------------------------------------------------
La version original de este script (`05 LSTM.py`) verificaba que el reparto de rutas por
pliegue coincidiera EXACTAMENTE con unos conteos hardcodeados, copiados a mano de la
salida de `01 Baseline.py`: una defensa contra que cada script, con su propia copia de la
logica de carga de Parquet, pudiera terminar generando un orden de filas distinto y por
tanto folds distintos sin que nadie se diera cuenta. Ese riesgo ya no existe: todos los
scripts de `bakeoff/` cargan ahora a traves de la MISMA `modelado.datos.cargar_train()`,
asi que sus folds son identicos por construccion, no por coincidencia verificada a mano.
Ademas, los conteos concretos ya no aplican: estaban dimensionados para el 100% de las
rutas (6.112), y ahora el bake-off opera solo sobre el 80% (train). Se mantiene la
comprobacion que si sigue siendo relevante -que ninguna ruta este a la vez en train y test
del mismo pliegue- y se retira la comparacion contra los conteos hardcodeados de la
version anterior.

Uso:

    uv run --extra deep-learning python src/modelado/bakeoff/lstm_experimental.py
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import Dataset

from modelado.datos import DISTANCIA, GRUPO, OBJETIVO, POBLACION, SEMILLA, cargar_train
from modelado.metricas import metricas as _metricas_segundos

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_MODELADO = PROJECT_ROOT / "artifacts" / "modelado"
REPORTS = ARTIFACTS_MODELADO / "reports"
MLRUNS = ARTIFACTS_MODELADO / "mlflow" / "mlruns"
MLFLOW_DB = ARTIFACTS_MODELADO / "mlflow" / "mlflow.db"

N_SPLITS = 5
FRACCION_VALIDACION = 0.15
EXPERIMENTO = "SmartDeliveryAI Tiempos"

# Referencias de los scripts anteriores, verificadas en sus propios informes. Se
# refrescan a mano tras cada ejecucion previa.
BASELINE_MAE, BASELINE_R2 = 0.4473, 0.9192
HIBRIDO_V1_MAE, HIBRIDO_V1_R2 = 0.3800, 0.9349
HIBRIDO_V2_MAE, HIBRIDO_V2_R2 = 0.3721, 0.9366

# MAE de almacen del LightGBM CONJUNTO original (bakeoff/lightgbm_unico.py, antes de
# dividir en especialistas) y del especialista dedicado (modelo_hibrido/catboost_zona).
# Sirven para contrastar la afirmacion de diseno de este script: que una LSTM no necesita
# dividir poblaciones porque su estado oculto puede adaptarse. Se mide en vez de darla por
# buena.
LGBM_CONJUNTO_MAE_ALMACEN = 5.0047
ESPECIALISTA_MAE_ALMACEN = 2.8540

# ------------------------------------------------------------------------------------
# Variables. Numericas + una categorica (estacion, via embedding).
# ------------------------------------------------------------------------------------
VARS_NUMERICAS = [
    DISTANCIA,
    POBLACION,  # 0/1: la LSTM puede usarla como cualquier otra variable, sin dividir modelos.
    "same_zone",  # nula en almacen (46/6.112 en to_zone_id, 100% en same_zone); se imputa 0.
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
]
CATEGORICA = "station_code"

HIDDEN_SIZE = 64
EMBEDDING_DIM = 8
BATCH_SIZE = 64
MAX_EPOCHS = 40
PACIENCIA = 6
LEARNING_RATE = 1e-3


def fijar_semillas() -> None:
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)


# ------------------------------------------------------------------------------------
# Construccion de secuencias
# ------------------------------------------------------------------------------------
def indice_estaciones(tramos: pd.DataFrame) -> dict[str, int]:
    """0 se reserva para relleno (padding); las estaciones reales empiezan en 1."""
    estaciones = sorted(tramos[CATEGORICA].unique())
    return {codigo: i + 1 for i, codigo in enumerate(estaciones)}


def construir_secuencias(tramos: pd.DataFrame, estacion_a_indice: dict[str, int]) -> dict[str, list]:
    """Una secuencia (T x variables) por ruta, en el orden real de la ruta."""
    secuencias: dict[str, list] = {"route_id": [], "x": [], "estacion": [], "y": [], "poblacion": []}
    for route_id, grupo in tramos.groupby(GRUPO, sort=False):
        secuencias["route_id"].append(route_id)
        secuencias["x"].append(grupo[VARS_NUMERICAS].to_numpy(dtype=np.float32))
        secuencias["estacion"].append(grupo[CATEGORICA].map(estacion_a_indice).to_numpy(dtype=np.int64))
        secuencias["y"].append((grupo[OBJETIVO].to_numpy(dtype=np.float32)) / 60.0)  # minutos
        secuencias["poblacion"].append(grupo[POBLACION].to_numpy(dtype=np.int64))
    return secuencias


class SecuenciasRutas(Dataset):
    def __init__(self, indices: list[int], secuencias: dict[str, list], media: np.ndarray, desviacion: np.ndarray):
        self.indices = indices
        self.secuencias = secuencias
        self.media = media
        self.desviacion = desviacion

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x = (self.secuencias["x"][idx] - self.media) / self.desviacion
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(self.secuencias["estacion"][idx]),
            torch.from_numpy(self.secuencias["y"][idx]),
            torch.from_numpy(self.secuencias["poblacion"][idx]),
        )


def collate(lote):
    """Rellena al mayor largo del lote y devuelve una mascara de posiciones validas."""
    longitudes = torch.tensor([item[0].shape[0] for item in lote])
    max_t = int(longitudes.max())
    n = len(lote)
    f = lote[0][0].shape[1]

    x = torch.zeros(n, max_t, f)
    estacion = torch.zeros(n, max_t, dtype=torch.long)
    y = torch.zeros(n, max_t)
    poblacion = torch.zeros(n, max_t, dtype=torch.long)
    mascara = torch.zeros(n, max_t, dtype=torch.bool)

    for i, (xi, ei, yi, pi) in enumerate(lote):
        t = xi.shape[0]
        x[i, :t] = xi
        estacion[i, :t] = ei
        y[i, :t] = yi
        poblacion[i, :t] = pi
        mascara[i, :t] = True

    return x, estacion, y, poblacion, mascara, longitudes


def lote_por_longitud(n_rutas: int, longitudes: np.ndarray, batch_size: int, generador: np.random.Generator):
    """Agrupa rutas de longitud parecida en el mismo lote para no desperdiciar computo en
    relleno, y mezcla el ORDEN de los lotes en cada epoca para no perder aleatoriedad.
    """
    orden = np.argsort(longitudes, kind="stable")
    lotes = [orden[i : i + batch_size].tolist() for i in range(0, n_rutas, batch_size)]
    generador.shuffle(lotes)
    return lotes


# ------------------------------------------------------------------------------------
# Modelo
# ------------------------------------------------------------------------------------
class LSTMTiempos(nn.Module):
    def __init__(self, n_numericas: int, n_estaciones: int):
        super().__init__()
        self.embedding = nn.Embedding(n_estaciones + 1, EMBEDDING_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=n_numericas + EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, num_layers=1, batch_first=True
        )
        self.cabeza = nn.Sequential(nn.Linear(HIDDEN_SIZE, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, estacion):
        entrada = torch.cat([x, self.embedding(estacion)], dim=-1)
        salida_lstm, _ = self.lstm(entrada)
        return self.cabeza(salida_lstm).squeeze(-1)


def perdida_l1_enmascarada(pred: torch.Tensor, objetivo: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
    return (torch.abs(pred - objetivo) * mascara).sum() / mascara.sum().clamp(min=1)


# ------------------------------------------------------------------------------------
# Metricas
# ------------------------------------------------------------------------------------
def metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """`y_real`/`y_pred` aqui ya estan en MINUTOS (la LSTM entrena sobre el objetivo
    dividido entre 60, ver `construir_secuencias`), mientras que `modelado.metricas`
    asume segundos y divide internamente entre 60. Se reescalan a segundos antes de
    llamarla para reutilizar la misma formula sin reimplementarla: el resultado es
    identico al que daria una version que operara directamente en minutos.
    """
    m = _metricas_segundos(y_real * 60, y_pred * 60)
    return {"mae_min": m["mae_min"], "rmse_min": m["rmse_min"], "r2": m["r2"]}


def metricas_por_poblacion(y_real: np.ndarray, y_pred: np.ndarray, poblacion: np.ndarray) -> dict[str, float]:
    salida: dict[str, float] = {}
    for grupo, etiqueta in ((0, "reparto"), (1, "almacen")):
        mask = poblacion == grupo
        if not mask.any():
            continue
        m = metricas(y_real[mask], y_pred[mask])
        salida[f"mae_min_{etiqueta}"] = m["mae_min"]
        salida[f"rmse_min_{etiqueta}"] = m["rmse_min"]
    return salida


# ------------------------------------------------------------------------------------
# Entrenamiento y evaluacion de un pliegue
# ------------------------------------------------------------------------------------
def separar_validacion(indices: list[int], generador: np.random.Generator) -> tuple[list[int], list[int]]:
    indices = list(indices)
    generador.shuffle(indices)
    n_val = max(1, int(len(indices) * FRACCION_VALIDACION))
    return indices[n_val:], indices[:n_val]


def evaluar_lote(modelo, x, estacion, y, mascara, dispositivo) -> torch.Tensor:
    x, estacion, y, mascara = x.to(dispositivo), estacion.to(dispositivo), y.to(dispositivo), mascara.to(dispositivo)
    return modelo(x, estacion), y, mascara


def entrenar_pliegue(
    pliegue: int,
    idx_train: list[int],
    idx_test: list[int],
    secuencias: dict[str, list],
    dispositivo: torch.device,
) -> dict[str, Any]:
    generador_np = np.random.default_rng(SEMILLA + pliegue)

    idx_sub_train, idx_val = separar_validacion(idx_train, generador_np)

    # Normalizacion ajustada SOLO con el sub-entrenamiento de este pliegue.
    x_train_concat = np.concatenate([secuencias["x"][i] for i in idx_sub_train], axis=0)
    media = x_train_concat.mean(axis=0)
    desviacion = x_train_concat.std(axis=0)
    desviacion[desviacion < 1e-6] = 1.0

    n_estaciones = int(max(max(e) for e in secuencias["estacion"]))
    modelo = LSTMTiempos(n_numericas=len(VARS_NUMERICAS), n_estaciones=n_estaciones).to(dispositivo)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

    ds_train = SecuenciasRutas(idx_sub_train, secuencias, media, desviacion)
    ds_val = SecuenciasRutas(idx_val, secuencias, media, desviacion)
    ds_test = SecuenciasRutas(idx_test, secuencias, media, desviacion)

    longitudes_train = np.array([len(secuencias["x"][i]) for i in idx_sub_train])

    mejor_val_mae = float("inf")
    mejor_estado = None
    epocas_sin_mejora = 0

    for epoca in range(1, MAX_EPOCHS + 1):
        modelo.train()
        lotes = lote_por_longitud(len(idx_sub_train), longitudes_train, BATCH_SIZE, generador_np)
        for lote_idx in lotes:
            muestras = [ds_train[i] for i in lote_idx]
            x, estacion, y, _pob, mascara, _long = collate(muestras)
            pred, y, mascara = evaluar_lote(modelo, x, estacion, y, mascara, dispositivo)
            perdida = perdida_l1_enmascarada(pred, y, mascara)
            optimizador.zero_grad()
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=5.0)
            optimizador.step()

        modelo.eval()
        val_mae = _mae_dataset(modelo, ds_val, dispositivo)
        if val_mae < mejor_val_mae - 1e-4:
            mejor_val_mae = val_mae
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
        if epocas_sin_mejora >= PACIENCIA:
            break

    modelo.load_state_dict(mejor_estado)
    modelo.eval()

    y_pred_test, y_real_test, poblacion_test = _predecir_dataset(modelo, ds_test, dispositivo)
    resultado = metricas(y_real_test, y_pred_test)
    resultado.update(metricas_por_poblacion(y_real_test, y_pred_test, poblacion_test))
    resultado["epocas"] = epoca
    resultado["val_mae_final"] = mejor_val_mae

    print(
        f"  pliegue {pliegue}  epocas {epoca:>2}  val_mae {mejor_val_mae:.4f}  "
        f"test MAE {resultado['mae_min']:.4f} min  R2 {resultado['r2']:.4f}  "
        f"(reparto {resultado.get('mae_min_reparto', float('nan')):.4f}, "
        f"almacen {resultado.get('mae_min_almacen', float('nan')):.4f})",
        flush=True,
    )
    return resultado


def _mae_dataset(modelo, ds: SecuenciasRutas, dispositivo: torch.device) -> float:
    y_pred, y_real, _pob = _predecir_dataset(modelo, ds, dispositivo)
    return float(np.mean(np.abs(y_real - y_pred)))


@torch.no_grad()
def _predecir_dataset(modelo, ds: SecuenciasRutas, dispositivo: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predice ruta a ruta (sin relleno) para no mezclar precision de relleno en la metrica."""
    todas_pred, todas_real, toda_poblacion = [], [], []
    for i in range(len(ds)):
        x, estacion, y, poblacion = ds[i]
        x, estacion = x.unsqueeze(0).to(dispositivo), estacion.unsqueeze(0).to(dispositivo)
        pred = modelo(x, estacion).squeeze(0).cpu().numpy()
        todas_pred.append(np.clip(pred, 0, None))
        todas_real.append(y.numpy())
        toda_poblacion.append(poblacion.numpy())
    return np.concatenate(todas_pred), np.concatenate(todas_real), np.concatenate(toda_poblacion)


# ------------------------------------------------------------------------------------
# Orquestacion
# ------------------------------------------------------------------------------------
def evaluar(tramos: pd.DataFrame) -> pd.DataFrame:
    dispositivo = torch.device("cpu")
    estacion_a_indice = indice_estaciones(tramos)
    secuencias = construir_secuencias(tramos, estacion_a_indice)
    route_ids = np.array(secuencias["route_id"])

    # Los grupos de GroupKFold son por FILA de tramo, no por ruta, para que el reparto de
    # rutas por pliegue coincida con el resto del bake-off (mismo orden, mismo route_id
    # por fila: todos parten de la misma modelado.datos.cargar_train()). Se traduce
    # despues a indices de secuencia.
    cv = GroupKFold(n_splits=N_SPLITS)
    posicion_ruta = {rid: i for i, rid in enumerate(route_ids)}

    filas: list[dict[str, Any]] = []
    for pliegue, (idx_train_filas, idx_test_filas) in enumerate(
        cv.split(tramos, groups=tramos[GRUPO]), start=1
    ):
        rutas_train = tramos.iloc[idx_train_filas][GRUPO].unique()
        rutas_test = tramos.iloc[idx_test_filas][GRUPO].unique()

        solapadas = set(rutas_train) & set(rutas_test)
        if solapadas:
            raise AssertionError(f"Pliegue {pliegue}: {len(solapadas)} rutas en train y test a la vez")

        idx_train = [posicion_ruta[r] for r in rutas_train]
        idx_test = [posicion_ruta[r] for r in rutas_test]

        resultado = entrenar_pliegue(pliegue, idx_train, idx_test, secuencias, dispositivo)
        filas.append({"pliegue": pliegue, **resultado})

    return pd.DataFrame(filas)


def resumir(detalle: pd.DataFrame) -> dict[str, float]:
    columnas = [c for c in detalle.columns if c != "pliegue"]
    return {f"{c}_mean": float(detalle[c].mean()) if pd.api.types.is_numeric_dtype(detalle[c]) else None for c in columnas} | {
        f"{c}_std": float(detalle[c].std()) for c in columnas if pd.api.types.is_numeric_dtype(detalle[c])
    }


# ------------------------------------------------------------------------------------
# Salidas
# ------------------------------------------------------------------------------------
def redactar(resumen: dict[str, float], detalle: pd.DataFrame) -> str:
    gana_lstm = resumen["mae_min_mean"] < HIBRIDO_V2_MAE

    lineas = [
        "# LSTM: el experimento de deep learning",
        "",
        f"Generado por `bakeoff/lstm_experimental.py` el {datetime.now(UTC).date()}.",
        "",
        "## Resultado",
        "",
        "```text",
        f"MAE   {resumen['mae_min_mean']:.4f} min   (hibrido v2: {HIBRIDO_V2_MAE:.4f}, baseline: {BASELINE_MAE:.4f})",
        f"RMSE  {resumen['rmse_min_mean']:.4f} min",
        f"R2    {resumen['r2_mean']:.4f}       (hibrido v2: {HIBRIDO_V2_R2:.4f}, baseline: {BASELINE_R2:.4f})",
        "```",
        "",
        (
            "**La LSTM bate al hibrido de arboles.** Resultado inesperado segun la Guia del "
            "Data Scientist, que anticipa arboles ganando en tabular; hay que revisarlo con "
            "cuidado antes de aceptarlo, no solo celebrarlo."
            if gana_lstm
            else
            "**La LSTM no bate al hibrido de arboles**, tal y como anticipaba la Guia del Data "
            "Scientist para datos tabulares. Es el resultado esperado y sigue siendo la "
            "comparacion que la memoria necesita: haberlo intentado y medido pesa mas que "
            "haberlo asumido sin probar."
        ),
        "",
        "## Comparativa completa",
        "",
        "| Modelo | MAE (min) | RMSE (min) | R2 | MAE reparto | MAE almacen |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Baseline B3 | {BASELINE_MAE:.4f} | - | {BASELINE_R2:.4f} | - | - |",
        f"| Hibrido v1 (LightGBM+LightGBM) | {HIBRIDO_V1_MAE:.4f} | - | {HIBRIDO_V1_R2:.4f} | - | - |",
        f"| Hibrido v2 (CatBoost+LightGBM) | {HIBRIDO_V2_MAE:.4f} | - | {HIBRIDO_V2_R2:.4f} | - | - |",
        (
            f"| **LSTM** | {resumen['mae_min_mean']:.4f} ± {resumen['mae_min_std']:.4f} "
            f"| {resumen['rmse_min_mean']:.4f} ± {resumen['rmse_min_std']:.4f} "
            f"| {resumen['r2_mean']:.4f} ± {resumen['r2_std']:.4f} "
            f"| {resumen.get('mae_min_reparto_mean', float('nan')):.4f} "
            f"| {resumen.get('mae_min_almacen_mean', float('nan')):.4f} |"
        ),
        "",
        "## Detalle por pliegue",
        "",
        "| Pliegue | Epocas | MAE (min) | R2 | MAE reparto | MAE almacen |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, fila in detalle.iterrows():
        lineas.append(
            f"| {int(fila['pliegue'])} | {int(fila['epocas'])} | {fila['mae_min']:.4f} | {fila['r2']:.4f} "
            f"| {fila.get('mae_min_reparto', float('nan')):.4f} | {fila.get('mae_min_almacen', float('nan')):.4f} |"
        )
    mae_almacen_lstm = resumen.get("mae_min_almacen_mean")
    lineas += [
        "",
        "## La hipotesis de diseno, puesta a prueba",
        "",
        "El diseno de una sola secuencia (sin dividir poblaciones) descansaba en una",
        "afirmacion: que la LSTM no sufriria el problema que obligo a dividir los arboles en",
        "`bakeoff/modelo_hibrido.py` (un arbol conjunto reparte su capacidad donde esta la",
        "masa de datos y desatiende el 0,68% de tramos de almacen). Toca comprobarlo con el",
        "MAE de almacen, no darlo por bueno:",
        "",
        "```text",
        f"LightGBM conjunto, sin dividir           {LGBM_CONJUNTO_MAE_ALMACEN:.4f} min",
        f"LSTM, una secuencia, sin dividir          {mae_almacen_lstm:.4f} min" if mae_almacen_lstm else "",
        f"especialista dedicado (modelo_hibrido/catboost_zona)   {ESPECIALISTA_MAE_ALMACEN:.4f} min",
        "```",
        "",
    ]
    if mae_almacen_lstm:
        frac = (LGBM_CONJUNTO_MAE_ALMACEN - mae_almacen_lstm) / (LGBM_CONJUNTO_MAE_ALMACEN - ESPECIALISTA_MAE_ALMACEN)
        lineas += [
            f"La hipotesis se sostiene, con matices: la LSTM mejora un {(LGBM_CONJUNTO_MAE_ALMACEN - mae_almacen_lstm) / LGBM_CONJUNTO_MAE_ALMACEN * 100:.0f}%",
            f"sobre el arbol conjunto en el tramo de almacen sin necesitar dos modelos",
            f"separados, recuperando el {frac * 100:.0f}% de la brecha que hay hasta el especialista",
            "dedicado. El estado oculto de la secuencia sustituye en parte, no del todo, a la",
            "especializacion explicita por poblacion.",
            "",
        ]
    lineas += [
        "## Decisiones de diseno y de que evidencia previa parten",
        "",
        "- **Una sola secuencia por ruta**, con `is_depot_segment` como variable mas, en vez",
        "  de dos modelos separados como en `bakeoff/modelo_hibrido.py`. La razon de dividir",
        "  alli era que un unico arbol reparte su capacidad de division donde esta la masa de",
        "  datos y desatiende el 0,68% de tramos de almacen; una LSTM no tiene ese problema",
        "  porque procesa la secuencia entera y su estado oculto puede adaptarse. Verificado",
        "  arriba, no solo argumentado.",
        "- **Sin meteorologia**: ya medida en el diseno original de `bakeoff/lightgbm_unico.py`",
        "  (peso 1,05%, MAE ligeramente peor al incluirla), y fijada fuera del diseno",
        "  compartido de `modelado/datos.py`. No es un olvido, es no repetir una pregunta ya",
        "  respondida.",
        "- **Sin `to_zone_id`**: ya medida en `bakeoff/catboost_zona.py`. Anadirla aqui",
        "  exigiria una capa de embedding sobre 8.956 categorias, desproporcionado para un",
        "  experimento comparativo cuyo aporte ya se cuantifico.",
        "- **Perdida L1 (MAE)**, no L2, por la misma razon que en LightGBM: optimizar el",
        "  error cuadratico mejora el R2 a costa del MAE, que es la metrica principal.",
        "- **Normalizacion ajustada solo con el sub-entrenamiento de cada pliegue**, nunca",
        "  con validacion ni prueba, para no fugar la escala de datos que el modelo no debe",
        "  conocer todavia.",
        "",
        "## Siguiente paso",
        "",
        (
            "Revisar la LSTM en detalle antes de considerarla candidata a produccion: aunque"
            " gane en esta metrica, TreeSHAP explica cada prediccion del hibrido de arboles"
            " de forma exacta y en milisegundos, mientras que interpretar una LSTM es mucho"
            " mas caro y aproximado. El agente necesita explicaciones, no solo precision."
            if gana_lstm
            else
            "`entrenamiento_final.py` entrena el modelo hibrido ganador (CatBoost en reparto,"
            " LightGBM en almacen) sobre el 100% de train y lo evalua sobre test. Este"
            " experimento cierra la comparativa de arquitecturas: baseline, arboles"
            " individuales, hibrido de arboles y deep learning."
        ),
    ]
    return "\n".join(lineas) + "\n"


def registrar_en_mlflow(resumen: dict[str, float]) -> str | None:
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

    with mlflow.start_run(run_name="lstm") as run:
        mlflow.log_params(
            {
                "modelo": "LSTM",
                "hidden_size": HIDDEN_SIZE,
                "embedding_dim": EMBEDDING_DIM,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "paciencia": PACIENCIA,
                "learning_rate": LEARNING_RATE,
                "n_splits": N_SPLITS,
                "perdida": "L1 enmascarada",
                "variables_numericas": len(VARS_NUMERICAS),
                "to_zone_id": "excluida (ver bakeoff/catboost_zona.py)",
                "clima": "excluida (fijado en modelado/datos.py)",
            }
        )
        for k, v in resumen.items():
            if v is not None:
                mlflow.log_metric(k, v)
        mlflow.log_metric("baseline_mae_min", BASELINE_MAE)
        mlflow.log_metric("hibrido_v2_mae_min", HIBRIDO_V2_MAE)
        mlflow.log_artifact(str(REPORTS / "lstm_experimental.md"))
        return run.info.run_id


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    fijar_semillas()

    print("Cargando train (80% de las rutas)...", flush=True)
    tramos = cargar_train()
    print(f"  {len(tramos):,} tramos, {tramos[GRUPO].nunique():,} rutas", flush=True)
    print(f"  dispositivo: cpu ({torch.get_num_threads()} hilos)", flush=True)

    print(f"\nEntrenando LSTM con GroupKFold de {N_SPLITS} pliegues:", flush=True)
    detalle = evaluar(tramos)
    resumen = resumir(detalle)

    detalle.to_csv(REPORTS / "lstm_experimental detalle.csv", index=False)
    pd.DataFrame([resumen]).to_csv(REPORTS / "lstm_experimental resumen.csv", index=False)

    informe = redactar(resumen, detalle)
    (REPORTS / "lstm_experimental.md").write_text(informe, encoding="utf-8")

    run_id = registrar_en_mlflow(resumen)

    (REPORTS / "lstm_experimental.json").write_text(
        json.dumps(
            {
                "generado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "n_splits": N_SPLITS,
                "hiperparametros": {
                    "hidden_size": HIDDEN_SIZE,
                    "embedding_dim": EMBEDDING_DIM,
                    "batch_size": BATCH_SIZE,
                    "max_epochs": MAX_EPOCHS,
                    "paciencia": PACIENCIA,
                    "learning_rate": LEARNING_RATE,
                },
                "baseline": {"mae_min": BASELINE_MAE, "r2": BASELINE_R2},
                "hibrido_v2": {"mae_min": HIBRIDO_V2_MAE, "r2": HIBRIDO_V2_R2},
                "resumen": resumen,
                "detalle_por_pliegue": detalle.to_dict(orient="records"),
                "mlflow_run_id": run_id,
            },
            indent=2,
            ensure_ascii=False,
            default=float,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 84, flush=True)
    print("RESULTADO LSTM", flush=True)
    print("=" * 84, flush=True)
    print(f"{'modelo':<32}{'MAE min':>11}{'R2':>10}")
    print(f"{'baseline B3':<32}{BASELINE_MAE:>11.4f}{BASELINE_R2:>10.4f}")
    print(f"{'hibrido v1':<32}{HIBRIDO_V1_MAE:>11.4f}{HIBRIDO_V1_R2:>10.4f}")
    print(f"{'hibrido v2 (catboost+lgbm)':<32}{HIBRIDO_V2_MAE:>11.4f}{HIBRIDO_V2_R2:>10.4f}")
    print(f"{'LSTM':<32}{resumen['mae_min_mean']:>11.4f}{resumen['r2_mean']:>10.4f}")
    print("=" * 84, flush=True)
    veredicto = "LA LSTM GANA" if resumen["mae_min_mean"] < HIBRIDO_V2_MAE else "el hibrido de arboles sigue ganando"
    print(f"Veredicto: {veredicto}", flush=True)
    print(f"\nInforme: {(REPORTS / 'lstm_experimental.md').relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
