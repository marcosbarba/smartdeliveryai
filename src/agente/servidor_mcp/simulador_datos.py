"""La solucion a "no tenemos datos de produccion": dos formas de generar un caso nuevo.

1. Replay de una ruta real reservada (`data/produccion_simulada/`): el 20% de rutas
   (por route_id, split fijo) que src/modelado/entrenamiento_final.py aparto ANTES del
   bake-off de arquitecturas. El modelo nunca las vio. Se sirven como si acabaran de llegar:
   el tiempo real (`travel_time_seconds`) se guarda aparte y solo se revela al final, para
   comparar contra la prediccion.

2. Alta manual con autocompletado: el usuario da lo que un repartidor conoceria de verdad
   (paradas, paquetes, coordenadas); lo que no podria conocer (tiempo de servicio, volumen,
   ventanas horarias, zona de planificacion) se muestrea de la distribucion historica de esa
   estacion. Esas distribuciones (artifacts/modelo/distribuciones_estacion.json) se
   calcularon SOLO con las rutas de train, nunca con el banco de produccion reservado (test).

Cada fila que sale de aqui declara que campos son reales y cuales son estimados
(`campos_estimados`), para que el agente pueda comunicarlo con honestidad.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# src/agente/servidor_mcp/simulador_datos.py -> servidor_mcp -> agente -> src -> smartdeliveryai
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "artifacts" / "modelo"
DATOS_PRODUCCION = PROJECT_ROOT / "data" / "produccion_simulada" / "datos_produccion.parquet"

SIN_ZONA = "SIN_ZONA"


class SinDatosProduccion(RuntimeError):
    """No existe el banco de produccion simulada: hay que ejecutar antes 'uv run sdai-train'."""


@lru_cache(maxsize=1)
def _distribuciones() -> dict:
    ruta = MODELS_DIR / "distribuciones_estacion.json"
    if not ruta.exists():
        raise SinDatosProduccion(
            f"No existe {ruta}. Ejecuta antes 'uv run sdai-train'."
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _banco_produccion() -> pd.DataFrame:
    if not DATOS_PRODUCCION.exists():
        raise SinDatosProduccion(
            f"No existe {DATOS_PRODUCCION}. Ejecuta antes 'uv run sdai-train'."
        )
    return pd.read_parquet(DATOS_PRODUCCION)


def listar_estaciones() -> list[dict]:
    """Estaciones disponibles, con cuantos tramos historicos de entrenamiento tiene cada una
    (para que la interfaz sepa cuales tienen distribuciones fiables)."""
    dist = _distribuciones()
    return [
        {"station_code": codigo, "n_tramos_historicos": info["n_tramos"]}
        for codigo, info in sorted(dist.items(), key=lambda kv: -kv[1]["n_tramos"])
    ]


def listar_rutas_produccion(station_code: str | None = None, limite: int = 20) -> list[dict]:
    """Rutas disponibles en el banco de produccion simulada, opcionalmente filtradas por
    estacion. Son rutas reales, nunca usadas para entrenar el modelo."""
    banco = _banco_produccion()
    if station_code:
        banco = banco[banco["station_code"] == station_code]
    resumen = (
        banco.groupby("route_id")
        .agg(
            station_code=("station_code", "first"),
            n_paradas=("segment_position", "max"),
            departure_hour=("departure_hour", "first"),
            weekday=("weekday", "first"),
            route_date=("route_date", "first"),
        )
        .reset_index()
        .head(limite)
    )
    return resumen.to_dict(orient="records")


def cargar_ruta_historica(route_id: str | None = None) -> dict:
    """Trae una ruta completa del banco de produccion simulada, ordenada por segment_position.
    Si no se da route_id, elige una al azar. El tiempo real de cada tramo se devuelve aparte
    (`tiempos_reales_ocultos`), nunca dentro de las variables de entrada del modelo.
    """
    banco = _banco_produccion()
    if route_id is None:
        route_id = str(np.random.default_rng().choice(banco["route_id"].unique()))
    ruta = banco[banco["route_id"] == route_id].sort_values("segment_position")
    if ruta.empty:
        raise ValueError(f"route_id '{route_id}' no esta en el banco de produccion simulada.")

    columnas_entrada = [
        "from_stop_id", "to_stop_id", "from_lat", "from_lng", "to_lat", "to_lng",
        "segment_distance_km", "is_depot_segment", "same_zone", "to_zone_id",
        "segment_position", "cumulative_distance_km", "packages_at_destination",
        "service_time_at_destination_seconds", "volume_at_destination_cm3",
        "packages_with_window_at_destination", "departure_hour", "route_total_stops",
        "route_package_count", "weekday", "station_code",
    ]
    tramos = ruta[columnas_entrada].to_dict(orient="records")
    for tramo in tramos:
        tramo["campos_estimados"] = []  # todo es real: viene del manifiesto de la ruta

    return {
        "route_id": route_id,
        "origen": "banco_produccion_simulada",
        "station_code": ruta["station_code"].iloc[0],
        "route_date": str(ruta["route_date"].iloc[0]),
        "n_paradas": int(ruta["segment_position"].max()),
        "tramos": tramos,
        "tiempos_reales_ocultos_segundos": ruta["travel_time_seconds"].tolist(),
    }


def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _muestrear(station_code: str, campo: str, generador: np.random.Generator) -> float | None:
    """Un valor al azar entre los percentiles historicos guardados para ese campo y esa
    estacion (no la media fija: asi el alta manual tiene variacion realista)."""
    entrada = _distribuciones().get(station_code, {}).get(campo)
    if not entrada:
        return None
    return float(generador.choice(list(entrada.values())))


def generar_ruta_manual(
    station_code: str, departure_hour: int, weekday: int, paradas: list[dict], semilla: int | None = None,
) -> dict:
    """paradas: lista de dicts con, como minimo, lat/lng/packages_at_destination. Los campos
    que falten se muestrean de la distribucion historica de station_code (calculada solo con
    datos de entrenamiento). El primer tramo (almacen -> primera parada) necesita coordenadas
    de almacen: si no estan disponibles para la estacion, se usa la primera parada como
    origen y se avisa en 'campos_estimados'.
    """
    if station_code not in _distribuciones():
        raise ValueError(f"station_code '{station_code}' no tiene distribuciones historicas.")
    if len(paradas) < 1:
        raise ValueError("Hace falta al menos una parada.")

    generador = np.random.default_rng(semilla)
    info_estacion = _distribuciones()[station_code]
    zonas_vistas = info_estacion.get("to_zone_id_vistas") or [SIN_ZONA]

    depot_lat = info_estacion.get("depot_lat")
    depot_lng = info_estacion.get("depot_lng")
    origen_estimado = depot_lat is None or depot_lng is None
    if origen_estimado:
        depot_lat, depot_lng = paradas[0]["lat"], paradas[0]["lng"]

    coords = [(depot_lat, depot_lng)] + [(p["lat"], p["lng"]) for p in paradas]
    route_package_count = sum(int(p.get("packages_at_destination", 1)) for p in paradas)
    # +1: el pipeline Gold cuenta el propio almacen como una parada mas (route_total_stops =
    # dropoff_stops + station_stops), no solo las entregas. Verificado contra el banco de
    # produccion: route_total_stops == n_tramos + 1 en las 611 rutas.
    route_total_stops = len(paradas) + 1

    tramos = []
    acumulado_km = 0.0
    zona_anterior_resuelta = None  # la de la parada anterior YA resuelta (dada o estimada), no el dict crudo de entrada
    for i, parada in enumerate(paradas):
        distancia = _haversine_km(*coords[i], *coords[i + 1])
        campos_estimados = []

        def _valor(campo, por_defecto=None):
            if campo in parada and parada[campo] is not None:
                return parada[campo]
            campos_estimados.append(campo)
            muestra = _muestrear(station_code, campo, generador)
            return muestra if muestra is not None else por_defecto

        to_zone_id = parada.get("to_zone_id")
        if not to_zone_id:
            to_zone_id = str(generador.choice(zonas_vistas))
            campos_estimados.append("to_zone_id")
        same_zone = 1 if (zona_anterior_resuelta and zona_anterior_resuelta == to_zone_id) else 0
        zona_anterior_resuelta = to_zone_id

        tramo = {
            "from_stop_id": "ALMACEN" if i == 0 else f"parada_{i}",
            "to_stop_id": f"parada_{i + 1}",
            "from_lat": coords[i][0], "from_lng": coords[i][1],
            "to_lat": coords[i + 1][0], "to_lng": coords[i + 1][1],
            "segment_distance_km": distancia,
            "is_depot_segment": 1 if i == 0 else 0,
            "same_zone": same_zone,
            "to_zone_id": to_zone_id,
            "segment_position": i + 1,
            # Km acumulados HASTA EL INICIO de este tramo, sin contar su propia distancia
            # (asi lo define el pipeline Gold: ventana hasta la fila anterior). Por eso se lee
            # "acumulado_km" antes de sumarle "distancia" mas abajo.
            "cumulative_distance_km": acumulado_km,
            "packages_at_destination": int(parada.get("packages_at_destination", 1)),
            "service_time_at_destination_seconds": _valor("service_time_at_destination_seconds"),
            "volume_at_destination_cm3": _valor("volume_at_destination_cm3"),
            "packages_with_window_at_destination": _valor("packages_with_window_at_destination", 0),
            "departure_hour": departure_hour,
            "route_total_stops": route_total_stops,
            "route_package_count": route_package_count,
            "weekday": weekday,
            "station_code": station_code,
            "campos_estimados": campos_estimados + (["from_lat/from_lng (origen almacen)"] if origen_estimado and i == 0 else []),
        }
        tramos.append(tramo)
        acumulado_km += distancia

    return {
        "route_id": "manual_" + "_".join(str(round(p["lat"], 3)) for p in paradas[:2]),
        "origen": "alta_manual",
        "station_code": station_code,
        "n_paradas": route_total_stops,
        "tramos": tramos,
        "tiempos_reales_ocultos_segundos": None,
    }
