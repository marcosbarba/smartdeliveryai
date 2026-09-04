"""Servidor MCP: el proveedor de herramientas del sistema multiagente.

Se ejecuta como proceso propio, aparte de Streamlit y del grafo de LangGraph, para poder
cargar el modelo de reparto (varios cientos de MB) una sola vez y mantenerlo en memoria
mientras dure la sesion de trabajo, en vez de recargarlo en cada turno de chat.

Guarda la "ruta activa" (la que el repartidor esta consultando ahora) como estado del
proceso: cargar_ruta_historica y generar_ruta_manual la fijan como efecto secundario, y el
resto de herramientas (predecir_ruta_activa, explicar_tramo_activo, optimizar_ruta_activa)
operan sobre ella sin que el LLM tenga que repetir el JSON de la ruta en cada llamada. Para
una demo de un solo usuario esto basta; en un despliegue multiusuario real esta variable
pasaria a ser un almacen por sesion.

Uso:

    uv run sdai-mcp

Transporte streamable-http en localhost, puerto configurable por MCP_SERVER_PORT (por
defecto 8765). agente/cliente_mcp.py y app/app.py se conectan a ese puerto.
"""

from __future__ import annotations

import os

import pandas as pd
from mcp.server.fastmcp import FastMCP

from . import explicabilidad, optimizador, simulador_datos
from .modelos import ModeloNoDisponible, modelos_disponibles, predecir_mixto

PUERTO = int(os.environ.get("MCP_SERVER_PORT", "8765"))

mcp = FastMCP("smartdeliveryai-reparto", host="127.0.0.1", port=PUERTO)

_RUTA_ACTIVA: dict | None = None


def _predecir_tramos(tramos: list[dict]) -> dict:
    df = pd.DataFrame(tramos)
    tiempos = predecir_mixto(df)
    acumulado = tiempos.cumsum()
    return {
        "tiempos_por_tramo_segundos": [round(t, 1) for t in tiempos.tolist()],
        "llegada_acumulada_segundos": [round(t, 1) for t in acumulado.tolist()],
        "tiempo_total_segundos": round(float(tiempos.sum()), 1),
        "to_stop_id": df["to_stop_id"].tolist() if "to_stop_id" in df.columns else None,
    }


def _paradas_desde_ruta_activa() -> tuple[list[dict], float, float]:
    tramos = _RUTA_ACTIVA["tramos"]
    depot_lat, depot_lng = tramos[0]["from_lat"], tramos[0]["from_lng"]
    paradas = [
        {
            "lat": t["to_lat"], "lng": t["to_lng"],
            "packages_at_destination": t.get("packages_at_destination", 1),
            "to_zone_id": t.get("to_zone_id"),
            "service_time_at_destination_seconds": t.get("service_time_at_destination_seconds"),
            "volume_at_destination_cm3": t.get("volume_at_destination_cm3"),
            "packages_with_window_at_destination": t.get("packages_with_window_at_destination"),
        }
        for t in tramos
    ]
    return paradas, depot_lat, depot_lng


def _requiere_ruta_activa() -> dict:
    if _RUTA_ACTIVA is None:
        raise ValueError(
            "No hay ninguna ruta cargada todavia. Usa primero cargar_ruta_historica o "
            "generar_ruta_manual."
        )
    return _RUTA_ACTIVA


@mcp.tool()
def listar_estaciones() -> list[dict]:
    """List the delivery stations available in the historical data, with how many
    historical segments each one has. Use this to help the user pick a station before
    loading or creating a route."""
    return simulador_datos.listar_estaciones()


@mcp.tool()
def listar_rutas_produccion(station_code: str | None = None, limite: int = 20) -> list[dict]:
    """List real routes available in the simulated-production bank (routes the model has
    never seen during training), optionally filtered by station. Use this to help the user
    pick a route_id for cargar_ruta_historica."""
    return simulador_datos.listar_rutas_produccion(station_code, limite)


@mcp.tool()
def cargar_ruta_historica(route_id: str | None = None) -> dict:
    """Load a real historical route from the simulated-production bank as if it had just
    come in, and set it as the active route. The model has never seen this route during
    training. If route_id is omitted, a random route is picked. The real travel times are
    returned separately (tiempos_reales_ocultos_segundos) only for comparison after
    predicting — never use them as model input or reveal them before predicting."""
    global _RUTA_ACTIVA
    _RUTA_ACTIVA = simulador_datos.cargar_ruta_historica(route_id)
    return _RUTA_ACTIVA


@mcp.tool()
def generar_ruta_manual(station_code: str, departure_hour: int, weekday: int, paradas: list[dict]) -> dict:
    """Create a brand-new route from stops a courier would actually know (lat, lng,
    packages_at_destination, and optionally to_zone_id), and set it as the active route.
    Fields a courier could not know in advance (service time, package volume, time-window
    packages, planning zone) are sampled from that station's historical distribution and
    marked in campos_estimados on each segment. weekday is 0=Monday..6=Sunday."""
    global _RUTA_ACTIVA
    _RUTA_ACTIVA = simulador_datos.generar_ruta_manual(station_code, departure_hour, weekday, paradas)
    return _RUTA_ACTIVA


@mcp.tool()
def estado_ruta_activa() -> dict:
    """Summarize the currently active route (station, number of stops, where it came
    from), without the full segment detail. Use this to check what is loaded before
    predicting, explaining or optimizing."""
    if _RUTA_ACTIVA is None:
        return {"hay_ruta_activa": False}
    return {
        "hay_ruta_activa": True,
        "route_id": _RUTA_ACTIVA["route_id"],
        "origen": _RUTA_ACTIVA["origen"],
        "station_code": _RUTA_ACTIVA["station_code"],
        "n_paradas": _RUTA_ACTIVA["n_paradas"],
    }


@mcp.tool()
def predecir_ruta_activa() -> dict:
    """Predict the travel time of every segment of the active route and the cumulative
    arrival time at each stop, in the route's current order. Predictions are a typical
    historical duration, not an exact time for one specific day; the reparto model has a
    known tendency to underpredict on average (see docs/modelado/instrucciones_de_servicio.md
    for the current measured bias) — mention this if promising a schedule, and give a small
    range rather than a single exact number."""
    ruta = _requiere_ruta_activa()
    return _predecir_tramos(ruta["tramos"])


@mcp.tool()
def explicar_tramo_activo(segment_position: int) -> dict:
    """Explain, with exact TreeSHAP contributions in seconds, why the segment at
    segment_position (1-based, 1 = the depot-to-first-stop segment) of the active route
    takes as long as predicted. Never attribute the result to rain/weather: the model does
    not use weather data. segment_distance_km is a straight line, not a road distance."""
    ruta = _requiere_ruta_activa()
    coincidencias = [t for t in ruta["tramos"] if t["segment_position"] == segment_position]
    if not coincidencias:
        raise ValueError(f"La ruta activa no tiene ningun tramo en segment_position={segment_position}.")
    return explicabilidad.explicar_tramo(coincidencias[0])


@mcp.tool()
def optimizar_ruta_activa(refinar: bool = True) -> dict:
    """Reorder the active route's stops to minimize the total predicted delivery time.
    Returns the original order, the proposed order (as indices into the route's current
    stop list, 0-based, depot excluded), and the time saved. Does not modify the active
    route — the caller decides whether to apply the suggestion. Uses straight-line
    coordinates only, not real road distances."""
    ruta = _requiere_ruta_activa()
    paradas, depot_lat, depot_lng = _paradas_desde_ruta_activa()
    tramos = ruta["tramos"]
    return optimizador.optimizar_ruta(
        paradas, ruta["station_code"], tramos[0]["departure_hour"], tramos[0]["weekday"],
        depot_lat=depot_lat, depot_lng=depot_lng, refinar=refinar,
    )


@mcp.tool()
def estado_modelos() -> dict:
    """Report whether the delivery-segment (reparto) and depot-segment (almacen) models are
    loaded. If reparto is false, most predictions will fail until the agent's model is
    (re)trained."""
    try:
        return modelos_disponibles()
    except ModeloNoDisponible as error:
        return {"reparto": False, "almacen": False, "error": str(error)}


def main() -> None:
    print(f"Servidor MCP arrancando en http://127.0.0.1:{PUERTO} ...")
    print("Estado de los modelos:", estado_modelos())
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
