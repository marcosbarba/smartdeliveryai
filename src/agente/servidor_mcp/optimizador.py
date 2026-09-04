"""Reordena una ruta para minimizar el tiempo total predicho.

Por que no una matriz de distancias NxN
------------------------------------------
El modelo no trata cada tramo A->B como un coste fijo: `segment_position` (que numero de
tramo es dentro de la ruta) y `cumulative_distance_km` (cuanto se ha recorrido ya) forman
parte de sus variables de entrada. El mismo salto A->B predice un tiempo distinto si ocurre
el tramo 2 o el tramo 15 de la ruta. Una matriz de costes precomputada asumiria que el coste
de un salto no depende de donde ocurre, lo cual no es cierto para este modelo.

En su lugar: construccion voraz paso a paso, evaluando en cada paso el tiempo que predeciria
el modelo para CADA parada restante EN LA POSICION REAL que ocuparia, y afinado con un 2-opt
acotado que recalcula con el modelo el tramo de sufijo afectado por cada intercambio
candidato, no con distancia bruta.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from .modelos import predecir_mixto

MAX_SEGUNDOS_2OPT = 8.0


def _haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlambda = np.radians(np.asarray(lng2) - np.asarray(lng1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _construir_filas(orden: list[int], paradas: list[dict], contexto: dict) -> pd.DataFrame:
    """Reconstruye, para un orden de visita dado, las variables dependientes de la posicion
    (segment_position, cumulative_distance_km, same_zone) tramo a tramo."""
    lat_prev, lng_prev = contexto["depot_lat"], contexto["depot_lng"]
    zona_prev = None
    acumulado = 0.0
    filas = []
    for posicion, idx in enumerate(orden, start=1):
        parada = paradas[idx]
        distancia = _haversine_km(lat_prev, lng_prev, parada["lat"], parada["lng"])
        filas.append(
            {
                **contexto["comunes"],
                # La posicion 1 es siempre el tramo almacen -> primera parada: ese modelo
                # (LightGBM) nunca vio tramos de reparto en su entrenamiento, y el de
                # reparto (CatBoost) nunca vio el tramo de almacen. predecir_mixto elige el
                # especialista correcto a partir de este campo.
                "is_depot_segment": 1 if posicion == 1 else 0,
                "segment_distance_km": float(distancia),
                "same_zone": 1 if (zona_prev and zona_prev == parada.get("to_zone_id")) else 0,
                "to_zone_id": parada.get("to_zone_id", "SIN_ZONA"),
                "segment_position": posicion,
                # Km acumulados HASTA EL INICIO de este tramo (sin contar su propia
                # distancia): asi lo define el pipeline Gold (ventana hasta la fila anterior,
                # ver Construir Gold Tramos.py). Por eso "acumulado" se actualiza DESPUES de
                # guardar la fila, no antes.
                "cumulative_distance_km": acumulado,
                "packages_at_destination": parada.get("packages_at_destination", 1),
                "service_time_at_destination_seconds": parada.get("service_time_at_destination_seconds", 0),
                "volume_at_destination_cm3": parada.get("volume_at_destination_cm3", 0),
                "packages_with_window_at_destination": parada.get("packages_with_window_at_destination", 0),
            }
        )
        acumulado += float(distancia)
        lat_prev, lng_prev = parada["lat"], parada["lng"]
        zona_prev = parada.get("to_zone_id")
    return pd.DataFrame(filas)


def _tiempo_total(orden: list[int], paradas: list[dict], contexto: dict) -> tuple[float, np.ndarray]:
    filas = _construir_filas(orden, paradas, contexto)
    tiempos = predecir_mixto(filas)
    return float(tiempos.sum()), tiempos


def _fila_candidata(lat_prev, lng_prev, zona_prev, acumulado_km, posicion, parada, contexto) -> dict:
    """Una unica fila (sin reconstruir el prefijo entero): la usan tanto la construccion
    voraz como cualquier evaluacion de 'anadir esta parada aqui' con el estado acumulado
    hasta el momento."""
    distancia = float(_haversine_km(lat_prev, lng_prev, parada["lat"], parada["lng"]))
    return {
        **contexto["comunes"],
        "is_depot_segment": 1 if posicion == 1 else 0,
        "segment_distance_km": distancia,
        "same_zone": 1 if (zona_prev and zona_prev == parada.get("to_zone_id")) else 0,
        "to_zone_id": parada.get("to_zone_id", "SIN_ZONA"),
        "segment_position": posicion,
        # Igual que en _construir_filas: km acumulados ANTES de este tramo, no incluyendolo.
        "cumulative_distance_km": acumulado_km,
        "packages_at_destination": parada.get("packages_at_destination", 1),
        "service_time_at_destination_seconds": parada.get("service_time_at_destination_seconds", 0),
        "volume_at_destination_cm3": parada.get("volume_at_destination_cm3", 0),
        "packages_with_window_at_destination": parada.get("packages_with_window_at_destination", 0),
    }, distancia


def _construccion_voraz(paradas: list[dict], contexto: dict) -> list[int]:
    """O(n^2): en cada paso, un unico predict vectorizado sobre todas las paradas restantes,
    construidas a partir del estado acumulado (sin reconstruir el prefijo completo cada vez)."""
    restantes = list(range(len(paradas)))
    orden: list[int] = []
    lat_prev, lng_prev = contexto["depot_lat"], contexto["depot_lng"]
    zona_prev = None
    acumulado_km = 0.0
    posicion = 1
    while restantes:
        filas_candidatas, distancias = [], []
        for idx in restantes:
            fila, distancia = _fila_candidata(lat_prev, lng_prev, zona_prev, acumulado_km, posicion, paradas[idx], contexto)
            filas_candidatas.append(fila)
            distancias.append(distancia)
        tiempos = predecir_mixto(pd.DataFrame(filas_candidatas))
        mejor = int(np.argmin(tiempos))
        elegido = restantes.pop(mejor)
        orden.append(elegido)

        acumulado_km += distancias[mejor]
        lat_prev, lng_prev = paradas[elegido]["lat"], paradas[elegido]["lng"]
        zona_prev = paradas[elegido].get("to_zone_id")
        posicion += 1
    return orden


def _2opt_acotado(orden: list[int], paradas: list[dict], contexto: dict, limite_segundos: float) -> list[int]:
    mejor_orden = orden[:]
    mejor_tiempo, _ = _tiempo_total(mejor_orden, paradas, contexto)
    inicio = time.monotonic()
    n = len(orden)
    mejorado = True
    while mejorado and time.monotonic() - inicio < limite_segundos:
        mejorado = False
        for i in range(n - 1):
            if time.monotonic() - inicio > limite_segundos:
                break
            for j in range(i + 1, n):
                if time.monotonic() - inicio > limite_segundos:
                    break
                candidato = mejor_orden[:i] + mejor_orden[i : j + 1][::-1] + mejor_orden[j + 1 :]
                tiempo_candidato, _ = _tiempo_total(candidato, paradas, contexto)
                if tiempo_candidato < mejor_tiempo:
                    mejor_orden, mejor_tiempo = candidato, tiempo_candidato
                    mejorado = True
    return mejor_orden


def optimizar_ruta(
    paradas: list[dict], station_code: str, departure_hour: int, weekday: int,
    route_package_count: int | None = None, depot_lat: float | None = None, depot_lng: float | None = None,
    refinar: bool = True,
) -> dict:
    """paradas: lista de dicts con lat/lng y, si se conocen, to_zone_id/packages_at_destination/etc.
    Devuelve el orden original y el propuesto, con el tiempo total predicho de cada uno.
    """
    if len(paradas) < 2:
        raise ValueError("Hacen falta al menos 2 paradas para poder optimizar el orden.")

    if depot_lat is None or depot_lng is None:
        depot_lat, depot_lng = paradas[0]["lat"], paradas[0]["lng"]

    contexto = {
        "depot_lat": depot_lat,
        "depot_lng": depot_lng,
        "comunes": {
            "departure_hour": departure_hour,
            # +1: el pipeline Gold cuenta el propio almacen como una parada mas
            # (route_total_stops = dropoff_stops + station_stops), no solo las entregas.
            # Verificado contra el banco de produccion: route_total_stops == n_tramos + 1
            # en las 611 rutas.
            "route_total_stops": len(paradas) + 1,
            "route_package_count": route_package_count or sum(p.get("packages_at_destination", 1) for p in paradas),
            "weekday": weekday,
            "station_code": station_code,
            # is_depot_segment NO va aqui: depende de la posicion (1 en el primer tramo,
            # 0 en el resto) y se fija por fila en _construir_filas/_fila_candidata.
        },
    }

    orden_original = list(range(len(paradas)))
    tiempo_original, tiempos_original_por_tramo = _tiempo_total(orden_original, paradas, contexto)

    orden_propuesto = _construccion_voraz(paradas, contexto)
    if refinar:
        orden_propuesto = _2opt_acotado(orden_propuesto, paradas, contexto, MAX_SEGUNDOS_2OPT)
    tiempo_propuesto, tiempos_propuesto_por_tramo = _tiempo_total(orden_propuesto, paradas, contexto)

    ahorro_segundos = tiempo_original - tiempo_propuesto
    return {
        "orden_original": orden_original,
        "tiempo_total_original_segundos": round(tiempo_original, 1),
        "orden_propuesto": orden_propuesto,
        "tiempo_total_propuesto_segundos": round(tiempo_propuesto, 1),
        "ahorro_segundos": round(ahorro_segundos, 1),
        "ahorro_pct": round(100 * ahorro_segundos / tiempo_original, 1) if tiempo_original > 0 else 0.0,
        "tiempos_por_tramo_propuesto_segundos": [round(t, 1) for t in tiempos_propuesto_por_tramo.tolist()],
    }
