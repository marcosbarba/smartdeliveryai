"""Helpers de render para la interfaz Streamlit: mapa, tabla de ruta y grafico SHAP."""

from __future__ import annotations

import pandas as pd
import pydeck as pdk


def mapa_ruta(tramos: list[dict], orden: list[int] | None = None) -> pdk.Deck:
    """Mapa con el almacen, las paradas (en el orden dado, o el original si orden es None) y
    la linea que las une."""
    if not tramos:
        return pdk.Deck(initial_view_state=pdk.ViewState(latitude=0, longitude=0, zoom=1))

    depot = {"lat": tramos[0]["from_lat"], "lng": tramos[0]["from_lng"]}
    paradas = [{"lat": t["to_lat"], "lng": t["to_lng"], "n": t["segment_position"]} for t in tramos]
    if orden is not None:
        paradas = [paradas[i] for i in orden]

    puntos = [depot] + paradas
    ruta_df = pd.DataFrame(puntos)
    paradas_df = pd.DataFrame(paradas)
    depot_df = pd.DataFrame([depot])

    camino = {"path": [[p["lng"], p["lat"]] for p in puntos]}

    capa_camino = pdk.Layer(
        "PathLayer", data=[camino], get_path="path", get_width=25, get_color=[80, 120, 220], width_min_pixels=2,
    )
    capa_paradas = pdk.Layer(
        "ScatterplotLayer", data=paradas_df, get_position="[lng, lat]", get_radius=40,
        get_fill_color=[220, 90, 60], pickable=True,
    )
    capa_almacen = pdk.Layer(
        "ScatterplotLayer", data=depot_df, get_position="[lng, lat]", get_radius=70,
        get_fill_color=[30, 160, 90], pickable=True,
    )

    vista = pdk.ViewState(latitude=float(ruta_df["lat"].mean()), longitude=float(ruta_df["lng"].mean()), zoom=11)
    return pdk.Deck(
        layers=[capa_camino, capa_paradas, capa_almacen], initial_view_state=vista,
        tooltip={"text": "parada {n}"}, map_style=None,
    )


def tabla_ruta(tramos: list[dict], prediccion: dict | None = None) -> pd.DataFrame:
    filas = []
    for i, t in enumerate(tramos):
        fila = {
            "Tramo": t["segment_position"],
            "Destino": t["to_stop_id"],
            "Distancia (km)": round(t["segment_distance_km"], 2),
            "Paquetes": t.get("packages_at_destination"),
            "Dato estimado": ", ".join(t.get("campos_estimados") or []) or "—",
        }
        if prediccion is not None:
            fila["Tiempo (min)"] = round(prediccion["tiempos_por_tramo_segundos"][i] / 60, 1)
            fila["Llegada acumulada (min)"] = round(prediccion["llegada_acumulada_segundos"][i] / 60, 1)
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_ruta_optimizada(tramos: list[dict], orden_propuesto: list[int], tiempos_por_tramo_segundos: list[float]) -> pd.DataFrame:
    """Igual que tabla_ruta, pero para el orden PROPUESTO: segment_position y
    segment_distance_km de cada tramo son del orden original (dependen de la posicion), asi
    que no se pueden reindexar tal cual. Solo se toman de cada tramo los campos que no
    dependen del orden (destino, paquetes, si fueron estimados); el tiempo por tramo viene ya
    recalculado para el nuevo orden por optimizar_ruta_activa."""
    filas = []
    acumulado = 0.0
    for nueva_posicion, idx in enumerate(orden_propuesto, start=1):
        t = tramos[idx]
        tiempo = tiempos_por_tramo_segundos[nueva_posicion - 1]
        acumulado += tiempo
        filas.append(
            {
                "Orden propuesto": nueva_posicion,
                "Destino": t["to_stop_id"],
                "Paquetes": t.get("packages_at_destination"),
                "Dato estimado": ", ".join(t.get("campos_estimados") or []) or "—",
                "Tiempo (min)": round(tiempo / 60, 1),
                "Llegada acumulada (min)": round(acumulado / 60, 1),
            }
        )
    return pd.DataFrame(filas)


def grafico_shap(explicacion: dict) -> pd.DataFrame:
    """DataFrame listo para st.bar_chart: contribucion en minutos por variable, mayor
    magnitud primero."""
    df = pd.DataFrame(explicacion["contribuciones"])
    df["contribucion_min"] = df["contribucion_segundos"] / 60
    df["etiqueta"] = df["variable"] + " = " + df["valor_entrada"].astype(str)
    return df.set_index("etiqueta")[["contribucion_min"]]
