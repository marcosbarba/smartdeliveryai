"""Interfaz Streamlit del sistema multiagente de optimizacion de reparto.

Una sola pagina: sidebar para la intendencia (cargar una ruta real reservada como
produccion simulada, o dar de alta una ruta manual con autocompletado), y area principal
con el mapa, la tabla de la ruta y el chat contra el grafo multiagente (agente/grafo.py).

Uso:

    uv run streamlit run src/app/app.py
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# src/app/app.py -> app -> src -> smartdeliveryai
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

from app import componentes, mcp_directo  # noqa: E402
from agente.grafo import responder  # noqa: E402

st.set_page_config(page_title="SmartDeliveryAI - Agente de reparto", layout="wide")

PREGUNTAS_RAPIDAS = [
    "¿A qué hora llego a la última parada?",
    "Optimiza el orden de esta ruta",
    "¿Por qué tarda tanto el primer tramo?",
]


@st.cache_resource
def asegurar_servidor_mcp() -> str:
    if mcp_directo.servidor_disponible():
        return "Ya estaba en marcha."
    # Se invoca el entry point instalado por uv (`sdai-mcp`, ver pyproject.toml) en vez de una
    # ruta de fichero directa: evita tener que recalcular a mano la ruta al interprete del
    # venv y a mcp_server.py, y es el mismo comando documentado para arrancarlo a mano al
    # depurar.
    subprocess.Popen(
        ["uv", "run", "sdai-mcp"], cwd=str(PROJECT_ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    for _ in range(60):
        if mcp_directo.servidor_disponible():
            return "Arrancado ahora."
        time.sleep(1)
    return "No ha respondido tras 60s; revisa la consola del servidor."


def _reiniciar_conversacion():
    st.session_state["historial_chat"] = []


def _cargar_en_sesion(ruta: dict):
    st.session_state["ruta_activa"] = ruta
    st.session_state["prediccion"] = None
    _reiniciar_conversacion()


def _procesar_turno(texto: str):
    historial = st.session_state.get("historial_chat", [])
    historial = historial + [{"role": "user", "content": texto}]
    with st.spinner("Pensando..."):
        try:
            historial = asyncio.run(responder(historial))
        except Exception as error:  # noqa: BLE001 - mostrar el error en el chat, no tumbar la app
            historial = historial + [{"role": "assistant", "content": f"Error: {error}", "agente": "sistema"}]
    st.session_state["historial_chat"] = historial


st.title("SmartDeliveryAI — agente de reparto")
st.caption(
    "Sistema multiagente (LangGraph + MCP) sobre el modelo hibrido CatBoost/LightGBM del TFM. "
    "Las predicciones son tiempos típicos históricos, no una promesa exacta para un día concreto."
)

estado_servidor = asegurar_servidor_mcp()

with st.sidebar:
    st.subheader("Estado")
    st.write(f"Servidor MCP: {estado_servidor}")
    if not os.environ.get("OPENAI_API_KEY"):
        clave = st.text_input("OPENAI_API_KEY", type="password")
        if clave:
            os.environ["OPENAI_API_KEY"] = clave
    if not os.environ.get("OPENAI_MODEL"):
        modelo = st.text_input("OPENAI_MODEL", placeholder="p.ej. el modelo que tengas disponible")
        if modelo:
            os.environ["OPENAI_MODEL"] = modelo

    if mcp_directo.servidor_disponible():
        disponibles = mcp_directo.estado_modelos()
        if not disponibles.get("reparto"):
            st.warning(
                "El modelo de reparto no está entrenado todavía. Ejecuta "
                "'uv run sdai-train' antes de predecir rutas de reparto."
            )

    st.divider()
    st.subheader("Ruta de trabajo")
    modo = st.radio("Origen de la ruta", ["Ruta histórica real (producción simulada)", "Ruta nueva manual"])

    if modo == "Ruta histórica real (producción simulada)":
        try:
            estaciones = mcp_directo.listar_estaciones()
        except Exception as error:
            estaciones = []
            st.error(f"No se pudo listar estaciones: {error}")
        codigos = [e["station_code"] for e in estaciones]
        estacion = st.selectbox("Estación", ["(cualquiera)"] + codigos)
        estacion_filtro = None if estacion == "(cualquiera)" else estacion
        if st.button("Cargar ruta al azar"):
            try:
                ruta = mcp_directo.cargar_ruta_historica(None)
                _cargar_en_sesion(ruta)
            except Exception as error:
                st.error(str(error))
        try:
            rutas = mcp_directo.listar_rutas_produccion(estacion_filtro, limite=15)
        except Exception:
            rutas = []
        if rutas:
            opciones = {f"{r['route_id']} ({int(r['n_paradas'])} paradas)": r["route_id"] for r in rutas}
            elegida = st.selectbox("O elige una ruta concreta", list(opciones))
            if st.button("Cargar ruta elegida"):
                try:
                    ruta = mcp_directo.cargar_ruta_historica(opciones[elegida])
                    _cargar_en_sesion(ruta)
                except Exception as error:
                    st.error(str(error))

    else:
        try:
            estaciones = mcp_directo.listar_estaciones()
        except Exception as error:
            estaciones = []
            st.error(f"No se pudo listar estaciones: {error}")
        codigos = [e["station_code"] for e in estaciones]
        if codigos:
            estacion = st.selectbox("Estación", codigos)
            departure_hour = st.slider("Hora de salida", 0, 23, 9)
            weekday = st.selectbox(
                "Día de la semana", options=list(range(7)),
                format_func=lambda d: ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"][d],
            )
            st.caption("Añade paradas (arrastra para más filas). lat/lng son obligatorios.")
            paradas_df = st.data_editor(
                [{"lat": 0.0, "lng": 0.0, "packages_at_destination": 1, "packages_with_window_at_destination": 0}],
                num_rows="dynamic", key="editor_paradas",
            )
            if st.button("Crear ruta manual"):
                paradas = [
                    p for p in paradas_df
                    if p.get("lat") not in (None, 0.0) or p.get("lng") not in (None, 0.0)
                ]
                if len(paradas) < 1:
                    st.error("Añade al menos una parada con coordenadas reales.")
                else:
                    try:
                        ruta = mcp_directo.generar_ruta_manual(estacion, departure_hour, weekday, paradas)
                        _cargar_en_sesion(ruta)
                    except Exception as error:
                        st.error(str(error))

if "ruta_activa" in st.session_state:
    ruta = st.session_state["ruta_activa"]
    st.subheader(f"Ruta activa: {ruta['route_id']}  ·  estación {ruta['station_code']}  ·  {ruta['n_paradas']} paradas")
    if ruta["origen"] == "banco_produccion_simulada":
        st.caption("Ruta real reservada como producción simulada: el modelo nunca la vio al entrenar.")
    else:
        st.caption("Ruta manual: los campos marcados como estimados se muestrearon del histórico, no los diste tú.")

    if st.session_state.get("prediccion") is None:
        try:
            st.session_state["prediccion"] = mcp_directo.predecir_ruta_activa()
        except Exception as error:
            st.error(f"No se pudo predecir la ruta: {error}")

    prediccion = st.session_state.get("prediccion")
    col_mapa, col_tabla = st.columns([3, 2])
    with col_mapa:
        st.pydeck_chart(componentes.mapa_ruta(ruta["tramos"]))
        if prediccion:
            st.metric("Tiempo total predicho", f"{prediccion['tiempo_total_segundos'] / 60:.1f} min")
            if ruta.get("tiempos_reales_ocultos_segundos"):
                real = sum(ruta["tiempos_reales_ocultos_segundos"]) / 60
                st.metric("Tiempo real (histórico, oculto al modelo)", f"{real:.1f} min")
    with col_tabla:
        st.dataframe(componentes.tabla_ruta(ruta["tramos"], prediccion), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Habla con el agente")
    cols = st.columns(len(PREGUNTAS_RAPIDAS))
    pregunta_rapida = None
    for c, pregunta in zip(cols, PREGUNTAS_RAPIDAS):
        if c.button(pregunta):
            pregunta_rapida = pregunta

    for mensaje in st.session_state.get("historial_chat", []):
        with st.chat_message(mensaje["role"]):
            etiqueta = f"*({mensaje['agente']})*  \n" if mensaje.get("agente") and mensaje["agente"] != "supervisor" else ""
            st.markdown(etiqueta + mensaje["content"])
            optimizacion = mensaje.get("optimizacion")
            if optimizacion:
                st.metric(
                    "Ahorro estimado con el nuevo orden",
                    f"{optimizacion['ahorro_segundos'] / 60:.1f} min ({optimizacion['ahorro_pct']}%)",
                )
                col_mapa_opt, col_tabla_opt = st.columns([3, 2])
                with col_mapa_opt:
                    st.pydeck_chart(componentes.mapa_ruta(ruta["tramos"], orden=optimizacion["orden_propuesto"]))
                with col_tabla_opt:
                    st.dataframe(
                        componentes.tabla_ruta_optimizada(
                            ruta["tramos"], optimizacion["orden_propuesto"], optimizacion["tiempos_por_tramo_propuesto_segundos"]
                        ),
                        use_container_width=True, hide_index=True,
                    )

    texto = st.chat_input("Pregunta al agente sobre esta ruta...")
    entrada = pregunta_rapida or texto
    if entrada:
        # Pintar la pregunta del usuario YA, antes de llamar al agente: si se guardara en
        # session_state solo dentro de _procesar_turno, no se veria hasta despues del
        # spinner, porque el bucle que dibuja el historial ya se ha ejecutado mas arriba
        # en esta misma pasada del script.
        with st.chat_message("user"):
            st.markdown(entrada)
        _procesar_turno(entrada)
        st.rerun()
else:
    st.info("Carga una ruta desde el panel de la izquierda para empezar.")
