"""Interfaz Streamlit del sistema multiagente de optimizacion de reparto.

Una sola pagina: sidebar para la intendencia (cargar una ruta real reservada como
produccion simulada, o dar de alta una ruta manual con autocompletado), y area principal
con el mapa, la tabla de la ruta y el chat contra el grafo multiagente (agente/grafo.py).

Uso:

    uv run streamlit run src/app/app.py
"""

from __future__ import annotations

import asyncio
import atexit
import os
import signal
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

ETIQUETAS_AGENTE = {
    "route_optimization_agent": "optimización de rutas",
    "eta_explanation_agent": "explicación de tiempos",
}


def _detener_arbol_proceso(pid: int) -> None:
    """Mata pid y toda su descendencia. Hace falta recorrer el arbol (y no solo matar pid)
    porque 'uv run sdai-mcp' no es el proceso final: uv lanza su propio hijo, que a su vez
    lanza el interprete del venv, y terminate() sobre el Popen de uv no se propaga a esos
    nietos ni en Windows ni en POSIX."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


@st.cache_resource
def asegurar_servidor_mcp() -> tuple[bool, str]:
    if mcp_directo.servidor_disponible():
        return True, "Conectado y listo."
    # Se invoca el entry point instalado por uv (`sdai-mcp`, ver pyproject.toml) en vez de una
    # ruta de fichero directa: evita tener que recalcular a mano la ruta al interprete del
    # venv y a mcp_server.py, y es el mismo comando documentado para arrancarlo a mano al
    # depurar. start_new_session/CREATE_NEW_PROCESS_GROUP lo deja como lider de su propio
    # grupo, requisito para poder matar el arbol entero en _detener_arbol_proceso.
    proceso = subprocess.Popen(
        ["uv", "run", "sdai-mcp"], cwd=str(PROJECT_ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        start_new_session=(os.name != "nt"),
    )
    # Solo se registra el cierre automatico si el servidor lo hemos arrancado nosotros: si ya
    # estaba en marcha (rama de arriba), puede que lo este usando otra sesion o que lo haya
    # arrancado el usuario a mano para depurar, y no nos corresponde matarlo.
    atexit.register(_detener_arbol_proceso, proceso.pid)
    for _ in range(60):
        if mcp_directo.servidor_disponible():
            return True, "Conectado y listo."
        time.sleep(1)
    return False, "El servicio no responde todavía. Prueba a recargar la página en unos segundos."


def _reiniciar_conversacion():
    st.session_state["historial_chat"] = []


def _cargar_en_sesion(ruta: dict):
    st.session_state["ruta_activa"] = ruta
    st.session_state["prediccion"] = None
    _reiniciar_conversacion()


def _procesar_turno(texto: str):
    historial = st.session_state.get("historial_chat", [])
    historial = historial + [{"role": "user", "content": texto}]
    with st.spinner("Analizando la ruta..."):
        try:
            historial = asyncio.run(responder(historial))
        except Exception as error:  # noqa: BLE001 - mostrar el error en el chat, no tumbar la app
            historial = historial + [{"role": "assistant", "content": f"Error: {error}", "agente": "sistema"}]
    st.session_state["historial_chat"] = historial


st.title("SmartDeliveryAI")
st.caption(
    "Asistente inteligente para planificar y optimizar rutas de reparto. Las estimaciones de "
    "tiempo se basan en el histórico de entregas y son orientativas, no una promesa exacta "
    "para un día concreto."
)

servicio_ok, estado_servidor = asegurar_servidor_mcp()

with st.sidebar:
    st.subheader("Estado del servicio")
    if servicio_ok:
        st.success(estado_servidor)
    else:
        st.warning(estado_servidor)
    if not os.environ.get("OPENAI_API_KEY"):
        clave = st.text_input("Clave de acceso (API key)", type="password")
        if clave:
            os.environ["OPENAI_API_KEY"] = clave
    if not os.environ.get("OPENAI_MODEL"):
        modelo = st.text_input("Modelo de lenguaje", placeholder="p.ej. gpt-4o-mini")
        if modelo:
            os.environ["OPENAI_MODEL"] = modelo

    if mcp_directo.servidor_disponible():
        disponibles = mcp_directo.estado_modelos()
        if not disponibles.get("reparto"):
            st.warning(
                "El servicio de predicción de rutas no está disponible en este momento. "
                "Ponte en contacto con el equipo técnico."
            )

    st.divider()
    st.subheader("Ruta de trabajo")
    modo = st.radio("Origen de la ruta", ["Ruta ya completada (histórico)", "Nueva ruta"])

    if modo == "Ruta ya completada (histórico)":
        try:
            estaciones = mcp_directo.listar_estaciones()
        except Exception as error:
            estaciones = []
            st.error(f"No se pudieron cargar las estaciones: {error}")
        codigos = [e["station_code"] for e in estaciones]
        estacion = st.selectbox("Estación", ["(cualquiera)"] + codigos)
        estacion_filtro = None if estacion == "(cualquiera)" else estacion
        if st.button("Cargar una ruta al azar"):
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
            st.error(f"No se pudieron cargar las estaciones: {error}")
        codigos = [e["station_code"] for e in estaciones]
        if codigos:
            estacion = st.selectbox("Estación", codigos)
            departure_hour = st.slider("Hora de salida", 0, 23, 9)
            weekday = st.selectbox(
                "Día de la semana", options=list(range(7)),
                format_func=lambda d: ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"][d],
            )
            st.caption("Añade las paradas de la ruta (puedes agregar más filas). Las coordenadas son obligatorias.")
            paradas_df = st.data_editor(
                [{"lat": 0.0, "lng": 0.0, "packages_at_destination": 1, "packages_with_window_at_destination": 0}],
                num_rows="dynamic", key="editor_paradas",
            )
            if st.button("Crear ruta"):
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
        st.caption("Ruta real extraída del histórico de entregas de la estación.")
    else:
        st.caption("Ruta nueva: los datos que no se indicaron manualmente se han estimado a partir del histórico de la estación.")

    if st.session_state.get("prediccion") is None:
        try:
            st.session_state["prediccion"] = mcp_directo.predecir_ruta_activa()
        except Exception as error:
            st.error(f"No se pudo calcular la predicción de la ruta: {error}")

    prediccion = st.session_state.get("prediccion")
    col_mapa, col_tabla = st.columns([3, 2])
    with col_mapa:
        st.pydeck_chart(componentes.mapa_ruta(ruta["tramos"]))
        if prediccion:
            st.metric("Tiempo total estimado", f"{prediccion['tiempo_total_segundos'] / 60:.1f} min")
            if ruta.get("tiempos_reales_ocultos_segundos"):
                real = sum(ruta["tiempos_reales_ocultos_segundos"]) / 60
                st.metric("Tiempo real registrado", f"{real:.1f} min")
    with col_tabla:
        st.dataframe(componentes.tabla_ruta(ruta["tramos"], prediccion), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Pregúntale al asistente")
    cols = st.columns(len(PREGUNTAS_RAPIDAS))
    pregunta_rapida = None
    for c, pregunta in zip(cols, PREGUNTAS_RAPIDAS):
        if c.button(pregunta):
            pregunta_rapida = pregunta

    for mensaje in st.session_state.get("historial_chat", []):
        with st.chat_message(mensaje["role"]):
            nombre_agente = ETIQUETAS_AGENTE.get(mensaje.get("agente"))
            etiqueta = f"*({nombre_agente})*  \n" if nombre_agente else ""
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

    texto = st.chat_input("Escribe tu pregunta sobre esta ruta...")
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
    st.info("Selecciona o crea una ruta en el panel lateral para empezar.")
