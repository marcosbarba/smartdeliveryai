"""Cliente MCP: conecta el grafo de LangGraph con servidor_mcp/mcp_server.py y traduce sus
herramientas a herramientas LangChain estandar (langchain-mcp-adapters).

servidor_mcp corre en un proceso aparte (streamable-http); este cliente solo abre peticiones
HTTP, no gestiona ningun subproceso, asi que reconectar en cada turno de chat es barato.
"""

from __future__ import annotations

import os

from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_SERVER_PORT = os.environ.get("MCP_SERVER_PORT", "8765")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", f"http://127.0.0.1:{MCP_SERVER_PORT}/mcp")


def crear_cliente() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "reparto": {
                "transport": "streamable_http",
                "url": MCP_SERVER_URL,
            }
        }
    )


async def obtener_tools_por_nombre() -> dict:
    """Todas las tools del servidor MCP, indexadas por nombre, para que cada nodo del grafo
    coja solo el subconjunto que necesita."""
    cliente = crear_cliente()
    tools = await cliente.get_tools()
    return {tool.name: tool for tool in tools}


NOMBRES_TOOLS_SUPERVISOR = [
    "listar_estaciones", "listar_rutas_produccion", "cargar_ruta_historica",
    "generar_ruta_manual", "estado_ruta_activa", "estado_modelos",
]
# Los especialistas tambien pueden cargar una ruta historica (no dar de alta una manual: esa
# necesita datos que el usuario da paso a paso, mejor que la resuelva el supervisor). Asi, si
# el supervisor delega directamente sobre una peticion compuesta ("carga la ruta de la
# estacion X y dime..."), el especialista no se queda bloqueado esperando una ruta activa que
# nadie cargo.
NOMBRES_TOOLS_OPTIMIZACION = [
    "optimizar_ruta_activa", "predecir_ruta_activa", "estado_ruta_activa",
    "listar_rutas_produccion", "cargar_ruta_historica",
]
NOMBRES_TOOLS_ETA = [
    "predecir_ruta_activa", "explicar_tramo_activo", "estado_ruta_activa",
    "listar_rutas_produccion", "cargar_ruta_historica",
]
