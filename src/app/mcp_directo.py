"""Llamadas directas al servidor MCP para las acciones del sidebar (cargar estacion, cargar
ruta historica, dar de alta una ruta manual): no necesitan pasar por el LLM, son formularios,
no lenguaje natural. El chat, en cambio, llega al mismo servidor a traves del grafo
multiagente (agente/grafo.py).
"""

from __future__ import annotations

import asyncio
import json

from agente.cliente_mcp import crear_cliente


def _ejecutar(coro):
    return asyncio.run(coro)


def _bloques_a_valores(resultado) -> list:
    """Las tools MCP devuelven una lista de bloques de contenido {"type": "text", "text":
    "<json>"}: un bloque por elemento cuando la tool devuelve una lista, un unico bloque
    cuando devuelve un dict. Aqui solo se deserializa cada bloque; quien llama decide si
    espera un dict (un bloque) o una lista (uno o mas bloques) segun la tool invocada, para
    no tener que adivinarlo por la longitud."""
    if not isinstance(resultado, list):
        return [resultado]
    salida = []
    for bloque in resultado:
        if isinstance(bloque, dict) and "text" in bloque:
            try:
                salida.append(json.loads(bloque["text"]))
            except (json.JSONDecodeError, TypeError):
                salida.append(bloque["text"])
        else:
            salida.append(bloque)
    return salida


async def _obtener_tool(nombre_tool: str):
    cliente = crear_cliente()
    tools = await cliente.get_tools()
    coincidencias = [t for t in tools if t.name == nombre_tool]
    if not coincidencias:
        raise RuntimeError(f"El servidor MCP no expone la herramienta '{nombre_tool}'.")
    return coincidencias[0]


async def _llamar_dict(nombre_tool: str, **kwargs) -> dict:
    """Para tools cuyo tipo de retorno es un unico dict."""
    tool = await _obtener_tool(nombre_tool)
    bloques = _bloques_a_valores(await tool.ainvoke(kwargs))
    return bloques[0] if bloques else {}


async def _llamar_lista(nombre_tool: str, **kwargs) -> list:
    """Para tools cuyo tipo de retorno es una lista."""
    tool = await _obtener_tool(nombre_tool)
    return _bloques_a_valores(await tool.ainvoke(kwargs))


def listar_estaciones() -> list[dict]:
    return _ejecutar(_llamar_lista("listar_estaciones"))


def listar_rutas_produccion(station_code: str | None = None, limite: int = 20) -> list[dict]:
    return _ejecutar(_llamar_lista("listar_rutas_produccion", station_code=station_code, limite=limite))


def cargar_ruta_historica(route_id: str | None = None) -> dict:
    return _ejecutar(_llamar_dict("cargar_ruta_historica", route_id=route_id))


def generar_ruta_manual(station_code: str, departure_hour: int, weekday: int, paradas: list[dict]) -> dict:
    return _ejecutar(
        _llamar_dict("generar_ruta_manual", station_code=station_code, departure_hour=departure_hour, weekday=weekday, paradas=paradas)
    )


def predecir_ruta_activa() -> dict:
    return _ejecutar(_llamar_dict("predecir_ruta_activa"))


def estado_modelos() -> dict:
    return _ejecutar(_llamar_dict("estado_modelos"))


def servidor_disponible() -> bool:
    try:
        estado_modelos()
        return True
    except Exception:
        return False
