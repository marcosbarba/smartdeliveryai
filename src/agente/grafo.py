"""El sistema multiagente: un supervisor que enruta a dos especialistas mediante handoffs
reales (no una unica llamada a LLM con lista de tools). Ver
docs/agente/arquitectura_multiagente.md para la justificacion de por que esto es
"multiagente" y un unico LLM con tools no lo seria.

Patron: el supervisor y cada especialista son agentes ReAct completos
(langchain.agents.create_agent), cada uno con su propio system prompt (agente/prompts.py) y
su propio subconjunto de herramientas MCP. El supervisor tiene ademas dos "tools de traspaso"
que devuelven un Command(goto=..., graph=Command.PARENT): al ejecutarlas, LangGraph desvia la
ejecucion al nodo especialista correspondiente en el grafo padre, en vez de que el supervisor
conteste el mismo.

Nota de version: langgraph.prebuilt.create_react_agent quedo obsoleto en langgraph >= 1.0 a
favor de langchain.agents.create_agent (mismo patron ReAct, ahora con un sistema de
middleware detras). La unica diferencia de uso relevante aqui es que el parametro se llama
system_prompt, no prompt.
"""

from __future__ import annotations

import json
import os

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from agente.cliente_mcp import (
    NOMBRES_TOOLS_ETA,
    NOMBRES_TOOLS_OPTIMIZACION,
    NOMBRES_TOOLS_SUPERVISOR,
    obtener_tools_por_nombre,
)
from agente.prompts import ETA_EXPLANATION_PROMPT, ROUTE_OPTIMIZATION_PROMPT, SUPERVISOR_PROMPT


def _modelo_openai() -> ChatOpenAI:
    modelo = os.environ.get("OPENAI_MODEL")
    if not modelo:
        raise RuntimeError(
            "Falta OPENAI_MODEL en el entorno (.env). No se asume ningun modelo por defecto "
            "para no arriesgarse a fijar un nombre incorrecto u obsoleto: pon el que tengas "
            "disponible con tu API key."
        )
    return ChatOpenAI(model=modelo, temperature=0)


async def construir_grafo() -> CompiledStateGraph:
    tools = await obtener_tools_por_nombre()
    modelo = _modelo_openai()

    route_optimization_agent = create_agent(
        modelo,
        [tools[n] for n in NOMBRES_TOOLS_OPTIMIZACION],
        system_prompt=ROUTE_OPTIMIZATION_PROMPT,
        name="route_optimization_agent",
    )
    eta_explanation_agent = create_agent(
        modelo,
        [tools[n] for n in NOMBRES_TOOLS_ETA],
        system_prompt=ETA_EXPLANATION_PROMPT,
        name="eta_explanation_agent",
    )

    @tool
    def transfer_to_route_optimization_agent() -> Command:
        """Hand off to the Route Optimization Agent. Use this when the user wants to
        reorder, reroute or otherwise optimize the active route's stop sequence."""
        return Command(goto="route_optimization_agent", graph=Command.PARENT)

    @tool
    def transfer_to_eta_explanation_agent() -> Command:
        """Hand off to the ETA & Explanation Agent. Use this when the user asks about
        arrival times, ETAs, or why a segment takes as long as it does."""
        return Command(goto="eta_explanation_agent", graph=Command.PARENT)

    tools_supervisor = [tools[n] for n in NOMBRES_TOOLS_SUPERVISOR] + [
        transfer_to_route_optimization_agent,
        transfer_to_eta_explanation_agent,
    ]
    supervisor_agent = create_agent(
        modelo, tools_supervisor, system_prompt=SUPERVISOR_PROMPT, name="supervisor",
    )

    grafo = StateGraph(MessagesState)
    grafo.add_node("supervisor", supervisor_agent)
    grafo.add_node("route_optimization_agent", route_optimization_agent)
    grafo.add_node("eta_explanation_agent", eta_explanation_agent)
    grafo.add_edge(START, "supervisor")
    grafo.add_edge("supervisor", END)
    grafo.add_edge("route_optimization_agent", END)
    grafo.add_edge("eta_explanation_agent", END)

    return grafo.compile()


def _contenido_tool_a_dict(contenido) -> dict | None:
    """El contenido de un ToolMessage que viene de servidor_mcp llega como lista de bloques
    {"type": "text", "text": "<json>"} (mismo formato que app/mcp_directo.py deserializa
    para las llamadas directas), no como dict ya parseado."""
    if isinstance(contenido, list):
        contenido = next((b.get("text") for b in contenido if isinstance(b, dict) and "text" in b), None)
    if not isinstance(contenido, str):
        return None
    try:
        return json.loads(contenido)
    except (json.JSONDecodeError, TypeError):
        return None


async def responder(historial: list[dict]) -> list[dict]:
    """historial: lista de {"role": "user"|"assistant", "content": str}. Devuelve el
    historial completo actualizado con la respuesta del turno (uno o mas mensajes: el
    supervisor y, si hubo traspaso, tambien el especialista). Si el turno llamo a
    optimizar_ruta_activa, el resultado estructurado (orden propuesto, ahorro) viaja en la
    clave "optimizacion" del ultimo mensaje, para que la interfaz pueda dibujarlo como mapa."""
    grafo = await construir_grafo()
    mensajes: list[BaseMessage] = [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in historial
    ]
    resultado = await grafo.ainvoke(
        {"messages": mensajes},
        config={"run_name": "chat_turn", "tags": [f"turno_{len(historial) + 1}"]},
    )
    nuevos = resultado["messages"][len(mensajes):]

    resultado_optimizacion = None
    for m in nuevos:
        if isinstance(m, ToolMessage) and m.name == "optimizar_ruta_activa":
            resultado_optimizacion = _contenido_tool_a_dict(m.content)

    turnos = [
        {"role": "assistant", "content": m.content, "agente": getattr(m, "name", None) or "supervisor"}
        for m in nuevos
        if isinstance(m, AIMessage) and m.content
    ]
    if resultado_optimizacion is not None and turnos:
        turnos[-1]["optimizacion"] = resultado_optimizacion
    return historial + turnos
