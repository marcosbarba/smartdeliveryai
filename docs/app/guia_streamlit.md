# La interfaz Streamlit: cómo se comunica con el resto

Una sola página (`src/app/app.py`): sidebar para la intendencia (cargar una ruta real reservada
como producción simulada, o dar de alta una ruta manual con autocompletado), y área principal con
el mapa, la tabla de la ruta y el chat contra el grafo multiagente (`src/agente/grafo.py`).

## El modelo de ejecución de Streamlit, en una frase

Streamlit vuelve a ejecutar el script **entero**, de arriba a abajo, cada vez que el usuario
interactúa con cualquier widget — no hay un bucle de eventos con callbacks persistentes como en
una app de escritorio. Todo lo que debe sobrevivir entre interacciones vive en
`st.session_state`, un diccionario que Streamlit conserva entre *reruns* (`ruta_activa`,
`prediccion`, `historial_chat`).

## Dos caminos hacia el servidor MCP, a propósito

| Camino | Uso |
|---|---|
| `src/app/mcp_directo.py` | Llamadas MCP directas para el sidebar (formularios, sin LLM de por medio). |
| `src/agente/grafo.py` | El chat llega al mismo servidor a través del grafo multiagente. |

Cargar una ruta histórica o dar de alta una manual es una acción determinista de formulario, no
una petición en lenguaje natural — pasarla por el LLM sería más lento y menos fiable sin ganar
nada. El chat, en cambio, necesita razonamiento (qué especialista, qué argumentos) y sí pasa por
`grafo.responder(historial)`.

## Bridging async→sync: por qué todo pasa por `asyncio.run(...)`

`langchain-mcp-adapters` es una librería async (las llamadas HTTP se hacen con `httpx` async), pero
Streamlit ejecuta el script de forma síncrona: no hay ningún `await` disponible ni un *event loop*
ya corriendo. `mcp_directo._ejecutar(coro)` y `app._procesar_turno` envuelven cada llamada en
`asyncio.run(coro)`: abre un *event loop* nuevo, ejecuta la corrutina entera (por dentro puede
haber varios `await` encadenados — conectar, listar tools, invocar una, esperar la respuesta), y lo
cierra. Es seguro precisamente porque Streamlit nunca mete el script en un loop propio ni ejecuta
dos interacciones a la vez: no hay riesgo de `asyncio.run() cannot be called from a running event
loop`. Cada llamada crea además un `MultiServerMCPClient` nuevo (`cliente_mcp.crear_cliente()`):
como cada `asyncio.run()` abre y cierra su propio loop, no hay forma barata de mantener una
conexión persistente entre *reruns* — se asume que reconectar (*streamable-http*, sin sesión
pesada) es barato, y lo es.

## Por qué el mensaje del usuario se pinta antes de llamar al agente

Bug real y ya corregido: si el turno se guarda en `session_state["historial_chat"]` solo dentro de
`_procesar_turno` (después de esperar la respuesta del LLM), el bucle que dibuja el historial —que
se ejecuta ANTES en esa misma pasada del script— no lo ve todavía. El usuario escribía una pregunta
y no aparecía en pantalla hasta que llegaba la respuesta completa. Arreglo: pintar la burbuja del
usuario con `st.chat_message("user")` en el momento en que se captura el texto, antes de llamar a
`_procesar_turno` (que dispara el spinner y la llamada real al agente).

## Cómo se renderiza el resultado de optimizar una ruta

`grafo.responder()` no solo devuelve el texto del LLM: si el turno llamó a
`optimizar_ruta_activa`, extrae el resultado estructurado del `ToolMessage` correspondiente (mismo
formato de bloques de texto JSON que el resto de MCP, ver [guía MCP](../agente/guia_mcp.md)) y lo
cuelga como clave `"optimizacion"` en el mensaje final. `app.py`, al pintar ese mensaje, si trae
`"optimizacion"`, dibuja debajo el mapa (`componentes.mapa_ruta`, que ya soporta un parámetro
`orden` para redibujar la ruta en cualquier secuencia) y una tabla con el nuevo orden — el LLM no
enumera las paradas en su respuesta de texto (instrucción explícita en
`ROUTE_OPTIMIZATION_PROMPT`): esa información ya la muestra la interfaz directamente desde el
resultado de la herramienta, repetirla en prosa solo añadiría ruido.

## Autoarranque del servidor MCP

`asegurar_servidor_mcp()` comprueba primero si el servidor ya responde
(`mcp_directo.servidor_disponible()`); si no, lo arranca con `uv run sdai-mcp` como subproceso y
espera hasta 60s a que conteste. Es el mismo entry point que se usa para arrancarlo a mano al
depurar — un único comando probado en los dos contextos, en vez de mantener dos formas distintas de
invocar el mismo servidor que podrían divergir.
