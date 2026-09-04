# MCP explicado desde cero, y lecciones reales de construirlo

Para quien conoce LangChain (ha usado `@tool` para exponer funciones a un LLM) pero nunca ha
tocado MCP.

## El puente desde `@tool`

Con `@tool`, la función vive en el mismo proceso que el LLM:

```python
from langchain_core.tools import tool

@tool
def predecir_tramo(distancia_km: float, estacion: str) -> float:
    """Predice cuanto tarda un tramo."""
    return modelo.predict(...)

agente = create_agent(modelo_llm, [predecir_tramo])
```

Con MCP, lo único que cambia es **dónde vive la función**: el cuerpo vive en
`src/agente/servidor_mcp/mcp_server.py`, en otro proceso. El agente no importa esa función;
importa un **cliente** (`langchain-mcp-adapters`) que le pregunta al servidor qué herramientas
tiene, y por cada una construye un `BaseTool` normal cuyo cuerpo hace una petición HTTP en vez de
ejecutar Python local. Todo lo que ya se sabe sobre `@tool`, tool-calling, `AIMessage`,
`ToolMessage` sigue aplicando sin cambios; solo cambia la fontanería de cómo se creó la lista.

## Por qué separar el servidor en un proceso aparte

El modelo de reparto (CatBoost) pesa varios cientos de MB y tarda segundos en cargar. Streamlit
vuelve a ejecutar el script *entero* en cada interacción del usuario — si el modelo viviera en
el mismo proceso, habría que resolver "cárgalo una vez y no lo repitas" dentro del propio proceso
de Streamlit (posible con `st.cache_resource`, pero frágil frente a reinicios). Con el modelo
detrás de un servidor MCP aparte:

- Se arranca una vez, carga el modelo una vez, se queda vivo indefinidamente.
- Es ajeno a que Streamlit se reinicie o a que LangGraph cambie.
- Es reutilizable por cualquier otro cliente MCP (otro framework de agentes, Claude Desktop, un
  script de pruebas) sin tocar `modelos.py`.

## El protocolo, en la práctica

Un servidor MCP expone **tools** (aquí, las únicas usadas), *resources* y *prompts*. Cada tool
tiene un nombre, una descripción en lenguaje natural (el docstring de la función — el LLM lo lee
para decidir si le sirve, así que están escritos como si le hablaran al modelo, no como
comentarios para otro programador) y un schema de argumentos generado por introspección de los
type hints (`FastMCP`, igual que hace `@tool`).

**La petición y la respuesta viajan como texto**, en bloques `{"type": "text", "text":
"<json>"}`, no como objetos Python serializados: MCP está diseñado para que el cliente pueda
estar en cualquier lenguaje.

## Lecciones reales (los errores que de verdad costaron tiempo)

1. **Los bloques de texto no son un dict ya parseado.** Al llamar a una tool que devuelve una
   lista, se recibe una lista de bloques de texto (uno por elemento), no `list[dict]`; al
   devolver un dict, un único bloque. `src/app/mcp_directo.py` resuelve la ambigüedad con
   `_llamar_dict`/`_llamar_lista`: el llamador declara qué forma espera, no se adivina por la
   longitud del resultado.
2. **LightGBM nativo predice sobre el CÓDIGO de la categoría, no sobre el texto**, y ese código
   depende de qué categorías estaban presentes al hacer `.astype("category")`. Sin fijar la
   lista de categorías del entrenamiento (guardada en `manifiesto.json`,
   `categorias_station_code`), una predicción de una sola fila recodificaría la única estación
   presente como código 0 — casi nunca el correcto. Solución en `modelos.py`:
   `pd.Categorical(x, categories=categorias_guardadas)` al servir, nunca `.astype("category")` a
   secas.
3. **Un servidor MCP que sigue vivo desde una sesión anterior sirve código antiguo, en
   silencio, para siempre.** El servidor importa sus módulos una vez al arrancar (es un proceso
   de larga duración a propósito). Si se edita `src/agente/servidor_mcp/` y no se reinicia el
   proceso, sigue respondiendo con la versión que tenía cargada en memoria — sin ningún error,
   solo resultados que parecen plausibles pero no lo son. `uv run streamlit run src/app/app.py`
   solo arranca el servidor MCP si detecta que el puerto **no** responde; si uno viejo ya está
   vivo, lo da por bueno. Comprobación rápida: `netstat -ano | grep :8765` para confirmar que
   solo hay un proceso escuchando, y revisar el log del servidor por si el `bind` falló
   silenciosamente (error de Windows "solo se permite un uso de cada dirección de socket").
4. **`optimizar_ruta_activa` no puede usar una matriz de costes NxN precomputada.** El modelo
   trata cada tramo A→B con coste dependiente de la posición: `segment_position` y
   `cumulative_distance_km` son variables de entrada. El mismo salto A→B predice un tiempo
   distinto en el tramo 2 que en el tramo 15. `src/agente/servidor_mcp/optimizador.py` construye
   la ruta de forma voraz paso a paso, evaluando en cada paso lo que el modelo predeciría para
   cada parada restante EN LA POSICIÓN REAL que ocuparía, con un 2-opt acotado que recalcula el
   sufijo afectado por cada intercambio candidato con el modelo real, no con distancia bruta.
5. **El tramo de almacén necesita el especialista de almacén, no el de reparto, incluso dentro
   del optimizador.** El modelo de reparto (CatBoost) nunca vio un tramo de almacén en
   entrenamiento; usarlo ahí es extrapolación fuera de dominio. `modelos.predecir_mixto()`
   centraliza la lógica "qué especialista según `is_depot_segment`" en un único sitio, para que
   no se repita (y se pueda volver a romper) en cada fichero que necesite predecir un tramo.
