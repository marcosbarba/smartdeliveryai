# SmartDeliveryAI

Predicción de la hora de llegada de un paquete (Amazon Last Mile Routing Research Challenge:
6.112 rutas reales, 5 áreas metropolitanas de EEUU) y un sistema multiagente que explica el
porqué y sugiere un mejor orden de reparto.

```text
Variable objetivo: travel_time_seconds — segundos entre dos paradas consecutivas
Se predice por TRAMO, no por ruta
Tabla del modelo: data/gold/Dataset Tramos Ruta/ — 898.415 filas
```

Documentación técnica completa (arquitectura de datos, decisiones y limitaciones de cada
módulo, el sistema multiagente, la interfaz): **`docs/README.md`**.

## Estructura

```text
src/
  preprocessing/   Bronze -> Silver -> Gold (PySpark) + EDA reproducible
  modelado/         split train/test fijo, bake-off de arquitecturas, entrenamiento final
  agente/           servidor MCP + grafo LangGraph (supervisor + 2 especialistas)
  app/              interfaz Streamlit
notebooks/          EDA narrado (ya ejecutado); bake-off de modelos, pendiente de narrar
data/               raw_amazon/, bronze/, silver/, gold/, produccion_simulada/
artifacts/          modelos entrenados, figuras, informes (todo regenerable, nada se edita a mano)
docs/               documentación técnica por módulo — léela primero
```

## Cómo poner esto en marcha

Requisitos previos (no los instala `uv`): Python 3.12 (PySpark no soporta 3.13+ todavía). Solo
para ejecutar el pipeline de datos con PySpark (no hace falta ni para el agente ni para la app):
JDK 17 — o bien instalado en el sistema, o bien un JDK portable (Eclipse Temurin, sin instalador)
descomprimido en `.jdk/` en la raíz del proyecto, que los scripts detectan solos — y en Windows
`winutils.exe`/`hadoop.dll` en `.hadoop/bin/`.

```text
1. Instalar dependencias (elige los extras que necesites, o todos):

   uv sync --all-extras

2. Descargar el dataset origen (una vez; el bloque de tiempos de viaje es obligatorio,
   ver data/raw_amazon/Descargar Dataset Amazon.sh):

   data/raw_amazon/Descargar Dataset Amazon.sh --con-tiempos-viaje

3. Construir el pipeline de datos (Bronze -> Silver -> Gold):

   uv run sdai-pipeline

4. Entrenar el modelo (split 80/20 fijo, bake-off ya decidido — ver docs/modelado/):

   uv run sdai-train

5. Copiar .env.example a .env en la raíz del proyecto y rellenar OPENAI_API_KEY/OPENAI_MODEL.

6. Arrancar la interfaz (arranca el servidor MCP sola si no está corriendo):

   uv run sdai-app
```

Equivalente a `uv run streamlit run src/app/app.py` (así es como lo ve Streamlit por dentro; `sdai-app`
es solo el entry point instalado por `uv`). El servidor MCP también se puede arrancar a mano para
depurar: `uv run sdai-mcp`.

## Reglas del proyecto

- No se paga ningún servicio cloud de infraestructura: todo corre en local. La única llamada de
  pago es a la API de OpenAI para el LLM del agente.
- No editar a mano nada dentro de `artifacts/` ni `data/{bronze,silver,gold,produccion_simulada}/`:
  se regeneran con `uv run sdai-pipeline` / `uv run sdai-train`.
- Sin PDF en este repositorio: toda la documentación queda en Markdown.
- Este proyecto es la base técnica para escribir la memoria académica del TFM (por separado, no
  incluida aquí) apoyándose en `docs/`.
