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

### Qué falta tras clonar el repositorio

Por `.gitignore`, quien clona el repo NO tiene todavía varias cosas que el agente necesita para
funcionar, y hay que generarlas o rellenarlas en local siguiendo los pasos de más abajo:

- **`.env`** — no se versiona nunca. Sin él el agente no arranca: hace falta como mínimo una
  API key de OpenAI (paso 5).
- **El dataset original** (`data/raw_amazon/.../*.json`, varios GB) — se descarga con el script
  del paso 2, no viene en git.
- **`data/silver/` y `data/gold/`** — salidas del pipeline de datos; se regeneran a partir del
  dataset descargado con `uv run sdai-pipeline` (paso 3).
- **El modelo de reparto** (`artifacts/modelo/reparto_catboost.cbm`, ~480 MB) **y el banco de
  rutas de producción simulada** (`data/produccion_simulada/*.parquet`) — no caben en GitHub
  (límite de 100 MB) y se generan entrenando en local con `uv run sdai-train` (paso 4, ~75 min).
  Sin esto, la app arranca pero avisa de que el modelo de reparto no está entrenado y las
  predicciones fallan.
- **`.venv/`** — se crea instalando las dependencias (paso 1).

Lo único que SÍ viene incluido y no hace falta regenerar es el modelo de almacén
(`artifacts/modelo/almacen_lightgbm.txt`, ~750 KB) y el resto de `artifacts/modelo/` (model
card, informe SHAP, gráficos): pesan poco y son documentación legible del entrenamiento ya
hecho.

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

5. Copiar .env.example a .env en la raíz del proyecto y rellenar, como mínimo:

   OPENAI_API_KEY   tu clave de https://platform.openai.com/api-keys
   OPENAI_MODEL     un modelo con tool-calling disponible en tu cuenta (p.ej. gpt-4o-mini)

   Sin estas dos variables el agente no puede arrancar: son las que usa el sistema
   multiagente (supervisor + especialistas) para razonar y llamar a las tools del servidor
   MCP. El resto de variables de .env.example (puerto del MCP, trazas de LangSmith) son
   opcionales.

6. Arrancar la interfaz (arranca el servidor MCP sola si no está corriendo):

   uv run sdai-app
```

Equivalente a `uv run streamlit run src/app/app.py` (así es como lo ve Streamlit por dentro; `sdai-app`
es solo el entry point instalado por `uv`). El servidor MCP también se puede arrancar a mano para
depurar: `uv run sdai-mcp`.

### Alternativa sin `uv` (pip + venv)

Mismos pasos, cambiando `uv sync`/`uv run` por un venv + pip corriente. Sigue haciendo falta
Python 3.12.

```text
1. Crear y activar el entorno virtual:

   py -3.12 -m venv .venv                  (Windows, PowerShell/cmd)
   python3.12 -m venv .venv                (Linux/macOS)

   .venv\Scripts\Activate.ps1              (Windows, PowerShell)
   source .venv/bin/activate               (Linux/macOS)

2. Instalar el proyecto en modo editable con los extras que necesites (la lista completa de
   extras — pipeline, modelado, deep-learning, agente, app, notebooks — está en pyproject.toml).
   Para levantar solo el agente y la interfaz:

   pip install -e ".[agente,app]"

   El extra deep-learning (torch) usa en uv un indice aparte (pytorch-cpu, ver
   [tool.uv.sources] en pyproject.toml) que pip no lee solo; hay que pasarlo a mano:

   pip install -e ".[deep-learning]" --extra-index-url https://download.pytorch.org/whl/cpu

3. Los pasos 2-6 de arriba (descargar el dataset, pipeline, entrenamiento, .env, arrancar la
   app) son iguales quitando el prefijo `uv run`: con el venv activado, los mismos comandos
   quedan instalados como scripts propios —

   sdai-pipeline
   sdai-train
   sdai-app
   sdai-mcp
```

La única diferencia real con `uv run` es que el venv hay que activarlo tú mismo en cada
terminal nueva; `uv run` lo hace por ti sin activación explícita.

## Reglas del proyecto

- No se paga ningún servicio cloud de infraestructura: todo corre en local. La única llamada de
  pago es a la API de OpenAI para el LLM del agente.
- No editar a mano nada dentro de `artifacts/` ni `data/{bronze,silver,gold,produccion_simulada}/`:
  se regeneran con `uv run sdai-pipeline` / `uv run sdai-train`.
- Sin PDF en este repositorio: toda la documentación queda en Markdown.
- Este proyecto es la base técnica para escribir la memoria académica del TFM (por separado, no
  incluida aquí) apoyándose en `docs/`.
