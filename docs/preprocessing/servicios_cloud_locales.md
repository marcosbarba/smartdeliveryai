# Servicios cloud replicados en local

> Versión corregida de "Servicios Cloud Locales.md" del proyecto original: aquel documento
> describía MLflow, el registro de modelos, la API y el dashboard como "pendientes", escrito antes
> de que existieran. Ya están todos implementados, con una diferencia deliberada respecto al plan
> inicial: no hay FastAPI, hay un servidor MCP (ver [guía MCP](../agente/guia_mcp.md)).

## Mapa de equivalencias, estado real

| Equivalente cloud | Réplica local | Estado |
|---|---|---|
| S3 / Azure Data Lake / GCS | `data/{bronze,silver,gold}/` | Hecho |
| EMR / Synapse / Databricks | PySpark local | Hecho |
| Glue Workflows / Data Factory | `src/preprocessing/pipeline/` (script) | Hecho |
| Glue Catalog / Purview | Contratos de datos en Markdown | Hecho |
| Glue Data Quality / Deequ | Informes de calidad JSON (propios) | Hecho |
| SageMaker / Azure ML Experiments | MLflow local (SQLite, `artifacts/modelado/`) | Hecho |
| Model Registry cloud | `artifacts/modelo/` + `manifiesto.json` | Hecho |
| SageMaker / Azure ML Endpoint | Servidor MCP (`src/agente/servidor_mcp/`) | Hecho — MCP, no REST |
| QuickSight / Power BI / Looker | Streamlit (`src/app/`) | Hecho |
| CloudWatch / Azure Monitor | Logs de proceso, sin *drift monitoring* | No hecho |
| MinIO / Docker / Prefect / Airflow | — | No hecho, no necesario |

## Por qué MCP y no FastAPI

El plan inicial preveía una API REST (`POST /predict`) consumida por un dashboard y por agentes.
Se construyó en su lugar un servidor MCP (Model Context Protocol): cumple el mismo papel
estructural —separar el modelo cargado en memoria (proceso de larga duración, ~480 MB el CatBoost
de reparto) del proceso que lo consume— pero con un protocolo diseñado específicamente para que un
agente LLM descubra y llame herramientas, en vez de una API REST genérica que habría necesitado
documentación OpenAPI aparte y un cliente HTTP escrito a mano en el lado del agente.
`langchain-mcp-adapters` traduce las herramientas del servidor MCP a herramientas LangChain
estándar automáticamente. Detalle completo en [guía MCP](../agente/guia_mcp.md).

## Qué no se implementó, y por qué no hacía falta

- **Monitorización de drift** (Evidently AI o similar): tendría sentido con datos de producción
  reales llegando de forma continua; este proyecto usa un banco de "producción simulada" fijo (ver
  [docs/modelado/](../modelado/)), no un flujo en vivo que pueda derivar con el tiempo.
- **MinIO/Docker**: el Data Lake por carpetas locales ya demuestra el patrón Bronze/Silver/Gold;
  montar un S3 local no cambia ninguna decisión de arquitectura, solo la sintaxis de las rutas.
- **Prefect/Airflow**: el pipeline tiene 4 pasos secuenciales con dependencias simples
  (Bronze → Silver → Gold → EDA); un orquestador dedicado añadiría infraestructura sin resolver un
  problema real a esta escala. El script `sdai-pipeline` cubre la reproducibilidad que un
  orquestador daría aquí.

## Frase resumen (para citar en la memoria)

La arquitectura replica en local, sin coste, los componentes principales de una plataforma Big
Data/MLOps empresarial: almacenamiento por capas Bronze/Silver/Gold, ingesta de fuentes externas,
procesamiento PySpark, contrato de datos, calidad de datos, seguimiento de experimentos (MLflow),
registro de modelos, un servicio que sirve el modelo (MCP, no REST) y un dashboard (Streamlit) —
con un diseño migrable a AWS/Azure/GCP si el volumen creciera en un escenario empresarial real.
