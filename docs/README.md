# SmartDeliveryAI — documentación técnica

Este proyecto predice **a qué hora llega un paquete al domicilio del cliente**, usando datos
reales de reparto de Amazon (Amazon Last Mile Routing Research Challenge: 6.112 rutas, 5 áreas
metropolitanas de EEUU, verano de 2018), y expone esa predicción a través de un sistema
multiagente que además explica el porqué y sugiere un mejor orden de reparto.

```text
Variable objetivo: travel_time_seconds — segundos entre dos paradas consecutivas
Se predice por TRAMO, no por ruta: sumando tramos se obtiene la hora de llegada a cualquier parada
Tabla del modelo: data/gold/Dataset Tramos Ruta/ — 898.415 filas
```

El tiempo de servicio (cuánto tarda el repartidor parado en la puerta) no se predice: viene
planificado por Amazon y es conocido antes de salir — es el 60% de la duración total de una
ruta, así que el modelo solo estima la parte de conducir.

## Cómo está organizada esta documentación

Cada carpeta es un módulo de `src/`, en el mismo orden en que los datos fluyen por el proyecto:

```text
preprocessing/   arquitectura de datos: Bronze -> Silver -> Gold, y el EDA que condiciona el modelado
modelado/        cómo se eligió la arquitectura del modelo y qué garantiza su evaluación
agente/          el sistema multiagente (LangGraph + MCP) que sirve el modelo
app/             la interfaz Streamlit que lo pone delante de un repartidor o dispatcher
```

Dentro de cada una, los documentos combinan **por qué** se decidió algo con **qué limitaciones**
tiene esa decisión — el objetivo es que sirvan de base directa para escribir la memoria
académica del TFM sin tener que reconstruir el razonamiento desde cero.

## El recorrido completo, en una imagen mental

```text
data/raw_amazon (JSON originales de Amazon)
  -> data/bronze     (fuentes originales, casi sin tocar: Amazon + meteorología + festivos)
  -> data/silver     (limpio, tipado, sin duplicados)
  -> data/gold       (dataset final: Dataset Tramos Ruta + Dataset Rutas Enriquecidas)
  -> notebooks/01_eda  (seis hallazgos que condicionan todo lo que viene después)
  -> src/modelado    (bake-off de arquitecturas -> modelo final: CatBoost + LightGBM, evaluado
                       sobre un 20% de rutas nunca visto en el ajuste)
  -> src/agente      (servidor MCP + grafo LangGraph: supervisor + 2 especialistas)
  -> src/app          (Streamlit: mapa, tabla, chat contra el agente)
```

## Qué NO es este proyecto (limitaciones que hay que citar siempre)

- El objetivo son **medias históricas** por par de coordenadas (el 90,1% de los pares repite el
  mismo tiempo exacto entre fechas distintas): el modelo predice la duración *típica* de un
  tramo, no una medición de un día concreto.
- **No hay meteorología ni tráfico en tiempo real** en el modelo: se midió explícitamente que la
  lluvia no aporta señal real (ver `preprocessing/hallazgos_eda.md`, hallazgo 5).
- **Sin GPS real**: las distancias son en línea recta (Haversine), no por carretera.
- **Sin estacionalidad**: cinco semanas de verano de 2018, una sola estación del año.
- No se paga ningún servicio cloud de infraestructura en todo el proyecto; la única llamada de
  pago es a la API de OpenAI para el LLM del agente, inherente a "usar un LLM".
