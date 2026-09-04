# Contrato de servicio del modelo

Adaptado de "Instrucciones Para El Agente.txt" del proyecto original — describe el contrato de
servicio del modelo en sí (variables, imputaciones, reglas de negocio), relevante para cualquiera
que sirva el modelo, no solo para quien construya un agente sobre él.

## Qué predice, en una frase

Cuánto tiempo (en segundos) se tarda en ir de una parada a la siguiente dentro de una ruta de
reparto. Sumando los tramos de una ruta se obtiene la hora estimada de llegada a cualquier
parada. No es un solo modelo: son DOS modelos que se reparten el trabajo según el tipo de tramo.

## Dónde está todo

```text
artifacts/modelo/
  reparto_catboost.cbm      modelo para tramos de reparto (CatBoost)
  almacen_lightgbm.txt      modelo para el tramo de almacen (LightGBM)
  manifiesto.json           que variables necesita cada modelo, en que orden, cuales son
                             categoricas, y los avisos que no se pueden ignorar. LEER ANTES
                             DE ESCRIBIR CODIGO, no derivarlo del nombre de las columnas.
  model_card.md              ficha resumen: rendimiento, variables mas influyentes, limitaciones
  informe_shap.md             interpretabilidad completa, con graficos
  resultado.json               metricas de test y de SHAP en formato estructurado

artifacts/modelado/reports/  informes tecnicos del bake-off (por que se eligio esta arquitectura)
data/gold/Dataset Tramos Ruta/Dataset Tramos Ruta.parquet   tabla completa, 898.415 tramos
```

## Los dos modelos, y cómo elegir cuál usar

```text
is_depot_segment = 1   el tramo que sale del ALMACEN (el primero de cada ruta). Dura de media
                        unos 30 minutos. Usa almacen_lightgbm.txt

is_depot_segment = 0   el resto de tramos, entre dos entregas. Dura de media ~1 minuto.
                        Usa reparto_catboost.cbm
```

Por qué dos modelos y no uno: un modelo único entrenado con las dos poblaciones a la vez rendía
mal en el tramo de almacén, porque es menos del 1% de las filas y el modelo conjunto no le
prestaba atención (bake-off, ver `bakeoff_resumen.md`). **No reimplementar esta elección en cada
sitio que necesite predecir un tramo** — `src/agente/servidor_mcp/modelos.py` centraliza la
lógica en `predecir_mixto()`.

## Cómo cargar los modelos (Python)

```python
from catboost import CatBoostRegressor
import lightgbm as lgb

modelo_reparto = CatBoostRegressor()
modelo_reparto.load_model("artifacts/modelo/reparto_catboost.cbm")

modelo_almacen = lgb.Booster(model_file="artifacts/modelo/almacen_lightgbm.txt")
```

No hace falta ninguna librería adicional para servir el modelo más allá de `catboost` y
`lightgbm`. No hace falta MLflow para esto: se usó solo para el historial de experimentos del
bake-off, no es el camino para consumir el modelo entrenado.

**Aviso sobre el tamaño**: `reparto_catboost.cbm` pesa varios cientos de MB. Cargarlo UNA VEZ al
arrancar el proceso (`src/agente/servidor_mcp/mcp_server.py` lo hace con `@lru_cache`), nunca
dentro de cada petición.

## Qué variables construir para cada predicción

La lista exacta, con nombre y orden, está en `manifiesto.json` (claves `reparto.variables` y
`almacen.variables`). Categóricas: `station_code` en los dos modelos, y además `to_zone_id` en el
de reparto — pasarlas como texto, no como código entero: los dos modelos las tratan de forma
nativa a partir del valor de texto (LightGBM predice sobre el CÓDIGO de la columna category, así
que hay que reconstruir el dtype exacto de entrenamiento al servir, ver
`docs/agente/guia_mcp.md`, lección 2).

**Nulos que hay que tratar antes de predecir**:
- `same_zone` ausente → imputar a 0.
- `to_zone_id` ausente → imputar al texto literal `"SIN_ZONA"` (no omitir ni poner `None`: el
  modelo aprendió esa categoría como una categoría más).

## Reglas que no se pueden romper

- **Nunca calcular una velocidad** (distancia / tiempo) como variable de entrada para pedir una
  nueva predicción: reconstruiría el objetivo por división — es la definición de fuga de datos.
- **Recortar siempre la predicción a un mínimo de 0 segundos.**
- **El modelo no usa meteorología.** Se probó explícitamente (bake-off, `lightgbm_unico`) y no
  aporta, además de que incluirla sin cuidado enseñaría relaciones falsas (la lluvia parece
  importar pero es un efecto de qué ciudad es cada estación, ver `docs/preprocessing/`). Nunca
  decir "tu paquete llega tarde por la lluvia".
- **El objetivo es una media histórica**, no el tiempo real de un día concreto (90,1% de los
  pares de coordenadas del dataset repiten el mismo tiempo exacto en fechas distintas). Comunicar
  "tiempo estimado típico", nunca precisión de minuto exacto para un día específico.
- **Sesgo conocido en reparto**: el modelo tiende a infrapredecir en los tramos de reparto (no en
  el de almacén) — la cifra exacta para esta versión del modelo está en
  `artifacts/modelado/reports/analisis_errores.md` (`src/modelado/analisis_errores.py`, ejecutado
  dentro de `train`, nunca sobre `test`). Si se promete un horario, comunicarlo como rango, no
  como cifra exacta.
- **`segment_distance_km` es distancia en línea recta** (Haversine), no por carretera.

## Rendimiento

Ver `artifacts/modelo/model_card.md` (generado por `uv run sdai-train`, con la métrica sobre el
20% de rutas de test — nunca visto en el ajuste, ver `docs/modelado/model_card.md`). Si al
probarlo el error medio observado se aleja mucho de esas cifras, algo está mal montado (variables
en otro orden, categóricas mal pasadas, nulos sin imputar) — no es normal que el modelo rinda
peor en producción que en la evaluación ya verificada.

## Regenerar el modelo desde cero

```text
uv run sdai-train
```

Es el único comando que reentrena y sobrescribe `artifacts/modelo/`. Los scripts de
`src/modelado/bakeoff/` y `src/modelado/analisis_errores.py` solo miden, no dejan modelo guardado
para servir.
