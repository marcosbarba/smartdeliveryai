# Contrato de servicio del modelo

> Adaptado de "Instrucciones Para El Agente.txt" del proyecto original. Describe el contrato de
> servicio del modelo en sí (variables, imputaciones, reglas de negocio) — relevante para
> cualquiera que sirva el modelo, no solo para quien construya un agente sobre él.

## Qué predice, en una frase

Cuánto tiempo (en segundos) se tarda en ir de una parada a la siguiente dentro de una ruta de
reparto. Sumando los tramos de una ruta se obtiene la hora estimada de llegada a cualquier parada.
No es un solo modelo: son **dos** modelos que se reparten el trabajo según el tipo de tramo.

## Dónde está todo

```text
artifacts/modelo/
  reparto_catboost.cbm      modelo para tramos de reparto (CatBoost)
  almacen_lightgbm.txt      modelo para el tramo de almacén (LightGBM)
  manifiesto.json           qué variables necesita cada modelo, en qué orden, cuáles son
                             categóricas, y los avisos que no se pueden ignorar.
                             LEER ANTES DE ESCRIBIR CÓDIGO, no derivarlo del nombre de columnas.
  model_card.md              ficha resumen: rendimiento, variables más influyentes, limitaciones
  informe_shap.md             interpretabilidad completa, con gráficos
  resultado.json               métricas de test y de SHAP en formato estructurado

artifacts/modelado/reports/  informes técnicos del bake-off (por qué se eligió esta arquitectura)
data/gold/Dataset Tramos Ruta/Dataset Tramos Ruta.parquet   tabla completa, 898 415 tramos
```

## Los dos modelos, y cómo elegir cuál usar

| `is_depot_segment` | Significado | Duración media | Modelo |
|---:|---|---|---|
| 1 | Tramo que sale del ALMACÉN (el primero de cada ruta) | ~30 min | `almacen_lightgbm.txt` |
| 0 | Resto de tramos, entre dos entregas | ~1 min | `reparto_catboost.cbm` |

Por qué dos modelos y no uno: un modelo único entrenado con las dos poblaciones a la vez rendía mal
en el tramo de almacén, porque es menos del 1% de las filas y el modelo conjunto no le prestaba
atención (bake-off, ver [bakeoff_resumen.md](bakeoff_resumen.md)). **No reimplementar esta
elección en cada sitio que necesite predecir un tramo** — `src/agente/servidor_mcp/modelos.py`
centraliza la lógica en `predecir_mixto()`.

## Cómo cargar los modelos (Python)

```python
from catboost import CatBoostRegressor
import lightgbm as lgb

modelo_reparto = CatBoostRegressor()
modelo_reparto.load_model("artifacts/modelo/reparto_catboost.cbm")

modelo_almacen = lgb.Booster(model_file="artifacts/modelo/almacen_lightgbm.txt")
```

No hace falta ninguna librería adicional para servir el modelo más allá de `catboost` y
`lightgbm`. Tampoco hace falta MLflow para esto: se usó solo para el historial de experimentos del
bake-off, no es el camino para consumir el modelo entrenado.

**Aviso sobre el tamaño**: `reparto_catboost.cbm` pesa varios cientos de MB. Cargarlo **una vez**
al arrancar el proceso (`src/agente/servidor_mcp/mcp_server.py` lo hace con `@lru_cache`), nunca
dentro de cada petición.

## Qué variables construir para cada predicción

La lista exacta, con nombre y orden, está en `manifiesto.json` (claves `reparto.variables` y
`almacen.variables`). Categóricas: `station_code` en los dos modelos, y además `to_zone_id` en el
de reparto — pasarlas como texto, no como código entero: los dos modelos las tratan de forma
nativa a partir del valor de texto (LightGBM predice sobre el código de la columna `category`, así
que hay que reconstruir el dtype exacto de entrenamiento al servir, ver
[guía MCP](../agente/guia_mcp.md), lección 2).

**Nulos que hay que tratar antes de predecir**:

- `same_zone` ausente → imputar a `0`.
- `to_zone_id` ausente → imputar al texto literal `"SIN_ZONA"` (no omitir ni poner `None`: el
  modelo aprendió esa categoría como una categoría más).

## Reglas que no se pueden romper

- **Nunca calcular una velocidad** (distancia / tiempo) como variable de entrada para pedir una
  nueva predicción: reconstruiría el objetivo por división — es la definición de fuga de datos.
- **Recortar siempre la predicción a un mínimo de 0 segundos.**
- **El modelo no usa meteorología.** Se probó explícitamente (bake-off, `lightgbm_unico`) y no
  aporta, además de que incluirla sin cuidado enseñaría relaciones falsas (la lluvia parece
  importar pero es un efecto de qué ciudad es cada estación, ver
  [docs/preprocessing/](../preprocessing/)). Nunca decir "tu paquete llega tarde por la lluvia".
- **El objetivo es una media histórica**, no el tiempo real de un día concreto (90,1% de los pares
  de coordenadas del dataset repiten el mismo tiempo exacto en fechas distintas). Comunicar "tiempo
  estimado típico", nunca precisión de minuto exacto para un día específico.
- **Sesgo conocido en reparto**: el modelo tiende a infrapredecir en los tramos de reparto (no en
  el de almacén) — la cifra exacta para esta versión del modelo está en
  `artifacts/modelado/reports/analisis_errores.md` (`src/modelado/analisis_errores.py`, ejecutado
  dentro de `train`, nunca sobre `test`). Si se promete un horario, comunicarlo como rango, no
  como cifra exacta.
- **`segment_distance_km` es distancia en línea recta** (Haversine), no por carretera.

## Rendimiento

Ver [model_card.md](model_card.md) (generado por `uv run sdai-train`, con la métrica sobre el 20%
de rutas de test — nunca visto en el ajuste). Si al probarlo el error medio observado se aleja
mucho de esas cifras, algo está mal montado (variables en otro orden, categóricas mal pasadas,
nulos sin imputar) — no es normal que el modelo rinda peor en producción que en la evaluación ya
verificada.

## Regenerar el modelo desde cero

```text
uv run sdai-train
```

Es el único comando que reentrena y sobrescribe `artifacts/modelo/`. Los scripts de
`src/modelado/bakeoff/` y `src/modelado/analisis_errores.py` solo miden, no dejan modelo guardado
para servir.
