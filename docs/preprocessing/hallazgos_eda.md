# Hallazgos del EDA que condicionan el modelado

Resumen de "Hallazgos EDA.md" (generado por `src/preprocessing/eda/eda_tiempos.py`, detalle
completo con cifras y gráficos en `artifacts/eda/` y en `notebooks/01_eda/eda_tiempos.ipynb`, la
versión narrativa ya ejecutada). Este documento recoge solo las conclusiones que **cambiaron**
una decisión de modelado — no un resumen genérico del análisis.

## Los datos

```text
898.415 tramos, 6.112 rutas, 5 áreas metropolitanas de EEUU
periodo: 2018-07-19 a 2018-08-26 (cinco semanas de verano)
31 variables, 37.744 celdas nulas, 0 duplicados
```

## Los seis hallazgos

**1. El objetivo son dos poblaciones, no una.** El tramo que sale del almacén dura de media ~30
minutos; los saltos entre entregas, ~1 minuto. Cualquier modelo o baseline que no distinga
`is_depot_segment` sirve mal a las dos realidades a la vez — condicionó directamente la
arquitectura híbrida de dos especialistas (ver `docs/modelado/`).

**2. El baseline intuitivo falla; el correcto es exigente.** Dividir distancia entre velocidad
media da **R² negativo** (peor que predecir la media): una única velocidad no sirve a la vez a
~35 km/h (reparto) y ~11 km/h (patrón del tramo de almacén). El baseline honesto — una recta por
población, ajustada solo en entrenamiento — da **R² 0,919, MAE 0,45 min**. Cualquier modelo
posterior se mide contra esa cifra, no contra la intuición.

**3. La distancia solo explica el 58% dentro de reparto**, no el 87% que sugiere la correlación
global (0,935) — esa cifra está inflada por la existencia de las dos poblaciones. El 42% restante
es lo que el modelo tiene que aprender de otras variables.

**4. `same_zone` es la mejor señal después de la distancia.** Compartir zona de planificación con
la parada anterior reduce el tiempo del tramo de forma consistente y es la segunda variable más
influyente en SHAP (ver `docs/modelado/model_card.md`).

**5. El clima no aporta, y engaña si no se controla por estación.** La lluvia correlaciona
+0,037 con el tiempo de conducción en bruto, y esa correlación **desaparece al controlar por
`station_code`** (+0,002 de media dentro de cada estación): las ciudades lluviosas del dataset
son las que tienen rutas más largas, así que la lluvia actuaba como disfraz de la variable
"ciudad" — paradoja de Simpson. El contraste estadístico lo confirma: diferencia significativa
(p = 3,57e-289 sobre 900.000 filas) pero con un tamaño del efecto (d de Cohen) de 0,077, es
decir, despreciable. Medido en el bake-off (`lightgbm_unico`): incluir variables de clima no
mejora el MAE. Se documenta como resultado negativo argumentado, no se descarta la variable sin
medir — la hipótesis inicial del proyecto era que el clima sí influiría. (Cifras recalculadas
sobre el Gold de este repositorio — `artifacts/eda/hallazgos_eda.md`, sección 12 —, distintas de
las del proyecto original: el nº de estaciones y el corte de datos no son exactamente los mismos.)

**6. No recortar atípicos ni tramos de 0 segundos: son casos reales.** Un tramo de 0 segundos es
una entrega en la misma coordenada (dos pisos de un mismo edificio), no un error de captura.
Recortarlos sesgaría el modelo contra un patrón que existe de verdad en la operación.

## Consecuencias directas para el modelado

- Partición siempre `GroupKFold` por `route_id` (los tramos de una ruta no son independientes).
- Métrica principal: MAE en minutos (interpretable ante un tribunal o un cliente).
- Arquitectura híbrida por `is_depot_segment`, no una variable más en un modelo conjunto (medido:
  empeora el R² aunque mejore el MAE — ver `docs/modelado/bakeoff_resumen.md`).
