# Ficha del modelo: tiempo de tramo de reparto

Generada automáticamente por `entrenamiento_final.py` el 2026-09-15
(se escribe a la vez en `artifacts/modelo/model_card.md` y aquí: mismo contenido,
sin mantenimiento manual duplicado). Arquitectura seleccionada en el bake-off — ver
[bakeoff_resumen.md](bakeoff_resumen.md); contrato de servicio para el agente en
[instrucciones_de_servicio.md](instrucciones_de_servicio.md).

## Qué predice

`travel_time_seconds`: segundos entre dos paradas consecutivas de una ruta de
reparto. Sumando los tramos de una ruta se obtiene la hora estimada de llegada a
cualquiera de sus paradas.

## Arquitectura

Híbrido de dos especialistas, seleccionados por `is_depot_segment`:

| Especialista | Modelo | Árboles | Categóricas nativas |
|---|---|---:|---|
| Reparto | CatBoost | 1865 | `station_code`, `to_zone_id` |
| Almacén | LightGBM | 532 | `station_code` |

La división existe porque un único árbol conjunto reparte su capacidad donde
está la masa de datos y desatiende el tramo de almacén (menos del 1% de los
tramos totales).

## Datos de entrenamiento y evaluación

Split fijo por `route_id`, hecho una sola vez antes del bake-off. El test set es
el mismo dato que sirve de banco de producción simulada del agente
(`data/produccion_simulada/datos_produccion.parquet`).

| Conjunto | Rutas | % |
|---|---:|---:|
| `train` | 4,890 | 80% |
| `test` (held-out, nunca visto en el ajuste) | 1,222 | 20% |

## Rendimiento (evaluado UNA VEZ sobre el 20% de test reservado)

| Población | MAE (min) | R² | n |
|---|---:|---:|---:|
| Global | 0.3683 | 0.9398 | — |
| Reparto | 0.3518 | 0.6487 | 179,875 |
| Almacén | 2.7868 | 0.8728 | 1,222 |

## Variables más influyentes (TreeSHAP)

`segment_distance_km` domina en las dos poblaciones. `to_zone_id` aporta
5.3% de la magnitud media de SHAP en reparto; es la razón de
usar CatBoost allí en vez de LightGBM (única de las dos librerías que trata esa
variable de forma nativa sin codificación manual). Detalle completo en
`artifacts/modelo/informe_shap.md`.

## Limitaciones conocidas

- El objetivo son medias históricas por par de coordenadas, no tiempos medidos
  ese día concreto: el modelo predice la duración típica de una ruta.
- La meteorología no se incluye (medida aparte, no mejora el error).
- Sin datos de tráfico en tiempo real ni festivos en el periodo cubierto.
- Cinco semanas de verano de 2018: no se puede evaluar estacionalidad.
- **`reparto_catboost.cbm` pesa varios cientos de MB.** Cargarlo una sola vez
  por proceso, nunca por petición.

## Ejemplos de explicación local

### Ejemplo: tramo de reparto tipico

```text
tiempo real:        0.77 min
prediccion:         0.45 min
valor base (media): 0.92 min

mayores contribuciones (segundos):
  segment_distance_km                       -15.9 s   (valor: 0.091973)
  station_code                               -6.2 s   (valor: DLA4)
  same_zone                                  -2.4 s   (valor: 1.0)
  service_time_at_destination_seconds        -1.9 s   (valor: 20.0)
  segment_position                           +0.7 s   (valor: 142)
  to_zone_id                                 -0.7 s   (valor: B-11.1C)
```

### Ejemplo: tramo de reparto largo (atipico)

```text
tiempo real:        9.63 min
prediccion:         1.78 min
valor base (media): 0.92 min

mayores contribuciones (segundos):
  segment_distance_km                       +24.5 s   (valor: 0.270769)
  station_code                              +14.8 s   (valor: DLA7)
  to_zone_id                                +12.3 s   (valor: C-15.2B)
  same_zone                                  -3.5 s   (valor: 1.0)
  service_time_at_destination_seconds        +2.3 s   (valor: 125.0)
  cumulative_distance_km                     +1.6 s   (valor: 23.970963)
```

### Ejemplo: tramo de almacen (mediana)

```text
tiempo real:        29.91 min
prediccion:         30.24 min
valor base (media): 30.00 min

mayores contribuciones (segundos):
  segment_distance_km                       +95.5 s   (valor: 18.671928)
  station_code                              -62.2 s   (valor: DLA5)
  volume_at_destination_cm3                 -24.6 s   (valor: 2135.0)
  weekday                                   +12.1 s   (valor: 6)
  route_total_stops                          -6.5 s   (valor: 156)
  departure_hour                             -6.1 s   (valor: 15)
```

## Cómo usarlo

1. Cargar `reparto_catboost.cbm` y `almacen_lightgbm.txt` (ver `manifiesto.json`).
2. Construir las variables de `manifiesto.json` para el tramo a predecir.
3. Si `is_depot_segment == 1`, usar el modelo de almacén; si no, el de reparto.
4. Recortar la predicción a un mínimo de 0 segundos.

## Reentrenamiento

Ejecutar de nuevo `uv run sdai-train` tras regenerar Gold (`uv run sdai-pipeline`).
No editar los ficheros de `artifacts/modelo/` a mano.
