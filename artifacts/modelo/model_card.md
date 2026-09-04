# Ficha del modelo: tiempo de tramo de reparto

Generada automaticamente por `entrenamiento_final.py` el 2026-09-04
(se escribe a la vez en `artifacts/modelo/model_card.md` y aqui: mismo contenido,
sin mantenimiento manual duplicado). Arquitectura seleccionada en el bake-off, ver
[bakeoff_resumen.md](bakeoff_resumen.md); contrato de servicio para el agente en
[instrucciones_de_servicio.md](instrucciones_de_servicio.md).

## Que predice

`travel_time_seconds`: segundos entre dos paradas consecutivas de una ruta de
reparto. Sumando los tramos de una ruta se obtiene la hora estimada de llegada a
cualquiera de sus paradas.

## Arquitectura

Hibrido de dos especialistas, seleccionados por `is_depot_segment`:

```text
reparto   CatBoost, 1865 arboles, con to_zone_id nativa
almacen   LightGBM, 532 arboles
```

La division existe porque un unico arbol conjunto reparte su capacidad donde
esta la masa de datos y desatiende el tramo de almacen (menos del 1% de los
tramos totales).

## Datos de entrenamiento y evaluacion

```text
Split fijo por route_id, hecho una sola vez antes del bake-off:
train   4,890 rutas (80%)
test    1,222 rutas (20%, nunca visto en el ajuste)

El test set es el mismo dato que sirve de banco de produccion simulada del agente
(data/produccion_simulada/datos_produccion.parquet).
```

## Rendimiento (evaluado UNA VEZ sobre el 20% de test reservado)

```text
global    MAE 0.3683 min   R2 0.9398
reparto   MAE 0.3518 min   R2 0.6487   (n=179,875)
almacen   MAE 2.7868 min   R2 0.8728   (n=1,222)
```

## Variables mas influyentes (TreeSHAP)

`segment_distance_km` domina en las dos poblaciones. `to_zone_id` aporta
5.3% de la magnitud media de SHAP en reparto; es la razon de
usar CatBoost alli en vez de LightGBM (unica de las dos librerias que trata esa
variable de forma nativa sin codificacion manual). Detalle completo en
`artifacts/modelo/informe_shap.md`.

## Limitaciones conocidas

- El objetivo son medias historicas por par de coordenadas, no tiempos medidos
  ese dia concreto: el modelo predice la duracion tipica de una ruta.
- La meteorologia no se incluye (medida aparte, no mejora el error).
- Sin datos de trafico en tiempo real ni festivos en el periodo cubierto.
- Cinco semanas de verano de 2018: no se puede evaluar estacionalidad.
- **`reparto_catboost.cbm` pesa varios cientos de MB.** Cargarlo una sola vez
  por proceso, nunca por peticion.

## Ejemplos de explicacion local

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

## Como usarlo

```text
1. Cargar reparto_catboost.cbm y almacen_lightgbm.txt (ver manifiesto.json).
2. Construir las variables de manifiesto.json para el tramo a predecir.
3. Si is_depot_segment == 1, usar el modelo de almacen; si no, el de reparto.
4. Recortar la prediccion a un minimo de 0 segundos.
```

## Reentrenamiento

Ejecutar de nuevo `uv run sdai-train` tras regenerar Gold (`uv run sdai-pipeline`).
No editar los ficheros de `artifacts/modelo/` a mano.
