# Contrato Datos Gold: tramos de ruta

Tabla para predecir cuanto se tarda en ir de una parada a la siguiente. Sumando los tramos de
una ruta se obtiene la hora estimada de llegada a cualquiera de sus paradas, que es lo que
responde el agente cuando un cliente pregunta por su paquete.

## Archivos

```text
data/Gold/Dataset Tramos Ruta/Dataset Tramos Ruta.csv
data/Gold/Dataset Tramos Ruta/Dataset Tramos Ruta.parquet
```

## Granularidad

Una fila por tramo entre dos paradas consecutivas de la secuencia real de entrega.
898.415 tramos, que salen de 904.527 paradas menos 6.112 rutas.

## Variable objetivo

- `travel_time_seconds`: segundos en ir de `from_stop_id` a `to_stop_id`.

## Variables del tramo

- `segment_distance_km`: distancia en linea recta entre las dos paradas (Haversine, no por carretera).
- `is_depot_segment`: 1 si el tramo sale de la estacion. Es el desplazamiento al barrio, de media 30 minutos frente a 1 minuto entre entregas.
- `from_zone_id`, `to_zone_id`, `same_zone`: zona de planificacion de origen y destino, y si coinciden.
- `from_lat`, `from_lng`, `to_lat`, `to_lng`: coordenadas.
- `segment_position`: numero de tramo dentro de la ruta.
- `segment_ratio`: proporcion de la ruta recorrida, de 0 a 1.
- `cumulative_distance_km`: kilometros acumulados desde el almacen hasta el inicio del tramo.

## Variables de la parada de destino

Todas se conocen al planificar, porque salen del manifiesto de carga:

- `packages_at_destination`: paquetes a entregar alli.
- `service_time_at_destination_seconds`: tiempo de servicio planificado.
- `volume_at_destination_cm3`: volumen total.
- `packages_with_window_at_destination`: paquetes con ventana horaria comprometida.

## Variables de contexto de la ruta

`station_code`, `route_date`, `departure_hour`, `route_total_stops`, `route_package_count`,
`weekday`, `is_weekend`, `temperature_2m_mean_c`, `rain_sum_mm`, `has_rain`,
`wind_speed_10m_max_kmh`.

## AVISOS IMPORTANTES

**Fuga de datos.** Todas las variables de esta tabla se conocen antes de que la ruta salga. No
anadir nada derivado de `travel_time_seconds`: en particular, **no calcular velocidades**
(velocidad = distancia / tiempo, y como la distancia esta aqui, velocidad y distancia
reconstruyen el objetivo por division). Si se quiere una variable de velocidad, debe calcularse
como historico de **otras** rutas y dentro de la validacion cruzada.

**Particion por ruta.** Los tramos de una misma ruta no son independientes. Hay que partir con
`GroupKFold` agrupando por `route_id`. Una particion aleatoria por filas dejaria tramos de la
misma ruta en entrenamiento y en prueba, y el resultado seria enganosamente bueno.

**Baseline obligatorio.** La posicion en la ruta explica gran parte del acumulado de forma
trivial. Hay que comparar siempre contra un baseline sencillo, por ejemplo tiempo medio por
kilometro, para saber si el modelo aporta algo de verdad.

**La meteorologia es enganosa.** La lluvia correlaciona +0,242 con el tiempo de conduccion,
pero es una correlacion espuria: las ciudades lluviosas del dataset son las que tienen rutas
mas largas. Controlando por estacion la correlacion cae a -0,017. Si se usa el clima **hay que
incluir `station_code`**, o el modelo aprendera la relacion falsa y las explicaciones al
usuario final seran incorrectas.

## Hace falta cruzar con la tabla de rutas?

**No para empezar.** Esta tabla ya trae el contexto de la ruta repetido en cada tramo
(`station_code`, `departure_hour`, `route_total_stops`, `route_package_count`, `weekday`,
`is_weekend` y las variables de meteorologia), que es la desnormalizacion habitual para
entrenar: una tabla plana con una fila por ejemplo.

Si se quiere enriquecer, se cruza por `route_id` con `Dataset Rutas Enriquecidas`. Variables de
esa tabla que aportarian y son legitimas, porque se conocen antes de que la ruta salga:

```text
unique_zones                      cuantas zonas cubre la ruta
lat_range, lng_range              dispersion geografica del reparto
executor_capacity_cm3             tamano del vehiculo
total_package_volume_cm3          carga total
packages_with_time_window_ratio   proporcion de entregas con hora comprometida
```

## Variables de la tabla de rutas que NO se pueden usar

Son resultados de la ruta, no datos de planificacion. Usarlas seria fuga de datos:

```text
total_travel_time_seconds          es la suma del objetivo
route_duration_seconds             contiene el objetivo
route_duration_hours               contiene el objetivo
mean_delivery_segment_distance_km  derivada del recorrido real
delivery_attempted_count           solo se sabe al terminar la ruta
route_score, route_score_numeric   solo se sabe al terminar la ruta
invalid_sequence_score             solo se sabe al terminar la ruta
```

Ademas, `total_stops`, `dropoff_stops`, `stop_rows`, `sequence_stop_count` y
`max_actual_sequence_position` son la misma magnitud con desplazamientos constantes, y
`route_total_stops` ya esta en esta tabla. Coger mas de una solo anade colinealidad.

## Reglas

Esta tabla se construye desde Silver, nunca desde Bronze. Se regenera con
`construir_gold_tramos.py`: no editarla a mano.
