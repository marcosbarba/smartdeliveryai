# Contrato Datos Gold

Dimensiones de esta ejecucion: **6,112 filas y 60 columnas**.

Este documento explica el dataset a nivel de ruta.

## Ojo: esta no es la tabla del modelo

El objetivo del proyecto es predecir a que hora llega un paquete, y eso se modela **por tramo**,
en la otra tabla de Gold:

```text
data/Gold/Dataset Tramos Ruta/Dataset Tramos Ruta.parquet
```

Esta tabla de rutas sirve para planificacion, analisis agregado y contexto: una fila resume
toda una ruta. `route_score` es una etiqueta de calidad de la secuencia que asigno Amazon y que
**ya no es el objetivo del proyecto**.

## Archivo principal para trabajar en local

```text
data/Gold/Dataset Rutas Enriquecidas/Dataset Rutas Enriquecidas.csv
```

## Archivo equivalente Big Data

```text
data/Gold/Dataset Rutas Enriquecidas/Dataset Rutas Enriquecidas.parquet
```

## Motor de construccion

Gold se construye con PySpark a partir de las tablas Silver.

## Granularidad

Cada fila representa una ruta de reparto de Amazon.

## Uso previsto

- EDA principal del TFM.
- Entrenamiento de modelos.
- Analisis para dashboard.
- Base para agentes o asistentes de consulta.

## Variables principales

- `route_id`: identificador unico de ruta.
- `station_code`: estacion logistica Amazon.
- `route_date`: fecha de la ruta.
- `route_score`: etiqueta original de dificultad/calidad de ruta.
- `route_score_numeric`: version numerica de `route_score`.
- `invalid_sequence_score`: puntuacion asociada a secuencia invalida.
- `total_stops`: numero total de paradas informado en rutas.
- `dropoff_stops`: numero de paradas de entrega.
- `package_count`: numero de paquetes de la ruta.
- `total_package_volume_cm3`: volumen total de paquetes.
- `total_planned_service_time_seconds`: tiempo total planificado de servicio.
- `unique_zones`: numero de zonas distintas en la ruta.
- `temperature_2m_mean_c`: temperatura media diaria.
- `precipitation_sum_mm`: precipitacion diaria.
- `rain_sum_mm`: lluvia diaria.
- `wind_speed_10m_max_kmh`: viento maximo diario.
- `is_weekend`: indica si la fecha cae en fin de semana.
- `is_us_public_holiday`: indica si la fecha es festivo oficial de EEUU.
- `delivered_package_count`: paquetes entregados con exito.
- `delivery_attempted_count`: intentos de entrega fallidos, es decir el repartidor paso y no habia nadie. Son el 0,85% de los paquetes. Un mismo paquete puede aparecer dos veces en la ruta: un intento fallido y una segunda pasada.
- `rejected_package_count`: paquetes rechazados por el cliente.
- `other_scan_status_count`: cajon de sastre. Deberia valer 0 siempre; si no, ha aparecido un estado de escaneo nuevo sin contemplar.
- `departure_time_utc`: hora de salida del almacen, en texto `HH:mm:ss`. Para operar con fecha y hora juntas hay que usar `departure_timestamp_utc`.
- `total_haversine_distance_km`: distancia total en linea recta recorrida siguiendo la secuencia real de entrega, desde la estacion hasta la ultima parada.
- `depot_to_first_stop_distance_km`: distancia del tramo que sale de la estacion hacia la primera parada de entrega, es decir el desplazamiento del almacen a la zona de reparto.
- `delivery_haversine_distance_km`: distancia recorrida solo entre paradas de entrega, sin contar el tramo inicial desde la estacion. Es la parte que corresponde al reparto en si.

## Variables de tiempo

- `total_travel_time_seconds`: suma de los tiempos reales de conduccion entre paradas consecutivas, siguiendo la secuencia real de entrega.
- `route_duration_seconds`: duracion total de la ruta, es decir conduccion mas tiempo de servicio en las paradas.
- `route_duration_hours`: lo mismo en horas, por comodidad de lectura.

Estas columnas solo tienen valor si se ha descargado `travel_times.json`, que es una descarga
opcional de 1,8 GB. Si no esta, salen a nulo y el resto de Gold no cambia.

## AVISO IMPORTANTE: fuga de datos

`route_duration_seconds` esta pensada como **variable objetivo** para predecir cuanto va a durar
una ruta. Eso obliga a tener cuidado con que se usa como predictor:

```text
route_duration_seconds = total_travel_time_seconds + total_planned_service_time_seconds
```

- **No usar `total_travel_time_seconds` como predictor de la duracion.** Es un sumando del
  objetivo: el modelo daria un acierto casi perfecto y no serviria para nada con datos nuevos.
- Tampoco derivar velocidades a partir de esos tiempos y usarlas como variable. Como la
  distancia ya esta en el dataset, velocidad y distancia reconstruyen el tiempo por division.
- Si se quiere una variable de velocidad, hay que calcularla con el historico de **otras**
  rutas (por estacion, zona u hora), nunca con la ruta que se esta prediciendo, y hacerlo
  dentro de la validacion cruzada. Por eso no viene ya calculada en Gold: hacerlo aqui
  contaminaria la particion entre entrenamiento y prueba.

Variables si legitimas como predictores, porque se conocen antes de que la ruta salga: numero
de paradas y paquetes, volumen, zonas, distancias Haversine, estacion, dia, hora de salida,
meteorologia y tiempo de servicio planificado.

## Resto de columnas

Identificacion y capacidad:

- `executor_capacity_cm3`: capacidad volumetrica del vehiculo asignado.
- `departure_timestamp_utc`: fecha y hora de salida juntas. Es la columna a usar para operar con fechas.

Composicion de la ruta:

- `station_stops`: paradas de tipo estacion. Vale 1 en todas las rutas: cada una sale de un unico almacen.
- `stop_rows`: paradas contadas en la tabla de paradas. Coincide con `total_stops`.
- `sequence_stop_count`: paradas presentes en la secuencia real. Coincide tambien con `total_stops`.
- `max_actual_sequence_position`: ultima posicion de la secuencia, es decir `total_stops` menos uno.
- `lat_range`, `lng_range`: amplitud en grados entre la parada mas al norte y la mas al sur, y lo mismo en longitud. Describen el rectangulo que ocupa la ruta, no el camino recorrido.

Cuidado al modelar: `total_stops`, `dropoff_stops`, `stop_rows`, `sequence_stop_count` y
`max_actual_sequence_position` miden **lo mismo** con desplazamientos constantes y correlacionan
a 1,0 entre si. Hay que quedarse con una sola.

Paquetes:

- `packages_with_time_window`: paquetes con ventana horaria comprometida.
- `packages_with_time_window_ratio`: proporcion de esos paquetes sobre el total de la ruta. Media 0,08.
- `mean_planned_service_time_seconds`: tiempo de servicio medio por paquete.
- `mean_package_volume_cm3`, `max_package_volume_cm3`: volumen medio y maximo de los paquetes.

Meteorologia del dia, por estacion:

- `temperature_2m_max_c`, `temperature_2m_min_c`: temperatura maxima y minima.
- `apparent_temperature_max_c`, `apparent_temperature_min_c`: sensacion termica maxima y minima.
- `snowfall_sum_cm`: nieve acumulada. Vale 0 en todo el dataset, que cubre julio y agosto.
- `weather_code`: codigo de condicion meteorologica de Open-Meteo.
- `wind_gusts_10m_max_kmh`: racha maxima de viento.

Calendario:

- `year`, `month`, `day`: componentes de la fecha. `year` vale 2018 siempre y `month` solo 7 u 8.
- `weekday`: dia de la semana en numero, de 0 para lunes a 6 para domingo.
- `weekday_name`: nombre del dia en ingles.

## Variables derivadas utiles para EDA

- `packages_per_dropoff_stop`: paquetes medios por parada de entrega.
- `volume_per_package_cm3`: volumen medio por paquete.
- `planned_service_time_per_package_seconds`: tiempo planificado medio por paquete.
- `mean_delivery_segment_distance_km`: distancia media entre dos paradas de entrega consecutivas. Mide lo dispersas que estan las entregas dentro de la zona.
- `has_rain`: indica si hubo lluvia ese dia.
- `has_precipitation`: indica si hubo precipitacion.

## Aviso sobre distancia Haversine

Estas variables son distancia en **linea recta** entre coordenadas, no distancia real conducida por carretera. Sirven como proxy de la dispersion geografica de la ruta, no como medida exacta de kilometros recorridos. Una mejora futura del proyecto seria sustituirlas por routing real (por ejemplo con OSRM) si el tiempo del TFM lo permite.

Por que se separa el tramo de la estacion: en este dataset la estacion ocupa siempre la posicion 0 de la secuencia, y ese primer tramo mide 17,75 km de media frente a 0,195 km entre paradas de entrega consecutivas, es decir el 38,4% de los 46,24 km medios por ruta. Son dos fenomenos distintos (lo lejos que esta el almacen de la zona, frente a lo denso que es el reparto dentro de la zona) y mezclarlos en una sola cifra dificulta interpretar el modelo.

Relaciones a tener en cuenta al modelar, para no introducir variables colineales sin darse cuenta:

```text
total_haversine_distance_km = depot_to_first_stop_distance_km + delivery_haversine_distance_km
mean_delivery_segment_distance_km = delivery_haversine_distance_km / (dropoff_stops - 1)
```

La distancia **no incluye el regreso a la estacion**: en las 6112 rutas la ultima parada de la secuencia es siempre de entrega, no la estacion, por lo que el dataset original no informa del trayecto de vuelta.

## Reglas

Gold se construye desde Silver. No debe leer directamente Bronze salvo que haya que reconstruir todo el pipeline.
