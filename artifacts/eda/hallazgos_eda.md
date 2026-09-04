# Hallazgos del analisis exploratorio

Generado por `eda_tiempos.py` el 2026-09-04.

## 0. Resumen ejecutivo

Analizados **898,415 tramos** de **6,112 rutas**, con 31 variables.
El objetivo es `travel_time_seconds`: cuanto se tarda en ir de una parada a la siguiente.

Cinco conclusiones que condicionan el modelado:

1. **El objetivo son dos poblaciones**, no una. El tramo que sale del almacen dura
   29.5 veces mas que un salto entre entregas.
2. **El baseline intuitivo es malo**: dividir la distancia entre una velocidad media da un R2 de
   -1.0344, peor que predecir la media. La referencia real es 0.9193.
3. **La distancia no lo explica todo**: dentro del reparto solo cubre el 58% de la variabilidad.
4. **`same_zone` es la mejor pista** despues de la distancia.
5. **El clima no aporta y ademas engana** por confusion geografica.

## 1. Vision general

```text
tramos                898,415
rutas                   6,112
variables                  31
memoria                 505.4 MB
periodo             2018-07-19 a 2018-08-26
estaciones                 17
```

## 2. Calidad de los datos

```text
celdas nulas                    37,744  (0.1355% del total)
filas duplicadas completas           0
duplicados por clave                 0
columnas sin variacion               0
```

Columnas constantes, que deben excluirse del modelo:

- ninguna

Los nulos se concentran en las variables de zona, y tienen explicacion: el almacen no pertenece
a ninguna zona de reparto, asi que en el tramo que sale de el no hay zona de origen ni se puede
comparar con la de destino.

No hay valores negativos en ninguna variable ni filas duplicadas.

Figura: `artifacts/eda/figuras/01 Calidad de los datos.png`

## 3. La variable objetivo

```text
                    tramos      media    mediana       p95      max   asimetria
reparto            892,303      1.02       0.76      2.83     39.1        3.60
almacen -> zona      6,112     30.19      29.91     48.54     95.8        0.33
```

**Este es el hallazgo principal.** El trayecto inicial es 29.5 veces mas largo: es un viaje
por carretera hasta el barrio, mientras que el resto son saltos de portal a portal. Mezclarlos en
un unico modelo sin distinguirlos obliga al modelo a servir a dos realidades incompatibles.

La distribucion del reparto esta muy sesgada a la derecha (asimetria 3.597, curtosis
30.129). El test de Shapiro-Wilk sobre una muestra rechaza la normalidad
(p = 2.51e-67), y aplicando logaritmo la asimetria baja a -0.461.

Eso importa para elegir modelo: los lineales sufren con esta forma, los de arboles no.

Figura: `artifacts/eda/figuras/02 Variable objetivo.png`

## 4. Variables numericas

Variables con asimetria fuerte, por encima de 2 en valor absoluto:

- `travel_time_seconds`
- `segment_distance_km`
- `is_depot_segment`
- `same_zone`
- `packages_at_destination`
- `service_time_at_destination_seconds`
- `volume_at_destination_cm3`
- `packages_with_window_at_destination`
- `rain_sum_mm`

Figura: `artifacts/eda/figuras/03 Distribuciones numericas.png`

## 5. Variables categoricas

```text
estaciones                      17
zonas de destino             8,962
zonas con un solo tramo         96
```

Las tres estaciones mas frecuentes concentran el 40.32% de los tramos, asi que hay
desequilibrio geografico que conviene tener presente al validar.

`to_zone_id` tiene cardinalidad muy alta. Para usarla no sirve una codificacion one-hot: hay que
recurrir a modelos que traten categoricas de forma nativa, como CatBoost, o a codificacion por
historico calculada fuera de muestra.

Figura: `artifacts/eda/figuras/04 Categoricas.png`

## 6. Valores atipicos

- `travel_time_seconds`: 48,830 atipicos por el criterio del rango intercuartilico (5.47%), maximo 2347.3
- `segment_distance_km`: 64,224 atipicos por el criterio del rango intercuartilico (7.2%), maximo 13.821
- `packages_at_destination`: 52,154 atipicos por el criterio del rango intercuartilico (5.84%), maximo 78.0
- `service_time_at_destination_seconds`: 72,617 atipicos por el criterio del rango intercuartilico (8.14%), maximo 8007.0

Hay 1,711 tramos de cero segundos (0.192%). **No son errores**: son
entregas consecutivas en la misma coordenada, como dos pisos del mismo edificio, donde el
repartidor no se desplaza.

Los atipicos del tiempo tampoco parecen errores, sino tramos genuinamente largos. La
recomendacion es **no recortarlos**: forman parte del fenomeno que se quiere predecir.

Figura: `artifacts/eda/figuras/05 Valores atipicos.png`

## 7. Correlaciones y multicolinealidad

Correlacion de Pearson con el objetivo:

- `segment_distance_km`: 0.935
- `is_depot_segment`: 0.8748
- `same_zone`: -0.3603
- `cumulative_distance_km`: -0.1476
- `segment_ratio`: -0.1253
- `segment_position`: -0.1174
- `service_time_at_destination_seconds`: 0.0896
- `packages_with_window_at_destination`: 0.0519

Pares de variables muy correlacionadas entre si, por encima de 0,7:

- `segment_position` y `segment_ratio`: 0.937
- `travel_time_seconds` y `segment_distance_km`: 0.935
- `segment_distance_km` y `is_depot_segment`: 0.8831
- `travel_time_seconds` y `is_depot_segment`: 0.8748
- `weekday` y `is_weekend`: 0.7563

Factor de inflacion de la varianza por encima del umbral habitual de 5: `segment_position`, `segment_ratio`.

Conviene no meter juntas variables que miden lo mismo. En particular, la posicion dentro de la
ruta esta representada por varias columnas equivalentes.

Figuras: `artifacts/eda/figuras/06 Matriz de correlaciones.png`, `artifacts/eda/figuras/07 Correlacion con el objetivo.png`

## 8. Distancia frente a tiempo

```text
                 correlacion   R2 con solo distancia   velocidad implicita
todos               0.935              0.874                  15.4 km/h
reparto             0.764              0.584                  11.4 km/h
almacen -> zona     0.870              0.757                  35.3 km/h
```

La correlacion global de 0.935 es enganosa: procede en buena parte de que existen dos grupos
separados. **Dentro del reparto baja a 0.764** y la distancia explica solo el
58% de la variabilidad. El 42% restante es lo que el modelo tiene que aprender.

Las velocidades son en linea recta: por carretera se recorre mas, asi que la velocidad real es
mayor. Sirven para comparar zonas y horas, no como medida absoluta.

Figura: `artifacts/eda/figuras/08 Distancia frente a tiempo.png`

## 9. Baselines de referencia

```text
distancia / velocidad media      R2 =  -1.034
una recta para todo              R2 =   0.874
una recta por poblacion          R2 =   0.919   MAE 0.45 min
```

El baseline mas intuitivo da un **R2 negativo**: es peor que predecir siempre la media, porque una
unica velocidad no puede describir dos poblaciones que circulan a 35 y a 11 km/h.

**La referencia a batir es la ultima**: R2 0.919 y 0.45 minutos de error medio.

Dos aclaraciones:

- **Estimar una velocidad media no es fuga de datos.** La regla es no usar el objetivo de una
  fila para predecir esa misma fila. Una media del conjunto es un parametro agregado, igual que
  la pendiente de una regresion. Lo prohibido seria dar al modelo la velocidad de cada tramo.
- Estas cifras se calculan sobre todos los datos. Al modelar hay que reajustar el baseline solo
  con el entrenamiento y medirlo en la prueba.

Figura: `artifacts/eda/figuras/09 Baselines.png`

## 10. Que explica lo que la distancia no explica

Correlacion con el residuo de la regresion sobre la distancia, dentro del reparto:

- `same_zone`: -0.1877
- `service_time_at_destination_seconds`: 0.084
- `packages_with_window_at_destination`: 0.0645
- `route_total_stops`: -0.0637
- `cumulative_distance_km`: -0.0472
- `packages_at_destination`: 0.0425

La mas informativa es `same_zone`. Moverse dentro de la misma zona de
planificacion sale mas rapido de lo que la distancia haria pensar:

```text
zonas distintas  0.88 min   0.165 km   759,904 tramos
zonas distintas  1.95 min   0.384 km   119,811 tramos
```

El 86.38% de los tramos ocurre dentro de una misma zona. Tiene sentido operativo: las
zonas agrupan calles proximas y bien conectadas.

Figuras: `artifacts/eda/figuras/10 Residuo tras la distancia.png`, `artifacts/eda/figuras/11 Efecto de la zona.png`

## 11. Contexto geografico y temporal

Minutos medios por tramo de reparto, por area metropolitana:

- LA: 0.972 min
- SE: 1.01 min
- CH: 1.054 min
- AU: 1.114 min
- BO: 1.141 min

## 12. El clima: paradoja de Simpson

```text
correlacion lluvia-tiempo, global              +0.0369
correlacion media dentro de cada estacion      +0.0017
```

La correlacion global sugiere que la lluvia ralentiza el reparto, pero **desaparece al controlar
por estacion**. Las ciudades lluviosas del dataset son las que tienen rutas mas largas, de modo
que la lluvia estaba actuando como disfraz de la variable "ciudad".

El contraste lo confirma: la diferencia es estadisticamente significativa
(p = 3.57e-289) pero con un tamano del efecto de 0.0771, es decir
**efecto despreciable**. Con 900.000 filas casi cualquier diferencia sale significativa,
por eso hay que mirar el tamano del efecto y no solo la p.

Si se entrena con meteorologia **sin incluir `station_code`**, el modelo aprendera la relacion
falsa y el agente acabara diciendo a un cliente que su paquete llega tarde por la lluvia, cuando
en realidad es por la ciudad en la que vive.

Figuras: `artifacts/eda/figuras/12 Contexto geografico y temporal.png`, `artifacts/eda/figuras/13 Paradoja de Simpson del clima.png`

## 13. Contrastes estadisticos

```text
deposito vs reparto     Mann-Whitney U     p = 0.0
misma zona              Mann-Whitney U     p = 0.0   d de Cohen = -0.8403
lluvia                  Mann-Whitney U     p = 3.57e-289   d de Cohen = 0.0771
estaciones              Kruskal-Wallis     p = 0.0   (17 grupos)
```

Se usan pruebas no parametricas porque la distribucion del objetivo dista mucho de la normal.

## 14. La ruta completa

```text
duracion media    7.51 h   (de 4.27 a 10.6 h)
conducir          39.9%
entregar          60.1%
```

El grueso de la jornada se va **entregando, no conduciendo**. Como el tiempo de servicio viene
planificado en el manifiesto y se conoce antes de salir, el modelo solo tiene que estimar el
39.9% restante.

Figura: `artifacts/eda/figuras/14 Duracion de ruta.png`

---

# Parte B: la tabla de rutas

Hasta aqui el analisis ha ido sobre los tramos, que es lo que alimenta el modelo. Esta segunda
parte describe la tabla por ruta, que sirve para planificacion, contexto y analisis agregado.

## B.1 Calidad

```text
rutas                      6,112
columnas                      60
celdas nulas                   0
identificadores repetidos      0
```

Columnas sin variacion, que deben excluirse:

- `station_stops`
- `other_scan_status_count`
- `snowfall_sum_cm`
- `year`
- `is_us_public_holiday`

Pares de columnas con correlacion practicamente perfecta, es decir que llevan la misma
informacion aunque difieran en una constante:

- `total_stops == dropoff_stops`
- `total_stops == stop_rows`
- `total_stops == sequence_stop_count`
- `total_stops == max_actual_sequence_position`
- `dropoff_stops == stop_rows`
- `dropoff_stops == sequence_stop_count`
- `dropoff_stops == max_actual_sequence_position`
- `stop_rows == sequence_stop_count`
- `stop_rows == max_actual_sequence_position`
- `mean_planned_service_time_seconds == planned_service_time_per_package_seconds`

Todas las variantes del numero de paradas son la misma magnitud: `total_stops` es
`dropoff_stops` mas la estacion, y las demas son recuentos equivalentes de la misma secuencia.
Hay que quedarse con una sola: meter varias juntas introduce colinealidad sin aportar nada.

## B.2 Como es una ruta tipica

- `total_stops`: media 147.99, mediana 151.0, rango de 33.0 a 238.0
- `package_count`: media 238.41, mediana 239.0, rango de 150.0 a 304.0
- `unique_zones`: media 20.04, mediana 20.0, rango de 4.0 a 47.0
- `total_package_volume_cm3`: media 2641162.73, mediana 2635624.31, rango de 1242909.09 a 4427701.53
- `packages_per_dropoff_stop`: media 1.69, mediana 1.56, rango de 1.16 a 7.34
- `route_duration_hours`: media 7.51, mediana 7.55, rango de 4.27 a 10.6

Figura: `artifacts/eda/figuras/15 Composicion de las rutas.png`

## B.3 Que determina la duracion de una ruta

Correlacion con `route_duration_seconds`, excluyendo sus propios componentes:

- `total_planned_service_time_seconds`: 0.6079
- `mean_planned_service_time_seconds`: 0.3948
- `planned_service_time_per_package_seconds`: 0.3948
- `total_stops`: 0.294
- `dropoff_stops`: 0.294
- `sequence_stop_count`: 0.294
- `stop_rows`: 0.294
- `max_actual_sequence_position`: 0.294

Figura: `artifacts/eda/figuras/16 Duracion de la ruta.png`

## B.4 route_score, la etiqueta que ya no es objetivo

Amazon etiqueto cada ruta historica segun la calidad de su secuencia. **Ya no es el objetivo del
proyecto**, pero conviene describirla porque estaba en el planteamiento anterior:

```text
Medium    3,292 rutas
High      2,718 rutas
Low         102 rutas
```

La clase minoritaria representa solo el 1.67% de las rutas. Ese desbalanceo severo fue una
de las razones para cambiar de objetivo: obligaba a pelearse con metricas por clase y particiones
estratificadas, mientras que el tiempo de entrega es una variable continua y bien repartida.

Duracion media segun la etiqueta:

```text
High       7.41 h
Low        7.81 h
Medium     7.60 h
```

---

## 15. Conclusiones para el modelado

1. **Distinguir las dos poblaciones.** Incluir `is_depot_segment` o entrenar por separado.
2. **Batir el baseline de R2 0.919 y MAE 0.45 minutos**, reajustado sobre el entrenamiento.
3. **Excluir las columnas constantes** y no duplicar variables que miden lo mismo.
4. **Aprovechar `same_zone`**, la mejor senal despues de la distancia.
5. **Incluir `station_code` si se usa meteorologia**, o las explicaciones seran falsas.
6. **Particionar con `GroupKFold` por `route_id`**: los tramos de una ruta no son independientes.
7. **No recortar los atipicos**: son casos reales, no errores.
8. **Metrica MAE en minutos**, que se explica sola ante un tribunal o un cliente.
9. **Priorizar arboles** (LightGBM, CatBoost): toleran la asimetria, manejan la alta cardinalidad
   de las zonas y permiten explicar cada prediccion con TreeSHAP, que es lo que necesita el agente.
