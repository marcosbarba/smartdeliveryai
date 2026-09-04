# Guía de modelado: qué predecir y las trampas que cuestan un TFM

Versión trimada de "Guia Data Scientist.md" del proyecto original: aquella guía cubría tres
carriles hipotéticos (Spark MLlib, ML tabular avanzado, Deep Learning distribuido) con ejemplos de
código genéricos que en la práctica no se usaron tal cual (no se entrenó ningún modelo con Spark
MLlib, ni hizo falta `DistributedDataParallel` ni lectura por lotes de un DataLoader masivo — el
dataset final, tras la agregación en Gold, cabe cómodamente en pandas). Este documento conserva
solo lo que de verdad condicionó el modelado: la variable objetivo, las trampas reales que había
que evitar, y qué se acabó entrenando.

## La variable objetivo

```text
travel_time_seconds     segundos en ir de una parada a la siguiente
```

Se predice **por tramo**, no por ruta:

```text
ALMACEN --tramo1--> parada#1 --tramo2--> parada#2 --tramo3--> parada#3
```

La hora de llegada a cualquier parada sale de sumar los tramos anteriores más los tiempos de
servicio de las paradas previas (conocidos de antemano, planificados por Amazon), y añadirlo a la
hora de salida del almacén. Predecir solo la duración total de la ruta no permitiría responder a
un cliente concreto; prediciendo tramos se obtienen las dos cosas, porque el total es la suma.

## Las cuatro trampas reales (verificadas sobre los datos, no solo advertidas)

**1. Fuga de datos por velocidad.** Nunca calcular una velocidad por tramo (`distancia / tiempo`)
como variable de entrada: la distancia ya está en la tabla, así que velocidad y distancia
reconstruirían el objetivo por división pura. Una velocidad **agregada** (media del conjunto de
entrenamiento) sí es legítima, igual que la pendiente de una regresión — lo prohibido es dar al
modelo la velocidad de *ese mismo tramo*.

**2. Partición por ruta, no por fila.** Los tramos de una misma ruta no son independientes:
`GroupKFold` agrupando por `route_id` en todo momento. Una partición aleatoria por filas dejaría
tramos de la misma ruta en entrenamiento y prueba, dando un resultado engañosamente bueno.

**3. Dos poblaciones, no una.** El tramo que sale del almacén dura de media 30 minutos; los
saltos entre entregas, 1 minuto — casi 30 veces menos. Un modelo único sin distinguirlas sirve mal
a las dos. Se resolvió con dos especialistas seleccionados por `is_depot_segment` (ver
`docs/modelado/`), no con una variable más en un modelo conjunto — un único LightGBM con
`is_depot_segment` como variable normal empeoró el R² pese a mejorar el MAE (bake-off, script
`lightgbm_unico`), señal de que la variable no bastaba para que el modelo repartiera bien su
capacidad entre las dos poblaciones.

**4. El baseline correcto no es el intuitivo.** El EDA comparó tres:

```text
distancia / velocidad media      R2 = -1,034   <- peor que predecir la media
una recta para todo               R2 =  0,874
una recta POR POBLACIÓN            R2 =  0,919   MAE 0,45 min   <- el baseline a batir
```

Dividir distancia entre una velocidad media da R2 **negativo** porque una única velocidad no
sirve a la vez a 35 km/h (reparto) y 11 km/h (patrón dentro del almacén). El baseline honesto se
ajusta solo con entrenamiento y se mide en prueba.

## Otras dos señales verificadas

- **La distancia no lo explica todo.** Su correlación global (0,935) es engañosa: viene de que
  hay dos grupos separados. Dentro de reparto baja a 0,764 (58% de la variabilidad). El 42%
  restante es lo que el modelo tiene que aprender de otras variables — la mejor pista es
  `same_zone` (misma zona de planificación que la parada anterior).
- **Las distancias son en línea recta (Haversine)**, no por carretera (ver
  `arquitectura_lakehouse.md`, decisión de descartar OSRM). Ojo con la colinealidad de variables
  derivadas de distancia si se usa un modelo lineal.

## Orden de modelos probado (bake-off, ver `docs/modelado/`)

```text
1. Baseline      recta por poblacion                     R2 0,919  MAE 0,45 min
2. LightGBM      unico, sin separar poblaciones           R2 peor que el baseline (ver arriba)
3. Hibrido        dos especialistas (reparto/almacen)
4. CatBoost       + to_zone_id nativo en reparto           arquitectura ganadora
5. LSTM/GRU       comparativo, no desplegado
```

Se recomienda **MAE en minutos** como métrica principal (se explica sola: "nos equivocamos de
media en 12 minutos"), con R² y RMSE como referencia secundaria. En datos tabulares los árboles
suelen ganar a las redes neuronales, así que el deep learning encajó como experimento comparativo
para la memoria, no como candidato serio a desplegar.

**Por qué el modelo final es de árboles, no solo por rendimiento**: TreeSHAP explica cada
predicción de forma exacta y en milisegundos — exactamente lo que necesita el agente para decir
*por qué* un paquete llega a esa hora. Con una LSTM esa explicación es mucho más cara y
aproximada. El requisito académico de interpretabilidad y la funcionalidad real del producto
resultaron ser la misma pieza.

## Variables objetivo alternativas descartadas

- `route_duration_seconds` (en la tabla de rutas): duración total. Útil para planificar turnos,
  no para responder a un cliente concreto sobre su paquete.
- `route_score` / `route_score_numeric`: calidad de la secuencia según Amazon — el objetivo
  original del proyecto antes del pivote (ver `arquitectura_lakehouse.md`). Queda como variable
  descriptiva; ojo si se usa, solo el 1,67% de las rutas son de clase "Low", desbalanceo severo.
