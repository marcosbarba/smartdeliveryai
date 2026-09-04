# Leer Primero

Tablas limpias de Amazon Last Mile. Cada una existe en `.csv`, para revisarla a mano, y en
`.parquet`, que es el formato que consume Gold.

## Archivos

- `Rutas.csv`: una fila por ruta. 6.112 filas.
- `Paradas.csv`: una fila por parada. 904.527 filas.
- `Paquetes.csv`: una fila por paquete. 1.457.175 filas.
- `Secuencias Reales.csv`: posicion real de cada parada dentro de su ruta.
- `Tiempos Viaje Segmentos.csv`: una fila por tramo entre dos paradas consecutivas de la
  secuencia real, con los segundos que costo recorrerlo. 898.415 filas.
- `Puntuaciones Secuencia Invalida.csv`: puntuacion por ruta.
- `Coordenadas Estaciones.csv`: coordenadas medias de cada estacion.

## Sobre la tabla de tramos

Es la base de la variable objetivo del proyecto. Sale de `travel_times.json`, que guarda el
tiempo entre **todas** las parejas de paradas de cada ruta: unos 156 millones de celdas. De ahi
solo se extraen los 898.415 tramos que el repartidor recorrio de verdad, es decir el 0,5%.

La cuenta cuadra asi: 904.527 paradas menos 6.112 rutas, porque una ruta de n paradas tiene n-1
saltos entre ellas.

Como el fichero de origen pesa 1,8 GB, se lee por streaming ruta a ruta en vez de cargarlo
entero en memoria.

## Uso

Gold une estas tablas con meteorologia y calendario para generar sus dos datasets: uno por ruta
y otro por tramo.
