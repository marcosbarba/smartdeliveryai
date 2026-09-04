# Leer Primero

Dataset final para EDA y modelos.

## Archivos

- `Dataset Rutas Enriquecidas.csv`: una fila por ruta con variables logisticas, de distancia, de tiempo, meteorologicas y de calendario. El numero exacto de filas y columnas figura al principio del contrato de datos, que se regenera con cada ejecucion.
- `Dataset Rutas Enriquecidas.parquet`: el mismo dataset en formato Parquet, para leerlo con Spark.
- `Contrato Datos Gold.md`: explicacion de columnas y uso previsto.

## Uso

Este es el punto de partida del EDA y del modelado. Conviene leer antes el contrato de datos,
sobre todo el aviso de que las distancias son en linea recta y no por carretera.

Para trabajo local sencillo, el CSV. Para mantener el enfoque Big Data, el Parquet con PySpark.
Los dos tienen exactamente el mismo contenido.

## No editar a mano

Los tres archivos los genera `Construir Gold.py`. Cualquier cambio manual se perdera en la
siguiente ejecucion del pipeline.
