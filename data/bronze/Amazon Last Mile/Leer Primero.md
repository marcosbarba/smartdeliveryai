# Leer Primero

Bronze de Amazon Last Mile.

## Archivos en Raw

- `Rutas Amazon.json`: rutas originales. 79 MB.
- `Paquetes Amazon.json`: paquetes originales. 375 MB.
- `Secuencias Reales Amazon.json`: orden real seguido por conductores. 10 MB.
- `Puntuaciones Secuencia Invalida Amazon.json`: puntuaciones de secuencias invalidas.
- `Tiempos Viaje Amazon.json`: tiempo real entre todas las parejas de paradas de cada ruta.
  **1,82 GB.** De aqui sale la variable objetivo del proyecto, asi que es obligatorio.

## Uso

Estos archivos son enlaces a los datos originales. Silver los transforma en tablas limpias.

## Si esta carpeta esta vacia

Los archivos no se descargan aqui: son los JSON originales de Amazon, que viven en
`02 Datos Amazon Last Mile/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Build Inputs/`,
enlazados a esta carpeta con los nombres que esperan los scripts de Silver.

Para prepararlos:

```text
python "04 Desarrollo Tecnico/01 Arquitectura Cloud Big Data/07 Pipeline Completo/Preparar Bronze Amazon.py"
```

No hace falta llamarlo a mano si se usa `Ejecutar Pipeline Completo.py`: el orquestador lo
ejecuta por su cuenta cuando detecta que faltan archivos.

El script crea enlaces duros para no duplicar los 2,3 GB del dataset. Si el dataset original
y la carpeta `Data` estan en discos distintos, copia los archivos en su lugar y avisa por pantalla.

Si tampoco existen los archivos originales, hay que descargarlos antes con
`02 Datos Amazon Last Mile/Descargar Dataset Amazon.sh`.

**Ojo con los tiempos de viaje**: ese fichero no se descarga por defecto por su tamano, y hay
que pedirlo expresamente:

```text
./"Descargar Dataset Amazon.sh" --con-tiempos-viaje
```

Sin el no se puede construir la tabla de tramos, que es la que alimenta el modelo.
