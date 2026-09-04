# Leer Primero

Gold contiene los datasets finales preparados para EDA, modelos, dashboard y agentes.

## Dos tablas, dos granos

```text
Dataset Tramos Ruta/          898.415 filas   una por tramo entre dos paradas
Dataset Rutas Enriquecidas/     6.112 filas   una por ruta completa
```

**`Dataset Tramos Ruta` es la tabla del modelo.** Su variable objetivo es
`travel_time_seconds`, el tiempo en ir de una parada a la siguiente. Sumando los tramos de una
ruta se obtiene la hora de llegada a cualquiera de sus paradas, que es lo que responde el
agente cuando un cliente pregunta por su paquete.

**`Dataset Rutas Enriquecidas`** agrega esa informacion por ruta: duracion total, distancias,
paquetes, meteorologia. Sirve para planificacion, analisis agregado y contexto.

## Carpetas

- `Dataset Tramos Ruta/`: dataset a nivel de tramo, con su contrato de datos.
- `Dataset Rutas Enriquecidas/`: dataset a nivel de ruta, con su contrato de datos.
- `Source Manifests/`: informes de calidad y trazabilidad de ambas tablas.

## Antes de usarlas

Leer el contrato de datos que acompana a cada tabla. Recogen avisos que condicionan el
modelado, sobre todo dos:

- **No usar variables derivadas del tiempo** (por ejemplo velocidades) como predictoras: el
  tiempo es el objetivo y se reconstruiria por division.
- **Particionar por `route_id`**, nunca al azar por filas: los tramos de una misma ruta no son
  independientes.

## Regla

Gold se construye desde Silver y se regenera con el pipeline. No editar estos archivos a mano:
cualquier cambio se pierde en la siguiente ejecucion.
