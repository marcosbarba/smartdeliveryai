# Contratos de datos: Gold y banco de producción simulada

> Consolida los tres contratos de datos que hoy viven repartidos entre `Data/Gold/*/` y
> `Data/Datos Simulados Produccion/` en el proyecto original. Se generan automáticamente por los
> scripts de `src/preprocessing/gold/` y `src/modelado/entrenamiento_final.py` respectivamente —
> no se editan a mano. Este documento resume el contenido para no tener que saltar entre tres
> ficheros para ver el contrato completo del pipeline.

## `data/gold/Dataset Tramos Ruta/` — la tabla del modelo

898 415 filas, una por tramo entre dos paradas consecutivas. Generada por
`src/preprocessing/gold/construir_gold_tramos.py`.

**Variable objetivo**: `travel_time_seconds` — segundos en ir de una parada a la siguiente.

**Variables principales**:

| Variable | Descripción |
|---|---|
| `route_id`, `segment_position` | Identificación y orden del tramo dentro de la ruta. |
| `is_depot_segment` | 1 si el tramo sale del almacén (primer tramo de la ruta), 0 si es entre dos entregas. Separa las dos poblaciones del modelo. |
| `segment_distance_km` | Distancia en línea recta (Haversine) entre las dos paradas del tramo — no distancia por carretera. |
| `same_zone`, `to_zone_id`, `from_zone_id` | Zona de planificación de destino/origen y si coinciden. Ausente (`NaN`) siempre en el tramo de almacén (no tiene zona de origen) y en un porcentaje pequeño de tramos de reparto. |
| `packages_at_destination`, `service_time_at_destination_seconds`, `volume_at_destination_cm3`, `packages_with_window_at_destination` | Carga de la parada de destino, conocida de antemano. |
| `departure_hour`, `route_total_stops`, `route_package_count`, `weekday` | Contexto de la ruta completa (constante para todos los tramos de una misma `route_id`). |
| `station_code` | Estación de origen. |
| `cumulative_distance_km` | Km recorridos hasta el INICIO de este tramo (sin contar su propia distancia) — ventana hasta la fila anterior, no hasta e incluyendo esta. |

**Avisos importantes**:

- **Fuga de datos**: nunca calcular velocidad (`distancia / tiempo`) como variable de entrada —
  reconstruiría el objetivo por división.
- **Partición**: siempre `GroupKFold` por `route_id`. Los tramos de una misma ruta no son
  independientes.
- El clima correlaciona espuriamente con el tiempo (ver [hallazgos_eda.md](hallazgos_eda.md),
  hallazgo 5); si se usa, incluir `station_code` para no aprender la relación falsa.

## `data/gold/Dataset Rutas Enriquecidas/` — contexto y planificación

6 112 filas, una por ruta completa. Generada por `src/preprocessing/gold/construir_gold.py`.

Agrega la información de `Dataset Tramos Ruta` por ruta: duración total, distancias, paquetes,
meteorología, calidad de secuencia (`route_score`, ya no es el objetivo del proyecto). Sirve para
planificación y análisis agregado, no para predecir un tramo concreto.

**Aviso importante**: `route_duration_seconds = total_travel_time_seconds +
total_planned_service_time_seconds` — no usar `total_travel_time_seconds` como predictor de
`route_duration_seconds`, es una fuga de datos directa (la suma que define el objetivo). Varias
columnas de recuento de paradas son la misma magnitud repetida (`total_stops`, `dropoff_stops`,
`stop_rows`, `sequence_stop_count`, `max_actual_sequence_position`): quedarse con una sola.

## `data/produccion_simulada/datos_produccion.parquet` — banco de producción simulada

Aproximadamente el 20% de las rutas de `Dataset Tramos Ruta` (mismo esquema, sin transformación
adicional). Generado por `src/modelado/datos.py` (`guardar_banco_produccion`), invocado desde
`src/modelado/entrenamiento_final.py` (`uv run sdai-train`).

**Por qué existe**: el proyecto no tiene acceso a un feed en vivo de rutas nuevas de Amazon. Este
fichero es literalmente **el test set del modelo** (ver [docs/modelado/](../modelado/)): un 20% de
rutas reservado por `route_id` **antes** de entrenar ninguna arquitectura, con semilla fija
(`SEMILLA_SPLIT = 7`). El modelo nunca lo ve en el ajuste, así que sirve honestamente para dos
cosas a la vez — la métrica de generalización reportada y la demo del agente — sin necesidad de
mantener dos bancos de datos ni dos modelos distintos.

**Qué contiene y qué no**: incluye `travel_time_seconds` intacto — el único sitio del proyecto
donde el objetivo se guarda junto a las variables de entrada, a propósito, para poder comparar
predicción contra realidad en la demo. **Nunca se usa como entrada del modelo**:
`src/agente/servidor_mcp/simulador_datos.py` separa `travel_time_seconds` en
`tiempos_reales_ocultos_segundos` al servir una ruta y no lo pasa nunca a los modelos.

**Reglas**: no editar a mano ni recortar filas — se regenera solo con `uv run sdai-train`. No usar
esta tabla para (re)entrenar ningún modelo: es precisamente lo que la vuelve honesta.
