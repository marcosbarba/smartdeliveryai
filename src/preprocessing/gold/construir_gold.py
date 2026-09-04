from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _detect_java_home() -> str | None:
    """Busca un JDK local. macOS: Homebrew. Windows: JDK portable en .jdk/ (ver README.md) o,
    si no existe, Eclipse Temurin instalado en el sistema."""
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]
    if platform.system() == "Windows":
        local_dir = PROJECT_ROOT / ".jdk"
        adoptium_dir = Path("C:/Program Files/Eclipse Adoptium")
        candidates = sorted(local_dir.glob("jdk-17*"), reverse=True) if local_dir.exists() else []
        candidates += sorted(adoptium_dir.glob("jdk-17*"), reverse=True) if adoptium_dir.exists() else []
    else:
        candidates = [Path("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home")]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


_java_home = _detect_java_home()
if _java_home:
    os.environ["JAVA_HOME"] = _java_home
    os.environ["PATH"] = f"{_java_home}{os.sep}bin{os.pathsep}{os.environ.get('PATH', '')}"
os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")
# Spark lanza sus workers con el 'python' del PATH. En Windows ese nombre suele resolver
# al stub de Microsoft Store, que no ejecuta nada: el worker no conecta de vuelta y Spark
# aborta con SocketTimeoutException. Apuntarlo al interprete que ya corre este script es
# correcto en cualquier sistema operativo.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

if platform.system() == "Windows":
    # winutils.exe/hadoop.dll: necesarios para que Spark escriba archivos en Windows.
    os.environ.setdefault("HADOOP_HOME", str(PROJECT_ROOT / ".hadoop"))
    os.environ["PATH"] = f"{os.environ['HADOOP_HOME']}{os.sep}bin{os.pathsep}{os.environ.get('PATH', '')}"

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

SILVER_DIR = PROJECT_ROOT / "data" / "silver"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
SPARK_TMP_DIR = Path(tempfile.gettempdir()) / "smartdeliveryai_spark_outputs"

AMAZON_DIR = SILVER_DIR / "Amazon Last Mile Clean"
WEATHER_DIR = SILVER_DIR / "Weather Open Meteo Clean"
CALENDAR_DIR = SILVER_DIR / "Calendar Nager Clean"

OUTPUT_DIR = GOLD_DIR / "Dataset Rutas Enriquecidas"
MANIFEST_DIR = GOLD_DIR / "Source Manifests"

GOLD_DATASET = OUTPUT_DIR / "Dataset Rutas Enriquecidas.csv"
GOLD_PARQUET = OUTPUT_DIR / "Dataset Rutas Enriquecidas.parquet"
DATA_CONTRACT = OUTPUT_DIR / "Contrato Datos Gold.md"
QUALITY_REPORT = MANIFEST_DIR / "Informe Calidad Gold.json"


GOLD_FIELDS = [
    "route_id",
    "station_code",
    "route_date",
    "departure_time_utc",
    "departure_timestamp_utc",
    "executor_capacity_cm3",
    "route_score",
    "route_score_numeric",
    "invalid_sequence_score",
    "total_stops",
    "dropoff_stops",
    "station_stops",
    "stop_rows",
    "unique_zones",
    "lat_range",
    "lng_range",
    "package_count",
    "delivered_package_count",
    "rejected_package_count",
    "delivery_attempted_count",
    "other_scan_status_count",
    "packages_with_time_window",
    "packages_with_time_window_ratio",
    "total_planned_service_time_seconds",
    "mean_planned_service_time_seconds",
    "total_package_volume_cm3",
    "mean_package_volume_cm3",
    "max_package_volume_cm3",
    "sequence_stop_count",
    "max_actual_sequence_position",
    "total_haversine_distance_km",
    "depot_to_first_stop_distance_km",
    "delivery_haversine_distance_km",
    "total_travel_time_seconds",
    "route_duration_seconds",
    "route_duration_hours",
    "temperature_2m_max_c",
    "temperature_2m_min_c",
    "temperature_2m_mean_c",
    "apparent_temperature_max_c",
    "apparent_temperature_min_c",
    "precipitation_sum_mm",
    "rain_sum_mm",
    "snowfall_sum_cm",
    "weather_code",
    "wind_speed_10m_max_kmh",
    "wind_gusts_10m_max_kmh",
    "year",
    "month",
    "day",
    "weekday",
    "weekday_name",
    "is_weekend",
    "is_us_public_holiday",
    "packages_per_dropoff_stop",
    "volume_per_package_cm3",
    "planned_service_time_per_package_seconds",
    "mean_delivery_segment_distance_km",
    "has_rain",
    "has_precipitation",
]

EARTH_RADIUS_KM = 6371.0088


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(text)


def recreate_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def write_gold_outputs(df: DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spark_run_dir = SPARK_TMP_DIR / f"gold_{uuid.uuid4().hex}"
    tmp_csv_dir = spark_run_dir / "csv"
    tmp_parquet_dir = spark_run_dir / "parquet"

    recreate_path(GOLD_DATASET)
    recreate_path(GOLD_PARQUET)
    recreate_path(spark_run_dir)
    spark_run_dir.mkdir(parents=True, exist_ok=True)

    selected = df.select(*GOLD_FIELDS)
    selected.write.mode("overwrite").parquet(str(tmp_parquet_dir))
    selected.coalesce(1).write.mode("overwrite").option("header", True).csv(str(tmp_csv_dir))

    part_files = sorted(tmp_csv_dir.glob("part-*.csv"))
    if not part_files:
        raise FileNotFoundError("Spark no genero el CSV Gold")
    shutil.move(str(part_files[0]), str(GOLD_DATASET))
    shutil.move(str(tmp_parquet_dir), str(GOLD_PARQUET))
    shutil.rmtree(spark_run_dir)


def create_contract(row_count: int) -> str:
    """El recuento de filas y columnas se inyecta desde la ejecucion real.

    Escribirlo a mano en el texto garantiza que tarde o temprano quede obsoleto: ya paso al
    anadir columnas nuevas y quedarse el contrato diciendo una cifra vieja.
    """
    return f"""# Contrato Datos Gold

Dimensiones de esta ejecucion: **{row_count:,} filas y {len(GOLD_FIELDS)} columnas**.

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
"""


def read_silver_table(spark: SparkSession, path: Path) -> DataFrame:
    parquet_path = path.with_suffix(".parquet")
    if parquet_path.exists():
        return spark.read.parquet(str(parquet_path))
    return spark.read.option("header", True).option("inferSchema", True).csv(str(path))


def haversine_distance_km(lat1: F.Column, lng1: F.Column, lat2: F.Column, lng2: F.Column) -> F.Column:
    """Distancia en linea recta (no por carretera) entre dos puntos lat/lng."""
    lat1_rad = F.radians(lat1)
    lat2_rad = F.radians(lat2)
    delta_lat = F.radians(lat2 - lat1)
    delta_lng = F.radians(lng2 - lng1)
    a = F.sin(delta_lat / 2) ** 2 + F.cos(lat1_rad) * F.cos(lat2_rad) * F.sin(delta_lng / 2) ** 2
    c = 2 * F.atan2(F.sqrt(a), F.sqrt(1 - a))
    return F.lit(EARTH_RADIUS_KM) * c


def build_distance_aggregates(stops: DataFrame, sequences: DataFrame) -> DataFrame:
    """Distancia Haversine entre paradas consecutivas segun la secuencia real de entrega.

    Se separa el tramo que sale de la estacion (almacen -> zona de reparto) del reparto en si,
    porque son fenomenos distintos: en este dataset ese primer tramo es ~38% de la distancia total.
    """
    ordered_stops = (
        stops.withColumn("lat_double", F.col("lat").cast("double"))
        .withColumn("lng_double", F.col("lng").cast("double"))
        .join(
            sequences.withColumn("actual_sequence_position_int", F.col("actual_sequence_position").cast("int")),
            ["route_id", "stop_id"],
        )
        .filter(F.col("lat_double").isNotNull() & F.col("lng_double").isNotNull())
        .select("route_id", "actual_sequence_position_int", "lat_double", "lng_double", "stop_type")
    )

    sequence_window = Window.partitionBy("route_id").orderBy("actual_sequence_position_int")
    segments = (
        ordered_stops.withColumn("prev_lat", F.lag("lat_double").over(sequence_window))
        .withColumn("prev_lng", F.lag("lng_double").over(sequence_window))
        .withColumn("prev_stop_type", F.lag("stop_type").over(sequence_window))
    )
    segments = segments.withColumn(
        "segment_distance_km",
        F.when(
            segments["prev_lat"].isNotNull(),
            haversine_distance_km(segments["prev_lat"], segments["prev_lng"], segments["lat_double"], segments["lng_double"]),
        ),
    )
    # El tramo que arranca en la estacion es el desplazamiento almacen -> zona de reparto.
    depot_segment = F.col("prev_stop_type") == "Station"
    delivery_segment = F.col("segment_distance_km").isNotNull() & ~depot_segment

    return segments.groupBy("route_id").agg(
        F.round(F.sum("segment_distance_km"), 6).alias("total_haversine_distance_km"),
        F.round(F.sum(F.when(depot_segment, F.col("segment_distance_km"))), 6).alias("depot_to_first_stop_distance_km"),
        F.round(F.sum(F.when(delivery_segment, F.col("segment_distance_km"))), 6).alias("delivery_haversine_distance_km"),
        F.round(F.avg(F.when(delivery_segment, F.col("segment_distance_km"))), 6).alias("mean_delivery_segment_distance_km"),
    )


def build_travel_time_aggregates(spark: SparkSession) -> DataFrame | None:
    """Suma por ruta los tiempos de viaje de los tramos realmente recorridos.

    Devuelve None si la tabla no existe, porque travel_times.json es una descarga
    opcional de 1,8 GB y el pipeline debe funcionar igual sin ella.
    """
    segments_csv = AMAZON_DIR / "Tiempos Viaje Segmentos.csv"
    if not segments_csv.exists() and not segments_csv.with_suffix(".parquet").exists():
        return None
    segments = read_silver_table(spark, segments_csv)
    return segments.groupBy("route_id").agg(
        F.round(F.sum(F.col("travel_time_seconds").cast("double")), 6).alias("total_travel_time_seconds"),
        F.count("*").alias("travel_segment_count"),
    )


def build_gold(spark: SparkSession) -> tuple[DataFrame, dict[str, int], dict[str, int]]:
    routes = read_silver_table(spark, AMAZON_DIR / "Rutas.csv")
    stops = read_silver_table(spark, AMAZON_DIR / "Paradas.csv")
    packages = read_silver_table(spark, AMAZON_DIR / "Paquetes.csv")
    sequences = read_silver_table(spark, AMAZON_DIR / "Secuencias Reales.csv")
    invalid_scores = read_silver_table(spark, AMAZON_DIR / "Puntuaciones Secuencia Invalida.csv")
    weather = read_silver_table(spark, WEATHER_DIR / "Meteorologia Diaria.csv")
    calendar = read_silver_table(spark, CALENDAR_DIR / "Calendario Diario.csv")

    stop_aggregates = (
        stops.withColumn("lat_double", F.col("lat").cast("double"))
        .withColumn("lng_double", F.col("lng").cast("double"))
        .groupBy("route_id")
        .agg(
            F.count("*").alias("stop_rows"),
            F.countDistinct("zone_id").alias("unique_zones"),
            F.sum(F.when(F.col("stop_type") == "Station", 1).otherwise(0)).alias("station_stop_count"),
            F.sum(F.when(F.col("stop_type") == "Dropoff", 1).otherwise(0)).alias("dropoff_stop_count"),
            F.min("lat_double").alias("lat_min"),
            F.max("lat_double").alias("lat_max"),
            F.min("lng_double").alias("lng_min"),
            F.max("lng_double").alias("lng_max"),
        )
        .withColumn("lat_range", F.round(F.col("lat_max") - F.col("lat_min"), 6))
        .withColumn("lng_range", F.round(F.col("lng_max") - F.col("lng_min"), 6))
        .select("route_id", "stop_rows", "unique_zones", "station_stop_count", "dropoff_stop_count", "lat_range", "lng_range")
    )

    package_aggregates = (
        packages.withColumn("planned_service_time_seconds_double", F.col("planned_service_time_seconds").cast("double"))
        .withColumn("volume_cm3_double", F.col("volume_cm3").cast("double"))
        .withColumn("has_time_window_int", F.col("has_time_window").cast("int"))
        .groupBy("route_id")
        .agg(
            F.count("*").alias("package_count"),
            F.sum(F.when(F.col("scan_status") == "DELIVERED", 1).otherwise(0)).alias("delivered_package_count"),
            F.sum(F.when(F.col("scan_status") == "REJECTED", 1).otherwise(0)).alias("rejected_package_count"),
            # Intento de entrega fallido: el repartidor paso y no habia nadie. Se cuenta
            # explicitamente y no por resta, para que siga siendo correcto si Amazon anadiera
            # algun estado nuevo. Es el 0,85% de los paquetes.
            F.sum(F.when(F.col("scan_status") == "DELIVERY_ATTEMPTED", 1).otherwise(0)).alias("delivery_attempted_count"),
            F.sum("has_time_window_int").alias("packages_with_time_window"),
            F.sum("planned_service_time_seconds_double").alias("total_planned_service_time_seconds"),
            F.avg("planned_service_time_seconds_double").alias("mean_planned_service_time_seconds"),
            F.sum("volume_cm3_double").alias("total_package_volume_cm3"),
            F.avg("volume_cm3_double").alias("mean_package_volume_cm3"),
            F.max("volume_cm3_double").alias("max_package_volume_cm3"),
        )
        .withColumn(
            # Cajon de sastre: deberia ser 0 siempre. Si deja de serlo es que ha aparecido un
            # estado de escaneo que no estamos contemplando.
            "other_scan_status_count",
            F.col("package_count")
            - F.col("delivered_package_count")
            - F.col("rejected_package_count")
            - F.col("delivery_attempted_count"),
        )
        .withColumn("packages_with_time_window_ratio", F.round(F.col("packages_with_time_window") / F.col("package_count"), 6))
    )

    sequence_aggregates = (
        sequences.withColumn("actual_sequence_position_int", F.col("actual_sequence_position").cast("int"))
        .groupBy("route_id")
        .agg(
            F.count("*").alias("sequence_stop_count"),
            F.max("actual_sequence_position_int").alias("max_actual_sequence_position"),
        )
    )

    distance_aggregates = build_distance_aggregates(stops, sequences)
    travel_aggregates = build_travel_time_aggregates(spark)

    joined = (
        routes.alias("r")
        .join(invalid_scores.alias("i"), "route_id", "left")
        .join(stop_aggregates.alias("s"), "route_id", "left")
        .join(package_aggregates.alias("p"), "route_id", "left")
        .join(sequence_aggregates.alias("seq"), "route_id", "left")
        .join(distance_aggregates.alias("d"), "route_id", "left")
        .join(
            weather.alias("w"),
            (F.col("r.station_code") == F.col("w.station_code")) & (F.col("r.route_date") == F.col("w.weather_date")),
            "left",
        )
        .join(calendar.alias("c"), F.col("r.route_date") == F.col("c.calendar_date"), "left")
    )

    # Los tiempos de viaje son una descarga opcional: si no estan, las columnas de
    # duracion se rellenan a nulo y el resto de Gold no cambia.
    if travel_aggregates is not None:
        joined = joined.join(travel_aggregates.alias("tt"), "route_id", "left")
    else:
        joined = joined.withColumn("total_travel_time_seconds", F.lit(None).cast("double"))

    package_count = F.col("package_count").cast("double")
    dropoff_stops = F.col("r.dropoff_stops").cast("double")
    total_volume = F.col("total_package_volume_cm3").cast("double")
    total_service = F.col("total_planned_service_time_seconds").cast("double")
    rain_sum = F.coalesce(F.col("w.rain_sum_mm").cast("double"), F.lit(0.0))
    precipitation_sum = F.coalesce(F.col("w.precipitation_sum_mm").cast("double"), F.lit(0.0))

    gold = (
        joined.withColumn("packages_per_dropoff_stop", F.when(dropoff_stops > 0, F.round(package_count / dropoff_stops, 6)))
        .withColumn("volume_per_package_cm3", F.when(package_count > 0, F.round(total_volume / package_count, 6)))
        .withColumn(
            "planned_service_time_per_package_seconds",
            F.when(package_count > 0, F.round(total_service / package_count, 6)),
        )
        .withColumn("has_rain", F.when(rain_sum > 0, 1).otherwise(0))
        .withColumn("has_precipitation", F.when(precipitation_sum > 0, 1).otherwise(0))
        # Duracion de la ruta = conducir entre paradas + atender cada entrega.
        .withColumn(
            "route_duration_seconds",
            F.round(F.col("total_travel_time_seconds").cast("double") + total_service, 6),
        )
        .withColumn("route_duration_hours", F.round(F.col("route_duration_seconds") / 3600, 6))
        .select(
            F.col("route_id"),
            F.col("r.station_code").alias("station_code"),
            F.col("r.route_date").alias("route_date"),
            F.col("departure_time_utc"),
            F.col("departure_timestamp_utc"),
            F.col("executor_capacity_cm3"),
            F.col("route_score"),
            F.col("route_score_numeric"),
            F.col("invalid_sequence_score"),
            F.col("total_stops"),
            F.col("r.dropoff_stops").alias("dropoff_stops"),
            F.col("r.station_stops").alias("station_stops"),
            F.col("stop_rows"),
            F.col("unique_zones"),
            F.col("lat_range"),
            F.col("lng_range"),
            F.col("package_count"),
            F.col("delivered_package_count"),
            F.col("rejected_package_count"),
            F.col("delivery_attempted_count"),
            F.col("other_scan_status_count"),
            F.col("packages_with_time_window"),
            F.col("packages_with_time_window_ratio"),
            F.round(F.col("total_planned_service_time_seconds"), 6).alias("total_planned_service_time_seconds"),
            F.round(F.col("mean_planned_service_time_seconds"), 6).alias("mean_planned_service_time_seconds"),
            F.round(F.col("total_package_volume_cm3"), 6).alias("total_package_volume_cm3"),
            F.round(F.col("mean_package_volume_cm3"), 6).alias("mean_package_volume_cm3"),
            F.round(F.col("max_package_volume_cm3"), 6).alias("max_package_volume_cm3"),
            F.col("sequence_stop_count"),
            F.col("max_actual_sequence_position"),
            F.col("total_haversine_distance_km"),
            F.col("depot_to_first_stop_distance_km"),
            F.col("delivery_haversine_distance_km"),
            F.round(F.col("total_travel_time_seconds"), 6).alias("total_travel_time_seconds"),
            F.col("route_duration_seconds"),
            F.col("route_duration_hours"),
            F.col("w.temperature_2m_max_c").alias("temperature_2m_max_c"),
            F.col("w.temperature_2m_min_c").alias("temperature_2m_min_c"),
            F.col("w.temperature_2m_mean_c").alias("temperature_2m_mean_c"),
            F.col("w.apparent_temperature_max_c").alias("apparent_temperature_max_c"),
            F.col("w.apparent_temperature_min_c").alias("apparent_temperature_min_c"),
            F.col("w.precipitation_sum_mm").alias("precipitation_sum_mm"),
            F.col("w.rain_sum_mm").alias("rain_sum_mm"),
            F.col("w.snowfall_sum_cm").alias("snowfall_sum_cm"),
            F.col("w.weather_code").alias("weather_code"),
            F.col("w.wind_speed_10m_max_kmh").alias("wind_speed_10m_max_kmh"),
            F.col("w.wind_gusts_10m_max_kmh").alias("wind_gusts_10m_max_kmh"),
            F.col("c.year").alias("year"),
            F.col("c.month").alias("month"),
            F.col("c.day").alias("day"),
            F.col("c.weekday").alias("weekday"),
            F.col("c.weekday_name").alias("weekday_name"),
            F.col("c.is_weekend").alias("is_weekend"),
            F.col("c.is_us_public_holiday").alias("is_us_public_holiday"),
            F.col("packages_per_dropoff_stop"),
            F.col("volume_per_package_cm3"),
            F.col("planned_service_time_per_package_seconds"),
            F.col("mean_delivery_segment_distance_km"),
            F.col("has_rain"),
            F.col("has_precipitation"),
        )
    )

    input_counts = {
        "routes": routes.count(),
        "stops": stops.count(),
        "packages": packages.count(),
        "sequences": sequences.count(),
        "invalid_scores": invalid_scores.count(),
        "weather_daily": weather.count(),
        "calendar_daily": calendar.count(),
    }
    validation = {
        "routes_without_weather": gold.filter(F.col("temperature_2m_mean_c").isNull()).count(),
        "routes_without_calendar": gold.filter(F.col("weekday").isNull()).count(),
        "routes_without_packages": gold.filter(F.col("package_count").isNull()).count(),
        "routes_without_stops": gold.filter(F.col("stop_rows").isNull()).count(),
        "routes_without_haversine_distance": gold.filter(F.col("total_haversine_distance_km").isNull()).count(),
        "routes_without_depot_distance": gold.filter(F.col("depot_to_first_stop_distance_km").isNull()).count(),
        "routes_without_duration": gold.filter(F.col("route_duration_seconds").isNull()).count(),
        # Deberia ser 0: si no, hay un estado de escaneo nuevo sin contemplar.
        "routes_with_unknown_scan_status": gold.filter(F.col("other_scan_status_count") != 0).count(),
        "routes_with_inconsistent_distance_split": gold.filter(
            F.abs(
                F.col("total_haversine_distance_km")
                - F.col("depot_to_first_stop_distance_km")
                - F.col("delivery_haversine_distance_km")
            )
            > 0.001
        ).count(),
    }
    return gold, input_counts, validation


def main() -> None:
    started_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    spark = (
        SparkSession.builder.appName("SmartDeliveryAI Gold")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.local.dir", str(Path(tempfile.gettempdir()) / "smartdeliveryai_spark_local"))
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        gold, input_counts, validation = build_gold(spark)
        output_rows = gold.count()
        output_columns = len(GOLD_FIELDS)
        write_gold_outputs(gold)
    finally:
        spark.stop()

    write_text(DATA_CONTRACT, create_contract(output_rows))
    finished_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    quality_report = {
        "layer": "gold",
        "status": "ok",
        "started_at_utc": started_at,
        "created_at_utc": finished_at,
        "engine": "pyspark",
        "spark_master": "local[*]",
        "format": "csv_and_parquet",
        "grain": "one row per route",
        "outputs": {
            "dataset_csv": GOLD_DATASET.relative_to(PROJECT_ROOT).as_posix(),
            "dataset_parquet": GOLD_PARQUET.relative_to(PROJECT_ROOT).as_posix(),
            "data_contract": DATA_CONTRACT.relative_to(PROJECT_ROOT).as_posix(),
        },
        "input_tables": input_counts,
        "output_rows": output_rows,
        "output_columns": output_columns,
        "validation": validation,
        "recommended_next_step": "Use Dataset Rutas Enriquecidas.csv for EDA local or Dataset Rutas Enriquecidas.parquet for Spark workflows.",
    }
    write_json(QUALITY_REPORT, quality_report)
    print(json.dumps(quality_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
