"""Construye la tabla Gold a nivel de tramo, que es la que alimenta el modelo de tiempos.

Cada fila es un salto entre dos paradas consecutivas de la secuencia real de entrega, y la
variable objetivo es lo que se tarda en recorrerlo. Sumando los tramos de una ruta se obtiene
la hora estimada de llegada a cualquiera de sus paradas.

Grano:

```text
Dataset Rutas Enriquecidas   una fila por ruta    6.112 filas    planificacion
Dataset Tramos Ruta          una fila por tramo   898.415 filas  prediccion de tiempos
```

Regla que se respeta aqui: todas las variables predictoras son cosas que se conocen **antes**
de que la ruta salga (paradas, paquetes, geografia, hora prevista, meteorologia). No se incluye
nada derivado del tiempo real, porque el tiempo real es justo lo que se quiere predecir.

Uso:

    uv run python src/preprocessing/gold/construir_gold_tramos.py
"""

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
    os.environ.setdefault("HADOOP_HOME", str(PROJECT_ROOT / ".hadoop"))
    os.environ["PATH"] = f"{os.environ['HADOOP_HOME']}{os.sep}bin{os.pathsep}{os.environ.get('PATH', '')}"

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

SILVER_DIR = PROJECT_ROOT / "data" / "silver"
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
SPARK_TMP_DIR = Path(tempfile.gettempdir()) / "smartdeliveryai_spark_outputs"

AMAZON_DIR = SILVER_DIR / "Amazon Last Mile Clean"
WEATHER_DIR = SILVER_DIR / "Weather Open Meteo Clean"
CALENDAR_DIR = SILVER_DIR / "Calendar Nager Clean"

OUTPUT_DIR = GOLD_DIR / "Dataset Tramos Ruta"
MANIFEST_DIR = GOLD_DIR / "Source Manifests"
GOLD_CSV = OUTPUT_DIR / "Dataset Tramos Ruta.csv"
GOLD_PARQUET = OUTPUT_DIR / "Dataset Tramos Ruta.parquet"
DATA_CONTRACT = OUTPUT_DIR / "Contrato Datos Gold Tramos.md"
QUALITY_REPORT = MANIFEST_DIR / "Informe Calidad Gold Tramos.json"

EARTH_RADIUS_KM = 6371.0088

GOLD_FIELDS = [
    # Identificacion
    "route_id",
    "segment_position",
    "from_stop_id",
    "to_stop_id",
    # OBJETIVO
    "travel_time_seconds",
    # Geografia del tramo
    "segment_distance_km",
    "is_depot_segment",
    "from_zone_id",
    "to_zone_id",
    "same_zone",
    "from_lat",
    "from_lng",
    "to_lat",
    "to_lng",
    # Carga en la parada de destino
    "packages_at_destination",
    "service_time_at_destination_seconds",
    "volume_at_destination_cm3",
    "packages_with_window_at_destination",
    # Posicion dentro de la ruta
    "segment_ratio",
    "cumulative_distance_km",
    # Contexto de la ruta
    "station_code",
    "route_date",
    "departure_hour",
    "route_total_stops",
    "route_package_count",
    "weekday",
    "is_weekend",
    "temperature_2m_mean_c",
    "rain_sum_mm",
    "has_rain",
    "wind_speed_10m_max_kmh",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def recreate_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def read_silver_table(spark: SparkSession, path: Path) -> DataFrame:
    parquet_path = path.with_suffix(".parquet")
    if parquet_path.exists():
        return spark.read.parquet(str(parquet_path))
    return spark.read.option("header", True).option("inferSchema", True).csv(str(path))


def haversine_distance_km(lat1, lng1, lat2, lng2):
    """Distancia en linea recta (no por carretera) entre dos puntos lat/lng."""
    lat1_rad, lat2_rad = F.radians(lat1), F.radians(lat2)
    delta_lat, delta_lng = F.radians(lat2 - lat1), F.radians(lng2 - lng1)
    a = F.sin(delta_lat / 2) ** 2 + F.cos(lat1_rad) * F.cos(lat2_rad) * F.sin(delta_lng / 2) ** 2
    return F.lit(EARTH_RADIUS_KM) * 2 * F.atan2(F.sqrt(a), F.sqrt(1 - a))


def write_outputs(df: DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = SPARK_TMP_DIR / f"gold_tramos_{uuid.uuid4().hex}"
    tmp_csv, tmp_parquet = run_dir / "csv", run_dir / "parquet"

    recreate_path(GOLD_CSV)
    recreate_path(GOLD_PARQUET)
    recreate_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    selected = df.select(*GOLD_FIELDS)
    selected.write.mode("overwrite").parquet(str(tmp_parquet))
    selected.coalesce(1).write.mode("overwrite").option("header", True).csv(str(tmp_csv))

    part_files = sorted(tmp_csv.glob("part-*.csv"))
    if not part_files:
        raise FileNotFoundError("Spark no genero el CSV de tramos")
    shutil.move(str(part_files[0]), str(GOLD_CSV))
    shutil.move(str(tmp_parquet), str(GOLD_PARQUET))
    shutil.rmtree(run_dir)


def build_gold_segments(spark: SparkSession) -> tuple[DataFrame, dict[str, Any]]:
    segments = read_silver_table(spark, AMAZON_DIR / "Tiempos Viaje Segmentos.csv")
    stops = read_silver_table(spark, AMAZON_DIR / "Paradas.csv")
    packages = read_silver_table(spark, AMAZON_DIR / "Paquetes.csv")
    routes = read_silver_table(spark, AMAZON_DIR / "Rutas.csv")
    weather = read_silver_table(spark, WEATHER_DIR / "Meteorologia Diaria.csv")
    calendar = read_silver_table(spark, CALENDAR_DIR / "Calendario Diario.csv")

    stop_geo = stops.select(
        "route_id",
        F.col("stop_id"),
        F.col("lat").cast("double").alias("lat"),
        F.col("lng").cast("double").alias("lng"),
        F.col("zone_id"),
        F.col("stop_type"),
    )

    # Carga de trabajo en cada parada: se conoce al planificar, sale del manifiesto.
    stop_load = packages.groupBy("route_id", "stop_id").agg(
        F.count("*").alias("packages_at_destination"),
        F.round(F.sum(F.col("planned_service_time_seconds").cast("double")), 3).alias("service_time_at_destination_seconds"),
        F.round(F.sum(F.col("volume_cm3").cast("double")), 3).alias("volume_at_destination_cm3"),
        F.sum(F.col("has_time_window").cast("int")).alias("packages_with_window_at_destination"),
    )

    origin = stop_geo.select(
        "route_id",
        F.col("stop_id").alias("from_stop_id"),
        F.col("lat").alias("from_lat"),
        F.col("lng").alias("from_lng"),
        F.col("zone_id").alias("from_zone_id"),
        F.col("stop_type").alias("from_stop_type"),
    )
    destination = stop_geo.select(
        "route_id",
        F.col("stop_id").alias("to_stop_id"),
        F.col("lat").alias("to_lat"),
        F.col("lng").alias("to_lng"),
        F.col("zone_id").alias("to_zone_id"),
    )

    df = (
        segments.join(origin, ["route_id", "from_stop_id"], "left")
        .join(destination, ["route_id", "to_stop_id"], "left")
        .join(stop_load.withColumnRenamed("stop_id", "to_stop_id"), ["route_id", "to_stop_id"], "left")
        .join(routes.alias("r"), "route_id", "left")
        .join(
            weather.alias("w"),
            (F.col("r.station_code") == F.col("w.station_code")) & (F.col("r.route_date") == F.col("w.weather_date")),
            "left",
        )
        .join(calendar.alias("c"), F.col("r.route_date") == F.col("c.calendar_date"), "left")
    )

    total_stops = F.col("r.total_stops").cast("double")
    df = (
        df.withColumn(
            "segment_distance_km",
            F.round(haversine_distance_km(F.col("from_lat"), F.col("from_lng"), F.col("to_lat"), F.col("to_lng")), 6),
        )
        .withColumn("is_depot_segment", F.when(F.col("from_stop_type") == "Station", 1).otherwise(0))
        .withColumn(
            "same_zone",
            F.when(F.col("from_zone_id").isNull() | F.col("to_zone_id").isNull(), None)
            .when(F.col("from_zone_id") == F.col("to_zone_id"), 1)
            .otherwise(0),
        )
        # Que porcentaje de la ruta lleva recorrido: 0 al principio, 1 al final.
        .withColumn("segment_ratio", F.round(F.col("segment_position").cast("double") / (total_stops - 1), 6))
        # Se usa departure_timestamp_utc y no departure_time_utc: el segundo guarda solo la
        # hora, y al leerlo Spark le pone la fecha de hoy delante, que es basura.
        .withColumn("departure_hour", F.hour(F.col("r.departure_timestamp_utc").cast("timestamp")))
        .withColumn("has_rain", F.when(F.coalesce(F.col("w.rain_sum_mm").cast("double"), F.lit(0.0)) > 0, 1).otherwise(0))
    )

    # Distancia acumulada desde el almacen hasta el inicio de este tramo. Se calcula con
    # las coordenadas, que se conocen al planificar, asi que no es fuga de datos.
    from pyspark.sql.window import Window

    running = Window.partitionBy("route_id").orderBy("segment_position").rowsBetween(Window.unboundedPreceding, -1)
    df = df.withColumn("cumulative_distance_km", F.round(F.coalesce(F.sum("segment_distance_km").over(running), F.lit(0.0)), 6))

    gold = df.select(
        F.col("route_id"),
        F.col("segment_position").cast("int").alias("segment_position"),
        F.col("from_stop_id"),
        F.col("to_stop_id"),
        F.round(F.col("travel_time_seconds").cast("double"), 3).alias("travel_time_seconds"),
        F.col("segment_distance_km"),
        F.col("is_depot_segment"),
        F.col("from_zone_id"),
        F.col("to_zone_id"),
        F.col("same_zone"),
        F.col("from_lat"),
        F.col("from_lng"),
        F.col("to_lat"),
        F.col("to_lng"),
        F.coalesce(F.col("packages_at_destination"), F.lit(0)).alias("packages_at_destination"),
        F.coalesce(F.col("service_time_at_destination_seconds"), F.lit(0.0)).alias("service_time_at_destination_seconds"),
        F.coalesce(F.col("volume_at_destination_cm3"), F.lit(0.0)).alias("volume_at_destination_cm3"),
        F.coalesce(F.col("packages_with_window_at_destination"), F.lit(0)).alias("packages_with_window_at_destination"),
        F.col("segment_ratio"),
        F.col("cumulative_distance_km"),
        F.col("r.station_code").alias("station_code"),
        F.col("r.route_date").alias("route_date"),
        F.col("departure_hour"),
        F.col("r.total_stops").alias("route_total_stops"),
        F.col("c.weekday").alias("weekday"),
        F.col("c.is_weekend").alias("is_weekend"),
        F.col("w.temperature_2m_mean_c").alias("temperature_2m_mean_c"),
        F.col("w.rain_sum_mm").alias("rain_sum_mm"),
        F.col("has_rain"),
        F.col("w.wind_speed_10m_max_kmh").alias("wind_speed_10m_max_kmh"),
    )

    # El numero de paquetes de toda la ruta se anade aparte para no repetir el join.
    route_packages = packages.groupBy("route_id").agg(F.count("*").alias("route_package_count"))
    gold = gold.join(route_packages, "route_id", "left")

    validation = {
        "tramos": gold.count(),
        "rutas": gold.select("route_id").distinct().count(),
        "tramos_sin_objetivo": gold.filter(F.col("travel_time_seconds").isNull()).count(),
        "tramos_sin_distancia": gold.filter(F.col("segment_distance_km").isNull()).count(),
        "tramos_sin_estacion": gold.filter(F.col("station_code").isNull()).count(),
        "tramos_sin_meteorologia": gold.filter(F.col("temperature_2m_mean_c").isNull()).count(),
        "tramos_del_deposito": gold.filter(F.col("is_depot_segment") == 1).count(),
    }
    return gold, validation


def create_contract() -> str:
    return """# Contrato Datos Gold: tramos de ruta

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
"""


def main() -> None:
    started_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    spark = (
        SparkSession.builder.appName("SmartDeliveryAI Gold Tramos")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.local.dir", str(Path(tempfile.gettempdir()) / "smartdeliveryai_spark_local"))
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        gold, validation = build_gold_segments(spark)
        write_outputs(gold)
    finally:
        spark.stop()

    DATA_CONTRACT.parent.mkdir(parents=True, exist_ok=True)
    DATA_CONTRACT.write_text(create_contract(), encoding="utf-8")

    report = {
        "layer": "gold",
        "table": "Dataset Tramos Ruta",
        "status": "ok",
        "started_at_utc": started_at,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "engine": "pyspark",
        "grain": "one row per segment between consecutive stops",
        "target": "travel_time_seconds",
        "outputs": {
            "dataset_csv": GOLD_CSV.relative_to(PROJECT_ROOT).as_posix(),
            "dataset_parquet": GOLD_PARQUET.relative_to(PROJECT_ROOT).as_posix(),
            "data_contract": DATA_CONTRACT.relative_to(PROJECT_ROOT).as_posix(),
        },
        "output_columns": len(GOLD_FIELDS),
        "validation": validation,
        "recommended_next_step": "EDA sobre el tiempo de tramo y modelado con GroupKFold por route_id.",
    }
    write_json(QUALITY_REPORT, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
