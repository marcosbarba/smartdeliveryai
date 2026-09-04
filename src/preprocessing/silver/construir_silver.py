from __future__ import annotations

import csv
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import uuid
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
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

BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
SPARK_TMP_DIR = Path(tempfile.gettempdir()) / "smartdeliveryai_spark_outputs"
SPARK_INPUT_TMP_DIR = Path(tempfile.gettempdir()) / "smartdeliveryai_spark_inputs"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    return value


def clean_rows(rows: list[dict[str, Any]], fieldnames: list[str]) -> list[dict[str, Any]]:
    return [{field: clean_value(row.get(field)) for field in fieldnames} for row in rows]


def recreate_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def write_spark_table(df: DataFrame, csv_path: Path, fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = csv_path.with_suffix(".parquet")
    spark_run_dir = SPARK_TMP_DIR / f"silver_{uuid.uuid4().hex}"
    tmp_csv_dir = spark_run_dir / "csv"
    tmp_parquet_dir = spark_run_dir / "parquet"

    recreate_path(csv_path)
    recreate_path(parquet_path)
    recreate_path(spark_run_dir)
    spark_run_dir.mkdir(parents=True, exist_ok=True)

    selected = df.select(*fieldnames)
    selected.write.mode("overwrite").parquet(str(tmp_parquet_dir))
    selected.coalesce(1).write.mode("overwrite").option("header", True).csv(str(tmp_csv_dir))

    part_files = sorted(tmp_csv_dir.glob("part-*.csv"))
    if not part_files:
        raise FileNotFoundError(f"Spark no genero CSV para {csv_path}")
    shutil.move(str(part_files[0]), str(csv_path))
    shutil.move(str(tmp_parquet_dir), str(parquet_path))
    shutil.rmtree(spark_run_dir)


def create_df(spark: SparkSession, rows: list[dict[str, Any]], fieldnames: list[str]) -> DataFrame:
    SPARK_INPUT_TMP_DIR.mkdir(parents=True, exist_ok=True)
    staging_path = SPARK_INPUT_TMP_DIR / f"silver_input_{uuid.uuid4().hex}.csv"
    with staging_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in clean_rows(rows, fieldnames):
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})

    return spark.read.option("header", True).option("inferSchema", True).csv(str(staging_path)).select(*fieldnames)


def build_amazon_tables(spark: SparkSession) -> dict[str, int]:
    route_data = read_json(BRONZE_DIR / "Amazon Last Mile" / "Raw" / "Rutas Amazon.json")
    package_data = read_json(BRONZE_DIR / "Amazon Last Mile" / "Raw" / "Paquetes Amazon.json")
    actual_sequences = read_json(BRONZE_DIR / "Amazon Last Mile" / "Raw" / "Secuencias Reales Amazon.json")
    invalid_scores = read_json(BRONZE_DIR / "Amazon Last Mile" / "Raw" / "Puntuaciones Secuencia Invalida Amazon.json")

    routes: list[dict[str, Any]] = []
    stops: list[dict[str, Any]] = []
    packages: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    invalid_score_rows: list[dict[str, Any]] = []
    station_points: dict[str, list[tuple[float, float]]] = defaultdict(list)
    route_score_map = {"Low": 1, "Medium": 2, "High": 3}

    for route_id, route in route_data.items():
        route_stops = route.get("stops", {})
        stop_type_counts = Counter(stop.get("type", "") for stop in route_stops.values())
        lat_values = [stop.get("lat") for stop in route_stops.values() if isinstance(stop.get("lat"), (int, float))]
        lng_values = [stop.get("lng") for stop in route_stops.values() if isinstance(stop.get("lng"), (int, float))]

        routes.append(
            {
                "route_id": route_id,
                "station_code": route.get("station_code"),
                "route_date": route.get("date_YYYY_MM_DD"),
                "departure_time_utc": route.get("departure_time_utc"),
                "departure_timestamp_utc": f"{route.get('date_YYYY_MM_DD')}T{route.get('departure_time_utc')}Z",
                "executor_capacity_cm3": route.get("executor_capacity_cm3"),
                "route_score": route.get("route_score"),
                "route_score_numeric": route_score_map.get(route.get("route_score")),
                "total_stops": len(route_stops),
                "dropoff_stops": stop_type_counts.get("Dropoff", 0),
                "station_stops": stop_type_counts.get("Station", 0),
                "avg_stop_lat": sum(lat_values) / len(lat_values) if lat_values else None,
                "avg_stop_lng": sum(lng_values) / len(lng_values) if lng_values else None,
            }
        )

        for stop_id, stop in route_stops.items():
            is_station = stop.get("type") == "Station"
            if is_station and isinstance(stop.get("lat"), (int, float)) and isinstance(stop.get("lng"), (int, float)):
                station_points[route.get("station_code")].append((stop["lat"], stop["lng"]))
            stops.append(
                {
                    "route_id": route_id,
                    "stop_id": stop_id,
                    "station_code": route.get("station_code"),
                    "route_date": route.get("date_YYYY_MM_DD"),
                    "lat": stop.get("lat"),
                    "lng": stop.get("lng"),
                    "stop_type": stop.get("type"),
                    "zone_id": stop.get("zone_id"),
                    "is_station": int(is_station),
                }
            )

    for route_id, route_packages in package_data.items():
        for stop_id, stop_packages in route_packages.items():
            for package_id, package in stop_packages.items():
                dimensions = package.get("dimensions", {})
                depth = dimensions.get("depth_cm")
                height = dimensions.get("height_cm")
                width = dimensions.get("width_cm")
                volume = depth * height * width if all(isinstance(x, (int, float)) for x in [depth, height, width]) else None
                time_window = package.get("time_window", {})
                start_time = time_window.get("start_time_utc")
                end_time = time_window.get("end_time_utc")
                packages.append(
                    {
                        "route_id": route_id,
                        "stop_id": stop_id,
                        "package_id": package_id,
                        "scan_status": package.get("scan_status"),
                        "time_window_start_utc": start_time,
                        "time_window_end_utc": end_time,
                        # Ojo: en el JSON las ventanas ausentes vienen como NaN, y en Python
                        # bool(nan) es True. Comprobar el tipo evita contar como ventana lo
                        # que en realidad es un hueco: solo el 8% de paquetes tiene ventana.
                        "has_time_window": int(isinstance(start_time, str) or isinstance(end_time, str)),
                        "planned_service_time_seconds": package.get("planned_service_time_seconds"),
                        "depth_cm": depth,
                        "height_cm": height,
                        "width_cm": width,
                        "volume_cm3": volume,
                    }
                )

    for route_id, sequence in actual_sequences.items():
        for stop_id, actual_position in sequence.get("actual", {}).items():
            sequence_rows.append({"route_id": route_id, "stop_id": stop_id, "actual_sequence_position": actual_position})

    for route_id, score in invalid_scores.items():
        invalid_score_rows.append({"route_id": route_id, "invalid_sequence_score": score})

    station_rows = [
        {
            "station_code": station_code,
            "station_lat": sum(point[0] for point in points) / len(points),
            "station_lng": sum(point[1] for point in points) / len(points),
            "routes_with_station_point": len(points),
        }
        for station_code, points in sorted(station_points.items())
    ]

    base = SILVER_DIR / "Amazon Last Mile Clean"
    tables = [
        (
            base / "Rutas.csv",
            routes,
            [
                "route_id",
                "station_code",
                "route_date",
                "departure_time_utc",
                "departure_timestamp_utc",
                "executor_capacity_cm3",
                "route_score",
                "route_score_numeric",
                "total_stops",
                "dropoff_stops",
                "station_stops",
                "avg_stop_lat",
                "avg_stop_lng",
            ],
        ),
        (
            base / "Paradas.csv",
            stops,
            ["route_id", "stop_id", "station_code", "route_date", "lat", "lng", "stop_type", "zone_id", "is_station"],
        ),
        (
            base / "Paquetes.csv",
            packages,
            [
                "route_id",
                "stop_id",
                "package_id",
                "scan_status",
                "time_window_start_utc",
                "time_window_end_utc",
                "has_time_window",
                "planned_service_time_seconds",
                "depth_cm",
                "height_cm",
                "width_cm",
                "volume_cm3",
            ],
        ),
        (base / "Secuencias Reales.csv", sequence_rows, ["route_id", "stop_id", "actual_sequence_position"]),
        (base / "Puntuaciones Secuencia Invalida.csv", invalid_score_rows, ["route_id", "invalid_sequence_score"]),
        (
            base / "Coordenadas Estaciones.csv",
            station_rows,
            ["station_code", "station_lat", "station_lng", "routes_with_station_point"],
        ),
    ]

    for path, rows, fields in tables:
        df = create_df(spark, rows, fields)
        if "departure_time_utc" in fields:
            # departure_time_utc guarda solo una hora ("16:02:10"). Spark la interpreta como
            # timestamp y le pone la fecha de HOY delante, lo que da lugar a fechas absurdas si
            # alguien opera con la columna. Se normaliza a texto para que solo contenga la hora.
            # Para trabajar con fecha y hora juntas esta departure_timestamp_utc.
            df = df.withColumn("departure_time_utc", F.date_format(F.col("departure_time_utc").cast("timestamp"), "HH:mm:ss"))
        write_spark_table(df, path, fields)

    return {
        "routes": len(routes),
        "stops": len(stops),
        "packages": len(packages),
        "actual_sequence_rows": len(sequence_rows),
        "invalid_sequence_scores": len(invalid_score_rows),
        "stations": len(station_rows),
    }


def build_travel_time_segments(spark: SparkSession) -> dict[str, Any]:
    """Extrae los tramos recorridos de la matriz de tiempos de viaje.

    travel_times.json guarda, por cada ruta, el tiempo entre TODAS las parejas de paradas:
    unos 156 millones de celdas en total, porque el reto de Amazon consistia en elegir el
    orden de las paradas y hacia falta poder evaluar cualquier orden candidato.

    Aqui solo interesan los tramos que el repartidor recorrio de verdad, es decir las parejas
    consecutivas de la secuencia real: unos 900.000 tramos, el 0,5% de la matriz.

    El fichero pesa 1,8 GB, asi que se lee por streaming ruta a ruta en vez de cargarlo
    entero en memoria. Si no esta descargado, se omite este paso sin romper el pipeline.
    """
    travel_path = BRONZE_DIR / "Amazon Last Mile" / "Raw" / "Tiempos Viaje Amazon.json"
    if not travel_path.exists():
        return {
            "status": "omitido",
            "motivo": "Tiempos Viaje Amazon.json no esta en Bronze. Descarga opcional de 1,8 GB.",
            "travel_time_segments": 0,
        }

    import ijson

    sequences = read_json(BRONZE_DIR / "Amazon Last Mile" / "Raw" / "Secuencias Reales Amazon.json")
    orders = {
        route_id: sorted(seq.get("actual", {}), key=seq.get("actual", {}).get)
        for route_id, seq in sequences.items()
    }

    rows: list[dict[str, Any]] = []
    routes_seen = 0
    routes_without_sequence = 0
    expected_segments = 0
    missing_segments = 0

    with travel_path.open("rb") as file:
        for route_id, matrix in ijson.kvitems(file, ""):
            routes_seen += 1
            order = orders.get(route_id)
            if not order:
                routes_without_sequence += 1
                continue
            expected_segments += len(order) - 1
            for position, (origin, destination) in enumerate(zip(order, order[1:]), start=1):
                seconds = matrix.get(origin, {}).get(destination)
                if seconds is None:
                    # No deberia pasar: la matriz cubre todas las parejas de la ruta.
                    # Se cuenta para que un hueco no pase desapercibido.
                    missing_segments += 1
                    continue
                rows.append(
                    {
                        "route_id": route_id,
                        "segment_position": position,
                        "from_stop_id": origin,
                        "to_stop_id": destination,
                        "travel_time_seconds": float(seconds),
                    }
                )

    fields = ["route_id", "segment_position", "from_stop_id", "to_stop_id", "travel_time_seconds"]
    write_spark_table(
        create_df(spark, rows, fields),
        SILVER_DIR / "Amazon Last Mile Clean" / "Tiempos Viaje Segmentos.csv",
        fields,
    )
    return {
        "status": "ok",
        "travel_time_routes": routes_seen,
        "travel_time_segments": len(rows),
        "expected_segments": expected_segments,
        "missing_segments": missing_segments,
        "routes_without_sequence": routes_without_sequence,
    }


def build_weather_table(spark: SparkSession) -> dict[str, int]:
    rows: list[dict[str, Any]] = []
    weather_files = sorted((BRONZE_DIR / "Weather Open Meteo" / "Raw").glob("*.json"))

    for weather_file in weather_files:
        payload = read_json(weather_file)
        station = payload["station"]
        raw_response = payload["raw_response"]
        daily = raw_response["daily"]
        units = raw_response.get("daily_units", {})

        for index, day in enumerate(daily["time"]):
            rows.append(
                {
                    "station_code": station["station_code"],
                    "weather_date": day,
                    "station_lat": station["latitude"],
                    "station_lng": station["longitude"],
                    "temperature_2m_max_c": daily.get("temperature_2m_max", [])[index],
                    "temperature_2m_min_c": daily.get("temperature_2m_min", [])[index],
                    "temperature_2m_mean_c": daily.get("temperature_2m_mean", [])[index],
                    "apparent_temperature_max_c": daily.get("apparent_temperature_max", [])[index],
                    "apparent_temperature_min_c": daily.get("apparent_temperature_min", [])[index],
                    "precipitation_sum_mm": daily.get("precipitation_sum", [])[index],
                    "rain_sum_mm": daily.get("rain_sum", [])[index],
                    "snowfall_sum_cm": daily.get("snowfall_sum", [])[index],
                    "weather_code": daily.get("weather_code", [])[index],
                    "wind_speed_10m_max_kmh": daily.get("wind_speed_10m_max", [])[index],
                    "wind_gusts_10m_max_kmh": daily.get("wind_gusts_10m_max", [])[index],
                    "source_timezone": raw_response.get("timezone"),
                    "temperature_unit": units.get("temperature_2m_mean"),
                    "precipitation_unit": units.get("precipitation_sum"),
                    "wind_speed_unit": units.get("wind_speed_10m_max"),
                }
            )

    fields = [
        "station_code",
        "weather_date",
        "station_lat",
        "station_lng",
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
        "source_timezone",
        "temperature_unit",
        "precipitation_unit",
        "wind_speed_unit",
    ]
    write_spark_table(
        create_df(spark, rows, fields),
        SILVER_DIR / "Weather Open Meteo Clean" / "Meteorologia Diaria.csv",
        fields,
    )
    return {"weather_daily_rows": len(rows), "weather_files": len(weather_files)}


def build_calendar_tables(spark: SparkSession) -> dict[str, int]:
    route_data = read_json(BRONZE_DIR / "Amazon Last Mile" / "Raw" / "Rutas Amazon.json")
    holiday_payload = read_json(BRONZE_DIR / "Calendar Nager" / "Raw" / "Festivos EEUU 2018.json")
    holiday_records = holiday_payload["raw_response"]

    holidays: list[dict[str, Any]] = []
    holiday_dates = set()
    for item in holiday_records:
        holiday_dates.add(item["date"])
        holidays.append(
            {
                "holiday_date": item.get("date"),
                "local_name": item.get("localName"),
                "name": item.get("name"),
                "country_code": item.get("countryCode"),
                "is_fixed": int(bool(item.get("fixed"))),
                "is_global": int(bool(item.get("global"))),
                "counties": item.get("counties") or [],
                "types": item.get("types") or [],
            }
        )

    route_dates = sorted({route["date_YYYY_MM_DD"] for route in route_data.values()})
    current = parse_date(route_dates[0])
    end_date = parse_date(route_dates[-1])
    calendar_rows: list[dict[str, Any]] = []
    while current <= end_date:
        day = current.isoformat()
        calendar_rows.append(
            {
                "calendar_date": day,
                "year": current.year,
                "month": current.month,
                "day": current.day,
                "weekday": current.weekday(),
                "weekday_name": current.strftime("%A"),
                "is_weekend": int(current.weekday() >= 5),
                "is_us_public_holiday": int(day in holiday_dates),
            }
        )
        current += timedelta(days=1)

    holiday_fields = ["holiday_date", "local_name", "name", "country_code", "is_fixed", "is_global", "counties", "types"]
    calendar_fields = ["calendar_date", "year", "month", "day", "weekday", "weekday_name", "is_weekend", "is_us_public_holiday"]
    holiday_df = create_df(spark, holidays, holiday_fields).withColumn("counties", F.concat_ws("|", F.col("counties"))).withColumn(
        "types", F.concat_ws("|", F.col("types"))
    )
    write_spark_table(holiday_df, SILVER_DIR / "Calendar Nager Clean" / "Festivos EEUU.csv", holiday_fields)
    write_spark_table(
        create_df(spark, calendar_rows, calendar_fields),
        SILVER_DIR / "Calendar Nager Clean" / "Calendario Diario.csv",
        calendar_fields,
    )
    return {"holiday_records": len(holidays), "calendar_daily_rows": len(calendar_rows)}


def validate_gold_join_keys(spark: SparkSession) -> dict[str, int]:
    routes = spark.read.option("header", True).csv(str(SILVER_DIR / "Amazon Last Mile Clean" / "Rutas.csv"))
    weather = spark.read.option("header", True).csv(str(SILVER_DIR / "Weather Open Meteo Clean" / "Meteorologia Diaria.csv"))
    calendar = spark.read.option("header", True).csv(str(SILVER_DIR / "Calendar Nager Clean" / "Calendario Diario.csv"))

    routes_without_weather = (
        routes.alias("r")
        .join(
            weather.alias("w"),
            (F.col("r.station_code") == F.col("w.station_code")) & (F.col("r.route_date") == F.col("w.weather_date")),
            "left_anti",
        )
        .count()
    )
    routes_without_calendar = (
        routes.alias("r")
        .join(calendar.alias("c"), F.col("r.route_date") == F.col("c.calendar_date"), "left_anti")
        .count()
    )
    return {
        "routes_without_weather_match": routes_without_weather,
        "routes_without_calendar_match": routes_without_calendar,
    }


def main() -> None:
    started_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    spark = (
        SparkSession.builder.appName("SmartDeliveryAI Silver")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.local.dir", str(Path(tempfile.gettempdir()) / "smartdeliveryai_spark_local"))
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        recreate_path(SPARK_INPUT_TMP_DIR)
        amazon_counts = build_amazon_tables(spark)
        travel_counts = build_travel_time_segments(spark)
        weather_counts = build_weather_table(spark)
        calendar_counts = build_calendar_tables(spark)
        join_validation = validate_gold_join_keys(spark)
    finally:
        recreate_path(SPARK_INPUT_TMP_DIR)
        spark.stop()

    finished_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    quality_report = {
        "layer": "silver",
        "created_at_utc": finished_at,
        "started_at_utc": started_at,
        "status": "ok",
        "engine": "pyspark",
        "spark_master": "local[*]",
        "format": "csv_and_parquet",
        "note": "Silver se construye con PySpark. Se mantiene CSV para revision local y Parquet para simular almacenamiento Big Data.",
        "tables": {**amazon_counts, **weather_counts, **calendar_counts},
        "travel_times": travel_counts,
        "join_validation": join_validation,
        "expected_joins_for_gold": {
            "routes_to_weather": "routes.station_code + routes.route_date = weather_daily.station_code + weather_daily.weather_date",
            "routes_to_calendar": "routes.route_date = calendar_daily.calendar_date",
            "routes_to_sequences": "routes.route_id + stops.stop_id = actual_sequences.route_id + actual_sequences.stop_id",
            "routes_to_packages": "routes.route_id + stops.stop_id = packages.route_id + packages.stop_id",
        },
    }
    write_json(SILVER_DIR / "Source Manifests" / "Informe Calidad Silver.json", quality_report)
    print(json.dumps(quality_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
