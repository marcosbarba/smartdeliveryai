from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable

BRONZE_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "preprocessing"
    / "bronze"
    / "descargar_fuentes_bronze.py"
)
SILVER_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "preprocessing"
    / "silver"
    / "construir_silver.py"
)
GOLD_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "preprocessing"
    / "gold"
    / "construir_gold.py"
)
GOLD_SEGMENTS_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "preprocessing"
    / "gold"
    / "construir_gold_tramos.py"
)
PREPARE_AMAZON_SCRIPT = (
    PROJECT_ROOT
    / "src"
    / "preprocessing"
    / "bronze"
    / "preparar_bronze_amazon.py"
)

AMAZON_REQUIRED_FILES = [
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "Amazon Last Mile"
    / "Raw"
    / "Rutas Amazon.json",
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "Amazon Last Mile"
    / "Raw"
    / "Paquetes Amazon.json",
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "Amazon Last Mile"
    / "Raw"
    / "Secuencias Reales Amazon.json",
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "Amazon Last Mile"
    / "Raw"
    / "Puntuaciones Secuencia Invalida Amazon.json",
    # De aqui sale la duracion de la ruta, que es la variable objetivo.
    PROJECT_ROOT / "data" / "bronze" / "Amazon Last Mile" / "Raw" / "Tiempos Viaje Amazon.json",
]

BRONZE_EXTERNAL_REQUIRED_FILES = [
    PROJECT_ROOT / "data" / "bronze" / "Calendar Nager" / "Raw" / "Festivos EEUU 2018.json",
    PROJECT_ROOT / "data" / "bronze" / "Source Manifests" / "Manifiesto Fuentes Bronze.json",
]

SILVER_QUALITY_REPORT = PROJECT_ROOT / "data" / "silver" / "Source Manifests" / "Informe Calidad Silver.json"
GOLD_QUALITY_REPORT = PROJECT_ROOT / "data" / "gold" / "Source Manifests" / "Informe Calidad Gold.json"
GOLD_SEGMENTS_QUALITY_REPORT = PROJECT_ROOT / "data" / "gold" / "Source Manifests" / "Informe Calidad Gold Tramos.json"
PIPELINE_REPORT = PROJECT_ROOT / "data" / "gold" / "Source Manifests" / "Informe Pipeline Completo.json"
GOLD_DATASET = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "Dataset Rutas Enriquecidas"
    / "Dataset Rutas Enriquecidas.csv"
)


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def check_files(label: str, paths: list[Path]) -> dict[str, Any]:
    missing = [path.relative_to(PROJECT_ROOT).as_posix() for path in paths if not path.exists()]
    return {
        "label": label,
        "status": "ok" if not missing else "missing",
        "missing": missing,
    }


def bronze_weather_file_count() -> int:
    weather_dir = PROJECT_ROOT / "data" / "bronze" / "Weather Open Meteo" / "Raw"
    return len(list(weather_dir.glob("Meteorologia *.json")))


def run_step(name: str, script: Path) -> dict[str, Any]:
    started_at = now_utc()
    result = subprocess.run(
        [PYTHON, str(script)],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
    )
    finished_at = now_utc()

    step = {
        "name": name,
        "script": script.relative_to(PROJECT_ROOT).as_posix(),
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "return_code": result.returncode,
        "status": "ok" if result.returncode == 0 else "error",
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {name}\n{result.stderr}\n{result.stdout}")
    return step


def validate_quality_reports() -> dict[str, Any]:
    silver = read_json(SILVER_QUALITY_REPORT)
    gold = read_json(GOLD_QUALITY_REPORT)
    segments = read_json(GOLD_SEGMENTS_QUALITY_REPORT)

    silver_join = silver.get("join_validation", {})
    gold_validation = gold.get("validation", {})
    segments_validation = segments.get("validation", {})

    checks = {
        "silver_status": silver.get("status"),
        "gold_status": gold.get("status"),
        "gold_segments_status": segments.get("status"),
        "gold_segments_rows": segments_validation.get("tramos"),
        "gold_segments_without_target": segments_validation.get("tramos_sin_objetivo"),
        "gold_segments_without_distance": segments_validation.get("tramos_sin_distancia"),
        "silver_routes_without_weather_match": silver_join.get("routes_without_weather_match"),
        "silver_routes_without_calendar_match": silver_join.get("routes_without_calendar_match"),
        "gold_routes_without_weather": gold_validation.get("routes_without_weather"),
        "gold_routes_without_calendar": gold_validation.get("routes_without_calendar"),
        "gold_routes_without_packages": gold_validation.get("routes_without_packages"),
        "gold_routes_without_stops": gold_validation.get("routes_without_stops"),
        "gold_output_rows": gold.get("output_rows"),
        "gold_output_columns": gold.get("output_columns"),
    }

    checks["status"] = (
        "ok"
        if checks["silver_status"] == "ok"
        and checks["gold_status"] == "ok"
        and checks["gold_segments_status"] == "ok"
        and checks["silver_routes_without_weather_match"] == 0
        and checks["silver_routes_without_calendar_match"] == 0
        and checks["gold_routes_without_weather"] == 0
        and checks["gold_routes_without_calendar"] == 0
        and checks["gold_routes_without_packages"] == 0
        and checks["gold_routes_without_stops"] == 0
        and checks["gold_segments_without_target"] == 0
        and checks["gold_segments_without_distance"] == 0
        else "error"
    )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline Bronze -> Silver -> Gold.")
    parser.add_argument(
        "--refresh-bronze",
        action="store_true",
        help="Vuelve a descargar fuentes externas Bronze. Necesita internet.",
    )
    parser.add_argument(
        "--skip-bronze",
        action="store_true",
        help="No ejecuta descarga Bronze externa. Falla si faltan fuentes Bronze.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = now_utc()
    steps: list[dict[str, Any]] = []

    # Bronze de Amazon no lo genera ninguna descarga: son los JSON originales colocados
    # en Bronze. Si faltan se preparan solos en vez de obligar a un paso manual.
    amazon_check = check_files("Bronze Amazon", AMAZON_REQUIRED_FILES)
    if amazon_check["status"] != "ok":
        steps.append(run_step("Preparar Bronze Amazon", PREPARE_AMAZON_SCRIPT))
        amazon_check = check_files("Bronze Amazon", AMAZON_REQUIRED_FILES)
        if amazon_check["status"] != "ok":
            raise FileNotFoundError(
                "Faltan archivos Bronze de Amazon y no se han podido preparar. "
                "Descarga el dataset original con 'data/raw_amazon/Descargar Dataset Amazon.sh'."
            )

    bronze_external_check = check_files("Bronze externo", BRONZE_EXTERNAL_REQUIRED_FILES)
    weather_count = bronze_weather_file_count()
    bronze_external_ready = bronze_external_check["status"] == "ok" and weather_count >= 17

    if args.skip_bronze and not bronze_external_ready:
        raise FileNotFoundError("Se ha usado --skip-bronze pero faltan fuentes Bronze externas.")

    if args.refresh_bronze or not bronze_external_ready:
        steps.append(run_step("Bronze externo", BRONZE_SCRIPT))
    else:
        steps.append(
            {
                "name": "Bronze externo",
                "status": "skipped",
                "reason": "Las fuentes externas Bronze ya existen. Usa --refresh-bronze para descargarlas de nuevo.",
                "weather_files": weather_count,
            }
        )

    steps.append(run_step("Construir Silver", SILVER_SCRIPT))
    steps.append(run_step("Construir Gold", GOLD_SCRIPT))
    # Tabla a nivel de tramo: es la que alimenta el modelo de tiempos de entrega.
    steps.append(run_step("Construir Gold Tramos", GOLD_SEGMENTS_SCRIPT))

    validation = validate_quality_reports()
    finished_at = now_utc()
    report = {
        "pipeline": "Bronze -> Silver -> Gold",
        "status": validation["status"],
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "steps": steps,
        "input_checks": {
            "amazon": amazon_check,
            "bronze_external_initial": bronze_external_check,
            "bronze_weather_files_initial": weather_count,
        },
        "validation": validation,
        "main_output": GOLD_DATASET.relative_to(PROJECT_ROOT).as_posix(),
        "next_step": "Realizar EDA sobre data/gold/Dataset Rutas Enriquecidas/Dataset Rutas Enriquecidas.csv",
    }
    write_json(PIPELINE_REPORT, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
