"""Prepara data/Bronze/Amazon Last Mile/Raw/ a partir del dataset ya descargado.

Los scripts de Silver esperan los cuatro JSON de Amazon dentro de Bronze y con nombres
en espanol, pero los archivos originales viven en 'data/raw_amazon' con los
nombres en ingles del reto de Amazon. Este script conecta ambos sitios.

Para no duplicar unos 465 MB se intenta crear un enlace duro (hard link), que hace que
el mismo archivo fisico aparezca en las dos rutas sin ocupar espacio extra. Si el
sistema no lo permite (por ejemplo si origen y destino estan en discos distintos) se
copia el archivo, que es mas lento y ocupa espacio pero funciona siempre.

El script es idempotente: si Bronze ya esta preparado no hace nada.

Uso:

    uv run python src/preprocessing/bronze/preparar_bronze_amazon.py

Normalmente no hace falta llamarlo a mano: 'ejecutar_pipeline_completo.py' lo ejecuta
por su cuenta si detecta que falta algun archivo en Bronze.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw_amazon"
    / "02 Dataset Original Amazon Last Mile"
    / "Dataset Entrenamiento 2021"
    / "Model Build Inputs"
)
TARGET_DIR = PROJECT_ROOT / "data" / "bronze" / "Amazon Last Mile" / "Raw"

# Nombre original del reto de Amazon -> nombre que esperan los scripts de Silver.
FILE_MAP = {
    "route_data.json": "Rutas Amazon.json",
    "package_data.json": "Paquetes Amazon.json",
    "actual_sequences.json": "Secuencias Reales Amazon.json",
    "invalid_sequence_scores.json": "Puntuaciones Secuencia Invalida Amazon.json",
    # Pesa 1,8 GB y hay que pedirlo expresamente al descargar, pero es obligatorio:
    # de aqui sale la duracion de la ruta, que es la variable objetivo del proyecto.
    "travel_times.json": "Tiempos Viaje Amazon.json",
}


def already_prepared(source: Path, target: Path) -> bool:
    """True si el destino ya apunta al mismo archivo o ya es una copia valida."""
    if not target.exists():
        return False
    try:
        if os.path.samefile(source, target):
            return True
    except OSError:
        pass
    return target.stat().st_size == source.stat().st_size


def link_or_copy(source: Path, target: Path) -> str:
    """Crea un hard link del origen en el destino. Si no se puede, copia el archivo."""
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
        return "enlazado"
    except OSError:
        shutil.copy2(source, target)
        return "copiado"


def main() -> int:
    missing_sources = [name for name in FILE_MAP if not (SOURCE_DIR / name).exists()]
    if missing_sources:
        print("ERROR: faltan archivos del dataset original de Amazon:", file=sys.stderr)
        for name in missing_sources:
            print(f"  - {(SOURCE_DIR / name).relative_to(PROJECT_ROOT).as_posix()}", file=sys.stderr)
        print(
            "\nDescargalos primero con 'data/raw_amazon/Descargar Dataset Amazon.sh'.",
            file=sys.stderr,
        )
        if "travel_times.json" in missing_sources:
            print(
                "\nOJO: travel_times.json no se descarga por defecto porque pesa 1,8 GB,\n"
                "pero es obligatorio: de el sale la duracion de la ruta, que es la\n"
                "variable objetivo. Hay que pedirlo expresamente:\n"
                "  ./'Descargar Dataset Amazon.sh' --con-tiempos-viaje",
                file=sys.stderr,
            )
        return 1

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}
    for source_name, target_name in FILE_MAP.items():
        source = SOURCE_DIR / source_name
        target = TARGET_DIR / target_name
        if already_prepared(source, target):
            results[target_name] = "ya estaba"
        else:
            results[target_name] = link_or_copy(source, target)

    print(f"Bronze Amazon preparado en {TARGET_DIR.relative_to(PROJECT_ROOT).as_posix()}")
    for target_name, action in results.items():
        print(f"  {action:10} {target_name}")

    if any(action == "copiado" for action in results.values()):
        print(
            "\nNota: algun archivo se ha copiado en vez de enlazado, probablemente porque el\n"
            "dataset original y la carpeta data estan en discos distintos. Funciona igual,\n"
            "pero ocupa espacio adicional en disco."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
