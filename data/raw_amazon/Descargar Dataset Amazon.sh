#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://amazon-last-mile-challenges.s3.us-west-2.amazonaws.com/almrrc2021"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${ROOT_DIR}/00 Fuente Licencia Y Readme"
mkdir -p "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Build Inputs"
mkdir -p "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Apply Inputs"
mkdir -p "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Score Inputs"

curl -L -o "${ROOT_DIR}/00 Fuente Licencia Y Readme/Licencia Amazon.txt" \
  "${BASE_URL}/License.txt"

curl -L -o "${ROOT_DIR}/00 Fuente Licencia Y Readme/Readme Amazon.txt" \
  "${BASE_URL}/Readme.txt"

curl -L -o "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Build Inputs/Readme Amazon.md" \
  "${BASE_URL}/almrrc2021-data-training/model_build_inputs/readme.md"

curl -L -o "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Build Inputs/route_data.json" \
  "${BASE_URL}/almrrc2021-data-training/model_build_inputs/route_data.json"

curl -L -o "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Build Inputs/actual_sequences.json" \
  "${BASE_URL}/almrrc2021-data-training/model_build_inputs/actual_sequences.json"

curl -L -o "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Build Inputs/invalid_sequence_scores.json" \
  "${BASE_URL}/almrrc2021-data-training/model_build_inputs/invalid_sequence_scores.json"

curl -L -o "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Build Inputs/package_data.json" \
  "${BASE_URL}/almrrc2021-data-training/model_build_inputs/package_data.json"

curl -L -o "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Apply Inputs/Readme Amazon.md" \
  "${BASE_URL}/almrrc2021-data-training/model_apply_inputs/readme.md"

curl -L -o "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Apply Inputs/new_route_data.json" \
  "${BASE_URL}/almrrc2021-data-training/model_apply_inputs/new_route_data.json"

curl -L -o "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Apply Inputs/new_package_data.json" \
  "${BASE_URL}/almrrc2021-data-training/model_apply_inputs/new_package_data.json"

curl -L -o "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Apply Inputs/new_travel_times.json" \
  "${BASE_URL}/almrrc2021-data-training/model_apply_inputs/new_travel_times.json"

curl -L -o "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Score Inputs/Readme Amazon.md" \
  "${BASE_URL}/almrrc2021-data-training/model_score_inputs/readme.md"

curl -L -o "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Score Inputs/new_actual_sequences.json" \
  "${BASE_URL}/almrrc2021-data-training/model_score_inputs/new_actual_sequences.json"

curl -L -o "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Score Inputs/new_invalid_sequence_scores.json" \
  "${BASE_URL}/almrrc2021-data-training/model_score_inputs/new_invalid_sequence_scores.json"

echo "Dataset Amazon Last Mile descargado."

# travel_times.json pesa 1.8 GB y no se descarga por defecto. Hace falta solo para
# calcular la duracion de las rutas, que es la variable objetivo de la prediccion de
# tiempos. Para incluirlo:  ./Descargar Dataset Amazon.sh --con-tiempos-viaje
if [[ "${1:-}" == "--con-tiempos-viaje" ]]; then
  echo "Descargando travel_times.json (1.8 GB). Puede tardar bastante..."
  curl -L -o "${ROOT_DIR}/02 Dataset Original Amazon Last Mile/Dataset Entrenamiento 2021/Model Build Inputs/travel_times.json" \
    "${BASE_URL}/almrrc2021-data-training/model_build_inputs/travel_times.json"
  echo "travel_times.json descargado."
else
  echo "Nota: Model Build Inputs/travel_times.json no se ha descargado porque pesa aproximadamente 1.8 GB."
  echo "      Es necesario para predecir duracion de rutas. Para incluirlo:"
  echo "      ./'Descargar Dataset Amazon.sh' --con-tiempos-viaje"
fi
