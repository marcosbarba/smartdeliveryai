# Bake-off de arquitecturas: por qué ganó CatBoost + LightGBM

Resumen de la comparativa de arquitecturas (`src/modelado/bakeoff/`), narrada con más detalle en
`notebooks/02_modelado/bakeoff_arquitecturas.ipynb`. Las cifras de esta tabla son las del bake-off
original del proyecto (comparando arquitecturas con `GroupKFold` sobre el 100% de las rutas, antes
de fijar el split train/test de 80/20 que usa este repositorio): sirven como referencia de qué
arquitectura ganó y por qué, pero al re-ejecutar `src/modelado/bakeoff/*.py` en este repositorio
las cifras exactas pueden variar ligeramente, porque ahora el bake-off compara arquitecturas solo
dentro del 80% de `train` (ver [model_card.md](model_card.md) para la razón de este cambio).

## La tabla

| # | Arquitectura | MAE (min) | R² | Nota |
|---:|---|---:|---:|---|
| 1 | Baseline (recta por población) | 0,4473 | 0,9192 | El suelo a batir |
| 2 | LightGBM único (sin separar población) | 0,3937 | 0,9121 | Mejora MAE, **empeora R²** |
| 3 | Híbrido (2 especialistas, sin zona) | 0,3800 | 0,9349 | Arregla el R² |
| 4 | CatBoost + `to_zone_id` en reparto | 0,3721 | 0,9366 | **Ganador** |
| 5 | LSTM/GRU | — | — | Descartado, no desplegado |

## Por qué el LightGBM único empeora el R² pese a mejorar el MAE

Un único modelo con `is_depot_segment` como variable normal (no como separador estructural de dos
especialistas) reduce el error medio absoluto pero empeora la varianza explicada: el modelo
reparte su capacidad donde está la masa de datos (los tramos de reparto, >99% de las filas) y
predice peor los tramos de almacén, que son pocos pero de magnitud muy distinta (30 min de media
frente a 1 min). Un error grande en esas pocas filas pesa mucho en R² (que penaliza el error al
cuadrado) aunque apenas mueva el MAE (que promedia en valor absoluto sobre casi 900 000 filas).

## Por qué separar en dos especialistas no basta — hace falta CatBoost, no solo dos LightGBM

El híbrido (paso 3) ya separa reparto/almacén, pero sigue sin usar `to_zone_id` (cardinalidad muy
alta, ~8 960 zonas) porque one-hot no es viable y LightGBM no la trata de forma nativa sin
codificación manual (con riesgo de fuga si se codifica mal). CatBoost sí trata categóricas de alta
cardinalidad de forma nativa (*ordered target encoding*, sin fuga): al añadirla en el especialista
de reparto, el MAE y el R² mejoran juntos. Verificado además con SHAP (no solo con importancia por
ganancia): `to_zone_id` aporta un porcentaje no trivial de la magnitud media de SHAP en reparto
(cifra exacta en [model_card.md](model_card.md), recalculada por este repositorio).

## Por qué LSTM se descartó como candidato a desplegar

Experimento comparativo con PyTorch, entrenado como secuencia por ruta (sin dividir poblaciones,
con embedding de estación). En datos tabulares los árboles suelen ganar a las redes neuronales, y
aquí no fue distinto: además, un modelo de árboles permite TreeSHAP nativo (explicación exacta en
milisegundos), que una LSTM no ofrece de forma barata — y esa interpretabilidad es un requisito
real del producto (el agente necesita explicar *por qué* un tramo tarda lo que tarda), no solo
académico.

## Reproducir el bake-off en este repositorio

```text
uv run python src/modelado/bakeoff/baseline.py
uv run python src/modelado/bakeoff/lightgbm_unico.py
uv run python src/modelado/bakeoff/modelo_hibrido.py
uv run python src/modelado/bakeoff/catboost_zona.py
uv run --extra deep-learning python src/modelado/bakeoff/lstm_experimental.py
```

Cada uno lee `modelado.datos.cargar_train()` (el 80% de las rutas, nunca el test set) y escribe su
informe a `artifacts/modelado/reports/`, sus figuras a `artifacts/modelado/figuras/`, y registra en
el mismo MLflow (`artifacts/modelado/mlflow/`) que `uv run sdai-train`.
