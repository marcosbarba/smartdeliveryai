# notebooks/ — los documentos de decisión

```text
01_eda/eda_tiempos.ipynb                  version narrativa del EDA, ya ejecutada (el script
                                           reproducible equivalente es
                                           src/preprocessing/eda/eda_tiempos.py)
02_modelado/bakeoff_arquitecturas.ipynb   PENDIENTE: por que gano CatBoost + LightGBM sobre
                                           baseline / LightGBM unico / LSTM. Se escribe despues
                                           de ejecutar los scripts de src/modelado/bakeoff/, con
                                           numeros de comparacion reales, no antes.
```

## Qué va aquí y qué no

Solo notebooks de **exploración y decisión**, ejecutados con sus gráficos y salidas
incrustados — son el razonamiento paso a paso detrás de una decisión, no código de producción.
Los 5 scripts del bake-off de arquitecturas (`src/modelado/bakeoff/`) y los 4 del pipeline de
datos (`src/preprocessing/`) son de producción/reproducibilidad y se quedan como scripts, no se
convierten en notebooks — se ejecutan con `uv run` desde la línea de comandos, sin necesidad de
abrir Jupyter.

Para abrir estos notebooks: `uv sync --extra notebooks` y luego `uv run jupyter lab`.
