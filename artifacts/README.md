# artifacts/ — todo lo que generan los scripts, nada escrito a mano

```text
eda/        figuras + informe del análisis exploratorio (src/preprocessing/eda/eda_tiempos.py)
modelado/   reports/figuras/mlflow del bake-off de arquitecturas (src/modelado/bakeoff/*.py)
modelo/     el modelo entrenado — reparto_catboost.cbm, almacen_lightgbm.txt, manifiesto.json,
            model_card.md, informe_shap.md (src/modelado/entrenamiento_final.py, uv run sdai-train)
```

## Regla

`docs/` es lo que un humano redacta a mano (decisiones, contexto, limitaciones); `artifacts/` es
lo que un script genera. Si un fichero de aquí parece "hecho a mano", es un bug — se regenera
ejecutando el script correspondiente, nunca se edita directamente.

## Por qué `modelo/` es singular (un solo modelo, no "oficial" + "del agente")

El diseño anterior de este proyecto mantenía dos modelos entrenados por separado: uno sobre el
100% de las rutas (sin test set propio) para la memoria, y otro sobre un 90% (reservando un 10%
aparte) para la demo del agente. Aquí hay un único split fijo (80% train / 20% test, ver
`docs/modelado/model_card.md`) y un único modelo: el mismo test set sirve a la vez de métrica de
generalización reportada y de banco de "producción simulada" del agente
(`data/produccion_simulada/`) — no hay razón para mantener dos artefactos ni dos historias
distintas de cómo se entrenó cada uno.

## Tamaños

```text
modelo/reparto_catboost.cbm   varios cientos de MB — gitignored, no cabe en git sin LFS
modelo/almacen_lightgbm.txt   menos de 1 MB
modelado/mlflow/mlflow.db     ~1 MB, backend SQLite de tracking — gitignored, regenerable
```
