# data/ — capas del pipeline, de crudo a transformado

```text
raw_amazon/          dataset origen (Amazon Last Mile). Solo licencia/diccionario/script aquí —
                     el JSON pesado (route_data, package_data, travel_times: ~1.7GB) se descarga
                     con Descargar Dataset Amazon.sh, no se versiona en git.
bronze/               fuentes originales casi sin tocar (Amazon + meteorología Open-Meteo +
                      festivos Nager.Date), generado por src/preprocessing/bronze/
silver/               datos limpios y tipados, generado por src/preprocessing/silver/
gold/                 dataset final (dos tablas: Dataset Tramos Ruta + Dataset Rutas Enriquecidas),
                      generado por src/preprocessing/gold/
produccion_simulada/  el 20% de rutas de test, generado por src/modelado/entrenamiento_final.py
                      (uv run sdai-train) — es a la vez la métrica de generalización del modelo
                      y el banco de "producción simulada" del agente (ver docs/modelado/)
```

## Regla

Nada de esto se edita a mano. `bronze/`, `silver/`, `gold/` se regeneran con `uv run
sdai-pipeline`; `produccion_simulada/` con `uv run sdai-train`. Los contratos de datos completos
(esquema de cada tabla, avisos de fuga de datos, qué columnas no usar como predictoras) están en
`docs/preprocessing/contratos_de_datos.md`, no repetidos aquí.

## Por qué el primer nivel está en minúsculas pero el resto se mantiene en español

Al migrar este proyecto desde su estructura original se renombró solo el segmento superior de
cada capa (`Data` → `data`, `Bronze` → `bronze`, `Silver` → `silver`, `Gold` → `gold`, `Datos
Simulados Produccion` → `produccion_simulada`), siguiendo la convención Python de minúsculas para
carpetas de primer nivel. Los nombres por debajo de ese primer nivel (`Amazon Last Mile`,
`Dataset Tramos Ruta`, `Source Manifests`, etc.) se mantienen tal cual porque decenas de scripts y
los contratos de datos ya versionados en `docs/` los citan letra a letra — renombrarlos
multiplicaría el riesgo de la migración sin ningún beneficio funcional.

## Tamaños (aproximados, no versionados en git salvo manifiestos/contratos)

```text
raw_amazon/   ~1.7 GB con el bloque de tiempos de viaje descargado (gitignored)
bronze/       ~440 KB (JSON de fuentes externas + manifiestos)
silver/       ~200 MB (CSV + Parquet, gitignored, regenerable)
gold/         ~60 MB (CSV + Parquet, gitignored, regenerable; contratos .md sí versionados)
produccion_simulada/  unos pocos MB (gitignored, regenerable)
```
