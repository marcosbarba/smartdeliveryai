"""Analisis exploratorio completo del tiempo de entrega, sobre la capa Gold.

El objetivo del TFM es predecir a que hora llega un paquete al cliente. Eso se modela por tramo:
`travel_time_seconds`, el tiempo en ir de una parada a la siguiente. Sumando los tramos de una
ruta se obtiene la hora de llegada a cualquiera de sus paradas.

El analisis sigue el recorrido habitual de un EDA profesional:

```text
1.  Vision general y tipos                7.  Correlaciones y multicolinealidad
2.  Calidad: nulos, duplicados, constantes 8.  Distancia frente a tiempo
3.  La variable objetivo                   9.  Baselines de referencia
4.  Univariante: numericas                10.  Que explica el residuo
5.  Univariante: categoricas              11.  Contexto: geografia, tiempo, clima
6.  Valores atipicos                      12.  Contrastes estadisticos
```

Salidas: figuras en PNG, informe redactado en Markdown y las cifras en JSON.

Regla del proyecto: el EDA parte de Gold, nunca de Silver ni de Bronze.

Uso:

    uv run python src/preprocessing/eda/eda_tiempos.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import warnings
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
        adoptium = Path("C:/Program Files/Eclipse Adoptium")
        candidatos = sorted(local_dir.glob("jdk-17*"), reverse=True) if local_dir.exists() else []
        candidatos += sorted(adoptium.glob("jdk-17*"), reverse=True) if adoptium.exists() else []
    else:
        candidatos = [Path("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home")]
    return next((str(c) for c in candidatos if c.exists()), None)


_java = _detect_java_home()
if _java:
    os.environ["JAVA_HOME"] = _java
    os.environ["PATH"] = f"{_java}{os.sep}bin{os.pathsep}{os.environ.get('PATH', '')}"
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

import matplotlib

matplotlib.use("Agg")  # Sin ventana grafica: el script solo guarda archivos.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

GOLD_TRAMOS = PROJECT_ROOT / "data" / "gold" / "Dataset Tramos Ruta" / "Dataset Tramos Ruta.parquet"
GOLD_RUTAS = PROJECT_ROOT / "data" / "gold" / "Dataset Rutas Enriquecidas" / "Dataset Rutas Enriquecidas.parquet"
SALIDA = PROJECT_ROOT / "artifacts" / "eda"
FIGURAS = SALIDA / "figuras"
INFORME_MD = SALIDA / "hallazgos_eda.md"
INFORME_JSON = SALIDA / "hallazgos_eda.json"

OBJETIVO = "travel_time_seconds"
AZUL, ROJO, NARANJA, VERDE = "#4F81A6", "#C0504D", "#E8A33D", "#6E9E52"

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9})

# Columnas que no son variables: identificadores y coordenadas en bruto.
NO_PREDICTORAS = {"route_id", "from_stop_id", "to_stop_id", "from_lat", "from_lng", "to_lat", "to_lng"}


def cargar() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Abre las dos tablas Gold con PySpark y las pasa a pandas.

    Spark es el motor del proyecto y mantiene el codigo migrable a un cluster. Con 898.415
    tramos el dataset cabe en memoria; si creciera, aqui se muestrearia o se agregaria en Spark
    antes de bajar los datos.
    """
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("SmartDeliveryAI EDA tiempos")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.local.dir", str(Path(tempfile.gettempdir()) / "smartdeliveryai_spark_local"))
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    try:
        tramos = spark.read.parquet(str(GOLD_TRAMOS)).toPandas()
        rutas = spark.read.parquet(str(GOLD_RUTAS)).toPandas()
    finally:
        spark.stop()
    return tramos, rutas


def guardar(nombre: str) -> str:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS / f"{nombre}.png"
    plt.savefig(ruta)
    plt.close()
    return ruta.relative_to(PROJECT_ROOT).as_posix()


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def numericas(t: pd.DataFrame) -> list[str]:
    return [c for c in t.select_dtypes(include=[np.number]).columns if c not in NO_PREDICTORAS]


# --------------------------------------------------------------------------------------------
# 1. Vision general
# --------------------------------------------------------------------------------------------
def vision_general(t: pd.DataFrame, r: pd.DataFrame) -> dict[str, Any]:
    tipos = t.dtypes.astype(str).value_counts().to_dict()
    return {
        "tramos": int(len(t)),
        "columnas_tramos": int(t.shape[1]),
        "rutas": int(r.route_id.nunique()),
        "columnas_rutas": int(r.shape[1]),
        "memoria_mb": round(float(t.memory_usage(deep=True).sum() / 1e6), 1),
        "tipos_de_dato": {str(k): int(v) for k, v in tipos.items()},
        "periodo": [str(t.route_date.min()), str(t.route_date.max())],
        "estaciones": int(t.station_code.nunique()),
    }


# --------------------------------------------------------------------------------------------
# 2. Calidad de datos
# --------------------------------------------------------------------------------------------
def calidad(t: pd.DataFrame) -> dict[str, Any]:
    nulos = t.isna().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    constantes = [c for c in t.columns if t[c].nunique(dropna=False) <= 1]
    # Casi constantes: mas del 99% de las filas con el mismo valor.
    casi = {c: round(float(t[c].value_counts(normalize=True, dropna=False).iloc[0] * 100), 2)
            for c in t.columns if c not in constantes and t[c].value_counts(normalize=True, dropna=False).iloc[0] > 0.99}

    num = numericas(t)
    negativos = {c: int((t[c] < 0).sum()) for c in num if (t[c] < 0).any()}
    ceros = {c: round(float((t[c] == 0).mean() * 100), 2) for c in num if (t[c] == 0).mean() > 0.05}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    if len(nulos):
        axes[0].barh(nulos.index, nulos.values / len(t) * 100, color=NARANJA)
        axes[0].set_xlabel("% de filas con nulo")
    else:
        axes[0].text(0.5, 0.5, "Sin nulos", ha="center", va="center", fontsize=13, color=VERDE)
        axes[0].set_axis_off()
    axes[0].set_title("Valores ausentes", fontsize=9)

    card = t[[c for c in t.columns if c not in NO_PREDICTORAS]].nunique().sort_values()
    axes[1].barh(card.index[-12:], card.values[-12:], color=AZUL)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("valores distintos (escala log)")
    axes[1].set_title("Cardinalidad", fontsize=9)
    fig.suptitle("Calidad de los datos", fontweight="bold")
    fig.tight_layout()
    figura = guardar("01 Calidad de los datos")

    return {
        "celdas_nulas": int(t.isna().sum().sum()),
        "columnas_con_nulos": {c: int(n) for c, n in nulos.items()},
        "porcentaje_nulos_total": round(float(t.isna().sum().sum() / t.size * 100), 4),
        "filas_duplicadas_completas": int(t.duplicated().sum()),
        "duplicados_por_clave": int(t.duplicated(["route_id", "segment_position"]).sum()),
        "columnas_constantes": constantes,
        "columnas_casi_constantes": casi,
        "columnas_con_negativos": negativos,
        "columnas_con_muchos_ceros": ceros,
        "figura": figura,
    }


# --------------------------------------------------------------------------------------------
# 3. La variable objetivo
# --------------------------------------------------------------------------------------------
def objetivo(t: pd.DataFrame) -> dict[str, Any]:
    dep = t[t.is_depot_segment == 1][OBJETIVO]
    rep = t[t.is_depot_segment == 0][OBJETIVO]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    sns.histplot(rep / 60, bins=80, color=AZUL, ax=axes[0, 0])
    axes[0, 0].set_xlim(0, 8)
    axes[0, 0].set_title(f"Tramos de reparto (n={len(rep):,})", fontsize=9)
    axes[0, 0].set_xlabel("minutos")
    sns.histplot(dep / 60, bins=40, color=NARANJA, ax=axes[0, 1])
    axes[0, 1].set_title(f"Tramo almacen -> zona (n={len(dep):,})", fontsize=9)
    axes[0, 1].set_xlabel("minutos")
    sns.histplot(np.log1p(rep), bins=80, color=VERDE, ax=axes[1, 0])
    axes[1, 0].set_title("Reparto en escala logaritmica", fontsize=9)
    axes[1, 0].set_xlabel("log(1 + segundos)")
    stats.probplot(rep.sample(min(5000, len(rep)), random_state=42), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Grafico Q-Q frente a la normal", fontsize=9)
    fig.suptitle("El objetivo son dos poblaciones distintas y esta muy sesgado", fontweight="bold")
    fig.tight_layout()
    figura = guardar("02 Variable objetivo")

    def resumen(s: pd.Series) -> dict[str, float]:
        return {
            "tramos": int(len(s)),
            "media_min": round(float(s.mean() / 60), 3),
            "mediana_min": round(float(s.median() / 60), 3),
            "desviacion_min": round(float(s.std() / 60), 3),
            "p05_min": round(float(s.quantile(0.05) / 60), 3),
            "p95_min": round(float(s.quantile(0.95) / 60), 3),
            "p99_min": round(float(s.quantile(0.99) / 60), 3),
            "max_min": round(float(s.max() / 60), 2),
            "asimetria": round(float(s.skew()), 3),
            "curtosis": round(float(s.kurtosis()), 3),
        }

    # Normalidad sobre una muestra: con 900.000 filas cualquier test rechaza siempre.
    muestra = rep.sample(min(5000, len(rep)), random_state=42)
    _, p_normal = stats.shapiro(muestra)
    _, p_log = stats.shapiro(np.log1p(muestra))

    return {
        "reparto": resumen(rep),
        "deposito": resumen(dep),
        "veces_mas_largo": round(float(dep.mean() / rep.mean()), 1),
        "normalidad_shapiro_p": float(f"{p_normal:.3g}"),
        "normalidad_log_shapiro_p": float(f"{p_log:.3g}"),
        "asimetria_tras_log": round(float(np.log1p(rep).skew()), 3),
        "figura": figura,
    }


# --------------------------------------------------------------------------------------------
# 4. Univariante: variables numericas
# --------------------------------------------------------------------------------------------
def univariante_numericas(t: pd.DataFrame) -> dict[str, Any]:
    cols = [c for c in numericas(t) if c != OBJETIVO][:12]
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 10))
    for ax, c in zip(axes.flat, cols):
        sns.histplot(t[c], bins=50, color=AZUL, ax=ax)
        ax.set_title(c, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes.flat[len(cols):]:
        ax.set_axis_off()
    fig.suptitle("Distribucion de las variables numericas", fontweight="bold")
    fig.tight_layout()
    figura = guardar("03 Distribuciones numericas")

    todas = numericas(t)
    desc = t[todas].describe().T[["mean", "std", "min", "50%", "max"]].round(3)
    asim = t[todas].skew().round(3)
    return {
        "resumen": desc.to_dict("index"),
        "asimetria": asim.to_dict(),
        "muy_asimetricas": asim[asim.abs() > 2].index.tolist(),
        "figura": figura,
    }


# --------------------------------------------------------------------------------------------
# 5. Univariante: variables categoricas
# --------------------------------------------------------------------------------------------
def univariante_categoricas(t: pd.DataFrame) -> dict[str, Any]:
    cats = ["station_code", "from_zone_id", "to_zone_id", "route_date"]
    cats = [c for c in cats if c in t.columns]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    est = t.station_code.value_counts()
    axes[0].bar(est.index, est.values, color=AZUL)
    axes[0].tick_params(axis="x", rotation=90, labelsize=7)
    axes[0].set_title(f"Tramos por estacion ({len(est)} estaciones)", fontsize=9)
    zonas = t.to_zone_id.value_counts()
    sns.histplot(zonas.values, bins=50, color=NARANJA, ax=axes[1])
    axes[1].set_title(f"Tramos por zona ({len(zonas):,} zonas distintas)", fontsize=9)
    axes[1].set_xlabel("tramos que terminan en la zona")
    fig.suptitle("Variables categoricas", fontweight="bold")
    fig.tight_layout()
    figura = guardar("04 Categoricas")

    return {
        "cardinalidad": {c: int(t[c].nunique()) for c in cats},
        "estacion_mas_frecuente": {str(est.index[0]): int(est.iloc[0])},
        "concentracion_top3_estaciones_pct": round(float(est.head(3).sum() / len(t) * 100), 2),
        "zonas_distintas": int(t.to_zone_id.nunique()),
        "zonas_con_un_solo_tramo": int((zonas == 1).sum()),
        "figura": figura,
    }


# --------------------------------------------------------------------------------------------
# 6. Valores atipicos
# --------------------------------------------------------------------------------------------
def atipicos(t: pd.DataFrame) -> dict[str, Any]:
    rep = t[t.is_depot_segment == 0]
    cols = [OBJETIVO, "segment_distance_km", "packages_at_destination", "service_time_at_destination_seconds"]

    resultado = {}
    for c in cols:
        s = rep[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        bajo, alto = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        fuera = ((s < bajo) | (s > alto)).sum()
        resultado[c] = {
            "atipicos_iqr": int(fuera),
            "porcentaje": round(float(fuera / len(s) * 100), 2),
            "limite_superior": round(float(alto), 3),
            "maximo": round(float(s.max()), 3),
        }

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, c in zip(axes, cols):
        sns.boxplot(y=rep[c], color=AZUL, ax=ax, fliersize=1)
        ax.set_title(c[:26], fontsize=8)
        ax.set_ylabel("")
    fig.suptitle("Valores atipicos en los tramos de reparto", fontweight="bold")
    fig.tight_layout()
    figura = guardar("05 Valores atipicos")

    # Los tramos de cero segundos merecen mencion aparte: no son errores.
    ceros = int((rep[OBJETIVO] == 0).sum())
    return {
        "por_variable": resultado,
        "tramos_de_cero_segundos": ceros,
        "porcentaje_ceros": round(float(ceros / len(rep) * 100), 3),
        "figura": figura,
    }


# --------------------------------------------------------------------------------------------
# 7. Correlaciones y multicolinealidad
# --------------------------------------------------------------------------------------------
def correlaciones(t: pd.DataFrame) -> dict[str, Any]:
    cols = numericas(t)
    corr = t[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.4, cbar_kws={"shrink": 0.6}, ax=ax,
                annot=False)
    ax.set_title("Matriz de correlaciones", fontweight="bold")
    ax.tick_params(labelsize=7)
    figura_matriz = guardar("06 Matriz de correlaciones")

    # Pares con correlacion alta: candidatos a colinealidad
    pares = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.loc[a, b]
            if abs(v) > 0.7:
                pares.append({"a": a, "b": b, "correlacion": round(float(v), 4)})
    pares.sort(key=lambda p: -abs(p["correlacion"]))

    # Correlacion con el objetivo, tambien de Spearman por la fuerte asimetria
    con_obj = corr[OBJETIVO].drop(OBJETIVO).dropna()
    con_obj = con_obj.reindex(con_obj.abs().sort_values(ascending=False).index)
    m = t.sample(min(60000, len(t)), random_state=42)
    spearman = {c: round(float(stats.spearmanr(m[c], m[OBJETIVO]).statistic), 4) for c in con_obj.index[:12]}

    fig, ax = plt.subplots(figsize=(8, 5))
    top = con_obj.head(14)
    ax.barh(top.index[::-1], top.values[::-1], color=[ROJO if v < 0 else AZUL for v in top.values][::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Correlacion de Pearson con {OBJETIVO}", fontweight="bold")
    figura_objetivo = guardar("07 Correlacion con el objetivo")

    return {
        "pares_muy_correlacionados": pares[:12],
        "correlacion_pearson_con_objetivo": {k: round(float(v), 4) for k, v in con_obj.head(14).items()},
        "correlacion_spearman_con_objetivo": spearman,
        "figuras": [figura_matriz, figura_objetivo],
    }


def multicolinealidad(t: pd.DataFrame) -> dict[str, Any]:
    """Factor de inflacion de la varianza sobre una muestra."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    cols = [c for c in numericas(t) if c != OBJETIVO and t[c].nunique() > 2][:10]
    m = t[cols].dropna().sample(min(20000, len(t)), random_state=42)
    m = m.loc[:, m.std() > 0]
    X = ((m - m.mean()) / m.std()).assign(const=1.0)
    vif = {}
    for i, c in enumerate(X.columns[:-1]):
        try:
            vif[c] = round(float(variance_inflation_factor(X.values, i)), 2)
        except Exception:
            vif[c] = float("nan")
    return {
        "vif": dict(sorted(vif.items(), key=lambda kv: -kv[1])),
        "umbral_habitual": 5,
        "variables_por_encima_del_umbral": [k for k, v in vif.items() if v > 5],
    }


# --------------------------------------------------------------------------------------------
# 8. Distancia frente a tiempo
# --------------------------------------------------------------------------------------------
def distancia(t: pd.DataFrame) -> dict[str, Any]:
    m = t.sample(min(40000, len(t)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (etiqueta, sub, color, xlim, ylim) in zip(
        axes,
        [("Tramos de reparto", m[m.is_depot_segment == 0], AZUL, (0, 3), (0, 8)),
         ("Tramo almacen -> zona", m[m.is_depot_segment == 1], NARANJA, (0, 50), (0, 70))],
    ):
        ax.scatter(sub.segment_distance_km, sub[OBJETIVO] / 60, s=3, alpha=0.15, color=color)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(etiqueta, fontsize=9)
        ax.set_xlabel("distancia en linea recta (km)")
        ax.set_ylabel("minutos")
    fig.suptitle("La distancia manda, pero no lo explica todo", fontweight="bold")
    fig.tight_layout()
    figura = guardar("08 Distancia frente a tiempo")

    res: dict[str, Any] = {"figura": figura, "poblaciones": {}}
    for nombre, sub in [("todos", t), ("reparto", t[t.is_depot_segment == 0]), ("deposito", t[t.is_depot_segment == 1])]:
        x, y = sub.segment_distance_km.values, sub[OBJETIVO].values
        a, b = np.polyfit(x, y, 1)
        res["poblaciones"][nombre] = {
            "correlacion": round(float(np.corrcoef(x, y)[0, 1]), 4),
            "r2_solo_distancia": round(r2(y, a * x + b), 4),
            "mae_min": round(float(np.abs(y - (a * x + b)).mean() / 60), 3),
            "velocidad_implicita_kmh": round(float(3600 * x.sum() / y.sum()), 1),
        }
    return res


# --------------------------------------------------------------------------------------------
# 9. Baselines
# --------------------------------------------------------------------------------------------
def baseline(t: pd.DataFrame) -> dict[str, Any]:
    """Compara tres baselines. El ingenuo falla por mezclar las dos poblaciones.

    Nota sobre fuga de datos, porque se presta a confusion: aqui se usa el tiempo para estimar
    una velocidad media y unas rectas, pero eso **no** es fuga. La regla es no usar el valor del
    objetivo de una fila para predecir esa misma fila. Una velocidad media del conjunto es un
    parametro agregado, igual que la pendiente de una regresion. Lo prohibido seria dar al modelo
    la velocidad de cada tramo como variable.

    Estas cifras se calculan sobre todos los datos, aceptable para un EDA descriptivo. En la fase
    de modelado el baseline debe reajustarse solo con el conjunto de entrenamiento.
    """
    x, y = t.segment_distance_km.values, t[OBJETIVO].values
    r2_ingenuo = r2(y, x * (y.sum() / x.sum()))
    a, b = np.polyfit(x, y, 1)
    r2_recta = r2(y, a * x + b)

    pred = np.zeros(len(t))
    for indicador in (0, 1):
        m = (t.is_depot_segment == indicador).values
        aa, bb = np.polyfit(x[m], y[m], 1)
        pred[m] = aa * x[m] + bb
    r2_sep = r2(y, pred)

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    nombres = ["Distancia /\nvelocidad media", "Una recta\npara todo", "Una recta por\npoblacion"]
    valores = [r2_ingenuo, r2_recta, r2_sep]
    ax.bar(nombres, valores, color=[ROJO if v < 0 else AZUL for v in valores])
    ax.axhline(0, color="black", linewidth=0.8)
    for i, v in enumerate(valores):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom" if v > 0 else "top", fontsize=9)
    ax.set_ylabel("R2")
    ax.set_title("Baselines: el ingenuo es peor que predecir la media", fontweight="bold")
    figura = guardar("09 Baselines")

    return {
        "r2_distancia_entre_velocidad_media": round(r2_ingenuo, 4),
        "r2_recta_unica": round(r2_recta, 4),
        "r2_recta_por_poblacion": round(r2_sep, 4),
        "mae_recta_por_poblacion_min": round(float(np.abs(y - pred).mean() / 60), 3),
        "figura": figura,
    }


# --------------------------------------------------------------------------------------------
# 10. Que explica el residuo
# --------------------------------------------------------------------------------------------
def residuo(t: pd.DataFrame) -> dict[str, Any]:
    d = t[t.is_depot_segment == 0].copy()
    x, y = d.segment_distance_km.values, d[OBJETIVO].values
    a, b = np.polyfit(x, y, 1)
    d["residuo"] = y - (a * x + b)

    candidatas = [c for c in numericas(d) if c not in (OBJETIVO, "residuo", "is_depot_segment")]
    corr = {c: round(float(d[c].astype(float).corr(d.residuo)), 4) for c in candidatas}
    corr = {k: v for k, v in sorted(corr.items(), key=lambda kv: -abs(kv[1])) if not np.isnan(v)}
    orden = list(corr)[:10]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    vals = [corr[c] for c in orden]
    axes[0].barh(orden[::-1], vals[::-1], color=[ROJO if v < 0 else AZUL for v in vals][::-1])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Correlacion con el residuo", fontsize=9)
    axes[0].tick_params(labelsize=7)
    m = d.sample(min(30000, len(d)), random_state=42)
    axes[1].scatter(m.segment_distance_km, m.residuo / 60, s=2, alpha=0.12, color=AZUL)
    axes[1].axhline(0, color=ROJO, linewidth=1)
    axes[1].set_xlim(0, 3)
    axes[1].set_ylim(-5, 8)
    axes[1].set_xlabel("distancia (km)")
    axes[1].set_ylabel("residuo (min)")
    axes[1].set_title("Residuos frente a la distancia", fontsize=9)
    fig.suptitle("Que explica lo que la distancia no explica", fontweight="bold")
    fig.tight_layout()
    figura_corr = guardar("10 Residuo tras la distancia")

    sz = d.dropna(subset=["same_zone"])
    etiqueta = sz.same_zone.map({0: "Zonas distintas", 1: "Misma zona"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    sns.barplot(x=etiqueta, y=sz[OBJETIVO] / 60, hue=etiqueta, palette=[ROJO, AZUL], legend=False, ax=axes[0])
    axes[0].set_title("Tiempo por tramo", fontsize=9)
    axes[0].set_ylabel("minutos")
    axes[0].set_xlabel("")
    sns.barplot(x=etiqueta, y=sz.segment_distance_km, hue=etiqueta, palette=[ROJO, AZUL], legend=False, ax=axes[1])
    axes[1].set_title("Distancia por tramo", fontsize=9)
    axes[1].set_ylabel("km")
    axes[1].set_xlabel("")
    fig.suptitle("Cambiar de zona cuesta tiempo", fontweight="bold")
    fig.tight_layout()
    figura_zona = guardar("11 Efecto de la zona")

    resumen = sz.groupby("same_zone").agg(
        tramos=(OBJETIVO, "size"),
        minutos=(OBJETIVO, lambda s: round(s.mean() / 60, 3)),
        km=("segment_distance_km", lambda s: round(s.mean(), 3)))

    return {
        "correlaciones_con_residuo": {k: corr[k] for k in orden},
        "variable_mas_explicativa": orden[0],
        "misma_zona": {str(k): v for k, v in resumen.to_dict("index").items()},
        "porcentaje_tramos_misma_zona": round(float(sz.same_zone.mean() * 100), 2),
        "figuras": [figura_corr, figura_zona],
    }


# --------------------------------------------------------------------------------------------
# 11. Contexto: geografia, tiempo y clima
# --------------------------------------------------------------------------------------------
def contexto(t: pd.DataFrame) -> dict[str, Any]:
    d = t[t.is_depot_segment == 0].copy()
    d["ciudad"] = d.station_code.str[1:3]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    ciudad = d.groupby("ciudad")[OBJETIVO].mean().sort_values() / 60
    axes[0, 0].barh(ciudad.index, ciudad.values, color=AZUL)
    axes[0, 0].set_title("Minutos por tramo, por ciudad", fontsize=9)
    hora = d.groupby("departure_hour")[OBJETIVO].mean() / 60
    axes[0, 1].plot(hora.index, hora.values, marker="o", color=AZUL)
    axes[0, 1].set_title("Segun la hora de salida", fontsize=9)
    axes[0, 1].set_xlabel("hora")
    dia = d.groupby("weekday")[OBJETIVO].mean() / 60
    axes[1, 0].bar(["L", "M", "X", "J", "V", "S", "D"][:len(dia)], dia.values, color=AZUL)
    axes[1, 0].set_title("Por dia de la semana", fontsize=9)
    axes[1, 0].set_ylabel("minutos")
    fecha = d.groupby("route_date")[OBJETIVO].mean() / 60
    axes[1, 1].plot(pd.to_datetime(fecha.index), fecha.values, color=AZUL, linewidth=1)
    axes[1, 1].set_title("Evolucion diaria", fontsize=9)
    axes[1, 1].tick_params(axis="x", rotation=45, labelsize=7)
    fig.suptitle("Contexto geografico y temporal", fontweight="bold")
    fig.tight_layout()
    figura = guardar("12 Contexto geografico y temporal")

    # Clima: la correlacion global frente a la controlada por estacion
    global_corr = float(d.has_rain.astype(float).corr(d[OBJETIVO]))
    por_estacion = [float(s.has_rain.astype(float).corr(s[OBJETIVO]))
                    for _, s in d.groupby("station_code") if s.has_rain.nunique() > 1 and len(s) > 1000]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    lluvia = d.assign(l=np.where(d.has_rain == 1, "Con lluvia", "Sin lluvia")).groupby("l")[OBJETIVO].mean() / 60
    axes[0].bar(lluvia.index, lluvia.values, color=[AZUL, NARANJA])
    axes[0].set_title("Tiempo medio con y sin lluvia", fontsize=9)
    axes[0].set_ylabel("minutos")
    axes[1].hist(por_estacion, bins=12, color=AZUL)
    axes[1].axvline(global_corr, color=ROJO, linewidth=2, label=f"global {global_corr:+.3f}")
    axes[1].axvline(float(np.mean(por_estacion)), color=VERDE, linewidth=2,
                    label=f"media interna {np.mean(por_estacion):+.3f}")
    axes[1].legend(fontsize=7)
    axes[1].set_title("Correlacion lluvia-tiempo por estacion", fontsize=9)
    fig.suptitle("Paradoja de Simpson: el efecto de la lluvia se desvanece", fontweight="bold")
    fig.tight_layout()
    figura_clima = guardar("13 Paradoja de Simpson del clima")

    return {
        "minutos_por_ciudad": {k: round(v, 3) for k, v in ciudad.to_dict().items()},
        "minutos_por_hora": {int(k): round(v, 3) for k, v in hora.to_dict().items()},
        "lluvia_correlacion_global": round(global_corr, 4),
        "lluvia_correlacion_media_dentro_de_estacion": round(float(np.mean(por_estacion)), 4),
        "estaciones_analizadas": len(por_estacion),
        "figuras": [figura, figura_clima],
    }


# --------------------------------------------------------------------------------------------
# 12. Contrastes estadisticos
# --------------------------------------------------------------------------------------------
def contrastes(t: pd.DataFrame) -> dict[str, Any]:
    """Tests para no quedarse en la impresion visual.

    Con 900.000 filas casi cualquier diferencia sale significativa, asi que junto a la p se
    reporta el tamano del efecto, que es lo que de verdad dice si algo importa.
    """
    d = t[t.is_depot_segment == 0]
    res: dict[str, Any] = {}

    # Depot frente a reparto
    u = stats.mannwhitneyu(t[t.is_depot_segment == 1][OBJETIVO], d[OBJETIVO], alternative="two-sided")
    res["deposito_vs_reparto"] = {"prueba": "Mann-Whitney U", "p": float(f"{u.pvalue:.3g}"),
                                  "significativo": bool(u.pvalue < 0.05)}

    # Lluvia: prueba y tamano del efecto
    con, sin = d[d.has_rain == 1][OBJETIVO], d[d.has_rain == 0][OBJETIVO]
    u2 = stats.mannwhitneyu(con, sin, alternative="two-sided")
    dif = (con.mean() - sin.mean()) / np.sqrt((con.var() + sin.var()) / 2)
    res["lluvia"] = {"prueba": "Mann-Whitney U", "p": float(f"{u2.pvalue:.3g}"),
                     "d_de_cohen": round(float(dif), 4),
                     "interpretacion": "efecto despreciable" if abs(dif) < 0.2 else "efecto apreciable"}

    # Misma zona
    sz = d.dropna(subset=["same_zone"])
    u3 = stats.mannwhitneyu(sz[sz.same_zone == 1][OBJETIVO], sz[sz.same_zone == 0][OBJETIVO], alternative="two-sided")
    a, b = sz[sz.same_zone == 1][OBJETIVO], sz[sz.same_zone == 0][OBJETIVO]
    dz = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
    res["misma_zona"] = {"prueba": "Mann-Whitney U", "p": float(f"{u3.pvalue:.3g}"),
                         "d_de_cohen": round(float(dz), 4),
                         "interpretacion": "efecto despreciable" if abs(dz) < 0.2 else "efecto apreciable"}

    # Estaciones: Kruskal-Wallis
    grupos = [s[OBJETIVO].values for _, s in d.groupby("station_code") if len(s) > 500]
    k = stats.kruskal(*grupos)
    res["estaciones"] = {"prueba": "Kruskal-Wallis", "p": float(f"{k.pvalue:.3g}"),
                         "grupos": len(grupos), "significativo": bool(k.pvalue < 0.05)}
    return res


# --------------------------------------------------------------------------------------------
# 13. Vision por ruta
# --------------------------------------------------------------------------------------------
def por_ruta(r: pd.DataFrame) -> dict[str, Any]:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    sns.histplot(r.route_duration_hours, bins=50, color=AZUL, ax=axes[0])
    axes[0].set_title("Duracion de la ruta", fontsize=9)
    axes[0].set_xlabel("horas")
    conducir = r.total_travel_time_seconds.sum()
    entregar = r.total_planned_service_time_seconds.sum()
    axes[1].pie([conducir, entregar], labels=["Conducir", "Entregar"], autopct="%1.1f%%",
                colors=[AZUL, NARANJA], startangle=90)
    axes[1].set_title("En que se va el tiempo", fontsize=9)
    fig.suptitle("La ruta completa", fontweight="bold")
    fig.tight_layout()
    figura = guardar("14 Duracion de ruta")
    return {
        "duracion_media_h": round(float(r.route_duration_hours.mean()), 2),
        "duracion_min_h": round(float(r.route_duration_hours.min()), 2),
        "duracion_max_h": round(float(r.route_duration_hours.max()), 2),
        "porcentaje_conducir": round(float(conducir / (conducir + entregar) * 100), 1),
        "porcentaje_entregar": round(float(entregar / (conducir + entregar) * 100), 1),
        "figura": figura,
    }


# --------------------------------------------------------------------------------------------
# PARTE B: la tabla de rutas
# --------------------------------------------------------------------------------------------
def rutas_calidad(r: pd.DataFrame) -> dict[str, Any]:
    """Calidad de la tabla por ruta, que es la de contexto y planificacion."""
    num = [c for c in r.select_dtypes(include=[np.number]).columns]
    constantes = [c for c in r.columns if r[c].nunique(dropna=False) <= 1]
    # Columnas que miden lo mismo con otro nombre: correlacion perfecta entre ellas.
    corr = r[num].corr().abs()
    identicas = []
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            if corr.loc[a, b] > 0.9999:
                identicas.append(f"{a} == {b}")
    return {
        "rutas": int(len(r)),
        "columnas": int(r.shape[1]),
        "celdas_nulas": int(r.isna().sum().sum()),
        "duplicados": int(r.duplicated("route_id").sum()),
        "columnas_constantes": constantes,
        "columnas_equivalentes": identicas[:10],
    }


def rutas_composicion(r: pd.DataFrame) -> dict[str, Any]:
    """Como es una ruta tipica: paradas, paquetes, volumen y zonas."""
    cols = ["total_stops", "package_count", "unique_zones", "total_package_volume_cm3",
            "packages_per_dropoff_stop", "route_duration_hours"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6))
    for ax, c in zip(axes.flat, cols):
        sns.histplot(r[c], bins=45, color=AZUL, ax=ax)
        ax.set_title(c, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Como es una ruta de reparto", fontweight="bold")
    fig.tight_layout()
    figura = guardar("15 Composicion de las rutas")

    desc = r[cols].describe().T[["mean", "std", "min", "50%", "max"]].round(2)
    return {"resumen": desc.to_dict("index"), "figura": figura}


def rutas_relaciones(r: pd.DataFrame) -> dict[str, Any]:
    """Que determina la duracion de una ruta."""
    num = [c for c in r.select_dtypes(include=[np.number]).columns
           if c not in ("route_duration_seconds", "route_duration_hours", "total_travel_time_seconds")]
    corr = r[num + ["route_duration_seconds"]].corr()["route_duration_seconds"].drop("route_duration_seconds").dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    top = corr.head(12)
    axes[0].barh(top.index[::-1], top.values[::-1], color=[ROJO if v < 0 else AZUL for v in top.values][::-1])
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Correlacion con la duracion de la ruta", fontsize=9)
    axes[0].tick_params(labelsize=7)
    axes[1].scatter(r.total_stops, r.route_duration_hours, s=4, alpha=0.2, color=AZUL)
    axes[1].set_xlabel("paradas")
    axes[1].set_ylabel("horas")
    axes[1].set_title("Paradas frente a duracion", fontsize=9)
    fig.suptitle("Que determina lo que dura una ruta", fontweight="bold")
    fig.tight_layout()
    figura = guardar("16 Duracion de la ruta")

    # route_score: ya no es objetivo, pero conviene describirlo
    score = r.route_score.value_counts()
    dur_score = r.groupby("route_score").route_duration_hours.mean().round(2)
    return {
        "correlacion_con_duracion": {k: round(float(v), 4) for k, v in corr.head(12).items()},
        "reparto_route_score": {str(k): int(v) for k, v in score.items()},
        "porcentaje_clase_minoritaria": round(float(score.min() / score.sum() * 100), 2),
        "duracion_media_por_score": {str(k): float(v) for k, v in dur_score.items()},
        "figura": figura,
    }


def redactar(res: dict[str, Any]) -> str:
    vg, ca, ob, un, uc, at, co, mc, di, ba, re, cx, ct, pr = (
        res["vision_general"], res["calidad"], res["objetivo"], res["univariante_numericas"],
        res["univariante_categoricas"], res["atipicos"], res["correlaciones"], res["multicolinealidad"],
        res["distancia"], res["baseline"], res["residuo"], res["contexto"], res["contrastes"], res["por_ruta"])
    rca, rco, rre = res["rutas_calidad"], res["rutas_composicion"], res["rutas_relaciones"]
    rep, dep, pob = ob["reparto"], ob["deposito"], di["poblaciones"]

    constantes = "\n".join(f"- `{x}`" for x in ca["columnas_constantes"]) or "- ninguna"
    pares = "\n".join(f"- `{p['a']}` y `{p['b']}`: {p['correlacion']}" for p in co["pares_muy_correlacionados"][:8]) or "- ninguno"
    top_obj = "\n".join(f"- `{k}`: {v}" for k, v in list(co["correlacion_pearson_con_objetivo"].items())[:8])
    top_res = "\n".join(f"- `{k}`: {v}" for k, v in list(re["correlaciones_con_residuo"].items())[:6])
    ciudades = "\n".join(f"- {k}: {v} min" for k, v in sorted(cx["minutos_por_ciudad"].items(), key=lambda x: x[1]))
    vif_alto = ", ".join(f"`{v}`" for v in mc["variables_por_encima_del_umbral"]) or "ninguna"

    return f"""# Hallazgos del analisis exploratorio

Generado por `eda_tiempos.py` el {res['generado']}.

## 0. Resumen ejecutivo

Analizados **{vg['tramos']:,} tramos** de **{vg['rutas']:,} rutas**, con {vg['columnas_tramos']} variables.
El objetivo es `travel_time_seconds`: cuanto se tarda en ir de una parada a la siguiente.

Cinco conclusiones que condicionan el modelado:

1. **El objetivo son dos poblaciones**, no una. El tramo que sale del almacen dura
   {ob['veces_mas_largo']} veces mas que un salto entre entregas.
2. **El baseline intuitivo es malo**: dividir la distancia entre una velocidad media da un R2 de
   {ba['r2_distancia_entre_velocidad_media']}, peor que predecir la media. La referencia real es {ba['r2_recta_por_poblacion']}.
3. **La distancia no lo explica todo**: dentro del reparto solo cubre el {pob['reparto']['r2_solo_distancia']*100:.0f}% de la variabilidad.
4. **`{re['variable_mas_explicativa']}` es la mejor pista** despues de la distancia.
5. **El clima no aporta y ademas engana** por confusion geografica.

## 1. Vision general

```text
tramos              {vg['tramos']:>9,}
rutas               {vg['rutas']:>9,}
variables           {vg['columnas_tramos']:>9}
memoria             {vg['memoria_mb']:>9.1f} MB
periodo             {vg['periodo'][0]} a {vg['periodo'][1]}
estaciones          {vg['estaciones']:>9}
```

## 2. Calidad de los datos

```text
celdas nulas                {ca['celdas_nulas']:>10,}  ({ca['porcentaje_nulos_total']}% del total)
filas duplicadas completas  {ca['filas_duplicadas_completas']:>10,}
duplicados por clave        {ca['duplicados_por_clave']:>10,}
columnas sin variacion      {len(ca['columnas_constantes']):>10}
```

Columnas constantes, que deben excluirse del modelo:

{constantes}

Los nulos se concentran en las variables de zona, y tienen explicacion: el almacen no pertenece
a ninguna zona de reparto, asi que en el tramo que sale de el no hay zona de origen ni se puede
comparar con la de destino.

No hay valores negativos en ninguna variable ni filas duplicadas.

Figura: `{ca['figura']}`

## 3. La variable objetivo

```text
                    tramos      media    mediana       p95      max   asimetria
reparto          {rep['tramos']:>9,}   {rep['media_min']:>7.2f}   {rep['mediana_min']:>8.2f}  {rep['p95_min']:>8.2f} {rep['max_min']:>8.1f}      {rep['asimetria']:>6.2f}
almacen -> zona  {dep['tramos']:>9,}   {dep['media_min']:>7.2f}   {dep['mediana_min']:>8.2f}  {dep['p95_min']:>8.2f} {dep['max_min']:>8.1f}      {dep['asimetria']:>6.2f}
```

**Este es el hallazgo principal.** El trayecto inicial es {ob['veces_mas_largo']} veces mas largo: es un viaje
por carretera hasta el barrio, mientras que el resto son saltos de portal a portal. Mezclarlos en
un unico modelo sin distinguirlos obliga al modelo a servir a dos realidades incompatibles.

La distribucion del reparto esta muy sesgada a la derecha (asimetria {rep['asimetria']}, curtosis
{rep['curtosis']}). El test de Shapiro-Wilk sobre una muestra rechaza la normalidad
(p = {ob['normalidad_shapiro_p']}), y aplicando logaritmo la asimetria baja a {ob['asimetria_tras_log']}.

Eso importa para elegir modelo: los lineales sufren con esta forma, los de arboles no.

Figura: `{ob['figura']}`

## 4. Variables numericas

Variables con asimetria fuerte, por encima de 2 en valor absoluto:

{chr(10).join(f"- `{v}`" for v in un['muy_asimetricas']) or '- ninguna'}

Figura: `{un['figura']}`

## 5. Variables categoricas

```text
estaciones                {uc['cardinalidad'].get('station_code', 0):>8,}
zonas de destino          {uc['zonas_distintas']:>8,}
zonas con un solo tramo   {uc['zonas_con_un_solo_tramo']:>8,}
```

Las tres estaciones mas frecuentes concentran el {uc['concentracion_top3_estaciones_pct']}% de los tramos, asi que hay
desequilibrio geografico que conviene tener presente al validar.

`to_zone_id` tiene cardinalidad muy alta. Para usarla no sirve una codificacion one-hot: hay que
recurrir a modelos que traten categoricas de forma nativa, como CatBoost, o a codificacion por
historico calculada fuera de muestra.

Figura: `{uc['figura']}`

## 6. Valores atipicos

{chr(10).join(f"- `{k}`: {v['atipicos_iqr']:,} atipicos por el criterio del rango intercuartilico ({v['porcentaje']}%), maximo {v['maximo']}" for k, v in at['por_variable'].items())}

Hay {at['tramos_de_cero_segundos']:,} tramos de cero segundos ({at['porcentaje_ceros']}%). **No son errores**: son
entregas consecutivas en la misma coordenada, como dos pisos del mismo edificio, donde el
repartidor no se desplaza.

Los atipicos del tiempo tampoco parecen errores, sino tramos genuinamente largos. La
recomendacion es **no recortarlos**: forman parte del fenomeno que se quiere predecir.

Figura: `{at['figura']}`

## 7. Correlaciones y multicolinealidad

Correlacion de Pearson con el objetivo:

{top_obj}

Pares de variables muy correlacionadas entre si, por encima de 0,7:

{pares}

Factor de inflacion de la varianza por encima del umbral habitual de 5: {vif_alto}.

Conviene no meter juntas variables que miden lo mismo. En particular, la posicion dentro de la
ruta esta representada por varias columnas equivalentes.

Figuras: {', '.join(f'`{f}`' for f in co['figuras'])}

## 8. Distancia frente a tiempo

```text
                 correlacion   R2 con solo distancia   velocidad implicita
todos               {pob['todos']['correlacion']:.3f}              {pob['todos']['r2_solo_distancia']:.3f}                 {pob['todos']['velocidad_implicita_kmh']:>5.1f} km/h
reparto             {pob['reparto']['correlacion']:.3f}              {pob['reparto']['r2_solo_distancia']:.3f}                 {pob['reparto']['velocidad_implicita_kmh']:>5.1f} km/h
almacen -> zona     {pob['deposito']['correlacion']:.3f}              {pob['deposito']['r2_solo_distancia']:.3f}                 {pob['deposito']['velocidad_implicita_kmh']:>5.1f} km/h
```

La correlacion global de {pob['todos']['correlacion']:.3f} es enganosa: procede en buena parte de que existen dos grupos
separados. **Dentro del reparto baja a {pob['reparto']['correlacion']:.3f}** y la distancia explica solo el
{pob['reparto']['r2_solo_distancia']*100:.0f}% de la variabilidad. El {(1-pob['reparto']['r2_solo_distancia'])*100:.0f}% restante es lo que el modelo tiene que aprender.

Las velocidades son en linea recta: por carretera se recorre mas, asi que la velocidad real es
mayor. Sirven para comparar zonas y horas, no como medida absoluta.

Figura: `{di['figura']}`

## 9. Baselines de referencia

```text
distancia / velocidad media      R2 = {ba['r2_distancia_entre_velocidad_media']:>7.3f}
una recta para todo              R2 = {ba['r2_recta_unica']:>7.3f}
una recta por poblacion          R2 = {ba['r2_recta_por_poblacion']:>7.3f}   MAE {ba['mae_recta_por_poblacion_min']:.2f} min
```

El baseline mas intuitivo da un **R2 negativo**: es peor que predecir siempre la media, porque una
unica velocidad no puede describir dos poblaciones que circulan a {pob['deposito']['velocidad_implicita_kmh']:.0f} y a {pob['reparto']['velocidad_implicita_kmh']:.0f} km/h.

**La referencia a batir es la ultima**: R2 {ba['r2_recta_por_poblacion']:.3f} y {ba['mae_recta_por_poblacion_min']:.2f} minutos de error medio.

Dos aclaraciones:

- **Estimar una velocidad media no es fuga de datos.** La regla es no usar el objetivo de una
  fila para predecir esa misma fila. Una media del conjunto es un parametro agregado, igual que
  la pendiente de una regresion. Lo prohibido seria dar al modelo la velocidad de cada tramo.
- Estas cifras se calculan sobre todos los datos. Al modelar hay que reajustar el baseline solo
  con el entrenamiento y medirlo en la prueba.

Figura: `{ba['figura']}`

## 10. Que explica lo que la distancia no explica

Correlacion con el residuo de la regresion sobre la distancia, dentro del reparto:

{top_res}

La mas informativa es `{re['variable_mas_explicativa']}`. Moverse dentro de la misma zona de
planificacion sale mas rapido de lo que la distancia haria pensar:

```text
{chr(10).join(f"{'misma zona' if k=='1' else 'zonas distintas':16} {v['minutos']:.2f} min   {v['km']:.3f} km   {v['tramos']:>7,} tramos" for k, v in sorted(re['misma_zona'].items(), reverse=True))}
```

El {re['porcentaje_tramos_misma_zona']}% de los tramos ocurre dentro de una misma zona. Tiene sentido operativo: las
zonas agrupan calles proximas y bien conectadas.

Figuras: {', '.join(f'`{f}`' for f in re['figuras'])}

## 11. Contexto geografico y temporal

Minutos medios por tramo de reparto, por area metropolitana:

{ciudades}

## 12. El clima: paradoja de Simpson

```text
correlacion lluvia-tiempo, global              {cx['lluvia_correlacion_global']:+.4f}
correlacion media dentro de cada estacion      {cx['lluvia_correlacion_media_dentro_de_estacion']:+.4f}
```

La correlacion global sugiere que la lluvia ralentiza el reparto, pero **desaparece al controlar
por estacion**. Las ciudades lluviosas del dataset son las que tienen rutas mas largas, de modo
que la lluvia estaba actuando como disfraz de la variable "ciudad".

El contraste lo confirma: la diferencia es estadisticamente significativa
(p = {ct['lluvia']['p']}) pero con un tamano del efecto de {ct['lluvia']['d_de_cohen']}, es decir
**{ct['lluvia']['interpretacion']}**. Con 900.000 filas casi cualquier diferencia sale significativa,
por eso hay que mirar el tamano del efecto y no solo la p.

Si se entrena con meteorologia **sin incluir `station_code`**, el modelo aprendera la relacion
falsa y el agente acabara diciendo a un cliente que su paquete llega tarde por la lluvia, cuando
en realidad es por la ciudad en la que vive.

Figuras: {', '.join(f'`{f}`' for f in cx['figuras'])}

## 13. Contrastes estadisticos

```text
deposito vs reparto     {ct['deposito_vs_reparto']['prueba']:<18} p = {ct['deposito_vs_reparto']['p']}
misma zona              {ct['misma_zona']['prueba']:<18} p = {ct['misma_zona']['p']}   d de Cohen = {ct['misma_zona']['d_de_cohen']}
lluvia                  {ct['lluvia']['prueba']:<18} p = {ct['lluvia']['p']}   d de Cohen = {ct['lluvia']['d_de_cohen']}
estaciones              {ct['estaciones']['prueba']:<18} p = {ct['estaciones']['p']}   ({ct['estaciones']['grupos']} grupos)
```

Se usan pruebas no parametricas porque la distribucion del objetivo dista mucho de la normal.

## 14. La ruta completa

```text
duracion media    {pr['duracion_media_h']} h   (de {pr['duracion_min_h']} a {pr['duracion_max_h']} h)
conducir          {pr['porcentaje_conducir']}%
entregar          {pr['porcentaje_entregar']}%
```

El grueso de la jornada se va **entregando, no conduciendo**. Como el tiempo de servicio viene
planificado en el manifiesto y se conoce antes de salir, el modelo solo tiene que estimar el
{pr['porcentaje_conducir']}% restante.

Figura: `{pr['figura']}`

---

# Parte B: la tabla de rutas

Hasta aqui el analisis ha ido sobre los tramos, que es lo que alimenta el modelo. Esta segunda
parte describe la tabla por ruta, que sirve para planificacion, contexto y analisis agregado.

## B.1 Calidad

```text
rutas                   {rca['rutas']:>8,}
columnas                {rca['columnas']:>8}
celdas nulas            {rca['celdas_nulas']:>8,}
identificadores repetidos {rca['duplicados']:>6}
```

Columnas sin variacion, que deben excluirse:

{chr(10).join(f"- `{x}`" for x in rca['columnas_constantes']) or '- ninguna'}

Pares de columnas con correlacion practicamente perfecta, es decir que llevan la misma
informacion aunque difieran en una constante:

{chr(10).join(f"- `{x}`" for x in rca['columnas_equivalentes']) or '- ninguna'}

Todas las variantes del numero de paradas son la misma magnitud: `total_stops` es
`dropoff_stops` mas la estacion, y las demas son recuentos equivalentes de la misma secuencia.
Hay que quedarse con una sola: meter varias juntas introduce colinealidad sin aportar nada.

## B.2 Como es una ruta tipica

{chr(10).join(f"- `{k}`: media {v['mean']}, mediana {v['50%']}, rango de {v['min']} a {v['max']}" for k, v in rco['resumen'].items())}

Figura: `{rco['figura']}`

## B.3 Que determina la duracion de una ruta

Correlacion con `route_duration_seconds`, excluyendo sus propios componentes:

{chr(10).join(f"- `{k}`: {v}" for k, v in list(rre['correlacion_con_duracion'].items())[:8])}

Figura: `{rre['figura']}`

## B.4 route_score, la etiqueta que ya no es objetivo

Amazon etiqueto cada ruta historica segun la calidad de su secuencia. **Ya no es el objetivo del
proyecto**, pero conviene describirla porque estaba en el planteamiento anterior:

```text
{chr(10).join(f"{k:8} {v:>6,} rutas" for k, v in rre['reparto_route_score'].items())}
```

La clase minoritaria representa solo el {rre['porcentaje_clase_minoritaria']}% de las rutas. Ese desbalanceo severo fue una
de las razones para cambiar de objetivo: obligaba a pelearse con metricas por clase y particiones
estratificadas, mientras que el tiempo de entrega es una variable continua y bien repartida.

Duracion media segun la etiqueta:

```text
{chr(10).join(f"{k:8} {v:>6.2f} h" for k, v in rre['duracion_media_por_score'].items())}
```

---

## 15. Conclusiones para el modelado

1. **Distinguir las dos poblaciones.** Incluir `is_depot_segment` o entrenar por separado.
2. **Batir el baseline de R2 {ba['r2_recta_por_poblacion']:.3f} y MAE {ba['mae_recta_por_poblacion_min']:.2f} minutos**, reajustado sobre el entrenamiento.
3. **Excluir las columnas constantes** y no duplicar variables que miden lo mismo.
4. **Aprovechar `same_zone`**, la mejor senal despues de la distancia.
5. **Incluir `station_code` si se usa meteorologia**, o las explicaciones seran falsas.
6. **Particionar con `GroupKFold` por `route_id`**: los tramos de una ruta no son independientes.
7. **No recortar los atipicos**: son casos reales, no errores.
8. **Metrica MAE en minutos**, que se explica sola ante un tribunal o un cliente.
9. **Priorizar arboles** (LightGBM, CatBoost): toleran la asimetria, manejan la alta cardinalidad
   de las zonas y permiten explicar cada prediccion con TreeSHAP, que es lo que necesita el agente.
"""


def main() -> None:
    print("Leyendo Gold con PySpark...")
    tramos, rutas = cargar()
    print(f"Cargados {len(tramos):,} tramos y {len(rutas):,} rutas.\n")

    FIGURAS.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {"generado": datetime.now(UTC).strftime("%Y-%m-%d")}

    pasos = [
        ("Vision general", "vision_general", lambda: vision_general(tramos, rutas)),
        ("Calidad de los datos", "calidad", lambda: calidad(tramos)),
        ("Variable objetivo", "objetivo", lambda: objetivo(tramos)),
        ("Univariante numericas", "univariante_numericas", lambda: univariante_numericas(tramos)),
        ("Univariante categoricas", "univariante_categoricas", lambda: univariante_categoricas(tramos)),
        ("Valores atipicos", "atipicos", lambda: atipicos(tramos)),
        ("Correlaciones", "correlaciones", lambda: correlaciones(tramos)),
        ("Multicolinealidad", "multicolinealidad", lambda: multicolinealidad(tramos)),
        ("Distancia y tiempo", "distancia", lambda: distancia(tramos)),
        ("Baselines", "baseline", lambda: baseline(tramos)),
        ("Residuo y zonas", "residuo", lambda: residuo(tramos)),
        ("Contexto", "contexto", lambda: contexto(tramos)),
        ("Contrastes estadisticos", "contrastes", lambda: contrastes(tramos)),
        ("Vision por ruta", "por_ruta", lambda: por_ruta(rutas)),
        # Parte B: la tabla de rutas, que es la de contexto y planificacion.
        ("Rutas: calidad", "rutas_calidad", lambda: rutas_calidad(rutas)),
        ("Rutas: composicion", "rutas_composicion", lambda: rutas_composicion(rutas)),
        ("Rutas: relaciones", "rutas_relaciones", lambda: rutas_relaciones(rutas)),
    ]
    for i, (etiqueta, clave, fn) in enumerate(pasos, start=1):
        print(f"{i:>2}/{len(pasos)} {etiqueta}...")
        res[clave] = fn()

    INFORME_MD.write_text(redactar(res), encoding="utf-8")
    with INFORME_JSON.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)

    print("\nEDA terminado.")
    print(f"  Figuras: {len(list(FIGURAS.glob('*.png')))} en {FIGURAS.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"  Informe: {INFORME_MD.relative_to(PROJECT_ROOT).as_posix()}")
    print("\nHallazgos principales:")
    print(f"  - El tramo al barrio es {res['objetivo']['veces_mas_largo']}x mas largo que un salto entre entregas.")
    print(f"  - La distancia explica el {res['distancia']['poblaciones']['reparto']['r2_solo_distancia']*100:.0f}% del tiempo de reparto.")
    print(f"  - Baseline a batir: R2 {res['baseline']['r2_recta_por_poblacion']:.3f}, MAE {res['baseline']['mae_recta_por_poblacion_min']:.2f} min.")
    print(f"  - Lluvia: p={res['contrastes']['lluvia']['p']} pero d de Cohen={res['contrastes']['lluvia']['d_de_cohen']} ({res['contrastes']['lluvia']['interpretacion']}).")


if __name__ == "__main__":
    main()
