"""Metricas de regresion compartidas por el bake-off y el entrenamiento final."""

from __future__ import annotations

import numpy as np


def metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_real - y_pred
    ss_tot = float(np.sum((y_real - y_real.mean()) ** 2))
    return {
        "mae_min": float(np.mean(np.abs(error)) / 60),
        "rmse_min": float(np.sqrt(np.mean(error**2)) / 60),
        "r2": 1 - float(np.sum(error**2)) / ss_tot if ss_tot > 0 else float("nan"),
    }
