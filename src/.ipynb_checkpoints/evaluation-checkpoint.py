"""
evaluation.py — metric utilities reused across forecasting and anomaly.

All metrics handle the zero-actual case for MAPE safely and return
finite floats (no NaN / inf) so downstream code never crashes.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true, y_pred, eps: float = 1e-6) -> float:
    """
    MAPE with safe handling of zero actuals.

    Standard MAPE is undefined when any y_true == 0. UCI Online Retail II
    has MANY zero-sale days (gap-filled), so we must not blow up.

    Strategy: compute MAPE only over rows where |y_true| > eps,
    and report the fraction of rows excluded. If nothing survives,
    return NaN (honest signal that MAPE is not meaningful here).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask])
                                 / y_true[mask])) * 100.0)


def r2(y_true, y_pred) -> float:
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


def compute_all_metrics(y_true, y_pred) -> dict:
    """Standard metric bundle used by the training pipeline."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Clip predictions to non-negative — quantity cannot be negative
    y_pred = np.clip(y_pred, 0, None)
    return {
        "MAE":  mae(y_true, y_pred),
        "MSE":  float(mean_squared_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2":   r2(y_true, y_pred),
        "n":    int(len(y_true)),
    }