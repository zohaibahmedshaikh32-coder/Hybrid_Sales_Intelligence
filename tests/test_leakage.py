# tests/test_leakage.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import PROCESSED_DIR
from src.leakage_audit import audit_leakage


def test_no_leakage():
    """
    Verify that every lag/rolling feature is past-only.

    CRITICAL: we pass the ORIGINAL pre-clip grid (`grid_filtered.parquet`).
    Without it, the audit reconstructs each product's series from the
    already-trimmed `feat` table and produces false positives near
    cold-start boundaries.
    """
    feat = pd.read_parquet(PROCESSED_DIR / "daily_product_features.parquet")
    grid = pd.read_parquet(PROCESSED_DIR / "grid_filtered.parquet")
    assert audit_leakage(feat, grid=grid, sample=200) is True


def test_split_is_chronological():
    tr = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    va = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    te = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    for code in tr["StockCode"].unique():
        t = tr[tr["StockCode"] == code]["Date"].max()
        v = va[va["StockCode"] == code]["Date"]
        s = te[te["StockCode"] == code]["Date"]
        if len(v):
            assert v.min() > t
        if len(s) and len(v):
            assert s.min() > v.max()


def test_all_splits_nonempty():
    for name in ("train", "val", "test"):
        df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
        assert len(df) > 0, f"{name} split is empty"
        assert df["StockCode"].nunique() > 100, f"{name} has too few products"


def test_lag_features_exist():
    feat = pd.read_parquet(PROCESSED_DIR / "daily_product_features.parquet")
    expected = [
        "lag_qty_1", "lag_qty_7", "lag_qty_14", "lag_qty_30",
        "roll_mean_7", "roll_mean_14", "roll_mean_30", "roll_mean_90",
        "roll_std_7", "roll_std_30", "cum_qty", "demand_trend",
    ]
    missing = [c for c in expected if c not in feat.columns]
    assert not missing, f"Missing feature columns: {missing}"

def test_models_are_saved():
    from src.config import BASE_DIR
    models_dir = BASE_DIR / "models" / "forecasting"
    assert (models_dir / "LinearRegression.joblib").exists()
    assert (models_dir / "RandomForest.joblib").exists()
    assert (models_dir / "metadata.json").exists()


def test_forecast_returns_finite_predictions():
    """Ensure recursive forecast produces sane, non-negative values."""
    from src.forecasting import forecast_future
    import numpy as np
    fc = forecast_future("22362", horizon_days=5)
    assert len(fc) == 5
    assert (fc["predicted_quantity"] >= 0).all()
    assert np.isfinite(fc["predicted_quantity"]).all()