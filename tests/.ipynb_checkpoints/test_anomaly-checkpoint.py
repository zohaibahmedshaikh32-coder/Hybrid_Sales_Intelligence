# tests/test_anomaly.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.config import BASE_DIR, PROCESSED_DIR
from src.anomaly_detection import (
    FEATURE_COLS, agreement_matrix, classify_anomaly, load_models)


def test_anomaly_models_are_saved():
    mdir = BASE_DIR / "models" / "anomaly_detection"
    assert (mdir / "kmeans.joblib").exists()
    assert (mdir / "hierarchical.joblib").exists()
    assert (mdir / "isolation_forest.joblib").exists()
    assert (mdir / "metadata.json").exists()


def test_anomaly_features_present():
    df = pd.read_parquet(PROCESSED_DIR / "anomaly_features.parquet")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    assert not missing, f"Missing anomaly features: {missing}"


def test_agreement_matrix_is_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    a = (rng.random(1000) > 0.98).astype(int)
    b = (rng.random(1000) > 0.98).astype(int)
    c = (rng.random(1000) > 0.98).astype(int)
    m = agreement_matrix(a, b, c)
    # Symmetric
    assert np.allclose(m.values, m.values.T)
    # Diagonal = 1 (self-agreement)
    assert np.allclose(np.diag(m.values), 1.0)
    # Bounded [0, 1]
    assert (m.values >= 0).all() and (m.values <= 1).all()


def test_classify_anomaly_returns_known_categories():
    sample = pd.Series({
        "IsReturn": 0, "AbsRevenue": 50.0,
        "qty_zscore": 4.0, "price_zscore": 0.0,
    })
    assert classify_anomaly(sample) == "Unusually high quantity"

    sample2 = pd.Series({
        "IsReturn": 0, "AbsRevenue": 50.0,
        "qty_zscore": 0.0, "price_zscore": 3.5,
    })
    assert classify_anomaly(sample2) == "Unusually high unit price"

    sample3 = pd.Series({
        "IsReturn": 0, "AbsRevenue": 50.0,
        "qty_zscore": 0.0, "price_zscore": 0.0,
    })
    assert classify_anomaly(sample3) == \
        "Unclassified (flagged by model only)"