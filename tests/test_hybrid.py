# tests/test_hybrid.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.config import PROCESSED_DIR


def test_hybrid_deviations_exist():
    p = PROCESSED_DIR / "hybrid_deviations.parquet"
    assert p.exists(), "run python -m scripts.run_hybrid first"
    df = pd.read_parquet(p)
    for col in ("StockCode", "Date", "Actual", "Predicted",
                "Deviation", "Deviation_Pct", "Perf_Class"):
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) > 0


def test_deviation_pct_is_zero_safe():
    """Deviation_Pct must never contain inf even when Predicted=0."""
    df = pd.read_parquet(PROCESSED_DIR / "hybrid_deviations.parquet")
    assert np.isfinite(df["Deviation_Pct"]).all(), \
        "Deviation_Pct contains NaN or inf"


def test_hybrid_xref_has_anomaly_columns():
    df = pd.read_parquet(PROCESSED_DIR / "hybrid_anomaly_xref.parquet")
    for col in ("has_anomaly", "n_anomalies", "hybrid_category"):
        assert col in df.columns
    # has_anomaly must be 0 or 1
    assert set(df["has_anomaly"].unique()).issubset({0, 1})


def test_insights_json_has_all_sections():
    import json
    p = PROCESSED_DIR / "hybrid_insights.json"
    assert p.exists()
    insights = json.loads(p.read_text(encoding="utf-8"))
    for key in ("overperforming_products", "underperforming_products",
                "anomaly_hotspots", "most_unstable_demand",
                "hybrid_category_counts", "scope_note"):
        assert key in insights, f"Missing insights key: {key}"