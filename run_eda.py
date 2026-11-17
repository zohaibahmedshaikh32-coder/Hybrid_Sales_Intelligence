"""run_eda.py — end-to-end data preparation.

Usage:
    python run_eda.py
"""
from __future__ import annotations
import logging, sys
import pandas as pd
from src.config import PROCESSED_DIR
from src.data_loader import load_raw_excel, inspect_dataset, parse_dates
from src.preprocessing import clean_and_split, data_quality_report
from src.feature_engineering import (
    build_daily_product_grid, filter_forecastable_products,
    build_forecasting_features,
    build_anomaly_features, compute_product_quality)
from src.split import product_time_split
from src.leakage_audit import audit_leakage
from src.eda import run_all_eda

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("run_eda")


def _save(df, name):
    if df is None or len(df) == 0:
        log.warning("Skipping empty table: %s", name)
        return
    try:
        path = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
    except Exception:
        path = PROCESSED_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
    log.info("Saved %s -> %s", name, path.name)


def main():
    # ---- 1. Load + inspect ----------------------------------------
    raw = load_raw_excel()
    understanding = inspect_dataset(raw)
    pd.DataFrame(
        list(understanding.items()), columns=["Metric", "Value"]
    ).to_csv(PROCESSED_DIR / "dataset_understanding.csv", index=False)
    raw = parse_dates(raw)

    # ---- 2. Clean + split -----------------------------------------
    split = clean_and_split(raw)
    dq = data_quality_report(raw, split)
    dq.to_csv(PROCESSED_DIR / "data_quality_report.csv", index=False)
    print(dq.to_string(index=False))

    # ---- 3. Gap-filled daily grid + product filter ----------------
    grid = build_daily_product_grid(split.valid_sales)
    filtered, stats = filter_forecastable_products(grid)

    # Persist the PRE-CLIP grid so leakage audit + tests can use it
    _save(filtered, "grid_filtered")

    # ---- 4. Forecasting feature store (order matters!) ------------
    feat_audit, forecast_feat = build_forecasting_features(filtered)

    quality = compute_product_quality(stats)
    _save(quality, "product_quality_scores")

    # ---- 5. Anomaly feature store ---------------------------------
    anom_feat = build_anomaly_features(split.valid_sales, split.returns)

    # ---- 6. Product-time split ------------------------------------
    splits = product_time_split(forecast_feat)

    # ---- 7. Leakage audit — against the PRE-CLIP grid -------------
    audit_leakage(feat_audit, grid=filtered, sample=300)

    # ---- 8. Task-driven EDA ---------------------------------------
    run_all_eda(split.valid_sales, split.returns, grid, stats, anom_feat)

    # ---- 9. Persist all artefacts ---------------------------------
    _save(split.valid_sales, "valid_sales")
    _save(split.returns,     "returns")
    _save(split.invalid,     "invalid")
    _save(forecast_feat,     "daily_product_features")
    _save(anom_feat,         "anomaly_features")
    _save(splits.train,      "train")
    _save(splits.val,        "val")
    _save(splits.test,       "test")

    log.info("Pipeline complete.")
    log.info("Forecasting table : %s", forecast_feat.shape)
    log.info("Anomaly table     : %s", anom_feat.shape)
    log.info("Products retained : %s",
             forecast_feat["StockCode"].nunique())


if __name__ == "__main__":
    main()