"""
forecasting.py

Train, evaluate, and deploy supervised product-demand forecasting models
on the Retail Intelligence dataset.

Design principles
-----------------
• Features used at prediction time are EXACTLY those available at train
  time (same column order, same scaler, same clipping already baked in).
• Linear Regression receives a StandardScaler inside a Pipeline — this
  preserves fitted preprocessing at load time (no separate scaler file
  to keep in sync).
• Random Forest does NOT need scaling; its feature importance is logged.
• Time-series split is already done (train/val/test parquet files).
  No random shuffling. Ever.
• Future forecasting is *recursive*: we predict tomorrow using lags
  that include our own prior predictions for the forecast horizon.
• Model selection uses the VALIDATION set; test set is touched only
  once for the final reported numbers.
• Models, scaler-in-pipeline, feature list, and metrics metadata are
  saved to models/forecasting/ for the Streamlit dashboard to load.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR, BASE_DIR, RANDOM_STATE
from src.evaluation import compute_all_metrics

log = logging.getLogger(__name__)

MODELS_DIR   = BASE_DIR / "models" / "forecasting"
REPORTS_DIR  = BASE_DIR / "reports" / "forecasting"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------
# What the models are allowed to see. MUST be identical between train and
# inference, and MUST be past-only (leakage_audit already proved the lags).
FEATURE_COLS: list[str] = [
    # Past demand lags
    "lag_qty_1", "lag_qty_7", "lag_qty_14", "lag_qty_30",
    # Past revenue lags
    "lag_rev_1", "lag_rev_7", "lag_rev_14", "lag_rev_30",
    # Rolling statistics (past-only)
    "roll_mean_7", "roll_mean_14", "roll_mean_30", "roll_mean_90",
    "roll_std_7",  "roll_std_14",  "roll_std_30",  "roll_std_90",
    "roll_max_7",  "roll_max_14",  "roll_max_30",  "roll_max_90",
    # Cumulative demand (past-only)
    "cum_qty", "cum_revenue",
    # Derived trend signal (past-only)
    "demand_trend",
    # Calendar (safe — derived from Date only, no target info)
    "DayOfWeek", "Month", "Day", "Quarter",
    "WeekOfYear", "IsWeekend", "IsMonthStart", "IsMonthEnd",
    # Product-identity signal
    "AvgPrice",
]

TARGET_COL = "Quantity"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    train: pd.DataFrame
    val:   pd.DataFrame
    test:  pd.DataFrame
    feature_cols: Sequence[str]


def load_splits() -> Dataset:
    """Load the three parquet splits produced by run_eda.py."""
    tr = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    va = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    te = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    # Defensive: ensure every feature exists
    missing = [c for c in FEATURE_COLS if c not in tr.columns]
    if missing:
        raise ValueError(
            f"Missing feature columns in training data: {missing}\n"
            f"Re-run run_eda.py or update FEATURE_COLS in forecasting.py."
        )

    # Drop any residual NaNs (should be none — run_eda drops cold-starts)
    for name, df in (("train", tr), ("val", va), ("test", te)):
        before = len(df)
        df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)
        if before != len(df):
            log.warning("%s: dropped %d NaN rows", name, before - len(df))

    log.info("Splits -> train=%d val=%d test=%d",
             len(tr), len(va), len(te))
    return Dataset(tr, va, te, FEATURE_COLS)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def make_linear_regression() -> Pipeline:
    """
    LinearRegression wrapped in a Pipeline with StandardScaler.
    The scaler is *inside* the pipeline so the saved .joblib already
    knows how to transform inference inputs — no separate scaler file.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LinearRegression()),
    ])


def make_random_forest() -> RandomForestRegressor:
    """
    Random Forest Regressor with carefully chosen hyperparameters:

    • n_estimators=200     → stable, good bias/variance trade-off on
                             1.4 M rows without excessive training time.
    • max_depth=18         → RF overfits heavily at depth>25 on this
                             data. 18 keeps leaves informative but bounded.
    • min_samples_leaf=5   → prevents 1-sample leaves that memorise noise.
    • max_features="sqrt"  → classic RF default, works well with 32 features.
    • n_jobs=-1            → parallel training.
    • random_state         → reproducibility.
    """
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------
def train_and_evaluate(ds: Dataset) -> dict:
    """
    Fit LR and RF, evaluate on val + test, return a result bundle.

    Model selection: whichever model has the lower VALIDATION RMSE is
    "best". Test metrics are reported for the *chosen* model only —
    that's the honest, no-peeking protocol.
    """
    X_tr = ds.train[ds.feature_cols].astype(float).values
    y_tr = ds.train[TARGET_COL].astype(float).values
    X_va = ds.val[ds.feature_cols].astype(float).values
    y_va = ds.val[TARGET_COL].astype(float).values
    X_te = ds.test[ds.feature_cols].astype(float).values
    y_te = ds.test[TARGET_COL].astype(float).values

    results: dict = {"feature_cols": list(ds.feature_cols),
                     "models":    {},
                     "metrics":   {}}

    # ---- Linear Regression ------------------------------------------
    log.info("Training Linear Regression ...")
    lr = make_linear_regression()
    lr.fit(X_tr, y_tr)
    results["models"]["LinearRegression"] = lr
    results["metrics"]["LinearRegression"] = {
        "val":  compute_all_metrics(y_va, lr.predict(X_va)),
        "test": compute_all_metrics(y_te, lr.predict(X_te)),
    }

    # ---- Random Forest ----------------------------------------------
    log.info("Training Random Forest (this can take a few minutes) ...")
    rf = make_random_forest()
    rf.fit(X_tr, y_tr)
    results["models"]["RandomForest"] = rf
    results["metrics"]["RandomForest"] = {
        "val":  compute_all_metrics(y_va, rf.predict(X_va)),
        "test": compute_all_metrics(y_te, rf.predict(X_te)),
    }

    # ---- Choose best by VALIDATION RMSE ----------------------------
    val_rmse = {name: m["val"]["RMSE"]
                for name, m in results["metrics"].items()}
    best_name = min(val_rmse, key=val_rmse.get)
    results["best_model"] = best_name
    log.info("Best model by validation RMSE: %s  (RMSE=%.4f)",
             best_name, val_rmse[best_name])

    # ---- Persist predictions for later analysis -------------------
    results["predictions"] = {
        "val":  {"y_true": y_va,
                 "lr":   lr.predict(X_va),
                 "rf":   rf.predict(X_va)},
        "test": {"y_true": y_te,
                 "lr":   lr.predict(X_te),
                 "rf":   rf.predict(X_te)},
    }

    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_models(results: dict) -> None:
    """Save each fitted model + metadata to models/forecasting/."""
    for name, model in results["models"].items():
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, path)
        log.info("Saved model -> %s", path)

    # Save metadata (features + metrics + best model + schema version)
    meta = {
        "feature_cols":  results["feature_cols"],
        "target":        TARGET_COL,
        "best_model":    results["best_model"],
        "metrics":       results["metrics"],
        "schema_version": 1,
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(meta, indent=2, default=float), encoding="utf-8")
    log.info("Saved metadata -> %s", MODELS_DIR / "metadata.json")


def load_models() -> dict:
    """Load models + metadata. Used by the dashboard and forecast CLI."""
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    bundle = {"feature_cols": meta["feature_cols"],
              "best_model":  meta["best_model"],
              "metrics":     meta["metrics"],
              "models":      {}}
    for name in ("LinearRegression", "RandomForest"):
        path = MODELS_DIR / f"{name}.joblib"
        if path.exists():
            bundle["models"][name] = joblib.load(path)
    return bundle


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_metrics_csv(results: dict) -> Path:
    rows = []
    for model_name, splits in results["metrics"].items():
        for split, metrics in splits.items():
            row = {"model": model_name, "split": split}
            row.update(metrics)
            rows.append(row)
    df = pd.DataFrame(rows)
    path = REPORTS_DIR / "metrics.csv"
    df.to_csv(path, index=False)
    log.info("Saved metrics -> %s", path)
    print(df.round(4).to_string(index=False))
    return path


def plot_model_comparison(results: dict) -> Path:
    df = pd.DataFrame({
        (m, s): results["metrics"][m][s]["RMSE"]
        for m in results["metrics"]
        for s in ("val", "test")
    }, index=["RMSE"]).T.reset_index()
    df.columns = ["model", "split", "RMSE"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=df, x="model", y="RMSE", hue="split",
                ax=axes[0], palette="Blues")
    axes[0].set_title("RMSE by model & split")

    df2 = pd.DataFrame({
        (m, s): results["metrics"][m][s]["MAE"]
        for m in results["metrics"]
        for s in ("val", "test")
    }, index=["MAE"]).T.reset_index()
    df2.columns = ["model", "split", "MAE"]
    sns.barplot(data=df2, x="model", y="MAE", hue="split",
                ax=axes[1], palette="Greens")
    axes[1].set_title("MAE by model & split")

    fig.tight_layout()
    path = REPORTS_DIR / "model_comparison.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure -> %s", path)
    return path


def plot_actual_vs_predicted(results: dict) -> Path:
    """Scatter of actual vs predicted for both models, test split only."""
    y_true = results["predictions"]["test"]["y_true"]
    lr     = results["predictions"]["test"]["lr"]
    rf     = results["predictions"]["test"]["rf"]

    # Subsample for readability (test has ~214 k rows)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y_true), size=min(10_000, len(y_true)),
                     replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, preds, name in ((axes[0], lr, "Linear Regression"),
                            (axes[1], rf, "Random Forest")):
        ax.scatter(y_true[idx], preds[idx], alpha=0.15, s=6,
                   color="#2563EB")
        mx = max(y_true[idx].max(), preds[idx].max())
        ax.plot([0, mx], [0, mx], "--", color="black", lw=1,
                label="perfect prediction")
        ax.set_xlabel("Actual quantity")
        ax.set_ylabel("Predicted quantity")
        ax.set_title(f"{name} — test set (10k sample)")
        ax.legend()
    fig.tight_layout()
    path = REPORTS_DIR / "actual_vs_predicted.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure -> %s", path)
    return path


def plot_residuals(results: dict) -> Path:
    y_true = results["predictions"]["test"]["y_true"]
    lr     = results["predictions"]["test"]["lr"]
    rf     = results["predictions"]["test"]["rf"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, preds, name in ((axes[0], lr, "Linear Regression"),
                            (axes[1], rf, "Random Forest")):
        residuals = y_true - preds
        sns.histplot(residuals, bins=100, ax=ax, color="#10B981")
        ax.axvline(0, color="black", lw=1)
        ax.set_title(f"{name} — residual distribution")
        ax.set_xlabel("actual − predicted")
    fig.tight_layout()
    path = REPORTS_DIR / "residuals.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure -> %s", path)
    return path


def plot_feature_importance(results: dict) -> Path:
    """RF feature importance — one of the most useful diagnostic charts."""
    rf = results["models"]["RandomForest"]
    feats = results["feature_cols"]
    imp = pd.Series(rf.feature_importances_, index=feats) \
            .sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=imp.values, y=imp.index, ax=ax, palette="viridis")
    ax.set_title("Random Forest — Top 20 feature importances")
    ax.set_xlabel("importance")
    fig.tight_layout()
    path = REPORTS_DIR / "feature_importance.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure -> %s", path)
    return path


# ---------------------------------------------------------------------------
# Future forecasting (the demo-critical piece)
# ---------------------------------------------------------------------------
def forecast_future(
    product_code: str,
    horizon_days: int = 14,
    model_name: str | None = None,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Forecast the NEXT `horizon_days` days for one product using the
    best saved model.

    Parameters
    ----------
    product_code : str   StockCode to forecast
    horizon_days : int   1..30 (guard-railed)
    model_name   : str   'LinearRegression' | 'RandomForest' (None = best)
    history      : pd.DataFrame
        The full feature table (daily_product_features.parquet) — must
        include rows up to the last known date for `product_code`.

    Returns
    -------
    pd.DataFrame with columns:
        Date, predicted_quantity, model_used, product_code, Description

    The forecast is RECURSIVE:
        Day 1: use the true history lags
        Day 2: use Day 1 prediction as lag_qty_1, etc.
    """
    if not 1 <= horizon_days <= 30:
        raise ValueError("horizon_days must be between 1 and 30.")

    bundle = load_models()
    if model_name is None:
        model_name = bundle["best_model"]
    if model_name not in bundle["models"]:
        raise ValueError(f"Unknown model: {model_name}")

    model = bundle["models"][model_name]
    feature_cols = bundle["feature_cols"]

    if history is None:
        history = pd.read_parquet(
            PROCESSED_DIR / "daily_product_features.parquet")

    prod = (history[history["StockCode"] == product_code]
            .sort_values("Date").reset_index(drop=True))
    if len(prod) < 90:
        raise ValueError(
            f"Product {product_code} has only {len(prod)} historical "
            f"days — need ≥ 90 for reliable lags.")
    description = prod["Description"].iloc[-1]

    # Internal state: extended Quantity / Revenue series
    qty_series = prod["Quantity"].astype(float).tolist()
    rev_series = prod["Revenue"].astype(float).tolist()
    dates      = prod["Date"].tolist()
    last_date  = dates[-1]

    # To build features for a hypothetical day, we need a "skeleton"
    # of the last known feature row (calendar / AvgPrice etc.).
    base_row = prod.iloc[-1].to_dict()

    forecasts: list[dict] = []

    for step in range(1, horizon_days + 1):
        next_date = last_date + pd.Timedelta(days=step)

        # ---- Build features for next_date ----------------------------
        row = dict(base_row)                       # calendar / AvgPrice
        row["Date"] = next_date
        dt = pd.Timestamp(next_date)
        row["Year"]         = dt.year
        row["Month"]        = dt.month
        row["Day"]          = dt.day
        row["DayOfWeek"]    = dt.dayofweek
        row["WeekOfYear"]   = int(dt.isocalendar().week)
        row["Quarter"]      = dt.quarter
        row["IsWeekend"]    = int(dt.dayofweek >= 5)
        row["IsMonthStart"] = int(dt.is_month_start)
        row["IsMonthEnd"]   = int(dt.is_month_end)

        # Lags — for step 1 these are the true last values;
        # for step > 1 they include our own predictions.
        def lag(series, k):
            idx = len(series) - k
            return float(series[idx]) if idx >= 0 else float("nan")

        for k in (1, 7, 14, 30):
            row[f"lag_qty_{k}"] = lag(qty_series, k)
            row[f"lag_rev_{k}"] = lag(rev_series, k)

        # Rolling stats — past-only, computed on the extended series
        def roll(series, w, stat):
            past = series[-w:] if len(series) >= w else series
            if not past:
                return 0.0
            if stat == "mean": return float(np.mean(past))
            if stat == "std":  return float(np.std(past, ddof=1)) \
                                      if len(past) > 1 else 0.0
            if stat == "max":  return float(np.max(past))

        for w in (7, 14, 30, 90):
            row[f"roll_mean_{w}"] = roll(qty_series, w, "mean")
            row[f"roll_std_{w}"]  = roll(qty_series, w, "std")
            row[f"roll_max_{w}"]  = roll(qty_series, w, "max")

        row["cum_qty"]      = float(np.sum(qty_series))
        row["cum_revenue"]  = float(np.sum(rev_series))
        row["demand_trend"] = row["roll_mean_7"] - row["roll_mean_30"]

        # ---- Predict -------------------------------------------------
        X = np.array([[row[c] for c in feature_cols]], dtype=float)
        pred = float(model.predict(X)[0])
        pred = max(0.0, pred)      # quantity cannot be negative

        forecasts.append({
            "Date":               next_date,
            "product_code":       product_code,
            "Description":        description,
            "predicted_quantity": pred,
            "model_used":         model_name,
        })

        # ---- Append to internal history for next step ----------------
        qty_series.append(pred)
        rev_series.append(pred * float(base_row.get("AvgPrice", 0.0)))
        dates.append(next_date)

    return pd.DataFrame(forecasts)


# ---------------------------------------------------------------------------
# Public entry point used by run_pipeline.py / train_forecasting.py
# ---------------------------------------------------------------------------
def run_training_pipeline(verbose: bool = True) -> dict:
    """One-call training + evaluation + plotting + persistence."""
    ds = load_splits()
    results = train_and_evaluate(ds)

    write_metrics_csv(results)
    plot_model_comparison(results)
    plot_actual_vs_predicted(results)
    plot_residuals(results)
    plot_feature_importance(results)
    save_models(results)

    if verbose:
        best = results["best_model"]
        print("\n" + "=" * 72)
        print(f"BEST MODEL: {best}")
        print("=" * 72)
        for split in ("val", "test"):
            m = results["metrics"][best][split]
            print(f"{split:5s}  MAE={m['MAE']:.4f}  "
                  f"RMSE={m['RMSE']:.4f}  "
                  f"R2={m['R2']:.4f}  MAPE={m['MAPE']:.2f}%")
        print("=" * 72)

    return results