"""
hybrid_analysis.py

The integration layer of the Retail Intelligence system.

Combines:
    - Forecasting model predictions (src.forecasting, Phase 4)
    - Anomaly detection flags (src.anomaly_detection, Phase 5)

Into a single, interpretable hybrid signal:

    Deviation   = Actual - Predicted
    Deviation % = Deviation / max(|Predicted|, eps) x 100

IMPORTANT - scope of claims
---------------------------
- Deviation is an ANALYTICAL SIGNAL, not proof of any business event.
- Anomalies detected by unsupervised models are "unusual patterns",
  NOT confirmed fraud or error.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PROCESSED_DIR, BASE_DIR, RANDOM_STATE

log = logging.getLogger(__name__)

REPORTS_DIR = BASE_DIR / "reports" / "hybrid_analysis"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Symmetric band for over/under-performance classification.
DEVIATION_BAND_PCT = 0.25


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    forecast_df = pd.read_parquet(
        PROCESSED_DIR / "daily_product_features.parquet")
    anomaly_df = pd.read_parquet(
        PROCESSED_DIR / "anomaly_features.parquet")

    fc_meta = json.loads(
        (BASE_DIR / "models" / "forecasting" / "metadata.json")
        .read_text(encoding="utf-8"))
    an_meta = json.loads(
        (BASE_DIR / "models" / "anomaly_detection" / "metadata.json")
        .read_text(encoding="utf-8"))

    models = {
        "forecasting": {
            "metadata": fc_meta,
            "linear": joblib.load(
                BASE_DIR / "models" / "forecasting" / "LinearRegression.joblib"),
            "rf": joblib.load(
                BASE_DIR / "models" / "forecasting" / "RandomForest.joblib"),
        },
        "anomaly": {
            "metadata": an_meta,
            "kmeans": joblib.load(
                BASE_DIR / "models" / "anomaly_detection" / "kmeans.joblib"),
            "iso_forest": joblib.load(
                BASE_DIR / "models" / "anomaly_detection" / "isolation_forest.joblib"),
        },
    }
    log.info("Loaded forecast table %s, anomaly table %s",
             forecast_df.shape, anomaly_df.shape)
    return forecast_df, anomaly_df, models


# ---------------------------------------------------------------------------
# Date normalisation (anomaly table uses InvoiceDate; forecast uses Date)
# ---------------------------------------------------------------------------
def _normalise_dates(forecast_df: pd.DataFrame,
                     anomaly_df: pd.DataFrame
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = forecast_df.copy()
    a = anomaly_df.copy()

    if "Date" not in f.columns:
        raise ValueError("forecast_df is missing 'Date' column.")
    f["Date"] = pd.to_datetime(f["Date"]).dt.normalize()

    if "Date" not in a.columns:
        if "InvoiceDate" in a.columns:
            a["Date"] = pd.to_datetime(a["InvoiceDate"]).dt.normalize()
        else:
            raise ValueError(
                "anomaly_df has neither 'Date' nor 'InvoiceDate'.")
    else:
        a["Date"] = pd.to_datetime(a["Date"]).dt.normalize()

    return f, a


# ---------------------------------------------------------------------------
# Step 1 - per-product-day forecast deviations
# ---------------------------------------------------------------------------
def compute_deviations(
    forecast_df: pd.DataFrame,
    models: dict,
    split: str = "test",
) -> pd.DataFrame:
    """Run the saved forecasting model over the given split and compute
    deviations (Actual - Predicted) at daily-product granularity."""
    path = PROCESSED_DIR / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run run_eda.py first.")
    df = pd.read_parquet(path)

    feat_cols = models["forecasting"]["metadata"]["feature_cols"]
    best_name = models["forecasting"]["metadata"]["best_model"]
    model = (models["forecasting"]["rf"]
             if best_name == "RandomForest"
             else models["forecasting"]["linear"])

    X = df[feat_cols].astype(float).values
    predicted = np.clip(model.predict(X), 0, None)
    actual = df["Quantity"].astype(float).values

    dev = actual - predicted
    eps = np.where(np.abs(predicted) > 1e-6, np.abs(predicted), 1e-6)
    dev_pct = (dev / eps) * 100.0

    out = pd.DataFrame({
        "StockCode":     df["StockCode"].values,
        "Description":   df["Description"].values,
        "Date":          pd.to_datetime(df["Date"]).dt.normalize(),
        "Actual":        actual,
        "Predicted":     predicted,
        "Deviation":     dev,
        "Deviation_Pct": dev_pct,
    })

    band = DEVIATION_BAND_PCT * 100.0
    out["Perf_Class"] = np.where(
        out["Actual"] == 0,
        "zero-sale-day",
        np.where(
            out["Deviation_Pct"] >  band, "overperforming",
            np.where(out["Deviation_Pct"] < -band, "underperforming",
                     "on-target"),
        ),
    )

    log.info("Deviations computed for %s split: %d rows, "
             "mean |dev%%| = %.1f%%",
             split, len(out), out["Deviation_Pct"].abs().mean())
    return out


# ---------------------------------------------------------------------------
# Step 2 - aggregate anomalies to product-day granularity
# ---------------------------------------------------------------------------
def aggregate_anomalies_to_daily(
    anomaly_df: pd.DataFrame,
    models: dict,
) -> pd.DataFrame:
    """Re-run the saved Isolation Forest over the anomaly features with
    the SAME StandardScaler that training used, then aggregate flags to
    (StockCode, Date) granularity."""
    from sklearn.preprocessing import StandardScaler

    feat_cols = models["anomaly"]["metadata"]["feature_cols"]
    iso = models["anomaly"]["iso_forest"]

    missing = [c for c in feat_cols if c not in anomaly_df.columns]
    if missing:
        raise ValueError(f"Anomaly table missing: {missing}")

    X_raw = anomaly_df[feat_cols].astype(float).values
    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)

    preds = iso.predict(X_scaled)
    flags = (preds == -1).astype(int)

    df = anomaly_df.copy()
    df["_flag"] = flags
    df["Date"] = pd.to_datetime(
        df["Date"] if "Date" in df.columns else df["InvoiceDate"]
    ).dt.normalize()

    agg = (df.groupby(["StockCode", "Date"], as_index=False)
             .agg(n_transactions=("_flag", "size"),
                  n_anomalies=("_flag", "sum")))
    agg["anomaly_rate"] = agg["n_anomalies"] / agg["n_transactions"]
    agg["has_anomaly"] = (agg["n_anomalies"] > 0).astype(int)
    log.info("Anomalies aggregated to %d product-days "
             "(%.2f%% of days have >=1 flagged transaction)",
             len(agg), 100 * agg["has_anomaly"].mean())
    return agg


# ---------------------------------------------------------------------------
# Step 3 - cross-reference deviations x anomalies
# ---------------------------------------------------------------------------
def cross_reference(
    deviations: pd.DataFrame,
    daily_anomalies: pd.DataFrame,
) -> pd.DataFrame:
    x = deviations.merge(
        daily_anomalies[["StockCode", "Date",
                         "n_transactions", "n_anomalies",
                         "anomaly_rate", "has_anomaly"]],
        on=["StockCode", "Date"],
        how="left",
    )
    for c in ("n_transactions", "n_anomalies", "has_anomaly"):
        x[c] = x[c].fillna(0).astype(int)
    x["anomaly_rate"] = x["anomaly_rate"].fillna(0.0)

    def _cat(row):
        if row["Actual"] == 0 and row["has_anomaly"] == 0:
            return "Zero-sale day (normal for sparse products)"
        dev = row["Deviation_Pct"]
        has_anom = row["has_anomaly"] == 1
        if has_anom and dev > DEVIATION_BAND_PCT * 100:
            return "Expected spike with unusual transactions"
        if has_anom and dev < -DEVIATION_BAND_PCT * 100:
            return "Missed demand with unusual transactions"
        if has_anom:
            return "Unexpected anomaly (no demand deviation)"
        if dev > DEVIATION_BAND_PCT * 100:
            return "Silent overperformance"
        if dev < -DEVIATION_BAND_PCT * 100:
            return "Silent underperformance"
        return "Normal (on-target, no anomalies)"

    x["hybrid_category"] = x.apply(_cat, axis=1)
    log.info("Hybrid categories assigned:\n%s",
             x["hybrid_category"].value_counts().to_string())
    return x


# ---------------------------------------------------------------------------
# Step 4 - per-product summary
# ---------------------------------------------------------------------------
def per_product_summary(hybrid: pd.DataFrame) -> pd.DataFrame:
    grp = hybrid.groupby(["StockCode", "Description"], as_index=False)
    summary = grp.agg(
        n_days=("Date", "nunique"),
        mean_actual=("Actual", "mean"),
        mean_predicted=("Predicted", "mean"),
        mean_deviation_pct=("Deviation_Pct", "mean"),
        abs_mean_deviation=("Deviation_Pct",
                            lambda s: float(np.mean(np.abs(s)))),
        n_anomalous_days=("has_anomaly", "sum"),
        anomaly_rate=("anomaly_rate", "mean"),
    ).sort_values("abs_mean_deviation", ascending=False)
    log.info("Per-product summary: %d products", len(summary))
    return summary


# ---------------------------------------------------------------------------
# Step 5 - business insight generator
# ---------------------------------------------------------------------------
def generate_insights(
    hybrid: pd.DataFrame,
    summary: pd.DataFrame,
    top_k: int = 10,
) -> dict:
    cat_counts = hybrid["hybrid_category"].value_counts().to_dict()

    over = summary[summary["mean_deviation_pct"] > DEVIATION_BAND_PCT * 100].head(top_k)
    under = summary[summary["mean_deviation_pct"] < -DEVIATION_BAND_PCT * 100].head(top_k)
    anomaly_hotspots = summary.sort_values("n_anomalous_days",
                                           ascending=False).head(top_k)
    unstable = summary.head(top_k)

    def _fmt(df: pd.DataFrame, cols: Sequence[str]) -> list[dict]:
        return df[list(cols)].round(2).to_dict(orient="records")

    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "scope_note":
            "Deviation signals are hypotheses, not confirmed events. "
            "Anomaly flags are unsupervised outputs without ground truth.",
        "hybrid_category_counts": cat_counts,
        "overperforming_products": _fmt(
            over, ("StockCode", "Description", "mean_deviation_pct")),
        "underperforming_products": _fmt(
            under, ("StockCode", "Description", "mean_deviation_pct")),
        "anomaly_hotspots": _fmt(
            anomaly_hotspots,
            ("StockCode", "Description", "n_anomalous_days", "anomaly_rate")),
        "most_unstable_demand": _fmt(
            unstable, ("StockCode", "Description", "abs_mean_deviation")),
        "rules": {
            "overperforming": "mean_deviation_pct > 25%",
            "underperforming": "mean_deviation_pct < -25%",
            "anomaly_hotspot": "highest n_anomalous_days",
        },
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
def _save(fig, name: str) -> None:
    path = REPORTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure -> %s", path)


def plot_deviation_distribution(hybrid: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dev = hybrid["Deviation_Pct"].clip(-200, 200)
    sns.histplot(dev, bins=120, ax=axes[0], color="#2563EB")
    axes[0].axvline(DEVIATION_BAND_PCT * 100, color="red", ls="--")
    axes[0].axvline(-DEVIATION_BAND_PCT * 100, color="red", ls="--")
    axes[0].set_title("Forecast deviation % distribution (clipped +/-200%)")
    axes[0].set_xlabel("Deviation %")

    counts = hybrid["Perf_Class"].value_counts()
    sns.barplot(x=counts.values, y=counts.index, ax=axes[1],
                hue=counts.index, palette="Blues_r", legend=False)
    axes[1].set_title("Perf class counts")
    fig.tight_layout()
    _save(fig, "01_deviation_distribution")


def plot_deviation_vs_anomaly(hybrid: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(hybrid), size=min(5000, len(hybrid)), replace=False)
    sample = hybrid.iloc[idx]
    colors = np.where(sample["has_anomaly"] == 1, "#EF4444", "#94A3B8")
    ax.scatter(sample["Deviation_Pct"].clip(-200, 200),
               sample["Actual"], c=colors, s=6, alpha=0.4)
    ax.set_xlabel("Deviation % (clipped +/-200%)")
    ax.set_ylabel("Actual quantity")
    ax.set_title("Deviation vs actual, coloured by anomaly presence")
    fig.tight_layout()
    _save(fig, "02_deviation_vs_anomaly")


def plot_category_breakdown(hybrid: pd.DataFrame) -> None:
    counts = hybrid["hybrid_category"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=counts.values, y=counts.index, ax=ax,
                hue=counts.index, palette="viridis", legend=False)
    ax.set_title("Hybrid categories across all product-days")
    ax.set_xlabel("# product-days")
    fig.tight_layout()
    _save(fig, "03_category_breakdown")


def plot_top_products(summary: pd.DataFrame, k: int = 20) -> None:
    top = summary.head(k)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top["Description"][::-1],
            top["abs_mean_deviation"][::-1], color="#2563EB")
    ax.set_xlabel("|mean deviation %|")
    ax.set_title(f"Top {k} most volatile products (largest |deviation %|)")
    fig.tight_layout()
    _save(fig, "04_top_unstable_products")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _save_df(df: pd.DataFrame, name: str) -> None:
    path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    log.info("Saved %s -> %s", name, path.name)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_hybrid_pipeline(verbose: bool = True) -> dict:
    forecast_df, anomaly_df, models = load_inputs()
    forecast_df, anomaly_df = _normalise_dates(forecast_df, anomaly_df)

    deviations = compute_deviations(forecast_df, models, split="test")
    daily_anom = aggregate_anomalies_to_daily(anomaly_df, models)
    hybrid = cross_reference(deviations, daily_anom)
    summary = per_product_summary(hybrid)
    insights = generate_insights(hybrid, summary)

    plot_deviation_distribution(hybrid)
    plot_deviation_vs_anomaly(hybrid)
    plot_category_breakdown(hybrid)
    plot_top_products(summary)

    _save_df(deviations, "hybrid_deviations")
    _save_df(hybrid,     "hybrid_anomaly_xref")
    _save_df(summary,    "hybrid_product_summary")
    (PROCESSED_DIR / "hybrid_insights.json").write_text(
        json.dumps(insights, indent=2, default=str), encoding="utf-8")

    if verbose:
        print("\n" + "=" * 72)
        print("HYBRID ANALYSIS SUMMARY")
        print("=" * 72)
        print(f"Rows analysed            : {len(hybrid):,}")
        print(f"Products                 : {len(summary):,}")
        print(f"Mean |deviation %|       : "
              f"{hybrid['Deviation_Pct'].abs().mean():.2f}%")
        print(f"Days with >=1 anomaly    : "
              f"{int(hybrid['has_anomaly'].sum()):,} "
              f"({100*hybrid['has_anomaly'].mean():.2f}%)")
        print("\nHybrid categories:")
        for cat, cnt in hybrid["hybrid_category"].value_counts().items():
            print(f"  {cat:45s}: {cnt:,}")
        print("=" * 72)

    return {
        "deviations": deviations,
        "hybrid": hybrid,
        "summary": summary,
        "insights": insights,
    }