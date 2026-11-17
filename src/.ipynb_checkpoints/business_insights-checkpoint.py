"""
business_insights.py

Translate the numeric hybrid analysis into business-facing mitigation
recommendations. Every recommendation is a HYPOTHESIS for human
review — never an automatic decision, never a fraud claim.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PROCESSED_DIR, BASE_DIR

log = logging.getLogger(__name__)

REPORTS_DIR = BASE_DIR / "reports" / "business_insights"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OVER_THRESHOLD_PCT  = 25.0
UNDER_THRESHOLD_PCT = -25.0
INSTABILITY_PCT     = 100.0
TOP_K               = 10


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load():
    hybrid  = pd.read_parquet(PROCESSED_DIR / "hybrid_anomaly_xref.parquet")
    summary = pd.read_parquet(PROCESSED_DIR / "hybrid_product_summary.parquet")
    returns = pd.read_parquet(PROCESSED_DIR / "returns.parquet")
    return hybrid, summary, returns


# ---------------------------------------------------------------------------
# Detection rules
# ---------------------------------------------------------------------------
def _stockout_risk(summary: pd.DataFrame) -> list[dict]:
    """
    Products overperforming consistently → stockout risk.
    Rule: mean_deviation_pct > 25% AND at least 5 anomalous days.
    """
    m = summary[
        (summary["mean_deviation_pct"] > OVER_THRESHOLD_PCT)
        & (summary["n_anomalous_days"] >= 5)
    ].head(TOP_K)
    return [_row_to_mitigation(
        row, category="stockout_risk", severity="high",
        evidence=(f"Mean deviation {row['mean_deviation_pct']:+.1f}%, "
                  f"{int(row['n_anomalous_days'])} anomalous days"),
        action="Increase reorder quantity for next 2 weeks; "
               "check supplier lead time; consider safety stock bump.",
        caveat="Deviation may reflect seasonality or bulk orders — "
               "verify against upcoming promotions.",
    ) for _, row in m.iterrows()]


def _overstock_risk(summary: pd.DataFrame) -> list[dict]:
    """Products underperforming consistently → overstock risk."""
    m = summary[
        (summary["mean_deviation_pct"] < UNDER_THRESHOLD_PCT)
        & (summary["n_anomalous_days"] >= 5)
    ].head(TOP_K)
    return [_row_to_mitigation(
        row, category="overstock_risk", severity="medium",
        evidence=(f"Mean deviation {row['mean_deviation_pct']:+.1f}%, "
                  f"{int(row['n_anomalous_days'])} anomalous days"),
        action="Reduce next reorder; consider promotional pricing or "
               "bundle with high-demand items.",
        caveat="Underperformance may reflect genuine demand drop or a "
               "stockout in the observation window — cross-check with "
               "inventory data.",
    ) for _, row in m.iterrows()]


def _demand_instability(summary: pd.DataFrame) -> list[dict]:
    """High |mean deviation| → erratic demand."""
    m = summary[summary["abs_mean_deviation"] > INSTABILITY_PCT].head(TOP_K)
    return [_row_to_mitigation(
        row, category="demand_instability", severity="medium",
        evidence=f"|mean deviation| = {row['abs_mean_deviation']:.1f}%",
        action="Move to manual demand-planning review; consider "
               "shortening forecast horizon and adding safety margin.",
        caveat="Instability may be structural (seasonal/bulk) — "
               "segment the customer base before assuming volatility.",
    ) for _, row in m.iterrows()]


def _anomaly_hotspots(summary: pd.DataFrame) -> list[dict]:
    """Most anomaly-flagged days."""
    m = summary.sort_values("n_anomalous_days", ascending=False).head(TOP_K)
    return [_row_to_mitigation(
        row, category="anomaly_hotspot", severity="high",
        evidence=(f"{int(row['n_anomalous_days'])} anomalous days, "
                  f"mean anomaly rate {row['anomaly_rate']:.3f}"),
        action="Audit transactions for this product: check for data-entry "
               "errors, bulk customer patterns, or reporting anomalies.",
        caveat="High anomaly count does not imply fraud — it means "
               "unusual transaction patterns warrant manual review.",
    ) for _, row in m.iterrows()]


def _return_anomaly_products(returns: pd.DataFrame,
                             summary: pd.DataFrame,
                             top_k: int = TOP_K) -> list[dict]:
    """
    Products with abnormal return ratios.

    Note on duplicates: some StockCodes have MULTIPLE descriptions in the
    UCI Online Retail II dataset (the description text was edited over
    time). We collapse to one row per StockCode — keeping the most
    frequent description — so the reindex below never sees duplicate
    labels.
    """
    if returns.empty:
        return []

    # Aggregate returns to StockCode level
    per_product = (returns.assign(AbsQty=returns["Quantity"].abs())
                          .groupby("StockCode")["AbsQty"].sum()
                          .sort_values(ascending=False)
                          .head(top_k))
    if per_product.empty:
        return []

    # Collapse the summary to one row per StockCode (most-frequent
    # description wins). This avoids duplicate-index errors downstream.
    summary_unique = (
        summary.groupby(["StockCode", "Description"], as_index=False)
               .agg(n_days=("n_days", "first"),
                    mean_actual=("mean_actual", "first"))
               .sort_values("n_days", ascending=False)
               .drop_duplicates("StockCode")
               .set_index("StockCode")
    )

    mitigations = []
    for code, ret_qty in per_product.items():
        if code not in summary_unique.index:
            continue
        row = summary_unique.loc[code]
        mitigations.append({
            "StockCode":        str(code),
            "Description":      str(row["Description"]),
            "category":         "return_anomaly_product",
            "severity":         "medium",
            "evidence":         f"{int(ret_qty)} units returned in total",
            "suggested_action": "Investigate return reasons (quality, "
                                "wrong listing, size mismatch). "
                                "Review product page and supplier.",
            "caveat":           "High returns may be legitimately seasonal "
                                "(e.g. post-holiday) — split by month to "
                                "confirm.",
        })
    return mitigations


def _price_anomaly_products(hybrid: pd.DataFrame) -> list[dict]:
    """
    Products with the highest proportion of 'Unusually high unit price'
    flags → potential pricing inconsistency.
    """
    mask = hybrid["hybrid_category"].str.contains(
        "unusual", case=False, na=False)
    flagged = hybrid[mask]
    if flagged.empty:
        return []
    top = (flagged.groupby(["StockCode", "Description"], as_index=False)
                  .size()
                  .sort_values("size", ascending=False)
                  .head(TOP_K))
    return [_row_to_mitigation(
        row, category="pricing_inconsistency", severity="medium",
        evidence=f"{int(row['size'])} unusual-transaction flags",
        action="Verify SKU-level price rules; look for manual overrides, "
               "currency mismatches, or regional pricing errors.",
        caveat="Unusual prices may be legitimate discounts — verify "
               "against the pricing master data.",
    ) for _, row in top.iterrows()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _row_to_mitigation(row: pd.Series, **kwargs) -> dict:
    return {
        "StockCode":        str(row.get("StockCode", "")),
        "Description":      str(row.get("Description", "")),
        "category":         kwargs["category"],
        "severity":         kwargs["severity"],
        "evidence":         kwargs["evidence"],
        "suggested_action": kwargs["action"],
        "caveat":           kwargs["caveat"],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_business_insights(verbose: bool = True) -> list[dict]:
    hybrid, summary, returns = _load()

    mitigations: list[dict] = []
    mitigations.extend(_stockout_risk(summary))
    mitigations.extend(_overstock_risk(summary))
    mitigations.extend(_demand_instability(summary))
    mitigations.extend(_anomaly_hotspots(summary))
    mitigations.extend(_return_anomaly_products(returns, summary))
    mitigations.extend(_price_anomaly_products(hybrid))

    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "scope_note":
            "All recommendations are hypotheses for human review. "
            "They are NOT confirmed business events and must be cross-"
            "checked against inventory, pricing, and customer data before "
            "operational action.",
        "n_mitigations": len(mitigations),
        "mitigations": mitigations,
    }
    (PROCESSED_DIR / "business_mitigations.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("Saved %d mitigations -> business_mitigations.json",
             len(mitigations))

    if mitigations:
        df = pd.DataFrame(mitigations)
        df.to_csv(REPORTS_DIR / "mitigations.csv", index=False)
        _plot_severity_breakdown(df)

    if verbose:
        print("\n" + "=" * 72)
        print("BUSINESS MITIGATION SUMMARY")
        print("=" * 72)
        df = pd.DataFrame(mitigations)
        if not df.empty:
            print(df.groupby(["category", "severity"]).size().to_string())
            print(f"\nTotal mitigations: {len(df)}")
        else:
            print("No mitigations generated.")
        print("=" * 72)

    return mitigations


def _plot_severity_breakdown(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    counts = df.groupby(["category", "severity"]).size().reset_index(name="n")
    sns.barplot(data=counts, x="n", y="category", hue="severity",
                ax=ax, palette={"high": "#EF4444", "medium": "#F59E0B"})
    ax.set_title("Business mitigation counts by category & severity")
    ax.set_xlabel("# mitigations")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "01_severity_breakdown.png",
                dpi=110, bbox_inches="tight")
    plt.close(fig)