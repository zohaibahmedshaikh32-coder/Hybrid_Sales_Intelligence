"""
evaluation_page.py — Page 8: Model Evaluation.

Consolidates evaluation artefacts for all trained models:
    • Forecasting: LR vs RF (MAE / MSE / RMSE / MAPE / R²)
    • Anomaly: K-Means / Hierarchical / Isolation Forest metrics
    • Reference figures from reports/ folders
    • Known limitations
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.ui import (
    PALETTE, empty_state, fmt_int, page_header, section,
)
from src.config import BASE_DIR, PROCESSED_DIR


PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color=PALETTE["text"]),
    margin=dict(l=10, r=10, t=40, b=10),
)


def _load_json_model(name: str) -> dict:
    p = BASE_DIR / "models" / name / "metadata.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def render() -> None:
    page_header(
        "Model Evaluation",
        "Consolidated performance metrics and comparison across all "
        "trained models.",
    )

    # =================================================================
    # 1. Forecasting metrics
    # =================================================================
    section("1. Forecasting Models — Linear Regression vs Random Forest")
    fc = _load_json_model("forecasting")
    if not fc or "metrics" not in fc:
        empty_state("Forecasting metrics not found. "
                    "Run `python -m scripts.train_forecasting`.", icon="⚙️")
    else:
        rows = []
        for model_name, splits in fc["metrics"].items():
            for split in ("val", "test"):
                m = splits.get(split, {})
                rows.append({
                    "Model":  model_name,
                    "Split":  split,
                    "MAE":    round(m.get("MAE", 0), 3),
                    "MSE":    round(m.get("MSE", 0), 3),
                    "RMSE":   round(m.get("RMSE", 0), 3),
                    "MAPE":   round(m.get("MAPE", 0), 2),
                    "R²":     round(m.get("R2",  0), 4),
                    "n":      m.get("n", 0),
                })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Side-by-side RMSE / R²
        test_df = df[df["Split"] == "test"]
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(test_df, x="Model", y="RMSE",
                         color="Model",
                         color_discrete_sequence=px.colors.qualitative.Set1)
            fig.update_layout(title="Test RMSE (lower is better)",
                              height=340, showlegend=False,
                              **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(test_df, x="Model", y="R²",
                         color="Model",
                         color_discrete_sequence=px.colors.qualitative.Set1)
            fig.update_layout(title="Test R² (higher is better)",
                              height=340, showlegend=False,
                              **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        best = fc.get("best_model", "?")
        st.success(f"**Best model (by validation RMSE): {best}**")

    # =================================================================
    # 2. Anomaly detection metrics
    # =================================================================
    section("2. Anomaly Detection Models")
    an = _load_json_model("anomaly_detection")
    if not an or "metrics" not in an:
        empty_state("Anomaly metrics not found. "
                    "Run `python -m scripts.train_anomaly`.", icon="⚙️")
    else:
        mdf = pd.DataFrame(an["metrics"])
        st.dataframe(mdf, use_container_width=True, hide_index=True)

        # Comparison bar chart (silhouette)
        if "silhouette" in mdf.columns:
            fig = px.bar(mdf, x="model", y="silhouette",
                         color="model",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(
                title="Silhouette score (higher is better)",
                height=340, showlegend=False,
                xaxis_title="Model", yaxis_title="Silhouette",
                **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "⚠️ **Unsupervised evaluation without ground truth** — "
            "these are *internal* clustering-quality metrics. There is "
            "no verified fraud label in the dataset, so we cannot "
            "compute precision or recall for any anomaly model.")

    # =================================================================
    # 3. Reference figures
    # =================================================================
    section("3. Reference Figures")
    figs_dir = BASE_DIR / "reports"
    tabs = st.tabs(["Forecasting", "Anomaly", "Hybrid", "Streaming"])

    def _show_first_png(directory, name_filter=""):
        if not directory.exists():
            st.info(f"No figures found in `{directory.relative_to(BASE_DIR)}`")
            return
        pngs = sorted(directory.glob("*.png"))
        if name_filter:
            pngs = [p for p in pngs if name_filter in p.name]
        if not pngs:
            st.info(f"No PNGs found in `{directory.relative_to(BASE_DIR)}`")
            return
        for p in pngs[:4]:
            st.image(str(p), caption=p.name, use_container_width=True)

    with tabs[0]:
        _show_first_png(figs_dir / "forecasting")
    with tabs[1]:
        _show_first_png(figs_dir / "anomaly_detection")
    with tabs[2]:
        _show_first_png(figs_dir / "hybrid_analysis")
    with tabs[3]:
        _show_first_png(figs_dir / "streaming")

    # =================================================================
    # 4. Known limitations (transparency)
    # =================================================================
    section("4. Known Limitations")
    st.markdown(
        """
        - **No ground-truth labels.** Forecasting is evaluated by
          split-based MAE / RMSE / R²; anomaly detection can only be
          evaluated by internal clustering metrics (Silhouette, DB-Index)
          and manual inspection of top-ranked flags.
        - **MAPE is inflated by zero-sale days.** Products with sparse
          demand have many `Actual = 0` rows; the percentage deviation
          metric is therefore reported with zero-safe handling, but the
          value is still dominated by these rows.
        - **Streaming is simulated.** The monitoring page replays
          historical transactions in chronological order through the
          trained Isolation Forest — it is not connected to any live
          retail system.
        - **Anomaly flags are hypotheses, not confirmations.** No
          transaction is treated as fraudulent; each flag is a signal
          that warrants human review.
        - **Description drift.** Some StockCodes have multiple
          descriptions in the UCI Online Retail II dataset (edited
          over time); we collapse to the most recent / most frequent.
        """)