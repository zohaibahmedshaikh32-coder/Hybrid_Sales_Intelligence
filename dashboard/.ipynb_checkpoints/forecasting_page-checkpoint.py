"""
forecasting_page.py — Page 4: Demand Forecasting.

Interactive panel that:
    - Lets the user pick a product and horizon
    - Runs the SAVED forecasting model (no retraining)
    - Plots history + forecast
    - Displays real test-set metrics for LR and RF
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.ui import (
    PALETTE, empty_state, fmt_int, load_parquet, page_header, section,
)
from src.config import BASE_DIR, PROCESSED_DIR


PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color=PALETTE["text"]),
    margin=dict(l=10, r=10, t=40, b=10),
)


@st.cache_data(show_spinner="Loading forecasting models …")
def _load_meta() -> dict:
    p = BASE_DIR / "models" / "forecasting" / "metadata.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Loading product catalogue …")
def _load_products() -> pd.DataFrame:
    q = PROCESSED_DIR / "product_quality_scores.parquet"
    if not q.exists():
        return pd.DataFrame()
    return pd.read_parquet(q)


def render() -> None:
    page_header(
        "Demand Forecasting",
        "Predict the next N days of demand for a selected product using "
        "the trained forecasting models.",
    )

    meta = _load_meta()
    products = _load_products()

    if not meta:
        empty_state(
            "Forecasting models not found. Run "
            "`python -m scripts.train_forecasting` first.",
            icon="⚙️")
        return
    if products.empty:
        empty_state(
            "Product quality scores not found. Run `python run_eda.py`.",
            icon="⚙️")
        return

    # -----------------------------------------------------------------
    # Controls
    # -----------------------------------------------------------------
    section("Forecast Configuration")
    c1, c2, c3 = st.columns([2, 1, 1])

    # Only show products that passed the forecastability filter
    forecastable = products[products["active_days"] >= 60].copy()
    if forecastable.empty:
        empty_state("No forecastable products found.", icon="📭")
        return

    forecastable = forecastable.sort_values("score", ascending=False)
    labels = (forecastable["StockCode"].astype(str)
              + "  —  " + forecastable["StockCode"].map(
                    lambda c: _description_for(c, products)))

    with c1:
        idx = st.selectbox(
            "Product",
            options=range(len(forecastable)),
            format_func=lambda i: labels.iloc[i],
            key="forecast_product",
        )
    with c2:
        horizon = st.slider("Forecast horizon (days)", 1, 30, 14,
                            key="forecast_horizon")
    with c3:
        model_choice = st.selectbox(
            "Model",
            options=["Best (auto)", "LinearRegression", "RandomForest"],
            key="forecast_model",
        )

    product_code = forecastable.iloc[idx]["StockCode"]
    model_name = None if model_choice.startswith("Best") else model_choice

    if st.button("🔮 Run Forecast", type="primary"):
        try:
            from src.forecasting import forecast_future
            with st.spinner("Running recursive forecast …"):
                fc = forecast_future(
                    product_code=product_code,
                    horizon_days=horizon,
                    model_name=model_name,
                )
            st.session_state["fc_result"] = fc
            st.session_state["fc_product"] = product_code
            st.session_state["fc_model"] = model_choice
        except Exception as e:
            st.error(f"**Forecast failed:** {e}")

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------
    fc = st.session_state.get("fc_result")
    if fc is None:
        st.info("Configure the options above and click **Run Forecast**.")
    else:
        _render_forecast(fc, product_code, meta)

    # -----------------------------------------------------------------
    # Model performance comparison (always shown)
    # -----------------------------------------------------------------
    section("Model Performance (test set)")
    _render_metrics_table(meta)


# ---------------------------------------------------------------------------
# Helper — resolve description for a product code
# ---------------------------------------------------------------------------
def _description_for(code: str, products: pd.DataFrame) -> str:
    try:
        row = products[products["StockCode"] == code]
        if not row.empty:
            if "Description" in row.columns:
                return str(row.iloc[0]["Description"])[:60]
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Forecast chart
# ---------------------------------------------------------------------------
def _render_forecast(fc: pd.DataFrame, product_code: str, meta: dict) -> None:
    section("Forecast Result")

    # Historical series (last 90 days)
    hist_path = PROCESSED_DIR / "daily_product_features.parquet"
    if hist_path.exists():
        hist = pd.read_parquet(hist_path,
                               columns=["StockCode", "Date", "Quantity"])
        hist = (hist[hist["StockCode"] == product_code]
                .sort_values("Date").tail(90))
    else:
        hist = pd.DataFrame(columns=["Date", "Quantity"])

    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["Date"], y=hist["Quantity"],
            mode="lines+markers", name="History (last 90 days)",
            line=dict(color=PALETTE["muted"], width=1.4),
            marker=dict(size=3),
        ))
    fig.add_trace(go.Scatter(
        x=fc["Date"], y=fc["predicted_quantity"],
        mode="lines+markers", name="Forecast",
        line=dict(color=PALETTE["primary"], width=2.4),
        marker=dict(size=6, color=PALETTE["primary"]),
    ))
    fig.update_layout(
        title=f"Demand forecast — {fc['Description'].iloc[0]} ({product_code})",
        xaxis_title="Date", yaxis_title="Predicted quantity / day",
        height=440, hovermode="x unified", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Summary KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Horizon (days)", len(fc))
    with c2: st.metric("Total predicted", f"{fc['predicted_quantity'].sum():.0f} units")
    with c3: st.metric("Average per day",   f"{fc['predicted_quantity'].mean():.1f} units")
    with c4: st.metric("Model used",        fc['model_used'].iloc[0])

    with st.expander("View forecast table"):
        out = fc[["Date", "predicted_quantity", "model_used"]].copy()
        out.columns = ["Date", "Predicted Quantity", "Model"]
        st.dataframe(out, use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ Forecast is an estimate, not a guarantee. Recursive lag "
        "features mean errors compound across the horizon.")


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------
def _render_metrics_table(meta: dict) -> None:
    metrics = meta.get("metrics", {})
    if not metrics:
        st.info("No metrics available — retrain the models.")
        return

    rows = []
    for model_name, splits in metrics.items():
        for split in ("val", "test"):
            m = splits.get(split, {})
            rows.append({
                "Model":  model_name,
                "Split":  split,
                "MAE":    round(m.get("MAE", float("nan")), 2),
                "RMSE":   round(m.get("RMSE", float("nan")), 2),
                "MAPE":   round(m.get("MAPE", float("nan")), 2),
                "R²":     round(m.get("R2",  float("nan")), 4),
                "n":      m.get("n", 0),
            })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    best = meta.get("best_model", "?")
    st.success(f"**Best model (by validation RMSE): {best}**")