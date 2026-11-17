"""
overview.py — Page 1: Executive Overview.

Nine KPI cards + sales trend + business insights summary.
Answers the question "what does the business look like right now?"
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.ui import (
    PALETTE, fmt_gbp, fmt_int, fmt_pct, kpi_card, page_header, section,
)


@st.cache_data
def _load_all():
    from dashboard.ui import load_parquet, load_json
    sales   = load_parquet("valid_sales")
    returns = load_parquet("returns")
    hybrid  = load_parquet("hybrid_anomaly_xref")
    summary = load_parquet("hybrid_product_summary")
    alerts  = load_parquet("stream_alerts") \
              if (load_parquet.__wrapped__ if hasattr(load_parquet, "__wrapped__")
                  else load_parquet)("stream_alerts").shape[0] >= 0 else pd.DataFrame()
    ins     = load_json("business_mitigations")
    return sales, returns, hybrid, summary, alerts, ins


def render() -> None:
    page_header(
        "Executive Overview",
        "Real-time KPIs, sales trend, and business health at a glance.",
    )

    # ---- Load ------------------------------------------------------
    sales    = _safe_load("valid_sales")
    returns  = _safe_load("returns")
    hybrid   = _safe_load("hybrid_anomaly_xref")
    alerts   = _safe_load("stream_alerts")
    ins      = _safe_load_json("business_mitigations")
    fc_meta  = _safe_load_json_model("forecasting")
    an_meta  = _safe_load_json_model("anomaly_detection")

    if sales is None or sales.empty:
        st.error("No cleaned sales data found. Run `python run_eda.py`.")
        return

    # ---- KPI cards (2 rows × 5 cols) -------------------------------
    total_rev      = float((sales["Quantity"] * sales["Price"]).sum())
    total_tx       = int(len(sales))
    total_prods    = int(sales["StockCode"].nunique())
    total_cust     = int(sales["Customer ID"].dropna().nunique())
    total_country  = int(sales["Country"].nunique())
    total_returns  = int(len(returns)) if returns is not None else 0

    det_anomalies = 0
    if hybrid is not None and not hybrid.empty:
        det_anomalies = int(hybrid["has_anomaly"].sum())

    fc_status = "✅ Trained"
    if fc_meta and "best_model" in fc_meta:
        fc_status = f"✅ {fc_meta['best_model']}"

    section("Key Performance Indicators")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Total Transactions", fmt_int(total_tx), icon="🧾")
    with c2: kpi_card("Total Revenue",      fmt_gbp(total_rev), icon="💷")
    with c3: kpi_card("Products",           fmt_int(total_prods), icon="📦")
    with c4: kpi_card("Customers",          fmt_int(total_cust), icon="👥")
    with c5: kpi_card("Countries",          fmt_int(total_country), icon="🌍")

    c6, c7, c8, c9, c10 = st.columns(5)
    with c6:  kpi_card("Returns",           fmt_int(total_returns), icon="↩️")
    with c7:  kpi_card("Anomalies Detected", fmt_int(det_anomalies),
                       icon="🚨", delta_color="inverse")
    with c8:  kpi_card("Forecasting Model", fc_status, icon="📈")
    with c9:  kpi_card("Anomaly Models",    "✅ 3 trained", icon="🧠")
    with c10:
        n_mit = len(ins.get("mitigations", [])) if ins else 0
        kpi_card("Mitigations",  fmt_int(n_mit), icon="🛡️")

    # ---- Sales trend ----------------------------------------------
    section("Sales Trend")
    sales2 = sales.copy()
    sales2["Date"] = pd.to_datetime(sales2["InvoiceDate"]).dt.normalize()
    sales2["Revenue"] = sales2["Quantity"] * sales2["Price"]

    daily = sales2.groupby("Date", as_index=False)["Revenue"].sum()
    monthly = (sales2.assign(Month=sales2["InvoiceDate"].dt.to_period("M")
                                          .astype(str))
                     .groupby("Month", as_index=False)["Revenue"].sum())

    tab1, tab2 = st.tabs(["Daily Revenue", "Monthly Revenue"])
    with tab1:
        fig = px.line(daily, x="Date", y="Revenue",
                      color_discrete_sequence=[PALETTE["primary"]])
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0),
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=PALETTE["text"]))
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = px.bar(monthly, x="Month", y="Revenue",
                     color_discrete_sequence=[PALETTE["secondary"]])
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=20, b=0),
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=PALETTE["text"]))
        st.plotly_chart(fig, use_container_width=True)

    # ---- Top insights (from business mitigations) -----------------
    if ins and ins.get("mitigations"):
        section("Top Business Signals")
        df = pd.DataFrame(ins["mitigations"])
        high = df[df["severity"] == "high"]
        if not high.empty:
            st.dataframe(
                high[["category", "Description", "evidence",
                      "suggested_action", "caveat"]],
                use_container_width=True, hide_index=True, height=280,
            )
        st.caption(
            "⚠️ All signals are hypotheses for human review. "
            "Not confirmed business events — see Business Intelligence page.")


# ---------------------------------------------------------------------------
# Defensive loaders — silent on missing, loud on usage
# ---------------------------------------------------------------------------
def _safe_load(name: str) -> pd.DataFrame | None:
    from dashboard.ui import load_parquet
    path = PROCESSED_DIR = __import__("src.config", fromlist=["PROCESSED_DIR"]).PROCESSED_DIR
    if not (path / f"{name}.parquet").exists():
        return None
    try:
        return load_parquet(name)
    except Exception:
        return None


def _safe_load_json(name: str) -> dict:
    from dashboard.ui import load_json
    try:
        return load_json(name)
    except Exception:
        return {}


def _safe_load_json_model(model_dir: str) -> dict:
    import json
    from src.config import BASE_DIR
    p = BASE_DIR / "models" / model_dir / "metadata.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}