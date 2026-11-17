"""
monitoring_page.py — Page 6: Live Monitoring Simulation.

Uses the StreamSimulator from src.streaming to replay transactions
in real-time. Full state lives in st.session_state so Start / Pause /
Reset work across Streamlit reruns.
"""
from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def _get_simulator():
    """Create (once) and return the StreamSimulator stored in session_state."""
    if "stream_sim" not in st.session_state:
        try:
            from src.streaming import StreamSimulator
            sim = StreamSimulator(n_per_batch=500, max_alerts=5000)
            st.session_state["stream_sim"] = sim
        except Exception as e:
            return None, str(e)
    return st.session_state["stream_sim"], None


def render() -> None:
    page_header(
        "Live Monitoring Simulation",
        "Historical transactions replayed through the trained Isolation "
        "Forest in simulated real-time.",
    )

    st.caption(
        "⚠️ **Simulated streaming** — this is a replay of the historical "
        "UCI Online Retail II dataset in chronological order, not a live "
        "connection to a retailer system.")

    sim, err = _get_simulator()
    if sim is None:
        empty_state(f"Simulator could not start: {err}", icon="⚠️")
        return

    # -----------------------------------------------------------------
    # 1. Controls
    # -----------------------------------------------------------------
    section("Controls")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

    with c1:
        if st.button("▶️ Start", type="primary", use_container_width=True):
            sim.start()
    with c2:
        if st.button("⏸️ Pause", use_container_width=True):
            sim.pause()
    with c3:
        if st.button("🔄 Reset", use_container_width=True):
            sim.reset()
            st.session_state.pop("stream_alerts_df", None)
    with c4:
        batch_size = st.selectbox(
            "Batch size per tick", options=[100, 500, 1000, 5000],
            index=1, key="monitor_batch")
        sim.n_per_batch = batch_size

    # -----------------------------------------------------------------
    # 2. Live KPIs
    # -----------------------------------------------------------------
    section("Live Status")
    s = sim.summary()

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("Status",        s["status"].title())
    with k2: st.metric("Processed",     fmt_int(s["n_processed"]))
    with k3: st.metric("Normal",        fmt_int(s["n_normal"]))
    with k4: st.metric("Anomalous",     fmt_int(s["n_anomalous"]),
                       delta_color="inverse")
    with k5: st.metric("Anomaly Rate",  f"{s['anomaly_rate_pct']:.2f}%")

    st.progress(s["progress_pct"] / 100.0,
                text=f"{s['progress_pct']:.1f}% of {fmt_int(s['total_rows'])} transactions")

    # -----------------------------------------------------------------
    # 3. Auto-tick loop
    # -----------------------------------------------------------------
    if sim.status == "running":
        # Advance ONE batch per rerun, then re-run after sleep
        sim.step()
        # Persist alerts snapshot for display
        st.session_state["stream_alerts_df"] = sim.alerts_df()

        time.sleep(0.05)
        st.rerun()

    # -----------------------------------------------------------------
    # 4. Alert rate over time
    # -----------------------------------------------------------------
    rate_df = sim.alert_rate_df()
    if not rate_df.empty:
        section("Alert Rate Over Time")
        fig = px.line(rate_df, x="batch", y="alert_rate",
                      color_discrete_sequence=[PALETTE["danger"]])
        fig.update_layout(
            title="Flagged transactions per batch",
            xaxis_title="Batch number",
            yaxis_title="Alert rate (fraction)",
            height=320, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------
    # 5. Latest alerts
    # -----------------------------------------------------------------
    alerts_df = st.session_state.get("stream_alerts_df",
                                     sim.alerts_df() if sim.alerts else pd.DataFrame())

    section(f"Alert History ({fmt_int(len(alerts_df))} stored)")
    if alerts_df.empty:
        st.info("No alerts yet — press **Start** to begin the simulation.")
    else:
        display_cols = [c for c in (
            "timestamp", "Invoice", "StockCode", "Description",
            "Quantity", "Price", "Revenue", "anomaly_score", "reason")
            if c in alerts_df.columns]
        st.dataframe(
            alerts_df.sort_values("anomaly_score", ascending=False)
                      .head(200)[display_cols],
            use_container_width=True, hide_index=True, height=420,
        )
        st.download_button(
            "⬇️ Download alert history (CSV)",
            data=alerts_df.to_csv(index=False).encode("utf-8"),
            file_name="stream_alerts.csv",
            mime="text/csv",
        )