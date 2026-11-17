"""
app.py — Retail Intelligence dashboard entry point.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from dashboard import (
    overview, data_explorer,
    sales_analytics, forecasting_page, anomaly_page,
    monitoring_page, business_insights_page,
    evaluation_page, reports_page,
)


st.set_page_config(
    page_title="Retail Intelligence – Hybrid Predictive System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:.75rem 0 1.25rem 0;'>
            <div style='font-size:2rem;'>🛒</div>
            <div style='font-size:1.1rem; font-weight:700;'>
                Retail Intelligence
            </div>
            <div style='font-size:.8rem; color:#64748B;'>
                Hybrid Predictive System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    PAGE = st.radio(
        "Navigation",
        options=[
            "📊 Executive Overview",
            "🔍 Dataset Explorer",
            "📈 Sales Analytics",
            "🔮 Demand Forecasting",
            "🚨 Anomaly Detection",
            "📡 Live Monitoring",
            "💼 Business Intelligence",
            "⚙️ Model Evaluation",
            "📥 Reports & Downloads",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("**UCI Online Retail II**\n\n"
               "Hybrid forecasting + anomaly detection")


ROUTES = {
    "📊 Executive Overview":    overview.render,
    "🔍 Dataset Explorer":      data_explorer.render,
    "📈 Sales Analytics":       sales_analytics.render,
    "🔮 Demand Forecasting":    forecasting_page.render,
    "🚨 Anomaly Detection":     anomaly_page.render,
    "📡 Live Monitoring":       monitoring_page.render,
    "💼 Business Intelligence": business_insights_page.render,
    "⚙️ Model Evaluation":      evaluation_page.render,
    "📥 Reports & Downloads":   reports_page.render,
}

try:
    ROUTES[PAGE]()
except Exception as e:
    st.error(f"**Error rendering page:** {e}")
    with st.expander("Traceback"):
        import traceback
        st.code(traceback.format_exc())