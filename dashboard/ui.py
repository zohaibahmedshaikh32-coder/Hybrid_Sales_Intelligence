"""
ui.py — shared UI helpers for the Retail Intelligence dashboard.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.config import PROCESSED_DIR


PALETTE = {
    "primary":   "#2563EB",
    "secondary": "#0EA5E9",
    "success":   "#10B981",
    "warning":   "#F59E0B",
    "danger":    "#EF4444",
    "muted":     "#64748B",
    "text":      "#E2E8F0",
}


@st.cache_data(show_spinner="Loading dataset …")
def load_parquet(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        st.error(f"File **{path.name}** not found. "
                 f"Run the training pipeline first.")
        st.stop()
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading JSON …")
def load_json(name: str) -> dict:
    path = PROCESSED_DIR / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div style="
            padding: 1rem 1.25rem;
            border-left: 4px solid {PALETTE['primary']};
            background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
            border-radius: 8px;
            margin-bottom: 1.5rem;">
            <h1 style="margin:0; color:{PALETTE['text']};
                       font-size:1.85rem; font-weight:700;">{title}</h1>
            <p style="margin:.35rem 0 0 0; color:{PALETTE['muted']};
                      font-size:.95rem;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, delta: str | None = None,
             delta_color: str = "normal", icon: str = "") -> None:
    st.metric(label=f"{icon} {label}".strip(), value=value,
              delta=delta, delta_color=delta_color)


def empty_state(message: str, icon: str = "📭") -> None:
    st.markdown(
        f"""
        <div style="
            padding: 3rem 1rem;
            text-align: center;
            color: {PALETTE['muted']};
            border: 1px dashed {PALETTE['muted']};
            border-radius: 12px;">
            <div style="font-size: 3rem;">{icon}</div>
            <p style="margin:.5rem 0 0 0;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(
        f"<h3 style='color:{PALETTE['text']}; margin-top:1.5rem;'>{title}</h3>",
        unsafe_allow_html=True)


def fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def fmt_gbp(n) -> str:
    try:
        return f"£{float(n):,.0f}"
    except Exception:
        return str(n)


def fmt_pct(n, digits: int = 2) -> str:
    try:
        return f"{float(n):.{digits}f}%"
    except Exception:
        return str(n)