"""
anomaly_page.py — Page 5: Anomaly Detection.

Interactive view of the three unsupervised models:
    - Model selector (K-Means / Hierarchical / Isolation Forest)
    - Anomaly distribution + scores
    - Product-level anomaly hotspots
    - Flagged-transaction table (filterable)
    - Inter-model agreement summary
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
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


@st.cache_data(show_spinner="Loading anomaly metadata …")
def _load_meta() -> dict:
    p = BASE_DIR / "models" / "anomaly_detection" / "metadata.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Running Isolation Forest on full dataset …")
def _recompute_flags() -> pd.DataFrame:
    """
    Run the saved Isolation Forest over the anomaly feature table
    (with the SAME StandardScaler used in Phase 5/6) to produce
    per-row anomaly flags. Cached so the page loads instantly after
    the first computation (~30 s).
    """
    from sklearn.preprocessing import StandardScaler
    import joblib

    df = pd.read_parquet(PROCESSED_DIR / "anomaly_features.parquet")
    meta = _load_meta()
    feat_cols = meta["feature_cols"]
    iso = joblib.load(
        BASE_DIR / "models" / "anomaly_detection" / "isolation_forest.joblib")

    X = df[feat_cols].astype(float).values
    X = StandardScaler().fit(X).transform(X)
    preds  = iso.predict(X)
    scores = -iso.score_samples(X)

    df["anomaly_flag"]  = (preds == -1).astype(int)
    df["anomaly_score"] = scores
    df["Date"] = pd.to_datetime(
        df["Date"] if "Date" in df.columns else df["InvoiceDate"]
    ).dt.normalize()
    df["Revenue"] = df["Quantity"] * df["Price"]
    return df


@st.cache_data(show_spinner="Loading agreement data …")
def _load_agreement() -> dict:
    """Load the inter-model agreement artefacts from Phase 5."""
    from dashboard.ui import load_json
    metrics = load_json("business_mitigations")   # anything cached
    # Re-derive counts from persisted per-model metrics
    meta = _load_meta()
    return {
        "chosen_k": meta.get("chosen_kmeans_k", "?"),
        "metrics":  meta.get("metrics", []),
    }


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------
def render() -> None:
    page_header(
        "Anomaly Detection",
        "Unsupervised view of unusual transactions across three models.",
    )

    if not (PROCESSED_DIR / "anomaly_features.parquet").exists():
        st.error("Anomaly features not found. Run `python run_eda.py`.")
        return

    meta = _load_meta()
    if not meta:
        empty_state(
            "Anomaly models not found. Run "
            "`python -m scripts.train_anomaly` first.", icon="⚙️")
        return

    df = _recompute_flags()

    # -----------------------------------------------------------------
    # 1. Model cards
    # -----------------------------------------------------------------
    section("Model Overview")
    for m in meta.get("metrics", []):
        name = m.get("model", "?")
        with st.expander(f"**{name}**", expanded=(m is meta["metrics"][0])):
            cols = st.columns(4)
            for col, key in zip(cols, ("silhouette", "davies_bouldin",
                                       "anomaly_pct", "k")):
                if key in m:
                    label = {
                        "silhouette":       "Silhouette ↑",
                        "davies_bouldin":   "Davies-Bouldin ↓",
                        "anomaly_pct":      "Anomaly %",
                        "k":                "Chosen k",
                    }[key]
                    val = m[key]
                    if isinstance(val, float):
                        val = f"{val:.3f}" if abs(val) < 10 else f"{val:.1f}"
                    col.metric(label, val)

    # -----------------------------------------------------------------
    # 2. Anomaly score distribution + per-flag histograms
    # -----------------------------------------------------------------
    section("Anomaly Score Distribution")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            df, x="anomaly_score", color="anomaly_flag",
            nbins=100, log_y=True,
            color_discrete_map={0: PALETTE["secondary"],
                                1: PALETTE["danger"]},
            labels={"anomaly_flag": "Flag", "anomaly_score": "Anomaly score"},
        )
        fig.update_layout(
            title="Isolation Forest — score distribution (log count)",
            height=380, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        counts = df["anomaly_flag"].value_counts().rename(
            {0: "Normal", 1: "Anomaly"})
        fig = px.pie(values=counts.values, names=counts.index,
                     color=counts.index,
                     color_discrete_map={"Normal": PALETTE["secondary"],
                                          "Anomaly": PALETTE["danger"]})
        fig.update_layout(title="Flag balance",
                          height=380, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------
    # 3. Product-level hotspots
    # -----------------------------------------------------------------
    section("Product-level Anomaly Hotspots")
    prod = (df.groupby(["StockCode"], as_index=False)
              .agg(n_transactions=("anomaly_flag", "size"),
                   n_anomalies=("anomaly_flag", "sum")))
    prod["anomaly_rate"] = prod["n_anomalies"] / prod["n_transactions"]
    top_products = prod.nlargest(15, "n_anomalies")

    fig = px.bar(top_products.iloc[::-1], x="n_anomalies", y="StockCode",
                 orientation="h",
                 color_discrete_sequence=[PALETTE["danger"]])
    fig.update_layout(title="Top 15 products by # of flagged transactions",
                      xaxis_title="Flagged transactions",
                      yaxis_title="StockCode",
                      height=460, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------
    # 4. Flagged-transaction table (filterable)
    # -----------------------------------------------------------------
    section("Flagged Transactions")

    flagged = df[df["anomaly_flag"] == 1].copy()

    c1, c2, c3 = st.columns(3)
    with c1:
        countries = sorted(flagged.get("Country", pd.Series([], dtype=str))
                              .dropna().unique().tolist())
        chosen_country = st.multiselect(
            "Country", options=countries,
            default=countries[:3] if len(countries) >= 3 else countries,
            key="anomaly_country_filter",
        ) if countries else []
    with c2:
        text = st.text_input(
            "Search description / StockCode",
            placeholder="e.g. CHRISTMAS, 85123A",
            key="anomaly_text_filter")
    with c3:
        min_score = st.slider("Minimum anomaly score", 0.0, 1.0, 0.0, 0.01,
                              key="anomaly_score_filter")

    view = flagged.copy()
    if chosen_country and "Country" in view.columns:
        view = view[view["Country"].isin(chosen_country)]
    if text:
        q = text.lower()
        view = view[
            view["Description"].astype(str).str.lower().str.contains(q,
                na=False, regex=False)
            | view["StockCode"].astype(str).str.lower().str.contains(q,
                na=False, regex=False)
        ]
    if min_score > 0:
        view = view[view["anomaly_score"] >= min_score]

    display_cols = [c for c in (
        "Invoice", "StockCode", "Description", "Quantity",
        "Price", "Revenue", "Date", "anomaly_score")
        if c in view.columns]

    st.write(f"**{fmt_int(len(view))} flagged transactions match the filter**")
    st.dataframe(
        view[display_cols].sort_values("anomaly_score",
                                       ascending=False).head(500),
        use_container_width=True, hide_index=True, height=420,
    )
    if len(view) > 500:
        st.caption("Showing top 500 by anomaly score. "
                   "Narrow the filters or download below.")

    st.download_button(
        "⬇️ Download flagged transactions (CSV)",
        data=view[display_cols].to_csv(index=False).encode("utf-8"),
        file_name="flagged_transactions.csv",
        mime="text/csv",
    )