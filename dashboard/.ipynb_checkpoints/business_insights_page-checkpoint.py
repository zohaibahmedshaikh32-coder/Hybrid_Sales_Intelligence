"""
business_insights_page.py — Page 7: Business Intelligence.

Translates the hybrid-analysis + business-mitigation artefacts into
an actionable dashboard with severity filters, category breakdowns,
and per-mitigation detail cards.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.ui import (
    PALETTE, empty_state, fmt_int, load_json, load_parquet,
    page_header, section,
)
from src.config import PROCESSED_DIR


PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color=PALETTE["text"]),
    margin=dict(l=10, r=10, t=40, b=10),
)


def render() -> None:
    page_header(
        "Business Intelligence",
        "Model-generated mitigation recommendations grouped by severity "
        "and business category.",
    )

    if not (PROCESSED_DIR / "business_mitigations.json").exists():
        empty_state(
            "Business mitigations not found. Run "
            "`python -m scripts.run_insights` first.", icon="⚙️")
        return

    payload = load_json("business_mitigations")
    mitigations = payload.get("mitigations", [])
    if not mitigations:
        empty_state("No mitigations were generated.", icon="📭")
        return

    df = pd.DataFrame(mitigations)

    # -----------------------------------------------------------------
    # Scope warning
    # -----------------------------------------------------------------
    st.warning(
        "⚠️ **Every recommendation below is a hypothesis for human "
        "review.** These are unsupervised model outputs, not confirmed "
        "business events. Cross-check against inventory, pricing, and "
        "customer data before acting.")

    # -----------------------------------------------------------------
    # 1. KPI cards
    # -----------------------------------------------------------------
    section("Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total mitigations", fmt_int(len(df)))
    with c2: st.metric("High severity",
                       fmt_int((df["severity"] == "high").sum()))
    with c3: st.metric("Medium severity",
                       fmt_int((df["severity"] == "medium").sum()))
    with c4: st.metric("Categories", fmt_int(df["category"].nunique()))

    # -----------------------------------------------------------------
    # 2. Category breakdown (stacked bars)
    # -----------------------------------------------------------------
    section("Mitigation Categories")
    counts = (df.groupby(["category", "severity"], as_index=False)
                .size().rename(columns={"size": "n"}))
    fig = px.bar(counts, x="n", y="category", color="severity",
                 orientation="h",
                 color_discrete_map={"high": PALETTE["danger"],
                                     "medium": PALETTE["warning"]},
                 barmode="stack")
    fig.update_layout(title="Mitigations by category and severity",
                      xaxis_title="Count", yaxis_title="",
                      height=380, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------
    # 3. Filters
    # -----------------------------------------------------------------
    section("Filter & Explore")
    c1, c2 = st.columns([1, 2])
    with c1:
        severities = st.multiselect(
            "Severity", options=sorted(df["severity"].unique()),
            default=sorted(df["severity"].unique()),
            key="bi_sev_filter")
    with c2:
        cats = st.multiselect(
            "Category", options=sorted(df["category"].unique()),
            default=sorted(df["category"].unique()),
            key="bi_cat_filter")

    view = df[df["severity"].isin(severities) & df["category"].isin(cats)]
    st.write(f"**{fmt_int(len(view))} mitigations match your filters**")

    # -----------------------------------------------------------------
    # 4. Table
    # -----------------------------------------------------------------
    st.dataframe(
        view[["category", "severity", "StockCode", "Description",
              "evidence", "suggested_action"]]
        .sort_values(["severity", "category"]),
        use_container_width=True, hide_index=True, height=340,
    )
    st.download_button(
        "⬇️ Download filtered mitigations (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="mitigations_filtered.csv",
        mime="text/csv",
    )

    # -----------------------------------------------------------------
    # 5. Detail cards per category
    # -----------------------------------------------------------------
    section("Detailed Recommendations")
    for cat in sorted(view["category"].unique()):
        with st.expander(f"**{cat}** "
                         f"({int((view['category'] == cat).sum())} items)",
                         expanded=False):
            for _, r in view[view["category"] == cat].iterrows():
                st.markdown(
                    f"**{r['Description']}**  (`{r['StockCode']}`)  "
                    f"— *{r['severity']}*  \n"
                    f"Evidence: {r['evidence']}  \n"
                    f"**Suggested action:** {r['suggested_action']}  \n"
                    f"*Caveat:* {r['caveat']}")
                st.divider()