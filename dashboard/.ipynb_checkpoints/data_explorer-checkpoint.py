"""
data_explorer.py — Page 2: Dataset Explorer.

Filterable view of the cleaned dataset, with a data-quality summary
report so the user can see exactly what was cleaned and why.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.ui import (
    empty_state, fmt_int, page_header, section,
)


def render() -> None:
    page_header(
        "Dataset Explorer",
        "Explore the cleaned transactions with filters, quality report, "
        "and CSV export.",
    )

    from dashboard.ui import load_parquet
    from src.config import PROCESSED_DIR

    if not (PROCESSED_DIR / "valid_sales.parquet").exists():
        st.error("No cleaned sales data. Run `python run_eda.py` first.")
        return
    sales = load_parquet("valid_sales")

    # ---- Data quality summary -------------------------------------
    section("Data Quality Summary")
    dq_path = PROCESSED_DIR / "data_quality_report.csv"
    if dq_path.exists():
        dq = pd.read_csv(dq_path)
        cols = st.columns(4)
        metrics = {row["Metric"]: row["Value"] for _, row in dq.iterrows()}
        with cols[0]:
            st.metric("Raw rows", fmt_int(metrics.get("Raw rows", "—")))
        with cols[1]:
            st.metric("Duplicates removed",
                      fmt_int(metrics.get("Duplicates removed", "—")))
        with cols[2]:
            st.metric("Valid sales rows",
                      fmt_int(metrics.get("Valid sales rows", "—")))
        with cols[3]:
            st.metric("Returns rows",
                      fmt_int(metrics.get("Returns rows", "—")))

        with st.expander("Full quality report"):
            st.dataframe(dq, use_container_width=True, hide_index=True)

    # ---- Filters ---------------------------------------------------
    section("Filter Transactions")
    c1, c2, c3 = st.columns(3)

    with c1:
        countries = sorted(sales["Country"].dropna().unique().tolist())
        chosen_countries = st.multiselect(
            "Country",
            options=countries,
            default=countries[:5] if len(countries) >= 5 else countries,
        )

    with c2:
        # Product search box
        query = st.text_input("Search product description or code",
                              placeholder="e.g. CHRISTMAS, 85123A")

    with c3:
        date_min = sales["InvoiceDate"].min().date()
        date_max = sales["InvoiceDate"].max().date()
        d1, d2 = st.date_input(
            "Date range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
        ) if False else (date_min, date_max)
        # Simple date range:
        dr = st.date_input(
            "Date range",
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
            key="explorer_dates",
        )

    # Apply filters
    df = sales.copy()
    if chosen_countries:
        df = df[df["Country"].isin(chosen_countries)]
    if query:
        q = query.lower().strip()
        mask = (df["Description"].astype(str).str.lower().str.contains(q,
                    na=False, regex=False)
                | df["StockCode"].astype(str).str.lower().str.contains(q,
                    na=False, regex=False))
        df = df[mask]
    if isinstance(dr, tuple) and len(dr) == 2:
        start, end = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
        df = df[(df["InvoiceDate"] >= start) & (df["InvoiceDate"] <= end + pd.Timedelta(days=1))]

    if df.empty:
        empty_state("No transactions match your filters.", icon="🔍")
        return

    # ---- Results table -------------------------------------------
    section(f"Transactions ({fmt_int(len(df))} rows)")

    display_cols = ["Invoice", "StockCode", "Description",
                    "Quantity", "Price", "InvoiceDate",
                    "Customer ID", "Country"]
    show_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[show_cols].head(1_000),
        use_container_width=True,
        hide_index=True,
        height=420,
    )
    if len(df) > 1_000:
        st.caption(f"Showing first 1,000 of {fmt_int(len(df))} rows. "
                   f"Download the full result below.")

    # ---- Download --------------------------------------------------
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=df[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="filtered_transactions.csv",
        mime="text/csv",
    )

    # ---- Missing-value summary -----------------------------------
    with st.expander("Missing values in the current selection"):
        miss = (df[show_cols].isna().sum().to_frame("# missing")
                              .assign(**{"% missing":
                                       (df[show_cols].isna().mean() * 100).round(2)}))
        st.dataframe(miss, use_container_width=True)