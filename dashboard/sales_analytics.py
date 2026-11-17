"""
sales_analytics.py — Page 3: Sales Analytics.

Six interactive charts answering the business questions:
    - What sells most (by revenue and by quantity)?
    - Where is the revenue coming from (countries)?
    - How does it trend month-over-month and year-over-year?
    - What does the revenue distribution look like?
    - How large is the return/cancellation stream?
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.ui import (
    PALETTE, empty_state, fmt_gbp, fmt_int, load_parquet,
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
        "Sales Analytics",
        "Product, geographic, temporal and returns-based view of the "
        "cleaned transaction stream.",
    )

    if not (PROCESSED_DIR / "valid_sales.parquet").exists():
        st.error("No cleaned sales data. Run `python run_eda.py` first.")
        return

    sales = load_parquet("valid_sales")
    returns = load_parquet("returns") \
              if (PROCESSED_DIR / "returns.parquet").exists() else pd.DataFrame()

    # Enrich once
    sales = sales.copy()
    sales["Revenue"] = sales["Quantity"] * sales["Price"]
    sales["Date"]    = pd.to_datetime(sales["InvoiceDate"]).dt.normalize()
    sales["Month"]   = sales["InvoiceDate"].dt.to_period("M").astype(str)
    sales["Year"]    = sales["InvoiceDate"].dt.year

    # =================================================================
    # 1. Top products — revenue vs quantity (side-by-side)
    # =================================================================
    section("1. Top Products")
    top_n = st.slider("Show top N products", 5, 30, 10, key="top_n_prod")

    by_rev = (sales.groupby("Description", as_index=False)["Revenue"]
                     .sum().nlargest(top_n, "Revenue"))
    by_qty = (sales.groupby("Description", as_index=False)["Quantity"]
                     .sum().nlargest(top_n, "Quantity"))

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(by_rev.iloc[::-1], x="Revenue", y="Description",
                     orientation="h",
                     color_discrete_sequence=[PALETTE["primary"]])
        fig.update_layout(title=f"Top {top_n} by Revenue (£)",
                          height=420, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(by_qty.iloc[::-1], x="Quantity", y="Description",
                     orientation="h",
                     color_discrete_sequence=[PALETTE["success"]])
        fig.update_layout(title=f"Top {top_n} by Quantity",
                          height=420, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # 2. Country-wise sales
    # =================================================================
    section("2. Country-wise Sales")
    top_countries = (sales.groupby("Country", as_index=False)["Revenue"]
                          .sum().nlargest(15, "Revenue"))
    fig = px.bar(top_countries.iloc[::-1], x="Revenue", y="Country",
                 orientation="h",
                 color_discrete_sequence=[PALETTE["secondary"]])
    fig.update_layout(title="Top 15 Countries by Revenue (£)",
                      height=460, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # 3. Monthly comparison across years
    # =================================================================
    section("3. Monthly Comparison (Year-over-Year)")
    monthly = (sales.groupby(["Year", "Month"], as_index=False)["Revenue"]
                    .sum())
    monthly["MonthShort"] = monthly["Month"].str[-2:]

    fig = px.line(monthly, x="MonthShort", y="Revenue",
                  color="Year", markers=True,
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(
        title="Monthly Revenue by Year",
        xaxis_title="Month", yaxis_title="Revenue (£)",
        height=420, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # 4. Product demand trends (7-day rolling, top 5 products)
    # =================================================================
    section("4. Demand Trends — Top 5 Products")
    top5 = (sales.groupby("Description")["Quantity"].sum()
                 .nlargest(5).index.tolist())
    sub = sales[sales["Description"].isin(top5)]
    daily = (sub.groupby(["Date", "Description"], as_index=False)["Quantity"]
                .sum().sort_values("Date"))
    daily["Roll7"] = (daily.groupby("Description")["Quantity"]
                           .transform(lambda s: s.rolling(7, min_periods=1).mean()))

    fig = px.line(daily, x="Date", y="Roll7", color="Description",
                  color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(title="7-day Rolling Demand — Top 5 Products",
                      xaxis_title="Date", yaxis_title="Units/day (7d avg)",
                      height=440, **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # 5. Revenue distribution (log-scale histogram)
    # =================================================================
    section("5. Revenue Distribution")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.histogram(sales, x="Revenue", nbins=120, log_y=True,
                           color_discrete_sequence=[PALETTE["primary"]])
        fig.update_layout(title="Revenue per line item (log-count)",
                          xaxis_title="Revenue (£)",
                          yaxis_title="Count (log)",
                          height=400, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("##### Summary statistics")
        stats = sales["Revenue"].describe(
            percentiles=[.5, .9, .99]).to_frame("Revenue (£)")
        stats["Revenue (£)"] = stats["Revenue (£)"].round(2)
        st.dataframe(stats, use_container_width=True)

    # =================================================================
    # 6. Returns / cancellations
    # =================================================================
    section("6. Returns & Cancellations")
    if returns.empty:
        empty_state("No returns recorded in the dataset.", icon="✅")
    else:
        returns = returns.copy()
        returns["AbsRevenue"] = returns["Quantity"].abs() * returns["Price"].abs()
        returns["Month"] = (pd.to_datetime(returns["InvoiceDate"])
                            .dt.to_period("M").astype(str))

        c1, c2 = st.columns(2)
        with c1:
            ret_monthly = (returns.groupby("Month", as_index=False)["AbsRevenue"]
                                  .sum())
            fig = px.bar(ret_monthly, x="Month", y="AbsRevenue",
                         color_discrete_sequence=[PALETTE["danger"]])
            fig.update_layout(title="Monthly return value (£)",
                              xaxis_title="Month",
                              yaxis_title="Value (£)",
                              height=380, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            top_returns = (returns.groupby("Description", as_index=False)["Quantity"]
                                  .sum()
                                  .assign(AbsQty=lambda d: d["Quantity"].abs())
                                  .nlargest(10, "AbsQty"))
            fig = px.bar(top_returns.iloc[::-1], x="AbsQty", y="Description",
                         orientation="h",
                         color_discrete_sequence=[PALETTE["warning"]])
            fig.update_layout(title="Top 10 most-returned products",
                              xaxis_title="Units returned",
                              height=380, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        total_sales = float(sales["Revenue"].sum())
        total_ret   = float(returns["AbsRevenue"].sum())
        ratio = 100 * total_ret / total_sales if total_sales else 0
        st.metric("Return value / sales value",
                  f"{ratio:.2f}%",
                  help="Total return/cancellation value ÷ total sales value")