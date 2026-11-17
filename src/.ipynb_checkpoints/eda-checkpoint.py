"""eda.py — task-driven EDA that answers MODEL questions."""
from __future__ import annotations
import json, logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.config import FIGURES_DIR, PROCESSED_DIR

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 110, "figure.autolayout": True})
INSIGHTS: dict = {}


def _save(fig, name):
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_forecastability(daily, stats):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    sns.histplot(stats["active_days"], bins=50, ax=axes[0], color="#2563EB")
    axes[0].axvline(60, color="#EF4444", ls="--", label="MIN_ACTIVE_DAYS=60")
    axes[0].set_title("Active days per product"); axes[0].legend()
    zero_frac = daily.groupby("StockCode")["Quantity"] \
                     .apply(lambda s: float((s == 0).mean()))
    sns.histplot(zero_frac, bins=50, ax=axes[1], color="#0EA5E9")
    axes[1].set_title("Fraction of zero-sale days per product")
    sns.scatterplot(data=stats, x="avg_daily", y="total_qty",
                    alpha=0.35, ax=axes[2], color="#10B981")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_title("Avg daily qty vs total qty")
    _save(fig, "01_forecastability")
    INSIGHTS["median_active_days"]        = float(stats["active_days"].median())
    INSIGHTS["median_zero_day_frac"]      = float(zero_frac.median())
    INSIGHTS["pct_products_forecastable"] = float(
        (stats["active_days"] >= 60).mean() * 100)


def plot_seasonality(daily):
    by_dow = daily.groupby("DayOfWeek")["Quantity"].mean()
    by_month = daily.groupby("Month")["Quantity"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.barplot(x=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                y=by_dow.values, ax=axes[0], palette="Blues")
    axes[0].set_title("Mean daily qty by weekday")
    sns.lineplot(x=by_month.index, y=by_month.values,
                 marker="o", ax=axes[1], color="#2563EB")
    axes[1].set_title("Mean daily qty by month")
    axes[1].set_xticks(range(1, 13))
    _save(fig, "02_seasonality")
    INSIGHTS["weekday_signal_cv"] = float(by_dow.std() / by_dow.mean())
    INSIGHTS["month_signal_cv"]   = float(by_month.std() / by_month.mean())


def plot_distribution_shape(daily):
    q = daily["Quantity"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    sns.histplot(q, bins=100, ax=axes[0], color="#2563EB")
    axes[0].set_title("Quantity (raw)")
    sns.histplot(np.log1p(q), bins=100, ax=axes[1], color="#10B981")
    axes[1].set_title("log1p(Quantity)")
    sns.boxplot(x=q, ax=axes[2])
    axes[2].set_title("Quantity (outliers)")
    _save(fig, "03_distribution_shape")
    INSIGHTS["qty_skew_raw"]   = float(q.skew())
    INSIGHTS["qty_skew_log1p"] = float(np.log1p(q).skew())
    INSIGHTS["qty_upper_p99"]  = float(q.quantile(0.99))


def plot_autocorrelation(daily, top_n=3):
    top = (daily.groupby("StockCode")["Quantity"].sum()
                .sort_values(ascending=False).head(top_n).index)
    fig, axes = plt.subplots(1, top_n, figsize=(16, 4), sharey=True)
    if top_n == 1: axes = [axes]
    for ax, code in zip(axes, top):
        s = daily[daily["StockCode"] == code].sort_values("Date")["Quantity"]
        acf = [s.autocorr(lag=k) for k in range(1, 31)]
        ax.bar(range(1, 31), acf, color="#2563EB")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"ACF: {code}")
        ax.set_xlabel("lag (days)")
    axes[0].set_ylabel("autocorrelation")
    _save(fig, "04_autocorrelation")
    acs = []
    for code in top:
        s = daily[daily["StockCode"] == code].sort_values("Date")["Quantity"]
        acs.append(float(s.autocorr(lag=7)))
    INSIGHTS["mean_acf_lag7_top_products"] = float(np.mean(acs))


def plot_anomaly_signature(anom_feat):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    sns.histplot(anom_feat["qty_zscore"].clip(-5, 5), bins=100,
                 ax=axes[0], color="#EF4444")
    axes[0].set_title("qty_zscore (clipped)")
    sns.histplot(anom_feat["price_zscore"].clip(-5, 5), bins=100,
                 ax=axes[1], color="#F59E0B")
    axes[1].set_title("price_zscore (clipped)")
    sns.histplot(np.log1p(anom_feat["AbsRevenue"]), bins=100,
                 ax=axes[2], color="#2563EB")
    axes[2].set_title("log1p(AbsRevenue)")
    _save(fig, "05_anomaly_signature")
    q1, q99 = anom_feat["qty_zscore"].quantile([0.01, 0.99])
    INSIGHTS["qty_zscore_p1_p99"] = [float(q1), float(q99)]
    INSIGHTS["pct_extreme_qty"]   = float(
        ((anom_feat["qty_zscore"] > 3)
         | (anom_feat["qty_zscore"] < -3)).mean() * 100)


def plot_returns_balance(sales, returns):
    if returns.empty:
        return
    sales   = sales.copy()
    returns = returns.copy()
    if "Revenue" not in sales.columns:
        sales["Revenue"] = sales["Quantity"] * sales["Price"]
    if "Revenue" not in returns.columns:
        returns["Revenue"] = returns["Quantity"].abs() * returns["Price"].abs()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    mr = returns.assign(M=returns["InvoiceDate"].dt.to_period("M")) \
                .groupby("M").size()
    ms = sales.assign(M=sales["InvoiceDate"].dt.to_period("M")) \
              .groupby("M").size()
    mr.index = mr.index.astype(str)
    ms.index = ms.index.astype(str)
    axes[0].plot(ms.index, ms.values, color="#2563EB", label="Sales lines")
    axes[0].plot(mr.index, mr.values, color="#EF4444", label="Returns")
    axes[0].set_title("Sales vs returns volume")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend()
    ratio = (mr / ms).dropna()
    sns.lineplot(x=ratio.index, y=ratio.values,
                 marker="o", ax=axes[1], color="#F59E0B")
    axes[1].set_title("Return / sale ratio per month")
    axes[1].tick_params(axis="x", rotation=45)
    _save(fig, "06_returns_balance")
    INSIGHTS["mean_return_ratio_pct"] = float(ratio.mean() * 100)



def plot_top_products(sales, stats, top_n=15):
    sales = sales.copy()
    # Defensive: ensure Revenue exists (sales may be the raw cleaned table)
    if "Revenue" not in sales.columns:
        sales["Revenue"] = sales["Quantity"] * sales["Price"]

    top = (sales.groupby("Description")["Revenue"].sum()
                .sort_values(ascending=False).head(top_n))
    codes = (sales[sales["Description"].isin(top.index)]
             .groupby("Description")["StockCode"].first())
    active = codes.map(stats.set_index("StockCode")["active_days"])
    colors = ["#10B981" if a >= 60 else "#F59E0B"
              for a in active.values]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
    ax.set_title(f"Top {top_n} by revenue - green = forecastable (>=60 days)")
    ax.set_xlabel("Revenue (GBP)")
    _save(fig, "07_top_products")


def plot_product_funnel(stats):
    steps = [
        ("All products", len(stats)),
        ("active_days>=60", int((stats["active_days"] >= 60).sum())),
        ("+ total_qty>=100",
            int(((stats["active_days"] >= 60)
                 & (stats["total_qty"] >= 100)).sum())),
        ("+ avg_daily>=1",
            int(((stats["active_days"] >= 60)
                 & (stats["total_qty"] >= 100)
                 & (stats["avg_daily"] >= 1)).sum())),
    ]
    labels, values = zip(*steps)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], values[::-1],
            color=["#EF4444","#F59E0B","#0EA5E9","#10B981"])
    for i, v in enumerate(values[::-1]):
        ax.text(v, i, f"  {v}", va="center")
    ax.set_title("Forecastable-product funnel")
    _save(fig, "08_product_funnel")
    INSIGHTS["products_all"]          = steps[0][1]
    INSIGHTS["products_forecastable"] = steps[-1][1]


def run_all_eda(sales, returns, daily, stats, anom_feat):
    log.info("Running task-driven EDA ...")
    plot_forecastability(daily, stats)
    plot_seasonality(daily)
    plot_distribution_shape(daily)
    plot_autocorrelation(daily)
    plot_anomaly_signature(anom_feat)
    plot_returns_balance(sales, returns)
    plot_top_products(sales, stats)
    plot_product_funnel(stats)
    with open(PROCESSED_DIR / "eda_insights.json", "w") as f:
        json.dump(INSIGHTS, f, indent=2)
    log.info("Insights written -> eda_insights.json")
