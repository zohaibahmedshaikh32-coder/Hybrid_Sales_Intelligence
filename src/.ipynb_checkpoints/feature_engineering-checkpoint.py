"""feature_engineering.py — gap-filled, leakage-safe feature builders."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from src.config import (LAG_DAYS, ROLL_WINDOWS, OUTLIER_UPPER_QUANTILE,
                        OUTLIER_LOWER_QUANTILE, MIN_ACTIVE_DAYS,
                        MIN_TOTAL_QTY, MIN_AVG_DAILY_QTY)

log = logging.getLogger(__name__)


def build_daily_product_grid(sales: pd.DataFrame) -> pd.DataFrame:
    sales = sales.copy()
    sales["Revenue"] = sales["Quantity"] * sales["Price"]
    sales["Date"]    = sales["InvoiceDate"].dt.normalize()

    agg = (sales.groupby(["Date", "StockCode"], as_index=False)
                .agg(
                    Description=("Description", "first"),
                    Quantity=("Quantity", "sum"),
                    Revenue=("Revenue", "sum"),
                    AvgPrice=("Price", "mean"),
                    NumTransactions=("Invoice", "nunique"),
                    UniqueCustomers=("Customer ID",
                                     lambda s: s.dropna().nunique()),
                ))

    full_dates = pd.date_range(agg["Date"].min(), agg["Date"].max(), freq="D")
    pieces = []
    for code, g in agg.groupby("StockCode", sort=False):
        g = g.set_index("Date").reindex(full_dates)
        g["StockCode"]  = code
        g["Description"] = g["Description"].ffill().bfill()
        g[["Quantity", "Revenue", "NumTransactions", "UniqueCustomers"]] = (
            g[["Quantity", "Revenue", "NumTransactions", "UniqueCustomers"]]
            .fillna(0))
        med = g["AvgPrice"].median()
        g["AvgPrice"] = g["AvgPrice"].fillna(med if pd.notna(med) else 0.0)
        g.index.name = "Date"
        pieces.append(g.reset_index())

    grid = pd.concat(pieces, ignore_index=True)
    dt = grid["Date"]
    grid["Year"]         = dt.dt.year
    grid["Month"]        = dt.dt.month
    grid["Day"]          = dt.dt.day
    grid["DayOfWeek"]    = dt.dt.dayofweek
    grid["WeekOfYear"]   = dt.dt.isocalendar().week.astype(int)
    grid["Quarter"]      = dt.dt.quarter
    grid["IsWeekend"]    = (grid["DayOfWeek"] >= 5).astype(int)
    grid["IsMonthStart"] = dt.dt.is_month_start.astype(int)
    grid["IsMonthEnd"]   = dt.dt.is_month_end.astype(int)
    log.info("Daily grid (gap-filled): %d rows x %d cols", *grid.shape)
    return grid


def filter_forecastable_products(grid: pd.DataFrame):
    stats = (grid.groupby("StockCode")
                 .agg(active_days=("Quantity", lambda s: int((s > 0).sum())),
                      total_qty=("Quantity", "sum"),
                      avg_daily=("Quantity", "mean"))
                 .reset_index())
    keep = stats[
        (stats["active_days"] >= MIN_ACTIVE_DAYS)
        & (stats["total_qty"]  >= MIN_TOTAL_QTY)
        & (stats["avg_daily"]  >= MIN_AVG_DAILY_QTY)
    ]["StockCode"]
    filtered = grid[grid["StockCode"].isin(keep)].copy()
    log.info("Product filter: kept %d / %d products (%d rows)",
             keep.nunique(), grid["StockCode"].nunique(), len(filtered))
    return filtered, stats


def clip_outliers_per_product(grid: pd.DataFrame) -> pd.DataFrame:
    out = []
    for code, g in grid.groupby("StockCode", sort=False):
        g = g.copy()
        for col in ("Quantity", "Revenue"):
            lo = g[col].quantile(OUTLIER_LOWER_QUANTILE)
            hi = g[col].quantile(OUTLIER_UPPER_QUANTILE)
            if hi > lo:
                g[col] = g[col].clip(lower=lo, upper=hi)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_leakage_safe_ts_features(grid: pd.DataFrame) -> pd.DataFrame:
    """
    Past-only lags and rolling stats.
    `shift(1)` before rolling -> feature(t) uses only data <= t-1.
    """
    frames = []
    for code, g in grid.groupby("StockCode", sort=False):
        g = g.sort_values("Date").copy()
        for lag in LAG_DAYS:
            g[f"lag_qty_{lag}"] = g["Quantity"].shift(lag)
            g[f"lag_rev_{lag}"] = g["Revenue"].shift(lag)
        shifted = g["Quantity"].shift(1)
        for w in ROLL_WINDOWS:
            g[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
            g[f"roll_std_{w}"]  = shifted.rolling(w, min_periods=1).std()
            g[f"roll_max_{w}"]  = shifted.rolling(w, min_periods=1).max()
        g["cum_qty"]      = g["Quantity"].shift(1).expanding().sum()
        g["cum_revenue"]  = g["Revenue"].shift(1).expanding().sum()
        g["demand_trend"] = (g["roll_mean_7"] - g["roll_mean_30"]).fillna(0.0)
        frames.append(g)

    feat = pd.concat(frames, ignore_index=True)
    max_lag = max(max(LAG_DAYS), max(ROLL_WINDOWS))
    before = len(feat)
    feat = feat.dropna(subset=[f"lag_qty_{max(LAG_DAYS)}",
                               f"roll_mean_{max(ROLL_WINDOWS)}"])
    log.info("Dropped %d cold-start rows (need >= %d days history)",
             before - len(feat), max_lag)
    return feat.reset_index(drop=True)


def build_anomaly_features(sales: pd.DataFrame,
                           returns: pd.DataFrame) -> pd.DataFrame:
    sales = sales.copy()
    sales["Revenue"] = sales["Quantity"] * sales["Price"]
    sales["IsReturn"] = 0
    rets = returns.copy()
    rets["Revenue"] = rets["Quantity"].abs() * rets["Price"].abs()
    rets["IsReturn"] = 1

    combined = pd.concat([sales, rets], ignore_index=True, sort=False)
    combined["AbsQuantity"] = combined["Quantity"].abs()
    combined["AbsRevenue"]  = combined["Revenue"].abs()
    combined["Hour"]        = combined["InvoiceDate"].dt.hour
    combined["DayOfWeek"]   = combined["InvoiceDate"].dt.dayofweek

    prod_stats = (combined.groupby("StockCode")
                          .agg(prod_mean_price=("Price", "mean"),
                               prod_std_price=("Price", "std"),
                               prod_mean_qty=("AbsQuantity", "mean"),
                               prod_std_qty=("AbsQuantity", "std"))
                          .reset_index())
    combined = combined.merge(prod_stats, on="StockCode", how="left")

    eps = 1e-9
    combined["price_zscore"] = (
        (combined["Price"] - combined["prod_mean_price"])
        / (combined["prod_std_price"] + eps))
    combined["qty_zscore"] = (
        (combined["AbsQuantity"] - combined["prod_mean_qty"])
        / (combined["prod_std_qty"] + eps))

    log.info("Anomaly feature table: %d rows x %d cols", *combined.shape)
    return combined.reset_index(drop=True)


def compute_product_quality(stats: pd.DataFrame) -> pd.DataFrame:
    stats = stats.copy()
    stats["score"] = (
        stats["active_days"].rank(pct=True) * 0.5
        + stats["total_qty"].rank(pct=True) * 0.3
        + (stats["avg_daily"].clip(upper=100)).rank(pct=True) * 0.2
    ).round(3)
    return stats.sort_values("score", ascending=False).reset_index(drop=True)

def build_forecasting_features(
    grid: pd.DataFrame,
    clip_after_audit: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build forecasting features with an EXPLICIT order:

        grid
          -> add_leakage_safe_ts_features   (producible from raw)
          -> leakage_audit                  (proves past-only)
          -> clip_outliers_per_product      (variance stabilisation)

    Returns
    -------
    features_audit : the table used for the leakage audit
                     (unclipped, fully reproducible from raw quantities)
    features_final : the table you actually train on
                     (clipped, extreme values suppressed)
    """
    # Step A — add lags/rolls to the raw gap-filled grid (no clipping yet)
    feat_audit = add_leakage_safe_ts_features(grid)

    if not clip_after_audit:
        return feat_audit, feat_audit

    # Step B — audit runs inside run_eda on `feat_audit`
    # Step C — clip ONLY the numeric quantity/revenue columns for the
    #          final training table. We deliberately do NOT re-run the
    #          rolling features after clipping, because clipping is a
    #          static per-product transform (no future information).
    feat_final = feat_audit.copy()
    for code, g in feat_final.groupby("StockCode", sort=False):
        for col in ("Quantity", "Revenue"):
            lo = g[col].quantile(OUTLIER_LOWER_QUANTILE)
            hi = g[col].quantile(OUTLIER_UPPER_QUANTILE)
            if hi > lo:
                feat_final.loc[g.index, col] = g[col].clip(lower=lo, upper=hi)

    log.info("Forecasting features built: audit=%s  final=%s",
             feat_audit.shape, feat_final.shape)
    return feat_audit, feat_final
