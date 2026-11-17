"""
bootstrap_project.py
Creates the complete Retail_Intelligence_System project tree with all files.
Run once:   python bootstrap_project.py
Then:       python -m pip install -r Retail_Intelligence_System/requirements.txt
"""
from pathlib import Path

ROOT = Path("Retail_Intelligence_System")

FILES: dict[str, str] = {}

# ─────────────────────────────────────────────────────────────────────
# requirements.txt
# ─────────────────────────────────────────────────────────────────────
FILES["requirements.txt"] = """pandas>=2.0
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
scikit-learn>=1.4
openpyxl>=3.1
pyarrow>=15.0
joblib>=1.3
scipy>=1.11
streamlit>=1.32
plotly>=5.20
reportlab>=4.1
pytest>=8.0
"""

# ─────────────────────────────────────────────────────────────────────
# README.md
# ─────────────────────────────────────────────────────────────────────
FILES["README.md"] = """# Retail Intelligence – Hybrid Predictive System

Sales Forecasting + Anomaly Detection on the UCI Online Retail II dataset.

## Quick start
1. `pip install -r requirements.txt`
2. Put `online_retail_II.xlsx` in `data/raw/`
3. `python run_eda.py` (cleans data, builds features, runs EDA, audits leakage)
4. `pytest tests/` (verifies no leakage, chronological split)
5. `streamlit run app.py` (dashboard – coming in the next phase)

## Folders
- `data/raw/`        → drop the Excel file here
- `data/processed/`  → parquet tables + reports produced by `run_eda.py`
- `reports/figures/` → EDA PNGs
- `src/`             → all pipeline modules
- `tests/`           → CI-ready checks
"""

# ─────────────────────────────────────────────────────────────────────
# src/__init__.py
# ─────────────────────────────────────────────────────────────────────
FILES["src/__init__.py"] = '"""Retail Intelligence source package."""\n'

# ─────────────────────────────────────────────────────────────────────
# src/config.py
# ─────────────────────────────────────────────────────────────────────
FILES["src/config.py"] = '''"""
config.py — ML-oriented configuration.
Every constant below is a *model-performance* decision, not aesthetic.
"""
from pathlib import Path

BASE_DIR       = Path(__file__).resolve().parent.parent
RAW_DIR        = BASE_DIR / "data" / "raw"
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
FIGURES_DIR    = BASE_DIR / "reports" / "figures"

for _p in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

RAW_FILE = RAW_DIR / "online_retail_II.xlsx"

COLUMN_ALIASES = {
    "InvoiceNo": "Invoice", "Invoice": "Invoice",
    "StockCode": "StockCode",
    "Description": "Description",
    "Quantity": "Quantity",
    "InvoiceDate": "InvoiceDate",
    "UnitPrice": "Price", "Price": "Price",
    "CustomerID": "Customer ID", "Customer ID": "Customer ID",
    "Country": "Country",
}

MIN_VALID_QTY       = 1
MIN_VALID_PRICE     = 0.01
CANCELLATION_PREFIX = "C"
ADMIN_STOCKCODES    = {
    "POST", "D", "M", "C2", "DOT", "BANK CHARGES",
    "PADS", "CRUK", "AMAZONFEE", "S", "B",
}

MIN_ACTIVE_DAYS   = 60
MIN_TOTAL_QTY     = 100
MIN_AVG_DAILY_QTY = 1.0

OUTLIER_UPPER_QUANTILE = 0.995
OUTLIER_LOWER_QUANTILE = 0.005

LAG_DAYS     = (1, 7, 14, 30)
ROLL_WINDOWS = (7, 14, 30, 90)
MIN_HISTORY  = 60

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

RANDOM_STATE = 42
'''

# ─────────────────────────────────────────────────────────────────────
# src/data_loader.py
# ─────────────────────────────────────────────────────────────────────
FILES["src/data_loader.py"] = '''"""data_loader.py — robust Excel loader with integrity checks."""
from __future__ import annotations
import logging
from typing import Dict, List
import pandas as pd
from src.config import RAW_FILE, COLUMN_ALIASES

log = logging.getLogger(__name__)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES}
    unknown = [c for c in df.columns if c not in COLUMN_ALIASES]
    if unknown:
        log.warning("Unknown columns ignored by schema: %s", unknown)
    return df.rename(columns=rename)


def load_raw_excel(path: str | None = None) -> pd.DataFrame:
    file_path = path or RAW_FILE
    xls = pd.ExcelFile(file_path)
    log.info("Sheets: %s", xls.sheet_names)
    frames: List[pd.DataFrame] = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        log.info("Sheet %-18s -> %6d x %2d cols", sheet, *df.shape)
        df = _normalise_columns(df)
        df["_source_sheet"] = sheet
        frames.append(df)
    if not frames:
        raise ValueError("Empty workbook.")
    raw = pd.concat(frames, ignore_index=True, sort=False)
    log.info("Combined raw shape: %s", raw.shape)
    return raw


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    bad = int(df["InvoiceDate"].isna().sum())
    if bad:
        log.warning("Dropped %d rows with unparseable InvoiceDate", bad)
        df = df.dropna(subset=["InvoiceDate"])
    return df


def inspect_dataset(df: pd.DataFrame) -> Dict[str, object]:
    rep = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "missing_pct": (df.isna().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "date_min": str(df["InvoiceDate"].min()) if "InvoiceDate" in df else None,
        "date_max": str(df["InvoiceDate"].max()) if "InvoiceDate" in df else None,
        "n_unique_customers": int(df["Customer ID"].nunique(dropna=True))
                              if "Customer ID" in df else None,
        "n_unique_products":  int(df["StockCode"].nunique())
                              if "StockCode" in df else None,
        "n_unique_invoices":  int(df["Invoice"].nunique())
                              if "Invoice" in df else None,
    }
    print("\\n" + "=" * 72)
    print("DATASET UNDERSTANDING")
    print("=" * 72)
    for k, v in rep.items():
        print(f"{k:22s} : {v}")
    print("=" * 72)
    return rep
'''

# ─────────────────────────────────────────────────────────────────────
# src/preprocessing.py
# ─────────────────────────────────────────────────────────────────────
FILES["src/preprocessing.py"] = '''"""preprocessing.py — hardened cleaning into 3 documented streams."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict
import numpy as np
import pandas as pd
from src.config import (ADMIN_STOCKCODES, CANCELLATION_PREFIX,
                        MIN_VALID_PRICE, MIN_VALID_QTY)

log = logging.getLogger(__name__)


@dataclass
class SplitResult:
    valid_sales: pd.DataFrame = field(default_factory=pd.DataFrame)
    returns:     pd.DataFrame = field(default_factory=pd.DataFrame)
    invalid:     pd.DataFrame = field(default_factory=pd.DataFrame)
    summary:     Dict[str, int] = field(default_factory=dict)


def clean_and_split(df: pd.DataFrame) -> SplitResult:
    df = df.copy()
    initial = len(df)

    dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log.info("Dropped %d duplicates", dupes)

    for col in ("Quantity", "Price", "Customer ID"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Invoice"]     = df["Invoice"].astype(str).str.strip().str.upper()
    df["StockCode"]   = df["StockCode"].astype(str).str.strip().str.upper()
    df["Description"] = (df["Description"].astype("string")
                         .str.strip().str.replace(r"\\s+", " ", regex=True))
    df["Country"]     = df["Country"].astype("string").str.strip().str.title()

    df["_is_cancel"]   = df["Invoice"].str.startswith(CANCELLATION_PREFIX)
    df["_is_admin"]    = df["StockCode"].isin(ADMIN_STOCKCODES)

    hard_invalid = (
        df["Quantity"].isna() | df["Price"].isna()
        | df["InvoiceDate"].isna()
        | df["StockCode"].isin({"", "NAN", "NONE"})
    )
    invalid_parts = [df[hard_invalid].assign(_reason="missing essential field")]
    remaining = df[~hard_invalid].copy()

    admin_mask = remaining["_is_admin"]
    if admin_mask.any():
        invalid_parts.append(
            remaining[admin_mask].assign(_reason="admin / non-product code"))
    remaining = remaining[~admin_mask]

    ret_mask = (remaining["Quantity"] < 0) | remaining["_is_cancel"]
    returns  = remaining[ret_mask].copy()
    returns["_reason"] = np.where(returns["_is_cancel"],
                                  "cancelled invoice", "negative quantity")
    remaining = remaining[~ret_mask]

    zpm = remaining["Price"] <= 0
    if zpm.any():
        invalid_parts.append(
            remaining[zpm].assign(_reason="non-positive price"))
    remaining = remaining[~zpm]

    lqm = remaining["Quantity"] < MIN_VALID_QTY
    if lqm.any():
        invalid_parts.append(
            remaining[lqm].assign(_reason="quantity below threshold"))
    remaining = remaining[~lqm]

    valid_sales = remaining.assign(_reason="valid sale")
    invalid = (pd.concat(invalid_parts, ignore_index=True, sort=False)
               if invalid_parts else pd.DataFrame())

    drop_cols = ["_is_cancel", "_is_admin"]
    valid_sales = valid_sales.drop(columns=drop_cols, errors="ignore")
    returns     = returns.drop(columns=drop_cols, errors="ignore")
    invalid     = invalid.drop(columns=drop_cols, errors="ignore")

    summary = {
        "initial": initial, "duplicates": dupes,
        "valid_sales": len(valid_sales), "returns": len(returns),
        "invalid": len(invalid),
        "check_sum": len(valid_sales) + len(returns) + len(invalid),
    }
    _log_summary(summary)
    return SplitResult(valid_sales, returns, invalid, summary)


def _log_summary(s: Dict[str, int]) -> None:
    print("\\n" + "=" * 72); print("CLEANING SUMMARY"); print("=" * 72)
    for k, v in s.items():
        pct = 100 * v / s["initial"] if s["initial"] else 0
        print(f"{k:15s}: {v:>10,d}  ({pct:5.2f}%)")
    print("=" * 72 + "\\n")


def data_quality_report(raw: pd.DataFrame, split: SplitResult) -> pd.DataFrame:
    rows = [
        ("Raw rows", len(raw)),
        ("Duplicates removed", int(raw.duplicated().sum())),
        ("Missing Customer ID", int(raw["Customer ID"].isna().sum())),
        ("Missing Description", int(raw["Description"].isna().sum())),
        ("Valid sales rows", len(split.valid_sales)),
        ("Returns rows", len(split.returns)),
        ("Invalid rows", len(split.invalid)),
        ("Unique products (valid)", split.valid_sales["StockCode"].nunique()),
        ("Unique customers (valid)",
            split.valid_sales["Customer ID"].dropna().nunique()),
        ("Unique invoices (valid)", split.valid_sales["Invoice"].nunique()),
        ("Date range",
            f"{split.valid_sales['InvoiceDate'].min().date()}"
            f" -> {split.valid_sales['InvoiceDate'].max().date()}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])
'''

# ─────────────────────────────────────────────────────────────────────
# src/feature_engineering.py
# ─────────────────────────────────────────────────────────────────────
FILES["src/feature_engineering.py"] = '''"""feature_engineering.py — gap-filled, leakage-safe feature builders."""
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
'''

# ─────────────────────────────────────────────────────────────────────
# src/split.py
# ─────────────────────────────────────────────────────────────────────
FILES["src/split.py"] = '''"""split.py — leakage-free product-time chronological split."""
from __future__ import annotations
import logging
from dataclasses import dataclass
import pandas as pd
from src.config import TRAIN_FRAC, VAL_FRAC

log = logging.getLogger(__name__)


@dataclass
class Splits:
    train: pd.DataFrame
    val:   pd.DataFrame
    test:  pd.DataFrame


def product_time_split(feat: pd.DataFrame) -> Splits:
    tr, va, te = [], [], []
    for _, g in feat.groupby("StockCode", sort=False):
        g = g.sort_values("Date")
        n = len(g)
        n_tr = int(n * TRAIN_FRAC)
        n_va = int(n * VAL_FRAC)
        tr.append(g.iloc[:n_tr])
        va.append(g.iloc[n_tr:n_tr + n_va])
        te.append(g.iloc[n_tr + n_va:])
    train = pd.concat(tr, ignore_index=True)
    val   = pd.concat(va, ignore_index=True)
    test  = pd.concat(te, ignore_index=True)
    log.info("Split -> train=%d val=%d test=%d",
             len(train), len(val), len(test))
    return Splits(train, val, test)
'''

# ─────────────────────────────────────────────────────────────────────
# src/leakage_audit.py
# ─────────────────────────────────────────────────────────────────────
FILES["src/leakage_audit.py"] = '''"""leakage_audit.py — proves lag/rolling features are past-only."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.config import LAG_DAYS, ROLL_WINDOWS


def audit_leakage(feat: pd.DataFrame, sample: int = 500) -> bool:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(feat), size=min(sample, len(feat)), replace=False)
    sub = feat.iloc[idx]

    errors = []
    for _, row in sub.iterrows():
        code = row["StockCode"]; date = row["Date"]
        prod = (feat[feat["StockCode"] == code]
                .set_index("Date").sort_index())
        for lag in LAG_DAYS:
            col = f"lag_qty_{lag}"
            try:
                expected = prod.loc[date - pd.Timedelta(days=lag), "Quantity"]
            except KeyError:
                continue
            if not np.isclose(row[col], expected, atol=1e-6):
                errors.append((code, date, col, row[col], expected))
        for w in ROLL_WINDOWS:
            col = f"roll_mean_{w}"
            expected = (prod["Quantity"].shift(1)
                        .rolling(w, min_periods=1).mean().loc[date])
            if not np.isclose(row[col], expected, atol=1e-6):
                errors.append((code, date, col, row[col], expected))

    if errors:
        raise AssertionError(
            f"LEAKAGE DETECTED in {len(errors)} places. Example: {errors[0]}")
    print("Leakage audit passed - all lag/rolling features are past-only.")
    return True
'''

# ─────────────────────────────────────────────────────────────────────
# src/eda.py
# ─────────────────────────────────────────────────────────────────────
FILES["src/eda.py"] = '''"""eda.py — task-driven EDA that answers MODEL questions."""
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
    zero_frac = daily.groupby("StockCode")["Quantity"] \\
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
    if returns.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    mr = returns.assign(M=returns["InvoiceDate"].dt.to_period("M")) \\
                .groupby("M").size()
    ms = sales.assign(M=sales["InvoiceDate"].dt.to_period("M")) \\
              .groupby("M").size()
    mr.index = mr.index.astype(str); ms.index = ms.index.astype(str)
    axes[0].plot(ms.index, ms.values, color="#2563EB", label="Sales lines")
    axes[0].plot(mr.index, mr.values, color="#EF4444", label="Returns")
    axes[0].set_title("Sales vs returns volume")
    axes[0].tick_params(axis="x", rotation=45); axes[0].legend()
    ratio = (mr / ms).dropna()
    sns.lineplot(x=ratio.index, y=ratio.values,
                 marker="o", ax=axes[1], color="#F59E0B")
    axes[1].set_title("Return / sale ratio per month")
    axes[1].tick_params(axis="x", rotation=45)
    _save(fig, "06_returns_balance")
    INSIGHTS["mean_return_ratio_pct"] = float(ratio.mean() * 100)


def plot_top_products(sales, stats, top_n=15):
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
'''

# ─────────────────────────────────────────────────────────────────────
# run_eda.py
# ─────────────────────────────────────────────────────────────────────
FILES["run_eda.py"] = '''"""run_eda.py — end-to-end data preparation.

Usage:
    python run_eda.py
"""
from __future__ import annotations
import logging, sys
import pandas as pd
from src.config import PROCESSED_DIR
from src.data_loader import load_raw_excel, inspect_dataset, parse_dates
from src.preprocessing import clean_and_split, data_quality_report
from src.feature_engineering import (
    build_daily_product_grid, filter_forecastable_products,
    clip_outliers_per_product, add_leakage_safe_ts_features,
    build_anomaly_features, compute_product_quality)
from src.split import product_time_split
from src.leakage_audit import audit_leakage
from src.eda import run_all_eda

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("run_eda")


def _save(df, name):
    if df is None or len(df) == 0:
        log.warning("Skipping empty table: %s", name); return
    try:
        path = PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
    except Exception:
        path = PROCESSED_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
    log.info("Saved %s -> %s", name, path.name)


def main():
    raw = load_raw_excel()
    understanding = inspect_dataset(raw)
    pd.DataFrame(list(understanding.items()),
                 columns=["Metric", "Value"]) \\
        .to_csv(PROCESSED_DIR / "dataset_understanding.csv", index=False)
    raw = parse_dates(raw)

    split = clean_and_split(raw)
    dq = data_quality_report(raw, split)
    dq.to_csv(PROCESSED_DIR / "data_quality_report.csv", index=False)
    print(dq.to_string(index=False))

    grid = build_daily_product_grid(split.valid_sales)
    filtered, stats = filter_forecastable_products(grid)
    clipped = clip_outliers_per_product(filtered)
    forecast_feat = add_leakage_safe_ts_features(clipped)

    quality = compute_product_quality(stats)
    _save(quality, "product_quality_scores")

    anom_feat = build_anomaly_features(split.valid_sales, split.returns)
    splits    = product_time_split(forecast_feat)
    audit_leakage(forecast_feat, sample=300)

    run_all_eda(split.valid_sales, split.returns, grid, stats, anom_feat)

    _save(split.valid_sales, "valid_sales")
    _save(split.returns,     "returns")
    _save(split.invalid,     "invalid")
    _save(forecast_feat,     "daily_product_features")
    _save(anom_feat,         "anomaly_features")
    _save(splits.train,      "train")
    _save(splits.val,        "val")
    _save(splits.test,       "test")

    log.info("Pipeline complete.")
    log.info("Forecasting table : %s", forecast_feat.shape)
    log.info("Anomaly table     : %s", anom_feat.shape)
    log.info("Products retained : %s",
             forecast_feat["StockCode"].nunique())


if __name__ == "__main__":
    main()
'''

# ─────────────────────────────────────────────────────────────────────
# tests/test_leakage.py
# ─────────────────────────────────────────────────────────────────────
FILES["tests/test_leakage.py"] = '''"""tests/test_leakage.py — CI guards against leakage and shuffled split."""
import pandas as pd
from src.config import PROCESSED_DIR
from src.leakage_audit import audit_leakage


def test_no_leakage():
    feat = pd.read_parquet(PROCESSED_DIR / "daily_product_features.parquet")
    assert audit_leakage(feat, sample=200) is True


def test_split_is_chronological():
    tr = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    va = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    te = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    for code in tr["StockCode"].unique():
        t = tr[tr["StockCode"] == code]["Date"].max()
        v = va[va["StockCode"] == code]["Date"]
        s = te[te["StockCode"] == code]["Date"]
        if len(v): assert v.min() > t
        if len(s) and len(v): assert s.min() > v.max()
'''

# ─────────────────────────────────────────────────────────────────────
# .gitignore
# ─────────────────────────────────────────────────────────────────────
FILES[".gitignore"] = """data/raw/*.xlsx
data/processed/*.parquet
data/processed/*.csv
reports/figures/*.png
__pycache__/
*.pyc
.pytest_cache/
.venv/
"""

# ─────────────────────────────────────────────────────────────────────
# data/raw/.gitkeep   (so the folder exists before you drop the xlsx)
# ─────────────────────────────────────────────────────────────────────
FILES["data/raw/.gitkeep"] = ""


def main() -> None:
    for rel, content in FILES.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    # placeholder folders
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    print(f"Created {len(FILES)} files under {ROOT.resolve()}")


if __name__ == "__main__":
    main()