"""preprocessing.py — hardened cleaning into 3 documented streams."""
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
                         .str.strip().str.replace(r"\s+", " ", regex=True))
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
    print("\n" + "=" * 72); print("CLEANING SUMMARY"); print("=" * 72)
    for k, v in s.items():
        pct = 100 * v / s["initial"] if s["initial"] else 0
        print(f"{k:15s}: {v:>10,d}  ({pct:5.2f}%)")
    print("=" * 72 + "\n")


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
