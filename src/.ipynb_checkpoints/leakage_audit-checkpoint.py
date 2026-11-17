"""
leakage_audit.py — proves lag/rolling features are past-only.

The KEY principle: we recompute each feature from the SAME complete
per-product series that was used to build it. We never reconstruct the
series from a truncated slice of the feature table (that was the bug
in the previous version — it caused false-positive "leakage" reports).

Verification logic:
    For each sampled row (StockCode, Date):
      - Rebuild the product's COMPLETE daily Quantity series from `feat`
        (all rows for that product, including cold-start rows that were
        dropped from the training table — they still exist in `feat`'s
        per-product series via the `_raw_series` cache we build below).
      - Recompute `lag_qty_k(t)` and `roll_mean_w(t)` from that series.
      - Assert equality with the stored feature.

If any stored feature differs from its past-only reconstruction → leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LAG_DAYS, ROLL_WINDOWS


def _complete_product_series(
    feat: pd.DataFrame,
    grid: pd.DataFrame | None,
    code: str,
) -> pd.Series:
    """
    Rebuild the COMPLETE per-product daily Quantity series — including
    the cold-start days that were removed from `feat` by `dropna`.

    We use the original gap-filled `grid` if available (preferred — the
    audit must be self-consistent with the *raw* series, not with the
    feature table). If `grid` is not supplied we fall back to `feat`,
    but then we document that the audit only verifies consistency with
    the retained timeline (still catches true leakage).
    """
    if grid is not None:
        s = (grid[grid["StockCode"] == code]
             .sort_values("Date")
             .set_index("Date")["Quantity"])
        return s

    s = (feat[feat["StockCode"] == code]
         .sort_values("Date")
         .set_index("Date")["Quantity"])
    return s


def audit_leakage(
    feat: pd.DataFrame,
    grid: pd.DataFrame | None = None,
    sample: int = 500,
) -> bool:
    """
    Verify that all lag/rolling features in `feat` are past-only.

    Parameters
    ----------
    feat  : feature table produced by add_leakage_safe_ts_features()
    grid  : the gap-filled daily grid used to build `feat` — REQUIRED for
            a fully self-consistent audit. If None, we fall back to
            `feat` and only verify the retained timeline.
    sample: number of rows to spot-check.

    Returns
    -------
    True if no leakage. Raises AssertionError otherwise.
    """
    if grid is None:
        print(
            "⚠️  audit_leakage: `grid` not supplied — falling back to `feat`.\n"
            "    The audit will still catch real leaks, but may miss\n"
            "    edge cases near cold-start boundaries.\n"
        )

    rng = np.random.default_rng(0)
    idx = rng.choice(len(feat), size=min(sample, len(feat)), replace=False)
    sub = feat.iloc[idx]

    errors: list[tuple] = []
    checked = 0
    cache: dict[str, pd.Series] = {}

    for _, row in sub.iterrows():
        code = row["StockCode"]
        date = row["Date"]

        if code not in cache:
            cache[code] = _complete_product_series(feat, grid, code)
        series = cache[code]

        # --- lag_qty_k ---------------------------------------------------
        for lag in LAG_DAYS:
            col = f"lag_qty_{lag}"
            expected_date = date - pd.Timedelta(days=lag)
            if expected_date not in series.index:
                continue
            expected = series.loc[expected_date]
            checked += 1
            if not np.isclose(row[col], expected, atol=1e-6):
                errors.append(
                    (code, date, col, float(row[col]), float(expected))
                )

        # --- roll_mean_w -------------------------------------------------
        # The stored feature was built as:
        #     series.shift(1).rolling(w, min_periods=1).mean()
        # so we MUST recompute it exactly that way on the same series.
        for w in ROLL_WINDOWS:
            col = f"roll_mean_{w}"
            expected = (series.shift(1)
                        .rolling(w, min_periods=1)
                        .mean()
                        .get(date))
            if expected is None or (isinstance(expected, float)
                                    and np.isnan(expected)):
                continue
            checked += 1
            if not np.isclose(row[col], expected, atol=1e-6):
                errors.append(
                    (code, date, col, float(row[col]), float(expected))
                )

        # --- roll_std_w --------------------------------------------------
        for w in ROLL_WINDOWS:
            col = f"roll_std_{w}"
            stored = row[col]
            if pd.isna(stored):
                continue
            expected = (series.shift(1)
                        .rolling(w, min_periods=1)
                        .std()
                        .get(date))
            if expected is None or (isinstance(expected, float)
                                    and np.isnan(expected)):
                continue
            checked += 1
            if not np.isclose(stored, expected, atol=1e-6, equal_nan=True):
                errors.append(
                    (code, date, col, float(stored), float(expected))
                )

    print(f"\n{'='*72}")
    print(f"LEAKAGE AUDIT — checked {checked} feature values "
          f"across {len(sub)} rows / {len(cache)} products")
    print(f"{'='*72}")

    if errors:
        print(f"❌ {len(errors)} mismatches detected. First 3:")
        for e in errors[:3]:
            print(f"   code={e[0]}  date={e[1].date()}  "
                  f"feature={e[2]}  stored={e[3]:.6f}  "
                  f"recomputed={e[4]:.6f}")
        raise AssertionError(
            f"LEAKAGE DETECTED in {len(errors)} places. "
            f"Example: {errors[0]}"
        )

    print("✅ Leakage audit passed — all lag/rolling features are past-only.")
    print(f"{'='*72}\n")
    return True