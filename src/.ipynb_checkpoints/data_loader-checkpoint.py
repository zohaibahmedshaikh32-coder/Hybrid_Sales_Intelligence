"""data_loader.py — robust Excel loader with integrity checks."""
from __future__ import annotations
import logging
from typing import Dict, List
import pandas as pd
from src.config import RAW_FILE, COLUMN_ALIASES
from pathlib import Path

log = logging.getLogger(__name__)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: COLUMN_ALIASES[c] for c in df.columns if c in COLUMN_ALIASES}
    unknown = [c for c in df.columns if c not in COLUMN_ALIASES]
    if unknown:
        log.warning("Unknown columns ignored by schema: %s", unknown)
    return df.rename(columns=rename)


def load_raw_excel(path: str | None = None) -> pd.DataFrame:
    file_path = Path(path) if path else RAW_FILE
    # normalize the path so `file_path.exists()` works on all OSes
    file_path = file_path.expanduser().resolve()

    # ---- friendly pre-checks ----------------------------------------
    if not file_path.exists():
        # Is there a similarly-named file the user probably meant?
        nearby = list(file_path.parent.glob("*.xlsx")) + \
                 list(file_path.parent.glob("*.xls"))
        hint = ""
        if nearby:
            hint = ("\nFound these Excel files in that folder:\n  "
                    + "\n  ".join(p.name for p in nearby)
                    + f"\n→ rename/move one to '{file_path.name}' "
                      f"or update RAW_FILE in src/config.py.")
        raise FileNotFoundError(
            f"\n\n❌ Dataset not found.\n"
            f"   Looked for : {file_path}\n"
            f"   Folder     : {file_path.parent}\n"
            f"   Folder exists? {file_path.parent.exists()}\n"
            f"{hint}\n\n"
            f"→ Download from: "
            f"https://archive.ics.uci.edu/dataset/502/online+retail+ii\n"
            f"→ Then place it at: {file_path}\n"
        )

    # ---- verify it's actually a readable Excel workbook -------------
    try:
        xls = pd.ExcelFile(file_path)
    except Exception as exc:
        raise ValueError(
            f"\n❌ Could not open '{file_path.name}' as an Excel workbook.\n"
            f"   Original error: {exc}\n"
            f"   Tip: the file may be corrupted, encrypted, or still being "
            f"downloaded. Re-download the UCI zip and try again.\n"
        ) from exc

    log.info("File      : %s", file_path)
    log.info("Size      : %.1f MB", file_path.stat().st_size / 1e6)
    log.info("Sheets    : %s", xls.sheet_names)

    frames: List[pd.DataFrame] = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        log.info("Sheet %-18s -> %6d x %2d cols", sheet, *df.shape)
        df = _normalise_columns(df)
        df["_source_sheet"] = sheet
        frames.append(df)

    if not frames:
        raise ValueError(
            f"Workbook '{file_path.name}' contains no sheets with data.")

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
    print("\n" + "=" * 72)
    print("DATASET UNDERSTANDING")
    print("=" * 72)
    for k, v in rep.items():
        print(f"{k:22s} : {v}")
    print("=" * 72)
    return rep
