"""split.py — leakage-free product-time chronological split."""
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
