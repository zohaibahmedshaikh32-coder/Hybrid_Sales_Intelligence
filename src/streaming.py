"""
streaming.py

Simulated streaming anomaly monitor.

HONESTY NOTE
------------
This is NOT a live connection to a retailer. It reads historical
transactions from `valid_sales.parquet` in chronological order and
replays them through the trained Isolation Forest in configurable
batches. The class is designed so the Streamlit dashboard can drive
it with Start / Pause / Reset buttons across reruns.

Design
------
• `StreamSimulator` holds *all* state internally — cursor position,
  counters, alert history. The dashboard can persist this object
  across Streamlit reruns using `st.session_state`.
• The scaler is cached once (fit on the full anomaly feature
  distribution) and re-used for every batch, so predictions are
  consistent with the trained model.
• Alerts are structured dicts (not just flags) so the dashboard can
  render them as cards with the fields the spec requires
  (Invoice, Description, StockCode, Quantity, Price, Revenue,
  Date, Score, Reason).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR, BASE_DIR

log = logging.getLogger(__name__)

ALERT_HISTORY_LIMIT = 5_000


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------
@dataclass
class Alert:
    """A single anomaly alert emitted by the stream."""
    timestamp:     str
    invoice:       str
    stock_code:    str
    description:   str
    quantity:      float
    price:         float
    revenue:       float
    anomaly_score: float
    reason:        str

    def to_dict(self) -> dict:
        return {
            "timestamp":     self.timestamp,
            "Invoice":       self.invoice,
            "StockCode":     self.stock_code,
            "Description":   self.description,
            "Quantity":      self.quantity,
            "Price":         self.price,
            "Revenue":       self.revenue,
            "anomaly_score": self.anomaly_score,
            "reason":        self.reason,
        }


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------
class StreamSimulator:
    """
    Replays historical transactions through a trained Isolation Forest.

    Parameters
    ----------
    n_per_batch  : how many rows per "arrival" tick (default 500)
    max_alerts   : cap on stored alert history (default 5000)
    random_state : reproducibility seed
    """

    def __init__(
        self,
        n_per_batch: int = 500,
        max_alerts: int = ALERT_HISTORY_LIMIT,
        random_state: int = 42,
    ):
        self.n_per_batch = int(n_per_batch)
        self.max_alerts  = int(max_alerts)

        # ---- Load the model bundle ----------------------------------
        meta_path = BASE_DIR / "models" / "anomaly_detection" / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Anomaly model metadata not found: {meta_path}\n"
                f"Run `python -m scripts.train_anomaly` first.")
        self.metadata   = json.loads(meta_path.read_text(encoding="utf-8"))
        self.feat_cols  = self.metadata["feature_cols"]
        self.iso_forest = joblib.load(
            BASE_DIR / "models" / "anomaly_detection" / "isolation_forest.joblib")

        # ---- Load the transaction stream -----------------------------
        raw = pd.read_parquet(
            PROCESSED_DIR / "anomaly_features.parquet"
        ).sort_values(["InvoiceDate", "Invoice"]).reset_index(drop=True)

        # Only sales lines (not returns) — returns are tracked separately
        self.df = raw[raw["IsReturn"] == 0].reset_index(drop=True)

        # ---- Fit scaler ONCE on the full anomaly distribution --------
        # (Matches the approach in hybrid_analysis.aggregate_anomalies_to_daily:
        #  the Isolation Forest was trained on standardized features, so we
        #  standardize using the same global feature distribution.)
        self.scaler = StandardScaler().fit(
            self.df[self.feat_cols].astype(float).values)

        # ---- State --------------------------------------------------
        self.cursor               = 0
        self.status               = "idle"          # idle | running | paused | completed
        self.n_processed          = 0
        self.n_normal             = 0
        self.n_anomalous          = 0
        self.alerts: list[Alert]  = []
        self.alert_rate_history: list[dict] = []    # for plotting
        self._rng                 = np.random.default_rng(random_state)

    # -----------------------------------------------------------------
    # Public API used by the dashboard and CLI
    # -----------------------------------------------------------------
    @property
    def total_rows(self) -> int:
        return len(self.df)

    @property
    def progress_pct(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return 100.0 * self.cursor / self.total_rows

    def start(self) -> None:
        if self.cursor >= self.total_rows:
            self.reset()
        self.status = "running"
        log.info("Stream started from row %d / %d",
                 self.cursor, self.total_rows)

    def pause(self) -> None:
        if self.status == "running":
            self.status = "paused"
            log.info("Stream paused at row %d / %d",
                     self.cursor, self.total_rows)

    def reset(self) -> None:
        self.cursor               = 0
        self.status               = "idle"
        self.n_processed          = 0
        self.n_normal             = 0
        self.n_anomalous          = 0
        self.alerts               = []
        self.alert_rate_history   = []
        log.info("Stream reset")

    def step(self) -> pd.DataFrame:
        """
        Advance one batch. Returns the batch DataFrame that was processed.
        Does nothing if status != 'running' or the stream is exhausted.
        """
        if self.status != "running":
            return pd.DataFrame()
        if self.cursor >= self.total_rows:
            self.status = "completed"
            log.info("Stream completed: %d rows processed", self.n_processed)
            return pd.DataFrame()

        end = min(self.cursor + self.n_per_batch, self.total_rows)
        batch = self.df.iloc[self.cursor:end].copy()
        self._process_batch(batch)
        self.cursor = end

        if self.cursor >= self.total_rows:
            self.status = "completed"
            log.info("Stream completed: %d rows processed", self.n_processed)

        return batch

    def run_until_complete(self) -> None:
        """Convenience for CLI: run every remaining batch, synchronously."""
        self.start()
        while self.status == "running":
            self.step()

    # -----------------------------------------------------------------
    # Batch processing
    # -----------------------------------------------------------------
    def _process_batch(self, batch: pd.DataFrame) -> None:
        X_raw    = batch[self.feat_cols].astype(float).values
        X_scaled = self.scaler.transform(X_raw)

        scores    = -self.iso_forest.score_samples(X_scaled)   # higher=more anomalous
        preds     = self.iso_forest.predict(X_scaled)           # -1 = anomaly
        flags     = (preds == -1).astype(int)

        self.n_processed += len(batch)
        self.n_normal    += int((flags == 0).sum())
        self.n_anomalous += int((flags == 1).sum())

        # Record per-batch alert rate for the dashboard time-series
        self.alert_rate_history.append({
            "batch":         len(self.alert_rate_history) + 1,
            "rows":          len(batch),
            "n_anomalies":   int(flags.sum()),
            "alert_rate":    float(flags.mean()),
        })

        # Materialise alerts
        for i, flag in enumerate(flags):
            if flag != 1:
                continue
            row = batch.iloc[i]
            alert = Alert(
                timestamp=pd.Timestamp(row["InvoiceDate"]).isoformat(),
                invoice=str(row.get("Invoice", "")),
                stock_code=str(row["StockCode"]),
                description=str(row.get("Description", "")),
                quantity=float(row["Quantity"]),
                price=float(row["Price"]),
                revenue=float(row.get("Revenue", row["Quantity"] * row["Price"])),
                anomaly_score=float(scores[i]),
                reason=self._infer_reason(row),
            )
            self.alerts.append(alert)

        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]

    # -----------------------------------------------------------------
    # Interpretable alert reason (same rules as anomaly_detection)
    # -----------------------------------------------------------------
    @staticmethod
    def _infer_reason(row: pd.Series) -> str:
        if row.get("qty_zscore", 0) > 3:
            return "Unusually high quantity"
        if row.get("qty_zscore", 0) < -3:
            return "Unusually low quantity"
        if row.get("price_zscore", 0) > 3:
            return "Unusually high unit price"
        if row.get("price_zscore", 0) < -3:
            return "Unusually low unit price"
        if row.get("AbsRevenue", 0) > 1_000:
            return "Unusual revenue"
        return "Flagged by Isolation Forest"

    # -----------------------------------------------------------------
    # Serialisation for the dashboard / tests
    # -----------------------------------------------------------------
    def alerts_df(self) -> pd.DataFrame:
        if not self.alerts:
            return pd.DataFrame(columns=[
                "timestamp", "Invoice", "StockCode", "Description",
                "Quantity", "Price", "Revenue", "anomaly_score", "reason"])
        return pd.DataFrame([a.to_dict() for a in self.alerts])

    def alert_rate_df(self) -> pd.DataFrame:
        if not self.alert_rate_history:
            return pd.DataFrame(columns=["batch", "rows",
                                         "n_anomalies", "alert_rate"])
        return pd.DataFrame(self.alert_rate_history)

    def summary(self) -> dict:
        return {
            "status":              self.status,
            "progress_pct":        round(self.progress_pct, 2),
            "n_processed":         self.n_processed,
            "n_normal":            self.n_normal,
            "n_anomalous":         self.n_anomalous,
            "anomaly_rate_pct":    round(
                100 * self.n_anomalous / max(self.n_processed, 1), 3),
            "total_rows":          self.total_rows,
            "n_alerts_stored":     len(self.alerts),
            "n_batches_completed": len(self.alert_rate_history),
        }