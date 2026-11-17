"""
scripts/run_streaming.py

Usage:
    python -m scripts.run_streaming                 # 500 per batch, all data
    python -m scripts.run_streaming --rows 20000    # only first 20k transactions
    python -m scripts.run_streaming --batch 250     # smaller batches

The CLI runs the full simulation synchronously, prints progress,
saves all alerts to data/processed/stream_alerts.parquet, and
generates the alert-rate chart.
"""
from __future__ import annotations
import argparse
import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import PROCESSED_DIR, BASE_DIR
from src.streaming import StreamSimulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])

REPORTS_DIR = BASE_DIR / "reports" / "streaming"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--rows", type=int, default=None,
                        help="cap total transactions (default: all)")
    args = parser.parse_args()

    sim = StreamSimulator(n_per_batch=args.batch)
    if args.rows is not None:
        sim.df = sim.df.iloc[:args.rows].reset_index(drop=True)

    print(f"\nStream simulator ready: {sim.total_rows:,} transactions "
          f"in {args.batch}-row batches "
          f"({-(-sim.total_rows // args.batch):,} batches expected)\n")

    sim.start()
    batch_num = 0
    while sim.status == "running":
        sim.step()
        batch_num += 1
        if batch_num % 20 == 0 or sim.status == "completed":
            print(f"  batch {batch_num:>4}  "
                  f"processed={sim.n_processed:>7,}  "
                  f"anomalies={sim.n_anomalous:>5,}  "
                  f"({sim.progress_pct:>5.1f}%)")

    summary = sim.summary()
    print("\n" + "=" * 72)
    print("STREAM SUMMARY")
    print("=" * 72)
    for k, v in summary.items():
        print(f"  {k:22s}: {v}")
    print("=" * 72)

    # Persist alerts + chart
    alerts_df = sim.alerts_df()
    alerts_df.to_parquet(PROCESSED_DIR / "stream_alerts.parquet", index=False)
    print(f"\nSaved {len(alerts_df)} alerts -> "
          f"stream_alerts.parquet")

    if not alerts_df.empty:
        alerts_df.head(100).to_csv(
            REPORTS_DIR / "alerts_sample.csv", index=False)

        rate_df = sim.alert_rate_df()
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(rate_df["batch"], rate_df["alert_rate"] * 100,
                color="#EF4444", lw=0.9)
        ax.set_xlabel("Batch number")
        ax.set_ylabel("Alert rate (%)")
        ax.set_title("Streaming alert rate over time")
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "01_alert_rate_over_time.png",
                    dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved alert-rate chart -> 01_alert_rate_over_time.png")


if __name__ == "__main__":
    main()