"""
scripts/train_forecasting.py

Usage:
    python -m scripts.train_forecasting
    python -m scripts.train_forecasting --forecast 22362 --horizon 14

The first form trains + evaluates + saves both models.
The second form additionally produces a future forecast for one product.
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.forecasting import forecast_future, run_training_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast", type=str, default=None,
                        help="StockCode to produce a future forecast for")
    parser.add_argument("--horizon", type=int, default=14,
                        help="Forecast horizon in days (1-30)")
    parser.add_argument("--model", type=str, default=None,
                        help="LinearRegression or RandomForest "
                             "(default: best)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use existing saved models)")
    args = parser.parse_args()

    if not args.skip_train:
        run_training_pipeline()

    if args.forecast:
        fc = forecast_future(args.forecast, horizon_days=args.horizon,
                             model_name=args.model)
        print("\n" + "=" * 72)
        print(f"FORECAST — {args.forecast} "
              f"({fc['Description'].iloc[0]})")
        print("=" * 72)
        print(fc[["Date", "predicted_quantity", "model_used"]]
              .to_string(index=False))
        print("=" * 72)


if __name__ == "__main__":
    main()