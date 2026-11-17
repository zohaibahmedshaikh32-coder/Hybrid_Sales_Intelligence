"""
scripts/train_anomaly.py

Usage:
    python -m scripts.train_anomaly
"""
from __future__ import annotations
import logging
import sys

from src.anomaly_detection import run_training_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])


def main():
    run_training_pipeline()


if __name__ == "__main__":
    main()