"""
scripts/run_hybrid.py

Usage:
    python -m scripts.run_hybrid
"""
from __future__ import annotations
import logging
import sys

from src.hybrid_analysis import run_hybrid_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])


def main():
    run_hybrid_pipeline()


if __name__ == "__main__":
    main()