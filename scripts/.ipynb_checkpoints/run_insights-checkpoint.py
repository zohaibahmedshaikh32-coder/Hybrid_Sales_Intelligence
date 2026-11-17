"""
scripts/run_insights.py

Usage:
    python -m scripts.run_insights
"""
from __future__ import annotations
import logging
import sys

from src.business_insights import run_business_insights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])


def main():
    run_business_insights()


if __name__ == "__main__":
    main()