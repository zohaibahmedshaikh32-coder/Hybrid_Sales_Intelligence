"""
config.py — ML-oriented configuration.
Every constant below is a *model-performance* decision, not aesthetic.
"""
from pathlib import Path

BASE_DIR       = Path(__file__).resolve().parent.parent
RAW_DIR        = BASE_DIR / "data" / "raw"
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
FIGURES_DIR    = BASE_DIR / "reports" / "figures"

for _p in (RAW_DIR, PROCESSED_DIR, FIGURES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

RAW_FILE = RAW_DIR / "online_retail_II.xlsx"

COLUMN_ALIASES = {
    "InvoiceNo": "Invoice", "Invoice": "Invoice",
    "StockCode": "StockCode",
    "Description": "Description",
    "Quantity": "Quantity",
    "InvoiceDate": "InvoiceDate",
    "UnitPrice": "Price", "Price": "Price",
    "CustomerID": "Customer ID", "Customer ID": "Customer ID",
    "Country": "Country",
}

MIN_VALID_QTY       = 1
MIN_VALID_PRICE     = 0.01
CANCELLATION_PREFIX = "C"
ADMIN_STOCKCODES    = {
    "POST", "D", "M", "C2", "DOT", "BANK CHARGES",
    "PADS", "CRUK", "AMAZONFEE", "S", "B",
}

MIN_ACTIVE_DAYS   = 60
MIN_TOTAL_QTY     = 100
MIN_AVG_DAILY_QTY = 1.0

OUTLIER_UPPER_QUANTILE = 0.995
OUTLIER_LOWER_QUANTILE = 0.005

LAG_DAYS     = (1, 7, 14, 30)
ROLL_WINDOWS = (7, 14, 30, 90)
MIN_HISTORY  = 60

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

RANDOM_STATE = 42
