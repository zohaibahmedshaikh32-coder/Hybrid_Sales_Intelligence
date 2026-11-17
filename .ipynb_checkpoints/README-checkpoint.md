Retail_Intelligence_System/
├── app.py # Streamlit entry point
├── README.md
├── requirements.txt
├── pytest.ini
├── conftest.py
├── run_eda.py # Data pipeline entry point
├── .streamlit/config.toml
│
├── data/
│ ├── raw/online_retail_II.xlsx # ← drop dataset here
│ └── processed/ # 15+ parquet/JSON artefacts
│
├── dashboard/
│ ├── ui.py, overview.py, data_explorer.py,
│ ├── sales_analytics.py, forecasting_page.py,
│ ├── anomaly_page.py, monitoring_page.py,
│ ├── business_insights_page.py, evaluation_page.py,
│ └── reports_page.py
│
├── src/
│ ├── config.py, data_loader.py, preprocessing.py,
│ ├── feature_engineering.py, split.py, leakage_audit.py,
│ ├── eda.py, forecasting.py, anomaly_detection.py,
│ ├── hybrid_analysis.py, streaming.py, business_insights.py,
│ └── evaluation.py
│
├── scripts/
│ ├── train_forecasting.py, train_anomaly.py,
│ ├── run_hybrid.py, run_streaming.py, run_insights.py
│
├── models/
│ ├── forecasting/LinearRegression.joblib
│ ├── forecasting/RandomForest.joblib
│ ├── anomaly_detection/{kmeans,hierarchical,isolation_forest}.joblib
│ └── */metadata.json
│
├── reports/
│ ├── figures/, forecasting/, anomaly_detection/,
│ ├── hybrid_analysis/, streaming/, business_insights/
│ ├── project_report.md
│ └── project_report.pdf
│
├── DEMO_WORKFLOW.md
└── tests/
├── test_leakage.py, test_anomaly.py, test_hybrid.py,
└── test_streaming.py, test_business_insights.py

text

---

## Installation

### Requirements
- **Python 3.11+** (tested on 3.13)
- Windows 10/11, macOS, or Linux
- ~2 GB free disk space

### Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# https://archive.ics.uci.edu/dataset/502/online+retail+ii
# Save as: data/raw/online_retail_II.xlsx
Running the Pipeline
Run these commands in order (each takes 2–10 min):

bash
# 1. Data cleaning + EDA + feature engineering
python run_eda.py

# 2. Train forecasting models (LR + RF)
python -m scripts.train_forecasting

# 3. Train anomaly detection models (KMeans + HC + IF)
python -m scripts.train_anomaly

# 4. Hybrid analysis (deviations × anomalies)
python -m scripts.run_hybrid

# 5. Business mitigations
python -m scripts.run_insights

# 6. Optional: streaming simulation
python -m scripts.run_streaming --rows 20000

# 7. Run all tests (should show 21 passed)
pytest tests/ -v
Launch the Dashboard

Launch the Dashboard
bash
streamlit run app.py
Browser opens at http://localhost:8501.