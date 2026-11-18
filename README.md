# Retail Intelligence — Hybrid Predictive System

**🔗 Live demo:** https://hybridsalesintelligence-qqzlreznhjksvhygq2uix4.streamlit.app/

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hybridsalesintelligence-qqzlreznhjksvhygq2uix4.streamlit.app/)

End-to-end ML application for **sales forecasting** and **anomaly
detection** on the UCI Online Retail II dataset, delivered as a 9-page
Streamlit dashboard. Built for BS AI - Lab 14 Complex Computing Activity.

## Features

- **Data pipeline**: 1,067,371 raw transactions -> 3 documented streams
  (valid sales / returns / invalid), gap-filled daily grid,
  leakage-audited time-series features.
- **Forecasting**: Linear Regression + Random Forest Regressor,
  chronological product-time splits, MAE/RMSE/R2, recursive forecasts.
- **Anomaly detection**: K-Means (Lab 10), Hierarchical Ward (Lab 11),
  Isolation Forest, 3-way agreement analysis.
- **Hybrid analysis**: Deviation = Actual - Predicted, cross-referenced
  with anomaly flags into 6 business categories.
- **Streaming simulation**: chronological batch replay through the
  trained Isolation Forest with Start / Pause / Reset.
- **Business intelligence**: 56 mitigation hypotheses with severity,
  suggested action, and explicit caveat per entry.
- **9-page dashboard**: interactive Plotly charts, live filtering, CSV
  and PDF exports.

## Project Structure
Retail_Intelligence_System/
├── app.py # Streamlit entry point
├── run_eda.py # Data pipeline entry point
├── requirements.txt
├── pytest.ini
├── conftest.py
├── README.md
├── DEMO_WORKFLOW.md
├── .streamlit/config.toml
├── data/raw/online_retail_II.xlsx
├── data/processed/ # 15+ parquet/JSON artefacts
├── dashboard/ # 10 UI modules
├── src/ # 13 ML modules
├── scripts/ # 5 CLIs
├── models/ # 5 joblib + metadata
├── reports/ # figures + this report
└── tests/ # 5 test modules

text

## Installation

Requires **Python 3.11+** (tested 3.13).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
Download dataset: https://archive.ics.uci.edu/dataset/502/online+retail+ii
Save as: data/raw/online_retail_II.xlsx

Running the pipeline
bash
python run_eda.py                          # data + EDA
python -m scripts.train_forecasting        # LR + RF
python -m scripts.train_anomaly            # KMeans + HC + IF
python -m scripts.run_hybrid               # deviation analysis
python -m scripts.run_insights             # mitigations
python -m scripts.run_streaming --rows 20000  # optional
pytest tests/ -v                           # 21 tests
Launch dashboard
bash
streamlit run app.py
Opens http://localhost:8501

Dashboard pages
Executive Overview - 10 KPIs + revenue trends

Dataset Explorer - filters, quality report, CSV

Sales Analytics - 6 charts

Demand Forecasting - live forecast via saved models

Anomaly Detection - metrics, distribution, flagged table

Live Monitoring - Start/Pause/Reset streaming sim

Business Intelligence - 56 mitigations with caveats

Model Evaluation - metrics + limitations

Reports & Downloads - artefacts + PDF

Testing
bash
pytest tests/ -v
21 tests pass. Covers: leakage audit, chronological split, feature
completeness, model persistence, forecast recursion, agreement matrix
symmetry, business rule classification, streaming state machine,
mitigation structure.

**Forecasting (test set):**
- LinearRegression — MAE 9.57, RMSE 29.37, R² 0.237
- RandomForest — MAE 9.29, RMSE 28.68, R² 0.272 (best)

Troubleshooting
Symptom	Fix
FileNotFoundError: online_retail_II.xlsx	Place at data/raw/
ModuleNotFoundError: src	Run from project root
IndentationError after paste	Use Ctrl+Shift+V in VS Code
Dashboard no data	Run full pipeline first
use_container_width warning	Harmless Streamlit 1.36 notice
.streamlit folder can't create	New-Item -ItemType Directory -Path .streamlit -Force
U+FEFF syntax error	PowerShell Set-Content adds BOM - use VS Code
Lab concepts (5 demonstrated)
Lab	Concept	File
1	Pre-processing + feature engineering	src/preprocessing.py
2	Linear Regression	src/forecasting.py
5	Random Forest Regressor	src/forecasting.py
10	K-Means clustering	src/anomaly_detection.py
11	Hierarchical clustering	src/anomaly_detection.py
Key results
1,025,079 valid sales rows (96.04%)

21,719 returns (2.03%)

8,440 invalid (0.79%)

2,004 forecastable products (42%)

56 business mitigations

21 tests passing

Anomaly detection:

K-Means k=3: silhouette 0.678, DB 0.650

Hierarchical k=4: silhouette 0.353, DB 1.010

Isolation Forest: 2% contamination

Inter-model agreement: 0.21-0.29 (Jaccard)

Consensus (all 3 flagged): 6,316 transactions

See reports/project_report.md and DEMO_WORKFLOW.md for full details.

References
Chen, D. (2012). Online Retail II. UCI Machine Learning Repository.

Breiman, L. (2001). Random Forests. Machine Learning 45(1), 5-32.

Liu et al. (2008). Isolation Forest. ICDM.

Pedregosa et al. (2011). Scikit-learn. JMLR 12, 2825-2830.

---

## Deployment

Live app: **https://hybridsalesintelligence-qqzlreznhjksvhygq2uix4.streamlit.app/**

Source: [GitHub — Hybrid_Sales_Intelligence](https://github.com/zohaibahmedshaikh32-coder/Hybrid_Sales_Intelligence)
