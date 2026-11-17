
---

## 📄 File 2: `reports/project_report.md`

**Location:** `C:\Users\zohai\Retail_Intelligence_System1\reports\project_report.md`

**Create/replace** with this **entire content**:

```markdown
# Retail Intelligence — Hybrid Predictive System

## Sales Forecasting and Anomaly Detection on the UCI Online Retail II Dataset

**Course:** BS Artificial Intelligence — Lab 14 Complex Computing Activity

**Author:** [Your Name]

**Date:** 2024

---

## 1. Introduction

Retail operations generate large volumes of transaction data in which demand forecasting and anomaly detection are both critical business functions. Traditional analytics treat these as separate problems: forecasting models predict demand, and anomaly detection flags unusual transactions. In practice, the two signals are intimately connected — a spike in demand is often accompanied by unusual transaction patterns, and an unusual pattern often signals a demand deviation worth investigating.

This project builds a **hybrid predictive system** that combines supervised demand forecasting with unsupervised anomaly detection into a single, coherent, interpretable application. The system processes 1,067,371 raw transactions from the UCI Online Retail II dataset, trains five machine learning models across three paradigms (linear regression, tree ensembles, and three unsupervised clustering/isolation methods), and delivers its output through a 9-page interactive Streamlit dashboard.

The project demonstrates five distinct lab concepts (pre-processing, linear regression, random forest, K-means, and hierarchical clustering) exceeding the Lab-14 requirement of at least three.

---

## 2. Problem Statement

UK-based online retailers face three operational challenges:

1. **Demand uncertainty** — insufficient stock during high-demand periods loses revenue; overstock ties up capital and requires markdowns.
2. **Transaction anomalies** — bulk orders, data-entry errors, return spikes, and pricing inconsistencies go unnoticed in raw transaction streams.
3. **Interpretability gap** — machine learning models produce predictions but rarely provide the actionable business guidance that operations teams need.

**Solution:** a hybrid system that produces forecasts, anomaly flags, categorised business mitigations, and exposes everything through a single, intuitive dashboard. The system treats all anomaly signals as hypotheses requiring human review — never as automated decisions.

---

## 3. Objectives

1. Build a clean, documented, leakage-free data pipeline from raw Excel to model-ready features.
2. Train and compare supervised forecasting models (Linear Regression, Random Forest).
3. Train and compare unsupervised anomaly detection models (K-Means, Hierarchical, Isolation Forest).
4. Combine both signals into a hybrid deviation analysis.
5. Simulate streaming anomaly monitoring.
6. Generate interpretable business mitigation recommendations.
7. Deliver a 9-page interactive dashboard.
8. Demonstrate at least three distinct Lab concepts (achieved: five).

---

## 4. Dataset

**UCI Online Retail II** — 1,067,371 transactions from a UK-based online retailer, covering 2009-12-01 to 2011-12-09.

| Property | Value |
|---|---|
| Raw rows | 1,067,371 |
| Columns | 8 (Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country) |
| Sheets | 2 ("Year 2009-2010", "Year 2010-2011") |
| Unique customers | 5,942 |
| Unique products | 5,305 |
| Unique invoices | 53,628 |
| Missing Customer ID | 243,007 (22.77%) |
| Missing Description | 4,382 (0.41%) |
| Duplicates | 12,133 (1.14%) |

**Source:** https://archive.ics.uci.edu/dataset/502/online+retail+ii

The dataset exhibits characteristic retail properties: heavy-tailed quantity distributions, many zero-sale product-days, a strong Q4 seasonal spike, near-zero weekend activity (B2B retailer), and multiple descriptions per StockCode due to product-name edits over the 2-year window.

---

## 5. Data Pre-processing and Feature Engineering

### 5.1 Cleaning Strategy

Three documented output streams. **Nothing is blindly dropped** — every exclusion is logged with a reason:

| Stream | Rows | % of raw | Description |
|---|---|---|---|
| **Valid sales** | 1,025,079 | 96.04% | Positive quantity, positive price, no cancellation prefix |
| **Returns** | 21,719 | 2.03% | Negative quantity OR invoice starting with `C` |
| **Invalid** | 8,440 | 0.79% | Admin codes (POST, M, DOT…), zero-price, unusable rows |

Duplicates removed: **12,133 (1.14%)**.

**Decision rationale:**
- Returns are a real business event (not noise) — they're preserved as a separate stream so anomaly detection can flag return patterns.
- Admin stock codes (POST, M, DOT, BANK CHARGES) are non-product lines — they're excluded from product analysis but preserved in the invalid stream for audit.
- Cancellations (`C`-prefixed invoices) are not simply "negative sales" — they represent a distinct workflow and are treated accordingly.

### 5.2 Feature Engineering

**Calendar features:** Year, Month, Day, DayOfWeek, WeekOfYear, Quarter, IsWeekend, IsMonthStart, IsMonthEnd.

**Sales features:** Revenue (Quantity × Price), AvgPrice, NumTransactions, UniqueCustomers.

**Time-series features (past-only):** lags 1, 7, 14, 30; rolling mean/std/max over 7, 14, 30, 90; cumulative demand; demand trend (7-day minus 30-day average).

**Gap-filling:** Every product's daily series is reindexed on the full calendar. A missing day is a real zero, not a dropped row. Without gap-filling, lag features would skip across missing days and produce meaningless values.

**Outlier clipping:** Per-product 0.5%–99.5% quantiles on Quantity and Revenue. This stabilises variance for both regression and clustering without removing genuine high-value orders (which are preserved in the anomaly stream via absolute z-scores).

### 5.3 Leakage Audit

An automated audit recomputes every lag and rolling feature from the raw series and asserts `feature(t) = function(data ≤ t−1)`. The audit passed on all 1,800 sampled feature values across 300 randomly chosen rows. The audit runs as part of the test suite (`test_no_leakage`), so any future change that introduces leakage fails CI.

**Critical detail:** the audit is run on the **pre-clip** grid, not the training table. Clipping is a static per-product transform (time-independent, order-independent) and therefore cannot introduce future information. Auditing after clipping mixes two independent concerns and produces false positives from floating-point drift.

### 5.4 Product Cohort Filter

Only products meeting **all three** criteria were retained for forecasting:

- ≥ 60 active days (distinct days with a sale)
- ≥ 100 total units sold
- ≥ 1 unit/day average

**Result:** 2,004 of 4,735 products retained (42%). This single filter lifts test R² by 15–30% on the retained cohort by removing products whose demand is too sparse to forecast meaningfully.

### 5.5 Chronological Split

Per-product temporal split, no shuffling:

| Split | Rows | Fraction |
|---|---|---|
| Train | 993,984 | 70% |
| Validation | 212,424 | 15% |
| Test | 214,428 | 15% |

Model selection uses validation RMSE. The test set is touched once, for the winning model only.

---

## 6. System Architecture

```mermaid
graph LR
    A[Raw Excel<br/>1.07 M rows] --> B[Data Loader]
    B --> C[Pre-processing<br/>3 streams]
    C --> D[Feature Engineering<br/>gap-fill + lags + audit]
    D --> E[Daily Product Grid<br/>1.42 M rows]
    D --> F[Anomaly Features<br/>1.05 M rows]

    E --> G[Forecasting<br/>LR + RF]
    F --> H[Anomaly Detection<br/>KMeans + HC + IF]

    G --> I[Hybrid Analysis<br/>Actual − Predicted]
    H --> I

    I --> J[Business Mitigations<br/>56 hypotheses]
    F --> K[Streaming Simulation]
    H --> K

    G --> L[(Saved models)]
    H --> L

    L --> M[Streamlit Dashboard<br/>9 pages]
    I --> M
    J --> M
    K --> M