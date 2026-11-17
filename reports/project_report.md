# Retail Intelligence - Hybrid Predictive System

Sales Forecasting and Anomaly Detection on the UCI Online Retail II Dataset
Course: BS Artificial Intelligence - Lab 14 Complex Computing Activity

Author: [Your Name] Date: 2024

1. Introduction
Retail operations generate large volumes of transaction data in which
demand forecasting and anomaly detection are both critical. Traditional
analytics treat these as separate problems. In practice the two signals
are intimately connected - a spike in demand is often accompanied by
unusual transaction patterns, and an unusual pattern often signals a
demand deviation worth investigating.

This project builds a hybrid predictive system that combines supervised
demand forecasting with unsupervised anomaly detection into a single,
coherent, interpretable application. It processes 1,067,371 raw
transactions from the UCI Online Retail II dataset, trains five machine
learning models across three paradigms, and delivers its output through
a 9-page interactive Streamlit dashboard.

The project demonstrates five distinct lab concepts (pre-processing,
linear regression, random forest, K-means, hierarchical clustering) -
exceeding the Lab-14 requirement of at least three.

2. Problem Statement
UK-based online retailers face three challenges:

Demand uncertainty - insufficient stock loses revenue; overstock
ties up capital.

Transaction anomalies - bulk orders, data-entry errors, return
spikes go unnoticed in raw streams.

Interpretability gap - ML models produce predictions but rarely
actionable business guidance.

Solution: a hybrid system producing forecasts, anomaly flags,
categorised business mitigations, exposed through a single dashboard.
All anomaly signals are hypotheses requiring human review, never
automated decisions.

3. Objectives
Build a clean, documented, leakage-free data pipeline.

Train and compare supervised forecasting models.

Train and compare unsupervised anomaly detection models.

Combine both signals into a hybrid deviation analysis.

Simulate streaming anomaly monitoring.

Generate interpretable business mitigation recommendations.

Deliver a 9-page interactive dashboard.

Demonstrate at least three distinct Lab concepts (achieved: five).

4. Dataset
UCI Online Retail II - 1,067,371 transactions, 2009-12-01 to 2011-12-09.

Property	Value
Raw rows	1,067,371
Columns	8 (Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country)
Unique customers	5,942
Unique products	5,305
Unique invoices	53,628
Missing Customer ID	243,007 (22.77%)
Missing Description	4,382 (0.41%)
Duplicates	12,133 (1.14%)
Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii

Key properties: heavy-tailed quantity, many zero-sale product-days,
strong Q4 seasonal spike, near-zero weekend activity (B2B retailer),
multiple descriptions per StockCode due to product-name edits.

5. Data Pre-processing and Feature Engineering
5.1 Cleaning Strategy
Three documented streams. Nothing blindly dropped.

Stream	Rows	% raw	Description
Valid sales	1,025,079	96.04%	Positive qty, positive price, no cancel
Returns	21,719	2.03%	Negative qty OR C-prefixed invoice
Invalid	8,440	0.79%	Admin codes, zero-price, unusable
Duplicates removed: 12,133 (1.14%).

Returns are real business events, preserved separately. Admin stock
codes (POST, M, DOT, BANK CHARGES) excluded from product analysis but
preserved for audit. Cancellations (C-prefixed) treated as distinct
workflow from negative sales.

5.2 Feature Engineering
Calendar: Year, Month, Day, DayOfWeek, WeekOfYear, Quarter, IsWeekend,
IsMonthStart, IsMonthEnd.

Sales: Revenue = Quantity x Price, AvgPrice, NumTransactions,
UniqueCustomers.

Time-series (past-only): lags 1/7/14/30, rolling mean/std/max over
7/14/30/90, cumulative demand, demand trend.

Gap-filling: every product reindexed on the full calendar. A missing
day is a real zero, not a dropped row. Without gap-filling, lag
features skip across missing days.

Outlier clipping: per-product 0.5%-99.5% quantiles on Quantity and
Revenue. Stabilises variance without removing high-value orders.

5.3 Leakage Audit
Automated audit recomputes every lag and rolling feature from raw series
and asserts feature(t) = function(data <= t-1). Passed 1,800 sampled
feature values. Runs in test suite (test_no_leakage). Audited on
pre-clip grid, not training table - clipping is time-independent.

5.4 Product Cohort Filter
= 60 active days, >= 100 total units, >= 1 unit/day average. Result:
2,004 of 4,735 products retained (42%). Lifts test R2 by 15-30% by
removing sparse products.

5.5 Chronological Split
Split	Rows	Fraction
Train	993,984	70%
Validation	212,424	15%
Test	214,428	15%
Model selection uses validation RMSE. Test set touched once.

6. System Architecture
text
Raw Excel (1.07M rows) -> Data Loader -> Pre-processing (3 streams)
    -> Feature Engineering (gap-fill + lags + audit)
    -> Daily Product Grid (1.42M rows)  -> Forecasting (LR + RF)
    -> Anomaly Features (1.05M rows)    -> Anomaly Detection (KMeans + HC + IF)
    -> Both feed into Hybrid Analysis (Actual - Predicted)
    -> Business Mitigations (56) + Streaming Simulation
    -> Streamlit Dashboard (9 pages)
7. Lab Concepts Integrated
Lab	Concept	Implementation	File
1	Pre-processing, scaling, feature engineering	Full pipeline + StandardScaler	src/preprocessing.py, src/feature_engineering.py
2	Linear Regression	Demand forecasting baseline	src/forecasting.py
5	Random Forest Regressor	Demand forecasting ensemble	src/forecasting.py
10	K-Means clustering	Sales anomaly detection	src/anomaly_detection.py
11	Hierarchical Agglomerative Clustering	Anomaly detection + dendrogram	src/anomaly_detection.py
Five distinct Lab concepts demonstrated - exceeds Lab-14 requirement
of at least three.

8. Forecasting Models
8.1 Linear Regression
32 engineered features. Pipeline: StandardScaler -> LinearRegression
(scaler inside pipeline so inference applies the same transform).
Purpose: interpretable baseline demonstrating Lab 2.

8.2 Random Forest Regressor
Hyperparameters: n_estimators=200, max_depth=18, min_samples_leaf=5,
max_features='sqrt', n_jobs=-1. max_depth=18 chosen because RF overfits
heavily at depth > 25. min_samples_leaf=5 prevents 1-sample leaves.
Feature importance analysis produced for viva defence (Lab 5 concept).

8.3 Evaluation (test set)
Fill in from reports/forecasting/metrics.csv before submission.

Model	MAE	RMSE	R2	MAPE
LinearRegression	(from CSV)	(from CSV)	(from CSV)	(from CSV)
RandomForest	(from CSV)	(from CSV)	(from CSV)	(from CSV)
Do not fabricate numbers. Open reports/forecasting/metrics.csv and
paste actual values. Your actual results are your viva defence.

MAPE caveat: many test rows have Actual = 0 (zero-sale days). Standard
MAPE undefined at zero; computed only over |Actual| > eps rows. High
value reflects true difficulty of predicting sparse demand.

8.4 Model Selection
Best model by lowest validation RMSE. Test set evaluated exactly once
for winner only. Standard no-peeking protocol.

8.5 Recursive Future Forecasting
Horizon N days: Day 1 uses true history, Day 2 uses Day 1 prediction as
lag_qty_1, ..., Day N uses predictions from N-1 days. Documented
limitation: forecast errors compound across horizon. Beyond 14 days,
reliability decreases.

9. Anomaly Detection Models
9.1 K-Means (Lab 10)
StandardScaler fit once. Elbow + silhouette sweep over k in
{2,3,4,5,6,7}. Safe silhouette with per-cluster stratified sampling -
critical because k=2 initially collapsed to single label after global
subsampling ("Number of labels is 1" error). Anomaly = top 2% by
distance-to-nearest-centroid. Final: k=3, silhouette=0.678, DB=0.650.

9.2 Hierarchical Agglomerative (Lab 11)
Ward linkage on 15,000-row subsample. All 1.05M rows assigned to
nearest centroid. Dendrogram on 500-row subsample. Anomaly = top 2% by
distance. Final: k=4, silhouette=0.353, DB=1.010.

9.3 Isolation Forest
n_estimators=200, contamination=0.02, random_state=42. Anomaly score =
-score_samples(X). Anomaly = predict(X) == -1.

9.4 Metrics
Model	Silhouette	Davies-Bouldin	Anomaly %
K-Means (k=3)	0.678	0.650	2.00%
Hierarchical (k=4, Ward)	0.353	1.010	2.00%
Isolation Forest	n/a	n/a	2.00%
9.5 Inter-Model Agreement (Jaccard)
K-Means	Hierarchical	Isolation Forest
K-Means	1.000	0.213	0.215
Hierarchical	0.213	1.000	0.293
Isolation Forest	0.215	0.293	1.000
Flag consensus:

# models	Count	%
0 (normal)	1,001,578	95.7%
1	33,265	3.2%
2	5,287	0.5%
3 (highest confidence)	6,316	0.6%
The 6,316 rows flagged by all three are the highest-confidence anomaly
set. Low agreement (21-29%) is expected - the three models optimise
different objectives (distance-based, connectivity-based,
isolation-based). They are complementary, not competing.

9.6 Limitations of Unsupervised Evaluation
No verified fraud labels. We report internal clustering metrics,
anomaly percentage, inter-model agreement, and samples for manual
inspection. We never claim any detected anomaly is confirmed fraud.
Precision and recall cannot be computed without ground truth.

10. Hybrid Predictive Analysis
For every test-set product-day:

text
Deviation    = Actual - Predicted
Deviation %  = Deviation / max(|Predicted|, eps) x 100
Six business categories:

Category	Rule
Zero-sale day	Actual = 0 and no anomaly
Normal	abs(dev%) < 25% and no anomaly
Silent overperformance	dev% > 25% and no anomaly
Silent underperformance	dev% < -25% and no anomaly
Unexpected anomaly	abs(dev%) < 25% and anomaly present
Expected spike with unusual txns	dev% > 25% and anomaly present
Missed demand with unusual txns	dev% < -25% and anomaly present
Zero-sale-day category is critical: without it, ~60% of product-days
would be misclassified as underperforming just because they had zero
sales (normal for sparse products).

10.1 Critical Bug Fixed - Training/Inference Parity
First run produced 99.9% anomaly rate instead of expected 2%.
Diagnosis: Isolation Forest trained on standardised features but
evaluated on raw features in hybrid module. Every transaction looked
anomalous due to input scale mismatch.

Fix: apply StandardScaler at inference time. Flag rate dropped to
2.00%. This is the classic training/inference parity bug that MLOps
pipelines exist to prevent. Documented as the project's most significant
challenge.

11. Simulated Streaming Monitor
StreamSimulator replays valid_sales.parquet in chronological order in
configurable batches (default 500 rows/tick) through trained Isolation
Forest.

Features: batch processing, per-transaction score, structured alerts
(Invoice, StockCode, Description, Quantity, Price, Revenue, Timestamp,
Score, Reason), alert history capped at 5,000, live counters (processed,
normal, anomalous, rate, progress), monitoring status (idle/running/
paused/completed), Start/Pause/Reset controls that persist across
Streamlit reruns, same preprocessing pipeline as training.

Honesty note: this is simulated streaming - a chronological replay
of historical data, NOT a live connection. In production, wired to
Kafka or similar.

12. Business Intelligence
56 mitigations across six categories:

Category	Severity	Count
Anomaly hotspot	high	10
Stockout risk	high	10
Overstock risk	medium	10
Demand instability	medium	10
Pricing inconsistency	medium	10
Return anomaly product	medium	6
Every mitigation carries a caveat field - a documented disclaimer that
the recommendation is a hypothesis. This is deliberate design for
responsible AI: no recommendation should be acted on without human
review.

Example:

Category: stockout_risk | Severity: high
Evidence: Mean deviation +45.2%, 8 anomalous days
Suggested action: Increase reorder quantity for next 2 weeks; check
supplier lead time; consider safety stock bump.
Caveat: Deviation may reflect seasonality or bulk orders - verify
against upcoming promotions.

13. Streamlit Dashboard
9 pages:

Page	Purpose
1. Executive Overview	10 KPI cards, revenue trends, top signals
2. Dataset Explorer	Filters, quality report, CSV download
3. Sales Analytics	6 interactive charts
4. Demand Forecasting	Product/model selectors, live forecast
5. Anomaly Detection	Metrics, distributions, flagged table
6. Live Monitoring	Start/Pause/Reset streaming sim
7. Business Intelligence	Filterable mitigation table
8. Model Evaluation	Consolidated metrics + limitations
9. Reports & Downloads	All artefacts + PDF generator
All charts use Plotly. All tables support CSV download. All data loads
are st.cache_data-cached for sub-2-second rendering.

14. Challenges and Limitations
14.1 Challenges Fixed
Silhouette on imbalanced clusters - fixed with per-cluster stratified
subsampling + fallback (-1.0).

Training/inference parity bug - fixed with StandardScaler at inference.

Duplicate StockCode descriptions - fixed by collapsing to most-
frequent (slowly-changing dimension pattern).

Zero-sale days inflating MAPE - fixed with zero-safe MAPE + zero-
sale-day hybrid category.

Leakage audit false positives - fixed by auditing against pre-clip
gap-filled grid.

14.2 Limitations
No ground-truth fraud labels - anomaly detection cannot be evaluated
by precision/recall.

Streaming is simulated, not live.

Forecast errors compound across recursive horizon.

Heavy-tailed distributions mean RMSE is dominated by a few high-
quantity days.

No fairness audit - dataset lacks protected attributes, but production
deployment should audit for disparate impact.

Single-country dominance - UK >80% of revenue, limits geographic
generalisation.

14.3 Fairness, Ethics, Responsible Use
No anomaly is called "fraud" - all are "unusual patterns warranting
review".

All mitigation recommendations are hypotheses, never automated
decisions.

No personal data used for modelling beyond CustomerID (pseudonymised).

System should never make unilateral decisions about customers or
suppliers without human oversight.

Streaming is explicitly labelled simulated.

Deploying on transaction data raises GDPR/CCPA questions requiring
legal review before production.

15. Conclusion
This project delivered a complete, working, end-to-end ML system for
retail demand forecasting and anomaly detection. It demonstrates five
distinct Lab concepts (1, 2, 5, 10, 11), a robust data pipeline with
verified leakage-free features (proven by automated tests), three
unsupervised anomaly detectors with quantified inter-model agreement, a
hybrid integration layer that connects forecasts to anomalies, a
simulated streaming monitor, and a professional 9-page dashboard.

Results Summary
1,067,371 raw transactions processed

1,025,079 valid sales retained (96% of raw)

2,004 forecastable products (42% of unique products)

5 trained models, all saved for reproducibility

56 business mitigations, all with documented caveats

21 automated tests, all passing

9 dashboard pages, all functional

Engineering Rigour
Every design decision documented with technical and business rationale.
Bugs found during development (training/inference parity, silhouette
degeneracy, duplicate descriptions, zero-sale MAPE inflation) are
diagnosed, fixed, and documented - demonstrating engineering rigour
appropriate for a professional ML system.

Future Work
Gradient boosting (XGBoost/LightGBM) for forecasting (typically
10-15% RMSE improvement on retail data)

Rolling-window cross-validation instead of single temporal split

Fairness auditing on customer segments

Real-time streaming via Kafka instead of simulated replay

Deliverables Checklist
☑ Complete Python source code (13 modules, 5 scripts)
☑ Data cleaning pipeline (3 documented streams)
☑ EDA (8 figures + insights JSON)
☑ Feature engineering with leakage audit
☑ Linear Regression forecasting
☑ Random Forest forecasting
☑ K-Means anomaly detection
☑ Hierarchical clustering + dendrogram
☑ Isolation Forest
☑ Model evaluation and comparison
☑ Hybrid predictive analysis
☑ Simulated streaming monitoring
☑ Business impact and mitigation analysis
☑ 9-page Streamlit dashboard
☑ Saved trained models
☑ Visualisations and architecture diagram
☑ 21 unit tests
☑ README.md
☑ requirements.txt
☑ Academic report (this document)
☑ Demo workflow (see DEMO_WORKFLOW.md)
