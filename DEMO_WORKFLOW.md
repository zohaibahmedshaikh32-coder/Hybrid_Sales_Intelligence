# Live Demonstration Workflow - Lab 14 Viva

Total: 10-12 minutes.

Pre-demo checklist
powershell
cd C:\Users\zohai\Retail_Intelligence_System1
pytest tests/ -q           # 21 passed
streamlit run app.py       # opens http://localhost:8501
Demo sequence
1. Introduction (30 s)
"This is Retail Intelligence - a hybrid predictive system for sales
forecasting and anomaly detection. It processes 1.07 million raw
transactions through five ML models and exposes results via a 9-page
dashboard. It demonstrates 5 Lab concepts: pre-processing, linear
regression, random forest, K-means, hierarchical clustering."

2. Executive Overview (60 s)
Show 10 KPI cards. Click Daily Revenue tab (hover). Click Monthly
Revenue tab (note Q4 spike). Scroll to Top Business Signals.

"KPIs are computed live from the cleaned dataset every time the app
loads - not cached."

3. Dataset Explorer (60 s)
Filter Country = United Kingdom. Type CHRISTMAS in search. Click
Download CSV. Expand Full quality report.

"Pipeline produced 1,025,079 valid sales rows from 1,067,371 raw.
Returns and invalid records are kept in separate streams, not
discarded."

4. Sales Analytics (90 s)
Scroll all 6 sections. Hover a bar chart. Point to return/sales ratio.

"Returns are 2% of sales value - healthy for UK retail. Preserved for
anomaly detection but excluded from forecasting."

5. Demand Forecasting - THE DEMO CENTREPIECE (2 min)
Pick top-scored product

Set horizon = 14 days

Model = Best (auto)

Click Run Forecast

"32 engineered features, recursive prediction - tomorrow's forecast
becomes today's lag feature."

Point to grey history line + blue forecast line. Read 4 KPI cards.
Scroll to Model Performance - point to best model callout.

"Random Forest outperforms Linear Regression on validation RMSE.
Selected by validation, not test - no test-set peeking."

6. Anomaly Detection (90 s)
Wait 30 s for IF computation (cached after). Point to 3 model metric
cards. Point to score distribution (red = flagged). Scroll to flagged
table. Set min-score = 0.6. Click Download flagged CSV.

"Three unsupervised models flag ~2% each, but only agree on ~21%.
This proves they're complementary. The 6,316 transactions flagged by
all three are the highest-confidence set."

7. Live Monitoring (90 s)
Click Start. Watch Processed counter climb. Anomaly Rate ~2%. Wait 15 s.
Click Pause - counters freeze. Click Start - resumes. Click Reset.

"Simulated streaming - chronological replay, not live connection. In
production this would be wired to Kafka. Anomaly rate of 2% matches
the training contamination."

8. Business Intelligence (60 s)
Point to yellow warning banner. Point to severity breakdown (20 high,
36 medium, 6 categories). Filter high. Expand Stockout risk card - read
suggested action + caveat aloud.

"Every recommendation has a caveat field. The model never claims an
anomaly is confirmed - it produces hypotheses for human review. This
is deliberate design for responsible AI."

9. Model Evaluation (60 s)
Scroll all 4 sections. Point to LR vs RF comparison. Point to Silhouette
comparison. Read Known Limitations aloud.

"We cannot compute precision/recall for anomaly models - no ground
truth. Fundamental limitation of unsupervised learning, explicitly
documented."

10. Reports & Downloads (30 s)
Click Generate PDF Report. Wait 2 s. Click Download PDF. Open - 5-page
summary.

"The system produces a self-contained academic report automatically."

Viva questions
Q1: Supervised vs unsupervised?

Forecasting is supervised (Quantity label, MAE/RMSE/R2). Anomaly
detection is unsupervised (no fraud label, use distance/connectivity/
isolation).

Q2: How did you prevent leakage?

Three ways: (1) shift(1) on all lags - verified by automated test;
(2) chronological per-product split - no shuffle; (3) model selection
uses validation RMSE, test set touched once.

Q3: Why RF over LR?

RF captures nonlinear interactions between lags and calendar. LR is
retained as interpretable Lab-2 baseline.

Q4: Why do 3 anomaly models disagree?

Different objectives: distance-based, connectivity-based,
isolation-based. Complementary, not competing. The 6,316 consensus
rows are highest-confidence.

Q5: How evaluate unsupervised without labels?

Internal clustering metrics + manual inspection. Never precision/recall
without ground truth.

Q6: Most challenging bug?

Training/inference parity bug. IF trained on standardized, evaluated
on raw -> 99.9% flag rate. Fixed with StandardScaler at inference ->
2%. Classic MLOps bug.

Q7: With more time?

Gradient boosting (XGBoost/LightGBM), rolling-window CV, fairness
auditing.

Q8: Why is MAPE high?

Many zero-sale days. Standard MAPE undefined at zero; computed only
over non-zero rows. High value reflects true difficulty of sparse
demand.

Pre-submission checklist
□ 21 tests pass
□ Dashboard runs, all 9 pages work
□ 9 screenshots saved to reports/figures/
□ PDF report generated
□ Real metrics filled in section 8.3 of project_report.md
□ Project folder zipped
Good luck!
