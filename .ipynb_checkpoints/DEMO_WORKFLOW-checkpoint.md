
---

## 📄 File 3: `DEMO_WORKFLOW.md`

**Location:** `C:\Users\zohai\Retail_Intelligence_System1\DEMO_WORKFLOW.md`

**Create/replace** with this **entire content**:

```markdown
# Live Demonstration Workflow — Lab 14 Viva

Total time: **10–12 minutes**. Follow this order for maximum impact.

---

## Pre-demo checklist (5 min before)

```powershell
cd C:\Users\zohai\Retail_Intelligence_System1

# Verify models and data are ready
Get-ChildItem .\models\forecasting\*.joblib
Get-ChildItem .\models\anomaly_detection\*.joblib
Get-ChildItem .\data\processed\*.parquet | Measure-Object

# Verify tests pass
pytest tests/ -q

# Launch the dashboard
streamlit run app.py