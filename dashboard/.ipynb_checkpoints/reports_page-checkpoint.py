"""
reports_page.py — Page 9: Reports & Downloads.

One-stop download hub for every processed artefact the pipeline has
generated. Optionally builds a PDF summary report.
"""
from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

from dashboard.ui import (
    empty_state, fmt_int, load_json, page_header, section,
)
from src.config import BASE_DIR, PROCESSED_DIR


_DOWNLOADS = [
    ("valid_sales",           "Cleaned sales transactions",       "parquet"),
    ("returns",               "Returns and cancellations",         "parquet"),
    ("invalid",               "Invalid / unusable records",        "parquet"),
    ("daily_product_features","Forecasting feature table",         "parquet"),
    ("anomaly_features",      "Anomaly feature table",             "parquet"),
    ("hybrid_deviations",     "Hybrid: per-row deviations",        "parquet"),
    ("hybrid_anomaly_xref",   "Hybrid: anomaly × deviation join",  "parquet"),
    ("hybrid_product_summary","Hybrid: per-product summary",       "parquet"),
    ("train",                 "Training split",                    "parquet"),
    ("val",                   "Validation split",                  "parquet"),
    ("test",                  "Test split",                         "parquet"),
    ("stream_alerts",         "Streaming alert history",           "parquet"),
]

_JSON_DOWNLOADS = [
    ("eda_insights",         "EDA insights"),
    ("hybrid_insights",      "Hybrid insights"),
    ("business_mitigations", "Business mitigations"),
]

_CSV_DOWNLOADS = [
    ("data_quality_report", "Data quality report"),
    ("dataset_understanding", "Dataset understanding"),
]


def render() -> None:
    page_header(
        "Reports & Downloads",
        "Download every cleaned, aggregated, and generated artefact "
        "from the pipeline.",
    )

    # =================================================================
    # 1. Processed tables (Parquet)
    # =================================================================
    section("1. Processed Data Tables (Parquet)")

    rows = []
    for name, label, ext in _DOWNLOADS:
        p = PROCESSED_DIR / f"{name}.{ext}"
        rows.append({
            "name": name,
            "label": label,
            "exists": p.exists(),
            "size_mb": (round(p.stat().st_size / 1e6, 2)
                        if p.exists() else 0.0),
            "path": p,
        })

    for r in rows:
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            status = "✅" if r["exists"] else "❌"
            st.write(f"{status} **{r['label']}**  \n"
                     f"`{r['name']}.parquet`")
        with c2:
            if r["exists"]:
                st.write(f"{r['size_mb']} MB")
            else:
                st.write("—")
        with c3:
            if r["exists"]:
                with open(r["path"], "rb") as f:
                    st.download_button(
                        "⬇️ Download",
                        data=f.read(),
                        file_name=f"{r['name']}.parquet",
                        mime="application/octet-stream",
                        key=f"dl_{r['name']}",
                    )

    # =================================================================
    # 2. JSON reports
    # =================================================================
    section("2. JSON Reports")
    for name, label in _JSON_DOWNLOADS:
        p = PROCESSED_DIR / f"{name}.json"
        if p.exists():
            st.download_button(
                f"⬇️ {label} (JSON)",
                data=p.read_bytes(),
                file_name=f"{name}.json",
                mime="application/json",
                key=f"dl_json_{name}",
            )
        else:
            st.caption(f"❌ {label} — not found")

    # =================================================================
    # 3. CSV reports
    # =================================================================
    section("3. CSV Reports")
    for name, label in _CSV_DOWNLOADS:
        p = PROCESSED_DIR / f"{name}.csv"
        if p.exists():
            st.download_button(
                f"⬇️ {label} (CSV)",
                data=p.read_bytes(),
                file_name=f"{name}.csv",
                mime="text/csv",
                key=f"dl_csv_{name}",
            )
        else:
            st.caption(f"❌ {label} — not found")

    # =================================================================
    # 4. Model artefacts
    # =================================================================
    section("4. Trained Models (.joblib)")
    model_rows = [
        ("forecasting/LinearRegression.joblib", "Linear Regression"),
        ("forecasting/RandomForest.joblib",     "Random Forest"),
        ("forecasting/metadata.json",           "Forecasting metadata"),
        ("anomaly_detection/kmeans.joblib",     "K-Means"),
        ("anomaly_detection/hierarchical.joblib","Hierarchical"),
        ("anomaly_detection/isolation_forest.joblib","Isolation Forest"),
        ("anomaly_detection/metadata.json",     "Anomaly metadata"),
    ]
    for rel, label in model_rows:
        p = BASE_DIR / "models" / rel
        if p.exists():
            size_mb = round(p.stat().st_size / 1e6, 2)
            st.download_button(
                f"⬇️ {label} ({size_mb} MB)",
                data=p.read_bytes(),
                file_name=p.name,
                key=f"dl_model_{rel.replace('/', '_')}",
            )
        else:
            st.caption(f"❌ {label} — not found")

    # =================================================================
    # 5. PDF report (optional, best-effort)
    # =================================================================
    section("5. Full PDF Summary Report")
    st.caption(
        "Generates a professional PDF summary of KPIs, model metrics, "
        "hybrid analysis, and top business recommendations.")

    if st.button("📄 Generate PDF Report", type="primary"):
        _generate_pdf()


# ---------------------------------------------------------------------------
# PDF generator (best-effort, ReportLab)
# ---------------------------------------------------------------------------
def _generate_pdf() -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak,
        )
    except ImportError:
        st.error("`reportlab` not installed. Run `pip install reportlab`.")
        return

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Retail Intelligence – Summary Report",
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                        fontSize=20, spaceAfter=14,
                        textColor=colors.HexColor("#1E3A8A"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                        fontSize=14, spaceAfter=8, spaceBefore=14,
                        textColor=colors.HexColor("#2563EB"))
    body = styles["BodyText"]

    story = [Paragraph("Retail Intelligence — Hybrid Predictive System",
                       h1),
             Paragraph("Summary Report", styles["Heading3"]),
             Spacer(1, 0.5*cm)]

    # -- Section 1: Dataset
    story.append(Paragraph("1. Dataset", h2))
    dq_path = PROCESSED_DIR / "data_quality_report.csv"
    if dq_path.exists():
        dq = pd.read_csv(dq_path)
        data = [["Metric", "Value"]] + dq.values.tolist()
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2563EB")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.whitesmoke),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        story.append(t)

    # -- Section 2: Forecasting metrics
    story.append(Paragraph("2. Forecasting Models", h2))
    fc = _load_json_raw("forecasting")
    if fc and "metrics" in fc:
        rows = [["Model", "Split", "MAE", "RMSE", "R²", "MAPE %"]]
        for m, splits in fc["metrics"].items():
            for s in ("val", "test"):
                x = splits.get(s, {})
                rows.append([
                    m, s,
                    f"{x.get('MAE', 0):.2f}",
                    f"{x.get('RMSE', 0):.2f}",
                    f"{x.get('R2', 0):.3f}",
                    f"{x.get('MAPE', 0):.2f}",
                ])
        t = Table(rows, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2563EB")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.whitesmoke),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        story.append(t)
        story.append(Paragraph(
            f"<b>Best model:</b> {fc.get('best_model','—')}", body))

    # -- Section 3: Anomaly detection
    story.append(PageBreak())
    story.append(Paragraph("3. Anomaly Detection Models", h2))
    an = _load_json_raw("anomaly_detection")
    if an and "metrics" in an:
        mdf = pd.DataFrame(an["metrics"])
        rows = [mdf.columns.tolist()] + mdf.round(3).astype(str).values.tolist()
        t = Table(rows, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2563EB")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.whitesmoke),
            ("FONTSIZE", (0,0), (-1,-1), 7),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        story.append(t)

    # -- Section 4: Business mitigations
    story.append(Paragraph("4. Top Business Recommendations", h2))
    bm = _load_json_raw("business_mitigations")
    if bm and bm.get("mitigations"):
        high = [m for m in bm["mitigations"] if m["severity"] == "high"][:10]
        for i, m in enumerate(high, 1):
            story.append(Paragraph(
                f"<b>{i}. {m['category']}</b> — {m['Description']}  \n"
                f"<i>Evidence:</i> {m['evidence']}  \n"
                f"<i>Action:</i> {m['suggested_action']}  \n"
                f"<i>Caveat:</i> {m['caveat']}", body))
            story.append(Spacer(1, 0.3*cm))

    # -- Section 5: Scope & caveats
    story.append(Paragraph("5. Scope and Disclaimer", h2))
    story.append(Paragraph(
        "All anomaly outputs are unsupervised model predictions. "
        "No transaction is confirmed as fraud or error. All recommended "
        "actions must be validated against inventory, pricing, and "
        "customer data before execution. Streaming is simulated using "
        "the historical UCI Online Retail II dataset; it is not "
        "connected to any live system.", body))

    doc.build(story)
    buf.seek(0)
    st.download_button(
        "📄 Download PDF report",
        data=buf.getvalue(),
        file_name="Retail_Intelligence_Report.pdf",
        mime="application/pdf",
    )
    st.success("PDF generated successfully.")


def _load_json_raw(name: str) -> dict:
    p = BASE_DIR / "models" / name / "metadata.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}