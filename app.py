"""Minimal front end for the CAPA trend report pipeline.

    streamlit run app.py
"""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import analyse
from cleaning import clean
from report import build_report, render_docx
from validate import compare

st.set_page_config(page_title="CAPA Trend Report", layout="centered")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)
st.title("CAPA Trend Report")


@st.cache_data(show_spinner="Generating report…")
def run_pipeline(name: str, data: bytes):
    work = Path(tempfile.mkdtemp())
    src = work / name
    src.write_bytes(data)
    ds = clean(src)
    res = analyse(ds)
    report = build_report(ds, res, work / "charts")
    docx = render_docx(report, work / "CAPA_Trend_Report.docx").read_bytes()
    try:
        checks = compare(src, ds, res, report)
    except Exception:          # independent check only works on TrackWise CSV exports
        checks = None
    return res, report, docx, checks


upload = st.file_uploader("Upload CAPA file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
if upload is None:
    st.stop()

try:
    res, report, docx, checks = run_pipeline(upload.name, upload.getvalue())
except Exception as e:
    st.error(f"Could not generate the report: {e}")
    st.stop()

k = res.kpis
c1, c2, c3, c4 = st.columns(4)
c1.metric("CAPAs", k.total)
c2.metric("Open", k.open)
c3.metric("Overdue", k.overdue_open)
c4.metric("Closure rate", f"{k.closure_rate_pct:.1f}%")

st.download_button("Download report", docx, "CAPA_Trend_Report.docx",
                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                   type="primary", use_container_width=True)

if checks:
    passed = sum(r["Result"] == "PASS" for r in checks)
    with st.expander(f"Accuracy: {passed}/{len(checks)} checks match"):
        st.dataframe(pd.DataFrame(checks), hide_index=True, use_container_width=True)

st.divider()
for s in report.sections:
    st.markdown(f"#### {s.heading}")
    for p in s.paragraphs:
        st.write(p.text)
    for b in s.bullets:
        st.markdown(f"- {b.text}")
    for chart in s.charts:
        st.image(chart, use_container_width=True)
    if s.tables:
        with st.expander("Tables"):
            for t in s.tables:
                st.dataframe(pd.DataFrame(t.rows, columns=t.headers), hide_index=True, use_container_width=True)
