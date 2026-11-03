"""Layer 3 - report generation.

Input : models.CleanedDataset + models.AnalysisResult
Output: models.Report, rendered to .docx (same format as the reference report)

All narrative is written as templates with {placeholders}. Paragraph refuses
literal digits, so every number in the text comes from AnalysisResult.facts()
(or the cleaning summary) and is therefore traceable to a computed field.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from docx import Document  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402

from models import AnalysisResult, CleanedDataset, Paragraph, Report, Section, Table  # noqa: E402

# Palette (validated categorical slots 1-2, neutral chrome)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=12)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return str(path)


def trend_chart(res: AnalysisResult, out: Path) -> str:
    s = res.series
    x = range(len(s))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [p.opened for p in s], w, color=BLUE, label="Opened")
    ax.bar([i + w / 2 for i in x], [p.closed for p in s], w, color=ORANGE, label="Closed")
    roll = [(i, p.rolling_mean_opened) for i, p in zip(x, s) if p.rolling_mean_opened is not None]
    if roll:
        ax.plot(*zip(*roll), color=INK2, linewidth=2, label="Opened, 3-period rolling mean")
    ax.axhline(res.ucl, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(len(s) - 0.5, res.ucl, f" UCL {res.ucl:.1f}", va="bottom", ha="right", fontsize=8, color=INK2)
    for i, p in zip(x, s):
        if p.above_ucl:
            ax.annotate("above UCL", (i - w / 2, p.opened), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=7, color=INK2)
    ax.set_xticks(list(x), [p.period for p in s], rotation=45 if len(s) > 8 else 0, ha="right" if len(s) > 8 else "center")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    ax.set_ylim(0, max([p.opened for p in s] + [p.closed for p in s] + [res.ucl]) * 1.25 + 0.5)
    _style(ax, f"CAPAs opened and closed per {res.granularity}")
    return _save(fig, out / "trend.png")


def backlog_chart(res: AnalysisResult, out: Path) -> str:
    s = res.series
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(range(len(s)), [p.backlog for p in s], color=BLUE, linewidth=2, marker="o", markersize=5)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(range(len(s)), [p.period for p in s], rotation=45 if len(s) > 8 else 0,
                  ha="right" if len(s) > 8 else "center")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.annotate(f"{s[-1].backlog}", (len(s) - 1, s[-1].backlog), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK)
    _style(ax, "Cumulative net backlog (opened minus closed within the period)")
    return _save(fig, out / "backlog.png")


def category_chart(b, out: Path) -> str:
    rows = b.rows[:8]
    labels, values = [r.category for r in rows], [r.count for r in rows]
    if len(b.rows) > 8:                       # fold the tail into "Other"
        labels.append("Other")
        values.append(sum(r.count for r in b.rows[8:]))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(labels) + 1.0))
    ax.barh(labels, values, color=BLUE, height=0.6)
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v, i, f" {v}", va="center", fontsize=8, color=INK2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlim(0, max(values) * 1.15)
    _style(ax, f"CAPAs by {b.dimension.lower()}")
    return _save(fig, out / f"by_{b.field}.png")


# --------------------------------------------------------------------------- #
# Report model assembly
# --------------------------------------------------------------------------- #
def _f(v, unit=""):
    if v is None:
        return "n/a"
    s = f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"
    return f"{s}{unit}"


def build_report(ds: CleanedDataset, res: AnalysisResult, chart_dir: Path) -> Report:
    chart_dir.mkdir(parents=True, exist_ok=True)
    facts = res.facts() | {
        "data.source": ds.source_file, "data.rows_in": str(ds.rows_in), "data.rows_out": str(ds.rows_out),
        "data.removed": str(ds.rows_in - ds.rows_out),
        "data.steps": str(sum(1 for s in ds.log if s.rows_affected)),
    }
    facts |= {f"assumption.{i}": a for i, a in enumerate(ds.assumptions)}

    def P(t: str) -> Paragraph:
        return Paragraph(template=t).render(facts)

    def I(*ids) -> list[Paragraph]:
        return [P(f"{{insight.{i}.interpretation}}") for i in ids if res.insight(i)]

    k, g = res.kpis, res.granularity
    sections: list[Section] = []

    # 1 ------------------------------------------------------------------
    sections.append(Section(heading="1. Purpose and Scope", paragraphs=[
        P("This report reviews Corrective and Preventive Action (CAPA) records initiated between {period_start} "
          "and {period_end}. It measures CAPA volume, closure performance and timeliness, tests whether CAPA "
          "intake is trending or shifting over time, and identifies the departments, sources and root causes "
          "that generate most CAPAs."),
        P("Open and overdue status is assessed as of {as_of}. Every figure is traceable to the accompanying "
          "analysis file, and every finding carries a confidence grade reflecting how much data supports it."),
    ]))

    # 2 ------------------------------------------------------------------
    sections.append(Section(heading="2. Executive Summary", paragraphs=[
        P("Between {period_start} and {period_end}, {kpi.total} CAPAs were initiated. As of {as_of}, "
          "{kpi.closed} are closed, {kpi.open} are open and {kpi.overdue_open} are overdue."),
    ], bullets=I("closure_rate", "on_time", "extensions", "overdue", "backlog", "trend", "control",
                 "top_site", "top_department", "top_root_cause")))

    # 3 ------------------------------------------------------------------
    sections.append(Section(heading="3. Data Sources and Preparation", paragraphs=[
        P("{data.rows_in} rows were read from {data.source}. After cleaning, {data.rows_out} valid CAPA records "
          "remain ({data.removed} rows removed). Each transformation is listed below with the number of rows "
          "affected and the reason."),
    ], tables=[
        Table(headers=["Step", "Field", "Rows affected", "Reason"],
              rows=[[s.step, s.column or "-", str(s.rows_affected), s.reason] for s in ds.log]),
        Table(headers=["Report field", "Source column"], rows=[[f, c] for f, c in ds.column_mapping.items()]),
    ]))

    # 4 ------------------------------------------------------------------
    sections.append(Section(heading="4. CAPA Status Overview", paragraphs=[
        P("As of {as_of}, the closure rate is {kpi.closure_rate_pct}% of non-cancelled CAPAs. Cancelled CAPAs "
          "({kpi.cancelled}) are excluded from the closure rate."),
    ], tables=[Table(headers=["Metric", "Value", "Basis"], rows=[
        ["CAPAs initiated", _f(k.total), "All records in period"],
        ["Closed", _f(k.closed), f"as of {res.as_of:%d %b %Y}"],
        ["Open", _f(k.open), f"as of {res.as_of:%d %b %Y}"],
        ["Cancelled", _f(k.cancelled), "Excluded from rates"],
        ["Overdue (open, past target)", _f(k.overdue_open), f"as of {res.as_of:%d %b %Y}"],
        ["Closure rate", _f(k.closure_rate_pct, "%"), "Closed / non-cancelled"],
        ["On-time closure (original target)", _f(k.on_time_closure_pct, "%"), "Closed on/before original target"],
        ["On-time closure (revised due date)", _f(k.on_time_revised_pct, "%"), "Closed on/before extended due date"],
        ["CAPAs extended at least once", _f(k.extended_pct, "%"), "Extension counter > 0"],
        ["Median days to close", _f(k.median_days_to_close), "Closed CAPAs with a closure date"],
        ["90th percentile days to close", _f(k.p90_days_to_close), "Closed CAPAs with a closure date"],
        ["Mean age of open CAPAs (days)", _f(k.mean_age_open_days), f"as of {res.as_of:%d %b %Y}"],
    ])]))

    # 5 ------------------------------------------------------------------
    sections.append(Section(heading="5. Trend Analysis", paragraphs=[
        P("CAPAs are counted per {granularity} of initiation (opened) and of closure (closed). The dashed line is "
          "the upper control limit of a c-chart ({ucl} per {granularity}, centre line {centre_line}); a "
          "{granularity} above it signals a special cause rather than normal variation."),
    ], bullets=I("volume", "pop_change", "trend", "shift", "control", "backlog"),
        charts=[trend_chart(res, chart_dir), backlog_chart(res, chart_dir)],
        tables=[Table(headers=["Period", "Opened", "Closed", "Net backlog", "3-period rolling mean", "Above UCL"],
                      rows=[[p.period, str(p.opened), str(p.closed), str(p.backlog), _f(p.rolling_mean_opened),
                             "Yes" if p.above_ucl else "-"] for p in res.series])]))

    # 6 ------------------------------------------------------------------
    c_bullets, c_charts, c_tables = [], [], []
    for b in res.breakdowns:
        c_bullets += I(f"top_{b.field}")
        c_charts.append(category_chart(b, chart_dir))
        c_tables.append(Table(
            headers=[b.dimension, "CAPAs", "% of total", "Open", "Overdue", "Median days to close"],
            rows=[[r.category, str(r.count), _f(r.pct, "%"), str(r.open), str(r.overdue), _f(r.median_days_to_close)]
                  for r in b.rows]))
    intro = (P("Categories are ranked by CAPA count. The few categories that account for most CAPAs (the Pareto "
               "set) are where corrective effort has the most effect.") if res.breakdowns else
             P("No department, source, root cause or priority fields were found in the data, so no category "
               "analysis is possible."))
    sections.append(Section(heading="6. Category Analysis", paragraphs=[intro], bullets=c_bullets,
                            charts=c_charts, tables=c_tables))

    # 7 ------------------------------------------------------------------
    timely = I("on_time", "extensions", "cycle_time", "overdue", "outliers")
    top_overdue = res.overdue_items[:20]
    sections.append(Section(
        heading="7. Timeliness and Overdue CAPAs",
        paragraphs=[P("No open CAPA is past its target date as of {as_of}.") if not res.overdue_items else
                    P("The {kpi.overdue_open} open CAPAs past their current (revised) due date are listed below, "
                  "most overdue first, for follow-up.")],
        bullets=timely or [P("Timeliness could not be assessed because the data has no closure or target dates.")],
        tables=[Table(headers=["CAPA ID", "Department", "Initiated", "Current due date", "Days overdue"],
                      rows=[[o.capa_id, o.department, f"{o.initiation_date:%d %b %Y}", f"{o.target_date:%d %b %Y}",
                             str(o.days_overdue)] for o in top_overdue])] if top_overdue else []))

    # 8 ------------------------------------------------------------------
    sections.append(Section(heading="8. Key Findings", paragraphs=[
        P("Each finding below is stated with its supporting number, the time window it covers and a confidence "
          "grade: High means at least thirty records support it, Moderate at least ten, Low fewer, and "
          "Insufficient data means no conclusion is drawn."),
    ], tables=[Table(headers=["Finding", "Value", "Window", "Confidence", "Interpretation"],
                     rows=[[i.title, f"{_f(i.value)} {i.unit}".strip(), i.window, i.confidence, i.interpretation]
                           for i in res.insights])]))

    # 9 ------------------------------------------------------------------
    recs: list[Paragraph] = []
    if k.overdue_open:
        recs.append(P("Clear the {kpi.overdue_open} overdue CAPAs, starting with the oldest listed in section seven, "
                      "and ensure any target-date extensions are justified and approved."))
    if k.on_time_closure_pct is not None and k.on_time_closure_pct < 90:
        recs.append(P("On-time closure is {kpi.on_time_closure_pct}%. Check that target dates are realistic when "
                      "set, and escalate CAPAs approaching their due date."))
    if k.extended_pct is not None and k.extended_pct >= 30:
        recs.append(P("{kpi.extended_pct}% of CAPAs needed a due-date extension. Set targets from a realistic "
                      "action plan at initiation and review repeat extensions at the QA management review."))
    bl = res.insight("backlog")
    if bl and bl.value and bl.value > 0:
        recs.append(P("The open backlog grew by {insight.backlog.value} CAPAs in the period; closure capacity "
                      "should be increased to at least match intake."))
    ctl = res.insight("control")
    if ctl and ctl.value:
        recs.append(P("Investigate the {insight.control.value} {granularity}(s) above the control limit for a "
                      "common trigger such as an audit, a batch failure or a process change."))
    tr = res.insight("trend")
    if tr and tr.confidence == "Insufficient data":
        recs.append(P("Repeat this analysis once more {granularity}s of data are available, so that trends can "
                      "be tested formally rather than inferred."))
    elif tr and "significant increasing" in tr.interpretation:
        recs.append(P("{insight.trend.interpretation} Identify the driver before it becomes systemic."))
    if "top.root_cause" in facts:
        recs.append(P("{top.root_cause} is the largest root-cause category ({top.root_cause.pct}% of CAPAs). "
                      "Review the effectiveness checks for these CAPAs and whether the same causes recur."))
    if "top.department" in facts:
        recs.append(P("{top.department} raises the most CAPAs ({top.department.pct}%). A focused review with "
                      "this department is likely to have the greatest impact."))
    if any(s.step in ("Missing category", "Malformed dates", "Drop undated rows", "Missing CAPA ID",
                      "Duplicate CAPA IDs") for s in ds.log):
        recs.append(P("Strengthen data entry in the CAPA system: {data.steps} cleaning steps had to correct "
                      "records in this extract (see section three)."))
    if not recs:
        recs.append(P("No signal requiring escalation was found; continue routine periodic monitoring."))
    sections.append(Section(heading="9. Conclusions and Recommendations", paragraphs=[
        P("Based on the findings above, the following actions are recommended:")], bullets=recs))

    # 10 -----------------------------------------------------------------
    sections.append(Section(heading="10. Limitations and Assumptions",
                            bullets=[P(f"{{limitation.{i}}}") for i in range(len(res.limitations))]
                            + [P(f"{{assumption.{i}}}") for i in range(len(ds.assumptions))]))

    return Report(title="CAPA Trend Analysis Report",
                  subtitle=f"Period: {res.period_start:%d %b %Y} to {res.period_end:%d %b %Y}  |  "
                           f"Status as of {res.as_of:%d %b %Y}  |  Generated {date.today():%d %b %Y}",
                  sections=sections)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _add_table(doc, t: Table):
    tbl = doc.add_table(rows=1, cols=len(t.headers))
    tbl.style = "Light Grid Accent 1"
    for cell, h in zip(tbl.rows[0].cells, t.headers):
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
    for r in t.rows:
        for cell, v in zip(tbl.add_row().cells, r):
            cell.text = v
    for row in tbl.rows:
        for cell in row.cells:
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(8.5)
    doc.add_paragraph()


def render_docx(report: Report, path: Path) -> Path:
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.add_heading(report.title, 0)
    sub = doc.add_paragraph().add_run(report.subtitle)
    sub.italic, sub.font.color.rgb = True, RGBColor(0x52, 0x51, 0x4E)
    for s in report.sections:
        doc.add_heading(s.heading, 1)
        for p in s.paragraphs:
            doc.add_paragraph(p.text)
        for b in s.bullets:
            doc.add_paragraph(b.text, style="List Bullet")
        if len(s.charts) == len(s.tables):          # chart + its table, pairwise
            for c, t in zip(s.charts, s.tables):
                doc.add_picture(c, width=Inches(6.3))
                _add_table(doc, t)
        else:
            for c in s.charts:
                doc.add_picture(c, width=Inches(6.3))
            for t in s.tables:
                _add_table(doc, t)
    doc.save(path)
    return path
