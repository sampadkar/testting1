"""Accuracy check: recompute the key figures independently from the raw TrackWise CSV
(plain pandas, explicit column names, no code shared with the pipeline) and compare
them with what the three-layer pipeline produced.

    python validate.py "capa_2025.csv"
"""
import sys
from pathlib import Path

import pandas as pd

from analysis import analyse
from cleaning import clean
from report import build_report


def ground_truth(path: Path) -> dict:
    raw = pd.read_csv(path)
    d = raw.drop_duplicates("capa_number", keep="last").copy()
    day = lambda c: pd.to_datetime(d[c], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    opened, done, orig, due = day("date_opened"), day("date_completed"), day("proposed_closure_date"), day("due_date")
    as_of = pd.to_datetime(raw["silver_load_time"], utc=True).dt.tz_localize(None).dt.normalize().max()
    status = d["record_status"].map({"Closed - Done": "Closed", "Closed - Cancelled": "Cancelled"}).fillna("Open")
    closed, open_ = status == "Closed", status == "Open"
    cur_due = due.fillna(orig)
    t = closed & done.notna() & orig.notna()
    r = closed & done.notna() & cur_due.notna()
    ext = pd.to_numeric(d["ext_counter"], errors="coerce").dropna()
    top = lambda c: d[c].fillna("Unknown").value_counts().head(3).to_dict()
    return {
        "raw rows": len(raw),
        "unique CAPAs": len(d),
        "open": int(open_.sum()), "closed": int(closed.sum()), "cancelled": int((status == "Cancelled").sum()),
        "overdue open": int((open_ & (cur_due < as_of)).sum()),
        "closure rate %": round(100 * closed.sum() / (status != "Cancelled").sum(), 1),
        "on-time vs original %": round(100 * (done[t] <= orig[t]).mean(), 1),
        "on-time vs revised %": round(100 * (done[r] <= cur_due[r]).mean(), 1),
        "extended %": round(100 * (ext > 0).mean(), 1),
        "median days to close": round(float((done - opened)[closed].dropna().dt.days.median()), 1),
        "as of": as_of.date(),
        "opened per month": opened.dt.to_period("M").value_counts().sort_index().astype(int).tolist(),
        "top sites": top("site_name"), "top departments": top("resp_dept"),
        "top sources": top("origin_qms"), "top classifications": top("capa_classification"),
    }


def pipeline_values(ds, res, report) -> dict:
    k = res.kpis
    tops = {b.field: {r.category: r.count for r in b.rows[:3]} for b in res.breakdowns}
    leftover = [p.text for s in report.sections for p in s.paragraphs + s.bullets if "{" in p.text]
    return {
        "raw rows": ds.rows_in, "unique CAPAs": ds.rows_out,
        "open": k.open, "closed": k.closed, "cancelled": k.cancelled, "overdue open": k.overdue_open,
        "closure rate %": k.closure_rate_pct, "on-time vs original %": k.on_time_closure_pct,
        "on-time vs revised %": k.on_time_revised_pct, "extended %": k.extended_pct,
        "median days to close": k.median_days_to_close, "as of": res.as_of,
        "opened per month": [p.opened for p in res.series] if res.granularity == "month" else "weekly",
        "top sites": tops.get("site"), "top departments": tops.get("department"),
        "top sources": tops.get("source"), "top classifications": tops.get("root_cause"),
        "_mapping": ds.column_mapping, "_unrendered": leftover,
    }


def compare(path: Path, ds, res, report) -> list[dict]:
    """One row per check: expected (independent) vs pipeline value. Needs a TrackWise-format CSV."""
    exp, got = ground_truth(path), pipeline_values(ds, res, report)
    rows = []
    for key, e in exp.items():
        g = got.get(key)
        ok = abs(e - g) <= 0.1 if isinstance(e, float) and isinstance(g, (int, float)) else e == g
        rows.append({"Check": key, "Expected": str(e), "Pipeline": str(g), "Result": "PASS" if ok else "FAIL"})
    rows.append({"Check": "unrendered placeholders", "Expected": "0", "Pipeline": str(len(got["_unrendered"])),
                 "Result": "PASS" if not got["_unrendered"] else "FAIL"})
    return rows


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "capa_2025.csv")
    ds = clean(path)
    res = analyse(ds)
    report = build_report(ds, res, Path("output/charts"))   # also proves the Report model validates
    rows = compare(path, ds, res, report)
    print(f"{'Check':<26}{'Expected':<40}{'Pipeline':<40}Result")
    for r in rows:
        print(f"{r['Check']:<26}{r['Expected'][:38]:<40}{r['Pipeline'][:38]:<40}{r['Result']}")
    passed = sum(r["Result"] == "PASS" for r in rows)
    print(f"\nColumn mapping used: {ds.column_mapping}")
    print(f"\nAccuracy: {passed}/{len(rows)} checks match ({100 * passed / len(rows):.0f}%)")


if __name__ == "__main__":
    main()
