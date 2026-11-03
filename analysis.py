"""Layer 2 - statistical insight.

Input : models.CleanedDataset
Output: models.AnalysisResult

Every insight carries its number, the window it covers, a plain-language
interpretation and a confidence grade. Small samples are graded down and
described as such rather than reported as trends.

Methods (no SciPy needed):
  * c-chart control limit on CAPAs opened per period: UCL = mean + 3*sqrt(mean)
  * Mann-Kendall test for monotonic trend (needs >= 8 complete periods)
  * Poisson two-count z-test for period-over-period and first-half vs second-half shifts
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from models import (AnalysisResult, Breakdown, CategoryRow, CleanedDataset, Insight, KPIs,
                    OverdueItem, PeriodPoint)

DIMENSIONS = {"site": "Site", "department": "Department", "source": "Source", "capa_type": "CAPA Type",
              "root_cause": "Root Cause / Classification", "priority": "Priority"}
MIN_TREND_PERIODS = 8


def _p_two_sided(z: float) -> float:
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def _r(x, nd=1):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def _confidence(n: int) -> str:
    return "High" if n >= 30 else "Moderate" if n >= 10 else "Low" if n >= 3 else "Insufficient data"


def mann_kendall(x: list[float]) -> tuple[float, float]:
    """Return (S statistic, two-sided p-value) with tie-corrected variance."""
    n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, ties = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in ties)) / 18
    if var <= 0:
        return float(s), 1.0
    z = (s - np.sign(s)) / math.sqrt(var)
    return float(s), _p_two_sided(z)


def poisson_shift(a: int, b: int) -> float | None:
    """p-value for H0: two equal-exposure counts come from the same Poisson rate."""
    return None if a + b == 0 else _p_two_sided((b - a) / math.sqrt(a + b))


def analyse(ds: CleanedDataset) -> AnalysisResult:
    df = pd.DataFrame([r.model_dump() for r in ds.records])
    for c in ("initiation_date", "target_date", "revised_due_date", "closure_date"):
        df[c] = pd.to_datetime(df[c])

    start, end = df["initiation_date"].min(), df["initiation_date"].max()
    if ds.as_of_date:
        as_of = pd.Timestamp(ds.as_of_date)
        basis = "the extract date in the file"
    else:
        as_of = max(end, df["closure_date"].max()) if df["closure_date"].notna().any() else end
        basis = "the latest date in the data (no extract date column found)"
    window = f"{start:%d %b %Y} - {end:%d %b %Y}"
    status_window = f"as of {as_of:%d %b %Y}"
    limitations: list[str] = [f"Open/overdue status is assessed as of {as_of:%d %b %Y}, {basis}."]
    due = df["revised_due_date"].fillna(df["target_date"])     # current commitment

    # ---- KPIs --------------------------------------------------------------
    active = df[df.status != "Cancelled"]
    closed = df[df.status == "Closed"]
    open_ = df[df.status == "Open"]
    days = (closed.closure_date - closed.initiation_date).dt.days.dropna()
    timed = closed.dropna(subset=["closure_date", "target_date"])
    revised = closed.assign(due=due[closed.index]).dropna(subset=["closure_date", "due"])
    overdue_mask = (df.status == "Open") & due.notna() & (due < as_of)
    ext = df.extension_count.dropna()
    kpis = KPIs(
        total=len(df), open=len(open_), closed=len(closed), cancelled=int((df.status == "Cancelled").sum()),
        overdue_open=int(overdue_mask.sum()),
        closure_rate_pct=_r(100 * len(closed) / len(active)) if len(active) else 0.0,
        on_time_closure_pct=_r(100 * (timed.closure_date <= timed.target_date).mean()) if len(timed) else None,
        on_time_revised_pct=(_r(100 * (revised.closure_date <= revised.due).mean())
                             if len(revised) and df.revised_due_date.notna().any() else None),
        extended_pct=_r(100 * (ext > 0).mean()) if len(ext) else None,
        median_days_to_close=_r(days.median()) if len(days) else None,
        p90_days_to_close=_r(days.quantile(0.9)) if len(days) else None,
        mean_age_open_days=_r((as_of - open_.initiation_date).dt.days.mean()) if len(open_) else None,
    )

    # ---- Time series -------------------------------------------------------
    granularity = "week" if (end - start).days <= 120 else "month"
    freq = "W-SUN" if granularity == "week" else "M"
    periods = pd.period_range(start.to_period(freq), end.to_period(freq), freq=freq)
    opened = df.initiation_date.dt.to_period(freq).value_counts().reindex(periods, fill_value=0)
    closed_ct = (closed.closure_date.dropna().dt.to_period(freq).value_counts()
                 .reindex(periods, fill_value=0))
    backlog = (opened - closed_ct).cumsum()
    rolling = opened.rolling(3, min_periods=3).mean()
    centre = float(opened.mean())
    ucl = centre + 3 * math.sqrt(centre) if centre > 0 else 0.0
    label = (lambda p: f"Wk {p.start_time:%d %b}") if granularity == "week" else (lambda p: f"{p.start_time:%b %Y}")
    series = [PeriodPoint(period=label(p), opened=int(opened[p]), closed=int(closed_ct[p]), backlog=int(backlog[p]),
                          rolling_mean_opened=_r(rolling[p]), above_ucl=bool(opened[p] > ucl))
              for p in periods]
    last_partial = end < periods[-1].end_time.normalize()
    complete = opened.iloc[:-1] if last_partial and len(opened) > 1 else opened
    if last_partial:
        limitations.append(f"The final {granularity} ({label(periods[-1])}) is incomplete in the data and is "
                           "excluded from period-over-period and trend tests.")
    if granularity == "week":
        limitations.append("The data spans four months or less, so weekly periods are used to give the trend "
                           "tests enough points; weekly counts are small and naturally noisy.")

    insights: list[Insight] = []
    unit_p = f"CAPAs/{granularity}"

    insights.append(Insight(
        id="volume", title="CAPAs initiated", value=kpis.total, unit="CAPAs", window=window,
        interpretation=f"{kpis.total} CAPAs were initiated over {len(periods)} {granularity}s, "
                       f"an average of {centre:.1f} per {granularity}.",
        confidence="High"))

    insights.append(Insight(
        id="closure_rate", title="Closure rate", value=kpis.closure_rate_pct, unit="%", window=status_window,
        interpretation=f"{kpis.closed} of {len(active)} non-cancelled CAPAs ({kpis.closure_rate_pct:.1f}%) are closed; "
                       f"{kpis.open} remain open.",
        confidence=_confidence(len(active))))

    if kpis.on_time_closure_pct is not None:
        late = int((timed.closure_date > timed.target_date).sum())
        extra = (f" Against the revised (extended) due date, {kpis.on_time_revised_pct:.1f}% were on time, so "
                 "extensions hide much of the slippage." if kpis.on_time_revised_pct is not None
                 and kpis.on_time_revised_pct - kpis.on_time_closure_pct >= 10 else
                 f" Against the revised due date, {kpis.on_time_revised_pct:.1f}% were on time."
                 if kpis.on_time_revised_pct is not None else "")
        insights.append(Insight(
            id="on_time", title="On-time closure (original target)", value=kpis.on_time_closure_pct, unit="%",
            window=status_window,
            interpretation=f"{kpis.on_time_closure_pct:.1f}% of closed CAPAs met their original target date; "
                           f"{late} closed after it." + extra,
            confidence=_confidence(len(timed))))
    else:
        limitations.append("No target dates are available, so on-time closure cannot be measured.")

    if kpis.extended_pct is not None:
        multi = int((ext >= 3).sum())
        insights.append(Insight(
            id="extensions", title="CAPAs needing a due-date extension", value=kpis.extended_pct, unit="%",
            window=status_window,
            interpretation=f"{kpis.extended_pct:.1f}% of CAPAs were extended at least once (mean "
                           f"{ext.mean():.1f} extensions per CAPA); {multi} were extended three or more times, "
                           "which suggests target dates are set too optimistically or actions are under-resourced.",
            confidence=_confidence(len(ext))))

    if kpis.median_days_to_close is not None:
        insights.append(Insight(
            id="cycle_time", title="Median days to close", value=kpis.median_days_to_close, unit="days",
            window=window,
            interpretation=f"Half of closed CAPAs closed within {kpis.median_days_to_close:.0f} days; the slowest "
                           f"tenth took more than {kpis.p90_days_to_close:.0f} days. Note that CAPAs still open are "
                           "excluded, so this understates true cycle time while many remain open.",
            confidence=_confidence(len(days))))

    if len(open_):
        share = 100 * kpis.overdue_open / len(open_)
        insights.append(Insight(
            id="overdue", title="Overdue open CAPAs", value=kpis.overdue_open, unit="CAPAs", window=status_window,
            interpretation=(f"{kpis.overdue_open} of {len(open_)} open CAPAs ({share:.0f}%) are past their target date; "
                            f"open CAPAs have been open {kpis.mean_age_open_days:.0f} days on average.")
            if kpis.overdue_open else f"None of the {len(open_)} open CAPAs is past its target date.",
            confidence=_confidence(len(open_))))

    net = int(opened.sum() - closed_ct.sum())
    insights.append(Insight(
        id="backlog", title="Net backlog change", value=net, unit="CAPAs", window=window,
        interpretation=(f"{int(opened.sum())} CAPAs were opened and {int(closed_ct.sum())} closed within the window, "
                        f"so the backlog {'grew' if net > 0 else 'shrank' if net < 0 else 'was unchanged'} by "
                        f"{abs(net)}. " + ("Closures are not keeping pace with new CAPAs." if net > 0 else
                                           "Closure capacity is keeping pace with intake.")),
        confidence=_confidence(int(opened.sum()))))

    # Period-over-period (last two complete periods)
    if len(complete) >= 2:
        a, b = int(complete.iloc[-2]), int(complete.iloc[-1])
        p = poisson_shift(a, b)
        pct = None if a == 0 else _r(100 * (b - a) / a)
        verdict = ("This change is statistically significant (p < 0.05)." if p is not None and p < 0.05 else
                   "This change is within normal random variation for counts this size and should not be read "
                   "as a trend on its own.")
        insights.append(Insight(
            id="pop_change", title=f"{granularity.title()}-over-{granularity} change", value=pct, unit="%",
            window=f"{label(complete.index[-2])} vs {label(complete.index[-1])}",
            interpretation=f"CAPAs opened went from {a} to {b}"
                           + (f" ({pct:+.0f}%). " if pct is not None else ". ") + verdict,
            confidence="Moderate" if p is not None and p < 0.05 else "Low"))

    # Monotonic trend
    if len(complete) >= MIN_TREND_PERIODS:
        s, p = mann_kendall(complete.tolist())
        direction = "increasing" if s > 0 else "decreasing" if s < 0 else "flat"
        slope = np.polyfit(np.arange(len(complete)), complete.to_numpy(float), 1)[0]
        insights.append(Insight(
            id="trend", title="Trend in CAPAs opened", value=_r(slope, 2), unit=f"change in {unit_p} per {granularity}",
            window=f"{label(complete.index[0])} - {label(complete.index[-1])}",
            interpretation=(f"There is a statistically significant {direction} trend in new CAPAs "
                            f"(Mann-Kendall p = {p:.3f}), about {slope:+.2f} per {granularity}."
                            if p < 0.05 else
                            f"No statistically significant trend in new CAPAs (Mann-Kendall p = {p:.2f}); "
                            "the volume is broadly stable with random variation."),
            confidence="High" if p < 0.05 and len(complete) >= 12 else "Moderate"))
    else:
        insights.append(Insight(
            id="trend", title="Trend in CAPAs opened", value=None, unit="", window=window,
            interpretation=f"Only {len(complete)} complete {granularity}s are available; at least "
                           f"{MIN_TREND_PERIODS} are needed for a reliable trend test, so no trend is claimed.",
            confidence="Insufficient data"))

    # First half vs second half shift
    h = len(complete) // 2
    if h >= 2:
        a, b = int(complete.iloc[:h].sum()), int(complete.iloc[-h:].sum())
        p = poisson_shift(a, b)
        insights.append(Insight(
            id="shift", title="First-half vs second-half volume", value=None if a == 0 else _r(100 * (b - a) / a),
            unit="%", window=f"{label(complete.index[0])} - {label(complete.index[-1])}",
            interpretation=(f"The first {h} {granularity}s had {a} new CAPAs and the last {h} had {b}. "
                            + ("This is a statistically significant shift (p = "
                               f"{p:.3f}) and warrants a look at what changed." if p is not None and p < 0.05 else
                               f"The difference is not statistically significant (p = {p:.2f}).")),
            confidence=_confidence(a + b) if p is not None and p < 0.05 else "Low"))

    # Control limit signals
    flagged = [pt.period for pt in series if pt.above_ucl]
    insights.append(Insight(
        id="control", title=f"{granularity.title()}s above control limit", value=len(flagged), unit=granularity + "s",
        window=window,
        interpretation=(f"New CAPAs exceeded the upper control limit of {ucl:.1f} per {granularity} in: "
                        f"{', '.join(flagged)}. These spikes are unlikely to be random and should be investigated."
                        if flagged else
                        f"No {granularity} exceeded the upper control limit of {ucl:.1f}; CAPA intake is in "
                        "statistical control."),
        confidence="Moderate" if len(series) >= MIN_TREND_PERIODS else "Low"))

    # ---- Category breakdowns ----------------------------------------------
    breakdowns: list[Breakdown] = []
    for col, name in DIMENSIONS.items():
        if (df[col] == "Unknown").all():
            limitations.append(f"No usable {name.lower()} data, so the {name.lower()} breakdown is omitted.")
            continue
        unknown = int((df[col] == "Unknown").sum())
        if unknown:
            limitations.append(f"{unknown} CAPAs have no {name.lower()} recorded and are shown as 'Unknown'.")
        rows = []
        for cat, g in df.groupby(col):
            g_days = (g.closure_date - g.initiation_date).dt.days.dropna()
            rows.append(CategoryRow(category=str(cat), count=len(g), pct=_r(100 * len(g) / len(df)),
                                    open=int((g.status == "Open").sum()), overdue=int(overdue_mask[g.index].sum()),
                                    median_days_to_close=_r(g_days.median()) if len(g_days) else None))
        rows.sort(key=lambda r: -r.count)
        cum, pareto = 0, []
        for r in rows:
            if cum >= 80:
                break
            pareto.append(r.category)
            cum += r.pct
        breakdowns.append(Breakdown(field=col, dimension=name, rows=rows, pareto_categories=pareto))
        top = rows[0]
        insights.append(Insight(
            id=f"top_{col}", title=f"Largest {name.lower()} category", value=top.pct, unit="% of CAPAs",
            window=window,
            interpretation=(f"{top.category} accounts for {top.count} CAPAs ({top.pct:.0f}%). "
                            f"{len(pareto)} of {len(rows)} {name.lower()} categories account for at least 80% of "
                            "all CAPAs" + (", so improvement effort can be concentrated there." if len(pareto) < len(rows)
                                           else ".")),
            confidence=_confidence(len(df))))

    n_out = int(df.cycle_time_outlier.sum())
    if n_out:
        insights.append(Insight(
            id="outliers", title="Unusually long closures", value=n_out, unit="CAPAs", window=window,
            interpretation=f"{n_out} CAPAs took far longer than typical to close (beyond Q3 + 3 x IQR); "
                           "review whether these reflect complex investigations or delayed record closure.",
            confidence="Moderate"))

    overdue_items = sorted(
        (OverdueItem(capa_id=r.capa_id, department=r.department, initiation_date=r.initiation_date.date(),
                     target_date=d.date(), days_overdue=int((as_of - d).days))
         for r, d in zip(df[overdue_mask].itertuples(), due[overdue_mask])), key=lambda o: -o.days_overdue)

    if len(df) < 30:
        limitations.append(f"Only {len(df)} CAPAs are in scope; percentages can move sharply with a single record.")

    return AnalysisResult(period_start=start.date(), period_end=end.date(), as_of=as_of.date(),
                          granularity=granularity, kpis=kpis, series=series, centre_line=round(centre, 2),
                          ucl=round(ucl, 2), breakdowns=breakdowns, overdue_items=overdue_items,
                          insights=insights, limitations=limitations)
