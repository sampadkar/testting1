"""Pydantic contracts between the three layers.

    cleaning.py  --CleanedDataset-->  analysis.py  --AnalysisResult-->  report.py  --Report--> .docx

Every handoff is a validated model, so a broken shape fails at the boundary
where it was produced, not at the end of the pipeline.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

STATUSES = ("Open", "Closed", "Cancelled")


# --------------------------------------------------------------------------- #
# Layer 1 -> 2 : cleaned data
# --------------------------------------------------------------------------- #
class CleaningStep(BaseModel):
    step: str
    column: Optional[str] = None
    rows_affected: int = Field(ge=0)
    reason: str


class CapaRecord(BaseModel):
    capa_id: str = Field(min_length=1)
    initiation_date: date
    target_date: Optional[date] = None          # original commitment
    revised_due_date: Optional[date] = None     # after extensions
    closure_date: Optional[date] = None
    status: Literal["Open", "Closed", "Cancelled"]
    site: str = "Unknown"
    department: str = "Unknown"
    source: str = "Unknown"
    capa_type: str = "Unknown"
    root_cause: str = "Unknown"
    priority: str = "Unknown"
    extension_count: Optional[int] = Field(default=None, ge=0)
    cycle_time_outlier: bool = False

    @model_validator(mode="after")
    def _dates_consistent(self):
        if self.closure_date and self.closure_date < self.initiation_date:
            raise ValueError("closure_date is before initiation_date")
        return self


class CleanedDataset(BaseModel):
    source_file: str
    rows_in: int = Field(ge=0)
    as_of_date: Optional[date] = None           # extract/snapshot date, if the file has one
    records: list[CapaRecord] = Field(min_length=1)
    column_mapping: dict[str, str]          # canonical field -> original header
    log: list[CleaningStep]
    assumptions: list[str]

    @property
    def rows_out(self) -> int:
        return len(self.records)


# --------------------------------------------------------------------------- #
# Layer 2 -> 3 : statistical insight
# --------------------------------------------------------------------------- #
Confidence = Literal["High", "Moderate", "Low", "Insufficient data"]


class KPIs(BaseModel):
    total: int = Field(ge=0)
    open: int = Field(ge=0)
    closed: int = Field(ge=0)
    cancelled: int = Field(ge=0)
    overdue_open: int = Field(ge=0)
    closure_rate_pct: float = Field(ge=0, le=100)
    on_time_closure_pct: Optional[float] = Field(default=None, ge=0, le=100)          # vs original target
    on_time_revised_pct: Optional[float] = Field(default=None, ge=0, le=100)          # vs revised due date
    extended_pct: Optional[float] = Field(default=None, ge=0, le=100)                 # CAPAs extended >= once
    median_days_to_close: Optional[float] = None
    p90_days_to_close: Optional[float] = None
    mean_age_open_days: Optional[float] = None

    @model_validator(mode="after")
    def _counts_add_up(self):
        if self.open + self.closed + self.cancelled != self.total:
            raise ValueError("open + closed + cancelled must equal total")
        return self


class PeriodPoint(BaseModel):
    period: str                      # e.g. "2025-04" or "2025-W15"
    opened: int = Field(ge=0)
    closed: int = Field(ge=0)
    backlog: int                     # cumulative opened - closed within the window
    rolling_mean_opened: Optional[float] = None
    above_ucl: bool = False


class CategoryRow(BaseModel):
    category: str
    count: int = Field(ge=0)
    pct: float = Field(ge=0, le=100)
    open: int = Field(ge=0)
    overdue: int = Field(ge=0)
    median_days_to_close: Optional[float] = None


class Breakdown(BaseModel):
    field: str = Field(pattern=r"^[a-z_]+$")   # CapaRecord field, e.g. "department"
    dimension: str                   # display name, e.g. "Department"
    rows: list[CategoryRow]
    pareto_categories: list[str]     # smallest set covering >= 80 % of CAPAs


class OverdueItem(BaseModel):
    capa_id: str
    department: str
    initiation_date: date
    target_date: date
    days_overdue: int = Field(gt=0)


class Insight(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    value: Optional[float]           # the supporting number (None if not computable)
    unit: str
    window: str                      # time window the number covers
    interpretation: str
    confidence: Confidence


class AnalysisResult(BaseModel):
    period_start: date
    period_end: date
    as_of: date
    granularity: Literal["week", "month"]
    kpis: KPIs
    series: list[PeriodPoint]
    centre_line: float               # c-chart mean of opened per period
    ucl: float                       # c-chart upper control limit
    breakdowns: list[Breakdown]
    overdue_items: list[OverdueItem]
    insights: list[Insight]
    limitations: list[str]

    def insight(self, id_: str) -> Optional[Insight]:
        return next((i for i in self.insights if i.id == id_), None)

    @model_validator(mode="after")
    def _period_order(self):
        if self.period_end < self.period_start:
            raise ValueError("period_end before period_start")
        return self

    def facts(self) -> dict[str, str]:
        """Flat registry of every number/text the narrative may cite.

        The report layer may only reference keys from this dict, which is what
        makes every narrative number traceable to a statistical-layer field.
        """
        k = self.kpis
        f = {
            "period_start": self.period_start.strftime("%d %b %Y"),
            "period_end": self.period_end.strftime("%d %b %Y"),
            "as_of": self.as_of.strftime("%d %b %Y"),
            "granularity": self.granularity,
            "n_periods": str(len(self.series)),
            "centre_line": f"{self.centre_line:.1f}",
            "ucl": f"{self.ucl:.1f}",
        }
        for name, val in k.model_dump().items():
            f[f"kpi.{name}"] = _fmt(val)
        for ins in self.insights:
            f[f"insight.{ins.id}.value"] = _fmt(ins.value)
            f[f"insight.{ins.id}.interpretation"] = ins.interpretation
            f[f"insight.{ins.id}.window"] = ins.window
            f[f"insight.{ins.id}.confidence"] = ins.confidence
        for b in self.breakdowns:
            if b.rows:
                f[f"top.{b.field}"] = b.rows[0].category
                f[f"top.{b.field}.pct"] = _fmt(b.rows[0].pct)
        for i, text in enumerate(self.limitations):
            f[f"limitation.{i}"] = text
        return f


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.0f}" if v.is_integer() else f"{v:.1f}"
    return str(v)


# --------------------------------------------------------------------------- #
# Layer 3 : report
# --------------------------------------------------------------------------- #
_PLACEHOLDER = re.compile(r"\{([a-zA-Z0-9_.]+)\}")


class Paragraph(BaseModel):
    """Narrative text written as a template, e.g. "{kpi.total} CAPAs were raised".

    Hard-coded digits are rejected, so a number can only enter the narrative
    through a placeholder that resolves to an AnalysisResult field.
    """
    template: str
    text: str = ""

    @field_validator("template")
    @classmethod
    def _no_literal_numbers(cls, v: str) -> str:
        if re.search(r"\d", _PLACEHOLDER.sub("", v)):
            raise ValueError(f"literal number in narrative (use a placeholder): {v!r}")
        return v

    def render(self, facts: dict[str, str]) -> "Paragraph":
        missing = [m for m in _PLACEHOLDER.findall(self.template) if m not in facts]
        if missing:
            raise ValueError(f"untraceable placeholders {missing} in {self.template!r}")
        return self.model_copy(update={"text": _PLACEHOLDER.sub(lambda m: facts[m.group(1)], self.template)})


class Table(BaseModel):
    headers: list[str] = Field(min_length=1)
    rows: list[list[str]]

    @model_validator(mode="after")
    def _rectangular(self):
        for r in self.rows:
            if len(r) != len(self.headers):
                raise ValueError("table row width does not match headers")
        return self


class Section(BaseModel):
    heading: str
    paragraphs: list[Paragraph] = []
    bullets: list[Paragraph] = []
    tables: list[Table] = []
    charts: list[str] = []           # PNG paths

    @model_validator(mode="after")
    def _has_content(self):
        if not (self.paragraphs or self.bullets or self.tables or self.charts):
            raise ValueError(f"section '{self.heading}' is empty")
        return self


REQUIRED_SECTIONS = (
    "1. Purpose and Scope",
    "2. Executive Summary",
    "3. Data Sources and Preparation",
    "4. CAPA Status Overview",
    "5. Trend Analysis",
    "6. Category Analysis",
    "7. Timeliness and Overdue CAPAs",
    "8. Key Findings",
    "9. Conclusions and Recommendations",
    "10. Limitations and Assumptions",
)


class Report(BaseModel):
    title: str
    subtitle: str
    sections: list[Section]

    @model_validator(mode="after")
    def _structure(self):
        headings = [s.heading for s in self.sections]
        if headings != list(REQUIRED_SECTIONS):
            raise ValueError(f"report sections must be exactly {REQUIRED_SECTIONS}, got {headings}")
        for s in self.sections:
            for p in s.paragraphs + s.bullets:
                if not p.text:
                    raise ValueError(f"unrendered paragraph in '{s.heading}'")
        return self
