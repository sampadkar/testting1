"""Layer 1 - data cleaning.

Input : path to a CAPA extract (.xlsx / .xls / .csv)
Output: models.CleanedDataset (validated records + transformation log)

Column names in CAPA exports vary between systems, so headers are matched
against an alias list instead of being hard-coded. Every change made to the
data is appended to the log with the number of rows it touched and why.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from models import CapaRecord, CleanedDataset, CleaningStep

# Canonical field -> header aliases (compared after lower-casing and stripping
# non-alphanumerics). Order matters: specific date fields are matched before
# the generic initiation date so "Closure Date" is never taken as "Date".
ALIASES: dict[str, list[str]] = {
    "capa_id": ["capano", "capanumber", "capaid", "caparefno", "capareference", "referenceno",
                "refno", "recordid", "recordno", "id", "capa", "no"],
    # Original commitment date. TrackWise exports "proposed_closure_date" (original) and
    # "due_date" (current, after extensions) - both are kept, see revised_due_date.
    "target_date": ["targetdate", "proposedclosuredate", "targetclosuredate", "plannedclosuredate",
                    "targetcompletiondate", "originalduedate", "commitmentdate", "duedate"],
    "revised_due_date": ["duedate", "revisedduedate", "extendedduedate", "currentduedate"],
    "closure_date": ["closuredate", "closeddate", "dateclosed", "datecompleted", "actualclosuredate",
                     "completiondate", "actualcompletiondate", "dateofclosure"],
    "initiation_date": ["initiationdate", "initiateddate", "dateofinitiation", "opendate", "openeddate",
                        "dateopened", "createddate", "creationdate", "raiseddate", "reporteddate",
                        "loggeddate", "capadate", "date"],
    "status": ["status", "capastatus", "currentstatus", "state", "recordstatus"],
    "site": ["site", "sitename", "plant", "location", "facility", "unit"],
    "department": ["department", "dept", "respdept", "responsibledepartment", "respdepartment",
                   "owningdepartment", "function", "area", "section"],
    "source": ["source", "capasource", "sourceofcapa", "originqms", "origin", "sourcetype"],
    "capa_type": ["capatype", "actiontype", "type"],
    "root_cause": ["rootcausecategory", "rootcause", "rccategory", "causecategory",
                   "capaclassification", "classification", "category"],
    "priority": ["priority", "criticality", "severity", "risk", "risklevel"],
    "extension_count": ["extcounter", "extensioncount", "noofextensions", "extensions"],
    "extract_date": ["silverloadtime", "loadtime", "extractdate", "snapshotdate", "reportdate"],
}
TEXT_FIELDS = ["site", "department", "source", "capa_type", "root_cause", "priority"]
DATE_FIELDS = ["initiation_date", "target_date", "revised_due_date", "closure_date"]
NO_FUTURE = {"initiation_date", "closure_date"}   # target/due dates may legitimately be in the future
NULL_TOKENS = {"", "na", "n/a", "nan", "none", "null", "-", "--", "nil", "tbd", "nat"}

ASSUMPTIONS = [
    "Column headers are matched to canonical CAPA fields by name aliases; unmatched columns are ignored.",
    "Ambiguous text dates (e.g. 03/04/2025) are read day-first, matching Indian/EU convention; "
    "ISO dates (YYYY-MM-DD) and Excel serial numbers are read unambiguously.",
    "Initiation/closure dates in the future, or any date before 2000, are treated as entry errors and blanked; "
    "target and due dates may be in the future.",
    "The original target is the proposed closure date; the due date (after extensions) is the revised target. "
    "Overdue status is judged against the revised due date.",
    "Exports repeat a CAPA once per child record / related customer; one row per CAPA ID is kept (the last). "
    "Where the classification differs between those rows, the last value is used.",
    "A closure date is treated as authoritative: a CAPA with a closure date is counted as Closed.",
    "A CAPA with no status is Closed if it has a closure date, otherwise Open.",
    "Category labels differing only in case, spacing or punctuation are merged into the most frequent spelling.",
    "Cycle-time outliers (beyond Q3 + 3 x IQR) are flagged, not removed; medians are used so they do not distort results.",
]


def _norm(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def map_columns(columns) -> dict[str, str]:
    """Return {canonical_field: original_header}. Exact alias match first, then containment."""
    normed = {c: _norm(c) for c in columns}
    mapping: dict[str, str] = {}
    for field, aliases in ALIASES.items():
        free = {c: n for c, n in normed.items() if c not in mapping.values()}
        hit = next((c for a in aliases for c, n in free.items() if n == a), None)
        if hit is None:  # e.g. "CAPA Initiation Date (dd/mm/yyyy)"
            hit = next((c for a in aliases if len(a) >= 5 for c, n in free.items() if a in n), None)
        if hit is not None:
            mapping[field] = hit
    return mapping


def load_table(path: Path) -> tuple[pd.DataFrame, str]:
    """Load the sheet/file that best matches a CAPA register.

    Handles title rows above the header by scanning the first 15 rows for the
    row that maps an initiation date column.
    """
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    else:
        sheets = {path.stem: pd.read_csv(path, header=None, dtype=object, encoding_errors="replace")}

    best, best_score = None, -1
    for name, raw in sheets.items():
        for h in range(min(15, len(raw))):
            header = [str(x).strip() if pd.notna(x) else f"col_{i}" for i, x in enumerate(raw.iloc[h])]
            seen: dict[str, int] = {}
            for i, name_ in enumerate(header):            # de-duplicate repeated headers
                seen[name_] = seen.get(name_, 0) + 1
                if seen[name_] > 1:
                    header[i] = f"{name_}_{seen[name_]}"
            m = map_columns(header)
            if "initiation_date" not in m:
                continue
            score = len(m) * 10_000 + len(raw)
            if score > best_score:
                body = raw.iloc[h + 1:].reset_index(drop=True)
                body.columns = header
                best, best_score = (body, f"{path.name} [sheet: {name}, header row {h + 1}]"), score
            break
    if best is None:
        raise ValueError(f"No sheet in {path.name} has a recognisable CAPA initiation-date column. "
                         f"Add its header to ALIASES['initiation_date'] in cleaning.py.")
    return best


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse a mixed column of datetimes, Excel serials, ISO strings and day-first strings."""
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    is_dt = s.map(lambda v: isinstance(v, (pd.Timestamp, date)))
    if is_dt.any():
        out[is_dt] = pd.to_datetime(s[is_dt].astype(str), errors="coerce")
    num = pd.to_numeric(s.where(~is_dt), errors="coerce")
    serial = num.between(20_000, 80_000)                       # 1954 .. 2119 as Excel serials
    if serial.any():
        out[serial] = pd.to_datetime(num[serial], unit="D", origin="1899-12-30")
    text = s.where(~is_dt & ~serial).dropna().astype(str).str.strip()
    iso = text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}")
    if iso.any():
        out[iso[iso].index] = pd.to_datetime(text[iso].str.split(r"[ T]", regex=True).str[0],
                                                errors="coerce", format="ISO8601")
    other = text[~iso]
    if len(other):
        out[other.index] = pd.to_datetime(other, errors="coerce", dayfirst=True, format="mixed")
    return out.dt.normalize()


def _status(v) -> str | None:
    n = _norm(v) if pd.notna(v) else ""
    if not n:
        return None
    # "Pending QA Closure" / "Awaiting ..." are still open, so check open-type words first.
    if any(k in n for k in ("pending", "awaiting", "progress", "wip", "ongoing", "investigation", "opened")):
        return "Open"
    if any(k in n for k in ("cancel", "void", "withdraw", "reject", "drop", "notrequired")):
        return "Cancelled"
    if any(k in n for k in ("clos", "complet", "done", "effective", "verified")):
        return "Closed"
    return "Open"


def _py(v):
    if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.date()
    if isinstance(v, np.generic):          # numpy bool/int -> plain Python for Pydantic
        return v.item()
    return v


def clean(path: str | Path) -> CleanedDataset:
    path = Path(path)
    log: list[CleaningStep] = []

    def note(step, rows, reason, column=None):
        log.append(CleaningStep(step=step, column=column, rows_affected=int(rows), reason=reason))

    raw, origin = load_table(path)
    rows_in = len(raw)
    mapping = map_columns(raw.columns)
    df = raw[list(mapping.values())].rename(columns={v: k for k, v in mapping.items()}).copy()
    note("Map columns", 0, "Matched headers: " + "; ".join(f"{v} -> {k}" for k, v in mapping.items()))
    ignored = [c for c in raw.columns if c not in mapping.values()]
    if ignored:
        note("Ignore columns", 0, f"{len(ignored)} column(s) not used by the analysis: {', '.join(map(str, ignored))}")
    for f in ALIASES:
        if f not in df.columns:
            df[f] = pd.NA
            if f not in ("status", "extract_date", "revised_due_date", "extension_count"):
                note("Missing column", 0, f"No '{f}' column found; field left blank / 'Unknown'.", f)

    # 1. Empty rows
    before = len(df)
    df = df.dropna(how="all", subset=[c for c in df.columns if c != "capa_id"] or None)
    note("Drop empty rows", before - len(df), "Rows with no data (blank lines, footer rows).")

    # 2. Text normalisation and null tokens
    for c in ["capa_id", "status"] + TEXT_FIELDS:
        s = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        nulls = s.str.lower().isin(NULL_TOKENS) & s.notna()
        df[c] = s.mask(nulls)
        if nulls.any():
            note("Null tokens", nulls.sum(), "Placeholder values such as 'NA', '-', 'nil' treated as missing.", c)

    # 3. Dates
    today = pd.Timestamp(date.today())
    for c in DATE_FIELDS:
        orig = df[c]
        parsed = _parse_dates(orig)
        bad = orig.notna() & parsed.isna()
        if bad.any():
            note("Malformed dates", bad.sum(), "Values that could not be parsed as dates were blanked.", c)
        implausible = parsed < pd.Timestamp("2000-01-01")
        if c in NO_FUTURE:
            implausible |= parsed > today
        if implausible.any():
            note("Implausible dates", implausible.sum(), "Future (for initiation/closure) or pre-2000 dates blanked.", c)
        df[c] = parsed.mask(implausible)

    # Extract/snapshot date -> "as of" date for open/overdue status (not a per-record field)
    extract = _parse_dates(df["extract_date"]).max()
    as_of = extract.date() if pd.notna(extract) else None
    if as_of:
        note("Status date", 0, f"Open/overdue status assessed as of the extract date {as_of:%d %b %Y}.", "extract_date")

    ext = pd.to_numeric(df["extension_count"], errors="coerce")
    bad = df["extension_count"].notna() & ext.isna()
    if bad.any():
        note("Non-numeric extensions", bad.sum(), "Extension counts that were not numbers were blanked.", "extension_count")
    df["extension_count"] = ext.astype("Int64")

    before = len(df)
    df = df[df["initiation_date"].notna()].copy()
    note("Drop undated rows", before - len(df), "Initiation date is required to place a CAPA on the timeline.")

    bad = df["closure_date"] < df["initiation_date"]
    df.loc[bad, "closure_date"] = pd.NaT
    note("Closure before initiation", bad.sum(), "Closure date earlier than initiation date is impossible; blanked.",
         "closure_date")
    for c in ("target_date", "revised_due_date"):
        bad = df[c] < df["initiation_date"]
        df.loc[bad, c] = pd.NaT
        note("Target before initiation", bad.sum(), "Target/due date earlier than initiation date; blanked.", c)

    # 4. IDs and duplicates
    missing_id = df["capa_id"].isna()
    df.loc[missing_id, "capa_id"] = [f"ROW-{i + 1}" for i in df.index[missing_id.to_numpy()]]
    note("Missing CAPA ID", missing_id.sum(), "Synthetic ID 'ROW-n' assigned so the record can be counted.", "capa_id")
    before = len(df)
    df = df.drop_duplicates()
    note("Exact duplicates", before - len(df), "Identical rows removed.")
    before = len(df)
    df = df.drop_duplicates(subset="capa_id", keep="last")
    note("Duplicate CAPA IDs", before - len(df),
         "Same CAPA ID repeated (child records / related customers); one row per CAPA kept.", "capa_id")

    # 5. Categorical labels
    for c in TEXT_FIELDS:
        s = df[c]
        keys = s.dropna().map(_norm)
        canon = s.dropna().groupby(keys).agg(lambda x: x.value_counts().index[0])
        new = s.map(lambda v: canon.get(_norm(v), v) if pd.notna(v) else v)
        changed = (new != s) & s.notna()
        if changed.any():
            note("Unify labels", changed.sum(), "Variants differing in case/spacing/punctuation merged.", c)
        missing = new.isna()
        if missing.any() and mapping.get(c):
            note("Missing category", missing.sum(), "Blank values set to 'Unknown'.", c)
        df[c] = new.fillna("Unknown")

    # 6. Status
    raw_status = df["status"]
    status = raw_status.map(_status)
    derived = status.isna()
    status = status.where(~derived, df["closure_date"].notna().map({True: "Closed", False: "Open"}))
    if derived.any():
        note("Derive status", derived.sum(), "Status missing; derived from presence of closure date.", "status")
    note("Standardise status", raw_status.notna().sum(),
         "Free-text statuses mapped to Open / Closed / Cancelled.", "status")
    fix = (status == "Open") & df["closure_date"].notna()
    status[fix] = "Closed"
    note("Status vs closure date", fix.sum(), "Marked Open but has a closure date; set to Closed.", "status")
    no_date = (status == "Closed") & df["closure_date"].isna()
    note("Closed without date", no_date.sum(), "Kept as Closed; excluded from cycle-time statistics.", "closure_date")
    df["status"] = status

    # 7. Cycle-time outliers (flag only)
    ct = (df["closure_date"] - df["initiation_date"]).dt.days
    df["cycle_time_outlier"] = False
    if ct.notna().sum() >= 8:
        q1, q3 = ct.quantile([0.25, 0.75])
        upper = q3 + 3 * (q3 - q1)
        df["cycle_time_outlier"] = (ct > upper).fillna(False)
        note("Flag outliers", df["cycle_time_outlier"].sum(),
             f"Cycle time above {upper:.0f} days (Q3 + 3 x IQR) flagged; retained in counts.", "closure_date")

    # 8. Row-level validation at the boundary
    records, rejected = [], 0
    for row in df[list(CapaRecord.model_fields)].to_dict("records"):
        try:
            records.append(CapaRecord(**{k: _py(v) for k, v in row.items()}))
        except ValidationError:
            rejected += 1
    note("Schema validation", rejected, "Rows failing the CapaRecord model were rejected.")

    return CleanedDataset(source_file=origin, rows_in=rows_in, records=records, as_of_date=as_of,
                          column_mapping=mapping, log=[s for s in log if s.rows_affected or s.step in
                                                       ("Map columns", "Ignore columns", "Missing column")],
                          assumptions=ASSUMPTIONS)
