"""Run the full pipeline:  python main.py [input.xlsx|.csv] [-o output_dir]

Produces in the output folder:
  CAPA_Trend_Report.docx   the report
  analysis.json            every statistic the report cites (traceability)
  cleaning_log.json        every cleaning transformation and assumption
  charts/*.png             the figures embedded in the report
"""
import argparse
from pathlib import Path

from analysis import analyse
from cleaning import clean
from report import build_report, render_docx


def find_input() -> Path:
    files = [p for p in Path.cwd().iterdir()
             if p.suffix.lower() in (".xlsx", ".xls", ".csv") and not p.name.startswith("~$")]
    if not files:
        raise SystemExit("No .xlsx/.csv file found; pass the input path as an argument.")
    return max(files, key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser(description="CAPA trend report generator")
    ap.add_argument("input", nargs="?", type=Path, help="CAPA data file (.xlsx/.csv)")
    ap.add_argument("-o", "--out", type=Path, default=Path("output"))
    args = ap.parse_args()
    src = args.input or find_input()
    args.out.mkdir(parents=True, exist_ok=True)

    data = clean(src)                                      # layer 1 -> CleanedDataset
    print(f"[clean]   {data.rows_in} rows in -> {data.rows_out} valid records ({data.source_file})")
    (args.out / "cleaning_log.json").write_text(
        data.model_dump_json(indent=2, exclude={"records"}), encoding="utf-8")

    result = analyse(data)                                 # layer 2 -> AnalysisResult
    print(f"[analyse] {len(result.insights)} insights, {len(result.series)} {result.granularity}ly periods")
    (args.out / "analysis.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")

    report = build_report(data, result, args.out / "charts")   # layer 3 -> Report
    path = render_docx(report, args.out / "CAPA_Trend_Report.docx")
    print(f"[report]  {path}")


if __name__ == "__main__":
    main()
