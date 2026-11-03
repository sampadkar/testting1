@echo off
REM Generates the CAPA trend report into the "output" folder.
REM Usage: double-click, or run:  run_report.bat capa_2025.csv
cd /d "%~dp0"
set INPUT=%~1
if "%INPUT%"=="" set INPUT=capa_2025.csv
if not exist "%INPUT%" (
  echo Input file "%INPUT%" not found. Save the CAPA CSV in this folder as capa_2025.csv.
  pause
  exit /b 1
)
if not exist output mkdir output
python -m pip install -q -r requirements.txt
python main.py "%INPUT%" -o output
python validate.py "%INPUT%" > output\validation_results.txt
type output\validation_results.txt
echo.
echo Report: %~dp0output\CAPA_Trend_Report.docx
pause
