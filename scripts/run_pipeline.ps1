$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$PWD\src"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Missing .venv. Run scripts/setup_windows.ps1 first." -ForegroundColor Red
    exit 1
}

& .\.venv\Scripts\python.exe -m matintel.pipeline

if ($LASTEXITCODE -eq 0) {
    Write-Host "Pipeline finished. Output at data/processed/scored_dataset.csv" -ForegroundColor Green
}
