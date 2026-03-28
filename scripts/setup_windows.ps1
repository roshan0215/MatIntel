$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    py -3 -m venv .venv
}

Write-Host "Upgrading pip..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host "Installing requirements..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Setup complete." -ForegroundColor Green
