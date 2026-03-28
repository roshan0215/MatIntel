$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$PWD\src"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

$streamlitDir = Join-Path $env:USERPROFILE ".streamlit"
if (-not (Test-Path $streamlitDir)) {
    New-Item -ItemType Directory -Path $streamlitDir | Out-Null
}
$credsPath = Join-Path $streamlitDir "credentials.toml"
if (-not (Test-Path $credsPath)) {
    @"
[general]
email = ""
"@ | Set-Content -Path $credsPath -Encoding UTF8
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Missing .venv. Run scripts/setup_windows.ps1 first." -ForegroundColor Red
    exit 1
}

& .\.venv\Scripts\python.exe -m streamlit run app.py
