$ErrorActionPreference = "Stop"

$targetDir = Join-Path $PWD "data\raw"
$targetFile = Join-Path $targetDir "stable_materials_summary.csv"
$url = "https://storage.googleapis.com/gdm_materials_discovery/gnome_data/stable_materials_summary.csv"

if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir | Out-Null
}

Write-Host "Downloading GNoME CSV from official public bucket..." -ForegroundColor Cyan
Write-Host "Source: $url"
Write-Host "Target: $targetFile"

$client = New-Object System.Net.WebClient
$client.DownloadFile($url, $targetFile)

$sizeMB = [math]::Round((Get-Item $targetFile).Length / 1MB, 2)
Write-Host "Download complete ($sizeMB MB)." -ForegroundColor Green