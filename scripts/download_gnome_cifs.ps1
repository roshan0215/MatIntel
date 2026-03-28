$ErrorActionPreference = "Stop"

$targetDir = Join-Path $PWD "data"
$zipPath = Join-Path $targetDir "by_id.zip"
$extractDir = Join-Path $targetDir "cifs"
$url = "https://storage.googleapis.com/gdm_materials_discovery/gnome_data/by_id.zip"

if (-not (Test-Path $targetDir)) {
    New-Item -ItemType Directory -Path $targetDir | Out-Null
}

if (-not (Test-Path $zipPath)) {
    Write-Host "Downloading GNoME by_id CIF archive..." -ForegroundColor Cyan
    Write-Host "Source: $url"
    Write-Host "Target: $zipPath"
    $client = New-Object System.Net.WebClient
    $client.DownloadFile($url, $zipPath)
    $sizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
    Write-Host "Download complete ($sizeMB MB)." -ForegroundColor Green
} else {
    Write-Host "Archive already exists: $zipPath" -ForegroundColor Yellow
}

if (-not (Test-Path $extractDir)) {
    New-Item -ItemType Directory -Path $extractDir | Out-Null
}

Write-Host "Extracting archive to $extractDir ..." -ForegroundColor Cyan
Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force

$cifCount = (Get-ChildItem $extractDir -Filter *.cif -Recurse | Measure-Object).Count
Write-Host "Extraction complete. CIF files found: $cifCount" -ForegroundColor Green
