$ErrorActionPreference = "Stop"

$repoRoot = $PWD
$externalDir = Join-Path $repoRoot "external"
$clscoreRepo = Join-Path $externalDir "Synthesizability-PU-CGCNN"

if (-not (Test-Path $externalDir)) {
    New-Item -ItemType Directory -Path $externalDir | Out-Null
}

if (-not (Test-Path $clscoreRepo)) {
    Write-Host "Cloning Synthesizability-PU-CGCNN..." -ForegroundColor Cyan
    git clone https://github.com/kaist-amsg/Synthesizability-PU-CGCNN $clscoreRepo
} else {
    Write-Host "Repo already present: $clscoreRepo" -ForegroundColor Yellow
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Missing .venv. Run scripts/setup_windows.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Installing CLscore dependencies into existing venv..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m pip install torch scikit-learn

$modelDir = Join-Path $clscoreRepo "trained_models"
$weightsCount = 0
if (Test-Path $modelDir) {
    $weightsCount = (Get-ChildItem $modelDir -Filter "checkpoint_bag_*.pth.tar" | Measure-Object).Count
}

Write-Host "Detected checkpoint files: $weightsCount"
if ($weightsCount -lt 1) {
    Write-Host "No pretrained checkpoints found in trained_models." -ForegroundColor Red
    exit 1
}

Write-Host "Validating first checkpoint load on CPU..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -c "import torch; ck=torch.load(r'external/Synthesizability-PU-CGCNN/trained_models/checkpoint_bag_1.pth.tar', map_location='cpu', weights_only=False); print('checkpoint_ok=', 'state_dict' in ck)"

Write-Host "CLscore setup complete." -ForegroundColor Green
Write-Host "Note: KAIST codebase is older; Python 3.13 + torch 2.x works in this project, but may emit deprecation warnings when loading legacy checkpoints." -ForegroundColor Yellow
