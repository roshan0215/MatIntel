# MatIntel (Windows-first)

End-to-end materials screening MVP with:
- Data ingestion and fallback demo dataset
- Feature engineering from composition formulas
- Application scoring across five domains
- Viability scoring (cost, abundance, supply risk)
- Streamlit UI with interactive filtering and AI summaries

## Quick Start (Windows PowerShell)

1. Setup environment and dependencies:

```powershell
./scripts/setup_windows.ps1
```

2. (Optional but recommended) Pull the official GNoME CSV directly:

```powershell
./scripts/download_gnome_csv.ps1
```

3. Run the full backend pipeline:

```powershell
./scripts/run_pipeline.ps1
```

4. Launch UI:

```powershell
./scripts/run_app.ps1
```

## CLscore (KAIST Synthesizability-PU-CGCNN)

### Setup

```powershell
./scripts/setup_clscore.ps1
```

This script will:
- clone `kaist-amsg/Synthesizability-PU-CGCNN` into `external/`
- install `torch` and `scikit-learn` into your existing `.venv`
- verify pretrained checkpoints exist and can be loaded on CPU

### Run on top candidates

```powershell
.\.venv\Scripts\python.exe run_clscore.py --app battery --top-n 1000 --cif-dir data/cifs
```

Notes:
- `run_clscore.py` ranks by `score_{app} * viability`, scores top-N, and merges `clscore` back into `data/processed/scored_dataset.csv`.
- Resume-safe batch output is written to `data/processed/clscore_results.csv`.
- Failed structures are logged in `data/processed/failed_clscore.csv`.

### Known compatibility notes

- KAIST code is from an older stack; this project runs it with modern `Python 3.13 + torch 2.x` on CPU.
- Legacy checkpoint loading may emit deprecation warnings from `torch`/`numpy`; these are non-fatal in this setup.
- Inference is CPU-only and slow for very large datasets; score top-N candidates per application first.

### GNoME-specific caveats

- Missing CIF files for some IDs will return `clscore = -1` (unknown) and be logged.
- Very large unit cells can make graph generation and inference slow.
- Unusual/disordered structures may fail parsing or neighbor graph construction.
- Elements with atomic number beyond common embedding range are clipped to a fallback feature index.

## Data behavior

- If `data/raw/stable_materials_summary.csv` exists, the pipeline uses it.
- If missing, the pipeline auto-generates a realistic demo dataset so everything still runs.
- To fetch the official dataset file directly, run `./scripts/download_gnome_csv.ps1`.

Expected outputs:
- `data/processed/working_dataset.csv`
- `data/processed/featured_dataset.csv`
- `data/processed/scored_dataset.csv`
- `logs/pipeline.log`

## Build Experimental Reference Dataset

To fetch experimentally sourced compounds (MP + optional matminer/JARVIS enrichment) and align to your
current `scored_dataset.csv` schema for direct concatenation:

```powershell
setx MATINTEL_MP_API_KEY "<your_mp_key>"
.\.venv\Scripts\python.exe scripts/build_experimental_reference.py
```

Output:
- `data/processed/experimental_compounds.csv`

## Optional real-data expansion

Replace demo data with real data by placing your CSV at:

`data/raw/stable_materials_summary.csv`

Minimum required columns:
- `MaterialId`
- `Reduced Formula`
- `Bandgap`
- `Formation Energy Per Atom`
- `Decomposition Energy Per Atom`
- `NSites`
- `Dimensionality Cheon`
- `Crystal System`

## Optional AI explanation support

Set environment variable before running app:

```powershell
$env:ANTHROPIC_API_KEY = "your_key_here"
```

If key is missing, app uses deterministic local fallback summaries.

## Project layout

- `app.py`: Streamlit interface
- `src/matintel/pipeline.py`: pipeline entry point
- `src/matintel/features.py`: composition feature engineering
- `src/matintel/scoring.py`: domain scoring logic
- `src/matintel/viability.py`: cost/abundance/supply scoring
- `src/matintel/explanations.py`: AI explanation adapter
- `scripts/*.ps1`: Windows setup and run scripts

## Notes

This build is intentionally Windows-first (PowerShell scripts, `.venv\\Scripts\\python.exe`, path handling via `pathlib`).
