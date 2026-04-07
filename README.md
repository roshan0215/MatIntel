# MatIntel (Windows-first)

End-to-end materials screening platform with provenance-aware ranking, viability constraints, and CLscore synthesizability inference.

## What is new

- Scoring system expanded from 31 to 71 active categories in code.
- Category registry now reflects the new full scoring spec in `MATINTEL_README_FULL.md`.
- Top-10 exports now produce 710 rows each (71 categories x 10 candidates).
- Provenance-aware ranking includes both Experimental and Synthesized records.
- Viability and CLscore multiplier consistency hardened in app loading and detail breakdown.

## Current capability summary

- Multi-source ingestion: base dataset plus synthesized references.
- 71 application scorers across energy, catalysis, electronics, magnetics, structural, quantum, biomedical, environmental, and advanced domains.
- Viability scoring from:
	- cost_score
	- abundance_score
	- supply_risk
	- element-level filters (radioactive and rare-earth penalties)
	- clscore_multiplier
- CLscore integration via KAIST Synthesizability-PU-CGCNN pipeline.
- Streamlit dashboard for filtering, provenance slicing, and top-candidate inspection.

## Quick start (PowerShell)

1. Setup environment and dependencies:

```powershell
./scripts/setup_windows.ps1
```

2. Optional: fetch GNoME CSV:

```powershell
./scripts/download_gnome_csv.ps1
```

3. Run pipeline:

```powershell
./scripts/run_pipeline.ps1
```

4. Launch website:

```powershell
./scripts/run_app.ps1
```

## CLscore setup and usage

Setup:

```powershell
./scripts/setup_clscore.ps1
```

Run CLscore on ranked candidates:

```powershell
.\.venv\Scripts\python.exe run_clscore.py --app battery --top-n 1000 --cif-dir data/cifs
```

Batch/all-material CLscore flow is supported through `run_clscore_all.py` and cache hydration in app loading.

## Experimental reference build

```powershell
setx MATINTEL_MP_API_KEY "<your_mp_key>"
.\.venv\Scripts\python.exe scripts/build_experimental_reference.py
```

This generates:

- `data/processed/experimental_compounds.csv`

## Rescore everything after scoring updates

Use this workflow when scoring logic changes:

1. Recompute scores + viability on processed datasets.
2. Regenerate top-10 files.

Top-10 regeneration command:

```powershell
.\.venv\Scripts\python.exe scripts/rebuild_top10.py
```

Current top-10 outputs:

- `data/processed/top10_per_category.csv` (weighted by viability)
- `data/processed/top10_per_category_raw_score.csv` (raw application score)

## Key output files

- `data/processed/working_dataset.csv`
- `data/processed/featured_dataset.csv`
- `data/processed/scored_dataset.csv`
- `data/processed/experimental_compounds.csv`
- `data/processed/top10_per_category.csv`
- `data/processed/top10_per_category_raw_score.csv`
- `logs/pipeline.log`

## Documentation map

- `README.md` (this file): operational quick guide.
- `README_COMPREHENSIVE.md`: implementation-level reference and provenance analysis.
- `MATINTEL_README_FULL.md`: full expanded category and scoring specification used for recent scorer implementation.

## Project layout

- `app.py`: Streamlit interface and data loading.
- `src/matintel/scoring.py`: 71-category scoring logic.
- `src/matintel/viability.py`: viability and penalty multipliers.
- `src/matintel/pipeline.py`: data pipeline orchestration.
- `scripts/rebuild_top10.py`: regenerate top-10 exports.
- `scripts/run_app.ps1`: launch website.

## Notes

- Windows-first setup using PowerShell scripts and `.venv\Scripts\python.exe`.
- Very large datasets can take significant time for full rescoring and CLscore inference.
- For CLscore, missing/invalid CIFs can yield `clscore = -1` before cache backfill.
