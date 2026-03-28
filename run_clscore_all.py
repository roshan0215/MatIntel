from __future__ import annotations

import argparse
from pathlib import Path
import os

import pandas as pd

from src.matintel.clscore import batch_clscore
from src.matintel.config import SCORED_CSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLscore for all materials and save standalone results")
    parser.add_argument("--input-csv", default=str(SCORED_CSV), help="Path to scored dataset CSV")
    parser.add_argument("--cif-dir", default="data/cifs", help="Directory containing <MaterialId>.cif files")
    parser.add_argument(
        "--output-csv",
        default="data/processed/clscore_all_results.csv",
        help="Standalone CLscore output with MaterialId and clscore",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--recompute-unknown",
        action="store_true",
        help="Recompute existing rows with clscore == -1 if CIFs are now available",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="Optional speed mode: use only first N bagging checkpoints (0 = all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, usecols=["MaterialId"])
    material_ids = df["MaterialId"].astype(str).tolist()

    if args.max_models > 0:
        os.environ["MATINTEL_CLSCORE_MAX_MODELS"] = str(args.max_models)

    results = batch_clscore(
        material_ids=material_ids,
        cif_dir=args.cif_dir,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        recompute_unknown=args.recompute_unknown,
    )

    valid = results[results["clscore"] >= 0]
    mean_val = float(valid["clscore"].mean()) if len(valid) else -1.0
    high_conf = float((valid["clscore"] > 0.5).mean() * 100) if len(valid) else 0.0

    print(f"Saved standalone CLscore file: {args.output_csv}")
    print(f"Rows scored: {len(results):,}")
    print(f"Mean CLscore (valid only): {mean_val:.4f}")
    print(f"High confidence (>0.5): {high_conf:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
