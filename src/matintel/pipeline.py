from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import APP_LABELS, FEATURED_CSV, LOG_DIR, SCORED_CSV, WORKING_CSV
from .data_sources import ensure_demo_dataset, load_raw
from .features import featurize
from .scoring import apply_application_scores
from .viability import apply_viability


def run_pipeline(output_dir: Path | None = None) -> Path:
    out_dir = output_dir or SCORED_CSV.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = _build_logger(LOG_DIR / "pipeline.log")
    logger.info("Starting MatIntel pipeline")

    raw_path = ensure_demo_dataset()
    logger.info("Raw dataset path: %s", raw_path)

    df_raw = load_raw(raw_path)
    logger.info("Loaded raw rows: %s", len(df_raw))

    df_working = df_raw.copy()
    df_working.to_csv(out_dir / WORKING_CSV.name, index=False)
    logger.info("Wrote working dataset: %s", out_dir / WORKING_CSV.name)

    df_featured = featurize(df_working)
    df_featured.to_csv(out_dir / FEATURED_CSV.name, index=False)
    logger.info("Wrote featured dataset: %s", out_dir / FEATURED_CSV.name)

    df_scored = apply_application_scores(df_featured)
    df_scored = apply_viability(df_scored)

    score_cols = [col for col in APP_LABELS.values() if col in df_scored.columns]
    df_scored["best_score"] = df_scored[score_cols].max(axis=1).round(3)

    final_path = out_dir / SCORED_CSV.name
    df_scored.to_csv(final_path, index=False)
    logger.info("Wrote scored dataset: %s", final_path)
    logger.info("Pipeline finished")

    return final_path


def _build_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("matintel.pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MatIntel data pipeline")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output folder for CSV artifacts",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    final_path = run_pipeline(args.output_dir)

    df = pd.read_csv(final_path)
    print(f"Scored dataset ready: {final_path}")
    print(f"Rows: {len(df):,}")
    print("Top columns:", ", ".join(df.columns[:10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
