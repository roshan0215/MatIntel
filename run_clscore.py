from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.matintel.clscore import batch_clscore
from src.matintel.config import SCORED_CSV
from src.matintel.viability import apply_viability

APP_TO_SCORE = {
    "battery": "score_battery_cathode_liion",
    "battery_cathode_liion": "score_battery_cathode_liion",
    "battery_anode": "score_battery_anode",
    "battery_cathode_naion": "score_battery_cathode_naion",
    "solid_electrolyte": "score_solid_electrolyte",
    "hydrogen_storage": "score_hydrogen_storage",
    "solar_singlejunction": "score_solar_singlejunction",
    "solar_tandem": "score_solar_tandem",
    "semiconductor": "score_semiconductor",
    "led": "score_led",
    "photodetector": "score_photodetector",
    "transparent_conductor": "score_transparent_conductor",
    "ferroelectric": "score_ferroelectric",
    "piezoelectric": "score_piezoelectric",
    "topological_insulator": "score_topological_insulator",
    "thermoelectric": "score_thermoelectric",
    "oer_electrocatalyst": "score_oer_electrocatalyst",
    "her_electrocatalyst": "score_her_electrocatalyst",
    "co2_reduction": "score_co2_reduction",
    "photocatalyst_h2o": "score_photocatalyst_h2o",
    "magnet": "score_permanent_magnet",
    "permanent_magnet": "score_permanent_magnet",
    "soft_magnet": "score_soft_magnet",
    "magnetic_semiconductor": "score_magnetic_semiconductor",
    "coating": "score_hard_coating",
    "thermal_barrier": "score_thermal_barrier",
    "thermal_interface": "score_thermal_interface",
    "hard_coating": "score_hard_coating",
    "corrosion_resistant": "score_corrosion_resistant",
    "refractory": "score_refractory",
    "superconductor": "score_superconductor",
    "radiation_detector": "score_radiation_detector",
    "sofc_electrolyte": "score_sofc_electrolyte",
    "multiferroic": "score_multiferroic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLscore on top candidates")
    parser.add_argument("--app", required=True, choices=sorted(APP_TO_SCORE.keys()))
    parser.add_argument("--top-n", type=int, default=1000)
    parser.add_argument("--cif-dir", default="data/cifs")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output-csv", default="data/processed/clscore_results.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    score_col = APP_TO_SCORE[args.app]
    scored_path = Path(SCORED_CSV)
    if not scored_path.exists():
        raise FileNotFoundError(f"Missing scored dataset: {scored_path}")

    df = pd.read_csv(scored_path)
    if score_col not in df.columns or "viability" not in df.columns:
        raise ValueError(f"Missing required columns for ranking ({score_col}, viability)")

    rank_col = f"rank_{args.app}"
    df[rank_col] = pd.to_numeric(df[score_col], errors="coerce").fillna(0) * pd.to_numeric(
        df["viability"], errors="coerce"
    ).fillna(0)

    top = df.sort_values(rank_col, ascending=False).head(args.top_n).copy()
    material_ids = top["MaterialId"].astype(str).tolist()

    clscore_df = batch_clscore(
        material_ids=material_ids,
        cif_dir=args.cif_dir,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
    )

    df = df.drop(columns=["clscore"], errors="ignore")
    df = df.merge(clscore_df[["MaterialId", "clscore"]], on="MaterialId", how="left")
    df["clscore"] = df["clscore"].fillna(-1.0)
    df = apply_viability(df)
    df.to_csv(scored_path, index=False)

    valid = clscore_df[clscore_df["clscore"] >= 0]
    scored_count = len(clscore_df)
    mean_val = float(valid["clscore"].mean()) if len(valid) else -1.0
    high_conf = float((valid["clscore"] > 0.5).mean() * 100) if len(valid) else 0.0

    print(f"Scored {scored_count} materials. Mean CLscore: {mean_val:.4f}. High confidence (>0.5): {high_conf:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
