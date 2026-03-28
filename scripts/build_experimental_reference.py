from __future__ import annotations

"""
Build a merged experimental reference table aligned to MatIntel scored_dataset schema.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\build_experimental_reference.py

Requirements:
  pip install mp-api matminer jarvis-tools pymatgen pandas tqdm

Env vars (either works):
  MATINTEL_MP_API_KEY=<materials project key>
  MP_API_KEY=<materials project key>
"""

import argparse
import os
from pathlib import Path
from typing import Any

import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm


def _safe_import_matminer_loader():
    try:
        mod = importlib.import_module("matminer.datasets")
        return getattr(mod, "load_dataset", None)
    except Exception:
        return None


def _safe_import_jarvis_data():
    try:
        mod = importlib.import_module("jarvis.db.figshare")
        return getattr(mod, "data", None)
    except Exception:
        return None


def _normalise_formula(formula: Any) -> str:
    try:
        from pymatgen.core import Composition

        return Composition(str(formula)).reduced_formula
    except Exception:
        return str(formula).strip()


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _add_key(df: pd.DataFrame, formula_col: str = "Reduced Formula") -> pd.DataFrame:
    out = df.copy()
    if formula_col in out.columns:
        out["_key"] = out[formula_col].map(_normalise_formula)
    else:
        out["_key"] = ""
    return out


def fetch_mp_experimental(api_key: str) -> pd.DataFrame:
    if not api_key or api_key == "YOUR_KEY_HERE":
        raise ValueError("Missing Materials Project key. Set MATINTEL_MP_API_KEY or MP_API_KEY.")

    from mp_api.client import MPRester

    rows: list[dict[str, Any]] = []
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            theoretical=False,
            fields=[
                "material_id",
                "formula_pretty",
                "band_gap",
                "formation_energy_per_atom",
                "energy_above_hull",
                "symmetry",
                "is_stable",
                "nsites",
            ],
        )

        for doc in tqdm(docs, desc="MP experimental"):
            try:
                symmetry = getattr(doc, "symmetry", None)
                cs = getattr(symmetry, "crystal_system", None) if symmetry is not None else None
                rows.append(
                    {
                        "MaterialId": str(getattr(doc, "material_id", "")),
                        "Reduced Formula": str(getattr(doc, "formula_pretty", "")),
                        "Bandgap": _as_float(getattr(doc, "band_gap", np.nan)),
                        "Formation Energy Per Atom": _as_float(getattr(doc, "formation_energy_per_atom", np.nan)),
                        "Decomposition Energy Per Atom": _as_float(getattr(doc, "energy_above_hull", np.nan)),
                        "Crystal System": str(cs).lower() if cs is not None else np.nan,
                        "NSites": _as_float(getattr(doc, "nsites", np.nan)),
                        "source": "MP_synthesized",
                        "is_experimental": True,
                        "is_stable_mp": bool(getattr(doc, "is_stable", False)),
                        "clscore": -1.0,
                    }
                )
            except Exception:
                continue

    out = pd.DataFrame(rows)
    return out.drop_duplicates(subset=["MaterialId"], keep="first")


def load_matminer_tables() -> list[pd.DataFrame]:
    load_dataset = _safe_import_matminer_loader()
    if load_dataset is None:
        print("matminer not available; skipping matminer datasets")
        return []

    tables: list[pd.DataFrame] = []

    def try_load(name: str) -> pd.DataFrame:
        try:
            return load_dataset(name)
        except Exception as exc:
            print(f"  skipped {name}: {exc}")
            return pd.DataFrame()

    print("Loading matminer experimental datasets...")

    expt_gap = try_load("expt_gap")
    if not expt_gap.empty:
        expt_gap = expt_gap.rename(columns={"formula": "Reduced Formula", "gap expt": "bandgap_experimental"})
        expt_gap = expt_gap[["Reduced Formula", "bandgap_experimental"]].dropna()
        tables.append(_add_key(expt_gap))

    mb_gap = try_load("matbench_expt_gap")
    if not mb_gap.empty:
        mb_gap = mb_gap.rename(
            columns={"composition": "Reduced Formula", "gap expt": "bandgap_experimental_mb"}
        )
        mb_gap = mb_gap[["Reduced Formula", "bandgap_experimental_mb"]].dropna()
        tables.append(_add_key(mb_gap))

    tc = try_load("citrine_thermal_conductivity")
    if not tc.empty and "formula" in tc.columns:
        tc_candidates = [c for c in tc.columns if "expt" in c.lower() and "k" in c.lower()]
        tc_col = tc_candidates[0] if tc_candidates else None
        if tc_col is not None:
            tc = tc.rename(columns={"formula": "Reduced Formula", tc_col: "thermal_conductivity_expt"})
            tc = tc[["Reduced Formula", "thermal_conductivity_expt"]].dropna()
            tables.append(_add_key(tc))

    expt_hf = try_load("expt_formation_enthalpy")
    if not expt_hf.empty and "formula" in expt_hf.columns:
        form_col = "expt_form_e" if "expt_form_e" in expt_hf.columns else None
        if form_col:
            expt_hf = expt_hf.rename(
                columns={"formula": "Reduced Formula", form_col: "formation_enthalpy_experimental"}
            )
            expt_hf = expt_hf[["Reduced Formula", "formation_enthalpy_experimental"]].dropna()
            tables.append(_add_key(expt_hf))

    elastic = try_load("elastic_tensor_2015")
    if not elastic.empty and "formula" in elastic.columns:
        if "K_VRH" in elastic.columns and "G_VRH" in elastic.columns:
            elastic = elastic.rename(columns={"formula": "Reduced Formula", "K_VRH": "bulk_modulus", "G_VRH": "shear_modulus"})
            elastic = elastic[["Reduced Formula", "bulk_modulus", "shear_modulus"]].dropna()
            tables.append(_add_key(elastic))

    dielec = try_load("dielectric_constant")
    if not dielec.empty and "formula" in dielec.columns:
        cols = ["formula"]
        if "n" in dielec.columns:
            cols.append("n")
        if "band_gap" in dielec.columns:
            cols.append("band_gap")
        dielec = dielec[cols].rename(
            columns={"formula": "Reduced Formula", "n": "refractive_index", "band_gap": "bandgap_dfpt"}
        )
        tables.append(_add_key(dielec))

    piezo = try_load("piezoelectric_tensor")
    if not piezo.empty and "formula" in piezo.columns and "eij_max" in piezo.columns:
        piezo = piezo.rename(columns={"formula": "Reduced Formula", "eij_max": "piezo_max"})
        piezo = piezo[["Reduced Formula", "piezo_max"]].dropna()
        tables.append(_add_key(piezo))

    print(f"  loaded {len(tables)} matminer tables")
    return tables


def load_jarvis_table() -> pd.DataFrame:
    jarvis_data = _safe_import_jarvis_data()
    if jarvis_data is None:
        print("jarvis-tools not available; skipping JARVIS")
        return pd.DataFrame()

    print("Loading JARVIS dft_3d dataset (first run may download cache)...")
    try:
        dft_3d = jarvis_data("dft_3d")
    except Exception as exc:
        print(f"  JARVIS load failed: {exc}")
        return pd.DataFrame()

    df = pd.DataFrame(dft_3d).replace("na", np.nan)
    keep_cols = [
        "jid",
        "formula",
        "icsd",
        "mbj_bandgap",
        "optb88vdw_bandgap",
        "formation_energy_peratom",
        "ehull",
        "n-Seebeck",
        "p-Seebeck",
        "n-powerfact",
        "p-powerfact",
        "bulk_modulus_kv",
        "shear_modulus_gv",
        "dfpt_piezo_max_eij",
        "magmom_outcar",
        "spg_number",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    if not keep_cols:
        return pd.DataFrame()

    df = df[keep_cols].copy()
    df["is_icsd_sourced"] = df.get("icsd").notna() & (df.get("icsd") != "")
    df = df.rename(
        columns={
            "formula": "Reduced Formula",
            "formation_energy_peratom": "formation_energy_jarvis",
            "ehull": "ehull_jarvis",
            "mbj_bandgap": "bandgap_mbj",
            "optb88vdw_bandgap": "bandgap_jarvis_gga",
            "bulk_modulus_kv": "bulk_modulus_jarvis",
            "shear_modulus_gv": "shear_modulus_jarvis",
            "dfpt_piezo_max_eij": "piezo_max_jarvis",
            "magmom_outcar": "magnetic_moment",
            "spg_number": "space_group_number",
        }
    )
    return _add_key(df)


def _best_band_gap(row: pd.Series) -> float:
    for col in ["bandgap_experimental", "bandgap_experimental_mb", "bandgap_mbj", "Bandgap"]:
        val = row.get(col)
        if pd.notna(val):
            return _as_float(val)
    return float("nan")


def _merge_property_tables(base: pd.DataFrame, tables: list[pd.DataFrame]) -> pd.DataFrame:
    out = base.copy()
    for prop in tables:
        if prop.empty or "_key" not in prop.columns:
            continue
        dedup = prop.drop_duplicates(subset=["_key"], keep="first")
        join_cols = [c for c in dedup.columns if c not in {"Reduced Formula"}]
        out = out.merge(dedup[join_cols], on="_key", how="left")
    return out


def _align_to_scored_schema(df: pd.DataFrame, scored_schema_path: Path) -> pd.DataFrame:
    if not scored_schema_path.exists():
        return df

    schema_cols = pd.read_csv(scored_schema_path, nrows=0).columns.tolist()

    out = df.copy()
    for col in schema_cols:
        if col not in out.columns:
            out[col] = np.nan

    # Ensure expected sentinel/defaults when present.
    if "clscore" in out.columns:
        out["clscore"] = pd.to_numeric(out["clscore"], errors="coerce").fillna(-1.0)

    # Keep schema columns first to allow direct concat with scored_dataset.
    ordered = schema_cols + [c for c in out.columns if c not in schema_cols]
    return out[ordered]


def build_experimental_reference(output_path: Path, scored_schema_path: Path, api_key: str) -> pd.DataFrame:
    print("Fetching Materials Project experimental structures...")
    mp_df = fetch_mp_experimental(api_key)
    print(f"  MP rows: {len(mp_df):,}")

    mp_df = _add_key(mp_df)

    matminer_tables = load_matminer_tables()
    jarvis_df = load_jarvis_table()

    merged = _merge_property_tables(mp_df, matminer_tables)

    if not jarvis_df.empty:
        dedup_jarvis = jarvis_df.drop_duplicates(subset=["_key"], keep="first")
        merged = merged.merge(dedup_jarvis.drop(columns=["Reduced Formula"], errors="ignore"), on="_key", how="left")

    merged["Bandgap"] = merged.apply(_best_band_gap, axis=1)
    merged["bandgap_source"] = np.where(
        merged.get("bandgap_experimental").notna(),
        "experimental",
        np.where(merged.get("bandgap_mbj").notna(), "mbj_dft", "mp_dft"),
    )

    if not jarvis_df.empty:
        mp_keys = set(merged["_key"].astype(str).tolist())
        jarvis_only = jarvis_df[jarvis_df["is_icsd_sourced"] & (~jarvis_df["_key"].isin(mp_keys))].copy()

        if len(jarvis_only):
            extra = pd.DataFrame(
                {
                    "MaterialId": "JARVIS_" + jarvis_only["jid"].astype(str),
                    "Reduced Formula": jarvis_only["Reduced Formula"],
                    "Bandgap": pd.to_numeric(jarvis_only.get("bandgap_mbj"), errors="coerce"),
                    "Formation Energy Per Atom": pd.to_numeric(
                        jarvis_only.get("formation_energy_jarvis"), errors="coerce"
                    ),
                    "Decomposition Energy Per Atom": pd.to_numeric(jarvis_only.get("ehull_jarvis"), errors="coerce"),
                    "Crystal System": np.nan,
                    "NSites": np.nan,
                    "source": "JARVIS_ICSD",
                    "is_experimental": True,
                    "clscore": -1.0,
                    "bandgap_source": "mbj_dft",
                    "n-Seebeck": jarvis_only.get("n-Seebeck"),
                    "p-Seebeck": jarvis_only.get("p-Seebeck"),
                    "bulk_modulus_jarvis": jarvis_only.get("bulk_modulus_jarvis"),
                    "magnetic_moment": jarvis_only.get("magnetic_moment"),
                }
            )
            extra = _add_key(extra)
            merged = pd.concat([merged, extra], ignore_index=True)
            print(f"  added JARVIS-only ICSD rows: {len(extra):,}")

    merged = merged.drop(columns=["_key"], errors="ignore")

    # Helpful provenance flags.
    merged["is_experimental"] = merged.get("is_experimental", True)
    merged["source"] = merged.get("source", "MP_synthesized")

    merged = _align_to_scored_schema(merged, scored_schema_path)
    merged = merged.drop_duplicates(subset=["MaterialId"], keep="first")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(merged):,}")
    if "bandgap_source" in merged.columns:
        print("Bandgap source counts:", merged["bandgap_source"].value_counts(dropna=False).to_dict())

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged experimental reference table for MatIntel")
    parser.add_argument(
        "--output",
        default="data/processed/experimental_compounds.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--scored-schema",
        default="data/processed/scored_dataset.csv",
        help="Path to scored_dataset CSV used for schema alignment",
    )
    parser.add_argument(
        "--mp-api-key",
        default="",
        help="Materials Project API key (overrides env vars)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = (
        args.mp_api_key.strip()
        or os.getenv("MATINTEL_MP_API_KEY", "").strip()
        or os.getenv("MP_API_KEY", "").strip()
    )

    build_experimental_reference(
        output_path=Path(args.output),
        scored_schema_path=Path(args.scored_schema),
        api_key=api_key,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
