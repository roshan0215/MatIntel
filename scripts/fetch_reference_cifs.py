from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch CIF files for reference IDs (MP + JARVIS)")
    parser.add_argument(
        "--input-csv",
        default="data/processed/experimental_compounds.csv",
        help="CSV containing MaterialId/source rows",
    )
    parser.add_argument("--cif-dir", default="data/cifs", help="Output CIF directory")
    parser.add_argument("--chunk-size", type=int, default=200, help="MP query chunk size")
    return parser.parse_args()


def _has_cif(cif_dir: Path, material_id: str) -> bool:
    return (cif_dir / f"{material_id}.CIF").exists() or (cif_dir / f"{material_id}.cif").exists()


def fetch_mp_cifs(mp_ids: list[str], cif_dir: Path, chunk_size: int) -> tuple[int, int]:
    if not mp_ids:
        return 0, 0

    api_key = os.getenv("MATINTEL_MP_API_KEY", "").strip() or os.getenv("MP_API_KEY", "").strip()
    if not api_key:
        print("No MP API key found (MATINTEL_MP_API_KEY/MP_API_KEY); skipping MP CIF fetch")
        return 0, len(mp_ids)

    from mp_api.client import MPRester  # type: ignore

    written = 0
    failed = 0

    with MPRester(api_key) as mpr:
        for i in range(0, len(mp_ids), chunk_size):
            chunk = mp_ids[i : i + chunk_size]
            try:
                docs = mpr.materials.summary.search(material_ids=chunk, fields=["material_id", "structure"])
            except Exception as exc:
                failed += len(chunk)
                print(f"MP chunk failed ({i}-{i+len(chunk)-1}): {exc}")
                continue

            found = set()
            for d in docs:
                mid = str(d.material_id)
                found.add(mid)
                struct = getattr(d, "structure", None)
                if struct is None:
                    failed += 1
                    continue
                try:
                    (cif_dir / f"{mid}.CIF").write_text(struct.to(fmt="cif"), encoding="utf-8")
                    written += 1
                except Exception:
                    failed += 1

            missing = len(set(chunk) - found)
            failed += missing

            if ((i // chunk_size) + 1) % 20 == 0:
                print(f"MP progress: {min(i + chunk_size, len(mp_ids))}/{len(mp_ids)} | written={written} failed={failed}")

    return written, failed


def fetch_jarvis_cifs(jarvis_ids: list[str], cif_dir: Path) -> tuple[int, int]:
    if not jarvis_ids:
        return 0, 0

    try:
        from jarvis.db.figshare import data as jarvis_data  # type: ignore
        from jarvis.core.atoms import Atoms  # type: ignore
    except Exception as exc:
        print(f"jarvis-tools unavailable; skipping JARVIS CIF fetch: {exc}")
        return 0, len(jarvis_ids)

    dft_3d = jarvis_data("dft_3d")
    index = {str(r.get("jid", "")): r for r in dft_3d}

    written = 0
    failed = 0

    for mid in jarvis_ids:
        jid = mid.replace("JARVIS_", "")
        row = index.get(jid)
        if not row or not row.get("atoms"):
            failed += 1
            continue
        try:
            atoms = Atoms.from_dict(row["atoms"])
            struct = atoms.pymatgen_converter()
            (cif_dir / f"{mid}.CIF").write_text(struct.to(fmt="cif"), encoding="utf-8")
            written += 1
        except Exception:
            failed += 1

    return written, failed


def main() -> int:
    args = parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    cif_dir = Path(args.cif_dir)
    cif_dir.mkdir(parents=True, exist_ok=True)

    exp_df = pd.read_csv(input_csv, usecols=["MaterialId", "source"])
    exp_df["MaterialId"] = exp_df["MaterialId"].astype(str)

    missing = exp_df[~exp_df["MaterialId"].map(lambda m: _has_cif(cif_dir, m))].copy()
    print(f"Reference rows total: {len(exp_df):,}")
    print(f"Missing CIFs before fetch: {len(missing):,}")

    mp_missing = missing[missing["MaterialId"].str.startswith("mp-")]["MaterialId"].drop_duplicates().tolist()
    jarvis_missing = (
        missing[missing["MaterialId"].str.startswith("JARVIS_JVASP-")]["MaterialId"].drop_duplicates().tolist()
    )

    print(f"MP IDs to fetch: {len(mp_missing):,}")
    print(f"JARVIS IDs to fetch: {len(jarvis_missing):,}")

    mp_written, mp_failed = fetch_mp_cifs(mp_missing, cif_dir, args.chunk_size)
    jar_written, jar_failed = fetch_jarvis_cifs(jarvis_missing, cif_dir)

    still_missing = exp_df[~exp_df["MaterialId"].map(lambda m: _has_cif(cif_dir, m))]

    print("Done fetching reference CIFs")
    print(f"  MP written={mp_written:,} failed={mp_failed:,}")
    print(f"  JARVIS written={jar_written:,} failed={jar_failed:,}")
    print(f"  Missing CIFs after fetch: {len(still_missing):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
