from __future__ import annotations

from pathlib import Path
import zipfile

from tqdm import tqdm


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    zip_path = root / "data" / "by_id.zip"
    out_dir = root / "data" / "cifs"

    if not zip_path.exists():
        raise FileNotFoundError(f"Missing archive: {zip_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        cif_members = [m for m in zf.infolist() if m.filename.lower().endswith(".cif")]

        extracted = 0
        skipped = 0

        for member in tqdm(cif_members, desc="Extract CIFs"):
            name = Path(member.filename).name
            target = out_dir / name
            if target.exists() and target.stat().st_size > 0:
                skipped += 1
                continue

            with zf.open(member, "r") as src, target.open("wb") as dst:
                dst.write(src.read())
            extracted += 1

    print(f"Done. Extracted: {extracted}, skipped existing: {skipped}, total target: {len(cif_members)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
