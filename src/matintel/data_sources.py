from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import FORMULA_COL, MATERIAL_ID_COL, RAW_CSV


REQUIRED_COLUMNS = [
    MATERIAL_ID_COL,
    FORMULA_COL,
    "Bandgap",
    "Formation Energy Per Atom",
    "Decomposition Energy Per Atom",
    "NSites",
    "Dimensionality Cheon",
    "Crystal System",
]


def ensure_demo_dataset(csv_path: Path = RAW_CSV) -> Path:
    """Create a small deterministic dataset if the raw file is missing."""
    if csv_path.exists():
        return csv_path

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ["gnome-000001", "LiCoO2", 0.2, -1.10, 0.02, 12, 3, "Trigonal"],
        ["gnome-000002", "LiFePO4", 0.4, -1.35, 0.01, 28, 3, "Orthorhombic"],
        ["gnome-000003", "Bi2Te3", 0.18, -0.62, 0.03, 15, 3, "Rhombohedral"],
        ["gnome-000004", "Si", 1.12, -0.45, 0.00, 2, 3, "Cubic"],
        ["gnome-000005", "GaAs", 1.42, -0.55, 0.01, 8, 3, "Cubic"],
        ["gnome-000006", "Nd2Fe14B", 0.0, -0.91, 0.04, 68, 3, "Tetragonal"],
        ["gnome-000007", "Fe", 0.0, -0.10, 0.01, 2, 3, "Cubic"],
        ["gnome-000008", "Al2O3", 8.8, -1.80, 0.00, 30, 3, "Trigonal"],
        ["gnome-000009", "TiN", 0.15, -1.60, 0.01, 8, 3, "Cubic"],
        ["gnome-000010", "CsPbI3", 1.65, -0.28, 0.08, 20, 3, "Orthorhombic"],
        ["gnome-000011", "ZnO", 3.3, -1.12, 0.01, 4, 3, "Hexagonal"],
        ["gnome-000012", "Co3O4", 1.8, -0.88, 0.05, 14, 3, "Cubic"],
    ]
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    df.to_csv(csv_path, index=False)
    return csv_path


def load_raw(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        ensure_demo_dataset(csv_path)
    df = pd.read_csv(csv_path)
    _add_missing_columns(df, REQUIRED_COLUMNS)
    return df


def filter_by_ids(df: pd.DataFrame, material_ids: Iterable[str] | None = None) -> pd.DataFrame:
    if not material_ids:
        return df.copy()
    ids = set(material_ids)
    return df[df[MATERIAL_ID_COL].isin(ids)].copy()


def _add_missing_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col not in df.columns:
            df[col] = None
