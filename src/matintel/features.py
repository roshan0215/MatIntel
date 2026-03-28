from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from .config import FORMULA_COL, TRANSITION_METALS


def get_elements(formula: str) -> set[str]:
    try:
        return {str(e) for e in Composition(formula).elements}
    except Exception:
        return set()


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["n_elements"] = out[FORMULA_COL].apply(_n_elements)
    out["mean_atomic_number"] = out[FORMULA_COL].apply(_mean_atomic_number)
    out["mean_atomic_mass"] = out[FORMULA_COL].apply(_mean_atomic_mass)
    out["mean_electronegativity"] = out[FORMULA_COL].apply(_mean_en)
    out["std_electronegativity"] = out[FORMULA_COL].apply(_std_en)
    out["contains_transition_metal"] = out[FORMULA_COL].apply(
        lambda f: int(bool(get_elements(str(f)) & TRANSITION_METALS))
    )
    out["formula_complexity"] = out[FORMULA_COL].apply(_formula_complexity)

    out["Bandgap"] = pd.to_numeric(out.get("Bandgap"), errors="coerce")
    out["Formation Energy Per Atom"] = pd.to_numeric(
        out.get("Formation Energy Per Atom"), errors="coerce"
    )
    out["Decomposition Energy Per Atom"] = pd.to_numeric(
        out.get("Decomposition Energy Per Atom"), errors="coerce"
    )
    out["NSites"] = pd.to_numeric(out.get("NSites"), errors="coerce")

    return out


def _safe_comp(formula: str) -> Composition | None:
    try:
        return Composition(formula)
    except Exception:
        return None


def _n_elements(formula: str) -> int:
    comp = _safe_comp(str(formula))
    return len(comp.elements) if comp else 0


def _weighted_values(comp: Composition, values: Iterable[float]) -> float:
    fractions = [comp.get_atomic_fraction(el) for el in comp.elements]
    return float(np.dot(np.array(list(values), dtype=float), np.array(fractions, dtype=float)))


def _mean_atomic_number(formula: str) -> float:
    comp = _safe_comp(str(formula))
    if not comp:
        return np.nan
    return _weighted_values(comp, [el.Z for el in comp.elements])


def _mean_atomic_mass(formula: str) -> float:
    comp = _safe_comp(str(formula))
    if not comp:
        return np.nan
    return _weighted_values(comp, [float(el.atomic_mass) for el in comp.elements])


def _mean_en(formula: str) -> float:
    comp = _safe_comp(str(formula))
    if not comp:
        return np.nan
    vals = [(el.X or 0.0) for el in comp.elements]
    return _weighted_values(comp, vals)


def _std_en(formula: str) -> float:
    comp = _safe_comp(str(formula))
    if not comp:
        return np.nan
    vals = np.array([(el.X or 0.0) for el in comp.elements], dtype=float)
    fractions = np.array([comp.get_atomic_fraction(el) for el in comp.elements], dtype=float)
    mean = float(np.dot(vals, fractions))
    var = float(np.dot((vals - mean) ** 2, fractions))
    return math.sqrt(max(var, 0.0))


def _formula_complexity(formula: str) -> float:
    comp = _safe_comp(str(formula))
    if not comp:
        return np.nan
    fractions = np.array([comp.get_atomic_fraction(el) for el in comp.elements], dtype=float)
    entropy = -np.sum([f * np.log(f) for f in fractions if f > 0])
    return float(entropy)
