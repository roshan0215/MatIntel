from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

from .config import CRITICAL_MINERALS, DEFAULT_ELEMENT_PRICE_USD_KG, FORMULA_COL


RADIOACTIVE_SYMBOLS = {
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Tc", "Po", "At", "Rn", "Fr", "Ra",
}

HARD_RARE_EARTH_PENALTY = {
    "Dy": 0.3,
    "Tb": 0.3,
    "Eu": 0.3,
    "Ho": 0.3,
    "Er": 0.3,
    "Tm": 0.3,
    "Lu": 0.3,
    "Yb": 0.3,
}

MODERATE_RARE_EARTH_PENALTY = {
    "Nd": 0.6,
    "Pr": 0.6,
    "Sm": 0.6,
    "Gd": 0.6,
    "Sc": 0.6,
}

MILD_RARE_EARTH_PENALTY = {
    "La": 0.85,
    "Ce": 0.85,
    "Y": 0.85,
}

RARE_EARTH_PENALTIES = {
    **HARD_RARE_EARTH_PENALTY,
    **MODERATE_RARE_EARTH_PENALTY,
    **MILD_RARE_EARTH_PENALTY,
}


def material_cost_score(formula: str, price_dict: dict[str, float] | None = None, max_cost: float = 120.0) -> float:
    price_map = price_dict or DEFAULT_ELEMENT_PRICE_USD_KG
    try:
        comp = Composition(formula)
        total_weight = sum(comp[el] * float(el.atomic_mass) for el in comp.elements)
        weighted_cost = sum(
            (comp[el] * float(el.atomic_mass) / total_weight) * price_map.get(str(el), 50.0)
            for el in comp.elements
        )
        return round(max(0.0, 1 - weighted_cost / max_cost), 3)
    except Exception:
        return 0.5


def abundance_score(formula: str) -> float:
    try:
        comp = Composition(formula)
        min_abund = min((Element(str(el)).abundance or 0.001) for el in comp.elements)
        score = math.log10(min_abund + 0.001) / math.log10(282000.0)
        return round(float(np.clip(score, 0.0, 1.0)), 3)
    except Exception:
        return 0.5


def supply_risk_score(formula: str) -> float:
    """Return 1.0 for low risk, 0.0 for high risk."""
    try:
        elems = {str(e) for e in Composition(formula).elements}
        critical_count = len(elems & CRITICAL_MINERALS)
        return round(1 - critical_count / max(len(elems), 1), 3)
    except Exception:
        return 0.5


def viability_filter_multiplier(structure: object | None, composition: dict[str, float] | None) -> float:
    """
    Return a multiplier to apply to base viability.

    Rules:
    1) Hard radioactive filter: return 0.0 immediately.
    2) Rare-earth penalty: multiply penalties for all present elements.
    """
    symbols = _extract_symbols(structure, composition)
    if not symbols:
        return 1.0

    # Hard reject: explicit radioactive symbols or any element with Z > 83.
    for sym in symbols:
        try:
            if sym in RADIOACTIVE_SYMBOLS or Element(sym).Z > 83:
                return 0.0
        except Exception:
            continue

    multiplier = 1.0
    for sym in symbols:
        multiplier *= RARE_EARTH_PENALTIES.get(sym, 1.0)
    return round(multiplier, 6)


def _extract_symbols(structure: object | None, composition: dict[str, float] | None) -> set[str]:
    symbols: set[str] = set()

    if composition:
        symbols.update(str(sym) for sym in composition.keys())

    if structure is not None:
        try:
            symbols.update(str(el) for el in structure.composition.elements)
        except Exception:
            pass

    return symbols


def viability_filter_multiplier_from_formula(formula: str) -> float:
    try:
        comp = Composition(formula)
        return viability_filter_multiplier(structure=None, composition=comp.fractional_composition.as_dict())
    except Exception:
        return 1.0


def clscore_penalty(clscore: float) -> float:
    """Map raw CLscore to viability multiplier."""
    try:
        value = float(clscore)
    except Exception:
        return 0.5

    if value == -1:
        return 0.5
    if value >= 0.5:
        return 1.0
    if value >= 0.3:
        return 0.7
    if value >= 0.1:
        return 0.4
    return 0.1


def apply_viability(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cost_score"] = out[FORMULA_COL].apply(material_cost_score)
    out["abundance_score"] = out[FORMULA_COL].apply(abundance_score)
    out["supply_risk"] = out[FORMULA_COL].apply(supply_risk_score)
    out["viability_filter_multiplier"] = out[FORMULA_COL].apply(viability_filter_multiplier_from_formula)
    out["viability"] = (
        0.45 * out["cost_score"] +
        0.35 * out["abundance_score"] +
        0.20 * out["supply_risk"]
    ) * out["viability_filter_multiplier"]

    if "clscore" in out.columns:
        out["clscore_multiplier"] = out["clscore"].apply(clscore_penalty)
    else:
        out["clscore_multiplier"] = 1.0

    out["viability"] = out["viability"] * out["clscore_multiplier"]
    out["viability"] = out["viability"].round(3)
    return out
