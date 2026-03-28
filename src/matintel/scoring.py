from __future__ import annotations

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

from .config import FORMULA_COL


def _num(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def _elements(composition: Composition) -> set[str]:
    return {str(e) for e in composition.elements}


def _in_range(value: float, low: float, high: float) -> bool:
    return (not np.isnan(value)) and low <= value <= high


def _clamp(score: float) -> float:
    return round(float(max(0.0, min(score, 1.0))), 3)


def _amounts(composition: Composition) -> dict[str, float]:
    return {k: float(v) for k, v in composition.get_el_amt_dict().items()}


def _is_metal(symbol: str) -> bool:
    try:
        return bool(Element(symbol).is_metal)
    except Exception:
        return False


def score_battery_cathode_liion(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    if "Li" not in elems:
        return 0.0
    score = 0.0
    if elems & {"Mn", "Fe", "Co", "Ni", "V", "Cr", "Cu", "Mo", "Ti"}:
        score += 0.35
    if "O" in elems and elems & {"P", "Si", "S", "B"}:
        score += 0.2
    elif "O" in elems:
        score += 0.1
    if _in_range(band_gap, 0.01, 3.0):
        score += 0.25
    elif not np.isnan(band_gap) and band_gap < 0.01:
        score += 0.1
    if not np.isnan(formation_energy) and formation_energy < -1.0:
        score += 0.1
    if "F" in elems:
        score *= 0.7
    return _clamp(score)


def score_battery_anode(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if elems & {"Si", "Sn", "Sb", "P", "Ge", "Bi", "Al"}:
        score += 0.4
    if (elems & {"Fe", "Co", "Ni", "Cu", "Mn", "Mo", "W"}) and (elems & {"O", "S", "Se", "F", "N"}):
        score += 0.25
    if not np.isnan(band_gap):
        if band_gap < 0.1:
            score += 0.2
        elif band_gap < 0.5:
            score += 0.12
        elif band_gap < 1.5:
            score += 0.05
    if "Fe" in elems and "Si" in elems:
        score += 0.15
    if "Li" in elems or "Na" in elems:
        score *= 0.5
    return _clamp(score)


def score_battery_cathode_naion(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    if "Na" not in elems:
        return 0.0
    score = 0.0
    if elems & {"Fe", "Mn"}:
        score += 0.4
    elif elems & {"Co", "Ni", "V", "Cr", "Cu", "Ti"}:
        score += 0.25
    if "O" in elems and elems & {"P", "Si", "S", "B"}:
        score += 0.2
    elif "O" in elems:
        score += 0.12
    if _in_range(band_gap, 0.0, 3.0):
        score += 0.2
    if "Li" not in elems:
        score += 0.08
    return _clamp(score)


def score_solid_electrolyte(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    amounts = _amounts(composition)
    if "Li" not in elems and "Na" not in elems:
        return 0.0
    if np.isnan(band_gap) or band_gap < 2.0:
        return 0.0
    score = 0.0
    if band_gap >= 4.0:
        score += 0.35
    elif band_gap >= 3.0:
        score += 0.25
    else:
        score += 0.1
    if elems & {"Zr", "La", "Al", "Ta", "Nb", "Ti"}:
        score += 0.25
    if "S" in elems and elems & {"P", "Si", "Ge", "Sn", "As"}:
        score += 0.2
    if elems & {"Cl", "Br", "I", "F"}:
        score += 0.2
    if "O" in elems:
        score += 0.1
    if not np.isnan(formation_energy) and formation_energy < -2.0:
        score += 0.1

    # Fluoride-rich frameworks are real but often less practical than oxide/sulfide systems.
    anion_total = sum(v for k, v in amounts.items() if not _is_metal(k))
    if anion_total > 0:
        f_fraction = amounts.get("F", 0.0) / anion_total
        if f_fraction > 0.4:
            score *= 0.65

    return _clamp(score)


def score_hydrogen_storage(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    if "H" not in elems:
        return 0.0
    score = 0.0
    if elems & {"Li", "Na", "Mg", "Al", "Ca", "K"}:
        score += 0.3
    if elems & {"Fe", "Ti", "Ni", "V", "Zr", "La", "Ce"}:
        score += 0.25
    if "B" in elems:
        score += 0.15
    if "Al" in elems:
        score += 0.1
    if _in_range(formation_energy, -1.5, -0.3):
        score += 0.2
    elif _in_range(formation_energy, -0.3, 0.0):
        score += 0.1
    return _clamp(score)


def score_solar_absorber_singlejunction(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    if np.isnan(band_gap):
        return 0.0
    score = 0.0
    if _in_range(band_gap, 1.0, 1.8):
        score += max(0.25, 0.5 - abs(band_gap - 1.34) * 0.5)
    elif _in_range(band_gap, 0.8, 1.0):
        score += 0.1
    else:
        return 0.0
    if len(elems & {"Cu", "In", "Ga", "Se", "S"}) >= 3:
        score += 0.25
    if len(elems & {"Cu", "Zn", "Sn", "S", "Se"}) >= 3:
        score += 0.2
    if elems & {"Mo", "W", "Sb", "Bi", "Ge"}:
        score += 0.1
    if "Cd" in elems:
        score *= 0.6
    if elems & {"As", "Hg"}:
        score *= 0.5
    if "Pb" in elems:
        score *= 0.7
    return _clamp(score)


def score_solar_absorber_tandem(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    amounts = _amounts(composition)
    if not _in_range(band_gap, 1.2, 2.3):
        return 0.0

    # Must include a plausible absorber metal center.
    absorber_metals = {"Pb", "Sn", "Ge", "Bi", "Sb", "In", "Ga", "Cu"}
    if not (elems & absorber_metals):
        return 0.0

    score = 0.0
    if _in_range(band_gap, 1.6, 2.0):
        score += max(0.25, 0.5 - abs(band_gap - 1.75) * 0.8)
    elif _in_range(band_gap, 1.4, 1.6) or _in_range(band_gap, 2.0, 2.2):
        score += 0.2
    if elems & {"I", "Br", "Cl"}:
        score += 0.2
    if elems & {"Pb", "Sn", "Ge"}:
        score += 0.15
    if "Pb" in elems:
        score *= 0.8

    # Thin-film tandem absorbers are usually chemically simpler.
    if len(elems) > 5:
        score *= 0.6

    # Halide-dominant compositions without a plausible A-site are poor thin-film absorber candidates.
    halogen_total = sum(amounts.get(x, 0.0) for x in ("F", "Cl", "Br", "I"))
    total = sum(amounts.values())
    a_site_cations = {"Cs", "Rb", "K", "Na", "Li", "Ba", "Sr", "Ca"}
    if total > 0 and (halogen_total / total) > 0.45 and not (elems & a_site_cations):
        score *= 0.25

    # Penalize toxic main constituents.
    primary_toxic = {"Hg", "Tl", "As"}
    if total > 0 and any((amounts.get(sym, 0.0) / total) >= 0.2 for sym in primary_toxic):
        score *= 0.4

    # If the heaviest element is N/P dominated chemistry, it's usually not a practical thin-film absorber.
    try:
        heaviest = max(elems, key=lambda s: Element(s).Z)
    except Exception:
        heaviest = ""
    if heaviest in {"N", "P"}:
        score *= 0.4

    return _clamp(score)


def score_thermoelectric(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if _in_range(band_gap, 0.05, 0.6):
        score += 0.35
    elif _in_range(band_gap, 0.6, 1.2):
        score += 0.15
    if elems & {"Bi", "Te", "Sb", "Se", "Pb", "Sn", "Ge"}:
        score += 0.25
    nsites = _num(features.get("NSites", np.nan))
    if np.isnan(nsites) and "NSites" in features:
        nsites = _num(features["NSites"])
    if np.isnan(nsites):
        nsites = _num(features.get("n_sites", np.nan))
    if np.isnan(nsites):
        nsites = 0
    if nsites > 10:
        score += 0.2
    if not np.isnan(formation_energy) and formation_energy < -0.5:
        score += 0.1
    if "O" in elems and not (elems & {"Co", "Ni", "Mn", "Fe"}):
        score *= 0.75
    return _clamp(score)


def score_oer_electrocatalyst(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if "O" in elems:
        score += 0.25
    if elems & {"Ni", "Fe", "Co", "Mn", "Ru", "Ir"}:
        score += 0.4
    if not np.isnan(band_gap) and band_gap < 2.0:
        score += 0.2
    if not np.isnan(formation_energy) and formation_energy < -1.0:
        score += 0.15
    if elems & {"Ir", "Ru"}:
        score *= 0.7

    # OER catalysts are generally oxides/oxyhydroxides in operating conditions.
    if "O" not in elems:
        score *= 0.75

    return _clamp(score)


def score_her_electrocatalyst(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    amounts = _amounts(composition)
    score = 0.0
    if elems & {"Pt", "Pd", "Ni", "Co", "Mo", "W", "Fe", "S", "Se", "P"}:
        score += 0.5
    if not np.isnan(band_gap) and band_gap < 1.5:
        score += 0.25
    if not np.isnan(formation_energy) and formation_energy < -0.5:
        score += 0.1
    if "Pt" in elems:
        score *= 0.6

    # Cap oxide-dominant compositions without chalcogenide/phosphide motifs.
    chalcogenides = amounts.get("S", 0.0) + amounts.get("Se", 0.0) + amounts.get("P", 0.0)
    oxygen = amounts.get("O", 0.0)
    if oxygen > chalcogenides and chalcogenides == 0:
        score = min(score, 0.5)

    return _clamp(score)


def score_co2_reduction(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if elems & {"Cu", "Ag", "Au", "Sn", "Bi", "In", "Zn", "Ni", "Fe", "Co"}:
        score += 0.45
    if not np.isnan(band_gap) and band_gap < 2.5:
        score += 0.2
    if elems & {"N", "S", "P"}:
        score += 0.15
    if not np.isnan(formation_energy) and formation_energy < -0.5:
        score += 0.1
    if elems & {"Au", "Ag"}:
        score *= 0.75

    # Cu is uniquely selective for deep CO2RR products; non-Cu systems are capped.
    if "Cu" not in elems:
        score = min(score, 0.5)

    return _clamp(score)


def score_photocatalyst_water_splitting(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    if "O" not in elems and "N" not in elems and "S" not in elems:
        return 0.0
    score = 0.0
    if _in_range(band_gap, 1.8, 3.2):
        score += 0.4
    elif _in_range(band_gap, 1.5, 3.6):
        score += 0.2
    if elems & {"Ti", "Fe", "Co", "Ni", "W", "Mo", "Bi", "Ta", "Nb"}:
        score += 0.3
    if not np.isnan(formation_energy) and formation_energy < -1.0:
        score += 0.2
    return _clamp(score)


def score_semiconductor_general(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    if np.isnan(band_gap):
        return 0.0
    score = 0.0
    if _in_range(band_gap, 0.5, 3.5):
        score += 0.45
    if _in_range(band_gap, 0.9, 1.8):
        score += 0.2
    if not np.isnan(e_hull) and e_hull <= 0.05:
        score += 0.2
    if not (elems & {"Hg", "Cd"}):
        score += 0.1
    return _clamp(score)


def score_led(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    if np.isnan(band_gap):
        return 0.0
    score = 0.0
    if _in_range(band_gap, 1.8, 3.2):
        score += 0.45
    elif _in_range(band_gap, 1.5, 3.6):
        score += 0.2
    if elems & {"Ga", "In", "N", "P", "As"}:
        score += 0.25
    if elems & {"Br", "I", "Cl", "Pb", "Sn"}:
        score += 0.15
    if "Pb" in elems:
        score *= 0.7
    return _clamp(score)


def score_photodetector(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if _in_range(band_gap, 0.1, 2.2):
        score += 0.45
    if elems & {"Se", "Te", "S", "Sb", "Bi", "Pb", "Hg", "In", "Ga"}:
        score += 0.25
    if not np.isnan(formation_energy) and formation_energy < -0.3:
        score += 0.15
    if elems & {"Hg", "Cd"}:
        score *= 0.65
    return _clamp(score)


def score_transparent_conductor(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    if np.isnan(band_gap):
        return 0.0
    score = 0.0
    if band_gap >= 3.0:
        score += 0.45
    if elems & {"In", "Sn", "Zn", "Ga", "Al", "Cd"} and "O" in elems:
        score += 0.35
    if not np.isnan(formation_energy) and formation_energy < -1.0:
        score += 0.1
    if "In" in elems:
        score *= 0.85
    return _clamp(score)


def score_ferroelectric(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if elems & {"Ti", "Nb", "Ta", "Zr"} and elems & {"Ba", "Pb", "Bi", "K", "Na", "Li"}:
        score += 0.5
    if "O" in elems:
        score += 0.2
    if band_gap >= 1.5:
        score += 0.15
    if not np.isnan(formation_energy) and formation_energy < -1.0:
        score += 0.1
    if "Pb" in elems:
        score *= 0.8
    return _clamp(score)


def score_piezoelectric(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if elems & {"Zn", "Al", "Ga", "In", "Ti", "Nb", "Ta"}:
        score += 0.35
    if elems & {"O", "N", "S"}:
        score += 0.2
    if band_gap >= 2.0:
        score += 0.2
    if not np.isnan(formation_energy) and formation_energy < -0.8:
        score += 0.15
    return _clamp(score)


def score_topological_insulator(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    heavy = {"Bi", "Sb", "Te", "Se", "Sn", "Pb", "Hg"}
    score = 0.0
    if len(elems & heavy) >= 2:
        score += 0.5
    if _in_range(band_gap, 0.0, 0.4):
        score += 0.25
    if not np.isnan(formation_energy) and formation_energy < -0.2:
        score += 0.1
    return _clamp(score)


def score_permanent_magnet(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if elems & {"Fe", "Co", "Ni"}:
        score += 0.4
    if elems & {"Nd", "Sm", "Dy", "Tb", "Pr"}:
        score += 0.25
    if elems & {"B", "N", "C"}:
        score += 0.15
    if not np.isnan(band_gap) and band_gap < 0.5:
        score += 0.1
    return _clamp(score)


def score_soft_magnet(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if elems & {"Fe", "Co", "Ni"}:
        score += 0.45
    if elems & {"Si", "Al", "B", "P"}:
        score += 0.2
    if not np.isnan(band_gap) and band_gap < 0.2:
        score += 0.15
    if not np.isnan(formation_energy) and formation_energy < -0.2:
        score += 0.1
    return _clamp(score)


def score_magnetic_semiconductor(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if elems & {"Fe", "Co", "Mn", "Cr", "Ni", "V"}:
        score += 0.35
    if _in_range(band_gap, 0.3, 2.5):
        score += 0.3
    if elems & {"O", "S", "Se", "Te", "N"}:
        score += 0.15
    if not np.isnan(formation_energy) and formation_energy < -0.5:
        score += 0.1
    return _clamp(score)


def score_tbc(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if elems & {"Zr", "Y", "La", "Ce", "Hf"} and "O" in elems:
        score += 0.45
    if not np.isnan(formation_energy) and formation_energy < -2.0:
        score += 0.2
    if band_gap >= 3.0:
        score += 0.1
    if elems & {"Si", "Al"}:
        score += 0.1
    return _clamp(score)


def score_thermal_interface(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if elems & {"C", "B", "N", "Al", "Cu", "Ag", "Si"}:
        score += 0.4
    if not np.isnan(band_gap) and (band_gap < 0.2 or band_gap > 3.0):
        score += 0.2
    if not np.isnan(formation_energy) and formation_energy < -0.3:
        score += 0.1
    if elems & {"Ag", "Au"}:
        score *= 0.8
    return _clamp(score)


def score_hard_coating(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if elems & {"Ti", "Cr", "V", "W", "Mo", "Ta", "Hf"}:
        score += 0.35
    if elems & {"N", "C", "B"}:
        score += 0.3
    if not np.isnan(formation_energy) and formation_energy < -1.0:
        score += 0.2
    if not np.isnan(band_gap) and band_gap > 0.2:
        score += 0.05
    return _clamp(score)


def score_corrosion_resistant(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if elems & {"Cr", "Al", "Ti", "Ni", "Mo", "Si"}:
        score += 0.35
    if "O" in elems:
        score += 0.2
    if not np.isnan(formation_energy) and formation_energy < -1.0:
        score += 0.2
    if elems & {"Cu", "Zn"}:
        score += 0.05
    return _clamp(score)


def score_refractory(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if elems & {"Hf", "Ta", "W", "Mo", "Nb", "Zr", "Ti"}:
        score += 0.4
    if elems & {"C", "N", "B", "O"}:
        score += 0.25
    if not np.isnan(formation_energy) and formation_energy < -1.5:
        score += 0.2
    return _clamp(score)


def score_superconductor(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0

    cuprate = ("Cu" in elems and "O" in elems and bool(elems & {"Ba", "Sr", "La"}))
    pnictide = ("Fe" in elems and bool(elems & {"As", "P"}))
    hydride = ("H" in elems and bool(elems & {"La", "Y", "Ce"}))
    conventional = bool(elems & {"Nb", "V", "Pb", "In", "Al", "Mo"})
    has_indicator = cuprate or pnictide or hydride or conventional

    if cuprate:
        score += 0.42
    if pnictide:
        score += 0.32
    if hydride:
        score += 0.28
    if conventional:
        score += 0.25

    if not np.isnan(band_gap) and band_gap < 0.2:
        score += 0.2
    if elems & {"O", "As", "Se", "Te"}:
        score += 0.15
    if not np.isnan(formation_energy) and formation_energy < -0.3:
        score += 0.1

    # If it's only a generic metallic chalcogenide without known SC indicators, cap it.
    is_metallic = (not np.isnan(band_gap)) and band_gap < 0.2
    has_chalcogenide = bool(elems & {"S", "Se", "Te"})
    if (not has_indicator) and is_metallic and has_chalcogenide:
        score = min(score, 0.4)

    # Bonus for high-Tc-associated element co-occurrence.
    if len(elems & {"Ba", "Sr", "La", "Y", "Cu", "Nb"}) >= 2:
        score += 0.12

    return _clamp(score)


def score_radiation_detector(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if elems & {"Tl", "I", "Br", "Cl", "Cs", "Na", "La", "Lu", "Y", "Gd"}:
        score += 0.35
    if elems & {"Pb", "Bi", "Cd", "Zn", "Se", "Te"}:
        score += 0.25
    if _in_range(band_gap, 1.5, 4.5):
        score += 0.2
    if not np.isnan(formation_energy) and formation_energy < -0.5:
        score += 0.1
    return _clamp(score)


def score_sofc_electrolyte(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if "O" in elems:
        score += 0.25
    if elems & {"Zr", "Ce", "La", "Y", "Sc", "Gd", "Sm"}:
        score += 0.35
    if band_gap >= 2.5:
        score += 0.15
    if not np.isnan(formation_energy) and formation_energy < -1.5:
        score += 0.15
    return _clamp(score)


def score_multiferroic(composition, structure, band_gap, formation_energy, e_hull, features):
    elems = _elements(composition)
    score = 0.0
    if elems & {"Bi", "Pb", "Ba", "La"}:
        score += 0.2
    if elems & {"Fe", "Mn", "Co", "Ni", "Cr"}:
        score += 0.25
    if elems & {"Ti", "Nb", "Ta", "Zr"}:
        score += 0.2
    if "O" in elems:
        score += 0.15
    if _in_range(band_gap, 1.0, 3.5):
        score += 0.1
    return _clamp(score)


SCORING_FUNCTIONS = {
    "battery_cathode_liion": score_battery_cathode_liion,
    "battery_anode": score_battery_anode,
    "battery_cathode_naion": score_battery_cathode_naion,
    "solid_electrolyte": score_solid_electrolyte,
    "hydrogen_storage": score_hydrogen_storage,
    "solar_singlejunction": score_solar_absorber_singlejunction,
    "solar_tandem": score_solar_absorber_tandem,
    "thermoelectric": score_thermoelectric,
    "oer_electrocatalyst": score_oer_electrocatalyst,
    "her_electrocatalyst": score_her_electrocatalyst,
    "co2_reduction": score_co2_reduction,
    "photocatalyst_h2o": score_photocatalyst_water_splitting,
    "semiconductor": score_semiconductor_general,
    "led": score_led,
    "photodetector": score_photodetector,
    "transparent_conductor": score_transparent_conductor,
    "ferroelectric": score_ferroelectric,
    "piezoelectric": score_piezoelectric,
    "topological_insulator": score_topological_insulator,
    "permanent_magnet": score_permanent_magnet,
    "soft_magnet": score_soft_magnet,
    "magnetic_semiconductor": score_magnetic_semiconductor,
    "thermal_barrier": score_tbc,
    "thermal_interface": score_thermal_interface,
    "hard_coating": score_hard_coating,
    "corrosion_resistant": score_corrosion_resistant,
    "refractory": score_refractory,
    "superconductor": score_superconductor,
    "radiation_detector": score_radiation_detector,
    "sofc_electrolyte": score_sofc_electrolyte,
    "multiferroic": score_multiferroic,
}


def score_all_applications(composition, structure, band_gap, formation_energy, e_hull, features):
    return {
        name: func(composition, structure, band_gap, formation_energy, e_hull, features)
        for name, func in SCORING_FUNCTIONS.items()
    }


def apply_application_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _score_row(row: pd.Series) -> dict[str, float]:
        formula = str(row.get(FORMULA_COL, ""))
        try:
            composition = Composition(formula)
        except Exception:
            return {name: 0.0 for name in SCORING_FUNCTIONS}
        band_gap = _num(row.get("Bandgap"))
        formation_energy = _num(row.get("Formation Energy Per Atom"))
        e_hull = _num(row.get("Decomposition Energy Per Atom"))
        features = row.to_dict()
        return score_all_applications(
            composition=composition,
            structure=None,
            band_gap=band_gap,
            formation_energy=formation_energy,
            e_hull=e_hull,
            features=features,
        )

    scores_df = out.apply(_score_row, axis=1, result_type="expand")
    scores_df = scores_df.add_prefix("score_")
    out = pd.concat([out, scores_df], axis=1)

    score_cols = list(scores_df.columns)
    out["top_application"] = out[score_cols].idxmax(axis=1).str.replace("score_", "", regex=False)
    out["best_score"] = out[score_cols].max(axis=1).round(3)
    return out
