from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

RAW_CSV = RAW_DIR / "stable_materials_summary.csv"
WORKING_CSV = PROCESSED_DIR / "working_dataset.csv"
FEATURED_CSV = PROCESSED_DIR / "featured_dataset.csv"
SCORED_CSV = PROCESSED_DIR / "scored_dataset.csv"

MATERIAL_ID_COL = "MaterialId"
FORMULA_COL = "Reduced Formula"

APP_LABELS = {
    "Battery Cathode (Li-ion)": "score_battery_cathode_liion",
    "Battery Anode": "score_battery_anode",
    "Battery Cathode (Na-ion)": "score_battery_cathode_naion",
    "Solid Electrolyte": "score_solid_electrolyte",
    "Hydrogen Storage": "score_hydrogen_storage",
    "Solar Absorber - Single Junction": "score_solar_singlejunction",
    "Solar Absorber - Tandem Top Cell": "score_solar_tandem",
    "Thermoelectric": "score_thermoelectric",
    "OER Electrocatalyst (Water Splitting)": "score_oer_electrocatalyst",
    "HER Electrocatalyst (Green Hydrogen)": "score_her_electrocatalyst",
    "CO2 Reduction Catalyst": "score_co2_reduction",
    "Photocatalyst (Water Splitting)": "score_photocatalyst_h2o",
    "Semiconductor (General)": "score_semiconductor",
    "LED / Light Emitter": "score_led",
    "Photodetector": "score_photodetector",
    "Transparent Conductor": "score_transparent_conductor",
    "Ferroelectric": "score_ferroelectric",
    "Piezoelectric": "score_piezoelectric",
    "Topological Insulator": "score_topological_insulator",
    "Permanent Magnet": "score_permanent_magnet",
    "Soft Magnet": "score_soft_magnet",
    "Magnetic Semiconductor / Spintronics": "score_magnetic_semiconductor",
    "Thermal Barrier Coating": "score_thermal_barrier",
    "Thermal Interface Material": "score_thermal_interface",
    "Hard Coating / Wear Resistant": "score_hard_coating",
    "Corrosion Resistant Coating": "score_corrosion_resistant",
    "Refractory / UHTC": "score_refractory",
    "Superconductor": "score_superconductor",
    "Radiation Detector / Scintillator": "score_radiation_detector",
    "Solid Oxide Fuel Cell Electrolyte": "score_sofc_electrolyte",
    "Multiferroic": "score_multiferroic",
}

DEFAULT_ELEMENT_PRICE_USD_KG = {
    "Li": 6.0,
    "Co": 33.0,
    "Ni": 14.0,
    "Mn": 2.0,
    "Fe": 0.1,
    "Cu": 8.5,
    "Al": 2.0,
    "Si": 2.5,
    "Ti": 11.0,
    "V": 30.0,
    "Cr": 9.0,
    "Zn": 2.5,
    "Ga": 220.0,
    "Ge": 1000.0,
    "Se": 21.0,
    "Nb": 42.0,
    "Mo": 40.0,
    "In": 167.0,
    "Sn": 26.0,
    "Sb": 6.5,
    "Te": 63.0,
    "Nd": 40.0,
    "Sm": 14.0,
    "Gd": 28.0,
    "Dy": 220.0,
    "Hf": 900.0,
    "W": 35.0,
    "Re": 3500.0,
    "Pt": 31000.0,
    "Pd": 49000.0,
    "Rh": 147000.0,
}

TRANSITION_METALS = {"Mn", "Fe", "Co", "Ni", "V", "Cr", "Cu", "Ti", "Zr"}
MAGNETIC_ELEMENTS = {"Fe", "Co", "Ni", "Nd", "Sm", "Gd", "Dy"}
EXPENSIVE_ELEMENTS = {"Pt", "Pd", "Rh", "Ir", "Ru", "Os", "Au", "Re"}
TOXIC_ELEMENTS = {"Hg", "Cd", "Pb", "As", "Tl"}
CRITICAL_MINERALS = {
    "Al", "Sb", "As", "Ba", "Be", "Bi", "Ce", "Cs", "Cr", "Co", "Dy", "Er",
    "Eu", "Gd", "Ga", "Ge", "Hf", "Ho", "In", "Ir", "La", "Li", "Lu", "Mg",
    "Mn", "Nd", "Ni", "Nb", "Pr", "Rh", "Ru", "Sm", "Sc", "Ta", "Te", "Tb",
    "Tl", "Tm", "Sn", "Ti", "W", "V", "Yb", "Y", "Zn", "Zr",
}
