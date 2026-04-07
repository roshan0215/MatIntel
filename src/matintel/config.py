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
    "Supercapacitor Electrode": "score_supercapacitor_electrode",
    "Redox Flow Battery Electrolyte": "score_redox_flow_battery_electrolyte",
    "Hydrogen Storage": "score_hydrogen_storage",
    "Sodium-Sulfur Battery Electrolyte/Electrode": "score_sodium_sulfur_battery",
    "Solar Absorber (Single Junction)": "score_solar_singlejunction",
    "Solar Absorber (Tandem Top Cell)": "score_solar_tandem",
    "Thermoelectric": "score_thermoelectric",
    "Photovoltaic Perovskite Stabiliser": "score_photovoltaic_perovskite_stabiliser",
    "Luminescent Solar Concentrator (LSC) Material": "score_luminescent_solar_concentrator",
    "OER Electrocatalyst (Water Splitting)": "score_oer_electrocatalyst",
    "HER Electrocatalyst (Green Hydrogen)": "score_her_electrocatalyst",
    "CO2 Reduction Catalyst (CO2RR)": "score_co2_reduction",
    "Photocatalyst (Water Splitting)": "score_photocatalyst_h2o",
    "Nitrogen Reduction Reaction (NRR) Catalyst": "score_nitrogen_reduction_catalyst",
    "Methane Activation Catalyst": "score_methane_activation_catalyst",
    "NOx Reduction Catalyst (SCR)": "score_nox_reduction_catalyst",
    "Semiconductor (General)": "score_semiconductor",
    "LED / Light Emitter": "score_led",
    "Photodetector": "score_photodetector",
    "Transparent Conductor": "score_transparent_conductor",
    "Ferroelectric": "score_ferroelectric",
    "Piezoelectric": "score_piezoelectric",
    "Phase Change Memory (PCM)": "score_phase_change_memory",
    "High-k Dielectric (Gate Oxide)": "score_highk_dielectric",
    "Nonlinear Optical (NLO) Material": "score_nonlinear_optical",
    "Permanent Magnet": "score_permanent_magnet",
    "Soft Magnet": "score_soft_magnet",
    "Magnetic Semiconductor / Spintronics": "score_magnetic_semiconductor",
    "Spintronic MTJ (Magnetic Tunnel Junction)": "score_spintronic_mtj",
    "Thermal Barrier Coating (TBC)": "score_thermal_barrier",
    "Thermal Interface Material (TIM)": "score_thermal_interface",
    "Hard Coating / Wear Resistant": "score_hard_coating",
    "Corrosion Resistant Coating": "score_corrosion_resistant",
    "Refractory / UHTC": "score_refractory",
    "Qubit Host Material": "score_qubit_host",
    "Topological Insulator": "score_topological_insulator",
    "Topological Qubit / Majorana Host": "score_topological_qubit_majorana",
    "Superconductor": "score_superconductor",
    "Radiation Detector / Scintillator": "score_radiation_detector",
    "Multiferroic": "score_multiferroic",
    "Biodegradable Implant": "score_biodegradable_implant",
    "Bone Scaffold / Hydroxyapatite Analog": "score_bone_scaffold",
    "Antibacterial Coating": "score_antibacterial_coating",
    "CO2 Capture Sorbent": "score_co2_capture_sorbent",
    "Desalination Membrane Material": "score_desalination_membrane",
    "Photocatalytic Pollutant Degradation": "score_photocatalytic_pollutant_degradation",
    "Nuclear Fuel Cladding": "score_nuclear_fuel_cladding",
    "Tritium Breeder Material": "score_tritium_breeder",
    "Radiation Shielding": "score_radiation_shielding",
    "Nuclear Waste Immobilisation": "score_nuclear_waste_immobilisation",
    "Battery Separator": "score_battery_separator",
    "Liquid Battery Electrolyte Component": "score_liquid_battery_electrolyte_component",
    "Solar Thermal Absorber": "score_solar_thermal_absorber",
    "Anti-Reflection Coating": "score_anti_reflection_coating",
    "Selective Hydrogenation Catalyst": "score_selective_hydrogenation_catalyst",
    "Fischer-Tropsch Catalyst": "score_fischer_tropsch_catalyst",
    "Dehydrogenation / LOHC Catalyst": "score_dehydrogenation_lohc_catalyst",
    "Memristor / Neuromorphic Computing Material": "score_memristor_neuromorphic",
    "2D Material": "score_material_2d",
    "Optical Fibre / Waveguide Material": "score_optical_fibre_waveguide",
    "Lightweight Structural Material": "score_lightweight_structural",
    "Shape Memory Alloy (SMA)": "score_shape_memory_alloy",
    "Metallic Glass / Amorphous Metal": "score_metallic_glass",
    "Superalloy": "score_superalloy",
    "Hydrogen Embrittlement Resistant Steel": "score_hydrogen_embrittlement_resistant_steel",
    "Photocatalytic CO2 Reduction": "score_photocatalytic_co2_reduction",
    "VOC Decomposition Catalyst": "score_voc_decomposition_catalyst",
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
