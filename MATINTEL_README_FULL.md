# MatIntel — Comprehensive Materials Intelligence Platform

**End-to-end computational materials screening and discovery platform** for identifying high-potential materials across 56 application domains. Combines domain-specific scoring with real-world viability assessment and AI-driven synthesizability predictions via CLscore.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Input Data Schema](#input-data-schema)
4. [The 56 Scoring Categories](#the-56-scoring-categories)
   - [Domain 1: Energy Storage](#domain-1-energy-storage)
   - [Domain 2: Energy Conversion](#domain-2-energy-conversion)
   - [Domain 3: Catalysis](#domain-3-catalysis)
   - [Domain 4: Electronics & Optoelectronics](#domain-4-electronics--optoelectronics)
   - [Domain 5: Magnetics & Spintronics](#domain-5-magnetics--spintronics)
   - [Domain 6: Thermal & Structural Coatings](#domain-6-thermal--structural-coatings)
   - [Domain 7: Quantum & Emerging Technologies](#domain-7-quantum--emerging-technologies)
   - [Domain 8: Biomedical & Environmental](#domain-8-biomedical--environmental)
5. [Viability Scoring](#viability-scoring)
6. [CLscore: Synthesizability Prediction](#clscore-synthesizability-prediction)
7. [Interactive Streamlit Dashboard](#interactive-streamlit-dashboard)
8. [Pipeline Architecture](#pipeline-architecture)
9. [Installation & Setup](#installation--setup)
10. [Running the System](#running-the-system)
11. [File Outputs & Data Structures](#file-outputs--data-structures)
12. [Advanced Usage](#advanced-usage)
13. [Technical Implementation Details](#technical-implementation-details)

---

## System Overview

MatIntel solves a critical problem in materials discovery: **given millions of computationally stable crystal structures, which ones are actually viable for specific applications and practically synthesizable?**

### Key Features

- **56 application-specific scorers** driven by rigorous materials science principles (band gap windows, crystal symmetry requirements, element chemistry, formation energy thresholds)
- **Viability multiplier** incorporating cost, elemental abundance, supply chain risk, radioactive exclusion, and rare-earth penalties
- **CLscore synthesis predictor** using deep graph neural networks (KAIST Synthesizability-PU-CGCNN) to estimate likelihood of successful synthesis
- **Interactive Streamlit dashboard** for filtering, ranking, and exploring candidates
- **Resume-safe batch processing** for scoring massive datasets with checkpointing
- **Multi-database support**: GNoME, Materials Project, AFLOW, JARVIS-DFT, OQMD, NOMAD

### Data Pipeline

```
Raw CSV (GNoME / MP / AFLOW / JARVIS / OQMD / NOMAD)
  ↓
Working Dataset (quality checks, formula validation, schema normalisation)
  ↓
Featured Dataset (element properties, structure metrics via matminer)
  ↓
Scored Dataset (56-category scores + viability + CLscore)
  ↓
Interactive Dashboard (filter, rank, export)
```

---

## Input Data Schema

MatIntel expects the following columns from source datasets. Optional columns improve scoring fidelity when present.

### Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `MaterialId` | string | Unique material identifier | `000006a8c4` |
| `Reduced Formula` | string | Reduced chemical formula | `Cs(ZrS2)3` |
| `Elements` | list[str] | Element symbols present | `['S', 'Zr', 'Cs']` |
| `Bandgap` | float (eV) | DFT band gap (GGA or HSE) | `0.0023` |
| `Formation Energy Per Atom` | float (eV/atom) | DFT formation energy | `-1.9058` |

### Strongly Recommended Columns

| Column | Type | Description | Used By |
|--------|------|-------------|---------|
| `Space Group` | string | Hermann–Mauguin symbol | `C2/m` | Ferroelectric, Piezoelectric, Multiferroic |
| `Space Group Number` | int | ITA number (1–230) | `12` | Symmetry-based scorers |
| `Crystal System` | string | Bravais lattice system | `monoclinic` | Structure family detection |
| `Point Group` | string | Crystallographic point group | `2/m` | Polar/non-centrosymmetric detection |
| `Decomposition Energy Per Atom` | float (eV/atom) | Distance above convex hull | `-0.1625` | Stability filters |
| `Density` | float (g/cm³) | Structural density | `3.2834` | Radiation detection, shielding |
| `NSites` | int | Atoms per unit cell | `10` | Complexity penalties |
| `Volume` | float (Å³) | Unit cell volume | `302.92` | Packing density proxies |

### Optional Enhancement Columns

| Column | Type | Source | Benefit |
|--------|------|--------|---------|
| `Magnetic Moment` | float (μB/f.u.) | MP, JARVIS | Magnet scorers |
| `Dielectric Constant` | float | MP, JARVIS | Ferroelectric, TIM scorers |
| `Elastic Modulus` | float (GPa) | MP | Hard coating, structural scorers |
| `Seebeck Coefficient` | float (μV/K) | JARVIS | Thermoelectric scorer |
| `Effective Mass (electron)` | float (mₑ) | JARVIS | Solar, LED, semiconductor |
| `Absorption Coefficient` | array | JARVIS optical | Solar absorber (replaces band gap proxy) |
| `Thermal Conductivity` | float (W/m·K) | JARVIS, Phono3py | Thermoelectric, TBC, TIM |
| `Dimensionality` | string | GNoME Cheon | 2D material scorer |
| `Is Train` | bool | GNoME | Provenance tracking |
| `source` | string | Pipeline | Raw origin label (`MP_synthesized`, `Experimental`, `JARVIS_ICSD`) |
| `is_experimental` | bool | Pipeline | **True = Experimental (NOT already synthesized); False = Synthesized (already synthesized/known)** |

### Provenance Semantics (Important)

- `Experimental` means the candidate is **not already present in synthesized reference datasets** and is treated as a novel/unsynthesized prediction.
- `Synthesized` means the material is **already reported as synthesized** in curated reference datasets (for example Materials Project / ICSD-linked sources).
- In short: `Experimental != Synthesized`.

### Schema Normalisation

The pipeline automatically handles:
- Missing `Bandgap` → scorer returns 0.0 or applies conservative estimate
- Missing `Space Group` → symmetry-dependent scorers use element-only heuristics
- Missing `Formation Energy` → stability bonuses skipped, not penalised
- Mixed-type columns → cast to float with NaN propagation
- Formula parsing failures → pymatgen fallback, then skip with warning

---

## The 56 Scoring Categories

Each material receives a score of **0.0–1.0** for each category, then multiplied by a **viability score** (0.0–1.0).

**General scoring principles applied across all categories:**
- Scores are **additive within a scorer** up to a cap of 1.0
- **Multiplicative penalties** can bring score below sum of bonuses
- **Hard returns** (score = 0.0) applied when a mandatory element or property is absent
- All thresholds are based on peer-reviewed experimental benchmarks, not arbitrary cutoffs
- Where band gap is used, GGA values are assumed (typically underestimated by ~40% vs experiment for semiconductors; HSE values should use tighter windows if available)


---

## Domain 1: Energy Storage

### 1. Battery Cathode (Li-ion)

**Scientific Context**: The positive electrode in a lithium-ion cell. During discharge, Li⁺ ions deintercalate from the anode and intercalate into the cathode, while electrons flow through the external circuit. The cathode determines cell voltage (from the redox potential of the transition metal), capacity (from how many Li sites are accessible), and rate capability (from Li⁺ diffusivity and electronic conductivity). The three dominant commercial families are layered oxides (LiCoO₂, NMC, NCA), spinels (LiMn₂O₄), and polyanionic compounds (LiFePO₄). The field is moving toward higher-Ni NMC (NMC811, NMC90) and LFP/LMFP for cost reasons.

**Key Requirements**:
- Li as the working ion (mandatory)
- Transition metal redox center(s) accessible in the 2.5–4.5 V window vs Li/Li⁺
- Electronic conductivity ≥ 10⁻⁴ S/cm (band gap should allow polaronic or band conduction)
- Li⁺ diffusivity ≥ 10⁻¹⁴ cm²/s (structural channels or layers)
- Thermodynamic stability under delithiation (low decomposition energy)
- Polyanionic frameworks (PO₄³⁻, SiO₄⁴⁻, SO₄²⁻) confer voltage stability via inductive effect

**Scoring Logic**:
```
Base score = 0.0

Mandatory check:
  if Li not in elements → return 0.0

Transition metal redox (pick highest applicable):
  +0.35 if Fe, Mn, Co, Ni, V, or Cr present (primary redox metals)
  +0.20 if Ti, Cu, Mo, or W present (secondary, lower voltage)
  +0.10 if only Zn, Al, Mg (non-redox, structural only — partial credit)

Framework bonus (additive):
  +0.22 if polyanionic framework: P+O, Si+O, S+O, B+O all present
  +0.12 if layered oxide: transition metal + O, no polyanionic anions
  +0.08 if spinel structure: space group Fd-3m (No. 227) detected

Band gap scoring:
  +0.25 if 0.01–2.5 eV (semiconducting to small-gap; polaronic conduction OK)
  +0.12 if 2.5–3.5 eV (marginal; needs doping or carbon coating)
  +0.00 if >3.5 eV (insulating; rate performance severely limited)
  +0.00 if 0.0 eV (metallic; good conductor but usually not intercalation host)

Stability bonus:
  +0.10 if formation energy < −2.0 eV/atom (thermodynamically robust)
  +0.05 if formation energy −1.0 to −2.0 eV/atom
  +0.00 if formation energy > −0.5 eV/atom (too metastable)

Decomposition penalty:
  ×0.80 if decomposition energy > 0.1 eV/atom (unstable against competing phases)
  ×0.60 if decomposition energy > 0.3 eV/atom (likely phase separates on cycling)

Element penalties:
  ×0.70 if F present without P (fluoride dissolution risk in liquid electrolyte)
  ×0.60 if S present without P (polysulfide dissolution)
  ×0.50 if toxic heavy metals (Tl, Hg, Cd, As, Pb) present
  ×0.85 if Co > 50 at% of transition metals (supply risk penalty here separate from viability)

final_score = min(sum_of_bonuses × product_of_penalties, 1.0)
```

**Commercial Benchmarks**: LiCoO₂ (3.9 V, 140 mAh/g), LiFePO₄ (3.4 V, 170 mAh/g), NMC811 (3.7 V, 200 mAh/g)

---

### 2. Battery Anode

**Scientific Context**: The negative electrode stores working ions at low electrochemical potential, maximising cell voltage. Graphite (372 mAh/g) dominates commercially, but silicon (theoretical 3579 mAh/g as Li₁₅Si₄) is the next generation, held back by ~300% volume expansion. Conversion-type anodes (Fe₂O₃, Fe₃O₄, CoO) offer high capacity via M + xLi⁺ + xe⁻ → Li_xM but suffer from large voltage hysteresis. Alloying anodes (Sn, Sb, Bi, Ge) sit between graphite and Si in capacity and volume expansion.

**Key Requirements**:
- Must NOT contain the working ion (Li or Na) — anode is in discharged state
- Near-metallic or metallic conductivity (band gap < 0.5 eV preferred; conversion types can be semiconducting)
- Alloying elements (Si, Sn, Sb, Ge, Bi, P, Al) for high capacity
- OR conversion-type transition metal + anion (oxide, sulfide, fluoride)
- Volume expansion tolerance (structural dimensionality, soft lattice)
- Low average delithiation voltage (< 1.0 V vs Li/Li⁺)

**Scoring Logic**:
```
Base score = 0.0

Mandatory disqualifiers:
  if Li present → ×0.5 (already lithiated; wrong state for anode screening)
  if Na present → ×0.5 (same logic for Na-ion)
  if radioactive elements → return 0.0

Alloying anode elements (additive, pick present):
  +0.40 if Si present (highest capacity theoretical)
  +0.35 if Sn, Sb, Bi, or Ge present
  +0.30 if P or Al present (moderate capacity alloys)
  +0.15 for Fe-Si combination specifically (well-studied FeSi₂ family, good cycle life)

Conversion-type detection:
  +0.25 if transition metal (Fe, Co, Ni, Mn, Cu) + anion (O, S, F, N) present
    and no alloying elements above already scored 0.35+

Electronic character:
  +0.20 if band gap < 0.1 eV (metallic — ideal electron conductor)
  +0.15 if band gap 0.1–0.5 eV (near-metallic — acceptable)
  +0.05 if band gap 0.5–1.5 eV (semiconducting — conversion types OK here)
  +0.00 if band gap > 1.5 eV (too insulating for anode kinetics)

Stability:
  +0.10 if formation energy < −1.0 eV/atom
  +0.05 if decomposition energy < 0.05 eV/atom (on or near hull)

Structure complexity penalty:
  ×0.85 if NSites > 20 (complex unit cells → slow Li diffusion)
  ×0.70 if NSites > 50

final_score = min(total × penalties, 1.0)
```

---

### 3. Battery Cathode (Na-ion)

**Scientific Context**: Sodium-ion batteries avoid Li, Co, and expensive rare earths by using Na⁺ (ionic radius 1.02 Å vs 0.76 Å for Li⁺) as the working ion. CATL commercialised Na-ion in 2023. The larger Na⁺ requires more open structural frameworks. The most promising families are O3/P2 layered oxides (NaNiO₂, NaMnO₂ variants), Prussian blue analogues (NaₓMFe(CN)₆), and polyanionic compounds (Na₃V₂(PO₄)₃, NASICON). Fe and Mn are strongly preferred over Co and Ni for cost reasons.

**Scoring Logic**:
```
Mandatory check:
  if Na not in elements → return 0.0

Transition metal redox (additive):
  +0.40 if Fe or Mn present (preferred — abundant, cheap, good Na-ion voltage)
  +0.20 if V or Ti present (good voltage but V has supply concerns)
  +0.15 if Co, Ni, Cu, or Cr present (work but add cost/supply risk)

Framework type:
  +0.22 if polyanionic (P+O, Si+O, S+O present) — NASICON/olivine type
  +0.15 if layered oxide (TM + O, hexagonal/trigonal crystal system)
  +0.12 if cyanide framework (C+N present) — Prussian blue analogue
  +0.08 if fluoride framework (F present with Na)

Band gap:
  +0.20 if 0.0–3.0 eV
  +0.10 if 3.0–4.0 eV
  +0.00 if > 4.0 eV

Li-free bonus:
  +0.08 if Li not present (pure Na chemistry, no cross-contamination risk)

Stability:
  +0.08 if formation energy < −1.5 eV/atom
  +0.04 if decomposition energy < 0.05 eV/atom

Penalties:
  ×0.70 if Co > 20 at% of transition metals (cost penalty)
  ×0.60 if S present without P (dissolution)
```

---

### 4. Solid Electrolyte

**Scientific Context**: Replaces flammable liquid electrolyte in all-solid-state batteries (ASSBs). Must conduct Li⁺ or Na⁺ (ionic conductivity ≥ 10⁻⁴ S/cm at RT) while remaining electronically insulating (σₑ < 10⁻¹⁰ S/cm). Three main families: oxides (LLZO, NASICON, LISICON) — wide gap, stable vs Li metal, brittle; sulfides (Li₆PS₅Cl, Li₁₀GeP₂S₁₂) — high conductivity (~10⁻² S/cm), narrow gap, reactive with moisture; halides (Li₃InCl₆, Li₃YCl₆) — emerging, oxidatively stable. The wide band gap requirement is absolute — any electronic conductivity causes self-discharge.

**Scoring Logic**:
```
Mandatory checks:
  if Li not in elements AND Na not in elements → return 0.0
  if band gap < 1.5 eV → return 0.0 (too electronically conductive)
  if band gap < 2.0 eV → ×0.3 (marginal; only sulfides acceptable here)

Band gap bonuses (pick highest):
  +0.35 if ≥ 5.0 eV (excellent insulation, oxide-type)
  +0.30 if 4.0–5.0 eV (very good)
  +0.20 if 3.0–4.0 eV (adequate for sulfides)
  +0.10 if 2.0–3.0 eV (sulfide-type; conditional on S present)

Framework elements (additive):
  +0.28 if Zr, La, Al, Ta, Nb, or Ti present (oxide framework backbone)
    additional +0.05 if both Zr and (La or Al) present (LLZO/NASICON-like)
  +0.22 if S present + (P, Si, Ge, Sn, or As) (sulfide framework — argyrodite/LGPS)
  +0.18 if halide (Cl, Br, I) + (In, Y, Er, Sc, Yb, La) (halide electrolyte family)
  +0.10 if F present (NASICON fluoride variant)

Oxygen bonus:
  +0.08 if O present (oxide stability)

Stability:
  +0.12 if formation energy < −2.5 eV/atom (must be very stable at interface)
  +0.06 if decomposition energy < 0.02 eV/atom (on hull)

Penalties:
  ×0.60 if decomposition energy > 0.2 eV/atom (metastable — interface degradation)
  ×0.50 if Mn present (oxidises sulfide electrolytes)
  ×0.40 if transition metal with variable oxidation state in contact with S (reduction risk)
```

---

### 5. Supercapacitor Electrode

**Scientific Context**: Supercapacitors (ultracapacitors) store energy via electrostatic double-layer capacitance (EDLC) or fast surface redox (pseudocapacitance). EDLC requires maximising accessible surface area (activated carbon, 1000–3000 m²/g). Pseudocapacitance from RuO₂ (~1500 F/g theoretical), MnO₂ (~1370 F/g), MXenes (Ti₃C₂Tₓ) and conducting polymers. Metallic conductivity is mandatory — resistive losses at high charge/discharge rates are the dominant performance limiter.

**Scoring Logic**:
```
Electronic conductivity (mandatory for high-rate performance):
  if band gap > 1.0 eV → ×0.3 (severe rate penalty)
  +0.35 if band gap < 0.1 eV (metallic)
  +0.25 if band gap 0.1–0.5 eV (near-metallic)
  +0.10 if band gap 0.5–1.0 eV (marginal)

Pseudocapacitance metals (additive):
  +0.30 if Ru or Ir present (highest specific capacitance — RuO₂ benchmark)
  +0.25 if Mn present (MnO₂ — cheap pseudocapacitor)
  +0.20 if V, Mo, or W present (vanadium oxides, MoO₃)
  +0.18 if Ni or Co present (NiO, Co₃O₄ alkaline pseudocapacitance)
  +0.15 if Ti + C present (MXene-type — Ti₃C₂ family)
  +0.12 if Fe present (Fe₂O₃/Fe₃O₄ pseudocapacitance)

Structural features:
  +0.15 if layered structure (trigonal/hexagonal crystal system + TM + O/S)
  +0.10 if 2D dimensionality (Dimensionality field = '2D' or 'layered')
  +0.08 if oxide present (O in elements — surface hydroxyl groups for redox)

Stability:
  +0.08 if formation energy < −1.0 eV/atom
  +0.05 if decomposition energy < 0.1 eV/atom

Penalties:
  ×0.70 if NSites > 30 (complex unit cells rarely expose clean pseudocapacitive surfaces)
  ×0.50 if Tl, Hg, Cd, or As present (toxicity incompatible with consumer devices)
  ×0.60 if band gap > 2.0 eV AND no pseudocapacitive metal (pure insulator — useless)
```

---

### 6. Redox Flow Battery Electrolyte

**Scientific Context**: Flow batteries store energy in dissolved redox-active species pumped through an electrochemical cell. The vanadium redox flow battery (VRFB) dominates for grid storage. The electrolyte material (dissolved species, not membrane) must be soluble at high concentration (> 1 M), stable over thousands of cycles, and have a large cell voltage (difference between posolyte and negolyte redox potentials). Organic redox molecules are an emerging alternative. This scorer targets inorganic candidate materials.

**Scoring Logic**:
```
Solubility proxy (redox-active species must dissolve):
  +0.30 if V present (vanadium — VRFB benchmark, V²⁺/V³⁺/VO²⁺/VO₂⁺)
  +0.25 if Fe present (Fe²⁺/Fe³⁺, iron-chromium systems)
  +0.22 if Cr present (Cr²⁺/Cr³⁺, iron-chromium system)
  +0.20 if Mn or Ce present (Mn²⁺/Mn³⁺, Ce³⁺/Ce⁴⁺ posolyte candidates)
  +0.15 if Zn present (Zn/Br₂, zinc-bromine systems)
  +0.10 if Ti, Mo, or W present (emerging inorganic candidates)

Anion suitability:
  +0.15 if SO₄ proxy: S+O present (sulfate electrolyte — VRFB standard)
  +0.10 if Cl present (chloride electrolyte systems)
  +0.08 if NO₃ proxy: N+O present (nitrate systems)

Water stability proxy:
  +0.10 if formation energy < −1.5 eV/atom (thermodynamically stable vs hydrolysis)

Penalties:
  ×0.50 if highly insoluble oxides: Zr, Al, Si, Ti + O only (sparingly soluble)
  ×0.40 if toxic elements (As, Hg, Tl, Cd, Pb) (electrolyte leaks = environmental hazard)
  ×0.30 if radioactive → return 0.0
  ×0.70 if no redox-active transition metal present
```

---

### 7. Hydrogen Storage

**Scientific Context**: Reversible solid-state hydrogen storage for fuel cell vehicles. DOE targets: ≥ 6.5 wt% gravimetric capacity, ≥ 50 g/L volumetric, release temperature 60–120°C. Incumbent: pressurised H₂ (700 bar) or liquid H₂ (−253°C). Solid-state alternatives: metal hydrides (TiFe, LaNi₅), complex hydrides (NaBH₄, LiAlH₄), chemical hydrides (NH₃BH₃). The main trade-off is between gravimetric density (light metals needed) and thermodynamic reversibility (moderate binding energy −0.3 to −0.5 eV/H₂ required).

**Scoring Logic**:
```
Mandatory check:
  if H not in elements → return 0.0

Light metal carriers (gravimetric density):
  +0.30 if Mg present (MgH₂: 7.6 wt%, slow kinetics but classic benchmark)
  +0.25 if Li or Na present (LiH, NaH, borohydrides)
  +0.20 if Al present (AlH₃, alanates)
  +0.18 if Ca or K present (CaH₂, Ca(BH₄)₂)
  +0.10 if B present (borohydride family — highest gravimetric but poor reversibility)

Kinetic enhancement metals:
  +0.22 if Ti present (TiH₂ catalyst; activates surface for H₂ dissociation)
  +0.20 if Fe present (TiFe intermetallic — commercial AB-type)
  +0.18 if Ni present (LaNi₅ — most studied AB₅)
  +0.15 if V, Zr, or La present (AB₂ Laves phase family)
  +0.10 if Ce or Y present (rare-earth nickel hydrides — high capacity but heavy)
  +0.08 if Pd present (low barrier H₂ dissociation, excellent catalyst, expensive)

Thermodynamics proxy:
  +0.20 if formation energy −1.5 to −0.3 eV/atom (moderate binding — reversible)
  +0.10 if formation energy −0.3 to 0.0 eV/atom (weak binding — too easy to release)
  +0.05 if formation energy < −1.5 eV/atom (strong binding — hard to release, not ideal)

Stability:
  +0.05 if decomposition energy < 0.05 eV/atom

Penalties:
  ×0.60 if no light metals AND no kinetic metals (heavy-only hydrides — poor gravimetric)
  ×0.50 if Hg, Tl, Cd, or As present (toxic)
  ×0.70 if NSites > 30 (structural complexity hurts diffusion kinetics)
```

---

### 8. Sodium-Sulfur Battery Electrolyte/Electrode

**Scientific Context**: High-temperature Na-S batteries (300°C) use molten Na anode and molten S cathode separated by a solid β-alumina ceramic electrolyte. These are mature grid-storage technology (NGK, Japan). Room-temperature Na-S is emerging but faces polysulfide dissolution issues similar to Li-S. This scorer targets β-alumina analogues and related sodium superionic conductors.

**Scoring Logic**:
```
Mandatory check:
  if Na not in elements → return 0.0
  if band gap < 2.0 eV → return 0.0 (must be electronically insulating)

Na-S electrolyte families:
  +0.40 if Al + O + Na present (β-alumina or β''-alumina family — THE benchmark)
  +0.30 if Zr + O + Na present (NASICON type for Na-S)
  +0.25 if Si + O + Na present (LISICON-like sodium conductor)
  +0.20 if Mg + Al + O + Na present (Mg-stabilised β''-alumina)

Band gap:
  +0.25 if ≥ 5.0 eV (high electrical insulation)
  +0.20 if 4.0–5.0 eV
  +0.10 if 3.0–4.0 eV

Stability:
  +0.15 if formation energy < −3.0 eV/atom (must survive 300°C repeated cycling)
  +0.08 if decomposition energy < 0.02 eV/atom

Penalties:
  ×0.60 if S present in electrolyte (electrolyte should not contain sulfur)
  ×0.50 if decomposition energy > 0.2 eV/atom
```


---

## Domain 2: Energy Conversion

### 9. Solar Absorber (Single Junction)

**Scientific Context**: The photovoltaic absorber layer must absorb sunlight efficiently and generate electron-hole pairs with minimal recombination. The Shockley-Queisser limit predicts maximum theoretical efficiency of ~33.7% at a band gap of ~1.34 eV under AM 1.5G illumination. Direct band gaps are strongly preferred: indirect gap materials (Si, Ge) require thicker absorber layers due to lower absorption coefficients. Key figures of merit: absorption coefficient α > 10⁴ cm⁻¹ near band edge, minority carrier diffusion length > grain size, p-type conductivity for most device architectures. The dominant thin-film technologies are CIGS (Cu(In,Ga)Se₂, ~23% record), CdTe (~22%), and emerging Cu₂ZnSn(S,Se)₄ (CZTS, ~13%). Perovskites have reached ~26% but face stability/Pb-toxicity challenges.

**Scoring Logic**:
```
Hard requirement:
  if band gap unavailable → return 0.0
  if band gap < 0.7 eV → return 0.0 (too much thermal generation; Voc too low)
  if band gap > 2.0 eV → return 0.0 (too little absorption of solar spectrum)

Gaussian band gap score (peak at 1.34 eV, FWHM ~0.6 eV):
  core_score = max(0.50 − |band_gap − 1.34| × 0.55, 0.15)
  Applied for band gap in [0.7, 2.0] eV

Material family bonuses (additive):
  +0.25 if chalcopyrite family: Cu + (In or Ga) + (S or Se) (≥ 3 of these elements)
  +0.22 if kesterite family: Cu + Zn + Sn + (S or Se) (≥ 3 of these elements)
  +0.18 if Cu + Bi + (S or Se) present (Cu-Bi chalcogenides — defect tolerant ns² family)
  +0.18 if Cu + Sb + (S or Se) present (Cu-Sb chalcogenides — same family)
  +0.15 if Ag + Bi + (S or Se) present (Ag-Bi chalcogenides)
  +0.12 if chalcogenide metals present: any of {Mo, W, Sb, Bi, Ge, Sn} + (S or Se)
  +0.10 if Cs or Rb present with Sn/Ge + halide (lead-free perovskite analog)
  +0.08 if Cu present generally (most working thin-film absorbers contain Cu)

ns² lone-pair bonus:
  +0.10 if Bi³⁺ or Sb³⁺ likely (Bi or Sb present with chalcogenide)
    (defect tolerance — lone pair raises VBM, disperses bands, reduces recombination)

Toxicity penalties:
  ×0.60 if Cd present (RoHS regulatory pressure; CdTe exemption narrowing)
  ×0.50 if As or Hg present (severe toxicity)
  ×0.70 if Pb present (REACH/RoHS regulatory pressure for non-perovskite uses)

Stability:
  +0.08 if formation energy < −1.0 eV/atom
  +0.05 if decomposition energy < 0.05 eV/atom

final_score = min((core_score + family_bonuses + stability_bonuses) × toxicity_penalties, 1.0)
```

**Example**: Ba₂Cu₂Bi₂SeS₅ — band gap 1.4247 eV → core_score ≈ 0.46; Cu present (+0.08); Bi + Se/S (+0.18 Cu-Bi chalcogenide); ns² bonus (+0.10); no toxic elements; formation energy bonus → raw score ≈ 1.0 ✓

---

### 10. Solar Absorber (Tandem Top Cell)

**Scientific Context**: In a two-terminal perovskite-silicon tandem (2025 record ~34.6%), the top cell must absorb photons above ~1.6 eV while being transparent below that energy to allow the Si bottom cell (~1.1 eV) to absorb the remainder. The ideal top cell band gap for a 2J tandem with Si is 1.67–1.75 eV (current-matched condition under AM 1.5G). Lead-halide perovskites (CsPbI₂Br, FA₀.₈Cs₀.₂PbI₂Br, ~1.67–1.77 eV) dominate but face Pb toxicity concerns and ion migration-induced instability. Pb-free alternatives (Sn-Pb mixed, Sb-based, Bi-based) are active research areas.

**Scoring Logic**:
```
Hard requirement:
  if band gap < 1.4 eV or > 2.2 eV → return 0.0

Gaussian band gap score (peak at 1.73 eV, FWHM ~0.35 eV):
  core_score = max(0.50 − |band_gap − 1.73| × 0.95, 0.15)

Perovskite structure bonus:
  +0.30 if halide perovskite ABC₃ indicators:
    halide (Cl, Br, I) + A-site (Cs, Rb, K, MA-proxy via formula) 
    + B-site (Pb, Sn, Ge, Bi, Sb, In, Ti)
  +0.15 if partial perovskite character (halide + B-site metals only, no A-site)

Pb-free bonus:
  +0.15 if no Pb present (strong regulatory and commercial driver)
  +0.10 if Sn or Ge present (Pb-replacement in perovskite B-site)
  +0.08 if Bi or Sb present as main cation (bismuth/antimony perovskite analogs)

Transparency window:
  +0.08 if no elements with strong sub-gap absorption (avoid narrow-gap metals)

Stability:
  +0.08 if formation energy < −1.5 eV/atom
  +0.05 if decomposition energy < 0.05 eV/atom

Penalties:
  ×0.70 if Pb present (regulatory — EU REACH, RoHS pressure intensifying)
  ×0.50 if Hg, Tl, or Cd present
  ×0.80 if I only (no Br/Cl) — pure iodide perovskites have phase instability at 1.73 eV
```

---

### 11. Thermoelectric

**Scientific Context**: Thermoelectrics convert heat directly to electricity (Seebeck effect) or vice versa (Peltier effect). Performance is quantified by the dimensionless figure of merit ZT = S²σT/κ, where S is the Seebeck coefficient (μV/K), σ is electrical conductivity (S/m), T is temperature (K), and κ is thermal conductivity (W/m·K). State-of-the-art materials reach ZT > 2.5 (SnSe single crystal, 773K). The band gap should be ~6–10 k_BT at the operating temperature for optimal power factor; at 300K that's ~0.15–0.26 eV, rising to ~0.5–0.8 eV at 800K. Heavy, anharmonically bonded atoms scatter phonons and reduce κ_lattice, which is why chalcogenides (Te, Se, S) and heavy pnictogens (Bi, Sb) dominate. Complex crystal structures (cage compounds, misfit layered, rattler modes) also suppress thermal conductivity.

**Scoring Logic**:
```
Band gap for optimal ZT (operating temperature dependent):
  +0.35 if 0.1–0.5 eV (optimal for room-temperature/low-temp thermoelectrics)
  +0.30 if 0.5–1.0 eV (optimal for mid-temperature, 500–800K)
  +0.20 if 0.05–0.1 eV (semimetallic — bipolar conduction penalty but high σ)
  +0.10 if 1.0–1.5 eV (too wide for low T; acceptable for high-T)
  +0.05 if > 1.5 eV (too insulating; requires heavy doping)
  +0.00 if < 0.05 eV (metallic; good σ but low S²)

Heavy element bonus (phonon scattering, low κ):
  heavy_elements = {Pb, Bi, Sb, Te, Se, Tl, In, Sn, Ge, Ba, Cs, I, Br, Ag, Hg, Cu}
  heavy_count = number of heavy elements present
  +min(heavy_count × 0.12, 0.38)

Key thermoelectric element bonuses:
  +0.12 if Te present (Bi₂Te₃ and PbTe families — commercial TE benchmark)
  +0.08 if Se present (chalcogenide low κ)
  +0.07 if S present (sulfide TE emerging class)
  +0.10 if Bi present (Bi₂Te₃, BiSbTe — commercial Peltier)
  +0.08 if Sb present (CoSb₃ skutterudite family)
  +0.07 if Ag present (AgSbTe₂, LAST compounds)
  +0.08 if halide present (I, Br) — soft, anharmonic bonding → low κ

Structural complexity bonus (low thermal conductivity via structural complexity):
  +0.08 if NSites > 15 (complex unit cell → additional phonon scattering)
  +0.05 if NSites > 30 (cage/clathrate-like complexity)

Penalties:
  ×0.40 if all elements have atomic mass < 40 (light elements → high κ → poor ZT)
  ×0.70 if band gap > 2.0 eV (poor power factor even with heavy atoms)
  ×0.60 if Tl present (highly toxic; regulatory barriers)
```

---

### 12. Photovoltaic Perovskite Stabiliser

**Scientific Context**: Lead-halide perovskites reach >26% PCE but degrade rapidly under moisture, heat, and illumination due to: (1) ion migration along grain boundaries; (2) phase transformation (e.g., α→δ CsPbI₃ at room temperature); (3) Pb²⁺ leaching. Stabiliser materials must form passivating interfaces, suppress ion migration, or act as encapsulants. Requirements: chemical stability, wide band gap (transparent), moisture resistance, and compatibility with low-temperature processing (< 150°C).

**Scoring Logic**:
```
Must be transparent (wide band gap):
  if band gap < 3.0 eV → ×0.3

Band gap:
  +0.35 if ≥ 5.0 eV (excellent transparency, oxide-type stabiliser)
  +0.25 if 3.5–5.0 eV
  +0.15 if 3.0–3.5 eV

Chemical stability:
  +0.25 if formation energy < −3.0 eV/atom (very stable — won't react with perovskite)
  +0.15 if formation energy −2.0 to −3.0 eV/atom
  +0.08 if decomposition energy < 0.01 eV/atom (on hull)

Moisture resistance proxy:
  +0.15 if F present (fluorides are moisture-resistant; LiF, MgF₂ used as interlayers)
  +0.12 if Al₂O₃ proxy: Al + O only (ALD Al₂O₃ is gold standard encapsulant)
  +0.10 if Si + O present (SiO₂ analogs — glass-like stability)
  +0.08 if Zr + O present (ZrO₂ — chemically inert)

Compatibility (no elements that react with perovskite):
  ×0.40 if Cu, Fe, or Co present (redox-active; can oxidise halide perovskite)
  ×0.50 if S or Se present (chalcogenides react with Pb halides)
  +0.05 if Cs, Rb, or K present (compatible A-site cations)
```

---

### 13. Luminescent Solar Concentrator (LSC) Material

**Scientific Context**: LSCs use fluorescent or phosphorescent materials embedded in a waveguide to redirect and concentrate sunlight onto small PV cells at the edges. Requirements: high photoluminescence quantum yield (PLQY > 70%), large Stokes shift (to avoid self-absorption), absorption in visible range, and stability under continuous illumination. Band gap must be in the visible (1.77–3.1 eV) with emission at lower energy (Stokes-shifted).

**Scoring Logic**:
```
Optical window (absorption):
  if band gap < 1.5 eV or > 3.5 eV → return 0.0

Band gap bonus:
  +0.35 if 1.77–3.1 eV (visible absorber)
  +0.20 if 1.5–1.77 eV (NIR; useful for bifacial LSC)
  +0.10 if 3.1–3.5 eV (UV absorber; partial solar spectrum)

Luminescent material families:
  +0.30 if Eu, Ce, Tb, or Dy present (rare-earth phosphors — high PLQY)
  +0.25 if Mn present in halide host (Mn²⁺ emission in halide perovskites; large Stokes shift)
  +0.20 if Cu + halide present (Cu⁺ luminescence in halide hosts)
  +0.18 if Cs + Sn or Cs + Pb + halide (perovskite quantum dot LSC — very active research)
  +0.15 if Ag + halide present (Ag⁺ luminescent centers)
  +0.10 if In + S or In + Se present (In-based quantum dot type)

Self-absorption mitigation proxy:
  +0.08 if heavy-atom host (Ba, Sr, Cs) with light luminescent dopant (Eu, Mn, Cu)
    (spatial separation of absorption and emission centers → Stokes shift)

Stability:
  +0.08 if formation energy < −2.0 eV/atom (photostability proxy)

Penalties:
  ×0.70 if Pb present (toxicity in consumer glazing applications)
  ×0.50 if Hg, Tl, Cd, or As present
  ×0.60 if no identified luminescent center
```


---

## Domain 3: Catalysis

### 14. OER Electrocatalyst (Water Splitting)

**Scientific Context**: Oxygen evolution reaction (OER): 2H₂O → O₂ + 4H⁺ + 4e⁻ (acidic) or 4OH⁻ → O₂ + 2H₂O + 4e⁻ (alkaline). Rate-limiting step in both PEM (acidic) and alkaline electrolysers. Overpotential minimum is ~0.37 V theoretically, but best catalysts achieve ~0.25 V excess. In acid (PEM), only Ir and Ru oxides are stable; in alkaline, Fe-Ni oxohydroxides are world-class. The key descriptor is the adsorption energy of *OH, *O, and *OOH intermediates on the surface — too strong or too weak and you fall off the volcano plot.

**Scoring Logic**:
```
Mandatory: metallic or near-metallic character
  if band gap > 2.5 eV → ×0.3 (insulators can't conduct electrons to electrode)

Electrocatalyst metal bonuses:
  +0.40 if Ir or Ru present (only stable OER catalysts in acid — PEM electrolyser)
  +0.30 if Fe, Co, or Ni present (best alkaline OER; NiFe-LDH benchmark)
  +0.20 if Mn present (Mn-based oxides — earth-abundant, moderate OER activity)
  +0.15 if Co alone present (Co₃O₄ alkaline OER)
  +0.10 if Cu present (emerging, lower activity but earth-abundant)
  +0.08 if Ti, V, Mo, or W present (support/synergy effects)

Oxide/hydroxide structure:
  +0.15 if O present (oxide surface — most OER catalysts are oxides/hydroxides)
  +0.10 if layered structure (trigonal/hexagonal crystal system) — LDH-type

Electronic character:
  +0.25 if band gap < 0.5 eV (metallic — essential for electron transfer)
  +0.10 if band gap 0.5–2.0 eV (semiconducting oxide — acceptable)
  ×0.3 if band gap > 2.5 eV

Structural bonuses:
  +0.08 if perovskite ABO₃ structure detected (LaCoO₃, BSCF family)
  +0.05 if spinel structure detected (Co₃O₄, Fe₃O₄)

Stability:
  +0.08 if formation energy < −2.0 eV/atom (chemical stability in aqueous environment)

Penalties:
  ×0.60 if Tl, Hg, Cd, or As present (toxic in aqueous catalyst)
  ×0.70 if only non-redox metals (Al, Si, Mg) — no active sites
```

---

### 15. HER Electrocatalyst (Green Hydrogen)

**Scientific Context**: Hydrogen evolution reaction (HER): 2H⁺ + 2e⁻ → H₂ (acid) or 2H₂O + 2e⁻ → H₂ + 2OH⁻ (alkaline). Pt is the benchmark (nearly zero overpotential), but at ~$31,000/kg it's economically unviable at scale. Earth-abundant alternatives: MoS₂ edge sites (~200 mV overpotential), CoP, Ni₂P, FeP, Mo₂C. The Sabatier principle dictates that the optimal catalyst has ΔG_H* ≈ 0 eV; Pt sits at −0.09 eV. Fe, Co, Ni sulfides and phosphides have been optimised to approach this.

**Scoring Logic**:
```
Noble metal bonus (highest activity):
  +0.40 if Pt or Pd present (benchmark HER catalysts)
  +0.30 if Rh, Ir, or Ru present (excellent but expensive)

Earth-abundant transition metal bonus:
  +0.30 if Mo or W present (MoS₂, WS₂ edge sites, Mo₂C)
  +0.28 if Ni or Co present (CoP, Ni₂P, NiMo alloys)
  +0.25 if Fe present (FeP, Fe₂P — lower activity but cheap)
  +0.20 if Cu present (Cu-based HER emerging)
  +0.15 if V or Cr present (supporting role)

Ligand/anion bonuses (surface active sites):
  +0.22 if S present + earth-abundant TM (sulfides — MoS₂ edge sites paradigm)
  +0.22 if P present + earth-abundant TM (phosphides — CoP, Ni₂P benchmark)
  +0.18 if N present + earth-abundant TM (nitrides — Mo₂N, W₂N)
  +0.15 if C present + TM (carbides — Mo₂C)
  +0.15 if Se present + earth-abundant TM (selenides, more conductive than sulfides)

Electronic character:
  +0.22 if band gap < 0.5 eV (metallic — essential for HER)
  +0.10 if band gap 0.5–1.5 eV (semiconducting sulfides/phosphides OK with edge sites)
  ×0.40 if band gap > 2.0 eV (insulating — no electron transfer)

Stability in acid/alkaline:
  +0.08 if formation energy < −1.5 eV/atom

Penalties:
  ×0.60 if only noble metals + no earth-abundant (too expensive to be commercially relevant)
  ×0.50 if Tl, Hg, Cd, or As present (toxic in electrolyser)
```

---

### 16. CO₂ Reduction Catalyst (CO2RR)

**Scientific Context**: Electrochemical CO₂ reduction to fuels (CO, formate, methanol, ethanol, ethylene, ethane). Cu is uniquely selective for multi-carbon (C₂+) products via C-C coupling; Ag and Au selectively produce CO; Zn and Sn produce formate. The challenge is selectivity over the competing HER. Surface oxidation state, grain boundaries, and subsurface oxygen all modulate Cu selectivity. Oxide-derived Cu (OD-Cu) typically shows enhanced C₂ selectivity.

**Scoring Logic**:
```
Cu as primary CO2RR metal:
  +0.35 if Cu present (only metal showing significant C₂+ selectivity)
  Additional +0.08 if Cu + O present (oxide-derived Cu — enhanced C₂ selectivity)

Secondary CO2RR metals:
  +0.22 if Ag or Au present (CO-selective, low overpotential)
  +0.20 if Zn or Sn present (formate-selective)
  +0.18 if Bi or In present (formate-selective, Pb-free)
  +0.15 if Pd or Pt present (CO-selective, expensive)
  +0.12 if Ni, Fe, or Co present (syngas-like products, lower selectivity)
  +0.08 if Mo or W present (emerging formate catalysts)

Surface chemistry bonuses:
  +0.15 if O present with CO2RR metal (oxide surface — activates CO₂)
  +0.10 if N present + TM (N-doped carbon analogs)
  +0.08 if noble metal free (+Earth-abundant only bonus)

Electronic:
  +0.20 if band gap < 0.5 eV (metallic — necessary for electron transfer)
  +0.10 if band gap 0.5–2.0 eV (semiconducting, possible with surface conduction)
  ×0.40 if band gap > 2.5 eV

Stability:
  +0.08 if formation energy < −1.0 eV/atom

Penalties:
  ×0.50 if no CO2RR-active metal present
  ×0.60 if Tl, Hg, Cd, or As present (toxic in aqueous electrochemical cell)
```

---

### 17. Photocatalyst (Water Splitting)

**Scientific Context**: Semiconductor photocatalysts absorb photons to generate electron-hole pairs that drive water oxidation (VB holes: H₂O → O₂) and reduction (CB electrons: H⁺ → H₂). Both half-reactions must be thermodynamically accessible: conduction band minimum must be more negative than H⁺/H₂ (0 V vs NHE) and valence band maximum must be more positive than O₂/H₂O (+1.23 V vs NHE). The band gap must exceed 1.23 eV with an additional overpotential; practical minimum ~1.8 eV. Visible-light-active photocatalysts (Eg < 3.0 eV) are preferred for solar efficiency. TiO₂ (anatase, 3.2 eV) is the benchmark but UV-only. BiVO₄ (2.4 eV), Ta₃N₅ (2.1 eV), and g-C₃N₄ (~2.7 eV) are prominent visible-light candidates.

**Scoring Logic**:
```
Hard requirements:
  if band gap < 1.23 eV → return 0.0 (thermodynamically impossible)
  if band gap > 3.5 eV → return 0.0 (UV-only; very low solar fraction)

Band gap score (Gaussian peak at 2.2 eV):
  core_score = max(0.40 − |band_gap − 2.2| × 0.12, 0.15)
  +0.10 if 1.23–1.8 eV (thermodynamically possible but marginal; useful with Z-scheme)

Material family bonuses:
  +0.28 if oxide photocatalyst: O + {Ti, Zn, Ga, In, Nb, Ta, W, Mo, Fe, Bi, Ce, Sn}
    additional +0.08 if Bi + V + O (BiVO₄ — leading visible-light photocatalyst)
    additional +0.08 if Ta + N present (Ta₃N₅ — 2.1 eV, excellent photocatalyst)
    additional +0.08 if Nb + N present
  +0.22 if nitride photocatalyst: N + {Ga, Ta, Ge, C, In, Ti}
  +0.15 if sulfide photocatalyst: S + {Zn, In, Ga, Cd, Cu, Ag}
  +0.12 if oxynitride: O + N + metal (wider visible absorption)
  +0.10 if Mo or W + O present (α-MoO₃, WO₃ — visible light active)
  +0.08 if Fe + O present (α-Fe₂O₃ — 2.2 eV, but poor hole mobility)

Stability:
  +0.10 if formation energy < −2.0 eV/atom (stable in aqueous environment)
  +0.05 if decomposition energy < 0.05 eV/atom

Penalties:
  ×0.60 if Cd present (CdS photocatalysts penalised for toxicity)
  ×0.50 if As, Tl, or Hg present
  ×0.70 if band gap < 1.8 eV (no penalty, but flag as Z-scheme only)
```

---

### 18. Nitrogen Reduction Reaction (NRR) Catalyst

**Scientific Context**: Electrochemical nitrogen fixation (N₂ + 6H⁺ + 6e⁻ → 2NH₃) at ambient conditions as an alternative to Haber-Bosch (400–500°C, 150–300 atm, 1–2% global energy consumption). The challenge is activating the N≡N triple bond (bond energy 945 kJ/mol) while suppressing HER competition. Active catalysts include: Fe (biological nitrogenase Fe-Mo cofactor), Mo single-atom catalysts, Ru, early transition metal nitrides (VN, MoN). The Sabatier principle predicts an optimal N-binding energy.

**Scoring Logic**:
```
Primary NRR metals (N₂ activation):
  +0.35 if Mo present (Mo SAC and MoN — current leading NRR candidates)
  +0.30 if Fe present (biological precedent — Fe-Mo-S nitrogenase cofactor)
  +0.25 if Ru present (high NRR activity, moderate HER selectivity issue)
  +0.22 if V present (VN — early transition metal nitride pathway)
  +0.20 if W present (WN, W SAC)
  +0.18 if Re present (high N₂ binding, expensive)
  +0.15 if Nb or Ta present (group V, similar to V)
  +0.12 if Cr present (CrN)
  +0.10 if Ti present (TiN — weaker but structurally relevant)

Nitride/sulfide enhancement:
  +0.20 if N present + NRR metal (pre-formed M-N bonds → Mars-van Krevelen pathway)
  +0.15 if S present + Fe/Mo (sulfide cofactor analogy to nitrogenase)

Electronic:
  +0.22 if band gap < 0.5 eV (metallic)
  +0.10 if band gap 0.5–1.5 eV
  ×0.40 if band gap > 2.5 eV

HER suppression proxy:
  +0.08 if formation energy < −2.0 eV/atom (stable nitride — less prone to proton reduction)

Penalties:
  ×0.60 if no N₂-activating metal present
  ×0.50 if Tl, Hg, Cd, or As present
  ×0.70 if only noble metals without Fe/Mo/V (selectivity issue)
```

---

### 19. Methane Activation Catalyst

**Scientific Context**: Partial oxidation of CH₄ to methanol (MTM) or syngas (CH₄ + O₂ → CO + 2H₂). At ~$0.03/m³ natural gas, selective activation of the C-H bond (bond dissociation energy 439 kJ/mol) could unlock trillion-dollar chemical value. Cu-zeolites (Cu-ZSM-5, Cu-SSZ-13) achieve selective MTM at 200°C. Fe-zeolites also active. Pd, Pt high-temperature partial oxidation. The challenge is stopping at methanol rather than over-oxidising to CO₂. This scorer targets inorganic catalyst materials.

**Scoring Logic**:
```
MTM-active metals:
  +0.35 if Cu present (Cu-zeolite MTM — most selective for methanol at low T)
  +0.28 if Fe present (Fe-zeolite; α-Fe-oxo sites for MTM)
  +0.22 if Pd or Pt present (high-T partial oxidation; less selective but high conversion)
  +0.20 if Rh present (syngas from CH₄; high activity)
  +0.18 if Ni present (steam reforming, less selective but industrial scale)
  +0.15 if Mo or V present (oxide catalysts for oxidative coupling)
  +0.12 if Ir or Ru present (high activity, expensive)
  +0.10 if Co present (FT syngas pathway)

Support/host structure:
  +0.18 if Al + Si + O present (aluminosilicate zeolite framework — Cu/Fe-ZSM-5 analog)
  +0.12 if Zr or Ce + O present (reducible oxide support; oxygen vacancy formation)
  +0.10 if O present with active metal (oxide surface — pre-activated)

Electronic:
  +0.20 if band gap < 0.5 eV (metallic)
  +0.10 if 0.5–2.0 eV
  ×0.40 if band gap > 3.0 eV (insulating — no redox activity)

Stability:
  +0.08 if formation energy < −2.0 eV/atom (survives reaction conditions)

Penalties:
  ×0.50 if no CH₄-activating metal
  ×0.60 if Tl, Hg, As present (toxic in industrial catalyst)
```

---

### 20. NOx Reduction Catalyst (SCR)

**Scientific Context**: Selective catalytic reduction (SCR) of NOx (NO + NO₂) with NH₃ or urea: 4NO + 4NH₃ + O₂ → 4N₂ + 6H₂O. Mandatory for diesel engines (Euro 6/VII, EPA Tier 4) and stationary power plants. Commercial catalyst: V₂O₅/WO₃/TiO₂ (300–400°C window). Zeolite-based catalysts (Cu-SSZ-13, Fe-ZSM-5) dominate for mobile applications (200–600°C, hydrothermal stable). Key properties: high N₂ selectivity, resistance to SO₂ poisoning, hydrothermal stability.

**Scoring Logic**:
```
SCR-active metals:
  +0.35 if V present (V₂O₅ — commercial benchmark, active 300–400°C)
  +0.30 if Cu present (Cu-zeolite — best mobile SCR, 200–600°C)
  +0.28 if Fe present (Fe-zeolite — good at high T, resistant to Cu poisoning)
  +0.22 if Mn present (MnOₓ — active at low T ~100°C, emerging)
  +0.20 if Ce present (CeO₂ co-catalyst; oxygen storage)
  +0.15 if W or Mo present (promoter, sulfur resistance)
  +0.12 if Cr present (Cr-based SCR catalyst)

Support materials:
  +0.18 if Ti + O present (TiO₂ — commercial V₂O₅/TiO₂ support)
  +0.15 if Al + Si + O present (zeolite framework — Cu/Fe-SSZ-13)
  +0.10 if Zr + O present (ZrO₂ — high-T stable support)

Electronic:
  +0.15 if band gap < 2.0 eV (conducting oxide — good for redox cycling)
  ×0.50 if band gap > 4.0 eV (insulating; no redox activity for NO activation)

Stability (hydrothermal conditions):
  +0.12 if formation energy < −2.5 eV/atom
  +0.08 if decomposition energy < 0.05 eV/atom

Penalties:
  ×0.60 if no SCR-active metal
  ×0.70 if S present (sulfur poisoning of SCR catalyst)
  ×0.50 if Tl, Hg, Cd, or As (toxic catalyst in automotive/industrial)
```


---

## Domain 4: Electronics & Optoelectronics

### 21. Semiconductor (General)

**Scientific Context**: Catch-all for semiconductor applications. Band gap range 0.1–4.0 eV covers all major device families from IR sensors to UV emitters. The most commercially important semiconductors are Group IV (Si, Ge, SiC), III-V (GaAs, GaN, InP), and II-VI (ZnO, CdTe, ZnSe). Wide-band-gap semiconductors (GaN, SiC, Ga₂O₃) are increasingly important for power electronics.

**Scoring Logic**:
```
Hard requirement:
  if band gap < 0.05 eV or > 4.5 eV → return 0.0

Core band gap score:
  +0.50 if 0.5–3.0 eV (primary device window)
  +0.30 if 3.0–4.0 eV (wide band gap power electronics)
  +0.25 if 0.1–0.5 eV (narrow gap; IR detectors, THz)
  +0.15 if 4.0–4.5 eV (ultrawide; UV applications)

Material family bonuses:
  +0.22 if Group IV: C, Si, Ge, Sn in compound (≥ 2 elements)
  +0.22 if III-V: (Ga, In, or Al) + (N, P, As, or Sb) (≥ 2 present)
  +0.18 if II-VI: (Zn, Cd, or Hg) + (O, S, Se, or Te) (≥ 2 present)
  +0.15 if transition metal oxide (TM + O): SnO₂, In₂O₃, WO₃ family
  +0.12 if chalcogenide perovskite (Ba/Sr/Ca + Ti/Zr + S/Se)

Stability:
  +0.08 if formation energy < −1.0 eV/atom
  +0.05 if decomposition energy < 0.05 eV/atom

Penalties:
  ×0.60 if Hg or Tl present without specific detector application context
  ×0.70 if NSites > 40 (complex structures rarely achieve commercial semiconductor status)
```

---

### 22. LED / Light Emitter

**Scientific Context**: LEDs require a direct band gap in the visible (1.77–3.1 eV) to emit efficiently. The key figure of merit is internal quantum efficiency (IQE), which depends on: radiative vs non-radiative recombination rates, defect density, and carrier injection efficiency. III-V semiconductors (GaN/InGaN for blue/green, AlGaInP for red) dominate commercial LEDs. II-VI compounds (ZnSe, CdSe) were historically important. Halide perovskites have achieved > 20% EQE for LEDs but face stability issues. The Stokes shift (difference between absorption and emission peaks) affects self-absorption losses.

**Scoring Logic**:
```
Hard requirement:
  if band gap < 1.6 eV → return 0.0 (too red/IR; not visible LED)
  if band gap > 4.5 eV → return 0.0 (deep UV; very specialised)

Band gap score:
  +0.45 if 1.77–3.1 eV (full visible range — blue to red)
  +0.25 if 3.1–4.0 eV (UV-A, near-UV — germicidal, curing applications)
  +0.15 if 1.6–1.77 eV (deep red; horticulture, NIR applications)
  +0.10 if 4.0–4.5 eV (UV-B — medical, disinfection)

Material family bonuses:
  +0.30 if III-V: (Ga, In, or Al) + (N, P, or As)
    additional +0.08 if GaN specifically (Ga + N, band gap ~3.4 eV — blue LED backbone)
    additional +0.06 if InGaN analog (In + Ga + N — tunable blue/green)
  +0.20 if II-VI: (Zn or Cd) + (S, Se, or Te)
  +0.20 if halide perovskite: halide + (Cs, Rb, or K) + (Pb, Sn, Ge, Bi, Sb, or In)
    additional +0.08 if Pb-free (Sn, Ge, Bi, Sb, In) (regulatory advantage)
  +0.15 if Eu, Tb, Ce, or Dy present (rare-earth phosphors for white LEDs)
  +0.10 if Cu + halide (Cu⁺ LED emitters — warm white, non-toxic)
  +0.08 if Mn + halide (Mn²⁺ emitters — orange-red phosphors)

Stability:
  +0.08 if formation energy < −1.5 eV/atom
  +0.05 if decomposition energy < 0.05 eV/atom

Penalties:
  ×0.30 if Si only (indirect band gap — very poor light emitter)
  ×0.60 if Hg or Tl present
  ×0.70 if Pb present (RoHS concerns for consumer lighting)
  ×0.50 if As present (GaAs widely used but regulatory pressure increasing)
```

---

### 23. Photodetector

**Scientific Context**: Converts incident photons to electrical current. Key figures of merit: responsivity (A/W), detectivity (D*, cm·Hz^0.5/W), response speed (bandwidth), and noise equivalent power. Different wavelength ranges require different materials: UV (GaN, SiC, Ga₂O₃), visible (Si, CdS, perovskites), NIR (InGaAs, Ge), SWIR (InGaAs), MWIR/LWIR (InSb, HgCdTe). Unlike solar absorbers, photodetectors need high carrier mobility and low dark current more than high absorption.

**Scoring Logic**:
```
Hard requirement:
  if band gap > 5.0 eV → return 0.0 (transparent to all useful radiation)
  if band gap < 0.0 eV → return 0.0

Base band gap score by detection window:
  +0.40 if band gap in [0.3, 4.5] eV (broad coverage)
  Sub-window bonuses:
    +0.12 if 0.3–1.0 eV (IR detection; InSb, HgCdTe range)
    +0.15 if 1.0–2.0 eV (visible; highest solar photon flux)
    +0.12 if 2.0–3.5 eV (UV-visible; high-energy photon detection)
    +0.10 if 3.5–4.5 eV (UV-B/C; solar-blind detection)

Material family bonuses:
  +0.25 if III-V high-mobility: (In + Ga) + (As or P) (InGaAs — telecom 1550 nm detector)
  +0.22 if halide perovskite type (fast response time; X-ray and visible)
  +0.18 if (Hg + Cd + Te) present (HgCdTe — MWIR/LWIR, military/thermal imaging)
  +0.15 if Si or Ge present (established semiconductor detectors, Si foundry-compatible)
  +0.15 if SiC or GaN present (UV-A, solar-blind)
  +0.12 if chalcogenide + heavy element (high Z stops radiation → X-ray detector)
  +0.10 if II-VI (Zn + S/Se/Te)

High-mobility proxy:
  +0.10 if III-V elements (GaAs, InP, InAs — highest electron mobilities)
  +0.08 if 2D dimensionality detected (high in-plane mobility)

Stability:
  +0.05 if formation energy < −1.0 eV/atom

Penalties:
  ×0.70 if band gap > 4.0 eV AND no heavy elements (UV-transparent without high-Z stopping)
  ×0.60 if Tl, Cd without justified use context (Cd still used in HgCdTe)
```

---

### 24. Transparent Conductor

**Scientific Context**: Wide-band-gap materials with degenerate doping enabling simultaneous optical transparency (> 80% in visible) and metallic-like conductivity (sheet resistance < 20 Ω/□). Commercial ITO (In₂O₃:Sn) dominates but In is scarce and expensive (~$167/kg). Alternatives: AZO (Al-doped ZnO), GZO (Ga-doped ZnO), FTO (F-doped SnO₂), and emerging amorphous IGZO. The optical band gap must exceed ~3.1 eV to be transparent to visible light (wavelengths < 400 nm). Electrical conductivity comes from donor doping — not intrinsic.

**Scoring Logic**:
```
Hard requirement:
  if band gap < 3.0 eV → return 0.0 (absorbs visible — not transparent)

Band gap score:
  +0.45 if ≥ 4.0 eV (excellent visual transparency; UV-stable too)
  +0.30 if 3.5–4.0 eV (very good transparency)
  +0.20 if 3.0–3.5 eV (borderline; slight yellow tint possible)

TCO element families:
  +0.35 if In + O (In₂O₃ host — ITO, IGZO; highest performance TCO)
  +0.30 if Zn + O (ZnO host — AZO/GZO; earth-abundant ITO replacement)
  +0.28 if Sn + O (SnO₂ host — FTO; high chemical stability, fluoride doped)
  +0.22 if Ga + O (Ga₂O₃ — ultrawide gap, β-Ga₂O₃ emerging power/UV transparent conductor)
  +0.18 if Al + Zn + O (AZO specifically)
  +0.15 if Cd + In + O or Cd + Sn + O (Cd-containing TCOs — functional but toxic)
  +0.12 if Ti + O (anatase TiO₂:Nb — amorphous, flexible substrate compatible)
  +0.10 if non-oxide present (×0.4 penalty applied — TCOs are almost universally oxides)

Earth-abundant bonus:
  +0.12 if Zn or Sn based WITHOUT In (avoids scarce element)

Stability:
  +0.10 if formation energy < −2.0 eV/atom (stable against reduction by electrode)
  +0.05 if decomposition energy < 0.05 eV/atom

Penalties:
  ×0.40 if no O present (non-oxide — very unusual TCO)
  ×0.60 if Cd present (CdO is a TCO but Cd toxicity is prohibitive for most applications)
  ×0.50 if only organic-framework-like complexity (>6 element types — synthesis too complex)
```

---

### 25. Ferroelectric

**Scientific Context**: Ferroelectrics exhibit a spontaneous electric polarisation that is switchable by an applied electric field. This requires a non-centrosymmetric crystal structure (one of 20 polar point groups). Applications: capacitors (MLCC), FeRAM (non-volatile memory), piezoelectric sensors/actuators, electrocaloric cooling. Commercial materials: PZT (Pb(Zr,Ti)O₃, Eg ~3.5 eV, polarisation ~60 μC/cm²), BaTiO₃ (Eg ~3.2 eV, Pb-free), BiFeO₃ (multiferroic). The Ba/Sr/Pb + Ti/Zr/Nb + O perovskite family overwhelmingly dominates.

**Scoring Logic**:
```
Symmetry requirement (most important criterion):
  if space group available:
    if space group in centrosymmetric list (11 Laue groups) → ×0.3
    if space group in polar list (10 polar point groups) → +0.40
    if space group in non-centrosymmetric but not polar → +0.20

Perovskite structure detection:
  A-site candidates: {Ba, Sr, Ca, Pb, Na, K, Bi, Li, La, Nd}
  B-site candidates: {Ti, Zr, Nb, Ta, Fe, Mn, W, Hf, Mo, V, Cr}
  +0.35 if ≥ 1 A-site + ≥ 1 B-site + O present
    additional +0.08 if BaTiO₃ proxy: Ba + Ti + O
    additional +0.08 if KNbO₃ proxy: K + Nb + O
    additional +0.08 if BiFeO₃ proxy: Bi + Fe + O (multiferroic bonus)
    additional +0.08 if LiNbO₃ proxy: Li + Nb + O

Aurivillius/Ruddlesden-Popper bonus:
  +0.15 if Bi + (TM) + O + layered crystal system (Aurivillius phase — high Tc)
  +0.12 if Sr or La + (TM) + O + tetragonal system (RP-phase)

Band gap:
  +0.20 if ≥ 3.0 eV (good insulation — leakage current suppressed)
  +0.12 if 1.5–3.0 eV (acceptable; multiferroics often in this range)
  ×0.30 if < 1.0 eV (too conductive; polarisation cannot be sustained)

Stability:
  +0.10 if formation energy < −2.5 eV/atom

Pb-free bonus:
  +0.12 if Pb not in elements (ROHS compliance for consumer electronics)

Penalties:
  ×0.70 if Pb present (regulatory pressure — EU ROHS exemption ending)
  ×0.50 if metallic (no band gap)
```

---

### 26. Piezoelectric

**Scientific Context**: Piezoelectrics generate electric charge under mechanical stress and deform under applied voltage. Like ferroelectrics, requires non-centrosymmetric structure. Piezoelectric coefficient d₃₃ (pC/N) is the key figure of merit. PZT achieves d₃₃ ~400–600 pC/N; BaTiO₃ ~190 pC/N; AlN ~5 pC/N (low but MEMS compatible). Applications: MEMS sensors, ultrasound transducers, energy harvesters, RF filters (BAW/SAW). AlN is dominant for MHz–GHz RF filters in smartphones.

**Scoring Logic**:
```
Symmetry requirement:
  if polar space group detected → +0.40
  if non-centrosymmetric, non-polar → +0.22 (piezoelectric but not pyroelectric)
  if centrosymmetric → ×0.3

Piezoelectric material families:
  +0.30 if perovskite: A-site + B-site + O (see ferroelectric above for lists)
    additional +0.08 if Pb + Zr/Ti + O (PZT family — highest d₃₃)
    additional +0.06 if Ba + Ti + O (BaTiO₃ — Pb-free benchmark)
    additional +0.06 if K + Nb + O or K + Nb + Na + O (KNN — leading Pb-free)
  +0.25 if Al + N present (AlN/ScAlN — RF MEMS dominant; high frequency applications)
    additional +0.08 if Sc + Al + N (ScAlN — 3× higher d₃₃ than pure AlN)
  +0.20 if Zn + O (ZnO — wide-gap, MEMS piezoelectric; ALD-compatible)
  +0.18 if Li + Nb/Ta + O (LiNbO₃/LiTaO₃ — SAW filter substrates)
  +0.15 if Bi + (TM) + O non-centrosymmetric (bismuth-based perovskites)

Band gap:
  +0.15 if ≥ 3.5 eV (wide gap → high resistivity → better piezo performance)
  +0.10 if 2.0–3.5 eV
  ×0.40 if < 1.0 eV (conductive; charge bleeds away immediately)

Stability:
  +0.10 if formation energy < −2.0 eV/atom

Pb-free bonus:
  +0.15 if no Pb (strong driver for automotive, medical, consumer electronics)

Penalties:
  ×0.70 if Pb present (regulatory)
  ×0.50 if metallic
```

---

### 27. Phase Change Memory (PCM)

**Scientific Context**: Non-volatile memory based on reversible amorphous↔crystalline phase transitions in chalcogenide alloys, changing resistance by 3–6 orders of magnitude. Benchmark material: Ge₂Sb₂Te₅ (GST, ~2.0 eV crystalline band gap, ~0.7 eV amorphous). Applications: Intel Optane (discontinued but influential), neuromorphic computing, optical data storage (CD/DVD/Blu-Ray). Key requirements: fast crystallisation (< 100 ns), large resistance contrast, thermal stability of amorphous phase (10-year retention at 85°C), cyclability (> 10⁸ cycles).

**Scoring Logic**:
```
PCM material family (GST-type):
  +0.40 if Ge + Sb + Te present (GST — THE benchmark; all variants score here)
  +0.30 if Sb + Te present without Ge (Sb₂Te₃ family — faster crystallisation)
  +0.28 if Ge + Te present (GeTe — high Tc amorphous, good retention)
  +0.25 if In + Sb + Te or Ag + In + Sb + Te (AIST — optical disc materials)
  +0.22 if Bi + Te or Bi + Se present (Bi-based PCM analogs)
  +0.20 if Sn + Te or Pb + Te present (IV-VI PCM candidates)
  +0.15 if Se + (Sb or As or Ge) present (chalcogenide glass analog)

Band gap considerations:
  +0.25 if 0.3–1.5 eV (narrow gap — PCM materials are typically narrow-gap semiconductors)
  +0.15 if 1.5–2.5 eV (wider gap; amorphous phase may have larger gap)
  +0.05 if > 2.5 eV (too insulating for PCM operation at reasonable voltage)

Crystallisation kinetics proxy:
  +0.10 if Sb present (fast crystalliser — promotes nucleation)
  +0.08 if In present (reduces crystallisation temperature)
  +0.07 if N present (nitrogen doping raises crystallisation temperature — better retention)

Stability:
  +0.08 if formation energy < −0.5 eV/atom (stable crystalline phase)

Penalties:
  ×0.60 if no chalcogenide element (Te, Se, S — PCM is exclusively chalcogenide-based)
  ×0.50 if Tl, Hg, Cd, or As present in non-Te/Se context
  ×0.70 if band gap > 3.0 eV (too insulating for PCM switching)
```

---

### 28. High-k Dielectric (Gate Oxide)

**Scientific Context**: As MOSFET gate oxides thin below 2 nm, quantum tunnelling current through SiO₂ becomes unacceptable. High-k dielectrics (k > 10) allow thicker physical layers while maintaining equivalent capacitance. Intel and TSMC adopted HfO₂ (k ~25) with HKMG in 2007. Requirements: high dielectric constant, large band gap (> 5 eV), wide conduction/valence band offsets vs Si (> 1 eV each), thermodynamic stability vs Si interface, amorphous phase stability. Emerging: ferroelectric HfO₂:Zr for FinFET-compatible memory gates.

**Scoring Logic**:
```
Hard requirements:
  if band gap < 4.0 eV → return 0.0 (leakage current unacceptable)
  if O not in elements → ×0.3 (non-oxide dielectrics very rare in CMOS)

Band gap score:
  +0.40 if ≥ 7.0 eV (excellent — SiO₂-like, excellent band offsets)
  +0.30 if 5.0–7.0 eV (very good — HfO₂ range)
  +0.15 if 4.0–5.0 eV (borderline; leakage may be acceptable at reduced temperature)

High-k oxide families:
  +0.35 if Hf + O (HfO₂ — industry standard; k~25, Eg~5.7 eV)
    additional +0.08 if Zr + Hf + O (HZO — ferroelectric gate dielectric)
  +0.28 if Zr + O (ZrO₂ — k~25, close relative to HfO₂)
  +0.25 if La + O (La₂O₃ — k~27, high band gap)
  +0.22 if Al + O (Al₂O₃ — k~9, lower but excellent interface with Si/III-V)
  +0.20 if Ta + O (Ta₂O₅ — k~25 but smaller band gap 4.4 eV)
  +0.18 if Nb + O (Nb₂O₅ — high k, lower band gap)
  +0.15 if Y + O or Sc + O (Y₂O₃, Sc₂O₃ — high Eg, moderate k)
  +0.12 if Ti + O (TiO₂ — very high k ~80 but small Eg ~3.0 eV; penalty applied)

Si-interface compatibility:
  +0.10 if no Si-silicide-forming elements at interface temperature (Hf, Zr, Al are safe)
  +0.08 if La or Y present (gettering of interface traps)

Stability:
  +0.10 if formation energy < −4.0 eV/atom (must survive 1000°C anneal)
  +0.05 if decomposition energy < 0.01 eV/atom

Penalties:
  ×0.60 if Ti + O as primary phase (TiO₂: high k but too small band gap for gate dielectric)
  ×0.50 if no oxide (non-oxide dielectrics incompatible with CMOS processing)
  ×0.70 if toxic elements (Tl, Hg, Cd, As)
```

---

### 29. Nonlinear Optical (NLO) Material

**Scientific Context**: Materials with large second-order (χ⁽²⁾) or third-order (χ⁽³⁾) susceptibilities for frequency conversion (SHG, THG, OPO), electro-optic modulation (Pockels effect), and all-optical switching. Second-order NLO requires non-centrosymmetric structure (same symmetry requirement as piezoelectrics). Key materials: KDP (KH₂PO₄), BBO (β-BaB₂O₄), LiNbO₃, KNbO₃. Wide band gap is essential — the material must be transparent at both the fundamental and harmonic frequencies.

**Scoring Logic**:
```
Symmetry requirement (mandatory for χ⁽²⁾):
  if centrosymmetric space group → ×0.3 (χ⁽²⁾ = 0 by symmetry)
  if polar space group → +0.40 (full bonus for SHG/Pockels)
  if non-centrosymmetric, non-polar → +0.20 (χ⁽²⁾ allowed but may be small)

Band gap (must be transparent at both ω and 2ω):
  Hard: if band gap < 2.0 eV → ×0.3 (absorbs visible/fundamental + harmonic)
  +0.30 if ≥ 5.0 eV (DUV transparent; critical for UV laser applications)
  +0.22 if 3.5–5.0 eV (UV transparent — BBO range)
  +0.15 if 2.5–3.5 eV (visible transparent — LiNbO₃ range)
  +0.10 if 2.0–2.5 eV (marginal; limits to IR fundamentals)

NLO material families:
  +0.30 if Li + Nb + O or Li + Ta + O (LiNbO₃/LiTaO₃ — electro-optic standard)
  +0.28 if K + Nb + O or K + Ti + O (KNbO₃, KTP — phase-matchable SHG)
  +0.25 if Ba + B + O (BBO family — DUV SHG benchmark)
  +0.22 if K + H + P + O (KDP/ADP — classic NLO crystals)
  +0.20 if Ag + Ga + S/Se/Te (AgGaS₂, AgGaSe₂ — mid-IR NLO)
  +0.18 if Cs + Pb + Br/I (halide perovskite NLO — large χ⁽²⁾ recently reported)
  +0.15 if Zn + Ge + P + (S or Se) (ZnGeP₂ — high-power IR NLO)
  +0.12 if B + O + alkali (borate NLO family)
  +0.10 if Se or Te + (Sb or Bi) (narrow-gap mid-IR NLO)

Stability:
  +0.08 if formation energy < −2.0 eV/atom

Penalties:
  ×0.50 if centrosymmetric (no χ⁽²⁾)
  ×0.60 if Tl, Hg, Cd, or As (toxic)
```


---

## Domain 5: Magnetics & Spintronics

### 30. Permanent Magnet

**Scientific Context**: Hard magnets with high remanence (Br > 1 T), high coercivity (Hc > 1 MA/m), and high energy product (BH)max > 200 kJ/m³. Used in EV motors (2–4 kg Nd₂Fe₁₄B per motor), wind turbines, hard drives. Nd₂Fe₁₄B (35–55 MGOe) dominates. Sm₂Co₁₇ preferred at high temperature. REEs provide magnetocrystalline anisotropy (single-ion anisotropy) — without them, coercivity collapses. Sm₂Fe₁₇N₃ (nitrided, 46 MGOe potential) is an emerging REE-lean alternative.

**Scoring Logic**:
```
Mandatory check:
  if no magnetic metals (Fe, Co, Ni, Mn) → return 0.0

Magnetic moment carriers:
  +0.30 if Fe present (highest Curie temp among 3d metals; Fe₁₄Nd₂B backbone)
  +0.20 if Co present (high Curie temp, corrosion resistance)
  +0.15 if Ni present (lower moment but important in SmCo/AlNiCo)
  +0.10 if Mn present (Mn-Al-C tetragonal magnet — RE-free candidate)

Rare-earth anisotropy elements (essential for high coercivity):
  +0.35 if Nd or Pr present (Nd₂Fe₁₄B — highest (BH)max; Pr slightly lower Tc)
  +0.30 if Sm present (SmCo₅, Sm₂Co₁₇ — high Tc, corrosion resistant)
  +0.20 if Dy or Tb present (coercivity enhancer; supply-critical heavy REE)
  +0.15 if Ho or Er present (anisotropy, specialised applications)
  +0.10 if Ce or La present (Ce substitution in Nd magnets — cost reduction strategy)

Structural bonuses:
  +0.15 if B present (Nd₂Fe₁₄B crystal structure enabler)
  +0.10 if N present (Sm₂Fe₁₇N₃ — interstitial nitriding enhances anisotropy)
  +0.08 if Co + Sm present (SmCo₅ or Sm₂Co₁₇ specifically)

Electronic character:
  +0.15 if band gap < 0.1 eV (metallic — required for ferromagnetism in 3d metals)
  ×0.50 if band gap > 0.5 eV (reduces exchange coupling and moment)

Stability:
  +0.08 if formation energy < −1.0 eV/atom

Penalties:
  ×0.70 if Dy or Tb present without Nd/Sm (heavy REE without host structure — high supply risk)
```

---

### 31. Soft Magnet

**Scientific Context**: Low coercivity (Hc < 1000 A/m), high permeability, low hysteresis loss. Used in transformer cores, inductors, electric motors, magnetic shielding. Fe-Si (3.2% Si) electrical steel is the global standard (40 million tons/year). Ferrites (MnZn, NiZn) dominate high-frequency applications due to high resistivity (no eddy currents). Amorphous/nanocrystalline metals (Fe-Si-B, Fe-Cu-Nb-Si-B / FINEMET) offer superior soft magnetic properties for medium frequency. No REEs needed — a cost advantage over permanent magnets.

**Scoring Logic**:
```
Mandatory:
  if no magnetic metals (Fe, Co, Ni) → return 0.0

Magnetic metals:
  +0.25 if Fe present (dominant soft magnet)
  +0.15 if Co present (permalloy, Co-based amorphous)
  +0.12 if Ni present (Ni-Fe permalloys: ~80% Ni, 20% Fe)

Specific material families:
  +0.30 if Fe + Si present (electrical steel — world's most important soft magnet)
    additional +0.08 if no other transition metals (pure Fe-Si steel)
  +0.25 if spinel ferrite: (Mn or Ni or Zn or Cu or Mg) + Fe + O
    (MnZn ferrite: 300 Hz–1 MHz; NiZn ferrite: 1–100 MHz)
  +0.20 if Fe + B present (Fe-B amorphous → FINEMET nanocrystalline family)
  +0.18 if Fe + Ni present (permalloy family: 78% Ni-Fe: highest permeability)
  +0.15 if Fe + Co + B or Fe + Co + Si + B (Co-based amorphous — very low losses)

Amorphous-forming indicators:
  +0.10 if Fe + B + P (Fe-B-P glass-forming compositions)
  +0.08 if Fe + Si + B + Cu + Nb (FINEMET indicator — optimised nanocrystalline)
  +0.07 if Fe + Zr or Fe + Hf (Fe-Zr amorphous — NANOPERM family)

Electronic:
  +0.15 if band gap < 0.5 eV (metallic)

Stability:
  +0.05 if formation energy < −0.5 eV/atom

Penalties:
  ×0.70 if REE present (Nd, Sm, Dy, Tb, Ho, Er, Tm, Yb, Lu) — soft magnets don't need REEs; penalise for cost
  ×0.60 if band gap > 1.0 eV (ferrite exception: ferrimagnetic oxides can have moderate gaps)
    NOTE: if spinel ferrite detected, waive the band gap penalty
```

---

### 32. Magnetic Semiconductor / Spintronics

**Scientific Context**: Dilute magnetic semiconductors (DMS) and spin-transport materials for spintronic devices: spin-LEDs, spin-valves, magnetic tunnel junctions (MTJ), and spin-orbit torque (SOT) MRAM. (Ga,Mn)As is the archetypal DMS but has Tc < 200K. EuO (Tc = 69K), chromium triiodide (CrI₃, 2D ferromagnet), and Fe₃GaTe₂ (Tc ~380K) are emerging. For practical spintronics: Tc > 300K essential. MTJ materials (MgO barriers, CoFeB electrodes) require specific band alignment for high TMR.

**Scoring Logic**:
```
Hard requirement:
  if band gap < 0.05 eV or > 4.0 eV → return 0.0

Magnetic 3d metals (spin carriers):
  +0.30 if Mn present (Mn-doped GaAs, Mn-based oxides)
  +0.25 if Fe present (Fe₃O₄, Fe₃GaTe₂)
  +0.22 if Co present (Co₂MnSi Heusler alloys)
  +0.20 if Cr present (CrI₃, Cr₂Ge₂Te₆, CrO₂)
  +0.18 if Ni present (NiMnSb Heusler)
  +0.15 if V present (VO₂ — metal-insulator transition)

Rare-earth magnetic elements:
  +0.25 if Eu present (EuO, EuS — classic magnetic semiconductors)
  +0.22 if Gd present (GdN — near-room-temperature TC, high spin polarisation)
  +0.18 if Dy, Nd, or Sm present (high anisotropy spintronic applications)

Semiconductor host families:
  +0.20 if (Ga or In) + (As or N or P) — III-V host
  +0.18 if (Zn or Cd) + (S, Se, or Te) — II-VI host
  +0.15 if chalcogenide 2D host (Te + Cr or Fe) — CrTe₂, FeTe₂ family
  +0.12 if I + halide + magnetic metal (halide magnetic semiconductors — CrI₃ family)

Band gap:
  +0.20 if 0.1–1.5 eV (narrow-gap spintronics)
  +0.15 if 1.5–3.0 eV (wide-gap DMS)
  +0.10 if 3.0–4.0 eV (UV magnetic semiconductor)

Stability:
  +0.08 if formation energy < −1.0 eV/atom

Penalties:
  ×0.60 if only non-magnetic elements
  ×0.70 if Tl, Hg, Cd (except CdTe) in non-justified context
```

---

### 33. Spintronic MTJ (Magnetic Tunnel Junction)

**Scientific Context**: MTJs consist of two ferromagnetic electrodes separated by a thin tunnel barrier. Tunnel magnetoresistance (TMR) ratio (up to 600% at RT in CoFeB/MgO/CoFeB) depends critically on the barrier material. MgO (001)-textured barriers enable coherent tunnelling through Δ₁ bands, producing record TMR. Key barrier requirements: wide band gap, lattice match to electrode, interface quality, pinhole-free at 1–3 nm. Applications: hard drive read heads (recording industry), STT-MRAM (non-volatile), and spin-orbit torque devices.

**Scoring Logic**:
```
MTJ barrier requirements:
  Must be wide-gap insulator:
  if band gap < 3.5 eV → return 0.0 (direct tunnelling dominates; loses coherence)

  +0.40 if Mg + O present (MgO — absolute benchmark; 600% TMR with CoFeB)
  +0.30 if Al + O present (Al₂O₃ — original MTJ barrier; ~70% TMR)
  +0.25 if Mg + Al + O or spinel MgAl₂O₄ (interface engineering variant)
  +0.22 if Sr + Ti + O (SrTiO₃ — epitaxial growth, band alignment studied)
  +0.20 if Hf + O or Zr + O (HfO₂, ZrO₂ — high-k candidate barriers)
  +0.18 if Zn + O (ZnO barrier — niche applications)
  +0.15 if Ti + O (TiO₂ — rutile MTJ barrier, studied for spintronics)
  +0.12 if V + O or Cr + O (emerging oxide barriers)

Band gap score:
  +0.35 if ≥ 7.0 eV (excellent barrier height)
  +0.25 if 5.0–7.0 eV
  +0.15 if 3.5–5.0 eV

Crystal compatibility:
  +0.10 if cubic or tetragonal crystal system (lattice match to BCC Fe/CoFeB electrode)
  +0.08 if rocksalt or perovskite structure (epitaxial growth feasibility)

Stability:
  +0.12 if formation energy < −3.0 eV/atom (barrier must survive annealing at 350°C)
  +0.06 if decomposition energy < 0.01 eV/atom

Penalties:
  ×0.50 if magnetic elements in barrier (kills TMR — barrier must be non-magnetic)
  ×0.60 if S, Se, Te present (chalcogenides reduce to metal under annealing)
```


---

## Domain 6: Thermal & Structural Coatings

### 34. Thermal Barrier Coating (TBC)

**Scientific Context**: TBCs insulate metal turbine blades from 1400–1700°C combustion gases, enabling turbine inlet temperatures above the metal's melting point (superalloy Tm ~1300°C). The 7 wt% yttria-stabilised zirconia (7YSZ) has been commercial standard for 40 years but degrades > 1200°C via sintering and CMAS (calcium-magnesium-alumino-silicate) attack. Next-generation: rare-earth zirconates (La₂Zr₂O₇, Gd₂Zr₂O₇), hexaaluminates (LaMgAl₁₁O₁₉), and pyrochlores (La₂Ce₂O₇). Requirements: low thermal conductivity (< 2 W/m·K), thermal expansion mismatch tolerance, phase stability to 1700°C.

**Scoring Logic**:
```
Mandatory: oxygen required (all TBCs are oxides)
  if O not in elements → return 0.0

Band gap:
  +0.35 if ≥ 5.0 eV (excellent electrical insulation)
  +0.22 if 4.0–5.0 eV
  +0.10 if 3.0–4.0 eV
  +0.00 if < 3.0 eV

Zirconia family (dominant TBC):
  +0.28 if Zr + O present (ZrO₂ host — central to all zirconate TBCs)
    additional +0.10 if Y + Zr + O (7YSZ — THE benchmark)
    additional +0.08 if Ce + Zr + O (CeO₂-ZrO₂ — higher temperature stability)
    additional +0.08 if La + Zr + O (La₂Zr₂O₇ — next-gen pyrochlore)
    additional +0.08 if Gd + Zr + O (Gd₂Zr₂O₇ — lower κ than YSZ)
    additional +0.06 if Nd, Sm, or Eu + Zr + O (RE-zirconate family)

Hexaaluminate family:
  +0.30 if Al + O + {Ba, Sr, La, Ce, Nd, Sm, Gd, Eu, Pr} (LaMgAl₁₁O₁₉ type)
    additional +0.08 if Mg + La + Al + O (Mg-modified hexaaluminate — most studied)

Pyrochlore family:
  +0.28 if A₂B₂O₇ structure proxied: {La, Nd, Sm, Gd, Er, Yb, Y} + {Zr, Ti, Hf, Ce, Sn} + O

Other refractory oxide bonuses:
  +0.15 if Hf + O (HfO₂ — high melting point 2758°C, phase stable)
  +0.12 if Ta + O or Nb + O (high-melting refractory oxides)
  +0.10 if Si + O (silicate TBCs — emerging environmental barrier coating)

Thermodynamic stability:
  +0.12 if formation energy < −4.0 eV/atom (essential for 1500°C stability)
  +0.06 if decomposition energy < 0.01 eV/atom

Penalties:
  ×0.40 if volatile elements present: Na, K, Li, Rb, Cs, Zn, Cd, Hg, S, Se, Te
    (evaporate at TBC temperatures — catastrophic coating failure)
  ×0.60 if no oxide-forming metal (impossible TBC chemistry)
  ×0.50 if band gap < 2.0 eV (electrically conducting at operating temperature)
```

---

### 35. Thermal Interface Material (TIM)

**Scientific Context**: Fills microscopic air gaps between heat source (CPU/GPU/power module) and heat sink, dramatically reducing thermal contact resistance. Thermal conductivity target > 10 W/m·K; electrical insulation required for most applications (band gap > 2.5 eV). Best materials: diamond (κ ~2000 W/m·K), cubic BN (κ ~750 W/m·K), AlN (κ ~180 W/m·K), SiC (κ ~150 W/m·K). The Debye model relates κ to phonon mean free path — light atoms in strong bonds have high phonon velocity and long mean free path.

**Scoring Logic**:
```
Hard requirement:
  if band gap < 2.5 eV → return 0.0 (electrically conducting — shorts power electronics)

Band gap score (electrical insulation):
  +0.35 if ≥ 7.0 eV
  +0.28 if 5.0–7.0 eV
  +0.18 if 3.5–5.0 eV
  +0.08 if 2.5–3.5 eV

High thermal conductivity material families:
  +0.50 if C present (diamond — κ ~2000 W/m·K; ultimate TIM)
    ×0.5 penalty if NSites > 3 (non-diamond carbon phases have lower κ)
  +0.42 if B + N present (cubic BN — κ ~750 W/m·K; next to diamond)
  +0.38 if Al + N present (AlN — κ ~180 W/m·K; commercial power electronics)
  +0.32 if Si + C present (SiC — κ ~150 W/m·K; established power semiconductor substrate)
  +0.25 if Be + O present (BeO — κ ~270 W/m·K; excellent but highly toxic)
    ×0.4 toxicity penalty for BeO (beryllium oxide inhalation hazard)
  +0.20 if Si + N present (Si₃N₄ — κ ~80 W/m·K; emerging substrate)
  +0.18 if Mg + O (MgO — κ ~50 W/m·K; moderate TIM)
  +0.15 if Al + O (Al₂O₃ — κ ~35 W/m·K; common but low κ for TIM)

Structural simplicity bonus (low complexity → high κ):
  +0.10 if NSites ≤ 4 (binary/simple ternary — minimal phonon scattering)
  +0.05 if NSites 5–8 (moderate complexity)
  ×0.70 if NSites > 15 (complex structure — phonon scattering suppresses κ)

Stability:
  +0.08 if formation energy < −3.0 eV/atom

Penalties:
  ×0.50 if Be present (BeO: effective TIM but toxic; restrict to specialised use)
  ×0.60 if band gap 2.5–3.0 eV AND electrically active dopants likely
  ×0.70 if S, Se, or Te present (chalcogenides are thermal insulators — opposite of what's needed)
```

---

### 36. Hard Coating / Wear Resistant

**Scientific Context**: Thin (1–10 μm) hard ceramic coatings deposited by PVD/CVD on cutting tools, dies, and engine components. Hardness > 20 GPa Vickers (diamond ~100 GPa, cubic BN ~60 GPa, TiN ~20 GPa). Key families: binary nitrides (TiN, CrN), ternary nitrides (TiAlN, CrAlN — form protective Al₂O₃ layer at > 800°C), carbides (TiC, WC), and diamond-like carbon (DLC). Coating must be hard, adhere well, and survive thermal shock and chemical attack from cutting fluids.

**Scoring Logic**:
```
Hard coating metal-anion combinations:
  Base check: metals = {Ti, Cr, W, Mo, V, Nb, Ta, Zr, Hf, Al}
              anions = {N, C, B}

  +0.45 if any coating metal + N (TiN/CrN/ZrN family)
    additional +0.12 if Ti + Al + N (TiAlN — best commercial multi-component)
    additional +0.10 if Cr + Al + N (CrAlN — good for high-Cr steels)
    additional +0.08 if Ti + Si + N (nanocomposite nc-TiN/a-Si₃N₄ — superhardness)
    additional +0.08 if V + N (VN — self-lubricating via V₂O₅ formation)
    additional +0.08 if Nb or Ta + N (hard nitrides, less studied)
  +0.40 if any coating metal + C (carbides)
    additional +0.10 if W + C (WC — cemented carbide, hardest conventional tool material)
    additional +0.10 if Ti + C (TiC — abrasive wear resistance)
  +0.35 if any coating metal + B (borides)
    additional +0.12 if Ti + B (TiB₂ — very high hardness ~33 GPa)
    additional +0.10 if Zr + B or Hf + B (UHTC borides, also hard)
  +0.40 if C only present (diamond or DLC — ultimate hardness)
  +0.22 if Al + O present (Al₂O₃ — oxide coating, good at high T, CVD inserts)
  +0.18 if Si + N present (Si₃N₄ — hard ceramic, bearing applications)

Band gap:
  +0.15 if ≥ 3.5 eV (indicates strong covalent/ionic bonding → hardness)
  +0.10 if 1.5–3.5 eV (metallic nitrides often in this range)
  +0.05 if < 1.5 eV (metallic — OK for WC/TiN)

Stability:
  +0.12 if formation energy < −2.5 eV/atom (must survive cutting temperatures)
  +0.06 if decomposition energy < 0.05 eV/atom

Ternary/quaternary bonus:
  +0.10 if ≥ 3 coating metals + anion (nanocomposite/superhardness designs)

Penalties:
  ×0.60 if no hard anion (N, C, B, O) present
  ×0.50 if volatile elements (Na, K, S, Se, Te, Zn) — evaporate during deposition
  ×0.70 if Tl, Hg, Cd, or As (toxic in cutting environment)
```

---

### 37. Corrosion Resistant Coating

**Scientific Context**: Protects metallic substrates from electrochemical oxidation, acid/base attack, and stress-corrosion cracking. Mechanisms: (1) passive oxide layer (Cr₂O₃ on stainless, Al₂O₃ on aluminium, TiO₂ on titanium); (2) noble metal coating (Pt, Au, Ir); (3) fluoride conversion coatings (chemical inertness); (4) ceramic coatings (plasma-spray Al₂O₃, WC-Co HVOF). The coating must be thermodynamically stable, low-defect (no pinholes), and have CTE match to substrate.

**Scoring Logic**:
```
Passive oxide families:
  +0.35 if Cr + O (Cr₂O₃ — gold standard passive film; stainless steel basis)
  +0.32 if Al + O (Al₂O₃ — AA alloys, aerospace; ALD conformal coating)
  +0.28 if Ti + O (TiO₂ — titanium, dental, aerospace)
  +0.25 if Zr + O (ZrO₂ — chemical reactor coatings, Zircaloy cladding)
  +0.22 if Ta + O (Ta₂O₅ — semiconductor wet processing)
  +0.20 if Nb + O (Nb₂O₅ — reactive metal analogue to Ta)
  +0.18 if Hf + O (HfO₂ — very chemically inert)
  +0.15 if Si + O (SiO₂ — glass, thermal oxidation, sol-gel coatings)
  +0.12 if W + O (WO₃ — emerging electrochemical corrosion protection)

Noble metal coatings:
  +0.30 if Pt present (platinum — ultimate corrosion resistance; very expensive)
  +0.25 if Ir or Rh present (acid-stable noble metals)
  +0.22 if Au present (gold — inert but expensive)
  +0.18 if Pd or Ru present (noble, moderately expensive)

Fluoride coatings:
  +0.28 if F + {Ca, Ba, Sr, Mg, La, Ce, Al} (fluoride conversion coatings — chemically inert)
    additional +0.08 if Ca + F (CaF₂ — optical coating and corrosion protection)
    additional +0.08 if La + F (LaF₃ — optical/corrosion coating)

Band gap:
  +0.20 if ≥ 4.0 eV (electrically insulating → cathodic protection under oxide)
  +0.12 if 3.0–4.0 eV
  +0.05 if 2.0–3.0 eV

Thermodynamic stability:
  +0.22 if formation energy < −4.0 eV/atom (extremely stable against reduction)
  +0.12 if formation energy −2.5 to −4.0 eV/atom

Penalties:
  ×0.50 if S or Se present (chalcogenides corrode in acidic environments)
  ×0.60 if Na, K, Ca, Mg chloride present (promotes pitting corrosion)
  ×0.70 if Tl, Hg, Cd, or As (toxic coating)
```

---

### 38. Refractory / UHTC

**Scientific Context**: Ultra-high temperature ceramics (UHTCs) for hypersonic vehicles (nose cones, leading edges), nuclear fuel cladding, and extreme industrial applications (> 2000°C in oxidising environments). Key families: HfC (Tm = 3958°C, highest of any binary compound), ZrB₂ (Tm = 3245°C, oxidation-resistant boride), TaC (Tm = 3880°C), HfB₂ (Tm = 3380°C), W-Re alloys. Requirements: Tm > 2000°C, structural integrity, thermal shock resistance, oxidation resistance.

**Scoring Logic**:
```
Refractory metal check:
  refractory_metals = {W, Re, Os, Ir, Mo, Hf, Nb, Ta, Zr, V, Cr, Ti, Ru, Rh}
  if no refractory metal → return 0.0 (room-temperature materials can't be UHTCs)

Refractory metal bonus:
  +0.30 if any refractory metal present
  UHTC-specific bonuses:
    +0.08 if Hf present (highest Tm carbide/boride)
    +0.08 if Ta present (TaC, TaB₂ — ultra-high Tm)
    +0.07 if W or Re present (highest Tm metals)
    +0.06 if Zr present (ZrB₂ — most commercially studied UHTC)
    +0.05 if Mo or Nb present

Hard anion bonuses:
  +0.35 if {Hf, Zr, Ta, Ti, Nb} + C present (UHTC carbides)
    additional +0.08 if multi-carbide (e.g., Hf + Ta + C: HfTaC solid solution)
  +0.32 if {Hf, Zr, Ta, Ti, Nb} + B present (UHTC diborides)
    additional +0.08 if SiC present with boride (SiC-ZrB₂ composites — best oxidation resistance)
  +0.25 if refractory metal + N (nitrides: TiN, ZrN, HfN)
  +0.20 if Si + C present (SiC — refractory workhorse; protective SiO₂ scale at T < 1600°C)

Stability:
  +0.22 if formation energy < −3.5 eV/atom (must be chemically stable at extreme T)
  +0.10 if decomposition energy < 0.01 eV/atom

Band gap:
  +0.15 if < 0.5 eV (metallic UHTCs — W, Re, refractory metals)
  +0.12 if > 5.0 eV (ceramic UHTCs — HfO₂, Al₂O₃-like: insulating)

Penalties:
  ×0.40 if volatile elements present: Na, K, Li, Rb, Cs, Zn, Cd, Hg, S, Se, Te
    (sublimation/decomposition at UHTC temperatures → catastrophic failure)
  ×0.50 if no hard anion (C, B, N) AND not refractory metal alloy (pure metals need reinforcement)
  ×0.60 if O present as primary anion (oxides not UHTCs — they ARE TBCs; overlap handled here)
```


---

## Domain 7: Quantum & Emerging Technologies

### 39. Qubit Host Material

**Scientific Context**: Solid-state qubits require extremely coherent quantum systems with long T₁ (energy relaxation) and T₂ (dephasing) times. Dominant platforms: (1) superconducting qubits (Al, Nb Josephson junctions on Si/sapphire — IBM, Google, Intel); (2) spin qubits in silicon (Si:P, Si/SiGe quantum dots — Intel, Delft); (3) NV centres in diamond (nitrogen-vacancy — sensing, communication); (4) neutral atoms (optical tweezers — Harvard, MIT). For solid-state hosts: zero nuclear spin isotopes preferred (²⁸Si, ¹²C), absence of paramagnetic impurities, and atomically clean surfaces.

**Scoring Logic**:
```
Qubit host material families:

Diamond/Carbon:
  +0.45 if C only present (diamond host — NV centres; highest coherence)
  +0.35 if C + N present (NV centre composition proxy — N-doped diamond)
  +0.30 if C + Si present (SiC — divacancy qubits; wafer-scale, foundry-compatible)

Silicon hosts (spin qubits):
  +0.40 if Si only present (isotopically pure ²⁸Si — longest T₂ for spin qubits)
  +0.35 if Si + Ge present (Si/SiGe heterostructure — Intel silicon spin qubit)
  +0.30 if Si + O only (SiO₂ — gate oxide for Si spin qubits; must be very clean)

Superconducting qubit substrates:
  +0.35 if Al + O (Al₂O₃ sapphire — best substrate for superconducting qubits, low dielectric loss)
  +0.30 if Si only (silicon substrate for superconducting qubits)
  +0.25 if Ti + O (TiO₂ — sapphire analog for some applications)

Rare-earth defect hosts:
  +0.30 if Y + Al + O (Y₃Al₅O₁₂ — YAG: Er-doped for quantum memory, telecom wavelength)
  +0.25 if Y + V + O (YVO₄:Eu — quantum memory at visible wavelengths)
  +0.22 if La + F (LaF₃:RE — low phonon energy, RE-doped quantum memory)

Nuclear spin environment (critical for coherence):
  +0.15 if ONLY zero-nuclear-spin elements dominant (²⁸Si, ¹²C, ¹⁶O)
    Proxy: Si, C, O only (even isotopes dominate in natural abundance after enrichment)
  ×0.60 if many high-spin nuclei present: {Co, Mn, V, Nb, Ta, In, Tl, Bi, As, P with I≠0}
    (nuclear spin bath → T₂ decoherence)

Band gap requirements:
  Hard: if band gap < 2.0 eV → ×0.3 (needs to be insulating; leakage destroys coherence)
  +0.25 if ≥ 5.5 eV (diamond-like wide gap)
  +0.20 if 3.5–5.5 eV (SiC/AlN-like)
  +0.12 if 2.0–3.5 eV (marginal; too much thermal excitation at mK temps)

Stability:
  +0.12 if formation energy < −3.0 eV/atom (ultra-pure materials need stable host)
  +0.08 if decomposition energy < 0.001 eV/atom (on hull)

Penalties:
  ×0.40 if magnetic transition metals present (Mn, Fe, Co, Ni, Cr) — magnetic noise
  ×0.50 if rare earths in host (unless explicitly RE-doped quantum memory use case)
  ×0.60 if S, Se, or Te present (no known qubit platforms in these hosts)
```

---

### 40. Topological Insulator

**Scientific Context**: Bulk-insulating materials with topologically protected metallic surface states arising from time-reversal symmetry and strong spin-orbit coupling. Surface states have linear (Dirac cone) dispersion and are protected against backscattering by non-magnetic impurities. Applications: quantum computing (Majorana-based qubits with magnetic proximity), spintronics (spin-momentum locking), and quantum metrology. Benchmark materials: Bi₂Te₃ (Eg ~0.17 eV), Bi₂Se₃ (Eg ~0.3 eV), Sb₂Te₃ (Eg ~0.3 eV), BiSb alloys.

**Scoring Logic**:
```
Hard requirements:
  if band gap < 0.02 eV → return 0.0 (metallic in bulk; no TI gap)
  if band gap > 2.0 eV → return 0.0 (too wide for strong SOC inversion)

Band gap score:
  +0.38 if 0.05–0.5 eV (ideal TI gap — Bi₂Te₃ range)
  +0.20 if 0.5–1.2 eV (wider TI gap — SnTe, Pb₁₋ₓSnₓSe)
  +0.10 if 1.2–2.0 eV (topological semimetal / weak TI range)

Heavy element SOC bonus (critical for band inversion):
  heavy_ti_elements = {Bi, Sb, Pb, Sn, Te, Se, Tl, Hg, In, Ge, As}
  heavy_count = number of these present
  +min(heavy_count × 0.15, 0.42)

Structural family bonuses:
  +0.22 if Bi + Te/Se/S (Bi₂Te₃ quintuple-layer family — paradigmatic 3D TI)
    additional +0.08 if Bi + Se (Bi₂Se₃ — largest bulk gap of family)
    additional +0.06 if Sb + Te (Sb₂Te₃ — topological)
  +0.20 if Pb + Sn + Se/Te (Pb₁₋ₓSnₓSe/Te — topological crystalline insulators, TCI)
  +0.18 if Sn + Te (SnTe — TCI, mirror symmetry protected)
  +0.15 if Bi + Sb (BiSb — first experimentally confirmed TI)
  +0.15 if Hg + Te/Se (HgTe — 2D TI in quantum well)
  +0.12 if In + Bi + (S or Se or Te) (Bi-In chalcogenide TI analogs)
  +0.10 if Tl + Bi + (S or Se) (TlBiSe₂ family)

2D TI bonus:
  +0.08 if 2D dimensionality detected (quantum spin Hall insulators — WTe₂, 1T'-MoS₂)

Stability:
  +0.08 if formation energy < −0.5 eV/atom
  +0.05 if decomposition energy < 0.05 eV/atom

Penalties:
  ×0.50 if magnetic elements (Fe, Co, Ni, Mn, Cr) — breaks time-reversal symmetry → destroys TI state
    Exception: intentional magnetic TI (MnBi₂Te₄) — complex case, apply 50% penalty only
  ×0.60 if only light elements (Z < 32) — insufficient SOC for band inversion
```

---

### 41. Topological Qubit / Majorana Host

**Scientific Context**: Majorana zero modes (MZMs) are non-Abelian anyons predicted to arise at interfaces between topological superconductors and conventional superconductors or in topological insulator-superconductor heterostructures. Microsoft's Station Q has invested heavily in InAs/Al and InSb/Al nanowire platforms. Key material requirements: large g-factor (heavy elements, spin-orbit coupling), induced superconducting gap, hard gap (no sub-gap states). Extremely active and contested research area (2023 retractions; still pursued).

**Scoring Logic**:
```
Topological superconductor host materials:
  +0.38 if In + As present (InAs — highest SOC III-V; primary Majorana platform)
  +0.35 if In + Sb present (InSb — larger g-factor than InAs)
  +0.28 if Bi + Se/Te present with indication of superconductivity (Cu_xBi₂Se₃ type)
  +0.25 if Bi + Te present (topological material for proximity effect)
  +0.22 if Sr + Ru + O (Sr₂RuO₄ — candidate chiral topological SC, contested)
  +0.20 if Fe + Te/Se (FeTe₁₋ₓSeₓ — topological surface states + intrinsic SC)
  +0.18 if Nb or Al or V present + topological material (SC proximity partner)

Band gap:
  +0.25 if 0.05–0.5 eV (TI-type gap enabling proximity effect)
  +0.15 if 0.5–1.5 eV (conventional III-V semiconductor range)
  +0.10 if < 0.05 eV (semimetallic — could work with careful band engineering)

SOC and g-factor proxy:
  +0.15 if {In, Sb, Bi, Pb, Te, Se} all heavy (high SOC elements)
  +0.10 if g-factor proxy: heavy-element III-V (In, Sb family)

Stability:
  +0.08 if formation energy < −0.5 eV/atom
  +0.05 if decomposition energy < 0.05 eV/atom

Penalties:
  ×0.50 if magnetic elements without justification (MZMs need non-magnetic host)
  ×0.60 if light elements only (no SOC → no topological protection)
  ×0.70 if O present as primary anion (oxides generally don't host Majoranas)
```

---

### 42. Superconductor

**Scientific Context**: Zero-resistance electrical conductors below critical temperature Tc. Conventional BCS superconductors (Nb, Nb₃Sn, MgB₂) need liquid He cooling. High-Tc cuprates (YBCO, BSCF) use liquid N₂ (77K). Iron-based superconductors (LaFeAsO, FeSe) discovered 2008. Hydrogen-rich compounds under pressure (LaH₁₀, ~250K) approach room temperature but require > 100 GPa. Applications: MRI (1.5–3 T: NbTi; 7+ T: Nb₃Sn), particle accelerators (LHC: NbTi), quantum computers (Al junctions at mK). The electronic structure requirement is metallic character with specific phonon-mediated or spin-fluctuation-mediated pairing.

**Scoring Logic**:
```
Hard requirement:
  if band gap > 0.5 eV → return 0.0 (must be metallic for Cooper pairing)

Metallic bonus:
  +0.20 if band gap < 0.1 eV

Superconductor families:
  Conventional BCS metals:
    +0.25 if Nb present (NbTi, Nb₃Sn — commercial SC wire; Tc 9–18K)
      additional +0.08 if Ti + Nb (NbTi — most widely deployed SC)
      additional +0.08 if Sn + Nb (Nb₃Sn — high-field SC, LHC)
    +0.22 if V + Si (V₃Si — A15 structure; Tc 17K)
    +0.20 if Mo or Re or W present (elemental/binary conventional SC)
    +0.18 if Mg + B (MgB₂ — Tc 39K, highest conventional SC; cheap)
    +0.15 if Al + Pb or Sn present (Josephson junction materials)

  Cuprate high-Tc:
    +0.35 if Cu + O + {Ba, Sr, La, Y, Bi, Tl, Hg} (cuprate family — highest Tc at ambient P)
      additional +0.10 if Y + Ba + Cu + O (YBCO — 92K; liquid N₂ cooled)
      additional +0.10 if Bi + Sr/Ba + Cu + O (BSCF family)
      additional +0.10 if Hg + Ba + Cu + O (Hg-1223 — highest Tc ambient P ~133K)

  Iron-based SC:
    +0.28 if Fe + (As or Se or P) (iron pnictide/chalcogenide family; Tc 5–56K)
      additional +0.08 if La/Nd/Sm + Fe + As + O (1111 type — LaFeAsO)
      additional +0.08 if Ba/Sr + Fe + As (122 type — BaFe₂As₂)
      additional +0.08 if Fe + Se only (11 type — FeSe, structurally simplest)

  Hydrogen-rich compounds:
    +0.25 if H + {La, Y, Ce, Th, Ca, Ba, Lu, Sc} (hydride SC; requires high pressure for synthesis)
      NOTE: CLscore will correctly penalise these for conventional synthesis difficulty

Stability:
  +0.08 if formation energy < −0.5 eV/atom
  +0.05 if decomposition energy < 0.05 eV/atom

Penalties:
  ×0.50 if band gap > 0.5 eV (not metallic — essential for SC)
  ×0.60 if large magnetic moment elements without Fe/Cu SC context (magnetic pairbreaking)
```

---

### 43. Radiation Detector / Scintillator

**Scientific Context**: Converts high-energy radiation (X-rays, γ-rays, neutrons, alpha/beta) to detectable signals. Two detector types: (1) Scintillators — absorb radiation, emit UV/visible photons (NaI:Tl, CsI:Tl, LYSO); (2) Semiconductor detectors — direct electron-hole pair generation (CdZnTe, high-purity Ge, Si). Scintillators need: high density (stopping power), wide band gap (3–6 eV for photon emission), high Z elements (photoelectric absorption), fast decay time (timing applications), and high light yield. Semiconductor detectors: narrow band gap (0.5–2.5 eV), excellent charge transport.

**Scoring Logic**:
```
High-Z element stopping power (critical for both types):
  high_z_elements = {Ba, I, Cs, Bi, Pb, Tl, Hg, W, Lu, Gd, Xe, La, Ce, In, Sn, Te}
  high_z_count = number of these present
  +min(high_z_count × 0.14, 0.42)

Scintillator path (band gap 2.5–7.0 eV):
  +0.30 if band gap in 3.0–6.0 eV (scintillator emission window)
  +0.15 if band gap 2.5–3.0 eV (lower edge; some scintillators)

  Alkali halide scintillators:
    +0.28 if {Na, Cs, K, Ba, Sr} + {I, Br, Cl, F} (NaI:Tl, CsI:Tl, BaI₂:Eu family)
      additional +0.10 if Ba + I (BaI₂:Eu — best energy resolution inorganic scintillator)
      additional +0.08 if Cs + I (CsI:Tl — widely deployed)

  Oxide scintillators:
    +0.25 if O + {Lu, Gd, Bi, Y, Ce, La} (LYSO, GSO, BGO family)
      additional +0.10 if Lu + Si + O (LYSO — fastest heavy oxide scintillator; PET/TOF)
      additional +0.08 if Bi + Ge + O (BGO — classic, high density, slow)
      additional +0.08 if Gd + Si + O (GSO — Gd for neutron capture)

  Neutron detection:
    +0.15 if Gd present (highest thermal neutron capture cross-section of stable elements)
    +0.12 if Li present (¹Li enrichment for neutron detection — Li-glass, LiI)
    +0.10 if B present (¹⁰B neutron absorber)

Semiconductor detector path (band gap 0.5–2.5 eV):
  +0.22 if Cd + Zn + Te (CdZnTe/CZT — room-temperature γ detector; best energy resolution)
  +0.20 if Hg + I present (HgI₂ — historically important room-T detector)
  +0.18 if Pb + I (perovskite X-ray detectors — high sensitivity)
  +0.15 if Tl + Br (TlBr — room-T detector with high stopping power)
  +0.12 if Ga + As or In + P (III-V semiconductor detectors)

Stability:
  +0.08 if formation energy < −2.0 eV/atom

Penalties:
  ×0.60 if no high-Z element AND low density proxy (light materials have poor stopping power)
  ×0.50 if Tl, Hg (if not explicitly the detector material — toxic)
```

---

### 44. Multiferroic

**Scientific Context**: Materials exhibiting two or more primary ferroic orders simultaneously — most commonly ferroelectricity + ferromagnetism (or antiferromagnetism). The magnetoelectric coupling enables electric-field control of magnetism (MERAM, low-power logic). BiFeO₃ is the benchmark (Tc ~640K ferroelectric, TN ~640K antiferromagnetic), but weak magnetoelectric coupling. Type-II multiferroics (RMnO₃ hexagonal family) have strong coupling but low Tc. Single-phase room-temperature multiferroics with large coupling remain an open materials challenge.

**Scoring Logic**:
```
Ferromagnetic/antiferromagnetic elements:
  if no magnetic element (Fe, Co, Ni, Mn, Cr, REE) → return 0.0
  +0.25 if Fe present (BiFeO₃ family; also Fe₃O₄ magnetoelectric)
  +0.22 if Mn present (hexagonal RMnO₃ family; TbMnO₃, DyMnO₃)
  +0.18 if Cr present (Cr₂O₃ — magnetoelectric, not strictly multiferroic)
  +0.15 if Co present (Co-based multiferroics)
  +0.20 if REE (Tb, Dy, Ho, Gd, Nd, Eu) + Mn present (type-II multiferroics)

Ferroelectric enablers:
  Symmetry: if polar space group → +0.30
  if non-centrosymmetric → +0.18
  if centrosymmetric → ×0.3
  +0.20 if Bi + Fe + O (BiFeO₃ — only known room-T single-phase multiferroic with large Ps)
  +0.18 if perovskite ABO₃ with magnetic B-site + O (e.g., PbFe₀.₅Nb₀.₅O₃)
  +0.15 if Bi + (TM) + O (Bi lone pair drives ferroelectricity — strong FE candidate)

Band gap:
  +0.20 if 1.0–3.5 eV (multiferroic window — FE + FM coexist)
  +0.10 if 3.5–5.0 eV (wider gap; possible but FE dominates over FM coupling)
  ×0.40 if < 0.5 eV (metallic; ferroelectric polarisation cannot be sustained)

Stability:
  +0.10 if formation energy < −2.5 eV/atom

Penalties:
  ×0.70 if Pb present (PbTiO₃-type ferroelectrics don't have magnetic order; regulatory)
  ×0.50 if no oxygen (most multiferroics are oxides)
  ×0.60 if metallic
```


---

## Domain 8: Biomedical & Environmental

### 45. Biodegradable Implant

**Scientific Context**: Metallic implants (orthopaedic screws, cardiovascular stents) that safely dissolve in the body at a controlled rate (0.1–0.5 mm/year), eliminating removal surgery. Mg alloys (Mg-Zn, Mg-Ca, Mg-Sr) are the primary candidate — Mg²⁺ is biocompatible, abundant in the body (0.3 mol/day dietary requirement). Zn alloys also promising (slower degradation, ~0.02 mm/year). Fe alloys degrade too slowly. Key requirements: biocompatibility, mechanical strength, controlled corrosion rate, non-toxic degradation products.

**Scoring Logic**:
```
Primary biodegradable metal families:
  +0.40 if Mg present (primary biodegradable metal — Mg-Zn, Mg-Ca, Mg-Sr)
    additional +0.10 if Mg + Zn present (Mg-Zn alloys — controlled corrosion, good strength)
    additional +0.08 if Mg + Ca present (Mg-Ca — Ca is essential bone mineral)
    additional +0.08 if Mg + Sr present (Mg-Sr — Sr promotes osteogenesis)
  +0.32 if Zn present without Mg (Zn-based — slower, better for load-bearing)
    additional +0.08 if Zn + Mg (see above)
    additional +0.06 if Zn + Cu (antimicrobial degradable stent)
  +0.20 if Fe present without Mg or Zn (Fe-based — very slow; limited applications)
    additional +0.08 if Fe + Mn (Fe-Mn — faster corrosion than pure Fe, antiferromagnetic)

Alloying elements (biocompatible):
  +0.12 if Ca present (osteogenic; bone regeneration)
  +0.10 if Si present (silicate hydrolysis products — biocompatible)
  +0.08 if Sr present (promotes bone cell activity)
  +0.06 if P + O present (phosphate — biomineralization affinity)

Band gap / electronic:
  +0.10 if < 0.5 eV (metallic character — structural integrity)

Biocompatibility stability:
  +0.10 if formation energy −1.0 to −3.0 eV/atom (stable enough to synthesise, not so stable it won't degrade)

Penalties:
  ×0.0 if ANY of: Tl, Hg, Cd, As, Be, Ba (systemic toxicity — absolute disqualifier for implant)
  ×0.0 if radioactive elements
  ×0.50 if Pb present (chronic toxicity)
  ×0.60 if REE present (uncertain in-vivo effects)
  ×0.70 if NSites > 20 (complex intermetallics — inconsistent degradation rates)
  ×0.70 if no biodegradable metal (Mg, Zn, Fe, Ca) present
```

---

### 46. Bone Scaffold / Hydroxyapatite Analog

**Scientific Context**: Calcium phosphate ceramics for bone regeneration, dental implant coatings, and drug delivery. Hydroxyapatite (HA: Ca₁₀(PO₄)₆(OH)₂) is the primary inorganic component of bone (70 wt%). Requirements: Ca/P ratio ~1.67 (HA stoichiometry), osseointegration, controlled resorption (tricalcium phosphate TCP: faster than HA), biocompatibility. Bioglass (45S5: SiO₂-CaO-Na₂O-P₂O₅) bonds to both hard and soft tissue. HA coatings on Ti/Co-Cr implants improve osseointegration.

**Scoring Logic**:
```
Mandatory elements:
  if Ca not in elements → ×0.3 (must contain calcium for bone affinity)
  if P not in elements AND Si not in elements → ×0.4 (needs phosphate or silicate)
  Hard disqualifier: if ANY toxic element present (Tl, Hg, Cd, As, Pb, Be) → return 0.0

Ca-P family (hydroxyapatite analogs):
  +0.40 if Ca + P + O present (calcium phosphate family)
    additional +0.12 if Ca + P + O + H present (HA analog — OH group proxy)
    additional +0.10 if Ca/P ratio ~1.67 (can be estimated from formula stoichiometry)
    additional +0.08 if Ca + P + O only, no Na/Mg (stoichiometric HA/TCP)

Bioglass analogs:
  +0.30 if Ca + Si + O + Na present (bioglass — 45S5 family)
  +0.25 if Ca + Si + O present without Na (calcium silicate — wollastonite, diopside)

Substitution-doped HA (enhances biological activity):
  +0.10 if F + Ca + P + O (fluorapatite — reduced solubility, harder)
  +0.10 if Sr + Ca + P + O (Sr-doped HA — enhanced osteoblast activity)
  +0.08 if Mg + Ca + P + O (Mg-doped HA — closer to biological apatite)
  +0.08 if Zn + Ca + P + O (Zn-doped HA — antibacterial)
  +0.07 if Si + Ca + P + O (Si-substituted HA — enhanced bone growth)

Stability:
  +0.10 if formation energy < −3.0 eV/atom (long-term in-vivo stability)

Band gap:
  +0.08 if ≥ 4.0 eV (insulating ceramic — appropriate for bone substitute)

Penalties:
  ×0.0 if ANY: Tl, Hg, Cd, As, Pb, Be (absolute biomedical disqualifier)
  ×0.60 if no Ca (calcium-free scaffold has poor osseointegration)
  ×0.70 if transition metals (Fe, Co, Ni) as primary elements (metals leach; cytotoxic)
```

---

### 47. Antibacterial Coating

**Scientific Context**: Surface coatings that kill or inhibit bacteria via ion release (Ag⁺, Cu²⁺, Zn²⁺), reactive oxygen species (TiO₂ photocatalysis), or contact killing (quaternary ammonium). Critical for hospital surfaces, medical devices, food processing, and water treatment. Ag is the gold standard (broad-spectrum, < 1 ppm effective). Cu is cheaper and effective against COVID-19 and influenza on touch surfaces. ZnO under UV is photocatalytic. Key requirement: sustained ion release without cytotoxicity.

**Scoring Logic**:
```
Antibacterial metal ions:
  +0.40 if Ag present (Ag⁺ — broadest spectrum antibacterial ion; gold standard)
  +0.35 if Cu present (Cu²⁺ — highly effective, cheaper than Ag; antiviral too)
  +0.28 if Zn present (Zn²⁺ — antibacterial, essential nutrient → lower toxicity)
  +0.20 if Ti + O present (TiO₂ — photocatalytic ROS generation under UV)
  +0.18 if Se present (Se nanoparticles — antifungal and antibacterial)
  +0.15 if Bi present (Bi³⁺ — antibacterial, used in Pepto-Bismol; minimal toxicity)
  +0.12 if Ce + O present (CeO₂ — antioxidant and antibacterial)
  +0.10 if Fe + O present (iron oxide nanoparticles — magnetic + antibacterial)
  +0.08 if Co or Ni + O (metal oxide antibacterials — lower efficacy)

Sustained release mechanisms:
  +0.12 if Ag or Cu + silicate (slow ion release from ceramic matrix)
  +0.10 if Ag or Cu + phosphate (sustained delivery in bone contact coatings)
  +0.08 if halide present with Ag (AgCl/AgBr — photosensitive ion release)

Band gap (photocatalytic applications):
  +0.10 if 2.8–3.5 eV + Ti or Zn (photocatalytic ROS under UV/near-UV)
  +0.08 if 2.0–2.8 eV (visible-light photocatalytic antibacterial — highly desirable)

Stability:
  +0.08 if formation energy < −1.0 eV/atom

Penalties:
  ×0.0 if: Tl, Hg, As, Be, radioactive elements (cytotoxic — disqualified for coating)
  ×0.50 if Pb present
  ×0.60 if only refractory metals with no ion release mechanism (Zr, W, Ta alone — not antibacterial)
  ×0.70 if band gap > 5.0 eV AND no metal ion release mechanism (inert — no antibacterial activity)
```

---

### 48. CO₂ Capture Sorbent

**Scientific Context**: Direct air capture (DAC) and post-combustion capture require materials that selectively bind CO₂ (400 ppm in air; 10–15% in flue gas) at ambient/elevated temperature and release it under mild heating (< 150°C). Commercial: amine-functionalized sorbents, KOH aqueous, CaO looping. Solid sorbents: zeolites, MOFs, alkali metal carbonates, amine-grafted silicas. Key descriptors: CO₂ adsorption enthalpy (−30 to −90 kJ/mol ideal), working capacity, cycle stability, and cost. This scorer targets inorganic crystalline sorbents.

**Scoring Logic**:
```
CO₂ capture material families:
  +0.35 if Ca + O present (CaO — calcium looping, high capacity but 900°C regeneration)
    additional +0.08 if Ca + O + Si (CaSiO₃ — silicate carbonation, geological analogue)
  +0.30 if K or Na + O present (K₂CO₃, Na₂CO₃ — mild regeneration ~150°C, humidity swing)
    additional +0.08 if K + O + Al + Si (zeolite-like framework — selective CO₂ adsorption)
  +0.28 if Li + O present (Li₄SiO₄, Li₂ZrO₃ — high-T sorbents, fast kinetics)
    additional +0.08 if Li + Si + O (Li₄SiO₄ — excellent high-T performance)
    additional +0.06 if Li + Zr + O (Li₂ZrO₃ — fast CO₂ uptake at 500°C)
  +0.25 if Mg + O present (MgO, dolomite — moderate-T sorbent)
  +0.22 if Ba + O present (BaO — CO₂ sorbent in chemical looping)
  +0.20 if Al + Si + O + (Na or K) (zeolite — selective physisorption, low T)
    additional +0.10 if Na + Al + Si + O: 13X, 4A zeolite type
  +0.18 if Zr + O present (ZrO₂ — emerging high-T CO₂ sorbent)

Alkalinity/basicity proxy:
  +0.10 if alkali or alkaline earth dominant (basic surface → acid CO₂ adsorption)

Stability:
  +0.12 if formation energy < −3.0 eV/atom
  +0.08 if decomposition energy < 0.05 eV/atom

Band gap:
  +0.05 if ≥ 4.0 eV (insulating sorbent — no competing redox reactions)

Penalties:
  ×0.60 if transition metals dominant without basic oxide character (no CO₂ affinity)
  ×0.50 if Tl, Hg, Cd, As (toxic sorbent — leaching risk in DAC contactors)
  ×0.70 if halide dominant (halides don't capture CO₂; no carbonate formation pathway)
  ×0.40 if acidic oxide dominant (SiO₂-only, TiO₂-only) — repels CO₂
```

---

### 49. Desalination Membrane Material

**Scientific Context**: Removes salt from seawater (35,000 ppm NaCl) or brackish water for drinking. Reverse osmosis (RO) uses pressure-driven transport through dense membranes (rejection > 99.7%). Emerging: graphene oxide membranes (sub-nm channels), MXene membranes (Ti₃C₂Tₓ), zeolite membranes (MFI, LTA type). Requirements: high water permeance, near-100% salt rejection, mechanical strength, chemical stability (chlorine resistance). This scorer targets inorganic crystalline membrane materials.

**Scoring Logic**:
```
Zeolite/molecular sieve framework:
  +0.38 if Al + Si + O (aluminosilicate zeolite — MFI/LTA type, proven water softening)
    additional +0.10 if Na + Al + Si + O (NaA/LTA — very tight 0.4 nm pores)
    additional +0.08 if Al + Si + O only (silicalite/MFI — hydrophobic, organics)
  +0.30 if Si + O only (pure silica zeolite — hydrophobic, lower Na⁺ rejection but organic selective)
  +0.25 if Al + P + O (AlPO₄ — AlPO molecular sieves, tunable pore size)

2D layered materials (angstrom channels):
  +0.32 if Ti + C present (MXene proxy: Ti₃C₂ — sub-nm channels, emerging RO membrane)
  +0.28 if C only (graphene oxide proxy — sub-nm ion sieving; record permeance)
  +0.25 if Mo + S present (MoS₂ — 2D layered, sub-nm water transport)
  +0.22 if V + O layered (V₂O₅ — 2D oxide membrane)

Oxide membranes:
  +0.20 if Al + O (γ-Al₂O₃ — nanofiltration support membrane)
  +0.18 if Zr + O (ZrO₂ — ultrafiltration ceramic membrane)
  +0.15 if Ti + O (TiO₂ — photocatalytic self-cleaning membrane)

Chemical stability:
  +0.12 if formation energy < −3.0 eV/atom (must survive chlorination cycles)
  +0.08 if decomposition energy < 0.02 eV/atom

Band gap:
  +0.05 if ≥ 5.0 eV (insulating — no electrochemical side reactions)

Penalties:
  ×0.50 if no structural framework (molecular sieves need periodic pore structure)
  ×0.60 if Tl, Hg, Cd, As, Pb (toxic — leaching into drinking water)
  ×0.40 if magnetic or metallic elements dominant (no filtration selectivity)
  ×0.70 if chalcogenide dominant (S, Se, Te) — dissolve in chlorinated water
```

---

### 50. Photocatalytic Pollutant Degradation

**Scientific Context**: UV or visible-light-driven mineralisation of organic pollutants (dyes, pharmaceuticals, pesticides, PFAS) in water via reactive oxygen species (•OH, O₂•⁻, h⁺). TiO₂ anatase (3.2 eV, UV) is the benchmark. The push is toward visible-light-active materials (Eg < 3.0 eV) to use solar energy efficiently. Key systems: Bi₂WO₆, g-C₃N₄, BiVO₄, Ag₃PO₄, Z-scheme composites. Band gap should be 2.0–3.2 eV for optimal visible-light activity. Distinct from water splitting: no H₂ production required, only oxidation needed.

**Scoring Logic**:
```
Hard requirements:
  if band gap < 1.5 eV → return 0.0 (no photogenerated holes at sufficient potential)
  if band gap > 4.0 eV → return 0.0 (UV only; very low solar fraction)

Band gap score (peak at 2.7 eV):
  core_score = max(0.35 − |band_gap − 2.7| × 0.12, 0.12)
  +0.10 if 2.0–3.2 eV (ideal visible-light photocatalyst range)
  +0.08 if 3.2–4.0 eV (UV-active but still partial solar; TiO₂ range)
  +0.05 if 1.5–2.0 eV (too narrow; VB holes may lack sufficient oxidation potential)

Material families:
  +0.28 if Ti + O (TiO₂ — benchmark; anatase phase preferred)
  +0.25 if Bi + V + O (BiVO₄ — 2.4 eV, leading visible photocatalyst)
  +0.22 if Bi + W + O (Bi₂WO₆ — 2.7 eV, excellent visible activity)
  +0.20 if Bi + Mo + O (Bi₂MoO₆)
  +0.18 if Ag + P + O (Ag₃PO₄ — very high activity but photocorrosion)
  +0.15 if Zn + O (ZnO — wide gap but excellent ROS generation)
  +0.15 if W + O (WO₃ — 2.7 eV, visible active, stable)
  +0.12 if Fe + O (α-Fe₂O₃ — 2.2 eV; poor hole mobility but earth-abundant)
  +0.10 if Ce + O (CeO₂ — 3.0–3.4 eV; redox Ce³⁺/Ce⁴⁺ for ROS)
  +0.08 if Sn + O (SnO₂ — UV active, good for coupling with visible sensitiser)

Visible-light absorption enhancement:
  +0.10 if halide present (halide perovskites — emerging pollutant degradation)
  +0.08 if S or N present (doped oxides — red-shifts absorption)

Stability:
  +0.10 if formation energy < −2.5 eV/atom (photocorrosion resistance)

Penalties:
  ×0.60 if Cd, Tl, Hg, or As (toxic — would leach into water being treated)
  ×0.50 if only non-photocatalytically-active elements (Zr, Al, Si, Mg alone)
  ×0.70 if band gap > 4.0 eV (transparent to solar radiation)
```


---

## Viability Scoring

The viability score is a multiplicative modifier applied to all raw application scores, capturing real-world constraints that determine whether a high-scoring material is actually usable.

$$\text{weighted\_score} = \text{application\_score} \times \text{viability\_multiplier}$$

The viability multiplier is itself a product of six independent components:

$$V = V_{\text{cost}} \times V_{\text{abundance}} \times V_{\text{supply}} \times V_{\text{radioactive}} \times V_{\text{REE}} \times V_{\text{CLscore}}$$

---

### Component 1: Material Cost Score

**Purpose**: Materials costing more than ~$120/kg cannot be commercially deployed at scale, regardless of performance.

$$V_{\text{cost}} = \max\!\left(0,\ 1 - \frac{\sum_i w_i \cdot p_i}{120}\right)$$

Where $w_i$ is the weight fraction of element $i$ and $p_i$ is its price in USD/kg.

**Element Price Reference Table** (USD/kg, update periodically):

| Element | Price ($/kg) | Notes |
|---------|-------------|-------|
| Li | 8.0 | Battery-grade carbonate ~$15/kg 2024 |
| Na, K | 0.5 | Commodity salts |
| Mg, Al, Ca | 2–3 | Industrial metals |
| Si | 2.5 | Abundant semiconductor |
| Fe, Mn | 0.5–1 | Commodity metals |
| Cu | 9 | LME spot price variable |
| Zn | 3 | Commodity |
| Co | 33 | Supply-concentrated (DRC); volatile |
| Ni | 14 | Battery demand driving price |
| Ti | 11 | Aerospace, industrial |
| Sn | 25 | Commodity semiconductor |
| Bi | 6 | Cheap heavy metal |
| Sb | 6 | Supply concerns |
| Ba | 0.3 | Barium carbonate cheap |
| Sr | 1 | Strontium carbonate |
| V | 30 | Ferrovanadium price-driven |
| Mo | 40 | Molybdenite mining |
| W | 35 | Tungsten carbide grinding |
| Nb | 40 | Brazilian monopoly |
| Ta | 150 | Coltan; very expensive |
| Hf | 900 | Hafnium — refinery byproduct |
| In | 167 | Scarce; ITO demand |
| Ge | 1000 | Very expensive semiconductor |
| Ga | 200 | Gallium — byproduct of Al smelting |
| Ru | 14,000 | PGM — electrolyser catalyst |
| Rh | 147,000 | Most expensive element |
| Ir | 52,000 | PGM — OER catalyst |
| Pt | 31,000 | PGM — HER benchmark |
| Pd | 49,000 | PGM |
| Au | 62,000 | Noble metal |
| REEs (average) | 30–200 | Wide range; Nd ~80, Dy ~350 |
| Unknown default | 50 | Conservative estimate |

---

### Component 2: Elemental Abundance Score

**Purpose**: Rare elements face supply shortages that persist regardless of price.

$$V_{\text{abundance}} = \text{clip}\!\left(\frac{\log_{10}(\text{min\_abundance} + 0.001)}{\log_{10}(282000)},\ 0,\ 1\right)$$

Where `min_abundance` is the Earth crustal abundance (ppm) of the least abundant element in the compound.

**Abundance Reference** (ppm by mass in Earth's crust):

| Element | Abundance (ppm) | Score |
|---------|----------------|-------|
| O | 461,000 | ~1.0 |
| Si | 282,000 | 1.0 |
| Al, Fe, Ca | 28,000–82,000 | ~0.9 |
| Na, K, Mg | 2,000–28,000 | ~0.8 |
| Ti, Mn | 600–900 | ~0.7 |
| P, Ba, Sr | 170–425 | ~0.7 |
| Zr, Cr, V | 100–165 | ~0.65 |
| Ni, Cu, Zn | 20–80 | ~0.6 |
| Co | 25 | ~0.6 |
| Sn, Nb | 2 | ~0.5 |
| Ga, Li | 15–20 | ~0.58 |
| In | 0.25 | ~0.35 |
| Te | 0.001 | ~0.1 |
| REEs (average) | 1–60 | ~0.4–0.6 |
| Pt, Ir, Rh | < 0.001 | ~0.1 |
| Au | 0.004 | ~0.15 |

---

### Component 3: Supply Risk Score

**Purpose**: Geopolitically concentrated supply chains create risk regardless of price or abundance.

$$V_{\text{supply}} = 1 - \frac{N_{\text{critical}}}{N_{\text{elements}}}$$

**USGS 2023 Critical Minerals List** (50+ elements including):

| Category | Elements |
|----------|----------|
| Light REEs | Nd, Pr, Sm, Gd, La, Ce, Y, Sc |
| Heavy REEs | Dy, Tb, Er, Ho, Tm, Yb, Lu, Eu |
| Battery metals | Li, Co, Ni |
| Specialty semiconductors | Ga, Ge, In |
| Refractory/strategic | Hf, Nb, Ta, Mo, W, Re, Sb |
| Chalcogenide-related | Se, Te |
| Platinum group | Pt, Pd, Rh, Ir, Ru, Os |
| Others | Be, Bi, Cs, Rb, Sn (for electronics) |

---

### Component 4: Radioactive Filter

**Purpose**: Hard reject for materials containing radioactive elements.

$$V_{\text{radioactive}} = \begin{cases} 0.0 & \text{if any radioactive element present} \\ 1.0 & \text{otherwise} \end{cases}$$

**Hard-reject elements**: Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr, Tc, Po, At, Rn, Fr, Ra

**Note**: Bi (Z=83) is technically the heaviest stable element. Elements Z>83 are all radioactive. Some scoring functions use Z>83 as an additional proxy for this filter.

---

### Component 5: Rare Earth Element Penalty

**Tier 1 — Heavy REE (×0.30 per element)**:
Dy, Tb, Eu, Ho, Er, Tm, Lu, Yb
*Rationale*: Extremely high supply concentration (China >95%), lowest crustal abundance among REEs, most geopolitically sensitive.

**Tier 2 — Moderate REE (×0.60 per element)**:
Nd, Pr, Sm, Gd, Sc
*Rationale*: Supply-constrained, essential for permanent magnets but subject to export controls.

**Tier 3 — Mild REE (×0.85 per element)**:
La, Ce, Y
*Rationale*: Relatively abundant (La/Ce are ~60 ppm), but strategic importance justifies mild penalty.

**Application override**: Permanent magnet scorer legitimately requires REEs for anisotropy — the viability penalty is accepted because (BH)max is so much higher that $/Watt still favours REE magnets for EV motors.

---

### Component 6: CLscore Penalty

$$V_{\text{CLscore}} = \begin{cases}
1.0 & \text{CLscore} \geq 0.50 \\
0.8 & 0.40 \leq \text{CLscore} < 0.50 \\
0.7 & 0.30 \leq \text{CLscore} < 0.40 \\
0.4 & 0.10 \leq \text{CLscore} < 0.30 \\
0.1 & \text{CLscore} < 0.10 \\
0.5 & \text{CLscore} = -1 \text{ (unknown)}
\end{cases}$$

---

## CLscore: Synthesizability Prediction

CLscore is a **synthesizability probability** (0.0–1.0) from KAIST's Synthesizability-PU-CGCNN: a graph neural network trained on crystal structures using Positive-Unlabeled learning. It predicts whether a computationally stable crystal structure is likely to be successfully synthesized under conventional laboratory conditions (solid-state, hydrothermal, CVD, sputtering).

### Architecture

1. **Input**: CIF crystal structure → PyMatGen → neighbour graph (radius=8Å, max_neighbours=12)
2. **Node features**: 92-dimensional one-hot element encoding
3. **Edge features**: interatomic distances + pair features (up to 41 dimensions)
4. **Graph convolution**: 3 stacked CGCNN layers with residual connections
5. **Global pooling**: mean aggregation over all atoms
6. **FC layers**: 128-dim hidden → 2-class softmax
7. **Ensemble**: 100 checkpoint bags → mean probability = CLscore

### Interpretation

| CLscore | Interpretation | Recommended Action |
|---------|---------------|-------------------|
| ≥ 0.80 | Very likely synthesisable | Prioritise for experimental validation |
| 0.50–0.80 | Likely synthesisable | Good candidate; assess synthesis route |
| 0.30–0.50 | Moderately optimistic | Creative synthesis strategy needed |
| 0.10–0.30 | Risky | High-pressure, unusual conditions may be required |
| < 0.10 | Very unlikely | Kinetically trapped; unlikely accessible |
| −1.0 | Unknown | CIF unavailable or parsing failed |

### Known Limitations

1. Does not model: hydrothermal, electrochemical, or sol-gel synthesis routes (solid-state bias)
2. Does not account for high-pressure synthesis advantages (important for hydrides)
3. Trained on DFT-relaxed structures; slightly different from experimental structures
4. No kinetic pathway prediction — only thermodynamic/structural likelihood
5. CIF required — materials without crystal structure files cannot be scored

---

## Additional Datasets

Beyond GNoME, the following databases can be integrated to improve coverage:

| Database | Structures | Unique Value | Integration Notes |
|----------|-----------|-------------|-----------------|
| **AFLOW** | ~3.5M | Largest raw count; alloy/intermetallic coverage | Different DFT settings; deduplication required |
| **JARVIS-DFT** | ~80k | Pre-computed optical properties, beyond-DFT (OptB88vdW, mbJ) | Best for solar/LED scorers (absorption spectra vs band gap proxy) |
| **OQMD** | ~1M | Alloy/intermetallic coverage; good for magnets, anodes | Older GGA; compatible with GNoME |
| **Materials Project** | ~154k | Rich property data: elastic, dielectric, piezo, magnetic moments | Highest data quality per entry; best for ferroelectric/piezo scorers |
| **NOMAD** | Variable | Entries linked to experimental papers (provenance) | Best for refining is_experimental flag accuracy |
| **CCDC (CSD)** | ~1.2M | Organic/MOF crystal structures | Relevant for CO₂ sorbent, membrane, drug delivery categories |
| **ICSD** | ~250k | Experimental crystal structures only | Ground truth for synthesisability assessment |

**Deduplication strategy**: Match on reduced formula + space group number + volume tolerance (±5%). When duplicates exist, prefer: ICSD > Materials Project > JARVIS > OQMD > AFLOW > GNoME for property data quality.

---

## Pipeline Architecture

```
src/matintel/
├── config.py              # APP_LABELS (56 apps), element prices, critical minerals
├── data_sources.py        # Multi-database load/validation, schema normalisation
├── features.py            # Matminer feature extraction (138 features)
├── scoring.py             # 56-category scoring functions + registry
├── viability.py           # Cost, abundance, supply risk, radioactivity, REE, CLscore
├── clscore.py             # CLscore prediction wrapper (KAIST integration)
├── pipeline.py            # Orchestration: load → featurize → score → viability
├── deduplication.py       # Cross-database duplicate detection and merging
└── explanations.py        # AI summary generation for top candidates
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+ (3.13 tested)
- Windows PowerShell 5.1+ (primary OS) or Linux/macOS bash
- ~8 GB disk (GNoME CIFs + pip packages + additional databases)

### Step 1: Clone and Environment

```powershell
cd c:\Users\rosha\Downloads\MatIntel
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Primary Data Download

```powershell
./scripts/download_gnome_csv.ps1        # GNoME metadata CSV (~500 MB)
./scripts/download_gnome_cifs.ps1       # CIF files (~455 MB, for CLscore)
```

### Step 3: Optional Additional Databases

```powershell
./scripts/download_jarvis.ps1           # JARVIS-DFT JSON (~200 MB)
./scripts/download_mp_experimental.ps1 # Materials Project experimental set
./scripts/build_experimental_reference.py  # Requires MP_API_KEY env var
```

### Step 4: CLscore Setup

```powershell
./scripts/setup_clscore.ps1  # Clones KAIST repo, installs torch, verifies checkpoints
```

---

## Running the System

### Full Pipeline

```powershell
./scripts/run_pipeline.ps1    # Load → Featurize → Score (56 apps) → Viability → Export
./scripts/run_app.ps1          # Launch Streamlit dashboard at http://localhost:8501
```

### CLscore Batch Processing

```powershell
# Score top 1000 candidates for a specific application
python run_clscore.py --app solar_singlejunction --top-n 1000 --cif-dir data/cifs

# Score all 554k materials in speed mode (1 checkpoint, ~7 hours)
python run_clscore_all.py --output-csv data/processed/clscore_all_results.csv --max-models 1

# Recompute unknown CLscores only
python run_clscore_all.py --recompute-unknown
```

### Export Outputs

From the Streamlit dashboard:
- Ranked CSV export for any application
- CIF ZIP archive for top candidates
- PDF summary report with scoring breakdown

---

## File Outputs

| File | Description |
|------|-------------|
| `data/processed/working_dataset.csv` | Validated input data, pre-feature |
| `data/processed/featured_dataset.csv` | + 138 matminer features |
| `data/processed/scored_dataset.csv` | + 56 score columns, viability, CLscore, best_score |
| `data/processed/clscore_all_results.csv` | CLscore cache (resumable) |
| `data/processed/experimental_compounds.csv` | MP/JARVIS experimental reference set |
| `data/processed/top10_per_category.csv` | Top 10 per app (viability-adjusted) with explicit provenance columns |
| `data/processed/top10_per_category_raw_score.csv` | Top 10 per app (raw score) with explicit provenance columns |
| `logs/pipeline.log` | Timestamped execution log |

---

## References

- **GNoME Dataset**: Merchant et al., Nature 624, 80–85 (2023). Google DeepMind.
- **CLscore Model**: Noh et al., Matter 3, 873–891 (2020). KAIST Synthesizability-PU-CGCNN.
- **Materials Project**: Jain et al., APL Materials 1, 011002 (2013).
- **JARVIS-DFT**: Choudhary et al., npj Computational Materials 6, 173 (2020).
- **AFLOW**: Curtarolo et al., Computational Materials Science 58, 218–226 (2012).
- **OQMD**: Kirklin et al., npj Computational Materials 1, 15010 (2015).
- **Matminer**: Ward et al., Computational Materials Science 152, 60–69 (2018).
- **Shockley-Queisser limit**: Shockley & Queisser, Journal of Applied Physics 32, 510 (1961).
- **USGS Critical Minerals 2023**: https://www.usgs.gov/critical-minerals


---

## Domain 9: Nuclear & Extreme Environment

### 51. Nuclear Fuel Cladding

**Scientific Context**: Fuel cladding encases uranium oxide fuel pellets in nuclear reactors, separating the fuel from the coolant while conducting heat out of the core. Must survive: (1) neutron bombardment causing radiation swelling and embrittlement; (2) operating temperatures of 300–400°C (LWR) or up to 1200°C during loss-of-coolant accidents; (3) zirconium-water reaction above 1200°C generating explosive H₂; (4) corrosion by high-pressure water/steam coolant. Commercial standard: Zircaloy-4 (Zr-1.5Sn-0.2Fe-0.1Cr). Accident-tolerant fuel (ATF) replacements: FeCrAl alloys, SiC/SiC composites, coated Zr. Key property: low neutron capture cross-section (must be nearly transparent to neutrons).

**Scoring Logic**:
```
Hard disqualifiers:
  if radioactive elements present → return 0.0
  if high neutron absorbers dominant: {B, Cd, Gd, Hf, In, Ir, Rh, Sm, Eu} as primary element → ×0.2
    (these capture neutrons — catastrophic for reactor neutron economy)

Zirconium-based cladding (commercial standard):
  +0.40 if Zr present (Zircaloy basis — lowest neutron capture σ among structural metals)
    additional +0.10 if Zr + Sn (Zircaloy-2/4 alloying)
    additional +0.08 if Zr + Nb (Zr-1%Nb, Zr-2.5%Nb — VVER/CANDU cladding)
    additional +0.08 if Zr + Fe + Cr (Zircaloy-4 composition)

Accident-tolerant fuel alternatives:
  +0.32 if Fe + Cr + Al present (FeCrAl — forms protective Al₂O₃ at 1200°C; ATF candidate)
    additional +0.08 if Fe + Cr + Al + Y/La (ODS steel — oxide dispersion strengthened)
  +0.28 if Si + C present (SiC — very low neutron activation, ceramic composite cladding)
    additional +0.10 if Si + C only (pure SiC/SiC composite)
  +0.22 if Cr-coated Zr proxy: Cr + Zr (Cr-coated Zircaloy — near-term ATF deployment)
  +0.18 if Mo present + Zr or Fe (Mo alloys — refractory cladding research)

Radiation resistance proxy:
  +0.12 if cubic crystal system (cubic symmetry → isotropic radiation swelling)
  +0.10 if high melting point metals: {W, Mo, Nb, Ta, Re} (radiation-stable refractory)
  +0.08 if formation energy < −3.0 eV/atom (thermodynamically stable under radiation)

Corrosion resistance:
  +0.10 if Cr or Al + refractory metal (protective oxide layer in steam)
  +0.08 if formation energy < −2.5 eV/atom

Band gap:
  +0.08 if < 0.5 eV (metallic — required for heat conduction through cladding)

Penalties:
  ×0.0 if B, Cd, Gd, Sm, Eu as primary cation (neutron poison — reactor killer)
  ×0.50 if alkali metals (Na, K, Li) present (react with water coolant)
  ×0.60 if S, Se, or Te present (corrosion accelerators in reactor environment)
  ×0.40 if low melting point metals (Sn, Pb, Bi, In) as primary element (melt during LOCA)
```

---

### 52. Tritium Breeder Material

**Scientific Context**: In deuterium-tritium fusion reactors (ITER, DEMO, ARC), the blanket must breed its own tritium fuel via neutron capture: ⁶Li + n → ⁴He + T (Q = 4.78 MeV). Lithium-containing ceramics are the primary candidates because: (1) Li has high tritium breeding ratio (TBR); (2) ceramic forms allow high-temperature operation (600–900°C); (3) tritium must be extracted by helium purge gas. Leading candidates: Li₄SiO₄ (lithium orthosilicate), Li₂TiO₃ (lithium metatitanate), Li₂ZrO₃, Li₂O. Li enrichment in ⁶Li (natural 7.5%, enriched to 60–90%) is assumed.

**Scoring Logic**:
```
Mandatory check:
  if Li not in elements → return 0.0 (Li is the tritium source via ⁶Li + n reaction)

Lithium ceramic families (ordered by TBR and thermal properties):
  +0.42 if Li + Si + O present (Li₄SiO₄ — best TBR, good thermal stability, ITER TBM)
    additional +0.10 if Li:Si ratio ~4:1 in formula (exact Li₄SiO₄ stoichiometry)
  +0.38 if Li + Ti + O present (Li₂TiO₃ — leading candidate; excellent tritium release)
    additional +0.08 if Li:Ti ratio ~2:1 (Li₂TiO₃ stoichiometry)
  +0.32 if Li + Zr + O present (Li₂ZrO₃ — good thermal stability, studied for DEMO)
  +0.28 if Li + Al + O present (LiAlO₂ — γ phase; studied extensively for ITER)
  +0.25 if Li + O only (Li₂O — highest TBR but poor thermal/chemical stability)
  +0.20 if Li + O + (Ba or Sr) (mixed lithium ceramics — improved sintering)

Tritium release properties proxy:
  +0.10 if open/porous crystal system (hexagonal or trigonal — facilitates T diffusion)
  +0.08 if low NSites (simple structure → clean tritium diffusion pathways)

Neutron multiplication:
  +0.12 if Be present (Be multiplier: n,2n reaction → doubles neutrons available)
  +0.10 if Pb present (Pb multiplier: n,2n reaction; cheaper than Be, lower toxicity concern at this use)
  +0.08 if Zr or Sn present (minor multiplier effect)

Thermal and radiation stability:
  +0.12 if formation energy < −4.0 eV/atom (must survive 14 MeV neutron bombardment)
  +0.08 if decomposition energy < 0.01 eV/atom
  +0.08 if cubic or trigonal crystal system (isotropic swelling under neutron irradiation)

Band gap:
  +0.08 if ≥ 4.0 eV (electrical insulation in magnetic confinement environment)

Penalties:
  ×0.50 if no O present (non-oxide lithium compounds — poor tritium retention)
  ×0.60 if high-σ neutron absorbers: {B, Cd, Gd, Hf} as co-elements without Li
  ×0.40 if metallic Li (Li metal → violent reaction with coolant water; safety hazard)
  ×0.30 if volatile elements (S, Se, Te, halides) — tritium contamination pathways
```

---

### 53. Radiation Shielding

**Scientific Context**: Attenuates ionising radiation (gamma rays, neutrons, alpha/beta particles) to protect personnel and equipment. Distinct from radiation detection — goal is attenuation, not signal generation. Gamma shielding: high-Z elements (Pb, W, Ba) maximise photoelectric absorption cross-section σ ∝ Z⁵. Neutron shielding: hydrogen-rich materials (polyethylene, water) for thermalisation + boron/cadmium/gadolinium for capture. Combined gamma + neutron: borated concrete, Pb-polyethylene composites. This scorer targets dense inorganic crystalline materials.

**Scoring Logic**:
```
Gamma shielding — high-Z elements (photoelectric absorption ∝ Z⁵):
  high_z_shield = {Pb, W, Ba, Bi, Ta, Hf, Re, Os, Ir, Pt, Au, Tl, Hg}
  high_z_count = number present
  +min(high_z_count × 0.18, 0.50)

  Specific bonuses:
    +0.15 if Pb present (cheapest, densest practical gamma shield; ρ=11.3 g/cm³)
    +0.14 if W present (W alloys — compact shielding for medical/industrial; ρ=19.3 g/cm³)
    +0.12 if Ba + O or Ba + S present (BaSO₄, BaO in concrete — non-toxic Pb alternative)
    +0.10 if Bi present (Bi₂O₃ — Pb-free gamma shield for medical)

Neutron shielding:
  +0.20 if B present (¹⁰B: σ_thermal = 3837 barn — extremely high capture cross-section)
    additional +0.08 if B + H (borated polyethylene proxy — combined thermalisation + capture)
    additional +0.06 if B + Ca + O (borated concrete — civil shielding)
  +0.18 if Gd present (Gd: σ_thermal = 49,000 barn — highest of stable elements)
  +0.15 if Cd present (Cd: σ_thermal = 2520 barn — thermal neutron absorber)
  +0.12 if Li present (⁶Li: σ_thermal = 940 barn; tritium production side effect)
  +0.10 if H present (hydrogen — neutron moderation/thermalisation)
  +0.08 if In or Sm or Eu present (high capture cross-section elements)

Density proxy (higher density → better shielding per unit volume):
  +0.12 if Density field > 10 g/cm³ (very dense)
  +0.08 if Density field > 6 g/cm³ (moderately dense)
  +0.05 if Density field > 3 g/cm³ (minimum useful density)

Stability under radiation:
  +0.08 if formation energy < −2.0 eV/atom
  +0.05 if no organic elements (C, N, H in organic compounds degrade under radiation)

Penalties:
  ×0.0 if radioactive elements present → return 0.0 (shield can't be radioactive itself)
  ×0.50 if low-density proxy: only light elements (Li, Be, B, C, N, O, F, Na, Mg, Al, Si)
    unless B present (borated light materials are good neutron shields)
  ×0.70 if no high-Z AND no neutron-capturing element (transparent to all radiation)
```

---

### 54. Nuclear Waste Immobilisation

**Scientific Context**: Vitrification or ceramic immobilisation of high-level radioactive waste (HLW) from nuclear fuel reprocessing. Borosilicate glass is the current standard but has limited waste loading (~25 wt%). Crystalline ceramics offer higher waste loading and better long-term stability: pyrochlore (A₂B₂O₇), zirconolite (CaZrTi₂O₇), hollandite (BaAl₂Ti₆O₁₆), perovskite (CaTiO₃). SYNROC (SYNthetic ROCk) is the benchmark crystalline wasteform. Must remain stable for > 10,000 years against leaching in geological repository conditions.

**Scoring Logic**:
```
Note: This scorer evaluates the HOST MATRIX material, not the waste itself.
Radioactive elements in the formula are acceptable HERE (they represent waste incorporation).

SYNROC mineral families:
  +0.40 if Ti + Zr + Ca + O present (zirconolite CaZrTi₂O₇ — primary actinide host in SYNROC)
  +0.38 if Ti + Ba + Al + O present (hollandite BaAl₂Ti₆O₁₆ — Cs/Rb host in SYNROC)
  +0.35 if Ca + Ti + O present (perovskite CaTiO₃ — Sr/REE host in SYNROC)
  +0.32 if A₂B₂O₇ pyrochlore proxy: {La, Nd, Sm, Gd, Ce, Pu, U} + {Zr, Ti, Sn, Hf} + O
    (pyrochlore — primary candidate for Pu/actinide immobilisation)
  +0.28 if Zr + Si + O (zircon ZrSiO₄ — natural actinide host; geologically proven stability)
  +0.25 if Zr + O only (ZrO₂ — baddeleyite; radiation-stable)
  +0.22 if Al + Si + O (aluminosilicate — glass-ceramic wasteform)
  +0.20 if Cs + Al + Si + O (pollucite CsAlSi₂O₆ — Cs waste host ceramic)
  +0.18 if Mn + Ba + O or Mn + Fe + O (spinel — minor actinide host)

Leach resistance and geological stability:
  +0.20 if formation energy < −4.5 eV/atom (extremely stable against hydrolysis)
  +0.15 if formation energy −3.0 to −4.5 eV/atom (very stable)
  +0.10 if decomposition energy < 0.01 eV/atom
  +0.12 if cubic crystal system (isotropic radiation swelling — stays monolithic)
  +0.08 if high density proxy (Density > 5 g/cm³ if available — slows leachant diffusion)

Radiation damage tolerance:
  +0.10 if fluorite or pyrochlore crystal structure (self-annealing under heavy-ion bombardment)
  +0.08 if simple stoichiometry (NSites ≤ 10) — fewer defect configurations

Band gap:
  +0.05 if ≥ 4.0 eV (insulating ceramics — no radiolytic decomposition)

Penalties:
  ×0.50 if volatile elements (Na, K, S, Se, Te, halogens except F) — leach preferentially
  ×0.40 if low formation energy > −1.0 eV/atom (too metastable for 10,000-year stability)
  ×0.60 if metallic (band gap < 0.5 eV) — metals corrode in geological repository
```

---

## Domain 10: Energy Storage Additions

### 55. Battery Separator

**Scientific Context**: The separator in a Li-ion cell is a porous polymeric or ceramic-coated membrane (20–25 μm thick) that physically prevents short circuits while allowing Li⁺ ion transport. Requirements: high ionic conductivity (tortuous but open pores), electrical insulation (band gap >> 0), thermal stability (shutdown at 130°C to prevent thermal runaway), chemical stability vs electrolyte, and good wettability. Commercial: polyethylene (PE), polypropylene (PP), trilayer PP/PE/PP. Ceramic-coated separators (Al₂O₃, SiO₂, BaTiO₃ on PE) improve thermal stability to > 200°C. This scorer targets the ceramic coating or inorganic separator material.

**Scoring Logic**:
```
Hard requirements:
  if band gap < 3.0 eV → ×0.2 (must be electrically insulating — separators cannot conduct electrons)
  if no O present AND no F present → ×0.5 (almost all ceramic separators are oxides or fluorides)

Ceramic coating families:
  +0.38 if Al + O present (Al₂O₃ — most widely used ceramic separator coating; excellent thermal stability)
    additional +0.08 if Al + O only (pure Al₂O₃ — anatase/γ phase coating)
  +0.32 if Si + O present (SiO₂ — colloidal silica coating; good wettability)
  +0.28 if Ti + O present (TiO₂ — ceramic coating; photocatalytic self-cleaning bonus)
  +0.25 if Ba + Ti + O present (BaTiO₃ — high dielectric constant → reduces dendrite formation)
  +0.22 if Mg + O present (MgO — emerging ceramic separator coating)
  +0.20 if Zr + O present (ZrO₂ — thermal stability > 1500°C)
  +0.18 if Al + Si + O (mullite — composite ceramic separator)
  +0.15 if B + O (BN-based — ultra-high thermal stability)
  +0.12 if Li + Al + O or Li + Si + O (Li-containing ceramic — ionic conductivity bonus)
  +0.10 if F + Li or F + Al (fluoride separator materials)

Band gap (electrical insulation):
  +0.30 if ≥ 6.0 eV (excellent insulation — SiO₂, Al₂O₃ range)
  +0.22 if 4.0–6.0 eV (very good)
  +0.12 if 3.0–4.0 eV (adequate)

Ionic transport proxy:
  +0.10 if Li present (Li-conducting ceramic → ionic conductivity)
  +0.08 if open crystal structure (high NSites but simple repeating unit — tortuous paths)

Chemical stability vs electrolyte:
  +0.10 if formation energy < −3.0 eV/atom (stable vs organic carbonate solvents)
  +0.08 if decomposition energy < 0.01 eV/atom

Thermal stability:
  +0.10 if refractory oxide: {Al₂O₃, ZrO₂, MgO proxy} + formation energy < −4.0 eV/atom
  +0.08 if melting point proxy: formation energy < −3.5 eV/atom (high-melting ceramics)

Penalties:
  ×0.0 if metallic (band gap < 0.5 eV) — would short-circuit the cell instantly
  ×0.50 if S, Se, or Te present (react with electrolyte → cell degradation)
  ×0.60 if alkali metals (K, Na, Rb, Cs) as primary cation without Li (reactive with electrolyte)
  ×0.40 if Pb, Tl, Hg, Cd, As (toxic in consumer battery environment)
```

---

### 56. Liquid Battery Electrolyte Component

**Scientific Context**: Liquid electrolytes in Li-ion batteries are solutions of Li salts (LiPF₆, LiTFSI, LiBF₄) in organic carbonate solvents (EC, DMC, DEC). The electrolyte must: conduct Li⁺ (ionic conductivity ~10 mS/cm), be stable at the anode (SEI formation) and cathode (oxidative stability > 4.5 V vs Li/Li⁺), have low viscosity, wide liquid temperature range, and low flammability. This scorer targets inorganic Li salt candidates and solid additive components (not organic solvents, which are outside the crystalline materials database).

**Scoring Logic**:
```
Mandatory check:
  if Li not in elements → return 0.0 (must contain Li as the working ion carrier)

Hard disqualifiers:
  if radioactive elements → return 0.0
  if band gap < 2.0 eV → return 0.0 (electrolyte must be insulating against electronic conductivity)

Li salt anion families:
  +0.38 if Li + P + F present (LiPF₆ proxy — industry standard; highest conductivity salt)
  +0.35 if Li + N + S + F + O present (LiTFSI proxy — Li[(CF₃SO₂)₂N]; best stability)
  +0.32 if Li + B + F present (LiBF₄ — low-temperature performance)
  +0.28 if Li + B + O present (LiBOB — wide electrochemical window, no F)
  +0.25 if Li + S + O + N + C (LiFSI/LiTFSI-type sulfonimide anion proxy)
  +0.22 if Li + Cl only (LiCl — simple, not practical alone but relevant for hybrid electrolytes)
  +0.20 if Li + I only (LiI — solid-state equivalent used in DSSCs, lower voltage window)

Inorganic additive components:
  +0.18 if Li + N + O (LiNO₃ — critical additive for Li-S batteries; protective SEI)
  +0.15 if B + O + Li (lithium borate — flame retardant additive)
  +0.12 if F + Li + P (LiPF₆ decomposition products — stable passivation additives)
  +0.10 if Mg or Al + O (ceramic additive for viscosity/stability control)

Electrochemical window proxy:
  +0.15 if formation energy < −3.0 eV/atom (thermodynamically stable vs Li metal anode)
  +0.10 if no oxidising elements at high valence (e.g., Mn⁴⁺, Cr⁶⁺ would degrade at cathode)

Band gap:
  +0.25 if ≥ 5.0 eV (excellent electrochemical insulation)
  +0.18 if 3.5–5.0 eV (very good)
  +0.10 if 2.0–3.5 eV (adequate)

Penalties:
  ×0.50 if Tl, Hg, Cd, or As (toxic in consumer battery)
  ×0.60 if transition metals with variable oxidation states (Fe, Co, Mn) as primary element
    (redox activity in electrolyte → decomposition, cell failure)
  ×0.40 if S present without N or F (polysulfide dissolution risk)
```

---

## Domain 11: Solar & Optical Additions

### 57. Solar Thermal Absorber

**Scientific Context**: Converts concentrated sunlight to heat for concentrated solar power (CSP) plants, solar water heaters, and industrial process heat. Unlike PV, no band gap optimisation needed — the goal is maximum broadband absorption across the full solar spectrum (300–2500 nm) combined with low thermal emittance (to retain heat). Key material classes: selective solar absorbers (high α, low ε), cermet coatings (TiN/AlN/TiAlN in Al₂O₃ matrix), black oxides (Fe₃O₄, Cu₂O), and carbon-based absorbers. Figure of merit: absorptance/emittance ratio at operating temperature (200–600°C for flat plate; 600–1000°C for CSP).

**Scoring Logic**:
```
Broadband absorber elements (materials absorb across solar spectrum):
  Metallic absorbers (high optical absorption coefficients):
  +0.35 if band gap < 0.5 eV (metallic — absorbs via free electron response; Drude model)
  +0.25 if band gap 0.5–1.5 eV (near-metallic — strong interband absorption in solar range)
  +0.15 if band gap 1.5–2.5 eV (partial absorption — absorbs visible but transparent in IR)
  ×0.3 if band gap > 3.0 eV (transparent to most solar spectrum — poor absorber)

Selective absorber cermet families:
  +0.30 if Ti + N present (TiN — gold-coloured, high solar absorptance, plasmonic resonance)
    additional +0.08 if Ti + Al + N (TiAlN — better thermal stability than TiN at 600°C)
  +0.28 if Al + N present (AlN — cermet matrix material; Al₂O₃ forms in-situ at T)
  +0.25 if Cr + O present (Cr₂O₃ / black chrome — classic flat-plate solar absorber)
  +0.22 if Cu + O present (Cu₂O, CuO — broadband absorber, cheap)
  +0.20 if Fe + O present (Fe₃O₄, Fe₂O₃ — black iron oxide absorbers)
  +0.18 if Co + O present (Co₃O₄ — spinel absorber, good selectivity)
  +0.15 if Ni + O present (NiO — nickel oxide absorber)
  +0.14 if Mo + S present (MoS₂ — layered absorber, emerging CSP research)
  +0.12 if C only or C + metal (carbon composites — near-perfect black body)

High-temperature stability (CSP requirement):
  +0.15 if formation energy < −3.0 eV/atom (stable at 600–1000°C operating temperature)
  +0.10 if refractory metals present (W, Mo, Ta, Re — high emittance retention at T)
  +0.08 if Cr or Al present (protective oxide scale formation in air)

Thermal emittance control:
  +0.08 if metallic + nitride together (cermet structure → tunable optical properties)
  +0.06 if layered structure (interference coating effect → wavelength-selective absorption)

Penalties:
  ×0.60 if band gap > 3.0 eV AND no plasmonic metal present (transparent — not an absorber)
  ×0.50 if volatile elements: Zn, Cd, Hg, S, Se, Te (evaporate at CSP temperatures)
  ×0.40 if Tl, Pb as primary element (toxic vapour at operating temperature)
```

---

### 58. Anti-Reflection Coating

**Scientific Context**: Minimises reflection losses at optical interfaces via destructive interference. For single-layer AR coating, the optimal refractive index is $n_{AR} = \sqrt{n_1 \cdot n_2}$ (geometric mean of adjacent media) and optimal thickness is λ/4 at the target wavelength. For solar cells on Si (n~3.5), ideal coating has n~1.87 (MgF₂: n=1.38 for glass; SiNₓ: n=1.9–2.3 widely used). Multi-layer designs (SiO₂/TiO₂, SiNₓ/SiO₂) achieve broadband AR. Requirements: wide band gap (transparent), low refractive index, good adhesion, durability, and low-cost deposition.

**Scoring Logic**:
```
Hard requirement (must be optically transparent):
  if band gap < 2.5 eV → return 0.0 (absorbs visible light — unsuitable AR coating)

Band gap:
  +0.35 if ≥ 5.0 eV (excellent transparency across full solar spectrum)
  +0.25 if 3.5–5.0 eV (very good)
  +0.15 if 2.5–3.5 eV (adequate for visible AR; absorbs some UV)

Low refractive index material families (n = 1.3–1.7 ideal for glass/air interface):
  +0.35 if Mg + F present (MgF₂ — n=1.38; lowest-n practical AR coating; space optics)
  +0.30 if Ca + F or Ba + F present (CaF₂ n=1.43, BaF₂ n=1.47 — UV/visible AR)
  +0.28 if Si + O present (SiO₂ — n=1.46; most common AR coating; sol-gel compatible)
  +0.25 if Al + F present (AlF₃ — n=1.35; deep UV AR)
  +0.22 if La + F present (LaF₃ — n=1.60; used in multi-layer AR stacks)
  +0.20 if Na + Al + F present (cryolite Na₃AlF₆ — n=1.35; historically important)

Medium/high refractive index (n = 1.8–2.5 for multi-layer designs):
  +0.25 if Si + N present (Si₃N₄/SiNₓ — n=1.9–2.3 tunable; standard silicon solar AR)
  +0.22 if Ti + O present (TiO₂ — n=2.3–2.5; top layer in multi-layer AR stacks)
  +0.20 if Zn + O present (ZnO — n=2.0; transparent conductor + AR function)
  +0.18 if Al + O present (Al₂O₃ — n=1.63; ALD conformal AR + passivation)
  +0.15 if Ta + O present (Ta₂O₅ — n=2.1; high-n layer in optical coatings)
  +0.12 if Hf + O present (HfO₂ — n=2.0; used in high-power laser coatings)

Stability:
  +0.10 if formation energy < −3.0 eV/atom (must survive outdoor weathering/UV exposure)
  +0.08 if decomposition energy < 0.01 eV/atom

Penalties:
  ×0.50 if band gap < 3.0 eV (absorbs visible — not a useful AR coating)
  ×0.60 if metallic elements dominant (metals are reflective, not AR)
  ×0.50 if S, Se, or Te present (chalcogenides not used as AR coatings — reactive)
  ×0.40 if Tl, Hg, Cd, Pb dominant (toxic in optical coating applications)
```

---

## Domain 12: Catalysis Additions

### 59. Selective Hydrogenation Catalyst

**Scientific Context**: Selective reduction of unsaturated functional groups (C=C, C=O, C≡N, NO₂) with H₂ in pharmaceutical, agrochemical, and fragrance synthesis. The challenge is chemoselectivity — reducing one functional group without touching others in complex molecules. Pd is the benchmark for C=C reduction (near-zero activation energy); Ni, Co, Cu for less selective but cheaper applications. Pt and Rh for ketone/aldehyde hydrogenation. Ru for stereoselective hydrogenation (pharmaceutical). Bimetallic catalysts (PdAg, PdAu, PtSn) tune selectivity.

**Scoring Logic**:
```
Noble metal hydrogenation catalysts:
  +0.38 if Pd present (C=C, C≡C benchmark; highest selectivity for partial hydrogenation)
    additional +0.10 if Pd + Ag (PdAg — selective hydrogenation of alkynes to alkenes)
    additional +0.08 if Pd + Au (PdAu — suppresses over-reduction)
  +0.32 if Pt present (ketone/aldehyde hydrogenation; stereoselective)
    additional +0.08 if Pt + Sn (PtSn — selective aldehyde hydrogenation vs ketone)
  +0.28 if Rh present (stereoselective; pharmaceutical synthesis)
  +0.25 if Ru present (stereoselective hydrogenation; chiral catalysts)
  +0.22 if Ir present (asymmetric hydrogenation; expensive)

Earth-abundant alternatives:
  +0.28 if Ni present (Raney Ni — workhorse for bulk hydrogenation; low selectivity)
    additional +0.08 if Ni + P (Ni₂P — selective vs Raney Ni)
    additional +0.06 if Ni + B (Ni-B — selective alkyne hydrogenation)
  +0.22 if Co present (Co-based hydrogenation; Fischer-Tropsch connection)
  +0.18 if Cu present (Cu — chemoselective for C=O over C=C; Ullmann-type)
  +0.15 if Fe present (Fe₃O₄ — HCHO hydrogenation; methanol synthesis direction)
  +0.12 if Mo present (MoS₂ — hydrodesulfurisation + selective hydrogenation)

Electronic character:
  +0.20 if band gap < 0.5 eV (metallic — electron transfer for H₂ dissociation)
  +0.10 if band gap 0.5–1.5 eV
  ×0.40 if band gap > 2.5 eV (insulating; H₂ dissociation blocked)

Support interaction proxy:
  +0.08 if Al + O (Al₂O₃ support — standard catalyst support)
  +0.08 if Si + O (SiO₂ support)
  +0.07 if Zr + O (ZrO₂ — bifunctional support for acid-catalysed hydrogenation)

Stability:
  +0.08 if formation energy < −1.5 eV/atom

Penalties:
  ×0.60 if no hydrogenation-active metal present
  ×0.50 if Tl, Hg, Cd, or As (catalyst poison and toxic)
  ×0.40 if high-spin oxide dominant without metal (too oxidised; needs reduction first)
```

---

### 60. Fischer-Tropsch Catalyst

**Scientific Context**: Converts syngas (CO + H₂, from coal, natural gas, or biomass gasification) to liquid hydrocarbons (e-fuels, synthetic diesel, waxes) via surface polymerisation: nCO + (2n+1)H₂ → CnH(2n+2) + nH₂O. Fe catalysts (promoted with K and Cu) are used for high-temperature FT (330–350°C, producing gasoline/olefins). Co catalysts (on Al₂O₃/SiO₂/TiO₂) for low-temperature FT (200–240°C, producing long-chain waxes/diesel). Ru is the most active but too expensive. The Anderson-Schulz-Flory distribution governs chain length selectivity.

**Scoring Logic**:
```
FT-active metals:
  +0.40 if Co present (LTFT workhorse; high chain growth probability α; diesel/wax)
    additional +0.08 if Co + Re (Re promoter — prevents cobalt oxidation)
    additional +0.08 if Co + Ru (mixed catalyst — enhanced activity)
    additional +0.07 if Co + Ti + O (Co/TiO₂ — strong metal-support interaction enhances selectivity)
  +0.35 if Fe present (HTFT; also methanol synthesis synergy; cheaper than Co)
    additional +0.10 if Fe + K (K promoter — shifts selectivity to olefins/alcohols)
    additional +0.08 if Fe + Cu (Cu promoter — facilitates carburisation to active FexC phase)
    additional +0.06 if Fe + Mn (Mn promoter — light olefin selectivity)
  +0.25 if Ru present (highest FT activity; benchmark but expensive)
  +0.18 if Ni present (methanation catalyst — high selectivity to CH₄, useful for SNG)
  +0.15 if Mo + S present (MoS₂ — alcohol synthesis from syngas; oxygenate selectivity)

Carbide phase detection (active phase for Fe FT):
  +0.12 if Fe + C present (iron carbide Hägg phase χ-Fe₅C₂ proxy — actual active FT phase)
  +0.10 if Co + C present (surface cobalt carbide forms under FT conditions)

Support effects:
  +0.10 if Al + O (Al₂O₃ — standard FT catalyst support)
  +0.08 if Si + O (SiO₂ — inert support, used for Co FT)
  +0.08 if Ti + O (TiO₂ — SMSI with Co → improved selectivity)
  +0.06 if Zr + O (ZrO₂ — acid-base bifunctional support)

Electronic:
  +0.18 if band gap < 0.5 eV (metallic — CO/H₂ activation requires electron transfer)
  +0.08 if 0.5–1.5 eV
  ×0.40 if > 2.5 eV

Stability:
  +0.08 if formation energy < −1.0 eV/atom (must survive reducing syngas atmosphere)

Penalties:
  ×0.60 if no FT-active metal (Co, Fe, Ru, Ni) present
  ×0.50 if S dominant (sulfur poisons Co/Fe FT catalysts; only MoS₂ exception)
  ×0.40 if Tl, Hg, Cd, or As (catalyst poisons)
```

---

### 61. Dehydrogenation / LOHC Catalyst

**Scientific Context**: Liquid organic hydrogen carriers (LOHCs) store H₂ chemically in aromatic compounds (toluene/methylcyclohexane, dibenzyltoluene/perhydrodibenzyltoluene) at ambient conditions, releasing H₂ on demand via catalytic dehydrogenation (e.g., methylcyclohexane → toluene + 3H₂, ΔH = +205 kJ/mol). Pt/Al₂O₃ is the benchmark dehydrogenation catalyst. Ni-based alternatives are cheaper. Key requirements: high activity at 300–400°C, selectivity against cracking, long lifetime, and thermal stability. Also relevant: dehydrogenation of propane to propylene (PDH) using Pt/Sn or VOₓ/Al₂O₃ catalysts.

**Scoring Logic**:
```
Dehydrogenation-active metals:
  +0.38 if Pt present (Pt — benchmark for MCH dehydrogenation; Pt-Sn for PDH)
    additional +0.10 if Pt + Sn (Pt₃Sn, PtSn — commercial PDH catalyst suppresses coking)
    additional +0.08 if Pt + In (PtIn — emerging Pt-group PDH catalyst)
  +0.30 if Pd present (high activity but sintering at dehydrogenation temperatures)
  +0.25 if Ni present (cheaper; NiAl₂O₄ — good for LOHC applications)
    additional +0.08 if Ni + Al + O (NiAl₂O₄ precursor — activated by reduction)
  +0.22 if V + O present (VOₓ — vanadium oxide for oxidative dehydrogenation ODH)
    additional +0.08 if V + Al + O (V₂O₅/Al₂O₃ — ODH catalyst)
  +0.20 if Cr + O present (Cr₂O₃ — classic PDH catalyst; CrOₓ/Al₂O₃)
  +0.18 if Fe + O present (FeOₓ — CO₂-assisted dehydrogenation)
  +0.15 if Mo + O present (MoOₓ — ODH active)
  +0.12 if Co + O present (CoOₓ — emerging for ethane dehydrogenation)
  +0.10 if Zn present (Zn/ZSM-5 type — propane aromatisation)

Support interactions:
  +0.12 if Al + O (Al₂O₃ — standard support for Pt/Pd dehydrogenation)
  +0.10 if Si + O + Al (zeolite — shape-selective dehydrogenation/aromatisation)
  +0.08 if Zr + O (ZrO₂ — thermal stability; ZrO₂-based PDH without Pt)

Electronic:
  +0.18 if band gap < 0.5 eV (metallic)
  +0.10 if 0.5–2.0 eV
  ×0.40 if > 3.0 eV (too insulating for C-H activation)

Stability:
  +0.10 if formation energy < −2.0 eV/atom (must survive repeated dehydrogenation cycles)

Penalties:
  ×0.60 if no dehydrogenation-active metal or oxide
  ×0.50 if S present (poisons Pt/Pd catalysts — sulfur tolerance is major challenge)
  ×0.40 if Tl, Hg, Cd, or As
```

---

## Domain 13: Electronics Additions

### 62. Memristor / Neuromorphic Computing Material

**Scientific Context**: Memristors (memory resistors) are two-terminal devices whose resistance depends on the history of current/voltage — effectively non-volatile memory. They enable neuromorphic (brain-inspired) computing by mimicking synaptic weights in neural networks. Resistive switching mechanisms: (1) filamentary switching via conductive bridge (cation migration in Ag/Cu-based devices: ECM cells); (2) interface-type switching via redox at electrode/oxide interface (HfO₂, TaOₓ, NbOₓ); (3) phase-change via Joule heating (GST-based PCM). Key metrics: ON/OFF ratio (> 10³), switching speed (< 1 ns for filamentary), endurance (> 10⁸ cycles), retention (> 10 years at 85°C), multi-level states (for analog computing).

**Scoring Logic**:
```
Filamentary (conductive bridge) memristors — highest ON/OFF ratio:
  +0.38 if Ag present (Ag⁺ migration in solid electrolyte — fastest ECM cell)
    additional +0.10 if Ag + S or Ag + Se (Ag₂S, Ag₂Se — classic ECM solid electrolytes)
    additional +0.08 if Ag + Te (Ag₂Te)
    additional +0.08 if Ag + Ge + S or Ag + Ge + Se (GeS, GeSe-based ECM)
  +0.30 if Cu present (Cu⁺ migration — cheaper than Ag; Cu/HfO₂ ECM cells)
    additional +0.08 if Cu + S (CuS/Cu₂S ECM)
    additional +0.07 if Cu + SiO₂ proxy: Cu + Si + O

Interface-type oxide memristors — best endurance for neuromorphic:
  +0.32 if Hf + O present (HfO₂ — leading RRAM material; compatible with CMOS)
  +0.28 if Ta + O present (TaOₓ — excellent endurance > 10¹² cycles; analog states)
  +0.25 if Nb + O present (NbOₓ — Mott insulator transition; oscillatory dynamics)
  +0.22 if Ti + O present (TiO₂ — original Strukov HP memristor; good analog gradation)
  +0.20 if W + O present (WOₓ — volatile threshold switching)
  +0.18 if Zr + O present (ZrO₂ — RRAM with good retention)
  +0.15 if Ni + O present (NiO — p-type RRAM; bipolar switching)
  +0.14 if Al + O present (Al₂O₃ — insulating layer in ECM cells)
  +0.12 if Mn + O present (MnO₂ — multilevel RRAM)

Phase-change type (see PCM scorer — overlap region):
  +0.18 if Ge + Sb + Te (GST — phase change neuromorphic; analog multilevel)

Band gap for switching layer:
  +0.25 if 2.0–5.0 eV (intermediate gap oxide — optimal for resistive switching)
  +0.15 if 1.0–2.0 eV (narrow gap; interface switching type)
  +0.10 if 5.0–8.0 eV (wide gap; tunnelling memristors)
  ×0.3 if < 0.5 eV (too conductive — no OFF state possible)

Stability and reliability proxy:
  +0.10 if formation energy < −3.0 eV/atom (stable oxide matrix)
  +0.08 if cubic or amorphous-compatible structure (uniform switching sites)

Penalties:
  ×0.60 if no switching-active element (all stable noble metals with no migration pathway)
  ×0.50 if Tl, Hg, Cd, or As (CMOS incompatible toxicity)
  ×0.40 if alkali metals as primary switching ion (Li, Na — unstable in CMOS environment)
```

---

### 63. 2D Material

**Scientific Context**: Materials with strong in-plane covalent bonding and weak van der Waals interlayer coupling, enabling exfoliation to monolayer or few-layer thickness. Properties change dramatically vs bulk: MoS₂ transitions from indirect (bulk, 1.3 eV) to direct (monolayer, 1.9 eV) band gap. Key families: graphene/BN (C, BN), TMDs (MX₂: MoS₂, WS₂, MoSe₂, WSe₂), MXenes (Ti₃C₂, Nb₂C), Xenes (silicene, germanene, phosphorene), and layered oxides (MoO₃, V₂O₅). Applications: field-effect transistors (ultimately thin channel), photodetectors, valleytronics, gas sensors, membranes.

**Scoring Logic**:
```
Dimensionality detection (primary signal):
  +0.35 if Dimensionality field = '2D' or 'layered' or 'intercalated ion'
  +0.20 if layered crystal system proxy: trigonal/hexagonal + transition metal + chalcogenide
    (MX₂ structure proxy)

TMD (transition metal dichalcogenide) families:
  +0.35 if Mo + S present (MoS₂ — paradigmatic semiconductor TMD)
  +0.32 if W + S present (WS₂ — direct gap ~2.0 eV monolayer; strong PL)
  +0.30 if Mo + Se present (MoSe₂ — 1.66 eV monolayer direct gap)
  +0.28 if W + Se present (WSe₂ — ambipolar FET; valleytronics)
  +0.25 if Mo + Te or W + Te (MoTe₂, WTe₂ — phase change; Weyl semimetal in Td)
  +0.22 if Nb + Se or Nb + S (NbSe₂, NbS₂ — charge density wave + superconductor)
  +0.20 if V + Se or V + S (VSe₂, VS₂ — magnetic 2D materials)
  +0.18 if Re + S or Re + Se (ReS₂, ReSe₂ — anisotropic 2D; broken symmetry)
  +0.15 if Sn + S or Sn + Se (SnS₂, SnSe₂ — n-type 2D semiconductor)
  +0.12 if In + Se or Ga + Se (InSe, GaSe — high mobility 2D semiconductors)

Graphene/hBN family:
  +0.40 if C only AND layered (graphene — ultimate 2D material)
  +0.35 if B + N AND hexagonal crystal system (hBN — 2D insulator, dielectric for vdW heterostructures)

MXene family (Ti₃C₂ₓ type):
  +0.32 if Ti + C present (Ti₃C₂ — most studied MXene; metallic, high capacitance)
  +0.28 if Nb + C or V + C or Mo + C (other MXenes)
  +0.25 if Ti + N or Nb + N (MXene nitrides)

Other 2D materials:
  +0.25 if P only (black phosphorus/phosphorene — direct gap 0.3–1.5 eV tunable)
  +0.22 if Bi + Se or Bi + Te (topological 2D — Bi₂Se₃, Bi₂Te₃ few-layer)
  +0.20 if Cr + I₃ proxy: Cr + I (CrI₃ — 2D ferromagnet; magnetic vdW)
  +0.18 if Fe + (Ge or Si) + Te (FGT Fe₃GeTe₂ — 2D ferromagnet, room T)

Band gap (2D monolayer values often ~0.5–1.0 eV higher than bulk DFT):
  +0.15 if 0.0–0.5 eV (semimetallic 2D — graphene, MXenes)
  +0.18 if 0.5–2.5 eV (semiconductor 2D — TMD window)
  +0.12 if 2.5–6.0 eV (insulating 2D — hBN range)

Stability:
  +0.08 if formation energy < −1.0 eV/atom
  +0.05 if decomposition energy < 0.05 eV/atom

Penalties:
  ×0.60 if 3D crystal system with no layered indicator (not a 2D material)
  ×0.50 if complex oxide without layering (perovskites, spinels — 3D structures)
```

---

### 64. Optical Fibre / Waveguide Material

**Scientific Context**: Ultra-pure SiO₂ glass (< 0.1 ppb transition metal impurities) is the foundation of global telecommunications — 5 billion km of fibre deployed. Attenuation minimum is 0.15 dB/km at 1550 nm (Rayleigh scattering limit). Specialty fibres: fluoride glass (ZBLAN: ZrF₄-BaF₂-LaF₃-AlF₃-NaF) for mid-IR transmission (2–8 μm); chalcogenide glass (As₂S₃, Ge-As-Se) for far-IR; phosphate glass for rare-earth-doped amplifiers (Nd, Er). Requirements: ultra-wide transparency window, low OH absorption, refractive index control via dopants.

**Scoring Logic**:
```
Hard requirement (must be transparent at telecom wavelengths):
  if band gap < 3.0 eV → return 0.0 (absorbs telecom/visible light)

Core material families:
  +0.42 if Si + O present (SiO₂ — telecom fibre basis; Rayleigh limit at 1550 nm)
    additional +0.10 if Si + O only (pure silica — highest transparency)
    additional +0.08 if Si + O + Ge (GeO₂-doped SiO₂ — core/cladding refractive index control)
    additional +0.06 if Si + O + F (F-doped SiO₂ — depressed cladding)
    additional +0.06 if Si + O + P (P-doped SiO₂ — phosphosilicate amplifier fibre)
    additional +0.06 if Si + O + B (B-doped SiO₂ — borosilicate cladding)
    additional +0.08 if Er + Si + O (Er-doped SiO₂ — EDFA amplifier; 1550 nm emission)

Fluoride fibres (mid-IR transparency):
  +0.35 if Zr + F or Ba + F present (ZBLAN — ZrF₄/BaF₂; 0.5–6 μm window)
    additional +0.08 if Zr + Ba + La + Al + Na + F (full ZBLAN composition)
  +0.28 if In + F present (InF₃ glass — 2–9 μm window)
  +0.25 if Al + F (AlF₃ glass)

Chalcogenide fibres (far-IR):
  +0.30 if As + S present (As₂S₃ — 1–6 μm; most common chalcogenide fibre)
  +0.28 if As + Se present (As₂Se₃ — 2–10 μm; broader IR window)
  +0.25 if Ge + As + Se (Ge-As-Se family — broadest IR window up to 12 μm)
  +0.22 if Ge + Sb + Se or Ge + Sb + Te (mid-IR chalcogenide)

Phosphate fibres (RE-doped amplifiers):
  +0.25 if P + O + (Na or K) (phosphate glass — high RE solubility for compact amplifiers)
    additional +0.10 if Er + P + O (Er:phosphate — short-length EDFA)
    additional +0.08 if Nd + P + O (Nd:phosphate — 1060 nm laser fibre)

Rare-earth dopants (amplifier/laser functionality):
  +0.15 if Er present (1550 nm telecom amplifier — EDFA)
  +0.12 if Nd present (1060/1340 nm laser fibre)
  +0.10 if Yb present (1000–1100 nm; high-power fibre laser)
  +0.08 if Tm present (2000 nm — medical, LIDAR)
  +0.07 if Ho present (2090 nm — surgical laser)

Band gap:
  +0.28 if ≥ 8.0 eV (SiO₂-like — ultra-wide transparency)
  +0.22 if 5.0–8.0 eV
  +0.12 if 3.0–5.0 eV (chalcogenide/fluoride range)

Stability:
  +0.10 if formation energy < −3.0 eV/atom (chemical durability)

Penalties:
  ×0.60 if transition metals (Fe, Co, Ni, Mn, Cu, Cr) present without RE context
    (catastrophic absorption at telecom wavelengths even at ppb levels)
  ×0.40 if band gap < 3.5 eV AND no chalcogenide/mid-IR justification
  ×0.50 if Tl, Hg, Cd, or As in non-chalcogenide context (toxic)
```

---

## Domain 14: Structural & Mechanical

### 65. Lightweight Structural Material

**Scientific Context**: High specific strength (σ/ρ) and specific stiffness (E/ρ) for aerospace, automotive, and sports applications. Key families: Al alloys (2xxx, 6xxx, 7xxx series — ρ~2.7 g/cm³), Ti alloys (Ti-6Al-4V — ρ~4.4 g/cm³, higher strength), Mg alloys (ρ~1.7 g/cm³, lightest structural metal), and advanced ceramics (SiC, Al₂O₃ — high stiffness/density). Metal matrix composites (Al-SiC, Ti-TiB₂) push the boundary. This scorer targets the bulk inorganic phase compositions rather than alloy microstructure details.

**Scoring Logic**:
```
Density proxy (low density → high specific properties):
  density_score = 0 (default)
  if Density field available:
    +0.35 if < 2.0 g/cm³ (ultra-light — Mg alloys, Li alloys)
    +0.28 if 2.0–3.5 g/cm³ (light — Al alloys, CFRP matrix)
    +0.18 if 3.5–5.0 g/cm³ (moderate — Ti alloys, lower end)
    +0.08 if 5.0–8.0 g/cm³ (heavier structural metals)
    +0.00 if > 8.0 g/cm³ (heavy — not lightweight structural)

Lightweight structural element families:
  +0.35 if Mg present (lightest structural metal; ρ~1.7 g/cm³; AZ/ZK/WE alloy systems)
    additional +0.08 if Mg + Al (AZ31/AZ91 — most common Mg alloys)
    additional +0.08 if Mg + Zn + Zr (ZK60 — highest-strength Mg wrought alloy)
    additional +0.08 if Mg + RE (Mg-Nd, Mg-Gd — creep-resistant for automotive)
  +0.32 if Al present (most widely used lightweight metal; extensive alloy library)
    additional +0.08 if Al + Cu (2xxx series — aerospace structural)
    additional +0.08 if Al + Mg + Si (6xxx — most common extruded; automotive)
    additional +0.08 if Al + Zn + Mg (7xxx series — highest-strength Al alloys)
    additional +0.06 if Al + Li (Al-Li — 10% density reduction; aerospace)
  +0.28 if Ti present (high strength-to-weight; ρ~4.5 g/cm³; aerospace, biomedical)
    additional +0.10 if Ti + Al + V (Ti-6Al-4V — most used Ti alloy in aerospace)
    additional +0.07 if Ti + Al + Mo or Ti + Al + Nb (high-T Ti alloys)

Lightweight ceramics (high stiffness/density):
  +0.22 if Si + C present (SiC — E/ρ among best structural ceramics)
  +0.20 if Al + O present (Al₂O₃ — high stiffness, low density vs metals)
  +0.18 if Si + N present (Si₃N₄ — excellent high-T lightweight structural ceramic)
  +0.15 if B + C present (B₄C — near-diamond hardness, very light; body armour)

Electronic character:
  +0.12 if band gap < 0.5 eV (metallic — necessary for ductile lightweight metals)
  +0.08 if > 4.0 eV (ceramic lightweight materials)

Stability:
  +0.10 if formation energy < −1.5 eV/atom

Penalties:
  ×0.50 if density > 8 g/cm³ (not lightweight by definition)
  ×0.60 if no lightweight structural element (Mg, Al, Ti, Si, B, Li, Be)
  ×0.40 if Tl, Hg, Pb as primary element (heavy and toxic)
  ×0.50 if Be present as primary element (BeO, Be metal — toxic despite being lightweight)
```

---

### 66. Shape Memory Alloy (SMA)

**Scientific Context**: SMAs recover their original shape upon heating (shape memory effect) or exhibit large recoverable strains under stress (superelasticity) via a martensitic phase transformation. NiTi (Nitinol, 50 at% Ni-Ti) is the workhorse — used in stents, orthodontic wires, actuators, and pipeline couplings. Cu-Al-Ni and Cu-Zn-Al are cheaper alternatives. Fe-Mn-Si-based alloys are emerging. The transformation temperature (As, Af, Ms, Mf) is highly composition-dependent and can be tuned from −100°C to +150°C.

**Scoring Logic**:
```
SMA material families:
  +0.42 if Ni + Ti present (NiTi / Nitinol — absolute benchmark; biocompatible, large recovery strain ~8%)
    additional +0.10 if Ni:Ti ratio approximately 1:1 in formula (equiatomic NiTi — maximum effect)
    additional +0.08 if Ni + Ti + Cu (NiTiCu — reduced hysteresis; fatigue-resistant)
    additional +0.08 if Ni + Ti + Nb (NiTiNb — wide hysteresis; pipeline couplings)
    additional +0.06 if Ni + Ti + Hf (NiTiHf — high-temperature SMA > 100°C)
    additional +0.06 if Ni + Ti + Zr (NiTiZr — high-temperature SMA)

  +0.30 if Cu + Al + Ni present (Cu-Al-Ni — cheaper than NiTi; brittle; large hysteresis)
  +0.28 if Cu + Zn + Al present (Cu-Zn-Al — cheapest SMA; poor fatigue)
  +0.25 if Cu + Al + Mn present (Cu-Al-Mn — ductile Cu-based SMA; emerging)
  +0.22 if Fe + Mn + Si present (Fe-Mn-Si — cheap ferrous SMA; lower recovery)
  +0.20 if Fe + Ni + Co + Ti present (Fe-Ni-Co-Ti — high-temperature ferromagnetic SMA)
  +0.18 if Ni + Mn + Ga present (Ni-Mn-Ga — magnetic SMA; field-driven actuation)
  +0.15 if Ni + Mn + In or Ni + Mn + Sn (Ni-Mn-based magnetic SMA — Heusler)

Electronic character:
  +0.15 if band gap < 0.1 eV (metallic — essential for SMA ductility and actuation)
  ×0.60 if band gap > 1.0 eV (ceramic/semiconductor — cannot be an SMA)

B2 / L2₁ structure indicators (most SMAs are ordered intermetallics):
  +0.10 if cubic crystal system (B2 austenite phase — NiTi parent phase)
  +0.08 if NSites ≤ 4 (simple ordered structure — NiTi, CuZn type)

Stability:
  +0.08 if formation energy −0.3 to −1.5 eV/atom (moderate — must transform reversibly; too stable → no transformation)

Penalties:
  ×0.0 if no SMA-active combination (all wrong element combinations)
  ×0.60 if band gap > 0.5 eV (not metallic — can't be an SMA)
  ×0.50 if Tl, Hg, Cd, or As (toxic — especially problematic for biomedical NiTi)
```

---

### 67. Metallic Glass / Amorphous Metal

**Scientific Context**: Amorphous metals produced by rapid quenching (> 10⁶ K/s for simple compositions) or by exploiting multi-component alloys that frustrate crystallisation. Unique properties: no grain boundaries (smooth, corrosion-resistant surfaces), ultra-high yield strength approaching theoretical maximum (~E/50), near-zero hysteresis loss for soft magnetic applications, and large elastic strain (>2%). Key families: Fe-Si-B (METGLAS — soft magnetic transformer cores), Zr-based (Zr₄₁Ti₁₄Cu₁₂.₅Ni₁₀Be₂₂.₅ / Vitreloy — structural), Fe-based (Fe-Cr-Mo-C-B — corrosion resistant), Pd-based.

**Scoring Logic**:
```
Glass-forming ability (GFA) indicators — multi-component systems frustrate crystallisation:

Zr-based structural metallic glasses (highest toughness):
  +0.38 if Zr present + ≥ 3 other elements (multi-component Zr glass)
    additional +0.10 if Zr + Cu + Ni + Al (Zr-based quaternary — Vitreloy type)
    additional +0.08 if Zr + Ti + Cu + Ni + Be (Vitreloy 1 — best BMG; Be reduces fragility)
    additional +0.08 if Zr + Cu + Al (ternary Zr BMG — without toxic Be)

Fe-based metallic glasses (soft magnetic + structural):
  +0.35 if Fe + Si + B present (METGLAS — commercial soft magnetic amorphous alloy)
    additional +0.10 if Fe + Si + B only or Fe + Si + B + C (2605 series)
  +0.30 if Fe + B + P present (Fe-B-P — lower critical cooling rate; easier amorphisation)
  +0.28 if Fe + Cr + Mo + C + B (Fe-Cr-Mo-C-B — corrosion-resistant metallic glass)

Cu-based metallic glasses:
  +0.28 if Cu + Zr + Al or Cu + Zr + Ti (Cu-Zr binary is prototype glass system)
  +0.25 if Cu + Hf + Ti (Cu-Hf-Ti BMG)

Pd/Pt-based (highest toughness BMGs):
  +0.32 if Pd + Cu + Ni + P (Pd-based — highest fracture toughness BMG; expensive)
  +0.28 if Pd + Si + Cu

Al/Mg-based (lightweight amorphous):
  +0.25 if Al + La + Ni or Al + Ce + Ni (Al-based amorphous — high strength, light)
  +0.22 if Mg + Cu + Y or Mg + Zn + Ca (Mg-based BMG — lightest metallic glass)

Component count bonus (critical mixing entropy):
  +0.10 if ≥ 4 different elements (multi-principal component → high GFA)
  +0.08 if ≥ 5 different elements

Electronic character:
  +0.20 if band gap < 0.1 eV (metallic — essential for metallic glasses)
  ×0.70 if band gap > 0.5 eV (ceramic/semiconductor — not a metallic glass)

Stability proxy:
  +0.10 if formation energy < −0.3 eV/atom (moderate stability — too stable → crystallises)
  ×0.50 if decomposition energy < −0.5 eV/atom (would phase separate → crystallise)

Penalties:
  ×0.50 if only 1–2 elements (binary alloys have poor GFA; high cooling rates needed)
  ×0.40 if Tl, Hg, Cd, or As (toxic in structural and consumer applications)
  ×0.60 if band gap > 1.0 eV
```

---

### 68. Superalloy

**Scientific Context**: Nickel-based superalloys maintain mechanical strength, creep resistance, and oxidation resistance at temperatures up to 1100°C (90% of Ni melting point). Used in turbine blades, discs, and combustion liners of jet engines and gas turbines. The key microstructural feature is coherent γ' (Ni₃Al, L1₂ ordered) precipitates in a γ (FCC Ni) matrix — the γ' provides strength via order hardening. Rhenium (Re) and ruthenium (Ru) additions provide creep resistance by reducing γ' rafting. Alloy compositions are closely guarded trade secrets (CMSX-4, IN718, René N6).

**Scoring Logic**:
```
Mandatory: Ni as primary matrix element
  if Ni not in elements → ×0.3 (Co-based superalloys exist but are secondary; apply partial credit)

Nickel superalloy system:
  +0.40 if Ni present as primary element
    additional +0.12 if Ni + Al (γ' Ni₃Al formation — essential strengthening phase)
    additional +0.10 if Ni + Al + Cr (classic ternary superalloy basis)
    additional +0.08 if Ni + Al + Co (Co partitions to γ, raises γ' solvus)
    additional +0.08 if Ni + Al + Ti (Ti substitutes for Al in γ', increases γ' volume fraction)
    additional +0.08 if Ni + Al + Ta (Ta in γ' — highest creep resistance single crystals)
    additional +0.07 if Ni + Re (Re — reduces γ' coarsening; 3rd/4th generation superalloys)
    additional +0.07 if Ni + Ru (Ru — 5th generation; suppresses TCP phase formation)
    additional +0.06 if Ni + Mo + W (Mo/W — solid solution strengthening of γ)
    additional +0.06 if Ni + Hf (Hf — improves oxide scale adhesion)

Cobalt superalloys (secondary):
  +0.30 if Co + Al + W present (Co₃(Al,W) — γ' in Co-based, discovered 2006)
  +0.25 if Co + Cr + W (classic Co-based superalloy; no γ')

Oxidation resistance:
  +0.10 if Cr present (Cr₂O₃ scale — oxidation protection up to ~950°C)
  +0.08 if Al + Cr (Al₂O₃ + Cr₂O₃ — dual oxide protection)

Electronic character:
  +0.15 if band gap < 0.1 eV (metallic — essential)
  ×0.60 if band gap > 0.5 eV

Stability:
  +0.10 if formation energy < −0.5 eV/atom (ordered intermetallic stability)

Penalties:
  ×0.50 if no Ni AND no Co (not a superalloy system)
  ×0.40 if Tl, Hg, Cd, Pb (toxic; also low-melting → catastrophic at turbine temperatures)
  ×0.60 if band gap > 0.5 eV
  ×0.70 if no refractory metal (Al, Cr, Mo, W, Re, Ru, Ta, Nb) to provide strengthening
```

---

### 69. Hydrogen Embrittlement Resistant Steel

**Scientific Context**: Hydrogen embrittlement (HE) causes catastrophic brittle fracture in high-strength steels exposed to hydrogen environments (pipelines, pressure vessels, electrolyser components, bolts in H₂ infrastructure). Mechanism: H atoms diffuse into the lattice, reduce cohesive energy at grain boundaries, and facilitate dislocation-mediated fracture. Austenitic stainless steels (FCC Fe-Cr-Ni) are more HE-resistant than martensitic/ferritic steels. Duplex steels (50% austenite / 50% ferrite) offer a compromise. Ni improves HE resistance by reducing H diffusivity. Coating with Ni, Pd, or noble metals can prevent H ingress.

**Scoring Logic**:
```
HE resistance — primary strategy is austenite stabilisation:
  +0.38 if Fe + Cr + Ni present (austenitic stainless basis — 304, 316 type)
    additional +0.10 if Fe + Cr + Ni + Mo (316 — best standard HE resistance)
    additional +0.08 if Fe + Cr + Ni + Mn (Mn-austenite — cheaper Ni replacement)
    additional +0.08 if Fe + Cr + Ni + N (N stabilises austenite, improves strength)
    additional +0.07 if high Ni fraction (≥ 30 at% Ni) — Invar/Hastelloy range
  +0.28 if Fe + Cr + Ni + Mo + N (super-austenitic — 904L, 6Mo — highest HE resistance)

Secondary HE-resistant elements:
  +0.20 if Fe + Al present (Al addition to ferritic steel → reduces H diffusivity)
    additional +0.08 if Fe + Cr + Al (FeCrAl — also oxidation resistant, dual purpose)
  +0.18 if Fe + Si present (Si reduces H permeability in ferritic steels)
  +0.15 if Pd + Fe or Pd + Ni (Pd surface coating — hydrogen dissociation without absorption)
  +0.12 if Cu + Fe or Cu + Ni (Cu reduces H uptake from acidic corrosion)

Grain boundary strengthening (trap sites):
  +0.10 if V + N present (VN precipitates — benign H traps prevent grain boundary accumulation)
  +0.08 if Ti + C + N (TiC, TiN — beneficial trapping precipitates)
  +0.07 if Nb + C (NbC — grain boundary pinning + H trapping)

Electronic character:
  +0.15 if band gap < 0.1 eV (metallic)
  ×0.60 if band gap > 0.5 eV (ceramic — not a structural steel)

Crystal structure proxy:
  +0.12 if cubic crystal system (FCC austenite → low H diffusivity ~10⁻¹⁵ m²/s vs BCC ferritic 10⁻⁹ m²/s)

Stability:
  +0.08 if formation energy < −0.5 eV/atom

Penalties:
  ×0.50 if no Fe present (this scorer is explicitly for Fe-based steels)
  ×0.60 if only BCC-forming elements without austenite stabilisers (Cr-only ferritic → high HE susceptibility)
  ×0.40 if Tl, Hg, Cd, Pb, or As (toxic and embrittling even without H)
```

---

## Domain 15: Environmental Additions

### 70. Photocatalytic CO₂ Reduction

**Scientific Context**: Converts CO₂ + H₂O + light → fuels (CH₄, CH₃OH, CO, HCOOH) — solar fuel synthesis. Distinct from water splitting photocatalysis: requires CO₂ activation (multiple electron/proton transfers, thermodynamically uphill by 1.33–7.9 eV depending on product). Band gap must straddle CO₂ reduction potentials: CB must be above −0.2 V (CO₂/CO), −0.38 V (CO₂/HCOOH), −0.52 V (CO₂/CH₃OH), or −0.61 V (CO₂/CH₄) vs NHE. Since CB needs to be more negative than water splitting HER, band gap typically ≥ 2.0 eV for selectivity. Cu-based cocatalysts strongly enhance CH₄/CH₃OH selectivity.

**Scoring Logic**:
```
Hard requirement:
  if band gap < 1.8 eV → return 0.0 (insufficient driving force for CO₂ reduction potential)
  if band gap > 4.0 eV → return 0.0 (UV-only; low solar fraction)

Band gap score (peak at 2.5 eV — balance of visible absorption and thermodynamic driving force):
  core_score = max(0.35 − |band_gap − 2.5| × 0.12, 0.12)
  +0.10 if 2.0–3.2 eV (visible light range)
  +0.08 if 3.2–4.0 eV (UV-A range; lower but still useful)

Photocatalyst families with CO₂RR selectivity:
  +0.28 if Ti + O present (TiO₂ — benchmark photocatalyst; CO₂ to CO/CH₄ with UV)
    additional +0.10 if Ti + O + Cu (Cu/TiO₂ — Cu cocatalyst dramatically shifts to CH₄/CH₃OH)
  +0.25 if Bi + V + O (BiVO₄ — visible photocatalyst; CO₂ to CO)
  +0.22 if Zn + Ga + O (ZnGa₂O₄ — CO₂ to CH₄ under UV)
  +0.20 if Zn + O (ZnO — wide gap but good electron affinity for CO₂RR)
  +0.18 if In + Ta + O or In + Nb + O (InTaO₄ — photocatalytic CO₂ to CH₃OH)
  +0.15 if Ga + N + O (GaON — oxynitride; visible light CO₂ reduction)
  +0.15 if Ce + O (CeO₂ — CO₂ activation via Ce³⁺/Ce⁴⁺ redox; oxygen vacancy mechanism)
  +0.12 if Mg + Al + O (hydrotalcite proxy — base sites for CO₂ adsorption)

CO₂ activation enhancement:
  +0.15 if Cu present (Cu⁰/Cu⁺ cocatalyst — strongest CO₂RR selectivity for C₁ products)
  +0.10 if Ag present (Ag — selective CO production from CO₂)
  +0.08 if Ru present (Ru cocatalyst — enhances CH₃OH selectivity)
  +0.08 if Zn present (Zn — CO selective CO₂RR)

CO₂ adsorption sites:
  +0.10 if basic oxide: Mg, Ca, Ba, Sr + O (basic sites capture CO₂; La₂O₃, MgO)
  +0.08 if oxygen vacancy likely (reducible oxides: CeO₂, TiO₂, In₂O₃)

Stability:
  +0.10 if formation energy < −2.5 eV/atom (photocorrosion resistance)

Penalties:
  ×0.60 if Cd, Tl, Hg, or As (toxic in photocatalytic water/air treatment)
  ×0.50 if no CO₂-activating element (only inert oxides like Al₂O₃, SiO₂ alone)
  ×0.70 if band gap < 2.0 eV without Cu/Ag/Ru cocatalyst indicator
```

---

### 71. VOC Decomposition Catalyst

**Scientific Context**: Catalytic oxidation of volatile organic compounds (VOCs: benzene, toluene, xylene, formaldehyde, acetaldehyde) to CO₂ and H₂O for air purification and industrial emission control. Noble metal catalysts (Pt, Pd, Au) achieve complete oxidation at < 200°C. Transition metal oxides (MnO₂, Co₃O₄, CeO₂) operate at 200–400°C — cheaper but higher light-off temperature. Photocatalytic degradation with TiO₂ achieves room-temperature VOC removal under UV/visible light. Applications: indoor air quality (formaldehyde from furniture), industrial VOC abatement, vehicle emissions (three-way catalyst connection).

**Scoring Logic**:
```
Thermal catalytic VOC oxidation metals:
  Noble metals (lowest light-off temperature):
    +0.35 if Pt present (Pt — best benzene/toluene oxidation; complete at 150°C)
    +0.32 if Pd present (Pd — methane oxidation + VOC; slightly higher T than Pt)
    +0.28 if Au present (Au nanoparticles — CO + HCHO oxidation at room temperature)
    +0.25 if Ru or Rh present (high activity, expensive)

  Transition metal oxides (earth-abundant):
    +0.30 if Mn + O present (MnO₂ — best earth-abundant VOC catalyst; OMS-2 structure)
      additional +0.08 if Mn + Ce + O (MnCeOₓ — synergistic; better than either alone)
    +0.28 if Co + O present (Co₃O₄ — spinel; excellent for toluene oxidation)
      additional +0.08 if Co + Ce + O (CoCeOₓ — highly active composite)
    +0.25 if Ce + O present (CeO₂ — oxygen storage; La-doped CeO₂ very active)
    +0.22 if Fe + O present (Fe₂O₃ — HCHO, acetaldehyde oxidation)
    +0.20 if Cu + O present (CuO — chlorinated VOC decomposition)
    +0.18 if V + O present (V₂O₅ — aromatic VOC oxidation)
    +0.15 if Cr + O present (Cr₂O₃ — active but Cr⁶⁺ toxicity concern in use)
    +0.12 if Ni + O present (NiO — moderate VOC activity)

  Photocatalytic room-temperature path:
    +0.22 if Ti + O present (TiO₂ — photocatalytic HCHO/VOC under UV)
      additional +0.08 if Ti + O + visible absorber proxy (Bi, W, V) — visible-light active
    +0.18 if Bi + W + O (Bi₂WO₆ — visible-light VOC decomposition)
    +0.15 if Zn + O (ZnO — UV photocatalytic VOC)

Band gap:
  +0.15 if < 0.5 eV (metallic noble metal — direct catalytic pathway)
  +0.18 if 1.5–3.5 eV (oxide photocatalyst window or semiconducting oxide)
  +0.08 if 0.5–1.5 eV (narrow gap oxide)
  ×0.3 if > 4.5 eV AND no noble metal (transparent insulator — no catalytic or photocatalytic activity)

Redox cycle proxy:
  +0.10 if Ce + O (Ce³⁺/Ce⁴⁺ — lattice oxygen mechanism for VOC oxidation)
  +0.08 if Mn with mixed valence (Mn²⁺/Mn³⁺/Mn⁴⁺ — oxygen vacancy formation)

Stability:
  +0.08 if formation energy < −2.0 eV/atom (catalyst stability under repeated oxidation cycles)

Penalties:
  ×0.60 if no catalytically active element (inert oxides: Al₂O₃, SiO₂, ZrO₂ alone)
  ×0.40 if Tl, Hg, Cd, or As (toxic catalyst — creates secondary pollution)
  ×0.50 if Cr as primary active element (Cr⁶⁺ leaching risk — regulatory concern)
  ×0.70 if high-melting refractory metals only (W, Ta, Nb alone — not active for VOC oxidation)
```

---

## Updated Summary

### Complete Category List (72 Total)

| # | Category | Domain | Key Elements | Band Gap Target |
|---|----------|--------|-------------|----------------|
| 1 | Battery Cathode (Li-ion) | Energy Storage | Li, Fe/Mn/Co/Ni, P/Si/S+O | 0.01–3.0 eV |
| 2 | Battery Anode | Energy Storage | Si, Sn, Sb, Fe+TM | < 0.5 eV |
| 3 | Battery Cathode (Na-ion) | Energy Storage | Na, Fe/Mn, P/Si+O | 0.0–3.0 eV |
| 4 | Solid Electrolyte | Energy Storage | Li/Na, Zr/Al/S+halide | > 3.0 eV |
| 5 | Supercapacitor Electrode | Energy Storage | Ru/Mn/V/Ti+C, Co | < 1.0 eV |
| 6 | Redox Flow Battery | Energy Storage | V/Fe/Cr/Zn, S/Cl+O | any |
| 7 | Hydrogen Storage | Energy Storage | Mg/Li/Na+H, Ti/Ni/La | any (H required) |
| 8 | Na-S Battery Electrolyte | Energy Storage | Na, Al+O, Zr+O | > 4.0 eV |
| 9 | Battery Separator | Energy Storage | Al/Si/Ti/Zr+O, BN | > 3.0 eV |
| 10 | Liquid Electrolyte Component | Energy Storage | Li+P+F, Li+B+F | > 2.0 eV |
| 11 | Solar Absorber (Single Junction) | Energy Conversion | Cu-chalcopyrite/kesterite, Bi | 0.7–2.0 eV |
| 12 | Solar Absorber (Tandem Top Cell) | Energy Conversion | halide perovskite, Pb-free | 1.4–2.2 eV |
| 13 | Thermoelectric | Energy Conversion | Bi/Sb/Te/Se, heavy elements | 0.05–1.0 eV |
| 14 | Perovskite Stabiliser | Energy Conversion | Al/Si/Zr+O, F | > 3.0 eV |
| 15 | Luminescent Solar Concentrator | Energy Conversion | Eu/Ce/Mn, Cs+halide | 1.5–3.5 eV |
| 16 | Solar Thermal Absorber | Energy Conversion | TiN, CrOₓ, Cu/Fe+O | < 2.5 eV |
| 17 | Anti-Reflection Coating | Energy Conversion | MgF₂, SiO₂, Si₃N₄, TiO₂ | > 2.5 eV |
| 18 | OER Electrocatalyst | Catalysis | Ir/Ru, Fe/Co/Ni+O | < 2.0 eV |
| 19 | HER Electrocatalyst | Catalysis | Pt/Pd, Mo/W/Ni+S/P | < 1.5 eV |
| 20 | CO₂ Reduction Catalyst | Catalysis | Cu, Ag/Au/Sn/Bi | < 2.5 eV |
| 21 | Photocatalyst (Water Splitting) | Catalysis | Ti/Bi/Ta+O/N | 1.8–3.2 eV |
| 22 | NRR Catalyst | Catalysis | Mo/Fe/V/Ru+N | < 1.5 eV |
| 23 | Methane Activation | Catalysis | Cu/Fe+zeolite, Pd | < 2.0 eV |
| 24 | NOx SCR Catalyst | Catalysis | V/Cu/Fe+Ti/Al+O | < 2.0 eV |
| 25 | Selective Hydrogenation | Catalysis | Pd/Pt/Rh, Ni/Co+P | < 0.5 eV |
| 26 | Fischer-Tropsch | Catalysis | Co/Fe/Ru+C promoters | < 0.5 eV |
| 27 | Dehydrogenation/LOHC | Catalysis | Pt+Sn, Ni/V+O | < 0.5 eV |
| 28 | Photocatalytic CO₂ Reduction | Environmental | Ti/Bi+O+Cu, CeO₂ | 1.8–4.0 eV |
| 29 | VOC Decomposition | Environmental | Mn/Co/Ce/Pt+O | any |
| 30 | Semiconductor (General) | Electronics | Si, GaAs, ZnO | 0.1–4.5 eV |
| 31 | LED / Light Emitter | Electronics | GaN, halide perovskite | 1.6–4.5 eV |
| 32 | Photodetector | Electronics | InGaAs, HgCdTe, Si | 0.3–4.5 eV |
| 33 | Transparent Conductor | Electronics | In/Sn/Zn+O | > 3.0 eV |
| 34 | Ferroelectric | Electronics | Ba/Sr/Pb+Ti/Zr+O, non-centro | > 1.5 eV |
| 35 | Piezoelectric | Electronics | AlN, PZT-type, LiNbO₃ | > 2.0 eV |
| 36 | Phase Change Memory | Electronics | Ge+Sb+Te, GST family | 0.3–2.5 eV |
| 37 | High-k Dielectric | Electronics | HfO₂, ZrO₂, La₂O₃ | > 4.0 eV |
| 38 | Nonlinear Optical | Electronics | LiNbO₃, BBO, KTP | > 2.5 eV |
| 39 | Memristor/Neuromorphic | Electronics | HfO₂, TaOₓ, Ag+S | 1.0–8.0 eV |
| 40 | 2D Material | Electronics | MoS₂, WS₂, C, BN, MXene | 0.0–6.0 eV |
| 41 | Optical Fibre/Waveguide | Electronics | SiO₂, ZBLAN, As₂S₃ | > 3.0 eV |
| 42 | Permanent Magnet | Magnetics | Nd/Sm+Fe/Co+B/N | < 0.1 eV |
| 43 | Soft Magnet | Magnetics | Fe+Si/B, ferrites | < 0.5 eV |
| 44 | Magnetic Semiconductor | Magnetics | Mn/Fe+III-V or II-VI | 0.1–4.0 eV |
| 45 | Spintronic MTJ | Magnetics | MgO, Al₂O₃, HfO₂ | > 3.5 eV |
| 46 | Thermal Barrier Coating | Coatings | YSZ, pyrochlore, hexaaluminate | > 3.0 eV |
| 47 | Thermal Interface Material | Coatings | AlN, SiC, BN, diamond | > 2.5 eV |
| 48 | Hard Coating / Wear Resistant | Coatings | TiN, TiAlN, WC, DLC | > 1.5 eV |
| 49 | Corrosion Resistant Coating | Coatings | Cr₂O₃, Al₂O₃, fluorides | > 3.0 eV |
| 50 | Refractory / UHTC | Coatings | HfC, ZrB₂, TaC, SiC | any |
| 51 | Nuclear Fuel Cladding | Nuclear | Zr alloys, FeCrAl, SiC | < 0.5 eV |
| 52 | Tritium Breeder | Nuclear | Li+Si/Ti/Zr+O | > 4.0 eV |
| 53 | Radiation Shielding | Nuclear | Pb/W/Ba, B/Gd+neutron | any |
| 54 | Nuclear Waste Immobilisation | Nuclear | Zirconolite, pyrochlore, SYNROC | > 4.0 eV |
| 55 | Qubit Host Material | Quantum | C, Si, SiC, Al₂O₃ | > 2.0 eV |
| 56 | Topological Insulator | Quantum | Bi₂Te₃, Bi₂Se₃, SnTe | 0.02–2.0 eV |
| 57 | Topological/Majorana Host | Quantum | InAs, InSb, FeTeSe | 0.05–1.5 eV |
| 58 | Superconductor | Quantum | Nb, cuprates, FeSe, hydrides | < 0.5 eV |
| 59 | Radiation Detector/Scintillator | Quantum | CsI, BaI₂, LYSO, CdZnTe | 0.5–7.0 eV |
| 60 | Multiferroic | Quantum | BiFeO₃, RMnO₃ | 1.0–3.5 eV |
| 61 | Biodegradable Implant | Biomedical | Mg/Zn/Fe alloys | < 0.5 eV |
| 62 | Bone Scaffold/HA Analog | Biomedical | Ca+P/Si+O | > 4.0 eV |
| 63 | Antibacterial Coating | Biomedical | Ag/Cu/Zn, TiO₂ | any |
| 64 | CO₂ Capture Sorbent | Environmental | CaO, Na/K/Li+O, zeolites | > 4.0 eV |
| 65 | Desalination Membrane | Environmental | Al+Si+O, MXene, SiO₂ | > 5.0 eV |
| 66 | Photocatalytic Pollutant Degradation | Environmental | TiO₂, BiVO₄, WO₃ | 1.5–4.0 eV |
| 67 | Lightweight Structural | Structural | Mg/Al/Ti alloys, SiC | any |
| 68 | Shape Memory Alloy | Structural | NiTi, Cu-Al-Ni | < 0.1 eV |
| 69 | Metallic Glass | Structural | Zr+multi, Fe+Si+B | < 0.1 eV |
| 70 | Superalloy | Structural | Ni+Al+Cr+Re, Co+Al+W | < 0.1 eV |
| 71 | H₂ Embrittlement Resistant | Structural | Fe+Cr+Ni+Mo, austenitic | < 0.1 eV |
| 72 | Nuclear Fuel Cladding | Nuclear | Zr, FeCrAl, SiC | < 0.5 eV |

