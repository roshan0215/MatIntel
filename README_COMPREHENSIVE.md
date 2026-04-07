# MatIntel — Comprehensive Materials Intelligence Platform

**End-to-end computational materials screening and discovery platform** for identifying high-potential materials across 31 application domains. Combines domain-specific scoring with real-world viability assessment and AI-driven synthesizability predictions via CLscore.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [The 31 Scoring Categories](#the-31-scoring-categories)
   - [Domain 1: Energy Storage](#domain-1--energy-storage)
   - [Domain 2: Energy Conversion](#domain-2--energy-conversion)
   - [Domain 3: Electronics & Optoelectronics](#domain-3--electronics--optoelectronics)
   - [Domain 4: Magnetics](#domain-4--magnetics)
   - [Domain 5: Thermal & Structural Coatings](#domain-5--thermal--structural-coatings)
   - [Domain 6: Emerging Applications](#domain-6--emerging--specialised)
4. [Viability Scoring](#viability-scoring)
5. [CLscore: Synthesizability Prediction](#clscore-synthesizability-prediction)
6. [Interactive Streamlit Dashboard](#interactive-streamlit-dashboard)
7. [Pipeline Architecture](#pipeline-architecture)
8. [Installation & Setup](#installation--setup)
9. [Running the System](#running-the-system)
10. [File Outputs & Data Structures](#file-outputs--data-structures)
11. [Advanced Usage](#advanced-usage)
12. [Technical Implementation Details](#technical-implementation-details)
13. [Recent Updates (March 2026)](#recent-updates-march-2026)
14. [Top-10 Provenance Findings (March 2026 Snapshot)](#top-10-provenance-findings-march-2026-snapshot)

---

## System Overview

MatIntel solves a critical problem in materials discovery: **given millions of computationally stable crystal structures, which ones are actually viable for specific applications and practically synthesizable?**

### Key Features

- **31 application-specific scorers** driven by materials science principles (band gap requirements, element chemistry, crystal structure indicators)
- **Viability multiplier** incorporating cost, elemental abundance, supply chain risk, radioactive exclusion, and rare-earth penalties
- **CLscore synthesis predictor** using deep graph neural networks (KAIST Synthesizability-PU-CGCNN) to estimate likelihood of successful synthesis from known materials
- **Interactive Streamlit dashboard** for filtering, ranking, and exploring candidates
- **Resume-safe batch processing** for scoring massive datasets with checkpointing
- **Windows-first design** with PowerShell automation scripts

### Data Pipeline

```
Raw CSV (GNoME dataset)
  ↓
Working Dataset (quality checks, formula validation)
  ↓
Featured Dataset (element properties, structure metrics via matminer)
  ↓
Scored Dataset (31-category scores + viability + CLscore)
  ↓
Interactive Dashboard (filter, rank, export)
```

### Recent Production Additions

- **Experimental reference ingestion** via `scripts/build_experimental_reference.py` (Materials Project + optional JARVIS/matminer enrichment)
- **Provenance-aware filtering** in the dashboard (`All`, `Experimental`, `Synthesized`)
- **CLscore cache hydration** in `app.py` from `data/processed/clscore_all_results.csv` when `scored_dataset.csv` lacks embedded CLscore
- **Export bundle support** from the dashboard: ranked CSV, matching CIF ZIP, and PDF summary report
- **Materials Project formula lookup** and in-app crystal structure preview (with `py3Dmol` installed)

---

## Architecture & Data Flow

### Pipeline Components

```
src/matintel/
├── config.py              # Application labels, element prices, critical minerals
├── data_sources.py        # Data load/validation, demo dataset fallback
├── features.py            # Matminer feature extraction
├── scoring.py             # 31-category scoring functions + registry
├── viability.py           # Cost, abundance, supply risk, radioactivity, REE penalties
├── clscore.py             # CLscore prediction wrapper (KAIST integration)
├── pipeline.py            # Orchestration: load → featurize → score → viability
└── explanations.py        # AI summary generation for materials
```

### High-Level Processing

1. **Data Ingestion**: Load CSV (GNoME or demo dataset)
2. **Feature Engineering**: Extract 138 matminer elemental features per material
3. **Application Scoring**: Run 31 domain-specific scorers (0.0–1.0 per app)
4. **Viability Assessment**: Multiply by real-world constraints (cost, abundance, supply risk)
5. **Synthesizability (CLscore)**: Predict likelihood of successful synthesis from crystal structure
6. **Interactive Ranking**: Filter by application, viability, and synthesizability in Streamlit

---

## The 31 Scoring Categories

Each material receives a score of **0.0–1.0** for each category, then multiplied by a **viability score** (0.0–1.0) that accounts for cost, abundance, supply risk, and synthesizability.

### DOMAIN 1 — ENERGY STORAGE

#### 1. Battery Cathode (Li-ion)
**Scientific Context**: Positive electrode in lithium-ion batteries. Must reversibly intercalate Li⁺ at high voltage while maintaining electronic conductivity.

**Key Requirements**:
- Contains Li (working ion)
- Transition metal redox centers (Mn, Fe, Co, Ni, V, Cr)
- Band gap: 0–3 eV (pure insulators inhibit rate performance)
- Polyanionic framework (P-O, Si-O, S-O) or layered oxide structure preferred

**Scoring Logic**:
- Base 0.35 for transition metals present
- +0.20 for polyanionic frameworks (phosphates, silicates, sulfates)
- +0.10 for layered oxide structures without polyanionic framework
- +0.25 for favorable band gap (0.01–3.0 eV)
- +0.10 for strong negative formation energy (<−1.0 eV/atom)
- ×0.7 penalty if fluoride present (electrolyte instability)
- ×0.6 penalty if sulfur without phosphorus (polysulfide dissolution)

**Commercial Examples**: LiCoO₂, LiFePO₄, NMC (nickel-manganese-cobalt oxide)

**Viability Considerations**: Co is expensive (~$33/kg) and high-risk. Fe and Mn are abundant; LFP is fastest-growing due to cost advantage.

---

#### 2. Battery Anode
**Scientific Context**: Negative electrode for Li/Na-ion cells. Stores working ions at low voltage. Must conduct electrons, not necessarily contain the working ion.

**Key Requirements**:
- Metallic or near-metallic conductivity (band gap <0.5 eV preferred)
- High-capacity alloying elements (Si ~3500 mAh/g, Sn, Sb, P, Ge)
- OR conversion-type materials (Fe/Co/Ni oxides, sulfides)
- NO working ion (Li/Na) already in structure (discharged state is metal only)

**Scoring Logic**:
- +0.40 for alloying anode elements (Si, Sn, Sb, P, Ge, Bi, Al)
- +0.25 for conversion-type (transition metal + anion)
- +0.20 for metallic character (band gap <0.1 eV)
- +0.12 for near-metallic (band gap 0.1–0.5 eV)
- +0.15 bonus for Fe-Si combination (well-studied family)
- ×0.5 penalty if Li or Na already present

**Commercial Examples**: Graphite (current standard), Si/SiO₂ composites (high capacity but volume expansion), FeSi₂

**Viability Considerations**: Si is abundant and cheap. Fe excellent. Sb and Ge carry supply risk.

---

#### 3. Battery Cathode (Na-ion)
**Scientific Context**: Positive electrode for sodium-ion batteries (CATL producing since 2023). Na⁺ is larger (1.02 Å vs 0.76 Å Li⁺), requiring more open frameworks. Cost advantage comes from avoiding Li, Co, and expensive rare earths.

**Key Requirements**:
- Contains Na (mandatory)
- Transition metal redox centers (Fe and Mn strongly preferred for cost)
- More open framework than Li-ion cathodes (larger interlayer spacing)
- Prussian blue analogues, layered oxides, or polyanionic compounds

**Scoring Logic**:
- +0.40 for preferred redox metals (Fe, Mn)
- +0.25 for other redox metals (Co, Ni, V, Cr, Cu, Ti)
- +0.20 for polyanionic framework + oxygen
- +0.12 for oxide layered structure
- +0.20 for favorable band gap (0.0–3.0 eV)
- +0.08 bonus for Li-free (pure Na-ion chemistry)

**Commercial Examples**: NMF (NaMn₀.₅Fe₀.₅O₂), NFSF (NaFeF₃), NVP (Na₃V₂(PO₄)₃)

**Viability Considerations**: Best commercial case rests on avoiding Li, Co, expensive REEs. Mn and Fe are ideal.

---

#### 4. Solid Electrolyte
**Scientific Context**: Solid-state ionic conductor replacing liquid electrolyte in all-solid-state batteries. Must conduct Li⁺ or Na⁺ while remaining electronically insulating.

**Key Requirements**:
- Wide band gap >3 eV (must be electronically insulating)
- Contains Li or Na (mandatory)
- Known framework elements: Zr, La, Al, P, Si, S
- Sulfide-based: 2–4 eV band gap acceptable (higher conductivity)
- Oxide-based: >4 eV (better stability vs Li metal)
- Halide-based: Cl/Br/I with Li/Na (active research)

**Scoring Logic**:
- Hard requirement: returns 0.0 if no Li/Na
- +0.35 for band gap ≥4.0 eV (oxide-based)
- +0.25 for band gap 3.0–4.0 eV
- +0.10 for band gap 2.0–3.0 eV (sulfide-based)
- Returns 0.0 if band gap <2.0 eV (too conducting)
- +0.25 for oxide framework elements (Zr, La, Al, Ta, Nb, Ti)
- +0.20 for sulfide framework + sulfur (P, Si, Ge, Sn, As + S)
- +0.20 for halide elements (Cl, Br, I, F)
- +0.10 if oxygen present
- +0.10 for very stable (formation energy <−2.0 eV/atom)

**Known Materials**: NASICON (Na₃Zr₂Si₂PO₁₂), LLZO (Li₇La₃Zr₂O₁₂), argyrodite (Li₆PS₅Cl)

---

#### 5. Hydrogen Storage
**Scientific Context**: Reversibly absorbs/releases hydrogen for fuel cell vehicle storage. Target: >6.5 wt% gravimetric capacity, release at 60–120°C.

**Key Requirements**:
- Contains H (mandatory)
- Light metals preferred for gravimetric density (Mg, Li, Na, Al, Ca)
- Transition metals for kinetics (Fe, Ti, Ni, V, Zr, La, Ce)
- Moderate stability: too stable = can't release H₂; too weak = loses storage
- Formation energy range: −0.3 to −1.5 eV/atom optimal

**Scoring Logic**:
- Hard requirement: returns 0.0 if no H
- +0.30 for light metals (Li, Na, Mg, Al, Ca, K)
- +0.25 for transition metal hydrides (Fe, Ti, Ni, V, Zr, La, Ce)
- +0.15 for borohydrides (B present)
- +0.10 for aluminum hydrides (Al + H)
- +0.20 for optimal formation energy range (−1.5 to −0.3 eV/atom)
- +0.10 for weak binding (near-zero formation energy)

**Commercial Examples**: Complex hydrides (NaBH₄, LiAlH₄), metal hydrides (FeTiH₂, LaNi₅H₆)

---

### DOMAIN 2 — ENERGY CONVERSION

#### 6. Solar Absorber (Single Junction)
**Scientific Context**: Photovoltaic absorber layer for single-junction solar cells. Must absorb visible light efficiently (~1.0–1.8 eV optimal band gap per Shockley-Queisser). Direct band gap strongly preferred for higher absorption coefficients.

**Key Requirements**:
- Band gap: 1.0–1.8 eV (tight range; optimum ~1.34 eV)
- Direct band gap (indirect gaps have lower absorption)
- No highly toxic elements (Cd, Pb penalized but not forbidden)
- Earth-abundant elements preferred

**Scoring Logic**:
- Returns 0.0 if band gap unavailable or outside 0.8–1.8 eV range
- Gaussian peak at 1.34 eV: score = max(0.50 − |E_g − 1.34| × 0.5, 0.25)
- +0.10 if narrowly outside range (0.8–1.0 eV, tandem-compatible)
- +0.25 for chalcopyrite family (Cu, In/Ga, Se/S; ≥3 elements)
- +0.20 for kesterite family (Cu, Zn, Sn, S/Se; ≥3 elements)
- +0.10 for chalcogenide metals (Mo, W, Sb, Bi, Ge)
- ×0.6 if Cd present (regulatory/toxicity barrier)
- ×0.5 if As/Hg present (severe toxicity)
- ×0.7 if Pb present (regulatory pressure)

**Example Candidates**: CIGS (Cu(In,Ga)Se₂), CZTS (Cu₂ZnSnS₄), Cu₆SiMoS₈ (1.03 eV from your dataset)

---

#### 7. Solar Absorber (Tandem Top Cell)
**Scientific Context**: Top cell in perovskite-silicon tandem solar cells (2025 record: ~34% efficiency). Must absorb photons >1.6 eV while transmitting <1.6 eV light to Si bottom cell.

**Key Requirements**:
- Band gap: 1.6–2.0 eV specifically (peak at ~1.75 eV)
- Direct band gap
- Thin-film processable
- Halide perovskite structure type bonus
- Pb-free preferred (but Pb-halide perovskites still dominate)

**Scoring Logic**:
- Returns 0.0 if band gap unavailable or outside useful range
- Gaussian peak at 1.75 eV: score = max(0.50 − |E_g − 1.75| × 0.8, 0.25)
- +0.15 if just outside range (1.5–1.6 eV)
- +0.30 for halide perovskite ABC₃ structure (halides + A-site Cs/Rb/K/Na + B-site Pb/Sn/Ge/Bi/Sb/In)
- +0.15 for partial perovskite character (halides + B-site metals)
- +0.15 for Pb-free compounds (strong regulatory driver)

**Example Materials**: CsPbI₃ (1.73 eV), CsSnI₃ (1.3 eV, too narrow but Pb-free), mixed halide variants

---

#### 8. Thermoelectric
**Scientific Context**: Converts heat↔electricity via Seebeck effect. Figure of merit ZT = S²σT/κ. Need high Seebeck, high conductivity, low thermal conductivity. Heavy atoms essential for phonon scattering.

**Key Requirements**:
- Near-metallic band gap (<0.5 eV optimal, up to 1.0 eV acceptable)
- Heavy elements present (atomic mass >100) — critical for low κ
- Anharmonic bonding (chalcogenides, halides especially)
- Complex unit cell or multiple inequivalent sites

**Scoring Logic**:
- Band gap weighting:
  - +0.30 for semimetallic (<0.1 eV)
  - +0.35 for ideal narrow gap (0.1–0.5 eV)
  - +0.20 for 0.5–1.0 eV
  - +0.05 for >1.0 eV (too insulating)
- Heavy element count: +min(heavy_count × 0.12, 0.35) for {Pb, Bi, Sb, Te, Se, Tl, In, Sn, Ge, Ba, Cs, I, Br, Ag, Hg}
- +0.10 for Te present (primary thermoelectric element)
- +0.07 for Se/S present (secondary)
- +0.08 for halides (I, Br, Cl)
- ×0.4 penalty if all light elements (high thermal conductivity)

**Commercial Examples**: PbTe, Bi₂Te₃, SnSe, Ba₁₂SiSn₃I₈ (top result in your dataset)

---

#### 9. OER Electrocatalyst (Oxygen Evolution)
**Scientific Context**: Catalyzes 2H₂O → O₂ + 4H⁺ + 4e⁻ (acidic) or 4OH⁻ → O₂ + 2H₂O + 4e⁻ (alkaline). Rate-limiting step in green hydrogen production via water electrolysis.

**Key Requirements**:
- Metallic or near-metallic (electronic conductivity required)
- OER-active transition metals:
  - Acidic: Ir, Ru essential (only metals stable in acidic OER)
  - Alkaline: Fe, Co, Ni, Mn well-established
- Oxide/hydroxide/oxyhydroxide structure preferred
- Must conduct electrons to electrode

**Scoring Logic**:
- +0.40 for acidic-active metals (Ir, Ru)
- +0.30 for alkaline-active metals (Fe, Co, Ni, Mn)
- Electronic conductivity bonus:
  - +0.25 for metallic (<0.5 eV band gap)
  - +0.10 for low-gap (<2.0 eV)
  - ×0.3 if insulating (>2.0 eV)
- +0.15 if oxygen present (oxide catalyst activation)
- +0.10 for Fe-Si combination (silicide catalysts; Fe₅Si₃ example)
- +0.10 bonus for perovskite B-site metals in oxide (structural family)

**Commercial/Promising**: IrO₂, RuO₂, NiFe-LDH, Fe₅Si₃ (your discovery)

---

#### 10. HER Electrocatalyst (Hydrogen Evolution)
**Scientific Context**: Catalyzes 2H⁺ + 2e⁻ → H₂ (acidic). Cathode reaction; different element preferences than OER. Pt is benchmark (expensive); MoS₂, CoP, Ni₂P are cheaper alternatives.

**Key Requirements**:
- HER-active elements:
  - Noble metals (Pt, Pd, Rh, Ir): expensive but highly active
  - Earth-abundant (Mo, W, Ni, Co, Fe, Cu): cheaper, need optimization
- Sulfides and phosphides excellent (MoS₂, CoP, Ni₂P edge sites)
- Metallic or semimetallic preferred

**Scoring Logic**:
- +0.40 for noble metals (Pt, Pd, Rh, Ir)
- +0.30 for earth-abundant transition metals (Mo, W, Ni, Co, Fe, Cu)
- Chalcogenide bonuses:
  - +0.20 for sulfides + earth-abundant metals
  - +0.20 for phosphides + earth-abundant metals
  - +0.15 for selenides + earth-abundant metals
  - +0.10 for nitrides + earth-abundant metals
- Electronic character bonus:
  - +0.20 for metallic (<0.5 eV)
  - +0.08 for low-gap (0.5–1.5 eV)

**Examples**: Pt, MoS₂, CoP, Ni₂P, Fe-Ni alloys

---

#### 11. CO₂ Reduction Catalyst
**Scientific Context**: Electrochemical reduction of CO₂ to fuels/chemicals (CO, formate, methanol, ethanol, ethylene). Cu uniquely selective for multi-carbon products (C-C coupling). Active research area for carbon utilization.

**Key Requirements**:
- Copper strongly preferred for C-C coupling selectivity
- Other CO₂RR-active metals (Ag, Au, Zn, Sn, Bi, In, Pb, Pd)
- Metallic conductivity required
- Oxide or oxide-derived surfaces

**Scoring Logic**:
- +0.35 for Cu (C-C coupling selectivity)
- +0.25 for other CO₂RR metals (Ag, Au, Zn, Sn, Bi, In, Pb, Pd)
- Electronic character:
  - +0.20 for metallic (<0.5 eV)
  - +0.08 for low-gap (<2.0 eV)
- +0.15 for oxide surface with CO₂RR metal
- +0.05 bonus if noble-metal free (earth-abundant)

**Examples**: Cu foil, Cu-M alloys, CuO-derived catalysts

---

#### 12. Photocatalyst (Water Splitting)
**Scientific Context**: Light-driven water splitting where both OER and HER occur on same particle. Band edges must straddle water redox potentials. Band gap >1.23 eV (thermodynamic minimum) but ideally <3 eV (visible light).

**Key Requirements**:
- Band gap: 1.8–3.2 eV optimal (peak at ~2.2 eV for best balance)
- Band edges must straddle:
  - Conduction band < −0.41 V vs NHE (H₂ evolution)
  - Valence band > +0.82 V vs NHE (O₂ evolution)
- Known photocatalyst families: oxides, nitrides, sulfides

**Scoring Logic**:
- Returns 0.0 if band gap <1.23 eV or >3.2 eV
- Gaussian peak at 2.2 eV: score = max(0.40 − |E_g − 2.2| × 0.15, 0.20)
- +0.10 if thermodynamically possible but suboptimal (1.23–1.8 eV)
- +0.25 for oxide photocatalysts (Ti, Zn, Ga, In, Nb, Ta, W, Mo, Fe, Bi + O)
- +0.20 for nitride photocatalysts (N + Ga/Ta/Ge/C)
- +0.10 for sulfide photocatalysts (S + anion)
- ×0.6 if Cd present (toxicity)

**Examples**: TiO₂ (anatase/rutile, 3.0–3.2 eV, limited visible), Ta₃N₅ (2.1 eV), BiVO₄ (2.4 eV), CdS (2.4 eV, toxic)

---

### DOMAIN 3 — ELECTRONICS & OPTOELECTRONICS

#### 13. Semiconductor (General)
**Scientific Context**: General-purpose catch-all for semiconductor applications not covered by more specific categories.

**Key Requirements**:
- Band gap: 0.1–4.0 eV (semiconductor range)
- Known semiconductor element families (IV, III-V, II-VI, TMO)

**Scoring Logic**:
- Returns 0.0 if band gap outside 0.1–4.0 eV range
- +0.50 if band gap 0.5–3.0 eV (ideal range)
- +0.25 if band gap outside ideal but in full range
- Element families:
  - +0.20 for Group IV (Si, Ge, C)
  - +0.20 for III-V (≥2 elements from {Ga, In, Al, N, P, As, Sb})
  - +0.15 for II-VI (≥2 elements from {Zn, Cd, Hg, O, S, Se, Te})
- +0.10 for TMOs (transition metal oxides)

**Examples**: Si, GaAs, GaN, ZnO, SnO₂, WO₃

---

#### 14. LED / Light Emitter
**Scientific Context**: Light-emitting diode material. Requires direct band gap in visible range with high quantum yield. Different from solar absorber despite potential overlap in band gap.

**Key Requirements**:
- Band gap: 1.77–3.1 eV (visible light, 700–400 nm)
- Direct band gap preferred
- III-V semiconductors (GaN, AlGaInP) or II-VI (ZnSe) or halide perovskites

**Scoring Logic**:
- +0.45 for visible range (1.77–3.1 eV)
- +0.20 for near-UV (3.1–4.0 eV)
- Returns 0.0 if outside 1.77–4.0 eV
- +0.30 for III-V metals + anions (Ga/In/Al + N/P/As)
- +0.15 for II-VI compounds (Zn/Cd + S/Se/Te)
- +0.20 for halide perovskites (halides + Cs/Rb + Pb/Sn/In)
- ×0.3 penalty if pure Si (indirect band gap, poor light emission)

**Commercial**: GaN (blue, 3.4 eV), AlGaInP (red/green, 1.8–2.2 eV), CsPbI₃ (perovskite, 1.73 eV)

---

#### 15. Photodetector
**Scientific Context**: Converts light to electrical signal. Band gap range wider than solar/LED (UV to infrared applications). Key families: Si (vis/NIR), InGaAs (telecom), HgCdTe (IR), perovskites (fast response).

**Key Requirements**:
- Band gap: 0.3–4.5 eV (broader range than absorbers/emitters)
- High carrier mobility preferred
- Fast response, high sensitivity

**Scoring Logic**:
- +0.40 for 0.3–4.5 eV range
- Sub-range bonuses:
  - +0.10 for IR (0.3–1.0 eV)
  - +0.15 for visible (1.0–2.0 eV)
  - +0.10 for UV (2.0–3.5 eV)
- +0.25 for III-V semiconductors (high-mobility)
- +0.20 for halide perovskites (fast response)
- +0.15 for HgCdTe (IR, but toxic)

**Examples**: Si photodiode, InGaAs (1.55 μm telecom), HgCdTe

---

#### 16. Transparent Conductor
**Scientific Context**: Wide band gap BUT electrically conductive via degenerate doping. Commercial: ITO (In₂O₃:Sn, ~$50/kg). Problem: In is expensive and scarce. Alternatives: AZO, FTO, Ga₂O₃.

**Key Requirements**:
- Band gap >3.0 eV (transparent in visible)
- Low effective electron mass (high mobility when doped)
- Oxides strongly preferred
- Earth-abundant alternatives to In

**Scoring Logic**:
- Returns 0.0 if band gap <3.0 eV (absorbs visible)
- +0.45 if band gap ≥3.5 eV
- +0.30 if band gap 3.0–3.5 eV
- +0.35 for oxide TCOs (In/Sn/Zn/Ga/Cd/Al/Ti + O)
- +0.15 for earth-abundant alternatives (Zn-O without In → AZO/GZO)
- +0.10 for Sn-O (FTO family)
- ×0.4 penalty if non-oxide (TCOs are almost all oxides)

**Commercial**: ITO, AZO (Al:ZnO), FTO (Sn:SnO₂)

---

#### 17. Ferroelectric
**Scientific Context**: Spontaneous electric polarization reversible by applied field. Used in capacitors, FeRAM, sensors, actuators. Requires non-centrosymmetric crystal symmetry.

**Key Requirements**:
- Non-centrosymmetric space group (68 of 230)
- Band gap >1.5 eV (insulators preferred)
- Perovskite ABO₃ structure type bonus

**Scoring Logic**:
- +0.35 if structure available and non-centrosymmetric (space group analysis)
- +0.35 if perovskite elements: A-site {Ba, Sr, Ca, Pb, Na, K, Bi} + B-site {Ti, Zr, Nb, Ta, Fe, Mn, W} + O
- Band gap bonus:
  - +0.20 if band gap ≥2.5 eV (good insulation)
  - +0.10 if band gap 1.5–2.5 eV
- +0.10 for Pb-free ferroelectrics (regulatory bonus)

**Examples**: BaTiO₃ (perovskite, 3.2 eV, polar), PZT (Pb(Zr,Ti)O₃, 3.5 eV)

---

#### 18. Piezoelectric
**Scientific Context**: Converts mechanical stress ↔ electrical charge. Same symmetry requirement as ferroelectric. Applications: sensors, actuators, energy harvesters. Industrial: PZT (lead-heavy); alternatives: AlN, KNbO₃, BaTiO₃.

**Key Requirements**:
- Non-centrosymmetric / polar space group
- Band gap >3.0 eV (good insulation)
- Known piezoelectric element combinations

**Scoring Logic**:
- +0.40 if polar space group detected
- +0.30 for known piezo metals in oxides (Ti, Zr, Nb, Ta, Al, Ga, Zn, Li, Ba, Pb, K + O)
- +0.20 for AlN (high-frequency MEMS standard)
- +0.15 for wide band gap (≥3.0 eV)
- +0.15 bonus for Pb-free materials

**Examples**: PZT, BaTiO₃, AlN, KNbO₃

---

#### 19. Topological Insulator
**Scientific Context**: Bulk insulating but surface/edge-conducting states protected by time-reversal symmetry. Applications in quantum computing, spintronics. Requires strong spin-orbit coupling.

**Key Requirements**:
- Small band gap (0.05–1.5 eV) with bulk insulation
- Heavy elements with strong spin-orbit coupling (Bi, Sb, Pb, Sn, Te, Se, Tl, Hg, In)
- Quintuple-layer (Bi₂Te₃ family) or IV-VI structures

**Scoring Logic**:
- Returns 0.0 if band gap <0.05 eV or >1.5 eV
- +0.35 if band gap 0.05–0.5 eV (ideal TI gap)
- +0.15 if band gap 0.5–1.5 eV (suboptimal but possible)
- Heavy element count: +min(heavy_count × 0.15, 0.40) for {Bi, Sb, Pb, Sn, Te, Se, Tl, Hg, In}
- +0.20 for quintuple-layer structure (Bi + Te/Se/S)
- +0.15 for IV-VI structure (Sn/Pb/Ge + Te/Se/S)

**Examples**: Bi₂Te₃, Bi₂Se₃, BiSb, SnTe, Pb₁₋ₓSnₓSe

---

### DOMAIN 4 — MAGNETICS

#### 20. Permanent Magnet
**Scientific Context**: Hard magnet with high coercivity/remanence (stays magnetized; used in EV motors, wind turbines, hard drives). Nd₂Fe₁₄B dominates (40% world market). Alternative: Sm₂Co₁₇ (high-temperature). Both require rare-earth elements.

**Key Requirements**:
- Magnetic moment carriers (Fe, Co, Ni)
- Rare-earth anisotropy (Nd, Sm, Dy, Pr for magnetocrystalline anisotropy)
- Metallic (band gap near 0)
- High density of magnetic atoms

**Scoring Logic**:
- Returns 0.0 if no magnetic metals
- +0.30 for magnetic metals (Fe, Co, Ni)
- +0.35 for REE anisotropy centers (Nd, Sm, Dy, Pr, Tb, Ho)
- +0.15 for boron (Nd₂Fe₁₄B type standard)
- Electronic character:
  - +0.15 if metallic (<0.1 eV)
  - ×0.5 if semiconducting (rare for PMs)
- +0.05 bonus for N interstitials (Sm₂Fe₁₇N₃ emerging class)

**Notable**: REE penalty in viability applies correctly here; permanent magnets legitimately require REEs. This is highest-value REE application.

**Commercial**: Nd₂Fe₁₄B, Sm₂Co₁₇, Sm₂Fe₁₇N₃

---

#### 21. Soft Magnet
**Scientific Context**: Low coercivity (easy magnetize/demagnetize). Used in transformer cores, inductors, shielding, sensors. Fe-Si electrical steel dominates. Ferrites for high-frequency. NO rare earths required (cost advantage).

**Key Requirements**:
- Magnetic metals (Fe, Co, Ni)
- NO rare earths (cost penalty if present)
- Fe-Si alloys, ferrites, amorphous/nanocrystalline

**Scoring Logic**:
- Returns 0.0 if no magnetic metals
- +0.25 for magnetic metals
- +0.30 for Fe-Si alloys (electrical steel)
- +0.25 for ferrites (Mn-Zn, Ni-Zn, spinel Fe-O + transition metals)
- Amorphous indicators:
  - +0.10 for Fe-B
  - +0.10 for Fe-P
- +0.15 if metallic (<0.5 eV)
- ×0.7 penalty if REE present (doesn't need them, adds cost)

**Commercial**: FeSi (electrical steel), Mn-Zn ferrite, Ni-Zn ferrite

---

#### 22. Magnetic Semiconductor / Spintronics
**Scientific Context**: Semiconductor with ferromagnetic/antiferromagnetic ordering. Applications in spin-LEDs, magnetic RAM, quantum computing. Dilute magnetic semiconductors (DMS) like (Ga,Mn)As are main family.

**Key Requirements**:
- Band gap: 0.1–3.5 eV (semiconductor range)
- Magnetic 3d transition metals OR rare-earth magnetic elements
- Semiconductor host matrix

**Scoring Logic**:
- Returns 0.0 if band gap outside 0.1–3.5 eV
- +0.20 for semiconductor gap range
- +0.30 for 3d magnetic metals (Mn, Fe, Co, Ni, Cr, V)
- +0.25 for REE magnetic elements (Gd, Eu, Dy, Nd, Sm, Tb, Ho, Er)
- +0.15 for semiconductor host (Ga, In, Ge, Si, Zn, Cd)
- +0.10 for chalcogenide magnetic semiconductors (Eu/Gd + S/Se/O/Te)

**Examples**: (Ga,Mn)As, DMS, rare-earth hydrides

---

### DOMAIN 5 — THERMAL & STRUCTURAL COATINGS

#### 23. Thermal Barrier Coating (TBC)
**Scientific Context**: Ceramic coating on turbine blades insulating from hot combustion gases (enables higher temperatures → better efficiency). YSZ standard but degrades >1200°C. Next-gen: hexaaluminates, pyrochlores.

**Key Requirements**:
- Oxide (mandatory for thermal properties)
- Wide band gap >3.0 eV
- Known TBC families: zirconia, hexaaluminates, pyrochlores
- Very stable (formation energy <−3 eV/atom)

**Scoring Logic**:
- Returns 0.0 if no oxygen
- Band gap bonus:
  - +0.35 if ≥4.0 eV (excellent transparency)
  - +0.20 if 3.0–4.0 eV
  - +0.05 if <3.0 eV (suboptimal)
- +0.25 for zirconia family (Zr + O)
- +0.10 bonus for YSZ (Y + Zr + O, the benchmark)
- +0.30 for hexaaluminates (Al + A-site {Ba, Sr, La, Ce, Pr, Nd, Sm} + O)
- +0.25 for pyrochlores (A₂B₂O₇: A ∈ {La, Nd, Sm, Gd, Er, Yb, Y}, B ∈ {Zr, Ti, Hf, Ce, Sn})
- +0.10 for high-melting refractory metals (Hf, Ta, W, Nb, Mo + O)
- +0.10 for very stable formation energy (<−3 eV/atom)

**Commercial**: YSZ (yttria-stabilized zirconia), hexaluminates, pyrochlores

---

#### 24. Thermal Interface Material
**Scientific Context**: High κ (>10 W/m·K), electrically insulating material between heat source and sink. Power electronics packaging; especially critical in modern CPUs/GPUs.

**Key Requirements**:
- Band gap >2.5 eV (good insulation)
- High thermal conductivity (correlates with light atoms in strong bonds)
- Simple stoichiometry preferred (lower complexity = higher κ)

**Scoring Logic**:
- Returns 0.0 if band gap <2.5 eV (conducting = poor TIM)
- +0.35 if band gap 5.0–∞ eV (excellent insulation)
- +0.25 if band gap 3.5–5.0 eV
- +0.10 if band gap 2.5–3.5 eV
- Material family bonuses:
  - +0.50 for C (diamond, κ ~2000 W/m·K, ultimate)
  - +0.40 for B-N (BN, κ ~300 W/m·K)
  - +0.35 for Al-N (AlN, κ ~180 W/m·K, common in power electronics)
  - +0.25 for Si-C (SiC, κ ~150 W/m·K)
  - +0.20 for Be-O (BeO, κ ~270 W/m·K, highly toxic)
- Stoichiometry bonus:
  - +0.10 for binary, ternary structures (simpler = higher κ)
  - ×0.7 for complex structures (≥5 elements)
- Toxicity penalty: ×0.5 for BeO (highly toxic)

**Commercial**: AlN substrates, Cu-diamond composites, SiC

---

#### 25. Hard Coating / Wear Resistant
**Scientific Context**: Hard ceramic coatings on cutting tools, dies, engine parts. Commercial: TiN (~20 GPa), TiAlN (higher hardness), CrN, DLC. Key property: hardness >20 GPa Vickers.

**Key Requirements**:
- Transition metal nitrides, carbides, or borides (short, strong bonds)
- Wide band gap >1.5 eV (correlates with hardness via bond strength)
- Very stable thermodynamically

**Scoring Logic**:
- +0.45 for transition metal + hard anion (nitrides/carbides/borides)
  - Metals: {Ti, Cr, W, Mo, V, Nb, Ta, Zr, Hf, Al}
  - Anions: {N, C, B}
- +0.40 for pure carbon (diamond, ultimate hardness)
- +0.25 for Al₂O₃ (oxide coatings)
- +0.15 for ternary nitrides/carbides (TiAlN better than TiN)
- Band gap bonus:
  - +0.15 if ≥3.0 eV (very hard, strong bonding)
  - +0.08 if 1.5–3.0 eV
- +0.10 for very stable (<−2.5 eV/atom formation energy)

**Commercial**: TiN, TiAlN, CrN, DLC (diamond-like carbon), Al₂O₃ CVD coatings

---

#### 26. Corrosion Resistant Coating
**Scientific Context**: Prevents oxidation, acid attack, electrochemical corrosion. Passive oxide layers (Cr₂O₃ on stainless steel), fluorides, noble oxides. Must be chemically inert, stable.

**Key Requirements**:
- Thermodynamically very stable (formation energy <<−2 eV/atom)
- Passivating oxide formers or noble metals
- Wide band gap (>3.5 eV, electrical insulation = corrosion resistance)

**Scoring Logic**:
- +0.35 for passive metals in oxides (Cr, Al, Ti, Zr, Ta, Nb, Hf, Si, W + O)
- +0.30 for fluoride coatings (F + {Ca, Ba, Sr, Mg, La, Ce, Al})
- +0.20 for wide band gap (≥3.5 eV)
- Formation energy stability:
  - +0.20 if <−4.0 eV/atom (extremely stable)
  - +0.10 if <−2.5 eV/atom (very stable)
- +0.15 for noble metals (Pt, Pd, Au, Ir, Rh, Ru)

**Commercial**: Cr₂O₃ (stainless steel), Al₂O₃, CaF₂ and BaF₂ fluoride coatings

---

#### 27. Refractory / UHTC (Ultra-High Temperature Ceramic)
**Scientific Context**: Structural use above 1500°C. Turbine hot sections, hypersonic vehicles, nuclear components. Key families: HfC, ZrB₂, TaC, HfB₂ (UHTCs).

**Key Requirements**:
- Refractory transition metals (melting point >2000°C)
- Carbides, nitrides, borides (strong covalent bonding)
- NO volatile elements (Na, K, Zn, Cd, Hg, S, Se, Te)
- Very negative formation energy

**Scoring Logic**:
- +0.30 for refractory metals (W, Re, Os, Ta, Mo, Hf, Nb, Zr, V, Cr, Ti)
- +0.35 for UHTC materials: {Hf, Zr, Ta, Ti, Nb} + {C, B, N}
- Formation energy stability:
  - +0.20 if <−3.0 eV/atom
  - +0.10 if <−1.5 eV/atom
- ×0.4 penalty if volatile elements present (Na, K, Li, Rb, Cs, Zn, Cd, Hg, S, Se, Te)
- Band gap bonus: +0.15 if <0.5 eV (metallic) or >4.0 eV (both types exist)

**Commercial**: HfC, ZrB₂, TaC, W-Re alloys

---

### DOMAIN 6 — EMERGING & SPECIALISED

#### 28. Superconductor
**Scientific Context**: Zero resistance below critical temperature Tc. Applications: MRI (1.5–3.0T), particle accelerators (>10T), quantum computing. Conventional BCS (Nb, Nb₃Sn) need liquid He. High-Tc cuprates (YBCO) use liquid N₂. Your rare-earth hydrides appear in superconductor prediction literature.

**Key Requirements**:
- Metallic (band gap <0.5 eV or zero)
- Known SC element families: conventional metals, cuprates, iron-based, hydrides
- Must be metallic conductor (no band gap)

**Scoring Logic**:
- Returns 0.0 if band gap >0.5 eV (must be metallic)
- +0.20 for metallic character (essential)
- +0.25 for conventional BCS metals (Nb, V, Pb, Sn, In, Al, Mo, Re, W)
- +0.35 for cuprates: Cu + O + A-site {Ba, Sr, La, Y, Bi, Tl, Hg}
- +0.25 for iron-based: Fe + {As, Se, P}
- +0.20 for hydride superconductors: H + host {La, Y, Ce, Th, Ca, Ba, Lu}

**Known Examples**: NbTi (conventional, 10K), YBa₂Cu₃O₇ (high-Tc, 92K), FeAs compounds (iron-based, ~50K), LuH₃ under pressure (potential room-temperature)

---

#### 29. Radiation Detector / Scintillator
**Scientific Context**: Converts high-energy radiation (X, γ, neutrons) to detectable signal. Medical (CT, PET), security screening, nuclear monitoring. Commercial scintillators: BaI₂:Eu, CsI:Tl, LYSO (Lu₂SiO₅:Ce), NaI:Tl.

**Key Requirements**:
- High density (heavy elements for stopping power)
- Band gap 3.0–6.0 eV (scintillator) or <3.0 eV (semiconductor detector)
- Known scintillator families: alkali halides, oxides
- High-Z elements (Ba, I, Cs, Bi, Pb, Tl, Hg, W, Lu, Gd, Xe)

**Scoring Logic**:
- High-Z element bonus: +min(high_z_count × 0.15, 0.40) for {Ba, I, Cs, Bi, Pb, Tl, Hg, W, Lu, Gd, Xe}
- Band gap weighting:
  - +0.30 if 3.0–6.0 eV (scintillator window)
  - +0.10 if <3.0 eV (semiconductor detector type)
- +0.25 for alkali halides (Na/Cs/K/Ba/Sr + I/Br/Cl/F)
- +0.20 for oxide scintillators (O + {Lu, Gd, Bi, Y, Ce})
- +0.10 if Gd present (neutron detection)

**Examples**: NaI(Tl), CsI(Tl), BaI₂:Eu, LYSO, BGO, GSO

---

#### 30. Solid Oxide Fuel Cell (SOFC) Electrolyte
**Scientific Context**: Oxide ion conductor at 600–1000°C. Standard: YSZ (Y₂O₃-stabilized ZrO₂). Score component based on electrolyte requirements (ionic conduction, electronic insulation, structural stability).

**Key Requirements**:
- Oxide (mandatory)
- Wide band gap (>3.0 eV, electronic insulation)
- Contains O²⁻ as mobile ion
- Known SOFC electrolyte framework: Zr, La, Ce, Y, Sc, Bi, Gd
- Very stable thermodynamically

**Scoring Logic**:
- Returns 0.0 if no oxygen
- Band gap bonus:
  - +0.35 if ≥4.0
  - +0.25 if 3.0–4.0
  - +0.10 if <3.0 eV
- +0.30 for Zr-based (ZrO₂ framework)
- +0.15 bonus for Y-stabilized (YSZ, the benchmark)
- +0.25 for other SOFC framework elements (La, Ce, Sc, Gd, Bi)
- +0.10 for very stable (<−4.0 eV/atom)

**Commercial**: YSZ (dominant), ceria-gadolinia, scandia-stabilized zirconia (SSZ)

---

#### 31. Multiferroic
**Scientific Context**: Material exhibiting two or more ferroic properties simultaneously (ferromagnetism + ferroelectricity is most common). Enables magnetoelectric coupling and new device concepts.

**Key Requirements**:
- Must exhibit ferromagnetism (Fe, Co, Ni, REEs, Mn)
- Must be ferroelectric (non-centrosymmetric structure, polarization)
- Band gap >1.0 eV (ferromagnetic metals are competing requirement)

**Scoring Logic**:
- +0.25 for ferromagnetic elements (Fe, Co, Ni, REEs, Mn)
- Non-centrosymmetric bonus: +0.30 if polar space group detected
- Band gap consideration:
  - +0.20 if ferromagnetic + gap 1.0–3.5 eV (multiferroic window)
  - Penalty if purely metallic (ferromagnetism dominates)
- +0.20 for perovskite ABO₃ structure with Fe, Mn, Co
- +0.15 if both magnetic and dielectric order confirmed

**Examples**: BiFeO₃ (Bi:ferroelectric, Fe:ferromagnetic), TbMnO₃, hexagonal manganites

---

## Viability Scoring

Synthesized from real-world material scientists' expertise. Takes each application score and multiplies by:

$$\text{final\_score} = \text{application\_score} \times \text{viability\_multiplier}$$

### Components of Viability

#### 1. Material Cost Score
**Purpose**: Materials that are too expensive won't be commercialized regardless of performance.

$$\text{cost\_score} = \max(0, 1 - \frac{\text{weighted\_cost}}{120})$$

Where:
- Weighted cost = Σ(weight fraction × element price)
- Max assumed cost: $120/kg (adjustable)
- Price dictionary includes 30+ elements, defaults to $50/kg for unknowns

**Impact**: 0.0 (extremely expensive, >$120/kg) to 1.0 (cheap, <$10/kg)

**Examples**:
- Si: ~0.9 (abundant, $2.5/kg)
- Co: ~0.7 (expensive, $33/kg)
- Pt: ~0.0 (extremely expensive, $31,000/kg)

#### 2. Elemental Abundance Score
**Purpose**: Rare elements have supply volatility and extraction challenges.

$$\text{abundance\_score} = \frac{\log_{10}(\text{min\_abundance} + 0.001)}{\log_{10}(282000)} \text{ clipped to } [0, 1]$$

Where min_abundance is the Earth crustal abundance (ppm) of the least abundant element.

**Impact**: 
- Common elements (Si, Al, Fe): ~0.9
- Scarce elements (In, REEs): ~0.3
- Ultra-rare (Au, Pt): ~0.0

#### 3. Supply Risk Score
**Purpose**: Critical minerals (USGS definition) face geopolitical risks and supply constraints.

$$\text{supply\_risk} = 1 - \frac{\text{critical\_minerals\_count}}{\text{total\_elements}}$$

**Critical Minerals List** (50+ elements including):
- Light REEs: Nd, Pr, Sm, Gd, La, Ce, Y, Sc
- Heavy REEs: Dy, Tb, Er, Ho, Tm, Yb, Lu, Eu
- Specialty metals: Li, Co, Ni, Be, Ga, Ge, In, Hf, Nb, Ta, Mo, W, Re, Sb, Sn, Te

**Impact**:
- No critical minerals (e.g., Al₂O₃): 1.0
- One critical mineral (e.g., Li₂CO₃): 0.5
- Multiple critical minerals: <0.3

#### 4. Radioactive Filter (Hard Reject)
**Purpose**: Radioactive elements unacceptable for most applications.

**Hard-reject elements** (return viability = 0.0):
- Naturally radioactive: Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr
- Unstable: Tc, Po, At, Rn, Fr, Ra
- Heavy metals with Z>83: Bi(83) and higher

**Impact**: Binary filter (0.0 or passes)

#### 5. Rare-Earth Element Penalty
**Purpose**: REEs are expensive, geopolitically risky, but sometimes essential (e.g., permanent magnets). Different penalty tiers based on criticality.

**Hard REE Penalty** (×0.3 for each):
- Dy, Tb, Eu, Ho, Er, Tm, Lu, Yb
- Highest supply risk, lowest abundance

**Moderate REE Penalty** (×0.6 for each):
- Nd, Pr, Sm, Gd, Sc
- Supply risk but more abundant

**Mild REE Penalty** (×0.85 for each):
- La, Ce, Y
- Relatively abundant but strategic importance

**Penalty application**: Multiplicative. E.g., Nd₂Fe₁₄B has two Nd atoms → ×0.6 × 0.6 = ×0.36

**Context**: Permanent magnets legitimately need REEs for anisotropy. Cost is accepted in EV motor (~$1k motor cost, ~$50 magnet cost). For other applications, REE penalty is major blocker.

#### 6. CLscore Penalty (Synthesizability)
**Purpose**: A material might be theoretically perfect but practically impossible to synthesize.

$$\text{clscore\_multiplier} = \begin{cases}
1.0 & \text{if } \text{CLscore} \geq 0.5 \text{ (likely synthesizable)} \\
0.7 & \text{if } 0.3 \leq \text{CLscore} < 0.5 \text{ (moderately optimistic)} \\
0.4 & \text{if } 0.1 \leq \text{CLscore} < 0.3 \text{ (risky)} \\
0.1 & \text{if } \text{CLscore} < 0.1 \text{ (very unlikely)} \\
0.5 & \text{if } \text{CLscore} = -1 \text{ (unknown, penalized but not rejected)}
\end{cases}$$

### Viability Multiplier Calculation

```python
viability_multiplier = (
    cost_score 
    × abundance_score 
    × supply_risk_score 
    × radioactive_filter_multiplier 
    × rare_earth_multiplier 
    × clscore_multiplier
)
```

**Typical ranges**:
- Excellent (0.7–1.0): Abundant, cheap, non-critical, no REEs (e.g., Si₃N₄, SiC)
- Moderate (0.3–0.7): One or two constraints (e.g., Ni-based alloys, some REE-containing)
- Challenged (0.0–0.3): Multiple constraints (e.g., heavy REE permanent magnets, Pt-based)
- Rejected (0.0): Radioactive or Z>83

---

## CLscore: Synthesizability Prediction

### What is CLscore?

CLscore is a **synthesizability probability** (0.0–1.0) predicting how likely a material is to be successfully synthesized given its crystal structure. Developed by KAIST (Korea Advanced Institute of Science and Technology) using the Synthesizability-PU-CGCNN (Positive Unlabeled learning with Crystal Graph Convolutional Neural Networks).

**Key insight**: Many computationally stable materials (on convex hull or near it) are actually very hard to synthesize due to kinetic barriers, intermediate phase competition, or extreme conditions needed.

### How CLscore is Computed

#### Step 1: Load CIF File
- User provides material ID (e.g., `000006a8c4`) and corresponding CIF crystal structure file
- PyMatGen parses the atomic coordinates, lattice parameters, and periodic boundary conditions
- Input: 3D atomic structure

#### Step 2: Build Crystal Graph
- For each atom, identify neighbors within 8.0 Å (configurable `radius`)
- Limit to 12 nearest neighbors (configurable `max_neighbors`) to avoid distance-based errors
- Edges represent chemical bonding/interactions

**Graph structure**:
- **Nodes**: Atoms with 92-dimensional one-hot features (element embedding from pre-trained checkpoint)
- **Edges**: Neighbor distances + atom pair features (up to 41 dimensions from checkpoint)

#### Step 3: Pass Through Graph Neural Network
The CrystalGraphConvNet architecture:
1. **Embedding layer**: 92-dim one-hot → 64-dim learned embeddings
2. **Convolutional layers** (3 stacks by default):
   - Aggregate neighbor features to central atoms
   - Pass through FC layers (atom_hidden dimension)
   - Residual connections
3. **Pooling**: Global mean/sum over nodes (material-level summary)
4. **FC layers** (h_fea_len = 128 hidden, then 1 output)
5. **Classification**: 2-class softmax → probability of "synthesizable"

#### Step 4: Ensemble Voting
- Run prediction through 100 trained checkpoint bags (bagging for robustness)
- Average probabilities across all checkpoints
- Final CLscore = mean probability

**Speed optimization**: For full dataset, use `--max-models 1` (single checkpoint) for speed, then refine top candidates with full ensemble.

### Model Training Data

KAIST trained on **GNoME** and **other high-throughput databases**:
- Materials that have been reported in literature (positive class) → label 1
- Computationally stable but unreported (unlabeled, some are synthesizable, some aren't) → positive-unlabeled learning

### Interpretation

- **CLscore ≥ 0.5**: "Likely synthesizable" — consider for experimental validation
- **CLscore 0.3–0.5**: "Moderately optimistic" — may require creative synthesis strategies
- **CLscore 0.1–0.3**: "Risky" — significant synthesis challenges but not impossible
- **CLscore < 0.1**: "Very unlikely" — would require extreme conditions or novel techniques
- **CLscore = −1.0**: "Unknown" — CIF file missing or parsing failed

### Practical Application in MatIntel

1. **Batch scoring**: Run `run_clscore_all.py` to compute CLscores for all 554,054 GNoME materials
2. **Top-N scoring**: Run `run_clscore.py --app [category] --top-n 1000` to refine top candidates with full ensemble
3. **Standalone cache**: Results stored in `data/processed/clscore_all_results.csv` for quick lookup
4. **Viability integration**: CLscore multiplies final scores — materials with low synthesizability are downranked

### Known Limitations

1. **CIF availability**: CLscore can't run without crystal structure. GNoME provides this, but gaps exist.
2. **Computational structures only**: Trained on DFT-relaxed structures; real synthesized structures may differ slightly.
3. **No kinetic pathway**: Predicts final-state likelihood, not synthesis route difficulty.
4. **No extreme conditions**: Doesn't account for hydrothermal, high-pressure, or high-temperature synthesis advantages.
5. **No batch chemistry**: Assumes solid-state synthesis; sol-gel or other wet methods unmodeled.

---

## Interactive Streamlit Dashboard

The dashboard provides a browser-based interface for exploration and filtering.

### User Interface

#### Sidebar Controls

1. **Target Application** (dropdown): Select from 31 categories or predefined shortcuts
2. **Min Application Score** (slider): 0.0–1.0, default 0.45
3. **Min Viability** (slider): 0.0–1.0, default 0.30
4. **Max Band Gap** (slider): 0.0–8.0 eV, default 4.0
5. **Min Supply Risk** (slider): 0.0–1.0, default 0.20
6. **Min CLscore** (slider): 0.0–1.0, default 0.0
7. **Compound Provenance** (dropdown): All / Experimental / Synthesized

#### Results Display

- **Metric Cards**: Count of results, mean application score, mean CLscore, mean viability
- **Results Table**: Sortable columns including:
  - MaterialId
  - Reduced Formula
  - Application Score (color-coded: green ≥0.5, yellow ≥0.3, red <0.3)
  - CLscore (color-coded: green ≥0.5, yellow ≥0.3, red <0.3, gray unknown)
  - Viability
  - Bandgap
  - Formation Energy
  - Source
  - is_experimental
  - Best Score (max across all 31 categories)

#### Color Coding

**CLscore Column**:
- Green (#cfeede): CLscore ≥ 0.5 (high confidence)
- Yellow (#f8efbe): 0.3 ≤ CLscore < 0.5 (moderate)
- Red (#f8d2d2): CLscore < 0.3 (low confidence)
- Gray (#f3efe5): CLscore = −1 (unknown/not computed)

**Application Score Column**: Similar color scheme

### Features

1. **Dynamic Category Selection**: Select any of 31 applications and see top candidates
2. **Multi-filter Ranking**: Results ranked by (application score × viability), sorted descending
3. **Provenance Split**: Toggle between computational base set and synthesized reference set
4. **Export Bundle**: Download ranked CSV, matching CIF archive, and PDF summary
5. **MP Lookup**: Optional Materials Project formula existence check via API key
6. **Structure Viewer**: Interactive CIF visualization for selected candidates
7. **AI Summaries**: (if explanations module enabled) Short text describing why top material is good
8. **Responsive Design**: Works on desktop, tablet, mobile

### Running the Dashboard

```powershell
./scripts/run_app.ps1
```

Opens in browser at http://localhost:8501

Manual equivalent:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

---

## Pipeline Architecture

### Data Flow

```
Raw Input CSV (GNoME or demo)
  ↓ [data_sources.ensure_demo_dataset()]
Working Dataset (MaterialId, Reduced Formula, properties)
  ↓ [features.featurize()]
Featured Dataset (+ 138 matminer features)
  ↓ [scoring.apply_application_scores()]
Scored Dataset (+ 31 score columns)
  ↓ [viability.apply_viability()]
Final Dataset (+ cost, abundance, supply_risk, viability, best_score)
  ↓ [clscore.batch_clscore()]
CLscore Cache (standalone file)
  ↓ [Merge back into featured_dataset]
Streamlit Dashboard
```

### Key Modules

#### config.py
- **APP_LABELS**: Readable names → internal score column mappings (31 applications)
- **Element prices**: USD/kg for 30+ elements
- **Critical minerals list**: USGS supply risk elements
- **Directory constants**: RAW_DIR, PROCESSED_DIR, LOG_DIR

#### data_sources.py
- **ensure_demo_dataset()**: Falls back to realistic demo if real CSV missing
- **load_raw()**: Load and validate GNoME CSV, handle missing columns

#### features.py
- **featurize()**: Apply matminer ElementProperty extractors
  - Atomic numbers, radii, electronegativities
  - Polarizabilities, valences, d-block character
  - 138 total features across composition and structure

#### scoring.py
- **31 scoring functions** (score_battery_cathode_liion, score_solar_singlejunction, etc.)
- **SCORING_FUNCTIONS**: Registry dict mapping function names
- **apply_application_scores()**: Iterate registry, generate columns for each app

#### viability.py
- **material_cost_score()**: Based on weighted element prices
- **abundance_score()**: Log-transform element crustal abundance
- **supply_risk_score()**: Critical minerals count penalty
- **viability_filter_multiplier()**: Radioactive rejection, REE penalties
- **clscore_penalty()**: Map CLscore to multiplier
- **apply_viability()**: Generate viability columns, multiply final scores

#### clscore.py
- **CLscorePredictor**: Singleton class managing model loading and inference
- **batch_clscore()**: Resume-safe batch processing with checkpointing
- **predict()**: Single-material inference
- Architecture shape derivation from checkpoint

#### pipeline.py
- **run_pipeline()**: Orchestrate load → featurize → score → viability
- **_build_logger()**: File + console logging

#### app.py
- **Streamlit UI**: Sidebar filters, results table, color-coded display

---

## Installation & Setup

### Prerequisites

- **Python 3.10+** (3.13 tested)
- **Windows PowerShell 5.1+** (native OS)
- **~5 GB disk** (GNoME CIFs + pip packages)

### Step 1: Clone and Activate

```powershell
cd c:\Users\rosha\Downloads\MatIntel
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

**Key packages**:
- pandas, numpy: Data manipulation
- pymatgen, matminer: Materials science
- torch, scikit-learn: CLscore neural network
- streamlit: Dashboard
- plotly: Visualization

### Step 3: Download GNoME Dataset (Optional)

```powershell
./scripts/download_gnome_csv.ps1  # ~500 MB, all materials metadata
```

If skipped, pipeline auto-generates demo dataset (10k materials).

### Step 4: Setup CLscore (Optional)

```powershell
./scripts/setup_clscore.ps1
```

This:
1. Clones KAIST Synthesizability-PU-CGCNN repo into `external/`
2. Installs torch and scikit-learn
3. Verifies trained checkpoints load on CPU

### Step 5: Download CIFs (Optional, Large)

```powershell
./scripts/download_gnome_cifs.ps1  # 455 MB, all 554k CIF files
```

---

## Running the System

### Full Pipeline (Data → Scores → Dashboard)

```powershell
# Option 1: Use provided PowerShell scripts
./scripts/run_pipeline.ps1       # Generates scored_dataset.csv
./scripts/run_app.ps1              # Launches Streamlit dashboard
```

Or manually:

```powershell
# Option 2: Manual Python commands
.\.venv\Scripts\python.exe -m src.matintel.pipeline

# Launch dashboard
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**Output**: `data/processed/scored_dataset.csv` with 31 score columns, viability, best_score

### Score Top Candidates for a Single Application

```powershell
# Score top 1000 battery cathode candidates with CLscore
.\.venv\Scripts\python.exe run_clscore.py `
  --app battery_cathode_liion `
  --top-n 1000 `
  --cif-dir data/cifs `
  --batch-size 100
```

**Output**: `data/processed/clscore_results.csv` with CLscores merged

### Score All Materials (Full Dataset)

```powershell
# Compute CLscores for all 554k materials (resumable)
.\.venv\Scripts\python.exe run_clscore_all.py `
  --output-csv data/processed/clscore_all_results.csv `
  --batch-size 2000 `
  --max-models 1                 # Speed mode: 1 checkpoint
```

**Speed modes**:
- `--max-models 1`: Single checkpoint, ~6–8 hours for 554k materials
- `--max-models 100` (default): Full ensemble, ~27 days (not recommended)

Resume if interrupted:
```powershell
# Rerun same command; automatically resumes from checkpoint
.\.venv\Scripts\python.exe run_clscore_all.py ...
```

### Build Experimental/Synthesized Reference Dataset

```powershell
.\.venv\Scripts\python.exe scripts/build_experimental_reference.py
```

Environment keys supported:
- `MATINTEL_MP_API_KEY`
- `MP_API_KEY`

Output:
- `data/processed/experimental_compounds.csv`

This dataset is automatically appended in `app.py` at load time and scored/viability-processed if score columns are missing.

### Recompute Unknown CLscores

```powershell
# Re-run only materials with CLscore = -1
.\.venv\Scripts\python.exe run_clscore_all.py `
  --recompute-unknown
```

---

## File Outputs & Data Structures

### CSVs Generated

#### data/processed/working_dataset.csv
- Raw data post-load, pre-feature
- Columns: MaterialId, Reduced Formula, Bandgap, Formation Energy, etc.

#### data/processed/featured_dataset.csv
- Post-featurization (matminer 138 features)
- Columns: (working columns) + elemental properties

#### data/processed/scored_dataset.csv
- **Main output**: Post-scoring and viability
- Columns include:
  - MaterialId, Reduced Formula
  - 31 score_* columns (one per application)
  - cost, abundance, supply_risk, viability, clscore
  - best_score (max of 31 scores × viability)
  - Band gap, Formation Energy, etc.
- **Rows**: All available materials (554,054 from GNoME or 10k demo)

#### data/processed/clscore_all_results.csv
- Standalone CLscore cache
- Columns: MaterialId, clscore
- **Resumable**: Appends new results on each run, deduplicates on close

#### data/processed/experimental_compounds.csv
- Experimental/synthesized reference dataset generated from MP/JARVIS pipeline
- Typical columns: MaterialId, Reduced Formula, Bandgap, source, is_experimental, thermodynamic fields
- Used by app loader to append additional rows to `scored_dataset.csv`
- If score columns are missing, app computes 31 category scores + viability for this file on load

#### data/processed/top10_per_category.csv
- Top 10 per application category ranked by `(application score × viability)`
- Includes all 31 categories for rapid shortlist review

#### data/processed/top10_per_category_raw_score.csv
- Top 10 per application category ranked by raw application score (without viability multiplier)
- Useful for separating technical fit from practical constraints

#### logs/pipeline.log
- Timestamped log of pipeline execution
- Rows logged: Load count, feature count, score count, timing

### Data Structure Example

```
MaterialId | Reduced Formula | Bandgap | score_battery_cathode_liion | score_solar_singlejunction | ... | viability | clscore | best_score
-----------|----|---------|----------|-----------|---|-----------|---------|------
000006a8c4 | Ca2SiMoO6 | 1.23 | 0.52 | 0.18 | ... | 0.68 | 0.5795 | 0.353
00006650ea | Li2Mn2O4 | 2.14 | 0.78 | 0.05 | ... | 0.82 | -1.0 | 0.64
```

---

## Advanced Usage

### Custom Element Prices

Edit `src/matintel/config.py`:

```python
CUSTOM_PRICES = {
    "Li": 8.0,      # Update Li price ($/kg)
    "Co": 40.0,
    "NewElement": 100.0
}
```

Then rerun pipeline with custom prices registered in cost scorer.

### Filtering by Multiple Criteria

In dashboard (sidebar):
1. **Application**: Battery Cathode (Li-ion)
2. **Min Score**: 0.50
3. **Min Viability**: 0.60
4. **Max Band Gap**: 2.0 eV
5. **Min Supply Risk**: 0.50
6. **Min CLscore**: 0.30

Results: Materials meeting ALL criteria, sorted by (score × viability)

### Exporting Results

From Streamlit:
- Select rows → right-click → copy table
- Or save CSV from dashboard

Or programmatically:

```python
import pandas as pd
df = pd.read_csv("data/processed/scored_dataset.csv")
battery_candidates = df[
    (df["score_battery_cathode_liion"] >= 0.50) &
    (df["viability"] >= 0.60) &
    (df["clscore"] >= 0.30)
].sort_values("score_battery_cathode_liion", ascending=False)
battery_candidates.to_csv("my_candidates.csv", index=False)
```

### Adding a New Scoring Category

1. **Define scoring function** in `src/matintel/scoring.py`:

```python
def score_my_new_app(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}
    
    # Your logic here
    if some_condition:
        score += 0.30
    
    return min(score, 1.0)
```

2. **Register in SCORING_FUNCTIONS dict**:

```python
SCORING_FUNCTIONS = {
    ...
    "score_my_new_app": score_my_new_app,
}
```

3. **Add to APP_LABELS** in `config.py`:

```python
APP_LABELS = {
    ...
    "My New App": "score_my_new_app",
}
```

4. **Rerun pipeline**:

```powershell
.\.venv\Scripts\python.exe -m src.matintel.pipeline
```

New score column automatically generated.

### Running on Subset of Data

```python
import pandas as pd

df = pd.read_csv("data/processed/scored_dataset.csv")
subset = df.query("Bandgap > 2.0 and Bandgap < 3.0")

# Then filter/process subset
```

---

## Technical Implementation Details

### Architecture Decisions

1. **Resume-safe batch processing**:
   - CLscore writes every N materials to checkpoint file
   - On rerun, reads existing file and skips completed rows
   - Final flush deduplicates before returning

2. **Viability as multiplier (not additive)**:
   - Application score × viability captures the idea that a perfect material is worthless if it costs $100k/kg
   - Multiplicative ensures that 0 in any component → 0 overall

3. **Element-level metadata** (cost, abundance, critical):
   - Lookup-based, not computed on-the-fly
   - 30+ elements manually curated with prices
   - CRITICAL_MINERALS set closely follows USGS 2023 list

4. **CLscore ensemble averaging**:
   - 100 bagging checkpoints reduce overfitting
   - Speed mode (1 checkpoint) for broad coverage, full ensemble for precision

5. **Streamlit caching**:
   - Dashboard loads CSV once, caches in memory
   - Interactive filtering is instant (<100ms)
   - On file update, cache invalidated

### Performance Characteristics

| Operation | Count | Time |
|-----------|-------|------|
| Load CSV | 554k | ~2 sec |
| Featurize | 554k | ~45 min (matminer) |
| Score (31 apps) | 554k | ~8 min |
| Viability | 554k | ~3 min |
| CLscore (1 model) | 554k | 6–8 hours |
| CLscore (100 models) | 554k | 25+ days |

### Checkpoint Format

CLscore uses PyTorch `torch.load()` with `weights_only=False` to load legacy checkpoints from KAIST (pre-2024 PyTorch versions). Modern security best practice would be `weights_only=True`, but compatibility with external checkpoints requires legacy mode.

### Error Handling

1. **Missing CIF**: Returns CLscore = -1.0, status = error message
2. **Parsing failure**: Returns -1.0, logs exception
3. **Model loading**: Catches torch.load errors, falls back to error message
4. **Formula parsing**: Pymatgen handles errors, returns 0.5 viability (conservative)
5. **Missing columns**: Pipeline auto-generates with sensible defaults
6. **CSV mixed dtypes in experimental file**: Pandas may emit a `DtypeWarning` for mixed-type columns during `read_csv`; loading remains functional

### Memory Usage

- **Full dataset in memory**: ~2–3 GB (554k rows × ~150 columns)
- **Model (all 100 checkpoints)**: ~500 MB on disk, ~200 MB peak RAM during inference
- **Streamlit user session**: ~1 GB (dataset + filter state)

---

## Summary

MatIntel solves materials discovery through:
1. **31 scientifically-grounded domain scorers** evaluating materials for specific applications
2. **Viability multiplier** incorporating cost, abundance, supply risk, and synthesizability
3. **Deep learning synthesizability prediction** (CLscore from KAIST) via crystal graph neural networks
4. **Interactive Streamlit dashboard** for exploration and ranking
5. **Resume-safe batch processing** for massive datasets

The system is extensible, Windows-native, and production-ready for computational materials screening workflows.

---

## Recent Updates (March 2026)

1. **CLscore integration hardening**
- Fixed model-loading path issues and completed full-dataset CLscore processing
- Dashboard now hydrates CLscore from `clscore_all_results.csv` if `scored_dataset.csv` lacks `clscore`

2. **Scoring-quality upgrades (Tier-1 subset implemented)**
- Updated category logic for:
  - Tandem solar top-cell (`score_solar_absorber_tandem`)
  - Superconductor specificity (`score_superconductor`)
  - Solid electrolyte fluoride-dominant penalty (`score_solid_electrolyte`)
  - HER/OER disambiguation (`score_her_electrocatalyst`, `score_oer_electrocatalyst`)
  - CO2 reduction selectivity shaping (`score_co2_reduction`)

3. **Experimental-reference pipeline shipped**
- Added `scripts/build_experimental_reference.py` with MP API compatibility updates
- Produced `data/processed/experimental_compounds.csv` and integrated automatic app ingestion

4. **Provenance filter and semantics finalized**
- Sidebar provenance selector added
- Current mapping in app:
  - Base scored rows (historically unlabeled) default to `Experimental`
  - `MP_synthesized` and `JARVIS_ICSD` rows are treated as `Synthesized`

5. **Export and analyst workflow improvements**
- Dashboard export bundle includes CSV, CIF ZIP, and PDF summary
- Added top-10-per-category exports for both viability-adjusted and raw-score ranking

---

## Top-10 Provenance Findings (March 2026 Snapshot)

This section summarizes a direct analysis of:
- `data/processed/top10_per_category_raw_score.csv` (raw application ranking)
- `data/processed/top10_per_category.csv` (viability-adjusted ranking)

### Headline Metrics

- **31 categories x 10 entries = 310 rows** in each export
- **Experimental share (raw top-10)**: **29.03%**
- **Experimental share (viability-adjusted top-10)**: **25.81%**
- Viability weighting shifts the frontier slightly toward synthesized references overall

### Where Experimental Excels vs Synthesized (Viability-Adjusted Top-10)

Using mean `weighted_score` within each category's viability-adjusted top-10:

- **Experimental advantage (7 categories)**:
  - Ferroelectric
  - Hard Coating / Wear Resistant
  - Hydrogen Storage
  - Refractory / UHTC
  - Soft Magnet
  - Solar Absorber - Single Junction
  - Thermal Barrier Coating

- **Synthesized advantage (15 categories)**:
  - Battery Anode
  - CO2 Reduction Catalyst
  - Corrosion Resistant Coating
  - HER Electrocatalyst (Green Hydrogen)
  - LED / Light Emitter
  - Multiferroic
  - Photodetector
  - Radiation Detector / Scintillator
  - Semiconductor (General)
  - Solar Absorber - Tandem Top Cell
  - Solid Oxide Fuel Cell Electrolyte
  - Thermal Interface Material
  - Thermoelectric
  - Topological Insulator
  - Transparent Conductor

- **Synthesized-only top-10 (9 categories, 0 experimental in weighted top-10)**:
  - Battery Cathode (Li-ion)
  - Battery Cathode (Na-ion)
  - Magnetic Semiconductor / Spintronics
  - OER Electrocatalyst (Water Splitting)
  - Permanent Magnet
  - Photocatalyst (Water Splitting)
  - Piezoelectric
  - Solid Electrolyte
  - Superconductor

### Largest Provenance Shifts After Viability Weighting

- **Experimental share increase**:
  - Hard Coating / Wear Resistant: **30% -> 90%**

- **Experimental share decreases**:
  - Superconductor: **50% -> 0%**
  - Hydrogen Storage: **90% -> 50%**
  - Solar Absorber - Single Junction: **100% -> 70%**
  - Permanent Magnet: **20% -> 0%**
  - Multiferroic: **70% -> 50%**

### Experimental Presence by Category (Raw vs Viability-Adjusted Top-10)

| Category | Experimental % (raw top-10) | Experimental % (weighted top-10) | Weighted advantage |
|---|---:|---:|---|
| Battery Anode | 40.0% | 40.0% | Synthesized |
| Battery Cathode (Li-ion) | 0.0% | 0.0% | Synthesized-only |
| Battery Cathode (Na-ion) | 0.0% | 0.0% | Synthesized-only |
| CO2 Reduction Catalyst | 50.0% | 50.0% | Synthesized |
| Corrosion Resistant Coating | 10.0% | 10.0% | Synthesized |
| Ferroelectric | 10.0% | 10.0% | Experimental |
| HER Electrocatalyst (Green Hydrogen) | 60.0% | 60.0% | Synthesized |
| Hard Coating / Wear Resistant | 30.0% | 90.0% | Experimental |
| Hydrogen Storage | 90.0% | 50.0% | Experimental |
| LED / Light Emitter | 10.0% | 10.0% | Synthesized |
| Magnetic Semiconductor / Spintronics | 0.0% | 0.0% | Synthesized-only |
| Multiferroic | 70.0% | 50.0% | Synthesized |
| OER Electrocatalyst (Water Splitting) | 0.0% | 0.0% | Synthesized-only |
| Permanent Magnet | 20.0% | 0.0% | Synthesized-only |
| Photocatalyst (Water Splitting) | 0.0% | 0.0% | Synthesized-only |
| Photodetector | 20.0% | 20.0% | Synthesized |
| Piezoelectric | 0.0% | 0.0% | Synthesized-only |
| Radiation Detector / Scintillator | 60.0% | 60.0% | Synthesized |
| Refractory / UHTC | 30.0% | 30.0% | Experimental |
| Semiconductor (General) | 10.0% | 10.0% | Synthesized |
| Soft Magnet | 10.0% | 10.0% | Experimental |
| Solar Absorber - Single Junction | 100.0% | 70.0% | Experimental |
| Solar Absorber - Tandem Top Cell | 20.0% | 20.0% | Synthesized |
| Solid Electrolyte | 0.0% | 0.0% | Synthesized-only |
| Solid Oxide Fuel Cell Electrolyte | 40.0% | 40.0% | Synthesized |
| Superconductor | 50.0% | 0.0% | Synthesized-only |
| Thermal Barrier Coating | 20.0% | 20.0% | Experimental |
| Thermal Interface Material | 20.0% | 20.0% | Synthesized |
| Thermoelectric | 50.0% | 50.0% | Synthesized |
| Topological Insulator | 70.0% | 70.0% | Synthesized |
| Transparent Conductor | 10.0% | 10.0% | Synthesized |

### Analyst Takeaway

Experimental candidates are still highly competitive in several categories, but viability weighting (cost, abundance, supply risk, and CLscore) generally favors synthesized references in more categories than not. The strongest experimental resilience appears in wear-resistant coatings and single-junction solar absorbers, while batteries (especially cathodes), solid electrolytes, and several catalytic categories remain synthesized-dominant under current viability constraints.

---

## References & Acknowledgments

- **GNoME Dataset**: Google DeepMind stable materials database
- **CLscore Model**: KAIST Synthesizability-PU-CGCNN (Juhwan Noh et al.)
- **Matminer**: Computational materials analysis library
- **Pymatgen**: Materials informatics platform
- **Streamlit**: Interactive app framework

