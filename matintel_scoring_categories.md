# MatIntel Scoring Categories — Full Implementation Guide

This document defines all 31 application scoring categories for MatIntel. Each entry covers the
scientific rationale, the scoring criteria as Python logic, element/structural rules, viability
modifiers, known caveats, and example compounds from the GNoME dataset.

---

## How Scores Are Structured

Every category function receives:
- `composition`: `pymatgen.Composition` object
- `structure`: `pymatgen.Structure` object (or `None` if unavailable)
- `band_gap`: float in eV (from CHGNet or matminer)
- `formation_energy`: float in eV/atom
- `e_hull`: float in eV/atom (energy above convex hull from GNoME CSV)
- `features`: dict of matminer ElementProperty features

Every function returns a float **0.0–1.0**.

Scores are then multiplied by the viability score (which incorporates element cost, abundance,
radioactive filter, rare earth penalty, CLscore penalty, and e_hull penalty separately).

---

## DOMAIN 1 — ENERGY STORAGE

### 1. Battery Cathode (Li-ion)

**What it is:** Positive electrode in lithium-ion batteries. Must intercalate Li⁺ reversibly at
high voltage. Commercially dominant examples: LiCoO₂, LiFePO₄, NMC.

**Key requirements:**
- Must contain Li in the structure (working ion present)
- Transition metal redox center: Mn, Fe, Co, Ni, V, Cr (change oxidation state reversibly)
- Band gap: 0 (metallic) to ~3 eV acceptable; fully insulating hurts rate performance
- Polyanionic framework (phosphate, silicate, sulfate) or layered oxide preferred for stability
- Avoid: purely metallic conductors with no intercalation sites

```python
def score_battery_cathode_liion(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Hard requirement: must contain Li
    if 'Li' not in elements:
        return 0.0

    # Redox-active transition metal present
    redox_metals = {'Mn', 'Fe', 'Co', 'Ni', 'V', 'Cr', 'Cu', 'Mo', 'Ti'}
    if elements & redox_metals:
        score += 0.35

    # Polyanionic framework bonus (P, Si, S, B as framework formers with O)
    polyanion_formers = {'P', 'Si', 'S', 'B'}
    if (elements & polyanion_formers) and ('O' in elements):
        score += 0.20

    # Layered oxide bonus (many cathodes are AMO2 type)
    if 'O' in elements and not (elements & polyanion_formers):
        score += 0.10

    # Band gap: conductivity needed but not fully metallic
    if band_gap is not None:
        if 0.01 <= band_gap <= 3.0:
            score += 0.25
        elif band_gap < 0.01:  # too metallic — might not have intercalation sites
            score += 0.10

    # Formation energy: should be strongly negative (stable)
    if formation_energy is not None and formation_energy < -1.0:
        score += 0.10

    # Penalise fluorides (stability issues in electrolyte)
    if 'F' in elements:
        score *= 0.7

    # Penalise sulfur without phosphorus (polysulfide dissolution)
    if 'S' in elements and 'P' not in elements and 'O' not in elements:
        score *= 0.6

    return min(score, 1.0)
```

**Viability notes:** Co is a critical mineral (~$33/kg, high supply risk). Ni is moderate.
Fe and Mn are excellent from a supply perspective — LFP (LiFePO₄) is the fastest-growing
cathode chemistry for this reason.

**Example targets from GNoME:** NaCa(Cu9Si4)₂ after Na-ion adaptation; any Li-Mn-P-O compound.

---

### 2. Battery Anode

**What it is:** Negative electrode in Li/Na-ion batteries. Stores ions at low voltage vs.
Li/Li⁺. Commercial: graphite (372 mAh/g), silicon (~3500 mAh/g but volume expansion problem).
Fe-Si alloys suppress Si expansion; iron silicides (FeSi₂) are studied as inactive matrix.

**Key requirements:**
- No working ion required in structure (anode inserts Li/Na from electrolyte)
- Metallic or near-metallic (band gap < 0.5 eV preferred for electronic conductivity)
- Contains Si, Sn, Sb, P, Ge (alloying anodes) OR conversion-type (Fe, Co, Ni oxides/sulfides)
- High theoretical capacity elements preferred
- Must NOT be purely an oxide without a metal — pure oxides are conversion type only

```python
def score_battery_anode(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # High-capacity alloying anode elements
    alloying = {'Si', 'Sn', 'Sb', 'P', 'Ge', 'Bi', 'Al'}
    if elements & alloying:
        score += 0.40

    # Conversion anode elements (transition metal + anion)
    conversion_metals = {'Fe', 'Co', 'Ni', 'Cu', 'Mn', 'Mo', 'W'}
    conversion_anions = {'O', 'S', 'Se', 'F', 'N'}
    if (elements & conversion_metals) and (elements & conversion_anions):
        score += 0.25

    # Metallic character preferred
    if band_gap is not None:
        if band_gap < 0.1:
            score += 0.20
        elif band_gap < 0.5:
            score += 0.12
        elif band_gap < 1.5:
            score += 0.05

    # Iron silicide specific bonus (well-studied anode family)
    if 'Fe' in elements and 'Si' in elements:
        score += 0.15

    # Penalise if Li or Na already in structure
    # (anode materials don't contain the working ion in their discharged state)
    if 'Li' in elements or 'Na' in elements:
        score *= 0.5

    return min(score, 1.0)
```

**Viability notes:** Si is abundant and cheap. Fe is excellent. Sn is moderate cost. Sb and Ge
carry supply risk. Avoid Co-heavy anodes.

**Example targets from GNoME:** Fe5Si3H (if the H variant is stable), Fe8Si15P, Cu30Si7P.

---

### 3. Battery Cathode (Na-ion)

**What it is:** Positive electrode specifically for sodium-ion batteries. Na-ion is commercially
emerging (CATL producing since 2023). Key difference from Li-ion: Na⁺ is larger (1.02 Å vs
0.76 Å for Li⁺), requiring a more open framework. Prussian blue analogues, layered oxides,
and polyanionic compounds are the main families.

**Key requirements:**
- Must contain Na in the structure
- Transition metal redox center (same as Li-ion cathode)
- More open framework preferred (larger interlayer spacing than Li-ion cathodes)
- Mn and Fe strongly preferred (abundant, Na-ion's cost advantage relies on cheap elements)
- Prussian blue analogue structure bonus (known excellent Na-ion cathode family)

```python
def score_battery_cathode_naion(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Hard requirement: must contain Na
    if 'Na' not in elements:
        return 0.0

    # Preferred redox metals for Na-ion (Fe and Mn especially)
    preferred_redox = {'Fe', 'Mn'}
    other_redox = {'Co', 'Ni', 'V', 'Cr', 'Cu', 'Ti'}
    if elements & preferred_redox:
        score += 0.40
    elif elements & other_redox:
        score += 0.25

    # Polyanionic framework (very common in Na-ion cathodes)
    polyanion_formers = {'P', 'Si', 'S', 'B'}
    if (elements & polyanion_formers) and ('O' in elements):
        score += 0.20

    # Oxide layered structure (P2 and O3 type)
    if 'O' in elements and not (elements & polyanion_formers):
        score += 0.12

    # Band gap
    if band_gap is not None:
        if 0.0 <= band_gap <= 3.0:
            score += 0.20

    # Bonus: no Li (pure Na-ion, not mixed)
    if 'Li' not in elements:
        score += 0.08

    return min(score, 1.0)
```

**Viability notes:** Na-ion's commercial case rests on avoiding Li, Co, and expensive REEs.
Mn and Fe are ideal. NaCa(Cu9Si4)₂ from your results is a candidate worth investigating.

---

### 4. Solid Electrolyte

**What it is:** Solid-state ionic conductor replacing liquid electrolyte in all-solid-state
batteries. Must conduct the working ion (Li⁺ or Na⁺) while being electronically insulating.
Key families: NASICON (Na₃Zr₂Si₂PO₁₂ type), garnet (Li₇La₃Zr₂O₁₂), LLZO, argyrodite (Li₆PS₅Cl).

**Key requirements:**
- Wide band gap (>3 eV) — must be electronically insulating
- Contains Li or Na (working ion)
- Contains Zr, La, Al, P, Si, S as framework elements (common in known solid electrolytes)
- Sulfide-based: lower band gap acceptable (2–4 eV), generally higher conductivity
- Oxide-based: higher band gap (>4 eV), better stability vs Li metal
- Halide-based: Cl, Br, I with Li/Na, very active research area

```python
def score_solid_electrolyte(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Hard requirement: Li or Na (or both)
    if 'Li' not in elements and 'Na' not in elements:
        return 0.0

    # Must be electronically insulating
    if band_gap is not None:
        if band_gap >= 4.0:
            score += 0.35
        elif band_gap >= 3.0:
            score += 0.25
        elif band_gap >= 2.0:
            score += 0.10
        else:
            return 0.0  # too conducting electronically

    # Known solid electrolyte framework elements
    oxide_framework = {'Zr', 'La', 'Al', 'Ta', 'Nb', 'Ti'}
    sulfide_framework = {'P', 'Si', 'Ge', 'Sn', 'As'}
    halide_elements = {'Cl', 'Br', 'I', 'F'}
    if elements & oxide_framework:
        score += 0.25
    if elements & sulfide_framework and 'S' in elements:
        score += 0.20
    if elements & halide_elements:
        score += 0.20

    # Oxygen presence (most known solid electrolytes are oxides)
    if 'O' in elements:
        score += 0.10

    # Formation energy: must be very stable
    if formation_energy is not None and formation_energy < -2.0:
        score += 0.10

    return min(score, 1.0)
```

---

### 5. Hydrogen Storage

**What it is:** Materials that absorb and release hydrogen reversibly for fuel cell vehicle
storage. Target: >6.5 wt% gravimetric capacity, release at 60–120°C. Known families: complex
hydrides (NaBH₄, LiAlH₄ type), metal hydrides (FeTiH₂, LaNi₅H₆), chemical hydrides.

**Key requirements:**
- Contains H in the structure
- Light metals preferred (Mg, Li, Na, Al, Ca) for gravimetric density
- Transition metal hydrides (Fe, Ti, Ni, La) for kinetics
- Moderate stability — too stable = can't release H₂ at practical temperatures
- Formation energy: moderately negative (−0.3 to −1.5 eV/atom ideal)

```python
def score_hydrogen_storage(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Hard requirement: must contain H
    if 'H' not in elements:
        return 0.0

    # Light metals (good for gravimetric density)
    light_metals = {'Li', 'Na', 'Mg', 'Al', 'Ca', 'K'}
    if elements & light_metals:
        score += 0.30

    # Transition metal hydrides (good kinetics)
    kinetic_metals = {'Fe', 'Ti', 'Ni', 'V', 'Zr', 'La', 'Ce', 'Mm'}
    if elements & kinetic_metals:
        score += 0.25

    # Borohydrides / alanates (complex hydrides)
    if 'B' in elements:
        score += 0.15
    if 'Al' in elements and 'H' in elements:
        score += 0.10

    # Formation energy: moderate stability preferred
    if formation_energy is not None:
        if -1.5 <= formation_energy <= -0.3:
            score += 0.20
        elif -0.3 < formation_energy <= 0.0:
            score += 0.10  # too weak binding

    return min(score, 1.0)
```

---

## DOMAIN 2 — ENERGY CONVERSION

### 6. Solar Absorber (Single Junction)

**What it is:** Photovoltaic absorber layer for single-junction solar cells. Must absorb visible
light efficiently and generate electron-hole pairs. Target band gap: 1.0–1.8 eV (Shockley-Queisser
optimum ~1.34 eV). Known materials: Si (1.1 eV), GaAs (1.4 eV), CdTe (1.45 eV), CIGS (1.0–1.7 eV),
perovskites (1.2–1.7 eV). Your Cu6SiMoS8 (1.03 eV) and Cu6SiWSe8 (0.96 eV) land here.

**Key requirements:**
- Band gap: 1.0–1.8 eV (tight range)
- Direct band gap strongly preferred over indirect
- High optical absorption coefficient
- No highly toxic elements (Cd, Pb penalised but not eliminated)
- Earth-abundant elements preferred (commercial viability)

```python
def score_solar_absorber_singlejunction(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    if band_gap is None:
        return 0.0

    # Core band gap requirement — tight Shockley-Queisser window
    if 1.0 <= band_gap <= 1.8:
        # Peak score at ~1.34 eV
        deviation = abs(band_gap - 1.34)
        score += max(0.50 - deviation * 0.5, 0.25)
    elif 0.8 <= band_gap < 1.0:
        score += 0.10  # too low but could work in tandem
    else:
        return 0.0  # outside useful range

    # Known good absorber element families
    chalcopyrite = {'Cu', 'In', 'Ga', 'Se', 'S'}  # CIGS family
    kesterite = {'Cu', 'Zn', 'Sn', 'S', 'Se'}      # CZTS family
    chalcogenide_metals = {'Mo', 'W', 'Sb', 'Bi', 'Ge'}
    if len(elements & chalcopyrite) >= 3:
        score += 0.25
    if len(elements & kesterite) >= 3:
        score += 0.20
    if elements & chalcogenide_metals:
        score += 0.10

    # Penalise very toxic elements (commercial barrier)
    if 'Cd' in elements:
        score *= 0.6
    if 'As' in elements or 'Hg' in elements:
        score *= 0.5

    # Penalise Pb (regulatory pressure, though perovskites still studied)
    if 'Pb' in elements:
        score *= 0.7

    return min(score, 1.0)
```

---

### 7. Solar Absorber (Tandem Top Cell)

**What it is:** Top cell in a silicon tandem solar cell stack. Must absorb high-energy photons
while transmitting lower-energy light to the Si bottom cell. The perovskite-silicon tandem is
the most commercially important variant, with target top cell band gap 1.6–2.0 eV. World record
efficiency (2025): ~34%.

**Key requirements:**
- Band gap: 1.6–2.0 eV specifically
- Direct gap strongly preferred
- Must be processable as thin film (not bulk crystal only)
- Halide perovskite structure type is a bonus
- Pb-free preferred but not required (Pb-perovskites still dominate research)

```python
def score_solar_absorber_tandem(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    if band_gap is None:
        return 0.0

    # Core band gap: tandem top cell window
    if 1.6 <= band_gap <= 2.0:
        deviation = abs(band_gap - 1.75)
        score += max(0.50 - deviation * 0.8, 0.25)
    elif 1.5 <= band_gap < 1.6:
        score += 0.15
    else:
        return 0.0

    # Halide perovskite indicators
    halides = {'Cl', 'Br', 'I'}
    a_site = {'Cs', 'Rb', 'K', 'Na'}
    b_site = {'Pb', 'Sn', 'Ge', 'Bi', 'Sb', 'In'}
    if (elements & halides) and (elements & a_site) and (elements & b_site):
        score += 0.30
    elif (elements & halides) and (elements & b_site):
        score += 0.15

    # Pb-free bonus (strong regulatory and environmental driver)
    if 'Pb' not in elements and 'Cd' not in elements:
        score += 0.15

    return min(score, 1.0)
```

---

### 8. Thermoelectric

**What it is:** Converts heat directly to electricity (or electricity to cooling) via Seebeck
effect. Figure of merit ZT = S²σT/κ. Need: high Seebeck coefficient S, high electrical
conductivity σ, low thermal conductivity κ. Best materials: PbTe, BiSb₂Te₃, SnSe, halide
perovskites. Near-zero band gap with heavy atoms is ideal.

**Key requirements:**
- Near-metallic band gap (< 0.5 eV optimal, up to ~1.0 eV acceptable)
- Heavy atoms present (atomic mass > 100) — key for low thermal conductivity
- Anharmonic bonding (chalcogenides, halides especially)
- Complex unit cell or multiple inequivalent sites (phonon scattering)

```python
def score_thermoelectric(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Band gap: narrow semiconductor or semimetal
    if band_gap is not None:
        if band_gap < 0.1:
            score += 0.30  # semimetallic — good electrical conductivity
        elif band_gap < 0.5:
            score += 0.35  # ideal narrow gap
        elif band_gap < 1.0:
            score += 0.20
        else:
            score += 0.05  # too insulating

    # Heavy elements (critical for phonon scattering / low κ)
    heavy_elements = {'Pb', 'Bi', 'Sb', 'Te', 'Se', 'Tl', 'In', 'Sn', 'Ge',
                      'Ba', 'Cs', 'I', 'Br', 'Ag', 'Hg'}
    heavy_count = len(elements & heavy_elements)
    score += min(heavy_count * 0.12, 0.35)

    # Chalcogenide / halide framework (highly anharmonic — key for low κ)
    if 'Te' in elements:
        score += 0.10
    elif 'Se' in elements or 'S' in elements:
        score += 0.07
    if elements & {'I', 'Br', 'Cl'}:
        score += 0.08

    # Penalise very light element-dominated structures (high κ)
    light_only = elements - {'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl'}
    if not light_only:
        score *= 0.4  # all light elements = high thermal conductivity

    return min(score, 1.0)
```

**Example targets from GNoME:** Ba12SiSn3I8 (#1 in your thermoelectric results), Cs3CeI6.

---

### 9. OER Electrocatalyst (Oxygen Evolution)

**What it is:** Catalyst for the oxygen evolution half-reaction in water electrolysis
(2H₂O → O₂ + 4H⁺ + 4e⁻). The rate-limiting step in green hydrogen production. Best known:
IrO₂ and RuO₂ (expensive), Fe/Co/Ni oxides (cheap but less active), Fe5Si3 (your result).
Operates in acidic (PEM electrolyzer) or alkaline (AEL) conditions.

**Key requirements:**
- Metallic or near-metallic (needs to conduct electrons to electrode)
- OER-active transition metals: Ir, Ru, Fe, Co, Ni, Mn
- Oxide, hydroxide, or oxyhydroxide structure preferred
- For acidic OER: Ir and Ru almost essential (stability requirement)
- For alkaline OER: Fe, Co, Ni oxides well-established

```python
def score_oer_electrocatalyst(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # OER-active transition metals
    acidic_active = {'Ir', 'Ru'}  # stable in acid
    alkaline_active = {'Fe', 'Co', 'Ni', 'Mn'}  # good in alkaline
    if elements & acidic_active:
        score += 0.40
    elif elements & alkaline_active:
        score += 0.30

    # Must be electronically conductive
    if band_gap is not None:
        if band_gap < 0.5:
            score += 0.25
        elif band_gap < 2.0:
            score += 0.10
        else:
            score *= 0.3  # insulating = bad catalyst support

    # Oxide / mixed oxide (most OER catalysts activate to oxide in situ)
    if 'O' in elements:
        score += 0.15

    # Silicide electrocatalyst bonus (your Fe5Si3 finding)
    if 'Si' in elements and (elements & alkaline_active):
        score += 0.10

    # Bonus for known catalyst structural families
    perovskite_b_sites = {'Fe', 'Co', 'Ni', 'Mn', 'Ir', 'Ru'}
    if 'O' in elements and len(elements & perovskite_b_sites) >= 1:
        score += 0.10

    return min(score, 1.0)
```

---

### 10. HER Electrocatalyst (Hydrogen Evolution)

**What it is:** Catalyst for the hydrogen evolution half-reaction (2H⁺ + 2e⁻ → H₂). 
The other half of water splitting. Best: Pt (expensive), MoS₂ edge sites, Ni, Co-P.
Operates at cathode; different element preferences from OER.

```python
def score_her_electrocatalyst(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # HER-active elements
    noble = {'Pt', 'Pd', 'Rh', 'Ir'}
    earth_abundant = {'Mo', 'W', 'Ni', 'Co', 'Fe', 'Cu'}
    if elements & noble:
        score += 0.40
    elif elements & earth_abundant:
        score += 0.30

    # Sulfides and phosphides are excellent HER catalysts (MoS₂, CoP, Ni₂P)
    if 'S' in elements and (elements & earth_abundant):
        score += 0.20
    if 'P' in elements and (elements & earth_abundant):
        score += 0.20
    if 'Se' in elements and (elements & earth_abundant):
        score += 0.15
    if 'N' in elements and (elements & earth_abundant):
        score += 0.10

    # Metallic or semimetallic
    if band_gap is not None:
        if band_gap < 0.5:
            score += 0.20
        elif band_gap < 1.5:
            score += 0.08

    return min(score, 1.0)
```

---

### 11. CO₂ Reduction Catalyst

**What it is:** Electrochemical or photochemical reduction of CO₂ to fuels/chemicals
(CO, formate, methanol, ethanol, ethylene). Cu is uniquely selective for multi-carbon products.
Active area of research for carbon utilisation.

```python
def score_co2_reduction(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Cu is uniquely active for C-C coupling in CO2RR
    if 'Cu' in elements:
        score += 0.35

    # Other CO2RR-active metals
    co2rr_metals = {'Ag', 'Au', 'Zn', 'Sn', 'Bi', 'In', 'Pb', 'Pd'}
    if elements & co2rr_metals:
        score += 0.25

    # Metallic character (needs to conduct electrons)
    if band_gap is not None:
        if band_gap < 0.5:
            score += 0.20
        elif band_gap < 2.0:
            score += 0.08

    # Oxide surface (many CO2RR catalysts are oxides or oxide-derived)
    if 'O' in elements and (elements & {'Cu', 'Ag', 'Zn', 'Sn', 'Bi'}):
        score += 0.15

    # Penalise if noble-metal free (earth-abundant bonus)
    noble = {'Pt', 'Pd', 'Rh', 'Ir', 'Au', 'Ru'}
    if not (elements & noble):
        score += 0.05

    return min(score, 1.0)
```

---

### 12. Photocatalyst (Water Splitting)

**What it is:** Light-driven water splitting using semiconductor photocatalyst suspended in
water or as thin film. Both OER and HER occur on the same particle. Band edges must straddle
water redox potentials: conduction band < −0.41 V vs NHE (for H₂), valence band > +0.82 V
vs NHE (for O₂). Band gap must be > 1.23 eV (thermodynamic minimum) but ideally < 3 eV
to absorb visible light.

```python
def score_photocatalyst_water_splitting(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    if band_gap is None:
        return 0.0

    # Band gap must straddle water redox: 1.8–3.2 eV optimal for visible light
    if 1.8 <= band_gap <= 3.2:
        # Peak at ~2.2 eV (best balance of absorption and driving force)
        deviation = abs(band_gap - 2.2)
        score += max(0.40 - deviation * 0.15, 0.20)
    elif 1.23 <= band_gap < 1.8:
        score += 0.10  # thermodynamically possible but low driving force
    else:
        return 0.0

    # Known photocatalyst families
    oxide_photocatalysts = {'Ti', 'Zn', 'Ga', 'In', 'Nb', 'Ta', 'W', 'Mo', 'Fe', 'Bi'}
    if 'O' in elements and (elements & oxide_photocatalysts):
        score += 0.25

    # Nitride photocatalysts (GaN, Ta₃N₅ family)
    if 'N' in elements and (elements & {'Ga', 'Ta', 'Ge', 'C'}):
        score += 0.20

    # Sulphide photocatalysts (CdS, ZnIn₂S₄ family — but penalise Cd)
    if 'S' in elements:
        score += 0.10
        if 'Cd' in elements:
            score *= 0.6

    return min(score, 1.0)
```

---

## DOMAIN 3 — ELECTRONICS AND OPTOELECTRONICS

### 13. Semiconductor (General)

**What it is:** Your existing semiconductor category, tightened. General-purpose scoring for
any semiconductor application — catch-all for compounds that don't fit more specific categories.

```python
def score_semiconductor_general(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0

    if band_gap is None:
        return 0.0

    # Semiconductor range
    if 0.1 <= band_gap <= 4.0:
        # Penalise extremes
        if 0.5 <= band_gap <= 3.0:
            score += 0.50
        else:
            score += 0.25
    else:
        return 0.0

    elements = {str(e) for e in composition.elements}

    # Known semiconductor families
    iv = {'Si', 'Ge', 'C'}
    iii_v = {'Ga', 'In', 'Al', 'N', 'P', 'As', 'Sb'}
    ii_vi = {'Zn', 'Cd', 'Hg', 'O', 'S', 'Se', 'Te'}
    if elements & iv:
        score += 0.20
    if len(elements & iii_v) >= 2:
        score += 0.20
    if len(elements & ii_vi) >= 2:
        score += 0.15

    # Transition metal oxides and chalcogenides
    tmo = {'Ti', 'Mo', 'W', 'V', 'Cr', 'Fe', 'Cu', 'Ni', 'Co'}
    if 'O' in elements and (elements & tmo):
        score += 0.10

    return min(score, 1.0)
```

---

### 14. LED / Light Emitter

**What it is:** Light-emitting diode material. Requires direct band gap in visible range.
Different from solar absorber even with same band gap: LED needs high quantum yield, while
solar needs high absorption coefficient. GaN (blue LED), AlGaInP (red/green), halide
perovskites (tunable colour).

```python
def score_led(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    if band_gap is None:
        return 0.0

    # Visible range: 1.77–3.1 eV (700–400 nm)
    if 1.77 <= band_gap <= 3.1:
        # Preference for middle visible (green ~2.3 eV)
        score += 0.45
    elif 3.1 < band_gap <= 4.0:
        score += 0.20  # near-UV LED applications
    else:
        return 0.0

    # III-V semiconductors (commercial LED family)
    iii_v_metals = {'Ga', 'In', 'Al'}
    iii_v_anions = {'N', 'P', 'As'}
    if (elements & iii_v_metals) and (elements & iii_v_anions):
        score += 0.30

    # II-VI compounds (ZnSe, ZnS, CdS etc.)
    if elements & {'Zn', 'Cd'} and elements & {'S', 'Se', 'Te'}:
        score += 0.15

    # Halide perovskites for LEDs (rapidly growing)
    halides = {'Cl', 'Br', 'I'}
    if elements & halides and elements & {'Cs', 'Rb'} and elements & {'Pb', 'Sn', 'In'}:
        score += 0.20

    # Penalise indirect gap indicators (common in Si-type structures)
    # (pymatgen space group analysis could improve this)
    if 'Si' in elements and len(elements) == 1:
        score *= 0.3  # pure Si is indirect gap

    return min(score, 1.0)
```

---

### 15. Photodetector

**What it is:** Semiconductor device that converts light to electrical signal. Wider band gap
range is acceptable than for solar or LED — applications span UV, visible, and infrared.
Key families: Si (visible/NIR), InGaAs (telecom NIR), HgCdTe (IR), perovskites (fast response).

```python
def score_photodetector(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    if band_gap is None:
        return 0.0

    # Broad range: UV to SWIR
    if 0.3 <= band_gap <= 4.5:
        score += 0.40
        # Sub-ranges have specific applications
        if 0.3 <= band_gap < 1.0:
            score += 0.10  # IR photodetector
        elif 1.0 <= band_gap < 2.0:
            score += 0.15  # visible
        elif 2.0 <= band_gap <= 3.5:
            score += 0.10  # UV
    else:
        return 0.0

    # High-mobility semiconductor families
    iii_v = {'Ga', 'In', 'Al'}
    anions = {'N', 'P', 'As', 'Sb'}
    if (elements & iii_v) and (elements & anions):
        score += 0.25

    # Perovskites (fast response, high sensitivity)
    if elements & {'Pb', 'Sn'} and elements & {'I', 'Br', 'Cl'} and elements & {'Cs', 'Rb'}:
        score += 0.20

    # HgCdTe family (IR detectors)
    if 'Hg' in elements and 'Cd' in elements and 'Te' in elements:
        score += 0.15  # technically excellent but toxic

    return min(score, 1.0)
```

---

### 16. Transparent Conductor

**What it is:** Wide band gap material that is also electrically conductive. Seemingly
contradictory — achieved by degenerate doping. ITO (In₂O₃:Sn) dominates commercially
but In is expensive and scarce. Alternatives: AZO (Al-doped ZnO), FTO (F-doped SnO₂),
Ga₂O₃, amorphous oxides.

**Key requirements:**
- Band gap > 3.0 eV (transparent to visible light)
- Low effective electron mass (for high conductivity when doped)
- Oxides strongly preferred (most known transparent conductors are oxides)
- Avoid highly toxic elements

```python
def score_transparent_conductor(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    if band_gap is None:
        return 0.0

    # Wide band gap requirement
    if band_gap >= 3.5:
        score += 0.45
    elif band_gap >= 3.0:
        score += 0.30
    else:
        return 0.0  # would absorb visible light

    # Known transparent conductor families
    tco_metals = {'In', 'Sn', 'Zn', 'Ga', 'Cd', 'Al', 'Ti'}
    if 'O' in elements and (elements & tco_metals):
        score += 0.35

    # Bonus for earth-abundant ITO alternatives (In is scarce)
    if 'Zn' in elements and 'O' in elements and 'In' not in elements:
        score += 0.15  # AZO/GZO family
    if 'Sn' in elements and 'O' in elements:
        score += 0.10  # FTO family

    # Penalise non-oxides (transparent conductors are almost all oxides)
    if 'O' not in elements:
        score *= 0.4

    return min(score, 1.0)
```

---

### 17. Ferroelectric

**What it is:** Material with spontaneous electric polarisation that can be reversed by
applied field. Used in capacitors, memory devices (FeRAM), sensors, actuators.
Key requirement: non-centrosymmetric crystal structure. BaTiO₃ is the archetypal example.
Perovskite ABO₃ structure is strongly associated.

```python
def score_ferroelectric(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Non-centrosymmetric space groups if structure available
    if structure is not None:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure)
            sg_num = sga.get_space_group_number()
            # Non-centrosymmetric space groups (rough filter)
            # Full list would be 68 of 230 space groups
            centrosymmetric_sg = set(range(1, 3)) | {10,11,12,13,14,15} | \
                                  set(range(47, 75)) | set(range(83, 89)) | \
                                  set(range(123, 143)) | set(range(147, 149)) | \
                                  set(range(162, 168)) | set(range(175, 177)) | \
                                  set(range(191, 195)) | set(range(200, 207)) | \
                                  set(range(221, 231))
            if sg_num not in centrosymmetric_sg:
                score += 0.35
        except:
            pass

    # Perovskite-type elements (ABO₃ structure)
    a_site = {'Ba', 'Sr', 'Ca', 'Pb', 'Na', 'K', 'Bi'}
    b_site = {'Ti', 'Zr', 'Nb', 'Ta', 'Fe', 'Mn', 'W'}
    if 'O' in elements and (elements & a_site) and (elements & b_site):
        score += 0.35

    # Band gap: ferroelectrics are insulators
    if band_gap is not None:
        if band_gap >= 2.5:
            score += 0.20
        elif band_gap >= 1.5:
            score += 0.10

    # Pb-free ferroelectrics (regulatory bonus)
    if 'Pb' not in elements and 'O' in elements:
        score += 0.10

    return min(score, 1.0)
```

---

### 18. Piezoelectric

**What it is:** Converts mechanical stress to electrical charge (and vice versa). Used in
sensors, actuators, ultrasound transducers, energy harvesters. Same non-centrosymmetric
requirement as ferroelectric. PZT (Pb-Zr-Ti oxide) dominates industrially but is lead-heavy.
KNbO₃, BaTiO₃, AlN are lead-free alternatives.

```python
def score_piezoelectric(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Non-centrosymmetric check (same as ferroelectric)
    if structure is not None:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure)
            sg_num = sga.get_space_group_number()
            # Polar space groups are a subset of non-centrosymmetric
            polar_sg = {1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 17, 18, 19, 26,
                        28, 29, 30, 31, 32, 33, 34, 36, 38, 39, 40, 41, 42, 44,
                        45, 46, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
                        109, 110, 111, 113, 115, 117, 119, 121, 156, 157, 158,
                        159, 160, 161, 166, 168, 169, 170, 171, 172, 173, 174,
                        177, 182, 183, 184, 185, 186, 187, 188}
            if sg_num in polar_sg:
                score += 0.40
        except:
            pass

    # Known piezoelectric element combinations
    if 'O' in elements:
        piezo_metals = {'Ti', 'Zr', 'Nb', 'Ta', 'Al', 'Ga', 'Zn', 'Li', 'Ba', 'Pb', 'K'}
        if elements & piezo_metals:
            score += 0.30

    # AlN — excellent for high-frequency MEMS
    if 'Al' in elements and 'N' in elements:
        score += 0.20

    # Wide band gap (insulators preferred)
    if band_gap is not None and band_gap >= 3.0:
        score += 0.15

    # Lead-free bonus
    if 'Pb' not in elements:
        score += 0.15

    return min(score, 1.0)
```

---

### 19. Topological Insulator

**What it is:** Material that is insulating in the bulk but has conducting surface/edge states
protected by time-reversal symmetry. Applications in quantum computing, spintronics,
low-dissipation electronics. Key families: Bi₂Te₃, Bi₂Se₃, Bi₂Se₂Te, SnTe, Pb₁₋ₓSnₓSe.
Heavy elements with strong spin-orbit coupling are essential.

```python
def score_topological_insulator(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    if band_gap is None:
        return 0.0

    # Must be an insulator but small gap
    if 0.05 <= band_gap <= 0.5:
        score += 0.35
    elif 0.5 < band_gap <= 1.5:
        score += 0.15
    else:
        return 0.0

    # Heavy elements with strong spin-orbit coupling (essential for topology)
    soc_heavy = {'Bi', 'Sb', 'Pb', 'Sn', 'Te', 'Se', 'Tl', 'Hg', 'In'}
    heavy_count = len(elements & soc_heavy)
    score += min(heavy_count * 0.15, 0.40)

    # Quintuple-layer structure indicator (Bi₂X₃ family)
    if 'Bi' in elements and elements & {'Te', 'Se', 'S'}:
        score += 0.20

    # IV-VI topological crystalline insulators (SnTe, PbTe type)
    if elements & {'Sn', 'Pb', 'Ge'} and elements & {'Te', 'Se', 'S'}:
        score += 0.15

    return min(score, 1.0)
```

---

## DOMAIN 4 — MAGNETICS

### 20. Permanent Magnet

**What it is:** Hard magnet with high coercivity and remanence — stays magnetised without
applied field. Applications: EV motors, wind turbines, hard drives. Nd₂Fe₁₄B (neodymium
magnet) dominates. Sm₂Co₁₇ is the high-temperature alternative. Both contain critical REEs.

**Key requirements:**
- Must contain Fe, Co, or Ni (magnetic moment carriers)
- REE for anisotropy (Nd, Sm, Dy, Pr) — but score accordingly
- No band gap (must be metallic)
- High density of magnetic atoms

```python
def score_permanent_magnet(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Core magnetic elements
    magnetic_metals = {'Fe', 'Co', 'Ni'}
    if not (elements & magnetic_metals):
        return 0.0
    score += 0.30

    # REE for magnetocrystalline anisotropy (what makes a hard magnet)
    anisotropy_ree = {'Nd', 'Sm', 'Dy', 'Pr', 'Tb', 'Ho'}
    if elements & anisotropy_ree:
        score += 0.35  # essential for permanent magnet — despite REE cost

    # Boron is almost universal in top permanent magnets (Nd₂Fe₁₄B type)
    if 'B' in elements:
        score += 0.15

    # Must be metallic
    if band_gap is not None:
        if band_gap < 0.1:
            score += 0.15
        else:
            score *= 0.5  # semiconducting permanent magnets are rare

    # Nitrogen interstitials (Sm₂Fe₁₇N₃ type — emerging class)
    if 'N' in elements and 'Fe' in elements and 'Sm' in elements:
        score += 0.05

    return min(score, 1.0)
```

**Viability notes:** REE penalty from your existing filter applies here and is correct.
Permanent magnets legitimately require REEs — this is the domain where the REE cost is
most commercially accepted. Consider reducing the REE penalty multiplier specifically for
this category in your viability.py.

---

### 21. Soft Magnet

**What it is:** Low coercivity magnetic material — easy to magnetise and demagnetise.
Used in transformer cores, inductors, magnetic shielding, sensors. Fe-Si alloys are the
commercial workhorse (electrical steel). Ferrites used in high-frequency applications.
No REE required.

```python
def score_soft_magnet(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    magnetic_metals = {'Fe', 'Co', 'Ni'}
    if not (elements & magnetic_metals):
        return 0.0
    score += 0.25

    # Fe-Si alloys (electrical steel — dominant soft magnet)
    if 'Fe' in elements and 'Si' in elements:
        score += 0.30

    # Ferrites (Mn-Zn, Ni-Zn, spinel ferrites)
    ferrite_metals = {'Mn', 'Zn', 'Ni', 'Cu'}
    if 'Fe' in elements and 'O' in elements and (elements & ferrite_metals):
        score += 0.25

    # Amorphous / nanocrystalline indicators (Fe + B, or Fe + P + C)
    if 'Fe' in elements and 'B' in elements:
        score += 0.10
    if 'Fe' in elements and 'P' in elements:
        score += 0.10

    # Metallic
    if band_gap is not None and band_gap < 0.5:
        score += 0.15

    # Penalise REE (soft magnets don't need them and they add cost)
    ree = {'Nd', 'Sm', 'Dy', 'Pr', 'Tb', 'Gd', 'La', 'Ce', 'Er', 'Ho', 'Tm', 'Yb', 'Lu'}
    if elements & ree:
        score *= 0.7

    return min(score, 1.0)
```

---

### 22. Magnetic Semiconductor / Spintronics

**What it is:** Material combining semiconductor band gap with magnetic ordering. Used in
spin-polarised LEDs, magnetic memory, quantum computing. Dilute magnetic semiconductors
(DMS) like (Ga,Mn)As are the main family. Rare earth hydrides from your results fall here.

```python
def score_magnetic_semiconductor(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    if band_gap is None:
        return 0.0

    # Must be a semiconductor
    if not (0.1 <= band_gap <= 3.5):
        return 0.0
    score += 0.20

    # Magnetic 3d transition metals (spin source)
    magnetic_3d = {'Mn', 'Fe', 'Co', 'Ni', 'Cr', 'V'}
    if elements & magnetic_3d:
        score += 0.30

    # Rare earth f-electron magnets (4f magnetic moments)
    magnetic_ree = {'Gd', 'Eu', 'Dy', 'Nd', 'Sm', 'Tb', 'Ho', 'Er'}
    if elements & magnetic_ree:
        score += 0.25

    # Semiconductor host matrix
    semiconductor_host = {'Ga', 'In', 'Ge', 'Si', 'Zn', 'Cd'}
    if elements & semiconductor_host:
        score += 0.15

    # Chalcogenide magnetic semiconductors (EuS, EuO type)
    if elements & {'Eu', 'Gd'} and elements & {'S', 'Se', 'O', 'Te'}:
        score += 0.10

    return min(score, 1.0)
```

---

## DOMAIN 5 — THERMAL AND STRUCTURAL COATINGS

### 23. Thermal Barrier Coating (TBC)

**What it is:** Ceramic coating on turbine blades to insulate metal from hot combustion gases.
Enables higher operating temperatures → better engine efficiency. YSZ (yttria-stabilised
zirconia) is the current standard but degrades above 1200°C. Hexaaluminates (LaMgAl₁₁O₁₉
type) and pyrochlores (La₂Zr₂O₇) are next-generation candidates. Ba12PrY3Al8O30 from your
results fits here.

```python
def score_tbc(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Must be an oxide
    if 'O' not in elements:
        return 0.0

    # Wide band gap (optical transparency, no electronic conduction at high T)
    if band_gap is not None:
        if band_gap >= 4.0:
            score += 0.35
        elif band_gap >= 3.0:
            score += 0.20
        else:
            score += 0.05

    # Zirconia family (current standard)
    if 'Zr' in elements:
        score += 0.25
        if 'Y' in elements:
            score += 0.10  # YSZ — the benchmark

    # Hexaaluminate family (next generation)
    if 'Al' in elements:
        a_site_hex = {'Ba', 'Sr', 'La', 'Ce', 'Pr', 'Nd', 'Sm'}
        if elements & a_site_hex:
            score += 0.30  # magnetoplumbite/β-alumina structure

    # Pyrochlore family (A₂B₂O₇)
    pyro_a = {'La', 'Nd', 'Sm', 'Gd', 'Er', 'Yb', 'Y'}
    pyro_b = {'Zr', 'Ti', 'Hf', 'Ce', 'Sn'}
    if (elements & pyro_a) and (elements & pyro_b):
        score += 0.25

    # High melting point indicator: refractory metal oxides
    refractory_ox = {'Hf', 'Ta', 'W', 'Nb', 'Mo'}
    if elements & refractory_ox:
        score += 0.10

    # Very stable formation energy (must survive high temperature)
    if formation_energy is not None and formation_energy < -3.0:
        score += 0.10

    return min(score, 1.0)
```

---

### 24. Thermal Interface Material

**What it is:** High thermal conductivity, electrically insulating material placed between
heat source and heatsink. Used in power electronics packaging. Key: high κ (>10 W/m·K),
wide band gap. AlN (κ ~180 W/m·K), BN (~300 W/m·K), diamond (2000 W/m·K), BeO (270 W/m·K).

```python
def score_thermal_interface(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Must be electrically insulating
    if band_gap is not None:
        if band_gap >= 5.0:
            score += 0.35
        elif band_gap >= 3.5:
            score += 0.25
        elif band_gap >= 2.5:
            score += 0.10
        else:
            return 0.0

    # High thermal conductivity material families
    # Diamond and cBN: highest thermal conductors
    if elements == {'C'}:  # diamond
        score += 0.50
    if 'B' in elements and 'N' in elements:
        score += 0.40  # BN family

    # Aluminium nitride (very common in power electronics)
    if 'Al' in elements and 'N' in elements:
        score += 0.35

    # Silicon carbide (also high κ)
    if 'Si' in elements and 'C' in elements:
        score += 0.25

    # Beryllium oxide (extremely high κ but toxic)
    if 'Be' in elements and 'O' in elements:
        score += 0.20
        score *= 0.5  # BeO is highly toxic — flag this

    # Simple binary/ternary structure (complex structures have lower κ)
    n_elements = len(composition.elements)
    if n_elements <= 2:
        score += 0.10
    elif n_elements >= 5:
        score *= 0.7

    return min(score, 1.0)
```

---

### 25. Hard Coating / Wear Resistant

**What it is:** Hard ceramic coating on cutting tools, dies, engine components.
TiN, TiAlN, CrN, DLC (diamond-like carbon), Al₂O₃ are commercial standards.
Key property: high hardness (>20 GPa Vickers), which correlates with high bulk modulus
and short strong bonds. Transition metal nitrides, carbides, and borides dominate.

```python
def score_hard_coating(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Hard material element families (nitrides, carbides, borides)
    transition_metals = {'Ti', 'Cr', 'W', 'Mo', 'V', 'Nb', 'Ta', 'Zr', 'Hf', 'Al'}
    hard_anions = {'N', 'C', 'B'}

    if (elements & transition_metals) and (elements & hard_anions):
        score += 0.45

    # Pure carbon (diamond, DLC)
    if elements == {'C'}:
        score += 0.40

    # Al₂O₃ (alumina coatings — CVD on cutting tools)
    if elements == {'Al', 'O'} or ('Al' in elements and 'O' in elements and len(elements) <= 3):
        score += 0.25

    # Ternary nitrides/carbides (better performance than binary)
    if len(elements & transition_metals) >= 2 and (elements & hard_anions):
        score += 0.15  # e.g. TiAlN better than TiN

    # Wide band gap (hardness correlates with ionic/covalent bonding)
    if band_gap is not None:
        if band_gap >= 3.0:
            score += 0.15
        elif band_gap >= 1.5:
            score += 0.08

    # Very negative formation energy (stable at high temperatures)
    if formation_energy is not None and formation_energy < -2.5:
        score += 0.10

    return min(score, 1.0)
```

---

### 26. Corrosion Resistant Coating

**What it is:** Protective coating to prevent oxidation, acid attack, or electrochemical
corrosion. Passive oxide layers (Cr₂O₃ on stainless steel), fluorides, and noble metal
oxides. Chemically inert, wide band gap, thermodynamically stable.

```python
def score_corrosion_resistant(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Chromia-forming / passivating oxide formers
    passive_metals = {'Cr', 'Al', 'Ti', 'Zr', 'Ta', 'Nb', 'Hf', 'Si', 'W'}
    if 'O' in elements and (elements & passive_metals):
        score += 0.35

    # Fluoride coatings (extremely chemically inert)
    if 'F' in elements:
        fluoride_metals = {'Ca', 'Ba', 'Sr', 'Mg', 'La', 'Ce', 'Al'}
        if elements & fluoride_metals:
            score += 0.30

    # Wide band gap and chemically stable
    if band_gap is not None and band_gap >= 3.5:
        score += 0.20

    # Very negative formation energy (thermodynamic stability = corrosion resistance)
    if formation_energy is not None and formation_energy < -4.0:
        score += 0.20
    elif formation_energy is not None and formation_energy < -2.5:
        score += 0.10

    # Noble metals (excellent corrosion resistance)
    noble = {'Pt', 'Pd', 'Au', 'Ir', 'Rh', 'Ru'}
    if elements & noble:
        score += 0.15

    return min(score, 1.0)
```

---

### 27. Refractory / Ultra-High Temperature Structural

**What it is:** Structural materials for use above 1500°C. Turbine hot sections, hypersonic
vehicles, nuclear reactor components. Key families: HfC, ZrB₂, TaC, HfB₂ (UHTCs —
ultra-high temperature ceramics), W and Re alloys, transition metal carbides.

```python
def score_refractory(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Refractory transition metals (melting point > 2000°C)
    refractory_metals = {'W', 'Re', 'Os', 'Ta', 'Mo', 'Hf', 'Nb', 'Zr', 'V', 'Cr', 'Ti'}
    if elements & refractory_metals:
        score += 0.30

    # UHTC carbides and borides (HfC, ZrB₂ family)
    uhtc = {'Hf', 'Zr', 'Ta', 'Ti', 'Nb'}
    if (elements & uhtc) and ('C' in elements or 'B' in elements or 'N' in elements):
        score += 0.35

    # Very stable thermodynamics
    if formation_energy is not None and formation_energy < -3.0:
        score += 0.20
    elif formation_energy is not None and formation_energy < -1.5:
        score += 0.10

    # No volatile elements (must survive high temperature)
    volatile = {'Na', 'K', 'Li', 'Rb', 'Cs', 'Zn', 'Cd', 'Hg', 'S', 'Se', 'Te'}
    if elements & volatile:
        score *= 0.4

    # Wide band gap or metallic (both types exist in refractory materials)
    if band_gap is not None:
        if band_gap < 0.5 or band_gap > 4.0:
            score += 0.15

    return min(score, 1.0)
```

---

## DOMAIN 6 — EMERGING AND SPECIALISED

### 28. Superconductor

**What it is:** Zero electrical resistance below critical temperature Tc. Applications:
MRI magnets, particle accelerators, quantum computing, lossless power transmission.
Conventional BCS superconductors (Nb, NbTi, Nb₃Sn) require liquid He cooling. High-Tc
cuprates (YBCO) work at 77K (liquid N₂). Your rare earth hydrides under pressure were
showing up in superconductor prediction papers.

```python
def score_superconductor(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Must be metallic
    if band_gap is not None and band_gap > 0.5:
        return 0.0
    score += 0.20

    # BCS/conventional superconductor elements
    conventional = {'Nb', 'V', 'Pb', 'Sn', 'In', 'Al', 'Mo', 'Re', 'W'}
    if elements & conventional:
        score += 0.25

    # Cuprate high-Tc indicators (Cu + O + REE or Ba/Sr)
    if 'Cu' in elements and 'O' in elements:
        cuprate_a_site = {'Ba', 'Sr', 'La', 'Y', 'Bi', 'Tl', 'Hg'}
        if elements & cuprate_a_site:
            score += 0.35

    # Iron-based superconductor indicators (FeAs, FeSe planes)
    if 'Fe' in elements and elements & {'As', 'Se', 'P'}:
        score += 0.25

    # Hydride superconductors (high-Tc under pressure)
    if 'H' in elements:
        hydride_hosts = {'La', 'Y', 'Ce', 'Th', 'Ca', 'Ba', 'Lu'}
        if elements & hydride_hosts:
            score += 0.20

    return min(score, 1.0)
```

---

### 29. Radiation Detector / Scintillator

**What it is:** Converts high-energy radiation (X-ray, gamma, neutron) into detectable
signal. Medical imaging (CT, PET), security screening, nuclear monitoring. BaI₂:Eu,
CsI:Tl, LYSO (Lu₂SiO₅:Ce), NaI:Tl are commercial scintillators. Your BaI₂ family
compounds from the thermoelectric results are actually relevant here.

**Key requirements:**
- High density (heavy elements for gamma absorption)
- Wide enough band gap to be a scintillator (>3 eV) OR narrow gap for semiconductor detector
- Contains luminescence-activatable sites (rare earth dopant sites)
- High Z elements (Ba, I, Lu, Bi, Hg, Pb) for radiation stopping power

```python
def score_radiation_detector(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # High-Z elements (high radiation stopping power)
    high_z = {'Ba', 'I', 'Cs', 'Bi', 'Pb', 'Tl', 'Hg', 'W', 'Lu', 'Gd', 'Xe'}
    high_z_count = len(elements & high_z)
    score += min(high_z_count * 0.15, 0.40)

    # Scintillator band gap (needs to emit visible photons)
    if band_gap is not None:
        if 3.0 <= band_gap <= 6.0:
            score += 0.30  # scintillator window
        elif band_gap < 3.0:
            score += 0.10  # semiconductor detector (Ge, CdTe type)

    # Known scintillator families
    # Alkali halides (NaI, CsI, BaI₂ type)
    alkali = {'Na', 'Cs', 'K', 'Ba', 'Sr'}
    halides = {'I', 'Br', 'Cl', 'F'}
    if (elements & alkali) and (elements & halides):
        score += 0.25

    # Oxide scintillators (LYSO, BGO, LSO type)
    scint_oxides = {'Lu', 'Gd', 'Bi', 'Y', 'Ce'}
    if 'O' in elements and (elements & scint_oxides):
        score += 0.20

    # Gd-containing for neutron detection
    if 'Gd' in elements:
        score += 0.10

    return min(score, 1.0)
```

---

### 30. Solid Oxide Fuel Cell (SOFC) Component

**What it is:** High-temperature electrochemical device converting fuel to electricity.
Operates at 600–1000°C. Three key components with different material requirements:
electrolyte (oxide ion conductor, wide gap), cathode (mixed ionic-electronic conductor),
anode (electronic conductor + fuel oxidation catalyst). YSZ is the standard electrolyte.

```python
def score_sofc_electrolyte(composition, structure, band_gap, formation_energy, e_hull, features):
    """Scores for SOFC electrolyte component (oxide ion conductor)."""
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Must be an oxide
    if 'O' not in elements:
        return 0.0

    # Wide band gap (electronic insulator)
    if band_gap is not None:
        if band_gap >= 4.0:
            score += 0.35
        elif band_gap >= 3.0:
            score += 0.20
        else:
            return 0.0

    # Known oxide ion conductor families
    # Fluorite structure (ZrO₂, CeO₂, Bi₂O₃)
    fluorite = {'Zr', 'Ce', 'Hf', 'Th'}
    if 'O' in elements and (elements & fluorite):
        score += 0.30
        if 'Y' in elements:
            score += 0.15  # YSZ benchmark

    # Perovskite oxide conductors (LaGaO₃ family)
    perovskite_sofc = {'La', 'Sr', 'Ga', 'Mg'}
    if 'O' in elements and len(elements & perovskite_sofc) >= 2:
        score += 0.25

    # BIMEVOX family (Bi₄V₂O₁₁ derivatives)
    if 'Bi' in elements and 'V' in elements and 'O' in elements:
        score += 0.20

    # High temperature stability (very negative formation energy)
    if formation_energy is not None and formation_energy < -3.5:
        score += 0.10

    return min(score, 1.0)
```

---

### 31. Multiferroic

**What it is:** Rare class of material simultaneously showing ferroelectric AND magnetic
ordering. Allows electric field control of magnetic properties — huge for low-power memory.
BiFeO₃ is the main known room-temperature multiferroic. Very few compounds qualify.

```python
def score_multiferroic(composition, structure, band_gap, formation_energy, e_hull, features):
    score = 0.0
    elements = {str(e) for e in composition.elements}

    # Must have magnetic transition metal
    magnetic_metals = {'Fe', 'Mn', 'Co', 'Ni', 'Cr', 'V', 'Cu'}
    if not (elements & magnetic_metals):
        return 0.0
    score += 0.20

    # Non-centrosymmetric structure check
    if structure is not None:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure)
            if not sga.is_laue():
                score += 0.25
        except:
            pass

    # Perovskite-type structure with both magnetic and polar B-site
    a_site = {'Bi', 'Pb', 'Ba', 'La', 'Ca', 'Sr'}
    polar_b = {'Ti', 'V', 'Nb', 'Ta', 'W', 'Mo'}
    magnetic_b = {'Fe', 'Mn', 'Co', 'Ni', 'Cr'}
    if 'O' in elements and (elements & a_site):
        if (elements & polar_b) and (elements & magnetic_b):
            score += 0.35  # classic multiferroic setup
        elif elements & magnetic_b:
            score += 0.15

    # BiFeO₃ family (the most studied multiferroic)
    if 'Bi' in elements and 'Fe' in elements and 'O' in elements:
        score += 0.20

    # Semiconductor gap (multiferroics tend to be insulators/semiconductors)
    if band_gap is not None and 1.0 <= band_gap <= 4.0:
        score += 0.10

    return min(score, 1.0)
```

---

## Integration Into scoring.py

Replace your current scoring block with the full registry:

```python
# scoring.py — Application Score Registry

SCORING_FUNCTIONS = {
    # Energy Storage
    'battery_cathode_liion':     score_battery_cathode_liion,
    'battery_anode':             score_battery_anode,
    'battery_cathode_naion':     score_battery_cathode_naion,
    'solid_electrolyte':         score_solid_electrolyte,
    'hydrogen_storage':          score_hydrogen_storage,

    # Energy Conversion
    'solar_singlejunction':      score_solar_absorber_singlejunction,
    'solar_tandem':              score_solar_absorber_tandem,
    'thermoelectric':            score_thermoelectric,
    'oer_electrocatalyst':       score_oer_electrocatalyst,
    'her_electrocatalyst':       score_her_electrocatalyst,
    'co2_reduction':             score_co2_reduction,
    'photocatalyst_h2o':         score_photocatalyst_water_splitting,

    # Electronics and Optoelectronics
    'semiconductor':             score_semiconductor_general,
    'led':                       score_led,
    'photodetector':             score_photodetector,
    'transparent_conductor':     score_transparent_conductor,
    'ferroelectric':             score_ferroelectric,
    'piezoelectric':             score_piezoelectric,
    'topological_insulator':     score_topological_insulator,

    # Magnetics
    'permanent_magnet':          score_permanent_magnet,
    'soft_magnet':               score_soft_magnet,
    'magnetic_semiconductor':    score_magnetic_semiconductor,

    # Thermal and Structural Coatings
    'thermal_barrier':           score_tbc,
    'thermal_interface':         score_thermal_interface,
    'hard_coating':              score_hard_coating,
    'corrosion_resistant':       score_corrosion_resistant,
    'refractory':                score_refractory,

    # Emerging and Specialised
    'superconductor':            score_superconductor,
    'radiation_detector':        score_radiation_detector,
    'sofc_electrolyte':          score_sofc_electrolyte,
    'multiferroic':              score_multiferroic,
}


def score_all_applications(composition, structure, band_gap,
                           formation_energy, e_hull, features):
    """Run all 31 scorers and return a dict of application → score."""
    return {
        name: func(composition, structure, band_gap,
                   formation_energy, e_hull, features)
        for name, func in SCORING_FUNCTIONS.items()
    }
```

---

## Streamlit Dropdown Labels

Add these human-readable labels to your app.py dropdown:

```python
APPLICATION_LABELS = {
    'battery_cathode_liion':   'Battery Cathode (Li-ion)',
    'battery_anode':           'Battery Anode',
    'battery_cathode_naion':   'Battery Cathode (Na-ion)',
    'solid_electrolyte':       'Solid Electrolyte',
    'hydrogen_storage':        'Hydrogen Storage',
    'solar_singlejunction':    'Solar Absorber — Single Junction',
    'solar_tandem':            'Solar Absorber — Tandem Top Cell',
    'thermoelectric':          'Thermoelectric',
    'oer_electrocatalyst':     'OER Electrocatalyst (Water Splitting)',
    'her_electrocatalyst':     'HER Electrocatalyst (Green Hydrogen)',
    'co2_reduction':           'CO₂ Reduction Catalyst',
    'photocatalyst_h2o':       'Photocatalyst (Water Splitting)',
    'semiconductor':           'Semiconductor (General)',
    'led':                     'LED / Light Emitter',
    'photodetector':           'Photodetector',
    'transparent_conductor':   'Transparent Conductor',
    'ferroelectric':           'Ferroelectric',
    'piezoelectric':           'Piezoelectric',
    'topological_insulator':   'Topological Insulator',
    'permanent_magnet':        'Permanent Magnet',
    'soft_magnet':             'Soft Magnet',
    'magnetic_semiconductor':  'Magnetic Semiconductor / Spintronics',
    'thermal_barrier':         'Thermal Barrier Coating',
    'thermal_interface':       'Thermal Interface Material',
    'hard_coating':            'Hard Coating / Wear Resistant',
    'corrosion_resistant':     'Corrosion Resistant Coating',
    'refractory':              'Refractory / UHTC',
    'superconductor':          'Superconductor',
    'radiation_detector':      'Radiation Detector / Scintillator',
    'sofc_electrolyte':        'Solid Oxide Fuel Cell Electrolyte',
    'multiferroic':            'Multiferroic',
}
```

---

## Viability Override: Permanent Magnet REE Exception

In viability.py, add a category-specific override so the rare earth penalty
is softened for permanent magnets (REEs are legitimately required and
commercially accepted there):

```python
def apply_ree_penalty(composition, base_viability, application_category):
    """
    Apply rare earth penalty with category-specific override.
    Permanent magnets legitimately need REEs — reduce penalty for them.
    """
    elements = {str(e) for e in composition.elements}

    # REE penalty tiers (from the main viability scorer)
    hard_ree    = {'Dy', 'Tb', 'Eu', 'Ho', 'Er', 'Tm', 'Lu', 'Yb'}
    moderate_ree = {'Nd', 'Pr', 'Sm', 'Gd', 'Sc'}
    mild_ree    = {'La', 'Ce', 'Y'}

    if application_category == 'permanent_magnet':
        # REE is expected here — halve the penalty
        hard_mult     = 0.65   # vs 0.3 normally
        moderate_mult = 0.80   # vs 0.6 normally
        mild_mult     = 0.95   # vs 0.85 normally
    else:
        hard_mult     = 0.30
        moderate_mult = 0.60
        mild_mult     = 0.85

    multiplier = 1.0
    for e in elements:
        if e in hard_ree:
            multiplier *= hard_mult
        elif e in moderate_ree:
            multiplier *= moderate_mult
        elif e in mild_ree:
            multiplier *= mild_mult

    return base_viability * multiplier
```

---

## Known Caveats and GNoME-Specific Issues

**Band gap underestimation:** DFT-predicted band gaps (including CHGNet estimates) typically
underestimate true values by 30–50%. A compound predicted at 0.8 eV may actually be 1.1–1.3 eV.
Consider applying a 1.3× correction factor to band gaps when comparing against target ranges,
or widen all target windows by ~0.3 eV on the lower bound.

**Disordered structures:** Some GNoME CIFs have partial occupancies. The space group symmetry
checks in ferroelectric/piezoelectric/multiferroic scorers will fail on disordered structures.
Wrap all `SpacegroupAnalyzer` calls in try/except and return a neutral 0.5 score if they fail.

**Structure=None fallback:** For categories that rely heavily on space group (ferroelectric,
piezoelectric, multiferroic), the score will be substantially lower when structure analysis
fails. This is intentional — these are structure-sensitive properties and a composition-only
score is genuinely uncertain.

**E_hull context:** GNoME guarantees all 240K compounds are below the convex hull (E_hull ≤ 0).
However, GNoME used its own DFT settings which may differ slightly from Materials Project.
For very borderline compounds (E_hull between -5 and 0 meV/atom), treat stability as uncertain.

**Radioactive elements:** The radioactive element hard filter (atomic number > 83, plus Tc)
should be applied BEFORE any application scoring — if a compound contains Np, Ac, Th, etc.,
return 0.0 viability regardless of how good the application score is. This is already
implemented in your viability.py.

---

*Document version: 1.0 — covers all 31 MatIntel application categories*
*Last updated for GNoME dataset compatibility*
