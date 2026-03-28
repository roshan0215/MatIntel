const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, ExternalHyperlink, PageBreak, TabStopType, TabStopPosition
} = require('docx');
const fs = require('fs');

// ── Helpers ──────────────────────────────────────────────────────────────────

const FONT = 'Arial';
const CONTENT_WIDTH = 9360; // US Letter, 1-inch margins
const COL_BORDER = { style: BorderStyle.SINGLE, size: 1, color: 'CCCCCC' };
const ALL_BORDERS = { top: COL_BORDER, bottom: COL_BORDER, left: COL_BORDER, right: COL_BORDER };
const NO_BORDERS = {
  top: { style: BorderStyle.NIL }, bottom: { style: BorderStyle.NIL },
  left: { style: BorderStyle.NIL }, right: { style: BorderStyle.NIL }
};

const ACCENT   = '1F4E8C'; // deep blue
const ACCENT2  = '2E7D56'; // dark green
const ACCENT3  = '6B3FA0'; // purple
const WARN     = '8B4513'; // brown
const LIGHT_BG = 'EEF4FB';
const GREEN_BG = 'EAF4EE';
const PURPLE_BG= 'F2EEFA';
const WARN_BG  = 'FEF6EC';
const GRAY_BG  = 'F5F5F5';
const HEADER_BG= '1F4E8C';

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 160 },
    children: [new TextRun({ text, font: FONT, size: 32, bold: true, color: ACCENT })]
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: ACCENT, space: 4 } },
    children: [new TextRun({ text, font: FONT, size: 26, bold: true, color: ACCENT })]
  });
}

function h3(text, color = '222222') {
  return new Paragraph({
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, font: FONT, size: 22, bold: true, color })]
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    children: [new TextRun({ text, font: FONT, size: 22, ...opts })]
  });
}

function link(label, url) {
  return new ExternalHyperlink({
    link: url,
    children: [new TextRun({ text: label, font: FONT, size: 22, color: '1155CC', underline: {} })]
  });
}

function bodyWithLink(prefix, label, url, suffix = '') {
  const runs = [];
  if (prefix) runs.push(new TextRun({ text: prefix, font: FONT, size: 22 }));
  runs.push(link(label, url));
  if (suffix) runs.push(new TextRun({ text: suffix, font: FONT, size: 22 }));
  return new Paragraph({ spacing: { before: 60, after: 60 }, children: runs });
}

function bullet(text, level = 0, bold = false) {
  const indent = level === 0 ? { left: 720, hanging: 360 } : { left: 1080, hanging: 360 };
  return new Paragraph({
    numbering: { reference: 'bullets', level },
    spacing: { before: 40, after: 40 },
    indent,
    children: [new TextRun({ text, font: FONT, size: 22, bold })]
  });
}

function bulletLink(prefix, label, url, suffix = '') {
  const runs = [];
  if (prefix) runs.push(new TextRun({ text: prefix, font: FONT, size: 22 }));
  runs.push(link(label, url));
  if (suffix) runs.push(new TextRun({ text: suffix, font: FONT, size: 22 }));
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { before: 40, after: 40 },
    indent: { left: 720, hanging: 360 },
    children: runs
  });
}

function numbered(text, bold = false) {
  return new Paragraph({
    numbering: { reference: 'numbers', level: 0 },
    spacing: { before: 60, after: 60 },
    indent: { left: 720, hanging: 360 },
    children: [new TextRun({ text, font: FONT, size: 22, bold })]
  });
}

function numberedWithLink(prefix, label, url, suffix = '') {
  const runs = [];
  if (prefix) runs.push(new TextRun({ text: prefix, font: FONT, size: 22 }));
  runs.push(link(label, url));
  if (suffix) runs.push(new TextRun({ text: suffix, font: FONT, size: 22 }));
  return new Paragraph({
    numbering: { reference: 'numbers', level: 0 },
    spacing: { before: 60, after: 60 },
    indent: { left: 720, hanging: 360 },
    children: runs
  });
}

function code(text) {
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    shading: { fill: '1E1E1E', type: ShadingType.CLEAR },
    children: [new TextRun({ text, font: 'Courier New', size: 18, color: 'D4D4D4' })]
  });
}

function note(text, bg = WARN_BG, accent = WARN) {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    shading: { fill: bg, type: ShadingType.CLEAR },
    border: { left: { style: BorderStyle.THICK, size: 12, color: accent, space: 8 } },
    indent: { left: 200 },
    children: [new TextRun({ text, font: FONT, size: 21, color: '333333' })]
  });
}

function spacer(pts = 160) {
  return new Paragraph({ spacing: { before: pts, after: 0 }, children: [] });
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function sectionBadge(text, bg, textColor) {
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: [CONTENT_WIDTH],
    rows: [new TableRow({ children: [new TableCell({
      borders: NO_BORDERS,
      shading: { fill: bg, type: ShadingType.CLEAR },
      margins: { top: 100, bottom: 100, left: 160, right: 160 },
      children: [new Paragraph({
        children: [new TextRun({ text, font: FONT, size: 24, bold: true, color: textColor })]
      })]
    })] })]
  });
}

function twoColTable(rows, widths = [2800, 6560]) {
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map(([left, right, bg]) => new TableRow({ children: [
      new TableCell({
        borders: ALL_BORDERS,
        width: { size: widths[0], type: WidthType.DXA },
        shading: { fill: bg || GRAY_BG, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: left, font: FONT, size: 20, bold: true, color: '333333' })] })]
      }),
      new TableCell({
        borders: ALL_BORDERS,
        width: { size: widths[1], type: WidthType.DXA },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: right, font: FONT, size: 20, color: '333333' })] })]
      })
    ]}))
  });
}

function threeColTable(headers, rows, colWidths = [2500, 3500, 3360]) {
  const headerRow = new TableRow({ children: headers.map((h, i) => new TableCell({
    borders: ALL_BORDERS,
    width: { size: colWidths[i], type: WidthType.DXA },
    shading: { fill: HEADER_BG, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({ children: [new TextRun({ text: h, font: FONT, size: 20, bold: true, color: 'FFFFFF' })] })]
  })) });

  const dataRows = rows.map(([c1, c2, c3]) => new TableRow({ children: [
    new TableCell({
      borders: ALL_BORDERS, width: { size: colWidths[0], type: WidthType.DXA },
      shading: { fill: GRAY_BG, type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({ children: [new TextRun({ text: c1, font: FONT, size: 20, bold: true })] })]
    }),
    new TableCell({
      borders: ALL_BORDERS, width: { size: colWidths[1], type: WidthType.DXA },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({ children: [new TextRun({ text: c2, font: FONT, size: 20 })] })]
    }),
    new TableCell({
      borders: ALL_BORDERS, width: { size: colWidths[2], type: WidthType.DXA },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({ children: [new TextRun({ text: c3, font: FONT, size: 20 })] })]
    })
  ]}));

  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows]
  });
}

// ── Document ──────────────────────────────────────────────────────────────────

const doc = new Document({
  numbering: {
    config: [
      { reference: 'bullets', levels: [
        { level: 0, format: LevelFormat.BULLET, text: '\u2022', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        { level: 1, format: LevelFormat.BULLET, text: '\u25E6', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1080, hanging: 360 } } } }
      ]},
      { reference: 'numbers', levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }
      ]}
    ]
  },
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: ACCENT },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: ACCENT },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    children: [

      // ── COVER ──────────────────────────────────────────────────────────────
      new Table({
        width: { size: CONTENT_WIDTH, type: WidthType.DXA },
        columnWidths: [CONTENT_WIDTH],
        rows: [new TableRow({ children: [new TableCell({
          borders: NO_BORDERS,
          shading: { fill: HEADER_BG, type: ShadingType.CLEAR },
          margins: { top: 320, bottom: 320, left: 320, right: 320 },
          children: [
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'MatIntel MVP', font: FONT, size: 52, bold: true, color: 'FFFFFF' })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, children: [new TextRun({ text: 'Materials Intelligence Platform', font: FONT, size: 30, color: 'B8D4F0' })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80 }, children: [new TextRun({ text: 'Full Build Guide — Step-by-Step', font: FONT, size: 24, color: 'B8D4F0' })] }),
          ]
        })] })]
      }),

      spacer(240),

      // Elevator pitch box
      new Table({
        width: { size: CONTENT_WIDTH, type: WidthType.DXA },
        columnWidths: [CONTENT_WIDTH],
        rows: [new TableRow({ children: [new TableCell({
          borders: { top: COL_BORDER, bottom: COL_BORDER, left: { style: BorderStyle.THICK, size: 16, color: ACCENT2 }, right: COL_BORDER },
          shading: { fill: GREEN_BG, type: ShadingType.CLEAR },
          margins: { top: 160, bottom: 160, left: 200, right: 200 },
          children: [
            new Paragraph({ children: [new TextRun({ text: 'What you\'re building', font: FONT, size: 22, bold: true, color: ACCENT2 })] }),
            spacer(60),
            new Paragraph({ children: [new TextRun({ text: 'A web app that takes 520,000+ predicted-stable crystal structures from Google DeepMind\'s GNoME database, runs open-source ML models to predict their key physical properties (band gap, bulk modulus, magnetic moment, ionic character), maps each material to the applications it\'s suited for (batteries, semiconductors, thermoelectrics, magnets, coatings), scores each one for real-world viability based on element cost, earth abundance, and supply chain risk, and presents everything through a clean search interface that any engineer or researcher can use without writing code.', font: FONT, size: 21, color: '333333' })] }),
          ]
        })] })]
      }),

      spacer(120),

      // Summary stats table
      threeColTable(
        ['Component', 'What it is', 'Where it comes from'],
        [
          ['Input dataset', '520K predicted-stable crystals', 'GNoME (Google DeepMind, free download)'],
          ['Starting subset', '33K–38K energy materials', 'Energy-GNoME paper (free, GitHub)'],
          ['Property prediction', 'Band gap, energy, bulk modulus, magnetism', 'CHGNet + matminer (pip install, free)'],
          ['Application mapping', 'Battery / semiconductor / thermoelectric / magnet score', 'Rules engine you build (Python)'],
          ['Viability scoring', 'Cost, abundance, supply chain risk', 'USGS data + pymatgen built-ins'],
          ['Interface', 'Search, filter, explore', 'Streamlit (free deployment)'],
          ['Total storage', '~200 MB (subset) | ~2 GB (full)', 'Your laptop'],
          ['Est. build time', '4–6 weeks solo', 'Python + basic web skills'],
        ],
        [2200, 3960, 3200]
      ),

      pageBreak(),

      // ── PHASE 0 ────────────────────────────────────────────────────────────
      sectionBadge('PHASE 0 — Environment Setup  (Day 1)', HEADER_BG, 'FFFFFF'),
      spacer(120),

      h2('0.1  Prerequisites'),
      body('You need Python 3.10+ and Node.js (optional, only if you later add a Next.js frontend). Everything else installs via pip.'),
      spacer(80),

      h3('Check your Python version'),
      code('python3 --version   # Must be 3.10 or higher'),
      spacer(80),

      h3('Create a virtual environment'),
      code('python3 -m venv matintel-env'),
      code('source matintel-env/bin/activate     # Mac/Linux'),
      code('matintel-env\\Scripts\\activate        # Windows'),
      spacer(80),

      h3('Install all Python dependencies in one shot'),
      code('pip install pymatgen matminer chgnet pandas numpy scikit-learn streamlit tqdm requests'),
      spacer(80),

      twoColTable([
        ['pymatgen', 'Core materials analysis library. Parses CIF files, holds element data, computes properties. Maintained by the Materials Project team.'],
        ['matminer', '70+ featurizers that convert a crystal formula into ML-ready numerical vectors. Built on top of pymatgen.'],
        ['chgnet', 'Pretrained Crystal Hamiltonian Graph Neural Network. Predicts energy, forces, stress, and magnetic moment from crystal structure. Most accurate universal MLIP currently available.'],
        ['pandas', 'Load and manipulate the GNoME CSV summary. Core data wrangling tool.'],
        ['scikit-learn', 'For any ML models you train on top of your feature vectors (band gap prediction, scoring models).'],
        ['streamlit', 'One-command web app framework. Write Python, get a working UI. Free cloud deployment.'],
        ['tqdm', 'Progress bars for batch processing loops (useful when processing 33K+ structures).'],
        ['requests', 'HTTP calls to the Materials Project API if you want to pull additional property data.'],
      ]),

      spacer(120),
      h2('0.2  Get a Materials Project API key'),
      body('You\'ll use this to pull DFT-validated property data for known materials, which you\'ll use to train your scoring models.'),
      spacer(60),
      numbered('Go to: '), // will linkify below
      numberedWithLink('Go to ', 'materialsproject.org', 'https://materialsproject.org', ' and create a free account'),
      numberedWithLink('Navigate to your dashboard and click ', '"Generate API Key"', 'https://materialsproject.org/dashboard'),
      numbered('Set it as an environment variable so you never hardcode it:'),
      code('export MP_API_KEY="your_key_here"   # Add to your .bashrc or .zshrc'),
      spacer(60),
      note('Docs: https://docs.materialsproject.org/downloading-data/using-the-api  |  Free tier: 1,000 requests/hour — more than enough for this project.'),

      pageBreak(),

      // ── PHASE 1 ────────────────────────────────────────────────────────────
      sectionBadge('PHASE 1 — Data Acquisition  (Days 1–3)', HEADER_BG, 'FFFFFF'),
      spacer(120),

      h2('1.1  Download the GNoME dataset'),
      body('The full GNoME dataset lives in a public Google Cloud Storage bucket. You have two options:'),
      spacer(80),

      h3('Option A — CSV summary only (recommended to start)'),
      body('The CSV contains composition, formation energy, decomposition energy, band gap (where available), space group, density, dimensionality, and more for all 520K materials. It\'s all you need for Phase 1 and 2. ~130 MB.'),
      code('# Install gsutil (part of Google Cloud SDK)'),
      code('# https://cloud.google.com/sdk/docs/install'),
      code(''),
      code('gsutil cp gs://gdm_materials_discovery/gnome_data/stable_materials_summary.csv ./data/'),
      spacer(60),
      note('Alternatively, use the Python download script from the GitHub repo — no Google Cloud account required:  github.com/google-deepmind/materials_discovery  →  scripts/download_data_wget.py'),
      spacer(80),

      h3('Option B — Full dataset including CIF structure files'),
      body('You only need CIF files when running CHGNet (which requires 3D atomic positions). For the MVP, download CIFs on-demand for a subset rather than all 520K at once.'),
      code('# Download the zipped CIF files (by_id is most useful)'),
      code('gsutil -m cp gs://gdm_materials_discovery/gnome_data/by_id.zip ./data/'),
      code('unzip data/by_id.zip -d data/cifs/'),
      spacer(60),

      twoColTable([
        ['stable_materials_summary.csv', '~130 MB. Your main working file. Has all metadata and pre-computed properties for all 520K materials. Start here.'],
        ['by_id.zip', '~800 MB compressed. CIF structure files named by unique material ID. Needed for CHGNet inference.'],
        ['by_reduced_formula.zip', '~800 MB. Same CIFs, named by chemical formula (e.g. Li2MnO4.cif). Useful for formula-based lookups.'],
        ['stable_materials_r2scan.csv', 'Additional validation calculations using r2SCAN functional. Optional — use for higher accuracy stability checks.'],
      ]),

      spacer(120),
      h2('1.2  Download the Energy-GNoME subset'),
      body('This is your recommended starting point. The Energy-GNoME paper already screened the full GNoME dataset and extracted 33,000–38,500 materials relevant to energy applications, with predicted suitability scores for thermoelectrics, battery cathodes, and perovskites.'),
      spacer(60),
      bulletLink('GitHub repository: ', 'github.com/paolodeangelis/Energy-GNoME', 'https://github.com/paolodeangelis/Energy-GNoME'),
      bulletLink('Paper (arXiv): ', 'arxiv.org/abs/2411.10125', 'https://arxiv.org/abs/2411.10125'),
      bulletLink('Interactive database: ', 'paolodeangelis.github.io/Energy-GNoME', 'https://paolodeangelis.github.io/Energy-GNoME'),
      spacer(60),
      code('git clone https://github.com/paolodeangelis/Energy-GNoME.git'),
      code('# The dataset CSV files are in the data/ directory of the repo'),
      spacer(60),
      note('This subset already has predicted thermoelectric figure-of-merit (zT), band gap (Eg), and cathode voltage for relevant candidates. Reading through the paper\'s methodology will save you significant work in Phase 3.'),

      spacer(120),
      h2('1.3  Download element cost and abundance data'),
      body('USGS publishes annual price and production data for 90+ minerals as downloadable CSVs. This is what you use to price-score each material.'),
      spacer(60),
      bulletLink('USGS Mineral Commodity Summaries 2025 data release: ', 'usgs.gov/data/...mcs2025', 'https://www.usgs.gov/data/us-geological-survey-mineral-commodity-summaries-2025-data-release-ver-20-april-2025'),
      bulletLink('Direct PDF (all commodity summaries): ', 'pubs.usgs.gov/publication/mcs2025', 'https://pubs.usgs.gov/publication/mcs2025'),
      spacer(60),
      note('pymatgen\'s built-in Element class already has crustal abundance data for every element. You only need the USGS CSV for price data. Call Element("Co").abundance to get cobalt\'s crustal abundance in parts per million — no separate download required for that field.'),

      pageBreak(),

      // ── PHASE 2 ────────────────────────────────────────────────────────────
      sectionBadge('PHASE 2 — Data Loading & Exploration  (Days 3–6)', HEADER_BG, 'FFFFFF'),
      spacer(120),

      h2('2.1  Load and inspect the GNoME CSV'),
      body('Start here before touching any CIF files or running any models. Get familiar with the data shape.'),
      spacer(60),
      code('import pandas as pd'),
      code(''),
      code('df = pd.read_csv("data/stable_materials_summary.csv")'),
      code('print(df.shape)            # Should be ~520,000 rows'),
      code('print(df.columns.tolist()) # See all available fields'),
      code('print(df.head(3))'),
      spacer(60),

      body('Key columns you\'ll work with most:'),
      twoColTable([
        ['Composition', 'Full elemental composition string, e.g. "Li2 Mn1 O4"'],
        ['Reduced Formula', 'Simplified formula, e.g. "Li2MnO4"'],
        ['Formation Energy Per Atom', 'How stable the material is. More negative = more stable. In eV/atom.'],
        ['Decomposition Energy Per Atom', 'Distance to the convex hull. Closer to 0 = more likely to be synthesizable.'],
        ['Bandgap', 'PBE-level band gap in eV. Available for a subset. 0 = metal, >0 = semiconductor/insulator.'],
        ['Dimensionality Cheon', 'Predicted dimensionality: 3D bulk, 2D layered, 1D chain, 0D molecular. Useful for filtering.'],
        ['NSites', 'Number of atoms in the unit cell. Larger = more complex structure.'],
        ['Space Group Number', 'Crystal symmetry. Useful for filtering by crystal system.'],
      ]),

      spacer(120),
      h2('2.2  Load a CIF file with pymatgen'),
      body('This is how you get a Structure object that CHGNet and matminer featurizers can work with.'),
      spacer(60),
      code('from pymatgen.core import Structure'),
      code(''),
      code('# Load one structure'),
      code('structure = Structure.from_file("data/cifs/gnome-12345.cif")'),
      code(''),
      code('# Inspect it'),
      code('print(structure.formula)        # e.g. "Li2 Mn1 O4"'),
      code('print(structure.lattice)        # Unit cell dimensions'),
      code('print(len(structure))           # Number of atoms'),
      code('print(structure.get_space_group_info())  # Space group'),

      spacer(120),
      h2('2.3  Build your master dataframe'),
      body('For the MVP, filter the CSV down to the Energy-GNoME subset and add a file path column pointing to each material\'s CIF file. This becomes your working dataset.'),
      spacer(60),
      code('import pandas as pd'),
      code('import os'),
      code(''),
      code('# Load full GNoME summary'),
      code('df_all = pd.read_csv("data/stable_materials_summary.csv")'),
      code(''),
      code('# Load Energy-GNoME IDs (from the cloned repo)'),
      code('df_energy = pd.read_csv("Energy-GNoME/data/energy_gnome_ids.csv")'),
      code(''),
      code('# Filter to energy subset'),
      code('df = df_all[df_all["MaterialId"].isin(df_energy["MaterialId"])].copy()'),
      code(''),
      code('# Add CIF file path'),
      code('df["cif_path"] = df["MaterialId"].apply('),
      code('    lambda mid: f"data/cifs/{mid}.cif" if os.path.exists(f"data/cifs/{mid}.cif") else None'),
      code(')'),
      code(''),
      code('print(f"Working dataset: {len(df)} materials")'),
      code('df.to_csv("data/working_dataset.csv", index=False)'),

      pageBreak(),

      // ── PHASE 3 ────────────────────────────────────────────────────────────
      sectionBadge('PHASE 3 — Property Prediction  (Days 6–14)', HEADER_BG, 'FFFFFF'),
      spacer(120),

      h2('3.1  Extract compositional features with matminer'),
      body('These features work on chemical formulas — you don\'t need CIF files for this step. Run this on all 33K+ materials in your working dataset.'),
      spacer(60),
      body('Reference: ', { bold: true }),
      bulletLink('matminer docs: ', 'hackingmaterials.lbl.gov/matminer', 'https://hackingmaterials.lbl.gov/matminer/'),
      bulletLink('Full featurizer table: ', 'hackingmaterials.lbl.gov/matminer/featurizer_summary.html', 'https://hackingmaterials.lbl.gov/matminer/featurizer_summary.html'),
      spacer(60),
      code('from matminer.featurizers.composition import ElementProperty'),
      code('from matminer.featurizers.conversions import StrToComposition'),
      code('import pandas as pd'),
      code('from tqdm import tqdm'),
      code(''),
      code('df = pd.read_csv("data/working_dataset.csv")'),
      code(''),
      code('# Step 1: Convert formula string to pymatgen Composition object'),
      code('stc = StrToComposition()'),
      code('df = stc.featurize_dataframe(df, "Reduced Formula", ignore_errors=True)'),
      code(''),
      code('# Step 2: Generate 132 element-property features per material'),
      code('# This uses data like electronegativity, atomic radius, ionization energy'),
      code('ep = ElementProperty.from_preset("magpie")'),
      code('df = ep.featurize_dataframe(df, "composition", ignore_errors=True)'),
      code(''),
      code('# Save — this is your feature matrix'),
      code('df.to_csv("data/featured_dataset.csv", index=False)'),
      code('print(f"Feature columns: {len(ep.feature_labels())}")  # ~132 features'),
      spacer(60),
      note('The ElementProperty "magpie" preset gives you 132 features per material: mean, max, min, range, and variance of ~26 elemental properties across all elements in the formula. This takes about 5–10 minutes to run on 33K materials on a laptop. Use ignore_errors=True to skip any malformed entries.'),

      spacer(120),
      h2('3.2  Run CHGNet for energy and magnetic predictions'),
      body('CHGNet requires 3D crystal structures (CIF files), not just formulas. It predicts energy per atom, forces, stress tensor, and magnetic moments. This step is optional for the MVP but adds significant predictive power.'),
      spacer(60),
      body('Reference: ', { bold: true }),
      bulletLink('CHGNet docs: ', 'chgnet.lbl.gov', 'https://chgnet.lbl.gov'),
      bulletLink('GitHub: ', 'github.com/CederGroupHub/chgnet', 'https://github.com/CederGroupHub/chgnet'),
      bulletLink('Paper (Nature Machine Intelligence): ', 'nature.com/articles/s42256-023-00716-3', 'https://www.nature.com/articles/s42256-023-00716-3'),
      spacer(60),
      code('from chgnet.model.model import CHGNet'),
      code('from pymatgen.core import Structure'),
      code('import pandas as pd, json, os'),
      code('from tqdm import tqdm'),
      code(''),
      code('chgnet = CHGNet.load()  # Downloads ~20 MB weights on first run'),
      code(''),
      code('df = pd.read_csv("data/working_dataset.csv")'),
      code('results = []'),
      code(''),
      code('for _, row in tqdm(df.iterrows(), total=len(df)):'),
      code('    if pd.isna(row["cif_path"]) or not os.path.exists(row["cif_path"]):'),
      code('        results.append({"MaterialId": row["MaterialId"]})'),
      code('        continue'),
      code('    try:'),
      code('        struct = Structure.from_file(row["cif_path"])'),
      code('        pred = chgnet.predict_structure(struct)'),
      code('        results.append({'),
      code('            "MaterialId": row["MaterialId"],'),
      code('            "chgnet_energy_ev_atom": float(pred["e"]),'),
      code('            "chgnet_magmom_mean": float(pred["m"].mean()) if pred["m"] is not None else 0.0,'),
      code('            "chgnet_magmom_max": float(pred["m"].max()) if pred["m"] is not None else 0.0,'),
      code('        })'),
      code('    except Exception as e:'),
      code('        results.append({"MaterialId": row["MaterialId"], "error": str(e)})'),
      code(''),
      code('df_chgnet = pd.DataFrame(results)'),
      code('df_chgnet.to_csv("data/chgnet_predictions.csv", index=False)'),
      spacer(60),
      note('Runtime estimate: ~0.5–2 seconds per structure on CPU, ~1–5 hours for 33K materials. Run overnight or batch it. CHGNet is the most accurate universal potential available — prefer it over M3GNet for this use case.'),

      spacer(120),
      h2('3.3  Pull additional properties from Materials Project'),
      body('Use the MP API to enrich your dataset with DFT-validated properties for any GNoME materials that overlap with known compounds.'),
      spacer(60),
      code('from mp_api.client import MPRester'),
      code('import os, pandas as pd'),
      code(''),
      code('df = pd.read_csv("data/working_dataset.csv")'),
      code('formulas = df["Reduced Formula"].unique().tolist()'),
      code(''),
      code('with MPRester(os.environ["MP_API_KEY"]) as mpr:'),
      code('    docs = mpr.materials.summary.search('),
      code('        formula=formulas[:1000],  # batch in groups of 1000'),
      code('        fields=["material_id", "formula_pretty", "band_gap",'),
      code('                "bulk_modulus", "energy_above_hull", "is_stable"]'),
      code('    )'),
      code(''),
      code('df_mp = pd.DataFrame([d.dict() for d in docs])'),
      code('df_mp.to_csv("data/mp_enrichment.csv", index=False)'),
      spacer(60),
      note('MP API docs: https://docs.materialsproject.org/downloading-data/using-the-api  |  The mp_api package installs via: pip install mp-api'),

      pageBreak(),

      // ── PHASE 4 ────────────────────────────────────────────────────────────
      sectionBadge('PHASE 4 — Application Mapping  (Days 14–21)', HEADER_BG, 'FFFFFF'),
      spacer(120),

      h2('4.1  Define application domain requirements'),
      body('Each application domain has known property requirements. You encode these as threshold rules to score each material. These are well-established in the literature — they\'re not arbitrary.'),
      spacer(80),

      h3('Battery cathode', ACCENT),
      twoColTable([
        ['Band gap', 'Low or zero (good electronic conductivity). Target: < 2.0 eV'],
        ['Ionic character', 'High — ionic compounds conduct Li+ ions. Measure via electronegativity difference.'],
        ['Transition metals', 'Must contain at least one: Mn, Fe, Co, Ni, V — enables redox reactions.'],
        ['Structural stability', 'Low decomposition energy (close to convex hull). Target: < 0.05 eV/atom'],
        ['Avoid', 'Platinum group metals, mercury, cadmium (toxicity / cost)'],
      ]),

      spacer(100),
      h3('Semiconductor / Solar cell', ACCENT),
      twoColTable([
        ['Band gap', 'Sweet spot: 0.9–1.8 eV for photovoltaics; 0.5–3.5 eV for general semiconductors'],
        ['Crystal quality proxy', 'Regular crystal system (cubic, tetragonal preferred). Check Crystal System column.'],
        ['Earth abundant', 'No rare earths, no platinum group. Check element abundance.'],
        ['Stability', 'Decomposition energy < 0.02 eV/atom (very stable)'],
        ['Benchmark', 'Silicon band gap = 1.1 eV. Perovskites: 1.5–1.7 eV is ideal for solar.'],
      ]),

      spacer(100),
      h3('Thermoelectric material', ACCENT),
      twoColTable([
        ['Band gap', 'Narrow: 0.1–0.5 eV (need both electrons and holes)'],
        ['Heavy elements', 'Higher atomic mass correlates with lower lattice thermal conductivity. Prefer Bi, Pb, Te, Sb.'],
        ['Crystal complexity', 'More complex unit cells tend to have lower thermal conductivity. NSites > 10 is a positive signal.'],
        ['Magnetic moment', 'Low — non-magnetic materials tend to have better thermoelectric performance.'],
        ['Benchmark', 'Best thermoelectrics: Bi2Te3 (zT ~ 1). Your model: predict zT proxy from above features.'],
      ]),

      spacer(100),
      h3('Permanent magnet', ACCENT),
      twoColTable([
        ['Magnetic elements', 'Must contain Fe, Co, Ni, Nd, Sm, Gd, or Dy'],
        ['Magnetic moment', 'High CHGNet-predicted mean magnetic moment. Target: > 1.5 μB/atom'],
        ['Crystal anisotropy', 'Non-cubic crystal system preferred (hexagonal, tetragonal)'],
        ['Curie temperature proxy', 'Materials with stronger exchange interactions tend to have higher Tc. Use magnetic moment as proxy.'],
        ['Benchmark', 'Nd2Fe14B is the gold standard. Your goal: find alternatives with less Nd (expensive/supply risk).'],
      ]),

      spacer(100),
      h3('Structural coating / hard material', ACCENT),
      twoColTable([
        ['Bulk modulus', 'High. Target: > 150 GPa. Pull from MP or predict with matminer.'],
        ['Formation energy', 'Very negative (very stable). Target: < -1.5 eV/atom'],
        ['Chemical stability', 'No easily oxidized elements (alkali metals). Check element groups.'],
        ['Density', 'Depends on use case — low for aerospace coatings, less important for wear-resistance.'],
      ]),

      spacer(120),
      h2('4.2  Implement the scoring engine'),
      body('Each material gets a score between 0 and 1 for each application domain. The score is a weighted sum of normalized sub-scores.'),
      spacer(60),
      code('import pandas as pd'),
      code('import numpy as np'),
      code('from pymatgen.core import Composition, Element'),
      code(''),
      code('df = pd.read_csv("data/featured_dataset.csv")'),
      code(''),
      code('TRANSITION_METALS = {"Mn","Fe","Co","Ni","V","Cr","Cu","Ti","Zr"}'),
      code('MAGNETIC_ELEMENTS = {"Fe","Co","Ni","Nd","Sm","Gd","Dy"}'),
      code('EXPENSIVE = {"Pt","Pd","Rh","Ir","Ru","Os","Au","Re"}'),
      code('TOXIC = {"Hg","Cd","Pb","As","Tl"}'),
      code(''),
      code('def get_elements(formula):'),
      code('    try: return set(str(e) for e in Composition(formula).elements)'),
      code('    except: return set()'),
      code(''),
      code('def score_battery(row):'),
      code('    score = 0.0'),
      code('    bg = row.get("Bandgap", np.nan)'),
      code('    if not np.isnan(bg):'),
      code('        score += 0.3 * max(0, 1 - bg / 3.0)   # low band gap better'),
      code('    elems = get_elements(row["Reduced Formula"])'),
      code('    if elems & TRANSITION_METALS: score += 0.3'),
      code('    if not elems & EXPENSIVE:     score += 0.2'),
      code('    if not elems & TOXIC:         score += 0.2'),
      code('    de = row.get("Decomposition Energy Per Atom", np.nan)'),
      code('    if not np.isnan(de):'),
      code('        score += 0.0 if de > 0.1 else 0.1 * (1 - de / 0.1)'),
      code('    return round(min(score, 1.0), 3)'),
      code(''),
      code('def score_semiconductor(row):'),
      code('    score = 0.0'),
      code('    bg = row.get("Bandgap", np.nan)'),
      code('    if not np.isnan(bg) and 0.5 <= bg <= 3.5:'),
      code('        # Peak score at 1.4 eV (ideal for solar)'),
      code('        score += 0.4 * (1 - abs(bg - 1.4) / 2.1)'),
      code('    elems = get_elements(row["Reduced Formula"])'),
      code('    if not elems & EXPENSIVE: score += 0.3'),
      code('    if not elems & TOXIC:     score += 0.2'),
      code('    if row.get("Dimensionality Cheon") == 3: score += 0.1'),
      code('    return round(min(score, 1.0), 3)'),
      code(''),
      code('# Apply all scorers'),
      code('df["score_battery"]       = df.apply(score_battery, axis=1)'),
      code('df["score_semiconductor"] = df.apply(score_semiconductor, axis=1)'),
      code('# Add score_thermoelectric, score_magnet, score_structural similarly'),
      code(''),
      code('df.to_csv("data/scored_dataset.csv", index=False)'),

      pageBreak(),

      // ── PHASE 5 ────────────────────────────────────────────────────────────
      sectionBadge('PHASE 5 — Viability Scoring  (Days 18–24)', HEADER_BG, 'FFFFFF'),
      spacer(120),

      h2('5.1  Element cost scoring'),
      body('Download the USGS MCS 2025 data and build a price lookup table. Then compute a weighted average material cost based on stoichiometry.'),
      spacer(60),
      code('# Build your element price dict manually from USGS MCS 2025 PDF'),
      code('# Or scrape the data CSVs from the USGS data release'),
      code('# Key prices to look up (USD/kg, approximate 2024 values):'),
      code(''),
      code('ELEMENT_PRICE_USD_KG = {'),
      code('    "Li": 6.0,    "Co": 33.0,   "Ni": 14.0,   "Mn": 2.0,'),
      code('    "Fe": 0.1,    "Cu": 8.5,    "Al": 2.0,    "Si": 2.5,'),
      code('    "Ti": 11.0,   "V":  30.0,   "Cr": 9.0,    "Zn": 2.5,'),
      code('    "Ga": 220.0,  "Ge": 1000.0, "Se": 21.0,   "Nb": 42.0,'),
      code('    "Mo": 40.0,   "In": 167.0,  "Sn": 26.0,   "Sb": 6.5,'),
      code('    "Te": 63.0,   "Cs": 72000., "Ba": 0.3,    "La": 2.0,'),
      code('    "Ce": 2.0,    "Nd": 40.0,   "Sm": 14.0,   "Gd": 28.0,'),
      code('    "Dy": 220.0,  "Yb": 14.0,   "Hf": 900.0,  "W":  35.0,'),
      code('    "Re": 3500.0, "Pt": 31000., "Pd": 49000., "Rh":147000.'),
      code('}'),
      code(''),
      code('from pymatgen.core import Composition'),
      code(''),
      code('def material_cost_score(formula, price_dict, max_cost=100):'),
      code('    try:'),
      code('        comp = Composition(formula)'),
      code('        total_weight = sum('),
      code('            comp[el] * el.atomic_mass for el in comp.elements'),
      code('        )'),
      code('        weighted_cost = sum('),
      code('            (comp[el] * el.atomic_mass / total_weight) *'),
      code('            price_dict.get(str(el), 50)   # default $50/kg if unknown'),
      code('            for el in comp.elements'),
      code('        )'),
      code('        # Score: 1.0 = very cheap (<$2/kg), 0.0 = very expensive (>$100/kg)'),
      code('        return round(max(0, 1 - weighted_cost / max_cost), 3)'),
      code('    except:'),
      code('        return 0.5'),

      spacer(120),
      h2('5.2  Earth abundance scoring'),
      body('pymatgen has crustal abundance data built directly into the Element class — no external data file needed.'),
      spacer(60),
      code('from pymatgen.core import Element, Composition'),
      code(''),
      code('# pymatgen abundance is in parts per million (ppm) of Earth\'s crust'),
      code('print(Element("Si").abundance)  # ~282,000 ppm — very abundant'),
      code('print(Element("Co").abundance)  # ~25 ppm — somewhat scarce'),
      code('print(Element("Te").abundance)  # ~0.001 ppm — very rare'),
      code(''),
      code('def abundance_score(formula):'),
      code('    try:'),
      code('        comp = Composition(formula)'),
      code('        # Use minimum abundance (bottleneck element)'),
      code('        min_abund = min('),
      code('            Element(str(el)).abundance or 0.001'),
      code('            for el in comp.elements'),
      code('        )'),
      code('        # Log scale: abundance ranges from ~0.001 to 282,000 ppm'),
      code('        import math'),
      code('        score = math.log10(min_abund + 0.001) / math.log10(282000)'),
      code('        return round(max(0, min(score, 1.0)), 3)'),
      code('    except:'),
      code('        return 0.5'),

      spacer(120),
      h2('5.3  Supply chain risk scoring'),
      body('Flag materials that contain elements on the US Critical Minerals List — these face geopolitical supply risk regardless of abundance or price.'),
      spacer(60),
      code('# US Critical Minerals List 2022 (updated periodically)'),
      code('# Full list: minerals.usgs.gov/minerals/pubs/mcs/2022/mcs2022.pdf  (Appendix C)'),
      code('CRITICAL_MINERALS_2022 = {'),
      code('    "Al","Sb","Ar","As","Ba","Be","Bi","Ce","Cs","Cr","Co","Dy","Er",'),
      code('    "Eu","Fl","Gd","Ga","Ge","Hf","Ho","In","Ir","La","Li","Lu","Mg",'),
      code('    "Mn","Nd","Ni","Nb","Pr","Rh","Rb","Ru","Sm","Sc","Ta","Te","Tb",'),
      code('    "Tl","Tm","Sn","Ti","W","V","Yb","Y","Zn","Zr"'),
      code('}'),
      code(''),
      code('def supply_risk_score(formula):'),
      code('    """Returns 1.0 = no risk, 0.0 = all critical minerals"""'),
      code('    try:'),
      code('        elems = set(str(e) for e in Composition(formula).elements)'),
      code('        critical_count = len(elems & CRITICAL_MINERALS_2022)'),
      code('        return round(1 - critical_count / max(len(elems), 1), 3)'),
      code('    except:'),
      code('        return 0.5'),
      spacer(60),
      note('Critical minerals list source: https://www.usgs.gov/centers/national-minerals-information-center/critical-minerals  |  Updated by USGS periodically — check for updates annually.'),

      spacer(120),
      h2('5.4  Combine into a final viability score'),
      code('def viability_score(row):'),
      code('    cost    = material_cost_score(row["Reduced Formula"], ELEMENT_PRICE_USD_KG)'),
      code('    abund   = abundance_score(row["Reduced Formula"])'),
      code('    supply  = supply_risk_score(row["Reduced Formula"])'),
      code('    # Weighted: cost matters most, then abundance, then supply risk'),
      code('    return round(0.45 * cost + 0.35 * abund + 0.20 * supply, 3)'),
      code(''),
      code('df["viability"] = df.apply(viability_score, axis=1)'),

      pageBreak(),

      // ── PHASE 6 ────────────────────────────────────────────────────────────
      sectionBadge('PHASE 6 — AI Explanations  (Days 22–26)', HEADER_BG, 'FFFFFF'),
      spacer(120),

      h2('6.1  Generate plain-English material summaries with Claude'),
      body('For each material a user clicks on, generate a 3–5 sentence plain-English explanation of why it scored well for a given application, what its trade-offs are, and what the next experimental steps would be. This is the "intelligence" layer that makes the tool genuinely useful to non-experts.'),
      spacer(60),
      note('Get your Anthropic API key at: console.anthropic.com  |  pip install anthropic'),
      spacer(60),
      code('import anthropic, os'),
      code(''),
      code('client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])'),
      code(''),
      code('def generate_material_summary(row, top_application):'),
      code('    prompt = f"""You are a materials scientist. Explain this material to a product engineer.'),
      code(''),
      code('Material: {row["Reduced Formula"]}'),
      code('Formation energy: {row.get("Formation Energy Per Atom", "unknown")} eV/atom'),
      code('Band gap: {row.get("Bandgap", "unknown")} eV'),
      code('Top application: {top_application}'),
      code('Application score: {row.get(f"score_{top_application}", "unknown")}'),
      code('Viability score: {row.get("viability", "unknown")}'),
      code(''),
      code('Write 3 sentences:'),
      code('1. Why this material is promising for {top_application} (cite its specific properties)'),
      code('2. Its main limitation or trade-off'),
      code('3. What a researcher would need to verify next in the lab'),
      code(''),
      code('Use plain English. No jargon. Be specific about numbers."""'),
      code(''),
      code('    response = client.messages.create('),
      code('        model="claude-sonnet-4-20250514",'),
      code('        max_tokens=300,'),
      code('        messages=[{"role": "user", "content": prompt}]'),
      code('    )'),
      code('    return response.content[0].text'),
      spacer(60),
      note('Call this only when a user clicks on a specific material — not in batch. At ~$0.003 per call, generating summaries on-demand for clicked materials costs fractions of a cent each. Do not pre-generate for all 33K materials.'),

      pageBreak(),

      // ── PHASE 7 ────────────────────────────────────────────────────────────
      sectionBadge('PHASE 7 — Streamlit Interface  (Days 24–30)', HEADER_BG, 'FFFFFF'),
      spacer(120),

      h2('7.1  App structure'),
      body('Build the app as a single Python file: app.py. Streamlit handles all the UI, routing, and state. You can deploy it to Streamlit Cloud for free.'),
      spacer(60),
      code('# app.py — full structure'),
      code('import streamlit as st'),
      code('import pandas as pd'),
      code('import numpy as np'),
      code('from pymatgen.core import Element'),
      code(''),
      code('st.set_page_config(page_title="MatIntel", layout="wide")'),
      code(''),
      code('@st.cache_data'),
      code('def load_data():'),
      code('    return pd.read_csv("data/scored_dataset.csv")'),
      code(''),
      code('df = load_data()'),
      code(''),
      code('# ── Sidebar filters ─────────────────────────────────────────────'),
      code('st.sidebar.header("Filter Materials")'),
      code(''),
      code('application = st.sidebar.selectbox('),
      code('    "Target application",'),
      code('    ["Battery cathode", "Semiconductor", "Thermoelectric", "Magnet", "Coating"]'),
      code(')'),
      code(''),
      code('score_col = {'),
      code('    "Battery cathode": "score_battery",'),
      code('    "Semiconductor": "score_semiconductor",'),
      code('    "Thermoelectric": "score_thermoelectric",'),
      code('    "Magnet": "score_magnet",'),
      code('    "Coating": "score_structural",'),
      code('}[application]'),
      code(''),
      code('min_score  = st.sidebar.slider("Min application score", 0.0, 1.0, 0.5)'),
      code('min_viab   = st.sidebar.slider("Min viability score",   0.0, 1.0, 0.4)'),
      code('exclude_critical = st.sidebar.checkbox("Exclude US critical minerals", True)'),
      code('max_bg     = st.sidebar.slider("Max band gap (eV)", 0.0, 5.0, 3.0)'),
      code(''),
      code('# ── Apply filters ───────────────────────────────────────────────'),
      code('filtered = df['),
      code('    (df[score_col] >= min_score) &'),
      code('    (df["viability"] >= min_viab)'),
      code('].copy()'),
      code(''),
      code('if exclude_critical:'),
      code('    # Filter implemented with supply_risk_score > 0.5'),
      code('    filtered = filtered[filtered["supply_risk"] > 0.5]'),
      code(''),
      code('if "Bandgap" in filtered.columns:'),
      code('    filtered = filtered[filtered["Bandgap"].fillna(0) <= max_bg]'),
      code(''),
      code('filtered = filtered.sort_values(score_col, ascending=False)'),
      code(''),
      code('# ── Main panel ──────────────────────────────────────────────────'),
      code('st.title("MatIntel — Materials Intelligence Platform")'),
      code('st.write(f"Showing {len(filtered):,} materials matching your criteria")'),
      code(''),
      code('# Results table'),
      code('display_cols = ["Reduced Formula", score_col, "viability",'),
      code('                "Bandgap", "Formation Energy Per Atom",'),
      code('                "Crystal System", "NSites"]'),
      code('st.dataframe('),
      code('    filtered[display_cols].head(100),'),
      code('    use_container_width=True'),
      code(')'),
      code(''),
      code('# ── Material detail card ────────────────────────────────────────'),
      code('selected = st.selectbox("Select a material for details", filtered["Reduced Formula"].head(50))'),
      code(''),
      code('if selected:'),
      code('    row = filtered[filtered["Reduced Formula"] == selected].iloc[0]'),
      code('    col1, col2, col3 = st.columns(3)'),
      code('    col1.metric("Application score", f\'{row[score_col]:.2f}\')" '),
      code('    col2.metric("Viability score",   f\'{row["viability"]:.2f}\')" '),
      code('    col3.metric("Band gap (eV)",      f\'{row.get("Bandgap", "N/A")}\')" '),
      code('    '),
      code('    if st.button("Generate AI explanation"):'),
      code('        with st.spinner("Asking Claude..."):'),
      code('            summary = generate_material_summary(row, application)'),
      code('            st.info(summary)'),

      spacer(120),
      h2('7.2  Deploy to Streamlit Cloud'),
      body('Streamlit Cloud hosts public apps for free. No server setup, no Docker, no cloud configuration.'),
      spacer(60),
      numberedWithLink('Push your code to a public GitHub repo', 'github.com/new', 'https://github.com/new'),
      numberedWithLink('Go to ', 'share.streamlit.io', 'https://share.streamlit.io', ' and connect your GitHub'),
      numbered('Select your repo and set the main file as app.py'),
      numbered('Add your environment secrets (MP_API_KEY, ANTHROPIC_API_KEY) in the Streamlit Cloud secrets manager'),
      numbered('Deploy — live in ~2 minutes'),
      spacer(60),
      note('For the MVP, include the processed scored_dataset.csv directly in your repo (it\'s only ~30–50 MB). For the full 520K dataset, use a database — Supabase free tier (500 MB) or PlanetScale free tier work well.'),

      pageBreak(),

      // ── PHASE 8 ────────────────────────────────────────────────────────────
      sectionBadge('PHASE 8 — Testing & Validation  (Days 28–35)', HEADER_BG, 'FFFFFF'),
      spacer(120),

      h2('8.1  Sanity checks'),
      body('Before shipping, verify your pipeline produces sensible results by checking known materials against their published properties.'),
      spacer(60),
      twoColTable([
        ['Si (Silicon)', 'Band gap ~1.1 eV (semiconductor). Your pipeline should give semiconductor score > 0.6'],
        ['LiCoO2', 'Classic Li-ion battery cathode. Battery score should be > 0.7. High cost score (cobalt).'],
        ['Fe (Iron)', 'Magnetic moment > 2 μB/atom. Magnet score should be high. Band gap = 0 (metal).'],
        ['Bi2Te3', 'Best thermoelectric known. zT ~ 1. Thermoelectric score should be your highest for this formula.'],
        ['Diamond (C)', 'Band gap ~5.5 eV (insulator). Semiconductor score should be low. High bulk modulus.'],
        ['Au (Gold)', 'Very expensive ($60,000/kg). Cost score should be near 0. Viability score should be low.'],
      ]),

      spacer(120),
      h2('8.2  Known limitations to document'),
      bullet('Band gap predictions from matminer features are rough estimates (±0.5 eV). The GNoME CSV has PBE-level band gaps for a subset — these are available but ~30% underestimated vs. experiment (PBE DFT systematic error).'),
      bullet('Synthesizability is not scored in the MVP. A material being thermodynamically stable doesn\'t mean it can be made in a lab. Add a note to the UI.'),
      bullet('CHGNet predictions are most reliable for materials compositionally similar to its training set (Materials Project). Novel chemistries may have higher error.'),
      bullet('Cost scores use 2024 USGS price data. Prices fluctuate significantly — lithium dropped 80% from 2022 to 2024. Build in a data update mechanism.'),
      bullet('The application thresholds are first-order approximations. Battery scientists will have more nuanced criteria. This is a screening tool, not a replacement for expert judgment.'),

      pageBreak(),

      // ── REFERENCE ──────────────────────────────────────────────────────────
      sectionBadge('REFERENCE — All Links, Packages, and Resources', HEADER_BG, 'FFFFFF'),
      spacer(120),

      h2('Core datasets'),
      twoColTable([
        ['GNoME GitHub', 'github.com/google-deepmind/materials_discovery'],
        ['GNoME Cloud Bucket', 'gs://gdm_materials_discovery  (gsutil or wget scripts)'],
        ['Energy-GNoME GitHub', 'github.com/paolodeangelis/Energy-GNoME'],
        ['Energy-GNoME Paper', 'arxiv.org/abs/2411.10125'],
        ['Energy-GNoME Explorer', 'paolodeangelis.github.io/Energy-GNoME'],
        ['Materials Project', 'materialsproject.org  (free API key from dashboard)'],
        ['USGS MCS 2025', 'pubs.usgs.gov/publication/mcs2025'],
        ['USGS Critical Minerals', 'minerals.usgs.gov/minerals/pubs/mcs/2022/mcs2022.pdf'],
      ]),

      spacer(100),
      h2('Python packages'),
      twoColTable([
        ['pymatgen', 'pip install pymatgen  |  pymatgen.org'],
        ['matminer', 'pip install matminer  |  hackingmaterials.lbl.gov/matminer'],
        ['chgnet', 'pip install chgnet  |  chgnet.lbl.gov'],
        ['mp-api', 'pip install mp-api  |  docs.materialsproject.org'],
        ['anthropic', 'pip install anthropic  |  docs.anthropic.com'],
        ['streamlit', 'pip install streamlit  |  docs.streamlit.io'],
        ['pandas', 'pip install pandas'],
        ['scikit-learn', 'pip install scikit-learn'],
        ['tqdm', 'pip install tqdm'],
      ]),

      spacer(100),
      h2('Key papers to read'),
      twoColTable([
        ['GNoME (DeepMind, 2023)', 'nature.com/articles/s41586-023-06735-9'],
        ['CHGNet (Deng et al., 2023)', 'nature.com/articles/s42256-023-00716-3'],
        ['MatGL library (2025)', 'nature.com/articles/s41524-025-01742-y'],
        ['matminer (Ward et al., 2018)', 'doi.org/10.1016/j.commatsci.2018.05.018'],
        ['Energy-GNoME (2024)', 'arxiv.org/abs/2411.10125'],
        ['Matbench benchmark', 'matbench.materialsproject.org'],
      ]),

      spacer(100),
      h2('Deployment'),
      twoColTable([
        ['Streamlit Cloud (free)', 'share.streamlit.io'],
        ['Streamlit docs', 'docs.streamlit.io'],
        ['Supabase (free DB tier)', 'supabase.com  — 500 MB PostgreSQL'],
        ['Anthropic Console', 'console.anthropic.com  — API keys & billing'],
        ['MP Dashboard', 'materialsproject.org/dashboard  — API key'],
      ]),

      spacer(120),
      h2('Suggested folder structure'),
      code('matintel/'),
      code('├── data/'),
      code('│   ├── stable_materials_summary.csv   # GNoME full CSV (download)'),
      code('│   ├── working_dataset.csv             # Your filtered subset'),
      code('│   ├── featured_dataset.csv            # + matminer features'),
      code('│   ├── chgnet_predictions.csv          # + CHGNet output'),
      code('│   └── scored_dataset.csv              # Final scored dataset'),
      code('├── cifs/                               # CIF files (download on demand)'),
      code('├── notebooks/'),
      code('│   ├── 01_explore.ipynb'),
      code('│   ├── 02_features.ipynb'),
      code('│   ├── 03_scoring.ipynb'),
      code('│   └── 04_validation.ipynb'),
      code('├── app.py                              # Streamlit app'),
      code('├── scoring.py                          # Application scoring functions'),
      code('├── viability.py                        # Cost/abundance/supply functions'),
      code('├── explanations.py                     # Claude API calls'),
      code('├── requirements.txt'),
      code('└── README.md'),

      spacer(100),
      h2('requirements.txt'),
      code('pymatgen>=2024.6.10'),
      code('matminer>=0.9.2'),
      code('chgnet>=0.3.0'),
      code('mp-api>=0.41.2'),
      code('anthropic>=0.34.0'),
      code('streamlit>=1.38.0'),
      code('pandas>=2.1.0'),
      code('numpy>=1.26.0'),
      code('scikit-learn>=1.5.0'),
      code('tqdm>=4.66.0'),
      code('requests>=2.31.0'),

      spacer(120),

      // Timeline table
      h2('Build timeline overview'),
      threeColTable(
        ['Phase', 'What you build', 'Time estimate'],
        [
          ['0 — Setup', 'Python env, API keys, folder structure', 'Day 1  (~2 hrs)'],
          ['1 — Data', 'Download GNoME CSV, Energy-GNoME, USGS data', 'Days 1–3  (~3 hrs)'],
          ['2 — Loading', 'Parse CSV, load CIFs, build working dataset', 'Days 3–6  (~4 hrs)'],
          ['3 — Properties', 'matminer features, CHGNet inference (overnight)', 'Days 6–14  (~6 hrs active)'],
          ['4 — App mapping', 'Scoring engine for 5 application domains', 'Days 14–21  (~8 hrs)'],
          ['5 — Viability', 'Cost, abundance, supply chain scoring', 'Days 18–24  (~5 hrs)'],
          ['6 — AI layer', 'Claude API integration for summaries', 'Days 22–26  (~3 hrs)'],
          ['7 — Interface', 'Streamlit app + cloud deployment', 'Days 24–30  (~8 hrs)'],
          ['8 — Testing', 'Sanity checks, known material validation', 'Days 28–35  (~4 hrs)'],
        ],
        [1800, 4200, 3360]
      ),

      spacer(80),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 200 },
        children: [new TextRun({ text: 'Total: ~4–6 weeks building part-time  |  ~45 hours of active coding', font: FONT, size: 20, color: '666666', italics: true })]
      }),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('/home/claude/MatIntel_MVP_Guide.docx', buf);
  console.log('Done');
});
