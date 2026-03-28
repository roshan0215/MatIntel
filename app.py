from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src.matintel.config import APP_LABELS, SCORED_CSV
from src.matintel.explanations import generate_material_summary
from src.matintel.scoring import apply_application_scores
from src.matintel.viability import apply_viability, clscore_penalty

st.set_page_config(page_title="MatIntel", page_icon="MI", layout="wide")


def data_cache_token(path: str) -> tuple[tuple[str, int, int], ...]:
    p = Path(path)
    deps = [
        p,
        p.parent / "experimental_compounds.csv",
        p.parent / "clscore_all_results.csv",
    ]
    token: list[tuple[str, int, int]] = []
    for dep in deps:
        if dep.exists():
            stat = dep.stat()
            token.append((dep.name, int(stat.st_mtime_ns), int(stat.st_size)))
        else:
            token.append((dep.name, -1, -1))
    return tuple(token)


@st.cache_data
def load_data(path: str, cache_token: object | None = None) -> pd.DataFrame:
    _ = cache_token
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)

    # Optionally append experimental compounds if generated.
    exp_path = p.parent / "experimental_compounds.csv"
    if exp_path.exists():
        exp_df = pd.read_csv(exp_path)
        if len(exp_df):
            exp_df["source"] = exp_df.get("source", "Synthesized")
            exp_df["is_experimental"] = exp_df.get("is_experimental", False)

            # Ensure experimental rows have score/viability columns populated.
            score_cols = list(APP_LABELS.values())
            needs_scoring = any(col not in exp_df.columns for col in score_cols)
            if not needs_scoring:
                try:
                    needs_scoring = exp_df[score_cols].isna().all(axis=1).any()
                except Exception:
                    needs_scoring = True

            if needs_scoring:
                drop_cols = [c for c in exp_df.columns if c.startswith("score_")]
                drop_cols += [c for c in ["top_application", "best_score"] if c in exp_df.columns]
                if drop_cols:
                    exp_df = exp_df.drop(columns=drop_cols, errors="ignore")
                exp_df = apply_application_scores(exp_df)
                exp_df = apply_viability(exp_df)

            df = pd.concat([df, exp_df], ignore_index=True, sort=False)

    # Hydrate CLscore from standalone cache; also backfill unknown (-1) rows when clscore column already exists.
    if "MaterialId" in df.columns:
        clscore_cache = p.parent / "clscore_all_results.csv"
        if clscore_cache.exists():
            cl_df = pd.read_csv(clscore_cache, usecols=["MaterialId", "clscore"])
            cl_df["MaterialId"] = cl_df["MaterialId"].astype(str)
            cl_df["clscore"] = pd.to_numeric(cl_df["clscore"], errors="coerce")
            cl_df = cl_df.drop_duplicates(subset=["MaterialId"], keep="last")

            if "clscore" not in df.columns:
                df = df.merge(cl_df, on="MaterialId", how="left")
            else:
                cl_map = cl_df.set_index("MaterialId")["clscore"]
                cache_vals = df["MaterialId"].astype(str).map(cl_map)
                current = pd.to_numeric(df["clscore"], errors="coerce")
                df["clscore"] = current.where(current >= 0, cache_vals)
                df["clscore"] = df["clscore"].fillna(cache_vals).fillna(-1.0)

    # Ensure viability fields stay consistent with the current CLscore mapping.
    if {"cost_score", "abundance_score", "supply_risk"}.issubset(df.columns):
        cost = pd.to_numeric(df["cost_score"], errors="coerce").fillna(0.5)
        abundance = pd.to_numeric(df["abundance_score"], errors="coerce").fillna(0.5)
        supply = pd.to_numeric(df["supply_risk"], errors="coerce").fillna(0.5)
        filter_mult = pd.to_numeric(df.get("viability_filter_multiplier", 1.0), errors="coerce").fillna(1.0)
        clscore_vals = pd.to_numeric(df.get("clscore", -1.0), errors="coerce").fillna(-1.0)

        expected_cl_mult = clscore_vals.apply(clscore_penalty).astype(float)
        if "clscore_multiplier" not in df.columns:
            df["clscore_multiplier"] = expected_cl_mult
        else:
            current_cl_mult = pd.to_numeric(df["clscore_multiplier"], errors="coerce")
            mismatch = current_cl_mult.isna() | (current_cl_mult.round(3) != expected_cl_mult.round(3))
            if mismatch.any():
                df.loc[mismatch, "clscore_multiplier"] = expected_cl_mult.loc[mismatch]

        base_viability = 0.45 * cost + 0.35 * abundance + 0.20 * supply
        final_viability = (
            base_viability
            * filter_mult
            * pd.to_numeric(df["clscore_multiplier"], errors="coerce").fillna(expected_cl_mult)
        ).round(3)
        if "viability" not in df.columns:
            df["viability"] = final_viability
        else:
            current_viability = pd.to_numeric(df["viability"], errors="coerce")
            mismatch = current_viability.isna() | ((current_viability.round(3) - final_viability).abs() > 0.001)
            if mismatch.any():
                df.loc[mismatch, "viability"] = final_viability.loc[mismatch]

    # Ensure source/provenance flags exist for filtering.
    if "source" in df.columns:
        source_lower = df["source"].astype("string").str.lower()
        inferred_experimental = ~source_lower.str.contains("mp_synthesized|mp_experimental|jarvis_icsd|synthesized", na=False)
        missing_source = df["source"].isna() | source_lower.str.strip().eq("")
    else:
        source_lower = pd.Series("", index=df.index, dtype="string")
        inferred_experimental = pd.Series(True, index=df.index)
        missing_source = pd.Series(True, index=df.index)

    if "is_experimental" in df.columns:
        parsed = (
            df["is_experimental"]
            .astype("string")
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        )
        parsed = parsed.where(parsed.notna(), df["is_experimental"])
        parsed = parsed.where(parsed.isin([True, False]), np.nan)
        df["is_experimental"] = parsed.where(parsed.notna(), np.where(missing_source, True, inferred_experimental))
    else:
        # Historical scored rows often have empty source; treat those as experimental by default.
        df["is_experimental"] = np.where(missing_source, True, inferred_experimental)

    df["is_experimental"] = df["is_experimental"].astype(bool)
    # Source labels are authoritative for synthesized reference rows.
    synthesized_source = source_lower.str.contains("mp_synthesized|mp_experimental|jarvis_icsd|synthesized", na=False)
    df.loc[synthesized_source, "is_experimental"] = False
    df.loc[missing_source, "is_experimental"] = True

    if "source" not in df.columns:
        df["source"] = np.where(df["is_experimental"], "Experimental", "Synthesized")
    else:
        source_lower = df["source"].astype("string").str.lower()
        missing_source = df["source"].isna() | source_lower.str.strip().eq("")
        df.loc[missing_source, "source"] = np.where(
            df.loc[missing_source, "is_experimental"],
            "Experimental",
            "Synthesized",
        )

    return df


def format_float(v: object, precision: int = 3) -> str:
    try:
        return f"{float(v):.{precision}f}"
    except Exception:
        return "N/A"


def format_clscore(v: object) -> str:
    try:
        value = float(v)
    except Exception:
        return "CLscore not yet computed"
    if value == -1:
        return "CLscore not yet computed"
    return f"{value:.3f}"


def style_clscore_column(v: object) -> str:
    try:
        value = float(v)
    except Exception:
        return "color: #1c1e1b; background-color: #efe7d6;"
    if value == -1:
        return "color: #4d514b; background-color: #f3efe5;"
    if value >= 0.5:
        return "color: #103424; background-color: #cfeede;"
    if value >= 0.3:
        return "color: #4a3b10; background-color: #f8efbe;"
    return "color: #4f1417; background-color: #f8d2d2;"


def clscore_interpretation(v: object) -> str:
    try:
        value = float(v)
    except Exception:
        return "Unknown synthesizability"
    if value == -1:
        return "Unknown synthesizability (not computed or failed)"
    if value >= 0.5:
        return "Likely synthesizable"
    if value >= 0.3:
        return "Moderately plausible"
    if value >= 0.1:
        return "Low-confidence synthesizability"
    return "Very unlikely to synthesize"


def viability_breakdown_html(row: dict[str, object]) -> str:
    """Generate HTML showing detailed viability calculation breakdown."""
    cost_f = 0.5
    abundance_f = 0.5
    supply_f = 0.5
    filter_mult_f = 1.0
    clscore_mult_f = 0.5
    clscore_f = -1.0

    try:
        cost_f = float(row.get("cost_score", 0))
        abundance_f = float(row.get("abundance_score", 0))
        supply_f = float(row.get("supply_risk", 0))
        filter_mult_f = float(row.get("viability_filter_multiplier", 1.0))
        clscore_f = float(row.get("clscore", -1.0))
        clscore_mult_f = float(clscore_penalty(clscore_f))

        base_viability = (0.45 * cost_f + 0.35 * abundance_f + 0.20 * supply_f)
        filtered_viability = base_viability * filter_mult_f
        final_viability = filtered_viability * clscore_mult_f
    except Exception:
        base_viability = 0
        filtered_viability = 0
        final_viability = 0

    cost = format_float(cost_f, 3)
    abundance = format_float(abundance_f, 3)
    supply = format_float(supply_f, 3)
    filter_mult = format_float(filter_mult_f, 4)
    clscore_mult = format_float(clscore_mult_f, 3)
    viability = format_float(final_viability, 3)

    # Interpret each component
    cost_text = "Good (low cost)" if cost_f > 0.7 else "Moderate (medium cost)" if cost_f > 0.4 else "High (expensive elements)"
    abund_text = "Abundant" if abundance_f > 0.7 else "Moderate" if abundance_f > 0.4 else "Rare/limited"
    supply_text = "Low-risk" if supply_f > 0.7 else "Moderate risk" if supply_f > 0.3 else "High critical mineral content"
    
    html = f"""
    <div style="background: linear-gradient(135deg, #f0f4f0 0%, #eff7f2 100%); border: 1px solid #d4e5dc; border-radius: 12px; padding: 16px; margin: 8px 0;">
        <div style="font-size: 0.9rem; margin-bottom: 14px;">
            <strong>📊 Viability Score Breakdown</strong>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; font-size: 0.85rem;">
            <div style="background: white; padding: 10px; border-radius: 8px; border-left: 3px solid #9bc583;">
                <div style="color: #666; margin-bottom: 2px;">Cost Score (45% weight)</div>
                <div style="font-weight: 600; color: #2d5016;">{cost}</div>
                <div style="color: #888; font-size: 0.75rem; margin-top: 3px;">{cost_text}</div>
            </div>
            <div style="background: white; padding: 10px; border-radius: 8px; border-left: 3px solid #84c984;">
                <div style="color: #666; margin-bottom: 2px;">Abundance Score (35% weight)</div>
                <div style="font-weight: 600; color: #2d5016;">{abundance}</div>
                <div style="color: #888; font-size: 0.75rem; margin-top: 3px;">{abund_text}</div>
            </div>
            <div style="background: white; padding: 10px; border-radius: 8px; border-left: 3px solid #6fb86f;">
                <div style="color: #666; margin-bottom: 2px;">Supply Risk Score (20% weight)</div>
                <div style="font-weight: 600; color: #2d5016;">{supply}</div>
                <div style="color: #888; font-size: 0.75rem; margin-top: 3px;">{supply_text}</div>
            </div>
            <div style="background: white; padding: 10px; border-radius: 8px; border-left: 3px solid #5a9e5a;">
                <div style="color: #666; margin-bottom: 2px;">Element Filter (radioactive/REE)</div>
                <div style="font-weight: 600; color: #2d5016;">{filter_mult}×</div>
                <div style="color: #888; font-size: 0.75rem; margin-top: 3px;">Penalty multiplier</div>
            </div>
        </div>
        
        <div style="background: white; padding: 10px; border-radius: 8px; border: 1px solid #e0e0e0; font-size: 0.85rem; margin-bottom: 10px;">
            <div style="color: #666; margin-bottom: 6px;">Calculation steps:</div>
            <div style="font-family: monospace; color: #333; line-height: 1.6;">
                <div>① Base = (0.45 × {cost}) + (0.35 × {abundance}) + (0.20 × {supply})</div>
                <div style="margin-left: 20px; color: #666;">= (0.45 × {cost}) + (0.35 × {abundance}) + (0.20 × {supply})</div>
                <div style="margin-left: 20px; color: #666;">= <strong>{format_float(base_viability, 3)}</strong></div>
                <div style="margin-top: 4px;">② Filtered = {format_float(base_viability, 3)} × {filter_mult} = <strong>{format_float(filtered_viability, 3)}</strong></div>
                <div style="margin-top: 4px;">③ CLscore = {format_clscore(clscore_f)} → multiplier: {clscore_mult}</div>
                <div style="margin-top: 4px;">④ Final = {format_float(filtered_viability, 3)} × {clscore_mult} = <strong style="color: #1f6f5e; font-size: 1.05em;">{viability}</strong></div>
            </div>
        </div>
        
        <div style="background: #f9fdf9; padding: 8px; border-radius: 6px; font-size: 0.8rem; color: #555;">
            <strong>💡 Note:</strong> Higher viability indicates real-world feasibility considering cost, elemental availability, supply chain risk, synthesis difficulty, and regulatory constraints.
        </div>
    </div>
    """
    return html


def google_scholar_url(formula: str) -> str:
    return f"https://scholar.google.com/scholar?q={quote_plus(formula)}"


@st.cache_data(show_spinner=False)
def lookup_materials_project_by_formula(formula: str) -> dict[str, object]:
    api_key = os.getenv("MATINTEL_MP_API_KEY", "").strip()
    if not api_key:
        return {"status": "no_api_key", "known": None, "mp_ids": []}

    try:
        from mp_api.client import MPRester  # type: ignore
    except Exception:
        return {"status": "mp_api_missing", "known": None, "mp_ids": []}

    try:
        with MPRester(api_key) as mpr:
            results = mpr.summary.search(formula=formula, fields=["material_id"], num_chunks=1, chunk_size=20)
            mp_ids = [str(r.material_id) for r in results]
            return {"status": "ok", "known": bool(mp_ids), "mp_ids": mp_ids[:10]}
    except Exception as exc:
        return {"status": f"lookup_error: {exc}", "known": None, "mp_ids": []}


def render_structure_viewer(material_id: str) -> None:
    cif_path = Path("data/cifs") / f"{material_id}.CIF"
    if not cif_path.exists():
        cif_path = Path("data/cifs") / f"{material_id}.cif"
    if not cif_path.exists():
        st.info("CIF file not found for this material.")
        return

    try:
        import py3Dmol  # type: ignore
    except Exception:
        st.caption(f"CIF available at: {cif_path}")
        st.info("Install `py3Dmol` to enable in-app crystal structure rendering.")
        return

    try:
        cif_text = cif_path.read_text(encoding="utf-8", errors="ignore")
        view = py3Dmol.view(width=560, height=380)
        view.addModel(cif_text, "cif")
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.2}})
        view.addUnitCell()
        view.zoomTo()
        components.html(view._make_html(), height=400)
    except Exception as exc:
        st.warning(f"Unable to render structure preview: {exc}")


def build_export_bundle(filtered: pd.DataFrame, app_choice: str, score_col: str, top_n: int) -> tuple[pd.DataFrame, bytes]:
    export_df = filtered.head(top_n).copy()
    score_cols = [c for c in filtered.columns if c.startswith("score_")]
    keep = [
        "MaterialId",
        "Reduced Formula",
        score_col,
        "viability",
        "clscore",
        "cost_score",
        "abundance_score",
        "supply_risk",
        "clscore_multiplier",
        "Bandgap",
        "Formation Energy Per Atom",
        "Decomposition Energy Per Atom",
        "Crystal System",
        "source",
        "is_experimental",
    ]
    keep = [c for c in keep if c in export_df.columns]
    keep = keep + [c for c in score_cols if c not in keep]
    export_df = export_df[keep]

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for material_id in export_df.get("MaterialId", pd.Series(dtype=str)).astype(str).tolist():
            cif_upper = Path("data/cifs") / f"{material_id}.CIF"
            cif_lower = Path("data/cifs") / f"{material_id}.cif"
            cif_path = cif_upper if cif_upper.exists() else cif_lower
            if cif_path.exists():
                zf.write(cif_path, arcname=f"cifs/{cif_path.name}")
    zip_buffer.seek(0)
    return export_df, zip_buffer.getvalue()


def generate_export_summary_text(export_df: pd.DataFrame, app_choice: str, score_col: str) -> str:
    top3 = export_df.head(3)
    lines = [
        f"Category: {app_choice}",
        f"Ranking field: {score_col} x viability",
        "Top 3 candidates:",
    ]
    for idx, row in top3.iterrows():
        lines.append(
            f"- {row.get('Reduced Formula', 'N/A')} ({row.get('MaterialId', 'N/A')}): "
            f"score={format_float(row.get(score_col), 3)}, "
            f"viability={format_float(row.get('viability'), 3)}, "
            f"clscore={format_clscore(row.get('clscore', -1))}"
        )
    fallback_text = "\n".join(lines)

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return fallback_text

    try:
        from anthropic import Anthropic  # type: ignore
    except Exception:
        return fallback_text

    try:
        client = Anthropic(api_key=api_key)
        prompt = (
            "Write a concise one-page research summary for materials screening results. "
            "Include why the top three are promising, likely synthesis risks, and one short next-step plan.\n\n"
            + fallback_text
        )
        msg = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=900,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        if msg.content and len(msg.content) > 0:
            return str(msg.content[0].text)
    except Exception:
        pass
    return fallback_text


def build_summary_pdf(summary_text: str, title: str) -> bytes | None:
    try:
        from reportlab.lib.pagesizes import letter  # type: ignore
        from reportlab.pdfgen import canvas  # type: ignore
    except Exception:
        return None

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 48

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, title)
    y -= 24
    c.setFont("Helvetica", 10)

    for line in summary_text.splitlines():
        if y < 50:
            c.showPage()
            y = height - 48
            c.setFont("Helvetica", 10)
        c.drawString(40, y, line[:120])
        y -= 14

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def app() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
            .stApp {
                font-family: 'Space Grotesk', sans-serif;
            }
            html[data-theme="light"] .stApp,
            body[data-theme="light"] .stApp,
            [data-theme="light"] .stApp {
                --bg: #f4f6f1;
                --card: #ffffff;
                --ink: #1c1e1b;
                --accent: #ad343e;
                --accent-2: #1f6f5e;
                --muted: #6b7069;

                background:
                    radial-gradient(circle at 10% 10%, #fbe9da 0%, transparent 30%),
                    radial-gradient(circle at 90% 30%, #e2f2e8 0%, transparent 34%),
                    var(--bg);
                color: var(--ink);
            }
            html[data-theme="light"] .hero,
            body[data-theme="light"] .hero,
            [data-theme="light"] .hero {
                background: linear-gradient(135deg, #fce2b4, #f4f0dc 45%, #dceee6);
                border: 1px solid #dacda8;
            }
            .hero {
                border-radius: 18px;
                padding: 18px 20px;
                margin-bottom: 12px;
            }
            .muted { color: var(--muted); font-size: 0.95rem; }
            .metric-card {
                background: var(--card);
                border: 1px solid #e3e3df;
                border-radius: 14px;
                padding: 12px;
                box-shadow: 0 6px 14px rgba(20, 20, 20, 0.04);
            }
            [data-testid="stMetric"] {
                background: var(--card);
                border: 1px solid #e3e3df;
                border-radius: 14px;
                padding: 10px 12px;
                box-shadow: 0 6px 14px rgba(20, 20, 20, 0.04);
            }
            [data-testid="stMetric"] * {
                color: var(--ink) !important;
            }
            .result-table {
                border-radius: 12px;
                overflow: hidden;
            }
            html[data-theme="light"] .stButton button,
            body[data-theme="light"] .stButton button,
            [data-theme="light"] .stButton button {
                background: #f6d98f;
                color: var(--ink);
            }
            .stButton button {
                border-radius: 10px;
                border: none;
                font-family: 'Space Grotesk', sans-serif;
                font-weight: 700;
            }
            .code-font { font-family: 'IBM Plex Mono', monospace; }

            html[data-theme="dark"] .stApp,
            body[data-theme="dark"] .stApp,
            [data-theme="dark"] .stApp {
                --bg: #0f1613;
                --card: #1a2620;
                --ink: #edf4ef;
                --accent: #f29b67;
                --accent-2: #84d1a7;
                --muted: #b6c6bc;

                background:
                    radial-gradient(circle at 12% 8%, #2f3a31 0%, transparent 33%),
                    radial-gradient(circle at 88% 26%, #22372d 0%, transparent 35%),
                    var(--bg);
                color: var(--ink);
            }
            html[data-theme="dark"] .hero,
            body[data-theme="dark"] .hero,
            [data-theme="dark"] .hero {
                background: linear-gradient(135deg, #3a3a28, #2f3327 45%, #24372e);
                border: 1px solid #445547;
            }
            html[data-theme="dark"] .metric-card,
            body[data-theme="dark"] .metric-card,
            [data-theme="dark"] .metric-card,
            html[data-theme="dark"] [data-testid="stMetric"],
            body[data-theme="dark"] [data-testid="stMetric"],
            [data-theme="dark"] [data-testid="stMetric"] {
                border-color: #3b4f43;
                box-shadow: 0 8px 18px rgba(0, 0, 0, 0.28);
            }
            html[data-theme="dark"] .stButton button,
            body[data-theme="dark"] .stButton button,
            [data-theme="dark"] .stButton button {
                background: #c5a14d;
                color: #172019;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h2 style="margin-bottom: 4px;">MatIntel</h2>
            <div class="muted">Materials screening with application fit and real-world viability scoring.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data(str(SCORED_CSV), cache_token=data_cache_token(str(SCORED_CSV)))
    if df.empty:
        st.warning("No scored dataset found. Run scripts/run_pipeline.ps1 first.")
        return

    st.sidebar.header("Filters")
    app_choice = st.sidebar.selectbox("Target application", list(APP_LABELS.keys()))
    score_col = APP_LABELS[app_choice]

    min_app_score = st.sidebar.slider("Min application score", 0.0, 1.0, 0.45, 0.01)
    min_viability = st.sidebar.slider("Min viability", 0.0, 1.0, 0.30, 0.01)
    max_bandgap = st.sidebar.slider("Max band gap", 0.0, 8.0, 4.0, 0.05)
    min_supply = st.sidebar.slider("Min supply score", 0.0, 1.0, 0.20, 0.01)
    min_clscore = st.sidebar.slider("Min CLscore", 0.0, 1.0, 0.0, 0.1)
    provenance_choice = st.sidebar.selectbox("Compound provenance", ["All", "Experimental", "Synthesized"])

    filtered = df[
        (df[score_col] >= min_app_score)
        & (df["viability"] >= min_viability)
        & (df["supply_risk"] >= min_supply)
    ].copy()

    if provenance_choice == "Experimental" and "is_experimental" in filtered.columns:
        filtered = filtered[filtered["is_experimental"].fillna(True)]
    elif provenance_choice == "Synthesized" and "is_experimental" in filtered.columns:
        filtered = filtered[~filtered["is_experimental"].fillna(True)]

    if "Bandgap" in filtered.columns:
        filtered = filtered[filtered["Bandgap"].fillna(0) <= max_bandgap]

    if "clscore" in filtered.columns:
        filtered["clscore"] = pd.to_numeric(filtered["clscore"], errors="coerce").fillna(-1.0)
        filtered = filtered[(filtered["clscore"] >= min_clscore) | (filtered["clscore"] == -1.0)]
    else:
        filtered["clscore"] = -1.0

    filtered = filtered.sort_values(by=[score_col, "viability"], ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Materials", f"{len(filtered):,}")
    c2.metric("Mean app score", format_float(filtered[score_col].mean() if len(filtered) else 0, 2))
    c3.metric("Mean viability", format_float(filtered["viability"].mean() if len(filtered) else 0, 2))
    c4.metric("Top candidate", str(filtered.iloc[0]["Reduced Formula"]) if len(filtered) else "N/A")

    st.subheader("Export")
    export_n = st.slider("Top N to export", min_value=10, max_value=500, value=100, step=10)
    export_df, cif_zip = build_export_bundle(filtered, app_choice, score_col, export_n)
    st.download_button(
        label="Download CSV (all scores + viability + CLscore)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"matintel_{app_choice.lower().replace(' ', '_')}_top{export_n}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        label="Download CIF ZIP (matching exported rows)",
        data=cif_zip,
        file_name=f"matintel_{app_choice.lower().replace(' ', '_')}_top{export_n}_cifs.zip",
        mime="application/zip",
        use_container_width=True,
    )

    summary_text = generate_export_summary_text(export_df, app_choice, score_col)
    pdf_bytes = build_summary_pdf(summary_text, f"MatIntel Summary - {app_choice}")
    if pdf_bytes is not None:
        st.download_button(
            label="Download one-page PDF summary",
            data=pdf_bytes,
            file_name=f"matintel_{app_choice.lower().replace(' ', '_')}_summary.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.info("Install `reportlab` to enable PDF summary export.")

    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Top candidates")
        show_cols = [
            "MaterialId", "Reduced Formula", "source", "is_experimental", score_col, "viability", "clscore", "Bandgap",
            "Formation Energy Per Atom", "Decomposition Energy Per Atom", "Crystal System",
        ]
        show_cols = [c for c in show_cols if c in filtered.columns]
        table_df = filtered[show_cols].head(200).copy()
        if "clscore" in table_df.columns:
            table_df["clscore_status"] = table_df["clscore"].apply(format_clscore)
            display_cols = show_cols + ["clscore_status"]
            styled = table_df[display_cols].style.map(style_clscore_column, subset=["clscore"])
            st.dataframe(styled, use_container_width=True, height=420)
        else:
            st.dataframe(table_df, use_container_width=True, height=420)

    with right:
        st.subheader("Score landscape")
        if len(filtered):
            fig = px.scatter(
                filtered,
                x="viability",
                y=score_col,
                color="Crystal System" if "Crystal System" in filtered.columns else None,
                hover_name="Reduced Formula",
                title="Viability vs Application Score",
                height=400,
            )
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rows match the filters.")

    st.subheader("Material detail")
    options = filtered["Reduced Formula"].astype(str).head(200).tolist() if len(filtered) else []
    selected_formula = st.selectbox("Choose a material", options)

    if selected_formula:
        row = filtered[filtered["Reduced Formula"].astype(str) == selected_formula].iloc[0].to_dict()
        row["selected_score"] = row.get(score_col)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Application score", format_float(row.get(score_col), 2))
        m2.metric("Viability", format_float(row.get("viability"), 2))
        m3.metric("Band gap", format_float(row.get("Bandgap"), 2))
        m4.metric("Decomp energy", format_float(row.get("Decomposition Energy Per Atom"), 3))

        with st.expander("📋 Viability calculation details", expanded=False):
            st.markdown(viability_breakdown_html(row), unsafe_allow_html=True)

        st.caption(f"CLscore: {format_clscore(row.get('clscore', -1))}")
        st.caption(clscore_interpretation(row.get("clscore", -1)))

        mp_info = lookup_materials_project_by_formula(str(row.get("Reduced Formula", "")))
        if mp_info.get("known") is True:
            st.success(f"Known to Materials Project ({len(mp_info.get('mp_ids', []))} match(es))")
            if mp_info.get("mp_ids"):
                st.caption("MP IDs: " + ", ".join([str(x) for x in mp_info.get("mp_ids", [])[:5]]))
        elif mp_info.get("known") is False:
            st.info("Novel (no Materials Project match found)")
        else:
            st.caption(f"Materials Project lookup status: {mp_info.get('status', 'unavailable')}")

        scholar = google_scholar_url(str(row.get("Reduced Formula", "")))
        st.markdown(f"[Search this formula on Google Scholar]({scholar})")

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="muted">Material ID</div>
                <div class="code-font">{row.get('MaterialId', 'N/A')}</div>
                <div class="muted" style="margin-top: 8px;">Top application</div>
                <strong>{app_choice}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Crystal structure viewer", expanded=False):
            render_structure_viewer(str(row.get("MaterialId", "")))

        if st.button("Generate AI explanation"):
            with st.spinner("Generating explanation..."):
                summary = generate_material_summary(row, app_choice)
            st.success(summary)


if __name__ == "__main__":
    app()
