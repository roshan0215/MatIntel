#!/usr/bin/env python
"""Rebuild top-10 rankings with current viability scores."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import app
from src.matintel.config import APP_LABELS

df = app.load_data(str(app.SCORED_CSV)).copy()

out_dir = Path(__file__).parent.parent / 'data' / 'processed'
out_dir.mkdir(parents=True, exist_ok=True)


def add_provenance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    source_series = frame.get('source', pd.Series('', index=frame.index))
    source_lower = source_series.astype('string').str.lower()

    is_exp = (
        frame.get('is_experimental', pd.Series(False, index=frame.index))
        .astype('string')
        .str.strip()
        .str.lower()
        .map({'true': True, 'false': False, '1': True, '0': False, 'yes': True, 'no': False})
    )

    fallback_exp = ~source_lower.str.contains('mp_synthesized|mp_experimental|jarvis_icsd|synthesized', na=False)
    is_exp = is_exp.where(is_exp.notna(), fallback_exp).fillna(True).astype(bool)

    out = frame.copy()
    out['is_experimental'] = is_exp
    out['provenance_status'] = is_exp.map(
        {
            True: 'Experimental (NOT synthesized yet)',
            False: 'Synthesized (already synthesized/known)',
        }
    )
    out['provenance_definition'] = is_exp.map(
        {
            True: 'Experimental means this candidate is not already in synthesized reference datasets.',
            False: 'Synthesized means this material is already reported as synthesized in reference datasets.',
        }
    )
    return out

def pick_cols(frame):
    wanted = [
        'application', 'score_column', 'MaterialId', 'Reduced Formula',
        'score_value', 'viability', 'weighted_score', 'clscore', 'Bandgap',
        'source', 'is_experimental', 'provenance_status', 'provenance_definition'
    ]
    return frame[[c for c in wanted if c in frame.columns]]

weighted_rows = []
raw_rows = []
for app_name, score_col in APP_LABELS.items():
    if score_col not in df.columns:
        continue
    sub = df.copy()
    sub['score_value'] = pd.to_numeric(sub[score_col], errors='coerce').fillna(0.0)
    sub['viability'] = pd.to_numeric(sub.get('viability', 1.0), errors='coerce').fillna(1.0)
    sub['weighted_score'] = sub['score_value'] * sub['viability']

    # Keep one representative per reduced formula to avoid repeated polymorph entries.
    top_w = (
        sub.sort_values(['weighted_score', 'score_value'], ascending=False)
        .drop_duplicates(subset=['Reduced Formula'], keep='first')
        .head(10)
        .copy()
    )
    top_w.insert(0, 'application', app_name)
    top_w.insert(1, 'score_column', score_col)
    weighted_rows.append(pick_cols(add_provenance_columns(top_w)))

    top_r = (
        sub.sort_values(['score_value', 'viability'], ascending=False)
        .drop_duplicates(subset=['Reduced Formula'], keep='first')
        .head(10)
        .copy()
    )
    top_r.insert(0, 'application', app_name)
    top_r.insert(1, 'score_column', score_col)
    raw_rows.append(pick_cols(add_provenance_columns(top_r)))

weighted_df = pd.concat(weighted_rows, ignore_index=True)
raw_df = pd.concat(raw_rows, ignore_index=True)
weighted_df.to_csv(out_dir / 'top10_per_category.csv', index=False)
raw_df.to_csv(out_dir / 'top10_per_category_raw_score.csv', index=False)

print(f'Regenerated top-10 files')
print(f'  weighted: {len(weighted_df)} rows, clscore unknown: {int((pd.to_numeric(weighted_df["clscore"], errors="coerce") < 0).sum())}')
print(f'  raw: {len(raw_df)} rows, clscore unknown: {int((pd.to_numeric(raw_df["clscore"], errors="coerce") < 0).sum())}')
print(f'weighted viability range: {weighted_df["viability"].min():.3f}-{weighted_df["viability"].max():.3f}')
print(f'raw viability range: {raw_df["viability"].min():.3f}-{raw_df["viability"].max():.3f}')
