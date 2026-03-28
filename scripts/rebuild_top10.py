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

def pick_cols(frame):
    wanted = [
        'application', 'score_column', 'MaterialId', 'Reduced Formula',
        'score_value', 'viability', 'weighted_score', 'clscore', 'Bandgap',
        'source', 'is_experimental'
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

    top_w = sub.sort_values(['weighted_score', 'score_value'], ascending=False).head(10).copy()
    top_w.insert(0, 'application', app_name)
    top_w.insert(1, 'score_column', score_col)
    weighted_rows.append(pick_cols(top_w))

    top_r = sub.sort_values(['score_value', 'viability'], ascending=False).head(10).copy()
    top_r.insert(0, 'application', app_name)
    top_r.insert(1, 'score_column', score_col)
    raw_rows.append(pick_cols(top_r))

weighted_df = pd.concat(weighted_rows, ignore_index=True)
raw_df = pd.concat(raw_rows, ignore_index=True)
weighted_df.to_csv(out_dir / 'top10_per_category.csv', index=False)
raw_df.to_csv(out_dir / 'top10_per_category_raw_score.csv', index=False)

print(f'Regenerated top-10 files')
print(f'  weighted: {len(weighted_df)} rows, clscore unknown: {int((pd.to_numeric(weighted_df["clscore"], errors="coerce") < 0).sum())}')
print(f'  raw: {len(raw_df)} rows, clscore unknown: {int((pd.to_numeric(raw_df["clscore"], errors="coerce") < 0).sum())}')
print(f'weighted viability range: {weighted_df["viability"].min():.3f}-{weighted_df["viability"].max():.3f}')
print(f'raw viability range: {raw_df["viability"].min():.3f}-{raw_df["viability"].max():.3f}')
