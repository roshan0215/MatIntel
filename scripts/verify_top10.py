#!/usr/bin/env python
import pandas as pd

df = pd.read_csv('data/processed/top10_per_category_raw_score.csv')
print(f'rows: {len(df)}')
print(f'clscore range: {df["clscore"].min():.3f}-{df["clscore"].max():.3f}')
print(f'viability range: {df["viability"].min():.3f}-{df["viability"].max():.3f}')
print(f'clscore unknown: {(df["clscore"]<0).sum()}')
print(f'viability nulls: {df["viability"].isna().sum()}')

high_clscore = df[df['clscore']>=0.5]
print(f'high clscore (>=0.5): {len(high_clscore)} rows, avg viability: {high_clscore["viability"].mean():.3f}')

low_clscore = df[df['clscore'] < 0.5]
print(f'low clscore (<0.5): {len(low_clscore)} rows, avg viability: {low_clscore["viability"].mean():.3f}')

# Check a sample row to verify multiplier logic
print('\nSample verification (first high-clscore row):')
if len(high_clscore) > 0:
    row = high_clscore.iloc[0]
    print(f'  clscore={row["clscore"]:.3f} (should have 1.0x multiplier, no penalty)')
    print(f'  viability={row["viability"]:.3f}')
