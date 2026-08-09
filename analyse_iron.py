"""
Full analysis of the iron ore flotation dataset (Case Study 2).
Covers: shape, dtypes, missing values, measurement interval,
assay-event extraction, leakage quantification, feature summary,
Spearman correlations, autocorrelation, and lag structure.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CSV = str(PROJECT_ROOT / 'data' / 'iron_ore_flotation.csv')

# ── 1. Load ──────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(CSV, sep=',', decimal=',', parse_dates=['date'])
print(f"\nShape: {df.shape[0]:,} rows  x  {df.shape[1]} columns")
print(f"Date range: {df['date'].min()}  →  {df['date'].max()}")
span_days = (df['date'].max() - df['date'].min()).days
print(f"Duration: {span_days} days (~{span_days/30:.1f} months)")

# ── 2. Column groups ──────────────────────────────────────────────────────────
target_cols   = ['% Iron Concentrate', '% Silica Concentrate']
lab_feed_cols = ['% Iron Feed', '% Silica Feed']
reagent_cols  = ['Starch Flow', 'Amina Flow']
pulp_cols     = ['Ore Pulp Flow', 'Ore Pulp pH', 'Ore Pulp Density']
air_cols      = [c for c in df.columns if 'Air Flow' in c]
level_cols    = [c for c in df.columns if 'Level' in c]
sensor_cols   = reagent_cols + pulp_cols + air_cols + level_cols

print(f"\n--- Column groups ---")
print(f"  Lab targets (hourly, forward-filled): {target_cols}")
print(f"  Lab feed inputs (hourly):             {lab_feed_cols}")
print(f"  Reagents ({len(reagent_cols)}):              {reagent_cols}")
print(f"  Pulp sensors ({len(pulp_cols)}):            {pulp_cols}")
print(f"  Column air flows ({len(air_cols)}):         {air_cols}")
print(f"  Column levels ({len(level_cols)}):          {level_cols}")

# ── 3. Missing values ─────────────────────────────────────────────────────────
missing = df.isnull().sum()
missing = missing[missing > 0]
print(f"\n--- Missing values ---")
print("  None" if len(missing) == 0 else missing.to_string())

# ── 4. Measurement interval ───────────────────────────────────────────────────
intervals = df['date'].diff().dropna().dt.total_seconds() / 60
print(f"\n--- Measurement interval ---")
print(f"  Median: {intervals.median():.1f} min  |  Mode: {intervals.mode().iloc[0]:.1f} min")
print(f"  Min: {intervals.min():.1f} min  |  Max: {intervals.max():.0f} min")

# ── 5. Target distributions ───────────────────────────────────────────────────
print(f"\n--- Target variable statistics ---")
print(df[target_cols + lab_feed_cols].describe().round(3).to_string())

# ── 6. Assay-event frequency (the leakage problem) ────────────────────────────
print(f"\n--- Assay-event frequency (core of the data leakage argument) ---")
for col in target_cols + lab_feed_cols:
    n_changes  = int((df[col].diff().fillna(1) != 0).sum())
    n_total    = len(df)
    n_filled   = n_total - n_changes
    pct_filled = 100 * n_filled / n_total
    print(f"  {col:<30}  {n_changes:>5,} real measurements  "
          f"({n_filled:>7,} rows forward-filled, {pct_filled:.1f}% of dataset)")

# ── 7. Assay-event dataset ────────────────────────────────────────────────────
# A row is a real measurement event if ANY lab column changes
lab_all = target_cols + lab_feed_cols
change_mask = df[lab_all].diff().fillna(1).abs().sum(axis=1) > 0
assay_df = df[change_mask].copy()
print(f"\n--- Assay-event dataset ---")
print(f"  Real events: {len(assay_df):,} rows  (from {len(df):,} historian rows)")
print(f"  Compression ratio: {len(df)/len(assay_df):.0f}x reduction")
print(f"  Approximate assay interval: {span_days*24*60 / len(assay_df):.0f} min")

# ── 8. Autocorrelation of targets (motivates lag features) ───────────────────
print(f"\n--- Autocorrelation of targets at assay-event level ---")
for col in target_cols:
    for lag in [1, 2, 3]:
        acf = assay_df[col].autocorr(lag=lag)
        print(f"  {col}  lag-{lag}: r = {acf:.4f}")

# ── 9. Spearman correlations (sensor features → targets) ─────────────────────
print(f"\n--- Spearman correlations (all sensor features vs targets) ---")
from scipy.stats import spearmanr
print(f"  {'Feature':<35}  {'rho vs % Fe Conc':>18}  {'rho vs % SiO2 Conc':>20}")
print(f"  {'-'*35}  {'-'*18}  {'-'*20}")
for feat in sensor_cols + lab_feed_cols:
    rho_fe,  _ = spearmanr(df[feat], df['% Iron Concentrate'])
    rho_si,  _ = spearmanr(df[feat], df['% Silica Concentrate'])
    print(f"  {feat:<35}  {rho_fe:>+18.4f}  {rho_si:>+20.4f}")

# ── 10. Lag feature autocorrelation at assay level ───────────────────────────
print(f"\n--- Lag feature Spearman correlations (assay-event level) ---")
for feat in sensor_cols:
    for lag in [1, 3, 6]:
        rho, _ = spearmanr(assay_df[feat].shift(lag).dropna(),
                            assay_df['% Iron Concentrate'].iloc[lag:])
        if abs(rho) > 0.3:
            print(f"  {feat:<35}  lag-{lag}: rho vs % Fe = {rho:+.3f}")

print("\nDone.")
