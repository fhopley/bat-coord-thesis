"""Unweighted sample shares by PV size band, by state, for comparison with the
population band shares in analysis/config/pv_size_bands.yaml (thesis
Methodology, Representativeness and Weighting).

Uses the same site metadata (validated sites in data/processed/edp/{year}_selected_sites_clean.csv,
less EXCLUDED_SITES), size resolution and band edges as financial_analysis.py.
Per state and year: count sites in each of the bands (sites outside 4.5-25 kW or
without a size are dropped), convert to shares summing to 1, then take the
straight mean across years, matching how weighted results are averaged.
Pooled-site shares (all state-years stacked) are also reported.

Usage: python3 analysis/scripts/sample_size_band_shares.py
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
# scipy is only used by financial_analysis.py for curve smoothing; not needed here.
try:
    import scipy.interpolate  # noqa: F401
except ModuleNotFoundError:
    import types
    _sp = types.ModuleType('scipy'); _si = types.ModuleType('scipy.interpolate')
    _si.make_interp_spline = None
    sys.modules['scipy'] = _sp; sys.modules['scipy.interpolate'] = _si
import financial_analysis as fa

DATA_ROOT = Path('.')
CFG = fa.load_weights_config(Path('analysis/config/pv_size_bands.yaml'))
BANDS = CFG['bands']
LABELS = [b[0] for b in BANDS]

years = ['2021', '2022', '2023', '2024', '2025']
scenarios = [(st, y) for st in ['NSW', 'QLD', 'SA', 'VIC'] for y in years]

# Pooled DC:AC ratio per year, as in make_scenario_builder
pooled = {}
for year in years:
    frames = []
    for st in CFG['by_state']:
        try:
            frames.append(fa.load_site_meta(DATA_ROOT, year, st))
        except FileNotFoundError:
            pass
    pm = pd.concat(frames, ignore_index=True)
    ratio = fa.compute_dc_ac_ratio(pm)
    pooled[year] = (ratio, fa.assign_direct_size(pm, ratio))

rows = []
for state, year in scenarios:
    if state not in CFG['by_state']:
        continue
    ratio, pm = pooled[year]
    meta = fa.load_site_meta(DATA_ROOT, year, state)
    meta = fa.assign_direct_size(meta, ratio)
    meta = fa.estimate_missing_sizes(meta, state, fallback_meta=pm)
    lo, hi = BANDS[0][1], BANDS[-1][2]

    def band_of(kw):
        if pd.isna(kw) or kw < lo or kw > hi:
            return None
        for label, a, b in BANDS:
            if a <= kw <= b:
                return label
    meta['band'] = meta['size_kw'].apply(band_of)
    n_all = len(meta)
    inb = meta.dropna(subset=['band'])
    for label in LABELS:
        rows.append({'state': state, 'year': year, 'band': label,
                     'n': int((inb['band'] == label).sum()),
                     'n_in_bands': len(inb), 'n_sites': n_all})

df = pd.DataFrame(rows)
df['share'] = df['n'] / df['n_in_bands']
pd.set_option('display.width', 200)
print("\nPer state-year counts:")
print(df.pivot_table(index=['state', 'year'], columns='band', values='n')[LABELS]
      .join(df.groupby(['state', 'year'])[['n_in_bands', 'n_sites']].first()))

mean_years = df.groupby(['state', 'band'])['share'].mean().unstack()[LABELS]
pool = df.groupby(['state', 'band'])['n'].sum().unstack()[LABELS]
pool_share = pool.div(pool.sum(axis=1), axis=0)
pop = pd.DataFrame(CFG['by_state'], index=LABELS).T

print("\nSample share, mean across years (rows sum to 1):")
print((mean_years * 100).round(1))
print("\nSample share, pooled state-years:")
print((pool_share * 100).round(1))
print("\nPopulation share (yaml):")
print((pop * 100).round(1))
print("\nPopulation / sample (mean across years):")
print((pop.loc[mean_years.index] / mean_years).round(2))
df.to_csv('analysis/output/sample_size_band_shares.csv', index=False)
