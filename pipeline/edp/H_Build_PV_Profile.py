"""
H_Build_PV_Profile.py

Builds a normalised PV generation profile library from all clean site files
for a given year. Used by I_Infill_Long_All_Sites.py to infill long PV gaps.

For each 5-minute AEST interval across the year, computes the average
normalised PV generation across all clean same-state sites:
  - Only uses intervals where pv_flag == 0 (original clean data)
  - Excludes Method 1 infilled intervals (pv_infill == 1)
  - Normalises each site's contribution by its pv_q3_wh from site stats
  - Averages normalised values across contributing sites

Output profile values are on a 0–1+ scale where 1.0 = Q3 generation for
a typical site in that state. Values > 1 occur during above-average
generation periods (clear days, high sun angle).

After building, prints a warning summary of intervals with fewer than
MIN_SITES_WARNING contributing sites, as these profiles may be
unrepresentative.

Inputs:  data/clean/edp/YYYY/edp_clean_{SITE_ID}_{YYYY}.csv (all sites)
         data/processed/edp/YYYY_site_pv_stats.csv
         data/processed/edp/YYYY_selected_sites_clean.csv (for state lookup)
Output:  data/processed/edp/YYYY_pv_profile_library.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys

# ── Parameters ─────────────────────────────────────────────────────────────────
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
MIN_SITES_WARNING = 20   # warn if fewer than this many sites contribute
                          # to an interval's profile
NEM_STATES        = ['NSW', 'QLD', 'SA', 'VIC']

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
ROOT_DIR      = SCRIPT_DIR.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "edp"
clean_dir     = ROOT_DIR / "data" / "clean" / "edp" / str(YEAR)

stats_file    = PROCESSED_DIR / f"{YEAR}_site_pv_stats.csv"
selected_file = PROCESSED_DIR / f"{YEAR}_selected_sites_clean.csv"
output_file   = PROCESSED_DIR / f"{YEAR}_pv_profile_library.csv"

# ── Load site metadata ─────────────────────────────────────────────────────────
stats    = pd.read_csv(stats_file).set_index('edp_site_id')
selected = pd.read_csv(selected_file, usecols=['edp_site_id', 'state'])
site_state = selected.set_index('edp_site_id')['state'].to_dict()

# ── Build complete AEST grid ───────────────────────────────────────────────────
EXPECTED_INTERVALS = 105_408 if YEAR in (2020, 2024) else 105_120
aest_grid = pd.date_range(
    start=pd.Timestamp(f'{YEAR}-01-01 00:05:00', tz='Australia/Brisbane'),
    periods=EXPECTED_INTERVALS,
    freq='5min'
)

# Accumulators: sum and count per state per interval
# Using numpy arrays for efficiency
n = EXPECTED_INTERVALS
state_sum   = {s: np.zeros(n) for s in NEM_STATES}
state_count = {s: np.zeros(n, dtype=int) for s in NEM_STATES}

# ── Single pass through all clean files ───────────────────────────────────────
clean_files = sorted(clean_dir.glob(f"edp_clean_*_{YEAR}.csv"))
print(f"Building PV profile library for {YEAR}")
print(f"Processing {len(clean_files)} clean files...\n")

t_start = time.time()
sites_used = {s: 0 for s in NEM_STATES}

for i, fpath in enumerate(clean_files):
    site_id = fpath.stem.replace('edp_clean_', '').replace(f'_{YEAR}', '')
    state   = site_state.get(site_id)

    if state not in NEM_STATES:
        continue

    # Get this site's Q3 for normalisation
    pv_q3 = stats.loc[site_id, 'pv_q3_wh'] if site_id in stats.index else np.nan
    if np.isnan(pv_q3) or pv_q3 <= 0:
        continue  # can't normalise without Q3

    df = pd.read_csv(fpath,
                     usecols=['datetime_aest', 'pv_site_net_wh',
                              'pv_flag', 'pv_infill'])
    df['datetime_aest'] = pd.to_datetime(df['datetime_aest'])

    # Only use clean original data — flag 0 and not Method 1 infilled
    mask = (df['pv_flag'] == 0) & (df['pv_infill'] == 0)
    clean = df[mask].copy()

    if len(clean) == 0:
        continue

    # Align to grid index
    clean['grid_idx'] = clean['datetime_aest'].map(
        {ts: idx for idx, ts in enumerate(aest_grid)}
    )
    clean = clean.dropna(subset=['grid_idx'])
    clean['grid_idx'] = clean['grid_idx'].astype(int)

    # Normalise and accumulate
    normalised = clean['pv_site_net_wh'].values / pv_q3
    idx = clean['grid_idx'].values

    np.add.at(state_sum[state],   idx, normalised)
    np.add.at(state_count[state], idx, 1)

    sites_used[state] += 1

    if (i + 1) % 100 == 0:
        print(f"  [{i+1:>3}/{len(clean_files)}] processed ({time.time()-t_start:.0f}s)")

print(f"\nSites contributing per state:")
for s in NEM_STATES:
    print(f"  {s}: {sites_used[s]} sites")

# ── Compute average normalised profiles ───────────────────────────────────────
print("\nComputing averages and building output...")

profile_df = pd.DataFrame({'datetime_aest': aest_grid})

for state in NEM_STATES:
    with np.errstate(invalid='ignore'):
        avg = np.where(
            state_count[state] > 0,
            state_sum[state] / state_count[state],
            np.nan
        )
    profile_df[f'{state}_profile']       = avg
    profile_df[f'{state}_n_sites']       = state_count[state]

# Fill zero-site intervals with 0 profile (all are overnight — physically correct)
for state in NEM_STATES:
    col = f'{state}_profile'
    profile_df[col] = profile_df[col].fillna(0.0)

# ── Warning: low contributor intervals ────────────────────────────────────────
print(f"\nWarning summary — intervals with < {MIN_SITES_WARNING} contributing sites:")
for state in NEM_STATES:
    col = f'{state}_n_sites'
    low = profile_df[
        (profile_df[col] > 0) &
        (profile_df[col] < MIN_SITES_WARNING)
    ]
    zero = (profile_df[col] == 0).sum()
    print(f"  {state}: {len(low):>5} intervals with 1–{MIN_SITES_WARNING-1} sites | "
          f"{zero:>5} intervals with 0 sites (NaN profile)")

# ── Save ───────────────────────────────────────────────────────────────────────
profile_df.to_csv(output_file, index=False)

total_elapsed = time.time() - t_start
print(f"\nSaved to: {output_file}")
print(f"Rows: {len(profile_df):,} | Columns: {list(profile_df.columns)}")
print(f"Completed in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
