"""
I_Infill_Long_All_Sites.py

Applies Method 2 (synthetic day) infill to all clean site files for a
given year.

For each site with remaining NaN or flagged intervals after Script G:
  - PV: looks up state normalised profile from YYYY_pv_profile_library.csv,
    scaled by site Q3 from YYYY_site_pv_stats.csv
  - Gross load: averages two closest same-day-of-week complete days from
    the site's own data (Flag 0 original or Method 1 infilled days eligible)
  - Replaces both NaN and flagged non-NaN intervals — flagged values are
    unreliable and should be replaced regardless of gap length
  - Sets pv_infill=2 and load_infill=2 for filled intervals
  - Back-calculates ac_load_net_wh after both channels are filled

Performance: both PV and load infill use vectorised numpy operations to
avoid Python-level per-interval loops, which were the main bottleneck
for sites with large gaps (10+ days).

PV profile library is loaded once upfront and shared across all sites.

Inputs:  data/clean/edp/YYYY/edp_clean_{SITE_ID}_{YYYY}.csv (all sites)
         data/processed/edp/YYYY_pv_profile_library.csv
         data/processed/edp/YYYY_site_pv_stats.csv
         data/processed/edp/YYYY_selected_sites_clean.csv
Output:  data/clean/edp/YYYY/edp_clean_{SITE_ID}_{YYYY}.csv (updated in place)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys

# ── Parameters ─────────────────────────────────────────────────────────────────
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
MIN_COMPLETE_DAY_INTERVALS = 288
NEM_STATES                 = ['NSW', 'QLD', 'SA', 'VIC']

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
ROOT_DIR      = SCRIPT_DIR.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "edp"
clean_dir     = ROOT_DIR / "data" / "clean" / "edp" / str(YEAR)

profile_file  = PROCESSED_DIR / f"{YEAR}_pv_profile_library.csv"
stats_file    = PROCESSED_DIR / f"{YEAR}_site_pv_stats.csv"
selected_file = PROCESSED_DIR / f"{YEAR}_selected_sites_clean.csv"

# ── Load shared inputs ─────────────────────────────────────────────────────────
print(f"Year: {YEAR}")
print("Loading PV profile library...")
profile = pd.read_csv(profile_file, parse_dates=['datetime_aest'])
profile['datetime_aest'] = pd.to_datetime(profile['datetime_aest'])
profile = profile.set_index('datetime_aest')

# Build per-state profile dicts once upfront — avoids repeated .to_dict()
# conversion inside the per-site loop (105,120 entries × 346 sites = slow)
print("Building profile lookup dicts...")
profile_dicts = {state: profile[f'{state}_profile'].to_dict()
                 for state in NEM_STATES}

print("Loading site stats and metadata...")
stats      = pd.read_csv(stats_file, index_col='edp_site_id')
selected   = pd.read_csv(selected_file, usecols=['edp_site_id', 'state'])
site_state = selected.set_index('edp_site_id')['state'].to_dict()

# ── Find all clean files ───────────────────────────────────────────────────────
clean_files = sorted(clean_dir.glob(f"edp_clean_*_{YEAR}.csv"))
print(f"Clean files found: {len(clean_files)}\n")

# ── Process all sites ──────────────────────────────────────────────────────────
t_start           = time.time()
total_pv_filled   = 0
total_load_filled = 0
total_pv_remain   = 0
total_load_remain = 0
sites_processed   = 0
sites_skipped     = 0

for i, fpath in enumerate(clean_files):
    site_id = fpath.stem.replace('edp_clean_', '').replace(f'_{YEAR}', '')
    state   = site_state.get(site_id)

    if state not in NEM_STATES:
        sites_skipped += 1
        continue

    df = pd.read_csv(fpath, parse_dates=['datetime_aest'])
    df['datetime_aest'] = pd.to_datetime(df['datetime_aest'])

    if 'pv_infill' not in df.columns:
        df['pv_infill'] = 0
    if 'load_infill' not in df.columns:
        df['load_infill'] = 0

    # Check if any intervals need replacement
    # Exclude intervals already filled by Method 1 (pv_infill==1 / load_infill==1)
    # — Method 1 interpolated between clean boundaries and should not be overwritten
    pv_needs   = ((df['pv_site_net_wh'].isna() | (df['pv_flag'] > 0)) &
                  (df['pv_infill'] == 0)).values
    load_needs = ((df['gross_load_wh'].isna()   | (df['load_flag'] > 0)) &
                  (df['load_infill'] == 0)).values

    if not pv_needs.any() and not load_needs.any():
        sites_skipped += 1
        continue

    sites_processed += 1
    pv_q3 = stats.loc[site_id, 'pv_q3_wh'] if site_id in stats.index else np.nan

    # ── PV infill — vectorised ─────────────────────────────────────────────────
    pv_values = df['pv_site_net_wh'].values.copy()
    pv_infill = df['pv_infill'].values.copy()

    if not np.isnan(pv_q3) and pv_q3 > 0 and pv_needs.any():
        # Map timestamps to profile values using pre-built dict
        profile_vals = df['datetime_aest'].map(
            profile_dicts[state]
        ).fillna(0.0).values

        # Apply to all needing-replacement intervals at once
        pv_values[pv_needs] = profile_vals[pv_needs] * pv_q3
        pv_infill[pv_needs] = 2

    pv_filled             = int(pv_needs.sum())
    df['pv_site_net_wh']  = pv_values
    df['pv_infill']       = pv_infill

    # ── Gross load infill — vectorised ─────────────────────────────────────────
    gross_values = df['gross_load_wh'].values.copy()
    load_infill  = df['load_infill'].values.copy()
    load_filled  = 0

    # Precompute date and day-of-week arrays
    aest_dt    = df['datetime_aest'].dt.tz_convert('Australia/Brisbane')
    dates      = aest_dt.dt.date.values
    dows       = aest_dt.dt.dayofweek.values
    time_slots = aest_dt.dt.time.values  # for alignment within day

    # Build reference pool: complete days (Flag 0 or Method 1 infilled)
    # Use a fast groupby on date to count qualifying intervals
    df['_date'] = dates
    df['_good'] = (
        df['gross_load_wh'].notna() &
        ((df['load_flag'] == 0) | (df['load_infill'] == 1))
    )
    day_counts    = df.groupby('_date')['_good'].sum()
    complete_days = set(day_counts[day_counts >= MIN_COMPLETE_DAY_INTERVALS].index)

    # Build day-of-week reference: {dow: [(date, profile_array), ...]}
    # Profile arrays are already sorted by time within day
    dow_reference = {}
    for date in sorted(complete_days):
        day_mask = dates == date
        if day_mask.sum() != MIN_COMPLETE_DAY_INTERVALS:
            continue
        dow = int(dows[day_mask][0])
        ref_vals = gross_values[day_mask]  # 288-element array in time order
        if dow not in dow_reference:
            dow_reference[dow] = []
        dow_reference[dow].append((date, ref_vals))

    # For each gap day, find reference profile and assign vectorised
    gap_dates = np.unique(dates[load_needs])
    for gap_date in gap_dates:
        gap_dow      = pd.Timestamp(gap_date).dayofweek
        gap_day_mask = (dates == gap_date) & load_needs

        if not gap_day_mask.any():
            continue
        if gap_dow not in dow_reference or len(dow_reference[gap_dow]) == 0:
            continue  # no reference days available — leave as NaN

        # Find two closest same-day-of-week reference days
        ref_days = sorted(
            dow_reference[gap_dow],
            key=lambda x: abs((pd.Timestamp(x[0]) -
                                pd.Timestamp(gap_date)).days)
        )
        ref_profile = (
            (ref_days[0][1] + ref_days[1][1]) / 2
            if len(ref_days) >= 2
            else ref_days[0][1].copy()
        )

        # Map reference profile positions to gap day positions
        # Both arrays are 288 intervals in time order — align by position
        gap_day_indices = np.where(dates == gap_date)[0]  # all 288 positions
        needs_indices   = np.where(gap_day_mask)[0]       # subset needing fill

        # Map each needs_index to its position within the day (0–287)
        day_start = gap_day_indices[0]
        positions = needs_indices - day_start  # offset within day

        # Vectorised assignment
        gross_values[needs_indices] = ref_profile[positions]
        load_infill[needs_indices]  = 2
        load_filled += len(needs_indices)

    df['gross_load_wh'] = gross_values
    df['load_infill']   = load_infill

    # ── Back-calculate ac_load_net_wh ──────────────────────────────────────────
    # Primary: Method 2 load infill
    method2_mask = load_infill == 2
    df.loc[method2_mask, 'ac_load_net_wh'] = (
        df.loc[method2_mask, 'gross_load_wh'] -
        df.loc[method2_mask, 'pv_site_net_wh']
    )
    # Secondary: intervals where gross_load was filled by Method 1 (Script G)
    # but pv_site_net was NaN at that time and filled here by Method 2.
    # Script G's back-calculation produced NaN for these; fix them now.
    missing_ac = (
        df['ac_load_net_wh'].isna() &
        df['gross_load_wh'].notna() &
        df['pv_site_net_wh'].notna()
    )
    if missing_ac.any():
        df.loc[missing_ac, 'ac_load_net_wh'] = (
            df.loc[missing_ac, 'gross_load_wh'] -
            df.loc[missing_ac, 'pv_site_net_wh']
        )

    # ── Save ───────────────────────────────────────────────────────────────────
    df = df.drop(columns=['_date', '_good'], errors='ignore')
    output_cols = ['datetime_aest', 'pv_site_net_wh', 'ac_load_net_wh',
                   'gross_load_wh', 'pv_flag', 'load_flag',
                   'pv_infill', 'load_infill',
                   'pv_original_wh', 'ac_load_net_original_wh']
    df[output_cols].to_csv(fpath, index=False)

    # Count remaining NaN (flags preserved, not used for remaining count)
    pv_remain   = int(np.isnan(pv_values).sum())
    load_remain = int(np.isnan(gross_values).sum())

    total_pv_filled   += pv_filled
    total_load_filled += load_filled
    total_pv_remain   += pv_remain
    total_load_remain += load_remain

    if sites_processed % 50 == 0 or (i + 1) == len(clean_files):
        elapsed = time.time() - t_start
        print(f"  [{i+1:>3}/{len(clean_files)}] processed "
              f"({elapsed:.0f}s) | "
              f"PV filled: {total_pv_filled:,} | "
              f"Load filled: {total_load_filled:,}")

# ── Final summary ──────────────────────────────────────────────────────────────
total_elapsed = time.time() - t_start
print(f"\nComplete in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"Sites processed: {sites_processed} | Skipped: {sites_skipped}")
print(f"\nTotal Method 2 intervals filled:")
print(f"  PV:   {total_pv_filled:,}")
print(f"  Load: {total_load_filled:,}")
print(f"\nTotal still remaining after Method 2:")
print(f"  PV:   {total_pv_remain:,}")
print(f"  Load: {total_load_remain:,}")