"""
J_Site_Stats.py

Reads all clean site files for a given year and produces two output files:

1. YYYY_flag_summary.csv — per-site flag counts and clean percentages for
   both pv_site_net and ac_load_net channels, before and after infill.
   Useful for thesis methodology reporting.

2. YYYY_site_pv_stats.csv — per-site statistics required by Script G for
   synthetic day infill:
     - pv_q3_wh: Q3 of daytime positive PV intervals (used to scale
       synthetic PV profile)
     - avg_daily_gross_load_wh: average daily gross load across complete
       days (used to scale cross-site load estimates)

Both files replace any previously incomplete versions written during
Script E processing.

Inputs:  data/clean/edp/YYYY/edp_clean_{SITE_ID}_{YYYY}.csv (all sites)
         data/processed/edp/YYYY_selected_sites_clean.csv (for state lookup)
Outputs: data/processed/edp/YYYY_flag_summary.csv
         data/processed/edp/YYYY_site_pv_stats.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys

# ── Parameters ─────────────────────────────────────────────────────────────────
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
SOLAR_HOUR_START = 9   # AEST inclusive — for PV Q3 calculation
SOLAR_HOUR_END   = 16  # AEST exclusive

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
ROOT_DIR      = SCRIPT_DIR.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "edp"
clean_dir     = ROOT_DIR / "data" / "clean" / "edp" / str(YEAR)

selected_file    = PROCESSED_DIR / f"{YEAR}_selected_sites_clean.csv"
flag_summary_file = PROCESSED_DIR / f"{YEAR}_flag_summary.csv"
site_stats_file  = PROCESSED_DIR / f"{YEAR}_site_pv_stats.csv"

# ── Load selected sites for state lookup ──────────────────────────────────────
selected = pd.read_csv(selected_file,
                       usecols=['edp_site_id', 'state', 'site_timezone'])
site_meta = selected.set_index('edp_site_id').to_dict('index')

# ── Find all clean files ───────────────────────────────────────────────────────
clean_files = sorted(clean_dir.glob(f"edp_clean_*_{YEAR}.csv"))
print(f"Found {len(clean_files)} clean files for {YEAR}")
print(f"Processing...\n")

EXPECTED_INTERVALS = 105_408 if YEAR in (2020, 2024) else 105_120
FLAG_CODES = list(range(7))  # 0 through 6

flag_rows = []
stats_rows = []
t_start = time.time()

for i, fpath in enumerate(clean_files):
    # Extract site ID from filename
    site_id = fpath.stem.replace(f'edp_clean_', '').replace(f'_{YEAR}', '')
    meta    = site_meta.get(site_id, {})
    state   = meta.get('state', '??')
    tz      = meta.get('site_timezone', '??')

    df = pd.read_csv(fpath, parse_dates=['datetime_aest'])
    df['datetime_aest'] = pd.to_datetime(df['datetime_aest'])

    # Add infill columns if not present (pre-F files)
    if 'pv_infill' not in df.columns:
        df['pv_infill']   = 0
    if 'load_infill' not in df.columns:
        df['load_infill'] = 0

    n = len(df)

    # ── Flag counts ────────────────────────────────────────────────────────────
    row = {'edp_site_id': site_id, 'state': state, 'n_intervals': n}

    for code in FLAG_CODES:
        row[f'pv_flag_{code}']   = (df['pv_flag']   == code).sum()
        row[f'load_flag_{code}'] = (df['load_flag'] == code).sum()

    row['pv_clean_pct']   = round(row['pv_flag_0']   / n * 100, 2)
    row['load_clean_pct'] = round(row['load_flag_0'] / n * 100, 2)

    # Infill tracking
    row['pv_method1']   = (df['pv_infill']   == 1).sum()
    row['pv_method2']   = (df['pv_infill']   == 2).sum()
    row['load_method1'] = (df['load_infill'] == 1).sum()
    row['load_method2'] = (df['load_infill'] == 2).sum()

    # NaN remaining (unfilled)
    row['pv_nan_remaining']   = df['pv_site_net_wh'].isna().sum()
    row['load_nan_remaining'] = df['gross_load_wh'].isna().sum()

    flag_rows.append(row)

    # ── PV Q3 (daytime positive intervals) ────────────────────────────────────
    hours = df['datetime_aest'].dt.tz_convert('Australia/Brisbane').dt.hour
    daytime_pv = df.loc[
        (hours >= SOLAR_HOUR_START) &
        (hours < SOLAR_HOUR_END) &
        (df['pv_site_net_wh'].notna()) &
        (df['pv_site_net_wh'] > 0),
        'pv_site_net_wh'
    ]
    if len(daytime_pv) > 0:
        q3 = daytime_pv.quantile(0.75)
    else:
        q3 = np.nan

    # ── Average daily gross load (complete days only) ─────────────────────────
    df['date'] = df['datetime_aest'].dt.date
    daily_gross  = df.groupby('date')['gross_load_wh'].sum()
    daily_counts = df.groupby('date')['gross_load_wh'].count()
    complete_days = daily_counts[daily_counts == 288].index
    avg_daily_gross = (daily_gross[complete_days].mean()
                       if len(complete_days) > 0 else np.nan)

    # ── Phase counts from flag summary ────────────────────────────────────────
    # Re-derive max phases from the data if available, else NaN
    # (phase info not stored in clean files — use flag 4 as proxy indicator)
    has_partial_phase_pv   = row['pv_flag_4'] > 0
    has_partial_phase_load = row['load_flag_4'] > 0

    stats_rows.append({
        'edp_site_id':             site_id,
        'state':                   state,
        'site_timezone':           tz,
        'pv_q3_wh':                round(q3, 2) if not np.isnan(q3) else np.nan,
        'avg_daily_gross_load_wh': round(avg_daily_gross, 1) if not np.isnan(avg_daily_gross) else np.nan,
        'has_partial_phase_pv':    has_partial_phase_pv,
        'has_partial_phase_load':  has_partial_phase_load,
    })

    if (i + 1) % 50 == 0 or (i + 1) == len(clean_files):
        print(f"  Processed {i+1:>3}/{len(clean_files)} sites "
              f"({time.time()-t_start:.0f}s)")

# ── Save outputs ───────────────────────────────────────────────────────────────
flag_df  = pd.DataFrame(flag_rows)
stats_df = pd.DataFrame(stats_rows)

flag_df.to_csv(flag_summary_file, index=False)
stats_df.to_csv(site_stats_file, index=False)

# ── Print summary ──────────────────────────────────────────────────────────────
total_elapsed = time.time() - t_start
print(f"\nCompleted in {total_elapsed:.1f}s")
print(f"\nFlag summary saved to:  {flag_summary_file}")
print(f"Site stats saved to:    {site_stats_file}")

print(f"\nOverall flag distribution across all sites:")
for label, prefix in [('pv_site_net', 'pv'), ('ac_load_net / gross_load', 'load')]:
    total = flag_df['n_intervals'].sum()
    print(f"\n  {label}:")
    flag_names = {0:'clean', 1:'missing', 2:'flatline', 3:'DST gap',
                  4:'partial phase', 5:'invalid', 6:'too large'}
    for code in FLAG_CODES:
        col   = f'{prefix}_flag_{code}'
        count = flag_df[col].sum()
        pct   = count / total * 100
        if count > 0:
            print(f"    Flag {code} ({flag_names[code]:<14}): "
                  f"{count:>8,}  ({pct:.2f}%)")

print(f"\nSites with NaN remaining after infill:")
pv_nan   = (flag_df['pv_nan_remaining']   > 0).sum()
load_nan = (flag_df['load_nan_remaining'] > 0).sum()
print(f"  PV NaN remaining:   {pv_nan} sites")
print(f"  Load NaN remaining: {load_nan} sites")

print(f"\nClean percentage distribution by state:")
state_summary = flag_df.groupby('state').agg(
    sites=('edp_site_id','count'),
    pv_clean_pct_mean=('pv_clean_pct','mean'),
    load_clean_pct_mean=('load_clean_pct','mean')
).round(1)
print(state_summary.to_string())
