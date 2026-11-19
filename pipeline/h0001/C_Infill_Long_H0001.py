"""
C_Infill_Long_H0001.py

Applies Method 2 (synthetic day) infill to the H0001 clean file. Adapted from
pipeline/edp/I_Infill_Long_All_Sites.py, with the two multi-site lookups
replaced:

  SITE_STATE   selects the state column of the PV profile library.
  SITE_PV_Q3   Q3 of daytime (09:00-16:00 AEST) positive PV in the original
               (pre-infill) data, used to scale the normalised PV profile.
               Computed from pv_original_wh rather than read from
               YYYY_site_pv_stats.csv.

PV:   intervals still NaN or flagged after Method 1 are replaced with the
      state profile for that interval times SITE_PV_Q3.
Load: intervals still NaN or flagged after Method 1 are replaced with the
      mean of the two closest complete same-day-of-week days.

ac_load_net_wh is back-calculated as gross_load_wh - pv_site_net_wh.

Infill tracking: pv_infill, load_infill = 2 for Method 2.

Usage from project root:
    python3 pipeline/h0001/C_Infill_Long_H0001.py 2022

Inputs:  data/clean/edp/{YYYY}/edp_clean_H0001_{YYYY}.csv
         data/processed/edp/{YYYY}_pv_profile_library.csv
           (built by pipeline/edp/H_Build_PV_Profile.py from the EDP cohort;
            not included in the public snapshot)
Output:  data/clean/edp/{YYYY}/edp_clean_H0001_{YYYY}.csv  (updated in place)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys

SITE_ID    = 'H0001'
SITE_STATE = 'NSW'
SOLAR_HOUR_START = 9    # AEST inclusive, for SITE_PV_Q3
SOLAR_HOUR_END   = 16   # AEST exclusive

# ── Parameters ─────────────────────────────────────────────────────────────────
YEAR                       = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
MIN_COMPLETE_DAY_INTERVALS = 288
NEM_STATES                 = ['NSW', 'QLD', 'SA', 'VIC']

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
ROOT_DIR      = SCRIPT_DIR.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "edp"
clean_dir     = ROOT_DIR / "data" / "clean" / "edp" / str(YEAR)

profile_file = PROCESSED_DIR / f"{YEAR}_pv_profile_library.csv"
fpath        = clean_dir / f"edp_clean_{SITE_ID}_{YEAR}.csv"

# ── Validate ───────────────────────────────────────────────────────────────────
if SITE_STATE not in NEM_STATES:
    print(f"ERROR: {SITE_STATE} is not a NEM state — cannot apply PV profile infill.")
    sys.exit(1)

if not fpath.exists():
    print(f"ERROR: Clean file not found: {fpath}")
    sys.exit(1)

if not profile_file.exists():
    print(f"ERROR: PV profile library not found: {profile_file}")
    sys.exit(1)

# ── Load PV profile library ────────────────────────────────────────────────────
print(f"Year: {YEAR} | Site: {SITE_ID} | State: {SITE_STATE}")
print("Loading PV profile library...")
profile = pd.read_csv(profile_file, parse_dates=['datetime_aest'])
profile['datetime_aest'] = pd.to_datetime(profile['datetime_aest'])
profile = profile.set_index('datetime_aest')
profile_dict = profile[f'{SITE_STATE}_profile'].to_dict()

# ── Load clean file ────────────────────────────────────────────────────────────
print(f"Loading {fpath.name}...")
df = pd.read_csv(fpath, parse_dates=['datetime_aest'])
df['datetime_aest'] = pd.to_datetime(df['datetime_aest'])

# ── PV scaling factor ──────────────────────────────────────────────────────────
_hours = df['datetime_aest'].dt.tz_convert('Australia/Brisbane').dt.hour
_daytime_pv = df.loc[(_hours >= SOLAR_HOUR_START) & (_hours < SOLAR_HOUR_END)
                     & (df['pv_original_wh'] > 0), 'pv_original_wh']
SITE_PV_Q3 = _daytime_pv.quantile(0.75) if len(_daytime_pv) else np.nan
print(f"SITE_PV_Q3 (daytime original PV): {SITE_PV_Q3:.2f} Wh")

if 'pv_infill' not in df.columns:
    df['pv_infill'] = 0
if 'load_infill' not in df.columns:
    df['load_infill'] = 0

# ── Identify intervals needing replacement ─────────────────────────────────────
# Exclude intervals already filled by Method 1 — those should not be overwritten.
pv_needs   = ((df['pv_site_net_wh'].isna() | (df['pv_flag'] > 0)) &
              (df['pv_infill'] == 0)).values
load_needs = ((df['gross_load_wh'].isna()   | (df['load_flag'] > 0)) &
              (df['load_infill'] == 0)).values

if not pv_needs.any() and not load_needs.any():
    print("No intervals need Method 2 infill — nothing to do.")
    sys.exit(0)

print(f"Intervals needing Method 2 infill: PV={pv_needs.sum():,}  Load={load_needs.sum():,}\n")

t_start = time.time()

# ── PV infill — vectorised ─────────────────────────────────────────────────────
pv_values = df['pv_site_net_wh'].values.copy()
pv_infill = df['pv_infill'].values.copy()
pv_filled = 0

if not np.isnan(SITE_PV_Q3) and SITE_PV_Q3 > 0 and pv_needs.any():
    profile_vals = df['datetime_aest'].map(profile_dict).fillna(0.0).values
    pv_values[pv_needs] = profile_vals[pv_needs] * SITE_PV_Q3
    pv_infill[pv_needs] = 2
    pv_filled = int(pv_needs.sum())
    print(f"PV:   {pv_filled:,} intervals filled (Method 2)")
else:
    print("PV:   nothing to fill" if not pv_needs.any() else "PV:   skipped (no daytime solar data for SITE_PV_Q3)")

df['pv_site_net_wh'] = pv_values
df['pv_infill']      = pv_infill

# ── Gross load infill — vectorised ─────────────────────────────────────────────
gross_values = df['gross_load_wh'].values.copy()
load_infill  = df['load_infill'].values.copy()
load_filled  = 0

aest_dt    = df['datetime_aest'].dt.tz_convert('Australia/Brisbane')
dates      = aest_dt.dt.date.values
dows       = aest_dt.dt.day_of_week.values

# Build reference pool: complete days with Flag 0 or Method 1 infilled data
df['_date'] = dates
df['_good'] = (
    df['gross_load_wh'].notna() &
    ((df['load_flag'] == 0) | (df['load_infill'] == 1))
)
day_counts    = df.groupby('_date')['_good'].sum()
complete_days = set(day_counts[day_counts >= MIN_COMPLETE_DAY_INTERVALS].index)

# Build per-day-of-week reference: {dow: [(date, 288-element array), ...]}
dow_reference = {}
for date in sorted(complete_days):
    day_mask = dates == date
    if day_mask.sum() != MIN_COMPLETE_DAY_INTERVALS:
        continue
    dow = int(dows[day_mask][0])
    ref_vals = gross_values[day_mask]
    if dow not in dow_reference:
        dow_reference[dow] = []
    dow_reference[dow].append((date, ref_vals))

# Fill each gap day
gap_dates = np.unique(dates[load_needs])
for gap_date in gap_dates:
    gap_dow      = pd.Timestamp(gap_date).day_of_week
    gap_day_mask = (dates == gap_date) & load_needs

    if not gap_day_mask.any():
        continue
    if gap_dow not in dow_reference or len(dow_reference[gap_dow]) == 0:
        print(f"  WARNING: no reference days for {gap_date} (dow={gap_dow}) — leaving as NaN")
        continue

    # Two closest same-day-of-week reference days
    ref_days = sorted(
        dow_reference[gap_dow],
        key=lambda x: abs((pd.Timestamp(x[0]) - pd.Timestamp(gap_date)).days)
    )
    ref_profile = (
        (ref_days[0][1] + ref_days[1][1]) / 2
        if len(ref_days) >= 2
        else ref_days[0][1].copy()
    )

    # Align by position within day (both arrays are 288 intervals in time order)
    gap_day_indices = np.where(dates == gap_date)[0]
    needs_indices   = np.where(gap_day_mask)[0]
    day_start       = gap_day_indices[0]
    positions       = needs_indices - day_start

    gross_values[needs_indices] = ref_profile[positions]
    load_infill[needs_indices]  = 2
    load_filled += len(needs_indices)

print(f"Load: {load_filled:,} intervals filled (Method 2)")

df['gross_load_wh'] = gross_values
df['load_infill']   = load_infill

# ── Back-calculate ac_load_net_wh ──────────────────────────────────────────────
method2_mask = load_infill == 2
df.loc[method2_mask, 'ac_load_net_wh'] = (
    df.loc[method2_mask, 'gross_load_wh'] -
    df.loc[method2_mask, 'pv_site_net_wh']
)
# Fix any remaining NaN ac_load_net where both gross and pv are now available
# (can arise where Method 1 filled gross but PV was only filled here)
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

# ── Save ───────────────────────────────────────────────────────────────────────
df = df.drop(columns=['_date', '_good'], errors='ignore')
output_cols = ['datetime_aest', 'pv_site_net_wh', 'ac_load_net_wh',
               'gross_load_wh', 'pv_flag', 'load_flag',
               'pv_infill', 'load_infill',
               'pv_original_wh', 'ac_load_net_original_wh']
df[output_cols].to_csv(fpath, index=False)

# ── Summary ────────────────────────────────────────────────────────────────────
pv_remain   = int(np.isnan(pv_values).sum())
load_remain = int(np.isnan(gross_values).sum())
elapsed     = time.time() - t_start

print(f"\nComplete in {elapsed:.1f}s")
print(f"Output: {fpath}")
print(f"\nStill remaining after Method 2:")
print(f"  PV:   {pv_remain:,} NaN intervals")
print(f"  Load: {load_remain:,} NaN intervals")
