"""
B_Infill_Short_H0001.py

Applies short-gap infill (Method 1, linear interpolation) to the H0001 clean
file. The rules and helpers are those of pipeline/edp/G_Infill_Short_All_Sites.py,
applied to one site rather than every clean CSV in the year folder.

Intervals needing replacement:
  pv_site_net_wh:  NaN or pv_flag > 0
  gross_load_wh:   NaN or load_flag > 0

Run length limits:
  PV, 21:00-03:59 AEST (dark):            any length
  PV, 04:00-20:59 AEST (possible sun):    <= 6 intervals
  Gross load:                             <= 12 intervals (1 hour)

Boundaries:
  Both clean (flag 0, non-NaN):  linear interpolation
  One at year start/end:         flatline from the available boundary
  Only one clean:                flatline from the clean boundary
  Neither clean:                 left for Method 2 (C_Infill_Long_H0001.py)

ac_load_net_wh is back-calculated as gross_load_wh - pv_site_net_wh where load
was filled.

Infill tracking:
  pv_infill, load_infill: 0 = original data, 1 = Method 1

Usage from project root:
    python3 pipeline/h0001/B_Infill_Short_H0001.py 2022

Input/output: data/clean/edp/{YYYY}/edp_clean_H0001_{YYYY}.csv (updated in place)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SITE_ID = 'H0001'

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
MAX_GAP_LOAD_INTERVALS   = 12  # 1 hour
MAX_GAP_PV_DAY_INTERVALS = 6   # 30 min during possible-sunlight hours
DARK_HOUR_START          = 21  # inclusive
DARK_HOUR_END            = 4   # exclusive

ROOT_DIR = Path(__file__).parent.parent.parent
fpath = ROOT_DIR / 'data' / 'clean' / 'edp' / str(YEAR) / f'edp_clean_{SITE_ID}_{YEAR}.csv'


def get_replace_runs(values, flags):
    """(start, end) inclusive index pairs for runs of NaN or flagged intervals."""
    needs_replace = np.isnan(values) | (flags > 0)
    runs = []
    i = 0
    while i < len(needs_replace):
        if needs_replace[i]:
            j = i
            while j < len(needs_replace) and needs_replace[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def is_dark_hour(hour):
    return hour >= DARK_HOUR_START or hour < DARK_HOUR_END


def interpolate_run(values, flags, start, end):
    """Fill values for one run, or None when neither boundary is clean."""
    n_total = len(values)
    n_fill = end - start + 1
    at_year_start = (start == 0)
    at_year_end = (end == n_total - 1)
    left_clean = at_year_start or (
        not np.isnan(values[start - 1]) and flags[start - 1] == 0)
    right_clean = at_year_end or (
        not np.isnan(values[end + 1]) and flags[end + 1] == 0)
    if not left_clean and not right_clean:
        return None
    if at_year_start:
        return np.full(n_fill, values[end + 1])
    if at_year_end:
        return np.full(n_fill, values[start - 1])
    if left_clean and right_clean:
        return np.linspace(values[start - 1], values[end + 1], n_fill + 2)[1:-1]
    if left_clean:
        return np.full(n_fill, values[start - 1])
    return np.full(n_fill, values[end + 1])


if not fpath.exists():
    print(f"ERROR: clean file not found: {fpath}")
    sys.exit(1)

t_start = time.time()
df = pd.read_csv(fpath)
df['datetime_aest'] = pd.to_datetime(df['datetime_aest'])
n = len(df)
hours = df['datetime_aest'].dt.tz_convert('Australia/Brisbane').dt.hour.values

# PV
pv_values = df['pv_site_net_wh'].values.copy()
pv_flags = df['pv_flag'].values.copy()
pv_infill = np.zeros(n, dtype=int)
for start, end in get_replace_runs(pv_values, pv_flags):
    run_len = end - start + 1
    all_dark = all(is_dark_hour(h) for h in hours[start:end + 1])
    max_gap = n if all_dark else MAX_GAP_PV_DAY_INTERVALS
    if run_len > max_gap:
        continue
    filled = interpolate_run(pv_values, pv_flags, start, end)
    if filled is not None:
        pv_values[start:end + 1] = filled
        pv_infill[start:end + 1] = 1
df['pv_site_net_wh'] = pv_values
df['pv_infill'] = pv_infill

# Gross load
gross_values = df['gross_load_wh'].values.copy()
load_flags = df['load_flag'].values.copy()
load_infill = np.zeros(n, dtype=int)
for start, end in get_replace_runs(gross_values, load_flags):
    if (end - start + 1) > MAX_GAP_LOAD_INTERVALS:
        continue
    filled = interpolate_run(gross_values, load_flags, start, end)
    if filled is not None:
        gross_values[start:end + 1] = filled
        load_infill[start:end + 1] = 1
df['gross_load_wh'] = gross_values
df['load_infill'] = load_infill

mask = load_infill == 1
df.loc[mask, 'ac_load_net_wh'] = df.loc[mask, 'gross_load_wh'] - df.loc[mask, 'pv_site_net_wh']

output_cols = ['datetime_aest', 'pv_site_net_wh', 'ac_load_net_wh',
               'gross_load_wh', 'pv_flag', 'load_flag',
               'pv_infill', 'load_infill',
               'pv_original_wh', 'ac_load_net_original_wh']
df[output_cols].to_csv(fpath, index=False)

pv_remain = int(((df['pv_flag'] > 0) | df['pv_site_net_wh'].isna()).sum() - (pv_infill == 1).sum())
load_remain = int(((df['load_flag'] > 0) | df['gross_load_wh'].isna()).sum() - (load_infill == 1).sum())
print(f"B_Infill_Short_H0001 — year {YEAR}")
print(f"  Method 1 filled: PV={int((pv_infill == 1).sum()):,}  Load={int((load_infill == 1).sum()):,}")
print(f"  Left for Method 2: PV={pv_remain:,}  Load={load_remain:,}")
print(f"  Output: {fpath}  ({time.time() - t_start:.1f}s)")
