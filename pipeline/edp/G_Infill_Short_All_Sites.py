"""
G_Infill_Short_All_Sites.py

Applies short-gap infill (Method 1, linear interpolation) to all clean site
files for a given year.

Replaces both NaN intervals AND flagged non-NaN intervals — a flagged value
is unreliable and should be replaced regardless of whether it is NaN.

For each site, identifies runs of "needs replacement" intervals:
  - pv_site_net_wh: NaN or pv_flag > 0
  - gross_load_wh:  NaN or load_flag > 0

Then applies Method 1 (linear interpolation) where the run is short enough
and has clean (Flag 0) boundaries:

  PV (pv_site_net_wh):
    - 21:00-03:59 AEST (definitely dark): any run length → Method 1
    - 04:00-21:00 AEST (possible sunlight): runs <= 6 intervals → Method 1

  Gross load (gross_load_wh):
    - Runs <= 12 intervals (1 hour) → Method 1

  Boundary requirements:
    - Both boundaries clean (Flag 0, non-NaN): linear interpolation
    - One boundary at year start/end: flatline from single boundary value
    - Neither boundary clean: leave for Script H (Method 2)

  ac_load_net_wh is back-calculated as gross_load_wh - pv_site_net_wh
  after both channels are filled.

Infill tracking:
  pv_infill:   0 = original data, 1 = Method 1 (this script)
  load_infill: 0 = original data, 1 = Method 1 (this script)

Clean files are updated in place.

Input/output: data/clean/edp/YYYY/edp_clean_{SITE_ID}_{YYYY}.csv (all sites)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys

# ── Parameters ─────────────────────────────────────────────────────────────────
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022

MAX_GAP_LOAD_INTERVALS   = 12  # 1 hour
MAX_GAP_PV_DAY_INTERVALS = 6   # 30 min during possible-sunlight hours
DARK_HOUR_START          = 21  # inclusive — start of definitely-dark window
DARK_HOUR_END            = 4   # exclusive (dark window: 21:00 to 03:59 AEST)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent.parent
clean_dir  = ROOT_DIR / "data" / "clean" / "edp" / str(YEAR)

clean_files = sorted(clean_dir.glob(f"edp_clean_*_{YEAR}.csv"))
print(f"Year: {YEAR} | Sites to process: {len(clean_files)}\n")

# ── Helper: find contiguous runs needing replacement ──────────────────────────
def get_replace_runs(values, flags):
    """
    Return list of (start_idx, end_idx) inclusive for each run of intervals
    that need replacement: NaN values OR flagged (flag > 0) values.
    """
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

# ── Helper: check if hour falls in dark window ────────────────────────────────
def is_dark_hour(hour):
    return hour >= DARK_HOUR_START or hour < DARK_HOUR_END

# ── Helper: interpolate a single run ──────────────────────────────────────────
def interpolate_run(values, flags, start, end):
    """
    Linear interpolation between clean boundary values.
    Boundaries must be Flag 0 and non-NaN.
    - Both boundaries clean: linear interpolation
    - One boundary at year edge: flatline from available boundary
    - Neither boundary clean: return None (leave for Method 2)
    """
    n_total = len(values)
    n_fill  = end - start + 1

    at_year_start = (start == 0)
    at_year_end   = (end == n_total - 1)

    left_clean  = at_year_start or (
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
    elif left_clean:
        return np.full(n_fill, values[start - 1])
    else:
        return np.full(n_fill, values[end + 1])

# ── Process all sites ──────────────────────────────────────────────────────────
t_start = time.time()
total_pv_filled   = 0
total_load_filled = 0
total_pv_remain   = 0
total_load_remain = 0

for i, fpath in enumerate(clean_files):
    site_id = fpath.stem.replace('edp_clean_', '').replace(f'_{YEAR}', '')

    df = pd.read_csv(fpath, parse_dates=['datetime_aest'])
    df['datetime_aest'] = pd.to_datetime(df['datetime_aest'])
    n = len(df)

    # Reset infill tracking — this run supersedes any previous Method 1 pass
    df['pv_infill']   = 0
    df['load_infill'] = 0

    hours = df['datetime_aest'].dt.tz_convert('Australia/Brisbane').dt.hour.values

    # ── PV infill ──────────────────────────────────────────────────────────────
    pv_values = df['pv_site_net_wh'].values.copy()
    pv_flags  = df['pv_flag'].values.copy()
    pv_infill = np.zeros(n, dtype=int)

    pv_filled = 0
    for start, end in get_replace_runs(pv_values, pv_flags):
        run_len   = end - start + 1
        run_hours = hours[start:end + 1]
        all_dark  = all(is_dark_hour(h) for h in run_hours)
        max_gap   = n if all_dark else MAX_GAP_PV_DAY_INTERVALS

        if run_len > max_gap:
            continue

        filled = interpolate_run(pv_values, pv_flags, start, end)
        if filled is not None:
            pv_values[start:end + 1] = filled
            pv_infill[start:end + 1] = 1
            pv_filled += run_len

    df['pv_site_net_wh'] = pv_values
    df['pv_infill']      = pv_infill

    # ── Gross load infill ──────────────────────────────────────────────────────
    gross_values = df['gross_load_wh'].values.copy()
    load_flags   = df['load_flag'].values.copy()
    load_infill  = np.zeros(n, dtype=int)

    load_filled = 0
    for start, end in get_replace_runs(gross_values, load_flags):
        if (end - start + 1) > MAX_GAP_LOAD_INTERVALS:
            continue
        filled = interpolate_run(gross_values, load_flags, start, end)
        if filled is not None:
            gross_values[start:end + 1] = filled
            load_infill[start:end + 1]  = 1
            load_filled += (end - start + 1)

    df['gross_load_wh'] = gross_values
    df['load_infill']   = load_infill

    # ── Back-calculate ac_load_net_wh ──────────────────────────────────────────
    # Recalculate wherever load was filled
    load_filled_mask = load_infill == 1
    df.loc[load_filled_mask, 'ac_load_net_wh'] = (
        df.loc[load_filled_mask, 'gross_load_wh'] -
        df.loc[load_filled_mask, 'pv_site_net_wh']
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    output_cols = ['datetime_aest', 'pv_site_net_wh', 'ac_load_net_wh',
                   'gross_load_wh', 'pv_flag', 'load_flag',
                   'pv_infill', 'load_infill',
                   'pv_original_wh', 'ac_load_net_original_wh']
    df[output_cols].to_csv(fpath, index=False)

    # Remaining = still NaN or still flagged after Method 1
    pv_remain   = (np.isnan(pv_values) | (pv_flags > 0)).sum() - pv_filled
    load_remain = (np.isnan(gross_values) | (load_flags > 0)).sum() - load_filled
    # Simpler: just count what's left needing replacement
    pv_remain   = int((np.isnan(df['pv_site_net_wh'].values) |
                       (df['pv_flag'].values > 0)).sum())
    load_remain = int((np.isnan(df['gross_load_wh'].values) |
                       (df['load_flag'].values > 0)).sum())

    total_pv_filled   += pv_filled
    total_load_filled += load_filled
    total_pv_remain   += pv_remain
    total_load_remain += load_remain

    if (i + 1) % 50 == 0 or (i + 1) == len(clean_files):
        elapsed = time.time() - t_start
        print(f"  [{i+1:>3}/{len(clean_files)}] processed "
              f"({elapsed:.0f}s) | "
              f"PV filled: {total_pv_filled:,} | "
              f"Load filled: {total_load_filled:,}")

# ── Final summary ──────────────────────────────────────────────────────────────
total_elapsed = time.time() - t_start
print(f"\nComplete. {len(clean_files)} sites in "
      f"{total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"\nTotal Method 1 intervals filled:")
print(f"  PV:   {total_pv_filled:,}")
print(f"  Load: {total_load_filled:,}")
print(f"\nTotal still needing replacement (NaN or flagged) for Method 2:")
print(f"  PV:   {total_pv_remain:,}")
print(f"  Load: {total_load_remain:,}")