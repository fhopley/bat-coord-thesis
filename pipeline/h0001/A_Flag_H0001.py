"""
A_Flag_H0001.py

Flags the pre-aggregated solar/load data for the sample household H0001 with
the same quality flags as pipeline/edp/E_Flag_All_Sites.py.

Differences from the all-sites script:
  - Input is already wide (one row per 5-min interval, not long-format per
    circuit). No phase aggregation or pivot step is needed.
  - Timestamps in the raw file are tz-aware ISO 8601 strings (e.g.
    "2022-01-31T00:05:00+11:00"), parsed via utc=True then converted to AEST.
    They are interval-end marks, so no +5 min offset is applied.
  - The data is pre-aggregated across the site's 3 phases, so Flag 4
    (partial phase) is not applied.
  - Site parameters are hard-coded below rather than read from a metadata file.

Usage from project root:
    python3 pipeline/h0001/A_Flag_H0001.py 2022
    python3 pipeline/h0001/A_Flag_H0001.py          (all years in ALL_YEARS)

Inputs:  data/raw/H0001/edp_raw_H0001_{YYYY}.csv
Outputs: data/clean/edp/{YYYY}/edp_clean_H0001_{YYYY}.csv  (overwritten)

Flag codes (priority order, higher overwrites lower):
  0 = Clean
  1 = Missing           interval absent after AEST reindex
  2 = Flatline          12+ consecutive intervals within ±0.01 Wh
  3 = DST gap           autumn-back overwritten hour (missing in AEST)
  4 = Partial phase     not applied (data pre-aggregated across phases)
  5 = Invalid           pv < -20 Wh, or gross_load < 0
  6 = Implausibly large pv or gross_load exceeds physical ceiling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pytz
import time
import sys

# ── Static site metadata ───────────────────────────────────────────────────────
SITE_ID         = 'H0001'
SITE_TZ_STRING  = 'Australia/Sydney'
STATE           = 'NSW'

# ── Year(s) to process ─────────────────────────────────────────────────────────
ALL_YEARS = [2022, 2023, 2024, 2025]
if len(sys.argv) > 1:
    years_to_process = [int(sys.argv[1])]
else:
    years_to_process = ALL_YEARS

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent.parent
RAW_DIR    = ROOT_DIR / "data" / "raw" / SITE_ID

# ── Thresholds ─────────────────────────────────────────────────────────────────
PV_NOISE_FLOOR_WH     = -20.0    # pv below this → Flag 5 (invalid)
                                  # -20 Wh = ~-240 W — allows for inverter standby draw
PV_MAX_WH             = 1667.0   # 20 kW × 5/60 hr  (site is 11.6 kWac — conservative ceiling)
GROSS_LOAD_MAX_WH     = 2500.0   # 30 kW × 5/60 hr
IQR_MULTIPLIER        = 3.0
FLATLINE_TOLERANCE_WH = 0.01
FLATLINE_MIN_RUN      = 12
SOLAR_HOUR_START      = 9        # AEST inclusive
SOLAR_HOUR_END        = 16       # AEST exclusive

# ── DST autumn-back UTC timestamps (Australia/Sydney) ──────────────────────────
# Each entry: first UTC timestamp of the duplicated AEST hour, and number of
# 5-min intervals in that duplicated window (always 12 for a 1-hour DST step).
DST_AUTUMN_BACK_UTC = {
    2020: ('2020-04-04 16:00:00', 12),
    2021: ('2021-04-03 16:00:00', 12),
    2022: ('2022-04-02 16:00:00', 12),
    2023: ('2023-04-01 16:00:00', 12),
    2024: ('2024-04-06 16:00:00', 12),
    2025: ('2025-04-05 16:00:00', 12),
    2026: ('2026-04-05 16:00:00', 12),
}

# ── Expected intervals per year ────────────────────────────────────────────────
def expected_intervals(year):
    return 105_408 if year in (2020, 2024) else 105_120

# ── Flatline detector (shared by PV and load) ──────────────────────────────────
def flag_flatline(values, hours_aest, solar_only=False,
                  curtailment_threshold=None):
    """
    Returns a boolean array marking intervals that are part of a flatline run
    of FLATLINE_MIN_RUN or more consecutive intervals where adjacent values
    differ by no more than FLATLINE_TOLERANCE_WH.

    solar_only=True  → only inspect intervals in SOLAR_HOUR_START–SOLAR_HOUR_END
    curtailment_threshold → exempt runs whose value equals or exceeds this
                            (genuine PV curtailment at ceiling looks like flatline)
    """
    flags = np.zeros(len(values), dtype=bool)
    run_start = None
    for i in range(1, len(values)):
        if np.isnan(values[i]) or np.isnan(values[i - 1]):
            run_start = None
            continue
        if solar_only:
            if hours_aest[i] < SOLAR_HOUR_START or hours_aest[i] >= SOLAR_HOUR_END:
                run_start = None
                continue
        if abs(values[i] - values[i - 1]) <= FLATLINE_TOLERANCE_WH:
            if run_start is None:
                run_start = i - 1
        else:
            if run_start is not None and (i - run_start) >= FLATLINE_MIN_RUN:
                if curtailment_threshold is None or values[i - 1] < curtailment_threshold:
                    flags[run_start:i] = True
            run_start = None
    # Handle run that reaches end of array
    if run_start is not None and (len(values) - run_start) >= FLATLINE_MIN_RUN:
        if curtailment_threshold is None or values[run_start] < curtailment_threshold:
            flags[run_start:] = True
    return flags


# ══════════════════════════════════════════════════════════════════════════════
# Main processing loop
# ══════════════════════════════════════════════════════════════════════════════
t_total = time.time()
print(f"A_Flag_H0001 — site: {SITE_ID} | tz: {SITE_TZ_STRING} | state: {STATE}")
print(f"Years to process: {years_to_process}\n")
print("=" * 60)

for YEAR in years_to_process:
    t_year = time.time()
    print(f"\nYear: {YEAR}")

    # ── Paths for this year ────────────────────────────────────────────────────
    raw_file   = RAW_DIR / f"edp_raw_{SITE_ID}_{YEAR}.csv"
    clean_dir  = ROOT_DIR / "data" / "clean" / "edp" / str(YEAR)
    clean_dir.mkdir(parents=True, exist_ok=True)
    output_file = clean_dir / f"edp_clean_{SITE_ID}_{YEAR}.csv"

    if not raw_file.exists():
        print(f"  Raw file not found — skipping: {raw_file}")
        continue

    # ── Load raw data ──────────────────────────────────────────────────────────
    # Incoming columns:
    #   datetime, pv_site_net_wh, ac_load_net_wh, gross_load_wh,
    #   pv_flag, load_flag, pv_original_wh, ac_load_net_original_wh
    #
    # pv_flag and load_flag in the raw file are blank placeholders — they will
    # be fully overwritten by this script.
    df = pd.read_csv(raw_file)

    # ── Parse tz-aware timestamps → AEST (Australia/Brisbane, no DST) ─────────
    # The raw datetime strings carry UTC offset info (e.g. +11:00 or +10:00).
    # Parse to UTC first, then convert to AEST.  No +5 min shift — timestamps
    # are already interval-end marks.
    df['datetime_aest'] = (
        pd.to_datetime(df['datetime'], utc=True)
          .dt.tz_convert('Australia/Brisbane')
    )

    # ── Build complete AEST grid for the year ──────────────────────────────────
    n_intervals = expected_intervals(YEAR)
    aest_grid = pd.date_range(
        start=pd.Timestamp(f'{YEAR}-01-01 00:05:00', tz='Australia/Brisbane'),
        periods=n_intervals,
        freq='5min'
    )

    # ── Reindex to full AEST grid (inserts NaN rows for missing intervals) ─────
    df = df.set_index('datetime_aest')
    df = df.reindex(aest_grid)
    df.index.name = 'datetime_aest'
    df = df.reset_index()

    # ── Ensure gross_load_wh is present (may be NaN after reindex) ────────────
    # Recompute from components where both are available; preserve existing value
    # where already populated (avoids rounding differences in source file).
    recompute_mask = df['gross_load_wh'].isna() & df['pv_site_net_wh'].notna() & df['ac_load_net_wh'].notna()
    df.loc[recompute_mask, 'gross_load_wh'] = (
        df.loc[recompute_mask, 'ac_load_net_wh'] +
        df.loc[recompute_mask, 'pv_site_net_wh']
    )

    # ── Store originals (preserve raw values before any future infill) ─────────
    df['pv_original_wh']          = df['pv_site_net_wh']
    df['ac_load_net_original_wh'] = df['ac_load_net_wh']

    # ── Compute PV IQR threshold ───────────────────────────────────────────────
    hours_aest = df['datetime_aest'].dt.hour.values
    daytime_pv = df.loc[
        (df['datetime_aest'].dt.hour >= SOLAR_HOUR_START) &
        (df['datetime_aest'].dt.hour <  SOLAR_HOUR_END) &
        (df['pv_site_net_wh'].notna()) &
        (df['pv_site_net_wh'] > 0),
        'pv_site_net_wh'
    ]
    if len(daytime_pv) > 0:
        q1  = daytime_pv.quantile(0.25)
        q3  = daytime_pv.quantile(0.75)
        iqr = q3 - q1
        pv_threshold = min(q3 + IQR_MULTIPLIER * iqr, PV_MAX_WH)
    else:
        q3 = np.nan
        pv_threshold = PV_MAX_WH

    # ── Average daily gross load (complete days only) ──────────────────────────
    df['_date'] = df['datetime_aest'].dt.date
    daily_gross  = df.groupby('_date')['gross_load_wh'].sum()
    daily_counts = df.groupby('_date')['gross_load_wh'].count()
    complete_days = daily_counts[daily_counts == 288].index
    avg_daily_gross = (daily_gross[complete_days].mean()
                       if len(complete_days) > 0 else np.nan)
    df = df.drop(columns='_date')

    # ── DST gap timestamps ─────────────────────────────────────────────────────
    dst_gap_timestamps = set()
    if YEAR in DST_AUTUMN_BACK_UTC:
        dst_utc_str, n_dst = DST_AUTUMN_BACK_UTC[YEAR]
        dst_utc = pd.Timestamp(dst_utc_str, tz='UTC')
        dst_aest_start = (dst_utc.tz_convert('Australia/Brisbane')
                          + pd.Timedelta(minutes=5))
        dst_gap_timestamps = set(
            pd.date_range(start=dst_aest_start, periods=n_dst, freq='5min')
        )

    # ── Assign flags ───────────────────────────────────────────────────────────
    pv_wh    = df['pv_site_net_wh'].values
    load_wh  = df['ac_load_net_wh'].values
    gross_wh = df['gross_load_wh'].values

    pv_flag   = np.zeros(len(df), dtype=int)
    load_flag = np.zeros(len(df), dtype=int)

    pv_missing   = np.isnan(pv_wh)
    load_missing = np.isnan(load_wh)

    # Flag 1: missing
    pv_flag[pv_missing]     = 1
    load_flag[load_missing] = 1

    # Flag 2: flatline
    pv_flatline   = flag_flatline(
        pv_wh, hours_aest, solar_only=True,
        curtailment_threshold=q3 if not np.isnan(q3) else None
    )
    load_flatline = flag_flatline(load_wh, hours_aest, solar_only=False)
    pv_flag[pv_flatline   & ~pv_missing]   = 2
    load_flag[load_flatline & ~load_missing] = 2

    # Flag 3: DST gap
    if dst_gap_timestamps:
        dst_mask = df['datetime_aest'].dt.tz_convert('UTC').isin(
            [ts.tz_convert('UTC') for ts in dst_gap_timestamps]
        ).values
        pv_flag[dst_mask   & pv_missing]   = 3
        load_flag[dst_mask & load_missing] = 3

    # Flag 4: partial phase — NOT APPLIED
    # Data is pre-aggregated across all 3 phases; no per-phase breakdown available.

    # Flag 5: invalid
    pv_flag[~pv_missing & (pv_wh < PV_NOISE_FLOOR_WH)]                   = 5
    load_flag[~load_missing & ~pv_missing & (gross_wh < 0)]               = 5

    # Flag 6: implausibly large
    pv_flag[~pv_missing & (pv_wh > pv_threshold)]                         = 6
    load_flag[~load_missing & ~pv_missing & (gross_wh > GROSS_LOAD_MAX_WH)] = 6

    df['pv_flag']   = pv_flag
    df['load_flag'] = load_flag

    # ── Write output ───────────────────────────────────────────────────────────
    output_cols = [
        'datetime_aest', 'pv_site_net_wh', 'ac_load_net_wh',
        'gross_load_wh', 'pv_flag', 'load_flag',
        'pv_original_wh', 'ac_load_net_original_wh'
    ]
    df[output_cols].to_csv(output_file, index=False)

    # ── Summary ────────────────────────────────────────────────────────────────
    n_pv_flagged   = (pv_flag > 0).sum()
    n_load_flagged = (load_flag > 0).sum()
    elapsed = time.time() - t_year
    print(f"  PV flagged:   {n_pv_flagged:>5} / {len(df)}")
    print(f"  Load flagged: {n_load_flagged:>5} / {len(df)}")
    print(f"  PV Q3 (daytime): {q3:.2f} Wh" if not np.isnan(q3) else "  PV Q3: n/a")
    print(f"  Avg daily gross load: {avg_daily_gross:.1f} Wh" if not np.isnan(avg_daily_gross) else "  Avg daily gross load: n/a")
    print(f"  Output: {output_file}  ({elapsed:.1f}s)")

print(f"\n{'='*60}")
print(f"Done. Total elapsed: {time.time() - t_total:.1f}s")
