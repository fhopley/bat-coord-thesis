"""
E_Flag_All_Sites.py

Processes all selected sites in a single scan of the raw monthly files:
each file is read once and data for all sites is extracted simultaneously.

Battery site correction (see docs/edp_data_spec.md, Battery sites): for sites
B_Site_Selection.py marked as a usable battery reconstruction, the extra
battery-labelled channel is pulled alongside ac_load_net and pv_site_net,
and the stored ac_load_net_wh column (and everything derived from it --
gross_load_wh, ac_load_net_original_wh, Flags 1/5/6) is the counterfactual
"no battery" net site flow, not the raw meter reading. This is the value
F through L and the coordination model consume, and is the entire point of
the reconstruction -- see C_Check_Consistency.py's docstring for the full
formula. Flag 2 (flatline / stuck sensor) is the one exception: it
deliberately continues to test the RAW ac_load_net signal, for the same
reason as Check 3 in C_Check_Consistency.py -- a genuinely stuck raw sensor
could be masked by a legitimately varying battery term if tested on the
corrected value instead. For ordinary (non-battery) sites, none of this
applies and behaviour is identical to before.

For each site produces:
  data/clean/edp/YYYY/edp_clean_{SITE_ID}_{YYYY}.csv

Also produces a site statistics lookup file for Script G:
  data/processed/edp/YYYY_site_pv_stats.csv

Flag codes (priority order — higher overwrites lower):
  0 = Clean
  1 = Missing     — interval absent after AEST reindex
  2 = Flatline    — 12+ consecutive intervals within ±0.01 Wh
  3 = DST gap     — autumn-back overwritten hour (missing in AEST)
  4 = Partial phase — fewer phases than site maximum for that channel
  5 = Invalid     — pv < -20 Wh, or gross_load < 0
  6 = Implausibly large — pv or gross_load exceeds physical ceiling

TEST_MODE: set to True to run on January only and a subset of sites,
for fast end-to-end verification before a full overnight run.

Inputs:  data/raw/edp/edp_data_YYYY_{MM}.csv (12 monthly files)
         data/raw/edp/edp_sites_metadata_2020_2025.csv
         data/processed/edp/YYYY_selected_sites_clean.csv
Outputs: data/clean/edp/YYYY/edp_clean_{SITE_ID}_{YYYY}.csv (one per site)
         data/processed/edp/YYYY_site_pv_stats.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import pytz
import time
import sys

# ── Year parameter ─────────────────────────────────────────────────────────────
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022

# ── Test mode ──────────────────────────────────────────────────────────────────
TEST_MODE        = False  # Set True for fast end-to-end test on subset
TEST_MONTHS      = [1]    # Months to process in test mode
TEST_SITES_PER_STATE = 3  # S and W sites per state in test mode

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR   = SCRIPT_DIR.parent.parent
RAW_DIR    = ROOT_DIR / "data" / "raw" / "edp"
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "edp"

clean_dir  = ROOT_DIR / "data" / "clean" / "edp" / str(YEAR)
clean_dir.mkdir(parents=True, exist_ok=True)

metadata_file    = RAW_DIR / "edp_sites_metadata_2020_2025.csv"
selected_file    = PROCESSED_DIR / f"{YEAR}_selected_sites_clean.csv"
site_stats_file  = PROCESSED_DIR / f"{YEAR}_site_pv_stats.csv"

# ── Thresholds ─────────────────────────────────────────────────────────────────
PV_NOISE_FLOOR_WH     = -20.0   # pv below this → Flag 5 (invalid)
                                 # -20 Wh = ~-240W — allows for inverter standby draw
PV_MAX_WH             = 1667.0   # 20 kW × 5/60 hr
GROSS_LOAD_MAX_WH     = 2500.0   # 30 kW × 5/60 hr
IQR_MULTIPLIER        = 3.0
FLATLINE_TOLERANCE_WH = 0.01
FLATLINE_MIN_RUN      = 12
SOLAR_HOUR_START      = 9        # AEST inclusive
SOLAR_HOUR_END        = 16       # AEST exclusive
CHUNK_SIZE            = 500_000
CORE_LABELS           = ['ac_load_net', 'pv_site_net']
BATTERY_LABELS         = ['battery_storage', 'load_battery', 'pv_site_net_battery']
USECOLS               = ['edp_site_id', 'unix_time', 'datetime',
                         'edp_device_and_circuit', 'circuit_label', 'real_energy']

# ── State-based timezone defaults (used when metadata timezone is missing) ─────
STATE_DEFAULT_TZ = {
    'NSW': 'Australia/Sydney',
    'VIC': 'Australia/Melbourne',
    'SA':  'Australia/Adelaide',
    'QLD': 'Australia/Brisbane',
    'WA':  'Australia/Perth',
    'TAS': 'Australia/Hobart',
    'NT':  'Australia/Darwin',
    'ACT': 'Australia/Sydney',
}
# See docs/edp_data_spec.md, Timestamps and daylight saving.
DST_AUTUMN_BACK_UTC = {
    ('Australia/Sydney',    2020): ('2020-04-04 16:00:00', 12),
    ('Australia/Sydney',    2021): ('2021-04-03 16:00:00', 12),
    ('Australia/Sydney',    2022): ('2022-04-02 16:00:00', 12),
    ('Australia/Sydney',    2023): ('2023-04-01 16:00:00', 12),
    ('Australia/Sydney',    2024): ('2024-04-06 16:00:00', 12),
    ('Australia/Sydney',    2025): ('2025-04-05 16:00:00', 12),
    ('Australia/Melbourne', 2020): ('2020-04-04 16:00:00', 12),
    ('Australia/Melbourne', 2021): ('2021-04-03 16:00:00', 12),
    ('Australia/Melbourne', 2022): ('2022-04-02 16:00:00', 12),
    ('Australia/Melbourne', 2023): ('2023-04-01 16:00:00', 12),
    ('Australia/Melbourne', 2024): ('2024-04-06 16:00:00', 12),
    ('Australia/Melbourne', 2025): ('2025-04-05 16:00:00', 12),
    ('Australia/Adelaide',  2020): ('2020-04-04 16:30:00', 12),
    ('Australia/Adelaide',  2021): ('2021-04-03 16:30:00', 12),
    ('Australia/Adelaide',  2022): ('2022-04-02 16:30:00', 12),
    ('Australia/Adelaide',  2023): ('2023-04-01 16:30:00', 12),
    ('Australia/Adelaide',  2024): ('2024-04-06 16:30:00', 12),
    ('Australia/Adelaide',  2025): ('2025-04-05 16:30:00', 12),
    ('Australia/Hobart',    2020): ('2020-04-04 16:00:00', 12),
    ('Australia/Hobart',    2021): ('2021-04-03 16:00:00', 12),
    ('Australia/Hobart',    2022): ('2022-04-02 16:00:00', 12),
    ('Australia/Hobart',    2023): ('2023-04-01 16:00:00', 12),
    ('Australia/Hobart',    2024): ('2024-04-06 16:00:00', 12),
    ('Australia/Hobart',    2025): ('2025-04-05 16:00:00', 12),
}

# ── Load metadata ──────────────────────────────────────────────────────────────
meta = pd.read_csv(metadata_file)
meta = (meta.sort_values('inverter_ac_rating_kw', ascending=False)
            .drop_duplicates(subset='edp_site_id', keep='first'))
meta_dict = meta.set_index('edp_site_id').to_dict('index')

# ── Load selected sites ────────────────────────────────────────────────────────
selected = pd.read_csv(selected_file)
NEM_STATES = ['NSW', 'QLD', 'SA', 'VIC']
selected = selected[selected['state'].isin(NEM_STATES)]

if TEST_MODE:
    print(f"*** TEST MODE: months {TEST_MONTHS}, "
          f"{TEST_SITES_PER_STATE} S+W sites per state ***\n")
    test_sites = []
    for state in NEM_STATES:
        for prefix in ['S', 'W']:
            subset = selected[
                (selected['state'] == state) &
                (selected['edp_site_id'].str.startswith(prefix))
            ].head(TEST_SITES_PER_STATE)
            test_sites.append(subset)
    selected = pd.concat(test_sites).reset_index(drop=True)
    months_to_process = TEST_MONTHS
else:
    months_to_process = list(range(1, 13))

site_ids  = set(selected['edp_site_id'])
print(f"Year: {YEAR} | Sites to process: {len(site_ids)}")
print(f"Months to process: {months_to_process}\n")

# ── Battery reconstruction lookup (see edp_data_spec.md, Battery sites) ──────────────
# {edp_site_id: (battery_channel_label, battery_sign_convention)}. Empty dict,
# and identical behaviour to before this change, if selected_sites_clean.csv
# predates B0_Battery_Site_Screen.py / doesn't have these columns.
if 'battery_channel_label' in selected.columns:
    battery_lookup = {
        row.edp_site_id: (row.battery_channel_label, row.battery_sign_convention)
        for row in selected.itertuples()
        if pd.notna(row.battery_channel_label)
    }
else:
    battery_lookup = {}

battery_labels_in_use = sorted({label for label, _ in battery_lookup.values()})
LABELS = CORE_LABELS + battery_labels_in_use
if battery_labels_in_use:
    print(f"Battery-corrected sites: {len(battery_lookup)} "
          f"(channels in use: {battery_labels_in_use})\n")

# ── Expected intervals ─────────────────────────────────────────────────────────
EXPECTED_INTERVALS = 105_408 if YEAR in (2020, 2024) else 105_120

# ── AEST grid (same for all sites) ────────────────────────────────────────────
aest_grid_start = pd.Timestamp(f'{YEAR}-01-01 00:05:00', tz='Australia/Brisbane')
aest_grid = pd.date_range(
    start=aest_grid_start, periods=EXPECTED_INTERVALS, freq='5min'
)
if TEST_MODE:
    # Restrict grid to test months
    test_start = pd.Timestamp(f'{YEAR}-01-01 00:05:00', tz='Australia/Brisbane')
    test_end   = pd.Timestamp(
        f'{YEAR}-{max(TEST_MONTHS):02d}-01 00:00:00', tz='Australia/Brisbane'
    ) + pd.DateOffset(months=1)
    aest_grid = aest_grid[(aest_grid >= test_start) & (aest_grid < test_end)]

# ── Accumulator: raw rows per site ────────────────────────────────────────────
site_raw = defaultdict(list)

# ── Single scan of monthly files ──────────────────────────────────────────────
t_start = time.time()
print("=" * 60)
print("Pass 1: Scanning raw files")
print("=" * 60)

for month in months_to_process:
    file_path = RAW_DIR / f"edp_data_{YEAR}_{month:02d}.csv"
    t_month = time.time()
    rows_extracted = 0

    for chunk in pd.read_csv(file_path, usecols=USECOLS, chunksize=CHUNK_SIZE):
        chunk['circuit_label'] = chunk['circuit_label'].str.strip()
        filtered = chunk[
            chunk['edp_site_id'].isin(site_ids) &
            chunk['circuit_label'].isin(LABELS)
        ].copy()
        if not filtered.empty:
            filtered['edp_device_and_circuit'] = (
                filtered['edp_device_and_circuit'].str.strip()
            )
            for site_id, group in filtered.groupby('edp_site_id'):
                site_raw[site_id].append(group)
            rows_extracted += len(filtered)

    elapsed = time.time() - t_month
    print(f"  Month {month:02d}: {rows_extracted:>8,} rows extracted "
          f"({elapsed:.1f}s)")

print(f"\nIngest complete. Total elapsed: {time.time()-t_start:.1f}s\n")

# ── Process each site ──────────────────────────────────────────────────────────
print("=" * 60)
print("Pass 2: Processing sites")
print("=" * 60)

flag_names = {0:'clean', 1:'missing', 2:'flatline', 3:'DST gap',
              4:'partial phase', 5:'invalid', 6:'too large'}

site_stats_rows = []
t_pass2 = time.time()

for i, site_id in enumerate(sorted(site_ids)):
    t_site = time.time()

    site_meta   = meta_dict.get(site_id, {})
    site_tz_str = site_meta.get('site_timezone', 'Australia/Sydney')
    state       = site_meta.get('state', '??')

    # Guard against NaN timezone (missing metadata) — derive from state
    if not isinstance(site_tz_str, str):
        site_tz_str = STATE_DEFAULT_TZ.get(state, 'Australia/Sydney')
        print(f"  [{i+1:>3}/{len(site_ids)}] {site_id} ({state}): "
              f"WARNING — no timezone in metadata, defaulting to {site_tz_str}")

    site_tz = pytz.timezone(site_tz_str)

    # Skip if clean file already exists (allows safe re-run after crash)
    output_file = clean_dir / f"edp_clean_{site_id}_{YEAR}.csv"
    if output_file.exists():
        print(f"  [{i+1:>3}/{len(site_ids)}] {site_id} ({state}): "
              f"already exists — skipping")
        continue

    # Concatenate all chunks for this site
    if site_id not in site_raw or not site_raw[site_id]:
        print(f"  [{i+1:>3}/{len(site_ids)}] {site_id} ({state}): "
              f"NO DATA — skipping")
        continue

    raw = pd.concat(site_raw[site_id]).reset_index(drop=True)

    # ── Convert timestamps to AEST interval-end ────────────────────────────────
    raw['datetime_local'] = pd.to_datetime(raw['datetime'])
    raw['datetime_local'] = raw['datetime_local'].dt.tz_localize(
        site_tz, ambiguous=True, nonexistent='NaT'
    )
    raw['datetime_aest'] = (raw['datetime_local']
                             .dt.tz_convert('Australia/Brisbane')
                             + pd.Timedelta(minutes=5))
    raw = raw.dropna(subset=['datetime_aest'])

    # ── Phase aggregation ──────────────────────────────────────────────────────
    phase_counts = (raw.groupby(['datetime_aest', 'circuit_label'])
                       ['edp_device_and_circuit'].nunique()
                       .rename('phase_count').reset_index())
    agg = (raw.groupby(['datetime_aest', 'circuit_label'])['real_energy']
               .sum().reset_index())
    agg = agg.merge(phase_counts, on=['datetime_aest', 'circuit_label'])

    wide = agg.pivot_table(
        index='datetime_aest',
        columns='circuit_label',
        values=['real_energy', 'phase_count']
    )
    # Flatten the (value, circuit_label) MultiIndex columns by name rather than
    # position — the fixed positional rename this replaced only worked because
    # exactly two labels (ac_load_net, pv_site_net) were ever present. A
    # battery site's raw rows also include its one battery-labelled channel
    # (see battery_lookup below), so the column set is now site-dependent and
    # must be named explicitly rather than assumed.
    wide.columns = [
        f'{label}_wh' if value == 'real_energy' else f'{label}_phases'
        for value, label in wide.columns
    ]
    wide = wide.reset_index()

    max_phases_load = int(wide['ac_load_net_phases'].max())
    max_phases_pv   = int(wide['pv_site_net_phases'].max())

    # ── Reindex to AEST grid ───────────────────────────────────────────────────
    wide = wide.set_index('datetime_aest').reindex(aest_grid)
    wide.index.name = 'datetime_aest'
    wide = wide.reset_index()

    # ── Battery correction (see edp_data_spec.md, Battery sites) ─────────────────────
    # ac_load_net_wh_raw is kept only for Flag 2's flatline (sensor-fault)
    # check below, which deliberately continues to test the raw meter signal
    # rather than the corrected one — same reasoning as Check 3 in
    # C_Check_Consistency.py: a genuinely stuck raw sensor could be masked by
    # a legitimately varying battery term if the corrected value were used
    # instead. Everywhere else (the stored ac_load_net_wh column itself, and
    # gross_load_wh, which feeds Flags 1, 5, and 6) uses the corrected value —
    # this is the counterfactual "no battery" net site flow that downstream
    # scripts (F through L) and the coordination model consume, and is the
    # entire point of doing the reconstruction. If the battery channel has no
    # value for a given interval, the corrected value silently falls back to
    # the raw ac_load_net_wh for that interval — same known simplification as
    # C_Check_Consistency.py.
    ac_load_net_wh_raw = wide['ac_load_net_wh'].copy()
    battery_label, battery_sign = battery_lookup.get(site_id, (None, None))
    if battery_label and f'{battery_label}_wh' in wide.columns:
        batt_col = wide[f'{battery_label}_wh']
        if battery_sign == 'subtract':
            corrected = wide['ac_load_net_wh'] - batt_col
        else:
            corrected = wide['ac_load_net_wh'] + batt_col
        corrected = corrected.where(batt_col.notna(), wide['ac_load_net_wh'])
        wide['ac_load_net_wh'] = corrected

    wide['gross_load_wh'] = wide['ac_load_net_wh'] + wide['pv_site_net_wh']

    # ── PV IQR threshold ───────────────────────────────────────────────────────
    hours_aest = wide['datetime_aest'].dt.hour.values
    daytime_pv = wide.loc[
        (wide['datetime_aest'].dt.hour >= SOLAR_HOUR_START) &
        (wide['datetime_aest'].dt.hour < SOLAR_HOUR_END) &
        (wide['pv_site_net_wh'].notna()) &
        (wide['pv_site_net_wh'] > 0),
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

    # ── Average daily gross load ───────────────────────────────────────────────
    wide['date'] = wide['datetime_aest'].dt.date
    daily_gross = wide.groupby('date')['gross_load_wh'].sum()
    # Only use days with complete data (288 intervals, none NaN)
    daily_counts = wide.groupby('date')['gross_load_wh'].count()
    complete_days = daily_counts[daily_counts == 288].index
    avg_daily_gross = daily_gross[complete_days].mean() if len(complete_days) > 0 else np.nan
    wide = wide.drop(columns='date')

    # ── DST gap timestamps ─────────────────────────────────────────────────────
    dst_key = (site_tz_str, YEAR)
    dst_gap_timestamps = set()
    if dst_key in DST_AUTUMN_BACK_UTC:
        dst_utc_str, n_intervals = DST_AUTUMN_BACK_UTC[dst_key]
        dst_utc = pd.Timestamp(dst_utc_str, tz='UTC')
        dst_aest_start = (dst_utc.tz_convert('Australia/Brisbane')
                          + pd.Timedelta(minutes=5))
        dst_gap_timestamps = set(
            pd.date_range(start=dst_aest_start, periods=n_intervals, freq='5min')
        )

    # ── Assign flags ───────────────────────────────────────────────────────────
    pv_wh    = wide['pv_site_net_wh'].values
    load_wh  = wide['ac_load_net_wh'].values          # corrected, for missing/Flag5/Flag6
    load_wh_raw = ac_load_net_wh_raw.values            # raw, for Flag 2 flatline only — see note above
    gross_wh = wide['gross_load_wh'].values
    pv_phases   = wide['pv_site_net_phases'].values
    load_phases = wide['ac_load_net_phases'].values

    pv_flag   = np.zeros(len(wide), dtype=int)
    load_flag = np.zeros(len(wide), dtype=int)

    pv_missing   = np.isnan(pv_wh)
    load_missing = np.isnan(load_wh)

    # Flag 1: missing
    pv_flag[pv_missing]     = 1
    load_flag[load_missing] = 1

    # Flag 2: flatline
    def flag_flatline(values, solar_only=False, curtailment_threshold=None):
        flags = np.zeros(len(values), dtype=bool)
        run_start = None
        for i in range(1, len(values)):
            if np.isnan(values[i]) or np.isnan(values[i-1]):
                run_start = None
                continue
            if solar_only:
                if hours_aest[i] < SOLAR_HOUR_START or hours_aest[i] >= SOLAR_HOUR_END:
                    run_start = None
                    continue
            if abs(values[i] - values[i-1]) <= FLATLINE_TOLERANCE_WH:
                if run_start is None:
                    run_start = i - 1
            else:
                if run_start is not None and (i - run_start) >= FLATLINE_MIN_RUN:
                    if curtailment_threshold is None or values[i-1] < curtailment_threshold:
                        flags[run_start:i] = True
                run_start = None
        if run_start is not None and (len(values) - run_start) >= FLATLINE_MIN_RUN:
            if curtailment_threshold is None or values[run_start] < curtailment_threshold:
                flags[run_start:] = True
        return flags

    pv_flatline   = flag_flatline(pv_wh, solar_only=True,
                                   curtailment_threshold=q3 if not np.isnan(q3) else None)
    load_flatline = flag_flatline(load_wh_raw, solar_only=False)
    pv_flag[pv_flatline & ~pv_missing]       = 2
    load_flag[load_flatline & ~load_missing] = 2

    # Flag 3: DST gap
    if dst_gap_timestamps:
        dst_mask = wide['datetime_aest'].dt.tz_convert('UTC').isin(
            [ts.tz_convert('UTC') for ts in dst_gap_timestamps]
        ).values
        pv_flag[dst_mask & pv_missing]     = 3
        load_flag[dst_mask & load_missing] = 3

    # Flag 4: partial phase
    if max_phases_pv > 1:
        pv_flag[~pv_missing & (pv_phases < max_phases_pv)] = 4
    if max_phases_load > 1:
        load_flag[~load_missing & (load_phases < max_phases_load)] = 4

    # Flag 5: invalid
    pv_flag[~pv_missing & (pv_wh < PV_NOISE_FLOOR_WH)]             = 5
    load_flag[~load_missing & ~pv_missing & (gross_wh < 0)]         = 5

    # Flag 6: implausibly large
    pv_flag[~pv_missing & (pv_wh > pv_threshold)]                   = 6
    load_flag[~load_missing & ~pv_missing & (gross_wh > GROSS_LOAD_MAX_WH)] = 6

    wide['pv_flag']   = pv_flag
    wide['load_flag'] = load_flag

    # ── Store original values before any infill ────────────────────────────────
    # These columns preserve the raw extracted values permanently.
    # Script G and H write to pv_site_net_wh / ac_load_net_wh but never
    # touch these columns, allowing comparison of original vs infilled values.
    wide['pv_original_wh']         = wide['pv_site_net_wh']
    wide['ac_load_net_original_wh'] = wide['ac_load_net_wh']

    # ── Save clean file ────────────────────────────────────────────────────────
    output_cols = ['datetime_aest', 'pv_site_net_wh', 'ac_load_net_wh',
                   'gross_load_wh', 'pv_flag', 'load_flag',
                   'pv_original_wh', 'ac_load_net_original_wh']
    wide[output_cols].to_csv(output_file, index=False)

    # ── Accumulate site stats ──────────────────────────────────────────────────
    site_stats_rows.append({
        'edp_site_id':            site_id,
        'state':                  state,
        'site_timezone':          site_tz_str,
        'pv_q3_wh':               round(q3, 2) if not np.isnan(q3) else np.nan,
        'avg_daily_gross_load_wh': round(avg_daily_gross, 1) if not np.isnan(avg_daily_gross) else np.nan,
        'max_phases_load':        max_phases_load,
        'max_phases_pv':          max_phases_pv,
    })

    # ── Progress print ─────────────────────────────────────────────────────────
    n_pv_flagged   = (pv_flag > 0).sum()
    n_load_flagged = (load_flag > 0).sum()
    elapsed_site   = time.time() - t_site
    elapsed_total  = time.time() - t_start
    print(f"  [{i+1:>3}/{len(site_ids)}] {site_id} ({state}) | "
          f"PV flagged: {n_pv_flagged:>5} | Load flagged: {n_load_flagged:>5} | "
          f"Site: {elapsed_site:.1f}s | Total: {elapsed_total:.0f}s")

# ── Save site stats ────────────────────────────────────────────────────────────
site_stats_df = pd.DataFrame(site_stats_rows)
site_stats_df.to_csv(site_stats_file, index=False)

# ── Final summary ──────────────────────────────────────────────────────────────
total_elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"Complete. {len(site_stats_rows)} sites processed in "
      f"{total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"\nSites processed by state:")
print(site_stats_df.groupby('state')['edp_site_id'].count().to_string())
print(f"\nSite stats saved to: {site_stats_file}")
if TEST_MODE:
    print(f"\n*** TEST MODE was active — re-run with TEST_MODE=False for full run ***")