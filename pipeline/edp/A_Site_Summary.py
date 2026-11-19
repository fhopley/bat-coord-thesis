"""
A_Site_Summary.py

First script in the data preparation pipeline. Scans all monthly raw data
files for a given year and computes per-site coverage statistics for the
ac_load_net and pv_site_net circuit labels, plus the three known
battery-labelled circuit labels (battery_storage, load_battery,
pv_site_net_battery). See docs/edp_data_spec.md, Battery sites.

For each site and label, records:
  - Coverage percentage: distinct 5-minute timestamps present as a fraction
    of the total expected intervals for the year. Uses unix_time (local time)
    for deduplication, which correctly handles DST overlaps and out-of-order
    data within files.
  - Maximum gap in minutes: largest consecutive gap in the sorted unix_time
    sequence, indicating the longest continuous outage.
  - Phase count: number of distinct edp_device_and_circuit values seen,
    indicating single-phase (1) or multi-phase (2-3) metering.

For the three battery labels only, also records aggregate sign/magnitude
stats (sum of positive values, sum of absolute negative values, counts of
positive/negative/total intervals), needed by B0_Battery_Site_Screen.py to
apply the screening gates (docs/edp_data_spec.md, Battery sites) without a
second raw-data scan. Phases are collapsed (summed) per (site, unix_time,
label) within each chunk before the positive/negative split, consistent
with the phase-aggregation convention used elsewhere in the pipeline. Known
simplifying assumption: if a site's phases for one timestamp were ever
split across a chunk boundary, that interval's sign classification could be
marginally wrong. Not expected to matter given chunk size (500,000 rows)
versus typical per-timestamp row count (2-10), and in the same spirit as
the existing accepted, unquantified partial-phase risk noted in
edp_data_spec.md item 9.

Expected intervals: 105,120 for non-leap years (365 x 288), 105,408 for
leap years (366 x 288). 2020 and 2024 are leap years.

Input:  data/raw/edp/edp_data_YYYY_{MM}.csv  (12 monthly files)
Output: data/processed/edp/YYYY_site_coverage.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "edp"
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "edp"

# ── Year parameter — change this to rerun for a different year ─────────────────
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022

# ── Derived file paths ─────────────────────────────────────────────────────────
input_file_template = str(RAW_DIR / f"edp_data_{YEAR}_{{:02d}}.csv")
output_file = PROCESSED_DIR / f"{YEAR}_site_coverage.csv"

# ── Settings ───────────────────────────────────────────────────────────────────
CHUNK_SIZE = 500_000
LABELS = ['ac_load_net', 'pv_site_net']
BATTERY_LABELS = ['battery_storage', 'load_battery', 'pv_site_net_battery']
ALL_LABELS = LABELS + BATTERY_LABELS
EXPECTED_INTERVALS = 105_408 if YEAR in (2020, 2024) else 105_120

# ── Accumulators ───────────────────────────────────────────────────────────────
# {(site, label): set of unix_times}
unix_sets = defaultdict(set)
# {(site, label): set of phase labels}
phase_sets = defaultdict(set)
# Battery-label-only aggregate sign/magnitude accumulators, see docstring.
battery_sum_pos_wh = defaultdict(float)
battery_sum_neg_abs_wh = defaultdict(float)
battery_n_pos = defaultdict(int)
battery_n_neg = defaultdict(int)
battery_n_total = defaultdict(int)

# ── Single pass through all 12 monthly files ───────────────────────────────────
for month in range(1, 13):
    file_path = input_file_template.format(month)
    print(f"Processing: {file_path}")

    for chunk in pd.read_csv(
        file_path,
        usecols=['edp_site_id', 'unix_time', 'circuit_label', 'edp_device_and_circuit', 'real_energy'],
        chunksize=CHUNK_SIZE
    ):
        chunk['circuit_label'] = chunk['circuit_label'].str.strip()
        filtered = chunk[chunk['circuit_label'].isin(ALL_LABELS)].copy()
        filtered['edp_device_and_circuit'] = filtered['edp_device_and_circuit'].str.strip()

        for (site, label), group in filtered.groupby(['edp_site_id', 'circuit_label']):
            unix_sets[(site, label)].update(group['unix_time'].values)
            phase_sets[(site, label)].update(group['edp_device_and_circuit'].values)
            if label in BATTERY_LABELS:
                # Collapse phases per timestamp before the sign split (see
                # docstring for the chunk-boundary caveat on this).
                collapsed = group.groupby('unix_time')['real_energy'].sum()
                key = (site, label)
                battery_sum_pos_wh[key] += collapsed[collapsed > 0].sum()
                battery_sum_neg_abs_wh[key] += (-collapsed[collapsed < 0]).sum()
                battery_n_pos[key] += int((collapsed > 0).sum())
                battery_n_neg[key] += int((collapsed < 0).sum())
                battery_n_total[key] += len(collapsed)

print("Computing summary statistics...")

# ── Build output rows ──────────────────────────────────────────────────────────
site_ids = sorted(set(site for site, label in unix_sets.keys()))

rows = []
for site in site_ids:
    row = {'edp_site_id': site}
    for label in ALL_LABELS:
        key = (site, label)
        if key in unix_sets:
            times = sorted(unix_sets[key])
            count = len(times)
            coverage = round(count / EXPECTED_INTERVALS * 100, 1)
            diffs = np.diff(times)
            max_gap_min = int(diffs.max() / 60) if len(diffs) > 0 else 0
            phases = len(phase_sets[key])
        else:
            coverage = 0.0
            max_gap_min = None
            phases = 0
        row[f'{label}_coverage_pct'] = coverage
        row[f'{label}_max_gap_min'] = max_gap_min
        row[f'{label}_phases'] = phases
        if label in BATTERY_LABELS:
            row[f'{label}_sum_pos_wh'] = round(battery_sum_pos_wh.get(key, 0.0), 1)
            row[f'{label}_sum_neg_abs_wh'] = round(battery_sum_neg_abs_wh.get(key, 0.0), 1)
            row[f'{label}_n_pos'] = battery_n_pos.get(key, 0)
            row[f'{label}_n_neg'] = battery_n_neg.get(key, 0)
            row[f'{label}_n_total'] = battery_n_total.get(key, 0)
    rows.append(row)

summary_df = pd.DataFrame(rows)
summary_df.to_csv(output_file, index=False)
print(f"Saved coverage summary to: {output_file}")
print(f"Total sites found: {len(summary_df)}")