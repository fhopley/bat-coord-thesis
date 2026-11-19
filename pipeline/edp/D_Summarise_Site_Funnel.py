"""
D_Summarise_Site_Funnel.py

Produces a summary table showing the number of sites at each stage of the
selection funnel, broken down by state.

Stages:
  1. Raw data (YYYY_site_coverage.csv) — all sites present in YYYY raw data
  2. Coverage, inverter rating, and battery screen passed (YYYY_selected_sites.csv)
     — sites with no battery evidence pass automatically, same as before.
     Sites with battery evidence (metadata flag or a battery-labelled
     circuit_label in the raw data) pass only if B0_Battery_Site_Screen.py
     marked them usable; see docs/edp_data_spec.md, Battery sites. This stage is no
     longer "no battery" -- some sites counted here have a battery and a
     reconstructed gross_load.
  3. Consistency checks passed (YYYY_selected_sites_clean.csv)

Output: data/processed/edp/YYYY_site_funnel_summary.csv
"""

import pandas as pd
import sys
from pathlib import Path

# ── Year parameter — change this to rerun for a different year ─────────────────
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "edp"
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "edp"

coverage_file = PROCESSED_DIR / f"{YEAR}_site_coverage.csv"
metadata_file = RAW_DIR / "edp_sites_metadata_2020_2025.csv"
selected_file = PROCESSED_DIR / f"{YEAR}_selected_sites.csv"
clean_file    = PROCESSED_DIR / f"{YEAR}_selected_sites_clean.csv"
battery_screen_file = PROCESSED_DIR / f"{YEAR}_battery_site_screen.csv"
output_file   = PROCESSED_DIR / f"{YEAR}_site_funnel_summary.csv"

# ── Load metadata (deduplicated) ───────────────────────────────────────────────
meta = pd.read_csv(metadata_file)
meta = (meta.sort_values('inverter_ac_rating_kw', ascending=False)
            .drop_duplicates(subset='edp_site_id', keep='first')
            [['edp_site_id', 'state']])

# ── Stage 1: raw data ──────────────────────────────────────────────────────────
coverage = pd.read_csv(coverage_file, usecols=['edp_site_id'])
coverage = coverage.merge(meta, on='edp_site_id', how='left')
stage1 = coverage.groupby('state')['edp_site_id'].count().rename('raw_data')

# ── Stage 2: coverage, inverter rating, and battery screen ────────────────────
selected = pd.read_csv(selected_file, usecols=['edp_site_id', 'state'])
stage2 = selected.groupby('state')['edp_site_id'].count().rename('coverage_battery_screen')

# ── Stage 3: consistency checks passed ────────────────────────────────────────
clean = pd.read_csv(clean_file, usecols=['edp_site_id', 'state'])
stage3 = clean.groupby('state')['edp_site_id'].count().rename('consistency_pass')

# ── Diagnostic: battery-recovered sites within stage 2 ──────────────────────────
# How many of the sites in coverage_battery_screen are there specifically
# because they passed the battery reconstruction screen (as opposed to having
# no battery evidence at all, which is most sites). Must be gated on actually
# appearing in stage 2 -- B0_Battery_Site_Screen.py's own usable count is not
# enough on its own, since a site can pass the battery screen and still be
# excluded by B_Site_Selection.py's coverage or inverter filters. Not a funnel
# stage in its own right -- a subset of stage 2 -- so not included in the
# totals row sum logic below beyond what pandas does automatically for a
# normal column.
if battery_screen_file.exists():
    battery_screen = pd.read_csv(battery_screen_file, usecols=['edp_site_id', 'usable'])
    battery_recovered_ids = set(battery_screen.loc[battery_screen['usable'], 'edp_site_id'])
    recovered_in_stage2 = selected[selected['edp_site_id'].isin(battery_recovered_ids)]
    battery_recovered = recovered_in_stage2.groupby('state')['edp_site_id'].count().rename('battery_recovered')
else:
    battery_recovered = pd.Series(dtype=int, name='battery_recovered')

# ── Combine ────────────────────────────────────────────────────────────────────
summary = (pd.concat([stage1, stage2, battery_recovered, stage3], axis=1)
             .fillna(0)
             .astype(int)
             .sort_values('raw_data', ascending=False))

# Add totals row
summary.loc['TOTAL'] = summary.sum()

summary.index.name = 'state'
summary = summary.reset_index()

# ── Save and print ─────────────────────────────────────────────────────────────
summary.to_csv(output_file, index=False)
print(summary.to_string(index=False))
print(f"\nSaved to: {output_file}")