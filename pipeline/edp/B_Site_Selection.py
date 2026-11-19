"""
B_Site_Selection.py

Filters sites from the annual coverage summary to produce a list of sites
suitable for analysis. Applies four sequential filters:

  1. ac_load_net coverage: site must have at least MIN_AC_LOAD_COVERAGE_PCT%
     of expected 5-minute intervals present across the year for ac_load_net,
     and no more than MAX_COVERAGE_PCT%. Coverage above 100% is physically
     impossible for one legitimate meter -- it means more distinct timestamps
     were observed than exist in the year, consistent with the documented
     W0017 pattern (edp_data_spec.md: two households sharing one site ID,
     each contributing its own near-complete year of readings, so the union
     of their timestamps exceeds what one meter could produce). Previously
     nothing caught this except luck at the consistency-check stage; this
     is now an explicit, auditable filter.

  2. pv_site_net coverage: same two-sided threshold applied to pv_site_net
     intervals.

  3. Inverter AC rating: site must have inverter_ac_rating_kw >= MIN_INVERTER_AC_RATING_KW
     in metadata, or have a null inverter rating (null is treated as unknown, not excluded).
     Sites with inverter_ac_rating_kw explicitly below the threshold are excluded.

  4. Battery screen: sites showing battery evidence (metadata flag OR a
     battery-labelled circuit_label present in the raw data -- metadata is
     known to be incomplete, see edp_data_spec.md item 11) are excluded
     UNLESS they pass the screening gates in docs/edp_data_spec.md, Battery sites,
     applied by B0_Battery_Site_Screen.py: exactly one distinct
     battery-labelled channel, bidirectional energy flow, sufficient annual
     throughput, and a plausible charge/discharge efficiency ratio. Sites
     with no battery evidence at all are unaffected by this filter, same as
     before. Sites that pass carry forward battery_channel_label and
     battery_sign_convention so C_Check_Consistency.py and
     E_Flag_All_Sites.py know to reconstruct gross_load for them.

Filters are applied independently so the printout shows how many sites each
filter would remove on its own, then the combined result of all filters applied
together. Sites must also have a metadata entry — those without are silently
dropped at the inner join step.

Inputs:  data/processed/edp/YYYY_site_coverage.csv
         data/raw/edp/edp_sites_metadata_2020_2025.csv
Outputs: data/processed/edp/YYYY_selected_sites.csv
"""

import pandas as pd
from pathlib import Path
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "edp"
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "edp"

# ── Year parameter — change this to rerun for a different year ─────────────────
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022

# ── Derived file paths ─────────────────────────────────────────────────────────
coverage_file = PROCESSED_DIR / f"{YEAR}_site_coverage.csv"
metadata_file = RAW_DIR / "edp_sites_metadata_2020_2025.csv"
battery_screen_file = PROCESSED_DIR / f"{YEAR}_battery_site_screen.csv"
output_file   = PROCESSED_DIR / f"{YEAR}_selected_sites.csv"

# ── Parameters ─────────────────────────────────────────────────────────────────
MIN_AC_LOAD_COVERAGE_PCT  = 85.0   # minimum ac_load_net coverage %
MIN_PV_COVERAGE_PCT       = 85.0   # minimum pv_site_net coverage %
MAX_COVERAGE_PCT          = 100.5  # maximum coverage % -- allows tiny rounding
                                    # slack but catches genuinely impossible
                                    # values (e.g. two households sharing a
                                    # site ID); see docstring item 1
MIN_INVERTER_AC_RATING_KW = 1.0    # minimum inverter AC rating (kW)

# ── Load coverage summary ──────────────────────────────────────────────────────
coverage_df = pd.read_csv(coverage_file)

# ── Load and deduplicate metadata ──────────────────────────────────────────────
meta_df = pd.read_csv(metadata_file)

# Strip whitespace from string columns used in filtering
meta_df['islandable'] = meta_df['islandable'].str.strip()

# Deduplicate: keep row with largest inverter_ac_rating_kw per site
meta_df = (
    meta_df
    .sort_values('inverter_ac_rating_kw', ascending=False)
    .drop_duplicates(subset='edp_site_id', keep='first')
)

# ── Battery flag ───────────────────────────────────────────────────────────────
meta_df['has_battery_flag'] = (
    (meta_df['islandable'] == 'Yes') |
    (meta_df['has_battery'] == 't') |
    (meta_df['battery_size_make_model'].notna())
)

# ── Merge coverage with metadata (inner join — drops sites missing from either)
merged_df = coverage_df.merge(meta_df, on='edp_site_id', how='inner')

# ── Battery screen (B0_Battery_Site_Screen.py output) ───────────────────────────
# Left join: most sites have no row here at all, meaning no battery evidence
# in the raw data, and are unaffected by the battery filter below. See
# docs/edp_data_spec.md, Battery sites, for the screening rules.
battery_screen = pd.read_csv(battery_screen_file, usecols=[
    'edp_site_id', 'category', 'usable', 'battery_channel_label', 'battery_sign_convention'
])
merged_df = merged_df.merge(
    battery_screen, on='edp_site_id', how='left', suffixes=('', '_screen')
)
merged_df['usable'] = merged_df['usable'].fillna(False).astype(bool)
# A site has battery evidence if metadata flags it OR it showed up in the
# screen at all (i.e. had >=1 battery-labelled channel in the raw data,
# regardless of whether it passed the screen). merge_indicator via 'category'
# being non-null distinguishes a real screen match from the left-join fill.
merged_df['battery_evidence'] = merged_df['has_battery_flag'] | merged_df['category'].notna()

n_coverage  = len(coverage_df)
n_metadata  = meta_df['edp_site_id'].nunique()
n_merged    = len(merged_df)
n_no_meta   = n_coverage - n_merged

print(f"Year: {YEAR}")
print(f"\nSite counts:")
print(f"  Sites in coverage data:             {n_coverage}")
print(f"  Sites in metadata:                  {n_metadata}")
print(f"  Sites with no metadata (dropped):   {n_no_meta}")
print(f"  Sites after inner join:             {n_merged}")

# ── Individual filter impact (each applied independently to full merged set) ───
f1_fail = merged_df['ac_load_net_coverage_pct'] < MIN_AC_LOAD_COVERAGE_PCT
f2_fail = merged_df['pv_site_net_coverage_pct'] < MIN_PV_COVERAGE_PCT
f3_fail = (merged_df['inverter_ac_rating_kw'] < MIN_INVERTER_AC_RATING_KW) & \
           (merged_df['inverter_ac_rating_kw'].notna())  # nulls treated as unknown, not excluded
f4_fail = merged_df['battery_evidence'] & ~merged_df['usable']
f5_fail = (
    (merged_df['ac_load_net_coverage_pct'] > MAX_COVERAGE_PCT) |
    (merged_df['pv_site_net_coverage_pct'] > MAX_COVERAGE_PCT)
)

print(f"\nIndividual filter impact (each applied independently to {n_merged} joined sites):")
print(f"  ac_load_net coverage < {MIN_AC_LOAD_COVERAGE_PCT}%:         "
      f"{f1_fail.sum():>4} would be removed ({f1_fail.sum()/n_merged:.1%})")
print(f"  pv_site_net coverage < {MIN_PV_COVERAGE_PCT}%:         "
      f"{f2_fail.sum():>4} would be removed ({f2_fail.sum()/n_merged:.1%})")
print(f"  inverter_ac_rating_kw < {MIN_INVERTER_AC_RATING_KW} kW:      "
      f"{f3_fail.sum():>4} would be removed ({f3_fail.sum()/n_merged:.1%})")
print(f"  battery (no usable reconstruction):"
      f"{f4_fail.sum():>4} would be removed ({f4_fail.sum()/n_merged:.1%})")
n_recovered = int((merged_df['battery_evidence'] & merged_df['usable']).sum())
print(f"  (of {int(merged_df['battery_evidence'].sum())} sites with battery evidence, "
      f"{n_recovered} pass the screen and are retained with a reconstructed gross_load)")
print(f"  coverage > {MAX_COVERAGE_PCT}% (implausible -- see item 1):"
      f"{f5_fail.sum():>4} would be removed ({f5_fail.sum()/n_merged:.1%})")
if f5_fail.sum() > 0:
    print(f"    Sites: {sorted(merged_df.loc[f5_fail, 'edp_site_id'].tolist())}")

# ── Combined filter ────────────────────────────────────────────────────────────
filtered_df = merged_df[
    ~f1_fail & ~f2_fail & ~f3_fail & ~f4_fail & ~f5_fail
].copy()

n_removed = n_merged - len(filtered_df)
print(f"\nAll filters combined:")
print(f"  Removed: {n_removed} ({n_removed/n_merged:.1%} of joined sites)")
print(f"  Final selected sites: {len(filtered_df)}")

# ── Summary by state ──────────────────────────────────────────────────────────
state_summary = (
    filtered_df.groupby('state')['edp_site_id']
    .count()
    .sort_values(ascending=False)
    .rename('site_count')
)
print("\nSites by state:")
print(state_summary.to_string())

# ── Save selected sites ────────────────────────────────────────────────────────
output_cols = [
    'edp_site_id', 'state', 'site_timezone',
    'inverter_manufacturer', 'inverter_model', 'inverter_ac_rating_kw',
    'subarray_dc_rating_kw',
    'ac_load_net_coverage_pct', 'ac_load_net_max_gap_min', 'ac_load_net_phases',
    'pv_site_net_coverage_pct', 'pv_site_net_max_gap_min', 'pv_site_net_phases',
    'has_battery_flag', 'islandable', 'has_battery', 'battery_size_make_model',
    'battery_channel_label', 'battery_sign_convention'
]

filtered_df[output_cols].to_csv(output_file, index=False)
print(f"\nSaved {len(filtered_df)} selected sites to: {output_file}")