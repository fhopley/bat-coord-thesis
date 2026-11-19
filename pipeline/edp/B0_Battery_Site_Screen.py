"""
B0_Battery_Site_Screen.py

New pipeline stage, inserted between A_Site_Summary.py and
B_Site_Selection.py. Implements the screening rules from
docs/edp_data_spec.md, Battery sites, using the aggregate battery-channel
statistics A_Site_Summary.py already computed -- no raw-data re-scan needed.

For every site with at least one of the three battery-labelled channels
present (battery_storage, load_battery, pv_site_net_battery), regardless of
what edp_sites_metadata_2020_2025.csv says (metadata is incomplete, so it is
not the gate), this applies:

  3.1 core channels present            -- both ac_load_net and pv_site_net
  3.2 exactly one battery-labelled channel -- 2+ is an automatic exclusion
  3.3 bidirectional                    -- >=5% of intervals on minority sign
  3.4 annual throughput >= 200,000 Wh
  3.5 efficiency ratio <= 3.0          -- also determines sign convention

Sites with zero battery-labelled channels never appear in the output at
all -- they are unaffected by this stage and pass through
B_Site_Selection.py exactly as before.

Input:  data/processed/edp/YYYY_site_coverage.csv (from A_Site_Summary.py,
        must include the battery aggregate columns)
        data/raw/edp/edp_sites_metadata_2020_2025.csv (diagnostic only)
Output: data/processed/edp/YYYY_battery_site_screen.csv
"""

import pandas as pd
from pathlib import Path
import sys

# ── Year parameter ─────────────────────────────────────────────────────────────
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "edp"
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "edp"

coverage_file = PROCESSED_DIR / f"{YEAR}_site_coverage.csv"
metadata_file = RAW_DIR / "edp_sites_metadata_2020_2025.csv"
output_file = PROCESSED_DIR / f"{YEAR}_battery_site_screen.csv"

# ── Thresholds (edp_data_spec.md, Battery sites) ────────────────────────────
MIN_MINORITY_SIGN_FRACTION = 0.05
MIN_ANNUAL_THROUGHPUT_WH = 200_000.0
PLAUSIBLE_RATIO_MAX = 3.0

CORE_LABELS = ['ac_load_net', 'pv_site_net']
BATTERY_LABELS = ['battery_storage', 'load_battery', 'pv_site_net_battery']

# ── Load coverage summary (must be the extended version from the updated
#    A_Site_Summary.py, with the battery aggregate columns) ────────────────────
coverage = pd.read_csv(coverage_file)
missing_cols = [
    f'{label}_{suffix}' for label in BATTERY_LABELS
    for suffix in ['sum_pos_wh', 'sum_neg_abs_wh', 'n_pos', 'n_neg', 'n_total']
    if f'{label}_{suffix}' not in coverage.columns
]
if missing_cols:
    raise RuntimeError(
        f"{coverage_file} is missing battery aggregate columns: {missing_cols}. "
        f"Re-run the updated A_Site_Summary.py for {YEAR} first."
    )

# ── Load metadata (diagnostic comparison only, not a gate) ─────────────────────
meta = pd.read_csv(metadata_file)
meta['islandable'] = meta['islandable'].str.strip()
meta = (meta.sort_values('inverter_ac_rating_kw', ascending=False)
            .drop_duplicates(subset='edp_site_id', keep='first'))
meta['has_battery_flag'] = (
    (meta['islandable'] == 'Yes') |
    (meta['has_battery'] == 't') |
    (meta['battery_size_make_model'].notna())
)
meta_flag = meta.set_index('edp_site_id')['has_battery_flag'].to_dict()

# ── Screen each site ─────────────────────────────────────────────────────────────
rows = []
for _, site_row in coverage.iterrows():
    site = site_row['edp_site_id']
    battery_labels_present = [
        label for label in BATTERY_LABELS if site_row.get(f'{label}_n_total', 0) > 0
    ]
    if not battery_labels_present:
        continue  # not a candidate at all, unaffected by this stage

    core_ok = all(site_row.get(f'{c}_coverage_pct', 0) > 0 for c in CORE_LABELS)
    n_batt = len(battery_labels_present)

    row = {
        'edp_site_id': site,
        'meta_has_battery_flag': meta_flag.get(site),
        'battery_labels_present': ','.join(battery_labels_present),
    }

    if not core_ok:
        row.update({'category': 'missing_core_unusable', 'usable': False})
        rows.append(row)
        continue

    if n_batt >= 2:
        row.update({'category': 'excluded_multi_channel', 'usable': False})
        rows.append(row)
        continue

    label = battery_labels_present[0]
    n_pos = site_row[f'{label}_n_pos']
    n_neg = site_row[f'{label}_n_neg']
    n_total = site_row[f'{label}_n_total']
    sum_pos = site_row[f'{label}_sum_pos_wh']
    sum_neg_abs = site_row[f'{label}_sum_neg_abs_wh']

    minority_frac = min(n_pos, n_neg) / n_total if n_total else 0.0
    bidirectional = minority_frac >= MIN_MINORITY_SIGN_FRACTION

    annual_throughput = sum_pos + sum_neg_abs
    throughput_ok = annual_throughput >= MIN_ANNUAL_THROUGHPUT_WH

    if sum_pos == 0 or sum_neg_abs == 0:
        sign_convention, ratio, ratio_ok = None, None, False
    elif sum_pos >= sum_neg_abs:
        sign_convention = 'subtract'   # positive = charge
        ratio = sum_pos / sum_neg_abs
        ratio_ok = ratio <= PLAUSIBLE_RATIO_MAX
    else:
        sign_convention = 'add'        # positive = discharge
        ratio = sum_neg_abs / sum_pos
        ratio_ok = ratio <= PLAUSIBLE_RATIO_MAX

    usable = bidirectional and throughput_ok and ratio_ok

    row.update({
        'battery_channel_label': label,
        'minority_frac_pct': round(minority_frac * 100, 2),
        'bidirectional': bidirectional,
        'annual_throughput_wh': round(annual_throughput, 0),
        'throughput_ok': throughput_ok,
        'battery_sign_convention': sign_convention,
        'efficiency_ratio': round(ratio, 3) if ratio is not None else None,
        'ratio_ok': ratio_ok,
        'category': 'usable' if usable else 'excluded_single_channel',
        'usable': usable,
    })
    rows.append(row)

screen_df = pd.DataFrame(rows)
screen_df.to_csv(output_file, index=False)

# ── Summary ──────────────────────────────────────────────────────────────────────
print(f"Year: {YEAR}")
print(f"Candidate sites (>=1 battery-labelled channel): {len(screen_df)}")
print()
print(screen_df['category'].value_counts().to_string())
print()
n_usable = int(screen_df['usable'].sum())
print(f"USABLE (passes all gates): {n_usable}")
if n_usable:
    usable_df = screen_df[screen_df['usable']]
    n_disagree = (~usable_df['meta_has_battery_flag'].fillna(False)).sum()
    print(f"  of which {n_disagree} were not flagged as battery sites in metadata "
          f"(recovered by the widened scope)")
print(f"\nSaved to: {output_file}")