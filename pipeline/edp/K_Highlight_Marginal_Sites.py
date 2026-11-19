"""
K_Highlight_Marginal_Sites.py

Reviews sites near the boundary of each consistency check threshold.
For each check prints:
  - The 5 overall-passing sites closest to the threshold (risk of false negative)
  - The 5 sites failing ONLY this check closest to the threshold (potential false positive)

Run after C_Check_Consistency.py. Does not require clean files.

Usage: python3 K_Highlight_Marginal_Sites.py [YEAR]

Input:  data/processed/edp/YYYY_consistency_checks.csv
        data/processed/edp/YYYY_selected_sites.csv
Output: printed summary to terminal
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ── Parameters ─────────────────────────────────────────────────────────────────
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
N    = 5   # sites to show per group

SCRIPT_DIR    = Path(__file__).parent
ROOT_DIR      = SCRIPT_DIR.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "edp"

cc_file       = PROCESSED_DIR / f"{YEAR}_consistency_checks.csv"
selected_file = PROCESSED_DIR / f"{YEAR}_selected_sites.csv"

# ── Load and merge ─────────────────────────────────────────────────────────────
cc  = pd.read_csv(cc_file)
sel = pd.read_csv(selected_file, usecols=['edp_site_id', 'state'])
cc  = cc.merge(sel, on='edp_site_id', how='left')

NEM = ['NSW', 'QLD', 'SA', 'VIC']
nem = cc[cc['state'].isin(NEM)].copy()

# All check pass columns
all_checks = ['corr_pass', 'export_pass', 'ac_flatline_pass', 'pv_flatline_pass',
              'generation_pass', 'pv_negative_pass', 'gross_pv_pass']

def marginal(df, col, threshold, fail_if_above, this_check_pass_col, label):
    """
    Print N passing sites closest to threshold, and N sites failing
    only this check closest to threshold.
    """
    print(f"\n── {label}")
    print(f"   threshold: {'>' if fail_if_above else '<'} {threshold}")

    display_cols = ['edp_site_id', 'state', col]

    # Distance from threshold (always positive)
    df = df.copy()
    df['_dist'] = (df[col] - threshold).abs()

    # Group 1: overall passing sites closest to threshold
    passing = df[df['overall_pass']].nsmallest(N, '_dist')[display_cols + ['_dist']]
    passing['_dist'] = passing['_dist'].round(4)
    print(f"\n  Passing sites closest to threshold (top {N}):")
    if len(passing) > 0:
        print(passing.to_string(index=False))
    else:
        print("  none")

    # Group 2: sites failing ONLY this check, closest to threshold
    other_checks = [c for c in all_checks if c != this_check_pass_col]
    only_this = df[
        ~df[this_check_pass_col] &
        df[other_checks].all(axis=1) &
        ~df['overall_pass'] &
        ~df.get('manual_exclusion', pd.Series(False, index=df.index))
    ].nsmallest(N, '_dist')[display_cols + ['_dist']]
    only_this['_dist'] = only_this['_dist'].round(4)
    print(f"\n  Failing this check only, closest to threshold (top {N}):")
    if len(only_this) > 0:
        print(only_this.to_string(index=False))
    else:
        print("  none")

# ── Header ─────────────────────────────────────────────────────────────────────
print(f"{'='*65}")
print(f"Marginal site review — {YEAR}  ({len(nem)} NEM sites)")
print(f"{'='*65}")

# ── Check 1 ───────────────────────────────────────────────────────────────────
marginal(nem, 'corr_coefficient', threshold=-0.2, fail_if_above=True,
         this_check_pass_col='corr_pass',
         label='Check 1 — PV/load correlation (fail if > -0.2)')

# ── Check 2a ──────────────────────────────────────────────────────────────────
marginal(nem, 'export_excess_intervals', threshold=10, fail_if_above=True,
         this_check_pass_col='export_pass',
         label='Check 2a — export excess intervals (fail if >= 10)')

# ── Check 2b ──────────────────────────────────────────────────────────────────
marginal(nem, 'gross_neg_fraction', threshold=0.10, fail_if_above=True,
         this_check_pass_col='export_pass',
         label='Check 2b — gross load < 0 fraction (fail if > 10%)')

# ── Check 3 ───────────────────────────────────────────────────────────────────
marginal(nem, 'ac_flatline_fraction', threshold=0.05, fail_if_above=True,
         this_check_pass_col='ac_flatline_pass',
         label='Check 3 — ac_load_net flatline fraction (fail if > 5%)')

# ── Check 4 ───────────────────────────────────────────────────────────────────
marginal(nem, 'pv_flatline_fraction', threshold=0.05, fail_if_above=True,
         this_check_pass_col='pv_flatline_pass',
         label='Check 4 — pv_site_net flatline fraction (fail if > 5%)')

# ── Check 5a ──────────────────────────────────────────────────────────────────
marginal(nem, 'annual_generation_kwh', threshold=500, fail_if_above=False,
         this_check_pass_col='generation_pass',
         label='Check 5a — annual generation (fail if < 500 kWh)')

# ── Check 5b ──────────────────────────────────────────────────────────────────
marginal(nem, 'avg_daily_gross_load_wh', threshold=5000, fail_if_above=False,
         this_check_pass_col='generation_pass',
         label='Check 5b — avg daily gross load (fail if < 5000 Wh/day)')

# ── Check 6 ───────────────────────────────────────────────────────────────────
marginal(nem, 'pv_daytime_negative_fraction', threshold=0.01, fail_if_above=True,
         this_check_pass_col='pv_negative_pass',
         label='Check 6 — daytime negative PV fraction (fail if > 1%)')

# ── Check 7 ───────────────────────────────────────────────────────────────────
marginal(nem, 'gross_pv_corr', threshold=0.475, fail_if_above=True,
         this_check_pass_col='gross_pv_pass',
         label='Check 7 — gross/pv correlation (fail if > 0.475)')

print(f"\n{'='*65}")
print(f"Done.")