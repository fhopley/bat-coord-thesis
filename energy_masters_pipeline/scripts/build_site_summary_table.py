"""build_site_summary_table.py

Produces one row per site with an include/exclude decision, the valid
REAL_TIME analysis period(s), and the data-quality diagnostics behind that
decision. This is the master table: everything before this script has been
exploratory, this is meant to be the thing we actually act on.

TUNABLE CONSTANTS (change these and rerun, rather than editing logic below):
"""
import sys
import traceback
from pathlib import Path

import pandas as pd
import numpy as np

DEFAULT_PATH = Path('data/energy_masters_amber_data/amber_telemetry.parquet')
OUTPUT_PATH = Path('data/processed/energy_masters/site_summary_output.txt')

# A REAL_TIME segment shorter than this doesn't count as a usable analysis
# period -- too short to be meaningful, likely noise. Reconsider this if it
# doesn't match what you need for a meaningful VCR comparison window.
MIN_SEGMENT_DAYS = 7

# A gap between two REAL_TIME runs shorter than this gets bridged over
# (treated as noise, not a genuine tariff change) before the minimum-length
# filter above is applied. Chosen to match the same "a real change wouldn't
# revert within a week" reasoning applied symmetrically to gaps. Empirically,
# most observed gaps cluster at almost the same timestamp across many
# unrelated sites (a shared data pipeline event, not independent tariff
# switches), and 80% of all gaps are <= 7 days.
MIN_GAP_TO_SPLIT_DAYS = 7

# Reconstruction-quality gate, computed within valid periods only.
MAX_PCT_NEGATIVE = 5.0
MAX_ABS_CORR_WITH_BATTERY = 0.3


class Tee:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def section(title):
    bar = '=' * 78
    print(f"\n{bar}\n{title}\n{bar}")


def safe(fn, default=None):
    try:
        return fn()
    except Exception:
        print(">>> FAILED:")
        traceback.print_exc()
        return default


def battery_family_for(df, site):
    labels = set(df.loc[df['em_id'] == site, 'appliance_label'].unique())
    if any(l.startswith('battery_storage') for l in labels):
        return 'battery_storage'
    if any(l.startswith('battery_dc') for l in labels):
        return 'battery_dc'
    return None


def raw_runs(timestamps, is_rt):
    """Return list of (start_idx, end_idx) for every contiguous REAL_TIME
    run, regardless of length -- no filtering yet."""
    runs = []
    start_idx = None
    for i, v in enumerate(is_rt):
        if v and start_idx is None:
            start_idx = i
        elif not v and start_idx is not None:
            runs.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        runs.append((start_idx, len(is_rt) - 1))
    return runs


def bridge_runs(runs, timestamps, max_gap_days):
    """Merge adjacent runs where the gap between them is <= max_gap_days.
    Operates on raw (unfiltered) runs so short segments separated only by
    noise get merged before any minimum-length filter is applied."""
    if not runs:
        return []
    merged = [list(runs[0])]
    for start_idx, end_idx in runs[1:]:
        prev_end_ts = timestamps.iloc[merged[-1][1]]
        this_start_ts = timestamps.iloc[start_idx]
        gap_days = (this_start_ts - prev_end_ts).total_seconds() / 86400
        if gap_days <= max_gap_days:
            merged[-1][1] = end_idx  # extend the previous run
        else:
            merged.append([start_idx, end_idx])
    return [tuple(m) for m in merged]


def find_valid_segments(timestamps, is_rt, min_days, max_gap_days):
    """Bridge short gaps first, then keep only merged segments spanning at
    least min_days. Returns list of (start, end, n_rows_in_span) where
    n_rows_in_span is the count of rows falling in [start, end] by index
    (includes any bridged-over non-REAL_TIME rows in that span)."""
    runs = raw_runs(timestamps, is_rt)
    merged = bridge_runs(runs, timestamps, max_gap_days)
    segments = []
    for start_idx, end_idx in merged:
        n_rows = end_idx - start_idx + 1
        span_days = (timestamps.iloc[end_idx] - timestamps.iloc[start_idx]).total_seconds() / 86400
        if span_days >= min_days:
            segments.append((timestamps.iloc[start_idx], timestamps.iloc[end_idx], n_rows))
    return segments


def build_family_wide(df, site):
    sub = df[df['em_id'] == site]
    wide = sub.pivot_table(
        index='utc_reading_datetime', columns='appliance_label',
        values='energy', aggfunc='first',
    )
    families = {}
    for col in wide.columns:
        family = col.split('__')[0]
        families.setdefault(family, []).append(col)
    fam_wide = pd.DataFrame({
        fam: wide[cols].sum(axis=1, skipna=True) for fam, cols in families.items()
    })
    return fam_wide, sub


def evaluate_site(df, site):
    row = {'site_id': site}
    battery_channel = battery_family_for(df, site)
    row['battery_channel'] = battery_channel

    if battery_channel is None:
        row.update({'include': False, 'exclude_reason': 'no battery channel present',
                    'valid_periods': [], 'total_valid_days': 0.0, 'n_valid_segments': 0,
                    'pct_negative_in_valid_period': None,
                    'corr_with_battery_in_valid_period': None,
                    'pct_bridged_noise_in_period': None,
                    'pct_rate_type_disagreement': None, 'notes': ''})
        return row

    ref = df[(df['em_id'] == site) & (df['appliance_label'] == 'ac_load_net')]
    ref = ref.sort_values('utc_reading_datetime')
    is_rt = (ref['rate_type'] == 'REAL_TIME').to_numpy()
    timestamps = ref['utc_reading_datetime']

    segments = find_valid_segments(timestamps, is_rt, MIN_SEGMENT_DAYS, MIN_GAP_TO_SPLIT_DAYS)

    if not segments:
        row.update({'include': False,
                    'exclude_reason': f'no REAL_TIME segment >= {MIN_SEGMENT_DAYS} days '
                                       '(possibly a rate_type labelling issue -- worth '
                                       'rechecking if final site count is too low)',
                    'valid_periods': [], 'total_valid_days': 0.0, 'n_valid_segments': 0,
                    'pct_negative_in_valid_period': None,
                    'corr_with_battery_in_valid_period': None,
                    'pct_bridged_noise_in_period': None,
                    'pct_rate_type_disagreement': None, 'notes': ''})
        return row

    valid_periods = [(s, e) for s, e, n in segments]
    total_days = sum((e - s).total_seconds() / 86400 for s, e, _ in segments)
    row['valid_periods'] = valid_periods
    row['total_valid_days'] = round(total_days, 1)
    row['n_valid_segments'] = len(segments)

    # Restrict to timestamps inside any bridged period's calendar span --
    # this includes the handful of rows that were bridged over because
    # rate_type briefly wasn't REAL_TIME (the noise we're treating as a
    # pipeline artifact, not a genuine tariff change).
    valid_mask = pd.Series(False, index=timestamps.index)
    for s, e, _ in segments:
        valid_mask |= (timestamps >= s) & (timestamps <= e)
    valid_timestamps = set(timestamps[valid_mask])

    # Of those, which rows are genuinely tagged REAL_TIME? Analysis below
    # uses only these -- the bridged-over rows are reported as a diagnostic
    # (pct_bridged_noise_in_period) but not silently treated as real-time
    # data, since what to do with them is a billing-pipeline decision for
    # later, not something to resolve in this site-selection pass.
    is_rt_series = pd.Series(is_rt, index=timestamps.index)
    real_time_mask = valid_mask & is_rt_series
    real_time_timestamps = set(timestamps[real_time_mask])

    n_window_rows = int(valid_mask.sum())
    n_real_time_rows = int(real_time_mask.sum())
    pct_bridged_noise = (round(100 * (1 - n_real_time_rows / n_window_rows), 2)
                         if n_window_rows else None)
    row['pct_bridged_noise_in_period'] = pct_bridged_noise

    fam_wide, sub = build_family_wide(df, site)
    fam_wide_valid = fam_wide[fam_wide.index.isin(real_time_timestamps)]

    notes = []

    # Reconstruction quality within valid periods.
    pct_negative = None
    corr_batt = None
    if ('ac_load_net' in fam_wide_valid.columns and 'pv_dc' in fam_wide_valid.columns
            and battery_channel in fam_wide_valid.columns):
        net = fam_wide_valid['ac_load_net']
        pv = fam_wide_valid['pv_dc']
        batt = fam_wide_valid[battery_channel]
        recon = (net + pv - batt).dropna()
        if len(recon):
            pct_negative = round((recon < 0).mean() * 100, 2)
            corr_batt = round(recon.corr(batt.reindex(recon.index)), 3)
    row['pct_negative_in_valid_period'] = pct_negative
    row['corr_with_battery_in_valid_period'] = corr_batt

    # Corrected rate_type disagreement check, restricted to timestamps
    # where ac_load_net is genuinely tagged REAL_TIME (not just inside the
    # bridged calendar window), both sides non-null required before
    # comparing (fixes the earlier bug).
    ref_rate = ref.set_index('utc_reading_datetime')['rate_type']
    ref_rate = ref_rate[ref_rate.index.isin(real_time_timestamps)]
    sub_valid = sub[sub['utc_reading_datetime'].isin(real_time_timestamps) &
                    (sub['appliance_label'] != 'ac_load_net')]
    merged = sub_valid.merge(ref_rate.rename('ref_rate_type'),
                              left_on='utc_reading_datetime', right_index=True, how='left')
    both_present = merged['rate_type'].notna() & merged['ref_rate_type'].notna()
    comparable = merged[both_present]
    if len(comparable):
        disagree = comparable[comparable['rate_type'] != comparable['ref_rate_type']]
        pct_disagree = round(100 * len(disagree) / len(comparable), 3)
    else:
        pct_disagree = None
    row['pct_rate_type_disagreement'] = pct_disagree
    if pct_disagree and pct_disagree > 0:
        notes.append(f'rate_type disagrees with ac_load_net on {pct_disagree}% of '
                      'comparable rows -- check for controlled-load contamination')

    # Decision.
    flags = []
    if pct_negative is not None and pct_negative > MAX_PCT_NEGATIVE:
        flags.append(f'pct_negative {pct_negative}% > {MAX_PCT_NEGATIVE}%')
    if corr_batt is not None and abs(corr_batt) > MAX_ABS_CORR_WITH_BATTERY:
        flags.append(f'|corr_with_battery| {abs(corr_batt)} > {MAX_ABS_CORR_WITH_BATTERY}')

    if flags:
        row['include'] = True
        row['exclude_reason'] = None
        notes.append('FLAGGED FOR REVIEW: ' + '; '.join(flags))
    else:
        row['include'] = True
        row['exclude_reason'] = None

    row['notes'] = ' | '.join(notes)
    return row


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUTPUT_PATH)

    section("LOAD")
    print(f"Path: {path.resolve()}")
    df = pd.read_parquet(path)
    df['utc_reading_datetime'] = pd.to_datetime(df['utc_reading_datetime'])
    print(f"Shape: {df.shape}")
    print(f"MIN_SEGMENT_DAYS={MIN_SEGMENT_DAYS}  MIN_GAP_TO_SPLIT_DAYS={MIN_GAP_TO_SPLIT_DAYS}  "
          f"MAX_PCT_NEGATIVE={MAX_PCT_NEGATIVE}  "
          f"MAX_ABS_CORR_WITH_BATTERY={MAX_ABS_CORR_WITH_BATTERY}")

    sites = sorted(df['em_id'].unique())

    section("PER-SITE EVALUATION")
    rows = []
    for site in sites:
        r = safe(lambda: evaluate_site(df, site))
        if r:
            rows.append(r)

    result = pd.DataFrame(rows)
    display_cols = ['site_id', 'battery_channel', 'include', 'exclude_reason',
                     'n_valid_segments', 'total_valid_days', 'pct_bridged_noise_in_period',
                     'pct_negative_in_valid_period', 'corr_with_battery_in_valid_period',
                     'pct_rate_type_disagreement', 'notes']
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 60)
    pd.set_option('display.width', 220)
    print(result[display_cols].to_string(index=False))

    # valid_periods has to go to CSV separately since it's a list column.
    result.to_csv('data/processed/energy_masters/site_summary_results.csv', index=False)
    print("\n(full table with valid_periods written to data/processed/energy_masters/site_summary_results.csv)")

    section("SHORTLIST: INCLUDED SITES, NO REVIEW FLAG, SORTED BY TOTAL VALID DAYS")
    def _shortlist():
        clean = result[result['include'] &
                        ~result['notes'].str.contains('FLAGGED', na=False)]
        clean = clean.sort_values('total_valid_days', ascending=False)
        print(clean[['site_id', 'battery_channel', 'n_valid_segments',
                      'total_valid_days']].to_string(index=False))
    safe(_shortlist)

    section("EXCLUDED SITES")
    def _excluded():
        excl = result[~result['include']]
        print(excl[['site_id', 'exclude_reason']].to_string(index=False))
    safe(_excluded)

    section("FLAGGED FOR REVIEW")
    def _flagged():
        flagged = result[result['notes'].str.contains('FLAGGED', na=False)]
        print(flagged[['site_id', 'battery_channel', 'total_valid_days', 'notes']]
              .to_string(index=False))
    safe(_flagged)

    section("DONE")
    print(f"Full report also written to: {OUTPUT_PATH.resolve()}")


if __name__ == '__main__':
    main()