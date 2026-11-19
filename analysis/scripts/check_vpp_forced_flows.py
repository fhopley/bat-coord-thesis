"""Quick diagnostic for vpp_forced_flows.parquet.

Usage (from project root):
    python3 analysis/scripts/check_vpp_forced_flows.py
    python3 analysis/scripts/check_vpp_forced_flows.py --path data/output/runs/example_vpp_tou_event/scenarios/NSW_2022/vpp_forced_flows.parquet
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def count_events(vpp_event_col: pd.Series) -> int:
    """Count distinct contiguous event windows in a boolean-like Series.

    A window is a maximal contiguous run of non-zero values.  Works on a
    single site's forced-flow column: any non-zero interval is 'in event'.
    """
    in_event = (vpp_event_col != 0).astype(int)
    # A new event starts where in_event goes from 0 to 1
    starts = ((in_event.diff() == 1) | ((in_event.iloc[:1] == 1))).sum()
    return int(starts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--path', type=Path,
        default=Path('data/output/runs/example_vpp_tou_event/scenarios'
                     '/NSW_2022/vpp_forced_flows.parquet'),
        help='Path to vpp_forced_flows.parquet',
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"ERROR: file not found: {args.path}")
        return 1

    df = pd.read_parquet(args.path)
    print(f"File: {args.path}")
    print(f"Shape: {df.shape[0]:,} event intervals  x  {df.shape[1]} sites")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print()

    # Split into forced import (positive) and forced export (negative)
    f_imp = df.clip(lower=0)   # kWh, positive
    f_exp = df.clip(upper=0).abs()  # kWh, positive (magnitude of export)

    # Per-site annual totals (kWh)
    imp_total = f_imp.sum()        # Series: site -> total forced import kWh
    exp_total = f_exp.sum()        # Series: site -> total forced export kWh
    combined  = imp_total + exp_total

    def stats(s: pd.Series, label: str) -> None:
        print(f"  {label}")
        print(f"    mean:   {s.mean():>10.2f} kWh")
        print(f"    median: {s.median():>10.2f} kWh")
        print(f"    std:    {s.std():>10.2f} kWh")
        print(f"    min:    {s.min():>10.2f} kWh")
        print(f"    max:    {s.max():>10.2f} kWh")
        # Show top 3 sites for context
        top = s.nlargest(3)
        print(f"    top 3 sites: " +
              ", ".join(f"{sid}={v:.1f}" for sid, v in top.items()))
        print()

    print("── Per-site annual totals (kWh) ──────────────────────────────")
    stats(imp_total, "Forced import  (operator charging battery)")
    stats(exp_total, "Forced export  (operator discharging to grid)")
    stats(combined,  "Combined       (import + export, cap-relevant)")

    # Event counting -- use the cohort-level forced flow series to find
    # contiguous event windows (same events for all sites since classification
    # is price-driven; use row-wise any() to find event intervals)
    cohort_active = (df != 0).any(axis=1)
    in_event = cohort_active.astype(int)
    event_starts = (in_event.diff() == 1).sum()
    if cohort_active.iloc[0]:
        event_starts += 1  # file starts mid-event

    # Event duration stats: find each contiguous run
    groups = (in_event.diff() != 0).cumsum()
    event_groups = groups[cohort_active == 1]
    if len(event_groups):
        durations_intervals = event_groups.groupby(event_groups).size()
        durations_hours = durations_intervals * (5 / 60)
        print("── VPP event summary ─────────────────────────────────────────")
        print(f"  Total distinct events (cohort level): {event_starts}")
        print(f"  Total event intervals:                {cohort_active.sum():,}")
        print(f"  Event duration (hours):")
        print(f"    mean:   {durations_hours.mean():.2f}")
        print(f"    median: {durations_hours.median():.2f}")
        print(f"    min:    {durations_hours.min():.2f}")
        print(f"    max:    {durations_hours.max():.2f}")
        print()

    # Per-site event counts (sites may have different active-interval counts
    # if some sites had zero forced flow in some events)
    site_event_counts = df.apply(count_events)
    print("── Per-site event counts ──────────────────────────────────────")
    print(f"  mean:   {site_event_counts.mean():.1f}")
    print(f"  median: {site_event_counts.median():.1f}")
    print(f"  min:    {site_event_counts.min()}")
    print(f"  max:    {site_event_counts.max()}")
    print()

    # Interval-level snapshot: mean forced import and export by hour of day
    print("── Mean forced flow by hour of day (cohort average, kWh/interval) ──")
    by_hour = pd.DataFrame({
        'f_imp': f_imp.mean(axis=1),
        'f_exp': f_exp.mean(axis=1),
    })
    by_hour['hour'] = by_hour.index.hour
    hourly = by_hour.groupby('hour')[['f_imp', 'f_exp']].mean()
    active_hours = hourly[(hourly['f_imp'] > 1e-6) | (hourly['f_exp'] > 1e-6)]
    if len(active_hours):
        print(f"  {'Hour':>5}  {'Forced import':>14}  {'Forced export':>14}")
        for hour, row in active_hours.iterrows():
            print(f"  {hour:>5}  {row['f_imp']:>14.5f}  {row['f_exp']:>14.5f}")
    else:
        print("  (no active hours found)")
    print()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
