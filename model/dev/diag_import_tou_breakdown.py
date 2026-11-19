"""Diagnostic: compare SC vs EOV natural imports by TOU time slot.

Investigates why EOV retail import (usage) is so much higher than SC despite
minimal forced imports.  Hypothesis: the TOU gap LP is shifting natural imports
into the expensive peak window.

Usage (from project root):
    python3 -m model.dev.diag_import_tou_breakdown SITE_ID YEAR
    python3 -m model.dev.diag_import_tou_breakdown SITE_ID YEAR \\
        --state NSW \\
        --retail-tariff agl_battery_rewards_tou_2025 \\
        --op-tariff agl_operator_wholesale_2025 \\
        --export-limit-kw 10

Prints a table of imports (kWh and $) by TOU slot for SC and EOV, then
breaks down the peak-shift within event vs non-event intervals.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model.core.paths import default_paths
from model.dev._dev_common import DEV_BATTERY
from model.dev.viz_vpp_tou_event import run_tou_analysis
from model.dev.viz_vpp_sc_event import load_inputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tou_slot(index: pd.DatetimeIndex, retail_tariff) -> pd.Series:
    """Return a string Series 'off_peak' / 'peak' for every interval.

    Reads slot boundaries directly from the tariff's usage_tariffs list so
    the diagnostic stays consistent with however the model prices things.
    """
    hour = index.hour + index.minute / 60.0  # fractional hour

    # Build slot label from tariff usage_tariffs
    # Each entry is [start_hour, end_hour, c_per_kwh]
    # We label by rate: cheapest = 'off_peak', rest = 'peak'
    # TariffPeriod fields: start_hour, end_hour, c_per_kwh
    periods = retail_tariff.usage_tariffs
    rate_values = sorted(set(p.c_per_kwh for p in periods))
    min_rate = rate_values[0]

    labels = pd.array(['peak'] * len(index), dtype=object)
    for p in periods:
        mask = (hour >= p.start_hour) & (hour < p.end_hour)
        if p.c_per_kwh == min_rate:
            labels[mask] = 'off_peak'

    return pd.Series(labels, index=index, name='slot')


def _rate_for_slot(slot: str, retail_tariff) -> float:
    """Return c/kWh for a given slot label."""
    periods = retail_tariff.usage_tariffs
    rate_values = sorted(set(p.c_per_kwh for p in periods))
    if slot == 'off_peak':
        return rate_values[0]
    else:
        return rate_values[-1]


def _slot_summary(
    import_kwh: pd.Series,
    slot: pd.Series,
    retail_tariff,
    label: str,
) -> pd.DataFrame:
    """Summarise import kWh and $ by TOU slot."""
    df = pd.DataFrame({'import_kwh': import_kwh, 'slot': slot})
    grp = df.groupby('slot')['import_kwh'].sum().rename(label)
    rows = []
    for s in ['off_peak', 'peak']:
        kwh = grp.get(s, 0.0)
        rate = _rate_for_slot(s, retail_tariff)
        rows.append({
            'slot': s,
            'rate_c_per_kwh': rate,
            f'{label}_kwh': kwh,
            f'{label}_dollars': kwh * rate / 100.0,
        })
    return pd.DataFrame(rows).set_index('slot')


def _merge_summaries(sc_df: pd.DataFrame, eov_df: pd.DataFrame) -> pd.DataFrame:
    # Drop duplicate rate column from right side before joining
    eov_df = eov_df.drop(columns=['rate_c_per_kwh'], errors='ignore')
    df = sc_df.join(eov_df)
    for col in ['kwh', 'dollars']:
        df[f'delta_{col}'] = df[f'eov_{col}'] - df[f'sc_{col}']
    return df


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def print_import_tou_breakdown(analysis) -> None:
    """Print full TOU import breakdown comparing SC and EOV traces."""

    rt = analysis.retail_tariff
    idx = analysis.sc_result.import_kwh.index
    slot = _tou_slot(idx, rt)

    # --- SC imports (all natural by definition) ---
    sc_import = analysis.sc_result.import_kwh

    # --- EOV imports: split natural vs forced ---
    eov_import   = analysis.eov_result.import_kwh
    forced_import = analysis.eov_vt.f_imp_kwh   # VPP forced import
    natural_import = eov_import - forced_import  # should be >= 0

    # --- Event mask ---
    event_mask = analysis.vpp_event.astype(bool)

    # -----------------------------------------------------------------------
    # Table 1: Overall import by TOU slot
    # -----------------------------------------------------------------------
    sc_sum  = _slot_summary(sc_import,     slot, rt, 'sc')
    eov_sum = _slot_summary(eov_import,    slot, rt, 'eov')
    merged  = _merge_summaries(sc_sum, eov_sum)

    print("\n" + "=" * 72)
    print("IMPORT BY TOU SLOT  (SC vs EOV, all intervals)")
    print("=" * 72)
    print(f"{'Slot':<12} {'Rate':>8}  "
          f"{'SC kWh':>10} {'SC $':>8}  "
          f"{'EOV kWh':>10} {'EOV $':>8}  "
          f"{'Delta kWh':>10} {'Delta $':>8}")
    print("-" * 72)
    for s, row in merged.iterrows():
        print(f"{s:<12} {row['rate_c_per_kwh']:>7.3f}c  "
              f"{row['sc_kwh']:>10.1f} {row['sc_dollars']:>8.2f}  "
              f"{row['eov_kwh']:>10.1f} {row['eov_dollars']:>8.2f}  "
              f"{row['delta_kwh']:>+10.1f} {row['delta_dollars']:>+8.2f}")
    print("-" * 72)
    # Totals row
    print(f"{'TOTAL':<12} {'':>8}  "
          f"{merged['sc_kwh'].sum():>10.1f} {merged['sc_dollars'].sum():>8.2f}  "
          f"{merged['eov_kwh'].sum():>10.1f} {merged['eov_dollars'].sum():>8.2f}  "
          f"{merged['delta_kwh'].sum():>+10.1f} {merged['delta_dollars'].sum():>+8.2f}")

    # -----------------------------------------------------------------------
    # Table 2: EOV natural vs forced import by slot
    # -----------------------------------------------------------------------
    nat_sum    = _slot_summary(natural_import, slot, rt, 'nat')
    forced_sum = _slot_summary(forced_import,  slot, rt, 'frc')

    nat_df = nat_sum[['nat_kwh', 'nat_dollars']]
    frc_df = forced_sum[['frc_kwh', 'frc_dollars']]
    eov_detail = nat_df.join(frc_df)
    eov_detail['total_kwh']     = eov_detail['nat_kwh']     + eov_detail['frc_kwh']
    eov_detail['total_dollars'] = eov_detail['nat_dollars'] + eov_detail['frc_dollars']

    print("\n" + "=" * 72)
    print("EOV IMPORT DETAIL: natural vs forced, by TOU slot")
    print("=" * 72)
    print(f"{'Slot':<12}  "
          f"{'Nat kWh':>10} {'Nat $':>8}  "
          f"{'Forced kWh':>10} {'Forced $':>8}  "
          f"{'Total kWh':>10} {'Total $':>8}")
    print("-" * 72)
    for s, row in eov_detail.iterrows():
        print(f"{s:<12}  "
              f"{row['nat_kwh']:>10.1f} {row['nat_dollars']:>8.2f}  "
              f"{row['frc_kwh']:>10.1f} {row['frc_dollars']:>8.2f}  "
              f"{row['total_kwh']:>10.1f} {row['total_dollars']:>8.2f}")
    print("-" * 72)
    print(f"{'TOTAL':<12}  "
          f"{eov_detail['nat_kwh'].sum():>10.1f} {eov_detail['nat_dollars'].sum():>8.2f}  "
          f"{eov_detail['frc_kwh'].sum():>10.1f} {eov_detail['frc_dollars'].sum():>8.2f}  "
          f"{eov_detail['total_kwh'].sum():>10.1f} {eov_detail['total_dollars'].sum():>8.2f}")

    # -----------------------------------------------------------------------
    # Table 3: Natural import shift, split by event vs non-event intervals
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("NATURAL IMPORT SHIFT: event vs non-event intervals")
    print("=" * 72)

    for segment, mask, label in [
        ('non-event', ~event_mask, 'Gap-fill LP behaviour'),
        ('event',      event_mask, 'VPP event LP behaviour'),
    ]:
        sc_seg   = _slot_summary(sc_import[mask],      slot[mask], rt, 'sc')
        nat_seg  = _slot_summary(natural_import[mask], slot[mask], rt, 'eov')
        seg_mrg  = _merge_summaries(sc_seg, nat_seg)

        n_intervals = mask.sum()
        n_hours = n_intervals * analysis.interval_hours
        print(f"\n  {label} ({n_intervals:,} intervals, {n_hours:.1f} h)")
        print(f"  {'Slot':<12} {'Rate':>8}  "
              f"{'SC kWh':>10} {'SC $':>8}  "
              f"{'EOV nat kWh':>12} {'EOV nat $':>10}  "
              f"{'Delta kWh':>10} {'Delta $':>8}")
        print("  " + "-" * 68)
        for s, row in seg_mrg.iterrows():
            print(f"  {s:<12} {row['rate_c_per_kwh']:>7.3f}c  "
                  f"{row['sc_kwh']:>10.1f} {row['sc_dollars']:>8.2f}  "
                  f"{row['eov_kwh']:>12.1f} {row['eov_dollars']:>10.2f}  "
                  f"{row['delta_kwh']:>+10.1f} {row['delta_dollars']:>+8.2f}")
        print("  " + "-" * 68)
        print(f"  {'TOTAL':<12} {'':>8}  "
              f"{seg_mrg['sc_kwh'].sum():>10.1f} {seg_mrg['sc_dollars'].sum():>8.2f}  "
              f"{seg_mrg['eov_kwh'].sum():>12.1f} {seg_mrg['eov_dollars'].sum():>10.2f}  "
              f"{seg_mrg['delta_kwh'].sum():>+10.1f} {seg_mrg['delta_dollars'].sum():>+8.2f}")

    # -----------------------------------------------------------------------
    # Table 4: SOC impact -- battery state during peak hours
    # -----------------------------------------------------------------------
    # Proxy: compare average SOC during peak vs off-peak for SC and EOV
    soc_sc  = analysis.sc_result.battery_soc_kwh
    soc_eov = analysis.eov_result.battery_soc_kwh

    peak_mask     = (slot == 'peak').values
    offpeak_mask  = (slot == 'off_peak').values

    print("\n" + "=" * 72)
    print("BATTERY SOC: average by slot (proxy for reserve availability)")
    print("=" * 72)
    print(f"{'Slot':<12}  {'SC avg SOC (kWh)':>18}  {'EOV avg SOC (kWh)':>18}  {'Delta':>8}")
    print("-" * 64)
    for mask, label in [(offpeak_mask, 'off_peak'), (peak_mask, 'peak')]:
        sc_avg  = soc_sc[mask].mean()
        eov_avg = soc_eov[mask].mean()
        print(f"{label:<12}  {sc_avg:>18.3f}  {eov_avg:>18.3f}  {eov_avg - sc_avg:>+8.3f}")

    # -----------------------------------------------------------------------
    # Summary diagnosis
    # -----------------------------------------------------------------------
    total_delta_kwh     = merged['delta_kwh'].sum()
    total_delta_dollars = merged['delta_dollars'].sum()
    peak_delta_kwh      = merged.loc['peak', 'delta_kwh']    if 'peak'     in merged.index else 0.0
    offpeak_delta_kwh   = merged.loc['off_peak', 'delta_kwh'] if 'off_peak' in merged.index else 0.0

    print("\n" + "=" * 72)
    print("DIAGNOSIS SUMMARY")
    print("=" * 72)
    print(f"  Total import delta (EOV - SC):  {total_delta_kwh:>+.1f} kWh  /  ${total_delta_dollars:>+.2f}")
    print(f"  Peak-slot import delta:         {peak_delta_kwh:>+.1f} kWh")
    print(f"  Off-peak import delta:          {offpeak_delta_kwh:>+.1f} kWh")
    if abs(total_delta_kwh) > 1.0:
        pct_explained_by_shift = abs(peak_delta_kwh) / abs(total_delta_kwh) * 100
        print(f"\n  Peak-slot share of total delta: {pct_explained_by_shift:.0f}%")
        if peak_delta_kwh > 50 and offpeak_delta_kwh < -20:
            print("\n  FINDING: Classic peak-shift pattern -- EOV is importing more in peak")
            print("  and less in off-peak vs SC.  The gap-fill TOU LP is likely holding")
            print("  SOC for events, leaving the battery unable to shift peak imports.")
        elif peak_delta_kwh > 50:
            print("\n  FINDING: EOV imports substantially more in peak windows.")
            print("  Off-peak delta is small, so the issue is not a simple shift --")
            print("  EOV is generating *new* peak imports (battery held for events?).")
        else:
            print("\n  FINDING: Import increase spread across slots; shift hypothesis")
            print("  may be secondary.  Check the event-vs-non-event breakdown above.")
    print()

    # -----------------------------------------------------------------------
    # Table 5: Top 10 days where EOV natural import exceeds SC import most
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("TOP 10 DAYS: EOV natural import most above SC import")
    print("(Use viz_vpp_sc_event / viz_vpp_tou_event to drill into these days)")
    print("=" * 72)

    sc_daily  = sc_import.resample('D').sum()
    nat_daily = natural_import.resample('D').sum()
    delta_daily = (nat_daily - sc_daily).sort_values(ascending=False)

    # Also compute per-day SOC stats and event flag
    soc_eov_daily_mean = soc_eov.resample('D').mean()
    soc_sc_daily_mean  = soc_sc.resample('D').mean()
    event_daily = pd.Series(event_mask.astype(int), index=idx).resample('D').max().astype(bool)

    print(f"  {'Rank':<5} {'Date':<12} {'SC kWh':>8} {'EOV nat kWh':>12} "
          f"{'Delta kWh':>10} {'SC SOC':>8} {'EOV SOC':>8} {'Event?':>7}")
    print("  " + "-" * 68)
    for rank, (date, delta) in enumerate(delta_daily.head(10).items(), 1):
        sc_kwh   = sc_daily.get(date, 0.0)
        nat_kwh  = nat_daily.get(date, 0.0)
        sc_soc   = soc_sc_daily_mean.get(date, float('nan'))
        eov_soc  = soc_eov_daily_mean.get(date, float('nan'))
        has_evt  = event_daily.get(date, False)
        print(f"  {rank:<5} {str(date.date()):<12} {sc_kwh:>8.2f} {nat_kwh:>12.2f} "
              f"{delta:>+10.2f} {sc_soc:>8.3f} {eov_soc:>8.3f} {'YES' if has_evt else 'no':>7}")
    print()


def print_soc_trace_detail(analysis, date_str: str, start_hour: int, end_hour: int) -> None:
    """Print interval-by-interval flow trace and SOC verification for a time window.

    For each interval in [start_hour, end_hour) on date_str, prints the raw
    flows from both SC and EOV traces, then recomputes the expected SOC change
    from those flows using the self-consumption battery rules:

        soc_new = soc_prev + sol_to_bat * charge_eff
                           + grid_to_bat * charge_eff
                           - bat_to_home / discharge_eff
                           - bat_to_grid / discharge_eff

    A MISMATCH flag is printed when the recomputed SOC delta does not match
    the actual SOC delta in the trace (tolerance 1e-6 kWh).
    """
    import numpy as np

    bs   = analysis.battery_spec
    c_e  = bs.charge_efficiency
    d_e  = bs.discharge_efficiency
    ih   = analysis.interval_hours

    sc  = analysis.sc_result
    eov = analysis.eov_result
    vt  = analysis.eov_vt

    # Build window mask
    idx = sc.import_kwh.index
    date = pd.Timestamp(date_str)
    mask = (
        (idx.date == date.date()) &
        (idx.hour >= start_hour) &
        (idx.hour < end_hour)
    )
    win_idx = idx[mask]
    if len(win_idx) == 0:
        print(f"No intervals found for {date_str} {start_hour:02d}:00-{end_hour:02d}:00")
        return

    print()
    print("=" * 100)
    print(f"SOC TRACE DETAIL  site={analysis.site_id}  {date_str}  "
          f"{start_hour:02d}:00-{end_hour:02d}:00")
    print(f"Battery: {bs.capacity_kwh} kWh, {bs.power_kw} kW, "
          f"charge_eff={c_e}, discharge_eff={d_e}, "
          f"reserve={bs.backup_reserve_pct}% ({bs.capacity_kwh * bs.backup_reserve_pct / 100:.2f} kWh)")
    print("=" * 100)

    # Header
    hdr = (f"{'Time':<8}  {'Solar':>7} {'Load':>7}  "
           f"{'SC s2b':>7} {'SC b2h':>7} {'SC SOC':>7} {'SC dSOC':>7} {'SC exp dSOC':>12} {'SC ok?':>6}  "
           f"{'EOV s2b':>7} {'EOV b2h':>7} {'EOV g2b':>7} {'EOV b2g':>7} "
           f"{'EOV n_imp':>9} {'EOV f_imp':>9} {'EOV f_exp':>9} "
           f"{'EOV SOC':>7} {'EOV dSOC':>8} {'EOV exp dSOC':>13} {'EOV ok?':>7}")
    print(hdr)
    print("-" * 180)

    # SOC at the interval before the window (for delta computation)
    # Find position of first window interval in full index
    first_pos = idx.get_loc(win_idx[0])

    def _prev_soc(traces, pos):
        if pos == 0:
            return analysis.soc_init_kwh
        return traces.battery_soc_kwh.iloc[pos - 1]

    sc_soc_prev  = _prev_soc(sc,  first_pos)
    eov_soc_prev = _prev_soc(eov, first_pos)

    solar_s = analysis.trace['pv_site_net_kwh']
    load_s  = analysis.trace['gross_load_kwh']

    for ts in win_idx:
        pos = idx.get_loc(ts)

        sol  = solar_s.loc[ts]
        load = load_s.loc[ts]

        # --- SC flows ---
        sc_s2b = sc.sol_to_bat_kwh.loc[ts]
        sc_b2h = sc.bat_to_home_kwh.loc[ts]
        sc_g2b = sc.grid_to_bat_kwh.loc[ts]
        sc_b2g = sc.bat_to_grid_kwh.loc[ts]
        sc_soc = sc.battery_soc_kwh.loc[ts]
        sc_dsoc_actual = sc_soc - sc_soc_prev
        sc_dsoc_expect = (sc_s2b + sc_g2b) * c_e - (sc_b2h + sc_b2g) / d_e
        sc_ok = abs(sc_dsoc_actual - sc_dsoc_expect) < 1e-6

        # --- EOV flows ---
        eov_s2b = eov.sol_to_bat_kwh.loc[ts]
        eov_b2h = eov.bat_to_home_kwh.loc[ts]
        eov_g2b = eov.grid_to_bat_kwh.loc[ts]
        eov_b2g = eov.bat_to_grid_kwh.loc[ts]
        eov_n_imp = vt.n_imp_kwh.loc[ts]
        eov_f_imp = vt.f_imp_kwh.loc[ts]
        eov_f_exp = vt.f_exp_kwh.loc[ts]
        eov_soc = eov.battery_soc_kwh.loc[ts]
        eov_dsoc_actual = eov_soc - eov_soc_prev
        eov_dsoc_expect = (eov_s2b + eov_g2b) * c_e - (eov_b2h + eov_b2g) / d_e
        eov_ok = abs(eov_dsoc_actual - eov_dsoc_expect) < 1e-6

        sc_flag  = "OK" if sc_ok  else "MISMATCH"
        eov_flag = "OK" if eov_ok else "MISMATCH"

        time_str = ts.strftime("%H:%M")
        print(
            f"{time_str:<8}  {sol:>7.4f} {load:>7.4f}  "
            f"{sc_s2b:>7.4f} {sc_b2h:>7.4f} {sc_soc:>7.4f} {sc_dsoc_actual:>+7.4f} {sc_dsoc_expect:>+12.4f} {sc_flag:>6}  "
            f"{eov_s2b:>7.4f} {eov_b2h:>7.4f} {eov_g2b:>7.4f} {eov_b2g:>7.4f} "
            f"{eov_n_imp:>9.4f} {eov_f_imp:>9.4f} {eov_f_exp:>9.4f} "
            f"{eov_soc:>7.4f} {eov_dsoc_actual:>+8.4f} {eov_dsoc_expect:>+13.4f} {eov_flag:>7}"
        )

        sc_soc_prev  = sc_soc
        eov_soc_prev = eov_soc

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('site_id')
    parser.add_argument('year', type=int)
    parser.add_argument('--state', default='NSW')
    parser.add_argument('--network-tariff', default='ausgrid_ea025_network_2024')
    parser.add_argument('--retail-tariff', default='ea_battery_ease_vpp_2025')
    parser.add_argument('--op-tariff', default='ea_operator_wholesale_2025')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--project-root', type=Path, default=None)
    parser.add_argument('--soc-detail-date', default=None,
                        help='Date for SOC trace detail, e.g. 2022-06-29')
    parser.add_argument('--soc-detail-start-hour', type=int, default=17,
                        help='Start hour (inclusive) for SOC detail window (default 17)')
    parser.add_argument('--soc-detail-end-hour', type=int, default=18,
                        help='End hour (exclusive) for SOC detail window (default 18)')
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, network, retail, op_tariff = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, args.op_tariff, paths,
    )

    analysis = run_tou_analysis(
        args.site_id, args.year, trace, prices_mwh,
        network, retail, op_tariff, DEV_BATTERY,
        export_limit_kw=args.export_limit_kw,
    )

    print_import_tou_breakdown(analysis)

    if args.soc_detail_date:
        print_soc_trace_detail(analysis, args.soc_detail_date,
                               args.soc_detail_start_hour,
                               args.soc_detail_end_hour)
    return 0


if __name__ == '__main__':
    sys.exit(main())