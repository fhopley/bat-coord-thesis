"""Diagnostic: price interval distribution inside vs outside VPP event windows.

Investigates whether the min_export_price threshold computed from full-year
prices is a good approximation when restricted to event-window intervals only.

For a given site/year/tariff, prints:
  - How many intervals exceed various price thresholds in total vs inside events
  - The budget allocation algorithm result using full-year vs event-only prices
  - How much cap would be used at each threshold under both approaches

Usage (from project root):
    python3 -m model.dev.check_min_export_price S0008 2022
    python3 -m model.dev.check_min_export_price S0008 2022 \\
        --retail-tariff ea_battery_ease_vpp_2025 \\
        --op-tariff agl_operator_wholesale_2025
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model.core.paths import default_paths
from model.core.tariff import load_network_tariff, load_retail_tariff
from model.components.vpp_event_classifier import classify_vpp_events
from model.dev._dev_common import DEV_BATTERY
from model.dev.viz_vpp_sc_event import load_inputs, _load_prices_mwh


def _budget_from_prices(
    prices: np.ndarray,
    avg_load_kw: np.ndarray,
    power_kw: float,
    interval_hours: float,
    kwh_limit: float,
) -> tuple[float, float]:
    """Run the budget allocation algorithm on a price/load subset.

    Returns (marginal_price, cumulative_at_exhaustion).
    cumulative_at_exhaustion is the total kWh that could be dispatched at
    or above the marginal price -- useful to see how close to the budget
    the algorithm gets.
    """
    effective_export_kw = np.maximum(0.0, power_kw - avg_load_kw)
    contribution = effective_export_kw * interval_hours

    order = np.argsort(-prices)
    cumulative = np.cumsum(contribution[order])

    exhausted_idx = np.searchsorted(cumulative, kwh_limit)
    if exhausted_idx >= len(order):
        return 0.0, float(cumulative[-1]) if len(cumulative) > 0 else 0.0

    return float(prices[order[exhausted_idx]]), float(cumulative[exhausted_idx])


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
    parser.add_argument('--op-tariff', default='agl_operator_wholesale_2025')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--project-root', type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, network, retail, op_tariff = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, args.op_tariff, paths,
    )

    vpp_cfg = retail.vpp
    if vpp_cfg is None:
        print(f"ERROR: tariff '{retail.name}' has no VppConfig block", file=sys.stderr)
        return 1

    interval_hours = 5 / 60
    power_kw       = DEV_BATTERY.power_kw
    kwh_limit      = vpp_cfg.kwh_limit

    # Classify events
    vpp_event = classify_vpp_events(
        prices_mwh,
        threshold_dollars_per_mwh=vpp_cfg.vpp_threshold_dollars_per_mwh,
        event_window_hours=vpp_cfg.event_window_hours,
        interval_hours=interval_hours,
    )
    in_event     = vpp_event.astype(bool)
    n_event      = int(in_event.sum())
    n_total      = len(prices_mwh)
    n_non_event  = n_total - n_event

    prices_all   = prices_mwh.to_numpy()
    load_kwh_arr = trace['gross_load_kwh'].to_numpy()
    # Cap load at power_kw: load above this contributes zero to effective export
    load_kw_all  = np.minimum(load_kwh_arr / interval_hours, power_kw)

    prices_event     = prices_all[in_event]
    load_kw_event    = load_kw_all[in_event]

    print(f"\n=== Min export price diagnostic: {args.site_id} {args.year} ===")
    print(f"Tariff:          {retail.name}")
    print(f"Battery:         {power_kw} kW, budget {kwh_limit:.0f} kWh ({vpp_cfg.limit_applies_to})")
    print(f"Event threshold: {vpp_cfg.vpp_threshold_dollars_per_mwh:.0f} $/MWh, "
          f"window {vpp_cfg.event_window_hours:.0f} h")
    print(f"Total intervals: {n_total:,}  |  "
          f"In events: {n_event:,} ({n_event/n_total*100:.1f}%)  |  "
          f"Outside: {n_non_event:,}")
    print()

    # --- Interval counts above price thresholds ---
    thresholds = [300, 500, 700, 750, 800, 900, 1000, 1500, 2000, 5000, 10000]
    print("--- Intervals above price threshold ---")
    print(f"  {'Threshold':>10}  {'All year':>10}  {'In events':>10}  "
          f"{'Outside':>10}  {'% in event':>12}")
    print(f"  {'':-<10}  {'':-<10}  {'':-<10}  {'':-<10}  {'':-<12}")
    for thr in thresholds:
        n_all   = int((prices_all >= thr).sum())
        n_ev    = int((prices_event >= thr).sum())
        n_out   = n_all - n_ev
        pct     = n_ev / n_all * 100 if n_all > 0 else 0.0
        print(f"  {thr:>9.0f}+  {n_all:>10,}  {n_ev:>10,}  {n_out:>10,}  {pct:>11.1f}%")
    print()

    # --- Budget allocation: full-year vs event-only ---
    avg_load_kw_all   = load_kw_all
    avg_load_kw_event = load_kw_event

    price_full, cum_full   = _budget_from_prices(
        prices_all, avg_load_kw_all, power_kw, interval_hours, kwh_limit)
    price_event, cum_event = _budget_from_prices(
        prices_event, avg_load_kw_event, power_kw, interval_hours, kwh_limit)

    print("--- Budget allocation algorithm ---")
    print(f"  {'':30}  {'Full year':>12}  {'Events only':>12}")
    print(f"  {'':30}  {'':-<12}  {'':-<12}")
    print(f"  {'Marginal price ($/MWh)':30}  {price_full:>12.1f}  {price_event:>12.1f}")
    print(f"  {'Cumul. throughput at margin':30}  {cum_full:>12.2f}  {cum_event:>12.2f}")
    print(f"  {'Budget (kWh)':30}  {kwh_limit:>12.1f}  {kwh_limit:>12.1f}")
    print()

    # --- Available throughput at each threshold (event-window only) ---
    # Shows how much of the budget would be used if min_export_price = threshold.
    print("--- Event-window throughput available at or above each threshold ---")
    print(f"  (battery {power_kw} kW, minus avg load at those intervals)")
    print(f"  {'Threshold':>10}  {'Intervals':>10}  "
          f"{'Avail kWh':>10}  {'% of budget':>12}  {'Threshold ok?':>14}")
    print(f"  {'':-<10}  {'':-<10}  {'':-<10}  {'':-<12}  {'':-<14}")
    for thr in thresholds:
        mask    = prices_event >= thr
        n_above = int(mask.sum())
        eff_kw  = np.maximum(0.0, power_kw - load_kw_event[mask])
        avail   = float((eff_kw * interval_hours).sum())
        pct     = avail / kwh_limit * 100
        ok      = 'YES' if avail >= kwh_limit else 'no (under-budget)'
        print(f"  {thr:>9.0f}+  {n_above:>10,}  {avail:>10.2f}  "
              f"{pct:>11.1f}%  {ok:>14}")
    print()

    # --- Summary ---
    if price_event > 0:
        print(f"Recommendation: use event-only marginal price "
              f"{price_event:.1f} $/MWh instead of full-year {price_full:.1f} $/MWh.")
        diff = price_full - price_event
        print(f"Full-year algorithm overstates threshold by {diff:.1f} $/MWh "
              f"({diff/price_event*100:.1f}% higher than event-only).")
    else:
        print("Budget cannot be exhausted within event windows -- no restriction needed.")

    return 0


if __name__ == '__main__':
    sys.exit(main())