"""Export a single site's per-interval billing data for manual verification.

Runs the real pipeline for one site/year (load trace -> self_consumption with
the hardcoded DEV_BATTERY -> billing) using the production network tariff and a
chosen production retail tariff. Writes:

  1. A per-interval CSV (full year, 5-min) with flows, wholesale price, and the
     resolved per-interval tariff rates, plus mapping-aid columns
     (interval_start, start_hour, is_weekend) so the period lookups can be
     spot-checked from the timestamps.
  2. A .txt summary with the per-component SiteBill breakdown AND a set of
     clearly-labelled intermediate sums (total kWh, loss-adjusted kWh, n_days)
     that are building blocks for the components - NOT additional bill lines.

The intent is to rebuild the bill independently in Excel and reconcile against
the summary.

Usage (from project root):
    python3 -m model.dev.export_billing_check SITE_ID YEAR [--retail wholesale|tou]

Network tariff is always ausgrid_ea025_network_2024.
Retail tariff defaults to wholesale (amber_wholesale_2024); tou selects
powershop_tou_2024.

CSV columns
-----------
  datetime_aest                       interval-end timestamp (the meaningful one)
  interval_start                      datetime_aest - 5 min (drives period lookup)
  start_hour                          hour (0-23) of interval_start
  is_weekend                          True if interval_start is Sat/Sun
  import_kwh, export_kwh              flows from self_consumption (with battery)
  wholesale_price_dollar_per_mwh      raw NEM RRP
  network_usage_rate_c_per_kwh        resolved network import period rate
  network_feedin_rate_c_per_kwh       resolved network export period rate
  retail_usage_period_rate_c_per_kwh  resolved retail period rate (0 for wholesale_exposed)
  retail_usage_other_c_per_kwh        flat retail import adder (constant)
  retail_feedin_period_rate_c_per_kwh resolved retail feedin period rate (0 for wholesale_exposed)
  retail_feedin_other_c_per_kwh       flat retail feedin adder (constant)
  import_loss_factor                  usage_dlf * import_tlf (constant)
  export_loss_factor                  export_dlf * export_tlf (constant)

Excel reconstruction (per the model sign conventions):
  wholesale_usage_cost   = SUM(import_kwh * price/1000 * import_loss_factor)
  wholesale_feedin_credit= -SUM(export_kwh * price/1000 * export_loss_factor)
  network_usage_cost     = SUM(import_kwh * network_usage_rate/100 * import_loss_factor)
  network_feedin_credit  = -SUM(export_kwh * network_feedin_rate/100 * export_loss_factor)
  retail_usage_cost      = SUM(import_kwh * (retail_usage_period_rate + retail_usage_other)/100)
  retail_feedin_credit   = -SUM(export_kwh * (retail_feedin_period_rate + retail_feedin_other)/100)
  network_daily_charges  = network_daily_c_per_day/100 * n_days
  retail_daily_charges   = retail_daily_c_per_day/100 * n_days
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from model.core.paths import default_paths
from model.core.tariff import (
    load_network_tariff, load_retail_tariff, period_rate_series,
)
from model.components.self_consumption import self_consumption
from model.components.billing import compute_bill
from model.dev._dev_common import DEV_BATTERY, load_site_trace


NETWORK_TARIFF_ID = 'ausgrid_ea025_network_2024'
RETAIL_TARIFF_IDS = {
    'wholesale': 'amber_wholesale_2024',
    'tou': 'powershop_tou_2024',
}
INTERVAL_HOURS = 5 / 60


def _resolved_rate(periods, index, weekend=None):
    """period_rate_series wrapper that returns zeros for an empty period set
    (wholesale_exposed has no period rates)."""
    if not periods:
        return np.zeros(len(index))
    return period_rate_series(periods, index, INTERVAL_HOURS, weekend).to_numpy()


def build_csv(traces, prices, network_tariff, retail_tariff) -> pd.DataFrame:
    index = traces.import_kwh.index
    interval_start = index - pd.Timedelta(hours=INTERVAL_HOURS)

    net_usage = _resolved_rate(network_tariff.usage_tariffs, index,
                               network_tariff.usage_tariffs_weekend)
    net_feedin = _resolved_rate(network_tariff.feedin_tariffs, index)

    ret_usage_period = _resolved_rate(retail_tariff.usage_tariffs, index,
                                      retail_tariff.usage_tariffs_weekend)
    ret_feedin_period = _resolved_rate(retail_tariff.feedin_tariffs, index)

    df = pd.DataFrame({
        'datetime_aest': index.strftime('%Y-%m-%d %H:%M:%S%z'),
        'interval_start': interval_start.strftime('%Y-%m-%d %H:%M:%S%z'),
        'start_hour': interval_start.hour,
        'is_weekend': interval_start.dayofweek >= 5,
        'import_kwh': traces.import_kwh.to_numpy(),
        'export_kwh': traces.export_kwh.to_numpy(),
        'wholesale_price_dollar_per_mwh': prices.to_numpy(),
        'network_usage_rate_c_per_kwh': net_usage,
        'network_feedin_rate_c_per_kwh': net_feedin,
        'retail_usage_period_rate_c_per_kwh': ret_usage_period,
        'retail_usage_other_c_per_kwh': retail_tariff.usage_other_c_per_kwh,
        'retail_feedin_period_rate_c_per_kwh': ret_feedin_period,
        'retail_feedin_other_c_per_kwh': retail_tariff.feedin_other_c_per_kwh,
        'import_loss_factor': network_tariff.import_loss_factor,
        'export_loss_factor': network_tariff.export_loss_factor,
    })
    return df


def build_summary(site_id, year, retail_choice, traces, prices,
                  network_tariff, retail_tariff, bill) -> str:
    index = traces.import_kwh.index
    interval_start = index - pd.Timedelta(hours=INTERVAL_HOURS)
    n_days = (interval_start[-1].normalize()
              - interval_start[0].normalize()).days + 1

    imp = traces.import_kwh.to_numpy()
    exp = traces.export_kwh.to_numpy()
    ilf = network_tariff.import_loss_factor
    elf = network_tariff.export_loss_factor

    L = []
    L.append(f"Billing verification summary")
    L.append(f"=" * 60)
    L.append(f"Site:            {site_id}")
    L.append(f"Year:            {year}")
    L.append(f"Network tariff:  {network_tariff.name} ({NETWORK_TARIFF_ID})")
    L.append(f"Retail tariff:   {retail_tariff.name} "
             f"({RETAIL_TARIFF_IDS[retail_choice]}, type={retail_tariff.type})")
    L.append(f"Battery:         {DEV_BATTERY.capacity_kwh} kWh / "
             f"{DEV_BATTERY.power_kw} kW / reserve {DEV_BATTERY.backup_reserve_pct}% / "
             f"eff {DEV_BATTERY.charge_efficiency}/{DEV_BATTERY.discharge_efficiency}")
    L.append("")

    L.append("INTERMEDIATE SUMS (building blocks, NOT bill line items - do not")
    L.append("add these to the component total; they feed into the components)")
    L.append("-" * 60)
    L.append(f"  total_import_kwh                 {imp.sum():14.4f}")
    L.append(f"  total_export_kwh                 {exp.sum():14.4f}")
    L.append(f"  loss_adj_import_kwh (x{ilf:.6f}) {imp.sum() * ilf:14.4f}")
    L.append(f"  loss_adj_export_kwh (x{elf:.6f}) {exp.sum() * elf:14.4f}")
    L.append(f"  n_days                          {n_days:14d}")
    L.append(f"  import_loss_factor              {ilf:14.6f}")
    L.append(f"  export_loss_factor             {elf:14.6f}")
    L.append(f"  network_daily_c_per_day        {network_tariff.daily_c_per_day:14.4f}")
    L.append(f"  retail_daily_c_per_day         {retail_tariff.daily_c_per_day:14.4f}")
    L.append("")

    L.append("BILL COMPONENTS ($) - these sum per the views below")
    L.append("-" * 60)
    L.append(f"  wholesale_usage_cost            {bill.wholesale_usage_cost:14.4f}")
    L.append(f"  wholesale_feedin_credit         {bill.wholesale_feedin_credit:14.4f}")
    L.append(f"  network_usage_cost              {bill.network_usage_cost:14.4f}")
    L.append(f"  network_feedin_credit           {bill.network_feedin_credit:14.4f}")
    L.append(f"  network_daily_charges           {bill.network_daily_charges:14.4f}")
    L.append(f"  retail_usage_cost               {bill.retail_usage_cost:14.4f}")
    L.append(f"  retail_feedin_credit            {bill.retail_feedin_credit:14.4f}")
    L.append(f"  retail_daily_charges            {bill.retail_daily_charges:14.4f}")
    L.append("")

    L.append("VIEWS ($)")
    L.append("-" * 60)
    L.append(f"  plan_type                       {bill.plan_type:>14}")
    L.append(f"  household_total                 {bill.household_total:14.4f}")
    L.append(f"  retailer_margin                 {bill.retailer_margin:14.4f}")
    L.append("")

    if retail_tariff.type == 'wholesale_exposed':
        L.append("  household_total (wholesale_exposed) = sum of ALL 8 components above")
    else:
        L.append("  household_total (tou) = retail_usage_cost + retail_feedin_credit")
        L.append("                          + retail_daily_charges  (retail layer only)")
    L.append("  retailer_margin = household_total")
    L.append("                    - (wholesale_usage_cost + wholesale_feedin_credit)")
    L.append("                    - (network_usage_cost + network_feedin_credit")
    L.append("                       + network_daily_charges)")
    L.append("")

    return "\n".join(L)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id')
    parser.add_argument('year', type=int)
    parser.add_argument('--retail', choices=['wholesale', 'tou'],
                        default='wholesale',
                        help="Retail tariff to apply (default: wholesale)")
    parser.add_argument('--project-root', type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)

    # Load trace and run self_consumption with the dev battery
    trace = load_site_trace(args.site_id, args.year, paths)
    if 'pv_site_net_kwh' not in trace or 'gross_load_kwh' not in trace:
        print("ERROR: trace missing pv_site_net_kwh or gross_load_kwh",
              file=sys.stderr)
        return 1

    result = self_consumption(
        site_id=args.site_id,
        solar_kwh=trace['pv_site_net_kwh'],
        load_kwh=trace['gross_load_kwh'],
        battery_spec=DEV_BATTERY,
    )

    # Load tariffs and prices
    network_tariff = load_network_tariff(NETWORK_TARIFF_ID, paths.tariffs_dir)
    retail_tariff = load_retail_tariff(RETAIL_TARIFF_IDS[args.retail],
                                       paths.tariffs_dir)
    price_path = paths.nem_prices_dir / f'{args.year}_nem_prices.parquet'
    if not price_path.exists():
        print(f"ERROR: price file not found: {price_path}", file=sys.stderr)
        return 1
    pdf = pd.read_parquet(price_path).set_index('datetime_aest')
    # map scenario state from tariff state
    rrp_col = {'NSW': 'NSW1_rrp', 'QLD': 'QLD1_rrp',
               'SA': 'SA1_rrp', 'VIC': 'VIC1_rrp'}[network_tariff.state]
    prices = pdf[rrp_col]
    prices.index = result.import_kwh.index  # align (both full-year 5-min AEST)

    # Compute the bill
    bill = compute_bill(result, prices, network_tariff, retail_tariff)

    # Build outputs
    csv_df = build_csv(result, prices, network_tariff, retail_tariff)
    summary = build_summary(args.site_id, args.year, args.retail, result,
                            prices, network_tariff, retail_tariff, bill)

    out_dir = (args.project_root or Path.cwd()) / 'data' / 'processed' / 'billing_check'
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"billing_check_{args.site_id}_{args.year}_{args.retail}"
    csv_path = out_dir / f"{base}.csv"
    txt_path = out_dir / f"{base}.txt"

    csv_df.to_csv(csv_path, index=False)
    txt_path.write_text(summary)

    print(summary)
    print(f"CSV written to:     {csv_path}  ({len(csv_df)} rows)")
    print(f"Summary written to: {txt_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())