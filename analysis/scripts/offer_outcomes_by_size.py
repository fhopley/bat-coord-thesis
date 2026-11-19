"""
offer_outcomes_by_size.py

Summary table for the battery-size sweep of the alternative offers
(docs/cap_account_battery_size_spec.md): one row per variant and battery
size, retailer and household outcomes side by side.

Retailer columns come from cap_account_retailer.py's output at each size
({retailer-root}/{battery}/x300/): retailer_cohort_months.csv for the
percentiles and the worst month, retailer_composition_means.csv for the mean
gross margin and its components, retailer_summary.csv for the months the
mechanism cost the retailer and the subscription change used. All are per
household per month across the 240 state-months, weighted by state mix.

Household columns come from cap_account_aggregate.py's
household_outcomes_weighted*.csv for the variant at that size
({aggregate-root}/{battery}/{variant}/national/).

    python3 analysis/scripts/offer_outcomes_by_size.py

Writes analysis/output/cap_account_retailer/by_size/offer_outcomes_by_size.csv.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cap_account_retailer import _weighted_percentile  # noqa: E402

THRESHOLD_GROUP = 'x300'
AMBER = 'Amber'
VALUE_COL = 'net_position_per_site'

# The sweep's variants (runners/run_cap_account_by_size.sh); other variants
# left in a size's output directory by earlier runs are not tabulated.
VARIANTS = ('cohort_x300_monthly_s0.01', 'cohort_x300_daily_s0.01',
            'retailer_x300_monthly_a0.9', 'retailer_x300_daily_a0.9',
            'dynamic_tesla_electric_mixed_2026_modelled')

PERCENTILES = (5, 25, 50, 75, 95)
COMPONENTS = ('subscription_or_net_fixed', 'import_margin', 'cost_absorbed',
              'credit_retained', 'surcharge_retained')
HOUSEHOLD_COLS = ('mean_bill_change', 'p5_bill_change', 'p95_bill_change',
                  'p5_pct_of_coordination_savings',
                  'p50_pct_of_coordination_savings',
                  'p95_pct_of_coordination_savings', 'share_better_off')


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--retailer-root', type=Path,
                   default=REPO_ROOT / 'analysis/output/cap_account_retailer')
    p.add_argument('--aggregate-root', type=Path,
                   default=REPO_ROOT / 'analysis/output/cap_account_aggregate')
    p.add_argument('--output', type=Path, default=None,
                   help="Default: {retailer-root}/by_size/"
                        "offer_outcomes_by_size.csv.")
    return p.parse_args(argv)


def _sizes(retailer_root: Path) -> list[int]:
    found = []
    for d in retailer_root.iterdir():
        m = re.fullmatch(r'(\d+)kWh', d.name)
        if m and (d / THRESHOLD_GROUP / 'retailer_summary.csv').exists():
            found.append(int(m.group(1)))
    return sorted(found)


def _offer_fields(variant: str) -> tuple[str, str]:
    """'cohort_x300_monthly_s0.01' -> ('Cohort Hedging', 'monthly')."""
    if variant.startswith('dynamic_'):
        return 'Tesla Electric', 'monthly'
    m = re.match(r'^(cohort|retailer)_x[\d.]+_(\w+?)_[sa][\d.]+$', variant)
    offer = {'cohort': 'Cohort Hedging',
             'retailer': 'Retailer Pool Retention'}[m.group(1)]
    return offer, m.group(2)


def _household(aggregate_root: Path, battery: str, variant: str) -> dict:
    folder = aggregate_root / battery / variant / 'national'
    files = sorted(folder.glob('household_outcomes_weighted*.csv'))
    if not files:
        print(f"WARNING: no household outcomes for {variant} {battery} "
              f"under {folder}", file=sys.stderr)
        return {c: np.nan for c in HOUSEHOLD_COLS}
    if len(files) > 1:
        print(f"WARNING: {len(files)} household outcome files in {folder}; "
              f"using {files[-1].name}", file=sys.stderr)
    row = pd.read_csv(files[-1]).iloc[0]
    return {c: float(row[c]) for c in HOUSEHOLD_COLS}


def _size_rows(size: int, args) -> list[dict]:
    battery = f'{size}kWh'
    folder = args.retailer_root / battery / THRESHOLD_GROUP
    months = pd.read_csv(folder / 'retailer_cohort_months.csv')
    comp = pd.read_csv(folder / 'retailer_composition_means.csv'
                       ).set_index('variant')
    summary = pd.read_csv(folder / 'retailer_summary.csv').set_index('variant')

    rows = []
    for variant in VARIANTS:
        if variant not in comp.index:
            print(f"WARNING: {variant} missing at {battery}", file=sys.stderr)
            continue
        m = months[months['variant'] == variant]
        v = m[VALUE_COL].to_numpy()
        w = m['weight'].to_numpy()
        worst = m.loc[m[VALUE_COL].idxmin()]
        offer, period = _offer_fields(variant)
        row = {
            'variant': variant,
            'offer': offer,
            'settlement_period': period,
            'battery_kwh': size,
            'n_cohort_months': len(m),
            'mean_gross_margin': float(comp.loc[variant, 'gross_margin']),
        }
        for c in COMPONENTS:
            row[c] = float(comp.loc[variant, c])
        for q in PERCENTILES:
            row[f'p{q}_gross_margin'] = _weighted_percentile(v, w, float(q))
        row['worst_month_gross_margin'] = float(worst[VALUE_COL])
        row['worst_month_state'] = worst['state']
        row['worst_month'] = worst['month']
        row['months_mechanism_cost'] = int(
            summary.loc[variant, 'months_retailer_lost_money'])
        row['months_below_zero'] = int((v < 0).sum())
        row['subscription_change_used'] = float(
            summary.loc[variant, 'subscription_change_used']
            if 'subscription_change_used' in summary.columns else np.nan)
        row['subscription_change_vs_amber'] = float(
            summary.loc[variant, 'subscription_change_vs_amber'])
        row.update(_household(args.aggregate_root, battery, variant))
        rows.append(row)
    return rows


def main(argv=None) -> int:
    args = _parse_args(argv)
    sizes = _sizes(args.retailer_root)
    if not sizes:
        print(f"No {{size}}kWh/{THRESHOLD_GROUP}/retailer_summary.csv under "
              f"{args.retailer_root}.", file=sys.stderr)
        return 1
    rows = []
    for size in sizes:
        rows.extend(_size_rows(size, args))
    out = pd.DataFrame(rows).sort_values(['offer', 'settlement_period',
                                          'battery_kwh']).reset_index(drop=True)
    path = args.output or (args.retailer_root / 'by_size'
                           / 'offer_outcomes_by_size.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Sizes: {sizes}\nWrote {len(out)} rows to {path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
