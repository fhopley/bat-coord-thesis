"""
relative_value_table.py

Writes the relative value of coordination table in Results, Societal and
Wholesale Value: the coordinated saving divided by the self-consumption saving
at one capacity, for each offer, on the household bill and on the net wholesale
cost.

Reads the curves financial_analysis.py writes for the conservative and
achievable run bases (<output-dir>/<basis folder>/curves_<weighting>.csv).
Household savings in those curves are against the Solar only bill on Amber, so
each offer's value at zero capacity is subtracted to give the saving against
the offer's own Solar only bill.  Wholesale savings are zero at zero capacity
on every offer and are used as they stand.

Self-consumption is the run without VPP participation where the offer has one,
matching the bars in savings_bar_figure.py.  The ratio runs from the
conservative to the achievable optimised saving over that same self-consumption
saving, which for the VPP and critical-rate offers is no events to maximum
dispatch and for Amber the two price information configurations.

Usage (from project root, after financial_analysis.py):
    python3 analysis/scripts/relative_value_table.py
    python3 analysis/scripts/relative_value_table.py --kwh 10 --scope VIC_all_years

Writes <output-dir>/capacity_comparison/relative_value_<kwh>kwh_<scope>_<weighting>.csv
and prints the table.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import financial_analysis as fa

SC_SERIES = ('self-consumption, no VPP', 'self-consumption')


def own_plan_values(output_dir: Path, basis: str, weight_tag: str,
                    scope: str, metric: str, kwh: int) -> pd.Series:
    """Saving at kwh by (offer, series), against the offer's own Solar only
    bill for the household metric.  An empty Series if the basis has no CSV."""
    path = output_dir / fa.RUN_BASES[basis]['folder'] / f'curves_{weight_tag}.csv'
    if not path.exists():
        print(f'  {basis}: no {path}, so its values are left out.')
        return pd.Series(dtype=float)
    d = pd.read_csv(path)
    d = d[(d['scope'] == scope) & (d['metric'] == metric)
          & d['kwh'].isin([0, kwh])]
    if d.empty:
        return pd.Series(dtype=float)
    wide = d.pivot_table(index=['offer', 'series'], columns='kwh',
                         values='value')
    if kwh not in wide.columns:
        return pd.Series(dtype=float)
    if metric == 'hh_saving' and 0 in wide.columns:
        return wide[kwh] - wide[0]
    return wide[kwh]


def self_consumption(values: pd.Series, offer: str) -> float:
    for series in SC_SERIES:
        if (offer, series) in values.index:
            return float(values[(offer, series)])
    return float('nan')


def build(output_dir: Path, weight_tag: str, scope: str,
          kwh: int) -> pd.DataFrame:
    rows = []
    for metric, label in (('hh_saving', 'household'),
                          ('ws_saving', 'wholesale')):
        cons = own_plan_values(output_dir, 'conservative', weight_tag, scope,
                               metric, kwh)
        ach = own_plan_values(output_dir, 'achievable', weight_tag, scope,
                              metric, kwh)
        if cons.empty:
            continue
        for offer in sorted({offer for offer, _ in cons.index}):
            sc = self_consumption(cons, offer)
            opt_cons = float(cons.get((offer, 'optimised'), float('nan')))
            opt_ach = (float(ach.get((offer, 'optimised'), float('nan')))
                       if not ach.empty else float('nan'))
            if pd.isna(sc) or sc == 0 or pd.isna(opt_cons):
                continue
            rows.append(dict(
                metric=label, offer=offer, kwh=kwh, scope=scope,
                self_consumption_saving=sc,
                optimised_saving_conservative=opt_cons,
                optimised_saving_achievable=opt_ach,
                relative_value_conservative=opt_cons / sc,
                relative_value_achievable=opt_ach / sc,
            ))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--output-dir', type=Path,
                        default=Path('analysis/output/financial'),
                        help="financial_analysis.py's output directory, read "
                             "for the curves and written under "
                             "capacity_comparison/")
    parser.add_argument('--kwh', type=int, default=20,
                        help='Battery capacity, one of the modelled sizes')
    parser.add_argument('--scope', default='national',
                        help="'national', '<STATE>_all_years' or "
                             "'<STATE>_<year>'")
    parser.add_argument('--weighting', default='weighted',
                        choices=('weighted', 'unweighted'))
    args = parser.parse_args(argv)

    table = build(args.output_dir, args.weighting, args.scope, args.kwh)
    if table.empty:
        sys.exit('No curves found; run financial_analysis.py first.')

    out_dir = args.output_dir / 'capacity_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (f'relative_value_{args.kwh}kwh_{args.scope}_'
                     f'{args.weighting}.csv')
    table.to_csv(out, index=False)

    pd.set_option('display.width', 200, 'display.float_format',
                  '{:.2f}'.format)
    for metric in table['metric'].unique():
        t = table[table['metric'] == metric].sort_values(
            'relative_value_achievable', ascending=False)
        print(f'\nRelative value of coordination at {args.kwh} kWh, '
              f'{metric}, {args.scope}, {args.weighting}:')
        for _, r in t.iterrows():
            print(f"  {r['offer']:30s} "
                  f"{r['relative_value_conservative']:.1f} - "
                  f"{r['relative_value_achievable']:.1f}")
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
