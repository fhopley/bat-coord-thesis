"""
offer_outcomes_text_values.py

Prints the values the Results text takes from the battery-size sweep
(docs/cap_account_battery_size_spec.md, "Values needed for the text") from
offer_outcomes_by_size.csv, written by offer_outcomes_by_size.py.

    python3 analysis/scripts/offer_outcomes_text_values.py
"""
import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTCOMES_CSV = (REPO_ROOT / 'analysis/output/cap_account_retailer/by_size/'
                'offer_outcomes_by_size.csv')
SIZES = (5, 10, 20, 50)
DESIGNED = ('Cohort Hedging', 'Retailer Pool Retention')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--outcomes', type=Path, default=OUTCOMES_CSV)
    args = ap.parse_args(argv)
    df = pd.read_csv(args.outcomes)
    at = df[df['battery_kwh'].isin(SIZES)]
    pd.set_option('display.width', 200, 'display.max_columns', 30,
                  'display.float_format', '{:.2f}'.format)

    def show(title, mask, cols):
        print(f'\n{title}')
        t = at[mask(at)].sort_values(['offer', 'settlement_period',
                                      'battery_kwh'])
        print(t[['offer', 'settlement_period', 'battery_kwh', *cols]]
              .to_string(index=False))

    show('Designed variants: mean gross margin, worst month, months with a '
         'net cost', lambda d: d['offer'].isin(DESIGNED),
         ['mean_gross_margin', 'worst_month_gross_margin', 'worst_month_state',
          'worst_month', 'months_mechanism_cost'])
    show('Retailer Pool Retention: subscription change against Amber '
         '($/month; the subscription is 25 plus this)',
         lambda d: d['offer'] == 'Retailer Pool Retention',
         ['subscription_change_used', 'subscription_or_net_fixed'])
    show('Designed variants: household 5th-95th percentile, % of '
         'coordination value', lambda d: d['offer'].isin(DESIGNED),
         ['p5_pct_of_coordination_savings', 'p95_pct_of_coordination_savings',
          'p5_bill_change', 'p95_bill_change'])
    show('Tesla Electric', lambda d: d['offer'] == 'Tesla Electric',
         ['mean_gross_margin', 'months_below_zero',
          'worst_month_gross_margin', 'worst_month_state', 'worst_month'])

    c = df[(df['offer'] == 'Cohort Hedging')
           & (df['settlement_period'] == 'monthly')].sort_values('battery_kwh')
    ok = (c['months_mechanism_cost'] <= 1)[::-1].cummin()[::-1]
    print('\nCohort Hedging (monthly): months with a net cost by size')
    print(c[['battery_kwh', 'months_mechanism_cost', 'mean_gross_margin']]
          .to_string(index=False))
    sizes_ok = c.loc[ok, 'battery_kwh']
    print('Smallest size from which at most one month has a net cost: '
          + (f'{int(sizes_ok.min())} kWh' if len(sizes_ok) else 'none'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
