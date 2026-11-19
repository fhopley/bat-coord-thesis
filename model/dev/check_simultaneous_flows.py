"""
Scan a custom run for simultaneous charge+discharge and negative flow values,
reporting affected intervals with prices and flow values.

Usage:
    python3 -m model.dev.check_simultaneous_flows SITE_ID YEAR \\
        [--date-from YYYY-MM-DD] [--date-to YYYY-MM-DD]
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from model.core.paths import default_paths
from model.core.prices import build_price_vectors
from model.components.self_consumption import self_consumption
from model.components.perfect_foresight_custom import perfect_foresight_custom
from model.dev._dev_common import DEV_BATTERY
from model.dev.viz_perfect_foresight_custom import load_inputs, filter_date_range

_SIMUL_TOL = 1e-6
_NEG_TOL   = 1e-9


def check_trace(tr, p_imp_np, p_exp_np, label="custom"):
    g2b = tr.grid_to_bat_kwh.to_numpy()
    b2h = tr.bat_to_home_kwh.to_numpy()
    b2g = tr.bat_to_grid_kwh.to_numpy()
    s2b = tr.sol_to_bat_kwh.to_numpy()
    g2h = tr.grid_to_home_kwh.to_numpy()
    cur = tr.curtail_kwh.to_numpy()
    idx = tr.import_kwh.index

    issues = []

    simul = (g2b > _SIMUL_TOL) & ((b2h + b2g) > _SIMUL_TOL)
    for i in np.where(simul)[0]:
        issues.append(dict(
            kind='SIMUL_CHG_DISC', time=idx[i],
            pi=p_imp_np[i], pe=p_exp_np[i],
            g2b=g2b[i], b2h=b2h[i], b2g=b2g[i], s2b=s2b[i], g2h=g2h[i], cur=cur[i],
        ))

    for arr, name in [(g2h, 'g2h'), (cur, 'curtail'), (b2h, 'b2h'), (g2b, 'g2b')]:
        for i in np.where(arr < -_NEG_TOL)[0]:
            issues.append(dict(
                kind=f'NEG_{name}', time=idx[i],
                pi=p_imp_np[i], pe=p_exp_np[i],
                value=arr[i],
                g2b=g2b[i], b2h=b2h[i], g2h=g2h[i], cur=cur[i],
            ))

    print(f"\n[{label}] {len(issues)} issue(s)")
    if not issues:
        print("  Clean.")
        return

    for iss in sorted(issues, key=lambda x: x['time']):
        t = iss['time']
        if iss['kind'] == 'SIMUL_CHG_DISC':
            print(f"  {t}  SIMUL  "
                  f"p_imp={iss['pi']*100:+7.2f}c  p_exp={iss['pe']*100:+7.2f}c  "
                  f"g2b={iss['g2b']:.5f}  b2h={iss['b2h']:.5f}  b2g={iss['b2g']:.5f}  "
                  f"s2b={iss['s2b']:.5f}")
        else:
            print(f"  {t}  {iss['kind']:12s}  "
                  f"p_imp={iss['pi']*100:+7.2f}c  p_exp={iss['pe']*100:+7.2f}c  "
                  f"val={iss['value']:.2e}  "
                  f"g2b={iss['g2b']:.5f}  g2h={iss['g2h']:.5f}  cur={iss['cur']:.5f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id')
    parser.add_argument('year', type=int)
    parser.add_argument('--state',           default='NSW')
    parser.add_argument('--network-tariff',  default='ausgrid_ea025_network_2024')
    parser.add_argument('--retail-tariff',   default='amber_wholesale_2024')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--date-from',       default=None, metavar='YYYY-MM-DD')
    parser.add_argument('--date-to',         default=None, metavar='YYYY-MM-DD')
    parser.add_argument('--project-root',    type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, nt, rt = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, paths,
    )
    trace, prices_mwh = filter_date_range(
        trace, prices_mwh, args.date_from, args.date_to,
    )

    spec    = DEV_BATTERY
    IH      = 5 / 60
    soc_min = spec.backup_reserve_pct / 100.0 * spec.capacity_kwh
    p_imp, p_exp = build_price_vectors(trace.index, prices_mwh, nt, rt, IH)
    pi_np = p_imp.to_numpy(); pe_np = p_exp.to_numpy()
    solar  = trace['pv_site_net_kwh']
    load_s = trace['gross_load_kwh']

    sc = self_consumption(args.site_id, solar, load_s, spec, IH,
                          neg_price_aware=True, price_imp=p_imp, price_exp=p_exp)
    sf = float(min(max(sc.battery_soc_kwh.to_numpy()[-1], soc_min), spec.capacity_kwh))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        r = perfect_foresight_custom(
            args.site_id, solar, load_s, p_imp, p_exp, spec,
            soc_init_kwh=soc_min, soc_final_kwh=sf,
            export_limit_kwh_per_interval=args.export_limit_kw * IH,
            interval_hours=IH,
        )

    print(f"\n=== Flow check: {args.site_id} {args.year} ===")
    if caught:
        print("Warnings from _assemble_traces:")
        for w in caught:
            print(f"  {w.message}")

    check_trace(r, pi_np, pe_np)
    return 0


if __name__ == '__main__':
    sys.exit(main())
