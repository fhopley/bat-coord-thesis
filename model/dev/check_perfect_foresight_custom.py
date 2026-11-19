"""Non-interactive perfect_foresight_custom verification on a real site file.

Usage (from project root):
    python3 -m model.dev.check_perfect_foresight_custom SITE_ID YEAR
    python3 -m model.dev.check_perfect_foresight_custom SITE_ID YEAR \\
        --state NSW --retail-tariff amber_wholesale_2024 --export-limit-kw 10 \\
        --opt-factor 1.0 --compare-lp

Loads a real site trace + NEM prices + tariffs, runs self_consumption then
perfect_foresight_custom (the greedy iterative-shift heuristic, with DEV_BATTERY,
no degradation cost, a per-phase export limit, SOC starting at the reserve floor
and ending at the self_consumption end-of-period SOC), prints verification stats,
and exits 0 if all checks pass, 1 otherwise.

With --compare-lp it also runs perfect_foresight_lp on identical inputs and
reports the heuristic-vs-LP gap (dispatch objective and household bill). The LP
is the global optimum, so the custom dispatch objective must be >= the LP and the
gap quantifies how much the heuristic leaves on the table.

Useful as a quick sanity check after touching perfect_foresight_custom.py,
self_consumption.py, prices.py, or billing.py.
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from model.core.paths import default_paths
from model.components.billing import compute_bill
from model.components.perfect_foresight_lp import perfect_foresight_lp
from model.dev._dev_common import DEV_BATTERY
from model.dev.viz_perfect_foresight_custom import (
    load_inputs,
    filter_date_range,
    run_custom_analysis,
    _print_verification,
)


def _compare_to_lp(analysis, prices_mwh) -> str:
    """Run perfect_foresight_lp on the same inputs and return a gap report.

    The LP is the global optimum for the dispatch objective, so the heuristic's
    objective must be >= the LP's; the difference is the heuristic gap.
    """
    a = analysis
    solar = a.trace['pv_site_net_kwh']
    load = a.trace['gross_load_kwh']
    export_limit_kwh = a.export_limit_kw * a.interval_hours

    lp_result = perfect_foresight_lp(
        a.site_id, solar, load, a.p_imp, a.p_exp, a.battery_spec,
        soc_init_kwh=a.soc_init_kwh, soc_final_kwh=a.soc_final_kwh,
        export_limit_kwh_per_interval=export_limit_kwh,
        interval_hours=a.interval_hours,
    )
    lp_bill = compute_bill(lp_result, prices_mwh, a.network_tariff,
                           a.retail_tariff, a.state, a.interval_hours)

    p_imp = a.p_imp.to_numpy()
    p_exp = a.p_exp.to_numpy()

    def dispatch_obj(r):
        return float(np.sum(r.import_kwh.to_numpy() * p_imp
                            - r.export_kwh.to_numpy() * p_exp))

    obj_custom = dispatch_obj(a.custom_result)
    obj_lp = dispatch_obj(lp_result)
    gap = obj_custom - obj_lp
    denom = abs(obj_lp) + 1e-9

    bill_custom = a.custom_bill.household_total
    bill_lp = lp_bill.household_total

    status = 'PASS' if obj_custom >= obj_lp - 1e-4 else 'FAIL'

    lines = []
    lines.append("--- Custom vs LP (perfect_foresight_lp) ---")
    lines.append(f"dispatch objective  custom=${obj_custom:,.4f}  "
                 f"lp=${obj_lp:,.4f}  gap=${gap:+,.4f} ({100 * gap / denom:+.2f}%)")
    lines.append(f"household bill       custom=${bill_custom:,.2f}  "
                 f"lp=${bill_lp:,.2f}  gap=${bill_custom - bill_lp:+,.2f}")
    lines.append(f"custom objective >= LP objective: {status}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id')
    parser.add_argument('year', type=int)
    parser.add_argument('--state', default='NSW')
    parser.add_argument('--network-tariff', default='ausgrid_ea025_network_2024')
    parser.add_argument('--retail-tariff', default='amber_wholesale_2024')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--opt-factor', type=float, default=1.0,
                        help='Fraction (0,1] of opportunities to process (default 1.0)')
    parser.add_argument('--compare-lp', action='store_true',
                        help='Also run perfect_foresight_lp and report the gap')
    parser.add_argument('--date-from', default=None, metavar='YYYY-MM-DD',
                        help='Start date inclusive (default: first day of trace)')
    parser.add_argument('--date-to', default=None, metavar='YYYY-MM-DD',
                        help='End date inclusive (default: last day of trace)')
    parser.add_argument('--project-root', type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, network_tariff, retail_tariff = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, paths,
    )
    trace, prices_mwh = filter_date_range(
        trace, prices_mwh, args.date_from, args.date_to,
    )

    analysis = run_custom_analysis(
        args.site_id, args.year, trace, prices_mwh,
        network_tariff, retail_tariff, DEV_BATTERY,
        export_limit_kw=args.export_limit_kw,
        state=args.state,
        opt_factor=args.opt_factor,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_verification(analysis)
        if args.compare_lp:
            print(_compare_to_lp(analysis, prices_mwh), end='')
    output = buf.getvalue()
    print(output, end='')

    if 'FAIL' in output:
        print("OVERALL: FAIL", file=sys.stderr)
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())