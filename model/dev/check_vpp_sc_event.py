"""Non-interactive vpp_sc_event verification on a real site file.

Usage (from project root):
    python3 -m model.dev.check_vpp_sc_event SITE_ID YEAR
    python3 -m model.dev.check_vpp_sc_event SITE_ID YEAR \\
        --state NSW \\
        --retail-tariff NSW_agl_battery_rewards_vpp_2026 \\
        --op-tariff NSW_agl_operator_wholesale_2026 \\
        --export-limit-kw 10

Loads a real site trace + NEM prices + tariffs, runs self_consumption for the
SC baseline bill, runs the VPP event classifier, then runs vpp_sc_event.
Prints verification stats including the VPP event summary, avoidable import
metric, and billing comparison, then exits 0 if all checks pass, 1 otherwise.

Useful as a quick sanity check after touching vpp_sc_event.py,
vpp_event_classifier.py, self_consumption.py, or billing.py.

The avoidable import metric quantifies pre-event SOC holding: it is the grid
import in non-event intervals where the battery had capacity to serve the load
but didn't. By construction this should be near zero for vpp_sc_event
(the battery follows self-consumption between events).
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

from model.core.paths import default_paths
from model.dev._dev_common import DEV_BATTERY
from model.dev.viz_vpp_sc_event import (
    load_inputs,
    run_eov_analysis,
    _print_verification,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id')
    parser.add_argument('year', type=int)
    parser.add_argument('--state', default='NSW')
    parser.add_argument('--network-tariff', default='NSW_ausgrid_ea025_ea029_2026')
    parser.add_argument('--retail-tariff', default='NSW_agl_battery_rewards_vpp_2026')
    parser.add_argument('--op-tariff', default='NSW_agl_operator_wholesale_2026')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--project-root', type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, network, retail, op_tariff = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, args.op_tariff, paths,
    )

    analysis = run_eov_analysis(
        args.site_id, args.year, trace, prices_mwh,
        network, retail, op_tariff, DEV_BATTERY,
        export_limit_kw=args.export_limit_kw,
        state=args.state,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_verification(analysis)
    output = buf.getvalue()
    print(output, end='')

    if 'FAIL' in output:
        print("OVERALL: FAIL", file=sys.stderr)
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())