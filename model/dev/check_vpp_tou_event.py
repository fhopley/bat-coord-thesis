"""Non-interactive vpp_tou_event verification on a real site file.

Usage (from project root):
    python3 -m model.dev.check_vpp_tou_event SITE_ID YEAR
    python3 -m model.dev.check_vpp_tou_event SITE_ID YEAR \\
        --state NSW \\
        --retail-tariff ea_battery_ease_vpp_2025 \\
        --op-tariff agl_operator_wholesale_2025 \\
        --export-limit-kw 10

Identical to check_vpp_sc_event except between-event gaps are filled by
TOU-optimised dispatch (perfect_foresight_lp_advanced) rather than
self-consumption. Runs the full vpp_tou_event pipeline, prints verification
stats, and exits 0 if all checks pass, 1 otherwise.

Useful as a quick sanity check after touching vpp_sc_event.py,
perfect_foresight_lp_advanced.py, vpp_event_classifier.py,
self_consumption.py, or billing.py.

The pre-event SOC hold analysis is particularly informative for the TOU
variant: unlike the SC gap strategy, the TOU LP may deliberately discharge
the battery during TOU peak periods between events, leaving less SOC for the
retailer to dispatch during events. The avoidable import metric will typically
be higher here than for vpp_sc_event.
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

from model.core.paths import default_paths
from model.dev._dev_common import DEV_BATTERY
from model.dev.viz_vpp_sc_event import load_inputs, _print_verification
from model.dev.viz_vpp_tou_event import run_tou_analysis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id')
    parser.add_argument('year', type=int)
    parser.add_argument('--state', default='NSW')
    parser.add_argument('--network-tariff', default='ausgrid_ea025_network_2024')
    parser.add_argument('--retail-tariff', default='agl_battery_rewards_vpp_2025')
    parser.add_argument('--op-tariff', default='agl_operator_wholesale_2025')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--project-root', type=Path, default=None)
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