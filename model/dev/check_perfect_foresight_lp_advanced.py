"""Non-interactive perfect_foresight_lp_advanced verification on a real site file.

Usage (from project root):
    python3 -m model.dev.check_perfect_foresight_lp_advanced SITE_ID YEAR
    python3 -m model.dev.check_perfect_foresight_lp_advanced SITE_ID YEAR \\
        --state NSW \\
        --retail-tariff globird_advanced_tou_2025 \\
        --export-limit-kw 10

Validation steps:

  Standard tariff equivalence (Step 18 -- confirmed passing):
    Run with --retail-tariff amber_wholesale_2024 (or any standard plan).
    The advanced LP produces identical household_total within 1e-4 because
    build_lp_price_vectors_advanced is a no-op for plans without advanced
    features and perfect_foresight_lp_advanced delegates to the same _build_lp.

  Advanced tariff (Step 19 -- confirmed passing on S0008/2022):
    Run with --retail-tariff globird_advanced_tou_2025. Expected output:
      - Block tier daily allocations: tier1 fills before spillover
      - Zero-import reward: ~363/365 qualifying days, reward reconciles with billing
      - Critical export / import: interval counts, critical-side kWh and value
      - LP price reconciliation: PASS -- the per-interval LP-price cost plus the
        block tier cost, less the critical rates' opposite-side shift, equals
        the bill's dispatch part

  c_deg_per_kwh recommendation:
    Set c_deg_per_kwh >= 0.005 $/kWh (in BatterySpec / DEV_BATTERY) when using
    plans with a negative critical import rate. At negative wholesale prices the LP can
    find it profitable to simultaneously charge and discharge the battery to
    maximise imports. c_deg >= 0.005 makes this unprofitable and produces
    physically sensible dispatch.

Exits 0 if all PASS/FAIL checks pass (INFO items do not affect exit code), 1 otherwise.
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

from model.core.paths import default_paths
from model.dev._dev_common import DEV_BATTERY
from model.dev.viz_perfect_foresight_lp import load_inputs
from model.dev.viz_perfect_foresight_lp_advanced import (
    run_lp_advanced_analysis,
    _print_verification_advanced,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('site_id')
    parser.add_argument('year', type=int)
    parser.add_argument('--state', default='NSW')
    parser.add_argument('--network-tariff', default='ausgrid_ea025_network_2024')
    parser.add_argument('--retail-tariff', default='globird_advanced_tou_2025',
                        help='Retail tariff ID (default: globird_advanced_tou_2025). '
                             'Use a standard tariff (e.g. amber_wholesale_2024) for '
                             'Step 18 equivalence validation against the standard LP.')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--project-root', type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, network_tariff, retail_tariff = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, paths,
    )

    analysis = run_lp_advanced_analysis(
        args.site_id, args.year, trace, prices_mwh,
        network_tariff, retail_tariff, DEV_BATTERY,
        export_limit_kw=args.export_limit_kw,
        state=args.state,
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_verification_advanced(analysis)
    output = buf.getvalue()
    print(output, end='')

    if 'FAIL' in output:
        print("OVERALL: FAIL", file=sys.stderr)
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())