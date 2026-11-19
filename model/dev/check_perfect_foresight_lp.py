"""Non-interactive perfect_foresight_lp verification on a real site file.

Usage (from project root):
    python3 -m model.dev.check_perfect_foresight_lp SITE_ID YEAR
    python3 -m model.dev.check_perfect_foresight_lp SITE_ID YEAR \\
        --state NSW --retail-tariff amber_wholesale_2024 --export-limit-kw 10

Loads a real site trace + NEM prices + tariffs, runs self_consumption then
perfect_foresight_lp (with DEV_BATTERY, no degradation cost, a per-phase export
limit, SOC starting at the reserve floor and ending at the self_consumption
end-of-period SOC), prints verification stats, and exits 0 if all checks pass,
1 otherwise.

Useful as a quick sanity check after touching perfect_foresight_lp.py,
prices.py, or billing.py.
"""
import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

from model.core.paths import default_paths
from model.dev._dev_common import DEV_BATTERY
from model.dev.viz_perfect_foresight_lp import (
    load_inputs,
    run_lp_analysis,
    _print_verification,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id')
    parser.add_argument('year', type=int)
    parser.add_argument('--state', default='NSW')
    parser.add_argument('--network-tariff', default='ausgrid_ea025_network_2024')
    parser.add_argument('--retail-tariff', default='amber_wholesale_2024')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--project-root', type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, network_tariff, retail_tariff = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, paths,
    )

    analysis = run_lp_analysis(
        args.site_id, args.year, trace, prices_mwh,
        network_tariff, retail_tariff, DEV_BATTERY,
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