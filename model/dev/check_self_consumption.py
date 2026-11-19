"""Non-interactive self_consumption verification on a real site file.

Usage (from project root):
    python3 -m model.dev.check_self_consumption SITE_ID YEAR
    python3 -m model.dev.check_self_consumption SITE_ID YEAR --no-battery

Loads a real site trace and runs self_consumption against it (with the
hardcoded DEV_BATTERY from viz_self_consumption, or no battery). Prints
the verification stats to stdout and exits with code 0 if all checks
pass, 1 otherwise.

Useful as a quick sanity check after touching self_consumption.py.
"""
import argparse
import io
import sys
from pathlib import Path
from contextlib import redirect_stdout

from model.dev.viz_self_consumption import (
    DEV_BATTERY,
    _load_site_trace,
    _print_verification,
)
from model.core.paths import default_paths
from model.components.self_consumption import self_consumption


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site_id')
    parser.add_argument('year', type=int)
    parser.add_argument('--no-battery', action='store_true')
    parser.add_argument('--project-root', type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace = _load_site_trace(args.site_id, args.year, paths)

    battery_spec = None if args.no_battery else DEV_BATTERY

    result = self_consumption(
        site_id=args.site_id,
        solar_kwh=trace['pv_site_net_kwh'],
        load_kwh=trace['gross_load_kwh'],
        battery_spec=battery_spec,
    )

    # Capture output to check for FAIL markers
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_verification(
            site_id=args.site_id,
            year=args.year,
            trace=trace,
            result=result,
            battery_present=battery_spec is not None,
            battery_spec=battery_spec,
        )
    output = buf.getvalue()
    print(output, end='')

    if 'FAIL' in output:
        print("OVERALL: FAIL", file=sys.stderr)
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
