"""Measured run times from existing scenarios, for sizing the sensitivity runs.

Reads every summary.txt under --data-root and reports, per run and scenario,
the site count, wall-clock elapsed, seconds per site and LP solve CPU seconds
per site. CPU seconds per site is the figure that transfers between machines
and worker counts; wall time for a new run is approximately

    sites * cpu_seconds_per_site / workers.

With --inventory it also reports which of the runs the sensitivity sets
reference are present for a given state and year, and at which battery sizes.

Usage (from project root):
    python3 analysis/scripts/sensitivity/run_times.py --data-root "/Volumes/Manual/Thesis Project"
    python3 analysis/scripts/sensitivity/run_times.py --data-root ... --runs full_run_rolling_amber --sizes 20,50
    python3 analysis/scripts/sensitivity/run_times.py --data-root ... --inventory
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# Runs the four sensitivity sets read, as references or as parents.
REFERENCE_RUNS = [
    'full_run_rolling_amber',
    'full_run_optimised_amber',
    'full_run_self_consumption_amber',
    'full_run_solar_only_amber',
    'full_run_optimised_origin_lite',
    '10hr_full_run_optimised_origin_lite',
    'full_run_self_consumption_origin_novpp',
    'full_run_solar_only_origin_starter',
    'full_run_rolling_origin_novpp',
    'full_run_rolling_ea_novpp',
    'full_run_self_consumption_ea',
    'full_run_solar_only_ea',
]

BATTERY_SIZES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

_DURATION = re.compile(r'(?:(\d+)h)?(?:(\d+)m)?([\d.]+)s')


def parse_duration(text: str) -> float | None:
    """Seconds from the summary's h/m/s form, e.g. '1h22m23.7s'."""
    m = _DURATION.fullmatch(text.strip())
    if not m:
        return None
    h, mins, secs = m.groups()
    return (int(h or 0) * 3600) + (int(mins or 0) * 60) + float(secs)


def read_summary(path: Path) -> dict | None:
    """Site count, elapsed, s/site and LP CPU s/site from one summary.txt."""
    text = path.read_text(errors='replace')

    def find(pattern: str) -> str | None:
        m = re.search(pattern, text)
        return m.group(1) if m else None

    sites = find(r'sites=(\d+)')
    elapsed = find(r'elapsed=(\S+)')
    if sites is None or elapsed is None:
        return None

    # Two historical shapes: 'LP solve=43m07.9s total (258.79s/site avg, ...'
    # and 'LP solve=15.8s  (15.80s/site)'. Both carry s/site.
    lp_per_site = find(r'LP solve=.*?\(([\d.]+)s/site')
    workers = find(r'LP solve=.*?,\s*(\d+) workers')

    return {
        'coordination_type': find(r'coordination_type=(\S+)') or '-',
        'sites': int(sites),
        'elapsed_s': parse_duration(elapsed),
        'per_site_s': float(find(r'elapsed=\S+\s+\(([\d.]+)s/site') or 'nan'),
        'lp_cpu_per_site_s': float(lp_per_site) if lp_per_site else None,
        'workers': int(workers) if workers else None,
    }


def scenario_key(name: str) -> tuple[str, str, int | None]:
    """('NSW', '2024', 20) from 'NSW_2024_20kWh'; size None when absent."""
    parts = name.split('_')
    size = None
    if len(parts) >= 3 and parts[2].endswith('kWh'):
        size = int(parts[2][:-3])
    return (parts[0], parts[1] if len(parts) > 1 else '', size)


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return '-'
    if value >= 3600:
        return f'{value / 3600:.1f}h'
    if value >= 60:
        return f'{value / 60:.1f}m'
    return f'{value:.0f}s'


def collect(runs_root: Path, runs: list[str] | None, state: str | None,
            year: str | None, sizes: list[int] | None) -> list[tuple]:
    rows = []
    for summary in sorted(runs_root.glob('*/scenarios/*/summary.txt')):
        run = summary.parts[-4]
        scenario = summary.parts[-2]
        if runs and run not in runs:
            continue
        st, yr, size = scenario_key(scenario)
        if state and st != state:
            continue
        if year and yr != year:
            continue
        if sizes and size not in sizes:
            continue
        data = read_summary(summary)
        if data:
            rows.append((run, scenario, data))
    return rows


def print_timings(rows: list[tuple]) -> None:
    if not rows:
        print('No matching scenarios found.')
        return
    header = (f"{'run':<42} {'scenario':<18} {'type':<38} {'sites':>5} "
              f"{'elapsed':>8} {'s/site':>8} {'LP cpu s/site':>14} {'workers':>8}")
    print(header)
    print('-' * len(header))
    for run, scenario, d in rows:
        lp = ('-' if d['lp_cpu_per_site_s'] is None
              else f"{d['lp_cpu_per_site_s']:.1f}")
        print(f"{run:<42} {scenario:<18} {d['coordination_type']:<38} "
              f"{d['sites']:>5} {fmt_seconds(d['elapsed_s']):>8} "
              f"{d['per_site_s']:>8.1f} {lp:>14} "
              f"{(d['workers'] if d['workers'] else '-'):>8}")


def print_inventory(runs_root: Path, state: str, year: str) -> None:
    print(f"\nReference runs present for {state} {year}")
    print('-' * 72)
    for run in REFERENCE_RUNS:
        scenarios = runs_root / run / 'scenarios'
        if not scenarios.exists():
            print(f'{run:<42} MISSING (no run directory)')
            continue
        plain = scenarios / f'{state}_{year}'
        sized = [s for s in BATTERY_SIZES
                 if (scenarios / f'{state}_{year}_{s}kWh' / 'bills.csv').exists()]
        if sized:
            missing = [s for s in BATTERY_SIZES if s not in sized]
            note = 'all sizes' if not missing else f"missing {missing}"
            print(f'{run:<42} {len(sized)} sizes ({note})')
        elif (plain / 'bills.csv').exists():
            print(f'{run:<42} present (no battery sizes)')
        else:
            print(f'{run:<42} MISSING for this state/year')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help='Project root holding data/output/runs '
                             '(default: current directory).')
    parser.add_argument('--runs', default=None,
                        help='Comma-separated run names to report '
                             '(default: the sensitivity reference runs).')
    parser.add_argument('--all-runs', action='store_true',
                        help='Report every run found, not just the '
                             'sensitivity reference runs.')
    parser.add_argument('--state', default='NSW')
    parser.add_argument('--year', default='2024')
    parser.add_argument('--sizes', default=None,
                        help='Comma-separated battery sizes in kWh '
                             '(default: all).')
    parser.add_argument('--inventory', action='store_true',
                        help='Also report which reference runs exist.')
    args = parser.parse_args(argv)

    runs_root = args.data_root / 'data' / 'output' / 'runs'
    if not runs_root.exists():
        parser.error(f'No runs directory at {runs_root}')

    if args.all_runs:
        runs = None
    elif args.runs:
        runs = args.runs.split(',')
    else:
        runs = REFERENCE_RUNS
    sizes = [int(s) for s in args.sizes.split(',')] if args.sizes else None

    rows = collect(runs_root, runs, args.state or None, args.year or None, sizes)
    print_timings(rows)

    if args.inventory:
        print_inventory(runs_root, args.state, args.year)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
