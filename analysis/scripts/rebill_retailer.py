#!/usr/bin/env python3
"""Re-bill completed runs under another retailer's tariff, keeping dispatch.

Reads each scenario's flows.parquet from a source run (full_run_{mode}_{source})
and bills every site under the target retail tariff with the core model's
billing (compute_bill_and_monthly_bills). Network tariff, prices and flows are
those of the source run. Writes a new run, full_run_{mode}_{target}, holding
bills only. The rolling_predispatchonly mode names its runs
full_run_rolling_{source}_predispatchonly and
full_run_rolling_{target}_predispatchonly (the source dispatches on
pre-dispatch forecasts only, current_interval_price: forecast).

    data/output/runs/full_run_{mode}_{target}/
      scenarios/{scenario_id}/
        bills.csv
        bills_01.csv .. bills_12.csv   # only where the source scenario has them
        summary.txt
      rebill_source.yaml               # source run, config, tariff ids, commit
      rebill_report.csv                # one row per scenario, with check results

No flows are copied. financial_analysis.py and retailer_briefing.py read only
bills.csv, so the new run is used exactly as a modelled run would be.

Dispatch is the source run's. For no CER, solar only and self consumption,
dispatch does not depend on the tariff, so the re-bill is exact. For optimised and
rolling it is an approximation: the source run optimised against its own
plan's prices, while a mixed plan's retailer dispatches against the
mixed-plan dispatch prices (docs/model_spec.md). A modelled run of the
target plan supersedes the re-billed one under the same run name.

Checks, per scenario; a scenario that fails any of them is reported and not
written:

  1. flows.parquet holds the same site set as the source bills.csv.
  2. wholesale_* and network_* columns match the source bills within _ABS_TOL,
     annual and monthly. These depend only on flows, prices and the network
     tariff, so a match confirms the re-bill used what the source was billed
     with. flows.parquet is float32, so an exact match is not expected.
  3. plan_type is the target tariff's type.
  4. Monthly bills sum to the annual bill, where monthly bills are written.

Existing target scenarios are skipped unless --overwrite, so an interrupted
batch resumes where it stopped.

Usage (from project root):
    python3 analysis/scripts/rebill_retailer.py \\
        --data-root "/Volumes/Manual/Thesis Project" --workers 6

    # one scenario, as a test
    python3 analysis/scripts/rebill_retailer.py \\
        --data-root "/Volumes/Manual/Thesis Project" \\
        --modes rolling --only NSW_2022_20kWh

    # Tesla equivalent of full_run_rolling_amber_predispatchonly
    python3 analysis/scripts/rebill_retailer.py \\
        --data-root "/Volumes/Manual/Thesis Project" \\
        --modes rolling_predispatchonly --workers 6

    # list what would be done and what is missing, without billing
    python3 analysis/scripts/rebill_retailer.py \\
        --data-root "/Volumes/Manual/Thesis Project" --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

INTERVAL_HOURS = 5 / 60
_ABS_TOL = 0.01   # $ per site per bill

MODES = ('no_cer', 'solar_only', 'self_consumption', 'optimised', 'rolling',
         'rolling_predispatchonly')
DEFAULT_MODES = ('solar_only', 'self_consumption', 'optimised', 'rolling',
                 'rolling_predispatchonly')
# Modes whose run name carries a suffix after the retailer:
# full_run_{base}_{retailer}{suffix}.
_MODE_SUFFIX = {'rolling_predispatchonly': ('rolling', '_predispatchonly')}


def mode_run_name(mode: str, retailer: str) -> str:
    """Run name for a mode and retailer suffix, e.g. 'rolling_predispatchonly'
    and 'tesla' -> 'full_run_rolling_tesla_predispatchonly'."""
    base, suffix = _MODE_SUFFIX.get(mode, (mode, ''))
    return f'full_run_{base}_{retailer}{suffix}'

CHECKED_COLS = [
    'wholesale_usage_cost', 'wholesale_feedin_credit',
    'network_usage_cost', 'network_feedin_credit', 'network_daily_charges',
]
NUMERIC_COLS = CHECKED_COLS + [
    'retail_usage_cost', 'retail_feedin_credit', 'retail_daily_charges',
    'retail_other_credits', 'household_total', 'retailer_margin',
]
MONTH_FILES = [f'bills_{m:02d}.csv' for m in range(1, 13)]
REPORT_COLS = ['run_name', 'scenario_id', 'status', 'n_sites',
               'max_checked_diff', 'max_month_sum_diff',
               'mean_source_household_total', 'mean_household_total',
               'message']


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, required=True,
                   help="Data root: source runs, NEM prices and the new runs "
                        "are under <data-root>/data/.")
    p.add_argument('--project-root', type=Path, default=Path.cwd(),
                   help="Code repo root; run configs and tariffs are read "
                        "from here. Default: cwd.")
    p.add_argument('--modes', nargs='+', choices=MODES,
                   default=list(DEFAULT_MODES),
                   help="Runs to re-bill. Default: all but no_cer.")
    p.add_argument('--source', default='amber',
                   help="Source run suffix: full_run_{mode}_{source}. "
                        "Default: amber.")
    p.add_argument('--target', default='tesla',
                   help="Target run suffix: full_run_{mode}_{target}. "
                        "Default: tesla.")
    p.add_argument('--retail-tariff-id',
                   default='{state}_tesla_electric_mixed_2026',
                   help="Target retail tariff, formatted with {state} and "
                        "{year}. Default: '{state}_tesla_electric_mixed_2026'.")
    p.add_argument('--out-root', type=Path, default=None,
                   help="Where the new runs are written. Default: "
                        "<data-root>/data/output/runs.")
    p.add_argument('--only', nargs='+', default=None, metavar='SCENARIO_ID',
                   help="Restrict to these scenario ids, e.g. NSW_2022_20kWh "
                        "or NSW_2022 for solar only.")
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--overwrite', action='store_true',
                   help="Re-bill scenarios already present in the target run.")
    p.add_argument('--dry-run', action='store_true',
                   help="List scenarios and missing inputs; bill nothing.")
    return p.parse_args(argv)


def _read_bills(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['site_id'] = df['site_id'].astype(str)
    return df.set_index('site_id')


def _git_commit(project_root: Path) -> str:
    try:
        out = subprocess.run(
            ['git', '-C', str(project_root), 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ['git', '-C', str(project_root), 'status', '--porcelain',
             '--untracked-files=no'],
            capture_output=True, text=True, check=True).stdout.strip()
        return f'{out}{"-dirty" if dirty else ""}'
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _rebill_scenario(task: dict) -> dict:
    """Re-bill one scenario. Writes the target scenario folder only if every
    check passes; returns a report row either way."""
    scenario = task['scenario']
    sid = scenario.scenario_id
    report = {'run_name': task['target_run'], 'scenario_id': sid,
              'status': 'FAIL', 'n_sites': 0, 'max_checked_diff': np.nan,
              'max_month_sum_diff': np.nan,
              'mean_source_household_total': np.nan,
              'mean_household_total': np.nan, 'message': ''}
    try:
        sys.path.insert(0, str(task['project_root']))
        from model.core import Paths
        from model.core.tariff import load_network_tariff, load_retail_tariff
        from model.core.site_result import SiteTraces
        from model.components.ingest import _load_prices
        from model.components.billing import compute_bill_and_monthly_bills
        from model.orchestrate import _bill_row

        src_dir: Path = task['src_dir']
        dst_dir: Path = task['dst_dir']
        flows_path = src_dir / 'flows.parquet'
        if not (src_dir / 'bills.csv').exists() or not flows_path.exists():
            report['message'] = 'missing flows.parquet or bills.csv in source'
            return report
        old = {'bills.csv': _read_bills(src_dir / 'bills.csv')}
        for f in MONTH_FILES:
            if (src_dir / f).exists():
                old[f] = _read_bills(src_dir / f)
        with_monthly = len(old) > 1

        flows = pd.read_parquet(flows_path)
        flows.columns = [str(c) for c in flows.columns]
        sites = list(old['bills.csv'].index)
        if set(sites) != set(flows.columns):
            report['message'] = (f'site mismatch: bills {len(sites)}, '
                                 f'flows {len(flows.columns)}')
            return report

        data_root: Path = task['data_root']
        paths = Paths(
            edp_clean_dir=data_root / 'data' / 'clean' / 'edp',
            edp_metadata_dir=data_root / 'data' / 'processed' / 'edp',
            nem_prices_dir=data_root / 'data' / 'clean' / 'nem_prices',
            tariffs_dir=task['project_root'] / 'model' / 'tariffs',
        )
        prices = _load_prices(scenario, paths)
        # flows.parquet can come back with a pytz FixedOffset(600) timezone
        # rather than Etc/GMT-10: same instants, different label. Compare
        # instants, then bill on the price index.
        index = flows.index
        if (len(prices) != len(index)
                or not (prices.index.asi8
                        == index.tz_convert(prices.index.tz).asi8).all()):
            report['message'] = 'price index does not match flows index'
            return report
        net = flows.astype(float)
        net.index = prices.index
        imp_df = net.clip(lower=0)
        exp_df = (-net).clip(lower=0)

        network = load_network_tariff(scenario.network_tariff_id,
                                      paths.tariffs_dir)
        retail = load_retail_tariff(task['target_tariff_id'],
                                    paths.tariffs_dir)

        zeros = pd.Series(0.0, index=prices.index)
        new_rows: dict[str, list] = {'bills.csv': []}
        for site in sites:
            traces = SiteTraces(
                site_id=site, import_kwh=imp_df[site], export_kwh=exp_df[site],
                battery_soc_kwh=zeros, sol_to_home_kwh=zeros,
                sol_to_bat_kwh=zeros, sol_to_grid_kwh=zeros,
                bat_to_home_kwh=zeros, grid_to_home_kwh=zeros,
                grid_to_bat_kwh=zeros, bat_to_grid_kwh=zeros,
                curtail_kwh=zeros,
            )
            annual, monthly = compute_bill_and_monthly_bills(
                traces, prices, network, retail, scenario.state,
                INTERVAL_HOURS)
            new_rows['bills.csv'].append(_bill_row(annual))
            if with_monthly:
                for m, b in monthly.items():
                    new_rows.setdefault(f'bills_{m:02d}.csv', []).append(
                        _bill_row(b))

        new = {}
        for f, rows in new_rows.items():
            df = pd.DataFrame(rows)
            df['site_id'] = df['site_id'].astype(str)
            new[f] = df.set_index('site_id')
        if set(new) != set(old):
            report['message'] = (f'bill file sets differ: new {sorted(new)} '
                                 f'source {sorted(old)}')
            return report

        max_checked = 0.0
        for f, o in old.items():
            n = new[f]
            if set(o.index) != set(n.index):
                report['message'] = f'{f}: site mismatch'
                return report
            n = n.loc[o.index]
            new[f] = n
            if not (n['plan_type'] == retail.type).all():
                report['message'] = f'{f}: plan_type is not {retail.type!r}'
                return report
            max_checked = max(max_checked, float(
                (n[CHECKED_COLS] - o[CHECKED_COLS]).abs().to_numpy().max()))

        max_msum = 0.0
        if with_monthly:
            month_sum = sum(new[f][NUMERIC_COLS] for f in new
                            if f != 'bills.csv')
            max_msum = float((month_sum - new['bills.csv'][NUMERIC_COLS])
                             .abs().to_numpy().max())

        report.update(
            n_sites=len(sites), max_checked_diff=max_checked,
            max_month_sum_diff=max_msum,
            mean_source_household_total=float(
                old['bills.csv']['household_total'].mean()),
            mean_household_total=float(new['bills.csv']['household_total']
                                       .mean()),
        )
        if max_checked >= _ABS_TOL or max_msum >= _ABS_TOL:
            report['message'] = 'tolerance exceeded (see max_* columns)'
            return report

        # Write to a sibling temp folder, then swap in, so an interrupted
        # write never leaves a partial scenario that looks complete.
        tmp_dir = dst_dir.with_name(dst_dir.name + '.tmp')
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        for f, df in new.items():
            df.reset_index().to_csv(tmp_dir / f, index=False)
        (tmp_dir / 'summary.txt').write_text(
            f"Re-billed scenario: {sid}\n"
            f"Source: {task['source_run']}/scenarios/{sid}\n"
            f"Retail tariff: {task['target_tariff_id']} ({retail.name}, "
            f"type {retail.type})\n"
            f"Network tariff: {scenario.network_tariff_id}\n"
            f"Sites: {len(sites)}\n"
            f"Max wholesale/network diff against source: ${max_checked:.2e}\n"
            f"Mean household_total: source ${report['mean_source_household_total']:,.2f}, "
            f"re-billed ${report['mean_household_total']:,.2f}\n")
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        tmp_dir.rename(dst_dir)
        report['status'] = 'PASS'
        return report
    except Exception as e:  # report and carry on with other scenarios
        report['message'] = f'{type(e).__name__}: {e}'
        return report


def _write_source_yaml(out_dir: Path, source_run: str, config_path: Path,
                       tariff_ids: dict, commit: str) -> None:
    """Provenance for the target run. Rewritten on every invocation; the
    report carries per-scenario history."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        'description': 'Bills only, re-billed from the source run flows under '
                       'the retail tariff below. Dispatch, prices and network '
                       'tariff are the source run\'s.',
        'source_run': source_run,
        'source_config': str(config_path),
        'retail_tariff_ids': tariff_ids,
        'script': 'analysis/scripts/rebill_retailer.py',
        'git_commit': commit,
        'written': dt.datetime.now().isoformat(timespec='seconds'),
    }
    (out_dir / 'rebill_source.yaml').write_text(
        yaml.safe_dump(doc, sort_keys=False))


def _merge_report(path: Path, rows: list[dict]) -> pd.DataFrame:
    """Merge this invocation's rows into an existing report, replacing rows
    for scenarios re-billed now."""
    new = pd.DataFrame(rows, columns=REPORT_COLS)
    if path.exists():
        prev = pd.read_csv(path)
        prev = prev[~prev['scenario_id'].isin(new['scenario_id'])]
        new = pd.concat([prev, new], ignore_index=True)
    new = new.sort_values('scenario_id').reset_index(drop=True)
    new.to_csv(path, index=False)
    return new


def main(argv=None) -> int:
    args = _parse_args(argv)
    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    sys.path.insert(0, str(project_root))
    from model.main import load_scenarios
    from model.core.tariff import load_retail_tariff

    if args.source == args.target:
        print("--source and --target are the same; refusing to overwrite the "
              "source run.", file=sys.stderr)
        return 2
    runs_root = data_root / 'data' / 'output' / 'runs'
    out_root = (args.out_root or runs_root).resolve()
    tariffs_dir = project_root / 'model' / 'tariffs'
    commit = _git_commit(project_root)

    plans = []   # (source_run, target_run, config_path, scenarios, tariff_ids)
    for mode in args.modes:
        source_run = mode_run_name(mode, args.source)
        target_run = mode_run_name(mode, args.target)
        config_path = project_root / 'model' / 'configs' / f'{source_run}.yaml'
        run_name, scenarios = load_scenarios(config_path)
        if run_name != source_run:
            print(f"{config_path} names its run {run_name!r}, expected "
                  f"{source_run!r}.", file=sys.stderr)
            return 2
        if args.only:
            scenarios = [s for s in scenarios if s.scenario_id in set(args.only)]
        # Resolve every target tariff before any billing, so a missing
        # tariff file fails the batch at the start rather than per scenario.
        tariff_ids = {}
        for s in scenarios:
            tid = args.retail_tariff_id.format(state=s.state, year=s.year)
            tariff_ids.setdefault(s.state, {})[s.year] = tid
        for tid in sorted({t for by_year in tariff_ids.values()
                           for t in by_year.values()}):
            load_retail_tariff(tid, tariffs_dir)
        plans.append((source_run, target_run, config_path, scenarios,
                      tariff_ids))

    tasks = []
    for source_run, target_run, config_path, scenarios, tariff_ids in plans:
        n_skip = n_missing = 0
        for s in scenarios:
            src_dir = runs_root / source_run / 'scenarios' / s.scenario_id
            dst_dir = out_root / target_run / 'scenarios' / s.scenario_id
            if (dst_dir / 'bills.csv').exists() and not args.overwrite:
                n_skip += 1
                continue
            if not ((src_dir / 'flows.parquet').exists()
                    and (src_dir / 'bills.csv').exists()):
                n_missing += 1
                if args.dry_run:
                    print(f"  missing input: {src_dir}")
            tasks.append(dict(
                scenario=s, src_dir=src_dir, dst_dir=dst_dir,
                source_run=source_run, target_run=target_run,
                target_tariff_id=tariff_ids[s.state][s.year],
                data_root=data_root, project_root=project_root))
        print(f"{source_run} -> {target_run}: {len(scenarios)} scenarios, "
              f"{n_skip} already done, {n_missing} missing flows or bills")
        if not args.dry_run and scenarios:
            _write_source_yaml(out_root / target_run, source_run,
                               config_path.relative_to(project_root),
                               tariff_ids, commit)

    if args.dry_run:
        print(f"\nDry run: {len(tasks)} scenarios would be re-billed.")
        return 0
    if not tasks:
        print("Nothing to do.")
        return 0

    print(f"\nRe-billing {len(tasks)} scenarios with {args.workers} worker(s)")
    rows = []

    def _log(r):
        print(f"  {r['run_name']:<34} {r['scenario_id']:<16} {r['status']}  "
              f"{r['message']}", flush=True)
        rows.append(r)

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_rebill_scenario, tasks):
                _log(r)
    else:
        for t in tasks:
            _log(_rebill_scenario(t))

    n_fail = 0
    for target_run in dict.fromkeys(r['run_name'] for r in rows):
        run_rows = [r for r in rows if r['run_name'] == target_run]
        rep = _merge_report(out_root / target_run / 'rebill_report.csv',
                            run_rows)
        run_fail = sum(r['status'] != 'PASS' for r in run_rows)
        n_fail += run_fail
        print(f"\n{target_run}: {len(run_rows) - run_fail}/{len(run_rows)} "
              f"PASS this invocation; report has {len(rep)} scenarios, "
              f"{int((rep['status'] == 'PASS').sum())} PASS")
        if len(run_rows) > run_fail:
            ok = pd.DataFrame([r for r in run_rows if r['status'] == 'PASS'])
            print(f"  max wholesale/network diff ${ok['max_checked_diff'].max():.2e}, "
                  f"max month-sum diff ${ok['max_month_sum_diff'].max():.2e}")
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
