#!/usr/bin/env python3
"""Add GST on spot imports to the bills of completed Amber runs.

Billing charges a household on a wholesale_exposed plan with
gst_on_wholesale_import: true GST on the spot energy charge of every import,
negative prices included (docs/model_spec.md, The settlement model). Runs
billed before that rule leave it out. Dispatch prices leave GST out under the
rule too, so the flows of those runs are what a new run would produce and only
their bills change: in bills.csv and every bills_MM.csv

    delta = GST_RATE * wholesale_usage_cost

is added to retail_usage_cost, household_total and retailer_margin. Every
other column is unchanged.

Per scenario, before anything is written, a sample of sites is re-billed from
flows.parquet with the core model's billing under the run's own tariffs, and
every bill column of the patched annual and monthly rows must match within
_ABS_TOL. flows.parquet is float32, so an exact match is not expected.

A marker, gst_patch.yaml, in each scenario folder records the md5 of every
bill file before and after the patch. It is written with status in_progress
before the first bill file is replaced and set to complete after the last.
A scenario with a complete marker is skipped. A scenario with an in_progress
marker (an interrupted write) is finished: each file whose md5 matches its
recorded post-patch value is left alone, each file matching its pre-patch
value is patched, and any other md5 stops that scenario with an error. No
file is patched twice.

Each run gets gst_patch_report.csv, one row per scenario, merged across
invocations.

Usage (from project root):
    python3 analysis/scripts/patch_amber_gst.py \\
        --data-root "/Volumes/Manual/Thesis Project" --workers 6

    # what would be done, without writing
    python3 analysis/scripts/patch_amber_gst.py \\
        --data-root "/Volumes/Manual/Thesis Project" --dry-run

    # one scenario, as a test
    python3 analysis/scripts/patch_amber_gst.py \\
        --data-root "/Volumes/Manual/Thesis Project" \\
        --runs full_run_no_cer_amber --only NSW_2022
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

INTERVAL_HOURS = 5 / 60
_ABS_TOL = 0.01   # $ per site per bill

DEFAULT_RUNS = (
    'full_run_no_cer_amber',
    'full_run_solar_only_amber',
    'full_run_self_consumption_amber',
    'full_run_optimised_amber',
    'full_run_rolling_amber',
)
MARKER = 'gst_patch.yaml'
REPORT = 'gst_patch_report.csv'
MONTH_FILES = [f'bills_{m:02d}.csv' for m in range(1, 13)]
PATCHED_COLS = ('retail_usage_cost', 'household_total', 'retailer_margin')
BILL_COLS = (
    'wholesale_usage_cost', 'wholesale_feedin_credit',
    'network_usage_cost', 'network_feedin_credit', 'network_daily_charges',
    'retail_usage_cost', 'retail_feedin_credit', 'retail_daily_charges',
    'retail_other_credits', 'household_total', 'retailer_margin',
)
REPORT_COLS = ['run_name', 'scenario_id', 'status', 'n_sites', 'n_files',
               'n_verified_sites', 'max_verify_diff',
               'mean_gst_per_site', 'message']


class PatchError(RuntimeError):
    """A scenario that cannot be patched safely; nothing more is written."""


# ----------------------------------------------------------------------
# Pure pieces
# ----------------------------------------------------------------------

def read_bills(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={'site_id': str})


def bills_bytes(df: pd.DataFrame) -> bytes:
    """The bytes a bill file is written as. Deterministic, so the md5 of a
    patched file can be predicted before it is written."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def patch_bills(df: pd.DataFrame, gst_rate: float) -> pd.DataFrame:
    """Add gst_rate x wholesale_usage_cost to PATCHED_COLS, row by row."""
    out = df.copy()
    delta = gst_rate * df['wholesale_usage_cost'].astype(float)
    for col in PATCHED_COLS:
        out[col] = df[col].astype(float) + delta
    return out


def bill_files(scn_dir: Path) -> list[str]:
    files = ['bills.csv'] + [f for f in MONTH_FILES if (scn_dir / f).exists()]
    if not (scn_dir / 'bills.csv').exists():
        raise PatchError(f'no bills.csv in {scn_dir}')
    return files


def sample_sites(sites: list[str], n: int) -> list[str]:
    """n sites spread evenly over the sorted site list."""
    ordered = sorted(sites)
    if n >= len(ordered):
        return ordered
    pick = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return [ordered[i] for i in sorted(set(pick))]


def max_bill_diff(patched: dict[str, pd.DataFrame],
                  reference: dict[str, pd.DataFrame]) -> float:
    """Largest absolute difference over BILL_COLS between the patched rows and
    the reference rows of the reference's sites. Raises PatchError when a
    reference file or site has no patched counterpart."""
    worst = 0.0
    for f, ref in reference.items():
        if f not in patched:
            raise PatchError(f'reference bill file {f} has no patched file')
        got = patched[f].set_index('site_id')
        ref = ref.set_index('site_id')
        missing = set(ref.index) - set(got.index)
        if missing:
            raise PatchError(f'{f}: sites {sorted(missing)} not in the bills')
        diff = (got.loc[ref.index, list(BILL_COLS)].astype(float)
                - ref[list(BILL_COLS)].astype(float)).abs().to_numpy()
        worst = max(worst, float(diff.max()))
    return worst


# ----------------------------------------------------------------------
# One scenario
# ----------------------------------------------------------------------

def patch_scenario(scn_dir: Path, gst_rate: float,
                   reference_bills: Callable[[list[str]], dict[str, pd.DataFrame]],
                   provenance: dict, dry_run: bool = False) -> dict:
    """Patch one scenario folder in place. Returns a report dict.

    reference_bills(files) returns, for the listed bill files, the bills of a
    sample of sites recomputed by the core model's billing. They are compared
    with the patched rows before anything is written.
    """
    report = {'status': 'FAIL', 'n_sites': 0, 'n_files': 0,
              'n_verified_sites': 0, 'max_verify_diff': np.nan,
              'mean_gst_per_site': np.nan, 'message': ''}
    marker_path = scn_dir / MARKER
    marker = None
    if marker_path.exists():
        marker = yaml.safe_load(marker_path.read_text())
        if marker.get('status') == 'complete':
            report.update(status='SKIP', message='already patched')
            return report
        if marker.get('status') != 'in_progress':
            raise PatchError(f'{marker_path}: unknown status '
                             f'{marker.get("status")!r}')

    files = bill_files(scn_dir)
    if marker is not None and sorted(marker['files']) != sorted(files):
        raise PatchError(f'{marker_path}: marker lists {sorted(marker["files"])}, '
                         f'folder has {sorted(files)}')

    current_bytes = {f: (scn_dir / f).read_bytes() for f in files}
    state = {}
    for f in files:
        h = md5(current_bytes[f])
        if marker is None:
            state[f] = 'pre'
        elif h == marker['files'][f]['post_md5']:
            state[f] = 'post'
        elif h == marker['files'][f]['pre_md5']:
            state[f] = 'pre'
        else:
            raise PatchError(f'{scn_dir / f} matches neither its recorded '
                             f'pre- nor post-patch md5; it changed after the '
                             f'interrupted patch')

    patched, new_bytes = {}, {}
    for f in files:
        df = read_bills(scn_dir / f)
        patched[f] = patch_bills(df, gst_rate) if state[f] == 'pre' else df
        new_bytes[f] = (bills_bytes(patched[f]) if state[f] == 'pre'
                        else current_bytes[f])
        if marker is not None and md5(new_bytes[f]) != marker['files'][f]['post_md5']:
            raise PatchError(f'{scn_dir / f}: re-patched content does not '
                             f'match the md5 recorded in {MARKER}')

    annual = patched['bills.csv']
    report['n_sites'] = len(annual)
    report['n_files'] = len(files)
    if not (annual['plan_type'] == 'wholesale_exposed').all():
        raise PatchError(f'{scn_dir}: plan_type is not wholesale_exposed')

    reference = reference_bills(files)
    report['n_verified_sites'] = len(reference['bills.csv'])
    worst = max_bill_diff(patched, reference)
    report['max_verify_diff'] = worst
    if marker is None:
        before = read_bills(scn_dir / 'bills.csv')['household_total'].astype(float)
    else:
        before = pd.Series(marker['mean_household_total_before'], index=[0])
    report['mean_gst_per_site'] = float(annual['household_total'].astype(float).mean()
                                        - before.mean())
    if worst >= _ABS_TOL:
        report['message'] = (f'patched bills differ from core billing by '
                             f'${worst:.4f} on the sampled sites')
        return report

    if dry_run:
        report.update(status='DRY', message=f'would patch '
                      f'{sum(s == "pre" for s in state.values())} files')
        return report

    if marker is None:
        marker = dict(provenance)
        marker.update(
            status='in_progress', gst_rate=gst_rate,
            mean_household_total_before=float(before.mean()),
            files={f: {'pre_md5': md5(current_bytes[f]),
                       'post_md5': md5(new_bytes[f])} for f in files})
        _write_atomic(marker_path, yaml.safe_dump(marker, sort_keys=False).encode())

    for f in files:
        if state[f] == 'pre':
            _write_atomic(scn_dir / f, new_bytes[f])

    marker.update(status='complete',
                  completed=dt.datetime.now().isoformat(timespec='seconds'),
                  mean_household_total_after=float(
                      annual['household_total'].astype(float).mean()),
                  n_verified_sites=report['n_verified_sites'],
                  max_verify_diff=worst)
    _write_atomic(marker_path, yaml.safe_dump(marker, sort_keys=False).encode())
    report.update(status='PASS',
                  message='resumed' if 'post' in state.values() else '')
    return report


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + '.gstpatch.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ----------------------------------------------------------------------
# Reference bills from the core model
# ----------------------------------------------------------------------

def core_reference_bills(flows: pd.DataFrame, prices: pd.Series,
                         network, retail, state: str,
                         files: list[str]) -> dict[str, pd.DataFrame]:
    """Bill every site column of `flows` with the core model's billing and
    return the requested bill files' rows. `flows` holds net kWh per interval,
    import positive, one column per site."""
    from model.core.site_result import SiteTraces
    from model.components.billing import compute_bill_and_monthly_bills
    from model.orchestrate import _bill_row

    net = flows.astype(float)
    zeros = pd.Series(0.0, index=net.index)
    rows: dict[str, list] = {f: [] for f in files}
    for site in net.columns:
        traces = SiteTraces(
            site_id=str(site), import_kwh=net[site].clip(lower=0),
            export_kwh=(-net[site]).clip(lower=0),
            battery_soc_kwh=zeros, sol_to_home_kwh=zeros,
            sol_to_bat_kwh=zeros, sol_to_grid_kwh=zeros,
            bat_to_home_kwh=zeros, grid_to_home_kwh=zeros,
            grid_to_bat_kwh=zeros, bat_to_grid_kwh=zeros,
            curtail_kwh=zeros,
        )
        annual, monthly = compute_bill_and_monthly_bills(
            traces, prices, network, retail, state, INTERVAL_HOURS)
        rows['bills.csv'].append(_bill_row(annual))
        for m, b in monthly.items():
            f = f'bills_{m:02d}.csv'
            if f in rows:
                rows[f].append(_bill_row(b))
    out = {}
    for f, r in rows.items():
        df = pd.DataFrame(r)
        df['site_id'] = df['site_id'].astype(str)
        out[f] = df
    return out


def _scenario_task(task: dict) -> dict:
    scenario = task['scenario']
    row = {'run_name': task['run_name'], 'scenario_id': scenario.scenario_id}
    try:
        sys.path.insert(0, str(task['project_root']))
        from model.core import Paths
        from model.core.tariff import load_network_tariff, load_retail_tariff
        from model.components.ingest import _load_prices
        from model.components.billing import GST_RATE

        scn_dir: Path = task['scn_dir']
        tariffs_dir = task['project_root'] / 'model' / 'tariffs'
        network = load_network_tariff(scenario.network_tariff_id, tariffs_dir)
        retail = load_retail_tariff(scenario.retail_tariff_id, tariffs_dir)
        if not retail.gst_on_wholesale_import:
            raise PatchError(f'{scenario.retail_tariff_id} does not set '
                             f'gst_on_wholesale_import')

        def reference(files: list[str]) -> dict[str, pd.DataFrame]:
            sites = list(read_bills(scn_dir / 'bills.csv')['site_id'])
            pick = sample_sites(sites, task['verify_sites'])
            flows = pd.read_parquet(scn_dir / 'flows.parquet', columns=pick)
            flows.columns = [str(c) for c in flows.columns]
            data_root: Path = task['data_root']
            paths = Paths(
                edp_clean_dir=data_root / 'data' / 'clean' / 'edp',
                edp_metadata_dir=data_root / 'data' / 'processed' / 'edp',
                nem_prices_dir=data_root / 'data' / 'clean' / 'nem_prices',
                tariffs_dir=tariffs_dir,
            )
            prices = _load_prices(scenario, paths)
            # flows.parquet can carry a pytz FixedOffset(600) label for the
            # same instants as the price index.
            if (len(prices) != len(flows)
                    or not (prices.index.asi8
                            == flows.index.tz_convert(prices.index.tz).asi8).all()):
                raise PatchError('price index does not match flows index')
            flows.index = prices.index
            return core_reference_bills(flows, prices, network, retail,
                                        scenario.state, files)

        rep = patch_scenario(scn_dir, GST_RATE, reference,
                             task['provenance'] | {
                                 'retail_tariff_id': scenario.retail_tariff_id,
                                 'network_tariff_id': scenario.network_tariff_id},
                             dry_run=task['dry_run'])
        row.update(rep)
    except Exception as e:   # report and carry on with other scenarios
        row.update({k: np.nan for k in REPORT_COLS if k not in row})
        row.update(status='FAIL', message=f'{type(e).__name__}: {e}')
    return row


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, required=True,
                   help="Data root: runs and NEM prices are under "
                        "<data-root>/data/.")
    p.add_argument('--project-root', type=Path, default=Path.cwd(),
                   help="Code repo root; run configs and tariffs are read "
                        "from here. Default: cwd.")
    p.add_argument('--runs', nargs='+', default=list(DEFAULT_RUNS),
                   help="Runs to patch. Default: the five Amber full runs.")
    p.add_argument('--only', nargs='+', default=None, metavar='SCENARIO_ID',
                   help="Restrict to these scenario ids.")
    p.add_argument('--verify-sites', type=int, default=5,
                   help="Sites per scenario re-billed from flows. Default 5.")
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--dry-run', action='store_true',
                   help="Verify and report; write nothing.")
    return p.parse_args(argv)


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


def _merge_report(path: Path, rows: list[dict]) -> pd.DataFrame:
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

    runs_root = data_root / 'data' / 'output' / 'runs'
    provenance = {'script': 'analysis/scripts/patch_amber_gst.py',
                  'git_commit': _git_commit(project_root),
                  'written': dt.datetime.now().isoformat(timespec='seconds')}
    tasks = []
    for run_name in args.runs:
        config_path = project_root / 'model' / 'configs' / f'{run_name}.yaml'
        name, scenarios = load_scenarios(config_path)
        if name != run_name:
            print(f"{config_path} names its run {name!r}, expected "
                  f"{run_name!r}.", file=sys.stderr)
            return 2
        if args.only:
            scenarios = [s for s in scenarios if s.scenario_id in set(args.only)]
        n_missing = 0
        for s in scenarios:
            scn_dir = runs_root / run_name / 'scenarios' / s.scenario_id
            if not ((scn_dir / 'bills.csv').exists()
                    and (scn_dir / 'flows.parquet').exists()):
                n_missing += 1
            tasks.append(dict(scenario=s, scn_dir=scn_dir, run_name=run_name,
                              data_root=data_root, project_root=project_root,
                              verify_sites=args.verify_sites,
                              dry_run=args.dry_run,
                              provenance=provenance | {'run_name': run_name}))
        print(f"{run_name}: {len(scenarios)} scenarios, {n_missing} missing "
              f"bills.csv or flows.parquet")

    if not tasks:
        print("Nothing to do.")
        return 0
    print(f"\n{'Checking' if args.dry_run else 'Patching'} {len(tasks)} "
          f"scenarios with {args.workers} worker(s)")
    rows = []

    def _log(r):
        print(f"  {r['run_name']:<34} {r['scenario_id']:<16} {r['status']:<5} "
              f"max diff {r['max_verify_diff']:.2e}  GST/site "
              f"{r['mean_gst_per_site']:.2f}  {r['message']}", flush=True)
        rows.append(r)

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_scenario_task, tasks):
                _log(r)
    else:
        for t in tasks:
            _log(_scenario_task(t))

    n_fail = sum(r['status'] == 'FAIL' for r in rows)
    if not args.dry_run:
        for run_name in dict.fromkeys(r['run_name'] for r in rows):
            run_rows = [r for r in rows if r['run_name'] == run_name]
            rep = _merge_report(runs_root / run_name / REPORT, run_rows)
            print(f"\n{run_name}: report has {len(rep)} scenarios, "
                  f"{int((rep['status'] == 'PASS').sum())} PASS, "
                  f"{int((rep['status'] == 'SKIP').sum())} SKIP, "
                  f"{int((rep['status'] == 'FAIL').sum())} FAIL")
    print(f"\n{len(rows) - n_fail}/{len(rows)} scenarios without failure")
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
