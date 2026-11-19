"""Bill reconciliation: prove household bills can be reconstituted from
already-modelled net meter flows.

For every (state, year, battery_name) scenario a run config YAML defines,
this script:

  1. Loads the site-level net import/export flows already written for
     that scenario (flows.parquet).
  2. Loads the same NEM price series and tariff pair the original run
     used (read straight out of the run config YAML, via
     model.main.load_scenarios, so there is no separate flag that could
     drift out of sync with what actually produced the flows).
  3. Reconstitutes each site's import_kwh / export_kwh by clipping the
     net flow at zero (exact at 5-minute NEM settlement granularity).
  4. Runs each site through the unmodified billing.compute_bill().
  5. Diffs the result against the scenario's existing bills.csv, per
     settlement-layer field, not just the household total.

Non-VPP coordination types only (perfect_foresight, perfect_foresight_advanced,
perfect_foresight_custom, self_consumption, self_hedging). VPP scenarios
(vpp_sc_event, vpp_tou_event) are skipped with a clear message: reconciling
them needs vpp_forced_flows.parquet and the VPP reward calculation on top of
the plain reconstitution.

Usage (from project root):
    python3 analysis/scripts/bill_reconciliation.py \\
        --run-config model/configs/full_run_optimised_amber.yaml \\
        --sample-sites 5

    python3 analysis/scripts/bill_reconciliation.py \\
        --run-config model/configs/full_run_optimised_amber.yaml \\
        --data-root "/Volumes/Manual/Thesis Project" \\
        --states NSW --years 2022 --battery-names 10kWh

    python3 analysis/scripts/bill_reconciliation.py --run-config model/configs/full_run_optimised_amber.yaml --data-root "/Volumes/Manual/Thesis Project" --states NSW --years 2022 --battery-names 10kWh --sample-sites 5

--sample-sites is for fast development iteration only. Omit it entirely
for any run whose results are meant to be reported as real (see spec
section 7) -- it is not carried over to the eventual capped-price impact
evaluation, which needs every household.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile household bills reconstituted from net "
                    "meter flows against the bills already computed by "
                    "the original model run.",
    )
    parser.add_argument('--project-root', type=Path, default=Path('.'),
                        help="Code repo root; model/tariffs/ and the run "
                             "config YAML (if given as a relative path) "
                             "are resolved here. Default: cwd.")
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help="Data root; run outputs, site metadata, and "
                             "NEM prices are found here. Default: cwd.")
    parser.add_argument('--run-config', type=Path, required=True,
                        help="Run config YAML (e.g. "
                             "model/configs/full_run_optimised_amber.yaml). "
                             "Resolved relative to --project-root if not "
                             "absolute. Everything else -- run directory "
                             "name, states, years, battery sizes, tariff "
                             "ids, coordination type -- is read out of "
                             "this file.")
    parser.add_argument('--states', nargs='+', default=None,
                        help="Restrict to a subset of the YAML's states "
                             "list (default: everything in the YAML).")
    parser.add_argument('--years', nargs='+', type=int, default=None,
                        help="Restrict to a subset of the YAML's years "
                             "list (default: everything in the YAML).")
    parser.add_argument('--battery-names', nargs='+', default=None,
                        help="Restrict to a subset of the YAML's "
                             "battery_specs (default: all of them).")
    parser.add_argument('--sample-sites', type=int, default=None,
                        help="Within each scenario, reconcile only the "
                             "first N site columns found in that "
                             "scenario's flows.parquet (default: all "
                             "sites). Dev-only -- omit for real results.")
    parser.add_argument('--output-dir', type=Path,
                        default=Path('analysis/output/bill_reconciliation'),
                        help="Where the diff report is written.")
    return parser.parse_args(argv)


# Coordination types this first version can reconcile. VPP types need
# vpp_forced_flows.parquet and the VPP reward calculation on top of the
# plain reconstitution below.
_UNSUPPORTED_COORDINATION_TYPES = {'vpp_sc_event', 'vpp_tou_event'}

# Reconciliation tolerance per SiteBill field. A field passes if EITHER
# threshold is met. Starting proposal,
# not fixed -- easy to tighten once real numbers are seen.
_ABS_TOL_DOLLARS = 0.01
_REL_TOL = 1e-6

# SiteBill fields compared, in report column order. Mirrors
# model.orchestrate._bill_row exactly, so bills.csv columns line up 1:1 --
# duplicated here (rather than importing the private helper) only because
# importing model.orchestrate pulls in the LP solver stack purely to reuse
# one 12-line dict builder; keep this list in sync with _bill_row if that
# ever changes.
_BILL_FIELDS = [
    'wholesale_usage_cost', 'wholesale_feedin_credit',
    'network_usage_cost', 'network_feedin_credit', 'network_daily_charges',
    'retail_usage_cost', 'retail_feedin_credit', 'retail_daily_charges',
    'retail_other_credits', 'household_total', 'retailer_margin',
]


def _bill_row(bill) -> dict:
    """Flatten a SiteBill into a dict, matching model.orchestrate._bill_row."""
    return {
        'site_id': bill.site_id,
        'plan_type': bill.plan_type,
        **{field: getattr(bill, field) for field in _BILL_FIELDS},
    }


# ----------------------------------------------------------------------
# Flow reconstitution
# ----------------------------------------------------------------------

def _split_net_flow(net: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split a net (import - export) flow series into non-negative
    import_kwh and export_kwh series.

    Exact at 5-minute NEM settlement granularity for non-VPP dispatch
    (confirmed for this scope).
    """
    import_kwh = net.clip(lower=0)
    export_kwh = (-net).clip(lower=0)
    return import_kwh, export_kwh


def _build_site_traces(site_id: str, net: pd.Series):
    """Construct a real SiteTraces for one site from a net flow series.

    Only import_kwh/export_kwh carry real data; every flow-decomposition
    field is an all-zero series aligned to the same index, matching the
    existing convention (SiteTraces docstring) that decomposition columns
    are zero-filled rather than None when not applicable. billing.compute_bill
    only reads site_id/import_kwh/export_kwh, so this is safe even though
    most fields here are placeholders.
    """
    from model.core.site_result import SiteTraces

    import_kwh, export_kwh = _split_net_flow(net)
    zeros = pd.Series(0.0, index=net.index)

    return SiteTraces(
        site_id=site_id,
        import_kwh=import_kwh,
        export_kwh=export_kwh,
        battery_soc_kwh=zeros,
        sol_to_home_kwh=zeros,
        sol_to_bat_kwh=zeros,
        sol_to_grid_kwh=zeros,
        bat_to_home_kwh=zeros,
        grid_to_home_kwh=zeros,
        grid_to_bat_kwh=zeros,
        bat_to_grid_kwh=zeros,
        curtail_kwh=zeros,
    )


# ----------------------------------------------------------------------
# Per-scenario reconciliation
# ----------------------------------------------------------------------

def _reconcile_scenario(scenario, run_name: str, data_root: Path,
                        tariffs_dir: Path, sample_sites: int | None
                        ) -> tuple[pd.DataFrame, dict] | None:
    """Reconcile one scenario. Returns (diff_df, scenario_stats) or None
    if the scenario's output files are missing (skipped, not failed)."""
    from model.core import Paths
    from model.core.tariff import load_network_tariff, load_retail_tariff
    from model.components.ingest import _load_prices
    from model.components.billing import compute_bill

    scenario_dir = (data_root / 'data' / 'output' / 'runs' / run_name
                    / 'scenarios' / scenario.scenario_id)
    flows_path = scenario_dir / 'flows.parquet'
    bills_path = scenario_dir / 'bills.csv'

    if not flows_path.exists() or not bills_path.exists():
        print(f"  SKIP  {scenario.scenario_id}: missing "
              f"{'flows.parquet' if not flows_path.exists() else 'bills.csv'} "
              f"in {scenario_dir}", file=sys.stderr)
        return None

    flows_df = pd.read_parquet(flows_path)
    bills_df = pd.read_csv(bills_path)
    bills_df['site_id'] = bills_df['site_id'].astype(str)
    bills_df = bills_df.set_index('site_id')

    paths = Paths(
        edp_clean_dir=data_root / 'data' / 'clean' / 'edp',
        edp_metadata_dir=data_root / 'data' / 'processed' / 'edp',
        nem_prices_dir=data_root / 'data' / 'clean' / 'nem_prices',
        tariffs_dir=tariffs_dir,
    )
    network_tariff = load_network_tariff(scenario.network_tariff_id,
                                        paths.tariffs_dir)
    retail_tariff = load_retail_tariff(scenario.retail_tariff_id,
                                       paths.tariffs_dir)
    prices = _load_prices(scenario, paths)

    site_ids = [str(c) for c in flows_df.columns]
    if sample_sites is not None:
        site_ids = site_ids[:sample_sites]

    computed_rows = []
    for site_id in site_ids:
        net = flows_df[site_id]
        if not net.index.equals(prices.index):
            raise ValueError(
                f"{scenario.scenario_id} site {site_id}: flows.parquet "
                f"index does not match the price series index. "
                f"flows len={len(net)}, prices len={len(prices)}. "
                f"First flow ts={net.index[0]}, first price ts={prices.index[0]}."
            )
        traces = _build_site_traces(site_id, net)
        bill = compute_bill(traces, prices, network_tariff, retail_tariff,
                            scenario.state)
        computed_rows.append(_bill_row(bill))

    computed_df = pd.DataFrame(computed_rows)
    computed_df['site_id'] = computed_df['site_id'].astype(str)
    computed_df = computed_df.set_index('site_id')

    missing_in_bills = [s for s in site_ids if s not in bills_df.index]
    if missing_in_bills:
        raise ValueError(
            f"{scenario.scenario_id}: {len(missing_in_bills)} site(s) in "
            f"flows.parquet have no row in bills.csv, e.g. "
            f"{missing_in_bills[:5]}. Data may be out of sync."
        )

    diff_rows = []
    for site_id in site_ids:
        computed = computed_df.loc[site_id]
        original = bills_df.loc[site_id]
        row = {
            'site_id': site_id,
            'plan_type_match': computed['plan_type'] == original['plan_type'],
        }
        any_fail = False
        for field in _BILL_FIELDS:
            c = float(computed[field])
            o = float(original[field])
            abs_diff = abs(c - o)
            rel_diff = abs_diff / max(abs(o), 1e-9)
            passed = (abs_diff < _ABS_TOL_DOLLARS) or (rel_diff < _REL_TOL)
            any_fail = any_fail or not passed
            row[f'{field}_computed'] = c
            row[f'{field}_original'] = o
            row[f'{field}_abs_diff'] = abs_diff
            row[f'{field}_rel_diff'] = rel_diff
            row[f'{field}_pass'] = passed
        row['all_pass'] = (not any_fail) and row['plan_type_match']
        diff_rows.append(row)

    diff_df = pd.DataFrame(diff_rows).set_index('site_id')

    stats = {
        'run_name': run_name,
        'scenario_id': scenario.scenario_id,
        'state': scenario.state,
        'year': scenario.year,
        'battery_name': scenario.battery_name,
        'coordination_type': scenario.coordination_type,
        'sites_checked': len(site_ids),
        'sites_passing': int(diff_df['all_pass'].sum()),
        'sites_failing': int((~diff_df['all_pass']).sum()),
    }
    for field in _BILL_FIELDS:
        fail_mask = ~diff_df[f'{field}_pass']
        stats[f'{field}_n_failing'] = int(fail_mask.sum())
        stats[f'{field}_max_abs_diff'] = (
            float(diff_df[f'{field}_abs_diff'].max()) if len(diff_df) else 0.0
        )

    return diff_df, stats


def _write_scenario_report(diff_df: pd.DataFrame, stats: dict,
                           output_dir: Path) -> None:
    scenario_dir = output_dir / stats['scenario_id']
    scenario_dir.mkdir(parents=True, exist_ok=True)

    diff_df.to_csv(scenario_dir / 'diffs.csv')

    lines = [
        f"Scenario: {stats['scenario_id']} (run: {stats['run_name']})",
        f"State: {stats['state']}  Year: {stats['year']}  "
        f"Battery: {stats['battery_name']}  "
        f"Coordination: {stats['coordination_type']}",
        "",
        f"Sites checked: {stats['sites_checked']}",
        f"Sites passing (all fields): {stats['sites_passing']}",
        f"Sites failing (>=1 field):  {stats['sites_failing']}",
        "",
        "Per-field failure counts and largest absolute diff:",
    ]
    for field in _BILL_FIELDS:
        n_fail = stats[f'{field}_n_failing']
        max_diff = stats[f'{field}_max_abs_diff']
        lines.append(f"  {field:28s} failing={n_fail:4d}  "
                     f"max_abs_diff=${max_diff:,.4f}")

    (scenario_dir / 'summary.txt').write_text('\n'.join(lines) + '\n')


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()

    # model.* is only importable once project_root is on sys.path -- do
    # this before any model.* import, including the deferred ones below.
    sys.path.insert(0, str(project_root))

    from model.main import load_scenarios

    run_config_path = args.run_config
    if not run_config_path.is_absolute():
        run_config_path = project_root / run_config_path

    run_name, scenarios = load_scenarios(run_config_path)

    if args.states is not None:
        scenarios = [s for s in scenarios if s.state in args.states]
    if args.years is not None:
        scenarios = [s for s in scenarios if s.year in args.years]
    if args.battery_names is not None:
        scenarios = [s for s in scenarios
                    if s.battery_name in args.battery_names]

    if not scenarios:
        print("No scenarios matched the given filters.", file=sys.stderr)
        return 1

    if args.sample_sites is not None:
        print(f"NOTE: --sample-sites {args.sample_sites} is active -- "
              f"this run is for development/sanity-checking only, not "
              f"a complete reconciliation. Omit --sample-sites for real "
              f"results.", file=sys.stderr)

    tariffs_dir = project_root / 'model' / 'tariffs'
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary_rows = []
    print(f"Reconciling {len(scenarios)} scenario(s) from run '{run_name}'...")
    for scenario in scenarios:
        if scenario.coordination_type in _UNSUPPORTED_COORDINATION_TYPES:
            print(f"  SKIP  {scenario.scenario_id}: coordination_type="
                 f"{scenario.coordination_type!r} not supported yet "
                 f"(VPP reconciliation needs the forced flows and VPP "
                 f"reward on top of the plain reconstitution).")
            continue

        result = _reconcile_scenario(
            scenario, run_name, data_root, tariffs_dir, args.sample_sites,
        )
        if result is None:
            continue
        diff_df, stats = result
        _write_scenario_report(diff_df, stats, output_dir)
        run_summary_rows.append(stats)

        status = "OK" if stats['sites_failing'] == 0 else "MISMATCH"
        print(f"  {status:8s} {scenario.scenario_id}: "
             f"{stats['sites_passing']}/{stats['sites_checked']} sites pass")

    if not run_summary_rows:
        print("No scenarios were reconciled (all skipped or missing "
             "output files).", file=sys.stderr)
        return 1

    run_summary_df = pd.DataFrame(run_summary_rows)
    run_summary_df.to_csv(output_dir / 'run_summary.csv', index=False)

    total_checked = run_summary_df['sites_checked'].sum()
    total_passing = run_summary_df['sites_passing'].sum()
    print(f"\nDone. {total_passing}/{total_checked} sites reconciled "
         f"across {len(run_summary_df)} scenario(s).")
    print(f"Report written to {output_dir}/")

    return 0


if __name__ == '__main__':
    sys.exit(main())
