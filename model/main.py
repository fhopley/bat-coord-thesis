"""CLI entry point for the coordination model.

Loads a scenario YAML, expands it into one or more ScenarioConfigs via
cross-product over states / years / battery_specs, validates ALL scenarios
strictly (any failure aborts), then runs each scenario sequentially with
per-site progress reporting.

Usage (from project root):
    python3 -u -m model.main configs/example.yaml [--save-trace SITE_ID]
                                                [--project-root PATH]
Examples :
    python3 -u -m model.main model/configs/example_perfect_foresight_lp.yaml --workers 6 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log
    python3 -u -m model.main model/configs/example_perfect_foresight_lp.yaml --max-sites 30 --workers 6 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log
    python3 -u -m model.main model/configs/example_single_scenario.yaml --save-trace S0008 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log
    python3 -u -m model.main model/configs/full_run_solar_only_amber.yaml --validate-only

YAML schema (see model/configs/ for examples)
---------------------------------------------
A minimal scenario file:

    name: example_run                  # optional; defaults to YAML filename stem
    states: [NSW]                      # required; list (cross-product dimension)
    years: [2022]                      # required; list (cross-product dimension)
    cohort: solar_battery              # required
    coordination_type: self_consumption  # required for solar_battery cohort

    # Tariffs: short form when states and years each have a single entry
    network_tariff_id: ausgrid_ea025_network_2024
    retail_tariff_id: amber_wholesale_2024

    # Battery: short form is a single spec applied to all expanded scenarios
    battery_spec:
      capacity_kwh: 13.5
      backup_reserve_pct: 10.0
      power_kw: 5.0
      charge_efficiency: 0.95
      discharge_efficiency: 0.95

Multi-state / multi-year / multi-battery (cross-product expansion):

    states: [NSW, VIC]
    years: [2022, 2023]

    # When states has more than one entry, tariffs must be per-state mappings.
    # Inner mapping keys are years.
    network_tariff_ids:
      NSW: {2022: ausgrid_ea025_network_2024, 2023: ausgrid_ea025_network_2024}
      VIC: {2022: vic_network_2024,           2023: vic_network_2024}
    retail_tariff_ids:
      NSW: {2022: amber_wholesale_2024, 2023: amber_wholesale_2024}
      VIC: {2022: amber_wholesale_2024, 2023: amber_wholesale_2024}

    # Multiple batteries: each becomes a separate expanded scenario.
    # Each entry needs a 'name' which becomes part of the scenario_id.
    battery_specs:
      - name: powerwall
        capacity_kwh: 13.5
        backup_reserve_pct: 10.0
        power_kw: 5.0
        charge_efficiency: 0.95
        discharge_efficiency: 0.95
      - name: smaller
        capacity_kwh: 6.0
        backup_reserve_pct: 10.0
        power_kw: 3.0
        charge_efficiency: 0.95
        discharge_efficiency: 0.95

Output structure (see docs/model_spec.md, Outputs)
----------------
    data/output/runs/{run_name}/
      scenarios/{scenario_id}/
        bills.csv
        summary.txt
        rolling_diagnostics.csv        # rolling coordination types only
        bills_01.csv .. bills_12.csv   # only with --monthly-bills
        flows.parquet, battery_flows.parquet,
        vpp_forced_flows.parquet       # only with --save-flows
        trace_{site_id}.csv            # only with --save-trace
      run_summary.txt
"""
import argparse
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from model.core import (
    ScenarioConfig, RollingConfig, BatterySpec, Paths, default_paths,
)
# UsageForecastConfig/SolarForecastConfig are NOT re-exported from model.core
# (unlike RollingConfig, which is) -- confirmed by the ImportError this line
# previously caused. Imported from their actual module directly rather than
# guessing at what model/core/__init__.py should export, matching how
# viz_rolling_perfect_foresight_lp.py already imports them successfully.
from model.core.scenario import UsageForecastConfig, SolarForecastConfig
from model.orchestrate import (
    run_scenario, validate_scenario, ScenarioValidationError,
    write_bills_csv, write_monthly_bills_csv, write_scenario_summary, write_single_trace, write_flows_parquet,
    write_vpp_forced_flows_parquet, write_battery_flows_parquet,
    write_rolling_diagnostics_csv,
    _format_elapsed,
)


# ----------------------------------------------------------------------
# YAML parsing and cross-product expansion
# ----------------------------------------------------------------------

def _require(data: dict, key: str, ctx: str) -> Any:
    if key not in data:
        raise ValueError(f"{ctx}: missing required field {key!r}")
    return data[key]


def _resolve_tariff(
    field_singular: str, field_mapping: str,
    data: dict, states: list, years: list, ctx: str,
) -> dict[tuple[str, int], str]:
    """Resolve a tariff specification to a (state, year) -> tariff_id mapping.

    Accepts either the singular string form (only valid when one state and
    one year) or the per-state-per-year mapping form. The result is always
    a dict keyed by (state, year).
    """
    has_singular = field_singular in data
    has_mapping = field_mapping in data
    if has_singular and has_mapping:
        raise ValueError(
            f"{ctx}: specify {field_singular!r} OR {field_mapping!r}, not both"
        )
    if not has_singular and not has_mapping:
        raise ValueError(
            f"{ctx}: must specify {field_singular!r} or {field_mapping!r}"
        )

    if has_singular:
        if len(states) > 1 or len(years) > 1:
            raise ValueError(
                f"{ctx}: {field_singular!r} (string form) requires exactly one "
                f"state and one year; got states={states}, years={years}. "
                f"Use {field_mapping!r} (mapping form) for multi-state/multi-year."
            )
        tid = data[field_singular]
        return {(states[0], years[0]): tid}

    # Mapping form
    mapping = data[field_mapping]
    out = {}
    for s in states:
        if s not in mapping:
            raise ValueError(f"{ctx}: {field_mapping!r} missing state {s!r}")
        per_state = mapping[s]
        for y in years:
            if y not in per_state:
                raise ValueError(
                    f"{ctx}: {field_mapping!r}[{s!r}] missing year {y}"
                )
            out[(s, y)] = per_state[y]
    return out


def _parse_battery_specs(data: dict, cohort: str, ctx: str
                        ) -> list[tuple[str | None, BatterySpec | None]]:
    """Return a list of (name, BatterySpec) pairs for cross-product expansion.

    For non-battery cohorts: returns [(None, None)] so the cross-product still
    yields one scenario per (state, year).
    For battery cohorts: returns a list of (name, spec) - one entry from a
    singular battery_spec block (name=None), or multiple entries from a
    battery_specs list (each must have a 'name').
    """
    if cohort != 'solar_battery':
        if 'battery_spec' in data or 'battery_specs' in data:
            raise ValueError(
                f"{ctx}: battery_spec/battery_specs not allowed for "
                f"cohort={cohort!r}"
            )
        return [(None, None)]

    has_singular = 'battery_spec' in data
    has_list = 'battery_specs' in data
    if has_singular and has_list:
        raise ValueError(
            f"{ctx}: specify 'battery_spec' OR 'battery_specs', not both"
        )
    if not has_singular and not has_list:
        raise ValueError(
            f"{ctx}: cohort=solar_battery requires 'battery_spec' or "
            f"'battery_specs'"
        )

    if has_singular:
        spec_data = data['battery_spec']
        return [(None, BatterySpec(**spec_data))]

    # battery_specs list - each must carry a 'name' field
    specs = data['battery_specs']
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"{ctx}: 'battery_specs' must be a non-empty list")
    out = []
    for i, entry in enumerate(specs):
        if 'name' not in entry:
            raise ValueError(
                f"{ctx}: battery_specs[{i}] missing required 'name' field"
            )
        name = entry.pop('name')
        try:
            spec = BatterySpec(**entry)
        except TypeError as e:
            raise ValueError(
                f"{ctx}: battery_specs[{i}] ({name!r}): {e}"
            ) from e
        out.append((name, spec))
    # Detect duplicate names that would collide in scenario_id
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(
            f"{ctx}: duplicate battery_specs names: {names}"
        )
    return out


def _parse_forecast_sub_config(raw, cls, key: str, ctx: str):
    """Parse an optional nested `rolling.usage_forecast` / `solar_forecast`
    block into its dataclass, or None when the key is absent.

    An empty mapping (`{}`) is a valid and meaningful value distinct from
    omitting the key: it means "enabled, all defaults" (perfect site
    knowledge is a rung of the ladder in its own right, not a degenerate
    absence -- see RollingConfig's docstring), whereas omitting the key
    entirely means the forecast is not used at all and the corresponding
    site input is known perfectly.

    Unknown sub-keys surface as the dataclass's own TypeError, wrapped here
    so the message names the YAML path rather than a Python constructor.
    """
    if key not in raw:
        return None
    sub = raw[key]
    if not isinstance(sub, dict):
        raise ValueError(f"{ctx}: 'rolling.{key}' must be a mapping")
    try:
        return cls(**sub)
    except TypeError as e:
        raise ValueError(f"{ctx}: invalid 'rolling.{key}' block: {e}") from e


def _parse_rolling(data: dict, ctx: str):
    """Parse the optional `rolling:` block into a RollingConfig.

    Returns None when absent. RollingConfig.__post_init__ validates the field
    values; ScenarioConfig.__post_init__ validates presence against the
    coordination type, so neither check is repeated here.

    forecast_source stays required even for an advanced TOU plan with no
    critical rate, where nothing reads a price forecast and the value is
    inert. This parser never loads the tariff, so it cannot tell those plans
    apart from the ones that do need it, and a key that is sometimes required
    depending on a file it cannot see would be worse than one that is always
    required and sometimes ignored.
    """
    raw = data.get('rolling')
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"{ctx}: 'rolling' must be a mapping")

    # usage_forecast / solar_forecast, commit_actual_load, on_degraded_period
    # and max_staleness_hours were added to RollingConfig (see
    # model/core/scenario.py) after this known-key set was first written, so
    # a YAML block using them was silently rejected as 'unknown key' even
    # though the dataclass itself already accepted them.
    known = {'forecast_source', 'horizon_hours', 'infill_lag_hours',
             'current_interval_price', 'retain_plans_from', 'retain_plans_to',
             'usage_forecast', 'solar_forecast', 'commit_actual_load',
             'on_degraded_period', 'max_staleness_hours'}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"{ctx}: unknown key(s) in 'rolling': {sorted(unknown)}. "
            f"Known keys: {sorted(known)}"
        )
    if 'forecast_source' not in raw:
        raise ValueError(f"{ctx}: 'rolling.forecast_source' is required")

    usage_forecast = _parse_forecast_sub_config(
        raw, UsageForecastConfig, 'usage_forecast', ctx)
    solar_forecast = _parse_forecast_sub_config(
        raw, SolarForecastConfig, 'solar_forecast', ctx)

    try:
        return RollingConfig(
            forecast_source=str(raw['forecast_source']),
            horizon_hours=float(raw.get('horizon_hours', 48.0)),
            infill_lag_hours=float(raw.get('infill_lag_hours', 24.0)),
            current_interval_price=str(
                raw.get('current_interval_price', 'forecast')),
            retain_plans_from=raw.get('retain_plans_from'),
            retain_plans_to=raw.get('retain_plans_to'),
            usage_forecast=usage_forecast,
            solar_forecast=solar_forecast,
            commit_actual_load=bool(raw.get('commit_actual_load', True)),
            on_degraded_period=str(
                raw.get('on_degraded_period', 'fail')),
            max_staleness_hours=float(
                raw.get('max_staleness_hours', 6.0)),
        )
    except ValueError as e:
        raise ValueError(f"{ctx}: invalid 'rolling' block: {e}") from e


# Top-level keys load_scenarios reads. Anything else is rejected, so a mistyped
# or stale key fails at load instead of running with its default.
_KNOWN_TOP_LEVEL_KEYS = frozenset({
    'name', 'states', 'years', 'cohort', 'coordination_type',
    'network_tariff_id', 'network_tariff_ids',
    'retail_tariff_id', 'retail_tariff_ids',
    'battery_spec', 'battery_specs',
    'rolling', 'export_limit_kw_per_phase', 'opt_factor',
    'op_retail_tariff_id', 'apply_export_order_nudge',
})


def load_scenarios(yaml_path: Path) -> tuple[str, list[ScenarioConfig]]:
    """Parse a scenario YAML and expand into a list of ScenarioConfig.

    Returns (run_name, scenarios).
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Scenario YAML not found: {yaml_path}")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    ctx = str(yaml_path)
    run_name = data.get('name', yaml_path.stem)

    states = _require(data, 'states', ctx)
    if not isinstance(states, list) or not states:
        raise ValueError(f"{ctx}: 'states' must be a non-empty list")
    years = _require(data, 'years', ctx)
    if not isinstance(years, list) or not years:
        raise ValueError(f"{ctx}: 'years' must be a non-empty list")
    years = [int(y) for y in years]
    cohort = _require(data, 'cohort', ctx)

    coordination_type = data.get('coordination_type')
    if cohort == 'solar_battery' and coordination_type is None:
        raise ValueError(
            f"{ctx}: coordination_type is required for cohort=solar_battery"
        )

    network_map = _resolve_tariff(
        'network_tariff_id', 'network_tariff_ids', data, states, years, ctx)
    retail_map = _resolve_tariff(
        'retail_tariff_id', 'retail_tariff_ids', data, states, years, ctx)

    battery_entries = _parse_battery_specs(data, cohort, ctx)

    # Removed in the rolling_optimisation work. These keys were parsed and
    # validated but never read by any component, so a config carrying them was
    # already a no-op. Rejecting them explicitly means a stale config fails
    # with a pointer rather than being silently ignored.
    for dead in ('use_rolling_window', 'window_days'):
        if dead in data:
            raise ValueError(
                f"{ctx}: '{dead}' was removed and had no effect. For rolling "
                "window dispatch use coordination_type: "
                "rolling_perfect_foresight_lp (wholesale-exposed plans) or "
                "rolling_perfect_foresight_lp_advanced (advanced TOU plans), "
                "with a 'rolling:' block (see rolling_optimisation_spec.md)."
            )

    unknown = set(data) - _KNOWN_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"{ctx}: unknown top-level key(s) {sorted(unknown)}. "
            f"Known keys: {sorted(_KNOWN_TOP_LEVEL_KEYS)}"
        )

    rolling = _parse_rolling(data, ctx)
    export_limit_kw_per_phase = data.get('export_limit_kw_per_phase')
    apply_export_order_nudge = data.get('apply_export_order_nudge', False)
    if not isinstance(apply_export_order_nudge, bool):
        raise ValueError(
            f"{ctx}: apply_export_order_nudge must be true or false, got "
            f"{apply_export_order_nudge!r}"
        )
    opt_factor = float(data.get('opt_factor', 1.0))
    op_retail_tariff_id = data.get('op_retail_tariff_id')
    if not (0.0 < opt_factor <= 1.0):
        raise ValueError(
            f"{ctx}: opt_factor must be in (0, 1], got {opt_factor}. "
            "Only meaningful for coordination_type=perfect_foresight_custom."
        )

    scenarios: list[ScenarioConfig] = []
    for state, year, (bname, bspec) in product(states, years, battery_entries):
        scenarios.append(ScenarioConfig(
            state=state,
            year=year,
            cohort=cohort,
            network_tariff_id=network_map[(state, year)],
            retail_tariff_id=retail_map[(state, year)],
            coordination_type=coordination_type,
            battery_spec=bspec,
            battery_name=bname,
            rolling=rolling,
            export_limit_kw_per_phase=export_limit_kw_per_phase,
            opt_factor=opt_factor,
            op_retail_tariff_id=op_retail_tariff_id,
            apply_export_order_nudge=apply_export_order_nudge,
        ))
    return run_name, scenarios


# ----------------------------------------------------------------------
# Run execution
# ----------------------------------------------------------------------

def _progress_printer(scenario_id: str, label: str = ""):
    """Returns a progress callback that prints '[i/n] scenario_id [label] site_id'.

    label is an optional short tag (e.g. 'ingest') to visually distinguish
    different phases when both ingest and dispatch are printing progress.
    """
    prefix = f"{label} " if label else ""
    def cb(i, n, site_id, note=""):
        suffix = f"  {note}" if note else ""
        print(f"  [{i}/{n}] {scenario_id} {prefix}{site_id}{suffix}", flush=True)
    return cb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run scenarios defined in a YAML configuration file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('yaml_path', type=Path,
                        help="Path to the scenario YAML")
    parser.add_argument('--save-trace', metavar='SITE_ID', default=None,
                        help="If given, save the full per-interval trace for "
                             "this single site in each scenario.")
    parser.add_argument('--save-flows', action='store_true', default=False,
                        help="If given, write flows.parquet for each scenario: "
                             "wide format with datetime_aest and one float32 "
                             "column per site (net import/export kWh, positive "
                             "= import). Omitted by default to avoid large "
                             "files on laptop runs.")
    parser.add_argument('--monthly-bills', action='store_true', default=False,
                        help="If given, also compute per-site monthly "
                             "SiteBills during dispatch and write "
                             "bills_01.csv .. bills_12.csv per scenario "
                             "(see run_scenario's with_monthly_bills and "
                             "billing.compute_bill_and_monthly_bills, which "
                             "share one settlement build with the annual "
                             "bill rather than re-deriving rates/local time "
                             "separately). Off by default: even with that "
                             "sharing, deriving monthly bills is not free, "
                             "and most runs only need the annual bill.")
    parser.add_argument('--workers', type=int, default=1,
                        help="Number of parallel worker processes for "
                             "perfect_foresight LP solves. Each site is "
                             "independent so this gives near-linear speedup "
                             "up to the number of physical cores. "
                             "Recommended: n_physical_cores - 1 "
                             "(default: 1, serial).")
    parser.add_argument('--sites', default=None, metavar='ID,ID,...',
                        help="Process only these comma-separated site ids "
                             "(applied before --max-sites). Every id must be "
                             "in each scenario's site set.")
    parser.add_argument('--max-sites', type=int, default=None,
                        metavar='N',
                        help="Process only the first N sites. Useful for "
                             "quick timing benchmarks without editing the "
                             "YAML (e.g. --max-sites 12 --workers 3).")
    parser.add_argument('--project-root', type=Path, default=None,
                        help="Override project root (default: cwd)")
    parser.add_argument('--output-dir', type=Path, default=None,
                        help="Override output directory "
                             "(default: data/output/runs/<run_name>)")
    parser.add_argument('--tariffs-dir', type=Path, default=None,
                        help="Override tariffs directory "
                             "(default: <project-root>/model/tariffs)")
    parser.add_argument('--validate-only', action='store_true', default=False,
                        help="Load, expand, and validate every scenario in "
                             "the YAML, then exit without running anything. "
                             "Unlike a normal run, does NOT abort on the "
                             "first failing scenario -- checks all of them "
                             "and reports every failure, so a full preflight "
                             "pass over a multi-day config surfaces every "
                             "problem in one go. Exit code 0 if all "
                             "scenarios are valid, 3 if any failed (same "
                             "convention as a normal run's fail-fast abort).")
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    if args.tariffs_dir is not None:
        paths = Paths(
            edp_clean_dir=paths.edp_clean_dir,
            edp_metadata_dir=paths.edp_metadata_dir,
            nem_prices_dir=paths.nem_prices_dir,
            tariffs_dir=args.tariffs_dir,
        )

    # Load and expand scenarios
    try:
        run_name, scenarios = load_scenarios(args.yaml_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR loading {args.yaml_path}: {e}", file=sys.stderr)
        return 2

    print(f"Run: {run_name}")
    print(f"Loaded {len(scenarios)} scenario(s) from {args.yaml_path}")
    for s in scenarios:
        print(f"  - {s.scenario_id}")
    print()

    # Pre-flight validation. Two modes:
    #   --validate-only: exhaustive -- check every scenario, never abort
    #     early, report every failure. For checking a config file (or many,
    #     via a wrapper script) before committing days of compute to it.
    #   normal run: fail-fast -- abort on the first bad scenario, so a bad
    #     config never wastes processing time mid-run.
    if args.validate_only:
        print(f"Validate-only: checking all {len(scenarios)} scenario(s), "
              f"not aborting on first failure...")
        n_ok = 0
        failures: list[tuple[str, str]] = []
        for s in scenarios:
            try:
                validate_scenario(s, paths)
                print(f"  OK    {s.scenario_id}")
                n_ok += 1
            except ScenarioValidationError as e:
                print(f"  FAIL  {s.scenario_id}: {e}", file=sys.stderr)
                failures.append((s.scenario_id, str(e)))
        print()
        print(f"Validate-only result: {n_ok}/{len(scenarios)} scenario(s) OK, "
              f"{len(failures)} failed")
        if failures:
            print("\nFailed scenarios:", file=sys.stderr)
            for sid, msg in failures:
                print(f"  {sid}: {msg}", file=sys.stderr)
            return 3
        return 0

    print("Pre-flight validation...")
    for s in scenarios:
        try:
            validate_scenario(s, paths)
            print(f"  OK    {s.scenario_id}")
        except ScenarioValidationError as e:
            print(f"  FAIL  {e}", file=sys.stderr)
            print(f"\nABORTED: pre-flight validation failed for {s.scenario_id}",
                  file=sys.stderr)
            return 3
    print()

    # Output directory
    if args.output_dir is not None:
        out_root = args.output_dir
    else:
        root = args.project_root or Path.cwd()
        out_root = root / 'data' / 'output' / 'runs' / run_name
    print(f"Output directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    # Execute each scenario
    summaries = []
    run_t0 = time.perf_counter()
    for idx, scenario in enumerate(scenarios, start=1):
        print(f"\n=== Scenario {idx}/{len(scenarios)}: {scenario.scenario_id} ===")
        result = run_scenario(
            scenario, paths,
            ingest_progress_callback=_progress_printer(scenario.scenario_id,
                                                      label='ingest'),
            progress_callback=_progress_printer(scenario.scenario_id),
            n_workers=args.workers,
            max_sites=args.max_sites,
            with_monthly_bills=args.monthly_bills,
            only_sites=(args.sites.split(',') if args.sites else None),
        )

        sdir = out_root / 'scenarios' / scenario.scenario_id
        write_bills_csv(result, sdir / 'bills.csv')
        # No-ops (writes nothing) when result.rolling_diagnostics is None,
        # i.e. for every coordination_type outside
        # ROLLING_COORDINATION_TYPES -- same "always call,
        # self-gates on the field" style as write_bills_csv above, rather
        # than an if-args.something flag.
        write_rolling_diagnostics_csv(result, sdir / 'rolling_diagnostics.csv')
        if args.monthly_bills:
            write_monthly_bills_csv(result, sdir)
        if args.save_flows:
            write_flows_parquet(result, sdir / 'flows.parquet')
            write_battery_flows_parquet(result, sdir / 'battery_flows.parquet')
            write_vpp_forced_flows_parquet(
                result, sdir / 'vpp_forced_flows.parquet')
        summary_text = write_scenario_summary(result, sdir / 'summary.txt')
        summaries.append(summary_text)
        print()
        print(summary_text)

        if args.save_trace is not None:
            trace_path = sdir / f'trace_{args.save_trace}.csv'
            try:
                write_single_trace(result, args.save_trace, trace_path)
                print(f"Saved trace: {trace_path}")
            except KeyError as e:
                print(f"WARNING: {e}", file=sys.stderr)

    run_elapsed = time.perf_counter() - run_t0
    run_header = (
        f"Run: {run_name}\n"
        f"Scenarios: {len(scenarios)}\n"
        f"Total elapsed: {_format_elapsed(run_elapsed)}\n\n"
    )

    # Run-level summary
    run_summary_path = out_root / 'run_summary.txt'
    with open(run_summary_path, 'w') as f:
        f.write(run_header)
        f.write("\n\n".join(summaries))
    print(f"\nTotal run time: {_format_elapsed(run_elapsed)}")
    print(f"Run summary written to: {run_summary_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())