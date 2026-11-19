"""Reconstruct battery_flows.parquet for historical battery-cohort runs.

This is a one-off backfill script. Runs produced before the model saved
battery flows directly only wrote flows.parquet (net site import/export
per site). This script reconstructs the net battery AC flow per site from:

  - flows.parquet (net site import/export, already on disk)
  - the raw EDP clean traces (solar generation and load, already on disk)
  - the battery spec recorded in summary.txt

Going forward, the live model should save battery_flows.parquet directly
during the run rather than needing this reconstruction. This script exists
only to backfill runs that predate that change.

Derivation
----------
Let, for a single site and a single 5-min interval:

    S  = pv_site_net_kwh          (solar; can be negative, parasitic draw)
    L  = gross_load_kwh           (load; always >= 0)
    F  = flows.parquet value      (import positive, export negative)

Because a site is never simultaneously importing and exporting within one
interval (true of both the EDP files and flows.parquet), F alone is enough
to recover import and export separately:

    Fi = max(F, 0)     # import_kwh
    Fe = Fi - F        # export_kwh

The full flow decomposition follows self-consumption priority order (solar
to load, then solar to battery, then solar to grid; battery to load before
battery to grid):

    solar_to_home = min(S, L)
    solar_to_grid = min(S - solar_to_home, Fe)
    solar_to_bat  = S - solar_to_home - solar_to_grid
    grid_to_home  = min(Fi, L - solar_to_home)
    grid_to_bat   = Fi - grid_to_home
    bat_to_home   = L - solar_to_home - grid_to_home
    bat_to_grid   = Fe - solar_to_grid

    B = solar_to_bat + grid_to_bat - bat_to_home - bat_to_grid

Substituting the intermediate definitions back in, every sol_to_home /
sol_to_grid / grid_to_home term cancels algebraically and B reduces to the
closed form:

    B = S - L + F

This means B does not depend on how the intermediate flows are split; it
falls straight out of energy conservation given only S, L, and F. The
script computes the full decomposition anyway (rather than just the
one-liner) because the intermediate values support real diagnostics: AC
power-cap violations, simultaneous charge+discharge, and a cross-check
against the closed form as an implementation bug catch (see
reconstruct_battery_flow below).

Known limitation: solar curtailment
------------------------------------
The identity above assumes zero curtailment. It is not: the true relation
(see model_spec.md's SiteTraces invariants) is

    F = (L - S) + curtail + B

Curtailment is exactly zero for coordination_type='self_consumption', but
can be non-zero for perfect_foresight / perfect_foresight_advanced /
perfect_foresight_custom / vpp_sc_event / vpp_tou_event, if an export limit
binds or (VPP only) a curtailment cost is configured. Per Fred (2026-07-08):
site-level export limits already curtail solar generation upstream of this
model, and any further economic curtailment inside the dispatch is rare
enough to ignore for this backfill. So this script deliberately treats
curtail as zero for every scenario, regardless of coordination_type. This
is a known, accepted approximation, not an oversight: the fix path, if this
ever needs revisiting, is to have the live model save curtail_kwh
per-site alongside flows.parquet (already computed internally by the LP
and vpp_sc_event components; simply not persisted today) and add it back
into the identity above.

One dispatch logic for every coordination_type
------------------------------------------------
Per Fred: use the same reconstruction logic regardless of which
coordination_type actually produced the run. This is consistent with the
curtailment approximation above: once curtailment is assumed away, the
flow-balance identity is the same shape for every coordination_type (self
consumption priority order is just one algebraically-equivalent way of
presenting the split; the net B is order-independent, as shown above).

Assumptions carried over from ingest.py / orchestrate.py
----------------------------------------------------------
  - solar_kwh in the original dispatch was pv_site_net_kwh * solar_scale,
    with solar_scale hardcoded to 1.0 for every site
    (orchestrate._resolve_per_site_solar_scaling). This script assumes 1.0.
  - load_kwh in the original dispatch was gross_load_kwh, unmodified.

Usage (from project root):
    python3 -m scripts.reconstruct_battery_flows
    python3 -m scripts.reconstruct_battery_flows my_run_name
    python3 -m scripts.reconstruct_battery_flows --check-only
    python3 -m scripts.reconstruct_battery_flows --project-root /path/to/project
"""
import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from model.core import Paths, default_paths


_INTERVAL_HOURS = 5 / 60

# Tolerance for the diagnostic checks below. Not a physical quantity, just
# slack for float32/float64 round-trip noise so we don't flag noise as a
# violation.
_EPS_KWH = 1e-6

_BATTERY_LINE_RE = re.compile(
    r"^\s*battery(?:\s+'[^']*')?:\s*"
    r"(?P<capacity_kwh>[\d.eE+-]+)\s*kWh\s*/\s*"
    r"(?P<power_kw>[\d.eE+-]+)\s*kW\s*/\s*"
    r"reserve\s+(?P<reserve_pct>[\d.eE+-]+)%\s*/\s*"
    r"eff\s+(?P<charge_eff>[\d.eE+-]+)/(?P<discharge_eff>[\d.eE+-]+)\s*$",
    re.MULTILINE,
)
_STATE_YEAR_LINE_RE = re.compile(
    r"^\s*state=(?P<state>\w+)\s+year=(?P<year>\d+)\s+cohort=(?P<cohort>\w+)\s*$",
    re.MULTILINE,
)


@dataclass
class BatteryMeta:
    """Battery spec and scenario identity parsed from summary.txt."""
    state: str
    year: int
    cohort: str
    capacity_kwh: float
    power_kw: float
    reserve_pct: float
    charge_eff: float
    discharge_eff: float


@dataclass
class ScenarioDiagnostics:
    """Aggregate diagnostics across all sites in one scenario."""
    n_sites: int = 0
    n_intervals: int = 0
    max_closed_form_diff_kwh: float = 0.0
    n_power_cap_violations: int = 0
    max_power_cap_violation_kwh: float = 0.0
    n_simultaneous_charge_discharge: int = 0
    n_unexpected_negative: int = 0
    n_negative_load: int = 0
    sites_with_issues: list = field(default_factory=list)

    def note(self, site_id: str, cap_viol: int, sim: int, neg: int, neg_load: int):
        if cap_viol or sim or neg or neg_load:
            self.sites_with_issues.append(site_id)

    def summary_line(self) -> str:
        flags = []
        if self.n_power_cap_violations:
            flags.append(f"{self.n_power_cap_violations} power-cap violations "
                         f"(max {self.max_power_cap_violation_kwh:.4f} kWh)")
        if self.n_simultaneous_charge_discharge:
            flags.append(f"{self.n_simultaneous_charge_discharge} simultaneous "
                         f"charge+discharge intervals")
        if self.n_unexpected_negative:
            flags.append(f"{self.n_unexpected_negative} unexpected-negative "
                         f"intervals")
        if self.n_negative_load:
            flags.append(f"{self.n_negative_load} negative-load intervals "
                         f"(raw data)")
        flag_text = "; ".join(flags) if flags else "clean"
        return (f"{self.n_sites} sites, {self.n_intervals} intervals/site, "
                f"closed-form max diff {self.max_closed_form_diff_kwh:.2e} kWh, "
                f"{flag_text}")


def parse_summary(summary_path: Path) -> Optional[BatteryMeta]:
    """Parse summary.txt for battery spec and scenario identity.

    Returns None if no battery line is present (non-battery cohort), which
    the caller treats as "skip, not an error".
    """
    text = summary_path.read_text()

    sy_match = _STATE_YEAR_LINE_RE.search(text)
    if sy_match is None:
        raise ValueError(
            f"{summary_path}: could not find 'state=... year=... cohort=...' "
            f"line. Is this a well-formed summary.txt?"
        )

    b_match = _BATTERY_LINE_RE.search(text)
    if b_match is None:
        return None

    return BatteryMeta(
        state=sy_match.group('state'),
        year=int(sy_match.group('year')),
        cohort=sy_match.group('cohort'),
        capacity_kwh=float(b_match.group('capacity_kwh')),
        power_kw=float(b_match.group('power_kw')),
        reserve_pct=float(b_match.group('reserve_pct')),
        charge_eff=float(b_match.group('charge_eff')),
        discharge_eff=float(b_match.group('discharge_eff')),
    )


def load_raw_solar_load(site_id: str, year: int, edp_clean_dir: Path) -> pd.DataFrame:
    """Load a site's raw solar (S) and load (L) traces in kWh.

    Mirrors ingest.py's Wh -> kWh conversion for these two columns only;
    does not go through ingest() since no tariffs or prices are needed here.
    """
    trace_path = edp_clean_dir / str(year) / f'edp_clean_{site_id}_{year}.parquet'
    if not trace_path.exists():
        raise FileNotFoundError(
            f"Clean trace file not found for site {site_id} year {year}: "
            f"{trace_path}"
        )

    df = pd.read_parquet(
        trace_path, columns=['datetime_aest', 'pv_site_net_wh', 'gross_load_wh']
    )
    idx = pd.to_datetime(df['datetime_aest']).dt.tz_convert('Etc/GMT-10')
    out = pd.DataFrame(
        {
            'S': df['pv_site_net_wh'].to_numpy(dtype='float64') / 1000.0,
            'L': df['gross_load_wh'].to_numpy(dtype='float64') / 1000.0,
        },
        index=idx,
    )
    out.index.name = 'datetime_aest'
    return out


def _check_alignment(raw_index: pd.DatetimeIndex, flows_index: pd.DatetimeIndex,
                     site_id: str) -> None:
    """Verify raw trace and flows.parquet share the same instants.

    Compares on absolute instants (UTC), not tz label, matching
    ingest._verify_index_alignment.
    """
    if len(raw_index) != len(flows_index):
        raise ValueError(
            f"Site {site_id}: raw trace has {len(raw_index)} intervals, "
            f"flows.parquet has {len(flows_index)}."
        )
    raw_utc = raw_index.tz_convert('UTC')
    flows_utc = flows_index.tz_convert('UTC')
    if not (raw_utc == flows_utc).all():
        mismatch = np.flatnonzero(raw_utc != flows_utc)
        first = mismatch[0]
        raise ValueError(
            f"Site {site_id}: raw trace index does not match flows.parquet "
            f"index. First mismatch at position {first}: "
            f"raw={raw_index[first]}, flows={flows_index[first]}."
        )


def reconstruct_battery_flow(
    S: np.ndarray, L: np.ndarray, F: np.ndarray,
    power_kw: float, interval_hours: float = _INTERVAL_HOURS,
) -> tuple[np.ndarray, dict]:
    """Reconstruct net battery AC flow for one site, plus diagnostics.

    Returns (B, diagnostics) where B is positive = net charging, matching
    flows.parquet's positive = import convention, and diagnostics is a dict
    of raw counts/values (aggregated by the caller across sites).

    Curtailment is assumed zero throughout (see module docstring).
    """
    Fi = np.clip(F, 0.0, None)
    Fe = Fi - F

    solar_to_home = np.minimum(S, L)
    solar_to_grid = np.minimum(S - solar_to_home, Fe)
    solar_to_bat = S - solar_to_home - solar_to_grid
    grid_to_home = np.minimum(Fi, L - solar_to_home)
    grid_to_bat = Fi - grid_to_home
    bat_to_home = L - solar_to_home - grid_to_home
    bat_to_grid = Fe - solar_to_grid

    charge = solar_to_bat + grid_to_bat
    discharge = bat_to_home + bat_to_grid
    B = charge - discharge

    # Cross-check against the closed form. Should match to float precision;
    # a real gap here means a bug in the decomposition above, not a data
    # issue.
    B_closed_form = S - L + F
    closed_form_diff = np.abs(B - B_closed_form)

    cap_kwh = power_kw * interval_hours
    cap_violation = np.maximum(charge, discharge) - cap_kwh
    n_cap_violations = int(np.sum(cap_violation > _EPS_KWH))
    max_cap_violation = float(np.max(cap_violation)) if len(cap_violation) else 0.0

    n_simultaneous = int(np.sum((charge > _EPS_KWH) & (discharge > _EPS_KWH)))

    # solar_to_home is allowed to be negative (parasitic overnight draw);
    # everything else should be non-negative by construction.
    unexpected_negative = (
        (solar_to_grid < -_EPS_KWH) | (solar_to_bat < -_EPS_KWH)
        | (grid_to_home < -_EPS_KWH) | (grid_to_bat < -_EPS_KWH)
        | (bat_to_home < -_EPS_KWH) | (bat_to_grid < -_EPS_KWH)
    )
    n_unexpected_negative = int(np.sum(unexpected_negative))

    n_negative_load = int(np.sum(L < -_EPS_KWH))

    diagnostics = {
        'max_closed_form_diff_kwh': float(np.max(closed_form_diff)) if len(closed_form_diff) else 0.0,
        'n_power_cap_violations': n_cap_violations,
        'max_power_cap_violation_kwh': max(max_cap_violation, 0.0),
        'n_simultaneous_charge_discharge': n_simultaneous,
        'n_unexpected_negative': n_unexpected_negative,
        'n_negative_load': n_negative_load,
    }
    return B, diagnostics


def iter_scenario_dirs(runs_root: Path, run_name: Optional[str]):
    """Yield (run_name, scenario_dir) for every scenario directory found."""
    if run_name is not None:
        run_dirs = [runs_root / run_name]
        if not run_dirs[0].is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dirs[0]}")
    else:
        if not runs_root.is_dir():
            raise FileNotFoundError(f"Runs root not found: {runs_root}")
        run_dirs = sorted(p for p in runs_root.iterdir() if p.is_dir())

    for run_dir in run_dirs:
        scenarios_dir = run_dir / 'scenarios'
        if not scenarios_dir.is_dir():
            continue
        for scen_dir in sorted(scenarios_dir.iterdir()):
            if scen_dir.is_dir():
                yield run_dir.name, scen_dir


def process_scenario(
    run_name: str, scen_dir: Path, paths: Paths, check_only: bool,
) -> Optional[ScenarioDiagnostics]:
    """Process one scenario directory.

    Returns None if the scenario was skipped (no flows.parquet, or no
    battery). Raises on any error found while processing a scenario that
    should have been processable; the caller catches this per-scenario so
    one bad scenario doesn't abort the whole batch.
    """
    flows_path = scen_dir / 'flows.parquet'
    if not flows_path.exists():
        return None

    summary_path = scen_dir / 'summary.txt'
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{scen_dir}: flows.parquet present but summary.txt missing. "
            f"Cannot confirm battery spec without it."
        )

    meta = parse_summary(summary_path)
    if meta is None:
        return None  # no battery: skip, not an error

    flows_df = pd.read_parquet(flows_path)
    flows_index = flows_df.index
    site_ids = list(flows_df.columns)

    diag = ScenarioDiagnostics(n_sites=len(site_ids), n_intervals=len(flows_index))
    out_cols: dict[str, np.ndarray] = {}

    for site_id in site_ids:
        raw = load_raw_solar_load(site_id, meta.year, paths.edp_clean_dir)
        _check_alignment(raw.index, flows_index, site_id)

        S = raw['S'].to_numpy(dtype='float64')
        L = raw['L'].to_numpy(dtype='float64')
        F = flows_df[site_id].to_numpy(dtype='float64')

        B, site_diag = reconstruct_battery_flow(S, L, F, meta.power_kw)

        diag.max_closed_form_diff_kwh = max(
            diag.max_closed_form_diff_kwh, site_diag['max_closed_form_diff_kwh'])
        diag.n_power_cap_violations += site_diag['n_power_cap_violations']
        diag.max_power_cap_violation_kwh = max(
            diag.max_power_cap_violation_kwh, site_diag['max_power_cap_violation_kwh'])
        diag.n_simultaneous_charge_discharge += site_diag['n_simultaneous_charge_discharge']
        diag.n_unexpected_negative += site_diag['n_unexpected_negative']
        diag.n_negative_load += site_diag['n_negative_load']
        diag.note(
            site_id,
            site_diag['n_power_cap_violations'],
            site_diag['n_simultaneous_charge_discharge'],
            site_diag['n_unexpected_negative'],
            site_diag['n_negative_load'],
        )

        out_cols[site_id] = B.astype('float32')

    if not check_only:
        out_df = pd.DataFrame(out_cols, index=flows_index)
        out_df.index.name = 'datetime_aest'
        out_path = scen_dir / 'battery_flows.parquet'
        out_df.to_parquet(out_path, compression='snappy')

    return diag


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct battery_flows.parquet for historical "
                    "battery-cohort runs from raw EDP traces and flows.parquet.",
    )
    parser.add_argument('run_name', nargs='?', default=None,
                        help="Process only this run under data/output/runs/. "
                             "Default: process every run found there.")
    parser.add_argument('--project-root', type=Path, default=None,
                        help="Override project root (default: cwd)")
    parser.add_argument('--runs-root', type=Path, default=None,
                        help="Override the runs directory "
                             "(default: <project-root>/data/output/runs)")
    parser.add_argument('--check-only', action='store_true', default=False,
                        help="Compute and print diagnostics without writing "
                             "battery_flows.parquet. Useful for a dry run "
                             "over a large tree before committing to writes.")
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    root = args.project_root or Path.cwd()
    runs_root = args.runs_root or (root / 'data' / 'output' / 'runs')

    print(f"Runs root: {runs_root}")
    if args.check_only:
        print("--check-only: computing diagnostics, not writing output")
    print()

    n_processed = 0
    n_skipped_no_flows_or_battery = 0
    failures: list[tuple[str, str]] = []

    for run_name, scen_dir in iter_scenario_dirs(runs_root, args.run_name):
        scenario_id = scen_dir.name
        try:
            diag = process_scenario(run_name, scen_dir, paths, args.check_only)
        except (FileNotFoundError, ValueError) as e:
            print(f"  FAIL  {run_name}/{scenario_id}: {e}", file=sys.stderr)
            failures.append((f"{run_name}/{scenario_id}", str(e)))
            continue

        if diag is None:
            n_skipped_no_flows_or_battery += 1
            continue

        n_processed += 1
        print(f"  OK    {run_name}/{scenario_id}: {diag.summary_line()}")
        if diag.sites_with_issues:
            shown = diag.sites_with_issues[:10]
            more = f" (+{len(diag.sites_with_issues) - 10} more)" \
                if len(diag.sites_with_issues) > 10 else ""
            print(f"          sites with issues: {shown}{more}")

    print()
    print(f"Processed: {n_processed}")
    print(f"Skipped (no flows.parquet or no battery): {n_skipped_no_flows_or_battery}")
    print(f"Failed: {len(failures)}")
    if failures:
        print("\nFailed scenarios:", file=sys.stderr)
        for sid, msg in failures:
            print(f"  {sid}: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
