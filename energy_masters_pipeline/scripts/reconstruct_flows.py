"""Actual-data flow decomposition and SOC reconstruction.

Given the three independently measured
series (solar, net grid, battery flow) for a real site, allocates them
into the full SiteTraces-style flow decomposition and reconstructs SOC,
with explicit diagnostics on where the allocation rule's assumptions
don't fit the data (rather than silently clipping).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

RECONCILIATION_TOLERANCE_KWH = 1e-9
"""bat_to_grid_a vs bat_to_grid_b should agree to floating-point
precision -- a discrepancy above this is a
code bug, not a data-quality signal."""


@dataclass
class FlowDiagnostics:
    """Per-flow negative-value characterization (pre-clip) and clipped
    energy (post-clip), plus the reconciliation and SOC ledgers."""
    negative_interval_counts: dict = field(default_factory=dict)
    negative_volume_kwh: dict = field(default_factory=dict)
    negative_worst_kwh: dict = field(default_factory=dict)
    clipped_volume_kwh: dict = field(default_factory=dict)

    total_intervals: int = 0
    """Total interval count for this site's valid period(s), for scaling
    all the counts below (e.g. '96 of 12483 intervals')."""

    gross_load_negative_intervals: int = 0
    gross_load_negative_volume_kwh: float = 0.0
    gross_load_negative_worst_kwh: float = 0.0
    """Informational only, not an error signal by itself: L = G + S - B
    coming out negative is legitimate and expected whenever it is fully
    explained by solar's own negative reading (parasitic inverter draw
    overnight) -- see sol_to_home_unexplained_negative_* below for the
    actual error signal.
    """

    sol_to_home_unexplained_negative_intervals: int = 0
    sol_to_home_unexplained_negative_volume_kwh: float = 0.0
    sol_to_home_unexplained_negative_worst_kwh: float = 0.0
    """sol_to_home = min(S, L) coming out negative WHILE S itself is >= 0
    is the real fundamental-inconsistency signal, not L<0 by itself.
    Negative sol_to_home is only physically legitimate when solar's own
    reading is negative (parasitic draw) -- a generating source cannot
    have negative consumption. When S >= 0 but sol_to_home < 0 anyway, it
    means L < S < 0 is false but L < S regardless of sign, i.e. gross
    load came out more negative than solar's own reading justifies: the
    three inputs conflict before any allocation choice is made, not a
    benign parasitic-draw case.
    """

    bat_to_grid_max_abs_discrepancy_kwh: float = 0.0
    bat_to_grid_reconciled: bool = True

    soc_overflow_kwh: float = 0.0
    soc_overflow_intervals: int = 0
    soc_underflow_kwh: float = 0.0
    soc_underflow_intervals: int = 0

    total_charge_attempted_dc_kwh: float = 0.0
    total_discharge_attempted_dc_kwh: float = 0.0
    """Total DC-side charge/discharge energy the reconstruction attempted
    to move into/out of the battery over the whole valid period (i.e. the
    sum of the two terms in the SOC update, each accumulated separately
    rather than netted per-interval). These are the natural denominators
    for judging whether soc_overflow_kwh/soc_underflow_kwh are a small or
    large fraction of total battery throughput -- an absolute kWh number
    on its own doesn't say whether it's negligible or substantial.
    """

    power_cap_charge_violation_kwh: float = 0.0
    power_cap_charge_violation_intervals: int = 0
    power_cap_discharge_violation_kwh: float = 0.0
    power_cap_discharge_violation_intervals: int = 0


@dataclass
class ReconstructedFlows:
    site_id: str
    sol_to_home_kwh: pd.Series
    sol_to_bat_kwh: pd.Series
    sol_to_grid_kwh: pd.Series
    bat_to_home_kwh: pd.Series
    grid_to_home_kwh: pd.Series
    grid_to_bat_kwh: pd.Series
    bat_to_grid_kwh: pd.Series
    curtail_kwh: pd.Series  # always zero; kept for SiteTraces field parity
    soc_kwh: pd.Series
    diagnostics: FlowDiagnostics


def _characterize_negatives(name: str, arr: np.ndarray, diag: FlowDiagnostics) -> None:
    neg_mask = arr < 0.0
    n_neg = int(neg_mask.sum())
    diag.negative_interval_counts[name] = n_neg
    diag.negative_volume_kwh[name] = float(-arr[neg_mask].sum()) if n_neg else 0.0
    diag.negative_worst_kwh[name] = float(arr.min()) if len(arr) else 0.0


def _clip_and_record(name: str, arr: np.ndarray, diag: FlowDiagnostics) -> np.ndarray:
    clipped = np.clip(arr, 0.0, None)
    removed = float((arr - clipped).sum())  # arr - clipped is <= 0 everywhere; this is negative of clipped volume
    diag.clipped_volume_kwh[name] = float(-removed)
    return clipped


def reconstruct_flows(
    site_id: str,
    solar_kwh: pd.Series,
    grid_kwh: pd.Series,
    battery_kwh: pd.Series,
    capacity_kwh: float,
    charge_eff: float,
    discharge_eff: float,
    initial_soc_kwh: float,
    max_power_kwh_per_interval: Optional[float] = None,
) -> ReconstructedFlows:
    """Reconstruct the full flow decomposition and SOC trace for one site.

    Args:
        solar_kwh, grid_kwh, battery_kwh: the three independently measured
            (already masked/interpolated) series. grid_kwh: +import,
            -export. battery_kwh: +charge, -discharge. Must share an index.
        capacity_kwh, charge_eff, discharge_eff: from
            estimate_battery_capacity.py (post capacity_review.py
            resolution for capacity_kwh).
        initial_soc_kwh: from estimate_battery_capacity.py's
            real_soc_trace, at the start of this series.
        max_power_kwh_per_interval: from estimate_battery_max_power.py, if
            available. Optional so this can run before that estimate
            exists; power-cap diagnostics are skipped (left at zero) if
            not provided, not silently treated as "no violations found".
    """
    if not (solar_kwh.index.equals(grid_kwh.index) and solar_kwh.index.equals(battery_kwh.index)):
        raise ValueError(f"{site_id}: solar_kwh, grid_kwh, battery_kwh must share the same index")

    index = solar_kwh.index
    S = solar_kwh.to_numpy(dtype=float, copy=False)
    G = grid_kwh.to_numpy(dtype=float, copy=False)
    B = battery_kwh.to_numpy(dtype=float, copy=False)

    L = G + S - B  # gross load, derived -- not independently measured

    diag = FlowDiagnostics()
    diag.total_intervals = len(index)

    l_negative_mask = L < 0.0
    diag.gross_load_negative_intervals = int(l_negative_mask.sum())
    diag.gross_load_negative_volume_kwh = float(-L[l_negative_mask].sum()) if diag.gross_load_negative_intervals else 0.0
    diag.gross_load_negative_worst_kwh = float(L.min()) if len(L) else 0.0

    # --- Step-by-step allocation, allowing negatives ---
    s2h = np.minimum(S, L)
    remaining_load = L - s2h

    # The real fundamental-inconsistency signal: sol_to_home negative
    # while solar itself is non-negative. This is NOT the same as L<0
    # (recorded above, informational only) -- L<0 alone is legitimate and
    # expected whenever it's fully explained by solar's own negative
    # (parasitic draw) reading, in which case S <= L and s2h = S, and
    # everything downstream is fine. Only when S >= 0 but s2h < 0 anyway
    # (which happens exactly when L < S, i.e. gross load came out more
    # negative than solar's own reading justifies) is there a genuine
    # problem: no generating source has negative consumption.
    unexplained_mask = (s2h < 0.0) & (S >= 0.0)
    diag.sol_to_home_unexplained_negative_intervals = int(unexplained_mask.sum())
    diag.sol_to_home_unexplained_negative_volume_kwh = (
        float(-s2h[unexplained_mask].sum()) if diag.sol_to_home_unexplained_negative_intervals else 0.0
    )
    diag.sol_to_home_unexplained_negative_worst_kwh = (
        float(s2h[unexplained_mask].min()) if diag.sol_to_home_unexplained_negative_intervals else 0.0
    )

    g_import = np.clip(G, 0.0, None)
    g2h = np.minimum(g_import, remaining_load)
    b2h = remaining_load - g2h  # can be negative if remaining_load < g2h is impossible by construction of min(),
    # but can exceed available discharge -- characterized below via -B comparison, not via sign alone.

    g2b = g_import - g2h

    is_charging = B > 0.0
    s2b = np.where(is_charging, B - g2b, 0.0)

    s2g = S - s2h - s2b

    g_export = np.clip(-G, 0.0, None)
    b_discharge = np.clip(-B, 0.0, None)
    bat_to_grid_a = g_export - s2g
    bat_to_grid_b = b_discharge - b2h

    # --- Reconciliation check: these must agree to floating precision ---
    discrepancy = np.abs(bat_to_grid_a - bat_to_grid_b)
    max_discrepancy = float(discrepancy.max()) if len(discrepancy) else 0.0
    diag.bat_to_grid_max_abs_discrepancy_kwh = max_discrepancy
    diag.bat_to_grid_reconciled = max_discrepancy <= RECONCILIATION_TOLERANCE_KWH
    if not diag.bat_to_grid_reconciled:
        raise AssertionError(
            f"{site_id}: bat_to_grid_a and bat_to_grid_b disagree by up to "
            f"{max_discrepancy:.6f} kWh, exceeding the floating-point tolerance of "
            f"{RECONCILIATION_TOLERANCE_KWH} kWh. This means "
            "solar_kwh/grid_kwh/battery_kwh here are NOT the same already-cleaned series "
            "gross_load was built from -- check the caller is passing the identical "
            "masked/interpolated channels em_ingest.py produced, not a re-derived copy."
        )
    bat_to_grid = bat_to_grid_a  # equivalently bat_to_grid_b, given the check above

    # --- Characterize negatives PRE-CLIP, on every flow that can go negative ---
    for name, arr in [
        ('sol_to_bat', s2b), ('bat_to_home', b2h), ('sol_to_grid', s2g),
        ('grid_to_bat', g2b), ('bat_to_grid', bat_to_grid),
    ]:
        _characterize_negatives(name, arr, diag)
    # sol_to_home is allowed negative (parasitic draw) -- not characterized as an error.

    # --- Clip everything to >= 0 except sol_to_home ---
    sol_to_home = s2h  # left as-is, negative allowed
    sol_to_bat = _clip_and_record('sol_to_bat', s2b, diag)
    sol_to_grid = _clip_and_record('sol_to_grid', s2g, diag)
    bat_to_home = _clip_and_record('bat_to_home', b2h, diag)
    grid_to_home = _clip_and_record('grid_to_home', g2h, diag)  # g2h is min() of two nonnegatives, never negative,
    # but recorded via the same helper for uniformity (clipped_volume will be 0).
    grid_to_bat = _clip_and_record('grid_to_bat', g2b, diag)
    bat_to_grid_clipped = _clip_and_record('bat_to_grid', bat_to_grid, diag)

    curtail = np.zeros(len(index))

    # --- Power-cap diagnostics on the ALLOCATED flows (see spec: this is
    # effectively a restatement of the same reconciliation, not a new
    # source of inconsistency) ---
    if max_power_kwh_per_interval is not None:
        total_charge = sol_to_bat + grid_to_bat
        total_discharge = bat_to_home + bat_to_grid_clipped
        charge_excess = np.clip(total_charge - max_power_kwh_per_interval, 0.0, None)
        discharge_excess = np.clip(total_discharge - max_power_kwh_per_interval, 0.0, None)
        diag.power_cap_charge_violation_kwh = float(charge_excess.sum())
        diag.power_cap_charge_violation_intervals = int((charge_excess > 0.0).sum())
        diag.power_cap_discharge_violation_kwh = float(discharge_excess.sum())
        diag.power_cap_discharge_violation_intervals = int((discharge_excess > 0.0).sum())

    # --- SOC reconstruction ---
    n = len(index)
    soc = np.zeros(n)
    current = initial_soc_kwh
    overflow_kwh = 0.0
    overflow_n = 0
    underflow_kwh = 0.0
    underflow_n = 0
    total_charge_attempted_dc = 0.0
    total_discharge_attempted_dc = 0.0
    for i in range(n):
        charge_component = (sol_to_bat[i] + grid_to_bat[i]) * charge_eff
        discharge_component = (bat_to_home[i] + bat_to_grid_clipped[i]) / discharge_eff
        total_charge_attempted_dc += charge_component
        total_discharge_attempted_dc += discharge_component
        delta = charge_component - discharge_component
        current = current + delta
        if current > capacity_kwh:
            overflow_kwh += current - capacity_kwh
            overflow_n += 1
            current = capacity_kwh
        elif current < 0.0:
            underflow_kwh += -current
            underflow_n += 1
            current = 0.0
        soc[i] = current

    diag.soc_overflow_kwh = overflow_kwh
    diag.soc_overflow_intervals = overflow_n
    diag.soc_underflow_kwh = underflow_kwh
    diag.soc_underflow_intervals = underflow_n
    diag.total_charge_attempted_dc_kwh = total_charge_attempted_dc
    diag.total_discharge_attempted_dc_kwh = total_discharge_attempted_dc

    def _series(arr: np.ndarray, name: str) -> pd.Series:
        return pd.Series(arr, index=index, name=name)

    return ReconstructedFlows(
        site_id=site_id,
        sol_to_home_kwh=_series(sol_to_home, 'sol_to_home_kwh'),
        sol_to_bat_kwh=_series(sol_to_bat, 'sol_to_bat_kwh'),
        sol_to_grid_kwh=_series(sol_to_grid, 'sol_to_grid_kwh'),
        bat_to_home_kwh=_series(bat_to_home, 'bat_to_home_kwh'),
        grid_to_home_kwh=_series(grid_to_home, 'grid_to_home_kwh'),
        grid_to_bat_kwh=_series(grid_to_bat, 'grid_to_bat_kwh'),
        bat_to_grid_kwh=_series(bat_to_grid_clipped, 'bat_to_grid_kwh'),
        curtail_kwh=_series(curtail, 'curtail_kwh'),
        soc_kwh=_series(soc, 'soc_kwh'),
        diagnostics=diag,
    )


def print_diagnostics(result: ReconstructedFlows) -> None:
    d = result.diagnostics
    print(f"=== {result.site_id} flow reconstruction diagnostics ===")
    print(f"gross_load (derived) negative: {d.gross_load_negative_intervals} of "
          f"{d.total_intervals} interval(s), total {d.gross_load_negative_volume_kwh:.3f} kWh, "
          f"worst {d.gross_load_negative_worst_kwh:.3f} kWh "
          "(informational only -- legitimate whenever fully explained by solar's own "
          "negative/parasitic-draw reading; see below for the actual error signal)")
    print(f"sol_to_home UNEXPLAINED negative (solar itself non-negative): "
          f"{d.sol_to_home_unexplained_negative_intervals} of {d.total_intervals} interval(s), "
          f"total {d.sol_to_home_unexplained_negative_volume_kwh:.3f} kWh, "
          f"worst {d.sol_to_home_unexplained_negative_worst_kwh:.3f} kWh "
          f"({'OK -- none' if d.sol_to_home_unexplained_negative_intervals == 0 else 'FUNDAMENTAL DATA INCONSISTENCY'})")
    print(f"bat_to_grid reconciliation: max |a - b| = "
          f"{d.bat_to_grid_max_abs_discrepancy_kwh:.2e} kWh  "
          f"({'OK' if d.bat_to_grid_reconciled else 'FAIL -- see error above'})")
    print("\nPre-clip negative-value characterization (allocation-rule mismatches):")
    for name in d.negative_interval_counts:
        n = d.negative_interval_counts[name]
        vol = d.negative_volume_kwh[name]
        worst = d.negative_worst_kwh[name]
        print(f"  {name:15s} {n:6d} of {d.total_intervals} interval(s)  total {vol:8.3f} kWh  worst {worst:8.3f} kWh")
    print("\nClipped energy (post-clip, all flows except sol_to_home):")
    for name, vol in d.clipped_volume_kwh.items():
        print(f"  {name:15s} {vol:8.3f} kWh clipped")
    overflow_pct = (100 * d.soc_overflow_kwh / d.total_charge_attempted_dc_kwh
                    if d.total_charge_attempted_dc_kwh > 0 else float('nan'))
    underflow_pct = (100 * d.soc_underflow_kwh / d.total_discharge_attempted_dc_kwh
                     if d.total_discharge_attempted_dc_kwh > 0 else float('nan'))
    print(f"\nSOC overflow (attempted charge beyond capacity_kwh): "
          f"{d.soc_overflow_kwh:.3f} kWh across {d.soc_overflow_intervals} of {d.total_intervals} interval(s)  "
          f"-- {overflow_pct:.2f}% of total attempted DC charge ({d.total_charge_attempted_dc_kwh:.1f} kWh)")
    print(f"SOC underflow (attempted discharge below 0): "
          f"{d.soc_underflow_kwh:.3f} kWh across {d.soc_underflow_intervals} of {d.total_intervals} interval(s)  "
          f"-- {underflow_pct:.2f}% of total attempted DC discharge ({d.total_discharge_attempted_dc_kwh:.1f} kWh)")
    if d.power_cap_charge_violation_intervals or d.power_cap_discharge_violation_intervals:
        print(f"\nPower cap violations (on allocated flows):")
        print(f"  charge:    {d.power_cap_charge_violation_kwh:.3f} kWh across "
              f"{d.power_cap_charge_violation_intervals} interval(s)")
        print(f"  discharge: {d.power_cap_discharge_violation_kwh:.3f} kWh across "
              f"{d.power_cap_discharge_violation_intervals} interval(s)")


def run_checks(
    site_id: str,
    summary_csv_path: Path = Path('data/processed/energy_masters/site_summary_results.csv'),
    telemetry_path: Optional[Path] = None,
    review_csv_path: Path = Path('energy_masters_pipeline/battery_capacity_review.csv'),
) -> ReconstructedFlows:
    """Run the full reconstruction for one real site, wiring together
    em_ingest.py (loading), capacity_review.py (the accept/exclude/override
    decision), estimate_battery_capacity.py (capacity, efficiencies,
    initial SOC), and estimate_battery_max_power.py (the power cap).

    Raises if the site has not been reviewed, or has been excluded --
    per capacity_review.py's contract, there is no sensible way to run
    this for a site without a resolved capacity decision.
    """
    # Local imports so this module doesn't hard-require the rest of the
    # project's package layout just to be imported for its own logic/tests.
    from energy_masters_pipeline.scripts.em_ingest import load_valid_periods_from_summary, load_em_site_component_flows, DEFAULT_TELEMETRY_PATH
    from energy_masters_pipeline.scripts.estimate_battery_capacity import estimate_battery_capacity
    from energy_masters_pipeline.scripts.estimate_battery_max_power import estimate_max_power
    from energy_masters_pipeline.scripts import capacity_review as cr

    if telemetry_path is None:
        telemetry_path = DEFAULT_TELEMETRY_PATH

    review = cr.get_review(site_id, review_csv_path)
    if review is None:
        raise ValueError(
            f"{site_id}: not yet reviewed in {review_csv_path}. Run "
            "estimate_battery_capacity.py's plot, review it by eye, and add a row "
            "via capacity_review.upsert_review() before reconstructing flows."
        )
    if review.decision == 'exclude':
        raise ValueError(f"{site_id}: excluded ({review.reason!r}); not reconstructing flows.")

    valid_periods = load_valid_periods_from_summary(site_id, summary_csv_path)

    print(f"Loading component flows for {site_id}...")
    flows_df = load_em_site_component_flows(site_id, valid_periods, telemetry_path)
    solar_kwh = flows_df['pv_dc_kwh']
    grid_kwh = flows_df['ac_load_net_kwh']
    battery_kwh = flows_df['battery_kwh']

    print("Estimating battery capacity and efficiency...")
    cap_estimate = estimate_battery_capacity(site_id, valid_periods, summary_csv_path, telemetry_path)
    capacity_kwh = cr.resolved_capacity_kwh(cap_estimate.capacity_kwh, review)
    charge_eff = cap_estimate.charge_eff
    discharge_eff = cap_estimate.discharge_eff
    initial_soc_kwh = cap_estimate.initial_soc_kwh
    if review.decision == 'override':
        # An overridden capacity may not agree with the capacity the
        # automated real_soc_trace was rebased against -- clip rather
        # than silently hand a possibly out-of-range value downstream.
        initial_soc_kwh = float(np.clip(initial_soc_kwh, 0.0, capacity_kwh))

    print("Estimating max AC power...")
    power_estimate = estimate_max_power(site_id, battery_kwh)

    print("Reconstructing flows...")
    result = reconstruct_flows(
        site_id=site_id, solar_kwh=solar_kwh, grid_kwh=grid_kwh, battery_kwh=battery_kwh,
        capacity_kwh=capacity_kwh, charge_eff=charge_eff, discharge_eff=discharge_eff,
        initial_soc_kwh=initial_soc_kwh,
        max_power_kwh_per_interval=power_estimate.max_power_kwh_per_interval,
    )
    print(f"\ncapacity_kwh used: {capacity_kwh:.3f}  "
          f"(review decision: {review.decision}, reason: {review.reason!r})")
    print(f"charge_eff=discharge_eff: {charge_eff:.4f}  initial_soc_kwh: {initial_soc_kwh:.3f}  "
          f"max_power_kw: {power_estimate.max_power_kw:.3f}\n")
    print_diagnostics(result)
    return result


def _run_synthetic_self_test() -> None:
    """Mechanics-only self-test: the reconciliation identity and the
    clipping/overflow ledgers, NOT a stand-in for real-data validation.
    Kept separate from run_checks so `python -m ... EM-427` runs the real
    driver by default; pass --synthetic to run this instead.
    """
    np.random.seed(0)
    n = 2000
    idx = pd.date_range('2025-01-01', periods=n, freq='5min')
    S = np.clip(np.random.normal(1.0, 1.5, n), 0, None)
    G = np.random.normal(0.0, 1.0, n)
    B = np.random.normal(0.0, 0.5, n)
    solar = pd.Series(S, index=idx)
    grid = pd.Series(G, index=idx)
    battery = pd.Series(B, index=idx)

    result = reconstruct_flows(
        site_id='SYNTH-001', solar_kwh=solar, grid_kwh=grid, battery_kwh=battery,
        capacity_kwh=10.0, charge_eff=0.95, discharge_eff=0.95,
        initial_soc_kwh=5.0, max_power_kwh_per_interval=0.5,
    )
    print_diagnostics(result)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--synthetic':
        _run_synthetic_self_test()
    elif len(sys.argv) > 1:
        run_checks(sys.argv[1])
    else:
        print("Usage: python3 -m energy_masters_pipeline.scripts.reconstruct_flows EM-427")
        print("       python3 -m energy_masters_pipeline.scripts.reconstruct_flows --synthetic")