"""Interactive visualisation of the actual-data flow reconstruction for a
single real Energy Masters site.

Adapted from viz_perfect_foresight_lp.py. Reused nearly unchanged: the day
slicing, the SOC+charge/discharge panel, the time-axis formatting, the two
matplotlib figures (FLOWS, PRICES), and the interactive day-by-day loop.
Changed for this context:

  - ONE result, not a pair. viz_perfect_foresight_lp.py compares
    self_consumption vs perfect_foresight_lp (two strategies). This script
    runs the existing visualization on the actual-data traces, so there is only one
    dataset here: what actually happened, reconstructed by
    reconstruct_flows.py. EMAnalysis therefore carries a single
    actual_result (a SiteTraces built from the reconstruction), not
    sc_result/lp_result.
  - SOC bounds are [0, capacity_kwh], not [backup_reserve_floor,
    capacity_kwh] -- this capacity is already the usable swing (from
    estimate_battery_capacity.py), not a BatterySpec modelling input.
    A BatterySpec with backup_reserve_pct=0 is constructed purely so
    _plot_soc_panel (copied verbatim) can be reused without modification.
  - Verification swaps the LP-specific checks (SOC bounds from
    optimisation, export-limit compliance, LP-vs-SC bill comparison) for
    the flow-reconstruction diagnostics from reconstruct_flows.py
    (negative-value characterization, bat_to_grid reconciliation, SOC
    overflow/underflow) -- same PASS/FAIL/INFO style, different substance.
  - CLI drops year (not applicable here): just site_id, since valid
    periods come from site_summary_results.csv and tariffs/state are the
    fixed EM defaults from em_ingest.py. --export-limit-kw IS present
    (unlike the original script's fixed default) but defaults to an
    auto-estimate from the site's own actual export data rather than a
    fixed 10 kW -- see estimate_export_limit.py.

Usage (from project root):
    python3 -m energy_masters_pipeline.scripts.viz_energy_masters EM-427
    python3 -m energy_masters_pipeline.scripts.viz_energy_masters EM-427 --simulate-lp
    python3 -m energy_masters_pipeline.scripts.viz_energy_masters EM-427 --simulate-lp \
        --export-limit-kw 10   # override the auto-estimate

--simulate-lp swaps which behaviour FLOWS/PRICES plot: instead of the
actual (Amber-coordinated) reconstruction, it runs perfect_foresight_lp on
this site's real gross_load/pv_dc trace with a battery matching the
estimated size/power/efficiencies, using the actual reconstruction's own
start/end SOC as the LP's soc_init_kwh/soc_final_kwh (the same fair-
comparison logic viz_perfect_foresight_lp.py uses with self_consumption's
end-of-period SOC). The actual reconstruction and its bill are still
computed either way, and the verification block prints a counterfactual
comparison (bills, self-sufficiency, import/export totals) when this flag
is used.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.core.paths import default_paths, Paths
from model.core.scenario import BatterySpec
from model.core.tariff import (
    NetworkTariff, RetailTariff, load_network_tariff, load_retail_tariff,
)
from model.core.prices import build_price_vectors
from model.core.site_result import SiteBill, SiteTraces
from model.components.billing import compute_bill
from model.components.self_consumption import run_sc_with_battery

from energy_masters_pipeline.scripts.em_ingest import (
    STATE, DEFAULT_NETWORK_TARIFF_ID, DEFAULT_RETAIL_TARIFF_ID,
    DEFAULT_TELEMETRY_PATH, DEFAULT_SITE_SUMMARY_PATH, INTERVAL_HOURS,
    load_valid_periods_from_summary, load_em_site_component_flows,
    _load_nem_prices_multi_year, _full_grid, _to_datetime_aest,
)
from energy_masters_pipeline.scripts.estimate_battery_capacity import estimate_battery_capacity
from energy_masters_pipeline.scripts.estimate_battery_max_power import estimate_max_power
from energy_masters_pipeline.scripts.estimate_export_limit import estimate_export_limit
from energy_masters_pipeline.scripts.reconstruct_flows import reconstruct_flows, ReconstructedFlows, FlowDiagnostics
from energy_masters_pipeline.scripts import capacity_review as cr


# Colour palette (matching viz_perfect_foresight_lp / viz_self_consumption)
COLOUR_SOLAR = '#F5A623'    # yellow/orange
COLOUR_BATTERY = '#5DA13E'  # green
COLOUR_GRID = '#888888'     # grey
COLOUR_LOAD = '#1F77B4'     # blue
COLOUR_CURTAIL = '#FBD38D'  # light orange (unused here -- curtail is always
                            # zero for actual-data reconstruction, kept only
                            # for visual parity with the LP script's panel)

# Scenario colours, used ONLY for the actual-vs-LP comparison figure.
# Deliberately outside the flow palette above: in the comparison view the
# pane and the price line carry the flow identity (export vs import), so
# hue is free to carry the SCENARIO instead. Reusing COLOUR_LOAD/
# COLOUR_BATTERY here (as the first version of the compare figure did)
# collides with "blue means home, green means battery" everywhere else.
COLOUR_ACTUAL = '#7B4FA8'   # purple -- actual (Amber-coordinated)
COLOUR_LP = '#0E9594'       # teal   -- perfect foresight (LP)

_C_PER_DOLLAR = 100.0       # $/kWh -> c/kWh for price axes

BATTERY_MODE = 'Actual (Amber-coordinated)'

# The analysis index timezone. em_ingest converts every trace to fixed-offset
# Etc/GMT-10, and every "day" downstream is a midnight in THIS zone: the day
# browser, the daily attribution grouping, the top-days list. Derived from
# _to_datetime_aest rather than written as a literal so the two cannot drift.
# Note this is NOT the site's local zone -- an SA site is analysed on AEST day
# boundaries, which is a pre-existing convention, not something introduced here.
ANALYSIS_TZ = _to_datetime_aest(pd.DatetimeIndex([pd.Timestamp('2000-01-01')])).tz

# Modelling defaults, in one greppable place because several scripts call
# load_and_analyze and MUST NOT drift apart on these. run_cohort_attribution.py
# and analyse_daily_value_gaps.py both call it; if these lived only in this
# module's argparse defaults, a cohort run and a CLI run of the same site would
# silently produce different numbers. Both analysis scripts print the
# curtailment mode and c_deg actually used on every run as a cross-check.
#
# PROVISIONAL: 0.039 is an estimate from cleaned site data via sweep_c_deg.py 
DEFAULT_SIMULATE_LP = True
DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS = None   # None = curtailment pinned to zero
DEFAULT_C_DEG_PER_KWH = 0.039

CURTAILMENT_ARG_HELP = (
    "LP curtailment mode: 'pinned' (default -- curtailment forced to zero, the "
    "like-for-like comparison against a coordinator with no inverter access), "
    "'free' (allowed at zero cost), or a number = dollars per kWh (NOT cents) "
    "to price it in the LP objective. See perfect_foresight_lp_spec.md, "
    "'Curtailment cost modes'."
)


def resolve_curtailment_arg(value: Optional[str]) -> Optional[float]:
    """Parse a --curtailment CLI value into curtail_cost_per_kwh_dollars.

    Shared by every script exposing --curtailment so the three modes cannot
    drift apart between them. Raises ValueError; callers wrap in
    parser.error() to get the usual argparse behaviour.
    """
    if value is None:
        return DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS
    v = value.strip().lower()
    if v == 'pinned':
        return None
    if v == 'free':
        return 0.0
    try:
        cost = float(v)
    except ValueError:
        raise ValueError(
            f"--curtailment must be 'pinned', 'free', or a number in dollars "
            f"per kWh; got {value!r}"
        )
    if cost < 0:
        raise ValueError(f"--curtailment cost must be >= 0; got {cost}")
    return cost


def describe_curtailment(cost: Optional[float]) -> str:
    """One-line description of the resolved curtailment mode, for run output."""
    if cost is None:
        return 'curtailment: PINNED to zero'
    if cost == 0.0:
        return 'curtailment: allowed, zero cost'
    return f'curtailment: allowed, priced at ${cost:.4f}/kWh'


# ---------------------------------------------------------------------------
# Data bundle and analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EMAnalysis:
    site_id: str
    trace: pd.DataFrame        # pv_dc_kwh, ac_load_net_kwh, battery_kwh, gross_load_kwh
    actual_result: SiteTraces  # from reconstructed_flows_to_site_traces
    actual_bill: SiteBill
    diagnostics: FlowDiagnostics
    p_imp: pd.Series           # $/kWh
    p_exp: pd.Series           # $/kWh
    battery_spec: BatterySpec  # backup_reserve_pct=0, for _plot_soc_panel reuse
    network_tariff: NetworkTariff
    retail_tariff: RetailTariff
    state: str
    interval_hours: float
    capacity_review_decision: str
    capacity_review_reason: str
    capacity_review_overrides: str
    display_mode: str = 'actual'   # 'actual' or 'lp' -- which result FLOWS/PRICES plot
    sc_result: Optional[SiteTraces] = None
    sc_bill: Optional[SiteBill] = None
    lp_result: Optional[SiteTraces] = None
    lp_bill: Optional[SiteBill] = None
    lp_backend: Optional[str] = None
    lp_solve_seconds: Optional[float] = None
    export_limit_kw: Optional[float] = None
    soc_init_kwh: Optional[float] = None
    soc_final_kwh: Optional[float] = None
    curtail_cost_per_kwh_dollars: Optional[float] = 0.0
    prices_mwh: Optional[pd.Series] = None
    """Wholesale RRP in $/MWh over this site's valid period(s).

    Same series compute_bill is given, so anything derived from it is
    consistent with the bills by construction. Added for the horizon
    sweep, which needs raw RRP to drive a ForecastProvider; p_imp and
    p_exp are already stacked and cannot serve.

    Optional only so that no existing construction site is forced to
    change; load_and_analyze always populates it.
    """

    @property
    def displayed_result(self) -> SiteTraces:
        if self.display_mode == 'lp':
            if self.lp_result is None:
                raise ValueError(f"{self.site_id}: display_mode='lp' but lp_result is None "
                                  "-- pass simulate_lp=True to load_and_analyze")
            return self.lp_result
        return self.actual_result

    @property
    def mode_label(self) -> str:
        return 'Perfect Foresight (simulated)' if self.display_mode == 'lp' else BATTERY_MODE


def reconstructed_flows_to_site_traces(result: ReconstructedFlows) -> SiteTraces:
    """Adapter: ReconstructedFlows -> SiteTraces, so compute_bill and the
    plotting code (which both expect SiteTraces, matching
    self_consumption.py's / perfect_foresight_lp's output shape) can be
    reused unchanged.

    import_kwh/export_kwh are derived from the same allocated flows
    reconstruct_flows.py already produced (grid_to_home+grid_to_bat,
    sol_to_grid+bat_to_grid respectively) -- not re-derived from the raw
    ac_load_net channel -- so this SiteTraces is internally consistent
    with the flow decomposition it's built from by construction.
    """
    import_kwh = result.grid_to_home_kwh + result.grid_to_bat_kwh
    export_kwh = result.sol_to_grid_kwh + result.bat_to_grid_kwh
    return SiteTraces(
        site_id=result.site_id,
        import_kwh=import_kwh,
        export_kwh=export_kwh,
        battery_soc_kwh=result.soc_kwh,
        sol_to_home_kwh=result.sol_to_home_kwh,
        sol_to_bat_kwh=result.sol_to_bat_kwh,
        sol_to_grid_kwh=result.sol_to_grid_kwh,
        bat_to_home_kwh=result.bat_to_home_kwh,
        grid_to_home_kwh=result.grid_to_home_kwh,
        grid_to_bat_kwh=result.grid_to_bat_kwh,
        bat_to_grid_kwh=result.bat_to_grid_kwh,
        curtail_kwh=result.curtail_kwh,
    )


def run_self_consumption_comparison(
    site_id: str,
    solar_kwh: pd.Series,
    gross_load_kwh: pd.Series,
    battery_spec: BatterySpec,
    soc_init_kwh: float,
    interval_hours: float,
) -> SiteTraces:
    """Self-consumption dispatch on the same real trace, for a three-way
    comparison against actual and (optionally) perfect-foresight.

    Uses run_sc_with_battery (not the public self_consumption() wrapper),
    since the wrapper always starts from the backup-reserve floor rather
    than an explicit SOC -- here we want the SAME real starting SOC as the
    actual reconstruction and the LP simulation, for a genuinely fair
    three-way comparison, not three different starting conditions.

    grid_to_bat and bat_to_grid are always zero under self-consumption
    (never charges from grid, never exports battery to grid) -- matching
    self_consumption.py's own documented contract, not assumed here.
    """
    soc_arr, flows = run_sc_with_battery(
        solar_kwh.to_numpy(dtype=float, copy=False),
        gross_load_kwh.to_numpy(dtype=float, copy=False),
        battery_spec, soc_init_kwh, interval_hours,
    )
    idx = solar_kwh.index

    def s(arr: np.ndarray) -> pd.Series:
        return pd.Series(arr, index=idx)

    zeros = pd.Series(0.0, index=idx)
    grid_to_home = s(flows['grid_to_home'])
    grid_to_bat = s(flows['grid_to_bat'])  # always zero, per self_consumption.py's contract
    sol_to_grid = s(flows['sol_to_grid'])
    bat_to_grid = zeros                    # always zero, per self_consumption.py's contract

    return SiteTraces(
        site_id=site_id,
        import_kwh=grid_to_home + grid_to_bat,
        export_kwh=sol_to_grid + bat_to_grid,
        battery_soc_kwh=s(soc_arr),
        sol_to_home_kwh=s(flows['sol_to_home']),
        sol_to_bat_kwh=s(flows['sol_to_bat']),
        sol_to_grid_kwh=sol_to_grid,
        bat_to_home_kwh=s(flows['bat_to_home']),
        grid_to_home_kwh=grid_to_home,
        grid_to_bat_kwh=grid_to_bat,
        bat_to_grid_kwh=bat_to_grid,
        curtail_kwh=s(flows['curtail']),
    )


def _load_em_price_vectors(
    site_id: str,
    valid_periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    paths: Paths,
) -> tuple[pd.Series, pd.Series]:
    """Derived (p_imp, p_exp) $/kWh across the site's valid period(s),
    same construction em_ingest.py's reconcile_prices uses internally
    (build_price_vectors against the multi-year NEM RRP series), but
    returning just the price vectors rather than the Amber-comparison
    diagnostic -- this script wants the prices to plot, not to re-run the
    settlement reconciliation check (already done in em_ingest.py).
    """
    years = sorted({s.year for s, _ in valid_periods} | {e.year for _, e in valid_periods})
    prices_mwh_full = _load_nem_prices_multi_year(STATE, years, paths)

    p_imp_frames, p_exp_frames = [], []
    for start, end in valid_periods:
        full_index = _full_grid(start, end)
        aest_index = _to_datetime_aest(full_index)
        prices_mwh = prices_mwh_full.reindex(aest_index)
        if prices_mwh.isna().any():
            n_missing = int(prices_mwh.isna().sum())
            raise ValueError(
                f"{site_id}: {n_missing} NEM price intervals missing over "
                f"{start} to {end}."
            )
        p_imp, p_exp = build_price_vectors(
            aest_index, prices_mwh, network_tariff, retail_tariff, STATE, INTERVAL_HOURS,
        )
        p_imp_frames.append(pd.Series(p_imp.to_numpy(), index=aest_index))
        p_exp_frames.append(pd.Series(p_exp.to_numpy(), index=aest_index))

    return pd.concat(p_imp_frames), pd.concat(p_exp_frames)


def load_and_analyze(
    site_id: str,
    summary_csv_path: Path = DEFAULT_SITE_SUMMARY_PATH,
    telemetry_path: Path = DEFAULT_TELEMETRY_PATH,
    review_csv_path: Path = Path('energy_masters_pipeline/battery_capacity_review.csv'),
    network_tariff_id: str = DEFAULT_NETWORK_TARIFF_ID,
    retail_tariff_id: str = DEFAULT_RETAIL_TARIFF_ID,
    paths: Optional[Paths] = None,
    simulate_lp: bool = DEFAULT_SIMULATE_LP,
    export_limit_kw: Optional[float] = None,
    curtail_cost_per_kwh_dollars: Optional[float] = DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS,
    c_deg_per_kwh_override: Optional[float] = None,
) -> EMAnalysis:
    """Load, reconstruct, and bill one real site's actual behaviour.

    Raises (via reconstruct_flows.run_checks's own logic, replicated
    here) if the site is not yet reviewed or has been excluded in
    battery_capacity_review.csv.

    If simulate_lp is True, ALSO runs perfect_foresight_lp on the same
    real gross_load/pv_dc trace using the estimated battery spec (size,
    power, efficiencies from estimate_battery_capacity.py /
    estimate_battery_max_power.py), with soc_init_kwh/soc_final_kwh set to
    the ACTUAL reconstruction's own start/end SOC -- same "fair
    comparison" logic viz_perfect_foresight_lp.py uses with
    self_consumption's end-of-period SOC as the LP's target. The returned
    EMAnalysis has display_mode='lp' in that case, so FLOWS/PRICES plot
    the perfect-foresight result; the actual reconstruction and its bill
    are still computed and retained for the verification block's
    counterfactual comparison.

    curtail_cost_per_kwh_dollars: passed straight through to
    perfect_foresight_lp. Defaults to DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS
    (None, i.e. curtailment PINNED to zero), because Amber cannot curtail
    without inverter access, so a benchmark that may curtail is not a like
    for like comparison. May raise RuntimeError if pinning makes the LP
    infeasible for a given site/period. 0.0 allows curtailment at zero
    cost; a positive value prices it in the LP objective. See
    perfect_foresight_lp_spec.md, "Curtailment cost modes".

    c_deg_per_kwh_override: None (default) uses DEFAULT_C_DEG_PER_KWH.
    A value overrides it for this call only -- used by sweep_c_deg.py to
    solve the LP at several candidate c_deg values without duplicating the
    battery spec / price vector / export limit setup logic.

    export_limit_kw: None (the default) falls back to export_kw_override
    in the review CSV, and failing that auto-estimates the site's real
    export limit from its own measured export data via
    estimate_export_limit.py, using the same iterative gap-based ceiling
    detection as estimate_battery_max_power.py. Pass an explicit value to
    override (e.g. a known network agreement limit). Auto-estimating
    matters: a fixed default (10 kW, carried over from
    viz_perfect_foresight_lp.py's own default) could be tighter OR looser
    than any given site's real limit, and either mismatch distorts the LP
    comparison -- tighter artificially caps its export revenue, looser
    lets it export more than the real system ever could.
    """
    paths = paths or default_paths()

    review = cr.get_review(site_id, review_csv_path)
    if review is None:
        raise ValueError(
            f"{site_id}: not yet reviewed in {review_csv_path}. See "
            "capacity_review.py / estimate_battery_capacity.py's plot."
        )
    if review.decision == 'exclude':
        raise ValueError(f"{site_id}: excluded ({review.reason!r}); not visualising.")

    valid_periods = load_valid_periods_from_summary(site_id, summary_csv_path)
    # Clip BEFORE anything downstream: estimate_battery_capacity takes
    # valid_periods directly, so clipping later would leave the capacity
    # estimate computed over the very stretch the override exists to drop.
    # Dates are interpreted in ANALYSIS_TZ, not in the periods' own zone
    # (which is UTC), so the boundary lands on the same midnight every other
    # day boundary in the analysis uses.
    valid_periods = cr.apply_period_overrides(valid_periods, review, ANALYSIS_TZ)

    flows_df = load_em_site_component_flows(site_id, valid_periods, telemetry_path)
    solar_kwh = flows_df['pv_dc_kwh']
    grid_kwh = flows_df['ac_load_net_kwh']
    battery_kwh = flows_df['battery_kwh']

    cap_estimate = estimate_battery_capacity(site_id, valid_periods, summary_csv_path, telemetry_path)
    capacity_kwh = cr.resolved_capacity_kwh(cap_estimate.capacity_kwh, review)
    charge_eff = cap_estimate.charge_eff
    discharge_eff = cap_estimate.discharge_eff
    initial_soc_kwh = float(np.clip(cap_estimate.initial_soc_kwh, 0.0, capacity_kwh))

    power_estimate = estimate_max_power(site_id, battery_kwh)
    # One resolved kW value feeding both the reconstruction's per-interval
    # limit and the LP's BatterySpec, so the two cannot describe different
    # inverters.
    power_kw = cr.resolved_power_kw(power_estimate.max_power_kw, review)
    max_power_kwh_per_interval = power_kw * INTERVAL_HOURS

    result = reconstruct_flows(
        site_id=site_id, solar_kwh=solar_kwh, grid_kwh=grid_kwh, battery_kwh=battery_kwh,
        capacity_kwh=capacity_kwh, charge_eff=charge_eff, discharge_eff=discharge_eff,
        initial_soc_kwh=initial_soc_kwh,
        max_power_kwh_per_interval=max_power_kwh_per_interval,
    )
    actual_result = reconstructed_flows_to_site_traces(result)

    network_tariff = load_network_tariff(network_tariff_id, paths.tariffs_dir)
    retail_tariff = load_retail_tariff(retail_tariff_id, paths.tariffs_dir)
    p_imp, p_exp = _load_em_price_vectors(
        site_id, valid_periods, network_tariff, retail_tariff, paths,
    )
    prices_mwh_full = _load_nem_prices_multi_year(
        STATE, sorted({s.year for s, _ in valid_periods} | {e.year for _, e in valid_periods}), paths,
    )
    # compute_bill validates that the price index matches the target index
    # EXACTLY (see prices.py's _validate_prices) -- prices_mwh_full is the
    # whole concatenated calendar year(s), not restricted to this site's
    # valid period(s), so it must be reindexed down to actual_result's
    # index (same index as trace/solar_kwh/grid_kwh) before billing, the
    # same reindex step _load_em_price_vectors already does internally for
    # p_imp/p_exp -- this was missing here, which is exactly what caused
    # the "prices len=210240, index len=58008" error.
    prices_mwh_for_bill = prices_mwh_full.reindex(actual_result.import_kwh.index)
    if prices_mwh_for_bill.isna().any():
        n_missing = int(prices_mwh_for_bill.isna().sum())
        raise ValueError(
            f"{site_id}: {n_missing} NEM price intervals missing when reindexed to the "
            "site's valid period(s) -- check the NEM price file(s) actually cover this span."
        )
    actual_bill = compute_bill(
        actual_result, prices_mwh_for_bill, network_tariff, retail_tariff, STATE, INTERVAL_HOURS,
    )

    battery_spec = BatterySpec(
        capacity_kwh=capacity_kwh,
        backup_reserve_pct=0.0,  # this capacity is already the usable swing -- see note above
        power_kw=power_kw,
        charge_efficiency=charge_eff,
        discharge_efficiency=discharge_eff,
        c_deg_per_kwh=(
            DEFAULT_C_DEG_PER_KWH if c_deg_per_kwh_override is None
            else c_deg_per_kwh_override
        ),  # Battery degradation cost, also serving as LP degeneracy regularisation
        # (at 0.0 the regularisation is disabled entirely, risking spurious simultaneous
        # charge/discharge in the solve). Overridable per-call via c_deg_per_kwh_override,
        # e.g. for sweep_c_deg.py.
    )
    gross_load_kwh = grid_kwh + solar_kwh - battery_kwh

    # self_consumption / perfect_foresight_lp expect load_kwh >= 0 always
    # (only solar is allowed negative, for parasitic inverter draw) --
    # perfect_foresight_lp's own _validate_inputs enforces this and raises
    # if violated; self_consumption's public wrapper does too, but
    # run_sc_with_battery (used below, since it needs an explicit starting
    # SOC) does NOT validate at all, so it would otherwise silently run
    # with undefined behaviour on these intervals rather than failing loudly.
    #
    # gross_load_kwh CAN legitimately go slightly negative for us: whenever a negative solar reading (parasitic
    # draw) is large enough that S <= L, sol_to_home = min(S, L) = S and
    # everything downstream is fine -- clipping L up to 0 in that case does
    # NOT change sol_to_home (S is already <= the new 0), it only slightly
    # increases the small residual load a few kWh have to come from
    # elsewhere. This is a negligible, well-understood adjustment ONLY
    # under that condition -- check it explicitly rather than assuming it,
    # since if any negative-load interval is NOT explained this way
    # (sol_to_home_unexplained_negative_intervals > 0 in the actual
    # reconstruction's own diagnostics), clipping here would be papering
    # over a genuine data inconsistency rather than a benign one.
    if result.diagnostics.sol_to_home_unexplained_negative_intervals > 0:
        print(
            f"WARNING: {site_id} has "
            f"{result.diagnostics.sol_to_home_unexplained_negative_intervals} unexplained "
            "negative-load interval(s) in the actual reconstruction "
            "-- clipping gross_load to >=0 for the self-consumption/LP simulation inputs is "
            "papering over a genuine inconsistency for those intervals, not just the benign "
            "parasitic-draw case. Treat the simulation comparison as more approximate for this site."
        )
    gross_load_for_sim_kwh = gross_load_kwh.clip(lower=0.0)
    n_clipped = int((gross_load_kwh < 0.0).sum())
    if n_clipped:
        print(f"Note: clipped {n_clipped} of {len(gross_load_kwh)} gross_load interval(s) "
              f"(total {-gross_load_kwh[gross_load_kwh < 0].sum():.3f} kWh) up to 0 for the "
              "self-consumption/LP simulation inputs only -- the actual reconstruction's own "
              "trace and verification keep the true (unclipped) value.")

    # Self-consumption comparison: ALWAYS computed (not gated by
    # simulate_lp), matching viz_perfect_foresight_lp.py's own pattern of
    # always having a self_consumption baseline available for comparison,
    # even though only the LP result gets plotted there. Uses the SAME
    # real starting SOC as the actual reconstruction (via
    # run_sc_with_battery, not the public self_consumption() wrapper,
    # which would otherwise force a reserve-floor start) so this is a
    # genuinely fair three-way comparison, not three different starting
    # conditions.
    sc_result = run_self_consumption_comparison(
        site_id, solar_kwh, gross_load_for_sim_kwh, battery_spec, initial_soc_kwh, INTERVAL_HOURS,
    )
    sc_bill = compute_bill(
        sc_result, prices_mwh_for_bill, network_tariff, retail_tariff, STATE, INTERVAL_HOURS,
    )

    lp_result = lp_bill = lp_backend_name = lp_solve_seconds = None
    soc_init_kwh = soc_final_kwh = None
    if simulate_lp:
        # Local import: only needed for this optional path, and keeps this
        # module importable even if perfect_foresight_lp / its solver
        # backend isn't set up in an environment that only wants the
        # actual-data reconstruction.
        import time as _time
        from model.components.perfect_foresight_lp import perfect_foresight_lp, lp_backend

        soc_init_kwh = float(actual_result.battery_soc_kwh.iloc[0])
        soc_final_kwh = float(actual_result.battery_soc_kwh.iloc[-1])
        # Clamp tiny float drift into bounds, same reasoning as
        # viz_perfect_foresight_lp.py's run_lp_analysis.
        soc_init_kwh = min(max(soc_init_kwh, 0.0), capacity_kwh)
        soc_final_kwh = min(max(soc_final_kwh, 0.0), capacity_kwh)

        # Export limit precedence: an explicit --export-limit-kw beats the
        # review CSV, which beats the automated estimate. Ordered that way so
        # a one-off command line experiment does not require editing the
        # review file, while a reviewed value still wins over an estimate the
        # reviewer has already judged wrong. The estimator is skipped entirely
        # when either override is present rather than being run and discarded.
        if export_limit_kw is not None:
            export_limit_was_estimated = False
            print(f"Export limit from --export-limit-kw: {export_limit_kw:.3f} kW "
                  "(overrides both the review CSV and the automated estimate).")
        elif review.export_kw_override is not None:
            export_limit_kw = review.export_kw_override
            export_limit_was_estimated = False
            print(f"Export limit from review override: {export_limit_kw:.3f} kW "
                  f"({review_csv_path.name}). Pass --export-limit-kw to override.")
        else:
            export_limit_was_estimated = True
            export_limit_estimate = estimate_export_limit(site_id, grid_kwh, INTERVAL_HOURS)
            export_limit_kw = export_limit_estimate.export_limit_kw
            print(f"Export limit auto-estimated from actual data: {export_limit_kw:.3f} kW "
                  f"({export_limit_estimate.n_intervals_near_ceiling} of "
                  f"{export_limit_estimate.n_intervals_total} nonzero-export intervals near "
                  f"ceiling, {export_limit_estimate.n_outliers_excluded} outlier(s) excluded). "
                  "Pass --export-limit-kw, or set export_kw_override, to override.")
            if export_limit_estimate.n_intervals_near_ceiling < 5:
                print("NOTE: fewer than 5 intervals near the ceiling -- this site may rarely "
                      "or never have actually been export-limited, so treat this estimate as "
                      "weakly supported; consider overriding with a known network agreement "
                      "value if you have one.")

        export_limit_kwh = export_limit_kw * INTERVAL_HOURS
        _lp_t0 = _time.perf_counter()
        lp_result = perfect_foresight_lp(
            site_id, solar_kwh, gross_load_for_sim_kwh,
            p_imp, p_exp, battery_spec,
            soc_init_kwh=soc_init_kwh, soc_final_kwh=soc_final_kwh,
            export_limit_kwh_per_interval=export_limit_kwh,
            interval_hours=INTERVAL_HOURS,
            curtail_cost_per_kwh_dollars=curtail_cost_per_kwh_dollars,
        )
        lp_solve_seconds = _time.perf_counter() - _lp_t0
        lp_backend_name = lp_backend()
        lp_bill = compute_bill(
            lp_result, prices_mwh_for_bill, network_tariff, retail_tariff, STATE, INTERVAL_HOURS,
        )

    trace = pd.DataFrame({
        'pv_dc_kwh': solar_kwh,
        'ac_load_net_kwh': grid_kwh,
        'battery_kwh': battery_kwh,
        'gross_load_kwh': gross_load_kwh,
    })

    return EMAnalysis(
        site_id=site_id, trace=trace, actual_result=actual_result, actual_bill=actual_bill,
        diagnostics=result.diagnostics, p_imp=p_imp, p_exp=p_exp,
        battery_spec=battery_spec, network_tariff=network_tariff, retail_tariff=retail_tariff,
        state=STATE, interval_hours=INTERVAL_HOURS,
        capacity_review_decision=review.decision, capacity_review_reason=review.reason,
        capacity_review_overrides=cr.describe_overrides(review),
        display_mode='lp' if simulate_lp else 'actual',
        sc_result=sc_result, sc_bill=sc_bill,
        lp_result=lp_result, lp_bill=lp_bill,
        lp_backend=lp_backend_name, lp_solve_seconds=lp_solve_seconds,
        export_limit_kw=export_limit_kw if simulate_lp else None,
        soc_init_kwh=soc_init_kwh, soc_final_kwh=soc_final_kwh,
        curtail_cost_per_kwh_dollars=curtail_cost_per_kwh_dollars if simulate_lp else 0.0,
        prices_mwh=prices_mwh_for_bill,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _print_verification(analysis: EMAnalysis) -> None:
    """Print startup verification stats, same PASS/FAIL/INFO style as
    viz_perfect_foresight_lp.py's _print_verification, but checking
    reconstruction-quality diagnostics instead of LP optimisation
    constraints (there is no LP here)."""
    a = analysis
    r = a.actual_result
    d = a.diagnostics
    solar = a.trace['pv_dc_kwh']
    load = a.trace['gross_load_kwh']
    cap = a.battery_spec.capacity_kwh
    tol = 1e-6

    def status(ok: bool) -> str:
        return 'PASS' if ok else 'FAIL'

    print()
    print(f"=== Verification: site {a.site_id} (actual-data reconstruction) ===")
    T = len(r.import_kwh)
    print(f"T (intervals): {T:,}")
    print(f"Battery: {cap:.3f} kWh, {a.battery_spec.power_kw:.3f} kW, "
          f"eff {a.battery_spec.charge_efficiency:.4f}/{a.battery_spec.discharge_efficiency:.4f}")
    print(f"Capacity review: {a.capacity_review_decision} ({a.capacity_review_reason!r})")
    print(f"  {a.capacity_review_overrides}")
    print(f"Network: {a.network_tariff.name} | Retail: {a.retail_tariff.name} "
          f"({a.retail_tariff.type})")
    print()

    print("--- Flow invariants (atol 1e-6) ---")
    load_recon = (r.sol_to_home_kwh + r.bat_to_home_kwh + r.grid_to_home_kwh).to_numpy()
    diff = np.abs(load_recon - load.to_numpy()).max()
    print(f"Load balance   max|recon-load|: {diff:.3e}  {status(diff < tol)}")

    solar_recon = (r.sol_to_home_kwh + r.sol_to_bat_kwh + r.sol_to_grid_kwh).to_numpy()
    diff = np.abs(solar_recon - solar.to_numpy()).max()
    print(f"Solar balance  max|recon-solar|: {diff:.3e}  {status(diff < tol)}  "
          "(curtail is always 0 here -- no curtailment logic for actual data)")

    diff = np.abs((r.grid_to_home_kwh + r.grid_to_bat_kwh - r.import_kwh).to_numpy()).max()
    print(f"Import recon   max|g2h+g2b-import|: {diff:.3e}  {status(diff < tol)}")

    diff = np.abs((r.sol_to_grid_kwh + r.bat_to_grid_kwh - r.export_kwh).to_numpy()).max()
    print(f"Export recon   max|s2g+b2g-export|: {diff:.3e}  {status(diff < tol)}")
    print()

    print("--- Reconstruction-quality diagnostics ---")
    print(f"gross_load negative (informational): {d.gross_load_negative_intervals} of "
          f"{d.total_intervals}, total {d.gross_load_negative_volume_kwh:.3f} kWh")
    print(f"sol_to_home UNEXPLAINED negative: {d.sol_to_home_unexplained_negative_intervals} of "
          f"{d.total_intervals}, total {d.sol_to_home_unexplained_negative_volume_kwh:.3f} kWh  "
          f"{status(d.sol_to_home_unexplained_negative_intervals == 0)}")
    print(f"bat_to_grid reconciliation: max|a-b|={d.bat_to_grid_max_abs_discrepancy_kwh:.2e} kWh  "
          f"{status(d.bat_to_grid_reconciled)}")
    overflow_pct = (100 * d.soc_overflow_kwh / d.total_charge_attempted_dc_kwh
                    if d.total_charge_attempted_dc_kwh > 0 else float('nan'))
    underflow_pct = (100 * d.soc_underflow_kwh / d.total_discharge_attempted_dc_kwh
                     if d.total_discharge_attempted_dc_kwh > 0 else float('nan'))
    print(f"SOC overflow:  {d.soc_overflow_kwh:.3f} kWh across {d.soc_overflow_intervals} "
          f"intervals ({overflow_pct:.2f}% of attempted DC charge)")
    print(f"SOC underflow: {d.soc_underflow_kwh:.3f} kWh across {d.soc_underflow_intervals} "
          f"intervals ({underflow_pct:.2f}% of attempted DC discharge)")
    print()

    print("--- Bill (actual, Amber-coordinated) ---")
    print(f"household_total: ${a.actual_bill.household_total:,.2f}")
    print()

    print("--- Period totals (kWh) ---")
    print(f"Solar generated:  {solar.sum():>11.1f}")
    print(f"Gross load:       {load.sum():>11.1f}")
    print(f"Solar -> home:    {r.sol_to_home_kwh.sum():>11.1f}")
    print(f"Solar -> battery: {r.sol_to_bat_kwh.sum():>11.1f}")
    print(f"Solar -> grid:    {r.sol_to_grid_kwh.sum():>11.1f}")
    print(f"Grid -> battery:  {r.grid_to_bat_kwh.sum():>11.1f}")
    print(f"Battery -> home:  {r.bat_to_home_kwh.sum():>11.1f}")
    print(f"Battery -> grid:  {r.bat_to_grid_kwh.sum():>11.1f}")
    print(f"Grid -> home:     {r.grid_to_home_kwh.sum():>11.1f}")
    print(f"Total import:     {r.import_kwh.sum():>11.1f}")
    print(f"Total export:     {r.export_kwh.sum():>11.1f}")
    self_suff = 100 * (1 - r.import_kwh.sum() / load.sum()) if load.sum() else float('nan')
    print(f"Self-sufficiency: {self_suff:>10.1f}%")
    print()

    print("--- Energy balance identities across scenarios (kWh) ---")
    print("Flow-side identity: import - export = (load - solar) + (C - Q) + curtail")
    print("  where C = sol_to_bat + grid_to_bat (AC charge), Q = bat_to_home + bat_to_grid (AC discharge).")
    print("This is just a recombination of the flow invariants already verified exactly above, "
          "so the residual should be ~0 for ALL scenarios regardless of any SOC clipping -- C and Q "
          "are the recorded AC-side flows, independent of what the downstream SOC bookkeeping does "
          "with them.")
    print()
    print("SOC-side identity: recorded delta_soc (= soc_final - soc_init) should equal "
          "C*charge_eff - Q/discharge_eff (the DC-side energy balance). This is where "
          "self-consumption/perfect-foresight (bounds enforced live during simulation, so this "
          "holds exactly) differ from actual (bounds enforced by CLIPPING an already-computed "
          "trace after the fact) -- actual's residual here should equal its own net "
          "overflow-minus-underflow.")
    print()

    load_for_sim = a.trace['gross_load_kwh'].clip(lower=0.0)  # matches what sc/lp were actually fed
    solar_sum = solar.sum()

    def _scenario_row(traces: SiteTraces, load_series: pd.Series) -> dict:
        C = float((traces.sol_to_bat_kwh + traces.grid_to_bat_kwh).sum())
        Q = float((traces.bat_to_home_kwh + traces.bat_to_grid_kwh).sum())
        curtail = float(traces.curtail_kwh.sum())
        load_sum = float(load_series.sum())
        import_sum = float(traces.import_kwh.sum())
        export_sum = float(traces.export_kwh.sum())
        observed_flow = import_sum - export_sum
        predicted_flow = (load_sum - solar_sum) + (C - Q) + curtail
        soc_init = float(traces.battery_soc_kwh.iloc[0])
        soc_final = float(traces.battery_soc_kwh.iloc[-1])
        recorded_delta_soc = soc_final - soc_init
        predicted_delta_soc = C * a.battery_spec.charge_efficiency - Q / a.battery_spec.discharge_efficiency
        return dict(
            load=load_sum, C=C, Q=Q, C_minus_Q=C - Q, curtail=curtail,
            import_sum=import_sum, export_sum=export_sum,
            observed_flow=observed_flow, predicted_flow=predicted_flow,
            flow_residual=observed_flow - predicted_flow,
            recorded_delta_soc=recorded_delta_soc, predicted_delta_soc=predicted_delta_soc,
            soc_residual=recorded_delta_soc - predicted_delta_soc,
        )

    scenario_rows_traces = {'actual': r, 'self-consumption': a.sc_result}
    scenario_rows = {'actual': _scenario_row(r, load)}
    scenario_rows['self-consumption'] = _scenario_row(a.sc_result, load_for_sim)
    if a.lp_result is not None:
        scenario_rows['perfect foresight'] = _scenario_row(a.lp_result, load_for_sim)
        scenario_rows_traces['perfect foresight'] = a.lp_result

    names = list(scenario_rows.keys())
    header = f"{'':22s}" + "".join(f"{n:>18s}" for n in names)
    print(header)
    field_labels = [
        ('load', 'load (as fed)'),
        ('C', 'AC charge C'),
        ('Q', 'AC discharge Q'),
        ('C_minus_Q', 'C - Q'),
        ('curtail', 'curtail'),
        ('import_sum', 'import'),
        ('export_sum', 'export'),
        ('observed_flow', 'import - export (obs)'),
        ('predicted_flow', 'predicted (load-solar)+(C-Q)+curtail'),
        ('flow_residual', 'flow residual'),
        ('recorded_delta_soc', 'recorded delta_soc'),
        ('predicted_delta_soc', 'predicted delta_soc'),
        ('soc_residual', 'soc residual'),
    ]
    for key, label in field_labels:
        row = f"{label:22s}" + "".join(f"{scenario_rows[n][key]:>18,.3f}" for n in names)
        print(row)
    print()
    actual_soc_residual = scenario_rows['actual']['soc_residual']
    net_overflow_underflow = d.soc_overflow_kwh - d.soc_underflow_kwh
    print(f"actual soc_residual ({actual_soc_residual:.3f}) vs net overflow-underflow "
          f"({net_overflow_underflow:.3f} = {d.soc_overflow_kwh:.3f} - {d.soc_underflow_kwh:.3f}): "
          f"{'MATCH' if abs(actual_soc_residual - (-net_overflow_underflow)) < 1e-2 else 'MISMATCH -- worth investigating'} "
          "(overflow means MORE was attempted than delivered, so it should show up as recorded_delta_soc "
          "being LOWER than predicted by the overflow amount, and higher by the underflow amount -- "
          "i.e. soc_residual should be approximately -(overflow - underflow))")
    print()

    print("--- Simultaneous charge+discharge check (per interval, not visible in period totals) ---")
    print("Period-total C/Q consistency (checked above) is blind to WITHIN-interval timing -- a "
          "degenerate LP solution could charge and discharge the same interval (e.g. to burn off "
          "round-trip losses deliberately during negative import prices) without changing any of "
          "the totals above. This checks per-interval, directly.")
    tol = 1e-6
    for name, traces in scenario_rows_traces.items():
        charge_t = traces.sol_to_bat_kwh + traces.grid_to_bat_kwh
        discharge_t = traces.bat_to_home_kwh + traces.bat_to_grid_kwh
        both = (charge_t > tol) & (discharge_t > tol)
        n_both = int(both.sum())
        overlap_kwh = float(np.minimum(charge_t[both], discharge_t[both]).sum())
        verdict = ('none -- no evidence of the degenerate simultaneous charge/discharge pattern'
                   if n_both == 0 else
                   f'NONZERO ({overlap_kwh:.3f} kWh overlap) -- inspect these intervals directly; '
                   'this is the pattern that caused billing/degeneracy issues in earlier work, '
                   'and may mean c_deg_per_kwh=0.015 is not enough (earlier project notes used 0.02)')
        print(f"{name:20s} {n_both:6d} interval(s) with charge>0 AND discharge>0 simultaneously  {verdict}")
    print()

    print("--- Simultaneous import+export check (per interval, distinct from battery cycling) ---")
    print("A single grid connection can only report one net direction at a time -- import_kwh and "
          "export_kwh both positive in the same interval is unphysical regardless of what the "
          "battery is doing (e.g. grid_to_home>0 while sol_to_grid>0, arbitraging a negative import "
          "price against a high export price with the battery untouched that interval). Checked "
          "separately from the charge/discharge check above since one does not imply the other.")
    for name, traces in scenario_rows_traces.items():
        both_ie = (traces.import_kwh > tol) & (traces.export_kwh > tol)
        n_both_ie = int(both_ie.sum())
        overlap_ie_kwh = float(np.minimum(traces.import_kwh[both_ie], traces.export_kwh[both_ie]).sum())
        verdict_ie = ('none -- no evidence of simultaneous import/export'
                      if n_both_ie == 0 else
                      f'NONZERO ({overlap_ie_kwh:.3f} kWh overlap) -- inspect these intervals directly, '
                      'this is unphysical for a single grid connection regardless of battery state')
        print(f"{name:20s} {n_both_ie:6d} interval(s) with import>0 AND export>0 simultaneously  {verdict_ie}")
    print()


    print("(self-consumption and perfect-foresight both use the SAME real gross_load/pv_dc "
          "trace and the same starting SOC as the actual reconstruction -- self-consumption "
          "and actual therefore differ only in DISPATCH POLICY, not starting conditions)")
    sc = a.sc_result
    sc_self_suff = 100 * (1 - sc.import_kwh.sum() / load.sum()) if load.sum() else float('nan')
    actual_total = a.actual_bill.household_total
    sc_total = a.sc_bill.household_total
    print(f"{'':20s} {'actual':>14s} {'self-consumption':>18s}" +
          (f" {'perfect foresight':>18s}" if a.lp_result is not None else ""))
    row_fmt = "{:20s} {:14,.2f} {:18,.2f}" + (" {:18,.2f}" if a.lp_result is not None else "")
    bill_row = ["household bill ($)", actual_total, sc_total]
    suff_row = ["self-sufficiency (%)", self_suff, sc_self_suff]
    import_row = ["total import (kWh)", r.import_kwh.sum(), sc.import_kwh.sum()]
    export_row = ["total export (kWh)", r.export_kwh.sum(), sc.export_kwh.sum()]
    if a.lp_result is not None:
        lp = a.lp_result
        lp_self_suff = 100 * (1 - lp.import_kwh.sum() / load.sum()) if load.sum() else float('nan')
        bill_row.append(a.lp_bill.household_total)
        suff_row.append(lp_self_suff)
        import_row.append(lp.import_kwh.sum())
        export_row.append(lp.export_kwh.sum())
    print(row_fmt.format(*bill_row))
    print(row_fmt.format(*suff_row))
    print(row_fmt.format(*import_row))
    print(row_fmt.format(*export_row))
    print()
    if sc_total <= actual_total + 1e-4:
        print(f"self-consumption vs actual: self-consumption is cheaper by "
              f"${actual_total - sc_total:,.2f}")
    else:
        print(f"self-consumption vs actual: self-consumption is MORE EXPENSIVE by "
              f"${sc_total - actual_total:,.2f}. Not necessarily a bug: under a "
              "wholesale_exposed tariff a price-aware 'actual' policy can legitimately beat "
              "self-consumption, since self_consumption.py's own contract forces "
              "grid_to_bat and bat_to_grid to always be zero -- it can never arbitrage via "
              "cheap/negative-price grid charging or high-price battery export, both of "
              "which 'actual' is doing here (see the self-sufficiency row: self-consumption "
              "is HIGHEST despite the WORST bill, consistent with prioritising self-"
              "sufficiency over profit under volatile wholesale pricing, not a sign of "
              "anything wrong).")

    if a.lp_result is not None:
        lp = a.lp_result
        lp_total = a.lp_bill.household_total
        actual_max_export_kw = float(r.export_kwh.max()) / a.interval_hours
        print(f"backend: {a.lp_backend}   solve time: {a.lp_solve_seconds:.2f}s   "
              f"export limit assumed for LP: {a.export_limit_kw:.1f} kW "
              f"({a.export_limit_kw * a.interval_hours:.4f} kWh/interval)")
        print(describe_curtailment(a.curtail_cost_per_kwh_dollars))
        print(f"c_deg_per_kwh used for LP: {a.battery_spec.c_deg_per_kwh:.4f} $/kWh")
        print(f"actual max export observed: {actual_max_export_kw:.2f} kW  "
              f"({'the assumed limit is TIGHTER than what was actually observed -- the LP may be more constrained than the real system, which could understate its true export volume/value' if actual_max_export_kw > a.export_limit_kw + 1e-6 else 'the assumed limit is not binding relative to what was observed'})")
        print(f"SOC: init={a.soc_init_kwh:.3f}  final={a.soc_final_kwh:.3f}  "
              "(both taken from the actual reconstruction's own start/end SOC)")
        print(f"perfect foresight <= actual: {'PASS' if lp_total <= actual_total + 1e-4 else 'FAIL'}  "
              f"(theoretical saving ${actual_total - lp_total:,.2f} vs actual -- this is an "
              "UPPER BOUND on what better dispatch could have achieved, not a claim about "
              "what any real retail plan or coordination scheme could realistically capture)")
        print(f"perfect foresight <= self-consumption: "
              f"{'PASS' if lp_total <= sc_total + 1e-4 else 'FAIL'}  "
              f"(saving vs self-consumption: ${sc_total - lp_total:,.2f})")

        vcr_denominator = lp_total - sc_total
        if abs(vcr_denominator) < 1e-6:
            print("Value Capture Ratio: undefined (perfect-foresight and self-consumption "
                  "bills are equal to within tolerance -- no separating value between the "
                  "two baselines to capture a share of)")
        else:
            vcr = (actual_total - sc_total) / vcr_denominator
            print(f"Value Capture Ratio (actual-sc)/(pf-sc): {vcr:.3f}  "
                  "(0 = actual performs like self-consumption; 1 = actual fully captures the "
                  "value perfect foresight could achieve beyond self-consumption; outside "
                  "[0,1] means actual did worse than self-consumption, or better than "
                  "perfect foresight -- the latter should not happen and is worth "
                  "investigating if seen)")
        print()


# ---------------------------------------------------------------------------
# Plotting helpers (copied near-verbatim from viz_perfect_foresight_lp.py --
# see that file for the original design notes on twin-axis handling etc.)
# ---------------------------------------------------------------------------

def _day_slice(analysis: EMAnalysis, day: pd.Timestamp):
    a = analysis
    day_start = day.normalize()
    day_end = day_start + pd.Timedelta(days=1)
    idx = a.trace.index
    mask = (idx >= day_start) & (idx < day_end)
    t = idx[mask]
    k = 1.0 / a.interval_hours

    r = a.displayed_result
    d = dict(
        solar=a.trace.loc[mask, 'pv_dc_kwh'].to_numpy() * k,
        load=a.trace.loc[mask, 'gross_load_kwh'].to_numpy() * k,
        s2h=r.sol_to_home_kwh.loc[mask].to_numpy() * k,
        s2b=r.sol_to_bat_kwh.loc[mask].to_numpy() * k,
        s2g=r.sol_to_grid_kwh.loc[mask].to_numpy() * k,
        b2h=r.bat_to_home_kwh.loc[mask].to_numpy() * k,
        b2g=r.bat_to_grid_kwh.loc[mask].to_numpy() * k,
        g2h=r.grid_to_home_kwh.loc[mask].to_numpy() * k,
        g2b=r.grid_to_bat_kwh.loc[mask].to_numpy() * k,
        curtail=r.curtail_kwh.loc[mask].to_numpy() * k,  # always zero here
    )
    soc = r.battery_soc_kwh.loc[mask].to_numpy()
    p_imp_c = a.p_imp.loc[mask].to_numpy() * _C_PER_DOLLAR
    p_exp_c = a.p_exp.loc[mask].to_numpy() * _C_PER_DOLLAR
    return t, d, soc, p_imp_c, p_exp_c, day_start, day_end, mask.sum()


def _reapply_labels(axes, titles, ylabels, xlabel='Hour of day'):
    for ax, title, yl in zip(axes, titles, ylabels):
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_ylabel(yl)
    axes[-1].set_xlabel(xlabel)


def _format_time_axis(ax, t, day_start, day_end):
    tz = t.tz if len(t) else None
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=tz))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3), tz=tz))
    ax.set_xlim(day_start, day_end)


# ---------------------------------------------------------------------------
# Price axis scaling
# ---------------------------------------------------------------------------
#
# Wholesale-exposed retail prices span roughly -100 c/kWh to several thousand
# c/kWh, and the interesting behaviour is at BOTH ends: the sign of a small
# negative export price decides whether exporting is a cost or a credit, while
# a 2,000 c/kWh spike decides most of the day's value. A linear axis renders
# the small values as a flat line against the spike; a pure log axis cannot
# show zero or negatives at all.
#
# symlog solves this: linear within +/- _PRICE_LINTHRESH_C of zero, logarithmic
# outside it. Two things the bare scale does not give us:
#
#  1. The transition point has to be visible. Explicit horizontal rules at 0
#     and at the threshold were tried and removed: on a four-pane figure they
#     added more clutter than they resolved. The changing TICK SPACING carries
#     it instead, which is quieter and always present.
#  2. Limits are SNAPPED to a small standard set rather than fitted to each
#     day. Free autoscale means the eye recalibrates on every step of the day
#     browser, which defeats browsing; fixed limits waste the pane on calm
#     days. Snapping means the axis only ever takes a few recognisable forms.
#
# Note the snap stops are in RETAIL c/kWh, not wholesale: these prices carry
# network and retail layers and loss factors on top of the wholesale rate, so
# they legitimately exceed the wholesale market cap. The top stop is therefore
# fitted to the data rather than hardcoded to a market ceiling.

_PRICE_LINTHRESH_C = 100.0
_PRICE_SNAP_STOPS_C = (100.0, 200.0, 1000.0)
_PRICE_TICK_CANDIDATES_C = (
    -5000.0, -2000.0, -1000.0, -500.0, -200.0, -100.0, -50.0, 0.0,
    50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0,
)


def _price_axis_limits(*price_arrays):
    """Snapped (lo, hi) price-axis limits in c/kWh.

    Uses the smallest standard stop that contains the day's data, falling
    back to a data-fitted limit above the largest stop (the "ceiling" case).
    """
    vals = [np.asarray(p, dtype=float).ravel() for p in price_arrays]
    vals = [v[np.isfinite(v)] for v in vals if v.size]
    if not vals:
        return -_PRICE_LINTHRESH_C, _PRICE_LINTHRESH_C
    allv = np.concatenate(vals)
    vmax = float(np.max(allv))
    vmin = float(np.min(allv))

    hi = next((s for s in _PRICE_SNAP_STOPS_C if vmax <= s), None)
    if hi is None:
        hi = vmax * 1.05
    lo = next((-s for s in _PRICE_SNAP_STOPS_C if vmin >= -s), None)
    if lo is None:
        lo = vmin * 1.05
    return lo, hi


def _apply_price_scale(ax, lo, hi):
    """Apply the symlog price scale with snapped limits and ticks.

    No horizontal rules at zero or at the linear/log threshold: the tick
    spacing already shows where the scale changes, and on a four-pane figure
    the extra lines read as noise.
    """
    ax.set_yscale('symlog', linthresh=_PRICE_LINTHRESH_C, linscale=1.0)
    ax.set_ylim(lo, hi)
    ticks = [v for v in _PRICE_TICK_CANDIDATES_C if lo <= v <= hi]
    if ticks:
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{v:,.0f}" for v in ticks])


def _plot_soc_panel(ax, t, soc, d, battery_spec, interval_hours):
    cap = battery_spec.capacity_kwh
    min_soc = (battery_spec.backup_reserve_pct / 100.0) * cap  # 0 here, by construction

    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    if len(t):
        tnum = mdates.date2num(t)
        w = (interval_hours / 24.0) * 0.45

        ax2.bar(tnum, d['s2b'], width=w, align='center',
                color=COLOUR_SOLAR, edgecolor='none', label='charge: sol_to_bat')
        ax2.bar(tnum, d['g2b'], width=w, bottom=d['s2b'], align='center',
                color=COLOUR_GRID, edgecolor='none', label='charge: grid_to_bat')

        ax2.bar(tnum, -d['b2h'], width=w, align='center',
                color=COLOUR_LOAD, edgecolor='none', label='discharge: bat_to_home')
        ax2.bar(tnum, -d['b2g'], width=w, bottom=-d['b2h'], align='center',
                color=COLOUR_BATTERY, edgecolor='none', label='discharge: bat_to_grid')

        ax2.axhline(0, color='black', linewidth=0.6, alpha=0.4)

    ax2.set_ylabel('Charge (+) / Discharge (\u2212) (kW)')
    pwr = battery_spec.power_kw * 1.1
    ax2.set_ylim(-max(pwr, 0.1), max(pwr, 0.1))

    ax.plot(t, soc, color=COLOUR_BATTERY, linewidth=1.8, label='SOC', zorder=10)
    ax.axhline(cap, ls='--', color='black', lw=0.8, alpha=0.5)
    ax.axhline(min_soc, ls='--', color='red', lw=0.8, alpha=0.5)
    ax.set_ylim(0, cap * 1.1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=7, ncol=1,
              framealpha=0.9, borderpad=0.4, facecolor='white').set_zorder(20)


def _clear_twins(ax):
    fig = ax.figure
    for other in list(fig.axes):
        if other is ax:
            continue
        if other.get_position().bounds == ax.get_position().bounds:
            other.remove()


def _set_window_title(fig, title):
    mgr = getattr(fig.canvas, 'manager', None)
    if mgr is not None:
        try:
            mgr.set_window_title(title)
        except Exception:
            pass


def _finalise(fig, day_start, suffix='', mode_label=BATTERY_MODE):
    kind = getattr(fig, '_viz_kind', '')
    site = getattr(fig, '_viz_site_id', '')
    fig.suptitle(f"Site {site}  {day_start.date()}  [{kind}]  {suffix} - {mode_label}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])


# ---------------------------------------------------------------------------
# FLOWS figure
# ---------------------------------------------------------------------------

def _setup_flows_figure(site_id, mode_suffix='actual'):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 3]})
    _set_window_title(fig, f'{site_id} \u2014 flows ({mode_suffix})')
    fig._viz_site_id = site_id
    fig._viz_kind = 'Generation, Load & Battery SOC'
    return fig, axes


def _plot_flows_day(axes, analysis, day):
    for ax in axes:
        ax.clear()
    _clear_twins(axes[2])

    t, d, soc, _, _, day_start, day_end, n = _day_slice(analysis, day)
    if n == 0:
        for ax in axes:
            ax.text(0.5, 0.5, f"No data for {day.normalize().date()}",
                    transform=ax.transAxes, ha='center', va='center')
        _finalise(axes[0].figure, day_start, analysis.retail_tariff.name, analysis.mode_label)
        return

    base = np.zeros_like(d['s2h'])
    l1 = d['s2h']
    l2 = d['s2h'] + d['s2b']
    l3 = l2 + d['s2g']
    axes[0].fill_between(t, base, l1, step='post', color=COLOUR_SOLAR, label='sol_to_home')
    axes[0].fill_between(t, l1, l2, step='post', color=COLOUR_BATTERY, label='sol_to_bat')
    axes[0].fill_between(t, l2, l3, step='post', color=COLOUR_GRID, label='sol_to_grid')
    axes[0].step(t, d['solar'], where='post', color=COLOUR_SOLAR, linewidth=1.8,
                 label='gross solar')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    p1 = d['s2h']
    p2 = d['s2h'] + d['b2h']
    p3 = p2 + d['g2h']
    axes[1].fill_between(t, base, p1, step='post', color=COLOUR_SOLAR, label='sol_to_home')
    axes[1].fill_between(t, p1, p2, step='post', color=COLOUR_BATTERY, label='bat_to_home')
    axes[1].fill_between(t, p2, p3, step='post', color=COLOUR_GRID, label='grid_to_home')
    axes[1].step(t, d['load'], where='post', color=COLOUR_LOAD, linewidth=1.8,
                 label='gross load')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    y_max = max(float(np.max(l3)) if len(l3) else 0,
                float(np.max(p3)) if len(p3) else 0,
                float(np.max(d['solar'])) if len(d['solar']) else 0,
                float(np.max(d['load'])) if len(d['load']) else 0, 0.1)
    y_min = min(0.0, float(np.min(d['solar'])) if len(d['solar']) else 0.0)
    axes[0].set_ylim(y_min * 1.05 if y_min < 0 else 0, y_max * 1.05)
    axes[1].set_ylim(0, y_max * 1.05)

    _plot_soc_panel(axes[2], t, soc, d, analysis.battery_spec, analysis.interval_hours)
    axes[2].grid(True, alpha=0.3)
    _format_time_axis(axes[2], t, day_start, day_end)

    _reapply_labels(
        axes,
        ['Solar Generation', 'Serviced Load', 'Battery SOC + Charge/Discharge'],
        ['Power (kW)', 'Power (kW)', 'SOC (kWh)'],
    )
    _finalise(axes[0].figure, day_start, analysis.retail_tariff.name, analysis.mode_label)


# ---------------------------------------------------------------------------
# PRICES figure
# ---------------------------------------------------------------------------

def _setup_prices_figure(site_id, mode_suffix='actual'):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 3]})
    _set_window_title(fig, f'{site_id} \u2014 prices ({mode_suffix})')
    fig._viz_site_id = site_id
    fig._viz_kind = 'Exports, Imports & Battery SOC'
    return fig, axes


def _plot_prices_day(axes, analysis, day):
    for ax in axes:
        ax.clear()
    _clear_twins(axes[0])
    _clear_twins(axes[1])
    _clear_twins(axes[2])

    t, d, soc, p_imp_c, p_exp_c, day_start, day_end, n = _day_slice(analysis, day)
    if n == 0:
        for ax in axes:
            ax.text(0.5, 0.5, f"No data for {day.normalize().date()}",
                    transform=ax.transAxes, ha='center', va='center')
        _finalise(axes[0].figure, day_start, analysis.retail_tariff.name, analysis.mode_label)
        return

    e1 = d['s2g']
    e2 = d['s2g'] + d['b2g']
    axes[0].fill_between(t, 0, e1, step='post', color=COLOUR_SOLAR, label='sol_to_grid')
    axes[0].fill_between(t, e1, e2, step='post', color=COLOUR_BATTERY, label='bat_to_grid')
    axes[0].step(t, e2, where='post', color=COLOUR_BATTERY, linewidth=1.5,
                 label='total export')
    axes[0].set_ylim(0, max(float(np.max(e2)) if len(e2) else 0, 0.1) * 1.1)
    axes[0].grid(True, alpha=0.3)
    axp0 = axes[0].twinx()
    axp0.step(t, p_exp_c, where='post', color=COLOUR_SOLAR, linestyle=':',
              linewidth=1.6, label='export price')
    axp0.set_ylabel('Export price (c/kWh)')
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axp0.get_legend_handles_labels()
    axp0.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    i1 = d['g2h']
    i2 = d['g2h'] + d['g2b']
    axes[1].fill_between(t, 0, i1, step='post', color=COLOUR_LOAD, label='grid_to_home')
    axes[1].fill_between(t, i1, i2, step='post', color=COLOUR_BATTERY, label='grid_to_bat')
    axes[1].step(t, i2, where='post', color=COLOUR_GRID, linewidth=1.5,
                 label='total import')
    axes[1].set_ylim(0, max(float(np.max(i2)) if len(i2) else 0, 0.1) * 1.1)
    axes[1].grid(True, alpha=0.3)
    axp1 = axes[1].twinx()
    axp1.step(t, p_imp_c, where='post', color=COLOUR_GRID, linestyle=':',
              linewidth=1.6, label='import price')
    axp1.set_ylabel('Import price (c/kWh)')
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = axp1.get_legend_handles_labels()
    axp1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    _plot_soc_panel(axes[2], t, soc, d, analysis.battery_spec, analysis.interval_hours)
    axes[2].grid(True, alpha=0.3)
    _format_time_axis(axes[2], t, day_start, day_end)

    _reapply_labels(
        axes,
        ['Export vs Export Price', 'Import vs Import Price',
         'Battery SOC + Charge/Discharge'],
        ['Export (kW)', 'Import (kW)', 'SOC (kWh)'],
    )
    _finalise(axes[0].figure, day_start, analysis.retail_tariff.name, analysis.mode_label)


# ---------------------------------------------------------------------------
# COMPARE figure (actual vs perfect foresight)
# ---------------------------------------------------------------------------

# Actual is drawn as a light filled area to zero on the two FLOW panes, with
# the LP as a heavier line over the top. Breaking the symmetry this way means
# the eye compares a line against a fill edge rather than two lines of equal
# weight, and it reads as "reality, with the benchmark laid on top". Not
# applied to the SOC pane: SOC is a level rather than a rate, so a fill to
# zero there implies a quantity that is not meaningful.
_ACTUAL_FILL_ALPHA = 0.18

# The signed difference band on the flow and SOC panes was tried and turned
# off: with two solid scenario traces already present it obscured them more
# than it clarified the gap. Kept behind a flag rather than deleted so it is
# one edit to re-evaluate. The cumulative-gap pane keeps its own signed infill
# regardless -- there the fill IS the series, not an overlay on top of one.
SHOW_DIFFERENCE_BAND = False


def _signed_difference_band(ax, t, series_actual, series_lp, alpha=0.22):
    """Shade the area between the two scenario traces, tinted by which is on top.

    This is the point of the whole figure: the GAP is made the visual subject
    rather than something inferred by comparing two edges. Two fills with
    complementary where= masks rather than one, so the colour states the sign
    directly (teal = LP above, purple = actual above) without introducing a
    third blended colour the reader has to learn.
    """
    a = np.asarray(series_actual, dtype=float)
    b = np.asarray(series_lp, dtype=float)
    ax.fill_between(t, a, b, where=(b >= a), step='post',
                    color=COLOUR_LP, alpha=alpha, linewidth=0)
    ax.fill_between(t, a, b, where=(b < a), step='post',
                    color=COLOUR_ACTUAL, alpha=alpha, linewidth=0)


def _compare_day_slice(analysis: EMAnalysis, day: pd.Timestamp):
    """Both scenarios' flows for one day, in kW, plus prices and the gap."""
    a = analysis
    day_start = pd.Timestamp(day)
    if day_start.tz is None:
        day_start = day_start.tz_localize(a.trace.index.tz)
    day_start = day_start.normalize()
    day_end = day_start + pd.Timedelta(days=1)
    idx = a.trace.index
    mask = (idx >= day_start) & (idx < day_end)

    t = idx[mask]
    k = 1.0 / a.interval_hours
    act, lp = a.actual_result, a.lp_result
    p_imp = a.p_imp.loc[mask].to_numpy()
    p_exp = a.p_exp.loc[mask].to_numpy()

    # Per-interval volumetric cost difference, in dollars. Positive means
    # actual spent more than the LP would have, i.e. the LP is ahead. Same
    # sign convention as analyse_daily_value_gaps' gap_lp_vs_actual.
    cost_act = act.import_kwh.loc[mask].to_numpy() * p_imp - act.export_kwh.loc[mask].to_numpy() * p_exp
    cost_lp = lp.import_kwh.loc[mask].to_numpy() * p_imp - lp.export_kwh.loc[mask].to_numpy() * p_exp

    d = dict(
        t=t,
        exp_act=act.export_kwh.loc[mask].to_numpy() * k,
        exp_lp=lp.export_kwh.loc[mask].to_numpy() * k,
        imp_act=act.import_kwh.loc[mask].to_numpy() * k,
        imp_lp=lp.import_kwh.loc[mask].to_numpy() * k,
        soc_act=act.battery_soc_kwh.loc[mask].to_numpy(),
        soc_lp=lp.battery_soc_kwh.loc[mask].to_numpy(),
        curtail_lp=lp.curtail_kwh.loc[mask].to_numpy() * k,
        p_imp_c=p_imp * _C_PER_DOLLAR,
        p_exp_c=p_exp * _C_PER_DOLLAR,
        gap=cost_act - cost_lp,
    )
    return d, day_start, day_end, int(mask.sum())


def _setup_compare_figure(site_id: str):
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True,
                             gridspec_kw={'height_ratios': [3, 3, 3, 2]})
    _set_window_title(fig, f'{site_id} \u2014 actual vs perfect foresight')
    fig._viz_site_id = site_id
    fig._viz_kind = 'Actual vs Perfect Foresight'
    return fig, axes


def _plot_compare_day(axes, analysis: EMAnalysis, day: pd.Timestamp) -> None:
    """Draw the actual-vs-LP comparison for one day onto existing axes.

    Structure deliberately mirrors _plot_prices_day: export against export
    price, import against import price, then SOC. The flow decomposition is
    collapsed to a single trace per scenario, because two overlaid stacked
    fills cannot be read. Hue therefore carries the SCENARIO here, while the
    pane and its price line carry the flow identity.

    Encoding on the two flow panes: actual as a light fill to zero with a thin
    edge, perfect foresight as a heavier line over the top. The SOC pane uses
    two lines instead, since a fill to zero under a level is not meaningful.

    Panes:
      1. export, both scenarios, against export price
      2. import, both scenarios, against import price
      3. SOC, both scenarios (the storage strategy difference)
      4. cumulative volumetric gap through the day -- answers "was this day
         lost in one window or all afternoon", which per-interval traces do
         not show directly.

    Solar availability had a pane in the first version of this figure and no
    longer does: it was mostly redundant against the flow panes. LP
    curtailment, which shared that pane, now appears in the export pane
    instead (curtailed energy is export that did not happen) and only when
    it is non-zero, so it vanishes entirely under pinned curtailment.

    Shares one drawing function between the interactive browser and the
    static --compare-day PNG so the two cannot drift apart.
    """
    a = analysis
    if a.lp_result is None:
        raise ValueError(f"{a.site_id}: no lp_result -- build with simulate_lp=True")

    for ax in axes:
        ax.clear()
    _clear_twins(axes[0])
    _clear_twins(axes[1])

    d, day_start, day_end, n = _compare_day_slice(a, day)
    if n == 0:
        for ax in axes:
            ax.text(0.5, 0.5, f"No data for {pd.Timestamp(day).date()}",
                    transform=ax.transAxes, ha='center', va='center')
        return
    t = d['t']

    # Both price axes share one transform and one set of limits, so that a
    # given height means the same c/kWh in both panes.
    p_lo, p_hi = _price_axis_limits(d['p_imp_c'], d['p_exp_c'])

    # --- 1. Export vs export price ---
    if SHOW_DIFFERENCE_BAND:
        _signed_difference_band(axes[0], t, d['exp_act'], d['exp_lp'])
    axes[0].fill_between(t, 0, d['exp_act'], step='post', color=COLOUR_ACTUAL,
                         alpha=_ACTUAL_FILL_ALPHA, linewidth=0, label='actual export')
    axes[0].step(t, d['exp_act'], where='post', color=COLOUR_ACTUAL, linewidth=1.2)
    axes[0].step(t, d['exp_lp'], where='post', color=COLOUR_LP,
                 linewidth=1.8, label='perfect foresight export')
    if float(np.sum(d['curtail_lp'])) > 1e-9:
        axes[0].fill_between(t, d['exp_lp'], d['exp_lp'] + d['curtail_lp'],
                             step='post', facecolor='none', hatch='///',
                             edgecolor=COLOUR_CURTAIL, linewidth=0.0,
                             label='LP curtailed (export forgone)')
    axes[0].set_ylim(0, max(float(np.max(d['exp_act'])), float(np.max(d['exp_lp'])), 0.1) * 1.1)
    axes[0].grid(True, alpha=0.3)
    axe0 = axes[0].twinx()
    axe0.step(t, d['p_exp_c'], where='post', color=COLOUR_SOLAR, linestyle=':',
              linewidth=1.6, label='export price')
    _apply_price_scale(axe0, p_lo, p_hi)
    axe0.set_ylabel('Export price (c/kWh)')
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axe0.get_legend_handles_labels()
    axe0.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)

    # --- 2. Import vs import price ---
    if SHOW_DIFFERENCE_BAND:
        _signed_difference_band(axes[1], t, d['imp_act'], d['imp_lp'])
    axes[1].fill_between(t, 0, d['imp_act'], step='post', color=COLOUR_ACTUAL,
                         alpha=_ACTUAL_FILL_ALPHA, linewidth=0, label='actual import')
    axes[1].step(t, d['imp_act'], where='post', color=COLOUR_ACTUAL, linewidth=1.2)
    axes[1].step(t, d['imp_lp'], where='post', color=COLOUR_LP,
                 linewidth=1.8, label='perfect foresight import')
    axes[1].set_ylim(0, max(float(np.max(d['imp_act'])), float(np.max(d['imp_lp'])), 0.1) * 1.1)
    axes[1].grid(True, alpha=0.3)
    axe1 = axes[1].twinx()
    axe1.step(t, d['p_imp_c'], where='post', color=COLOUR_GRID, linestyle=':',
              linewidth=1.6, label='import price')
    _apply_price_scale(axe1, p_lo, p_hi)
    axe1.set_ylabel('Import price (c/kWh)')
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = axe1.get_legend_handles_labels()
    axe1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8)

    # --- 3. SOC ---
    if SHOW_DIFFERENCE_BAND:
        _signed_difference_band(axes[2], t, d['soc_act'], d['soc_lp'])
    axes[2].plot(t, d['soc_act'], color=COLOUR_ACTUAL, linewidth=1.8, label='actual SOC')
    axes[2].plot(t, d['soc_lp'], color=COLOUR_LP, linewidth=1.8,
                 label='perfect foresight SOC')
    cap = a.battery_spec.capacity_kwh
    min_soc = (a.battery_spec.backup_reserve_pct / 100.0) * cap
    axes[2].axhline(cap, ls='--', color='black', lw=0.8, alpha=0.5)
    axes[2].axhline(min_soc, ls='--', color='red', lw=0.8, alpha=0.5)
    axes[2].set_ylim(0, cap * 1.1)
    axes[2].legend(loc='upper left', fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # --- 4. Cumulative volumetric gap ---
    cum = np.cumsum(d['gap'])
    axes[3].fill_between(t, 0, cum, where=(cum >= 0), step='post',
                         color=COLOUR_LP, alpha=0.25, linewidth=0)
    axes[3].fill_between(t, 0, cum, where=(cum < 0), step='post',
                         color=COLOUR_ACTUAL, alpha=0.25, linewidth=0)
    axes[3].step(t, cum, where='post', color='#333333', linewidth=1.4)
    axes[3].axhline(0, color='black', linewidth=0.6, alpha=0.4)
    axes[3].legend(handles=[
        mpatches.Patch(facecolor=COLOUR_LP, alpha=0.25,
                       label='perfect foresight ahead'),
        mpatches.Patch(facecolor=COLOUR_ACTUAL, alpha=0.25,
                       label='actual ahead'),
    ], loc='upper left', fontsize=8)
    axes[3].grid(True, alpha=0.3)
    _format_time_axis(axes[3], t, day_start, day_end)

    _reapply_labels(
        axes,
        ['Export vs Export Price', 'Import vs Import Price', 'Battery SOC',
         'Cumulative volumetric gap (positive = perfect foresight ahead)'],
        ['Export (kW)', 'Import (kW)', 'SOC (kWh)', 'Gap ($)'],
    )

    day_gap = float(np.sum(d['gap']))
    day_curtail_kwh = float(np.sum(d['curtail_lp'])) * a.interval_hours
    fig = axes[0].figure
    fig.suptitle(
        f"Site {a.site_id}  {day_start.date()}  actual vs perfect foresight  "
        f"(volumetric gap ${day_gap:,.2f}, LP curtailed {day_curtail_kwh:.1f} kWh)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def plot_compare_day(analysis: EMAnalysis, day: pd.Timestamp,
                     out_path: Optional[Path] = None):
    """Static single-day comparison figure, optionally saved to out_path."""
    fig, axes = _setup_compare_figure(analysis.site_id)
    _plot_compare_day(axes, analysis, day)
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Saved comparison plot to {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Day ordering for the browser
# ---------------------------------------------------------------------------

ORDER_MODES = {
    'c': 'chronological',
    'ag': 'absolute gap (LP vs actual), largest first',
    'rg': 'relative gap (gap / headroom), largest first',
    'sc': 'days actual did worse than self-consumption, worst first',
}


def build_day_ranking(analysis: EMAnalysis) -> pd.DataFrame:
    """Per-day attribution table used to order the day browser.

    Deferred import: analyse_daily_value_gaps imports load_and_analyze from
    THIS module (inside run_checks), so a module-level import here would
    close the cycle. Both sides must stay deferred.
    """
    from energy_masters_pipeline.scripts.analyse_daily_value_gaps import (
        attribution_from_analysis, VCR_DENOMINATOR_FLOOR_DOLLARS,
    )
    daily = attribution_from_analysis(analysis).daily.copy()
    headroom = daily['gap_lp_vs_sc']
    # The relative-gap ordering needs a headroom floor. Without it the
    # denominator approaches zero on quiet days and the ratio explodes,
    # filling the top of the ranking with days that had no opportunity to
    # capture in the first place. Same small-denominator trap that ended the
    # arbitrage-cycle work, so the floor is mandatory rather than a polish.
    # Reuses that module's existing floor constant rather than inventing a
    # second threshold for the same problem.
    daily['relative_gap'] = np.where(
        headroom > VCR_DENOMINATOR_FLOOR_DOLLARS,
        daily['gap_lp_vs_actual'] / headroom.where(headroom != 0, np.nan),
        np.nan,
    )
    return daily


def day_order(daily: pd.DataFrame, mode: str) -> list:
    """Ordered day list for a given ORDER_MODES key.

    The ranked modes can return FEWER days than the period contains: 'rg'
    drops days below the headroom floor, 'sc' keeps only days where actual
    underperformed self-consumption. That is intended -- those days are not
    meaningfully rankable on that criterion -- but it means the browser must
    not assume every mode covers every day.
    """
    if mode == 'c':
        return list(daily.index)
    if mode == 'ag':
        return list(daily.sort_values('gap_lp_vs_actual', ascending=False).index)
    if mode == 'rg':
        d = daily.dropna(subset=['relative_gap'])
        return list(d.sort_values('relative_gap', ascending=False).index)
    if mode == 'sc':
        d = daily[daily['gap_actual_vs_sc'] < 0]
        return list(d.sort_values('gap_actual_vs_sc').index)
    raise ValueError(f"unknown order mode {mode!r}; expected one of {list(ORDER_MODES)}")


def _describe_day(daily: pd.DataFrame, day) -> str:
    if day not in daily.index:
        return '(no attribution row for this day)'
    r = daily.loc[day]
    rel = r['relative_gap']
    rel_s = 'n/a' if not np.isfinite(rel) else f"{rel:.2f}"
    return (f"gap ${r['gap_lp_vs_actual']:>8,.2f}   headroom ${r['gap_lp_vs_sc']:>8,.2f}   "
            f"rel {rel_s:>5}   act-sc ${r['gap_actual_vs_sc']:>8,.2f}")


def _print_top_days(daily: pd.DataFrame, order: list, mode: str, n: int = 15) -> None:
    print(f"\nTop {min(n, len(order))} of {len(order)} days -- {ORDER_MODES[mode]}")
    for i, day in enumerate(order[:n], 1):
        print(f"  {i:>3}. {day.date()}  {_describe_day(daily, day)}")
    print()
# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def _interactive_loop(analysis: EMAnalysis) -> None:
    a = analysis
    has_lp = a.lp_result is not None

    # FLOWS and PRICES show ONE scenario, chosen by display_mode. That is a
    # field on a frozen dataclass, so the toggle keeps its own state here and
    # hands the plotters a view via dataclasses.replace. The view is a shallow
    # copy sharing every Series, so making one per redraw is cheap.
    display_mode = a.display_mode

    fig_flows, axes_flows = _setup_flows_figure(a.site_id, display_mode)
    fig_prices, axes_prices = _setup_prices_figure(a.site_id, display_mode)
    fig_compare = axes_compare = None
    daily = None
    mode = 'c'
    order = []

    if has_lp:
        fig_compare, axes_compare = _setup_compare_figure(a.site_id)
        # Ranking is computed once up front: it is a whole-period attribution
        # over every interval, not a per-day calculation, so recomputing it on
        # each step would be wasteful and would not change between days.
        try:
            daily = build_day_ranking(a)
            order = day_order(daily, mode)
        except Exception as e:
            print(f"(day ranking unavailable, chronological only: "
                  f"{type(e).__name__}: {e})")
            daily = None

    plt.ion()

    first_day = a.trace.index[0].normalize()
    last_day = a.trace.index[-1].normalize()
    if not order:
        order = list(pd.date_range(first_day, last_day, freq='D', tz=a.trace.index.tz))
    pos = 0
    current = order[0]

    def _redraw():
        # COMPARE always draws both scenarios, so the toggle does not touch it.
        view = replace(a, display_mode=display_mode)
        _plot_flows_day(axes_flows, view, current)
        _plot_prices_day(axes_prices, view, current)
        if axes_compare is not None:
            _plot_compare_day(axes_compare, a, current)

    _redraw()
    plt.show()
    plt.pause(0.1)

    print(f"Days: {first_day.date()} to {last_day.date()}")
    print("Controls: Enter=next, b=back, YYYY-MM-DD=jump, t=top days, q=quit")
    if has_lp:
        print("  m = swap FLOWS/PRICES between actual and perfect foresight "
              "(COMPARE always shows both)")
    if daily is not None:
        print("Order modes: " + ", ".join(f"{k}={v}" for k, v in ORDER_MODES.items()))
    print(f"Order: {ORDER_MODES[mode]} ({len(order)} days)")

    while True:
        try:
            raw = input(
                f"[{current.date()} {pos + 1}/{len(order)} {display_mode}] > "
            ).strip().lower()
        except EOFError:
            break

        if raw == 'q':
            break

        elif raw == 'm':
            if not has_lp:
                print("  (no LP result to swap to -- run without --no-simulate-lp)")
                continue
            display_mode = 'actual' if display_mode == 'lp' else 'lp'
            # The suptitle is rewritten by _finalise on every draw, but the
            # window title is only set at figure setup, so it needs updating
            # here or it goes stale against what is being plotted.
            _set_window_title(fig_flows, f'{a.site_id} \u2014 flows ({display_mode})')
            _set_window_title(fig_prices, f'{a.site_id} \u2014 prices ({display_mode})')
            label = ('perfect foresight (simulated)' if display_mode == 'lp'
                     else 'actual (Amber-coordinated)')
            print(f"  FLOWS/PRICES now showing: {label}")

        elif raw == 't':
            if daily is not None:
                _print_top_days(daily, order, mode)
            else:
                print("  (no ranking available)")
            continue

        elif raw in ORDER_MODES and daily is not None:
            try:
                new_order = day_order(daily, raw)
            except Exception as e:
                print(f"  (could not order by {raw}: {e})")
                continue
            if not new_order:
                print(f"  (no days match '{raw}' -- order unchanged)")
                continue
            mode, order = raw, new_order
            pos = 0
            current = order[0]
            print(f"Order: {ORDER_MODES[mode]} ({len(order)} days)")

        elif raw in ORDER_MODES and daily is None:
            print("  (ranking unavailable -- chronological only)")
            continue

        elif raw == 'b':
            pos = max(0, pos - 1)
            current = order[pos]

        elif raw in ('', 'n'):
            pos = min(len(order) - 1, pos + 1)
            current = order[pos]

        else:
            try:
                jump = pd.Timestamp(raw).tz_localize(a.trace.index.tz).normalize()
            except Exception as e:
                print(f"  (unrecognised input: {e})")
                continue
            if jump < first_day or jump > last_day:
                print(f"  (out of range {first_day.date()}..{last_day.date()})")
                continue
            current = jump
            # A jumped-to day need not be in the current order: 'rg' drops
            # days below the headroom floor and 'sc' keeps only underperforming
            # days. Show it anyway, and keep pos meaningful where we can.
            if current in order:
                pos = order.index(current)
            else:
                print(f"  (not in the current '{mode}' order -- showing anyway)")

        if daily is not None:
            print(f"  {current.date()}  {_describe_day(daily, current)}")

        _redraw()
        fig_flows.canvas.draw_idle()
        fig_prices.canvas.draw_idle()
        if fig_compare is not None:
            fig_compare.canvas.draw_idle()
        plt.pause(0.05)

    plt.close(fig_flows)
    plt.close(fig_prices)
    if fig_compare is not None:
        plt.close(fig_compare)



def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('site_id', help='Site identifier, e.g. EM-427')
    parser.add_argument('--simulate-lp', action=argparse.BooleanOptionalAction,
                        default=DEFAULT_SIMULATE_LP,
                        help='Simulate perfect-foresight dispatch on this site\'s real '
                             'load/solar trace with a similarly-sized battery, and plot '
                             'that instead of the actual (Amber-coordinated) behaviour. '
                             'The actual reconstruction is still computed and printed as '
                             'a counterfactual comparison in the verification block. '
                             'On by default; --no-simulate-lp skips the LP solve.')
    parser.add_argument('--export-limit-kw', type=float, default=None,
                        help='Per-phase export limit kW for the LP simulation, 1 phase '
                             'assumed. Default: auto-estimated from the site\'s actual '
                             'export data (see estimate_export_limit.py). Only used with '
                             '--simulate-lp.')
    parser.add_argument('--curtailment', type=str, default=None, metavar='MODE',
                        help=CURTAILMENT_ARG_HELP + ' Only used with --simulate-lp.')
    parser.add_argument('--c-deg-per-kwh', type=float, default=None,
                        help='Override the LP battery degradation cost, in dollars '
                             '(not cents) per kWh of throughput. Default: '
                             f'{DEFAULT_C_DEG_PER_KWH}. Only used with --simulate-lp. '
                             'See sweep_c_deg.py for calibrating this value against '
                             'actual battery-use metrics.')
    parser.add_argument('--compare-day', type=str, default=None, metavar='YYYY-MM-DD',
                        help='Render an actual-vs-perfect-foresight comparison figure for '
                             'this single day and save it as a PNG, instead of opening the '
                             'interactive browser. Implies --simulate-lp. Repeatable via '
                             'comma separation, e.g. --compare-day 2025-11-03,2025-11-04. '
                             'Intended for investigating the top days from '
                             'analyse_daily_value_gaps.py.')
    parser.add_argument('--project-root', type=Path, default=None)
    args = parser.parse_args(argv)

    simulate_lp = args.simulate_lp or (args.compare_day is not None)

    try:
        curtail_cost_per_kwh_dollars = resolve_curtailment_arg(args.curtailment)
    except ValueError as e:
        parser.error(str(e))

    paths = default_paths(args.project_root)
    analysis = load_and_analyze(
        args.site_id, paths=paths,
        simulate_lp=simulate_lp, export_limit_kw=args.export_limit_kw,
        curtail_cost_per_kwh_dollars=curtail_cost_per_kwh_dollars,
        c_deg_per_kwh_override=args.c_deg_per_kwh,
    )
    _print_verification(analysis)

    if args.compare_day is not None:
        days = [d.strip() for d in args.compare_day.split(',') if d.strip()]
        for day_str in days:
            out_dir = Path('energy_masters_pipeline/charts/diagnostics')
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'{args.site_id}_compare_{day_str}.png'
            try:
                plot_compare_day(analysis, pd.Timestamp(day_str), out_path)
            except ValueError as e:
                print(f"  skipped {day_str}: {e}")
        return 0

    _interactive_loop(analysis)
    return 0


if __name__ == '__main__':
    sys.exit(main())