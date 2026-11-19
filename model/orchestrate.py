"""Scenario routing and component wiring for the coordination model.

A scenario is a single (state, year, cohort, coordination_type, battery,
network_tariff, retail_tariff) combination. `run_scenario` executes the
full pipeline for one scenario and returns a ScenarioResult holding per-
site flow traces and bills.

Multi-scenario runs (cross-product over states / years / batteries) are
handled in `model/main.py`, which calls `run_scenario` for each expanded
combination.

Pre-flight validation
---------------------
`validate_scenario` checks all required inputs exist and load before any
heavy computation. For a multi-scenario run, main.py validates every
expanded scenario first and aborts on the first failure (strict mode), so
a bad config or missing file doesn't waste hours of processing.

Per-site resolution
-------------------
For now, every site in the scenario receives the same BatterySpec
(scenario.battery_spec) and a uniform solar scaling factor of 1.0. The
orchestrator already routes through per-site resolution helpers
(`_resolve_per_site_batteries`, `_resolve_per_site_solar_scaling`) so
future complexity (per-site battery sizing, per-site solar scaling for
larger-array studies) slots in without restructuring the orchestrator.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import pandas as pd

from model.core.scenario import (
    ROLLING_COORDINATION_TYPES, VPP_COORDINATION_TYPES, MIXED_COORDINATION_TYPES,
)
from model.core import (
    ScenarioConfig, BatterySpec, DataContext, Paths,
    SiteTraces, SiteBill,
    NetworkTariff, RetailTariff,
    RollingConfig,
    load_network_tariff, load_retail_tariff,
)
from model.core.site_result import VppTraces
from model.core.prices import (
    build_price_vectors, build_lp_price_vectors_advanced, charge_window_mask,
    build_price_stack_offsets, PriceStackOffsets,
    build_mixed_dispatch_stack, dispatch_curtail_cost,
    check_export_order_nudge_degradation,
)
from model.components.ingest import ingest
from model.components.self_consumption import self_consumption
from model.components.perfect_foresight_lp import perfect_foresight_lp
from model.components.perfect_foresight_lp_advanced import perfect_foresight_lp_advanced
from model.components.perfect_foresight_custom import perfect_foresight_custom
from model.components.vpp_event_classifier import classify_vpp_events
from model.components.vpp_sc_event import vpp_sc_event
from model.components.rolling_optimisation import rolling_optimisation, RollingDiagnostics
from model.components.rolling_vpp_event import rolling_vpp_event
from model.components.rolling_advanced import AdvancedWindowPlanner
from model.components.billing import compute_bill, compute_bill_and_monthly_bills
from model.forecast.registry import build_provider, region_for_state, is_reportable
from model.forecast.usage import UsageForecastProvider
from model.forecast.solar import SolarForecastProvider

# Default location of the canonical pre-dispatch store, matching the same
# constant name and value used throughout the Energy Masters scripts
# (sweep_horizon.py, check_rolling_optimisation.py, run_cohort_rungs.py).
# ScenarioConfig / RollingConfig carry no store_root field -- the store is a
# single project-wide resource, not a per-scenario setting -- so this is the
# one place the orchestrator needs to know where it lives.
DEFAULT_PREDISPATCH_STORE = Path('data/clean/predispatch')


# ----------------------------------------------------------------------
# Result type
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioResult:
    """In-memory results from a single scenario run.

    Attributes:
        scenario: The ScenarioConfig that was run.
        site_traces: Per-site SiteTraces produced by the chosen dispatch.
        site_bills: Per-site SiteBills computed against the dispatch flows.
        elapsed_seconds: Wall-clock time of the whole run_scenario call
            (ingest + per-site dispatch + billing). 0.0 if not measured.
        ingest_seconds: Wall-clock time of the ingest phase alone. Useful
            for separating I/O cost from computation cost. 0.0 if not
            measured. Dispatch + billing time is elapsed_seconds - ingest_seconds.
        lp_seconds: Cumulative time spent in the dispatch component (LP,
            greedy, event or rolling solve) across all sites. 0.0 for the
            plain self_consumption path. Cumulative worker-seconds when
            n_workers > 1, so it can exceed wall-clock time.
        baseline_traces: For every optimising coordination type, the
            self_consumption SiteTraces per site (the comparator). None for
            the plain self_consumption path and non-battery cohorts.
        baseline_bills: The self_consumption SiteBill per site, populated
            alongside baseline_traces. site_bills holds the dispatched
            strategy's bills.
        n_workers: Number of parallel worker processes used for the
            dispatch phase. 1 = serial. Stored so the summary can format
            timing correctly.
    """
    scenario: ScenarioConfig
    site_traces: dict[str, SiteTraces]
    site_bills: dict[str, SiteBill]
    elapsed_seconds: float = 0.0
    ingest_seconds: float = 0.0
    lp_seconds: float = 0.0
    baseline_traces: Optional[dict[str, SiteTraces]] = None
    baseline_bills: Optional[dict[str, SiteBill]] = None
    n_workers: int = 1
    vpp_traces: Optional[dict[str, VppTraces]] = None
    """Per-site VppTraces (natural/forced flow split) from a VPP run.
    Populated for the four VPP coordination types; None otherwise. Contains
    the natural/forced flow split that billing used to compute VPP rewards.
    Stored so downstream analysis can compute operator P&L and cap utilisation
    without re-running the LP.
    """
    vpp_event: Optional['pd.Series'] = None
    """Boolean Series (datetime_aest index) marking VPP event intervals.
    True = operator-controlled; False = default regime. Shared across all
    sites for a run (events are price-driven, not site-specific). None for
    all non-VPP coordination types, and for the rolling VPP types, whose
    declared events are per-site. Stored so write_vpp_forced_flows_parquet
    can select event rows without re-running the classifier.
    """
    site_monthly_bills: Optional[dict[str, dict[int, SiteBill]]] = None
    """Per-site monthly SiteBills (site_id -> {month: SiteBill}), populated
    only when run_scenario(..., with_monthly_bills=True). Computed inline
    during dispatch via compute_bill_and_monthly_bills, sharing one
    settlement build with the annual site_bills entry for the same site
    (rather than a separate pass re-deriving rates/local time from
    scratch) -- see billing.py's module docstring. write_monthly_bills_csv
    reads this directly; it does not re-bill anything. None (not an empty
    dict) when with_monthly_bills was False for this run, so callers can
    distinguish "not requested" from "requested, nothing to write" (e.g. a
    zero-site scenario).

    Superseded fields removed here: this ScenarioResult previously also
    stashed `prices` / `network_tariff` / `retail_tariff` / `interval_hours`
    solely so write_monthly_bills_csv could reconstruct a second, separate
    settlement build after the fact. Now that monthly bills are computed
    inline (sharing the SAME build as the annual bill, see
    compute_bill_and_monthly_bills), there is nothing left needing those
    stashed inputs, and keeping them around would just be unused
    complexity with no current reader.
    """
    rolling_diagnostics: Optional[dict[str, RollingDiagnostics]] = None
    """Per-site RollingDiagnostics, keyed by site_id. Populated for the
    ROLLING_COORDINATION_TYPES; None for every other coordination type. RollingDiagnostics.retained is always empty here
    unless scenario.rolling.retain_plans_from is set, which validate_scenario
    only permits for a single-site scenario. write_rolling_diagnostics_csv
    reads this field and writes nothing when it is None.
    """
    min_export_price_resolution: Optional['MinExportPriceResolution'] = None
    """How the VPP minimum export price was arrived at. Populated on all four
    VPP coordination types, which share resolve_min_export_price; None
    elsewhere. write_scenario_summary reads it and writes nothing when it is
    None.
    """


# ----------------------------------------------------------------------
# Pre-flight validation
# ----------------------------------------------------------------------

class ScenarioValidationError(Exception):
    """Raised when a scenario fails pre-flight validation."""


def validate_scenario(scenario: ScenarioConfig, paths: Paths) -> None:
    """Verify all inputs for a scenario exist and load.

    Does NOT execute the model. Catches missing files, malformed tariffs,
    NaN prices, and absent site trace files. Designed to fail fast so a
    multi-scenario run aborts before any heavy computation begins.

    Raises:
        ScenarioValidationError: if any input is missing or invalid. The
            message identifies the scenario_id and the specific problem.
    """
    sid = scenario.scenario_id

    try:
        # Tariff files load successfully (catches missing files and bad YAML)
        network = load_network_tariff(scenario.network_tariff_id, paths.tariffs_dir)
        retail = load_retail_tariff(scenario.retail_tariff_id, paths.tariffs_dir)

        # Metadata file exists with at least one site for the scenario state
        meta_path = paths.edp_metadata_dir / f'{scenario.year}_selected_sites_clean.csv'
        if not meta_path.exists():
            raise ScenarioValidationError(
                f"{sid}: metadata file not found: {meta_path}"
            )
        meta = pd.read_csv(meta_path)
        sites = meta[meta['state'] == scenario.state]['edp_site_id'].tolist()
        if not sites:
            raise ScenarioValidationError(
                f"{sid}: no sites in metadata for state={scenario.state!r}"
            )

        # All listed sites' trace files exist (existence check only)
        trace_dir = paths.edp_clean_dir / str(scenario.year)
        missing = [s for s in sites
                   if not (trace_dir / f'edp_clean_{s}_{scenario.year}.parquet').exists()]
        if missing:
            raise ScenarioValidationError(
                f"{sid}: {len(missing)} site trace files missing (first 5: "
                f"{missing[:5]})"
            )

        # Price file exists and contains the required RRP column, no NaN
        price_path = paths.nem_prices_dir / f'{scenario.year}_nem_prices.parquet'
        if not price_path.exists():
            raise ScenarioValidationError(
                f"{sid}: price file not found: {price_path}"
            )
        rrp_col = {'NSW': 'NSW1_rrp', 'QLD': 'QLD1_rrp',
                   'SA': 'SA1_rrp', 'VIC': 'VIC1_rrp'}[scenario.state]
        prices = pd.read_parquet(price_path, columns=['datetime_aest', rrp_col])
        if prices[rrp_col].isna().any():
            n = int(prices[rrp_col].isna().sum())
            raise ScenarioValidationError(
                f"{sid}: {rrp_col} contains {n} NaN values in {price_path}"
            )

        # Soft check: warn on tariff/state mismatch (not fatal but worth noting)
        for label, t in (('Network', network), ('Retail', retail)):
            if t.state != scenario.state:
                warnings.warn(
                    f"{sid}: {label} tariff is configured for state {t.state!r} "
                    f"but scenario state is {scenario.state!r}."
                )

        # A mixed plan is dispatched against its own dispatch prices
        # (model_spec.md, Mixed-plan dispatch prices), which only the
        # advanced optimisers build. Cohorts without a battery and
        # self-consumption do not read prices at all.
        if (retail.type == 'mixed'
                and scenario.coordination_type not in MIXED_COORDINATION_TYPES):
            raise ScenarioValidationError(
                f"{sid}: retail tariff '{scenario.retail_tariff_id}' is type "
                f"'mixed', which coordination_type="
                f"'{scenario.coordination_type}' does not support. A mixed "
                f"plan runs under the no_cer or solar cohort or one of "
                f"{sorted(t for t in MIXED_COORDINATION_TYPES if t)}."
            )
        # The export-order nudge acts on TOU plans only. Checked here so a
        # degradation cost too low to cover it fails before ingest rather
        # than inside the first site's LP.
        if scenario.apply_export_order_nudge and retail.type == 'tou':
            try:
                check_export_order_nudge_degradation(
                    scenario.battery_spec.c_deg_per_kwh, retail.name)
            except ValueError as e:
                raise ScenarioValidationError(f"{sid}: {e}") from e
        if (retail.type == 'mixed'
                and scenario.coordination_type == 'perfect_foresight_advanced'):
            # The stack proves price ordering over the market range; build it
            # here so a tariff pair that fails does so before ingest.
            mixed_idx = pd.DatetimeIndex(prices['datetime_aest'])
            try:
                build_mixed_dispatch_stack(
                    mixed_idx, network, retail, scenario.state,
                    (mixed_idx[1] - mixed_idx[0]).total_seconds() / 3600.0)
            except ValueError as e:
                raise ScenarioValidationError(f"{sid}: {e}") from e

        # VPP-specific checks
        if scenario.coordination_type in ('vpp', 'vpp_sc_event', 'vpp_tou_event'):
            if not retail.has_vpp:
                raise ScenarioValidationError(
                    f"{sid}: coordination_type='{scenario.coordination_type}' but retail tariff "
                    f"'{scenario.retail_tariff_id}' has no VppConfig block. "
                    f"Add a 'vpp:' section to the tariff YAML."
                )
            op_tariff = load_retail_tariff(
                scenario.op_retail_tariff_id, paths.tariffs_dir)
            if op_tariff.type != 'wholesale_exposed':
                raise ScenarioValidationError(
                    f"{sid}: op_retail_tariff_id='{scenario.op_retail_tariff_id}' "
                    f"must be type 'wholesale_exposed', got '{op_tariff.type}'."
                )

        # Minimum export price rules, shared by the perfect-foresight and
        # rolling VPP types. Both resolve through resolve_min_export_price, so
        # both must reject the same configurations -- and they must reject them
        # identically, because the value capture ratio between a rolling run
        # and its perfect-foresight benchmark is only interpretable when the
        # two resolved the same YAML to the same threshold.
        if scenario.coordination_type in VPP_COORDINATION_TYPES:
            vpp_cfg = retail.vpp
            if vpp_cfg is None:
                raise ScenarioValidationError(
                    f"{sid}: coordination_type="
                    f"'{scenario.coordination_type}' requires retail tariff "
                    f"'{scenario.retail_tariff_id}' to carry a vpp block; "
                    f"it has none."
                )

            # Rejected here rather than at the resolution point so a
            # multi-scenario run aborts before any ingest.
            if vpp_cfg.min_export_price_mode == 'based_on_last_year':
                raise ScenarioValidationError(
                    f"{sid}: retail tariff '{scenario.retail_tariff_id}' sets "
                    f"min_export_price_mode='based_on_last_year', which is "
                    f"not implemented. Use 'fixed' with a stated "
                    f"min_export_price_dollars_per_mwh, or "
                    f"'based_on_this_year' to derive the threshold from the "
                    f"modelled year."
                )

            # 'fixed' is the default, so this also catches a tariff that names
            # no mode and states no value. That combination used to mean
            # "derive from this year" by implication; it now fails rather than
            # resolving to something different from what it used to, which is
            # the whole reason the rule is a rejection and not a fallback.
            # Under 'based_on_this_year' the configured value is a floor
            # beneath the derived estimate and may be omitted, meaning zero.
            if (vpp_cfg.min_export_price_mode == 'fixed'
                    and vpp_cfg.min_export_price_dollars_per_mwh is None):
                raise ScenarioValidationError(
                    f"{sid}: retail tariff '{scenario.retail_tariff_id}' "
                    f"omits min_export_price_dollars_per_mwh under "
                    f"min_export_price_mode='fixed' (the default when the key "
                    f"is absent), which leaves nothing to derive the "
                    f"threshold from. State a value, or set "
                    f"min_export_price_mode='based_on_this_year' to compute "
                    f"it from the modelled year's own prices."
                )

        # Checks shared by both rolling coordination types.
        if scenario.coordination_type in ROLLING_COORDINATION_TYPES:
            idx = pd.DatetimeIndex(prices['datetime_aest'])
            interval_hours = (idx[1] - idx[0]).total_seconds() / 3600.0

            if scenario.rolling is None:
                raise ScenarioValidationError(
                    f"{sid}: coordination_type="
                    f"'{scenario.coordination_type}' requires a rolling "
                    f"config; none was supplied."
                )

        # rolling_perfect_foresight_lp_advanced-specific checks
        if scenario.coordination_type == 'rolling_perfect_foresight_lp_advanced':
            # No price-stack guard here. On a TOU plan the customer price does
            # not depend on RRP at all, so build_price_stack_offsets is never
            # called and its wholesale_exposed guard would reject every valid
            # scenario on this path. The ordering property that matters here
            # -- p_imp >= p_exp, which a block window masked below the feed-in
            # rate can break -- is checked by AdvancedWindowPlanner at
            # construction, against the same vectors the run itself uses.
            if retail.type not in ('tou', 'mixed'):
                raise ScenarioValidationError(
                    f"{sid}: coordination_type="
                    f"'rolling_perfect_foresight_lp_advanced' is for TOU "
                    f"and mixed plans; retail tariff "
                    f"'{scenario.retail_tariff_id}' is type "
                    f"'{retail.type}'. Wholesale-exposed plans use "
                    f"'rolling_perfect_foresight_lp'."
                )

            # A rolling run must have at least one genuine source of
            # imperfect information, or it is perfect foresight with an
            # artificially truncated horizon wearing a rolling label. On this
            # path that means site forecasts, or critical rates (which
            # intervals are critical depends on the price forecast). The wholesale path gets
            # this for free because its whole price stack moves with RRP.
            has_site_forecast = (scenario.rolling.usage_forecast is not None
                                 or scenario.rolling.solar_forecast is not None)
            needs_prices = advanced_plan_needs_price_forecast(retail)
            if not has_site_forecast and not needs_prices:
                raise ScenarioValidationError(
                    f"{sid}: retail tariff "
                    f"'{scenario.retail_tariff_id}' has no critical rate "
                    f"and no usage_forecast or solar_forecast is configured, "
                    f"so nothing in this run is uncertain. That is perfect "
                    f"foresight over a truncated horizon, not a rolling "
                    f"result. Configure a site forecast, or use "
                    f"coordination_type='perfect_foresight_advanced'."
                )

            # The planner is what the run actually builds, so constructing it
            # here fails fast on a tariff it would reject later (a block
            # window masked below the feed-in rate makes importing to export
            # profitable) rather than after a full ingest. Cost is a couple
            # of vectorised passes over the year, not a model execution.
            try:
                AdvancedWindowPlanner(
                    index=idx, retail_tariff=retail, network_tariff=network,
                    state=scenario.state,
                    battery_spec=scenario.battery_spec,
                    horizon_intervals=int(round(
                        scenario.rolling.horizon_hours / interval_hours)),
                    interval_hours=interval_hours,
                    apply_export_order_nudge=(
                        scenario.apply_export_order_nudge),
                )
            except ValueError as e:
                raise ScenarioValidationError(f"{sid}: {e}") from e

        # rolling VPP-specific checks
        if scenario.coordination_type in ('rolling_vpp_sc_event',
                                          'rolling_vpp_tou_event'):
            vpp_cfg = retail.vpp
            if vpp_cfg is None:
                raise ScenarioValidationError(
                    f"{sid}: coordination_type="
                    f"'{scenario.coordination_type}' requires retail tariff "
                    f"'{scenario.retail_tariff_id}' to carry a vpp block; "
                    f"it has none."
                )

            # The declaration lookahead is min(W, n). A horizon shorter than
            # one event window truncates EVERY lookahead, so the run would
            # declare fewer events for a configuration reason and the result
            # would read as a forecast effect.
            # Compared in INTERVALS, using the same arithmetic
            # rolling_optimisation applies to horizon_hours, so validation
            # cannot accept a horizon the run then treats as shorter (or
            # reject one it treats as long enough).
            window_intervals = math.ceil(
                vpp_cfg.event_window_hours / interval_hours)
            horizon_intervals = max(1, int(round(
                scenario.rolling.horizon_hours / interval_hours)))
            if horizon_intervals < window_intervals:
                raise ScenarioValidationError(
                    f"{sid}: rolling.horizon_hours="
                    f"{scenario.rolling.horizon_hours} is shorter than the "
                    f"VPP event window of {vpp_cfg.event_window_hours}h "
                    f"({horizon_intervals} intervals against "
                    f"{window_intervals}). Every event declaration lookahead "
                    f"would be truncated. Raise the horizon to at least the "
                    f"event window length."
                )

            if (scenario.coordination_type == 'rolling_vpp_tou_event'
                    and retail.type != 'tou'):
                raise ScenarioValidationError(
                    f"{sid}: coordination_type='rolling_vpp_tou_event' "
                    f"optimises the household against a TOU plan between "
                    f"events; retail tariff "
                    f"'{scenario.retail_tariff_id}' is type "
                    f"'{retail.type}'. Use 'rolling_vpp_sc_event' for a "
                    f"household that self-consumes between events."
                )

        # rolling_perfect_foresight_lp-specific checks
        if scenario.coordination_type == 'rolling_perfect_foresight_lp':

            # Tariff type and price-ordering guards. Deliberately NOT
            # duplicated as separate checks here: build_price_stack_offsets
            # already raises ValueError on a non-wholesale_exposed retail
            # tariff and on a price-ordering violation across the market
            # price range (see prices.py, "Guard 1" / "Guard 2"), and it is
            # the exact function the run itself calls, so calling it here
            # both fails fast AND cannot drift from what the run enforces.
            # Cost is one vectorised pass over the year's index (~100k rows),
            # not a model execution, so it belongs in validate_scenario
            # alongside the existing NaN-price check on the same data.
            try:
                build_price_stack_offsets(
                    idx, network, retail, scenario.state, interval_hours)
            except ValueError as e:
                raise ScenarioValidationError(f"{sid}: {e}") from e


        # Checks that apply to every rolling coordination type.
        if scenario.coordination_type in ROLLING_COORDINATION_TYPES:
            # Non-causal correctness-gate check, ported from
            # run_cohort_rungs.py's per-rung guard (registry.is_reportable).
            # 'perfect' returns actual prices; that is only a defensible
            # reference at a FINITE horizon (perfect price foresight within
            # a rolling window, isolating window-mechanics cost from every
            # forecast effect). At a horizon covering the whole scenario
            # period it degenerates into full-period perfect foresight
            # wearing a rolling label, which is the correctness gate this
            # component is tested against, not a result anyone should file.
            #
            # It applies only where a price forecast is actually used. An
            # advanced plan with no critical rate reads no price forecast
            # at all, so forecast_source is inert and reportability is
            # governed by the site forecasts instead.
            needs_store = (
                scenario.coordination_type == 'rolling_perfect_foresight_lp'
                or advanced_plan_needs_price_forecast(retail))
            period_hours = len(idx) * interval_hours
            if needs_store and not is_reportable(
                    scenario.rolling.forecast_source,
                                 scenario.rolling.horizon_hours, period_hours):
                raise ScenarioValidationError(
                    f"{sid}: forecast_source="
                    f"'{scenario.rolling.forecast_source}' at horizon "
                    f"{scenario.rolling.horizon_hours:g} h over a "
                    f"{period_hours:.0f} h period sees the whole period and "
                    "is the correctness gate this component is tested "
                    "against, not a reportable result. Reduce horizon_hours "
                    "or use a different forecast_source."
                )

            # retain_plans_from/to hold full forecast vectors and plans for
            # the retention window IN MEMORY for every site in the scenario
            # (RollingConfig is one object per scenario, not per site). For a
            # bulk EDP scenario that is the same multi-gigabyte-through-the-
            # pool problem the coverage check below exists to keep out of the
            # store; for a single-site diagnostic run it is exactly the
            # forecast-evolution use case the field exists for. So it is
            # permitted only when the scenario resolves to exactly one site.
            if (scenario.rolling.retain_plans_from is not None
                    and len(sites) != 1):
                raise ScenarioValidationError(
                    f"{sid}: rolling.retain_plans_from/to is only permitted "
                    f"for a single-site scenario; {scenario.state} "
                    f"{scenario.year} resolves to {len(sites)} sites. Restrict "
                    "the scenario to one site (e.g. via metadata) or drop "
                    "retention for a bulk run."
                )

            # Pre-dispatch store coverage. Only the 'predispatch' forecast
            # source touches the store; 'perfect' and 'truncated_perfect*'
            # are built from prices_mwh alone (see registry.build_provider)
            # and need no store at all. Checked against the SAME partition
            # naming convention the pipeline writes to
            # (pipeline.pre_dispatch.B_Clean_Predispatch.partition_path) and
            # the SAME known-gap table stage C validates against
            # (pipeline.pre_dispatch.C_Validate_Predispatch.KNOWN_ARCHIVE_GAPS)
            # rather than a second, independently-maintained list that could
            # drift from it. Imported lazily so a non-rolling scenario, or a
            # rolling scenario on a non-predispatch source, never pulls the
            # pipeline package in at all.
            #
            # Full calendar-year coverage only: this scenario always starts
            # 1 January with no prior-December preload (see RollingConfig's
            # infill_lag_hours), so only the scenario's own 12 months matter.
            if scenario.rolling.forecast_source == 'predispatch':
                from pipeline.pre_dispatch.B_Clean_Predispatch import (
                    partition_path,
                )
                from pipeline.pre_dispatch.C_Validate_Predispatch import (
                    KNOWN_ARCHIVE_GAPS,
                )
                region = region_for_state(scenario.state)
                store_root = DEFAULT_PREDISPATCH_STORE
                missing_months = []
                for period in pd.period_range(
                        f'{scenario.year}-01', f'{scenario.year}-12', freq='M'):
                    if (('predispatch', str(period)) in KNOWN_ARCHIVE_GAPS):
                        continue
                    path = partition_path(
                        store_root, 'predispatch', region, period)
                    if not path.exists():
                        missing_months.append(str(period))
                if missing_months:
                    raise ScenarioValidationError(
                        f"{sid}: pre-dispatch store {store_root} is missing "
                        f"{len(missing_months)} month(s) for region {region}: "
                        f"{missing_months}. Run pipeline.pre_dispatch stages "
                        "A-C for this range first."
                    )

    except ScenarioValidationError:
        raise
    except Exception as e:
        # Wrap any other unexpected failure as a validation error so main.py
        # treats it uniformly with the strict-abort rule.
        raise ScenarioValidationError(f"{sid}: {type(e).__name__}: {e}") from e


# ----------------------------------------------------------------------
# Per-site resolution helpers (currently trivial; structured for future
# programmatic variation per-site)
# ----------------------------------------------------------------------

def _resolve_per_site_batteries(
    scenario: ScenarioConfig,
    site_metadata: pd.DataFrame,
) -> dict[str, Optional[BatterySpec]]:
    """Resolve the BatterySpec each site receives.

    Today: every site gets the same scenario.battery_spec (or None for
    non-battery cohorts).

    Future expansion this hook supports without restructuring:
      - per-site battery sizing (e.g. by site avg_daily_gross_load_kwh)
      - randomised allocation across a fleet of spec options
      - cohort-stratified batteries
    """
    return {s: scenario.battery_spec
            for s in site_metadata['edp_site_id']}


def _resolve_per_site_solar_scaling(
    scenario: ScenarioConfig,
    site_metadata: pd.DataFrame,
) -> dict[str, float]:
    """Resolve the solar scaling multiplier each site receives.

    Today: every site gets 1.0 (no scaling).

    Future expansion this hook supports:
      - uniform upsize (e.g. factor 1.5 to model larger arrays universally)
      - per-site scaling by inverter rating or subarray capacity
      - cohort-stratified scaling
    """
    return {s: 1.0 for s in site_metadata['edp_site_id']}


def _resolve_per_site_export_limits(
    scenario: ScenarioConfig,
    site_metadata: pd.DataFrame,
) -> dict[str, float]:
    """Resolve the export power limit (kW) each site receives.

    Per-site limit:
        export_limit_kw[site] = export_limit_kw_per_phase
            * max(ac_load_net_phases[site], pv_site_net_phases[site])

    If scenario.export_limit_kw_per_phase is None, the limit is infinite
    (non-binding) for every site. The returned value is in kW; the LP
    component converts to a per-interval kWh cap via interval_hours.

    Mirrors _resolve_per_site_batteries: a per-site resolution hook the
    orchestrator owns. Consumed by every optimising runner; the
    self_consumption path does not apply an export limit.

    Raises:
        ValueError: if a limit is requested but the phase columns are absent
            from site_metadata, or a site has missing phase data.
    """
    epp = scenario.export_limit_kw_per_phase
    if epp is None:
        return {s: float('inf') for s in site_metadata['edp_site_id']}

    required = ['ac_load_net_phases', 'pv_site_net_phases']
    missing = [c for c in required if c not in site_metadata.columns]
    if missing:
        raise ValueError(
            f"export_limit_kw_per_phase is set but site_metadata is missing "
            f"phase columns {missing}. Ensure ingest retains them "
            f"(_METADATA_COLUMNS)."
        )

    limits: dict[str, float] = {}
    for site_id, ac_ph, pv_ph in zip(
        site_metadata['edp_site_id'],
        site_metadata['ac_load_net_phases'],
        site_metadata['pv_site_net_phases'],
    ):
        if pd.isna(ac_ph) or pd.isna(pv_ph):
            raise ValueError(
                f"Site {site_id} has missing phase data "
                f"(ac_load_net_phases={ac_ph}, pv_site_net_phases={pv_ph}); "
                f"cannot resolve export limit."
            )
        limits[site_id] = epp * max(float(ac_ph), float(pv_ph))
    return limits


# ----------------------------------------------------------------------
# Dispatch routing
# ----------------------------------------------------------------------

def _dispatch_for_site(
    site_id: str,
    site_trace: pd.DataFrame,
    battery_spec: Optional[BatterySpec],
    solar_scale: float,
    scenario: ScenarioConfig,
    prices: pd.Series,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
) -> SiteTraces:
    """Run the chosen dispatch strategy for one site.

    Handles the single-trace cohorts. perfect_foresight is NOT routed here:
    it needs a self_consumption baseline (for terminal SOC and VCR) and so is
    handled directly in run_scenario via _run_perfect_foresight, which returns
    both the LP and baseline traces.

    Routing (see model_spec.md, Orchestration):
        no_cer:         self_consumption with solar = zeros
        solar:          self_consumption with no battery
        solar_battery + self_consumption: self_consumption with battery
        solar_battery + self_hedging:     NotImplementedError
        every other coordination type is routed to its own runner in
        run_scenario and never reaches here.
    """
    load_kwh = site_trace['gross_load_kwh']
    solar_kwh = site_trace['pv_site_net_kwh'] * solar_scale

    if scenario.cohort == 'no_cer':
        zero_solar = solar_kwh * 0.0  # preserve index/tz
        return self_consumption(site_id, zero_solar, load_kwh, battery_spec=None)

    if scenario.cohort == 'solar':
        return self_consumption(site_id, solar_kwh, load_kwh, battery_spec=None)

    # solar_battery
    ct = scenario.coordination_type
    if ct == 'self_consumption':
        return self_consumption(site_id, solar_kwh, load_kwh,
                                battery_spec=battery_spec)
    if ct == 'perfect_foresight':
        raise RuntimeError(
            "perfect_foresight is handled in run_scenario (_run_perfect_foresight), "
            "not _dispatch_for_site; this branch should be unreachable."
        )
    if ct == 'perfect_foresight_custom':
        raise RuntimeError(
            "perfect_foresight_custom is handled in run_scenario "
            "(_run_perfect_foresight_custom), not _dispatch_for_site; "
            "this branch should be unreachable."
        )
    if ct == 'perfect_foresight_advanced':
        raise RuntimeError(
            "perfect_foresight_advanced is handled in run_scenario "
            "(_run_perfect_foresight_advanced), not _dispatch_for_site; "
            "this branch should be unreachable."
        )
    if ct in ('vpp', 'vpp_sc_event', 'vpp_tou_event', 'self_hedging'):
        if ct == 'self_hedging':
            raise NotImplementedError(
                f"coordination_type='self_hedging' not yet implemented"
            )
        raise RuntimeError(
            f"{ct!r} is handled in run_scenario, not _dispatch_for_site; "
            "this branch should be unreachable."
        )
    raise ValueError(f"Unknown coordination_type: {ct!r}")


# ----------------------------------------------------------------------
# Main entry: run a single scenario
# ----------------------------------------------------------------------

def run_scenario(
    scenario: ScenarioConfig,
    paths: Paths,
    *,
    progress_callback=None,
    ingest_progress_callback=None,
    n_workers: int = 1,
    max_sites: Optional[int] = None,
    with_monthly_bills: bool = False,
    only_sites: Optional[list[str]] = None,
) -> ScenarioResult:
    """Execute one scenario end-to-end and return per-site traces and bills.

    Args:
        scenario: Fully-validated ScenarioConfig.
        paths: Filesystem paths to inputs and tariffs.
        progress_callback: Optional callable(i, n, site_id, note="") called
            once per site during the dispatch+billing phase.
        ingest_progress_callback: Optional callable(i, n, site_id) called
            once per site during the ingest phase.
        n_workers: Number of parallel worker processes for perfect_foresight
            scenarios. Each site's LP solve is independent, so n_workers>1
            gives near-linear speedup up to the number of physical cores.
            Defaults to 1 (serial). Ignored for non-LP coordination types.
            Recommended: n_physical_cores - 1 (leave one core for the OS and
            main process). For an 8-core i9: n_workers=6.
        max_sites: If set, process only the first N sites after ingest.
        only_sites: If set, process only these sites, in the ingest order.
            Applied before max_sites. Raises ValueError if any is not in the
            scenario's site set.
            Useful for quick timing benchmarks without editing the YAML.
        with_monthly_bills: If True, also compute per-site monthly SiteBills
            (result.site_monthly_bills) for the main dispatch bill only (not
            the self_consumption baseline). Uses
            billing.compute_bill_and_monthly_bills so the annual and monthly
            bills for a site share one settlement build rather than each
            independently re-deriving rates/local time -- see that
            function's docstring. Off by default: even with that sharing,
            deriving monthly bills is not free, and most callers only need
            the annual bill.

    Returns:
        ScenarioResult with per-site SiteTraces and SiteBills, plus timing
        fields elapsed_seconds, ingest_seconds, and lp_seconds.
    """
    t0 = time.perf_counter()
    ctx = ingest(scenario, paths,
                 progress_callback=ingest_progress_callback)
    ingest_elapsed = time.perf_counter() - t0

    per_site_battery = _resolve_per_site_batteries(scenario, ctx.site_metadata)
    per_site_scale = _resolve_per_site_solar_scaling(scenario, ctx.site_metadata)

    site_ids = list(ctx.site_traces.keys())
    if only_sites is not None:
        missing = sorted(set(only_sites) - set(site_ids))
        if missing:
            raise ValueError(
                f"{scenario.scenario_id}: requested site(s) not in the "
                f"scenario's site set: {missing}")
        wanted = set(only_sites)
        site_ids = [s for s in site_ids if s in wanted]
    if max_sites is not None:
        site_ids = site_ids[:max_sites]
    n = len(site_ids)

    # perfect_foresight needs the self_consumption baseline (terminal SOC + VCR
    # comparison), so it is handled in a dedicated path that returns both the
    # LP result and the baseline.
    if scenario.coordination_type == 'perfect_foresight':
        return _run_perfect_foresight(
            scenario=scenario, ctx=ctx,
            per_site_battery=per_site_battery, per_site_scale=per_site_scale,
            site_ids=site_ids, t0=t0, ingest_elapsed=ingest_elapsed,
            progress_callback=progress_callback,
            workers=n_workers, with_monthly_bills=with_monthly_bills,
        )

    # perfect_foresight_custom uses the same baseline/VCR structure as LP
    # but runs the greedy heuristic instead, passing sc_traces to avoid a
    # double self_consumption call per site.
    if scenario.coordination_type == 'perfect_foresight_custom':
        return _run_perfect_foresight_custom(
            scenario=scenario, ctx=ctx,
            per_site_battery=per_site_battery, per_site_scale=per_site_scale,
            site_ids=site_ids, t0=t0, ingest_elapsed=ingest_elapsed,
            progress_callback=progress_callback,
            workers=n_workers, with_monthly_bills=with_monthly_bills,
        )

    if scenario.coordination_type == 'perfect_foresight_advanced':
        return _run_perfect_foresight_advanced(
            scenario=scenario, ctx=ctx,
            per_site_battery=per_site_battery, per_site_scale=per_site_scale,
            site_ids=site_ids, t0=t0, ingest_elapsed=ingest_elapsed,
            progress_callback=progress_callback,
            workers=n_workers, with_monthly_bills=with_monthly_bills,
        )

    if scenario.coordination_type in ('rolling_vpp_sc_event',
                                      'rolling_vpp_tou_event'):
        # One runner for both; the worker selects gap_strategy from
        # coordination_type. This branch sits BEFORE the
        # ROLLING_COORDINATION_TYPES branch below, which would otherwise catch
        # these types and dispatch them to the wholesale/advanced runner --
        # a plausible wrong answer rather than a failure.
        return _run_rolling_vpp_event(
            scenario=scenario, ctx=ctx, paths=paths,
            per_site_battery=per_site_battery, per_site_scale=per_site_scale,
            site_ids=site_ids, t0=t0, ingest_elapsed=ingest_elapsed,
            progress_callback=progress_callback,
            workers=n_workers, with_monthly_bills=with_monthly_bills,
        )

    if scenario.coordination_type in ROLLING_COORDINATION_TYPES:
        # One bulk runner for both non-VPP rolling types. Everything it does --
        # per-site args assembly, the serial/parallel split, causality
        # abort, diagnostics collection -- is identical; only the
        # scenario-level price setup and the per-site runner differ, and
        # both branch inside. A second near-copy would drift on causality,
        # which is the property that must not.
        return _run_rolling_perfect_foresight_lp(
            scenario=scenario, ctx=ctx,
            per_site_battery=per_site_battery, per_site_scale=per_site_scale,
            site_ids=site_ids, t0=t0, ingest_elapsed=ingest_elapsed,
            progress_callback=progress_callback,
            workers=n_workers, with_monthly_bills=with_monthly_bills,
        )

    if scenario.coordination_type == 'vpp_sc_event':
        return _run_vpp_sc_event(
            scenario=scenario, ctx=ctx, paths=paths,
            per_site_battery=per_site_battery, per_site_scale=per_site_scale,
            site_ids=site_ids, t0=t0, ingest_elapsed=ingest_elapsed,
            progress_callback=progress_callback,
            workers=n_workers, with_monthly_bills=with_monthly_bills,
        )

    if scenario.coordination_type == 'vpp_tou_event':
        return _run_vpp_sc_event(
            scenario=scenario, ctx=ctx, paths=paths,
            per_site_battery=per_site_battery, per_site_scale=per_site_scale,
            site_ids=site_ids, t0=t0, ingest_elapsed=ingest_elapsed,
            progress_callback=progress_callback,
            workers=n_workers, with_monthly_bills=with_monthly_bills,
            gap_strategy='tou_advanced',
        )

    # Standard single-trace path: no_cer / solar / self_consumption.
    traces: dict[str, SiteTraces] = {}
    bills: dict[str, SiteBill] = {}
    monthly_bills: Optional[dict[str, dict[int, SiteBill]]] = (
        {} if with_monthly_bills else None
    )
    for i, site_id in enumerate(site_ids, start=1):
        st = _dispatch_for_site(
            site_id=site_id,
            site_trace=ctx.site_traces[site_id],
            battery_spec=per_site_battery[site_id],
            solar_scale=per_site_scale[site_id],
            scenario=scenario,
            prices=ctx.prices,
            network_tariff=ctx.network_tariff,
            retail_tariff=ctx.retail_tariff,
        )
        if with_monthly_bills:
            bill, site_monthly = compute_bill_and_monthly_bills(
                st, ctx.prices, ctx.network_tariff, ctx.retail_tariff,
                scenario.state,
            )
            monthly_bills[site_id] = site_monthly
        else:
            bill = compute_bill(st, ctx.prices, ctx.network_tariff,
                                ctx.retail_tariff, scenario.state)
        traces[site_id] = st
        bills[site_id] = bill
        if progress_callback is not None:
            progress_callback(i, n, site_id)

    elapsed = time.perf_counter() - t0
    return ScenarioResult(
        scenario=scenario, site_traces=traces, site_bills=bills,
        elapsed_seconds=elapsed, ingest_seconds=ingest_elapsed,
        site_monthly_bills=monthly_bills,
    )


def _perfect_foresight_site_worker(args: tuple) -> tuple:
    """Module-level worker for ProcessPoolExecutor — must be picklable.

    Runs self_consumption then perfect_foresight_lp for a single site and
    returns everything the caller needs to assemble ScenarioResult.

    Warnings from perfect_foresight_lp (e.g. solver degeneracy) are caught
    and returned as strings so the main process can emit them cleanly rather
    than having them appear attributed to concurrent.futures/process.py.

    Args:
        args: (site_id, solar_kwh, load_kwh, p_imp, p_exp, battery_spec,
               prices_mwh, network_tariff, retail_tariff,
               soc_min, export_limit_kwh, interval_hours, state,
               with_monthly_bills)

    Returns:
        (site_id, sc_t, sc_bill, lp_t, lp_bill, lp_dt, warning_msgs,
         lp_monthly_bills)
        lp_monthly_bills is None unless with_monthly_bills was True, in
        which case it's the dict[int, SiteBill] for lp_t/lp_bill (the
        self_consumption baseline is never split by month -- see
        run_scenario's with_monthly_bills docstring).
    """
    import warnings as _warnings
    (site_id, solar_kwh, load_kwh, p_imp, p_exp, battery_spec,
     prices_mwh, network_tariff, retail_tariff,
     soc_min, export_limit_kwh, interval_hours, state,
     with_monthly_bills) = args

    sc_t = self_consumption(site_id, solar_kwh, load_kwh, battery_spec,
                            interval_hours)
    sc_bill = compute_bill(sc_t, prices_mwh, network_tariff, retail_tariff,
                           state, interval_hours)

    soc_final = float(sc_t.battery_soc_kwh.iloc[-1])
    soc_final = min(max(soc_final, soc_min), battery_spec.capacity_kwh)

    lp_start = time.perf_counter()
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        lp_t = perfect_foresight_lp(
            site_id, solar_kwh, load_kwh, p_imp, p_exp, battery_spec,
            soc_init_kwh=soc_min, soc_final_kwh=soc_final,
            export_limit_kwh_per_interval=export_limit_kwh,
            interval_hours=interval_hours,
        )
    lp_dt = time.perf_counter() - lp_start

    lp_monthly_bills = None
    if with_monthly_bills:
        lp_bill, lp_monthly_bills = compute_bill_and_monthly_bills(
            lp_t, prices_mwh, network_tariff, retail_tariff,
            state, interval_hours,
        )
    else:
        lp_bill = compute_bill(lp_t, prices_mwh, network_tariff, retail_tariff,
                               state, interval_hours)

    warning_msgs = [str(w.message) for w in caught]
    return site_id, sc_t, sc_bill, lp_t, lp_bill, lp_dt, warning_msgs, lp_monthly_bills


def _perfect_foresight_custom_site_worker(args: tuple) -> tuple:
    """Module-level worker for ProcessPoolExecutor — must be picklable.

    Runs self_consumption then perfect_foresight_custom for a single site.
    The self_consumption result is passed directly as sc_traces so the
    greedy does not rerun it internally (avoids the double SC cost).

    Args:
        args: (site_id, solar_kwh, load_kwh, p_imp, p_exp, battery_spec,
               prices_mwh, network_tariff, retail_tariff,
               soc_min, export_limit_kwh, interval_hours, opt_factor, state,
               with_monthly_bills)

    Returns:
        (site_id, sc_t, sc_bill, custom_t, custom_bill, custom_dt, [],
         custom_monthly_bills)
        custom_monthly_bills is None unless with_monthly_bills was True.
    """
    (site_id, solar_kwh, load_kwh, p_imp, p_exp, battery_spec,
     prices_mwh, network_tariff, retail_tariff,
     soc_min, export_limit_kwh, interval_hours, opt_factor, state,
     with_monthly_bills) = args

    sc_t = self_consumption(site_id, solar_kwh, load_kwh, battery_spec,
                            interval_hours,
                            neg_price_aware=True,
                            price_imp=p_imp, price_exp=p_exp)
    sc_bill = compute_bill(sc_t, prices_mwh, network_tariff, retail_tariff,
                           state, interval_hours)

    soc_final = float(sc_t.battery_soc_kwh.iloc[-1])
    soc_final = min(max(soc_final, soc_min), battery_spec.capacity_kwh)

    custom_start = time.perf_counter()
    custom_t = perfect_foresight_custom(
        site_id, solar_kwh, load_kwh, p_imp, p_exp, battery_spec,
        soc_init_kwh=soc_min, soc_final_kwh=soc_final,
        export_limit_kwh_per_interval=export_limit_kwh,
        interval_hours=interval_hours,
        opt_factor=opt_factor,
        sc_traces=sc_t,
    )
    custom_dt = time.perf_counter() - custom_start

    custom_monthly_bills = None
    if with_monthly_bills:
        custom_bill, custom_monthly_bills = compute_bill_and_monthly_bills(
            custom_t, prices_mwh, network_tariff, retail_tariff,
            state, interval_hours,
        )
    else:
        custom_bill = compute_bill(custom_t, prices_mwh, network_tariff,
                                   retail_tariff, state, interval_hours)

    return (site_id, sc_t, sc_bill, custom_t, custom_bill, custom_dt, [],
            custom_monthly_bills)


def _run_perfect_foresight(
    *,
    scenario: ScenarioConfig,
    ctx,
    per_site_battery: dict[str, Optional[BatterySpec]],
    per_site_scale: dict[str, float],
    site_ids: list[str],
    t0: float,
    ingest_elapsed: float,
    progress_callback=None,
    workers: int = 1,
    with_monthly_bills: bool = False,
) -> ScenarioResult:
    """Per-site self_consumption baseline then perfect_foresight_lp.

    For each site, sequentially (workers=1) or in parallel (workers>1):
      1. Run self_consumption -> baseline trace + bill, and its end-of-period
         SOC (used to seed the LP's terminal SOC for a fair comparison).
      2. Run perfect_foresight_lp seeded with that terminal SOC -> optimal
         trace + bill.

    Sequential per-site (workers=1) keeps only one site's traces alive at a
    time. Parallel execution (workers>1) uses a ProcessPoolExecutor so each
    worker runs in its own process — safe for both HiGHS and Gurobi, each of
    which is independently thread-safe within a single process. Workers
    default to 1 (serial) so behaviour is unchanged unless explicitly set.

    The LP traces/bills become the main ScenarioResult.site_traces/site_bills;
    the self_consumption traces/bills are stored as baseline_traces/baseline_bills
    for VCR reporting.
    """
    n = len(site_ids)

    per_site_export_limit = _resolve_per_site_export_limits(
        scenario, ctx.site_metadata)

    idx = ctx.prices.index
    interval_hours = (idx[1] - idx[0]).total_seconds() / 3600.0

    # Price vectors are identical across sites for a scenario -> build once.
    p_imp, p_exp = build_price_vectors(
        idx, ctx.prices, ctx.network_tariff, ctx.retail_tariff,
        scenario.state, interval_hours)

    # Build the per-site argument tuples for the worker function.
    # p_imp/p_exp are passed as Series so the worker has the correct index;
    # prices_mwh and tariffs are passed for billing inside the worker.
    site_args = []
    for site_id in site_ids:
        battery_spec = per_site_battery[site_id]
        site_trace = ctx.site_traces[site_id]
        solar_kwh = site_trace['pv_site_net_kwh'] * per_site_scale[site_id]
        load_kwh = site_trace['gross_load_kwh']
        soc_min = (battery_spec.backup_reserve_pct / 100.0) * battery_spec.capacity_kwh
        export_limit_kwh = per_site_export_limit[site_id] * interval_hours
        site_args.append((
            site_id, solar_kwh, load_kwh, p_imp, p_exp, battery_spec,
            ctx.prices, ctx.network_tariff, ctx.retail_tariff,
            soc_min, export_limit_kwh, interval_hours, scenario.state,
            with_monthly_bills,
        ))

    traces: dict[str, SiteTraces] = {}
    bills: dict[str, SiteBill] = {}
    baseline_traces: dict[str, SiteTraces] = {}
    baseline_bills: dict[str, SiteBill] = {}
    monthly_bills: Optional[dict[str, dict[int, SiteBill]]] = (
        {} if with_monthly_bills else None
    )
    lp_seconds = 0.0

    if workers == 1:
        # Serial path: predictable progress reporting, minimal memory footprint.
        for i, (site_id, args) in enumerate(zip(site_ids, site_args), start=1):
            sid, sc_t, sc_bill, lp_t, lp_bill, lp_dt, warning_msgs, lp_monthly = (
                _perfect_foresight_site_worker(args))
            for msg in warning_msgs:
                warnings.warn(msg, stacklevel=2)
            traces[sid] = lp_t
            bills[sid] = lp_bill
            baseline_traces[sid] = sc_t
            baseline_bills[sid] = sc_bill
            if with_monthly_bills:
                monthly_bills[sid] = lp_monthly
            lp_seconds += lp_dt
            if progress_callback is not None:
                progress_callback(i, n, sid, note=f"LP {lp_dt:.1f}s")
    else:
        # Parallel path: submit all sites, collect as futures complete.
        # Results arrive out of order; progress counter increments on each
        # completion. site order in dicts is insertion order (py3.7+) so
        # we restore it after collection.
        import multiprocessing as _mp
        completed = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_mp.get_context('spawn'),
        ) as executor:
            future_to_site = {
                executor.submit(_perfect_foresight_site_worker, args): site_id
                for site_id, args in zip(site_ids, site_args)
            }
            for future in as_completed(future_to_site):
                sid, sc_t, sc_bill, lp_t, lp_bill, lp_dt, warning_msgs, lp_monthly = (
                    future.result())
                for msg in warning_msgs:
                    warnings.warn(msg, stacklevel=2)
                traces[sid] = lp_t
                bills[sid] = lp_bill
                baseline_traces[sid] = sc_t
                baseline_bills[sid] = sc_bill
                if with_monthly_bills:
                    monthly_bills[sid] = lp_monthly
                lp_seconds += lp_dt
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, n, sid,
                                      note=f"LP {lp_dt:.1f}s")

    elapsed = time.perf_counter() - t0
    return ScenarioResult(
        scenario=scenario, site_traces=traces, site_bills=bills,
        elapsed_seconds=elapsed, ingest_seconds=ingest_elapsed,
        lp_seconds=lp_seconds, n_workers=workers,
        baseline_traces=baseline_traces, baseline_bills=baseline_bills,
        site_monthly_bills=monthly_bills,
    )


def _run_perfect_foresight_custom(
    *,
    scenario: ScenarioConfig,
    ctx,
    per_site_battery: dict[str, Optional[BatterySpec]],
    per_site_scale: dict[str, float],
    site_ids: list[str],
    t0: float,
    ingest_elapsed: float,
    progress_callback=None,
    workers: int = 1,
    with_monthly_bills: bool = False,
) -> ScenarioResult:
    """Per-site self_consumption baseline then perfect_foresight_custom.

    Mirrors _run_perfect_foresight exactly, substituting the greedy heuristic
    for the LP solver. The self_consumption result is passed directly to the
    custom component as sc_traces to avoid running it twice per site.

    opt_factor is taken from scenario.opt_factor if available, defaulting to
    1.0 (full optimisation). Set a lower value in the scenario YAML to trade
    solution quality for speed once ScenarioConfig exposes the field.
    """
    n = len(site_ids)

    per_site_export_limit = _resolve_per_site_export_limits(
        scenario, ctx.site_metadata)

    idx = ctx.prices.index
    interval_hours = (idx[1] - idx[0]).total_seconds() / 3600.0

    p_imp, p_exp = build_price_vectors(
        idx, ctx.prices, ctx.network_tariff, ctx.retail_tariff,
        scenario.state, interval_hours)

    opt_factor = getattr(scenario, 'opt_factor', 1.0)

    site_args = []
    for site_id in site_ids:
        battery_spec = per_site_battery[site_id]
        site_trace = ctx.site_traces[site_id]
        solar_kwh = site_trace['pv_site_net_kwh'] * per_site_scale[site_id]
        load_kwh = site_trace['gross_load_kwh']
        soc_min = (battery_spec.backup_reserve_pct / 100.0) * battery_spec.capacity_kwh
        export_limit_kwh = per_site_export_limit[site_id] * interval_hours
        site_args.append((
            site_id, solar_kwh, load_kwh, p_imp, p_exp, battery_spec,
            ctx.prices, ctx.network_tariff, ctx.retail_tariff,
            soc_min, export_limit_kwh, interval_hours, opt_factor,
            scenario.state, with_monthly_bills,
        ))

    traces: dict[str, SiteTraces] = {}
    bills: dict[str, SiteBill] = {}
    baseline_traces: dict[str, SiteTraces] = {}
    baseline_bills: dict[str, SiteBill] = {}
    monthly_bills: Optional[dict[str, dict[int, SiteBill]]] = (
        {} if with_monthly_bills else None
    )
    custom_seconds = 0.0

    if workers == 1:
        for i, (site_id, args) in enumerate(zip(site_ids, site_args), start=1):
            sid, sc_t, sc_bill, custom_t, custom_bill, custom_dt, _, custom_monthly = (
                _perfect_foresight_custom_site_worker(args))
            traces[sid] = custom_t
            bills[sid] = custom_bill
            baseline_traces[sid] = sc_t
            baseline_bills[sid] = sc_bill
            if with_monthly_bills:
                monthly_bills[sid] = custom_monthly
            custom_seconds += custom_dt
            if progress_callback is not None:
                progress_callback(i, n, sid, note=f"custom {custom_dt:.1f}s")
    else:
        import multiprocessing as _mp
        completed = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_mp.get_context('spawn'),
        ) as executor:
            future_to_site = {
                executor.submit(_perfect_foresight_custom_site_worker, args): site_id
                for site_id, args in zip(site_ids, site_args)
            }
            for future in as_completed(future_to_site):
                sid, sc_t, sc_bill, custom_t, custom_bill, custom_dt, _, custom_monthly = (
                    future.result())
                traces[sid] = custom_t
                bills[sid] = custom_bill
                baseline_traces[sid] = sc_t
                baseline_bills[sid] = sc_bill
                if with_monthly_bills:
                    monthly_bills[sid] = custom_monthly
                custom_seconds += custom_dt
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, n, sid,
                                      note=f"custom {custom_dt:.1f}s")

    elapsed = time.perf_counter() - t0
    return ScenarioResult(
        scenario=scenario, site_traces=traces, site_bills=bills,
        elapsed_seconds=elapsed, ingest_seconds=ingest_elapsed,
        lp_seconds=custom_seconds, n_workers=workers,
        baseline_traces=baseline_traces, baseline_bills=baseline_bills,
        site_monthly_bills=monthly_bills,
    )


def _perfect_foresight_advanced_site_worker(args: tuple) -> tuple:
    """Module-level worker for ProcessPoolExecutor — must be picklable.

    Mirrors _perfect_foresight_site_worker, substituting
    perfect_foresight_lp_advanced for perfect_foresight_lp and passing the
    two extra arguments it requires: retail_tariff and prices_mwh.

    The price vectors (p_imp, p_exp) are pre-built once per scenario by the
    orchestrator via build_lp_price_vectors_advanced, then passed to each
    worker as Series (with the correct index). This matches the standard
    worker pattern exactly.

    Args:
        args: (site_id, solar_kwh, load_kwh, p_imp, p_exp, battery_spec,
               prices_mwh, network_tariff, retail_tariff,
               soc_min, export_limit_kwh, interval_hours,
               apply_export_order_nudge, state, with_monthly_bills)

    Returns:
        (site_id, sc_t, sc_bill, adv_t, adv_bill, adv_dt, warning_msgs,
         adv_monthly_bills)
        adv_monthly_bills is None unless with_monthly_bills was True.
    """
    import warnings as _warnings
    (site_id, solar_kwh, load_kwh, p_imp, p_exp, battery_spec,
     prices_mwh, network_tariff, retail_tariff,
     soc_min, export_limit_kwh, interval_hours,
     apply_export_order_nudge, state, with_monthly_bills) = args

    sc_t = self_consumption(site_id, solar_kwh, load_kwh, battery_spec,
                            interval_hours)
    sc_bill = compute_bill(sc_t, prices_mwh, network_tariff, retail_tariff,
                           state, interval_hours)

    soc_final = float(sc_t.battery_soc_kwh.iloc[-1])
    soc_final = min(max(soc_final, soc_min), battery_spec.capacity_kwh)

    adv_start = time.perf_counter()
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        adv_t = perfect_foresight_lp_advanced(
            site_id, solar_kwh, load_kwh, p_imp, p_exp, battery_spec,
            soc_init_kwh=soc_min, soc_final_kwh=soc_final,
            retail_tariff=retail_tariff,
            prices_mwh=prices_mwh,
            state=state,
            export_limit_kwh_per_interval=export_limit_kwh,
            interval_hours=interval_hours,
            apply_export_order_nudge=apply_export_order_nudge,
            curtail_cost_per_kwh_dollars=dispatch_curtail_cost(retail_tariff),
        )
    adv_dt = time.perf_counter() - adv_start

    adv_monthly_bills = None
    if with_monthly_bills:
        adv_bill, adv_monthly_bills = compute_bill_and_monthly_bills(
            adv_t, prices_mwh, network_tariff, retail_tariff,
            state, interval_hours,
        )
    else:
        adv_bill = compute_bill(adv_t, prices_mwh, network_tariff, retail_tariff,
                                state, interval_hours)

    warning_msgs = [str(w.message) for w in caught]
    return site_id, sc_t, sc_bill, adv_t, adv_bill, adv_dt, warning_msgs, adv_monthly_bills


def _run_perfect_foresight_advanced(
    *,
    scenario: ScenarioConfig,
    ctx,
    per_site_battery: dict[str, Optional[BatterySpec]],
    per_site_scale: dict[str, float],
    site_ids: list[str],
    t0: float,
    ingest_elapsed: float,
    progress_callback=None,
    workers: int = 1,
    with_monthly_bills: bool = False,
) -> ScenarioResult:
    """Per-site self_consumption baseline then perfect_foresight_lp_advanced.

    Mirrors _run_perfect_foresight exactly, with two differences:
      1. Price vectors are built via build_lp_price_vectors_advanced (applies
         block masking and critical rates) rather than build_price_vectors.
      2. The advanced worker is used, which passes retail_tariff and prices_mwh
         to perfect_foresight_lp_advanced.

    All other logic (baseline SOC, serial/parallel dispatch, VCR output
    structure) is identical to the standard LP orchestrator.

    The advanced LP supports all standard tariff plans in addition to plans
    with has_advanced_features=True -- the advanced and standard functions
    produce identical results for standard plans.
    """
    n = len(site_ids)

    per_site_export_limit = _resolve_per_site_export_limits(
        scenario, ctx.site_metadata)

    idx = ctx.prices.index
    interval_hours = (idx[1] - idx[0]).total_seconds() / 3600.0

    # Price vectors built once per scenario -- advanced variant handles block
    # masking and critical rates. For standard plans this is equivalent to
    # build_price_vectors.
    p_imp, p_exp = build_lp_price_vectors_advanced(
        idx, ctx.prices, ctx.network_tariff, ctx.retail_tariff,
        scenario.state, interval_hours)

    apply_export_order_nudge = scenario.apply_export_order_nudge

    site_args = []
    for site_id in site_ids:
        battery_spec = per_site_battery[site_id]
        site_trace = ctx.site_traces[site_id]
        solar_kwh = site_trace['pv_site_net_kwh'] * per_site_scale[site_id]
        load_kwh = site_trace['gross_load_kwh']
        soc_min = (battery_spec.backup_reserve_pct / 100.0) * battery_spec.capacity_kwh
        export_limit_kwh = per_site_export_limit[site_id] * interval_hours
        site_args.append((
            site_id, solar_kwh, load_kwh, p_imp, p_exp, battery_spec,
            ctx.prices, ctx.network_tariff, ctx.retail_tariff,
            soc_min, export_limit_kwh, interval_hours,
            apply_export_order_nudge, scenario.state, with_monthly_bills,
        ))

    traces: dict[str, SiteTraces] = {}
    bills: dict[str, SiteBill] = {}
    baseline_traces: dict[str, SiteTraces] = {}
    baseline_bills: dict[str, SiteBill] = {}
    monthly_bills: Optional[dict[str, dict[int, SiteBill]]] = (
        {} if with_monthly_bills else None
    )
    lp_seconds = 0.0

    if workers == 1:
        for i, (site_id, args) in enumerate(zip(site_ids, site_args), start=1):
            sid, sc_t, sc_bill, adv_t, adv_bill, adv_dt, warning_msgs, adv_monthly = (
                _perfect_foresight_advanced_site_worker(args))
            for msg in warning_msgs:
                warnings.warn(msg, stacklevel=2)
            traces[sid] = adv_t
            bills[sid] = adv_bill
            baseline_traces[sid] = sc_t
            baseline_bills[sid] = sc_bill
            if with_monthly_bills:
                monthly_bills[sid] = adv_monthly
            lp_seconds += adv_dt
            if progress_callback is not None:
                progress_callback(i, n, sid, note=f"adv {adv_dt:.1f}s")
    else:
        import multiprocessing as _mp
        completed = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_mp.get_context('spawn'),
        ) as executor:
            future_to_site = {
                executor.submit(_perfect_foresight_advanced_site_worker, args): site_id
                for site_id, args in zip(site_ids, site_args)
            }
            for future in as_completed(future_to_site):
                sid, sc_t, sc_bill, adv_t, adv_bill, adv_dt, warning_msgs, adv_monthly = (
                    future.result())
                for msg in warning_msgs:
                    warnings.warn(msg, stacklevel=2)
                traces[sid] = adv_t
                bills[sid] = adv_bill
                baseline_traces[sid] = sc_t
                baseline_bills[sid] = sc_bill
                if with_monthly_bills:
                    monthly_bills[sid] = adv_monthly
                lp_seconds += adv_dt
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, n, sid,
                                      note=f"adv {adv_dt:.1f}s")

    elapsed = time.perf_counter() - t0
    return ScenarioResult(
        scenario=scenario, site_traces=traces, site_bills=bills,
        elapsed_seconds=elapsed, ingest_seconds=ingest_elapsed,
        lp_seconds=lp_seconds, n_workers=workers,
        baseline_traces=baseline_traces, baseline_bills=baseline_bills,
        site_monthly_bills=monthly_bills,
    )


def build_rolling_price_provider(
    rolling_config: RollingConfig,
    prices_mwh: pd.Series,
    interval_hours: float,
    *,
    region: Optional[str] = None,
    store_root: Path = DEFAULT_PREDISPATCH_STORE,
):
    """Build the price-forecast provider for a RollingConfig.

    The single place the horizon_hours/interval_hours -> interval_minutes and
    infill_lag_hours -> infill_lag_intervals conversions happen. Both the
    orchestrator's per-scenario setup (_run_rolling_perfect_foresight_lp,
    via the flat-primitives copy of this logic in _init_rolling_worker -- see
    that function's docstring for why it can't just call this) and any
    single-site script (viz_rolling_perfect_foresight_lp.py) derive from the
    same arithmetic rather than each re-deriving it -- exactly the kind of
    independently-maintained duplicate that has caused drift elsewhere in
    this project (see B_Clean_Predispatch/C_Validate_Predispatch's shared
    KNOWN_ARCHIVE_GAPS for the same reasoning applied to a different pair).

    region is required for forecast_source='predispatch' only (see
    registry.build_provider), and unused otherwise.

    on_degraded_period and max_staleness_hours are read from rolling_config
    (both are real RollingConfig fields -- see model/core/scenario.py) rather
    than taken as separate keyword arguments here, so there is exactly one
    place a scenario's choice on the October-2022-shaped problem is recorded,
    not a config field the orchestrator ignores in favour of a
    separately-passed override.
    """
    interval_minutes = int(round(interval_hours * 60))
    infill_lag_intervals = int(round(
        rolling_config.infill_lag_hours * 60 / interval_minutes))
    return build_provider(
        rolling_config.forecast_source, prices_mwh,
        interval_minutes=interval_minutes,
        mode=rolling_config.current_interval_price,
        infill_lag_intervals=infill_lag_intervals,
        store_root=store_root, region=region,
        on_degraded_period=rolling_config.on_degraded_period,
        max_staleness_hours=rolling_config.max_staleness_hours,
    )


def run_rolling_perfect_foresight_lp_site(
    site_id: str,
    solar_kwh: pd.Series,
    load_kwh: pd.Series,
    battery_spec: BatterySpec,
    offsets: Optional[PriceStackOffsets],
    provider,
    rolling_config: RollingConfig,
    *,
    export_limit_kwh_per_interval: float = float('inf'),
    interval_hours: float = 5 / 60,
    state: str,
    sc_result: Optional[SiteTraces] = None,
    planner=None,
) -> tuple[SiteTraces, SiteTraces, RollingDiagnostics]:
    """Run self_consumption (unless supplied) then rolling_optimisation for
    ONE site, with every piece of wiring this project has settled on for the
    rolling_perfect_foresight_lp coordination type.

    This is THE place the following conventions live, so that the
    orchestrator worker and any single-site diagnostic script call the exact
    same code rather than hand-mirroring it (which is how the two-copies risk
    the person flagged would actually happen):
      - SOC pins: soc_init_kwh is the battery's reserve floor;
        soc_final_kwh is the self_consumption baseline's own terminal SOC,
        clamped to [soc_min, capacity] -- the same convention
        _perfect_foresight_site_worker uses for perfect_foresight_lp, so the
        VCR comparison against both self_consumption and perfect_foresight_lp
        is fair (see rolling_optimisation's own soc_final_kwh docstring).
      - commit_actual_load is read from rolling_config, never hardcoded.
      - Usage/solar forecast providers are built from the site's OWN load/
        solar trace with NO valid= mask: every interval enters the profile
        history irrespective of infill status (confirmed decision -- fitting
        those forecasters was restricted to clean data, but a site being
        DISPATCHED needs a forecast at every interval regardless of how that
        interval's data was sourced). Built only when rolling_config.
        usage_forecast / solar_forecast is not None.

    Args:
        provider: A ForecastProvider already built via
            build_rolling_price_provider (or equivalent) for the SAME prices
            this site's solar_kwh/load_kwh index covers.
        sc_result: Pass the already-computed self_consumption SiteTraces to
            avoid rerunning it (the orchestrator worker needs it separately
            for the baseline bill anyway). None runs it here.
        planner: Window planner for an advanced TOU plan. None uses the
            default wholesale planner built from offsets, which is what
            rolling_perfect_foresight_lp wants.

            Threaded through here rather than given its own near-identical
            twin function, because everything above -- the SOC pins, the
            usage/solar provider construction, commit_actual_load -- is
            identical for both coordination types, and a second copy would
            drift on exactly the conventions this function exists to hold in
            one place. For the advanced path offsets is None and provider may
            be None; see run_rolling_perfect_foresight_lp_advanced_site.

    Returns:
        (sc_result, rolling_traces, diagnostics). sc_result is returned even
        when supplied, so callers can use one return signature either way.
    """
    if sc_result is None:
        sc_result = self_consumption(site_id, solar_kwh, load_kwh,
                                     battery_spec, interval_hours)

    soc_min = (battery_spec.backup_reserve_pct / 100.0) * battery_spec.capacity_kwh
    soc_final = float(sc_result.battery_soc_kwh.iloc[-1])
    soc_final = min(max(soc_final, soc_min), battery_spec.capacity_kwh)

    n_h = int(round(rolling_config.horizon_hours / interval_hours))
    usage_provider = None
    if rolling_config.usage_forecast is not None:
        usage_provider = UsageForecastProvider.from_series(
            load_kwh, state, rolling_config.usage_forecast,
            max_horizon_intervals=n_h, interval_hours=interval_hours)
    solar_provider = None
    if rolling_config.solar_forecast is not None:
        solar_provider = SolarForecastProvider.from_series(
            solar_kwh, state, rolling_config.solar_forecast,
            max_horizon_intervals=n_h, interval_hours=interval_hours)

    roll_t, diagnostics = rolling_optimisation(
        site_id, solar_kwh, load_kwh, provider, offsets, battery_spec,
        soc_init_kwh=soc_min, soc_final_kwh=soc_final, config=rolling_config,
        planner=planner,
        export_limit_kwh_per_interval=export_limit_kwh_per_interval,
        interval_hours=interval_hours,
        load_forecast_provider=usage_provider,
        solar_forecast_provider=solar_provider,
        commit_actual_load=rolling_config.commit_actual_load,
    )
    return sc_result, roll_t, diagnostics


def run_rolling_perfect_foresight_lp_advanced_site(
    site_id: str,
    solar_kwh: pd.Series,
    load_kwh: pd.Series,
    battery_spec: BatterySpec,
    network_tariff: NetworkTariff,
    retail_tariff: RetailTariff,
    provider,
    rolling_config: RollingConfig,
    *,
    export_limit_kwh_per_interval: float = float('inf'),
    interval_hours: float = 5 / 60,
    state: str,
    sc_result: Optional[SiteTraces] = None,
    settled_rrp_mwh: Optional[pd.Series] = None,
    apply_export_order_nudge: bool = False,
) -> tuple[SiteTraces, SiteTraces, RollingDiagnostics]:
    """The advanced TOU twin of run_rolling_perfect_foresight_lp_site.

    Builds the window planner for this site and delegates everything else --
    SOC pins, usage/solar provider construction, commit_actual_load -- to the
    shared function, so the two coordination types cannot drift apart on any
    of it.

    The planner is built per site even though its expensive parts (base price
    vectors, block window masks, day map) depend only on the index, tariff and
    state, so a scenario with many sites on one tariff repeats that work. Left
    that way deliberately: it is a fraction of one site's solve time and
    caching it across a process pool is a real complication to add before
    there is a measurement asking for it.

    Args:
        provider: price-forecast provider, or None. Only needed when the plan
            has critical rates; on a plan without them nothing in the price
            vectors reads the wholesale price and the loop never asks.
        settled_rrp_mwh: the settled wholesale price for the period. Required
            when the plan has a critical rate on a side with a block window:
            the planner's day state counts block usage by the settled
            criticality of each committed interval, as billing does. Read
            only at interval k once k has been committed.
        apply_export_order_nudge: passed to AdvancedWindowPlanner
            (ScenarioConfig.apply_export_order_nudge).
    """
    planner = AdvancedWindowPlanner(
        index=solar_kwh.index,
        retail_tariff=retail_tariff,
        network_tariff=network_tariff,
        state=state,
        battery_spec=battery_spec,
        horizon_intervals=int(round(rolling_config.horizon_hours
                                    / interval_hours)),
        export_limit_kwh_per_interval=export_limit_kwh_per_interval,
        interval_hours=interval_hours,
        curtail_cost_per_kwh_dollars=dispatch_curtail_cost(retail_tariff),
        settled_rrp_mwh=settled_rrp_mwh,
        apply_export_order_nudge=apply_export_order_nudge,
    )
    return run_rolling_perfect_foresight_lp_site(
        site_id, solar_kwh, load_kwh, battery_spec,
        offsets=None, provider=provider, rolling_config=rolling_config,
        export_limit_kwh_per_interval=export_limit_kwh_per_interval,
        interval_hours=interval_hours, state=state, sc_result=sc_result,
        planner=planner,
    )


def advanced_plan_needs_price_forecast(retail_tariff: RetailTariff) -> bool:
    """Whether a rolling advanced run on this plan needs a price forecast.

    True when the plan has a critical rate, or is mixed (whose window prices
    are built from the forecast RRP). Used by validate_scenario
    and by the per-scenario setup so a plain block-TOU plan does not build a
    predispatch provider, check store coverage, or fall back to
    self-consumption over a forecast it could not have used.

    Reads the tariff rather than duplicating the rule, and the planner's own
    requires_price_forecast is the same predicate applied to the same fields.
    """
    return retail_tariff.type == 'mixed' or retail_tariff.has_critical_rates


def _init_rolling_worker(
    forecast_source: str, prices_mwh: pd.Series, interval_minutes: int,
    mode: str, infill_lag_intervals: int, store_root: Path, region: str,
    on_degraded_period: str, max_staleness_hours: float,
) -> None:
    """ProcessPoolExecutor initializer: build the price-forecast provider
    ONCE per worker process rather than once per site.

    Takes the already-converted (interval_minutes, infill_lag_intervals)
    rather than a RollingConfig, because THIS function's signature is also
    the ProcessPoolExecutor initargs tuple -- it must stay a flat, cheaply-
    picklable set of primitives and a Series, not a dataclass carrying
    Optional[UsageForecastConfig] etc. build_rolling_price_provider (called
    by viz_rolling_perfect_foresight_lp.py, which has no picklability
    constraint) does the RollingConfig -> primitives conversion via the same
    arithmetic; this just calls registry.build_provider directly with what
    it's given. on_degraded_period and max_staleness_hours are read from
    rolling_config at the call site in _run_rolling_perfect_foresight_lp,
    same as forecast_source/mode/infill_lag_intervals already are -- there is
    exactly one place (RollingConfig) that owns what a scenario does about a
    store gap, this function just receives the two extra scalars alongside
    the ones it already took.

    Why this exists (and the other per-site inputs below do not need it):
    for forecast_source='predispatch' the provider reads roughly 18 MB out of
    the canonical store (see rolling_optimisation_spec.md, "Memory"). Every
    other per-site input this worker uses (offsets, prices_mwh, tariffs) is
    small enough that duplicating it into every site's pickled args tuple is
    what _perfect_foresight_site_worker already does for p_imp/p_exp, so this
    module follows that precedent rather than special-casing everything.
    The provider is the one input where that precedent would actually cost
    something: naively embedding it in every site's args tuple would pickle
    ~18 MB per site, and a 436-site EDP scenario would move gigabytes.

    Runs in the child process under spawn, so it gets its own copy of this
    module's globals -- setting _ROLLING_PRICE_PROVIDER here does not touch
    the parent process or any other worker. For workers=1 (no executor), the
    caller invokes this directly in-process before the per-site loop, which
    has the identical effect via the identical code path.
    """
    global _ROLLING_PRICE_PROVIDER
    _ROLLING_PRICE_PROVIDER = build_provider(
        forecast_source, prices_mwh,
        interval_minutes=interval_minutes, mode=mode,
        infill_lag_intervals=infill_lag_intervals,
        store_root=store_root, region=region,
        on_degraded_period=on_degraded_period,
        max_staleness_hours=max_staleness_hours,
    )


_ROLLING_PRICE_PROVIDER = None  # set by _init_rolling_worker before use


def _rolling_perfect_foresight_site_worker(args: tuple) -> tuple:
    """Module-level worker for ProcessPoolExecutor -- must be picklable.

    Thin wrapper around run_rolling_perfect_foresight_lp_site: computes the
    baseline bill (which that function does not need but this worker's
    caller does, for baseline_bills) and the billing/monthly-bills glue,
    and reads the price provider from the module global rather than taking
    it as an argument -- see _init_rolling_worker's docstring for why.

    Args:
        args: (site_id, solar_kwh, load_kwh, battery_spec, offsets,
               prices_mwh, network_tariff, retail_tariff, soc_min,
               export_limit_kwh, interval_hours, state, rolling_config,
               with_monthly_bills, coordination_type,
               apply_export_order_nudge)

            coordination_type selects which per-site runner is called. An
            explicit field rather than inferring the advanced path from
            offsets being None: a reader meeting `offsets=None` in a tuple
            would reasonably take it for a bug, and this tuple crosses a
            pickle boundary where a wrong guess is expensive to trace.

    Returns:
        (site_id, sc_t, sc_bill, roll_t, roll_bill, roll_dt, warning_msgs,
         roll_monthly_bills, diagnostics)
        roll_monthly_bills is None unless with_monthly_bills was True.
        diagnostics is the RollingDiagnostics for this site; the caller is
        responsible for checking causality_ok (see rolling_optimisation_spec.md
        -- a causality violation invalidates the run, not just this site).
    """
    import warnings as _warnings
    (site_id, solar_kwh, load_kwh, battery_spec, offsets,
     prices_mwh, network_tariff, retail_tariff,
     soc_min, export_limit_kwh, interval_hours, state, rolling_config,
     with_monthly_bills, coordination_type,
     apply_export_order_nudge) = args
    advanced = coordination_type == 'rolling_perfect_foresight_lp_advanced'

    needs_prices = (not advanced
                    or advanced_plan_needs_price_forecast(retail_tariff))
    if needs_prices and _ROLLING_PRICE_PROVIDER is None:
        raise RuntimeError(
            "_ROLLING_PRICE_PROVIDER is unset in this worker process. "
            "_init_rolling_worker must run first -- either as a "
            "ProcessPoolExecutor initializer, or called directly before the "
            "per-site loop in the workers=1 (serial) path."
        )

    sc_t = self_consumption(site_id, solar_kwh, load_kwh, battery_spec,
                            interval_hours)
    sc_bill = compute_bill(sc_t, prices_mwh, network_tariff, retail_tariff,
                           state, interval_hours)

    roll_start = time.perf_counter()
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        provider = _ROLLING_PRICE_PROVIDER if needs_prices else None
        if advanced:
            _, roll_t, diagnostics = (
                run_rolling_perfect_foresight_lp_advanced_site(
                    site_id, solar_kwh, load_kwh, battery_spec,
                    network_tariff, retail_tariff, provider, rolling_config,
                    export_limit_kwh_per_interval=export_limit_kwh,
                    interval_hours=interval_hours, state=state,
                    sc_result=sc_t, settled_rrp_mwh=prices_mwh,
                    apply_export_order_nudge=apply_export_order_nudge,
                )
            )
        else:
            _, roll_t, diagnostics = run_rolling_perfect_foresight_lp_site(
                site_id, solar_kwh, load_kwh, battery_spec, offsets,
                provider, rolling_config,
                export_limit_kwh_per_interval=export_limit_kwh,
                interval_hours=interval_hours, state=state, sc_result=sc_t,
            )
    roll_dt = time.perf_counter() - roll_start

    roll_monthly_bills = None
    if with_monthly_bills:
        roll_bill, roll_monthly_bills = compute_bill_and_monthly_bills(
            roll_t, prices_mwh, network_tariff, retail_tariff,
            state, interval_hours,
        )
    else:
        roll_bill = compute_bill(roll_t, prices_mwh, network_tariff,
                                 retail_tariff, state, interval_hours)

    warning_msgs = [str(w.message) for w in caught]
    return (site_id, sc_t, sc_bill, roll_t, roll_bill, roll_dt, warning_msgs,
            roll_monthly_bills, diagnostics)


def _run_rolling_perfect_foresight_lp(
    *,
    scenario: ScenarioConfig,
    ctx,
    per_site_battery: dict[str, Optional[BatterySpec]],
    per_site_scale: dict[str, float],
    site_ids: list[str],
    t0: float,
    ingest_elapsed: float,
    progress_callback=None,
    workers: int = 1,
    with_monthly_bills: bool = False,
) -> ScenarioResult:
    """Per-site self_consumption baseline then rolling_optimisation.

    Mirrors _run_perfect_foresight's structure (baseline first for a fair
    terminal-SOC comparison, serial vs ProcessPoolExecutor branch, VCR output
    shape) with two differences:

      1. price_offsets and the price-forecast provider are scenario-level
         (built once, not per site) rather than replacing build_price_vectors
         -- see prices.build_price_stack_offsets's own docstring for why this
         is the correct chokepoint. validate_scenario has already called it
         once for its own guards; this call is independent (validate_scenario
         does not hand its result forward) and just as cheap.
      2. The provider is NOT included in each site's args tuple. See
         _init_rolling_worker's docstring for why, and how workers=1 and
         workers>1 both reach the same initialization path.

    No PF benchmark is run alongside rolling here -- per the integration
    plan, VCR for a rolling scenario is assembled downstream by joining
    against a separate perfect_foresight run's bills over the same sites,
    not computed inline. baseline_traces/baseline_bills therefore hold the
    self_consumption comparison only, same meaning as every other
    coordination type that populates them.

    Raises:
        RuntimeError: if any site's RollingDiagnostics.causality_ok is False.
            A causality violation means the result is not a legitimate
            rolling-optimisation output (see rolling_optimisation_spec.md),
            so this aborts the scenario rather than returning bills computed
            from an invalid dispatch. In the workers>1 case, futures already
            submitted before the violation is detected are allowed to finish
            (ProcessPoolExecutor gives no cheap way to cancel a running task)
            but no further site is queued and the scenario still fails.
    """
    n = len(site_ids)
    rolling_config = scenario.rolling

    per_site_export_limit = _resolve_per_site_export_limits(
        scenario, ctx.site_metadata)

    idx = ctx.prices.index
    interval_hours = (idx[1] - idx[0]).total_seconds() / 3600.0
    interval_minutes = int(round(interval_hours * 60))

    # Scenario-level, built once: both depend only on the tariffs/state/prices
    # shared by every site in the scenario, never on a site's own trace.
    advanced = (scenario.coordination_type
                == 'rolling_perfect_foresight_lp_advanced')
    # The advanced path builds no price-stack offsets: on a TOU plan the
    # customer price does not depend on RRP, and build_price_stack_offsets
    # would refuse the tariff. Its per-window prices come from the planner,
    # which each site builds for itself.
    offsets = None if advanced else build_price_stack_offsets(
        idx, ctx.network_tariff, ctx.retail_tariff, scenario.state,
        interval_hours)
    # A plan with no critical rate reads no wholesale price at all, so the
    # provider is never built and the workers need no initializer. That also
    # means a store gap cannot push such a run into self-consumption for a
    # reason unconnected to the plan.
    needs_prices = (not advanced
                    or advanced_plan_needs_price_forecast(ctx.retail_tariff))
    infill_lag_intervals = int(round(
        rolling_config.infill_lag_hours * 60 / interval_minutes))
    # This conversion is duplicated (not called via
    # build_rolling_price_provider) because _init_rolling_worker's signature
    # must stay a flat, picklable primitives tuple for
    # ProcessPoolExecutor's initargs -- it cannot take a RollingConfig. If
    # this arithmetic ever changes, change it in build_rolling_price_provider
    # too; the two must agree.
    region = region_for_state(scenario.state)
    init_args = (
        rolling_config.forecast_source, ctx.prices, interval_minutes,
        rolling_config.current_interval_price, infill_lag_intervals,
        DEFAULT_PREDISPATCH_STORE, region,
        rolling_config.on_degraded_period, rolling_config.max_staleness_hours,
    )

    site_args = []
    for site_id in site_ids:
        battery_spec = per_site_battery[site_id]
        site_trace = ctx.site_traces[site_id]
        solar_kwh = site_trace['pv_site_net_kwh'] * per_site_scale[site_id]
        load_kwh = site_trace['gross_load_kwh']
        soc_min = (battery_spec.backup_reserve_pct / 100.0) * battery_spec.capacity_kwh
        export_limit_kwh = per_site_export_limit[site_id] * interval_hours
        site_args.append((
            site_id, solar_kwh, load_kwh, battery_spec, offsets,
            ctx.prices, ctx.network_tariff, ctx.retail_tariff,
            soc_min, export_limit_kwh, interval_hours, scenario.state,
            rolling_config, with_monthly_bills, scenario.coordination_type,
            scenario.apply_export_order_nudge,
        ))

    traces: dict[str, SiteTraces] = {}
    bills: dict[str, SiteBill] = {}
    baseline_traces: dict[str, SiteTraces] = {}
    baseline_bills: dict[str, SiteBill] = {}
    diagnostics_by_site: dict[str, RollingDiagnostics] = {}
    monthly_bills: Optional[dict[str, dict[int, SiteBill]]] = (
        {} if with_monthly_bills else None
    )
    roll_seconds = 0.0

    def _check_causality(sid: str, diag: RollingDiagnostics) -> None:
        if not diag.causality_ok:
            raise RuntimeError(
                f"{scenario.scenario_id} site {sid}: RollingDiagnostics."
                f"causality_ok is False (max run-time lead "
                f"{diag.max_run_time_lead_minutes:.1f} min). Aborting the "
                "scenario -- see rolling_optimisation_spec.md."
            )

    if workers == 1:
        if needs_prices:
            _init_rolling_worker(*init_args)
        for i, (site_id, args) in enumerate(zip(site_ids, site_args), start=1):
            (sid, sc_t, sc_bill, roll_t, roll_bill, roll_dt, warning_msgs,
             roll_monthly, diag) = _rolling_perfect_foresight_site_worker(args)
            _check_causality(sid, diag)
            for msg in warning_msgs:
                warnings.warn(msg, stacklevel=2)
            traces[sid] = roll_t
            bills[sid] = roll_bill
            baseline_traces[sid] = sc_t
            baseline_bills[sid] = sc_bill
            diagnostics_by_site[sid] = diag
            if with_monthly_bills:
                monthly_bills[sid] = roll_monthly
            roll_seconds += roll_dt
            if progress_callback is not None:
                progress_callback(i, n, sid, note=f"rolling {roll_dt:.1f}s")
    else:
        import multiprocessing as _mp
        completed = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_mp.get_context('spawn'),
            initializer=_init_rolling_worker if needs_prices else None,
            initargs=init_args if needs_prices else (),
        ) as executor:
            future_to_site = {
                executor.submit(_rolling_perfect_foresight_site_worker, args): site_id
                for site_id, args in zip(site_ids, site_args)
            }
            for future in as_completed(future_to_site):
                (sid, sc_t, sc_bill, roll_t, roll_bill, roll_dt, warning_msgs,
                 roll_monthly, diag) = future.result()
                _check_causality(sid, diag)
                for msg in warning_msgs:
                    warnings.warn(msg, stacklevel=2)
                traces[sid] = roll_t
                bills[sid] = roll_bill
                baseline_traces[sid] = sc_t
                baseline_bills[sid] = sc_bill
                diagnostics_by_site[sid] = diag
                if with_monthly_bills:
                    monthly_bills[sid] = roll_monthly
                roll_seconds += roll_dt
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, n, sid,
                                      note=f"rolling {roll_dt:.1f}s")

    elapsed = time.perf_counter() - t0
    return ScenarioResult(
        scenario=scenario, site_traces=traces, site_bills=bills,
        elapsed_seconds=elapsed, ingest_seconds=ingest_elapsed,
        lp_seconds=roll_seconds, n_workers=workers,
        baseline_traces=baseline_traces, baseline_bills=baseline_bills,
        site_monthly_bills=monthly_bills,
        rolling_diagnostics=diagnostics_by_site,
    )



# ---------------------------------------------------------------------------
# Rolling VPP coordination types (Part C)
# ---------------------------------------------------------------------------

def run_rolling_vpp_event_site(
    site_id,
    solar_kwh,
    load_kwh,
    prices_mwh,
    battery_spec,
    op_price_offsets,
    forecast_provider,
    rolling_config,
    *,
    vpp_threshold_dollars_per_mwh,
    event_window_hours,
    vpp_exp_reward,
    vpp_imp_reward,
    vpp_kwh_limit,
    vpp_limit_applies_to,
    min_export_price_dollars_per_mwh,
    gap_strategy,
    export_limit_kwh_per_interval=float('inf'),
    interval_hours=5 / 60,
    p_curtail=None,
    no_grid_charge_mask=None,
    retail_tariff=None,
    network_tariff=None,
    state=None,
    sc_result=None,
    apply_export_order_nudge=False,
):
    """One site through rolling_vpp_event. Shared by the bulk worker and viz.

    Public for the same reason run_rolling_perfect_foresight_lp_site is: the
    single-site tool and the bulk run must execute literally the same code, or
    a difference between them is a difference nobody is looking for.

    Returns (sc_result, SiteTraces, VppTraces, RollingDiagnostics). The
    baseline is computed here when not supplied so callers that already have
    it do not pay for it twice.

    apply_export_order_nudge is passed to the household AdvancedWindowPlanner
    on gap_strategy='tou_advanced' (ScenarioConfig.apply_export_order_nudge).
    """
    if sc_result is None:
        sc_result = self_consumption(site_id, solar_kwh, load_kwh,
                                     battery_spec, interval_hours)

    # Built exactly as run_rolling_perfect_foresight_lp_site builds them, from
    # the site's OWN load/solar trace with no valid= mask. Omitting this is
    # silent: rolling_vpp_event defaults both providers to None and
    # commit_actual_load to True, so the run completes with perfect site
    # knowledge, every invariant holds, and only load_forecast_used /
    # solar_forecast_used in the diagnostics show it.
    n_h = int(round(rolling_config.horizon_hours / interval_hours))
    if state is None and (rolling_config.usage_forecast is not None
                          or rolling_config.solar_forecast is not None):
        raise ValueError(
            f"[{site_id}] a usage_forecast or solar_forecast is configured "
            f"but state is None. Both providers resolve local days from it, "
            f"so it is required here even on gap_strategy='self_consumption', "
            f"where nothing else reads it."
        )
    usage_provider = None
    if rolling_config.usage_forecast is not None:
        usage_provider = UsageForecastProvider.from_series(
            load_kwh, state, rolling_config.usage_forecast,
            max_horizon_intervals=n_h, interval_hours=interval_hours)
    solar_provider = None
    if rolling_config.solar_forecast is not None:
        solar_provider = SolarForecastProvider.from_series(
            solar_kwh, state, rolling_config.solar_forecast,
            max_horizon_intervals=n_h, interval_hours=interval_hours)

    traces, vpp_traces, diagnostics = rolling_vpp_event(
        site_id=site_id,
        solar_kwh=solar_kwh,
        load_kwh=load_kwh,
        prices_mwh=prices_mwh,
        op_price_offsets=op_price_offsets,
        forecast_provider=forecast_provider,
        battery_spec=battery_spec,
        soc_init_kwh=(battery_spec.backup_reserve_pct / 100.0)
        * battery_spec.capacity_kwh,
        soc_final_kwh=(battery_spec.backup_reserve_pct / 100.0)
        * battery_spec.capacity_kwh,
        config=rolling_config,
        vpp_threshold_dollars_per_mwh=vpp_threshold_dollars_per_mwh,
        event_window_hours=event_window_hours,
        vpp_exp_reward=vpp_exp_reward,
        vpp_imp_reward=vpp_imp_reward,
        vpp_kwh_limit=vpp_kwh_limit,
        vpp_limit_applies_to=vpp_limit_applies_to,
        min_export_price_dollars_per_mwh=min_export_price_dollars_per_mwh,
        gap_strategy=gap_strategy,
        export_limit_kwh_per_interval=export_limit_kwh_per_interval,
        interval_hours=interval_hours,
        p_curtail=p_curtail,
        no_grid_charge_mask=no_grid_charge_mask,
        retail_tariff=retail_tariff,
        network_tariff=network_tariff,
        state=state,
        load_forecast_provider=usage_provider,
        solar_forecast_provider=solar_provider,
        commit_actual_load=rolling_config.commit_actual_load,
        apply_export_order_nudge=apply_export_order_nudge,
    )
    return sc_result, traces, vpp_traces, diagnostics


# Field order of the rolling VPP worker's args tuple. Defined once and used by
# both the builder and the unpacker below, so the two cannot drift: arity alone
# would not catch two swapped fields, which keeps the length identical, unpacks
# silently, and sends the wrong value to the wrong parameter.
_ROLLING_VPP_ARG_FIELDS = (
    'site_id', 'solar_kwh', 'load_kwh', 'battery_spec', 'op_offsets',
    'prices_mwh', 'network_tariff', 'retail_tariff', 'soc_min',
    'export_limit_kwh', 'interval_hours', 'state', 'rolling_config',
    'with_monthly_bills', 'coordination_type',
    'vpp_threshold_dollars_per_mwh', 'event_window_hours',
    'vpp_exp_reward', 'vpp_imp_reward', 'vpp_kwh_limit',
    'vpp_limit_applies_to', 'min_export_price_dollars_per_mwh',
    'p_curtail', 'no_grid_charge_mask', 'apply_export_order_nudge',
)


def _rolling_vpp_site_worker(args: tuple) -> tuple:
    """Module-level worker for ProcessPoolExecutor -- must be picklable.

    Separate from _rolling_perfect_foresight_site_worker rather than a branch
    inside it: this path carries nine more fields that the non-VPP types never
    read, and returns VppTraces, which reaches compute_bill and the scenario's
    vpp_traces. Folding the two together would make both harder to follow
    across a pickle boundary where a wrong field is expensive to trace.

    Args:
        args: a tuple in _ROLLING_VPP_ARG_FIELDS order.

            coordination_type selects gap_strategy. An explicit field rather
            than an inference, for the same reason the perfect-foresight
            worker takes one.

    Returns:
        (site_id, sc_t, sc_bill, roll_t, roll_vt, roll_bill, roll_dt,
         warning_msgs, roll_monthly_bills, diagnostics)
        roll_monthly_bills is None unless with_monthly_bills was True.
        The caller checks diagnostics.causality_ok -- a violation invalidates
        the run, not just this site.
    """
    import warnings as _warnings
    (site_id, solar_kwh, load_kwh, battery_spec, op_offsets,
     prices_mwh, network_tariff, retail_tariff, soc_min,
     export_limit_kwh, interval_hours, state, rolling_config,
     with_monthly_bills, coordination_type,
     vpp_threshold_dollars_per_mwh, event_window_hours,
     vpp_exp_reward, vpp_imp_reward, vpp_kwh_limit,
     vpp_limit_applies_to, min_export_price_dollars_per_mwh,
     p_curtail, no_grid_charge_mask, apply_export_order_nudge) = args

    if _ROLLING_PRICE_PROVIDER is None:
        raise RuntimeError(
            "_ROLLING_PRICE_PROVIDER is unset in this worker process. "
            "_init_rolling_worker must run first -- either as a "
            "ProcessPoolExecutor initializer, or called directly before the "
            "per-site loop in the workers=1 (serial) path. The rolling VPP "
            "types always need it: the operator declares events from the "
            "price forecast even when the household plan never reads prices."
        )

    gap_strategy = ('tou_advanced'
                    if coordination_type == 'rolling_vpp_tou_event'
                    else 'self_consumption')

    sc_t = self_consumption(site_id, solar_kwh, load_kwh, battery_spec,
                            interval_hours)
    sc_bill = compute_bill(sc_t, prices_mwh, network_tariff, retail_tariff,
                           state, interval_hours)

    roll_start = time.perf_counter()
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        _, roll_t, roll_vt, diagnostics = run_rolling_vpp_event_site(
            site_id, solar_kwh, load_kwh, prices_mwh, battery_spec,
            op_offsets, _ROLLING_PRICE_PROVIDER, rolling_config,
            vpp_threshold_dollars_per_mwh=vpp_threshold_dollars_per_mwh,
            event_window_hours=event_window_hours,
            vpp_exp_reward=vpp_exp_reward,
            vpp_imp_reward=vpp_imp_reward,
            vpp_kwh_limit=vpp_kwh_limit,
            vpp_limit_applies_to=vpp_limit_applies_to,
            min_export_price_dollars_per_mwh=min_export_price_dollars_per_mwh,
            gap_strategy=gap_strategy,
            export_limit_kwh_per_interval=export_limit_kwh,
            interval_hours=interval_hours, p_curtail=p_curtail,
            no_grid_charge_mask=no_grid_charge_mask,
            retail_tariff=retail_tariff, network_tariff=network_tariff,
            state=state, sc_result=sc_t,
            apply_export_order_nudge=apply_export_order_nudge,
        )
    roll_dt = time.perf_counter() - roll_start

    roll_monthly_bills = None
    if with_monthly_bills:
        roll_bill, roll_monthly_bills = compute_bill_and_monthly_bills(
            roll_t, prices_mwh, network_tariff, retail_tariff,
            state, interval_hours, vpp_traces=roll_vt,
        )
    else:
        roll_bill = compute_bill(roll_t, prices_mwh, network_tariff,
                                 retail_tariff, state, interval_hours,
                                 vpp_traces=roll_vt)

    warning_msgs = [str(w.message) for w in caught]
    return (site_id, sc_t, sc_bill, roll_t, roll_vt, roll_bill, roll_dt,
            warning_msgs, roll_monthly_bills, diagnostics)


def _run_rolling_vpp_event(
    *,
    scenario: ScenarioConfig,
    ctx,
    paths: Paths,
    per_site_battery: dict[str, Optional[BatterySpec]],
    per_site_scale: dict[str, float],
    site_ids: list[str],
    t0: float,
    ingest_elapsed: float,
    progress_callback=None,
    workers: int = 1,
    with_monthly_bills: bool = False,
) -> ScenarioResult:
    """Per-site self_consumption baseline then rolling_vpp_event.

    One runner for both rolling VPP coordination types; the worker selects
    gap_strategy from scenario.coordination_type.

    Structure follows _run_rolling_perfect_foresight_lp -- scenario-level
    offsets and provider built once, the provider reaching workers through
    _init_rolling_worker rather than the args tuple, causality aborting the
    scenario -- with the VPP additions from _run_vpp_sc_event: operator price
    offsets, reward conversion from cents, the no-grid-charge mask, and
    VppTraces carried through to billing and into the ScenarioResult.

    The price forecast provider is ALWAYS built here, unlike the advanced
    perfect-foresight path which skips it for a plan with no critical rates. The
    operator declares events from the price forecast whatever the household
    plan reads, so a rolling VPP run without one is not a rolling VPP run.

    Raises:
        RuntimeError: if any site's RollingDiagnostics.causality_ok is False,
            for the reason given in _run_rolling_perfect_foresight_lp.
    """
    n = len(site_ids)
    rolling_config = scenario.rolling
    vpp_cfg = ctx.retail_tariff.vpp

    per_site_export_limit = _resolve_per_site_export_limits(
        scenario, ctx.site_metadata)

    idx = ctx.prices.index
    interval_hours = (idx[1] - idx[0]).total_seconds() / 3600.0
    interval_minutes = int(round(interval_hours * 60))

    # Operator price stack, scenario-level: depends on the operator tariff and
    # the state, never on a site's own trace. The household side needs no
    # offsets -- on the SC variant there is no household objective, and on the
    # TOU variant the planner builds its own vectors.
    op_tariff = load_retail_tariff(scenario.op_retail_tariff_id,
                                   paths.tariffs_dir)
    op_offsets = build_price_stack_offsets(
        idx, ctx.network_tariff, op_tariff, scenario.state, interval_hours)

    _C = 100.0
    vpp_exp_reward = vpp_cfg.exp_reward_c_per_kwh / _C
    vpp_imp_reward = vpp_cfg.imp_reward_c_per_kwh / _C
    p_curtail = (vpp_cfg.curtail_c_per_kwh / _C
                 if vpp_cfg.curtail_c_per_kwh is not None else None)

    # Fleet-wide scalar, resolved once here so every site's planner reads the
    # same number. $/MWh: VppWindowPlanner applies its own /1000 against the
    # operator export price, so no conversion belongs here.
    min_export_resolution = resolve_min_export_price(
        vpp_cfg=vpp_cfg,
        prices_mwh=ctx.prices,
        site_traces=ctx.site_traces,
        site_ids=site_ids,
        power_kw=scenario.battery_spec.power_kw,
        interval_hours=interval_hours,
    )
    min_export_price_dollars_per_mwh = (
        min_export_resolution.value_dollars_per_mwh)

    no_grid_charge_mask = None
    if vpp_cfg.no_grid_charge_windows:
        import numpy as _np
        no_grid_charge_mask = _np.zeros(len(idx), dtype=bool)
        for window in vpp_cfg.no_grid_charge_windows:
            no_grid_charge_mask |= charge_window_mask(
                idx, window, scenario.state, interval_hours)

    infill_lag_intervals = int(round(
        rolling_config.infill_lag_hours * 60 / interval_minutes))
    region = region_for_state(scenario.state)
    init_args = (
        rolling_config.forecast_source, ctx.prices, interval_minutes,
        rolling_config.current_interval_price, infill_lag_intervals,
        DEFAULT_PREDISPATCH_STORE, region,
        rolling_config.on_degraded_period, rolling_config.max_staleness_hours,
    )

    site_args = []
    for site_id in site_ids:
        battery_spec = per_site_battery[site_id]
        site_trace = ctx.site_traces[site_id]
        solar_kwh = site_trace['pv_site_net_kwh'] * per_site_scale[site_id]
        load_kwh = site_trace['gross_load_kwh']
        soc_min = ((battery_spec.backup_reserve_pct / 100.0)
                   * battery_spec.capacity_kwh)
        export_limit_kwh = per_site_export_limit[site_id] * interval_hours
        # Field order must match _ROLLING_VPP_ARG_FIELDS.
        site_args.append((
            site_id, solar_kwh, load_kwh, battery_spec, op_offsets,
            ctx.prices, ctx.network_tariff, ctx.retail_tariff, soc_min,
            export_limit_kwh, interval_hours, scenario.state, rolling_config,
            with_monthly_bills, scenario.coordination_type,
            vpp_cfg.vpp_threshold_dollars_per_mwh, vpp_cfg.event_window_hours,
            vpp_exp_reward, vpp_imp_reward, vpp_cfg.kwh_limit,
            vpp_cfg.limit_applies_to, min_export_price_dollars_per_mwh,
            p_curtail, no_grid_charge_mask, scenario.apply_export_order_nudge,
        ))

    traces: dict[str, SiteTraces] = {}
    vpp_traces_all: dict[str, VppTraces] = {}
    bills: dict[str, SiteBill] = {}
    baseline_traces: dict[str, SiteTraces] = {}
    baseline_bills: dict[str, SiteBill] = {}
    diagnostics_by_site: dict[str, RollingDiagnostics] = {}
    monthly_bills: Optional[dict[str, dict[int, SiteBill]]] = (
        {} if with_monthly_bills else None
    )
    roll_seconds = 0.0

    def _check_causality(sid: str, diag: RollingDiagnostics) -> None:
        if not diag.causality_ok:
            raise RuntimeError(
                f"{scenario.scenario_id} site {sid}: RollingDiagnostics."
                f"causality_ok is False (max run-time lead "
                f"{diag.max_run_time_lead_minutes:.1f} min). Aborting the "
                "scenario -- see rolling_optimisation_spec.md."
            )

    def _collect(result) -> tuple[str, float]:
        (sid, sc_t, sc_bill, roll_t, roll_vt, roll_bill, roll_dt,
         warning_msgs, roll_monthly, diag) = result
        _check_causality(sid, diag)
        for msg in warning_msgs:
            warnings.warn(msg, stacklevel=2)
        traces[sid] = roll_t
        vpp_traces_all[sid] = roll_vt
        bills[sid] = roll_bill
        baseline_traces[sid] = sc_t
        baseline_bills[sid] = sc_bill
        diagnostics_by_site[sid] = diag
        if with_monthly_bills:
            monthly_bills[sid] = roll_monthly
        return sid, roll_dt

    if workers == 1:
        _init_rolling_worker(*init_args)
        for i, args in enumerate(site_args, start=1):
            sid, roll_dt = _collect(_rolling_vpp_site_worker(args))
            roll_seconds += roll_dt
            if progress_callback is not None:
                progress_callback(i, n, sid, note=f"rolling {roll_dt:.1f}s")
    else:
        import multiprocessing as _mp
        completed = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_mp.get_context('spawn'),
            initializer=_init_rolling_worker,
            initargs=init_args,
        ) as executor:
            futures = [executor.submit(_rolling_vpp_site_worker, a)
                       for a in site_args]
            for future in as_completed(futures):
                sid, roll_dt = _collect(future.result())
                roll_seconds += roll_dt
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, n, sid,
                                      note=f"rolling {roll_dt:.1f}s")

    elapsed = time.perf_counter() - t0
    return ScenarioResult(
        scenario=scenario, site_traces=traces, site_bills=bills,
        elapsed_seconds=elapsed, ingest_seconds=ingest_elapsed,
        lp_seconds=roll_seconds, n_workers=workers,
        baseline_traces=baseline_traces, baseline_bills=baseline_bills,
        site_monthly_bills=monthly_bills,
        rolling_diagnostics=diagnostics_by_site,
        vpp_traces=vpp_traces_all,
        min_export_price_resolution=min_export_resolution,
    )

def _vpp_sc_event_site_worker(args: tuple) -> tuple:
    """Module-level worker for ProcessPoolExecutor -- must be picklable.

    Runs self_consumption for the SC baseline bill, then classify_vpp_events,
    then vpp_sc_event for a single site.

    Args:
        args: (site_id, solar_kwh, load_kwh,
               op_p_imp, op_p_exp,
               battery_spec, prices_mwh, network_tariff, retail_tariff,
               soc_min, export_limit_kwh, interval_hours,
               vpp_exp_reward, vpp_imp_reward, vpp_kwh_limit,
               vpp_limit_applies_to, threshold_mwh, event_window_hours,
               p_curtail, gap_strategy, hh_p_imp, hh_p_exp,
               min_export_price_per_kwh, no_grid_charge_mask,
               apply_export_order_nudge, state, with_monthly_bills)

    Returns:
        (site_id, sc_t, sc_bill, eov_t, eov_vt, eov_bill, eov_dt,
         warning_msgs, eov_monthly_bills)
        eov_monthly_bills is None unless with_monthly_bills was True.
    """
    import warnings as _warnings
    (site_id, solar_kwh, load_kwh,
     op_p_imp, op_p_exp,
     battery_spec, prices_mwh, network_tariff, retail_tariff,
     soc_min, export_limit_kwh, interval_hours,
     vpp_exp_reward, vpp_imp_reward, vpp_kwh_limit,
     vpp_limit_applies_to, threshold_mwh, event_window_hours,
     p_curtail, gap_strategy, hh_p_imp, hh_p_exp,
     min_export_price_per_kwh, no_grid_charge_mask,
     apply_export_order_nudge, state, with_monthly_bills) = args

    # Self-consumption baseline for the VCR delta bill comparison.
    sc_t = self_consumption(site_id, solar_kwh, load_kwh, battery_spec,
                            interval_hours)
    sc_bill = compute_bill(sc_t, prices_mwh, network_tariff, retail_tariff,
                           state, interval_hours)

    # Event classification.
    vpp_event = classify_vpp_events(
        prices_mwh,
        threshold_dollars_per_mwh=threshold_mwh,
        event_window_hours=event_window_hours,
        interval_hours=interval_hours,
    )

    eov_start = time.perf_counter()
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        eov_t, eov_vt = vpp_sc_event(
            site_id, solar_kwh, load_kwh,
            op_p_imp, op_p_exp,
            vpp_event=vpp_event,
            battery_spec=battery_spec,
            soc_init_kwh=soc_min,
            vpp_exp_reward=vpp_exp_reward,
            vpp_imp_reward=vpp_imp_reward,
            vpp_kwh_limit=vpp_kwh_limit,
            vpp_limit_applies_to=vpp_limit_applies_to,
            export_limit_kwh_per_interval=export_limit_kwh,
            interval_hours=interval_hours,
            p_curtail=p_curtail,
            gap_strategy=gap_strategy,
            retail_tariff=retail_tariff if gap_strategy == 'tou_advanced' else None,
            prices_mwh=prices_mwh      if gap_strategy == 'tou_advanced' else None,
            price_imp=hh_p_imp         if gap_strategy == 'tou_advanced' else None,
            price_exp=hh_p_exp         if gap_strategy == 'tou_advanced' else None,
            min_export_price_per_kwh=min_export_price_per_kwh,
            no_grid_charge_mask=no_grid_charge_mask,
            apply_export_order_nudge=apply_export_order_nudge,
            state=state,
        )
    eov_dt = time.perf_counter() - eov_start

    eov_monthly_bills = None
    if with_monthly_bills:
        eov_bill, eov_monthly_bills = compute_bill_and_monthly_bills(
            eov_t, prices_mwh, network_tariff, retail_tariff,
            state, interval_hours, vpp_traces=eov_vt,
        )
    else:
        eov_bill = compute_bill(eov_t, prices_mwh, network_tariff, retail_tariff,
                                state, interval_hours, vpp_traces=eov_vt)

    warning_msgs = [str(w.message) for w in caught]
    return (site_id, sc_t, sc_bill, eov_t, eov_vt, eov_bill, eov_dt,
            warning_msgs, eov_monthly_bills)


@dataclass(frozen=True)
class MinExportPriceEstimate:
    """A minimum export price estimate with the context needed to read it.

    The price alone is ambiguous at zero: the algorithm returns zero both when
    the mask selected no intervals and when it selected intervals whose total
    throughput fell short of kwh_limit. Those are different facts about a
    scenario -- no events at all, against events too few or too load-covered to
    spend the budget -- and only the second says anything about the cap.

    Attributes:
        price_dollars_per_mwh: The estimate. Zero when not exhausted.
        n_mask_intervals: Intervals the event mask selected, or the whole
            period when no mask was supplied.
        available_kwh: Total effective export throughput across those
            intervals, at full power after cohort-average load.
        exhausted: Whether the cumulative contribution reached kwh_limit. When
            False the price is zero by definition rather than by arithmetic.
    """
    price_dollars_per_mwh: float
    n_mask_intervals: int
    available_kwh: float
    exhausted: bool


def _compute_min_export_price(
    prices_mwh: pd.Series,
    site_traces: dict,
    site_ids: list[str],
    power_kw: float,
    interval_hours: float,
    kwh_limit: float,
    vpp_event: 'np.ndarray | None' = None,
    *,
    return_details: bool = False,
) -> 'float | MinExportPriceEstimate':
    """Compute the marginal wholesale price ($/MWh) that exhausts the annual
    VPP export budget if the battery exports at full effective power at every
    higher-priced interval.

    The effective export rate at each interval accounts for average household
    load competing with battery discharge:
        effective_export_kw[t] = max(0, power_kw - avg_load_kw[t])

    where avg_load_kw[t] is the mean across all sites in the cohort at
    interval t. This gives a single conservative threshold that is
    representative across the cohort.

    When vpp_event is provided, the budget allocation is restricted to
    event-window intervals only. This is important because the LP can only
    dispatch forced export inside event windows -- counting all-year intervals
    overstates the available throughput and produces a threshold that is too
    high, leaving cap unused.

    Returns 0.0 if the budget can never be exhausted (sum of all
    contributions < kwh_limit), meaning no restriction is needed.

    Args:
        return_details: When True, return a MinExportPriceEstimate instead of
            a bare float. Keyword-only and defaulting False so the existing
            perfect-foresight call sites are unaffected.
    """
    import numpy as np
    prices = prices_mwh.to_numpy()
    T = len(prices)

    # Cohort-average load at each interval (kW), capped at power_kw per site.
    # Load above power_kw contributes zero to effective export (the battery
    # can't export when load already exceeds its full discharge power), so
    # including uncapped load would understate available throughput.
    load_sum = np.zeros(T)
    n_sites = 0
    for sid in site_ids:
        if sid in site_traces:
            load_kwh = site_traces[sid]['gross_load_kwh']
            if hasattr(load_kwh, 'to_numpy'):
                load_kwh = load_kwh.to_numpy()
            load_kw = load_kwh / interval_hours
            load_sum += np.minimum(load_kw, power_kw)
            n_sites += 1
    avg_load_kw = (load_sum / n_sites) if n_sites > 0 else np.zeros(T)

    # Restrict to event-window intervals if provided.
    # The LP can only dispatch forced export inside event windows, so the
    # budget allocation should only count those intervals.
    if vpp_event is not None and vpp_event.any():
        mask = vpp_event.astype(bool)
        prices      = prices[mask]
        avg_load_kw = avg_load_kw[mask]

    def _result(price: float, available_kwh: float,
                exhausted: bool) -> 'float | MinExportPriceEstimate':
        if not return_details:
            return price
        return MinExportPriceEstimate(
            price_dollars_per_mwh=price,
            n_mask_intervals=int(len(prices)),
            available_kwh=float(available_kwh),
            exhausted=exhausted,
        )

    if len(prices) == 0:
        return _result(0.0, 0.0, False)

    # Effective export kWh per interval after serving average load
    effective_export_kw = np.maximum(0.0, power_kw - avg_load_kw)
    contribution = effective_export_kw * interval_hours  # kWh per interval

    # Sort by descending price and accumulate contributions
    order = np.argsort(-prices)
    cumulative = np.cumsum(contribution[order])
    available_kwh = float(cumulative[-1])

    # Find the interval where the budget is exhausted
    exhausted_idx = np.searchsorted(cumulative, kwh_limit)
    if exhausted_idx >= len(order):
        # Budget cannot be exhausted -- no restriction needed
        return _result(0.0, available_kwh, False)

    marginal_price = float(prices[order[exhausted_idx]])
    return _result(marginal_price, available_kwh, True)


@dataclass(frozen=True)
class MinExportPriceResolution:
    """How a rolling VPP run arrived at its minimum export price.

    Recorded on ScenarioResult and written to the scenario summary. Without it
    a run whose derived estimate collapsed to the configured floor is
    indistinguishable from one configured 'fixed', and a run whose threshold
    was derived from the modelled year's own prices is indistinguishable from
    a fully causal one after the fact.

    Attributes:
        mode: The VppConfig.min_export_price_mode that produced the value.
        value_dollars_per_mwh: The operative threshold handed to every site's
            planner. $/MWh; the planner applies its own /1000.
        configured_dollars_per_mwh: The value stated in the tariff, or None.
            Under a derived mode this is a floor, not the value.
        derived_dollars_per_mwh: The estimate, or None under 'fixed'.
        source: Which of the two the floor rule selected -- 'configured',
            'derived', or 'equal' when they tie.
        n_mask_intervals: Event-mask intervals the estimate ran over, or None
            under 'fixed'.
        available_kwh: Effective export throughput across those intervals, or
            None under 'fixed'.
        budget_exhausted: Whether that throughput reached kwh_limit. False
            with a zero derived estimate means the cap could not be spent
            within the available event intervals, which is a different finding
            from a low marginal price.
    """
    mode: str
    value_dollars_per_mwh: float
    configured_dollars_per_mwh: Optional[float]
    derived_dollars_per_mwh: Optional[float] = None
    source: str = 'configured'
    n_mask_intervals: Optional[int] = None
    available_kwh: Optional[float] = None
    budget_exhausted: Optional[bool] = None


def resolve_min_export_price(
    vpp_cfg,
    prices_mwh: pd.Series,
    site_traces: dict,
    site_ids: list[str],
    power_kw: float,
    interval_hours: float,
) -> MinExportPriceResolution:
    """Resolve the fleet-wide minimum export price for a rolling VPP scenario.

    Called once per scenario, before the worker pool spawns, so every site's
    planner reads the same number. Kept separate from the runner so it can be
    exercised on plain Series and a traces dict without a scenario fixture.

    Under a derived mode the operative value is max(derived, configured or 0):
    the configured value is the operator's own price floor and the derivation
    can only tighten it. That also covers the case where the derivation returns
    zero because the available throughput could not exhaust kwh_limit, which
    without a floor would remove the restriction entirely at exactly the
    configurations where event intervals are scarcest.

    Returns $/MWh. The rolling planner takes $/MWh and applies its own /1000
    against the operator export price, so the value passes through unconverted.
    The perfect-foresight runners divide by 1000 because vpp_sc_event takes
    $/kWh; that conversion has no place on this path.

    Raises:
        ValueError: on 'based_on_last_year', which is not implemented, and on
            an absent configured value under 'fixed'. validate_scenario catches
            both before any ingest; this is the backstop for a caller that
            bypassed it.

    See rolling_vpp_event_spec.md, "Minimum export price".
    """
    mode = vpp_cfg.min_export_price_mode
    configured = vpp_cfg.min_export_price_dollars_per_mwh

    if mode == 'based_on_last_year':
        raise ValueError(
            "min_export_price_mode='based_on_last_year' is not implemented. "
            "Use 'fixed' or 'based_on_this_year'."
        )

    if mode == 'fixed':
        if configured is None:
            raise ValueError(
                "min_export_price_mode='fixed' requires "
                "min_export_price_dollars_per_mwh to be set."
            )
        return MinExportPriceResolution(
            mode=mode,
            value_dollars_per_mwh=float(configured),
            configured_dollars_per_mwh=float(configured),
            source='configured',
        )

    if mode != 'based_on_this_year':
        raise ValueError(
            f"unrecognised min_export_price_mode {mode!r}")

    # The mask comes from classify_vpp_events rather than from the rolling
    # component's own forward declaration rule, so a rolling run and its
    # perfect-foresight benchmark resolve to the same number for the same
    # scenario. Deriving them differently would put a threshold difference
    # inside every value-capture ratio quoted between the two.
    vpp_event = classify_vpp_events(
        prices_mwh,
        threshold_dollars_per_mwh=vpp_cfg.vpp_threshold_dollars_per_mwh,
        event_window_hours=vpp_cfg.event_window_hours,
        interval_hours=interval_hours,
    )

    floor = 0.0 if configured is None else float(configured)

    # An all-False mask is handled here rather than passed down.
    # _compute_min_export_price reads `vpp_event is not None and
    # vpp_event.any()`, so an empty mask falls through to the whole period and
    # derives a threshold from all-year prices -- the exact overstatement the
    # mask exists to prevent, and the resolution record would report the
    # period length as the mask size. Both the rolling and perfect-foresight
    # VPP runners resolve through this function, so neither reaches that
    # fall-through; the estimator itself is left unchanged.
    if not vpp_event.any():
        return MinExportPriceResolution(
            mode=mode,
            value_dollars_per_mwh=floor,
            configured_dollars_per_mwh=(None if configured is None
                                        else float(configured)),
            derived_dollars_per_mwh=0.0,
            source='configured' if floor > 0.0 else 'equal',
            n_mask_intervals=0,
            available_kwh=0.0,
            budget_exhausted=False,
        )

    estimate = _compute_min_export_price(
        prices_mwh=prices_mwh,
        site_traces=site_traces,
        site_ids=site_ids,
        power_kw=power_kw,
        interval_hours=interval_hours,
        kwh_limit=vpp_cfg.kwh_limit,
        vpp_event=vpp_event,
        return_details=True,
    )

    derived = estimate.price_dollars_per_mwh
    if derived > floor:
        source = 'derived'
    elif derived < floor:
        source = 'configured'
    else:
        source = 'equal'

    return MinExportPriceResolution(
        mode=mode,
        value_dollars_per_mwh=max(derived, floor),
        configured_dollars_per_mwh=(None if configured is None
                                    else float(configured)),
        derived_dollars_per_mwh=derived,
        source=source,
        n_mask_intervals=estimate.n_mask_intervals,
        available_kwh=estimate.available_kwh,
        budget_exhausted=estimate.exhausted,
    )


def _run_vpp_sc_event(
    *,
    scenario: ScenarioConfig,
    ctx,
    paths: Paths,
    per_site_battery: dict[str, Optional[BatterySpec]],
    per_site_scale: dict[str, float],
    site_ids: list[str],
    t0: float,
    ingest_elapsed: float,
    progress_callback=None,
    workers: int = 1,
    gap_strategy: str = 'self_consumption',
    with_monthly_bills: bool = False,
) -> ScenarioResult:
    """Per-site SC baseline then vpp_sc_event dispatch.

    Design notes (this replaced the earlier perfect_foresight_vpp single-LP
    approach, removed once vpp_sc_event's per-event formulation superseded it):
      - No household price vectors for the event LP: vpp_sc_event runs SC
        internally for event-window bound construction.
      - No soc_final: the component has no terminal SOC constraint.
      - No export preflight: SC handles any infeasible intervals naturally.

    When gap_strategy='tou_advanced', household TOU price vectors are built
    and passed to the component for between-event TOU gap optimisation.
    """
    n = len(site_ids)

    per_site_export_limit = _resolve_per_site_export_limits(
        scenario, ctx.site_metadata)

    idx = ctx.prices.index
    interval_hours = (idx[1] - idx[0]).total_seconds() / 3600.0

    # Operator price vectors only (always needed).
    op_tariff = load_retail_tariff(scenario.op_retail_tariff_id, paths.tariffs_dir)
    op_p_imp, op_p_exp = build_price_vectors(
        idx, ctx.prices, ctx.network_tariff, op_tariff,
        scenario.state, interval_hours)

    # Household TOU price vectors (only for tou_advanced gap strategy).
    hh_p_imp = hh_p_exp = None
    if gap_strategy == 'tou_advanced':
        from model.core.prices import build_lp_price_vectors_advanced
        hh_p_imp, hh_p_exp = build_lp_price_vectors_advanced(
            idx, ctx.prices, ctx.network_tariff, ctx.retail_tariff,
            scenario.state, interval_hours)

    vpp_cfg = ctx.retail_tariff.vpp
    _C = 100.0
    vpp_exp_reward = vpp_cfg.exp_reward_c_per_kwh / _C
    vpp_imp_reward = vpp_cfg.imp_reward_c_per_kwh / _C
    p_curtail = (vpp_cfg.curtail_c_per_kwh / _C
                 if vpp_cfg.curtail_c_per_kwh is not None else None)

    # Resolved through the same function the rolling path uses, so the same
    # YAML produces the same threshold on both. The value capture ratio between
    # a rolling run and this benchmark is only interpretable if it does: a
    # threshold that differed between the two would sit inside that ratio with
    # nothing to attribute it to.
    #
    # This replaced a branch that derived the threshold whenever the YAML
    # omitted the value. That combination now fails validation instead, so the
    # implicit "absent means derive" convention is gone rather than silently
    # resolving differently than it used to.
    min_export_resolution = resolve_min_export_price(
        vpp_cfg=vpp_cfg,
        prices_mwh=ctx.prices,
        site_traces=ctx.site_traces,
        site_ids=site_ids,
        power_kw=scenario.battery_spec.power_kw,
        interval_hours=interval_hours,
    )
    # /1000 here and NOT on the rolling path: vpp_sc_event takes $/kWh, while
    # VppWindowPlanner takes $/MWh and applies its own conversion.
    min_export_price_per_kwh = (
        min_export_resolution.value_dollars_per_mwh / 1000.0)

    # No-grid-charge mask: full-year boolean, identical across sites, built once
    # from the plan's windows. None when the plan defines no windows.
    no_grid_charge_mask = None
    if vpp_cfg.no_grid_charge_windows:
        import numpy as np
        no_grid_charge_mask = np.zeros(len(idx), dtype=bool)
        for window in vpp_cfg.no_grid_charge_windows:
            no_grid_charge_mask |= charge_window_mask(
                idx, window, scenario.state, interval_hours)

    site_args = []
    for site_id in site_ids:
        battery_spec = per_site_battery[site_id]
        site_trace   = ctx.site_traces[site_id]
        solar_kwh    = site_trace['pv_site_net_kwh'] * per_site_scale[site_id]
        load_kwh     = site_trace['gross_load_kwh']
        soc_min      = (battery_spec.backup_reserve_pct / 100.0) * battery_spec.capacity_kwh
        export_limit_kwh = per_site_export_limit[site_id] * interval_hours
        site_args.append((
            site_id, solar_kwh, load_kwh,
            op_p_imp, op_p_exp,
            battery_spec, ctx.prices, ctx.network_tariff, ctx.retail_tariff,
            soc_min, export_limit_kwh, interval_hours,
            vpp_exp_reward, vpp_imp_reward,
            vpp_cfg.kwh_limit, vpp_cfg.limit_applies_to,
            vpp_cfg.vpp_threshold_dollars_per_mwh, vpp_cfg.event_window_hours,
            p_curtail, gap_strategy, hh_p_imp, hh_p_exp,
            min_export_price_per_kwh, no_grid_charge_mask,
            scenario.apply_export_order_nudge, scenario.state,
            with_monthly_bills,
        ))

    # Classify events once using the shared price series. Stored on
    # ScenarioResult for downstream writers (same for all sites).
    vpp_event_arr = classify_vpp_events(
        ctx.prices,
        threshold_dollars_per_mwh=vpp_cfg.vpp_threshold_dollars_per_mwh,
        event_window_hours=vpp_cfg.event_window_hours,
        interval_hours=interval_hours,
    )
    vpp_event_series = pd.Series(vpp_event_arr, index=idx, dtype=bool)

    traces: dict[str, SiteTraces]  = {}
    bills:  dict[str, SiteBill]   = {}
    vpp_traces_all: dict[str, VppTraces] = {}
    def _emit_eov_warnings(warning_msgs: list[str]) -> None:
        """Re-emit EOV worker warnings, suppressing or formatting EOV_SKIP tokens."""
        for msg in warning_msgs:
            if msg.startswith('EOV_SKIP:'):
                # EOV_SKIP:site_id:win_start:remaining_cap:rhs_total:n_imp_hi_total:status=N
                parts = msg.split(':')
                remaining_cap = float(parts[3])
                if remaining_cap < 20.0:
                    # Cap near-exhausted -- skip is expected and low consequence;
                    # suppress to avoid alarming output on normal runs.
                    continue
                # Substantial remaining cap -- unexpected artefact, surface it.
                rhs   = float(parts[4])
                n_imp = float(parts[5])
                warnings.warn(
                    f"EOV infeasible fallback to SC [{parts[1]}] "
                    f"t={parts[2]}  remaining_cap={remaining_cap:.1f} kWh  "
                    f"load-solar={rhs:.1f} kWh  sc_n_imp_budget={n_imp:.1f} kWh  "
                    f"{parts[6]}",
                    stacklevel=3,
                )
            else:
                warnings.warn(msg, stacklevel=3)

    baseline_traces: dict[str, SiteTraces] = {}
    baseline_bills:  dict[str, SiteBill]  = {}
    monthly_bills: Optional[dict[str, dict[int, SiteBill]]] = (
        {} if with_monthly_bills else None
    )
    lp_seconds = 0.0

    if workers == 1:
        for i, (site_id, args) in enumerate(zip(site_ids, site_args), start=1):
            (sid, sc_t, sc_bill, eov_t, eov_vt,
             eov_bill, eov_dt, warning_msgs, eov_monthly) = _vpp_sc_event_site_worker(args)
            _emit_eov_warnings(warning_msgs)
            traces[sid]          = eov_t
            bills[sid]           = eov_bill
            vpp_traces_all[sid]  = eov_vt
            baseline_traces[sid] = sc_t
            baseline_bills[sid]  = sc_bill
            if with_monthly_bills:
                monthly_bills[sid] = eov_monthly
            lp_seconds += eov_dt
            if progress_callback is not None:
                progress_callback(i, n, sid, note=f"eov {eov_dt:.1f}s")
    else:
        import multiprocessing as _mp
        completed = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_mp.get_context('spawn'),
        ) as executor:
            future_to_site = {
                executor.submit(_vpp_sc_event_site_worker, args): site_id
                for site_id, args in zip(site_ids, site_args)
            }
            for future in as_completed(future_to_site):
                (sid, sc_t, sc_bill, eov_t, eov_vt,
                 eov_bill, eov_dt, warning_msgs, eov_monthly) = future.result()
                _emit_eov_warnings(warning_msgs)
                traces[sid]          = eov_t
                bills[sid]           = eov_bill
                vpp_traces_all[sid]  = eov_vt
                baseline_traces[sid] = sc_t
                baseline_bills[sid]  = sc_bill
                if with_monthly_bills:
                    monthly_bills[sid] = eov_monthly
                lp_seconds += eov_dt
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, n, sid, note=f"eov {eov_dt:.1f}s")

    elapsed = time.perf_counter() - t0
    return ScenarioResult(
        scenario=scenario, site_traces=traces, site_bills=bills,
        elapsed_seconds=elapsed, ingest_seconds=ingest_elapsed,
        lp_seconds=lp_seconds, n_workers=workers,
        baseline_traces=baseline_traces, baseline_bills=baseline_bills,
        vpp_traces=vpp_traces_all,
        vpp_event=vpp_event_series,
        site_monthly_bills=monthly_bills,
        min_export_price_resolution=min_export_resolution,
    )


# ----------------------------------------------------------------------
# Result writers
# ----------------------------------------------------------------------

def _format_elapsed(seconds: float) -> str:
    """Render elapsed seconds as a human-friendly string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{s:04.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{s:04.1f}s"


def _bill_row(bill: SiteBill) -> dict:
    """Flatten a SiteBill into a CSV row dict. Shared by write_bills_csv and
    write_monthly_bills_csv so the two column sets never drift apart."""
    return {
        'site_id': bill.site_id,
        'plan_type': bill.plan_type,
        'wholesale_usage_cost': bill.wholesale_usage_cost,
        'wholesale_feedin_credit': bill.wholesale_feedin_credit,
        'network_usage_cost': bill.network_usage_cost,
        'network_feedin_credit': bill.network_feedin_credit,
        'network_daily_charges': bill.network_daily_charges,
        'retail_usage_cost': bill.retail_usage_cost,
        'retail_feedin_credit': bill.retail_feedin_credit,
        'retail_daily_charges': bill.retail_daily_charges,
        'retail_other_credits': bill.retail_other_credits,
        'household_total': bill.household_total,
        'retailer_margin': bill.retailer_margin,
    }


def write_bills_csv(result: ScenarioResult, out_path: Path) -> None:
    """Write one row per site with all SiteBill fields."""
    rows = [_bill_row(bill) for bill in result.site_bills.values()]
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def write_monthly_bills_csv(result: ScenarioResult, out_dir: Path) -> None:
    """Write bills_01.csv .. bills_12.csv into out_dir, one row per site.

    Purely a writer: reads result.site_monthly_bills (computed inline during
    run_scenario, via with_monthly_bills=True, sharing one settlement build
    per site with the annual site_bills entry -- see
    billing.compute_bill_and_monthly_bills). Does not re-bill anything; a
    ScenarioResult from a run without with_monthly_bills=True has
    site_monthly_bills=None and this function raises rather than silently
    doing nothing.

    A month the scenario doesn't span (e.g. a partial-year run) is absent
    from a given site's monthly dict; no bills_MM.csv is written for a month
    absent from every site.

    See run_scenario's with_monthly_bills docstring for what's covered: the
    main dispatch site_bills only, not baseline_bills (the self_consumption
    VCR comparator); and compute_monthly_bills' docstring for how the
    monthly daily-charge counts reconcile with the annual bill.
    """
    if result.site_monthly_bills is None:
        raise ValueError(
            "write_monthly_bills_csv requires result.site_monthly_bills to "
            "be populated -- re-run this scenario with "
            "run_scenario(..., with_monthly_bills=True)."
        )

    rows_by_month: dict[int, list[dict]] = {m: [] for m in range(1, 13)}
    for site_id, monthly_bills in result.site_monthly_bills.items():
        for month, bill in monthly_bills.items():
            rows_by_month[month].append(_bill_row(bill))

    out_dir.mkdir(parents=True, exist_ok=True)
    for month in range(1, 13):
        rows = rows_by_month[month]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"bills_{month:02d}.csv", index=False)


def write_rolling_diagnostics_csv(result: ScenarioResult, out_path: Path) -> None:
    """Write one row per site with the RollingDiagnostics summary fields.

    Unlike write_monthly_bills_csv, this does NOT raise when the field is
    absent -- it writes nothing and returns. main.py calls it unconditionally
    for every scenario (same style as write_bills_csv), and result.
    rolling_diagnostics is None for every coordination_type except the
    rolling ones, which is the normal case for most runs rather than a
    missing-data error.

    RollingDiagnostics.retained is deliberately excluded: it holds full
    per-window forecast vectors and plans (see rolling_optimisation.py) and
    is only ever non-empty for the single-site scenarios where
    validate_scenario permits rolling.retain_plans_from -- a diagnostic
    consumed directly from the ScenarioResult in that workflow, not through
    this summary CSV.
    """
    if result.rolling_diagnostics is None:
        return
    rows = []
    for site_id, diag in result.rolling_diagnostics.items():
        rows.append({
            'site_id': site_id,
            'n_intervals': diag.n_intervals,
            'n_windows_solved': diag.n_windows_solved,
            'n_fallbacks': diag.n_fallbacks,
            'sc_fallback_no_forecast': diag.sc_fallback_no_forecast,
            'sc_fallback_infeasible': diag.sc_fallback_infeasible,
            'sc_fallback_no_load_forecast': diag.sc_fallback_no_load_forecast,
            'sc_fallback_no_solar_forecast': diag.sc_fallback_no_solar_forecast,
            'infill_fraction_mean': diag.infill_fraction_mean,
            'staleness_minutes_mean': diag.staleness_minutes_mean,
            'staleness_minutes_max': diag.staleness_minutes_max,
            'coarse_staleness_minutes_mean': diag.coarse_staleness_minutes_mean,
            'coarse_staleness_minutes_max': diag.coarse_staleness_minutes_max,
            'solve_seconds_total': diag.solve_seconds_total,
            'solve_ms_median': diag.solve_ms_median,
            'solve_ms_p95': diag.solve_ms_p95,
            'terminal_soc_kwh': diag.terminal_soc_kwh,
            'terminal_soc_target_kwh': diag.terminal_soc_target_kwh,
            'max_run_time_lead_minutes': diag.max_run_time_lead_minutes,
            'causality_ok': diag.causality_ok,
            'load_forecast_used': diag.load_forecast_used,
            'load_windows_warm_up': diag.load_windows_warm_up,
            'load_positions_clipped': diag.load_positions_clipped,
            'infeasible_from_site_forecast': diag.infeasible_from_site_forecast,
            'solar_forecast_used': diag.solar_forecast_used,
            'solar_windows_warm_up': diag.solar_windows_warm_up,
            'solar_positions_clipped': diag.solar_positions_clipped,
            # Advanced TOU plans only. Empty for every other coordination
            # type, and empty for an advanced plan that lacks the feature --
            # a blank cell and a 0 mean different things here, so the None
            # is written through rather than coerced.
            'reward_days_lost': diag.reward_days_lost,
            'block_tier_days_exhausted': (
                None if diag.block_tier_days_exhausted is None
                else ';'.join(str(v)
                              for v in diag.block_tier_days_exhausted)),
            'windows_with_tier_exhausted': diag.windows_with_tier_exhausted,
            # VPP fields (Part C). None on every non-VPP path.
            'events_anchored': diag.events_anchored,
            'event_intervals': diag.event_intervals,
            'event_intervals_fallback': diag.event_intervals_fallback,
            'event_blocks_no_spike': diag.event_blocks_no_spike,
            'event_block_len_min': diag.event_block_len_min,
            'event_block_len_max': diag.event_block_len_max,
            'event_block_len_mean': diag.event_block_len_mean,
            'event_blocks_len_1': diag.event_blocks_len_1,
            'n_lt_W': diag.n_lt_W,
            'n_lt_W_first': diag.n_lt_W_first,
            'n_lt_W_last': diag.n_lt_W_last,
            'n_imp_budget_clamps': diag.n_imp_budget_clamps,
            'n_imp_resettled_kwh': diag.n_imp_resettled_kwh,
            'blocks_resettled': diag.blocks_resettled,
            'f_imp_total_kwh': diag.f_imp_total_kwh,
            'f_exp_total_kwh': diag.f_exp_total_kwh,
            'f_exp_blocked_by_price': diag.f_exp_blocked_by_price,
            'operator_infeasible': diag.operator_infeasible,
            'cap_limit_kwh': diag.cap_limit_kwh,
            'cap_used_kwh': diag.cap_used_kwh,
            'cap_remaining_kwh': diag.cap_remaining_kwh,
            'cap_exhausted_at': diag.cap_exhausted_at,
        })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def write_scenario_summary(result: ScenarioResult, out_path: Path) -> str:
    """Write a human-readable summary of the scenario; return the text."""
    s = result.scenario
    bills = list(result.site_bills.values())
    n = len(bills)

    # Aggregate across sites
    def total(field):
        return sum(getattr(b, field) for b in bills)

    L = []
    L.append(f"Scenario: {s.scenario_id}")
    L.append("=" * 60)
    L.append(f"  state={s.state}  year={s.year}  cohort={s.cohort}")
    L.append(f"  coordination_type={s.coordination_type}")
    if s.battery_spec is not None:
        b = s.battery_spec
        bn = f" '{s.battery_name}'" if s.battery_name else ""
        L.append(f"  battery{bn}: {b.capacity_kwh} kWh / {b.power_kw} kW / "
                 f"reserve {b.backup_reserve_pct}% / "
                 f"eff {b.charge_efficiency}/{b.discharge_efficiency}")
    L.append(f"  network_tariff={s.network_tariff_id}")
    L.append(f"  retail_tariff={s.retail_tariff_id}")
    L.append(f"  sites={n}")
    if result.elapsed_seconds > 0:
        per_site = result.elapsed_seconds / n if n else 0.0
        dispatch_elapsed = result.elapsed_seconds - result.ingest_seconds
        dispatch_per_site = dispatch_elapsed / n if n else 0.0
        L.append(f"  elapsed={_format_elapsed(result.elapsed_seconds)}"
                 f"  ({per_site:.2f}s/site)")
        L.append(f"  ingest={_format_elapsed(result.ingest_seconds)}"
                 f"  dispatch+billing={_format_elapsed(dispatch_elapsed)}"
                 f"  ({dispatch_per_site:.2f}s/site)")
        if result.lp_seconds > 0:
            lp_per_site = result.lp_seconds / n if n else 0.0
            if result.n_workers > 1:
                # Cumulative worker-seconds exceed wall-clock in parallel;
                # report cumulative + per-site avg + worker count instead.
                lp_wallclock = result.elapsed_seconds - result.ingest_seconds
                L.append(f"  LP solve={_format_elapsed(result.lp_seconds)}"
                         f" total ({lp_per_site:.2f}s/site avg,"
                         f" {result.n_workers} workers,"
                         f" {_format_elapsed(lp_wallclock)} wall-clock)")
            else:
                rest = dispatch_elapsed - result.lp_seconds
                L.append(f"  LP solve={_format_elapsed(result.lp_seconds)}"
                         f"  ({lp_per_site:.2f}s/site)"
                         f"  baseline+billing={_format_elapsed(rest)}")
    L.append("")
    L.append(f"COHORT TOTALS ($)")
    L.append("-" * 60)
    for field in (
        'wholesale_usage_cost', 'wholesale_feedin_credit',
        'network_usage_cost', 'network_feedin_credit', 'network_daily_charges',
        'retail_usage_cost', 'retail_feedin_credit', 'retail_daily_charges',
        'household_total', 'retailer_margin',
    ):
        L.append(f"  {field:30}{total(field):14.2f}")
    L.append("")
    L.append(f"PER-SITE AVERAGES ($)")
    L.append("-" * 60)
    L.append(f"  household_total                {total('household_total')/n:14.2f}")
    L.append(f"  retailer_margin                {total('retailer_margin')/n:14.2f}")
    L.append("")

    # VCR comparison: present for both perfect_foresight (LP) and
    # perfect_foresight_custom (greedy). site_bills holds the optimised bills;
    # baseline_bills holds the self_consumption bills for comparison.
    if result.baseline_bills is not None and n:
        coord = result.scenario.coordination_type
        opt_label = 'CUSTOM GREEDY' if coord == 'perfect_foresight_custom' else 'LP'
        dispatch_label = 'custom' if coord == 'perfect_foresight_custom' else 'LP'
        lp_total = total('household_total')
        sc_total = sum(b.household_total
                       for b in result.baseline_bills.values())
        saving = sc_total - lp_total
        pct = (saving / sc_total * 100.0) if sc_total else 0.0
        n_worse = sum(
            1 for sid in result.site_bills
            if result.site_bills[sid].household_total
            > result.baseline_bills[sid].household_total + 1e-4
        )
        L.append(f"{opt_label} vs SELF-CONSUMPTION (perfect-foresight value capture)")
        L.append("-" * 60)
        L.append(f"  avg self_consumption household_total {sc_total/n:14.2f}")
        L.append(f"  avg {dispatch_label} household_total        {lp_total/n:14.2f}")
        L.append(f"  avg saving per site                  {saving/n:14.2f}")
        L.append(f"  total saving ({pct:5.1f}%)              {saving:14.2f}")
        L.append(f"  sites where {dispatch_label} worse than SC  {n_worse:14d}   (expect 0)")
        L.append("")

    r = result.min_export_price_resolution
    if r is not None:
        L.append("MINIMUM EXPORT PRICE")
        L.append("-" * 60)
        L.append(f"  mode                                 {r.mode}")
        L.append(f"  operative ($/MWh)                    "
                 f"{r.value_dollars_per_mwh:14.1f}   (from {r.source})")
        cfg = ('none' if r.configured_dollars_per_mwh is None
               else f"{r.configured_dollars_per_mwh:.1f}")
        L.append(f"  configured ($/MWh)                   {cfg:>14}")
        if r.derived_dollars_per_mwh is not None:
            L.append(f"  derived ($/MWh)                      "
                     f"{r.derived_dollars_per_mwh:14.1f}")
            L.append(f"  event-mask intervals                 "
                     f"{r.n_mask_intervals:14d}")
            # A zero derived threshold means one of two different things, and
            # only this line separates them: the budget could not be spent
            # within the available event intervals, or it could and the
            # marginal price was genuinely zero.
            L.append(f"  throughput available (kWh)           "
                     f"{r.available_kwh:14.1f}"
                     f"   (cap {'reached' if r.budget_exhausted else 'NOT reached'})")
        L.append("")

    text = "\n".join(L)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return text


def write_flows_parquet(result: ScenarioResult, out_path: Path) -> None:
    """Write per-interval net import/export for every site to a wide Parquet file.

    Columns: datetime_aest (timestamp[us, tz=Etc/GMT-10]) plus one float32
    column per site_id holding net kWh at each 5-minute interval, where
    positive = net import and negative = net export (import_kwh - export_kwh).

    Uses result.site_traces (the dispatched coordination strategy outcome).
    Compression: snappy, matching the NEM price and EDP clean files.
    """
    if not result.site_traces:
        import warnings as _w
        _w.warn(f"write_flows_parquet: no site traces in result for "
                f"{result.scenario.scenario_id}; skipping.", stacklevel=2)
        return

    first = next(iter(result.site_traces.values()))
    index = first.import_kwh.index

    cols = {
        site_id: (traces.import_kwh - traces.export_kwh).astype('float32').to_numpy()
        for site_id, traces in result.site_traces.items()
    }
    df = pd.DataFrame(cols, index=index)
    df.index.name = 'datetime_aest'

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression='snappy')


def write_battery_flows_parquet(result: ScenarioResult, out_path: Path) -> None:
    """Write per-interval net battery AC flow for every site to a wide Parquet file.

    Columns: datetime_aest (timestamp[us, tz=Etc/GMT-10]) plus one float32
    column per site_id holding net battery kWh at each 5-minute interval,
    where positive = net charging and negative = net discharging
    (sol_to_bat_kwh + grid_to_bat_kwh - bat_to_home_kwh - bat_to_grid_kwh).
    Mirrors flows.parquet's positive = import sign convention.

    Uses result.site_traces (the dispatched coordination strategy outcome).
    Only written for battery cohorts (result.scenario.battery_spec is not
    None); skipped otherwise, since the battery flow columns are all-zero
    by convention for no_cer/solar cohorts (see SiteTraces docstring) and
    the file would carry no information.

    Compression: snappy, matching flows.parquet and the NEM price files.
    """
    if result.scenario.battery_spec is None:
        return

    if not result.site_traces:
        import warnings as _w
        _w.warn(f"write_battery_flows_parquet: no site traces in result for "
                f"{result.scenario.scenario_id}; skipping.", stacklevel=2)
        return

    first = next(iter(result.site_traces.values()))
    index = first.import_kwh.index

    cols = {
        site_id: (traces.sol_to_bat_kwh + traces.grid_to_bat_kwh
                  - traces.bat_to_home_kwh - traces.bat_to_grid_kwh
                  ).astype('float32').to_numpy()
        for site_id, traces in result.site_traces.items()
    }
    df = pd.DataFrame(cols, index=index)
    df.index.name = 'datetime_aest'

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression='snappy')


def write_vpp_forced_flows_parquet(result: ScenarioResult, out_path: Path) -> None:
    """Write per-interval forced net flow for every site to a wide Parquet file.

    Only written for VPP coordination types (result.vpp_traces and
    result.vpp_event must both be non-None). Rows are all intervals inside a
    VPP event window, as determined by result.vpp_event (the boolean Series
    produced by classify_vpp_events during the run). This includes intervals
    within an event where all forced flows happen to be zero, so the event
    boundaries are fully preserved.

    Columns: datetime_aest (timestamp[us, tz=Etc/GMT-10]) plus one float32
    column per site_id holding forced net kWh at each event interval, where
    positive = forced import (operator charging battery) and negative = forced
    export (operator discharging battery to grid).

    Computed as f_imp_kwh - f_exp_kwh from VppTraces. Combined with
    flows.parquet (total net import = n_imp - n_exp + f_imp - f_exp), this
    allows reconstruction of natural and forced components for billing analysis.

    Compression: snappy, matching flows.parquet and NEM price files.
    """
    if result.vpp_traces is None or result.vpp_event is None:
        return

    if not result.vpp_traces:
        import warnings as _w
        _w.warn(f"write_vpp_forced_flows_parquet: empty vpp_traces for "
                f"{result.scenario.scenario_id}; skipping.", stacklevel=2)
        return

    event_mask = result.vpp_event.to_numpy(dtype=bool)
    event_index = result.vpp_event.index[event_mask]

    cols = {
        site_id: (vt.f_imp_kwh - vt.f_exp_kwh
                  ).loc[event_index].astype('float32').to_numpy()
        for site_id, vt in result.vpp_traces.items()
    }
    df = pd.DataFrame(cols, index=event_index)
    df.index.name = 'datetime_aest'

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression='snappy')


def write_single_trace(result: ScenarioResult, site_id: str, out_path: Path) -> None:
    """Write one site's per-interval SiteTraces to CSV."""
    if site_id not in result.site_traces:
        raise KeyError(
            f"Site {site_id!r} not in scenario {result.scenario.scenario_id}. "
            f"Available: {sorted(result.site_traces)[:5]}..."
        )
    t = result.site_traces[site_id]
    df = pd.DataFrame({
        'datetime_aest': t.import_kwh.index.strftime('%Y-%m-%d %H:%M:%S%z'),
        'import_kwh': t.import_kwh.values,
        'export_kwh': t.export_kwh.values,
        'battery_soc_kwh': t.battery_soc_kwh.values,
        'sol_to_home_kwh': t.sol_to_home_kwh.values,
        'sol_to_bat_kwh': t.sol_to_bat_kwh.values,
        'sol_to_grid_kwh': t.sol_to_grid_kwh.values,
        'bat_to_home_kwh': t.bat_to_home_kwh.values,
        'grid_to_home_kwh': t.grid_to_home_kwh.values,
        'grid_to_bat_kwh': t.grid_to_bat_kwh.values,
        'bat_to_grid_kwh': t.bat_to_grid_kwh.values,
        'curtail_kwh': t.curtail_kwh.values,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)