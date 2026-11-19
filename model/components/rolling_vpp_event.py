"""The rolling VPP coordination component.

A thin wrapper over `rolling_optimisation`. The loop is unchanged and still
returns `(SiteTraces, RollingDiagnostics)`; this builds the `VppWindowPlanner`,
hands it to the loop through the existing `planner` argument, and afterwards
reads the natural/forced split off the planner to assemble `VppTraces`. The
loop stays ignorant of VPP.

Two coordination types share this component, differing only in what the
household does between events:

    gap_strategy='self_consumption'  -> rolling_vpp_sc_event
    gap_strategy='tou_advanced'      -> rolling_vpp_tou_event

which mirrors the `gap_strategy` parameter on the perfect-foresight
`vpp_sc_event`, where the same single component serves `vpp_sc_event` and
`vpp_tou_event`.

Phantom events
--------------
`event_blocks_no_spike` counts declared blocks in which no SETTLED price ever
exceeded the threshold: blocks the operator opened on a forecast spike that
never arrived. It is the direct measure of what imperfect operator foresight
costs, and the headline quantity this coordination type exists to produce. The
planner cannot compute it, because it only ever sees forecasts. The wrapper
can, because it holds the settled series, so it feeds the settled comparison
in after the run.

See rolling_vpp_event_spec.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.core.prices import PriceStackOffsets
from model.core.scenario import BatterySpec, RollingConfig
from model.core.site_result import SiteTraces, VppTraces
from model.components.rolling_advanced import AdvancedWindowPlanner
from model.components.rolling_optimisation import (
    RollingDiagnostics, rolling_optimisation,
)
from model.components.vpp_declaration import CapState, window_intervals
from model.components.vpp_window_planner import OPERATOR, VppWindowPlanner


def rolling_vpp_event(*args, **kwargs):
    """Run a rolling VPP dispatch. See run_with_planner for the arguments.

    Returns (SiteTraces, VppTraces, RollingDiagnostics).
    """
    traces, vpp, diagnostics, _planner = run_with_planner(*args, **kwargs)
    return traces, vpp, diagnostics


def run_with_planner(
    site_id: str,
    solar_kwh: pd.Series,
    load_kwh: pd.Series,
    prices_mwh: pd.Series,
    op_price_offsets: PriceStackOffsets,
    forecast_provider,
    battery_spec: BatterySpec,
    soc_init_kwh: float,
    soc_final_kwh: float,
    config: RollingConfig,
    vpp_threshold_dollars_per_mwh: float,
    event_window_hours: float,
    vpp_exp_reward: float,
    vpp_imp_reward: float,
    vpp_kwh_limit: float,
    vpp_limit_applies_to: str,
    min_export_price_dollars_per_mwh: float,
    gap_strategy: str = 'self_consumption',
    export_limit_kwh_per_interval: float = np.inf,
    interval_hours: float = 5 / 60,
    p_curtail: float | None = None,
    no_grid_charge_mask: np.ndarray | None = None,
    household_planner=None,
    retail_tariff=None,
    network_tariff=None,
    state: str | None = None,
    load_forecast_provider=None,
    solar_forecast_provider=None,
    commit_actual_load: bool = True,
    progress=None,
    apply_export_order_nudge: bool = False,
) -> tuple[SiteTraces, VppTraces, RollingDiagnostics]:
    """Dispatch a VPP-enrolled battery under imperfect foresight on both sides.

    Args:
        prices_mwh: SETTLED wholesale prices for the period. Used only after
            the run, to identify blocks that were declared on a forecast spike
            that never materialised. Never reaches the planner or the LP.
        op_price_offsets: Operator tariff price stack. The operator LP's prices
            come from the FORECAST RRP through this.
        forecast_provider: Price forecasts. Required: the operator reads them
            to declare events even when the household plan's prices are fixed
            by the clock.
        min_export_price_dollars_per_mwh: Fleet-wide scalar for the whole
            period, resolved by the orchestrator before the worker pool spawns.
        gap_strategy: 'self_consumption' or 'tou_advanced'.
        household_planner: Optional. On gap_strategy='tou_advanced' an
            AdvancedWindowPlanner is built from retail_tariff, network_tariff
            and state unless one is supplied. Must be None for
            'self_consumption', whose dispatch is deterministic.
        retail_tariff, network_tariff, state: Required on 'tou_advanced' to
            build the household planner. The household is billed on TOTAL
            import and export, including the flows the operator forces during
            events, so this planner's day state -- block tiers, the zero-import
            reward -- must observe event intervals. VppWindowPlanner forwards
            observe_commit to it unconditionally for exactly that reason.
        apply_export_order_nudge: Passed to the household
            AdvancedWindowPlanner when this function builds it. Ignored when
            household_planner is supplied, which carries its own setting.

    Returns:
        (SiteTraces, VppTraces, RollingDiagnostics, VppWindowPlanner).

        The planner is returned because the per-interval mode array is not
        recoverable from the outputs: forced flows are non-zero only inside
        events, but an event interval where the operator chose to do nothing,
        or where the LP was infeasible and self-consumption was committed,
        carries no forced flow. Anything that needs the declared mask -- the
        viz shading, a mask comparison against classify_vpp_events -- has to
        read it here. rolling_vpp_event drops it and returns the documented
        triple.
    """
    if gap_strategy not in ('self_consumption', 'tou_advanced'):
        raise ValueError(
            f"gap_strategy must be 'self_consumption' or 'tou_advanced', "
            f"got {gap_strategy!r}"
        )
    if gap_strategy == 'tou_advanced' and household_planner is None:
        # Built here rather than demanded from the caller: it needs the same
        # index, battery and horizon the loop is about to run on, and deriving
        # them twice is how the two drift apart.
        if retail_tariff is None or network_tariff is None or state is None:
            raise ValueError(
                "gap_strategy='tou_advanced' needs retail_tariff, "
                "network_tariff and state to build the household planner, or "
                "a household_planner passed in directly"
            )
        household_planner = AdvancedWindowPlanner(
            index=solar_kwh.index,
            retail_tariff=retail_tariff,
            network_tariff=network_tariff,
            state=state,
            battery_spec=battery_spec,
            horizon_intervals=max(1, int(round(
                config.horizon_hours / interval_hours))),
            export_limit_kwh_per_interval=export_limit_kwh_per_interval,
            interval_hours=interval_hours,
            curtail_cost_per_kwh_dollars=p_curtail,
            # Settled prices for the household day state: block usage leaves
            # out flow that was critical by the settled price, as billing does.
            settled_rrp_mwh=prices_mwh,
            apply_export_order_nudge=apply_export_order_nudge,
        )
    if gap_strategy == 'self_consumption' and household_planner is not None:
        raise ValueError(
            "gap_strategy='self_consumption' dispatches deterministically "
            "between events and takes no household_planner"
        )
    if forecast_provider is None:
        raise ValueError(
            "rolling_vpp_event requires a price forecast provider: the "
            "operator declares events from it"
        )

    index = solar_kwh.index
    T = len(index)
    if len(prices_mwh) != T:
        raise ValueError(
            f"prices_mwh covers {len(prices_mwh)} intervals but the site "
            f"traces cover {T}"
        )

    w = window_intervals(event_window_hours, interval_hours)
    horizon_intervals = max(1, int(round(
        config.horizon_hours / interval_hours)))
    if horizon_intervals < w:
        raise ValueError(
            f"horizon of {config.horizon_hours}h ({horizon_intervals} "
            f"intervals) is shorter than the event window of "
            f"{event_window_hours}h ({w} intervals); every declaration "
            f"lookahead would be truncated"
        )

    planner = VppWindowPlanner(
        T=T,
        op_price_offsets=op_price_offsets,
        battery_spec=battery_spec,
        w=w,
        vpp_threshold_dollars_per_mwh=vpp_threshold_dollars_per_mwh,
        vpp_exp_reward=vpp_exp_reward,
        vpp_imp_reward=vpp_imp_reward,
        cap=CapState(limit_kwh=vpp_kwh_limit,
                     limit_applies_to=vpp_limit_applies_to),
        min_export_price_dollars_per_mwh=min_export_price_dollars_per_mwh,
        export_limit_kwh_per_interval=export_limit_kwh_per_interval,
        interval_hours=interval_hours,
        site_id=site_id,
        p_curtail=p_curtail,
        no_grid_charge_mask=no_grid_charge_mask,
        household_planner=household_planner,
    )

    traces, diagnostics = rolling_optimisation(
        site_id=site_id,
        solar_kwh=solar_kwh,
        load_kwh=load_kwh,
        forecast_provider=forecast_provider,
        price_offsets=op_price_offsets,
        battery_spec=battery_spec,
        soc_init_kwh=soc_init_kwh,
        soc_final_kwh=soc_final_kwh,
        config=config,
        export_limit_kwh_per_interval=export_limit_kwh_per_interval,
        interval_hours=interval_hours,
        curtail_cost_per_kwh_dollars=p_curtail,
        progress=progress,
        load_forecast_provider=load_forecast_provider,
        solar_forecast_provider=solar_forecast_provider,
        commit_actual_load=commit_actual_load,
        planner=planner,
    )

    # A block still open at the period end has never been settled: its import
    # split would otherwise keep whatever the LP booked against a budget the
    # event was still projecting forward.
    planner.finalize()

    diagnostics = _apply_settled_price_diagnostics(
        planner, diagnostics, prices_mwh, vpp_threshold_dollars_per_mwh)

    vpp_traces = VppTraces(
        site_id=site_id,
        n_imp_kwh=pd.Series(planner.n_imp, index=index),
        f_imp_kwh=pd.Series(planner.f_imp, index=index),
        n_exp_kwh=pd.Series(planner.n_exp, index=index),
        f_exp_kwh=pd.Series(planner.f_exp, index=index),
    )
    _check_split_invariants(traces, vpp_traces, site_id)
    return traces, vpp_traces, diagnostics, planner


def count_blocks_without_a_settled_spike(
    mode: np.ndarray,
    prices_mwh: np.ndarray,
    threshold_dollars_per_mwh: float,
) -> int:
    """Declared blocks containing no settled price above the threshold.

    Each maximal run of operator-mode intervals is one block. A block whose
    settled prices never exceeded the threshold was opened on a forecast spike
    that never arrived -- the operator took control of a battery it then had
    no reason to dispatch.

    Separated from the wrapper so it can be applied to any (mode, price) pair,
    including in the viz where the two are plotted together.
    """
    op = np.asarray(mode) == OPERATOR
    prices = np.asarray(prices_mwh, dtype=float)
    if op.size != prices.size:
        raise ValueError(
            f"mode covers {op.size} intervals, prices {prices.size}")
    if not op.any():
        return 0

    spike = prices > threshold_dollars_per_mwh
    # Block boundaries: where op flips on, and where it flips off.
    padded = np.concatenate([[False], op, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return int(sum(1 for s, e in zip(starts, ends) if not spike[s:e].any()))


def _apply_settled_price_diagnostics(
    planner, diagnostics: RollingDiagnostics,
    prices_mwh: pd.Series, threshold: float,
) -> RollingDiagnostics:
    """Refresh the VPP counters after finalize(), and fill the ones that need
    settled prices rather than forecasts.

    The loop reads plan_diagnostics() and builds RollingDiagnostics BEFORE the
    wrapper calls finalize(), so any counter touched while settling a block
    still open at the period end -- the block length, the resettled volume --
    is missing from what the loop returned. Re-reading the planner here rather
    than patching individual fields keeps this correct if a counter is added
    later.
    """
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(RollingDiagnostics)}
    refreshed = {k: v for k, v in planner.plan_diagnostics().items()
                 if k in field_names}
    refreshed['event_blocks_no_spike'] = count_blocks_without_a_settled_spike(
        planner.mode, prices_mwh.to_numpy(), threshold)
    return dataclasses.replace(diagnostics, **refreshed)


def _check_split_invariants(
    traces: SiteTraces, vpp: VppTraces, site_id: str, atol: float = 1e-6,
) -> None:
    """The invariant VppTraces documents and billing relies on.

    Checked rather than assumed because the split and the headline series come
    from different places -- the split from the LP's own variables, the
    headline series from the loop's flow reconstruction -- and a packing error
    would leave both individually plausible.
    """
    for name, split, headline in (
        ('import', vpp.n_imp_kwh + vpp.f_imp_kwh, traces.import_kwh),
        ('export', vpp.n_exp_kwh + vpp.f_exp_kwh, traces.export_kwh),
    ):
        diff = np.abs(split.to_numpy() - headline.to_numpy())
        worst = int(np.argmax(diff)) if diff.size else 0
        if diff.size and diff[worst] > atol:
            raise AssertionError(
                f"[{site_id}] VppTraces {name} split does not reconstruct "
                f"SiteTraces.{name}_kwh: worst mismatch {diff[worst]:.6g} kWh "
                f"at interval {worst} ({headline.index[worst]}), "
                f"split={split.to_numpy()[worst]:.6g} vs "
                f"headline={headline.to_numpy()[worst]:.6g}; "
                f"{int((diff > atol).sum())} interval(s) exceed {atol}"
            )
