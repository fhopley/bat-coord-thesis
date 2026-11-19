"""rolling_optimisation: battery dispatch under imperfect price foresight.

At each 5-minute decision instant the optimiser plans over a finite horizon
using the price forecast available at that moment, commits the current
interval's battery action, advances one interval, and re-plans against the
updated forecast.

Returns a SiteTraces in exactly the same shape as self_consumption and
perfect_foresight_lp, so billing and every downstream consumer are unchanged.

See rolling_optimisation_spec.md for the design. Two things are worth knowing
before reading the code.

Why a stateful solver
---------------------
The window subproblem IS perfect_foresight_lp over H intervals: same 6T
variable layout, same 2T equality rows, same objective. Crucially, A_eq
depends only on (T, eta_c, eta_d) and not on solar, load, prices, the export
limit or any SOC, so for a fixed horizon the constraint matrix is identical
for every window of every site. The model is therefore built once and only its
cost, bounds and row bounds are mutated between windows, with the basis
retained. Measured at roughly 4.4x faster than rebuilding, which is the whole
argument for depending on highspy directly rather than going through
scipy.optimize.linprog.

That makes the component internally stateful while still presenting as a pure
function, which is a deliberate documented exception to the "components are
pure functions" convention rather than an accident.

Terminal SOC
------------
Two rules, applied automatically by position in the period, with no policy
parameter:

  - A window reaching the END of the period pins soc[k_end] to the terminal
    SOC of a self-consumption run over the WHOLE period, which is the same
    value perfect_foresight_lp is seeded with. All three dispatches therefore
    finish at the same state and the bill comparison needs no residual
    adjustment.
  - Any other window pins soc[k_end] to the SOC reached by running
    self-consumption forward from the current committed SOC across the window.

The forward pin is a hard constraint, but it constrains the ENDPOINT, not the
path. It is harmless when the profitable cycle completes inside the window: the
battery can charge overnight, discharge into a morning peak, and still arrive at
a pinned empty state. It bites only when a cycle would have to straddle the
window boundary. The horizon therefore needs to be long enough to contain a
complete charge-discharge cycle from any decision instant. RollingConfig
defaults to 48 hours; the production configs use 18 hours, set by a real-data
sweep. See rolling_optimisation_spec.md, "Horizon and infill".
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from model.core.forecast import (
    ForecastProvider, LoadForecastProvider, SolarForecastProvider,
)
from model.core.prices import PriceStackOffsets
from model.core.scenario import BatterySpec, RollingConfig
from model.core.site_result import SiteTraces
from model.components.perfect_foresight_lp import (
    _clip_nonneg, _net_simultaneous_charge_discharge, _reconstruct_flows,
)
from model.components.self_consumption import run_sc_with_battery

try:
    import highspy
except ImportError as exc:                                  # pragma: no cover
    raise ImportError(
        "rolling_optimisation requires highspy. Install with "
        "`pip install highspy` and add it to pyproject dependencies. It is "
        "the same HiGHS solver scipy.optimize.linprog already calls, used "
        "directly so the model can be reused across windows."
    ) from exc


_MAX_FALLBACK_TIMESTAMPS = 20      # retained for diagnosis, not a full list


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetainedWindow:
    """One decision instant's full context, for the forecast evolution matrix.

    Retained only for decision instants inside the configured window, because
    the point is to reconstruct what the model knew approaching a price spike,
    not to keep everything. Three days at 5-minute cadence with a 48-hour
    horizon is roughly 2.5 MB, so retention around spike days can be generous.

    A row is retained for EVERY decision instant in the window, including the
    ones that fell back to self-consumption. Skipping those would leave holes
    in the retained sequence exactly where the controller failed, and a
    forecast surface plotted from it would show a gap that reads as missing
    data rather than as a fallback. On a fallback, solved is False and the
    plan arrays are None; where the fallback was a missing forecast the
    forecast arrays are empty as well, because there was nothing to retain.

    Attributes:
        run_time: Label of the fine run, the freshest thing in the window.
        coarse_run_time: Label of the coarse run, which supplies everything
            beyond the fine segment and can be up to 25 minutes older.
        is_fine: True where a value came from the fine segment.
        plan_import, plan_export, plan_curtail: the planned GRID flows. Taken
            from the solution vector rather than derived from solar, load and
            the battery flows, because the LP already solves for them and
            re-deriving them downstream would mean re-implementing the
            efficiency, curtailment and export-limit conventions outside the
            component that owns them.
        solved: False where the window fell back to self-consumption.
    """
    decision_instant: pd.Timestamp
    run_time: pd.Timestamp
    coarse_run_time: pd.Timestamp
    rrp_mwh: np.ndarray
    is_infill: np.ndarray
    is_fine: np.ndarray
    plan_charge: np.ndarray | None
    plan_discharge: np.ndarray | None
    plan_soc: np.ndarray | None
    plan_import: np.ndarray | None
    plan_export: np.ndarray | None
    plan_curtail: np.ndarray | None
    committed_soc: float
    solved: bool = True
    # Empty when load is known perfectly. Retained alongside the price
    # forecast so a per-interval post-mortem can separate a price-driven
    # mistake from a load-driven one; with only one of the two retained the
    # two are indistinguishable in the artefact.
    load_forecast_kwh: np.ndarray | None = None
    solar_forecast_kwh: np.ndarray | None = None


@dataclass(frozen=True)
class RollingDiagnostics:
    """Counters and context from one rolling run.

    Returned rather than logged so the check script and any caller can both
    consume it.

    On the two staleness pairs. staleness_minutes_* measures the FINE run and
    is zero by construction whenever fine coverage is complete, because a
    5-minute source publishes a run at every decision instant. That is not a
    useless number: it goes sharply positive when a store ends early and the
    nearest run is carried forward, which is the failure it exists to catch.
    But it says nothing about how old the forward window is.
    coarse_staleness_minutes_* measures the coarse run, which is routinely 0
    to 25 minutes old by design and is what shapes everything the plan is
    built on beyond the first hour.
    """
    site_id: str
    n_intervals: int
    n_windows_solved: int
    sc_fallback_no_forecast: int
    sc_fallback_infeasible: int
    fallback_timestamps: tuple[pd.Timestamp, ...]
    infill_fraction_mean: float
    infill_fraction_by_hour: dict[int, float]
    staleness_minutes_mean: float
    staleness_minutes_max: float
    solve_seconds_total: float
    solve_ms_median: float
    solve_ms_p95: float
    terminal_soc_kwh: float
    terminal_soc_target_kwh: float
    max_run_time_lead_minutes: float
    causality_ok: bool
    retained: tuple[RetainedWindow, ...] = field(default=())
    # Appended rather than placed next to the fine-segment staleness so that
    # no existing field is reordered and no existing field is forced to take
    # a default it should not have.
    coarse_staleness_minutes_mean: float = 0.0
    coarse_staleness_minutes_max: float = 0.0
    # Load forecast. All zero / False when load is known perfectly.
    load_forecast_used: bool = False
    sc_fallback_no_load_forecast: int = 0
    load_windows_warm_up: int = 0
    load_positions_clipped: int = 0
    # Of the infeasible windows, how many became feasible when re-solved with
    # ACTUAL load. Those are attributable to the load forecast; the rest would
    # have happened anyway. The re-solve fires only on failure, so it costs
    # nothing in the common case, and its solution is discarded rather than
    # used -- using it would be exactly the leak the forecast exists to avoid.
    infeasible_from_site_forecast: int = 0
    # Solar forecast. All zero / False when solar is known perfectly.
    solar_forecast_used: bool = False
    sc_fallback_no_solar_forecast: int = 0
    solar_windows_warm_up: int = 0
    solar_positions_clipped: int = 0
    # Advanced TOU plans only, supplied by the planner. None means the plan
    # does not have the feature at all, which a reader needs to tell apart
    # from a feature that was never triggered -- "no zero-import reward on
    # this plan" and "the reward was never lost" are very different readings
    # of a zero.
    reward_days_lost: int | None = None
    block_tier_days_exhausted: tuple[int, ...] | None = None
    windows_with_tier_exhausted: int | None = None

    # VPP fields, populated only by VppWindowPlanner. None on every other
    # path, which a reader needs to tell apart from a counter that stayed at
    # zero because nothing triggered it. See rolling_vpp_event_spec.md,
    # "Diagnostics".
    events_anchored: int | None = None
    event_intervals: int | None = None
    event_intervals_fallback: int | None = None
    event_blocks_no_spike: int | None = None
    event_block_len_min: int | None = None
    event_block_len_max: int | None = None
    event_block_len_mean: float | None = None
    event_blocks_len_1: int | None = None
    n_lt_W: int | None = None
    n_lt_W_first: int | None = None
    n_lt_W_last: int | None = None
    n_imp_budget_clamps: int | None = None
    n_imp_resettled_kwh: float | None = None
    blocks_resettled: int | None = None
    f_imp_total_kwh: float | None = None
    f_exp_total_kwh: float | None = None
    f_exp_blocked_by_price: int | None = None
    operator_infeasible: int | None = None
    cap_limit_kwh: float | None = None
    cap_used_kwh: float | None = None
    cap_remaining_kwh: float | None = None
    cap_exhausted_at: int | None = None

    @property
    def n_fallbacks(self) -> int:
        return (self.sc_fallback_no_forecast + self.sc_fallback_infeasible
                + self.sc_fallback_no_load_forecast
                + self.sc_fallback_no_solar_forecast)

    def summary(self) -> str:
        """Human-readable block for check scripts and run logs."""
        lines = [
            f"rolling_optimisation [{self.site_id}]",
            f"  intervals            {self.n_intervals:,}",
            f"  windows solved       {self.n_windows_solved:,}",
            f"  SC fallbacks         {self.n_fallbacks:,} "
            f"(no price forecast {self.sc_fallback_no_forecast:,}, "
            f"no load forecast {self.sc_fallback_no_load_forecast:,}, "
            f"no solar forecast {self.sc_fallback_no_solar_forecast:,}, "
            f"infeasible {self.sc_fallback_infeasible:,})",
            f"  infill fraction      {self.infill_fraction_mean:.1%} mean",
            f"  fine run staleness   {self.staleness_minutes_mean:.1f} min mean, "
            f"{self.staleness_minutes_max:.0f} max",
            f"  coarse run staleness {self.coarse_staleness_minutes_mean:.1f} "
            f"min mean, {self.coarse_staleness_minutes_max:.0f} max",
            f"  solve time           {self.solve_ms_median:.2f} ms median, "
            f"{self.solve_ms_p95:.2f} ms p95, "
            f"{self.solve_seconds_total:.1f} s total",
            f"  terminal SOC         {self.terminal_soc_kwh:.3f} kWh "
            f"(target {self.terminal_soc_target_kwh:.3f})",
            f"  causality            {'OK' if self.causality_ok else 'VIOLATED'} "
            f"(max run-time lead {self.max_run_time_lead_minutes:.1f} min)",
        ]
        if self.load_forecast_used:
            lines.append(
                f"  load forecast        ON, {self.load_windows_warm_up:,} "
                f"warm-up window(s), {self.load_positions_clipped:,} position(s) "
                f"clipped at zero")
            if self.sc_fallback_infeasible:
                lines.append(
                    f"  infeasible from site {self.infeasible_from_site_forecast:,} "
                    f"of {self.sc_fallback_infeasible:,} "
                    f"(rest would have occurred under perfect load)")
        if self.solar_forecast_used:
            lines.append(
                f"  solar forecast       ON, {self.solar_windows_warm_up:,} "
                f"warm-up window(s), {self.solar_positions_clipped:,} "
                f"position(s) clipped at an index bound")
        if self.reward_days_lost is not None:
            lines.append(
                f"  zero-import reward   lost on {self.reward_days_lost:,} day(s)")
        if self.block_tier_days_exhausted is not None:
            per_window = ', '.join(
                f"w{i}: {v:,}" for i, v in
                enumerate(self.block_tier_days_exhausted))
            lines.append(
                f"  block tiers used up  {per_window} day(s); "
                f"{self.windows_with_tier_exhausted:,} window(s) planned with "
                f"no capacity left")
        if self.fallback_timestamps:
            shown = ', '.join(str(t) for t in self.fallback_timestamps[:5])
            lines.append(f"  first fallbacks      {shown}")
        if self.retained:
            lines.append(f"  retained windows     {len(self.retained):,}")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Window solver
# ---------------------------------------------------------------------------

class _WindowSolver:
    """A HiGHS model for a fixed window length T, mutated between windows.

    The constraint matrix is built once. Between windows only the objective
    coefficients for import and export, the flow-balance right-hand side, the
    initial SOC, the curtailment ceiling and the terminal SOC pin change; the
    basis is retained, so consecutive solves start from a near-optimal point.
    """

    def __init__(self, T: int, bspec: BatterySpec, export_limit: float,
                 interval_hours: float, curtail_cost: float | None):
        self.T = T
        self._curtail_pinned = curtail_cost is None
        self.OC, self.OD, self.OI, self.OE, self.OS, self.OQ = (
            0, T, 2 * T, 3 * T, 4 * T, 5 * T)
        N = 6 * T
        inf = highspy.kHighsInf

        c = np.zeros(N)
        c[self.OC:self.OC + T] = bspec.c_deg_per_kwh
        c[self.OD:self.OD + T] = bspec.c_deg_per_kwh
        if curtail_cost:
            c[self.OQ:self.OQ + T] = curtail_cost

        power_kwh = bspec.power_kw * interval_hours
        soc_min = (bspec.backup_reserve_pct / 100.0) * bspec.capacity_kwh
        lo = np.zeros(N)
        hi = np.full(N, inf)
        hi[self.OC:self.OC + T] = power_kwh
        hi[self.OD:self.OD + T] = power_kwh
        hi[self.OE:self.OE + T] = (inf if not np.isfinite(export_limit)
                                   else export_limit)
        lo[self.OS:self.OS + T] = soc_min
        hi[self.OS:self.OS + T] = bspec.capacity_kwh
        hi[self.OQ:self.OQ + T] = 0.0 if self._curtail_pinned else inf

        A = self._build_matrix(T, bspec.charge_efficiency,
                               bspec.discharge_efficiency)

        lp = highspy.HighsLp()
        lp.num_col_ = N
        lp.num_row_ = 2 * T
        lp.col_cost_ = c
        lp.col_lower_ = lo
        lp.col_upper_ = hi
        lp.row_lower_ = np.zeros(2 * T)
        lp.row_upper_ = np.zeros(2 * T)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr.astype(np.int32)
        lp.a_matrix_.index_ = A.indices.astype(np.int32)
        lp.a_matrix_.value_ = A.data

        self._h = highspy.Highs()
        self._h.setOptionValue('output_flag', False)
        self._h.passModel(lp)

        # Index vectors reused every call, so the hot path allocates nothing.
        self._price_cols = np.arange(self.OI, self.OI + 2 * T, dtype=np.int32)
        self._curtail_cols = np.arange(self.OQ, self.OQ + T, dtype=np.int32)
        self._soc_last_col = np.array([self.OS + T - 1], dtype=np.int32)
        self._fb_rows = np.arange(T + 1, dtype=np.int32)   # flow balance + soc[0]
        self._soc_lo_const = soc_min
        self._soc_hi_const = bspec.capacity_kwh
        self._inf = inf

    @staticmethod
    def _build_matrix(T: int, eta_c: float, eta_d: float):
        """A_eq, identical to perfect_foresight_lp._build_lp.

        Depends only on T, eta_c and eta_d, which is precisely why it can be
        built once and reused for every window of every site in a scenario.
        """
        import scipy.sparse as sp
        t = np.arange(T, dtype=np.intp)
        t1 = np.arange(1, T, dtype=np.intp)
        OC, OD, OI, OE, OS, OQ = 0, T, 2 * T, 3 * T, 4 * T, 5 * T

        fb_rows = np.tile(t, 5)
        fb_cols = np.concatenate([OC + t, OD + t, OI + t, OE + t, OQ + t])
        fb_data = np.concatenate([-np.ones(T), np.ones(T), np.ones(T),
                                  -np.ones(T), -np.ones(T)])
        sc_rows = np.concatenate([T + t, T + t, T + t])
        sc_cols = np.concatenate([OC + t, OD + t, OS + t])
        sc_data = np.concatenate([-np.full(T, eta_c),
                                  np.full(T, 1.0 / eta_d),
                                  np.ones(T)])
        if T > 1:
            sc_rows = np.concatenate([sc_rows, T + t1])
            sc_cols = np.concatenate([sc_cols, OS + t1 - 1])
            sc_data = np.concatenate([sc_data, -np.ones(T - 1)])

        return sp.csc_matrix(
            (np.concatenate([fb_data, sc_data]),
             (np.concatenate([fb_rows, sc_rows]),
              np.concatenate([fb_cols, sc_cols]))),
            shape=(2 * T, 6 * T),
        )

    def solve(self, solar, load, p_imp, p_exp, soc_init, soc_final):
        """Solve one window. Returns the primal vector, or None if infeasible."""
        T = self.T
        h = self._h

        cost = np.empty(2 * T)
        cost[:T] = p_imp
        cost[T:] = -p_exp
        h.changeColsCost(2 * T, self._price_cols, cost)

        if not self._curtail_pinned:
            h.changeColsBounds(T, self._curtail_cols,
                               np.zeros(T), np.maximum(0.0, solar))

        h.changeColsBounds(1, self._soc_last_col,
                           np.array([soc_final]), np.array([soc_final]))

        rhs = np.empty(T + 1)
        rhs[:T] = load - solar
        rhs[T] = soc_init
        h.changeRowsBounds(T + 1, self._fb_rows, rhs, rhs)

        h.run()
        status = h.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            return None
        return np.asarray(h.getSolution().col_value, dtype=float)

    def restore_soc_bounds(self):
        """Undo the terminal pin on soc[T-1] before reusing the model.

        The pin is applied by tightening bounds on a single column. Without
        restoring them, a later window would inherit the previous window's
        target as a hard constraint, which is exactly the kind of state leak
        that reusing a solver invites.
        """
        self._h.changeColsBounds(
            1, self._soc_last_col,
            np.array([self._soc_lo_const]), np.array([self._soc_hi_const]))


class WindowPlanner(Protocol):
    """What the rolling loop needs from a plan in order to solve one window.

    The loop owns window geometry, the forecast providers, causality and
    staleness accounting, the terminal SOC pin, the self-consumption fallback
    and retention. Only three things differ between plan types: how the
    window's price vectors are built, what the window LP looks like, and
    whether any state has to be carried from one committed interval to the
    next. Those three are this protocol.

    Keeping one loop rather than a second, largely-copied component matters
    because the property a copy would drift on is causality.
    """

    @property
    def requires_price_forecast(self) -> bool:
        """False when nothing in the price vectors reads the wholesale price.

        The loop then skips the price provider entirely, so a plan whose
        prices are fixed by the clock does not fall back to self-consumption
        over a missing forecast that could not have changed its answer.
        """
        ...

    def window_prices(self, rrp_mwh, k: int, n: int):
        """(p_imp, p_exp) in $/kWh for the window starting at interval k."""
        ...

    def solve_window(self, k: int, n: int, solar, load, p_imp, p_exp,
                     soc_init: float, soc_final: float):
        """Primal vector for the window, or None if infeasible.

        The vector starts with the standard 6n block, so the loop extracts
        charge, discharge, import, export, SOC and curtailment at the usual
        offsets without knowing which planner ran.
        """
        ...

    def observe_commit(self, k: int, import_kwh: float,
                       export_kwh: float) -> None:
        """Record what was actually committed in interval k."""
        ...

    def plan_diagnostics(self) -> dict | None:
        """Plan-specific counters to fold into RollingDiagnostics, or None.

        Read once at the end of a run and never mutating, so it can also be
        inspected mid-run by a check script.
        """
        ...


class _WholesalePlanner:
    """The default planner: a wholesale-exposed price stack and the standard LP.

    Reproduces exactly what this component did before planners existed. It has
    no day state, so observe_commit does nothing, and its prices always depend
    on the wholesale price, so it always requires a forecast.
    """

    def __init__(self, price_offsets: PriceStackOffsets,
                 battery_spec: BatterySpec, export_limit: float,
                 interval_hours: float, curtail_cost: float | None):
        self._offsets = price_offsets
        self._bspec = battery_spec
        self._export_limit = export_limit
        self._interval_hours = interval_hours
        self._curtail_cost = curtail_cost
        self._solver: _WindowSolver | None = None
        self._solver_T = -1

    @property
    def requires_price_forecast(self) -> bool:
        return True

    def window_prices(self, rrp_mwh, k: int, n: int):
        return self._offsets.window_price_vectors(rrp_mwh[:n], start=k)

    def solve_window(self, k: int, n: int, solar, load, p_imp, p_exp,
                     soc_init: float, soc_final: float):
        if self._solver is None or self._solver_T != n:
            self._solver = _WindowSolver(
                n, self._bspec, self._export_limit, self._interval_hours,
                self._curtail_cost)
            self._solver_T = n
        x = self._solver.solve(solar, load, p_imp, p_exp, soc_init, soc_final)
        self._solver.restore_soc_bounds()
        return x

    def observe_commit(self, k: int, import_kwh: float,
                       export_kwh: float) -> None:
        return None

    def plan_diagnostics(self) -> dict | None:
        return None


# ---------------------------------------------------------------------------
# Component
# ---------------------------------------------------------------------------

def rolling_optimisation(
    site_id: str,
    solar_kwh: pd.Series,
    load_kwh: pd.Series,
    forecast_provider: ForecastProvider | None,
    price_offsets: PriceStackOffsets | None,
    battery_spec: BatterySpec,
    soc_init_kwh: float,
    soc_final_kwh: float,
    config: RollingConfig,
    export_limit_kwh_per_interval: float = np.inf,
    interval_hours: float = 5 / 60,
    curtail_cost_per_kwh_dollars: float | None = 0.0,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = 2000,
    load_forecast_provider: LoadForecastProvider | None = None,
    solar_forecast_provider: SolarForecastProvider | None = None,
    commit_actual_load: bool = True,
    planner: WindowPlanner | None = None,
) -> tuple[SiteTraces, RollingDiagnostics]:
    """Dispatch a battery under imperfect price foresight.

    Args:
        site_id: Passed through to the output SiteTraces.
        solar_kwh, load_kwh: Gross per-interval energy, kWh, sharing one index.
        forecast_provider: Answers "what was the forecast at this instant".
            Owns causality, splicing, carry-forward, infill and the
            current-interval price mode.
        price_offsets: Affine price stack for the period, from
            prices.build_price_stack_offsets. Must cover the same index.
        battery_spec: Battery specification, including c_deg_per_kwh.
        soc_init_kwh: DC SOC at the start of the period.
        soc_final_kwh: DC SOC required at the end of the period. Set to the
            self-consumption end-of-period SOC so the bill comparison against
            self_consumption and perfect_foresight_lp is fair.
        config: RollingConfig. horizon_hours drives the window length; the
            current-interval price mode is a property of the provider and is
            not read here.
        export_limit_kwh_per_interval: Per-interval export cap, kWh.
        interval_hours: Interval length in hours.
        curtail_cost_per_kwh_dollars: As perfect_foresight_lp. None pins
            curtailment to zero, which carries a real infeasibility risk; a
            window that cannot be solved falls back to self-consumption for
            that interval and is counted.
        progress: Optional callback(intervals_done, intervals_total), fired
            every progress_every intervals. A site-year is minutes of silent
            work and a sweep is many of those, so callers need a way to see
            that something is happening. One integer comparison per interval
            is unmeasurable against a millisecond LP solve, so this is always
            on when a callback is supplied and costs nothing when it is not.
        progress_every: Intervals between callbacks.
        load_forecast_provider: Optional household load forecast. None means
            load is known perfectly over the horizon, which is the original
            scope of this component and a legitimate rung of the measurement
            ladder rather than a degenerate case. Appended at the end of the
            signature rather than placed beside forecast_provider so that no
            existing positional call site shifts.
        solar_forecast_provider: Optional generation forecast. None means
            solar is known perfectly, the original scope of this component.
            Every method behind it is naive relative to an operator
            forecasting from satellite imagery, so a run using it bounds the
            cost of solar forecast error from above; see
            solar_forecast_spec.md.
        commit_actual_load: When True the committed interval is solved against
            ACTUAL load and solar, and only positions 1 onward use forecasts,
            so the site forecasts affect the plan rather than the execution.
            The flag governs BOTH site inputs -- it names what the committed
            interval is solved against, and treating load and solar
            differently there would make the committed interval a mixture of
            two execution models. See usage_forecast_spec.md, "What the
            component commits". Ignored when neither provider is supplied.

        planner: Window planner. None builds the wholesale planner from
            price_offsets, which is what every existing caller wants and
            reproduces this component's previous behaviour exactly. An
            advanced TOU plan passes AdvancedWindowPlanner instead.

    Returns:
        (SiteTraces, RollingDiagnostics).

    Raises:
        ValueError: on misaligned inputs or an out-of-range initial SOC.
    """
    index = solar_kwh.index
    if not load_kwh.index.equals(index):
        raise ValueError("solar_kwh and load_kwh must share the same index")
    if price_offsets is not None and len(price_offsets) != len(index):
        raise ValueError(
            f"price_offsets covers {len(price_offsets)} intervals but the "
            f"traces cover {len(index)}"
        )
    if planner is None and price_offsets is None:
        raise ValueError(
            "rolling_optimisation needs either price_offsets (to build the "
            "default wholesale planner) or an explicit planner"
        )
    solar = solar_kwh.to_numpy(dtype=float, copy=False)
    load = load_kwh.to_numpy(dtype=float, copy=False)
    if np.isnan(solar).any() or np.isnan(load).any():
        raise ValueError("solar_kwh and load_kwh must not contain NaN")

    soc_min = (battery_spec.backup_reserve_pct / 100.0) * battery_spec.capacity_kwh
    for name, v in (('soc_init_kwh', soc_init_kwh), ('soc_final_kwh', soc_final_kwh)):
        if not (soc_min - 1e-9 <= v <= battery_spec.capacity_kwh + 1e-9):
            raise ValueError(
                f"{name}={v} outside battery bounds "
                f"[{soc_min}, {battery_spec.capacity_kwh}]"
            )

    T_p = len(index)
    H = max(1, int(round(config.horizon_hours / interval_hours)))

    # Opt-in, read once: a planner that does not declare it gets the pin, so
    # every planner written before this property behaves exactly as before.
    needs_terminal_soc = bool(getattr(planner, 'needs_terminal_soc', True)) \
        if planner is not None else True
    interval = pd.Timedelta(hours=interval_hours)

    retain_from, retain_to = _retention_bounds(config, index.tz)

    charge_c = np.zeros(T_p)
    discharge_c = np.zeros(T_p)
    curtail_c = np.zeros(T_p)
    soc_c = np.zeros(T_p)

    if planner is None:
        planner = _WholesalePlanner(
            price_offsets, battery_spec, export_limit_kwh_per_interval,
            interval_hours, curtail_cost_per_kwh_dollars)
    needs_prices = planner.requires_price_forecast
    if needs_prices and forecast_provider is None:
        raise ValueError(
            "rolling_optimisation: this plan's prices depend on the wholesale "
            "price, so a forecast_provider is required"
        )

    n_solved = 0
    fb_no_forecast = 0
    fb_no_load_forecast = 0
    fb_infeasible = 0
    fb_infeasible_from_load = 0
    load_warm_up = 0
    load_clipped = 0
    fb_no_solar_forecast = 0
    solar_warm_up = 0
    solar_clipped = 0
    fb_stamps: list[pd.Timestamp] = []
    infill_fracs: list[float] = []
    infill_hours: list[int] = []
    staleness: list[float] = []
    coarse_staleness: list[float] = []
    solve_times: list[float] = []
    max_lead_minutes = -np.inf
    causality_ok = True
    retained: list[RetainedWindow] = []

    soc_prev = float(soc_init_kwh)

    for k in range(T_p):
        if progress is not None and k % progress_every == 0:
            progress(k, T_p)
        decision_instant = index[k] - interval
        k_end = min(k + H - 1, T_p - 1)
        n = k_end - k + 1
        retaining = (retain_from is not None
                     and retain_from <= decision_instant < retain_to)

        slice_ = (forecast_provider.get(decision_instant, n)
                  if needs_prices else None)
        if needs_prices and slice_ is None:
            soc_prev = _commit_sc(
                k, solar, load, battery_spec, soc_prev, interval_hours,
                charge_c, discharge_c, curtail_c, soc_c, planner)
            fb_no_forecast += 1
            if len(fb_stamps) < _MAX_FALLBACK_TIMESTAMPS:
                fb_stamps.append(index[k])
            if retaining:
                # No forecast existed, so the forecast arrays are empty rather
                # than absent. The row exists so the retained sequence stays
                # contiguous.
                retained.append(RetainedWindow(
                    decision_instant=decision_instant,
                    **_retained_forecast_fields(None, n, decision_instant),
                    plan_charge=None, plan_discharge=None, plan_soc=None,
                    plan_import=None, plan_export=None, plan_curtail=None,
                    committed_soc=soc_prev, solved=False,
                ))
            continue

        if slice_ is not None:
            n = len(slice_)                  # provider may shorten the window
            k_end = k + n - 1

        # Window load. Obtained through a single point rather than sliced
        # inline at each use, so that the perfect-load and forecast-load paths
        # cannot drift apart: the terminal SOC pin and the LP must see the
        # SAME vector or the pin is computed against information the plan does
        # not have.
        load_window = load[k:k_end + 1]
        load_slice = None
        if load_forecast_provider is not None:
            load_slice = load_forecast_provider.get(k, n)
            if load_slice is None:
                # No usable history. Self-consumption is what a battery does
                # with no plannable signal, matching the no-price-forecast
                # case. With min_history_days=1 this reaches only the first
                # local day of a period.
                soc_prev = _commit_sc(
                    k, solar, load, battery_spec, soc_prev, interval_hours,
                    charge_c, discharge_c, curtail_c, soc_c, planner)
                fb_no_load_forecast += 1
                if len(fb_stamps) < _MAX_FALLBACK_TIMESTAMPS:
                    fb_stamps.append(index[k])
                continue
            if len(load_slice) < n:
                n = len(load_slice)
                k_end = k + n - 1
            load_window = load_slice.load_kwh[:n].copy()
            if commit_actual_load:
                load_window[0] = load[k]
            load_warm_up += int(load_slice.warm_up)
            load_clipped += int(load_slice.n_clipped)

        # Window solar, obtained the same way and for the same reason: the
        # terminal SOC pin and the LP must see one vector, or the pin is
        # computed against information the plan does not have.
        solar_window = solar[k:k_end + 1]
        solar_slice = None
        if solar_forecast_provider is not None:
            solar_slice = solar_forecast_provider.get(k, n)
            if solar_slice is None:
                soc_prev = _commit_sc(
                    k, solar, load, battery_spec, soc_prev, interval_hours,
                    charge_c, discharge_c, curtail_c, soc_c, planner)
                fb_no_solar_forecast += 1
                if len(fb_stamps) < _MAX_FALLBACK_TIMESTAMPS:
                    fb_stamps.append(index[k])
                continue
            if len(solar_slice) < n:
                n = len(solar_slice)
                k_end = k + n - 1
                load_window = load_window[:n]
            solar_window = solar_slice.solar_kwh[:n].copy()
            if commit_actual_load:
                solar_window[0] = solar[k]
            solar_warm_up += int(solar_slice.warm_up)
            solar_clipped += int(solar_slice.n_clipped)

        if slice_ is not None:
            lead = (slice_.run_time - decision_instant).total_seconds() / 60.0
            max_lead_minutes = max(max_lead_minutes, lead)
            if lead > interval_hours * 60 + 1e-9:
                causality_ok = False
            staleness.append(max(0.0, -lead))
            coarse_lead = (slice_.coarse_run_time
                           - decision_instant).total_seconds() / 60.0
            coarse_staleness.append(max(0.0, -coarse_lead))
            infill_fracs.append(slice_.infill_fraction)
            infill_hours.append(index[k].hour)

        p_imp, p_exp = planner.window_prices(
            None if slice_ is None else slice_.rrp_mwh, k, n)

        # Terminal SOC. A window reaching the period end pins the global
        # self-consumption target, so the rolling, perfect-foresight and
        # self-consumption dispatches all finish at the same state.
        #
        # A planner may declare that it never reads the pin, in which case the
        # self-consumption forward run that produces it is skipped. It is a
        # full-window run at every interval -- roughly 30 s per site-year at an
        # 18-hour horizon -- and on the VPP path nothing consumes it: the
        # operator LP deliberately has no terminal constraint, and
        # self-consumption between events is deterministic. Absent the
        # property the pin is computed as before, so no existing planner
        # changes behaviour.
        if not needs_terminal_soc:
            soc_target = float(soc_prev)
        elif k_end == T_p - 1:
            soc_target = float(soc_final_kwh)
        else:
            soc_fwd, _ = run_sc_with_battery(
                solar_window, load_window,
                battery_spec, soc_prev, interval_hours)
            soc_target = float(soc_fwd[-1])

        t0 = time.perf_counter()
        x = planner.solve_window(k, n, solar_window, load_window,
                                 p_imp, p_exp, soc_prev, soc_target)
        solve_times.append(time.perf_counter() - t0)

        if x is None:
            soc_prev = _commit_sc(
                k, solar, load, battery_spec, soc_prev, interval_hours,
                charge_c, discharge_c, curtail_c, soc_c, planner)
            fb_infeasible += 1
            # Attribution. With curtailment pinned to zero, a site forecast
            # that under-predicts load or over-predicts solar leaves the LP
            # surplus generation it cannot place, so either forecast makes
            # infeasibility strictly more likely. Re-solving the same window
            # with ACTUAL load and solar says whether THIS failure is one of
            # those. The solution is discarded: using it would be the very
            # leak the forecasts exist to model.
            if load_slice is not None or solar_slice is not None:
                if planner.solve_window(
                        k, n, solar[k:k_end + 1], load[k:k_end + 1],
                        p_imp, p_exp, soc_prev, soc_target) is not None:
                    fb_infeasible_from_load += 1
            if len(fb_stamps) < _MAX_FALLBACK_TIMESTAMPS:
                fb_stamps.append(index[k])
            if retaining:
                # A forecast existed and the window was infeasible, so the
                # forecast is retained and only the plan is missing.
                retained.append(RetainedWindow(
                    decision_instant=decision_instant,
                    **_retained_forecast_fields(slice_, n, decision_instant),
                    plan_charge=None, plan_discharge=None, plan_soc=None,
                    plan_import=None, plan_export=None, plan_curtail=None,
                    committed_soc=soc_prev, solved=False,
                    load_forecast_kwh=(None if load_slice is None
                                       else load_window.copy()),
                    solar_forecast_kwh=(None if solar_slice is None
                                        else solar_window.copy()),
                ))
            continue

        n_solved += 1
        OC, OD, OI, OE, OS, OQ = 0, n, 2 * n, 3 * n, 4 * n, 5 * n
        charge_c[k] = x[OC]
        discharge_c[k] = x[OD]
        soc_prev = _clamp_soc_residue(float(x[OS]), battery_spec, site_id, k)
        soc_c[k] = soc_prev
        curtail_c[k] = x[OQ]
        # The LP's own import and export for the committed interval, not a
        # reconstruction: these are the quantities the linking constraints and
        # the reward test are written against.
        planner.observe_commit(k, float(x[OI]), float(x[OE]))

        if retaining:
            retained.append(RetainedWindow(
                decision_instant=decision_instant,
                **_retained_forecast_fields(slice_, n, decision_instant),
                plan_charge=x[OC:OC + n].copy(),
                plan_discharge=x[OD:OD + n].copy(),
                plan_soc=x[OS:OS + n].copy(),
                plan_import=x[OI:OI + n].copy(),
                plan_export=x[OE:OE + n].copy(),
                plan_curtail=x[OQ:OQ + n].copy(),
                committed_soc=soc_prev,
                solved=True,
                load_forecast_kwh=(None if load_slice is None
                                   else load_window.copy()),
                solar_forecast_kwh=(None if solar_slice is None
                                    else solar_window.copy()),
            ))

    if progress is not None:
        progress(T_p, T_p)

    charge_c = _clip_nonneg(charge_c, 'charge', site_id)
    discharge_c = _clip_nonneg(discharge_c, 'discharge', site_id)
    curtail_c = _clip_nonneg(curtail_c, 'curtail', site_id)
    charge_c, discharge_c = _net_simultaneous_charge_discharge(charge_c, discharge_c)

    traces = _reconstruct_flows(site_id, index, solar, load,
                                charge_c, discharge_c, soc_c, curtail_c)

    if fb_infeasible:
        warnings.warn(
            f"rolling_optimisation [{site_id}]: {fb_infeasible} window(s) were "
            "infeasible and fell back to self-consumption. With curtailment "
            "pinned this is expected occasionally; a large count means the "
            "run should be treated as suspect.",
            stacklevel=2,
        )

    diagnostics = RollingDiagnostics(
        site_id=site_id,
        n_intervals=T_p,
        n_windows_solved=n_solved,
        sc_fallback_no_forecast=fb_no_forecast,
        sc_fallback_infeasible=fb_infeasible,
        load_forecast_used=load_forecast_provider is not None,
        sc_fallback_no_load_forecast=fb_no_load_forecast,
        load_windows_warm_up=load_warm_up,
        load_positions_clipped=load_clipped,
        infeasible_from_site_forecast=fb_infeasible_from_load,
        solar_forecast_used=solar_forecast_provider is not None,
        sc_fallback_no_solar_forecast=fb_no_solar_forecast,
        solar_windows_warm_up=solar_warm_up,
        solar_positions_clipped=solar_clipped,
        fallback_timestamps=tuple(fb_stamps),
        infill_fraction_mean=float(np.mean(infill_fracs)) if infill_fracs else 0.0,
        infill_fraction_by_hour=_by_hour(infill_hours, infill_fracs),
        staleness_minutes_mean=float(np.mean(staleness)) if staleness else 0.0,
        staleness_minutes_max=float(np.max(staleness)) if staleness else 0.0,
        solve_seconds_total=float(np.sum(solve_times)) if solve_times else 0.0,
        solve_ms_median=float(np.median(solve_times) * 1e3) if solve_times else 0.0,
        solve_ms_p95=float(np.percentile(solve_times, 95) * 1e3) if solve_times else 0.0,
        terminal_soc_kwh=float(soc_c[-1]) if T_p else float('nan'),
        terminal_soc_target_kwh=float(soc_final_kwh),
        max_run_time_lead_minutes=(float(max_lead_minutes)
                                   if np.isfinite(max_lead_minutes) else 0.0),
        causality_ok=causality_ok,
        retained=tuple(retained),
        coarse_staleness_minutes_mean=(float(np.mean(coarse_staleness))
                                       if coarse_staleness else 0.0),
        coarse_staleness_minutes_max=(float(np.max(coarse_staleness))
                                      if coarse_staleness else 0.0),
        **(planner.plan_diagnostics() or {}),
    )
    return traces, diagnostics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retained_forecast_fields(slice_, n, decision_instant) -> dict:
    """Forecast columns of a RetainedWindow.

    A plan whose prices do not depend on the wholesale price runs with no
    price forecast at all, so the forecast columns are empty rather than
    absent -- the same shape the no-forecast fallback already uses, so a
    reader cannot mistake "not needed" for "missing".
    """
    if slice_ is None:
        return dict(run_time=decision_instant,
                    coarse_run_time=decision_instant,
                    rrp_mwh=np.empty(0),
                    is_infill=np.empty(0, dtype=bool),
                    is_fine=np.empty(0, dtype=bool))
    return dict(run_time=slice_.run_time,
                coarse_run_time=slice_.coarse_run_time,
                rrp_mwh=slice_.rrp_mwh[:n].copy(),
                is_infill=slice_.is_infill[:n].copy(),
                is_fine=slice_.is_fine[:n].copy())


# HiGHS returns a primal solution to its feasibility tolerance (1e-7), so a
# committed SOC at the backup reserve can come back fractionally below it --
# 4.999999977 against a 5.0 kWh floor was seen on a real site-year. The
# committed SOC seeds the next window's soc_init and anchors the VPP shadow,
# both of which reject values outside the battery bounds, so the residue is
# clamped here, once, at the point it enters the committed trajectory.
# Anything beyond the solver tolerance is a real bound problem and raises.
_SOC_RESIDUE_TOL = 1e-7


def _clamp_soc_residue(soc: float, bspec: BatterySpec, site_id, k: int) -> float:
    """Clamp a committed LP SOC into [soc_min, capacity] within solver tolerance."""
    soc_min = (bspec.backup_reserve_pct / 100.0) * bspec.capacity_kwh
    cap = bspec.capacity_kwh
    if soc < soc_min - _SOC_RESIDUE_TOL or soc > cap + _SOC_RESIDUE_TOL:
        raise ValueError(
            f"rolling_optimisation [{site_id}] k={k}: committed SOC {soc!r} "
            f"outside [{soc_min}, {cap}] by more than the "
            f"{_SOC_RESIDUE_TOL:.0e} solver tolerance -- check the LP bounds."
        )
    return min(max(soc, soc_min), cap)


def _commit_sc(k, solar, load, bspec, soc_prev, interval_hours,
               charge_c, discharge_c, curtail_c, soc_c,
               planner=None) -> float:
    """Commit one interval of self-consumption dispatch, and return the new SOC.

    The single fallback for every degraded case. Self-consumption is what the
    battery does when no useful price signal is available, so a fallback
    interval is a modelling statement rather than a gap to be patched.
    """
    soc_arr, flows = run_sc_with_battery(
        solar[k:k + 1], load[k:k + 1], bspec, soc_prev, interval_hours)
    charge_c[k] = flows['sol_to_bat'][0] + flows['grid_to_bat'][0]
    discharge_c[k] = flows['bat_to_home'][0]
    curtail_c[k] = flows['curtail'][0]
    soc_c[k] = soc_arr[0]
    if planner is not None:
        # A fallback is a statement about what the battery did, not a gap, so
        # its flows count against block tiers and the zero-import reward like
        # any others. Skipping them would let the day state drift away from
        # what actually happened.
        planner.observe_commit(
            k,
            float(flows['grid_to_home'][0] + flows['grid_to_bat'][0]),
            float(flows['sol_to_grid'][0]))
    return float(soc_arr[0])


def _retention_bounds(config: RollingConfig, tz):
    """Resolve the retention window to a half-open interval of instants.

    The config fields are DATES and both ends are INCLUSIVE, so
    from='2026-01-05', to='2026-01-28' retains all 24 days.

    This is worth stating because the obvious implementation gets it wrong.
    Parsing 'to' straight to a Timestamp gives midnight at the START of the
    28th, so an inclusive comparison against it retains 23 days and a single
    interval. Nothing errors, the artefact looks complete, and the missing day
    is the last one, which is exactly where a reader is least likely to check.
    Returning a half-open bound at the following midnight is what makes the
    inclusive reading true.

    A value carrying a time component is rejected rather than reinterpreted.
    Silently extending '2026-01-28 12:00' to the end of the day would be
    guessing at intent, and silently honouring it would reintroduce the short
    window this function exists to prevent.

    Returns (lo, hi) to be used as lo <= decision_instant < hi, or
    (None, None) when retention is off.
    """
    if config.retain_plans_from is None:
        return None, None
    for name in ('retain_plans_from', 'retain_plans_to'):
        raw = getattr(config, name)
        ts = pd.Timestamp(raw)
        if ts != ts.normalize():
            raise ValueError(
                f"{name}={raw!r} carries a time of day. Retention bounds are "
                "whole dates and both ends are inclusive, e.g. "
                "'2026-01-28'. Pass a date."
            )
    lo = pd.Timestamp(config.retain_plans_from, tz=tz)
    hi = pd.Timestamp(config.retain_plans_to, tz=tz)
    if hi < lo:
        raise ValueError(
            f"retain_plans_to={config.retain_plans_to!r} precedes "
            f"retain_plans_from={config.retain_plans_from!r}"
        )
    # The model index is market time (a fixed offset, no DST), so adding a
    # calendar day to a localised midnight is unambiguous here.
    return lo, hi + pd.Timedelta(days=1)


def _by_hour(hours, values) -> dict[int, float]:
    """Mean of `values` grouped by hour of day.

    The infill fraction has a strong daily rhythm, because the pre-dispatch
    horizon sawtooths between roughly 15 and 39 hours. Reporting it by hour is
    what stops that publication-schedule artefact being attributed to battery
    behaviour in any time-of-day analysis.
    """
    if not values:
        return {}
    out: dict[int, list[float]] = {}
    for h, v in zip(hours, values):
        out.setdefault(int(h), []).append(v)
    return {h: float(np.mean(v)) for h, v in sorted(out.items())}