"""Window solver for the rolling advanced (TOU) path.

See rolling_perfect_foresight_lp_advanced_spec.md. This module holds the
solver only; the planner that precomputes period data, builds window price
vectors and carries the day state is built on top of it.

Why this exists rather than calling perfect_foresight_lp_advanced per window
--------------------------------------------------------------------------
That function does period-level work on every call -- input validation on
pandas Series, the zero-import penalty, the day map, and a block_window_mask
per block window, several of which convert to local time independently. At
roughly 105,000 windows per site-year it would repeat work whose answer is
fixed for the whole period.

The subproblem itself is unchanged: the standard 6T core from
perfect_foresight_lp._build_lp, plus one tier variable per (block window, day
slot, tier) and one linking row per (block window, day slot), exactly as
_build_block_tier_data builds them for a whole period.

Why the model is kept and mutated
---------------------------------
Unlike the wholesale path, A_eq is not invariant: the linking rows carry a -1
for each interval inside a block window, and which intervals those are moves
as the window slides. But it moves in a restricted way. Window position j maps
to absolute interval k + j, and a block window is a contiguous run in absolute
time, so its image in window coordinates is contiguous too and advancing k by
one shifts that run left by exactly one position. Only the two ends move.

On a plan with critical rates, positions forecast to be critical on a window's
side are left out of its linking row (critical flow does not count towards a
block tier). Membership is then the run minus those positions, and a forecast
that moves a spike adds or removes a few more coefficients.

So the model is built once per site and mutated: costs, column bounds, row
bounds and tier bounds as on the wholesale path, plus the symmetric difference
of each linking row's membership per step.

Day slots
---------
A tier variable belongs to a local calendar date inside the window, indexed by
a slot where slot 0 is the date of the window's first interval. The slot count
is a per-site constant because the model's column count cannot change between
windows; the planner computes it from the day map rather than deriving it from
the horizon, since local days are 23 or 25 hours long across daylight saving.

A window that touches fewer dates than there are slots leaves the surplus
slots with an empty linking row. That forces the sum of their tier variables
to zero, and since the variables are non-negative each is pinned to zero, so
the surplus costs nothing and needs no special case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

import highspy

from model.core.localtime import local_wall_clock
from model.core.scenario import BatterySpec
from model.core.prices import (
    C_TO_DOLLARS, apply_tou_export_order_nudge, block_window_mask,
    build_base_price_vectors_advanced, build_mixed_dispatch_stack,
    build_tou_export_order_nudge, build_zero_import_penalty,
    check_export_order_nudge_degradation, critical_masks,
    critical_rates_dollars,
)
from model.core.tariff import RetailTariff
from model.components.perfect_foresight_lp_advanced import _build_day_map
from model.components.rolling_optimisation import _WindowSolver


@dataclass(frozen=True)
class BlockWindowSpec:
    """One block window's tier structure, lifted out of the tariff.

    capacities is per tier in kWh, the width of that tier rather than its
    cumulative threshold, with inf for the final tier. costs is per tier in
    $/kWh with the objective sign already applied: positive for an import
    window (a cost), negative for an export window (a credit). Both mirror
    _build_block_tier_data so the two builders cannot disagree about order.
    """
    is_import: bool
    capacities: tuple[float, ...]
    costs: tuple[float, ...]

    @property
    def n_tiers(self) -> int:
        return len(self.capacities)


def block_window_specs(retail_tariff: RetailTariff) -> tuple[BlockWindowSpec, ...]:
    """Block window specs in _build_block_tier_data's canonical order.

    Import windows first, then export windows, each in tariff order. Every
    index used elsewhere in this module -- tier variable positions, linking
    row positions, the planner's per-window accumulators -- is an index into
    this tuple.
    """
    specs: list[BlockWindowSpec] = []
    for is_import, windows in ((True, retail_tariff.block_import_tiers or ()),
                               (False, retail_tariff.block_export_tiers or ())):
        for window in windows:
            caps, costs, prev = [], [], 0.0
            for tier in window.tiers:
                if np.isinf(tier.threshold_kwh):
                    caps.append(np.inf)
                else:
                    caps.append(tier.threshold_kwh - prev)
                    prev = tier.threshold_kwh
                cost = tier.rate_c_per_kwh / C_TO_DOLLARS
                costs.append(cost if is_import else -cost)
            specs.append(BlockWindowSpec(is_import, tuple(caps), tuple(costs)))
    return tuple(specs)


def tier_upper_bounds(
    specs: tuple[BlockWindowSpec, ...],
    n_slots: int,
    day0_used: dict[int, float] | None = None,
) -> np.ndarray:
    """Upper bounds for every tier variable, slot 0 reduced by carry-in.

    day0_used maps block window index (into `specs`) to kWh already flowed
    through that window earlier on the window's first local day. This is the
    same reduction _build_block_tier_data applies for
    first_day_block_import_used / first_day_block_export_used, kept here as
    one implementation so the rolling path and the whole-period path cannot
    drift on the spill rule from one tier into the next.
    """
    used = day0_used or {}
    hi: list[float] = []
    for w_idx, spec in enumerate(specs):
        for slot in range(n_slots):
            remaining = float(used.get(w_idx, 0.0)) if slot == 0 else 0.0
            for cap in spec.capacities:
                if remaining > 0 and not np.isinf(cap):
                    hi.append(max(0.0, cap - remaining))
                    remaining = max(0.0, remaining - cap)
                else:
                    hi.append(cap)
    return np.array(hi, dtype=float)


class AdvancedWindowSolver:
    """A HiGHS model for a fixed window length T and slot count, mutated
    between windows.

    Membership is passed per (block window, day slot) as either one
    half-open run of window positions (lo, hi), or an array of window
    positions (a run with forecast-critical positions removed). None means
    that pair has no intervals in this window, which happens for surplus
    slots and for weekday-only windows on a weekend.
    """

    def __init__(self, T: int, bspec: BatterySpec, export_limit: float,
                 interval_hours: float, curtail_cost: float | None,
                 specs: tuple[BlockWindowSpec, ...], n_slots: int):
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}")
        self.T = T
        self.specs = specs
        self.n_slots = n_slots
        self._curtail_pinned = curtail_cost is None
        self.OC, self.OD, self.OI, self.OE, self.OS, self.OQ = (
            0, T, 2 * T, 3 * T, 4 * T, 5 * T)
        N_CORE = 6 * T
        self.N_CORE = N_CORE
        inf = highspy.kHighsInf

        # Tier variables and linking rows, in block_window_specs order.
        # tier_col[(w, slot)] is the column of that pair's FIRST tier variable;
        # link_row[(w, slot)] is its linking row, offset past the 2T core rows.
        self._tier_col: dict[tuple[int, int], int] = {}
        self._link_row: dict[tuple[int, int], int] = {}
        tier_costs: list[float] = []
        col = N_CORE
        row = 2 * T
        for w_idx, spec in enumerate(specs):
            for slot in range(n_slots):
                self._tier_col[(w_idx, slot)] = col
                self._link_row[(w_idx, slot)] = row
                tier_costs.extend(spec.costs)
                col += spec.n_tiers
                row += 1
        self.n_tier_vars = col - N_CORE
        self.n_linking = row - 2 * T
        N = N_CORE + self.n_tier_vars
        n_rows = 2 * T + self.n_linking

        c = np.zeros(N)
        c[self.OC:self.OC + T] = bspec.c_deg_per_kwh
        c[self.OD:self.OD + T] = bspec.c_deg_per_kwh
        if curtail_cost:
            c[self.OQ:self.OQ + T] = curtail_cost
        if self.n_tier_vars:
            c[N_CORE:] = np.array(tier_costs, dtype=float)

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
        # Tier bounds are set per window; start them at their uncarried values.
        if self.n_tier_vars:
            hi[N_CORE:] = tier_upper_bounds(specs, n_slots)

        core = _WindowSolver._build_matrix(
            T, bspec.charge_efficiency, bspec.discharge_efficiency)
        if self.n_tier_vars:
            # Linking rows start with only their +1 tier entries. The -1
            # entries on imp/exp columns are installed by the first
            # set_membership call and moved incrementally after that.
            rows, cols = [], []
            for (w_idx, slot), r in self._link_row.items():
                base = self._tier_col[(w_idx, slot)]
                for i in range(specs[w_idx].n_tiers):
                    rows.append(r - 2 * T)
                    cols.append(base + i)
            linking = sp.csc_matrix(
                (np.ones(len(rows)), (np.array(rows), np.array(cols))),
                shape=(self.n_linking, N))
            A = sp.vstack(
                [sp.hstack([core, sp.csc_matrix((2 * T, self.n_tier_vars))],
                           format='csc'), linking], format='csc')
        else:
            A = core

        lp = highspy.HighsLp()
        lp.num_col_ = N
        lp.num_row_ = n_rows
        lp.col_cost_ = c
        lp.col_lower_ = lo
        lp.col_upper_ = hi
        lp.row_lower_ = np.zeros(n_rows)
        lp.row_upper_ = np.zeros(n_rows)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr.astype(np.int32)
        lp.a_matrix_.index_ = A.indices.astype(np.int32)
        lp.a_matrix_.value_ = A.data

        self._h = highspy.Highs()
        self._h.setOptionValue('output_flag', False)
        self._h.passModel(lp)

        self._price_cols = np.arange(self.OI, self.OI + 2 * T, dtype=np.int32)
        self._curtail_cols = np.arange(self.OQ, self.OQ + T, dtype=np.int32)
        self._soc_last_col = np.array([self.OS + T - 1], dtype=np.int32)
        self._fb_rows = np.arange(T + 1, dtype=np.int32)
        self._tier_cols = np.arange(N_CORE, N, dtype=np.int32)
        self._tier_lo = np.zeros(self.n_tier_vars)
        self._soc_lo_const = soc_min
        self._soc_hi_const = bspec.capacity_kwh
        # Current membership per (window, slot), as a set of window positions.
        self._members: dict[tuple[int, int], frozenset] = {
            key: frozenset() for key in self._link_row}
        self.n_coeff_changes = 0     # cumulative, for diagnostics and tests

    # -- membership ---------------------------------------------------------

    def _positions(self, key, new) -> frozenset:
        """Membership value -> frozenset of window positions, validated."""
        T = self.T
        if new is None:
            return frozenset()
        if isinstance(new, tuple):
            lo, hi = new
            if not (0 <= lo <= hi <= T):
                raise ValueError(
                    f"set_membership: run {new} for window/slot {key} is "
                    f"not a valid half-open range within [0, {T}]"
                )
            return frozenset(range(lo, hi))
        pos = np.asarray(new, dtype=int).ravel()
        if pos.size and (pos.min() < 0 or pos.max() >= T):
            raise ValueError(
                f"set_membership: positions for window/slot {key} fall "
                f"outside [0, {T}): min {pos.min()}, max {pos.max()}"
            )
        return frozenset(int(p) for p in pos)

    def set_membership(
        self, runs: dict[tuple[int, int], object],
    ) -> None:
        """Install the window's block membership, changing only what moved.

        runs maps (block window index, day slot) to a half-open run of window
        positions (lo, hi), an array of window positions, or None for no
        intervals. Every (window, slot) pair must be present: a missing key
        would silently leave the previous window's coefficients in place,
        which is the failure this method exists to prevent.
        """
        if runs.keys() != self._link_row.keys():
            missing = sorted(self._link_row.keys() - runs.keys())
            extra = sorted(runs.keys() - self._link_row.keys())
            raise ValueError(
                f"set_membership: runs must cover every (window, slot) pair. "
                f"missing={missing} unexpected={extra}"
            )
        for key, new in runs.items():
            new_set = self._positions(key, new)
            old_set = self._members[key]
            if new_set == old_set:
                continue
            core_off = (self.OI if self.specs[key[0]].is_import else self.OE)
            row = self._link_row[key]
            # Symmetric difference of the two memberships. Contiguity of the
            # underlying run is what keeps this to a couple of coefficients
            # per step; forecast-critical positions add a few more.
            for pos in sorted(old_set - new_set):
                self._h.changeCoeff(row, core_off + pos, 0.0)
                self.n_coeff_changes += 1
            for pos in sorted(new_set - old_set):
                self._h.changeCoeff(row, core_off + pos, -1.0)
                self.n_coeff_changes += 1
            self._members[key] = new_set

    # -- solve --------------------------------------------------------------

    def solve(self, solar, load, p_imp, p_exp, soc_init, soc_final,
              tier_hi=None):
        """Solve one window. Returns the primal vector, or None if infeasible.

        The returned vector starts with the standard 6T block, so callers
        extract charge, discharge, import, export, SOC and curtailment at the
        usual offsets and need not know that tier variables exist.
        """
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

        if self.n_tier_vars and tier_hi is not None:
            if len(tier_hi) != self.n_tier_vars:
                raise ValueError(
                    f"solve: tier_hi has {len(tier_hi)} entries, expected "
                    f"{self.n_tier_vars}"
                )
            h.changeColsBounds(self.n_tier_vars, self._tier_cols,
                               self._tier_lo, np.asarray(tier_hi, dtype=float))

        rhs = np.empty(T + 1)
        rhs[:T] = load - solar
        rhs[T] = soc_init
        h.changeRowsBounds(T + 1, self._fb_rows, rhs, rhs)

        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None
        return np.asarray(h.getSolution().col_value, dtype=float)

    def restore_soc_bounds(self):
        """Undo the terminal pin on soc[T-1] before reusing the model."""
        self._h.changeColsBounds(
            1, self._soc_last_col,
            np.array([self._soc_lo_const]), np.array([self._soc_hi_const]))

    def objective_value(self) -> float:
        return float(self._h.getObjectiveValue())

    # -- introspection for the structural gate ------------------------------

    def constraint_matrix(self) -> sp.csc_matrix:
        """The solver's current constraint matrix, read back from HiGHS.

        For the structural test: objective agreement can mask a wrong matrix
        when the wrong entries happen not to bind, so the incremental update
        is checked against a freshly built model coefficient by coefficient.
        """
        lp = self._h.getLp()
        return sp.csc_matrix(
            (np.asarray(lp.a_matrix_.value_, dtype=float),
             np.asarray(lp.a_matrix_.index_),
             np.asarray(lp.a_matrix_.start_)),
            shape=(lp.num_row_, lp.num_col_))

    def linking_row_positions(self, key: tuple[int, int]) -> np.ndarray:
        """Window positions currently carrying a -1 in a linking row."""
        A = self.constraint_matrix().tocsr()
        row = A[self._link_row[key]].toarray().ravel()
        core_off = self.OI if self.specs[key[0]].is_import else self.OE
        return np.where(row[core_off:core_off + self.T] != 0.0)[0]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def _contiguous_runs(mask: np.ndarray) -> np.ndarray:
    """Half-open [start, stop) runs of True in a boolean mask, as an (n, 2)
    integer array."""
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges.reshape(-1, 2)


def _split_runs_by_day(runs: np.ndarray, day_map: np.ndarray) -> np.ndarray:
    """Cut every run at local-day boundaries.

    window_runs keys each run to a single day slot, taken from the day of its
    first interval, and the solver holds one tier variable per (window, day).
    A run spanning midnight would therefore post a whole day's extra flow to
    the first day's tier variable -- a wrong linking row rather than an error.

    Most block windows never need this: a half-open hour-of-day window inside
    a day already breaks at midnight, so the split is a no-op and behaviour is
    unchanged. It is needed for a window covering the WHOLE day, such as EA
    Battery Ease's 0-24 block export tier, whose mask is unbroken across the
    entire period and would otherwise be one run spanning every local day.

    Splitting rather than rejecting keeps the two-coefficient incremental
    update intact: each (window, day slot) still holds exactly one run, which
    is what the update assumes.
    """
    if len(runs) == 0:
        return runs
    out = []
    for start, stop in runs:
        days = day_map[start:stop]
        cuts = np.flatnonzero(days[1:] != days[:-1]) + 1 + start
        edges = np.concatenate(([start], cuts, [stop]))
        for lo, hi in zip(edges[:-1], edges[1:]):
            out.append((int(lo), int(hi)))
    return np.asarray(out, dtype=runs.dtype).reshape(-1, 2)


class AdvancedWindowPlanner:
    """Window planner for an advanced TOU plan.

    Built once per site. Holds everything about the plan that is fixed for
    the whole period, produces the price vectors and the LP for one window,
    and carries the two pieces of state that reset at local midnight.

    What is uncertain on this path
    ------------------------------
    The retail rates are known exactly: on a TOU plan build_price_vectors
    returns the retail layer alone, so the per-interval price depends on the
    clock and nothing else. Load and solar are forecast by their own
    providers. The wholesale price forecast matters for exactly one thing --
    which intervals are critical -- so on a plan with no critical rates the
    price forecast cannot change any decision and is not requested at all.

    Day state
    ---------
    Two quantities reset at local midnight and cannot be recovered from the
    window alone: how much of each block window's tier capacity has already
    been used today, and whether today's zero-import reward is still
    available. Both are running totals updated once per committed interval.

    They are NOT a running reproduction of the bill. The LP objective and the
    bill are different functions on purpose -- the objective carries a
    zero-import penalty no retailer charges and shifts the opposite side of
    critical intervals where the retailer does not. compute_bill on the committed
    flows remains the source of truth. These totals exist only so the
    objective expresses the right incentive at each decision instant.

    They must agree with billing in two places. The reward threshold is
    copied from _zero_import_day_credits: a bare "> 0" test would declare
    the reward lost on a 1e-14 solver residual while billing still paid it,
    and the rest of that day would be planned against an incentive billing
    disagrees with. And block usage leaves out flow that was critical on the
    window's side by the SETTLED price (settled_rrp_mwh), which is what
    billing's tier masks use; the forecast decides only the window LP.
    """

    def __init__(
        self,
        index: pd.DatetimeIndex,
        retail_tariff: RetailTariff,
        network_tariff,
        state: str,
        battery_spec: BatterySpec,
        horizon_intervals: int,
        export_limit_kwh_per_interval: float = np.inf,
        interval_hours: float = 5 / 60,
        curtail_cost_per_kwh_dollars: float | None = 0.0,
        settled_rrp_mwh=None,
        apply_export_order_nudge: bool = False,
    ):
        """
        Args:
            apply_export_order_nudge: On a 'tou' plan, add
                build_tou_export_order_nudge to the base export price, capped
                at the base import price (apply_tou_export_order_nudge), as
                perfect_foresight_lp_advanced does for the whole period. Every
                window's p_exp is a slice of the base, so each window prefers
                charging from the first surplus to exporting it. Raises
                ValueError when battery_spec.c_deg_per_kwh does not cover the
                nudge. No effect on a 'mixed' plan, whose window prices come
                from the dispatch stack.
        """
        if retail_tariff.type not in ('tou', 'mixed'):
            raise ValueError(
                f"AdvancedWindowPlanner is for TOU and mixed plans; got "
                f"{retail_tariff.type!r}. Wholesale-exposed plans are handled "
                f"by rolling_perfect_foresight_lp, which owns the "
                f"price-dependent stack."
            )
        self.index = index
        self.retail_tariff = retail_tariff
        self.state = state
        self.interval_hours = interval_hours
        self._bspec = battery_spec
        self._export_limit = export_limit_kwh_per_interval
        self._curtail_cost = curtail_cost_per_kwh_dollars
        T_p = len(index)
        self.T_p = T_p

        # Base price vectors: block-masked, critical-rate-free, fixed for the
        # period. A TOU plan's vectors do not depend on the wholesale price
        # (guarded by the type check above), so the zero series is not an
        # approximation -- it is the same answer any price would give.
        zero_prices = pd.Series(np.zeros(T_p), index=index)
        base_imp, base_exp = build_base_price_vectors_advanced(
            index, zero_prices, network_tariff, retail_tariff, state,
            interval_hours)
        self.base_imp = base_imp.to_numpy(dtype=float)
        self.base_exp = base_exp.to_numpy(dtype=float)

        # Price ordering. p_imp < p_exp anywhere is free money: the LP will
        # import and export in the same interval, and since the committed
        # traces net simultaneous flows while the LP's own variables do not,
        # the day-state accumulator and the bill would then disagree about how
        # much of a block tier was used.
        #
        # Checking the BASE is sufficient for every forecast. The critical
        # rates' delta is added to both sides, so it leaves the spread
        # untouched, and the zero-import penalty only raises p_imp.
        #
        # The likely cause is a block window whose masked import price
        # (usage_other) sits below the feed-in rate for the same interval,
        # which is a tariff definition error rather than a modelling one.
        bad = np.flatnonzero(self.base_imp < self.base_exp - 1e-12)
        if bad.size:
            j = int(bad[0])
            raise ValueError(
                f"{retail_tariff.name}: p_imp < p_exp at {bad.size} of "
                f"{T_p} intervals, first at {index[j]} "
                f"(p_imp={self.base_imp[j]:.6f}, p_exp={self.base_exp[j]:.6f} "
                f"$/kWh). Importing and exporting in the same interval would "
                f"be profitable, so the LP has no physical solution. Check "
                f"whether a block window's usage_other rate sits below the "
                f"feed-in rate for those hours."
            )

        # Export-order nudge. Added after the ordering check so a tariff error
        # is still reported against the real rates. The cap keeps
        # base_imp >= base_exp, so the argument above for checking the base
        # alone still holds.
        self.apply_export_order_nudge = apply_export_order_nudge
        if apply_export_order_nudge:
            nudge = build_tou_export_order_nudge(
                index, retail_tariff, state, interval_hours)
            if nudge is not None:
                check_export_order_nudge_degradation(
                    battery_spec.c_deg_per_kwh, retail_tariff.name)
                self.base_exp = apply_tou_export_order_nudge(
                    self.base_imp, self.base_exp, nudge.to_numpy())

        # A mixed plan's window prices depend on the forecast RRP, so they come
        # from the dispatch stack, not from base_imp/base_exp (which are built
        # above for every plan but not read on a mixed one). The stack proves
        # p_imp >= p_exp for every RRP in the market range at construction,
        # which covers any forecast the loop can pass (model_spec.md,
        # Mixed-plan dispatch prices).
        self.dispatch_stack = (
            build_mixed_dispatch_stack(index, network_tariff, retail_tariff,
                                       state, interval_hours)
            if retail_tariff.type == 'mixed' else None)

        # Critical rates: rho includes that side's flat adder
        # (critical_rates_dollars), matching apply_critical_rates.
        self._rho_x, self._rho_m = critical_rates_dollars(retail_tariff)

        # Settled criticality, for the day state only: observe_commit(k) reads
        # entry k once interval k has been committed, when its price is
        # settled. Needed only where a critical rate shares a side with a
        # block window; validation builds the planner without prices and
        # never commits.
        has_imp_windows = bool(retail_tariff.block_import_tiers)
        has_exp_windows = bool(retail_tariff.block_export_tiers)
        self._needs_settled = (
            (retail_tariff.critical_export_rate is not None and has_exp_windows)
            or (retail_tariff.critical_import_rate is not None and has_imp_windows))
        if settled_rrp_mwh is not None and retail_tariff.has_critical_rates:
            settled = np.asarray(settled_rrp_mwh, dtype=float)
            if settled.shape != (T_p,):
                raise ValueError(
                    f"settled_rrp_mwh has shape {settled.shape}, expected "
                    f"({T_p},) to match index"
                )
            self._settled_crit_x, self._settled_crit_m = critical_masks(
                settled, retail_tariff)
        else:
            self._settled_crit_x = self._settled_crit_m = None
        # Forecast criticality of the window last priced by window_prices,
        # consumed by solve_window for the same (k, n).
        self._window_crit: tuple | None = None

        # Zero-import reward: the penalty vector is the same one the
        # perfect-foresight path adds to p_imp; the availability rule is new.
        penalty = build_zero_import_penalty(
            index, retail_tariff, state, interval_hours)
        self._penalty = (None if penalty is None
                         else penalty.to_numpy(dtype=float))
        reward = retail_tariff.zero_import_reward
        if reward is None:
            self._reward_mask = None
            self._reward_threshold = 0.0
        else:
            lwc = local_wall_clock(index, state, interval_hours)
            self._reward_mask = ((lwc.hour >= reward.start_hour)
                                 & (lwc.hour < reward.end_hour))
            self._reward_threshold = (reward.tolerance_kwh
                                      if reward.tolerance_kwh > 0 else 1e-9)

        # Day map and block window membership.
        _, self.day_map = _build_day_map(index, state, interval_hours)
        self.specs = block_window_specs(retail_tariff)
        windows = list(retail_tariff.block_import_tiers or ())
        windows += list(retail_tariff.block_export_tiers or ())
        self._runs: list[np.ndarray] = []
        for w_idx, window in enumerate(windows):
            mask = block_window_mask(index, window, state, interval_hours)
            runs = _split_runs_by_day(_contiguous_runs(mask), self.day_map)
            # One run per local day is what makes the incremental coefficient
            # update two changes per step. _split_runs_by_day establishes it
            # for any window, including one covering the whole day, so this is
            # now an invariant on the splitter rather than a restriction on
            # the tariff. Kept because a violation would produce a silently
            # wrong linking row, not an error.
            for start, stop in runs:
                if self.day_map[start] != self.day_map[stop - 1]:
                    raise ValueError(
                        f"block window {w_idx} has a run spanning local days "
                        f"{self.day_map[start]}..{self.day_map[stop - 1]} "
                        f"(intervals {start}..{stop - 1}); the window solver "
                        f"assumes one contiguous run per local day"
                    )
            self._runs.append(runs)
            if len(runs) and len(runs) != len(np.unique(self.day_map[runs[:, 0]])):
                raise ValueError(
                    f"block window {w_idx} has more than one run on some "
                    f"local day; the window solver assumes one contiguous "
                    f"run per local day"
                )

        self.n_slots = self._max_slots(horizon_intervals)

        # Day state. Held for one local day at a time; a query about any other
        # day answers zero, so the accumulator cannot be read out of order.
        self._accum_day = -1
        self._accum_used = np.zeros(len(self.specs))
        self._accum_reward = 0.0

        # Run counters. Days are counted as they close, plus the day still
        # open when the counters are read, so plan_diagnostics() can be called
        # at any point without mutating anything.
        self._reward_days_lost = 0
        self._tier_days_exhausted = np.zeros(len(self.specs), dtype=int)
        self._windows_with_tier_exhausted = 0
        # Total FINITE tier capacity per block window: the point past which
        # every further kWh is charged (or credited) at the unbounded rate.
        self._finite_capacity = np.array(
            [sum(c for c in spec.capacities if np.isfinite(c))
             for spec in self.specs], dtype=float)

        self._solver: AdvancedWindowSolver | None = None
        self._solver_T = -1
        self.last_objective: float | None = None

    def _max_slots(self, H: int) -> int:
        """Largest number of local dates any window of length H can touch.

        Computed from the day map rather than from the horizon: local days are
        23 or 25 hours long across daylight saving, so any closed-form bound
        is either wrong at the edges or loose enough to allocate slots that
        never fill.
        """
        T_p, day_map = self.T_p, self.day_map
        if T_p == 0:
            return 1
        starts = np.arange(T_p)
        ends = np.minimum(starts + H, T_p) - 1
        return int((day_map[ends] - day_map[starts]).max()) + 1

    # -- what the loop asks before it does anything -------------------------

    @property
    def requires_price_forecast(self) -> bool:
        """False when nothing in the price vectors reads the wholesale price.

        On a plan with no critical rates the loop skips the price provider
        entirely: no query, no staleness accounting, and no self-consumption
        fallback triggered by a missing forecast that could not have changed
        the answer. A mixed plan always needs one: both of its window prices
        are built from the forecast RRP.
        """
        return self.dispatch_stack is not None or self._has_critical_rates

    @property
    def _has_critical_rates(self) -> bool:
        """Whether forecast criticality moves prices and tier membership.
        Not the same as requires_price_forecast: a mixed plan needs a
        forecast but has no critical rates."""
        return self._rho_x is not None or self._rho_m is not None

    # -- prices -------------------------------------------------------------

    def window_prices(self, rrp_mwh, k: int, n: int):
        """(p_imp, p_exp) in $/kWh for the window starting at interval k.

        rrp_mwh is the FORECAST wholesale price over the window, or None when
        requires_price_forecast is False. The critical test is applied to
        whatever is passed, which is the whole point: an interval takes the
        critical rate, and leaves its block tier, if the forecast said it
        would, not if it turned out to. The forecast criticality is kept for
        the solve_window call for the same window.
        """
        self._window_crit = None
        if self.dispatch_stack is not None:
            # Mixed plan: no critical rates, block tiers or reward to apply.
            if rrp_mwh is None:
                raise ValueError(
                    "window_prices: a mixed plan's window prices are built "
                    "from the price forecast, so one is required"
                )
            return self.dispatch_stack.window_price_vectors(
                np.asarray(rrp_mwh, dtype=float)[:n], k)

        sl = slice(k, k + n)
        p_imp = self.base_imp[sl].copy()
        p_exp = self.base_exp[sl].copy()

        if self._has_critical_rates:
            if rrp_mwh is None:
                raise ValueError(
                    "window_prices: this plan has critical rates, so a "
                    "price forecast is required"
                )
            rrp = np.asarray(rrp_mwh, dtype=float)[:n]
            crit_x, crit_m = critical_masks(rrp, self.retail_tariff)
            # Same arithmetic as apply_critical_rates(shift_opposite=True):
            # the critical side takes rho, the other side moves by the same
            # delta, so the spread p_imp - p_exp is unchanged and price
            # ordering needs no separate check.
            if self._rho_x is not None and crit_x.any():
                delta = self._rho_x - p_exp[crit_x]
                p_exp[crit_x] = self._rho_x
                p_imp[crit_x] += delta
            if self._rho_m is not None and crit_m.any():
                delta = self._rho_m - p_imp[crit_m]
                p_imp[crit_m] = self._rho_m
                p_exp[crit_m] += delta
            self._window_crit = (k, n, crit_x, crit_m)

        if self._penalty is not None:
            pen = self._penalty[sl].copy()
            if not self.reward_available(k):
                # Today's reward is already lost, so stop penalising today's
                # remaining imports. Positions on the next local day keep the
                # penalty: that day's reward is still live.
                pen[self.day_map[sl] == self.day_map[k]] = 0.0
            p_imp += pen

        return p_imp, p_exp

    # -- day state ----------------------------------------------------------

    def day0_used(self, k: int) -> dict[int, float]:
        """Committed flow through each block window earlier on interval k's
        local day, keyed by index into self.specs."""
        if self.day_map[k] != self._accum_day:
            return {}
        return {w: float(v) for w, v in enumerate(self._accum_used) if v > 0.0}

    def reward_available(self, k: int) -> bool:
        """Whether the zero-import reward for interval k's local day can still
        be earned given what has already been committed."""
        if self._reward_mask is None:
            return False
        if self.day_map[k] != self._accum_day:
            return True
        return self._accum_reward <= self._reward_threshold

    def _day_contribution(self) -> tuple[int, np.ndarray]:
        """What the currently open day would add to the counters.

        Kept separate from the counters themselves so plan_diagnostics() can
        be read mid-run without closing a day that is still accumulating.
        """
        if self._accum_day < 0:
            return 0, np.zeros(len(self.specs), dtype=int)
        lost = int(self._reward_mask is not None
                   and self._accum_reward > self._reward_threshold)
        exhausted = ((self._finite_capacity > 0)
                     & (self._accum_used >= self._finite_capacity - 1e-9))
        return lost, exhausted.astype(int)

    def _close_day(self) -> None:
        lost, exhausted = self._day_contribution()
        self._reward_days_lost += lost
        self._tier_days_exhausted += exhausted

    def plan_diagnostics(self) -> dict:
        """Advanced-plan counters for RollingDiagnostics.

        None-valued entries mean the plan does not have that feature, which a
        reader needs to tell apart from a feature that was never triggered.
        """
        lost, exhausted = self._day_contribution()
        has_reward = self._reward_mask is not None
        has_tiers = len(self.specs) > 0
        return {
            'reward_days_lost': (self._reward_days_lost + lost
                                 if has_reward else None),
            'block_tier_days_exhausted': (
                tuple(int(v) for v in self._tier_days_exhausted + exhausted)
                if has_tiers else None),
            'windows_with_tier_exhausted': (self._windows_with_tier_exhausted
                                            if has_tiers else None),
        }

    def observe_commit(self, k: int, import_kwh: float,
                       export_kwh: float) -> None:
        """Record the flows actually committed in interval k.

        Called for every committed interval, including ones that fell back to
        self-consumption: a fallback is a statement about what the battery
        did, and those flows count against the tiers and the reward like any
        others.
        """
        day = self.day_map[k]
        if day != self._accum_day:
            self._close_day()
            self._accum_day = int(day)
            self._accum_used[:] = 0.0
            self._accum_reward = 0.0
        if self._needs_settled and self._settled_crit_x is None:
            raise ValueError(
                "observe_commit: this plan has a critical rate on a side "
                "with a block window, so the planner needs settled_rrp_mwh "
                "to count block usage as billing does"
            )
        for w_idx, spec in enumerate(self.specs):
            runs = self._runs[w_idx]
            if _in_runs(runs, k):
                # Flow that was critical on this window's side by the settled
                # price does not count towards the tier.
                if self._settled_crit_x is not None and (
                        self._settled_crit_m[k] if spec.is_import
                        else self._settled_crit_x[k]):
                    continue
                self._accum_used[w_idx] += (import_kwh if spec.is_import
                                            else export_kwh)
        if self._reward_mask is not None and self._reward_mask[k]:
            self._accum_reward += import_kwh

    # -- membership and solve ----------------------------------------------

    def window_runs(self, k: int, n: int):
        """Membership run per (block window, day slot) for the window at k."""
        day0 = self.day_map[k]
        out: dict[tuple[int, int], tuple[int, int] | None] = {}
        for w_idx in range(len(self.specs)):
            for slot in range(self.n_slots):
                out[(w_idx, slot)] = None
            runs = self._runs[w_idx]
            if len(runs) == 0:
                continue
            # Runs overlapping [k, k+n): the last run starting before k+n,
            # walking back while it still reaches past k.
            j = int(np.searchsorted(runs[:, 0], k + n, side='left')) - 1
            while j >= 0 and runs[j, 1] > k:
                start, stop = int(runs[j, 0]), int(runs[j, 1])
                lo, hi = max(start, k) - k, min(stop, k + n) - k
                if hi > lo:
                    slot = int(self.day_map[max(start, k)] - day0)
                    if not (0 <= slot < self.n_slots):
                        raise ValueError(
                            f"window at k={k} puts block window {w_idx} on "
                            f"day slot {slot}, outside [0, {self.n_slots}); "
                            f"the slot count was computed for a shorter "
                            f"horizon than this window"
                        )
                    out[(w_idx, slot)] = (lo, hi)
                j -= 1
        return out

    def window_members(self, k: int, n: int):
        """Membership per (block window, day slot) for the window at k, with
        positions forecast-critical on each window's side removed.

        Uses the forecast criticality recorded by window_prices for this same
        window. Without critical rates it is window_runs.
        """
        runs = self.window_runs(k, n)
        if not self._has_critical_rates:
            return runs
        wc = self._window_crit
        if wc is None or wc[0] != k or wc[1] != n:
            raise ValueError(
                f"window_members: window_prices was not called for the "
                f"window at k={k}, n={n}, so its forecast criticality is "
                f"unknown"
            )
        _, _, crit_x, crit_m = wc
        out = {}
        for key, run in runs.items():
            if run is None:
                out[key] = None
                continue
            crit = crit_m if self.specs[key[0]].is_import else crit_x
            lo, hi = run
            pos = np.arange(lo, hi)
            out[key] = pos[~crit[lo:hi]] if crit[lo:hi].any() else run
        return out

    def solve_window(self, k: int, n: int, solar, load, p_imp, p_exp,
                     soc_init: float, soc_final: float):
        """Solve the window at k. Returns the primal vector, or None."""
        if self._solver is None or self._solver_T != n:
            # Windows shrink only in the final H intervals of the period.
            self._solver = AdvancedWindowSolver(
                n, self._bspec, self._export_limit, self.interval_hours,
                self._curtail_cost, self.specs, self.n_slots)
            self._solver_T = n
        solver = self._solver
        solver.set_membership(self.window_members(k, n))
        used = self.day0_used(k)
        tier_hi = (tier_upper_bounds(self.specs, self.n_slots, used)
                   if self.specs else None)
        if used and any(
                self._finite_capacity[w] > 0
                and v >= self._finite_capacity[w] - 1e-9
                for w, v in used.items()):
            self._windows_with_tier_exhausted += 1
        x = solver.solve(solar, load, p_imp, p_exp, soc_init, soc_final,
                         tier_hi)
        # Read the objective BEFORE restoring the SOC bounds: any change to
        # the model invalidates the solution HiGHS reports, so reading it
        # afterwards silently returns zero.
        self.last_objective = None if x is None else solver.objective_value()
        solver.restore_soc_bounds()
        return x


def _in_runs(runs: np.ndarray, k: int) -> bool:
    """Whether interval k falls inside any half-open run."""
    if len(runs) == 0:
        return False
    j = int(np.searchsorted(runs[:, 0], k, side='right')) - 1
    return j >= 0 and k < runs[j, 1]
