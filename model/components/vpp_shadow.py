"""The self-consumption shadow trajectory and its natural-flow bounds.

Under perfect foresight the natural/forced split comes from one
self-consumption run across a known event window (`vpp_sc_event.py`,
`_compute_bounds`). Under rolling the window boundary is not known: an event
may end at the next interval or run for hours, so there is no fixed span to
run across.

This module supplies the rolling equivalent. A shadow battery is anchored at
the state of charge the household handed over when the event was declared and
advanced one interval per committed interval, on the same site data the LP
saw. It is a counterfactual dispatch and never touches the real battery, so
its bounds do not depend on what the LP does -- which is what lets the natural
import budget stay aggregate without the feedback the perfect-foresight spec
describes.

The two bounds keep the asymmetry of the perfect-foresight component, and for
the same reasons:

  Natural export is a PER-INTERVAL ceiling. Forced export is the rewarded
  flow, so the operator wants less of it on the books. With a budget the LP
  could suppress export early -- curtail, or charge instead -- to bank
  headroom, then book a later larger export as natural and under-report what
  the household is owed. A per-interval ceiling makes that impossible.

  Natural import is an AGGREGATE budget over the expected event duration.
  Constraining each interval separately would penalise shifting import in
  time without increasing it, and would make the split depend on when within
  the event the LP chose to import -- a distinction the contract does not
  make.

Nothing here solves or knows about an LP. See rolling_vpp_event_spec.md,
"The shadow trajectory".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from model.core.scenario import BatterySpec
from model.components.self_consumption import run_sc_with_battery


@dataclass
class ShadowBounds:
    """Natural-flow bounds for one operator LP solve.

    Attributes:
        n_exp_hi: Per-interval natural export ceiling, kWh, length n_eff.
            Bounds n_exp[j] in the operator LP.
        n_imp_budget: Aggregate natural import allowance for the REMAINDER of
            the event, kWh, after subtracting import already committed as
            natural. Bounds sum(n_imp) over the LP's horizon. Clamped at zero.
        full_budget: The unclamped aggregate over the whole expected event
            span [k0, t_star], before subtracting committed import. Reported
            so a clamp can be attributed: a small full_budget and a large
            committed total mean different things.
        committed_n_imp: Natural import committed so far in this event, kWh.
        clamped: True when subtracting committed import drove the budget
            below zero. Every further kWh of import in this event is then
            forced, which is the right answer -- the counterfactual allowance
            really is spent -- but a clamp firing often means the projection
            is unstable, so it is counted.
    """
    n_exp_hi: np.ndarray
    n_imp_budget: float
    full_budget: float
    committed_n_imp: float
    clamped: bool


@dataclass
class ShadowDiagnostics:
    """Counters over a whole run. Read without mutating."""
    events_anchored: int = 0
    committed_intervals: int = 0
    n_imp_budget_clamps: int = 0
    n_imp_resettled_kwh: float = 0.0
    blocks_resettled: int = 0
    block_lengths: list[int] = field(default_factory=list)


class ShadowTrajectory:
    """A self-consumption battery running alongside the real one, per event.

    Lifecycle, once per contiguous event block:

        anchor(k0, soc)      at the interval the event is declared
        bounds(...)          once per interval, before the LP solve
        commit(...)          once per interval, after the interval is committed
        settle()             at the interval the event ends

    `bounds` is pure. All state changes go through `anchor`, `commit` and
    `settle`, so a diagnostic read mid-run cannot disturb the trajectory.
    """

    def __init__(self, battery_spec: BatterySpec, interval_hours: float = 5 / 60):
        self.bspec = battery_spec
        self.interval_hours = interval_hours

        capacity = battery_spec.capacity_kwh
        self.capacity = capacity
        self.soc_min = (battery_spec.backup_reserve_pct / 100.0) * capacity
        self.power_kwh = battery_spec.power_kw * interval_hours
        self.eta_c = battery_spec.charge_efficiency
        self.eta_d = battery_spec.discharge_efficiency

        self.diagnostics = ShadowDiagnostics()

        # Per-event state. `active` is False between events.
        self.active = False
        self.k0: int | None = None
        self._soc_prev: float | None = None   # shadow SOC entering the next interval
        self._accrued_n_imp = 0.0             # committed shadow natural import
        self._committed_n_imp = 0.0           # committed LP natural import
        self._committed_import = 0.0          # committed LP total import
        self._n_committed = 0

    # -- lifecycle ----------------------------------------------------------

    def anchor(self, k0: int, soc_kwh: float) -> None:
        """Start a new event block at interval k0 from the real battery's SOC.

        Anchoring at the handed-over SOC, rather than running one shadow from
        the start of the period, matches what vpp_sc_event and vpp_tou_event
        do: the counterfactual is "what would THIS battery have done, from the
        state the household left it in". A period-long shadow would be the
        truer never-joined-the-VPP counterfactual but would drift permanently
        from the real battery, so its bounds would stop describing a reachable
        state.
        """
        if self.active:
            raise RuntimeError(
                f"anchor(k0={k0}) called while an event anchored at "
                f"{self.k0} is still active; call settle() first"
            )
        if not (self.soc_min - 1e-9 <= soc_kwh <= self.capacity + 1e-9):
            raise ValueError(
                f"anchor SOC {soc_kwh} outside [{self.soc_min}, "
                f"{self.capacity}]"
            )
        self.active = True
        self.k0 = k0
        self._soc_prev = float(soc_kwh)
        self._accrued_n_imp = 0.0
        self._committed_n_imp = 0.0
        self._committed_import = 0.0
        self._n_committed = 0
        self.diagnostics.events_anchored += 1

    def settle(self) -> tuple[float, float]:
        """End the current event block and settle the import split.

        At this point the true span is known and the shadow's committed prefix
        over it was built entirely on actuals, so the true budget is exactly
        computable -- unlike at any interval during the event, where the tail
        was still a projection.

            true_budget   = accrued shadow natural import over [k0, k_end]
            n_imp_settled = min(total committed import, true_budget)
            f_imp_settled = total committed import - n_imp_settled

        Returns (n_imp_settled, f_imp_settled) as block TOTALS. The caller
        redistributes them across the block's intervals in proportion to
        committed import, which is exact for billing because the reward rates
        are flat in $/kWh, so only the totals reach retail_other_credits.

        The correction is symmetric: it moves volume to f_imp when the LP
        over-claimed against a budget the event never delivered, and to n_imp
        when it under-claimed. Only the natural/forced split changes; total
        import per interval is untouched, so dispatch, SOC and the energy
        component of the bill are unaffected.

        Export needs no equivalent. n_exp_hi at a committed interval is built
        from the shadow SOC entering that interval, which is in the committed
        prefix and therefore already on actuals.
        """
        if not self.active:
            raise RuntimeError("settle() called with no active event")

        true_budget = self._accrued_n_imp
        total_import = self._committed_import
        n_imp_settled = min(total_import, true_budget)
        f_imp_settled = total_import - n_imp_settled

        moved = n_imp_settled - self._committed_n_imp
        if abs(moved) > 1e-9:
            self.diagnostics.n_imp_resettled_kwh += moved
            self.diagnostics.blocks_resettled += 1
        self.diagnostics.block_lengths.append(self._n_committed)

        self.active = False
        self.k0 = None
        self._soc_prev = None
        return n_imp_settled, f_imp_settled

    # -- per-interval -------------------------------------------------------

    def bounds(
        self,
        solar_window: np.ndarray,
        load_window: np.ndarray,
        n_eff: int,
    ) -> ShadowBounds:
        """Bounds for a solve over [k, k + n_eff - 1]. Does not mutate state.

        solar_window and load_window are the arrays the LP sees, positions 0
        onward from the committed interval. Position 0 carries actuals when
        commit_actual_load is True; everything beyond it is forecast. The
        shadow therefore degrades exactly as the dispatch does, with no
        branch here.

        The projection over [k, t_star] is rebuilt every interval and thrown
        away. Only the committed step, taken in commit(), persists.
        """
        if not self.active:
            raise RuntimeError("bounds() called with no active event")
        if n_eff < 1:
            raise ValueError(f"n_eff must be >= 1, got {n_eff}")
        if len(solar_window) < n_eff or len(load_window) < n_eff:
            raise ValueError(
                f"window arrays shorter than n_eff={n_eff}: "
                f"solar={len(solar_window)}, load={len(load_window)}"
            )

        solar = np.asarray(solar_window[:n_eff], dtype=float)
        load = np.asarray(load_window[:n_eff], dtype=float)

        # Project the shadow across the horizon from its current SOC. Returns
        # end-of-interval SOC, so soc_prev is the entry SOC shifted by one.
        soc_arr, _ = run_sc_with_battery(
            solar, load, self.bspec, self._soc_prev, self.interval_hours)
        soc_prev = np.empty(n_eff)
        soc_prev[0] = self._soc_prev
        if n_eff > 1:
            soc_prev[1:] = soc_arr[:-1]

        # Same formulas as vpp_sc_event._compute_bounds, on the shadow SOC.
        discharge_avail = np.minimum(
            (soc_prev - self.soc_min) * self.eta_d, self.power_kwh)
        projected_n_imp = np.maximum(0.0, load - solar - discharge_avail)

        n_exp_hi = np.maximum(
            0.0,
            solar - load - np.minimum(
                (self.capacity - soc_prev) / self.eta_c, self.power_kwh),
        )

        # The budget spans the whole expected event, not just what has been
        # committed: accrued prefix (actuals) plus projected tail (forecast),
        # less what the LP has already booked as natural.
        full_budget = self._accrued_n_imp + float(projected_n_imp.sum())
        raw = full_budget - self._committed_n_imp
        clamped = raw < 0.0

        return ShadowBounds(
            n_exp_hi=n_exp_hi,
            n_imp_budget=max(0.0, raw),
            full_budget=full_budget,
            committed_n_imp=self._committed_n_imp,
            clamped=clamped,
        )

    def commit(
        self,
        solar_actual: float,
        load_actual: float,
        n_imp: float,
        import_total: float,
    ) -> None:
        """Advance the shadow one interval and record what the LP committed.

        solar_actual and load_actual are position 0 of the window the LP saw.
        n_imp and import_total are the committed natural import and total
        import for that interval.
        """
        if not self.active:
            raise RuntimeError("commit() called with no active event")

        solar = np.array([float(solar_actual)])
        load = np.array([float(load_actual)])
        soc_prev = self._soc_prev

        discharge_avail = min(
            (soc_prev - self.soc_min) * self.eta_d, self.power_kwh)
        self._accrued_n_imp += max(
            0.0, float(load[0]) - float(solar[0]) - discharge_avail)

        soc_arr, _ = run_sc_with_battery(
            solar, load, self.bspec, soc_prev, self.interval_hours)
        self._soc_prev = float(soc_arr[0])

        self._committed_n_imp += float(n_imp)
        self._committed_import += float(import_total)
        self._n_committed += 1

        self.diagnostics.committed_intervals += 1

    def note_clamp(self) -> None:
        """Record that the budget handed to a solve was clamped at zero.

        Separate from bounds() so that bounds() stays pure: the caller counts
        the clamp only for the solve it actually ran, not for any bounds it
        computed and discarded.
        """
        self.diagnostics.n_imp_budget_clamps += 1

    # -- read-only ----------------------------------------------------------

    @property
    def soc_prev(self) -> float | None:
        """Shadow SOC entering the next interval, or None between events."""
        return self._soc_prev

    def plan_diagnostics(self) -> dict:
        """Snapshot of the counters. Does not mutate."""
        d = self.diagnostics
        lengths = d.block_lengths
        return {
            'events_anchored': d.events_anchored,
            'event_intervals': d.committed_intervals,
            'n_imp_budget_clamps': d.n_imp_budget_clamps,
            'n_imp_resettled_kwh': d.n_imp_resettled_kwh,
            'blocks_resettled': d.blocks_resettled,
            'event_block_len_min': min(lengths) if lengths else 0,
            'event_block_len_max': max(lengths) if lengths else 0,
            'event_block_len_mean': (sum(lengths) / len(lengths)
                                     if lengths else 0.0),
            'event_blocks_len_1': sum(1 for n in lengths if n == 1),
        }
