"""The VPP window planner: one loop, one planner, two modes.

`rolling_optimisation` runs unchanged. This is a `WindowPlanner` supplied
through its existing `planner` argument, holding a household sub-planner, the
operator LP, the shadow trajectory, the event state and the cap accumulator,
and deciding per interval which mode governs.

Why one planner rather than two
-------------------------------
The obvious alternative -- an operator planner and a household planner run
alternately -- breaks the TOU variant. On `rolling_vpp_tou_event` the household
is billed on total import and export INCLUDING the flows the operator forces
during events, so the household's block-tier and zero-import day state has to
observe event intervals. Under two planners it would go blind at exactly the
intervals carrying the largest flows. A single planner that forwards
`observe_commit` unconditionally gets this right by construction.

Mode is decided inside `window_prices`, because that is where the forecast
slice arrives, and cached for the `solve_window(k)` that immediately follows.
The price vectors returned are therefore the ones that enter the objective
actually solved, so a retained plan is never annotated with prices from the
other mode.

Call order the loop guarantees, and what this planner does with it

    window_prices(k)   decide the mode; settle a block that has just ended
    solve_window(k)    build the vector; stash the committed split
    observe_commit(k)  advance the shadow, decrement the cap, record

`observe_commit` runs for EVERY committed interval, including self-consumption
fallbacks, where `solve_window` returned None or was never called. A fallback
is a statement about what the battery did, not a gap: its flows are entirely
natural, they do not touch the cap, and the household sub-planner still sees
them.

Degraded intervals reach `observe_commit` without a preceding
`window_prices` -- the loop falls back before asking for prices when no
forecast exists. Mode defaults to household there, which is right: an operator
that cannot see prices cannot declare an event.

See rolling_vpp_event_spec.md, "Architecture".
"""
from __future__ import annotations

import numpy as np

from model.core.scenario import BatterySpec
from model.core.prices import PriceStackOffsets
from model.components.self_consumption import run_sc_with_battery
from model.components.vpp_declaration import CapState, declare
from model.components.vpp_operator_lp_highs import HighsOperatorLp
from model.components.vpp_shadow import ShadowTrajectory

HOUSEHOLD = 'household'
OPERATOR = 'operator'


class VppWindowPlanner:
    """Two-mode planner for the rolling VPP coordination types.

    Args:
        T: Intervals in the period.
        op_price_offsets: Operator tariff price stack, used to turn a forecast
            RRP window into operator import/export prices in $/kWh.
        battery_spec: The site's battery.
        w: Event window length in intervals, from
            vpp_declaration.window_intervals().
        vpp_threshold_dollars_per_mwh: Declaration threshold.
        vpp_exp_reward, vpp_imp_reward: Household reward rates, $/kWh.
        cap: Per-site annual cap state.
        min_export_price_dollars_per_mwh: Fleet-wide scalar for the whole
            period. Converted to $/kWh once here.
        export_limit_kwh_per_interval: Site export limit, or inf.
        p_curtail: Curtailment penalty in $/kWh, or None to disable.
        no_grid_charge_mask: Length-T boolean, or None.
        household_planner: A WindowPlanner for between-event intervals, or
            None for the self-consumption variant, where dispatch is
            deterministic and reads neither prices nor the forecast.
    """

    def __init__(
        self,
        T: int,
        op_price_offsets: PriceStackOffsets,
        battery_spec: BatterySpec,
        w: int,
        vpp_threshold_dollars_per_mwh: float,
        vpp_exp_reward: float,
        vpp_imp_reward: float,
        cap: CapState,
        min_export_price_dollars_per_mwh: float,
        export_limit_kwh_per_interval: float = np.inf,
        interval_hours: float = 5 / 60,
        site_id: str = '',
        p_curtail: float | None = None,
        no_grid_charge_mask: np.ndarray | None = None,
        household_planner=None,
    ):
        self.T = int(T)
        self.op_offsets = op_price_offsets
        self.bspec = battery_spec
        self.w = int(w)
        self.threshold = float(vpp_threshold_dollars_per_mwh)
        self.exp_reward = float(vpp_exp_reward)
        self.imp_reward = float(vpp_imp_reward)
        self.cap = cap
        self.min_export_price_per_kwh = (
            float(min_export_price_dollars_per_mwh) / 1000.0)
        self.export_limit = export_limit_kwh_per_interval
        self.interval_hours = interval_hours
        self.site_id = site_id
        self.p_curtail = p_curtail
        self.no_grid_charge_mask = no_grid_charge_mask
        self.household = household_planner

        self.shadow = ShadowTrajectory(battery_spec, interval_hours)

        # Per-interval record, assembled into VppTraces by the wrapper.
        self.mode = np.array([HOUSEHOLD] * self.T, dtype=object)
        self.n_imp = np.zeros(self.T)
        self.f_imp = np.zeros(self.T)
        self.n_exp = np.zeros(self.T)
        self.f_exp = np.zeros(self.T)
        self.n_eff = np.zeros(self.T, dtype=int)

        # Set in window_prices, consumed in solve_window.
        self._pending_mode: tuple[int, str, int] | None = None
        self._op_prices: tuple[np.ndarray, np.ndarray] | None = None
        # Set in solve_window, consumed in observe_commit.
        self._pending_split: tuple[int, float, float, float, float] | None = None
        self._pending_site: tuple[int, float, float] | None = None

        self._counts = {
            'event_intervals_fallback': 0,
            'n_lt_w': 0,
            'n_lt_w_first': None,
            'n_lt_w_last': None,
            'f_exp_blocked_by_price': 0,
            'operator_infeasible': 0,
        }
        self._block_had_spike = False
        self._blocks_no_spike = 0

        # One HiGHS model per site, built here and reused for every event
        # interval. Consecutive windows differ by one interval of shift and a
        # data refresh, so the retained basis is nearly optimal for the next
        # solve: measured 5.9x-13.8x against rebuilding through
        # scipy.optimize.linprog, and nearly flat in event span rather than
        # scaling with it.
        #
        # Not picklable, which is fine: the planner is constructed inside the
        # worker by rolling_vpp_event, never sent across the pool boundary.
        self._op_lp = HighsOperatorLp(
            w=self.w,
            battery_spec=battery_spec,
            interval_hours=interval_hours,
            vpp_limit_applies_to=cap.limit_applies_to,
            has_no_grid_charge=no_grid_charge_mask is not None,
            p_curtail=p_curtail,
            export_limit=export_limit_kwh_per_interval,
        )

    # -- WindowPlanner protocol --------------------------------------------

    @property
    def needs_terminal_soc(self) -> bool:
        """False: nothing on this path reads the loop's terminal SOC pin.

        The operator LP deliberately has no terminal constraint -- it
        dispatches on event economics and leaves whatever remains -- and
        self-consumption between events is deterministic. The loop's
        SC-forward run that produces the pin costs ~30.5 s per site-year at an
        18-hour horizon, measured, and every second of it is discarded here.

        The TOU variant needs it True, because AdvancedWindowPlanner does pin
        its terminal SOC -- hence the test on the sub-planner rather than a
        constant.
        """
        return self.household is not None

    @property
    def requires_price_forecast(self) -> bool:
        """Always True: the operator reads the forecast to declare events even
        when the household plan's prices never move with the wholesale price.
        This also discharges the rule that a rolling run must have at least one
        genuine source of imperfect information, with no special case."""
        return True

    def window_prices(self, rrp_mwh, k: int, n: int):
        # A block that ran up to k-1 is settled here rather than in
        # observe_commit, because this is the first point at which the mode
        # for k is known and therefore the first point at which the block can
        # be said to have ended.
        active, j_star = self._declare(rrp_mwh, k, n)

        if not active and self.shadow.active:
            self._settle_block()

        if active:
            self.mode[k] = OPERATOR
            if not self.shadow.active:
                self._block_had_spike = False
            rrp = np.asarray(rrp_mwh, dtype=float)[:n]
            p_imp, p_exp = self.op_offsets.window_price_vectors(rrp, k)
            self._op_prices = (p_imp, p_exp)
            self._pending_mode = (k, OPERATOR, j_star)
            return p_imp, p_exp

        self.mode[k] = HOUSEHOLD
        self._op_prices = None
        self._pending_mode = (k, HOUSEHOLD, -1)
        if self.household is not None:
            return self.household.window_prices(rrp_mwh, k, n)
        # Self-consumption variant: dispatch is deterministic and no objective
        # is solved, so there are no prices to report. The mode array is what
        # distinguishes these intervals.
        return np.zeros(n), np.zeros(n)

    def solve_window(self, k: int, n: int, solar, load, p_imp, p_exp,
                     soc_init: float, soc_final: float):
        mode, j_star = self._mode_for(k)

        if mode == OPERATOR:
            return self._solve_operator(
                k, n, solar, load, j_star, soc_init)

        if self.household is not None:
            x = self.household.solve_window(
                k, n, solar, load, p_imp, p_exp, soc_init, soc_final)
            if x is not None:
                self._stash_household_split(k, x, n)
            return x
        return self._solve_household_sc(k, n, solar, load, soc_init)

    def observe_commit(self, k: int, import_kwh: float,
                       export_kwh: float) -> None:
        # A degraded interval reaches here with no window_prices call, so the
        # mode stays HOUSEHOLD and any open block must be closed.
        if self.shadow.active and self.mode[k] != OPERATOR:
            self._settle_block()

        if self._pending_split is not None and self._pending_split[0] == k:
            _, n_i, f_i, n_e, f_e = self._pending_split
        else:
            # Fallback: no solution for this interval. Every flow is natural,
            # nothing touches the cap.
            n_i, f_i, n_e, f_e = float(import_kwh), 0.0, float(export_kwh), 0.0
            if self.mode[k] == OPERATOR:
                self._counts['event_intervals_fallback'] += 1

        self.n_imp[k], self.f_imp[k] = n_i, f_i
        self.n_exp[k], self.f_exp[k] = n_e, f_e

        if self.mode[k] == OPERATOR and self.shadow.active:
            solar_a, load_a = self._site_for(k)
            self.shadow.commit(solar_a, load_a,
                               n_imp=n_i, import_total=float(import_kwh))
            self.cap.commit(k, f_imp=f_i, f_exp=f_e)

        if self.household is not None:
            # Forwarded unconditionally, including during events and on
            # fallbacks: the household is billed on these flows, so its day
            # state must include them.
            self.household.observe_commit(k, import_kwh, export_kwh)

        self._pending_split = None
        self._pending_site = None

    def plan_diagnostics(self) -> dict:
        """Snapshot. Does not mutate, so it can be read mid-run."""
        out = dict(self._counts)
        out.pop('n_lt_w', None)
        out['n_lt_W'] = self._counts['n_lt_w']
        out['n_lt_W_first'] = self._counts['n_lt_w_first']
        out['n_lt_W_last'] = self._counts['n_lt_w_last']
        out.pop('n_lt_w_first', None)
        out.pop('n_lt_w_last', None)
        out.update(self.shadow.plan_diagnostics())
        out.update(self.cap.plan_diagnostics())
        out['event_blocks_no_spike'] = self._blocks_no_spike
        out['f_imp_total_kwh'] = float(self.f_imp.sum())
        out['f_exp_total_kwh'] = float(self.f_exp.sum())
        if self.household is not None:
            hh = self.household.plan_diagnostics()
            if hh:
                out.update(hh)
        return out

    def finalize(self) -> None:
        """Close a block still open at the end of the period.

        Without this the last event's import split would never be settled, and
        its intervals would keep whatever the LP booked against a budget the
        event was still projecting.
        """
        if self.shadow.active:
            self._settle_block()

    # -- internals ----------------------------------------------------------

    def _declare(self, rrp_mwh, k: int, n: int) -> tuple[bool, int]:
        if self.cap.exhausted:
            # Terminal: the site stops participating and the household
            # strategy runs to the end of the period.
            return False, -1
        if rrp_mwh is None:
            return False, -1
        if n < self.w:
            c = self._counts
            c['n_lt_w'] += 1
            if c['n_lt_w_first'] is None:
                c['n_lt_w_first'] = k
            c['n_lt_w_last'] = k
        return declare(rrp_mwh, n, self.w, self.threshold)

    def _mode_for(self, k: int) -> tuple[str, int]:
        if self._pending_mode is not None and self._pending_mode[0] == k:
            return self._pending_mode[1], self._pending_mode[2]
        return HOUSEHOLD, -1

    def _solve_operator(self, k, n, solar, load, j_star, soc_init):
        if not self.shadow.active:
            self.shadow.anchor(k0=k, soc_kwh=soc_init)
        n_eff = j_star + 1

        # Stashed BEFORE the solve, not after. On an infeasible window the
        # loop falls back to self-consumption and still calls observe_commit,
        # where the shadow must advance on the site data the LP saw. Stashing
        # only on success left that path with nothing to advance on.
        self._pending_site = (k, float(solar[0]), float(load[0]))

        bounds = self.shadow.bounds(solar, load, n_eff)
        if bounds.clamped:
            self.shadow.note_clamp()

        p_imp, p_exp = self._op_prices
        ngc = (None if self.no_grid_charge_mask is None
               else self.no_grid_charge_mask[k:k + n_eff])

        sol = self._op_lp.solve(
            solar=solar, load=load, op_imp=p_imp, op_exp=p_exp,
            n=n, n_eff=n_eff,
            n_imp_budget=bounds.n_imp_budget, n_exp_hi=bounds.n_exp_hi,
            soc_entry=soc_init,
            vpp_exp_reward=self.exp_reward, vpp_imp_reward=self.imp_reward,
            remaining_cap=max(0.0, self.cap.remaining_kwh),
            min_export_price_per_kwh=self.min_export_price_per_kwh,
            no_grid_charge_window=ngc,
        )
        if sol is None:
            self._counts['operator_infeasible'] += 1
            return None

        self.n_eff[k] = n_eff
        if self.min_export_price_per_kwh > 0.0 and p_exp[0] < self.min_export_price_per_kwh:
            self._counts['f_exp_blocked_by_price'] += 1
        c = sol.committed
        self._pending_split = (
            k, c['n_imp'], c['f_imp'], c['n_exp'], c['f_exp'])
        return sol.x

    def _solve_household_sc(self, k, n, solar, load, soc_init):
        """Deterministic self-consumption over the window.

        Returns a vector rather than None. None is the loop's infeasibility
        signal and would land in sc_fallback_infeasible, reporting a failure
        where there was a normal deterministic dispatch.
        """
        # Only interval 0 is ever committed, and self-consumption is
        # deterministic, so the remaining n-1 intervals of the window are
        # computed and thrown away. run_sc_with_battery costs 0.290 ms over a
        # 216-interval window against 0.004 ms over one -- 75.7x, measured --
        # and at ~96k household intervals per site-year that is ~27.5 s of a
        # 186.6 s run. Solving one interval and holding the tail flat is
        # exact for everything the loop reads.
        #
        # The cost is that a retained plan on this path shows the committed
        # interval rather than a full SC window. There is no plan to inspect
        # here in any case: between events the dispatch is a fixed function of
        # solar, load and SOC, not the output of an optimisation.
        soc_arr, flows = run_sc_with_battery(
            np.asarray(solar[:1], dtype=float),
            np.asarray(load[:1], dtype=float),
            self.bspec, soc_init, self.interval_hours)

        x = np.zeros(6 * n)
        x[0*n] = flows['sol_to_bat'][0] + flows['grid_to_bat'][0]
        x[1*n] = flows['bat_to_home'][0]
        x[2*n] = flows['grid_to_home'][0] + flows['grid_to_bat'][0]
        x[3*n] = flows['sol_to_grid'][0]
        x[4*n:5*n] = soc_arr[0]          # SOC is a state: hold it flat
        x[5*n] = flows['curtail'][0]

        self._pending_split = (k, float(x[2*n]), 0.0, float(x[3*n]), 0.0)
        self._pending_site = (k, float(solar[0]), float(load[0]))
        return x

    def _stash_household_split(self, k, x, n):
        """Household LP intervals are entirely natural by definition."""
        self._pending_split = (k, float(x[2*n]), 0.0, float(x[3*n]), 0.0)

    def _site_for(self, k):
        if self._pending_site is not None and self._pending_site[0] == k:
            return self._pending_site[1], self._pending_site[2]
        raise RuntimeError(
            f"no site data recorded for interval {k}; the shadow cannot be "
            f"advanced without the solar and load the LP saw"
        )

    def _settle_block(self) -> None:
        """Close the open block and apply retrospective import settlement.

        Only the natural/forced split moves. Total import per interval is
        untouched, so dispatch, SOC and the energy component of the bill
        cannot change. Distribution across the block is proportional to
        committed import, which is exact for billing because the reward rates
        are flat in $/kWh and only the block totals reach
        retail_other_credits.
        """
        k0 = self.shadow.k0
        n_block = self.shadow._n_committed
        n_imp_settled, f_imp_settled = self.shadow.settle()

        if not self._block_had_spike:
            self._blocks_no_spike += 1

        if n_block <= 0:
            return
        sl = slice(k0, k0 + n_block)
        imports = self.n_imp[sl] + self.f_imp[sl]
        total = float(imports.sum())
        if total <= 0.0:
            pre_f_imp = float(self.f_imp[sl].sum())
            self.n_imp[sl] = 0.0
            self.f_imp[sl] = 0.0
            self.cap.apply_settlement_delta(k0 + n_block - 1, -pre_f_imp)
            return
        share = imports / total
        pre_f_imp = float(self.f_imp[sl].sum())
        self.n_imp[sl] = n_imp_settled * share
        self.f_imp[sl] = f_imp_settled * share
        # The cap was decremented per interval from the pre-settlement split.
        # Settlement has just rewritten it, so the cap must follow or the
        # constraint and the bill disagree by exactly this delta.
        self.cap.apply_settlement_delta(
            k0 + n_block - 1, f_imp_settled - pre_f_imp)

    def note_spike(self, settled_above_threshold: bool) -> None:
        """Record that a settled price in the current block exceeded the
        threshold.

        Not called anywhere: the planner only ever sees forecasts, so the
        in-run count of blocks without a spike counts every closed block.
        The reported event_blocks_no_spike is computed after the run by
        rolling_vpp_event.count_blocks_without_a_settled_spike from settled
        prices, and overwrites the in-run value."""
        if settled_above_threshold:
            self._block_had_spike = True
