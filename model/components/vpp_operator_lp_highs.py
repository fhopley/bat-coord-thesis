"""The operator event LP as a persistent HiGHS model.

`vpp_operator_lp.solve_operator_window` builds a fresh model and calls
`scipy.optimize.linprog` at every decision instant. That is correct, and is the
reference this module is checked against, but it dominates the run: measured at
2.79 ms for a 10-interval span and 6.80 ms at 120, against roughly 8,800 event
intervals in a site-year.

Where the cost actually is
--------------------------
Assembling the sparse matrix takes 0.25 ms -- between 3 and 8 per cent of a
solve. So rebuilding the matrix is NOT the cost, and a version of this module
that only avoided the rebuild was measured at 0.6x to 1.0x, i.e. slower. The
cost is the solve, and specifically the fact that `linprog` constructs a fresh
HiGHS instance per call and throws the basis away.

Consecutive windows differ by one interval of shift and a data refresh, so the
previous optimal basis is nearly optimal for the next problem. Retaining it is
the lever. This is the same reason `rolling_optimisation._WindowSolver` talks
to `highspy` directly rather than through scipy, and this module follows its
structure deliberately: matrix built once, `changeColsCost` /
`changeColsBounds` / `changeRowsBounds` between windows, index vectors
preallocated so the hot path allocates nothing.

Fixed size, deactivated tail
----------------------------
The model is built at `W` intervals -- the event window length -- not at
`n_eff`. A window whose event span is shorter deactivates its tail:

    for j >= n_eff:
        every flow variable bounded to [0, 0]
        flow balance RHS (load - solar) set to 0

With the flows pinned and the balance RHS zero, the tail rows are satisfied by
the zero vector and SOC continuity carries `soc[n_eff-1]` forward unchanged.
The tail is feasible, inert, and contributes nothing to the objective, so
solving `W` intervals with a dead tail is exactly equivalent to solving
`n_eff`. That equivalence is what makes the fixed size sound, and it is
asserted against the from-scratch builder rather than assumed.

`_WindowSolver` is also fixed-size and always solves its full horizon, so this
is not a new compromise. Whether the dead tail is cheap under warm starting is
an empirical question and is measured, not asserted.

Two rows need care to keep the structure invariant:

  - No-grid-charge windows differ between windows, so the rows are always
    present and made non-binding where the mask is off, by raising the RHS
    rather than dropping the row.
  - The minimum export price pins `f_exp` through a variable BOUND, never a
    row, so a changing set of blocked intervals never changes the matrix.

State leak is the hazard of reusing a solver. Every bound and right-hand side
this class touches is rewritten in full on every call, including the dead tail
and including bounds that happen to be unchanged, so no window can inherit a
constraint from its predecessor. `test_repeated_solves_do_not_leak_state`
exists to catch a regression in exactly that.

See rolling_vpp_event_spec.md, "The operator LP" > "Solver".
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from model.core.scenario import BatterySpec
from model.components.vpp_operator_lp import (
    OperatorSolution, pack_to_loop_layout, read_committed,
)

try:
    import highspy
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "vpp_operator_lp_highs requires highspy. Install with "
        "`pip install highspy`. It is the same HiGHS solver "
        "scipy.optimize.linprog already calls, used directly so the basis "
        "can be carried between windows."
    ) from exc


class HighsOperatorLp:
    """One reusable HiGHS operator LP for a single site.

    Args:
        w: Event window length in intervals. The model is built at this size;
            every solve deactivates the tail beyond n_eff.
        has_no_grid_charge: Whether to include the no-grid-charge rows. They
            cost a row per interval, so a plan without such windows omits them
            rather than carrying rows that are always slack.
    """

    def __init__(
        self,
        w: int,
        battery_spec: BatterySpec,
        interval_hours: float,
        vpp_limit_applies_to: str,
        has_no_grid_charge: bool = False,
        p_curtail: float | None = None,
        export_limit: float = np.inf,
    ):
        if w < 1:
            raise ValueError(f"w must be >= 1, got {w}")
        if vpp_limit_applies_to not in ('export_only', 'import_and_export'):
            raise ValueError(
                f"vpp_limit_applies_to must be 'export_only' or "
                f"'import_and_export', got {vpp_limit_applies_to!r}")

        self.W = W = int(w)
        self.bspec = battery_spec
        self.interval_hours = interval_hours
        self.limit_applies_to = vpp_limit_applies_to
        self.has_ngc = bool(has_no_grid_charge)
        self.p_curtail = p_curtail
        self.export_limit = export_limit
        self.solves = 0

        self.capacity = battery_spec.capacity_kwh
        self.soc_min = (battery_spec.backup_reserve_pct / 100.0) * self.capacity
        self.power_kwh = battery_spec.power_kw * interval_hours
        self.eta_c = battery_spec.charge_efficiency
        self.eta_d = battery_spec.discharge_efficiency

        inf = highspy.kHighsInf
        self._inf = inf

        # Block order matches vpp_sc_event._solve_event_lp exactly.
        self.OC, self.OD = 0, W
        self.ONI, self.OFI = 2 * W, 3 * W
        self.ONE, self.OFE = 4 * W, 5 * W
        self.OS, self.OQ = 6 * W, 7 * W
        N = 8 * W

        c = np.zeros(N)
        c[self.OC:self.OC + W] = battery_spec.c_deg_per_kwh
        c[self.OD:self.OD + W] = battery_spec.c_deg_per_kwh
        if p_curtail:
            c[self.OQ:self.OQ + W] = p_curtail

        lo = np.zeros(N)
        hi = np.full(N, inf)
        hi[self.OC:self.OC + W] = self.power_kwh
        hi[self.OD:self.OD + W] = self.power_kwh
        lo[self.OS:self.OS + W] = self.soc_min
        hi[self.OS:self.OS + W] = self.capacity
        if p_curtail is None:
            hi[self.OQ:self.OQ + W] = 0.0

        A, n_rows, self._row = self._build_matrix()
        self._n_rows = n_rows

        row_lo = np.full(n_rows, -inf)
        row_hi = np.zeros(n_rows)
        row_lo[:2 * W] = 0.0        # equalities, RHS set per solve

        lp = highspy.HighsLp()
        lp.num_col_ = N
        lp.num_row_ = n_rows
        lp.col_cost_ = c
        lp.col_lower_ = lo
        lp.col_upper_ = hi
        lp.row_lower_ = row_lo
        lp.row_upper_ = row_hi
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr.astype(np.int32)
        lp.a_matrix_.index_ = A.indices.astype(np.int32)
        lp.a_matrix_.value_ = A.data

        self._h = highspy.Highs()
        self._h.setOptionValue('output_flag', False)
        self._h.passModel(lp)

        # Preallocated index vectors and buffers: the hot path allocates
        # nothing, matching _WindowSolver.
        self._price_cols = np.arange(self.ONI, self.ONI + 4 * W,
                                     dtype=np.int32)
        self._flow_cols = np.concatenate([
            np.arange(off, off + W, dtype=np.int32)
            for off in (self.OC, self.OD, self.ONI, self.OFI,
                        self.ONE, self.OFE, self.OQ)])
        self._soc_cols = np.arange(self.OS, self.OS + W, dtype=np.int32)
        self._eq_rows = np.arange(2 * W, dtype=np.int32)
        self._ub_rows = np.arange(2 * W, n_rows, dtype=np.int32)
        self._price_buf = np.empty(4 * W)
        self._flow_lo = np.zeros(self._flow_cols.size)
        self._flow_hi = np.empty(self._flow_cols.size)
        self._eq_rhs = np.empty(2 * W)
        self._ub_lo = np.full(n_rows - 2 * W, -inf)
        self._ub_hi = np.empty(n_rows - 2 * W)
        self._soc_lo = np.full(W, self.soc_min)
        self._soc_hi = np.full(W, self.capacity)

    # -- structure ----------------------------------------------------------

    def _build_matrix(self):
        """Assemble A once. Depends only on W, the efficiencies and which
        optional row groups are present."""
        W = self.W
        t = np.arange(W, dtype=np.intp)
        rows, cols, data = [], [], []

        # Flow balance, rows [0, W).
        rows.append(np.tile(t, 7))
        cols.append(np.concatenate([self.OC + t, self.OD + t, self.ONI + t,
                                    self.OFI + t, self.ONE + t, self.OFE + t,
                                    self.OQ + t]))
        data.append(np.concatenate([
            -np.ones(W), np.ones(W), np.ones(W), np.ones(W),
            -np.ones(W), -np.ones(W), -np.ones(W)]))

        # SOC continuity, rows [W, 2W).
        rows.append(np.concatenate([W + t, W + t, W + t]))
        cols.append(np.concatenate([self.OC + t, self.OD + t, self.OS + t]))
        data.append(np.concatenate([
            np.full(W, -self.eta_c), np.full(W, 1.0 / self.eta_d),
            np.ones(W)]))
        if W > 1:
            t1 = np.arange(1, W, dtype=np.intp)
            rows.append(W + t1)
            cols.append(self.OS + t1 - 1)
            data.append(-np.ones(W - 1))

        nxt = 2 * W
        index = {}

        # Natural import budget: sum(n_imp) <= B_k.
        index['budget'] = nxt
        rows.append(np.full(W, nxt, dtype=np.intp))
        cols.append(self.ONI + t)
        data.append(np.ones(W))
        nxt += 1

        # Annual cap.
        cap_cols = (self.OFE + t if self.limit_applies_to == 'export_only'
                    else np.concatenate([self.OFI + t, self.OFE + t]))
        index['cap'] = nxt
        rows.append(np.full(cap_cols.size, nxt, dtype=np.intp))
        cols.append(cap_cols)
        data.append(np.ones(cap_cols.size))
        nxt += 1

        # Export limit per interval: n_exp + f_exp <= limit.
        index['export'] = nxt
        rows.append(np.tile(nxt + t, 2))
        cols.append(np.concatenate([self.ONE + t, self.OFE + t]))
        data.append(np.ones(2 * W))
        nxt += W

        # No-grid-charge: charge + curtail <= max(0, solar - load).
        if self.has_ngc:
            index['ngc'] = nxt
            rows.append(np.tile(nxt + t, 2))
            cols.append(np.concatenate([self.OC + t, self.OQ + t]))
            data.append(np.ones(2 * W))
            nxt += W

        A = sp.csc_matrix(
            (np.concatenate(data),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(nxt, 8 * W))
        return A, nxt, index

    # -- per-window ----------------------------------------------------------

    def solve(
        self, solar, load, op_imp, op_exp, n, n_eff, n_imp_budget, n_exp_hi,
        soc_entry, vpp_exp_reward, vpp_imp_reward, remaining_cap,
        min_export_price_per_kwh=0.0, no_grid_charge_window=None,
    ) -> OperatorSolution | None:
        """Refresh data for one window and re-solve, retaining the basis."""
        W, e = self.W, int(n_eff)
        if not (1 <= e <= W):
            raise ValueError(f"n_eff must be in [1, {W}], got {e}")
        if e > n:
            raise ValueError(f"n_eff={e} exceeds window length n={n}")

        solar = np.asarray(solar[:e], dtype=float)
        load = np.asarray(load[:e], dtype=float)
        op_imp = np.asarray(op_imp[:e], dtype=float)
        op_exp = np.asarray(op_exp[:e], dtype=float)
        n_exp_hi = np.asarray(n_exp_hi[:e], dtype=float)
        h, inf = self._h, self._inf

        # Objective: the four priced blocks. c_deg and p_curtail never move.
        buf = self._price_buf
        buf[0:e] = op_imp
        buf[e:W] = 0.0
        buf[W:W + e] = op_imp + vpp_imp_reward
        buf[W + e:2 * W] = 0.0
        buf[2 * W:2 * W + e] = -op_exp
        buf[2 * W + e:3 * W] = 0.0
        buf[3 * W:3 * W + e] = -(op_exp - vpp_exp_reward)
        buf[3 * W + e:4 * W] = 0.0
        h.changeColsCost(4 * W, self._price_cols, buf)

        # Flow bounds, written in full so nothing survives from the last call.
        fhi = self._flow_hi
        fhi[0:e] = self.power_kwh                       # charge
        fhi[e:W] = 0.0
        fhi[W:W + e] = self.power_kwh                   # discharge
        fhi[W + e:2 * W] = 0.0
        fhi[2 * W:2 * W + e] = inf                      # n_imp
        fhi[2 * W + e:3 * W] = 0.0
        fhi[3 * W:3 * W + e] = inf                      # f_imp
        fhi[3 * W + e:4 * W] = 0.0
        fhi[4 * W:4 * W + e] = n_exp_hi                 # n_exp
        fhi[4 * W + e:5 * W] = 0.0
        fhi[5 * W:5 * W + e] = np.where(                # f_exp
            op_exp < min_export_price_per_kwh, 0.0, inf)
        fhi[5 * W + e:6 * W] = 0.0
        if self.p_curtail is None:
            fhi[6 * W:7 * W] = 0.0                      # curtail pinned off
        else:
            # Clamped at zero: solar is genuinely negative at some intervals
            # (overnight inverter parasitic draw -- see SiteTraces on
            # sol_to_home_kwh). An unclamped negative upper bound against a
            # zero lower bound is an infeasible variable, which made 8,106 of
            # 8,296 operator windows on a real site-year infeasible while
            # every synthetic test passed, because generated solar was never
            # negative. Both reference implementations clamp the same way:
            # vpp_sc_event._solve_event_lp and _WindowSolver.solve.
            fhi[6 * W:6 * W + e] = np.maximum(0.0, solar)
            fhi[6 * W + e:7 * W] = 0.0
        h.changeColsBounds(self._flow_cols.size, self._flow_cols,
                           self._flow_lo, fhi)

        # SOC bounds are constant, but rewritten so a future change to them
        # cannot leak across windows.
        h.changeColsBounds(W, self._soc_cols, self._soc_lo, self._soc_hi)

        # Equality RHS: flow balance then SOC continuity.
        rhs = self._eq_rhs
        rhs[0:e] = load - solar
        rhs[e:W] = 0.0
        rhs[W:] = 0.0
        rhs[W] = soc_entry
        h.changeRowsBounds(2 * W, self._eq_rows, rhs, rhs)

        # Inequality RHS.
        ub = self._ub_hi
        r = self._row
        ub[r['budget'] - 2 * W] = max(0.0, float(n_imp_budget))
        ub[r['cap'] - 2 * W] = max(0.0, float(remaining_cap))
        xo = r['export'] - 2 * W
        ub[xo:xo + W] = (self.export_limit
                         if np.isfinite(self.export_limit) else inf)
        if self.has_ngc:
            go = r['ngc'] - 2 * W
            ub[go:go + W] = inf
            if no_grid_charge_window is not None:
                mask = np.asarray(no_grid_charge_window[:e], dtype=bool)
                idx = np.flatnonzero(mask)
                if idx.size:
                    ub[go + idx] = np.maximum(0.0, solar - load)[idx]
        h.changeRowsBounds(self._n_rows - 2 * W, self._ub_rows,
                           self._ub_lo, ub)

        h.run()
        self.solves += 1
        if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None

        x_full = np.asarray(h.getSolution().col_value, dtype=float)
        x_event = np.empty(8 * e)
        for i, off in enumerate((self.OC, self.OD, self.ONI, self.OFI,
                                 self.ONE, self.OFE, self.OS, self.OQ)):
            x_event[i * e:(i + 1) * e] = x_full[off:off + e]

        x = pack_to_loop_layout(x_event, n_eff=e, n=n)
        return OperatorSolution(x=x, n_eff=e, committed=read_committed(x, n))
