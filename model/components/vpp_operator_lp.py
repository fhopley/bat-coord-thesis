"""The operator event LP in rolling form.

The LP itself is `vpp_sc_event._solve_event_lp`, unchanged and imported rather
than reimplemented. Under perfect foresight it is solved once over a known
event window; under rolling it is solved once per interval over the span the
operator can currently see, and only the first interval is committed. The
formulation is identical either way, so sharing the builder is what keeps the
two paths from drifting apart -- the objective, the flow balance, the SOC
continuity, the cap row and the no-grid-charge coupling all have exactly one
definition.

What this module adds is the two things rolling needs and perfect foresight
does not.

Horizon
-------
The LP runs `[k, t_star]`, where `t_star` is the last forecast spike inside the
declaration lookahead. Beyond it the operator has nothing to dispatch into, and
whether the event continues past it is the next interval's question, decided
against the next forecast. Planning further would be planning against an event
extension that has not been declared. There is no terminal SOC pin: the
operator dispatches on event economics and leaves whatever remains, and the
backstop against emptying the battery at low value is the minimum export price,
not a terminal constraint. `soc_final` from the rolling loop is therefore
ignored on this path -- the only place a planner ignores an argument the loop
computes.

Layout
------
`rolling_optimisation` reads the committed interval at fixed offsets on
`n`-sized blocks: charge, discharge, import, export, soc, curtail at
`0, n, 2n, 3n, 4n, 5n`. The event LP's layout is different -- eight blocks with
natural and forced flows separate -- so the solution is repacked. This is
mechanical, not a modelling choice: a vector laid out on `n_eff`-sized blocks
would put `x[n]` in the middle of the discharge block and the loop would
silently commit the wrong numbers.

The four split blocks are appended after `x[6n:]`, the same way
AdvancedWindowPlanner appends its tier variables. The loop never learns what is
there; the VPP planner reads them back to build VppTraces.

Feasibility
-----------
`vpp_sc_event` Section 4 guarantees the LP is always feasible, because its
bounds come from a self-consumption run started at the same SOC, making the SC
dispatch a valid assignment. That guarantee does not survive the move to
rolling: the shadow is anchored at the event's first interval and advances on
its own SOC, so it diverges from the real battery and SC is no longer a
certificate. A window can genuinely be infeasible -- battery full, surplus
solar, curtailment pinned to zero, and export blocked by a finite export limit,
an exhausted cap, or the minimum export price.

`_solve_event_lp` emits an `EOV_SKIP` warning on an infeasible solve, whose
wording calls this a bug because on the perfect-foresight path it is one. On
this path it is an expected condition: the caller falls back to
self-consumption for that interval and counts it. The warning token is
unchanged because orchestrate.py and three test modules match on it.

See rolling_vpp_event_spec.md, "The operator LP".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.core.scenario import BatterySpec
from model.components.vpp_sc_event import _solve_event_lp


@dataclass(frozen=True)
class OperatorSolution:
    """A packed operator solve, ready for the rolling loop.

    Attributes:
        x: Solution vector of length 10*n. The first 6*n is the loop's block
            layout; the remaining 4*n holds n_imp, f_imp, n_exp, f_exp.
        n_eff: Intervals actually solved. Positions n_eff..n-1 of every block
            are zero, except SOC which holds flat at its final solved value.
            Recorded so a flat tail is never read as a plan.
        committed: The first interval's flows, the only ones that will be
            committed. Keys: charge, discharge, import, export, soc, curtail,
            n_imp, f_imp, n_exp, f_exp.
    """
    x: np.ndarray
    n_eff: int
    committed: dict[str, float]


# HiGHS reports a primal solution to a feasibility tolerance, so a variable at
# a zero lower bound can come back fractionally negative -- -1.2e-14 was seen
# on a real site-year. Downstream that is not cosmetic: CapState rejects a
# negative forced flow, and VppTraces documents every series as non-negative.
# Set to HiGHS's own primal feasibility tolerance: a value the solver is
# entitled to return is noise by definition, and anything beyond it is a real
# bound problem that must surface rather than be silently absorbed. An earlier
# 1e-9 was tighter than the solver's own guarantee and rejected legitimate
# residue of -1.1e-9 mid-run.
_FLOW_RESIDUE_TOL = 1e-7


def _clamp_flow_residue(x_event: np.ndarray, e: int) -> np.ndarray:
    """Clamp tiny negative flow residues to zero, leaving SOC untouched.

    Applied before the blocks are summed into the headline import and export
    series, so the split and the headline stay exactly consistent: import is
    derived from the clamped n_imp and f_imp, not from the raw solution.

    SOC is skipped -- it is a state with its own bounds, not a flow, and a
    negative SOC would never be near zero in any case.
    """
    out = x_event.copy()
    flow = np.r_[0:6*e, 7*e:8*e]          # every block except soc
    vals = out[flow]
    worst = vals.min() if vals.size else 0.0
    if worst < -_FLOW_RESIDUE_TOL:
        raise ValueError(
            f"operator LP returned a flow of {worst:.6g}, beyond the "
            f"{_FLOW_RESIDUE_TOL:.0e} tolerance for solver residue. This is "
            f"not rounding noise -- check the variable bounds."
        )
    out[flow] = np.maximum(0.0, vals)
    return out


def pack_to_loop_layout(x_event: np.ndarray, n_eff: int, n: int) -> np.ndarray:
    """Repack an 8-block event LP solution into the loop's layout.

    Args:
        x_event: Raw solution from _solve_event_lp, length 8 * n_eff, blocks
            in the order charge, discharge, n_imp, f_imp, n_exp, f_exp, soc,
            curtail.
        n_eff: Intervals solved.
        n: The loop's window length. n_eff <= n.

    Returns:
        Vector of length 10 * n.
    """
    if n_eff < 1:
        raise ValueError(f"n_eff must be >= 1, got {n_eff}")
    if n_eff > n:
        raise ValueError(f"n_eff={n_eff} exceeds window length n={n}")
    if x_event.size != 8 * n_eff:
        raise ValueError(
            f"x_event has {x_event.size} entries, expected {8 * n_eff} "
            f"for n_eff={n_eff}"
        )

    e = n_eff
    x_event = _clamp_flow_residue(x_event, e)
    charge, discharge = x_event[0:e], x_event[e:2*e]
    n_imp, f_imp = x_event[2*e:3*e], x_event[3*e:4*e]
    n_exp, f_exp = x_event[4*e:5*e], x_event[5*e:6*e]
    soc, curtail = x_event[6*e:7*e], x_event[7*e:8*e]

    out = np.zeros(10 * n)
    out[0*n:0*n + e] = charge
    out[1*n:1*n + e] = discharge
    out[2*n:2*n + e] = n_imp + f_imp        # loop reads total import here
    out[3*n:3*n + e] = n_exp + f_exp        # and total export here
    out[4*n:4*n + e] = soc
    out[5*n:5*n + e] = curtail
    # SOC is a state, not a flow: holding it flat past the solved span keeps
    # the tail physically readable. Every other block stays zero.
    if e < n:
        out[4*n + e:5*n] = soc[-1]
    out[6*n:6*n + e] = n_imp
    out[7*n:7*n + e] = f_imp
    out[8*n:8*n + e] = n_exp
    out[9*n:9*n + e] = f_exp
    return out


def read_committed(x: np.ndarray, n: int) -> dict[str, float]:
    """The committed interval's flows from a packed vector."""
    if x.size != 10 * n:
        raise ValueError(
            f"x has {x.size} entries, expected {10 * n} for n={n}")
    return {
        'charge': float(x[0*n]),
        'discharge': float(x[1*n]),
        'import': float(x[2*n]),
        'export': float(x[3*n]),
        'soc': float(x[4*n]),
        'curtail': float(x[5*n]),
        'n_imp': float(x[6*n]),
        'f_imp': float(x[7*n]),
        'n_exp': float(x[8*n]),
        'f_exp': float(x[9*n]),
    }


def solve_operator_window(
    solar_window: np.ndarray,
    load_window: np.ndarray,
    op_imp: np.ndarray,
    op_exp: np.ndarray,
    n: int,
    n_eff: int,
    n_imp_budget: float,
    n_exp_hi: np.ndarray,
    battery_spec: BatterySpec,
    soc_entry: float,
    vpp_exp_reward: float,
    vpp_imp_reward: float,
    remaining_cap: float,
    vpp_limit_applies_to: str,
    export_limit: float,
    interval_hours: float,
    site_id: str,
    k: int,
    p_curtail: float | None = None,
    min_export_price_per_kwh: float = 0.0,
    no_grid_charge_window: np.ndarray | None = None,
) -> OperatorSolution | None:
    """Solve the operator LP over [k, k + n_eff - 1] and pack the result.

    Args:
        solar_window, load_window: The arrays the loop supplied, length >= n.
            Position 0 carries actuals under commit_actual_load.
        op_imp, op_exp: Operator import and export prices in $/kWh over the
            window, built from the FORECAST wholesale price through the
            operator tariff's price stack. Length >= n_eff.
        n: The loop's window length, which sets the packed layout.
        n_eff: Intervals to solve, t_star - k + 1 from the declaration.
        n_imp_budget: Aggregate natural import allowance for the remainder of
            the event, from ShadowBounds. Already clamped at zero.
        n_exp_hi: Per-interval natural export ceiling, length >= n_eff.
        soc_entry: The real battery's SOC entering interval k.
        remaining_cap: The site's remaining annual cap.
        min_export_price_per_kwh: Threshold below which f_exp is pinned to
            zero. Applied to the FORECAST op_exp, so the operator may forgo
            export at an interval that settles above the threshold, or export
            at one that settles below. That is an information effect this path
            exists to measure, not an error.

    Returns:
        OperatorSolution, or None when the LP is infeasible. The caller falls
        back to self-consumption for that interval and counts it.
    """
    if n_eff < 1 or n_eff > n:
        raise ValueError(f"n_eff must be in [1, {n}], got {n_eff}")
    for name, arr in (('solar_window', solar_window),
                      ('load_window', load_window),
                      ('op_imp', op_imp), ('op_exp', op_exp),
                      ('n_exp_hi', n_exp_hi)):
        if len(arr) < n_eff:
            raise ValueError(
                f"{name} covers {len(arr)} intervals, need at least "
                f"n_eff={n_eff}"
            )

    e = n_eff
    x_event = _solve_event_lp(
        solar=np.asarray(solar_window[:e], dtype=float),
        load=np.asarray(load_window[:e], dtype=float),
        op_imp=np.asarray(op_imp[:e], dtype=float),
        op_exp=np.asarray(op_exp[:e], dtype=float),
        sc_n_imp_total=float(n_imp_budget),
        n_exp_hi=np.asarray(n_exp_hi[:e], dtype=float),
        bspec=battery_spec,
        soc_entry=float(soc_entry),
        vpp_exp_reward=vpp_exp_reward,
        vpp_imp_reward=vpp_imp_reward,
        remaining_cap=float(remaining_cap),
        vpp_limit_applies_to=vpp_limit_applies_to,
        export_limit=export_limit,
        interval_hours=interval_hours,
        site_id=site_id,
        win_start=k,
        p_curtail=p_curtail,
        min_export_price_per_kwh=min_export_price_per_kwh,
        no_grid_charge_ew=(None if no_grid_charge_window is None
                           else np.asarray(no_grid_charge_window[:e], dtype=bool)),
    )
    if x_event is None:
        return None

    x = pack_to_loop_layout(x_event, n_eff=e, n=n)
    return OperatorSolution(x=x, n_eff=e, committed=read_committed(x, n))
