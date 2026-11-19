"""Shared window assembly. Every provider runs this; only run supply differs.

Four stages, per forecast_providers_spec.md:

    1. select runs   latest fine and coarse runs available at the decision instant
    2. splice        coarse expanded flat to 5 min, fine overlaid on top
    3. infill        extrapolate beyond coarse coverage at a fixed lag
    4. assert        causality and shape, then return

Splice, infill, mode selection and the causality assertion are where subtle
errors hide, so they exist once here and are tested once, rather than being
reimplemented per source.

There is no price clipping. Bounding forecasts to the market price range was
considered and rejected: the cap changes annually, which would put
year-indexed knowledge in a path executed ~105,000 times per site-year, and
nothing in this subsystem manufactures an out-of-range value. Range validation
belongs in store ingest, where the correct cap for each year is known and the
check runs once.
"""
from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from model.forecast.runset import RunSet

CurrentIntervalPrice = Literal['forecast', 'price_known', 'price_and_next',
                               'next_forecast']
_MODES = ('forecast', 'price_known', 'price_and_next', 'next_forecast')


class AssembledWindow(NamedTuple):
    """One assembled window, plus the provenance a diagnostic needs.

    A named tuple rather than a bare tuple so that adding provenance does not
    churn every call site again. Callers should use attribute access.

    Attributes:
        rrp: Forecast RRP in $/MWh, in target order.
        is_infill: True where the value was extrapolated rather than forecast.
        is_fine: True where the value came from the fine segment. Recorded
            here rather than derived downstream because the boundary moves
            with the mode allowance and with fine-run staleness.
        run_pos: Interval position of the selected fine run, or of the coarse
            run when there is no fine segment. This is the causality-relevant
            label and the freshest thing in the window.
        coarse_run_pos: Interval position of the selected coarse run. Can be
            up to one coarse cadence older than run_pos, and is what shapes
            everything beyond the first hour.

    rrp, is_infill and is_fine may be SHORTER than the requested n_intervals
    when infill cannot reach far enough back into history.
    """
    rrp: np.ndarray
    is_infill: np.ndarray
    is_fine: np.ndarray
    run_pos: int
    coarse_run_pos: int

# Modes whose committed interval is priced at the settled dispatch price rather
# than at a forecast of it. Measured: AEMO writes the price for an interval
# about 4.85 minutes BEFORE that interval ends, i.e. ~9 seconds after it
# begins, so a controller genuinely has it with almost the whole interval left
# to act. This is not a bound; it is what a real battery can do.
_USES_ACTUAL_FOR_COMMITTED = ('price_known', 'price_and_next')


def uses_actual_for_committed(mode: str) -> bool:
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {_MODES}")
    return mode in _USES_ACTUAL_FOR_COMMITTED


def mode_allowance_intervals(mode: str) -> int:
    """Intervals after the decision instant a run label may legitimately carry.

    Zero for modes that use only the run available when the interval begins;
    one interval for modes that also use the run arriving ~39 s in.

    The three coherent information sets within interval k, by the moment they
    become actionable:

        0 s     `forecast`        the run labelled index[k-1], which forecast
                                  interval k five minutes ahead
        ~9 s    `price_known`     + the settled dispatch price for k, which
                                  AEMO publishes ~9 s after k begins
        ~39 s   `price_and_next`  + the run labelled index[k], refreshing the
                                  whole forward window

    `next_forecast` is retained for decomposition but is NOT a controller
    anybody would build: it uses the run arriving at ~39 s while pricing
    interval k at that run's own estimate, ignoring the settled price it has
    had for 30 seconds. P5MIN makes approximations to speed computation and is
    not the same algorithm as dispatch, so its lead-0 row is measurably worse
    than the published price it duplicates (MAE $70/MWh, $1,707 on spikes).
    Its only use is isolating that approximation cost against `price_and_next`
    with the forward window held constant.

    P5MIN publishes about 4.35 minutes before its nominal label, so the run
    labelled index[k] arrives roughly 0.65 minutes into interval k, carrying a
    lead-0 estimate for the interval in progress plus refreshed forward
    prices. A real controller can act on it within the interval.

    This is expressed as a per-mode allowance rather than as slack in the
    general causality rule so that it stays a narrow, named exception. The
    worst publication lead observed was 3.12 minutes, which still lands inside
    interval k, so the allowance holds at the tail of the distribution and not
    just at the mean.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {_MODES}")
    return 1 if mode in ('next_forecast', 'price_and_next') else 0


def expand_coarse(target_pos: np.ndarray, values: np.ndarray,
                  resolution_intervals: int) -> tuple[np.ndarray, np.ndarray]:
    """Expand coarse targets flat across their constituent 5-minute intervals.

    A 30-minute target labelled T covers the six 5-minute intervals labelled
    T-25min .. T, following the interval-end convention used throughout the
    model. Confirmed against the data: the half hour labelled 14:30 covers the
    5-minute actuals labelled 14:05 to 14:30.

    Flat expansion leaves the LP indifferent about placement within a block.
    That degeneracy only ever reaches tail intervals, because the committed
    interval always sits inside the fine segment.
    """
    r = int(resolution_intervals)
    if r == 1:
        return target_pos, values
    back = np.arange(r - 1, -1, -1, dtype=np.int64)          # r-1 .. 0
    pos = (target_pos[:, None] - back[None, :]).ravel()
    val = np.repeat(values, r)
    return pos, val


def assemble_window(
    *,
    k: int,
    n_intervals: int,
    coarse: RunSet,
    fine: RunSet | None,
    actuals: np.ndarray,
    infill_lag_intervals: int,
    mode: str = 'forecast',
) -> "AssembledWindow | None":
    """Assemble one forecast window.

    Args:
        k: Interval position of the window's first target, i.e. the interval
            about to be committed. The decision instant is the interval-end
            label of position k-1.
        n_intervals: Requested window length.
        coarse: Coarse segment runs. Required.
        fine: Fine segment runs, or None. Optional by design: the coarse
            segment alone spans the whole window from the decision instant, so
            a missing fine run degrades resolution over the first hour rather
            than invalidating the forecast.
        actuals: Settled prices for the whole period, indexed by interval
            position. Read only for infill, and only strictly before the
            decision instant.
        infill_lag_intervals: Lag L used to extrapolate beyond forecast
            coverage.
        mode: Current-interval price mode.

    Returns:
        An AssembledWindow, or None when no run is available at or before the
        selection instant.
    """
    if k < 1:
        raise ValueError(
            f"k must be >= 1: position k-1 is the decision instant, so a "
            f"window cannot start at the first interval of the period (got {k})"
        )
    if n_intervals < 1:
        raise ValueError(f"n_intervals must be >= 1, got {n_intervals}")
    if infill_lag_intervals < 1:
        raise ValueError(
            f"infill_lag_intervals must be >= 1, got {infill_lag_intervals}")

    # ---- stage 1: select runs -------------------------------------------
    decision_pos = k - 1
    select_pos = decision_pos + mode_allowance_intervals(mode)

    ci = coarse.latest_at(select_pos)
    if ci is None:
        return None
    fi = fine.latest_at(select_pos) if fine is not None else None

    # ---- stage 2: splice -------------------------------------------------
    # Paint into a buffer that starts ONE interval early, at the decision
    # instant. buf[0] is the interval ending at the decision instant, which is
    # not part of the window (it is already committed) but is the one piece of
    # known history that cannot come from settled actuals, because its price
    # is not published yet. Carrying it here means whichever segment covers it
    # supplies it, with fine winning, and infill needs no special case.
    buf = np.full(n_intervals + 1, np.nan, dtype=np.float64)
    is_infill = np.zeros(n_intervals, dtype=bool)
    fine_buf = np.zeros(n_intervals + 1, dtype=bool)

    c_run, c_targets, c_values = coarse.run(ci)
    c_pos, c_val = expand_coarse(c_targets, c_values, coarse.resolution_intervals)
    _paint(buf, c_pos, c_val, decision_pos, n_intervals + 1)

    run_pos = c_run
    coarse_run_pos = c_run
    if fi is not None:
        f_run, f_targets, f_values = fine.run(fi)
        f_pos, f_val = expand_coarse(f_targets, f_values, fine.resolution_intervals)
        # fine wins, and reports which slots it took, which is the splice
        # boundary. Infill cannot muddy this: it only fills slots nothing
        # painted, so those stay False without a special case.
        fine_buf[_paint(buf, f_pos, f_val, decision_pos, n_intervals + 1)] = True
        run_pos = f_run

    lead0_value = float(buf[0])
    out = buf[1:]
    is_fine = fine_buf[1:]

    # ---- stage 3: infill -------------------------------------------------
    n_filled = _infill(
        out, is_infill, k, n_intervals, actuals,
        infill_lag_intervals, decision_pos, lead0_value,
    )
    if n_filled == 0:
        return None
    if n_filled < n_intervals:
        out, is_infill = out[:n_filled], is_infill[:n_filled]
        is_fine = is_fine[:n_filled]

    # ---- stage 4: assert -------------------------------------------------
    if run_pos > select_pos:
        raise AssertionError(
            f"causality violation: selected run at position {run_pos} is "
            f"later than the permitted selection position {select_pos} "
            f"(decision position {decision_pos}, mode {mode!r})"
        )
    if not np.isfinite(out).all():
        raise AssertionError(
            f"{int((~np.isfinite(out)).sum())} unfilled interval(s) survived "
            "assembly; splice and infill must jointly cover the window"
        )
    return AssembledWindow(rrp=out, is_infill=is_infill, is_fine=is_fine,
                           run_pos=run_pos, coarse_run_pos=coarse_run_pos)


def _paint(buf: np.ndarray, pos: np.ndarray, val: np.ndarray,
           base: int, n: int) -> np.ndarray:
    """Write values whose positions fall inside [base, base+n).

    Called with base = decision_pos, so targets strictly before the decision
    instant are dropped. The lead-0 row lands in buf[0] and is trimmed off the
    returned window, which is what removes it from the plan while keeping it
    available to infill.

    Returns the buffer slots written, so a caller can record which segment
    supplied which position without recomputing the overlap arithmetic.
    """
    m = (pos >= base) & (pos < base + n)
    if not m.any():
        return np.empty(0, dtype=np.int64)
    idx = pos[m] - base
    buf[idx] = val[m]
    return idx


def _infill(out: np.ndarray, is_infill: np.ndarray, k: int, n: int,
            actuals: np.ndarray, L: int, decision_pos: int,
            lead0_value: float) -> int:
    """Fill gaps at lag L, on the assumption that tomorrow looks like today.

    A single forward pass resolves the recursion to any depth: because L > 0,
    position j always sources from j-L which is lower and therefore already
    settled. No explicit recursion is needed.

    For j-L < 0 the source lies at or before the decision instant, so it comes
    from known history:

        source position <= decision_pos - 1   settled actual
        source position == decision_pos       fine run's lead-0 estimate

    Past is not the same as published. The settled price for the interval
    ending at the decision instant is not available at that instant, which is
    why the second case exists rather than reading actuals[decision_pos]. At
    the default parameters exactly one window position (j = L-1) lands there.

    Returns the number of leading positions successfully filled. A shortfall
    means history does not reach back far enough, which happens only near the
    start of the available series.
    """
    for j in range(n):
        if np.isfinite(out[j]):
            continue
        s = j - L
        if s >= 0:
            out[j] = out[s]
        else:
            src = k + s              # absolute interval position
            if src == decision_pos:
                if not np.isfinite(lead0_value):
                    return j
                out[j] = lead0_value
            elif 0 <= src < decision_pos and np.isfinite(actuals[src]):
                out[j] = actuals[src]
            else:
                return j             # history exhausted; window truncates here
        is_infill[j] = True
    return n
