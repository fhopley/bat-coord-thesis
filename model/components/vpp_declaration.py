"""Causal event declaration for the rolling VPP coordination types.

`classify_vpp_events` decides the whole year at once by scanning backwards over
settled prices. A rolling operator cannot do that: at interval `k` it holds a
forecast and must decide, now, whether it is taking control. This module is
that decision.

The rule
--------
An event is active at `k` when a forecast price above the threshold lies within
one event window length ahead:

    lookahead(k) = [k, min(k + W - 1, k + n - 1)]
    event(k)     = any(rrp_forecast[t] > threshold for t in lookahead(k))

The operator takes control for up to `event_window_hours` before a spike in
order to position the battery, so the question at any interval is whether a
spike is within that positioning window. The horizon is normally far longer
than the window -- 18 hours against 10, or against 2 -- so the operator
routinely sees spikes it cannot yet act on. `W` is the reach of the
declaration; `n` only bounds it near the end of the period.

Relationship to classify_vpp_events
-----------------------------------
This produces the SKIP-FREE union of the backwards windows. `classify_vpp_events`
skips a spike already covered by a window opened by a later spike, which depends
on which spikes have already been processed -- knowledge of the future that a
causal rule cannot have. The forward rule is therefore strictly more permissive
wherever spikes cluster closer than `W`, and a rolling run compared against
`vpp_sc_event` differs for two reasons at once: forecast error, and a mask that
starts events earlier around clusters. The two are not separable in a VCR.

Un-declaration
--------------
When the rule stops holding the event ends immediately. Committing for a fixed
duration instead would manufacture phantom events: one interval of forecast
spike that evaporates on the next run would hand the operator a full
`event_window_hours` of control with nothing to dispatch into. The cost is that
a forecast hovering near the threshold inside the lookahead can toggle control
every five minutes, which is measured rather than assumed -- see the block
length counters on ShadowTrajectory.

See rolling_vpp_event_spec.md, "Event declaration".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def window_intervals(event_window_hours: float, interval_hours: float) -> int:
    """W: intervals in one event window, rounded up.

    Rounded up so a fractional window always covers at least the requested
    duration, matching classify_vpp_events.
    """
    if event_window_hours <= 0:
        raise ValueError(
            f"event_window_hours must be > 0, got {event_window_hours}")
    return math.ceil(event_window_hours / interval_hours)


def declare(
    rrp_forecast: np.ndarray,
    n: int,
    w: int,
    threshold_dollars_per_mwh: float,
) -> tuple[bool, int]:
    """Decide whether an event is active at the window's first interval.

    Args:
        rrp_forecast: Forecast wholesale price over the window, $/MWh,
            position 0 being the interval about to be committed. May be
            LONGER than n when the loop shrank the window after the price
            provider was queried; only the first n positions are read.
            Reading the whole slice would let declaration see prices for
            intervals outside the window.
        n: Window length in intervals.
        w: Event window length in intervals, from window_intervals().
        threshold_dollars_per_mwh: VppConfig.vpp_threshold_dollars_per_mwh.

    Returns:
        (active, j_star) where j_star is the window-relative position of the
        LAST forecast spike inside the lookahead, or -1 when inactive. The
        operator LP runs [0, j_star]: beyond the last spike it has nothing to
        dispatch into, and whether the event continues past it is the next
        interval's question, decided against the next forecast.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if w < 1:
        raise ValueError(f"w must be >= 1, got {w}")

    reach = min(w, n)
    prices = np.asarray(rrp_forecast[:reach], dtype=float)
    if prices.size < reach:
        raise ValueError(
            f"rrp_forecast covers {prices.size} intervals but the lookahead "
            f"needs {reach} (n={n}, w={w})"
        )
    if not np.isfinite(prices).all():
        n_bad = int((~np.isfinite(prices)).sum())
        raise ValueError(
            f"rrp_forecast contains {n_bad} non-finite value(s) inside the "
            f"lookahead; a provider must return a fully populated window"
        )

    spikes = np.flatnonzero(prices > threshold_dollars_per_mwh)
    if spikes.size == 0:
        return False, -1
    return True, int(spikes[-1])


def declared_mask(
    prices: np.ndarray,
    w: int,
    threshold_dollars_per_mwh: float,
) -> np.ndarray:
    """The mask the forward rule produces over a whole price series.

    Not used by the rolling loop, which calls declare() once per interval
    against a forecast. This applies the same rule to a settled series, so the
    declaration rule can be compared against classify_vpp_events directly and
    the mask difference measured on its own rather than being left inside a
    VCR alongside forecast error.
    """
    p = np.asarray(prices, dtype=float)
    T = p.size
    out = np.zeros(T, dtype=bool)
    for k in range(T):
        n = T - k
        active, _ = declare(p[k:], n, w, threshold_dollars_per_mwh)
        out[k] = active
    return out


@dataclass
class CapState:
    """Per-site annual VPP energy cap. Exact, and decremented only on commit.

    The cap is per site even though the minimum export price is fleet-wide:
    it is a contract term on that site's battery. What counts toward it varies
    by plan and is read from VppConfig.limit_applies_to -- 'export_only'
    counts forced discharge alone, 'import_and_export' counts forced charge
    and forced discharge together. Both are in use across real offerings.

    Once exhausted the site stops participating and the household strategy
    runs to the end of the period. The state is terminal: `remaining` only
    shrinks, so no re-entry is possible and no reset applies.
    """
    limit_kwh: float
    limit_applies_to: str
    remaining_kwh: float = None          # type: ignore[assignment]
    exhausted_at: int | None = None
    used_kwh: float = 0.0

    def __post_init__(self):
        if self.limit_applies_to not in ('export_only', 'import_and_export'):
            raise ValueError(
                f"limit_applies_to must be 'export_only' or "
                f"'import_and_export', got {self.limit_applies_to!r}"
            )
        if self.limit_kwh < 0:
            raise ValueError(f"limit_kwh must be >= 0, got {self.limit_kwh}")
        if self.remaining_kwh is None:
            self.remaining_kwh = float(self.limit_kwh)

    # Solver tolerance, not physics: the remaining cap is the difference of
    # accumulated LP outputs, so a cap that is exactly spent can land either
    # side of zero. It was observed at +8.3e-17 kWh after a solver change,
    # which left the site declaring events forever on a cap it had used up.
    # 1e-9 kWh is a microjoule; nothing physical lives below it.
    EXHAUSTION_TOL_KWH = 1e-9

    @property
    def exhausted(self) -> bool:
        return self.remaining_kwh <= self.EXHAUSTION_TOL_KWH

    def charge_to_cap(self, f_imp: float, f_exp: float) -> float:
        """What the given forced flows consume, per limit_applies_to."""
        if self.limit_applies_to == 'export_only':
            return float(f_exp)
        return float(f_imp) + float(f_exp)

    def commit(self, k: int, f_imp: float, f_exp: float) -> None:
        """Decrement by a COMMITTED interval's forced flows.

        Only committed flows count. Planned forced flows in the discarded tail
        of a window must never reach here: a cap decremented from the plan
        would exhaust roughly one horizon's worth of intervals too fast, and
        the error would look like a plausible under-dispatch rather than a
        bug.
        """
        used = self.charge_to_cap(f_imp, f_exp)
        if used < 0:
            raise ValueError(
                f"forced flows must be non-negative, got "
                f"f_imp={f_imp}, f_exp={f_exp}"
            )
        self.used_kwh += used
        self.remaining_kwh -= used
        if self.exhausted and self.exhausted_at is None:
            self.exhausted_at = k

    def apply_settlement_delta(self, k: int, d_f_imp: float) -> None:
        """Adjust the cap when retrospective settlement rewrites forced import.

        commit() decrements per interval from the split the LP produced. When
        a block closes, settlement recomputes that split against the event's
        true span and can move volume between n_imp and f_imp. On an
        'import_and_export' plan the cap counts the very quantity that moves,
        so without this the constraint is enforced against one number while
        VppTraces, billing and the reward report another -- and the cap can be
        exceeded while still reporting headroom.

        The cap CAN be exceeded retrospectively as a result. That is honest
        rather than a defect: the operator dispatched against a projected
        budget and settlement revised it after the fact. It is counted, and
        exhausted_at records the block-closing interval that tipped it.

        No-op under 'export_only', where settlement cannot change what the cap
        counts.
        """
        if self.limit_applies_to == 'export_only':
            return
        self.used_kwh += float(d_f_imp)
        self.remaining_kwh -= float(d_f_imp)
        if self.exhausted and self.exhausted_at is None:
            self.exhausted_at = k

    def plan_diagnostics(self) -> dict:
        """Snapshot. Does not mutate."""
        return {
            'cap_limit_kwh': self.limit_kwh,
            'cap_used_kwh': self.used_kwh,
            'cap_remaining_kwh': max(0.0, self.remaining_kwh),
            'cap_exhausted_at': self.exhausted_at,
        }
