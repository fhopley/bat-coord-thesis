"""Battery usable-capacity estimation.

Estimates a self-consistent
(capacity_kwh, charge_eff, discharge_eff) triple from a site's measured
AC-side battery charge/discharge trace, using plateau detection at the
real physical top/bottom of the free-running SOC trace and a
cumulative-charge-throughput correction to separate an efficiency-estimate
bias from genuine capacity drift.

USAGE
-----
Run as a module from the project root, same reasoning as em_ingest.py's
own usage note (avoids sys.path issues from running by file path):

    python3 -m energy_masters_pipeline.scripts.estimate_battery_capacity EM-427
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from energy_masters_pipeline.scripts.em_ingest import (
    DEFAULT_TELEMETRY_PATH, DEFAULT_SITE_SUMMARY_PATH,
    MAX_PLAUSIBLE_WH_PER_INTERVAL,
    _load_full_telemetry, _site_rows, battery_family_for, _family_sum,
    _reindex_interpolate, _to_datetime_aest, _full_grid,
    load_valid_periods_from_summary,
)

# Plateau-detection starting parameters.
# Not yet validated against a real trace -- treat as first-guess values to
# check against the EM-427 plot, not settled constants.
PLATEAU_BAND_FRACTION = 0.01       # 1% of current best capacity estimate
PLATEAU_MIN_DWELL_INTERVALS = 4    # 4 x 5-min = 20 minutes
MAX_CORRECTION_ITERATIONS = 12
RESIDUAL_SLOPE_CONVERGENCE_TOL = 1e-4  # kWh of plateau-level drift per kWh of cumulative throughput

INITIAL_NOMINAL_CAPACITY_KWH = 100.0
INITIAL_SOC_KWH = 50.0


@dataclass
class PlateauInstance:
    level_kwh: float
    timestamp: pd.Timestamp
    dwell_intervals: int
    cumulative_charge_kwh_at_midpoint: float


@dataclass
class BatteryCapacityEstimate:
    site_id: str
    capacity_kwh: float
    charge_eff: float
    discharge_eff: float
    soc_trace: pd.Series
    top_plateaus: list = field(default_factory=list)
    bottom_plateaus: list = field(default_factory=list)
    paired_amplitudes: list = field(default_factory=list)
    initial_rte_estimate: float = float('nan')
    n_correction_iterations: int = 0
    residual_throughput_slope: float = float('nan')
    residual_time_slope: float = float('nan')
    equivalent_full_cycles: float = float('nan')
    reliable: bool = False
    one_sided: bool = False
    real_soc_trace: pd.Series = None
    """SOC rebased onto the real [0, capacity_kwh] scale, for feeding a
    real-data SOC/flow reconstruction. See _rebase_to_real_soc for the
    method: the free-running trace's own scale is arbitrary (it starts at
    an offset of 50 kWh purely to avoid clipping during estimation), so it
    is rebased against the tracked upper envelope at each point in time
    (envelope(t) -> capacity_kwh, i.e. 'full', at each t) rather than a
    single fixed offset -- this follows a genuine step change in capacity
    (e.g. a battery added mid-window) instead of assuming one constant
    offset for the whole valid period. Clipped to [0, capacity_kwh].
    """
    initial_soc_kwh: float = float('nan')
    """real_soc_trace's value at the first timestamp of the valid period.
    The starting point to hand to a real-data SOC/flow reconstruction."""


# ---------------------------------------------------------------------------
# Loading the battery trace
# ---------------------------------------------------------------------------

def _load_battery_trace_kwh(
    site_id: str,
    valid_periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    telemetry_path: Path = DEFAULT_TELEMETRY_PATH,
    raw_df: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Battery kwh per 5-min interval, AEST-indexed, over the site's valid
    period(s). Reuses em_ingest.py's own channel-selection, family-sum,
    masking, and interpolation logic rather than duplicating it, so this
    stays consistent with load_em_site_trace's battery handling by
    construction.
    """
    df = raw_df if raw_df is not None else _load_full_telemetry(telemetry_path)
    sub = _site_rows(df, site_id)

    battery_family = battery_family_for(sub)
    if battery_family is None:
        raise ValueError(f"{site_id}: no battery channel present, cannot estimate capacity")

    battery_wh = _family_sum(sub, 'energy', battery_family)

    frames = []
    for start, end in valid_periods:
        full_index = _full_grid(start, end)
        bat = _reindex_interpolate(battery_wh, full_index, MAX_PLAUSIBLE_WH_PER_INTERVAL)
        frames.append(bat)

    out = pd.concat(frames) / 1000.0
    out.index = _to_datetime_aest(out.index)
    out.index.name = 'datetime_aest'
    return out


# ---------------------------------------------------------------------------
# SOC integration
# ---------------------------------------------------------------------------

def _integrate_soc(
    battery_kwh: pd.Series, charge_eff: float, discharge_eff: float,
) -> tuple[pd.Series, pd.Series]:
    """Free-running DC-side SOC trace on an oversized nominal battery, plus
    the cumulative AC-side charge throughput series (needed later for the
    throughput-based efficiency correction).

    soc = INITIAL_SOC_KWH + cumsum(dsoc): each interval's flow is applied
    on top of the running total, starting from INITIAL_SOC_KWH before the
    first interval.

    Returns (soc_trace, cumulative_charge_kwh), both indexed like
    battery_kwh.
    """
    charge = battery_kwh.clip(lower=0.0)
    discharge = battery_kwh.clip(upper=0.0)  # already <= 0
    dsoc = charge * charge_eff + discharge / discharge_eff
    soc = INITIAL_SOC_KWH + dsoc.cumsum()
    cumulative_charge = charge.cumsum()
    return soc, cumulative_charge


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------

def _detect_plateaus(
    soc: pd.Series,
    cumulative_charge: pd.Series,
    band_fraction: float,
    current_capacity_estimate: float,
    min_dwell: int,
    boundary: str,
) -> list[PlateauInstance]:
    """Detect local flat dwells and classify each as 'top' or 'bottom'
    type by the SOC trend immediately before it.

    A naive version of this compared the whole trace to a single GLOBAL
    max/min band, which a synthetic test caught as broken: with an
    imperfect efficiency estimate, repeated daily full/empty dwells DRIFT
    across the window, so only the single highest (or lowest) day's dwell
    ever falls within a band of the global extreme -- every other day's
    otherwise-genuine plateau is silently missed. That left only one
    instance per boundary, which degenerates into a two-point line that
    exactly fits any two points, giving no real evidence about whether a
    trend is genuine bias or noise.

    Fixed here by finding LOCAL flatness first (a run where the rolling
    range of soc stays within the band, regardless of its absolute
    level), then classifying each flat run as a top-type or bottom-type
    plateau using the sign of the SOC change immediately before the run
    started (rising into a flat run means a top/full dwell; falling into
    one means a bottom/empty dwell). A flat run with no discernible
    trend before it, e.g. right at the start of the series, is skipped
    rather than guessed at.
    """
    if boundary not in ('top', 'bottom'):
        raise ValueError(f"boundary must be 'top' or 'bottom', got {boundary!r}")

    band = band_fraction * current_capacity_estimate
    values = soc.to_numpy()
    idx = soc.index
    n = len(values)

    # Rolling range over a min_dwell window: True where the window is
    # flat enough to be a candidate plateau tick.
    s = pd.Series(values)
    rolling_range = s.rolling(min_dwell).max() - s.rolling(min_dwell).min()
    flat_tick = (rolling_range <= band).to_numpy()
    # rolling(...).max()/.min() label each window by its LAST index, so
    # flat_tick[i] means "the min_dwell-length window ending at i is flat".

    instances = []
    run_start = None
    for i in range(n):
        if flat_tick[i] and run_start is None:
            # window ending at i is flat -> the run actually starts
            # min_dwell-1 back from here.
            run_start = max(0, i - (min_dwell - 1))
        elif not flat_tick[i] and run_start is not None:
            _maybe_add_classified_instance(
                instances, values, idx, cumulative_charge, run_start, i - 1, min_dwell, boundary,
            )
            run_start = None
    if run_start is not None:
        _maybe_add_classified_instance(
            instances, values, idx, cumulative_charge, run_start, n - 1, min_dwell, boundary,
        )
    return instances


def _maybe_add_classified_instance(instances, values, idx, cumulative_charge, start, end, min_dwell, boundary):
    dwell = end - start + 1
    if dwell < min_dwell:
        return
    if start == 0:
        return  # no trend available before the run to classify it by
    trend = values[start] - values[start - 1]
    is_top = trend > 0
    if boundary == 'top' and not is_top:
        return
    if boundary == 'bottom' and is_top:
        return
    level = float(np.median(values[start:end + 1]))
    mid = start + (end - start) // 2
    ts = idx[mid]
    cum_charge_at_mid = float(cumulative_charge.iloc[mid])
    instances.append(PlateauInstance(
        level_kwh=level, timestamp=ts, dwell_intervals=dwell,
        cumulative_charge_kwh_at_midpoint=cum_charge_at_mid,
    ))


# ---------------------------------------------------------------------------
# Nearest-in-time pairing and amplitude / outlier logic
# ---------------------------------------------------------------------------

@dataclass
class PairedAmplitude:
    amplitude_kwh: float
    timestamp: pd.Timestamp  # midpoint of the paired top/bottom timestamps
    cumulative_charge_kwh_at_midpoint: float
    top_timestamp: pd.Timestamp
    bottom_timestamp: pd.Timestamp
    is_outlier: bool = False


def _pair_amplitudes(
    top_plateaus: list[PlateauInstance], bottom_plateaus: list[PlateauInstance],
) -> list[PairedAmplitude]:
    """Pair each top plateau instance with its nearest-in-time bottom
    instance (and vice versa), giving amplitude_i = |top - bottom| for
    each pair.

    A single amplitude reading is only a LOWER bound on true capacity --
    a top and bottom that don't belong to the same full swing (e.g. a
    partial overnight discharge that doesn't reach empty before the next
    day's charge) will read smaller than the real capacity. True full
    swings should cluster near the top of the amplitude distribution;
    the maximum of the clean set (see _flag_outliers) is therefore the
    estimator, not the mean or median.

    Deliberately not a strict bijection -- some bottoms may be nearest to
    more than one top and vice versa. That's fine here: the goal is a
    representative sample of local swings, not a unique cycle count.
    """
    if not top_plateaus or not bottom_plateaus:
        return []

    bottom_sorted = sorted(bottom_plateaus, key=lambda p: p.timestamp)
    bottom_ts = np.array([b.timestamp.value for b in bottom_sorted])
    top_sorted = sorted(top_plateaus, key=lambda p: p.timestamp)
    top_ts = np.array([t.timestamp.value for t in top_sorted])

    pairs = []
    for t in top_sorted:
        j = int(np.searchsorted(bottom_ts, t.timestamp.value))
        candidates = [j - 1, j]
        candidates = [c for c in candidates if 0 <= c < len(bottom_sorted)]
        b = min(candidates, key=lambda c: abs(bottom_sorted[c].timestamp.value - t.timestamp.value))
        b = bottom_sorted[b]
        pairs.append(_make_pair(t, b))
    for b in bottom_sorted:
        i = int(np.searchsorted(top_ts, b.timestamp.value))
        candidates = [i - 1, i]
        candidates = [c for c in candidates if 0 <= c < len(top_sorted)]
        t = min(candidates, key=lambda c: abs(top_sorted[c].timestamp.value - b.timestamp.value))
        t = top_sorted[t]
        pairs.append(_make_pair(t, b))

    # De-duplicate identical (top_timestamp, bottom_timestamp) pairs that
    # arise from being found from both directions.
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = (p.top_timestamp, p.bottom_timestamp)
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)
    return sorted(unique_pairs, key=lambda p: p.timestamp)


def _make_pair(top: PlateauInstance, bottom: PlateauInstance) -> PairedAmplitude:
    mid_value = top.timestamp.value + (bottom.timestamp.value - top.timestamp.value) // 2
    mid_ts = pd.Timestamp(mid_value, tz=top.timestamp.tz)
    cum_mid = (top.cumulative_charge_kwh_at_midpoint + bottom.cumulative_charge_kwh_at_midpoint) / 2
    return PairedAmplitude(
        amplitude_kwh=abs(top.level_kwh - bottom.level_kwh),
        timestamp=mid_ts,
        cumulative_charge_kwh_at_midpoint=cum_mid,
        top_timestamp=top.timestamp,
        bottom_timestamp=bottom.timestamp,
    )


def _flag_upper_outliers(
    pairs: list[PairedAmplitude],
    current_capacity_estimate: float,
    window_frac: float = 0.15,
    min_window: int = 15,
    gap_multiplier: float = 4.0,
    min_absolute_gap_fraction: float = 0.02,
) -> None:
    """Iterative local-gap outlier check, applied in place.

    Superseded the earlier global Tukey-IQR fence: a single multiplier
    (whether 1.5x or 3x) can't work well across sites, because it
    measures a candidate's distance from the BULK of the (mostly
    partial-cycle) distribution, and how far that bulk sits from the
    near-capacity cluster varies site to site with the distribution's
    skew. Confirmed on two real sites: EM-427 needed 3x (1.5x clipped
    genuine near-capacity swings), but EM-414 showed the opposite problem
    at 3x -- a single isolated 17.5 kWh point, sitting in a large empty
    gap above a tight ~12.5-13.5 kWh cluster of genuine readings, passed
    the 3x fence and inflated the capacity estimate.

    What actually distinguishes a genuine near-capacity swing from a data
    artifact, by eye, is not distance from the bulk -- it's ISOLATION:
    a real outlier sits in an unusually large gap above its neighbours,
    while genuine near-capacity swings sit close together in a cluster.
    This checks that directly and locally (only within the top of the
    distribution, not the whole skewed population), so it doesn't need a
    per-site multiplier:

    1. Sort descending, take the top window_frac (or min_window, if
       larger) of points -- restricted to the near-capacity region, not
       the long partial-cycle tail.
    2. Compare the gap between the current top point and its neighbour to
       the median gap among the next several points in that window (a
       local, robust "typical spacing" measure, itself insensitive to one
       outlier gap).
    3. If that top gap exceeds gap_multiplier times the typical local
       spacing, AND exceeds a minimum absolute floor (min_absolute_gap_fraction
       of the current capacity estimate, so near-identical points don't
       get flagged just because their tiny spacing is numerically small),
       the top point is an isolated outlier. Drop it and repeat.
    4. Stop as soon as a gap is NOT unusually large -- that's where the
       real clustered top of the distribution begins.
    """
    if len(pairs) < 8:
        return
    sorted_pairs = sorted(pairs, key=lambda p: -p.amplitude_kwh)
    amps = [p.amplitude_kwh for p in sorted_pairs]
    n = len(amps)
    window = min(n, max(min_window, int(window_frac * n)))
    min_absolute_gap = min_absolute_gap_fraction * current_capacity_estimate

    idx = 0
    while idx < window - 3 and idx < n - 3:
        local = amps[idx:idx + window]
        gaps = [local[i] - local[i + 1] for i in range(len(local) - 1)]
        if len(gaps) < 3:
            break
        current_gap = gaps[0]
        baseline_gaps = gaps[1:]
        median_gap = float(np.median(baseline_gaps))
        threshold = max(gap_multiplier * median_gap, min_absolute_gap)
        if current_gap > threshold and current_gap > 0:
            sorted_pairs[idx].is_outlier = True
            idx += 1
        else:
            break


def _fit_slope(xs: list[float], ys: list[float]) -> float:
    """Plain least-squares slope. Returns 0.0 if fewer than 2 points (no
    basis for a slope, not an error)."""
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def estimate_battery_capacity(
    site_id: str,
    valid_periods: Optional[list[tuple[pd.Timestamp, pd.Timestamp]]] = None,
    summary_csv_path: Path = DEFAULT_SITE_SUMMARY_PATH,
    telemetry_path: Path = DEFAULT_TELEMETRY_PATH,
    raw_df: Optional[pd.DataFrame] = None,
) -> BatteryCapacityEstimate:
    if valid_periods is None:
        valid_periods = load_valid_periods_from_summary(site_id, summary_csv_path)
    if len(valid_periods) != 1:
        raise ValueError(
            f"{site_id}: expected exactly one valid segment, found "
            f"{len(valid_periods)}. This design does not yet handle "
            "multi-segment sites."
        )

    battery_kwh = _load_battery_trace_kwh(site_id, valid_periods, telemetry_path, raw_df)

    total_charge = battery_kwh.clip(lower=0.0).sum()
    total_discharge = -battery_kwh.clip(upper=0.0).sum()
    if total_charge <= 0:
        raise ValueError(f"{site_id}: no charging intervals found, cannot estimate RTE")
    initial_rte = total_discharge / total_charge
    initial_rte = min(initial_rte, 0.999)  # RTE >= 1 is unphysical for a real battery
    rte = initial_rte

    # First-guess capacity from the raw free-running extremes, used only
    # to set the initial plateau band width.
    charge_eff = discharge_eff = float(np.sqrt(rte))
    soc, cumulative_charge = _integrate_soc(battery_kwh, charge_eff, discharge_eff)
    capacity_guess = float(soc.max() - soc.min())

    n_iterations = 0
    residual_slope = float('nan')
    top_plateaus: list[PlateauInstance] = []
    bottom_plateaus: list[PlateauInstance] = []
    pairs: list[PairedAmplitude] = []

    for n_iterations in range(1, MAX_CORRECTION_ITERATIONS + 1):
        top_plateaus = _detect_plateaus(
            soc, cumulative_charge, PLATEAU_BAND_FRACTION, capacity_guess,
            PLATEAU_MIN_DWELL_INTERVALS, 'top',
        )
        bottom_plateaus = _detect_plateaus(
            soc, cumulative_charge, PLATEAU_BAND_FRACTION, capacity_guess,
            PLATEAU_MIN_DWELL_INTERVALS, 'bottom',
        )

        pairs = _pair_amplitudes(top_plateaus, bottom_plateaus)
        _flag_upper_outliers(pairs, capacity_guess)
        clean_pairs = [p for p in pairs if not p.is_outlier]

        if len(clean_pairs) < 2:
            # Not enough clean anchors to fit a throughput correction --
            # stop iterating, report what we have, let 'reliable'
            # downstream communicate the weak evidence.
            residual_slope = float('nan')
            break

        capacity_guess = max(p.amplitude_kwh for p in clean_pairs)

        xs = [p.cumulative_charge_kwh_at_midpoint for p in clean_pairs]
        ys = [p.amplitude_kwh for p in clean_pairs]
        residual_slope = _fit_slope(xs, ys)

        if abs(residual_slope) < RESIDUAL_SLOPE_CONVERGENCE_TOL:
            break

        # A positive slope means achieved swing amplitude grows with more
        # cumulative charge throughput -- consistent with charge_eff being
        # set too high (each kWh charged is over-credited, so a full
        # charge/discharge swing measures as bigger than it really is, more
        # so the more throughput has passed). Nudge RTE down a bit and
        # re-integrate. This regresses AMPLITUDE against throughput, not
        # absolute plateau level -- absolute level is confounded by
        # ordinary envelope drift (varying solar/load week to week), which
        # doesn't change the achieved swing size the way an efficiency
        # bias does. Step size is a plain damped update, not a real
        # solver -- fine for a handful of iterations against a slope that
        # should be small to start with.
        rte = float(np.clip(rte * (1 - 0.5 * np.sign(residual_slope) * min(abs(residual_slope), 0.2)), 0.05, 0.999))
        charge_eff = discharge_eff = float(np.sqrt(rte))
        soc, cumulative_charge = _integrate_soc(battery_kwh, charge_eff, discharge_eff)
        capacity_guess = float(soc.max() - soc.min())

    reliable = len(top_plateaus) > 0 and len(bottom_plateaus) > 0
    one_sided = (len(top_plateaus) > 0) != (len(bottom_plateaus) > 0)

    clean_pairs = [p for p in pairs if not p.is_outlier]
    if clean_pairs:
        capacity_kwh = max(p.amplitude_kwh for p in clean_pairs)
    elif pairs:
        # Only outliers were found (extremely unusual) -- fall back to
        # the outlier set rather than reporting nothing, but this should
        # essentially never happen in practice and is worth a hard look
        # at the data if it does.
        capacity_kwh = max(p.amplitude_kwh for p in pairs)
    else:
        # Fallback per the "no plateaus" edge case: raw extremes, clearly
        # unsupported by a physical dwell.
        capacity_kwh = float(soc.max() - soc.min())

    # Residual trend against elapsed calendar time, using the clean paired
    # amplitudes (not absolute plateau levels, for the same envelope-drift
    # reason as above), after the throughput correction.
    if len(clean_pairs) >= 2:
        t0 = clean_pairs[0].timestamp
        time_xs = [(p.timestamp - t0).total_seconds() / 86400 for p in clean_pairs]
        time_ys = [p.amplitude_kwh for p in clean_pairs]
        residual_time_slope = _fit_slope(time_xs, time_ys)
    else:
        residual_time_slope = float('nan')

    equivalent_full_cycles = total_discharge / capacity_kwh if capacity_kwh > 0 else float('nan')

    envelope = _upper_envelope(soc)
    real_soc_trace = _rebase_to_real_soc(soc, envelope, capacity_kwh)
    initial_soc_kwh = float(real_soc_trace.iloc[0])

    return BatteryCapacityEstimate(
        site_id=site_id,
        capacity_kwh=capacity_kwh,
        charge_eff=charge_eff,
        discharge_eff=discharge_eff,
        soc_trace=soc,
        top_plateaus=top_plateaus,
        bottom_plateaus=bottom_plateaus,
        paired_amplitudes=pairs,
        initial_rte_estimate=initial_rte,
        n_correction_iterations=n_iterations,
        residual_throughput_slope=residual_slope,
        residual_time_slope=residual_time_slope,
        equivalent_full_cycles=equivalent_full_cycles,
        reliable=reliable,
        one_sided=one_sided,
        real_soc_trace=real_soc_trace,
        initial_soc_kwh=initial_soc_kwh,
    )


# ---------------------------------------------------------------------------
# Upper envelope, for the sense-check plot
# ---------------------------------------------------------------------------

def _upper_envelope(
    soc: pd.Series, short_window_days: float = 3.0, smooth_window_days: float = 4.0,
) -> pd.Series:
    """A slowly-varying upper envelope of the free-running SOC trace, for
    overlaying a capacity-height shaded band on the plot.

    Morphological opening: rolling max over a short window, picks out the
    peak within that window regardless of exact timing, then rolling min
    of THAT sequence over a longer window (a few days), which smooths
    away day-to-day noise in the daily peaks while still tracking genuine
    slow drift. Unlike a plain long-window rolling max (which would just
    report the single highest point seen so far, never coming back down)
    this can and does come back down when the real peaks genuinely drop,
    and unlike a simple decay-based envelope follower it doesn't need a
    hand-tuned decay rate -- it responds to a sustained level shift (like
    the EM-427 late-Oct gap) about as fast as the smoothing window
    allows, rather than creeping down slowly.

    short_window_days was initially 1.0 (a single day), which tracked
    individual daily peaks too tightly -- confirmed against the real
    EM-427 plot as visibly too sensitive to single-day variation rather
    than reading as a smooth upper bound by eye. Widened to 3.0 days so
    the top of the band reflects the peak over a few days at a time, not
    any one day's specific high.

    This is a visual aid only, not part of the capacity estimate itself.
    """
    if len(soc) < 3:
        return soc.copy()
    interval_minutes = 5
    short_n = max(3, int(short_window_days * 24 * 60 / interval_minutes))
    smooth_n = max(3, int(smooth_window_days * 24 * 60 / interval_minutes))
    daily_peak = soc.rolling(short_n, center=True, min_periods=1).max()
    envelope = daily_peak.rolling(smooth_n, center=True, min_periods=1).min()
    return envelope


def _rebase_to_real_soc(
    soc: pd.Series, envelope: pd.Series, capacity_kwh: float,
) -> pd.Series:
    """Rebase the free-running SOC trace onto the real [0, capacity_kwh]
    scale, for feeding a real-data SOC/flow reconstruction.

    The free-running trace's own scale is arbitrary (an offset of 50 kWh
    on a 100 kWh nominal container, chosen purely to avoid clipping during
    estimation), so a real SOC needs a real reference point. Using the
    tracked upper envelope as that reference, per timestamp, rather than a
    single fixed offset for the whole valid period, means a genuine step
    change in capacity (e.g. a battery added mid-window, which shows up as
    a step in the envelope) gets carried through correctly instead of
    silently assuming one constant offset throughout.

    real_soc(t) = soc(t) - envelope(t) + capacity_kwh
        so that envelope(t) always maps to capacity_kwh ("full") at t.

    Clipped to [0, capacity_kwh] since plateau-detection imprecision or
    noise can push a rebased value just outside the physical range.
    """
    real_soc = soc - envelope + capacity_kwh
    return real_soc.clip(lower=0.0, upper=capacity_kwh)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_capacity_estimate(estimate: BatteryCapacityEstimate, out_path: Optional[Path] = None):
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1], sharex=False)
    ax.plot(estimate.soc_trace.index, estimate.soc_trace.to_numpy(),
            color='0.4', linewidth=0.8, label='free-running SOC')

    envelope = _upper_envelope(estimate.soc_trace)
    ax.fill_between(
        envelope.index, envelope.to_numpy() - estimate.capacity_kwh, envelope.to_numpy(),
        color='tab:green', alpha=0.15, zorder=1,
        label=f'capacity band ({estimate.capacity_kwh:.2f} kWh)',
    )
    ax.plot(envelope.index, envelope.to_numpy(), color='tab:green', linewidth=1.0,
            alpha=0.6, zorder=2, label='tracked upper envelope')

    if estimate.top_plateaus:
        ax.scatter([p.timestamp for p in estimate.top_plateaus],
                   [p.level_kwh for p in estimate.top_plateaus],
                   color='tab:red', s=15, zorder=5, label='top plateau instances')
    if estimate.bottom_plateaus:
        ax.scatter([p.timestamp for p in estimate.bottom_plateaus],
                   [p.level_kwh for p in estimate.bottom_plateaus],
                   color='tab:blue', s=15, zorder=5, label='bottom plateau instances')

    ax.set_ylabel('free-running SOC (kWh)')
    ax.legend(loc='lower right')

    # Second panel: amplitude of each paired top/bottom swing over time,
    # the actual quantity the capacity estimate is drawn from, with the
    # IQR-flagged outliers (data problems, like a missed charge event)
    # marked distinctly from the clean set they're excluded from.
    clean = [p for p in estimate.paired_amplitudes if not p.is_outlier]
    outliers = [p for p in estimate.paired_amplitudes if p.is_outlier]
    if clean:
        ax2.scatter([p.timestamp for p in clean], [p.amplitude_kwh for p in clean],
                    color='0.2', s=10, label='paired amplitude')
    if outliers:
        ax2.scatter([p.timestamp for p in outliers], [p.amplitude_kwh for p in outliers],
                    color='tab:orange', s=30, marker='x', label='flagged outliers')
    ax2.axhline(estimate.capacity_kwh, color='tab:green', linestyle='--', linewidth=1.0,
                label=f'capacity estimate ({estimate.capacity_kwh:.2f} kWh)')
    ax2.set_ylabel('paired swing amplitude (kWh)')
    ax2.legend(loc='upper right')

    ax.set_title(
        f"{estimate.site_id}: capacity={estimate.capacity_kwh:.2f} kWh  "
        f"charge_eff=discharge_eff={estimate.charge_eff:.3f}  "
        f"reliable={estimate.reliable}  cycles={estimate.equivalent_full_cycles:.1f}  "
        f"({len(outliers)} outlier pair(s) excluded)"
    )
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Verification / CLI entry point
# ---------------------------------------------------------------------------

def run_checks(
    site_id: str = 'EM-427',
    summary_csv_path: Path = DEFAULT_SITE_SUMMARY_PATH,
    telemetry_path: Path = DEFAULT_TELEMETRY_PATH,
) -> None:
    print(f"=== {site_id} battery capacity estimation check ===")
    estimate = estimate_battery_capacity(site_id, summary_csv_path=summary_csv_path,
                                          telemetry_path=telemetry_path)

    print(f"\ncapacity_kwh: {estimate.capacity_kwh:.3f}  "
          "(max of the IQR-clean paired top/bottom swing amplitudes)")
    print(f"charge_eff = discharge_eff: {estimate.charge_eff:.4f} "
          f"(initial RTE guess: {estimate.initial_rte_estimate:.4f})")
    print(f"correction iterations: {estimate.n_correction_iterations}")
    print(f"residual slope vs cumulative charge throughput: "
          f"{estimate.residual_throughput_slope:.6f} kWh/kWh "
          f"({'FAIL -- did not converge, treat capacity as approximate' if abs(estimate.residual_throughput_slope) >= RESIDUAL_SLOPE_CONVERGENCE_TOL else 'OK'})")
    print(f"residual slope vs elapsed time: {estimate.residual_time_slope:.6f} kWh/day "
          "(non-negligible => possible genuine drift, inspect the plot)")
    print(f"equivalent full cycles: {estimate.equivalent_full_cycles:.2f}")
    print(f"top plateau instances: {len(estimate.top_plateaus)}  "
          f"bottom plateau instances: {len(estimate.bottom_plateaus)}")

    outliers = [p for p in estimate.paired_amplitudes if p.is_outlier]
    clean = [p for p in estimate.paired_amplitudes if not p.is_outlier]
    print(f"paired amplitudes: {len(estimate.paired_amplitudes)} total, "
          f"{len(clean)} clean, {len(outliers)} flagged as upper-IQR outliers")
    if outliers:
        print("Flagged outlier pairs (likely data issues, e.g. a missed charge/discharge "
              "event -- worth inspecting each of these against the raw telemetry):")
        for p in sorted(outliers, key=lambda p: -p.amplitude_kwh):
            print(f"  amplitude={p.amplitude_kwh:.2f} kWh  top@{p.top_timestamp}  "
                  f"bottom@{p.bottom_timestamp}")

    print(f"reliable: {estimate.reliable}  one_sided: {estimate.one_sided}")
    print(f"\ninitial_soc_kwh (for modelling, start of valid period): "
          f"{estimate.initial_soc_kwh:.3f} kWh  "
          f"({100*estimate.initial_soc_kwh/estimate.capacity_kwh:.1f}% of capacity)")
    if not estimate.reliable:
        print("FAIL (soft) -- no plateau at one or both boundaries; "
              "capacity_kwh above is the raw free-running max-min fallback, "
              "not a plateau-anchored estimate.")

    out_dir = Path('energy_masters_pipeline/charts/diagnostics')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{site_id}_battery_capacity.png'
    plot_capacity_estimate(estimate, out_path)
    print("\n=== Done ===")


if __name__ == '__main__':
    import sys
    site = sys.argv[1] if len(sys.argv) > 1 else 'EM-427'
    run_checks(site)