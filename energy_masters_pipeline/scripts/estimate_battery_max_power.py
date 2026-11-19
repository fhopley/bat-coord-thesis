"""Max AC battery power estimation.

Implements Part A from the flow-reconstruction design discussion: one
symmetric max AC power value (assumed the same for charge and discharge),
estimated from the measured battery flow trace.

Unlike capacity, this doesn't need SOC integration. A genuine power-limit
hit shows up directly as the measured flow itself repeatedly clipping at
the same ceiling -- the BMS/inverter simply won't move more than its rated
power, whenever the household tries to. So the method is: take |battery_kwh|
across all intervals (charge and discharge combined, since one symmetric
value is wanted), and apply the same iterative local-gap outlier check
built for estimate_battery_capacity.py's capacity estimate -- sort
descending, check whether the top value is isolated from the next cluster
by an unusually large local gap, strip it if so, repeat. The clean max is
the power estimate.

This duplicates a small (~30 line) piece of estimate_battery_capacity.py's
gap-detection logic rather than importing its private (underscore-prefixed)
function, to keep the two modules independent -- the algorithm is short
enough that the duplication is cheaper than coupling two dev scripts
through a private interface.

A synthetic self-test (see __main__) covers the isolated-outlier case: a
tight cluster near a true power ceiling with one isolated outlier
correctly collapses to the clean ceiling value.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MaxPowerEstimate:
    site_id: str
    max_power_kwh_per_interval: float
    max_power_kw: float
    n_intervals_total: int
    n_intervals_near_ceiling: int
    """Count of intervals within 5% of the estimated ceiling -- a
    reliability signal, same role as equivalent_full_cycles in the
    capacity estimate: more repeated hits at the ceiling means more
    confidence in the number."""
    n_outliers_excluded: int


def _iterative_gap_outlier_filter(
    values: list[float],
    scale_reference: float,
    window_frac: float = 0.15,
    min_window: int = 15,
    gap_multiplier: float = 4.0,
    min_absolute_gap_fraction: float = 0.02,
) -> tuple[list[float], int]:
    """Same method as estimate_battery_capacity.py's _flag_upper_outliers,
    applied here to raw |battery_kwh| magnitudes rather than paired SOC
    swing amplitudes. Returns (clean_values, n_excluded).
    """
    n = len(values)
    if n < 8:
        return list(values), 0
    order = sorted(range(n), key=lambda i: -values[i])
    sorted_vals = [values[i] for i in order]
    window = min(n, max(min_window, int(window_frac * n)))
    min_abs_gap = min_absolute_gap_fraction * scale_reference

    n_excluded = 0
    idx = 0
    while idx < window - 3 and idx < n - 3:
        local = sorted_vals[idx:idx + window]
        gaps = [local[i] - local[i + 1] for i in range(len(local) - 1)]
        if len(gaps) < 3:
            break
        current_gap = gaps[0]
        baseline_gaps = gaps[1:]
        median_gap = float(np.median(baseline_gaps))
        threshold = max(gap_multiplier * median_gap, min_abs_gap)
        if current_gap > threshold and current_gap > 0:
            n_excluded += 1
            idx += 1
        else:
            break

    clean_vals = sorted_vals[n_excluded:]
    return clean_vals, n_excluded


def estimate_max_power(
    site_id: str,
    battery_kwh: pd.Series,
    interval_hours: float = 5 / 60,
) -> MaxPowerEstimate:
    """Estimate one symmetric max AC power (kW) for charge and discharge.

    Args:
        battery_kwh: measured battery flow, +charge/-discharge, over the
            site's valid period(s).
        interval_hours: interval length, for converting kWh/interval to kW
            (matching BatterySpec.power_kw's definition in self_consumption.py).
    """
    magnitudes = battery_kwh.abs().to_numpy(dtype=float, copy=False)
    magnitudes = magnitudes[magnitudes > 0.0]  # zero-flow intervals carry no power-limit info
    if len(magnitudes) == 0:
        raise ValueError(f"{site_id}: no nonzero battery flow intervals, cannot estimate max power")

    # First-guess scale for the outlier filter's absolute floor, refined
    # by using the raw max initially (same bootstrapping pattern as
    # estimate_battery_capacity.py's capacity_guess).
    scale_guess = float(magnitudes.max())
    clean_vals, n_excluded = _iterative_gap_outlier_filter(list(magnitudes), scale_guess)

    max_power_kwh_per_interval = max(clean_vals) if clean_vals else scale_guess
    max_power_kw = max_power_kwh_per_interval / interval_hours

    near_ceiling_threshold = 0.95 * max_power_kwh_per_interval
    n_near_ceiling = int((magnitudes >= near_ceiling_threshold).sum())

    return MaxPowerEstimate(
        site_id=site_id,
        max_power_kwh_per_interval=max_power_kwh_per_interval,
        max_power_kw=max_power_kw,
        n_intervals_total=len(magnitudes),
        n_intervals_near_ceiling=n_near_ceiling,
        n_outliers_excluded=n_excluded,
    )


def print_estimate(estimate: MaxPowerEstimate) -> None:
    print(f"=== {estimate.site_id} max AC power estimate ===")
    print(f"max_power_kw: {estimate.max_power_kw:.3f}  "
          f"({estimate.max_power_kwh_per_interval:.4f} kWh/interval)")
    print(f"intervals near ceiling (>=95%): {estimate.n_intervals_near_ceiling} "
          f"of {estimate.n_intervals_total} nonzero-flow intervals")
    print(f"outliers excluded: {estimate.n_outliers_excluded}")
    if estimate.n_intervals_near_ceiling < 5:
        print("NOTE: fewer than 5 intervals near the ceiling -- treat this estimate as "
              "weakly supported, similar to a low equivalent_full_cycles count in the "
              "capacity estimate.")


def plot_max_power_estimate(
    estimate: MaxPowerEstimate, battery_kwh: pd.Series, out_path: Optional[Path] = None,
):
    """Two-panel sense-check, same spirit as
    estimate_battery_capacity.py's plot_capacity_estimate:

      top: |battery_kwh| over time, with the estimated ceiling and the
           near-ceiling band marked, so a real repeated power-limit hit
           (a horizontal band of points sitting at the line) is visually
           distinguishable from an isolated one-off spike above it.
      bottom: histogram of |battery_kwh|, since the whole POINT of this
           estimate is "is there a real cluster right at some ceiling
           value" -- a histogram makes that cluster (or its absence)
           obvious in a way the time series alone doesn't.
    """
    import matplotlib.pyplot as plt

    magnitudes = battery_kwh.abs()
    nonzero = magnitudes[magnitudes > 0.0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.plot(magnitudes.index, magnitudes.to_numpy(), color='0.4', linewidth=0.5,
              label='|battery flow|')
    ax1.axhline(estimate.max_power_kwh_per_interval, color='tab:green', linestyle='--',
                linewidth=1.2, label=f'estimated ceiling ({estimate.max_power_kw:.2f} kW)')
    near_ceiling_mask = magnitudes >= 0.95 * estimate.max_power_kwh_per_interval
    ax1.scatter(magnitudes.index[near_ceiling_mask], magnitudes.to_numpy()[near_ceiling_mask],
                color='tab:red', s=10, zorder=5, label='near-ceiling interval (>=95%)')
    ax1.set_ylabel('|battery flow| (kWh/interval)')
    ax1.set_title(
        f"{estimate.site_id}: max_power={estimate.max_power_kw:.2f} kW  "
        f"({estimate.n_intervals_near_ceiling} near-ceiling of {estimate.n_intervals_total} "
        f"nonzero intervals, {estimate.n_outliers_excluded} outlier(s) excluded)"
    )
    ax1.legend(loc='upper right')

    ax2.hist(nonzero.to_numpy(), bins=80, color='0.5')
    ax2.axvline(estimate.max_power_kwh_per_interval, color='tab:green', linestyle='--',
                linewidth=1.2, label='estimated ceiling')
    ax2.set_xlabel('|battery flow| (kWh/interval)')
    ax2.set_ylabel('interval count')
    ax2.legend(loc='upper right')
    ax2.set_yscale('log')  # the near-ceiling cluster is usually a small fraction of all intervals

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    return fig


def run_checks(
    site_id: str,
    summary_csv_path: Path = Path('data/processed/energy_masters/site_summary_results.csv'),
    telemetry_path: Optional[Path] = None,
) -> MaxPowerEstimate:
    """Real-site driver: loads the battery trace via em_ingest.py, runs
    the estimate, prints diagnostics, and saves the sense-check plot --
    same shape as estimate_battery_capacity.py's run_checks, so this can
    be sense-checked by eye BEFORE reconstruct_flows.py consumes the
    number, the same way the capacity estimate was checked before being
    trusted.
    """
    from energy_masters_pipeline.scripts.em_ingest import load_valid_periods_from_summary, load_em_site_component_flows, DEFAULT_TELEMETRY_PATH

    if telemetry_path is None:
        telemetry_path = DEFAULT_TELEMETRY_PATH

    valid_periods = load_valid_periods_from_summary(site_id, summary_csv_path)
    flows_df = load_em_site_component_flows(site_id, valid_periods, telemetry_path)
    battery_kwh = flows_df['battery_kwh']

    estimate = estimate_max_power(site_id, battery_kwh)
    print_estimate(estimate)

    out_dir = Path('energy_masters_pipeline/charts/diagnostics')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{site_id}_max_power.png'
    plot_max_power_estimate(estimate, battery_kwh, out_path)
    return estimate


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] != '--synthetic':
        run_checks(sys.argv[1])
        sys.exit(0)

    # Synthetic self-test: a tight cluster near a true 5.0 kW ceiling
    # (5.0 kW * 5/60 h = 0.4167 kWh/interval), plus a scatter of smaller
    # partial-power intervals, plus one isolated outlier from a data glitch.
    np.random.seed(0)
    true_power_kwh = 5.0 * (5 / 60)
    n_ceiling_hits = 40
    n_partial = 500
    ceiling_cluster = true_power_kwh - np.random.uniform(0, 0.01, n_ceiling_hits)
    partial = np.random.exponential(scale=true_power_kwh * 0.3, size=n_partial)
    partial = np.clip(partial, 0, true_power_kwh * 0.9)  # physically bounded below the true ceiling
    outlier = np.array([true_power_kwh * 1.6])  # isolated data-glitch spike
    signs = np.random.choice([1, -1], size=n_ceiling_hits + n_partial + 1)
    all_vals = np.concatenate([ceiling_cluster, partial, outlier]) * signs
    np.random.shuffle(all_vals)

    idx = pd.date_range('2025-01-01', periods=len(all_vals), freq='5min')
    battery = pd.Series(all_vals, index=idx)

    estimate = estimate_max_power('SYNTH-001', battery)
    print_estimate(estimate)
    print(f"\n(true power was 5.0 kW = {true_power_kwh:.4f} kWh/interval, "
          f"outlier was {true_power_kwh*1.6:.4f} kWh/interval)")