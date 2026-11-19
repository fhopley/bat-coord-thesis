"""Site export limit estimation.

Same method as estimate_battery_max_power.py, applied to the site's total
export (sol_to_grid + bat_to_grid) instead of battery flow. A genuine
export-limit hit (inverter/network connection agreement cap) shows up as
repeated clipping at a ceiling value, exactly like a battery power limit.

Simpler than the battery case: site export magnitude is max(-ac_load_net, 0)
directly -- ac_load_net is a single independently-measured channel (+import/
-export), not something reconstruct_flows.py has to allocate among several
possible destinations, so there's no ambiguity to resolve first.

Duplicates the small (~30 line) gap-detection helper from
estimate_battery_max_power.py rather than importing it, for the same reason
that file gives: keeping dev scripts independent is cheaper than coupling
them through a private interface for an algorithm this short.

A synthetic self-test (see __main__) covers the isolated-outlier case,
same as estimate_battery_max_power.py.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ExportLimitEstimate:
    site_id: str
    export_limit_kwh_per_interval: float
    export_limit_kw: float
    n_intervals_total: int
    n_intervals_near_ceiling: int
    n_outliers_excluded: int


def _iterative_gap_outlier_filter(
    values: list[float],
    scale_reference: float,
    window_frac: float = 0.15,
    min_window: int = 15,
    gap_multiplier: float = 4.0,
    min_absolute_gap_fraction: float = 0.02,
) -> tuple[list[float], int]:
    """Identical method to estimate_battery_max_power.py's version -- see
    that file's docstring for the full reasoning. Returns
    (clean_values, n_excluded)."""
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


def estimate_export_limit(
    site_id: str,
    ac_load_net_kwh: pd.Series,
    interval_hours: float = 5 / 60,
) -> ExportLimitEstimate:
    """Estimate the site's real export limit (kW) from its measured net
    grid flow.

    Args:
        ac_load_net_kwh: measured net grid flow, +import/-export, over the
            site's valid period(s). Same already-cleaned channel
            em_ingest.py produces (ac_load_net_kwh column of
            load_em_site_component_flows), not re-derived here.
    """
    export_magnitudes = (-ac_load_net_kwh).clip(lower=0.0).to_numpy(dtype=float, copy=False)
    export_magnitudes = export_magnitudes[export_magnitudes > 0.0]
    if len(export_magnitudes) == 0:
        raise ValueError(f"{site_id}: no nonzero export intervals, cannot estimate export limit")

    scale_guess = float(export_magnitudes.max())
    clean_vals, n_excluded = _iterative_gap_outlier_filter(list(export_magnitudes), scale_guess)

    export_limit_kwh_per_interval = max(clean_vals) if clean_vals else scale_guess
    export_limit_kw = export_limit_kwh_per_interval / interval_hours

    near_ceiling_threshold = 0.95 * export_limit_kwh_per_interval
    n_near_ceiling = int((export_magnitudes >= near_ceiling_threshold).sum())

    return ExportLimitEstimate(
        site_id=site_id,
        export_limit_kwh_per_interval=export_limit_kwh_per_interval,
        export_limit_kw=export_limit_kw,
        n_intervals_total=len(export_magnitudes),
        n_intervals_near_ceiling=n_near_ceiling,
        n_outliers_excluded=n_excluded,
    )


def print_estimate(estimate: ExportLimitEstimate) -> None:
    print(f"=== {estimate.site_id} export limit estimate ===")
    print(f"export_limit_kw: {estimate.export_limit_kw:.3f}  "
          f"({estimate.export_limit_kwh_per_interval:.4f} kWh/interval)")
    print(f"intervals near ceiling (>=95%): {estimate.n_intervals_near_ceiling} "
          f"of {estimate.n_intervals_total} nonzero-export intervals")
    print(f"outliers excluded: {estimate.n_outliers_excluded}")
    if estimate.n_intervals_near_ceiling < 5:
        print("NOTE: fewer than 5 intervals near the ceiling -- treat this estimate as "
              "weakly supported (the site may rarely or never have actually been "
              "export-limited within its valid period).")


def plot_export_limit_estimate(
    estimate: ExportLimitEstimate, ac_load_net_kwh: pd.Series,
    interval_hours: float = 5 / 60, out_path: Optional[Path] = None,
):
    import matplotlib.pyplot as plt

    export_magnitudes = (-ac_load_net_kwh).clip(lower=0.0)
    nonzero = export_magnitudes[export_magnitudes > 0.0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.plot(export_magnitudes.index, export_magnitudes.to_numpy(), color='0.4',
              linewidth=0.5, label='site export (sol_to_grid + bat_to_grid)')
    ax1.axhline(estimate.export_limit_kwh_per_interval, color='tab:green', linestyle='--',
                linewidth=1.2, label=f'estimated limit ({estimate.export_limit_kw:.2f} kW)')
    near_ceiling_mask = export_magnitudes >= 0.95 * estimate.export_limit_kwh_per_interval
    ax1.scatter(export_magnitudes.index[near_ceiling_mask],
                export_magnitudes.to_numpy()[near_ceiling_mask],
                color='tab:red', s=10, zorder=5, label='near-ceiling interval (>=95%)')
    ax1.set_ylabel('export (kWh/interval)')
    ax1.set_title(
        f"{estimate.site_id}: export_limit={estimate.export_limit_kw:.2f} kW  "
        f"({estimate.n_intervals_near_ceiling} near-ceiling of {estimate.n_intervals_total} "
        f"nonzero intervals, {estimate.n_outliers_excluded} outlier(s) excluded)"
    )
    ax1.legend(loc='upper right')

    ax2.hist(nonzero.to_numpy(), bins=80, color='0.5')
    ax2.axvline(estimate.export_limit_kwh_per_interval, color='tab:green', linestyle='--',
                linewidth=1.2, label='estimated limit')
    ax2.set_xlabel('export (kWh/interval)')
    ax2.set_ylabel('interval count')
    ax2.legend(loc='upper right')
    ax2.set_yscale('log')

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    return fig


def run_checks(
    site_id: str,
    summary_csv_path: Path = Path('data/processed/energy_masters/site_summary_results.csv'),
    telemetry_path: Optional[Path] = None,
) -> ExportLimitEstimate:
    from energy_masters_pipeline.scripts.em_ingest import load_valid_periods_from_summary, load_em_site_component_flows, DEFAULT_TELEMETRY_PATH

    if telemetry_path is None:
        telemetry_path = DEFAULT_TELEMETRY_PATH

    valid_periods = load_valid_periods_from_summary(site_id, summary_csv_path)
    flows_df = load_em_site_component_flows(site_id, valid_periods, telemetry_path)
    ac_load_net_kwh = flows_df['ac_load_net_kwh']

    estimate = estimate_export_limit(site_id, ac_load_net_kwh)
    print_estimate(estimate)

    out_dir = Path('energy_masters_pipeline/charts/diagnostics')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{site_id}_export_limit.png'
    plot_export_limit_estimate(estimate, ac_load_net_kwh, out_path=out_path)
    return estimate


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] != '--synthetic':
        run_checks(sys.argv[1])
        sys.exit(0)

    # Synthetic self-test: a tight cluster near a true 5.0 kW export limit
    # (5.0 kW * 5/60 h = 0.4167 kWh/interval), plus a scatter of smaller
    # partial exports, plus one isolated outlier from a data glitch.
    np.random.seed(0)
    true_limit_kwh = 5.0 * (5 / 60)
    n_ceiling_hits = 40
    n_partial = 500
    ceiling_cluster = true_limit_kwh - np.random.uniform(0, 0.01, n_ceiling_hits)
    partial = np.random.exponential(scale=true_limit_kwh * 0.3, size=n_partial)
    partial = np.clip(partial, 0, true_limit_kwh * 0.9)
    outlier = np.array([true_limit_kwh * 1.6])
    export_vals = np.concatenate([ceiling_cluster, partial, outlier])
    np.random.shuffle(export_vals)
    # Build ac_load_net = -export (all export, no import, for this simple test)
    idx = pd.date_range('2025-01-01', periods=len(export_vals), freq='5min')
    ac_load_net = pd.Series(-export_vals, index=idx)

    estimate = estimate_export_limit('SYNTH-001', ac_load_net)
    print_estimate(estimate)
    print(f"\n(true limit was 5.0 kW = {true_limit_kwh:.4f} kWh/interval, "
          f"outlier was {true_limit_kwh*1.6:.4f} kWh/interval)")