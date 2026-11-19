"""Four-way dispatch comparison: actual, self-consumption, rolling, perfect foresight.

The pure core behind check_rolling_optimisation.py and, later, the horizon
sweep. Takes already-loaded inputs and returns a comparison; does no I/O and
knows nothing about the Energy Masters pipeline or the EDP pipeline, so both
can feed it and it can be tested on synthetic data.

Why energy cost rather than the full bill for the headline
----------------------------------------------------------
Daily supply and network charges are identical across all four dispatches, so
they cancel in every difference that matters. Using energy cost keeps the
per-day decomposition coherent, where apportioning a daily charge would be
arbitrary. The full bill is still computed and reported, and VCR from bills and
VCR from energy cost agree by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from model.core.site_result import SiteBill, SiteTraces


@dataclass(frozen=True)
class DispatchMetrics:
    """Headline numbers for one dispatch over one period."""
    label: str
    energy_cost: float          # $ import cost less export credit, no daily charges
    bill_total: Optional[float]  # $ household_total, None if not billed
    import_kwh: float
    export_kwh: float
    grid_to_bat_kwh: float      # arbitrage charging: the decision that needs foresight
    bat_to_grid_kwh: float      # arbitrage discharging
    sol_to_bat_kwh: float
    curtail_kwh: float
    throughput_kwh: float       # charge + discharge, AC
    equivalent_cycles: float
    terminal_soc_kwh: float


def dispatch_metrics(label: str, traces: SiteTraces, p_imp: pd.Series,
                     p_exp: pd.Series, capacity_kwh: float,
                     bill: Optional[SiteBill] = None) -> DispatchMetrics:
    charge = traces.sol_to_bat_kwh + traces.grid_to_bat_kwh
    discharge = traces.bat_to_home_kwh + traces.bat_to_grid_kwh
    throughput = float(charge.sum() + discharge.sum())
    return DispatchMetrics(
        label=label,
        energy_cost=float((traces.import_kwh * p_imp
                           - traces.export_kwh * p_exp).sum()),
        bill_total=None if bill is None else float(bill.household_total),
        import_kwh=float(traces.import_kwh.sum()),
        export_kwh=float(traces.export_kwh.sum()),
        grid_to_bat_kwh=float(traces.grid_to_bat_kwh.sum()),
        bat_to_grid_kwh=float(traces.bat_to_grid_kwh.sum()),
        sol_to_bat_kwh=float(traces.sol_to_bat_kwh.sum()),
        curtail_kwh=float(traces.curtail_kwh.sum()),
        throughput_kwh=throughput,
        equivalent_cycles=throughput / (2 * capacity_kwh) if capacity_kwh else float('nan'),
        terminal_soc_kwh=float(traces.battery_soc_kwh.iloc[-1]),
    )


@dataclass(frozen=True)
class RollingComparison:
    """Rolling optimisation set against its three reference dispatches."""
    site_id: str
    horizon_hours: float
    coarse_resolution: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    sc: DispatchMetrics
    pf: DispatchMetrics
    rolling: DispatchMetrics
    actual: Optional[DispatchMetrics]
    daily: pd.DataFrame          # per-day energy cost and headroom
    checks: tuple[tuple[str, bool, str], ...]

    @property
    def headroom(self) -> float:
        """What perfect foresight captures over self-consumption. The VCR
        denominator, and the thing every gap is measured against."""
        return self.sc.energy_cost - self.pf.energy_cost

    def vcr(self, m: DispatchMetrics) -> float:
        h = self.headroom
        return (self.sc.energy_cost - m.energy_cost) / h if abs(h) > 1e-9 else float('nan')

    @property
    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.checks)


def build_daily(sc: SiteTraces, pf: SiteTraces, rolling: SiteTraces,
                actual: Optional[SiteTraces],
                p_imp: pd.Series, p_exp: pd.Series) -> pd.DataFrame:
    """Per-day energy cost for each dispatch, plus headroom and VCR.

    The per-day view is where a rolling run either makes sense or does not.
    Value is concentrated on a handful of extreme price days, so a headline
    VCR can look reasonable while the dispatch is wrong on exactly the days
    that matter. Sorting by headroom puts those days first.
    """
    def cost(t):
        return (t.import_kwh * p_imp - t.export_kwh * p_exp).groupby(
            t.import_kwh.index.date).sum()

    df = pd.DataFrame({'sc': cost(sc), 'pf': cost(pf), 'rolling': cost(rolling)})
    if actual is not None:
        df['actual'] = cost(actual)
    df['headroom'] = df['sc'] - df['pf']
    with np.errstate(divide='ignore', invalid='ignore'):
        df['vcr_rolling'] = (df['sc'] - df['rolling']) / df['headroom']
        if actual is not None:
            df['vcr_actual'] = (df['sc'] - df['actual']) / df['headroom']
    df.index.name = 'date'
    return df.sort_values('headroom', ascending=False)


def run_checks(sc: DispatchMetrics, pf: DispatchMetrics, rolling: DispatchMetrics,
               traces: SiteTraces, solar: np.ndarray, load: np.ndarray,
               diagnostics, capacity_kwh: float, soc_min_kwh: float,
               soc_target_kwh: float, n_intervals: int,
               fallback_tolerance: float = 0.01) -> tuple:
    """Checks that must hold for the result to be worth interpreting.

    Ordered so that the most fundamental failures appear first: a broken
    invariant makes every downstream number meaningless, whereas a slightly
    high fallback count merely warrants caution.
    """
    out = []

    def chk(name, ok, detail=''):
        out.append((name, bool(ok), detail))

    tol = 1e-7
    chk('import == grid_to_home + grid_to_bat',
        np.allclose(traces.import_kwh, traces.grid_to_home_kwh + traces.grid_to_bat_kwh, atol=tol))
    chk('export == sol_to_grid + bat_to_grid',
        np.allclose(traces.export_kwh, traces.sol_to_grid_kwh + traces.bat_to_grid_kwh, atol=tol))
    chk('load balance',
        np.allclose(load, traces.sol_to_home_kwh + traces.bat_to_home_kwh
                    + traces.grid_to_home_kwh, atol=tol))
    chk('solar balance',
        np.allclose(solar, traces.sol_to_home_kwh + traces.sol_to_bat_kwh
                    + traces.sol_to_grid_kwh + traces.curtail_kwh, atol=tol))
    chk('no NaN in traces',
        not any(np.isnan(getattr(traces, f).to_numpy()).any()
                for f in ('import_kwh', 'export_kwh', 'battery_soc_kwh')))

    soc = traces.battery_soc_kwh
    chk('SOC within [reserve, capacity]',
        soc.min() >= soc_min_kwh - 1e-6 and soc.max() <= capacity_kwh + 1e-6,
        f"min {soc.min():.3f}, max {soc.max():.3f}, "
        f"bounds [{soc_min_kwh:.3f}, {capacity_kwh:.3f}]")
    chk('terminal SOC hits the global target',
        abs(rolling.terminal_soc_kwh - soc_target_kwh) < 1e-5,
        f"{rolling.terminal_soc_kwh:.5f} vs {soc_target_kwh:.5f}")

    chk('rolling does not beat perfect foresight',
        rolling.energy_cost >= pf.energy_cost - 1e-6,
        f"rolling ${rolling.energy_cost:,.2f} vs PF ${pf.energy_cost:,.2f}")
    chk('rolling is at least as good as self-consumption',
        rolling.energy_cost <= sc.energy_cost + 1e-6,
        f"rolling ${rolling.energy_cost:,.2f} vs SC ${sc.energy_cost:,.2f}")

    chk('causality', diagnostics.causality_ok,
        f"max run-time lead {diagnostics.max_run_time_lead_minutes:.2f} min")
    frac = diagnostics.n_fallbacks / max(1, n_intervals)
    chk(f'self-consumption fallbacks below {fallback_tolerance:.0%}',
        frac <= fallback_tolerance,
        f"{diagnostics.n_fallbacks:,} of {n_intervals:,} ({frac:.2%})")
    return tuple(out)


def format_report(cmp_: RollingComparison, diagnostics, n_days: int = 8) -> str:
    """The block a human reads to decide whether the run makes sense."""
    L = []
    L.append(f"=== rolling_optimisation check: {cmp_.site_id} ===")
    L.append(f"period   {cmp_.period_start:%Y-%m-%d} to {cmp_.period_end:%Y-%m-%d}"
             f"   horizon {cmp_.horizon_hours:g} h   coarse {cmp_.coarse_resolution}")
    L.append("")

    rows = [cmp_.sc, cmp_.rolling, cmp_.pf]
    if cmp_.actual is not None:
        rows.insert(1, cmp_.actual)
    L.append(f"{'dispatch':<22}{'energy $':>10}{'VCR':>8}{'grid>bat':>10}"
             f"{'bat>grid':>10}{'cycles':>8}{'curtail':>9}")
    L.append("-" * 77)
    for m in rows:
        v = cmp_.vcr(m)
        L.append(f"{m.label:<22}{m.energy_cost:>10,.2f}{v:>8.3f}"
                 f"{m.grid_to_bat_kwh:>10,.0f}{m.bat_to_grid_kwh:>10,.0f}"
                 f"{m.equivalent_cycles:>8.1f}{m.curtail_kwh:>9,.0f}")
    L.append("-" * 77)
    L.append(f"headroom (SC - PF): ${cmp_.headroom:,.2f}")
    L.append("")

    L.append("Top days by headroom -- value concentrates here, so this is where")
    L.append("a rolling result either makes sense or does not. A single day's VCR")
    L.append("can exceed 1.0: the daily split does not carry the global terminal")
    L.append("SOC constraint, so a day can beat perfect foresight locally by")
    L.append("borrowing charge from its neighbours. Only the period VCR is bounded.")
    cols = [c for c in ('headroom', 'sc', 'actual', 'rolling', 'pf',
                        'vcr_actual', 'vcr_rolling') if c in cmp_.daily.columns]
    L.append(cmp_.daily.head(n_days)[cols].to_string(
        float_format=lambda v: f"{v:,.3f}"))
    L.append("")

    L.append(diagnostics.summary())
    L.append("")

    L.append("Checks:")
    for name, ok, detail in cmp_.checks:
        tag = 'PASS' if ok else 'FAIL'
        L.append(f"  [{tag}] {name}" + (f"   {detail}" if detail else ''))
    L.append("")
    L.append(f"OVERALL: {'PASS' if cmp_.passed else 'FAIL'}")
    return "\n".join(L)
