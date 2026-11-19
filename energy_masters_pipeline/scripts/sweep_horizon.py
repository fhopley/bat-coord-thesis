"""Sweep the rolling optimisation horizon, and optionally battery size.

Answers one question: how short can the planning horizon be before value is
lost? It matters for three reasons.

  1. If the answer is at or below 15 hours, the horizon fits inside the
     15.08 h that AEMO pre-dispatch guarantees from any decision instant, so
     the infill rule never fires for that source and no reported number rests
     on extrapolated prices.
  2. Compute scales with horizon. 12 h against 48 h is roughly fourfold across
     the full EDP run.
  3. The horizon that suffices should scale with storage duration, and the EDP
     battery sweep runs from 5 kWh to 50 kWh.

Uses TruncatedPerfectForecastProvider, so no pre-dispatch store is required:
real prices restricted to the pre-dispatch availability structure, with zero
forecast error. That is the point. Any value lost here is lost to the horizon,
the terminal pin or coarse resolution, never to a bad forecast.

DECISION RULE, fixed before looking at results because the outcome determines a
fourfold compute difference and would otherwise be easy to rationalise after
the fact:

    Adopt the shortest horizon H such that, across every battery size and
    season tested, cohort VCR at H is within 0.005 of the best cohort VCR
    observed, and no individual site above the headroom floor loses more
    than 0.02.

Cohort VCR is computed from summed numerator over summed denominator, never a
mean of per-site ratios, matching run_cohort_attribution: a mean of ratios
silently overweights sites with little headroom.

Usage (from project root):
    python3 -m energy_masters_pipeline.scripts.sweep_horizon
    python3 -m energy_masters_pipeline.scripts.sweep_horizon --sites EM-427,EM-414
    python3 -m energy_masters_pipeline.scripts.sweep_horizon --horizons 6,12,24,48
    python3 -m energy_masters_pipeline.scripts.sweep_horizon --sites EM-427 \\
        --battery-kwh 5,10,20,35,50 --horizons 6,12,24,48
"""
from __future__ import annotations

import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from model.components.rolling_optimisation import rolling_optimisation
from model.core.prices import build_price_stack_offsets
from model.core.scenario import RollingConfig
from model.dev.progress import ConsoleProgress
from model.forecast import TruncatedPerfectForecastProvider

DEFAULT_HORIZONS = (3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 24.0, 48.0)

# Power rating applied when capacity is swept, matching the EDP battery specs
# in model/configs/full_run_optimised_amber.yaml: half the capacity in kW,
# capped at 10 kW. The cap is what makes STORAGE DURATION rise with capacity
# (5 kWh is 2 h, 50 kWh is 5 h), and duration is the thing the horizon is
# expected to scale with. Using an uncapped capacity/2 would hold duration
# fixed at 2 h for every size and the sweep would measure nothing.
POWER_KW_RATIO = 0.5
POWER_KW_CAP = 10.0
DEFAULT_COARSE = '30min'
DEFAULT_OUT_DIR = Path('data/processed/energy_masters')
DEFAULT_CSV = DEFAULT_OUT_DIR / 'horizon_sweep.csv'
DEFAULT_DAILY_CSV = DEFAULT_OUT_DIR / 'horizon_sweep_daily.csv'
# EDP results go to their own directory. Sharing the Energy Masters paths would
# let one run silently overwrite the other, and the two are not comparable:
# different cohorts, different periods, different batteries.
DEFAULT_EDP_OUT_DIR = Path('data/processed/edp')
DEFAULT_EDP_CSV = DEFAULT_EDP_OUT_DIR / 'horizon_sweep_edp.csv'
DEFAULT_EDP_DAILY_CSV = DEFAULT_EDP_OUT_DIR / 'horizon_sweep_edp_daily.csv'

# Below this, VCR is a ratio of two small numbers and says nothing about
# coordination quality. Same threshold and reasoning as run_cohort_attribution.
SITE_HEADROOM_FLOOR_DOLLARS = 5.0

# Decision rule thresholds.
COHORT_TOLERANCE = 0.005
SITE_TOLERANCE = 0.02


# ---------------------------------------------------------------------------
# Site inputs
# ---------------------------------------------------------------------------

class SiteInputs:
    """Everything a sweep needs for one site, loaded once.

    Loading is the expensive part and none of it depends on horizon or battery
    size, so it is hoisted out of the loop. The forecast provider matters most:
    it materialises its RunSets at construction and is invariant to both swept
    dimensions, so building it once per site is most of the saving.
    """

    def __init__(self, site_id, index, solar, load, prices_mwh, p_imp, p_exp,
                 offsets, battery_spec, soc_init_kwh, soc_final_kwh,
                 export_limit_kwh_per_interval, interval_hours,
                 curtail_cost_per_kwh_dollars, sc_cost, pf_cost, actual_cost,
                 daily_baseline=None):
        self.site_id = site_id
        self.index = index
        self.solar = solar
        self.load = load
        self.prices_mwh = prices_mwh
        self.p_imp = p_imp
        self.p_exp = p_exp
        self.offsets = offsets
        self.battery_spec = battery_spec
        self.soc_init_kwh = soc_init_kwh
        self.soc_final_kwh = soc_final_kwh
        self.export_limit_kwh_per_interval = export_limit_kwh_per_interval
        self.interval_hours = interval_hours
        self.curtail_cost_per_kwh_dollars = curtail_cost_per_kwh_dollars
        self.sc_cost = sc_cost
        self.pf_cost = pf_cost
        self.actual_cost = actual_cost
        # Per-day SC and PF costs for THIS site's own battery. Only used on the
        # native-battery path; when capacity is swept, baselines_for produces a
        # fresh frame per size.
        self.daily_baseline = daily_baseline
        self._providers: dict[str, TruncatedPerfectForecastProvider] = {}

    @property
    def headroom(self) -> float:
        return self.sc_cost - self.pf_cost

    def provider(self, coarse_resolution: str):
        if coarse_resolution not in self._providers:
            self._providers[coarse_resolution] = TruncatedPerfectForecastProvider(
                self.prices_mwh,
                interval_minutes=int(round(self.interval_hours * 60)),
                coarse_resolution=coarse_resolution,
            )
        return self._providers[coarse_resolution]


def baselines_for(inputs: "SiteInputs", spec):
    """Recompute the self-consumption and perfect foresight baselines.

    Needed whenever battery capacity is swept, because both baselines are
    properties of the battery, not just of the site. Returns
    (sc_cost, pf_cost, soc_init_kwh, soc_final_kwh, daily), where `daily` is a
    DataFrame indexed by date with columns cost_sc and cost_pf.

    Costs about 8 seconds per size against roughly 10 minutes for that size's
    horizon ladder, so it is cheap insurance against comparing a dispatch to
    the wrong benchmark.
    """
    from model.components.perfect_foresight_lp import perfect_foresight_lp
    from model.components.self_consumption import self_consumption, run_sc_with_battery

    soc_init = (spec.backup_reserve_pct / 100.0) * spec.capacity_kwh
    soc_arr, _ = run_sc_with_battery(
        inputs.solar.to_numpy(), inputs.load.to_numpy(), spec, soc_init,
        inputs.interval_hours)
    soc_final = float(soc_arr[-1])

    sc = self_consumption(inputs.site_id, inputs.solar, inputs.load, spec,
                          interval_hours=inputs.interval_hours)
    pf = perfect_foresight_lp(
        inputs.site_id, inputs.solar, inputs.load, inputs.p_imp, inputs.p_exp,
        spec, soc_init, soc_final,
        export_limit_kwh_per_interval=inputs.export_limit_kwh_per_interval,
        interval_hours=inputs.interval_hours,
        curtail_cost_per_kwh_dollars=inputs.curtail_cost_per_kwh_dollars)
    daily = pd.DataFrame({
        'cost_sc': daily_energy_cost(sc, inputs.p_imp, inputs.p_exp),
        'cost_pf': daily_energy_cost(pf, inputs.p_imp, inputs.p_exp),
    })
    return (energy_cost(sc, inputs.p_imp, inputs.p_exp),
            energy_cost(pf, inputs.p_imp, inputs.p_exp),
            soc_init, soc_final, daily)


def energy_cost(traces, p_imp, p_exp) -> float:
    """Import cost less export credit, excluding daily charges.

    Daily charges are identical across dispatches so they cancel in every
    difference that matters, and excluding them keeps the arithmetic legible.
    VCR from energy cost and VCR from full bills agree by construction.
    """
    return float((traces.import_kwh * p_imp - traces.export_kwh * p_exp).sum())


def daily_energy_cost(traces, p_imp, p_exp) -> pd.Series:
    """energy_cost grouped by calendar date.

    Retained because a full-year sweep is an overnight run and the seasonal
    question -- does an 18 hour horizon cost more in winter than in summer --
    is answerable from per-day costs without re-running any of it. Aggregating
    to a period total at write time would throw that away and there would be no
    cheap way to get it back.

    Grouped on `index.date` of the interval-END label, matching
    run_cohort_rungs, so the midnight interval falls in the preceding day. The
    convention only has to be consistent, and it is the one already in use.
    """
    cost = traces.import_kwh * p_imp - traces.export_kwh * p_exp
    return cost.groupby(cost.index.date).sum()


def simulation_inputs(a) -> tuple[pd.Series, pd.Series]:
    """Recover the EXACT solar and load the SC and PF baselines were given.

    load_and_analyze clips gross_load to >= 0 before simulating, because a
    handful of intervals carry unexplained negative load, but it leaves
    a.trace['gross_load_kwh'] unclipped. Taking solar and load from the trace
    therefore hands rolling optimisation a marginally different problem from
    the one its own benchmarks solved, which is an unfair comparison however
    small the volume.

    Rather than re-implement the clipping and risk drifting from it, both
    series are reconstructed from sc_result via the SiteTraces invariants:

        load  == sol_to_home + bat_to_home + grid_to_home
        solar == sol_to_home + sol_to_bat + sol_to_grid   (SC never curtails)

    These hold exactly by construction, so whatever load_and_analyze did to
    its inputs is reproduced here without knowing what it was. sol_to_home
    mirrors the sign of solar, so a negative parasitic-draw interval survives
    correctly.
    """
    sc = a.sc_result
    load = sc.sol_to_home_kwh + sc.bat_to_home_kwh + sc.grid_to_home_kwh
    solar = sc.sol_to_home_kwh + sc.sol_to_bat_kwh + sc.sol_to_grid_kwh
    if (load < -1e-9).any():
        raise ValueError(
            f"reconstructed load has {int((load < -1e-9).sum())} negative "
            "interval(s); the SiteTraces invariant should make this impossible"
        )
    return solar.astype(float), load.astype(float)


def load_em_site(site_id: str, c_deg_per_kwh_override=None,
                 curtail_cost_per_kwh_dollars=None) -> SiteInputs:
    """Load one Energy Masters site, including its SC, PF and actual baselines."""
    from energy_masters_pipeline.scripts.viz_energy_masters import (
        load_and_analyze, DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS,
    )
    if curtail_cost_per_kwh_dollars is None:
        curtail_cost_per_kwh_dollars = DEFAULT_CURTAIL_COST_PER_KWH_DOLLARS

    a = load_and_analyze(
        site_id, simulate_lp=True,
        curtail_cost_per_kwh_dollars=curtail_cost_per_kwh_dollars,
        c_deg_per_kwh_override=c_deg_per_kwh_override,
    )
    if a.prices_mwh is None:
        raise RuntimeError(
            "EMAnalysis.prices_mwh is None; this needs the version of "
            "viz_energy_masters.py that populates it."
        )
    index = a.actual_result.import_kwh.index
    sim_solar, sim_load = simulation_inputs(a)
    return SiteInputs(
        site_id=site_id,
        index=index,
        solar=sim_solar,
        load=sim_load,
        prices_mwh=a.prices_mwh,
        p_imp=a.p_imp, p_exp=a.p_exp,
        offsets=build_price_stack_offsets(index, a.network_tariff,
                                          a.retail_tariff, a.state,
                                          a.interval_hours),
        battery_spec=a.battery_spec,
        soc_init_kwh=a.soc_init_kwh,
        soc_final_kwh=a.soc_final_kwh,
        export_limit_kwh_per_interval=(
            np.inf if a.export_limit_kw is None
            else a.export_limit_kw * a.interval_hours),
        interval_hours=a.interval_hours,
        curtail_cost_per_kwh_dollars=a.curtail_cost_per_kwh_dollars,
        sc_cost=energy_cost(a.sc_result, a.p_imp, a.p_exp),
        pf_cost=energy_cost(a.lp_result, a.p_imp, a.p_exp),
        actual_cost=energy_cost(a.actual_result, a.p_imp, a.p_exp),
        daily_baseline=pd.DataFrame({
            'cost_sc': daily_energy_cost(a.sc_result, a.p_imp, a.p_exp),
            'cost_pf': daily_energy_cost(a.lp_result, a.p_imp, a.p_exp),
        }),
    )


def load_edp_site(
    site_id: str,
    year: int,
    state: str,
    network_tariff_id: str,
    retail_tariff_id: str,
    battery_spec,
    export_limit_kw: float = 10.0,
    months: Optional[tuple] = None,
    project_root=None,
    curtail_cost_per_kwh_dollars: float = 0.0,
) -> SiteInputs:
    """Load one EDP site-year, optionally restricted to selected months.

    Exists to answer the one question the Energy Masters cohort cannot. The EM
    window is September 2025 to March 2026, southern-hemisphere summer, and the
    failure mode that makes short horizons lose value is the overnight-charge
    into-morning-peak cycle, which is largely a WINTER phenomenon. So the EM
    sweep is systematically biased toward concluding short horizons are fine.
    EDP spans 2021 to 2025 across four states and every season.

    Needs no forecast store: TruncatedPerfectForecastProvider is built from
    actual prices, which we already have for the whole EDP period.

    Args:
        months: 1-based month numbers to keep, e.g. (6, 7, 8) for winter. The
            slice is applied BEFORE the baselines are computed, so the
            self-consumption and perfect foresight benchmarks cover exactly the
            same intervals as the rolling run. Slicing afterwards would compare
            against a benchmark for a different period.
        export_limit_kw: Per-phase export limit. EDP has no per-site review
            override, so this is a modelling choice rather than a site fact.
    """
    from model.core.paths import default_paths
    from model.dev.viz_perfect_foresight_lp import load_inputs

    paths = default_paths(project_root)
    trace, prices_mwh, network_tariff, retail_tariff = load_inputs(
        site_id, year, state, network_tariff_id, retail_tariff_id, paths)

    if months:
        # Restrict to months WITHIN the requested year. Under interval-end
        # labelling an EDP year file ends at YYYY+1-01-01 00:00, so a bare
        # month mask on month 1 would also match that trailing interval and
        # produce a spurious year-long gap.
        keep = (trace.index.month.isin(tuple(months))
                & (trace.index.year == year))
        if keep.sum() < 288:
            raise ValueError(
                f"{site_id} {year}: months {months} leave "
                f"{int(keep.sum())} intervals, under one day")
        trace = trace[keep]
        prices_mwh = prices_mwh[keep]
        # A month selection that wraps the year end, e.g. 12,1,2, concatenates
        # December to the FOLLOWING January of the same calendar year, leaving
        # a nine-month hole in the middle. Rolling optimisation assumes a
        # contiguous period and the forecast providers index by position on a
        # regular grid, so a hole does not raise: windows past it simply find
        # no run and fall back to self-consumption, capping VCR at a value that
        # looks like a horizon result and is not. Refuse it.
        step = pd.Timedelta(minutes=5)
        gaps = trace.index.to_series().diff().dropna()
        if (gaps != step).any():
            bad = gaps[gaps != step]
            raise ValueError(
                f"{site_id} {year}: months {tuple(months)} leave a "
                f"non-contiguous period ({len(bad)} gap(s), largest "
                f"{bad.max()}). Rolling optimisation requires a contiguous "
                "run. Use consecutive months within one calendar year, e.g. "
                "6,7,8 for winter; for a summer window use a single year's "
                "1,2 or 12 rather than wrapping."
            )

    index = trace.index
    # EDP column names differ from Energy Masters: pv_site_net_kwh rather than
    # pv_dc_kwh. Matches what viz_perfect_foresight_lp.run_lp_analysis uses.
    solar = trace['pv_site_net_kwh'].astype(float)
    load = trace['gross_load_kwh'].astype(float)
    interval_hours = 5 / 60

    p_imp, p_exp = _build_price_vectors(index, prices_mwh, network_tariff,
                                        retail_tariff, state, interval_hours)
    inputs = SiteInputs(
        site_id=f"{site_id}_{year}" + (f"_m{'-'.join(str(m) for m in months)}"
                                       if months else ""),
        index=index, solar=solar, load=load, prices_mwh=prices_mwh,
        p_imp=p_imp, p_exp=p_exp,
        offsets=build_price_stack_offsets(index, network_tariff, retail_tariff,
                                          state, interval_hours),
        battery_spec=battery_spec,
        soc_init_kwh=0.0, soc_final_kwh=0.0,       # replaced below
        export_limit_kwh_per_interval=export_limit_kw * interval_hours,
        interval_hours=interval_hours,
        curtail_cost_per_kwh_dollars=curtail_cost_per_kwh_dollars,
        sc_cost=np.nan, pf_cost=np.nan,
        actual_cost=np.nan,                         # EDP has no actual dispatch
    )
    sc_cost, pf_cost, soc_init, soc_final, daily = baselines_for(
        inputs, battery_spec)
    inputs.sc_cost, inputs.pf_cost = sc_cost, pf_cost
    inputs.soc_init_kwh, inputs.soc_final_kwh = soc_init, soc_final
    inputs.daily_baseline = daily
    return inputs


def _build_price_vectors(index, prices_mwh, network_tariff, retail_tariff,
                         state, interval_hours):
    from model.core.prices import build_price_vectors
    return build_price_vectors(index, prices_mwh, network_tariff,
                               retail_tariff, state, interval_hours)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_site(
    inputs: SiteInputs,
    horizons=DEFAULT_HORIZONS,
    coarse_resolutions=(DEFAULT_COARSE,),
    battery_kwh: Optional[tuple] = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every (horizon, coarse resolution, battery size) for one site.

    Fail-soft per combination. A single infeasible or otherwise broken run must
    not lose the rest of the site's ladder, and a gap changes which horizons
    are comparable, so failures become NaN rows rather than disappearing.

    When capacity is swept, the SC and PF baselines are recomputed for each
    size, so VCR is comparable both across horizons and across sizes.

    Returns (per-run rows, per-run-per-day rows). The second frame exists so a
    full-year sweep can be sliced by season afterwards without re-running.
    """
    sizes = battery_kwh or (None,)
    rows = []
    daily_rows = []
    total = len(horizons) * len(coarse_resolutions) * len(sizes)
    n = 0
    for size in sizes:
        if size is None:
            spec = inputs.battery_spec
            soc_init, soc_final = inputs.soc_init_kwh, inputs.soc_final_kwh
            sc_cost, pf_cost = inputs.sc_cost, inputs.pf_cost
            daily_base = inputs.daily_baseline
        else:
            spec = replace(inputs.battery_spec, capacity_kwh=float(size),
                           power_kw=min(float(size) * POWER_KW_RATIO, POWER_KW_CAP))
            # Baselines MUST be recomputed for the swept capacity. The site's
            # own SC and PF runs used its own battery, so reusing them would
            # measure a big battery's dispatch against a small battery's
            # benchmark: VCR would run far above 1 and the decision rule's
            # absolute 0.005 tolerance would become proportionally far tighter,
            # biasing every size toward a longer horizon.
            print(f"  recomputing SC/PF baselines for {size:g} kWh "
                  f"/ {spec.power_kw:g} kW ...", flush=True)
            sc_cost, pf_cost, soc_init, soc_final, daily_base = baselines_for(
                inputs, spec)
        headroom = sc_cost - pf_cost
        for coarse in coarse_resolutions:
            provider = inputs.provider(coarse)
            for hh in horizons:
                n += 1
                label = (f"[{n}/{total}] {inputs.site_id} h={hh:g}h "
                         f"{coarse}" + (f" {size:g}kWh" if size else ""))
                cfg = RollingConfig(forecast_source='truncated_perfect',
                                    horizon_hours=hh)
                t0 = time.perf_counter()
                try:
                    with ConsoleProgress(label, enabled=show_progress) as prog:
                        traces, dg = rolling_optimisation(
                            inputs.site_id, inputs.solar, inputs.load,
                            provider, inputs.offsets, spec,
                            soc_init, soc_final, cfg,
                            export_limit_kwh_per_interval=(
                                inputs.export_limit_kwh_per_interval),
                            interval_hours=inputs.interval_hours,
                            curtail_cost_per_kwh_dollars=(
                                inputs.curtail_cost_per_kwh_dollars),
                            progress=prog,
                        )
                    cost = energy_cost(traces, inputs.p_imp, inputs.p_exp)
                    if daily_base is not None:
                        d = daily_base.copy()
                        d['cost_rolling'] = daily_energy_cost(
                            traces, inputs.p_imp, inputs.p_exp)
                        d['headroom'] = d['cost_sc'] - d['cost_pf']
                        with np.errstate(divide='ignore', invalid='ignore'):
                            d['vcr'] = ((d['cost_sc'] - d['cost_rolling'])
                                        / d['headroom'])
                        d.insert(0, 'horizon_hours', hh)
                        d.insert(0, 'coarse_resolution', coarse)
                        d.insert(0, 'battery_group',
                                 'site_native' if size is None
                                 else f"{float(size):g}kWh")
                        d.insert(0, 'site_id', inputs.site_id)
                        daily_rows.append(d.rename_axis('date').reset_index())
                    row = dict(
                        cost_rolling=cost,
                        vcr=((sc_cost - cost) / headroom
                             if abs(headroom) > 1e-9 else np.nan),
                        infill_fraction=dg.infill_fraction_mean,
                        n_fallbacks=dg.n_fallbacks,
                        solve_ms_median=dg.solve_ms_median,
                        causality_ok=dg.causality_ok,
                        error='',
                    )
                except Exception as e:                      # noqa: BLE001
                    row = dict(cost_rolling=np.nan, vcr=np.nan,
                               infill_fraction=np.nan, n_fallbacks=-1,
                               solve_ms_median=np.nan, causality_ok=False,
                               error=f"{type(e).__name__}: {e}")
                    print(f"  {label} FAILED: {row['error']}")
                    traceback.print_exc()
                wall = time.perf_counter() - t0
                rows.append(dict(
                    site_id=inputs.site_id, horizon_hours=hh,
                    coarse_resolution=coarse,
                    battery_kwh=spec.capacity_kwh,
                    # Grouping key for cohort aggregation. When sizes are swept
                    # explicitly, every site runs the same capacity and it is a
                    # meaningful group. When each site uses its OWN battery,
                    # capacity differs per site, so grouping on it would split
                    # the cohort into one row per site and silently defeat the
                    # aggregation. That is a real trap: the table still renders.
                    battery_group=('site_native' if size is None
                                   else f"{float(size):g}kWh"),
                    n_intervals=len(inputs.index),
                    cost_sc=sc_cost, cost_pf=pf_cost,
                    cost_actual=inputs.actual_cost,
                    headroom=headroom,
                    vcr_actual=((sc_cost - inputs.actual_cost) / headroom
                                if (size is None and abs(headroom) > 1e-9)
                                else np.nan),
                    wall_seconds=wall, **row))
                print(f"  {label}: VCR {row['vcr']:.4f}  "
                      f"infill {row['infill_fraction']:.0%}  {wall:.0f} s",
                      flush=True)
    daily = (pd.concat(daily_rows, ignore_index=True) if daily_rows
             else pd.DataFrame())
    return pd.DataFrame(rows), daily


def run_sweep(
    sites: Optional[list] = None,
    horizons=DEFAULT_HORIZONS,
    coarse_resolutions=(DEFAULT_COARSE,),
    battery_kwh: Optional[tuple] = None,
    csv_path: Optional[Path] = DEFAULT_CSV,
    daily_csv_path: Optional[Path] = DEFAULT_DAILY_CSV,
    c_deg_per_kwh_override=None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Sweep every site. Fail-soft per site."""
    from energy_masters_pipeline.scripts.run_cohort_attribution import accepted_sites
    if sites is None:
        sites = accepted_sites()

    n_runs = len(sites) * len(horizons) * len(coarse_resolutions) * len(battery_kwh or (None,))
    print(f"Horizon sweep: {len(sites)} site(s) x {len(horizons)} horizon(s)"
          + (f" x {len(battery_kwh)} battery size(s)" if battery_kwh else "")
          + (f" x {len(coarse_resolutions)} resolution(s)" if len(coarse_resolutions) > 1 else "")
          + f" = {n_runs} rolling run(s)")
    print(f"Sites: {', '.join(sites)}\n")

    frames, daily_frames, failures = [], [], []
    t_start = time.perf_counter()
    for i, site_id in enumerate(sites, 1):
        print(f"[site {i}/{len(sites)}] {site_id}: loading ...", flush=True)
        try:
            inputs = load_em_site(site_id, c_deg_per_kwh_override=c_deg_per_kwh_override)
        except Exception as e:                              # noqa: BLE001
            failures.append((site_id, e))
            print(f"  SITE FAILED: {type(e).__name__}: {e}")
            continue
        print(f"  headroom ${inputs.headroom:,.2f}, "
              f"actual VCR {(inputs.sc_cost - inputs.actual_cost) / inputs.headroom:.3f}"
              if abs(inputs.headroom) > 1e-9 else "  headroom ~0", flush=True)
        f, d = sweep_site(inputs, horizons, coarse_resolutions,
                          battery_kwh, show_progress)
        frames.append(f)
        if not d.empty:
            daily_frames.append(d)

    if not frames:
        print("\nNo usable results.")
        return pd.DataFrame()

    long = pd.concat(frames, ignore_index=True)
    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        long.to_csv(csv_path, index=False)
        print(f"\nPer-run detail written to {Path(csv_path).resolve()}")
    _write_daily(daily_frames, daily_csv_path)

    print(f"\nTotal wall time {(time.perf_counter() - t_start) / 60:.1f} min")
    if failures:
        print(f"\n{len(failures)} site(s) failed:")
        for site_id, e in failures:
            print(f"  {site_id}: {type(e).__name__}: {e}")

    print_cohort_table(long)
    return long


def run_edp_sweep(
    sites: list,
    years,
    state: str,
    network_tariff_id: str,
    retail_tariff_id: str,
    battery_kwh=None,
    months: Optional[tuple] = None,
    horizons=DEFAULT_HORIZONS,
    coarse_resolutions=(DEFAULT_COARSE,),
    export_limit_kw: float = 10.0,
    c_deg_per_kwh: float = 0.039,
    csv_path: Optional[Path] = None,
    daily_csv_path: Optional[Path] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Sweep EDP sites. Same machinery, different loader.

    c_deg defaults to the Energy Masters fitted value rather than the model
    default of 0.005, so that an EDP horizon result is comparable with the EM
    one. A different degradation cost changes how much cycling is worthwhile
    and therefore how much horizon is worth having.

    `years` accepts an int or a sequence. Each (site, year) becomes its own
    cohort member, since load_edp_site already stamps the year into site_id.
    Price volatility differs more between years than load shape does between
    sites, so a second year is usually worth more than five more sites.
    """
    from model.core.scenario import BatterySpec

    if isinstance(years, int):
        years = (years,)
    pairs = [(s, y) for y in years for s in sites]

    frames, daily_frames, failures = [], [], []
    t_start = time.perf_counter()
    label = (f"{state} {','.join(str(y) for y in years)}"
             + (f" months {months}" if months else ""))
    print(f"EDP horizon sweep: {len(sites)} site(s) x {len(years)} year(s), "
          f"{label}, {len(horizons)} horizon(s)"
          + (f" x {len(battery_kwh)} battery size(s)" if battery_kwh else ""))
    for i, (site_id, year) in enumerate(pairs, 1):
        print(f"[site {i}/{len(pairs)}] {site_id} {year}: loading ...",
              flush=True)
        try:
            base = BatterySpec(
                capacity_kwh=float(battery_kwh[0]) if battery_kwh else 13.5,
                backup_reserve_pct=0.0,
                power_kw=min(float(battery_kwh[0]) * POWER_KW_RATIO, POWER_KW_CAP)
                if battery_kwh else 5.0,
                charge_efficiency=0.95, discharge_efficiency=0.95,
                c_deg_per_kwh=c_deg_per_kwh)
            inputs = load_edp_site(site_id, year, state, network_tariff_id,
                                   retail_tariff_id, base,
                                   export_limit_kw=export_limit_kw,
                                   months=months)
        except Exception as e:                              # noqa: BLE001
            failures.append((site_id, e))
            print(f"  SITE FAILED: {type(e).__name__}: {e}")
            continue
        print(f"  {len(inputs.index):,} intervals, headroom "
              f"${inputs.headroom:,.2f}", flush=True)
        f, d = sweep_site(inputs, horizons, coarse_resolutions,
                          battery_kwh, show_progress)
        frames.append(f)
        if not d.empty:
            daily_frames.append(d)
        # Written after every site rather than only at the end. A full-year
        # multi-size sweep is an overnight run; losing all of it to a failure
        # on the last site would be avoidable waste.
        if csv_path is not None:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
        _write_daily(daily_frames, daily_csv_path, quiet=True)

    if not frames:
        print("\nNo usable results.")
        return pd.DataFrame()
    long = pd.concat(frames, ignore_index=True)
    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        long.to_csv(csv_path, index=False)
        print(f"\nPer-run detail written to {Path(csv_path).resolve()}")
    _write_daily(daily_frames, daily_csv_path)
    print(f"\nTotal wall time {(time.perf_counter() - t_start) / 60:.1f} min")
    for site_id, e in failures:
        print(f"  {site_id}: {type(e).__name__}: {e}")
    print_cohort_table(long)
    return long


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def cohort_vcr(long: pd.DataFrame) -> pd.DataFrame:
    """Cohort VCR by (battery size, coarse resolution, horizon).

    Summed numerator over summed denominator, never a mean of per-site ratios.
    Sites below the headroom floor are excluded: their VCR is a ratio of two
    small numbers and including it imports noise.
    """
    thick = long[long['headroom'] >= SITE_HEADROOM_FLOOR_DOLLARS].copy()
    thick['num'] = thick['cost_sc'] - thick['cost_rolling']
    if 'battery_group' not in thick.columns:       # tolerate older CSVs
        thick['battery_group'] = 'site_native'
    g = thick.groupby(['battery_group', 'coarse_resolution', 'horizon_hours'],
                      dropna=False)
    out = g.agg(cohort_vcr=('num', 'sum'), denom=('headroom', 'sum'),
                n_sites=('site_id', 'nunique'),
                infill=('infill_fraction', 'mean'),
                wall_s=('wall_seconds', 'sum')).reset_index()
    out['cohort_vcr'] = out['cohort_vcr'] / out['denom']
    return out.drop(columns='denom')


def _write_daily(daily_frames, path, quiet: bool = False) -> None:
    if path is None or not daily_frames:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(daily_frames, ignore_index=True).to_csv(path, index=False)
    if not quiet:
        print(f"Per-day detail written to {Path(path).resolve()}")


def loss_at_horizon(long: pd.DataFrame, horizon: float = 18.0) -> pd.DataFrame:
    """What holding the horizon at `horizon` costs, by battery size.

    recommend_horizon answers "what is the shortest horizon meeting the rule".
    This answers the inverse and is the one that decides compute: the run cost
    rises roughly linearly with horizon, so a loss of 0.001 VCR at 18 h is a
    reason to stay there even when the rule would recommend 24.

    Columns: the best cohort VCR at any horizon tested, the cohort VCR at
    `horizon`, the shortfall, and the largest shortfall borne by any single
    site above the headroom floor.
    """
    coh = cohort_vcr(long)
    thick = long[long['headroom'] >= SITE_HEADROOM_FLOOR_DOLLARS]
    if 'battery_group' not in thick.columns:
        thick = thick.assign(battery_group='site_native')

    out = []
    for (size, coarse), grp in coh.groupby(['battery_group', 'coarse_resolution']):
        at_h = grp[grp['horizon_hours'] == horizon]
        if at_h.empty:
            continue
        sel = ((thick['battery_group'] == size)
               & (thick['coarse_resolution'] == coarse))
        site_best = thick[sel].groupby('site_id')['vcr'].max()
        site_at_h = (thick[sel & (thick['horizon_hours'] == horizon)]
                     .set_index('site_id')['vcr'])
        site_loss = (site_best - site_at_h).dropna()
        best = float(grp['cohort_vcr'].max())
        here = float(at_h['cohort_vcr'].iloc[0])
        out.append(dict(
            battery_group=size, coarse_resolution=coarse,
            best_horizon_hours=float(
                grp.loc[grp['cohort_vcr'].idxmax(), 'horizon_hours']),
            best_cohort_vcr=best,
            cohort_vcr_at_h=here,
            cohort_loss=best - here,
            worst_site_loss=float(site_loss.max()) if len(site_loss) else np.nan,
            worst_site=(site_loss.idxmax() if len(site_loss) else ''),
        ))
    return pd.DataFrame(out)


def recommend_horizon(long: pd.DataFrame) -> dict:
    """Apply the pre-registered decision rule.

    Returns the shortest horizon meeting both the cohort and per-site
    tolerances, per battery size and coarse resolution.
    """
    coh = cohort_vcr(long)
    thick = long[long['headroom'] >= SITE_HEADROOM_FLOOR_DOLLARS]
    out = {}
    if 'battery_group' not in thick.columns:
        thick = thick.assign(battery_group='site_native')
    for (size, coarse), grp in coh.groupby(['battery_group', 'coarse_resolution']):
        best = grp['cohort_vcr'].max()
        site_best = (thick[(thick['battery_group'] == size)
                           & (thick['coarse_resolution'] == coarse)]
                     .groupby('site_id')['vcr'].max())
        chosen = None
        for hh in sorted(grp['horizon_hours']):
            row = grp[grp['horizon_hours'] == hh].iloc[0]
            if row['cohort_vcr'] < best - COHORT_TOLERANCE:
                continue
            at_h = thick[(thick['battery_group'] == size)
                         & (thick['coarse_resolution'] == coarse)
                         & (thick['horizon_hours'] == hh)
                         ].set_index('site_id')['vcr']
            if ((site_best - at_h).dropna() > SITE_TOLERANCE).any():
                continue
            chosen = hh
            break
        out[(size, coarse)] = chosen
    return out


def print_cohort_table(long: pd.DataFrame, hold_horizon: float = 18.0) -> None:
    coh = cohort_vcr(long)
    print("\n=== Cohort VCR by horizon ===")
    print("Summed numerator / summed denominator; sites below "
          f"${SITE_HEADROOM_FLOOR_DOLLARS:.0f} headroom excluded.")
    print(coh.to_string(index=False,
                        float_format=lambda v: f"{v:,.4f}"))

    print(f"\n=== Decision rule (cohort within {COHORT_TOLERANCE}, "
          f"no site losing more than {SITE_TOLERANCE}) ===")
    for (size, coarse), hh in sorted(recommend_horizon(long).items()):
        if hh is None:
            print(f"  {size}, {coarse}: no horizon tested meets the rule")
        else:
            note = ("  <- fits inside pre-dispatch's 15.08 h guaranteed "
                    "coverage, so infill never fires" if hh <= 15 else
                    "  <- ABOVE the 15.08 h guarantee, so infill fires")
            print(f"  {size}, {coarse}: {hh:g} h{note}")

    loss = loss_at_horizon(long, hold_horizon)
    if not loss.empty:
        print(f"\n=== Cost of holding the horizon at {hold_horizon:g} h ===")
        print("Compute rises roughly linearly with horizon, so a small loss "
              "here is worth taking.")
        print(loss.to_string(index=False,
                             float_format=lambda v: f"{v:,.4f}"))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sites', type=str, default=None,
                        help='Comma-separated. Default: all accepted sites.')
    parser.add_argument('--horizons', type=str,
                        default=','.join(f'{h:g}' for h in DEFAULT_HORIZONS))
    parser.add_argument('--coarse-resolutions', type=str, default=DEFAULT_COARSE,
                        help="Comma-separated subset of 5min,30min. '30min' is "
                             "what PREDISPATCH publishes; add '5min' to "
                             "separate resolution loss from horizon effects.")
    parser.add_argument('--battery-kwh', type=str, default=None,
                        help='Comma-separated capacities, e.g. 5,10,20,35,50. '
                             'Power is set to capacity/2. Omit to use each '
                             "site's own battery.")
    parser.add_argument('--c-deg-per-kwh', type=float, default=None)
    parser.add_argument('--csv', type=Path, default=None,
                        help='Per-run CSV. Defaults to the Energy Masters or '
                             'EDP path depending on --edp.')
    parser.add_argument('--daily-csv', type=Path, default=None,
                        help='Per-run-per-day CSV, for slicing a full-year '
                             'sweep by season without re-running it.')
    parser.add_argument('--hold-horizon', type=float, default=18.0,
                        help='Horizon whose cost is reported in the final '
                             'table (default 18).')
    parser.add_argument('--no-progress', action='store_true')
    parser.add_argument('--edp', action='store_true',
                        help='Sweep EDP sites instead of Energy Masters sites. '
                             'Requires --year, --state and both tariff ids.')
    parser.add_argument('--year', type=str, default=None,
                        help='Calendar year, or comma-separated years.')
    parser.add_argument('--state', type=str, default=None)
    parser.add_argument('--network-tariff', type=str, default=None)
    parser.add_argument('--retail-tariff', type=str, default=None)
    parser.add_argument('--months', type=str, default=None,
                        help='Comma-separated month numbers, e.g. 6,7,8 for '
                             'winter. EDP only.')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    args = parser.parse_args()

    if args.edp:
        missing = [f for f, v in (('--year', args.year), ('--state', args.state),
                                  ('--network-tariff', args.network_tariff),
                                  ('--retail-tariff', args.retail_tariff))
                   if v is None]
        if missing or not args.sites:
            parser.error("--edp requires --sites and " + ', '.join(missing or []))
        run_edp_sweep(
            sites=[s.strip() for s in args.sites.split(',')],
            years=tuple(int(y) for y in str(args.year).split(',')),
            state=args.state,
            network_tariff_id=args.network_tariff,
            retail_tariff_id=args.retail_tariff,
            battery_kwh=(tuple(float(b) for b in args.battery_kwh.split(','))
                         if args.battery_kwh else None),
            months=(tuple(int(m) for m in args.months.split(','))
                    if args.months else None),
            horizons=tuple(float(h) for h in args.horizons.split(',')),
            coarse_resolutions=tuple(c.strip() for c in args.coarse_resolutions.split(',')),
            export_limit_kw=args.export_limit_kw,
            # Previously dropped on this path, so --c-deg-per-kwh silently had
            # no effect on an EDP sweep and every run used the 0.039 default.
            **({'c_deg_per_kwh': args.c_deg_per_kwh}
               if args.c_deg_per_kwh is not None else {}),
            csv_path=args.csv or DEFAULT_EDP_CSV,
            daily_csv_path=args.daily_csv or DEFAULT_EDP_DAILY_CSV,
            show_progress=not args.no_progress,
        )
        raise SystemExit(0)

    run_sweep(
        sites=[s.strip() for s in args.sites.split(',')] if args.sites else None,
        horizons=tuple(float(h) for h in args.horizons.split(',')),
        coarse_resolutions=tuple(c.strip() for c in args.coarse_resolutions.split(',')),
        battery_kwh=(tuple(float(b) for b in args.battery_kwh.split(','))
                     if args.battery_kwh else None),
        csv_path=args.csv or DEFAULT_CSV,
        daily_csv_path=args.daily_csv or DEFAULT_DAILY_CSV,
        c_deg_per_kwh_override=args.c_deg_per_kwh,
        show_progress=not args.no_progress,
    )