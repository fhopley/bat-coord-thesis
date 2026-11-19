"""Targeted diagnostic: trace the backward window search for a specific sink.

After running perfect_foresight_custom on a date-filtered trace, this script
replays the greedy state AT THE POINT where a nominated sink interval is first
processed, then calls the window and source machinery directly and logs every
candidate interval in the backward window with its full decision data.

This answers definitively:
  1. Is the target source interval (e.g. 16:40 Jun 10) inside the backward window?
  2. If yes — what is its head_charge and source cost, and does it pass the hurdle?
  3. If no  — which earlier interval is the full-battery barrier that terminates the window?

Usage (from project root):
    python3 -m model.dev.diagnose_shift_window S0088 2022 \\
        --date-from 2022-06-09 --date-to 2022-06-11 \\
        --sink-time 2022-06-10T22:30 --sink-kind EXPORT \\
        --source-time 2022-06-10T16:40

All times are interpreted in the trace timezone (AEST / Etc/GMT-10).
--sink-time   : the interval you want to examine as a sink (nearest 5-min boundary)
--sink-kind   : EXPORT or AVOID
--source-time : a specific interval you want to check explicitly (e.g. 16:40)
                Shown with a *** marker in the window table.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from model.core.paths import default_paths
from model.core.prices import build_price_vectors
from model.components.self_consumption import self_consumption
from model.components.perfect_foresight_custom import (
    _State, _baseline_curtail, _evaluate_sink, _find_best_source,
    _apply_shift, _backward_window_start, _forward_window_end,
    _source_arrays, _AVOID, _EXPORT, _EPS, _SIMUL_TOL,
)
from model.dev._dev_common import DEV_BATTERY
from model.dev.viz_perfect_foresight_custom import load_inputs, filter_date_range


_C = 100.0  # $/kWh → c/kWh for display


def _build_state_from_scratch(solar, load, p_imp, p_exp, spec, export_kw, IH):
    """Build a fresh _State from self_consumption baseline + curtailment."""
    soc_min = spec.backup_reserve_pct / 100.0 * spec.capacity_kwh
    sc = self_consumption('S', pd.Series(solar), pd.Series(load), spec, IH)
    st = _State(
        solar=solar.copy(), load=load.copy(),
        p_imp=p_imp.copy(), p_exp=p_exp.copy(),
        sol_to_home=sc.sol_to_home_kwh.to_numpy().copy(),
        sol_to_bat=sc.sol_to_bat_kwh.to_numpy().copy(),
        sol_to_grid=sc.sol_to_grid_kwh.to_numpy().copy(),
        bat_to_home=sc.bat_to_home_kwh.to_numpy().copy(),
        grid_to_home=sc.grid_to_home_kwh.to_numpy().copy(),
        grid_to_bat=sc.grid_to_bat_kwh.to_numpy().copy(),
        bat_to_grid=sc.bat_to_grid_kwh.to_numpy().copy(),
        soc=sc.battery_soc_kwh.to_numpy().copy(),
        curtail=sc.curtail_kwh.to_numpy().copy(),
        cap=spec.capacity_kwh, soc_min=soc_min,
        eta_c=spec.charge_efficiency, eta_d=spec.discharge_efficiency,
        power=spec.power_kw * IH, export_cap=export_kw * IH,
    )
    _baseline_curtail(st)
    return st, soc_min, float(sc.battery_soc_kwh.iloc[-1])


def _run_greedy_until(st, threshold, stop_j, stop_kind, opt_factor=1.0):
    """Run the greedy loop, stopping just BEFORE the nominated sink is first
    processed. Returns the number of shifts applied and the opportunity-list
    position where the sink appears.

    Stops at the first occurrence of stop_j / stop_kind in the sorted list
    (before calling _service_sink on it), so the state represents exactly
    what the algorithm sees when it first examines that sink.
    """
    T = st.soc.shape[0]
    eta_c, eta_d = st.eta_c, st.eta_d
    seed_values = np.concatenate([st.p_imp, st.p_exp])
    seed_index  = np.concatenate([np.arange(T), np.arange(T)]).astype(np.intp)
    seed_kind   = np.concatenate(
        [np.full(T, _AVOID, dtype=np.intp), np.full(T, _EXPORT, dtype=np.intp)]
    )
    order = np.argsort(-seed_values, kind='stable')
    limit = int(np.floor(len(order) * opt_factor))

    n_shifts = 0
    for pos in range(limit):
        oi   = order[pos]
        j    = int(seed_index[oi])
        kind = int(seed_kind[oi])
        if j == stop_j and kind == stop_kind:
            return n_shifts, pos   # stop BEFORE processing this sink

        while True:
            sv, sa = _evaluate_sink(st, j, kind)
            if sa <= _EPS:
                break
            is_r = kind == _EXPORT and st.sol_to_bat[j] > _SIMUL_TOL
            hr   = (sv - threshold) if is_r else (sv * eta_c * eta_d - threshold)
            sw   = sa * eta_c if is_r else sa / eta_d
            src  = _find_best_source(st, j, hr)
            if src is None:
                break
            k, _, sdc = src
            sw = min(sw, sdc)
            if sw <= _EPS:
                break
            _apply_shift(st, k, j, kind, is_r, sw)
            n_shifts += 1

    return n_shifts, limit   # sink not found in opportunity list


def _inspect_backward_window(st, j, hurdle, mark_k, index):
    """Print the full backward window for sink j with per-interval detail."""
    ws   = _backward_window_start(st, j)
    k_range = np.arange(ws, j, dtype=np.intp)
    if len(k_range) == 0:
        print("  (backward window is empty)")
        return

    diffs      = st.cap - st.soc[ws:j]
    head_charge = np.minimum.accumulate(diffs[::-1])[::-1]  # min over [k, j-1]
    cost, dc   = _source_arrays(st, k_range)
    dc_capped  = np.minimum(dc, head_charge)
    viable     = (head_charge > _EPS) & (cost < hurdle) & (dc_capped > _EPS)

    # Find the best source (matches _find_best_source logic)
    rank = np.where(viable, cost + 1e-12 * (j - k_range), np.inf)
    best_i = int(np.argmin(rank)) if viable.any() else -1

    def hstr(i):
        ts = index[i]
        return f"{ts.strftime('%b %d %H:%M')}"

    def stype(k):
        if st.curtail[k] > _EPS:    return 'curtail'
        if st.sol_to_grid[k] > _EPS: return 'solar'
        return 'grid'

    print(f"  Backward window: [{hstr(ws)} .. {hstr(j-1)}]  "
          f"({len(k_range)} intervals)")
    print(f"  Hurdle: {hurdle:.4f} $/kWh  (source must be cheaper)")
    print()
    print(f"  {'Time':>15} {'type':>8} {'cost c/kWh':>11} "
          f"{'head_dc':>9} {'viable':>7}  note")
    print("  " + "-" * 68)

    # Print every interval but condense long runs of identical rows
    prev_type = prev_cost = prev_viable = None
    run = 0

    def flush_run(run, prev_time, prev_type, prev_cost, prev_viable):
        if run > 1:
            print(f"  {'... ×' + str(run):>15}")

    for ii, k in enumerate(k_range):
        kt   = stype(k)
        kc   = float(cost[ii])
        khc  = float(head_charge[ii])
        kdc  = float(dc_capped[ii])
        kv   = bool(viable[ii])
        is_mark = (k == mark_k)
        is_best = (ii == best_i)

        note = ''
        if is_best:   note += ' ← BEST SOURCE CHOSEN'
        if is_mark:   note += ' *** TARGET SOURCE ***'
        if khc < _EPS: note += ' [head_charge≈0: full-battery barrier]'

        # Only print if interesting: mark, best, viable, near barrier edge
        prev_hc = float(head_charge[ii-1]) if ii > 0 else None
        barrier_edge = (prev_hc is not None and (
            (prev_hc < _EPS) != (khc < _EPS)
        ))

        if is_mark or is_best or kv or barrier_edge or ii == 0 or ii == len(k_range)-1:
            ts_str = hstr(k)
            print(f"  {ts_str:>15} {kt:>8} {kc*_C:>10.2f}c "
                  f"{khc:>9.4f} {'YES' if kv else 'no':>7}{note}")
        elif khc < _EPS and (prev_hc is None or prev_hc >= _EPS):
            # Print the first interval where barrier hits
            ts_str = hstr(k)
            print(f"  {ts_str:>15} {kt:>8} {kc*_C:>10.2f}c "
                  f"{khc:>9.4f} {'YES' if kv else 'no':>7}"
                  f" [full-battery barrier starts here]")

    print()
    if viable.any():
        bk = int(k_range[best_i])
        print(f"  Best source: {hstr(bk)} "
              f"cost={float(cost[best_i])*_C:.2f}c  "
              f"head_dc={float(dc_capped[best_i]):.4f}")
    else:
        print("  No viable source found in backward window.")

    if mark_k >= ws and mark_k < j:
        mi = mark_k - ws
        print(f"\n  Target interval {hstr(mark_k)}:")
        print(f"    type={stype(mark_k)}  cost={float(cost[mi])*_C:.2f}c  "
              f"head_charge={float(head_charge[mi]):.4f}  "
              f"dc_capped={float(dc_capped[mi]):.4f}  "
              f"viable={bool(viable[mi])}")
        if not viable[mi]:
            if head_charge[mi] < _EPS:
                print("    BLOCKED: head_charge ≈ 0 (full-battery barrier "
                      "between here and the sink)")
            elif cost[mi] >= hurdle:
                print(f"    BLOCKED: cost {float(cost[mi])*_C:.2f}c >= "
                      f"hurdle {hurdle*_C:.2f}c")
            elif dc_capped[mi] < _EPS:
                print("    BLOCKED: dc headroom ≈ 0 after carry cap")
    else:
        print(f"\n  Target interval {hstr(mark_k)} is OUTSIDE the backward window.")
        if mark_k < ws:
            print(f"    It is before ws={hstr(ws)} — blocked by a full-battery "
                  f"barrier between {hstr(mark_k)} and {hstr(ws-1)}.")
            # Find the barrier: the interval in [mark_k, ws-1] where soc >= cap
            barrier_range = np.arange(mark_k, ws)
            full = np.where(st.soc[barrier_range] >= st.cap - _EPS)[0]
            if len(full):
                bi = barrier_range[full[-1]]
                print(f"    Last full-battery interval in gap: {hstr(bi)}  "
                      f"SOC={st.soc[bi]:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('site_id')
    parser.add_argument('year', type=int)
    parser.add_argument('--state',           default='NSW')
    parser.add_argument('--network-tariff',  default='ausgrid_ea025_network_2024')
    parser.add_argument('--retail-tariff',   default='amber_wholesale_2024')
    parser.add_argument('--export-limit-kw', type=float, default=10.0)
    parser.add_argument('--opt-factor',      type=float, default=1.0)
    parser.add_argument('--date-from',       default=None, metavar='YYYY-MM-DD')
    parser.add_argument('--date-to',         default=None, metavar='YYYY-MM-DD')
    parser.add_argument('--sink-time',       required=True, metavar='YYYY-MM-DDTHH:MM',
                        help='Sink interval to examine (nearest 5-min boundary, AEST)')
    parser.add_argument('--sink-kind',       default='EXPORT', choices=['EXPORT', 'AVOID'],
                        help='Sink type (default EXPORT)')
    parser.add_argument('--source-time',     required=True, metavar='YYYY-MM-DDTHH:MM',
                        help='Source interval to check explicitly in the window (AEST)')
    parser.add_argument('--project-root',    type=Path, default=None)
    args = parser.parse_args(argv)

    paths = default_paths(args.project_root)
    trace, prices_mwh, nt, rt = load_inputs(
        args.site_id, args.year, args.state,
        args.network_tariff, args.retail_tariff, paths,
    )
    trace, prices_mwh = filter_date_range(
        trace, prices_mwh, args.date_from, args.date_to,
    )

    spec    = DEV_BATTERY
    IH      = 5 / 60
    export  = args.export_limit_kw
    opt     = args.opt_factor
    threshold = 2.0 * spec.c_deg_per_kwh

    p_imp, p_exp = build_price_vectors(trace.index, prices_mwh, nt, rt, IH)
    solar  = trace['pv_site_net_kwh'].to_numpy()
    load_a = trace['gross_load_kwh'].to_numpy()
    p_imp_np = p_imp.to_numpy()
    p_exp_np = p_exp.to_numpy()
    index  = trace.index

    # Resolve sink and source intervals to array indices
    tz = index.tz
    sink_ts   = pd.Timestamp(args.sink_time,   tz=tz)
    source_ts = pd.Timestamp(args.source_time, tz=tz)
    # Snap to nearest 5-min boundary present in the index
    sink_j   = int(index.get_indexer([sink_ts],   method='nearest')[0])
    source_k = int(index.get_indexer([source_ts], method='nearest')[0])
    kind     = _EXPORT if args.sink_kind == 'EXPORT' else _AVOID

    print()
    print(f"=== Shift-window diagnostic: {args.site_id} "
          f"{args.date_from or args.year}–{args.date_to or ''} ===")
    print(f"Sink:   {index[sink_j]}  ({args.sink_kind})  "
          f"p={'p_exp' if kind==_EXPORT else 'p_imp'}="
          f"{(p_exp_np[sink_j] if kind==_EXPORT else p_imp_np[sink_j])*_C:.1f}c")
    print(f"Source: {index[source_k]}  "
          f"p_imp={p_imp_np[source_k]*_C:.1f}c")
    print()

    # Build fresh state and advance greedy to just before the sink is processed
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        st, soc_min, _ = _build_state_from_scratch(
            solar, load_a, p_imp_np, p_exp_np, spec, export, IH,
        )

    print("Running greedy up to (but not including) the sink...")
    n_shifts, pos = _run_greedy_until(st, threshold, sink_j, kind, opt)
    print(f"  {n_shifts} shifts applied before reaching sink "
          f"(opportunity-list position {pos}).")
    print()

    # Evaluate the sink in its current state
    sv, sa = _evaluate_sink(st, sink_j, kind)
    is_r   = kind == _EXPORT and st.sol_to_bat[sink_j] > _SIMUL_TOL
    hurdle = (sv - threshold) if is_r else (sv * spec.charge_efficiency *
                                             spec.discharge_efficiency - threshold)
    print(f"Sink evaluation at this point:")
    print(f"  sink_value={sv*_C:.2f}c  sink_amount_ac={sa:.4f} kWh  "
          f"is_redirect={is_r}")
    print(f"  hurdle = {hurdle*_C:.2f}c  (source must be cheaper than this)")
    print(f"  SOC at sink:   {st.soc[sink_j]:.4f} kWh")
    print(f"  SOC at source: {st.soc[source_k]:.4f} kWh")
    if sa <= _EPS:
        print("  Sink already exhausted — no shift to make.")
        return 0
    print()

    # Full backward window inspection
    print("Backward window (all candidate source intervals):")
    _inspect_backward_window(st, sink_j, hurdle, source_k, index)
    return 0


if __name__ == '__main__':
    sys.exit(main())
