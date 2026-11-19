"""Validation harness for perfect_foresight_custom.

Checks invariants/bounds and benchmarks the heuristic's dispatch cost against an
LP solve of the same problem (the LP gives the global optimum, so the custom
dispatch cost must be >= LP and should be close, and <= the self_consumption
baseline cost).
"""
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

sys.path.insert(0, '.')
from model.core.scenario import BatterySpec
from model.components.self_consumption import self_consumption
from model.components.perfect_foresight_custom import perfect_foresight_custom


def make_index(n, freq_min=30):
    return pd.date_range('2023-01-01', periods=n, freq=f'{freq_min}min')


def dispatch_cost(traces, p_imp, p_exp):
    """Dispatch-dependent cost (the objective the LP minimises, c_deg=0)."""
    imp = traces.import_kwh.to_numpy()
    exp = traces.export_kwh.to_numpy()
    return float(np.sum(imp * p_imp - exp * p_exp))


def feasible_baseline_cost(sc, p_imp, p_exp, export_cap):
    """Baseline dispatch cost with the same export-cap and negative-price
    curtailment the component applies, so the comparison is apples-to-apples.
    self_consumption itself is export-limit- and price-sign-agnostic."""
    imp = sc.import_kwh.to_numpy().copy()
    exp = sc.export_kwh.to_numpy().copy()
    overflow = np.maximum(0.0, exp - export_cap)
    neg = np.where(p_exp < 0.0, exp, 0.0)
    cut = np.maximum(overflow, neg)
    exp = exp - cut
    return float(np.sum(imp * p_imp - exp * p_exp))


def full_objective(traces, p_imp, p_exp, spec):
    """Dispatch cost + c_deg * throughput (charge AC + discharge AC)."""
    imp = traces.import_kwh.to_numpy()
    exp = traces.export_kwh.to_numpy()
    charge = traces.sol_to_bat_kwh.to_numpy() + traces.grid_to_bat_kwh.to_numpy()
    discharge = traces.bat_to_home_kwh.to_numpy() + traces.bat_to_grid_kwh.to_numpy()
    return float(np.sum(imp*p_imp - exp*p_exp)
                 + spec.c_deg_per_kwh * np.sum(charge + discharge))


def lp_optimum(solar, load, p_imp, p_exp, spec, soc_init, soc_final,
               export_cap, interval_hours):
    """Reference LP (matches the perfect_foresight_lp formulation, c_deg=0).
    Returns the optimal dispatch cost."""
    T = len(solar)
    OC, OD, OI, OE, OS, OQ = 0, T, 2*T, 3*T, 4*T, 5*T
    N = 6 * T
    eta_c, eta_d = spec.charge_efficiency, spec.discharge_efficiency
    power = spec.power_kw * interval_hours
    soc_min = spec.backup_reserve_pct / 100.0 * spec.capacity_kwh

    c = np.zeros(N)
    c[OC:OC+T] = spec.c_deg_per_kwh
    c[OD:OD+T] = spec.c_deg_per_kwh
    c[OI:OI+T] = p_imp
    c[OE:OE+T] = -p_exp

    t = np.arange(T)
    fb_rows = np.tile(t, 5)
    fb_cols = np.concatenate([OC+t, OD+t, OI+t, OE+t, OQ+t])
    fb_data = np.concatenate([-np.ones(T), np.ones(T), np.ones(T), -np.ones(T), -np.ones(T)])
    b_fb = load - solar

    sc_rows = np.concatenate([T+t, T+t, T+t])
    sc_cols = np.concatenate([OC+t, OD+t, OS+t])
    sc_data = np.concatenate([-np.full(T, eta_c), np.full(T, 1.0/eta_d), np.ones(T)])
    if T > 1:
        t1 = np.arange(1, T)
        sc_rows = np.concatenate([sc_rows, T+t1])
        sc_cols = np.concatenate([sc_cols, OS+t1-1])
        sc_data = np.concatenate([sc_data, -np.ones(T-1)])
    b_sc = np.zeros(T); b_sc[0] = soc_init

    A = sp.csc_matrix((np.concatenate([fb_data, sc_data]),
                       (np.concatenate([fb_rows, sc_rows]),
                        np.concatenate([fb_cols, sc_cols]))), shape=(2*T, N))
    b = np.concatenate([b_fb, b_sc])

    lo = np.zeros(N); hi = np.full(N, np.inf)
    hi[OC:OC+T] = power; hi[OD:OD+T] = power
    hi[OE:OE+T] = export_cap
    lo[OS:OS+T] = soc_min; hi[OS:OS+T] = spec.capacity_kwh
    lo[OS+T-1] = soc_final; hi[OS+T-1] = soc_final
    hi[OQ:OQ+T] = np.maximum(0.0, solar)

    res = linprog(c, A_eq=A, b_eq=b, bounds=np.column_stack([lo, hi]), method='highs')
    assert res.status == 0, f"LP failed: {res.message}"
    x = res.x
    dispatch = float(np.sum(x[OI:OI+T]*p_imp - x[OE:OE+T]*p_exp))
    deg = float(spec.c_deg_per_kwh * np.sum(x[OC:OC+T] + x[OD:OD+T]))
    return dispatch + deg


def baseline_soc_final(solar, load, spec, interval_hours, index):
    sc = self_consumption('S', pd.Series(solar, index=index),
                          pd.Series(load, index=index), spec, interval_hours)
    return float(sc.battery_soc_kwh.to_numpy()[-1]), sc


def run_case(name, solar, load, p_imp, p_exp, spec, export_cap=np.inf,
             freq_min=30, check_lp=True):
    n = len(solar)
    ih = freq_min / 60.0
    index = make_index(n, freq_min)
    soc_min = spec.backup_reserve_pct / 100.0 * spec.capacity_kwh
    soc_final, sc = baseline_soc_final(solar, load, spec, ih, index)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        out = perfect_foresight_custom(
            'S', pd.Series(solar, index=index), pd.Series(load, index=index),
            pd.Series(p_imp, index=index), pd.Series(p_exp, index=index),
            spec, soc_init_kwh=soc_min, soc_final_kwh=soc_final,
            export_limit_kwh_per_interval=export_cap, interval_hours=ih,
        )
    warn_msgs = [str(x.message) for x in w]

    # Invariant checks (hard)
    s2h = out.sol_to_home_kwh.to_numpy(); s2b = out.sol_to_bat_kwh.to_numpy()
    s2g = out.sol_to_grid_kwh.to_numpy(); cur = out.curtail_kwh.to_numpy()
    b2h = out.bat_to_home_kwh.to_numpy(); g2h = out.grid_to_home_kwh.to_numpy()
    g2b = out.grid_to_bat_kwh.to_numpy(); b2g = out.bat_to_grid_kwh.to_numpy()
    soc = out.battery_soc_kwh.to_numpy()
    imp = out.import_kwh.to_numpy(); exp = out.export_kwh.to_numpy()

    errs = []
    if np.max(np.abs(solar - (s2h + s2b + s2g + cur))) > 1e-6:
        errs.append('solar split')
    if np.max(np.abs(load - (s2h + b2h + g2h))) > 1e-6:
        errs.append('load split')
    if np.max(np.abs(imp - (g2h + g2b))) > 1e-9:
        errs.append('import identity')
    if np.max(np.abs(exp - (s2g + b2g))) > 1e-9:
        errs.append('export identity')
    # SOC continuity
    eta_c, eta_d = spec.charge_efficiency, spec.discharge_efficiency
    delta = eta_c*(s2b+g2b) - (b2h+b2g)/eta_d
    cont = soc.copy()
    cont[0] = soc[0] - (soc_min + delta[0])
    cont[1:] = soc[1:] - soc[:-1] - delta[1:]
    if np.max(np.abs(cont)) > 1e-6:
        errs.append(f'soc continuity ({np.max(np.abs(cont)):.2e})')
    if (soc < soc_min - 1e-6).any() or (soc > spec.capacity_kwh + 1e-6).any():
        errs.append('soc bounds')
    if abs(soc[-1] - soc_final) > 1e-6:
        errs.append(f'terminal soc ({soc[-1]:.6f} vs {soc_final:.6f})')
    if (g2h < -1e-6).any() or (s2g < -1e-6).any() or (b2h < -1e-6).any():
        errs.append('negative flow')

    cost_custom = full_objective(out, p_imp, p_exp, spec)
    # baseline full objective with the same curtailment applied
    base_charge = sc.sol_to_bat_kwh.to_numpy() + sc.grid_to_bat_kwh.to_numpy()
    base_discharge = sc.bat_to_home_kwh.to_numpy() + sc.bat_to_grid_kwh.to_numpy()
    cost_base = (feasible_baseline_cost(sc, p_imp, p_exp, export_cap)
                 + spec.c_deg_per_kwh * float(np.sum(base_charge + base_discharge)))
    line = f"[{name}]"
    print(f"{line}")
    print(f"    baseline cost = {cost_base:10.4f}   custom cost = {cost_custom:10.4f}")
    if check_lp:
        cost_lp = lp_optimum(solar, load, p_imp, p_exp, spec, soc_min, soc_final,
                             export_cap, ih)
        gap = cost_custom - cost_lp
        gap_pct = 100.0 * gap / (abs(cost_lp) + 1e-9)
        print(f"    LP optimum    = {cost_lp:10.4f}   gap = {gap:+.4f} ({gap_pct:+.2f}%)")
        if cost_custom < cost_lp - 1e-4:
            errs.append(f'custom beats LP optimum (impossible): {cost_custom} < {cost_lp}')
    if cost_custom > cost_base + 1e-6:
        errs.append(f'custom worse than baseline: {cost_custom} > {cost_base}')

    status = 'PASS' if not errs else 'FAIL: ' + '; '.join(errs)
    print(f"    invariants: {status}")
    if warn_msgs:
        for m in warn_msgs:
            print(f"    warn: {m}")
    print()
    return not errs


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def main():
    all_ok = True

    # Case 1: simple arbitrage, no solar. Cheap night, expensive evening.
    # 8 intervals. Battery should charge cheap, discharge expensive.
    spec = BatterySpec(capacity_kwh=10, power_kw=5, charge_efficiency=0.95,
                       discharge_efficiency=0.95, backup_reserve_pct=0.0,
                       c_deg_per_kwh=0.0)
    n = 8
    solar = np.zeros(n)
    load = np.full(n, 2.0)
    p_imp = np.array([0.10, 0.10, 0.10, 0.10, 0.50, 0.50, 0.50, 0.50])
    p_exp = p_imp - 0.05
    all_ok &= run_case('arbitrage no-solar', solar, load, p_imp, p_exp, spec, freq_min=30)

    # Case 2: solar surplus midday, expensive evening, cheap overnight.
    n = 12
    solar = np.array([0,0,0, 6,6,6, 0,0,0, 0,0,0], dtype=float)
    load = np.array([1,1,1, 1,1,1, 3,3,3, 1,1,1], dtype=float)
    p_imp = np.array([.12,.12,.12, .15,.15,.15, .55,.55,.55, .20,.20,.20])
    p_exp = np.array([.05,.05,.05, .04,.04,.04, .40,.40,.40, .08,.08,.08])
    all_ok &= run_case('solar + evening peak', solar, load, p_imp, p_exp, spec, freq_min=30)

    # Case 3: negative price midday -> baseline curtails; later peak -> wind-back
    # capture into battery should occur if capacity frees up.
    n = 12
    solar = np.array([0,0, 8,8,8,8, 0,0, 0,0, 0,0], dtype=float)
    load = np.array([1,1, 1,1,1,1, 1,1, 4,4, 1,1], dtype=float)
    p_imp = np.array([.15,.15, .02,.02,.02,.02, .20,.20, .60,.60, .20,.20])
    p_exp = np.array([.05,.05, -.10,-.10,-.10,-.10, .10,.10, .45,.45, .10,.10])
    spec2 = BatterySpec(capacity_kwh=12, power_kw=6, charge_efficiency=0.95,
                        discharge_efficiency=0.95, backup_reserve_pct=10.0,
                        c_deg_per_kwh=0.0)
    all_ok &= run_case('negative price + curtail wind-back', solar, load, p_imp, p_exp,
                       spec2, freq_min=30)

    # Case 4: export limit binding.
    n = 10
    solar = np.array([0,0, 10,10,10, 0,0, 0,0, 0], dtype=float)
    load = np.array([1,1, 1,1,1, 1,1, 5,5, 1], dtype=float)
    p_imp = np.array([.15,.15, .10,.10,.10, .25,.25, .60,.60, .20])
    p_exp = np.array([.06,.06, .05,.05,.05, .12,.12, .45,.45, .10])
    all_ok &= run_case('export limit 2kWh/int', solar, load, p_imp, p_exp, spec2,
                       export_cap=2.0, freq_min=30)

    # Case 5: flat prices -> no beneficial shift; custom == baseline.
    n = 8
    solar = np.zeros(n); load = np.full(n, 2.0)
    p_imp = np.full(n, 0.30); p_exp = np.full(n, 0.05)
    ok = run_case('flat price (no shift)', solar, load, p_imp, p_exp, spec, freq_min=30)
    all_ok &= ok

    # Case 6: degradation threshold suppresses marginal cycling.
    n = 8
    solar = np.zeros(n); load = np.full(n, 2.0)
    p_imp = np.array([.20,.20,.20,.20, .26,.26,.26,.26])  # only 6c spread
    p_exp = p_imp - 0.02
    spec_deg = BatterySpec(capacity_kwh=10, power_kw=5, charge_efficiency=0.95,
                           discharge_efficiency=0.95, backup_reserve_pct=0.0,
                           c_deg_per_kwh=0.05)  # threshold 0.10 > 6c spread => no shift
    all_ok &= run_case('c_deg suppresses cycling', solar, load, p_imp, p_exp,
                       spec_deg, freq_min=30)

    # Case 7: larger randomised 5-min day, LP-gap benchmark.
    rng = np.random.default_rng(42)
    n = 288  # one day at 5-min
    index = make_index(n, 5)
    hours = np.arange(n) * 5 / 60.0
    solar = np.maximum(0, 4*np.sin((hours-6)/12*np.pi)) * (hours > 6) * (hours < 18)
    solar += rng.normal(0, 0.05, n); solar = np.maximum(0, solar)
    load = 1.0 + 0.5*np.sin((hours-18)/24*2*np.pi) + rng.normal(0, 0.1, n)
    load = np.maximum(0.1, load)
    base_price = 0.10 + 0.30*np.clip(np.sin((hours-17)/24*2*np.pi), 0, 1)
    p_imp = base_price + rng.normal(0, 0.02, n); p_imp = np.maximum(0.02, p_imp)
    p_exp = p_imp - 0.07
    all_ok &= run_case('random 5-min day (288)', solar, load, p_imp, p_exp, spec2,
                       freq_min=5)

    print('=' * 60)
    print('ALL TESTS PASS' if all_ok else 'SOME TESTS FAILED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
