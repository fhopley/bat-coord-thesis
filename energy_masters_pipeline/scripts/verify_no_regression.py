"""Prove that commits 0 to 3 changed no dispatch number.

Commits 0 to 3 add provenance and diagnostics. Nothing in them touches the
forecast values, the LP, or the committed action, so every figure in
cohort_rungs_site.csv must be unchanged. This script is what turns "should be"
into "is".

A plain `diff` on the CSV is not usable: two columns are inserted in the middle
of the row, so every line differs textually while every value is identical.
This compares on the shared columns instead, keyed on (site_id, dispatch).

Usage (from project root):
    python3 -m energy_masters_pipeline.scripts.verify_no_regression \\
        --before path/to/cohort_rungs_site_BEFORE.csv \\
        --after  data/processed/energy_masters/cohort_rungs_site.csv

Exits 0 if nothing changed, 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ['site_id', 'dispatch']

# Columns commits 0 to 3 are ALLOWED to add. Anything else appearing or
# disappearing is a finding, not a detail.
EXPECTED_NEW = {'coarse_staleness_min_mean', 'coarse_staleness_min_max'}

# Columns that measure wall clock rather than dispatch. These CANNOT be
# expected to reproduce: they depend on machine load, thermal state and what
# else is running, and they vary between two runs of identical code. Comparing
# them turns every run into a failure and trains the reader to ignore the
# verdict, which is worse than not checking at all.
#
# They are still reported, because a large one-directional shift is worth
# seeing even though it is not a regression. The distinguishing test is
# whether the ratio is consistently above 1 (a real cost) or scattered either
# side of it (machine variance), so the report gives the spread rather than
# just the mean.
VOLATILE = {'solve_ms_median', 'solve_ms_p95', 'solve_seconds_total',
            'wall_seconds'}

# Tolerance for float comparison. Deliberately tight: these runs are
# deterministic given the same inputs, so a difference at 1e-9 is a real
# difference and not solver noise.
ATOL = 1e-9
RTOL = 1e-12


def compare(before: Path, after: Path, atol: float = ATOL,
            rtol: float = RTOL) -> tuple[bool, list[str]]:
    """Compare two cohort CSVs on their shared columns. Returns (ok, report)."""
    b = pd.read_csv(before).set_index(KEY).sort_index()
    a = pd.read_csv(after).set_index(KEY).sort_index()
    out: list[str] = []
    ok = True

    # --- shape -----------------------------------------------------------
    missing_rows = b.index.difference(a.index)
    added_rows = a.index.difference(b.index)
    if len(missing_rows):
        ok = False
        out.append(f"ROWS LOST ({len(missing_rows)}): "
                   f"{[tuple(x) for x in missing_rows[:10]]}")
    if len(added_rows):
        ok = False
        out.append(f"ROWS ADDED ({len(added_rows)}): "
                   f"{[tuple(x) for x in added_rows[:10]]}")
    out.append(f"rows: {len(b)} before, {len(a)} after, "
               f"{len(b.index.intersection(a.index))} shared")

    # --- columns ---------------------------------------------------------
    new_cols = set(a.columns) - set(b.columns)
    lost_cols = set(b.columns) - set(a.columns)
    unexpected = new_cols - EXPECTED_NEW
    absent = EXPECTED_NEW - new_cols
    if lost_cols:
        ok = False
        out.append(f"COLUMNS LOST: {sorted(lost_cols)}")
    if unexpected:
        ok = False
        out.append(f"UNEXPECTED NEW COLUMNS: {sorted(unexpected)}")
    if absent:
        ok = False
        out.append(f"EXPECTED NEW COLUMNS MISSING: {sorted(absent)}")
    out.append(f"new columns: {sorted(new_cols) or 'none'}")

    # --- values ----------------------------------------------------------
    shared_idx = b.index.intersection(a.index)
    shared_cols = [c for c in b.columns if c in a.columns]
    compared = [c for c in shared_cols if c not in VOLATILE]
    skipped = [c for c in shared_cols if c in VOLATILE]
    for col in compared:
        bs, as_ = b.loc[shared_idx, col], a.loc[shared_idx, col]
        if pd.api.types.is_numeric_dtype(bs) and pd.api.types.is_numeric_dtype(as_):
            bv, av = bs.to_numpy(float), as_.to_numpy(float)
            same = np.isclose(bv, av, atol=atol, rtol=rtol,
                              equal_nan=True)
        else:
            same = (bs.fillna('').astype(str).to_numpy()
                    == as_.fillna('').astype(str).to_numpy())
        if same.all():
            continue
        ok = False
        n = int((~same).sum())
        worst = ''
        if pd.api.types.is_numeric_dtype(bs):
            d = np.abs(bv - av)
            d[np.isnan(d)] = 0.0
            j = int(np.argmax(d))
            worst = (f"; worst {tuple(shared_idx[j])} "
                     f"{bv[j]!r} -> {av[j]!r} (delta {d[j]:.3e})")
        out.append(f"CHANGED: {col} in {n} of {len(shared_idx)} rows{worst}")

    if ok:
        out.append(f"all {len(compared)} dispatch columns identical across "
                   f"{len(shared_idx)} rows")

    # --- volatile columns, reported but not judged -----------------------
    for col in skipped:
        bv = pd.to_numeric(b.loc[shared_idx, col], errors='coerce').to_numpy(float)
        av = pd.to_numeric(a.loc[shared_idx, col], errors='coerce').to_numpy(float)
        m = np.isfinite(bv) & np.isfinite(av) & (bv != 0)
        if not m.any():
            continue
        r = av[m] / bv[m]
        verdict = ('consistently slower, worth a look'
                   if (r > 1).all() else 'scattered either side of 1, machine variance')
        out.append(f"timing (not compared): {col} ratio after/before "
                   f"min {r.min():.3f}, median {np.median(r):.3f}, "
                   f"max {r.max():.3f} over {m.sum()} rows -- {verdict}")

    # --- sanity on the new columns themselves ----------------------------
    # Not a regression check: a positive assertion that the thing we added
    # actually measures something. A coarse staleness of exactly zero
    # everywhere would mean the new column is as blind as the old one.
    if 'coarse_staleness_min_max' in a.columns:
        s = pd.to_numeric(a['coarse_staleness_min_max'], errors='coerce').dropna()
        if len(s):
            out.append(f"coarse staleness max: min {s.min():.1f}, "
                       f"max {s.max():.1f} min across {len(s)} rolling rows")
            if s.max() <= 0:
                ok = False
                out.append("COARSE STALENESS IS ZERO EVERYWHERE: the new "
                           "column is not measuring the coarse run")
            if s.max() > 30.0 + 1e-9:
                ok = False
                out.append("COARSE STALENESS EXCEEDS ONE CADENCE (30 min): "
                           "expected 0 to 25 min on a healthy store")
    if 'staleness_min_max' in a.columns:
        s = pd.to_numeric(a['staleness_min_max'], errors='coerce').dropna()
        if len(s):
            out.append(f"fine staleness max:   min {s.min():.1f}, "
                       f"max {s.max():.1f} min (0.0 expected on a full store)")
    return ok, out


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    args = parser.parse_args(argv)

    ok, report = compare(args.before, args.after)
    print(f"=== cohort regression check ===")
    print(f"before {args.before}")
    print(f"after  {args.after}\n")
    for line in report:
        print(f"  {line}")
    print(f"\nOVERALL: {'PASS -- no dispatch number changed' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
