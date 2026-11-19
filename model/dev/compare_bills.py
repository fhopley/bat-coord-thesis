"""Compare per-site bills between two scenario runs (e.g. perfect_foresight_custom vs LP).

Usage (from project root):
    python3 -m model.dev.compare_bills \\
        data/output/runs/perfect_foresight_custom/scenarios/NSW_2022/bills.csv \\
        data/output/runs/perfect_foresight_lp/scenarios/NSW_2022/bills.csv \\
        [--label-a CUSTOM] [--label-b LP] [--top 20] [--csv out.csv]

Outputs:
  - Summary statistics: mean, median, p95 gap ($ and %) in household_total
  - Distribution of site-level gaps (histogram-style decile table)
  - Top-N sites by gap magnitude (sorted by household_total difference a - b)
  - Breakdown of gap by plan_type if multiple types present
  - Optional CSV export of the full joined table with gap columns added
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"site_id", "household_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} bills.csv missing columns: {missing}")
    df = df.set_index("site_id")
    return df


def _pct(a, b):
    """(a - b) / |b| * 100, safe against b == 0."""
    denom = np.where(np.abs(b) > 1e-9, np.abs(b), np.nan)
    return (a - b) / denom * 100


def _decile_table(gaps: np.ndarray) -> str:
    """Build a simple decile distribution table as a string."""
    deciles = np.percentile(gaps, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100])
    labels = ["min", "p10", "p20", "p30", "p40", "p50", "p60", "p70", "p80", "p90",
              "p95", "p99", "max"]
    rows = [f"  {l:>4s}: {v:+.4f}" for l, v in zip(labels, deciles)]
    return "\n".join(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("bills_a", type=Path, help="bills.csv for run A (shown as positive gap when worse)")
    parser.add_argument("bills_b", type=Path, help="bills.csv for run B (the reference / benchmark)")
    parser.add_argument("--label-a", default="A", help="Short label for run A (default: A)")
    parser.add_argument("--label-b", default="B", help="Short label for run B (default: B)")
    parser.add_argument("--top", type=int, default=20, help="Number of worst/best sites to show (default 20)")
    parser.add_argument("--csv", type=Path, default=None, help="Optional: write full joined table to this CSV")
    args = parser.parse_args(argv)

    la, lb = args.label_a, args.label_b

    a = _load(args.bills_a, la)
    b = _load(args.bills_b, lb)

    # Align on common sites
    common = a.index.intersection(b.index)
    only_a = a.index.difference(b.index)
    only_b = b.index.difference(a.index)

    if len(only_a) or len(only_b):
        print(f"WARNING: {len(only_a)} sites only in {la}, {len(only_b)} only in {lb}. "
              f"Analysis restricted to {len(common)} common sites.\n")

    a = a.loc[common]
    b = b.loc[common]
    n = len(common)

    ht_a = a["household_total"].to_numpy(dtype=float)
    ht_b = b["household_total"].to_numpy(dtype=float)
    gap  = ht_a - ht_b          # positive = A is more expensive
    pct  = _pct(ht_a, ht_b)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"=== Per-site bill comparison: {la} vs {lb} ===")
    print(f"Sites: {n}  (common to both runs)\n")

    print(f"--- Aggregate ---")
    print(f"  Total  {la}:  ${ht_a.sum():>12,.2f}")
    print(f"  Total  {lb}:  ${ht_b.sum():>12,.2f}")
    print(f"  Net gap ({la} - {lb}):  ${gap.sum():>+12,.2f}  "
          f"({100*gap.sum()/abs(ht_b.sum()):+.3f}% of {lb} total)")
    print()

    print(f"--- Per-site gap (household_total  {la} − {lb}, $) ---")
    print(f"  Mean:    {gap.mean():+.4f}   ({pct[~np.isnan(pct)].mean():+.2f}%)")
    print(f"  Median:  {np.median(gap):+.4f}")
    print(f"  Std:      {gap.std():.4f}")
    print()

    n_worse  = int((gap > 1e-4).sum())
    n_better = int((gap < -1e-4).sum())
    n_equal  = n - n_worse - n_better
    print(f"  Sites where {la} worse  (gap > 0): {n_worse:>4d} ({100*n_worse/n:.1f}%)")
    print(f"  Sites where {la} better (gap < 0): {n_better:>4d} ({100*n_better/n:.1f}%)")
    print(f"  Sites essentially equal:           {n_equal:>4d} ({100*n_equal/n:.1f}%)")
    print()

    print(f"--- Gap distribution ($ per site) ---")
    print(_decile_table(gap))
    print()

    # ── Top-N worst ────────────────────────────────────────────────────────────
    order = np.argsort(-gap)   # descending (worst first)
    site_ids = np.array(common)

    print(f"--- Top {args.top} sites where {la} is most expensive vs {lb} ---")
    print(f"  {'Site':>8}  {'gap ($)':>10}  {'gap (%)':>8}  "
          f"  {la} ($)  {lb} ($)  {'plan':>18}")
    print("  " + "-" * 72)
    for i in order[:args.top]:
        sid = site_ids[i]
        plan = a.loc[sid, "plan_type"] if "plan_type" in a.columns else "?"
        print(f"  {sid:>8}  {gap[i]:>+10.2f}  {pct[i]:>+7.2f}%  "
              f"  {ht_a[i]:>9.2f}  {ht_b[i]:>8.2f}  {plan:>18}")
    print()

    # ── Top-N best ─────────────────────────────────────────────────────────────
    order_best = np.argsort(gap)   # ascending (most negative = A is cheapest)
    n_show_best = min(args.top // 4, n_better, 10)
    if n_show_best > 0:
        print(f"--- Top {n_show_best} sites where {la} is cheapest vs {lb} ---")
        print(f"  {'Site':>8}  {'gap ($)':>10}  {'gap (%)':>8}  "
              f"  {la} ($)  {lb} ($)  {'plan':>18}")
        print("  " + "-" * 72)
        for i in order_best[:n_show_best]:
            sid = site_ids[i]
            plan = a.loc[sid, "plan_type"] if "plan_type" in a.columns else "?"
            print(f"  {sid:>8}  {gap[i]:>+10.2f}  {pct[i]:>+7.2f}%  "
                  f"  {ht_a[i]:>9.2f}  {ht_b[i]:>8.2f}  {plan:>18}")
        print()

    # ── Savings comparison ─────────────────────────────────────────────────────
    # If both runs have baseline_bills context we can compare VCR, but here
    # we just compare household_total directly.
    # Check for a summary.txt sibling to extract SC baseline if present.
    for run_label, bills_path in [(la, args.bills_a), (lb, args.bills_b)]:
        summary_path = bills_path.parent / "summary.txt"
        if summary_path.exists():
            txt = summary_path.read_text()
            # Pull out the saving line if present
            for line in txt.splitlines():
                if "avg saving per site" in line.lower() or "total saving" in line.lower():
                    print(f"  [{run_label} summary] {line.strip()}")
    print()

    # ── Bill component breakdown for worst sites ───────────────────────────────
    component_cols = [
        "wholesale_usage_cost", "wholesale_feedin_credit",
        "network_usage_cost", "network_feedin_credit",
        "retail_usage_cost", "retail_feedin_credit",
    ]
    shared_cols = [c for c in component_cols if c in a.columns and c in b.columns]

    if shared_cols and n_worse > 0:
        print(f"--- Bill component gap breakdown (mean across {n_worse} sites where {la} is worse) ---")
        worse_mask = gap > 1e-4
        for col in shared_cols:
            ca = a.loc[common[worse_mask], col].to_numpy(dtype=float)
            cb = b.loc[common[worse_mask], col].to_numpy(dtype=float)
            mean_diff = (ca - cb).mean()
            print(f"  {col:35s}: {mean_diff:+.4f}")
        print()

    # ── Gap percentile histogram ───────────────────────────────────────────────
    print(f"--- Gap magnitude histogram ({la} − {lb}, $) ---")
    bins = [-np.inf, -5, -2, -1, -0.1, 0.1, 1, 2, 5, 10, 20, np.inf]
    labels_hist = ["< -5", "-5 to -2", "-2 to -1", "-1 to -0.1", "≈ 0",
                   "0.1 to 1", "1 to 2", "2 to 5", "5 to 10", "10 to 20", "> 20"]
    counts, _ = np.histogram(gap, bins=bins)
    for label, count in zip(labels_hist, counts):
        bar = "█" * int(count / max(counts) * 30) if count else ""
        print(f"  {label:>12s}: {count:>4d} {bar}")
    print()

    # ── Optional CSV export ────────────────────────────────────────────────────
    if args.csv is not None:
        joined = pd.DataFrame({
            "site_id": common,
            f"household_total_{la}": ht_a,
            f"household_total_{lb}": ht_b,
            "gap_dollars": gap,
            "gap_pct": pct,
        })
        for col in shared_cols:
            joined[f"{col}_{la}"] = a.loc[common, col].to_numpy(dtype=float)
            joined[f"{col}_{lb}"] = b.loc[common, col].to_numpy(dtype=float)
            joined[f"{col}_gap"] = joined[f"{col}_{la}"] - joined[f"{col}_{lb}"]
        if "plan_type" in a.columns:
            joined["plan_type"] = a.loc[common, "plan_type"].to_numpy()
        joined = joined.sort_values("gap_dollars", ascending=False)
        joined.to_csv(args.csv, index=False)
        print(f"Full joined table written to: {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
