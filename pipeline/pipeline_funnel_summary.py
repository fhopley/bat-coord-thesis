#!/usr/bin/env python3
"""
pipeline_funnel_summary.py

Parses the site-selection pipeline logs (logs/pipeline_<YEAR>.log) and builds
three summary tables across states (NSW, QLD, SA, VIC) and years:

  1. Incoming sites  - raw_data count from the D_Summarise_Site_Funnel step
  2. Outgoing sites  - consistency_pass count (final clean sites) from the
                        same step
  3. Pass-through %  - outgoing / incoming * 100

The logs are UTF-16 encoded (Windows PowerShell transcripts). The relevant
data lives in the "D_Summarise_Site_Funnel.py" section of each log, e.g.:

    state  raw_data  coverage_no_battery  consistency_pass
      NSW       351                  127               105
      QLD       141                   80                67
       SA       137                   84                73
      VIC       125                   30                19
       WA        30                   19                18
      ...
    TOTAL       805                  347               289

Run from the project root:
    python pipeline_funnel_summary.py

Optional arguments:
    --logs-dir DIR     directory containing pipeline_<YEAR>.log files (default: logs)
    --years Y1 Y2 ...   years to include (default: 2021 2022 2023 2024 2025)
    --states S1 S2 ...  states to include (default: NSW QLD SA VIC)
    --out-dir DIR       directory to write CSV outputs to (default: current dir)
"""

import argparse
import re
import sys
from pathlib import Path

DEFAULT_YEARS = [2021, 2022, 2023, 2024, 2025]
DEFAULT_STATES = ["NSW", "QLD", "SA", "VIC"]

# Matches the state column of a D_Summarise_Site_Funnel table row, e.g.
# "  NSW       351                  195      10       139" -- the number of
# numeric columns between state and the end of the line is deliberately not
# fixed here. D_Summarise_Site_Funnel.py's column count has already changed
# once (coverage_no_battery -> coverage_battery_screen, plus a new
# battery_recovered column) and may change again; this only assumes the
# FIRST numeric column is raw_data and the LAST is consistency_pass,
# regardless of how many columns sit between them.
STATE_RE = re.compile(r"^[A-Z]{2,3}$")


def read_log_text(path: Path) -> str:
    """Read a pipeline log, handling the UTF-16 (with BOM) encoding used by
    the PowerShell transcripts, falling back to UTF-8 if needed."""
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-16")


def parse_funnel_table(text: str, year: int):
    """Extract per-state raw_data / consistency_pass counts from the
    D_Summarise_Site_Funnel.py section of a single year's log. Robust to
    the number of columns between raw_data and consistency_pass -- takes
    the first numeric column as raw_data and the last as consistency_pass
    by position, not by a fixed count or column name."""
    marker = "Running D_Summarise_Site_Funnel.py"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError(f"Could not find '{marker}' section for YEAR={year}")

    # The funnel table appears shortly after the marker, terminated by a
    # blank line following the TOTAL row (or by the "Saved to:" line).
    section = text[idx: idx + 3000]
    lines = section.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("state") and "raw_data" in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find funnel table header for YEAR={year}")

    results = {}
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            break
        if stripped.startswith("TOTAL"):
            break
        parts = stripped.split()
        if len(parts) < 3:
            continue
        state = parts[0]
        nums = parts[1:]
        if not STATE_RE.match(state) or not all(n.isdigit() for n in nums):
            continue
        results[state] = {
            "incoming": int(nums[0]),
            "outgoing": int(nums[-1]),
        }
    return results


def build_tables(logs_dir: Path, years, states):
    incoming = {s: {} for s in states}
    outgoing = {s: {} for s in states}

    for year in years:
        log_path = logs_dir / f"pipeline_{year}.log"
        if not log_path.exists():
            print(f"WARNING: {log_path} not found, skipping YEAR={year}", file=sys.stderr)
            continue
        text = read_log_text(log_path)
        funnel = parse_funnel_table(text, year)
        for state in states:
            if state not in funnel:
                print(f"WARNING: state {state} not found for YEAR={year}", file=sys.stderr)
                continue
            incoming[state][year] = funnel[state]["incoming"]
            outgoing[state][year] = funnel[state]["outgoing"]

    return incoming, outgoing


def format_table(title, data, years, states, fmt=lambda v: str(v),
                  col_totals=None, row_totals=None, grand_total=None):
    """Render a text table. For counts, totals are summed automatically.
    For percentages (or any derived metric), pass precomputed col_totals
    (dict year->value), row_totals (dict state->value) and grand_total,
    since summing percentages is not meaningful."""
    col_width = max(8, max((len(fmt(v)) for s in states for v in [data[s].get(y) for y in years] if v is not None), default=8) + 2)
    state_width = max(6, max(len(s) for s in states) + 2)

    lines = [title]
    header = "State".ljust(state_width) + "".join(str(y).rjust(col_width) for y in years) + "TOTAL".rjust(col_width)
    lines.append(header)
    lines.append("-" * len(header))

    computed_col_totals = {y: 0 for y in years}
    for state in states:
        row_vals = []
        row_sum = 0
        for y in years:
            v = data[state].get(y)
            row_vals.append(fmt(v) if v is not None else "n/a")
            if v is not None:
                row_sum += v
                computed_col_totals[y] += v
        row_total_val = row_totals[state] if row_totals is not None else row_sum
        row = state.ljust(state_width) + "".join(rv.rjust(col_width) for rv in row_vals) + fmt(row_total_val).rjust(col_width)
        lines.append(row)

    lines.append("-" * len(header))
    final_col_totals = col_totals if col_totals is not None else computed_col_totals
    total_row = "TOTAL".ljust(state_width) + "".join(fmt(final_col_totals[y]).rjust(col_width) for y in years)
    final_grand_total = grand_total if grand_total is not None else sum(final_col_totals.values())
    total_row += fmt(final_grand_total).rjust(col_width)
    lines.append(total_row)
    return "\n".join(lines)


def write_csv(path: Path, title, data, years, states, fmt=lambda v: v,
              col_totals=None, row_totals=None, grand_total=None):
    with open(path, "w") as f:
        f.write("state," + ",".join(str(y) for y in years) + ",total\n")
        computed_col_totals = {y: 0 for y in years}
        for state in states:
            row_vals = []
            row_sum = 0
            for y in years:
                v = data[state].get(y)
                row_vals.append("" if v is None else str(fmt(v)))
                if v is not None:
                    row_sum += v
                    computed_col_totals[y] += v
            row_total_val = row_totals[state] if row_totals is not None else row_sum
            f.write(f"{state}," + ",".join(row_vals) + f",{fmt(row_total_val)}\n")
        final_col_totals = col_totals if col_totals is not None else computed_col_totals
        final_grand_total = grand_total if grand_total is not None else sum(final_col_totals.values())
        f.write("TOTAL," + ",".join(str(fmt(final_col_totals[y])) for y in years) + f",{fmt(final_grand_total)}\n")


def main():
    parser = argparse.ArgumentParser(description="Summarise pipeline funnel results by state and year.")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing pipeline_<YEAR>.log files")
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--states", nargs="+", default=DEFAULT_STATES)
    parser.add_argument("--out-dir", default=".", help="Directory to write CSV outputs")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    incoming, outgoing = build_tables(logs_dir, args.years, args.states)

    pct = {s: {} for s in args.states}
    for s in args.states:
        for y in args.years:
            if y in incoming[s] and y in outgoing[s] and incoming[s][y] > 0:
                pct[s][y] = round(100.0 * outgoing[s][y] / incoming[s][y], 1)

    # Correct totals for the percentage table: recompute from summed counts
    # rather than summing percentages (which is not meaningful).
    pct_col_totals = {}
    for y in args.years:
        inc_sum = sum(incoming[s][y] for s in args.states if y in incoming[s])
        out_sum = sum(outgoing[s][y] for s in args.states if y in outgoing[s])
        pct_col_totals[y] = round(100.0 * out_sum / inc_sum, 1) if inc_sum else 0.0
    pct_row_totals = {}
    for s in args.states:
        inc_sum = sum(incoming[s].values())
        out_sum = sum(outgoing[s].values())
        pct_row_totals[s] = round(100.0 * out_sum / inc_sum, 1) if inc_sum else 0.0
    total_inc = sum(sum(incoming[s].values()) for s in args.states)
    total_out = sum(sum(outgoing[s].values()) for s in args.states)
    pct_grand_total = round(100.0 * total_out / total_inc, 1) if total_inc else 0.0

    print()
    print(format_table("TABLE 1: Incoming sites (raw_data) by state and year", incoming, args.years, args.states))
    print()
    print(format_table("TABLE 2: Outgoing sites (consistency_pass, final clean) by state and year", outgoing, args.years, args.states))
    print()
    print(format_table(
        "TABLE 3: Pass-through percentage (outgoing / incoming) by state and year",
        pct, args.years, args.states,
        fmt=lambda v: f"{v:.1f}%" if v is not None else "n/a",
        col_totals=pct_col_totals, row_totals=pct_row_totals, grand_total=pct_grand_total,
    ))
    print()

    write_csv(out_dir / "table1_incoming_sites.csv", "incoming", incoming, args.years, args.states)
    write_csv(out_dir / "table2_outgoing_sites.csv", "outgoing", outgoing, args.years, args.states)
    write_csv(out_dir / "table3_pass_through_pct.csv", "pct", pct, args.years, args.states,
              fmt=lambda v: f"{v:.1f}" if v is not None else "",
              col_totals=pct_col_totals, row_totals=pct_row_totals, grand_total=pct_grand_total)

    print(f"CSV files written to: {out_dir.resolve()}")
    print("  table1_incoming_sites.csv")
    print("  table2_outgoing_sites.csv")
    print("  table3_pass_through_pct.csv")


if __name__ == "__main__":
    main()