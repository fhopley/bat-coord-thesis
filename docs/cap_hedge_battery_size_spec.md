# Cap hedge position by battery size — Specification

## Scope

A battery-size sweep of the cap hedge analysis (`docs/cap_hedge_spec.md`, `analysis/scripts/cap_hedge_analysis.py`) for one thesis figure in Results, Retailer Sources of Value (`thesis/04_results.md`, Figure B).  It answers how much storage per site a coordinated cohort needs before it needs no $300/MWh cap cover across the 2021–2025 price history.

## Method

- Run the existing cap hedge analysis at each battery size 5, 10, …, 50 kWh (`--battery-kwh`), national tier, same weighting, price data and premium setting as the 20 kWh results already in the thesis (`analysis/output/cap_hedge/`).
- Cohorts: coordinated Amber (achievable imperfect foresight, `full_run_rolling_amber`) and self-consumption (`full_run_self_consumption_amber`) at each size.  No CER and Solar only do not depend on battery size and are reference values only.
- For each cohort and size collect:
  - x<sub>0</sub> (kW/site), from `hedge_summary.csv`;
  - from the state-month table (`state_months.csv`, no cap cover): the number of state-months with a net cost above $300/MWh, with net earnings, and with no intervals above $300/MWh, and the worst state-month net cost per site with its state and month.
- Write one CSV with a row per cohort and size holding these values.

## Chart

- Two panels, `docs/thesis_figure_spec.md` format, height ~3.0 in, figure title "Cap position by battery size, imperfect foresight".
- Left panel: x<sub>0</sub> (kW/site) against battery size, one line per cohort (self-consumption, Amber conservative, Amber achievable), No CER and Solar only x<sub>0</sub> as dashed reference lines, zero line emphasised.
- Right panel: number of state-months (of 240) with a net cost above $300/MWh against battery size, one line per cohort.
- Colours and names as in the other Results charts ("Self-consumption", "Amber (conservative)", "Amber (achievable)"), legend in that order.  No notes drawn in the figure.
- Output: `thesis/figures/results/retailer_cap_by_battery_size.png` and the CSV beside the cap hedge outputs.

## Values needed for the text

The smallest battery size at which the coordinated cohort's x<sub>0</sub> is negative; the smallest size from which it has no state-month with a net cost; x<sub>0</sub> and the worst state-month at 10, 20 and 50 kWh.

## Notes

- If the conservative Amber runs are available at all sizes, add them as a second coordinated line (dashed), since the threshold may be sensitive to forecast quality.
- Run time: each size repeats the 20 kWh analysis; `--charts-only` can be used once the per-size summaries exist.
