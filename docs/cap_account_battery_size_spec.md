# Alternative offer outcomes by battery size — Specification

## Scope

A battery-size sweep of the alternative offer analysis (`docs/cap_account_spec.md`, `analysis/scripts/cap_account_analysis.py`, `cap_account_retailer.py`, `cap_account_aggregate.py`) for Results, Alternative Coordination Offering Outcomes (`thesis/04_results.md`).  It shows how the household and retailer outcomes of the designed offers and Tesla Electric change from 5 to 50 kWh, the 20 kWh results already in the thesis being one point on each curve.

## Variants

All at a $300/MWh cap, on the coordinated Amber cohort (`full_run_rolling_amber`, achievable imperfect foresight), at each battery size 5, 10, …, 50 kWh, for every state and year 2021–2025:

| Variant | Settings |
|---|---|
| Cohort Hedging, monthly | `--offer cohort --settlement-period monthly --surcharge-rate-per-kwh 0.01` |
| Cohort Hedging, daily | `--offer cohort --settlement-period daily --surcharge-rate-per-kwh 0.01` |
| Retailer Pool Retention, monthly | `--offer retailer --settlement-period monthly --share-fraction 0.9` |
| Retailer Pool Retention, daily | `--offer retailer --settlement-period daily --share-fraction 0.9` |
| Tesla Electric | `--offer dynamic` (2026 trial rates, `{state}_tesla_electric_mixed_2026`) |

The surcharge (1 c/kWh) and α (0.9) are the same at every size.  Cohort Hedging keeps Amber's $25 subscription, so its mean gross margin is free to move away from $25 with battery size.  Retailer Pool Retention's subscription is set at each size so that its mean gross margin is $25 (below).

Tesla Electric is billed on its own modelled run, `full_run_rolling_tesla`, at each size (the `cap_account_analysis.py` default, `--dynamic-dispatch own`).

## Retailer Pool Retention subscription

The equivalent subscription has a closed form (`docs/cap_account_spec.md` section 14, `subscription_change_vs_amber` in `retailer_summary.csv`), and the wholesale-layer settlement does not depend on the subscription.  With `--subscription-change auto`, `cap_account_retailer.py` sets each Retailer Pool Retention variant's subscription change to that variant's own `subscription_change_vs_amber` before the gross margin, spread and composition outputs are calculated, and writes the value used to `retailer_summary.csv`.  The other variants are unchanged.  At 20 kWh this gives −5.79 (monthly) and −5.16 (daily).

`cap_account_aggregate.py` takes the same value per size (`--subscription-change-per-month`, read from that size's `retailer_summary.csv`), so the household outcomes are at the same subscription as the retailer outcomes.

## Runs

`runners/run_cap_account_by_size.sh` loops over battery size (`SIZES`, default 5 to 50 kWh) and, for each size:

1. runs `cap_account_analysis.py` for each variant, state and year (`--battery-name {size}kWh`, `--data-root` from `DATA_ROOT`);
2. runs `cap_account_retailer.py --battery-name {size}kWh --threshold-per-mwh 300 --subscription-change auto --dynamic dynamic_tesla_electric_mixed_2026_modelled`;
3. runs `cap_account_aggregate.py` for each variant, with the Retailer Pool Retention subscription change from step 2.

Outputs are separated by battery name.  It stops on the first failure.  Each size repeats the 20 kWh analysis for the five variants (100 state-year runs), and each `cap_account_analysis.py` run reads the scenario's `flows.parquet` from the data root.

## Summary table

`analysis/scripts/offer_outcomes_by_size.py` writes `analysis/output/cap_account_retailer/by_size/offer_outcomes_by_size.csv`, a row per variant and battery size:

- Retailer, per household per month across the 240 state-months, weighted by state mix: mean gross margin and its components (subscription or net fixed charges, import margin, cost absorbed, credit retained, surcharge retained), 5th, 25th, 50th, 75th and 95th percentiles, worst month with its state and month, months in which the mechanism cost the retailer, months below zero, and the subscription change used.  From `cap_account_retailer.py`'s `retailer_cohort_months.csv`, `retailer_composition_means.csv` and `retailer_summary.csv`.
- Household, weighted by PV size band and state as in `cap_account_aggregate.py`: mean annual bill change against Amber ($), 5th and 95th percentiles of the bill change ($), its 5th, 50th and 95th percentiles as a share of coordination value, and the share of site-years better off.  From the aggregate script's `household_outcomes_weighted*.csv`.

## Charts

`analysis/scripts/alt_offers_by_size_figures.py` draws three figures from the summary table to `analysis/output/alt_offers/` and copies each to `thesis/figures/results/`.  `docs/thesis_figure_spec.md` format, height 3.0 in, offer colours as in `alt_household_bill_impact_figure.py`, no notes drawn in the figures.  Each has two panels, daily settlement (left) and monthly settlement (right), against battery size, with Tesla Electric on the monthly panel only.

- **`alt_retailer_margin_by_size`.**  Mean gross margin per household per month, one line per offer, with Retailer Pool Retention's subscription as a dotted line in the same colour.
- **`alt_retailer_spread_by_size`.**  For each offer, the 25th–75th and 5th–95th percentile bands of the monthly gross margin, the median as a line and the worst month as markers.  Vertical scale linear within ±50 and logarithmic beyond, as in `alt_retailer_spread_figure.py`.
- **`alt_household_bill_impact_by_size`.**  The 5th–95th percentile band of the annual bill change against Amber as a share of coordination value, with the median as a line.

## Values for the text

`analysis/scripts/offer_outcomes_text_values.py` prints them from the summary table.  For each designed variant: mean gross margin, worst month and months with a net cost at 5, 10, 20 and 50 kWh; the smallest battery size from which Cohort Hedging (monthly) has at most one month with a net cost; Retailer Pool Retention's subscription at 5, 10, 20 and 50 kWh; the household 5th–95th percentile range at 5, 10, 20 and 50 kWh.  For Tesla Electric: mean gross margin, months below zero and worst month at the same sizes.

## Notes

- Cohort Hedging keeps a 1 c/kWh surcharge at every size, so at small sizes its mean gross margin can fall below $25.  The summary table gives the shortfall at each size.
- Where Retailer Pool Retention's retained share is smaller than its cost absorbed, the subscription change is positive (the fee rises above $25).
