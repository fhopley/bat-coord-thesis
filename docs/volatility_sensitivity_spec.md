# Coordination value and price volatility — Specification

## Scope

An analysis of existing runs for Results, Sensitivity Analysis (`thesis/04_results.md`, Years and price events).  It tests how the relative value of coordination and the value captured with imperfect foresight change with price volatility, for the household bill and the wholesale saving.  The expected pattern is that in calm periods Amber's relative value and value capture both fall, and the time-of-use and VPP offers lose mainly wholesale value, so the relative value of coordination is smaller when prices are less volatile.  The result links to the Discussion on the lower volatility of 2026.

No new model runs.  A new script, `analysis/scripts/volatility_sensitivity.py`.

## Unit of analysis

The state-month (4 states × 60 months = 240), at 20 kWh.  Months vary far more in volatility than years, and the monthly bills already exist (`bills_{MM}.csv` beside each scenario's `bills.csv`).  State-months are grouped into quartiles of cap value (Q1 lowest volatility to Q4 highest), and each metric is calculated for each group.

## Inputs

- Monthly bills, `data/output/runs/{run}/scenarios/{STATE}_{YEAR}[_20kWh]/bills_{MM}.csv`, for each offer's Solar only, self-consumption reference, achievable optimised and perfect foresight optimised runs, and for the Amber Solar only run (`financial_analysis.DEFAULT_BASELINE_RUN`).  Run names come from `financial_analysis.RETAIL_PLANS` and `PLAN_RUNS`.  The self-consumption reference is the offer's self-consumption run without VPP events or critical rates, `sc_no_vpp_dir` in `financial_analysis.RETAIL_PLANS`:

  | Offer | Self-consumption reference |
  |---|---|
  | Amber | `full_run_self_consumption_amber` |
  | GloBird ZeroHero | `full_run_self_consumption_globird_novpp` |
  | AGL Battery Rewards | `full_run_self_consumption_agl_novpp` |
  | EnergyAustralia Battery Ease | `full_run_self_consumption_ea` |
  | Origin Battery Lite | `full_run_self_consumption_origin_novpp` |
  An offer's `excluded_states` in `financial_analysis.RETAIL_PLANS` are respected, so EnergyAustralia (NSW only) has NSW state-months only, and the notes on the offer charts carry the line "EnergyAustralia results for NSW only".  Check that every run has monthly bills; a run without them is listed and left out.  The sites of a state-month are those present in all of an offer's runs and the Amber Solar only run, and that have a weight.
- Site weights per state-year (`compute_site_weight_for_scenario` or the `financial_analysis.py` pipeline it calls).
- NEM prices, `data/clean/nem_prices/{year}_nem_prices.parquet`, 2021 to 2026, regional RRP.

## Volatility measures per state-month

- **Cap value**: mean over the month's intervals of max(RRP − 300, 0), $/MWh, the payout rate of a $300/MWh cap and the value available to batteries in price spikes.
- **Daily spread**: mean over the month's days of the mean of the highest two hours of RRP less the mean of the lowest two hours (local time days), the value available from daily arbitrage outside spikes.

Quartiles are formed on cap value.  A value on a boundary belongs to the lower group, so that state-months with no cap value, which can be more than a quarter of them, fall in Q1.  The same measures are computed for each state-month of 2026 to date (prices only), and their quartile under the 2021–2025 boundaries is reported, so the current market can be placed against the modelled months.

## Metrics

For each offer and state-month, the PV-size-weighted mean per site of the monthly household saving (Solar only `household_total` less the battery run's) and wholesale saving (the same, using the net wholesale cost as defined for `ws_saving` in `financial_analysis.py`), for the self-consumption reference, achievable optimised and perfect foresight optimised, against the offer's own Solar only run.  The reference has no VPP events or critical rates, so the coordination value (optimised less self-consumption) includes everything the offer's coordination earns: event dispatch, critical rates, VPP rewards and, for AGL, the participation credit (absent from the `_novpp` tariff).  Achievable and perfect foresight are measured from the same reference.

The same two savings for self-consumption and achievable optimised against the Amber Solar only run (the common baseline).

For each offer and volatility group, from the sums over the group's state-months (sums rather than means of monthly ratios, as single-month self-consumption savings can be close to zero):

- relative household value: Σ achievable optimised saving ÷ Σ self-consumption saving, achievable;
- relative wholesale value: the same for the wholesale saving;
- household value capture: Σ (achievable optimised − self-consumption) ÷ Σ (perfect foresight optimised − self-consumption);
- wholesale value capture: the same for the wholesale saving;
- share of the 2021–2025 coordination value (achievable optimised − self-consumption) earned in the group's months, household and wholesale;
- mean annual saving against the Amber Solar only run, for self-consumption and achievable optimised, household and wholesale: 12 × the mean over the group's state-months of the monthly saving.

A ratio whose denominator is not positive is not defined and is left empty.

For the offers with VPP events or critical rates the achievable runs are at maximum dispatch, as in the rest of Results.

## Charts

`docs/thesis_figure_spec.md` format, colours and offer names as in the other Results charts.  Both bar charts have two panels stacked at full width, household bill above net wholesale, grouped bars by cap value quartile, each panel titled so that it reads alone and no figure title.  Every title ends ", imperfect foresight (achievable), 20 kWh" (the capacity set by `--battery-kwh`); a title wider than its panel breaks before that suffix.  The x axis is labelled `$300 cap value quartiles ($/MWh)` and the ticks give the quartile ranges from the 2021–2025 boundaries, e.g. `< 1.0`, `1.0 - 5.0`, `5.0 - 10.0`, `> 10.0`.  Height 3.6 in.

**Relative value by volatility** (`analysis/output/sensitivity/volatility/coordination_by_volatility.png`).  One bar per offer, the relative value of coordination (achievable optimised ÷ self-consumption saving, against the offer's own Solar only run) with a reference line at 1.  Titles "Relative value of coordination by volatility, household bill" and "Relative value of coordination by volatility, net wholesale saving", each with the suffix.  Value labels horizontal, one decimal.  Value capture is not charted.

**Saving by volatility, common baseline** (`saving_by_volatility_common.png`).  The mean annual saving against the Amber Solar only run, as in `household_saving_by_offer_common.png` and the wholesale equivalent: Amber self-consumption (hatched) and each offer's achievable optimised run, one bar each.  Titles "Mean annual household bill saving by volatility" and "Mean annual wholesale saving by volatility", each with the suffix.  No value labels.  The y axis tops out at 5,000 with a tick every 1,000, and is extended to the next 1,000 if a bar exceeds it.

**Amber coordination value against cap value** (`coordination_vs_cap_value.png`, appendix candidate, height 3.0 in).  A single full-width panel titled "Amber household bill coordination value against cap value, imperfect foresight (achievable), 20 kWh".  Amber's monthly coordination value (achievable optimised − self-consumption, $ per site) against cap value (log axis, ticks 0.001, 0.01, … written as plain decimals), one marker per state-month coloured by state.  The quartile boundaries are dashed vertical lines.  State-months with no cap value sit on the left edge as open triangles.  The 2026-to-date state-months are square markers of the state's colour, the size of the data markers, at a y value halfway between zero and the lower y limit, at their cap value.

Data notes are drawn top left of every panel in the data note format of `thesis_figure_spec.md`, and `--no-notes` leaves them off for the thesis.  Three lines: the scope ("National (NSW, QLD, SA, VIC), 2021-2025"), the weighting ("Weighted by PV size, state-months pooled equally"), and the baseline ("Savings vs Amber Solar only" on the common baseline chart, "Savings vs each offer's own Solar only" on the others).

## Outputs

`analysis/output/sensitivity/volatility/`: the charts above, `state_month_metrics.csv` (one row per state-month and offer with both volatility measures, the quartile, and every monthly saving, against the offer's own and the Amber Solar only run), `volatility_group_metrics.csv` (one row per offer and quartile with every group metric and the quartile's cap value range), and `volatility_2026_to_date.csv` (one row per 2026 state-month with both measures, its quartile, its percentile among the modelled months' cap values, and a flag for a month not fully covered by prices).

## Values needed for the text

- Amber's relative household and wholesale value, and its value capture, in Q1 and Q4.
- The same for each time-of-use and VPP offer, separating household and wholesale.
- The share of Amber's coordination value earned in Q4 months.
- Where the 2026-to-date state-months fall against the 2021–2025 quartiles.

Measured.  Amber's relative household value runs 1.6, 1.8, 2.2 and 3.2 across Q1 to Q4, and its relative wholesale value 3.6, 3.9, 5.0 and 6.0, against a pooled household figure of 2.3 over all months.  Its household value capture stays between 0.66 and 0.75 across the four quartiles, so what falls in calm months is the headroom rather than the share captured.  57% of its 2021–2025 household coordination value is earned in Q4.  No time-of-use or VPP offer's household relative value moves by more than about 0.2 from Q1 to Q4, and the direction is not common to all of them, while on the wholesale cost GloBird runs 2.4 to 5.2, EnergyAustralia (NSW only) 1.1 to 4.2 and Origin 1.3 to 1.0.  Of the 36 complete 2026 state-months to September, 27 fall in Q1 and 7 in the top half, with a mean cap value of \$5.7/MWh against \$18.2/MWh over 2021–2025.

## Notes

- VIC 2021, QLD and SA 2025 and SA 2024 rest on fewer than 20 sites each (as noted in Household Value of Coordination).  They are included and flagged in the CSV.
- Before 1 October 2021 prices are 30-minute averages repeated across their 5-minute intervals (`nem_price_spec.md`), which lowers the daily spread of the early 2021 months.
- The relationship is a sensitivity of the 2021–2025 results to volatility, not a forecast for 2026 and later, since dispatch and prices interact as battery uptake grows.
