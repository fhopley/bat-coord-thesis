# Battery cycling and VPP forced flows chart — Specification

## Scope

One two-panel thesis figure for Results, Battery Utilisation and Cycling (`thesis/04_results.md`), and an exploratory chart of cycles per day against battery size.  Format follows `docs/thesis_figure_spec.md` (`analysis/scripts/thesis_style.py`).

`flow_exposure_analysis.py` computes the values on the achievable basis and writes them as CSVs.  `battery_cycling_figure.py` draws both charts from those CSVs and prints the values the text cites.

```bash
python3 analysis/scripts/flow_exposure_analysis.py --basis achievable --cycling-sweep
python3 analysis/scripts/battery_cycling_figure.py
```

## Common settings

| Item | Value |
|---|---|
| Battery size | 20 kWh for the thesis figure (`--kwh`); 5–50 kWh in 5 kWh steps for the size chart |
| Years and regions | 2021–2025, NSW, QLD, SA, VIC |
| Weighting | PV size band within each state, years averaged without weighting within a state, states combined by the national state mix, as in `financial_analysis.py` |
| Configuration | Achievable imperfect foresight only |
| Offers | The `flow_exposure_analysis.py` registry: Amber, GloBird, AGL, EnergyAustralia, Origin (Battery Lite).  Origin Battery Maximiser and Tesla Electric are not in it |
| Offer names on the axis | Self-consumption, Amber, GloBird, AGL, EnergyAustralia, Origin (`OFFER_NAMES`) |
| Colours | The registry colours used by the other Results charts |

Each offer's run is its achievable optimised run from `financial_analysis.PLAN_RUNS`.  The VPP offers use their VPP runs, and GloBird ZeroHero has critical rates on:

| Offer | Run |
|---|---|
| Self-consumption | `full_run_self_consumption_amber` (self-consumption flows are the same on every offer) |
| Amber | `full_run_rolling_amber` |
| GloBird | `full_run_rolling_globird` |
| AGL | `full_run_rolling_agl` |
| EnergyAustralia | `full_run_rolling_ea` |
| Origin | `full_run_rolling_origin` |

## Left panel: battery cycling against warranty allowances

- **Quantity.** Equivalent full cycles per day = weighted mean annual battery discharge per site (kWh, from `battery_flows.parquet`) ÷ (battery size × 365).  The divisor is the modelled battery capacity, consistent with the warranty allowance calculation in the Methodology.
- **Bars.** One bar per offer in the order above, Self-consumption in its outlined style, each labelled with cycles per day to two decimal places.
- **Axis.** "Equivalent full cycles per day", 0 to at least 1.3.
- **Reference lines.** Warranty allowances (Methodology, Battery Utilisation and Warranty Implications), given in MWh/yr of discharge for a 20 kWh battery and drawn as dashed grey lines in cycles per day, labelled at the right: Tesla Powerwall 5.6 MWh/yr (0.77), SigEnergy and FoxESS (low) 7.4 MWh/yr (1.01), FoxESS (high) 8.4 MWh/yr (1.15).  In cycles per day they are the same at every battery size.
- **Panel subtitle.** "Battery cycling against warranty allowances".

## Right panel: VPP forced flows against the annual caps

- **Offers.** The VPP offers (those with a cap): AGL, EnergyAustralia, Origin.
- **Quantities.** Weighted mean per site per year of forced discharge (`f_exp_total_kwh`) and forced charge (`f_imp_total_kwh`) from `rolling_diagnostics.csv` in each scenario folder, and their total.  Forced flows are the export and import beyond what self-consumption would have produced in each event window (Methodology, VPP Event Dispatch).
- **Columns.** A three-column waterfall per offer: discharge from zero, charge floating on top of discharge, total from zero.  Discharge starts from zero so that a discharge-only cap can be marked directly on it.
- **Caps.** A short dashed line across the column the cap applies to, labelled with its value.  The cap is the `kwh_limit` of the retail tariff's VPP block, on the total column under `limit_applies_to: import_and_export` and on the discharge column under `export_only`, checked against `cap_limit_kwh` in `rolling_diagnostics.csv`.  An offer whose tariffs disagree across scenarios gets no cap mark.
- **Legend.** The discharge, charge and total column styles, once, below the panel.
- **Axis.** "Forced flow (kWh/yr)", 0 to at least 300.
- **Panel subtitle.** "VPP forced flows against annual caps".

## Layout and output

- Two panels, figure title "Battery cycling and VPP forced flows at 20 kWh", height 3.0 in.
- No notes drawn in the figure (data, period, weighting and configuration go in the thesis caption).
- Outputs in `analysis/output/exposure/imperfect_foresight_achievable/national/`:
  - `battery_cycling_20kwh.csv`: per offer, annual discharge (kWh), cycles per day, forced discharge, forced charge, total forced flow, cap and the column it applies to;
  - `battery_cycling_and_forced_flows.png`, also written to `thesis/figures/results/` (`--thesis-dir`).

## Size chart: cycles per day against battery size

- `--cycling-sweep` writes `battery_cycling_by_size.csv` (national annual discharge and cycles per day for each offer at each size).  It reads every size's `battery_flows.parquet`, so it is slow.
- One line per offer, same order and colours as the left panel, cycles per day = annual discharge ÷ (battery size × 365).
- Warranty allowances as faint dashed lines, each with a marker at every usable capacity its product line is installed at up to 50 kWh (Tesla Powerwall 13.5 kWh units; SigEnergy and FoxESS stack sizes, `PRODUCT_UNITS`).
- Output `cycles_by_battery_size.png` beside the CSVs.  Used for the battery-size sentence in the text.

## Values printed for the text

Cycles per day and annual discharge for each offer at 20 kWh; for each VPP offer, the forced flow on its capped quantity (discharge for EnergyAustralia and Origin, total for AGL) as a share of its annual discharge; cycles per day on Self-consumption and Amber at 10, 20 and 50 kWh.
