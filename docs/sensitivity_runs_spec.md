# Sensitivity runs — Specification

## Scope

Four sets of model runs for Results, Sensitivity Analysis (`thesis/04_results.md`), each on a small subset of the full runs, compared against the full run's own scenario for the same state, year and battery size:

1. Degradation cost (Amber, achievable imperfect foresight).
2. Rolling optimisation horizon (Amber, achievable imperfect foresight, every battery size).
3. VPP control window (Origin Battery Lite, perfect foresight).
4. Impact of nudge (Origin & EnergyAustralia, conservative imperfect foresight, every battery size)

Full runs at every variant would take too long.  The subset is NSW 2024, chosen as a recent year with frequent price spikes and the largest state sample.  

NSW 2024 holds 101 sites after the study-wide exclusions (`analysis/scripts/site_exclusions.py`).

Each run is a copy of the production config restricted to `states: [NSW]`, `years: [2024]` and the battery sizes listed, with the one setting changed and a `sens_` run name, run with `--save-flows` so battery throughput and wholesale flows are available.  Configs go in `model/configs/sensitivity/`, commands in `runners/run_sensitivity.sh`.

Where the run time demands it, a run covers a subset of the 101 sites.  `runners/run_sensitivity.sh` carries a `SITES` variable for this and `analysis/scripts/sensitivity/select_sites.py` produces a subset that holds the PV size band mix of the full set, taking sites at evenly spaced ranks within each band so the subset spans each band.  Every comparison is restricted to the sites common to all of the runs it reads, with the weights renormalised over that set, so a subset stays a paired comparison against the full run rather than a comparison of two different site mixes.

Results are processed by `analysis/scripts/sensitivity/`, one module per set over the shared loading and weighting in `sensitivity_common.py`, each writing its CSV and figure to `analysis/output/sensitivity/<set>/` and printing the values the text cites under `--print-values`.  Site weights are `financial_analysis.py`'s PV size band weights, so a site carries the same weight here as in the results these are compared against.

## 1. Degradation cost

| Run | Base config | Change |
|---|---|---|
| `sens_cdeg034_rolling_amber` | `full_run_rolling_amber.yaml` | `c_deg_per_kwh: 0.034` |
| `sens_cdeg046_rolling_amber` | `full_run_rolling_amber.yaml` | `c_deg_per_kwh: 0.046` |

20 kWh only.  3.4 and 4.6 c/kWh are the ends of the calibration range against the 3.9 c/kWh used in every full run (Battery Degradation Calibration).  The 3.9 c/kWh reference is `full_run_rolling_amber/scenarios/NSW_2024_20kWh`.

Self-consumption and Solar only do not depend on the degradation cost, so the existing `full_run_self_consumption_amber` and `full_run_solar_only_amber` NSW 2024 scenarios are the references for the savings.

Outputs at each of the three degradation costs:

- household battery saving against Solar only, and the coordination value (that saving less the self-consumption saving);
- net wholesale saving against Solar only;
- annual battery discharge throughput (kWh, from `battery_flows.parquet`) and equivalent full cycles per day.

The result is reported as a table of the four metrics at the three degradation costs.  A chart of three bars per metric adds little to it, so one is produced for the appendix rather than the body.

Chart (`docs/thesis_figure_spec.md`, height ~3.0 in): two bar panels, three bars each, one per degradation cost, with 3.9 c/kWh emphasised.  A single battery size gives one value per run, so the panels are bars rather than lines against battery size.  The two panels are whichever metrics move most across the calibration range, taken from different families so the panels are not two views of one number, since equivalent full cycles per day is annual discharge rescaled and the coordination value is the household saving less a reference that does not move with the degradation cost.  Bars are drawn from zero at their measured value.  The self-consumption value is available as a flat reference on every panel under `--sc-reference` and is off by default, because self consumption does not depend on the degradation cost and so carries nothing about the sensitivity the chart is about.  Output `analysis/output/sensitivity/degradation/degradation_sensitivity.png` and a CSV with every output above, which also carries the self-consumption saving and cycling for the text.

## 2. Rolling horizon

| Run | Base config | Change |
|---|---|---|
| `sens_h06_rolling_amber` | `full_run_rolling_amber.yaml` | `horizon_hours: 6.0` |
| `sens_h12_rolling_amber` | `full_run_rolling_amber.yaml` | `horizon_hours: 12.0` |
| `sens_h24_rolling_amber` | `full_run_rolling_amber.yaml` | `horizon_hours: 24.0` |
| `sens_h48_rolling_amber` | `full_run_rolling_amber.yaml` | `horizon_hours: 48.0` |

Every battery size, 5 to 50 kWh.  The 18 h reference is `full_run_rolling_amber/scenarios/NSW_2024_{size}kWh`.  Beyond pre-dispatch coverage (15.1 to 39 h ahead) the price provider infills from earlier prices (`docs/rolling_optimisation_spec.md`, Horizon and infill), so the 24 and 48 h runs test a longer window with partly infilled prices, which is the choice a real controller faces.

Outputs per run and battery size:

- household battery saving and coordination value, net wholesale saving (as above);
- value capture ratio against perfect foresight (`full_run_optimised_amber`, NSW 2024, same size): coordinated less self-consumption saving, rolling over perfect foresight, for the household bill and the wholesale saving.  It is pooled, the weighted mean of the numerator over the weighted mean of the denominator, so a site whose own headroom is near zero cannot blow the ratio up;
- share of window entries using infilled prices (`rolling_diagnostics.csv`);
- median and 95th percentile window solve time.  `rolling_diagnostics.csv` carries one median and one 95th percentile per site, and each is summarised as the median over sites, so both read as the typical site's figure.

Chart (`docs/thesis_figure_spec.md`, height ~3.0 in, two panels): household (left) and wholesale (right) value capture ratio against battery capacity, one line per horizon, 18 h in the emphasis colour and not labelled as the production setting, as on the other sensitivity charts.  The other horizons take `thesis_style.CATEGORICAL_COLOURS` in horizon order, and the legend is unboxed below the panels in that same order, so it reads as a sequence.  The title carries the offer and the foresight basis and the data note carries the scope, the site count and the weighting, following `volatility_sensitivity.py`; `--no-notes` leaves the note off.  Output `analysis/output/sensitivity/horizon/horizon_value_capture.png` and a CSV with every output above.

A horizon missing at one battery capacity drops that horizon at that capacity only, and the others are still charted there, so a partially complete run set charts what exists rather than losing the capacity entirely.

## 3. VPP control window

| Run | Base config | Change |
|---|---|---|
| `sens_w15min_optimised_origin_lite` | `full_run_optimised_origin_lite.yaml` | retail tariff `NSW_origin_battery_lite_vpp_2026_w15min` |

`NSW_origin_battery_lite_vpp_2026_w15min.yaml` is a copy of `model/tariffs/NSW_origin_battery_lite_vpp_2026.yaml` with `event_window_hours: 0.25` and nothing else changed.  20 kWh only.

The 1 h reference is `full_run_optimised_origin_lite/scenarios/NSW_2024_20kWh` and the 10 h comparison is `10hr_full_run_optimised_origin_lite/scenarios/NSW_2024_20kWh`, run on the current 2026 rates with `event_window_hours: 10.0` (its daily charges match the current supply charge of 139.04 c/day).

Subset check: the 1 h and 10 h full runs exist for every state, year and size, so the difference between them in NSW 2024 at 20 kWh is compared with the national 20 kWh difference (weighted as in `financial_analysis.py`).

Measured, the two are not similar.  The 10 h less 1 h difference is -\$23 on the household bill and +\$326 on the net wholesale saving in NSW 2024, against -\$8 and +\$112 nationally, so the subset carries about three times the national effect.  VPP events are anchored on price spikes and NSW in 2024 had them often, which is the same property the subset was chosen for.  NSW 2024 is therefore an upper bound rather than a representative case.  The household result survives this, since it is small in the amplified case and smaller nationally, while the wholesale gain from a longer window is quoted at the national figure.

Outputs for 15 min, 1 h and 10 h (NSW 2024, 20 kWh), and for 1 h and 10 h nationally:

- household battery saving against Solar only (Origin Solar only run), and the VPP reward within it;
- forced discharge and forced charge (kWh/yr, `rolling_diagnostics.csv` or `vpp_forced_flows.parquet` as in `flow_exposure_analysis.load_forced_totals`);
- net wholesale saving against Solar only;
- retailer margin (`bills.csv`).

Chart (`docs/thesis_figure_spec.md`, height ~3.0 in): two bar panels, three bars each, one per control window, 1 h emphasised, selected from the outputs above the same way as the degradation chart.  These runs are perfect foresight, so the quantity charted is the saving itself rather than value capture against perfect foresight.  Output `analysis/output/sensitivity/control_window/control_window_sensitivity.png`, a CSV with every output above, and a second CSV holding the subset against national check.

## 4. Impact of nudge

| Run | Base config | Change |
|---|---|---|
| `sens_nudge_rolling_origin_lite_novpp` | `full_run_rolling_origin_novpp.yaml` | nudge enabled |
| `sens_nudge_rolling_ea_novpp` | `full_run_rolling_ea_novpp.yaml` | nudge enabled |

The nudge is `apply_export_order_nudge: true`, which the planner adds to its base export price so every rolling window breaks the flat feed-in tie in favour of charging first (`rolling_perfect_foresight_lp_advanced_spec.md`, Export-order nudge).

No arm carries a VPP enrolment payment.  Every series is on the retailer's no-VPP tariff, and `retailer_briefing.py` draws the same no-nudge run as its `TOU optimised` line without one, adding EnergyAustralia's \$180 a year only to the band edge that stands for an enrolled but undispatched battery, which is not a series here.  Adding it lifts the two rolling curves against self consumption by a fixed amount and hides the sizes at which imperfect foresight falls below self consumption, which is the artefact the nudge exists to remove.  It leaves the nudge gain unchanged, since it applies to both rolling arms equally.

Savings are against each retailer's own Solar only run, `full_run_solar_only_origin_starter` and `full_run_solar_only_ea`, so each curve is the value of the battery on that offer.  The self-consumption references are `full_run_self_consumption_origin_novpp` and `full_run_self_consumption_ea`, neither of which depends on the nudge.

Chart (`docs/thesis_figure_spec.md`, height 3.0 in, two panels): Origin (left) and EnergyAustralia (right) showing household bill saving by battery size, with lines for self consumption, imperfect foresight without the nudge and imperfect foresight with it.

The encoding is `retailer_briefing.py`'s, so the two charts read alike.  Each retailer keeps its own hue, dark for the coordinated arms and mid for self consumption, and the series are named as that script names them: `Self-consumption`, `TOU optimised` and `TOU optimised + nudge`.  The foresight basis is one per chart and sits in the caption rather than in every series name.  The two arms the briefing also draws keep its solid line, so a series that appears on both charts looks the same on both.  The nudge arm takes a long dash, `(0, (9, 3))`, which is clear of the briefing's short dash for perfect foresight and its dotted VPP band edges.  Curves start at the zero-capacity anchor, where the saving against that retailer's own Solar only run is zero by construction, and are smoothed between the modelled capacities by `financial_analysis._smooth_xy_piecewise` with a marker at each capacity run.

Each panel carries its own legend, in that retailer's own hues, ordered highest-saving first so it matches the vertical stacking of the curves.  Its keys carry no marker, since every arm uses the same one, and are set to `handlelength=4.4` rather than the default 1.6.  The dashed key is meant to read as line, gap, line, which needs dash plus gap plus dash, 27.3 pt at this line width, against a default key of 10.4 pt; below that the key shows no gap at all and reads identically to the solid arm above it.  The data note follows the briefing's format and is left off with `--no-notes`.  Output `analysis/output/sensitivity/nudge/nudge_impact.png` and a CSV with the saving on each arm by battery size, the nudge gain and that gain as a share of the self-consumption saving.

## Values needed for the text

- Degradation: change in the 20 kWh household saving, coordination value, wholesale saving and cycles per day at 3.4 and 4.6 c/kWh.  At one battery size there is no statement about how the degradation cost interacts with battery size.
  - Measured (NSW 2024, 101 sites): household saving +0.2% and -0.4%, coordination value +0.3% and -0.6%, net wholesale saving +0.7% and -1.1%, cycles per day +4.0% and -5.4%, against 3.9 c/kWh.  Self consumption at the same size saves \$1,188 on the household bill and \$455 on the net wholesale cost and cycles 0.443 times a day.
- Horizon: value capture at 6, 12, 18, 24 and 48 h for 20 and 50 kWh, and the battery size, if any, above which a longer horizon than 18 h adds more than ~1% of value capture.
  - Measured (NSW 2024, 101 sites).  18 h is the best of the five at every capacity on both metrics, so no size gains from a longer horizon.  Household value capture ratio at 20 kWh runs 0.643, 0.657, 0.665, 0.664 and 0.664 across 6, 12, 18, 24 and 48 h, and at 50 kWh 0.739, 0.852, 0.891 and 0.890 across 6, 12, 18 and 24 h.  Infill rises 0%, 0%, 1.0%, 6.8%, 43.3% and the median window solve time 0.20, 0.36, 0.49, 0.71, 1.45 ms.  Under the Energy Masters sweep's own rule, the shortest horizon within 0.005 of the best, 6 h suffices at 5 kWh, 12 h at 10 kWh and 18 h from 15 kWh up.
- Control window: household saving, VPP reward, forced flows, wholesale saving and retailer margin at 15 min, 1 h and 10 h, and whether the NSW 2024 1 h to 10 h difference matches the national one.  At one battery size these are levels at 20 kWh, with no statement about how the control window interacts with battery size.
  - Measured (NSW 2024, 101 sites, 20 kWh): household saving \$1,599, \$1,612 and \$1,590; forced discharge 131, 150 and 168 kWh/yr; forced charge 39, 66 and 111 kWh/yr; net wholesale saving \$2,134, \$2,509 and \$2,834; retailer margin \$1,503, \$1,859 and \$2,216.  The household saving spans 1.4% and is highest at 1 h, as the reward on forced discharge offsets the optimisation the operator displaces.
  - The VPP reward equals the forced discharge in kWh, since Origin pays \$1/kWh on it, so a table carrying both repeats one quantity.
- Impact of nudge: the battery size after which the nudge has limited impact on the results.
  - Measured (NSW 2024, 101 sites), nudge gain as a share of the self-consumption saving falling monotonically: Origin below 5% from 35 kWh, EnergyAustralia from 25 kWh.  Without the nudge, imperfect foresight is below self consumption at 5 kWh on Origin (-\$44) and at 5 and 10 kWh on EnergyAustralia (-\$78 and -\$11), and above it at every size with the nudge.  On Origin the gain turns slightly negative by 50 kWh (-\$6).

## Notes

- The Methodology's control window sentence previously read "limited impact on household or retailer outcomes" up to the 10-hour period.  The retailer's margin rises by almost a half over that range, so it now claims limited impact on the household saving alone and points to Results, Sensitivity Analysis for the retailer and wholesale effects.  For the degradation cost, Battery Degradation Calibration now points to Results, Sensitivity Analysis for what the uncertainty in the fitted value costs.
- Run time: the horizon set is 40 scenarios (4 horizons × 10 sizes) and the nudge set 20 (2 retailers × 10 sizes), against two scenarios for degradation and one for the control window.  The horizon set dominates, with the 48 h windows the slowest, since window solve cost rises with the number of intervals in the window.  `runners/run_sensitivity.sh` runs the sets cheapest first and the longest horizon last, so the horizon set can be stopped part way without losing the other three.  `analysis/scripts/sensitivity/run_times.py` reports measured site counts and per-site solve times from existing runs' `summary.txt`.
- Measured on the production runs (NSW 2024, 101 sites, 6 workers): `full_run_rolling_amber` takes 29m36s at 20 kWh and 29m57s at 50 kWh, so battery size moves the rolling run's cost by about 1%, and the ten-size ladder costs about ten times one scenario.  `full_run_rolling_origin_novpp` takes 21m33s and `full_run_rolling_ea_novpp` 27m13s at 20 kWh.  The perfect-foresight VPP runs are far cheaper, `full_run_optimised_origin_lite` taking 2m06s.
- Window solve cost is linear in horizon length.  Measured at 20 kWh on three sites, LP solve is 102.3 s/site at 24 h and 203.1 s/site at 48 h, an exponent of 0.99 on the window length.  Both points imply the same 77.0 s/site at 18 h for those sites, against the 101-site average of 103.8 s/site, so `--max-sites` takes a cheaper than average subset and its absolute per-site times are not representative; the ratio between two horizons on the same subset is.  `runners/run_sensitivity.sh calibrate` reruns this with the configs in `model/configs/sensitivity/calibration/`.
- At 101 sites and every battery size the four sets take about 36 hours on six workers: 25 h for the horizon set, 10 h for the nudge set, 1 h for degradation and under 10 minutes for the control window.
