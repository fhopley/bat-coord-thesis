# Retail Models for Increased Consumer Energy Resource Flexibility

Code for the UNSW Sydney Master of Engineering thesis *Retail Models for
Increased Consumer Energy Resource Flexibility* (Fred Hopley, 2026).

The repository models residential solar and battery economics in Australia's
National Electricity Market (NEM):

- **Data pipelines** (`pipeline/`) clean and infill 5-minute household energy
  data, and prepare NEM wholesale prices and AEMO pre-dispatch price forecasts.
- **Coordination model** (`model/`) dispatches each household's battery under
  a coordination strategy (self-consumption, perfect-foresight optimisation,
  VPP operation, rolling optimisation on forecasts) and bills the result at the
  wholesale, network and retail layers.
- **Analysis** (`analysis/`) turns model outputs into the financial, exposure
  and retailer results and figures in the thesis.

---

## Data

The thesis results use household data that cannot be shared. The repository
includes one sample household, H0001, so that the pipelines, model and
analyses can be run.

| Data | Included | Source |
|---|---|---|
| EDP household data (Solar Analytics, Wattwatchers), NSW, QLD, SA, VIC, 2021–2025 | No | Proprietary |
| Energy Masters household data | No | Proprietary |
| Sample household H0001, NSW, 2022–2024: raw and clean 5-minute traces, site metadata | Yes | `data/raw/H0001/`, `data/clean/edp/`, `data/processed/edp/` |
| NEM prices, 2020–2026 | Yes | AEMO, via NEMOSIS. `data/clean/nem_prices/` |
| Pre-dispatch price forecasts, NSW, 2022–2024 | Yes | AEMO NEMWeb. `data/clean/predispatch/` |
| AEMO MCL calculators | Yes | AEMO. `docs/references/aemo_prudential/` |
| Literature review figure data | Yes | Sources cited in the thesis. `data/processed/literature_review/` |

H0001 is a NSW household with an 11.6 kWac / 14.8 kWdc PV system on three
phases and no battery. Its data has the same format, flagging and infill as the
EDP households (`docs/edp_data_spec.md`).

`model/configs/full_run_*.yaml` are the thesis runs over the EDP households and
fail validation without that data. Each has a counterpart in
`model/configs/demo/` that runs H0001 for NSW 2022–2024. A demo config writes
to the same output folder as its full run, so the analysis scripts read it
unchanged.

`energy_masters_pipeline/` is included as code only.

---

## Setup

Python 3.12 or later. From the project root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
python3 -m pytest model/tests -m "not slow"
```

All commands run from the project root.

---

## Running on the sample household

**Model.** Validate and run the demo configs. `run_demo_runs.sh` takes config
names to run a subset.

```bash
./runners/validate_demo_runs.sh
./runners/run_demo_runs.sh
```

**Analysis.** With the demo runs complete, the analysis scripts run on the
sample household, for example:

```bash
python3 analysis/scripts/financial_analysis.py
python3 analysis/scripts/savings_bar_figure.py --scope NSW_all_years
python3 analysis/scripts/flow_exposure_analysis.py
python3 analysis/scripts/import_price_analysis.py
python3 analysis/scripts/reliability_response_analysis.py
python3 analysis/scripts/retailer_briefing.py --retailer amber
python3 analysis/scripts/cap_hedge_analysis.py
```

Scripts with a `--scope` option default to the national result, which needs
more than one state. Scripts that read only the national result, such as
`battery_cycling_figure.py`, do not run on the sample household.

---

## Repository structure

```
pipeline/
  edp/                    EDP cleaning scripts A_ to L_
  h0001/                  cleaning scripts A_ to D_ for the sample household
  nem_prices/             NEM price download and cleaning
  pre_dispatch/           AEMO pre-dispatch download (A), clean (B), validate (C)
  run_pipeline.py         runs the EDP scripts for one year
model/
  core/                   shared types: scenario, tariffs, local time, prices, outputs
  components/             ingest, self-consumption, billing, dispatch components
  forecast/               price, usage and solar forecast providers
  configs/                scenario YAML files: examples, full runs, demo/, sensitivity/
  tariffs/                tariff YAML files
  dev/                    single-site views, checks and diagnostics
  scripts/                output utilities, including make_demo_configs.py
  tests/                  pytest suite and fixtures
  orchestrate.py          validation, routing, runners, output writers
  main.py                 CLI entry point
analysis/
  scripts/                analyses and thesis figure scripts
  config/                 analysis configuration
energy_masters_pipeline/  Energy Masters ingest and value-capture analysis (code only)
docs/                     specifications
runners/                  batch runners (runners/README.md)
data/                     sample data; pipeline and model outputs are written here
```

The thesis text is not included. Specs refer to its sections, and the thesis
figure scripts write a copy of each figure to `thesis/figures/`.

---

## Data pipelines

**Sample household.** Rebuilds the clean H0001 traces and metadata from
`data/raw/H0001/`. Method 2 PV infill needs the PV profile library built from
the EDP households, which is not included; the shipped clean files already
contain the infill.

```bash
python3 pipeline/h0001/run_h0001.py [YEAR]
```

**EDP households**, one year at a time (`docs/edp_data_spec.md`):

```bash
python3 -u pipeline/run_pipeline.py 2022
```

**NEM prices** (`docs/nem_price_spec.md`):

```bash
python3 pipeline/nem_prices/A_Download_NEM_Prices.py
python3 pipeline/nem_prices/B_Clean_NEM_Prices.py
```

**Pre-dispatch forecasts**, needed for `forecast_source: predispatch`
(`docs/predispatch_data_spec.md`):

```bash
python3 -u -m pipeline.pre_dispatch.A_Download_Predispatch --start 2022-01 --end 2024-12
python3 -u -m pipeline.pre_dispatch.B_Clean_Predispatch --start 2022-01 --end 2024-12
python3 -u -m pipeline.pre_dispatch.C_Validate_Predispatch
```

---

## Coordination model

```bash
python3 -u -m model.main model/configs/example_perfect_foresight_lp.yaml
```

Options:

- `--validate-only` checks every scenario in a YAML without running anything.
- `--workers N` runs sites in parallel.
- `--max-sites N` runs the first N sites only.
- `--monthly-bills` adds monthly bill files.
- `--save-flows` writes per-interval flow Parquet files.
- `--save-trace SITE_ID` writes one site's full trace.

Output goes to `data/output/runs/{run_name}/scenarios/{scenario_id}/`.
The YAML schema, CLI and output format are in `docs/model_spec.md`.

| Coordination type | Dispatch | Example config |
|---|---|---|
| `self_consumption` | solar → load → battery → grid, no optimisation | `example_single_scenario.yaml` |
| `perfect_foresight` | full-year LP on known prices | `example_perfect_foresight_lp.yaml` |
| `perfect_foresight_advanced` | full-year LP with block tiers, zero-import reward and critical rates | `example_perfect_foresight_lp_advanced.yaml` |
| `perfect_foresight_custom` | greedy heuristic alternative to the LP | `example_perfect_foresight_custom.yaml` |
| `vpp_sc_event` | operator LP during VPP events, self-consumption between | |
| `vpp_tou_event` | operator LP during VPP events, TOU LP between | `example_vpp_tou_event.yaml` |
| `rolling_perfect_foresight_lp` | rolling LP on price, load and solar forecasts (wholesale-exposed plans) | |
| `rolling_perfect_foresight_lp_advanced` | rolling advanced LP on forecasts (TOU plans) | `example_rolling_perfect_foresight_lp_advanced.yaml` |
| `rolling_vpp_sc_event`, `rolling_vpp_tou_event` | VPP types with events declared from forecasts | `example_rolling_vpp_sc_event.yaml`, `example_rolling_vpp_tou_event.yaml` |

`example_no_cer.yaml` runs the household without solar or battery, and
`example_multi_battery.yaml` runs several battery specifications.

`model/dev/` holds single-site views and checks, run as
`python3 -m model.dev.<name>`; each script's `--help` lists its options.

---

## Analysis

`analysis/scripts/` reads model outputs from `data/output/runs/`.

| Script | Spec |
|---|---|
| `financial_analysis.py`, `savings_bar_figure.py` | `docs/financial_analysis_spec.md` |
| `flow_exposure_analysis.py` | `docs/flow_exposure_spec.md` |
| `battery_cycling_figure.py` | `docs/battery_cycling_chart_spec.md` |
| `average_day_profile.py` | `docs/average_day_profile_spec.md` |
| `cap_account_analysis.py`, `cap_account_aggregate.py`, `cap_account_retailer.py` | `docs/cap_account_spec.md` |
| `offer_outcomes_by_size.py`, `alt_offers_by_size_figures.py`, `offer_outcomes_text_values.py` | `docs/cap_account_battery_size_spec.md` |
| `cap_hedge_analysis.py`, `asx_caps.py` | `docs/cap_hedge_spec.md` |
| `cap_hedge_battery_size.py` | `docs/cap_hedge_battery_size_spec.md` |
| `reliability_response_analysis.py` | `docs/reliability_response_spec.md` |
| `import_price_analysis.py` | `docs/import_price_spec.md` |
| `prudential_analysis.py` | `docs/prudential_spec.md` |
| `sensitivity/` | `docs/sensitivity_runs_spec.md` |
| `volatility_sensitivity.py` | `docs/volatility_sensitivity_spec.md` |
| `thesis_style.py` | `docs/thesis_figure_spec.md` |
| the rest | docstrings |

`site_exclusions.py` lists the households excluded from every analysis.

---

## Documentation

| Document | Contents |
|---|---|
| `docs/model_spec.md` | Model architecture, sign conventions, scenario and tariff schema, local time, prices, output types, orchestration, CLI and outputs |
| `docs/edp_data_spec.md` | Household data, flags, infill and the clean trace format |
| `docs/nem_price_spec.md` | NEM price source, processing and validation |
| `docs/predispatch_data_spec.md` | Pre-dispatch forecast store |
| `docs/perfect_foresight_lp_spec.md` | `perfect_foresight` |
| `docs/perfect_foresight_advanced_spec.md` | `perfect_foresight_advanced` and the advanced tariff features |
| `docs/perfect_foresight_custom_spec.md` | `perfect_foresight_custom` |
| `docs/vpp_sc_event_spec.md`, `docs/vpp_tou_event_spec.md` | Perfect-foresight VPP types |
| `docs/rolling_optimisation_spec.md` | The rolling loop, `RollingConfig`, `rolling_perfect_foresight_lp` |
| `docs/rolling_perfect_foresight_lp_advanced_spec.md` | `rolling_perfect_foresight_lp_advanced` |
| `docs/rolling_vpp_event_spec.md` | Rolling VPP types |
| `docs/forecast_providers_spec.md`, `docs/usage_forecast_spec.md`, `docs/solar_forecast_spec.md` | Price, load and solar forecasts |
| `docs/financial_analysis_spec.md`, `docs/flow_exposure_spec.md`, `docs/cap_account_spec.md`, `docs/cap_hedge_spec.md`, `docs/import_price_spec.md`, `docs/prudential_spec.md`, `docs/reliability_response_spec.md` | Analysis designs |
| `docs/average_day_profile_spec.md`, `docs/battery_cycling_chart_spec.md`, `docs/cap_account_battery_size_spec.md`, `docs/cap_hedge_battery_size_spec.md`, `docs/sensitivity_runs_spec.md`, `docs/volatility_sensitivity_spec.md` | Analyses and figures for specific Results sections |
| `docs/thesis_figure_spec.md` | Format of the thesis figures |

---

## Licence and citation

Code: MIT licence (`LICENSE`). Sample household data: CC BY 4.0. AEMO data is
subject to AEMO's copyright permissions.

To cite this work, see `CITATION.cff`.
