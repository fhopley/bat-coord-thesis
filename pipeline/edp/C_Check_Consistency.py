"""
C_Check_Consistency.py

Programmatic consistency checks for EDP site data across all 12 months.
Applies seven checks per site.

Battery site correction (see docs/edp_data_spec.md, Battery sites): for sites that
B_Site_Selection.py marked as a usable battery reconstruction (non-null
battery_channel_label / battery_sign_convention), the raw battery-labelled
channel is pulled alongside ac_load_net and pv_site_net, and an
ac_load_net_corrected column is computed as the counterfactual "no battery"
net site flow:
    ac_load_net_corrected = ac_load_net - battery_channel   (sign=subtract)
    ac_load_net_corrected = ac_load_net + battery_channel   (sign=add)
For ordinary (non-battery) sites, ac_load_net_corrected is identical to
ac_load_net. Checks 1, 2, 2b, 5b, and 7 use ac_load_net_corrected wherever
they previously used ac_load_net directly, since they test physical/
behavioural plausibility of the load (export consistency, correlation with
PV) and the counterfactual battery-free value is the physically meaningful
quantity for that purpose. Check 3 deliberately continues to test the RAW
ac_load_net, not the corrected value, because it is a sensor-fault check
(stuck/flatlined meter reading), and a genuinely stuck raw sensor could be
masked by a legitimately varying battery term if the corrected value were
used instead -- correcting it would hide the exact fault the check exists to
catch. This is a specific design decision, not an oversight; flag it if you
disagree with the reasoning. If a battery-labelled channel is present in raw
data but has no value for a given interval, ac_load_net_corrected silently
falls back to the raw ac_load_net for that interval rather than becoming
NaN -- a known simplification, acceptable since gaps in the battery channel
specifically are expected to be rare (B0_Battery_Site_Screen.py already
requires >=5% bidirectional coverage and a real annual throughput before a
site reaches this stage at all).

Check 1 — PV/load correlation during solar hours:
    During daylight hours when PV generation is meaningful, ac_load_net should
    decrease as pv_site_net increases. A weakly negative or positive correlation
    indicates misconfigured metering (e.g. ac_load_net measuring gross load).

Check 2 — Export magnitude consistency:
    A PV-only site cannot export more than it generates. Two sub-checks are
    applied: (2a) intervals where -ac_load_net exceeds pv_site_net by more than
    EXPORT_NOISE_WH, consistent with an undisclosed battery or unmonitored
    inverter; and (2b) fraction of all intervals where gross_load < 0 (any
    amount), which catches partial phase metering where ac_load_net is
    systematically too small in magnitude to balance PV generation.

Check 3 — Flatline / zero-value ac_load_net during solar hours:
    During solar hours, PV generation drives continuous variation in ac_load_net
    even for unoccupied houses — so near-zero ac_load_net during these hours is
    suspicious and likely indicates a stuck or faulty sensor. Both numerator and
    denominator use solar hours only. Threshold of 2 Wh allows flexible loads
    (e.g. EV solar-following, residuals 5–15 Wh) while catching stuck sensors
    (~0 Wh exactly). Uses raw ac_load_net, not the battery-corrected value —
    see battery site correction note above.

Check 4 — Flatline / zero-value pv_site_net during solar hours:
    Extended periods of zero or near-zero pv_site_net during daylight hours
    indicate a sensor fault or extended outage not captured by the coverage
    filter. Only daytime intervals (SOLAR_HOUR_START to SOLAR_HOUR_END) are
    counted — nighttime zeros are expected and excluded.

Check 5 — Minimum annual generation and minimum average daily gross load:
    Two sub-checks: (5a) total annual pv_site_net must exceed
    MIN_ANNUAL_GENERATION_WH — catches non-functional or implausibly small
    systems; (5b) average daily gross load must exceed MIN_AVG_DAILY_GROSS_LOAD_WH
    — catches empty or unoccupied sites where load data is unreliable.

Check 6 — Daytime negative pv_site_net:
    A small number of negative PV readings overnight is normal inverter standby
    draw. However, negative PV during known daylight hours (SOLAR_HOUR_START to
    SOLAR_HOUR_END) indicates a genuine metering fault such as reversed polarity.
    Sites where more than PV_NEGATIVE_THRESHOLD fraction of daytime pv_site_net
    intervals are below PV_NEGATIVE_WH are excluded.

Check 7 — Gross load vs PV correlation during solar hours:
    On a correctly metered site, gross_load = ac_load_net + pv_site_net reflects
    household consumption and is driven by behaviour, not solar generation. A
    high positive correlation between gross_load and pv_site_net during solar
    hours indicates partial phase metering — where ac_load_net captures only a
    fraction of actual load (e.g. one of three phases), causing gross_load to
    track pv_site_net rather than true consumption. Sites exceeding
    GROSS_PV_CORR_THRESHOLD are excluded. The threshold is set to pass sites
    with solar-correlated flexible loads (e.g. EV chargers, export-limited
    inverters) where correlation is incidental and bounded.

Manual exclusions are read from manual_exclusions.csv and applied after
programmatic checks. A site/year combination in that file is always excluded
regardless of check results.

Outputs: data/processed/edp/YYYY_consistency_checks.csv  — raw check results for all sites
         data/processed/edp/YYYY_selected_sites_clean.csv — sites passing all checks
         data/processed/edp/YYYY_selected_sites_excluded.csv — sites failing one or more checks
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw" / "edp"
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "edp"

# ── Year parameter — change this to rerun for a different year ─────────────────
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2022

# ── Derived file paths ─────────────────────────────────────────────────────────
selected_sites_file = PROCESSED_DIR / f"{YEAR}_selected_sites.csv"
output_file         = PROCESSED_DIR / f"{YEAR}_consistency_checks.csv"
clean_file          = PROCESSED_DIR / f"{YEAR}_selected_sites_clean.csv"
excluded_file       = PROCESSED_DIR / f"{YEAR}_selected_sites_excluded.csv"
exclusions_file     = RAW_DIR / "manual_exclusions.csv"

# ── Parameters (processing mechanics) ─────────────────────────────────────────
SOLAR_HOUR_START   = 9          # inclusive — local time hour to start solar window
SOLAR_HOUR_END     = 16         # exclusive — local time hour to end solar window
EXPORT_NOISE_WH    = 50         # export excess below this (Wh) treated as noise/rounding
FLATLINE_NOISE_WH  = 2          # values below this (Wh) treated as zero/flatline
                                # for Check 3 (solar hours ac_load_net) — 2 Wh chosen
                                # to allow flexible loads (e.g. EV solar-following)
                                # which leave residuals of 5–15 Wh, while catching
                                # stuck sensors which report ~0 Wh exactly
CHUNK_SIZE         = 500_000
USECOLS = ['edp_site_id', 'unix_time', 'datetime',
           'circuit_label', 'real_energy']

# ── Pass/fail thresholds ───────────────────────────────────────────────────────
CORR_THRESHOLD          = -0.2      # fail if corr_coefficient > this (less negative)
MIN_PV_THRESHOLD_WH     = 100       # minimum pv_site_net (Wh) to qualify for Check 1
                                    # (~1.2 kW over 5 min — filters low/cloudy periods)
EXPORT_COUNT_THRESHOLD  = 10        # fail if export_excess_intervals >= this
                                    # note: export_excess_max_w retained for information
                                    # but not used as a disqualifying criterion
GROSS_NEG_THRESHOLD     = 0.10      # fail if >10% of intervals have gross_load < 0
                                    # catches partial phase metering (only 1 of 3
                                    # ac_load_net phases metered while PV fully measured)
FLATLINE_THRESHOLD      = 0.05      # fail if flatline fraction > this (Checks 3 & 4)
MIN_ANNUAL_GENERATION_WH    = 500_000  # fail if total annual pv_site_net < this (Check 5)
                                    # 500 kWh — conservative floor for a functional system
MIN_AVG_DAILY_GROSS_LOAD_WH = 5_000   # fail if avg daily gross load < 5 kWh (Check 5)
                                    # catches empty/unoccupied sites with no real load
PV_NEGATIVE_WH          = -20.0    # pv below this (Wh) counted as negative for Check 6
                                    # -20 Wh = ~-240W — consistent with Flag 5 noise floor
PV_NEGATIVE_THRESHOLD   = 0.01     # fail if fraction of daytime pv intervals below
                                    # PV_NEGATIVE_WH exceeds this (Check 6)
                                    # 1% threshold — reversed polarity faults produce
                                    # persistent negative PV throughout the day, well
                                    # above this floor; genuine noise stays below it
GROSS_PV_CORR_THRESHOLD = 0.475    # fail if gross_load vs pv_site_net correlation
                                    # during solar hours exceeds this (Check 7)
                                    # catches partial phase metering where gross_load
                                    # tracks PV because ac_load_net ≈ -k*pv_site_net
                                    # threshold of 0.475 set after manual review of 2022
                                    # marginal sites; EV solar-following sites pass
                                    # (W0143=0.518 excluded — boundary reviewed and
                                    # accepted; S0379=0.478 passes as flexible load)

# ── Load selected sites ────────────────────────────────────────────────────────
selected = pd.read_csv(selected_sites_file)
site_set = set(selected['edp_site_id'])
print(f"Year: {YEAR}")
print(f"Running consistency checks for {len(site_set)} selected sites across 12 months.")

# ── Battery reconstruction lookup ───────────────────────────────────────────────
# {edp_site_id: (battery_channel_label, battery_sign_convention)} for sites
# B_Site_Selection.py marked as a usable battery reconstruction. Empty dict
# (and identical behaviour to before this change) if the selected sites file
# predates B0_Battery_Site_Screen.py / doesn't have these columns.
if 'battery_channel_label' in selected.columns:
    battery_lookup = {
        row.edp_site_id: (row.battery_channel_label, row.battery_sign_convention)
        for row in selected.itertuples()
        if pd.notna(row.battery_channel_label)
    }
else:
    battery_lookup = {}

battery_labels_in_use = sorted({label for label, _ in battery_lookup.values()})
if battery_labels_in_use:
    print(f"Battery-corrected sites: {len(battery_lookup)} "
          f"(channels in use: {battery_labels_in_use})")

# ── Load manual exclusions ─────────────────────────────────────────────────────
if exclusions_file.exists():
    exclusions = pd.read_csv(exclusions_file)
    manual_excluded = set(
        exclusions[
            (exclusions['year'].astype(str) == str(YEAR)) |
            (exclusions['year'].astype(str).str.lower() == 'all')
        ]['edp_site_id']
    )
    if manual_excluded:
        print(f"Manual exclusions for {YEAR}: {sorted(manual_excluded)}")
else:
    manual_excluded = set()
    print(f"No manual exclusions file found at {exclusions_file} — skipping.")

# ── Accumulators ───────────────────────────────────────────────────────────────
# Check 1: {site: {month: {'pv': [], 'load': []}}}
corr_data = defaultdict(lambda: defaultdict(lambda: {'pv': [], 'load': []}))

# Check 2: {site: {month: {'count': int, 'max_wh': float}}}
export_data = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'max_wh': 0.0}))

# Check 2b: {site: {'gross_negative_count': int, 'total_count': int}}
# Counts all intervals where gross_load < 0 (any amount, including noise)
# Used to detect partial phase metering where ac_load_net < pv_site_net
gross_negative = defaultdict(lambda: {'gross_negative_count': 0, 'total_count': 0})

# Check 3: {site: {'zero_count': int, 'total_count': int}}
flatline_ac = defaultdict(lambda: {'zero_count': 0, 'total_count': 0})

# Check 4: {site: {'zero_count': int, 'total_count': int}} — daytime only
flatline_pv = defaultdict(lambda: {'zero_count': 0, 'total_count': 0})

# Check 5: {site: total_wh}
annual_generation = defaultdict(float)

# Check 5b: {site: {'gross_load_wh': float, 'complete_days': int}}
# Tracks average daily gross load for minimum load check
avg_gross_load = defaultdict(lambda: {'gross_load_wh': 0.0, 'intervals': 0})

# Check 6: {site: {'negative_count': int, 'total_count': int}} — daytime only
pv_daytime_negative = defaultdict(lambda: {'negative_count': 0, 'total_count': 0})

# Check 7: {site: {'pv': list, 'gross': list}} — solar hours
# Correlation between gross_load and pv_site_net during solar hours.
# High positive correlation indicates partial phase metering — gross_load
# tracks PV because ac_load_net ≈ -k*pv_site_net (missing load phases).
gross_pv_corr_data = defaultdict(lambda: {'pv': [], 'gross': []})

# ── Main loop ──────────────────────────────────────────────────────────────────
for month in range(1, 13):
    file_path = RAW_DIR / f"edp_data_{YEAR}_{month:02d}.csv"
    print(f"Processing: {file_path.name}")

    for chunk in pd.read_csv(file_path, usecols=USECOLS, chunksize=CHUNK_SIZE):

        # Strip whitespace from circuit_label before filtering
        chunk['circuit_label'] = chunk['circuit_label'].str.strip()

        # Filter to selected sites and relevant labels only (core labels
        # plus any battery-labelled channels in use by battery-corrected sites)
        chunk = chunk[
            chunk['edp_site_id'].isin(site_set) &
            chunk['circuit_label'].isin(['ac_load_net', 'pv_site_net'] + battery_labels_in_use)
        ].copy()

        if chunk.empty:
            continue

        # Parse datetime and extract hour
        chunk['datetime'] = pd.to_datetime(chunk['datetime'])
        chunk['hour'] = chunk['datetime'].dt.hour

        # Sum across phases per (site, timestamp, label)
        agg = (chunk.groupby(['edp_site_id', 'unix_time', 'hour', 'circuit_label'])
                    ['real_energy'].sum().reset_index())

        # Pivot to wide: one row per (site, timestamp) with pv, load, and
        # (where relevant) battery channel columns
        wide = agg.pivot_table(
            index=['edp_site_id', 'unix_time', 'hour'],
            columns='circuit_label',
            values='real_energy'
        ).reset_index()
        wide.columns.name = None

        # Ensure all label columns exist — a chunk may contain only some of them
        for col in ['ac_load_net', 'pv_site_net'] + battery_labels_in_use:
            if col not in wide.columns:
                wide[col] = np.nan

        # ── Battery correction: counterfactual "no battery" net site flow ─────
        # ac_load_net_corrected == ac_load_net for every site with no battery
        # reconstruction. See docstring for the Check 3 exception and the
        # missing-interval fallback behaviour.
        wide['ac_load_net_corrected'] = wide['ac_load_net']
        for label in battery_labels_in_use:
            subtract_sites = {s for s, (l, conv) in battery_lookup.items()
                               if l == label and conv == 'subtract'}
            add_sites = {s for s, (l, conv) in battery_lookup.items()
                         if l == label and conv == 'add'}
            mask_sub = wide['edp_site_id'].isin(subtract_sites) & wide[label].notna()
            mask_add = wide['edp_site_id'].isin(add_sites) & wide[label].notna()
            wide.loc[mask_sub, 'ac_load_net_corrected'] = (
                wide.loc[mask_sub, 'ac_load_net'] - wide.loc[mask_sub, label]
            )
            wide.loc[mask_add, 'ac_load_net_corrected'] = (
                wide.loc[mask_add, 'ac_load_net'] + wide.loc[mask_add, label]
            )

        # ── Check 3: ac_load_net flatline — solar hours only ─────────────────
        # During solar hours PV drives variation in ac_load_net even for
        # empty houses, so near-zero ac_load_net is suspicious. Limiting to
        # solar hours avoids penalising sites with genuine low overnight load.
        # Numerator and denominator both use solar hours only.
        ac_rows = wide[
            wide['ac_load_net'].notna() &
            (wide['hour'] >= SOLAR_HOUR_START) &
            (wide['hour'] < SOLAR_HOUR_END)
        ]
        for site, group in ac_rows.groupby('edp_site_id'):
            flatline_ac[site]['total_count'] += len(group)
            flatline_ac[site]['zero_count'] += (
                group['ac_load_net'].abs() <= FLATLINE_NOISE_WH
            ).sum()

        # ── Check 4: pv_site_net flatline — solar hours only ──────────────────
        pv_rows = wide[
            wide['pv_site_net'].notna() &
            (wide['hour'] >= SOLAR_HOUR_START) &
            (wide['hour'] < SOLAR_HOUR_END)
        ]
        for site, group in pv_rows.groupby('edp_site_id'):
            flatline_pv[site]['total_count'] += len(group)
            flatline_pv[site]['zero_count'] += (
                group['pv_site_net'].abs() <= FLATLINE_NOISE_WH
            ).sum()

        # ── Check 5: annual generation — accumulate pv_site_net ───────────────
        pv_all = wide[wide['pv_site_net'].notna()]
        for site, group in pv_all.groupby('edp_site_id'):
            annual_generation[site] += group['pv_site_net'].sum()

        # ── Check 6: daytime negative pv_site_net ─────────────────────────────
        pv_daytime = wide[
            wide['pv_site_net'].notna() &
            (wide['hour'] >= SOLAR_HOUR_START) &
            (wide['hour'] < SOLAR_HOUR_END)
        ]
        for site, group in pv_daytime.groupby('edp_site_id'):
            pv_daytime_negative[site]['total_count'] += len(group)
            pv_daytime_negative[site]['negative_count'] += (
                group['pv_site_net'] < PV_NEGATIVE_WH
            ).sum()

        # ── Check 7: gross_load vs pv correlation during solar hours ───────────
        # Accumulate pv_site_net and gross_load for solar hours where both present
        # and pv_site_net > 0 (genuine generation periods only)
        solar_both = wide[
            (wide['hour'] >= SOLAR_HOUR_START) &
            (wide['hour'] < SOLAR_HOUR_END) &
            (wide['pv_site_net'] > 0) &
            wide['ac_load_net_corrected'].notna()
        ].copy()
        solar_both['gross_load'] = solar_both['ac_load_net_corrected'] + solar_both['pv_site_net']

        for site, group in solar_both.groupby('edp_site_id'):
            gross_pv_corr_data[site]['pv'].extend(group['pv_site_net'].tolist())
            gross_pv_corr_data[site]['gross'].extend(group['gross_load'].tolist())

        # Only process rows where both (corrected) labels are present for Checks 1 & 2
        wide = wide.dropna(subset=['ac_load_net_corrected', 'pv_site_net'])

        if wide.empty:
            continue

        # ── Check 1: correlation during solar hours ────────────────────────────
        solar = wide[
            (wide['hour'] >= SOLAR_HOUR_START) &
            (wide['hour'] < SOLAR_HOUR_END) &
            (wide['pv_site_net'] >= MIN_PV_THRESHOLD_WH)
        ]

        for site, group in solar.groupby('edp_site_id'):
            corr_data[site][month]['pv'].extend(group['pv_site_net'].tolist())
            corr_data[site][month]['load'].extend(group['ac_load_net_corrected'].tolist())

        # ── Check 2: export excess ─────────────────────────────────────────────
        exporting = wide[wide['ac_load_net_corrected'] < 0].copy()
        exporting['export_excess_wh'] = (-exporting['ac_load_net_corrected']
                                          - exporting['pv_site_net'])
        excess = exporting[exporting['export_excess_wh'] > EXPORT_NOISE_WH]

        for site, group in excess.groupby('edp_site_id'):
            export_data[site][month]['count'] += len(group)
            export_data[site][month]['max_wh'] = max(
                export_data[site][month]['max_wh'],
                group['export_excess_wh'].max()
            )

        # ── Check 2b: gross load < 0 fraction ─────────────────────────────────
        # Catches partial phase metering — persistent negative gross load
        # even when export excess (Check 2) is not triggered
        for site, group in wide.groupby('edp_site_id'):
            gross_negative[site]['total_count'] += len(group)
            gross_negative[site]['gross_negative_count'] += (
                (group['ac_load_net_corrected'] + group['pv_site_net']) < 0
            ).sum()

        # ── Check 5b: average daily gross load ─────────────────────────────────
        for site, group in wide.groupby('edp_site_id'):
            avg_gross_load[site]['gross_load_wh'] += (
                group['ac_load_net_corrected'] + group['pv_site_net']
            ).sum()
            avg_gross_load[site]['intervals'] += len(group)

print("\nComputing summary statistics...")

# ── Compute per-site summary ───────────────────────────────────────────────────
rows = []
for site in sorted(site_set):

    # -- Check 1: correlation --------------------------------------------------
    month_corrs = {}
    all_pv, all_load = [], []
    for month in range(1, 13):
        pv = corr_data[site][month]['pv']
        load = corr_data[site][month]['load']
        if len(pv) >= 10:
            month_corrs[month] = np.corrcoef(pv, load)[0, 1]
            all_pv.extend(pv)
            all_load.extend(load)

    if len(all_pv) >= 10:
        overall_corr = np.corrcoef(all_pv, all_load)[0, 1]
    else:
        overall_corr = np.nan

    if month_corrs:
        worst_corr_month = max(month_corrs, key=month_corrs.get)
        worst_corr_value = month_corrs[worst_corr_month]
    else:
        worst_corr_month = np.nan
        worst_corr_value = np.nan

    # -- Check 2: export excess ------------------------------------------------
    total_excess_intervals = sum(
        export_data[site][m]['count'] for m in range(1, 13)
    )
    max_excess_wh = max(
        (export_data[site][m]['max_wh'] for m in range(1, 13)),
        default=0.0
    )
    max_excess_w = round(max_excess_wh * 12, 1)

    month_excess_counts = {m: export_data[site][m]['count']
                           for m in range(1, 13)
                           if export_data[site][m]['count'] > 0}
    worst_export_month = (max(month_excess_counts, key=month_excess_counts.get)
                          if month_excess_counts else np.nan)

    # -- Check 3: ac_load_net flatline -----------------------------------------
    ac_total = flatline_ac[site]['total_count']
    ac_zero  = flatline_ac[site]['zero_count']
    ac_flatline_fraction = round(ac_zero / ac_total, 4) if ac_total > 0 else np.nan

    # -- Check 4: pv_site_net flatline (solar hours only) ----------------------
    pv_total = flatline_pv[site]['total_count']
    pv_zero  = flatline_pv[site]['zero_count']
    pv_flatline_fraction = round(pv_zero / pv_total, 4) if pv_total > 0 else np.nan

    # -- Check 2b: gross load < 0 fraction -------------------------------------
    gn_total   = gross_negative[site]['total_count']
    gn_count   = gross_negative[site]['gross_negative_count']
    gross_neg_fraction = round(gn_count / gn_total, 4) if gn_total > 0 else 0.0

    # -- Check 5: annual generation --------------------------------------------
    total_generation_wh = annual_generation.get(site, 0.0)
    total_generation_kwh = round(total_generation_wh / 1000, 1)

    # -- Check 5b: average daily gross load ------------------------------------
    gl = avg_gross_load[site]
    avg_daily_gross_load_wh = round(
        gl['gross_load_wh'] / gl['intervals'] * 288, 1
    ) if gl['intervals'] > 0 else 0.0

    # -- Check 6: daytime negative PV ------------------------------------------
    pv_day_total   = pv_daytime_negative[site]['total_count']
    pv_day_neg     = pv_daytime_negative[site]['negative_count']
    pv_neg_fraction = round(pv_day_neg / pv_day_total, 4) if pv_day_total > 0 else np.nan

    # -- Check 7: gross_load vs pv_site_net correlation during solar hours ------
    gpc = gross_pv_corr_data[site]
    if len(gpc['pv']) >= 100:
        gross_pv_corr = round(float(np.corrcoef(gpc['pv'], gpc['gross'])[0, 1]), 3)
    else:
        gross_pv_corr = np.nan

    rows.append({
        'edp_site_id': site,
        # Check 1
        'corr_coefficient': round(overall_corr, 3) if not np.isnan(overall_corr) else np.nan,
        'corr_worst_month': worst_corr_month,
        'corr_worst_month_value': round(worst_corr_value, 3) if not np.isnan(worst_corr_value) else np.nan,
        # Check 2
        'export_excess_intervals': total_excess_intervals,
        'export_excess_max_w': max_excess_w,
        'export_excess_worst_month': worst_export_month,
        'gross_neg_fraction': gross_neg_fraction,
        # Check 3
        'ac_flatline_fraction': ac_flatline_fraction,
        # Check 4
        'pv_flatline_fraction': pv_flatline_fraction,
        # Check 5
        'annual_generation_kwh': total_generation_kwh,
        'avg_daily_gross_load_wh': avg_daily_gross_load_wh,
        # Check 6
        'pv_daytime_negative_fraction': pv_neg_fraction,
        # Check 7
        'gross_pv_corr': gross_pv_corr,
    })

results_df = pd.DataFrame(rows)

# ── Apply pass/fail thresholds ─────────────────────────────────────────────────
results_df['corr_pass']         = results_df['corr_coefficient'] <= CORR_THRESHOLD
results_df['export_pass']       = (
    (results_df['export_excess_intervals'] < EXPORT_COUNT_THRESHOLD) &
    (results_df['gross_neg_fraction'] < 0.10)
)
results_df['ac_flatline_pass']  = results_df['ac_flatline_fraction'] <= FLATLINE_THRESHOLD
results_df['pv_flatline_pass']  = results_df['pv_flatline_fraction'] <= FLATLINE_THRESHOLD
results_df['generation_pass']   = (
    (results_df['annual_generation_kwh'] >= (MIN_ANNUAL_GENERATION_WH / 1000)) &
    (results_df['avg_daily_gross_load_wh'] >= MIN_AVG_DAILY_GROSS_LOAD_WH)
)
results_df['pv_negative_pass']  = results_df['pv_daytime_negative_fraction'] <= PV_NEGATIVE_THRESHOLD
results_df['gross_pv_pass']     = (
    results_df['gross_pv_corr'].isna() |
    (results_df['gross_pv_corr'] < GROSS_PV_CORR_THRESHOLD)
)
results_df['overall_pass']      = (results_df['corr_pass'] &
                                    results_df['export_pass'] &
                                    results_df['ac_flatline_pass'] &
                                    results_df['pv_flatline_pass'] &
                                    results_df['generation_pass'] &
                                    results_df['pv_negative_pass'] &
                                    results_df['gross_pv_pass'])

# ── Save raw consistency results ───────────────────────────────────────────────
results_df.to_csv(output_file, index=False)

# ── Merge onto selected sites ──────────────────────────────────────────────────
selected_full = pd.read_csv(selected_sites_file)
merged = selected_full.merge(results_df, on='edp_site_id', how='left')
merged['overall_pass'] = merged['overall_pass'].fillna(False)
merged['manual_exclusion'] = merged['edp_site_id'].isin(manual_excluded)

# Manual exclusions always excluded regardless of check results
merged.loc[merged['manual_exclusion'], 'overall_pass'] = False

# ── Split and save ─────────────────────────────────────────────────────────────
clean_df    = merged[merged['overall_pass']].copy()
excluded_df = merged[~merged['overall_pass']].copy()

clean_df.to_csv(clean_file, index=False)
excluded_df.to_csv(excluded_file, index=False)

# ── Summary ────────────────────────────────────────────────────────────────────
n_manual     = merged['manual_exclusion'].sum()
n_corr       = (~results_df['corr_pass']).sum()
n_export     = (~results_df['export_pass']).sum()
n_ac_flat    = (~results_df['ac_flatline_pass']).sum()
n_pv_flat    = (~results_df['pv_flatline_pass']).sum()
n_generation = (~results_df['generation_pass']).sum()
n_pv_neg     = (~results_df['pv_negative_pass']).sum()
n_gross_pv   = (~results_df['gross_pv_pass']).sum()

only_corr    = (~results_df['corr_pass'] & results_df['export_pass'] & results_df['ac_flatline_pass'] & results_df['pv_flatline_pass'] & results_df['generation_pass'] & results_df['pv_negative_pass'] & results_df['gross_pv_pass']).sum()
only_export  = (results_df['corr_pass'] & ~results_df['export_pass'] & results_df['ac_flatline_pass'] & results_df['pv_flatline_pass'] & results_df['generation_pass'] & results_df['pv_negative_pass'] & results_df['gross_pv_pass']).sum()
only_ac_flat = (results_df['corr_pass'] & results_df['export_pass'] & ~results_df['ac_flatline_pass'] & results_df['pv_flatline_pass'] & results_df['generation_pass'] & results_df['pv_negative_pass'] & results_df['gross_pv_pass']).sum()
only_pv_flat = (results_df['corr_pass'] & results_df['export_pass'] & results_df['ac_flatline_pass'] & ~results_df['pv_flatline_pass'] & results_df['generation_pass'] & results_df['pv_negative_pass'] & results_df['gross_pv_pass']).sum()
only_gen     = (results_df['corr_pass'] & results_df['export_pass'] & results_df['ac_flatline_pass'] & results_df['pv_flatline_pass'] & ~results_df['generation_pass'] & results_df['pv_negative_pass'] & results_df['gross_pv_pass']).sum()
only_pv_neg  = (results_df['corr_pass'] & results_df['export_pass'] & results_df['ac_flatline_pass'] & results_df['pv_flatline_pass'] & results_df['generation_pass'] & ~results_df['pv_negative_pass'] & results_df['gross_pv_pass']).sum()
only_gross_pv= (results_df['corr_pass'] & results_df['export_pass'] & results_df['ac_flatline_pass'] & results_df['pv_flatline_pass'] & results_df['generation_pass'] & results_df['pv_negative_pass'] & ~results_df['gross_pv_pass']).sum()
multi        = (~results_df['overall_pass']).sum() - only_corr - only_export - only_ac_flat - only_pv_flat - only_gen - only_pv_neg - only_gross_pv

print(f"\nThresholds applied:")
print(f"  Check 1 (correlation):          fail if corr_coefficient > {CORR_THRESHOLD}")
print(f"  Check 2 (export excess):        fail if export_excess_intervals >= {EXPORT_COUNT_THRESHOLD}, or gross_load < 0 fraction > {GROSS_NEG_THRESHOLD:.0%}")
print(f"  Check 3 (ac_load_net flatline): fail if ac_flatline_fraction > {FLATLINE_THRESHOLD:.0%} (solar hours only, {FLATLINE_NOISE_WH} Wh threshold)")
print(f"  Check 4 (pv_site_net flatline): fail if pv_flatline_fraction > {FLATLINE_THRESHOLD:.0%} (solar hours only)")
print(f"  Check 5 (annual generation):    fail if annual_generation_kwh < {MIN_ANNUAL_GENERATION_WH/1000:.0f} kWh, or avg_daily_gross_load < {MIN_AVG_DAILY_GROSS_LOAD_WH/1000:.0f} kWh")
print(f"  Check 6 (daytime negative PV):  fail if pv_daytime_negative_fraction > {PV_NEGATIVE_THRESHOLD:.0%} (solar hours only)")
print(f"  Check 7 (gross/pv correlation): fail if gross_load vs pv_site_net correlation > {GROSS_PV_CORR_THRESHOLD} (solar hours)")
print(f"  Max excess magnitude:           retained for information only — not used to exclude")
print(f"  Manual exclusions:              applied from {exclusions_file.name}")
print(f"\nIndividual check failures (each independently):")
print(f"  Check 1 (correlation):          {n_corr:>4}")
print(f"  Check 2 (export excess):        {n_export:>4}")
print(f"  Check 3 (ac_load_net flatline): {n_ac_flat:>4}")
print(f"  Check 4 (pv_site_net flatline): {n_pv_flat:>4}")
print(f"  Check 5 (annual generation):    {n_generation:>4}")
print(f"  Check 6 (daytime negative PV):  {n_pv_neg:>4}")
print(f"  Check 7 (gross/pv correlation): {n_gross_pv:>4}")
print(f"\nResults:")
print(f"  Total sites:                    {len(merged)}")
print(f"  Passing (clean):                {len(clean_df)}")
print(f"  Excluded:                       {len(excluded_df)}")
print(f"    Failed Check 1 only:          {only_corr}")
print(f"    Failed Check 2 only:          {only_export}")
print(f"    Failed Check 3 only:          {only_ac_flat}")
print(f"    Failed Check 4 only:          {only_pv_flat}")
print(f"    Failed Check 5 only:          {only_gen}")
print(f"    Failed Check 6 only:          {only_pv_neg}")
print(f"    Failed Check 7 only:          {only_gross_pv}")
print(f"    Failed multiple checks:       {multi}")
print(f"    Manual exclusions:            {n_manual}")
print(f"\nSaved:")
print(f"  {output_file}")
print(f"  {clean_file}")
print(f"  {excluded_file}")