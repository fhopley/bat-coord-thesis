"""Impact of the export-order nudge (docs/sensitivity_runs_spec.md, 4).

Origin Battery Starter and EnergyAustralia Battery Ease, conservative
imperfect foresight (the no-VPP rolling runs), NSW 2024, every battery size,
with and without the export-order nudge.

On a flat time-of-use feed-in rate the planner is indifferent between
exporting surplus solar now and holding it to charge later. The nudge breaks
that tie in favour of charging first (docs/rolling_perfect_foresight_lp_advanced_spec.md,
Export-order nudge).

Every arm is on the retailer's no-VPP tariff and none carries a VPP
enrolment payment. retailer_briefing.py draws the same no-nudge run as its
'TOU optimised' line without one, and adds EnergyAustralia's $180 a year
only to the band edge standing for an enrolled but undispatched battery,
which is not a series here. Adding it would lift two of the three curves
against the third and hide where imperfect foresight falls below self
consumption. Self-consumption is unaffected by the nudge and is drawn once
per retailer.

Colours, line styles, series names, the zero-capacity anchor, the piecewise
smoothing and the data note follow retailer_briefing.py, so the two charts
read alike. The two arms that briefing also draws keep its solid line; the
nudge takes a long dash, clear of the briefing's short dash and dotted VPP
edges. Each panel carries its own legend, in that retailer's hues, ordered
highest-saving first.

Outputs, in --output-dir:
  nudge_impact.csv   savings by retailer, battery size and arm
  nudge_impact.png   household bill saving by battery size, Origin and
                     EnergyAustralia panels

Usage (from project root):
    python3 analysis/scripts/sensitivity/nudge.py --data-root "/Volumes/Manual/Thesis Project"
    python3 analysis/scripts/sensitivity/nudge.py --data-root ... --print-values
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sensitivity_common as sc          # noqa: E402
import financial_analysis as fa          # noqa: E402
import thesis_style as ts                # noqa: E402
import matplotlib.lines as mlines        # noqa: E402
import matplotlib.ticker as mticker      # noqa: E402

HEIGHT_IN = 3.0

# Nudge is judged to have limited further impact once its gain falls below
# this share of the self-consumption saving at that battery size.
LIMITED_IMPACT_SHARE = 0.05

# Colours and line styles are retailer_briefing.py's, so a reader moving
# between the two charts reads the same encoding. Dark is TOU optimised and
# mid is self consumption, in each retailer's own hue.
ORIGIN_DARK, ORIGIN_MID = '#a85400', '#f0a030'
EA_DARK, EA_MID = '#197e2b', '#5aae61'
SOLID = '-'
MARKER = 'o'

# The two arms the briefing also draws keep its solid line, so a series that
# is the same series on both charts looks the same on both. The nudge is new
# here and takes a long dash, which is clear of the briefing's short dash for
# perfect foresight, (5, 2.4), and its dotted VPP band edges, (1.6, 1.6).
LONG_DASH = (0, (9, 3))

# No enrolment payment on any arm. Every series here is on the no-VPP tariff,
# and retailer_briefing draws its own 'TOU optimised' line, which is the same
# run as the no-nudge arm, without one: the payment belongs to the band edge
# standing for an enrolled but undispatched battery, which is not drawn here.
RETAILERS = [
    dict(key='origin',
         label='Origin Battery Starter',
         solar=sc.REF_SOLAR_ORIGIN,
         self_cons=sc.REF_SC_ORIGIN,
         no_nudge=sc.REF_ROLLING_ORIGIN,
         nudge='sens_nudge_rolling_origin_lite_novpp',
         dark=ORIGIN_DARK, mid=ORIGIN_MID),
    dict(key='ea',
         label='EnergyAustralia Battery Ease',
         solar=sc.REF_SOLAR_EA,
         self_cons=sc.REF_SC_EA,
         no_nudge=sc.REF_ROLLING_EA,
         nudge='sens_nudge_rolling_ea_novpp',
         dark=EA_DARK, mid=EA_MID),
]

# Arm names follow retailer_briefing.py's series labels, so the same series
# is called the same thing on both charts. They are the values written to the
# CSV as well as the legend text.
SELF_CONSUMPTION = 'Self-consumption'
NO_NUDGE = 'TOU optimised'
WITH_NUDGE = 'TOU optimised + nudge'

# Highest-saving first, matching the vertical stacking on the chart and the
# order retailer_briefing.py lists its series in. Curves are drawn in the
# reverse of this, so the nudge arm, the subject of the chart, ends on top.
ARMS = [WITH_NUDGE, NO_NUDGE, SELF_CONSUMPTION]

YLABEL = r'Mean annual saving (\$/site)'

# (hue key, line style) per arm. The hue key picks the retailer's own colour
# on the panels and a grey on the shared legend.
ARM_ENCODING = {
    SELF_CONSUMPTION: ('mid', SOLID),
    NO_NUDGE: ('dark', SOLID),
    WITH_NUDGE: ('dark', LONG_DASH),
}

def arm_style(arm: str, retailer: dict) -> dict:
    """Line colour and style for one arm, in the retailer's own hue."""
    hue, ls = ARM_ENCODING[arm]
    return dict(color=retailer[hue], ls=ls)


def collect(data_root: Path, state: str, year: str,
            weighted: bool) -> pd.DataFrame:
    runs = sc.runs_root(data_root)
    weight = sc.site_weights(data_root, state, year) if weighted else None

    rows = []
    for r in RETAILERS:
        solar = sc.load_solar_only(runs, r['solar'], state, year)
        for size in sc.BATTERY_SIZES:
            arms = {
                SELF_CONSUMPTION: sc.load_bills(runs, r['self_cons'], size,
                                                state, year),
                NO_NUDGE: sc.load_bills(runs, r['no_nudge'], size,
                                        state, year),
                WITH_NUDGE: sc.load_bills(runs, r['nudge'], size,
                                          state, year),
            }
            if any(v is None for v in arms.values()):
                missing = [k for k, v in arms.items() if v is None]
                print(f'  Skipping {r["label"]} {size} kWh: missing '
                      f'{missing}.', file=sys.stderr)
                continue
            idx = sc.common_index(solar, *arms.values())
            if len(idx) == 0:
                continue

            for arm, bills in arms.items():
                sav = sc.savings(bills.loc[idx], solar.loc[idx])
                rows.append({
                    'retailer': r['label'], 'retailer_key': r['key'],
                    'battery_kwh': size, 'arm': arm,
                    'n_sites': len(idx),
                    'hh_saving': sc.wmean(sav['hh_saving'], weight),
                    'ws_saving': sc.wmean(sav['ws_saving'], weight),
                })

    if not rows:
        raise SystemExit('No retailer and size had a complete run set.')
    df = pd.DataFrame(rows)

    # Nudge gain, and the self-consumption saving it is judged against.
    wide = df.pivot_table(index=['retailer_key', 'retailer', 'battery_kwh'],
                          columns='arm', values='hh_saving')
    gain = (wide[WITH_NUDGE] - wide[NO_NUDGE]).rename('nudge_gain')
    share = (gain / wide[SELF_CONSUMPTION].replace(0, np.nan)
             ).rename('nudge_gain_share_of_sc')
    extra = pd.concat([gain, share], axis=1).reset_index()
    return df.merge(extra, on=['retailer_key', 'retailer', 'battery_kwh'],
                    how='left')


def draw(df: pd.DataFrame, out_path: Path, thesis_dir: Path | None,
         notes: str | None = None) -> None:
    """Household bill saving by battery size, one panel per retailer.

    Encoding follows retailer_briefing.py so the two read alike: each curve
    starts at the zero-capacity anchor, where saving against the retailer's
    own Solar only run is zero by construction, is smoothed between the
    modelled capacities by fa._smooth_xy_piecewise and carries a marker at
    each capacity actually run.
    """
    ts.apply_style()
    fig = ts.new_figure(height_in=3.0)
    axes = fig.subplots(1, 2, sharey=True)

    present = [r for r in RETAILERS if r['label'] in set(df['retailer'])]
    xs_base = np.concatenate([[0], np.array(sc.BATTERY_SIZES, dtype=float)])

    for ax, r in zip(axes, present):
        sub = df[df['retailer'] == r['label']]
        handles = []
        # Drawn lowest-saving first so the nudge arm ends on top, then keyed
        # in ARMS order so the legend reads highest first.
        for zorder, arm in enumerate(reversed(ARMS), start=2):
            rows = sub[sub['arm'] == arm].sort_values('battery_kwh')
            if rows.empty:
                continue
            # The zero-capacity anchor is $0: with no battery every arm is
            # the Solar only bill on that retailer's own offer.
            ys_raw = np.concatenate([[0.0], rows['hh_saving'].to_numpy()])
            style = arm_style(arm, r)
            xs_sm, ys_sm = fa._smooth_xy_piecewise(xs_base, ys_raw)
            ax.plot(xs_sm, ys_sm, lw=ts.DATA_LW, zorder=zorder, **style)
            ax.plot(xs_base, ys_raw, ls='none', marker=MARKER,
                    ms=ts.MARKER_SIZE, color=style['color'],
                    zorder=zorder + 10)
            # No marker on the key: every arm uses the same one, so it
            # states nothing, and it would sit over the gap that identifies
            # the dashed arm.
            handles.append(mlines.Line2D([], [], lw=ts.DATA_LW, label=arm,
                                         **style))

        # The dashed key is meant to read as line, gap, line, which needs
        # dash + gap + dash = 27.3 pt at this line width against a default
        # key of 10.4 pt. 4.4 clears that with a little margin, so the
        # second dash is not clipped by rounding.
        ax.legend(handles=list(reversed(handles)), loc='lower right', ncol=1,
                  handlelength=4.4)
        ax.set_title(r['label'], fontsize=ts.SUBTITLE_SIZE)
        ax.set_xlabel('Battery capacity (kWh)')
        ax.set_xlim(0, max(sc.BATTERY_SIZES))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f'${v:,.0f}' if v >= 0 else f'\u2212${-v:,.0f}'))
        if notes:
            ax.text(0.02, 0.97, notes, transform=ax.transAxes,
                    ha='left', va='top', fontsize=ts.NOTE_SIZE,
                    color=ts.REFERENCE_COLOUR, bbox=ts.LABEL_BOX, zorder=6)

    axes[0].set_ylabel(YLABEL)
    ts.panel_suptitle(fig, 'Household bill savings by battery size, with and '
                           'without the imperfect foresight export-order nudge')
    sc.save_figure(fig, out_path, thesis_dir)


def print_values(df: pd.DataFrame) -> None:
    print('\nValues for the text (NSW 2024, conservative imperfect foresight)')
    print('-' * 78)
    for r in RETAILERS:
        sub = df[df['retailer'] == r['label']]
        if sub.empty:
            continue
        gains = (sub[['battery_kwh', 'nudge_gain', 'nudge_gain_share_of_sc']]
                 .drop_duplicates().sort_values('battery_kwh'))
        print(f'\n{r["label"]} (n = {int(sub["n_sites"].iloc[0])} sites)')
        print(f'  {"size":>5} {"nudge gain":>11} {"share of SC saving":>20}')
        for _, g in gains.iterrows():
            print(f'  {int(g["battery_kwh"]):>3} kWh '
                  f'{g["nudge_gain"]:>+10,.0f} '
                  f'{g["nudge_gain_share_of_sc"]:>19.1%}')
        limited = gains[gains['nudge_gain_share_of_sc'] < LIMITED_IMPACT_SHARE]
        if limited.empty:
            print(f'  The nudge adds more than '
                  f'{LIMITED_IMPACT_SHARE:.0%} of the self-consumption saving '
                  f'at every battery size.')
        else:
            # The first size from which it stays below the threshold.
            sizes = gains['battery_kwh'].tolist()
            below = gains['nudge_gain_share_of_sc'].lt(LIMITED_IMPACT_SHARE).tolist()
            start = None
            for i in range(len(sizes)):
                if all(below[i:]):
                    start = sizes[i]
                    break
            if start is None:
                print(f'  Falls below {LIMITED_IMPACT_SHARE:.0%} at some sizes '
                      f'but not from any size onward.')
            else:
                print(f'  Limited impact from {int(start)} kWh: the gain stays '
                      f'below {LIMITED_IMPACT_SHARE:.0%} of the '
                      f'self-consumption saving at and above that size.')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sc.add_common_args(parser)
    parser.add_argument('--no-notes', action='store_true',
                        help='Leave the data notes off, for the thesis, '
                             'where they belong in the caption.')
    args = parser.parse_args(argv)

    out_dir = args.output_dir or (sc.OUTPUT_ROOT / 'nudge')
    df = collect(args.data_root, args.state, args.year, not args.unweighted)
    sc.write_csv(df, out_dir / 'nudge_impact.csv')

    # Same note as retailer_briefing.py: scope and site count, then the
    # weighting and what the saving is measured against.
    weight_label = 'Unweighted' if args.unweighted else 'Weighted by PV size'
    note = (f'{args.state} {args.year} (n = {int(df["n_sites"].iloc[0])})\n'
            f'{weight_label} vs solar only')
    print(f'  Note: {note.replace(chr(10), ", ")}')
    draw(df, out_dir / 'nudge_impact.png', args.thesis_dir,
         None if args.no_notes else note)
    if args.print_values:
        print_values(df)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
