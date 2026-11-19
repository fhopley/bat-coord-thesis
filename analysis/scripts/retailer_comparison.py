#!/usr/bin/env python3
"""Retailer comparison chart — coordination outcomes for several retailers on
one pair of axes, measured from one zero point.

Purpose
-------
retailer_briefing.py draws one retailer's coordination result per figure, each
measured against that retailer's own solar-only run. This script draws two or
more retailers together in a single two-panel figure:

    left   household bill saving by battery size
    right  net wholesale saving by battery size

Each retailer takes its colour; the coordination modes are the same
encodings as in the briefing charts and are keyed once, in grey. Perfect
foresight is the dashed line, imperfect foresight the range between the two
rolling runs, and self consumption the solid line, as in the briefings.

Retailers are the coordination briefings registered in retailer_briefing.py
(currently amber and tesla). Adding one there makes it available here.

Baseline
--------
Every series is measured against Amber's solar-only run
(financial_analysis.DEFAULT_BASELINE_RUN), the 'amber' baseline mode of the
scenario builder, so the retailers share one zero point. Each curve starts at
the saving of its own plan with no battery, against Amber solar only: $0 for
Amber and, for a retailer on a different tariff, the household bill
difference between that tariff and Amber's on the same solar-only flows. The
net wholesale saving at zero capacity is $0 for all, since a solar-only site
has the same wholesale flows on every plan.

Because a battery dispatches identically under the Amber and Tesla self
consumption runs, their net wholesale savings are the same curve, and one
line is drawn over the other.

Amber naming
------------
The Amber runs carry Amber's own tariff and also the constructed alternative
coordination offerings built on the same runs, so the Amber series is named
and coloured as the alternatives by default. --alt-name Amber restores
Amber's own name and colour for a plain Amber against Tesla figure.

Usage
-----
    python3 analysis/scripts/retailer_comparison.py --compare amber tesla
    python3 analysis/scripts/retailer_comparison.py --compare amber tesla \
        --alt-name Amber
    python3 analysis/scripts/retailer_comparison.py --compare amber tesla \
        --alt-name "Amber & capped import price alternatives" \
        --alt-colour '#8e3fb5' --no-notes --data-root "/Volumes/Manual/Thesis Project"

Writes to analysis/output/retailer_briefings/comparison/<keys joined by _vs_>/
by default, two figures per scope: coordination_comparison_<weighting>.png with
perfect foresight, and coordination_comparison_no_perfect_foresight_<weighting>.png
without it. Figure format follows docs/thesis_figure_spec.md through
thesis_style. Data notes are drawn in the bottom right of each panel for
presentation use and left off with --no-notes, for the thesis, where they go
in the caption.
"""

import argparse
import colorsys
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

import financial_analysis as fa
import retailer_briefing as rb
import thesis_style as ts


# Name and colour of the Amber series when it stands for Amber and the
# constructed alternative offerings. The colour is not in the retailer
# registry (Amber, Tesla, AGL, GloBird, Origin, EnergyAustralia) and is not
# grey, which is the colour of the keys common to every retailer.
ALT_NAME = 'Amber & capped import price alternatives'
ALT_COLOUR = '#8e3fb5'

# The step darker used for marker edges, as AMBER_INK is to AMBER.
INK_LIGHTNESS = 0.7

# The figures drawn: (file name suffix, series labels left off). Both are
# written on every run so that either can be chosen.
VARIANTS = (('', frozenset()),
            ('_no_perfect_foresight', frozenset({'Perfect foresight'})))

PANELS = (('hh_saving', 'Household bill saving'),
          ('ws_saving', 'Net wholesale saving'))
YLABEL = r'Mean annual saving (\$/site)'

COMPARABLE = sorted(k for k, b in rb.BRIEFINGS.items()
                    if b['subject'] == 'coordination mode')

FIGURE_CHOICES = ['national', 'state_all_years', 'state_year']
OUTPUT_ROOT = Path('analysis/output/retailer_briefings/comparison')
BASELINE_NOTE = 'Savings vs Amber solar only'
NOTE_WIDTH_CHARS = 36
# More legend entries than this do not fit in one row at the figure width.
MAX_ONE_ROW_ENTRIES = 4
KEY_GREY = ts.REFERENCE_COLOUR
KEY_GREY_EDGE = '0.15'


def _ink(colour: str) -> str:
    r, g, b = mcolors.to_rgb(colour)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return mcolors.to_hex(colorsys.hls_to_rgb(h, l * INK_LIGHTNESS, s))


def _is_amber_name(name: str) -> bool:
    return name.strip().lower() == 'amber'


def _entries(keys: list[str], alt_name: str | None, alt_colour: str | None,
             dropped: frozenset = frozenset()) -> list[dict]:
    """One entry per compared retailer: name, colour, ink and the briefing's
    series, recoloured, without the dropped series."""
    entries = []
    for key in keys:
        briefing = rb.BRIEFINGS[key]
        lead = briefing['series'][0]
        name, colour, ink = (briefing['display_name'], lead['color'],
                             lead['marker_edge'])
        if key == 'amber':
            if alt_name is None or not _is_amber_name(alt_name):
                name = alt_name or ALT_NAME
                colour = alt_colour or ALT_COLOUR
                ink = _ink(colour)
            else:
                name = 'Amber'
                if alt_colour:
                    colour, ink = alt_colour, _ink(alt_colour)
        series = []
        for spec in briefing['series']:
            if spec['label'] in dropped:
                continue
            s = dict(spec)
            s['color'], s['marker_edge'] = colour, ink
            series.append(s)
        entries.append(dict(key=key, name=name, colour=colour, series=series,
                            briefing=briefing))
    return entries


def _legend(entries: list[dict]):
    """Handles and labels: the grey keys for what every retailer shares,
    then one entry per retailer colour."""
    handles, labels = [], []
    for spec in entries[0]['series']:
        if spec['kind'] == 'band':
            handles.append(rb._RangeKey(KEY_GREY,
                                        both_edges=bool(spec.get('stroke_lower')),
                                        lw=ts.DATA_LW))
        else:
            handles.append(mlines.Line2D(
                [], [], color=KEY_GREY, lw=ts.DATA_LW, ls=spec['linestyle'],
                marker=spec.get('marker', rb.MARKER), ms=ts.MARKER_SIZE,
                markeredgecolor=KEY_GREY_EDGE,
                markeredgewidth=ts.MARKER_EDGE_WIDTH))
        labels.append(spec['label'])
    for e in entries:
        handles.append(mlines.Line2D([], [], color=e['colour'], lw=4.0,
                                     solid_capstyle='butt'))
        labels.append(e['name'])
    return handles, labels


def _two_rows(handles: list, labels: list[str], n_retailers: int):
    """Reorder the legend into two rows: the grey keys on the first, the
    retailers on the second, each row left to right.

    Matplotlib fills legend columns top to bottom, so the entries are laid
    out column by column, with blank entries padding the shorter row. With
    all keys and retailers on one row the legend is wider than the figure.
    """
    n_keys = len(handles) - n_retailers
    ncol = max(n_keys, n_retailers)
    blank = mlines.Line2D([], [], alpha=0)
    keys = list(zip(handles[:n_keys], labels[:n_keys]))
    rets = list(zip(handles[n_keys:], labels[n_keys:]))
    keys += [(blank, '')] * (ncol - n_keys)
    rets += [(blank, '')] * (ncol - n_retailers)
    ordered = [e for pair in zip(keys, rets) for e in pair]
    return [h for h, _ in ordered], [l for _, l in ordered], ncol


def make_comparison_figure(curve_set: dict, entries: list[dict], title: str,
                           notes: str | None):
    """Two panels on their own y ranges, under one title, one y label and one
    legend. notes, where given, go in the bottom right of each panel."""
    fig = ts.new_figure(height_in=3.3)
    axes = fig.subplots(1, len(PANELS))
    for ax, (metric, subtitle) in zip(axes, PANELS):
        curves = curve_set[metric]
        for e in entries:
            rb._draw_series(ax, curves, e['series'], 0.0)
        ax.set_title(subtitle, fontsize=ts.SUBTITLE_SIZE)
        ax.set_xlabel('Battery capacity (kWh)')
        if notes:
            rb._draw_notes(ax, notes, 'upper left')
    fig.supylabel(YLABEL, fontsize=matplotlib.rcParams['axes.labelsize'])
    handles, labels = _legend(entries)
    ncol = len(handles)
    if len(handles) > MAX_ONE_ROW_ENTRIES:
        handles, labels, ncol = _two_rows(handles, labels, len(entries))
    fig.legend(handles=handles, labels=labels, loc='outside lower center',
               ncol=ncol, frameon=False, handleheight=1.5, handlelength=2.8,
               handler_map={rb._RangeKey: rb._RangeKeyHandler()})
    ts.panel_suptitle(fig, title)
    return fig


def _write(curve_set: dict, entry_sets: list[tuple[str, list[dict]]],
           title: str, out_dir: Path, weight_tag: str, info: str,
           range_note: str | None, draw_notes: bool) -> None:
    """One figure per (suffix, entries) in entry_sets, in out_dir."""
    notes = [n for n in (info, BASELINE_NOTE, range_note) if n]
    # Wrapped so the box in the corner of each panel stays narrow.
    wrapped = '\n'.join(textwrap.fill(line, NOTE_WIDTH_CHARS)
                        for n in notes for line in n.split('\n'))
    for suffix, entries in entry_sets:
        fig = make_comparison_figure(
            curve_set, entries, title, wrapped if draw_notes else None)
        ts.save_figure(
            fig, out_dir / f'coordination_comparison{suffix}_{weight_tag}')
    print('    ' + '; '.join(n.replace('\n', ', ') for n in notes))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--compare', nargs='+', default=['amber', 'tesla'],
                        type=str.lower, choices=COMPARABLE, metavar='RETAILER',
                        help=f"Retailers to compare, from {COMPARABLE}, in "
                             f"drawing order (default: amber tesla).")
    parser.add_argument('--alt-name', default=None,
                        help=f"Name of the amber series (default: "
                             f"'{ALT_NAME}'). 'Amber' restores Amber's own "
                             f"name and colour.")
    parser.add_argument('--alt-colour', default=None,
                        help=f"Colour of the amber series (default: "
                             f"{ALT_COLOUR}, or Amber's registry colour with "
                             f"--alt-name Amber).")
    parser.add_argument('--title', default=None,
                        help="Figure title (default: the retailer names "
                             "joined by 'vs', then 'offer by battery size and "
                             "coordination type').")
    parser.add_argument('--data-root', type=Path, default=Path('.'),
                        help='Project root (default: cwd)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory (default: '
                             f'{OUTPUT_ROOT}/<retailers joined by _vs_>)')
    parser.add_argument('--figures', nargs='+', default=['national'],
                        choices=FIGURE_CHOICES,
                        help="'national': pooled state-weighted view. "
                             "'state_all_years': one per state, mean across "
                             "years. 'state_year': one per state-year. "
                             "Default: national.")
    parser.add_argument('--states', nargs='+', default=None,
                        help='Restrict to these states (default: all found)')
    parser.add_argument('--years', nargs='+', default=None,
                        help='Restrict to these years (default: all found)')
    parser.add_argument('--unweighted', action='store_true',
                        help='Disable PV size weighting (no national view)')
    parser.add_argument('--weights-file', type=Path,
                        default=Path('analysis/config/pv_size_bands.yaml'),
                        help='PV size band weights config')
    parser.add_argument('--no-notes', dest='notes', action='store_false',
                        help='Leave the data notes off the chart, for the '
                             'thesis. They are printed either way.')
    args = parser.parse_args(argv)

    keys = list(dict.fromkeys(k.lower() for k in args.compare))
    if len(keys) < 2:
        print('ERROR: --compare needs at least two different retailers.',
              file=sys.stderr)
        return 1
    if args.alt_name and 'amber' not in keys:
        print("NOTE: --alt-name and --alt-colour apply to the amber series, "
              "which is not in --compare, so they have no effect.")

    matplotlib.rcdefaults()
    ts.apply_style()

    entry_sets = [(suffix, _entries(keys, args.alt_name, args.alt_colour,
                                    dropped))
                  for suffix, dropped in VARIANTS]
    entries = entry_sets[0][1]
    title = args.title or (' vs '.join(e['name'] for e in entries)
                           + ' offer by battery size and coordination type')
    figures = set(args.figures)
    weighted = not args.unweighted
    weight_tag = 'weighted' if weighted else 'unweighted'
    output_dir = args.output_dir or (OUTPUT_ROOT / '_vs_'.join(keys))

    plans = [p for e in entries for p in e['briefing']['plans']]
    placeholders = [m for e in entries
                    for m in e['briefing'].get('placeholders', [])]
    modes = [m for p in plans for m in (p['sc_mode'], p['lp_mode'])]
    if len(set(modes)) != len(modes):
        # A run repeated into a spare slot is suffixed __UNUSED, so a
        # repeat here is two briefings sharing a mode key.
        raise KeyError(f'Mode keys are shared between the compared briefings: '
                       f'{sorted({m for m in modes if modes.count(m) > 1})}.')
    _validate_all(entries, plans)

    scenarios = fa.discover_scenarios(args.data_root)
    for e in entries:
        states = e['briefing'].get('states')
        if states:
            scenarios = [(s, y) for s, y in scenarios if s in states]
    if args.states:
        scenarios = [(s, y) for s, y in scenarios if s in args.states]
    if args.years:
        scenarios = [(s, y) for s, y in scenarios if y in args.years]
    if not scenarios:
        print('ERROR: no scenarios match the requested filters.',
              file=sys.stderr)
        return 1

    problems = []
    for e in entries:
        for p in rb._check_runs_exist(args.data_root, e['briefing']['plans'],
                                      e['briefing']['baseline_run'],
                                      scenarios):
            if p not in problems:
                problems.append(p)
    if problems:
        print('ERROR: the following runs this comparison needs were not '
              'found:\n' + '\n'.join(problems), file=sys.stderr)
        return 1
    rb._check_placeholders(args.data_root, placeholders, scenarios)

    fa.install_plans(plans, plans[0]['sc_mode'])

    weights_cfg = None
    if weighted:
        if not args.weights_file.exists():
            print(f'ERROR: weighted mode requires a weights file, none found '
                  f'at {args.weights_file}', file=sys.stderr)
            return 1
        weights_cfg = fa.load_weights_config(args.weights_file)

    print('=' * 72)
    print(f"RETAILER COMPARISON — {', '.join(e['name'] for e in entries)}")
    print(f"  Baseline:    {fa.DEFAULT_BASELINE_RUN} for every series")
    print(f"  Figures:     {', '.join(sorted(figures))}")
    print(f"  Output:      {output_dir}")
    print('=' * 72)

    # 'amber': every plan against Amber's solar-only run, which puts the
    # compared retailers on one zero point.
    build_scenario = fa.make_scenario_builder(args.data_root, weights_cfg,
                                              'amber')
    by_state: dict[str, list[str]] = {}
    for state, year in scenarios:
        by_state.setdefault(state, []).append(year)

    weight_label = 'Weighted by PV size' if weighted else 'Unweighted'
    range_note = next((c.get('range_note') for e in entries
                       for c in e['briefing']['charts']
                       if c.get('range_note')), None)
    per_state_curves: dict[str, dict] = {}

    for state, years in by_state.items():
        year_sets, counts = [], []
        for year in years:
            print(f'\nLoading {state} {year}...')
            built = build_scenario(state, year)
            if built is None:
                continue
            curves = built['curves']
            rb._add_placeholder_curves(curves, placeholders)
            year_sets.append(curves)
            counts.append(built['n_sites'])
            if 'state_year' in figures:
                print(f'  writing {state}_{year}')
                _write(curves, entry_sets, title, output_dir / f'{state}_{year}',
                       weight_tag,
                       f'{state} {year} (n = {built["n_sites"]})\n'
                       f'{weight_label}', range_note, args.notes)
        if not year_sets:
            continue
        state_avg = (fa.average_curves_across_years(year_sets)
                     if len(year_sets) > 1 else year_sets[0])
        per_state_curves[state] = state_avg
        if 'state_all_years' in figures:
            n_label = (f'n = {min(counts)}–{max(counts)} per year'
                       if len(set(counts)) > 1 else f'n = {counts[0]}')
            ints = sorted(int(y) for y in years)
            span = (f'{ints[0]}–{ints[-1]}'
                    if len(ints) > 1 and ints == list(range(ints[0], ints[-1] + 1))
                    else ', '.join(years))
            print(f'  writing {state}_all_years')
            _write(state_avg, entry_sets, title, output_dir / f'{state}_all_years',
                   weight_tag, f'{state}, {span} ({n_label})\n{weight_label}',
                   range_note, args.notes)

    if 'national' in figures:
        if len(per_state_curves) < 2:
            print('\nNOTE: national view needs more than one state; only '
                  f'{sorted(per_state_curves)} loaded, so it was skipped.')
        elif not weighted:
            print('\nNOTE: --unweighted given, so the national view was not '
                  'produced: pooling per-state means needs state_mix.')
        else:
            all_years = sorted({y for ys in by_state.values() for y in ys})
            year_label = (f'{all_years[0]}–{all_years[-1]}'
                          if len(all_years) > 1 else all_years[0])
            print('  writing national')
            national = fa.combine_curves_state_weighted(
                per_state_curves, weights_cfg['state_mix'])
            _write(national, entry_sets, title, output_dir / 'national',
                   weight_tag,
                   f"National ({', '.join(sorted(per_state_curves))}), "
                   f"{year_label}\nWeighted by PV size & state mix",
                   range_note, args.notes)

    print('\nDone.')
    return 0


def _validate_all(entries: list[dict], plans: list[dict]) -> None:
    rb._validate_plan_keys(plans)
    for e in entries:
        rb._validate_series_modes(e['briefing'])


if __name__ == '__main__':
    sys.exit(main())
