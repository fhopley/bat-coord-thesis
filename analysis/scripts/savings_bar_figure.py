"""Battery saving at one capacity by offer, under imperfect foresight.

Reads the curves financial_analysis.py writes for the conservative and
achievable run bases (<output-dir>/<basis folder>/curves_<weighting>.csv) and
draws, for one scope and one modelled capacity, a self-consumption and an
optimised bar per offer, in two figures:

  household_saving_by_offer_own_plan  the household bill saving against the
                                      offer's own Solar only bill
  wholesale_saving_by_offer           the net wholesale saving

Self-consumption is the run without VPP participation (the _novpp run where
the offer has a VPP, without its enrolment payment). The optimised bar runs
to the conservative value and a shaded segment continues it to the
achievable value: for the VPP and critical-rate offers no events to maximum
dispatch, for Amber the predispatch-only to the dispatch price & new
predispatch configuration, as the range bands of the line charts. Each bar is
labelled with its top value.

Offers are ordered by the household achievable optimised saving, highest
first, with the conservative value standing in for an offer whose achievable
run is not yet available, and the wholesale figure keeps the same order.

The curves are measured against Solar only on Amber. The value at zero
capacity is each offer's Solar only bill against Amber's, so subtracting it
gives the saving against the offer's own Solar only bill; curve values are
weighted means, so the subtraction commutes with the weighting and pooling.
Wholesale savings are zero at zero capacity on every offer.

It also prints, from the site savings financial_analysis.py writes beside
the curves (site_savings_<weighting>.csv), the weighted share of site-years
in which each offer gives the lowest bill with an optimised battery of that
capacity, and writes those shares to lowest_bill_shares_<kwh>kwh_<scope>_
<weighting>.csv beside the figures.

Usage (from project root, after financial_analysis.py):
    python3 analysis/scripts/savings_bar_figure.py
    python3 analysis/scripts/savings_bar_figure.py --kwh 10
    python3 analysis/scripts/savings_bar_figure.py --scope NSW_all_years
    python3 analysis/scripts/savings_bar_figure.py --no-notes

The curves must come from a run against the default Amber baseline
(financial_analysis.py without --baseline own or --only-plan).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import financial_analysis as fa  # noqa: E402
import thesis_style as ts        # noqa: E402

BAR_WIDTH = 0.34                 # the pair fills 0.68 of the offer spacing
BAR_GAP = 0.04                   # white gap between the pair
HEIGHT_IN = 3.2
# The range segment is the offer colour taken this far towards white, the
# bar equivalent of the line charts' range band.
RANGE_TINT = 0.6
SC_EDGE_LW = 1.0
SC_HATCH = '////'
SC_HATCH_TINT = 0.55
KEY_COLOUR = fa.STYLE_KEY_COLOUR
# White hatch across the optimised column of an offer whose optimisation is
# done by the user, as in the flow_exposure_analysis.py column charts; a
# VPP-optimised column is plain.
HATCH_USER = 'xxx'
PENDING_COLOUR = '0.45'

FIGURES = (
    dict(metric='hh_saving', own_plan=True,
         stem='household_saving_by_offer_own_plan',
         title='Battery savings at {kwh} kWh by offer, imperfect foresight, '
               'optimised',
         baseline='vs own solar only'),
    dict(metric='ws_saving', own_plan=False,
         stem='wholesale_saving_by_offer',
         title='Net wholesale savings at {kwh} kWh by offer, '
               'imperfect foresight, optimised',
         baseline='vs solar only'),
)
RANGE_LABEL = 'Imperfect foresight & VPP dispatch range'

# Offer tick labels on two lines where one would crowd.
OFFER_LABELS = {
    'GloBird ZeroHero': 'GloBird\nZeroHero',
    'AGL Battery Rewards': 'AGL Battery\nRewards',
    'EnergyAustralia Battery Ease': 'EnergyAustralia\nBattery Ease',
    'Origin Battery Lite': 'Origin\nBattery Lite',
}


def _tint(colour: str, frac: float) -> tuple:
    rgb = np.array(mcolors.to_rgb(colour))
    return tuple(rgb + (1 - rgb) * frac)


def load_values(output_dir: Path, weight_tag: str, scope: str,
                kwh: int) -> pd.DataFrame:
    """Rows of (basis, metric, offer, series, value) at the capacity, plus
    metric hh_own: the household saving against the offer's own Solar only
    bill. Self-consumption on the conservative basis has its enrolment
    payment taken off, which leaves the no-VPP run. A basis whose CSV is
    missing contributes no rows."""
    frames = []
    for basis in ('conservative', 'achievable'):
        path = (output_dir / fa.RUN_BASES[basis]['folder']
                / f'curves_{weight_tag}.csv')
        if not path.exists():
            print(f'  {basis}: no {path}, so its values are left out.')
            continue
        d = pd.read_csv(path)
        d = d[(d['scope'] == scope) & d['kwh'].isin([0, kwh])]
        wide = d.pivot_table(index=['metric', 'offer', 'series', 'mode'],
                             columns='kwh', values='value').reset_index()
        wide['basis'] = basis
        frames.append(wide)
    if not frames:
        sys.exit('No curves found; run financial_analysis.py first.')
    wide = pd.concat(frames, ignore_index=True)
    if kwh not in wide.columns:
        sys.exit(f'No curves for scope {scope!r} at {kwh} kWh.')
    wide['value'] = wide[kwh]
    sc_pay = ((wide['basis'] == 'conservative')
              & (wide['series'] == 'self-consumption')
              & (wide['metric'] == 'hh_saving'))
    if kwh > 0:
        wide.loc[sc_pay, 'value'] -= (wide.loc[sc_pay, 'mode']
                                      .map(fa.CONSERVATIVE_PAYMENTS).fillna(0))
    hh = wide['metric'] == 'hh_saving'
    hh_own = wide[hh].assign(metric='hh_own',
                             value=wide.loc[hh, 'value'] - wide.loc[hh, 0])
    return pd.concat([wide, hh_own], ignore_index=True)[
        ['basis', 'metric', 'offer', 'series', 'value']]


def _bars(values: pd.DataFrame, metric: str) -> dict[str, dict]:
    """Per offer: its colour, the no-VPP self-consumption value, and the
    conservative and achievable optimised values (None where missing), and
    its opt_type ('vpp' or 'user')."""
    v = values[values['metric'] == metric]
    get = {(r.basis, r.offer, r.series): r.value for r in v.itertuples()}
    colours = {fa._offer_name(p): p['color'] for p in fa._BASE_PLANS}
    inks = {fa._offer_name(p): fa.PLAN_INK.get(p['sc_mode'], p['color'])
            for p in fa._BASE_PLANS}
    opt_types = {fa._offer_name(p): p['opt_type'] for p in fa._BASE_PLANS}
    return {offer: dict(colour=colours.get(offer, KEY_COLOUR),
                        ink=inks.get(offer, KEY_COLOUR),
                        opt_type=opt_types.get(offer, 'vpp'),
                        sc=get.get(('conservative', offer, 'self-consumption')),
                        lo=get.get(('conservative', offer, 'optimised')),
                        hi=get.get(('achievable', offer, 'optimised')))
            for offer in dict.fromkeys(v['offer'])}


def offer_order(values: pd.DataFrame) -> list[str]:
    """Highest household achievable optimised saving first."""
    bars = _bars(values, 'hh_own')
    rank = {o: (b['hi'] if b['hi'] is not None else b['lo']) or 0.0
            for o, b in bars.items()}
    return sorted(rank, key=lambda o: -rank[o])


def _sc_patches(ink) -> tuple[dict, dict]:
    """Keyword sets for the two layers of a self-consumption bar: a light
    hatch in a tint of the offer colour, then the outline in the colour
    itself. A matplotlib hatch takes the edge colour, so the two need
    separate layers to differ."""
    return (dict(facecolor='white', edgecolor=_tint(ink, SC_HATCH_TINT),
                 hatch=SC_HATCH, linewidth=0),
            dict(facecolor='none', edgecolor=ink, linewidth=SC_EDGE_LW))


def _sc_bar(ax, x: float, y: float, ink) -> None:
    """Outlined and lightly hatched, so it cannot be read as part of the
    optimised bar beside it."""
    for kw in _sc_patches(ink):
        ax.bar([x], [y], width=BAR_WIDTH, zorder=3, **kw)


def _label(ax, x: float, y: float, text: str, **kw) -> None:
    ax.annotate(text, (x, max(y, 0)), xytext=(0, 1.5),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=kw.pop('fontsize', ts.ANNOTATION_SIZE),
                linespacing=1.0, zorder=4, **kw)


def make_figure(values: pd.DataFrame, spec: dict, order: list[str], kwh: int,
                notes: str | None, out_path: Path) -> None:
    bars = _bars(values, 'hh_own' if spec['own_plan'] else spec['metric'])
    offers = [o for o in order if o in bars]
    fig = ts.new_figure(HEIGHT_IN)
    ax = fig.subplots()
    top = 0.0
    x_sc = -(BAR_WIDTH + BAR_GAP) / 2
    x_opt = (BAR_WIDTH + BAR_GAP) / 2
    for i, offer in enumerate(offers):
        b = bars[offer]
        if b['sc'] is not None:
            _sc_bar(ax, i + x_sc, b['sc'], b['ink'])
            _label(ax, i + x_sc, b['sc'], f"{b['sc']:,.0f}")
            top = max(top, b['sc'])
        present = [v for v in (b['lo'], b['hi']) if v is not None]
        if not present:
            continue
        base, peak = min(present), max(present)
        opt_style = dict(color=b['colour'])
        if b['opt_type'] == 'user':
            opt_style.update(edgecolor='white', hatch=HATCH_USER, linewidth=0)
        ax.bar([i + x_opt], [base], width=BAR_WIDTH, zorder=3, **opt_style)
        if peak - base >= 0.5:
            ax.bar([i + x_opt], [peak - base], bottom=[base], width=BAR_WIDTH,
                   color=_tint(b['colour'], RANGE_TINT), zorder=3)
        _label(ax, i + x_opt, peak, f'{peak:,.0f}')
        top = max(top, peak)
        if b['hi'] is None:
            ax.annotate('range\npending', (i + x_opt, peak), xytext=(0, 11),
                        textcoords='offset points', ha='center', va='bottom',
                        color=PENDING_COLOUR, fontsize=ts.NOTE_SIZE,
                        linespacing=1.0)
    ax.set_xticks(range(len(offers)))
    ax.set_xticklabels([OFFER_LABELS.get(o, o) for o in offers])
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.6, len(offers) - 0.4)
    ax.grid(axis='x', visible=False)
    ax.set_ylim(0, top * 1.12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda y, _: f'${y:,.0f}'))
    ax.set_ylabel(fa.YLABEL)
    ax.set_title(spec['title'].format(kwh=kwh))
    if notes:
        ax.text(0.99, 0.97, notes, transform=ax.transAxes, ha='right',
                va='top', fontsize=ts.NOTE_SIZE, color=ts.REFERENCE_COLOUR,
                bbox=ts.LABEL_BOX, zorder=6)
    opt_types = {bars[o]['opt_type'] for o in offers}
    handles = [tuple(mpatches.Patch(**kw) for kw in _sc_patches(KEY_COLOUR))]
    labels = ['Self-consumption']
    if 'vpp' in opt_types:
        handles.append(mpatches.Patch(color=KEY_COLOUR))
        labels.append('VPP optimised')
    if 'user' in opt_types:
        handles.append(mpatches.Patch(facecolor=KEY_COLOUR, edgecolor='white',
                                      hatch=HATCH_USER, linewidth=0))
        labels.append('User optimised')
    handles.append(mpatches.Patch(color=_tint(KEY_COLOUR, RANGE_TINT)))
    labels.append(RANGE_LABEL)
    fig.legend(handles=handles, labels=labels, loc='outside lower center',
               ncol=len(handles), frameon=False)
    ts.save_figure(fig, out_path)


def lowest_bill_shares(output_dir: Path, weight_tag: str, scope: str,
                       kwh: int, state_mix: dict[str, float],
                       out_dir: Path | None = None) -> None:
    """Print, and write to lowest_bill_shares_<kwh>kwh_<scope>_<weighting>.csv
    under out_dir, the weighted share of site-years in which each offer gives
    the lowest bill with an optimised battery of kwh, for each basis.

    Every saving is against the same Solar only bill on Amber, so the highest
    saving is the lowest bill. Site-years are compared over the offers
    modelled for them. A site-year's weight is its PV size weight
    normalised within its scenario, divided by its state's number of years
    and, for the national scope, multiplied by the state's share of the
    state mix, which is the weighting behind the national curves.
    """
    rows = []
    for basis in ('achievable', 'conservative'):
        path = (output_dir / fa.RUN_BASES[basis]['folder']
                / f'site_savings_{weight_tag}.csv')
        if not path.exists():
            print(f'\n{basis}: no {path}; rerun financial_analysis.py for the '
                  f'lowest-bill shares.')
            continue
        d = pd.read_csv(path)
        d = d[(d['series'] == 'optimised') & (d['kwh'] == kwh)]
        if scope != 'national':
            state, rest = scope.split('_', 1)
            d = d[d['state'] == state]
            if rest != 'all_years':
                d = d[d['year'].astype(str) == rest]
        wide = d.pivot_table(index=['state', 'year', 'site_id'],
                             columns='offer', values='hh_saving')
        wide = wide[wide['Amber'].notna()]
        weight = (d.groupby(['state', 'year', 'site_id'])['weight'].first()
                  .reindex(wide.index))
        weight = weight / weight.groupby(level=['state', 'year']).transform('sum')
        years = weight.index.to_frame(index=False).groupby('state')['year'].nunique()
        states = weight.index.get_level_values('state')
        weight = weight / states.map(years).to_numpy()
        mix = pd.Series({s: state_mix.get(s, 0.0) for s in years.index})
        if scope == 'national':
            weight = weight * states.map(mix / mix.sum()).to_numpy()
        # Two populations: every site-year on its own, and each site with
        # its savings averaged over the years it appears in, which is the
        # share the Results text quotes. A site's weight in the second is the
        # mean of its yearly weights, renormalised within its state.
        site = wide.groupby(level=['state', 'site_id']).mean()
        site_weight = weight.groupby(level=['state', 'site_id']).mean()
        site_weight = (site_weight
                       / site_weight.groupby(level='state').transform('sum'))
        if scope == 'national':
            site_states = site_weight.index.get_level_values('state')
            site_weight = site_weight * site_states.map(mix / mix.sum()).to_numpy()
        for population, table, w, n in (
                ('site-year', wide, weight, len(wide)),
                ('site averaged across years', site, site_weight, len(site))):
            shares = w.groupby(table.idxmax(axis=1)).sum() / w.sum()
            print(f'\nLowest bill at {kwh} kWh, optimised, {basis}, '
                  f'{population} ({n}; offers compared: '
                  f"{', '.join(wide.columns)}):")
            for offer, share in shares.sort_values(ascending=False).items():
                print(f'  {offer:30s} {share:6.1%}')
                rows.append(dict(basis=basis, population=population,
                                 scope=scope, kwh=kwh, offer=offer,
                                 share_lowest_bill=share, n=n,
                                 offers_compared=', '.join(wide.columns)))
    if rows and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / (f'lowest_bill_shares_{kwh}kwh_{scope}_'
                         f'{weight_tag}.csv')
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'\nwrote {out}')


def _scope_note(scope: str, all_scopes: list[str]) -> str:
    """Where and when, for the first line of the notes."""
    def years_of(state=None):
        return sorted({s.split('_')[1] for s in all_scopes
                       if s != 'national' and s.split('_')[1].isdigit()
                       and (state is None or s.startswith(f'{state}_'))})
    if scope == 'national':
        states = sorted({s.split('_')[0] for s in all_scopes
                         if s != 'national'})
        return f"National ({', '.join(states)}), {fa._span(years_of())}"
    state, rest = scope.split('_', 1)
    if rest == 'all_years':
        return f'{state}, {fa._span(years_of(state))}'
    return f'{state} {rest}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--output-dir', type=Path,
                        default=Path('analysis/output/financial'),
                        help="financial_analysis.py's output directory, read "
                             "for the curves and written under "
                             "capacity_comparison/")
    parser.add_argument('--kwh', type=int, default=20,
                        help='Battery capacity, one of the modelled sizes')
    parser.add_argument('--scope', default='national',
                        help="'national', '<STATE>_all_years' or "
                             "'<STATE>_<YEAR>'")
    parser.add_argument('--unweighted', action='store_true',
                        help='Read the unweighted curves')
    parser.add_argument('--weights-file', type=Path,
                        default=Path('analysis/config/pv_size_bands.yaml'),
                        help='Weighting YAML, for the state mix')
    parser.add_argument('--no-notes', dest='notes', action='store_false',
                        help='Leave the data notes off, for the thesis')
    args = parser.parse_args(argv)
    if args.kwh not in fa.BATTERY_SIZES:
        sys.exit(f'--kwh must be one of {fa.BATTERY_SIZES}')

    # financial_analysis sets its own rcParams on import.
    matplotlib.rcdefaults()
    ts.apply_style()

    weight_tag = 'unweighted' if args.unweighted else 'weighted'
    values = load_values(args.output_dir, weight_tag, args.scope, args.kwh)
    scopes = pd.read_csv(
        args.output_dir / fa.RUN_BASES[values['basis'].iloc[0]]['folder']
        / f'curves_{weight_tag}.csv', usecols=['scope'])['scope'].unique()
    weighting = ('Unweighted' if args.unweighted else
                 'Weighted by PV size & state mix' if args.scope == 'national'
                 else 'Weighted by PV size')
    scope_note = _scope_note(args.scope, list(scopes))
    restricted = (fa.restricted_offer_notes(
        {s.split('_')[0] for s in scopes if s != 'national'})
        if args.scope == 'national' else [])
    order = offer_order(values)
    out_dir = args.output_dir / 'capacity_comparison'
    for spec in FIGURES:
        notes = '\n'.join([scope_note, f"{weighting} {spec['baseline']}"]
                          + restricted)
        print(f"\n{spec['stem']} notes (for the caption):\n{notes}")
        make_figure(values, spec, order, args.kwh,
                    notes if args.notes else None,
                    out_dir / f"{spec['stem']}_{args.kwh}kwh_{args.scope}_"
                              f"{weight_tag}")
    state_mix = (fa.load_weights_config(args.weights_file)['state_mix']
                 if not args.unweighted else {})
    lowest_bill_shares(args.output_dir, weight_tag, args.scope, args.kwh,
                       state_mix, out_dir)


if __name__ == '__main__':
    main()
