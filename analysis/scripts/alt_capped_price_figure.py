"""
alt_capped_price_figure.py

Thesis figure for the capped import price: the NSW wholesale-passthrough
import price with a $300/MWh cap (left) beside the uncapped export price
(right), on a shared broken y-axis, each with a two-hour spike at the market
price cap added to the median day.

Base curves are the median and interquartile range by local time of day from
data/output/cap_slides/nsw_profile_pooled.csv (export_cap_slide_profiles.py,
NSW 2021-2025, Ausgrid EA025/EA029, Amber 2026 adders, GST on spot imports).

Spike and cap levels are the same price stack evaluated at a flat spot price,
so they carry the network period, loss factors, adders and GST of each bin:

    cap level     stack at spot = --cap-mwh                  (import only)
    spike         stack at spot = market price cap for --mpc-date
    import        median, spike bins clamped to the cap level
    export        median, spike bins at the uncapped spike price

Bins are interval-start local time (local_wall_clock), drawn as steps.

Format from thesis_style.py (docs/thesis_figure_spec.md).

  python3 analysis/scripts/alt_capped_price_figure.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.core.market_prices import market_price_cap  # noqa: E402
from model.core.tariff import load_network_tariff, load_retail_tariff  # noqa: E402
from model.scripts.export_cap_slide_profiles import stack_for_year, tod_bin  # noqa: E402
from thesis_style import (ANNOTATION_SIZE, BAND_ALPHA, LINE_COLOUR,  # noqa: E402
                          REFERENCE_COLOUR, REFERENCE_LW, SUBTITLE_SIZE,
                          apply_style, new_figure, panel_suptitle,
                          save_figure)

BINS = 288
BIN_H = 24 / BINS

# Axis ranges. The lower band holds every base curve; the upper band holds
# the export spike.
LOWER_YLIM = (-5, 90)
LOWER_TICKS = [0, 20, 40, 60, 80]
UPPER_YLIM = (2250, 2420)
UPPER_TICKS = [2400]
HEIGHT_RATIOS = (1, 4)
FIG_HEIGHT_IN = 2.2


def flat_stack(spot_mwh: float, network, retail, day: str) -> pd.DataFrame:
    """Import and export c/kWh by time-of-day bin at a constant spot price."""
    idx = pd.date_range(f'{day} 00:05', periods=BINS, freq='5min',
                        tz='Etc/GMT-10')
    df = stack_for_year(pd.Series(spot_mwh, index=idx), network, retail,
                        'NSW', 5 / 60)
    df['tod_bin'] = tod_bin(idx, 'NSW', 5 / 60)
    out = df.groupby('tod_bin')[['import_c_kwh', 'export_c_kwh']].first()
    if len(out) != BINS:
        raise ValueError(f"{day}: expected {BINS} bins, got {len(out)} "
                         "(pick a day without a DST transition)")
    return out.reindex(range(BINS))


def steps(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x in hours and y for a post-step line spanning 0..24."""
    x = np.arange(BINS + 1) * BIN_H
    return x, np.append(values, values[-1])


def break_marks(ax_top, ax_bottom) -> None:
    """Diagonal marks on both side spines where the axis is broken."""
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=6, linestyle='none',
              color='k', mec='k', mew=0.6, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kw)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kw)


def draw_panel(ax_top, ax_bot, line, lo, hi, title, cap_c, show_y,
               cap_label=None):
    """One price panel on a broken axis. cap_c=None draws no cap line."""
    x, y = steps(line)
    _, ylo = steps(lo)
    _, yhi = steps(hi)
    for ax in (ax_top, ax_bot):
        ax.fill_between(x, ylo, yhi, step='post', color=LINE_COLOUR,
                        alpha=BAND_ALPHA, lw=0, zorder=1,
                        label='Interquartile range')
        ax.plot(x, y, drawstyle='steps-post', color=LINE_COLOUR,
                zorder=3, label='Median, with spike')
        if cap_c is not None:
            ax.axhline(cap_c, color=REFERENCE_COLOUR, ls='--', lw=REFERENCE_LW,
                       zorder=2, label=cap_label)
        ax.set_xlim(0, 24)

    ax_top.set_ylim(*UPPER_YLIM)
    ax_top.set_yticks(UPPER_TICKS)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_top.set_title(title, fontsize=SUBTITLE_SIZE)

    ax_bot.set_ylim(*LOWER_YLIM)
    ax_bot.set_yticks(LOWER_TICKS)
    ax_bot.spines['top'].set_visible(False)
    ax_bot.set_xticks([0, 6, 12, 18, 24])
    ax_bot.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])

    if not show_y:
        for ax in (ax_top, ax_bot):
            ax.tick_params(axis='y', labelleft=False)
    break_marks(ax_top, ax_bot)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--profile', type=Path,
                    default=REPO_ROOT / 'data/output/cap_slides/nsw_profile_pooled.csv')
    ap.add_argument('--tariffs-dir', type=Path, default=REPO_ROOT / 'model/tariffs')
    ap.add_argument('--network-tariff-id', default='NSW_ausgrid_ea025_ea029_2026')
    ap.add_argument('--retail-tariff-id', default='NSW_amber_wholesale_2026')
    ap.add_argument('--cap-mwh', type=float, default=300.0)
    ap.add_argument('--mpc-date', default='2026-07-15',
                    help="day whose market price cap sets the spike and on "
                         "which the flat stacks are evaluated")
    ap.add_argument('--spike-start', default='18:00')
    ap.add_argument('--spike-hours', type=float, default=2.0)
    ap.add_argument('--out', type=Path,
                    default=REPO_ROOT / 'analysis/output/alt_offers/alt_capped_import_price.png')
    args = ap.parse_args(argv)

    p = pd.read_csv(args.profile)
    if len(p) != BINS:
        raise ValueError(f"{args.profile}: expected {BINS} rows, got {len(p)}")

    network = load_network_tariff(args.network_tariff_id, args.tariffs_dir)
    retail = load_retail_tariff(args.retail_tariff_id, args.tariffs_dir)
    mpc = market_price_cap(pd.Timestamp(args.mpc_date))
    cap = flat_stack(args.cap_mwh, network, retail, args.mpc_date)
    spike = flat_stack(mpc, network, retail, args.mpc_date)

    h, m = (int(v) for v in args.spike_start.split(':'))
    b0 = h * 12 + m // 5
    window = np.zeros(BINS, dtype=bool)
    window[b0:b0 + int(round(args.spike_hours * 12))] = True

    cap_imp = cap['import_c_kwh'].to_numpy()
    imp_spike = spike['import_c_kwh'].to_numpy()
    exp_spike = spike['export_c_kwh'].to_numpy()
    imp = np.where(window, imp_spike, p['import_c_kwh_p50'].to_numpy())
    imp = np.minimum(imp, cap_imp)
    exp = np.where(window, exp_spike, p['export_c_kwh_p50'].to_numpy())

    cap_c = float(cap_imp[window].max())
    if np.ptp(cap_imp[window]) > 1e-9:
        raise ValueError("cap level varies within the spike window; "
                         "a single dashed line would misstate it")
    top = max(p['import_c_kwh_p75'].max(), p['export_c_kwh_p75'].max())
    if top >= LOWER_YLIM[1] or exp[window].max() >= UPPER_YLIM[1]:
        raise ValueError("data exceeds the fixed axis bands; adjust LOWER_YLIM/UPPER_YLIM")

    apply_style()
    fig = new_figure(FIG_HEIGHT_IN, layout=None)
    axes = fig.subplots(2, 2, sharex='col',
                        gridspec_kw=dict(height_ratios=HEIGHT_RATIOS,
                                         left=0.07, right=0.97, bottom=0.12,
                                         top=0.84, hspace=0.07, wspace=0.17))
    draw_panel(axes[0, 0], axes[1, 0], imp, p['import_c_kwh_p25'].to_numpy(),
               p['import_c_kwh_p75'].to_numpy(),
               'Import price (c/kWh) with cap', cap_c, True,
               cap_label=f'Cap (\\${args.cap_mwh:,.0f}/MWh wholesale)')
    draw_panel(axes[0, 1], axes[1, 1], exp, p['export_c_kwh_p25'].to_numpy(),
               p['export_c_kwh_p75'].to_numpy(), 'Export reward (c/kWh) uncapped',
               None, False)
    panel_suptitle(fig, 'Import and export rates under both designed offers',
                   y=0.985)

    b1 = b0 + int(window.sum())
    axes[1, 0].annotate(f'{cap_c:.1f}', xy=(b1 * BIN_H, cap_c), xytext=(3, 2),
                        textcoords='offset points', ha='left', va='bottom',
                        fontsize=ANNOTATION_SIZE, color=LINE_COLOUR)
    axes[0, 1].annotate(f'{exp[window].max():,.0f}',
                        xy=(b1 * BIN_H, exp[window].max()), xytext=(3, 0),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=ANNOTATION_SIZE, color=LINE_COLOUR)


    # Legend in the gap between the overnight curve and the cap line.
    leg = axes[1, 0].legend(loc='upper left', bbox_to_anchor=(0.0, 0.79),
                            handlelength=1.2, handletextpad=0.5,
                            borderpad=0.3, borderaxespad=0.3)
    leg.set_zorder(20)

    save_figure(fig, args.out)

    print(f"market price cap {mpc:,.0f} $/MWh ({args.mpc_date}); "
          f"cap {args.cap_mwh:,.0f} $/MWh")
    print(f"import cap level in spike window : {cap_c:.2f} c/kWh")
    print(f"uncapped import in spike window  : {imp_spike[window].max():,.1f} c/kWh")
    print(f"export in spike window           : {exp[window].max():,.1f} c/kWh")
    print(f"median import peak               : {p['import_c_kwh_p50'].max():.2f} c/kWh")
    print(f"import p75 peak                  : {p['import_c_kwh_p75'].max():.2f} c/kWh")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
