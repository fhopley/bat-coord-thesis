"""
poster_offers_figure.py

Poster Figure 4 as one figure (thesis/poster/poster_spec.md): the capped
import price and uncapped export reward (top pane) above the self-hedging
illustration (bottom pane). Both panes share the left and right plot edges.
Data and constructed flows are those of alt_capped_price_figure.py and
alt_self_hedge_figure.py; only the layout and the poster format differ.

  python3 analysis/scripts/poster_offers_figure.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / 'analysis/scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import alt_capped_price_figure as cp  # noqa: E402
import alt_self_hedge_figure as sh  # noqa: E402
import poster_style as ps  # noqa: E402
import thesis_style as ts  # noqa: E402
from model.core.market_prices import market_price_cap  # noqa: E402
from model.core.tariff import load_network_tariff, load_retail_tariff  # noqa: E402

WIDTH_CM, HEIGHT_CM = ps.WIDE_W_CM, 24.9
LEFT_CM, RIGHT_CM = ps.PLOT_LEFT_CM, ps.PLOT_RIGHT_CM   # shared with Fig 2
GAP_CM = 2.0
# Vertical layout, cm from the top of the figure.
TITLE_A_TOP = ps.TITLE_TOP_CM
A_AXES_TOP, A_AXES_H, A_BREAK_GAP = 2.6, 8.6, 0.3
B_AXES_TOP, B_AXES_H = 14.1, 8.2
BOTTOM_WIDTH_RATIO = (1.6, 1.0)


def axes_cm(fig, left, top, width, height, **kw):
    return fig.add_axes([left / WIDTH_CM, 1 - (top + height) / HEIGHT_CM,
                         width / WIDTH_CM, height / HEIGHT_CM], **kw)


def break_marks(ax_top, ax_bottom):
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=12, linestyle='none',
              color='k', mec='k', mew=ps.FRAME_LW, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kw)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kw)


def price_panel(fig, left, width, line, lo, hi, title, cap_c, show_y,
                cap_label=None):
    top_h = (A_AXES_H - A_BREAK_GAP) / 5
    ax_t = axes_cm(fig, left, A_AXES_TOP, width, top_h)
    ax_b = axes_cm(fig, left, A_AXES_TOP + top_h + A_BREAK_GAP, width,
                   A_AXES_H - top_h - A_BREAK_GAP)
    x, y = cp.steps(line)
    _, ylo = cp.steps(lo)
    _, yhi = cp.steps(hi)
    for ax in (ax_t, ax_b):
        ax.fill_between(x, ylo, yhi, step='post', color=ts.LINE_COLOUR,
                        alpha=ts.BAND_ALPHA, lw=0, zorder=1,
                        label='Interquartile range')
        ax.plot(x, y, drawstyle='steps-post', color=ts.LINE_COLOUR,
                lw=ps.DATA_LW, zorder=3, label='Median, with spike')
        if cap_c is not None:
            ax.axhline(cap_c, color=ts.REFERENCE_COLOUR, ls='--',
                       lw=ps.REFERENCE_LW, zorder=2, label=cap_label)
        ax.set_xlim(0, 24)
    for ax in (ax_t, ax_b):
        ax.set_xticks([0, 6, 12, 18, 24])
    ax_t.set_ylim(*cp.UPPER_YLIM)
    ax_t.set_yticks(cp.UPPER_TICKS)
    ax_t.spines['bottom'].set_visible(False)
    ax_t.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_t.set_title(title, fontsize=ps.TEXT)
    ax_b.set_ylim(*cp.LOWER_YLIM)
    ax_b.set_yticks(cp.LOWER_TICKS)
    ax_b.spines['top'].set_visible(False)
    ax_b.set_xticks([0, 6, 12, 18, 24])
    ax_b.set_xticklabels(['00:00', '06:00', '12:00', '18:00', '24:00'])
    if not show_y:
        for ax in (ax_t, ax_b):
            ax.tick_params(axis='y', labelleft=False)
    break_marks(ax_t, ax_b)
    return ax_t, ax_b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--profile', type=Path, default=REPO_ROOT /
                    'data/output/cap_slides/nsw_profile_pooled.csv')
    ap.add_argument('--tariffs-dir', type=Path,
                    default=REPO_ROOT / 'model/tariffs')
    ap.add_argument('--cap-mwh', type=float, default=300.0)
    ap.add_argument('--mpc-date', default='2026-07-15')
    ap.add_argument('--out', type=Path, default=REPO_ROOT /
                    'analysis/output/poster/poster_capped_offers_poster.png')
    args = ap.parse_args(argv)

    p = pd.read_csv(args.profile)
    network = load_network_tariff('NSW_ausgrid_ea025_ea029_2026',
                                  args.tariffs_dir)
    retail = load_retail_tariff('NSW_amber_wholesale_2026', args.tariffs_dir)
    mpc = market_price_cap(pd.Timestamp(args.mpc_date))
    cap = cp.flat_stack(args.cap_mwh, network, retail, args.mpc_date)
    spike = cp.flat_stack(mpc, network, retail, args.mpc_date)
    b0 = 18 * 12
    window = np.zeros(cp.BINS, dtype=bool)
    window[b0:b0 + 24] = True
    cap_imp = cap['import_c_kwh'].to_numpy()
    imp = np.minimum(np.where(window, spike['import_c_kwh'].to_numpy(),
                              p['import_c_kwh_p50'].to_numpy()), cap_imp)
    exp = np.where(window, spike['export_c_kwh'].to_numpy(),
                   p['export_c_kwh_p50'].to_numpy())
    cap_c = float(cap_imp[window].max())

    ps.apply_style()
    fig = ps.new_figure(WIDTH_CM, HEIGHT_CM, layout=None)
    pw = (RIGHT_CM - LEFT_CM - GAP_CM) / 2

    # ---- Top pane
    t_l, b_l = price_panel(
        fig, LEFT_CM, pw, imp, p['import_c_kwh_p25'].to_numpy(),
        p['import_c_kwh_p75'].to_numpy(), 'Import price (c/kWh) with cap',
        cap_c, True, f'Cap (${args.cap_mwh:,.0f}/MWh wholesale)')
    t_r, b_r = price_panel(
        fig, LEFT_CM + pw + GAP_CM, pw, exp, p['export_c_kwh_p25'].to_numpy(),
        p['export_c_kwh_p75'].to_numpy(), 'Export reward (c/kWh) uncapped',
        None, False)
    ps.add_title(fig, None, 'Import and export rates under the capped '
                 'offers, and how self-hedging operates')
    b1 = b0 + int(window.sum())
    b_l.annotate(f'{cap_c:.1f}', xy=(b1 * cp.BIN_H, cap_c), xytext=(4, 3),
                 textcoords='offset points', ha='left', va='bottom',
                 fontsize=ps.TEXT, color=ts.LINE_COLOUR)
    t_r.annotate(f'{exp[window].max():,.0f}',
                 xy=(b1 * cp.BIN_H, exp[window].max()), xytext=(4, 0),
                 textcoords='offset points', ha='left', va='center',
                 fontsize=ps.TEXT, color=ts.LINE_COLOUR)
    b_l.legend(loc='upper left', bbox_to_anchor=(0.0, 0.79),
               handlelength=1.2, handletextpad=0.5, borderpad=0.3,
               borderaxespad=0.3).set_zorder(20)

    # ---- Bottom pane
    wl = (RIGHT_CM - LEFT_CM - GAP_CM) * BOTTOM_WIDTH_RATIO[0] / sum(
        BOTTOM_WIDTH_RATIO)
    wr = (RIGHT_CM - LEFT_CM - GAP_CM) - wl
    ax_f = axes_cm(fig, LEFT_CM, B_AXES_TOP, wl, B_AXES_H)
    ax_v = axes_cm(fig, LEFT_CM + wl + GAP_CM, B_AXES_TOP, wr, B_AXES_H)
    t = sh.START_HOUR + sh.INTERVAL_H * np.arange(len(sh.FLOWS_KWH))
    in_spike = (t >= sh.SPIKE_START_H) & (t < sh.SPIKE_END_H)
    flows = sh.FLOWS_KWH
    exp_kwh = -flows[in_spike & (flows < 0)].sum()
    imp_kwh = flows[in_spike & (flows > 0)].sum()
    net_kwh = exp_kwh - imp_kwh
    colours = np.where(~in_spike, sh.OUTSIDE_COLOUR,
                       np.where(flows < 0, sh.EXPORT_COLOUR, sh.IMPORT_COLOUR))
    ax_f.axvspan(sh.SPIKE_START_H, sh.SPIKE_END_H, color=sh.WINDOW_SHADE,
                 lw=0, zorder=0)
    ax_f.bar(t + sh.INTERVAL_H / 2, flows, width=sh.BAR_WIDTH * sh.INTERVAL_H,
             color=colours, zorder=3)
    ax_f.axhline(0, color=ts.REFERENCE_COLOUR, lw=ps.REFERENCE_LW, zorder=4)
    lim = 1.15 * np.abs(flows).max()
    ax_f.set_ylim(-lim, 1.05 * lim)
    ax_f.set_xlim(sh.START_HOUR, t[-1] + sh.INTERVAL_H)
    hours = np.arange(sh.START_HOUR, t[-1] + sh.INTERVAL_H + 1e-9, 1.0)
    ax_f.set_xticks(hours)
    ax_f.set_xticklabels([f'{int(h):02d}:00' for h in hours])
    ax_f.set_yticks([])
    ax_f.grid(False)
    ax_f.text((sh.SPIKE_START_H + sh.SPIKE_END_H) / 2, 0.97 * 1.05 * lim,
              'Spot price above cap', ha='center', va='top',
              fontsize=ps.TEXT, color=ts.REFERENCE_COLOUR)
    for h0, h1, text in sh.PHASES:
        y = max(flows[(t >= h0) & (t < h1)].max(), 0.0) + 0.06 * lim
        ax_f.plot([h0 + 0.03, h1 - 0.03], [y, y], color=ts.REFERENCE_COLOUR,
                  lw=ps.REFERENCE_LW, solid_capstyle='butt', zorder=4)
        ax_f.text((h0 + h1) / 2, y + 0.03 * lim, text, ha='center',
                  va='bottom', fontsize=ps.TEXT, color=ts.REFERENCE_COLOUR,
                  linespacing=1.1)
    handles = [Patch(color=sh.EXPORT_COLOUR), Patch(color=sh.IMPORT_COLOUR),
               Patch(color=sh.OUTSIDE_COLOUR)]
    ax_f.legend(handles, ['Export while above cap', 'Import while above cap',
                          'Not in the above-cap balance'], loc='lower left',
                bbox_to_anchor=(0.0, 0.02), handlelength=1.0).set_zorder(20)
    ax_f.set_title('Grid flows (kWh) in each 15 minutes', fontsize=ps.TEXT)

    x = np.arange(3)
    ax_v.bar(x[0], exp_kwh, width=0.6, color=sh.EXPORT_COLOUR, zorder=3)
    ax_v.bar(x[1], -imp_kwh, bottom=exp_kwh, width=0.6,
             color=sh.IMPORT_COLOUR, zorder=3)
    ax_v.bar(x[2], net_kwh, width=0.6, color=sh.NET_COLOUR, zorder=3)
    for x0, x1, y in ((x[0], x[1], exp_kwh), (x[1], x[2], net_kwh)):
        ax_v.plot([x0 + 0.3, x1 - 0.3], [y, y], ls='--', lw=ps.GRID_LW * 1.2,
                  color=ts.SEPARATOR_COLOUR, zorder=2)
    ax_v.axhline(0, color=ts.REFERENCE_COLOUR, lw=ps.REFERENCE_LW, zorder=4)
    ax_v.set_ylim(0, 1.15 * exp_kwh)
    ax_v.set_xlim(-0.6, 2.6)
    ax_v.set_xticks(x)
    ax_v.set_xticklabels(['Export\nearnings', 'Import\ncosts', 'Net\ncredit'])
    ax_v.tick_params(axis='x', length=0)
    ax_v.set_yticks([])
    ax_v.grid(False)
    ax_v.set_title('Above-cap value ($)', fontsize=ps.TEXT)

    ps.save_figure(fig, args.out)
    print('Saved', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
