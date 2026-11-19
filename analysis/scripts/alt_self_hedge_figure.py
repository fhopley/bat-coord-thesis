"""
alt_self_hedge_figure.py

Illustrative thesis figure of self-hedging under the designed offers
(docs/thesis_figure_spec.md, Illustrative figures). No model data: the flows
are constructed below and can be edited directly.

Left: grid flows at 15-minute resolution through an evening with a two-hour
price spike (18:00-20:00, as in alt_capped_price_figure.py). The battery
exports into the early part of the spike, empties, and the household imports
for the rest of it. Only flows inside the spike window count towards the
above-cap balance; flows either side are drawn in grey. Before the spike
the grid charges the battery and then the battery supplies the load; after
it empties, the load is imported, easing later in the evening.

Right: the above-cap value of those flows over the settlement period. With
the spot price at one fixed level through the spike, each interval's
above-cap value is (P - cap) x flow, so the bars are the window's export and
import totals scaled by one constant: above-cap export earnings, less
above-cap import costs, leaves the net credit (B < 0). Loss factors and GST
are omitted.

  python3 analysis/scripts/alt_self_hedge_figure.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_style import (ANNOTATION_SIZE, REFERENCE_COLOUR,  # noqa: E402
                          SEPARATOR_COLOUR, SUBTITLE_SIZE, apply_style,
                          new_figure, panel_suptitle, save_figure)

# ---- Constructed flows, kWh per 15-minute interval (import +, export -),
# from START_HOUR. Edit here.
START_HOUR = 15.0
INTERVAL_H = 0.25
FLOWS_KWH = np.array([
    0.70, 0.75, 0.70, 0.70, 0.75, 0.70, 0.65,  # 15:00  grid charging the battery
    0.00, 0.00, 0.00, 0.00, 0.00,       # 16:45  battery supplies the load
    -1.10, -1.10, -1.05, -1.00, -0.90, -0.60,  # 18:00  spike: battery exports
    0.35, 0.35,                         # 19:30  battery empty: load imported
    0.32, 0.30, 0.27, 0.25,             # 20:00  spike over, load easing
])
# Phase labels: (start hour, end hour, text). A thin line spans the phase
# just above its bars, with the text centred over it.
PHASES = (
    (15.0, 16.75, 'Battery charging'),
    (16.75, 18.0, 'Battery\nsupplies load'),
    (19.5, 21.0, 'Battery empty'),
)
SPIKE_START_H, SPIKE_END_H = 18.0, 20.0

FIG_HEIGHT_IN = 2.2
EXPORT_COLOUR = '#2a78d6'
IMPORT_COLOUR = '#eb6834'
NET_COLOUR = '#1baf7a'
OUTSIDE_COLOUR = '0.78'
WINDOW_SHADE = '0.93'
BAR_WIDTH = 0.8                         # fraction of an interval


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--out', type=Path, default=REPO_ROOT /
                    'analysis/output/alt_offers/alt_self_hedge_offsetting.png')
    args = ap.parse_args(argv)

    t = START_HOUR + INTERVAL_H * np.arange(len(FLOWS_KWH))   # interval start
    in_spike = (t >= SPIKE_START_H) & (t < SPIKE_END_H)
    exp_kwh = -FLOWS_KWH[in_spike & (FLOWS_KWH < 0)].sum()
    imp_kwh = FLOWS_KWH[in_spike & (FLOWS_KWH > 0)].sum()
    net_kwh = exp_kwh - imp_kwh
    if net_kwh <= 0:
        raise ValueError("constructed flows give no net credit; the figure "
                         "is drawn for B < 0")

    apply_style()
    fig = new_figure(FIG_HEIGHT_IN)
    ax_f, ax_v = fig.subplots(1, 2, width_ratios=(1.6, 1))

    # ---- Left: flows
    colours = np.where(~in_spike, OUTSIDE_COLOUR,
                       np.where(FLOWS_KWH < 0, EXPORT_COLOUR, IMPORT_COLOUR))
    ax_f.axvspan(SPIKE_START_H, SPIKE_END_H, color=WINDOW_SHADE, lw=0, zorder=0)
    ax_f.bar(t + INTERVAL_H / 2, FLOWS_KWH, width=BAR_WIDTH * INTERVAL_H,
             color=colours, zorder=3)
    ax_f.axhline(0, color=REFERENCE_COLOUR, lw=0.6, zorder=4)
    lim = 1.15 * np.abs(FLOWS_KWH).max()
    ax_f.set_ylim(-lim, 1.05 * lim)
    ax_f.set_xlim(START_HOUR, t[-1] + INTERVAL_H)
    hours = np.arange(START_HOUR, t[-1] + INTERVAL_H + 1e-9, 1.0)
    ax_f.set_xticks(hours)
    ax_f.set_xticklabels([f'{int(h):02d}:00' for h in hours])
    ax_f.set_yticks([])
    ax_f.grid(False)
    ax_f.text((SPIKE_START_H + SPIKE_END_H) / 2, 0.97 * 1.05 * lim,
              'Spot price above cap', ha='center', va='top',
              fontsize=ANNOTATION_SIZE, color=REFERENCE_COLOUR)
    for h0, h1, text in PHASES:
        y = max(FLOWS_KWH[(t >= h0) & (t < h1)].max(), 0.0) + 0.06 * lim
        ax_f.plot([h0 + 0.03, h1 - 0.03], [y, y], color=REFERENCE_COLOUR,
                  lw=0.6, solid_capstyle='butt', zorder=4)
        ax_f.text((h0 + h1) / 2, y + 0.03 * lim, text, ha='center',
                  va='bottom', fontsize=ANNOTATION_SIZE,
                  color=REFERENCE_COLOUR, linespacing=1.1)
    for y, label, va in ((0.97, 'Import', 'top'), (0.03, 'Export', 'bottom')):
        # Direction labels belong to the axis, so they take axis text colour
        # rather than the grey of the notes.
        ax_f.text(0.015, y, label, transform=ax_f.transAxes, ha='left', va=va,
                  fontsize=ANNOTATION_SIZE, color='black')
    handles = [Patch(color=EXPORT_COLOUR), Patch(color=IMPORT_COLOUR),
               Patch(color=OUTSIDE_COLOUR)]
    leg = ax_f.legend(handles, ['Export while above cap',
                                'Import while above cap',
                                'Not in the above-cap balance'],
                      loc='lower left', bbox_to_anchor=(0.0, 0.1),
                      handlelength=1.0)
    leg.set_zorder(20)
    ax_f.set_title('Grid flows (kWh) in each 15 minutes',
                   fontsize=SUBTITLE_SIZE)

    # ---- Right: above-cap value over the period, as a waterfall
    x = np.arange(3)
    ax_v.bar(x[0], exp_kwh, width=0.6, color=EXPORT_COLOUR, zorder=3)
    ax_v.bar(x[1], -imp_kwh, bottom=exp_kwh, width=0.6, color=IMPORT_COLOUR,
             zorder=3)
    ax_v.bar(x[2], net_kwh, width=0.6, color=NET_COLOUR, zorder=3)
    for x0, x1, y in ((x[0], x[1], exp_kwh), (x[1], x[2], net_kwh)):
        ax_v.plot([x0 + 0.3, x1 - 0.3], [y, y], ls='--', lw=0.6,
                  color=SEPARATOR_COLOUR, zorder=2)
    ax_v.axhline(0, color=REFERENCE_COLOUR, lw=0.6, zorder=4)
    ax_v.set_ylim(0, 1.15 * exp_kwh)
    ax_v.set_xlim(-0.6, 2.6)
    ax_v.set_xticks(x)
    ax_v.set_xticklabels(['Above-cap\nexport earnings',
                          'Above-cap\nimport costs', 'Net credit\n(B < 0)'])
    ax_v.tick_params(axis='x', length=0)
    ax_v.set_yticks([])
    ax_v.grid(False)
    ax_v.set_title('Above-cap value (\\$) over the settlement period',
                   fontsize=SUBTITLE_SIZE)

    panel_suptitle(fig,
                   'Self-hedging of above-cap imports by above-cap exports')
    save_figure(fig, args.out)
    print(f"spike-window export {exp_kwh:.2f} kWh, import {imp_kwh:.2f} kWh, "
          f"net {net_kwh:.2f} kWh")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
