"""
poster_style.py

Draft format for poster figures (thesis/poster/poster_spec.md, Figures).
Text is 18 pt at the printed size, so a figure is drawn at the width it is
printed (one column or two) and inserted at 100 %. Marks are the thesis
marks scaled by TEXT_SCALE.

    import poster_style as ps
    ps.apply_style()
    fig = ps.new_figure(ps.ONE_COL_CM, 8.5)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

import thesis_style as ts

CM = 1 / 2.54
ONE_COL_CM = 17.13
TWO_COL_CM = 35.76

TEXT_SCALE = 18 / 6.5
AXIS_LABEL = 20
TEXT = 18                       # ticks, legend, annotations
SMALL = 16                      # minimum anywhere in a figure

# Thesis mark sizes scaled with the text.
DATA_LW = round(ts.DATA_LW * TEXT_SCALE, 1)        # 3.6
MARKER = round(ts.MARKER_SIZE * TEXT_SCALE, 1)     # 8.4
FRAME_LW = round(ts.RC['axes.linewidth'] * TEXT_SCALE, 1)   # 1.7
GRID_LW = round(0.5 * TEXT_SCALE, 1)               # 1.4
BOX_LW = round(ts.RC['axes.linewidth'] * TEXT_SCALE, 1)
MEDIAN_LW = round(BOX_LW * 1.6, 1)
REFERENCE_LW = round(ts.REFERENCE_LW * TEXT_SCALE, 1)
TICK_LEN = round(3 * TEXT_SCALE, 1)
LABEL_BOX = dict(facecolor='white', edgecolor='none', pad=2.0)

# Figure titles. 'figure' sets the title in a band at the top-left of the
# figure, above the y-axis tick labels, so it can be longer than the plot
# box; 'axes' sets it left-aligned with the plot box (the layout engine
# reserves the room). Change TITLE_MODE to switch every poster figure.
TITLE_MODE = 'figure'
TITLE_PT = 24
TITLE_BAND_CM = 1.7             # same band on every figure
TITLE_TOP_CM = 0.15

# Wide (hero-width) figures share a frame width and the plot-box edges, so
# the boxes of Figures 2 and 4 line up when both frames sit at the same left.
WIDE_W_CM = 37.2
PLOT_LEFT_CM = 2.8              # frame edge to plot box, room for y labels
PLOT_RIGHT_CM = 36.3            # leaves room for an end tick label


def _font() -> str:
    """Calibri where installed (the poster body font), then its metric-
    compatible Carlito, then DejaVu Sans."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Calibri', 'Carlito', 'DejaVu Sans'):
        if name in have:
            return name
    return 'DejaVu Sans'


def apply_style() -> None:
    rc = dict(ts.RC)
    rc.update({
        'font.sans-serif': [_font()],
        'font.size': TEXT,
        'axes.titlesize': AXIS_LABEL,
        'axes.labelsize': AXIS_LABEL,
        'xtick.labelsize': TEXT,
        'ytick.labelsize': TEXT,
        'legend.fontsize': TEXT,
        'figure.titlesize': AXIS_LABEL,
        'axes.linewidth': FRAME_LW,
        'xtick.major.width': FRAME_LW,
        'ytick.major.width': FRAME_LW,
        'xtick.major.size': TICK_LEN,
        'ytick.major.size': TICK_LEN,
        'xtick.major.pad': 5,
        'ytick.major.pad': 5,
        'lines.linewidth': DATA_LW,
        'lines.markersize': MARKER,
        'grid.linewidth': GRID_LW,
        'legend.handlelength': 1.6,
    })
    mpl.rcParams.update(rc)


def new_figure(width_cm: float, height_cm: float,
               layout: str | None = 'constrained', title: bool = False):
    """Figure at its printed size. title=True reserves the title band at the
    top when TITLE_MODE is 'figure' (constrained layout only)."""
    fig = plt.figure(figsize=(width_cm * CM, height_cm * CM), layout=layout)
    if title and layout == 'constrained' and TITLE_MODE == 'figure':
        fig.get_layout_engine().set(
            rect=(0, 0, 1, 1 - TITLE_BAND_CM / height_cm))
    return fig


def add_title(fig, ax, text: str, top_cm: float | None = None) -> None:
    """Figure title in the current TITLE_MODE. top_cm places a 'figure' title
    lower down the figure (a second pane title)."""
    if TITLE_MODE == 'axes':
        ax.set_title(text, loc='left', fontsize=TITLE_PT, fontweight='bold')
        return
    w, h = (v / CM for v in fig.get_size_inches())
    t = TITLE_TOP_CM if top_cm is None else top_cm
    fig.text(0.1 / w, 1 - t / h, text, ha='left', va='top',
             fontsize=TITLE_PT, fontweight='bold')


def save_figure(fig, out_path) -> None:
    ts.save_figure(fig, out_path)
