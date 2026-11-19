"""
thesis_style.py

Default format for thesis figures (docs/thesis_figure_spec.md).

    from thesis_style import apply_style, new_figure, save_figure
    apply_style()
    fig = new_figure(height_in=2.0)
    ...
    save_figure(fig, out_path)

Scripts outside analysis/scripts import it as analysis.scripts.thesis_style.
"""
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt

# A4 (21.0 cm) less 2.54 cm margins each side = 15.92 cm.
TEXT_WIDTH_IN = 15.92 / 2.54
DPI = 300
ANNOTATION_SIZE = 6.5
SUBTITLE_SIZE = 6.5               # panel titles under a figure title
NOTE_SIZE = 5.5                   # data notes, presentation versions only

# Colours
LINE_COLOUR = '#2a78d6'          # single series
EMPHASIS_COLOUR = 'black'        # aggregate drawn over its members
REFERENCE_COLOUR = '0.3'         # cap, threshold, target, derived value
BAND_ALPHA = 0.22
# Categorical series with no fixed identity (sites, years), assigned in
# sorted order. Checked for colour-vision separation; the lighter hues sit
# below 3:1 against white, so identity is also carried by the legend.
CATEGORICAL_COLOURS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                       '#e87ba4', '#008300', '#6250d6', '#e34948')

# Line widths (pt) and marker size
DATA_LW = 1.3                    # single or primary series
MEMBER_LW = 1.0                  # members under an emphasised aggregate
EMPHASIS_LW = 1.8                # the aggregate
REFERENCE_LW = 0.9               # dashed reference lines
MARKER_SIZE = 3
MARKER_EDGE_WIDTH = 0.6          # darker outline on a light series colour
SEPARATOR_LW = 0.8               # dotted separators between groups
SEPARATOR_COLOUR = '0.65'


class Marks(NamedTuple):
    """Series line width, marker size and marker edge width (pt)."""
    lw: float
    ms: float
    mew: float


# Full-size charts, and the lighter set for small-multiples grids, where the
# panels are small and many.
MARKS = Marks(DATA_LW, MARKER_SIZE, MARKER_EDGE_WIDTH)
GRID_MARKS = Marks(1.0, 2.2, 0.4)
GRID_SPINE_LW = 0.5
GRID_TICK_LEN = 2

# Background for a value label that must sit over data.
LABEL_BOX = dict(facecolor='white', edgecolor='none', pad=1.0)

RC = {
    'font.family':         'sans-serif',
    'font.sans-serif':     ['DejaVu Sans'],
    'font.size':           6.5,
    'axes.titlesize':      7.5,
    'axes.titleweight':    'normal',
    'axes.labelsize':      6.5,
    'xtick.labelsize':     6.5,
    'ytick.labelsize':     6.5,
    'legend.fontsize':     6.5,
    'figure.titlesize':    7.5,
    'axes.linewidth':      0.6,
    'xtick.major.width':   0.6,
    'ytick.major.width':   0.6,
    'xtick.major.size':    3,
    'ytick.major.size':    3,
    'lines.linewidth':     DATA_LW,
    'lines.markersize':    MARKER_SIZE,
    'mathtext.fontset':    'dejavusans',
    'axes.spines.top':     True,
    'axes.spines.right':   True,
    'axes.grid':           True,
    'grid.color':          '0.85',
    'grid.linewidth':      0.5,
    'grid.linestyle':      '-',
    'axes.axisbelow':      True,
    'legend.frameon':      True,
    'legend.framealpha':   1.0,
    'legend.facecolor':    'white',
    'legend.edgecolor':    '0.7',
    'legend.fancybox':     False,
    'legend.borderpad':    0.4,
    'legend.handlelength': 1.6,
    'savefig.dpi':         DPI,
    'hatch.linewidth':     0.6,
    'svg.fonttype':        'none',
}


def apply_style() -> None:
    mpl.rcParams.update(RC)


def new_figure(height_in: float, width_in: float = TEXT_WIDTH_IN,
               layout: str | None = 'constrained'):
    """Figure at its final printed size. layout=None for figures that set
    their own margins (e.g. broken axes, where constrained layout pads the
    break)."""
    return plt.figure(figsize=(width_in, height_in), layout=layout)


def save_figure(fig, out_path: Path) -> None:
    """PNG at DPI. The canvas is not cropped, so the saved width is the
    figure width and text prints at its rcParams size when inserted at 100%."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix('.png'), dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {out_path.with_suffix('.png')}")


def panel_suptitle(fig, title: str, **kwargs):
    """Figure title centred over the panels rather than the whole figure.

    fig.suptitle centres on the figure, which includes the strip a shared y
    label (fig.supylabel) and the tick labels take on the left, so over
    side-by-side panels it sits left of the panel titles. The title is drawn
    first so that constrained layout reserves room for it, then moved to the
    centre of the panels' combined extent, which moving it does not change.
    Call it after everything else is on the figure, since a legend or label
    added later moves the panels. kwargs go to fig.suptitle, e.g. y for a
    figure laid out with explicit margins.
    """
    fig.suptitle(title, **kwargs)
    fig.draw_without_rendering()
    x0 = min(ax.get_position().x0 for ax in fig.axes)
    x1 = max(ax.get_position().x1 for ax in fig.axes)
    return fig.suptitle(title, x=(x0 + x1) / 2, **kwargs)


def grid_axes(fig, n_rows: int, n_cols: int, xlabel: str):
    """Panels for a small-multiples grid, as an n_rows x n_cols array: shared
    axes, left and bottom spines only, and one x label centred under the
    bottom row.

    The x label belongs to an empty axes spanning the bottom row, so
    constrained layout reserves its room between the panels' tick labels and
    a figure legend below. fig.supxlabel shares the bottom strip with an
    'outside lower center' legend and the two overlap.
    """
    gs = fig.add_gridspec(n_rows, n_cols)
    axes = gs.subplots(sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_linewidth(GRID_SPINE_LW)
        ax.tick_params(width=GRID_SPINE_LW, length=GRID_TICK_LEN)
    span = fig.add_subplot(gs[-1, :])
    span.set_frame_on(False)
    span.patch.set_visible(False)
    span.grid(False)
    span.tick_params(left=False, bottom=False,
                     labelleft=False, labelbottom=False)
    span.set_navigate(False)
    # The span has no ticks, so its label is padded clear of the panels'.
    pad = (GRID_TICK_LEN + mpl.rcParams['xtick.major.pad']
           + 1.4 * mpl.rcParams['xtick.labelsize'])
    span.set_xlabel(xlabel, labelpad=pad)
    return axes
