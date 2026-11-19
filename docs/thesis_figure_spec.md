# Thesis figure format — Specification

## Scope

Default format for every matplotlib figure inserted in the thesis.
Implemented in `analysis/scripts/thesis_style.py`: `apply_style()` sets the
rcParams, `new_figure(height_in)` creates the figure at printed size,
`save_figure(fig, path)` writes it, `panel_suptitle(fig, title)` draws a
multi-panel figure title, `grid_axes(fig, n_rows, n_cols, xlabel)` creates
the panels of a small-multiples grid, and the module constants give the
colours, line widths and annotation size below. Scripts in
`analysis/scripts` import it as `thesis_style`; scripts elsewhere as
`analysis.scripts.thesis_style`.

Reference implementations:

- `analysis/scripts/alt_capped_price_figure.py`: two panels, broken axis,
  band.
- `energy_masters_pipeline/scripts/sweep_c_deg.py::plot_thesis_throughput`:
  single panel, categorical series under an aggregate.
- `energy_masters_pipeline/scripts/chart_cohort_rungs.py::chart_thesis_ladders`:
  bar chart with ordered categories, variants and separators.
- `energy_masters_pipeline/scripts/chart_cohort_rungs.py::chart_thesis_model_vs_amber`:
  scatter panels with area-scaled markers, size keys and a shared legend.

## Size and output

| Item | Value |
|---|---|
| Width | 15.92 cm / 6.27 in: A4 text width with 2.54 cm margins |
| Height | per figure: 2.0 in for an illustrative row of panels, 3.0 in for a single results chart, more only where the content needs it |
| File | PNG, 300 dpi |
| Cropping | none: the saved image is exactly the figure size |
| Insertion | at 100 % in Word, so printed text sizes equal the sizes below |

Layout is matplotlib constrained layout (`new_figure` default), or explicit
margins (`new_figure(..., layout=None)` and `gridspec_kw`) where constrained
layout cannot place the elements, e.g. broken axes.

## Text

DejaVu Sans, regular weight throughout. Mathtext uses the DejaVu Sans set,
so symbols match the surrounding text.

| Element | Size (pt) |
|---|---|
| Figure or panel title | 7.5 |
| Panel subtitle (under a figure title) | 6.5 |
| Axis label | 6.5 |
| Tick label | 6.5 |
| Legend | 6.5 |
| Annotation | 6.5 |
| Data notes (presentation versions only) | 5.5 |

Sentence case for titles, labels and legend entries. Thousands separators
on numbers of four or more digits in annotations. Dollar signs escaped
(`\$`) so matplotlib does not read them as mathtext. Model parameters are
written in the thesis notation as mathtext, e.g. `$c_{\mathrm{deg}}$`.

## Titles and captions

Every figure carries a short title naming what is plotted:

- single panel: `ax.set_title`, e.g. "LP vs actual battery throughput by
  $c_{\mathrm{deg}}$";
- multiple panels: a figure title (`panel_suptitle`, centred over the
  panels rather than the whole figure, which includes the y label strip)
  naming what the panels show together, e.g. "Model vs Amber value capture
  ratio", and a subtitle
  per panel naming what distinguishes it: the quantity, with the unit in
  brackets where the y axis has no label (e.g. "Import price (c/kWh) with
  cap"), or the subset (e.g. "Whole period, by site (n = 8)").

The caption in the thesis holds:

1. a first sentence naming what is shown;
2. the notes: data, period, method choices that affect reading the figure,
   and "illustrative" for constructed examples.

No notes or footnotes are drawn in the thesis figure. Context already given
in the text leading up to the figure (network area, offer) need not be
repeated in the figure.

A figure also used in presentations may carry the data notes (data, period,
weighting, what a range spans) in its presentation version, with a switch
that leaves them off for the thesis: grey 0.3 at the data note size, on a
white box with no edge (`LABEL_BOX`), in the corner opposite the legend, or
in the bottom right of each panel where panels share a legend below them.
In a small-multiples grid the notes are the panel's site count alone, top
left of each panel, and the rest go in the figure title.

## Axes

- Frame: all four spines, 0.6 pt black; left and bottom only in a
  small-multiples grid.
- Ticks: outward, 3 pt long, 0.6 pt.
- Grid: solid, 0.5 pt, grey 0.85, behind the data, on both axes; y only on
  bar charts.
- Axis labels: `Quantity (unit)`; a dimensionless ratio has no unit. Where
  each panel title carries the unit, the y axis has no label.
- Shared axes: panels showing the same quantity share their range; inner
  panels drop their tick labels. A y label common to side-by-side panels is
  drawn once (`fig.supylabel`); each panel keeps its own x label, except in
  a small-multiples grid.
- Sweep axes: ticks at the swept values only, labelled with `:g`, with 3 %
  padding either side.
- Time-of-day axes: 00:00 to 24:00, ticks every 6 hours.
- Side-by-side panels are spaced so the last tick label of one panel and
  the first of the next do not touch.

Broken axes: two stacked axes, the lower band holding the bulk of the data
and the upper band the excursion, height ratio 1:4, with a small diagonal
mark on both side spines at the break. Each band shows at least one tick
label.

## Small-multiples grids

A grid of more than two panels sharing both axes (e.g. states, years, or
states by years) has lighter marks and frame than a single chart. A pair of
panels keeps the single-chart style.

| Element | Style |
|---|---|
| Panel | about 1.4 times as wide as it is tall, e.g. 0.82 in high in a grid five panels wide |
| Frame | left and bottom spines, 0.5 pt; ticks 2 pt long, 0.5 pt |
| Series (`GRID_MARKS`) | lines 1.0 pt, markers 2.2 pt, marker edge 0.4 pt |
| Panel title | what distinguishes the panel only, e.g. "NSW 2021", at the panel subtitle size |
| Tick labels | on the outer panels only |
| Axis labels | each drawn once: x centred under the bottom row, y with `fig.supylabel` |
| Site count | a data note, top left of each panel (presentation versions only) |
| Legend | one figure legend below the panels |

## Marks

| Mark | Colour | Width / size |
|---|---|---|
| Single or primary series | `LINE_COLOUR` `#2a78d6` | 1.3 pt |
| Categorical members (sites, years) | `CATEGORICAL_COLOURS`, assigned in sorted order | 1.0 pt |
| Aggregate over its members (pooled, mean) | black, round markers at the computed points | 1.8 pt, markers 3 pt |
| Reference line (cap, threshold, target, derived value) | grey 0.3, dashed | 0.9 pt |
| Range band (IQR etc.) | series colour, alpha 0.22, no edge | — |

Scenario and retailer colours come from one mapping and are the same in
every figure. A reference line is drawn only in the panels it applies to.

## Bar charts

| Element | Style |
|---|---|
| Bar | width 0.72 of the category spacing, no edge |
| Ordered categories (best to worst information, etc.) | sequential blue ramp, darkest first |
| Variant of the bar before it | parent's colour, white hatch (`///`, then `xxx`), hatch 0.6 pt |
| Observed data beside modelled bars | its scenario colour (Amber `ACTUAL_COLOUR`) |
| Value label | black, annotation size, two decimals, centred 0.02 above the bar |
| Separator between groups | dotted, `SEPARATOR_COLOUR` grey 0.65, 0.8 pt, in a gap of 0.375 |
| Separator before observed data | dotted, grey 0.3, 0.8 pt |
| Bar clipped at the axis limit | white triangle on the limit; the CSV keeps the true value |

Grid on y only; x ticks have no tick marks. Tick labels on two lines where
one would crowd. Every bar carries its value label. A category keeps the
same tick label in every thesis figure it appears in; working figures not
in the thesis (e.g. per-site grids) may use shorter labels.

## Scatter plots

| Element | Style |
|---|---|
| Point | colour by category (`CATEGORICAL_COLOURS` for sites), alpha 0.8, white edge 0.5 pt |
| Marker area | proportional to the weighting quantity (e.g. headroom), never to its square root |
| Size key | bottom right of each panel, three reference values in one row, titled with the quantity and its unit of observation (e.g. "Site-day headroom"); each panel has its own scale where the quantity differs by panel; the box clears the largest marker and each marker is level with its label |
| Fitted line | dashed, grey 0.3, 0.9 pt, "Least squares fit" |
| Zero lines | where the range spans zero: solid, grey 0.65, 0.5 pt |
| Point outside the window | open triangle on the boundary, counted in the legend |
| Statistics (r, ρ) | top left, `r = 0.94` form, annotation size, white background (`LABEL_BOX`); n in the panel title |

Where x and y are the same quantity the panel is square (`set_box_aspect(1)`)
with equal x and y ranges, padded at the low end where needed to keep the
size key clear of the points.

## Legend

Inside the axes, in an area clear of data, reference lines and value
labels; opaque white box, 0.7 grey edge, square corners. Entries common to
several panels (e.g. site colours) go in one figure legend below the panels
(`loc='outside lower center'`), one row, no box; legend markers there are a
fixed size. Two columns when
there are more than five entries; an aggregate entry sits at the foot of
the first column. Omitted when the figure has one series and the caption
names it.

## Annotations

Key values are labelled in the figure: those the text cites, and the value
a figure exists to establish (e.g. the crossing that sets a parameter). Bar
value labels are covered under Bar charts. Otherwise the label is in the
colour of the mark it labels, at annotation size, and sits
at the feature it identifies: beside a spike or step on the side clear of
other marks, at the crossing of two reference lines. A label with clear
space around it has no background; one that must sit over data takes a
white box with no edge (`LABEL_BOX`).

## Illustrative figures

Figures that explain a mechanism with constructed numbers rather than
model results. Everything above applies, with these differences:

- The constructed values are named constants at the top of the script, so
  the figure is edited there.
- Value axes whose magnitudes are arbitrary have no tick labels; the
  quantity and unit stay in the subtitle or axis label, and direction is
  labelled in the panel (e.g. "Import" above zero, "Export" below) in black,
  as axis text.
- Notes that explain what is happening (phases, windows) are grey 0.3. A
  note that applies over a span sits centred over a thin grey line (0.6 pt)
  marking the span, just above the marks it describes.
- Where one panel shows quantities and another the money they produce, the
  subtitles name both units, and each quantity keeps its colour in both
  panels.
- Marks that sit outside the mechanism being shown (e.g. flows outside the
  window that counts) are grey 0.78 and named in the legend.
- A window or period that defines the mechanism is shaded grey 0.93 and
  labelled at its top; the data shown extend past it on both sides.
- The caption ends "Illustrative."

Reference implementation: `analysis/scripts/alt_self_hedge_figure.py`.

## Files

Each figure is drawn by a named function or script that reads saved results
(CSV), so it can be restyled without re-running the model. Output goes to
`analysis/output/<topic>/` or the pipeline's charts folder under the file
name used in `thesis/figures/`, and is copied into `thesis/figures/` for
insertion.
