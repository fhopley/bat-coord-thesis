#!/usr/bin/env python3
"""
poster_build.py

Refreshes the figures and applies the page margins to the hand-edited poster
layout, leaving text content as edited. Pictures are matched by their
top-left position in the layout and replaced at their grid position; text
boxes are moved so their ink sits on the same left and right margins.

Grid (cm): ink margin 2.5 each side; three columns 17.13 wide at 2.5, 21.13,
39.76 (the third ends at 56.89). Wide figures (Figures 2 and 4) share a frame
at x = 2.5 - 0.1 and width 37.2, so their plot boxes align (poster_style).

    python3 analysis/scripts/poster_build.py \\
        --layout thesis/poster/poster_draft_4.pptx \\
        --out thesis/poster/poster_draft_5.pptx
"""
import argparse
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Cm

FIG = Path('thesis/poster/figures')
MARGIN = 2.5
RIGHT = 59.4 - MARGIN                    # 56.9
COL3 = 39.76
COL_W = 17.13
WIDE_LEFT = MARGIN - 0.1
WIDE_W = 37.2
TEXT_INSET = 0.25                        # autoshape zones
FIG_RIGHT = 38.9                         # right end of caption under wide figs

CM = 360000


def cmv(emu):
    return round(emu / CM, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--layout', type=Path,
                    default=Path('thesis/poster/poster_draft_4.pptx'))
    ap.add_argument('--out', type=Path,
                    default=Path('thesis/poster/poster_draft_5.pptx'))
    args = ap.parse_args(argv)
    prs = Presentation(str(args.layout))
    slide = prs.slides[0]

    def put(sh, left, width=None):
        sh.left = Cm(left)
        if width is not None:
            sh.width = Cm(width)

    def new_pic(old, file, left, width):
        top = old.top
        old._element.getparent().remove(old._element)
        im = Image.open(file)
        slide.shapes.add_picture(str(file), Cm(left), top, Cm(width),
                                 Cm(width * im.size[1] / im.size[0]))

    for sh in list(slide.shapes):
        t = sh.shape_type
        key = (cmv(sh.left), cmv(sh.top))
        if t == 9:                                    # temporary guide lines
            sh._element.getparent().remove(sh._element)
        elif t == 13:
            if key == (19.8, 12.2):                    # taxonomy
                w = RIGHT - sh.left / CM
                h = w * sh.height / sh.width
                sh.width, sh.height = Cm(w), Cm(h)
            elif key == (1.3, 33.6):                   # Figure 2
                new_pic(sh, FIG / 'poster_value_ratio_iqr_poster.png',
                        WIDE_LEFT, WIDE_W)
            elif key == (40.6, 33.5) or key == (39.8, 33.5):
                new_pic(sh, FIG / 'poster_event_exposure_poster.png',
                        COL3, COL_W)
            elif key == (40.6, 55.2) or key == (39.8, 55.2):
                new_pic(sh, FIG / 'poster_retailer_margin_spread_poster.png',
                        COL3, COL_W)
            elif key in ((2.5, 54.8), (2.4, 54.8)):    # Figure 4
                new_pic(sh, FIG / 'poster_capped_offers_poster.png',
                        WIDE_LEFT, WIDE_W)
        elif sh.has_text_frame:
            txt = sh.text_frame.text.strip()
            ml = sh.text_frame.margin_left / CM
            if txt.startswith('Why'):
                put(sh, MARGIN - TEXT_INSET)
            elif txt.startswith(('Impact', 'Conclusions')):
                put(sh, COL3 - TEXT_INSET, COL_W + 2 * TEXT_INSET)
            elif txt.startswith('Figure 1'):
                put(sh, sh.left / CM, RIGHT - sh.left / CM + ml)
            elif txt.startswith(('Figure 2', 'Figure 4')):
                put(sh, MARGIN - ml, FIG_RIGHT - MARGIN + ml)
            elif txt.startswith(('Figure 3', 'Figure 5')):
                put(sh, COL3 - ml, COL_W + 2 * ml)
            elif txt.startswith('1') and 'Sunwiz' in txt:
                put(sh, MARGIN - ml, RIGHT - MARGIN + 2 * ml)
            elif not txt and sh.height / CM < 0.5:     # yellow heading rules
                x = sh.left / CM
                put(sh, MARGIN if x < 10 else COL3)
        elif t == 1 and sh.height / CM < 0.5:          # rules without text
            x = sh.left / CM
            put(sh, MARGIN if x < 10 else COL3)

    prs.save(str(args.out))
    print('Saved', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
