"""Shared typography for the paper's 6.5-inch figure export canvases.

The ICLR manuscript displays full-width plots at 5.5 inches: the exported 9
and 11 pt sizes become approximately 7.6 and 9.3 pt on the page. Dense
figures should reserve space for labels/legends rather than use smaller type.
Additional LaTeX scaling would also shrink these carefully sized labels.
"""
import matplotlib as mpl

FONT_SIZE = 10
LABEL_SIZE = 9
TICK_SIZE = 9
LEGEND_SIZE = 9
TITLE_SIZE = 11
ANNOTATION_SIZE = 9
FIGURE_WIDTH = 6.5


def apply_style():
    """Set legible paper-scale typography without changing plotted values."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": FONT_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "legend.title_fontsize": LEGEND_SIZE,
        "figure.titlesize": TITLE_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
