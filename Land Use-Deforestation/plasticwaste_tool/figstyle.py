"""
One figure style for every plate in the methodology document.

WHY A SHARED MODULE RATHER THAN RCPARAMS IN EACH NOTEBOOK

Figures from different notebooks end up side by side in one PDF. If each
notebook sets its own fonts and sizes, the document looks assembled rather
than written, and a reader who notices the seams starts wondering what else
was stitched together. One import fixes that.

The palette is colour-blind safe (Okabe & Ito 2008, "Color Universal Design"),
because roughly 1 in 12 men has some form of colour vision deficiency and a
peer-reviewed methodology document should not be unreadable to them. Where a
figure encodes an ordered quantity it uses `viridis`, which is perceptually
uniform and also survives greyscale printing.

Figures are saved at 200 dpi: sharp in print at the single-column widths used
here, without producing files too large to email.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIGDIR = Path(__file__).resolve().parent / "manual_qa" / "figures"

# Okabe-Ito, assigned to the meanings this module uses repeatedly so that
# "assessed" is the same blue in every plate it appears in.
OKABE_ITO = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermillion": "#D55E00", "sky": "#56B4E9", "yellow": "#F0E442",
    "purple": "#CC79A7", "black": "#000000", "grey": "#999999",
}

VERDICT_COLOURS = {
    "valid": OKABE_ITO["blue"],
    "no_water_mask": OKABE_ITO["orange"],
    "implausible": OKABE_ITO["vermillion"],
    "guard_blocked": OKABE_ITO["grey"],
}

SCENARIO_COLOURS = {
    "waw_2007": OKABE_ITO["sky"],
    "conakry_recent": OKABE_ITO["vermillion"],
}


def use() -> None:
    """Apply the document style. Call once per notebook."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.autolayout": False,
    })
    FIGDIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str, caption: str = "") -> Path:
    """
    Write a figure and print the path plus its caption.

    The caption is printed rather than drawn onto the figure: a caption baked
    into the image cannot be edited during document review, cannot be
    translated, and will not reflow. It belongs in the document, and printing
    it here means it travels with the figure instead of being reinvented.
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path)
    print(f"saved  {path.relative_to(FIGDIR.parent.parent)}")
    if caption:
        print(f"caption: {caption}")
    return path


def thousands(ax, axis: str = "y") -> None:
    """Thousands separators -- 18,166 reads faster than 18166."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


__all__ = ["use", "save", "thousands", "FIGDIR",
           "OKABE_ITO", "VERDICT_COLOURS", "SCENARIO_COLOURS", "plt"]
