"""Shared matplotlib style for paper figures.

Palette: validated default from the dataviz skill (light mode):
  categorical slots 1-3: blue #2a78d6, orange #eb6834, aqua #1baf7a
  sequential blue ramp, diverging blue<->red with gray midpoint.
Ink and chrome follow the reference chart chrome.  All figures are light-mode
(paper surfaces), white page plane.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# categorical slots (fixed order, never cycled)
C1 = "#2a78d6"  # blue
C2 = "#eb6834"  # orange
C3 = "#1baf7a"  # aqua
C4 = "#eda100"  # yellow
C5 = "#e87ba4"  # magenta

# sequential blue ramp (light->dark), for magnitude
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]

# diverging blue <-> red with gray midpoint (for lift around 1.0)
DIVERGING = ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#7a1d1c"]
DIVERGING_CMAP = LinearSegmentedColormap.from_list("palette_diverging", DIVERGING)

# chrome
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"


def apply(style: str = "paper") -> None:
    """Set global matplotlib rcParams for light paper figures."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 8.0,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": SURFACE,
    })


def save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] {path}")
