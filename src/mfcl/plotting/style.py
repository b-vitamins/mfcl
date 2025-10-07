from __future__ import annotations

import matplotlib as mpl


def apply():
    mpl.rcParams.update(
        {
            "figure.figsize": (6.0, 4.0),
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "savefig.bbox": "tight",
        }
    )
