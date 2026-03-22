"""
Shared matplotlib defaults for publication-style distribution plots:
unfilled step histograms and ticks on all sides pointing inward (ATLAS-like).
Import and call apply_publication_style() once at startup (scripts) or in a notebook
before plotting.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Keyword args for overlaid normalized histograms (no fill).
STEP_HIST_KW = {"histtype": "step", "linewidth": 2}


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
        }
    )
