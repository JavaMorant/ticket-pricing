"""How a report READS: markdown tables and figures. Phases 1-3 only.

    md_table                a DataFrame as a GitHub markdown table
    save_fig / style / log_ticks / BRAND_COLOURS    the plot house style

Where a report gets WRITTEN, which data it ran on and what its banner says are all in
`phaseio.py` — one module, one answer, for all six phases. Nothing in either file does
any analysis: each phase owns its own numbers, so that reading `phase1.py` top to bottom
tells you everything Phase 1 does.

Phases 4-6 import none of this. They report tables they build by hand and draw nothing,
so matplotlib is never imported on their path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Headless: these scripts only ever write PNG files, and importing pyplot without this
# tries to open a window on a machine that may not have one.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

# Plot house style. Deliberately plain: these are research figures that go in a markdown
# report, not a dashboard. Legible beats decorated.
BRAND_COLOURS = {"Brand A": "#3d6bb3", "Brand B": "#c4622d"}
GREY = "#666666"
FIG_DPI = 130


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    """Render a DataFrame as a GitHub markdown table.

    Hand-rolled because pandas' own `.to_markdown()` needs the `tabulate` package, and a
    whole dependency for fifteen lines is not a trade worth making.
    """
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = [
        "| " + " | ".join(_cell(v, floatfmt) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body])


def _cell(value, floatfmt: str) -> str:
    """One markdown cell. bool before int, because in Python a bool IS an int."""
    if isinstance(value, str):
        # A raw pipe inside a cell splits the row into extra columns.
        return value.replace("|", "/")
    if value is None or value is pd.NaT:
        return ""
    if isinstance(value, bool | np.bool_):
        return "yes" if value else "no"
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, int | np.integer):
        return f"{int(value):,}"
    if isinstance(value, float | np.floating):
        return "" if np.isnan(value) else floatfmt.format(float(value))
    return str(value)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def save_fig(fig, out_dir: Path, name: str) -> Path:
    """Save and close a figure. Closing matters: a long run otherwise leaks them."""
    path = Path(out_dir) / name
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI)
    plt.close(fig)
    return path.resolve()


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    """The same axis treatment everywhere: title, labels, a faint horizontal grid."""
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def log_ticks(ax) -> None:
    """Plain numbers on log axes, at 1/2/5 per decade.

    Matplotlib's default labels every minor tick with 3 x 10^2 notation, which collides
    into an unreadable smear on a narrow range like 70..700 tickets — exactly the range
    these plots live in.
    """
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(labelsize=8)
