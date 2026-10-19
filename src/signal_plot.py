"""Styling and plotting helpers for the test analysis notebook (05).

All the color/marker styling and matplotlib plotting logic lives here so the
notebook itself only keeps the things that matter for an experiment: the
ticker, the date range and which views to render. The notebook builds the
aligned ``(date, signal, price)`` frame and then calls into these helpers.

The styling is scheme-agnostic: it is driven entirely by the ordered class
list (most-bearish -> most-bullish) recovered from ``feature_metadata.json``,
so it works for the 3class (``SELL / HOLD / BUY``) and 5class
(``SELL_100% ... BUY_1000USD``) schemes, and degrades gracefully for any other
ordered class list.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HOLD_NAME = "HOLD"

# Explicit palettes per scheme (sell-side -> HOLD -> buy-side).
PALETTES: dict[str, dict[str, str]] = {
    "5class": {
        "SELL_100%": "darkred",
        "SELL_50%": "red",
        "HOLD": "blue",
        "BUY_500USD": "lightgreen",
        "BUY_1000USD": "darkgreen",
    },
    "3class": {
        "SELL": "red",
        "HOLD": "blue",
        "BUY": "green",
    },
}


def _fallback_palette(
    ordered_classes: Sequence[str], hold_name: str = HOLD_NAME
) -> dict:
    """Red -> blue -> green gradient for an arbitrary ordered class list."""
    hold_idx = (
        list(ordered_classes).index(hold_name)
        if hold_name in ordered_classes
        else len(ordered_classes) // 2
    )
    n_sell = max(hold_idx, 1)
    n_buy = max(len(ordered_classes) - hold_idx - 1, 1)
    reds = plt.cm.Reds(np.linspace(0.9, 0.5, n_sell))
    greens = plt.cm.Greens(np.linspace(0.5, 0.9, n_buy))
    palette = {}
    for i, name in enumerate(ordered_classes):
        if name == hold_name:
            palette[name] = "blue"
        elif i < hold_idx:
            palette[name] = reds[i]
        else:
            palette[name] = greens[i - hold_idx - 1]
    return palette


def build_style_map(
    ordered_classes: Sequence[str],
    scheme: str,
    hold_name: str = HOLD_NAME,
    logger=None,
) -> dict[str, dict]:
    """Per-class style map ``{name: {color, marker, size, alpha}}``.

    Classes before ``hold_name`` are sell-side (``v``), classes after are
    buy-side (``^``) and ``hold_name`` is a small-ish, transparent ``o`` since
    it dominates. Colors come from the explicit per-scheme palette when it
    matches the class set, otherwise from a red -> blue -> green gradient.
    """
    palette = PALETTES.get(scheme)
    if palette is None or set(palette) != set(ordered_classes):
        if logger is not None:
            logger.info("No explicit palette for scheme %r; using gradient fallback.", scheme)
        palette = _fallback_palette(ordered_classes, hold_name)
    hold_idx = (
        list(ordered_classes).index(hold_name)
        if hold_name in ordered_classes
        else len(ordered_classes) // 2
    )
    styles = {}
    for i, name in enumerate(ordered_classes):
        if name == hold_name:
            marker, size, alpha = "o", 70, 0.25
        elif i < hold_idx:
            marker, size, alpha = "v", 70, 0.9
        else:
            marker, size, alpha = "^", 70, 0.9
        styles[name] = {"color": palette[name], "marker": marker, "size": size, "alpha": alpha}
    return styles


def plot_signals(
    price_win: pd.DataFrame,
    sig_win: pd.DataFrame,
    classes_to_show: Sequence[str],
    feature_metadata: dict,
    ticker: str,
    title: str,
):
    """Plot the price line with predicted signals scattered on top.

    - ``price_win`` - columns ``date`` and ``price`` (the black line).
    - ``sig_win`` - columns ``date``, ``price`` and ``signal`` (class name).
    - ``classes_to_show`` - which classes to scatter (and list in the legend).
    - ``style_map`` - output of :func:`build_style_map`.

    The legend lists every requested class (with its count in the window),
    even classes that never occur in the current selection.
    """

    LABEL_SCHEME = feature_metadata.get("label_scheme")
    ORDERED_CLASSES = feature_metadata.get("ordered_classes")

    
    style_map = build_style_map(classes_to_show, LABEL_SCHEME)

    for name in ORDERED_CLASSES:
        s = style_map[name]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(
        price_win["date"], price_win["price"],
        color="black", linewidth=1.2, zorder=1,
    )
    for name in classes_to_show:
        st = style_map[name]
        pts = sig_win[sig_win["signal"] == name]
        if pts.empty:
            continue
        ax.scatter(
            pts["date"], pts["price"],
            c=[st["color"]], marker=st["marker"], s=st["size"],
            alpha=st["alpha"], edgecolors="none", zorder=3,
        )

    # Custom legend so every requested class is listed (even with 0 occurrences).
    handles = [Line2D([0], [0], color="black", lw=1.2, label=f"{ticker} price")]
    for name in classes_to_show:
        st = style_map[name]
        cnt = int((sig_win["signal"] == name).sum())
        handles.append(
            Line2D(
                [0], [0], marker=st["marker"], linestyle="none",
                markerfacecolor=st["color"], markeredgecolor=st["color"],
                markersize=15, label=f"{name} ({cnt})",
            )
        )
    ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Adj Close price")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()
    return fig, ax
