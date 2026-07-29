#!/usr/bin/env python
# coding: utf-8
"""Shared plotting style and figure helpers for overparameterization scripts."""


import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# APS / PRX-style figure settings
# ============================================================
INCH_PER_CM = 1.0 / 2.54
FIGSIZE_SINGLE = (8.5 * INCH_PER_CM, 6.2 * INCH_PER_CM)
FIGSIZE_DOUBLE = (17.0 * INCH_PER_CM, 8.5 * INCH_PER_CM)
FIGURE_WIDTH_DEFAULT = "double"

SAVE_DPI = 600

# Numerical result figures: PDF only
NUMERICAL_SAVE_PNG = False
NUMERICAL_SAVE_PDF = True

# Quantum circuit figures: PNG only
CIRCUIT_SAVE_PNG = True
CIRCUIT_SAVE_PDF = False

SHOW_FIGURE_TITLES = False

BASE_FONT_SIZE = 10
TITLE_FONT_SIZE = 10
AXIS_LABEL_FONT_SIZE = 13
TICK_LABEL_FONT_SIZE = 11
LEGEND_FONT_SIZE = 12

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": BASE_FONT_SIZE,
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_FONT_SIZE,
    "xtick.labelsize": TICK_LABEL_FONT_SIZE,
    "ytick.labelsize": TICK_LABEL_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "figure.titlesize": TITLE_FONT_SIZE,

    "axes.linewidth": 0.6,
    "axes.labelpad": 2.5,
    "axes.unicode_minus": False,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,

    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,

    "lines.linewidth": 1.0,
    "lines.markersize": 4.0,
    "errorbar.capsize": 2.5,

    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.3,
    "legend.labelspacing": 0.25,
    "legend.columnspacing": 0.8,

    "grid.linewidth": 0.4,
    "grid.alpha": 0.25,

    "savefig.dpi": SAVE_DPI,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

_DEFAULT_AXES_MARGINS_PRX = {
    "left": 0.14,
    "right": 0.03,
    "bottom": 0.17,
    "top": 0.04,
}

_DEFAULT_AXES_MARGINS_PRX_OUTSIDE_LEGEND = {
    "left": 0.14,
    "right": 0.28,
    "bottom": 0.17,
    "top": 0.04,
}


def _figsize_from_width(width: str = FIGURE_WIDTH_DEFAULT):
    if width == "single":
        return FIGSIZE_SINGLE
    if width == "double":
        return FIGSIZE_DOUBLE
    raise ValueError("width must be either 'single' or 'double'.")


def apply_axes_prx(
    fig,
    ax,
    *,
    outside_legend: bool = False,
    legend_space_frac: Optional[float] = None,
    margins: Optional[dict] = None,
) -> None:
    if margins is None:
        margins = (
            _DEFAULT_AXES_MARGINS_PRX_OUTSIDE_LEGEND
            if outside_legend
            else _DEFAULT_AXES_MARGINS_PRX
        )
    else:
        margins = dict(margins)

    if legend_space_frac is not None and outside_legend:
        margins["right"] = max(
            float(margins.get("right", 0.03)),
            float(legend_space_frac),
        )

    left = float(margins.get("left", 0.14))
    right = float(margins.get("right", 0.03))
    bottom = float(margins.get("bottom", 0.17))
    top = float(margins.get("top", 0.04))

    fig.subplots_adjust(
        left=left,
        right=max(left + 0.05, 1.0 - right),
        bottom=bottom,
        top=min(0.995, 1.0 - top),
    )


def new_fig_ax(
    *,
    outside_legend: bool = False,
    legend_space_frac: Optional[float] = None,
    margins: Optional[dict] = None,
    width: str = FIGURE_WIDTH_DEFAULT,
):
    fig, ax = plt.subplots(figsize=_figsize_from_width(width))
    apply_axes_prx(
        fig,
        ax,
        outside_legend=outside_legend,
        legend_space_frac=legend_space_frac,
        margins=margins,
    )
    return fig, ax


def apply_fontsizes(
    ax,
    *,
    title_size: int = TITLE_FONT_SIZE,
    label_size: int = AXIS_LABEL_FONT_SIZE,
    tick_size: int = TICK_LABEL_FONT_SIZE,
    legend_size: int = LEGEND_FONT_SIZE,
) -> None:
    ax.title.set_fontsize(title_size)
    ax.xaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontsize(label_size)
    ax.tick_params(axis="both", labelsize=tick_size)

    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontsize(legend_size)


def style_axes_for_prx(
    ax,
    *,
    use_minor_ticks: bool = True,
    grid_axis: str = "y",
    grid: bool = True,
) -> None:
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        pad=2.0,
    )

    ax.tick_params(axis="both", which="major", length=3.0, width=0.6)
    ax.tick_params(axis="both", which="minor", length=1.8, width=0.5)

    if use_minor_ticks:
        ax.minorticks_on()

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    ax.xaxis.labelpad = 2.5
    ax.yaxis.labelpad = 2.5

    if grid:
        ax.grid(True, axis=grid_axis, alpha=0.25, linewidth=0.4)
    else:
        ax.grid(False)


def style_legend_for_prx(
    ax,
    *,
    frameon: bool = False,
):
    leg = ax.get_legend()

    if leg is None:
        return None

    leg.set_frame_on(frameon)

    if frameon:
        leg.get_frame().set_linewidth(0.4)
        leg.get_frame().set_alpha(0.9)

    for txt in leg.get_texts():
        txt.set_fontsize(LEGEND_FONT_SIZE)

    return leg


def save_fig(
    fig,
    ax,
    outpath: str,
    *,
    outside_legend: bool = False,
    legend_space_frac: Optional[float] = None,
    save_png: bool = NUMERICAL_SAVE_PNG,
    save_pdf: bool = NUMERICAL_SAVE_PDF,
) -> None:
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    if not SHOW_FIGURE_TITLES:
        ax.set_title("")

    apply_axes_prx(
        fig,
        ax,
        outside_legend=outside_legend,
        legend_space_frac=legend_space_frac,
    )
    apply_fontsizes(ax)
    style_axes_for_prx(ax)
    style_legend_for_prx(ax, frameon=False)

    root, _ = os.path.splitext(outpath)

    if save_png:
        fig.savefig(
            root + ".png",
            dpi=SAVE_DPI,
            bbox_inches="tight",
            pad_inches=0.02,
        )

    if save_pdf:
        fig.savefig(
            root + ".pdf",
            bbox_inches="tight",
            pad_inches=0.02,
        )

    plt.close(fig)


def plot_qfim_grad_alignment_table(
    table,
    *,
    title,
    outpath,
    log_x=True,
    log_y=False,
    eig_floor=1e-16,
    weight_floor=1e-16,
    color_by=None,
    point_size=16.0,
    alpha=0.50,
    annotate_top_k=0,
):
    """Plot gradient coefficients in the QFIM eigenbasis."""
    lambdas = np.asarray(table["lambda"], dtype=float)
    weights = np.asarray(table["w_grad"], dtype=float)

    finite = (
        np.isfinite(lambdas)
        & np.isfinite(weights)
        & (lambdas >= 0.0)
        & (weights >= 0.0)
    )

    x = np.maximum(lambdas[finite], eig_floor)
    y_raw = weights[finite]
    y = np.maximum(y_raw, weight_floor) if log_y else y_raw

    fig, ax = new_fig_ax(outside_legend=False)

    if color_by is not None and color_by in table:
        color_values = np.asarray(table[color_by])[finite]

        sc = ax.scatter(
            x,
            y,
            c=color_values,
            cmap="viridis",
            s=point_size,
            alpha=alpha,
            edgecolors="none",
        )

        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_by.replace("_", " "))
    else:
        ax.scatter(
            x,
            y,
            s=point_size,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.25,
        )

    if annotate_top_k > 0 and "eig_index" in table:
        eig_indices = np.asarray(table["eig_index"])[finite]
        top_order = np.argsort(y_raw)[::-1][:annotate_top_k]

        for loc in top_order:
            ax.annotate(
                str(int(eig_indices[loc])),
                xy=(x[loc], y[loc]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
            )

    if log_x:
        ax.set_xscale("log")

    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel("QFIM eigenvalue")
    ax.set_ylabel("Gradient coeff.")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.30)

    style_axes_for_prx(ax, grid_axis="both", grid=True)

    save_fig(fig, ax, outpath, outside_legend=False)


def plot_qfim_grad_alignment_layer_overlay(
    table_by_layer,
    layers,
    *,
    title,
    outpath,
    log_x=True,
    log_y=False,
    eig_floor=1e-16,
    weight_floor=1e-16,
    point_size=14.0,
    alpha=0.45,
):
    """Overlay QFIM-gradient alignment scatter plots for multiple layers."""
    valid_layers = [
        int(L)
        for L in layers
        if table_by_layer.get(int(L)) is not None
    ]

    if not valid_layers:
        return

    overlay_cmap = matplotlib.colormaps.get_cmap("viridis")

    fig, ax = new_fig_ax(outside_legend=True)

    for layer_idx, L in enumerate(valid_layers):
        table = table_by_layer[L]

        lambdas = np.asarray(table["lambda"], dtype=float)
        weights = np.asarray(table["w_grad"], dtype=float)

        finite = (
            np.isfinite(lambdas)
            & np.isfinite(weights)
            & (lambdas >= 0.0)
            & (weights >= 0.0)
        )

        x = np.maximum(lambdas[finite], eig_floor)
        y_raw = weights[finite]
        y = np.maximum(y_raw, weight_floor) if log_y else y_raw

        color = overlay_cmap(layer_idx / max(len(valid_layers) - 1, 1))

        ax.scatter(
            x,
            y,
            s=point_size,
            alpha=alpha,
            color=color,
            edgecolors="none",
            label=f"L={L}",
        )

    if log_x:
        ax.set_xscale("log")

    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel("QFIM eigenvalue")
    ax.set_ylabel("Gradient coeff.")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.30)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
    )

    style_axes_for_prx(ax, grid_axis="both", grid=True)

    save_fig(fig, ax, outpath, outside_legend=True)
