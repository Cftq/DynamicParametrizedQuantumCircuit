#!/usr/bin/env python
# coding: utf-8
"""Shared plotting style and figure helpers for overparameterization scripts."""


import configparser
import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# APS / PRX-style figure settings loaded from plot.config
# ============================================================
_PLOT_MODULE_DIR = Path(__file__).resolve().parent
PLOT_CONFIG_PATH = _PLOT_MODULE_DIR / "plot.config"
if not PLOT_CONFIG_PATH.exists():
    PLOT_CONFIG_PATH = _PLOT_MODULE_DIR.parent / "plot.config"
if not PLOT_CONFIG_PATH.exists():
    PLOT_CONFIG_PATH = _PLOT_MODULE_DIR.parent.parent / "plot.config"

_DEFAULT_PLOT_CONFIG = {
    "figure": {
        "width_default": "double",
        "single_width_cm": "8.5",
        "single_height_cm": "6.2",
        "double_width_cm": "17.0",
        "double_height_cm": "8.5",
        "dpi": "600",
        "display_dpi": "150",
    },
    "save": {
        "numerical_png": "false",
        "numerical_pdf": "true",
        "circuit_png": "true",
        "circuit_pdf": "false",
        "pad_inches": "0.02",
    },
    "text": {
        "show_titles": "false",
        "show_redundant_layer_legends": "false",
        "base_font_size": "10",
        "title_font_size": "10",
        "axis_label_font_size": "13",
        "tick_label_font_size": "11",
        "legend_font_size": "12",
        "font_family": "serif",
        "font_serif": "STIXGeneral, Times New Roman, DejaVu Serif",
        "mathtext_fontset": "stix",
    },
    "axes": {
        "linewidth": "0.6",
        "labelpad": "2.5",
        "unicode_minus": "false",
        "tick_direction": "in",
        "tick_top": "true",
        "tick_right": "true",
        "tick_pad": "2.0",
        "tick_major_size": "3.0",
        "tick_minor_size": "1.8",
        "tick_major_width": "0.6",
        "tick_minor_width": "0.5",
        "margin_left": "0.14",
        "margin_right": "0.03",
        "margin_bottom": "0.17",
        "margin_top": "0.04",
        "outside_legend_margin_left": "0.14",
        "outside_legend_margin_right": "0.28",
        "outside_legend_margin_bottom": "0.17",
        "outside_legend_margin_top": "0.04",
    },
    "lines": {
        "linewidth": "1.0",
        "markersize": "4.0",
        "errorbar_capsize": "2.5",
    },
    "legend": {
        "frameon": "false",
        "handlelength": "1.4",
        "handletextpad": "0.4",
        "borderaxespad": "0.3",
        "labelspacing": "0.25",
        "columnspacing": "0.8",
    },
    "grid": {
        "linewidth": "0.4",
        "alpha": "0.25",
    },
}


def _load_plot_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read_dict(_DEFAULT_PLOT_CONFIG)
    config.read(PLOT_CONFIG_PATH, encoding="utf-8")
    return config


_PLOT_CONFIG = _load_plot_config()


def _cfg_str(section: str, key: str) -> str:
    return _PLOT_CONFIG.get(section, key)


def _cfg_float(section: str, key: str) -> float:
    return _PLOT_CONFIG.getfloat(section, key)


def _cfg_int(section: str, key: str) -> int:
    return _PLOT_CONFIG.getint(section, key)


def _cfg_bool(section: str, key: str) -> bool:
    return _PLOT_CONFIG.getboolean(section, key)


def _cfg_csv(section: str, key: str) -> list[str]:
    return [
        item.strip()
        for item in _cfg_str(section, key).split(",")
        if item.strip()
    ]


INCH_PER_CM = 1.0 / 2.54
FIGSIZE_SINGLE = (
    _cfg_float("figure", "single_width_cm") * INCH_PER_CM,
    _cfg_float("figure", "single_height_cm") * INCH_PER_CM,
)
FIGSIZE_DOUBLE = (
    _cfg_float("figure", "double_width_cm") * INCH_PER_CM,
    _cfg_float("figure", "double_height_cm") * INCH_PER_CM,
)
FIGURE_WIDTH_DEFAULT = _cfg_str("figure", "width_default")

SAVE_DPI = _cfg_int("figure", "dpi")
FIGURE_DPI = _cfg_int("figure", "display_dpi")
SAVEFIG_PAD_INCHES = _cfg_float("save", "pad_inches")

NUMERICAL_SAVE_PNG = _cfg_bool("save", "numerical_png")
NUMERICAL_SAVE_PDF = _cfg_bool("save", "numerical_pdf")
CIRCUIT_SAVE_PNG = _cfg_bool("save", "circuit_png")
CIRCUIT_SAVE_PDF = _cfg_bool("save", "circuit_pdf")

SHOW_FIGURE_TITLES = _cfg_bool("text", "show_titles")
SHOW_REDUNDANT_LAYER_LEGENDS = _cfg_bool("text", "show_redundant_layer_legends")

BASE_FONT_SIZE = _cfg_int("text", "base_font_size")
TITLE_FONT_SIZE = _cfg_int("text", "title_font_size")
AXIS_LABEL_FONT_SIZE = _cfg_int("text", "axis_label_font_size")
TICK_LABEL_FONT_SIZE = _cfg_int("text", "tick_label_font_size")
LEGEND_FONT_SIZE = _cfg_int("text", "legend_font_size")

FONT_FAMILY = _cfg_str("text", "font_family")
FONT_SERIF = _cfg_csv("text", "font_serif")
MATHTEXT_FONTSET = _cfg_str("text", "mathtext_fontset")

AXES_LINEWIDTH = _cfg_float("axes", "linewidth")
AXES_LABELPAD = _cfg_float("axes", "labelpad")
AXES_UNICODE_MINUS = _cfg_bool("axes", "unicode_minus")
TICK_DIRECTION = _cfg_str("axes", "tick_direction")
TICK_TOP = _cfg_bool("axes", "tick_top")
TICK_RIGHT = _cfg_bool("axes", "tick_right")
TICK_PAD = _cfg_float("axes", "tick_pad")
TICK_MAJOR_SIZE = _cfg_float("axes", "tick_major_size")
TICK_MINOR_SIZE = _cfg_float("axes", "tick_minor_size")
TICK_MAJOR_WIDTH = _cfg_float("axes", "tick_major_width")
TICK_MINOR_WIDTH = _cfg_float("axes", "tick_minor_width")

LINE_WIDTH = _cfg_float("lines", "linewidth")
MARKER_SIZE = _cfg_float("lines", "markersize")
ERRORBAR_CAPSIZE = _cfg_float("lines", "errorbar_capsize")

LEGEND_FRAMEON = _cfg_bool("legend", "frameon")
LEGEND_HANDLELENGTH = _cfg_float("legend", "handlelength")
LEGEND_HANDLETEXTPAD = _cfg_float("legend", "handletextpad")
LEGEND_BORDERAXESPAD = _cfg_float("legend", "borderaxespad")
LEGEND_LABELSPACING = _cfg_float("legend", "labelspacing")
LEGEND_COLUMNSPACING = _cfg_float("legend", "columnspacing")

GRID_LINEWIDTH = _cfg_float("grid", "linewidth")
GRID_ALPHA = _cfg_float("grid", "alpha")

_DEFAULT_AXES_MARGINS_PRX = {
    "left": _cfg_float("axes", "margin_left"),
    "right": _cfg_float("axes", "margin_right"),
    "bottom": _cfg_float("axes", "margin_bottom"),
    "top": _cfg_float("axes", "margin_top"),
}

_DEFAULT_AXES_MARGINS_PRX_OUTSIDE_LEGEND = {
    "left": _cfg_float("axes", "outside_legend_margin_left"),
    "right": _cfg_float("axes", "outside_legend_margin_right"),
    "bottom": _cfg_float("axes", "outside_legend_margin_bottom"),
    "top": _cfg_float("axes", "outside_legend_margin_top"),
}


def apply_plot_config() -> None:
    """Apply plot.config settings to matplotlib.rcParams."""
    matplotlib.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.serif": FONT_SERIF,
        "mathtext.fontset": MATHTEXT_FONTSET,
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_LABEL_FONT_SIZE,
        "ytick.labelsize": TICK_LABEL_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "figure.titlesize": TITLE_FONT_SIZE,
        "axes.linewidth": AXES_LINEWIDTH,
        "axes.labelpad": AXES_LABELPAD,
        "axes.unicode_minus": AXES_UNICODE_MINUS,
        "xtick.direction": TICK_DIRECTION,
        "ytick.direction": TICK_DIRECTION,
        "xtick.top": TICK_TOP,
        "ytick.right": TICK_RIGHT,
        "xtick.major.size": TICK_MAJOR_SIZE,
        "ytick.major.size": TICK_MAJOR_SIZE,
        "xtick.minor.size": TICK_MINOR_SIZE,
        "ytick.minor.size": TICK_MINOR_SIZE,
        "xtick.major.width": TICK_MAJOR_WIDTH,
        "ytick.major.width": TICK_MAJOR_WIDTH,
        "xtick.minor.width": TICK_MINOR_WIDTH,
        "ytick.minor.width": TICK_MINOR_WIDTH,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "errorbar.capsize": ERRORBAR_CAPSIZE,
        "legend.frameon": LEGEND_FRAMEON,
        "legend.handlelength": LEGEND_HANDLELENGTH,
        "legend.handletextpad": LEGEND_HANDLETEXTPAD,
        "legend.borderaxespad": LEGEND_BORDERAXESPAD,
        "legend.labelspacing": LEGEND_LABELSPACING,
        "legend.columnspacing": LEGEND_COLUMNSPACING,
        "grid.linewidth": GRID_LINEWIDTH,
        "grid.alpha": GRID_ALPHA,
        "savefig.dpi": SAVE_DPI,
        "figure.dpi": FIGURE_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": SAVEFIG_PAD_INCHES,
    })


apply_plot_config()


def _figsize_from_width(
    width: str = FIGURE_WIDTH_DEFAULT,
    *,
    height_cm: Optional[float] = None,
):
    if width == "single":
        base_w, base_h = FIGSIZE_SINGLE
    elif width == "double":
        base_w, base_h = FIGSIZE_DOUBLE
    else:
        raise ValueError("width must be either 'single' or 'double'.")

    if height_cm is None:
        return (base_w, base_h)
    return (base_w, float(height_cm) * INCH_PER_CM)


def apply_axes_prx(
    fig,
    ax=None,
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


def new_prx_figure(
    *,
    width: str = FIGURE_WIDTH_DEFAULT,
    height_cm: Optional[float] = None,
):
    """Create a figure with dimensions controlled by plot.config."""
    return plt.figure(figsize=_figsize_from_width(width, height_cm=height_cm))


def set_prx_title(title: str, *, ax=None) -> None:
    """Keep figure titles switchable; journal captions usually carry titles."""
    if not SHOW_FIGURE_TITLES:
        return
    if ax is None:
        plt.title(title)
    else:
        ax.set_title(title)


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


def apply_prx_axis_style(ax) -> None:
    """Apply configured font sizes to a single axis."""
    apply_fontsizes(ax)


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
        direction=TICK_DIRECTION,
        top=TICK_TOP,
        right=TICK_RIGHT,
        pad=TICK_PAD,
    )

    ax.tick_params(
        axis="both",
        which="major",
        length=TICK_MAJOR_SIZE,
        width=TICK_MAJOR_WIDTH,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        length=TICK_MINOR_SIZE,
        width=TICK_MINOR_WIDTH,
    )

    if use_minor_ticks:
        ax.minorticks_on()

    for spine in ax.spines.values():
        spine.set_linewidth(AXES_LINEWIDTH)

    ax.xaxis.labelpad = AXES_LABELPAD
    ax.yaxis.labelpad = AXES_LABELPAD

    if grid:
        ax.grid(True, axis=grid_axis, alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)
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
    style_legend_for_prx(ax, frameon=LEGEND_FRAMEON)

    root, _ = os.path.splitext(outpath)

    if save_png:
        fig.savefig(
            root + ".png",
            dpi=SAVE_DPI,
            bbox_inches="tight",
            pad_inches=SAVEFIG_PAD_INCHES,
        )

    if save_pdf:
        fig.savefig(
            root + ".pdf",
            bbox_inches="tight",
            pad_inches=SAVEFIG_PAD_INCHES,
        )

    plt.close(fig)


def save_current_figure(
    outpath: str,
    *,
    outside_legend: bool = False,
    legend_space_frac: Optional[float] = None,
    save_png: bool = NUMERICAL_SAVE_PNG,
    save_pdf: bool = NUMERICAL_SAVE_PDF,
) -> None:
    """Save the current matplotlib figure using plot.config appearance settings."""
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    fig = plt.gcf()
    if not SHOW_FIGURE_TITLES:
        for ax in fig.axes:
            ax.set_title("")

    apply_axes_prx(
        fig,
        outside_legend=outside_legend,
        legend_space_frac=legend_space_frac,
    )

    for ax in fig.axes:
        apply_prx_axis_style(ax)
        style_axes_for_prx(ax)
        style_legend_for_prx(ax, frameon=LEGEND_FRAMEON)

    root, ext = os.path.splitext(outpath)
    ext = ext.lower()
    if not ext:
        root = outpath
        outpath = root + ".png"
        ext = ".png"

    if save_png:
        png_path = outpath if ext == ".png" else root + ".png"
        fig.savefig(
            png_path,
            dpi=SAVE_DPI,
            bbox_inches="tight",
            pad_inches=SAVEFIG_PAD_INCHES,
        )

    if save_pdf:
        fig.savefig(
            root + ".pdf",
            bbox_inches="tight",
            pad_inches=SAVEFIG_PAD_INCHES,
        )

    plt.close(fig)


def make_violin_ready(
    x,
    *,
    ensure_positive: bool = False,
    tiny: float = 1e-12,
) -> np.ndarray:
    """Return a plotting-only array that is safe for matplotlib.violinplot."""
    arr = np.asarray(x, dtype=float).ravel()

    if arr.size == 0:
        base = tiny if ensure_positive else 0.0
        return np.array([base, base + tiny], dtype=float)

    if ensure_positive:
        arr = np.where(arr <= 0.0, tiny, arr)

    if arr.size == 1:
        base = arr[0]
        scale = max(abs(base), 1.0)
        delta = tiny * scale
        if ensure_positive:
            return np.array([base, base + delta], dtype=float)
        return np.array([base - 0.5 * delta, base + 0.5 * delta], dtype=float)

    if np.allclose(arr, arr[0], rtol=0.0, atol=0.0):
        base = arr[0]
        scale = max(abs(base), 1.0)
        delta = tiny * scale
        if ensure_positive:
            jitter = delta * np.linspace(0.0, 1.0, arr.size, dtype=float)
        else:
            jitter = delta * np.linspace(-0.5, 0.5, arr.size, dtype=float)
        return arr + jitter

    return arr


def style_violin(
    vp,
    *,
    facecolor=None,
    edgecolor=None,
    alpha: float = 0.20,
    linewidth: float = LINE_WIDTH,
    hatch: Optional[str] = None,
    linecolor=None,
    linealpha: float = 0.7,
) -> None:
    """Apply configured violin-plot styling to matplotlib's violinplot result."""
    for body in vp["bodies"]:
        if facecolor is not None:
            body.set_facecolor(facecolor)
        if edgecolor is not None:
            body.set_edgecolor(edgecolor)
        body.set_alpha(alpha)
        body.set_linewidth(linewidth)
        if hatch is not None:
            body.set_hatch(hatch)

    line_color = linecolor if linecolor is not None else (
        edgecolor if edgecolor is not None else "black"
    )

    for key in ["cmeans", "cmins", "cmaxes", "cbars", "cmedians"]:
        if key in vp:
            vp[key].set_color(line_color)
            vp[key].set_linewidth(linewidth)
            vp[key].set_alpha(linealpha)


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

