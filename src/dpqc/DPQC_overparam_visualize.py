#!/usr/bin/env python
# coding: utf-8
"""Visualize saved DPQC overparameterization numerical results.

Run DPQC_overparam_compute.py first to create the .npz files under
figs/dpqc/h_<h_param>/numerical_results. This script loads those saved
results and generates the figures without recomputing VQE/QFIM quantities.
"""


import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple


# Support direct execution from any working directory.  Shared configuration,
# plotting helpers, and numerical utilities live in ``src/common`` rather than
# beside this visualization entry point.
_MODULE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _MODULE_DIR.parent
_COMMON_DIR = _SRC_DIR / "common"
for _path in (_MODULE_DIR, _COMMON_DIR):
    _path_string = str(_path)
    if _path_string not in sys.path:
        sys.path.insert(0, _path_string)


import config_overparam as cfg

# ------------------------------------------------------------
# IMPORTANT: env vars should be set BEFORE importing jax
# ------------------------------------------------------------
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.ticker as mticker
import numpy as np
import tensorcircuit as tc
from matplotlib.patches import Patch
from plot import (
    new_fig_ax,
    plot_qfim_grad_alignment_layer_overlay,
    plot_qfim_grad_alignment_table,
    save_fig,
    style_axes_for_prx,
)
from tqdm.auto import tqdm

jax.config.update("jax_enable_x64", True)

tc.set_backend("jax")
tc.set_dtype("complex128")

REAL_DTYPE = jnp.float64
COMPLEX_DTYPE = jnp.complex128
NP_REAL_DTYPE = np.float64
NP_COMPLEX_DTYPE = np.complex128
NP_INT_DTYPE = np.int64

from dpqc_overparam_common import (
    _thr_tag,
    build_H_matrix_jax,
    build_layer_list,
    hamiltonian_terms,
    load_npz_result,
    rho_zero_state,
    threshold_psd_eigvals_for_rank,
)

# ============================================================
# Shared constants / helpers
# ============================================================
num_system_qubits = 5
h_param = cfg.H_PARAM
tolerance = cfg.TOLERANCE
steps = cfg.STEPS
num_runs = int(cfg.NUM_RUNS)
if num_runs <= 0:
    raise ValueError("cfg.NUM_RUNS must be a positive integer.")
lr = cfg.LEARNING_RATE

# Optimization-history sampling points used for history plots and
# QFIM-gradient sector diagnostics.
eps = 1e-12
sample_every = cfg.SAMPLE_EVERY

# Upper cutoff for the zoomed final-energy-error distribution.  The h=0.1
# results have a dense lowest-error cloud below 0.6; changing this value
# regenerates the detailed figure with a different cutoff.
FINAL_ENERGY_ERROR_DETAIL_THRESHOLD = 6e-1

# Optimization-history sampling points used for history plots and
# QFIM-gradient sector diagnostics.
#
# qfim_grad_weight_scatter is generated at each of these iterations.
# We intentionally use iteration 1 instead of iteration 0 because the
# user request is for optimized-path parameters at
#   1, 1000, 2000, ..., 10000.
sample_iters = np.asarray(
    [1] + list(range(sample_every, steps + 1, sample_every)),
    dtype=NP_INT_DTYPE,
)

sample_iters = sample_iters[
    (sample_iters >= 1) & (sample_iters <= steps)
]

sample_iters = np.unique(sample_iters).astype(NP_INT_DTYPE)
sample_iter_set = set(int(t) for t in sample_iters.tolist())

NUM_BLOCKS = 4
PARAMS_PER_BLOCK = 3
EXTRA_PARAMS_PER_LAYER = 2
n_param_per_layer = NUM_BLOCKS * PARAMS_PER_BLOCK + EXTRA_PARAMS_PER_LAYER

TOP, LEFT, RIGHT, BOTTOM, ANC_CENTER, FRESH_ANCILLA = 0, 1, 2, 3, 4, 5

LAYER_PAIRS = (
    (LEFT, BOTTOM),
    (RIGHT, BOTTOM),
    (TOP, RIGHT),
    (TOP, ANC_CENTER),
)

RED4_COLOR = "blue"

# Reduced-system QFIM identifiers used in filenames and figure titles.
# Define these near the top so later cells/sections cannot hit NameError.
keep_key = "keep0123"
keep_label = "Reduced (0,1,2,3)"
keep_key_5 = "keep01234"
keep_label_5 = "Reduced (0,1,2,3,4)"


def make_layer_legend_handles(layer_list, cmap, *, alpha=0.25):
    n = max(len(layer_list), 1)

    return [
        Patch(
            facecolor=cmap(idx / n),
            edgecolor=cmap(idx / n),
            alpha=alpha,
            label=f"L{L}",
        )
        for idx, L in enumerate(layer_list)
    ]


def violin_width_from_positions(positions, *, scale=0.60, default=0.75):
    positions = np.asarray(positions, dtype=NP_REAL_DTYPE)

    return (
        scale * np.min(np.diff(np.sort(positions)))
        if positions.size > 1
        else default
    )


def style_violin(vp, color, *, alpha=0.20, lw=1.0, line_color=None):
    for body in vp["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(alpha)
        body.set_linewidth(lw)

    line_color = color if line_color is None else line_color

    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        artist = vp.get(key)
        if artist is not None:
            artist.set_color(line_color)
            artist.set_linewidth(lw)


def style_violin_bodies(vp, colors, *, alpha=0.20, lw=1.0, line_color=None):
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(alpha)
        body.set_linewidth(lw)

    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        artist = vp.get(key)
        if artist is not None:
            if line_color is not None:
                artist.set_color(line_color)
            artist.set_linewidth(lw)


def style_boxplot(bp, color, *, alpha=0.20, lw=1.0):
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(alpha)
        box.set_linewidth(lw)

    for key in ("whiskers", "caps", "medians"):
        for artist in bp[key]:
            artist.set_color(color)
            artist.set_linewidth(lw)


def _beeswarm_offsets_1d(
    y_for_layout: np.ndarray,
    *,
    max_width: float = 0.32,
    nbins: Optional[int] = None,
) -> np.ndarray:
    y = np.asarray(y_for_layout, dtype=NP_REAL_DTYPE).reshape(-1)
    n = int(y.size)

    if n == 0:
        return np.asarray([], dtype=NP_REAL_DTYPE)

    if n == 1:
        return np.zeros(1, dtype=NP_REAL_DTYPE)

    if nbins is None:
        nbins = max(6, int(np.ceil(np.sqrt(n))))

    ymin = float(np.min(y))
    ymax = float(np.max(y))

    if np.isclose(ymin, ymax):
        bin_ids = np.zeros(n, dtype=NP_INT_DTYPE)
    else:
        edges = np.linspace(ymin, ymax, nbins + 1)
        bin_ids = np.digitize(y, edges[1:-1], right=False).astype(NP_INT_DTYPE)

    offsets = np.zeros(n, dtype=NP_REAL_DTYPE)

    for b in np.unique(bin_ids):
        inds = np.where(bin_ids == b)[0]
        m = int(inds.size)

        if m <= 1:
            offsets[inds] = 0.0
            continue

        inds = inds[np.argsort(y[inds])]

        order = np.zeros(m, dtype=NP_REAL_DTYPE)

        for k in range(1, m):
            amp = (k + 1) // 2
            sign = 1.0 if k % 2 == 1 else -1.0
            order[k] = sign * amp

        max_abs = np.max(np.abs(order))

        if max_abs > 0.0:
            order = order / max_abs * max_width

        offsets[inds] = order

    return offsets


def plot_beeswarm_by_layer(
    datasets,
    layer_list,
    *,
    cmap,
    ylabel,
    title,
    outpath,
    point_size: float = 18.0,
    alpha: float = 0.65,
    edge_lw: float = 0.25,
    max_width: float = 0.32,
    log_scale: bool = False,
    eps: float = 1e-12,
):
    positions = np.asarray(layer_list, dtype=NP_REAL_DTYPE)

    fig, ax = new_fig_ax(outside_legend=False)

    for idx, (L, values) in enumerate(zip(layer_list, datasets)):
        color = cmap(idx / len(layer_list))

        y = np.asarray(values, dtype=NP_REAL_DTYPE).reshape(-1)

        # A thresholded detail view can legitimately leave a layer empty.
        # Skip non-finite/empty inputs before computing layout bins or a
        # median, while keeping every layer position on the x-axis.
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue

        if log_scale:
            y_plot = np.maximum(y, eps)
            y_for_layout = np.log10(y_plot)
        else:
            y_plot = y
            y_for_layout = y

        offsets = _beeswarm_offsets_1d(
            y_for_layout,
            max_width=max_width,
        )

        x = np.asarray(L, dtype=NP_REAL_DTYPE) + offsets

        ax.scatter(
            x,
            y_plot,
            s=point_size,
            alpha=alpha,
            color=color,
            edgecolors="black",
            linewidths=edge_lw,
            zorder=3,
        )

        median_y = np.median(y_plot)

        ax.hlines(
            median_y,
            xmin=float(L) - max_width,
            xmax=float(L) + max_width,
            color="black",
            linewidth=1.0,
            zorder=4,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(L) for L in layer_list])
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.3)
    else:
        ax.grid(True, axis="y", alpha=0.3)

    save_fig(fig, ax, outpath, outside_legend=False)


def plot_history_violin(
    data_by_layer,
    layer_list,
    *,
    sample_iters,
    x_groups,
    offset_span,
    violin_width,
    cmap,
    ylabel,
    title,
    outpath,
    transform,
    log_scale: bool = True,
):
    fig, ax = new_fig_ax(outside_legend=True)
    n = len(layer_list)
    handles = []

    for idx, L in enumerate(layer_list):
        color = cmap(idx / n)
        handles.append(
            Patch(
                facecolor=color,
                edgecolor=color,
                alpha=0.25,
                label=f"L{L}",
            )
        )

        offset = (idx - (n - 1) / 2) * (offset_span / n)
        positions = x_groups + offset

        runs = np.asarray(data_by_layer[L], dtype=NP_REAL_DTYPE)
        datasets = [transform(runs[:, t]) for t in sample_iters]

        vp = ax.violinplot(
            datasets,
            positions=positions,
            widths=violin_width,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        style_violin(vp, color, alpha=0.20, lw=1.0)

    ax.set_xlabel("Iterations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_groups)
    ax.set_xticklabels([str(t) for t in sample_iters], rotation=45, ha="right")

    if log_scale:
        ax.set_yscale("log")

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

    save_fig(fig, ax, outpath, outside_legend=True)


def plot_violin_by_layer(
    datasets,
    layer_list,
    *,
    cmap,
    ylabel,
    title,
    outpath,
    alpha=0.20,
    lw=1.0,
    line_color=None,
    show_legend: bool = True,
    log_scale: bool = False,
):
    positions = np.asarray(layer_list, dtype=NP_REAL_DTYPE)
    colors = [cmap(idx / len(layer_list)) for idx in range(len(layer_list))]

    fig, ax = new_fig_ax(outside_legend=show_legend)

    vp = ax.violinplot(
        datasets,
        positions=positions,
        widths=violin_width_from_positions(positions),
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    style_violin_bodies(vp, colors, alpha=alpha, lw=lw, line_color=line_color)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(L) for L in layer_list])
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.3)
    else:
        ax.grid(True, axis="y", alpha=0.3)

    if show_legend:
        ax.legend(
            handles=make_layer_legend_handles(layer_list, cmap),
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

    save_fig(fig, ax, outpath, outside_legend=show_legend)


def plot_single_line_by_layer(
    x,
    y,
    *,
    ylabel,
    title,
    outpath,
    label,
    ylim=None,
):
    fig, ax = new_fig_ax(outside_legend=False)

    ax.plot(x, y, marker="o", linestyle="-", label=label)
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.3)

    save_fig(fig, ax, outpath, outside_legend=False)


SUCCESS_PROBABILITY_FIGURE_THRESHOLDS = np.asarray(
    cfg.SUCCESS_PROBABILITY_FIGURE_THRESHOLDS,
    dtype=NP_REAL_DTYPE,
)


def _warn_skip_success_probability_figure(message: str) -> None:
    warnings.warn(
        "Skipping multiple-accuracy success-probability figure: "
        f"{message}",
        RuntimeWarning,
        stacklevel=2,
    )


def _success_probability_result_for_figure(result: dict) -> dict:
    """Recompute plot summaries at the figure-specific thresholds."""
    thresholds = np.asarray(
        SUCCESS_PROBABILITY_FIGURE_THRESHOLDS,
        dtype=NP_REAL_DTYPE,
    )
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError(
            "SUCCESS_PROBABILITY_FIGURE_THRESHOLDS must be a non-empty "
            "one-dimensional sequence"
        )
    if not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0.0):
        raise ValueError(
            "SUCCESS_PROBABILITY_FIGURE_THRESHOLDS must contain only "
            "positive finite values"
        )
    if np.any(np.diff(thresholds) >= 0.0):
        raise ValueError(
            "SUCCESS_PROBABILITY_FIGURE_THRESHOLDS must be strictly decreasing"
        )

    if "layers" not in result or "num_trials" not in result:
        raise ValueError("archive is missing 'layers' or 'num_trials'")

    layers = np.asarray(result["layers"])
    if layers.ndim != 1 or layers.size == 0:
        raise ValueError("'layers' must be a non-empty one-dimensional array")

    try:
        num_trials_array = np.asarray(
            result["num_trials"],
            dtype=NP_REAL_DTYPE,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("'num_trials' must be a numeric scalar") from exc
    if (
        num_trials_array.size != 1
        or not np.isfinite(num_trials_array.reshape(-1)[0])
        or num_trials_array.reshape(-1)[0]
        != np.rint(num_trials_array.reshape(-1)[0])
    ):
        raise ValueError("'num_trials' must be one finite integer scalar")
    num_trials = int(num_trials_array.reshape(-1)[0])
    if num_trials <= 0:
        raise ValueError("'num_trials' must be positive")

    if "final_energy_errors" in result:
        try:
            final_energy_errors = np.asarray(
                result["final_energy_errors"],
                dtype=NP_REAL_DTYPE,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "'final_energy_errors' must be a numeric array"
            ) from exc
    else:
        if "final_energies" not in result:
            raise ValueError(
                "archive is missing both 'final_energy_errors' and "
                "'final_energies'"
            )
        ground_energy_key = (
            "ground_energy"
            if "ground_energy" in result
            else "smallest_eigval"
        )
        if ground_energy_key not in result:
            raise ValueError(
                "archive with 'final_energies' is missing 'ground_energy' "
                "and 'smallest_eigval'"
            )
        try:
            final_energies = np.asarray(
                result["final_energies"],
                dtype=NP_REAL_DTYPE,
            )
            ground_energy_array = np.asarray(
                result[ground_energy_key],
                dtype=NP_REAL_DTYPE,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "'final_energies' and the ground energy must be numeric"
            ) from exc
        if ground_energy_array.size != 1 or not np.isfinite(
            ground_energy_array.reshape(-1)[0]
        ):
            raise ValueError("ground energy must be one finite scalar")
        final_energy_errors = np.maximum(
            final_energies - ground_energy_array.reshape(-1)[0],
            0.0,
        )

    expected_error_shape = (layers.size, num_trials)
    if final_energy_errors.shape != expected_error_shape:
        raise ValueError(
            f"'final_energy_errors' has shape {final_energy_errors.shape}; "
            f"expected {expected_error_shape}"
        )
    if not np.all(np.isfinite(final_energy_errors)):
        raise ValueError("'final_energy_errors' contains a non-finite value")
    if np.any(final_energy_errors < 0.0):
        raise ValueError("'final_energy_errors' contains a negative value")

    success_indicators = (
        final_energy_errors[:, :, None] <= thresholds[None, None, :]
    ).astype(NP_INT_DTYPE)
    success_counts = np.sum(
        success_indicators,
        axis=1,
        dtype=NP_INT_DTYPE,
    )
    success_probabilities = (
        success_counts.astype(NP_REAL_DTYPE) / NP_REAL_DTYPE(num_trials)
    )

    figure_result = dict(result)
    figure_result.update(
        thresholds=thresholds,
        tolerances=thresholds,
        success_indicators=success_indicators,
        success_counts=success_counts,
        success_probabilities=success_probabilities,
    )
    return figure_result


def _validated_success_probability_data(result: dict):
    """Validate and normalize the dedicated multiple-accuracy archive."""
    required_keys = {
        "layers",
        "num_trials",
        "success_counts",
        "success_probabilities",
    }
    missing_keys = sorted(required_keys.difference(result))
    if missing_keys:
        raise ValueError(
            "archive is missing required key(s): "
            + ", ".join(repr(key) for key in missing_keys)
        )

    if "thresholds" not in result and "tolerances" not in result:
        raise ValueError(
            "archive is missing both 'thresholds' and 'tolerances'"
        )

    try:
        layers_float = np.asarray(
            result["layers"],
            dtype=NP_REAL_DTYPE,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("'layers' must be a numeric one-dimensional array") from exc

    if layers_float.ndim != 1 or layers_float.size == 0:
        raise ValueError("'layers' must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(layers_float)):
        raise ValueError("'layers' contains a non-finite value")
    if not np.all(layers_float == np.rint(layers_float)):
        raise ValueError("'layers' must contain only integer values")

    try:
        layers = layers_float.astype(NP_INT_DTYPE)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("'layers' contains a value outside the integer range") from exc

    if np.any(layers <= 0):
        raise ValueError("'layers' must contain only positive integers")
    if np.unique(layers).size != layers.size:
        raise ValueError("'layers' contains duplicate entries")

    threshold_key = "thresholds" if "thresholds" in result else "tolerances"
    try:
        thresholds = np.asarray(
            result[threshold_key],
            dtype=NP_REAL_DTYPE,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"'{threshold_key}' must be a numeric one-dimensional array"
        ) from exc

    if thresholds.ndim != 1:
        raise ValueError(f"'{threshold_key}' must be one-dimensional")
    if (
        thresholds.size != SUCCESS_PROBABILITY_FIGURE_THRESHOLDS.size
        or not np.all(np.isfinite(thresholds))
        or np.any(thresholds <= 0.0)
    ):
        raise ValueError(
            f"'{threshold_key}' must contain the "
            f"{SUCCESS_PROBABILITY_FIGURE_THRESHOLDS.size} configured positive "
            "finite accuracy levels"
        )

    if "thresholds" in result and "tolerances" in result:
        try:
            tolerance_alias = np.asarray(
                result["tolerances"],
                dtype=NP_REAL_DTYPE,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "'tolerances' must be a numeric one-dimensional array"
            ) from exc

        if (
            tolerance_alias.shape != np.asarray(result["thresholds"]).shape
            or not np.allclose(
                tolerance_alias,
                np.asarray(result["thresholds"], dtype=NP_REAL_DTYPE),
                rtol=1e-12,
                atol=0.0,
            )
        ):
            raise ValueError(
                "'thresholds' and its 'tolerances' alias are inconsistent"
            )

    threshold_order = []
    for expected_threshold in SUCCESS_PROBABILITY_FIGURE_THRESHOLDS:
        matching = np.flatnonzero(
            np.isclose(
                thresholds,
                expected_threshold,
                rtol=1e-12,
                atol=0.0,
            )
        )
        if matching.size != 1:
            raise ValueError(
                f"'{threshold_key}' must contain exactly one "
                f"{expected_threshold:.8g} entry"
            )
        threshold_order.append(int(matching[0]))

    try:
        num_trials_array = np.asarray(
            result["num_trials"],
            dtype=NP_REAL_DTYPE,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("'num_trials' must be a numeric scalar") from exc

    if (
        num_trials_array.size != 1
        or not np.isfinite(num_trials_array.reshape(-1)[0])
        or num_trials_array.reshape(-1)[0]
        != np.rint(num_trials_array.reshape(-1)[0])
    ):
        raise ValueError("'num_trials' must be one finite integer scalar")
    num_trials = int(num_trials_array.reshape(-1)[0])

    expected_shape = (layers.size, thresholds.size)
    try:
        success_probabilities = np.asarray(
            result["success_probabilities"],
            dtype=NP_REAL_DTYPE,
        )
        success_counts_float = np.asarray(
            result["success_counts"],
            dtype=NP_REAL_DTYPE,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "'success_probabilities' and 'success_counts' must be numeric arrays"
        ) from exc

    if success_probabilities.shape != expected_shape:
        raise ValueError(
            "'success_probabilities' has shape "
            f"{success_probabilities.shape}; expected {expected_shape}"
        )
    if success_counts_float.shape != expected_shape:
        raise ValueError(
            f"'success_counts' has shape {success_counts_float.shape}; "
            f"expected {expected_shape}"
        )
    if not np.all(np.isfinite(success_probabilities)):
        raise ValueError("'success_probabilities' contains a non-finite value")

    probability_slack = 1e-12
    if np.any(success_probabilities < -probability_slack) or np.any(
        success_probabilities > 1.0 + probability_slack
    ):
        raise ValueError(
            "'success_probabilities' contains a value outside [0, 1]"
        )
    success_probabilities = np.clip(success_probabilities, 0.0, 1.0)

    if (
        not np.all(np.isfinite(success_counts_float))
        or not np.all(success_counts_float == np.rint(success_counts_float))
    ):
        raise ValueError("'success_counts' must contain only finite integers")
    if np.any(success_counts_float < 0) or np.any(
        success_counts_float > num_trials
    ):
        raise ValueError(
            "'success_counts' contains a value outside [0, num_trials]"
        )
    success_counts = success_counts_float.astype(NP_INT_DTYPE)

    if not np.allclose(
        success_probabilities,
        success_counts.astype(NP_REAL_DTYPE) / num_trials,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "'success_probabilities' is inconsistent with "
            "'success_counts / num_trials'"
        )

    threshold_order = np.asarray(threshold_order, dtype=NP_INT_DTYPE)
    thresholds = thresholds[threshold_order]
    success_probabilities = success_probabilities[:, threshold_order]
    success_counts = success_counts[:, threshold_order]

    # A smaller tolerance cannot have more successes than a larger one.
    if np.any(np.diff(success_counts, axis=1) > 0):
        raise ValueError(
            "'success_counts' is not monotone across decreasing tolerances"
        )

    layer_order = np.argsort(layers, kind="stable")
    return (
        layers[layer_order],
        thresholds,
        success_probabilities[layer_order],
        success_counts[layer_order],
        num_trials,
    )


def _success_probability_threshold_label(threshold: float) -> str:
    """Format the requested decade-spanning tolerances exactly."""
    threshold = float(threshold)
    hundredths = int(np.rint(100.0 * threshold))
    if 1 <= hundredths <= 10 and np.isclose(
        threshold,
        hundredths * 1e-2,
        rtol=1e-12,
        atol=0.0,
    ):
        if hundredths == 1:
            return r"$\delta=10^{-2}$"
        if hundredths == 10:
            return r"$\delta=10^{-1}$"
        return rf"$\delta={hundredths}\times 10^{{-2}}$"

    exponent = int(np.floor(np.log10(threshold)))
    mantissa = threshold / (10.0 ** exponent)
    if np.isclose(mantissa, 1.0, rtol=1e-12, atol=0.0):
        return rf"$\delta=10^{{{exponent}}}$"
    return rf"$\delta={mantissa:.3g}\times 10^{{{exponent}}}$"


def plot_success_probability_multiple_tolerances(
    layers,
    thresholds,
    success_probabilities,
    *,
    num_trials: int,
    outpath: str,
) -> None:
    """Plot one empirical VQE success-probability curve per tolerance."""
    layers = np.asarray(layers, dtype=NP_INT_DTYPE)
    thresholds = np.asarray(thresholds, dtype=NP_REAL_DTYPE)
    success_probabilities = np.asarray(
        success_probabilities,
        dtype=NP_REAL_DTYPE,
    )

    colors = matplotlib.colormaps.get_cmap("viridis")(
        np.linspace(0.08, 0.92, thresholds.size)
    )
    markers = ("o", "s", "^", "D", "P", "v", "X")
    linestyles = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))

    fig, ax = new_fig_ax(outside_legend=True)

    threshold_plot_order = np.argsort(thresholds, kind="stable")
    for plot_index, threshold_index in enumerate(threshold_plot_order):
        threshold = thresholds[threshold_index]
        ax.plot(
            layers,
            success_probabilities[:, threshold_index],
            color=colors[plot_index],
            marker=markers[plot_index % len(markers)],
            linestyle=linestyles[plot_index % len(linestyles)],
            linewidth=1.35,
            markersize=4.5,
            label=_success_probability_threshold_label(threshold),
            zorder=3,
        )

    ax.set_xlabel(r"Number of Layers $L$")
    ax.set_ylabel(
        r"Empirical success probability $\widehat{S}_L(\delta)$"
    )
    ax.set_title(
        "Success probability at multiple accuracy levels "
        f"({num_trials} independent trials)"
    )
    ax.set_xticks(layers)
    ax.set_xticklabels([str(int(layer)) for layer in layers])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        title=rf"$R={int(num_trials)}$ independent trials",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    save_fig(fig, ax, outpath, outside_legend=True)


def render_success_probability_multiple_tolerances_figure(
    *,
    result_path: str,
    outpath: str,
) -> bool:
    """Load, validate, and render the optional multiple-accuracy result."""
    if not os.path.exists(result_path):
        _warn_skip_success_probability_figure(
            f"result file was not found: {result_path}"
        )
        return False

    try:
        result = load_npz_result(result_path)
    except (OSError, ValueError, KeyError) as exc:
        _warn_skip_success_probability_figure(
            f"could not load result file {result_path}: {exc}"
        )
        return False

    recompute_error = None
    try:
        result_for_figure = _success_probability_result_for_figure(result)
    except (TypeError, ValueError, OverflowError) as exc:
        # Backward compatibility for summary-only archives that already use
        # the configured figure thresholds.
        recompute_error = exc
        result_for_figure = result

    try:
        (
            layers,
            thresholds,
            success_probabilities,
            _success_counts,
            num_trials,
        ) = _validated_success_probability_data(result_for_figure)
    except (TypeError, ValueError, OverflowError) as exc:
        recompute_detail = (
            ""
            if recompute_error is None
            else f"; could not recompute from final energies: {recompute_error}"
        )
        _warn_skip_success_probability_figure(
            f"invalid archive {result_path}: {exc}{recompute_detail}"
        )
        return False

    plot_success_probability_multiple_tolerances(
        layers,
        thresholds,
        success_probabilities,
        num_trials=num_trials,
        outpath=outpath,
    )
    return True


# ==============================
# Circuit blocks TensorCircuit
# ==============================
# ==============================
# TC -> Qiskit drawing
# ==============================
PARAMETER_FREE_GATE_LABELS = {
    "rz": r"$R_z$",
    "rx": r"$R_x$",
    "ry": r"$R_y$",
    "rxx": r"$R_{xx}$",
    "crx": r"$CR_x$",
    "cry": r"$CR_y$",
    "crz": r"$CR_z$",
    "cx": r"$CX$",
    "x": r"$X$",
    "z": r"$Z$",
    "h": r"$H$",
}


# ==============================
# Hamiltonian & ground truth
# ==============================
H_terms = tuple(hamiltonian_terms(h_param))

H_matrix = build_H_matrix_jax(H_terms, num_system_qubits)

smallest_eigval = float(
    np.linalg.eigvalsh(np.array(H_matrix, dtype=NP_COMPLEX_DTYPE)).min().real
)


# ==============================
# Parameter wrapping
# ==============================
# ============================================================
# Sequential trace-out machinery
# ============================================================
I2 = jnp.eye(2, dtype=COMPLEX_DTYPE)
X2 = jnp.array([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)
P0 = jnp.array([[1, 0], [0, 0]], dtype=COMPLEX_DTYPE)
P1 = jnp.array([[0, 0], [0, 1]], dtype=COMPLEX_DTYPE)


_U_CX = jnp.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=COMPLEX_DTYPE,
)


_RHO_KEEP_INIT = rho_zero_state(num_system_qubits, dtype=COMPLEX_DTYPE)


# ==============================
# Optimization loop per layer
# ==============================
success_rates_history = {}

final_stats = {
    "layer": [],
    "success_rate": [],
    "mean_energy": [],
    "std_energy": [],
}

# ------------------------------------------------------------
# Layer schedules
# ------------------------------------------------------------
vqe_dense_until_layer = cfg.VQE_DENSE_UNTIL_LAYER
vqe_max_layer = cfg.VQE_MAX_LAYER
vqe_sparse_step = cfg.VQE_SPARSE_STEP

vqe_layer_list = build_layer_list(
    vqe_max_layer,
    vqe_dense_until_layer,
    vqe_sparse_step,
)

qfim_dense_until_layer = cfg.QFIM_DENSE_UNTIL_LAYER
qfim_max_layer = cfg.QFIM_MAX_LAYER
qfim_sparse_step = cfg.QFIM_SPARSE_STEP

qfim_layer_list = build_layer_list(
    qfim_max_layer,
    qfim_dense_until_layer,
    qfim_sparse_step,
)

if not vqe_layer_list:
    raise ValueError(
        "vqe_layer_list is empty. Check vqe_max_layer, "
        "vqe_dense_until_layer, and vqe_sparse_step."
    )

if not qfim_layer_list:
    raise ValueError(
        "qfim_layer_list is empty. Check qfim_max_layer, "
        "qfim_dense_until_layer, and qfim_sparse_step."
    )

save_dir = f"./figs/dpqc/h_{h_param}"

energy_fig_dir = os.path.join(save_dir, "energy_figures")
qfim_fig_dir = os.path.join(save_dir, "qfim_figures")
qfim_trace_dir = os.path.join(qfim_fig_dir, "qfim_trace")
qfim_eigs_dir = os.path.join(qfim_fig_dir, "qfim_eigs")
qfim_eigs_dir_red4 = os.path.join(qfim_eigs_dir, "reduced_keep_0123")
qfim_eigs_dir_red5 = os.path.join(qfim_eigs_dir, "reduced_keep_01234")
qfim_rank_dir = os.path.join(qfim_fig_dir, "qfim_rank")
qfim_rank_random_dir = os.path.join(qfim_rank_dir, "random_points")
qfim_rank_optimization_path_dir = os.path.join(qfim_rank_dir, "optimization_path")
qfim_rank_optimization_path_mean_dir = os.path.join(
    qfim_rank_optimization_path_dir,
    "mean",
)
qfim_rank_optimization_path_min_dir = os.path.join(
    qfim_rank_optimization_path_dir,
    "min",
)
qfim_eigcount_dir = os.path.join(qfim_fig_dir, "qfim_eigcount")
qfim_eigcount_random_dir = os.path.join(qfim_eigcount_dir, "random_points")
qfim_eigcount_optimization_path_dir = os.path.join(
    qfim_eigcount_dir,
    "optimization_path",
)
qfim_eigcount_optimization_path_mean_dir = os.path.join(
    qfim_eigcount_optimization_path_dir,
    "mean",
)
qfim_eigcount_optimization_path_min_dir = os.path.join(
    qfim_eigcount_optimization_path_dir,
    "min",
)
circuit_dir = os.path.join(save_dir, "optimized_circuits")
numerical_results_dir = os.path.join(save_dir, "numerical_results")
energy_results_dir = os.path.join(numerical_results_dir, "energy")
qfim_results_dir = os.path.join(numerical_results_dir, "qfim")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(energy_fig_dir, exist_ok=True)
os.makedirs(qfim_fig_dir, exist_ok=True)
os.makedirs(qfim_trace_dir, exist_ok=True)
os.makedirs(qfim_eigs_dir, exist_ok=True)
os.makedirs(qfim_eigs_dir_red4, exist_ok=True)
os.makedirs(qfim_eigs_dir_red5, exist_ok=True)
os.makedirs(qfim_rank_dir, exist_ok=True)
os.makedirs(qfim_rank_random_dir, exist_ok=True)
os.makedirs(qfim_rank_optimization_path_dir, exist_ok=True)
os.makedirs(qfim_rank_optimization_path_mean_dir, exist_ok=True)
os.makedirs(qfim_rank_optimization_path_min_dir, exist_ok=True)
os.makedirs(qfim_eigcount_dir, exist_ok=True)
os.makedirs(qfim_eigcount_random_dir, exist_ok=True)
os.makedirs(qfim_eigcount_optimization_path_dir, exist_ok=True)
os.makedirs(qfim_eigcount_optimization_path_mean_dir, exist_ok=True)
os.makedirs(qfim_eigcount_optimization_path_min_dir, exist_ok=True)
os.makedirs(circuit_dir, exist_ok=True)
os.makedirs(numerical_results_dir, exist_ok=True)
os.makedirs(energy_results_dir, exist_ok=True)
os.makedirs(qfim_results_dir, exist_ok=True)


def _load_layer_arrays_from_npz(
    result: dict,
    layers,
    suffix: Optional[str] = None,
    *,
    dtype=None,
) -> dict:
    arrays_by_layer = {}

    for L in layers:
        L_int = int(L)
        key = f"L{L_int}" if suffix is None else f"L{L_int}_{suffix}"

        if key not in result:
            continue

        arr = np.asarray(result[key])
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        arrays_by_layer[L_int] = arr

    return arrays_by_layer


cmap = matplotlib.colormaps.get_cmap("viridis")


# ============================================================
# Load VQE numerical results and generate energy figures
# ============================================================
vqe_optimization_result_path = os.path.join(
    energy_results_dir,
    "vqe_optimization_histories.npz",
)

vqe_optimization_results = load_npz_result(vqe_optimization_result_path)

vqe_layer_list = [
    int(L)
    for L in np.asarray(vqe_optimization_results["vqe_layers"], dtype=NP_INT_DTYPE)
]
qfim_layer_list = [
    int(L)
    for L in np.asarray(vqe_optimization_results["qfim_layers"], dtype=NP_INT_DTYPE)
]
sample_iters = np.asarray(vqe_optimization_results["sample_iters"], dtype=NP_INT_DTYPE)
steps = int(np.asarray(vqe_optimization_results["steps"]).item())
num_runs = int(np.asarray(vqe_optimization_results["num_runs"]).item())
smallest_eigval = NP_REAL_DTYPE(
    np.asarray(vqe_optimization_results["smallest_eigval"]).item()
)

energy_traces_by_layer = _load_layer_arrays_from_npz(
    vqe_optimization_results,
    vqe_layer_list,
    "energy_traces",
    dtype=NP_REAL_DTYPE,
)
grad_norm_traces_by_layer = _load_layer_arrays_from_npz(
    vqe_optimization_results,
    vqe_layer_list,
    "grad_norm_traces",
    dtype=NP_REAL_DTYPE,
)
theta_history = _load_layer_arrays_from_npz(
    vqe_optimization_results,
    vqe_layer_list,
    "theta_final",
    dtype=NP_REAL_DTYPE,
)
theta_sample_traces_by_layer = _load_layer_arrays_from_npz(
    vqe_optimization_results,
    vqe_layer_list,
    "theta_samples",
    dtype=NP_REAL_DTYPE,
)
grad_sample_traces_by_layer = _load_layer_arrays_from_npz(
    vqe_optimization_results,
    vqe_layer_list,
    "grad_samples",
    dtype=NP_REAL_DTYPE,
)
success_rates_history = _load_layer_arrays_from_npz(
    vqe_optimization_results,
    vqe_layer_list,
    "success_rates",
    dtype=NP_REAL_DTYPE,
)
final_theta_periodic_only_rmsdist_by_layer = _load_layer_arrays_from_npz(
    vqe_optimization_results,
    vqe_layer_list,
    "theta_rmsdist",
    dtype=NP_REAL_DTYPE,
)
ancilla_p1_stats_by_layer = {
    int(L): {
        "ancilla_qubits": np.asarray(
            vqe_optimization_results[f"L{int(L)}_ancilla_qubits"],
            dtype=NP_INT_DTYPE,
        ),
        "p1_runs": np.asarray(
            vqe_optimization_results[f"L{int(L)}_ancilla_p1_runs"],
            dtype=NP_REAL_DTYPE,
        ),
        "mean": np.asarray(
            vqe_optimization_results[f"L{int(L)}_ancilla_p1_mean"],
            dtype=NP_REAL_DTYPE,
        ),
        "var": np.asarray(
            vqe_optimization_results[f"L{int(L)}_ancilla_p1_var"],
            dtype=NP_REAL_DTYPE,
        ),
        "std": np.asarray(
            vqe_optimization_results[f"L{int(L)}_ancilla_p1_std"],
            dtype=NP_REAL_DTYPE,
        ),
    }
    for L in vqe_layer_list
}
final_stats = {
    "layer": np.asarray(vqe_optimization_results["final_stats_layer"], dtype=NP_INT_DTYPE),
    "success_rate": np.asarray(vqe_optimization_results["final_stats_success_rate"], dtype=NP_REAL_DTYPE),
    "mean_energy": np.asarray(vqe_optimization_results["final_stats_mean_energy"], dtype=NP_REAL_DTYPE),
    "std_energy": np.asarray(vqe_optimization_results["final_stats_std_energy"], dtype=NP_REAL_DTYPE),
}


# ==============================
# Plotting & saving figures
# ==============================
os.makedirs(save_dir, exist_ok=True)
os.makedirs(energy_fig_dir, exist_ok=True)
os.makedirs(qfim_fig_dir, exist_ok=True)

x_groups = np.arange(len(sample_iters), dtype=NP_REAL_DTYPE)
num_vqe_layers = len(vqe_layer_list)
offset_span = 0.75
box_width = 0.85 * (offset_span / num_vqe_layers)

plot_history_violin(
    energy_traces_by_layer,
    vqe_layer_list,
    sample_iters=sample_iters,
    x_groups=x_groups,
    offset_span=offset_span,
    violin_width=box_width,
    cmap=cmap,
    ylabel="Absolute energy error",
    title=f"Loss history over {num_runs} runs",
    outpath=os.path.join(energy_fig_dir, "loss_history.pdf"),
    transform=lambda x: np.abs(x - smallest_eigval) + eps,
)

plot_history_violin(
    grad_norm_traces_by_layer,
    vqe_layer_list,
    sample_iters=sample_iters,
    x_groups=x_groups,
    offset_span=offset_span,
    violin_width=box_width,
    cmap=cmap,
    ylabel="Gradient norm",
    title=f"Gradient-norm history over {num_runs} runs",
    outpath=os.path.join(energy_fig_dir, "grad_norm_history.pdf"),
    transform=lambda x: x + eps,
)

final_energy_error_by_layer = [
    np.abs(
        np.asarray(
            energy_traces_by_layer[L][:, -1],
            dtype=NP_REAL_DTYPE,
        )
        - smallest_eigval
    )
    for L in vqe_layer_list
]

plot_violin_by_layer(
    final_energy_error_by_layer,
    vqe_layer_list,
    cmap=cmap,
    ylabel="Final energy error",
    title="Final energy-error distributions",
    outpath=os.path.join(energy_fig_dir, "final_energy_error.pdf"),
    show_legend=False,
    log_scale=False,
)

plot_violin_by_layer(
    [
        np.maximum(err, eps)
        for err in final_energy_error_by_layer
    ],
    vqe_layer_list,
    cmap=cmap,
    ylabel="Final energy error",
    title="Final energy-error distributions",
    outpath=os.path.join(energy_fig_dir, "final_energy_error_logscale.pdf"),
    show_legend=False,
    log_scale=True,
)

plot_beeswarm_by_layer(
    final_energy_error_by_layer,
    vqe_layer_list,
    cmap=cmap,
    ylabel="Final energy error",
    title="Final energy-error distributions",
    outpath=os.path.join(energy_fig_dir, "final_energy_error_beeswarm.pdf"),
    point_size=18.0,
    alpha=0.65,
    max_width=0.32,
    log_scale=False,
)

plot_beeswarm_by_layer(
    [
        np.maximum(err, eps)
        for err in final_energy_error_by_layer
    ],
    vqe_layer_list,
    cmap=cmap,
    ylabel="Final energy error",
    title="Final energy-error distributions",
    outpath=os.path.join(energy_fig_dir, "final_energy_error_beeswarm_logscale.pdf"),
    point_size=18.0,
    alpha=0.65,
    max_width=0.32,
    log_scale=True,
    eps=eps,
)

final_energy_error_detail_threshold = float(
    FINAL_ENERGY_ERROR_DETAIL_THRESHOLD
)

if (
    not np.isfinite(final_energy_error_detail_threshold)
    or final_energy_error_detail_threshold <= 0.0
):
    raise ValueError(
        "FINAL_ENERGY_ERROR_DETAIL_THRESHOLD must be finite and positive."
    )

final_energy_error_detail_by_layer = [
    err[
        np.isfinite(err)
        & (err <= final_energy_error_detail_threshold)
    ]
    for err in final_energy_error_by_layer
]

if not any(values.size for values in final_energy_error_detail_by_layer):
    warnings.warn(
        "No final energy errors satisfy "
        f"error <= {final_energy_error_detail_threshold:g}; "
        "skipping the detailed beeswarm figure.",
        RuntimeWarning,
        stacklevel=1,
    )
else:
    plot_beeswarm_by_layer(
        final_energy_error_detail_by_layer,
        vqe_layer_list,
        cmap=cmap,
        ylabel=(
            "Final energy error "
            rf"($\Delta E \leq {final_energy_error_detail_threshold:g}$)"
        ),
        title=(
            "Detailed final energy-error distributions "
            rf"($\Delta E \leq {final_energy_error_detail_threshold:g}$)"
        ),
        outpath=os.path.join(
            energy_fig_dir,
            (
                "final_energy_error_beeswarm_below_"
                f"{_thr_tag(final_energy_error_detail_threshold)}.pdf"
            ),
        ),
        point_size=18.0,
        alpha=0.65,
        max_width=0.32,
        log_scale=False,
    )

plot_single_line_by_layer(
    np.array(final_stats["layer"], dtype=NP_REAL_DTYPE),
    np.array(final_stats["success_rate"], dtype=NP_REAL_DTYPE),
    ylabel="Success Rate",
    title=f"Success Rate over {num_runs} runs (Tol={tolerance})",
    outpath=os.path.join(energy_fig_dir, "success_rate.pdf"),
    label="Success Rate",
    ylim=(-0.02, 1.02),
)

render_success_probability_multiple_tolerances_figure(
    result_path=os.path.join(
        energy_results_dir,
        "vqe_success_probability_multiple_tolerances.npz",
    ),
    outpath=os.path.join(
        energy_fig_dir,
        "success_probability_multiple_tolerances.pdf",
    ),
)

fig, ax = new_fig_ax(outside_legend=True)
legend_handles = []
all_x_ticks = set()

for idx, L in enumerate(vqe_layer_list):
    color = cmap(idx / len(vqe_layer_list))

    legend_handles.append(
        Patch(
            facecolor=color,
            edgecolor=color,
            alpha=0.25,
            label=f"L{L}",
        )
    )

    anc_ids = ancilla_p1_stats_by_layer[L]["ancilla_qubits"]
    p1_runs = ancilla_p1_stats_by_layer[L]["p1_runs"]

    positions = anc_ids.astype(float) + (
        idx - (len(vqe_layer_list) - 1) / 2
    ) * (0.12 / len(vqe_layer_list))

    all_x_ticks.update(anc_ids.tolist())

    bp = ax.boxplot(
        [p1_runs[:, j] for j in range(p1_runs.shape[1])],
        positions=positions,
        widths=0.08,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
    )

    style_boxplot(bp, color, alpha=0.20, lw=1.0)

ax.set_xlabel("Ancilla Qubit Index")
ax.set_ylabel("Ancilla probability")
ax.set_title(rf"Ancilla $P(1)$ over {num_runs} runs")
ax.grid(True, alpha=0.3)
ax.set_xticks(sorted(all_x_ticks))
ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left")

save_fig(
    fig,
    ax,
    os.path.join(energy_fig_dir, "ancilla_p1.pdf"),
    outside_legend=True,
)


# ============================================================

# ============================================================
# Load random-parameter QFIM results and generate QFIM figures
# ============================================================
# QFIM rank + eigenvalue plots for both retained subsystems.
# ============================================================
KEEP_WIRES_4 = (0, 1, 2, 3)
assert KEEP_WIRES_4 == tuple(range(num_system_qubits - 1))
KEEP_WIRES_5 = tuple(range(num_system_qubits))

QFIM_EFFECTIVE_RANK_THRESHOLD = cfg.QFIM_EFFECTIVE_RANK_THRESHOLD

EIG_SUM_EPS = cfg.EIG_SUM_EPS
QFIM_EIG_PLOT_EPS = cfg.QFIM_EIG_PLOT_EPS
NUM_QFIM_SAMPLES = cfg.NUM_QFIM_SAMPLES
QFIM_SAMPLE_SEED_BASE = cfg.QFIM_SAMPLE_SEED_BASE
RED_JVP_CHUNK = cfg.RED_JVP_CHUNK

# Thresholds used for large-sector gradient-weight diagnostics.
# Keep this broad set unless you also want to reduce the gradient-sector plots.
THRESHOLDS = tuple(float(t) for t in cfg.GRADIENT_SECTOR_THRESHOLDS)
QFIM_PATH_EIGCOUNT_THRESHOLDS = tuple(
    float(t) for t in cfg.QFIM_PATH_EIGCOUNT_THRESHOLDS
)


def _rank_thresholds_for_eigs_sorted_desc(
    eigs_sorted_desc: np.ndarray,
) -> np.ndarray:
    eigs_arr = np.asarray(eigs_sorted_desc, dtype=NP_REAL_DTYPE)

    if eigs_arr.ndim == 1:
        eigs_arr = eigs_arr[None, :]

    eigs_jnp = jnp.asarray(eigs_arr, dtype=REAL_DTYPE)

    thresholds = jax.vmap(
        lambda evals_1d: threshold_psd_eigvals_for_rank(evals_1d)[1]
    )(eigs_jnp)

    return np.asarray(jax.device_get(thresholds), dtype=NP_REAL_DTYPE)


def eigenvalue_index_ticks(n_params: int, *, max_ticks: int = 11) -> np.ndarray:
    n_params = int(n_params)

    if n_params <= 0:
        return np.asarray([], dtype=NP_INT_DTYPE)

    max_ticks = max(2, int(max_ticks))

    if n_params <= max_ticks:
        return np.arange(1, n_params + 1, dtype=NP_INT_DTYPE)

    ticks = np.rint(
        np.linspace(1, n_params, num=max_ticks)
    ).astype(NP_INT_DTYPE)

    ticks = np.unique(ticks)
    ticks[0] = 1
    ticks[-1] = n_params

    if ticks.size >= 3:
        gaps = np.diff(ticks).astype(NP_REAL_DTYPE)
        typical_gap = float(np.median(gaps))

        if typical_gap > 0.0 and float(gaps[-1]) < 0.60 * typical_gap:
            ticks = np.delete(ticks, -2)

    return ticks


def save_qfim_eigs_by_index(
    eigs_sorted_desc: np.ndarray,
    *,
    title: str,
    outpath: str,
    eps: float = QFIM_EIG_PLOT_EPS,
    point_size: float = 14.0,
    alpha: float = 0.55,
) -> None:
    eigs_raw = np.asarray(eigs_sorted_desc, dtype=NP_REAL_DTYPE)

    if eigs_raw.ndim == 1:
        eigs_raw = eigs_raw[None, :]

    eigs_plot = eigs_raw.copy()
    eigs_plot[eigs_plot <= 0.0] = eps

    n_params = int(eigs_plot.shape[1])

    rank_thresholds = _rank_thresholds_for_eigs_sorted_desc(eigs_raw)
    rank_thresholds_plot = np.maximum(rank_thresholds, eps)

    fig, ax = new_fig_ax(outside_legend=False)

    positions = np.arange(1, n_params + 1, dtype=NP_REAL_DTYPE)

    for i, x0 in enumerate(positions):
        y_plot = eigs_plot[:, i]
        ax.scatter(
            np.full_like(y_plot, x0, dtype=NP_REAL_DTYPE),
            y_plot,
            s=point_size,
            color="C0",
            alpha=alpha,
            edgecolors="black",
            linewidths=0.20,
            rasterized=True,
        )

    ticks = eigenvalue_index_ticks(n_params, max_ticks=11)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(0.5, n_params + 0.5)
    ax.set_yscale("log")

    for thr in np.unique(rank_thresholds_plot):
        ax.axhline(
            thr,
            linestyle="-",
            linewidth=1.2,
            color="red",
            alpha=0.75,
            zorder=4,
        )

    ymin = min(float(np.min(eigs_plot)), float(np.min(rank_thresholds_plot)))
    ymax = max(float(np.max(eigs_plot)), float(np.max(rank_thresholds_plot)))
    ymin = max(eps, ymin)

    if ymax > ymin:
        ax.set_ylim(max(eps, ymin / 1.5), ymax * 1.5)

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("QFIM eigenvalue")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    save_fig(fig, ax, outpath, outside_legend=False)


def save_qfim_eigs_by_index_colored_by_layer(
    eigs_sorted_desc_by_layer: dict,
    layer_list,
    *,
    title: str,
    outpath: str,
    eps: float = QFIM_EIG_PLOT_EPS,
    cmap=None,
    alpha: float = 0.50,
    point_size: float = 10.0,
) -> None:
    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap

    layers = [
        int(L)
        for L in layer_list
        if eigs_sorted_desc_by_layer.get(L) is not None
    ]

    if not layers:
        return

    max_n_params = max(
        int(
            np.asarray(
                eigs_sorted_desc_by_layer[L],
                dtype=NP_REAL_DTYPE,
            ).shape[1]
        )
        for L in layers
    )

    fig, ax = new_fig_ax(outside_legend=True)

    handles = []

    for idx, L in enumerate(layers):
        color = cmap(idx / len(layers))

        handles.append(
            Patch(
                facecolor=color,
                edgecolor=color,
                alpha=0.25,
                label=f"L{L}",
            )
        )

        eigs = np.asarray(
            eigs_sorted_desc_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )

        for i in range(max_n_params):
            if i < eigs.shape[1]:
                v = eigs[:, i].astype(float)
                v[v <= 0.0] = eps
                x = np.full_like(v, i + 1, dtype=NP_REAL_DTYPE)
                ax.scatter(
                    x,
                    v,
                    s=point_size,
                    color=color,
                    alpha=alpha,
                    edgecolors="black",
                    linewidths=0.15,
                    rasterized=True,
                )

    ticks = eigenvalue_index_ticks(max_n_params, max_ticks=11)

    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(0.5, max_n_params + 0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("QFIM eigenvalue")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.85,
    )

    save_fig(fig, ax, outpath, outside_legend=True)


qfim_random_points_result_path = os.path.join(
    qfim_results_dir,
    f"qfim_random_points_{keep_key}.npz",
)

qfim_random_points_results = load_npz_result(qfim_random_points_result_path)
qfim_layer_list = [
    int(L)
    for L in np.asarray(qfim_random_points_results["layers"], dtype=NP_INT_DTYPE)
]
qfim_rank_reduced_0123_by_layer = _load_layer_arrays_from_npz(
    qfim_random_points_results,
    qfim_layer_list,
    "rank",
    dtype=NP_INT_DTYPE,
)
qfim_eigs_reduced_0123_by_layer = _load_layer_arrays_from_npz(
    qfim_random_points_results,
    qfim_layer_list,
    "eigs_desc",
    dtype=NP_REAL_DTYPE,
)
qfim_rho_rank_reduced_0123_by_layer = _load_layer_arrays_from_npz(
    qfim_random_points_results,
    qfim_layer_list,
    "rho_rank",
    dtype=NP_INT_DTYPE,
)
qfim_eigsum_reduced_0123_by_layer = _load_layer_arrays_from_npz(
    qfim_random_points_results,
    qfim_layer_list,
    "trace",
    dtype=NP_REAL_DTYPE,
)
qfim_abs_entry_sum_reduced_0123_by_layer = _load_layer_arrays_from_npz(
    qfim_random_points_results,
    qfim_layer_list,
    "abs_entry_sum",
    dtype=NP_REAL_DTYPE,
)

for L in qfim_layer_list:
    save_qfim_eigs_by_index(
        qfim_eigs_reduced_0123_by_layer[L],
        title=rf"QFIM eigenvalues at {NUM_QFIM_SAMPLES} random points (L={L})",
        outpath=os.path.join(qfim_eigs_dir_red4, f"L{L}_reduced_0123.pdf"),
    )


save_qfim_eigs_by_index_colored_by_layer(
    qfim_eigs_reduced_0123_by_layer,
    qfim_layer_list,
    title=rf"QFIM eigenvalues at {NUM_QFIM_SAMPLES} random points",
    outpath=os.path.join(
        qfim_eigs_dir,
        "qfim_eigs_by_index_layers_reduced_keep_0123.pdf",
    ),
    cmap=cmap,
)


# ============================================================
# Mean/minimum QFIM rank + density-rank upper bound by layer
#   reduced keep=(0,1,2,3) only
# ============================================================
def _qfim_rank_mean_min_sem_upper_bound_xy(
    rank_by_layer: dict,
    rho_rank_by_layer: dict,
    layers,
    *,
    d_keep: int = 2 ** len(KEEP_WIRES_4),
):
    valid_items = []

    for L in layers:
        ranks_L = rank_by_layer.get(L)

        if ranks_L is None:
            continue

        ranks_arr = np.asarray(ranks_L, dtype=NP_REAL_DTYPE).reshape(-1)

        if ranks_arr.size == 0:
            continue

        valid_items.append((int(L), ranks_arr))

    if not valid_items:
        return (
            np.asarray([], dtype=NP_REAL_DTYPE),
            np.asarray([], dtype=NP_REAL_DTYPE),
            np.asarray([], dtype=NP_REAL_DTYPE),
            np.asarray([], dtype=NP_REAL_DTYPE),
            np.asarray([], dtype=NP_REAL_DTYPE),
            [],
        )

    valid_layers = [L for L, _ in valid_items]
    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)

    min_ranks = np.asarray(
        [np.min(ranks_arr) for _, ranks_arr in valid_items],
        dtype=NP_REAL_DTYPE,
    )

    mean_ranks = np.asarray(
        [np.mean(ranks_arr) for _, ranks_arr in valid_items],
        dtype=NP_REAL_DTYPE,
    )

    sem_ranks = np.asarray(
        [
            NP_REAL_DTYPE(0.0)
            if ranks_arr.size <= 1
            else NP_REAL_DTYPE(np.std(ranks_arr, ddof=1) / np.sqrt(ranks_arr.size))
            for _, ranks_arr in valid_items
        ],
        dtype=NP_REAL_DTYPE,
    )

    upper_bounds = []

    for L in valid_layers:
        rho_ranks_L = rho_rank_by_layer.get(L)

        if rho_ranks_L is None:
            r_for_bound = d_keep
        else:
            rho_ranks_arr = np.asarray(rho_ranks_L, dtype=NP_REAL_DTYPE).reshape(-1)
            rho_ranks_arr = rho_ranks_arr[np.isfinite(rho_ranks_arr)]

            if rho_ranks_arr.size == 0:
                r_for_bound = d_keep
            else:
                r_for_bound = int(np.max(rho_ranks_arr))

        r_for_bound = int(np.clip(r_for_bound, 1, d_keep))
        density_manifold_bound = 2 * d_keep * r_for_bound - r_for_bound**2 - 1
        parameter_bound = n_param_per_layer * int(L)
        upper_bounds.append(min(parameter_bound, density_manifold_bound))

    upper_bounds = np.asarray(upper_bounds, dtype=NP_REAL_DTYPE)

    return x, min_ranks, mean_ranks, sem_ranks, upper_bounds, valid_layers


def plot_single_qfim_rank_mean_min_sem_by_layer(
    rank_by_layer: dict,
    rho_rank_by_layer: dict,
    layers,
    *,
    d_keep: int = 2 ** len(KEEP_WIRES_4),
    color_min,
    color_mean,
    label=None,
    title,
    outpath,
    marker_min: str = "o",
    marker_mean: str = "s",
    lw: float = 1.4,
):
    (
        x,
        min_ranks,
        mean_ranks,
        sem_ranks,
        upper_bounds,
        valid_layers,
    ) = _qfim_rank_mean_min_sem_upper_bound_xy(
        rank_by_layer,
        rho_rank_by_layer,
        layers,
        d_keep=d_keep,
    )

    if x.size == 0:
        return

    label_suffix = "" if label in (None, "") else rf" ({label})"

    fig, ax = new_fig_ax(outside_legend=False)

    ax.plot(
        x,
        min_ranks,
        marker=marker_min,
        linestyle="-",
        linewidth=lw,
        markersize=6.0,
        color=color_min,
        label=rf"Minimum effective rank{label_suffix}",
    )

    ax.errorbar(
        x,
        mean_ranks,
        yerr=sem_ranks,
        marker=marker_mean,
        linestyle="-",
        linewidth=lw,
        markersize=5.0,
        capsize=4.0,
        elinewidth=1.0,
        color=color_mean,
        label=rf"Mean effective rank $\pm$ SEM{label_suffix}",
    )

    ax.plot(
        x,
        upper_bounds,
        marker=None,
        linestyle="--",
        linewidth=lw,
        color="black",
        label=(
            rf"Upper bound $\min({n_param_per_layer}L, "
            rf"{2 * int(d_keep)}r_{{\max}}-r_{{\max}}^2-1)$"
        ),
    )

    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("QFIM rank")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in valid_layers])
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    save_fig(fig, ax, outpath, outside_legend=False)


plot_single_qfim_rank_mean_min_sem_by_layer(
    qfim_rank_reduced_0123_by_layer,
    qfim_rho_rank_reduced_0123_by_layer,
    qfim_layer_list,
    color_min="C0",
    color_mean="C1",
    label=None,
    title=rf"QFIM effective rank mean/minimum and upper bound at {NUM_QFIM_SAMPLES} random points",
    outpath=os.path.join(
        qfim_rank_random_dir,
        "qfim_rank_mean_min_upper_bound_random_points_reduced_0123.pdf",
    ),
    marker_min="o",
    marker_mean="s",
    lw=1.4,
)


def _qfim_threshold_tex_for_label(threshold: float) -> str:
    threshold = float(threshold)

    if threshold <= 0.0:
        return f"{threshold:g}"

    exp = int(np.floor(np.log10(threshold)))
    mant = threshold / (10.0 ** exp)

    if np.isclose(mant, 1.0):
        return rf"10^{{{exp}}}"

    return rf"{mant:g}\times 10^{{{exp}}}"


def qfim_random_eigcount_by_threshold_by_layer(
    eigs_by_layer: dict,
    layers,
    thresholds,
) -> dict:
    eigcount_by_threshold = {}

    for threshold in thresholds:
        threshold = float(threshold)
        count_by_layer = {}

        for L in layers:
            L_int = int(L)
            if eigs_by_layer.get(L_int) is None:
                continue

            eigs_L = np.asarray(eigs_by_layer[L_int], dtype=NP_REAL_DTYPE)

            if eigs_L.ndim != 2:
                raise ValueError(
                    "Each random-point QFIM eigenvalue array must be 2D: "
                    "(num_samples, num_params)."
                )

            count_by_layer[L_int] = np.sum(
                eigs_L >= threshold,
                axis=1,
            ).astype(NP_REAL_DTYPE)

        eigcount_by_threshold[threshold] = count_by_layer

    return eigcount_by_threshold


def plot_qfim_random_eigcount_threshold_overlay(
    eigs_by_layer: dict,
    layers,
    thresholds,
    *,
    title: str,
    outpath: str,
    cmap=None,
):
    thresholds = tuple(float(thr) for thr in thresholds)

    if not thresholds:
        return

    eigcount_by_threshold = qfim_random_eigcount_by_threshold_by_layer(
        eigs_by_layer,
        layers,
        thresholds,
    )

    valid_layers = [
        int(L)
        for L in layers
        if any(
            eigcount_by_threshold[threshold].get(int(L)) is not None
            for threshold in thresholds
        )
    ]

    if not valid_layers:
        return

    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)
    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap

    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.24)
    n_thresholds = len(thresholds)

    for threshold_idx, threshold in enumerate(thresholds):
        means = []
        sems = []
        layers_for_threshold = []

        for L in valid_layers:
            counts = eigcount_by_threshold[threshold].get(L)

            if counts is None:
                continue

            counts = np.asarray(counts, dtype=NP_REAL_DTYPE).reshape(-1)
            counts = counts[np.isfinite(counts)]

            if counts.size == 0:
                continue

            layers_for_threshold.append(L)
            means.append(NP_REAL_DTYPE(np.mean(counts)))
            sems.append(
                NP_REAL_DTYPE(0.0)
                if counts.size <= 1
                else NP_REAL_DTYPE(np.std(counts, ddof=1) / np.sqrt(counts.size))
            )

        if not layers_for_threshold:
            continue

        x_thr = np.asarray(layers_for_threshold, dtype=NP_REAL_DTYPE)
        color = cmap(threshold_idx / max(n_thresholds - 1, 1))
        label = rf"$\lambda_i \geq {_qfim_threshold_tex_for_label(threshold)}$"

        ax.errorbar(
            x_thr,
            np.asarray(means, dtype=NP_REAL_DTYPE),
            yerr=np.asarray(sems, dtype=NP_REAL_DTYPE),
            marker="o",
            linestyle="-",
            linewidth=1.2,
            markersize=4.8,
            capsize=3.0,
            elinewidth=0.8,
            color=color,
            label=label,
        )

    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Mean QFIM eigenvalue count")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in valid_layers])
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
    )

    save_fig(
        fig,
        ax,
        outpath,
        outside_legend=True,
        legend_space_frac=0.24,
    )


plot_qfim_random_eigcount_threshold_overlay(
    qfim_eigs_reduced_0123_by_layer,
    qfim_layer_list,
    QFIM_PATH_EIGCOUNT_THRESHOLDS,
    title=rf"QFIM eigenvalue count at random points by threshold ({keep_label})",
    outpath=os.path.join(
        qfim_eigcount_random_dir,
        f"qfim_eigcount_threshold_overlay_random_points_{keep_key}.pdf",
    ),
    cmap=cmap,
)


# ============================================================
# Maximum QFIM trace + mean ± SEM by layer
#   Trace(F) is computed as the sum of QFIM eigenvalues at each
#   random parameter point, using qfim_eigsum_reduced_0123_by_layer.
#   reduced keep=(0,1,2,3) only
# ============================================================
def _qfim_metric_max_mean_sem_xy(metric_by_layer: dict, layers):
    valid_items = []

    for L in layers:
        values_L = metric_by_layer.get(L)

        if values_L is None:
            continue

        values_arr = np.asarray(values_L, dtype=NP_REAL_DTYPE).reshape(-1)
        values_arr = values_arr[np.isfinite(values_arr)]

        if values_arr.size == 0:
            continue

        valid_items.append((int(L), values_arr))

    if not valid_items:
        return (
            np.asarray([], dtype=NP_REAL_DTYPE),
            np.asarray([], dtype=NP_REAL_DTYPE),
            np.asarray([], dtype=NP_REAL_DTYPE),
            np.asarray([], dtype=NP_REAL_DTYPE),
            [],
        )

    valid_layers = [L for L, _ in valid_items]
    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)

    max_values = np.asarray(
        [np.max(values_arr) for _, values_arr in valid_items],
        dtype=NP_REAL_DTYPE,
    )

    mean_values = np.asarray(
        [np.mean(values_arr) for _, values_arr in valid_items],
        dtype=NP_REAL_DTYPE,
    )

    sem_values = np.asarray(
        [
            NP_REAL_DTYPE(0.0)
            if values_arr.size <= 1
            else NP_REAL_DTYPE(np.std(values_arr, ddof=1) / np.sqrt(values_arr.size))
            for _, values_arr in valid_items
        ],
        dtype=NP_REAL_DTYPE,
    )

    return x, max_values, mean_values, sem_values, valid_layers


def plot_qfim_trace_max_mean_sem_by_layer(
    trace_by_layer: dict,
    layers,
    *,
    title: str,
    outpath: str,
    color_max="C0",
    color_mean="C1",
    marker_max: str = "o",
    marker_mean: str = "s",
    lw: float = 1.4,
    log_scale: bool = False,
):
    (
        x,
        max_values,
        mean_values,
        sem_values,
        valid_layers,
    ) = _qfim_metric_max_mean_sem_xy(trace_by_layer, layers)

    if x.size == 0:
        return

    fig, ax = new_fig_ax(outside_legend=False)

    ax.plot(
        x,
        max_values,
        marker=marker_max,
        linestyle="-",
        linewidth=lw,
        markersize=6.0,
        color=color_max,
        label="Maximum QFIM trace",
    )

    ax.errorbar(
        x,
        mean_values,
        yerr=sem_values,
        marker=marker_mean,
        linestyle="-",
        linewidth=lw,
        markersize=5.0,
        capsize=4.0,
        elinewidth=1.0,
        color=color_mean,
        label=r"Mean QFIM trace $\pm$ SEM",
    )

    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("QFIM trace")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in valid_layers])

    if log_scale:
        ax.set_yscale("log")

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    save_fig(fig, ax, outpath, outside_legend=False)


plot_qfim_trace_max_mean_sem_by_layer(
    qfim_eigsum_reduced_0123_by_layer,
    qfim_layer_list,
    title=rf"QFIM trace maximum and mean $\pm$ SEM at {NUM_QFIM_SAMPLES} random points",
    outpath=os.path.join(
        qfim_trace_dir,
        "qfim_trace_max_mean_sem_random_points_reduced_0123.pdf",
    ),
    color_max="C0",
    color_mean="C1",
    marker_max="o",
    marker_mean="s",
    lw=1.4,
    log_scale=False,
)


# ============================================================

# ============================================================
# Load large-sector gradient-weight results
# ============================================================
# Large-sector gradient weight along the VQE optimization path
#   color  = layer number
#   marker = QFIM eigenvalue threshold
# ============================================================
GRADIENT_SECTOR_NORM_EPS = 1e-24


def _finite_mean_sem_over_runs_by_time(values_2d: np.ndarray):
    values = np.asarray(values_2d, dtype=NP_REAL_DTYPE)

    if values.ndim == 1:
        values = values[None, :]

    valid = np.isfinite(values)
    counts = np.sum(valid, axis=0).astype(NP_REAL_DTYPE)
    sums = np.nansum(np.where(valid, values, 0.0), axis=0)

    means = np.divide(
        sums,
        counts,
        out=np.full(values.shape[1], np.nan, dtype=NP_REAL_DTYPE),
        where=counts > 0,
    )

    centered = np.where(valid, values - means[None, :], np.nan)
    sq = np.nansum(centered**2, axis=0)

    stds = np.sqrt(
        np.divide(
            sq,
            counts - 1.0,
            out=np.zeros_like(sq, dtype=NP_REAL_DTYPE),
            where=counts > 1,
        )
    )

    sems = np.divide(
        stds,
        np.sqrt(counts),
        out=np.zeros_like(stds, dtype=NP_REAL_DTYPE),
        where=counts > 1,
    )

    return means, sems, counts


qfim_large_sector_gradient_weight_result_path = os.path.join(
    qfim_results_dir,
    f"qfim_large_sector_gradient_weight_{keep_key}.npz",
)

qfim_large_sector_gradient_weight_results = load_npz_result(
    qfim_large_sector_gradient_weight_result_path
)
qfim_gradient_large_sector_weight_by_layer = {
    int(L): {
        float(thr): np.asarray(
            qfim_large_sector_gradient_weight_results[
                f"L{int(L)}_thr_{_thr_tag(float(thr))}"
            ],
            dtype=NP_REAL_DTYPE,
        )
        for thr in np.asarray(
            qfim_large_sector_gradient_weight_results["thresholds"],
            dtype=NP_REAL_DTYPE,
        )
    }
    for L in np.asarray(
        qfim_large_sector_gradient_weight_results["layers"],
        dtype=NP_INT_DTYPE,
    )
}


# ============================================================

# ============================================================
# Load optimization-path QFIM results and generate path figures
# ============================================================
# QFIM rank along the VQE optimization path
#   x-axis: sampled optimization iteration
#   y-axis: run-mean QFIM effective rank at theta(iteration)
#   color: layer number
# ============================================================
def plot_qfim_rank_history_mean_by_layer(
    rank_history_by_layer: dict,
    layers,
    sample_iters,
    *,
    title: str,
    outpath: str,
    cmap=None,
):
    valid_layers = [
        int(L)
        for L in layers
        if rank_history_by_layer.get(int(L)) is not None
    ]

    if not valid_layers:
        return

    x = np.asarray(sample_iters, dtype=NP_REAL_DTYPE)
    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap

    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.22)
    n_layers = len(valid_layers)

    for layer_idx, L in enumerate(valid_layers):
        ranks = np.asarray(rank_history_by_layer[L], dtype=NP_REAL_DTYPE)

        if ranks.ndim != 2:
            raise ValueError(
                "Each rank history array must be 2D: "
                "(num_runs, num_sample_iters)."
            )

        if ranks.shape[1] != x.size and ranks.shape[0] == x.size:
            ranks = ranks.T

        if ranks.shape[1] != x.size:
            raise ValueError(
                f"Shape mismatch for L={L}: "
                f"ranks.shape={ranks.shape}, len(sample_iters)={x.size}."
            )

        means, sems, counts = _finite_mean_sem_over_runs_by_time(ranks)
        finite_mask = np.isfinite(means) & (counts > 0)

        if not np.any(finite_mask):
            continue

        color = cmap(layer_idx / max(n_layers - 1, 1))

        ax.errorbar(
            x[finite_mask],
            means[finite_mask],
            yerr=sems[finite_mask],
            marker="o",
            linestyle="-",
            linewidth=1.2,
            markersize=4.5,
            capsize=3.0,
            elinewidth=0.8,
            color=color,
            label=f"L={L}",
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Mean QFIM rank")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in x], rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
    )

    save_fig(
        fig,
        ax,
        outpath,
        outside_legend=True,
        legend_space_frac=0.22,
    )


def plot_qfim_rank_history_min_by_layer(
    rank_history_by_layer: dict,
    layers,
    sample_iters,
    *,
    title: str,
    outpath: str,
    cmap=None,
):
    valid_layers = [
        int(L)
        for L in layers
        if rank_history_by_layer.get(int(L)) is not None
    ]

    if not valid_layers:
        return

    x = np.asarray(sample_iters, dtype=NP_REAL_DTYPE)
    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap

    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.22)
    n_layers = len(valid_layers)

    for layer_idx, L in enumerate(valid_layers):
        ranks = np.asarray(rank_history_by_layer[L], dtype=NP_REAL_DTYPE)

        if ranks.ndim != 2:
            raise ValueError(
                "Each rank history array must be 2D: "
                "(num_runs, num_sample_iters)."
            )

        if ranks.shape[1] != x.size and ranks.shape[0] == x.size:
            ranks = ranks.T

        if ranks.shape[1] != x.size:
            raise ValueError(
                f"Shape mismatch for L={L}: "
                f"ranks.shape={ranks.shape}, len(sample_iters)={x.size}."
            )

        valid = np.isfinite(ranks)
        counts = np.sum(valid, axis=0)
        ranks_for_min = np.where(valid, ranks, np.inf)
        min_ranks = np.min(ranks_for_min, axis=0)
        min_ranks = np.where(counts > 0, min_ranks, np.nan)
        finite_mask = np.isfinite(min_ranks) & (counts > 0)

        if not np.any(finite_mask):
            continue

        color = cmap(layer_idx / max(n_layers - 1, 1))

        ax.plot(
            x[finite_mask],
            min_ranks[finite_mask],
            marker="o",
            linestyle="-",
            linewidth=1.2,
            markersize=4.5,
            color=color,
            label=f"L={L}",
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Minimum QFIM rank")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in x], rotation=45, ha="right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
    )

    save_fig(
        fig,
        ax,
        outpath,
        outside_legend=True,
        legend_space_frac=0.22,
    )


def plot_qfim_trace_history_mean_by_layer(
    trace_history_by_layer: dict,
    layers,
    sample_iters,
    *,
    title: str,
    outpath: str,
    cmap=None,
    log_scale: bool = False,
):
    valid_layers = [
        int(L)
        for L in layers
        if trace_history_by_layer.get(int(L)) is not None
    ]

    if not valid_layers:
        return

    x = np.asarray(sample_iters, dtype=NP_REAL_DTYPE)
    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap

    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.22)
    n_layers = len(valid_layers)

    for layer_idx, L in enumerate(valid_layers):
        traces = np.asarray(trace_history_by_layer[L], dtype=NP_REAL_DTYPE)

        if traces.ndim != 2:
            raise ValueError(
                "Each QFIM trace history array must be 2D: "
                "(num_runs, num_sample_iters)."
            )

        if traces.shape[1] != x.size and traces.shape[0] == x.size:
            traces = traces.T

        if traces.shape[1] != x.size:
            raise ValueError(
                f"Shape mismatch for L={L}: "
                f"traces.shape={traces.shape}, len(sample_iters)={x.size}."
            )

        means, sems, counts = _finite_mean_sem_over_runs_by_time(traces)
        finite_mask = np.isfinite(means) & (counts > 0)

        if not np.any(finite_mask):
            continue

        color = cmap(layer_idx / max(n_layers - 1, 1))

        ax.errorbar(
            x[finite_mask],
            means[finite_mask],
            yerr=sems[finite_mask],
            marker="o",
            linestyle="-",
            linewidth=1.2,
            markersize=4.5,
            capsize=3.0,
            elinewidth=0.8,
            color=color,
            label=f"L={L}",
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Mean QFIM trace")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in x], rotation=45, ha="right")

    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.3)
    else:
        ax.grid(True, axis="y", alpha=0.3)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
    )

    save_fig(
        fig,
        ax,
        outpath,
        outside_legend=True,
        legend_space_frac=0.22,
    )


qfim_rank_history_result_path = os.path.join(
    qfim_results_dir,
    f"qfim_rank_history_optimization_path_{keep_key}.npz",
)

qfim_rank_history_results = load_npz_result(qfim_rank_history_result_path)
vqe_layer_list = [
    int(L)
    for L in np.asarray(qfim_rank_history_results["layers"], dtype=NP_INT_DTYPE)
]
sample_iters = np.asarray(qfim_rank_history_results["sample_iters"], dtype=NP_INT_DTYPE)
qfim_rank_history_by_layer = _load_layer_arrays_from_npz(
    qfim_rank_history_results,
    vqe_layer_list,
    suffix=None,
    dtype=NP_REAL_DTYPE,
)

plot_qfim_rank_history_mean_by_layer(
    qfim_rank_history_by_layer,
    vqe_layer_list,
    sample_iters,
    title=rf"Mean QFIM effective rank along optimization path ({keep_label})",
    outpath=os.path.join(
        qfim_rank_optimization_path_mean_dir,
        f"qfim_rank_mean_history_optimization_path_{keep_key}.pdf",
    ),
    cmap=cmap,
)

plot_qfim_rank_history_min_by_layer(
    qfim_rank_history_by_layer,
    vqe_layer_list,
    sample_iters,
    title=rf"Minimum QFIM effective rank along optimization path ({keep_label})",
    outpath=os.path.join(
        qfim_rank_optimization_path_min_dir,
        f"qfim_rank_min_history_optimization_path_{keep_key}.pdf",
    ),
    cmap=cmap,
)


qfim_eigs_history_result_path = os.path.join(
    qfim_results_dir,
    f"qfim_eigs_history_optimization_path_{keep_key}.npz",
)

qfim_eigs_history_results = load_npz_result(qfim_eigs_history_result_path)
qfim_eigs_history_optimization_path_by_layer = _load_layer_arrays_from_npz(
    qfim_eigs_history_results,
    vqe_layer_list,
    suffix=None,
    dtype=NP_REAL_DTYPE,
)

qfim_trace_history_result_path = os.path.join(
    qfim_results_dir,
    f"qfim_trace_history_optimization_path_{keep_key}.npz",
)

if os.path.exists(qfim_trace_history_result_path):
    qfim_trace_history_results = load_npz_result(qfim_trace_history_result_path)
    qfim_trace_history_layer_list = [
        int(L)
        for L in np.asarray(qfim_trace_history_results["layers"], dtype=NP_INT_DTYPE)
    ]
    qfim_trace_history_optimization_path_by_layer = _load_layer_arrays_from_npz(
        qfim_trace_history_results,
        qfim_trace_history_layer_list,
        suffix=None,
        dtype=NP_REAL_DTYPE,
    )
else:
    qfim_trace_history_layer_list = list(vqe_layer_list)
    qfim_trace_history_optimization_path_by_layer = {
        int(L): np.sum(np.asarray(eigs, dtype=NP_REAL_DTYPE), axis=2)
        for L, eigs in qfim_eigs_history_optimization_path_by_layer.items()
    }

plot_qfim_trace_history_mean_by_layer(
    qfim_trace_history_optimization_path_by_layer,
    qfim_trace_history_layer_list,
    sample_iters,
    title=rf"Mean QFIM trace along optimization path ({keep_label})",
    outpath=os.path.join(
        qfim_trace_dir,
        f"qfim_trace_mean_history_optimization_path_{keep_key}.pdf",
    ),
    cmap=cmap,
    log_scale=False,
)


def qfim_path_eig_target_iterations(
    sample_iters_for_labels,
    *,
    every: int = sample_every,
) -> np.ndarray:
    every = int(every)
    if every <= 0:
        raise ValueError("every must be a positive integer.")

    sample_iters_arr = np.asarray(sample_iters_for_labels, dtype=NP_INT_DTYPE)
    return sample_iters_arr[(sample_iters_arr % every) == 0]


def _sample_time_index_for_iteration(sample_iters_for_labels, iteration: int) -> int:
    sample_iters_arr = np.asarray(sample_iters_for_labels, dtype=NP_INT_DTYPE)
    hit = np.where(sample_iters_arr == int(iteration))[0]

    if hit.size == 0:
        raise ValueError(
            f"iteration {iteration} is not included in sample_iters. "
            "Add it to sample_iters before running the VQE optimization loop."
        )

    return int(hit[0])


def save_qfim_eigs_optimization_path_by_iteration(
    eigs_history_by_layer: dict,
    layers,
    sample_iters_for_labels,
    *,
    outdir: str,
    result_key: str = keep_key,
    result_label: str = keep_label,
    target_iterations=None,
    eps: float = QFIM_EIG_PLOT_EPS,
):
    os.makedirs(outdir, exist_ok=True)

    sample_iters_arr = np.asarray(sample_iters_for_labels, dtype=NP_INT_DTYPE)
    if target_iterations is None:
        target_iterations = qfim_path_eig_target_iterations(sample_iters_arr)

    target_iterations = np.asarray(target_iterations, dtype=NP_INT_DTYPE)
    if target_iterations.size == 0:
        return []

    saved_paths = []

    for L in tqdm(
        layers,
        desc="QFIM eig distributions along optimization path",
        unit="layer",
    ):
        L_int = int(L)
        if eigs_history_by_layer.get(L_int) is None:
            continue

        eigs_L = np.asarray(
            eigs_history_by_layer[L_int],
            dtype=NP_REAL_DTYPE,
        )

        if eigs_L.ndim != 3:
            raise ValueError(
                "Each QFIM eigenvalue history array must be 3D: "
                "(num_runs, num_sample_iters, num_params)."
            )

        if (
            eigs_L.shape[1] != sample_iters_arr.size
            and eigs_L.shape[0] == sample_iters_arr.size
        ):
            eigs_L = np.transpose(eigs_L, (1, 0, 2))

        if eigs_L.shape[1] != sample_iters_arr.size:
            raise ValueError(
                f"Shape mismatch for L={L_int}: "
                f"eigs_L.shape={eigs_L.shape}, len(sample_iters)={sample_iters_arr.size}."
            )

        layer_outdir = os.path.join(outdir, f"L{L_int}")
        os.makedirs(layer_outdir, exist_ok=True)

        for iteration in target_iterations:
            iteration_int = int(iteration)
            time_idx = _sample_time_index_for_iteration(
                sample_iters_arr,
                iteration_int,
            )

            outpath = os.path.join(
                layer_outdir,
                f"iter{iteration_int:06d}_{result_key}.pdf",
            )

            save_qfim_eigs_by_index(
                eigs_L[:, time_idx, :],
                title=(
                    rf"QFIM eigenvalues along optimization path "
                    rf"(L={L_int}, iteration={iteration_int}, {result_label})"
                ),
                outpath=outpath,
                eps=eps,
            )
            saved_paths.append(outpath)

    return saved_paths


qfim_eigs_optimization_path_target_iterations = qfim_path_eig_target_iterations(
    sample_iters,
    every=sample_every,
)

qfim_eigs_optimization_path_dir = os.path.join(
    qfim_eigs_dir,
    f"optimization_path_{keep_key}",
)

qfim_eigs_optimization_path_files = save_qfim_eigs_optimization_path_by_iteration(
    qfim_eigs_history_optimization_path_by_layer,
    vqe_layer_list,
    sample_iters,
    outdir=qfim_eigs_optimization_path_dir,
    target_iterations=qfim_eigs_optimization_path_target_iterations,
    eps=QFIM_EIG_PLOT_EPS,
)


def _sample_mean_sem(samples: np.ndarray) -> Tuple[NP_REAL_DTYPE, NP_REAL_DTYPE]:
    samples = np.asarray(samples, dtype=NP_REAL_DTYPE).reshape(-1)
    n = int(samples.size)

    if n == 0:
        return NP_REAL_DTYPE(0.0), NP_REAL_DTYPE(0.0)

    mean = NP_REAL_DTYPE(np.mean(samples))

    if n <= 1:
        sem = NP_REAL_DTYPE(0.0)
    else:
        sem = NP_REAL_DTYPE(np.std(samples, ddof=1) / np.sqrt(n))

    return mean, sem


def _mean_sem_arrays_from_by_layer(stats_by_layer: dict, layers):
    valid_layers = [
        int(L)
        for L in layers
        if stats_by_layer.get(L) is not None
    ]

    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)

    means = np.asarray(
        [stats_by_layer[L]["mean"] for L in valid_layers],
        dtype=NP_REAL_DTYPE,
    )

    sems = np.asarray(
        [stats_by_layer[L]["sem"] for L in valid_layers],
        dtype=NP_REAL_DTYPE,
    )

    return x, means, sems



def _threshold_tex(threshold: float) -> str:
    threshold = float(threshold)

    if threshold <= 0.0:
        return f"{threshold:g}"

    exp = int(np.floor(np.log10(threshold)))
    mant = threshold / (10.0 ** exp)

    if np.isclose(mant, 1.0):
        return rf"10^{{{exp}}}"

    return rf"{mant:g}\times 10^{{{exp}}}"


def qfim_eigcount_history_by_layer(
    eigs_history_by_layer: dict,
    layers,
    sample_iters_for_labels,
    *,
    threshold: float,
) -> dict:
    sample_iters_arr = np.asarray(sample_iters_for_labels, dtype=NP_INT_DTYPE)
    threshold = float(threshold)
    count_history_by_layer = {}

    for L in layers:
        L_int = int(L)
        if eigs_history_by_layer.get(L_int) is None:
            continue

        eigs_L = np.asarray(eigs_history_by_layer[L_int], dtype=NP_REAL_DTYPE)

        if eigs_L.ndim != 3:
            raise ValueError(
                "Each QFIM eigenvalue history array must be 3D: "
                "(num_runs, num_sample_iters, num_params)."
            )

        if eigs_L.shape[1] != sample_iters_arr.size and eigs_L.shape[0] == sample_iters_arr.size:
            eigs_L = np.transpose(eigs_L, (1, 0, 2))

        if eigs_L.shape[1] != sample_iters_arr.size:
            raise ValueError(
                f"Shape mismatch for L={L_int}: "
                f"eigs_L.shape={eigs_L.shape}, len(sample_iters)={sample_iters_arr.size}."
            )

        count_history_by_layer[L_int] = np.sum(
            eigs_L >= threshold,
            axis=2,
        ).astype(NP_REAL_DTYPE)

    return count_history_by_layer


def plot_qfim_eigcount_history_mean_by_layer(
    eigcount_history_by_layer: dict,
    layers,
    sample_iters_for_labels,
    *,
    threshold: float,
    title: str,
    outpath: str,
    cmap=None,
):
    valid_layers = [
        int(L)
        for L in layers
        if eigcount_history_by_layer.get(int(L)) is not None
    ]

    if not valid_layers:
        return

    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap
    threshold_label = _threshold_tex(threshold)
    x = np.asarray(sample_iters_for_labels, dtype=NP_REAL_DTYPE)
    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.22)
    n_layers = len(valid_layers)

    for layer_idx, L in enumerate(valid_layers):
        counts_L = np.asarray(
            eigcount_history_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )

        if counts_L.ndim != 2:
            raise ValueError(
                "Each QFIM eig-count history array must be 2D: "
                "(num_runs, num_sample_iters)."
            )

        if counts_L.shape[1] != x.size:
            raise ValueError(
                f"Shape mismatch for L={L}: "
                f"counts_L.shape={counts_L.shape}, len(sample_iters)={x.size}."
            )

        valid = np.isfinite(counts_L)
        n_valid = np.sum(valid, axis=0)
        sums = np.sum(np.where(valid, counts_L, 0.0), axis=0)
        means = np.divide(
            sums,
            n_valid,
            out=np.full(x.shape, np.nan, dtype=NP_REAL_DTYPE),
            where=n_valid > 0,
        )

        centered = np.where(valid, counts_L - means[None, :], np.nan)
        sq = np.nansum(centered**2, axis=0)
        stds = np.sqrt(
            np.divide(
                sq,
                n_valid - 1.0,
                out=np.zeros_like(sq, dtype=NP_REAL_DTYPE),
                where=n_valid > 1,
            )
        )
        sems = np.divide(
            stds,
            np.sqrt(n_valid),
            out=np.zeros_like(stds, dtype=NP_REAL_DTYPE),
            where=n_valid > 1,
        )

        finite_mask = np.isfinite(means) & (n_valid > 0)
        if not np.any(finite_mask):
            continue

        color = cmap(layer_idx / max(n_layers - 1, 1))
        ax.plot(
            x[finite_mask],
            means[finite_mask],
            marker="o",
            linestyle="-",
            linewidth=1.2,
            markersize=4.5,
            color=color,
            label=f"L={L}",
        )
        ax.fill_between(
            x[finite_mask],
            means[finite_mask] - sems[finite_mask],
            means[finite_mask] + sems[finite_mask],
            color=color,
            alpha=0.16,
            linewidth=0.0,
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel(rf"Mean QFIM eigenvalue count ($\lambda_i \geq {threshold_label}$)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in x], rotation=45, ha="right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
    )

    save_fig(
        fig,
        ax,
        outpath,
        outside_legend=True,
        legend_space_frac=0.22,
    )


def plot_qfim_eigcount_history_min_by_layer(
    eigcount_history_by_layer: dict,
    layers,
    sample_iters_for_labels,
    *,
    threshold: float,
    title: str,
    outpath: str,
    cmap=None,
):
    valid_layers = [
        int(L)
        for L in layers
        if eigcount_history_by_layer.get(int(L)) is not None
    ]

    if not valid_layers:
        return

    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap
    threshold_label = _threshold_tex(threshold)
    x = np.asarray(sample_iters_for_labels, dtype=NP_REAL_DTYPE)
    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.22)
    n_layers = len(valid_layers)

    for layer_idx, L in enumerate(valid_layers):
        counts_L = np.asarray(
            eigcount_history_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )

        if counts_L.ndim != 2:
            raise ValueError(
                "Each QFIM eig-count history array must be 2D: "
                "(num_runs, num_sample_iters)."
            )

        if counts_L.shape[1] != x.size:
            raise ValueError(
                f"Shape mismatch for L={L}: "
                f"counts_L.shape={counts_L.shape}, len(sample_iters)={x.size}."
            )

        valid = np.isfinite(counts_L)
        n_valid = np.sum(valid, axis=0)
        counts_for_min = np.where(valid, counts_L, np.inf)
        min_counts = np.min(counts_for_min, axis=0)
        min_counts = np.where(n_valid > 0, min_counts, np.nan)
        finite_mask = np.isfinite(min_counts) & (n_valid > 0)

        if not np.any(finite_mask):
            continue

        color = cmap(layer_idx / max(n_layers - 1, 1))
        ax.plot(
            x[finite_mask],
            min_counts[finite_mask],
            marker="o",
            linestyle="-",
            linewidth=1.2,
            markersize=4.5,
            color=color,
            label=f"L={L}",
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel(rf"Minimum QFIM eigenvalue count ($\lambda_i \geq {threshold_label}$)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in x], rotation=45, ha="right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
    )

    save_fig(
        fig,
        ax,
        outpath,
        outside_legend=True,
        legend_space_frac=0.22,
    )


qfim_eigcount_sample_iters = np.asarray(sample_iters, dtype=NP_INT_DTYPE)
qfim_eigcount_time_indices = np.arange(
    qfim_eigcount_sample_iters.size,
    dtype=NP_INT_DTYPE,
)

for threshold in QFIM_PATH_EIGCOUNT_THRESHOLDS:
    eigcount_history_all_by_layer = qfim_eigcount_history_by_layer(
        qfim_eigs_history_optimization_path_by_layer,
        vqe_layer_list,
        sample_iters,
        threshold=threshold,
    )
    eigcount_history_by_layer = {
        int(L): np.asarray(counts, dtype=NP_REAL_DTYPE)[
            :,
            qfim_eigcount_time_indices,
        ]
        for L, counts in eigcount_history_all_by_layer.items()
    }
    threshold_tag = _thr_tag(threshold)
    threshold_label = _threshold_tex(threshold)

    plot_qfim_eigcount_history_mean_by_layer(
        eigcount_history_by_layer,
        vqe_layer_list,
        qfim_eigcount_sample_iters,
        threshold=threshold,
        title=(
            rf"Mean QFIM eigenvalue count along optimization path "
            rf"($\lambda_i \geq {threshold_label}$, {keep_label})"
        ),
        outpath=os.path.join(
            qfim_eigcount_optimization_path_mean_dir,
            f"qfim_eigcount_mean_history_optimization_path_thr_{threshold_tag}_{keep_key}.pdf",
        ),
        cmap=cmap,
    )

    plot_qfim_eigcount_history_min_by_layer(
        eigcount_history_by_layer,
        vqe_layer_list,
        qfim_eigcount_sample_iters,
        threshold=threshold,
        title=(
            rf"Minimum QFIM eigenvalue count along optimization path "
            rf"($\lambda_i \geq {threshold_label}$, {keep_label})"
        ),
        outpath=os.path.join(
            qfim_eigcount_optimization_path_min_dir,
            f"qfim_eigcount_min_history_optimization_path_thr_{threshold_tag}_{keep_key}.pdf",
        ),
        cmap=cmap,
    )


# ============================================================
# Mean ± SEM scalar QFIM diagnostics
#   1. Sum of QFIM eigenvalues
#   2. Sum of absolute values of QFIM matrix entries
#   reduced keep=(0,1,2,3) only
# ============================================================
# The trace summary below uses optimization-path samples. The absolute-entry
# summary remains based on the random-point QFIM data saved above.
def _metric_mean_sem_by_layer(metric_by_layer: dict, layers) -> dict:
    stats_by_layer = {}

    for L in layers:
        if metric_by_layer.get(L) is None:
            continue

        mean_L, sem_L = _sample_mean_sem(metric_by_layer[L])

        stats_by_layer[L] = {
            "mean": mean_L,
            "sem": sem_L,
            "n": NP_INT_DTYPE(np.asarray(metric_by_layer[L]).size),
        }

    return stats_by_layer


def plot_metric_mean_sem_by_layer(
    metric_by_layer: dict,
    layers,
    *,
    ylabel: str,
    title: str,
    outpath: str,
    label: str,
    color="C0",
    marker: str = "o",
    log_scale: bool = False,
):
    stats_by_layer = _metric_mean_sem_by_layer(metric_by_layer, layers)

    x, means, sems = _mean_sem_arrays_from_by_layer(
        stats_by_layer,
        layers,
    )

    if x.size == 0:
        return

    fig, ax = new_fig_ax(outside_legend=False)

    ax.errorbar(
        x,
        means,
        yerr=sems,
        marker=marker,
        linestyle="-",
        linewidth=1.2,
        markersize=6.0,
        capsize=4.0,
        elinewidth=1.0,
        color=color,
        label=label,
    )

    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(L)) for L in x])

    if log_scale:
        ax.set_yscale("log")

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    save_fig(fig, ax, outpath, outside_legend=False)


plot_metric_mean_sem_by_layer(
    qfim_trace_history_optimization_path_by_layer,
    qfim_trace_history_layer_list,
    ylabel="Mean QFIM trace",
    title=rf"QFIM trace mean $\pm$ SEM vs Layers along optimization path ({keep_label})",
    outpath=os.path.join(
        qfim_trace_dir,
        f"qfim_trace_mean_errorbar_optimization_path_{keep_key}.pdf",
    ),
    label=r"$\sum_k \lambda_k(F)$",
    color="C0",
    marker="o",
    log_scale=False,
)

plot_metric_mean_sem_by_layer(
    qfim_abs_entry_sum_reduced_0123_by_layer,
    qfim_layer_list,
    ylabel="Mean elementwise absolute sum",
    title=rf"QFIM elementwise-absolute-sum mean $\pm$ SEM vs Layers ({keep_label}) at {NUM_QFIM_SAMPLES} random points",
    outpath=os.path.join(
        qfim_fig_dir,
        f"qfim_abs_entry_sum_mean_errorbar_{keep_key}.pdf",
    ),
    label=r"$\sum_{i,j} |F_{ij}|$",
    color="C1",
    marker="s",
    log_scale=False,
)

# ============================================================

# ============================================================
# QFIM spectral/effective-rank summary figures
# ============================================================
# These diagnostics are additive: older numerical-result directories do not
# contain the two summary archives below.  Missing or incomplete archives must
# therefore skip only the affected new figure, while all legacy plots continue
# to be generated.
qfim_effective_rank_fig_dir = os.path.join(qfim_fig_dir, "effective_rank")
qfim_spectral_gradient_fig_dir = os.path.join(
    qfim_fig_dir,
    "qfim_grad_alignment",
)
qfim_cumulative_alignment_fig_dir = os.path.join(
    qfim_fig_dir,
    "cumulative_alignment",
)

for _diagnostic_fig_dir in (
    qfim_effective_rank_fig_dir,
    qfim_spectral_gradient_fig_dir,
    qfim_cumulative_alignment_fig_dir,
):
    os.makedirs(_diagnostic_fig_dir, exist_ok=True)


def qfim_grad_alignment_dirs_for_key(result_key: str):
    result_root = os.path.join(qfim_results_dir, "qfim_grad_alignment")

    if result_key == keep_key:
        return qfim_spectral_gradient_fig_dir, result_root

    figure_dir = os.path.join(qfim_spectral_gradient_fig_dir, result_key)
    result_dir = os.path.join(result_root, result_key)
    os.makedirs(figure_dir, exist_ok=True)
    return figure_dir, result_dir


def _warn_skip_new_figure(message: str) -> None:
    warnings.warn(
        f"Skipping new QFIM diagnostic figure: {message}",
        RuntimeWarning,
        stacklevel=2,
    )


def _load_optional_npz_result(path: str, *, description: str):
    if not os.path.exists(path):
        _warn_skip_new_figure(
            f"{description} result file was not found: {path}"
        )
        return None

    try:
        return load_npz_result(path)
    except (OSError, ValueError, KeyError) as exc:
        _warn_skip_new_figure(
            f"could not load {description} result file {path}: {exc}"
        )
        return None


def _summary_layers_or_none(result: dict, *, description: str):
    if "layers" not in result:
        _warn_skip_new_figure(f"{description} archive has no 'layers' key")
        return None

    layers = [
        int(L)
        for L in np.asarray(result["layers"], dtype=NP_INT_DTYPE).reshape(-1)
    ]

    if not layers:
        _warn_skip_new_figure(f"{description} archive has an empty layer list")
        return None

    return layers


def _summary_sample_iters_or_none(result: dict, *, description: str):
    if "sample_iters" not in result:
        _warn_skip_new_figure(
            f"{description} archive has no 'sample_iters' key"
        )
        return None

    sample_iters_result = np.asarray(
        result["sample_iters"],
        dtype=NP_INT_DTYPE,
    ).reshape(-1)

    if sample_iters_result.size == 0:
        _warn_skip_new_figure(
            f"{description} archive has an empty 'sample_iters' array"
        )
        return None

    return sample_iters_result


def render_qfim_keep01234_core_figures() -> None:
    """Render the keep=(0,1,2,3,4) counterparts of the core QFIM figures."""
    random_result_path = os.path.join(
        qfim_results_dir,
        f"qfim_random_points_{keep_key_5}.npz",
    )
    random_result = _load_optional_npz_result(
        random_result_path,
        description=(
            "keep01234 random-point QFIM result; rerun "
            "DPQC_overparam_compute.py for complete keep01234 figures"
        ),
    )
    random_rank_rendered = False

    if random_result is not None:
        random_layers = _summary_layers_or_none(
            random_result,
            description="keep01234 random-point QFIM result",
        )

        if random_layers is not None:
            num_random_samples = int(
                np.asarray(
                    random_result.get("num_qfim_samples", NUM_QFIM_SAMPLES),
                    dtype=NP_INT_DTYPE,
                ).reshape(-1)[0]
            )
            rank_by_layer = _load_layer_arrays_from_npz(
                random_result,
                random_layers,
                "rank",
                dtype=NP_REAL_DTYPE,
            )
            eigs_by_layer = _load_layer_arrays_from_npz(
                random_result,
                random_layers,
                "eigs_desc",
                dtype=NP_REAL_DTYPE,
            )
            rho_rank_by_layer = _load_layer_arrays_from_npz(
                random_result,
                random_layers,
                "rho_rank",
                dtype=NP_REAL_DTYPE,
            )
            trace_by_layer = _load_layer_arrays_from_npz(
                random_result,
                random_layers,
                "trace",
                dtype=NP_REAL_DTYPE,
            )
            abs_entry_sum_by_layer = _load_layer_arrays_from_npz(
                random_result,
                random_layers,
                "abs_entry_sum",
                dtype=NP_REAL_DTYPE,
            )

            for L in random_layers:
                if int(L) not in eigs_by_layer:
                    continue
                save_qfim_eigs_by_index(
                    eigs_by_layer[int(L)],
                    title=(
                        rf"QFIM eigenvalues at {num_random_samples} random "
                        rf"points (L={int(L)}, {keep_label_5})"
                    ),
                    outpath=os.path.join(
                        qfim_eigs_dir_red5,
                        f"L{int(L)}_reduced_01234.pdf",
                    ),
                )

            save_qfim_eigs_by_index_colored_by_layer(
                eigs_by_layer,
                random_layers,
                title=(
                    rf"QFIM eigenvalues at {num_random_samples} random "
                    rf"points ({keep_label_5})"
                ),
                outpath=os.path.join(
                    qfim_eigs_dir,
                    "qfim_eigs_by_index_layers_reduced_keep_01234.pdf",
                ),
                cmap=cmap,
            )

            if any(
                np.asarray(values).size > 0
                for values in rank_by_layer.values()
            ):
                plot_single_qfim_rank_mean_min_sem_by_layer(
                    rank_by_layer,
                    rho_rank_by_layer,
                    random_layers,
                    d_keep=2 ** len(KEEP_WIRES_5),
                    color_min="C0",
                    color_mean="C1",
                    label=None,
                    title=(
                        rf"QFIM effective rank mean/minimum and upper bound "
                        rf"at {num_random_samples} random points "
                        rf"({keep_label_5})"
                    ),
                    outpath=os.path.join(
                        qfim_rank_random_dir,
                        (
                            "qfim_rank_mean_min_upper_bound_random_points_"
                            "reduced_01234.pdf"
                        ),
                    ),
                    marker_min="o",
                    marker_mean="s",
                    lw=1.4,
                )
                random_rank_rendered = True

            plot_qfim_random_eigcount_threshold_overlay(
                eigs_by_layer,
                random_layers,
                QFIM_PATH_EIGCOUNT_THRESHOLDS,
                title=(
                    "QFIM eigenvalue count at random points by threshold "
                    f"({keep_label_5})"
                ),
                outpath=os.path.join(
                    qfim_eigcount_random_dir,
                    (
                        "qfim_eigcount_threshold_overlay_random_points_"
                        f"{keep_key_5}.pdf"
                    ),
                ),
                cmap=cmap,
            )

            plot_qfim_trace_max_mean_sem_by_layer(
                trace_by_layer,
                random_layers,
                title=(
                    rf"QFIM trace maximum and mean $\pm$ SEM at "
                    rf"{num_random_samples} random points ({keep_label_5})"
                ),
                outpath=os.path.join(
                    qfim_trace_dir,
                    (
                        "qfim_trace_max_mean_sem_random_points_"
                        "reduced_01234.pdf"
                    ),
                ),
                color_max="C0",
                color_mean="C1",
                marker_max="o",
                marker_mean="s",
                lw=1.4,
                log_scale=False,
            )

            plot_metric_mean_sem_by_layer(
                abs_entry_sum_by_layer,
                random_layers,
                ylabel="Mean elementwise absolute sum",
                title=(
                    rf"QFIM elementwise-absolute-sum mean $\pm$ SEM vs "
                    rf"Layers ({keep_label_5}) at {num_random_samples} "
                    "random points"
                ),
                outpath=os.path.join(
                    qfim_fig_dir,
                    f"qfim_abs_entry_sum_mean_errorbar_{keep_key_5}.pdf",
                ),
                label=r"$\sum_{i,j} |F_{ij}|$",
                color="C1",
                marker="s",
                log_scale=False,
            )
    if not random_rank_rendered:
        # Older result directories contain the active threshold rank in the
        # sensitivity archive even though the full keep01234 spectrum was not
        # persisted.  Use it so the requested rank figures remain available.
        random_rank_fallback_path = os.path.join(
            qfim_results_dir,
            (
                "hamiltonian_qfim_normalized_sensitivity_random_points_"
                f"{keep_key_5}.npz"
            ),
        )
        random_rank_fallback = _load_optional_npz_result(
            random_rank_fallback_path,
            description="keep01234 random-point QFIM-rank fallback",
        )
        if random_rank_fallback is not None:
            fallback_layers = _summary_layers_or_none(
                random_rank_fallback,
                description="keep01234 random-point QFIM-rank fallback",
            )
            if fallback_layers is not None:
                fallback_rank_by_layer = _load_layer_arrays_from_npz(
                    random_rank_fallback,
                    fallback_layers,
                    "active_rank",
                    dtype=NP_REAL_DTYPE,
                )
                plot_single_qfim_rank_mean_min_sem_by_layer(
                    fallback_rank_by_layer,
                    {},
                    fallback_layers,
                    d_keep=2 ** len(KEEP_WIRES_5),
                    color_min="C0",
                    color_mean="C1",
                    label=None,
                    title=(
                        "QFIM effective rank mean/minimum and upper bound "
                        f"at random points ({keep_label_5})"
                    ),
                    outpath=os.path.join(
                        qfim_rank_random_dir,
                        (
                            "qfim_rank_mean_min_upper_bound_random_points_"
                            "reduced_01234.pdf"
                        ),
                    ),
                )

    rank_history_path = os.path.join(
        qfim_results_dir,
        f"qfim_rank_history_optimization_path_{keep_key_5}.npz",
    )
    rank_history_result = _load_optional_npz_result(
        rank_history_path,
        description=(
            "keep01234 optimization-path QFIM-rank result; rerun "
            "DPQC_overparam_compute.py for complete keep01234 figures"
        ),
    )

    def render_rank_history_result(
        result,
        *,
        suffix,
        description: str,
    ) -> bool:
        if result is None:
            return False

        result_layers = _summary_layers_or_none(
            result,
            description=description,
        )
        result_sample_iters = _summary_sample_iters_or_none(
            result,
            description=description,
        )

        if result_layers is None or result_sample_iters is None:
            return False

        result_rank_by_layer = _load_layer_arrays_from_npz(
            result,
            result_layers,
            suffix,
            dtype=NP_REAL_DTYPE,
        )
        if not any(
            np.asarray(values).size > 0
            for values in result_rank_by_layer.values()
        ):
            return False

        plot_qfim_rank_history_mean_by_layer(
            result_rank_by_layer,
            result_layers,
            result_sample_iters,
            title=(
                "Mean QFIM effective rank along optimization path "
                f"({keep_label_5})"
            ),
            outpath=os.path.join(
                qfim_rank_optimization_path_mean_dir,
                (
                    "qfim_rank_mean_history_optimization_path_"
                    f"{keep_key_5}.pdf"
                ),
            ),
            cmap=cmap,
        )
        plot_qfim_rank_history_min_by_layer(
            result_rank_by_layer,
            result_layers,
            result_sample_iters,
            title=(
                "Minimum QFIM effective rank along optimization path "
                f"({keep_label_5})"
            ),
            outpath=os.path.join(
                qfim_rank_optimization_path_min_dir,
                (
                    "qfim_rank_min_history_optimization_path_"
                    f"{keep_key_5}.pdf"
                ),
            ),
            cmap=cmap,
        )
        return True

    rank_history_rendered = render_rank_history_result(
        rank_history_result,
        suffix=None,
        description="keep01234 optimization-path QFIM-rank result",
    )

    if not rank_history_rendered:
        rank_history_fallback_path = os.path.join(
            qfim_results_dir,
            (
                "hamiltonian_qfim_normalized_sensitivity_optimization_path_"
                f"{keep_key_5}.npz"
            ),
        )
        rank_history_result = _load_optional_npz_result(
            rank_history_fallback_path,
            description="keep01234 optimization-path QFIM-rank fallback",
        )
        render_rank_history_result(
            rank_history_result,
            suffix="active_rank",
            description="keep01234 optimization-path QFIM-rank fallback",
        )

    eigs_history_path = os.path.join(
        qfim_results_dir,
        f"qfim_eigs_history_optimization_path_{keep_key_5}.npz",
    )
    eigs_history_result = _load_optional_npz_result(
        eigs_history_path,
        description=(
            "keep01234 optimization-path QFIM-eigenvalue result; rerun "
            "DPQC_overparam_compute.py for spectrum-dependent figures"
        ),
    )

    eigs_history_layers = None
    eigs_history_sample_iters = None
    eigs_history_by_layer = {}

    if eigs_history_result is not None:
        eigs_history_layers = _summary_layers_or_none(
            eigs_history_result,
            description="keep01234 optimization-path QFIM-eigenvalue result",
        )
        eigs_history_sample_iters = _summary_sample_iters_or_none(
            eigs_history_result,
            description="keep01234 optimization-path QFIM-eigenvalue result",
        )

        if (
            eigs_history_layers is not None
            and eigs_history_sample_iters is not None
        ):
            eigs_history_by_layer = _load_layer_arrays_from_npz(
                eigs_history_result,
                eigs_history_layers,
                suffix=None,
                dtype=NP_REAL_DTYPE,
            )
            target_iterations = qfim_path_eig_target_iterations(
                eigs_history_sample_iters,
                every=sample_every,
            )
            save_qfim_eigs_optimization_path_by_iteration(
                eigs_history_by_layer,
                eigs_history_layers,
                eigs_history_sample_iters,
                outdir=os.path.join(
                    qfim_eigs_dir,
                    f"optimization_path_{keep_key_5}",
                ),
                result_key=keep_key_5,
                result_label=keep_label_5,
                target_iterations=target_iterations,
                eps=QFIM_EIG_PLOT_EPS,
            )

            for threshold in QFIM_PATH_EIGCOUNT_THRESHOLDS:
                eigcount_history_by_layer = qfim_eigcount_history_by_layer(
                    eigs_history_by_layer,
                    eigs_history_layers,
                    eigs_history_sample_iters,
                    threshold=threshold,
                )
                threshold_tag = _thr_tag(threshold)
                threshold_label = _threshold_tex(threshold)

                plot_qfim_eigcount_history_mean_by_layer(
                    eigcount_history_by_layer,
                    eigs_history_layers,
                    eigs_history_sample_iters,
                    threshold=threshold,
                    title=(
                        "Mean QFIM eigenvalue count along optimization path "
                        rf"($\lambda_i \geq {threshold_label}$, "
                        f"{keep_label_5})"
                    ),
                    outpath=os.path.join(
                        qfim_eigcount_optimization_path_mean_dir,
                        (
                            "qfim_eigcount_mean_history_optimization_path_"
                            f"thr_{threshold_tag}_{keep_key_5}.pdf"
                        ),
                    ),
                    cmap=cmap,
                )
                plot_qfim_eigcount_history_min_by_layer(
                    eigcount_history_by_layer,
                    eigs_history_layers,
                    eigs_history_sample_iters,
                    threshold=threshold,
                    title=(
                        "Minimum QFIM eigenvalue count along optimization "
                        rf"path ($\lambda_i \geq {threshold_label}$, "
                        f"{keep_label_5})"
                    ),
                    outpath=os.path.join(
                        qfim_eigcount_optimization_path_min_dir,
                        (
                            "qfim_eigcount_min_history_optimization_path_"
                            f"thr_{threshold_tag}_{keep_key_5}.pdf"
                        ),
                    ),
                    cmap=cmap,
                )

    trace_history_path = os.path.join(
        qfim_results_dir,
        f"qfim_trace_history_optimization_path_{keep_key_5}.npz",
    )
    trace_history_result = (
        _load_optional_npz_result(
            trace_history_path,
            description="keep01234 optimization-path QFIM-trace result",
        )
        if os.path.exists(trace_history_path)
        else None
    )

    trace_history_layers = None
    trace_history_sample_iters = None
    trace_history_by_layer = {}

    if trace_history_result is not None:
        trace_history_layers = _summary_layers_or_none(
            trace_history_result,
            description="keep01234 optimization-path QFIM-trace result",
        )
        trace_history_sample_iters = _summary_sample_iters_or_none(
            trace_history_result,
            description="keep01234 optimization-path QFIM-trace result",
        )
        if trace_history_layers is not None:
            trace_history_by_layer = _load_layer_arrays_from_npz(
                trace_history_result,
                trace_history_layers,
                suffix=None,
                dtype=NP_REAL_DTYPE,
            )

    if (
        (
            trace_history_layers is None
            or trace_history_sample_iters is None
            or not trace_history_by_layer
        )
        and eigs_history_layers is not None
        and eigs_history_sample_iters is not None
    ):
        trace_history_layers = eigs_history_layers
        trace_history_sample_iters = eigs_history_sample_iters
        trace_history_by_layer = {
            int(L): np.sum(np.asarray(eigs, dtype=NP_REAL_DTYPE), axis=2)
            for L, eigs in eigs_history_by_layer.items()
        }

    if (
        trace_history_layers is not None
        and trace_history_sample_iters is not None
    ):
        plot_qfim_trace_history_mean_by_layer(
            trace_history_by_layer,
            trace_history_layers,
            trace_history_sample_iters,
            title=(
                "Mean QFIM trace along optimization path "
                f"({keep_label_5})"
            ),
            outpath=os.path.join(
                qfim_trace_dir,
                f"qfim_trace_mean_history_optimization_path_{keep_key_5}.pdf",
            ),
            cmap=cmap,
            log_scale=False,
        )
        plot_metric_mean_sem_by_layer(
            trace_history_by_layer,
            trace_history_layers,
            ylabel="Mean QFIM trace",
            title=(
                rf"QFIM trace mean $\pm$ SEM vs Layers along optimization "
                rf"path ({keep_label_5})"
            ),
            outpath=os.path.join(
                qfim_trace_dir,
                (
                    "qfim_trace_mean_errorbar_optimization_path_"
                    f"{keep_key_5}.pdf"
                ),
            ),
            label=r"$\sum_k \lambda_k(F)$",
            color="C0",
            marker="o",
            log_scale=False,
        )


render_qfim_keep01234_core_figures()


def _finite_sample_mean_sem(samples):
    samples = np.asarray(samples, dtype=NP_REAL_DTYPE).reshape(-1)
    samples = samples[np.isfinite(samples)]
    n = int(samples.size)

    if n == 0:
        return NP_REAL_DTYPE(np.nan), NP_REAL_DTYPE(np.nan), 0

    mean = NP_REAL_DTYPE(np.mean(samples))
    sem = (
        NP_REAL_DTYPE(np.std(samples, ddof=1) / np.sqrt(n))
        if n > 1
        else NP_REAL_DTYPE(0.0)
    )
    return mean, sem, n


def plot_qfim_threshold_vs_participation_random_points(
    threshold_rank_by_layer: dict,
    participation_rank_by_layer: dict,
    layers,
    *,
    state_label: str = keep_label,
    outpath: str,
) -> bool:
    valid_layers = [
        int(L)
        for L in layers
        if int(L) in threshold_rank_by_layer
        and int(L) in participation_rank_by_layer
    ]

    if not valid_layers:
        _warn_skip_new_figure(
            "random-point summary has no layer containing both "
            "'threshold_rank' and 'participation_rank'"
        )
        return False

    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)
    threshold_stats = [
        _finite_sample_mean_sem(threshold_rank_by_layer[L])
        for L in valid_layers
    ]
    participation_stats = [
        _finite_sample_mean_sem(participation_rank_by_layer[L])
        for L in valid_layers
    ]

    threshold_mean = np.asarray(
        [item[0] for item in threshold_stats],
        dtype=NP_REAL_DTYPE,
    )
    threshold_sem = np.asarray(
        [item[1] for item in threshold_stats],
        dtype=NP_REAL_DTYPE,
    )
    participation_mean = np.asarray(
        [item[0] for item in participation_stats],
        dtype=NP_REAL_DTYPE,
    )
    participation_sem = np.asarray(
        [item[1] for item in participation_stats],
        dtype=NP_REAL_DTYPE,
    )

    fig, ax = new_fig_ax(outside_legend=False)

    for means, sems, color, marker, label in (
        (
            threshold_mean,
            threshold_sem,
            "C0",
            "o",
            "Threshold rank",
        ),
        (
            participation_mean,
            participation_sem,
            "C1",
            "s",
            "Participation rank",
        ),
    ):
        finite = np.isfinite(means) & np.isfinite(sems)
        if not np.any(finite):
            continue

        ax.errorbar(
            x[finite],
            means[finite],
            yerr=sems[finite],
            marker=marker,
            linestyle="-",
            linewidth=1.2,
            markersize=5.5,
            capsize=3.0,
            elinewidth=0.8,
            color=color,
            label=label,
        )

    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("QFIM effective dimension")
    ax.set_title(
        rf"Threshold and participation QFIM ranks at random points "
        rf"({state_label})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in valid_layers])
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    save_fig(fig, ax, outpath, outside_legend=False)
    return True


def _normalize_history_array_for_plot(
    values,
    *,
    layer: int,
    metric_name: str,
    num_sample_iters: int,
):
    array = np.asarray(values, dtype=NP_REAL_DTYPE)

    if array.ndim == 1:
        array = array[None, :]

    if array.ndim != 2:
        _warn_skip_new_figure(
            f"{metric_name} for L={layer} must have shape "
            "(num_runs, num_sample_iters); "
            f"received {array.shape}"
        )
        return None

    if array.shape[1] != num_sample_iters and array.shape[0] == num_sample_iters:
        array = array.T

    if array.shape[1] != num_sample_iters:
        _warn_skip_new_figure(
            f"{metric_name} for L={layer} has shape {array.shape}, but "
            f"sample_iters has length {num_sample_iters}"
        )
        return None

    return array


def _history_values_span_orders_of_magnitude(
    history_arrays,
    *,
    minimum_ratio: float = 100.0,
) -> bool:
    positive_parts = []

    for values in history_arrays:
        values = np.asarray(values, dtype=NP_REAL_DTYPE)
        positive = values[np.isfinite(values) & (values > 0.0)]
        if positive.size:
            positive_parts.append(positive)

    if not positive_parts:
        return False

    positive_values = np.concatenate(positive_parts)
    return bool(
        np.max(positive_values)
        / max(float(np.min(positive_values)), np.finfo(NP_REAL_DTYPE).tiny)
        >= float(minimum_ratio)
    )


def plot_qfim_summary_history_mean_sem(
    metric_by_layer: dict,
    layers,
    sample_iters_for_labels,
    *,
    ylabel: str,
    title: str,
    outpath: str,
    metric_name: str,
    auto_log_scale: bool = False,
    cmap=None,
) -> bool:
    x = np.asarray(
        sample_iters_for_labels,
        dtype=NP_REAL_DTYPE,
    ).reshape(-1)
    normalized_by_layer = {}

    for L in layers:
        L_int = int(L)
        if L_int not in metric_by_layer:
            continue

        values = _normalize_history_array_for_plot(
            metric_by_layer[L_int],
            layer=L_int,
            metric_name=metric_name,
            num_sample_iters=int(x.size),
        )
        if values is not None:
            normalized_by_layer[L_int] = values

    valid_layers = [
        int(L)
        for L in layers
        if int(L) in normalized_by_layer
    ]

    if x.size == 0 or not valid_layers:
        _warn_skip_new_figure(
            f"no valid optimization-path arrays for {metric_name}"
        )
        return False

    curve_data = []
    for L in valid_layers:
        means, sems, counts = _finite_mean_sem_over_runs_by_time(
            normalized_by_layer[L]
        )
        finite = np.isfinite(means) & np.isfinite(sems) & (counts > 0)
        curve_data.append((L, means, sems, finite))

    if not any(np.any(item[3]) for item in curve_data):
        _warn_skip_new_figure(
            f"all optimization-path values are non-finite for {metric_name}"
        )
        return False

    log_scale = (
        _history_values_span_orders_of_magnitude(
            [normalized_by_layer[L] for L in valid_layers]
        )
        if auto_log_scale
        else False
    )

    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap
    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.22)
    n_layers = len(valid_layers)

    for layer_idx, (L, means, sems, finite) in enumerate(curve_data):
        if log_scale:
            finite = finite & (means > 0.0)

        if not np.any(finite):
            continue

        color = cmap(layer_idx / max(n_layers - 1, 1))

        if log_scale:
            lower = np.maximum(
                means[finite] - sems[finite],
                np.finfo(NP_REAL_DTYPE).tiny,
            )
            upper = means[finite] + sems[finite]

            ax.plot(
                x[finite],
                means[finite],
                marker="o",
                linestyle="-",
                linewidth=1.2,
                markersize=4.5,
                color=color,
                label=f"L={L}",
            )
            ax.fill_between(
                x[finite],
                lower,
                upper,
                color=color,
                alpha=0.18,
                linewidth=0.0,
            )
        else:
            ax.errorbar(
                x[finite],
                means[finite],
                yerr=sems[finite],
                marker="o",
                linestyle="-",
                linewidth=1.2,
                markersize=4.5,
                capsize=3.0,
                elinewidth=0.8,
                color=color,
                label=f"L={L}",
            )

    ax.set_xlabel("Iterations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(int(iteration)) for iteration in x],
        rotation=45,
        ha="right",
    )

    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.3)
    else:
        ax.grid(True, axis="y", alpha=0.3)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
    )
    save_fig(
        fig,
        ax,
        outpath,
        outside_legend=True,
        legend_space_frac=0.22,
    )
    return True


def _alignment_table_mean_sem_by_eig_index(
    eig_indices,
    values,
    row_mask,
):
    eig_indices = np.asarray(eig_indices, dtype=NP_INT_DTYPE)
    values = np.asarray(values, dtype=NP_REAL_DTYPE)
    row_mask = np.asarray(row_mask, dtype=bool)
    indices = np.unique(eig_indices[row_mask])
    indices = np.sort(indices)

    means = np.full(indices.size, np.nan, dtype=NP_REAL_DTYPE)
    sems = np.full(indices.size, np.nan, dtype=NP_REAL_DTYPE)
    counts = np.zeros(indices.size, dtype=NP_INT_DTYPE)

    for index_position, eig_index in enumerate(indices):
        index_values = values[row_mask & (eig_indices == eig_index)]
        mean, sem, count = _finite_sample_mean_sem(index_values)
        means[index_position] = mean
        sems[index_position] = sem
        counts[index_position] = count

    return indices, means, sems, counts


def plot_qfim_vs_gradient_cumulative_table(
    table: dict,
    *,
    layer: int,
    iteration: int,
    outpath: str,
    source_path: str,
    state_label: str = keep_label,
) -> bool:
    required_keys = (
        "eig_index",
        "cumulative_lambda_fraction",
        "cumulative_gradient_weight",
    )
    missing_keys = [key for key in required_keys if key not in table]

    if missing_keys:
        _warn_skip_new_figure(
            f"cumulative alignment table {source_path} is missing keys "
            f"{missing_keys}"
        )
        return False

    eig_indices = np.asarray(table["eig_index"], dtype=NP_INT_DTYPE).reshape(-1)
    cumulative_lambda = np.asarray(
        table["cumulative_lambda_fraction"],
        dtype=NP_REAL_DTYPE,
    ).reshape(-1)
    cumulative_gradient = np.asarray(
        table["cumulative_gradient_weight"],
        dtype=NP_REAL_DTYPE,
    ).reshape(-1)

    if not (
        eig_indices.size
        == cumulative_lambda.size
        == cumulative_gradient.size
    ):
        _warn_skip_new_figure(
            f"cumulative alignment arrays in {source_path} have "
            "inconsistent lengths"
        )
        return False

    row_mask = np.ones(eig_indices.size, dtype=bool)

    if "layer" in table:
        table_layers = np.asarray(table["layer"], dtype=NP_INT_DTYPE).reshape(-1)
        if table_layers.size != row_mask.size:
            _warn_skip_new_figure(
                f"'layer' has an inconsistent length in {source_path}"
            )
            return False
        row_mask &= table_layers == int(layer)

    if "iteration" in table:
        table_iterations = np.asarray(
            table["iteration"],
            dtype=NP_INT_DTYPE,
        ).reshape(-1)
        if table_iterations.size != row_mask.size:
            _warn_skip_new_figure(
                f"'iteration' has an inconsistent length in {source_path}"
            )
            return False
        row_mask &= table_iterations == int(iteration)

    if not np.any(row_mask):
        return False

    (
        indices_lambda,
        lambda_means,
        lambda_sems,
        lambda_counts,
    ) = _alignment_table_mean_sem_by_eig_index(
        eig_indices,
        cumulative_lambda,
        row_mask,
    )
    (
        indices_gradient,
        gradient_means,
        gradient_sems,
        gradient_counts,
    ) = _alignment_table_mean_sem_by_eig_index(
        eig_indices,
        cumulative_gradient,
        row_mask,
    )

    lambda_finite = np.isfinite(lambda_means) & (lambda_counts > 0)
    gradient_finite = np.isfinite(gradient_means) & (gradient_counts > 0)

    if not np.any(lambda_finite) and not np.any(gradient_finite):
        _warn_skip_new_figure(
            f"cumulative alignment table {source_path} has no finite "
            f"data for L={layer}, iteration={iteration}"
        )
        return False

    fig, ax = new_fig_ax(outside_legend=False)

    for indices, means, sems, finite, color, label in (
        (
            indices_lambda,
            lambda_means,
            lambda_sems,
            lambda_finite,
            "C0",
            "Cumulative QFIM eigenvalue fraction",
        ),
        (
            indices_gradient,
            gradient_means,
            gradient_sems,
            gradient_finite,
            "C1",
            "Cumulative gradient weight",
        ),
    ):
        if not np.any(finite):
            continue

        lower = np.clip(means[finite] - sems[finite], 0.0, 1.0)
        upper = np.clip(means[finite] + sems[finite], 0.0, 1.0)

        ax.plot(
            indices[finite],
            means[finite],
            marker="o",
            linestyle="-",
            linewidth=1.2,
            markersize=3.5,
            color=color,
            label=label,
        )
        ax.fill_between(
            indices[finite],
            lower,
            upper,
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(
        rf"Cumulative QFIM and gradient concentration, "
        rf"L={int(layer)}, iteration {int(iteration)} ({state_label})"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, axis="both", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    save_fig(fig, ax, outpath, outside_legend=False)
    return True


def _load_alignment_table_for_cumulative_figure(
    *,
    layer: int,
    iteration: int,
    final_iteration: int,
    cache: dict,
    result_key: str = keep_key,
):
    alignment_figure_dir, alignment_result_dir = (
        qfim_grad_alignment_dirs_for_key(result_key)
    )
    alignment_roots = (alignment_result_dir, alignment_figure_dir)
    iter_tag = f"iter{int(iteration):06d}"
    exact_candidates = [
        os.path.join(
            root,
            f"L{int(layer)}",
            (
                f"qfim_grad_alignment_scatter_data_L{int(layer)}_"
                f"{iter_tag}.npz"
            ),
        )
        for root in alignment_roots
    ]

    aggregate_candidates = []
    for root in alignment_roots:
        aggregate_candidates.append(
            os.path.join(
                root,
                f"qfim_grad_alignment_scatter_data_L{int(layer)}_all_times.npz",
            )
        )
        if int(iteration) == int(final_iteration):
            aggregate_candidates.append(
                os.path.join(
                    root,
                    (
                        f"qfim_grad_alignment_scatter_data_L{int(layer)}_"
                        "final_iter.npz"
                    ),
                )
            )

    for candidate in exact_candidates + aggregate_candidates:
        if not os.path.exists(candidate):
            continue

        if candidate not in cache:
            try:
                cache[candidate] = load_npz_result(candidate)
            except (OSError, ValueError, KeyError) as exc:
                _warn_skip_new_figure(
                    f"could not load alignment table {candidate}: {exc}"
                )
                cache[candidate] = None

        table = cache[candidate]
        if table is None:
            continue

        is_exact_iteration_file = candidate in exact_candidates
        if is_exact_iteration_file:
            return table, candidate

        if "iteration" not in table:
            continue

        table_iterations = np.asarray(
            table["iteration"],
            dtype=NP_INT_DTYPE,
        ).reshape(-1)
        if np.any(table_iterations == int(iteration)):
            return table, candidate

    return None, None


def render_qfim_spectral_effective_rank_figures(
    result_key: str = keep_key,
    state_label: str = keep_label,
    *,
    include_cumulative_alignment: bool = True,
) -> None:
    random_summary_path = os.path.join(
        qfim_results_dir,
        f"qfim_effective_rank_random_points_{result_key}.npz",
    )
    random_summary = _load_optional_npz_result(
        random_summary_path,
        description="random-point effective-rank summary",
    )

    if random_summary is not None:
        random_layers = _summary_layers_or_none(
            random_summary,
            description="random-point effective-rank summary",
        )
        if random_layers is not None:
            threshold_rank_by_layer = _load_layer_arrays_from_npz(
                random_summary,
                random_layers,
                "threshold_rank",
                dtype=NP_REAL_DTYPE,
            )
            participation_rank_by_layer = _load_layer_arrays_from_npz(
                random_summary,
                random_layers,
                "participation_rank",
                dtype=NP_REAL_DTYPE,
            )
            plot_qfim_threshold_vs_participation_random_points(
                threshold_rank_by_layer,
                participation_rank_by_layer,
                random_layers,
                state_label=state_label,
                outpath=os.path.join(
                    qfim_effective_rank_fig_dir,
                    (
                        "qfim_threshold_vs_participation_rank_"
                        f"random_points_{result_key}.pdf"
                    ),
                ),
            )

    path_summary_path = os.path.join(
        qfim_results_dir,
        (
            "qfim_spectral_gradient_summary_optimization_path_"
            f"{result_key}.npz"
        ),
    )
    path_summary = _load_optional_npz_result(
        path_summary_path,
        description="optimization-path spectral-gradient summary",
    )

    cumulative_layers = [int(L) for L in vqe_layer_list]
    cumulative_sample_iters = np.asarray(sample_iters, dtype=NP_INT_DTYPE)

    if path_summary is not None:
        path_layers = _summary_layers_or_none(
            path_summary,
            description="optimization-path spectral-gradient summary",
        )

        if "sample_iters" not in path_summary:
            _warn_skip_new_figure(
                "optimization-path spectral-gradient summary has no "
                "'sample_iters' key"
            )
            path_sample_iters = None
        else:
            path_sample_iters = np.asarray(
                path_summary["sample_iters"],
                dtype=NP_INT_DTYPE,
            ).reshape(-1)
            if path_sample_iters.size == 0:
                _warn_skip_new_figure(
                    "optimization-path spectral-gradient summary has an "
                    "empty 'sample_iters' array"
                )
                path_sample_iters = None

        if path_layers is not None:
            cumulative_layers = path_layers
        if path_sample_iters is not None:
            cumulative_sample_iters = path_sample_iters

        if path_layers is not None and path_sample_iters is not None:
            history_figure_specs = (
                (
                    "participation_rank",
                    r"QFIM participation rank $r_{\mathrm{part}}(F)$",
                    rf"QFIM participation rank along optimization path "
                    rf"({state_label})",
                    os.path.join(
                        qfim_effective_rank_fig_dir,
                        f"qfim_participation_rank_history_{result_key}.pdf",
                    ),
                    "QFIM participation rank",
                    False,
                ),
                (
                    "gradient_participation_rank",
                    r"Gradient participation rank $r_{\mathrm{grad}}$",
                    rf"Gradient participation rank in the QFIM eigenbasis "
                    rf"({state_label})",
                    os.path.join(
                        qfim_spectral_gradient_fig_dir,
                        f"gradient_participation_rank_history_{result_key}.pdf",
                    ),
                    "gradient participation rank",
                    False,
                ),
                (
                    "gradient_weighted_qfim_eigenvalue",
                    r"Gradient-weighted QFIM eigenvalue $\overline{\lambda}_g$",
                    rf"Gradient-weighted QFIM eigenvalue along optimization "
                    rf"path ({state_label})",
                    os.path.join(
                        qfim_spectral_gradient_fig_dir,
                        (
                            "gradient_weighted_qfim_eigenvalue_history_"
                            f"{result_key}.pdf"
                        ),
                    ),
                    "gradient-weighted QFIM eigenvalue",
                    True,
                ),
            )

            for (
                suffix,
                ylabel,
                title,
                outpath,
                metric_name,
                auto_log_scale,
            ) in history_figure_specs:
                metric_by_layer = _load_layer_arrays_from_npz(
                    path_summary,
                    path_layers,
                    suffix,
                    dtype=NP_REAL_DTYPE,
                )
                plot_qfim_summary_history_mean_sem(
                    metric_by_layer,
                    path_layers,
                    path_sample_iters,
                    ylabel=ylabel,
                    title=title,
                    outpath=outpath,
                    metric_name=metric_name,
                    auto_log_scale=auto_log_scale,
                    cmap=cmap,
                )

    if not include_cumulative_alignment:
        return

    if cumulative_sample_iters.size == 0 or not cumulative_layers:
        _warn_skip_new_figure(
            "no layers or sampled iterations are available for cumulative "
            "QFIM-gradient figures"
        )
        return

    alignment_cache = {}
    missing_alignment_tables = 0
    rendered_cumulative_figures = 0
    final_iteration = int(cumulative_sample_iters[-1])
    cumulative_output_dir = (
        qfim_cumulative_alignment_fig_dir
        if result_key == keep_key
        else os.path.join(qfim_cumulative_alignment_fig_dir, result_key)
    )
    os.makedirs(cumulative_output_dir, exist_ok=True)

    for L in cumulative_layers:
        for iteration in cumulative_sample_iters:
            table, source_path = _load_alignment_table_for_cumulative_figure(
                layer=int(L),
                iteration=int(iteration),
                final_iteration=final_iteration,
                cache=alignment_cache,
                result_key=result_key,
            )
            if table is None:
                missing_alignment_tables += 1
                continue

            rendered = plot_qfim_vs_gradient_cumulative_table(
                table,
                layer=int(L),
                iteration=int(iteration),
                source_path=source_path,
                state_label=state_label,
                outpath=os.path.join(
                    cumulative_output_dir,
                    (
                        f"qfim_vs_gradient_cumulative_L{int(L)}_"
                        f"iter{int(iteration)}.pdf"
                    ),
                ),
            )
            rendered_cumulative_figures += int(rendered)

    if missing_alignment_tables:
        _warn_skip_new_figure(
            f"{missing_alignment_tables} cumulative alignment table(s) "
            "were not found; those figures were skipped"
        )
    elif rendered_cumulative_figures == 0:
        _warn_skip_new_figure(
            "alignment tables were found, but none contained usable "
            "cumulative QFIM-gradient data"
        )


render_qfim_spectral_effective_rank_figures()
render_qfim_spectral_effective_rank_figures(
    keep_key_5,
    keep_label_5,
    include_cumulative_alignment=True,
)

# ============================================================

# ============================================================
# Hamiltonian-direction QFIM-normalized sensitivity
# ============================================================
# The compute entry point stores the threshold-regularized
# chi_H^(tau)(theta) = g(theta)^T F_tau(theta)^+ g(theta) for random points
# and for every run/sampled optimization time.  Individual finite values are
# shown faintly so the aggregation represented by each error bar is visible.
hamiltonian_qfim_sensitivity_fig_dir = os.path.join(
    qfim_fig_dir,
    "hamiltonian_qfim_normalized_sensitivity",
)
os.makedirs(hamiltonian_qfim_sensitivity_fig_dir, exist_ok=True)


def plot_hamiltonian_qfim_sensitivity_final_by_layer(
    chi_by_layer: dict,
    layers,
    *,
    final_iteration: int,
    qfim_threshold: float,
    state_label: str,
    outpath: str,
) -> bool:
    valid_layers = []
    samples_by_layer = {}

    for L in layers:
        L = int(L)
        if L not in chi_by_layer:
            continue

        chi_history = np.asarray(
            chi_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )
        if chi_history.ndim != 2 or chi_history.shape[1] == 0:
            _warn_skip_new_figure(
                f"Hamiltonian-QFIM sensitivity for L={L} must have shape "
                f"(runs, sampled times), got {chi_history.shape}"
            )
            continue

        final_samples = chi_history[:, -1].reshape(-1)
        final_samples = final_samples[np.isfinite(final_samples)]
        if final_samples.size == 0:
            _warn_skip_new_figure(
                f"Hamiltonian-QFIM sensitivity for L={L} has no finite "
                "run values at the final sampled iteration"
            )
            continue

        valid_layers.append(L)
        samples_by_layer[L] = final_samples

    if not valid_layers:
        _warn_skip_new_figure(
            "Hamiltonian-QFIM sensitivity archive has no usable layer data"
        )
        return False

    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)
    stats = [
        _finite_sample_mean_sem(samples_by_layer[L])
        for L in valid_layers
    ]
    means = np.asarray([item[0] for item in stats], dtype=NP_REAL_DTYPE)
    sems = np.asarray([item[1] for item in stats], dtype=NP_REAL_DTYPE)

    fig, ax = new_fig_ax(outside_legend=False)

    for L in valid_layers:
        samples = samples_by_layer[L]
        ax.scatter(
            np.full(samples.shape, float(L), dtype=NP_REAL_DTYPE),
            samples,
            color="C0",
            s=18.0,
            alpha=0.25,
            edgecolors="none",
            zorder=2,
        )

    ax.errorbar(
        x,
        means,
        yerr=sems,
        color="C0",
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=5.0,
        capsize=3.0,
        label=(
            rf"Run mean $\pm$ SEM; {state_label}; "
            rf"$\tau_F={_threshold_tex(qfim_threshold)}$"
        ),
        zorder=3,
    )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(
        r"$\chi_H^{(\tau_F)}="
        r"\boldsymbol{g}^{\mathsf{T}}F_{\tau_F}^{+}\boldsymbol{g}$"
    )
    ax.set_title(
        "Hamiltonian-direction QFIM-normalized sensitivity\n"
        f"final sampled iteration {int(final_iteration)}, "
        rf"$\tau_F={_threshold_tex(qfim_threshold)}$ ({state_label})"
    )
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")

    save_fig(fig, ax, outpath, outside_legend=False)
    return True


def plot_hamiltonian_qfim_sensitivity_random_by_layer(
    chi_by_layer: dict,
    layers,
    *,
    qfim_threshold: float,
    state_label: str,
    outpath: str,
) -> bool:
    valid_layers = []
    samples_by_layer = {}

    for L in layers:
        L = int(L)
        if L not in chi_by_layer:
            continue

        samples = np.asarray(
            chi_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )
        if samples.ndim != 1:
            _warn_skip_new_figure(
                f"random-point Hamiltonian-QFIM sensitivity for L={L} "
                f"must have shape (samples,), got {samples.shape}"
            )
            continue

        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            _warn_skip_new_figure(
                f"random-point Hamiltonian-QFIM sensitivity for L={L} "
                "has no finite sample values"
            )
            continue

        valid_layers.append(L)
        samples_by_layer[L] = samples

    if not valid_layers:
        _warn_skip_new_figure(
            "random-point Hamiltonian-QFIM sensitivity archive has no "
            "usable layer data"
        )
        return False

    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)
    stats = [
        _finite_sample_mean_sem(samples_by_layer[L])
        for L in valid_layers
    ]
    means = np.asarray([item[0] for item in stats], dtype=NP_REAL_DTYPE)
    sems = np.asarray([item[1] for item in stats], dtype=NP_REAL_DTYPE)

    fig, ax = new_fig_ax(outside_legend=False)

    for L in valid_layers:
        samples = samples_by_layer[L]
        ax.scatter(
            np.full(samples.shape, float(L), dtype=NP_REAL_DTYPE),
            samples,
            color="C0",
            s=18.0,
            alpha=0.25,
            edgecolors="none",
            zorder=2,
        )

    ax.errorbar(
        x,
        means,
        yerr=sems,
        color="C0",
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=5.0,
        capsize=3.0,
        label=(
            rf"Sample mean $\pm$ SEM; {state_label}; "
            rf"$\tau_F={_threshold_tex(qfim_threshold)}$"
        ),
        zorder=3,
    )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(
        r"$\chi_H^{(\tau_F)}="
        r"\boldsymbol{g}^{\mathsf{T}}F_{\tau_F}^{+}\boldsymbol{g}$"
    )
    ax.set_title(
        "Hamiltonian-direction QFIM-normalized sensitivity\n"
        "random parameter points, "
        rf"$\tau_F={_threshold_tex(qfim_threshold)}$ ({state_label})"
    )
    ax.set_xticks(x)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best")

    save_fig(fig, ax, outpath, outside_legend=False)
    return True


def render_hamiltonian_qfim_sensitivity_figure(
    *,
    keep_key: str,
    state_label: str,
) -> None:
    description = (
        "Hamiltonian-direction QFIM-normalized sensitivity "
        f"for {state_label} [{keep_key}]"
    )
    result_path = os.path.join(
        qfim_results_dir,
        (
            "hamiltonian_qfim_normalized_sensitivity_optimization_path_"
            f"{keep_key}.npz"
        ),
    )
    result = _load_optional_npz_result(
        result_path,
        description=description,
    )
    if result is None:
        return

    layers = _summary_layers_or_none(
        result,
        description=description,
    )
    if layers is None:
        return

    if "sample_iters" not in result:
        _warn_skip_new_figure(
            "Hamiltonian-QFIM sensitivity archive has no 'sample_iters' key"
        )
        return

    result_sample_iters = np.asarray(
        result["sample_iters"],
        dtype=NP_INT_DTYPE,
    ).reshape(-1)
    if result_sample_iters.size == 0:
        _warn_skip_new_figure(
            "Hamiltonian-QFIM sensitivity archive has an empty "
            "'sample_iters' array"
        )
        return

    if "qfim_eigenvalue_threshold" not in result:
        _warn_skip_new_figure(
            "Hamiltonian-QFIM sensitivity archive has no "
            "'qfim_eigenvalue_threshold' key"
        )
        return

    threshold_values = np.asarray(
        result["qfim_eigenvalue_threshold"],
        dtype=NP_REAL_DTYPE,
    ).reshape(-1)
    if (
        threshold_values.size != 1
        or not np.isfinite(threshold_values[0])
        or threshold_values[0] <= 0.0
    ):
        _warn_skip_new_figure(
            "Hamiltonian-QFIM sensitivity archive has an invalid "
            "'qfim_eigenvalue_threshold' value"
        )
        return
    qfim_threshold = float(threshold_values[0])

    chi_by_layer = _load_layer_arrays_from_npz(
        result,
        layers,
        "chi_hamiltonian",
        dtype=NP_REAL_DTYPE,
    )
    if not chi_by_layer:
        _warn_skip_new_figure(
            "Hamiltonian-QFIM sensitivity archive has no "
            "'L{layer}_chi_hamiltonian' arrays"
        )
        return

    expected_times = int(result_sample_iters.size)
    shape_valid_chi_by_layer = {}
    for L, values in chi_by_layer.items():
        values = np.asarray(values, dtype=NP_REAL_DTYPE)
        if values.ndim != 2 or values.shape[1] != expected_times:
            _warn_skip_new_figure(
                f"Hamiltonian-QFIM sensitivity for L={int(L)} has shape "
                f"{values.shape}; expected (runs, {expected_times})"
            )
            continue
        shape_valid_chi_by_layer[int(L)] = values

    plot_hamiltonian_qfim_sensitivity_final_by_layer(
        shape_valid_chi_by_layer,
        layers,
        final_iteration=int(result_sample_iters[-1]),
        qfim_threshold=qfim_threshold,
        state_label=state_label,
        outpath=os.path.join(
            hamiltonian_qfim_sensitivity_fig_dir,
            (
                "hamiltonian_qfim_normalized_sensitivity_"
                "final_sampled_iteration_"
                f"{keep_key}.pdf"
            ),
        ),
    )


def render_hamiltonian_qfim_sensitivity_random_figure(
    *,
    keep_key: str,
    state_label: str,
) -> None:
    description = (
        "random-point Hamiltonian-direction QFIM-normalized sensitivity "
        f"for {state_label} [{keep_key}]"
    )
    result_path = os.path.join(
        qfim_results_dir,
        (
            "hamiltonian_qfim_normalized_sensitivity_random_points_"
            f"{keep_key}.npz"
        ),
    )
    result = _load_optional_npz_result(
        result_path,
        description=description,
    )
    if result is None:
        return

    layers = _summary_layers_or_none(
        result,
        description=description,
    )
    if layers is None:
        return

    if "qfim_eigenvalue_threshold" not in result:
        _warn_skip_new_figure(
            "random-point Hamiltonian-QFIM sensitivity archive has no "
            "'qfim_eigenvalue_threshold' key"
        )
        return

    if "num_qfim_samples" not in result:
        _warn_skip_new_figure(
            "random-point Hamiltonian-QFIM sensitivity archive has no "
            "'num_qfim_samples' key"
        )
        return

    sample_count_values = np.asarray(
        result["num_qfim_samples"],
        dtype=NP_REAL_DTYPE,
    ).reshape(-1)
    if (
        sample_count_values.size != 1
        or not np.isfinite(sample_count_values[0])
        or sample_count_values[0] <= 0
        or float(sample_count_values[0]) != int(sample_count_values[0])
    ):
        _warn_skip_new_figure(
            "random-point Hamiltonian-QFIM sensitivity archive has an "
            "invalid 'num_qfim_samples' value"
        )
        return
    expected_samples = int(sample_count_values[0])

    threshold_values = np.asarray(
        result["qfim_eigenvalue_threshold"],
        dtype=NP_REAL_DTYPE,
    ).reshape(-1)
    if (
        threshold_values.size != 1
        or not np.isfinite(threshold_values[0])
        or threshold_values[0] <= 0.0
    ):
        _warn_skip_new_figure(
            "random-point Hamiltonian-QFIM sensitivity archive has an "
            "invalid 'qfim_eigenvalue_threshold' value"
        )
        return
    qfim_threshold = float(threshold_values[0])

    chi_by_layer = _load_layer_arrays_from_npz(
        result,
        layers,
        "chi_hamiltonian",
        dtype=NP_REAL_DTYPE,
    )
    if not chi_by_layer:
        _warn_skip_new_figure(
            "random-point Hamiltonian-QFIM sensitivity archive has no "
            "'L{layer}_chi_hamiltonian' arrays"
        )
        return

    shape_valid_chi_by_layer = {}
    for L, values in chi_by_layer.items():
        values = np.asarray(values, dtype=NP_REAL_DTYPE)
        if values.shape != (expected_samples,):
            _warn_skip_new_figure(
                f"random-point Hamiltonian-QFIM sensitivity for L={int(L)} "
                f"has shape {values.shape}; expected ({expected_samples},)"
            )
            continue
        shape_valid_chi_by_layer[int(L)] = values

    plot_hamiltonian_qfim_sensitivity_random_by_layer(
        shape_valid_chi_by_layer,
        layers,
        qfim_threshold=qfim_threshold,
        state_label=state_label,
        outpath=os.path.join(
            hamiltonian_qfim_sensitivity_fig_dir,
            (
                "hamiltonian_qfim_normalized_sensitivity_random_points_"
                f"{keep_key}.pdf"
            ),
        ),
    )


HAMILTONIAN_QFIM_SENSITIVITY_STATES = (
    ("keep0123", "Reduced (0,1,2,3)"),
    ("keep01234", "Reduced (0,1,2,3,4)"),
)

for _sensitivity_keep_key, _sensitivity_state_label in (
    HAMILTONIAN_QFIM_SENSITIVITY_STATES
):
    render_hamiltonian_qfim_sensitivity_figure(
        keep_key=_sensitivity_keep_key,
        state_label=_sensitivity_state_label,
    )
    render_hamiltonian_qfim_sensitivity_random_figure(
        keep_key=_sensitivity_keep_key,
        state_label=_sensitivity_state_label,
    )

# ============================================================

# ============================================================
# QFIM-gradient alignment figures from saved scatter data
# ============================================================
# QFIM eigenvalue vs gradient-direction weight scatter plots
#   x-axis: QFIM eigenvalue lambda_i
#   y-axis: w_i^grad = |v_i^T g|^2 / sum_j |v_j^T g|^2
#
# This section uses optimization-path samples already stored in
#   theta_sample_traces_by_layer[L]
#   grad_sample_traces_by_layer[L]
# and constructs one scatter plot per available VQE layer.
#
# Mathematical meaning of the diagnostic
# --------------------------------------
# Fix one optimization point theta and let
#
#     L(theta) : loss / energy objective,
#     g(theta) = grad_theta L(theta),
#     F(theta) : QFIM at theta.
#
# Since the QFIM is Hermitian positive semidefinite, we diagonalize it as
#
#     F(theta) v_i = lambda_i v_i,
#     v_i^dagger v_j = delta_ij,
#     lambda_i >= 0.
#
# The eigenvectors v_i give orthonormal directions in parameter space, while
# lambda_i measures how strongly an infinitesimal parameter displacement in
# that direction changes the quantum state. Large lambda_i are geometrically
# sensitive directions; very small lambda_i are nearly redundant / flat
# directions of the variational state manifold.
#
# The ordinary loss gradient is expanded in this QFIM eigenbasis:
#
#     g(theta) = sum_i c_i v_i,
#     c_i = v_i^dagger g(theta).
#
# We then plot the normalized squared component
#
#     w_i^grad = |c_i|^2 / sum_j |c_j|^2.
#
# Thus w_i^grad is the fraction of the Euclidean gradient norm carried by
# the i-th QFIM eigen-direction. The weights satisfy sum_i w_i^grad = 1
# whenever the gradient norm is nonzero.
#
# Each scatter point is one eigen-direction at one sampled optimization
# state: x = lambda_i, y = w_i^grad. If many high-weight points lie at large
# lambda_i, the loss gradient mainly points along directions that strongly
# change the quantum state. If high-weight points lie at tiny lambda_i, the
# gradient is dominated by directions that barely move the represented state,
# which can indicate overparameterization or geometric redundancy.
#
# This is a diagnostic projection of the ordinary gradient onto the QFIM
# eigenbasis. It is not the natural-gradient update F^{-1} g; no inverse QFIM
# is applied here. The small positive floors below are only for numerical
# safety in log-scale visualization.
# ============================================================

QFIM_GRAD_ALIGN_EIG_FLOOR = 1e-16
QFIM_GRAD_ALIGN_WEIGHT_FLOOR = 1e-16
QFIM_GRAD_ALIGN_NORM_EPS = 1e-24

qfim_grad_align_dir = os.path.join(qfim_fig_dir, "qfim_grad_alignment")
qfim_grad_align_results_dir = os.path.join(qfim_results_dir, "qfim_grad_alignment")
os.makedirs(qfim_grad_align_dir, exist_ok=True)
os.makedirs(qfim_grad_align_results_dir, exist_ok=True)


# ------------------------------------------------------------

# ------------------------------------------------------------
# Execution settings for visualization from saved alignment data
# ------------------------------------------------------------
RUN_QFIM_GRAD_ALIGNMENT_FINAL_ITER = cfg.RUN_QFIM_GRAD_ALIGNMENT_FINAL_ITER
RUN_QFIM_GRAD_ALIGNMENT_ALL_TIMES = cfg.RUN_QFIM_GRAD_ALIGNMENT_ALL_TIMES
RUN_QFIM_GRAD_ALIGNMENT_PER_ITERATION = cfg.RUN_QFIM_GRAD_ALIGNMENT_PER_ITERATION

LOG_X_QFIM_GRAD_ALIGNMENT = cfg.LOG_X_QFIM_GRAD_ALIGNMENT
LOG_Y_QFIM_GRAD_ALIGNMENT = cfg.LOG_Y_QFIM_GRAD_ALIGNMENT
if cfg.QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS is None:
    QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS = tuple(int(t) for t in sample_iters)
else:
    QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS = tuple(
        int(t) for t in cfg.QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS
    )

def _load_alignment_table_or_none(path: str):
    if not os.path.exists(path):
        return None
    return load_npz_result(path)

def render_saved_qfim_grad_alignment_figures(
    result_key: str,
    state_label: str,
) -> None:
    alignment_figure_dir, alignment_result_dir = (
        qfim_grad_alignment_dirs_for_key(result_key)
    )

    if RUN_QFIM_GRAD_ALIGNMENT_FINAL_ITER or RUN_QFIM_GRAD_ALIGNMENT_ALL_TIMES:
        for use_all_times in (False, True):
            if use_all_times and not RUN_QFIM_GRAD_ALIGNMENT_ALL_TIMES:
                continue
            if (not use_all_times) and not RUN_QFIM_GRAD_ALIGNMENT_FINAL_ITER:
                continue

            time_tag = "all_times" if use_all_times else "final_iter"
            title_time = (
                "all sampled iterations"
                if use_all_times
                else f"final iteration {int(sample_iters[-1])}"
            )
            color_by = "iteration" if use_all_times else None
            point_size = 12.0 if use_all_times else 14.0
            scatter_alpha = 0.40 if use_all_times else 0.45

            table_by_layer = {}
            for L in vqe_layer_list:
                path = os.path.join(
                    alignment_result_dir,
                    (
                        "qfim_grad_alignment_scatter_data_"
                        f"L{int(L)}_{time_tag}.npz"
                    ),
                )
                table = _load_alignment_table_or_none(path)
                if table is None:
                    continue
                table_by_layer[int(L)] = table
                plot_qfim_grad_alignment_table(
                    table,
                    title=(
                        "QFIM eigenvalue vs gradient weight, "
                        f"L={int(L)}, {title_time} ({state_label})"
                    ),
                    outpath=os.path.join(
                        alignment_figure_dir,
                        (
                            "qfim_grad_weight_scatter_"
                            f"L{int(L)}_{time_tag}.pdf"
                        ),
                    ),
                    log_x=LOG_X_QFIM_GRAD_ALIGNMENT,
                    log_y=LOG_Y_QFIM_GRAD_ALIGNMENT,
                    color_by=color_by,
                    point_size=point_size,
                    alpha=scatter_alpha,
                )

            if table_by_layer:
                plot_qfim_grad_alignment_layer_overlay(
                    table_by_layer,
                    sorted(table_by_layer),
                    title=(
                        "QFIM eigenvalue vs gradient weight across layers, "
                        f"{time_tag.replace('_', ' ')} ({state_label})"
                    ),
                    outpath=os.path.join(
                        alignment_figure_dir,
                        (
                            "qfim_grad_weight_scatter_overlay_layers_"
                            f"{time_tag}.pdf"
                        ),
                    ),
                    log_x=LOG_X_QFIM_GRAD_ALIGNMENT,
                    log_y=LOG_Y_QFIM_GRAD_ALIGNMENT,
                    point_size=12.0,
                    alpha=0.40,
                )

    if RUN_QFIM_GRAD_ALIGNMENT_PER_ITERATION:
        for L in vqe_layer_list:
            layer_dir = os.path.join(
                alignment_figure_dir,
                f"L{int(L)}",
            )
            os.makedirs(layer_dir, exist_ok=True)
            for iteration in QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS:
                iteration = int(iteration)
                iter_tag = f"iter{iteration:06d}"
                path = os.path.join(
                    alignment_result_dir,
                    f"L{int(L)}",
                    (
                        "qfim_grad_alignment_scatter_data_"
                        f"L{int(L)}_{iter_tag}.npz"
                    ),
                )
                table = _load_alignment_table_or_none(path)
                if table is None:
                    continue
                plot_qfim_grad_alignment_table(
                    table,
                    title=(
                        "QFIM eigenvalue vs gradient weight, "
                        f"L={int(L)}, iteration {iteration} ({state_label})"
                    ),
                    outpath=os.path.join(
                        layer_dir,
                        (
                            "qfim_grad_weight_scatter_"
                            f"L{int(L)}_{iter_tag}.pdf"
                        ),
                    ),
                    log_x=LOG_X_QFIM_GRAD_ALIGNMENT,
                    log_y=LOG_Y_QFIM_GRAD_ALIGNMENT,
                    color_by=None,
                    point_size=14.0,
                    alpha=0.45,
                )


for _alignment_result_key, _alignment_state_label in (
    (keep_key, keep_label),
    (keep_key_5, keep_label_5),
):
    render_saved_qfim_grad_alignment_figures(
        _alignment_result_key,
        _alignment_state_label,
    )

print(f"Saved figures to: {save_dir}")
