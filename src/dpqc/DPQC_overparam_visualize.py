#!/usr/bin/env python
# coding: utf-8
"""Visualize saved DPQC overparameterization numerical results.

Run DPQC_overparam_compute.py first to create the .npz files under
figs/dpqc/h_<h_param>/numerical_results. This script loads those saved
results and generates the figures without recomputing VQE/QFIM quantities.
"""


import os
import sys
from pathlib import Path
from typing import Optional, Tuple

_MODULE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _MODULE_DIR.parent
_COMMON_DIR = _SRC_DIR / "common"
for _path in (_MODULE_DIR, _COMMON_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import config_overparam as cfg

# ------------------------------------------------------------
# IMPORTANT: env vars should be set BEFORE importing jax
# ------------------------------------------------------------
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import matplotlib
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch
from plot import (
    new_fig_ax,
    plot_qfim_grad_alignment_layer_overlay,
    plot_qfim_grad_alignment_table,
    save_eigenvalue_histogram_across_trials,
    save_eigenvalue_histograms_by_trial,
    save_fig,
)
from tqdm.auto import tqdm

NP_REAL_DTYPE = np.float64
NP_COMPLEX_DTYPE = np.complex128
NP_INT_DTYPE = np.int64

# Shared visual language: metric determines color; statistic determines line.
METRIC_COLORS = {
    "qfim": "#0072B2",
    "hs": "#D55E00",
    "ortk": "#009E73",
    "hessian": "#CC79A7",
    "energy": "#E69F00",
}
STATISTIC_LINESTYLES = {"min": ":", "mean": "-", "max": "--"}

def run_dpqc_overparam_visualize() -> None:
    import jax
    import jax.numpy as jnp
    import tensorcircuit as tc
    from dpqc_overparam_common import (
        _thr_tag,
        build_H_matrix_jax,
        build_layer_list,
        hamiltonian_terms,
        load_npz_result,
        rho_zero_state,
    )
    from qfim import rank_threshold_from_eigvals

    jax.config.update("jax_enable_x64", True)

    tc.set_backend("jax")
    tc.set_dtype("complex128")

    REAL_DTYPE = jnp.float64
    COMPLEX_DTYPE = jnp.complex128
    NP_REAL_DTYPE = np.float64
    NP_COMPLEX_DTYPE = np.complex128
    NP_INT_DTYPE = np.int64

    # ============================================================
    # Shared constants / helpers
    # ============================================================
    num_system_qubits = 5
    h_param = cfg.H_PARAM
    tolerance = cfg.TOLERANCE
    steps = cfg.STEPS
    num_runs = cfg.NUM_RUNS
    lr = cfg.LEARNING_RATE

    # Optimization-history sampling points used for history plots and
    # QFIM-gradient sector diagnostics.
    eps = 1e-12
    sample_every = cfg.SAMPLE_EVERY

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
    num_params_per_layer = NUM_BLOCKS * PARAMS_PER_BLOCK + EXTRA_PARAMS_PER_LAYER

    TOP, LEFT, RIGHT, BOTTOM, ANC_CENTER, FRESH_ANCILLA = 0, 1, 2, 3, 4, 5

    LAYER_PAIRS = (
        (LEFT, BOTTOM),
        (RIGHT, BOTTOM),
        (TOP, RIGHT),
        (TOP, ANC_CENTER),
    )

    RED4_COLOR = "blue"

    # Reduced-system QFIM identifier used in filenames and figure titles.
    # Define these near the top so later cells/sections cannot hit NameError.
    keep_key = "keep0123"
    keep_label = "Reduced (0,1,2,3)"


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

    figures_dir = os.path.join(save_dir, "figures")
    energy_fig_dir = os.path.join(figures_dir, "energy")
    qfim_fig_dir = os.path.join(figures_dir, "qfim")
    hs_fig_dir = os.path.join(figures_dir, "hs")
    ortk_fig_dir = os.path.join(figures_dir, "ortk")
    hessian_fig_dir = os.path.join(figures_dir, "hessian")
    qfim_keep0123_fig_dir = os.path.join(qfim_fig_dir, "reduced_keep_0123")
    qfim_keep01234_fig_dir = os.path.join(qfim_fig_dir, "reduced_keep_01234")
    hs_keep0123_fig_dir = os.path.join(hs_fig_dir, "reduced_keep_0123")
    hs_keep01234_fig_dir = os.path.join(hs_fig_dir, "reduced_keep_01234")
    ortk_keep01234_fig_dir = os.path.join(ortk_fig_dir, "reduced_keep_01234")
    hessian_keep01234_fig_dir = os.path.join(hessian_fig_dir, "reduced_keep_01234")
    qfim_eigs_dir = os.path.join(qfim_keep0123_fig_dir, "eigs")
    qfim_eigs_dir_red4 = qfim_eigs_dir
    qfim_rank_dir = os.path.join(qfim_keep0123_fig_dir, "rank")
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
    qfim_eigcount_dir = os.path.join(qfim_keep0123_fig_dir, "eigcount")
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
    hs_eigs_dir = os.path.join(hs_keep0123_fig_dir, "eigs")
    hs_eigs_dir_red4 = hs_eigs_dir
    hs_rank_dir = os.path.join(hs_keep0123_fig_dir, "rank")
    hs_rank_random_dir = os.path.join(hs_rank_dir, "random_points")
    hs_rank_optimization_path_dir = os.path.join(hs_rank_dir, "optimization_path")
    hs_rank_optimization_path_mean_dir = os.path.join(
        hs_rank_optimization_path_dir,
        "mean",
    )
    hs_rank_optimization_path_min_dir = os.path.join(
        hs_rank_optimization_path_dir,
        "min",
    )
    hs_eigcount_dir = os.path.join(hs_keep0123_fig_dir, "eigcount")
    hs_eigcount_random_dir = os.path.join(hs_eigcount_dir, "random_points")
    hs_eigcount_optimization_path_dir = os.path.join(
        hs_eigcount_dir,
        "optimization_path",
    )
    hs_eigcount_optimization_path_mean_dir = os.path.join(
        hs_eigcount_optimization_path_dir,
        "mean",
    )
    hs_eigcount_optimization_path_min_dir = os.path.join(
        hs_eigcount_optimization_path_dir,
        "min",
    )
    ortk_eigs_dir = os.path.join(ortk_keep01234_fig_dir, "eigs")
    ortk_rank_dir = os.path.join(ortk_keep01234_fig_dir, "rank")
    ortk_rank_random_dir = os.path.join(ortk_rank_dir, "random_points")
    ortk_rank_optimization_path_dir = os.path.join(
        ortk_rank_dir,
        "optimization_path",
    )
    ortk_rank_optimization_path_mean_dir = os.path.join(
        ortk_rank_optimization_path_dir,
        "mean",
    )
    ortk_rank_optimization_path_min_dir = os.path.join(
        ortk_rank_optimization_path_dir,
        "min",
    )
    ortk_effective_rank_dir = os.path.join(ortk_keep01234_fig_dir, "effective_rank")
    ortk_effective_rank_random_dir = os.path.join(
        ortk_effective_rank_dir,
        "random_points",
    )
    ortk_effective_rank_optimization_path_dir = os.path.join(
        ortk_effective_rank_dir,
        "optimization_path",
    )
    ortk_effective_rank_optimization_path_mean_dir = os.path.join(
        ortk_effective_rank_optimization_path_dir,
        "mean",
    )
    ortk_effective_rank_optimization_path_min_dir = os.path.join(
        ortk_effective_rank_optimization_path_dir,
        "min",
    )
    hessian_eigs_dir = os.path.join(hessian_keep01234_fig_dir, "eigs")
    hessian_rank_dir = os.path.join(hessian_keep01234_fig_dir, "rank")
    hessian_rank_random_dir = os.path.join(hessian_rank_dir, "random_points")
    hessian_rank_optimization_path_dir = os.path.join(
        hessian_rank_dir,
        "optimization_path",
    )
    hessian_rank_optimization_path_mean_dir = os.path.join(
        hessian_rank_optimization_path_dir,
        "mean",
    )
    hessian_rank_optimization_path_min_dir = os.path.join(
        hessian_rank_optimization_path_dir,
        "min",
    )
    qfim_trace_dir = os.path.join(qfim_keep0123_fig_dir, "trace")
    qfim_trace_random_dir = os.path.join(qfim_trace_dir, "random_points")
    qfim_trace_optimization_path_dir = os.path.join(
        qfim_trace_dir,
        "optimization_path",
    )
    qfim_abs_entry_sum_dir = os.path.join(qfim_keep0123_fig_dir, "abs_entry_sum")
    qfim_abs_entry_sum_random_dir = os.path.join(
        qfim_abs_entry_sum_dir,
        "random_points",
    )
    hs_trace_dir = os.path.join(hs_keep0123_fig_dir, "trace")
    hs_trace_random_dir = os.path.join(hs_trace_dir, "random_points")
    hs_trace_optimization_path_dir = os.path.join(
        hs_trace_dir,
        "optimization_path",
    )
    hs_abs_entry_sum_dir = os.path.join(hs_keep0123_fig_dir, "abs_entry_sum")
    hs_abs_entry_sum_random_dir = os.path.join(
        hs_abs_entry_sum_dir,
        "random_points",
    )
    ortk_trace_dir = os.path.join(ortk_keep01234_fig_dir, "trace")
    ortk_trace_optimization_path_dir = os.path.join(
        ortk_trace_dir,
        "optimization_path",
    )
    hessian_trace_dir = os.path.join(hessian_keep01234_fig_dir, "trace")
    hessian_trace_optimization_path_dir = os.path.join(
        hessian_trace_dir,
        "optimization_path",
    )
    hessian_abs_eigsum_dir = os.path.join(hessian_keep01234_fig_dir, "abs_eigsum")
    hessian_abs_eigsum_random_dir = os.path.join(
        hessian_abs_eigsum_dir,
        "random_points",
    )
    hessian_abs_eigsum_optimization_path_dir = os.path.join(
        hessian_abs_eigsum_dir,
        "optimization_path",
    )
    circuit_dir = os.path.join(save_dir, "optimized_circuits")
    numerical_results_dir = os.path.join(save_dir, "numerical_results")
    energy_results_dir = os.path.join(numerical_results_dir, "energy")
    qfim_results_dir = os.path.join(numerical_results_dir, "qfim")
    hs_results_dir = os.path.join(numerical_results_dir, "hs")
    ortk_results_dir = os.path.join(numerical_results_dir, "ortk")
    hessian_results_dir = os.path.join(numerical_results_dir, "hessian")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(energy_fig_dir, exist_ok=True)
    os.makedirs(qfim_fig_dir, exist_ok=True)
    os.makedirs(hs_fig_dir, exist_ok=True)
    os.makedirs(ortk_fig_dir, exist_ok=True)
    os.makedirs(hessian_fig_dir, exist_ok=True)
    os.makedirs(qfim_eigs_dir, exist_ok=True)
    os.makedirs(qfim_eigs_dir_red4, exist_ok=True)
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
    os.makedirs(hs_eigs_dir, exist_ok=True)
    os.makedirs(hs_eigs_dir_red4, exist_ok=True)
    os.makedirs(hs_rank_dir, exist_ok=True)
    os.makedirs(hs_rank_random_dir, exist_ok=True)
    os.makedirs(hs_rank_optimization_path_dir, exist_ok=True)
    os.makedirs(hs_rank_optimization_path_mean_dir, exist_ok=True)
    os.makedirs(hs_rank_optimization_path_min_dir, exist_ok=True)
    os.makedirs(hs_eigcount_dir, exist_ok=True)
    os.makedirs(hs_eigcount_random_dir, exist_ok=True)
    os.makedirs(hs_eigcount_optimization_path_dir, exist_ok=True)
    os.makedirs(hs_eigcount_optimization_path_mean_dir, exist_ok=True)
    os.makedirs(hs_eigcount_optimization_path_min_dir, exist_ok=True)
    os.makedirs(ortk_eigs_dir, exist_ok=True)
    os.makedirs(ortk_rank_dir, exist_ok=True)
    os.makedirs(ortk_rank_random_dir, exist_ok=True)
    os.makedirs(ortk_rank_optimization_path_dir, exist_ok=True)
    os.makedirs(ortk_rank_optimization_path_mean_dir, exist_ok=True)
    os.makedirs(ortk_rank_optimization_path_min_dir, exist_ok=True)
    os.makedirs(ortk_effective_rank_dir, exist_ok=True)
    os.makedirs(ortk_effective_rank_random_dir, exist_ok=True)
    os.makedirs(ortk_effective_rank_optimization_path_dir, exist_ok=True)
    os.makedirs(ortk_effective_rank_optimization_path_mean_dir, exist_ok=True)
    os.makedirs(ortk_effective_rank_optimization_path_min_dir, exist_ok=True)
    os.makedirs(hessian_eigs_dir, exist_ok=True)
    os.makedirs(hessian_rank_dir, exist_ok=True)
    os.makedirs(hessian_rank_random_dir, exist_ok=True)
    os.makedirs(hessian_rank_optimization_path_dir, exist_ok=True)
    os.makedirs(hessian_rank_optimization_path_mean_dir, exist_ok=True)
    os.makedirs(hessian_rank_optimization_path_min_dir, exist_ok=True)
    os.makedirs(qfim_trace_random_dir, exist_ok=True)
    os.makedirs(qfim_trace_optimization_path_dir, exist_ok=True)
    os.makedirs(qfim_abs_entry_sum_random_dir, exist_ok=True)
    os.makedirs(hs_trace_random_dir, exist_ok=True)
    os.makedirs(hs_trace_optimization_path_dir, exist_ok=True)
    os.makedirs(hs_abs_entry_sum_random_dir, exist_ok=True)
    os.makedirs(ortk_trace_optimization_path_dir, exist_ok=True)
    os.makedirs(hessian_trace_optimization_path_dir, exist_ok=True)
    os.makedirs(hessian_abs_eigsum_random_dir, exist_ok=True)
    os.makedirs(hessian_abs_eigsum_optimization_path_dir, exist_ok=True)
    os.makedirs(circuit_dir, exist_ok=True)
    os.makedirs(numerical_results_dir, exist_ok=True)
    os.makedirs(energy_results_dir, exist_ok=True)
    os.makedirs(qfim_results_dir, exist_ok=True)
    os.makedirs(hs_results_dir, exist_ok=True)
    os.makedirs(ortk_results_dir, exist_ok=True)
    os.makedirs(hessian_results_dir, exist_ok=True)


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

    plot_single_line_by_layer(
        np.array(final_stats["layer"], dtype=NP_REAL_DTYPE),
        np.array(final_stats["success_rate"], dtype=NP_REAL_DTYPE),
        ylabel="Success Rate",
        title=f"Success Rate over {num_runs} runs (Tol={tolerance})",
        outpath=os.path.join(energy_fig_dir, "success_rate.pdf"),
        label="Success Rate",
        ylim=(-0.02, 1.02),
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
    # QFIM rank + eigenvalue plots
    #   Only reduced-system QFIM for keep=(0,1,2,3)
    # ============================================================
    KEEP_WIRES_4 = (0, 1, 2, 3)
    assert KEEP_WIRES_4 == tuple(range(num_system_qubits - 1))

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
        *,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        eigs_arr = np.asarray(eigs_sorted_desc, dtype=NP_REAL_DTYPE)

        if eigs_arr.ndim == 1:
            eigs_arr = eigs_arr[None, :]

        eigs_jnp = jnp.asarray(eigs_arr, dtype=REAL_DTYPE)

        thresholds = jax.vmap(
            lambda evals_1d: rank_threshold_from_eigvals(
                evals_1d,
                threshold=threshold,
            )
        )(eigs_jnp)

        return np.asarray(jax.device_get(thresholds), dtype=NP_REAL_DTYPE)


    def eigenvalue_index_ticks(num_params: int, *, max_ticks: int = 11) -> np.ndarray:
        num_params = int(num_params)

        if num_params <= 0:
            return np.asarray([], dtype=NP_INT_DTYPE)

        max_ticks = max(2, int(max_ticks))

        if num_params <= max_ticks:
            return np.arange(1, num_params + 1, dtype=NP_INT_DTYPE)

        ticks = np.rint(
            np.linspace(1, num_params, num=max_ticks)
        ).astype(NP_INT_DTYPE)

        ticks = np.unique(ticks)
        ticks[0] = 1
        ticks[-1] = num_params

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
        ylabel: str = "QFIM eigenvalue",
        rank_threshold: Optional[float] = None,
    ) -> None:
        eigs_raw = np.asarray(eigs_sorted_desc, dtype=NP_REAL_DTYPE)

        if eigs_raw.ndim == 1:
            eigs_raw = eigs_raw[None, :]

        eigs_plot = eigs_raw.copy()
        eigs_plot[eigs_plot <= 0.0] = eps

        num_params = int(eigs_plot.shape[1])

        rank_thresholds = _rank_thresholds_for_eigs_sorted_desc(
            eigs_raw,
            threshold=rank_threshold,
        )
        rank_thresholds_plot = np.maximum(rank_thresholds, eps)

        fig, ax = new_fig_ax(outside_legend=False)

        positions = np.arange(1, num_params + 1, dtype=NP_REAL_DTYPE)

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

        ticks = eigenvalue_index_ticks(num_params, max_ticks=11)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlim(0.5, num_params + 0.5)
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
        ax.set_ylabel(ylabel)
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
        ylabel: str = "QFIM eigenvalue",
    ) -> None:
        cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap

        layers = [
            int(L)
            for L in layer_list
            if eigs_sorted_desc_by_layer.get(L) is not None
        ]

        if not layers:
            return

        max_num_params = max(
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

            for i in range(max_num_params):
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

        ticks = eigenvalue_index_ticks(max_num_params, max_ticks=11)

        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlim(0.5, max_num_params + 0.5)
        ax.set_yscale("log")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel(ylabel)
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


    def save_signed_eigs_by_index(
        eigs_sorted_desc: np.ndarray,
        *,
        title: str,
        outpath: str,
        threshold: float = QFIM_EFFECTIVE_RANK_THRESHOLD,
        eps: float = QFIM_EIG_PLOT_EPS,
        point_size: float = 14.0,
        alpha: float = 0.55,
        ylabel: str = "Hessian eigenvalue",
    ) -> None:
        eigs = np.asarray(eigs_sorted_desc, dtype=NP_REAL_DTYPE)

        if eigs.ndim == 1:
            eigs = eigs[None, :]

        num_params = int(eigs.shape[1])
        threshold = float(threshold)
        linthresh = max(float(eps), threshold)

        fig, ax = new_fig_ax(outside_legend=False)
        positions = np.arange(1, num_params + 1, dtype=NP_REAL_DTYPE)

        for i, x0 in enumerate(positions):
            y = eigs[:, i]
            finite = np.isfinite(y)
            if not np.any(finite):
                continue
            ax.scatter(
                np.full(np.sum(finite), x0, dtype=NP_REAL_DTYPE),
                y[finite],
                s=point_size,
                color="C6",
                alpha=alpha,
                edgecolors="black",
                linewidths=0.20,
                rasterized=True,
            )

        ticks = eigenvalue_index_ticks(num_params, max_ticks=11)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlim(0.5, num_params + 0.5)
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.axhline(0.0, linestyle="-", linewidth=0.8, color="black", alpha=0.45)

        if threshold > 0.0:
            ax.axhline(
                threshold,
                linestyle="-",
                linewidth=1.2,
                color="red",
                alpha=0.75,
                zorder=4,
            )
            ax.axhline(
                -threshold,
                linestyle="-",
                linewidth=1.2,
                color="red",
                alpha=0.75,
                zorder=4,
            )

        finite_vals = eigs[np.isfinite(eigs)]
        if finite_vals.size > 0:
            max_abs = max(float(np.max(np.abs(finite_vals))), threshold, linthresh)
            ax.set_ylim(-1.5 * max_abs, 1.5 * max_abs)

        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)

        save_fig(fig, ax, outpath, outside_legend=False)


    def save_signed_eigs_by_index_colored_by_layer(
        eigs_sorted_desc_by_layer: dict,
        layer_list,
        *,
        title: str,
        outpath: str,
        threshold: float = QFIM_EFFECTIVE_RANK_THRESHOLD,
        eps: float = QFIM_EIG_PLOT_EPS,
        cmap=None,
        alpha: float = 0.50,
        point_size: float = 10.0,
        ylabel: str = "Hessian eigenvalue",
    ) -> None:
        cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap

        layers = [
            int(L)
            for L in layer_list
            if eigs_sorted_desc_by_layer.get(int(L)) is not None
        ]

        if not layers:
            return

        max_num_params = max(
            int(np.asarray(eigs_sorted_desc_by_layer[L]).shape[1])
            for L in layers
        )
        threshold = float(threshold)
        linthresh = max(float(eps), threshold)

        fig, ax = new_fig_ax(outside_legend=True)
        handles = []
        max_abs = max(linthresh, threshold)

        for idx, L in enumerate(layers):
            color = cmap(idx / max(len(layers) - 1, 1))
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
            finite_vals = eigs[np.isfinite(eigs)]
            if finite_vals.size > 0:
                max_abs = max(max_abs, float(np.max(np.abs(finite_vals))))

            for i in range(max_num_params):
                if i >= eigs.shape[1]:
                    continue

                y = eigs[:, i]
                finite = np.isfinite(y)
                if not np.any(finite):
                    continue

                ax.scatter(
                    np.full(np.sum(finite), i + 1, dtype=NP_REAL_DTYPE),
                    y[finite],
                    s=point_size,
                    color=color,
                    alpha=alpha,
                    edgecolors="black",
                    linewidths=0.15,
                    rasterized=True,
                )

        ticks = eigenvalue_index_ticks(max_num_params, max_ticks=11)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlim(0.5, max_num_params + 0.5)
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.axhline(0.0, linestyle="-", linewidth=0.8, color="black", alpha=0.45)

        if threshold > 0.0:
            ax.axhline(threshold, linestyle="-", linewidth=1.2, color="red", alpha=0.75)
            ax.axhline(-threshold, linestyle="-", linewidth=1.2, color="red", alpha=0.75)

        ax.set_ylim(-1.5 * max_abs, 1.5 * max_abs)
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel(ylabel)
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

    hs_random_points_result_path = os.path.join(
        hs_results_dir,
        f"hs_random_points_{keep_key}.npz",
    )

    if os.path.exists(hs_random_points_result_path):
        hs_random_points_results = load_npz_result(hs_random_points_result_path)
        hs_layer_list = [
            int(L)
            for L in np.asarray(hs_random_points_results["layers"], dtype=NP_INT_DTYPE)
        ]
        hs_rank_reduced_0123_by_layer = _load_layer_arrays_from_npz(
            hs_random_points_results,
            hs_layer_list,
            "rank",
            dtype=NP_INT_DTYPE,
        )
        hs_eigs_reduced_0123_by_layer = _load_layer_arrays_from_npz(
            hs_random_points_results,
            hs_layer_list,
            "eigs_desc",
            dtype=NP_REAL_DTYPE,
        )
        hs_rho_rank_reduced_0123_by_layer = _load_layer_arrays_from_npz(
            hs_random_points_results,
            hs_layer_list,
            "rho_rank",
            dtype=NP_INT_DTYPE,
        )
        hs_eigsum_reduced_0123_by_layer = _load_layer_arrays_from_npz(
            hs_random_points_results,
            hs_layer_list,
            "trace",
            dtype=NP_REAL_DTYPE,
        )
        hs_abs_entry_sum_reduced_0123_by_layer = _load_layer_arrays_from_npz(
            hs_random_points_results,
            hs_layer_list,
            "abs_entry_sum",
            dtype=NP_REAL_DTYPE,
        )
    else:
        hs_layer_list = []
        hs_rank_reduced_0123_by_layer = {}
        hs_eigs_reduced_0123_by_layer = {}
        hs_rho_rank_reduced_0123_by_layer = {}
        hs_eigsum_reduced_0123_by_layer = {}
        hs_abs_entry_sum_reduced_0123_by_layer = {}

    ortk_random_points_result_path = os.path.join(
        ortk_results_dir,
        "ortk_random_points.npz",
    )

    if os.path.exists(ortk_random_points_result_path):
        ortk_random_points_results = load_npz_result(ortk_random_points_result_path)
        ortk_layer_list = [
            int(L)
            for L in np.asarray(
                ortk_random_points_results["layers"],
                dtype=NP_INT_DTYPE,
            )
        ]
        ortk_rank_by_layer = _load_layer_arrays_from_npz(
            ortk_random_points_results,
            ortk_layer_list,
            "rank",
            dtype=NP_INT_DTYPE,
        )
        ortk_effective_rank_by_layer = _load_layer_arrays_from_npz(
            ortk_random_points_results,
            ortk_layer_list,
            "effective_rank",
            dtype=NP_REAL_DTYPE,
        )
        ortk_eigs_by_layer = _load_layer_arrays_from_npz(
            ortk_random_points_results,
            ortk_layer_list,
            "eigs_desc",
            dtype=NP_REAL_DTYPE,
        )
        ortk_trace_by_layer = _load_layer_arrays_from_npz(
            ortk_random_points_results,
            ortk_layer_list,
            "trace",
            dtype=NP_REAL_DTYPE,
        )
        ortk_rank_threshold = float(
            np.asarray(ortk_random_points_results["ortk_rank_threshold"]).item()
        )
    else:
        ortk_layer_list = []
        ortk_rank_by_layer = {}
        ortk_effective_rank_by_layer = {}
        ortk_eigs_by_layer = {}
        ortk_trace_by_layer = {}
        ortk_rank_threshold = float(QFIM_EFFECTIVE_RANK_THRESHOLD)

    hessian_random_points_result_path = os.path.join(
        hessian_results_dir,
        "hessian_random_points.npz",
    )

    if os.path.exists(hessian_random_points_result_path):
        hessian_random_points_results = load_npz_result(
            hessian_random_points_result_path
        )
        hessian_layer_list = [
            int(L)
            for L in np.asarray(
                hessian_random_points_results["layers"],
                dtype=NP_INT_DTYPE,
            )
        ]
        hessian_rank_by_layer = _load_layer_arrays_from_npz(
            hessian_random_points_results,
            hessian_layer_list,
            "rank",
            dtype=NP_INT_DTYPE,
        )
        hessian_eigs_by_layer = _load_layer_arrays_from_npz(
            hessian_random_points_results,
            hessian_layer_list,
            "eigs_desc",
            dtype=NP_REAL_DTYPE,
        )
        hessian_trace_by_layer = _load_layer_arrays_from_npz(
            hessian_random_points_results,
            hessian_layer_list,
            "trace",
            dtype=NP_REAL_DTYPE,
        )
        hessian_abs_eigsum_by_layer = _load_layer_arrays_from_npz(
            hessian_random_points_results,
            hessian_layer_list,
            "abs_eigsum",
            dtype=NP_REAL_DTYPE,
        )
        hessian_rank_threshold = float(
            np.asarray(
                hessian_random_points_results["hessian_effective_rank_threshold"]
            ).item()
        )
    else:
        hessian_layer_list = []
        hessian_rank_by_layer = {}
        hessian_eigs_by_layer = {}
        hessian_trace_by_layer = {}
        hessian_abs_eigsum_by_layer = {}
        hessian_rank_threshold = float(QFIM_EFFECTIVE_RANK_THRESHOLD)

    for L in qfim_layer_list:
        save_qfim_eigs_by_index(
            qfim_eigs_reduced_0123_by_layer[L],
            title=rf"QFIM eigenvalues at {NUM_QFIM_SAMPLES} random points (L={L})",
            outpath=os.path.join(qfim_eigs_dir_red4, f"L{L}_reduced_0123.pdf"),
        )
        save_eigenvalue_histograms_by_trial(
            qfim_eigs_reduced_0123_by_layer[L],
            outdir=os.path.join(
                qfim_eigs_dir_red4,
                "histograms",
                "random_points",
                f"L{L}",
            ),
            matrix_tag="dpqc_qfim",
            matrix_label="QFIM",
            num_layers=L,
            context_tag="random",
            context_label="random point",
            condition_tag="reduced0123",
            condition_label="reduced keep=(0,1,2,3)",
            color=METRIC_COLORS["qfim"],
        )

    for L in hs_layer_list:
        save_qfim_eigs_by_index(
            hs_eigs_reduced_0123_by_layer[L],
            title=(
                rf"HS tangent Gram eigenvalues at {NUM_QFIM_SAMPLES} random "
                rf"points (L={L})"
            ),
            outpath=os.path.join(hs_eigs_dir_red4, f"L{L}_reduced_0123.pdf"),
            ylabel="HS tangent Gram eigenvalue",
        )
        save_eigenvalue_histograms_by_trial(
            hs_eigs_reduced_0123_by_layer[L],
            outdir=os.path.join(
                hs_eigs_dir_red4,
                "histograms",
                "random_points",
                f"L{L}",
            ),
            matrix_tag="dpqc_hs_gram",
            matrix_label="HS tangent Gram",
            num_layers=L,
            context_tag="random",
            context_label="random point",
            condition_tag="reduced0123",
            condition_label="reduced keep=(0,1,2,3)",
            color=METRIC_COLORS["hs"],
        )

    for L in ortk_layer_list:
        save_qfim_eigs_by_index(
            ortk_eigs_by_layer[L],
            title=(
                rf"Observable-Relevant Tangent Kernel eigenvalues at "
                rf"{NUM_QFIM_SAMPLES} random points (L={L})"
            ),
            outpath=os.path.join(ortk_eigs_dir, f"L{L}.pdf"),
            ylabel="ORTK eigenvalue",
            rank_threshold=ortk_rank_threshold,
        )

    for L in hessian_layer_list:
        save_signed_eigs_by_index(
            hessian_eigs_by_layer[L],
            title=rf"Energy Hessian eigenvalues at {NUM_QFIM_SAMPLES} random points (L={L})",
            outpath=os.path.join(hessian_eigs_dir, f"L{L}.pdf"),
            threshold=hessian_rank_threshold,
            ylabel="Energy Hessian eigenvalue",
        )
        save_eigenvalue_histograms_by_trial(
            hessian_eigs_by_layer[L],
            outdir=os.path.join(
                hessian_eigs_dir,
                "histograms",
                "random_points",
                f"L{L}",
            ),
            matrix_tag="dpqc_energy_hessian",
            matrix_label="Energy Hessian",
            num_layers=L,
            context_tag="random",
            context_label="random point",
            color=METRIC_COLORS["hessian"],
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

    if hs_layer_list:
        save_qfim_eigs_by_index_colored_by_layer(
            hs_eigs_reduced_0123_by_layer,
            hs_layer_list,
            title=rf"HS tangent Gram eigenvalues at {NUM_QFIM_SAMPLES} random points",
            outpath=os.path.join(
                hs_eigs_dir,
                "hs_eigs_by_index_layers_reduced_keep_0123.pdf",
            ),
            cmap=cmap,
            ylabel="HS tangent Gram eigenvalue",
        )

    if ortk_layer_list:
        save_qfim_eigs_by_index_colored_by_layer(
            ortk_eigs_by_layer,
            ortk_layer_list,
            title=(
                rf"Observable-Relevant Tangent Kernel eigenvalues at "
                rf"{NUM_QFIM_SAMPLES} random points"
            ),
            outpath=os.path.join(
                ortk_eigs_dir,
                "ortk_eigs_by_index_layers_random_points.pdf",
            ),
            cmap=cmap,
            ylabel="ORTK eigenvalue",
        )

    if hessian_layer_list:
        save_signed_eigs_by_index_colored_by_layer(
            hessian_eigs_by_layer,
            hessian_layer_list,
            title=rf"Energy Hessian eigenvalues at {NUM_QFIM_SAMPLES} random points",
            outpath=os.path.join(
                hessian_eigs_dir,
                "hessian_eigs_by_index_layers_random_points.pdf",
            ),
            threshold=hessian_rank_threshold,
            cmap=cmap,
            ylabel="Energy Hessian eigenvalue",
        )


    # ============================================================
    # Mean/minimum/maximum QFIM rank + density-rank upper bound by layer
    #   reduced keep=(0,1,2,3) only
    # ============================================================
    def _qfim_rank_mean_min_max_sem_upper_bound_xy(
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
                np.asarray([], dtype=NP_REAL_DTYPE),
                [],
            )

        valid_layers = [L for L, _ in valid_items]
        x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)

        min_ranks = np.asarray(
            [np.min(ranks_arr) for _, ranks_arr in valid_items],
            dtype=NP_REAL_DTYPE,
        )

        max_ranks = np.asarray(
            [np.max(ranks_arr) for _, ranks_arr in valid_items],
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
            parameter_bound = num_params_per_layer * int(L)
            upper_bounds.append(min(parameter_bound, density_manifold_bound))

        upper_bounds = np.asarray(upper_bounds, dtype=NP_REAL_DTYPE)

        return x, min_ranks, max_ranks, mean_ranks, sem_ranks, upper_bounds, valid_layers


    def plot_single_qfim_rank_mean_min_max_sem_by_layer(
        rank_by_layer: dict,
        rho_rank_by_layer: dict,
        layers,
        *,
        color_min,
        color_max,
        color_mean,
        label=None,
        title,
        outpath,
        ylabel: str = "QFIM rank",
        rank_label: str = "effective rank",
        marker_min: str = "o",
        marker_max: str = "^",
        marker_mean: str = "s",
        lw: float = 1.4,
        integer_y_axis: bool = True,
    ):
        (
            x,
            min_ranks,
            max_ranks,
            mean_ranks,
            sem_ranks,
            upper_bounds,
            valid_layers,
        ) = _qfim_rank_mean_min_max_sem_upper_bound_xy(
            rank_by_layer,
            rho_rank_by_layer,
            layers,
        )

        if x.size == 0:
            return

        label_suffix = "" if label in (None, "") else rf" ({label})"

        fig, ax = new_fig_ax(outside_legend=False)

        ax.plot(
            x,
            min_ranks,
            marker=marker_min,
            linestyle=STATISTIC_LINESTYLES["min"],
            linewidth=lw,
            markersize=6.0,
            color=color_min,
            label=rf"Minimum {rank_label}{label_suffix}",
        )

        ax.plot(
            x,
            max_ranks,
            marker=marker_max,
            linestyle=STATISTIC_LINESTYLES["max"],
            linewidth=lw,
            markersize=6.0,
            color=color_max,
            label=rf"Maximum {rank_label}{label_suffix}",
        )

        ax.errorbar(
            x,
            mean_ranks,
            yerr=sem_ranks,
            marker=marker_mean,
            linestyle=STATISTIC_LINESTYLES["mean"],
            linewidth=lw,
            markersize=5.0,
            capsize=4.0,
            elinewidth=1.0,
            color=color_mean,
            label=rf"Mean {rank_label} $\pm$ SEM{label_suffix}",
        )

        ax.plot(
            x,
            upper_bounds,
            marker=None,
            linestyle="--",
            linewidth=lw,
            color="black",
            label=r"Upper bound $\min(14L, 32r_{\max}-r_{\max}^2-1)$",
        )

        ax.set_xlabel("Number of Layers")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(L) for L in valid_layers])
        if integer_y_axis:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best", frameon=True, framealpha=0.9)

        save_fig(fig, ax, outpath, outside_legend=False)


    def plot_rank_mean_min_max_sem_by_layer(
        rank_by_layer: dict,
        layers,
        *,
        color_min,
        color_max,
        color_mean,
        title,
        outpath,
        ylabel: str,
        rank_label: str,
        marker_min: str = "o",
        marker_max: str = "^",
        marker_mean: str = "s",
        lw: float = 1.4,
    ):
        valid_items = []

        for L in layers:
            ranks_L = rank_by_layer.get(int(L))
            if ranks_L is None:
                continue

            ranks_arr = np.asarray(ranks_L, dtype=NP_REAL_DTYPE).reshape(-1)
            ranks_arr = ranks_arr[np.isfinite(ranks_arr)]

            if ranks_arr.size == 0:
                continue

            valid_items.append((int(L), ranks_arr))

        if not valid_items:
            return

        valid_layers = [L for L, _ in valid_items]
        x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)
        min_ranks = np.asarray(
            [np.min(ranks_arr) for _, ranks_arr in valid_items],
            dtype=NP_REAL_DTYPE,
        )
        max_ranks = np.asarray(
            [np.max(ranks_arr) for _, ranks_arr in valid_items],
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

        fig, ax = new_fig_ax(outside_legend=False)
        ax.plot(
            x,
            min_ranks,
            marker=marker_min,
            linestyle=STATISTIC_LINESTYLES["min"],
            linewidth=lw,
            markersize=6.0,
            color=color_min,
            label=rf"Minimum {rank_label}",
        )
        ax.plot(
            x,
            max_ranks,
            marker=marker_max,
            linestyle=STATISTIC_LINESTYLES["max"],
            linewidth=lw,
            markersize=6.0,
            color=color_max,
            label=rf"Maximum {rank_label}",
        )
        ax.errorbar(
            x,
            mean_ranks,
            yerr=sem_ranks,
            marker=marker_mean,
            linestyle=STATISTIC_LINESTYLES["mean"],
            linewidth=lw,
            markersize=5.0,
            capsize=4.0,
            elinewidth=1.0,
            color=color_mean,
            label=rf"Mean {rank_label} $\pm$ SEM",
        )

        ax.set_xlabel("Number of Layers")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(L) for L in valid_layers])
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best", frameon=True, framealpha=0.9)

        save_fig(fig, ax, outpath, outside_legend=False)


    plot_single_qfim_rank_mean_min_max_sem_by_layer(
        qfim_rank_reduced_0123_by_layer,
        qfim_rho_rank_reduced_0123_by_layer,
        qfim_layer_list,
        color_min="C0",
        color_max="C2",
        color_mean="C1",
        label=None,
        title=rf"QFIM effective rank mean/minimum/maximum and upper bound at {NUM_QFIM_SAMPLES} random points",
        outpath=os.path.join(
            qfim_rank_random_dir,
            "qfim_rank_mean_min_upper_bound_random_points_reduced_0123.pdf",
        ),
        marker_min="o",
        marker_max="^",
        marker_mean="s",
        lw=1.4,
    )

    if hs_layer_list:
        plot_single_qfim_rank_mean_min_max_sem_by_layer(
            hs_rank_reduced_0123_by_layer,
            hs_rho_rank_reduced_0123_by_layer,
            hs_layer_list,
            color_min="C3",
            color_max="C5",
            color_mean="C4",
            label=None,
            title=(
                rf"HS tangent Gram effective rank mean/minimum/maximum and "
                rf"upper bound at {NUM_QFIM_SAMPLES} random points"
            ),
            outpath=os.path.join(
                hs_rank_random_dir,
                "hs_rank_mean_min_upper_bound_random_points_reduced_0123.pdf",
            ),
            ylabel="HS tangent Gram rank",
            rank_label="HS effective rank",
            marker_min="o",
            marker_max="^",
            marker_mean="s",
            lw=1.4,
        )

    if ortk_layer_list:
        plot_rank_mean_min_max_sem_by_layer(
            ortk_rank_by_layer,
            ortk_layer_list,
            color_min="C9",
            color_max="C2",
            color_mean="C0",
            title=(
                rf"Observable-Relevant Tangent Kernel rank "
                rf"mean/minimum/maximum at {NUM_QFIM_SAMPLES} random points"
            ),
            outpath=os.path.join(
                ortk_rank_random_dir,
                "ortk_rank_mean_min_max_random_points.pdf",
            ),
            ylabel="ORTK rank",
            rank_label="ORTK rank",
            marker_min="o",
            marker_max="^",
            marker_mean="s",
            lw=1.4,
            integer_y_axis=True,
        )

        plot_rank_mean_min_max_sem_by_layer(
            ortk_effective_rank_by_layer,
            ortk_layer_list,
            color_min="C9",
            color_max="C2",
            color_mean="C0",
            title=(
                rf"Observable-Relevant Tangent Kernel participation effective "
                rf"rank mean/minimum/maximum at {NUM_QFIM_SAMPLES} random points"
            ),
            outpath=os.path.join(
                ortk_effective_rank_random_dir,
                "ortk_effective_rank_mean_min_max_random_points.pdf",
            ),
            ylabel="ORTK participation effective rank",
            rank_label="ORTK effective rank",
            marker_min="o",
            marker_max="^",
            marker_mean="s",
            lw=1.4,
            integer_y_axis=False,
        )

    if hessian_layer_list:
        plot_rank_mean_min_max_sem_by_layer(
            hessian_rank_by_layer,
            hessian_layer_list,
            color_min="C6",
            color_max="C8",
            color_mean="C7",
            title=(
                rf"Energy Hessian rank mean/minimum/maximum at "
                rf"{NUM_QFIM_SAMPLES} random points"
            ),
            outpath=os.path.join(
                hessian_rank_random_dir,
                "hessian_rank_mean_min_max_random_points.pdf",
            ),
            ylabel="Energy Hessian rank",
            rank_label="Hessian rank",
            marker_min="o",
            marker_max="^",
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
        *,
        use_absolute_values: bool = False,
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

                if eigs_L.ndim < 2:
                    raise ValueError(
                        "Each eigenvalue array must have a sample axis and "
                        "an eigenvalue axis."
                    )

                # Optimization-path data have (run, iteration, eigenvalue).
                # Both run and iteration are valid samples for a layer-wise
                # summary, so collapse all leading axes here.
                eigs_L = eigs_L.reshape(-1, eigs_L.shape[-1])
                if use_absolute_values:
                    eigs_L = np.abs(eigs_L)

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
        ylabel: str = "Mean QFIM eigenvalue count",
        eigenvalue_symbol: str = r"\lambda_i",
        cmap=None,
        use_absolute_values: bool = False,
    ):
        thresholds = tuple(float(thr) for thr in thresholds)

        if not thresholds:
            return

        eigcount_by_threshold = qfim_random_eigcount_by_threshold_by_layer(
            eigs_by_layer,
            layers,
            thresholds,
            use_absolute_values=use_absolute_values,
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
        num_thresholds = len(thresholds)

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
            color = cmap(threshold_idx / max(num_thresholds - 1, 1))
            label = rf"${eigenvalue_symbol} \geq {_qfim_threshold_tex_for_label(threshold)}$"

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
        ax.set_ylabel(ylabel)
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

    if hs_layer_list:
        plot_qfim_random_eigcount_threshold_overlay(
            hs_eigs_reduced_0123_by_layer,
            hs_layer_list,
            QFIM_PATH_EIGCOUNT_THRESHOLDS,
            title=(
                rf"HS tangent Gram eigenvalue count at random points by "
                rf"threshold ({keep_label})"
            ),
            outpath=os.path.join(
                hs_eigcount_random_dir,
                f"hs_eigcount_threshold_overlay_random_points_{keep_key}.pdf",
            ),
            ylabel="Mean HS eigenvalue count",
            eigenvalue_symbol=r"\mu_i",
            cmap=cmap,
        )

    if ortk_eigs_by_layer:
        plot_qfim_random_eigcount_threshold_overlay(
            ortk_eigs_by_layer,
            qfim_layer_list,
            QFIM_PATH_EIGCOUNT_THRESHOLDS,
            title="ORTK eigenvalue count at random points by threshold",
            outpath=os.path.join(ortk_eigs_dir, "ortk_eigcount_threshold_overlay_random_points.pdf"),
            ylabel="Mean ORTK eigenvalue count",
            eigenvalue_symbol=r"\kappa_i",
            cmap=cmap,
        )

    if hessian_eigs_by_layer:
        plot_qfim_random_eigcount_threshold_overlay(
            hessian_eigs_by_layer,
            qfim_layer_list,
            QFIM_PATH_EIGCOUNT_THRESHOLDS,
            title="Absolute Hessian eigenvalue count at random points by threshold",
            outpath=os.path.join(hessian_eigs_dir, "hessian_abs_eigcount_threshold_overlay_random_points.pdf"),
            ylabel="Mean absolute Hessian eigenvalue count",
            eigenvalue_symbol=r"|\eta_i|",
            cmap=cmap,
            use_absolute_values=True,
        )


    # ============================================================
    # Maximum QFIM trace + mean ﾂｱ SEM by layer
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
        ylabel: str = "QFIM trace",
        max_label: str = "Maximum QFIM trace",
        mean_label: str = r"Mean QFIM trace $\pm$ SEM",
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
            linestyle=STATISTIC_LINESTYLES["max"],
            linewidth=lw,
            markersize=6.0,
            color=color_max,
            label=max_label,
        )

        ax.errorbar(
            x,
            mean_values,
            yerr=sem_values,
            marker=marker_mean,
            linestyle=STATISTIC_LINESTYLES["mean"],
            linewidth=lw,
            markersize=5.0,
            capsize=4.0,
            elinewidth=1.0,
            color=color_mean,
            label=mean_label,
        )

        ax.set_xlabel("Number of Layers")
        ax.set_ylabel(ylabel)
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
            qfim_trace_random_dir,
            "qfim_trace_max_mean_sem_random_points_reduced_0123.pdf",
        ),
        color_max="C0",
        color_mean="C1",
        marker_max="o",
        marker_mean="s",
        lw=1.4,
        log_scale=False,
    )

    if hs_layer_list:
        plot_qfim_trace_max_mean_sem_by_layer(
            hs_eigsum_reduced_0123_by_layer,
            hs_layer_list,
            title=(
                rf"HS tangent Gram trace maximum and mean $\pm$ SEM at "
                rf"{NUM_QFIM_SAMPLES} random points"
            ),
            outpath=os.path.join(
                hs_trace_random_dir,
                "hs_trace_max_mean_sem_random_points_reduced_0123.pdf",
            ),
            ylabel="HS tangent Gram trace",
            max_label="Maximum HS trace",
            mean_label=r"Mean HS trace $\pm$ SEM",
            color_max="C3",
            color_mean="C4",
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
        ylabel: str = "Mean QFIM rank",
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
        num_layers = len(valid_layers)

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

            color = cmap(layer_idx / max(num_layers - 1, 1))

            ax.errorbar(
                x[finite_mask],
                means[finite_mask],
                yerr=sems[finite_mask],
                marker="o",
                linestyle=STATISTIC_LINESTYLES["mean"],
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

        # The callers historically saved mean and minimum histories only.
        # Always add the matching maximum history so every summary is present.
        max_outpath = outpath.replace(
            f"{os.sep}mean{os.sep}", f"{os.sep}max{os.sep}"
        ).replace("_mean_", "_max_")
        os.makedirs(os.path.dirname(max_outpath), exist_ok=True)
        plot_qfim_rank_history_extreme_by_layer(
            rank_history_by_layer,
            layers,
            sample_iters,
            statistic="max",
            title=title.replace("Mean ", "Maximum ").replace("mean ", "maximum "),
            outpath=max_outpath,
            ylabel=ylabel.replace("Mean ", "Maximum "),
            cmap=cmap,
            integer_y_axis="rank" in ylabel.lower(),
        )


    def plot_qfim_rank_history_min_by_layer(
        rank_history_by_layer: dict,
        layers,
        sample_iters,
        *,
        title: str,
        outpath: str,
        ylabel: str = "Minimum QFIM rank",
        cmap=None,
        integer_y_axis: bool = True,
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
        num_layers = len(valid_layers)

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

            color = cmap(layer_idx / max(num_layers - 1, 1))

            ax.plot(
                x[finite_mask],
                min_ranks[finite_mask],
                marker="o",
                linestyle=STATISTIC_LINESTYLES["min"],
                linewidth=1.2,
                markersize=4.5,
                color=color,
                label=f"L={L}",
            )

        ax.set_xlabel("Iterations")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(t)) for t in x], rotation=45, ha="right")
        if integer_y_axis:
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


    def plot_qfim_rank_history_extreme_by_layer(
        rank_history_by_layer: dict,
        layers,
        sample_iters,
        *,
        statistic: str,
        title: str,
        outpath: str,
        ylabel: str,
        cmap=None,
        integer_y_axis: bool = False,
    ):
        if statistic not in {"min", "max"}:
            raise ValueError("statistic must be 'min' or 'max'.")
        valid_layers = [int(L) for L in layers if rank_history_by_layer.get(int(L)) is not None]
        if not valid_layers:
            return
        x = np.asarray(sample_iters, dtype=NP_REAL_DTYPE)
        cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap
        fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.22)
        for layer_idx, L in enumerate(valid_layers):
            values = np.asarray(rank_history_by_layer[L], dtype=NP_REAL_DTYPE)
            if values.ndim != 2:
                raise ValueError("Each history array must be 2D (num_runs, num_sample_iters).")
            if values.shape[1] != x.size and values.shape[0] == x.size:
                values = values.T
            if values.shape[1] != x.size:
                raise ValueError(f"Shape mismatch for L={L}: values.shape={values.shape}, len(sample_iters)={x.size}.")
            valid = np.isfinite(values)
            counts = np.sum(valid, axis=0)
            fill = np.inf if statistic == "min" else -np.inf
            reduced = getattr(np, statistic)(np.where(valid, values, fill), axis=0)
            reduced = np.where(counts > 0, reduced, np.nan)
            mask = np.isfinite(reduced)
            if np.any(mask):
                ax.plot(x[mask], reduced[mask], marker="o",
                        linestyle=STATISTIC_LINESTYLES[statistic], linewidth=1.2,
                        markersize=4.5, color=cmap(layer_idx / max(len(valid_layers) - 1, 1)),
                        label=f"L={L}")
        ax.set(xlabel="Iterations", ylabel=ylabel, title=title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(t)) for t in x], rotation=45, ha="right")
        if integer_y_axis:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        save_fig(fig, ax, outpath, outside_legend=True, legend_space_frac=0.22)


    def plot_qfim_trace_history_mean_by_layer(
        trace_history_by_layer: dict,
        layers,
        sample_iters,
        *,
        title: str,
        outpath: str,
        ylabel: str = "Mean QFIM trace",
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
        num_layers = len(valid_layers)

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

            color = cmap(layer_idx / max(num_layers - 1, 1))

            ax.errorbar(
                x[finite_mask],
                means[finite_mask],
                yerr=sems[finite_mask],
                marker="o",
                linestyle=STATISTIC_LINESTYLES["mean"],
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

        for statistic, adjective in (("min", "Minimum"), ("max", "Maximum")):
            sibling = outpath.replace("_mean_", f"_{statistic}_")
            plot_qfim_rank_history_extreme_by_layer(
                trace_history_by_layer,
                layers,
                sample_iters,
                statistic=statistic,
                title=title.replace("Mean ", f"{adjective} ").replace("mean ", f"{adjective.lower()} "),
                outpath=sibling,
                ylabel=ylabel.replace("Mean ", f"{adjective} "),
                cmap=cmap,
                integer_y_axis=False,
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
            qfim_trace_optimization_path_dir,
            f"qfim_trace_mean_history_optimization_path_{keep_key}.pdf",
        ),
        cmap=cmap,
        log_scale=False,
    )

    hs_rank_history_result_path = os.path.join(
        hs_results_dir,
        f"hs_rank_history_optimization_path_{keep_key}.npz",
    )
    hs_eigs_history_result_path = os.path.join(
        hs_results_dir,
        f"hs_eigs_history_optimization_path_{keep_key}.npz",
    )
    hs_trace_history_result_path = os.path.join(
        hs_results_dir,
        f"hs_trace_history_optimization_path_{keep_key}.npz",
    )

    if (
        os.path.exists(hs_rank_history_result_path)
        and os.path.exists(hs_eigs_history_result_path)
    ):
        hs_rank_history_results = load_npz_result(hs_rank_history_result_path)
        hs_path_layer_list = [
            int(L)
            for L in np.asarray(hs_rank_history_results["layers"], dtype=NP_INT_DTYPE)
        ]
        hs_sample_iters = np.asarray(
            hs_rank_history_results["sample_iters"],
            dtype=NP_INT_DTYPE,
        )
        hs_rank_history_by_layer = _load_layer_arrays_from_npz(
            hs_rank_history_results,
            hs_path_layer_list,
            suffix=None,
            dtype=NP_REAL_DTYPE,
        )

        plot_qfim_rank_history_mean_by_layer(
            hs_rank_history_by_layer,
            hs_path_layer_list,
            hs_sample_iters,
            title=rf"Mean HS tangent Gram effective rank along optimization path ({keep_label})",
            outpath=os.path.join(
                hs_rank_optimization_path_mean_dir,
                f"hs_rank_mean_history_optimization_path_{keep_key}.pdf",
            ),
            ylabel="Mean HS tangent Gram rank",
            cmap=cmap,
        )

        plot_qfim_rank_history_min_by_layer(
            hs_rank_history_by_layer,
            hs_path_layer_list,
            hs_sample_iters,
            title=rf"Minimum HS tangent Gram effective rank along optimization path ({keep_label})",
            outpath=os.path.join(
                hs_rank_optimization_path_min_dir,
                f"hs_rank_min_history_optimization_path_{keep_key}.pdf",
            ),
            ylabel="Minimum HS tangent Gram rank",
            cmap=cmap,
        )

        hs_eigs_history_results = load_npz_result(hs_eigs_history_result_path)
        hs_eigs_history_optimization_path_by_layer = _load_layer_arrays_from_npz(
            hs_eigs_history_results,
            hs_path_layer_list,
            suffix=None,
            dtype=NP_REAL_DTYPE,
        )

        if os.path.exists(hs_trace_history_result_path):
            hs_trace_history_results = load_npz_result(hs_trace_history_result_path)
            hs_trace_history_layer_list = [
                int(L)
                for L in np.asarray(
                    hs_trace_history_results["layers"],
                    dtype=NP_INT_DTYPE,
                )
            ]
            hs_trace_history_optimization_path_by_layer = _load_layer_arrays_from_npz(
                hs_trace_history_results,
                hs_trace_history_layer_list,
                suffix=None,
                dtype=NP_REAL_DTYPE,
            )
        else:
            hs_trace_history_layer_list = list(hs_path_layer_list)
            hs_trace_history_optimization_path_by_layer = {
                int(L): np.sum(np.asarray(eigs, dtype=NP_REAL_DTYPE), axis=2)
                for L, eigs in hs_eigs_history_optimization_path_by_layer.items()
            }

        plot_qfim_trace_history_mean_by_layer(
            hs_trace_history_optimization_path_by_layer,
            hs_trace_history_layer_list,
            hs_sample_iters,
            title=rf"Mean HS tangent Gram trace along optimization path ({keep_label})",
            outpath=os.path.join(
                hs_trace_optimization_path_dir,
                f"hs_trace_mean_history_optimization_path_{keep_key}.pdf",
            ),
            ylabel="Mean HS tangent Gram trace",
            cmap=cmap,
            log_scale=False,
        )
    else:
        hs_path_layer_list = []
        hs_sample_iters = np.asarray([], dtype=NP_INT_DTYPE)
        hs_rank_history_by_layer = {}
        hs_eigs_history_optimization_path_by_layer = {}
        hs_trace_history_layer_list = []
        hs_trace_history_optimization_path_by_layer = {}

    ortk_rank_history_result_path = os.path.join(
        ortk_results_dir,
        "ortk_rank_history_optimization_path.npz",
    )
    ortk_effective_rank_history_result_path = os.path.join(
        ortk_results_dir,
        "ortk_effective_rank_history_optimization_path.npz",
    )
    ortk_eigs_history_result_path = os.path.join(
        ortk_results_dir,
        "ortk_eigs_history_optimization_path.npz",
    )
    ortk_trace_history_result_path = os.path.join(
        ortk_results_dir,
        "ortk_trace_history_optimization_path.npz",
    )

    if (
        os.path.exists(ortk_rank_history_result_path)
        and os.path.exists(ortk_effective_rank_history_result_path)
        and os.path.exists(ortk_eigs_history_result_path)
    ):
        ortk_rank_history_results = load_npz_result(ortk_rank_history_result_path)
        ortk_path_layer_list = [
            int(L)
            for L in np.asarray(
                ortk_rank_history_results["layers"],
                dtype=NP_INT_DTYPE,
            )
        ]
        ortk_sample_iters = np.asarray(
            ortk_rank_history_results["sample_iters"],
            dtype=NP_INT_DTYPE,
        )
        ortk_rank_history_by_layer = _load_layer_arrays_from_npz(
            ortk_rank_history_results,
            ortk_path_layer_list,
            suffix=None,
            dtype=NP_REAL_DTYPE,
        )

        plot_qfim_rank_history_mean_by_layer(
            ortk_rank_history_by_layer,
            ortk_path_layer_list,
            ortk_sample_iters,
            title="Mean Observable-Relevant Tangent Kernel rank along optimization path",
            outpath=os.path.join(
                ortk_rank_optimization_path_mean_dir,
                "ortk_rank_mean_history_optimization_path.pdf",
            ),
            ylabel="Mean ORTK rank",
            cmap=cmap,
        )

        plot_qfim_rank_history_min_by_layer(
            ortk_rank_history_by_layer,
            ortk_path_layer_list,
            ortk_sample_iters,
            title="Minimum Observable-Relevant Tangent Kernel rank along optimization path",
            outpath=os.path.join(
                ortk_rank_optimization_path_min_dir,
                "ortk_rank_min_history_optimization_path.pdf",
            ),
            ylabel="Minimum ORTK rank",
            cmap=cmap,
            integer_y_axis=True,
        )

        ortk_effective_rank_history_results = load_npz_result(
            ortk_effective_rank_history_result_path
        )
        ortk_effective_rank_history_by_layer = _load_layer_arrays_from_npz(
            ortk_effective_rank_history_results,
            ortk_path_layer_list,
            suffix=None,
            dtype=NP_REAL_DTYPE,
        )

        plot_qfim_rank_history_mean_by_layer(
            ortk_effective_rank_history_by_layer,
            ortk_path_layer_list,
            ortk_sample_iters,
            title=(
                "Mean Observable-Relevant Tangent Kernel participation "
                "effective rank along optimization path"
            ),
            outpath=os.path.join(
                ortk_effective_rank_optimization_path_mean_dir,
                "ortk_effective_rank_mean_history_optimization_path.pdf",
            ),
            ylabel="Mean ORTK participation effective rank",
            cmap=cmap,
        )

        plot_qfim_rank_history_min_by_layer(
            ortk_effective_rank_history_by_layer,
            ortk_path_layer_list,
            ortk_sample_iters,
            title=(
                "Minimum Observable-Relevant Tangent Kernel participation "
                "effective rank along optimization path"
            ),
            outpath=os.path.join(
                ortk_effective_rank_optimization_path_min_dir,
                "ortk_effective_rank_min_history_optimization_path.pdf",
            ),
            ylabel="Minimum ORTK participation effective rank",
            cmap=cmap,
            integer_y_axis=False,
        )

        ortk_eigs_history_results = load_npz_result(ortk_eigs_history_result_path)
        ortk_eigs_history_optimization_path_by_layer = _load_layer_arrays_from_npz(
            ortk_eigs_history_results,
            ortk_path_layer_list,
            suffix=None,
            dtype=NP_REAL_DTYPE,
        )

        if os.path.exists(ortk_trace_history_result_path):
            ortk_trace_history_results = load_npz_result(
                ortk_trace_history_result_path
            )
            ortk_trace_history_layer_list = [
                int(L)
                for L in np.asarray(
                    ortk_trace_history_results["layers"],
                    dtype=NP_INT_DTYPE,
                )
            ]
            ortk_trace_history_optimization_path_by_layer = _load_layer_arrays_from_npz(
                ortk_trace_history_results,
                ortk_trace_history_layer_list,
                suffix=None,
                dtype=NP_REAL_DTYPE,
            )
        else:
            ortk_trace_history_layer_list = list(ortk_path_layer_list)
            ortk_trace_history_optimization_path_by_layer = {
                int(L): np.sum(np.asarray(eigs, dtype=NP_REAL_DTYPE), axis=2)
                for L, eigs in ortk_eigs_history_optimization_path_by_layer.items()
            }

        plot_qfim_trace_history_mean_by_layer(
            ortk_trace_history_optimization_path_by_layer,
            ortk_trace_history_layer_list,
            ortk_sample_iters,
            title="Mean Observable-Relevant Tangent Kernel trace along optimization path",
            outpath=os.path.join(
                ortk_trace_optimization_path_dir,
                "ortk_trace_mean_history_optimization_path.pdf",
            ),
            ylabel="Mean ORTK trace",
            cmap=cmap,
            log_scale=False,
        )
    else:
        ortk_path_layer_list = []
        ortk_sample_iters = np.asarray([], dtype=NP_INT_DTYPE)
        ortk_rank_history_by_layer = {}
        ortk_effective_rank_history_by_layer = {}
        ortk_eigs_history_optimization_path_by_layer = {}
        ortk_trace_history_layer_list = []
        ortk_trace_history_optimization_path_by_layer = {}

    hessian_rank_history_result_path = os.path.join(
        hessian_results_dir,
        "hessian_rank_history_optimization_path.npz",
    )
    hessian_eigs_history_result_path = os.path.join(
        hessian_results_dir,
        "hessian_eigs_history_optimization_path.npz",
    )
    hessian_trace_history_result_path = os.path.join(
        hessian_results_dir,
        "hessian_trace_history_optimization_path.npz",
    )
    hessian_abs_eigsum_history_result_path = os.path.join(
        hessian_results_dir,
        "hessian_abs_eigsum_history_optimization_path.npz",
    )

    if (
        os.path.exists(hessian_rank_history_result_path)
        and os.path.exists(hessian_eigs_history_result_path)
    ):
        hessian_rank_history_results = load_npz_result(
            hessian_rank_history_result_path
        )
        hessian_path_layer_list = [
            int(L)
            for L in np.asarray(
                hessian_rank_history_results["layers"],
                dtype=NP_INT_DTYPE,
            )
        ]
        hessian_sample_iters = np.asarray(
            hessian_rank_history_results["sample_iters"],
            dtype=NP_INT_DTYPE,
        )
        hessian_rank_history_by_layer = _load_layer_arrays_from_npz(
            hessian_rank_history_results,
            hessian_path_layer_list,
            suffix=None,
            dtype=NP_REAL_DTYPE,
        )
        hessian_path_rank_threshold = float(
            np.asarray(
                hessian_rank_history_results["hessian_effective_rank_threshold"]
            ).item()
        )

        plot_qfim_rank_history_mean_by_layer(
            hessian_rank_history_by_layer,
            hessian_path_layer_list,
            hessian_sample_iters,
            title="Mean energy Hessian rank along optimization path",
            outpath=os.path.join(
                hessian_rank_optimization_path_mean_dir,
                "hessian_rank_mean_history_optimization_path.pdf",
            ),
            ylabel="Mean energy Hessian rank",
            cmap=cmap,
        )

        plot_qfim_rank_history_min_by_layer(
            hessian_rank_history_by_layer,
            hessian_path_layer_list,
            hessian_sample_iters,
            title="Minimum energy Hessian rank along optimization path",
            outpath=os.path.join(
                hessian_rank_optimization_path_min_dir,
                "hessian_rank_min_history_optimization_path.pdf",
            ),
            ylabel="Minimum energy Hessian rank",
            cmap=cmap,
        )

        hessian_eigs_history_results = load_npz_result(
            hessian_eigs_history_result_path
        )
        hessian_eigs_history_optimization_path_by_layer = (
            _load_layer_arrays_from_npz(
                hessian_eigs_history_results,
                hessian_path_layer_list,
                suffix=None,
                dtype=NP_REAL_DTYPE,
            )
        )

        if os.path.exists(hessian_trace_history_result_path):
            hessian_trace_history_results = load_npz_result(
                hessian_trace_history_result_path
            )
            hessian_trace_history_layer_list = [
                int(L)
                for L in np.asarray(
                    hessian_trace_history_results["layers"],
                    dtype=NP_INT_DTYPE,
                )
            ]
            hessian_trace_history_optimization_path_by_layer = (
                _load_layer_arrays_from_npz(
                    hessian_trace_history_results,
                    hessian_trace_history_layer_list,
                    suffix=None,
                    dtype=NP_REAL_DTYPE,
                )
            )
        else:
            hessian_trace_history_layer_list = list(hessian_path_layer_list)
            hessian_trace_history_optimization_path_by_layer = {
                int(L): np.sum(np.asarray(eigs, dtype=NP_REAL_DTYPE), axis=2)
                for L, eigs in hessian_eigs_history_optimization_path_by_layer.items()
            }

        if os.path.exists(hessian_abs_eigsum_history_result_path):
            hessian_abs_eigsum_history_results = load_npz_result(
                hessian_abs_eigsum_history_result_path
            )
            hessian_abs_eigsum_history_layer_list = [
                int(L)
                for L in np.asarray(
                    hessian_abs_eigsum_history_results["layers"],
                    dtype=NP_INT_DTYPE,
                )
            ]
            hessian_abs_eigsum_history_optimization_path_by_layer = (
                _load_layer_arrays_from_npz(
                    hessian_abs_eigsum_history_results,
                    hessian_abs_eigsum_history_layer_list,
                    suffix=None,
                    dtype=NP_REAL_DTYPE,
                )
            )
        else:
            hessian_abs_eigsum_history_layer_list = list(hessian_path_layer_list)
            hessian_abs_eigsum_history_optimization_path_by_layer = {
                int(L): np.sum(
                    np.abs(np.asarray(eigs, dtype=NP_REAL_DTYPE)),
                    axis=2,
                )
                for L, eigs in hessian_eigs_history_optimization_path_by_layer.items()
            }
    else:
        hessian_path_layer_list = []
        hessian_sample_iters = np.asarray([], dtype=NP_INT_DTYPE)
        hessian_rank_history_by_layer = {}
        hessian_eigs_history_optimization_path_by_layer = {}
        hessian_path_rank_threshold = hessian_rank_threshold
        hessian_trace_history_layer_list = []
        hessian_trace_history_optimization_path_by_layer = {}
        hessian_abs_eigsum_history_layer_list = []
        hessian_abs_eigsum_history_optimization_path_by_layer = {}

    # Counts are meaningful for spectra, but not for scalar summaries such as
    # trace, energy, success rate, or participation effective rank.
    spectral_path_summaries = (
        (qfim_eigs_history_optimization_path_by_layer, qfim_eigs_dir,
         "QFIM", "qfim", r"\lambda_i", False),
        (hs_eigs_history_optimization_path_by_layer, hs_eigs_dir,
         "HS tangent Gram", "hs", r"\mu_i", False),
        (ortk_eigs_history_optimization_path_by_layer, ortk_eigs_dir,
         "ORTK", "ortk", r"\kappa_i", False),
        (hessian_eigs_history_optimization_path_by_layer, hessian_eigs_dir,
         "Absolute Hessian", "hessian_abs", r"|\eta_i|", True),
    )
    for eigs_by_layer, output_dir, label, tag, symbol, use_abs in spectral_path_summaries:
        if not eigs_by_layer:
            continue
        plot_qfim_random_eigcount_threshold_overlay(
            eigs_by_layer,
            sorted(eigs_by_layer),
            QFIM_PATH_EIGCOUNT_THRESHOLDS,
            title=f"{label} eigenvalue count along optimization path by threshold",
            outpath=os.path.join(
                output_dir,
                f"{tag}_eigcount_threshold_overlay_optimization_path_by_layer.pdf",
            ),
            ylabel=f"Mean {label} eigenvalue count",
            eigenvalue_symbol=symbol,
            cmap=cmap,
            use_absolute_values=use_abs,
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
        target_iterations=None,
        eps: float = QFIM_EIG_PLOT_EPS,
        matrix_tag: str = "dpqc_qfim",
        quantity_name: str = "QFIM",
        ylabel: str = "QFIM eigenvalue",
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
            desc=f"{quantity_name} eig distributions along optimization path",
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
                    f"iter{iteration_int:06d}_{keep_key}.pdf",
                )

                save_qfim_eigs_by_index(
                    eigs_L[:, time_idx, :],
                    title=(
                        rf"{quantity_name} eigenvalues along optimization path "
                        rf"(L={L_int}, iteration={iteration_int}, {keep_label})"
                    ),
                    outpath=outpath,
                    eps=eps,
                    ylabel=ylabel,
                )
                save_eigenvalue_histogram_across_trials(
                    eigs_L[:, time_idx, :],
                    outdir=os.path.join(
                        outdir,
                        "histograms",
                        f"L{L_int}",
                    ),
                    matrix_tag=matrix_tag,
                    matrix_label=quantity_name,
                    num_layers=L_int,
                    context_tag="opt_path",
                    context_label="optimization path",
                    iteration=iteration_int,
                    condition_tag="reduced0123",
                    condition_label="reduced keep=(0,1,2,3)",
                    color=(METRIC_COLORS["hs"] if "Gram" in quantity_name else METRIC_COLORS["qfim"]),
                )
                saved_paths.append(outpath)

        return saved_paths


    def save_signed_eigs_optimization_path_by_iteration(
        eigs_history_by_layer: dict,
        layers,
        sample_iters_for_labels,
        *,
        outdir: str,
        target_iterations=None,
        threshold: float = QFIM_EFFECTIVE_RANK_THRESHOLD,
        eps: float = QFIM_EIG_PLOT_EPS,
        matrix_tag: str = "dpqc_energy_hessian",
        quantity_name: str = "Energy Hessian",
        ylabel: str = "Energy Hessian eigenvalue",
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
            desc=f"{quantity_name} eig distributions along optimization path",
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
                    "Each Hessian eigenvalue history array must be 3D: "
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
                    f"iter{iteration_int:06d}.pdf",
                )

                save_signed_eigs_by_index(
                    eigs_L[:, time_idx, :],
                    title=(
                        rf"{quantity_name} eigenvalues along optimization path "
                        rf"(L={L_int}, iteration={iteration_int})"
                    ),
                    outpath=outpath,
                    threshold=threshold,
                    eps=eps,
                    ylabel=ylabel,
                )
                save_eigenvalue_histogram_across_trials(
                    eigs_L[:, time_idx, :],
                    outdir=os.path.join(
                        outdir,
                        "histograms",
                        f"L{L_int}",
                    ),
                    matrix_tag=matrix_tag,
                    matrix_label=quantity_name,
                    num_layers=L_int,
                    context_tag="opt_path",
                    context_label="optimization path",
                    iteration=iteration_int,
                    color=METRIC_COLORS["hessian"],
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

    if hs_path_layer_list:
        hs_eigs_optimization_path_target_iterations = qfim_path_eig_target_iterations(
            hs_sample_iters,
            every=sample_every,
        )

        hs_eigs_optimization_path_dir = os.path.join(
            hs_eigs_dir,
            f"optimization_path_{keep_key}",
        )

        hs_eigs_optimization_path_files = save_qfim_eigs_optimization_path_by_iteration(
            hs_eigs_history_optimization_path_by_layer,
            hs_path_layer_list,
            hs_sample_iters,
            outdir=hs_eigs_optimization_path_dir,
            target_iterations=hs_eigs_optimization_path_target_iterations,
            eps=QFIM_EIG_PLOT_EPS,
            matrix_tag="dpqc_hs_gram",
            quantity_name="HS tangent Gram",
            ylabel="HS tangent Gram eigenvalue",
        )
    else:
        hs_eigs_optimization_path_files = []

    if hessian_path_layer_list:
        hessian_eigs_optimization_path_target_iterations = (
            qfim_path_eig_target_iterations(
                hessian_sample_iters,
                every=sample_every,
            )
        )
        hessian_eigs_optimization_path_files = (
            save_signed_eigs_optimization_path_by_iteration(
                hessian_eigs_history_optimization_path_by_layer,
                hessian_path_layer_list,
                hessian_sample_iters,
                outdir=os.path.join(hessian_eigs_dir, "optimization_path"),
                target_iterations=hessian_eigs_optimization_path_target_iterations,
                threshold=hessian_path_rank_threshold,
                eps=QFIM_EIG_PLOT_EPS,
                quantity_name="Energy Hessian",
                ylabel="Energy Hessian eigenvalue",
            )
        )
    else:
        hessian_eigs_optimization_path_files = []


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
        ylabel_prefix: str = "Mean QFIM eigenvalue count",
        eigenvalue_symbol: str = r"\lambda_i",
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
        threshold_label = _qfim_threshold_tex_for_label(threshold)
        x = np.asarray(sample_iters_for_labels, dtype=NP_REAL_DTYPE)
        fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.22)
        num_layers = len(valid_layers)

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
            num_valid = np.sum(valid, axis=0)
            sums = np.sum(np.where(valid, counts_L, 0.0), axis=0)
            means = np.divide(
                sums,
                num_valid,
                out=np.full(x.shape, np.nan, dtype=NP_REAL_DTYPE),
                where=num_valid > 0,
            )

            centered = np.where(valid, counts_L - means[None, :], np.nan)
            sq = np.nansum(centered**2, axis=0)
            stds = np.sqrt(
                np.divide(
                    sq,
                    num_valid - 1.0,
                    out=np.zeros_like(sq, dtype=NP_REAL_DTYPE),
                    where=num_valid > 1,
                )
            )
            sems = np.divide(
                stds,
                np.sqrt(num_valid),
                out=np.zeros_like(stds, dtype=NP_REAL_DTYPE),
                where=num_valid > 1,
            )

            finite_mask = np.isfinite(means) & (num_valid > 0)
            if not np.any(finite_mask):
                continue

            color = cmap(layer_idx / max(num_layers - 1, 1))
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
        ax.set_ylabel(
            rf"{ylabel_prefix} (${eigenvalue_symbol} \geq {threshold_label}$)"
        )
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
        ylabel_prefix: str = "Minimum QFIM eigenvalue count",
        eigenvalue_symbol: str = r"\lambda_i",
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
        threshold_label = _qfim_threshold_tex_for_label(threshold)
        x = np.asarray(sample_iters_for_labels, dtype=NP_REAL_DTYPE)
        fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.22)
        num_layers = len(valid_layers)

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
            num_valid = np.sum(valid, axis=0)
            counts_for_min = np.where(valid, counts_L, np.inf)
            min_counts = np.min(counts_for_min, axis=0)
            min_counts = np.where(num_valid > 0, min_counts, np.nan)
            finite_mask = np.isfinite(min_counts) & (num_valid > 0)

            if not np.any(finite_mask):
                continue

            color = cmap(layer_idx / max(num_layers - 1, 1))
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
        ax.set_ylabel(
            rf"{ylabel_prefix} (${eigenvalue_symbol} \geq {threshold_label}$)"
        )
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
        threshold_label = _qfim_threshold_tex_for_label(threshold)

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

    if hs_path_layer_list:
        hs_eigcount_sample_iters = np.asarray(hs_sample_iters, dtype=NP_INT_DTYPE)
        hs_eigcount_time_indices = np.arange(
            hs_eigcount_sample_iters.size,
            dtype=NP_INT_DTYPE,
        )

        for threshold in QFIM_PATH_EIGCOUNT_THRESHOLDS:
            eigcount_history_all_by_layer = qfim_eigcount_history_by_layer(
                hs_eigs_history_optimization_path_by_layer,
                hs_path_layer_list,
                hs_sample_iters,
                threshold=threshold,
            )
            eigcount_history_by_layer = {
                int(L): np.asarray(counts, dtype=NP_REAL_DTYPE)[
                    :,
                    hs_eigcount_time_indices,
                ]
                for L, counts in eigcount_history_all_by_layer.items()
            }
            threshold_tag = _thr_tag(threshold)
            threshold_label = _qfim_threshold_tex_for_label(threshold)

            plot_qfim_eigcount_history_mean_by_layer(
                eigcount_history_by_layer,
                hs_path_layer_list,
                hs_eigcount_sample_iters,
                threshold=threshold,
                title=(
                    rf"Mean HS tangent Gram eigenvalue count along optimization path "
                    rf"($\mu_i \geq {threshold_label}$, {keep_label})"
                ),
                outpath=os.path.join(
                    hs_eigcount_optimization_path_mean_dir,
                    f"hs_eigcount_mean_history_optimization_path_thr_{threshold_tag}_{keep_key}.pdf",
                ),
                ylabel_prefix="Mean HS eigenvalue count",
                eigenvalue_symbol=r"\mu_i",
                cmap=cmap,
            )

            plot_qfim_eigcount_history_min_by_layer(
                eigcount_history_by_layer,
                hs_path_layer_list,
                hs_eigcount_sample_iters,
                threshold=threshold,
                title=(
                    rf"Minimum HS tangent Gram eigenvalue count along optimization path "
                    rf"($\mu_i \geq {threshold_label}$, {keep_label})"
                ),
                outpath=os.path.join(
                    hs_eigcount_optimization_path_min_dir,
                    f"hs_eigcount_min_history_optimization_path_thr_{threshold_tag}_{keep_key}.pdf",
                ),
                ylabel_prefix="Minimum HS eigenvalue count",
                eigenvalue_symbol=r"\mu_i",
                cmap=cmap,
            )


    # ============================================================
    # Mean ﾂｱ SEM scalar QFIM diagnostics
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
        color=METRIC_COLORS["qfim"],
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

        # Save the two complementary summaries alongside the mean figure.
        for statistic, adjective in (("min", "Minimum"), ("max", "Maximum")):
            xs, ys = [], []
            for L in layers:
                values = metric_by_layer.get(L)
                if values is None:
                    continue
                finite = np.asarray(values, dtype=NP_REAL_DTYPE).reshape(-1)
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    xs.append(L)
                    ys.append(getattr(np, statistic)(finite))
            if not xs:
                continue
            sibling = outpath.replace("_mean_", f"_{statistic}_")
            fig_extreme, ax_extreme = new_fig_ax(outside_legend=False)
            ax_extreme.plot(xs, ys, marker=marker, linestyle="-", color=color,
                            linewidth=1.2, markersize=6.0, label=f"{adjective} {label}")
            ax_extreme.set_xlabel("Number of Layers")
            ax_extreme.set_ylabel(ylabel.replace("Mean ", f"{adjective} "))
            ax_extreme.set_title(title.replace("mean ", f"{adjective.lower()} ").replace("Mean ", f"{adjective} "))
            ax_extreme.set_xticks(xs)
            if log_scale:
                ax_extreme.set_yscale("log")
            ax_extreme.grid(True, axis="y", alpha=0.3)
            ax_extreme.legend(loc="best", frameon=True, framealpha=0.9)
            save_fig(fig_extreme, ax_extreme, sibling, outside_legend=False)


    plot_metric_mean_sem_by_layer(
        qfim_trace_history_optimization_path_by_layer,
        qfim_trace_history_layer_list,
        ylabel="Mean QFIM trace",
        title=rf"QFIM trace mean $\pm$ SEM vs Layers along optimization path ({keep_label})",
        outpath=os.path.join(
            qfim_trace_optimization_path_dir,
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
            qfim_abs_entry_sum_random_dir,
            f"qfim_abs_entry_sum_mean_errorbar_{keep_key}.pdf",
        ),
        label=r"$\sum_{i,j} |F_{ij}|$",
        color=METRIC_COLORS["qfim"],
        marker="s",
        log_scale=False,
    )

    if hs_path_layer_list:
        plot_metric_mean_sem_by_layer(
            hs_trace_history_optimization_path_by_layer,
            hs_trace_history_layer_list,
            ylabel="Mean HS tangent Gram trace",
            title=(
                rf"HS tangent Gram trace mean $\pm$ SEM vs Layers along "
                rf"optimization path ({keep_label})"
            ),
            outpath=os.path.join(
                hs_trace_optimization_path_dir,
                f"hs_trace_mean_errorbar_optimization_path_{keep_key}.pdf",
            ),
            label=r"$\sum_k \mu_k(G)$",
            color=METRIC_COLORS["hs"],
            marker="o",
            log_scale=False,
        )

    if hs_layer_list:
        plot_metric_mean_sem_by_layer(
            hs_abs_entry_sum_reduced_0123_by_layer,
            hs_layer_list,
            ylabel="Mean HS elementwise absolute sum",
            title=(
                rf"HS tangent Gram elementwise-absolute-sum mean $\pm$ SEM "
                rf"vs Layers ({keep_label}) at {NUM_QFIM_SAMPLES} random points"
            ),
            outpath=os.path.join(
                hs_abs_entry_sum_random_dir,
                f"hs_abs_entry_sum_mean_errorbar_{keep_key}.pdf",
            ),
            label=r"$\sum_{i,j} |G_{ij}|$",
            color=METRIC_COLORS["hs"],
            marker="s",
            log_scale=False,
        )

    if hessian_path_layer_list:
        plot_metric_mean_sem_by_layer(
            hessian_trace_history_optimization_path_by_layer,
            hessian_trace_history_layer_list,
            ylabel="Mean energy Hessian trace",
            title="Energy Hessian trace mean $\\pm$ SEM vs Layers along optimization path",
            outpath=os.path.join(
                hessian_trace_optimization_path_dir,
                "hessian_trace_mean_errorbar_optimization_path.pdf",
            ),
            label=r"$\sum_k \eta_k(\nabla^2 E)$",
            color=METRIC_COLORS["hessian"],
            marker="o",
            log_scale=False,
        )

        plot_metric_mean_sem_by_layer(
            hessian_abs_eigsum_history_optimization_path_by_layer,
            hessian_abs_eigsum_history_layer_list,
            ylabel="Mean Hessian absolute eigenvalue sum",
            title=(
                "Energy Hessian absolute-eigenvalue-sum mean $\\pm$ SEM "
                "vs Layers along optimization path"
            ),
            outpath=os.path.join(
                hessian_abs_eigsum_optimization_path_dir,
                "hessian_abs_eigsum_mean_errorbar_optimization_path.pdf",
            ),
            label=r"$\sum_k |\eta_k(\nabla^2 E)|$",
            color=METRIC_COLORS["hessian"],
            marker="s",
            log_scale=False,
        )

    if hessian_layer_list:
        plot_metric_mean_sem_by_layer(
            hessian_abs_eigsum_by_layer,
            hessian_layer_list,
            ylabel="Mean Hessian absolute eigenvalue sum",
            title=(
                rf"Energy Hessian absolute-eigenvalue-sum mean $\pm$ SEM "
                rf"vs Layers at {NUM_QFIM_SAMPLES} random points"
            ),
            outpath=os.path.join(
                hessian_abs_eigsum_random_dir,
                "hessian_abs_eigsum_mean_errorbar_random_points.pdf",
            ),
            label=r"$\sum_k |\eta_k(\nabla^2 E)|$",
            color=METRIC_COLORS["hessian"],
            marker="s",
            log_scale=False,
        )

    # ============================================================
    # Reduced keep=(0,1,2,3,4): retain the center ancilla qubit 4.
    # ============================================================
    keep5_key = "keep01234"
    keep5_label = "Reduced (0,1,2,3,4)"
    qfim5_random = load_npz_result(
        os.path.join(qfim_results_dir, "qfim_random_points_keep01234.npz")
    )
    hs5_random = load_npz_result(
        os.path.join(hs_results_dir, "hs_random_points_keep01234.npz")
    )
    keep5_layers = [int(L) for L in np.asarray(qfim5_random["layers"], dtype=NP_INT_DTYPE)]
    qfim5_rank = _load_layer_arrays_from_npz(qfim5_random, keep5_layers, "rank", dtype=NP_REAL_DTYPE)
    qfim5_eigs = _load_layer_arrays_from_npz(qfim5_random, keep5_layers, "eigs_desc", dtype=NP_REAL_DTYPE)
    qfim5_trace = _load_layer_arrays_from_npz(qfim5_random, keep5_layers, "trace", dtype=NP_REAL_DTYPE)
    qfim5_abs = _load_layer_arrays_from_npz(qfim5_random, keep5_layers, "abs_entry_sum", dtype=NP_REAL_DTYPE)
    hs5_rank = _load_layer_arrays_from_npz(hs5_random, keep5_layers, "rank", dtype=NP_REAL_DTYPE)
    hs5_eigs = _load_layer_arrays_from_npz(hs5_random, keep5_layers, "eigs_desc", dtype=NP_REAL_DTYPE)

    for metric, eigs, representation_dir, symbol in (
        ("QFIM", qfim5_eigs, qfim_keep01234_fig_dir, r"\lambda_i"),
        ("HS tangent Gram", hs5_eigs, hs_keep01234_fig_dir, r"\mu_i"),
    ):
        eigs_output_dir = os.path.join(representation_dir, "eigs")
        eigcount_output_dir = os.path.join(
            representation_dir, "eigcount", "random_points"
        )
        os.makedirs(eigs_output_dir, exist_ok=True)
        os.makedirs(eigcount_output_dir, exist_ok=True)
        plot_qfim_random_eigcount_threshold_overlay(
            eigs, keep5_layers, QFIM_PATH_EIGCOUNT_THRESHOLDS,
            title=f"{metric} eigenvalue count at random points by threshold ({keep5_label})",
            outpath=os.path.join(eigcount_output_dir, f"{metric.lower().split()[0]}_eigcount_random_{keep5_key}.pdf"),
            ylabel=f"Mean {metric} eigenvalue count", eigenvalue_symbol=symbol, cmap=cmap,
        )
        for L in keep5_layers:
            save_eigenvalue_histograms_by_trial(
                eigs[L], outdir=os.path.join(eigs_output_dir, "histograms", "random_points", f"L{L}"),
                matrix_tag=f"dpqc_{metric.lower().replace(' ', '_')}", matrix_label=metric,
                num_layers=L, context_tag="random", context_label="random point",
                condition_tag="reduced01234", condition_label="reduced keep=(0,1,2,3,4)",
                color=METRIC_COLORS["qfim" if metric == "QFIM" else "hs"],
            )

    for tag, label, values, color in (
        ("qfim_rank", "QFIM effective rank", qfim5_rank, METRIC_COLORS["qfim"]),
        ("qfim_trace", "QFIM trace", qfim5_trace, METRIC_COLORS["qfim"]),
        ("qfim_abs_entry_sum", "QFIM elementwise absolute sum", qfim5_abs, METRIC_COLORS["qfim"]),
        ("hs_rank", "HS effective rank", hs5_rank, METRIC_COLORS["hs"]),
    ):
        representation_dir = (
            qfim_keep01234_fig_dir if tag.startswith("qfim") else hs_keep01234_fig_dir
        )
        if tag.endswith("rank"):
            category = "rank"
        elif "trace" in tag:
            category = "trace"
        else:
            category = "abs_entry_sum"
        random_output_dir = os.path.join(
            representation_dir, category, "random_points"
        )
        os.makedirs(random_output_dir, exist_ok=True)
        plot_metric_mean_sem_by_layer(
            values, keep5_layers, ylabel=f"Mean {label}",
            title=f"{label} mean ± SEM vs Layers ({keep5_label})",
            outpath=os.path.join(random_output_dir, f"{tag}_mean_random_{keep5_key}.pdf"),
            label=label, color=color,
        )

    for metric, result_dir, color in (
        ("qfim", qfim_results_dir, METRIC_COLORS["qfim"]),
        ("hs", hs_results_dir, METRIC_COLORS["hs"]),
    ):
        rank_result = load_npz_result(os.path.join(result_dir, f"{metric}_rank_history_optimization_path_keep01234.npz"))
        eigs_result = load_npz_result(os.path.join(result_dir, f"{metric}_eigs_history_optimization_path_keep01234.npz"))
        trace_result = load_npz_result(os.path.join(result_dir, f"{metric}_trace_history_optimization_path_keep01234.npz"))
        layers5 = [int(L) for L in np.asarray(rank_result["layers"], dtype=NP_INT_DTYPE)]
        iters5 = np.asarray(rank_result["sample_iters"], dtype=NP_INT_DTYPE)
        ranks5 = _load_layer_arrays_from_npz(rank_result, layers5, suffix=None, dtype=NP_REAL_DTYPE)
        eigs5 = _load_layer_arrays_from_npz(eigs_result, layers5, suffix=None, dtype=NP_REAL_DTYPE)
        traces5 = _load_layer_arrays_from_npz(trace_result, layers5, suffix=None, dtype=NP_REAL_DTYPE)
        representation_dir = qfim_keep01234_fig_dir if metric == "qfim" else hs_keep01234_fig_dir
        rank_path_dir = os.path.join(representation_dir, "rank", "optimization_path")
        rank_mean_dir = os.path.join(rank_path_dir, "mean")
        rank_min_dir = os.path.join(rank_path_dir, "min")
        trace_path_dir = os.path.join(representation_dir, "trace", "optimization_path")
        eigcount_path_dir = os.path.join(representation_dir, "eigcount", "optimization_path")
        for output_dir in (rank_mean_dir, rank_min_dir, trace_path_dir, eigcount_path_dir):
            os.makedirs(output_dir, exist_ok=True)
        plot_qfim_rank_history_mean_by_layer(
            ranks5, layers5, iters5, title=f"Mean {metric.upper()} rank along optimization path ({keep5_label})",
            outpath=os.path.join(rank_mean_dir, f"{metric}_rank_mean_{keep5_key}.pdf"), cmap=cmap,
        )
        plot_qfim_rank_history_min_by_layer(
            ranks5, layers5, iters5, title=f"Minimum {metric.upper()} rank along optimization path ({keep5_label})",
            outpath=os.path.join(rank_min_dir, f"{metric}_rank_min_{keep5_key}.pdf"), cmap=cmap,
        )
        plot_qfim_trace_history_mean_by_layer(
            traces5, layers5, iters5,
            title=f"Mean {metric.upper()} trace along optimization path ({keep5_label})",
            outpath=os.path.join(trace_path_dir, f"{metric}_trace_mean_{keep5_key}.pdf"),
            ylabel=f"Mean {metric.upper()} trace", cmap=cmap,
        )
        plot_qfim_random_eigcount_threshold_overlay(
            eigs5, layers5, QFIM_PATH_EIGCOUNT_THRESHOLDS,
            title=f"{metric.upper()} eigenvalue count along optimization path ({keep5_label})",
            outpath=os.path.join(eigcount_path_dir, f"{metric}_eigcount_by_layer_{keep5_key}.pdf"),
            ylabel=f"Mean {metric.upper()} eigenvalue count", cmap=cmap,
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
    # is applied here. Plot-level positive floors are only for numerical safety
    # in log-scale visualization.
    # ============================================================

    qfim_grad_align_dir = os.path.join(qfim_keep0123_fig_dir, "grad_alignment")
    qfim_grad_align_results_dir = os.path.join(qfim_results_dir, "grad_alignment")
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

    if RUN_QFIM_GRAD_ALIGNMENT_FINAL_ITER or RUN_QFIM_GRAD_ALIGNMENT_ALL_TIMES:
        for use_all_times in (False, True):
            if use_all_times and not RUN_QFIM_GRAD_ALIGNMENT_ALL_TIMES:
                continue
            if (not use_all_times) and not RUN_QFIM_GRAD_ALIGNMENT_FINAL_ITER:
                continue

            time_tag = "all_times" if use_all_times else "final_iter"
            title_time = "all sampled iterations" if use_all_times else f"final iteration {int(sample_iters[-1])}"
            color_by = "iteration" if use_all_times else None
            point_size = 12.0 if use_all_times else 14.0
            scatter_alpha = 0.40 if use_all_times else 0.45

            table_by_layer = {}
            for L in vqe_layer_list:
                path = os.path.join(
                    qfim_grad_align_results_dir,
                    f"qfim_grad_alignment_scatter_data_L{int(L)}_{time_tag}.npz",
                )
                table = _load_alignment_table_or_none(path)
                if table is None:
                    continue
                table_by_layer[int(L)] = table
                plot_qfim_grad_alignment_table(
                    table,
                    title=rf"QFIM eigenvalue vs gradient weight, L={int(L)}, {title_time}",
                    outpath=os.path.join(
                        qfim_grad_align_dir,
                        f"qfim_grad_weight_scatter_L{int(L)}_{time_tag}.pdf",
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
                    title=rf"QFIM eigenvalue vs gradient weight across layers, {time_tag.replace('_', ' ')}",
                    outpath=os.path.join(
                        qfim_grad_align_dir,
                        f"qfim_grad_weight_scatter_overlay_layers_{time_tag}.pdf",
                    ),
                    log_x=LOG_X_QFIM_GRAD_ALIGNMENT,
                    log_y=LOG_Y_QFIM_GRAD_ALIGNMENT,
                    point_size=12.0,
                    alpha=0.40,
                )

    if RUN_QFIM_GRAD_ALIGNMENT_PER_ITERATION:
        for L in vqe_layer_list:
            layer_dir = os.path.join(qfim_grad_align_dir, f"L{int(L)}")
            os.makedirs(layer_dir, exist_ok=True)
            for iteration in QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS:
                iteration = int(iteration)
                iter_tag = f"iter{iteration:06d}"
                path = os.path.join(
                    qfim_grad_align_results_dir,
                    f"L{int(L)}",
                    f"qfim_grad_alignment_scatter_data_L{int(L)}_{iter_tag}.npz",
                )
                table = _load_alignment_table_or_none(path)
                if table is None:
                    continue
                plot_qfim_grad_alignment_table(
                    table,
                    title=rf"QFIM eigenvalue vs gradient weight, L={int(L)}, iteration {iteration}",
                    outpath=os.path.join(
                        layer_dir,
                        f"qfim_grad_weight_scatter_L{int(L)}_{iter_tag}.pdf",
                    ),
                    log_x=LOG_X_QFIM_GRAD_ALIGNMENT,
                    log_y=LOG_Y_QFIM_GRAD_ALIGNMENT,
                    color_by=None,
                    point_size=14.0,
                    alpha=0.45,
                )

    print(f"Saved figures to: {save_dir}")



if __name__ == "__main__":
    run_dpqc_overparam_visualize()




