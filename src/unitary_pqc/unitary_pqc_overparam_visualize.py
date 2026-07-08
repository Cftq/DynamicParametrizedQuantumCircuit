#!/usr/bin/env python
# coding: utf-8
"""Visualize saved Unitary-PQC numerical results.

Run unitary_pqc_overparam_compute.py first. This script loads saved .npz
results under figs/unitary_pqc/h_<h_param>/numerical_results and generates
figures/circuit drawings without recomputing VQE or QFIM quantities.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _MODULE_DIR.parent
_COMMON_DIR = _SRC_DIR / "common"
for _path in (_MODULE_DIR, _COMMON_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from matplotlib.patches import Patch

import config_overparam as cfg
import unitary_pqc_overparam_compute as upqc
from dpqc_overparam_common import load_npz_result, save_circuit_matplotlib_png


NP_REAL_DTYPE = np.float64
NP_INT_DTYPE = np.int64


def _load_unitary_vqe_results() -> dict:
    path = os.path.join(
        upqc.energy_results_dir,
        "vqe_optimization_results.npz",
    )
    result = load_npz_result(path)

    upqc.layer_list = [
        int(L)
        for L in np.asarray(result["layers"], dtype=NP_INT_DTYPE)
    ]
    upqc.sample_iters = np.asarray(result["sample_iters"], dtype=NP_INT_DTYPE)
    upqc.sample_iter_set = set(int(t) for t in upqc.sample_iters.tolist())
    upqc.steps = int(np.asarray(result["steps"]).item())
    upqc.num_runs = int(np.asarray(result["num_runs"]).item())
    upqc.tolerance = float(np.asarray(result["tolerance"]).item())
    upqc.lr = float(np.asarray(result["learning_rate"]).item())
    upqc.smallest_eigval = float(np.asarray(result["smallest_eigval"]).item())
    upqc.cmap = matplotlib.colormaps.get_cmap("viridis")

    upqc.final_stats = {
        "layer": np.asarray(result["final_stats_layer"], dtype=NP_INT_DTYPE).tolist(),
        "success_rate": np.asarray(
            result["final_stats_success_rate"],
            dtype=NP_REAL_DTYPE,
        ).tolist(),
        "mean_energy": np.asarray(
            result["final_stats_mean_energy"],
            dtype=NP_REAL_DTYPE,
        ).tolist(),
        "std_energy": np.asarray(
            result["final_stats_std_energy"],
            dtype=NP_REAL_DTYPE,
        ).tolist(),
    }

    upqc.theta_history = {}
    upqc.best_theta_by_layer = {}
    upqc.final_theta_wrapped_rmsdist_by_layer = {}
    upqc.energy_traces_by_layer = {}
    upqc.grad_norm_traces_by_layer = {}
    upqc.theta_sample_traces_by_layer = {}
    upqc.grad_sample_traces_by_layer = {}
    upqc.energy_mean_history = {}
    upqc.energy_std_history = {}
    upqc.success_rates_history = {}

    for L in upqc.layer_list:
        upqc.theta_history[L] = np.asarray(
            result[f"L{L}_theta_history"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.best_theta_by_layer[L] = np.asarray(
            result[f"L{L}_best_theta"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.final_theta_wrapped_rmsdist_by_layer[L] = np.asarray(
            result[f"L{L}_final_theta_wrapped_rmsdist"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.energy_traces_by_layer[L] = np.asarray(
            result[f"L{L}_energy_traces"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.grad_norm_traces_by_layer[L] = np.asarray(
            result[f"L{L}_grad_norm_traces"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.theta_sample_traces_by_layer[L] = np.asarray(
            result[f"L{L}_theta_samples"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.grad_sample_traces_by_layer[L] = np.asarray(
            result[f"L{L}_grad_samples"],
            dtype=NP_REAL_DTYPE,
        )

        energy_data = upqc.energy_traces_by_layer[L]
        upqc.energy_mean_history[L] = np.mean(energy_data, axis=0)
        upqc.energy_std_history[L] = np.std(energy_data, axis=0)
        diffs = np.abs(energy_data - upqc.smallest_eigval)
        upqc.success_rates_history[L] = np.mean(diffs <= upqc.tolerance, axis=0)

    return result


def _save_optimized_circuit_drawings() -> None:
    os.makedirs(upqc.circuit_dir, exist_ok=True)

    for L in upqc.layer_list:
        best_tc_circ = upqc.create_unitary_pqc(
            jnp.asarray(upqc.best_theta_by_layer[L], dtype=upqc.REAL_DTYPE),
            num_layers=L,
            num_qubits=upqc.num_total_qubits,
        )
        save_circuit_matplotlib_png(
            best_tc_circ,
            os.path.join(upqc.circuit_dir, f"optimized_circuit_L{L}.png"),
            num_qubits=upqc.num_total_qubits,
            dpi=upqc.SAVE_DPI,
            pad_inches=upqc.SAVEFIG_PAD_INCHES,
            save_pdf=upqc.CIRCUIT_SAVE_PDF,
            hide_params=True,
        )


def _load_random_qfim_results() -> None:
    qfim_path = os.path.join(upqc.qfim_results_dir, "qfim_random_points.npz")
    qfim_result = load_npz_result(qfim_path)
    layers = [int(L) for L in np.asarray(qfim_result["layers"], dtype=NP_INT_DTYPE)]

    upqc.layer_list = layers
    upqc.NUM_QFIM_SAMPLES = int(
        np.asarray(qfim_result["num_qfim_samples"]).item()
    )
    upqc.QFIM_SAMPLE_SEED_BASE = int(
        np.asarray(qfim_result["qfim_sample_seed_base"]).item()
    )
    upqc.RED_JVP_CHUNK = int(np.asarray(qfim_result["red_jvp_chunk"]).item())
    upqc.PURE_QFIM_LAYER_THRESHOLD = int(
        np.asarray(qfim_result["pure_qfim_layer_threshold"]).item()
    )

    upqc.qfim_random_thetas_by_layer = {}
    upqc.qfim_rank_reduced_by_layer = {}
    upqc.qfim_eigs_reduced_by_layer = {}
    upqc.qfim_thresh_reduced_by_layer = {}
    upqc.qfim_rank_pure_by_layer = {}
    upqc.qfim_eigs_pure_by_layer = {}
    upqc.qfim_thresh_pure_by_layer = {}

    for L in layers:
        upqc.qfim_random_thetas_by_layer[L] = np.asarray(
            qfim_result[f"L{L}_theta"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.qfim_rank_reduced_by_layer[L] = np.asarray(
            qfim_result[f"L{L}_rank_reduced"],
            dtype=NP_INT_DTYPE,
        )
        upqc.qfim_eigs_reduced_by_layer[L] = np.asarray(
            qfim_result[f"L{L}_eigs_reduced_desc"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.qfim_thresh_reduced_by_layer[L] = np.asarray(
            qfim_result[f"L{L}_rank_threshold_reduced"],
            dtype=NP_REAL_DTYPE,
        )

        pure_rank_key = f"L{L}_rank_pure"
        if pure_rank_key in qfim_result:
            upqc.qfim_rank_pure_by_layer[L] = np.asarray(
                qfim_result[pure_rank_key],
                dtype=NP_INT_DTYPE,
            )
            upqc.qfim_eigs_pure_by_layer[L] = np.asarray(
                qfim_result[f"L{L}_eigs_pure_desc"],
                dtype=NP_REAL_DTYPE,
            )
            upqc.qfim_thresh_pure_by_layer[L] = np.asarray(
                qfim_result[f"L{L}_rank_threshold_pure"],
                dtype=NP_REAL_DTYPE,
            )
        else:
            upqc.qfim_rank_pure_by_layer[L] = None
            upqc.qfim_eigs_pure_by_layer[L] = None
            upqc.qfim_thresh_pure_by_layer[L] = None

    hs_path = os.path.join(
        upqc.qfim_results_dir,
        "hs_random_points_reduced_0123.npz",
    )
    hs_result = load_npz_result(hs_path)

    upqc.hs_rank_reduced_by_layer = {}
    upqc.hs_eigs_reduced_by_layer = {}
    upqc.hs_thresh_reduced_by_layer = {}

    for L in layers:
        upqc.hs_rank_reduced_by_layer[L] = np.asarray(
            hs_result[f"L{L}_rank"],
            dtype=NP_INT_DTYPE,
        )
        upqc.hs_eigs_reduced_by_layer[L] = np.asarray(
            hs_result[f"L{L}_eigs_desc"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.hs_thresh_reduced_by_layer[L] = np.asarray(
            hs_result[f"L{L}_rank_threshold"],
            dtype=NP_REAL_DTYPE,
        )


def _plot_random_qfim_results() -> None:
    qfim_eigs_dir = os.path.join(upqc.save_dir, "qfim_eigs")
    qfim_eigs_pure_dir = os.path.join(qfim_eigs_dir, "pure_full")
    qfim_eigs_reduced_0123_dir = os.path.join(
        qfim_eigs_dir,
        "reduced_keep_0123",
    )
    hs_eigs_dir = os.path.join(upqc.save_dir, "hs_eigs")
    hs_eigs_reduced_0123_dir = os.path.join(hs_eigs_dir, "reduced_keep_0123")

    os.makedirs(qfim_eigs_pure_dir, exist_ok=True)
    os.makedirs(qfim_eigs_reduced_0123_dir, exist_ok=True)
    os.makedirs(hs_eigs_reduced_0123_dir, exist_ok=True)

    for L in upqc.layer_list:
        upqc._save_qfim_eigs_violinplot_by_index(
            upqc.qfim_eigs_reduced_by_layer[L],
            title=rf"QFIM eigenvalues at {upqc.NUM_QFIM_SAMPLES} random points (L={L})",
            outpath=os.path.join(qfim_eigs_reduced_0123_dir, f"L{L}_reduced_0123.pdf"),
            rank_thresholds=upqc.qfim_thresh_reduced_by_layer[L],
        )
        upqc._save_qfim_eigs_violinplot_by_index(
            upqc.hs_eigs_reduced_by_layer[L],
            title=rf"HS tangent Gram eigenvalues at {upqc.NUM_QFIM_SAMPLES} random points (L={L})",
            outpath=os.path.join(hs_eigs_reduced_0123_dir, f"L{L}_reduced_0123.pdf"),
            rank_thresholds=upqc.hs_thresh_reduced_by_layer[L],
            ylabel="HS tangent Gram eigenvalue",
        )
        if upqc.qfim_eigs_pure_by_layer[L] is not None:
            upqc._save_qfim_eigs_violinplot_by_index(
                upqc.qfim_eigs_pure_by_layer[L],
                title=rf"QFIM eigenvalues (Pure full-state) at {upqc.NUM_QFIM_SAMPLES} random points (L={L})",
                outpath=os.path.join(qfim_eigs_pure_dir, f"L{L}_pure_full.pdf"),
                rank_thresholds=upqc.qfim_thresh_pure_by_layer[L],
            )

    x_all = np.array(upqc.layer_list, dtype=NP_REAL_DTYPE)
    x_labels = [str(L) for L in upqc.layer_list]
    dx = 0.25
    violin_w_rank = 0.20
    num_layers = len(upqc.layer_list)

    upqc.new_prx_figure(width="double")
    ax = plt.gca()

    for idx, L in enumerate(upqc.layer_list):
        color = upqc.cmap(idx / num_layers)
        red_dataset = upqc._make_violin_ready(
            upqc.qfim_rank_reduced_by_layer[L],
            ensure_positive=False,
            tiny=1e-12,
        )
        vp_red = plt.violinplot(
            [red_dataset],
            positions=[float(L) + dx],
            widths=violin_w_rank,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        upqc._style_violin(
            vp_red,
            facecolor=color,
            edgecolor=color,
            alpha=0.12,
            linewidth=1.0,
            hatch="///",
            linecolor=color,
            linealpha=0.7,
        )

    pure_layers = [
        L for L in upqc.layer_list
        if upqc.qfim_rank_pure_by_layer[L] is not None
    ]
    for L in pure_layers:
        idx = upqc.layer_list.index(L)
        color = upqc.cmap(idx / num_layers)
        pure_dataset = upqc._make_violin_ready(
            upqc.qfim_rank_pure_by_layer[L],
            ensure_positive=False,
            tiny=1e-12,
        )
        vp_pure = plt.violinplot(
            [pure_dataset],
            positions=[float(L) - dx],
            widths=violin_w_rank,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        upqc._style_violin(
            vp_pure,
            facecolor=color,
            edgecolor=color,
            alpha=0.20,
            linewidth=1.0,
            linecolor=color,
            linealpha=0.7,
        )

    ax.set_xticks(x_all)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(r"QFIM effective rank $(\lambda_k > 10^{-12})$")
    upqc.set_prx_title(
        rf"QFIM rank at {upqc.NUM_QFIM_SAMPLES} random points",
        ax=ax,
    )
    ax.grid(True, axis="y", alpha=0.3)
    type_handles = [
        Patch(facecolor="white", edgecolor="black", label="Pure(full)"),
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch="///",
            label=f"Reduced (keep={upqc.KEEP_WIRES})",
        ),
    ]
    ax.legend(handles=type_handles, loc="best", frameon=True, framealpha=0.9)
    upqc.save_current_figure(
        os.path.join(upqc.save_dir, "qfim_rank_violinplot_random_points.pdf"),
        outside_legend=False,
    )

    upqc.new_prx_figure(width="double")
    ax = plt.gca()

    for idx, L in enumerate(upqc.layer_list):
        color = upqc.cmap(idx / num_layers)
        hs_dataset = upqc._make_violin_ready(
            upqc.hs_rank_reduced_by_layer[L],
            ensure_positive=False,
            tiny=1e-12,
        )
        vp_hs = plt.violinplot(
            [hs_dataset],
            positions=[float(L)],
            widths=violin_w_rank,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        upqc._style_violin(
            vp_hs,
            facecolor=color,
            edgecolor=color,
            alpha=0.18,
            linewidth=1.0,
            linecolor=color,
            linealpha=0.7,
        )

    ax.set_xticks(x_all)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(r"HS effective rank $(\lambda_k > 10^{-12})$")
    upqc.set_prx_title(
        rf"HS tangent Gram rank at {upqc.NUM_QFIM_SAMPLES} random points",
        ax=ax,
    )
    ax.grid(True, axis="y", alpha=0.3)
    upqc.save_current_figure(
        os.path.join(upqc.save_dir, "hs_rank_violinplot_random_points_reduced_0123.pdf"),
        outside_legend=False,
    )

    upqc.plot_qfim_rank_max_by_layer(
        upqc.qfim_rank_pure_by_layer,
        upqc.layer_list,
        color="C0",
        title=rf"Maximum pure full-state QFIM rank at {upqc.NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum QFIM effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(upqc.save_dir, "qfim_rank_max_random_points_pure_full.pdf"),
        marker="s",
        lw=1.0,
    )
    upqc.plot_qfim_rank_max_by_layer(
        upqc.qfim_rank_reduced_by_layer,
        upqc.layer_list,
        color="C0",
        title=rf"Maximum QFIM rank at {upqc.NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum QFIM effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(upqc.save_dir, "qfim_rank_max_random_points_reduced_0123.pdf"),
        marker="o",
        lw=1.0,
    )
    upqc.plot_qfim_rank_max_by_layer(
        upqc.hs_rank_reduced_by_layer,
        upqc.layer_list,
        color="C3",
        title=rf"Maximum HS tangent Gram rank at {upqc.NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum HS effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(upqc.save_dir, "hs_rank_max_random_points_reduced_0123.pdf"),
        marker="D",
        lw=1.0,
    )


def _load_optimization_path_results() -> None:
    qfim_rank_path = os.path.join(
        upqc.qfim_results_dir,
        "qfim_rank_history_optimization_path_reduced_0123.npz",
    )
    qfim_result = load_npz_result(qfim_rank_path)
    layers = [int(L) for L in np.asarray(qfim_result["layers"], dtype=NP_INT_DTYPE)]
    upqc.sample_iters = np.asarray(qfim_result["sample_iters"], dtype=NP_INT_DTYPE)
    upqc.layer_list = layers
    upqc.qfim_rank_history_by_layer = {
        L: np.asarray(qfim_result[f"L{L}_rank"], dtype=NP_REAL_DTYPE)
        for L in layers
    }

    hs_rank_path = os.path.join(
        upqc.qfim_results_dir,
        "hs_rank_history_optimization_path_reduced_0123.npz",
    )
    hs_result = load_npz_result(hs_rank_path)
    upqc.hs_rank_history_by_layer = {
        L: np.asarray(hs_result[f"L{L}_rank"], dtype=NP_REAL_DTYPE)
        for L in layers
    }


def _plot_optimization_path_results() -> None:
    upqc.plot_qfim_rank_history_mean_by_layer(
        upqc.qfim_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Mean QFIM effective rank along optimization path (keep=(0,1,2,3))",
        outpath=os.path.join(
            upqc.save_dir,
            "qfim_rank_mean_history_optimization_path_reduced_0123.pdf",
        ),
        cmap=upqc.cmap,
    )
    upqc.plot_qfim_rank_history_min_by_layer(
        upqc.qfim_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Minimum QFIM effective rank along optimization path (keep=(0,1,2,3))",
        outpath=os.path.join(
            upqc.save_dir,
            "qfim_rank_min_history_optimization_path_reduced_0123.pdf",
        ),
        cmap=upqc.cmap,
    )
    upqc.plot_qfim_rank_history_mean_by_layer(
        upqc.hs_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Mean HS tangent Gram effective rank along optimization path (keep=(0,1,2,3))",
        outpath=os.path.join(
            upqc.save_dir,
            "hs_rank_mean_history_optimization_path_reduced_0123.pdf",
        ),
        ylabel=r"Mean HS effective rank $(\lambda_k > 10^{-12})$",
        cmap=upqc.cmap,
    )
    upqc.plot_qfim_rank_history_min_by_layer(
        upqc.hs_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Minimum HS tangent Gram effective rank along optimization path (keep=(0,1,2,3))",
        outpath=os.path.join(
            upqc.save_dir,
            "hs_rank_min_history_optimization_path_reduced_0123.pdf",
        ),
        ylabel=r"Minimum HS effective rank $(\lambda_k > 10^{-12})$",
        cmap=upqc.cmap,
    )


def _plot_qfim_grad_alignment_results() -> None:
    if cfg.QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS is None:
        target_iterations = tuple(int(t) for t in upqc.sample_iters)
    else:
        target_iterations = tuple(
            int(t) for t in cfg.QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS
        )

    for L in upqc.layer_list:
        layer_data_dir = os.path.join(upqc.qfim_grad_align_results_dir, f"L{L}")
        layer_plot_dir = os.path.join(upqc.qfim_grad_align_dir, f"L{L}")
        os.makedirs(layer_plot_dir, exist_ok=True)

        for iteration in target_iterations:
            iter_tag = f"iter{iteration:06d}"
            data_path = os.path.join(
                layer_data_dir,
                f"qfim_grad_alignment_scatter_data_L{L}_{iter_tag}.npz",
            )
            if not os.path.exists(data_path):
                continue

            table = load_npz_result(data_path)
            upqc.plot_qfim_grad_alignment_table(
                table,
                title=(
                    rf"QFIM eigenvalue vs gradient weight, "
                    rf"L={L}, iteration {iteration}"
                ),
                outpath=os.path.join(
                    layer_plot_dir,
                    f"qfim_grad_weight_scatter_L{L}_{iter_tag}.pdf",
                ),
                log_x=cfg.LOG_X_QFIM_GRAD_ALIGNMENT,
                log_y=cfg.LOG_Y_QFIM_GRAD_ALIGNMENT,
                color_by=None,
                point_size=14.0,
                alpha=0.45,
            )


def run_unitary_pqc_visualization() -> dict:
    upqc.configure_unitary_pqc_overparam()
    _load_unitary_vqe_results()
    _save_optimized_circuit_drawings()
    upqc.plot_vqe_optimization_results()

    _load_random_qfim_results()
    _plot_random_qfim_results()

    _load_optimization_path_results()
    _plot_optimization_path_results()
    _plot_qfim_grad_alignment_results()

    return upqc.collect_unitary_pqc_result()


if __name__ == "__main__":
    run_unitary_pqc_visualization()
