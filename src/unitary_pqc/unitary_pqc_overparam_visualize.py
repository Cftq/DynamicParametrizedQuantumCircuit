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
from typing import Optional

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


def _load_required_result(path: str) -> dict:
    """Load a compute-stage result or explain how to generate it."""
    result_path = Path(path).resolve()
    if not result_path.is_file():
        compute_script = _MODULE_DIR / "unitary_pqc_overparam_compute.py"
        raise FileNotFoundError(
            "Required Unitary-PQC numerical result is missing:\n"
            f"  {result_path}\n"
            "Run the numerical pipeline to successful completion before "
            "visualizing:\n"
            f'  "{sys.executable}" "{compute_script}"'
        )
    return load_npz_result(str(result_path))


def _plot_spectral_count_by_layer(
    eigs_by_layer: dict,
    *,
    title: str,
    ylabel: str,
    outpath: str,
    use_absolute_values: bool = False,
) -> None:
    """Plot mean component counts above each threshold against layer count."""
    layers = [int(L) for L in sorted(eigs_by_layer) if eigs_by_layer[L] is not None]
    thresholds = tuple(float(t) for t in cfg.QFIM_PATH_EIGCOUNT_THRESHOLDS)
    if not layers or not thresholds:
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    cmap = matplotlib.colormaps.get_cmap("viridis")
    for idx, threshold in enumerate(thresholds):
        means, sems = [], []
        for L in layers:
            eigs = np.asarray(eigs_by_layer[L], dtype=NP_REAL_DTYPE)
            if eigs.ndim < 2:
                raise ValueError("Spectral arrays need sample and eigenvalue axes.")
            eigs = eigs.reshape(-1, eigs.shape[-1])
            if use_absolute_values:
                eigs = np.abs(eigs)
            counts = np.sum(eigs >= threshold, axis=1).astype(NP_REAL_DTYPE)
            means.append(np.mean(counts))
            sems.append(0.0 if counts.size < 2 else np.std(counts, ddof=1) / np.sqrt(counts.size))
        ax.errorbar(
            layers, means, yerr=sems, marker="o", linewidth=1.2,
            capsize=3.0, color=cmap(idx / max(len(thresholds) - 1, 1)),
            label=rf"threshold $\geq {threshold:g}$",
        )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def _load_unitary_vqe_results() -> dict:
    path = os.path.join(
        upqc.energy_results_dir,
        "vqe_optimization_results.npz",
    )
    result = _load_required_result(path)

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
    qfim_result = _load_required_result(qfim_path)
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
        upqc.hs_results_dir,
        "hs_random_points_reduced_0123.npz",
    )
    hs_result = _load_required_result(hs_path)

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

    ortk_path = os.path.join(
        upqc.ortk_results_dir,
        "ortk_random_points.npz",
    )
    ortk_result = _load_required_result(ortk_path)
    upqc.ORTK_RANK_THRESHOLD = float(
        np.asarray(ortk_result["ortk_rank_threshold"]).item()
    )
    upqc.ORTK_PARTICIPATION_EPS = float(
        np.asarray(ortk_result["ortk_participation_eps"]).item()
    )
    upqc.ortk_rank_by_layer = {}
    upqc.ortk_effective_rank_by_layer = {}
    upqc.ortk_eigs_by_layer = {}
    upqc.ortk_trace_by_layer = {}

    for L in layers:
        upqc.ortk_rank_by_layer[L] = np.asarray(
            ortk_result[f"L{L}_rank"],
            dtype=NP_INT_DTYPE,
        )
        upqc.ortk_effective_rank_by_layer[L] = np.asarray(
            ortk_result[f"L{L}_effective_rank"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.ortk_eigs_by_layer[L] = np.asarray(
            ortk_result[f"L{L}_eigs_desc"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.ortk_trace_by_layer[L] = np.asarray(
            ortk_result[f"L{L}_trace"],
            dtype=NP_REAL_DTYPE,
        )

    hessian_path = os.path.join(
        upqc.hessian_results_dir,
        "hessian_random_points.npz",
    )
    upqc.hessian_rank_by_layer = {}
    upqc.hessian_eigs_by_layer = {}
    upqc.hessian_thresh_by_layer = {}
    upqc.hessian_trace_by_layer = {}
    upqc.hessian_abs_eigsum_by_layer = {}

    if os.path.exists(hessian_path):
        hessian_result = _load_required_result(hessian_path)

        for L in layers:
            upqc.hessian_rank_by_layer[L] = np.asarray(
                hessian_result[f"L{L}_rank"],
                dtype=NP_INT_DTYPE,
            )
            upqc.hessian_eigs_by_layer[L] = np.asarray(
                hessian_result[f"L{L}_eigs_desc"],
                dtype=NP_REAL_DTYPE,
            )
            upqc.hessian_thresh_by_layer[L] = np.asarray(
                hessian_result[f"L{L}_rank_threshold"],
                dtype=NP_REAL_DTYPE,
            )
            upqc.hessian_trace_by_layer[L] = np.asarray(
                hessian_result[f"L{L}_trace"],
                dtype=NP_REAL_DTYPE,
            )
            upqc.hessian_abs_eigsum_by_layer[L] = np.asarray(
                hessian_result[f"L{L}_abs_eigsum"],
                dtype=NP_REAL_DTYPE,
            )


def _plot_random_qfim_results() -> None:
    qfim_eigs_pure_dir = upqc.qfim_eigs_pure_dir
    qfim_eigs_reduced_0123_dir = upqc.qfim_eigs_reduced_0123_dir
    hs_eigs_reduced_0123_dir = upqc.hs_eigs_reduced_0123_dir
    ortk_eigs_dir = upqc.ortk_eigs_dir
    hessian_eigs_dir = upqc.hessian_eigs_dir
    qfim_rank_random_dir = upqc.qfim_rank_random_dir
    hs_rank_random_dir = upqc.hs_rank_random_dir
    ortk_rank_random_dir = upqc.ortk_rank_random_dir
    ortk_effective_rank_random_dir = upqc.ortk_effective_rank_random_dir
    hessian_rank_random_dir = upqc.hessian_rank_random_dir

    os.makedirs(qfim_eigs_pure_dir, exist_ok=True)
    os.makedirs(qfim_eigs_reduced_0123_dir, exist_ok=True)
    os.makedirs(hs_eigs_reduced_0123_dir, exist_ok=True)
    os.makedirs(ortk_eigs_dir, exist_ok=True)
    os.makedirs(hessian_eigs_dir, exist_ok=True)
    os.makedirs(qfim_rank_random_dir, exist_ok=True)
    os.makedirs(hs_rank_random_dir, exist_ok=True)
    os.makedirs(ortk_rank_random_dir, exist_ok=True)
    os.makedirs(ortk_effective_rank_random_dir, exist_ok=True)
    os.makedirs(hessian_rank_random_dir, exist_ok=True)

    for L in upqc.layer_list:
        upqc._save_qfim_eigs_violinplot_by_index(
            upqc.qfim_eigs_reduced_by_layer[L],
            title=rf"QFIM eigenvalues at {upqc.NUM_QFIM_SAMPLES} random points (L={L})",
            outpath=os.path.join(qfim_eigs_reduced_0123_dir, f"L{L}_reduced_0123.pdf"),
            rank_thresholds=upqc.qfim_thresh_reduced_by_layer[L],
        )
        upqc.plot_style.save_eigenvalue_histograms_by_trial(
            upqc.qfim_eigs_reduced_by_layer[L],
            outdir=os.path.join(
                qfim_eigs_reduced_0123_dir,
                "histograms",
                "random_points",
                f"L{L}",
            ),
            matrix_tag="unitary_pqc_qfim",
            matrix_label="QFIM",
            num_layers=L,
            context_tag="random",
            context_label="random point",
            condition_tag="reduced0123",
            condition_label="reduced keep=(0,1,2,3)",
            color="C0",
        )
        upqc._save_qfim_eigs_violinplot_by_index(
            upqc.hs_eigs_reduced_by_layer[L],
            title=rf"HS tangent Gram eigenvalues at {upqc.NUM_QFIM_SAMPLES} random points (L={L})",
            outpath=os.path.join(hs_eigs_reduced_0123_dir, f"L{L}_reduced_0123.pdf"),
            rank_thresholds=upqc.hs_thresh_reduced_by_layer[L],
            ylabel="HS tangent Gram eigenvalue",
        )
        upqc.plot_style.save_eigenvalue_histograms_by_trial(
            upqc.hs_eigs_reduced_by_layer[L],
            outdir=os.path.join(
                hs_eigs_reduced_0123_dir,
                "histograms",
                "random_points",
                f"L{L}",
            ),
            matrix_tag="unitary_pqc_hs_gram",
            matrix_label="HS tangent Gram",
            num_layers=L,
            context_tag="random",
            context_label="random point",
            condition_tag="reduced0123",
            condition_label="reduced keep=(0,1,2,3)",
            color="C3",
        )
        upqc._save_qfim_eigs_violinplot_by_index(
            upqc.ortk_eigs_by_layer[L],
            title=(
                rf"Observable-Relevant Tangent Kernel eigenvalues at "
                rf"{upqc.NUM_QFIM_SAMPLES} random points (L={L})"
            ),
            outpath=os.path.join(ortk_eigs_dir, f"L{L}.pdf"),
            rank_thresholds=np.asarray(
                [upqc.ORTK_RANK_THRESHOLD],
                dtype=NP_REAL_DTYPE,
            ),
            ylabel="ORTK eigenvalue",
        )
        if upqc.hessian_eigs_by_layer.get(L) is not None:
            upqc._save_signed_eigs_scatterplot_by_index(
                upqc.hessian_eigs_by_layer[L],
                title=rf"Energy Hessian eigenvalues at {upqc.NUM_QFIM_SAMPLES} random points (L={L})",
                outpath=os.path.join(hessian_eigs_dir, f"L{L}.pdf"),
                rank_thresholds=upqc.hessian_thresh_by_layer[L],
                ylabel="Energy Hessian eigenvalue",
            )
            upqc.plot_style.save_eigenvalue_histograms_by_trial(
                upqc.hessian_eigs_by_layer[L],
                outdir=os.path.join(
                    hessian_eigs_dir,
                    "histograms",
                    "random_points",
                    f"L{L}",
                ),
                matrix_tag="unitary_pqc_energy_hessian",
                matrix_label="Energy Hessian",
                num_layers=L,
                context_tag="random",
                context_label="random point",
                color="C6",
            )
        if upqc.qfim_eigs_pure_by_layer[L] is not None:
            upqc._save_qfim_eigs_violinplot_by_index(
                upqc.qfim_eigs_pure_by_layer[L],
                title=rf"QFIM eigenvalues (Pure full-state) at {upqc.NUM_QFIM_SAMPLES} random points (L={L})",
                outpath=os.path.join(qfim_eigs_pure_dir, f"L{L}_pure_full.pdf"),
                rank_thresholds=upqc.qfim_thresh_pure_by_layer[L],
            )
            upqc.plot_style.save_eigenvalue_histograms_by_trial(
                upqc.qfim_eigs_pure_by_layer[L],
                outdir=os.path.join(
                    qfim_eigs_pure_dir,
                    "histograms",
                    "random_points",
                    f"L{L}",
                ),
                matrix_tag="unitary_pqc_qfim",
                matrix_label="QFIM",
                num_layers=L,
                context_tag="random",
                context_label="random point",
                condition_tag="pure_full",
                condition_label="pure full state",
                color="C0",
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
        os.path.join(qfim_rank_random_dir, "qfim_rank_violinplot_random_points.pdf"),
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
        os.path.join(hs_rank_random_dir, "hs_rank_violinplot_random_points_reduced_0123.pdf"),
        outside_legend=False,
    )

    upqc.plot_qfim_rank_max_by_layer(
        upqc.qfim_rank_pure_by_layer,
        upqc.layer_list,
        color="C0",
        title=rf"Maximum pure full-state QFIM rank at {upqc.NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum QFIM effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(qfim_rank_random_dir, "qfim_rank_max_random_points_pure_full.pdf"),
        marker="s",
        lw=1.0,
    )
    upqc.plot_qfim_rank_max_by_layer(
        upqc.qfim_rank_reduced_by_layer,
        upqc.layer_list,
        color="C0",
        title=rf"Maximum QFIM rank at {upqc.NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum QFIM effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(qfim_rank_random_dir, "qfim_rank_max_random_points_reduced_0123.pdf"),
        marker="o",
        lw=1.0,
    )
    upqc.plot_qfim_rank_max_by_layer(
        upqc.hs_rank_reduced_by_layer,
        upqc.layer_list,
        color="C3",
        title=rf"Maximum HS tangent Gram rank at {upqc.NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum HS effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(hs_rank_random_dir, "hs_rank_max_random_points_reduced_0123.pdf"),
        marker="D",
        lw=1.0,
    )

    upqc.plot_scalar_violin_by_layer(
        upqc.ortk_rank_by_layer,
        upqc.layer_list,
        title=(
            rf"Observable-Relevant Tangent Kernel rank at "
            rf"{upqc.NUM_QFIM_SAMPLES} random points"
        ),
        ylabel="ORTK rank",
        outpath=os.path.join(
            ortk_rank_random_dir,
            "ortk_rank_violinplot_random_points.pdf",
        ),
        integer_y_axis=True,
    )
    upqc.plot_scalar_violin_by_layer(
        upqc.ortk_effective_rank_by_layer,
        upqc.layer_list,
        title=(
            rf"Observable-Relevant Tangent Kernel participation effective "
            rf"rank at {upqc.NUM_QFIM_SAMPLES} random points"
        ),
        ylabel="ORTK participation effective rank",
        outpath=os.path.join(
            ortk_effective_rank_random_dir,
            "ortk_effective_rank_violinplot_random_points.pdf",
        ),
        integer_y_axis=False,
    )

    if upqc.hessian_rank_by_layer:
        upqc.new_prx_figure(width="double")
        ax = plt.gca()

        for idx, L in enumerate(upqc.layer_list):
            if upqc.hessian_rank_by_layer.get(L) is None:
                continue
            color = upqc.cmap(idx / num_layers)
            hessian_dataset = upqc._make_violin_ready(
                upqc.hessian_rank_by_layer[L],
                ensure_positive=False,
                tiny=1e-12,
            )
            vp_hessian = plt.violinplot(
                [hessian_dataset],
                positions=[float(L)],
                widths=violin_w_rank,
                showmeans=False,
                showmedians=True,
                showextrema=True,
            )
            upqc._style_violin(
                vp_hessian,
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
        ax.set_ylabel(r"Hessian rank $(|\eta_k| > 10^{-12})$")
        upqc.set_prx_title(
            rf"Energy Hessian rank at {upqc.NUM_QFIM_SAMPLES} random points",
            ax=ax,
        )
        ax.grid(True, axis="y", alpha=0.3)
        upqc.save_current_figure(
            os.path.join(hessian_rank_random_dir, "hessian_rank_violinplot_random_points.pdf"),
            outside_legend=False,
        )

        upqc.plot_qfim_rank_max_by_layer(
            upqc.hessian_rank_by_layer,
            upqc.layer_list,
            color="C6",
            title=rf"Maximum energy Hessian rank at {upqc.NUM_QFIM_SAMPLES} random points",
            ylabel=r"Maximum Hessian rank $(|\eta_k| > 10^{-12})$",
            outpath=os.path.join(hessian_rank_random_dir, "hessian_rank_max_random_points.pdf"),
            marker="P",
            lw=1.0,
        )

    spectral_random_summaries = (
        (upqc.qfim_eigs_reduced_by_layer, upqc.qfim_eigs_reduced_0123_dir,
         "Reduced QFIM", "qfim_reduced", False),
        (upqc.qfim_eigs_pure_by_layer, upqc.qfim_eigs_pure_dir,
         "Pure-state QFIM", "qfim_pure", False),
        (upqc.hs_eigs_reduced_by_layer, upqc.hs_eigs_reduced_0123_dir,
         "HS tangent Gram", "hs", False),
        (upqc.ortk_eigs_by_layer, upqc.ortk_eigs_dir,
         "ORTK", "ortk", False),
        (upqc.hessian_eigs_by_layer, upqc.hessian_eigs_dir,
         "Absolute Hessian", "hessian_abs", True),
    )
    for eigs, directory, label, tag, use_abs in spectral_random_summaries:
        if not eigs or not any(value is not None for value in eigs.values()):
            continue
        _plot_spectral_count_by_layer(
            eigs,
            title=f"{label} eigenvalue count at random points by threshold",
            ylabel=f"Mean {label} eigenvalue count",
            outpath=os.path.join(directory, f"{tag}_eigcount_threshold_overlay_random_points.pdf"),
            use_absolute_values=use_abs,
        )


def _load_optimization_path_results() -> None:
    qfim_rank_path = os.path.join(
        upqc.qfim_results_dir,
        "qfim_rank_history_optimization_path_reduced_0123.npz",
    )
    qfim_result = _load_required_result(qfim_rank_path)
    layers = [int(L) for L in np.asarray(qfim_result["layers"], dtype=NP_INT_DTYPE)]
    upqc.sample_iters = np.asarray(qfim_result["sample_iters"], dtype=NP_INT_DTYPE)
    upqc.layer_list = layers
    upqc.qfim_rank_history_by_layer = {
        L: np.asarray(qfim_result[f"L{L}_rank"], dtype=NP_REAL_DTYPE)
        for L in layers
    }
    upqc.qfim_eigs_history_by_layer = {
        L: np.asarray(qfim_result[f"L{L}_eigs"], dtype=NP_REAL_DTYPE)
        for L in layers
    }

    hs_rank_path = os.path.join(
        upqc.hs_results_dir,
        "hs_rank_history_optimization_path_reduced_0123.npz",
    )
    hs_result = _load_required_result(hs_rank_path)
    upqc.hs_rank_history_by_layer = {
        L: np.asarray(hs_result[f"L{L}_rank"], dtype=NP_REAL_DTYPE)
        for L in layers
    }
    upqc.hs_eigs_history_by_layer = {
        L: np.asarray(hs_result[f"L{L}_eigs"], dtype=NP_REAL_DTYPE)
        for L in layers
    }

    ortk_rank_result = _load_required_result(
        os.path.join(
            upqc.ortk_results_dir,
            "ortk_rank_history_optimization_path.npz",
        )
    )
    ortk_effective_rank_result = _load_required_result(
        os.path.join(
            upqc.ortk_results_dir,
            "ortk_effective_rank_history_optimization_path.npz",
        )
    )
    ortk_eigs_result = _load_required_result(
        os.path.join(
            upqc.ortk_results_dir,
            "ortk_eigs_history_optimization_path.npz",
        )
    )
    ortk_trace_result = _load_required_result(
        os.path.join(
            upqc.ortk_results_dir,
            "ortk_trace_history_optimization_path.npz",
        )
    )
    upqc.ortk_rank_history_by_layer = {
        L: np.asarray(ortk_rank_result[f"L{L}"], dtype=NP_REAL_DTYPE)
        for L in layers
    }
    upqc.ortk_effective_rank_history_by_layer = {
        L: np.asarray(
            ortk_effective_rank_result[f"L{L}"],
            dtype=NP_REAL_DTYPE,
        )
        for L in layers
    }
    upqc.ortk_eigs_history_by_layer = {
        L: np.asarray(ortk_eigs_result[f"L{L}"], dtype=NP_REAL_DTYPE)
        for L in layers
    }
    upqc.ortk_trace_history_by_layer = {
        L: np.asarray(ortk_trace_result[f"L{L}"], dtype=NP_REAL_DTYPE)
        for L in layers
    }

    hessian_rank_path = os.path.join(
        upqc.hessian_results_dir,
        "hessian_rank_history_optimization_path.npz",
    )
    upqc.hessian_rank_history_by_layer = {}
    upqc.hessian_eigs_history_by_layer = {}
    upqc.hessian_thresh_history_by_layer = {}

    if os.path.exists(hessian_rank_path):
        hessian_result = _load_required_result(hessian_rank_path)
        upqc.hessian_rank_history_by_layer = {
            L: np.asarray(hessian_result[f"L{L}_rank"], dtype=NP_REAL_DTYPE)
            for L in layers
        }
        upqc.hessian_eigs_history_by_layer = {
            L: np.asarray(hessian_result[f"L{L}_eigs"], dtype=NP_REAL_DTYPE)
            for L in layers
        }
        upqc.hessian_thresh_history_by_layer = {
            L: np.asarray(
                hessian_result[f"L{L}_rank_threshold"],
                dtype=NP_REAL_DTYPE,
            )
            for L in layers
        }


def _plot_optimization_path_results() -> None:
    def save_optimization_path_eigenvalue_histograms(
        eigs_history_by_layer: dict,
        *,
        outdir: str,
        matrix_tag: str,
        matrix_label: str,
        color: str,
        condition_tag: Optional[str] = None,
        condition_label: Optional[str] = None,
    ) -> None:
        target_iterations = np.asarray(upqc.sample_iters, dtype=NP_INT_DTYPE)

        for L in upqc.layer_list:
            eigs_L = eigs_history_by_layer.get(L)
            if eigs_L is None:
                continue

            eigs_L = np.asarray(eigs_L, dtype=NP_REAL_DTYPE)
            if eigs_L.ndim != 3:
                raise ValueError(
                    "Each optimization-path eigenvalue array must have shape "
                    "(num_runs, num_sample_iters, num_params)."
                )
            if (
                eigs_L.shape[1] != target_iterations.size
                and eigs_L.shape[0] == target_iterations.size
            ):
                eigs_L = np.transpose(eigs_L, (1, 0, 2))
            if eigs_L.shape[1] != target_iterations.size:
                raise ValueError(
                    f"Shape mismatch for L={L}: eigs_L.shape={eigs_L.shape}, "
                    f"len(sample_iters)={target_iterations.size}."
                )

            for time_idx, iteration in enumerate(target_iterations):
                upqc.plot_style.save_eigenvalue_histogram_across_trials(
                    eigs_L[:, time_idx, :],
                    outdir=os.path.join(outdir, "histograms", f"L{L}"),
                    matrix_tag=matrix_tag,
                    matrix_label=matrix_label,
                    num_layers=L,
                    context_tag="opt_path",
                    context_label="optimization path",
                    iteration=int(iteration),
                    condition_tag=condition_tag,
                    condition_label=condition_label,
                    color=color,
                )

    save_optimization_path_eigenvalue_histograms(
        upqc.qfim_eigs_history_by_layer,
        outdir=os.path.join(
            upqc.qfim_eigs_dir,
            "optimization_path_keep0123",
        ),
        matrix_tag="unitary_pqc_qfim",
        matrix_label="QFIM",
        color="C0",
        condition_tag="reduced0123",
        condition_label="reduced keep=(0,1,2,3)",
    )
    save_optimization_path_eigenvalue_histograms(
        upqc.hs_eigs_history_by_layer,
        outdir=os.path.join(
            upqc.hs_eigs_dir,
            "optimization_path_keep0123",
        ),
        matrix_tag="unitary_pqc_hs_gram",
        matrix_label="HS tangent Gram",
        color="C3",
        condition_tag="reduced0123",
        condition_label="reduced keep=(0,1,2,3)",
    )
    save_optimization_path_eigenvalue_histograms(
        upqc.ortk_eigs_history_by_layer,
        outdir=os.path.join(
            upqc.ortk_eigs_dir,
            "optimization_path",
        ),
        matrix_tag="unitary_pqc_ortk",
        matrix_label="Observable-Relevant Tangent Kernel",
        color="C2",
    )
    if upqc.hessian_eigs_history_by_layer:
        save_optimization_path_eigenvalue_histograms(
            upqc.hessian_eigs_history_by_layer,
            outdir=os.path.join(
                upqc.hessian_eigs_dir,
                "optimization_path",
            ),
            matrix_tag="unitary_pqc_energy_hessian",
            matrix_label="Energy Hessian",
            color="C6",
        )

    spectral_path_summaries = (
        (upqc.qfim_eigs_history_by_layer, upqc.qfim_eigs_dir, "QFIM", "qfim", False),
        (upqc.hs_eigs_history_by_layer, upqc.hs_eigs_dir, "HS tangent Gram", "hs", False),
        (upqc.ortk_eigs_history_by_layer, upqc.ortk_eigs_dir, "ORTK", "ortk", False),
        (upqc.hessian_eigs_history_by_layer, upqc.hessian_eigs_dir,
         "Absolute Hessian", "hessian_abs", True),
    )
    for eigs, directory, label, tag, use_abs in spectral_path_summaries:
        if not eigs:
            continue
        _plot_spectral_count_by_layer(
            eigs,
            title=f"{label} eigenvalue count along optimization path by threshold",
            ylabel=f"Mean {label} eigenvalue count",
            outpath=os.path.join(
                directory,
                f"{tag}_eigcount_threshold_overlay_optimization_path_by_layer.pdf",
            ),
            use_absolute_values=use_abs,
        )

    upqc.plot_qfim_rank_history_mean_by_layer(
        upqc.qfim_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Mean QFIM effective rank along optimization path (keep=(0,1,2,3))",
        outpath=os.path.join(
            upqc.qfim_rank_optimization_path_mean_dir,
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
            upqc.qfim_rank_optimization_path_min_dir,
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
            upqc.hs_rank_optimization_path_mean_dir,
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
            upqc.hs_rank_optimization_path_min_dir,
            "hs_rank_min_history_optimization_path_reduced_0123.pdf",
        ),
        ylabel=r"Minimum HS effective rank $(\lambda_k > 10^{-12})$",
        cmap=upqc.cmap,
    )

    upqc.plot_qfim_rank_history_mean_by_layer(
        upqc.ortk_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Mean Observable-Relevant Tangent Kernel rank along optimization path",
        outpath=os.path.join(
            upqc.ortk_rank_optimization_path_mean_dir,
            "ortk_rank_mean_history_optimization_path.pdf",
        ),
        ylabel="Mean ORTK rank",
        cmap=upqc.cmap,
    )
    upqc.plot_qfim_rank_history_min_by_layer(
        upqc.ortk_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Minimum Observable-Relevant Tangent Kernel rank along optimization path",
        outpath=os.path.join(
            upqc.ortk_rank_optimization_path_min_dir,
            "ortk_rank_min_history_optimization_path.pdf",
        ),
        ylabel="Minimum ORTK rank",
        cmap=upqc.cmap,
        integer_y_axis=True,
    )
    upqc.plot_qfim_rank_history_mean_by_layer(
        upqc.ortk_effective_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title=(
            "Mean Observable-Relevant Tangent Kernel participation effective "
            "rank along optimization path"
        ),
        outpath=os.path.join(
            upqc.ortk_effective_rank_optimization_path_mean_dir,
            "ortk_effective_rank_mean_history_optimization_path.pdf",
        ),
        ylabel="Mean ORTK participation effective rank",
        cmap=upqc.cmap,
    )
    upqc.plot_qfim_rank_history_min_by_layer(
        upqc.ortk_effective_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title=(
            "Minimum Observable-Relevant Tangent Kernel participation "
            "effective rank along optimization path"
        ),
        outpath=os.path.join(
            upqc.ortk_effective_rank_optimization_path_min_dir,
            "ortk_effective_rank_min_history_optimization_path.pdf",
        ),
        ylabel="Minimum ORTK participation effective rank",
        cmap=upqc.cmap,
        integer_y_axis=False,
    )
    upqc.plot_qfim_rank_history_mean_by_layer(
        upqc.ortk_trace_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Mean Observable-Relevant Tangent Kernel trace along optimization path",
        outpath=os.path.join(
            upqc.ortk_trace_optimization_path_dir,
            "ortk_trace_mean_history_optimization_path.pdf",
        ),
        ylabel="Mean ORTK trace",
        cmap=upqc.cmap,
    )
    upqc.plot_qfim_rank_history_min_by_layer(
        upqc.ortk_trace_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Minimum Observable-Relevant Tangent Kernel trace along optimization path",
        outpath=os.path.join(
            upqc.ortk_trace_optimization_path_dir,
            "ortk_trace_min_history_optimization_path.pdf",
        ),
        ylabel="Minimum ORTK trace",
        cmap=upqc.cmap,
        integer_y_axis=False,
    )

    if upqc.hessian_rank_history_by_layer:
        upqc.plot_qfim_rank_history_mean_by_layer(
            upqc.hessian_rank_history_by_layer,
            upqc.layer_list,
            upqc.sample_iters,
            title="Mean energy Hessian rank along optimization path",
            outpath=os.path.join(
                upqc.hessian_rank_optimization_path_mean_dir,
                "hessian_rank_mean_history_optimization_path.pdf",
            ),
            ylabel=r"Mean Hessian rank $(|\eta_k| > 10^{-12})$",
            cmap=upqc.cmap,
        )
        upqc.plot_qfim_rank_history_min_by_layer(
            upqc.hessian_rank_history_by_layer,
            upqc.layer_list,
            upqc.sample_iters,
            title="Minimum energy Hessian rank along optimization path",
            outpath=os.path.join(
                upqc.hessian_rank_optimization_path_min_dir,
                "hessian_rank_min_history_optimization_path.pdf",
            ),
            ylabel=r"Minimum Hessian rank $(|\eta_k| > 10^{-12})$",
            cmap=upqc.cmap,
        )

        hessian_path_dir = os.path.join(
            upqc.hessian_eigs_dir,
            "optimization_path",
        )
        os.makedirs(hessian_path_dir, exist_ok=True)

        target_iterations = np.asarray(upqc.sample_iters, dtype=NP_INT_DTYPE)

        for L in upqc.layer_list:
            eigs_L = upqc.hessian_eigs_history_by_layer.get(L)
            if eigs_L is None:
                continue

            eigs_L = np.asarray(eigs_L, dtype=NP_REAL_DTYPE)
            if eigs_L.ndim != 3:
                continue

            if (
                eigs_L.shape[1] != target_iterations.size
                and eigs_L.shape[0] == target_iterations.size
            ):
                eigs_L = np.transpose(eigs_L, (1, 0, 2))

            if eigs_L.shape[1] != target_iterations.size:
                continue

            layer_dir = os.path.join(hessian_path_dir, f"L{L}")
            os.makedirs(layer_dir, exist_ok=True)
            thresholds_L = upqc.hessian_thresh_history_by_layer.get(L)
            thresholds_L_arr = None
            if thresholds_L is not None:
                thresholds_L_arr = np.asarray(thresholds_L, dtype=NP_REAL_DTYPE)
                if (
                    thresholds_L_arr.ndim == 2
                    and thresholds_L_arr.shape[1] != target_iterations.size
                    and thresholds_L_arr.shape[0] == target_iterations.size
                ):
                    thresholds_L_arr = thresholds_L_arr.T
                if (
                    thresholds_L_arr.ndim != 2
                    or thresholds_L_arr.shape[1] != target_iterations.size
                ):
                    thresholds_L_arr = None

            for time_idx, iteration in enumerate(target_iterations):
                iteration_int = int(iteration)
                upqc._save_signed_eigs_scatterplot_by_index(
                    eigs_L[:, time_idx, :],
                    title=(
                        rf"Energy Hessian eigenvalues along optimization path "
                        rf"(L={L}, iteration={iteration_int})"
                    ),
                    outpath=os.path.join(
                        layer_dir,
                        f"iter{iteration_int:06d}.pdf",
                    ),
                    rank_thresholds=(
                        None
                        if thresholds_L_arr is None
                        else thresholds_L_arr[:, time_idx]
                    ),
                    ylabel="Energy Hessian eigenvalue",
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

            table = _load_required_result(data_path)
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
