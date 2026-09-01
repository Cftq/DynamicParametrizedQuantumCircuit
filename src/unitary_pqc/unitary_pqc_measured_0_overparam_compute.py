#!/usr/bin/env python
# coding: utf-8
"""Measurement-outcome-0 Unitary-PQC overparameterization pipeline.

This module was extracted from unitary_pqc_overparam.ipynb so the notebook can
call a Python function instead of carrying the full optimization/QFIM program
inline. Independent VQE trials and QFIM/HS/Hessian parameter points are
evaluated in configurable fixed-size JAX batches.  The command-line entry
point runs VQE and the post-VQE analyses in separate Python subprocesses so
the VQE JAX/XLA runtime is released before QFIM compilation begins.
Because the full circuit is closed and unitary, its numerical evolution uses a
32-amplitude statevector; reduced density matrices are formed only at subsystem
analysis boundaries.
Numerical plots are handled by
``unitary_pqc_measured_0_overparam_visualize.py``, and circuit drawings by
``unitary_pqc_measured_0_overparam_draw_circuits.py``.

Example::

    python unitary_pqc_measured_0_overparam_compute.py --h-param 0.1 --vqe-batch-size 20 --analysis-batch-size 1
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional

_MODULE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _MODULE_DIR.parent
_COMMON_DIR = _SRC_DIR / "common"
_PROJECT_ROOT = _SRC_DIR.parent
for _path in (_MODULE_DIR, _COMMON_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


import config_overparam as cfg


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _coerce_finite_h_param(value) -> float:
    """Return a finite Hamiltonian parameter without mutating the config."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("h_param must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError("h_param must be a finite number")
    return parsed


def _finite_float(value: str) -> float:
    try:
        return _coerce_finite_h_param(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _resolve_h_param(cli_value: Optional[float]) -> float:
    """Use an explicit CLI value, otherwise fall back to the config value."""
    value = cfg.H_PARAM if cli_value is None else cli_value
    return _coerce_finite_h_param(value)


def _parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the measurement-outcome-0 Unitary-PQC numerical pipeline. "
            "By default VQE and the "
            "post-VQE QFIM/HS/Hessian analyses run in separate subprocesses."
        )
    )
    parser.add_argument(
        "--h-param",
        type=_finite_float,
        default=None,
        help=(
            "Hamiltonian parameter h (default: H_PARAM from "
            "config_overparam.py)."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("all", "vqe", "qfim"),
        default="all",
        help=(
            "all: run VQE and then post-VQE analyses in separate processes; "
            "vqe/qfim: run only the selected stage"
        ),
    )
    parser.add_argument(
        "--vqe-batch-size",
        type=_positive_int,
        default=int(getattr(cfg, "VQE_BATCH_SIZE", 5)),
        help="Number of independent VQE trials evaluated by each vmap call.",
    )
    parser.add_argument(
        "--analysis-batch-size",
        type=_positive_int,
        default=int(getattr(cfg, "ANALYSIS_BATCH_SIZE", 5)),
        help=(
            "Number of independent parameter points evaluated by each "
            "QFIM/HS/Hessian vmap call."
        ),
    )
    args = parser.parse_args(argv)
    args.h_param = _resolve_h_param(args.h_param)
    return args


def _launch_stage_subprocess(stage: str, args: argparse.Namespace) -> int:
    """Run one numerical stage in a fresh interpreter and return its status."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--h-param",
        str(args.h_param),
    ]
    if stage == "vqe":
        command.extend(("--vqe-batch-size", str(args.vqe_batch_size)))
    elif stage == "qfim":
        command.extend(
            ("--analysis-batch-size", str(args.analysis_batch_size))
        )
    else:
        raise ValueError(f"Unsupported subprocess stage: {stage!r}")

    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def _run_split_cli_pipeline(args: argparse.Namespace) -> int:
    """Run VQE to completion, then start analyses in a clean process."""
    print(f"Selected Hamiltonian parameter h: {args.h_param}", flush=True)
    return_code = _launch_stage_subprocess("vqe", args)
    if return_code:
        return return_code
    return _launch_stage_subprocess("qfim", args)


if __name__ == "__main__":
    _CLI_ARGS = _parse_cli_args()
    if _CLI_ARGS.stage == "all":
        # Exit before importing the numerical dependencies below. Only the two child
        # processes own numerical runtimes, and the VQE child is gone before
        # the QFIM child starts.
        raise SystemExit(_run_split_cli_pipeline(_CLI_ARGS))
else:
    _CLI_ARGS = argparse.Namespace(
        h_param=_resolve_h_param(None),
        stage=None,
        vqe_batch_size=int(getattr(cfg, "VQE_BATCH_SIZE", 5)),
        analysis_batch_size=int(getattr(cfg, "ANALYSIS_BATCH_SIZE", 5)),
    )

VQE_BATCH_SIZE = int(_CLI_ARGS.vqe_batch_size)
if VQE_BATCH_SIZE <= 0:
    raise ValueError("VQE_BATCH_SIZE must be a positive integer.")

ANALYSIS_BATCH_SIZE = int(_CLI_ARGS.analysis_batch_size)
if ANALYSIS_BATCH_SIZE <= 0:
    raise ValueError("ANALYSIS_BATCH_SIZE must be a positive integer.")


# ------------------------------------------------------------
# IMPORTANT: env vars should be set BEFORE importing jax
# ------------------------------------------------------------
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import optax
from matplotlib.patches import Patch
from tqdm.auto import tqdm as _tqdm

import jax
import plot as plot_style
from hamiltonian import (
    PAULI,
    build_H_matrix_jax,
    hamiltonian_terms,
)
from qfim import (
    effective_rank_from_eigvals,
    hermitian as _hermitian,
    hermitian_eigvals_desc,
    make_hilbert_schmidt_metric_fn,
    make_mixed_state_qfim_fn,
    make_pure_state_qfim_fn,
    mask_psd_eigvals_for_rank,
    matrix_rank_psd,
    psd_eigvals_desc,
)

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def tqdm(*args, **kwargs):
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("dynamic_ncols", True)
    return _tqdm(*args, **kwargs)


def _release_jax_compilation_cache() -> None:
    """Release completed layer/metric executables before compiling the next."""
    gc.collect()
    jax.clear_caches()
    gc.collect()


jit = jax.jit

REAL_DTYPE = jnp.float64
COMPLEX_DTYPE = jnp.complex128
NP_REAL_DTYPE = np.float64
NP_COMPLEX_DTYPE = np.complex128
NP_INT_DTYPE = np.int64
MEASUREMENT_OUTCOME = 0
ANSATZ_NAME = "unitary_pqc_measured_0"


def build_layer_list(max_layer: int, dense_until_layer: int, sparse_step: int):
    """Build the dense-then-sparse layer schedule without simulator imports."""
    dense_end = min(int(dense_until_layer), int(max_layer))
    return list(range(1, dense_end + 1)) + list(
        range(dense_end + int(sparse_step), int(max_layer) + 1, int(sparse_step))
    )


def _unitary_pqc_save_dir(h_value: float) -> str:
    """Return this outcome branch's output directory independent of CWD."""
    return str(
        _PROJECT_ROOT
        / "figs"
        / ANSATZ_NAME
        / f"h_{h_value}"
    )

INCH_PER_CM = plot_style.INCH_PER_CM
FIGSIZE_SINGLE = plot_style.FIGSIZE_SINGLE
FIGSIZE_DOUBLE = plot_style.FIGSIZE_DOUBLE
FIGURE_WIDTH_DEFAULT = plot_style.FIGURE_WIDTH_DEFAULT

SAVE_DPI = plot_style.SAVE_DPI
SAVEFIG_PAD_INCHES = plot_style.SAVEFIG_PAD_INCHES
SAVE_PNG = plot_style.NUMERICAL_SAVE_PNG
SAVE_PDF = plot_style.NUMERICAL_SAVE_PDF
CIRCUIT_SAVE_PDF = plot_style.CIRCUIT_SAVE_PDF
SHOW_FIGURE_TITLES = plot_style.SHOW_FIGURE_TITLES
SHOW_REDUNDANT_LAYER_LEGENDS = plot_style.SHOW_REDUNDANT_LAYER_LEGENDS

BASE_FONT_SIZE = plot_style.BASE_FONT_SIZE
TITLE_FONT_SIZE = plot_style.TITLE_FONT_SIZE
AXIS_LABEL_FONT_SIZE = plot_style.AXIS_LABEL_FONT_SIZE
TICK_LABEL_FONT_SIZE = plot_style.TICK_LABEL_FONT_SIZE
LEGEND_FONT_SIZE = plot_style.LEGEND_FONT_SIZE

_DEFAULT_AXES_MARGINS_PRX = plot_style._DEFAULT_AXES_MARGINS_PRX
_DEFAULT_AXES_MARGINS_PRX_OUTSIDE_LEGEND = (
    plot_style._DEFAULT_AXES_MARGINS_PRX_OUTSIDE_LEGEND
)

num_system_qubits = 4
ANCILLA_QUBIT = 4
num_total_qubits = num_system_qubits + 1
SYSTEM_WIRES = tuple(range(num_system_qubits))
FULL_WIRES = tuple(range(num_total_qubits))
KEEP_WIRES_4 = SYSTEM_WIRES
KEEP_WIRES_5 = FULL_WIRES
QFIM_KEEP0123_KEY = "keep0123"
QFIM_KEEP01234_KEY = "keep01234"
QFIM_KEEP0123_LABEL = "Reduced state keep=(0,1,2,3)"
QFIM_KEEP01234_LABEL = "Pure full state keep=(0,1,2,3,4)"

h_param = _coerce_finite_h_param(_CLI_ARGS.h_param)
tolerance = cfg.TOLERANCE
steps = cfg.STEPS
num_runs = cfg.NUM_RUNS
lr = cfg.LEARNING_RATE

NUM_BLOCKS = 4
PARAMS_PER_BLOCK = 3
num_params_per_layer = NUM_BLOCKS * PARAMS_PER_BLOCK
LAYER_PAIRS = (
    (1, 3),
    (2, 3),
    (0, 2),
    (0, ANCILLA_QUBIT),
)

dense_until_layer = cfg.UNITARY_PQC_DENSE_UNTIL_LAYER
max_layer = cfg.UNITARY_PQC_MAX_LAYER
sparse_step = cfg.UNITARY_PQC_SPARSE_STEP
dense_end = min(dense_until_layer, max_layer)
layer_list = build_layer_list(max_layer, dense_until_layer, sparse_step)

qfim_dense_until_layer = cfg.UNITARY_PQC_QFIM_DENSE_UNTIL_LAYER
qfim_max_layer = cfg.UNITARY_PQC_QFIM_MAX_LAYER
qfim_sparse_step = cfg.UNITARY_PQC_QFIM_SPARSE_STEP
qfim_layer_list = build_layer_list(
    qfim_max_layer,
    qfim_dense_until_layer,
    qfim_sparse_step,
)

save_dir = _unitary_pqc_save_dir(h_param)
figures_dir = os.path.join(save_dir, "figures")
energy_fig_dir = os.path.join(figures_dir, "energy")
qfim_fig_dir = os.path.join(figures_dir, "qfim")
hs_fig_dir = os.path.join(figures_dir, "hs")
hessian_fig_dir = os.path.join(figures_dir, "hessian")
circuit_dir = os.path.join(save_dir, "optimized_circuits")
numerical_results_dir = os.path.join(save_dir, "numerical_results")
energy_results_dir = os.path.join(numerical_results_dir, "energy")
qfim_results_dir = os.path.join(numerical_results_dir, "qfim")
hs_results_dir = os.path.join(numerical_results_dir, "hs")
hessian_results_dir = os.path.join(numerical_results_dir, "hessian")

sample_every = cfg.SAMPLE_EVERY
sample_iters = np.asarray([], dtype=NP_INT_DTYPE)
sample_iter_set = set()

KEEP_WIRES = KEEP_WIRES_4
QFIM_EFFECTIVE_RANK_THRESHOLD = cfg.QFIM_EFFECTIVE_RANK_THRESHOLD
HESSIAN_RANDOM_SCHEMA_VERSION = 1
HESSIAN_RANK_DEFINITION = (
    "count(abs(eigenvalue) >= hessian_rank_threshold)"
)
HESSIAN_CONDITION_NUMBER_DEFINITION = (
    "max(abs(active eigenvalue)) / min(abs(active eigenvalue)); NaN if rank == 0"
)
EIG_SUM_EPS = cfg.EIG_SUM_EPS
QFIM_EIG_PLOT_EPS = cfg.QFIM_EIG_PLOT_EPS
NUM_QFIM_SAMPLES = cfg.NUM_QFIM_SAMPLES
QFIM_SAMPLE_SEED_BASE = cfg.UNITARY_PQC_QFIM_SAMPLE_SEED_BASE
PURE_QFIM_LAYER_THRESHOLD = cfg.PURE_QFIM_LAYER_THRESHOLD
RED_JVP_CHUNK = cfg.RED_JVP_CHUNK

key = None
H_terms = ()
PAULI = {}
H_matrix = None
smallest_eigval = None
X2 = None
_PSI_FULL_INIT = None
cmap = None

success_rates_history = {}
energy_mean_history = {}
energy_std_history = {}
final_stats = {}
theta_history = {}
best_theta_by_layer = {}
final_theta_wrapped_rmsdist_by_layer = {}
energy_traces_by_layer = {}
grad_norm_traces_by_layer = {}
theta_sample_traces_by_layer = {}

qfim_rank_pure_by_layer = {}
qfim_rank_reduced_by_layer = {}
qfim_random_thetas_by_layer = {}
qfim_eigs_pure_by_layer = {}
qfim_eigs_reduced_by_layer = {}
qfim_thresh_pure_by_layer = {}
qfim_thresh_reduced_by_layer = {}
qfim_rank_history_by_layer = {}
qfim_eigs_history_by_layer = {}
qfim_thresh_history_by_layer = {}
qfim_random_result_paths_by_keep = {}
qfim_optimization_path_result_paths_by_keep = {}
hs_rank_reduced_by_layer = {}
hs_eigs_reduced_by_layer = {}
hs_thresh_reduced_by_layer = {}
hs_rank_history_by_layer = {}
hs_eigs_history_by_layer = {}
hs_thresh_history_by_layer = {}
hessian_rank_by_layer = {}
hessian_condition_by_layer = {}

# Path attributes are retained for the separate visualizer's import API. This
# compute module does not create these directories or write QFIM eigenspectrum
# figures to them.
qfim_eigs_dir = os.path.join(qfim_fig_dir, "eigs")
qfim_eigs_pure_dir = os.path.join(qfim_eigs_dir, "pure_full")
qfim_eigs_reduced_0123_dir = os.path.join(qfim_eigs_dir, "reduced_keep_0123")
qfim_rank_dir = os.path.join(qfim_fig_dir, "rank")
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
hs_eigs_dir = os.path.join(hs_fig_dir, "eigs")
hs_eigs_reduced_0123_dir = os.path.join(hs_eigs_dir, "reduced_keep_0123")
hs_rank_dir = os.path.join(hs_fig_dir, "rank")
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
_figsize_from_width = plot_style._figsize_from_width
new_prx_figure = plot_style.new_prx_figure
set_prx_title = plot_style.set_prx_title
apply_axes_prx = plot_style.apply_axes_prx
apply_prx_axis_style = plot_style.apply_prx_axis_style
save_current_figure = plot_style.save_current_figure


def _ensure_unitary_result_dirs() -> None:
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(energy_fig_dir, exist_ok=True)
    os.makedirs(qfim_fig_dir, exist_ok=True)
    os.makedirs(hs_fig_dir, exist_ok=True)
    os.makedirs(hessian_fig_dir, exist_ok=True)
    os.makedirs(qfim_rank_dir, exist_ok=True)
    os.makedirs(qfim_rank_random_dir, exist_ok=True)
    os.makedirs(qfim_rank_optimization_path_dir, exist_ok=True)
    os.makedirs(qfim_rank_optimization_path_mean_dir, exist_ok=True)
    os.makedirs(qfim_rank_optimization_path_min_dir, exist_ok=True)
    os.makedirs(hs_eigs_dir, exist_ok=True)
    os.makedirs(hs_eigs_reduced_0123_dir, exist_ok=True)
    os.makedirs(hs_rank_dir, exist_ok=True)
    os.makedirs(hs_rank_random_dir, exist_ok=True)
    os.makedirs(hs_rank_optimization_path_dir, exist_ok=True)
    os.makedirs(hs_rank_optimization_path_mean_dir, exist_ok=True)
    os.makedirs(hs_rank_optimization_path_min_dir, exist_ok=True)
    os.makedirs(circuit_dir, exist_ok=True)
    os.makedirs(numerical_results_dir, exist_ok=True)
    os.makedirs(energy_results_dir, exist_ok=True)
    os.makedirs(qfim_results_dir, exist_ok=True)
    os.makedirs(hs_results_dir, exist_ok=True)
    os.makedirs(hessian_results_dir, exist_ok=True)


def save_npz_result(outpath: str, **arrays) -> None:
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    np.savez(outpath, **arrays)


def _validated_qfim_layers(
    layers,
    rank_by_layer: dict,
    eigs_by_layer: dict,
    threshold_by_layer: dict,
    *,
    expected_rank_ndim: int,
    theta_by_layer: Optional[dict] = None,
    context: str,
) -> list[int]:
    """Return layers with a complete, shape-consistent QFIM result set."""
    valid_layers = []
    for layer in layers:
        L = int(layer)
        values = (
            rank_by_layer.get(L),
            eigs_by_layer.get(L),
            threshold_by_layer.get(L),
        )
        present = tuple(value is not None for value in values)
        if not any(present):
            raise ValueError(f"Missing {context} QFIM results for L={L}.")
        if not all(present):
            raise ValueError(
                f"Incomplete {context} QFIM results for L={L}: "
                f"rank/eigs/threshold presence={present}."
            )

        ranks = np.asarray(values[0])
        eigs = np.asarray(values[1])
        thresholds = np.asarray(values[2])
        for array_name, array in (
            ("rank", ranks),
            ("eigs", eigs),
            ("threshold", thresholds),
        ):
            if (
                not np.issubdtype(array.dtype, np.number)
                or np.iscomplexobj(array)
            ):
                raise TypeError(
                    f"{context} {array_name} array for L={L} must be "
                    f"real numeric data, got dtype={array.dtype}."
                )
        if ranks.ndim != int(expected_rank_ndim):
            raise ValueError(
                f"{context} rank array for L={L} must be "
                f"{expected_rank_ndim}D, got {ranks.shape}."
            )
        if thresholds.shape != ranks.shape:
            raise ValueError(
                f"{context} rank/threshold shape mismatch for L={L}: "
                f"{ranks.shape} != {thresholds.shape}."
            )
        if eigs.ndim != ranks.ndim + 1 or eigs.shape[:-1] != ranks.shape:
            raise ValueError(
                f"{context} eigenspectrum shape mismatch for L={L}: "
                f"rank={ranks.shape}, eigs={eigs.shape}."
            )
        expected_num_params = num_params_per_layer * L
        if eigs.shape[-1] != expected_num_params:
            raise ValueError(
                f"{context} parameter-axis mismatch for L={L}: "
                f"expected {expected_num_params}, got {eigs.shape[-1]}."
            )

        if theta_by_layer is not None:
            theta = theta_by_layer.get(L)
            if theta is None:
                raise ValueError(f"Missing {context} theta samples for L={L}.")
            theta = np.asarray(theta)
            if (
                not np.issubdtype(theta.dtype, np.number)
                or np.iscomplexobj(theta)
            ):
                raise TypeError(
                    f"{context} theta array for L={L} must be real numeric "
                    f"data, got dtype={theta.dtype}."
                )
            if theta.ndim != 2 or theta.shape != (
                ranks.shape[0],
                eigs.shape[-1],
            ):
                raise ValueError(
                    f"{context} theta shape mismatch for L={L}: "
                    f"theta={theta.shape}, rank={ranks.shape}, eigs={eigs.shape}."
                )

        valid_layers.append(L)

    if not valid_layers:
        raise ValueError(f"No complete {context} QFIM results are available.")
    return valid_layers


def save_qfim_random_point_results_by_keep(
    *,
    layers,
    theta_by_layer: dict,
    rank_keep0123_by_layer: dict,
    eigs_keep0123_by_layer: dict,
    threshold_keep0123_by_layer: dict,
    rank_keep01234_by_layer: dict,
    eigs_keep01234_by_layer: dict,
    threshold_keep01234_by_layer: dict,
    analysis_batch_size: Optional[int] = None,
) -> dict[str, str]:
    """Save canonical DPQC-style random-point archives for both kept states."""
    specifications = (
        (
            QFIM_KEEP0123_KEY,
            KEEP_WIRES_4,
            QFIM_KEEP0123_LABEL,
            "reduced_mixed",
            rank_keep0123_by_layer,
            eigs_keep0123_by_layer,
            threshold_keep0123_by_layer,
        ),
        (
            QFIM_KEEP01234_KEY,
            KEEP_WIRES_5,
            QFIM_KEEP01234_LABEL,
            "pure_full",
            rank_keep01234_by_layer,
            eigs_keep01234_by_layer,
            threshold_keep01234_by_layer,
        ),
    )
    result_paths = {}
    for (
        keep_key,
        keep_wires,
        state_label,
        representation,
        rank_by_layer,
        eigs_by_layer,
        threshold_by_layer,
    ) in specifications:
        valid_layers = _validated_qfim_layers(
            layers,
            rank_by_layer,
            eigs_by_layer,
            threshold_by_layer,
            expected_rank_ndim=1,
            theta_by_layer=theta_by_layer,
            context=f"random-point {keep_key}",
        )
        for L in valid_layers:
            sample_count = np.asarray(rank_by_layer[L]).shape[0]
            if sample_count != int(NUM_QFIM_SAMPLES):
                raise ValueError(
                    f"random-point {keep_key} sample-count mismatch for L={L}: "
                    f"expected {NUM_QFIM_SAMPLES}, got {sample_count}."
                )
        outpath = os.path.join(
            qfim_results_dir,
            f"qfim_random_points_{keep_key}.npz",
        )
        arrays = {
            "schema_version": np.asarray(1, dtype=NP_INT_DTYPE),
            "ansatz": np.asarray(ANSATZ_NAME),
            "measurement_outcome": np.asarray(
                MEASUREMENT_OUTCOME,
                dtype=NP_INT_DTYPE,
            ),
            "analysis_kind": np.asarray("random_points"),
            "h_param": np.asarray(h_param, dtype=NP_REAL_DTYPE),
            "num_total_qubits": np.asarray(
                num_total_qubits,
                dtype=NP_INT_DTYPE,
            ),
            "num_params_per_layer": np.asarray(
                num_params_per_layer,
                dtype=NP_INT_DTYPE,
            ),
            "num_qfim_samples": np.asarray(
                NUM_QFIM_SAMPLES,
                dtype=NP_INT_DTYPE,
            ),
            "qfim_sample_seed_base": np.asarray(
                QFIM_SAMPLE_SEED_BASE,
                dtype=NP_INT_DTYPE,
            ),
            "qfim_effective_rank_threshold": np.asarray(
                QFIM_EFFECTIVE_RANK_THRESHOLD,
                dtype=NP_REAL_DTYPE,
            ),
            "red_jvp_chunk": np.asarray(RED_JVP_CHUNK, dtype=NP_INT_DTYPE),
            "analysis_batch_size": np.asarray(
                _resolve_analysis_batch_size(analysis_batch_size),
                dtype=NP_INT_DTYPE,
            ),
            "keep_key": np.asarray(keep_key),
            "keep_wires": np.asarray(keep_wires, dtype=NP_INT_DTYPE),
            "traced_wires": np.asarray(
                tuple(wire for wire in FULL_WIRES if wire not in keep_wires),
                dtype=NP_INT_DTYPE,
            ),
            "state_label": np.asarray(state_label),
            "representation": np.asarray(representation),
            "qfim_definition": np.asarray("SLD_QFIM"),
            "qfim_implementation": np.asarray(
                "mixed_state_sld_jvp"
                if keep_key == QFIM_KEEP0123_KEY
                else "pure_state_wavefunction"
            ),
            "layers": np.asarray(valid_layers, dtype=NP_INT_DTYPE),
            "eigenvalue_order": np.asarray("descending"),
            "eigenvalues_threshold_masked": np.asarray(False),
            "eig_sum_eps": np.asarray(EIG_SUM_EPS, dtype=NP_REAL_DTYPE),
        }
        for L in valid_layers:
            eigs = np.asarray(eigs_by_layer[L], dtype=NP_REAL_DTYPE)
            ranks = np.asarray(rank_by_layer[L])
            rank_dtype = (
                NP_INT_DTYPE
                if np.issubdtype(ranks.dtype, np.integer)
                else NP_REAL_DTYPE
            )
            arrays.update(
                {
                    f"L{L}_theta": np.asarray(
                        theta_by_layer[L],
                        dtype=NP_REAL_DTYPE,
                    ),
                    f"L{L}_rank": np.asarray(ranks, dtype=rank_dtype),
                    f"L{L}_threshold_rank": np.asarray(
                        ranks,
                        dtype=rank_dtype,
                    ),
                    f"L{L}_eigs_desc": eigs,
                    f"L{L}_rank_threshold": np.asarray(
                        threshold_by_layer[L],
                        dtype=NP_REAL_DTYPE,
                    ),
                    f"L{L}_trace": np.sum(eigs, axis=-1, dtype=NP_REAL_DTYPE),
                }
            )
        save_npz_result(outpath, **arrays)
        result_paths[keep_key] = outpath
    return result_paths


def save_qfim_optimization_path_results_by_keep(
    *,
    layers,
    sample_iterations,
    rank_keep0123_by_layer: dict,
    eigs_keep0123_by_layer: dict,
    threshold_keep0123_by_layer: dict,
    rank_keep01234_by_layer: dict,
    eigs_keep01234_by_layer: dict,
    threshold_keep01234_by_layer: dict,
    analysis_batch_size: Optional[int] = None,
) -> dict[str, dict[str, str]]:
    """Save rank, eigenspectrum, and trace histories for both kept states."""
    specifications = (
        (
            QFIM_KEEP0123_KEY,
            KEEP_WIRES_4,
            QFIM_KEEP0123_LABEL,
            "reduced_mixed",
            rank_keep0123_by_layer,
            eigs_keep0123_by_layer,
            threshold_keep0123_by_layer,
        ),
        (
            QFIM_KEEP01234_KEY,
            KEEP_WIRES_5,
            QFIM_KEEP01234_LABEL,
            "pure_full",
            rank_keep01234_by_layer,
            eigs_keep01234_by_layer,
            threshold_keep01234_by_layer,
        ),
    )
    sample_iterations = np.asarray(sample_iterations, dtype=NP_INT_DTYPE)
    if sample_iterations.ndim != 1 or sample_iterations.size == 0:
        raise ValueError("sample_iterations must be a non-empty 1D array.")
    plot_iterations = _qfim_history_plot_iterations(sample_iterations)
    result_paths = {}
    for (
        keep_key,
        keep_wires,
        state_label,
        representation,
        rank_by_layer,
        eigs_by_layer,
        threshold_by_layer,
    ) in specifications:
        valid_layers = _validated_qfim_layers(
            layers,
            rank_by_layer,
            eigs_by_layer,
            threshold_by_layer,
            expected_rank_ndim=2,
            context=f"optimization-path {keep_key}",
        )
        expected_num_runs = np.asarray(rank_by_layer[valid_layers[0]]).shape[0]
        for L in valid_layers:
            history_shape = np.asarray(rank_by_layer[L]).shape
            expected_shape = (expected_num_runs, sample_iterations.size)
            if history_shape != expected_shape:
                raise ValueError(
                    f"optimization-path {keep_key} history shape mismatch "
                    f"for L={L}: expected {expected_shape}, got {history_shape}."
                )
        metadata = {
            "schema_version": np.asarray(1, dtype=NP_INT_DTYPE),
            "ansatz": np.asarray(ANSATZ_NAME),
            "measurement_outcome": np.asarray(
                MEASUREMENT_OUTCOME,
                dtype=NP_INT_DTYPE,
            ),
            "analysis_kind": np.asarray("optimization_path"),
            "h_param": np.asarray(h_param, dtype=NP_REAL_DTYPE),
            "num_total_qubits": np.asarray(
                num_total_qubits,
                dtype=NP_INT_DTYPE,
            ),
            "num_params_per_layer": np.asarray(
                num_params_per_layer,
                dtype=NP_INT_DTYPE,
            ),
            "num_runs": np.asarray(
                expected_num_runs,
                dtype=NP_INT_DTYPE,
            ),
            "sample_iters": sample_iterations,
            "plot_iters": plot_iterations,
            "sample_semantics": np.asarray("pre_update_theta_t"),
            "layers": np.asarray(valid_layers, dtype=NP_INT_DTYPE),
            "keep_key": np.asarray(keep_key),
            "keep_wires": np.asarray(keep_wires, dtype=NP_INT_DTYPE),
            "traced_wires": np.asarray(
                tuple(wire for wire in FULL_WIRES if wire not in keep_wires),
                dtype=NP_INT_DTYPE,
            ),
            "state_label": np.asarray(state_label),
            "representation": np.asarray(representation),
            "qfim_definition": np.asarray("SLD_QFIM"),
            "analysis_batch_size": np.asarray(
                _resolve_analysis_batch_size(analysis_batch_size),
                dtype=NP_INT_DTYPE,
            ),
            "qfim_implementation": np.asarray(
                "mixed_state_sld_jvp"
                if keep_key == QFIM_KEEP0123_KEY
                else "pure_state_wavefunction"
            ),
            "qfim_effective_rank_threshold": np.asarray(
                QFIM_EFFECTIVE_RANK_THRESHOLD,
                dtype=NP_REAL_DTYPE,
            ),
            "eigenvalues_threshold_masked": np.asarray(False),
        }
        rank_path = os.path.join(
            qfim_results_dir,
            f"qfim_rank_history_optimization_path_{keep_key}.npz",
        )
        eigs_path = os.path.join(
            qfim_results_dir,
            f"qfim_eigs_history_optimization_path_{keep_key}.npz",
        )
        trace_path = os.path.join(
            qfim_results_dir,
            f"qfim_trace_history_optimization_path_{keep_key}.npz",
        )
        save_npz_result(
            rank_path,
            **metadata,
            **{
                f"L{L}": np.asarray(
                    rank_by_layer[L],
                    dtype=NP_REAL_DTYPE,
                )
                for L in valid_layers
            },
            **{
                f"L{L}_rank_threshold": np.asarray(
                    threshold_by_layer[L],
                    dtype=NP_REAL_DTYPE,
                )
                for L in valid_layers
            },
        )
        save_npz_result(
            eigs_path,
            **metadata,
            eigenvalue_order=np.asarray("descending"),
            **{
                f"L{L}": np.asarray(
                    eigs_by_layer[L],
                    dtype=NP_REAL_DTYPE,
                )
                for L in valid_layers
            },
        )
        save_npz_result(
            trace_path,
            **metadata,
            **{
                f"L{L}": np.sum(
                    np.asarray(eigs_by_layer[L], dtype=NP_REAL_DTYPE),
                    axis=-1,
                    dtype=NP_REAL_DTYPE,
                )
                for L in valid_layers
            },
        )
        result_paths[keep_key] = {
            "rank": rank_path,
            "eigs": eigs_path,
            "trace": trace_path,
        }
    return result_paths


def create_unitary_pqc(theta: jnp.ndarray, num_layers: int, num_qubits: int):
    """
    5-qubit circuit with one central ancilla.

    Qubits:
      - System qubits : 0,1,2,3
      - Center ancilla: 4

    One layer consists of four qg_layer blocks:
      1. (1,3)
      2. (2,3)
      3. (0,2)
      4. (0,4) = added system-ancilla block

    Each qg_layer(q0,q1) applies
        Rz(q0), Rz(q1), Rxx(q0,q1).
    Thus the added ancilla block applies single Rz gates on qubit 0
    and on the ancilla, followed by Rxx(0, ancilla).
    """
    num_qubits_int = int(num_qubits)
    if num_qubits_int != num_total_qubits:
        raise ValueError(
            f"This script is configured for {num_total_qubits} total qubits: "
            f"system={SYSTEM_WIRES}, ancilla={ANCILLA_QUBIT}."
        )

    # Numerical VQE/QFIM execution uses the direct JAX statevector path below.
    # Keep TensorCircuit optional so its unused PyTorch integrations are not
    # imported (and their CUDA DLLs are not loaded) during normal execution.
    try:
        # TensorCircuit eagerly imports optional ML/Qiskit integrations. They
        # are not used here and can load large CUDA DLLs before JAX starts.
        blocked_modules = ("tensorflow", "torch", "qiskit")
        missing = object()
        previous_modules = {
            name: sys.modules.get(name, missing) for name in blocked_modules
        }
        for name in blocked_modules:
            sys.modules[name] = None
        try:
            import tensorcircuit as tc
        finally:
            for name, previous in previous_modules.items():
                if previous is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
        tc.set_backend("jax")
        tc.set_dtype("complex128")
        qc = tc.Circuit(num_qubits_int)
    except (ImportError, OSError, MemoryError) as exc:
        raise RuntimeError(
            "Optional TensorCircuit circuit construction is unavailable. "
            "Use unitary_pqc_measured_0_overparam_draw_circuits.py for "
            "circuit drawings; numerical computation does not require "
            "TensorCircuit."
        ) from exc

    cursor = [0]

    def grab(n: int):
        seg = theta[cursor[0] : cursor[0] + n]
        cursor[0] += n
        return seg

    for _layer_idx in range(int(num_layers)):
        for q0, q1 in LAYER_PAIRS:
            p = grab(PARAMS_PER_BLOCK)
            qc.rz(int(q0), theta=p[0])
            qc.rz(int(q1), theta=p[1])
            qc.rxx(int(q0), int(q1), theta=p[2])

    return qc

@jit
def wrap_to_pi(x: jnp.ndarray) -> jnp.ndarray:
    two_pi = jnp.array(2.0 * jnp.pi, dtype=x.dtype)
    return (x + jnp.pi) % two_pi - jnp.pi


def U_rz(theta: jnp.ndarray) -> jnp.ndarray:
    """Return the complex128 single-qubit Rz unitary."""
    th = jnp.asarray(theta, dtype=REAL_DTYPE)
    return jnp.asarray(
        [
            [jnp.exp(-0.5j * th), 0.0],
            [0.0, jnp.exp(0.5j * th)],
        ],
        dtype=COMPLEX_DTYPE,
    )

def U_rxx(theta: jnp.ndarray) -> jnp.ndarray:
    th = jnp.asarray(theta, dtype=REAL_DTYPE)
    c = jnp.cos(0.5 * th).astype(COMPLEX_DTYPE)
    s = jnp.sin(0.5 * th).astype(COMPLEX_DTYPE)
    XX = jnp.kron(X2, X2)
    return c * jnp.eye(4, dtype=COMPLEX_DTYPE) - 1j * s * XX


def apply_unitary_on_statevector(
    statevector: jnp.ndarray,
    unitary: jnp.ndarray,
    wires,
    num_qubits: int,
) -> jnp.ndarray:
    """Apply a local unitary directly to a statevector."""
    wires = tuple(int(wire) for wire in wires)
    num_qubits = int(num_qubits)
    num_target_wires = len(wires)
    if unitary.shape != (2**num_target_wires, 2**num_target_wires):
        raise ValueError(
            "unitary shape does not match the number of target wires: "
            f"shape={unitary.shape}, wires={wires}."
        )
    if len(set(wires)) != num_target_wires or any(
        wire < 0 or wire >= num_qubits for wire in wires
    ):
        raise ValueError(
            f"wires must be unique indices in [0, {num_qubits}), got {wires}."
        )

    other_wires = [wire for wire in range(num_qubits) if wire not in wires]
    permutation = list(wires) + other_wires
    inverse_permutation = [0] * num_qubits
    for position, axis in enumerate(permutation):
        inverse_permutation[axis] = position

    state_tensor = jnp.reshape(statevector, (2,) * num_qubits)
    state_permuted = jnp.transpose(state_tensor, permutation)
    target_dimension = 2**num_target_wires
    other_dimension = 2 ** (num_qubits - num_target_wires)
    state_permuted = jnp.reshape(
        state_permuted,
        (target_dimension, other_dimension),
    )
    state_updated = unitary @ state_permuted
    state_updated = jnp.reshape(state_updated, (2,) * num_qubits)
    state_updated = jnp.transpose(state_updated, inverse_permutation)
    return jnp.reshape(state_updated, (2**num_qubits,))


def apply_unitary_on_rho(rho: jnp.ndarray, U: jnp.ndarray, wires, k: int) -> jnp.ndarray:
    """Apply a local unitary to a density matrix (compatibility helper)."""
    wires = tuple(int(w) for w in wires)
    m = len(wires)
    assert U.shape == (2**m, 2**m)

    others = [i for i in range(k) if i not in wires]
    perm_ket = list(wires) + others
    perm_bra = [w + k for w in wires] + [o + k for o in others]
    perm = perm_ket + perm_bra

    inv_perm = [0] * (2 * k)
    for i, a in enumerate(perm):
        inv_perm[a] = i

    rho_t = jnp.reshape(rho, (2,) * (2 * k))
    rho_p = jnp.transpose(rho_t, perm)

    dk = 2**m
    dr = 2 ** (k - m)
    rho_p = jnp.reshape(rho_p, (dk, dr, dk, dr))

    # ket-side: U @ rho
    rho1 = jnp.einsum("ij,jrks->irks", U, rho_p)
    # bra-side: rho @ U^\dagger
    rho2 = jnp.einsum("irps,bp->irbs", rho1, jnp.conjugate(U))

    rho2 = jnp.reshape(
        rho2,
        (2,) * m + (2,) * (k - m) + (2,) * m + (2,) * (k - m),
    )
    rho2 = jnp.transpose(rho2, inv_perm)
    return jnp.reshape(rho2, (2**k, 2**k))

def partial_trace_keep(
    rho: jnp.ndarray,
    *,
    keep_wires=SYSTEM_WIRES,
    num_qubits: int = num_total_qubits,
) -> jnp.ndarray:
    """
    Trace out all qubits except keep_wires.

    For keep_wires=(0,1,2,3) and num_qubits=5, this traces out
    the center ancilla qubit 4 and returns a 16x16 reduced density matrix.
    """
    keep_wires = tuple(int(w) for w in keep_wires)
    trace_wires = tuple(i for i in range(int(num_qubits)) if i not in keep_wires)

    num_keep = len(keep_wires)
    num_trace = len(trace_wires)

    perm_ket = list(keep_wires) + list(trace_wires)
    perm_bra = [w + num_qubits for w in keep_wires] + [w + num_qubits for w in trace_wires]
    perm = perm_ket + perm_bra

    rho_t = jnp.reshape(rho, (2,) * (2 * num_qubits))
    rho_p = jnp.transpose(rho_t, perm)

    dk = 2**num_keep
    dt = 2**num_trace
    rho_p = jnp.reshape(rho_p, (dk, dt, dk, dt))

    rho_keep = jnp.trace(rho_p, axis1=1, axis2=3)
    return jnp.reshape(rho_keep, (dk, dk))


def density_matrix_from_statevector(statevector: jnp.ndarray) -> jnp.ndarray:
    """Construct ``|psi><psi|`` only when a full density matrix is required."""
    statevector = jnp.asarray(statevector, dtype=COMPLEX_DTYPE).reshape((-1,))
    return jnp.outer(statevector, jnp.conjugate(statevector))


def reduced_density_matrix_from_statevector(
    statevector: jnp.ndarray,
    *,
    keep_wires=SYSTEM_WIRES,
    num_qubits: int = num_total_qubits,
) -> jnp.ndarray:
    """Return a reduced density matrix without constructing the full density."""
    keep_wires = tuple(int(wire) for wire in keep_wires)
    num_qubits = int(num_qubits)
    if len(set(keep_wires)) != len(keep_wires) or any(
        wire < 0 or wire >= num_qubits for wire in keep_wires
    ):
        raise ValueError(
            "keep_wires must be unique valid qubit indices, "
            f"got keep_wires={keep_wires}, num_qubits={num_qubits}."
        )

    trace_wires = tuple(
        wire for wire in range(num_qubits) if wire not in keep_wires
    )
    permutation = list(keep_wires) + list(trace_wires)
    state_tensor = jnp.reshape(statevector, (2,) * num_qubits)
    state_permuted = jnp.transpose(state_tensor, permutation)
    keep_dimension = 2 ** len(keep_wires)
    trace_dimension = 2 ** len(trace_wires)
    state_matrix = jnp.reshape(
        state_permuted,
        (keep_dimension, trace_dimension),
    )
    rho_keep = state_matrix @ jnp.conjugate(state_matrix.T)
    return _hermitian(rho_keep)


def statevector_sequential_unitary_pqc(
    theta: jnp.ndarray,
    num_layers: int,
) -> jnp.ndarray:
    """Propagate the closed five-qubit circuit as a 32-component statevector."""
    num_layers = int(num_layers)
    theta = jnp.asarray(theta, dtype=REAL_DTYPE)
    theta_layers = jnp.reshape(theta, (num_layers, num_params_per_layer))

    def one_layer(statevector: jnp.ndarray, layer_theta: jnp.ndarray):
        blocks = jnp.reshape(
            layer_theta,
            (NUM_BLOCKS, PARAMS_PER_BLOCK),
        )
        for block_index, (q0, q1) in enumerate(LAYER_PAIRS):
            parameters = blocks[block_index]
            statevector = apply_unitary_on_statevector(
                statevector,
                U_rz(parameters[0]),
                (q0,),
                num_total_qubits,
            )
            statevector = apply_unitary_on_statevector(
                statevector,
                U_rz(parameters[1]),
                (q1,),
                num_total_qubits,
            )
            statevector = apply_unitary_on_statevector(
                statevector,
                U_rxx(parameters[2]),
                (q0, q1),
                num_total_qubits,
            )
        return statevector, None

    statevector_final, _ = jax.lax.scan(
        one_layer,
        _PSI_FULL_INIT,
        theta_layers,
    )
    return jnp.asarray(statevector_final, dtype=COMPLEX_DTYPE)


def rho_full_sequential_unitary_pqc(theta: jnp.ndarray, num_layers: int) -> jnp.ndarray:
    """
    Return the full 5-qubit density matrix through a statevector outer product.

    The circuit itself is propagated only as a 32-component statevector.  This
    compatibility function constructs the 32x32 density matrix at the end.
    """
    statevector = statevector_sequential_unitary_pqc(
        theta,
        num_layers=num_layers,
    )
    return density_matrix_from_statevector(statevector)

def rho_keep_sequential_unitary_pqc(
    theta: jnp.ndarray,
    num_layers: int,
    *,
    keep_wires=SYSTEM_WIRES,
) -> jnp.ndarray:
    """
    Reduced density matrix on keep_wires.

    Default keep_wires=(0,1,2,3), so the added center ancilla is traced out.
    This is used for the reduced mixed-state QFIM.
    """
    statevector = statevector_sequential_unitary_pqc(
        theta,
        num_layers=num_layers,
    )
    return reduced_density_matrix_from_statevector(
        statevector,
        keep_wires=keep_wires,
        num_qubits=num_total_qubits,
    )

@jit
def _energy_from_rho_full_jit(
    rho_full: jnp.ndarray,
    hamiltonian_matrix: jnp.ndarray,
) -> jnp.ndarray:
    e = jnp.einsum("ij,ji->", rho_full, hamiltonian_matrix)
    return jnp.real(e)


def energy_from_rho_full(
    rho_full: jnp.ndarray,
    hamiltonian_matrix: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Evaluate ``Tr[rho_full H]`` using the current or supplied Hamiltonian."""
    if hamiltonian_matrix is None:
        if H_matrix is None:
            raise RuntimeError(
                "configure_unitary_pqc_overparam() must be called before "
                "energy_from_rho_full()."
            )
        hamiltonian_matrix = H_matrix
    return _energy_from_rho_full_jit(rho_full, hamiltonian_matrix)


@jit
def _energy_from_statevector_jit(
    statevector: jnp.ndarray,
    hamiltonian_matrix: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.real(
        jnp.vdot(statevector, hamiltonian_matrix @ statevector)
    )


def energy_from_statevector(
    statevector: jnp.ndarray,
    hamiltonian_matrix: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Evaluate ``<psi|H|psi>`` without constructing a density matrix."""
    if hamiltonian_matrix is None:
        if H_matrix is None:
            raise RuntimeError(
                "configure_unitary_pqc_overparam() must be called before "
                "energy_from_statevector()."
            )
        hamiltonian_matrix = H_matrix
    return _energy_from_statevector_jit(statevector, hamiltonian_matrix)


def make_energy_fn_for_layer(num_layers: int):
    def energy_fn(theta: jnp.ndarray) -> jnp.ndarray:
        statevector = statevector_sequential_unitary_pqc(
            theta,
            num_layers=num_layers,
        )
        return energy_from_statevector(statevector, H_matrix)

    return energy_fn

_make_violin_ready = plot_style.make_violin_ready
_style_violin = plot_style.style_violin

threshold_psd_eigvals_for_rank = mask_psd_eigvals_for_rank
_matrix_rank_psd = matrix_rank_psd

def make_pure_qfim_matrix_fn_for_layer(num_layers: int):
    def psi_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return statevector_sequential_unitary_pqc(
            theta,
            num_layers=num_layers,
        )

    return make_pure_state_qfim_fn(psi_fn)

def make_reduced_qfim_matrix_fn_for_layer_sequential(
    num_layers: int,
    *,
    keep_wires=KEEP_WIRES,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    keep_wires = tuple(int(w) for w in keep_wires)
    if keep_wires != KEEP_WIRES:
        raise NotImplementedError("Only keep_wires=(0,1,2,3) is supported here.")

    @jit
    def rho_sub_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return rho_keep_sequential_unitary_pqc(
            theta,
            num_layers=num_layers,
            keep_wires=keep_wires,
        )

    return make_mixed_state_qfim_fn(
        rho_sub_fn,
        eig_sum_eps=EIG_SUM_EPS,
        jvp_chunk=jvp_chunk,
    )


def make_reduced_hs_matrix_fn_for_layer_sequential(
    num_layers: int,
    *,
    keep_wires=KEEP_WIRES,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    @jax.jit
    def rho_keep_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return _hermitian(
            rho_keep_sequential_unitary_pqc(
                theta,
                num_layers=num_layers,
                keep_wires=keep_wires,
            )
        )

    return make_hilbert_schmidt_metric_fn(
        rho_keep_fn,
        jvp_chunk=jvp_chunk,
    )


def make_pure_full_hs_matrix_fn_for_layer(
    num_layers: int,
    *,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    """HS tangent Gram matrix derived from the full pure-state QFIM.

    For a normalized pure state, ``F_Q = 2 G_HS``.  Deriving the HS matrix
    from the statevector QFIM avoids differentiating a 32x32 density matrix.
    ``jvp_chunk`` remains in the signature for call compatibility.
    """
    del jvp_chunk
    pure_qfim_fn = make_pure_qfim_matrix_fn_for_layer(num_layers=num_layers)

    @jax.jit
    def pure_hs(theta: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * pure_qfim_fn(theta)

    return pure_hs

def make_reduced_qfim_rank_fn_for_layer(
    num_layers: int,
    keep_wires=KEEP_WIRES,
    reverse_axes: bool = False,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    qfim_fn = make_reduced_qfim_matrix_fn_for_layer_sequential(
        num_layers=num_layers,
        keep_wires=keep_wires,
        jvp_chunk=jvp_chunk,
    )

    def rank_reduced(theta: jnp.ndarray) -> jnp.ndarray:
        F = qfim_fn(theta)
        return _matrix_rank_psd(F)

    return rank_reduced

def _save_qfim_eigs_violinplot_by_index(
    eigs_sorted_desc: np.ndarray,
    *,
    title: str,
    outpath: str,
    rank_thresholds: Optional[np.ndarray] = None,
    eps: float = QFIM_EIG_PLOT_EPS,
    ylabel: str = "QFIM eigenvalue",
) -> None:
    """
    Eigenvalue distribution plot:
      - x-axis: eigenvalue index (sorted descending per sample)
      - y-axis: eigenvalue magnitude (log scale)
      - violin plots aggregated across random samples for a fixed layer

    IMPORTANT:
      - Input eigs_sorted_desc is expected to ALREADY be threshold-zeroed:
          eig[i] == 0.0 for entries deemed "zero" by the SAME rank rule.
      - For log plotting only, zeros are mapped to eps (visual-only).
        Stored eigenvalues are NOT modified.
      - rank_thresholds, if provided, should contain the fixed threshold
        computed by threshold_psd_eigvals_for_rank. Unique positive thresholds
        are drawn as red solid horizontal lines.
    """
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    eigs = np.asarray(eigs_sorted_desc, dtype=NP_REAL_DTYPE)  # keep stored values (includes exact zeros)
    eigs_plot = eigs.copy()

    mask = eigs_plot <= 0.0
    eigs_plot[mask] = eps

    num_params = int(eigs_plot.shape[1])
    datasets = [
        _make_violin_ready(eigs_plot[:, i], ensure_positive=True, tiny=eps)
        for i in range(num_params)
    ]

    new_prx_figure(width="double")
    ax = plt.gca()

    vp = plt.violinplot(
        datasets,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    _style_violin(vp, alpha=0.20, linewidth=1.0, linecolor="black", linealpha=0.7)

    # Draw the fixed rank threshold as a red solid line.
    if rank_thresholds is not None:
        thr = np.asarray(rank_thresholds, dtype=NP_REAL_DTYPE).ravel()
        thr = thr[np.isfinite(thr) & (thr > 0.0)]
        for t in np.unique(thr):
            ax.axhline(
                t,
                linestyle="-",
                linewidth=1.2,
                color="red",
                alpha=0.75,
                zorder=4,
            )

    step = max(1, num_params // 10)
    ticks = np.arange(1, num_params + 1, step, dtype=int)
    if ticks.size == 0 or ticks[0] != 1:
        ticks = np.concatenate([[1], ticks])
    if ticks[-1] != num_params:
        ticks = np.concatenate([ticks, [num_params]])
    plt.xticks(ticks=ticks, labels=[str(t) for t in ticks])

    plt.yscale("log")
    plt.xlabel("Eigenvalue index")
    plt.ylabel(ylabel)
    set_prx_title(title)
    plt.grid(True, which="both", alpha=0.3)

    save_current_figure(outpath, outside_legend=False)

def _qfim_rank_max_xy(rank_by_layer: dict, layers):
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
            [],
        )

    valid_layers = [L for L, _ in valid_items]
    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)
    max_ranks = np.asarray(
        [np.max(ranks_arr) for _, ranks_arr in valid_items],
        dtype=NP_REAL_DTYPE,
    )

    return x, max_ranks, valid_layers

def plot_qfim_rank_max_by_layer(
    rank_by_layer: dict,
    layers,
    *,
    color,
    title: str,
    ylabel: str,
    outpath: str,
    marker: str = "s",
    lw: float = 1.0,
):
    x, max_ranks, valid_layers = _qfim_rank_max_xy(rank_by_layer, layers)

    if x.size == 0:
        return

    new_prx_figure(width="double")
    ax = plt.gca()

    ax.plot(
        x,
        max_ranks,
        marker=marker,
        linestyle="--",
        linewidth=lw,
        markersize=4.0,
        color=color,
    )

    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    set_prx_title(title, ax=ax)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in valid_layers])
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)

    save_current_figure(outpath, outside_legend=False)


def make_qfim_eigvals_fn_for_layer(
    num_layers: int,
    *,
    keep_wires=KEEP_WIRES,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    qfim_fn = make_reduced_qfim_matrix_fn_for_layer_sequential(
        num_layers=num_layers,
        keep_wires=keep_wires,
        jvp_chunk=jvp_chunk,
    )

    @jit
    def qfim_eigvals(theta: jnp.ndarray):
        F = qfim_fn(theta)
        return psd_eigvals_desc(F)

    return qfim_eigvals

def make_qfim_rank_fn_for_layer(
    num_layers: int,
    *,
    keep_wires=KEEP_WIRES,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    qfim_eigvals_fn = make_qfim_eigvals_fn_for_layer(
        num_layers=num_layers,
        keep_wires=keep_wires,
        jvp_chunk=jvp_chunk,
    )

    @jit
    def qfim_rank(theta: jnp.ndarray):
        return effective_rank_from_eigvals(qfim_eigvals_fn(theta))

    return qfim_rank

def make_hs_eigvals_fn_for_layer(
    num_layers: int,
    *,
    keep_wires=KEEP_WIRES,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    hs_fn = make_reduced_hs_matrix_fn_for_layer_sequential(
        num_layers=num_layers,
        keep_wires=keep_wires,
        jvp_chunk=jvp_chunk,
    )

    @jit
    def hs_eigvals(theta: jnp.ndarray):
        G = hs_fn(theta)
        return psd_eigvals_desc(G)

    return hs_eigvals

def make_hs_rank_fn_for_layer(
    num_layers: int,
    *,
    keep_wires=KEEP_WIRES,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    hs_eigvals_fn = make_hs_eigvals_fn_for_layer(
        num_layers=num_layers,
        keep_wires=keep_wires,
        jvp_chunk=jvp_chunk,
    )

    @jit
    def hs_rank(theta: jnp.ndarray):
        return effective_rank_from_eigvals(hs_eigvals_fn(theta))

    return hs_rank


def make_energy_hessian_eigvals_fn_for_layer(num_layers: int):
    energy_fn = make_energy_fn_for_layer(num_layers)
    hessian_fn = jax.jit(jax.hessian(energy_fn))

    @jit
    def hessian_eigvals(theta: jnp.ndarray):
        return hermitian_eigvals_desc(hessian_fn(theta))

    return hessian_eigvals


def _resolve_analysis_batch_size(batch_size: Optional[int]) -> int:
    """Resolve and validate the fixed batch size used by matrix analyses."""
    resolved = ANALYSIS_BATCH_SIZE if batch_size is None else int(batch_size)
    if resolved <= 0:
        raise ValueError("analysis_batch_size must be a positive integer.")
    return resolved


def _pad_analysis_theta_batch(
    theta_batch: jnp.ndarray,
    batch_size: int,
) -> tuple[jnp.ndarray, int]:
    """Pad a final analysis batch so every JIT call has the same shape."""
    batch_size = _resolve_analysis_batch_size(batch_size)
    theta_batch = jnp.asarray(theta_batch, dtype=REAL_DTYPE)
    if theta_batch.ndim != 2:
        raise ValueError("theta_batch must have shape (batch, num_params).")

    valid_count = int(theta_batch.shape[0])
    if not 1 <= valid_count <= batch_size:
        raise ValueError(
            f"Expected between 1 and {batch_size} parameter points, "
            f"got {valid_count}."
        )
    if valid_count == batch_size:
        return theta_batch, valid_count

    padding = jnp.repeat(
        theta_batch[-1:, :],
        repeats=batch_size - valid_count,
        axis=0,
    )
    return jnp.concatenate((theta_batch, padding), axis=0), valid_count


def _evaluate_analysis_in_batches(
    theta_points,
    batched_fn,
    *,
    batch_size: Optional[int] = None,
    description: str,
) -> tuple[np.ndarray, ...]:
    """Evaluate a tuple-valued ``jit(vmap(...))`` function in fixed batches."""
    effective_batch_size = _resolve_analysis_batch_size(batch_size)
    theta_points = np.asarray(theta_points, dtype=NP_REAL_DTYPE)
    if theta_points.ndim != 2:
        raise ValueError("theta_points must have shape (num_points, num_params).")
    num_points = int(theta_points.shape[0])
    if num_points <= 0:
        raise ValueError("theta_points must contain at least one point.")

    output_parts = None
    batch_starts = range(0, num_points, effective_batch_size)
    for batch_start in tqdm(
        batch_starts,
        total=(num_points + effective_batch_size - 1)
        // effective_batch_size,
        desc=description,
        unit="batch",
        leave=False,
    ):
        batch_end = min(batch_start + effective_batch_size, num_points)
        theta_batch, valid_count = _pad_analysis_theta_batch(
            jnp.asarray(
                theta_points[batch_start:batch_end],
                dtype=REAL_DTYPE,
            ),
            effective_batch_size,
        )
        host_outputs = jax.device_get(batched_fn(theta_batch))
        if not isinstance(host_outputs, tuple):
            raise TypeError("An analysis batch runner must return a tuple.")
        if output_parts is None:
            output_parts = tuple([] for _ in host_outputs)
        if len(host_outputs) != len(output_parts):
            raise AssertionError("Analysis batch output structure changed.")
        for parts, values in zip(output_parts, host_outputs):
            parts.append(np.asarray(values[:valid_count]))

    return tuple(np.concatenate(parts, axis=0) for parts in output_parts)


def _make_psd_analysis_batch_runner(matrix_fn):
    """Batch PSD-matrix rank, masked/raw spectrum, and threshold metrics."""

    def metrics_one(theta: jnp.ndarray):
        eigs_desc = psd_eigvals_desc(matrix_fn(theta))
        masked_desc, threshold = threshold_psd_eigvals_for_rank(eigs_desc)
        rank_value = jnp.sum(eigs_desc > threshold)
        return rank_value, masked_desc, eigs_desc, threshold

    return jax.jit(jax.vmap(metrics_one))


def make_qfim_analysis_batch_runner(
    num_layers: int,
    *,
    keep_wires=KEEP_WIRES,
    jvp_chunk: int = RED_JVP_CHUNK,
    representation: str = "reduced",
):
    """Create a fixed-shape QFIM ``jit(vmap(...))`` metrics runner."""
    if representation == "pure_full":
        matrix_fn = make_pure_qfim_matrix_fn_for_layer(int(num_layers))
    elif representation == "reduced":
        matrix_fn = make_reduced_qfim_matrix_fn_for_layer_sequential(
            num_layers=int(num_layers),
            keep_wires=keep_wires,
            jvp_chunk=jvp_chunk,
        )
    else:
        raise ValueError("representation must be 'pure_full' or 'reduced'.")
    return _make_psd_analysis_batch_runner(matrix_fn)


def make_hs_analysis_batch_runner(
    num_layers: int,
    *,
    keep_wires=KEEP_WIRES,
    jvp_chunk: int = RED_JVP_CHUNK,
    representation: str = "reduced",
):
    """Create a fixed-shape HS ``jit(vmap(...))`` metrics runner."""
    if representation == "pure_full":
        matrix_fn = make_pure_full_hs_matrix_fn_for_layer(
            num_layers=int(num_layers),
            jvp_chunk=jvp_chunk,
        )
    elif representation == "reduced":
        matrix_fn = make_reduced_hs_matrix_fn_for_layer_sequential(
            num_layers=int(num_layers),
            keep_wires=keep_wires,
            jvp_chunk=jvp_chunk,
        )
    else:
        raise ValueError("representation must be 'pure_full' or 'reduced'.")
    return _make_psd_analysis_batch_runner(matrix_fn)


def hessian_rank_and_condition_from_eigvals(eigenvalues: jnp.ndarray):
    """Reduce a signed Hessian spectrum to the two requested metrics."""
    abs_eigs = jnp.abs(jnp.asarray(eigenvalues, dtype=REAL_DTYPE))
    if abs_eigs.ndim != 1:
        raise ValueError("Hessian eigenvalues must be one-dimensional.")
    threshold = jnp.asarray(
        QFIM_EFFECTIVE_RANK_THRESHOLD,
        dtype=abs_eigs.dtype,
    )
    active = abs_eigs >= threshold
    rank_value = jnp.sum(active, dtype=jnp.int64)
    max_active = jnp.max(jnp.where(active, abs_eigs, 0.0))
    min_active = jnp.min(jnp.where(active, abs_eigs, jnp.inf))
    condition_number = jnp.where(
        rank_value > 0,
        max_active / min_active,
        jnp.asarray(jnp.nan, dtype=abs_eigs.dtype),
    )
    return rank_value, condition_number


def make_hessian_analysis_batch_runner(num_layers: int):
    """Return fixed-cutoff rank and active-spectrum condition number."""
    eigvals_fn = make_energy_hessian_eigvals_fn_for_layer(int(num_layers))

    def metrics_one(theta: jnp.ndarray):
        return hessian_rank_and_condition_from_eigvals(eigvals_fn(theta))

    return jax.jit(jax.vmap(metrics_one))


def compute_qfim_rank_history_by_layer(
    theta_samples_by_layer: dict,
    layers,
    *,
    keep_wires=KEEP_WIRES,
    jvp_chunk: int = RED_JVP_CHUNK,
    representation: str = "reduced",
    batch_size: Optional[int] = None,
):
    rank_history_by_layer = {}
    eigs_history_by_layer = {}
    thresh_history_by_layer = {}

    for L in tqdm(
        layers,
        desc="QFIM rank history along optimization path",
        unit="layer",
    ):
        L_int = int(L)
        if theta_samples_by_layer.get(L_int) is None:
            continue

        theta_samples = np.asarray(
            theta_samples_by_layer[L_int],
            dtype=NP_REAL_DTYPE,
        )

        if theta_samples.ndim != 3:
            raise ValueError(
                "theta_samples must have shape "
                "(num_runs, num_sample_iters, num_params)."
            )

        num_runs, num_times, num_params = theta_samples.shape
        batch_runner = make_qfim_analysis_batch_runner(
            num_layers=L_int,
            keep_wires=keep_wires,
            jvp_chunk=jvp_chunk,
            representation=representation,
        )
        ranks_flat, _, eigs_flat, thresholds_flat = (
            _evaluate_analysis_in_batches(
                theta_samples.reshape((-1, num_params)),
                batch_runner,
                batch_size=batch_size,
                description=(
                    f"QFIM batches ({representation}, L={L_int}, "
                    f"batch={_resolve_analysis_batch_size(batch_size)})"
                ),
            )
        )
        ranks_L = np.asarray(ranks_flat, dtype=NP_REAL_DTYPE).reshape(
            (num_runs, num_times)
        )
        eigs_L = np.asarray(eigs_flat, dtype=NP_REAL_DTYPE).reshape(
            (num_runs, num_times, num_params)
        )
        thresh_L = np.asarray(
            thresholds_flat,
            dtype=NP_REAL_DTYPE,
        ).reshape((num_runs, num_times))

        rank_history_by_layer[L_int] = ranks_L
        eigs_history_by_layer[L_int] = eigs_L
        thresh_history_by_layer[L_int] = thresh_L

        del batch_runner
        _release_jax_compilation_cache()

    return rank_history_by_layer, eigs_history_by_layer, thresh_history_by_layer


def compute_hs_rank_history_by_layer(
    theta_samples_by_layer: dict,
    layers,
    *,
    keep_wires=KEEP_WIRES,
    jvp_chunk: int = RED_JVP_CHUNK,
    representation: str = "reduced",
    batch_size: Optional[int] = None,
):
    rank_history_by_layer = {}
    eigs_history_by_layer = {}
    thresh_history_by_layer = {}

    for L in tqdm(
        layers,
        desc="HS rank history along optimization path",
        unit="layer",
    ):
        L_int = int(L)
        if theta_samples_by_layer.get(L_int) is None:
            continue

        theta_samples = np.asarray(
            theta_samples_by_layer[L_int],
            dtype=NP_REAL_DTYPE,
        )

        if theta_samples.ndim != 3:
            raise ValueError(
                "theta_samples must have shape "
                "(num_runs, num_sample_iters, num_params)."
            )

        num_runs, num_times, num_params = theta_samples.shape
        batch_runner = make_hs_analysis_batch_runner(
            num_layers=L_int,
            keep_wires=keep_wires,
            jvp_chunk=jvp_chunk,
            representation=representation,
        )
        ranks_flat, _, eigs_flat, thresholds_flat = (
            _evaluate_analysis_in_batches(
                theta_samples.reshape((-1, num_params)),
                batch_runner,
                batch_size=batch_size,
                description=(
                    f"HS batches ({representation}, L={L_int}, "
                    f"batch={_resolve_analysis_batch_size(batch_size)})"
                ),
            )
        )
        ranks_L = np.asarray(ranks_flat, dtype=NP_REAL_DTYPE).reshape(
            (num_runs, num_times)
        )
        eigs_L = np.asarray(eigs_flat, dtype=NP_REAL_DTYPE).reshape(
            (num_runs, num_times, num_params)
        )
        thresh_L = np.asarray(
            thresholds_flat,
            dtype=NP_REAL_DTYPE,
        ).reshape((num_runs, num_times))

        rank_history_by_layer[L_int] = ranks_L
        eigs_history_by_layer[L_int] = eigs_L
        thresh_history_by_layer[L_int] = thresh_L

        del batch_runner
        _release_jax_compilation_cache()

    return rank_history_by_layer, eigs_history_by_layer, thresh_history_by_layer

def _qfim_history_plot_iterations(sample_iters_for_plot) -> np.ndarray:
    x = np.asarray(sample_iters_for_plot, dtype=NP_REAL_DTYPE).copy()
    if x.size > 0 and int(x[-1]) == int(steps) - 1:
        x[-1] = NP_REAL_DTYPE(steps)
    return x

def _finite_mean_sem_over_runs_by_time(values_2d: np.ndarray):
    values = np.asarray(values_2d, dtype=NP_REAL_DTYPE)
    if values.ndim != 2:
        raise ValueError("values_2d must have shape (num_runs, num_times).")

    valid = np.isfinite(values)
    counts = np.sum(valid, axis=0)
    sums = np.sum(np.where(valid, values, 0.0), axis=0)

    means = np.full(values.shape[1], np.nan, dtype=NP_REAL_DTYPE)
    np.divide(sums, counts, out=means, where=counts > 0)

    centered = np.where(valid, values - means[None, :], 0.0)
    denom = np.maximum(counts - 1, 1)
    var = np.sum(centered**2, axis=0) / denom
    std = np.sqrt(var)

    sems = np.full(values.shape[1], np.nan, dtype=NP_REAL_DTYPE)
    np.divide(std, np.sqrt(counts), out=sems, where=counts > 1)
    sems = np.where(counts == 1, 0.0, sems)

    return means, sems, counts

def _rank_history_for_plot(rank_history_by_layer: dict, L: int, x: np.ndarray):
    ranks = np.asarray(rank_history_by_layer[int(L)], dtype=NP_REAL_DTYPE)

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

    return ranks

def plot_qfim_rank_history_mean_by_layer(
    rank_history_by_layer: dict,
    layers,
    sample_iters_for_plot,
    *,
    title: str,
    outpath: str,
    ylabel: str = r"Mean QFIM effective rank $(\lambda_k > 10^{-12})$",
    cmap=None,
):
    valid_layers = [
        int(L)
        for L in layers
        if rank_history_by_layer.get(int(L)) is not None
    ]

    if not valid_layers:
        return

    x = _qfim_history_plot_iterations(sample_iters_for_plot)
    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap

    new_prx_figure(width="double")
    ax = plt.gca()
    num_layers = len(valid_layers)

    for layer_idx, L in enumerate(valid_layers):
        ranks = _rank_history_for_plot(rank_history_by_layer, L, x)
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
    set_prx_title(title, ax=ax)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in x], rotation=45, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
    )

    save_current_figure(
        outpath,
        outside_legend=True,
        legend_space_frac=0.26,
    )

def plot_qfim_rank_history_min_by_layer(
    rank_history_by_layer: dict,
    layers,
    sample_iters_for_plot,
    *,
    title: str,
    outpath: str,
    ylabel: str = r"Minimum QFIM effective rank $(\lambda_k > 10^{-12})$",
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

    x = _qfim_history_plot_iterations(sample_iters_for_plot)
    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap

    new_prx_figure(width="double")
    ax = plt.gca()
    num_layers = len(valid_layers)

    for layer_idx, L in enumerate(valid_layers):
        ranks = _rank_history_for_plot(rank_history_by_layer, L, x)
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
            linestyle=":",
            linewidth=1.2,
            markersize=4.5,
            color=color,
            label=f"L={L}",
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel(ylabel)
    set_prx_title(title, ax=ax)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in x], rotation=45, ha="right")
    ax.set_ylim(bottom=0.0)
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

    save_current_figure(
        outpath,
        outside_legend=True,
        legend_space_frac=0.26,
    )

def configure_unitary_pqc_overparam(
    *,
    h_value: Optional[float] = None,
) -> None:
    """Initialize runtime constants, backend settings, Hamiltonian, and initial state."""
    global REAL_DTYPE, COMPLEX_DTYPE, NP_REAL_DTYPE, NP_COMPLEX_DTYPE, NP_INT_DTYPE, INCH_PER_CM
    global FIGSIZE_SINGLE, FIGSIZE_DOUBLE, FIGURE_WIDTH_DEFAULT, SAVE_DPI, SAVEFIG_PAD_INCHES, SAVE_PNG
    global SAVE_PDF
    global CIRCUIT_SAVE_PDF, SHOW_FIGURE_TITLES, SHOW_REDUNDANT_LAYER_LEGENDS, BASE_FONT_SIZE, TITLE_FONT_SIZE, AXIS_LABEL_FONT_SIZE
    global TICK_LABEL_FONT_SIZE, LEGEND_FONT_SIZE, _DEFAULT_AXES_MARGINS_PRX, _DEFAULT_AXES_MARGINS_PRX_OUTSIDE_LEGEND, key, num_system_qubits
    global ANCILLA_QUBIT, num_total_qubits, SYSTEM_WIRES, FULL_WIRES, h_param, tolerance
    global steps, num_runs, lr, NUM_BLOCKS, PARAMS_PER_BLOCK, num_params_per_layer
    global LAYER_PAIRS, H_terms, PAULI, H_matrix, eigvals_np, smallest_eigval
    global X2, _PSI_FULL_INIT
    global save_dir, figures_dir, energy_fig_dir, qfim_fig_dir, hs_fig_dir, hessian_fig_dir
    global circuit_dir, numerical_results_dir, energy_results_dir, qfim_results_dir
    global hs_results_dir, hessian_results_dir
    global qfim_eigs_dir, qfim_eigs_pure_dir, qfim_eigs_reduced_0123_dir
    global qfim_rank_dir, qfim_rank_random_dir, qfim_rank_optimization_path_dir
    global qfim_rank_optimization_path_mean_dir, qfim_rank_optimization_path_min_dir
    global hs_eigs_dir, hs_eigs_reduced_0123_dir
    global hs_rank_dir, hs_rank_random_dir, hs_rank_optimization_path_dir
    global hs_rank_optimization_path_mean_dir, hs_rank_optimization_path_min_dir
    # ============================================================
    # DPQC optimization + plots + QFIM rank (pure + reduced)
    #
    # 5-QUBIT CENTER-ANCILLA VERSION:
    #   - Original physical system qubits are (0,1,2,3).
    #   - A central ancilla qubit is added as qubit 4.
    #   - Each layer contains 4 two-qubit blocks (NUM_BLOCKS=4):
    #       (1,3), (2,3), (0,2), (0,4)
    #     where (0,4) is the added system-ancilla block.
    #   - Each block applies Rz on both qubits and Rxx between them.
    #   - Per-layer parameter count:
    #       num_params_per_layer = NUM_BLOCKS * PARAMS_PER_BLOCK = 12
    #   - The closed 5-qubit circuit is propagated as a 32-amplitude statevector.
    #     A 16x16 reduced density matrix is formed only for mixed subsystem
    #     quantities after tracing out the ancilla.
    #   - Hamiltonian acts nontrivially only on qubits 0..3 and as identity on
    #     the center ancilla qubit 4, i.e. H_total = H_system 竓・I_ancilla.
    #   - QFIM analyses:
    #       * Reduced (mixed-state) QFIM for keep=(0,1,2,3), tracing out ancilla 4
    #       * Pure(full) QFIM for the full 5-qubit state for small layers
    #   - Eigenvalue thresholding rule identical to DPQC_overparam.ipynb:
    #       thresh = 1e-12
    #     and eigenvalues <= thresh are zeroed before numerical storage.
    #
    #   - NEW:
    #       * All box plots are replaced by violin plots.
    #       * For final_energy_error.pdf, an additional beeswarm version is saved as:
    #           final_energy_error_beeswarm.pdf
    #       * A log-scale final-energy-error violin plot is also saved as:
    #           final_energy_error_log.pdf
    #   - DTYPE UPDATE:
    #       * Unified to float64 / complex128
    #       * JAX x64 enabled
    #       * The optional TensorCircuit compatibility builder uses complex128
    # ============================================================
    
    # ==============================
    # Global dtype setup
    # ==============================
    REAL_DTYPE = jnp.float64
    COMPLEX_DTYPE = jnp.complex128
    NP_REAL_DTYPE = np.float64
    NP_COMPLEX_DTYPE = np.complex128
    NP_INT_DTYPE = np.int64
    
    
    # ============================================================
    # APS / PRX-style figure settings
    # ============================================================
    # These settings are centralized in plot.config and loaded by plot.py.
    # They only affect figure appearance/export; numerical computations are
    # intentionally unchanged.
    # ============================================================
    plot_style.apply_plot_config()
    
    INCH_PER_CM = plot_style.INCH_PER_CM
    FIGSIZE_SINGLE = plot_style.FIGSIZE_SINGLE
    FIGSIZE_DOUBLE = plot_style.FIGSIZE_DOUBLE
    FIGURE_WIDTH_DEFAULT = plot_style.FIGURE_WIDTH_DEFAULT
    
    SAVE_DPI = plot_style.SAVE_DPI
    SAVEFIG_PAD_INCHES = plot_style.SAVEFIG_PAD_INCHES
    SAVE_PNG = plot_style.NUMERICAL_SAVE_PNG
    SAVE_PDF = plot_style.NUMERICAL_SAVE_PDF
    CIRCUIT_SAVE_PDF = plot_style.CIRCUIT_SAVE_PDF
    SHOW_FIGURE_TITLES = plot_style.SHOW_FIGURE_TITLES
    SHOW_REDUNDANT_LAYER_LEGENDS = plot_style.SHOW_REDUNDANT_LAYER_LEGENDS
    
    BASE_FONT_SIZE = plot_style.BASE_FONT_SIZE
    TITLE_FONT_SIZE = plot_style.TITLE_FONT_SIZE
    AXIS_LABEL_FONT_SIZE = plot_style.AXIS_LABEL_FONT_SIZE
    TICK_LABEL_FONT_SIZE = plot_style.TICK_LABEL_FONT_SIZE
    LEGEND_FONT_SIZE = plot_style.LEGEND_FONT_SIZE
    
    _DEFAULT_AXES_MARGINS_PRX = plot_style._DEFAULT_AXES_MARGINS_PRX
    _DEFAULT_AXES_MARGINS_PRX_OUTSIDE_LEGEND = (
        plot_style._DEFAULT_AXES_MARGINS_PRX_OUTSIDE_LEGEND
    )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ==============================
    # Backend & Seed
    # ==============================
    key = jax.random.PRNGKey(42)
    
    # ==============================
    # Common Setup
    # ==============================
    # Physical system qubits: 0,1,2,3.
    # A single central ancilla qubit is added as qubit 4.
    num_system_qubits = 4
    ANCILLA_QUBIT = 4
    num_total_qubits = num_system_qubits + 1  # 5 qubits: system (0..3) + center ancilla (4)
    
    SYSTEM_WIRES = tuple(range(num_system_qubits))
    FULL_WIRES = tuple(range(num_total_qubits))
    
    h_param = _resolve_h_param(h_value)
    tolerance = cfg.TOLERANCE
    steps = cfg.STEPS
    num_runs = cfg.NUM_RUNS
    lr = cfg.LEARNING_RATE  # Adam base lr
    
    save_dir = _unitary_pqc_save_dir(h_param)
    figures_dir = os.path.join(save_dir, "figures")
    energy_fig_dir = os.path.join(figures_dir, "energy")
    qfim_fig_dir = os.path.join(figures_dir, "qfim")
    hs_fig_dir = os.path.join(figures_dir, "hs")
    hessian_fig_dir = os.path.join(figures_dir, "hessian")
    circuit_dir = os.path.join(save_dir, "optimized_circuits")
    numerical_results_dir = os.path.join(save_dir, "numerical_results")
    energy_results_dir = os.path.join(numerical_results_dir, "energy")
    qfim_results_dir = os.path.join(numerical_results_dir, "qfim")
    hs_results_dir = os.path.join(numerical_results_dir, "hs")
    hessian_results_dir = os.path.join(numerical_results_dir, "hessian")
    qfim_eigs_dir = os.path.join(qfim_fig_dir, "eigs")
    qfim_eigs_pure_dir = os.path.join(qfim_eigs_dir, "pure_full")
    qfim_eigs_reduced_0123_dir = os.path.join(qfim_eigs_dir, "reduced_keep_0123")
    qfim_rank_dir = os.path.join(qfim_fig_dir, "rank")
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
    hs_eigs_dir = os.path.join(hs_fig_dir, "eigs")
    hs_eigs_reduced_0123_dir = os.path.join(hs_eigs_dir, "reduced_keep_0123")
    hs_rank_dir = os.path.join(hs_fig_dir, "rank")
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
    _ensure_unitary_result_dirs()
    
    # Block structure constants
    # Existing lattice blocks: (1,3), (2,3), (0,2)
    # Added center-ancilla block: (0, ANCILLA_QUBIT)
    NUM_BLOCKS = 4
    PARAMS_PER_BLOCK = 3
    num_params_per_layer = NUM_BLOCKS * PARAMS_PER_BLOCK  # 12
    
    LAYER_PAIRS = (
        (1, 3),
        (2, 3),
        (0, 2),
        (0, ANCILLA_QUBIT),
    )
    
    # ==============================
    # Optional TensorCircuit compatibility builder
    # ==============================
    
    
    
    
    # ==============================
    # Hamiltonian & Ground Truth
    #   - The physical Hamiltonian acts on system qubits 0..3.
    #   - The added center ancilla qubit 4 is acted on by identity.
    #   - Therefore the matrix is built on all 5 qubits as H_system 竓・I_ancilla.
    # ==============================
    
    
    
    
    
    
    H_terms = tuple(hamiltonian_terms(h_param))

    # 5-qubit Hamiltonian matrix (32x32): H_system 竓・I_ancilla
    H_matrix = build_H_matrix_jax(H_terms, num_total_qubits)

    eigvals_np, _ = np.linalg.eigh(np.array(H_matrix, dtype=NP_COMPLEX_DTYPE))
    smallest_eigval = float(eigvals_np.min().real)
    
    # ==============================
    # Wrap
    # ==============================
    
    
    # ============================================================
    # 5-qubit statevector propagation
    #
    # The closed full state lives on 5 qubits: system (0,1,2,3) + ancilla (4),
    # hence the propagated statevector has 32 complex amplitudes.
    #
    # Each layer applies the existing lattice blocks plus the added center-ancilla
    # block (0,4):
    #     (1,3), (2,3), (0,2), (0,4).
    #
    # The energy is evaluated directly as <psi|H_system tensor I|psi>.
    # For the reduced mixed-state QFIM on the original 4-qubit system,
    # rho_keep_sequential_unitary_pqc constructs only the required 16x16 state.
    # ============================================================
    
    X2 = jnp.array([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Precompute the initial five-qubit |00000> statevector.
    _PSI_FULL_INIT = jnp.zeros(
        (2**num_total_qubits,),
        dtype=COMPLEX_DTYPE,
    ).at[0].set(jnp.asarray(1.0, dtype=COMPLEX_DTYPE))
    
    
    
    
    
    
    
    
    # ==============================


def _vqe_sample_slot_by_iteration(
    num_steps: int,
    sample_iterations,
) -> np.ndarray:
    """Map each pre-update iteration to its fixed sample-buffer slot."""
    num_steps = int(num_steps)
    sample_iterations = np.asarray(sample_iterations, dtype=NP_INT_DTYPE)

    if num_steps <= 0:
        raise ValueError("num_steps must be a positive integer.")
    if sample_iterations.ndim != 1:
        raise ValueError("sample_iterations must be one-dimensional.")
    if sample_iterations.size == 0:
        raise ValueError("sample_iterations must not be empty.")
    if np.unique(sample_iterations).size != sample_iterations.size:
        raise ValueError("sample_iterations must not contain duplicates.")
    if np.any(sample_iterations < 0) or np.any(sample_iterations >= num_steps):
        raise ValueError(
            "sample_iterations must lie in the half-open range "
            f"[0, {num_steps})."
        )

    slot_by_iteration = np.full(num_steps, -1, dtype=np.int32)
    slot_by_iteration[sample_iterations] = np.arange(
        sample_iterations.size,
        dtype=np.int32,
    )
    return slot_by_iteration


def make_vqe_batch_runner(
    current_layer: int,
    *,
    num_steps: int,
    sample_iterations,
    optimizer,
):
    """Compile a fixed-size batch of independent Unitary-PQC VQE runs.

    The trace convention intentionally matches the historical Unitary-PQC
    archive: index ``t`` stores energy/gradient data at ``theta_t`` before the
    corresponding optimizer update, while ``theta_final`` is ``theta_steps``.
    """
    current_layer = int(current_layer)
    num_steps = int(num_steps)
    num_total_params = num_params_per_layer * current_layer
    sample_slot_by_iteration = _vqe_sample_slot_by_iteration(
        num_steps,
        sample_iterations,
    )
    scan_sample_slots = jnp.asarray(
        sample_slot_by_iteration,
        dtype=jnp.int32,
    )
    num_samples = int(np.asarray(sample_iterations).size)
    energy_and_grad = jax.value_and_grad(
        make_energy_fn_for_layer(current_layer)
    )

    def optimize_one_run(theta_initial: jnp.ndarray):
        theta = jnp.asarray(theta_initial, dtype=REAL_DTYPE)
        opt_state = optimizer.init(theta)
        theta_samples = jnp.zeros(
            (num_samples, num_total_params),
            dtype=REAL_DTYPE,
        )

        def one_step(carry, sample_slot):
            theta_old, opt_state_old, theta_samples_old = carry
            energy, grad = energy_and_grad(theta_old)
            grad_norm = jnp.linalg.norm(grad)
            updates, opt_state_new = optimizer.update(
                grad,
                opt_state_old,
                theta_old,
            )
            theta_new = wrap_to_pi(
                optax.apply_updates(theta_old, updates)
            )

            theta_samples_new = jax.lax.cond(
                sample_slot >= 0,
                lambda buffer: buffer.at[sample_slot].set(theta_old),
                lambda buffer: buffer,
                theta_samples_old,
            )
            new_carry = (
                theta_new,
                opt_state_new,
                theta_samples_new,
            )
            return new_carry, (energy, grad_norm)

        (
            (
                theta_final,
                _,
                theta_samples_final,
            ),
            (energy_trace, grad_norm_trace),
        ) = jax.lax.scan(
            one_step,
            (theta, opt_state, theta_samples),
            scan_sample_slots,
        )
        return (
            theta_final,
            energy_trace,
            grad_norm_trace,
            theta_samples_final,
        )

    return jax.jit(jax.vmap(optimize_one_run))


def _pad_vqe_theta_batch(
    theta_batch: jnp.ndarray,
    batch_size: int,
) -> tuple[jnp.ndarray, int]:
    """Pad a final partial batch without changing any valid vmap lane."""
    batch_size = int(batch_size)
    valid_count = int(theta_batch.shape[0])
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not 1 <= valid_count <= batch_size:
        raise ValueError(
            f"Expected between 1 and {batch_size} runs, got {valid_count}."
        )
    if valid_count == batch_size:
        return theta_batch, valid_count

    padding = jnp.repeat(
        theta_batch[-1:, :],
        repeats=batch_size - valid_count,
        axis=0,
    )
    return jnp.concatenate((theta_batch, padding), axis=0), valid_count


def _resolve_vqe_batch_size(batch_size: Optional[int]) -> int:
    resolved = VQE_BATCH_SIZE if batch_size is None else int(batch_size)
    if resolved <= 0:
        raise ValueError("vqe_batch_size must be a positive integer.")
    return resolved


def run_vqe_optimization(
    *,
    save_circuits: bool = False,
    vqe_batch_size: Optional[int] = None,
) -> None:
    """Run VQE optimization for every configured layer and collect traces.

    ``save_circuits`` is retained for call compatibility. Circuit drawing is
    now always deferred to
    ``unitary_pqc_measured_0_overparam_draw_circuits.py``.
    Independent trials are compiled as ``jit(vmap(scan))`` batches.
    """
    global success_rates_history, energy_mean_history, energy_std_history, final_stats, dense_until_layer, max_layer
    global sparse_step, dense_end, layer_list, save_dir, figures_dir, energy_fig_dir, qfim_fig_dir, hs_fig_dir, hessian_fig_dir, circuit_dir, optimizer
    global qfim_eigs_dir, qfim_eigs_pure_dir, qfim_eigs_reduced_0123_dir
    global qfim_rank_dir, qfim_rank_random_dir, qfim_rank_optimization_path_dir
    global qfim_rank_optimization_path_mean_dir, qfim_rank_optimization_path_min_dir
    global hs_eigs_dir, hs_eigs_reduced_0123_dir
    global hs_rank_dir, hs_rank_random_dir, hs_rank_optimization_path_dir
    global hs_rank_optimization_path_mean_dir, hs_rank_optimization_path_min_dir
    global theta_history, best_theta_by_layer, final_theta_wrapped_rmsdist_by_layer, energy_traces_by_layer, grad_norm_traces_by_layer, sample_every
    global sample_iters, sample_iter_set, theta_sample_traces_by_layer, cmap
    global numerical_results_dir, energy_results_dir, qfim_results_dir, hs_results_dir, hessian_results_dir
    effective_batch_size = _resolve_vqe_batch_size(vqe_batch_size)
    if int(num_runs) <= 0:
        raise ValueError("cfg.NUM_RUNS must be a positive integer.")
    if save_circuits:
        warnings.warn(
            "save_circuits is no longer executed during VQE. Run "
            "src/unitary_pqc/"
            "unitary_pqc_measured_0_overparam_draw_circuits.py after "
            "the numerical results have been saved.",
            UserWarning,
            stacklevel=2,
        )

    # Optimization Loop per Layer
    # ==============================
    success_rates_history = {}
    energy_mean_history = {}
    energy_std_history = {}
    
    final_stats = {"layer": [], "success_rate": [], "mean_energy": [], "std_energy": []}
    
    dense_until_layer = cfg.UNITARY_PQC_DENSE_UNTIL_LAYER
    max_layer = cfg.UNITARY_PQC_MAX_LAYER
    sparse_step = cfg.UNITARY_PQC_SPARSE_STEP
    
    dense_end = min(dense_until_layer, max_layer)
    layer_list = build_layer_list(max_layer, dense_until_layer, sparse_step)
    
    # --- Result directories ---
    save_dir = _unitary_pqc_save_dir(h_param)
    figures_dir = os.path.join(save_dir, "figures")
    energy_fig_dir = os.path.join(figures_dir, "energy")
    qfim_fig_dir = os.path.join(figures_dir, "qfim")
    hs_fig_dir = os.path.join(figures_dir, "hs")
    hessian_fig_dir = os.path.join(figures_dir, "hessian")
    circuit_dir = os.path.join(save_dir, "optimized_circuits")
    numerical_results_dir = os.path.join(save_dir, "numerical_results")
    energy_results_dir = os.path.join(numerical_results_dir, "energy")
    qfim_results_dir = os.path.join(numerical_results_dir, "qfim")
    hs_results_dir = os.path.join(numerical_results_dir, "hs")
    hessian_results_dir = os.path.join(numerical_results_dir, "hessian")
    qfim_eigs_dir = os.path.join(qfim_fig_dir, "eigs")
    qfim_eigs_pure_dir = os.path.join(qfim_eigs_dir, "pure_full")
    qfim_eigs_reduced_0123_dir = os.path.join(qfim_eigs_dir, "reduced_keep_0123")
    qfim_rank_dir = os.path.join(qfim_fig_dir, "rank")
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
    hs_eigs_dir = os.path.join(hs_fig_dir, "eigs")
    hs_eigs_reduced_0123_dir = os.path.join(hs_eigs_dir, "reduced_keep_0123")
    hs_rank_dir = os.path.join(hs_fig_dir, "rank")
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
    _ensure_unitary_result_dirs()
    
    optimizer = optax.adam(learning_rate=lr)
    
    theta_history = {L: [] for L in layer_list}  # final theta of each run
    best_theta_by_layer = {}
    
    # Final RMS wrapped parameter distance per layer (distribution over runs)
    #   Reference theta_ref(L): best-run final parameters at the same layer
    final_theta_wrapped_rmsdist_by_layer = {}  # L -> (num_runs,) array of d_theta(theta_final, theta_ref)
    
    energy_traces_by_layer = {}
    grad_norm_traces_by_layer = {}  # L -> (num_runs, steps) gradient-norm traces
    
    # Sampled optimization-time states feed the post-VQE matrix analyses.
    sample_every = cfg.SAMPLE_EVERY
    sample_iters = np.arange(0, steps, sample_every, dtype=NP_INT_DTYPE)
    if sample_iters.size == 0 or sample_iters[0] != 0:
        sample_iters = np.concatenate([[0], sample_iters]).astype(NP_INT_DTYPE)
    
    if sample_iters[-1] != steps - 1:
        sample_iters = np.concatenate([sample_iters, [steps - 1]]).astype(NP_INT_DTYPE)
    
    sample_iters = np.unique(sample_iters).astype(NP_INT_DTYPE)
    sample_iter_set = set(int(t) for t in sample_iters.tolist())
    
    theta_sample_traces_by_layer = {}
    cmap = matplotlib.colormaps.get_cmap("viridis")
    
    # tqdm: Layers (VQE)
    for current_layer in tqdm(layer_list, desc="Layers (VQE)", unit="layer"):
        num_total_params = num_params_per_layer * current_layer
    
        run_vqe_batch = make_vqe_batch_runner(
            current_layer,
            num_steps=steps,
            sample_iterations=sample_iters,
            optimizer=optimizer,
        )

        # Generate every run before batching so batch size never changes the
        # historical random-key sequence or run ordering.
        base_key = jax.random.PRNGKey(current_layer * 1000)
        keys = jax.random.split(base_key, num_runs)
        theta_initial_runs = jnp.stack(
            [
                jax.random.uniform(
                    keys[run_index],
                    shape=(num_total_params,),
                    minval=-jnp.pi,
                    maxval=jnp.pi,
                    dtype=REAL_DTYPE,
                )
                for run_index in range(num_runs)
            ],
            axis=0,
        )

        output_parts = tuple([] for _ in range(4))
        batch_starts = range(0, num_runs, effective_batch_size)
        for batch_start in tqdm(
            batch_starts,
            total=(num_runs + effective_batch_size - 1) // effective_batch_size,
            desc=(
                f"Run batches (L={current_layer}, "
                f"batch={effective_batch_size})"
            ),
            unit="batch",
            leave=False,
        ):
            batch_end = min(batch_start + effective_batch_size, num_runs)
            theta_batch, valid_count = _pad_vqe_theta_batch(
                theta_initial_runs[batch_start:batch_end],
                effective_batch_size,
            )
            host_outputs = jax.device_get(run_vqe_batch(theta_batch))
            for parts, values in zip(output_parts, host_outputs):
                parts.append(
                    np.asarray(
                        values[:valid_count],
                        dtype=NP_REAL_DTYPE,
                    )
                )

        (
            theta_final_data,
            energy_data,
            gradnorm_data,
            theta_sample_data,
        ) = (
            np.concatenate(parts, axis=0)
            for parts in output_parts
        )
        expected_shapes = (
            (num_runs, num_total_params),
            (num_runs, steps),
            (num_runs, steps),
            (num_runs, sample_iters.size, num_total_params),
        )
        actual_shapes = tuple(
            array.shape
            for array in (
                theta_final_data,
                energy_data,
                gradnorm_data,
                theta_sample_data,
            )
        )
        if actual_shapes != expected_shapes:
            raise AssertionError(
                f"Unexpected VQE output shapes for L={current_layer}: "
                f"{actual_shapes} != {expected_shapes}."
            )

        final_energies = energy_data[:, -1]
        finite_run_indices = np.flatnonzero(np.isfinite(final_energies))
        if finite_run_indices.size == 0:
            raise FloatingPointError(
                f"No finite final VQE energy was produced for L={current_layer}."
            )
        best_local_index = int(np.argmin(final_energies[finite_run_indices]))
        best_run_index = int(finite_run_indices[best_local_index])
        best_final_theta = theta_final_data[best_run_index].copy()

        theta_history[current_layer] = theta_final_data

        # Final RMS wrapped distance distribution over runs
        theta_runs_jnp = jnp.asarray(theta_history[current_layer], dtype=REAL_DTYPE)     # (num_runs, num_params)
        theta_ref_jnp = jnp.asarray(best_final_theta, dtype=REAL_DTYPE)[None, :]         # (1, num_params)
    
        wrapped_diff = wrap_to_pi(theta_runs_jnp - theta_ref_jnp)       # (num_runs, num_params)
        d_theta_runs = jnp.sqrt(jnp.mean(wrapped_diff ** 2, axis=1))    # (num_runs,)
    
        final_theta_wrapped_rmsdist_by_layer[current_layer] = np.asarray(
            jax.device_get(d_theta_runs), dtype=NP_REAL_DTYPE
        )
    
        energy_traces_by_layer[current_layer] = energy_data
        grad_norm_traces_by_layer[current_layer] = gradnorm_data
        theta_sample_traces_by_layer[current_layer] = theta_sample_data
    
        best_theta_by_layer[current_layer] = best_final_theta.copy()
    
        # stats (energy mean/std)
        mean_trace = np.mean(energy_data, axis=0)
        std_trace = np.std(energy_data, axis=0)
        energy_mean_history[current_layer] = mean_trace
        energy_std_history[current_layer] = std_trace
    
        diffs = np.abs(energy_data - smallest_eigval)
        success_flags = diffs <= tolerance
        success_rate_per_step = np.mean(success_flags, axis=0)
        success_rates_history[current_layer] = success_rate_per_step
    
        final_stats["layer"].append(current_layer)
        final_stats["success_rate"].append(success_rate_per_step[-1])
        final_stats["mean_energy"].append(mean_trace[-1])
        final_stats["std_energy"].append(std_trace[-1])

        del run_vqe_batch
        _release_jax_compilation_cache()

    save_unitary_vqe_results()
    
    
    # ==============================


def save_unitary_vqe_results() -> str:
    outpath = os.path.join(energy_results_dir, "vqe_optimization_results.npz")
    layers_arr = np.asarray(layer_list, dtype=NP_INT_DTYPE)

    save_npz_result(
        outpath,
        ansatz=np.asarray(ANSATZ_NAME),
        measurement_outcome=np.asarray(
            MEASUREMENT_OUTCOME,
            dtype=NP_INT_DTYPE,
        ),
        num_params_per_layer=np.asarray(
            num_params_per_layer,
            dtype=NP_INT_DTYPE,
        ),
        h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
        tolerance=np.asarray(tolerance, dtype=NP_REAL_DTYPE),
        steps=np.asarray(steps, dtype=NP_INT_DTYPE),
        num_runs=np.asarray(num_runs, dtype=NP_INT_DTYPE),
        learning_rate=np.asarray(lr, dtype=NP_REAL_DTYPE),
        layers=layers_arr,
        sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        smallest_eigval=np.asarray(smallest_eigval, dtype=NP_REAL_DTYPE),
        final_stats_layer=np.asarray(final_stats["layer"], dtype=NP_INT_DTYPE),
        final_stats_success_rate=np.asarray(
            final_stats["success_rate"],
            dtype=NP_REAL_DTYPE,
        ),
        final_stats_mean_energy=np.asarray(
            final_stats["mean_energy"],
            dtype=NP_REAL_DTYPE,
        ),
        final_stats_std_energy=np.asarray(
            final_stats["std_energy"],
            dtype=NP_REAL_DTYPE,
        ),
        **{
            f"L{int(L)}_theta_history": np.asarray(
                theta_history[int(L)],
                dtype=NP_REAL_DTYPE,
            )
            for L in layer_list
        },
        **{
            f"L{int(L)}_best_theta": np.asarray(
                best_theta_by_layer[int(L)],
                dtype=NP_REAL_DTYPE,
            )
            for L in layer_list
        },
        **{
            f"L{int(L)}_final_theta_wrapped_rmsdist": np.asarray(
                final_theta_wrapped_rmsdist_by_layer[int(L)],
                dtype=NP_REAL_DTYPE,
            )
            for L in layer_list
        },
        **{
            f"L{int(L)}_energy_traces": np.asarray(
                energy_traces_by_layer[int(L)],
                dtype=NP_REAL_DTYPE,
            )
            for L in layer_list
        },
        **{
            f"L{int(L)}_grad_norm_traces": np.asarray(
                grad_norm_traces_by_layer[int(L)],
                dtype=NP_REAL_DTYPE,
            )
            for L in layer_list
        },
        **{
            f"L{int(L)}_theta_samples": np.asarray(
                theta_sample_traces_by_layer[int(L)],
                dtype=NP_REAL_DTYPE,
            )
            for L in layer_list
        },
    )

    return outpath


def load_unitary_vqe_samples(inpath: Optional[str] = None) -> str:
    """Restore only the saved VQE arrays required by post-VQE analyses.

    The QFIM stage runs in a fresh process, so its optimization-path analyses
    cannot use the dictionaries populated by ``run_vqe_optimization``
    directly. The VQE archive is the process boundary: validate it before
    publishing any of its values as module globals.
    """
    global layer_list, sample_iters, sample_iter_set, steps, num_runs, cmap
    global theta_sample_traces_by_layer

    result_path = (
        os.path.join(energy_results_dir, "vqe_optimization_results.npz")
        if inpath is None
        else os.fspath(inpath)
    )
    if not os.path.isfile(result_path):
        raise FileNotFoundError(
            "The QFIM stage requires an existing VQE archive. Run "
            f"--stage vqe first: {result_path}"
        )

    with np.load(result_path, allow_pickle=False) as data:
        metadata_keys = (
            "ansatz",
            "measurement_outcome",
            "num_params_per_layer",
            "h_param",
            "steps",
            "num_runs",
            "layers",
            "sample_iters",
        )
        missing_metadata = [key for key in metadata_keys if key not in data]
        if missing_metadata:
            raise KeyError(
                "VQE archive is missing required metadata: "
                + ", ".join(missing_metadata)
            )

        archived_h_param = np.asarray(data["h_param"])
        archived_steps = np.asarray(data["steps"])
        archived_num_runs = np.asarray(data["num_runs"])
        archived_ansatz = np.asarray(data["ansatz"])
        archived_measurement_outcome = np.asarray(
            data["measurement_outcome"]
        )
        archived_num_params_per_layer = np.asarray(
            data["num_params_per_layer"]
        )
        if (
            archived_h_param.shape != ()
            or archived_steps.shape != ()
            or archived_num_runs.shape != ()
            or archived_ansatz.shape != ()
            or archived_measurement_outcome.shape != ()
            or archived_num_params_per_layer.shape != ()
        ):
            raise ValueError(
                "Saved variant metadata, h_param, steps, and num_runs must "
                "be scalar values."
            )
        if str(archived_ansatz.item()) != ANSATZ_NAME:
            raise ValueError(
                "Saved VQE ansatz does not match this program: "
                f"{archived_ansatz.item()!r} != {ANSATZ_NAME!r}."
            )
        if not np.issubdtype(archived_measurement_outcome.dtype, np.integer):
            raise TypeError("Saved measurement_outcome must use an integer dtype.")
        if int(archived_measurement_outcome.item()) != MEASUREMENT_OUTCOME:
            raise ValueError(
                "Saved VQE measurement outcome does not match this program: "
                f"{int(archived_measurement_outcome.item())} != "
                f"{MEASUREMENT_OUTCOME}."
            )
        if not np.issubdtype(
            archived_num_params_per_layer.dtype,
            np.integer,
        ):
            raise TypeError(
                "Saved num_params_per_layer must use an integer dtype."
            )
        if int(archived_num_params_per_layer.item()) != num_params_per_layer:
            raise ValueError(
                "Saved VQE parameter count does not match this program: "
                f"{int(archived_num_params_per_layer.item())} != "
                f"{num_params_per_layer}."
            )
        if (
            not np.issubdtype(archived_h_param.dtype, np.number)
            or np.issubdtype(archived_h_param.dtype, np.complexfloating)
        ):
            raise TypeError("Saved h_param must use a real numeric dtype.")
        if not np.issubdtype(archived_steps.dtype, np.integer):
            raise TypeError("Saved steps must use an integer dtype.")
        if not np.issubdtype(archived_num_runs.dtype, np.integer):
            raise TypeError("Saved num_runs must use an integer dtype.")

        archived_h_value = NP_REAL_DTYPE(archived_h_param.item())
        if not np.isfinite(archived_h_value):
            raise ValueError("Saved h_param must be finite.")
        if archived_h_value != NP_REAL_DTYPE(h_param):
            raise ValueError(
                "Saved VQE h_param does not match the selected Hamiltonian: "
                f"{float(archived_h_value)} != {float(h_param)}."
            )

        archived_steps_value = int(archived_steps.item())
        archived_num_runs_value = int(archived_num_runs.item())
        if archived_steps_value <= 0:
            raise ValueError("Saved steps must be positive.")
        if archived_num_runs_value <= 0:
            raise ValueError("Saved num_runs must be positive.")

        layers_raw = np.asarray(data["layers"])
        sample_iters_raw = np.asarray(data["sample_iters"])
        if not np.issubdtype(layers_raw.dtype, np.integer):
            raise TypeError("Saved layers must use an integer dtype.")
        if not np.issubdtype(sample_iters_raw.dtype, np.integer):
            raise TypeError("Saved sample_iters must use an integer dtype.")

        archived_layers = np.array(
            layers_raw,
            dtype=NP_INT_DTYPE,
            copy=True,
        )
        archived_sample_iters = np.array(
            sample_iters_raw,
            dtype=NP_INT_DTYPE,
            copy=True,
        )
        if archived_layers.ndim != 1 or archived_layers.size == 0:
            raise ValueError("Saved layers must be a non-empty 1D array.")
        if np.any(archived_layers <= 0):
            raise ValueError("Saved layers must be positive.")
        if np.unique(archived_layers).size != archived_layers.size:
            raise ValueError("Saved layers must not contain duplicates.")
        if (
            archived_sample_iters.ndim != 1
            or archived_sample_iters.size == 0
        ):
            raise ValueError(
                "Saved sample_iters must be a non-empty 1D array."
            )
        if np.any(archived_sample_iters < 0) or np.any(
            np.diff(archived_sample_iters) <= 0
        ):
            raise ValueError(
                "Saved sample_iters must be nonnegative and strictly "
                "increasing."
            )
        if int(archived_sample_iters[-1]) >= archived_steps_value:
            raise ValueError("Saved sample_iters must be smaller than steps.")

        theta_samples = {}
        expected_real_dtype = np.dtype(NP_REAL_DTYPE)
        for layer_value in archived_layers:
            layer = int(layer_value)
            theta_key = f"L{layer}_theta_samples"
            if theta_key not in data:
                raise KeyError(
                    f"VQE archive is missing required array: {theta_key}"
                )

            theta_raw = data[theta_key]
            if theta_raw.dtype != expected_real_dtype:
                raise TypeError(
                    f"Saved L={layer} theta samples must be float64, "
                    f"got {theta_raw.dtype}."
                )

            theta = np.array(theta_raw, dtype=NP_REAL_DTYPE, copy=True)
            expected_shape = (
                archived_num_runs_value,
                archived_sample_iters.size,
                num_params_per_layer * layer,
            )
            if theta.shape != expected_shape:
                raise ValueError(
                    f"Saved L={layer} theta sample shape mismatch: "
                    f"{theta.shape} != {expected_shape}."
                )
            if not np.all(np.isfinite(theta)):
                raise ValueError(
                    f"Saved L={layer} theta samples contain non-finite values."
                )

            theta_samples[layer] = theta

    # Publish the validated archive atomically at the Python-object level.
    layer_list = [int(layer) for layer in archived_layers.tolist()]
    sample_iters = archived_sample_iters
    sample_iter_set = set(int(value) for value in sample_iters.tolist())
    steps = archived_steps_value
    num_runs = archived_num_runs_value
    theta_sample_traces_by_layer = theta_samples
    cmap = matplotlib.colormaps.get_cmap("viridis")
    return result_path


def plot_vqe_optimization_results() -> None:
    """Generate VQE energy, gradient, final-error, and success-rate plots."""
    # Plotting helpers (violin)
    # ==============================
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(energy_fig_dir, exist_ok=True)
    
    
    
    
    
    # ------------------------------------------------------------
    # Plot A: Loss history as VIOLIN (ABSOLUTE ERROR, LOG SCALE)
    # ------------------------------------------------------------
    new_prx_figure(width="double")
    
    eps = 1e-12  # for log-scale safety
    
    x_groups = np.arange(len(sample_iters), dtype=NP_REAL_DTYPE)
    
    num_layers = len(layer_list)
    offset_span = 0.75
    violin_width = 0.85 * (offset_span / num_layers)
    
    legend_handles = []
    for idx, L in enumerate(layer_list):
        color = cmap(idx / num_layers)
        legend_handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.25, label=f"L{L}"))
    
        offset = (idx - (num_layers - 1) / 2) * (offset_span / num_layers)
        positions = x_groups + offset
    
        e_runs = np.asarray(energy_traces_by_layer[L], dtype=NP_REAL_DTYPE)
        datasets = [
            _make_violin_ready(np.abs(e_runs[:, t] - smallest_eigval) + eps, ensure_positive=True, tiny=eps)
            for t in sample_iters
        ]
    
        vp = plt.violinplot(
            datasets,
            positions=positions,
            widths=violin_width,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        _style_violin(vp, facecolor=color, edgecolor=color, alpha=0.20, linewidth=1.0, linecolor=color)
    
    plt.xlabel("Iterations")
    plt.ylabel(r"Absolute error  $|E(\theta) - E_{\mathrm{GT}}|$")
    set_prx_title(f"Loss history over {num_runs} runs")
    xtick_labels = [str(t) for t in sample_iters]
    xtick_labels[-1] = str(steps)
    plt.xticks(x_groups, xtick_labels, rotation=45, ha="right")
    
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure(
        os.path.join(energy_fig_dir, "loss_history.pdf"),
        outside_legend=True,
    )
    
    
    # ------------------------------------------------------------
    # Plot A2: Gradient-norm history as VIOLIN (LOG SCALE)
    # ------------------------------------------------------------
    new_prx_figure(width="double")
    
    eps_g = 1e-12  # for log-scale safety
    
    legend_handles_g = []
    for idx, L in enumerate(layer_list):
        color = cmap(idx / num_layers)
        legend_handles_g.append(Patch(facecolor=color, edgecolor=color, alpha=0.25, label=f"L{L}"))
    
        offset = (idx - (num_layers - 1) / 2) * (offset_span / num_layers)
        positions = x_groups + offset
    
        g_runs = np.asarray(grad_norm_traces_by_layer[L], dtype=NP_REAL_DTYPE)
        datasets = [
            _make_violin_ready(g_runs[:, t] + eps_g, ensure_positive=True, tiny=eps_g)
            for t in sample_iters
        ]
    
        vp = plt.violinplot(
            datasets,
            positions=positions,
            widths=violin_width,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        _style_violin(vp, facecolor=color, edgecolor=color, alpha=0.20, linewidth=1.0, linecolor=color)
    
    plt.xlabel("Iterations")
    plt.ylabel(r"Gradient norm  $\|\nabla_\theta E(\theta)\|_2$")
    set_prx_title(f"Gradient-norm history over {num_runs} runs")
    xtick_labels = [str(t) for t in sample_iters]
    xtick_labels[-1] = str(steps)
    plt.xticks(x_groups, xtick_labels, rotation=45, ha="right")
    
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(handles=legend_handles_g, bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure(
        os.path.join(energy_fig_dir, "grad_norm_history.pdf"),
        outside_legend=True,
    )
    
    
    # ------------------------------------------------------------
    # Plot 1: Energy Error History (Log-Statistics)
    # ------------------------------------------------------------
    new_prx_figure(width="double")
    eps = 1e-12
    
    for idx, n in enumerate(layer_list):
        color = cmap(idx / len(layer_list))
    
        mean_data = np.asarray(energy_mean_history[n], dtype=NP_REAL_DTYPE)
        std_data = np.asarray(energy_std_history[n], dtype=NP_REAL_DTYPE)
        x_axis = np.arange(len(mean_data))
    
        err_mean = np.abs(mean_data - smallest_eigval) + eps
        err_std = std_data
    
        var = err_std**2
        mu_log = np.log(err_mean**2 / np.sqrt(var + err_mean**2))
        sigma_log = np.sqrt(np.log(1.0 + var / err_mean**2))
    
        err_center = np.exp(mu_log)
        err_low = np.exp(mu_log - sigma_log)
        err_high = np.exp(mu_log + sigma_log)
    
        plt.semilogy(
            x_axis,
            err_center,
            label=f"L{n}",
            color=color,
            linewidth=1.0,
        )
        plt.fill_between(
            x_axis,
            err_low,
            err_high,
            color=color,
            alpha=0.15,
        )
    
    plt.xlabel("Iterations")
    plt.ylabel(r"Absolute Error  $|E - E_{\mathrm{GT}}|$")
    set_prx_title(rf"Error (log-mean $\pm$ log-std over {num_runs} runs)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", alpha=0.3)
    save_current_figure(
        os.path.join(energy_fig_dir, "energy_error_history_logstat.pdf"),
        outside_legend=True,
    )
    
    
    # ------------------------------------------------------------
    # Plot 2a: Final Energy ERROR distribution (VIOLIN, LINEAR) vs depth
    #   - saved as final_energy_error.pdf
    # ------------------------------------------------------------
    new_prx_figure(width="double")
    
    num_layers = len(layer_list)
    final_error_runs = [
        np.abs(np.asarray(energy_traces_by_layer[L][:, -1], dtype=NP_REAL_DTYPE) - smallest_eigval)
        for L in layer_list
    ]
    
    positions = np.arange(1, len(layer_list) + 1, dtype=NP_REAL_DTYPE)
    
    datasets_2a = [_make_violin_ready(v, ensure_positive=False, tiny=1e-12) for v in final_error_runs]
    
    vp = plt.violinplot(
        datasets_2a,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    
    legend_handles = []
    for idx, (L, body) in enumerate(zip(layer_list, vp["bodies"])):
        color = cmap(idx / num_layers)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.20)
        body.set_linewidth(1.0)
        legend_handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.25, label=f"L{L}"))
    
    for k in ["cmins", "cmaxes", "cbars", "cmedians"]:
        if k in vp:
            vp[k].set_linewidth(1.0)
            vp[k].set_color("black")
            vp[k].set_alpha(0.7)
    
    plt.xticks(ticks=positions, labels=[str(L) for L in layer_list])
    plt.xlabel("Number of Layers")
    plt.ylabel(r"Final error $|E(\theta_{\mathrm{final}})-E_{\mathrm{GT}}|$")
    set_prx_title(f"Final Energy Error Distribution over {num_runs} runs")
    plt.grid(True, axis="y", alpha=0.3)
    
    if SHOW_REDUNDANT_LAYER_LEGENDS:
        plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure(
        os.path.join(energy_fig_dir, "final_energy_error.pdf"),
        outside_legend=SHOW_REDUNDANT_LAYER_LEGENDS,
    )
    
    
    # ------------------------------------------------------------
    # Plot 2a-log: Final Energy ERROR distribution (VIOLIN, LOG) vs depth
    #   - saved as final_energy_error_log.pdf
    # ------------------------------------------------------------
    new_prx_figure(width="double")
    
    datasets_2a_log = [
        _make_violin_ready(v, ensure_positive=True, tiny=1e-12)
        for v in final_error_runs
    ]
    
    vp = plt.violinplot(
        datasets_2a_log,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    
    legend_handles_log = []
    for idx, (L, body) in enumerate(zip(layer_list, vp["bodies"])):
        color = cmap(idx / num_layers)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.20)
        body.set_linewidth(1.0)
        legend_handles_log.append(
            Patch(facecolor=color, edgecolor=color, alpha=0.25, label=f"L{L}")
        )
    
    for k in ["cmins", "cmaxes", "cbars", "cmedians"]:
        if k in vp:
            vp[k].set_linewidth(1.0)
            vp[k].set_color("black")
            vp[k].set_alpha(0.7)
    
    plt.xticks(ticks=positions, labels=[str(L) for L in layer_list])
    plt.xlabel("Number of Layers")
    plt.ylabel(r"Final error $|E(\theta_{\mathrm{final}})-E_{\mathrm{GT}}|$")
    plt.yscale("log")
    set_prx_title(f"Final Energy Error Distribution over {num_runs} runs (log scale)")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    
    if SHOW_REDUNDANT_LAYER_LEGENDS:
        plt.legend(handles=legend_handles_log, bbox_to_anchor=(1.05, 1), loc="upper left")
    save_current_figure(
        os.path.join(energy_fig_dir, "final_energy_error_log.pdf"),
        outside_legend=SHOW_REDUNDANT_LAYER_LEGENDS,
    )
    
    
    # ------------------------------------------------------------
    # Plot 2a-extra: Final Energy ERROR distribution (BEESWARM, LINEAR) vs depth
    #   - saved as final_energy_error_beeswarm.pdf
    # ------------------------------------------------------------
    new_prx_figure(width="double")
    
    num_layers = len(layer_list)
    positions = np.arange(1, len(layer_list) + 1, dtype=NP_REAL_DTYPE)
    
    final_error_runs = [
        np.abs(np.asarray(energy_traces_by_layer[L][:, -1], dtype=NP_REAL_DTYPE) - smallest_eigval)
        for L in layer_list
    ]
    
    rng_beeswarm = np.random.default_rng(12345)
    
    legend_handles = []
    for idx, (L, errs) in enumerate(zip(layer_list, final_error_runs)):
        color = cmap(idx / num_layers)
    
        errs = np.asarray(errs, dtype=NP_REAL_DTYPE).ravel()
    
        # Deterministic horizontal jitter for beeswarm-like visualization.
        # This is only for plotting; numerical data are unchanged.
        jitter_width = 0.28
        x_jitter = rng_beeswarm.uniform(
            low=-jitter_width,
            high=jitter_width,
            size=errs.size,
        )
        x_vals = positions[idx] + x_jitter
    
        plt.scatter(
            x_vals,
            errs,
            s=12,
            alpha=0.75,
            edgecolors="none",
            color=color,
        )
    
        # Median marker
        median_err = np.median(errs)
        plt.plot(
            [positions[idx] - 0.32, positions[idx] + 0.32],
            [median_err, median_err],
            linewidth=1.2,
            color="black",
            alpha=0.8,
        )
    
        legend_handles.append(
            Patch(facecolor=color, edgecolor=color, alpha=0.25, label=f"L{L}")
        )
    
    plt.xticks(ticks=positions, labels=[str(L) for L in layer_list])
    plt.xlabel("Number of Layers")
    plt.ylabel(r"Final error $|E(\theta_{\mathrm{final}})-E_{\mathrm{GT}}|$")
    set_prx_title(f"Final Energy Error Beeswarm Plot over {num_runs} runs")
    plt.grid(True, axis="y", alpha=0.3)
    
    if SHOW_REDUNDANT_LAYER_LEGENDS:
        plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    
    save_current_figure(
        os.path.join(energy_fig_dir, "final_energy_error_beeswarm.pdf"),
        outside_legend=SHOW_REDUNDANT_LAYER_LEGENDS,
    )
    
    
    # ------------------------------------------------------------
    # Plot 2.5: Final RMS wrapped parameter distance (VIOLIN) vs depth
    #   - overwrite: final_theta_wrapped_rmsdist_vs_layers.pdf
    # ------------------------------------------------------------
    new_prx_figure(width="double")
    
    num_layers = len(layer_list)
    positions = np.array(layer_list, dtype=NP_REAL_DTYPE)
    
    final_theta_dist_runs = [
        np.asarray(final_theta_wrapped_rmsdist_by_layer[L], dtype=NP_REAL_DTYPE) for L in layer_list
    ]
    
    if positions.size > 1:
        violin_width_25 = 0.75 * np.min(np.diff(np.sort(positions)))
    else:
        violin_width_25 = 0.75
    
    datasets_25 = [_make_violin_ready(v, ensure_positive=False, tiny=1e-12) for v in final_theta_dist_runs]
    
    vp = plt.violinplot(
        datasets_25,
        positions=positions,
        widths=violin_width_25,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    
    legend_handles = []
    for idx, (L, body) in enumerate(zip(layer_list, vp["bodies"])):
        color = cmap(idx / num_layers)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.20)
        body.set_linewidth(1.0)
        legend_handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.25, label=f"L{L}"))
    
    for k in ["cmins", "cmaxes", "cbars", "cmedians"]:
        if k in vp:
            vp[k].set_linewidth(1.0)
            vp[k].set_color("black")
            vp[k].set_alpha(0.7)
    
    plt.xticks(ticks=positions, labels=[str(L) for L in layer_list])
    plt.xlabel("Number of Layers")
    plt.ylabel(r"$d_\theta(\theta_{\mathrm{final}}, \theta_{\mathrm{ref}})$  (RMS wrapped)")
    set_prx_title(f"Final RMS Wrapped Parameter Distance over {num_runs} runs (ref = best-run per layer)")
    plt.grid(True, axis="y", alpha=0.3)
    if SHOW_REDUNDANT_LAYER_LEGENDS:
        plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left")
    
    save_current_figure(
        os.path.join(energy_fig_dir, "final_theta_wrapped_rmsdist_vs_layers.pdf"),
        outside_legend=SHOW_REDUNDANT_LAYER_LEGENDS,
    )
    
    
    # ------------------------------------------------------------
    # Plot 3: Success Rate
    # ------------------------------------------------------------
    new_prx_figure(width="double")
    
    layers3 = np.array(final_stats["layer"], dtype=NP_REAL_DTYPE)
    success_rates3 = np.array(final_stats["success_rate"], dtype=NP_REAL_DTYPE)
    
    plt.plot(
        layers3,
        success_rates3,
        marker="o",
        linestyle="-",
        label="Success Rate",
    )
    
    plt.xlabel("Number of Layers")
    plt.ylabel("Success Rate")
    set_prx_title(f"Success Rate (Tol={tolerance})")
    plt.xticks(layers3)
    plt.ylim(-0.02, 1.02)
    
    plt.grid(True, alpha=0.3)
    save_current_figure(
        os.path.join(energy_fig_dir, "success_rate.pdf"),
        outside_legend=False,
    )
    
    
    # ============================================================

def run_random_qfim_analysis(
    *,
    make_plots: bool = False,
    analysis_batch_size: Optional[int] = None,
) -> None:
    """Compute random-point matrix metrics with fixed-size JAX batches."""
    global KEEP_WIRES, QFIM_EFFECTIVE_RANK_THRESHOLD, EIG_SUM_EPS, QFIM_EIG_PLOT_EPS, NUM_QFIM_SAMPLES, QFIM_SAMPLE_SEED_BASE
    global PURE_QFIM_LAYER_THRESHOLD, RED_JVP_CHUNK, qfim_rank_pure_by_layer, qfim_rank_reduced_by_layer, qfim_random_thetas_by_layer, qfim_eigs_pure_by_layer
    global qfim_eigs_reduced_by_layer, qfim_thresh_pure_by_layer, qfim_thresh_reduced_by_layer, qfim_fig_dir, qfim_eigs_dir, qfim_eigs_pure_dir, qfim_eigs_reduced_0123_dir
    global qfim_rank_dir, qfim_rank_random_dir
    global hs_rank_reduced_by_layer, hs_eigs_reduced_by_layer, hs_thresh_reduced_by_layer, hs_eigs_dir, hs_eigs_reduced_0123_dir
    global hs_rank_dir, hs_rank_random_dir
    global hessian_rank_by_layer, hessian_condition_by_layer
    global qfim_random_result_paths_by_keep
    global qfim_dense_until_layer, qfim_max_layer, qfim_sparse_step
    global qfim_layer_list
    effective_analysis_batch_size = _resolve_analysis_batch_size(
        analysis_batch_size
    )
    qfim_dense_until_layer = cfg.UNITARY_PQC_QFIM_DENSE_UNTIL_LAYER
    qfim_max_layer = cfg.UNITARY_PQC_QFIM_MAX_LAYER
    qfim_sparse_step = cfg.UNITARY_PQC_QFIM_SPARSE_STEP
    qfim_layer_list = build_layer_list(
        qfim_max_layer,
        qfim_dense_until_layer,
        qfim_sparse_step,
    )
    if not qfim_layer_list:
        raise ValueError(
            "qfim_layer_list is empty. Check "
            "UNITARY_PQC_QFIM_MAX_LAYER, "
            "UNITARY_PQC_QFIM_DENSE_UNTIL_LAYER, and "
            "UNITARY_PQC_QFIM_SPARSE_STEP."
        )
    # QFIM rank (pure + reduced)
    #   - evaluated at RANDOM points in parameter space (per layer)
    #
    # Reduced(mixed):
    #   - keep=(0,1,2,3): trace out the added center ancilla qubit 4
    #   - rho_keep_sequential_unitary_pqc returns the 16x16 reduced system state
    #   - d rho / dﾎｸ via linearize + chunked JVPs
    #
    # keep01234 / pure(full):
    #   - Computed on the full 5-qubit pure state for every configured layer.
    #   - No wire is traced, so the pure-state QFIM is the efficient SLD-QFIM.
    #
    # ============================================================
    
    # ------------------------------
    # Subsystem to keep for reduced QFIM
    # ------------------------------
    # We keep the original 4-qubit system and trace out the center ancilla qubit.
    KEEP_WIRES = SYSTEM_WIRES
    assert KEEP_WIRES == (0, 1, 2, 3), "Reduced QFIM keeps only the original system qubits."
    
    # ------------------------------
    # Rank / numerical knobs
    # ------------------------------
    QFIM_EFFECTIVE_RANK_THRESHOLD = cfg.QFIM_EFFECTIVE_RANK_THRESHOLD
    EIG_SUM_EPS = cfg.EIG_SUM_EPS
    
    # For eigenvalue plots (log safety; DISPLAY ONLY for true zeros)
    QFIM_EIG_PLOT_EPS = cfg.QFIM_EIG_PLOT_EPS
    
    # ------------------------------
    # Random sampling knobs
    # ------------------------------
    NUM_QFIM_SAMPLES = cfg.NUM_QFIM_SAMPLES
    QFIM_SAMPLE_SEED_BASE = cfg.UNITARY_PQC_QFIM_SAMPLE_SEED_BASE
    
    # ------------------------------
    # Pure QFIM compute cutoff (kept)
    # ------------------------------
    # Compute pure-full metrics for every configured layer.  The previous
    # cutoff produced an asymmetric pure/reduced result set.
    PURE_QFIM_LAYER_THRESHOLD = max(qfim_layer_list, default=0) + 1
    
    # ------------------------------
    # Reduced-QFIM derivative chunk size
    # ------------------------------
    RED_JVP_CHUNK = cfg.RED_JVP_CHUNK
    
    
    
    
    
    
    # ------------------------------
    # Pure(full) QFIM matrix function on all 5 qubits
    # ------------------------------
    
    
    # ------------------------------
    # Reduced(mixed) QFIM matrix function
    #   - Propagate full 5-qubit state.
    #   - Trace out the center ancilla.
    #   - Build SLD-QFIM from the reduced 4-qubit system density matrix.
    # ------------------------------
    
    
    # Backward-compatible wrapper (kept signature)
    
    
    
    
    # ============================================================
    # Compute QFIM ranks + eigenvalue distributions at RANDOM samples per layer
    # ============================================================
    qfim_rank_pure_by_layer = {}        # L -> (NUM_QFIM_SAMPLES,) or None
    qfim_rank_reduced_by_layer = {}     # L -> (NUM_QFIM_SAMPLES,) for keep=(0..3)
    qfim_random_thetas_by_layer = {}
    
    qfim_eigs_pure_by_layer = {}        # L -> (NUM_QFIM_SAMPLES, num_params) or None
    qfim_eigs_reduced_by_layer = {}     # L -> (NUM_QFIM_SAMPLES, num_params)
    # Raw clipped spectra feed the canonical keep archives.  The dictionaries
    # above retain the historical threshold-masked representation.
    qfim_eigs_pure_raw_by_layer = {}
    qfim_eigs_reduced_raw_by_layer = {}
    
    # fixed thresholds used in rank computation
    qfim_thresh_pure_by_layer = {}      # L -> (NUM_QFIM_SAMPLES,) or None
    qfim_thresh_reduced_by_layer = {}   # L -> (NUM_QFIM_SAMPLES,)
    hs_rank_reduced_by_layer = {}       # L -> (NUM_QFIM_SAMPLES,)
    hs_eigs_reduced_by_layer = {}       # L -> (NUM_QFIM_SAMPLES, num_params)
    hs_thresh_reduced_by_layer = {}     # L -> (NUM_QFIM_SAMPLES,)
    hs_rank_pure_by_layer = {}
    hs_eigs_pure_by_layer = {}
    hs_thresh_pure_by_layer = {}
    hessian_rank_by_layer = {}             # L -> (NUM_QFIM_SAMPLES,)
    hessian_condition_by_layer = {}        # L -> (NUM_QFIM_SAMPLES,)
    
    qfim_eigs_dir = os.path.join(qfim_fig_dir, "eigs")
    qfim_eigs_pure_dir = os.path.join(qfim_eigs_dir, "pure_full")
    qfim_eigs_reduced_0123_dir = os.path.join(qfim_eigs_dir, "reduced_keep_0123")
    qfim_rank_dir = os.path.join(qfim_fig_dir, "rank")
    qfim_rank_random_dir = os.path.join(qfim_rank_dir, "random_points")
    hs_eigs_dir = os.path.join(hs_fig_dir, "eigs")
    hs_eigs_reduced_0123_dir = os.path.join(hs_eigs_dir, "reduced_keep_0123")
    hs_rank_dir = os.path.join(hs_fig_dir, "rank")
    hs_rank_random_dir = os.path.join(hs_rank_dir, "random_points")
    
    os.makedirs(qfim_rank_random_dir, exist_ok=True)
    os.makedirs(hs_eigs_dir, exist_ok=True)
    os.makedirs(hs_eigs_reduced_0123_dir, exist_ok=True)
    os.makedirs(hs_rank_random_dir, exist_ok=True)
    _ensure_unitary_result_dirs()
    
    # tqdm: Layers (QFIM)
    for L in tqdm(qfim_layer_list, desc="Layers (QFIM)", unit="layer"):
        num_params = num_params_per_layer * L
    
        key_L = jax.random.PRNGKey(QFIM_SAMPLE_SEED_BASE + int(L))
        thetas_L = jax.random.uniform(
            key_L,
            shape=(NUM_QFIM_SAMPLES, num_params),
            minval=-jnp.pi,
            maxval=jnp.pi,
            dtype=REAL_DTYPE,
        )
        qfim_random_thetas_by_layer[L] = np.asarray(jax.device_get(thetas_L), dtype=NP_REAL_DTYPE)
    
        # --------------------------
        # Reduced QFIM (keep 0..3)
        # --------------------------
        red_qfim_batch_runner = make_qfim_analysis_batch_runner(
            num_layers=L,
            keep_wires=KEEP_WIRES,
            jvp_chunk=RED_JVP_CHUNK,
            representation="reduced",
        )
        (
            reduced_qfim_ranks,
            reduced_qfim_eigs_masked,
            reduced_qfim_eigs_raw,
            reduced_qfim_thresholds,
        ) = _evaluate_analysis_in_batches(
            qfim_random_thetas_by_layer[L],
            red_qfim_batch_runner,
            batch_size=effective_analysis_batch_size,
            description=(
                f"Reduced QFIM batches (L={L}, "
                f"batch={effective_analysis_batch_size})"
            ),
        )
        qfim_rank_reduced_by_layer[L] = np.asarray(
            reduced_qfim_ranks,
            dtype=NP_INT_DTYPE,
        )
        qfim_eigs_reduced_by_layer[L] = np.asarray(
            reduced_qfim_eigs_masked,
            dtype=NP_REAL_DTYPE,
        )
        qfim_eigs_reduced_raw_by_layer[L] = np.asarray(
            reduced_qfim_eigs_raw,
            dtype=NP_REAL_DTYPE,
        )
        qfim_thresh_reduced_by_layer[L] = np.asarray(
            reduced_qfim_thresholds,
            dtype=NP_REAL_DTYPE,
        )

        del red_qfim_batch_runner
        _release_jax_compilation_cache()
    
        # --------------------------
        # Hilbert-Schmidt tangent Gram matrix (keep 0..3)
        #   G_ij = Re Tr[(partial_i rho)(partial_j rho)]
        #   computed via the equivalent Frobenius form after Hermitian symmetrization.
        # --------------------------
        red_hs_batch_runner = make_hs_analysis_batch_runner(
            num_layers=L,
            keep_wires=KEEP_WIRES,
            jvp_chunk=RED_JVP_CHUNK,
            representation="reduced",
        )
        (
            reduced_hs_ranks,
            reduced_hs_eigs_masked,
            _,
            reduced_hs_thresholds,
        ) = _evaluate_analysis_in_batches(
            qfim_random_thetas_by_layer[L],
            red_hs_batch_runner,
            batch_size=effective_analysis_batch_size,
            description=(
                f"Reduced HS batches (L={L}, "
                f"batch={effective_analysis_batch_size})"
            ),
        )
        hs_rank_reduced_by_layer[L] = np.asarray(
            reduced_hs_ranks,
            dtype=NP_INT_DTYPE,
        )
        hs_eigs_reduced_by_layer[L] = np.asarray(
            reduced_hs_eigs_masked,
            dtype=NP_REAL_DTYPE,
        )
        hs_thresh_reduced_by_layer[L] = np.asarray(
            reduced_hs_thresholds,
            dtype=NP_REAL_DTYPE,
        )

        del red_hs_batch_runner
        _release_jax_compilation_cache()

        # For a normalized pure state, G_HS = F_Q / 2 exactly.  Evaluate the
        # pure QFIM once in batches and derive both pure result families from
        # its spectrum, avoiding a duplicate state-Jacobian calculation.
        pure_qfim_batch_runner = make_qfim_analysis_batch_runner(
            num_layers=L,
            jvp_chunk=RED_JVP_CHUNK,
            representation="pure_full",
        )
        (
            pure_qfim_ranks,
            pure_qfim_eigs_masked,
            pure_qfim_eigs_raw,
            pure_qfim_thresholds,
        ) = _evaluate_analysis_in_batches(
            qfim_random_thetas_by_layer[L],
            pure_qfim_batch_runner,
            batch_size=effective_analysis_batch_size,
            description=(
                f"Pure(full) QFIM/HS batches (L={L}, "
                f"batch={effective_analysis_batch_size})"
            ),
        )
        if L >= PURE_QFIM_LAYER_THRESHOLD:
            qfim_rank_pure_by_layer[L] = None
            qfim_eigs_pure_by_layer[L] = None
            qfim_eigs_pure_raw_by_layer[L] = None
            qfim_thresh_pure_by_layer[L] = None
        else:
            qfim_rank_pure_by_layer[L] = np.asarray(
                pure_qfim_ranks,
                dtype=NP_INT_DTYPE,
            )
            qfim_eigs_pure_by_layer[L] = np.asarray(
                pure_qfim_eigs_masked,
                dtype=NP_REAL_DTYPE,
            )
            qfim_eigs_pure_raw_by_layer[L] = np.asarray(
                pure_qfim_eigs_raw,
                dtype=NP_REAL_DTYPE,
            )
            qfim_thresh_pure_by_layer[L] = np.asarray(
                pure_qfim_thresholds,
                dtype=NP_REAL_DTYPE,
            )

        pure_hs_eigs_raw = 0.5 * np.asarray(
            pure_qfim_eigs_raw,
            dtype=NP_REAL_DTYPE,
        )
        pure_hs_thresholds = np.full(
            (pure_hs_eigs_raw.shape[0],),
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        )
        pure_hs_eigs_masked = np.where(
            pure_hs_eigs_raw > pure_hs_thresholds[:, None],
            pure_hs_eigs_raw,
            NP_REAL_DTYPE(0.0),
        )
        pure_hs_ranks = np.sum(
            pure_hs_eigs_raw > pure_hs_thresholds[:, None],
            axis=1,
        )
        hs_rank_pure_by_layer[L] = np.asarray(
            pure_hs_ranks,
            dtype=NP_INT_DTYPE,
        )
        hs_eigs_pure_by_layer[L] = np.asarray(
            pure_hs_eigs_masked,
            dtype=NP_REAL_DTYPE,
        )
        hs_thresh_pure_by_layer[L] = np.asarray(
            pure_hs_thresholds, dtype=NP_REAL_DTYPE
        )

        del pure_qfim_batch_runner
        _release_jax_compilation_cache()

        if make_plots:
            _save_qfim_eigs_violinplot_by_index(
                hs_eigs_reduced_by_layer[L],
                title=rf"HS tangent Gram eigenvalues at {NUM_QFIM_SAMPLES} random points (L={L})",
                outpath=os.path.join(hs_eigs_reduced_0123_dir, f"L{L}_reduced_0123.pdf"),
                rank_thresholds=hs_thresh_reduced_by_layer[L],
                ylabel="HS tangent Gram eigenvalue",
            )
            plot_style.save_eigenvalue_histograms_by_trial(
                hs_eigs_reduced_by_layer[L],
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

        # --------------------------
        # Energy Hessian
        #   H_ij = partial_i partial_j E(theta)
        #   Signed eigenvalues are reduced immediately to the two requested
        #   random-point metrics using the fixed inclusive QFIM cutoff.
        # --------------------------
        hessian_batch_runner = make_hessian_analysis_batch_runner(L)
        hessian_ranks, hessian_condition_numbers = _evaluate_analysis_in_batches(
            qfim_random_thetas_by_layer[L],
            hessian_batch_runner,
            batch_size=effective_analysis_batch_size,
            description=(
                f"Hessian batches (L={L}, "
                f"batch={effective_analysis_batch_size})"
            ),
        )
        hessian_rank_by_layer[L] = np.asarray(
            hessian_ranks,
            dtype=NP_INT_DTYPE,
        )
        hessian_condition_by_layer[L] = np.asarray(
            hessian_condition_numbers,
            dtype=NP_REAL_DTYPE,
        )

        del hessian_batch_runner
        _release_jax_compilation_cache()

    
    save_npz_result(
        os.path.join(qfim_results_dir, "qfim_random_points.npz"),
        h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
        num_qfim_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
        qfim_sample_seed_base=np.asarray(QFIM_SAMPLE_SEED_BASE, dtype=NP_INT_DTYPE),
        qfim_effective_rank_threshold=np.asarray(
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
        pure_qfim_layer_threshold=np.asarray(
            PURE_QFIM_LAYER_THRESHOLD,
            dtype=NP_INT_DTYPE,
        ),
        red_jvp_chunk=np.asarray(RED_JVP_CHUNK, dtype=NP_INT_DTYPE),
        analysis_batch_size=np.asarray(
            effective_analysis_batch_size,
            dtype=NP_INT_DTYPE,
        ),
        **{
            f"L{int(L)}_theta": arr
            for L, arr in qfim_random_thetas_by_layer.items()
        },
        **{
            f"L{int(L)}_rank_reduced": arr
            for L, arr in qfim_rank_reduced_by_layer.items()
        },
        **{
            f"L{int(L)}_eigs_reduced_desc": arr
            for L, arr in qfim_eigs_reduced_by_layer.items()
        },
        **{
            f"L{int(L)}_rank_threshold_reduced": arr
            for L, arr in qfim_thresh_reduced_by_layer.items()
        },
        **{
            f"L{int(L)}_rank_pure": arr
            for L, arr in qfim_rank_pure_by_layer.items()
            if arr is not None
        },
        **{
            f"L{int(L)}_eigs_pure_desc": arr
            for L, arr in qfim_eigs_pure_by_layer.items()
            if arr is not None
        },
        **{
            f"L{int(L)}_rank_threshold_pure": arr
            for L, arr in qfim_thresh_pure_by_layer.items()
            if arr is not None
        },
    )

    # Canonical per-keep archives mirror the DPQC core rank/eigenvalue/trace
    # naming.  The historical combined archive remains available to loaders.
    qfim_random_result_paths_by_keep = save_qfim_random_point_results_by_keep(
        layers=qfim_layer_list,
        theta_by_layer=qfim_random_thetas_by_layer,
        rank_keep0123_by_layer=qfim_rank_reduced_by_layer,
        eigs_keep0123_by_layer=qfim_eigs_reduced_raw_by_layer,
        threshold_keep0123_by_layer=qfim_thresh_reduced_by_layer,
        rank_keep01234_by_layer=qfim_rank_pure_by_layer,
        eigs_keep01234_by_layer=qfim_eigs_pure_raw_by_layer,
        threshold_keep01234_by_layer=qfim_thresh_pure_by_layer,
        analysis_batch_size=effective_analysis_batch_size,
    )

    save_npz_result(
        os.path.join(hs_results_dir, "hs_random_points_reduced_0123.npz"),
        h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
        num_hs_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
        hs_sample_seed_base=np.asarray(QFIM_SAMPLE_SEED_BASE, dtype=NP_INT_DTYPE),
        hs_effective_rank_threshold=np.asarray(
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
        red_jvp_chunk=np.asarray(RED_JVP_CHUNK, dtype=NP_INT_DTYPE),
        analysis_batch_size=np.asarray(
            effective_analysis_batch_size,
            dtype=NP_INT_DTYPE,
        ),
        **{
            f"L{int(L)}_theta": arr
            for L, arr in qfim_random_thetas_by_layer.items()
        },
        **{
            f"L{int(L)}_rank": arr
            for L, arr in hs_rank_reduced_by_layer.items()
        },
        **{
            f"L{int(L)}_eigs_desc": arr
            for L, arr in hs_eigs_reduced_by_layer.items()
        },
        **{
            f"L{int(L)}_rank_threshold": arr
            for L, arr in hs_thresh_reduced_by_layer.items()
        },
    )

    save_npz_result(
        os.path.join(hs_results_dir, "hs_random_points_pure_full.npz"),
        h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
        num_hs_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
        hs_sample_seed_base=np.asarray(QFIM_SAMPLE_SEED_BASE, dtype=NP_INT_DTYPE),
        hs_effective_rank_threshold=np.asarray(QFIM_EFFECTIVE_RANK_THRESHOLD, dtype=NP_REAL_DTYPE),
        layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
        representation=np.asarray("pure_full"),
        hs_implementation=np.asarray("pure_qfim_spectrum_over_2"),
        analysis_batch_size=np.asarray(
            effective_analysis_batch_size,
            dtype=NP_INT_DTYPE,
        ),
        **{f"L{int(L)}_rank": arr for L, arr in hs_rank_pure_by_layer.items()},
        **{f"L{int(L)}_eigs_desc": arr for L, arr in hs_eigs_pure_by_layer.items()},
        **{f"L{int(L)}_rank_threshold": arr for L, arr in hs_thresh_pure_by_layer.items()},
    )

    save_npz_result(
        os.path.join(hessian_results_dir, "hessian_random_points.npz"),
        schema_version=np.asarray(
            HESSIAN_RANDOM_SCHEMA_VERSION,
            dtype=NP_INT_DTYPE,
        ),
        analysis_kind=np.asarray("random_points"),
        ansatz=np.asarray(ANSATZ_NAME),
        measurement_outcome=np.asarray(
            MEASUREMENT_OUTCOME,
            dtype=NP_INT_DTYPE,
        ),
        h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
        layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
        num_hessian_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
        hessian_sample_seed_base=np.asarray(QFIM_SAMPLE_SEED_BASE, dtype=NP_INT_DTYPE),
        hessian_rank_threshold=np.asarray(
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        hessian_rank_definition=np.asarray(HESSIAN_RANK_DEFINITION),
        hessian_condition_number_definition=np.asarray(
            HESSIAN_CONDITION_NUMBER_DEFINITION
        ),
        num_params_per_layer=np.asarray(
            num_params_per_layer,
            dtype=NP_INT_DTYPE,
        ),
        analysis_batch_size=np.asarray(
            effective_analysis_batch_size,
            dtype=NP_INT_DTYPE,
        ),
        **{
            f"L{int(L)}_rank": arr
            for L, arr in hessian_rank_by_layer.items()
        },
        **{
            f"L{int(L)}_condition_number": arr
            for L, arr in hessian_condition_by_layer.items()
        },
    )

    if not make_plots:
        return

    # ============================================================
    # Plot: QFIM rank vs depth  (VIOLIN)
    #   - reduced 縺ｨ pure(full) 縺ｮ縺ｿ繧定｡ｨ遉ｺ
    #   - upper/lower bound 縺ｮ險育ｮ励・謠冗判縺ｯ陦後ｏ縺ｪ縺・
    # ============================================================
    
    new_prx_figure(width="double")
    ax = plt.gca()
    
    x_all = np.array(qfim_layer_list, dtype=NP_REAL_DTYPE)
    x_labels = [str(L) for L in qfim_layer_list]
    
    dx = 0.25
    violin_w_rank = 0.20
    num_layers = len(qfim_layer_list)
    
    # ------------------------------
    # Reduced keep (0..3)
    # ------------------------------
    for idx, L in enumerate(qfim_layer_list):
        color = cmap(idx / num_layers)
        pos_red = float(L) + dx
        red_dataset = _make_violin_ready(
            qfim_rank_reduced_by_layer[L],
            ensure_positive=False,
            tiny=1e-12,
        )
    
        vp_red = plt.violinplot(
            [red_dataset],
            positions=[pos_red],
            widths=violin_w_rank,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        _style_violin(
            vp_red,
            facecolor=color,
            edgecolor=color,
            alpha=0.12,
            linewidth=1.0,
            hatch="///",
            linecolor=color,
            linealpha=0.7,
        )
    
    # ------------------------------
    # Pure(full) (only where computed)
    # ------------------------------
    pure_layers = [
        L for L in qfim_layer_list if qfim_rank_pure_by_layer[L] is not None
    ]
    
    for L in pure_layers:
        idx = qfim_layer_list.index(L)
        color = cmap(idx / num_layers)
        pos_pure = float(L) - dx
        pure_dataset = _make_violin_ready(
            qfim_rank_pure_by_layer[L],
            ensure_positive=False,
            tiny=1e-12,
        )
    
        vp_pure = plt.violinplot(
            [pure_dataset],
            positions=[pos_pure],
            widths=violin_w_rank,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        _style_violin(
            vp_pure,
            facecolor=color,
            edgecolor=color,
            alpha=0.20,
            linewidth=1.0,
            linecolor=color,
            linealpha=0.7,
        )
    
    # ------------------------------
    # Axes & grid
    # ------------------------------
    ax.set_xticks(x_all)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(r"QFIM effective rank $(\lambda_k > 10^{-12})$")
    set_prx_title(rf"QFIM rank at {NUM_QFIM_SAMPLES} random points", ax=ax)
    ax.grid(True, axis="y", alpha=0.3)
    
    # ------------------------------
    # Legends
    # ------------------------------
    layer_handles = [
        Patch(
            facecolor=cmap(i / num_layers),
            edgecolor=cmap(i / num_layers),
            alpha=0.25,
            label=f"L{L}",
        )
        for i, L in enumerate(qfim_layer_list)
    ]
    
    type_handles = [
        Patch(facecolor="white", edgecolor="black", label="Pure(full)"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label=f"Reduced (keep={KEEP_WIRES})"),
    ]
    
    if SHOW_REDUNDANT_LAYER_LEGENDS:
        leg_layers = ax.legend(
            handles=layer_handles,
            bbox_to_anchor=(1.05, 1.0),
            loc="upper left",
            frameon=True,
        )
        ax.add_artist(leg_layers)
    
    ax.legend(
        handles=type_handles,
        loc="best",
        frameon=True,
        framealpha=0.9,
    )
    
    save_current_figure(
        os.path.join(qfim_rank_random_dir, "qfim_rank_violinplot_random_points.pdf"),
        outside_legend=SHOW_REDUNDANT_LAYER_LEGENDS,
    )
    
    
    # ============================================================
    # Plot: Maximum QFIM rank vs layer
    #   - Uses already-computed QFIM rank dictionaries.
    #   - No additional QFIM computation is performed here.
    #   - Saves separate figures for:
    #       * pure_full
    #       * reduced_0123
    #   - Upper/lower bound lines are not drawn.
    # ============================================================
    
    
    
    
    plot_qfim_rank_max_by_layer(
        qfim_rank_pure_by_layer,
        qfim_layer_list,
        color="C0",
        title=rf"Maximum pure full-state QFIM rank at {NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum QFIM effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(qfim_rank_random_dir, "qfim_rank_max_random_points_pure_full.pdf"),
        marker="s",
        lw=1.0,
    )
    
    
    plot_qfim_rank_max_by_layer(
        qfim_rank_reduced_by_layer,
        qfim_layer_list,
        color="C0",
        title=rf"Maximum QFIM rank at {NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum QFIM effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(qfim_rank_random_dir, "qfim_rank_max_random_points_reduced_0123.pdf"),
        marker="o",
        lw=1.0,
    )

    new_prx_figure(width="double")
    ax = plt.gca()

    for idx, L in enumerate(qfim_layer_list):
        color = cmap(idx / num_layers)
        hs_dataset = _make_violin_ready(
            hs_rank_reduced_by_layer[L],
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
        _style_violin(
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
    set_prx_title(
        rf"HS tangent Gram rank at {NUM_QFIM_SAMPLES} random points",
        ax=ax,
    )
    ax.grid(True, axis="y", alpha=0.3)

    save_current_figure(
        os.path.join(hs_rank_random_dir, "hs_rank_violinplot_random_points_reduced_0123.pdf"),
        outside_legend=False,
    )

    plot_qfim_rank_max_by_layer(
        hs_rank_reduced_by_layer,
        qfim_layer_list,
        color="C3",
        title=rf"Maximum HS tangent Gram rank at {NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum HS effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(hs_rank_random_dir, "hs_rank_max_random_points_reduced_0123.pdf"),
        marker="D",
        lw=1.0,
    )

    # ============================================================

def run_optimization_path_qfim_analysis(
    *,
    make_plots: bool = False,
    analysis_batch_size: Optional[int] = None,
) -> None:
    """Compute trajectory matrix metrics with fixed-size JAX batches."""
    global qfim_rank_history_by_layer, qfim_eigs_history_by_layer, qfim_thresh_history_by_layer, qfim_rank_history_npz
    global hs_rank_history_by_layer, hs_eigs_history_by_layer, hs_thresh_history_by_layer, hs_rank_history_npz
    global qfim_optimization_path_result_paths_by_keep
    effective_analysis_batch_size = _resolve_analysis_batch_size(
        analysis_batch_size
    )
    # QFIM rank/eigenvalues along the Unitary-PQC VQE optimization path
    #   x-axis: sampled optimization iteration
    #   y-axis: run-mean/run-min QFIM effective rank at theta(iteration)
    #   color: layer number
    # ============================================================
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    qfim_rank_history_by_layer, qfim_eigs_history_by_layer, qfim_thresh_history_by_layer = (
        compute_qfim_rank_history_by_layer(
            theta_sample_traces_by_layer,
            layer_list,
            keep_wires=KEEP_WIRES,
            jvp_chunk=RED_JVP_CHUNK,
            batch_size=effective_analysis_batch_size,
        )
    )
    
    qfim_rank_history_npz = {
        "sample_iters": np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        "plot_iters": _qfim_history_plot_iterations(sample_iters),
        "layers": np.asarray(layer_list, dtype=NP_INT_DTYPE),
        "analysis_batch_size": np.asarray(
            effective_analysis_batch_size,
            dtype=NP_INT_DTYPE,
        ),
    }
    qfim_rank_history_npz.update(
        {
            f"L{int(L)}_rank": arr
            for L, arr in qfim_rank_history_by_layer.items()
        }
    )
    qfim_rank_history_npz.update(
        {
            f"L{int(L)}_eigs": arr
            for L, arr in qfim_eigs_history_by_layer.items()
        }
    )
    qfim_rank_history_npz.update(
        {
            f"L{int(L)}_rank_threshold": arr
            for L, arr in qfim_thresh_history_by_layer.items()
        }
    )
    
    save_npz_result(
        os.path.join(qfim_results_dir, "qfim_rank_history_optimization_path_reduced_0123.npz"),
        **qfim_rank_history_npz,
    )

    qfim_rank_history_pure, qfim_eigs_history_pure, qfim_thresh_history_pure = (
        compute_qfim_rank_history_by_layer(
            theta_sample_traces_by_layer,
            layer_list,
            jvp_chunk=RED_JVP_CHUNK,
            representation="pure_full",
            batch_size=effective_analysis_batch_size,
        )
    )
    save_npz_result(
        os.path.join(qfim_results_dir, "qfim_rank_history_optimization_path_pure_full.npz"),
        sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        plot_iters=_qfim_history_plot_iterations(sample_iters),
        layers=np.asarray(layer_list, dtype=NP_INT_DTYPE),
        representation=np.asarray("pure_full"),
        analysis_batch_size=np.asarray(
            effective_analysis_batch_size,
            dtype=NP_INT_DTYPE,
        ),
        **{f"L{int(L)}_rank": arr for L, arr in qfim_rank_history_pure.items()},
        **{f"L{int(L)}_eigs": arr for L, arr in qfim_eigs_history_pure.items()},
        **{f"L{int(L)}_rank_threshold": arr for L, arr in qfim_thresh_history_pure.items()},
    )

    # Save DPQC-style rank/eigenvalue/trace archives for each kept state.  The
    # existing reduced_0123 and pure_full archives above remain compatibility
    # views over the same in-memory arrays.
    qfim_optimization_path_result_paths_by_keep = (
        save_qfim_optimization_path_results_by_keep(
            layers=layer_list,
            sample_iterations=sample_iters,
            rank_keep0123_by_layer=qfim_rank_history_by_layer,
            eigs_keep0123_by_layer=qfim_eigs_history_by_layer,
            threshold_keep0123_by_layer=qfim_thresh_history_by_layer,
            rank_keep01234_by_layer=qfim_rank_history_pure,
            eigs_keep01234_by_layer=qfim_eigs_history_pure,
            threshold_keep01234_by_layer=qfim_thresh_history_pure,
            analysis_batch_size=effective_analysis_batch_size,
        )
    )

    hs_rank_history_by_layer, hs_eigs_history_by_layer, hs_thresh_history_by_layer = (
        compute_hs_rank_history_by_layer(
            theta_sample_traces_by_layer,
            layer_list,
            keep_wires=KEEP_WIRES,
            jvp_chunk=RED_JVP_CHUNK,
            batch_size=effective_analysis_batch_size,
        )
    )

    hs_rank_history_npz = {
        "sample_iters": np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        "plot_iters": _qfim_history_plot_iterations(sample_iters),
        "layers": np.asarray(layer_list, dtype=NP_INT_DTYPE),
        "analysis_batch_size": np.asarray(
            effective_analysis_batch_size,
            dtype=NP_INT_DTYPE,
        ),
    }
    hs_rank_history_npz.update(
        {
            f"L{int(L)}_rank": arr
            for L, arr in hs_rank_history_by_layer.items()
        }
    )
    hs_rank_history_npz.update(
        {
            f"L{int(L)}_eigs": arr
            for L, arr in hs_eigs_history_by_layer.items()
        }
    )
    hs_rank_history_npz.update(
        {
            f"L{int(L)}_rank_threshold": arr
            for L, arr in hs_thresh_history_by_layer.items()
        }
    )

    save_npz_result(
        os.path.join(hs_results_dir, "hs_rank_history_optimization_path_reduced_0123.npz"),
        **hs_rank_history_npz,
    )

    # Reuse the already-batched pure QFIM spectra: G_HS = F_Q / 2.
    hs_eigs_history_pure = {
        int(L): 0.5 * np.asarray(eigs, dtype=NP_REAL_DTYPE)
        for L, eigs in qfim_eigs_history_pure.items()
    }
    hs_thresh_history_pure = {
        int(L): np.full(
            np.asarray(eigs).shape[:2],
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        )
        for L, eigs in hs_eigs_history_pure.items()
    }
    hs_rank_history_pure = {
        int(L): np.sum(
            np.asarray(eigs, dtype=NP_REAL_DTYPE)
            > hs_thresh_history_pure[int(L)][..., None],
            axis=-1,
            dtype=NP_INT_DTYPE,
        ).astype(NP_REAL_DTYPE)
        for L, eigs in hs_eigs_history_pure.items()
    }
    save_npz_result(
        os.path.join(hs_results_dir, "hs_rank_history_optimization_path_pure_full.npz"),
        sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        plot_iters=_qfim_history_plot_iterations(sample_iters),
        layers=np.asarray(layer_list, dtype=NP_INT_DTYPE),
        representation=np.asarray("pure_full"),
        hs_implementation=np.asarray("pure_qfim_spectrum_over_2"),
        analysis_batch_size=np.asarray(
            effective_analysis_batch_size,
            dtype=NP_INT_DTYPE,
        ),
        **{f"L{int(L)}_rank": arr for L, arr in hs_rank_history_pure.items()},
        **{f"L{int(L)}_eigs": arr for L, arr in hs_eigs_history_pure.items()},
        **{f"L{int(L)}_rank_threshold": arr for L, arr in hs_thresh_history_pure.items()},
    )

    if not make_plots:
        return

    plot_qfim_rank_history_mean_by_layer(
        qfim_rank_history_by_layer,
        layer_list,
        sample_iters,
        title="Mean QFIM effective rank along optimization path (keep=(0,1,2,3))",
        outpath=os.path.join(
            qfim_rank_optimization_path_mean_dir,
            "qfim_rank_mean_history_optimization_path_reduced_0123.pdf",
        ),
        cmap=cmap,
    )
    
    plot_qfim_rank_history_min_by_layer(
        qfim_rank_history_by_layer,
        layer_list,
        sample_iters,
        title="Minimum QFIM effective rank along optimization path (keep=(0,1,2,3))",
        outpath=os.path.join(
            qfim_rank_optimization_path_min_dir,
            "qfim_rank_min_history_optimization_path_reduced_0123.pdf",
        ),
        cmap=cmap,
    )

    plot_qfim_rank_history_mean_by_layer(
        hs_rank_history_by_layer,
        layer_list,
        sample_iters,
        title="Mean HS tangent Gram effective rank along optimization path (keep=(0,1,2,3))",
        outpath=os.path.join(
            hs_rank_optimization_path_mean_dir,
            "hs_rank_mean_history_optimization_path_reduced_0123.pdf",
        ),
        ylabel=r"Mean HS effective rank $(\lambda_k > 10^{-12})$",
        cmap=cmap,
    )

    plot_qfim_rank_history_min_by_layer(
        hs_rank_history_by_layer,
        layer_list,
        sample_iters,
        title="Minimum HS tangent Gram effective rank along optimization path (keep=(0,1,2,3))",
        outpath=os.path.join(
            hs_rank_optimization_path_min_dir,
            "hs_rank_min_history_optimization_path_reduced_0123.pdf",
        ),
        ylabel=r"Minimum HS effective rank $(\lambda_k > 10^{-12})$",
        cmap=cmap,
    )

    # ============================================================

def collect_unitary_pqc_result() -> dict:
    """Return the compact summary shown by the notebook after execution."""
    return {
        "ansatz": ANSATZ_NAME,
        "measurement_outcome": MEASUREMENT_OUTCOME,
        "num_params_per_layer": num_params_per_layer,
        "save_dir": save_dir,
        "figures_dir": figures_dir,
        "energy_fig_dir": energy_fig_dir,
        "qfim_fig_dir": qfim_fig_dir,
        "hs_fig_dir": hs_fig_dir,
        "hessian_fig_dir": hessian_fig_dir,
        "circuit_dir": circuit_dir,
        "numerical_results_dir": numerical_results_dir,
        "energy_results_dir": energy_results_dir,
        "qfim_results_dir": qfim_results_dir,
        "hs_results_dir": hs_results_dir,
        "hessian_results_dir": hessian_results_dir,
        "qfim_keep_keys": (QFIM_KEEP0123_KEY, QFIM_KEEP01234_KEY),
        "qfim_random_result_paths_by_keep": qfim_random_result_paths_by_keep,
        "qfim_optimization_path_result_paths_by_keep": (
            qfim_optimization_path_result_paths_by_keep
        ),
        "h_param": h_param,
        "vqe_batch_size": VQE_BATCH_SIZE,
        "analysis_batch_size": ANALYSIS_BATCH_SIZE,
        "layer_list": layer_list,
        "qfim_layer_list": qfim_layer_list,
        "sample_iters": sample_iters,
        "smallest_eigval": smallest_eigval,
    }


def run_unitary_pqc_vqe_stage(
    *,
    h_param: Optional[float] = None,
    vqe_batch_size: Optional[int] = None,
) -> str:
    """Run only VQE and return the archive used by later processes."""
    configure_unitary_pqc_overparam(h_value=h_param)
    run_vqe_optimization(vqe_batch_size=vqe_batch_size)
    return os.path.join(energy_results_dir, "vqe_optimization_results.npz")


def run_unitary_pqc_qfim_stage(
    *,
    h_param: Optional[float] = None,
    analysis_batch_size: Optional[int] = None,
) -> dict:
    """Load saved VQE samples and run every post-VQE numerical analysis."""
    effective_analysis_batch_size = _resolve_analysis_batch_size(
        analysis_batch_size
    )
    configure_unitary_pqc_overparam(h_value=h_param)
    vqe_input_path = load_unitary_vqe_samples()
    print(
        "Loaded saved float64 VQE samples for post-VQE analyses: "
        f"{vqe_input_path}",
        flush=True,
    )
    run_random_qfim_analysis(
        make_plots=False,
        analysis_batch_size=effective_analysis_batch_size,
    )
    run_optimization_path_qfim_analysis(
        make_plots=False,
        analysis_batch_size=effective_analysis_batch_size,
    )
    result = collect_unitary_pqc_result()
    result["analysis_batch_size"] = effective_analysis_batch_size
    result["vqe_input_path"] = vqe_input_path
    return result


def run_unitary_pqc_overparam(
    *,
    h_param: Optional[float] = None,
    vqe_batch_size: Optional[int] = None,
    analysis_batch_size: Optional[int] = None,
) -> dict:
    """Run all computations in-process for the existing Python API.

    The command-line entry point uses the isolated VQE/QFIM subprocess stages
    instead.  This function remains in-process so callers that rely on its
    populated module globals and returned dictionary keep their old behavior.
    """
    effective_analysis_batch_size = _resolve_analysis_batch_size(
        analysis_batch_size
    )
    configure_unitary_pqc_overparam(h_value=h_param)
    run_vqe_optimization(vqe_batch_size=vqe_batch_size)
    run_random_qfim_analysis(
        make_plots=False,
        analysis_batch_size=effective_analysis_batch_size,
    )
    run_optimization_path_qfim_analysis(
        make_plots=False,
        analysis_batch_size=effective_analysis_batch_size,
    )
    result = collect_unitary_pqc_result()
    result["analysis_batch_size"] = effective_analysis_batch_size
    return result


def _run_selected_cli_stage(args: argparse.Namespace) -> int:
    """Execute a worker stage after the numerical dependencies are imported."""
    print(f"Selected Hamiltonian parameter h: {args.h_param}", flush=True)
    if args.stage == "vqe":
        archive_path = run_unitary_pqc_vqe_stage(
            h_param=args.h_param,
            vqe_batch_size=args.vqe_batch_size,
        )
        print(f"Hamiltonian parameter h: {h_param}")
        print(f"VQE batch size: {args.vqe_batch_size}")
        print(f"Saved VQE numerical results to: {archive_path}")
        print(
            "Run --stage qfim to refresh QFIM/HS/Hessian results from this "
            "archive."
        )
        return 0

    if args.stage == "qfim":
        run_unitary_pqc_qfim_stage(
            h_param=args.h_param,
            analysis_batch_size=args.analysis_batch_size,
        )
        print(f"Hamiltonian parameter h: {h_param}")
        print(f"Analysis batch size: {args.analysis_batch_size}")
        print(
            "Saved keep0123 and keep01234 QFIM numerical results to: "
            f"{qfim_results_dir}"
        )
        print(
            "Circuit drawing was skipped. To draw the saved optimized "
            "circuits, run "
            "src/unitary_pqc/"
            "unitary_pqc_measured_0_overparam_draw_circuits.py "
            "separately."
        )
        return 0

    raise ValueError(f"Unsupported worker stage: {args.stage!r}")


if __name__ == "__main__":
    raise SystemExit(_run_selected_cli_stage(_CLI_ARGS))


