#!/usr/bin/env python
# coding: utf-8
"""Unitary-PQC overparameterization numerical pipeline.

This module was extracted from unitary_pqc_overparam.ipynb so the notebook can
call a Python function instead of carrying the full optimization/QFIM program
inline. Plot generation is handled by unitary_pqc_overparam_visualize.py.
"""
from __future__ import annotations

import os
import sys
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
import tensorcircuit as tc
from matplotlib.patches import Patch
from tqdm.auto import tqdm as _tqdm

import config_overparam as cfg
import jax
import plot as plot_style
from dpqc_overparam_common import (
    _normalize_index_list,
    _time_index_from_iteration,
    qfim_grad_alignment_at_point,
    qfim_grad_alignment_one_to_table,
    qg_layer,
    rho_zero_state,
    save_circuit_matplotlib_png,
    U_rz,
)
from hamiltonian import (
    PAULI,
    build_H_matrix_jax,
    hamiltonian_terms,
    local_term_to_matrix,
)
from qfim import (
    effective_rank_from_eigvals,
    effective_abs_rank_from_eigvals,
    hermitian as _hermitian,
    hermitian_eigvals_desc,
    make_hilbert_schmidt_metric_fn,
    make_mixed_state_qfim_fn,
    make_pure_state_qfim_fn,
    mask_psd_eigvals_for_rank,
    matrix_rank_psd,
    psd_eigvals_desc,
    participation_effective_rank_from_eigvals,
    rank_threshold_from_eigvals,
)

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def tqdm(*args, **kwargs):
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("dynamic_ncols", True)
    return _tqdm(*args, **kwargs)


jit = jax.jit

REAL_DTYPE = jnp.float64
COMPLEX_DTYPE = jnp.complex128
NP_REAL_DTYPE = np.float64
NP_COMPLEX_DTYPE = np.complex128
NP_INT_DTYPE = np.int64


def _unitary_pqc_save_dir(h_value: float) -> str:
    """Return the Unitary-PQC output directory independently of process CWD."""
    return str(
        _PROJECT_ROOT
        / "figs"
        / "unitary_pqc"
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

h_param = cfg.H_PARAM
tolerance = cfg.TOLERANCE
steps = cfg.STEPS
num_runs = cfg.NUM_RUNS
lr = cfg.LEARNING_RATE

NUM_BLOCKS = 4
PARAMS_PER_BLOCK = 3
num_params_per_layer = NUM_BLOCKS * PARAMS_PER_BLOCK
LAYER_PAIRS = (
    (2, 3),
    (0, 2),
    (1, 3),
    (0, ANCILLA_QUBIT),
)

dense_until_layer = cfg.UNITARY_PQC_DENSE_UNTIL_LAYER
max_layer = cfg.UNITARY_PQC_MAX_LAYER
sparse_step = cfg.UNITARY_PQC_SPARSE_STEP
dense_end = min(dense_until_layer, max_layer)
layer_list = list(range(1, dense_end + 1))
if max_layer > dense_end:
    layer_list += list(range(dense_end + sparse_step, max_layer + 1, sparse_step))

save_dir = _unitary_pqc_save_dir(h_param)
figures_dir = os.path.join(save_dir, "figures")
energy_fig_dir = os.path.join(figures_dir, "energy")
qfim_fig_dir = os.path.join(figures_dir, "qfim")
hs_fig_dir = os.path.join(figures_dir, "hs")
ortk_fig_dir = os.path.join(figures_dir, "ortk")
hessian_fig_dir = os.path.join(figures_dir, "hessian")
circuit_dir = os.path.join(save_dir, "optimized_circuits")
numerical_results_dir = os.path.join(save_dir, "numerical_results")
energy_results_dir = os.path.join(numerical_results_dir, "energy")
qfim_results_dir = os.path.join(numerical_results_dir, "qfim")
hs_results_dir = os.path.join(numerical_results_dir, "hs")
ortk_results_dir = os.path.join(numerical_results_dir, "ortk")
hessian_results_dir = os.path.join(numerical_results_dir, "hessian")

sample_every = cfg.SAMPLE_EVERY
sample_iters = np.asarray([], dtype=NP_INT_DTYPE)
sample_iter_set = set()

KEEP_WIRES = SYSTEM_WIRES
QFIM_EFFECTIVE_RANK_THRESHOLD = cfg.QFIM_EFFECTIVE_RANK_THRESHOLD
EIG_SUM_EPS = cfg.EIG_SUM_EPS
QFIM_EIG_PLOT_EPS = cfg.QFIM_EIG_PLOT_EPS
NUM_QFIM_SAMPLES = cfg.NUM_QFIM_SAMPLES
QFIM_SAMPLE_SEED_BASE = cfg.UNITARY_PQC_QFIM_SAMPLE_SEED_BASE
PURE_QFIM_LAYER_THRESHOLD = cfg.PURE_QFIM_LAYER_THRESHOLD
RED_JVP_CHUNK = cfg.RED_JVP_CHUNK
ORTK_RANK_THRESHOLD = cfg.ORTK_RANK_THRESHOLD
ORTK_PARTICIPATION_EPS = cfg.ORTK_PARTICIPATION_EPS

QFIM_GRAD_ALIGN_EIG_FLOOR = 1e-16
QFIM_GRAD_ALIGN_WEIGHT_FLOOR = 1e-16

key = None
H_terms = ()
PAULI = {}
H_matrix = None
H_OBSERVABLE_MATRICES = None
smallest_eigval = None
X2 = None
_RHO_FULL_INIT = None
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
grad_sample_traces_by_layer = {}

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
hs_rank_reduced_by_layer = {}
hs_eigs_reduced_by_layer = {}
hs_thresh_reduced_by_layer = {}
hs_rank_history_by_layer = {}
hs_eigs_history_by_layer = {}
hs_thresh_history_by_layer = {}
ortk_rank_by_layer = {}
ortk_effective_rank_by_layer = {}
ortk_eigs_by_layer = {}
ortk_trace_by_layer = {}
ortk_rank_history_by_layer = {}
ortk_effective_rank_history_by_layer = {}
ortk_eigs_history_by_layer = {}
ortk_trace_history_by_layer = {}
hessian_rank_by_layer = {}
hessian_eigs_by_layer = {}
hessian_thresh_by_layer = {}
hessian_trace_by_layer = {}
hessian_abs_eigsum_by_layer = {}
hessian_rank_history_by_layer = {}
hessian_eigs_history_by_layer = {}
hessian_thresh_history_by_layer = {}
hessian_trace_history_by_layer = {}
hessian_abs_eigsum_history_by_layer = {}
qfim_grad_alignment_table_by_layer_iteration = {}

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
ortk_eigs_dir = os.path.join(ortk_fig_dir, "eigs")
ortk_rank_dir = os.path.join(ortk_fig_dir, "rank")
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
ortk_effective_rank_dir = os.path.join(ortk_fig_dir, "effective_rank")
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
ortk_trace_dir = os.path.join(ortk_fig_dir, "trace")
ortk_trace_optimization_path_dir = os.path.join(
    ortk_trace_dir,
    "optimization_path",
)
hessian_eigs_dir = os.path.join(hessian_fig_dir, "eigs")
hessian_rank_dir = os.path.join(hessian_fig_dir, "rank")
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
qfim_grad_align_dir = os.path.join(qfim_fig_dir, "grad_alignment")
qfim_grad_align_results_dir = os.path.join(qfim_results_dir, "grad_alignment")


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
    os.makedirs(ortk_fig_dir, exist_ok=True)
    os.makedirs(hessian_fig_dir, exist_ok=True)
    os.makedirs(qfim_eigs_dir, exist_ok=True)
    os.makedirs(qfim_eigs_pure_dir, exist_ok=True)
    os.makedirs(qfim_eigs_reduced_0123_dir, exist_ok=True)
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
    os.makedirs(ortk_trace_dir, exist_ok=True)
    os.makedirs(ortk_trace_optimization_path_dir, exist_ok=True)
    os.makedirs(hessian_eigs_dir, exist_ok=True)
    os.makedirs(hessian_rank_dir, exist_ok=True)
    os.makedirs(hessian_rank_random_dir, exist_ok=True)
    os.makedirs(hessian_rank_optimization_path_dir, exist_ok=True)
    os.makedirs(hessian_rank_optimization_path_mean_dir, exist_ok=True)
    os.makedirs(hessian_rank_optimization_path_min_dir, exist_ok=True)
    os.makedirs(qfim_grad_align_dir, exist_ok=True)
    os.makedirs(circuit_dir, exist_ok=True)
    os.makedirs(numerical_results_dir, exist_ok=True)
    os.makedirs(energy_results_dir, exist_ok=True)
    os.makedirs(qfim_results_dir, exist_ok=True)
    os.makedirs(hs_results_dir, exist_ok=True)
    os.makedirs(ortk_results_dir, exist_ok=True)
    os.makedirs(hessian_results_dir, exist_ok=True)
    os.makedirs(qfim_grad_align_results_dir, exist_ok=True)


def save_npz_result(outpath: str, **arrays) -> None:
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    np.savez(outpath, **arrays)


def create_unitary_pqc(theta: jnp.ndarray, num_layers: int, num_qubits: int) -> tc.Circuit:
    """
    5-qubit circuit with one central ancilla.

    Qubits:
      - System qubits : 0,1,2,3
      - Center ancilla: 4

    One layer consists of four qg_layer blocks:
      1. (2,3)
      2. (0,2)
      3. (1,3)
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

    qc = tc.Circuit(num_qubits_int)

    cursor = [0]

    def grab(n: int):
        seg = theta[cursor[0] : cursor[0] + n]
        cursor[0] += n
        return seg

    for _layer_idx in range(int(num_layers)):
        for q0, q1 in LAYER_PAIRS:
            p = grab(PARAMS_PER_BLOCK)
            qg_layer(qc, q0, q1, p)

    return qc

@jit
def wrap_to_pi(x: jnp.ndarray) -> jnp.ndarray:
    two_pi = jnp.array(2.0 * jnp.pi, dtype=x.dtype)
    return (x + jnp.pi) % two_pi - jnp.pi

def U_rxx(theta: jnp.ndarray) -> jnp.ndarray:
    th = jnp.asarray(theta, dtype=REAL_DTYPE)
    c = jnp.cos(0.5 * th).astype(COMPLEX_DTYPE)
    s = jnp.sin(0.5 * th).astype(COMPLEX_DTYPE)
    XX = jnp.kron(X2, X2)
    return c * jnp.eye(4, dtype=COMPLEX_DTYPE) - 1j * s * XX

def apply_unitary_on_rho(rho: jnp.ndarray, U: jnp.ndarray, wires, k: int) -> jnp.ndarray:
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

def rho_full_sequential_unitary_pqc(theta: jnp.ndarray, num_layers: int) -> jnp.ndarray:
    """
    Full 5-qubit propagation returning rho_full with shape 32x32.
    """
    k = num_total_qubits

    # shape: (num_layers, 12)
    theta_layers = jnp.reshape(theta, (num_layers, num_params_per_layer))

    def one_layer(rho: jnp.ndarray, layer_theta: jnp.ndarray):
        # 4 blocks: three original lattice blocks + one center-ancilla block.
        blocks = jnp.reshape(layer_theta, (NUM_BLOCKS, PARAMS_PER_BLOCK))

        for bi, (q0, q1) in enumerate(LAYER_PAIRS):
            p = blocks[bi]
            rho = apply_unitary_on_rho(rho, U_rz(p[0]), (q0,), k)
            rho = apply_unitary_on_rho(rho, U_rz(p[1]), (q1,), k)
            rho = apply_unitary_on_rho(rho, U_rxx(p[2]), (q0, q1), k)

        return rho, None

    rho_final, _ = jax.lax.scan(one_layer, _RHO_FULL_INIT, theta_layers)
    return _hermitian(rho_final)

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
    rho_full = rho_full_sequential_unitary_pqc(theta, num_layers=num_layers)
    rho_keep = partial_trace_keep(
        rho_full,
        keep_wires=keep_wires,
        num_qubits=num_total_qubits,
    )
    return _hermitian(rho_keep)

@jit
def energy_from_rho_full(rho_full: jnp.ndarray) -> jnp.ndarray:
    """
    Energy evaluated as Tr[rho_full * (H_system 竓・I_ancilla)] on 5 qubits.
    """
    e = jnp.einsum("ij,ji->", rho_full, H_matrix)
    return jnp.real(e)

def make_energy_fn_for_layer(num_layers: int):
    def energy_fn(theta: jnp.ndarray) -> jnp.ndarray:
        rho_full = rho_full_sequential_unitary_pqc(theta, num_layers=num_layers)
        return energy_from_rho_full(rho_full)

    return energy_fn

_make_violin_ready = plot_style.make_violin_ready
_style_violin = plot_style.style_violin

threshold_psd_eigvals_for_rank = mask_psd_eigvals_for_rank
_matrix_rank_psd = matrix_rank_psd

def make_pure_qfim_matrix_fn_for_layer(num_layers: int):
    def psi_fn(theta: jnp.ndarray) -> jnp.ndarray:
        c = create_unitary_pqc(theta, num_layers=num_layers, num_qubits=num_total_qubits)
        return c.wavefunction()

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
    """HS tangent Gram matrix of the full five-qubit pure-state density matrix."""
    @jax.jit
    def rho_full_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return _hermitian(
            rho_full_sequential_unitary_pqc(theta, num_layers=num_layers)
        )

    return make_hilbert_schmidt_metric_fn(
        rho_full_fn,
        jvp_chunk=jvp_chunk,
    )

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

def _save_signed_eigs_scatterplot_by_index(
    eigs_sorted_desc: np.ndarray,
    *,
    title: str,
    outpath: str,
    rank_thresholds: Optional[np.ndarray] = None,
    eps: float = QFIM_EIG_PLOT_EPS,
    ylabel: str = "Hessian eigenvalue",
    point_size: float = 12.0,
    alpha: float = 0.55,
) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    eigs = np.asarray(eigs_sorted_desc, dtype=NP_REAL_DTYPE)
    if eigs.ndim == 1:
        eigs = eigs[None, :]

    num_params = int(eigs.shape[1])
    thresholds = np.asarray(
        [] if rank_thresholds is None else rank_thresholds,
        dtype=NP_REAL_DTYPE,
    ).reshape(-1)
    thresholds = thresholds[np.isfinite(thresholds) & (thresholds > 0.0)]
    threshold = (
        float(np.max(thresholds))
        if thresholds.size > 0
        else float(QFIM_EFFECTIVE_RANK_THRESHOLD)
    )
    linthresh = max(float(eps), threshold)

    new_prx_figure(width="double")
    ax = plt.gca()

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

    step = max(1, num_params // 10)
    ticks = np.arange(1, num_params + 1, step, dtype=int)
    if ticks.size == 0 or ticks[0] != 1:
        ticks = np.concatenate([[1], ticks])
    if ticks[-1] != num_params:
        ticks = np.concatenate([ticks, [num_params]])
    plt.xticks(ticks=ticks, labels=[str(t) for t in ticks])

    ax.set_yscale("symlog", linthresh=linthresh)
    ax.axhline(0.0, linestyle="-", linewidth=0.8, color="black", alpha=0.45)

    if threshold > 0.0:
        ax.axhline(threshold, linestyle="-", linewidth=1.2, color="red", alpha=0.75)
        ax.axhline(-threshold, linestyle="-", linewidth=1.2, color="red", alpha=0.75)

    finite_vals = eigs[np.isfinite(eigs)]
    if finite_vals.size > 0:
        max_abs = max(float(np.max(np.abs(finite_vals))), threshold, linthresh)
        ax.set_ylim(-1.5 * max_abs, 1.5 * max_abs)

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


def plot_scalar_violin_by_layer(
    values_by_layer: dict,
    layers,
    *,
    title: str,
    ylabel: str,
    outpath: str,
    integer_y_axis: bool = False,
):
    valid_layers = [
        int(L)
        for L in layers
        if values_by_layer.get(int(L)) is not None
    ]
    if not valid_layers:
        return

    new_prx_figure(width="double")
    ax = plt.gca()
    num_layers = len(valid_layers)

    for index, L in enumerate(valid_layers):
        color = cmap(index / max(1, num_layers))
        dataset = _make_violin_ready(
            values_by_layer[L],
            ensure_positive=False,
            tiny=1e-12,
        )
        violin = ax.violinplot(
            [dataset],
            positions=[float(L)],
            widths=0.35,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        _style_violin(
            violin,
            facecolor=color,
            edgecolor=color,
            alpha=0.18,
            linewidth=1.0,
            linecolor=color,
            linealpha=0.7,
        )

    ax.set_xticks(valid_layers)
    ax.set_xticklabels([str(L) for L in valid_layers])
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    set_prx_title(title, ax=ax)
    if integer_y_axis:
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


def make_observable_expectation_vector_fn_for_layer(num_layers: int):
    """Return the weighted Hamiltonian-term expectation vector."""

    @jax.jit
    def observable_expectation_vector(theta: jnp.ndarray) -> jnp.ndarray:
        rho_keep = rho_keep_sequential_unitary_pqc(
            theta,
            num_layers=num_layers,
            keep_wires=KEEP_WIRES,
        )
        values = jnp.einsum("aij,ji->a", H_OBSERVABLE_MATRICES, rho_keep)
        return jnp.real(values)

    return observable_expectation_vector


def make_observable_tangent_kernel_matrix_fn_for_layer(num_layers: int):
    """Return K_obs(theta) = J_obs(theta) J_obs(theta)^T."""
    observable_fn = make_observable_expectation_vector_fn_for_layer(num_layers)
    observable_jacobian_fn = jax.jacrev(observable_fn)

    @jax.jit
    def observable_tangent_kernel(theta: jnp.ndarray) -> jnp.ndarray:
        jacobian = observable_jacobian_fn(theta)
        kernel = jacobian @ jacobian.T
        return 0.5 * (kernel + kernel.T)

    return observable_tangent_kernel


def make_observable_tangent_kernel_eigvals_fn_for_layer(num_layers: int):
    ortk_fn = make_observable_tangent_kernel_matrix_fn_for_layer(num_layers)

    @jax.jit
    def observable_tangent_kernel_eigvals(theta: jnp.ndarray) -> jnp.ndarray:
        return psd_eigvals_desc(ortk_fn(theta))

    return observable_tangent_kernel_eigvals


def make_ortk_rank_effective_eigvals_fn_for_layer(num_layers: int):
    ortk_eigvals_fn = make_observable_tangent_kernel_eigvals_fn_for_layer(
        num_layers,
    )

    @jax.jit
    def ortk_rank_effective_eigvals(theta: jnp.ndarray):
        eigs_desc = ortk_eigvals_fn(theta)
        rank_value = effective_rank_from_eigvals(
            eigs_desc,
            threshold=ORTK_RANK_THRESHOLD,
        )
        effective_rank_value = participation_effective_rank_from_eigvals(
            eigs_desc,
            eps=ORTK_PARTICIPATION_EPS,
        )
        return rank_value, effective_rank_value, eigs_desc

    return ortk_rank_effective_eigvals

def make_energy_hessian_eigvals_fn_for_layer(num_layers: int):
    energy_fn = make_energy_fn_for_layer(num_layers)
    hessian_fn = jax.jit(jax.hessian(energy_fn))

    @jit
    def hessian_eigvals(theta: jnp.ndarray):
        return hermitian_eigvals_desc(hessian_fn(theta))

    return hessian_eigvals

def compute_qfim_rank_history_by_layer(
    theta_samples_by_layer: dict,
    layers,
    *,
    keep_wires=KEEP_WIRES,
    jvp_chunk: int = RED_JVP_CHUNK,
    representation: str = "reduced",
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
        if representation == "pure_full":
            matrix_fn = make_pure_qfim_matrix_fn_for_layer(L_int)
            eigvals_fn = jax.jit(lambda theta: psd_eigvals_desc(matrix_fn(theta)))
        elif representation == "reduced":
            eigvals_fn = make_qfim_eigvals_fn_for_layer(
                num_layers=L_int, keep_wires=keep_wires, jvp_chunk=jvp_chunk
            )
        else:
            raise ValueError("representation must be 'pure_full' or 'reduced'.")

        ranks_L = np.full((num_runs, num_times), np.nan, dtype=NP_REAL_DTYPE)
        eigs_L = np.full((num_runs, num_times, num_params), np.nan, dtype=NP_REAL_DTYPE)
        thresh_L = np.full((num_runs, num_times), np.nan, dtype=NP_REAL_DTYPE)

        for run_idx in tqdm(
            range(num_runs),
            desc=f"QFIM-rank runs (L={L_int})",
            unit="run",
            leave=False,
        ):
            for time_idx in range(num_times):
                eigs_desc = eigvals_fn(
                    jnp.asarray(theta_samples[run_idx, time_idx], dtype=REAL_DTYPE)
                )
                rank_value = effective_rank_from_eigvals(eigs_desc)
                thresh = rank_threshold_from_eigvals(eigs_desc)
                ranks_L[run_idx, time_idx] = NP_REAL_DTYPE(
                    jax.device_get(rank_value)
                )
                eigs_L[run_idx, time_idx, :] = np.asarray(
                    jax.device_get(eigs_desc),
                    dtype=NP_REAL_DTYPE,
                )
                thresh_L[run_idx, time_idx] = NP_REAL_DTYPE(
                    jax.device_get(thresh)
                )

        rank_history_by_layer[L_int] = ranks_L
        eigs_history_by_layer[L_int] = eigs_L
        thresh_history_by_layer[L_int] = thresh_L

    return rank_history_by_layer, eigs_history_by_layer, thresh_history_by_layer


def compute_ortk_rank_history_by_layer(
    theta_samples_by_layer: dict,
    layers,
):
    rank_history_by_layer = {}
    effective_rank_history_by_layer = {}
    eigs_history_by_layer = {}

    for L in tqdm(
        layers,
        desc="ORTK rank history along optimization path",
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

        num_runs, num_times, _ = theta_samples.shape
        ortk_metrics_fn = make_ortk_rank_effective_eigvals_fn_for_layer(L_int)
        num_observables = int(H_OBSERVABLE_MATRICES.shape[0])

        ranks_L = np.full((num_runs, num_times), np.nan, dtype=NP_REAL_DTYPE)
        effective_ranks_L = np.full(
            (num_runs, num_times),
            np.nan,
            dtype=NP_REAL_DTYPE,
        )
        eigs_L = np.full(
            (num_runs, num_times, num_observables),
            np.nan,
            dtype=NP_REAL_DTYPE,
        )

        for run_idx in tqdm(
            range(num_runs),
            desc=f"ORTK-rank runs (L={L_int})",
            unit="run",
            leave=False,
        ):
            for time_idx in range(num_times):
                rank_value, effective_rank_value, eigs_desc = ortk_metrics_fn(
                    jnp.asarray(
                        theta_samples[run_idx, time_idx],
                        dtype=REAL_DTYPE,
                    )
                )
                ranks_L[run_idx, time_idx] = NP_REAL_DTYPE(
                    jax.device_get(rank_value)
                )
                effective_ranks_L[run_idx, time_idx] = NP_REAL_DTYPE(
                    jax.device_get(effective_rank_value)
                )
                eigs_L[run_idx, time_idx, :] = np.asarray(
                    jax.device_get(eigs_desc),
                    dtype=NP_REAL_DTYPE,
                )

        rank_history_by_layer[L_int] = ranks_L
        effective_rank_history_by_layer[L_int] = effective_ranks_L
        eigs_history_by_layer[L_int] = eigs_L

    return (
        rank_history_by_layer,
        effective_rank_history_by_layer,
        eigs_history_by_layer,
    )

def compute_hs_rank_history_by_layer(
    theta_samples_by_layer: dict,
    layers,
    *,
    keep_wires=KEEP_WIRES,
    jvp_chunk: int = RED_JVP_CHUNK,
    representation: str = "reduced",
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
        if representation == "pure_full":
            matrix_fn = make_pure_full_hs_matrix_fn_for_layer(
                num_layers=L_int, jvp_chunk=jvp_chunk
            )
            eigvals_fn = jax.jit(lambda theta: psd_eigvals_desc(matrix_fn(theta)))
        elif representation == "reduced":
            eigvals_fn = make_hs_eigvals_fn_for_layer(
                num_layers=L_int, keep_wires=keep_wires, jvp_chunk=jvp_chunk
            )
        else:
            raise ValueError("representation must be 'pure_full' or 'reduced'.")

        ranks_L = np.full((num_runs, num_times), np.nan, dtype=NP_REAL_DTYPE)
        eigs_L = np.full((num_runs, num_times, num_params), np.nan, dtype=NP_REAL_DTYPE)
        thresh_L = np.full((num_runs, num_times), np.nan, dtype=NP_REAL_DTYPE)

        for run_idx in tqdm(
            range(num_runs),
            desc=f"HS-rank runs (L={L_int})",
            unit="run",
            leave=False,
        ):
            for time_idx in range(num_times):
                eigs_desc = eigvals_fn(
                    jnp.asarray(theta_samples[run_idx, time_idx], dtype=REAL_DTYPE)
                )
                rank_value = effective_rank_from_eigvals(eigs_desc)
                thresh = rank_threshold_from_eigvals(eigs_desc)
                ranks_L[run_idx, time_idx] = NP_REAL_DTYPE(
                    jax.device_get(rank_value)
                )
                eigs_L[run_idx, time_idx, :] = np.asarray(
                    jax.device_get(eigs_desc),
                    dtype=NP_REAL_DTYPE,
                )
                thresh_L[run_idx, time_idx] = NP_REAL_DTYPE(
                    jax.device_get(thresh)
                )

        rank_history_by_layer[L_int] = ranks_L
        eigs_history_by_layer[L_int] = eigs_L
        thresh_history_by_layer[L_int] = thresh_L

    return rank_history_by_layer, eigs_history_by_layer, thresh_history_by_layer

def compute_hessian_rank_history_by_layer(
    theta_samples_by_layer: dict,
    layers,
):
    rank_history_by_layer = {}
    eigs_history_by_layer = {}
    thresh_history_by_layer = {}

    for L in tqdm(
        layers,
        desc="Energy Hessian rank history along optimization path",
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
        eigvals_fn = make_energy_hessian_eigvals_fn_for_layer(num_layers=L_int)

        ranks_L = np.full((num_runs, num_times), np.nan, dtype=NP_REAL_DTYPE)
        eigs_L = np.full((num_runs, num_times, num_params), np.nan, dtype=NP_REAL_DTYPE)
        thresh_L = np.full((num_runs, num_times), np.nan, dtype=NP_REAL_DTYPE)

        for run_idx in tqdm(
            range(num_runs),
            desc=f"Energy Hessian-rank runs (L={L_int})",
            unit="run",
            leave=False,
        ):
            for time_idx in range(num_times):
                eigs_desc = eigvals_fn(
                    jnp.asarray(theta_samples[run_idx, time_idx], dtype=REAL_DTYPE)
                )
                rank_value = effective_abs_rank_from_eigvals(eigs_desc)
                thresh = rank_threshold_from_eigvals(eigs_desc)
                ranks_L[run_idx, time_idx] = NP_REAL_DTYPE(
                    jax.device_get(rank_value)
                )
                eigs_L[run_idx, time_idx, :] = np.asarray(
                    jax.device_get(eigs_desc),
                    dtype=NP_REAL_DTYPE,
                )
                thresh_L[run_idx, time_idx] = NP_REAL_DTYPE(
                    jax.device_get(thresh)
                )

        rank_history_by_layer[L_int] = ranks_L
        eigs_history_by_layer[L_int] = eigs_L
        thresh_history_by_layer[L_int] = thresh_L

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

def compute_qfim_grad_alignment_table_for_layer(
    L,
    theta_samples_by_layer,
    grad_samples_by_layer,
    *,
    run_indices=None,
    time_indices=None,
    sample_iters_for_labels=None,
    jvp_chunk=RED_JVP_CHUNK,
    sort_desc=True,
):
    theta_samples = np.asarray(theta_samples_by_layer[L], dtype=NP_REAL_DTYPE)
    grad_samples = np.asarray(grad_samples_by_layer[L], dtype=NP_REAL_DTYPE)

    if theta_samples.shape != grad_samples.shape:
        raise ValueError(
            f"theta and grad must have the same shape for L={L}. "
            f"Got {theta_samples.shape} and {grad_samples.shape}."
        )

    if theta_samples.ndim == 2:
        theta_samples = theta_samples[:, None, :]
        grad_samples = grad_samples[:, None, :]
    elif theta_samples.ndim != 3:
        raise ValueError(
            "theta and grad arrays must have shape "
            "(num_runs, num_sample_times, num_params) or "
            "(num_samples, num_params)."
        )

    num_runs, num_times, _ = theta_samples.shape
    run_ids = _normalize_index_list(run_indices, num_runs)
    time_ids = _normalize_index_list(time_indices, num_times)

    if sample_iters_for_labels is None:
        sample_iters_arr = np.arange(num_times, dtype=NP_INT_DTYPE)
    else:
        sample_iters_arr = np.asarray(sample_iters_for_labels, dtype=NP_INT_DTYPE)

    qfim_fn = make_reduced_qfim_matrix_fn_for_layer_sequential(
        num_layers=int(L),
        keep_wires=KEEP_WIRES,
        jvp_chunk=jvp_chunk,
    )

    rows = {
        "lambda": [],
        "w_grad": [],
        "coeff_abs2": [],
        "eig_index": [],
        "layer": [],
        "run": [],
        "time_index": [],
        "iteration": [],
    }

    for run_idx in tqdm(
        run_ids,
        desc=f"QFIM-gradient scatter data (L={L})",
        unit="run",
        leave=False,
    ):
        for time_idx in time_ids:
            iteration = (
                int(sample_iters_arr[time_idx])
                if time_idx < sample_iters_arr.size
                else int(time_idx)
            )

            alignment = qfim_grad_alignment_at_point(
                theta_samples[run_idx, time_idx],
                grad_samples[run_idx, time_idx],
                qfim_fn,
                sort_desc=sort_desc,
            )

            table_one = qfim_grad_alignment_one_to_table(
                alignment,
                layer=L,
                run=run_idx,
                time_index=time_idx,
                iteration=iteration,
            )

            for key in rows:
                rows[key].append(table_one[key])

    table = {}
    for key, values in rows.items():
        if key in ("lambda", "w_grad", "coeff_abs2"):
            table[key] = np.concatenate(values).astype(NP_REAL_DTYPE)
        else:
            table[key] = np.concatenate(values).astype(NP_INT_DTYPE)

    return table

def plot_qfim_grad_alignment_table(
    table,
    *,
    title,
    outpath,
    log_x=True,
    log_y=False,
    eig_floor=QFIM_GRAD_ALIGN_EIG_FLOOR,
    weight_floor=QFIM_GRAD_ALIGN_WEIGHT_FLOOR,
    color_by=None,
    point_size=14.0,
    alpha=0.45,
):
    lambdas = np.asarray(table["lambda"], dtype=NP_REAL_DTYPE)
    weights = np.asarray(table["w_grad"], dtype=NP_REAL_DTYPE)
    finite = (
        np.isfinite(lambdas)
        & np.isfinite(weights)
        & (lambdas >= 0.0)
        & (weights >= 0.0)
    )

    x = np.maximum(lambdas[finite], eig_floor)
    y_raw = weights[finite]
    y = np.maximum(y_raw, weight_floor) if log_y else y_raw

    new_prx_figure(width="double")
    ax = plt.gca()

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
        cbar = plt.gcf().colorbar(sc, ax=ax, pad=0.02)
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

    if log_x:
        ax.set_xscale("log")

    if log_y:
        ax.set_yscale("log")
    else:
        ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel(r"QFIM eigenvalue $\lambda_i$")
    ax.set_ylabel("Gradient coeff.")
    set_prx_title(title, ax=ax)
    ax.grid(True, which="both", alpha=0.30)

    save_current_figure(outpath, outside_legend=False)

def run_qfim_grad_alignment_by_layer_iteration_folders(
    *,
    layers=None,
    target_iterations=None,
    run_indices=None,
    sample_iters_for_labels=None,
    jvp_chunk=RED_JVP_CHUNK,
    log_x=True,
    log_y=False,
    save_npz=True,
    make_plots=True,
    data_dir=None,
    plot_dir=None,
):
    if layers is None:
        candidate_layers = layer_list
    else:
        candidate_layers = layers

    if sample_iters_for_labels is None:
        sample_iters_for_labels = sample_iters

    if target_iterations is None:
        target_iterations = tuple(int(t) for t in sample_iters_for_labels)
    else:
        target_iterations = tuple(int(t) for t in target_iterations)

    available_layers = []
    for L in candidate_layers:
        L = int(L)
        if L in theta_sample_traces_by_layer and L in grad_sample_traces_by_layer:
            available_layers.append(L)

    if not available_layers:
        raise ValueError(
            "No layers are available in theta_sample_traces_by_layer and "
            "grad_sample_traces_by_layer. Run the VQE optimization first."
        )

    data_root = qfim_grad_align_dir if data_dir is None else data_dir
    plot_root = qfim_grad_align_dir if plot_dir is None else plot_dir

    table_by_layer_iteration = {}
    for L in tqdm(
        available_layers,
        desc="QFIM eigenvalue-gradient scatter by layer/iteration",
        unit="layer",
    ):
        layer_data_dir = os.path.join(data_root, f"L{L}")
        layer_plot_dir = os.path.join(plot_root, f"L{L}")
        os.makedirs(layer_data_dir, exist_ok=True)
        if make_plots:
            os.makedirs(layer_plot_dir, exist_ok=True)
        table_by_layer_iteration[L] = {}

        for iteration in tqdm(
            target_iterations,
            desc=f"Iterations (L={L})",
            unit="iter",
            leave=False,
        ):
            time_idx = _time_index_from_iteration(
                sample_iters_for_labels,
                iteration,
            )
            table_L_iter = compute_qfim_grad_alignment_table_for_layer(
                L,
                theta_sample_traces_by_layer,
                grad_sample_traces_by_layer,
                run_indices=run_indices,
                time_indices=[time_idx],
                sample_iters_for_labels=sample_iters_for_labels,
                jvp_chunk=jvp_chunk,
                sort_desc=True,
            )
            table_by_layer_iteration[L][iteration] = table_L_iter

            iter_tag = f"iter{iteration:06d}"
            if save_npz:
                np.savez(
                    os.path.join(
                        layer_data_dir,
                        f"qfim_grad_alignment_scatter_data_L{L}_{iter_tag}.npz",
                    ),
                    **table_L_iter,
                )

            if make_plots:
                plot_qfim_grad_alignment_table(
                    table_L_iter,
                    title=(
                        rf"QFIM eigenvalue vs gradient weight, "
                        rf"L={L}, iteration {iteration}"
                    ),
                    outpath=os.path.join(
                        layer_plot_dir,
                        f"qfim_grad_weight_scatter_L{L}_{iter_tag}.pdf",
                    ),
                    log_x=log_x,
                    log_y=log_y,
                    color_by=None,
                    point_size=14.0,
                    alpha=0.45,
                )

    return table_by_layer_iteration

def configure_unitary_pqc_overparam() -> None:
    """Initialize runtime constants, backend settings, Hamiltonian, and initial state."""
    global REAL_DTYPE, COMPLEX_DTYPE, NP_REAL_DTYPE, NP_COMPLEX_DTYPE, NP_INT_DTYPE, INCH_PER_CM
    global FIGSIZE_SINGLE, FIGSIZE_DOUBLE, FIGURE_WIDTH_DEFAULT, SAVE_DPI, SAVEFIG_PAD_INCHES, SAVE_PNG
    global SAVE_PDF
    global CIRCUIT_SAVE_PDF, SHOW_FIGURE_TITLES, SHOW_REDUNDANT_LAYER_LEGENDS, BASE_FONT_SIZE, TITLE_FONT_SIZE, AXIS_LABEL_FONT_SIZE
    global TICK_LABEL_FONT_SIZE, LEGEND_FONT_SIZE, _DEFAULT_AXES_MARGINS_PRX, _DEFAULT_AXES_MARGINS_PRX_OUTSIDE_LEGEND, key, num_system_qubits
    global ANCILLA_QUBIT, num_total_qubits, SYSTEM_WIRES, FULL_WIRES, h_param, tolerance
    global steps, num_runs, lr, NUM_BLOCKS, PARAMS_PER_BLOCK, num_params_per_layer
    global LAYER_PAIRS, H_terms, PAULI, H_matrix, H_OBSERVABLE_MATRICES, eigvals_np, smallest_eigval
    global X2, _RHO_FULL_INIT
    global save_dir, figures_dir, energy_fig_dir, qfim_fig_dir, hs_fig_dir, ortk_fig_dir, hessian_fig_dir
    global circuit_dir, numerical_results_dir, energy_results_dir, qfim_results_dir
    global hs_results_dir, ortk_results_dir, hessian_results_dir
    global qfim_eigs_dir, qfim_eigs_pure_dir, qfim_eigs_reduced_0123_dir
    global qfim_rank_dir, qfim_rank_random_dir, qfim_rank_optimization_path_dir
    global qfim_rank_optimization_path_mean_dir, qfim_rank_optimization_path_min_dir
    global hs_eigs_dir, hs_eigs_reduced_0123_dir
    global hs_rank_dir, hs_rank_random_dir, hs_rank_optimization_path_dir
    global hs_rank_optimization_path_mean_dir, hs_rank_optimization_path_min_dir
    global ortk_eigs_dir, ortk_rank_dir, ortk_rank_random_dir
    global ortk_rank_optimization_path_dir
    global ortk_rank_optimization_path_mean_dir, ortk_rank_optimization_path_min_dir
    global ortk_effective_rank_dir, ortk_effective_rank_random_dir
    global ortk_effective_rank_optimization_path_dir
    global ortk_effective_rank_optimization_path_mean_dir
    global ortk_effective_rank_optimization_path_min_dir
    global ortk_trace_dir, ortk_trace_optimization_path_dir
    global hessian_eigs_dir, hessian_rank_dir, hessian_rank_random_dir
    global hessian_rank_optimization_path_dir
    global hessian_rank_optimization_path_mean_dir, hessian_rank_optimization_path_min_dir
    global qfim_grad_align_dir, qfim_grad_align_results_dir
    # ============================================================
    # DPQC optimization + plots + QFIM rank (pure + reduced)
    #
    # 5-QUBIT CENTER-ANCILLA VERSION:
    #   - Original physical system qubits are (0,1,2,3).
    #   - A central ancilla qubit is added as qubit 4.
    #   - Each layer contains 4 two-qubit blocks (NUM_BLOCKS=4):
    #       (2,3), (0,2), (1,3), (0,4)
    #     where (0,4) is the added system-ancilla block.
    #   - Each block applies Rz on both qubits and Rxx between them.
    #   - Per-layer parameter count:
    #       num_params_per_layer = NUM_BLOCKS * PARAMS_PER_BLOCK = 12
    #   - Density-matrix propagation is performed on all 5 qubits: 32x32 rho.
    #   - Hamiltonian acts nontrivially only on qubits 0..3 and as identity on
    #     the center ancilla qubit 4, i.e. H_total = H_system 竓・I_ancilla.
    #   - QFIM analyses:
    #       * Reduced (mixed-state) QFIM for keep=(0,1,2,3), tracing out ancilla 4
    #       * Pure(full) QFIM for the full 5-qubit state for small layers
    #   - Eigenvalue thresholding rule identical to DPQC_overparam.ipynb:
    #       thresh = 1e-12
    #     and eigenvalues <= thresh are zeroed BEFORE storage/plotting.
    #
    #   - NEW:
    #       * All box plots are replaced by violin plots.
    #       * For final_energy_error.pdf, an additional beeswarm version is saved as:
    #           final_energy_error_beeswarm.pdf
    #       * A log-scale final-energy-error violin plot is also saved as:
    #           final_energy_error_log.pdf
    #       * For each per-layer QFIM eigenvalue plot saved in qfim_eigs/,
    #         a horizontal red solid line is drawn at the fixed rank threshold.
    #       * pure_full and reduced_keep_0123 are saved into separate folders:
    #           qfim_eigs/pure_full/
    #           qfim_eigs/reduced_keep_0123/
    #
    #   - DTYPE UPDATE:
    #       * Unified to float64 / complex128
    #       * JAX x64 enabled
    #       * TensorCircuit dtype set to complex128
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
    tc.set_backend("jax")
    tc.set_dtype("complex128")
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
    
    h_param = cfg.H_PARAM
    tolerance = cfg.TOLERANCE
    steps = cfg.STEPS
    num_runs = cfg.NUM_RUNS
    lr = cfg.LEARNING_RATE  # Adam base lr
    
    save_dir = _unitary_pqc_save_dir(h_param)
    figures_dir = os.path.join(save_dir, "figures")
    energy_fig_dir = os.path.join(figures_dir, "energy")
    qfim_fig_dir = os.path.join(figures_dir, "qfim")
    hs_fig_dir = os.path.join(figures_dir, "hs")
    ortk_fig_dir = os.path.join(figures_dir, "ortk")
    hessian_fig_dir = os.path.join(figures_dir, "hessian")
    circuit_dir = os.path.join(save_dir, "optimized_circuits")
    numerical_results_dir = os.path.join(save_dir, "numerical_results")
    energy_results_dir = os.path.join(numerical_results_dir, "energy")
    qfim_results_dir = os.path.join(numerical_results_dir, "qfim")
    hs_results_dir = os.path.join(numerical_results_dir, "hs")
    ortk_results_dir = os.path.join(numerical_results_dir, "ortk")
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
    ortk_eigs_dir = os.path.join(ortk_fig_dir, "eigs")
    ortk_rank_dir = os.path.join(ortk_fig_dir, "rank")
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
    ortk_effective_rank_dir = os.path.join(ortk_fig_dir, "effective_rank")
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
    ortk_trace_dir = os.path.join(ortk_fig_dir, "trace")
    ortk_trace_optimization_path_dir = os.path.join(
        ortk_trace_dir,
        "optimization_path",
    )
    hessian_eigs_dir = os.path.join(hessian_fig_dir, "eigs")
    hessian_rank_dir = os.path.join(hessian_fig_dir, "rank")
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
    qfim_grad_align_dir = os.path.join(qfim_fig_dir, "grad_alignment")
    qfim_grad_align_results_dir = os.path.join(qfim_results_dir, "grad_alignment")
    _ensure_unitary_result_dirs()
    
    # Block structure constants
    # Existing lattice blocks: (2,3), (0,2), (1,3)
    # Added center-ancilla block: (0, ANCILLA_QUBIT)
    NUM_BLOCKS = 4
    PARAMS_PER_BLOCK = 3
    num_params_per_layer = NUM_BLOCKS * PARAMS_PER_BLOCK  # 12
    
    LAYER_PAIRS = (
        (2, 3),
        (0, 2),
        (1, 3),
        (0, ANCILLA_QUBIT),
    )
    
    # ==============================
    # Circuit blocks (TensorCircuit)
    # ==============================
    
    
    
    
    # ==============================
    # TC -> Qiskit (for drawing)
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
    H_OBSERVABLE_MATRICES = jnp.stack(
        [
            jnp.asarray(coef, dtype=REAL_DTYPE)
            * local_term_to_matrix(local_ops, num_system_qubits)
            for coef, local_ops in H_terms
        ],
        axis=0,
    )
    
    eigvals_np, _ = np.linalg.eigh(np.array(H_matrix, dtype=NP_COMPLEX_DTYPE))
    smallest_eigval = float(eigvals_np.min().real)
    
    # ==============================
    # Wrap
    # ==============================
    
    
    # ============================================================
    # 5-qubit density-matrix propagation
    #
    # The full state rho lives on 5 qubits: system (0,1,2,3) + ancilla (4),
    # hence rho has shape 32x32.
    #
    # Each layer applies the existing lattice blocks plus the added center-ancilla
    # block (0,4):
    #     (2,3), (0,2), (1,3), (0,4).
    #
    # The energy is evaluated as Tr[rho_full (H_system 竓・I_ancilla)].
    # For the reduced mixed-state QFIM on the original 4-qubit system,
    # rho_keep_sequential_unitary_pqc traces out the ancilla qubit.
    # ============================================================
    
    X2 = jnp.array([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Precompute initial 5-qubit |00000><00000| density matrix
    _RHO_FULL_INIT = rho_zero_state(num_total_qubits, dtype=COMPLEX_DTYPE)
    
    
    
    
    
    
    
    
    # ==============================

def run_vqe_optimization(*, save_circuits: bool = False) -> None:
    """Run VQE optimization for every configured layer and collect traces."""
    global success_rates_history, energy_mean_history, energy_std_history, final_stats, dense_until_layer, max_layer
    global sparse_step, dense_end, layer_list, save_dir, figures_dir, energy_fig_dir, qfim_fig_dir, hs_fig_dir, ortk_fig_dir, hessian_fig_dir, circuit_dir, optimizer
    global qfim_eigs_dir, qfim_eigs_pure_dir, qfim_eigs_reduced_0123_dir
    global qfim_rank_dir, qfim_rank_random_dir, qfim_rank_optimization_path_dir
    global qfim_rank_optimization_path_mean_dir, qfim_rank_optimization_path_min_dir
    global hs_eigs_dir, hs_eigs_reduced_0123_dir
    global hs_rank_dir, hs_rank_random_dir, hs_rank_optimization_path_dir
    global hs_rank_optimization_path_mean_dir, hs_rank_optimization_path_min_dir
    global hessian_eigs_dir, hessian_rank_dir, hessian_rank_random_dir
    global hessian_rank_optimization_path_dir
    global hessian_rank_optimization_path_mean_dir, hessian_rank_optimization_path_min_dir
    global theta_history, best_theta_by_layer, final_theta_wrapped_rmsdist_by_layer, energy_traces_by_layer, grad_norm_traces_by_layer, sample_every
    global sample_iters, sample_iter_set, theta_sample_traces_by_layer, grad_sample_traces_by_layer, cmap
    global numerical_results_dir, energy_results_dir, qfim_results_dir, hs_results_dir, ortk_results_dir, hessian_results_dir, qfim_grad_align_dir, qfim_grad_align_results_dir
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
    
    layer_list = list(range(1, dense_end + 1))
    if max_layer > dense_end:
        layer_list += list(range(dense_end + sparse_step, max_layer + 1, sparse_step))
    
    # --- Save dir for optimized circuit diagrams ---
    save_dir = _unitary_pqc_save_dir(h_param)
    figures_dir = os.path.join(save_dir, "figures")
    energy_fig_dir = os.path.join(figures_dir, "energy")
    qfim_fig_dir = os.path.join(figures_dir, "qfim")
    hs_fig_dir = os.path.join(figures_dir, "hs")
    ortk_fig_dir = os.path.join(figures_dir, "ortk")
    hessian_fig_dir = os.path.join(figures_dir, "hessian")
    circuit_dir = os.path.join(save_dir, "optimized_circuits")
    numerical_results_dir = os.path.join(save_dir, "numerical_results")
    energy_results_dir = os.path.join(numerical_results_dir, "energy")
    qfim_results_dir = os.path.join(numerical_results_dir, "qfim")
    hs_results_dir = os.path.join(numerical_results_dir, "hs")
    ortk_results_dir = os.path.join(numerical_results_dir, "ortk")
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
    hessian_eigs_dir = os.path.join(hessian_fig_dir, "eigs")
    hessian_rank_dir = os.path.join(hessian_fig_dir, "rank")
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
    qfim_grad_align_dir = os.path.join(qfim_fig_dir, "grad_alignment")
    qfim_grad_align_results_dir = os.path.join(qfim_results_dir, "grad_alignment")
    _ensure_unitary_result_dirs()
    
    optimizer = optax.adam(learning_rate=lr)
    
    theta_history = {L: [] for L in layer_list}  # final theta of each run
    best_theta_by_layer = {}
    
    # Final RMS wrapped parameter distance per layer (distribution over runs)
    #   Reference theta_ref(L): best-run final parameters at the same layer
    final_theta_wrapped_rmsdist_by_layer = {}  # L -> (num_runs,) array of d_theta(theta_final, theta_ref)
    
    energy_traces_by_layer = {}
    grad_norm_traces_by_layer = {}  # L -> (num_runs, steps) gradient-norm traces
    
    # Sampled optimization-time states and gradients used later for
    # QFIM-eigenbasis gradient-weight plots.
    sample_every = cfg.SAMPLE_EVERY
    sample_iters = np.arange(0, steps, sample_every, dtype=NP_INT_DTYPE)
    if sample_iters.size == 0 or sample_iters[0] != 0:
        sample_iters = np.concatenate([[0], sample_iters]).astype(NP_INT_DTYPE)
    
    if sample_iters[-1] != steps - 1:
        sample_iters = np.concatenate([sample_iters, [steps - 1]]).astype(NP_INT_DTYPE)
    
    sample_iters = np.unique(sample_iters).astype(NP_INT_DTYPE)
    sample_iter_set = set(int(t) for t in sample_iters.tolist())
    
    theta_sample_traces_by_layer = {}
    grad_sample_traces_by_layer = {}
    cmap = matplotlib.colormaps.get_cmap("viridis")
    
    # tqdm: Layers (VQE)
    for current_layer in tqdm(layer_list, desc="Layers (VQE)", unit="layer"):
        num_total_params = num_params_per_layer * current_layer
    
        # Energy function:
        #   - propagate rho on all 5 qubits
        #   - evaluate energy as Tr[rho_full * (H_system 竓・I_ancilla)]
        energy_fn = make_energy_fn_for_layer(current_layer)
        energy_and_grad = jax.jit(jax.value_and_grad(energy_fn))
    
        @jax.jit
        def optimization_step(theta, opt_state):
            e, g = energy_and_grad(theta)
            g_norm = jnp.linalg.norm(g)  # ||竏㍉theta E||_2
            updates, new_opt_state = optimizer.update(g, opt_state, theta)
            theta = optax.apply_updates(theta, updates)
            theta = wrap_to_pi(theta)
            return theta, new_opt_state, e, g, g_norm
    
        best_final_theta = None
        best_final_energy = np.inf
        all_energy_traces = []
        all_gradnorm_traces = []
        all_theta_sample_traces = []
        all_grad_sample_traces = []
    
        base_key = jax.random.PRNGKey(current_layer * 1000)
        keys = jax.random.split(base_key, num_runs)
    
        # tqdm: Runs
        for i in tqdm(range(num_runs), desc=f"Runs (L={current_layer})", unit="run", leave=False):
            key_i = keys[i]
            theta = jax.random.uniform(
                key_i,
                shape=(num_total_params,),
                minval=-jnp.pi,
                maxval=jnp.pi,
                dtype=REAL_DTYPE,
            )
            opt_state = optimizer.init(theta)
    
            trace = []
            grad_trace = []
            theta_sample_trace = []
            grad_sample_trace = []
    
            for step_idx in range(steps):
                theta_before_step = theta
                theta, opt_state, e, g, g_norm = optimization_step(theta, opt_state)
                trace.append(e)
                grad_trace.append(g_norm)
    
                if step_idx in sample_iter_set:
                    theta_sample_trace.append(
                        np.asarray(
                            jax.device_get(theta_before_step),
                            dtype=NP_REAL_DTYPE,
                        )
                    )
                    grad_sample_trace.append(
                        np.asarray(jax.device_get(g), dtype=NP_REAL_DTYPE)
                    )
    
            all_energy_traces.append(np.array(trace, dtype=NP_REAL_DTYPE))
            all_gradnorm_traces.append(np.array(grad_trace, dtype=NP_REAL_DTYPE))
            all_theta_sample_traces.append(
                np.asarray(theta_sample_trace, dtype=NP_REAL_DTYPE)
            )
            all_grad_sample_traces.append(
                np.asarray(grad_sample_trace, dtype=NP_REAL_DTYPE)
            )
            theta_history[current_layer].append(np.array(theta, dtype=NP_REAL_DTYPE))
    
            final_e = float(e)
            if final_e < best_final_energy:
                best_final_energy = final_e
                best_final_theta = np.array(theta, dtype=NP_REAL_DTYPE)
    
        theta_history[current_layer] = np.stack(theta_history[current_layer], axis=0)  # (num_runs, num_params)
    
        # Final RMS wrapped distance distribution over runs
        theta_runs_jnp = jnp.asarray(theta_history[current_layer], dtype=REAL_DTYPE)     # (num_runs, num_params)
        theta_ref_jnp = jnp.asarray(best_final_theta, dtype=REAL_DTYPE)[None, :]         # (1, num_params)
    
        wrapped_diff = wrap_to_pi(theta_runs_jnp - theta_ref_jnp)       # (num_runs, num_params)
        d_theta_runs = jnp.sqrt(jnp.mean(wrapped_diff ** 2, axis=1))    # (num_runs,)
    
        final_theta_wrapped_rmsdist_by_layer[current_layer] = np.asarray(
            jax.device_get(d_theta_runs), dtype=NP_REAL_DTYPE
        )
    
        energy_data = np.stack(all_energy_traces, axis=0)  # (num_runs, steps)
        energy_traces_by_layer[current_layer] = energy_data
    
        gradnorm_data = np.stack(all_gradnorm_traces, axis=0)  # (num_runs, steps)
        grad_norm_traces_by_layer[current_layer] = gradnorm_data
    
        theta_sample_data = np.stack(all_theta_sample_traces, axis=0)
        grad_sample_data = np.stack(all_grad_sample_traces, axis=0)
        theta_sample_traces_by_layer[current_layer] = theta_sample_data
        grad_sample_traces_by_layer[current_layer] = grad_sample_data
    
        best_theta_by_layer[current_layer] = best_final_theta.copy()
    
        if save_circuits:
            num_qubits_for_drawing = num_total_qubits
            best_tc_circ = create_unitary_pqc(
                jnp.asarray(best_final_theta, dtype=REAL_DTYPE),
                num_layers=current_layer,
                num_qubits=num_total_qubits,
            )
            out_png = os.path.join(circuit_dir, f"optimized_circuit_L{current_layer}.png")
            save_circuit_matplotlib_png(
                best_tc_circ,
                out_png,
                num_qubits=num_qubits_for_drawing,
                dpi=SAVE_DPI,
                pad_inches=SAVEFIG_PAD_INCHES,
                save_pdf=CIRCUIT_SAVE_PDF,
                hide_params=True,
            )
    
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

    save_unitary_vqe_results()
    
    
    # ==============================


def save_unitary_vqe_results() -> str:
    outpath = os.path.join(energy_results_dir, "vqe_optimization_results.npz")
    layers_arr = np.asarray(layer_list, dtype=NP_INT_DTYPE)

    save_npz_result(
        outpath,
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
        **{
            f"L{int(L)}_grad_samples": np.asarray(
                grad_sample_traces_by_layer[int(L)],
                dtype=NP_REAL_DTYPE,
            )
            for L in layer_list
        },
    )

    return outpath


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

def run_random_qfim_analysis(*, make_plots: bool = False) -> None:
    """Compute random-point QFIM ranks/eigenvalues and save QFIM summary plots."""
    global KEEP_WIRES, QFIM_EFFECTIVE_RANK_THRESHOLD, EIG_SUM_EPS, QFIM_EIG_PLOT_EPS, NUM_QFIM_SAMPLES, QFIM_SAMPLE_SEED_BASE
    global PURE_QFIM_LAYER_THRESHOLD, RED_JVP_CHUNK, qfim_rank_pure_by_layer, qfim_rank_reduced_by_layer, qfim_random_thetas_by_layer, qfim_eigs_pure_by_layer
    global qfim_eigs_reduced_by_layer, qfim_thresh_pure_by_layer, qfim_thresh_reduced_by_layer, qfim_fig_dir, qfim_eigs_dir, qfim_eigs_pure_dir, qfim_eigs_reduced_0123_dir
    global qfim_rank_dir, qfim_rank_random_dir
    global hs_rank_reduced_by_layer, hs_eigs_reduced_by_layer, hs_thresh_reduced_by_layer, hs_eigs_dir, hs_eigs_reduced_0123_dir
    global hs_rank_dir, hs_rank_random_dir
    global ORTK_RANK_THRESHOLD, ORTK_PARTICIPATION_EPS
    global ortk_rank_by_layer, ortk_effective_rank_by_layer, ortk_eigs_by_layer, ortk_trace_by_layer
    global ortk_eigs_dir, ortk_rank_dir, ortk_rank_random_dir
    global ortk_effective_rank_dir, ortk_effective_rank_random_dir
    global hessian_rank_by_layer, hessian_eigs_by_layer, hessian_thresh_by_layer, hessian_trace_by_layer, hessian_abs_eigsum_by_layer, hessian_eigs_dir
    global hessian_rank_dir, hessian_rank_random_dir
    # QFIM rank (pure + reduced) + eig plots
    #   - evaluated at RANDOM points in parameter space (per layer)
    #
    # Reduced(mixed):
    #   - keep=(0,1,2,3): trace out the added center ancilla qubit 4
    #   - rho_keep_sequential_unitary_pqc returns the 16x16 reduced system state
    #   - d rho / dﾎｸ via linearize + chunked JVPs
    #
    # Pure(full):
    #   - Computed on the full 5-qubit pure state only for L < PURE_QFIM_LAYER_THRESHOLD
    #
    # Eigenvalue plots:
    #   - eigenvalues <= thresh are set to 0.0 BEFORE plotting/storage
    #   - log-safety epsilon applied ONLY at plotting time
    #   - the fixed rank threshold 10^{-12} is overlaid as a horizontal line
    #     in each saved per-layer plot
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
    PURE_QFIM_LAYER_THRESHOLD = max(layer_list, default=0) + 1
    
    # ------------------------------
    # Reduced-QFIM derivative chunk size
    # ------------------------------
    RED_JVP_CHUNK = cfg.RED_JVP_CHUNK
    ORTK_RANK_THRESHOLD = cfg.ORTK_RANK_THRESHOLD
    ORTK_PARTICIPATION_EPS = cfg.ORTK_PARTICIPATION_EPS
    
    
    
    
    
    
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
    
    # fixed thresholds used in rank computation
    qfim_thresh_pure_by_layer = {}      # L -> (NUM_QFIM_SAMPLES,) or None
    qfim_thresh_reduced_by_layer = {}   # L -> (NUM_QFIM_SAMPLES,)
    hs_rank_reduced_by_layer = {}       # L -> (NUM_QFIM_SAMPLES,)
    hs_eigs_reduced_by_layer = {}       # L -> (NUM_QFIM_SAMPLES, num_params)
    hs_thresh_reduced_by_layer = {}     # L -> (NUM_QFIM_SAMPLES,)
    hs_rank_pure_by_layer = {}
    hs_eigs_pure_by_layer = {}
    hs_thresh_pure_by_layer = {}
    ortk_rank_by_layer = {}             # L -> (NUM_QFIM_SAMPLES,)
    ortk_effective_rank_by_layer = {}   # L -> (NUM_QFIM_SAMPLES,)
    ortk_eigs_by_layer = {}             # L -> (NUM_QFIM_SAMPLES, num_observables)
    ortk_trace_by_layer = {}            # L -> (NUM_QFIM_SAMPLES,)
    hessian_rank_by_layer = {}          # L -> (NUM_QFIM_SAMPLES,)
    hessian_eigs_by_layer = {}          # L -> (NUM_QFIM_SAMPLES, num_params)
    hessian_thresh_by_layer = {}        # L -> (NUM_QFIM_SAMPLES,)
    hessian_trace_by_layer = {}         # L -> (NUM_QFIM_SAMPLES,)
    hessian_abs_eigsum_by_layer = {}    # L -> (NUM_QFIM_SAMPLES,)
    
    qfim_eigs_dir = os.path.join(qfim_fig_dir, "eigs")
    qfim_eigs_pure_dir = os.path.join(qfim_eigs_dir, "pure_full")
    qfim_eigs_reduced_0123_dir = os.path.join(qfim_eigs_dir, "reduced_keep_0123")
    qfim_rank_dir = os.path.join(qfim_fig_dir, "rank")
    qfim_rank_random_dir = os.path.join(qfim_rank_dir, "random_points")
    hs_eigs_dir = os.path.join(hs_fig_dir, "eigs")
    hs_eigs_reduced_0123_dir = os.path.join(hs_eigs_dir, "reduced_keep_0123")
    hs_rank_dir = os.path.join(hs_fig_dir, "rank")
    hs_rank_random_dir = os.path.join(hs_rank_dir, "random_points")
    ortk_eigs_dir = os.path.join(ortk_fig_dir, "eigs")
    ortk_rank_dir = os.path.join(ortk_fig_dir, "rank")
    ortk_rank_random_dir = os.path.join(ortk_rank_dir, "random_points")
    ortk_effective_rank_dir = os.path.join(ortk_fig_dir, "effective_rank")
    ortk_effective_rank_random_dir = os.path.join(
        ortk_effective_rank_dir,
        "random_points",
    )
    hessian_eigs_dir = os.path.join(hessian_fig_dir, "eigs")
    hessian_rank_dir = os.path.join(hessian_fig_dir, "rank")
    hessian_rank_random_dir = os.path.join(hessian_rank_dir, "random_points")
    
    os.makedirs(qfim_eigs_dir, exist_ok=True)
    os.makedirs(qfim_eigs_pure_dir, exist_ok=True)
    os.makedirs(qfim_eigs_reduced_0123_dir, exist_ok=True)
    os.makedirs(qfim_rank_random_dir, exist_ok=True)
    os.makedirs(hs_eigs_dir, exist_ok=True)
    os.makedirs(hs_eigs_reduced_0123_dir, exist_ok=True)
    os.makedirs(hs_rank_random_dir, exist_ok=True)
    os.makedirs(ortk_eigs_dir, exist_ok=True)
    os.makedirs(ortk_rank_random_dir, exist_ok=True)
    os.makedirs(ortk_effective_rank_random_dir, exist_ok=True)
    os.makedirs(hessian_eigs_dir, exist_ok=True)
    os.makedirs(hessian_rank_random_dir, exist_ok=True)
    _ensure_unitary_result_dirs()
    
    # tqdm: Layers (QFIM)
    for L in tqdm(layer_list, desc="Layers (QFIM)", unit="layer"):
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
        red_qfim_fn = make_reduced_qfim_matrix_fn_for_layer_sequential(
            num_layers=L,
            keep_wires=KEEP_WIRES,
            jvp_chunk=RED_JVP_CHUNK,
        )
    
        rr_list = []
        eigs_list = []
        thresh_list = []
    
        for s in tqdm(
            range(NUM_QFIM_SAMPLES),
            desc=f"Reduced QFIM samples (rank+eigs) (L={L})",
            unit="sample",
            leave=False,
        ):
            th = thetas_L[s]
    
            F = red_qfim_fn(th)
            evals = jnp.linalg.eigvalsh(_hermitian(F))
            evals = jnp.clip(evals, a_min=0.0)
    
            evals_masked, thresh = threshold_psd_eigvals_for_rank(evals)
            r = jnp.sum(evals > thresh)
    
            rr_list.append(int(jax.device_get(r)))
            eigs_list.append(np.asarray(jax.device_get(evals_masked[::-1]), dtype=NP_REAL_DTYPE))
            thresh_list.append(float(jax.device_get(thresh)))
    
        qfim_rank_reduced_by_layer[L] = np.asarray(rr_list, dtype=int)
        qfim_eigs_reduced_by_layer[L] = np.stack(eigs_list, axis=0)
        qfim_thresh_reduced_by_layer[L] = np.asarray(thresh_list, dtype=NP_REAL_DTYPE)
    
        if make_plots:
            _save_qfim_eigs_violinplot_by_index(
                qfim_eigs_reduced_by_layer[L],
                title=rf"QFIM eigenvalues at {NUM_QFIM_SAMPLES} random points (L={L})",
                outpath=os.path.join(qfim_eigs_reduced_0123_dir, f"L{L}_reduced_0123.pdf"),
                rank_thresholds=qfim_thresh_reduced_by_layer[L],
            )
            plot_style.save_eigenvalue_histograms_by_trial(
                qfim_eigs_reduced_by_layer[L],
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

        # --------------------------
        # Hilbert-Schmidt tangent Gram matrix (keep 0..3)
        #   G_ij = Re Tr[(partial_i rho)(partial_j rho)]
        #   computed via the equivalent Frobenius form after Hermitian symmetrization.
        # --------------------------
        red_hs_fn = make_reduced_hs_matrix_fn_for_layer_sequential(
            num_layers=L,
            keep_wires=KEEP_WIRES,
            jvp_chunk=RED_JVP_CHUNK,
        )

        hs_rank_list = []
        hs_eigs_list = []
        hs_thresh_list = []

        for s in tqdm(
            range(NUM_QFIM_SAMPLES),
            desc=f"Reduced HS samples (rank+eigs) (L={L})",
            unit="sample",
            leave=False,
        ):
            th = thetas_L[s]

            G = red_hs_fn(th)
            evals_hs = jnp.linalg.eigvalsh(_hermitian(G))
            evals_hs = jnp.clip(evals_hs, a_min=0.0)

            evals_hs_masked, thresh_hs = threshold_psd_eigvals_for_rank(evals_hs)
            rank_hs = jnp.sum(evals_hs > thresh_hs)

            hs_rank_list.append(int(jax.device_get(rank_hs)))
            hs_eigs_list.append(
                np.asarray(
                    jax.device_get(evals_hs_masked[::-1]),
                    dtype=NP_REAL_DTYPE,
                )
            )
            hs_thresh_list.append(float(jax.device_get(thresh_hs)))

        hs_rank_reduced_by_layer[L] = np.asarray(hs_rank_list, dtype=int)
        hs_eigs_reduced_by_layer[L] = np.stack(hs_eigs_list, axis=0)
        hs_thresh_reduced_by_layer[L] = np.asarray(
            hs_thresh_list,
            dtype=NP_REAL_DTYPE,
        )

        pure_hs_fn = make_pure_full_hs_matrix_fn_for_layer(
            num_layers=L,
            jvp_chunk=RED_JVP_CHUNK,
        )
        pure_hs_ranks, pure_hs_eigs, pure_hs_thresholds = [], [], []
        for s in tqdm(
            range(NUM_QFIM_SAMPLES),
            desc=f"Pure(full) HS samples (rank+eigs) (L={L})",
            unit="sample",
            leave=False,
        ):
            evals_hs_pure = jnp.clip(
                jnp.linalg.eigvalsh(_hermitian(pure_hs_fn(thetas_L[s]))),
                a_min=0.0,
            )
            masked_hs_pure, threshold_hs_pure = threshold_psd_eigvals_for_rank(
                evals_hs_pure
            )
            pure_hs_ranks.append(
                int(jax.device_get(jnp.sum(evals_hs_pure > threshold_hs_pure)))
            )
            pure_hs_eigs.append(
                np.asarray(jax.device_get(masked_hs_pure[::-1]), dtype=NP_REAL_DTYPE)
            )
            pure_hs_thresholds.append(float(jax.device_get(threshold_hs_pure)))
        hs_rank_pure_by_layer[L] = np.asarray(pure_hs_ranks, dtype=NP_INT_DTYPE)
        hs_eigs_pure_by_layer[L] = np.stack(pure_hs_eigs, axis=0)
        hs_thresh_pure_by_layer[L] = np.asarray(
            pure_hs_thresholds, dtype=NP_REAL_DTYPE
        )

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
        # Observable-Relevant Tangent Kernel
        #   o_a(theta) = Tr[(c_a O_a) rho_system(theta)]
        #   K_obs(theta) = J_obs(theta) J_obs(theta)^T
        # --------------------------
        ortk_metrics_fn = make_ortk_rank_effective_eigvals_fn_for_layer(L)
        ortk_rank_list = []
        ortk_effective_rank_list = []
        ortk_eigs_list = []
        ortk_trace_list = []

        for s in tqdm(
            range(NUM_QFIM_SAMPLES),
            desc=f"Observable tangent kernel samples (L={L})",
            unit="sample",
            leave=False,
        ):
            rank_value, effective_rank_value, eigs_desc = ortk_metrics_fn(
                thetas_L[s]
            )
            eigs_np = np.asarray(
                jax.device_get(eigs_desc),
                dtype=NP_REAL_DTYPE,
            )
            ortk_rank_list.append(int(jax.device_get(rank_value)))
            ortk_effective_rank_list.append(
                NP_REAL_DTYPE(jax.device_get(effective_rank_value))
            )
            ortk_eigs_list.append(eigs_np)
            ortk_trace_list.append(NP_REAL_DTYPE(np.sum(eigs_np)))

        ortk_rank_by_layer[L] = np.asarray(ortk_rank_list, dtype=NP_INT_DTYPE)
        ortk_effective_rank_by_layer[L] = np.asarray(
            ortk_effective_rank_list,
            dtype=NP_REAL_DTYPE,
        )
        ortk_eigs_by_layer[L] = np.stack(ortk_eigs_list, axis=0)
        ortk_trace_by_layer[L] = np.asarray(
            ortk_trace_list,
            dtype=NP_REAL_DTYPE,
        )

        if make_plots:
            _save_qfim_eigs_violinplot_by_index(
                ortk_eigs_by_layer[L],
                title=(
                    rf"Observable-Relevant Tangent Kernel eigenvalues at "
                    rf"{NUM_QFIM_SAMPLES} random points (L={L})"
                ),
                outpath=os.path.join(ortk_eigs_dir, f"L{L}.pdf"),
                rank_thresholds=np.asarray(
                    [ORTK_RANK_THRESHOLD],
                    dtype=NP_REAL_DTYPE,
                ),
                ylabel="ORTK eigenvalue",
            )

        # --------------------------
        # Energy Hessian
        #   H_ij = partial_i partial_j E(theta)
        #   Hessian eigenvalues are signed; rank counts |eta_i| > threshold.
        # --------------------------
        hessian_eigvals_fn = make_energy_hessian_eigvals_fn_for_layer(num_layers=L)

        hessian_rank_list = []
        hessian_eigs_list = []
        hessian_thresh_list = []
        hessian_trace_list = []
        hessian_abs_eigsum_list = []

        for s in tqdm(
            range(NUM_QFIM_SAMPLES),
            desc=f"Energy Hessian samples (rank+eigs) (L={L})",
            unit="sample",
            leave=False,
        ):
            th = thetas_L[s]

            evals_hessian = hessian_eigvals_fn(th)
            rank_hessian = effective_abs_rank_from_eigvals(evals_hessian)
            thresh_hessian = rank_threshold_from_eigvals(evals_hessian)
            evals_hessian_np = np.asarray(
                jax.device_get(evals_hessian),
                dtype=NP_REAL_DTYPE,
            )

            hessian_rank_list.append(int(jax.device_get(rank_hessian)))
            hessian_eigs_list.append(evals_hessian_np)
            hessian_thresh_list.append(float(jax.device_get(thresh_hessian)))
            hessian_trace_list.append(NP_REAL_DTYPE(np.sum(evals_hessian_np)))
            hessian_abs_eigsum_list.append(
                NP_REAL_DTYPE(np.sum(np.abs(evals_hessian_np)))
            )

        hessian_rank_by_layer[L] = np.asarray(hessian_rank_list, dtype=int)
        hessian_eigs_by_layer[L] = np.stack(hessian_eigs_list, axis=0)
        hessian_thresh_by_layer[L] = np.asarray(
            hessian_thresh_list,
            dtype=NP_REAL_DTYPE,
        )
        hessian_trace_by_layer[L] = np.asarray(
            hessian_trace_list,
            dtype=NP_REAL_DTYPE,
        )
        hessian_abs_eigsum_by_layer[L] = np.asarray(
            hessian_abs_eigsum_list,
            dtype=NP_REAL_DTYPE,
        )

        if make_plots:
            _save_signed_eigs_scatterplot_by_index(
                hessian_eigs_by_layer[L],
                title=rf"Energy Hessian eigenvalues at {NUM_QFIM_SAMPLES} random points (L={L})",
                outpath=os.path.join(hessian_eigs_dir, f"L{L}.pdf"),
                rank_thresholds=hessian_thresh_by_layer[L],
                ylabel="Energy Hessian eigenvalue",
            )
            plot_style.save_eigenvalue_histograms_by_trial(
                hessian_eigs_by_layer[L],
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
    
        # --------------------------
        # Pure(full) QFIM (only for small layers; logic kept unchanged)
        # --------------------------
        if L >= PURE_QFIM_LAYER_THRESHOLD:
            qfim_rank_pure_by_layer[L] = None
            qfim_eigs_pure_by_layer[L] = None
            qfim_thresh_pure_by_layer[L] = None
        else:
            pure_qfim_fn = make_pure_qfim_matrix_fn_for_layer(num_layers=L)
    
            rp_list = []
            eigs_pure_list = []
            thresh_p_list = []
    
            for s in tqdm(
                range(NUM_QFIM_SAMPLES),
                desc=f"Pure(full) QFIM samples (rank+eigs) (L={L})",
                unit="sample",
                leave=False,
            ):
                th = thetas_L[s]
                Fp = pure_qfim_fn(th)
    
                evals_p = jnp.linalg.eigvalsh(_hermitian(Fp))
                evals_p = jnp.clip(evals_p, a_min=0.0)
    
                evals_p_masked, thresh_p = threshold_psd_eigvals_for_rank(evals_p)
                rp = jnp.sum(evals_p > thresh_p)
                rp_list.append(int(jax.device_get(rp)))
    
                eigs_pure_list.append(np.asarray(jax.device_get(evals_p_masked[::-1]), dtype=NP_REAL_DTYPE))
                thresh_p_list.append(float(jax.device_get(thresh_p)))
    
            qfim_rank_pure_by_layer[L] = np.asarray(rp_list, dtype=int)
            qfim_eigs_pure_by_layer[L] = np.stack(eigs_pure_list, axis=0)
            qfim_thresh_pure_by_layer[L] = np.asarray(thresh_p_list, dtype=NP_REAL_DTYPE)
    
            if make_plots:
                _save_qfim_eigs_violinplot_by_index(
                    qfim_eigs_pure_by_layer[L],
                    title=rf"QFIM eigenvalues (Pure full-state) at {NUM_QFIM_SAMPLES} random points (L={L})",
                    outpath=os.path.join(qfim_eigs_pure_dir, f"L{L}_pure_full.pdf"),
                    rank_thresholds=qfim_thresh_pure_by_layer[L],
                )
                plot_style.save_eigenvalue_histograms_by_trial(
                    qfim_eigs_pure_by_layer[L],
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

    save_npz_result(
        os.path.join(qfim_results_dir, "qfim_random_points.npz"),
        h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
        num_qfim_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
        qfim_sample_seed_base=np.asarray(QFIM_SAMPLE_SEED_BASE, dtype=NP_INT_DTYPE),
        qfim_effective_rank_threshold=np.asarray(
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        layers=np.asarray(layer_list, dtype=NP_INT_DTYPE),
        pure_qfim_layer_threshold=np.asarray(
            PURE_QFIM_LAYER_THRESHOLD,
            dtype=NP_INT_DTYPE,
        ),
        red_jvp_chunk=np.asarray(RED_JVP_CHUNK, dtype=NP_INT_DTYPE),
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

    save_npz_result(
        os.path.join(hs_results_dir, "hs_random_points_reduced_0123.npz"),
        h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
        num_hs_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
        hs_sample_seed_base=np.asarray(QFIM_SAMPLE_SEED_BASE, dtype=NP_INT_DTYPE),
        hs_effective_rank_threshold=np.asarray(
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        layers=np.asarray(layer_list, dtype=NP_INT_DTYPE),
        red_jvp_chunk=np.asarray(RED_JVP_CHUNK, dtype=NP_INT_DTYPE),
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
        layers=np.asarray(layer_list, dtype=NP_INT_DTYPE),
        representation=np.asarray("pure_full"),
        **{f"L{int(L)}_rank": arr for L, arr in hs_rank_pure_by_layer.items()},
        **{f"L{int(L)}_eigs_desc": arr for L, arr in hs_eigs_pure_by_layer.items()},
        **{f"L{int(L)}_rank_threshold": arr for L, arr in hs_thresh_pure_by_layer.items()},
    )

    save_npz_result(
        os.path.join(ortk_results_dir, "ortk_random_points.npz"),
        representation_equivalence=np.asarray(
            "pure_full == reduced_keep_0123 for system-observable expectations"
        ),
        h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
        num_ortk_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
        ortk_sample_seed_base=np.asarray(
            QFIM_SAMPLE_SEED_BASE,
            dtype=NP_INT_DTYPE,
        ),
        ortk_rank_threshold=np.asarray(
            ORTK_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        ortk_participation_eps=np.asarray(
            ORTK_PARTICIPATION_EPS,
            dtype=NP_REAL_DTYPE,
        ),
        ortk_num_observables=np.asarray(
            H_OBSERVABLE_MATRICES.shape[0],
            dtype=NP_INT_DTYPE,
        ),
        layers=np.asarray(layer_list, dtype=NP_INT_DTYPE),
        **{
            f"L{int(L)}_rank": arr
            for L, arr in ortk_rank_by_layer.items()
        },
        **{
            f"L{int(L)}_effective_rank": arr
            for L, arr in ortk_effective_rank_by_layer.items()
        },
        **{
            f"L{int(L)}_eigs_desc": arr
            for L, arr in ortk_eigs_by_layer.items()
        },
        **{
            f"L{int(L)}_trace": arr
            for L, arr in ortk_trace_by_layer.items()
        },
    )

    save_npz_result(
        os.path.join(hessian_results_dir, "hessian_random_points.npz"),
        representation_equivalence=np.asarray(
            "pure_full == reduced_keep_0123 for E=Tr[(H_system tensor I) rho_full]"
        ),
        h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
        num_hessian_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
        hessian_sample_seed_base=np.asarray(QFIM_SAMPLE_SEED_BASE, dtype=NP_INT_DTYPE),
        hessian_effective_rank_threshold=np.asarray(
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        layers=np.asarray(layer_list, dtype=NP_INT_DTYPE),
        **{
            f"L{int(L)}_theta": arr
            for L, arr in qfim_random_thetas_by_layer.items()
        },
        **{
            f"L{int(L)}_rank": arr
            for L, arr in hessian_rank_by_layer.items()
        },
        **{
            f"L{int(L)}_eigs_desc": arr
            for L, arr in hessian_eigs_by_layer.items()
        },
        **{
            f"L{int(L)}_rank_threshold": arr
            for L, arr in hessian_thresh_by_layer.items()
        },
        **{
            f"L{int(L)}_trace": arr
            for L, arr in hessian_trace_by_layer.items()
        },
        **{
            f"L{int(L)}_abs_eigsum": arr
            for L, arr in hessian_abs_eigsum_by_layer.items()
        },
    )

    if not make_plots:
        return

    plot_scalar_violin_by_layer(
        ortk_rank_by_layer,
        layer_list,
        title=(
            rf"Observable-Relevant Tangent Kernel rank at "
            rf"{NUM_QFIM_SAMPLES} random points"
        ),
        ylabel="ORTK rank",
        outpath=os.path.join(
            ortk_rank_random_dir,
            "ortk_rank_violinplot_random_points.pdf",
        ),
        integer_y_axis=True,
    )
    plot_scalar_violin_by_layer(
        ortk_effective_rank_by_layer,
        layer_list,
        title=(
            rf"Observable-Relevant Tangent Kernel participation effective "
            rf"rank at {NUM_QFIM_SAMPLES} random points"
        ),
        ylabel="ORTK participation effective rank",
        outpath=os.path.join(
            ortk_effective_rank_random_dir,
            "ortk_effective_rank_violinplot_random_points.pdf",
        ),
        integer_y_axis=False,
    )
    
    
    # ============================================================
    # Plot: QFIM rank vs depth  (VIOLIN)
    #   - reduced 縺ｨ pure(full) 縺ｮ縺ｿ繧定｡ｨ遉ｺ
    #   - upper/lower bound 縺ｮ險育ｮ励・謠冗判縺ｯ陦後ｏ縺ｪ縺・
    # ============================================================
    
    new_prx_figure(width="double")
    ax = plt.gca()
    
    x_all = np.array(layer_list, dtype=NP_REAL_DTYPE)
    x_labels = [str(L) for L in layer_list]
    
    dx = 0.25
    violin_w_rank = 0.20
    num_layers = len(layer_list)
    
    # ------------------------------
    # Reduced keep (0..3)
    # ------------------------------
    for idx, L in enumerate(layer_list):
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
    pure_layers = [L for L in layer_list if qfim_rank_pure_by_layer[L] is not None]
    
    for L in pure_layers:
        idx = layer_list.index(L)
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
        for i, L in enumerate(layer_list)
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
        layer_list,
        color="C0",
        title=rf"Maximum pure full-state QFIM rank at {NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum QFIM effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(qfim_rank_random_dir, "qfim_rank_max_random_points_pure_full.pdf"),
        marker="s",
        lw=1.0,
    )
    
    
    plot_qfim_rank_max_by_layer(
        qfim_rank_reduced_by_layer,
        layer_list,
        color="C0",
        title=rf"Maximum QFIM rank at {NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum QFIM effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(qfim_rank_random_dir, "qfim_rank_max_random_points_reduced_0123.pdf"),
        marker="o",
        lw=1.0,
    )

    new_prx_figure(width="double")
    ax = plt.gca()

    for idx, L in enumerate(layer_list):
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
        layer_list,
        color="C3",
        title=rf"Maximum HS tangent Gram rank at {NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum HS effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(hs_rank_random_dir, "hs_rank_max_random_points_reduced_0123.pdf"),
        marker="D",
        lw=1.0,
    )

    new_prx_figure(width="double")
    ax = plt.gca()

    for idx, L in enumerate(layer_list):
        color = cmap(idx / num_layers)
        hessian_dataset = _make_violin_ready(
            hessian_rank_by_layer[L],
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
        _style_violin(
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
    set_prx_title(
        rf"Energy Hessian rank at {NUM_QFIM_SAMPLES} random points",
        ax=ax,
    )
    ax.grid(True, axis="y", alpha=0.3)

    save_current_figure(
        os.path.join(hessian_rank_random_dir, "hessian_rank_violinplot_random_points.pdf"),
        outside_legend=False,
    )

    plot_qfim_rank_max_by_layer(
        hessian_rank_by_layer,
        layer_list,
        color="C6",
        title=rf"Maximum energy Hessian rank at {NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum Hessian rank $(|\eta_k| > 10^{-12})$",
        outpath=os.path.join(hessian_rank_random_dir, "hessian_rank_max_random_points.pdf"),
        marker="P",
        lw=1.0,
    )
    
    
    # ============================================================

def run_optimization_path_qfim_analysis(*, make_plots: bool = False) -> None:
    """Compute and plot reduced QFIM rank along sampled VQE trajectories."""
    global qfim_rank_history_by_layer, qfim_eigs_history_by_layer, qfim_thresh_history_by_layer, qfim_rank_history_npz
    global hs_rank_history_by_layer, hs_eigs_history_by_layer, hs_thresh_history_by_layer, hs_rank_history_npz
    global ortk_rank_history_by_layer, ortk_effective_rank_history_by_layer
    global ortk_eigs_history_by_layer, ortk_trace_history_by_layer
    global hessian_rank_history_by_layer, hessian_eigs_history_by_layer, hessian_thresh_history_by_layer, hessian_trace_history_by_layer, hessian_abs_eigsum_history_by_layer, hessian_rank_history_npz
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
        )
    )
    
    qfim_rank_history_npz = {
        "sample_iters": np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        "plot_iters": _qfim_history_plot_iterations(sample_iters),
        "layers": np.asarray(layer_list, dtype=NP_INT_DTYPE),
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
        )
    )
    save_npz_result(
        os.path.join(qfim_results_dir, "qfim_rank_history_optimization_path_pure_full.npz"),
        sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        plot_iters=_qfim_history_plot_iterations(sample_iters),
        layers=np.asarray(layer_list, dtype=NP_INT_DTYPE),
        representation=np.asarray("pure_full"),
        **{f"L{int(L)}_rank": arr for L, arr in qfim_rank_history_pure.items()},
        **{f"L{int(L)}_eigs": arr for L, arr in qfim_eigs_history_pure.items()},
        **{f"L{int(L)}_rank_threshold": arr for L, arr in qfim_thresh_history_pure.items()},
    )

    hs_rank_history_by_layer, hs_eigs_history_by_layer, hs_thresh_history_by_layer = (
        compute_hs_rank_history_by_layer(
            theta_sample_traces_by_layer,
            layer_list,
            keep_wires=KEEP_WIRES,
            jvp_chunk=RED_JVP_CHUNK,
        )
    )

    hs_rank_history_npz = {
        "sample_iters": np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        "plot_iters": _qfim_history_plot_iterations(sample_iters),
        "layers": np.asarray(layer_list, dtype=NP_INT_DTYPE),
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

    hs_rank_history_pure, hs_eigs_history_pure, hs_thresh_history_pure = (
        compute_hs_rank_history_by_layer(
            theta_sample_traces_by_layer,
            layer_list,
            jvp_chunk=RED_JVP_CHUNK,
            representation="pure_full",
        )
    )
    save_npz_result(
        os.path.join(hs_results_dir, "hs_rank_history_optimization_path_pure_full.npz"),
        sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        plot_iters=_qfim_history_plot_iterations(sample_iters),
        layers=np.asarray(layer_list, dtype=NP_INT_DTYPE),
        representation=np.asarray("pure_full"),
        **{f"L{int(L)}_rank": arr for L, arr in hs_rank_history_pure.items()},
        **{f"L{int(L)}_eigs": arr for L, arr in hs_eigs_history_pure.items()},
        **{f"L{int(L)}_rank_threshold": arr for L, arr in hs_thresh_history_pure.items()},
    )

    (
        ortk_rank_history_by_layer,
        ortk_effective_rank_history_by_layer,
        ortk_eigs_history_by_layer,
    ) = compute_ortk_rank_history_by_layer(
        theta_sample_traces_by_layer,
        layer_list,
    )
    ortk_trace_history_by_layer = {
        int(L): np.sum(np.asarray(arr, dtype=NP_REAL_DTYPE), axis=2)
        for L, arr in ortk_eigs_history_by_layer.items()
    }

    ortk_history_metadata = {
        "sample_iters": np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        "plot_iters": _qfim_history_plot_iterations(sample_iters),
        "layers": np.asarray(layer_list, dtype=NP_INT_DTYPE),
    }
    save_npz_result(
        os.path.join(ortk_results_dir, "ortk_rank_history_optimization_path.npz"),
        **ortk_history_metadata,
        ortk_rank_threshold=np.asarray(
            ORTK_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        **{
            f"L{int(L)}": arr
            for L, arr in ortk_rank_history_by_layer.items()
        },
    )
    save_npz_result(
        os.path.join(
            ortk_results_dir,
            "ortk_effective_rank_history_optimization_path.npz",
        ),
        **ortk_history_metadata,
        ortk_participation_eps=np.asarray(
            ORTK_PARTICIPATION_EPS,
            dtype=NP_REAL_DTYPE,
        ),
        **{
            f"L{int(L)}": arr
            for L, arr in ortk_effective_rank_history_by_layer.items()
        },
    )
    save_npz_result(
        os.path.join(ortk_results_dir, "ortk_eigs_history_optimization_path.npz"),
        **ortk_history_metadata,
        ortk_rank_threshold=np.asarray(
            ORTK_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        ortk_participation_eps=np.asarray(
            ORTK_PARTICIPATION_EPS,
            dtype=NP_REAL_DTYPE,
        ),
        **{
            f"L{int(L)}": arr
            for L, arr in ortk_eigs_history_by_layer.items()
        },
    )
    save_npz_result(
        os.path.join(ortk_results_dir, "ortk_trace_history_optimization_path.npz"),
        **ortk_history_metadata,
        **{
            f"L{int(L)}": arr
            for L, arr in ortk_trace_history_by_layer.items()
        },
    )

    (
        hessian_rank_history_by_layer,
        hessian_eigs_history_by_layer,
        hessian_thresh_history_by_layer,
    ) = compute_hessian_rank_history_by_layer(
        theta_sample_traces_by_layer,
        layer_list,
    )

    hessian_trace_history_by_layer = {
        int(L): np.sum(np.asarray(arr, dtype=NP_REAL_DTYPE), axis=2)
        for L, arr in hessian_eigs_history_by_layer.items()
    }
    hessian_abs_eigsum_history_by_layer = {
        int(L): np.sum(np.abs(np.asarray(arr, dtype=NP_REAL_DTYPE)), axis=2)
        for L, arr in hessian_eigs_history_by_layer.items()
    }

    hessian_rank_history_npz = {
        "sample_iters": np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        "plot_iters": _qfim_history_plot_iterations(sample_iters),
        "layers": np.asarray(layer_list, dtype=NP_INT_DTYPE),
        "hessian_effective_rank_threshold": np.asarray(
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
    }
    hessian_rank_history_npz.update(
        {
            f"L{int(L)}_rank": arr
            for L, arr in hessian_rank_history_by_layer.items()
        }
    )
    hessian_rank_history_npz.update(
        {
            f"L{int(L)}_eigs": arr
            for L, arr in hessian_eigs_history_by_layer.items()
        }
    )
    hessian_rank_history_npz.update(
        {
            f"L{int(L)}_rank_threshold": arr
            for L, arr in hessian_thresh_history_by_layer.items()
        }
    )
    hessian_rank_history_npz.update(
        {
            f"L{int(L)}_trace": arr
            for L, arr in hessian_trace_history_by_layer.items()
        }
    )
    hessian_rank_history_npz.update(
        {
            f"L{int(L)}_abs_eigsum": arr
            for L, arr in hessian_abs_eigsum_history_by_layer.items()
        }
    )

    save_npz_result(
        os.path.join(hessian_results_dir, "hessian_rank_history_optimization_path.npz"),
        **hessian_rank_history_npz,
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

    plot_qfim_rank_history_mean_by_layer(
        ortk_rank_history_by_layer,
        layer_list,
        sample_iters,
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
        layer_list,
        sample_iters,
        title="Minimum Observable-Relevant Tangent Kernel rank along optimization path",
        outpath=os.path.join(
            ortk_rank_optimization_path_min_dir,
            "ortk_rank_min_history_optimization_path.pdf",
        ),
        ylabel="Minimum ORTK rank",
        cmap=cmap,
        integer_y_axis=True,
    )
    plot_qfim_rank_history_mean_by_layer(
        ortk_effective_rank_history_by_layer,
        layer_list,
        sample_iters,
        title=(
            "Mean Observable-Relevant Tangent Kernel participation effective "
            "rank along optimization path"
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
        layer_list,
        sample_iters,
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
    plot_qfim_rank_history_mean_by_layer(
        ortk_trace_history_by_layer,
        layer_list,
        sample_iters,
        title="Mean Observable-Relevant Tangent Kernel trace along optimization path",
        outpath=os.path.join(
            ortk_trace_optimization_path_dir,
            "ortk_trace_mean_history_optimization_path.pdf",
        ),
        ylabel="Mean ORTK trace",
        cmap=cmap,
    )

    plot_qfim_rank_history_mean_by_layer(
        hessian_rank_history_by_layer,
        layer_list,
        sample_iters,
        title="Mean energy Hessian rank along optimization path",
        outpath=os.path.join(
            hessian_rank_optimization_path_mean_dir,
            "hessian_rank_mean_history_optimization_path.pdf",
        ),
        ylabel=r"Mean Hessian rank $(|\eta_k| > 10^{-12})$",
        cmap=cmap,
    )

    plot_qfim_rank_history_min_by_layer(
        hessian_rank_history_by_layer,
        layer_list,
        sample_iters,
        title="Minimum energy Hessian rank along optimization path",
        outpath=os.path.join(
            hessian_rank_optimization_path_min_dir,
            "hessian_rank_min_history_optimization_path.pdf",
        ),
        ylabel=r"Minimum Hessian rank $(|\eta_k| > 10^{-12})$",
        cmap=cmap,
    )
    
    
    # ============================================================

def run_qfim_grad_alignment_analysis(*, make_plots: bool = False) -> None:
    """Compute QFIM-gradient alignment tables and scatter plots."""
    global QFIM_GRAD_ALIGN_EIG_FLOOR, QFIM_GRAD_ALIGN_WEIGHT_FLOOR, qfim_grad_align_dir, qfim_grad_align_results_dir, RUN_QFIM_GRAD_ALIGNMENT_PER_ITERATION, LOG_X_QFIM_GRAD_ALIGNMENT
    global LOG_Y_QFIM_GRAD_ALIGNMENT, QFIM_GRAD_ALIGNMENT_RUN_INDICES, QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS, qfim_grad_alignment_table_by_layer_iteration
    # QFIM eigenvalue vs gradient-direction weight scatter plots
    #   x-axis: QFIM eigenvalue lambda_i
    #   y-axis: w_i^grad = |v_i^T g|^2 / sum_j |v_j^T g|^2
    #
    # This section uses optimization-path samples already stored in
    #   theta_sample_traces_by_layer[L]
    #   grad_sample_traces_by_layer[L]
    # and constructs one scatter plot per available layer/iteration.
    # ============================================================
    
    QFIM_GRAD_ALIGN_EIG_FLOOR = 1e-16
    QFIM_GRAD_ALIGN_WEIGHT_FLOOR = 1e-16
    
    qfim_grad_align_dir = os.path.join(qfim_fig_dir, "grad_alignment")
    os.makedirs(qfim_grad_align_dir, exist_ok=True)
    os.makedirs(qfim_grad_align_results_dir, exist_ok=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ------------------------------------------------------------
    # Execution settings
    # ------------------------------------------------------------
    # Saved as:
    #   qfim_grad_alignment/L{L}/qfim_grad_weight_scatter_L{L}_iterXXXXXX.pdf
    RUN_QFIM_GRAD_ALIGNMENT_PER_ITERATION = cfg.RUN_QFIM_GRAD_ALIGNMENT_PER_ITERATION
    LOG_X_QFIM_GRAD_ALIGNMENT = cfg.LOG_X_QFIM_GRAD_ALIGNMENT
    LOG_Y_QFIM_GRAD_ALIGNMENT = cfg.LOG_Y_QFIM_GRAD_ALIGNMENT
    
    # Use None for all VQE runs, or e.g. range(5) for a quick test.
    QFIM_GRAD_ALIGNMENT_RUN_INDICES = cfg.QFIM_GRAD_ALIGNMENT_RUN_INDICES
    target_iterations_cfg = cfg.QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS
    if target_iterations_cfg is None:
        QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS = tuple(int(t) for t in sample_iters)
    else:
        QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS = tuple(
            int(t) for t in target_iterations_cfg
        )
    
    if RUN_QFIM_GRAD_ALIGNMENT_PER_ITERATION:
        qfim_grad_alignment_table_by_layer_iteration = (
            run_qfim_grad_alignment_by_layer_iteration_folders(
                layers=layer_list,
                target_iterations=QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS,
                run_indices=QFIM_GRAD_ALIGNMENT_RUN_INDICES,
                sample_iters_for_labels=sample_iters,
                jvp_chunk=RED_JVP_CHUNK,
                log_x=LOG_X_QFIM_GRAD_ALIGNMENT,
                log_y=LOG_Y_QFIM_GRAD_ALIGNMENT,
                save_npz=True,
                make_plots=make_plots,
                data_dir=qfim_grad_align_results_dir,
                plot_dir=qfim_grad_align_dir,
            )
        )


def collect_unitary_pqc_result() -> dict:
    """Return the compact summary shown by the notebook after execution."""
    return {
        "save_dir": save_dir,
        "figures_dir": figures_dir,
        "energy_fig_dir": energy_fig_dir,
        "qfim_fig_dir": qfim_fig_dir,
        "hs_fig_dir": hs_fig_dir,
        "ortk_fig_dir": ortk_fig_dir,
        "hessian_fig_dir": hessian_fig_dir,
        "circuit_dir": circuit_dir,
        "numerical_results_dir": numerical_results_dir,
        "energy_results_dir": energy_results_dir,
        "qfim_results_dir": qfim_results_dir,
        "hs_results_dir": hs_results_dir,
        "ortk_results_dir": ortk_results_dir,
        "hessian_results_dir": hessian_results_dir,
        "layer_list": layer_list,
        "sample_iters": sample_iters,
        "smallest_eigval": smallest_eigval,
    }


def run_unitary_pqc_overparam() -> dict:
    """Run Unitary-PQC numerical computations and save reusable .npz results."""
    configure_unitary_pqc_overparam()
    run_vqe_optimization(save_circuits=False)
    run_random_qfim_analysis(make_plots=False)
    run_optimization_path_qfim_analysis(make_plots=False)
    run_qfim_grad_alignment_analysis(make_plots=False)
    return collect_unitary_pqc_result()


if __name__ == "__main__":
    run_unitary_pqc_overparam()


