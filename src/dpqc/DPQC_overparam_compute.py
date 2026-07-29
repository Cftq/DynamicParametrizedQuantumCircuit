#!/usr/bin/env python
# coding: utf-8
"""Run DPQC overparameterization numerical calculations and save results.

This script is split from DPQC_overparam.ipynb. It performs the expensive
VQE/QFIM calculations and writes reusable .npz files under
figs/dpqc/h_<h_param>/numerical_results. Plot generation is handled by
DPQC_overparam_visualize.py.
"""


import os
import sys
import warnings
from copy import copy as shallow_copy
from pathlib import Path
from typing import Optional, Tuple


# Support direct execution from any working directory, for example:
#     python C:\...\src\dpqc\DPQC_overparam_compute.py
# Shared configuration and helpers live in ``src/common`` rather than beside
# this entry-point script, so add both source directories before importing
# project modules.
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
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorcircuit as tc
from plot import (
    CIRCUIT_SAVE_PDF,
    CIRCUIT_SAVE_PNG,
    SAVE_DPI,
    new_fig_ax,
    plot_qfim_grad_alignment_layer_overlay,
    plot_qfim_grad_alignment_table,
    save_fig,
    style_axes_for_prx,
)
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
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
    ket0_density,
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
success_probability_thresholds = np.asarray(
    cfg.SUCCESS_PROBABILITY_THRESHOLDS,
    dtype=NP_REAL_DTYPE,
)

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
n_param_per_layer = NUM_BLOCKS * PARAMS_PER_BLOCK + EXTRA_PARAMS_PER_LAYER

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
keep_key_5 = "keep01234"
keep_label_5 = "Reduced (0,1,2,3,4)"


def jax_to_np(x, dtype=None):
    return np.asarray(jax.device_get(x), dtype=dtype)


# ==============================
# Circuit blocks TensorCircuit
# ==============================
def qg_layer(c: tc.Circuit, q0: int, q1: int, p: jnp.ndarray) -> None:
    c.rz(q0, theta=p[0])
    c.rz(q1, theta=p[1])
    c.rxx(q0, q1, theta=p[2])


def qg_dyn_delay(
    c: tc.Circuit,
    ctrl: int,
    tgt: int,
    varphi: float,
    phi: float,
) -> None:
    c.cx(ctrl, tgt)
    c.crz(tgt, ctrl, theta=varphi)
    c.crx(tgt, ctrl, theta=2.0 * phi)
    c.crz(tgt, ctrl, theta=varphi)


def create_dpqc(theta: jnp.ndarray, n_layer: int, num_system: int) -> tc.Circuit:
    qc = tc.Circuit(num_system + n_layer)
    theta_layers = jnp.reshape(theta, (n_layer, n_param_per_layer))

    for layer_idx, layer_theta in enumerate(theta_layers):
        blocks = jnp.reshape(
            layer_theta[:-EXTRA_PARAMS_PER_LAYER],
            (NUM_BLOCKS, PARAMS_PER_BLOCK),
        )

        for (q0, q1), p in zip(LAYER_PAIRS, blocks):
            qg_layer(qc, q0, q1, p)

        varphi = layer_theta[-2]
        phi = layer_theta[-1]

        qg_dyn_delay(
            qc,
            ANC_CENTER,
            num_system + layer_idx,
            varphi,
            phi,
        )

    return qc


# ==============================
# TC -> Qiskit drawing
# ==============================
def tc_to_qiskit_qc(
    tc_circ: tc.Circuit,
    n_qubits: Optional[int] = None,
) -> QuantumCircuit:
    if hasattr(tc_circ, "to_qiskit"):
        try:
            return tc_circ.to_qiskit()
        except Exception:
            pass

    if n_qubits is None:
        n_qubits = next(
            (
                int(getattr(tc_circ, attr))
                for attr in ("nqubits", "_nqubits", "n")
                if hasattr(tc_circ, attr)
            ),
            None,
        )

    if n_qubits is None:
        raise ValueError("n_qubits could not be inferred. Pass n_qubits explicitly.")

    return tc.translation.qir2qiskit(tc_circ.to_qir(), n_qubits)


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


def make_parameter_free_qiskit_for_drawing(qc: QuantumCircuit) -> QuantumCircuit:
    if qc.num_clbits > 0:
        qc_draw = QuantumCircuit(qc.num_qubits, qc.num_clbits, name=qc.name)
    else:
        qc_draw = QuantumCircuit(qc.num_qubits, name=qc.name)

    qc_draw.global_phase = 0.0

    for inst in qc.data:
        if hasattr(inst, "operation"):
            operation = inst.operation
            qargs = inst.qubits
            cargs = inst.clbits
        else:
            operation, qargs, cargs = inst

        if hasattr(operation, "to_mutable"):
            op_draw = operation.to_mutable()
        else:
            try:
                op_draw = operation.copy()
            except Exception:
                op_draw = shallow_copy(operation)

        gate_name = str(getattr(op_draw, "name", "")).lower()

        if gate_name in PARAMETER_FREE_GATE_LABELS:
            op_draw.label = PARAMETER_FREE_GATE_LABELS[gate_name]
        elif getattr(op_draw, "params", None):
            op_draw.label = rf"${gate_name}$"

        q_indices = [qc.find_bit(q).index for q in qargs]
        c_indices = [qc.find_bit(c).index for c in cargs]

        qc_draw.append(op_draw, q_indices, c_indices)

    return qc_draw


def save_circuit_matplotlib_png(
    tc_circ: tc.Circuit,
    outpath: str,
    n_qubits: Optional[int] = None,
    dpi: int = SAVE_DPI,
    save_png: bool = CIRCUIT_SAVE_PNG,
    save_pdf: bool = CIRCUIT_SAVE_PDF,
    hide_params: bool = True,
) -> None:
    qc = tc_to_qiskit_qc(tc_circ, n_qubits=n_qubits)

    if hide_params:
        qc_for_draw = make_parameter_free_qiskit_for_drawing(qc)
        drawer_style = {
            "displaytext": PARAMETER_FREE_GATE_LABELS,
        }
    else:
        qc_for_draw = qc
        drawer_style = None

    fig = circuit_drawer(
        qc_for_draw,
        output="mpl",
        fold=-1,
        style=drawer_style,
    )

    root, _ = os.path.splitext(outpath)

    if save_png:
        fig.savefig(
            root + ".png",
            dpi=dpi,
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
@jax.jit
def wrap_to_pi(x: jnp.ndarray) -> jnp.ndarray:
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=x.dtype)
    return (x + jnp.pi) % two_pi - jnp.pi


def wrap_theta_periodic_only(theta: jnp.ndarray, n_layer: int) -> jnp.ndarray:
    theta_layers = jnp.reshape(theta, (n_layer, n_param_per_layer))
    theta_layers = theta_layers.at[:, :-EXTRA_PARAMS_PER_LAYER].set(
        wrap_to_pi(theta_layers[:, :-EXTRA_PARAMS_PER_LAYER])
    )
    return jnp.reshape(theta_layers, (-1,))


def theta_difference_periodic_only(
    theta_a: jnp.ndarray,
    theta_b: jnp.ndarray,
    n_layer: int,
) -> jnp.ndarray:
    d_layers = jnp.reshape(theta_a - theta_b, (n_layer, n_param_per_layer))
    d_periodic = wrap_to_pi(d_layers[:, :-EXTRA_PARAMS_PER_LAYER])
    d_nonperiodic = d_layers[:, -EXTRA_PARAMS_PER_LAYER:]
    d_layers = jnp.concatenate([d_periodic, d_nonperiodic], axis=1)

    return jnp.reshape(d_layers, (-1,))


def rms_theta_distance_periodic_only(
    theta_a: jnp.ndarray,
    theta_b: jnp.ndarray,
    n_layer: int,
) -> jnp.ndarray:
    d = theta_difference_periodic_only(theta_a, theta_b, n_layer=n_layer)
    return jnp.sqrt(jnp.mean(d**2))


# ============================================================
# Sequential trace-out machinery
# ============================================================
I2 = jnp.eye(2, dtype=COMPLEX_DTYPE)
X2 = jnp.array([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)
P0 = jnp.array([[1, 0], [0, 0]], dtype=COMPLEX_DTYPE)
P1 = jnp.array([[0, 0], [0, 1]], dtype=COMPLEX_DTYPE)


def U_rz(theta: jnp.ndarray) -> jnp.ndarray:
    th = jnp.asarray(theta, dtype=REAL_DTYPE)

    return jnp.array(
        [
            [jnp.exp(-0.5j * th), 0.0],
            [0.0, jnp.exp(0.5j * th)],
        ],
        dtype=COMPLEX_DTYPE,
    )


def U_rx(theta: jnp.ndarray) -> jnp.ndarray:
    th = jnp.asarray(theta, dtype=REAL_DTYPE)
    c = jnp.cos(0.5 * th).astype(COMPLEX_DTYPE)
    s = jnp.sin(0.5 * th).astype(COMPLEX_DTYPE)

    return jnp.array(
        [
            [c, -1j * s],
            [-1j * s, c],
        ],
        dtype=COMPLEX_DTYPE,
    )


def U_rxx(theta: jnp.ndarray) -> jnp.ndarray:
    th = jnp.asarray(theta, dtype=REAL_DTYPE)
    c = jnp.cos(0.5 * th).astype(COMPLEX_DTYPE)
    s = jnp.sin(0.5 * th).astype(COMPLEX_DTYPE)

    return c * jnp.eye(4, dtype=COMPLEX_DTYPE) - 1j * s * jnp.kron(X2, X2)


_U_CX = jnp.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=COMPLEX_DTYPE,
)


def U_crx(theta: jnp.ndarray, control_pos: int, target_pos: int) -> jnp.ndarray:
    U = U_rx(theta)

    if (control_pos, target_pos) == (0, 1):
        return jnp.kron(P0, I2) + jnp.kron(P1, U)

    if (control_pos, target_pos) == (1, 0):
        return jnp.kron(I2, P0) + jnp.kron(U, P1)

    raise ValueError("control_pos and target_pos must be different and in {0,1}.")


def U_crz(theta: jnp.ndarray, control_pos: int, target_pos: int) -> jnp.ndarray:
    U = U_rz(theta)

    if (control_pos, target_pos) == (0, 1):
        return jnp.kron(P0, I2) + jnp.kron(P1, U)

    if (control_pos, target_pos) == (1, 0):
        return jnp.kron(I2, P0) + jnp.kron(U, P1)

    raise ValueError("control_pos and target_pos must be different and in {0,1}.")


def apply_unitary_on_rho(
    rho: jnp.ndarray,
    U: jnp.ndarray,
    wires,
    k: int,
) -> jnp.ndarray:
    wires = tuple(int(w) for w in wires)
    m = len(wires)

    assert U.shape == (2**m, 2**m)

    others = [i for i in range(k) if i not in wires]
    perm = list(wires) + others + [w + k for w in wires] + [o + k for o in others]

    inv_perm = [0] * (2 * k)
    for i, a in enumerate(perm):
        inv_perm[a] = i

    rho_p = jnp.transpose(jnp.reshape(rho, (2,) * (2 * k)), perm)

    dk = 2**m
    dr = 2 ** (k - m)
    rho_p = jnp.reshape(rho_p, (dk, dr, dk, dr))

    rho1 = jnp.einsum("ij,jrks->irks", U, rho_p)
    rho2 = jnp.einsum("irps,bp->irbs", rho1, jnp.conjugate(U))

    rho2 = jnp.reshape(
        rho2,
        (2,) * m + (2,) * (k - m) + (2,) * m + (2,) * (k - m),
    )

    return jnp.reshape(jnp.transpose(rho2, inv_perm), (2**k, 2**k))


def append_fresh_ancilla_zero(rho_k: jnp.ndarray) -> jnp.ndarray:
    return jnp.kron(rho_k, ket0_density(dtype=rho_k.dtype))


def partial_trace_last_qubit(rho_kplus1: jnp.ndarray, k: int) -> jnp.ndarray:
    dimk = 2**k
    rho = jnp.reshape(rho_kplus1, (dimk, 2, dimk, 2))
    return rho[:, 0, :, 0] + rho[:, 1, :, 1]


def z_expectation_last_qubit(rho_kplus1: jnp.ndarray, k: int) -> jnp.ndarray:
    dimk = 2**k
    rho = jnp.reshape(rho_kplus1, (dimk, 2, dimk, 2))
    z = jnp.trace(rho[:, 0, :, 0]) - jnp.trace(rho[:, 1, :, 1])
    return jnp.real(z)


def _hermitian(a: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (a + jnp.conjugate(a.T))


_RHO_KEEP_INIT = rho_zero_state(num_system_qubits, dtype=COMPLEX_DTYPE)


def _apply_kept_blocks(
    rho: jnp.ndarray,
    layer_theta: jnp.ndarray,
    *,
    k: int = num_system_qubits,
) -> jnp.ndarray:
    blocks = jnp.reshape(
        layer_theta[:-EXTRA_PARAMS_PER_LAYER],
        (NUM_BLOCKS, PARAMS_PER_BLOCK),
    )

    for (q0, q1), p in zip(LAYER_PAIRS, blocks):
        rho = apply_unitary_on_rho(rho, U_rz(p[0]), (q0,), k)
        rho = apply_unitary_on_rho(rho, U_rz(p[1]), (q1,), k)
        rho = apply_unitary_on_rho(rho, U_rxx(p[2]), (q0, q1), k)

    return rho


def _rho6_after_layer(rho: jnp.ndarray, layer_theta: jnp.ndarray) -> jnp.ndarray:
    rho = _apply_kept_blocks(rho, layer_theta)

    varphi = layer_theta[-2]
    phi = layer_theta[-1]

    rho6 = append_fresh_ancilla_zero(rho)

    rho6 = apply_unitary_on_rho(
        rho6,
        _U_CX,
        (ANC_CENTER, FRESH_ANCILLA),
        6,
    )

    rho6 = apply_unitary_on_rho(
        rho6,
        U_crz(varphi, control_pos=1, target_pos=0),
        (ANC_CENTER, FRESH_ANCILLA),
        6,
    )

    rho6 = apply_unitary_on_rho(
        rho6,
        U_crx(2.0 * phi, control_pos=1, target_pos=0),
        (ANC_CENTER, FRESH_ANCILLA),
        6,
    )

    rho6 = apply_unitary_on_rho(
        rho6,
        U_crz(varphi, control_pos=1, target_pos=0),
        (ANC_CENTER, FRESH_ANCILLA),
        6,
    )

    return rho6


def rho_keep_sequential_dpqc(theta: jnp.ndarray, n_layer: int) -> jnp.ndarray:
    theta_layers = jnp.reshape(theta, (n_layer, n_param_per_layer))

    def one_layer(rho: jnp.ndarray, layer_theta: jnp.ndarray):
        rho6 = _rho6_after_layer(rho, layer_theta)
        rho_next = partial_trace_last_qubit(rho6, k=num_system_qubits)
        return rho_next, None

    rho_final, _ = jax.lax.scan(one_layer, _RHO_KEEP_INIT, theta_layers)

    return _hermitian(rho_final)


def ancilla_p1_sequential_dpqc(theta: jnp.ndarray, n_layer: int) -> jnp.ndarray:
    theta_layers = jnp.reshape(theta, (n_layer, n_param_per_layer))

    def one_layer(rho: jnp.ndarray, layer_theta: jnp.ndarray):
        rho6 = _rho6_after_layer(rho, layer_theta)
        p1 = 0.5 * (1.0 - z_expectation_last_qubit(rho6, k=num_system_qubits))
        rho_next = partial_trace_last_qubit(rho6, k=num_system_qubits)
        return rho_next, p1

    _, p1_vec = jax.lax.scan(one_layer, _RHO_KEEP_INIT, theta_layers)

    return p1_vec


@jax.jit
def energy_from_rho_keep(rho_keep: jnp.ndarray) -> jnp.ndarray:
    return jnp.real(jnp.einsum("ij,ji->", rho_keep, H_matrix))


def partial_trace_one_qubit(
    rho: jnp.ndarray,
    n_qubits: int,
    trace_wire: int,
) -> jnp.ndarray:
    n_qubits = int(n_qubits)
    trace_wire = int(trace_wire)

    if not 0 <= trace_wire < n_qubits:
        raise ValueError("trace_wire out of range.")

    keep = [i for i in range(n_qubits) if i != trace_wire]
    perm = keep + [trace_wire] + [i + n_qubits for i in keep] + [
        trace_wire + n_qubits
    ]

    rho_p = jnp.transpose(jnp.reshape(rho, (2,) * (2 * n_qubits)), perm)

    dim_keep = 2 ** (n_qubits - 1)
    rho_p = jnp.reshape(rho_p, (dim_keep, 2, dim_keep, 2))

    return rho_p[:, 0, :, 0] + rho_p[:, 1, :, 1]


def rho4_from_rho5(rho5: jnp.ndarray) -> jnp.ndarray:
    return partial_trace_one_qubit(rho5, n_qubits=5, trace_wire=4)


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
circuit_dir = os.path.join(save_dir, "optimized_circuits")
numerical_results_dir = os.path.join(save_dir, "numerical_results")
energy_results_dir = os.path.join(numerical_results_dir, "energy")
qfim_results_dir = os.path.join(numerical_results_dir, "qfim")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(energy_fig_dir, exist_ok=True)
os.makedirs(qfim_fig_dir, exist_ok=True)
os.makedirs(circuit_dir, exist_ok=True)
os.makedirs(numerical_results_dir, exist_ok=True)
os.makedirs(energy_results_dir, exist_ok=True)
os.makedirs(qfim_results_dir, exist_ok=True)


def save_npz_result(outpath: str, **arrays) -> None:
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    np.savez(outpath, **arrays)


def _layer_arrays_for_npz(data_by_layer: dict, suffix: str) -> dict:
    return {
        f"L{int(L)}_{suffix}": np.asarray(arr)
        for L, arr in data_by_layer.items()
    }


def _finite_mean_sem(values, *, axis=0):
    """Return finite-value mean, sample SEM, and contributing counts."""
    values = np.asarray(values, dtype=NP_REAL_DTYPE)
    values = np.moveaxis(values, axis, 0)
    output_shape = values.shape[1:]
    means = np.full(output_shape, np.nan, dtype=NP_REAL_DTYPE)
    sems = np.full(output_shape, np.nan, dtype=NP_REAL_DTYPE)
    counts = np.zeros(output_shape, dtype=NP_INT_DTYPE)

    for output_index in np.ndindex(output_shape):
        sample = values[(slice(None),) + output_index]
        sample = sample[np.isfinite(sample)]
        count = int(sample.size)
        counts[output_index] = count
        if count:
            means[output_index] = NP_REAL_DTYPE(np.mean(sample))
        if count > 1:
            sems[output_index] = NP_REAL_DTYPE(
                np.std(sample, ddof=1) / np.sqrt(count)
            )

    return means, sems, counts


def _multiple_tolerance_success_statistics(
    final_energies_by_layer,
    *,
    layers,
    thresholds,
    ground_energy,
    num_trials,
):
    """Compute final-energy success statistics at several error thresholds.

    The trial error is clipped only from below,
    ``max(E_final - E0, 0)``, and threshold equality counts as success.
    """
    layers = np.asarray(layers, dtype=NP_INT_DTYPE)
    thresholds = np.asarray(thresholds, dtype=NP_REAL_DTYPE)
    ground_energy = NP_REAL_DTYPE(ground_energy)
    num_trials = int(num_trials)

    if layers.ndim != 1 or layers.size == 0:
        raise ValueError("layers must be a non-empty one-dimensional array.")
    if np.unique(layers).size != layers.size:
        raise ValueError("layers must not contain duplicates.")
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0.0):
        raise ValueError("All success-probability thresholds must be finite and positive.")
    if np.any(np.diff(thresholds) >= 0.0):
        raise ValueError(
            "Success-probability thresholds must be strictly decreasing."
        )
    if not np.isfinite(ground_energy):
        raise ValueError("ground_energy must be finite.")
    if num_trials <= 0:
        raise ValueError("num_trials must be positive.")

    final_energies = np.stack(
        [
            np.asarray(final_energies_by_layer[int(layer)], dtype=NP_REAL_DTYPE)
            for layer in layers
        ],
        axis=0,
    )
    expected_energy_shape = (layers.size, num_trials)
    if final_energies.shape != expected_energy_shape:
        raise ValueError(
            "Final-energy matrix has shape "
            f"{final_energies.shape}; expected {expected_energy_shape} "
            "(one final energy per independent trial and layer)."
        )
    if not np.all(np.isfinite(final_energies)):
        raise ValueError("All final energies must be finite.")

    raw_final_energy_errors = final_energies - ground_energy
    final_energy_errors = np.maximum(raw_final_energy_errors, 0.0)
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

    expected_indicator_shape = (
        layers.size,
        num_trials,
        thresholds.size,
    )
    expected_summary_shape = (layers.size, thresholds.size)
    if success_indicators.shape != expected_indicator_shape:
        raise AssertionError(
            "Success-indicator shape mismatch: "
            f"{success_indicators.shape} != {expected_indicator_shape}."
        )
    if success_counts.shape != expected_summary_shape:
        raise AssertionError(
            "Success-count shape mismatch: "
            f"{success_counts.shape} != {expected_summary_shape}."
        )
    if success_probabilities.shape != expected_summary_shape:
        raise AssertionError(
            "Success-probability shape mismatch: "
            f"{success_probabilities.shape} != {expected_summary_shape}."
        )
    if not np.array_equal(
        success_counts,
        np.sum(success_indicators, axis=1, dtype=NP_INT_DTYPE),
    ):
        raise AssertionError("Success counts do not equal the indicator sums.")
    if not np.array_equal(
        success_probabilities,
        success_counts.astype(NP_REAL_DTYPE) / NP_REAL_DTYPE(num_trials),
    ):
        raise AssertionError(
            "Success probabilities do not equal success_counts / num_trials."
        )
    if np.any(np.diff(success_indicators, axis=2) > 0):
        raise AssertionError(
            "Indicators must be nonincreasing for decreasing thresholds."
        )
    if np.any(np.diff(success_probabilities, axis=1) > 0.0):
        raise AssertionError(
            "Success probabilities must be nonincreasing for decreasing thresholds."
        )

    return {
        "layers": layers,
        "thresholds": thresholds,
        "num_trials": np.asarray(num_trials, dtype=NP_INT_DTYPE),
        "ground_energy": np.asarray(ground_energy, dtype=NP_REAL_DTYPE),
        "final_energies": final_energies,
        "raw_final_energy_errors": raw_final_energy_errors,
        "final_energy_errors": final_energy_errors,
        "success_indicators": success_indicators,
        "success_counts": success_counts,
        "success_probabilities": success_probabilities,
    }


# ============================================================
# VQE optimization: compute and save numerical results
# ============================================================
optimizer = optax.adam(learning_rate=lr)

theta_history = {L: [] for L in vqe_layer_list}
ancilla_p1_stats_by_layer = {}
final_theta_periodic_only_rmsdist_by_layer = {}
energy_traces_by_layer = {}
grad_norm_traces_by_layer = {}

# These store the optimization-time states and gradients at sample_iters.
# They are later used to compute the QFIM eigenbasis gradient-sector weights.
theta_sample_traces_by_layer = {}
grad_sample_traces_by_layer = {}

cmap = matplotlib.colormaps.get_cmap("viridis")

for current_layer in tqdm(vqe_layer_list, desc="Layers (VQE)", unit="layer"):
    n_total_params = n_param_per_layer * current_layer

    def energy_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return energy_from_rho_keep(
            rho_keep_sequential_dpqc(theta, n_layer=current_layer)
        )

    energy_and_grad = jax.jit(jax.value_and_grad(energy_fn))

    @jax.jit
    def optimization_step_and_measure(theta, opt_state, g_old):
        updates, new_opt_state = optimizer.update(g_old, opt_state, theta)

        theta_new = wrap_theta_periodic_only(
            optax.apply_updates(theta, updates),
            n_layer=current_layer,
        )

        e_new, g_new = energy_and_grad(theta_new)
        g_new_norm = jnp.linalg.norm(g_new)

        return theta_new, new_opt_state, e_new, g_new, g_new_norm

    best_final_theta = None
    best_final_energy = np.inf
    all_energy_traces = []
    all_gradnorm_traces = []
    all_theta_sample_traces = []
    all_grad_sample_traces = []

    keys = jax.random.split(jax.random.PRNGKey(current_layer), num_runs)

    for i in tqdm(
        range(num_runs),
        desc=f"Runs (L={current_layer})",
        unit="run",
        leave=False,
    ):
        theta = jax.random.uniform(
            keys[i],
            shape=(n_total_params,),
            dtype=REAL_DTYPE,
            minval=jnp.asarray(-jnp.pi, dtype=REAL_DTYPE),
            maxval=jnp.asarray(jnp.pi, dtype=REAL_DTYPE),
        )

        opt_state = optimizer.init(theta)

        e_current, g_current = energy_and_grad(theta)
        trace = [float(e_current)]
        grad_trace = [float(jnp.linalg.norm(g_current))]

        theta_sample_trace = []
        grad_sample_trace = []

        if 0 in sample_iter_set:
            theta_sample_trace.append(jax_to_np(theta, dtype=NP_REAL_DTYPE))
            grad_sample_trace.append(jax_to_np(g_current, dtype=NP_REAL_DTYPE))

        for step_idx in range(1, steps + 1):
            (
                theta,
                opt_state,
                e_current,
                g_current,
                g_current_norm,
            ) = optimization_step_and_measure(
                theta,
                opt_state,
                g_current,
            )

            trace.append(float(e_current))
            grad_trace.append(float(g_current_norm))

            if step_idx in sample_iter_set:
                theta_sample_trace.append(jax_to_np(theta, dtype=NP_REAL_DTYPE))
                grad_sample_trace.append(jax_to_np(g_current, dtype=NP_REAL_DTYPE))

        all_energy_traces.append(np.asarray(trace, dtype=NP_REAL_DTYPE))
        all_gradnorm_traces.append(np.asarray(grad_trace, dtype=NP_REAL_DTYPE))
        all_theta_sample_traces.append(
            np.asarray(theta_sample_trace, dtype=NP_REAL_DTYPE)
        )
        all_grad_sample_traces.append(
            np.asarray(grad_sample_trace, dtype=NP_REAL_DTYPE)
        )
        theta_history[current_layer].append(jax_to_np(theta))

        final_e = trace[-1]

        if final_e < best_final_energy:
            best_final_energy = final_e
            best_final_theta = jax_to_np(theta)

    theta_history[current_layer] = np.stack(theta_history[current_layer], axis=0)
    theta_runs_jnp = jnp.asarray(theta_history[current_layer], dtype=REAL_DTYPE)

    theta_ref_jnp = jnp.asarray(best_final_theta, dtype=REAL_DTYPE)

    d_theta_runs = jax.vmap(
        lambda th: rms_theta_distance_periodic_only(
            th,
            theta_ref_jnp,
            n_layer=current_layer,
        )
    )(theta_runs_jnp)

    final_theta_periodic_only_rmsdist_by_layer[current_layer] = jax_to_np(
        d_theta_runs,
        dtype=NP_REAL_DTYPE,
    )

    energy_data = np.stack(all_energy_traces, axis=0)
    gradnorm_data = np.stack(all_gradnorm_traces, axis=0)
    theta_sample_data = np.stack(all_theta_sample_traces, axis=0)
    grad_sample_data = np.stack(all_grad_sample_traces, axis=0)

    energy_traces_by_layer[current_layer] = energy_data
    grad_norm_traces_by_layer[current_layer] = gradnorm_data
    theta_sample_traces_by_layer[current_layer] = theta_sample_data
    grad_sample_traces_by_layer[current_layer] = grad_sample_data

    best_tc_circ = create_dpqc(
        jnp.asarray(best_final_theta, dtype=REAL_DTYPE),
        n_layer=current_layer,
        num_system=num_system_qubits,
    )

    save_circuit_matplotlib_png(
        best_tc_circ,
        os.path.join(circuit_dir, f"optimized_circuit_L{current_layer}.png"),
        n_qubits=num_system_qubits + current_layer,
        dpi=SAVE_DPI,
        hide_params=True,
    )

    p1_runs = np.array(
        jax.jit(
            jax.vmap(
                lambda th: ancilla_p1_sequential_dpqc(
                    th,
                    n_layer=current_layer,
                )
            )
        )(theta_runs_jnp),
        dtype=NP_REAL_DTYPE,
    )

    anc_ids = np.arange(
        num_system_qubits,
        num_system_qubits + current_layer,
        dtype=NP_INT_DTYPE,
    )

    ancilla_p1_stats_by_layer[current_layer] = {
        "ancilla_qubits": anc_ids.copy(),
        "p1_runs": p1_runs,
        "mean": np.mean(p1_runs, axis=0),
        "var": np.var(p1_runs, axis=0, ddof=0),
        "std": np.std(p1_runs, axis=0, ddof=0),
    }

    mean_trace = np.mean(energy_data, axis=0)
    std_trace = np.std(energy_data, axis=0)

    success_rate_per_step = np.mean(
        np.abs(energy_data - smallest_eigval) <= tolerance,
        axis=0,
    )

    success_rates_history[current_layer] = success_rate_per_step

    final_stats["layer"].append(current_layer)
    final_stats["success_rate"].append(success_rate_per_step[-1])
    final_stats["mean_energy"].append(mean_trace[-1])
    final_stats["std_energy"].append(std_trace[-1])


vqe_optimization_result_path = os.path.join(
    energy_results_dir,
    "vqe_optimization_histories.npz",
)

save_npz_result(
    vqe_optimization_result_path,
    h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
    tolerance=np.asarray(tolerance, dtype=NP_REAL_DTYPE),
    steps=np.asarray(steps, dtype=NP_INT_DTYPE),
    num_runs=np.asarray(num_runs, dtype=NP_INT_DTYPE),
    lr=np.asarray(lr, dtype=NP_REAL_DTYPE),
    smallest_eigval=np.asarray(smallest_eigval, dtype=NP_REAL_DTYPE),
    sample_every=np.asarray(sample_every, dtype=NP_INT_DTYPE),
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    vqe_layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    qfim_layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
    final_stats_layer=np.asarray(final_stats["layer"], dtype=NP_INT_DTYPE),
    final_stats_success_rate=np.asarray(final_stats["success_rate"], dtype=NP_REAL_DTYPE),
    final_stats_mean_energy=np.asarray(final_stats["mean_energy"], dtype=NP_REAL_DTYPE),
    final_stats_std_energy=np.asarray(final_stats["std_energy"], dtype=NP_REAL_DTYPE),
    **_layer_arrays_for_npz(energy_traces_by_layer, "energy_traces"),
    **_layer_arrays_for_npz(grad_norm_traces_by_layer, "grad_norm_traces"),
    **_layer_arrays_for_npz(theta_history, "theta_final"),
    **_layer_arrays_for_npz(theta_sample_traces_by_layer, "theta_samples"),
    **_layer_arrays_for_npz(grad_sample_traces_by_layer, "grad_samples"),
    **_layer_arrays_for_npz(success_rates_history, "success_rates"),
    **_layer_arrays_for_npz(final_theta_periodic_only_rmsdist_by_layer, "theta_rmsdist"),
    **{
        f"L{int(L)}_ancilla_qubits": data["ancilla_qubits"]
        for L, data in ancilla_p1_stats_by_layer.items()
    },
    **{
        f"L{int(L)}_ancilla_p1_runs": data["p1_runs"]
        for L, data in ancilla_p1_stats_by_layer.items()
    },
    **{
        f"L{int(L)}_ancilla_p1_mean": data["mean"]
        for L, data in ancilla_p1_stats_by_layer.items()
    },
    **{
        f"L{int(L)}_ancilla_p1_var": data["var"]
        for L, data in ancilla_p1_stats_by_layer.items()
    },
    **{
        f"L{int(L)}_ancilla_p1_std": data["std"]
        for L, data in ancilla_p1_stats_by_layer.items()
    },
)


# ------------------------------------------------------------
# Final-energy success probabilities at multiple error thresholds
# ------------------------------------------------------------
# Keep this archive separate from the legacy single-tolerance history above:
# the latter is an optimization-step time series, while this analysis uses
# only the final energy from each independent optimization trial.
final_energies_by_layer = {
    int(layer): np.asarray(
        energy_traces_by_layer[int(layer)][:, -1],
        dtype=NP_REAL_DTYPE,
    )
    for layer in vqe_layer_list
}

multiple_tolerance_success = _multiple_tolerance_success_statistics(
    final_energies_by_layer,
    layers=vqe_layer_list,
    thresholds=success_probability_thresholds,
    ground_energy=smallest_eigval,
    num_trials=num_runs,
)

multiple_tolerance_success_result_path = os.path.join(
    energy_results_dir,
    "vqe_success_probability_multiple_tolerances.npz",
)

save_npz_result(
    multiple_tolerance_success_result_path,
    h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
    final_iteration=np.asarray(steps, dtype=NP_INT_DTYPE),
    layers=multiple_tolerance_success["layers"],
    thresholds=multiple_tolerance_success["thresholds"],
    # ``tolerances`` is an explicit compatibility alias for visualization.
    tolerances=multiple_tolerance_success["thresholds"],
    num_trials=multiple_tolerance_success["num_trials"],
    num_runs=multiple_tolerance_success["num_trials"],
    R=multiple_tolerance_success["num_trials"],
    trial_indices=np.arange(1, num_runs + 1, dtype=NP_INT_DTYPE),
    ground_energy=multiple_tolerance_success["ground_energy"],
    smallest_eigval=multiple_tolerance_success["ground_energy"],
    final_energies=multiple_tolerance_success["final_energies"],
    # Final energies are stored without post-processing; retain this alias to
    # make the distinction from the clipped error matrix self-documenting.
    raw_final_energies=multiple_tolerance_success["final_energies"],
    raw_final_energy_errors=multiple_tolerance_success[
        "raw_final_energy_errors"
    ],
    final_energy_errors=multiple_tolerance_success["final_energy_errors"],
    success_indicators=multiple_tolerance_success["success_indicators"],
    success_counts=multiple_tolerance_success["success_counts"],
    success_probabilities=multiple_tolerance_success[
        "success_probabilities"
    ],
    energy_error_clipped_at_zero=np.asarray(True, dtype=np.bool_),
    success_comparison=np.asarray("<="),
    probability_denominator=np.asarray("num_trials"),
    **_layer_arrays_for_npz(
        {
            int(layer): multiple_tolerance_success["final_energies"][layer_idx]
            for layer_idx, layer in enumerate(
                multiple_tolerance_success["layers"]
            )
        },
        "final_energies",
    ),
    **_layer_arrays_for_npz(
        {
            int(layer): multiple_tolerance_success[
                "raw_final_energy_errors"
            ][layer_idx]
            for layer_idx, layer in enumerate(
                multiple_tolerance_success["layers"]
            )
        },
        "raw_final_energy_errors",
    ),
    **_layer_arrays_for_npz(
        {
            int(layer): multiple_tolerance_success["final_energy_errors"][
                layer_idx
            ]
            for layer_idx, layer in enumerate(
                multiple_tolerance_success["layers"]
            )
        },
        "final_energy_errors",
    ),
    **_layer_arrays_for_npz(
        {
            int(layer): multiple_tolerance_success["success_indicators"][
                layer_idx
            ]
            for layer_idx, layer in enumerate(
                multiple_tolerance_success["layers"]
            )
        },
        "success_indicators",
    ),
    **_layer_arrays_for_npz(
        {
            int(layer): multiple_tolerance_success["success_counts"][layer_idx]
            for layer_idx, layer in enumerate(
                multiple_tolerance_success["layers"]
            )
        },
        "success_counts",
    ),
    **_layer_arrays_for_npz(
        {
            int(layer): multiple_tolerance_success[
                "success_probabilities"
            ][layer_idx]
            for layer_idx, layer in enumerate(
                multiple_tolerance_success["layers"]
            )
        },
        "success_probabilities",
    ),
)


# ============================================================
# Random-parameter QFIM: compute and save numerical results
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
PARTICIPATION_EFFECTIVE_RANK_EPS = cfg.PARTICIPATION_EFFECTIVE_RANK_EPS
QFIM_GRAD_ALIGNMENT_NORM_EPS = cfg.QFIM_GRAD_ALIGNMENT_NORM_EPS
QFIM_DEGENERACY_RTOL = cfg.QFIM_DEGENERACY_RTOL
QFIM_DEGENERACY_ATOL = cfg.QFIM_DEGENERACY_ATOL

RUN_QFIM_EFFECTIVE_RANK_RANDOM_POINTS = (
    cfg.RUN_QFIM_EFFECTIVE_RANK_RANDOM_POINTS
)
RUN_QFIM_EFFECTIVE_RANK_OPTIMIZATION_PATH = (
    cfg.RUN_QFIM_EFFECTIVE_RANK_OPTIMIZATION_PATH
)
RUN_QFIM_SPECTRAL_GRADIENT_SUMMARY = (
    cfg.RUN_QFIM_SPECTRAL_GRADIENT_SUMMARY
)

# Thresholds used for large-sector gradient-weight diagnostics.
# Keep this broad set unless you also want to reduce the gradient-sector plots.
THRESHOLDS = tuple(float(t) for t in cfg.GRADIENT_SECTOR_THRESHOLDS)


def psd_rank_and_desc_eigs(F: jnp.ndarray):
    evals = jnp.clip(jnp.linalg.eigvalsh(_hermitian(F)), a_min=0.0)
    rank, _ = threshold_psd_eigvals_for_rank(evals)

    return rank, evals[::-1]


def participation_effective_rank_from_eigvals(
    eigvals,
    *,
    eps=PARTICIPATION_EFFECTIVE_RANK_EPS,
):
    """Return ``Tr(A)^2 / Tr(A^2)`` for a PSD spectrum.

    The zero-matrix convention is participation rank 0.  Inputs are clipped
    at zero so that round-off-sized negative eigenvalues cannot inflate the
    denominator.
    """
    eigvals = np.clip(np.asarray(eigvals, dtype=NP_REAL_DTYPE), 0.0, None)
    trace = NP_REAL_DTYPE(np.sum(eigvals))
    frobenius_norm_sq = NP_REAL_DTYPE(np.sum(eigvals * eigvals))

    if frobenius_norm_sq <= float(eps):
        return NP_REAL_DTYPE(0.0)

    return NP_REAL_DTYPE((trace * trace) / frobenius_norm_sq)


def qfim_spectral_summary_from_eigvals(
    eigvals,
    *,
    rank_threshold=QFIM_EFFECTIVE_RANK_THRESHOLD,
    participation_eps=PARTICIPATION_EFFECTIVE_RANK_EPS,
):
    """Derive all trace/rank diagnostics from one clipped QFIM spectrum."""
    eigvals = np.clip(np.asarray(eigvals, dtype=NP_REAL_DTYPE), 0.0, None)
    eigvals_desc = np.sort(eigvals)[::-1]
    active = eigvals_desc > float(rank_threshold)
    threshold_rank = int(np.count_nonzero(active))
    trace = NP_REAL_DTYPE(np.sum(eigvals_desc))
    frobenius_norm_sq = NP_REAL_DTYPE(
        np.sum(eigvals_desc * eigvals_desc)
    )
    participation_rank = participation_effective_rank_from_eigvals(
        eigvals_desc,
        eps=participation_eps,
    )
    largest = (
        NP_REAL_DTYPE(eigvals_desc[0])
        if eigvals_desc.size
        else NP_REAL_DTYPE(np.nan)
    )

    if threshold_rank:
        smallest_active = NP_REAL_DTYPE(eigvals_desc[active][-1])
        condition_active = NP_REAL_DTYPE(largest / smallest_active)
    else:
        smallest_active = NP_REAL_DTYPE(np.nan)
        condition_active = NP_REAL_DTYPE(np.nan)

    return {
        "evals": eigvals_desc,
        "qfim_threshold_rank": NP_INT_DTYPE(threshold_rank),
        "threshold_rank": NP_INT_DTYPE(threshold_rank),
        "qfim_participation_rank": participation_rank,
        "participation_rank": participation_rank,
        "qfim_trace": trace,
        "trace": trace,
        "qfim_frobenius_norm_sq": frobenius_norm_sq,
        "frobenius_norm_sq": frobenius_norm_sq,
        "largest_qfim_eigenvalue": largest,
        "largest_eigenvalue": largest,
        "smallest_active_qfim_eigenvalue": smallest_active,
        "smallest_active_eigenvalue": smallest_active,
        "condition_number_active": condition_active,
        "active_condition_number": condition_active,
    }


def _spectral_threshold_tag(value):
    """Return compact stable tags such as ``1e0`` and ``5e-1``."""
    mantissa, exponent = f"{float(value):.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


_QFIM_DEGENERACY_WARNING_EMITTED = False


def _warn_if_qfim_spectrum_is_degenerate(eigvals_desc):
    global _QFIM_DEGENERACY_WARNING_EMITTED

    eigvals_desc = np.asarray(eigvals_desc, dtype=NP_REAL_DTYPE)
    if _QFIM_DEGENERACY_WARNING_EMITTED or eigvals_desc.size < 2:
        return

    adjacent_is_degenerate = np.isclose(
        eigvals_desc[:-1],
        eigvals_desc[1:],
        rtol=QFIM_DEGENERACY_RTOL,
        atol=QFIM_DEGENERACY_ATOL,
    )
    if np.any(adjacent_is_degenerate):
        warnings.warn(
            "Individual gradient weights can be basis-dependent inside a "
            "degenerate QFIM eigenspace. Aggregated weight over the full "
            "degenerate subspace is basis-independent.",
            RuntimeWarning,
            stacklevel=2,
        )
        _QFIM_DEGENERACY_WARNING_EMITTED = True


def qfim_spectral_gradient_diagnostics_from_matrix(
    F,
    grad,
    *,
    rank_threshold=QFIM_EFFECTIVE_RANK_THRESHOLD,
    participation_eps=PARTICIPATION_EFFECTIVE_RANK_EPS,
    grad_norm_eps=QFIM_GRAD_ALIGNMENT_NORM_EPS,
    sector_thresholds=THRESHOLDS,
    warn_degenerate=True,
):
    """Compute all QFIM/gradient diagnostics from one eigendecomposition.

    Individual gradient weights can be basis-dependent inside a degenerate
    QFIM eigenspace. Aggregated weight over the full degenerate subspace is
    basis-independent.

    The QFIM is Hermitianized before ``eigh`` and round-off-sized negative
    eigenvalues are clipped to zero.  Gradient weights are normalized by the
    directly evaluated Euclidean norm ``||g||^2``.  If that norm is too
    small, all normalized-gradient diagnostics are represented by NaN.
    """
    F = jnp.asarray(F)
    F = 0.5 * (F + jnp.conjugate(F.T))
    grad = jnp.asarray(grad, dtype=REAL_DTYPE).reshape((-1,))

    if int(F.shape[0]) != int(F.shape[1]) or int(F.shape[0]) != int(grad.size):
        raise ValueError(
            f"Dimension mismatch: F.shape={F.shape}, grad.shape={grad.shape}"
        )

    evals, evecs = jnp.linalg.eigh(F)
    evals = jnp.clip(jnp.real(evals), a_min=0.0)
    order = jnp.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    grad_projection = grad.astype(evecs.dtype)
    coeffs = jnp.conjugate(evecs).T @ grad_projection
    coeff_abs2 = jnp.real(coeffs * jnp.conjugate(coeffs))
    gradient_norm_sq = jnp.real(jnp.vdot(grad, grad))
    projected_norm_sq = jnp.sum(coeff_abs2)
    is_valid_gradient = gradient_norm_sq > float(grad_norm_eps)
    weights = jnp.where(
        is_valid_gradient,
        coeff_abs2 / gradient_norm_sq,
        jnp.full_like(coeff_abs2, jnp.nan),
    )

    evals_np = jax_to_np(evals, dtype=NP_REAL_DTYPE)
    coeffs_np = jax_to_np(coeffs, dtype=NP_COMPLEX_DTYPE)
    coeff_abs2_np = jax_to_np(coeff_abs2, dtype=NP_REAL_DTYPE)
    weights_np = jax_to_np(weights, dtype=NP_REAL_DTYPE)
    gradient_norm_sq_np = NP_REAL_DTYPE(jax.device_get(gradient_norm_sq))
    projected_norm_sq_np = NP_REAL_DTYPE(jax.device_get(projected_norm_sq))
    valid_gradient = bool(jax.device_get(is_valid_gradient))
    active_by_rank_threshold = evals_np > float(rank_threshold)

    # Truncated Moore--Penrose inverse in the QFIM eigenbasis:
    #
    #   chi_H^(tau) = sum_{lambda_i > tau} |v_i^dagger g|^2 / lambda_i.
    #
    # Evaluating this directly in the eigenbasis avoids explicitly forming an
    # ill-conditioned pseudoinverse.  The complementary squared norm measures
    # the part of the Hamiltonian gradient outside the numerically retained
    # image of F.  Both quantities reuse this routine's single QFIM
    # eigendecomposition.
    if np.any(active_by_rank_threshold):
        hamiltonian_qfim_sensitivity = NP_REAL_DTYPE(
            np.sum(
                coeff_abs2_np[active_by_rank_threshold]
                / evals_np[active_by_rank_threshold]
            )
        )
        gradient_image_projection_norm_sq = NP_REAL_DTYPE(
            np.sum(coeff_abs2_np[active_by_rank_threshold])
        )
    else:
        hamiltonian_qfim_sensitivity = NP_REAL_DTYPE(0.0)
        gradient_image_projection_norm_sq = NP_REAL_DTYPE(0.0)

    gradient_image_residual_norm_sq = NP_REAL_DTYPE(
        np.sum(coeff_abs2_np[~active_by_rank_threshold])
    )
    gradient_image_residual_norm = NP_REAL_DTYPE(
        np.sqrt(max(float(gradient_image_residual_norm_sq), 0.0))
    )
    gradient_image_residual_fraction = (
        NP_REAL_DTYPE(
            gradient_image_residual_norm_sq / gradient_norm_sq_np
        )
        if valid_gradient
        else NP_REAL_DTYPE(np.nan)
    )

    spectral = qfim_spectral_summary_from_eigvals(
        evals_np,
        rank_threshold=rank_threshold,
        participation_eps=participation_eps,
    )
    trace = float(spectral["qfim_trace"])

    if trace > float(participation_eps):
        lambda_fraction = evals_np / trace
        cumulative_lambda_fraction = np.cumsum(lambda_fraction)
    else:
        lambda_fraction = np.full_like(evals_np, np.nan)
        cumulative_lambda_fraction = np.full_like(evals_np, np.nan)

    if valid_gradient:
        gradient_weight_sum = NP_REAL_DTYPE(np.sum(weights_np))
        weight_square_sum = NP_REAL_DTYPE(np.sum(weights_np * weights_np))
        gradient_participation_rank = (
            NP_REAL_DTYPE(1.0 / weight_square_sum)
            if weight_square_sum > float(participation_eps)
            else NP_REAL_DTYPE(np.nan)
        )
        gradient_weighted_eigenvalue = NP_REAL_DTYPE(
            np.sum(weights_np * evals_np)
        )
        cumulative_gradient_weight = np.cumsum(weights_np)
        parseval_relative_error = NP_REAL_DTYPE(
            abs(projected_norm_sq_np - gradient_norm_sq_np)
            / max(abs(float(gradient_norm_sq_np)), float(grad_norm_eps))
        )
    else:
        gradient_weight_sum = NP_REAL_DTYPE(np.nan)
        gradient_participation_rank = NP_REAL_DTYPE(np.nan)
        gradient_weighted_eigenvalue = NP_REAL_DTYPE(np.nan)
        cumulative_gradient_weight = np.full_like(evals_np, np.nan)
        parseval_relative_error = NP_REAL_DTYPE(np.nan)

    diagnostics = {
        **spectral,
        "weights": weights_np,
        "w_grad": weights_np,
        "coeffs": coeffs_np,
        "coeff_abs2": coeff_abs2_np,
        "eig_index": np.arange(1, evals_np.size + 1, dtype=NP_INT_DTYPE),
        "lambda_fraction": np.asarray(
            lambda_fraction,
            dtype=NP_REAL_DTYPE,
        ),
        "cumulative_lambda_fraction": np.asarray(
            cumulative_lambda_fraction,
            dtype=NP_REAL_DTYPE,
        ),
        "cumulative_gradient_weight": np.asarray(
            cumulative_gradient_weight,
            dtype=NP_REAL_DTYPE,
        ),
        "active_by_rank_threshold": np.asarray(
            active_by_rank_threshold,
            dtype=np.bool_,
        ),
        "gradient_norm_sq": gradient_norm_sq_np,
        "projected_gradient_norm_sq": projected_norm_sq_np,
        "hamiltonian_qfim_sensitivity": hamiltonian_qfim_sensitivity,
        "chi_hamiltonian": hamiltonian_qfim_sensitivity,
        "gradient_image_projection_norm_sq": (
            gradient_image_projection_norm_sq
        ),
        "gradient_image_residual_norm_sq": gradient_image_residual_norm_sq,
        "gradient_image_residual_norm": gradient_image_residual_norm,
        "gradient_image_residual_fraction": gradient_image_residual_fraction,
        "gradient_weight_sum": gradient_weight_sum,
        "gradient_participation_rank": gradient_participation_rank,
        "gradient_effective_dimension": gradient_participation_rank,
        "gradient_weighted_qfim_eigenvalue": gradient_weighted_eigenvalue,
        "lambda_grad_mean": gradient_weighted_eigenvalue,
        "parseval_relative_error": parseval_relative_error,
    }

    sector_weights = {}
    for threshold in sector_thresholds:
        metric_name = f"grad_weight_above_{_spectral_threshold_tag(threshold)}"
        sector_weight = (
            NP_REAL_DTYPE(np.sum(weights_np[evals_np > float(threshold)]))
            if valid_gradient
            else NP_REAL_DTYPE(np.nan)
        )
        diagnostics[metric_name] = sector_weight
        sector_weights[float(threshold)] = sector_weight
    diagnostics["sector_weights"] = sector_weights

    if warn_degenerate:
        _warn_if_qfim_spectrum_is_degenerate(evals_np)

    return diagnostics


def qfim_spectral_gradient_diagnostics_at_point(
    theta,
    grad,
    qfim_fn,
    **kwargs,
):
    """Evaluate one QFIM once and derive all spectral-gradient diagnostics."""
    theta = jnp.asarray(theta, dtype=REAL_DTYPE)
    F = qfim_fn(theta)
    return qfim_spectral_gradient_diagnostics_from_matrix(
        F,
        grad,
        **kwargs,
    )


def make_mixed_state_qfim_matrix_fn_for_layer_sequential(
    n_layer: int,
    *,
    rho_from_rho5_fn,
    n_keep: int,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    """Build the SLD-QFIM for a reduced state derived from the kept rho5.

    ``rho_from_rho5_fn`` is a static Python callable applied to the five-qubit
    state returned by ``rho_keep_sequential_dpqc``.  This keeps the SLD
    construction identical for keep=(0,1,2,3) and keep=(0,1,2,3,4), while
    allowing the former to trace out wire 4 and the latter to use rho5
    directly.
    """
    n_keep = int(n_keep)
    jvp_chunk = int(jvp_chunk)
    if not 1 <= n_keep <= num_system_qubits:
        raise ValueError(
            f"n_keep must be in [1, {num_system_qubits}], got {n_keep}."
        )
    if jvp_chunk <= 0:
        raise ValueError(f"jvp_chunk must be positive, got {jvp_chunk}.")

    dim_state = 2**n_keep
    dim_vec = (2**n_keep) ** 2

    @jax.jit
    def rho_sub_fn(theta: jnp.ndarray) -> jnp.ndarray:
        rho5 = rho_keep_sequential_dpqc(theta, n_layer=n_layer)
        rho_sub = rho_from_rho5_fn(rho5)
        return _hermitian(rho_sub)

    def qfim_mixed_state(theta: jnp.ndarray) -> jnp.ndarray:
        rho, rho_jvp = jax.linearize(rho_sub_fn, theta)
        rho = _hermitian(rho)
        if rho.shape != (dim_state, dim_state):
            raise ValueError(
                "rho_from_rho5_fn returned an incompatible density-matrix "
                f"shape {rho.shape}; expected {(dim_state, dim_state)}."
            )

        lam, U = jnp.linalg.eigh(rho)
        lam = jnp.clip(lam, a_min=0.0)

        lam_sum = lam[:, None] + lam[None, :]

        sqrtW = jnp.sqrt(
            jnp.where(
                lam_sum > EIG_SUM_EPS,
                2.0 / lam_sum,
                0.0,
            )
        )

        Udag = jnp.conjugate(U).T
        n_params = int(theta.shape[0])
        eye = jnp.eye(n_params, dtype=theta.dtype)

        def _to_eig(d: jnp.ndarray) -> jnp.ndarray:
            return Udag @ d @ U

        blocks = []

        for s in range(0, n_params, jvp_chunk):
            V = eye[s: min(s + jvp_chunk, n_params), :]
            drho_B = jax.vmap(rho_jvp)(V)

            drho_B = 0.5 * (
                drho_B + jnp.conjugate(jnp.swapaxes(drho_B, 1, 2))
            )

            Cflat_B = jnp.reshape(
                jax.vmap(_to_eig)(drho_B) * sqrtW[None, :, :],
                (V.shape[0], dim_vec),
            )

            blocks.append(Cflat_B)

        Cflat = jnp.concatenate(blocks, axis=0)
        F_red = jnp.real(Cflat @ jnp.conjugate(Cflat).T)

        return 0.5 * (F_red + F_red.T)

    return qfim_mixed_state


def make_reduced0123_qfim_matrix_fn_for_layer_sequential(
    n_layer: int,
    *,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    """Compatibility wrapper for the SLD-QFIM of reduced keep=(0,1,2,3)."""
    return make_mixed_state_qfim_matrix_fn_for_layer_sequential(
        n_layer=n_layer,
        rho_from_rho5_fn=rho4_from_rho5,
        n_keep=4,
        jvp_chunk=jvp_chunk,
    )


def make_reduced01234_qfim_matrix_fn_for_layer_sequential(
    n_layer: int,
    *,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    """Return the SLD-QFIM for the full kept state on wires (0,1,2,3,4)."""
    return make_mixed_state_qfim_matrix_fn_for_layer_sequential(
        n_layer=n_layer,
        rho_from_rho5_fn=lambda rho5: rho5,
        n_keep=5,
        jvp_chunk=jvp_chunk,
    )


def make_reduced0123_rho_rank_fn_for_layer_sequential(n_layer: int):
    @jax.jit
    def rho_rank_reduced0123(theta: jnp.ndarray) -> jnp.ndarray:
        rho5 = rho_keep_sequential_dpqc(theta, n_layer=n_layer)
        rho4 = _hermitian(rho4_from_rho5(rho5))
        evals = jnp.clip(jnp.linalg.eigvalsh(rho4), a_min=0.0)
        threshold = jnp.asarray(QFIM_EFFECTIVE_RANK_THRESHOLD, dtype=evals.dtype)

        return jnp.sum(evals > threshold)

    return rho_rank_reduced0123


qfim_rank_reduced_0123_by_layer = {}
qfim_eigs_reduced_0123_by_layer = {}
qfim_rho_rank_reduced_0123_by_layer = {}

qfim_eigsum_reduced_0123_by_layer = {}
qfim_abs_entry_sum_reduced_0123_by_layer = {}
qfim_participation_rank_random_by_layer = {}
qfim_frobenius_norm_sq_random_by_layer = {}
qfim_largest_eigenvalue_random_by_layer = {}
qfim_smallest_active_eigenvalue_random_by_layer = {}
qfim_active_condition_number_random_by_layer = {}
hamiltonian_qfim_sensitivity_random_by_layer = {}
gradient_image_residual_norm_sq_random_by_layer = {}
qfim_active_rank_random_by_layer = {}

# Hamiltonian-direction sensitivity for the full five-qubit kept state.
# These have dedicated names so every existing keep0123 variable and archive
# remains unchanged.
hamiltonian_qfim_sensitivity_random_keep01234_by_layer = {}
gradient_image_residual_norm_sq_random_keep01234_by_layer = {}
qfim_active_rank_random_keep01234_by_layer = {}

qfim_eigs_dir = os.path.join(qfim_fig_dir, "qfim_eigs")
qfim_eigs_dir_red4 = os.path.join(qfim_eigs_dir, "reduced_keep_0123")

os.makedirs(qfim_eigs_dir_red4, exist_ok=True)

for L in tqdm(
    qfim_layer_list,
    desc="Layers (QFIM; keep=(0,1,2,3) and keep=(0,1,2,3,4))",
    unit="layer",
):
    n_params = n_param_per_layer * L

    thetas_L = jax.random.uniform(
        jax.random.PRNGKey(QFIM_SAMPLE_SEED_BASE + int(L)),
        shape=(NUM_QFIM_SAMPLES, n_params),
        dtype=REAL_DTYPE,
        minval=jnp.asarray(-jnp.pi, dtype=REAL_DTYPE),
        maxval=jnp.asarray(jnp.pi, dtype=REAL_DTYPE),
    )

    red4_qfim_fn = make_reduced0123_qfim_matrix_fn_for_layer_sequential(
        n_layer=L,
        jvp_chunk=RED_JVP_CHUNK,
    )
    red4_rho_rank_fn = make_reduced0123_rho_rank_fn_for_layer_sequential(
        n_layer=L,
    )
    red5_qfim_fn = make_reduced01234_qfim_matrix_fn_for_layer_sequential(
        n_layer=L,
        jvp_chunk=RED_JVP_CHUNK,
    )

    def random_point_energy_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return energy_from_rho_keep(
            rho_keep_sequential_dpqc(theta, n_layer=L)
        )

    # One compiled Hamiltonian-gradient callable per layer.  The Hamiltonian
    # acts only on kept wires (0,1,2,3), so this is the gradient associated
    # with the same reduced state used to construct F4.
    random_point_energy_grad_fn = jax.jit(jax.grad(random_point_energy_fn))

    rr4_list = []
    eigs4_list = []
    rho_rank4_list = []
    eigsum4_list = []
    abs_entry_sum4_list = []
    participation_rank4_list = []
    frobenius_norm_sq4_list = []
    largest_eigenvalue4_list = []
    smallest_active_eigenvalue4_list = []
    active_condition_number4_list = []
    chi_hamiltonian4_list = []
    gradient_image_residual_norm_sq4_list = []
    active_rank4_list = []
    chi_hamiltonian5_list = []
    gradient_image_residual_norm_sq5_list = []
    active_rank5_list = []

    for s in tqdm(
        range(NUM_QFIM_SAMPLES),
        desc=f"QFIM samples for both kept states (L={L})",
        unit="sample",
        leave=False,
    ):
        th = thetas_L[s]

        F4 = red4_qfim_fn(th)
        grad_hamiltonian = random_point_energy_grad_fn(th)
        diagnostics4 = qfim_spectral_gradient_diagnostics_from_matrix(
            F4,
            grad_hamiltonian,
            sector_thresholds=(),
            warn_degenerate=False,
        )
        rho_rank4 = red4_rho_rank_fn(th)

        # The Hamiltonian gradient is the same derivative used above because
        # H acts on the full kept rho5.  F5 and its eigendecomposition are each
        # evaluated exactly once for this sample.
        F5 = red5_qfim_fn(th)
        diagnostics5 = qfim_spectral_gradient_diagnostics_from_matrix(
            F5,
            grad_hamiltonian,
            sector_thresholds=(),
            warn_degenerate=False,
        )

        evals4_np = np.asarray(
            diagnostics4["evals"],
            dtype=NP_REAL_DTYPE,
        )
        F4_np = jax_to_np(F4, dtype=NP_REAL_DTYPE)

        rr4_list.append(int(diagnostics4["qfim_threshold_rank"]))
        eigs4_list.append(evals4_np)
        rho_rank4_list.append(int(jax.device_get(rho_rank4)))

        eigsum4_list.append(NP_REAL_DTYPE(np.sum(evals4_np)))
        abs_entry_sum4_list.append(NP_REAL_DTYPE(np.sum(np.abs(F4_np))))

        # Rank, spectrum, trace-based diagnostics, and Hamiltonian sensitivity
        # all reuse the single eigendecomposition inside ``diagnostics4``.
        spectral4 = diagnostics4
        participation_rank4_list.append(
            spectral4["qfim_participation_rank"]
        )
        frobenius_norm_sq4_list.append(
            spectral4["qfim_frobenius_norm_sq"]
        )
        largest_eigenvalue4_list.append(
            spectral4["largest_qfim_eigenvalue"]
        )
        smallest_active_eigenvalue4_list.append(
            spectral4["smallest_active_qfim_eigenvalue"]
        )
        active_condition_number4_list.append(
            spectral4["condition_number_active"]
        )
        chi_hamiltonian4_list.append(
            diagnostics4["hamiltonian_qfim_sensitivity"]
        )
        gradient_image_residual_norm_sq4_list.append(
            diagnostics4["gradient_image_residual_norm_sq"]
        )
        active_rank4_list.append(
            diagnostics4["qfim_threshold_rank"]
        )
        chi_hamiltonian5_list.append(
            diagnostics5["hamiltonian_qfim_sensitivity"]
        )
        gradient_image_residual_norm_sq5_list.append(
            diagnostics5["gradient_image_residual_norm_sq"]
        )
        active_rank5_list.append(
            diagnostics5["qfim_threshold_rank"]
        )

    qfim_rank_reduced_0123_by_layer[L] = np.asarray(
        rr4_list,
        dtype=NP_INT_DTYPE,
    )

    qfim_eigs_reduced_0123_by_layer[L] = np.stack(eigs4_list, axis=0)

    qfim_rho_rank_reduced_0123_by_layer[L] = np.asarray(
        rho_rank4_list,
        dtype=NP_INT_DTYPE,
    )

    qfim_eigsum_reduced_0123_by_layer[L] = np.asarray(
        eigsum4_list,
        dtype=NP_REAL_DTYPE,
    )

    qfim_abs_entry_sum_reduced_0123_by_layer[L] = np.asarray(
        abs_entry_sum4_list,
        dtype=NP_REAL_DTYPE,
    )

    qfim_participation_rank_random_by_layer[L] = np.asarray(
        participation_rank4_list,
        dtype=NP_REAL_DTYPE,
    )
    qfim_frobenius_norm_sq_random_by_layer[L] = np.asarray(
        frobenius_norm_sq4_list,
        dtype=NP_REAL_DTYPE,
    )
    qfim_largest_eigenvalue_random_by_layer[L] = np.asarray(
        largest_eigenvalue4_list,
        dtype=NP_REAL_DTYPE,
    )
    qfim_smallest_active_eigenvalue_random_by_layer[L] = np.asarray(
        smallest_active_eigenvalue4_list,
        dtype=NP_REAL_DTYPE,
    )
    qfim_active_condition_number_random_by_layer[L] = np.asarray(
        active_condition_number4_list,
        dtype=NP_REAL_DTYPE,
    )
    hamiltonian_qfim_sensitivity_random_by_layer[L] = np.asarray(
        chi_hamiltonian4_list,
        dtype=NP_REAL_DTYPE,
    )
    gradient_image_residual_norm_sq_random_by_layer[L] = np.asarray(
        gradient_image_residual_norm_sq4_list,
        dtype=NP_REAL_DTYPE,
    )
    qfim_active_rank_random_by_layer[L] = np.asarray(
        active_rank4_list,
        dtype=NP_INT_DTYPE,
    )
    hamiltonian_qfim_sensitivity_random_keep01234_by_layer[L] = np.asarray(
        chi_hamiltonian5_list,
        dtype=NP_REAL_DTYPE,
    )
    gradient_image_residual_norm_sq_random_keep01234_by_layer[L] = np.asarray(
        gradient_image_residual_norm_sq5_list,
        dtype=NP_REAL_DTYPE,
    )
    qfim_active_rank_random_keep01234_by_layer[L] = np.asarray(
        active_rank5_list,
        dtype=NP_INT_DTYPE,
    )



qfim_random_points_result_path = os.path.join(
    qfim_results_dir,
    f"qfim_random_points_{keep_key}.npz",
)

save_npz_result(
    qfim_random_points_result_path,
    h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
    num_qfim_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
    qfim_sample_seed_base=np.asarray(QFIM_SAMPLE_SEED_BASE, dtype=NP_INT_DTYPE),
    qfim_effective_rank_threshold=np.asarray(QFIM_EFFECTIVE_RANK_THRESHOLD, dtype=NP_REAL_DTYPE),
    eig_sum_eps=np.asarray(EIG_SUM_EPS, dtype=NP_REAL_DTYPE),
    qfim_eig_plot_eps=np.asarray(QFIM_EIG_PLOT_EPS, dtype=NP_REAL_DTYPE),
    red_jvp_chunk=np.asarray(RED_JVP_CHUNK, dtype=NP_INT_DTYPE),
    layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
    grad_sector_thresholds=np.asarray(THRESHOLDS, dtype=NP_REAL_DTYPE),
    **_layer_arrays_for_npz(qfim_rank_reduced_0123_by_layer, "rank"),
    **_layer_arrays_for_npz(qfim_eigs_reduced_0123_by_layer, "eigs_desc"),
    **_layer_arrays_for_npz(qfim_rho_rank_reduced_0123_by_layer, "rho_rank"),
    **_layer_arrays_for_npz(qfim_eigsum_reduced_0123_by_layer, "trace"),
    **_layer_arrays_for_npz(qfim_abs_entry_sum_reduced_0123_by_layer, "abs_entry_sum"),
    **_layer_arrays_for_npz(qfim_rank_reduced_0123_by_layer, "threshold_rank"),
    **_layer_arrays_for_npz(
        qfim_participation_rank_random_by_layer,
        "participation_rank",
    ),
    **_layer_arrays_for_npz(
        qfim_frobenius_norm_sq_random_by_layer,
        "frobenius_norm_sq",
    ),
    **_layer_arrays_for_npz(
        qfim_largest_eigenvalue_random_by_layer,
        "largest_eigenvalue",
    ),
    **_layer_arrays_for_npz(
        qfim_smallest_active_eigenvalue_random_by_layer,
        "smallest_active_eigenvalue",
    ),
    **_layer_arrays_for_npz(
        qfim_active_condition_number_random_by_layer,
        "active_condition_number",
    ),
    **_layer_arrays_for_npz(
        hamiltonian_qfim_sensitivity_random_by_layer,
        "chi_hamiltonian",
    ),
    **_layer_arrays_for_npz(
        gradient_image_residual_norm_sq_random_by_layer,
        "gradient_image_residual_norm_sq",
    ),
    **_layer_arrays_for_npz(
        qfim_active_rank_random_by_layer,
        "active_rank",
    ),
)


def _build_hamiltonian_sensitivity_layer_statistics(
    layers,
    chi_by_layer,
    residual_norm_sq_by_layer,
    active_rank_by_layer,
):
    """Build the common per-layer NPZ schema for random/path sensitivity."""
    npz_arrays = {}
    chi_mean_by_layer = []
    chi_sem_by_layer = []
    chi_count_by_layer = []

    for L in layers:
        L = int(L)
        chi_hamiltonian = np.asarray(
            chi_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )
        residual_norm_sq = np.asarray(
            residual_norm_sq_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )
        active_rank = np.asarray(
            active_rank_by_layer[L],
            dtype=NP_INT_DTYPE,
        )
        chi_mean, chi_sem, chi_count = _finite_mean_sem(
            chi_hamiltonian,
            axis=0,
        )
        residual_mean, residual_sem, residual_count = _finite_mean_sem(
            residual_norm_sq,
            axis=0,
        )
        active_rank_mean, active_rank_sem, active_rank_count = _finite_mean_sem(
            active_rank,
            axis=0,
        )
        L_tag = f"L{L}"
        npz_arrays.update(
            {
                f"{L_tag}_chi_hamiltonian": chi_hamiltonian,
                f"{L_tag}_chi_hamiltonian_mean": np.asarray(
                    chi_mean,
                    dtype=NP_REAL_DTYPE,
                ),
                f"{L_tag}_chi_hamiltonian_sem": np.asarray(
                    chi_sem,
                    dtype=NP_REAL_DTYPE,
                ),
                f"{L_tag}_chi_hamiltonian_count": np.asarray(
                    chi_count,
                    dtype=NP_INT_DTYPE,
                ),
                f"{L_tag}_gradient_image_residual_norm_sq": residual_norm_sq,
                f"{L_tag}_gradient_image_residual_norm_sq_mean": np.asarray(
                    residual_mean,
                    dtype=NP_REAL_DTYPE,
                ),
                f"{L_tag}_gradient_image_residual_norm_sq_sem": np.asarray(
                    residual_sem,
                    dtype=NP_REAL_DTYPE,
                ),
                f"{L_tag}_gradient_image_residual_norm_sq_count": np.asarray(
                    residual_count,
                    dtype=NP_INT_DTYPE,
                ),
                f"{L_tag}_active_rank": active_rank,
                f"{L_tag}_active_rank_mean": np.asarray(
                    active_rank_mean,
                    dtype=NP_REAL_DTYPE,
                ),
                f"{L_tag}_active_rank_sem": np.asarray(
                    active_rank_sem,
                    dtype=NP_REAL_DTYPE,
                ),
                f"{L_tag}_active_rank_count": np.asarray(
                    active_rank_count,
                    dtype=NP_INT_DTYPE,
                ),
            }
        )
        chi_mean_by_layer.append(chi_mean)
        chi_sem_by_layer.append(chi_sem)
        chi_count_by_layer.append(chi_count)

    return (
        npz_arrays,
        chi_mean_by_layer,
        chi_sem_by_layer,
        chi_count_by_layer,
    )


(
    hamiltonian_qfim_sensitivity_random_npz_arrays,
    random_chi_hamiltonian_mean_by_layer,
    random_chi_hamiltonian_sem_by_layer,
    random_chi_hamiltonian_count_by_layer,
) = _build_hamiltonian_sensitivity_layer_statistics(
    qfim_layer_list,
    hamiltonian_qfim_sensitivity_random_by_layer,
    gradient_image_residual_norm_sq_random_by_layer,
    qfim_active_rank_random_by_layer,
)

hamiltonian_qfim_sensitivity_random_result_path = os.path.join(
    qfim_results_dir,
    f"hamiltonian_qfim_normalized_sensitivity_random_points_{keep_key}.npz",
)
save_npz_result(
    hamiltonian_qfim_sensitivity_random_result_path,
    h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
    layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
    keep_wires=np.asarray((0, 1, 2, 3), dtype=NP_INT_DTYPE),
    state_label=np.asarray(keep_label),
    qfim_eigenvalue_threshold=np.asarray(
        QFIM_EFFECTIVE_RANK_THRESHOLD,
        dtype=NP_REAL_DTYPE,
    ),
    num_qfim_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
    qfim_sample_seed_base=np.asarray(
        QFIM_SAMPLE_SEED_BASE,
        dtype=NP_INT_DTYPE,
    ),
    definition=np.asarray(
        "chi_H^(tau) = sum_{lambda_i > tau} "
        "|v_i^dagger g|^2 / lambda_i"
    ),
    gradient_image_residual_definition=np.asarray(
        "sum_{lambda_i <= tau} |v_i^dagger g|^2"
    ),
    chi_hamiltonian_mean_by_layer=np.asarray(
        random_chi_hamiltonian_mean_by_layer,
        dtype=NP_REAL_DTYPE,
    ),
    chi_hamiltonian_sem_by_layer=np.asarray(
        random_chi_hamiltonian_sem_by_layer,
        dtype=NP_REAL_DTYPE,
    ),
    chi_hamiltonian_count_by_layer=np.asarray(
        random_chi_hamiltonian_count_by_layer,
        dtype=NP_INT_DTYPE,
    ),
    **hamiltonian_qfim_sensitivity_random_npz_arrays,
)


(
    hamiltonian_qfim_sensitivity_random_keep01234_npz_arrays,
    random_chi_hamiltonian_mean_keep01234_by_layer,
    random_chi_hamiltonian_sem_keep01234_by_layer,
    random_chi_hamiltonian_count_keep01234_by_layer,
) = _build_hamiltonian_sensitivity_layer_statistics(
    qfim_layer_list,
    hamiltonian_qfim_sensitivity_random_keep01234_by_layer,
    gradient_image_residual_norm_sq_random_keep01234_by_layer,
    qfim_active_rank_random_keep01234_by_layer,
)

hamiltonian_qfim_sensitivity_random_keep01234_result_path = os.path.join(
    qfim_results_dir,
    (
        "hamiltonian_qfim_normalized_sensitivity_random_points_"
        f"{keep_key_5}.npz"
    ),
)
save_npz_result(
    hamiltonian_qfim_sensitivity_random_keep01234_result_path,
    h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
    layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
    keep_wires=np.asarray((0, 1, 2, 3, 4), dtype=NP_INT_DTYPE),
    state_label=np.asarray(keep_label_5),
    qfim_eigenvalue_threshold=np.asarray(
        QFIM_EFFECTIVE_RANK_THRESHOLD,
        dtype=NP_REAL_DTYPE,
    ),
    num_qfim_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
    qfim_sample_seed_base=np.asarray(
        QFIM_SAMPLE_SEED_BASE,
        dtype=NP_INT_DTYPE,
    ),
    definition=np.asarray(
        "chi_H^(tau) = sum_{lambda_i > tau} "
        "|v_i^dagger g|^2 / lambda_i"
    ),
    gradient_image_residual_definition=np.asarray(
        "sum_{lambda_i <= tau} |v_i^dagger g|^2"
    ),
    chi_hamiltonian_mean_by_layer=np.asarray(
        random_chi_hamiltonian_mean_keep01234_by_layer,
        dtype=NP_REAL_DTYPE,
    ),
    chi_hamiltonian_sem_by_layer=np.asarray(
        random_chi_hamiltonian_sem_keep01234_by_layer,
        dtype=NP_REAL_DTYPE,
    ),
    chi_hamiltonian_count_by_layer=np.asarray(
        random_chi_hamiltonian_count_keep01234_by_layer,
        dtype=NP_INT_DTYPE,
    ),
    **hamiltonian_qfim_sensitivity_random_keep01234_npz_arrays,
)


qfim_effective_rank_random_points_result_path = os.path.join(
    qfim_results_dir,
    f"qfim_effective_rank_random_points_{keep_key}.npz",
)

if RUN_QFIM_EFFECTIVE_RANK_RANDOM_POINTS:
    save_npz_result(
        qfim_effective_rank_random_points_result_path,
        h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
        num_qfim_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
        qfim_sample_seed_base=np.asarray(
            QFIM_SAMPLE_SEED_BASE,
            dtype=NP_INT_DTYPE,
        ),
        qfim_effective_rank_threshold=np.asarray(
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        participation_effective_rank_eps=np.asarray(
            PARTICIPATION_EFFECTIVE_RANK_EPS,
            dtype=NP_REAL_DTYPE,
        ),
        layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
        **_layer_arrays_for_npz(
            qfim_rank_reduced_0123_by_layer,
            "threshold_rank",
        ),
        **_layer_arrays_for_npz(
            qfim_participation_rank_random_by_layer,
            "participation_rank",
        ),
        **_layer_arrays_for_npz(
            qfim_eigsum_reduced_0123_by_layer,
            "trace",
        ),
        **_layer_arrays_for_npz(
            qfim_frobenius_norm_sq_random_by_layer,
            "frobenius_norm_sq",
        ),
        **_layer_arrays_for_npz(
            qfim_largest_eigenvalue_random_by_layer,
            "largest_eigenvalue",
        ),
        **_layer_arrays_for_npz(
            qfim_smallest_active_eigenvalue_random_by_layer,
            "smallest_active_eigenvalue",
        ),
        **_layer_arrays_for_npz(
            qfim_active_condition_number_random_by_layer,
            "active_condition_number",
        ),
    )


# ============================================================
# Large-sector gradient weights: compute and save numerical results
# ============================================================
# Large-sector gradient weight along the VQE optimization path
#   color  = layer number
#   marker = QFIM eigenvalue threshold
# ============================================================
GRADIENT_SECTOR_NORM_EPS = QFIM_GRAD_ALIGNMENT_NORM_EPS


def make_large_sector_gradient_weight_fn_for_layer(
    n_layer: int,
    thresholds,
    *,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    qfim_fn = make_reduced0123_qfim_matrix_fn_for_layer_sequential(
        n_layer=n_layer,
        jvp_chunk=jvp_chunk,
    )

    thresholds_jnp = jnp.asarray(thresholds, dtype=REAL_DTYPE)

    @jax.jit
    def large_sector_gradient_weight(theta: jnp.ndarray, grad: jnp.ndarray):
        F = qfim_fn(theta)
        F = 0.5 * (F + F.T)

        evals, evecs = jnp.linalg.eigh(F)
        evals = jnp.clip(evals, a_min=0.0)

        grad = jnp.asarray(grad, dtype=REAL_DTYPE)
        grad_norm_sq = jnp.real(jnp.vdot(grad, grad))

        coeffs = evecs.T @ grad
        coeff_abs2 = jnp.real(coeffs * jnp.conjugate(coeffs))
        valid_gradient = grad_norm_sq > GRADIENT_SECTOR_NORM_EPS
        weights = jnp.where(
            valid_gradient,
            coeff_abs2 / grad_norm_sq,
            jnp.full_like(coeff_abs2, jnp.nan),
        )

        large_masks = evals[None, :] > thresholds_jnp[:, None]
        large_weights = jnp.sum(
            jnp.where(large_masks, weights[None, :], 0.0),
            axis=1,
        )

        return jnp.where(
            valid_gradient,
            jnp.clip(large_weights, a_min=0.0, a_max=1.0),
            jnp.full_like(large_weights, jnp.nan),
        )

    return large_sector_gradient_weight


def compute_large_sector_gradient_weight_by_layer(
    theta_samples_by_layer: dict,
    grad_samples_by_layer: dict,
    layers,
    thresholds,
    *,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    thresholds = tuple(float(thr) for thr in thresholds)
    result = {}

    for L in tqdm(
        layers,
        desc="Large-sector gradient weights",
        unit="layer",
    ):
        if theta_samples_by_layer.get(L) is None:
            continue
        if grad_samples_by_layer.get(L) is None:
            continue

        theta_samples = np.asarray(
            theta_samples_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )
        grad_samples = np.asarray(
            grad_samples_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )

        if theta_samples.shape != grad_samples.shape:
            raise ValueError(
                f"theta_samples and grad_samples must have the same shape for L={L}. "
                f"Got {theta_samples.shape} and {grad_samples.shape}."
            )

        if theta_samples.ndim != 3:
            raise ValueError(
                "theta_samples and grad_samples must have shape "
                "(num_runs, num_sample_iters, num_params)."
            )

        n_runs, n_times, _ = theta_samples.shape

        weight_fn = make_large_sector_gradient_weight_fn_for_layer(
            n_layer=int(L),
            thresholds=thresholds,
            jvp_chunk=jvp_chunk,
        )

        weights_L = {
            thr: np.full((n_runs, n_times), np.nan, dtype=NP_REAL_DTYPE)
            for thr in thresholds
        }

        for run_idx in tqdm(
            range(n_runs),
            desc=f"Gradient-sector runs (L={L})",
            unit="run",
            leave=False,
        ):
            for time_idx in range(n_times):
                weights = weight_fn(
                    jnp.asarray(theta_samples[run_idx, time_idx], dtype=REAL_DTYPE),
                    jnp.asarray(grad_samples[run_idx, time_idx], dtype=REAL_DTYPE),
                )
                weights_np = jax_to_np(weights, dtype=NP_REAL_DTYPE)

                for threshold_idx, threshold in enumerate(thresholds):
                    weights_L[threshold][run_idx, time_idx] = weights_np[threshold_idx]

        result[int(L)] = weights_L

    return result


def compute_qfim_spectral_gradient_history_by_layer(
    theta_samples_by_layer,
    grad_samples_by_layer,
    layers,
    *,
    jvp_chunk=RED_JVP_CHUNK,
    sector_thresholds=THRESHOLDS,
    qfim_matrix_fn_factory=make_reduced0123_qfim_matrix_fn_for_layer_sequential,
    progress_description="QFIM spectral diagnostics along optimization path",
):
    """Compute one QFIM/eigendecomposition per optimization-path point.

    The returned cache is reused by the legacy rank/eigenvalue outputs, the
    large-sector output, the new spectral summary, and all requested alignment
    tables.  QFIM eigenvectors are used only transiently and are not stored.
    """
    scalar_metric_names = (
        "qfim_threshold_rank",
        "qfim_participation_rank",
        "qfim_trace",
        "qfim_frobenius_norm_sq",
        "gradient_norm_sq",
        "hamiltonian_qfim_sensitivity",
        "gradient_image_projection_norm_sq",
        "gradient_image_residual_norm_sq",
        "gradient_image_residual_norm",
        "gradient_image_residual_fraction",
        "gradient_participation_rank",
        "gradient_weighted_qfim_eigenvalue",
        "gradient_weight_sum",
        "largest_qfim_eigenvalue",
        "smallest_active_qfim_eigenvalue",
        "condition_number_active",
        "parseval_relative_error",
    )
    sector_thresholds = tuple(float(value) for value in sector_thresholds)
    diagnostics_cache = {}
    summary_by_layer = {}
    eigs_by_layer = {}
    large_sector_by_layer = {}

    for L in tqdm(
        layers,
        desc=progress_description,
        unit="layer",
    ):
        L = int(L)
        if theta_samples_by_layer.get(L) is None:
            continue
        if grad_samples_by_layer.get(L) is None:
            continue

        theta_samples = np.asarray(
            theta_samples_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )
        grad_samples = np.asarray(
            grad_samples_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )
        if theta_samples.shape != grad_samples.shape:
            raise ValueError(
                f"theta_samples and grad_samples must match for L={L}: "
                f"{theta_samples.shape} != {grad_samples.shape}."
            )
        if theta_samples.ndim != 3:
            raise ValueError(
                "theta_samples and grad_samples must have shape "
                "(num_runs, num_sample_iters, num_params)."
            )

        n_runs_L, n_times_L, n_params_L = theta_samples.shape
        expected_n_params = n_param_per_layer * L
        if n_params_L != expected_n_params:
            raise ValueError(
                f"Expected M=14L={expected_n_params} parameters for L={L}, "
                f"got {n_params_L}."
            )

        # Compiling the complete QFIM callable once per layer avoids dispatching
        # every internal JVP block as a separate eager operation.
        qfim_fn = jax.jit(
            qfim_matrix_fn_factory(
                n_layer=L,
                jvp_chunk=jvp_chunk,
            )
        )
        summary_L = {
            name: np.full(
                (n_runs_L, n_times_L),
                np.nan,
                dtype=NP_REAL_DTYPE,
            )
            for name in scalar_metric_names
        }
        for threshold in sector_thresholds:
            summary_L[
                f"grad_weight_above_{_spectral_threshold_tag(threshold)}"
            ] = np.full(
                (n_runs_L, n_times_L),
                np.nan,
                dtype=NP_REAL_DTYPE,
            )
        eigs_L = np.full(
            (n_runs_L, n_times_L, n_params_L),
            np.nan,
            dtype=NP_REAL_DTYPE,
        )

        for run_idx in tqdm(
            range(n_runs_L),
            desc=f"QFIM spectral runs (L={L})",
            unit="run",
            leave=False,
        ):
            for time_idx in range(n_times_L):
                diagnostics = qfim_spectral_gradient_diagnostics_at_point(
                    theta_samples[run_idx, time_idx],
                    grad_samples[run_idx, time_idx],
                    qfim_fn,
                    sector_thresholds=sector_thresholds,
                )
                diagnostics_cache[(L, run_idx, time_idx)] = diagnostics
                eigs_L[run_idx, time_idx, :] = diagnostics["evals"]
                for metric_name in summary_L:
                    summary_L[metric_name][run_idx, time_idx] = diagnostics[
                        metric_name
                    ]

        summary_by_layer[L] = summary_L
        eigs_by_layer[L] = eigs_L
        large_sector_by_layer[L] = {
            threshold: summary_L[
                f"grad_weight_above_{_spectral_threshold_tag(threshold)}"
            ]
            for threshold in sector_thresholds
        }

    parseval_parts = [
        np.ravel(metrics["parseval_relative_error"])
        for metrics in summary_by_layer.values()
    ]
    finite_parseval_errors = (
        np.concatenate(parseval_parts)
        if parseval_parts
        else np.empty(0, dtype=NP_REAL_DTYPE)
    )
    finite_parseval_errors = finite_parseval_errors[
        np.isfinite(finite_parseval_errors)
    ]
    max_parseval_error = (
        float(np.max(finite_parseval_errors))
        if finite_parseval_errors.size
        else float("nan")
    )
    print(
        "QFIM gradient Parseval check: "
        f"max relative error={max_parseval_error:.3e}"
    )

    return (
        diagnostics_cache,
        summary_by_layer,
        eigs_by_layer,
        large_sector_by_layer,
    )


def validate_qfim_spectral_diagnostics_smoke(
    diagnostics,
    *,
    layer,
    atol=1e-10,
    rtol=1e-8,
):
    """Validate the required one-point numerical invariants."""
    n_params = n_param_per_layer * int(layer)
    eigvals = np.asarray(diagnostics["evals"], dtype=NP_REAL_DTYPE)
    if eigvals.shape != (n_params,):
        raise AssertionError(
            f"Expected {n_params} QFIM eigenvalues, got {eigvals.shape}."
        )
    if np.min(eigvals) < -atol:
        raise AssertionError("Clipped QFIM eigenvalues must be nonnegative.")

    threshold_rank = int(diagnostics["qfim_threshold_rank"])
    if not 0 <= threshold_rank <= n_params:
        raise AssertionError("QFIM threshold rank is outside [0, M].")

    participation_rank = float(diagnostics["qfim_participation_rank"])
    frobenius_norm_sq = float(diagnostics["qfim_frobenius_norm_sq"])
    if frobenius_norm_sq <= PARTICIPATION_EFFECTIVE_RANK_EPS:
        if participation_rank != 0.0:
            raise AssertionError("The zero-spectrum participation rank must be 0.")
    elif not 1.0 - atol <= participation_rank <= n_params + atol:
        raise AssertionError("QFIM participation rank is outside [1, M].")

    if participation_effective_rank_from_eigvals(
        np.zeros(n_params, dtype=NP_REAL_DTYPE)
    ) != 0.0:
        raise AssertionError("Zero-spectrum participation-rank smoke test failed.")

    gradient_norm_sq = float(diagnostics["gradient_norm_sq"])
    chi_hamiltonian = float(diagnostics["hamiltonian_qfim_sensitivity"])
    image_projection_norm_sq = float(
        diagnostics["gradient_image_projection_norm_sq"]
    )
    image_residual_norm_sq = float(
        diagnostics["gradient_image_residual_norm_sq"]
    )
    if not np.isfinite(chi_hamiltonian) or chi_hamiltonian < -atol:
        raise AssertionError(
            "Hamiltonian-direction QFIM sensitivity must be finite and nonnegative."
        )
    if not np.isclose(
        image_projection_norm_sq + image_residual_norm_sq,
        gradient_norm_sq,
        rtol=rtol,
        atol=atol,
    ):
        raise AssertionError(
            "QFIM-image projection and residual fail the gradient-norm "
            "decomposition check."
        )
    weights = np.asarray(diagnostics["weights"], dtype=NP_REAL_DTYPE)
    if gradient_norm_sq > QFIM_GRAD_ALIGNMENT_NORM_EPS:
        if not np.all(np.isfinite(weights)):
            raise AssertionError("Nonzero-gradient weights must be finite.")
        if not np.isclose(np.sum(weights), 1.0, rtol=rtol, atol=atol):
            raise AssertionError("Gradient weights fail the Parseval sum check.")
        gradient_rank = float(diagnostics["gradient_participation_rank"])
        if not 1.0 - atol <= gradient_rank <= n_params + atol:
            raise AssertionError(
                "Gradient participation rank is outside [1, M]."
            )
        weighted_eigenvalue = float(
            diagnostics["gradient_weighted_qfim_eigenvalue"]
        )
        weighted_tolerance = atol + rtol * max(1.0, float(eigvals[0]))
        if not (
            float(eigvals[-1]) - weighted_tolerance
            <= weighted_eigenvalue
            <= float(eigvals[0]) + weighted_tolerance
        ):
            raise AssertionError(
                "Gradient-weighted eigenvalue is outside the QFIM spectrum."
            )
    elif not np.all(np.isnan(weights)):
        raise AssertionError("Near-zero-gradient weights must be NaN.")

    parseval_error = float(diagnostics["parseval_relative_error"])
    if np.isfinite(parseval_error) and parseval_error > 1e-7:
        raise AssertionError(
            f"Parseval relative error is unexpectedly large: {parseval_error}."
        )

    print(
        f"QFIM spectral smoke (L={int(layer)}, run=0, time=0): "
        f"M={n_params}, threshold_rank={threshold_rank}, "
        f"participation_rank={participation_rank:.6g}, "
        f"gradient_weight_sum={float(diagnostics['gradient_weight_sum']):.12g}, "
        f"Parseval relative error={parseval_error:.3e}"
    )


(
    qfim_spectral_gradient_diagnostics_cache,
    qfim_spectral_gradient_summary_by_layer,
    qfim_eigs_history_optimization_path_by_layer,
    qfim_gradient_large_sector_weight_by_layer,
) = compute_qfim_spectral_gradient_history_by_layer(
    theta_sample_traces_by_layer,
    grad_sample_traces_by_layer,
    vqe_layer_list,
    jvp_chunk=RED_JVP_CHUNK,
    sector_thresholds=THRESHOLDS,
)

if qfim_spectral_gradient_diagnostics_cache:
    _smoke_cache_key = (
        (1, 0, 0)
        if (1, 0, 0) in qfim_spectral_gradient_diagnostics_cache
        else next(iter(qfim_spectral_gradient_diagnostics_cache))
    )
    validate_qfim_spectral_diagnostics_smoke(
        qfim_spectral_gradient_diagnostics_cache[_smoke_cache_key],
        layer=_smoke_cache_key[0],
    )

# Repeat the same one-QFIM/one-eigendecomposition analysis for the full kept
# five-qubit state.  The previously sampled VQE parameters and gradients are
# reused, so this adds no optimizer or Hamiltonian-gradient evaluations.
(
    qfim_spectral_gradient_diagnostics_cache_keep01234,
    qfim_spectral_gradient_summary_keep01234_by_layer,
    qfim_eigs_history_optimization_path_keep01234_by_layer,
    qfim_gradient_large_sector_weight_keep01234_by_layer,
) = compute_qfim_spectral_gradient_history_by_layer(
    theta_sample_traces_by_layer,
    grad_sample_traces_by_layer,
    vqe_layer_list,
    jvp_chunk=RED_JVP_CHUNK,
    sector_thresholds=(),
    qfim_matrix_fn_factory=(
        make_reduced01234_qfim_matrix_fn_for_layer_sequential
    ),
    progress_description=(
        "QFIM spectral diagnostics along optimization path; "
        "keep=(0,1,2,3,4)"
    ),
)

if qfim_spectral_gradient_diagnostics_cache_keep01234:
    _smoke_cache_key_keep01234 = (
        (1, 0, 0)
        if (1, 0, 0)
        in qfim_spectral_gradient_diagnostics_cache_keep01234
        else next(iter(qfim_spectral_gradient_diagnostics_cache_keep01234))
    )
    validate_qfim_spectral_diagnostics_smoke(
        qfim_spectral_gradient_diagnostics_cache_keep01234[
            _smoke_cache_key_keep01234
        ],
        layer=_smoke_cache_key_keep01234[0],
    )

np.savez(
    os.path.join(qfim_fig_dir, f"qfim_large_sector_gradient_weight_{keep_key}.npz"),
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    thresholds=np.asarray(THRESHOLDS, dtype=NP_REAL_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    **{
        f"L{int(L)}_thr_{_thr_tag(thr)}": arr
        for L, data_L in qfim_gradient_large_sector_weight_by_layer.items()
        for thr, arr in data_L.items()
    },
)


qfim_large_sector_gradient_weight_result_path = os.path.join(
    qfim_results_dir,
    f"qfim_large_sector_gradient_weight_{keep_key}.npz",
)

save_npz_result(
    qfim_large_sector_gradient_weight_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    thresholds=np.asarray(THRESHOLDS, dtype=NP_REAL_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    **{
        f"L{int(L)}_thr_{_thr_tag(thr)}": arr
        for L, data_L in qfim_gradient_large_sector_weight_by_layer.items()
        for thr, arr in data_L.items()
    },
)


# ============================================================
# Optimization-path QFIM rank/eigenvalues: compute and save numerical results
# ============================================================
# QFIM rank along the VQE optimization path
#   x-axis: sampled optimization iteration
#   y-axis: run-mean QFIM effective rank at theta(iteration)
#   color: layer number
# ============================================================
def make_qfim_rank_and_eigs_fn_for_layer(
    n_layer: int,
    *,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    qfim_fn = make_reduced0123_qfim_matrix_fn_for_layer_sequential(
        n_layer=n_layer,
        jvp_chunk=jvp_chunk,
    )

    @jax.jit
    def qfim_rank_and_eigs(theta: jnp.ndarray):
        F = qfim_fn(theta)
        return psd_rank_and_desc_eigs(F)

    return qfim_rank_and_eigs


def compute_qfim_rank_history_by_layer(
    theta_samples_by_layer: dict,
    layers,
    *,
    jvp_chunk: int = RED_JVP_CHUNK,
    return_eigs: bool = False,
):
    rank_history_by_layer = {}
    eigs_history_by_layer = {}

    for L in tqdm(
        layers,
        desc="QFIM rank history along optimization path",
        unit="layer",
    ):
        if theta_samples_by_layer.get(L) is None:
            continue

        theta_samples = np.asarray(
            theta_samples_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )

        if theta_samples.ndim != 3:
            raise ValueError(
                "theta_samples must have shape "
                "(num_runs, num_sample_iters, num_params)."
            )

        n_runs, n_times, n_params = theta_samples.shape
        rank_eigs_fn = make_qfim_rank_and_eigs_fn_for_layer(
            n_layer=int(L),
            jvp_chunk=jvp_chunk,
        )

        ranks_L = np.full((n_runs, n_times), np.nan, dtype=NP_REAL_DTYPE)
        eigs_L = None
        if return_eigs:
            eigs_L = np.full(
                (n_runs, n_times, n_params),
                np.nan,
                dtype=NP_REAL_DTYPE,
            )

        for run_idx in tqdm(
            range(n_runs),
            desc=f"QFIM-rank runs (L={L})",
            unit="run",
            leave=False,
        ):
            for time_idx in range(n_times):
                rank_value, eigs_desc = rank_eigs_fn(
                    jnp.asarray(theta_samples[run_idx, time_idx], dtype=REAL_DTYPE)
                )
                ranks_L[run_idx, time_idx] = NP_REAL_DTYPE(
                    jax.device_get(rank_value)
                )
                if return_eigs:
                    eigs_L[run_idx, time_idx, :] = jax_to_np(
                        eigs_desc,
                        dtype=NP_REAL_DTYPE,
                    )

        rank_history_by_layer[int(L)] = ranks_L
        if return_eigs:
            eigs_history_by_layer[int(L)] = eigs_L

    if return_eigs:
        return rank_history_by_layer, eigs_history_by_layer

    return rank_history_by_layer


# Reuse the consolidated per-point diagnostics above.  The compatibility
# ``compute_qfim_rank_history_by_layer`` function remains available for
# callers, but the main program deliberately does not perform a second pass.
qfim_rank_history_by_layer = {
    int(L): np.asarray(
        metrics["qfim_threshold_rank"],
        dtype=NP_REAL_DTYPE,
    )
    for L, metrics in qfim_spectral_gradient_summary_by_layer.items()
}

np.savez(
    os.path.join(qfim_fig_dir, f"qfim_rank_history_optimization_path_{keep_key}.npz"),
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    **{
        f"L{int(L)}": arr
        for L, arr in qfim_rank_history_by_layer.items()
    },
)

qfim_rank_history_result_path = os.path.join(
    qfim_results_dir,
    f"qfim_rank_history_optimization_path_{keep_key}.npz",
)

save_npz_result(
    qfim_rank_history_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    **{
        f"L{int(L)}": arr
        for L, arr in qfim_rank_history_by_layer.items()
    },
)


qfim_eigs_history_result_path = os.path.join(
    qfim_results_dir,
    f"qfim_eigs_history_optimization_path_{keep_key}.npz",
)

save_npz_result(
    qfim_eigs_history_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    **{
        f"L{int(L)}": arr
        for L, arr in qfim_eigs_history_optimization_path_by_layer.items()
    },
)

qfim_trace_history_optimization_path_by_layer = {
    int(L): np.sum(np.asarray(arr, dtype=NP_REAL_DTYPE), axis=2)
    for L, arr in qfim_eigs_history_optimization_path_by_layer.items()
}

qfim_trace_history_result_path = os.path.join(
    qfim_results_dir,
    f"qfim_trace_history_optimization_path_{keep_key}.npz",
)

save_npz_result(
    qfim_trace_history_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    **{
        f"L{int(L)}": arr
        for L, arr in qfim_trace_history_optimization_path_by_layer.items()
    },
)


# ============================================================
# Trace-based / gradient-participation summary (new schema)
# ============================================================
qfim_spectral_summary_npz_arrays = {}
for L, metrics in qfim_spectral_gradient_summary_by_layer.items():
    L_tag = f"L{int(L)}"
    qfim_spectral_summary_npz_arrays.update(
        {
            f"{L_tag}_threshold_rank": np.asarray(
                metrics["qfim_threshold_rank"],
                dtype=NP_INT_DTYPE,
            ),
            f"{L_tag}_qfim_threshold_rank": np.asarray(
                metrics["qfim_threshold_rank"],
                dtype=NP_INT_DTYPE,
            ),
            f"{L_tag}_participation_rank": np.asarray(
                metrics["qfim_participation_rank"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_qfim_participation_rank": np.asarray(
                metrics["qfim_participation_rank"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_trace": np.asarray(
                metrics["qfim_trace"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_qfim_trace": np.asarray(
                metrics["qfim_trace"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_frobenius_norm_sq": np.asarray(
                metrics["qfim_frobenius_norm_sq"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_qfim_frobenius_norm_sq": np.asarray(
                metrics["qfim_frobenius_norm_sq"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_gradient_norm_sq": np.asarray(
                metrics["gradient_norm_sq"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_hamiltonian_qfim_sensitivity": np.asarray(
                metrics["hamiltonian_qfim_sensitivity"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_chi_hamiltonian": np.asarray(
                metrics["hamiltonian_qfim_sensitivity"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_gradient_image_projection_norm_sq": np.asarray(
                metrics["gradient_image_projection_norm_sq"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_gradient_image_residual_norm_sq": np.asarray(
                metrics["gradient_image_residual_norm_sq"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_gradient_image_residual_norm": np.asarray(
                metrics["gradient_image_residual_norm"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_gradient_image_residual_fraction": np.asarray(
                metrics["gradient_image_residual_fraction"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_gradient_participation_rank": np.asarray(
                metrics["gradient_participation_rank"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_gradient_weighted_qfim_eigenvalue": np.asarray(
                metrics["gradient_weighted_qfim_eigenvalue"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_gradient_weight_sum": np.asarray(
                metrics["gradient_weight_sum"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_largest_eigenvalue": np.asarray(
                metrics["largest_qfim_eigenvalue"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_largest_qfim_eigenvalue": np.asarray(
                metrics["largest_qfim_eigenvalue"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_smallest_active_eigenvalue": np.asarray(
                metrics["smallest_active_qfim_eigenvalue"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_smallest_active_qfim_eigenvalue": np.asarray(
                metrics["smallest_active_qfim_eigenvalue"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_active_condition_number": np.asarray(
                metrics["condition_number_active"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_condition_number_active": np.asarray(
                metrics["condition_number_active"],
                dtype=NP_REAL_DTYPE,
            ),
            f"{L_tag}_parseval_relative_error": np.asarray(
                metrics["parseval_relative_error"],
                dtype=NP_REAL_DTYPE,
            ),
        }
    )
    for metric_name, values in metrics.items():
        if metric_name.startswith("grad_weight_above_"):
            qfim_spectral_summary_npz_arrays[
                f"{L_tag}_{metric_name}"
            ] = np.asarray(values, dtype=NP_REAL_DTYPE)


# ============================================================
# Hamiltonian-direction QFIM-normalized sensitivity
# ============================================================
# This is evaluated along the already-computed VQE optimization path because
# those points provide the Hamiltonian gradient g without any extra gradient
# or QFIM evaluations.  The final sampled iteration is the default layer-axis
# summary used by the visualization script, while the full run/time arrays are
# retained for reproducibility and alternative plots.
hamiltonian_sensitivity_layers = np.asarray(
    sorted(qfim_spectral_gradient_summary_by_layer),
    dtype=NP_INT_DTYPE,
)
hamiltonian_sensitivity_npz_arrays = {}
final_chi_hamiltonian_mean = []
final_chi_hamiltonian_sem = []
final_chi_hamiltonian_count = []

for L in hamiltonian_sensitivity_layers:
    L = int(L)
    metrics = qfim_spectral_gradient_summary_by_layer[L]
    chi_hamiltonian = np.asarray(
        metrics["hamiltonian_qfim_sensitivity"],
        dtype=NP_REAL_DTYPE,
    )
    residual_norm_sq = np.asarray(
        metrics["gradient_image_residual_norm_sq"],
        dtype=NP_REAL_DTYPE,
    )
    active_rank = np.asarray(
        metrics["qfim_threshold_rank"],
        dtype=NP_INT_DTYPE,
    )
    chi_mean, chi_sem, chi_count = _finite_mean_sem(
        chi_hamiltonian,
        axis=0,
    )
    residual_mean, residual_sem, residual_count = _finite_mean_sem(
        residual_norm_sq,
        axis=0,
    )
    active_rank_mean, active_rank_sem, active_rank_count = _finite_mean_sem(
        active_rank,
        axis=0,
    )

    L_tag = f"L{L}"
    hamiltonian_sensitivity_npz_arrays.update(
        {
            f"{L_tag}_chi_hamiltonian": chi_hamiltonian,
            f"{L_tag}_chi_hamiltonian_mean": chi_mean,
            f"{L_tag}_chi_hamiltonian_sem": chi_sem,
            f"{L_tag}_chi_hamiltonian_count": chi_count,
            f"{L_tag}_gradient_image_residual_norm_sq": residual_norm_sq,
            f"{L_tag}_gradient_image_residual_norm_sq_mean": residual_mean,
            f"{L_tag}_gradient_image_residual_norm_sq_sem": residual_sem,
            f"{L_tag}_gradient_image_residual_norm_sq_count": residual_count,
            f"{L_tag}_active_rank": active_rank,
            f"{L_tag}_active_rank_mean": active_rank_mean,
            f"{L_tag}_active_rank_sem": active_rank_sem,
            f"{L_tag}_active_rank_count": active_rank_count,
        }
    )
    final_chi_hamiltonian_mean.append(chi_mean[-1])
    final_chi_hamiltonian_sem.append(chi_sem[-1])
    final_chi_hamiltonian_count.append(chi_count[-1])

hamiltonian_qfim_sensitivity_result_path = os.path.join(
    qfim_results_dir,
    (
        "hamiltonian_qfim_normalized_sensitivity_optimization_path_"
        f"{keep_key}.npz"
    ),
)
save_npz_result(
    hamiltonian_qfim_sensitivity_result_path,
    h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
    layers=hamiltonian_sensitivity_layers,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    num_runs=np.asarray(num_runs, dtype=NP_INT_DTYPE),
    keep_wires=np.asarray((0, 1, 2, 3), dtype=NP_INT_DTYPE),
    state_label=np.asarray(keep_label),
    qfim_eigenvalue_threshold=np.asarray(
        QFIM_EFFECTIVE_RANK_THRESHOLD,
        dtype=NP_REAL_DTYPE,
    ),
    definition=np.asarray(
        "chi_H^(tau) = sum_{lambda_i > tau} "
        "|v_i^dagger g|^2 / lambda_i"
    ),
    gradient_image_residual_definition=np.asarray(
        "sum_{lambda_i <= tau} |v_i^dagger g|^2"
    ),
    final_sample_iter=np.asarray(sample_iters[-1], dtype=NP_INT_DTYPE),
    final_chi_hamiltonian_mean=np.asarray(
        final_chi_hamiltonian_mean,
        dtype=NP_REAL_DTYPE,
    ),
    final_chi_hamiltonian_sem=np.asarray(
        final_chi_hamiltonian_sem,
        dtype=NP_REAL_DTYPE,
    ),
    final_chi_hamiltonian_count=np.asarray(
        final_chi_hamiltonian_count,
        dtype=NP_INT_DTYPE,
    ),
    **hamiltonian_sensitivity_npz_arrays,
)


hamiltonian_sensitivity_layers_keep01234 = np.asarray(
    sorted(qfim_spectral_gradient_summary_keep01234_by_layer),
    dtype=NP_INT_DTYPE,
)
(
    hamiltonian_sensitivity_keep01234_npz_arrays,
    chi_hamiltonian_mean_keep01234_by_layer,
    chi_hamiltonian_sem_keep01234_by_layer,
    chi_hamiltonian_count_keep01234_by_layer,
) = _build_hamiltonian_sensitivity_layer_statistics(
    hamiltonian_sensitivity_layers_keep01234,
    {
        int(L): metrics["hamiltonian_qfim_sensitivity"]
        for L, metrics
        in qfim_spectral_gradient_summary_keep01234_by_layer.items()
    },
    {
        int(L): metrics["gradient_image_residual_norm_sq"]
        for L, metrics
        in qfim_spectral_gradient_summary_keep01234_by_layer.items()
    },
    {
        int(L): metrics["qfim_threshold_rank"]
        for L, metrics
        in qfim_spectral_gradient_summary_keep01234_by_layer.items()
    },
)
final_chi_hamiltonian_mean_keep01234 = [
    np.asarray(values, dtype=NP_REAL_DTYPE)[-1]
    for values in chi_hamiltonian_mean_keep01234_by_layer
]
final_chi_hamiltonian_sem_keep01234 = [
    np.asarray(values, dtype=NP_REAL_DTYPE)[-1]
    for values in chi_hamiltonian_sem_keep01234_by_layer
]
final_chi_hamiltonian_count_keep01234 = [
    np.asarray(values, dtype=NP_INT_DTYPE)[-1]
    for values in chi_hamiltonian_count_keep01234_by_layer
]

hamiltonian_qfim_sensitivity_keep01234_result_path = os.path.join(
    qfim_results_dir,
    (
        "hamiltonian_qfim_normalized_sensitivity_optimization_path_"
        f"{keep_key_5}.npz"
    ),
)
save_npz_result(
    hamiltonian_qfim_sensitivity_keep01234_result_path,
    h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
    layers=hamiltonian_sensitivity_layers_keep01234,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    num_runs=np.asarray(num_runs, dtype=NP_INT_DTYPE),
    keep_wires=np.asarray((0, 1, 2, 3, 4), dtype=NP_INT_DTYPE),
    state_label=np.asarray(keep_label_5),
    qfim_eigenvalue_threshold=np.asarray(
        QFIM_EFFECTIVE_RANK_THRESHOLD,
        dtype=NP_REAL_DTYPE,
    ),
    definition=np.asarray(
        "chi_H^(tau) = sum_{lambda_i > tau} "
        "|v_i^dagger g|^2 / lambda_i"
    ),
    gradient_image_residual_definition=np.asarray(
        "sum_{lambda_i <= tau} |v_i^dagger g|^2"
    ),
    final_sample_iter=np.asarray(sample_iters[-1], dtype=NP_INT_DTYPE),
    final_chi_hamiltonian_mean=np.asarray(
        final_chi_hamiltonian_mean_keep01234,
        dtype=NP_REAL_DTYPE,
    ),
    final_chi_hamiltonian_sem=np.asarray(
        final_chi_hamiltonian_sem_keep01234,
        dtype=NP_REAL_DTYPE,
    ),
    final_chi_hamiltonian_count=np.asarray(
        final_chi_hamiltonian_count_keep01234,
        dtype=NP_INT_DTYPE,
    ),
    **hamiltonian_sensitivity_keep01234_npz_arrays,
)


qfim_effective_rank_optimization_path_result_path = os.path.join(
    qfim_results_dir,
    f"qfim_effective_rank_optimization_path_{keep_key}.npz",
)
if RUN_QFIM_EFFECTIVE_RANK_OPTIMIZATION_PATH:
    save_npz_result(
        qfim_effective_rank_optimization_path_result_path,
        h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
        layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
        sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        qfim_effective_rank_threshold=np.asarray(
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        participation_effective_rank_eps=np.asarray(
            PARTICIPATION_EFFECTIVE_RANK_EPS,
            dtype=NP_REAL_DTYPE,
        ),
        **{
            key: value
            for key, value in qfim_spectral_summary_npz_arrays.items()
            if (
                key.endswith("_threshold_rank")
                or key.endswith("_participation_rank")
                or key.endswith("_trace")
                or key.endswith("_frobenius_norm_sq")
            )
        },
    )


qfim_spectral_gradient_summary_result_path = os.path.join(
    qfim_results_dir,
    f"qfim_spectral_gradient_summary_optimization_path_{keep_key}.npz",
)
if RUN_QFIM_SPECTRAL_GRADIENT_SUMMARY:
    save_npz_result(
        qfim_spectral_gradient_summary_result_path,
        h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
        layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
        sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        qfim_effective_rank_threshold=np.asarray(
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        participation_effective_rank_eps=np.asarray(
            PARTICIPATION_EFFECTIVE_RANK_EPS,
            dtype=NP_REAL_DTYPE,
        ),
        qfim_grad_alignment_norm_eps=np.asarray(
            QFIM_GRAD_ALIGNMENT_NORM_EPS,
            dtype=NP_REAL_DTYPE,
        ),
        gradient_norm_eps=np.asarray(
            QFIM_GRAD_ALIGNMENT_NORM_EPS,
            dtype=NP_REAL_DTYPE,
        ),
        grad_sector_thresholds=np.asarray(
            THRESHOLDS,
            dtype=NP_REAL_DTYPE,
        ),
        gradient_sector_thresholds=np.asarray(
            THRESHOLDS,
            dtype=NP_REAL_DTYPE,
        ),
        threshold_rank_definition=np.asarray(
            "number of QFIM eigenvalues strictly above "
            "qfim_effective_rank_threshold"
        ),
        participation_rank_definition=np.asarray(
            "(sum(lambda))^2 / sum(lambda^2), with zero-matrix value 0"
        ),
        gradient_weight_definition=np.asarray(
            "|v_i^dagger g|^2 / ||g||_2^2; NaN when gradient norm is small"
        ),
        **qfim_spectral_summary_npz_arrays,
    )

# ============================================================
# QFIM-gradient alignment: compute and save numerical results
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
QFIM_GRAD_ALIGN_NORM_EPS = QFIM_GRAD_ALIGNMENT_NORM_EPS

qfim_grad_align_dir = os.path.join(qfim_fig_dir, "qfim_grad_alignment")
qfim_grad_align_results_dir = os.path.join(qfim_results_dir, "qfim_grad_alignment")
os.makedirs(qfim_grad_align_dir, exist_ok=True)
os.makedirs(qfim_grad_align_results_dir, exist_ok=True)


def _normalize_index_list(indices, n):
    if indices is None:
        return list(range(n))

    normalized = []

    for idx in indices:
        idx = int(idx)

        if idx < 0:
            idx = n + idx

        normalized.append(idx)

    if any((idx < 0 or idx >= n) for idx in normalized):
        raise IndexError("Index out of range.")

    return normalized


def qfim_grad_alignment_at_point(
    theta,
    grad,
    qfim_fn,
    *,
    sort_desc=True,
    norm_eps=QFIM_GRAD_ALIGN_NORM_EPS,
):
    """Compute alignment data from one QFIM evaluation.

    Individual gradient weights can be basis-dependent inside a degenerate
    QFIM eigenspace. Aggregated weight over the full degenerate subspace is
    basis-independent.
    """
    diagnostics = qfim_spectral_gradient_diagnostics_at_point(
        theta,
        grad,
        qfim_fn,
        grad_norm_eps=norm_eps,
    )

    if not sort_desc:
        reverse = slice(None, None, -1)
        for key in (
            "evals",
            "weights",
            "coeffs",
            "coeff_abs2",
            "lambda_fraction",
            "active_by_rank_threshold",
        ):
            diagnostics[key] = diagnostics[key][reverse]
        diagnostics["cumulative_lambda_fraction"] = np.cumsum(
            diagnostics["lambda_fraction"]
        )
        diagnostics["cumulative_gradient_weight"] = np.cumsum(
            diagnostics["weights"]
        )
        diagnostics["eig_index"] = np.arange(
            1,
            diagnostics["evals"].size + 1,
            dtype=NP_INT_DTYPE,
        )

    # Legacy name retained; unlike the old implementation, normalization uses
    # the directly computed ||g||^2 as required by the definition.
    diagnostics["grad_weight_denominator"] = diagnostics["gradient_norm_sq"]
    return diagnostics


def qfim_grad_alignment_one_to_table(
    alignment,
    *,
    layer=None,
    run=None,
    time_index=None,
    iteration=None,
):
    """
    Convert one-point alignment result into a table-like dictionary.
    """
    n = alignment["evals"].size

    layer_value = -1 if layer is None else int(layer)
    run_value = -1 if run is None else int(run)
    time_value = -1 if time_index is None else int(time_index)
    iter_value = -1 if iteration is None else int(iteration)

    table = {
        "lambda": np.asarray(
            alignment["evals"],
            dtype=NP_REAL_DTYPE,
        ),
        "w_grad": np.asarray(
            alignment["weights"],
            dtype=NP_REAL_DTYPE,
        ),
        "coeff_abs2": np.asarray(
            alignment["coeff_abs2"],
            dtype=NP_REAL_DTYPE,
        ),
        "coeff": np.asarray(
            alignment["coeffs"],
            dtype=NP_COMPLEX_DTYPE,
        ),
        "lambda_fraction": np.asarray(
            alignment["lambda_fraction"],
            dtype=NP_REAL_DTYPE,
        ),
        "cumulative_lambda_fraction": np.asarray(
            alignment["cumulative_lambda_fraction"],
            dtype=NP_REAL_DTYPE,
        ),
        "cumulative_gradient_weight": np.asarray(
            alignment["cumulative_gradient_weight"],
            dtype=NP_REAL_DTYPE,
        ),
        "active_by_rank_threshold": np.asarray(
            alignment["active_by_rank_threshold"],
            dtype=np.bool_,
        ),
        "eig_index": np.asarray(
            alignment["eig_index"],
            dtype=NP_INT_DTYPE,
        ),
        "layer": np.full(n, layer_value, dtype=NP_INT_DTYPE),
        "run": np.full(n, run_value, dtype=NP_INT_DTYPE),
        "time_index": np.full(n, time_value, dtype=NP_INT_DTYPE),
        "iteration": np.full(n, iter_value, dtype=NP_INT_DTYPE),
    }
    for threshold in THRESHOLDS:
        metric_name = f"grad_weight_above_{_spectral_threshold_tag(threshold)}"
        table[metric_name] = np.full(
            n,
            alignment.get(metric_name, np.nan),
            dtype=NP_REAL_DTYPE,
        )

    return table


def compute_qfim_grad_alignment_table_for_layer(
    L,
    theta_samples_by_layer,
    grad_samples_by_layer,
    *,
    run_indices=None,
    time_indices=None,
    sample_iters=None,
    jvp_chunk=RED_JVP_CHUNK,
    sort_desc=True,
    diagnostics_cache=None,
):
    """
    Compute QFIM-gradient alignment scatter data for one layer L.

    Expected shapes:
        theta_samples_by_layer[L]: (num_runs, num_sample_times, num_params)
        grad_samples_by_layer[L]:  (num_runs, num_sample_times, num_params)

    A 2D shape (num_samples, num_params) is also accepted and treated as
    one sampled time point.
    """
    theta_samples = np.asarray(
        theta_samples_by_layer[L],
        dtype=NP_REAL_DTYPE,
    )

    grad_samples = np.asarray(
        grad_samples_by_layer[L],
        dtype=NP_REAL_DTYPE,
    )

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

    n_runs, n_times, _ = theta_samples.shape

    run_ids = _normalize_index_list(run_indices, n_runs)
    time_ids = _normalize_index_list(time_indices, n_times)

    if sample_iters is None:
        sample_iters_arr = np.arange(n_times, dtype=NP_INT_DTYPE)
    else:
        sample_iters_arr = np.asarray(sample_iters, dtype=NP_INT_DTYPE)

    qfim_fn = None
    if diagnostics_cache is None:
        qfim_fn = jax.jit(
            make_reduced0123_qfim_matrix_fn_for_layer_sequential(
                n_layer=int(L),
                jvp_chunk=jvp_chunk,
            )
        )

    rows = {
        "lambda": [],
        "w_grad": [],
        "coeff_abs2": [],
        "coeff": [],
        "lambda_fraction": [],
        "cumulative_lambda_fraction": [],
        "cumulative_gradient_weight": [],
        "active_by_rank_threshold": [],
        "eig_index": [],
        "layer": [],
        "run": [],
        "time_index": [],
        "iteration": [],
    }
    for threshold in THRESHOLDS:
        rows[
            f"grad_weight_above_{_spectral_threshold_tag(threshold)}"
        ] = []

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

            cache_key = (int(L), int(run_idx), int(time_idx))
            if diagnostics_cache is not None:
                if not sort_desc:
                    raise ValueError(
                        "Cached alignment diagnostics are stored in descending "
                        "eigenvalue order."
                    )
                if cache_key not in diagnostics_cache:
                    raise KeyError(
                        f"No cached QFIM diagnostics for key {cache_key}."
                    )
                alignment = diagnostics_cache[cache_key]
            else:
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
        if key == "coeff":
            table[key] = np.concatenate(values).astype(NP_COMPLEX_DTYPE)
        elif key == "active_by_rank_threshold":
            table[key] = np.concatenate(values).astype(np.bool_)
        elif key in (
            "lambda",
            "w_grad",
            "coeff_abs2",
            "lambda_fraction",
            "cumulative_lambda_fraction",
            "cumulative_gradient_weight",
        ) or key.startswith("grad_weight_above_"):
            table[key] = np.concatenate(values).astype(NP_REAL_DTYPE)
        else:
            table[key] = np.concatenate(values).astype(NP_INT_DTYPE)

    return table


def _time_index_from_iteration(sample_iters_for_labels, target_iteration: int):
    """
    Return the sampled-time index corresponding to a requested optimization
    iteration.
    """
    sample_iters_arr = np.asarray(sample_iters_for_labels, dtype=NP_INT_DTYPE)
    target_iteration = int(target_iteration)

    hit = np.where(sample_iters_arr == target_iteration)[0]

    if hit.size == 0:
        raise ValueError(
            f"iteration {target_iteration} is not included in sample_iters. "
            "Add it to sample_iters before running the VQE optimization loop."
        )

    return int(hit[0])


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
    diagnostics_cache=None,
):
    """
    Generate qfim_grad_weight_scatter plots layer by layer.

    For each available layer L, this creates

        qfim_grad_alignment/L{L}/

    and saves one scatter plot for each requested optimization iteration.

    The scatter plot uses
        x = QFIM eigenvalue lambda_i,
        y = gradient weight w_i^grad,
    evaluated at the parameters sampled during optimization at that iteration.
    """
    if layers is None:
        candidate_layers = vqe_layer_list
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

    table_by_layer_iteration = {}

    for L in tqdm(
        available_layers,
        desc="QFIM eigenvalue-gradient scatter by layer/iteration",
        unit="layer",
    ):
        layer_dir = os.path.join(qfim_grad_align_dir, f"L{L}")
        os.makedirs(layer_dir, exist_ok=True)

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
                sample_iters=sample_iters_for_labels,
                jvp_chunk=jvp_chunk,
                sort_desc=True,
                diagnostics_cache=diagnostics_cache,
            )

            table_by_layer_iteration[L][iteration] = table_L_iter

            iter_tag = f"iter{iteration:06d}"
            result_npz_path = os.path.join(
                qfim_grad_align_results_dir,
                f"L{L}",
                f"qfim_grad_alignment_scatter_data_L{L}_{iter_tag}.npz",
            )

            if save_npz:
                np.savez(
                    os.path.join(
                        layer_dir,
                        f"qfim_grad_alignment_scatter_data_L{L}_{iter_tag}.npz",
                    ),
                    **table_L_iter,
                )
                save_npz_result(
                    result_npz_path,
                    **table_L_iter,
                )

            if make_plots:
                table_for_plot = (
                    load_npz_result(result_npz_path)
                    if save_npz
                    else table_L_iter
                )
                plot_qfim_grad_alignment_table(
                    table_for_plot,
                    title=(
                        rf"QFIM eigenvalue vs gradient weight, "
                        rf"L={L}, iteration {iteration}"
                    ),
                    outpath=os.path.join(
                        layer_dir,
                        f"qfim_grad_weight_scatter_L{L}_{iter_tag}.pdf",
                    ),
                    log_x=log_x,
                    log_y=log_y,
                    color_by=None,
                    point_size=14.0,
                    alpha=0.45,
                )

    return table_by_layer_iteration


def run_qfim_grad_alignment_by_layer(
    *,
    layers=None,
    use_all_sampled_times=False,
    run_indices=None,
    sample_iters_for_labels=None,
    jvp_chunk=RED_JVP_CHUNK,
    log_x=True,
    log_y=False,
    save_npz=True,
    make_per_layer_plots=True,
    make_overlay_plot=True,
    diagnostics_cache=None,
):
    """
    Run QFIM eigenvalue-gradient alignment analysis layer by layer.

    Parameters
    ----------
    layers : iterable or None
        Layers to analyze. If None, use vqe_layer_list.
    use_all_sampled_times : bool
        False: use final sampled iteration only.
        True: use all sampled iterations.
    run_indices : iterable or None
        Runs to include. If None, use all runs.
    sample_iters_for_labels : array or None
        Iteration labels. Usually pass sample_iters.
    jvp_chunk : int
        JVP chunk size for QFIM construction.
    log_x : bool
        Use log scale on QFIM eigenvalue axis.
    log_y : bool
        Use log scale on gradient-weight axis.
    save_npz : bool
        Save table data for each layer.
    make_per_layer_plots : bool
        Save one scatter plot per layer.
    make_overlay_plot : bool
        Save one combined scatter plot across layers.

    Returns
    -------
    dict
        table_by_layer[L] = table dictionary.
    """
    if layers is None:
        candidate_layers = vqe_layer_list
    else:
        candidate_layers = layers

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

    table_by_layer = {}

    for L in tqdm(
        available_layers,
        desc="QFIM eigenvalue-gradient scatter by layer",
        unit="layer",
    ):
        theta_samples_L = np.asarray(theta_sample_traces_by_layer[L])
        n_times_L = theta_samples_L.shape[1] if theta_samples_L.ndim == 3 else 1

        if use_all_sampled_times:
            time_indices = range(n_times_L)
            time_tag = "all_times"
            title_time = "all sampled iterations"
            color_by = "iteration"
            point_size = 12.0
            scatter_alpha = 0.40
        else:
            time_indices = [-1]
            time_tag = "final_iter"
            color_by = None
            point_size = 14.0
            scatter_alpha = 0.45

            if sample_iters_for_labels is not None:
                title_time = f"final iteration {int(np.asarray(sample_iters_for_labels)[-1])}"
            else:
                title_time = "final iteration"

        table_L = compute_qfim_grad_alignment_table_for_layer(
            L,
            theta_sample_traces_by_layer,
            grad_sample_traces_by_layer,
            run_indices=run_indices,
            time_indices=time_indices,
            sample_iters=sample_iters_for_labels,
            jvp_chunk=jvp_chunk,
            sort_desc=True,
            diagnostics_cache=diagnostics_cache,
        )

        result_npz_path = os.path.join(
            qfim_grad_align_results_dir,
            f"qfim_grad_alignment_scatter_data_L{L}_{time_tag}.npz",
        )

        if save_npz:
            np.savez(
                os.path.join(
                    qfim_grad_align_dir,
                    f"qfim_grad_alignment_scatter_data_L{L}_{time_tag}.npz",
                ),
                **table_L,
            )
            save_npz_result(
                result_npz_path,
                **table_L,
            )

        table_for_plot = (
            load_npz_result(result_npz_path)
            if save_npz
            else table_L
        )
        table_by_layer[L] = table_for_plot

        if make_per_layer_plots:
            plot_qfim_grad_alignment_table(
                table_for_plot,
                title=(
                    rf"QFIM eigenvalue vs gradient weight, "
                    rf"L={L}, {title_time}"
                ),
                outpath=os.path.join(
                    qfim_grad_align_dir,
                    f"qfim_grad_weight_scatter_L{L}_{time_tag}.pdf",
                ),
                log_x=log_x,
                log_y=log_y,
                color_by=color_by,
                point_size=point_size,
                alpha=scatter_alpha,
            )

    if make_overlay_plot:
        overlay_tag = "all_times" if use_all_sampled_times else "final_iter"

        plot_qfim_grad_alignment_layer_overlay(
            table_by_layer,
            available_layers,
            title=(
                rf"QFIM eigenvalue vs gradient weight "
                rf"across layers, {overlay_tag.replace('_', ' ')}"
            ),
            outpath=os.path.join(
                qfim_grad_align_dir,
                f"qfim_grad_weight_scatter_overlay_layers_{overlay_tag}.pdf",
            ),
            log_x=log_x,
            log_y=log_y,
            point_size=12.0,
            alpha=0.40,
        )

    return table_by_layer


# ------------------------------------------------------------

# ------------------------------------------------------------
# Execution settings for numerical alignment data
# ------------------------------------------------------------
RUN_QFIM_GRAD_ALIGNMENT_FINAL_ITER = cfg.RUN_QFIM_GRAD_ALIGNMENT_FINAL_ITER
RUN_QFIM_GRAD_ALIGNMENT_ALL_TIMES = cfg.RUN_QFIM_GRAD_ALIGNMENT_ALL_TIMES
RUN_QFIM_GRAD_ALIGNMENT_PER_ITERATION = cfg.RUN_QFIM_GRAD_ALIGNMENT_PER_ITERATION

LOG_X_QFIM_GRAD_ALIGNMENT = cfg.LOG_X_QFIM_GRAD_ALIGNMENT
LOG_Y_QFIM_GRAD_ALIGNMENT = cfg.LOG_Y_QFIM_GRAD_ALIGNMENT
QFIM_GRAD_ALIGNMENT_RUN_INDICES = cfg.QFIM_GRAD_ALIGNMENT_RUN_INDICES
if cfg.QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS is None:
    QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS = tuple(int(t) for t in sample_iters)
else:
    QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS = tuple(
        int(t) for t in cfg.QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS
    )

if RUN_QFIM_GRAD_ALIGNMENT_FINAL_ITER:
    run_qfim_grad_alignment_by_layer(
        layers=vqe_layer_list,
        use_all_sampled_times=False,
        run_indices=QFIM_GRAD_ALIGNMENT_RUN_INDICES,
        sample_iters_for_labels=sample_iters,
        jvp_chunk=RED_JVP_CHUNK,
        log_x=LOG_X_QFIM_GRAD_ALIGNMENT,
        log_y=LOG_Y_QFIM_GRAD_ALIGNMENT,
        save_npz=True,
        make_per_layer_plots=False,
        make_overlay_plot=False,
        diagnostics_cache=qfim_spectral_gradient_diagnostics_cache,
    )

if RUN_QFIM_GRAD_ALIGNMENT_ALL_TIMES:
    run_qfim_grad_alignment_by_layer(
        layers=vqe_layer_list,
        use_all_sampled_times=True,
        run_indices=QFIM_GRAD_ALIGNMENT_RUN_INDICES,
        sample_iters_for_labels=sample_iters,
        jvp_chunk=RED_JVP_CHUNK,
        log_x=LOG_X_QFIM_GRAD_ALIGNMENT,
        log_y=LOG_Y_QFIM_GRAD_ALIGNMENT,
        save_npz=True,
        make_per_layer_plots=False,
        make_overlay_plot=False,
        diagnostics_cache=qfim_spectral_gradient_diagnostics_cache,
    )

if RUN_QFIM_GRAD_ALIGNMENT_PER_ITERATION:
    run_qfim_grad_alignment_by_layer_iteration_folders(
        layers=vqe_layer_list,
        target_iterations=QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS,
        run_indices=QFIM_GRAD_ALIGNMENT_RUN_INDICES,
        sample_iters_for_labels=sample_iters,
        jvp_chunk=RED_JVP_CHUNK,
        log_x=LOG_X_QFIM_GRAD_ALIGNMENT,
        log_y=LOG_Y_QFIM_GRAD_ALIGNMENT,
        save_npz=True,
        make_plots=False,
        diagnostics_cache=qfim_spectral_gradient_diagnostics_cache,
    )

print(f"Saved numerical results to: {numerical_results_dir}")
