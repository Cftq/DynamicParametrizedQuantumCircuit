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
from pathlib import Path

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

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
import tensorcircuit as tc
from plot import (
    CIRCUIT_SAVE_PDF,
    CIRCUIT_SAVE_PNG,
    SAVE_DPI,
    SAVEFIG_PAD_INCHES,
    plot_qfim_grad_alignment_layer_overlay,
    plot_qfim_grad_alignment_table,
)
from qfim import (
    effective_rank_from_eigvals,
    effective_abs_rank_from_eigvals,
    hermitian as _hermitian,
    hermitian_eigvals_desc,
    make_hilbert_schmidt_metric_fn,
    make_mixed_state_qfim_fn,
    participation_effective_rank_from_eigvals,
    psd_eigvals_desc,
)
from hamiltonian import local_term_to_matrix
from tqdm.auto import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("dynamic_ncols", True)
    return _tqdm(*args, **kwargs)

jax.config.update("jax_enable_x64", True)

tc.set_backend("jax")
tc.set_dtype("complex128")

REAL_DTYPE = jnp.float64
COMPLEX_DTYPE = jnp.complex128
NP_REAL_DTYPE = np.float64
NP_COMPLEX_DTYPE = np.complex128
NP_INT_DTYPE = np.int64

from dpqc_overparam_common import (
    _normalize_index_list,
    _thr_tag,
    _time_index_from_iteration,
    build_H_matrix_jax,
    build_layer_list,
    hamiltonian_terms,
    ket0_density,
    load_npz_result,
    qfim_grad_alignment_at_point,
    qfim_grad_alignment_one_to_table,
    qg_layer,
    rho_zero_state,
    save_circuit_matplotlib_png,
    U_rz,
)

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


def jax_to_np(x, dtype=None):
    return np.asarray(jax.device_get(x), dtype=dtype)


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


def create_dpqc(theta: jnp.ndarray, num_layers: int, num_system: int) -> tc.Circuit:
    qc = tc.Circuit(num_system + num_layers)
    theta_layers = jnp.reshape(theta, (num_layers, num_params_per_layer))

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
# Hamiltonian & ground truth
# ==============================
H_terms = tuple(hamiltonian_terms(h_param))

H_matrix = build_H_matrix_jax(H_terms, num_system_qubits)

H_OBSERVABLE_MATRICES = jnp.stack(
    [
        jnp.asarray(coef, dtype=REAL_DTYPE)
        * local_term_to_matrix(local_ops, num_system_qubits)
        for coef, local_ops in H_terms
    ],
    axis=0,
)

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


def wrap_theta_periodic_only(theta: jnp.ndarray, num_layers: int) -> jnp.ndarray:
    theta_layers = jnp.reshape(theta, (num_layers, num_params_per_layer))
    theta_layers = theta_layers.at[:, :-EXTRA_PARAMS_PER_LAYER].set(
        wrap_to_pi(theta_layers[:, :-EXTRA_PARAMS_PER_LAYER])
    )
    return jnp.reshape(theta_layers, (-1,))


def theta_difference_periodic_only(
    theta_a: jnp.ndarray,
    theta_b: jnp.ndarray,
    num_layers: int,
) -> jnp.ndarray:
    d_layers = jnp.reshape(theta_a - theta_b, (num_layers, num_params_per_layer))
    d_periodic = wrap_to_pi(d_layers[:, :-EXTRA_PARAMS_PER_LAYER])
    d_nonperiodic = d_layers[:, -EXTRA_PARAMS_PER_LAYER:]
    d_layers = jnp.concatenate([d_periodic, d_nonperiodic], axis=1)

    return jnp.reshape(d_layers, (-1,))


def rms_theta_distance_periodic_only(
    theta_a: jnp.ndarray,
    theta_b: jnp.ndarray,
    num_layers: int,
) -> jnp.ndarray:
    d = theta_difference_periodic_only(theta_a, theta_b, num_layers=num_layers)
    return jnp.sqrt(jnp.mean(d**2))


# ============================================================
# Sequential trace-out machinery
# ============================================================
I2 = jnp.eye(2, dtype=COMPLEX_DTYPE)
X2 = jnp.array([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)
P0 = jnp.array([[1, 0], [0, 0]], dtype=COMPLEX_DTYPE)
P1 = jnp.array([[0, 0], [0, 1]], dtype=COMPLEX_DTYPE)


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


def rho_keep_sequential_dpqc(theta: jnp.ndarray, num_layers: int) -> jnp.ndarray:
    theta_layers = jnp.reshape(theta, (num_layers, num_params_per_layer))

    def one_layer(rho: jnp.ndarray, layer_theta: jnp.ndarray):
        rho6 = _rho6_after_layer(rho, layer_theta)
        rho_next = partial_trace_last_qubit(rho6, k=num_system_qubits)
        return rho_next, None

    rho_final, _ = jax.lax.scan(one_layer, _RHO_KEEP_INIT, theta_layers)

    return _hermitian(rho_final)


def ancilla_p1_sequential_dpqc(theta: jnp.ndarray, num_layers: int) -> jnp.ndarray:
    theta_layers = jnp.reshape(theta, (num_layers, num_params_per_layer))

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


def make_energy_fn_for_layer(num_layers: int):
    def energy_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return energy_from_rho_keep(
            rho_keep_sequential_dpqc(theta, num_layers=num_layers)
        )

    return energy_fn


def partial_trace_one_qubit(
    rho: jnp.ndarray,
    num_qubits: int,
    trace_wire: int,
) -> jnp.ndarray:
    num_qubits = int(num_qubits)
    trace_wire = int(trace_wire)

    if not 0 <= trace_wire < num_qubits:
        raise ValueError("trace_wire out of range.")

    keep = [i for i in range(num_qubits) if i != trace_wire]
    perm = keep + [trace_wire] + [i + num_qubits for i in keep] + [
        trace_wire + num_qubits
    ]

    rho_p = jnp.transpose(jnp.reshape(rho, (2,) * (2 * num_qubits)), perm)

    dim_keep = 2 ** (num_qubits - 1)
    rho_p = jnp.reshape(rho_p, (dim_keep, 2, dim_keep, 2))

    return rho_p[:, 0, :, 0] + rho_p[:, 1, :, 1]


def rho4_from_rho5(rho5: jnp.ndarray) -> jnp.ndarray:
    return partial_trace_one_qubit(rho5, num_qubits=5, trace_wire=4)


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
os.makedirs(circuit_dir, exist_ok=True)
os.makedirs(numerical_results_dir, exist_ok=True)
os.makedirs(energy_results_dir, exist_ok=True)
os.makedirs(qfim_results_dir, exist_ok=True)
os.makedirs(hs_results_dir, exist_ok=True)
os.makedirs(ortk_results_dir, exist_ok=True)
os.makedirs(hessian_results_dir, exist_ok=True)


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
    num_total_params = num_params_per_layer * current_layer

    energy_fn = make_energy_fn_for_layer(current_layer)
    energy_and_grad = jax.jit(jax.value_and_grad(energy_fn))

    @jax.jit
    def optimization_step_and_measure(theta, opt_state, g_old):
        updates, new_opt_state = optimizer.update(g_old, opt_state, theta)

        theta_new = wrap_theta_periodic_only(
            optax.apply_updates(theta, updates),
            num_layers=current_layer,
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
            shape=(num_total_params,),
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
            num_layers=current_layer,
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
        num_layers=current_layer,
        num_system=num_system_qubits,
    )

    save_circuit_matplotlib_png(
        best_tc_circ,
        os.path.join(circuit_dir, f"optimized_circuit_L{current_layer}.png"),
        num_qubits=num_system_qubits + current_layer,
        dpi=SAVE_DPI,
        pad_inches=SAVEFIG_PAD_INCHES,
        save_png=CIRCUIT_SAVE_PNG,
        save_pdf=CIRCUIT_SAVE_PDF,
        hide_params=True,
    )

    p1_runs = np.array(
        jax.jit(
            jax.vmap(
                lambda th: ancilla_p1_sequential_dpqc(
                    th,
                    num_layers=current_layer,
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
ORTK_RANK_THRESHOLD = cfg.ORTK_RANK_THRESHOLD
ORTK_PARTICIPATION_EPS = cfg.ORTK_PARTICIPATION_EPS

# Thresholds used for large-sector gradient-weight diagnostics.
# Keep this broad set unless you also want to reduce the gradient-sector plots.
THRESHOLDS = tuple(float(t) for t in cfg.GRADIENT_SECTOR_THRESHOLDS)


def make_reduced0123_qfim_matrix_fn_for_layer_sequential(
    num_layers: int,
    *,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    @jax.jit
    def rho_sub_fn(theta: jnp.ndarray) -> jnp.ndarray:
        rho5 = rho_keep_sequential_dpqc(theta, num_layers=num_layers)
        rho4 = rho4_from_rho5(rho5)
        return _hermitian(rho4)

    return make_mixed_state_qfim_fn(
        rho_sub_fn,
        eig_sum_eps=EIG_SUM_EPS,
        jvp_chunk=jvp_chunk,
    )


def make_reduced0123_hs_matrix_fn_for_layer_sequential(
    num_layers: int,
    *,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    @jax.jit
    def rho_sub_fn(theta: jnp.ndarray) -> jnp.ndarray:
        rho5 = rho_keep_sequential_dpqc(theta, num_layers=num_layers)
        rho4 = rho4_from_rho5(rho5)
        return _hermitian(rho4)

    return make_hilbert_schmidt_metric_fn(
        rho_sub_fn,
        jvp_chunk=jvp_chunk,
    )


def make_reduced0123_rho_rank_fn_for_layer_sequential(num_layers: int):
    @jax.jit
    def rho_rank_reduced0123(theta: jnp.ndarray) -> jnp.ndarray:
        rho5 = rho_keep_sequential_dpqc(theta, num_layers=num_layers)
        rho4 = _hermitian(rho4_from_rho5(rho5))
        evals = jnp.clip(jnp.linalg.eigvalsh(rho4), a_min=0.0)
        threshold = jnp.asarray(QFIM_EFFECTIVE_RANK_THRESHOLD, dtype=evals.dtype)

        return jnp.sum(evals > threshold)

    return rho_rank_reduced0123


def make_observable_expectation_vector_fn_for_layer(num_layers: int):
    """Create m(theta)=Tr[(c_a O_a) rho(theta)] for Hamiltonian terms."""

    @jax.jit
    def observable_expectation_vector(theta: jnp.ndarray) -> jnp.ndarray:
        rho_keep = rho_keep_sequential_dpqc(theta, num_layers=num_layers)
        values = jnp.einsum("aij,ji->a", H_OBSERVABLE_MATRICES, rho_keep)
        return jnp.real(values)

    return observable_expectation_vector


def make_observable_tangent_kernel_matrix_fn_for_layer(num_layers: int):
    """Create K_obs(theta)=J_obs(theta) J_obs(theta)^T."""
    obs_fn = make_observable_expectation_vector_fn_for_layer(num_layers)
    obs_jac_fn = jax.jacrev(obs_fn)

    @jax.jit
    def observable_tangent_kernel(theta: jnp.ndarray) -> jnp.ndarray:
        jac = obs_jac_fn(theta)
        kernel = jac @ jac.T
        return 0.5 * (kernel + kernel.T)

    return observable_tangent_kernel


def make_observable_tangent_kernel_eigvals_fn_for_layer(num_layers: int):
    ortk_fn = make_observable_tangent_kernel_matrix_fn_for_layer(
        num_layers=num_layers,
    )

    @jax.jit
    def observable_tangent_kernel_eigvals(theta: jnp.ndarray):
        return psd_eigvals_desc(ortk_fn(theta))

    return observable_tangent_kernel_eigvals


def make_energy_hessian_eigvals_fn_for_layer(num_layers: int):
    energy_fn = make_energy_fn_for_layer(num_layers)
    hessian_fn = jax.jit(jax.hessian(energy_fn))

    @jax.jit
    def hessian_eigvals(theta: jnp.ndarray):
        return hermitian_eigvals_desc(hessian_fn(theta))

    return hessian_eigvals


qfim_rank_reduced_0123_by_layer = {}
qfim_eigs_reduced_0123_by_layer = {}
qfim_rho_rank_reduced_0123_by_layer = {}

qfim_eigsum_reduced_0123_by_layer = {}
qfim_abs_entry_sum_reduced_0123_by_layer = {}
hs_rank_reduced_0123_by_layer = {}
hs_eigs_reduced_0123_by_layer = {}
hs_rho_rank_reduced_0123_by_layer = {}
hs_eigsum_reduced_0123_by_layer = {}
hs_abs_entry_sum_reduced_0123_by_layer = {}
ortk_rank_by_layer = {}
ortk_effective_rank_by_layer = {}
ortk_eigs_by_layer = {}
ortk_trace_by_layer = {}
hessian_rank_by_layer = {}
hessian_eigs_by_layer = {}
hessian_trace_by_layer = {}
hessian_abs_eigsum_by_layer = {}

qfim_eigs_dir = os.path.join(qfim_fig_dir, "eigs")
qfim_eigs_dir_red4 = os.path.join(qfim_eigs_dir, "reduced_keep_0123")
hs_eigs_dir = os.path.join(hs_fig_dir, "eigs")
hs_eigs_dir_red4 = os.path.join(hs_eigs_dir, "reduced_keep_0123")

os.makedirs(qfim_eigs_dir_red4, exist_ok=True)
os.makedirs(hs_eigs_dir_red4, exist_ok=True)

for L in tqdm(
    qfim_layer_list,
    desc="Layers (QFIM; reduced keep=(0,1,2,3))",
    unit="layer",
):
    num_params = num_params_per_layer * L

    thetas_L = jax.random.uniform(
        jax.random.PRNGKey(QFIM_SAMPLE_SEED_BASE + int(L)),
        shape=(NUM_QFIM_SAMPLES, num_params),
        dtype=REAL_DTYPE,
        minval=jnp.asarray(-jnp.pi, dtype=REAL_DTYPE),
        maxval=jnp.asarray(jnp.pi, dtype=REAL_DTYPE),
    )

    red4_qfim_fn = make_reduced0123_qfim_matrix_fn_for_layer_sequential(
        num_layers=L,
        jvp_chunk=RED_JVP_CHUNK,
    )
    red4_rho_rank_fn = make_reduced0123_rho_rank_fn_for_layer_sequential(
        num_layers=L,
    )

    rr4_list = []
    eigs4_list = []
    rho_rank4_list = []
    eigsum4_list = []
    abs_entry_sum4_list = []

    for s in tqdm(
        range(NUM_QFIM_SAMPLES),
        desc=f"Reduced QFIM samples keep=(0,1,2,3) (L={L})",
        unit="sample",
        leave=False,
    ):
        th = thetas_L[s]

        F4 = red4_qfim_fn(th)
        evals4_desc = psd_eigvals_desc(F4)
        r4 = effective_rank_from_eigvals(evals4_desc)
        rho_rank4 = red4_rho_rank_fn(th)

        evals4_np = jax_to_np(evals4_desc, dtype=NP_REAL_DTYPE)
        F4_np = jax_to_np(F4, dtype=NP_REAL_DTYPE)

        rr4_list.append(int(jax.device_get(r4)))
        eigs4_list.append(evals4_np)
        rho_rank4_list.append(int(jax.device_get(rho_rank4)))

        eigsum4_list.append(NP_REAL_DTYPE(np.sum(evals4_np)))
        abs_entry_sum4_list.append(NP_REAL_DTYPE(np.sum(np.abs(F4_np))))

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

    red4_hs_fn = make_reduced0123_hs_matrix_fn_for_layer_sequential(
        num_layers=L,
        jvp_chunk=RED_JVP_CHUNK,
    )

    hs_rank_list = []
    hs_eigs_list = []
    hs_rho_rank_list = []
    hs_eigsum_list = []
    hs_abs_entry_sum_list = []

    for s in tqdm(
        range(NUM_QFIM_SAMPLES),
        desc=f"Reduced HS samples keep=(0,1,2,3) (L={L})",
        unit="sample",
        leave=False,
    ):
        th = thetas_L[s]

        G4 = red4_hs_fn(th)
        hs_evals_desc = psd_eigvals_desc(G4)
        hs_rank = effective_rank_from_eigvals(hs_evals_desc)
        rho_rank4 = red4_rho_rank_fn(th)

        hs_evals_np = jax_to_np(hs_evals_desc, dtype=NP_REAL_DTYPE)
        G4_np = jax_to_np(G4, dtype=NP_REAL_DTYPE)

        hs_rank_list.append(int(jax.device_get(hs_rank)))
        hs_eigs_list.append(hs_evals_np)
        hs_rho_rank_list.append(int(jax.device_get(rho_rank4)))
        hs_eigsum_list.append(NP_REAL_DTYPE(np.sum(hs_evals_np)))
        hs_abs_entry_sum_list.append(NP_REAL_DTYPE(np.sum(np.abs(G4_np))))

    hs_rank_reduced_0123_by_layer[L] = np.asarray(
        hs_rank_list,
        dtype=NP_INT_DTYPE,
    )
    hs_eigs_reduced_0123_by_layer[L] = np.stack(hs_eigs_list, axis=0)
    hs_rho_rank_reduced_0123_by_layer[L] = np.asarray(
        hs_rho_rank_list,
        dtype=NP_INT_DTYPE,
    )
    hs_eigsum_reduced_0123_by_layer[L] = np.asarray(
        hs_eigsum_list,
        dtype=NP_REAL_DTYPE,
    )
    hs_abs_entry_sum_reduced_0123_by_layer[L] = np.asarray(
        hs_abs_entry_sum_list,
        dtype=NP_REAL_DTYPE,
    )

    ortk_eigvals_fn = make_observable_tangent_kernel_eigvals_fn_for_layer(
        num_layers=L,
    )

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
        th = thetas_L[s]

        ortk_eigs_desc = ortk_eigvals_fn(th)
        ortk_rank = effective_rank_from_eigvals(
            ortk_eigs_desc,
            threshold=ORTK_RANK_THRESHOLD,
        )
        ortk_effective_rank = participation_effective_rank_from_eigvals(
            ortk_eigs_desc,
            eps=ORTK_PARTICIPATION_EPS,
        )

        ortk_eigs_np = jax_to_np(ortk_eigs_desc, dtype=NP_REAL_DTYPE)

        ortk_rank_list.append(int(jax.device_get(ortk_rank)))
        ortk_effective_rank_list.append(
            NP_REAL_DTYPE(jax.device_get(ortk_effective_rank))
        )
        ortk_eigs_list.append(ortk_eigs_np)
        ortk_trace_list.append(NP_REAL_DTYPE(np.sum(ortk_eigs_np)))

    ortk_rank_by_layer[L] = np.asarray(
        ortk_rank_list,
        dtype=NP_INT_DTYPE,
    )
    ortk_effective_rank_by_layer[L] = np.asarray(
        ortk_effective_rank_list,
        dtype=NP_REAL_DTYPE,
    )
    ortk_eigs_by_layer[L] = np.stack(ortk_eigs_list, axis=0)
    ortk_trace_by_layer[L] = np.asarray(
        ortk_trace_list,
        dtype=NP_REAL_DTYPE,
    )

    hessian_eigvals_fn = make_energy_hessian_eigvals_fn_for_layer(num_layers=L)

    hessian_rank_list = []
    hessian_eigs_list = []
    hessian_trace_list = []
    hessian_abs_eigsum_list = []

    for s in tqdm(
        range(NUM_QFIM_SAMPLES),
        desc=f"Energy Hessian samples (rank+eigs) (L={L})",
        unit="sample",
        leave=False,
    ):
        th = thetas_L[s]

        hessian_eigs_desc = hessian_eigvals_fn(th)
        hessian_rank = effective_abs_rank_from_eigvals(hessian_eigs_desc)

        hessian_eigs_np = jax_to_np(hessian_eigs_desc, dtype=NP_REAL_DTYPE)

        hessian_rank_list.append(int(jax.device_get(hessian_rank)))
        hessian_eigs_list.append(hessian_eigs_np)
        hessian_trace_list.append(NP_REAL_DTYPE(np.sum(hessian_eigs_np)))
        hessian_abs_eigsum_list.append(NP_REAL_DTYPE(np.sum(np.abs(hessian_eigs_np))))

    hessian_rank_by_layer[L] = np.asarray(
        hessian_rank_list,
        dtype=NP_INT_DTYPE,
    )
    hessian_eigs_by_layer[L] = np.stack(hessian_eigs_list, axis=0)
    hessian_trace_by_layer[L] = np.asarray(
        hessian_trace_list,
        dtype=NP_REAL_DTYPE,
    )
    hessian_abs_eigsum_by_layer[L] = np.asarray(
        hessian_abs_eigsum_list,
        dtype=NP_REAL_DTYPE,
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
)

hs_random_points_result_path = os.path.join(
    hs_results_dir,
    f"hs_random_points_{keep_key}.npz",
)

save_npz_result(
    hs_random_points_result_path,
    h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
    num_hs_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
    hs_sample_seed_base=np.asarray(QFIM_SAMPLE_SEED_BASE, dtype=NP_INT_DTYPE),
    hs_effective_rank_threshold=np.asarray(QFIM_EFFECTIVE_RANK_THRESHOLD, dtype=NP_REAL_DTYPE),
    hs_eig_plot_eps=np.asarray(QFIM_EIG_PLOT_EPS, dtype=NP_REAL_DTYPE),
    red_jvp_chunk=np.asarray(RED_JVP_CHUNK, dtype=NP_INT_DTYPE),
    layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
    **_layer_arrays_for_npz(hs_rank_reduced_0123_by_layer, "rank"),
    **_layer_arrays_for_npz(hs_eigs_reduced_0123_by_layer, "eigs_desc"),
    **_layer_arrays_for_npz(hs_rho_rank_reduced_0123_by_layer, "rho_rank"),
    **_layer_arrays_for_npz(hs_eigsum_reduced_0123_by_layer, "trace"),
    **_layer_arrays_for_npz(hs_abs_entry_sum_reduced_0123_by_layer, "abs_entry_sum"),
)

ortk_random_points_result_path = os.path.join(
    ortk_results_dir,
    "ortk_random_points.npz",
)

save_npz_result(
    ortk_random_points_result_path,
    h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
    num_ortk_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
    ortk_sample_seed_base=np.asarray(QFIM_SAMPLE_SEED_BASE, dtype=NP_INT_DTYPE),
    ortk_rank_threshold=np.asarray(ORTK_RANK_THRESHOLD, dtype=NP_REAL_DTYPE),
    ortk_participation_eps=np.asarray(ORTK_PARTICIPATION_EPS, dtype=NP_REAL_DTYPE),
    ortk_num_observables=np.asarray(
        H_OBSERVABLE_MATRICES.shape[0],
        dtype=NP_INT_DTYPE,
    ),
    layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
    **_layer_arrays_for_npz(ortk_rank_by_layer, "rank"),
    **_layer_arrays_for_npz(ortk_effective_rank_by_layer, "effective_rank"),
    **_layer_arrays_for_npz(ortk_eigs_by_layer, "eigs_desc"),
    **_layer_arrays_for_npz(ortk_trace_by_layer, "trace"),
)

hessian_random_points_result_path = os.path.join(
    hessian_results_dir,
    "hessian_random_points.npz",
)

save_npz_result(
    hessian_random_points_result_path,
    h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
    num_hessian_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
    hessian_sample_seed_base=np.asarray(QFIM_SAMPLE_SEED_BASE, dtype=NP_INT_DTYPE),
    hessian_effective_rank_threshold=np.asarray(
        QFIM_EFFECTIVE_RANK_THRESHOLD,
        dtype=NP_REAL_DTYPE,
    ),
    hessian_eig_plot_eps=np.asarray(QFIM_EIG_PLOT_EPS, dtype=NP_REAL_DTYPE),
    layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
    **_layer_arrays_for_npz(hessian_rank_by_layer, "rank"),
    **_layer_arrays_for_npz(hessian_eigs_by_layer, "eigs_desc"),
    **_layer_arrays_for_npz(hessian_trace_by_layer, "trace"),
    **_layer_arrays_for_npz(hessian_abs_eigsum_by_layer, "abs_eigsum"),
)


# ============================================================
# Large-sector gradient weights: compute and save numerical results
# ============================================================
# Large-sector gradient weight along the VQE optimization path
#   color  = layer number
#   marker = QFIM eigenvalue threshold
# ============================================================
GRADIENT_SECTOR_NORM_EPS = 1e-24


def make_large_sector_gradient_weight_fn_for_layer(
    num_layers: int,
    thresholds,
    *,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    qfim_fn = make_reduced0123_qfim_matrix_fn_for_layer_sequential(
        num_layers=num_layers,
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
        weights = jnp.where(
            grad_norm_sq > GRADIENT_SECTOR_NORM_EPS,
            (coeffs**2) / grad_norm_sq,
            jnp.zeros_like(coeffs),
        )

        large_masks = evals[None, :] > thresholds_jnp[:, None]
        large_weights = jnp.sum(
            jnp.where(large_masks, weights[None, :], 0.0),
            axis=1,
        )

        return jnp.clip(large_weights, a_min=0.0, a_max=1.0)

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

        num_runs, num_times, _ = theta_samples.shape

        weight_fn = make_large_sector_gradient_weight_fn_for_layer(
            num_layers=int(L),
            thresholds=thresholds,
            jvp_chunk=jvp_chunk,
        )

        weights_L = {
            thr: np.full((num_runs, num_times), np.nan, dtype=NP_REAL_DTYPE)
            for thr in thresholds
        }

        for run_idx in tqdm(
            range(num_runs),
            desc=f"Gradient-sector runs (L={L})",
            unit="run",
            leave=False,
        ):
            for time_idx in range(num_times):
                weights = weight_fn(
                    jnp.asarray(theta_samples[run_idx, time_idx], dtype=REAL_DTYPE),
                    jnp.asarray(grad_samples[run_idx, time_idx], dtype=REAL_DTYPE),
                )
                weights_np = jax_to_np(weights, dtype=NP_REAL_DTYPE)

                for threshold_idx, threshold in enumerate(thresholds):
                    weights_L[threshold][run_idx, time_idx] = weights_np[threshold_idx]

        result[int(L)] = weights_L

    return result


qfim_gradient_large_sector_weight_by_layer = compute_large_sector_gradient_weight_by_layer(
    theta_sample_traces_by_layer,
    grad_sample_traces_by_layer,
    vqe_layer_list,
    THRESHOLDS,
    jvp_chunk=RED_JVP_CHUNK,
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
def make_qfim_eigvals_fn_for_layer(
    num_layers: int,
    *,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    qfim_fn = make_reduced0123_qfim_matrix_fn_for_layer_sequential(
        num_layers=num_layers,
        jvp_chunk=jvp_chunk,
    )

    @jax.jit
    def qfim_eigvals(theta: jnp.ndarray):
        F = qfim_fn(theta)
        return psd_eigvals_desc(F)

    return qfim_eigvals


def make_qfim_rank_fn_for_layer(
    num_layers: int,
    *,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    qfim_eigvals_fn = make_qfim_eigvals_fn_for_layer(
        num_layers=num_layers,
        jvp_chunk=jvp_chunk,
    )

    @jax.jit
    def qfim_rank(theta: jnp.ndarray):
        return effective_rank_from_eigvals(qfim_eigvals_fn(theta))

    return qfim_rank


def make_hs_eigvals_fn_for_layer(
    num_layers: int,
    *,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    hs_fn = make_reduced0123_hs_matrix_fn_for_layer_sequential(
        num_layers=num_layers,
        jvp_chunk=jvp_chunk,
    )

    @jax.jit
    def hs_eigvals(theta: jnp.ndarray):
        G = hs_fn(theta)
        return psd_eigvals_desc(G)

    return hs_eigvals


def make_hs_rank_fn_for_layer(
    num_layers: int,
    *,
    jvp_chunk: int = RED_JVP_CHUNK,
):
    hs_eigvals_fn = make_hs_eigvals_fn_for_layer(
        num_layers=num_layers,
        jvp_chunk=jvp_chunk,
    )

    @jax.jit
    def hs_rank(theta: jnp.ndarray):
        return effective_rank_from_eigvals(hs_eigvals_fn(theta))

    return hs_rank


def make_ortk_rank_effective_eigvals_fn_for_layer(num_layers: int):
    ortk_eigvals_fn = make_observable_tangent_kernel_eigvals_fn_for_layer(
        num_layers=num_layers,
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

        num_runs, num_times, num_params = theta_samples.shape
        eigvals_fn = make_qfim_eigvals_fn_for_layer(
            num_layers=int(L),
            jvp_chunk=jvp_chunk,
        )

        ranks_L = np.full((num_runs, num_times), np.nan, dtype=NP_REAL_DTYPE)
        eigs_L = None
        if return_eigs:
            eigs_L = np.full(
                (num_runs, num_times, num_params),
                np.nan,
                dtype=NP_REAL_DTYPE,
            )

        for run_idx in tqdm(
            range(num_runs),
            desc=f"QFIM-rank runs (L={L})",
            unit="run",
            leave=False,
        ):
            for time_idx in range(num_times):
                eigs_desc = eigvals_fn(
                    jnp.asarray(theta_samples[run_idx, time_idx], dtype=REAL_DTYPE)
                )
                rank_value = effective_rank_from_eigvals(eigs_desc)
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


def compute_ortk_rank_history_by_layer(
    theta_samples_by_layer: dict,
    layers,
    *,
    return_eigs: bool = False,
):
    rank_history_by_layer = {}
    effective_rank_history_by_layer = {}
    eigs_history_by_layer = {}

    for L in tqdm(
        layers,
        desc="ORTK rank history along optimization path",
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

        num_runs, num_times, _ = theta_samples.shape
        rank_effective_eigvals_fn = make_ortk_rank_effective_eigvals_fn_for_layer(
            num_layers=int(L),
        )

        ranks_L = np.full((num_runs, num_times), np.nan, dtype=NP_REAL_DTYPE)
        effective_ranks_L = np.full(
            (num_runs, num_times),
            np.nan,
            dtype=NP_REAL_DTYPE,
        )
        eigs_L = None
        if return_eigs:
            eigs_L = np.full(
                (num_runs, num_times, H_OBSERVABLE_MATRICES.shape[0]),
                np.nan,
                dtype=NP_REAL_DTYPE,
            )

        for run_idx in tqdm(
            range(num_runs),
            desc=f"ORTK-rank runs (L={L})",
            unit="run",
            leave=False,
        ):
            for time_idx in range(num_times):
                rank_value, effective_rank_value, eigs_desc = (
                    rank_effective_eigvals_fn(
                        jnp.asarray(
                            theta_samples[run_idx, time_idx],
                            dtype=REAL_DTYPE,
                        )
                    )
                )
                ranks_L[run_idx, time_idx] = NP_REAL_DTYPE(
                    jax.device_get(rank_value)
                )
                effective_ranks_L[run_idx, time_idx] = NP_REAL_DTYPE(
                    jax.device_get(effective_rank_value)
                )
                if return_eigs:
                    eigs_L[run_idx, time_idx, :] = jax_to_np(
                        eigs_desc,
                        dtype=NP_REAL_DTYPE,
                    )

        rank_history_by_layer[int(L)] = ranks_L
        effective_rank_history_by_layer[int(L)] = effective_ranks_L
        if return_eigs:
            eigs_history_by_layer[int(L)] = eigs_L

    if return_eigs:
        return (
            rank_history_by_layer,
            effective_rank_history_by_layer,
            eigs_history_by_layer,
        )

    return rank_history_by_layer, effective_rank_history_by_layer


def compute_hs_rank_history_by_layer(
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
        desc="HS rank history along optimization path",
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

        num_runs, num_times, num_params = theta_samples.shape
        eigvals_fn = make_hs_eigvals_fn_for_layer(
            num_layers=int(L),
            jvp_chunk=jvp_chunk,
        )

        ranks_L = np.full((num_runs, num_times), np.nan, dtype=NP_REAL_DTYPE)
        eigs_L = None
        if return_eigs:
            eigs_L = np.full(
                (num_runs, num_times, num_params),
                np.nan,
                dtype=NP_REAL_DTYPE,
            )

        for run_idx in tqdm(
            range(num_runs),
            desc=f"HS-rank runs (L={L})",
            unit="run",
            leave=False,
        ):
            for time_idx in range(num_times):
                eigs_desc = eigvals_fn(
                    jnp.asarray(theta_samples[run_idx, time_idx], dtype=REAL_DTYPE)
                )
                rank_value = effective_rank_from_eigvals(eigs_desc)
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


def compute_hessian_rank_history_by_layer(
    theta_samples_by_layer: dict,
    layers,
    *,
    return_eigs: bool = False,
):
    rank_history_by_layer = {}
    eigs_history_by_layer = {}

    for L in tqdm(
        layers,
        desc="Energy Hessian rank history along optimization path",
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

        num_runs, num_times, num_params = theta_samples.shape
        eigvals_fn = make_energy_hessian_eigvals_fn_for_layer(num_layers=int(L))

        ranks_L = np.full((num_runs, num_times), np.nan, dtype=NP_REAL_DTYPE)
        eigs_L = None
        if return_eigs:
            eigs_L = np.full(
                (num_runs, num_times, num_params),
                np.nan,
                dtype=NP_REAL_DTYPE,
            )

        for run_idx in tqdm(
            range(num_runs),
            desc=f"Energy Hessian-rank runs (L={L})",
            unit="run",
            leave=False,
        ):
            for time_idx in range(num_times):
                eigs_desc = eigvals_fn(
                    jnp.asarray(theta_samples[run_idx, time_idx], dtype=REAL_DTYPE)
                )
                rank_value = effective_abs_rank_from_eigvals(eigs_desc)
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


qfim_rank_history_by_layer, qfim_eigs_history_optimization_path_by_layer = (
    compute_qfim_rank_history_by_layer(
        theta_sample_traces_by_layer,
        vqe_layer_list,
        jvp_chunk=RED_JVP_CHUNK,
        return_eigs=True,
    )
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

hs_rank_history_by_layer, hs_eigs_history_optimization_path_by_layer = (
    compute_hs_rank_history_by_layer(
        theta_sample_traces_by_layer,
        vqe_layer_list,
        jvp_chunk=RED_JVP_CHUNK,
        return_eigs=True,
    )
)

hs_rank_history_result_path = os.path.join(
    hs_results_dir,
    f"hs_rank_history_optimization_path_{keep_key}.npz",
)

save_npz_result(
    hs_rank_history_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    **{
        f"L{int(L)}": arr
        for L, arr in hs_rank_history_by_layer.items()
    },
)

hs_eigs_history_result_path = os.path.join(
    hs_results_dir,
    f"hs_eigs_history_optimization_path_{keep_key}.npz",
)

save_npz_result(
    hs_eigs_history_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    **{
        f"L{int(L)}": arr
        for L, arr in hs_eigs_history_optimization_path_by_layer.items()
    },
)

hs_trace_history_optimization_path_by_layer = {
    int(L): np.sum(np.asarray(arr, dtype=NP_REAL_DTYPE), axis=2)
    for L, arr in hs_eigs_history_optimization_path_by_layer.items()
}

hs_trace_history_result_path = os.path.join(
    hs_results_dir,
    f"hs_trace_history_optimization_path_{keep_key}.npz",
)

save_npz_result(
    hs_trace_history_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    **{
        f"L{int(L)}": arr
        for L, arr in hs_trace_history_optimization_path_by_layer.items()
    },
)

(
    ortk_rank_history_by_layer,
    ortk_effective_rank_history_by_layer,
    ortk_eigs_history_optimization_path_by_layer,
) = compute_ortk_rank_history_by_layer(
    theta_sample_traces_by_layer,
    vqe_layer_list,
    return_eigs=True,
)

ortk_rank_history_result_path = os.path.join(
    ortk_results_dir,
    "ortk_rank_history_optimization_path.npz",
)

save_npz_result(
    ortk_rank_history_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    ortk_rank_threshold=np.asarray(ORTK_RANK_THRESHOLD, dtype=NP_REAL_DTYPE),
    **{
        f"L{int(L)}": arr
        for L, arr in ortk_rank_history_by_layer.items()
    },
)

ortk_effective_rank_history_result_path = os.path.join(
    ortk_results_dir,
    "ortk_effective_rank_history_optimization_path.npz",
)

save_npz_result(
    ortk_effective_rank_history_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    ortk_participation_eps=np.asarray(ORTK_PARTICIPATION_EPS, dtype=NP_REAL_DTYPE),
    **{
        f"L{int(L)}": arr
        for L, arr in ortk_effective_rank_history_by_layer.items()
    },
)

ortk_eigs_history_result_path = os.path.join(
    ortk_results_dir,
    "ortk_eigs_history_optimization_path.npz",
)

save_npz_result(
    ortk_eigs_history_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    ortk_rank_threshold=np.asarray(ORTK_RANK_THRESHOLD, dtype=NP_REAL_DTYPE),
    ortk_participation_eps=np.asarray(ORTK_PARTICIPATION_EPS, dtype=NP_REAL_DTYPE),
    **{
        f"L{int(L)}": arr
        for L, arr in ortk_eigs_history_optimization_path_by_layer.items()
    },
)

ortk_trace_history_optimization_path_by_layer = {
    int(L): np.sum(np.asarray(arr, dtype=NP_REAL_DTYPE), axis=2)
    for L, arr in ortk_eigs_history_optimization_path_by_layer.items()
}

ortk_trace_history_result_path = os.path.join(
    ortk_results_dir,
    "ortk_trace_history_optimization_path.npz",
)

save_npz_result(
    ortk_trace_history_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    **{
        f"L{int(L)}": arr
        for L, arr in ortk_trace_history_optimization_path_by_layer.items()
    },
)

hessian_rank_history_by_layer, hessian_eigs_history_optimization_path_by_layer = (
    compute_hessian_rank_history_by_layer(
        theta_sample_traces_by_layer,
        vqe_layer_list,
        return_eigs=True,
    )
)

hessian_rank_history_result_path = os.path.join(
    hessian_results_dir,
    "hessian_rank_history_optimization_path.npz",
)

save_npz_result(
    hessian_rank_history_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    hessian_effective_rank_threshold=np.asarray(
        QFIM_EFFECTIVE_RANK_THRESHOLD,
        dtype=NP_REAL_DTYPE,
    ),
    **{
        f"L{int(L)}": arr
        for L, arr in hessian_rank_history_by_layer.items()
    },
)

hessian_eigs_history_result_path = os.path.join(
    hessian_results_dir,
    "hessian_eigs_history_optimization_path.npz",
)

save_npz_result(
    hessian_eigs_history_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    hessian_effective_rank_threshold=np.asarray(
        QFIM_EFFECTIVE_RANK_THRESHOLD,
        dtype=NP_REAL_DTYPE,
    ),
    **{
        f"L{int(L)}": arr
        for L, arr in hessian_eigs_history_optimization_path_by_layer.items()
    },
)

hessian_trace_history_optimization_path_by_layer = {
    int(L): np.sum(np.asarray(arr, dtype=NP_REAL_DTYPE), axis=2)
    for L, arr in hessian_eigs_history_optimization_path_by_layer.items()
}
hessian_abs_eigsum_history_optimization_path_by_layer = {
    int(L): np.sum(np.abs(np.asarray(arr, dtype=NP_REAL_DTYPE)), axis=2)
    for L, arr in hessian_eigs_history_optimization_path_by_layer.items()
}

hessian_trace_history_result_path = os.path.join(
    hessian_results_dir,
    "hessian_trace_history_optimization_path.npz",
)

save_npz_result(
    hessian_trace_history_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    **{
        f"L{int(L)}": arr
        for L, arr in hessian_trace_history_optimization_path_by_layer.items()
    },
)

hessian_abs_eigsum_history_result_path = os.path.join(
    hessian_results_dir,
    "hessian_abs_eigsum_history_optimization_path.npz",
)

save_npz_result(
    hessian_abs_eigsum_history_result_path,
    sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
    layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
    **{
        f"L{int(L)}": arr
        for L, arr in hessian_abs_eigsum_history_optimization_path_by_layer.items()
    },
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
# is applied here. Plot-level positive floors are only for numerical safety
# in log-scale visualization.
# ============================================================

qfim_grad_align_dir = os.path.join(qfim_fig_dir, "grad_alignment")
qfim_grad_align_results_dir = os.path.join(qfim_results_dir, "grad_alignment")
os.makedirs(qfim_grad_align_dir, exist_ok=True)
os.makedirs(qfim_grad_align_results_dir, exist_ok=True)


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

    num_runs, num_times, _ = theta_samples.shape

    run_ids = _normalize_index_list(run_indices, num_runs)
    time_ids = _normalize_index_list(time_indices, num_times)

    if sample_iters is None:
        sample_iters_arr = np.arange(num_times, dtype=NP_INT_DTYPE)
    else:
        sample_iters_arr = np.asarray(sample_iters, dtype=NP_INT_DTYPE)

    qfim_fn = make_reduced0123_qfim_matrix_fn_for_layer_sequential(
        num_layers=int(L),
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
        num_times_L = theta_samples_L.shape[1] if theta_samples_L.ndim == 3 else 1

        if use_all_sampled_times:
            time_indices = range(num_times_L)
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
    )

print(f"Saved numerical results to: {numerical_results_dir}")


