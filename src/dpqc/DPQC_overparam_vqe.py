#!/usr/bin/env python
# coding: utf-8
"""Run the DPQC VQE optimization and save its numerical histories.

This is the VQE half of the former DPQC_overparam_compute.py workflow.
It writes the archive consumed by DPQC_overparam_qfim.py as well as the
energy/success-probability result files. QFIM calculations are never run here.

Examples::

    python DPQC_overparam_vqe.py
    python DPQC_overparam_vqe.py --h-param 0.10 --vqe-batch-size 20
"""

import argparse
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Tuple


# Support direct execution from any working directory, for example:
#     python C:\...\src\dpqc\DPQC_overparam_vqe.py
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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be a finite number")
    return parsed



def _parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run DPQC VQE optimization and save histories for the later "
            "QFIM, visualization, and circuit-drawing stages."
        )
    )
    parser.add_argument(
        "--h-param",
        type=_finite_float,
        default=float(cfg.H_PARAM),
        help=(
            "Hamiltonian parameter H_PARAM (default: value from "
            "config_overparam.py)."
        ),
    )
    parser.add_argument(
        "--vqe-batch-size",
        type=_positive_int,
        default=int(getattr(cfg, "VQE_BATCH_SIZE", 5)),
        help="Number of independent VQE trials evaluated by each vmap call.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    _CLI_ARGS = _parse_cli_args()
else:
    _CLI_ARGS = argparse.Namespace(
        h_param=float(cfg.H_PARAM),
        vqe_batch_size=int(getattr(cfg, "VQE_BATCH_SIZE", 5)),
    )

VQE_BATCH_SIZE = int(_CLI_ARGS.vqe_batch_size)
RUN_VQE_STAGE = True
RUN_QFIM_STAGE = False

# ------------------------------------------------------------
# IMPORTANT: env vars should be set BEFORE importing jax
# ------------------------------------------------------------
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc
from tqdm.auto import tqdm

jax.config.update("jax_enable_x64", True)

# The shared Hamiltonian helper obtains Pauli tensors from TensorCircuit.
# Preserve its historical JAX/complex128 backend; no circuit is constructed or
# drawn in this VQE script.
tc.set_backend("jax")
tc.set_dtype("complex128")

REAL_DTYPE = jnp.float64
COMPLEX_DTYPE = jnp.complex128
NP_REAL_DTYPE = np.float64
NP_COMPLEX_DTYPE = np.complex128
NP_INT_DTYPE = np.int64

from dpqc_overparam_common import (
    _thr_tag,
    build_dpqc_vqe_optimizer,
    build_H_matrix_jax,
    build_layer_list,
    dpqc_vqe_optimizer_display_name,
    hamiltonian_terms,
    load_npz_result,
    normalize_dpqc_vqe_optimizer_name,
    rho_zero_state,
    threshold_psd_eigvals_for_rank,
)

# ============================================================
# Shared constants / helpers
# ============================================================
num_system_qubits = 5
h_param = float(_CLI_ARGS.h_param)
tolerance = cfg.TOLERANCE
steps = cfg.STEPS
num_runs = int(cfg.NUM_RUNS)
if RUN_VQE_STAGE and num_runs <= 0:
    raise ValueError("cfg.NUM_RUNS must be a positive integer.")
lr = cfg.LEARNING_RATE
vqe_optimizer_name = normalize_dpqc_vqe_optimizer_name(
    getattr(cfg, "DPQC_VQE_OPTIMIZER", "adam")
)
success_probability_thresholds = np.asarray(
    cfg.SUCCESS_PROBABILITY_THRESHOLDS,
    dtype=NP_REAL_DTYPE,
)

# Optimization-history sampling points saved for downstream diagnostics.
eps = 1e-12
sample_every = cfg.SAMPLE_EVERY

# Store optimized-path parameters at these iterations for the QFIM stage.
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

TOP, LEFT, RIGHT, BOTTOM, ANC_CENTER = 0, 1, 2, 3, 4

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


def jax_to_np(x, dtype=None):
    return np.asarray(jax.device_get(x), dtype=dtype)


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
# Sequential Kraus-channel machinery
# ============================================================
X2 = jnp.array([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)
_RHO_QUBIT_ZERO = jnp.array([[1, 0], [0, 0]], dtype=COMPLEX_DTYPE)


def U_rz(theta: jnp.ndarray) -> jnp.ndarray:
    th = jnp.asarray(theta, dtype=REAL_DTYPE)

    return jnp.array(
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

    return c * jnp.eye(4, dtype=COMPLEX_DTYPE) - 1j * s * jnp.kron(X2, X2)


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


def _apply_dynamic_delay_kraus(
    rho: jnp.ndarray,
    varphi: jnp.ndarray,
    phi: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the fresh-ancilla interaction without constructing the ancilla.

    The original dilation initializes a fresh ancilla in ``|0>``, applies

        CX(center -> ancilla),
        CRZ(ancilla -> center, varphi),
        CRX(ancilla -> center, 2 * phi),
        CRZ(ancilla -> center, varphi),

    and then traces out the ancilla.  Because the ancilla is only a control
    after the CX, the exactly equivalent channel on the center qubit has

        K0 = |0><0|,
        K1 = |psi><1|,
        |psi> = -i sin(phi)|0> + exp(i varphi) cos(phi)|1>.

    ``ANC_CENTER`` is the final kept-system wire, so the density matrix can be
    viewed as blocks indexed by that qubit.  The off-diagonal blocks vanish
    on tracing out the fresh ancilla.  The returned probability is the same
    final fresh-ancilla ``p(1)`` used by the historical implementation.
    """
    expected_shape = (2**num_system_qubits, 2**num_system_qubits)
    if rho.shape != expected_shape:
        raise ValueError(
            f"Expected a kept-state density matrix of shape {expected_shape}, "
            f"got {rho.shape}."
        )
    if ANC_CENTER != num_system_qubits - 1:
        raise ValueError("The direct Kraus channel requires ANC_CENTER last.")

    dim_rest = 2 ** (num_system_qubits - 1)
    rho_blocks = jnp.reshape(rho, (dim_rest, 2, dim_rest, 2))
    rho00 = rho_blocks[:, 0, :, 0]
    rho11 = rho_blocks[:, 1, :, 1]

    sin_phi = jnp.sin(jnp.asarray(phi, dtype=REAL_DTYPE)).astype(COMPLEX_DTYPE)
    cos_phi = jnp.cos(jnp.asarray(phi, dtype=REAL_DTYPE)).astype(COMPLEX_DTYPE)
    phase = jnp.exp(
        1j * jnp.asarray(varphi, dtype=REAL_DTYPE)
    ).astype(COMPLEX_DTYPE)
    psi = jnp.stack((-1j * sin_phi, phase * cos_phi))
    rho_psi = psi[:, None] * jnp.conjugate(psi[None, :])

    rho_next = (
        jnp.einsum("rs,ab->rasb", rho00, _RHO_QUBIT_ZERO)
        + jnp.einsum("rs,ab->rasb", rho11, rho_psi)
    )
    p1 = jnp.real(jnp.trace(rho11))

    return jnp.reshape(rho_next, expected_shape), p1


def _rho5_after_layer(
    rho: jnp.ndarray,
    layer_theta: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    rho = _apply_kept_blocks(rho, layer_theta)

    varphi = layer_theta[-2]
    phi = layer_theta[-1]

    return _apply_dynamic_delay_kraus(rho, varphi, phi)


def rho_keep_sequential_dpqc(theta: jnp.ndarray, n_layer: int) -> jnp.ndarray:
    theta_layers = jnp.reshape(theta, (n_layer, n_param_per_layer))

    def one_layer(rho: jnp.ndarray, layer_theta: jnp.ndarray):
        rho_next, _ = _rho5_after_layer(rho, layer_theta)
        return rho_next, None

    rho_final, _ = jax.lax.scan(one_layer, _RHO_KEEP_INIT, theta_layers)

    return _hermitian(rho_final)


def ancilla_p1_sequential_dpqc(theta: jnp.ndarray, n_layer: int) -> jnp.ndarray:
    theta_layers = jnp.reshape(theta, (n_layer, n_param_per_layer))

    def one_layer(rho: jnp.ndarray, layer_theta: jnp.ndarray):
        rho_next, p1 = _rho5_after_layer(rho, layer_theta)
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
    """Trace wire 4 from rho5, preserving any leading batch dimensions."""
    rho5 = jnp.asarray(rho5)
    expected_shape = (2**num_system_qubits, 2**num_system_qubits)
    if rho5.ndim < 2 or tuple(rho5.shape[-2:]) != expected_shape:
        raise ValueError(
            "rho5 must end in density-matrix dimensions "
            f"{expected_shape}, got {rho5.shape}."
        )

    dim_keep = 2 ** (num_system_qubits - 1)
    rho5_tensor = jnp.reshape(
        rho5,
        rho5.shape[:-2] + (dim_keep, 2, dim_keep, 2),
    )

    # The final two-dimensional axes are ket/bra wire 4.  This works for both
    # rho5 with shape (32, 32) and d_rho5 chunks with shape (B, 32, 32).
    return jnp.trace(rho5_tensor, axis1=-3, axis2=-1)


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

qfim_dense_until_layer = cfg.DPQC_QFIM_DENSE_UNTIL_LAYER
qfim_max_layer = cfg.DPQC_QFIM_MAX_LAYER
qfim_sparse_step = cfg.DPQC_QFIM_SPARSE_STEP

qfim_layer_list = build_layer_list(
    qfim_max_layer,
    qfim_dense_until_layer,
    qfim_sparse_step,
)

if RUN_VQE_STAGE and not vqe_layer_list:
    raise ValueError(
        "vqe_layer_list is empty. Check vqe_max_layer, "
        "vqe_dense_until_layer, and vqe_sparse_step."
    )

if RUN_QFIM_STAGE and not qfim_layer_list:
    raise ValueError(
        "qfim_layer_list is empty. Check DPQC_QFIM_MAX_LAYER, "
        "DPQC_QFIM_DENSE_UNTIL_LAYER, and DPQC_QFIM_SPARSE_STEP."
    )

save_dir = f"./figs/dpqc/h_{h_param}"

energy_fig_dir = os.path.join(save_dir, "energy_figures")
qfim_fig_dir = os.path.join(save_dir, "qfim_figures")
numerical_results_dir = os.path.join(save_dir, "numerical_results")
energy_results_dir = os.path.join(numerical_results_dir, "energy")
qfim_results_dir = os.path.join(numerical_results_dir, "qfim")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(energy_fig_dir, exist_ok=True)
os.makedirs(qfim_fig_dir, exist_ok=True)
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
vqe_optimization_result_path = os.path.join(
    energy_results_dir,
    "vqe_optimization_histories.npz",
)


def _vqe_sample_slot_by_iteration(
    num_steps: int,
    sample_iterations,
) -> np.ndarray:
    """Map an optimization iteration to its fixed sample-buffer slot."""
    num_steps = int(num_steps)
    sample_iterations = np.asarray(sample_iterations, dtype=NP_INT_DTYPE)

    if num_steps < 0:
        raise ValueError("num_steps must be nonnegative.")
    if sample_iterations.ndim != 1:
        raise ValueError("sample_iterations must be one-dimensional.")
    if np.unique(sample_iterations).size != sample_iterations.size:
        raise ValueError("sample_iterations must not contain duplicates.")
    if np.any(sample_iterations < 0) or np.any(sample_iterations > num_steps):
        raise ValueError(
            "sample_iterations must lie in the inclusive range "
            f"[0, {num_steps}]."
        )

    slot_by_iteration = np.full(num_steps + 1, -1, dtype=np.int32)
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
    """Compile one fixed-size batch of independent VQE optimization runs."""
    current_layer = int(current_layer)
    num_steps = int(num_steps)
    n_total_params = n_param_per_layer * current_layer
    slot_by_iteration = _vqe_sample_slot_by_iteration(
        num_steps,
        sample_iterations,
    )
    num_samples = int(np.asarray(sample_iterations).size)
    initial_sample_slot = int(slot_by_iteration[0])
    scan_sample_slots = jnp.asarray(
        slot_by_iteration[1:],
        dtype=jnp.int32,
    )

    def energy_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return energy_from_rho_keep(
            rho_keep_sequential_dpqc(theta, n_layer=current_layer)
        )

    # The outer jit(vmap(scan)) compiles this differentiation together with
    # the complete optimizer loop, avoiding one Python dispatch per step.
    energy_and_grad = jax.value_and_grad(energy_fn)

    def optimize_one_run(theta_initial: jnp.ndarray):
        theta = jnp.asarray(theta_initial, dtype=REAL_DTYPE)
        opt_state = optimizer.init(theta)
        energy_initial, grad = energy_and_grad(theta)

        theta_samples = jnp.zeros(
            (num_samples, n_total_params),
            dtype=REAL_DTYPE,
        )

        if initial_sample_slot >= 0:
            theta_samples = theta_samples.at[initial_sample_slot].set(theta)

        def one_step(carry, sample_slot):
            (
                theta_old,
                opt_state_old,
                grad_old,
                theta_samples_old,
            ) = carry

            updates, opt_state_new = optimizer.update(
                grad_old,
                opt_state_old,
                theta_old,
            )
            theta_new = wrap_theta_periodic_only(
                optax.apply_updates(theta_old, updates),
                n_layer=current_layer,
            )
            energy_new, grad_new = energy_and_grad(theta_new)
            grad_norm_new = jnp.linalg.norm(grad_new)

            theta_samples_new = jax.lax.cond(
                sample_slot >= 0,
                lambda theta_buffer: theta_buffer.at[sample_slot].set(theta_new),
                lambda theta_buffer: theta_buffer,
                theta_samples_old,
            )

            new_carry = (
                theta_new,
                opt_state_new,
                grad_new,
                theta_samples_new,
            )
            measurements = (energy_new, grad_norm_new)
            return new_carry, measurements

        (
            (
                theta_final,
                _,
                _,
                theta_samples_final,
            ),
            (energy_after_steps, grad_norm_after_steps),
        ) = jax.lax.scan(
            one_step,
            (
                theta,
                opt_state,
                grad,
                theta_samples,
            ),
            scan_sample_slots,
        )

        energy_trace = jnp.concatenate(
            (energy_initial[None], energy_after_steps),
            axis=0,
        )
        grad_norm_trace = jnp.concatenate(
            (jnp.linalg.norm(grad)[None], grad_norm_after_steps),
            axis=0,
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
) -> Tuple[jnp.ndarray, int]:
    """Pad a final partial batch without changing any valid vmap lane."""
    batch_size = int(batch_size)
    valid_count = int(theta_batch.shape[0])

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


def _run_vqe_optimization():
    """Run scan/vmap VQE, save optimization histories, and return samples."""
    optimizer = build_dpqc_vqe_optimizer(vqe_optimizer_name, lr)
    print(
        "VQE optimizer: "
        f"{dpqc_vqe_optimizer_display_name(vqe_optimizer_name)} "
        f"(learning_rate={float(lr):g})"
    )

    theta_history = {}
    ancilla_p1_stats_by_layer = {}
    final_theta_periodic_only_rmsdist_by_layer = {}
    energy_traces_by_layer = {}
    grad_norm_traces_by_layer = {}
    theta_sample_traces_by_layer = {}
    success_rates_history = {}
    final_stats = {
        "layer": [],
        "success_rate": [],
        "mean_energy": [],
        "std_energy": [],
    }

    for current_layer in tqdm(
        vqe_layer_list,
        desc="Layers (VQE)",
        unit="layer",
    ):
        n_total_params = n_param_per_layer * current_layer
        run_vqe_batch = make_vqe_batch_runner(
            current_layer,
            num_steps=steps,
            sample_iterations=sample_iters,
            optimizer=optimizer,
        )

        # Preserve the exact historical key sequence and run ordering.
        keys = jax.random.split(
            jax.random.PRNGKey(current_layer),
            num_runs,
        )
        theta_initial_runs = jnp.stack(
            [
                jax.random.uniform(
                    keys[run_index],
                    shape=(n_total_params,),
                    dtype=REAL_DTYPE,
                    minval=jnp.asarray(-jnp.pi, dtype=REAL_DTYPE),
                    maxval=jnp.asarray(jnp.pi, dtype=REAL_DTYPE),
                )
                for run_index in range(num_runs)
            ],
            axis=0,
        )

        output_parts = tuple([] for _ in range(4))
        batch_starts = range(0, num_runs, VQE_BATCH_SIZE)

        for batch_start in tqdm(
            batch_starts,
            total=(num_runs + VQE_BATCH_SIZE - 1) // VQE_BATCH_SIZE,
            desc=f"Run batches (L={current_layer}, batch={VQE_BATCH_SIZE})",
            unit="batch",
            leave=False,
        ):
            batch_end = min(batch_start + VQE_BATCH_SIZE, num_runs)
            theta_batch, valid_count = _pad_vqe_theta_batch(
                theta_initial_runs[batch_start:batch_end],
                VQE_BATCH_SIZE,
            )

            # One transfer/synchronization per five runs replaces the former
            # two scalar synchronizations at every optimizer step.
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
            (num_runs, n_total_params),
            (num_runs, steps + 1),
            (num_runs, steps + 1),
            (num_runs, sample_iters.size, n_total_params),
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

        best_final_theta = None
        best_final_energy = np.inf
        for run_index in range(num_runs):
            final_energy = energy_data[run_index, -1]
            if final_energy < best_final_energy:
                best_final_energy = final_energy
                best_final_theta = theta_final_data[run_index].copy()

        if best_final_theta is None:
            raise FloatingPointError(
                f"No finite final VQE energy was produced for L={current_layer}."
            )

        theta_history[current_layer] = theta_final_data
        theta_runs_jnp = jnp.asarray(theta_final_data, dtype=REAL_DTYPE)
        theta_ref_jnp = jnp.asarray(best_final_theta, dtype=REAL_DTYPE)

        d_theta_runs = jax.vmap(
            lambda th: rms_theta_distance_periodic_only(
                th,
                theta_ref_jnp,
                n_layer=current_layer,
            )
        )(theta_runs_jnp)

        final_theta_periodic_only_rmsdist_by_layer[current_layer] = (
            jax_to_np(d_theta_runs, dtype=NP_REAL_DTYPE)
        )
        energy_traces_by_layer[current_layer] = energy_data
        grad_norm_traces_by_layer[current_layer] = gradnorm_data
        theta_sample_traces_by_layer[current_layer] = theta_sample_data

        p1_runs = jax_to_np(
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

    save_npz_result(
        vqe_optimization_result_path,
        h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
        tolerance=np.asarray(tolerance, dtype=NP_REAL_DTYPE),
        steps=np.asarray(steps, dtype=NP_INT_DTYPE),
        num_runs=np.asarray(num_runs, dtype=NP_INT_DTYPE),
        optimizer_name=np.asarray(vqe_optimizer_name),
        lr=np.asarray(lr, dtype=NP_REAL_DTYPE),
        smallest_eigval=np.asarray(
            smallest_eigval,
            dtype=NP_REAL_DTYPE,
        ),
        sample_every=np.asarray(sample_every, dtype=NP_INT_DTYPE),
        sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        vqe_layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
        qfim_layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
        final_stats_layer=np.asarray(
            final_stats["layer"],
            dtype=NP_INT_DTYPE,
        ),
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
        **_layer_arrays_for_npz(energy_traces_by_layer, "energy_traces"),
        **_layer_arrays_for_npz(
            grad_norm_traces_by_layer,
            "grad_norm_traces",
        ),
        **_layer_arrays_for_npz(theta_history, "theta_final"),
        **_layer_arrays_for_npz(
            theta_sample_traces_by_layer,
            "theta_samples",
        ),
        **_layer_arrays_for_npz(success_rates_history, "success_rates"),
        **_layer_arrays_for_npz(
            final_theta_periodic_only_rmsdist_by_layer,
            "theta_rmsdist",
        ),
        **{
            f"L{int(layer)}_ancilla_qubits": data["ancilla_qubits"]
            for layer, data in ancilla_p1_stats_by_layer.items()
        },
        **{
            f"L{int(layer)}_ancilla_p1_runs": data["p1_runs"]
            for layer, data in ancilla_p1_stats_by_layer.items()
        },
        **{
            f"L{int(layer)}_ancilla_p1_mean": data["mean"]
            for layer, data in ancilla_p1_stats_by_layer.items()
        },
        **{
            f"L{int(layer)}_ancilla_p1_var": data["var"]
            for layer, data in ancilla_p1_stats_by_layer.items()
        },
        **{
            f"L{int(layer)}_ancilla_p1_std": data["std"]
            for layer, data in ancilla_p1_stats_by_layer.items()
        },
    )

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
        optimizer_name=np.asarray(vqe_optimizer_name),
        lr=np.asarray(lr, dtype=NP_REAL_DTYPE),
        layers=multiple_tolerance_success["layers"],
        thresholds=multiple_tolerance_success["thresholds"],
        tolerances=multiple_tolerance_success["thresholds"],
        num_trials=multiple_tolerance_success["num_trials"],
        num_runs=multiple_tolerance_success["num_trials"],
        R=multiple_tolerance_success["num_trials"],
        trial_indices=np.arange(1, num_runs + 1, dtype=NP_INT_DTYPE),
        ground_energy=multiple_tolerance_success["ground_energy"],
        smallest_eigval=multiple_tolerance_success["ground_energy"],
        final_energies=multiple_tolerance_success["final_energies"],
        raw_final_energies=multiple_tolerance_success["final_energies"],
        raw_final_energy_errors=multiple_tolerance_success[
            "raw_final_energy_errors"
        ],
        final_energy_errors=multiple_tolerance_success[
            "final_energy_errors"
        ],
        success_indicators=multiple_tolerance_success[
            "success_indicators"
        ],
        success_counts=multiple_tolerance_success["success_counts"],
        success_probabilities=multiple_tolerance_success[
            "success_probabilities"
        ],
        energy_error_clipped_at_zero=np.asarray(True, dtype=np.bool_),
        success_comparison=np.asarray("<="),
        probability_denominator=np.asarray("num_trials"),
        **_layer_arrays_for_npz(
            {
                int(layer): multiple_tolerance_success["final_energies"][
                    layer_index
                ]
                for layer_index, layer in enumerate(
                    multiple_tolerance_success["layers"]
                )
            },
            "final_energies",
        ),
        **_layer_arrays_for_npz(
            {
                int(layer): multiple_tolerance_success[
                    "raw_final_energy_errors"
                ][layer_index]
                for layer_index, layer in enumerate(
                    multiple_tolerance_success["layers"]
                )
            },
            "raw_final_energy_errors",
        ),
        **_layer_arrays_for_npz(
            {
                int(layer): multiple_tolerance_success[
                    "final_energy_errors"
                ][layer_index]
                for layer_index, layer in enumerate(
                    multiple_tolerance_success["layers"]
                )
            },
            "final_energy_errors",
        ),
        **_layer_arrays_for_npz(
            {
                int(layer): multiple_tolerance_success[
                    "success_indicators"
                ][layer_index]
                for layer_index, layer in enumerate(
                    multiple_tolerance_success["layers"]
                )
            },
            "success_indicators",
        ),
        **_layer_arrays_for_npz(
            {
                int(layer): multiple_tolerance_success["success_counts"][
                    layer_index
                ]
                for layer_index, layer in enumerate(
                    multiple_tolerance_success["layers"]
                )
            },
            "success_counts",
        ),
        **_layer_arrays_for_npz(
            {
                int(layer): multiple_tolerance_success[
                    "success_probabilities"
                ][layer_index]
                for layer_index, layer in enumerate(
                    multiple_tolerance_success["layers"]
                )
            },
            "success_probabilities",
        ),
    )

    return theta_sample_traces_by_layer


def run_vqe():
    """Execute VQE, save all results, and return sampled parameter histories."""
    return _run_vqe_optimization()


def main():
    run_vqe()
    print(f"Saved VQE numerical results to: {energy_results_dir}")
    print(
        "QFIM was not run. To compute it from the saved VQE archive, run "
        "src/dpqc/DPQC_overparam_qfim.py."
    )
    print(
        "Circuit drawing was skipped. To draw the saved optimized circuits, "
        "run src/dpqc/DPQC_overparam_draw_circuits.py separately."
    )


if __name__ == "__main__":
    main()
