#!/usr/bin/env python
# coding: utf-8
"""Shared DPQC overparameterization helpers used by compute and visualize."""


import os
from functools import reduce
from typing import Tuple

import config_overparam as cfg
import jax
import jax.numpy as jnp
import numpy as np
import tensorcircuit as tc


REAL_DTYPE = jnp.float64
COMPLEX_DTYPE = jnp.complex128
NP_REAL_DTYPE = np.float64
NP_INT_DTYPE = np.int64


def build_layer_list(max_layer: int, dense_until_layer: int, sparse_step: int):
    dense_end = min(dense_until_layer, max_layer)
    return list(range(1, dense_end + 1)) + list(
        range(dense_end + sparse_step, max_layer + 1, sparse_step)
    )


def _Z(i):
    return (tc.gates.z(), [i])


def _X(i):
    return (tc.gates.x(), [i])


def hamiltonian_terms(h: float):
    zz_edges = ((0, 1), (0, 2), (1, 3), (2, 3))

    terms = [(-(1.0 - h), tuple(_Z(i) for i in edge)) for edge in zz_edges]
    terms.append((-(1.0 - h), tuple(_X(i) for i in range(4))))
    terms.extend((-h, (_Z(i),)) for i in range(4))

    return terms


PAULI = {
    "I": tc.gates.i().tensor,
    "X": tc.gates.x().tensor,
    "Y": tc.gates.y().tensor,
    "Z": tc.gates.z().tensor,
}


def local_term_to_matrix(local_ops, n_qubits):
    mats = [PAULI["I"]] * n_qubits

    for gate, wires in local_ops:
        for qubit in wires:
            mats[qubit] = gate.tensor

    return reduce(jnp.kron, mats)


def build_H_matrix_jax(H_terms, n_qubits):
    dim = 2**n_qubits
    H = jnp.zeros((dim, dim), dtype=COMPLEX_DTYPE)

    for coef, local_ops in H_terms:
        H = H + jnp.asarray(coef, dtype=REAL_DTYPE) * local_term_to_matrix(
            local_ops,
            n_qubits,
        )

    return H


def _kron_all(mats):
    return reduce(jnp.kron, mats)


def ket0_density(dtype=COMPLEX_DTYPE):
    v = jnp.array([1.0, 0.0], dtype=dtype)
    return jnp.outer(v, jnp.conjugate(v))


def rho_zero_state(k: int, dtype=COMPLEX_DTYPE) -> jnp.ndarray:
    return _kron_all([ket0_density(dtype=dtype) for _ in range(k)])


def qg_layer(circuit, q0: int, q1: int, params: jnp.ndarray) -> None:
    """Apply the shared Rz-Rz-Rxx three-parameter circuit block."""
    if int(np.shape(params)[0]) != 3:
        raise ValueError(
            "qg_layer expects exactly three parameters: Rz, Rz, and Rxx."
        )
    circuit.rz(int(q0), theta=params[0])
    circuit.rz(int(q1), theta=params[1])
    circuit.rxx(int(q0), int(q1), theta=params[2])


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


def _jax_to_np(value, dtype=None) -> np.ndarray:
    array = np.asarray(jax.device_get(value))
    if dtype is not None:
        array = array.astype(dtype)
    return array


def _normalize_index_list(indices, n: int):
    """Normalize optional positive/negative indices while preserving order."""
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative.")
    if indices is None:
        return list(range(n))

    normalized = []
    for index in indices:
        index = int(index)
        if index < 0:
            index += n
        normalized.append(index)

    if any(index < 0 or index >= n for index in normalized):
        raise IndexError("Index out of range.")
    return normalized


def qfim_grad_alignment_at_point(
    theta,
    grad,
    qfim_fn,
    *,
    sort_desc: bool = True,
    norm_eps: float = 1e-24,
) -> dict:
    """Compute QFIM eigenvalues and normalized gradient-direction weights."""
    theta = jnp.asarray(theta, dtype=REAL_DTYPE)
    grad = jnp.asarray(grad, dtype=REAL_DTYPE).reshape((-1,))
    qfim_matrix = jnp.asarray(qfim_fn(theta))
    if qfim_matrix.ndim != 2 or qfim_matrix.shape[0] != qfim_matrix.shape[1]:
        raise ValueError(f"QFIM must be square, got shape={qfim_matrix.shape}.")
    if int(qfim_matrix.shape[0]) != int(grad.shape[0]):
        raise ValueError(
            "Dimension mismatch: "
            f"QFIM shape={qfim_matrix.shape}, grad shape={grad.shape}."
        )

    qfim_matrix = 0.5 * (qfim_matrix + jnp.conjugate(qfim_matrix.T))
    evals, evecs = jnp.linalg.eigh(qfim_matrix)
    evals = jnp.clip(jnp.real(evals), a_min=0.0)

    coeffs = jnp.conjugate(evecs).T @ grad.astype(evecs.dtype)
    coeff_abs2 = jnp.real(coeffs * jnp.conjugate(coeffs))
    denominator = jnp.real(jnp.vdot(grad, grad))
    norm_eps_value = jnp.asarray(norm_eps, dtype=denominator.dtype)
    weights = jnp.where(
        denominator > norm_eps_value,
        coeff_abs2 / denominator,
        jnp.full_like(coeff_abs2, jnp.nan),
    )

    if sort_desc:
        order = jnp.argsort(evals)[::-1]
        evals = evals[order]
        weights = weights[order]
        coeffs = coeffs[order]
        coeff_abs2 = coeff_abs2[order]

    evals_np = _jax_to_np(evals, dtype=NP_REAL_DTYPE)
    return {
        "evals": evals_np,
        "weights": _jax_to_np(weights, dtype=NP_REAL_DTYPE),
        "coeffs": _jax_to_np(coeffs),
        "coeff_abs2": _jax_to_np(coeff_abs2, dtype=NP_REAL_DTYPE),
        "eig_index": np.arange(1, evals_np.size + 1, dtype=NP_INT_DTYPE),
        "grad_weight_denominator": NP_REAL_DTYPE(
            jax.device_get(denominator)
        ),
    }


def qfim_grad_alignment_one_to_table(
    alignment,
    *,
    layer=None,
    run=None,
    time_index=None,
    iteration=None,
) -> dict:
    """Convert one-point QFIM-gradient alignment output to table arrays."""
    evals = np.asarray(alignment["evals"], dtype=NP_REAL_DTYPE)
    n = int(evals.size)
    layer_value = -1 if layer is None else int(layer)
    run_value = -1 if run is None else int(run)
    time_value = -1 if time_index is None else int(time_index)
    iteration_value = -1 if iteration is None else int(iteration)
    return {
        "lambda": evals,
        "w_grad": np.asarray(alignment["weights"], dtype=NP_REAL_DTYPE),
        "coeff_abs2": np.asarray(
            alignment["coeff_abs2"],
            dtype=NP_REAL_DTYPE,
        ),
        "eig_index": np.asarray(alignment["eig_index"], dtype=NP_INT_DTYPE),
        "layer": np.full(n, layer_value, dtype=NP_INT_DTYPE),
        "run": np.full(n, run_value, dtype=NP_INT_DTYPE),
        "time_index": np.full(n, time_value, dtype=NP_INT_DTYPE),
        "iteration": np.full(n, iteration_value, dtype=NP_INT_DTYPE),
    }


def _time_index_from_iteration(
    sample_iters_for_labels,
    target_iteration: int,
) -> int:
    """Return the sampled-time index for an exact optimization iteration."""
    sample_iters = np.asarray(sample_iters_for_labels, dtype=NP_INT_DTYPE)
    if sample_iters.ndim != 1:
        raise ValueError("sample_iters_for_labels must be one-dimensional.")
    target_iteration = int(target_iteration)
    matches = np.flatnonzero(sample_iters == target_iteration)
    if matches.size == 0:
        raise ValueError(
            f"iteration {target_iteration} is not included in sample_iters. "
            "Add it to sample_iters before running the VQE optimization loop."
        )
    return int(matches[0])


def load_npz_result(inpath: str) -> dict:
    if not os.path.exists(inpath):
        raise FileNotFoundError(f"Required numerical result file is missing: {inpath}")

    with np.load(inpath, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


@jax.jit
def threshold_psd_eigvals_for_rank(
    evals: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Count QFIM eigenvalues strictly larger than the fixed rank threshold."""
    thresh = jnp.array(
        cfg.QFIM_EFFECTIVE_RANK_THRESHOLD,
        dtype=evals.dtype,
    )
    rank = jnp.sum(evals > thresh)
    return rank, thresh


def _thr_tag(thr: float) -> str:
    return f"{float(thr):.0e}".replace("+", "")
