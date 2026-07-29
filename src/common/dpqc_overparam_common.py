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
