#!/usr/bin/env python
# coding: utf-8
"""Hamiltonian construction helpers for DPQC and Unitary-PQC analyses."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce

if __package__:
    from .dpqc_precision import COMPLEX_DTYPE, REAL_DTYPE, ensure_jax_x64
else:
    from dpqc_precision import COMPLEX_DTYPE, REAL_DTYPE, ensure_jax_x64

ensure_jax_x64()

import jax.numpy as jnp

PAULI = {
    "I": jnp.eye(2, dtype=COMPLEX_DTYPE),
    "X": jnp.asarray([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE),
    "Y": jnp.asarray([[0, -1j], [1j, 0]], dtype=COMPLEX_DTYPE),
    "Z": jnp.asarray([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE),
}


@dataclass(frozen=True)
class _DenseGate:
    """Minimal TensorCircuit-gate substitute for matrix-only workflows."""

    tensor: jnp.ndarray


H_RAW_PAULI_LABELS_4 = (
    "Z0Z1",
    "Z0Z2",
    "Z1Z3",
    "Z2Z3",
    "X0X1X2X3",
    "Z0",
    "Z1",
    "Z2",
    "Z3",
)


def z_term(wire: int):
    return (_DenseGate(PAULI["Z"]), [int(wire)])


def x_term(wire: int):
    return (_DenseGate(PAULI["X"]), [int(wire)])


def hamiltonian_terms(h: float):
    """Return the 4-system-qubit Hamiltonian terms used in the experiments."""
    zz_edges = ((0, 1), (0, 2), (1, 3), (2, 3))

    terms = [(-(1.0 - h), tuple(z_term(i) for i in edge)) for edge in zz_edges]
    terms.append((-(1.0 - h), tuple(x_term(i) for i in range(4))))
    terms.extend((-h, (z_term(i),)) for i in range(4))

    return terms


def local_term_to_matrix(local_ops, num_qubits: int):
    """Embed one local Pauli product into a num_qubits-qubit Hilbert space."""
    ensure_jax_x64()
    mats = [PAULI["I"]] * int(num_qubits)

    for gate, wires in local_ops:
        for qubit in wires:
            mats[int(qubit)] = gate.tensor

    return reduce(jnp.kron, mats)


def build_H_matrix_jax(H_terms, num_qubits: int):
    """Build a dense JAX matrix for a Hamiltonian term list."""
    ensure_jax_x64()
    num_qubits = int(num_qubits)
    dim = 2**num_qubits
    H = jnp.zeros((dim, dim), dtype=COMPLEX_DTYPE)

    for coef, local_ops in H_terms:
        H = H + jnp.asarray(coef, dtype=REAL_DTYPE) * local_term_to_matrix(
            local_ops,
            num_qubits,
        )

    return H


def hamiltonian_coefficient_vector(h: float) -> jnp.ndarray:
    """Return coefficients in the fixed raw-Pauli order used by the DPQC."""
    ensure_jax_x64()
    h = jnp.asarray(h, dtype=REAL_DTYPE)
    one_minus_h = -(1.0 - h)
    minus_h = -h
    return jnp.asarray(
        [one_minus_h] * 5 + [minus_h] * 4,
        dtype=REAL_DTYPE,
    )


def raw_hamiltonian_pauli_matrices_4() -> jnp.ndarray:
    """Return the nine unweighted Hamiltonian Pauli strings on four qubits."""
    ensure_jax_x64()
    # h changes only the coefficients, so any finite value gives the same
    # ordered local-operator list.
    return jnp.stack(
        [
            local_term_to_matrix(local_ops, num_qubits=4)
            for _, local_ops in hamiltonian_terms(0.5)
        ],
        axis=0,
    ).astype(COMPLEX_DTYPE)


def parity_z_matrix_4() -> jnp.ndarray:
    """Return Pi_Z = Z0 Z1 Z2 Z3 on the four retained system qubits."""
    ensure_jax_x64()
    return reduce(jnp.kron, [PAULI["Z"]] * 4).astype(COMPLEX_DTYPE)
