#!/usr/bin/env python
# coding: utf-8
"""Hamiltonian construction helpers for DPQC and Unitary-PQC analyses."""

from __future__ import annotations

from functools import reduce

import jax.numpy as jnp
import tensorcircuit as tc


REAL_DTYPE = jnp.float64
COMPLEX_DTYPE = jnp.complex128

PAULI = {
    "I": tc.gates.i().tensor,
    "X": tc.gates.x().tensor,
    "Y": tc.gates.y().tensor,
    "Z": tc.gates.z().tensor,
}


def z_term(wire: int):
    return (tc.gates.z(), [int(wire)])


def x_term(wire: int):
    return (tc.gates.x(), [int(wire)])


def hamiltonian_terms(h: float):
    """Return the 4-system-qubit Hamiltonian terms used in the experiments."""
    zz_edges = ((0, 1), (0, 2), (1, 3), (2, 3))

    terms = [(-(1.0 - h), tuple(z_term(i) for i in edge)) for edge in zz_edges]
    terms.append((-(1.0 - h), tuple(x_term(i) for i in range(4))))
    terms.extend((-h, (z_term(i),)) for i in range(4))

    return terms


def local_term_to_matrix(local_ops, num_qubits: int):
    """Embed one local Pauli product into a num_qubits-qubit Hilbert space."""
    mats = [PAULI["I"]] * int(num_qubits)

    for gate, wires in local_ops:
        for qubit in wires:
            mats[int(qubit)] = gate.tensor

    return reduce(jnp.kron, mats)


def build_H_matrix_jax(H_terms, num_qubits: int):
    """Build a dense JAX matrix for a Hamiltonian term list."""
    num_qubits = int(num_qubits)
    dim = 2**num_qubits
    H = jnp.zeros((dim, dim), dtype=COMPLEX_DTYPE)

    for coef, local_ops in H_terms:
        H = H + jnp.asarray(coef, dtype=REAL_DTYPE) * local_term_to_matrix(
            local_ops,
            num_qubits,
        )

    return H
