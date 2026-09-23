#!/usr/bin/env python
# coding: utf-8
"""Lightweight shared definition of the 60-parameter U3-Cartan circuit.

Each layer contains four independent 15-angle pair blocks and no additional
ancilla rotation, measurement, reset, or feed-forward operation. Importing this module does not
load a numerical backend or a circuit drawing library.
"""

from __future__ import annotations


ANSATZ_NAME = "unitary_pqc"
OUTPUT_VARIANT = "u3_cartan"
ANCILLA_QUBIT = 4
NUM_BLOCKS = 4
BLOCK_PARAMETER_NAMES = (
    "pre_Ry1_q0", "pre_Rz_q0", "pre_Ry2_q0",
    "pre_Ry1_q1", "pre_Rz_q1", "pre_Ry2_q1",
    "Rxx", "Ryy", "Rzz",
    "post_Ry1_q0", "post_Rz_q0", "post_Ry2_q0",
    "post_Ry1_q1", "post_Rz_q1", "post_Ry2_q1",
)
PARAMS_PER_BLOCK = len(BLOCK_PARAMETER_NAMES)
UNITARY_PARAMS_PER_LAYER = NUM_BLOCKS * PARAMS_PER_BLOCK
NUM_PARAMS_PER_LAYER = UNITARY_PARAMS_PER_LAYER
LAYER_PAIRS = (
    (1, 3),
    (2, 3),
    (0, 2),
    (0, ANCILLA_QUBIT),
)


def unitary_block_matrix(block, jnp):
    """Return the 15-angle pair unitary in first-wire/second-wire basis order.

    ``U3`` means RY(a), RZ(b), RY(c) in application order, matching reset
    DPQC. Both wires and both sides of Rxx/Ryy/Rzz use independent triples.
    The caller supplies the array module to keep imports lightweight.
    """
    if block.shape != (PARAMS_PER_BLOCK,):
        raise ValueError(f"A unitary block requires {PARAMS_PER_BLOCK} angles.")

    def ry(theta):
        c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
        return jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)

    def rz(theta):
        return jnp.diag(jnp.exp(jnp.array([-0.5j, 0.5j]) * theta))

    def u3(start):
        return ry(block[start + 2]) @ rz(block[start + 1]) @ ry(block[start])

    x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
    z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
    identity = jnp.eye(4, dtype=jnp.complex128)
    unitary = jnp.kron(u3(0), u3(3))
    for theta, pauli in zip(block[6:9], (x, y, z)):
        rotation = (
            jnp.cos(theta / 2) * identity
            - 1j * jnp.sin(theta / 2) * jnp.kron(pauli, pauli)
        )
        unitary = rotation @ unitary
    return jnp.kron(u3(9), u3(12)) @ unitary
