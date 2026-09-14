"""Differentiable Rz/Rxx density updates without small matrix products.

Wire zero is the most significant computational-basis bit, matching the DPQC
reshape/transposition convention.  Callers configure JAX and enable float64
before using these functions.  Wire indices and ``num_qubits`` are static;
``theta`` remains a differentiable scalar.  Leading density batch dimensions
are supported, and independent angles can be handled with ``jax.vmap``.
"""

import operator

import jax.numpy as jnp


def _basis_size(rho, num_qubits):
    num_qubits = operator.index(num_qubits)
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive.")
    dimension = 1 << num_qubits
    if rho.shape[-2:] != (dimension, dimension):
        raise ValueError("rho shape must end in (2**num_qubits, 2**num_qubits).")
    return num_qubits, dimension


def _wire_mask(wire, num_qubits):
    wire = operator.index(wire)
    if not 0 <= wire < num_qubits:
        raise ValueError("wire must be in range(num_qubits).")
    return 1 << (num_qubits - 1 - wire)


def apply_rz_density(rho, theta, wire, num_qubits=5):
    """Apply exp(-i theta Z_wire / 2) by row/column basis phases."""
    rho = jnp.asarray(rho)
    num_qubits, dimension = _basis_size(rho, num_qubits)
    mask = _wire_mask(wire, num_qubits)
    bit = (jnp.arange(dimension) & mask) != 0
    z_eigenvalue = 1 - 2 * bit.astype(rho.real.dtype)
    theta = jnp.asarray(theta, dtype=rho.real.dtype)
    phase = jnp.exp(-0.5j * theta * z_eigenvalue)
    return rho * phase[:, None] * jnp.conjugate(phase)[None, :]


def apply_rxx_density(rho, theta, wires, num_qubits=5):
    """Apply exp(-i theta X_a X_b / 2) using XOR basis permutations.

    With P = X_a X_b and U = c I - i s P, the exact update is
    c**2 rho + s**2 P rho P + i c s (rho P - P rho).
    """
    rho = jnp.asarray(rho)
    num_qubits, dimension = _basis_size(rho, num_qubits)
    wires = tuple(wires)
    if len(wires) != 2 or wires[0] == wires[1]:
        raise ValueError("Rxx requires two distinct wires.")
    mask = _wire_mask(wires[0], num_qubits) | _wire_mask(wires[1], num_qubits)
    permutation = jnp.arange(dimension) ^ mask
    # Every XOR index is valid; clip avoids out-of-bounds fill machinery.
    p_rho = jnp.take(rho, permutation, axis=-2, mode="clip")
    rho_p = jnp.take(rho, permutation, axis=-1, mode="clip")
    p_rho_p = jnp.take(p_rho, permutation, axis=-1, mode="clip")
    theta = jnp.asarray(theta, dtype=rho.real.dtype)
    c, s = jnp.cos(0.5 * theta), jnp.sin(0.5 * theta)
    return c * c * rho + s * s * p_rho_p + 1j * c * s * (rho_p - p_rho)
