#!/usr/bin/env python
# coding: utf-8
"""Analyze energy Hessians at optimized DPQC endpoints.

This program is a post-processing stage for ``DPQC_overparam_vqe.py``.  It
loads ``vqe_optimization_histories.npz`` and does not rerun the optimizer.
Because that archive has no basin labels, endpoints at a fixed depth are
grouped into empirical basins using both final-energy proximity and the
Frobenius distance between their four-qubit reduced states.

By default, the endpoint with the smallest archived gradient norm in each
empirical basin is used as its representative.  Pass ``--all-endpoints`` to
compute a Hessian for every selected run instead.  The signed full Hessian
spectrum is retained: negative eigenvalues are never clipped.

The last two parameters of the final DPQC layer are exact objective-null
directions.  Consequently, the full Hessian always has at least two zero
modes (up to floating-point error) and cannot be positive definite.  The
requested full-space quantities are the primary output; diagnostics for the
matrix obtained by removing those two known directions are also saved under
the ``quotient_*`` names.

Examples::

    python src/dpqc/DPQC_overparam_hessian.py
    python src/dpqc/DPQC_overparam_hessian.py --layers 1,2,4
    python src/dpqc/DPQC_overparam_hessian.py --layers 1 --runs 0,1 --all-endpoints
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Iterable, Sequence


_MODULE_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _MODULE_DIR.parent / "common"
_common_dir_string = str(_COMMON_DIR)
if _common_dir_string not in sys.path:
    sys.path.insert(0, _common_dir_string)

import config_overparam as cfg


# Match the VQE scripts' deterministic, double-precision CPU default while
# still allowing a caller to select another JAX platform through the
# environment before starting this process.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)

REAL_DTYPE = jnp.float64
COMPLEX_DTYPE = jnp.complex128
NP_REAL_DTYPE = np.float64
NP_INT_DTYPE = np.int64

NUM_OBSERVED_QUBITS = 4
NUM_KEPT_QUBITS = 5
NUM_BLOCKS = 4
PARAMS_PER_BLOCK = 3
NUM_CHANNEL_PARAMS = 2
NUM_PARAMS_PER_LAYER = NUM_BLOCKS * PARAMS_PER_BLOCK + NUM_CHANNEL_PARAMS
STRUCTURAL_NULL_COUNT = NUM_CHANNEL_PARAMS

TOP, LEFT, RIGHT, BOTTOM, CENTRE = 0, 1, 2, 3, 4
LAYER_PAIRS = (
    (LEFT, BOTTOM),
    (RIGHT, BOTTOM),
    (TOP, RIGHT),
    (TOP, CENTRE),
)

DEFAULT_EIGENVALUE_EPSILON = 1e-8
DEFAULT_STATIONARITY_TOLERANCE = 1e-6
DEFAULT_HVP_CHUNK_SIZE = 8
# Fixed-step Adam endpoints have appreciably more spread than fully refined
# stationary points.  Both values remain explicit command-line parameters.
DEFAULT_BASIN_ENERGY_TOLERANCE = 1e-3
DEFAULT_BASIN_STATE_TOLERANCE = 5e-2
DEFAULT_ENERGY_CHECK_TOLERANCE = 1e-8

HESSIAN_METHOD = "chunked_forward_over_reverse_hvp"

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SpectrumMetrics:
    """Signed-eigenvalue summary at one fixed numerical threshold."""

    dimension: int
    epsilon: float
    negative_count: int
    zero_count: int
    positive_count: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    minimum_positive_eigenvalue: float
    positive_spectrum_condition_number: float
    condition_number_defined: bool
    zero_fraction: float
    eigenvalues: np.ndarray


@dataclass(frozen=True)
class Basin:
    """One deterministic empirical endpoint cluster at a fixed depth."""

    basin_id: int
    member_positions: tuple[int, ...]
    seed_position: int
    representative_position: int


@dataclass
class PointResult:
    """Hessian diagnostics for one analyzed endpoint."""

    layer: int
    basin_id: int
    basin_size: int
    run_index: int
    is_representative: bool
    archive_energy: float
    recomputed_energy: float
    energy_residual: float
    archived_gradient_norm: float
    gradient_norm: float
    stationary: bool
    epsilon: float
    structural_null_count: int
    excess_zero_count: int
    hessian_symmetry_residual: float
    structural_null_residual: float
    curvature_classification: str
    stationary_classification: str
    full: SpectrumMetrics
    quotient: SpectrumMetrics
    theta: np.ndarray
    hessian: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Side-effect-free DPQC endpoint and energy map
# ---------------------------------------------------------------------------


def _kron_all(matrices: Iterable[jnp.ndarray]) -> jnp.ndarray:
    return reduce(jnp.kron, matrices)


I2 = jnp.eye(2, dtype=COMPLEX_DTYPE)
X2 = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=COMPLEX_DTYPE)
Z2 = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=COMPLEX_DTYPE)
ZERO_RHO = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=COMPLEX_DTYPE)
INITIAL_STATE = _kron_all([ZERO_RHO] * NUM_KEPT_QUBITS)


def _rz(theta: jnp.ndarray) -> jnp.ndarray:
    theta = jnp.asarray(theta, dtype=REAL_DTYPE)
    return jnp.asarray(
        [
            [jnp.exp(-0.5j * theta), 0.0],
            [0.0, jnp.exp(0.5j * theta)],
        ],
        dtype=COMPLEX_DTYPE,
    )


def _rxx(theta: jnp.ndarray) -> jnp.ndarray:
    theta = jnp.asarray(theta, dtype=REAL_DTYPE)
    cosine = jnp.cos(0.5 * theta).astype(COMPLEX_DTYPE)
    sine = jnp.sin(0.5 * theta).astype(COMPLEX_DTYPE)
    return cosine * jnp.eye(4, dtype=COMPLEX_DTYPE) - 1j * sine * jnp.kron(
        X2, X2
    )


def _apply_unitary(
    rho: jnp.ndarray,
    unitary: jnp.ndarray,
    wires: Sequence[int],
    *,
    num_qubits: int = NUM_KEPT_QUBITS,
) -> jnp.ndarray:
    wires = tuple(int(wire) for wire in wires)
    num_targets = len(wires)
    if unitary.shape != (2**num_targets, 2**num_targets):
        raise ValueError("Unitary shape does not match the selected wires.")

    other_wires = [wire for wire in range(num_qubits) if wire not in wires]
    permutation = (
        list(wires)
        + other_wires
        + [wire + num_qubits for wire in wires]
        + [wire + num_qubits for wire in other_wires]
    )
    inverse_permutation = [0] * (2 * num_qubits)
    for position, axis in enumerate(permutation):
        inverse_permutation[axis] = position

    rho_permuted = jnp.transpose(
        jnp.reshape(rho, (2,) * (2 * num_qubits)),
        permutation,
    )
    target_dimension = 2**num_targets
    rest_dimension = 2 ** (num_qubits - num_targets)
    rho_permuted = jnp.reshape(
        rho_permuted,
        (target_dimension, rest_dimension, target_dimension, rest_dimension),
    )
    left_applied = jnp.einsum("ij,jrks->irks", unitary, rho_permuted)
    both_applied = jnp.einsum(
        "irps,bp->irbs",
        left_applied,
        jnp.conjugate(unitary),
    )
    both_applied = jnp.reshape(
        both_applied,
        (2,) * num_targets
        + (2,) * (num_qubits - num_targets)
        + (2,) * num_targets
        + (2,) * (num_qubits - num_targets),
    )
    return jnp.reshape(
        jnp.transpose(both_applied, inverse_permutation),
        (2**num_qubits, 2**num_qubits),
    )


def _apply_dynamic_delay(
    rho: jnp.ndarray,
    varphi: jnp.ndarray,
    phi: jnp.ndarray,
) -> jnp.ndarray:
    """Apply the VQE program's exact two-Kraus fresh-ancilla channel."""

    kept_dimension = 2**NUM_KEPT_QUBITS
    if rho.shape != (kept_dimension, kept_dimension):
        raise ValueError(
            f"Expected rho shape {(kept_dimension, kept_dimension)}, "
            f"got {rho.shape}."
        )

    rest_dimension = 2 ** (NUM_KEPT_QUBITS - 1)
    blocks = jnp.reshape(rho, (rest_dimension, 2, rest_dimension, 2))
    rho00 = blocks[:, 0, :, 0]
    rho11 = blocks[:, 1, :, 1]

    varphi = jnp.asarray(varphi, dtype=REAL_DTYPE)
    phi = jnp.asarray(phi, dtype=REAL_DTYPE)
    psi = jnp.stack(
        (
            -1j * jnp.sin(phi).astype(COMPLEX_DTYPE),
            jnp.exp(1j * varphi).astype(COMPLEX_DTYPE)
            * jnp.cos(phi).astype(COMPLEX_DTYPE),
        )
    )
    rho_psi = psi[:, None] * jnp.conjugate(psi[None, :])
    output = (
        jnp.einsum("rs,ab->rasb", rho00, ZERO_RHO)
        + jnp.einsum("rs,ab->rasb", rho11, rho_psi)
    )
    return jnp.reshape(output, (kept_dimension, kept_dimension))


def _hermitian(matrix: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (matrix + jnp.conjugate(matrix.T))


def _one_layer(rho: jnp.ndarray, layer_theta: jnp.ndarray) -> jnp.ndarray:
    blocks = jnp.reshape(
        layer_theta[:-NUM_CHANNEL_PARAMS],
        (NUM_BLOCKS, PARAMS_PER_BLOCK),
    )
    for (left_wire, right_wire), block in zip(LAYER_PAIRS, blocks):
        rho = _apply_unitary(rho, _rz(block[0]), (left_wire,))
        rho = _apply_unitary(rho, _rz(block[1]), (right_wire,))
        rho = _apply_unitary(rho, _rxx(block[2]), (left_wire, right_wire))
    return _apply_dynamic_delay(rho, layer_theta[-2], layer_theta[-1])


def full_state(theta: jnp.ndarray, layer: int) -> jnp.ndarray:
    """Return the five-retained-qubit endpoint state used by VQE."""

    layer = int(layer)
    if layer <= 0:
        raise ValueError("layer must be positive.")
    theta = jnp.asarray(theta, dtype=REAL_DTYPE)
    expected_parameters = layer * NUM_PARAMS_PER_LAYER
    if theta.shape != (expected_parameters,):
        raise ValueError(
            f"Expected theta shape {(expected_parameters,)}, got {theta.shape}."
        )
    layer_parameters = jnp.reshape(theta, (layer, NUM_PARAMS_PER_LAYER))

    def scan_layer(rho: jnp.ndarray, parameters: jnp.ndarray):
        return _one_layer(rho, parameters), None

    endpoint, _ = jax.lax.scan(scan_layer, INITIAL_STATE, layer_parameters)
    return _hermitian(endpoint)


def reduced_state(theta: jnp.ndarray, layer: int) -> jnp.ndarray:
    """Trace the retained centre qubit and return the observed state."""

    rho5 = full_state(theta, layer)
    observed_dimension = 2**NUM_OBSERVED_QUBITS
    tensor = jnp.reshape(rho5, (observed_dimension, 2, observed_dimension, 2))
    return _hermitian(jnp.trace(tensor, axis1=1, axis2=3))


def _pauli_string_matrix(operators: dict[int, jnp.ndarray]) -> jnp.ndarray:
    return _kron_all(
        [operators.get(wire, I2) for wire in range(NUM_OBSERVED_QUBITS)]
    )


HAMILTONIAN_PAULI_MATRICES = jnp.stack(
    (
        _pauli_string_matrix({0: Z2, 1: Z2}),
        _pauli_string_matrix({0: Z2, 2: Z2}),
        _pauli_string_matrix({1: Z2, 3: Z2}),
        _pauli_string_matrix({2: Z2, 3: Z2}),
        _pauli_string_matrix({0: X2, 1: X2, 2: X2, 3: X2}),
        _pauli_string_matrix({0: Z2}),
        _pauli_string_matrix({1: Z2}),
        _pauli_string_matrix({2: Z2}),
        _pauli_string_matrix({3: Z2}),
    ),
    axis=0,
)


def hamiltonian4(h_param: float) -> jnp.ndarray:
    h_value = jnp.asarray(h_param, dtype=REAL_DTYPE)
    coefficients = jnp.concatenate(
        (
            jnp.full((5,), -(1.0 - h_value), dtype=REAL_DTYPE),
            jnp.full((4,), -h_value, dtype=REAL_DTYPE),
        )
    )
    return jnp.einsum("a,aij->ij", coefficients, HAMILTONIAN_PAULI_MATRICES)


def make_energy_function(layer: int, h_param: float):
    """Construct the exact scalar objective for one fixed DPQC depth."""

    layer = int(layer)
    hamiltonian = hamiltonian4(float(h_param))

    def energy_function(theta: jnp.ndarray) -> jnp.ndarray:
        rho4 = reduced_state(theta, layer)
        return jnp.real(jnp.einsum("ij,ji->", hamiltonian, rho4))

    return energy_function


def make_hessian_vector_chunk_function(energy_function):
    """Compile fixed-size batches of exact Hessian-vector products.

    ``jax.hessian`` materializes all tangent directions together and needs a
    prohibitively large XLA work buffer for the deepest archived circuits.
    A forward-mode JVP of the reverse-mode gradient computes the same Hessian
    columns while letting the caller bound device memory with a small batch of
    tangent vectors.
    """

    gradient_function = jax.grad(energy_function)

    def hessian_vector_chunk(
        theta: jnp.ndarray,
        tangent_vectors: jnp.ndarray,
    ) -> jnp.ndarray:
        def one_hessian_vector_product(tangent_vector: jnp.ndarray):
            return jax.jvp(
                gradient_function,
                (theta,),
                (tangent_vector,),
            )[1]

        return jax.vmap(one_hessian_vector_product)(tangent_vectors)

    return jax.jit(hessian_vector_chunk)


def assemble_hessian_from_hvp_chunks(
    theta: jnp.ndarray,
    hessian_vector_chunk_function,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Assemble a dense Hessian from fixed-shape, bounded-memory HVP calls."""

    if (
        isinstance(chunk_size, (bool, np.bool_))
        or not isinstance(chunk_size, (int, np.integer))
        or int(chunk_size) <= 0
    ):
        raise ValueError("chunk_size must be a positive integer.")
    chunk_size = int(chunk_size)

    theta = jnp.asarray(theta, dtype=REAL_DTYPE)
    if theta.ndim != 1 or int(theta.size) == 0:
        raise ValueError("theta must be a nonempty one-dimensional array.")
    dimension = int(theta.size)
    hessian = np.empty((dimension, dimension), dtype=NP_REAL_DTYPE)

    # Every call has exactly the same tangent shape.  The final partial chunk
    # is zero-padded so JAX reuses one compiled executable for the whole layer.
    tangent_chunk = np.zeros(
        (chunk_size, dimension),
        dtype=NP_REAL_DTYPE,
    )
    for start in range(0, dimension, chunk_size):
        stop = min(start + chunk_size, dimension)
        valid_count = stop - start
        tangent_chunk.fill(0.0)
        local_indices = np.arange(valid_count)
        tangent_chunk[local_indices, start + local_indices] = 1.0
        hessian_columns = np.asarray(
            jax.device_get(
                hessian_vector_chunk_function(
                    theta,
                    jnp.asarray(tangent_chunk, dtype=REAL_DTYPE),
                )
            ),
            dtype=NP_REAL_DTYPE,
        )
        expected_shape = (chunk_size, dimension)
        if hessian_columns.shape != expected_shape:
            raise AssertionError(
                "Unexpected Hessian-vector chunk shape: "
                f"{hessian_columns.shape} != {expected_shape}."
            )

        # Row i of the batched result is H @ e_i, hence it is column i of H.
        hessian[:, start:stop] = hessian_columns[:valid_count].T

    return hessian


# ---------------------------------------------------------------------------
# Spectrum, classification, and empirical-basin helpers
# ---------------------------------------------------------------------------


def summarize_spectrum(
    matrix: np.ndarray,
    epsilon: float,
) -> SpectrumMetrics:
    """Return the quantities requested in the Hessian analysis definition."""

    matrix = np.asarray(matrix, dtype=NP_REAL_DTYPE)
    epsilon = float(epsilon)
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be a finite positive number.")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError("The Hessian contains non-finite values.")

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric).astype(NP_REAL_DTYPE, copy=False)
    negative_count = int(np.count_nonzero(eigenvalues < -epsilon))
    zero_count = int(np.count_nonzero(np.abs(eigenvalues) <= epsilon))
    positive_count = int(np.count_nonzero(eigenvalues > epsilon))

    if eigenvalues.size:
        minimum = float(eigenvalues[0])
        maximum = float(eigenvalues[-1])
    else:
        minimum = math.nan
        maximum = math.nan

    positive_eigenvalues = eigenvalues[eigenvalues > epsilon]
    condition_defined = bool(positive_eigenvalues.size)
    if condition_defined:
        minimum_positive = float(positive_eigenvalues[0])
        condition_number = float(maximum / minimum_positive)
    else:
        minimum_positive = math.nan
        condition_number = math.nan

    dimension = int(eigenvalues.size)
    zero_fraction = float(zero_count / dimension) if dimension else math.nan
    return SpectrumMetrics(
        dimension=dimension,
        epsilon=epsilon,
        negative_count=negative_count,
        zero_count=zero_count,
        positive_count=positive_count,
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=maximum,
        minimum_positive_eigenvalue=minimum_positive,
        positive_spectrum_condition_number=condition_number,
        condition_number_defined=condition_defined,
        zero_fraction=zero_fraction,
        eigenvalues=eigenvalues,
    )


def _curvature_classification(metrics: SpectrumMetrics) -> str:
    if metrics.negative_count and metrics.positive_count:
        return "indefinite_negative_curvature"
    if metrics.negative_count:
        if metrics.zero_count:
            return "negative_semidefinite"
        return "negative_definite"
    if metrics.zero_count:
        if metrics.positive_count:
            return "positive_semidefinite_flat"
        return "numerically_zero"
    if metrics.positive_count == metrics.dimension:
        return "positive_definite"
    return "inconclusive"


def _stationary_classification(
    metrics: SpectrumMetrics,
    gradient_norm: float,
    stationarity_tolerance: float,
) -> tuple[bool, str]:
    stationary = bool(
        math.isfinite(gradient_norm)
        and gradient_norm <= float(stationarity_tolerance)
    )
    if not stationary:
        return False, "nonstationary"
    if metrics.negative_count and metrics.positive_count:
        return True, "saddle"
    if metrics.negative_count:
        return True, "local_maximum_candidate"
    if (
        metrics.zero_count == 0
        and metrics.positive_count == metrics.dimension
    ):
        return True, "strict_local_minimum"
    if metrics.zero_count:
        return True, "flat_stationary_minimum_candidate"
    return True, "inconclusive_stationary_point"


def assign_empirical_basins(
    final_energies: np.ndarray,
    reduced_states: np.ndarray | None,
    run_indices: np.ndarray,
    archived_gradient_norms: np.ndarray,
    *,
    energy_tolerance: float,
    state_tolerance: float | None,
) -> tuple[np.ndarray, list[Basin], np.ndarray, np.ndarray]:
    """Cluster endpoints deterministically by energy and physical state.

    This is an endpoint clustering rule, not a proof that two starts belong to
    the same dynamical attraction basin.  Basin seeds and IDs are ordered by
    increasing final energy and then by run index.
    """

    final_energies = np.asarray(final_energies, dtype=NP_REAL_DTYPE)
    run_indices = np.asarray(run_indices, dtype=NP_INT_DTYPE)
    archived_gradient_norms = np.asarray(
        archived_gradient_norms, dtype=NP_REAL_DTYPE
    )
    count = int(final_energies.size)
    if final_energies.shape != (count,):
        raise ValueError("final_energies must be one-dimensional.")
    if run_indices.shape != (count,):
        raise ValueError("run_indices must align with final_energies.")
    if archived_gradient_norms.shape != (count,):
        raise ValueError("archived_gradient_norms must align with final_energies.")
    if count == 0:
        raise ValueError("At least one endpoint is required.")
    if not np.all(np.isfinite(final_energies)):
        raise FloatingPointError("Final energies contain non-finite values.")
    if energy_tolerance < 0.0 or not math.isfinite(energy_tolerance):
        raise ValueError("energy_tolerance must be finite and nonnegative.")
    if state_tolerance is not None:
        if state_tolerance < 0.0 or not math.isfinite(state_tolerance):
            raise ValueError("state_tolerance must be finite and nonnegative.")
        reduced_states = np.asarray(reduced_states)
        if reduced_states.shape[0] != count or reduced_states.ndim != 3:
            raise ValueError("reduced_states must have shape (runs, dim, dim).")
    else:
        reduced_states = None

    basin_ids = np.full(count, -1, dtype=NP_INT_DTYPE)
    seed_positions: list[int] = []
    order = np.lexsort((run_indices, final_energies))
    for position in order.tolist():
        assigned_id = None
        for candidate_id, seed_position in enumerate(seed_positions):
            energy_close = (
                abs(final_energies[position] - final_energies[seed_position])
                <= energy_tolerance
            )
            if not energy_close:
                continue
            if reduced_states is not None:
                state_distance = np.linalg.norm(
                    reduced_states[position] - reduced_states[seed_position],
                    ord="fro",
                )
                if state_distance > float(state_tolerance):
                    continue
            assigned_id = candidate_id
            break
        if assigned_id is None:
            assigned_id = len(seed_positions)
            seed_positions.append(position)
        basin_ids[position] = assigned_id

    seed_energy_distances = np.empty(count, dtype=NP_REAL_DTYPE)
    seed_state_distances = np.full(count, np.nan, dtype=NP_REAL_DTYPE)
    basins: list[Basin] = []
    for basin_id, seed_position in enumerate(seed_positions):
        member_positions = tuple(
            int(position)
            for position in np.flatnonzero(basin_ids == basin_id).tolist()
        )
        for position in member_positions:
            seed_energy_distances[position] = abs(
                final_energies[position] - final_energies[seed_position]
            )
            if reduced_states is not None:
                seed_state_distances[position] = np.linalg.norm(
                    reduced_states[position] - reduced_states[seed_position],
                    ord="fro",
                )

        def representative_key(position: int):
            gradient_norm = archived_gradient_norms[position]
            gradient_score = (
                float(gradient_norm) if math.isfinite(gradient_norm) else math.inf
            )
            return (
                gradient_score,
                float(final_energies[position]),
                int(run_indices[position]),
            )

        representative_position = min(member_positions, key=representative_key)
        basins.append(
            Basin(
                basin_id=basin_id,
                member_positions=member_positions,
                seed_position=int(seed_position),
                representative_position=int(representative_position),
            )
        )

    return basin_ids, basins, seed_energy_distances, seed_state_distances


# ---------------------------------------------------------------------------
# Archive validation and output helpers
# ---------------------------------------------------------------------------


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _parse_integer_list(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(token.strip()) for token in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integers"
        ) from exc
    if not parsed or any(token.strip() == "" for token in value.split(",")):
        raise argparse.ArgumentTypeError(
            "expected a nonempty comma-separated list of integers"
        )
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("indices must not be duplicated")
    return parsed


def _h_tag(h_param: float) -> str:
    return f"{float(h_param):.12g}"


def _default_archive_path(h_param: float) -> Path:
    # The compute programs use Python's float string (for example ``1.0``),
    # while older outputs sometimes used the compact ``.12g`` form (``1``).
    # Accept both layouts and prefer the compute program's current spelling.
    tags = list(dict.fromkeys((str(float(h_param)), _h_tag(h_param))))
    candidates = [
        Path.cwd()
        / "figs"
        / "dpqc"
        / f"h_{tag}"
        / "numerical_results"
        / "energy"
        / "vqe_optimization_histories.npz"
        for tag in tags
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def _available_layers(archive: np.lib.npyio.NpzFile) -> list[int]:
    if "vqe_layers" in archive.files:
        layers = [
            int(value)
            for value in np.asarray(archive["vqe_layers"], dtype=NP_INT_DTYPE)
        ]
    else:
        pattern = re.compile(r"^L([1-9][0-9]*)_theta_final$")
        layers = sorted(
            int(match.group(1))
            for key in archive.files
            if (match := pattern.match(key)) is not None
        )
    layers = list(dict.fromkeys(layers))
    if not layers:
        raise ValueError("The VQE archive contains no theta_final layer arrays.")
    return layers


def _select_layers(
    requested_layers: Sequence[int] | None,
    available_layers: Sequence[int],
) -> list[int]:
    available = [int(layer) for layer in available_layers]
    if requested_layers is None:
        return available
    requested = [int(layer) for layer in requested_layers]
    if any(layer <= 0 for layer in requested):
        raise ValueError("Layer numbers must be positive.")
    missing = [layer for layer in requested if layer not in available]
    if missing:
        raise ValueError(
            f"Requested layers are absent from the VQE archive: {missing}. "
            f"Available layers: {available}."
        )
    return requested


def _select_runs(
    requested_runs: Sequence[int] | None,
    run_count: int,
) -> np.ndarray:
    if requested_runs is None:
        return np.arange(run_count, dtype=NP_INT_DTYPE)
    normalized: list[int] = []
    for requested in requested_runs:
        index = int(requested)
        if index < 0:
            index += run_count
        if index < 0 or index >= run_count:
            raise IndexError(
                f"Run index {requested} is outside the valid range "
                f"[-{run_count}, {run_count - 1}]."
            )
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Normalized run indices must not be duplicated.")
    return np.asarray(normalized, dtype=NP_INT_DTYPE)


def _load_layer_inputs(
    archive: np.lib.npyio.NpzFile,
    layer: int,
    requested_runs: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta_key = f"L{layer}_theta_final"
    energy_key = f"L{layer}_energy_traces"
    gradient_key = f"L{layer}_grad_norm_traces"
    missing = [
        key for key in (theta_key, energy_key) if key not in archive.files
    ]
    if missing:
        raise KeyError(f"Missing required VQE archive arrays: {missing}.")

    all_theta = np.asarray(archive[theta_key], dtype=NP_REAL_DTYPE)
    energy_traces = np.asarray(archive[energy_key], dtype=NP_REAL_DTYPE)
    expected_parameters = NUM_PARAMS_PER_LAYER * int(layer)
    if all_theta.ndim != 2 or all_theta.shape[1] != expected_parameters:
        raise ValueError(
            f"{theta_key} must have shape (runs, {expected_parameters}); "
            f"got {all_theta.shape}."
        )
    if energy_traces.ndim != 2 or energy_traces.shape[0] != all_theta.shape[0]:
        raise ValueError(
            f"{energy_key} must have shape (runs, iterations) and align with "
            f"{theta_key}; got {energy_traces.shape}."
        )
    if energy_traces.shape[1] == 0:
        raise ValueError(f"{energy_key} has no optimization iterations.")

    if gradient_key in archive.files:
        gradient_traces = np.asarray(
            archive[gradient_key], dtype=NP_REAL_DTYPE
        )
        if gradient_traces.shape != energy_traces.shape:
            raise ValueError(
                f"{gradient_key} shape {gradient_traces.shape} does not match "
                f"{energy_key} shape {energy_traces.shape}."
            )
        final_gradient_norms = gradient_traces[:, -1]
    else:
        final_gradient_norms = np.full(
            all_theta.shape[0], np.nan, dtype=NP_REAL_DTYPE
        )

    run_indices = _select_runs(requested_runs, int(all_theta.shape[0]))
    theta = all_theta[run_indices]
    final_energies = energy_traces[run_indices, -1]
    final_gradient_norms = final_gradient_norms[run_indices]
    if not np.all(np.isfinite(theta)):
        raise FloatingPointError(f"{theta_key} contains non-finite values.")
    if not np.all(np.isfinite(final_energies)):
        raise FloatingPointError(f"{energy_key} contains non-finite final values.")
    return run_indices, theta, final_energies, final_gradient_norms


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fieldnames = list(rows[0].keys())
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _safe_float(value: float) -> float | str:
    value = float(value)
    return value if math.isfinite(value) else "nan"


def _point_to_csv_row(point: PointResult) -> dict:
    full = point.full
    quotient = point.quotient
    return {
        "layer": point.layer,
        "basin_id": point.basin_id,
        "basin_size": point.basin_size,
        "run_index": point.run_index,
        "is_basin_representative": int(point.is_representative),
        "parameter_count": full.dimension,
        "archive_final_energy": point.archive_energy,
        "recomputed_energy": point.recomputed_energy,
        "energy_residual": point.energy_residual,
        "archive_final_gradient_norm": _safe_float(
            point.archived_gradient_norm
        ),
        "recomputed_gradient_norm": point.gradient_norm,
        "stationarity_tolerance": "",
        "stationary": int(point.stationary),
        "epsilon": point.epsilon,
        "negative_eigenvalue_count": full.negative_count,
        "zero_eigenvalue_count": full.zero_count,
        "positive_eigenvalue_count": full.positive_count,
        "minimum_eigenvalue": full.minimum_eigenvalue,
        "maximum_eigenvalue": full.maximum_eigenvalue,
        "minimum_positive_eigenvalue": _safe_float(
            full.minimum_positive_eigenvalue
        ),
        "positive_spectrum_condition_number": _safe_float(
            full.positive_spectrum_condition_number
        ),
        "condition_number_defined": int(full.condition_number_defined),
        "zero_fraction": full.zero_fraction,
        "structural_null_count": point.structural_null_count,
        "excess_zero_eigenvalue_count": point.excess_zero_count,
        "hessian_symmetry_residual": point.hessian_symmetry_residual,
        "structural_null_residual": point.structural_null_residual,
        "curvature_classification": point.curvature_classification,
        "classification": point.stationary_classification,
        "quotient_dimension": quotient.dimension,
        "quotient_negative_eigenvalue_count": quotient.negative_count,
        "quotient_zero_eigenvalue_count": quotient.zero_count,
        "quotient_positive_eigenvalue_count": quotient.positive_count,
        "quotient_minimum_eigenvalue": quotient.minimum_eigenvalue,
        "quotient_maximum_eigenvalue": quotient.maximum_eigenvalue,
        "quotient_minimum_positive_eigenvalue": _safe_float(
            quotient.minimum_positive_eigenvalue
        ),
        "quotient_positive_spectrum_condition_number": _safe_float(
            quotient.positive_spectrum_condition_number
        ),
    }


def _finite_statistics(values: Sequence[float]) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=NP_REAL_DTYPE)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return math.nan, math.nan, math.nan
    return float(np.min(finite)), float(np.median(finite)), float(np.max(finite))


def _basin_summary_rows(
    assignments: list[dict],
    points: list[PointResult],
) -> list[dict]:
    rows: list[dict] = []
    point_groups: dict[tuple[int, int], list[PointResult]] = {}
    assignment_groups: dict[tuple[int, int], list[dict]] = {}
    for point in points:
        point_groups.setdefault((point.layer, point.basin_id), []).append(point)
    for assignment in assignments:
        key = (int(assignment["layer"]), int(assignment["basin_id"]))
        assignment_groups.setdefault(key, []).append(assignment)

    for key in sorted(assignment_groups):
        layer, basin_id = key
        members = assignment_groups[key]
        analyzed = point_groups[key]
        representative = next(point for point in analyzed if point.is_representative)
        energies = np.asarray(
            [member["archive_final_energy"] for member in members],
            dtype=NP_REAL_DTYPE,
        )
        negative_stats = _finite_statistics(
            [point.full.negative_count for point in analyzed]
        )
        zero_stats = _finite_statistics(
            [point.full.zero_count for point in analyzed]
        )
        minimum_stats = _finite_statistics(
            [point.full.minimum_eigenvalue for point in analyzed]
        )
        condition_stats = _finite_statistics(
            [point.full.positive_spectrum_condition_number for point in analyzed]
        )
        classification_counts: dict[str, int] = {}
        for point in analyzed:
            classification_counts[point.stationary_classification] = (
                classification_counts.get(point.stationary_classification, 0) + 1
            )

        rows.append(
            {
                "layer": layer,
                "basin_id": basin_id,
                "member_count": len(members),
                "member_run_indices": ";".join(
                    str(member["run_index"]) for member in members
                ),
                "representative_run_index": representative.run_index,
                "energy_min": float(np.min(energies)),
                "energy_mean": float(np.mean(energies)),
                "energy_std": float(np.std(energies)),
                "energy_max": float(np.max(energies)),
                "analyzed_endpoint_count": len(analyzed),
                "representative_gradient_norm": representative.gradient_norm,
                "representative_negative_eigenvalue_count": (
                    representative.full.negative_count
                ),
                "representative_zero_eigenvalue_count": (
                    representative.full.zero_count
                ),
                "representative_minimum_eigenvalue": (
                    representative.full.minimum_eigenvalue
                ),
                "representative_minimum_positive_eigenvalue": _safe_float(
                    representative.full.minimum_positive_eigenvalue
                ),
                "representative_condition_number": _safe_float(
                    representative.full.positive_spectrum_condition_number
                ),
                "representative_classification": (
                    representative.stationary_classification
                ),
                "negative_count_min": negative_stats[0],
                "negative_count_median": negative_stats[1],
                "negative_count_max": negative_stats[2],
                "zero_count_min": zero_stats[0],
                "zero_count_median": zero_stats[1],
                "zero_count_max": zero_stats[2],
                "minimum_eigenvalue_min": minimum_stats[0],
                "minimum_eigenvalue_median": minimum_stats[1],
                "minimum_eigenvalue_max": minimum_stats[2],
                "condition_number_min": _safe_float(condition_stats[0]),
                "condition_number_median": _safe_float(condition_stats[1]),
                "condition_number_max": _safe_float(condition_stats[2]),
                "classification_counts": json.dumps(
                    classification_counts,
                    ensure_ascii=True,
                    sort_keys=True,
                ),
            }
        )
    return rows


def _layer_result_arrays(
    *,
    layer: int,
    h_param: float,
    run_indices: np.ndarray,
    theta: np.ndarray,
    final_energies: np.ndarray,
    archived_gradient_norms: np.ndarray,
    basin_ids: np.ndarray,
    basins: Sequence[Basin],
    points: Sequence[PointResult],
    epsilon: float,
    stationarity_tolerance: float,
    basin_energy_tolerance: float,
    basin_state_tolerance: float | None,
    hvp_chunk_size: int,
    all_endpoints: bool,
    save_hessians: bool,
) -> dict:
    arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=NP_INT_DTYPE),
        "h_param": np.asarray(h_param, dtype=NP_REAL_DTYPE),
        "layer": np.asarray(layer, dtype=NP_INT_DTYPE),
        "num_params_per_layer": np.asarray(
            NUM_PARAMS_PER_LAYER, dtype=NP_INT_DTYPE
        ),
        "epsilon": np.asarray(epsilon, dtype=NP_REAL_DTYPE),
        "stationarity_tolerance": np.asarray(
            stationarity_tolerance, dtype=NP_REAL_DTYPE
        ),
        "basin_energy_tolerance": np.asarray(
            basin_energy_tolerance, dtype=NP_REAL_DTYPE
        ),
        "basin_state_tolerance": np.asarray(
            math.nan if basin_state_tolerance is None else basin_state_tolerance,
            dtype=NP_REAL_DTYPE,
        ),
        "basin_definition": np.asarray(
            "empirical_endpoint_cluster: final energy and reduced-state "
            "Frobenius distance"
            if basin_state_tolerance is not None
            else "empirical_endpoint_cluster: final energy only"
        ),
        "structural_null_count": np.asarray(
            STRUCTURAL_NULL_COUNT, dtype=NP_INT_DTYPE
        ),
        "hessian_method": np.asarray(HESSIAN_METHOD),
        "hvp_chunk_size": np.asarray(
            hvp_chunk_size, dtype=NP_INT_DTYPE
        ),
        "analysis_mode": np.asarray(
            "all_selected_endpoints"
            if all_endpoints
            else "one_representative_per_empirical_basin"
        ),
        "selected_run_indices": np.asarray(run_indices, dtype=NP_INT_DTYPE),
        "selected_theta_final": np.asarray(theta, dtype=NP_REAL_DTYPE),
        "selected_final_energies": np.asarray(
            final_energies, dtype=NP_REAL_DTYPE
        ),
        "selected_archived_gradient_norms": np.asarray(
            archived_gradient_norms, dtype=NP_REAL_DTYPE
        ),
        "selected_basin_ids": np.asarray(basin_ids, dtype=NP_INT_DTYPE),
        "basin_seed_run_indices": np.asarray(
            [run_indices[basin.seed_position] for basin in basins],
            dtype=NP_INT_DTYPE,
        ),
        "basin_representative_run_indices": np.asarray(
            [run_indices[basin.representative_position] for basin in basins],
            dtype=NP_INT_DTYPE,
        ),
        "basin_member_counts": np.asarray(
            [len(basin.member_positions) for basin in basins],
            dtype=NP_INT_DTYPE,
        ),
        "analyzed_run_indices": np.asarray(
            [point.run_index for point in points], dtype=NP_INT_DTYPE
        ),
        "analyzed_basin_ids": np.asarray(
            [point.basin_id for point in points], dtype=NP_INT_DTYPE
        ),
        "analyzed_is_representative": np.asarray(
            [point.is_representative for point in points], dtype=np.bool_
        ),
        "analyzed_theta": np.stack([point.theta for point in points], axis=0),
        "recomputed_energies": np.asarray(
            [point.recomputed_energy for point in points], dtype=NP_REAL_DTYPE
        ),
        "recomputed_gradient_norms": np.asarray(
            [point.gradient_norm for point in points], dtype=NP_REAL_DTYPE
        ),
        "stationary": np.asarray(
            [point.stationary for point in points], dtype=np.bool_
        ),
        "negative_eigenvalue_counts": np.asarray(
            [point.full.negative_count for point in points], dtype=NP_INT_DTYPE
        ),
        "zero_eigenvalue_counts": np.asarray(
            [point.full.zero_count for point in points], dtype=NP_INT_DTYPE
        ),
        "positive_eigenvalue_counts": np.asarray(
            [point.full.positive_count for point in points], dtype=NP_INT_DTYPE
        ),
        "minimum_eigenvalues": np.asarray(
            [point.full.minimum_eigenvalue for point in points],
            dtype=NP_REAL_DTYPE,
        ),
        "maximum_eigenvalues": np.asarray(
            [point.full.maximum_eigenvalue for point in points],
            dtype=NP_REAL_DTYPE,
        ),
        "minimum_positive_eigenvalues": np.asarray(
            [point.full.minimum_positive_eigenvalue for point in points],
            dtype=NP_REAL_DTYPE,
        ),
        "positive_spectrum_condition_numbers": np.asarray(
            [
                point.full.positive_spectrum_condition_number
                for point in points
            ],
            dtype=NP_REAL_DTYPE,
        ),
        "condition_number_defined": np.asarray(
            [point.full.condition_number_defined for point in points],
            dtype=np.bool_,
        ),
        "zero_fractions": np.asarray(
            [point.full.zero_fraction for point in points], dtype=NP_REAL_DTYPE
        ),
        "excess_zero_eigenvalue_counts": np.asarray(
            [point.excess_zero_count for point in points], dtype=NP_INT_DTYPE
        ),
        "hessian_eigenvalues": np.stack(
            [point.full.eigenvalues for point in points], axis=0
        ),
        "hessian_symmetry_residuals": np.asarray(
            [point.hessian_symmetry_residual for point in points],
            dtype=NP_REAL_DTYPE,
        ),
        "structural_null_residuals": np.asarray(
            [point.structural_null_residual for point in points],
            dtype=NP_REAL_DTYPE,
        ),
        "curvature_classifications": np.asarray(
            [point.curvature_classification for point in points]
        ),
        "classifications": np.asarray(
            [point.stationary_classification for point in points]
        ),
        "quotient_negative_eigenvalue_counts": np.asarray(
            [point.quotient.negative_count for point in points],
            dtype=NP_INT_DTYPE,
        ),
        "quotient_zero_eigenvalue_counts": np.asarray(
            [point.quotient.zero_count for point in points], dtype=NP_INT_DTYPE
        ),
        "quotient_positive_eigenvalue_counts": np.asarray(
            [point.quotient.positive_count for point in points],
            dtype=NP_INT_DTYPE,
        ),
        "quotient_minimum_eigenvalues": np.asarray(
            [point.quotient.minimum_eigenvalue for point in points],
            dtype=NP_REAL_DTYPE,
        ),
        "quotient_minimum_positive_eigenvalues": np.asarray(
            [point.quotient.minimum_positive_eigenvalue for point in points],
            dtype=NP_REAL_DTYPE,
        ),
        "quotient_positive_spectrum_condition_numbers": np.asarray(
            [
                point.quotient.positive_spectrum_condition_number
                for point in points
            ],
            dtype=NP_REAL_DTYPE,
        ),
        "quotient_hessian_eigenvalues": np.stack(
            [point.quotient.eigenvalues for point in points], axis=0
        ),
    }
    if save_hessians:
        arrays["hessians"] = np.stack(
            [np.asarray(point.hessian, dtype=NP_REAL_DTYPE) for point in points],
            axis=0,
        )
    return arrays


# ---------------------------------------------------------------------------
# Main analysis workflow
# ---------------------------------------------------------------------------


def run_hessian_analysis(
    *,
    input_path: Path,
    output_dir: Path | None = None,
    h_param: float | None = None,
    layers: Sequence[int] | None = None,
    runs: Sequence[int] | None = None,
    epsilon: float = DEFAULT_EIGENVALUE_EPSILON,
    stationarity_tolerance: float = DEFAULT_STATIONARITY_TOLERANCE,
    basin_energy_tolerance: float = DEFAULT_BASIN_ENERGY_TOLERANCE,
    basin_state_tolerance: float | None = DEFAULT_BASIN_STATE_TOLERANCE,
    all_endpoints: bool = False,
    save_hessians: bool = False,
    energy_check_tolerance: float = DEFAULT_ENERGY_CHECK_TOLERANCE,
    hvp_chunk_size: int = DEFAULT_HVP_CHUNK_SIZE,
) -> dict[str, Path]:
    """Run the saved-endpoint Hessian analysis and return generated paths."""

    epsilon = float(epsilon)
    stationarity_tolerance = float(stationarity_tolerance)
    basin_energy_tolerance = float(basin_energy_tolerance)
    energy_check_tolerance = float(energy_check_tolerance)
    if (
        isinstance(hvp_chunk_size, (bool, np.bool_))
        or not isinstance(hvp_chunk_size, (int, np.integer))
        or int(hvp_chunk_size) <= 0
    ):
        raise ValueError("hvp_chunk_size must be a positive integer.")
    hvp_chunk_size = int(hvp_chunk_size)
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive.")
    if stationarity_tolerance <= 0.0 or not math.isfinite(
        stationarity_tolerance
    ):
        raise ValueError("stationarity_tolerance must be finite and positive.")
    if basin_energy_tolerance < 0.0 or not math.isfinite(
        basin_energy_tolerance
    ):
        raise ValueError("basin_energy_tolerance must be finite and nonnegative.")
    if basin_state_tolerance is not None:
        basin_state_tolerance = float(basin_state_tolerance)
        if basin_state_tolerance < 0.0 or not math.isfinite(
            basin_state_tolerance
        ):
            raise ValueError(
                "basin_state_tolerance must be finite and nonnegative."
            )
    if energy_check_tolerance <= 0.0 or not math.isfinite(
        energy_check_tolerance
    ):
        raise ValueError("energy_check_tolerance must be finite and positive.")

    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"VQE archive not found: {input_path}")
    if output_dir is None:
        output_dir = input_path.parent.parent / "hessian"
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    assignment_rows: list[dict] = []
    point_results: list[PointResult] = []
    layer_paths: list[Path] = []

    with np.load(input_path, allow_pickle=False) as archive:
        if "h_param" not in archive.files:
            raise KeyError("The VQE archive is missing scalar metadata 'h_param'.")
        archived_h_array = np.asarray(archive["h_param"])
        if archived_h_array.shape != ():
            raise ValueError("Archive h_param must be scalar.")
        archived_h = float(archived_h_array)
        if not math.isfinite(archived_h):
            raise ValueError("Archive h_param must be finite.")
        if h_param is not None and not math.isclose(
            archived_h,
            float(h_param),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Requested h_param={h_param} does not match archive "
                f"h_param={archived_h}."
            )
        h_param = archived_h

        available_layers = _available_layers(archive)
        selected_layers = _select_layers(layers, available_layers)
        print(
            "Hessian analysis: "
            f"h={h_param}, layers={selected_layers}, "
            f"mode={'all endpoints' if all_endpoints else 'basin representatives'}, "
            f"method={HESSIAN_METHOD}, hvp_chunk_size={hvp_chunk_size}",
            flush=True,
        )

        for layer_number, layer in enumerate(selected_layers, start=1):
            run_indices, theta, final_energies, archived_gradient_norms = (
                _load_layer_inputs(archive, layer, runs)
            )
            print(
                f"[{layer_number}/{len(selected_layers)}] L={layer}: "
                f"clustering {len(run_indices)} endpoints",
                flush=True,
            )

            if basin_state_tolerance is None:
                states_np = None
            else:
                state_batch_function = jax.jit(
                    jax.vmap(lambda value: reduced_state(value, int(layer)))
                )
                states_np = np.asarray(
                    jax.device_get(
                        state_batch_function(jnp.asarray(theta, dtype=REAL_DTYPE))
                    )
                )

            (
                basin_ids,
                basins,
                seed_energy_distances,
                seed_state_distances,
            ) = assign_empirical_basins(
                final_energies,
                states_np,
                run_indices,
                archived_gradient_norms,
                energy_tolerance=basin_energy_tolerance,
                state_tolerance=basin_state_tolerance,
            )

            representative_positions = {
                basin.representative_position for basin in basins
            }
            basin_by_id = {basin.basin_id: basin for basin in basins}
            for position, run_index in enumerate(run_indices.tolist()):
                basin = basin_by_id[int(basin_ids[position])]
                assignment_rows.append(
                    {
                        "layer": int(layer),
                        "run_index": int(run_index),
                        "basin_id": int(basin_ids[position]),
                        "basin_size": len(basin.member_positions),
                        "is_basin_seed": int(position == basin.seed_position),
                        "is_basin_representative": int(
                            position == basin.representative_position
                        ),
                        "archive_final_energy": float(final_energies[position]),
                        "archive_final_gradient_norm": _safe_float(
                            archived_gradient_norms[position]
                        ),
                        "energy_distance_to_basin_seed": float(
                            seed_energy_distances[position]
                        ),
                        "state_distance_to_basin_seed": _safe_float(
                            seed_state_distances[position]
                        ),
                    }
                )

            if all_endpoints:
                analyzed_positions = list(range(len(run_indices)))
            else:
                analyzed_positions = [
                    basin.representative_position for basin in basins
                ]

            energy_function = make_energy_function(layer, h_param)
            value_and_gradient = jax.jit(jax.value_and_grad(energy_function))
            hessian_vector_chunk_function = (
                make_hessian_vector_chunk_function(energy_function)
            )
            layer_points: list[PointResult] = []

            print(
                f"L={layer}: {len(basins)} empirical basins; "
                f"computing {len(analyzed_positions)} Hessians",
                flush=True,
            )
            for point_number, position in enumerate(analyzed_positions, start=1):
                theta_jax = jnp.asarray(theta[position], dtype=REAL_DTYPE)
                energy_jax, gradient_jax = value_and_gradient(theta_jax)
                recomputed_energy = float(jax.device_get(energy_jax))
                gradient = np.asarray(
                    jax.device_get(gradient_jax), dtype=NP_REAL_DTYPE
                )
                if not np.all(np.isfinite(gradient)):
                    raise FloatingPointError(
                        f"Non-finite gradient at L={layer}, "
                        f"run={run_indices[position]}."
                    )
                archive_energy = float(final_energies[position])
                energy_residual = recomputed_energy - archive_energy
                if abs(energy_residual) > energy_check_tolerance:
                    raise ValueError(
                        "Recomputed energy does not match the saved VQE result: "
                        f"L={layer}, run={run_indices[position]}, "
                        f"saved={archive_energy:.17g}, "
                        f"recomputed={recomputed_energy:.17g}, "
                        f"residual={energy_residual:.3e}."
                    )

                # Validate the inexpensive value/gradient calculation before
                # spending time on all Hessian-vector chunks at this point.
                raw_hessian = assemble_hessian_from_hvp_chunks(
                    theta_jax,
                    hessian_vector_chunk_function,
                    chunk_size=hvp_chunk_size,
                )
                if not np.all(np.isfinite(raw_hessian)):
                    raise FloatingPointError(
                        f"Non-finite Hessian at L={layer}, "
                        f"run={run_indices[position]}."
                    )

                hessian_norm = float(np.linalg.norm(raw_hessian, ord="fro"))
                symmetry_residual = float(
                    np.linalg.norm(raw_hessian - raw_hessian.T, ord="fro")
                    / max(1.0, hessian_norm)
                )
                hessian = 0.5 * (raw_hessian + raw_hessian.T)
                structural_null_residual = float(
                    max(
                        np.max(np.abs(hessian[-STRUCTURAL_NULL_COUNT:, :])),
                        np.max(np.abs(hessian[:, -STRUCTURAL_NULL_COUNT:])),
                    )
                )
                full_metrics = summarize_spectrum(hessian, epsilon)
                quotient_hessian = hessian[
                    :-STRUCTURAL_NULL_COUNT, :-STRUCTURAL_NULL_COUNT
                ]
                quotient_metrics = summarize_spectrum(
                    quotient_hessian, epsilon
                )
                gradient_norm = float(np.linalg.norm(gradient))
                stationary, stationary_classification = (
                    _stationary_classification(
                        full_metrics,
                        gradient_norm,
                        stationarity_tolerance,
                    )
                )
                basin_id = int(basin_ids[position])
                basin = basin_by_id[basin_id]
                point = PointResult(
                    layer=int(layer),
                    basin_id=basin_id,
                    basin_size=len(basin.member_positions),
                    run_index=int(run_indices[position]),
                    is_representative=position in representative_positions,
                    archive_energy=archive_energy,
                    recomputed_energy=recomputed_energy,
                    energy_residual=energy_residual,
                    archived_gradient_norm=float(
                        archived_gradient_norms[position]
                    ),
                    gradient_norm=gradient_norm,
                    stationary=stationary,
                    epsilon=epsilon,
                    structural_null_count=STRUCTURAL_NULL_COUNT,
                    excess_zero_count=max(
                        full_metrics.zero_count - STRUCTURAL_NULL_COUNT, 0
                    ),
                    hessian_symmetry_residual=symmetry_residual,
                    structural_null_residual=structural_null_residual,
                    curvature_classification=_curvature_classification(
                        full_metrics
                    ),
                    stationary_classification=stationary_classification,
                    full=full_metrics,
                    quotient=quotient_metrics,
                    theta=np.asarray(theta[position], dtype=NP_REAL_DTYPE),
                    hessian=hessian if save_hessians else None,
                )
                layer_points.append(point)
                point_results.append(point)
                print(
                    f"  [{point_number}/{len(analyzed_positions)}] "
                    f"run={point.run_index}, basin={basin_id}, "
                    f"n-={full_metrics.negative_count}, "
                    f"n0={full_metrics.zero_count}, "
                    f"mu_min={full_metrics.minimum_eigenvalue:.3e}, "
                    f"class={stationary_classification}",
                    flush=True,
                )

            layer_path = output_dir / f"hessian_final_points_L{layer}.npz"
            _atomic_savez(
                layer_path,
                **_layer_result_arrays(
                    layer=layer,
                    h_param=h_param,
                    run_indices=run_indices,
                    theta=theta,
                    final_energies=final_energies,
                    archived_gradient_norms=archived_gradient_norms,
                    basin_ids=basin_ids,
                    basins=basins,
                    points=layer_points,
                    epsilon=epsilon,
                    stationarity_tolerance=stationarity_tolerance,
                    basin_energy_tolerance=basin_energy_tolerance,
                    basin_state_tolerance=basin_state_tolerance,
                    hvp_chunk_size=hvp_chunk_size,
                    all_endpoints=all_endpoints,
                    save_hessians=save_hessians,
                ),
            )
            layer_paths.append(layer_path)
            # Dense matrices are already persisted in the layer archive and
            # are not used by the cross-layer CSV summaries.  Releasing them
            # avoids retaining O(sum_L runs * (14L)^2) memory in
            # --all-endpoints --save-hessians mode.
            if save_hessians:
                for point in layer_points:
                    point.hessian = None
            del states_np
            jax.clear_caches()
            gc.collect()

    point_rows = [_point_to_csv_row(point) for point in point_results]
    for row in point_rows:
        row["stationarity_tolerance"] = stationarity_tolerance
    basin_rows = _basin_summary_rows(assignment_rows, point_results)

    assignment_path = output_dir / "hessian_basin_assignments.csv"
    point_path = output_dir / "hessian_endpoint_summary.csv"
    basin_path = output_dir / "hessian_basin_summary.csv"
    metadata_path = output_dir / "hessian_analysis_metadata.json"
    _atomic_write_csv(assignment_path, assignment_rows)
    _atomic_write_csv(point_path, point_rows)
    _atomic_write_csv(basin_path, basin_rows)
    _atomic_write_json(
        metadata_path,
        {
            "schema_version": SCHEMA_VERSION,
            "input_archive": str(input_path),
            "h_param": h_param,
            "layers": selected_layers,
            "requested_runs": None if runs is None else [int(run) for run in runs],
            "analysis_mode": (
                "all_selected_endpoints"
                if all_endpoints
                else "one_representative_per_empirical_basin"
            ),
            "epsilon": epsilon,
            "stationarity_tolerance": stationarity_tolerance,
            "energy_check_tolerance": energy_check_tolerance,
            "basin_energy_tolerance": basin_energy_tolerance,
            "basin_state_tolerance": basin_state_tolerance,
            "basin_definition": (
                "Same layer, final-energy difference from the basin seed at "
                "or below basin_energy_tolerance, and reduced four-qubit "
                "state Frobenius distance from the seed at or below "
                "basin_state_tolerance. This is an empirical endpoint "
                "cluster, not a certified dynamical attraction basin."
                if basin_state_tolerance is not None
                else "Same layer and final-energy difference from the basin "
                "seed at or below basin_energy_tolerance. This is an "
                "empirical endpoint cluster, not a certified dynamical "
                "attraction basin."
            ),
            "representative_definition": (
                "Minimum archived final gradient norm in the basin; ties are "
                "broken by final energy and run index."
            ),
            "condition_number_definition": (
                "lambda_max / lambda_min_positive, where "
                "lambda_min_positive > epsilon; NaN in NPZ and 'nan' in CSV "
                "when no such positive eigenvalue exists."
            ),
            "structural_null_count": STRUCTURAL_NULL_COUNT,
            "hessian_method": HESSIAN_METHOD,
            "hvp_chunk_size": hvp_chunk_size,
            "structural_null_note": (
                "The final layer's two dynamic-channel parameters are exact "
                "energy-null directions. Full-space metrics include them; "
                "quotient_* metrics remove only these final two coordinates."
            ),
            "save_hessians": bool(save_hessians),
            "layer_result_files": [str(path) for path in layer_paths],
        },
    )

    print(f"Saved Hessian results to: {output_dir}", flush=True)
    return {
        "assignments": assignment_path,
        "endpoints": point_path,
        "basins": basin_path,
        "metadata": metadata_path,
        **{f"layer_{index}": path for index, path in enumerate(layer_paths)},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Path to vqe_optimization_histories.npz. By default, use "
            "./figs/dpqc/h_<h>/numerical_results/energy/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. By default, use the hessian directory beside "
            "the input archive's energy directory."
        ),
    )
    parser.add_argument(
        "--h-param",
        type=_finite_float,
        default=None,
        help=(
            "Expected Hamiltonian parameter. It is validated against the "
            "archive. When --input is omitted, the config value is used to "
            "locate the archive."
        ),
    )
    parser.add_argument(
        "--layers",
        type=_parse_integer_list,
        default=None,
        help="Comma-separated archive layers to analyze (default: all).",
    )
    parser.add_argument(
        "--runs",
        type=_parse_integer_list,
        default=None,
        help=(
            "Comma-separated zero-based run indices to include before basin "
            "clustering (default: all; negative indices are accepted)."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=_positive_float,
        default=DEFAULT_EIGENVALUE_EPSILON,
        help=(
            "Absolute eigenvalue threshold used for n-, n0, and positive "
            f"eigenvalues (default: {DEFAULT_EIGENVALUE_EPSILON:g})."
        ),
    )
    parser.add_argument(
        "--stationarity-tolerance",
        type=_positive_float,
        default=DEFAULT_STATIONARITY_TOLERANCE,
        help=(
            "Maximum gradient norm for stationary-point labels "
            f"(default: {DEFAULT_STATIONARITY_TOLERANCE:g})."
        ),
    )
    parser.add_argument(
        "--hvp-chunk-size",
        type=_positive_int,
        default=DEFAULT_HVP_CHUNK_SIZE,
        help=(
            "Number of Hessian-vector products evaluated together while "
            "assembling the dense Hessian "
            f"(default: {DEFAULT_HVP_CHUNK_SIZE})."
        ),
    )
    parser.add_argument(
        "--basin-energy-tolerance",
        type=_nonnegative_float,
        default=DEFAULT_BASIN_ENERGY_TOLERANCE,
        help=(
            "Maximum final-energy difference from a basin seed "
            f"(default: {DEFAULT_BASIN_ENERGY_TOLERANCE:g})."
        ),
    )
    parser.add_argument(
        "--basin-state-tolerance",
        type=_nonnegative_float,
        default=DEFAULT_BASIN_STATE_TOLERANCE,
        help=(
            "Maximum reduced-state Frobenius distance from a basin seed "
            f"(default: {DEFAULT_BASIN_STATE_TOLERANCE:g})."
        ),
    )
    parser.add_argument(
        "--energy-only-basins",
        action="store_true",
        help="Cluster by final energy only, without the reduced-state test.",
    )
    parser.add_argument(
        "--all-endpoints",
        action="store_true",
        help="Compute every selected endpoint Hessian instead of representatives.",
    )
    parser.add_argument(
        "--save-hessians",
        action="store_true",
        help="Also store dense Hessian matrices in each layer NPZ file.",
    )
    parser.add_argument(
        "--energy-check-tolerance",
        type=_positive_float,
        default=DEFAULT_ENERGY_CHECK_TOLERANCE,
        help=(
            "Maximum absolute difference between archived and recomputed "
            f"endpoint energy (default: {DEFAULT_ENERGY_CHECK_TOLERANCE:g})."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lookup_h = float(cfg.H_PARAM) if args.h_param is None else args.h_param
    input_path = (
        _default_archive_path(lookup_h) if args.input is None else args.input
    )
    run_hessian_analysis(
        input_path=input_path,
        output_dir=args.output_dir,
        h_param=args.h_param,
        layers=args.layers,
        runs=args.runs,
        epsilon=args.epsilon,
        stationarity_tolerance=args.stationarity_tolerance,
        basin_energy_tolerance=args.basin_energy_tolerance,
        basin_state_tolerance=(
            None if args.energy_only_basins else args.basin_state_tolerance
        ),
        all_endpoints=args.all_endpoints,
        save_hessians=args.save_hessians,
        energy_check_tolerance=args.energy_check_tolerance,
        hvp_chunk_size=args.hvp_chunk_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
