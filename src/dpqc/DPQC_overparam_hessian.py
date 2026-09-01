#!/usr/bin/env python
# coding: utf-8
"""Compute DPQC energy-Hessian rank and condition at random parameters.

For every requested circuit depth, this program regenerates exactly the
random-parameter convention used by ``DPQC_overparam_qfim.py``: one batched
``jax.random.uniform`` call with key ``QFIM_SAMPLE_SEED_BASE + layer`` and
parameters drawn uniformly from ``[-pi, pi)`` in float64. It then computes
the full signed energy Hessian at every point.

The Hessian is generally indefinite and has exact structural zero modes, so
both reported quantities use the absolute spectrum at the fixed QFIM rank
threshold ``tau = cfg.QFIM_EFFECTIVE_RANK_THRESHOLD``:

    rank_tau(H) = number of eigenvalues with |lambda_i| >= tau,
    kappa_tau(H) = max_active |lambda_i| / min_active |lambda_i|.

The condition number is NaN only when the active spectrum is empty. This is
the condition number on the threshold-active subspace, rather than the
ordinary full-space condition number (which is infinite in the presence of
exact zero modes).

Only the rank and active condition number samples needed by the corresponding
plots are saved, in one ``hessian_random_points.npz`` archive.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from functools import reduce
from pathlib import Path
from typing import Iterable, Sequence


_MODULE_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _MODULE_DIR.parent / "common"
_common_dir_string = str(_COMMON_DIR)
if _common_dir_string not in sys.path:
    sys.path.insert(0, _common_dir_string)

import config_overparam as cfg


# Match the QFIM and VQE programs' deterministic double-precision CPU default
# while allowing callers to select a different JAX platform before launch.
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

TOP, LEFT, RIGHT, BOTTOM, CENTRE = 0, 1, 2, 3, 4
LAYER_PAIRS = (
    (LEFT, BOTTOM),
    (RIGHT, BOTTOM),
    (TOP, RIGHT),
    (TOP, CENTRE),
)

HESSIAN_RANK_THRESHOLD = float(cfg.QFIM_EFFECTIVE_RANK_THRESHOLD)
DEFAULT_NUM_SAMPLES = int(cfg.NUM_QFIM_SAMPLES)
DEFAULT_SEED_BASE = int(cfg.QFIM_SAMPLE_SEED_BASE)
DEFAULT_HVP_CHUNK_SIZE = 8
HESSIAN_METHOD = "chunked_forward_over_reverse_hvp"
SCHEMA_VERSION = 1

if not math.isfinite(HESSIAN_RANK_THRESHOLD) or HESSIAN_RANK_THRESHOLD <= 0.0:
    raise ValueError("QFIM_EFFECTIVE_RANK_THRESHOLD must be finite and positive.")


# ---------------------------------------------------------------------------
# Side-effect-free DPQC state and energy map
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
    """Return the five-retained-qubit DPQC state at one fixed depth."""

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


# ---------------------------------------------------------------------------
# Bounded-memory Hessian construction
# ---------------------------------------------------------------------------


def make_hessian_vector_chunk_function(energy_function):
    """Compile fixed-size batches of exact Hessian-vector products.

    A forward-mode JVP of the reverse-mode gradient gives Hessian columns
    without asking XLA to materialize every tangent direction simultaneously.
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

    chunk_size = _require_positive_int(chunk_size, "chunk_size")
    theta = jnp.asarray(theta, dtype=REAL_DTYPE)
    if theta.ndim != 1 or int(theta.size) == 0:
        raise ValueError("theta must be a nonempty one-dimensional array.")

    dimension = int(theta.size)
    hessian = np.empty((dimension, dimension), dtype=NP_REAL_DTYPE)
    tangent_chunk = np.zeros((chunk_size, dimension), dtype=NP_REAL_DTYPE)

    # Zero-padding the last chunk keeps every compiled call at the same shape.
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
        hessian[:, start:stop] = hessian_columns[:valid_count].T

    return hessian


# ---------------------------------------------------------------------------
# Fixed-threshold spectrum and QFIM-matched random samples
# ---------------------------------------------------------------------------


def hessian_rank_and_condition_from_eigenvalues(
    eigenvalues: np.ndarray,
) -> tuple[int, NP_REAL_DTYPE]:
    """Return inclusive-threshold rank and active-spectrum condition number."""

    eigenvalues = np.asarray(eigenvalues, dtype=NP_REAL_DTYPE)
    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be a one-dimensional array.")
    if not np.all(np.isfinite(eigenvalues)):
        raise FloatingPointError("Hessian eigenvalues contain non-finite values.")

    absolute_eigenvalues = np.abs(eigenvalues)
    active = absolute_eigenvalues >= HESSIAN_RANK_THRESHOLD
    rank = int(np.count_nonzero(active))
    if rank == 0:
        return rank, NP_REAL_DTYPE(np.nan)

    active_absolute_eigenvalues = absolute_eigenvalues[active]
    condition_number = NP_REAL_DTYPE(
        np.max(active_absolute_eigenvalues)
        / np.min(active_absolute_eigenvalues)
    )
    return rank, condition_number


def generate_qfim_random_theta_samples(
    layer: int,
    *,
    num_samples: int,
    seed_base: int,
) -> jnp.ndarray:
    """Regenerate exactly the random-theta convention used by the QFIM stage."""

    layer = _require_positive_int(layer, "layer")
    num_samples = _require_positive_int(num_samples, "num_samples")
    seed_base = _require_nonnegative_int(seed_base, "seed_base")
    n_params = NUM_PARAMS_PER_LAYER * layer
    return jax.random.uniform(
        jax.random.PRNGKey(seed_base + layer),
        shape=(num_samples, n_params),
        dtype=REAL_DTYPE,
        minval=jnp.asarray(-jnp.pi, dtype=REAL_DTYPE),
        maxval=jnp.asarray(jnp.pi, dtype=REAL_DTYPE),
    )


# ---------------------------------------------------------------------------
# Validation and output helpers
# ---------------------------------------------------------------------------


def _require_positive_int(value, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _require_nonnegative_int(value, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def _positive_int(value: str) -> int:
    try:
        return _require_positive_int(int(value), "value")
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from exc


def _nonnegative_int(value: str) -> int:
    try:
        return _require_nonnegative_int(int(value), "value")
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            "value must be a nonnegative integer"
        ) from exc


def _parse_layer_list(value: str) -> tuple[int, ...]:
    tokens = value.split(",")
    if not tokens or any(token.strip() == "" for token in tokens):
        raise argparse.ArgumentTypeError(
            "expected a nonempty comma-separated list of positive integers"
        )
    try:
        parsed = tuple(int(token.strip()) for token in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        ) from exc
    if any(layer <= 0 for layer in parsed):
        raise argparse.ArgumentTypeError("layers must be positive")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("layers must not be duplicated")
    return parsed


def _default_layers() -> list[int]:
    max_layer = int(cfg.DPQC_QFIM_MAX_LAYER)
    dense_until_layer = int(cfg.DPQC_QFIM_DENSE_UNTIL_LAYER)
    sparse_step = int(cfg.DPQC_QFIM_SPARSE_STEP)
    dense_end = min(dense_until_layer, max_layer)
    return list(range(1, dense_end + 1)) + list(
        range(dense_end + sparse_step, max_layer + 1, sparse_step)
    )


def _validated_layers(layers: Sequence[int] | None) -> list[int]:
    selected = _default_layers() if layers is None else list(layers)
    if not selected:
        raise ValueError("At least one Hessian layer is required.")
    normalized = [
        _require_positive_int(layer, f"layers[{index}]")
        for index, layer in enumerate(selected)
    ]
    if len(set(normalized)) != len(normalized):
        raise ValueError("layers must not contain duplicates.")
    return sorted(normalized)


def _default_output_dir(h_param: float) -> Path:
    return (
        Path.cwd()
        / "figs"
        / "dpqc"
        / f"h_{float(h_param)}"
        / "numerical_results"
        / "hessian"
    )


def _atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Main random-point workflow
# ---------------------------------------------------------------------------


def run_hessian_analysis(
    *,
    output_dir: Path | None = None,
    h_param: float = float(cfg.H_PARAM),
    layers: Sequence[int] | None = None,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    seed_base: int = DEFAULT_SEED_BASE,
    hvp_chunk_size: int = DEFAULT_HVP_CHUNK_SIZE,
) -> dict[str, Path]:
    """Compute random-point Hessian rank/condition samples and save one NPZ."""

    h_param = float(h_param)
    if not math.isfinite(h_param):
        raise ValueError("h_param must be finite.")
    selected_layers = _validated_layers(layers)
    num_samples = _require_positive_int(num_samples, "num_samples")
    seed_base = _require_nonnegative_int(seed_base, "seed_base")
    hvp_chunk_size = _require_positive_int(hvp_chunk_size, "hvp_chunk_size")

    if output_dir is None:
        output_dir = _default_output_dir(h_param)
    output_dir = Path(output_dir).expanduser().resolve()
    output_path = output_dir / "hessian_random_points.npz"

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=NP_INT_DTYPE),
        "h_param": np.asarray(h_param, dtype=NP_REAL_DTYPE),
        "layers": np.asarray(selected_layers, dtype=NP_INT_DTYPE),
        "num_hessian_samples": np.asarray(num_samples, dtype=NP_INT_DTYPE),
        "hessian_sample_seed_base": np.asarray(seed_base, dtype=NP_INT_DTYPE),
        "hessian_rank_threshold": np.asarray(
            HESSIAN_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        "hessian_rank_definition": np.asarray(
            "count(abs(eigenvalue) >= hessian_rank_threshold)"
        ),
        "hessian_condition_number_definition": np.asarray(
            "max(active abs(eigenvalue)) / min(active abs(eigenvalue)); "
            "NaN when rank is zero"
        ),
        "parameter_distribution": np.asarray(
            "jax.random.uniform[-pi, pi), float64, "
            "PRNGKey(hessian_sample_seed_base + layer)"
        ),
        "parameters_per_layer": np.asarray(
            NUM_PARAMS_PER_LAYER,
            dtype=NP_INT_DTYPE,
        ),
        "hessian_method": np.asarray(HESSIAN_METHOD),
        "hvp_chunk_size": np.asarray(hvp_chunk_size, dtype=NP_INT_DTYPE),
    }

    print(
        "Random-point Hessian analysis: "
        f"h={h_param}, layers={selected_layers}, samples={num_samples}, "
        f"seed_base={seed_base}, threshold={HESSIAN_RANK_THRESHOLD:.3e}, "
        f"hvp_chunk_size={hvp_chunk_size}",
        flush=True,
    )

    for layer_number, layer in enumerate(selected_layers, start=1):
        n_params = NUM_PARAMS_PER_LAYER * layer
        theta_samples = generate_qfim_random_theta_samples(
            layer,
            num_samples=num_samples,
            seed_base=seed_base,
        )
        energy_function = make_energy_function(layer, h_param)
        hessian_vector_chunk_function = make_hessian_vector_chunk_function(
            energy_function
        )
        ranks = np.empty(num_samples, dtype=NP_INT_DTYPE)
        condition_numbers = np.empty(num_samples, dtype=NP_REAL_DTYPE)

        print(
            f"[{layer_number}/{len(selected_layers)}] L={layer}: "
            f"{num_samples} Hessians of dimension {n_params}",
            flush=True,
        )
        progress_interval = max(1, num_samples // 10)
        for sample_index in range(num_samples):
            raw_hessian = assemble_hessian_from_hvp_chunks(
                theta_samples[sample_index],
                hessian_vector_chunk_function,
                chunk_size=hvp_chunk_size,
            )
            if not np.all(np.isfinite(raw_hessian)):
                raise FloatingPointError(
                    f"Non-finite Hessian at L={layer}, sample={sample_index}."
                )
            hessian = 0.5 * (raw_hessian + raw_hessian.T)
            eigenvalues = np.linalg.eigvalsh(hessian).astype(
                NP_REAL_DTYPE,
                copy=False,
            )
            rank, condition_number = (
                hessian_rank_and_condition_from_eigenvalues(eigenvalues)
            )
            ranks[sample_index] = rank
            condition_numbers[sample_index] = condition_number

            completed = sample_index + 1
            if completed == num_samples or completed % progress_interval == 0:
                print(
                    f"  L={layer}: {completed}/{num_samples} samples",
                    flush=True,
                )

        arrays[f"L{layer}_rank"] = ranks
        arrays[f"L{layer}_condition_number"] = condition_numbers

        del theta_samples
        del energy_function
        del hessian_vector_chunk_function
        jax.clear_caches()
        gc.collect()

    _atomic_savez(output_path, **arrays)
    print(f"Saved random-point Hessian results to: {output_path}", flush=True)
    return {"random_points": output_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. By default, use "
            "./figs/dpqc/h_<h>/numerical_results/hessian."
        ),
    )
    parser.add_argument(
        "--h-param",
        type=_finite_float,
        default=float(cfg.H_PARAM),
        help="Hamiltonian parameter (default: config_overparam.H_PARAM).",
    )
    parser.add_argument(
        "--layers",
        type=_parse_layer_list,
        default=None,
        help=(
            "Comma-separated positive layers (default: the DPQC QFIM layer "
            "schedule from config_overparam.py)."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=_positive_int,
        default=DEFAULT_NUM_SAMPLES,
        help=(
            "Random points per layer "
            f"(default: NUM_QFIM_SAMPLES={DEFAULT_NUM_SAMPLES})."
        ),
    )
    parser.add_argument(
        "--seed-base",
        type=_nonnegative_int,
        default=DEFAULT_SEED_BASE,
        help=(
            "Base random seed; layer L uses PRNGKey(seed_base + L) "
            f"(default: QFIM_SAMPLE_SEED_BASE={DEFAULT_SEED_BASE})."
        ),
    )
    parser.add_argument(
        "--hvp-chunk-size",
        type=_positive_int,
        default=DEFAULT_HVP_CHUNK_SIZE,
        help=(
            "Hessian-vector products per compiled batch "
            f"(default: {DEFAULT_HVP_CHUNK_SIZE})."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_hessian_analysis(
        output_dir=args.output_dir,
        h_param=args.h_param,
        layers=args.layers,
        num_samples=args.num_samples,
        seed_base=args.seed_base,
        hvp_chunk_size=args.hvp_chunk_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
