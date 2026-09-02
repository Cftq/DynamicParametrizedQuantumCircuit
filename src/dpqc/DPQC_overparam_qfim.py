#!/usr/bin/env python
# coding: utf-8
"""Compute and save DPQC QFIM diagnostics from a saved VQE archive.

This is the QFIM half of the former DPQC_overparam_compute.py workflow.
It never runs VQE optimization. Run DPQC_overparam_vqe.py first with the same
Hamiltonian parameter and working directory so that its saved history can be
validated and reused.

Examples::

    python DPQC_overparam_qfim.py
    python DPQC_overparam_qfim.py --h-param 0.10
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Tuple


# Support direct execution from any working directory, for example:
#     python C:\...\src\dpqc\DPQC_overparam_qfim.py
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
            "Load saved DPQC VQE histories, compute QFIM diagnostics for "
            "keep=(0,1,2,3) and keep=(0,1,2,3,4), and save the results."
        )
    )
    parser.add_argument(
        "--h-param",
        type=_finite_float,
        default=float(cfg.H_PARAM),
        help=(
            "Hamiltonian parameter H_PARAM (default: value from "
            "config_overparam.py). It must match the saved VQE archive."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    _CLI_ARGS = _parse_cli_args()
else:
    _CLI_ARGS = argparse.Namespace(h_param=float(cfg.H_PARAM))

RUN_VQE_STAGE = False
RUN_QFIM_STAGE = True


# ------------------------------------------------------------
# IMPORTANT: env vars should be set BEFORE importing jax
# ------------------------------------------------------------
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import tensorcircuit as tc
from tqdm.auto import tqdm

jax.config.update("jax_enable_x64", True)

# The shared Hamiltonian helper obtains Pauli tensors from TensorCircuit.
# Preserve its historical JAX/complex128 backend; no circuit is constructed or
# drawn in this QFIM script.
tc.set_backend("jax")
tc.set_dtype("complex128")

REAL_DTYPE = jnp.float64
COMPLEX_DTYPE = jnp.complex128
NP_REAL_DTYPE = np.float64
NP_INT_DTYPE = np.int64

from dpqc_overparam_common import (
    build_layer_list,
    rho_zero_state,
    threshold_psd_eigvals_for_rank,
)

# ============================================================
# Shared constants / helpers
# ============================================================
num_system_qubits = 5
h_param = float(_CLI_ARGS.h_param)
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


vqe_optimization_result_path = os.path.join(
    energy_results_dir,
    "vqe_optimization_histories.npz",
)


def _load_saved_vqe_samples(inpath: str):
    """Load the float64 VQE parameter arrays required by the QFIM stage."""
    if not os.path.isfile(inpath):
        raise FileNotFoundError(
            "The QFIM-only stage requires an existing VQE archive: "
            f"{inpath}"
        )

    with np.load(inpath, allow_pickle=False) as data:
        metadata_keys = (
            "h_param",
            "num_runs",
            "sample_iters",
            "vqe_layers",
        )
        missing_metadata = [key for key in metadata_keys if key not in data]
        if missing_metadata:
            raise KeyError(
                "VQE archive is missing required metadata: "
                + ", ".join(missing_metadata)
            )

        archived_h_param = np.asarray(data["h_param"])
        archived_num_runs = np.asarray(data["num_runs"])
        archived_sample_iters = np.array(
            data["sample_iters"],
            dtype=NP_INT_DTYPE,
            copy=True,
        )
        archived_layers = np.array(
            data["vqe_layers"],
            dtype=NP_INT_DTYPE,
            copy=True,
        )

        if archived_h_param.shape != () or archived_num_runs.shape != ():
            raise ValueError(
                "h_param and num_runs must be scalar values in the VQE archive."
            )
        archived_h_value = NP_REAL_DTYPE(archived_h_param.item())
        if archived_h_value != NP_REAL_DTYPE(h_param):
            raise ValueError(
                "Saved VQE h_param does not match the current Hamiltonian: "
                f"{float(archived_h_value)} != {float(h_param)}."
            )

        archived_num_runs_int = int(archived_num_runs.item())
        if archived_num_runs_int <= 0:
            raise ValueError("Saved num_runs must be positive.")
        if archived_layers.ndim != 1 or archived_layers.size == 0:
            raise ValueError("Saved vqe_layers must be a non-empty 1D array.")
        if np.any(archived_layers <= 0):
            raise ValueError("Saved vqe_layers must be positive.")
        if np.unique(archived_layers).size != archived_layers.size:
            raise ValueError("Saved vqe_layers must not contain duplicates.")
        if (
            archived_sample_iters.ndim != 1
            or archived_sample_iters.size == 0
        ):
            raise ValueError(
                "Saved sample_iters must be a non-empty 1D array."
            )
        if np.any(archived_sample_iters < 0) or np.any(
            np.diff(archived_sample_iters) <= 0
        ):
            raise ValueError(
                "Saved sample_iters must be nonnegative and strictly increasing."
            )

        theta_by_layer = {}
        expected_real_dtype = np.dtype(NP_REAL_DTYPE)

        for layer_value in archived_layers:
            layer = int(layer_value)
            theta_key = f"L{layer}_theta_samples"
            if theta_key not in data:
                raise KeyError(
                    "VQE archive is missing required array: " + theta_key
                )

            theta_raw = data[theta_key]
            if theta_raw.dtype != expected_real_dtype:
                raise TypeError(
                    f"Saved L={layer} theta samples must be float64, "
                    f"got {theta_raw.dtype}."
                )

            theta = np.array(theta_raw, dtype=NP_REAL_DTYPE, copy=True)
            expected_shape = (
                archived_num_runs_int,
                archived_sample_iters.size,
                n_param_per_layer * layer,
            )
            if theta.shape != expected_shape:
                raise ValueError(
                    f"Saved L={layer} theta sample shape mismatch: "
                    f"theta={theta.shape}, expected={expected_shape}."
                )
            if not np.all(np.isfinite(theta)):
                raise ValueError(
                    f"Saved L={layer} theta samples contain non-finite values."
                )

            theta_by_layer[layer] = theta

    return (
        theta_by_layer,
        [int(layer) for layer in archived_layers.tolist()],
        archived_sample_iters.astype(NP_INT_DTYPE, copy=True),
    )


def run_qfim(*, include_optimization_path: bool = True):
    """Compute and save QFIM diagnostics.

    Random-point diagnostics are always computed. Optimization-path
    diagnostics additionally require the saved VQE parameter histories and can
    be disabled by callers that do not need them.
    """
    if include_optimization_path:
        (
            theta_sample_traces_by_layer,
            vqe_layer_list,
            sample_iters,
        ) = _load_saved_vqe_samples(vqe_optimization_result_path)
        print(
            "Loaded saved float64 VQE samples for the QFIM calculation: "
            f"{vqe_optimization_result_path}"
        )

    # ============================================================
    # Random-parameter QFIM: compute and save numerical results
    # ============================================================
    # QFIM rank + eigenvalue diagnostics for both retained subsystems.
    # ============================================================
    KEEP_WIRES_4 = (0, 1, 2, 3)
    assert KEEP_WIRES_4 == tuple(range(num_system_qubits - 1))
    KEEP_WIRES_5 = tuple(range(num_system_qubits))

    QFIM_EFFECTIVE_RANK_THRESHOLD = cfg.QFIM_EFFECTIVE_RANK_THRESHOLD

    EIG_SUM_EPS = cfg.EIG_SUM_EPS
    QFIM_EIG_PLOT_EPS = cfg.QFIM_EIG_PLOT_EPS
    NUM_QFIM_SAMPLES = cfg.NUM_QFIM_SAMPLES
    QFIM_SAMPLE_SEED_BASE = cfg.QFIM_SAMPLE_SEED_BASE
    RED_JVP_CHUNK = cfg.RED_JVP_CHUNK
    PARTICIPATION_EFFECTIVE_RANK_EPS = cfg.PARTICIPATION_EFFECTIVE_RANK_EPS

    RUN_QFIM_EFFECTIVE_RANK_RANDOM_POINTS = (
        cfg.RUN_QFIM_EFFECTIVE_RANK_RANDOM_POINTS
    )

    def psd_desc_eigs(F: jnp.ndarray):
        evals = jnp.clip(jnp.linalg.eigvalsh(_hermitian(F)), a_min=0.0)

        return evals[::-1]


    def psd_rank_and_desc_eigs(F: jnp.ndarray):
        evals_desc = psd_desc_eigs(F)
        rank, _ = threshold_psd_eigvals_for_rank(evals_desc)

        return rank, evals_desc


    def participation_effective_rank_from_eigvals(
        eigvals,
        *,
        threshold=QFIM_EFFECTIVE_RANK_THRESHOLD,
        eps=PARTICIPATION_EFFECTIVE_RANK_EPS,
    ):
        """Return ``Tr(A)^2 / Tr(A^2)`` for the active PSD spectrum.

        The zero-matrix convention is participation rank 0.  Inputs are clipped
        at zero so that round-off-sized negative eigenvalues cannot inflate the
        denominator, then values not strictly larger than ``threshold`` are
        discarded.
        """
        eigvals = np.clip(np.asarray(eigvals, dtype=NP_REAL_DTYPE), 0.0, None)
        eigvals = np.where(
            eigvals > float(threshold),
            eigvals,
            NP_REAL_DTYPE(0.0),
        )
        trace = NP_REAL_DTYPE(np.sum(eigvals))
        frobenius_norm_sq = NP_REAL_DTYPE(np.sum(eigvals * eigvals))

        if frobenius_norm_sq <= float(eps):
            return NP_REAL_DTYPE(0.0)

        return NP_REAL_DTYPE((trace * trace) / frobenius_norm_sq)


    def qfim_spectral_summary_from_eigvals(
        eigvals,
        *,
        rank_threshold=QFIM_EFFECTIVE_RANK_THRESHOLD,
        participation_eps=PARTICIPATION_EFFECTIVE_RANK_EPS,
    ):
        """Derive all trace/rank diagnostics from one clipped QFIM spectrum."""
        eigvals = np.clip(np.asarray(eigvals, dtype=NP_REAL_DTYPE), 0.0, None)
        eigvals_desc = np.sort(eigvals)[::-1]
        active = eigvals_desc > float(rank_threshold)
        threshold_rank = int(np.count_nonzero(active))
        trace = NP_REAL_DTYPE(np.sum(eigvals_desc))
        frobenius_norm_sq = NP_REAL_DTYPE(
            np.sum(eigvals_desc * eigvals_desc)
        )
        participation_rank = participation_effective_rank_from_eigvals(
            eigvals_desc,
            threshold=rank_threshold,
            eps=participation_eps,
        )
        largest = (
            NP_REAL_DTYPE(eigvals_desc[0])
            if eigvals_desc.size
            else NP_REAL_DTYPE(np.nan)
        )

        if threshold_rank:
            smallest_active = NP_REAL_DTYPE(eigvals_desc[active][-1])
            condition_active = NP_REAL_DTYPE(largest / smallest_active)
        else:
            smallest_active = NP_REAL_DTYPE(np.nan)
            condition_active = NP_REAL_DTYPE(np.nan)

        return {
            "evals": eigvals_desc,
            "qfim_threshold_rank": NP_INT_DTYPE(threshold_rank),
            "threshold_rank": NP_INT_DTYPE(threshold_rank),
            "qfim_participation_rank": participation_rank,
            "participation_rank": participation_rank,
            "qfim_trace": trace,
            "trace": trace,
            "qfim_frobenius_norm_sq": frobenius_norm_sq,
            "frobenius_norm_sq": frobenius_norm_sq,
            "largest_qfim_eigenvalue": largest,
            "largest_eigenvalue": largest,
            "smallest_active_qfim_eigenvalue": smallest_active,
            "smallest_active_eigenvalue": smallest_active,
            "condition_number_active": condition_active,
            "active_condition_number": condition_active,
        }


    def make_mixed_state_qfim_matrix_fn_for_layer_sequential(
        n_layer: int,
        *,
        rho_from_rho5_fn,
        n_keep: int,
        jvp_chunk: int = RED_JVP_CHUNK,
    ):
        """Build the SLD-QFIM for a reduced state derived from the kept rho5.

        ``rho_from_rho5_fn`` is a static Python callable applied to the five-qubit
        state returned by ``rho_keep_sequential_dpqc``.  This keeps the SLD
        construction identical for keep=(0,1,2,3) and keep=(0,1,2,3,4), while
        allowing the former to trace out wire 4 and the latter to use rho5
        directly.
        """
        n_keep = int(n_keep)
        jvp_chunk = int(jvp_chunk)
        if not 1 <= n_keep <= num_system_qubits:
            raise ValueError(
                f"n_keep must be in [1, {num_system_qubits}], got {n_keep}."
            )
        if jvp_chunk <= 0:
            raise ValueError(f"jvp_chunk must be positive, got {jvp_chunk}.")

        dim_state = 2**n_keep
        dim_vec = (2**n_keep) ** 2

        @jax.jit
        def rho_sub_fn(theta: jnp.ndarray) -> jnp.ndarray:
            rho5 = rho_keep_sequential_dpqc(theta, n_layer=n_layer)
            rho_sub = rho_from_rho5_fn(rho5)
            return _hermitian(rho_sub)

        def qfim_mixed_state(theta: jnp.ndarray) -> jnp.ndarray:
            rho, rho_jvp = jax.linearize(rho_sub_fn, theta)
            rho = _hermitian(rho)
            if rho.shape != (dim_state, dim_state):
                raise ValueError(
                    "rho_from_rho5_fn returned an incompatible density-matrix "
                    f"shape {rho.shape}; expected {(dim_state, dim_state)}."
                )

            lam, U = jnp.linalg.eigh(rho)
            lam = jnp.clip(lam, a_min=0.0)

            lam_sum = lam[:, None] + lam[None, :]

            sqrtW = jnp.sqrt(
                jnp.where(
                    lam_sum > EIG_SUM_EPS,
                    2.0 / lam_sum,
                    0.0,
                )
            )

            Udag = jnp.conjugate(U).T
            n_params = int(theta.shape[0])
            eye = jnp.eye(n_params, dtype=theta.dtype)

            def _to_eig(d: jnp.ndarray) -> jnp.ndarray:
                return Udag @ d @ U

            blocks = []

            for s in range(0, n_params, jvp_chunk):
                V = eye[s: min(s + jvp_chunk, n_params), :]
                drho_B = jax.vmap(rho_jvp)(V)

                drho_B = 0.5 * (
                    drho_B + jnp.conjugate(jnp.swapaxes(drho_B, 1, 2))
                )

                Cflat_B = jnp.reshape(
                    jax.vmap(_to_eig)(drho_B) * sqrtW[None, :, :],
                    (V.shape[0], dim_vec),
                )

                blocks.append(Cflat_B)

            Cflat = jnp.concatenate(blocks, axis=0)
            F_red = jnp.real(Cflat @ jnp.conjugate(Cflat).T)

            return 0.5 * (F_red + F_red.T)

        return qfim_mixed_state


    def make_reduced0123_qfim_matrix_fn_for_layer_sequential(
        n_layer: int,
        *,
        jvp_chunk: int = RED_JVP_CHUNK,
    ):
        """Compatibility wrapper for the SLD-QFIM of reduced keep=(0,1,2,3)."""
        return make_mixed_state_qfim_matrix_fn_for_layer_sequential(
            n_layer=n_layer,
            rho_from_rho5_fn=rho4_from_rho5,
            n_keep=4,
            jvp_chunk=jvp_chunk,
        )


    def make_reduced01234_qfim_matrix_fn_for_layer_sequential(
        n_layer: int,
        *,
        jvp_chunk: int = RED_JVP_CHUNK,
    ):
        """Return the SLD-QFIM for the full kept state on wires (0,1,2,3,4)."""
        return make_mixed_state_qfim_matrix_fn_for_layer_sequential(
            n_layer=n_layer,
            rho_from_rho5_fn=lambda rho5: rho5,
            n_keep=5,
            jvp_chunk=jvp_chunk,
        )


    def make_reduced_rho_rank_fn_for_layer_sequential(
        n_layer: int,
        *,
        rho_from_rho5_fn,
    ):
        @jax.jit
        def rho_rank(theta: jnp.ndarray) -> jnp.ndarray:
            rho5 = rho_keep_sequential_dpqc(theta, n_layer=n_layer)
            rho_reduced = _hermitian(rho_from_rho5_fn(rho5))
            evals = jnp.clip(jnp.linalg.eigvalsh(rho_reduced), a_min=0.0)
            threshold = jnp.asarray(QFIM_EFFECTIVE_RANK_THRESHOLD, dtype=evals.dtype)

            return jnp.sum(evals > threshold)

        return rho_rank


    def make_reduced0123_rho_rank_fn_for_layer_sequential(n_layer: int):
        return make_reduced_rho_rank_fn_for_layer_sequential(
            n_layer,
            rho_from_rho5_fn=rho4_from_rho5,
        )


    def make_reduced01234_rho_rank_fn_for_layer_sequential(n_layer: int):
        return make_reduced_rho_rank_fn_for_layer_sequential(
            n_layer,
            rho_from_rho5_fn=lambda rho5: rho5,
        )


    def make_joint_reduced_qfim_data_fn_for_layer_sequential(
        n_layer: int,
        *,
        jvp_chunk: int = RED_JVP_CHUNK,
    ):
        """Build F4/F5 and density-matrix ranks from one rho5 linearization.

        At each parameter point, ``rho5`` and its linear map are evaluated
        once. Every parameter-direction chunk ``d_rho5`` is also evaluated
        once; the corresponding ``rho4`` and ``d_rho4`` are obtained by
        tracing wire 4. The same state eigendecompositions provide both
        density-matrix ranks.

        The returned tuple is ``(F4, F5, rho_rank4, rho_rank5)``.
        Callers should JIT the returned function once per layer.
        """
        n_layer = int(n_layer)
        jvp_chunk = int(jvp_chunk)
        if n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {n_layer}.")
        if jvp_chunk <= 0:
            raise ValueError(f"jvp_chunk must be positive, got {jvp_chunk}.")

        n_params = n_param_per_layer * n_layer
        dim4 = 2 ** (num_system_qubits - 1)
        dim5 = 2**num_system_qubits

        def _state_sld_factors(rho: jnp.ndarray):
            rho = _hermitian(rho)
            eigenvalues, eigenvectors = jnp.linalg.eigh(rho)
            eigenvalues = jnp.clip(eigenvalues, a_min=0.0)
            pair_sums = eigenvalues[:, None] + eigenvalues[None, :]
            sqrt_weight = jnp.sqrt(
                jnp.where(
                    pair_sums > EIG_SUM_EPS,
                    2.0 / pair_sums,
                    0.0,
                )
            )
            rho_rank = jnp.sum(eigenvalues > QFIM_EFFECTIVE_RANK_THRESHOLD)
            return eigenvectors, sqrt_weight, rho_rank

        def _feature_block(
            d_rho_block: jnp.ndarray,
            eigenvectors: jnp.ndarray,
            sqrt_weight: jnp.ndarray,
        ) -> jnp.ndarray:
            eigenvectors_dagger = jnp.conjugate(eigenvectors).T
            d_rho_eigenbasis = jax.vmap(
                lambda d_rho: eigenvectors_dagger @ d_rho @ eigenvectors
            )(d_rho_block)
            return jnp.reshape(
                d_rho_eigenbasis * sqrt_weight[None, :, :],
                (d_rho_block.shape[0], -1),
            )

        def joint_qfim_data(theta: jnp.ndarray):
            theta = jnp.asarray(theta, dtype=REAL_DTYPE)
            if theta.shape != (n_params,):
                raise ValueError(
                    f"Expected theta shape {(n_params,)}, got {theta.shape}."
                )

            def rho5_fn(theta_value: jnp.ndarray) -> jnp.ndarray:
                return _hermitian(
                    rho_keep_sequential_dpqc(theta_value, n_layer=n_layer)
                )

            rho5, rho5_jvp = jax.linearize(rho5_fn, theta)
            rho4 = _hermitian(rho4_from_rho5(rho5))

            eigenvectors4, sqrt_weight4, rho_rank4 = _state_sld_factors(rho4)
            eigenvectors5, sqrt_weight5, rho_rank5 = _state_sld_factors(rho5)

            identity_tangents = jnp.eye(n_params, dtype=theta.dtype)
            feature4_blocks = []
            feature5_blocks = []

            for start in range(0, n_params, jvp_chunk):
                tangent_block = identity_tangents[
                    start: min(start + jvp_chunk, n_params), :
                ]

                d_rho5_block = jax.vmap(rho5_jvp)(tangent_block)
                d_rho5_block = 0.5 * (
                    d_rho5_block
                    + jnp.conjugate(jnp.swapaxes(d_rho5_block, -2, -1))
                )
                d_rho4_block = rho4_from_rho5(d_rho5_block)
                d_rho4_block = 0.5 * (
                    d_rho4_block
                    + jnp.conjugate(jnp.swapaxes(d_rho4_block, -2, -1))
                )

                feature4_blocks.append(
                    _feature_block(
                        d_rho4_block,
                        eigenvectors4,
                        sqrt_weight4,
                    )
                )
                feature5_blocks.append(
                    _feature_block(
                        d_rho5_block,
                        eigenvectors5,
                        sqrt_weight5,
                    )
                )

            feature4 = jnp.concatenate(feature4_blocks, axis=0)
            feature5 = jnp.concatenate(feature5_blocks, axis=0)

            if feature4.shape != (n_params, dim4 * dim4):
                raise AssertionError(
                    f"Unexpected F4 feature shape {feature4.shape}."
                )
            if feature5.shape != (n_params, dim5 * dim5):
                raise AssertionError(
                    f"Unexpected F5 feature shape {feature5.shape}."
                )

            F4 = jnp.real(feature4 @ jnp.conjugate(feature4).T)
            F5 = jnp.real(feature5 @ jnp.conjugate(feature5).T)
            F4 = 0.5 * (F4 + F4.T)
            F5 = 0.5 * (F5 + F5.T)

            return F4, F5, rho_rank4, rho_rank5

        return joint_qfim_data


    qfim_rank_reduced_0123_by_layer = {}
    qfim_eigs_reduced_0123_by_layer = {}
    qfim_rho_rank_reduced_0123_by_layer = {}
    qfim_eigsum_reduced_0123_by_layer = {}
    qfim_abs_entry_sum_reduced_0123_by_layer = {}
    qfim_participation_rank_random_by_layer = {}
    qfim_frobenius_norm_sq_random_by_layer = {}
    qfim_largest_eigenvalue_random_by_layer = {}
    qfim_smallest_active_eigenvalue_random_by_layer = {}
    qfim_active_condition_number_random_by_layer = {}

    qfim_rank_reduced_01234_by_layer = {}
    qfim_eigs_reduced_01234_by_layer = {}
    qfim_rho_rank_reduced_01234_by_layer = {}
    qfim_eigsum_reduced_01234_by_layer = {}
    qfim_abs_entry_sum_reduced_01234_by_layer = {}
    qfim_participation_rank_random_keep01234_by_layer = {}
    qfim_frobenius_norm_sq_random_keep01234_by_layer = {}
    qfim_largest_eigenvalue_random_keep01234_by_layer = {}
    qfim_smallest_active_eigenvalue_random_keep01234_by_layer = {}
    qfim_active_condition_number_random_keep01234_by_layer = {}

    for L in tqdm(
        qfim_layer_list,
        desc="Layers (QFIM; keep=(0,1,2,3) and keep=(0,1,2,3,4))",
        unit="layer",
    ):
        n_params = n_param_per_layer * L
        thetas_L = jax.random.uniform(
            jax.random.PRNGKey(QFIM_SAMPLE_SEED_BASE + int(L)),
            shape=(NUM_QFIM_SAMPLES, n_params),
            dtype=REAL_DTYPE,
            minval=jnp.asarray(-jnp.pi, dtype=REAL_DTYPE),
            maxval=jnp.asarray(jnp.pi, dtype=REAL_DTYPE),
        )
        joint_qfim_data_fn = jax.jit(
            make_joint_reduced_qfim_data_fn_for_layer_sequential(
                n_layer=L,
                jvp_chunk=RED_JVP_CHUNK,
            )
        )

        state_lists = [
            {
                "rank": [],
                "eigs": [],
                "rho_rank": [],
                "trace": [],
                "abs_entry_sum": [],
                "participation_rank": [],
                "frobenius_norm_sq": [],
                "largest_eigenvalue": [],
                "smallest_active_eigenvalue": [],
                "active_condition_number": [],
            }
            for _ in range(2)
        ]

        for sample_index in tqdm(
            range(NUM_QFIM_SAMPLES),
            desc=f"QFIM samples for both kept states (L={L})",
            unit="sample",
            leave=False,
        ):
            F4, F5, rho_rank4, rho_rank5 = joint_qfim_data_fn(
                thetas_L[sample_index]
            )
            for state_index, (F, rho_rank) in enumerate(
                ((F4, rho_rank4), (F5, rho_rank5))
            ):
                _, eigs_desc = psd_rank_and_desc_eigs(F)
                eigs_np = jax_to_np(eigs_desc, dtype=NP_REAL_DTYPE)
                spectral = qfim_spectral_summary_from_eigvals(eigs_np)
                values = state_lists[state_index]
                values["rank"].append(int(spectral["qfim_threshold_rank"]))
                values["eigs"].append(eigs_np)
                values["rho_rank"].append(int(jax.device_get(rho_rank)))
                values["trace"].append(spectral["qfim_trace"])
                values["abs_entry_sum"].append(
                    NP_REAL_DTYPE(
                        np.sum(np.abs(jax_to_np(F, dtype=NP_REAL_DTYPE)))
                    )
                )
                values["participation_rank"].append(
                    spectral["qfim_participation_rank"]
                )
                values["frobenius_norm_sq"].append(
                    spectral["qfim_frobenius_norm_sq"]
                )
                values["largest_eigenvalue"].append(
                    spectral["largest_qfim_eigenvalue"]
                )
                values["smallest_active_eigenvalue"].append(
                    spectral["smallest_active_qfim_eigenvalue"]
                )
                values["active_condition_number"].append(
                    spectral["condition_number_active"]
                )

        values4, values5 = state_lists
        qfim_rank_reduced_0123_by_layer[L] = np.asarray(
            values4["rank"], dtype=NP_INT_DTYPE
        )
        qfim_eigs_reduced_0123_by_layer[L] = np.stack(values4["eigs"], axis=0)
        qfim_rho_rank_reduced_0123_by_layer[L] = np.asarray(
            values4["rho_rank"], dtype=NP_INT_DTYPE
        )
        qfim_eigsum_reduced_0123_by_layer[L] = np.asarray(
            values4["trace"], dtype=NP_REAL_DTYPE
        )
        qfim_abs_entry_sum_reduced_0123_by_layer[L] = np.asarray(
            values4["abs_entry_sum"], dtype=NP_REAL_DTYPE
        )
        qfim_participation_rank_random_by_layer[L] = np.asarray(
            values4["participation_rank"], dtype=NP_REAL_DTYPE
        )
        qfim_frobenius_norm_sq_random_by_layer[L] = np.asarray(
            values4["frobenius_norm_sq"], dtype=NP_REAL_DTYPE
        )
        qfim_largest_eigenvalue_random_by_layer[L] = np.asarray(
            values4["largest_eigenvalue"], dtype=NP_REAL_DTYPE
        )
        qfim_smallest_active_eigenvalue_random_by_layer[L] = np.asarray(
            values4["smallest_active_eigenvalue"], dtype=NP_REAL_DTYPE
        )
        qfim_active_condition_number_random_by_layer[L] = np.asarray(
            values4["active_condition_number"], dtype=NP_REAL_DTYPE
        )

        qfim_rank_reduced_01234_by_layer[L] = np.asarray(
            values5["rank"], dtype=NP_INT_DTYPE
        )
        qfim_eigs_reduced_01234_by_layer[L] = np.stack(values5["eigs"], axis=0)
        qfim_rho_rank_reduced_01234_by_layer[L] = np.asarray(
            values5["rho_rank"], dtype=NP_INT_DTYPE
        )
        qfim_eigsum_reduced_01234_by_layer[L] = np.asarray(
            values5["trace"], dtype=NP_REAL_DTYPE
        )
        qfim_abs_entry_sum_reduced_01234_by_layer[L] = np.asarray(
            values5["abs_entry_sum"], dtype=NP_REAL_DTYPE
        )
        qfim_participation_rank_random_keep01234_by_layer[L] = np.asarray(
            values5["participation_rank"], dtype=NP_REAL_DTYPE
        )
        qfim_frobenius_norm_sq_random_keep01234_by_layer[L] = np.asarray(
            values5["frobenius_norm_sq"], dtype=NP_REAL_DTYPE
        )
        qfim_largest_eigenvalue_random_keep01234_by_layer[L] = np.asarray(
            values5["largest_eigenvalue"], dtype=NP_REAL_DTYPE
        )
        qfim_smallest_active_eigenvalue_random_keep01234_by_layer[L] = np.asarray(
            values5["smallest_active_eigenvalue"], dtype=NP_REAL_DTYPE
        )
        qfim_active_condition_number_random_keep01234_by_layer[L] = np.asarray(
            values5["active_condition_number"], dtype=NP_REAL_DTYPE
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
        participation_effective_rank_threshold=np.asarray(
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        participation_effective_rank_eps=np.asarray(
            PARTICIPATION_EFFECTIVE_RANK_EPS,
            dtype=NP_REAL_DTYPE,
        ),
        eig_sum_eps=np.asarray(EIG_SUM_EPS, dtype=NP_REAL_DTYPE),
        qfim_eig_plot_eps=np.asarray(QFIM_EIG_PLOT_EPS, dtype=NP_REAL_DTYPE),
        red_jvp_chunk=np.asarray(RED_JVP_CHUNK, dtype=NP_INT_DTYPE),
        layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
        **_layer_arrays_for_npz(qfim_rank_reduced_0123_by_layer, "rank"),
        **_layer_arrays_for_npz(qfim_eigs_reduced_0123_by_layer, "eigs_desc"),
        **_layer_arrays_for_npz(qfim_rho_rank_reduced_0123_by_layer, "rho_rank"),
        **_layer_arrays_for_npz(qfim_eigsum_reduced_0123_by_layer, "trace"),
        **_layer_arrays_for_npz(qfim_abs_entry_sum_reduced_0123_by_layer, "abs_entry_sum"),
        **_layer_arrays_for_npz(qfim_rank_reduced_0123_by_layer, "threshold_rank"),
        **_layer_arrays_for_npz(
            qfim_participation_rank_random_by_layer,
            "participation_rank",
        ),
        **_layer_arrays_for_npz(
            qfim_frobenius_norm_sq_random_by_layer,
            "frobenius_norm_sq",
        ),
        **_layer_arrays_for_npz(
            qfim_largest_eigenvalue_random_by_layer,
            "largest_eigenvalue",
        ),
        **_layer_arrays_for_npz(
            qfim_smallest_active_eigenvalue_random_by_layer,
            "smallest_active_eigenvalue",
        ),
        **_layer_arrays_for_npz(
            qfim_active_condition_number_random_by_layer,
            "active_condition_number",
        ),
    )

    qfim_random_points_keep01234_result_path = os.path.join(
        qfim_results_dir,
        f"qfim_random_points_{keep_key_5}.npz",
    )

    save_npz_result(
        qfim_random_points_keep01234_result_path,
        h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
        num_qfim_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
        qfim_sample_seed_base=np.asarray(QFIM_SAMPLE_SEED_BASE, dtype=NP_INT_DTYPE),
        qfim_effective_rank_threshold=np.asarray(
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        participation_effective_rank_threshold=np.asarray(
            QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=NP_REAL_DTYPE,
        ),
        participation_effective_rank_eps=np.asarray(
            PARTICIPATION_EFFECTIVE_RANK_EPS,
            dtype=NP_REAL_DTYPE,
        ),
        eig_sum_eps=np.asarray(EIG_SUM_EPS, dtype=NP_REAL_DTYPE),
        qfim_eig_plot_eps=np.asarray(QFIM_EIG_PLOT_EPS, dtype=NP_REAL_DTYPE),
        red_jvp_chunk=np.asarray(RED_JVP_CHUNK, dtype=NP_INT_DTYPE),
        keep_wires=np.asarray(KEEP_WIRES_5, dtype=NP_INT_DTYPE),
        state_label=np.asarray(keep_label_5),
        representation=np.asarray("reduced_keep_01234"),
        layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
        **_layer_arrays_for_npz(qfim_rank_reduced_01234_by_layer, "rank"),
        **_layer_arrays_for_npz(
            qfim_eigs_reduced_01234_by_layer,
            "eigs_desc",
        ),
        **_layer_arrays_for_npz(
            qfim_rho_rank_reduced_01234_by_layer,
            "rho_rank",
        ),
        **_layer_arrays_for_npz(qfim_eigsum_reduced_01234_by_layer, "trace"),
        **_layer_arrays_for_npz(
            qfim_abs_entry_sum_reduced_01234_by_layer,
            "abs_entry_sum",
        ),
        **_layer_arrays_for_npz(
            qfim_rank_reduced_01234_by_layer,
            "threshold_rank",
        ),
        **_layer_arrays_for_npz(
            qfim_participation_rank_random_keep01234_by_layer,
            "participation_rank",
        ),
        **_layer_arrays_for_npz(
            qfim_frobenius_norm_sq_random_keep01234_by_layer,
            "frobenius_norm_sq",
        ),
        **_layer_arrays_for_npz(
            qfim_largest_eigenvalue_random_keep01234_by_layer,
            "largest_eigenvalue",
        ),
        **_layer_arrays_for_npz(
            qfim_smallest_active_eigenvalue_random_keep01234_by_layer,
            "smallest_active_eigenvalue",
        ),
        **_layer_arrays_for_npz(
            qfim_active_condition_number_random_keep01234_by_layer,
            "active_condition_number",
        ),
    )


    qfim_effective_rank_random_points_result_path = os.path.join(
        qfim_results_dir,
        f"qfim_effective_rank_random_points_{keep_key}.npz",
    )
    qfim_effective_rank_random_points_keep01234_result_path = os.path.join(
        qfim_results_dir,
        f"qfim_effective_rank_random_points_{keep_key_5}.npz",
    )

    if RUN_QFIM_EFFECTIVE_RANK_RANDOM_POINTS:
        save_npz_result(
            qfim_effective_rank_random_points_result_path,
            h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
            num_qfim_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
            qfim_sample_seed_base=np.asarray(
                QFIM_SAMPLE_SEED_BASE,
                dtype=NP_INT_DTYPE,
            ),
            qfim_effective_rank_threshold=np.asarray(
                QFIM_EFFECTIVE_RANK_THRESHOLD,
                dtype=NP_REAL_DTYPE,
            ),
            participation_effective_rank_threshold=np.asarray(
                QFIM_EFFECTIVE_RANK_THRESHOLD,
                dtype=NP_REAL_DTYPE,
            ),
            participation_effective_rank_eps=np.asarray(
                PARTICIPATION_EFFECTIVE_RANK_EPS,
                dtype=NP_REAL_DTYPE,
            ),
            layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
            **_layer_arrays_for_npz(
                qfim_rank_reduced_0123_by_layer,
                "threshold_rank",
            ),
            **_layer_arrays_for_npz(
                qfim_participation_rank_random_by_layer,
                "participation_rank",
            ),
            **_layer_arrays_for_npz(
                qfim_eigsum_reduced_0123_by_layer,
                "trace",
            ),
            **_layer_arrays_for_npz(
                qfim_frobenius_norm_sq_random_by_layer,
                "frobenius_norm_sq",
            ),
            **_layer_arrays_for_npz(
                qfim_largest_eigenvalue_random_by_layer,
                "largest_eigenvalue",
            ),
            **_layer_arrays_for_npz(
                qfim_smallest_active_eigenvalue_random_by_layer,
                "smallest_active_eigenvalue",
            ),
            **_layer_arrays_for_npz(
                qfim_active_condition_number_random_by_layer,
                "active_condition_number",
            ),
        )
        save_npz_result(
            qfim_effective_rank_random_points_keep01234_result_path,
            h_param=np.asarray(h_param, dtype=NP_REAL_DTYPE),
            num_qfim_samples=np.asarray(NUM_QFIM_SAMPLES, dtype=NP_INT_DTYPE),
            qfim_sample_seed_base=np.asarray(
                QFIM_SAMPLE_SEED_BASE,
                dtype=NP_INT_DTYPE,
            ),
            qfim_effective_rank_threshold=np.asarray(
                QFIM_EFFECTIVE_RANK_THRESHOLD,
                dtype=NP_REAL_DTYPE,
            ),
            participation_effective_rank_threshold=np.asarray(
                QFIM_EFFECTIVE_RANK_THRESHOLD,
                dtype=NP_REAL_DTYPE,
            ),
            participation_effective_rank_eps=np.asarray(
                PARTICIPATION_EFFECTIVE_RANK_EPS,
                dtype=NP_REAL_DTYPE,
            ),
            keep_wires=np.asarray(KEEP_WIRES_5, dtype=NP_INT_DTYPE),
            state_label=np.asarray(keep_label_5),
            layers=np.asarray(qfim_layer_list, dtype=NP_INT_DTYPE),
            **_layer_arrays_for_npz(
                qfim_rank_reduced_01234_by_layer,
                "threshold_rank",
            ),
            **_layer_arrays_for_npz(
                qfim_participation_rank_random_keep01234_by_layer,
                "participation_rank",
            ),
            **_layer_arrays_for_npz(
                qfim_eigsum_reduced_01234_by_layer,
                "trace",
            ),
            **_layer_arrays_for_npz(
                qfim_frobenius_norm_sq_random_keep01234_by_layer,
                "frobenius_norm_sq",
            ),
            **_layer_arrays_for_npz(
                qfim_largest_eigenvalue_random_keep01234_by_layer,
                "largest_eigenvalue",
            ),
            **_layer_arrays_for_npz(
                qfim_smallest_active_eigenvalue_random_keep01234_by_layer,
                "smallest_active_eigenvalue",
            ),
            **_layer_arrays_for_npz(
                qfim_active_condition_number_random_keep01234_by_layer,
                "active_condition_number",
            ),
        )


    if not include_optimization_path:
        print(
            "Saved random-point QFIM numerical results to: "
            f"{qfim_results_dir}"
        )
        return

    # ============================================================
    # Optimization-path QFIM eigenvalues: compute and save numerical results
    # ============================================================
    def compute_joint_qfim_eigs_history_by_layer(
        theta_samples_by_layer: dict,
        layers,
        *,
        jvp_chunk: int = RED_JVP_CHUNK,
    ):
        eig_histories = ({}, {})

        for L in tqdm(
            layers,
            desc=(
                "QFIM eigenvalue history; "
                "keep=(0,1,2,3) and keep=(0,1,2,3,4)"
            ),
            unit="layer",
        ):
            L = int(L)
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

            n_runs_L, n_times_L, n_params_L = theta_samples.shape
            expected_n_params = n_param_per_layer * L
            if n_params_L != expected_n_params:
                raise ValueError(
                    f"Expected M=14L={expected_n_params} parameters for L={L}, "
                    f"got {n_params_L}."
                )

            joint_qfim_data_fn = jax.jit(
                make_joint_reduced_qfim_data_fn_for_layer_sequential(
                    n_layer=L,
                    jvp_chunk=jvp_chunk,
                )
            )
            eigs_L = tuple(
                np.full(
                    (n_runs_L, n_times_L, n_params_L),
                    np.nan,
                    dtype=NP_REAL_DTYPE,
                )
                for _ in range(2)
            )

            for run_idx in tqdm(
                range(n_runs_L),
                desc=f"QFIM-eigenvalue runs (L={L})",
                unit="run",
                leave=False,
            ):
                for time_idx in range(n_times_L):
                    F4, F5, _, _ = joint_qfim_data_fn(
                        jnp.asarray(
                            theta_samples[run_idx, time_idx],
                            dtype=REAL_DTYPE,
                        )
                    )
                    for state_index, F in enumerate((F4, F5)):
                        eigs_desc = psd_desc_eigs(F)
                        eigs_np = jax_to_np(eigs_desc, dtype=NP_REAL_DTYPE)
                        eigs_L[state_index][run_idx, time_idx, :] = eigs_np

            for state_index in range(2):
                eig_histories[state_index][L] = eigs_L[state_index]

        return eig_histories


    (
        qfim_eigs_history_optimization_path_by_layer,
        qfim_eigs_history_optimization_path_keep01234_by_layer,
    ) = compute_joint_qfim_eigs_history_by_layer(
        theta_sample_traces_by_layer,
        vqe_layer_list,
        jvp_chunk=RED_JVP_CHUNK,
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

    qfim_eigs_history_keep01234_result_path = os.path.join(
        qfim_results_dir,
        f"qfim_eigs_history_optimization_path_{keep_key_5}.npz",
    )

    save_npz_result(
        qfim_eigs_history_keep01234_result_path,
        sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
        keep_wires=np.asarray(KEEP_WIRES_5, dtype=NP_INT_DTYPE),
        state_label=np.asarray(keep_label_5),
        **{
            f"L{int(L)}": arr
            for L, arr
            in qfim_eigs_history_optimization_path_keep01234_by_layer.items()
        },
    )

    qfim_trace_history_optimization_path_by_layer = {
        int(L): np.sum(np.asarray(arr, dtype=NP_REAL_DTYPE), axis=2)
        for L, arr in qfim_eigs_history_optimization_path_by_layer.items()
    }
    qfim_trace_history_optimization_path_keep01234_by_layer = {
        int(L): np.sum(np.asarray(arr, dtype=NP_REAL_DTYPE), axis=2)
        for L, arr
        in qfim_eigs_history_optimization_path_keep01234_by_layer.items()
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

    qfim_trace_history_keep01234_result_path = os.path.join(
        qfim_results_dir,
        f"qfim_trace_history_optimization_path_{keep_key_5}.npz",
    )

    save_npz_result(
        qfim_trace_history_keep01234_result_path,
        sample_iters=np.asarray(sample_iters, dtype=NP_INT_DTYPE),
        layers=np.asarray(vqe_layer_list, dtype=NP_INT_DTYPE),
        keep_wires=np.asarray(KEEP_WIRES_5, dtype=NP_INT_DTYPE),
        state_label=np.asarray(keep_label_5),
        **{
            f"L{int(L)}": arr
            for L, arr
            in qfim_trace_history_optimization_path_keep01234_by_layer.items()
        },
    )

    print(
        "Saved keep0123 and keep01234 numerical results to: "
        f"{numerical_results_dir}"
    )
    print(
        "Circuit drawing was skipped. To draw the saved optimized circuits, run "
        "src/dpqc/DPQC_overparam_draw_circuits.py separately."
    )


def main():
    run_qfim()


if __name__ == "__main__":
    main()
