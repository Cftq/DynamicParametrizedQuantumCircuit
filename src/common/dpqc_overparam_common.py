#!/usr/bin/env python
# coding: utf-8
"""Shared DPQC overparameterization helpers used by compute and visualize."""

import os
from copy import copy as shallow_copy
from functools import reduce
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import hamiltonian as _hamiltonian
from qfim import rank_threshold_from_eigvals


REAL_DTYPE = jnp.float64
COMPLEX_DTYPE = jnp.complex128
NP_REAL_DTYPE = np.float64
NP_INT_DTYPE = np.int64

build_H_matrix_jax = _hamiltonian.build_H_matrix_jax
hamiltonian_terms = _hamiltonian.hamiltonian_terms

__all__ = [
    "COMPLEX_DTYPE",
    "NP_INT_DTYPE",
    "NP_REAL_DTYPE",
    "REAL_DTYPE",
    "_normalize_index_list",
    "_thr_tag",
    "_time_index_from_iteration",
    "build_H_matrix_jax",
    "build_layer_list",
    "hamiltonian_terms",
    "ket0_density",
    "load_npz_result",
    "make_parameter_free_qiskit_for_drawing",
    "qfim_grad_alignment_at_point",
    "qfim_grad_alignment_one_to_table",
    "qg_layer",
    "rho_zero_state",
    "save_circuit_matplotlib_png",
    "tc_to_qiskit_qc",
    "threshold_psd_eigvals_for_rank",
    "U_rz",
]

PARAMETER_FREE_GATE_LABELS = {
    "rz": r"$R_z$",
    "rx": r"$R_x$",
    "ry": r"$R_y$",
    "rxx": r"$R_{xx}$",
    "crx": r"$CR_x$",
    "cry": r"$CR_y$",
    "crz": r"$CR_z$",
    "cx": r"$CX$",
    "x": r"$X$",
    "z": r"$Z$",
    "h": r"$H$",
}


def build_layer_list(max_layer: int, dense_until_layer: int, sparse_step: int):
    dense_end = min(dense_until_layer, max_layer)
    return list(range(1, dense_end + 1)) + list(
        range(dense_end + sparse_step, max_layer + 1, sparse_step)
    )


def _kron_all(mats):
    return reduce(jnp.kron, mats)


def qg_layer(c, q0: int, q1: int, p: jnp.ndarray) -> None:
    c.rz(q0, theta=p[0])
    c.rz(q1, theta=p[1])
    c.rxx(q0, q1, theta=p[2])


def U_rz(theta: jnp.ndarray) -> jnp.ndarray:
    th = jnp.asarray(theta, dtype=REAL_DTYPE)
    return jnp.array(
        [[jnp.exp(-0.5j * th), 0.0], [0.0, jnp.exp(0.5j * th)]],
        dtype=COMPLEX_DTYPE,
    )


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


def tc_to_qiskit_qc(
    tc_circ,
    num_qubits: Optional[int] = None,
) -> "QuantumCircuit":
    """Convert a TensorCircuit circuit into a Qiskit QuantumCircuit."""
    try:
        return tc_circ.to_qiskit()
    except AttributeError:
        pass
    except Exception:
        pass

    if num_qubits is None:
        raise ValueError("Pass num_qubits explicitly for QIR-to-Qiskit conversion.")

    import tensorcircuit as tc

    return tc.translation.qir2qiskit(tc_circ.to_qir(), int(num_qubits))


def make_parameter_free_qiskit_for_drawing(qc) -> "QuantumCircuit":
    """Return a drawing-only Qiskit circuit with gate parameters hidden."""
    from qiskit import QuantumCircuit

    if qc.num_clbits > 0:
        qc_draw = QuantumCircuit(qc.num_qubits, qc.num_clbits, name=qc.name)
    else:
        qc_draw = QuantumCircuit(qc.num_qubits, name=qc.name)

    qc_draw.global_phase = 0.0

    for inst in qc.data:
        try:
            operation = inst.operation
            qargs = inst.qubits
            cargs = inst.clbits
        except AttributeError:
            operation, qargs, cargs = inst

        try:
            op_draw = operation.to_mutable()
        except AttributeError:
            try:
                op_draw = operation.copy()
            except Exception:
                op_draw = shallow_copy(operation)

        gate_name = str(op_draw.name).lower()

        if gate_name in PARAMETER_FREE_GATE_LABELS:
            op_draw.label = PARAMETER_FREE_GATE_LABELS[gate_name]
        elif op_draw.params:
            op_draw.label = rf"${gate_name}$"

        q_indices = [qc.find_bit(q).index for q in qargs]
        c_indices = [qc.find_bit(c).index for c in cargs]

        qc_draw.append(op_draw, q_indices, c_indices)

    return qc_draw


def save_circuit_matplotlib_png(
    tc_circ,
    outpath: str,
    num_qubits: Optional[int] = None,
    *,
    dpi: int = 600,
    pad_inches: float = 0.02,
    save_png: bool = True,
    save_pdf: bool = False,
    hide_params: bool = True,
) -> None:
    """Save a TensorCircuit circuit drawing through Qiskit/matplotlib."""
    import matplotlib.pyplot as plt
    from qiskit.visualization import circuit_drawer

    qc = tc_to_qiskit_qc(tc_circ, num_qubits=num_qubits)

    if hide_params:
        qc_for_draw = make_parameter_free_qiskit_for_drawing(qc)
        drawer_style = {"displaytext": PARAMETER_FREE_GATE_LABELS}
    else:
        qc_for_draw = qc
        drawer_style = None

    fig = circuit_drawer(
        qc_for_draw,
        output="mpl",
        fold=-1,
        style=drawer_style,
    )

    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    root, ext = os.path.splitext(outpath)
    if ext.lower() == ".png":
        png_path = outpath
    else:
        png_path = root + ".png"

    if save_png:
        fig.savefig(
            png_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=pad_inches,
        )

    if save_pdf:
        fig.savefig(
            root + ".pdf",
            bbox_inches="tight",
            pad_inches=pad_inches,
        )

    plt.close(fig)


def _jax_to_np(x, dtype=None):
    arr = np.asarray(jax.device_get(x))
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _normalize_index_list(indices, n):
    if indices is None:
        return list(range(n))

    normalized = []

    for idx in indices:
        idx = int(idx)

        if idx < 0:
            idx = n + idx

        normalized.append(idx)

    if any((idx < 0 or idx >= n) for idx in normalized):
        raise IndexError("Index out of range.")

    return normalized


def qfim_grad_alignment_at_point(
    theta,
    grad,
    qfim_fn,
    *,
    sort_desc=True,
    norm_eps=1e-24,
):
    """Compute QFIM eigenvalues and gradient weights at one parameter point."""
    theta = jnp.asarray(theta, dtype=REAL_DTYPE)
    grad = jnp.asarray(grad, dtype=REAL_DTYPE).reshape((-1,))

    F = qfim_fn(theta)
    F = 0.5 * (F + jnp.conjugate(F.T))

    if int(F.shape[0]) != int(grad.shape[0]):
        raise ValueError(
            f"Dimension mismatch: F.shape={F.shape}, grad.shape={grad.shape}"
        )

    evals, evecs = jnp.linalg.eigh(F)
    evals = jnp.clip(jnp.real(evals), a_min=0.0)

    grad_for_projection = grad.astype(evecs.dtype)
    coeffs = jnp.conjugate(evecs).T @ grad_for_projection
    coeff_abs2 = jnp.real(coeffs * jnp.conjugate(coeffs))
    denom = jnp.sum(coeff_abs2)

    weights = jnp.where(
        denom > norm_eps,
        coeff_abs2 / denom,
        jnp.full_like(coeff_abs2, jnp.nan),
    )

    if sort_desc:
        order = jnp.argsort(evals)[::-1]
        evals = evals[order]
        weights = weights[order]
        coeffs = coeffs[order]
        coeff_abs2 = coeff_abs2[order]

    evals_np = _jax_to_np(evals, dtype=NP_REAL_DTYPE)
    weights_np = _jax_to_np(weights, dtype=NP_REAL_DTYPE)
    coeff_abs2_np = _jax_to_np(coeff_abs2, dtype=NP_REAL_DTYPE)

    return {
        "evals": evals_np,
        "weights": weights_np,
        "coeffs": np.asarray(jax.device_get(coeffs)),
        "coeff_abs2": coeff_abs2_np,
        "eig_index": np.arange(1, evals_np.size + 1, dtype=NP_INT_DTYPE),
        "grad_weight_denominator": NP_REAL_DTYPE(jax.device_get(denom)),
    }


def qfim_grad_alignment_one_to_table(
    alignment,
    *,
    layer=None,
    run=None,
    time_index=None,
    iteration=None,
):
    """Convert one-point QFIM-gradient alignment output into table arrays."""
    n = alignment["evals"].size

    layer_value = -1 if layer is None else int(layer)
    run_value = -1 if run is None else int(run)
    time_value = -1 if time_index is None else int(time_index)
    iter_value = -1 if iteration is None else int(iteration)

    return {
        "lambda": np.asarray(alignment["evals"], dtype=NP_REAL_DTYPE),
        "w_grad": np.asarray(alignment["weights"], dtype=NP_REAL_DTYPE),
        "coeff_abs2": np.asarray(alignment["coeff_abs2"], dtype=NP_REAL_DTYPE),
        "eig_index": np.asarray(alignment["eig_index"], dtype=NP_INT_DTYPE),
        "layer": np.full(n, layer_value, dtype=NP_INT_DTYPE),
        "run": np.full(n, run_value, dtype=NP_INT_DTYPE),
        "time_index": np.full(n, time_value, dtype=NP_INT_DTYPE),
        "iteration": np.full(n, iter_value, dtype=NP_INT_DTYPE),
    }


def _time_index_from_iteration(sample_iters_for_labels, target_iteration: int):
    """Return the sampled-time index for a requested optimization iteration."""
    sample_iters_arr = np.asarray(sample_iters_for_labels, dtype=NP_INT_DTYPE)
    target_iteration = int(target_iteration)
    hit = np.where(sample_iters_arr == target_iteration)[0]

    if hit.size == 0:
        raise ValueError(
            f"iteration {target_iteration} is not included in sample_iters. "
            "Add it to sample_iters before running the VQE optimization loop."
        )

    return int(hit[0])


@jax.jit
def threshold_psd_eigvals_for_rank(
    evals: jnp.ndarray,
) -> jnp.ndarray:
    """Return the fixed effective-rank threshold for QFIM eigenvalues."""
    return rank_threshold_from_eigvals(evals)


def _thr_tag(thr: float) -> str:
    return f"{float(thr):.0e}".replace("+", "")

