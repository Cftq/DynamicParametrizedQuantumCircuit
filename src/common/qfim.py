#!/usr/bin/env python
# coding: utf-8
"""QFIM algebra helpers shared by DPQC and Unitary-PQC scripts."""

from __future__ import annotations

from typing import Optional, Tuple

import config_overparam as cfg
import jax
import jax.numpy as jnp


REAL_DTYPE = jnp.float64
COMPLEX_DTYPE = jnp.complex128


def hermitian(a: jnp.ndarray) -> jnp.ndarray:
    """Return the Hermitian part of a square matrix."""
    return 0.5 * (a + jnp.conjugate(a.T))


def _threshold_value(
    evals: jnp.ndarray,
    threshold: Optional[float] = None,
) -> jnp.ndarray:
    if threshold is None:
        threshold = cfg.QFIM_EFFECTIVE_RANK_THRESHOLD
    return jnp.asarray(threshold, dtype=evals.dtype)


@jax.jit
def _mask_psd_eigvals_default_threshold(evals: jnp.ndarray):
    threshold = jnp.asarray(cfg.QFIM_EFFECTIVE_RANK_THRESHOLD, dtype=evals.dtype)
    masked = jnp.where(evals > threshold, evals, jnp.zeros_like(evals))
    return masked, threshold


def mask_psd_eigvals_for_rank(
    evals: jnp.ndarray,
    *,
    threshold: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Zero eigenvalues that are not counted in the effective rank."""
    if threshold is None:
        return _mask_psd_eigvals_default_threshold(evals)

    threshold_jnp = rank_threshold_from_eigvals(evals, threshold=threshold)
    masked = jnp.where(evals > threshold_jnp, evals, jnp.zeros_like(evals))
    return masked, threshold_jnp


def rank_threshold_from_eigvals(
    evals: jnp.ndarray,
    *,
    threshold: Optional[float] = None,
) -> jnp.ndarray:
    """Return the effective-rank threshold for a PSD eigenvalue array."""
    return _threshold_value(evals, threshold)


def effective_rank_from_eigvals(
    evals: jnp.ndarray,
    *,
    threshold: Optional[float] = None,
) -> jnp.ndarray:
    """Count PSD eigenvalues strictly larger than the effective-rank threshold."""
    threshold_jnp = rank_threshold_from_eigvals(evals, threshold=threshold)
    return jnp.sum(evals > threshold_jnp)


def participation_effective_rank_from_eigvals(
    evals: jnp.ndarray,
    *,
    eps: float = 1e-30,
) -> jnp.ndarray:
    """Return the participation effective rank (sum lambda)^2 / sum lambda^2."""
    evals = jnp.clip(jnp.real(evals), a_min=0.0)
    eigsum = jnp.sum(evals)
    eigsq_sum = jnp.sum(evals**2)
    eps_jnp = jnp.asarray(eps, dtype=evals.dtype)
    return jnp.where(eigsq_sum > eps_jnp, (eigsum**2) / eigsq_sum, 0.0)


def psd_eigvals(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return clipped ascending eigenvalues of a Hermitian PSD matrix."""
    return jnp.clip(jnp.linalg.eigvalsh(hermitian(matrix)), a_min=0.0)


def psd_eigvals_desc(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return clipped descending eigenvalues of a Hermitian PSD matrix."""
    return psd_eigvals(matrix)[::-1]


def hermitian_eigvals_desc(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return signed descending eigenvalues of a Hermitian matrix."""
    evals = jnp.real(jnp.linalg.eigvalsh(hermitian(matrix)))
    return evals[::-1]


def effective_abs_rank_from_eigvals(
    evals: jnp.ndarray,
    *,
    threshold: Optional[float] = None,
) -> jnp.ndarray:
    """Count signed eigenvalues whose absolute value exceeds the threshold."""
    threshold_jnp = rank_threshold_from_eigvals(evals, threshold=threshold)
    return jnp.sum(jnp.abs(evals) > threshold_jnp)


def matrix_rank_psd(
    matrix: jnp.ndarray,
    *,
    threshold: Optional[float] = None,
) -> jnp.ndarray:
    """Effective rank of a Hermitian PSD matrix."""
    return effective_rank_from_eigvals(psd_eigvals(matrix), threshold=threshold)


def pure_state_qfim_from_state_jacobian(
    psi: jnp.ndarray,
    jacobian: jnp.ndarray,
) -> jnp.ndarray:
    """Pure-state QFIM from |psi(theta)> and d|psi>/dtheta."""
    jtj = jnp.matmul(jnp.conjugate(jacobian).T, jacobian)
    b = jnp.matmul(jnp.conjugate(psi), jacobian)
    qfim_matrix = 4.0 * jnp.real(jtj - jnp.outer(jnp.conjugate(b), b))
    return 0.5 * (qfim_matrix + qfim_matrix.T)


def make_pure_state_qfim_fn(psi_fn):
    """Create a pure-state QFIM function from a state-vector function."""
    jac_psi = jax.jacfwd(psi_fn)

    @jax.jit
    def qfim_pure(theta: jnp.ndarray) -> jnp.ndarray:
        psi = psi_fn(theta)
        jacobian = jac_psi(theta)
        return pure_state_qfim_from_state_jacobian(psi, jacobian)

    return qfim_pure


def mixed_state_qfim_from_rho_jvp(
    rho: jnp.ndarray,
    rho_jvp,
    theta: jnp.ndarray,
    *,
    eig_sum_eps: float,
    jvp_chunk: int,
) -> jnp.ndarray:
    """
    Mixed-state SLD QFIM from rho(theta) and a JAX linearized JVP callable.

    rho_jvp is the second return value of jax.linearize(rho_fn, theta).
    """
    rho = hermitian(rho)
    evals, evecs = jnp.linalg.eigh(rho)
    evals = jnp.clip(evals, a_min=0.0)

    eval_sum = evals[:, None] + evals[None, :]
    sqrt_weight = jnp.sqrt(
        jnp.where(
            eval_sum > eig_sum_eps,
            2.0 / eval_sum,
            0.0,
        )
    )

    evecs_dag = jnp.conjugate(evecs).T
    num_params = int(theta.shape[0])
    dim_vec = int(rho.shape[0] * rho.shape[1])
    eye = jnp.eye(num_params, dtype=theta.dtype)

    def to_eig_basis(drho: jnp.ndarray) -> jnp.ndarray:
        return evecs_dag @ drho @ evecs

    blocks = []
    for start in range(0, num_params, int(jvp_chunk)):
        basis_block = eye[start : min(start + int(jvp_chunk), num_params), :]
        drho_block = jax.vmap(rho_jvp)(basis_block)
        drho_block = 0.5 * (
            drho_block + jnp.conjugate(jnp.swapaxes(drho_block, 1, 2))
        )

        coeff_block = jax.vmap(to_eig_basis)(drho_block)
        flat_block = jnp.reshape(
            coeff_block * sqrt_weight[None, :, :],
            (basis_block.shape[0], dim_vec),
        )
        blocks.append(flat_block)

    coeffs_flat = jnp.concatenate(blocks, axis=0)
    qfim_matrix = jnp.real(coeffs_flat @ jnp.conjugate(coeffs_flat).T)
    return 0.5 * (qfim_matrix + qfim_matrix.T)


def hilbert_schmidt_metric_from_rho_jvp(
    rho: jnp.ndarray,
    rho_jvp,
    theta: jnp.ndarray,
    *,
    jvp_chunk: int,
) -> jnp.ndarray:
    """
    Hilbert-Schmidt tangent Gram matrix from rho(theta).

    The matrix elements are

        G_ij = Re Tr[(partial_i rho)(partial_j rho)].

    The implementation symmetrizes density-matrix derivatives as Hermitian and
    evaluates the equivalent Frobenius inner product
    Re Tr[(partial_i rho)^dagger (partial_j rho)] for numerical stability.
    """
    rho = hermitian(rho)
    num_params = int(theta.shape[0])
    dim_vec = int(rho.shape[0] * rho.shape[1])
    eye = jnp.eye(num_params, dtype=theta.dtype)

    blocks = []
    for start in range(0, num_params, int(jvp_chunk)):
        basis_block = eye[start : min(start + int(jvp_chunk), num_params), :]
        drho_block = jax.vmap(rho_jvp)(basis_block)
        drho_block = 0.5 * (
            drho_block + jnp.conjugate(jnp.swapaxes(drho_block, 1, 2))
        )
        flat_block = jnp.reshape(drho_block, (basis_block.shape[0], dim_vec))
        blocks.append(flat_block)

    derivs_flat = jnp.concatenate(blocks, axis=0)
    hs_matrix = jnp.real(derivs_flat @ jnp.conjugate(derivs_flat).T)
    return 0.5 * (hs_matrix + hs_matrix.T)


def make_mixed_state_qfim_fn(
    rho_fn,
    *,
    eig_sum_eps: float,
    jvp_chunk: int,
):
    """Create a mixed-state SLD-QFIM function from rho(theta)."""

    def qfim_mixed(theta: jnp.ndarray) -> jnp.ndarray:
        rho, rho_jvp = jax.linearize(rho_fn, theta)
        return mixed_state_qfim_from_rho_jvp(
            rho,
            rho_jvp,
            theta,
            eig_sum_eps=eig_sum_eps,
            jvp_chunk=jvp_chunk,
        )

    return qfim_mixed


def make_hilbert_schmidt_metric_fn(
    rho_fn,
    *,
    jvp_chunk: int,
):
    """Create G_ij = Re Tr[(partial_i rho)(partial_j rho)]."""

    def hs_metric(theta: jnp.ndarray) -> jnp.ndarray:
        rho, rho_jvp = jax.linearize(rho_fn, theta)
        return hilbert_schmidt_metric_from_rho_jvp(
            rho,
            rho_jvp,
            theta,
            jvp_chunk=jvp_chunk,
        )

    return hs_metric
