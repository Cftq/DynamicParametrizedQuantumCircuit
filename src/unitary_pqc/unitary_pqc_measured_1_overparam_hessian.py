#!/usr/bin/env python
# coding: utf-8
"""HESSIAN stage for the measurement-outcome-1 unitary PQC.

Numerical dependencies are imported only when a computation is requested.
"""
from __future__ import annotations

from typing import Optional


def main(argv=None) -> int:
    if __package__:
        from . import unitary_pqc_measured_1_overparam_cli as _cli
    else:
        import unitary_pqc_measured_1_overparam_cli as _cli

    args = _cli.parse_stage_args("hessian", argv)
    routed_status = _cli.maybe_relaunch_stage_in_wsl("hessian", args, __file__, argv)
    if routed_status is not None:
        return routed_status
    result = run_unitary_pqc_hessian_stage(
        h_param=args.h_param,
        analysis_batch_size=args.analysis_batch_size,
        device=args.device,
    )
    print(f"Saved Hessian numerical results to: {result['hessian_results_dir']}")
    return 0


def run_unitary_pqc_hessian_stage(
    *,
    h_param: Optional[float] = None,
    analysis_batch_size: Optional[int] = None,
    device: Optional[str] = None,
) -> dict:
    """Compute signed Hessians independently of VQE and QFIM stages."""
    if __package__:
        from .unitary_pqc_measured_1_overparam_cli import load_stage_common
    else:
        from unitary_pqc_measured_1_overparam_cli import load_stage_common

    _common = load_stage_common("hessian", device)
    effective_analysis_batch_size = _common._resolve_analysis_batch_size(
        analysis_batch_size
    )
    _common.configure_unitary_pqc_overparam(h_value=h_param)
    archive_path = run_random_hessian_analysis(
        analysis_batch_size=effective_analysis_batch_size,
    )
    result = _common.collect_unitary_pqc_result()
    result["analysis_batch_size"] = effective_analysis_batch_size
    result["hessian_result_path"] = archive_path
    return result


def make_energy_hessian_fn_for_layer(num_layers: int):
    """Return the signed energy Hessian, symmetrized before storage."""
    if __package__:
        from . import unitary_pqc_measured_1_overparam_common as _common
    else:
        import unitary_pqc_measured_1_overparam_common as _common
    energy_fn = _common.make_energy_fn_for_layer(num_layers)
    raw_hessian_fn = _common.jax.hessian(energy_fn)

    @_common.jit
    def energy_hessian(theta: jnp.ndarray):
        matrix = _common.jnp.asarray(raw_hessian_fn(theta), dtype=_common.REAL_DTYPE)
        return (matrix + matrix.T) * 0.5

    return energy_hessian


def make_energy_hessian_eigvals_fn_for_layer(num_layers: int):
    if __package__:
        from . import unitary_pqc_measured_1_overparam_common as _common
    else:
        import unitary_pqc_measured_1_overparam_common as _common
    hessian_fn = make_energy_hessian_fn_for_layer(num_layers)

    @_common.jit
    def hessian_eigvals(theta: jnp.ndarray):
        return _common.hermitian_eigvals_desc(hessian_fn(theta))

    return hessian_eigvals


def hessian_rank_and_condition_from_eigvals(eigvals: jnp.ndarray):
    """Return fixed-cutoff Hessian rank and active-spectrum condition number."""
    if __package__:
        from . import unitary_pqc_measured_1_overparam_common as _common
    else:
        import unitary_pqc_measured_1_overparam_common as _common
    abs_eigvals = _common.jnp.abs(_common.jnp.asarray(eigvals, dtype=_common.REAL_DTYPE))
    threshold = _common.jnp.asarray(_common.QFIM_EFFECTIVE_RANK_THRESHOLD, dtype=_common.REAL_DTYPE)
    active = abs_eigvals >= threshold
    rank_value = _common.jnp.sum(active, dtype=_common.jnp.int64)
    largest_active = _common.jnp.max(_common.jnp.where(active, abs_eigvals, 0.0))
    smallest_active = _common.jnp.min(_common.jnp.where(active, abs_eigvals, _common.jnp.inf))
    condition_number = _common.jnp.where(
        rank_value > 0,
        largest_active / smallest_active,
        _common.jnp.asarray(_common.jnp.nan, dtype=_common.REAL_DTYPE),
    )
    return rank_value, condition_number


def make_hessian_analysis_batch_runner(num_layers: int):
    """Return rank, active-spectrum condition number, and the full Hessian.

    Raw signed matrices permit threshold-free curvature analysis at plotting
    time. Existing summary metrics are computed from the same matrix.
    """
    if __package__:
        from . import unitary_pqc_measured_1_overparam_common as _common
    else:
        import unitary_pqc_measured_1_overparam_common as _common
    hessian_fn = make_energy_hessian_fn_for_layer(int(num_layers))

    def metrics_one(theta: jnp.ndarray):
        matrix = hessian_fn(theta)
        rank_value, condition_number = hessian_rank_and_condition_from_eigvals(
            _common.hermitian_eigvals_desc(matrix)
        )
        return rank_value, condition_number, matrix

    return _common.jax.jit(_common.jax.vmap(metrics_one))


def run_random_hessian_analysis(
    *,
    analysis_batch_size: Optional[int] = None,
) -> str:
    """Save signed Hessians at the same seeded random points used by QFIM."""
    if __package__:
        from . import unitary_pqc_measured_1_overparam_common as _common
    else:
        import unitary_pqc_measured_1_overparam_common as _common


    effective_analysis_batch_size = _common._resolve_analysis_batch_size(
        analysis_batch_size
    )
    _common.qfim_dense_until_layer = _common.cfg.UNITARY_PQC_QFIM_DENSE_UNTIL_LAYER
    _common.qfim_max_layer = _common.cfg.UNITARY_PQC_QFIM_MAX_LAYER
    _common.qfim_sparse_step = _common.cfg.UNITARY_PQC_QFIM_SPARSE_STEP
    _common.qfim_layer_list = _common.build_layer_list(
        _common.qfim_max_layer,
        _common.qfim_dense_until_layer,
        _common.qfim_sparse_step,
    )
    if not _common.qfim_layer_list:
        raise ValueError(
            "qfim_layer_list is empty. Check "
            "UNITARY_PQC_QFIM_MAX_LAYER, "
            "UNITARY_PQC_QFIM_DENSE_UNTIL_LAYER, and "
            "UNITARY_PQC_QFIM_SPARSE_STEP."
        )
    _common.NUM_QFIM_SAMPLES = _common.cfg.NUM_QFIM_SAMPLES
    _common.QFIM_SAMPLE_SEED_BASE = _common.cfg.UNITARY_PQC_QFIM_SAMPLE_SEED_BASE
    _common.QFIM_EFFECTIVE_RANK_THRESHOLD = _common.cfg.QFIM_EFFECTIVE_RANK_THRESHOLD
    _common.hessian_random_thetas_by_layer = {}
    _common.hessian_by_layer = {}
    _common.hessian_rank_by_layer = {}
    _common.hessian_condition_by_layer = {}
    _common._ensure_unitary_result_dirs()
    # tqdm: Layers (Hessian)
    for L in _common.tqdm(_common.qfim_layer_list, desc="Layers (Hessian)", unit="layer"):
        num_params = _common.num_params_per_layer * L

        key_L = _common.jax.random.PRNGKey(_common.QFIM_SAMPLE_SEED_BASE + int(L))
        thetas_L = _common.jax.random.uniform(
            key_L,
            shape=(_common.NUM_QFIM_SAMPLES, num_params),
            minval=-_common.jnp.pi,
            maxval=_common.jnp.pi,
            dtype=_common.REAL_DTYPE,
        )
        _common.hessian_random_thetas_by_layer[L] = _common.np.asarray(_common.jax.device_get(thetas_L), dtype=_common.NP_REAL_DTYPE)

        # --------------------------
        # Full signed energy Hessian at the same random points as the QFIM.
        # Rank counts |lambda_i| >= the fixed QFIM rank threshold.  The
        # condition number is max(|lambda_i|) / min(|lambda_i|) over that
        # active spectrum and is NaN only when the rank is zero.
        # --------------------------
        hessian_batch_runner = make_hessian_analysis_batch_runner(L)
        hessian_ranks, hessian_conditions, hessian_matrices = _common._evaluate_analysis_in_batches(
            _common.hessian_random_thetas_by_layer[L],
            hessian_batch_runner,
            batch_size=effective_analysis_batch_size,
            description=(
                f"Hessian batches (L={L}, "
                f"batch={effective_analysis_batch_size})"
            ),
        )
        _common.hessian_by_layer[L] = _common.np.asarray(
            hessian_matrices, dtype=_common.NP_REAL_DTYPE
        )
        _common.hessian_rank_by_layer[L] = _common.np.asarray(
            hessian_ranks,
            dtype=_common.NP_INT_DTYPE,
        )
        _common.hessian_condition_by_layer[L] = _common.np.asarray(
            hessian_conditions,
            dtype=_common.NP_REAL_DTYPE,
        )

        del hessian_batch_runner
        _common._release_jax_compilation_cache()

    _common.save_npz_result(
        _common.os.path.join(_common.hessian_results_dir, "hessian_random_points.npz"),
        schema_version=_common.np.asarray(
            _common.HESSIAN_RANDOM_SCHEMA_VERSION,
            dtype=_common.NP_INT_DTYPE,
        ),
        analysis_kind=_common.np.asarray("random_points"),
        ansatz=_common.np.asarray(_common.ANSATZ_NAME),
        measurement_outcome=_common.np.asarray(_common.MEASUREMENT_OUTCOME, dtype=_common.NP_INT_DTYPE),
        h_param=_common.np.asarray(_common.h_param, dtype=_common.NP_REAL_DTYPE),
        layers=_common.np.asarray(_common.qfim_layer_list, dtype=_common.NP_INT_DTYPE),
        num_hessian_samples=_common.np.asarray(_common.NUM_QFIM_SAMPLES, dtype=_common.NP_INT_DTYPE),
        hessian_sample_seed_base=_common.np.asarray(_common.QFIM_SAMPLE_SEED_BASE, dtype=_common.NP_INT_DTYPE),
        hessian_rank_threshold=_common.np.asarray(
            _common.QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=_common.NP_REAL_DTYPE,
        ),
        hessian_rank_definition=_common.np.asarray(_common.HESSIAN_RANK_DEFINITION),
        hessian_condition_number_definition=_common.np.asarray(
            _common.HESSIAN_CONDITION_NUMBER_DEFINITION
        ),
        num_params_per_layer=_common.np.asarray(
            _common.num_params_per_layer,
            dtype=_common.NP_INT_DTYPE,
        ),
        analysis_batch_size=_common.np.asarray(
            effective_analysis_batch_size,
            dtype=_common.NP_INT_DTYPE,
        ),
        hessian_matrix_definition=_common.np.asarray("d2 E(theta) / dtheta_i dtheta_j"),
        **{
            f"L{int(L)}_hessian": arr
            for L, arr in _common.hessian_by_layer.items()
        },
        **{
            f"L{int(L)}_theta": _common.np.asarray(
                _common.hessian_random_thetas_by_layer[L], dtype=_common.NP_REAL_DTYPE
            )
            for L in _common.hessian_by_layer
        },
        **{
            f"L{int(L)}_rank": arr
            for L, arr in _common.hessian_rank_by_layer.items()
        },
        **{
            f"L{int(L)}_condition_number": arr
            for L, arr in _common.hessian_condition_by_layer.items()
        },
    )
    return _common.os.path.join(_common.hessian_results_dir, "hessian_random_points.npz")


if __name__ == "__main__":
    raise SystemExit(main())
