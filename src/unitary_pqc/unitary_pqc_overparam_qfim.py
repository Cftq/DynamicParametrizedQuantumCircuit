#!/usr/bin/env python
# coding: utf-8
"""QFIM stage for the 60-parameter Cartan unitary PQC.

Numerical dependencies are imported only when a computation is requested.
"""
from __future__ import annotations

from typing import Optional


def main(argv=None) -> int:
    if __package__:
        from . import unitary_pqc_overparam_cli as _cli
    else:
        import unitary_pqc_overparam_cli as _cli

    args = _cli.parse_stage_args("qfim", argv)
    routed_status = _cli.maybe_relaunch_stage_in_wsl("qfim", args, __file__, argv)
    if routed_status is not None:
        return routed_status
    result = run_unitary_pqc_qfim_stage(
        h_param=args.h_param,
        analysis_batch_size=args.analysis_batch_size,
        include_optimization_path=not args.random_only,
        device=args.device,
    )
    print(f"Saved QFIM numerical results to: {result['qfim_results_dir']}")
    return 0


def run_unitary_pqc_qfim_stage(
    *,
    h_param: Optional[float] = None,
    analysis_batch_size: Optional[int] = None,
    include_optimization_path: bool = True,
    device: Optional[str] = None,
) -> dict:
    """Compute QFIM/HS, reusing saved training samples only for the path."""
    if __package__:
        from .unitary_pqc_overparam_cli import load_stage_common
    else:
        from unitary_pqc_overparam_cli import load_stage_common

    _common = load_stage_common("qfim", device)
    effective_analysis_batch_size = _common._resolve_analysis_batch_size(
        analysis_batch_size
    )
    _common.configure_unitary_pqc_overparam(h_value=h_param)
    vqe_input_path = None
    if include_optimization_path:
        # Validate the archive before any potentially expensive matrix work.
        vqe_input_path = _common.load_unitary_vqe_samples()
        print(f"Loaded saved float64 VQE samples: {vqe_input_path}", flush=True)
    run_random_qfim_analysis(
        make_plots=False,
        analysis_batch_size=effective_analysis_batch_size,
    )
    if include_optimization_path:
        run_optimization_path_qfim_analysis(
            analysis_batch_size=effective_analysis_batch_size,
        )
    result = _common.collect_unitary_pqc_result()
    result["analysis_batch_size"] = effective_analysis_batch_size
    result["vqe_input_path"] = vqe_input_path
    return result


def make_pure_qfim_matrix_fn_for_layer(num_layers: int):
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common
    def psi_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return _common.statevector_sequential_unitary_pqc(
            theta,
            num_layers=num_layers,
        )

    return _common.make_pure_state_qfim_fn(psi_fn)


def make_reduced_qfim_matrix_fn_for_layer_sequential(
    num_layers: int,
    *,
    keep_wires=None,
    jvp_chunk: int = None,
):
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common
    if keep_wires is None:
        keep_wires = _common.KEEP_WIRES
    if jvp_chunk is None:
        jvp_chunk = _common.RED_JVP_CHUNK
    keep_wires = tuple(int(w) for w in keep_wires)
    if keep_wires != _common.KEEP_WIRES:
        raise NotImplementedError("Only keep_wires=(0,1,2,3) is supported here.")

    @_common.jit
    def rho_sub_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return _common.rho_keep_sequential_unitary_pqc(
            theta,
            num_layers=num_layers,
            keep_wires=keep_wires,
        )

    return _common.make_mixed_state_qfim_fn(
        rho_sub_fn,
        eig_sum_eps=_common.EIG_SUM_EPS,
        jvp_chunk=jvp_chunk,
    )


def make_reduced_hs_matrix_fn_for_layer_sequential(
    num_layers: int,
    *,
    keep_wires=None,
    jvp_chunk: int = None,
):
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common
    if keep_wires is None:
        keep_wires = _common.KEEP_WIRES
    if jvp_chunk is None:
        jvp_chunk = _common.RED_JVP_CHUNK
    @_common.jax.jit
    def rho_keep_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return _common._hermitian(
            _common.rho_keep_sequential_unitary_pqc(
                theta,
                num_layers=num_layers,
                keep_wires=keep_wires,
            )
        )

    return _common.make_hilbert_schmidt_metric_fn(
        rho_keep_fn,
        jvp_chunk=jvp_chunk,
    )


def make_pure_full_hs_matrix_fn_for_layer(
    num_layers: int,
    *,
    jvp_chunk: int = None,
):
    """HS tangent Gram matrix derived from the full pure-state QFIM.

    For a normalized pure state, ``F_Q = 2 G_HS``.  Deriving the HS matrix
    from the statevector QFIM avoids differentiating a 32x32 density matrix.
    ``jvp_chunk`` remains in the signature for call compatibility.
    """
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common
    if jvp_chunk is None:
        jvp_chunk = _common.RED_JVP_CHUNK
    del jvp_chunk
    pure_qfim_fn = make_pure_qfim_matrix_fn_for_layer(num_layers=num_layers)

    @_common.jax.jit
    def pure_hs(theta: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * pure_qfim_fn(theta)

    return pure_hs


def make_reduced_qfim_rank_fn_for_layer(
    num_layers: int,
    keep_wires=None,
    reverse_axes: bool = False,
    jvp_chunk: int = None,
):
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common
    if keep_wires is None:
        keep_wires = _common.KEEP_WIRES
    if jvp_chunk is None:
        jvp_chunk = _common.RED_JVP_CHUNK
    qfim_fn = make_reduced_qfim_matrix_fn_for_layer_sequential(
        num_layers=num_layers,
        keep_wires=keep_wires,
        jvp_chunk=jvp_chunk,
    )

    def rank_reduced(theta: jnp.ndarray) -> jnp.ndarray:
        F = qfim_fn(theta)
        return _common._matrix_rank_psd(F)

    return rank_reduced


def make_qfim_eigvals_fn_for_layer(
    num_layers: int,
    *,
    keep_wires=None,
    jvp_chunk: int = None,
):
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common
    if keep_wires is None:
        keep_wires = _common.KEEP_WIRES
    if jvp_chunk is None:
        jvp_chunk = _common.RED_JVP_CHUNK
    qfim_fn = make_reduced_qfim_matrix_fn_for_layer_sequential(
        num_layers=num_layers,
        keep_wires=keep_wires,
        jvp_chunk=jvp_chunk,
    )

    @_common.jit
    def qfim_eigvals(theta: jnp.ndarray):
        F = qfim_fn(theta)
        return _common.psd_eigvals_desc(F)

    return qfim_eigvals


def make_qfim_rank_fn_for_layer(
    num_layers: int,
    *,
    keep_wires=None,
    jvp_chunk: int = None,
):
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common
    if keep_wires is None:
        keep_wires = _common.KEEP_WIRES
    if jvp_chunk is None:
        jvp_chunk = _common.RED_JVP_CHUNK
    qfim_eigvals_fn = make_qfim_eigvals_fn_for_layer(
        num_layers=num_layers,
        keep_wires=keep_wires,
        jvp_chunk=jvp_chunk,
    )

    @_common.jit
    def qfim_rank(theta: jnp.ndarray):
        return _common.effective_rank_from_eigvals(qfim_eigvals_fn(theta))

    return qfim_rank


def make_hs_eigvals_fn_for_layer(
    num_layers: int,
    *,
    keep_wires=None,
    jvp_chunk: int = None,
):
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common
    if keep_wires is None:
        keep_wires = _common.KEEP_WIRES
    if jvp_chunk is None:
        jvp_chunk = _common.RED_JVP_CHUNK
    hs_fn = make_reduced_hs_matrix_fn_for_layer_sequential(
        num_layers=num_layers,
        keep_wires=keep_wires,
        jvp_chunk=jvp_chunk,
    )

    @_common.jit
    def hs_eigvals(theta: jnp.ndarray):
        G = hs_fn(theta)
        return _common.psd_eigvals_desc(G)

    return hs_eigvals


def make_hs_rank_fn_for_layer(
    num_layers: int,
    *,
    keep_wires=None,
    jvp_chunk: int = None,
):
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common
    if keep_wires is None:
        keep_wires = _common.KEEP_WIRES
    if jvp_chunk is None:
        jvp_chunk = _common.RED_JVP_CHUNK
    hs_eigvals_fn = make_hs_eigvals_fn_for_layer(
        num_layers=num_layers,
        keep_wires=keep_wires,
        jvp_chunk=jvp_chunk,
    )

    @_common.jit
    def hs_rank(theta: jnp.ndarray):
        return _common.effective_rank_from_eigvals(hs_eigvals_fn(theta))

    return hs_rank


def _make_psd_analysis_batch_runner(matrix_fn):
    """Batch PSD-matrix rank, masked/raw spectrum, and threshold metrics."""
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common

    def metrics_one(theta: jnp.ndarray):
        eigs_desc = _common.psd_eigvals_desc(matrix_fn(theta))
        masked_desc, threshold = _common.threshold_psd_eigvals_for_rank(eigs_desc)
        rank_value = _common.jnp.sum(eigs_desc > threshold)
        return rank_value, masked_desc, eigs_desc, threshold

    return _common.jax.jit(_common.jax.vmap(metrics_one))


def _make_psd_eigenvalue_batch_runner(matrix_fn):
    """Batch raw PSD eigenspectra without rank or threshold calculations."""
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common

    def eigenvalues_one(theta: jnp.ndarray):
        return (_common.psd_eigvals_desc(matrix_fn(theta)),)

    return _common.jax.jit(_common.jax.vmap(eigenvalues_one))


def make_qfim_analysis_batch_runner(
    num_layers: int,
    *,
    keep_wires=None,
    jvp_chunk: int = None,
    representation: str = "reduced",
    eigenvalues_only: bool = False,
):
    """Create a fixed-shape QFIM ``jit(vmap(...))`` analysis runner."""
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common
    if keep_wires is None:
        keep_wires = _common.KEEP_WIRES
    if jvp_chunk is None:
        jvp_chunk = _common.RED_JVP_CHUNK
    if representation == "pure_full":
        matrix_fn = make_pure_qfim_matrix_fn_for_layer(int(num_layers))
    elif representation == "reduced":
        matrix_fn = make_reduced_qfim_matrix_fn_for_layer_sequential(
            num_layers=int(num_layers),
            keep_wires=keep_wires,
            jvp_chunk=jvp_chunk,
        )
    else:
        raise ValueError("representation must be 'pure_full' or 'reduced'.")
    if eigenvalues_only:
        return _make_psd_eigenvalue_batch_runner(matrix_fn)
    return _make_psd_analysis_batch_runner(matrix_fn)


def make_hs_analysis_batch_runner(
    num_layers: int,
    *,
    keep_wires=None,
    jvp_chunk: int = None,
    representation: str = "reduced",
    eigenvalues_only: bool = False,
):
    """Create a fixed-shape HS ``jit(vmap(...))`` analysis runner."""
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common
    if keep_wires is None:
        keep_wires = _common.KEEP_WIRES
    if jvp_chunk is None:
        jvp_chunk = _common.RED_JVP_CHUNK
    if representation == "pure_full":
        matrix_fn = make_pure_full_hs_matrix_fn_for_layer(
            num_layers=int(num_layers),
            jvp_chunk=jvp_chunk,
        )
    elif representation == "reduced":
        matrix_fn = make_reduced_hs_matrix_fn_for_layer_sequential(
            num_layers=int(num_layers),
            keep_wires=keep_wires,
            jvp_chunk=jvp_chunk,
        )
    else:
        raise ValueError("representation must be 'pure_full' or 'reduced'.")
    if eigenvalues_only:
        return _make_psd_eigenvalue_batch_runner(matrix_fn)
    return _make_psd_analysis_batch_runner(matrix_fn)


def compute_qfim_eigenvalue_history_by_layer(
    theta_samples_by_layer: dict,
    layers,
    *,
    keep_wires=None,
    jvp_chunk: int = None,
    representation: str = "reduced",
    batch_size: Optional[int] = None,
):
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common
    if keep_wires is None:
        keep_wires = _common.KEEP_WIRES
    if jvp_chunk is None:
        jvp_chunk = _common.RED_JVP_CHUNK
    eigs_history_by_layer = {}

    for L in _common.tqdm(
        layers,
        desc="QFIM eigenvalue history along optimization path",
        unit="layer",
    ):
        L_int = int(L)
        if theta_samples_by_layer.get(L_int) is None:
            continue

        theta_samples = _common.np.asarray(
            theta_samples_by_layer[L_int],
            dtype=_common.NP_REAL_DTYPE,
        )

        if theta_samples.ndim != 3:
            raise ValueError(
                "theta_samples must have shape "
                "(num_runs, num_sample_iters, num_params)."
            )

        num_runs, num_times, num_params = theta_samples.shape
        batch_runner = make_qfim_analysis_batch_runner(
            num_layers=L_int,
            keep_wires=keep_wires,
            jvp_chunk=jvp_chunk,
            representation=representation,
            eigenvalues_only=True,
        )
        (eigs_flat,) = _common._evaluate_analysis_in_batches(
            theta_samples.reshape((-1, num_params)),
            batch_runner,
            batch_size=batch_size,
            description=(
                f"QFIM eigenvalue batches ({representation}, L={L_int}, "
                f"batch={_common._resolve_analysis_batch_size(batch_size)})"
            ),
        )
        eigs_L = _common.np.asarray(eigs_flat, dtype=_common.NP_REAL_DTYPE).reshape(
            (num_runs, num_times, num_params)
        )

        eigs_history_by_layer[L_int] = eigs_L

        del batch_runner
        _common._release_jax_compilation_cache()

    return eigs_history_by_layer


def compute_hs_eigenvalue_history_by_layer(
    theta_samples_by_layer: dict,
    layers,
    *,
    keep_wires=None,
    jvp_chunk: int = None,
    representation: str = "reduced",
    batch_size: Optional[int] = None,
):
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common
    if keep_wires is None:
        keep_wires = _common.KEEP_WIRES
    if jvp_chunk is None:
        jvp_chunk = _common.RED_JVP_CHUNK
    eigs_history_by_layer = {}

    for L in _common.tqdm(
        layers,
        desc="HS eigenvalue history along optimization path",
        unit="layer",
    ):
        L_int = int(L)
        if theta_samples_by_layer.get(L_int) is None:
            continue

        theta_samples = _common.np.asarray(
            theta_samples_by_layer[L_int],
            dtype=_common.NP_REAL_DTYPE,
        )

        if theta_samples.ndim != 3:
            raise ValueError(
                "theta_samples must have shape "
                "(num_runs, num_sample_iters, num_params)."
            )

        num_runs, num_times, num_params = theta_samples.shape
        batch_runner = make_hs_analysis_batch_runner(
            num_layers=L_int,
            keep_wires=keep_wires,
            jvp_chunk=jvp_chunk,
            representation=representation,
            eigenvalues_only=True,
        )
        (eigs_flat,) = _common._evaluate_analysis_in_batches(
            theta_samples.reshape((-1, num_params)),
            batch_runner,
            batch_size=batch_size,
            description=(
                f"HS eigenvalue batches ({representation}, L={L_int}, "
                f"batch={_common._resolve_analysis_batch_size(batch_size)})"
            ),
        )
        eigs_L = _common.np.asarray(eigs_flat, dtype=_common.NP_REAL_DTYPE).reshape(
            (num_runs, num_times, num_params)
        )

        eigs_history_by_layer[L_int] = eigs_L

        del batch_runner
        _common._release_jax_compilation_cache()

    return eigs_history_by_layer


def run_random_qfim_analysis(
    *,
    make_plots: bool = False,
    analysis_batch_size: Optional[int] = None,
) -> None:
    """Compute only random-point QFIM/HS metrics with fixed-size JAX batches."""
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common


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
    # QFIM rank (pure + reduced)
    #   - evaluated at RANDOM points in parameter space (per layer)
    #
    # Reduced(mixed):
    #   - keep=(0,1,2,3): trace out the added center ancilla qubit 4
    #   - rho_keep_sequential_unitary_pqc returns the 16x16 reduced system state
    #   - d rho / dﾎｸ via linearize + chunked JVPs
    #
    # keep01234 / pure(full):
    #   - Computed on the full 5-qubit pure state for every configured layer.
    #   - No wire is traced, so the pure-state QFIM is the efficient SLD-QFIM.
    #
    # ============================================================

    # ------------------------------
    # Subsystem to keep for reduced QFIM
    # ------------------------------
    # We keep the original 4-qubit system and trace out the center ancilla qubit.
    _common.KEEP_WIRES = _common.SYSTEM_WIRES
    assert _common.KEEP_WIRES == (0, 1, 2, 3), "Reduced QFIM keeps only the original system qubits."

    # ------------------------------
    # Rank / numerical knobs
    # ------------------------------
    _common.QFIM_EFFECTIVE_RANK_THRESHOLD = _common.cfg.QFIM_EFFECTIVE_RANK_THRESHOLD
    _common.EIG_SUM_EPS = _common.cfg.EIG_SUM_EPS

    # For eigenvalue plots (log safety; DISPLAY ONLY for true zeros)
    _common.QFIM_EIG_PLOT_EPS = _common.cfg.QFIM_EIG_PLOT_EPS

    # ------------------------------
    # Random sampling knobs
    # ------------------------------
    _common.NUM_QFIM_SAMPLES = _common.cfg.NUM_QFIM_SAMPLES
    _common.QFIM_SAMPLE_SEED_BASE = _common.cfg.UNITARY_PQC_QFIM_SAMPLE_SEED_BASE

    # ------------------------------
    # Pure QFIM compute cutoff (kept)
    # ------------------------------
    # Compute pure-full metrics for every configured layer.  The previous
    # cutoff produced an asymmetric pure/reduced result set.
    _common.PURE_QFIM_LAYER_THRESHOLD = max(_common.qfim_layer_list, default=0) + 1

    # ------------------------------
    # Reduced-QFIM derivative chunk size
    # ------------------------------
    _common.RED_JVP_CHUNK = _common.cfg.RED_JVP_CHUNK


    # ------------------------------
    # Pure(full) QFIM matrix function on all 5 qubits
    # ------------------------------


    # ------------------------------
    # Reduced(mixed) QFIM matrix function
    #   - Propagate full 5-qubit state.
    #   - Trace out the center ancilla.
    #   - Build SLD-QFIM from the reduced 4-qubit system density matrix.
    # ------------------------------


    # Backward-compatible wrapper (kept signature)


    # ============================================================
    # Compute QFIM ranks + eigenvalue distributions at RANDOM samples per layer
    # ============================================================
    _common.qfim_rank_pure_by_layer = {}        # L -> (NUM_QFIM_SAMPLES,) or None
    _common.qfim_rank_reduced_by_layer = {}     # L -> (NUM_QFIM_SAMPLES,) for keep=(0..3)
    _common.qfim_random_thetas_by_layer = {}

    _common.qfim_eigs_pure_by_layer = {}        # L -> (NUM_QFIM_SAMPLES, num_params) or None
    _common.qfim_eigs_reduced_by_layer = {}     # L -> (NUM_QFIM_SAMPLES, num_params)
    # Raw clipped spectra feed the canonical keep archives.  The dictionaries
    # above retain the historical threshold-masked representation.
    qfim_eigs_pure_raw_by_layer = {}
    qfim_eigs_reduced_raw_by_layer = {}

    # fixed thresholds used in rank computation
    _common.qfim_thresh_pure_by_layer = {}      # L -> (NUM_QFIM_SAMPLES,) or None
    _common.qfim_thresh_reduced_by_layer = {}   # L -> (NUM_QFIM_SAMPLES,)
    _common.hs_rank_reduced_by_layer = {}       # L -> (NUM_QFIM_SAMPLES,)
    _common.hs_eigs_reduced_by_layer = {}       # L -> (NUM_QFIM_SAMPLES, num_params)
    _common.hs_thresh_reduced_by_layer = {}     # L -> (NUM_QFIM_SAMPLES,)
    hs_rank_pure_by_layer = {}
    hs_eigs_pure_by_layer = {}
    hs_thresh_pure_by_layer = {}

    _common.qfim_eigs_dir = _common.os.path.join(_common.qfim_fig_dir, "eigs")
    _common.qfim_eigs_pure_dir = _common.os.path.join(_common.qfim_eigs_dir, "pure_full")
    _common.qfim_eigs_reduced_0123_dir = _common.os.path.join(_common.qfim_eigs_dir, "reduced_keep_0123")
    _common.qfim_rank_dir = _common.os.path.join(_common.qfim_fig_dir, "rank")
    _common.qfim_rank_random_dir = _common.os.path.join(_common.qfim_rank_dir, "random_points")
    _common.hs_eigs_dir = _common.os.path.join(_common.hs_fig_dir, "eigs")
    _common.hs_eigs_reduced_0123_dir = _common.os.path.join(_common.hs_eigs_dir, "reduced_keep_0123")
    _common.hs_rank_dir = _common.os.path.join(_common.hs_fig_dir, "rank")
    _common.hs_rank_random_dir = _common.os.path.join(_common.hs_rank_dir, "random_points")

    _common.os.makedirs(_common.qfim_rank_random_dir, exist_ok=True)
    _common.os.makedirs(_common.hs_eigs_dir, exist_ok=True)
    _common.os.makedirs(_common.hs_eigs_reduced_0123_dir, exist_ok=True)
    _common.os.makedirs(_common.hs_rank_random_dir, exist_ok=True)
    _common._ensure_unitary_result_dirs()

    # tqdm: Layers (QFIM)
    for L in _common.tqdm(_common.qfim_layer_list, desc="Layers (QFIM)", unit="layer"):
        num_params = _common.num_params_per_layer * L

        key_L = _common.jax.random.PRNGKey(_common.QFIM_SAMPLE_SEED_BASE + int(L))
        thetas_L = _common.jax.random.uniform(
            key_L,
            shape=(_common.NUM_QFIM_SAMPLES, num_params),
            minval=-_common.jnp.pi,
            maxval=_common.jnp.pi,
            dtype=_common.REAL_DTYPE,
        )
        _common.qfim_random_thetas_by_layer[L] = _common.np.asarray(_common.jax.device_get(thetas_L), dtype=_common.NP_REAL_DTYPE)

        # --------------------------
        # Reduced QFIM (keep 0..3)
        # --------------------------
        red_qfim_batch_runner = make_qfim_analysis_batch_runner(
            num_layers=L,
            keep_wires=_common.KEEP_WIRES,
            jvp_chunk=_common.RED_JVP_CHUNK,
            representation="reduced",
        )
        (
            reduced_qfim_ranks,
            reduced_qfim_eigs_masked,
            reduced_qfim_eigs_raw,
            reduced_qfim_thresholds,
        ) = _common._evaluate_analysis_in_batches(
            _common.qfim_random_thetas_by_layer[L],
            red_qfim_batch_runner,
            batch_size=effective_analysis_batch_size,
            description=(
                f"Reduced QFIM batches (L={L}, "
                f"batch={effective_analysis_batch_size})"
            ),
        )
        _common.qfim_rank_reduced_by_layer[L] = _common.np.asarray(
            reduced_qfim_ranks,
            dtype=_common.NP_INT_DTYPE,
        )
        _common.qfim_eigs_reduced_by_layer[L] = _common.np.asarray(
            reduced_qfim_eigs_masked,
            dtype=_common.NP_REAL_DTYPE,
        )
        qfim_eigs_reduced_raw_by_layer[L] = _common.np.asarray(
            reduced_qfim_eigs_raw,
            dtype=_common.NP_REAL_DTYPE,
        )
        _common.qfim_thresh_reduced_by_layer[L] = _common.np.asarray(
            reduced_qfim_thresholds,
            dtype=_common.NP_REAL_DTYPE,
        )

        del red_qfim_batch_runner
        _common._release_jax_compilation_cache()

        # --------------------------
        # Hilbert-Schmidt tangent Gram matrix (keep 0..3)
        #   G_ij = Re Tr[(partial_i rho)(partial_j rho)]
        #   computed via the equivalent Frobenius form after Hermitian symmetrization.
        # --------------------------
        red_hs_batch_runner = make_hs_analysis_batch_runner(
            num_layers=L,
            keep_wires=_common.KEEP_WIRES,
            jvp_chunk=_common.RED_JVP_CHUNK,
            representation="reduced",
        )
        (
            reduced_hs_ranks,
            reduced_hs_eigs_masked,
            _,
            reduced_hs_thresholds,
        ) = _common._evaluate_analysis_in_batches(
            _common.qfim_random_thetas_by_layer[L],
            red_hs_batch_runner,
            batch_size=effective_analysis_batch_size,
            description=(
                f"Reduced HS batches (L={L}, "
                f"batch={effective_analysis_batch_size})"
            ),
        )
        _common.hs_rank_reduced_by_layer[L] = _common.np.asarray(
            reduced_hs_ranks,
            dtype=_common.NP_INT_DTYPE,
        )
        _common.hs_eigs_reduced_by_layer[L] = _common.np.asarray(
            reduced_hs_eigs_masked,
            dtype=_common.NP_REAL_DTYPE,
        )
        _common.hs_thresh_reduced_by_layer[L] = _common.np.asarray(
            reduced_hs_thresholds,
            dtype=_common.NP_REAL_DTYPE,
        )

        del red_hs_batch_runner
        _common._release_jax_compilation_cache()

        # For a normalized pure state, G_HS = F_Q / 2 exactly.  Evaluate the
        # pure QFIM once in batches and derive both pure result families from
        # its spectrum, avoiding a duplicate state-Jacobian calculation.
        pure_qfim_batch_runner = make_qfim_analysis_batch_runner(
            num_layers=L,
            jvp_chunk=_common.RED_JVP_CHUNK,
            representation="pure_full",
        )
        (
            pure_qfim_ranks,
            pure_qfim_eigs_masked,
            pure_qfim_eigs_raw,
            pure_qfim_thresholds,
        ) = _common._evaluate_analysis_in_batches(
            _common.qfim_random_thetas_by_layer[L],
            pure_qfim_batch_runner,
            batch_size=effective_analysis_batch_size,
            description=(
                f"Pure(full) QFIM/HS batches (L={L}, "
                f"batch={effective_analysis_batch_size})"
            ),
        )
        if L >= _common.PURE_QFIM_LAYER_THRESHOLD:
            _common.qfim_rank_pure_by_layer[L] = None
            _common.qfim_eigs_pure_by_layer[L] = None
            qfim_eigs_pure_raw_by_layer[L] = None
            _common.qfim_thresh_pure_by_layer[L] = None
        else:
            _common.qfim_rank_pure_by_layer[L] = _common.np.asarray(
                pure_qfim_ranks,
                dtype=_common.NP_INT_DTYPE,
            )
            _common.qfim_eigs_pure_by_layer[L] = _common.np.asarray(
                pure_qfim_eigs_masked,
                dtype=_common.NP_REAL_DTYPE,
            )
            qfim_eigs_pure_raw_by_layer[L] = _common.np.asarray(
                pure_qfim_eigs_raw,
                dtype=_common.NP_REAL_DTYPE,
            )
            _common.qfim_thresh_pure_by_layer[L] = _common.np.asarray(
                pure_qfim_thresholds,
                dtype=_common.NP_REAL_DTYPE,
            )

        pure_hs_eigs_raw = 0.5 * _common.np.asarray(
            pure_qfim_eigs_raw,
            dtype=_common.NP_REAL_DTYPE,
        )
        pure_hs_thresholds = _common.np.full(
            (pure_hs_eigs_raw.shape[0],),
            _common.QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=_common.NP_REAL_DTYPE,
        )
        pure_hs_eigs_masked = _common.np.where(
            pure_hs_eigs_raw > pure_hs_thresholds[:, None],
            pure_hs_eigs_raw,
            _common.NP_REAL_DTYPE(0.0),
        )
        pure_hs_ranks = _common.np.sum(
            pure_hs_eigs_raw > pure_hs_thresholds[:, None],
            axis=1,
        )
        hs_rank_pure_by_layer[L] = _common.np.asarray(
            pure_hs_ranks,
            dtype=_common.NP_INT_DTYPE,
        )
        hs_eigs_pure_by_layer[L] = _common.np.asarray(
            pure_hs_eigs_masked,
            dtype=_common.NP_REAL_DTYPE,
        )
        hs_thresh_pure_by_layer[L] = _common.np.asarray(
            pure_hs_thresholds, dtype=_common.NP_REAL_DTYPE
        )

        del pure_qfim_batch_runner
        _common._release_jax_compilation_cache()

        if make_plots:
            _common._save_qfim_eigs_violinplot_by_index(
                _common.hs_eigs_reduced_by_layer[L],
                title=rf"HS tangent Gram eigenvalues at {_common.NUM_QFIM_SAMPLES} random points (L={L})",
                outpath=_common.os.path.join(_common.hs_eigs_reduced_0123_dir, f"L{L}_reduced_0123.pdf"),
                rank_thresholds=_common.hs_thresh_reduced_by_layer[L],
                ylabel="HS tangent Gram eigenvalue",
            )
            _common.plot_style.save_eigenvalue_histograms_by_trial(
                _common.hs_eigs_reduced_by_layer[L],
                outdir=_common.os.path.join(
                    _common.hs_eigs_reduced_0123_dir,
                    "histograms",
                    "random_points",
                    f"L{L}",
                ),
                matrix_tag="unitary_pqc_hs_gram",
                matrix_label="HS tangent Gram",
                num_layers=L,
                context_tag="random",
                context_label="random point",
                condition_tag="reduced0123",
                condition_label="reduced keep=(0,1,2,3)",
                color="C3",
            )


    _common.save_npz_result(
        _common.os.path.join(_common.qfim_results_dir, "qfim_random_points.npz"),
        h_param=_common.np.asarray(_common.h_param, dtype=_common.NP_REAL_DTYPE),
        num_qfim_samples=_common.np.asarray(_common.NUM_QFIM_SAMPLES, dtype=_common.NP_INT_DTYPE),
        qfim_sample_seed_base=_common.np.asarray(_common.QFIM_SAMPLE_SEED_BASE, dtype=_common.NP_INT_DTYPE),
        qfim_effective_rank_threshold=_common.np.asarray(
            _common.QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=_common.NP_REAL_DTYPE,
        ),
        layers=_common.np.asarray(_common.qfim_layer_list, dtype=_common.NP_INT_DTYPE),
        pure_qfim_layer_threshold=_common.np.asarray(
            _common.PURE_QFIM_LAYER_THRESHOLD,
            dtype=_common.NP_INT_DTYPE,
        ),
        red_jvp_chunk=_common.np.asarray(_common.RED_JVP_CHUNK, dtype=_common.NP_INT_DTYPE),
        analysis_batch_size=_common.np.asarray(
            effective_analysis_batch_size,
            dtype=_common.NP_INT_DTYPE,
        ),
        **{
            f"L{int(L)}_theta": arr
            for L, arr in _common.qfim_random_thetas_by_layer.items()
        },
        **{
            f"L{int(L)}_rank_reduced": arr
            for L, arr in _common.qfim_rank_reduced_by_layer.items()
        },
        **{
            f"L{int(L)}_eigs_reduced_desc": arr
            for L, arr in _common.qfim_eigs_reduced_by_layer.items()
        },
        **{
            f"L{int(L)}_rank_threshold_reduced": arr
            for L, arr in _common.qfim_thresh_reduced_by_layer.items()
        },
        **{
            f"L{int(L)}_rank_pure": arr
            for L, arr in _common.qfim_rank_pure_by_layer.items()
            if arr is not None
        },
        **{
            f"L{int(L)}_eigs_pure_desc": arr
            for L, arr in _common.qfim_eigs_pure_by_layer.items()
            if arr is not None
        },
        **{
            f"L{int(L)}_rank_threshold_pure": arr
            for L, arr in _common.qfim_thresh_pure_by_layer.items()
            if arr is not None
        },
    )

    # Canonical per-keep archives mirror the DPQC core rank/eigenvalue/trace
    # naming.  The historical combined archive remains available to loaders.
    _common.qfim_random_result_paths_by_keep = _common.save_qfim_random_point_results_by_keep(
        layers=_common.qfim_layer_list,
        theta_by_layer=_common.qfim_random_thetas_by_layer,
        rank_keep0123_by_layer=_common.qfim_rank_reduced_by_layer,
        eigs_keep0123_by_layer=qfim_eigs_reduced_raw_by_layer,
        threshold_keep0123_by_layer=_common.qfim_thresh_reduced_by_layer,
        rank_keep01234_by_layer=_common.qfim_rank_pure_by_layer,
        eigs_keep01234_by_layer=qfim_eigs_pure_raw_by_layer,
        threshold_keep01234_by_layer=_common.qfim_thresh_pure_by_layer,
        analysis_batch_size=effective_analysis_batch_size,
    )

    _common.save_npz_result(
        _common.os.path.join(_common.hs_results_dir, "hs_random_points_reduced_0123.npz"),
        h_param=_common.np.asarray(_common.h_param, dtype=_common.NP_REAL_DTYPE),
        num_hs_samples=_common.np.asarray(_common.NUM_QFIM_SAMPLES, dtype=_common.NP_INT_DTYPE),
        hs_sample_seed_base=_common.np.asarray(_common.QFIM_SAMPLE_SEED_BASE, dtype=_common.NP_INT_DTYPE),
        hs_effective_rank_threshold=_common.np.asarray(
            _common.QFIM_EFFECTIVE_RANK_THRESHOLD,
            dtype=_common.NP_REAL_DTYPE,
        ),
        layers=_common.np.asarray(_common.qfim_layer_list, dtype=_common.NP_INT_DTYPE),
        red_jvp_chunk=_common.np.asarray(_common.RED_JVP_CHUNK, dtype=_common.NP_INT_DTYPE),
        analysis_batch_size=_common.np.asarray(
            effective_analysis_batch_size,
            dtype=_common.NP_INT_DTYPE,
        ),
        **{
            f"L{int(L)}_theta": arr
            for L, arr in _common.qfim_random_thetas_by_layer.items()
        },
        **{
            f"L{int(L)}_rank": arr
            for L, arr in _common.hs_rank_reduced_by_layer.items()
        },
        **{
            f"L{int(L)}_eigs_desc": arr
            for L, arr in _common.hs_eigs_reduced_by_layer.items()
        },
        **{
            f"L{int(L)}_rank_threshold": arr
            for L, arr in _common.hs_thresh_reduced_by_layer.items()
        },
    )

    _common.save_npz_result(
        _common.os.path.join(_common.hs_results_dir, "hs_random_points_pure_full.npz"),
        h_param=_common.np.asarray(_common.h_param, dtype=_common.NP_REAL_DTYPE),
        num_hs_samples=_common.np.asarray(_common.NUM_QFIM_SAMPLES, dtype=_common.NP_INT_DTYPE),
        hs_sample_seed_base=_common.np.asarray(_common.QFIM_SAMPLE_SEED_BASE, dtype=_common.NP_INT_DTYPE),
        hs_effective_rank_threshold=_common.np.asarray(_common.QFIM_EFFECTIVE_RANK_THRESHOLD, dtype=_common.NP_REAL_DTYPE),
        layers=_common.np.asarray(_common.qfim_layer_list, dtype=_common.NP_INT_DTYPE),
        representation=_common.np.asarray("pure_full"),
        hs_implementation=_common.np.asarray("pure_qfim_spectrum_over_2"),
        analysis_batch_size=_common.np.asarray(
            effective_analysis_batch_size,
            dtype=_common.NP_INT_DTYPE,
        ),
        **{f"L{int(L)}_rank": arr for L, arr in hs_rank_pure_by_layer.items()},
        **{f"L{int(L)}_eigs_desc": arr for L, arr in hs_eigs_pure_by_layer.items()},
        **{f"L{int(L)}_rank_threshold": arr for L, arr in hs_thresh_pure_by_layer.items()},
    )


    if not make_plots:
        return

    if _common.cmap is None:
        _common.cmap = _common.matplotlib.colormaps.get_cmap("viridis")

    # ============================================================
    # Plot: QFIM rank vs depth  (VIOLIN)
    #   - reduced 縺ｨ pure(full) 縺ｮ縺ｿ繧定｡ｨ遉ｺ
    #   - upper/lower bound 縺ｮ險育ｮ励・謠冗判縺ｯ陦後ｏ縺ｪ縺・
    # ============================================================

    _common.new_prx_figure(width="double")
    ax = _common.plt.gca()

    x_all = _common.np.array(_common.qfim_layer_list, dtype=_common.NP_REAL_DTYPE)
    x_labels = [str(L) for L in _common.qfim_layer_list]

    dx = 0.25
    violin_w_rank = 0.20
    num_layers = len(_common.qfim_layer_list)

    # ------------------------------
    # Reduced keep (0..3)
    # ------------------------------
    for idx, L in enumerate(_common.qfim_layer_list):
        color = _common.cmap(idx / num_layers)
        pos_red = float(L) + dx
        red_dataset = _common._make_violin_ready(
            _common.qfim_rank_reduced_by_layer[L],
            ensure_positive=False,
            tiny=1e-12,
        )

        vp_red = _common.plt.violinplot(
            [red_dataset],
            positions=[pos_red],
            widths=violin_w_rank,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        _common._style_violin(
            vp_red,
            facecolor=color,
            edgecolor=color,
            alpha=0.12,
            linewidth=1.0,
            hatch="///",
            linecolor=color,
            linealpha=0.7,
        )

    # ------------------------------
    # Pure(full) (only where computed)
    # ------------------------------
    pure_layers = [
        L for L in _common.qfim_layer_list if _common.qfim_rank_pure_by_layer[L] is not None
    ]

    for L in pure_layers:
        idx = _common.qfim_layer_list.index(L)
        color = _common.cmap(idx / num_layers)
        pos_pure = float(L) - dx
        pure_dataset = _common._make_violin_ready(
            _common.qfim_rank_pure_by_layer[L],
            ensure_positive=False,
            tiny=1e-12,
        )

        vp_pure = _common.plt.violinplot(
            [pure_dataset],
            positions=[pos_pure],
            widths=violin_w_rank,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        _common._style_violin(
            vp_pure,
            facecolor=color,
            edgecolor=color,
            alpha=0.20,
            linewidth=1.0,
            linecolor=color,
            linealpha=0.7,
        )

    # ------------------------------
    # Axes & grid
    # ------------------------------
    ax.set_xticks(x_all)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(r"QFIM effective rank $(\lambda_k > 10^{-12})$")
    _common.set_prx_title(rf"QFIM rank at {_common.NUM_QFIM_SAMPLES} random points", ax=ax)
    ax.grid(True, axis="y", alpha=0.3)

    # ------------------------------
    # Legends
    # ------------------------------
    layer_handles = [
        _common.Patch(
            facecolor=_common.cmap(i / num_layers),
            edgecolor=_common.cmap(i / num_layers),
            alpha=0.25,
            label=f"L{L}",
        )
        for i, L in enumerate(_common.qfim_layer_list)
    ]

    type_handles = [
        _common.Patch(facecolor="white", edgecolor="black", label="Pure(full)"),
        _common.Patch(facecolor="white", edgecolor="black", hatch="///", label=f"Reduced (keep={_common.KEEP_WIRES})"),
    ]

    if _common.SHOW_REDUNDANT_LAYER_LEGENDS:
        leg_layers = ax.legend(
            handles=layer_handles,
            bbox_to_anchor=(1.05, 1.0),
            loc="upper left",
            frameon=True,
        )
        ax.add_artist(leg_layers)

    ax.legend(
        handles=type_handles,
        loc="best",
        frameon=True,
        framealpha=0.9,
    )

    _common.save_current_figure(
        _common.os.path.join(_common.qfim_rank_random_dir, "qfim_rank_violinplot_random_points.pdf"),
        outside_legend=_common.SHOW_REDUNDANT_LAYER_LEGENDS,
    )


    # ============================================================
    # Plot: Maximum QFIM rank vs layer
    #   - Uses already-computed QFIM rank dictionaries.
    #   - No additional QFIM computation is performed here.
    #   - Saves separate figures for:
    #       * pure_full
    #       * reduced_0123
    #   - Upper/lower bound lines are not drawn.
    # ============================================================


    _common.plot_qfim_rank_max_by_layer(
        _common.qfim_rank_pure_by_layer,
        _common.qfim_layer_list,
        color="C0",
        title=rf"Maximum pure full-state QFIM rank at {_common.NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum QFIM effective rank $(\lambda_k > 10^{-12})$",
        outpath=_common.os.path.join(_common.qfim_rank_random_dir, "qfim_rank_max_random_points_pure_full.pdf"),
        marker="s",
        lw=1.0,
    )


    _common.plot_qfim_rank_max_by_layer(
        _common.qfim_rank_reduced_by_layer,
        _common.qfim_layer_list,
        color="C0",
        title=rf"Maximum QFIM rank at {_common.NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum QFIM effective rank $(\lambda_k > 10^{-12})$",
        outpath=_common.os.path.join(_common.qfim_rank_random_dir, "qfim_rank_max_random_points_reduced_0123.pdf"),
        marker="o",
        lw=1.0,
    )

    _common.new_prx_figure(width="double")
    ax = _common.plt.gca()

    for idx, L in enumerate(_common.qfim_layer_list):
        color = _common.cmap(idx / num_layers)
        hs_dataset = _common._make_violin_ready(
            _common.hs_rank_reduced_by_layer[L],
            ensure_positive=False,
            tiny=1e-12,
        )

        vp_hs = _common.plt.violinplot(
            [hs_dataset],
            positions=[float(L)],
            widths=violin_w_rank,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        _common._style_violin(
            vp_hs,
            facecolor=color,
            edgecolor=color,
            alpha=0.18,
            linewidth=1.0,
            linecolor=color,
            linealpha=0.7,
        )

    ax.set_xticks(x_all)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(r"HS effective rank $(\lambda_k > 10^{-12})$")
    _common.set_prx_title(
        rf"HS tangent Gram rank at {_common.NUM_QFIM_SAMPLES} random points",
        ax=ax,
    )
    ax.grid(True, axis="y", alpha=0.3)

    _common.save_current_figure(
        _common.os.path.join(_common.hs_rank_random_dir, "hs_rank_violinplot_random_points_reduced_0123.pdf"),
        outside_legend=False,
    )

    _common.plot_qfim_rank_max_by_layer(
        _common.hs_rank_reduced_by_layer,
        _common.qfim_layer_list,
        color="C3",
        title=rf"Maximum HS tangent Gram rank at {_common.NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum HS effective rank $(\lambda_k > 10^{-12})$",
        outpath=_common.os.path.join(_common.hs_rank_random_dir, "hs_rank_max_random_points_reduced_0123.pdf"),
        marker="D",
        lw=1.0,
    )


def run_optimization_path_qfim_analysis(
    *,
    analysis_batch_size: Optional[int] = None,
) -> None:
    """Compute trajectory QFIM/HS eigenspectra with fixed-size JAX batches."""
    if __package__:
        from . import unitary_pqc_overparam_common as _common
    else:
        import unitary_pqc_overparam_common as _common


    effective_analysis_batch_size = _common._resolve_analysis_batch_size(
        analysis_batch_size
    )
    # QFIM/HS eigenspectra along the Unitary-PQC VQE optimization path.
    # Rank metrics are intentionally excluded; the saved spectra support the
    # separate QFIM Trace and eigenvalue-count visualizations.
    # ============================================================


    _common.qfim_eigs_history_by_layer = compute_qfim_eigenvalue_history_by_layer(
        _common.theta_sample_traces_by_layer,
        _common.layer_list,
        keep_wires=_common.KEEP_WIRES,
        jvp_chunk=_common.RED_JVP_CHUNK,
        batch_size=effective_analysis_batch_size,
    )
    qfim_eigs_history_pure = compute_qfim_eigenvalue_history_by_layer(
        _common.theta_sample_traces_by_layer,
        _common.layer_list,
        jvp_chunk=_common.RED_JVP_CHUNK,
        representation="pure_full",
        batch_size=effective_analysis_batch_size,
    )

    # Save the canonical eigenvalue/trace archives consumed by the visualizer.
    _common.qfim_optimization_path_result_paths_by_keep = (
        _common.save_qfim_optimization_path_results_by_keep(
            layers=_common.layer_list,
            sample_iterations=_common.sample_iters,
            eigs_keep0123_by_layer=_common.qfim_eigs_history_by_layer,
            eigs_keep01234_by_layer=qfim_eigs_history_pure,
            analysis_batch_size=effective_analysis_batch_size,
        )
    )

    _common.hs_eigs_history_by_layer = compute_hs_eigenvalue_history_by_layer(
        _common.theta_sample_traces_by_layer,
        _common.layer_list,
        keep_wires=_common.KEEP_WIRES,
        jvp_chunk=_common.RED_JVP_CHUNK,
        batch_size=effective_analysis_batch_size,
    )
    _common.save_npz_result(
        _common.os.path.join(
            _common.hs_results_dir,
            "hs_eigs_history_optimization_path_reduced_0123.npz",
        ),
        sample_iters=_common.np.asarray(_common.sample_iters, dtype=_common.NP_INT_DTYPE),
        plot_iters=_common._qfim_history_plot_iterations(_common.sample_iters),
        layers=_common.np.asarray(_common.layer_list, dtype=_common.NP_INT_DTYPE),
        representation=_common.np.asarray("reduced_mixed"),
        analysis_batch_size=_common.np.asarray(
            effective_analysis_batch_size,
            dtype=_common.NP_INT_DTYPE,
        ),
        **{
            f"L{int(L)}_eigs": arr
            for L, arr in _common.hs_eigs_history_by_layer.items()
        },
    )

    # Reuse the already-batched pure QFIM spectra: G_HS = F_Q / 2.
    hs_eigs_history_pure = {
        int(L): 0.5 * _common.np.asarray(eigs, dtype=_common.NP_REAL_DTYPE)
        for L, eigs in qfim_eigs_history_pure.items()
    }
    _common.save_npz_result(
        _common.os.path.join(
            _common.hs_results_dir,
            "hs_eigs_history_optimization_path_pure_full.npz",
        ),
        sample_iters=_common.np.asarray(_common.sample_iters, dtype=_common.NP_INT_DTYPE),
        plot_iters=_common._qfim_history_plot_iterations(_common.sample_iters),
        layers=_common.np.asarray(_common.layer_list, dtype=_common.NP_INT_DTYPE),
        representation=_common.np.asarray("pure_full"),
        hs_implementation=_common.np.asarray("pure_qfim_spectrum_over_2"),
        analysis_batch_size=_common.np.asarray(
            effective_analysis_batch_size,
            dtype=_common.NP_INT_DTYPE,
        ),
        **{f"L{int(L)}_eigs": arr for L, arr in hs_eigs_history_pure.items()},
    )


if __name__ == "__main__":
    raise SystemExit(main())
