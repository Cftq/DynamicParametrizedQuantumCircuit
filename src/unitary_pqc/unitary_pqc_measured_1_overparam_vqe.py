#!/usr/bin/env python
# coding: utf-8
"""VQE stage for the measurement-outcome-1 unitary PQC.

Numerical dependencies are imported only when a computation is requested.
"""
from __future__ import annotations

from typing import Optional


def main(argv=None) -> int:
    if __package__:
        from . import unitary_pqc_measured_1_overparam_cli as _cli
    else:
        import unitary_pqc_measured_1_overparam_cli as _cli

    args = _cli.parse_stage_args("vqe", argv)
    routed_status = _cli.maybe_relaunch_stage_in_wsl("vqe", args, __file__, argv)
    if routed_status is not None:
        return routed_status
    archive_path = run_unitary_pqc_vqe_stage(
        h_param=args.h_param,
        vqe_batch_size=args.vqe_batch_size,
        device=args.device,
    )
    print(f"Saved VQE numerical results to: {archive_path}")
    return 0


def run_unitary_pqc_vqe_stage(
    *,
    h_param: Optional[float] = None,
    vqe_batch_size: Optional[int] = None,
    device: Optional[str] = None,
) -> str:
    """Train once and save the float64 archive consumed by later analyses."""
    if __package__:
        from .unitary_pqc_measured_1_overparam_cli import load_stage_common
    else:
        from unitary_pqc_measured_1_overparam_cli import load_stage_common

    _common = load_stage_common("vqe", device)
    _common.configure_unitary_pqc_overparam(h_value=h_param)
    run_vqe_optimization(vqe_batch_size=vqe_batch_size)
    return _common.os.path.join(
        _common.energy_results_dir, "vqe_optimization_results.npz"
    )


def _vqe_sample_slot_by_iteration(
    num_steps: int,
    sample_iterations,
) -> np.ndarray:
    """Map each pre-update iteration to its fixed sample-buffer slot."""
    if __package__:
        from . import unitary_pqc_measured_1_overparam_common as _common
    else:
        import unitary_pqc_measured_1_overparam_common as _common
    num_steps = int(num_steps)
    sample_iterations = _common.np.asarray(sample_iterations, dtype=_common.NP_INT_DTYPE)

    if num_steps <= 0:
        raise ValueError("num_steps must be a positive integer.")
    if sample_iterations.ndim != 1:
        raise ValueError("sample_iterations must be one-dimensional.")
    if sample_iterations.size == 0:
        raise ValueError("sample_iterations must not be empty.")
    if _common.np.unique(sample_iterations).size != sample_iterations.size:
        raise ValueError("sample_iterations must not contain duplicates.")
    if _common.np.any(sample_iterations < 0) or _common.np.any(sample_iterations >= num_steps):
        raise ValueError(
            "sample_iterations must lie in the half-open range "
            f"[0, {num_steps})."
        )

    slot_by_iteration = _common.np.full(num_steps, -1, dtype=_common.np.int32)
    slot_by_iteration[sample_iterations] = _common.np.arange(
        sample_iterations.size,
        dtype=_common.np.int32,
    )
    return slot_by_iteration


def make_vqe_batch_runner(
    current_layer: int,
    *,
    num_steps: int,
    sample_iterations,
    optimizer,
):
    """Compile a fixed-size batch of independent Unitary-PQC VQE runs.

    The trace convention intentionally matches the historical Unitary-PQC
    archive: index ``t`` stores energy/gradient data at ``theta_t`` before the
    corresponding optimizer update, while ``theta_final`` is ``theta_steps``.
    """
    if __package__:
        from . import unitary_pqc_measured_1_overparam_common as _common
    else:
        import unitary_pqc_measured_1_overparam_common as _common
    import optax
    current_layer = int(current_layer)
    num_steps = int(num_steps)
    num_total_params = _common.num_params_per_layer * current_layer
    sample_slot_by_iteration = _vqe_sample_slot_by_iteration(
        num_steps,
        sample_iterations,
    )
    scan_sample_slots = _common.jnp.asarray(
        sample_slot_by_iteration,
        dtype=_common.jnp.int32,
    )
    num_samples = int(_common.np.asarray(sample_iterations).size)
    energy_and_grad = _common.jax.value_and_grad(
        _common.make_energy_fn_for_layer(current_layer)
    )

    def optimize_one_run(theta_initial: jnp.ndarray):
        theta = _common.jnp.asarray(theta_initial, dtype=_common.REAL_DTYPE)
        opt_state = optimizer.init(theta)
        theta_samples = _common.jnp.zeros(
            (num_samples, num_total_params),
            dtype=_common.REAL_DTYPE,
        )

        def one_step(carry, sample_slot):
            theta_old, opt_state_old, theta_samples_old = carry
            energy, grad = energy_and_grad(theta_old)
            grad_norm = _common.jnp.linalg.norm(grad)
            updates, opt_state_new = optimizer.update(
                grad,
                opt_state_old,
                theta_old,
            )
            theta_new = _common.wrap_to_pi(
                optax.apply_updates(theta_old, updates)
            )

            theta_samples_new = _common.jax.lax.cond(
                sample_slot >= 0,
                lambda buffer: buffer.at[sample_slot].set(theta_old),
                lambda buffer: buffer,
                theta_samples_old,
            )
            new_carry = (
                theta_new,
                opt_state_new,
                theta_samples_new,
            )
            return new_carry, (energy, grad_norm)

        (
            (
                theta_final,
                _,
                theta_samples_final,
            ),
            (energy_trace, grad_norm_trace),
        ) = _common.jax.lax.scan(
            one_step,
            (theta, opt_state, theta_samples),
            scan_sample_slots,
        )
        return (
            theta_final,
            energy_trace,
            grad_norm_trace,
            theta_samples_final,
        )

    return _common.jax.jit(_common.jax.vmap(optimize_one_run))


def _pad_vqe_theta_batch(
    theta_batch: jnp.ndarray,
    batch_size: int,
) -> tuple[jnp.ndarray, int]:
    """Pad a final partial batch without changing any valid vmap lane."""
    if __package__:
        from . import unitary_pqc_measured_1_overparam_common as _common
    else:
        import unitary_pqc_measured_1_overparam_common as _common
    batch_size = int(batch_size)
    valid_count = int(theta_batch.shape[0])
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not 1 <= valid_count <= batch_size:
        raise ValueError(
            f"Expected between 1 and {batch_size} runs, got {valid_count}."
        )
    if valid_count == batch_size:
        return theta_batch, valid_count

    padding = _common.jnp.repeat(
        theta_batch[-1:, :],
        repeats=batch_size - valid_count,
        axis=0,
    )
    return _common.jnp.concatenate((theta_batch, padding), axis=0), valid_count


def _resolve_vqe_batch_size(batch_size: Optional[int]) -> int:
    if __package__:
        from . import unitary_pqc_measured_1_overparam_common as _common
    else:
        import unitary_pqc_measured_1_overparam_common as _common
    resolved = _common.VQE_BATCH_SIZE if batch_size is None else int(batch_size)
    if resolved <= 0:
        raise ValueError("vqe_batch_size must be a positive integer.")
    return resolved


def run_vqe_optimization(
    *,
    save_circuits: bool = False,
    vqe_batch_size: Optional[int] = None,
) -> None:
    """Run VQE optimization for every configured layer and collect traces.

    ``save_circuits`` is retained for call compatibility. Circuit drawing is
    now always deferred to
    ``unitary_pqc_measured_1_overparam_draw_circuits.py``.
    Independent trials are compiled as ``jit(vmap(scan))`` batches.
    """
    if __package__:
        from . import unitary_pqc_measured_1_overparam_common as _common
    else:
        import unitary_pqc_measured_1_overparam_common as _common
    import optax


    effective_batch_size = _resolve_vqe_batch_size(vqe_batch_size)
    if int(_common.num_runs) <= 0:
        raise ValueError("cfg.NUM_RUNS must be a positive integer.")
    if save_circuits:
        _common.warnings.warn(
            "save_circuits is no longer executed during VQE. Run "
            "src/unitary_pqc/"
            "unitary_pqc_measured_1_overparam_draw_circuits.py after "
            "the numerical results have been saved.",
            UserWarning,
            stacklevel=2,
        )

    # Optimization Loop per Layer
    # ==============================
    _common.success_rates_history = {}
    _common.energy_mean_history = {}
    _common.energy_std_history = {}

    _common.final_stats = {"layer": [], "success_rate": [], "mean_energy": [], "std_energy": []}

    _common.dense_until_layer = _common.cfg.UNITARY_PQC_DENSE_UNTIL_LAYER
    _common.max_layer = _common.cfg.UNITARY_PQC_MAX_LAYER
    _common.sparse_step = _common.cfg.UNITARY_PQC_SPARSE_STEP

    _common.dense_end = min(_common.dense_until_layer, _common.max_layer)
    _common.layer_list = _common.build_layer_list(_common.max_layer, _common.dense_until_layer, _common.sparse_step)

    # --- Result directories ---
    _common.save_dir = _common._unitary_pqc_save_dir(_common.h_param)
    _common.figures_dir = _common.os.path.join(_common.save_dir, "figures")
    _common.energy_fig_dir = _common.os.path.join(_common.figures_dir, "energy")
    _common.qfim_fig_dir = _common.os.path.join(_common.figures_dir, "qfim")
    _common.hs_fig_dir = _common.os.path.join(_common.figures_dir, "hs")
    _common.hessian_fig_dir = _common.os.path.join(_common.figures_dir, "hessian")
    _common.circuit_dir = _common.os.path.join(_common.save_dir, "optimized_circuits")
    _common.numerical_results_dir = _common.os.path.join(_common.save_dir, "numerical_results")
    _common.energy_results_dir = _common.os.path.join(_common.numerical_results_dir, "energy")
    _common.qfim_results_dir = _common.os.path.join(_common.numerical_results_dir, "qfim")
    _common.hs_results_dir = _common.os.path.join(_common.numerical_results_dir, "hs")
    _common.hessian_results_dir = _common.os.path.join(_common.numerical_results_dir, "hessian")
    _common.qfim_eigs_dir = _common.os.path.join(_common.qfim_fig_dir, "eigs")
    _common.qfim_eigs_pure_dir = _common.os.path.join(_common.qfim_eigs_dir, "pure_full")
    _common.qfim_eigs_reduced_0123_dir = _common.os.path.join(_common.qfim_eigs_dir, "reduced_keep_0123")
    _common.qfim_rank_dir = _common.os.path.join(_common.qfim_fig_dir, "rank")
    _common.qfim_rank_random_dir = _common.os.path.join(_common.qfim_rank_dir, "random_points")
    _common.hs_eigs_dir = _common.os.path.join(_common.hs_fig_dir, "eigs")
    _common.hs_eigs_reduced_0123_dir = _common.os.path.join(_common.hs_eigs_dir, "reduced_keep_0123")
    _common.hs_rank_dir = _common.os.path.join(_common.hs_fig_dir, "rank")
    _common.hs_rank_random_dir = _common.os.path.join(_common.hs_rank_dir, "random_points")
    _common._ensure_unitary_result_dirs()

    _common.optimizer = optax.adam(learning_rate=_common.lr)

    _common.theta_history = {L: [] for L in _common.layer_list}  # final theta of each run
    _common.best_theta_by_layer = {}

    # Final RMS wrapped parameter distance per layer (distribution over runs)
    #   Reference theta_ref(L): best-run final parameters at the same layer
    _common.final_theta_wrapped_rmsdist_by_layer = {}  # L -> (num_runs,) array of d_theta(theta_final, theta_ref)

    _common.energy_traces_by_layer = {}
    _common.grad_norm_traces_by_layer = {}  # L -> (num_runs, steps) gradient-norm traces

    # Sampled optimization-time states feed the post-VQE matrix analyses.
    _common.sample_every = _common.cfg.SAMPLE_EVERY
    _common.sample_iters = _common.np.arange(0, _common.steps, _common.sample_every, dtype=_common.NP_INT_DTYPE)
    if _common.sample_iters.size == 0 or _common.sample_iters[0] != 0:
        _common.sample_iters = _common.np.concatenate([[0], _common.sample_iters]).astype(_common.NP_INT_DTYPE)

    if _common.sample_iters[-1] != _common.steps - 1:
        _common.sample_iters = _common.np.concatenate([_common.sample_iters, [_common.steps - 1]]).astype(_common.NP_INT_DTYPE)

    _common.sample_iters = _common.np.unique(_common.sample_iters).astype(_common.NP_INT_DTYPE)
    _common.sample_iter_set = set(int(t) for t in _common.sample_iters.tolist())

    _common.theta_sample_traces_by_layer = {}
    _common.cmap = _common.matplotlib.colormaps.get_cmap("viridis")

    # tqdm: Layers (VQE)
    for current_layer in _common.tqdm(_common.layer_list, desc="Layers (VQE)", unit="layer"):
        num_total_params = _common.num_params_per_layer * current_layer

        run_vqe_batch = make_vqe_batch_runner(
            current_layer,
            num_steps=_common.steps,
            sample_iterations=_common.sample_iters,
            optimizer=_common.optimizer,
        )

        # Generate every run before batching so batch size never changes the
        # historical random-key sequence or run ordering.
        base_key = _common.jax.random.PRNGKey(current_layer * 1000)
        keys = _common.jax.random.split(base_key, _common.num_runs)
        theta_initial_runs = _common.jnp.stack(
            [
                _common.jax.random.uniform(
                    keys[run_index],
                    shape=(num_total_params,),
                    minval=-_common.jnp.pi,
                    maxval=_common.jnp.pi,
                    dtype=_common.REAL_DTYPE,
                )
                for run_index in range(_common.num_runs)
            ],
            axis=0,
        )

        output_parts = tuple([] for _ in range(4))
        batch_starts = range(0, _common.num_runs, effective_batch_size)
        for batch_start in _common.tqdm(
            batch_starts,
            total=(_common.num_runs + effective_batch_size - 1) // effective_batch_size,
            desc=(
                f"Run batches (L={current_layer}, "
                f"batch={effective_batch_size})"
            ),
            unit="batch",
            leave=False,
        ):
            batch_end = min(batch_start + effective_batch_size, _common.num_runs)
            theta_batch, valid_count = _pad_vqe_theta_batch(
                theta_initial_runs[batch_start:batch_end],
                effective_batch_size,
            )
            host_outputs = _common.jax.device_get(run_vqe_batch(theta_batch))
            for parts, values in zip(output_parts, host_outputs):
                parts.append(
                    _common.np.asarray(
                        values[:valid_count],
                        dtype=_common.NP_REAL_DTYPE,
                    )
                )

        (
            theta_final_data,
            energy_data,
            gradnorm_data,
            theta_sample_data,
        ) = (
            _common.np.concatenate(parts, axis=0)
            for parts in output_parts
        )
        expected_shapes = (
            (_common.num_runs, num_total_params),
            (_common.num_runs, _common.steps),
            (_common.num_runs, _common.steps),
            (_common.num_runs, _common.sample_iters.size, num_total_params),
        )
        actual_shapes = tuple(
            array.shape
            for array in (
                theta_final_data,
                energy_data,
                gradnorm_data,
                theta_sample_data,
            )
        )
        if actual_shapes != expected_shapes:
            raise AssertionError(
                f"Unexpected VQE output shapes for L={current_layer}: "
                f"{actual_shapes} != {expected_shapes}."
            )

        final_energies = energy_data[:, -1]
        finite_run_indices = _common.np.flatnonzero(_common.np.isfinite(final_energies))
        if finite_run_indices.size == 0:
            raise FloatingPointError(
                f"No finite final VQE energy was produced for L={current_layer}."
            )
        best_local_index = int(_common.np.argmin(final_energies[finite_run_indices]))
        best_run_index = int(finite_run_indices[best_local_index])
        best_final_theta = theta_final_data[best_run_index].copy()

        _common.theta_history[current_layer] = theta_final_data

        # Final RMS wrapped distance distribution over runs
        theta_runs_jnp = _common.jnp.asarray(_common.theta_history[current_layer], dtype=_common.REAL_DTYPE)     # (num_runs, num_params)
        theta_ref_jnp = _common.jnp.asarray(best_final_theta, dtype=_common.REAL_DTYPE)[None, :]         # (1, num_params)

        wrapped_diff = _common.wrap_to_pi(theta_runs_jnp - theta_ref_jnp)       # (num_runs, num_params)
        d_theta_runs = _common.jnp.sqrt(_common.jnp.mean(wrapped_diff ** 2, axis=1))    # (num_runs,)

        _common.final_theta_wrapped_rmsdist_by_layer[current_layer] = _common.np.asarray(
            _common.jax.device_get(d_theta_runs), dtype=_common.NP_REAL_DTYPE
        )

        _common.energy_traces_by_layer[current_layer] = energy_data
        _common.grad_norm_traces_by_layer[current_layer] = gradnorm_data
        _common.theta_sample_traces_by_layer[current_layer] = theta_sample_data

        _common.best_theta_by_layer[current_layer] = best_final_theta.copy()

        # stats (energy mean/std)
        mean_trace = _common.np.mean(energy_data, axis=0)
        std_trace = _common.np.std(energy_data, axis=0)
        _common.energy_mean_history[current_layer] = mean_trace
        _common.energy_std_history[current_layer] = std_trace

        diffs = _common.np.abs(energy_data - _common.smallest_eigval)
        success_flags = diffs <= _common.tolerance
        success_rate_per_step = _common.np.mean(success_flags, axis=0)
        _common.success_rates_history[current_layer] = success_rate_per_step

        _common.final_stats["layer"].append(current_layer)
        _common.final_stats["success_rate"].append(success_rate_per_step[-1])
        _common.final_stats["mean_energy"].append(mean_trace[-1])
        _common.final_stats["std_energy"].append(std_trace[-1])

        del run_vqe_batch
        _common._release_jax_compilation_cache()

    _common.save_unitary_vqe_results()


if __name__ == "__main__":
    raise SystemExit(main())
