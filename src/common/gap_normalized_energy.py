"""Optimization accuracy and success rates from saved final VQE energies.

Only NumPy is needed for the numerical routines. No circuit, training, or QFIM
code is imported. The gap is to the first energy above the entire ground-state
subspace, so adding a spectator ancilla does not turn it into a zero gap.
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil

import numpy as np

if __package__:
    from .hessian_curvature import hamiltonian_matrix_numpy
else:
    from hessian_curvature import hamiltonian_matrix_numpy


DEFAULT_THRESHOLDS = (1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)


def spectral_gap_from_hamiltonian(hamiltonian_matrix):
    """Return the ground energy and first distinct excited energy of H.

    Eigenvalues within 64 * dimension * float64 epsilon * spectral scale of
    the minimum belong to the numerically degenerate ground-state subspace.
    A spectrum without a resolvable excited level is rejected.
    """
    matrix = np.asarray(hamiltonian_matrix)
    if (
        matrix.ndim != 2 or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or not np.issubdtype(matrix.dtype, np.number)
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("Hamiltonian must be a finite Hermitian square matrix.")
    scale = float(np.max(np.abs(matrix)))
    scaled = matrix / scale if scale > 0.0 else matrix
    if not np.allclose(scaled, scaled.conj().T, rtol=1e-12, atol=1e-14):
        raise ValueError("Hamiltonian must be a finite Hermitian square matrix.")
    # Explicit float64/complex128 avoids float32 eigensolver rounding changing
    # which levels count as degenerate.
    matrix = matrix.astype(np.complex128 if np.iscomplexobj(matrix) else np.float64)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Hamiltonian eigenvalues must be finite.")
    spectral_scale = float(np.max(np.abs(eigenvalues)))
    tolerance = 64.0 * matrix.shape[0] * np.finfo(np.float64).eps * spectral_scale
    ground_energy = float(eigenvalues[0])
    excited = eigenvalues - ground_energy > tolerance
    if not np.any(excited):
        raise ValueError("Hamiltonian has no numerically resolvable spectral gap.")
    first_excited = float(eigenvalues[np.flatnonzero(excited)[0]])
    gap = first_excited - ground_energy
    if not math.isfinite(gap) or gap <= 0.0:
        raise ValueError("Hamiltonian spectral gap must be finite and positive.")
    return {
        "ground_energy": ground_energy,
        "first_excited_energy": first_excited,
        "spectral_gap": gap,
        "ground_degeneracy": int(np.count_nonzero(~excited)),
        "degeneracy_tolerance": tolerance,
    }


def compute_gap_normalized_energy(
    final_energies_by_layer,
    *,
    h_param,
    thresholds=DEFAULT_THRESHOLDS,
    hamiltonian_matrix=None,
):
    """Compute per-run (E_final - E0)/gap and strict-threshold success rates.

    Every layer must have a nonempty vector of finite real final energies.
    Only negative errors consistent with eigensolver roundoff are clipped to
    zero. Materially sub-ground energies are rejected, rather than counted as
    successes. SEM uses the sample standard deviation, with zero for one run.
    Raw saved energies and all normalized samples are returned unchanged apart
    from that negative-roundoff clipping; no failed/large errors are excluded.
    """
    if not final_energies_by_layer:
        raise ValueError("Saved final energies are required for optimization metrics.")
    if any(
        isinstance(layer, (bool, np.bool_))
        or not isinstance(layer, (int, np.integer)) or layer < 1
        for layer in final_energies_by_layer
    ):
        raise ValueError("Layer keys must be positive integers.")
    layers = np.asarray(sorted(final_energies_by_layer), dtype=np.int64)
    h_param = float(h_param)
    if not math.isfinite(h_param):
        raise ValueError("h_param must be finite.")
    thresholds = np.asarray(thresholds)
    if (
        thresholds.ndim != 1 or thresholds.size == 0
        or not np.issubdtype(thresholds.dtype, np.number)
        or np.iscomplexobj(thresholds) or not np.all(np.isfinite(thresholds))
        or np.any(thresholds <= 0.0)
    ):
        raise ValueError("Thresholds must be a nonempty vector of finite positive numbers.")
    thresholds = thresholds.astype(np.float64, copy=True)
    spectrum = spectral_gap_from_hamiltonian(
        hamiltonian_matrix_numpy(h_param)
        if hamiltonian_matrix is None else hamiltonian_matrix
    )
    tolerance = spectrum["degeneracy_tolerance"]
    ground_energy = spectrum["ground_energy"]
    gap = spectrum["spectral_gap"]
    final_energies = {}
    normalized_errors = {}
    statistics = []
    success_probabilities = []
    clipped_counts = []
    for layer in layers:
        values = np.asarray(final_energies_by_layer[layer])
        if (
            values.ndim != 1 or values.size == 0
            or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values) or not np.all(np.isfinite(values))
        ):
            raise ValueError(f"Layer {layer}: final energies must be a nonempty finite real vector.")
        values = values.astype(np.float64, copy=True)
        energy_errors = values - ground_energy
        if np.any(energy_errors < -tolerance):
            raise ValueError(
                f"Layer {layer}: saved final energy lies below the ground energy "
                f"by more than roundoff tolerance ({tolerance:.3g})."
            )
        clipped_counts.append(int(np.count_nonzero(energy_errors < 0.0)))
        errors = np.maximum(energy_errors, 0.0) / gap
        if not np.all(np.isfinite(errors)):
            raise ValueError(f"Layer {layer}: normalized energy errors are non-finite.")
        final_energies[int(layer)] = values
        normalized_errors[int(layer)] = errors
        sem = np.std(errors, ddof=1) / np.sqrt(errors.size) if errors.size > 1 else 0.0
        statistics.append((np.mean(errors), sem, np.min(errors), np.max(errors), np.median(errors)))
        success_probabilities.append(np.mean(errors[:, None] < thresholds[None, :], axis=0))
    stats = np.asarray(statistics)
    return {
        "h_param": h_param,
        "layers": layers,
        "thresholds": thresholds,
        **spectrum,
        "negative_roundoff_tolerance": tolerance,
        "normalized_errors_by_layer": normalized_errors,
        "final_energies_by_layer": final_energies,
        "success_probabilities": np.asarray(success_probabilities),
        "num_runs_by_layer": np.asarray([final_energies[int(layer)].size for layer in layers]),
        "num_clipped_negative_by_layer": np.asarray(clipped_counts, dtype=np.int64),
        **dict(zip(("mean", "sem", "min", "max", "median"), stats.T)),
    }


def _threshold_label(value):
    exponent = int(round(np.log10(value)))
    if np.isclose(value / 10.0 ** exponent, 1.0, rtol=1e-12, atol=0.0):
        return rf"$\epsilon=10^{{{exponent}}}$"
    return rf"$\epsilon={value:g}$"


def _layer_ticks(layers):
    """Keep readable ticks at actual layer values, including both endpoints."""
    indices = np.unique(np.linspace(0, len(layers) - 1, min(len(layers), 7)).round().astype(int))
    return layers[indices]


def _beeswarm_x_positions(ax, layer, values, *, half_width, marker_area=10.0):
    """Pack dots deterministically using distances on the rendered y axis.

    Call after setting the final axes limits/layout and drawing the canvas.
    Only x changes; every sample retains its original y value and order.
    Candidates tangent to already placed markers keep dots separated whenever
    space permits. If the swarm is wider than its layer cell, all x offsets
    are compressed by the same factor. Crowded identical values can therefore
    overlap, but no samples are dropped, randomly jittered, or moved in y.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Beeswarm values must be a nonempty finite vector.")
    if not math.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("Beeswarm half_width must be finite and positive.")
    if not math.isfinite(marker_area) or marker_area <= 0.0:
        raise ValueError("Beeswarm marker_area must be finite and positive.")
    pixels = ax.transData.transform(np.column_stack((np.full(values.size, layer), values)))
    center_x = pixels[0, 0]
    diameter = math.sqrt(marker_area) * ax.figure.dpi / 72.0 * 1.1
    order = np.argsort(pixels[:, 1], kind="stable")
    placed_y = []
    placed_x = []
    offsets = np.zeros(values.size, dtype=np.float64)
    for index in order:
        y_position = pixels[index, 1]
        delta_y = y_position - np.asarray(placed_y)
        nearby = np.abs(delta_y) < diameter
        neighbor_x = np.asarray(placed_x)[nearby]
        neighbor_dy = delta_y[nearby]
        if neighbor_x.size:
            tangent = np.sqrt(np.maximum(diameter ** 2 - neighbor_dy ** 2, 0.0))
            candidates = np.concatenate(([0.0], neighbor_x - tangent, neighbor_x + tangent))
            separation_sq = (candidates[:, None] - neighbor_x[None, :]) ** 2 + neighbor_dy[None, :] ** 2
            valid = np.all(separation_sq >= diameter ** 2 * (1.0 - 1e-12), axis=1)
            candidates = candidates[valid]
            # Alternate exact left/right ties deterministically.
            sign_priority = candidates < 0.0 if len(placed_x) % 2 else candidates > 0.0
            choice = np.lexsort((sign_priority, np.abs(candidates)))[0]
            offset = candidates[choice]
        else:
            offset = 0.0
        offsets[index] = offset
        placed_x.append(offset)
        placed_y.append(y_position)
    pixel_bound = abs(ax.transData.transform((layer + half_width, values[0]))[0] - center_x)
    max_offset = float(np.max(np.abs(offsets)))
    if max_offset > pixel_bound:
        offsets *= pixel_bound / max_offset
    pixels[:, 0] = center_x + offsets
    return ax.transData.inverted().transform(pixels)[:, 0]


def _swarm_half_widths(layers):
    """Limit each swarm to 36% of the nearest adjacent layer spacing."""
    layers = np.asarray(layers, dtype=np.float64)
    if layers.size == 1:
        return np.asarray([0.36])
    intervals = np.diff(layers)
    return 0.36 * np.minimum(np.r_[intervals[0], intervals], np.r_[intervals, intervals[-1]])


def save_gap_normalized_energy_outputs(
    final_energies_by_layer,
    *,
    h_param,
    figures_dir,
    statistics_outpath=None,
    thresholds=DEFAULT_THRESHOLDS,
    hamiltonian_matrix=None,
    metadata=None,
):
    """Save a beeswarm, two names for success curves, and pickle-free NPZ.

    The error figure uses a symlog axis with a linear segment below one tenth
    of the strictest success threshold. All normalized samples, including true
    zeros, appear in a deterministic horizontal beeswarm with a median bar at
    each layer. Crowded swarms are compressed to remain in their layer cells;
    dense ties may overlap, but no values or samples are removed or moved in y.
    """
    if __package__:
        from .plot import new_fig_ax, save_fig
    else:
        from plot import new_fig_ax, save_fig

    results = compute_gap_normalized_energy(
        final_energies_by_layer, h_param=h_param, thresholds=thresholds,
        hamiltonian_matrix=hamiltonian_matrix,
    )
    layers = results["layers"]
    thresholds = results["thresholds"]
    linear_threshold = float(np.min(thresholds) / 10.0)
    arrays = {
        key: np.asarray(value) for key, value in results.items()
        if key not in ("normalized_errors_by_layer", "final_energies_by_layer")
    }
    arrays.update({
        "normalization": np.asarray("(E_final - E0) / (E_first_distinct_excited - E0)"),
        "success_criterion": np.asarray("normalized_error < epsilon (strict inequality)"),
        "ground_subspace_policy": np.asarray("first distinct excited eigenvalue above the degenerate ground subspace"),
        "negative_error_policy": np.asarray("clip negative roundoff only; reject materially sub-ground energies"),
        "error_axis_scale": np.asarray("symlog; true zeros retained"),
        "error_axis_linear_threshold": np.asarray(linear_threshold),
        "error_plot_style": np.asarray("deterministic screen-coordinate beeswarm with per-layer median bars"),
        "beeswarm_overflow_policy": np.asarray("uniformly compress x offsets to 36% of nearest layer spacing; crowded ties may overlap; preserve every y value"),
        "energy_sample_source": np.asarray("saved final VQE energies; no training or QFIM recomputation"),
    })
    for layer in layers:
        arrays[f"L{layer}_final_energies"] = results["final_energies_by_layer"][int(layer)]
        arrays[f"L{layer}_normalized_errors"] = results["normalized_errors_by_layer"][int(layer)]
    for key, value in (metadata or {}).items():
        if not isinstance(key, str) or key in arrays:
            raise ValueError(f"Invalid or reserved metadata key: {key!r}.")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"Metadata {key!r} must not require pickle storage.")
        arrays[key] = array

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_path = (
        figures_dir / "gap_normalized_energy_statistics.npz"
        if statistics_outpath is None else Path(statistics_outpath)
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        "normalized_energy_error": figures_dir / "final_gap_normalized_energy_error.pdf",
        "success_probability": figures_dir / "gap_normalized_success_probability.pdf",
        "success_probability_multiple_tolerances": figures_dir / "success_probability_multiple_tolerances_gap_normalized.pdf",
    }

    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.31)
    half_widths = _swarm_half_widths(layers)
    marker_area = 10.0
    threshold_linestyles = ("--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (1, 1)))
    for index, threshold in enumerate(thresholds):
        ax.axhline(
            threshold, color=f"C{(index + 1) % 10}",
            ls=threshold_linestyles[index % len(threshold_linestyles)], lw=0.8,
            label=_threshold_label(threshold),
        )
    ax.set_yscale("symlog", linthresh=linear_threshold, linscale=0.5)
    # A linear autoscale margin becomes imperceptible across many log decades.
    # Give the largest run/threshold explicit multiplicative headroom.
    upper_value = max(
        float(np.max(results["max"])),
        float(np.max(thresholds)),
    )
    ax.set_ylim(bottom=0.0, top=2.0 * upper_value)
    ax.set_xlim(float(layers[0] - half_widths[0] * 1.8), float(layers[-1] + half_widths[-1] * 1.8))
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(r"$(E_{\mathrm{final}}-E_0)/\Delta$")
    ax.set_xticks(_layer_ticks(layers))
    # Allocate a marker radius below true zero in display coordinates. This
    # prevents dots at zero being cut in half by the bottom axes boundary.
    fig.canvas.draw()
    padding_pixels = math.sqrt(marker_area) * fig.dpi / 144.0 + 2.0
    y_transform = ax.yaxis.get_transform()
    transformed_top = float(y_transform.transform(2.0 * upper_value))
    transformed_bottom = -transformed_top * padding_pixels / (ax.bbox.height - padding_pixels)
    lower_limit = float(y_transform.inverted().transform(transformed_bottom))
    ax.set_ylim(bottom=lower_limit)
    fig.canvas.draw()
    for index, layer in enumerate(layers):
        values = results["normalized_errors_by_layer"][int(layer)]
        positions = _beeswarm_x_positions(
            ax, float(layer), values, half_width=float(half_widths[index]), marker_area=marker_area,
        )
        arrays[f"L{layer}_beeswarm_x"] = positions
        ax.scatter(
            positions, values, s=marker_area, color="C0", alpha=0.55,
            linewidths=0.0, label="Individual runs" if index == 0 else None, zorder=2,
        )
        median_half_width = half_widths[index] * 0.65
        ax.plot(
            [layer - median_half_width, layer + median_half_width],
            [results["median"][index]] * 2, color="black", lw=1.5,
            label="Median" if index == 0 else None, zorder=4,
        )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save_fig(fig, ax, str(figure_paths["normalized_energy_error"]), outside_legend=True, legend_space_frac=0.31)

    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.31)
    success_markers = ("o", "s", "^", "D", "v", "P")
    success_linestyles = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)))
    for index, threshold in enumerate(thresholds):
        ax.plot(
            layers, results["success_probabilities"][:, index],
            marker=success_markers[index % len(success_markers)],
            linestyle=success_linestyles[index % len(success_linestyles)],
            markerfacecolor="none", color=f"C{(index + 1) % 10}", label=_threshold_label(threshold),
        )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(r"$P_{\mathrm{succ}}^{(\epsilon)}(L)$")
    ax.set_xticks(_layer_ticks(layers))
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save_fig(fig, ax, str(figure_paths["success_probability"]), outside_legend=True, legend_space_frac=0.31)
    shutil.copyfile(figure_paths["success_probability"], figure_paths["success_probability_multiple_tolerances"])

    # An explicit file object keeps the exact requested filename (np.savez
    # would otherwise append an extra .npz extension).
    with data_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    print(f"Saved gap-normalized optimization figures to: {figures_dir}", flush=True)
    print(f"Saved gap-normalized optimization statistics to: {data_path}", flush=True)
    return {**results, "figure_paths": figure_paths, "data_path": data_path}
