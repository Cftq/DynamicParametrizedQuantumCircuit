#!/usr/bin/env python
# coding: utf-8
"""First-passage convergence statistics and figures for VQE histories.

For an absolute energy-error tolerance ``delta``, this module computes

``T[L, r]``
    The first stored optimizer step whose error is at most ``delta``.  A run
    that never reaches the tolerance is represented by ``np.inf``.
``median_T[L]``
    The median over *all* runs, including the infinite values.
``p_fail[L]``
    The fraction of runs whose first-passage time is infinite.
``F_hat[L, t]``
    The fraction of all runs that have reached the tolerance by step ``t``.

The implementation deliberately uses the dense energy histories rather than
the sparse ``sample_iters`` arrays used by several other figures.  Non-finite
energy samples are treated as non-hits; a finite hit earlier in the same run
still determines its first-passage time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib
import numpy as np

try:  # Package import (``src.common.convergence_time``).
    from .plot import new_fig_ax, save_fig
except ImportError:  # Direct scripts add ``src/common`` to ``sys.path``.
    from plot import new_fig_ax, save_fig


NP_REAL_DTYPE = np.float64
NP_INT_DTYPE = np.int64

DEFAULT_CONVERGENCE_TOLERANCES = (1.0,)
CONVERGENCE_STATISTICS_SCHEMA_VERSION = 1


def normalize_convergence_tolerances(
    tolerances: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return a validated, order-preserving array of positive tolerances."""
    if tolerances is None:
        tolerances = DEFAULT_CONVERGENCE_TOLERANCES
    elif np.isscalar(tolerances):
        tolerances = (tolerances,)

    try:
        values = np.asarray(tuple(tolerances), dtype=NP_REAL_DTYPE)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("convergence tolerances must be real numbers") from exc

    if values.ndim != 1 or values.size == 0:
        raise ValueError("convergence tolerances must be a non-empty 1D sequence")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("convergence tolerances must be finite and positive")
    if np.unique(values).size != values.size:
        raise ValueError("convergence tolerances must not contain duplicates")
    return values


def _validated_layers(layers: Sequence[int]) -> np.ndarray:
    try:
        values = np.asarray(tuple(layers), dtype=NP_REAL_DTYPE)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("layers must be a numeric one-dimensional sequence") from exc

    if values.ndim != 1 or values.size == 0:
        raise ValueError("layers must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
        raise ValueError("layers must contain only finite integers")
    normalized = values.astype(NP_INT_DTYPE)
    if np.any(normalized <= 0):
        raise ValueError("layers must contain only positive integers")
    if np.unique(normalized).size != normalized.size:
        raise ValueError("layers must not contain duplicates")
    return np.sort(normalized, kind="stable")


def _validated_positive_int(value, *, name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or numeric != np.rint(numeric):
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric)


def _validated_nonnegative_int(value, *, name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if not np.isfinite(numeric) or numeric < 0.0 or numeric != np.rint(numeric):
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(numeric)


def _validated_energy_histories(
    energy_traces_by_layer: Mapping[int, np.ndarray],
    layers: np.ndarray,
    *,
    num_runs: Optional[int],
    optimizer_steps: Optional[int],
) -> tuple[list[np.ndarray], int, int, int]:
    if not isinstance(energy_traces_by_layer, Mapping):
        raise TypeError("energy_traces_by_layer must be a layer-to-array mapping")

    expected_num_runs = (
        None
        if num_runs is None
        else _validated_positive_int(num_runs, name="num_runs")
    )
    expected_optimizer_steps = (
        None
        if optimizer_steps is None
        else _validated_nonnegative_int(optimizer_steps, name="optimizer_steps")
    )

    histories: list[np.ndarray] = []
    history_shape = None
    for layer in layers:
        layer_int = int(layer)
        if layer_int not in energy_traces_by_layer:
            raise KeyError(f"missing energy traces for L={layer_int}")
        raw_history = np.asarray(energy_traces_by_layer[layer_int])
        if not np.issubdtype(raw_history.dtype, np.number) or np.iscomplexobj(
            raw_history
        ):
            raise TypeError(f"energy traces for L={layer_int} must be real numeric data")
        history = np.asarray(raw_history, dtype=NP_REAL_DTYPE)
        if history.ndim != 2 or history.shape[0] == 0 or history.shape[1] == 0:
            raise ValueError(
                f"energy traces for L={layer_int} must have shape "
                "(num_runs>0, num_time_points>0); "
                f"got {history.shape}"
            )
        if history_shape is None:
            history_shape = history.shape
        elif history.shape != history_shape:
            raise ValueError(
                "all layer histories must have the same shape; "
                f"L={layer_int} has {history.shape}, expected {history_shape}"
            )
        histories.append(history)

    assert history_shape is not None
    actual_num_runs, num_time_points = (int(v) for v in history_shape)
    if expected_num_runs is not None and actual_num_runs != expected_num_runs:
        raise ValueError(
            f"history run count is {actual_num_runs}; expected {expected_num_runs}"
        )
    if expected_optimizer_steps is not None and num_time_points not in {
        expected_optimizer_steps,
        expected_optimizer_steps + 1,
    }:
        raise ValueError(
            "history time-axis length must be optimizer_steps or "
            "optimizer_steps + 1; "
            f"got {num_time_points} for optimizer_steps={expected_optimizer_steps}"
        )

    return (
        histories,
        actual_num_runs,
        num_time_points,
        -1 if expected_optimizer_steps is None else expected_optimizer_steps,
    )


def compute_convergence_time_statistics(
    energy_traces_by_layer: Mapping[int, np.ndarray],
    layers: Sequence[int],
    *,
    ground_energy: float,
    tolerances: Optional[Sequence[float]] = None,
    num_runs: Optional[int] = None,
    optimizer_steps: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Compute exact first-passage statistics from dense energy histories.

    Array axes in the returned dictionary are documented by the
    ``*_axis_order`` fields.  In particular, failures remain ``np.inf`` in
    ``first_passage_steps`` and therefore also affect the ordinary median.
    """
    layer_values = _validated_layers(layers)
    tolerance_values = normalize_convergence_tolerances(tolerances)

    try:
        ground_energy_value = float(ground_energy)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("ground_energy must be one finite real scalar") from exc
    if not np.isfinite(ground_energy_value):
        raise ValueError("ground_energy must be one finite real scalar")

    (
        histories,
        actual_num_runs,
        num_time_points,
        archived_optimizer_steps,
    ) = _validated_energy_histories(
        energy_traces_by_layer,
        layer_values,
        num_runs=num_runs,
        optimizer_steps=optimizer_steps,
    )

    num_tolerances = int(tolerance_values.size)
    num_layers = int(layer_values.size)
    t_max = num_time_points - 1
    time_steps = np.arange(num_time_points, dtype=NP_INT_DTYPE)

    first_passage_steps = np.full(
        (num_tolerances, num_layers, actual_num_runs),
        np.inf,
        dtype=NP_REAL_DTYPE,
    )
    median_first_passage_steps = np.full(
        (num_tolerances, num_layers),
        np.inf,
        dtype=NP_REAL_DTYPE,
    )
    failure_probabilities = np.empty(
        (num_tolerances, num_layers),
        dtype=NP_REAL_DTYPE,
    )
    attainment_cdf = np.zeros(
        (num_tolerances, num_layers, num_time_points),
        dtype=NP_REAL_DTYPE,
    )
    nonfinite_energy_sample_counts = np.empty(num_layers, dtype=NP_INT_DTYPE)
    runs_with_nonfinite_energy_counts = np.empty(
        num_layers,
        dtype=NP_INT_DTYPE,
    )

    for layer_index, energy_history in enumerate(histories):
        finite_energy_samples = np.isfinite(energy_history)
        nonfinite_energy_sample_counts[layer_index] = np.count_nonzero(
            ~finite_energy_samples
        )
        runs_with_nonfinite_energy_counts[layer_index] = np.count_nonzero(
            np.any(~finite_energy_samples, axis=1)
        )
        with np.errstate(invalid="ignore", over="ignore"):
            absolute_errors = np.abs(energy_history - ground_energy_value)
        for tolerance_index, tolerance in enumerate(tolerance_values):
            within_tolerance = finite_energy_samples & (
                absolute_errors <= tolerance
            )
            reached = np.any(within_tolerance, axis=1)
            first_indices = np.argmax(within_tolerance, axis=1).astype(
                NP_REAL_DTYPE
            )
            first_indices[~reached] = np.inf
            first_passage_steps[tolerance_index, layer_index] = first_indices
            median_first_passage_steps[tolerance_index, layer_index] = np.median(
                first_indices
            )
            failure_probabilities[tolerance_index, layer_index] = np.mean(
                ~reached,
                dtype=NP_REAL_DTYPE,
            )

            finite_first_indices = first_indices[reached].astype(
                NP_INT_DTYPE,
                copy=False,
            )
            first_hit_counts = np.bincount(
                finite_first_indices,
                minlength=num_time_points,
            )
            attainment_cdf[tolerance_index, layer_index] = (
                np.cumsum(first_hit_counts, dtype=NP_INT_DTYPE)
                / NP_REAL_DTYPE(actual_num_runs)
            )

    if np.any(np.diff(attainment_cdf, axis=-1) < -1e-15):
        raise AssertionError("attainment CDF must be nondecreasing in time")
    if not np.allclose(
        attainment_cdf[:, :, -1],
        1.0 - failure_probabilities,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("F_hat(t_max) must equal 1 - p_fail")

    history_includes_optimizer_endpoint = (
        archived_optimizer_steps >= 0
        and num_time_points == archived_optimizer_steps + 1
    )
    return {
        "schema_version": np.asarray(
            CONVERGENCE_STATISTICS_SCHEMA_VERSION,
            dtype=NP_INT_DTYPE,
        ),
        "analysis_kind": np.asarray("vqe_first_passage"),
        "layers": layer_values,
        "tolerances": tolerance_values,
        "num_runs": np.asarray(actual_num_runs, dtype=NP_INT_DTYPE),
        "optimizer_steps": np.asarray(
            archived_optimizer_steps,
            dtype=NP_INT_DTYPE,
        ),
        "num_time_points": np.asarray(num_time_points, dtype=NP_INT_DTYPE),
        "t_max": np.asarray(t_max, dtype=NP_INT_DTYPE),
        "history_includes_optimizer_endpoint": np.asarray(
            history_includes_optimizer_endpoint,
            dtype=np.bool_,
        ),
        "ground_energy": np.asarray(ground_energy_value, dtype=NP_REAL_DTYPE),
        "energy_error_definition": np.asarray("abs(E - E_ground)"),
        "unreached_encoding": np.asarray("positive_infinity"),
        "nonfinite_energy_sample_counts": nonfinite_energy_sample_counts,
        "runs_with_nonfinite_energy_counts": runs_with_nonfinite_energy_counts,
        "time_steps": time_steps,
        "first_passage_steps": first_passage_steps,
        "median_first_passage_steps": median_first_passage_steps,
        "failure_probabilities": failure_probabilities,
        "attainment_cdf": attainment_cdf,
        "first_passage_axis_order": np.asarray("tolerance,layer,run"),
        "summary_axis_order": np.asarray("tolerance,layer"),
        "attainment_cdf_axis_order": np.asarray("tolerance,layer,time"),
    }


def save_convergence_time_statistics(
    statistics: Mapping[str, np.ndarray],
    outpath,
) -> Path:
    """Atomically save one derived convergence-statistics archive."""
    output_path = Path(outpath)
    if output_path.suffix.lower() != ".npz":
        raise ValueError("convergence statistics output path must end in .npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp{output_path.suffix}"
    )
    payload = {}
    for key, value in statistics.items():
        if not isinstance(key, str) or not key:
            raise ValueError("convergence-statistics keys must be non-empty strings")
        array_value = np.asarray(value)
        if array_value.dtype.hasobject:
            raise TypeError(
                f"convergence-statistics field {key!r} cannot use object dtype"
            )
        payload[key] = array_value
    np.savez_compressed(
        temporary_path,
        **payload,
    )
    os.replace(temporary_path, output_path)
    return output_path


def _delta_tag(tolerance: float) -> str:
    # ``repr(float)`` is the shortest round-trip representation, so two
    # distinct validated tolerances cannot silently target the same files.
    text = repr(float(tolerance)).lower()
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def _delta_mathtext(tolerance: float) -> str:
    text = f"{float(tolerance):.8g}".lower()
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e", maxsplit=1)
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def _plot_median_first_passage_steps(
    statistics: Mapping[str, np.ndarray],
    *,
    tolerance_index: int,
    outpath,
) -> None:
    layers = np.asarray(statistics["layers"], dtype=NP_INT_DTYPE)
    tolerances = np.asarray(statistics["tolerances"], dtype=NP_REAL_DTYPE)
    medians = np.asarray(
        statistics["median_first_passage_steps"],
        dtype=NP_REAL_DTYPE,
    )[tolerance_index]
    t_max = int(np.asarray(statistics["t_max"]).item())
    tolerance = float(tolerances[tolerance_index])

    finite = np.isfinite(medians)
    fig, ax = new_fig_ax(outside_legend=False)
    ax.plot(
        layers,
        np.where(finite, medians, np.nan),
        color="#0072B2",
        marker="o",
        linestyle="-",
        zorder=3,
    )

    time_scale = max(float(t_max), 1.0)
    infinity_offset = max(1.0, 0.08 * time_scale)
    infinity_y = float(t_max) + infinity_offset
    if np.any(~finite):
        ax.scatter(
            layers[~finite],
            np.full(np.count_nonzero(~finite), infinity_y),
            color="#0072B2",
            marker="^",
            s=32.0,
            zorder=4,
        )

    ax.set_xlabel(r"Number of layers $L$")
    ax.set_ylabel(
        "Median first-passage step "
        rf"$\widetilde{{T}}_L({_delta_mathtext(tolerance)})$"
    )
    ax.set_title(
        "Median convergence step over all runs at "
        rf"$\delta={_delta_mathtext(tolerance)}$"
    )
    ax.set_xticks(layers)
    ax.set_xticklabels([str(int(layer)) for layer in layers])

    if np.any(~finite):
        base_ticks = np.unique(
            np.rint(np.linspace(0.0, float(t_max), 6)).astype(NP_INT_DTYPE)
        )
        ax.set_yticks(
            np.concatenate((base_ticks.astype(NP_REAL_DTYPE), [infinity_y]))
        )
        ax.set_yticklabels(
            [str(int(value)) for value in base_ticks] + [r"$\infty$"]
        )
        ax.set_ylim(0.0, infinity_y + 0.45 * infinity_offset)
    else:
        ax.set_ylim(0.0, max(1.0, 1.03 * float(t_max)))
        ax.yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=6, integer=True)
        )
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, ax, str(outpath), outside_legend=False)


def _plot_failure_probability(
    statistics: Mapping[str, np.ndarray],
    *,
    tolerance_index: int,
    outpath,
) -> None:
    layers = np.asarray(statistics["layers"], dtype=NP_INT_DTYPE)
    tolerances = np.asarray(statistics["tolerances"], dtype=NP_REAL_DTYPE)
    failure_probabilities = np.asarray(
        statistics["failure_probabilities"],
        dtype=NP_REAL_DTYPE,
    )[tolerance_index]
    tolerance = float(tolerances[tolerance_index])

    fig, ax = new_fig_ax(outside_legend=False)
    ax.plot(
        layers,
        failure_probabilities,
        color="#D55E00",
        marker="s",
        linestyle="-",
        zorder=3,
    )
    ax.set_xlabel(r"Number of layers $L$")
    ax.set_ylabel(
        "Failure probability "
        rf"$p_{{\mathrm{{fail}},L}}({_delta_mathtext(tolerance)})$"
    )
    ax.set_title(
        "Failure probability at "
        rf"$\delta={_delta_mathtext(tolerance)}$"
    )
    ax.set_xticks(layers)
    ax.set_xticklabels([str(int(layer)) for layer in layers])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, ax, str(outpath), outside_legend=False)


def _plot_attainment_cdf(
    statistics: Mapping[str, np.ndarray],
    *,
    tolerance_index: int,
    outpath,
) -> None:
    layers = np.asarray(statistics["layers"], dtype=NP_INT_DTYPE)
    tolerances = np.asarray(statistics["tolerances"], dtype=NP_REAL_DTYPE)
    time_steps = np.asarray(statistics["time_steps"], dtype=NP_INT_DTYPE)
    attainment_cdf = np.asarray(
        statistics["attainment_cdf"],
        dtype=NP_REAL_DTYPE,
    )[tolerance_index]
    tolerance = float(tolerances[tolerance_index])

    base_cmap = matplotlib.colormaps.get_cmap("viridis")
    layer_colors = base_cmap(np.linspace(0.05, 0.95, layers.size))
    layer_cmap = matplotlib.colors.ListedColormap(layer_colors)
    layer_norm = matplotlib.colors.BoundaryNorm(
        np.arange(layers.size + 1, dtype=NP_REAL_DTYPE) - 0.5,
        layer_cmap.N,
    )

    fig, ax = new_fig_ax(
        outside_legend=True,
        legend_space_frac=0.18,
    )
    for layer_index in range(layers.size):
        ax.step(
            time_steps,
            attainment_cdf[layer_index],
            where="post",
            color=layer_colors[layer_index],
            linewidth=1.15,
        )

    scalar_mappable = matplotlib.cm.ScalarMappable(
        norm=layer_norm,
        cmap=layer_cmap,
    )
    scalar_mappable.set_array(np.arange(layers.size))
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        fraction=0.045,
        pad=0.025,
    )
    if layers.size <= 9:
        selected_indices = np.arange(layers.size, dtype=NP_INT_DTYPE)
    else:
        selected_indices = np.unique(
            np.rint(np.linspace(0, layers.size - 1, 8)).astype(NP_INT_DTYPE)
        )
    colorbar.set_ticks(selected_indices)
    colorbar.set_ticklabels([str(int(layers[index])) for index in selected_indices])
    colorbar.set_label(r"Number of layers $L$")

    ax.set_xlabel(r"Optimization step $t$")
    ax.set_ylabel(
        "Cumulative attainment "
        rf"$\widehat{{F}}_L(t;{_delta_mathtext(tolerance)})$"
    )
    ax.set_title(
        "First-passage cumulative attainment at "
        rf"$\delta={_delta_mathtext(tolerance)}$"
    )
    if time_steps.size == 1:
        ax.set_xlim(-0.05, 0.05)
        ax.set_xticks([0])
    else:
        ax.set_xlim(float(time_steps[0]), float(time_steps[-1]))
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(True, alpha=0.3)
    save_fig(
        fig,
        ax,
        str(outpath),
        outside_legend=True,
        legend_space_frac=0.18,
    )


def render_convergence_time_figures(
    statistics: Mapping[str, np.ndarray],
    figure_dir,
) -> tuple[Path, ...]:
    """Render median, failure-probability, and attainment-CDF figures."""
    output_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tolerances = np.asarray(statistics["tolerances"], dtype=NP_REAL_DTYPE)
    output_paths: list[Path] = []

    for tolerance_index, tolerance in enumerate(tolerances):
        tag = _delta_tag(float(tolerance))
        median_path = output_dir / f"convergence_median_steps_delta_{tag}.pdf"
        failure_path = (
            output_dir / f"convergence_failure_probability_delta_{tag}.pdf"
        )
        cdf_path = output_dir / f"convergence_attainment_cdf_delta_{tag}.pdf"

        _plot_median_first_passage_steps(
            statistics,
            tolerance_index=tolerance_index,
            outpath=median_path,
        )
        _plot_failure_probability(
            statistics,
            tolerance_index=tolerance_index,
            outpath=failure_path,
        )
        _plot_attainment_cdf(
            statistics,
            tolerance_index=tolerance_index,
            outpath=cdf_path,
        )
        output_paths.extend((median_path, failure_path, cdf_path))

    return tuple(output_paths)


def generate_convergence_time_outputs(
    energy_traces_by_layer: Mapping[int, np.ndarray],
    layers: Sequence[int],
    *,
    ground_energy: float,
    figure_dir,
    statistics_outpath=None,
    tolerances: Optional[Sequence[float]] = None,
    num_runs: Optional[int] = None,
    optimizer_steps: Optional[int] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> dict[str, np.ndarray]:
    """Compute, optionally save, and render all convergence-time outputs."""
    statistics = compute_convergence_time_statistics(
        energy_traces_by_layer,
        layers,
        ground_energy=ground_energy,
        tolerances=tolerances,
        num_runs=num_runs,
        optimizer_steps=optimizer_steps,
    )
    if metadata is not None:
        collisions = sorted(set(metadata).intersection(statistics))
        if collisions:
            raise ValueError(
                "convergence metadata collides with core field(s): "
                + ", ".join(collisions)
            )
        for key, value in metadata.items():
            if not isinstance(key, str) or not key:
                raise ValueError("convergence metadata keys must be non-empty strings")
            metadata_value = np.asarray(value)
            if metadata_value.dtype.hasobject:
                raise TypeError(
                    f"convergence metadata field {key!r} cannot use object dtype"
                )
            statistics[key] = metadata_value
    if statistics_outpath is not None:
        save_convergence_time_statistics(statistics, statistics_outpath)
    render_convergence_time_figures(statistics, figure_dir)
    return statistics
