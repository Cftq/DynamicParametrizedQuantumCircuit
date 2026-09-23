"""Fixed-cutoff QFIM rank summaries from archived random-point spectra.

This is the eigenvalue count at or above a fixed positive threshold, matching
the DPQC/reset-DPQC rank figures. It is not participation effective rank.
Numerical routines require only NumPy; no quantum or training code is loaded.
"""

from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np


DEFAULT_THRESHOLD = 1e-12


def _validated_threshold(threshold):
    value = np.asarray(threshold)
    if (
        value.ndim != 0 or not np.issubdtype(value.dtype, np.number)
        or np.iscomplexobj(value) or not np.isfinite(value)
    ):
        raise ValueError("QFIM rank threshold must be a finite positive real scalar.")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("QFIM rank threshold must be a finite positive real scalar.")
    return value


def compute_qfim_rank(eigenvalues_by_layer, *, threshold=DEFAULT_THRESHOLD):
    """Count lambda >= threshold for every saved sample, then summarize.

    Equality is included. Zeros and negative values do not pass a positive
    cutoff. Spectra with any nonfinite or nonreal entries are rejected rather
    than silently excluded from the sample population. Sample standard
    deviation uses ddof=1; a single sample has zero standard deviation/SEM.
    """
    threshold = _validated_threshold(threshold)
    if not eigenvalues_by_layer:
        raise ValueError("Saved random-point QFIM spectra are required.")
    if any(
        isinstance(layer, (bool, np.bool_))
        or not isinstance(layer, (int, np.integer)) or layer < 1
        for layer in eigenvalues_by_layer
    ):
        raise ValueError("Layer keys must be positive integers.")
    layers = np.asarray(sorted(eigenvalues_by_layer), dtype=np.int64)
    rank_by_layer = {}
    statistics = []
    minima = []
    maxima = []
    sample_counts = []
    parameter_counts = []
    for layer in layers:
        values = np.asarray(eigenvalues_by_layer[int(layer)])
        if (
            values.ndim != 2 or min(values.shape) == 0
            or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values) or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                f"Layer {layer}: eigenvalues must be a nonempty finite real "
                "array with shape (samples, parameters)."
            )
        # Compare in float64, as in the DPQC visualizer. Do not clip or mask
        # eigenvalues before applying this inclusive fixed threshold.
        with np.errstate(over="ignore"):
            values = values.astype(np.float64, copy=False)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Layer {layer}: eigenvalues must be finite float64 values.")
        ranks = np.count_nonzero(values >= threshold, axis=1).astype(np.int64)
        std = float(np.std(ranks, ddof=1)) if ranks.size > 1 else 0.0
        rank_by_layer[int(layer)] = ranks
        statistics.append((np.mean(ranks), std, std / math.sqrt(ranks.size)))
        minima.append(int(np.min(ranks)))
        maxima.append(int(np.max(ranks)))
        sample_counts.append(ranks.size)
        parameter_counts.append(values.shape[1])
    return {
        "threshold": threshold,
        "layers": layers,
        "rank_by_layer": rank_by_layer,
        "num_samples_by_layer": np.asarray(sample_counts, dtype=np.int64),
        "num_parameters_by_layer": np.asarray(parameter_counts, dtype=np.int64),
        "min": np.asarray(minima, dtype=np.int64),
        "max": np.asarray(maxima, dtype=np.int64),
        **dict(zip(("mean", "std", "sem"), np.asarray(statistics).T)),
    }


def _threshold_tex(threshold):
    exponent = int(math.floor(math.log10(threshold)))
    power = 10.0 ** exponent
    if power > 0.0 and math.isclose(threshold / power, 1.0, rel_tol=1e-12, abs_tol=0.0):
        return rf"10^{{{exponent}}}"
    return f"{threshold:g}"


def save_qfim_rank_outputs(
    result, output_dir, *, keep_key, title=None, source_path=None, metadata=None,
):
    """Save DPQC-style rank mean/SEM/min/max PDF and integer sample ranks.

    Archive metadata is preserved under ``source_`` keys so input values
    cannot overwrite the rank definition or the derived statistics.
    """
    if __package__:
        from .plot import new_fig_ax, save_fig
    else:
        from plot import new_fig_ax, save_fig
    from matplotlib.ticker import MaxNLocator

    if not isinstance(keep_key, str) or not re.fullmatch(r"[A-Za-z0-9_]+", keep_key):
        raise ValueError("keep_key must contain only letters, digits, and underscores.")
    threshold = _validated_threshold(result["threshold"])
    # Match DPQC's ge_1e-12 convention while retaining significant digits for
    # arbitrary custom thresholds, avoiding accidental filename collisions.
    threshold_tag = np.format_float_scientific(threshold, unique=True, trim="-", exp_digits=2).replace("+", "")
    basename = f"qfim_rank_mean_sem_min_max_random_points_ge_{threshold_tag}_{keep_key}"
    output_dir = Path(output_dir)
    figure_path = output_dir / f"{basename}.pdf"
    statistics_path = output_dir / f"{basename}.npz"
    arrays = {
        key: np.asarray(value) for key, value in result.items()
        if key != "rank_by_layer"
    }
    arrays.update({
        "keep_key": np.asarray(keep_key),
        "rank_criterion": np.asarray("eigenvalue >= threshold (inclusive)"),
        "rank_definition": np.asarray("number of saved QFIM eigenvalues at or above a fixed positive cutoff"),
        "sample_source": np.asarray("saved random-point QFIM spectra; no QFIM or training recomputation"),
    })
    if title is not None:
        arrays["title"] = np.asarray(title)
    if source_path is not None:
        arrays["source_path"] = np.asarray(str(source_path))
    for layer in result["layers"]:
        arrays[f"L{layer}_rank"] = result["rank_by_layer"][int(layer)]
    for key, value in (metadata or {}).items():
        if not isinstance(key, str) or f"source_{key}" in arrays:
            raise ValueError(f"Invalid or reserved metadata key: {key!r}.")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"Metadata {key!r} must not require pickle storage.")
        arrays[f"source_{key}"] = array
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(statistics_path, **arrays)

    layers = result["layers"]
    fig, ax = new_fig_ax(outside_legend=False)
    ax.errorbar(
        layers, result["mean"], yerr=result["sem"], marker="o", linestyle="-",
        linewidth=1.4, markersize=5.5, capsize=3.0, elinewidth=0.9,
        color="C0", label=r"Mean $\pm$ SEM", zorder=3,
    )
    ax.plot(
        layers, result["min"], marker="v", linestyle="--", linewidth=1.2,
        markersize=5.0, color="C2", label="Minimum",
    )
    ax.plot(
        layers, result["max"], marker="^", linestyle="--", linewidth=1.2,
        markersize=5.0, color="C3", label="Maximum",
    )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(rf"QFIM rank ($\lambda_i \geq {_threshold_tex(threshold)}$)")
    ax.set_xticks(layers)
    ax.set_xticklabels([str(int(layer)) for layer in layers])
    ax.set_ylim(bottom=0.0)
    if np.max(result["max"]) == 0:
        ax.set_ylim(top=1.0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    if title is not None:
        ax.set_title(title)
    save_fig(fig, ax, str(figure_path), outside_legend=False)
    return {"figure_path": figure_path, "statistics_path": statistics_path}
