#!/usr/bin/env python
# coding: utf-8
"""Visualize saved Unitary-PQC numerical results.

Run unitary_pqc_overparam_compute.py first. This script loads saved .npz
results under figs/unitary_pqc/h_<h_param>/numerical_results and generates
numerical figures without recomputing VQE or QFIM quantities. Circuit drawings
are handled independently by unitary_pqc_overparam_draw_circuits.py.
QFIM traces are reconstructed from the saved raw eigenspectra and include only
eigenvalues at or above the configured effective-rank threshold.

    python src/unitary_pqc/unitary_pqc_overparam_visualize.py --h-param 0.1
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

_MODULE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _MODULE_DIR.parent
_COMMON_DIR = _SRC_DIR / "common"
for _path in (_MODULE_DIR, _COMMON_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


import config_overparam as cfg


def _finite_float(value: str) -> float:
    """Parse one finite floating-point command-line value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("value must be a finite number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be a finite number")
    return parsed


def _parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Visualize saved Unitary-PQC results for one Hamiltonian "
            "parameter h."
        )
    )
    parser.add_argument(
        "--h-param",
        type=_finite_float,
        default=float(cfg.H_PARAM),
        help=(
            "Hamiltonian parameter h whose saved results are loaded and "
            "visualized (default: H_PARAM from config_overparam.py)."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    _CLI_ARGS = _parse_cli_args()
else:
    _CLI_ARGS = argparse.Namespace(h_param=float(cfg.H_PARAM))

_SELECTED_H_PARAM = float(_CLI_ARGS.h_param)
if not math.isfinite(_SELECTED_H_PARAM):
    raise ValueError("h_param must be a finite number.")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import unitary_pqc_overparam_compute as upqc
from dpqc_overparam_common import load_npz_result as _load_npz_result_unchecked


NP_REAL_DTYPE = np.float64
NP_INT_DTYPE = np.int64
ENERGY_ERROR_PLOT_EPS = NP_REAL_DTYPE(1e-12)
QFIM_TRACE_EIGENVALUE_THRESHOLD = NP_REAL_DTYPE(
    cfg.QFIM_EFFECTIVE_RANK_THRESHOLD
)
HESSIAN_RANDOM_SCHEMA_VERSION = int(upqc.HESSIAN_RANDOM_SCHEMA_VERSION)
HESSIAN_RANK_DEFINITION = str(upqc.HESSIAN_RANK_DEFINITION)
HESSIAN_CONDITION_NUMBER_DEFINITION = str(
    upqc.HESSIAN_CONDITION_NUMBER_DEFINITION
)
FINAL_ENERGY_ERROR_DETAIL_THRESHOLD = NP_REAL_DTYPE(
    getattr(cfg, "FINAL_ENERGY_ERROR_DETAIL_THRESHOLD", 6e-1)
)
SUCCESS_PROBABILITY_FIGURE_THRESHOLDS = np.asarray(
    cfg.SUCCESS_PROBABILITY_FIGURE_THRESHOLDS,
    dtype=NP_REAL_DTYPE,
)

# Shared visual language: metric determines color; statistic determines line.
METRIC_COLORS = {
    "qfim": "#0072B2",
    "hs": "#D55E00",
    "hessian": "#CC79A7",
    "energy": "#E69F00",
}
STATISTIC_LINESTYLES = {"mean": "-"}


def _validate_result_h_param(
    result: dict,
    result_path,
    *,
    expected_h_param: float,
    required: bool = False,
) -> None:
    """Ensure an archive belongs to the selected Hamiltonian parameter."""
    if "h_param" not in result:
        if required:
            raise KeyError(
                "The required Unitary-PQC archive does not contain h_param "
                f"metadata: {Path(result_path).resolve()}"
            )
        return

    archived_value = np.asarray(result["h_param"])
    if (
        archived_value.size != 1
        or not np.issubdtype(archived_value.dtype, np.number)
        or np.iscomplexobj(archived_value)
    ):
        raise ValueError(
            "Saved h_param must be one real numeric scalar in archive: "
            f"{Path(result_path).resolve()}"
        )
    archived_h_param = float(archived_value.reshape(-1)[0])
    if not math.isfinite(archived_h_param):
        raise ValueError(
            "Saved h_param must be finite in archive: "
            f"{Path(result_path).resolve()}"
        )
    if NP_REAL_DTYPE(archived_h_param) != NP_REAL_DTYPE(expected_h_param):
        raise ValueError(
            "Saved Unitary-PQC h_param does not match --h-param: "
            f"{archived_h_param} != {expected_h_param} in "
            f"{Path(result_path).resolve()}"
        )


def _load_required_result(
    path: str,
    *,
    expected_h_param: Optional[float] = None,
    require_h_param: bool = False,
) -> dict:
    """Load a compute-stage result or explain how to generate it."""
    result_path = Path(path).resolve()
    selected_h_param = float(
        upqc.h_param if expected_h_param is None else expected_h_param
    )
    if not result_path.is_file():
        compute_script = _MODULE_DIR / "unitary_pqc_overparam_compute.py"
        raise FileNotFoundError(
            "Required Unitary-PQC numerical result is missing:\n"
            f"  {result_path}\n"
            "Run the numerical pipeline to successful completion before "
            "visualizing:\n"
            f'  "{sys.executable}" "{compute_script}" '
            f"--h-param {selected_h_param}"
        )
    result = _load_npz_result_unchecked(str(result_path))
    _validate_result_h_param(
        result,
        result_path,
        expected_h_param=selected_h_param,
        required=require_h_param,
    )
    return result


def _validated_num_trials(value) -> int:
    """Return a positive integer trial count without lossy coercion."""
    raw_value = np.asarray(value)
    if (
        raw_value.size != 1
        or not np.issubdtype(raw_value.dtype, np.number)
        or np.iscomplexobj(raw_value)
    ):
        raise TypeError("num_trials must be one real numeric scalar.")
    numeric_value = float(raw_value.reshape(-1)[0])
    if (
        not np.isfinite(numeric_value)
        or numeric_value != np.rint(numeric_value)
        or numeric_value <= 0.0
    ):
        raise ValueError("num_trials must be one finite positive integer.")
    return int(numeric_value)


def _validated_final_energy_samples(
    energy_traces_by_layer: dict,
    layers,
    *,
    ground_energy: float,
    num_trials: int,
):
    """Validate VQE histories and return their final sampled energies."""
    layer_values = np.asarray(layers)
    if layer_values.ndim != 1 or layer_values.size == 0:
        raise ValueError("layers must be a non-empty one-dimensional array.")
    if not np.issubdtype(layer_values.dtype, np.number) or np.iscomplexobj(
        layer_values
    ):
        raise TypeError("layers must contain real numeric values.")
    layer_values_float = np.asarray(layer_values, dtype=NP_REAL_DTYPE)
    if (
        not np.all(np.isfinite(layer_values_float))
        or not np.all(layer_values_float == np.rint(layer_values_float))
    ):
        raise ValueError("layers must contain only finite integer values.")
    layer_values = layer_values_float.astype(NP_INT_DTYPE)
    if np.any(layer_values <= 0) or np.unique(layer_values).size != layer_values.size:
        raise ValueError("layers must be positive and contain no duplicates.")

    ground_energy = float(ground_energy)
    if not np.isfinite(ground_energy):
        raise ValueError("ground_energy must be finite.")
    num_trials = _validated_num_trials(num_trials)

    final_energies_by_layer = {}
    for layer in layer_values:
        L = int(layer)
        if L not in energy_traces_by_layer:
            raise ValueError(f"Missing energy traces for L={L}.")
        raw_traces = np.asarray(energy_traces_by_layer[L])
        if not np.issubdtype(raw_traces.dtype, np.number) or np.iscomplexobj(
            raw_traces
        ):
            raise TypeError(f"Energy traces for L={L} must be real numeric data.")
        traces = np.asarray(raw_traces, dtype=NP_REAL_DTYPE)
        if traces.ndim != 2 or traces.shape[0] != num_trials or traces.shape[1] == 0:
            raise ValueError(
                f"Energy traces for L={L} must have shape "
                f"({num_trials}, num_iterations>0), got {traces.shape}."
            )
        if not np.all(np.isfinite(traces)):
            raise ValueError(f"Energy traces for L={L} are not all finite.")
        final_energies = traces[:, -1]
        final_energies_by_layer[L] = final_energies

    return layer_values, final_energies_by_layer, ground_energy, num_trials


def _validated_success_probability_thresholds(thresholds) -> np.ndarray:
    """Normalize the configured DPQC-style final-energy tolerances."""
    raw_thresholds = np.asarray(thresholds)
    if not np.issubdtype(raw_thresholds.dtype, np.number) or np.iscomplexobj(
        raw_thresholds
    ):
        raise TypeError("Success-probability thresholds must be real numeric data.")
    normalized = np.asarray(raw_thresholds, dtype=NP_REAL_DTYPE)
    if normalized.ndim != 1 or normalized.size == 0:
        raise ValueError(
            "Success-probability thresholds must be a non-empty 1D array."
        )
    if not np.all(np.isfinite(normalized)) or np.any(normalized <= 0.0):
        raise ValueError(
            "Success-probability thresholds must be positive and finite."
        )
    if np.any(np.diff(normalized) >= 0.0):
        raise ValueError(
            "Success-probability thresholds must be strictly decreasing."
        )
    return normalized


def _multiple_tolerance_success_statistics(
    final_energies_by_layer: dict,
    layers,
    *,
    ground_energy: float,
    num_trials: int,
    thresholds,
) -> dict:
    """Compute DPQC-compatible final-energy success fractions."""
    layers = np.asarray(layers, dtype=NP_INT_DTYPE)
    thresholds = _validated_success_probability_thresholds(thresholds)
    num_trials = _validated_num_trials(num_trials)
    final_energies = np.stack(
        [
            np.asarray(final_energies_by_layer[int(L)], dtype=NP_REAL_DTYPE)
            for L in layers
        ],
        axis=0,
    )
    expected_shape = (layers.size, num_trials)
    if final_energies.shape != expected_shape:
        raise ValueError(
            f"Final-energy shape mismatch: expected {expected_shape}, "
            f"got {final_energies.shape}."
        )
    if not np.all(np.isfinite(final_energies)):
        raise ValueError("Final energies must all be finite.")

    raw_final_energy_errors = final_energies - NP_REAL_DTYPE(ground_energy)
    # Match the DPQC success metric: a tiny numerical undershoot below the
    # exact ground energy is treated as zero error.  Distribution figures use
    # the absolute error separately and therefore retain the undershoot size.
    final_energy_errors = np.maximum(raw_final_energy_errors, 0.0)
    success_indicators = (
        final_energy_errors[:, :, None] <= thresholds[None, None, :]
    ).astype(NP_INT_DTYPE)
    success_counts = np.sum(success_indicators, axis=1, dtype=NP_INT_DTYPE)
    success_probabilities = (
        success_counts.astype(NP_REAL_DTYPE) / NP_REAL_DTYPE(num_trials)
    )

    if np.any(np.diff(success_counts, axis=1) > 0):
        raise AssertionError(
            "Success counts must be nonincreasing for decreasing thresholds."
        )
    if not np.allclose(
        success_probabilities,
        success_counts.astype(NP_REAL_DTYPE) / NP_REAL_DTYPE(num_trials),
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Success probabilities do not match count / trials.")

    return {
        "layers": layers,
        "thresholds": thresholds,
        "num_trials": int(num_trials),
        "final_energies": final_energies,
        "raw_final_energy_errors": raw_final_energy_errors,
        "final_energy_errors": final_energy_errors,
        "success_indicators": success_indicators,
        "success_counts": success_counts,
        "success_probabilities": success_probabilities,
    }


def _violin_width_from_positions(
    positions,
    *,
    scale: float = 0.60,
    default: float = 0.75,
) -> float:
    positions = np.asarray(positions, dtype=NP_REAL_DTYPE)
    if positions.size <= 1:
        return float(default)
    return float(scale * np.min(np.diff(np.sort(positions))))


def _plot_final_energy_error_violin(
    errors_by_layer: dict,
    layers,
    *,
    outpath: str,
    log_scale: bool,
) -> None:
    """Plot the DPQC-style final ground-energy-error violin figure."""
    layers = np.asarray(layers, dtype=NP_INT_DTYPE)
    positions = layers.astype(NP_REAL_DTYPE)
    datasets = []
    for L in layers:
        values = np.asarray(errors_by_layer[int(L)], dtype=NP_REAL_DTYPE)
        if log_scale:
            values = np.maximum(values, ENERGY_ERROR_PLOT_EPS)
        datasets.append(
            upqc._make_violin_ready(
                values,
                ensure_positive=log_scale,
                tiny=float(ENERGY_ERROR_PLOT_EPS),
            )
        )

    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    violin = ax.violinplot(
        datasets,
        positions=positions,
        widths=_violin_width_from_positions(positions),
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    colors = [
        upqc.cmap(idx / max(len(layers), 1))
        for idx in range(len(layers))
    ]
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.20)
        body.set_linewidth(1.0)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        artist = violin.get(key)
        if artist is not None:
            artist.set_color("black")
            artist.set_linewidth(1.0)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(int(L)) for L in layers])
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Final energy error")
    upqc.set_prx_title("Final energy-error distributions", ax=ax)
    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.3)
    else:
        ax.grid(True, axis="y", alpha=0.3)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=False)


def _beeswarm_offsets_1d(
    y_for_layout: np.ndarray,
    *,
    max_width: float = 0.32,
    nbins: Optional[int] = None,
) -> np.ndarray:
    """Return deterministic within-bin offsets for a one-dimensional swarm."""
    y = np.asarray(y_for_layout, dtype=NP_REAL_DTYPE).reshape(-1)
    n = int(y.size)
    if n == 0:
        return np.asarray([], dtype=NP_REAL_DTYPE)
    if n == 1:
        return np.zeros(1, dtype=NP_REAL_DTYPE)
    if nbins is None:
        nbins = max(6, int(np.ceil(np.sqrt(n))))

    ymin = float(np.min(y))
    ymax = float(np.max(y))
    if np.isclose(ymin, ymax):
        bin_ids = np.zeros(n, dtype=NP_INT_DTYPE)
    else:
        edges = np.linspace(ymin, ymax, int(nbins) + 1)
        bin_ids = np.digitize(y, edges[1:-1], right=False).astype(NP_INT_DTYPE)

    offsets = np.zeros(n, dtype=NP_REAL_DTYPE)
    for bin_id in np.unique(bin_ids):
        indices = np.where(bin_ids == bin_id)[0]
        count = int(indices.size)
        if count <= 1:
            continue
        indices = indices[np.argsort(y[indices])]
        order = np.zeros(count, dtype=NP_REAL_DTYPE)
        for index in range(1, count):
            amplitude = (index + 1) // 2
            order[index] = amplitude if index % 2 == 1 else -amplitude
        max_abs = float(np.max(np.abs(order)))
        if max_abs > 0.0:
            order = order / max_abs * float(max_width)
        offsets[indices] = order
    return offsets


def _plot_final_energy_error_beeswarm(
    errors_by_layer: dict,
    layers,
    *,
    outpath: str,
    ylabel: str = "Final energy error",
    title: str = "Final energy-error distributions",
    log_scale: bool = False,
) -> bool:
    """Plot deterministic DPQC-style final-error samples and their medians."""
    layers = np.asarray(layers, dtype=NP_INT_DTYPE)
    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    plotted = False
    for layer_index, L in enumerate(layers):
        values = np.asarray(errors_by_layer[int(L)], dtype=NP_REAL_DTYPE).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        plotted = True
        y_plot = (
            np.maximum(values, ENERGY_ERROR_PLOT_EPS)
            if log_scale
            else values
        )
        y_for_layout = np.log10(y_plot) if log_scale else y_plot
        x = NP_REAL_DTYPE(L) + _beeswarm_offsets_1d(y_for_layout)
        color = upqc.cmap(layer_index / max(len(layers), 1))
        ax.scatter(
            x,
            y_plot,
            s=18.0,
            alpha=0.65,
            color=color,
            edgecolors="black",
            linewidths=0.25,
            zorder=3,
        )
        median = float(np.median(y_plot))
        ax.hlines(
            median,
            xmin=float(L) - 0.32,
            xmax=float(L) + 0.32,
            color="black",
            linewidth=1.0,
            zorder=4,
        )

    if not plotted:
        plt.close(plt.gcf())
        return False
    ax.set_xticks(layers)
    ax.set_xticklabels([str(int(L)) for L in layers])
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    upqc.set_prx_title(title, ax=ax)
    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.3)
    else:
        ax.grid(True, axis="y", alpha=0.3)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=False)
    return True


def _success_probability_threshold_label(threshold: float) -> str:
    threshold = float(threshold)
    hundredths = int(np.rint(100.0 * threshold))
    if 1 <= hundredths <= 10 and np.isclose(
        threshold,
        hundredths * 1e-2,
        rtol=1e-12,
        atol=0.0,
    ):
        if hundredths == 1:
            return r"$\delta=10^{-2}$"
        if hundredths == 10:
            return r"$\delta=10^{-1}$"
        return rf"$\delta={hundredths}\times 10^{{-2}}$"
    exponent = int(np.floor(np.log10(threshold)))
    mantissa = threshold / (10.0 ** exponent)
    if np.isclose(mantissa, 1.0, rtol=1e-12, atol=0.0):
        return rf"$\delta=10^{{{exponent}}}$"
    return rf"$\delta={mantissa:.3g}\times 10^{{{exponent}}}$"


def _plot_success_probability_multiple_tolerances(
    statistics: dict,
    *,
    outpath: str,
) -> None:
    """Plot the fraction of trials below each final-energy-error threshold."""
    layers = np.asarray(statistics["layers"], dtype=NP_INT_DTYPE)
    thresholds = np.asarray(statistics["thresholds"], dtype=NP_REAL_DTYPE)
    probabilities = np.asarray(
        statistics["success_probabilities"],
        dtype=NP_REAL_DTYPE,
    )
    num_trials = int(statistics["num_trials"])
    if probabilities.shape != (layers.size, thresholds.size):
        raise ValueError("Success-probability array has an inconsistent shape.")

    colors = matplotlib.colormaps.get_cmap("viridis")(
        np.linspace(0.08, 0.92, thresholds.size)
    )
    markers = ("o", "s", "^", "D", "P", "v", "X")
    linestyles = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))
    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    for plot_index, threshold_index in enumerate(
        np.argsort(thresholds, kind="stable")
    ):
        threshold = thresholds[threshold_index]
        ax.plot(
            layers,
            probabilities[:, threshold_index],
            color=colors[plot_index],
            marker=markers[plot_index % len(markers)],
            linestyle=linestyles[plot_index % len(linestyles)],
            linewidth=1.35,
            markersize=4.5,
            label=_success_probability_threshold_label(threshold),
            zorder=3,
        )

    ax.set_xlabel(r"Number of Layers $L$")
    ax.set_ylabel(r"Empirical success probability $\widehat{S}_L(\delta)$")
    upqc.set_prx_title(
        "Success probability at multiple accuracy levels "
        f"({num_trials} independent trials)",
        ax=ax,
    )
    ax.set_xticks(layers)
    ax.set_xticklabels([str(int(L)) for L in layers])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        title=rf"$R={num_trials}$ independent trials",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=True)


def _energy_error_threshold_tag(threshold: float) -> str:
    return f"{float(threshold):.0e}".replace("+", "")


def _plot_vqe_ground_truth_error_results() -> dict:
    """Create the DPQC-style final-error and success-fraction figures."""
    (
        layers,
        final_energies_by_layer,
        ground_energy,
        num_trials,
    ) = _validated_final_energy_samples(
        upqc.energy_traces_by_layer,
        upqc.layer_list,
        ground_energy=upqc.smallest_eigval,
        num_trials=upqc.num_runs,
    )
    final_absolute_errors_by_layer = {
        int(L): np.abs(final_energies_by_layer[int(L)] - ground_energy)
        for L in layers
    }
    energy_dir = upqc.energy_fig_dir

    _plot_final_energy_error_violin(
        final_absolute_errors_by_layer,
        layers,
        outpath=os.path.join(energy_dir, "final_energy_error.pdf"),
        log_scale=False,
    )
    _plot_final_energy_error_beeswarm(
        final_absolute_errors_by_layer,
        layers,
        outpath=os.path.join(energy_dir, "final_energy_error_beeswarm.pdf"),
        log_scale=False,
    )
    _plot_final_energy_error_beeswarm(
        final_absolute_errors_by_layer,
        layers,
        outpath=os.path.join(
            energy_dir,
            "final_energy_error_beeswarm_logscale.pdf",
        ),
        log_scale=True,
    )

    detail_threshold = float(FINAL_ENERGY_ERROR_DETAIL_THRESHOLD)
    if not np.isfinite(detail_threshold) or detail_threshold <= 0.0:
        raise ValueError(
            "FINAL_ENERGY_ERROR_DETAIL_THRESHOLD must be finite and positive."
        )
    detail_errors_by_layer = {
        int(L): final_absolute_errors_by_layer[int(L)][
            final_absolute_errors_by_layer[int(L)] <= detail_threshold
        ]
        for L in layers
    }
    detail_path = os.path.join(
        energy_dir,
        "final_energy_error_beeswarm_below_"
        f"{_energy_error_threshold_tag(detail_threshold)}.pdf",
    )
    if not _plot_final_energy_error_beeswarm(
        detail_errors_by_layer,
        layers,
        outpath=detail_path,
        ylabel=(
            "Final energy error "
            rf"($\Delta E \leq {detail_threshold:g}$)"
        ),
        title=(
            "Detailed final energy-error distributions "
            rf"($\Delta E \leq {detail_threshold:g}$)"
        ),
        log_scale=False,
    ):
        warnings.warn(
            "No final energy errors satisfy "
            f"error <= {detail_threshold:g}; skipping the detailed "
            "beeswarm figure.",
            RuntimeWarning,
            stacklevel=2,
        )

    success_statistics = _multiple_tolerance_success_statistics(
        final_energies_by_layer,
        layers,
        ground_energy=ground_energy,
        num_trials=num_trials,
        thresholds=SUCCESS_PROBABILITY_FIGURE_THRESHOLDS,
    )
    _plot_success_probability_multiple_tolerances(
        success_statistics,
        outpath=os.path.join(
            energy_dir,
            "success_probability_multiple_tolerances.pdf",
        ),
    )
    return success_statistics


def _validated_positive_integer_scalar(value, *, name: str) -> int:
    """Return one positive integer scalar without silently truncating it."""
    raw = np.asarray(value)
    if (
        raw.size != 1
        or not np.issubdtype(raw.dtype, np.number)
        or np.iscomplexobj(raw)
    ):
        raise TypeError(f"{name} must be one real numeric scalar.")
    scalar = float(raw.reshape(-1)[0])
    if not np.isfinite(scalar) or scalar <= 0.0 or scalar != np.rint(scalar):
        raise ValueError(f"{name} must be one finite positive integer.")
    return int(scalar)


def _validated_qfim_layers(result: dict, *, description: str) -> list[int]:
    """Validate and return an archive's ordered layer vector."""
    if "layers" not in result:
        raise KeyError(f"{description} is missing the 'layers' array.")
    raw = np.asarray(result["layers"])
    if (
        raw.ndim != 1
        or raw.size == 0
        or not np.issubdtype(raw.dtype, np.number)
        or np.iscomplexobj(raw)
    ):
        raise TypeError(f"{description} layers must be a non-empty real 1D array.")
    values = np.asarray(raw, dtype=NP_REAL_DTYPE)
    if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
        raise ValueError(f"{description} layers must contain finite integers.")
    layers = values.astype(NP_INT_DTYPE)
    if np.any(layers <= 0) or np.unique(layers).size != layers.size:
        raise ValueError(
            f"{description} layers must be positive and contain no duplicates."
        )
    return [int(L) for L in layers]


def _validated_qfim_sample_iters(result: dict, *, description: str) -> np.ndarray:
    """Validate and return an archive's strictly increasing sample iterations."""
    if "sample_iters" not in result:
        raise KeyError(f"{description} is missing the 'sample_iters' array.")
    raw = np.asarray(result["sample_iters"])
    if (
        raw.ndim != 1
        or raw.size == 0
        or not np.issubdtype(raw.dtype, np.number)
        or np.iscomplexobj(raw)
    ):
        raise TypeError(
            f"{description} sample_iters must be a non-empty real 1D array."
        )
    values = np.asarray(raw, dtype=NP_REAL_DTYPE)
    if not np.all(np.isfinite(values)) or not np.all(values == np.rint(values)):
        raise ValueError(
            f"{description} sample_iters must contain finite integers."
        )
    sample_iters = values.astype(NP_INT_DTYPE)
    if np.any(sample_iters < 0) or np.any(np.diff(sample_iters) <= 0):
        raise ValueError(
            f"{description} sample_iters must be nonnegative and strictly increasing."
        )
    return sample_iters


def _require_matching_integer_sequence(
    actual,
    expected,
    *,
    actual_name: str,
    expected_name: str,
) -> None:
    actual_array = np.asarray(actual, dtype=NP_INT_DTYPE)
    expected_array = np.asarray(expected, dtype=NP_INT_DTYPE)
    if not np.array_equal(actual_array, expected_array):
        raise ValueError(
            f"{actual_name} does not match {expected_name}: "
            f"{actual_array.tolist()} != {expected_array.tolist()}."
        )


def _validate_raw_qfim_archive_metadata(
    result: dict,
    *,
    description: str,
    expected_keep_key: str,
    expected_analysis_kind: str,
) -> None:
    """Reject threshold-masked or otherwise incompatible QFIM archives."""
    for key in (
        "keep_key",
        "analysis_kind",
        "num_params_per_layer",
        "qfim_effective_rank_threshold",
        "eigenvalues_threshold_masked",
        "eigenvalue_order",
    ):
        if key not in result:
            raise KeyError(f"{description} is missing the {key!r} metadata.")

    keep_key = np.asarray(result["keep_key"])
    if keep_key.size != 1 or str(keep_key.reshape(-1)[0]) != expected_keep_key:
        raise ValueError(
            f"{description} keep_key does not match {expected_keep_key!r}."
        )

    analysis_kind = np.asarray(result["analysis_kind"])
    if (
        analysis_kind.size != 1
        or str(analysis_kind.reshape(-1)[0]) != expected_analysis_kind
    ):
        raise ValueError(
            f"{description} analysis_kind does not match "
            f"{expected_analysis_kind!r}."
        )

    params_per_layer = _validated_positive_integer_scalar(
        result["num_params_per_layer"],
        name=f"{description} num_params_per_layer",
    )
    if params_per_layer != int(upqc.num_params_per_layer):
        raise ValueError(
            f"{description} num_params_per_layer {params_per_layer} does not "
            f"match {int(upqc.num_params_per_layer)}."
        )

    threshold = np.asarray(result["qfim_effective_rank_threshold"])
    if (
        threshold.size != 1
        or not np.issubdtype(threshold.dtype, np.number)
        or np.iscomplexobj(threshold)
    ):
        raise TypeError(
            f"{description} qfim_effective_rank_threshold must be real scalar."
        )
    archived_threshold = float(threshold.reshape(-1)[0])
    if (
        not np.isfinite(archived_threshold)
        or NP_REAL_DTYPE(archived_threshold)
        != QFIM_TRACE_EIGENVALUE_THRESHOLD
    ):
        raise ValueError(
            f"{description} QFIM threshold {archived_threshold!r} does not "
            f"match {float(QFIM_TRACE_EIGENVALUE_THRESHOLD)!r}."
        )

    masked = np.asarray(result["eigenvalues_threshold_masked"])
    if masked.size != 1 or bool(masked.reshape(-1)[0]):
        raise ValueError(
            f"{description} must contain raw, non-threshold-masked eigenvalues."
        )

    eigenvalue_order = np.asarray(result["eigenvalue_order"])
    if (
        eigenvalue_order.size != 1
        or str(eigenvalue_order.reshape(-1)[0]) != "descending"
    ):
        raise ValueError(f"{description} eigenvalues must be stored in descending order.")


def _validated_random_qfim_eigs(
    result: dict,
    layers,
    *,
    num_samples: int,
    description: str,
) -> dict[int, np.ndarray]:
    """Load finite raw random-point spectra with shape sample x eigenvalue."""
    eigs_by_layer = {}
    for L in layers:
        key = f"L{int(L)}_eigs_desc"
        if key not in result:
            raise KeyError(f"{description} is missing {key!r}.")
        raw = np.asarray(result[key])
        if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
            raise TypeError(f"{description} {key} must contain real numeric data.")
        eigs = np.asarray(raw, dtype=NP_REAL_DTYPE)
        expected_shape = (
            int(num_samples),
            int(upqc.num_params_per_layer) * int(L),
        )
        if eigs.shape != expected_shape:
            raise ValueError(
                f"{description} {key} must have shape {expected_shape}, "
                f"got {eigs.shape}."
            )
        if not np.all(np.isfinite(eigs)):
            raise FloatingPointError(f"{description} {key} contains non-finite values.")
        eigs_by_layer[int(L)] = eigs
    return eigs_by_layer


def _validated_qfim_eigs_history(
    result: dict,
    layers,
    sample_iters,
    *,
    num_runs: int,
    description: str,
) -> dict[int, np.ndarray]:
    """Load finite raw histories with shape run x sampled-time x eigenvalue."""
    eigs_by_layer = {}
    num_sample_iters = int(np.asarray(sample_iters).size)
    for L in layers:
        key = f"L{int(L)}"
        if key not in result:
            raise KeyError(f"{description} is missing {key!r}.")
        raw = np.asarray(result[key])
        if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
            raise TypeError(f"{description} {key} must contain real numeric data.")
        eigs = np.asarray(raw, dtype=NP_REAL_DTYPE)
        expected_shape = (
            int(num_runs),
            num_sample_iters,
            int(upqc.num_params_per_layer) * int(L),
        )
        if eigs.shape != expected_shape:
            raise ValueError(
                f"{description} {key} must have shape {expected_shape}, "
                f"got {eigs.shape}."
            )
        if not np.all(np.isfinite(eigs)):
            raise FloatingPointError(f"{description} {key} contains non-finite values.")
        eigs_by_layer[int(L)] = eigs
    return eigs_by_layer


def qfim_trace_at_or_above_rank_threshold(eigenvalues: np.ndarray) -> np.ndarray:
    """Sum eigenvalues at/above the cutoff; preserve invalid spectra as NaN."""
    raw = np.asarray(eigenvalues)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise TypeError("QFIM eigenvalues must be real numeric data.")
    eigs = np.asarray(raw, dtype=NP_REAL_DTYPE)
    if eigs.ndim == 0:
        raise ValueError("QFIM eigenvalues must have an eigenvalue axis.")
    finite_spectrum = np.all(np.isfinite(eigs), axis=-1)
    trace = np.sum(
        np.where(
            eigs >= QFIM_TRACE_EIGENVALUE_THRESHOLD,
            eigs,
            NP_REAL_DTYPE(0.0),
        ),
        axis=-1,
        dtype=NP_REAL_DTYPE,
    )
    return np.where(finite_spectrum, trace, NP_REAL_DTYPE(np.nan))


def _qfim_threshold_tex(threshold: float) -> str:
    threshold = float(threshold)
    if threshold <= 0.0:
        return f"{threshold:g}"
    exponent = int(np.floor(np.log10(threshold)))
    mantissa = threshold / (10.0**exponent)
    if np.isclose(mantissa, 1.0):
        return rf"10^{{{exponent}}}"
    return rf"{mantissa:g}\times 10^{{{exponent}}}"


QFIM_TRACE_THRESHOLD_TEX = _qfim_threshold_tex(
    QFIM_TRACE_EIGENVALUE_THRESHOLD
)
QFIM_TRACE_THRESHOLD_FILE_TAG = (
    f"ge_{float(QFIM_TRACE_EIGENVALUE_THRESHOLD):.12g}".replace("+", "")
)
QFIM_TRACE_YLABEL = (
    rf"QFIM trace "
    rf"($\sum_{{\lambda_i \geq {QFIM_TRACE_THRESHOLD_TEX}}}\lambda_i$)"
)
QFIM_TRACE_MEAN_YLABEL = (
    rf"Mean QFIM trace "
    rf"($\sum_{{\lambda_i \geq {QFIM_TRACE_THRESHOLD_TEX}}}\lambda_i$)"
)


def _eigenvalue_index_ticks(n_params: int, *, max_ticks: int = 11) -> np.ndarray:
    n_params = int(n_params)
    if n_params <= 0:
        return np.asarray([], dtype=NP_INT_DTYPE)
    if n_params <= max(2, int(max_ticks)):
        return np.arange(1, n_params + 1, dtype=NP_INT_DTYPE)
    ticks = np.unique(
        np.rint(np.linspace(1, n_params, num=max(2, int(max_ticks)))).astype(
            NP_INT_DTYPE
        )
    )
    if ticks[0] != 1:
        ticks = np.insert(ticks, 0, 1)
    if ticks[-1] != n_params:
        ticks = np.append(ticks, n_params)
    return ticks


def _save_qfim_eigs_by_index(
    eigs_sorted_desc: np.ndarray,
    *,
    title: str,
    outpath: str,
) -> None:
    """Plot all random-point eigenvalues against their descending index."""
    eigs = np.asarray(eigs_sorted_desc, dtype=NP_REAL_DTYPE)
    if eigs.ndim != 2 or eigs.shape[0] == 0 or eigs.shape[1] == 0:
        raise ValueError("Random-point QFIM spectra must be a non-empty 2D array.")
    eigs_plot = np.where(
        np.isfinite(eigs) & (eigs > 0.0),
        eigs,
        NP_REAL_DTYPE(cfg.QFIM_EIG_PLOT_EPS),
    )

    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    for index in range(eigs_plot.shape[1]):
        values = eigs_plot[:, index]
        ax.scatter(
            np.full(values.shape, index + 1, dtype=NP_REAL_DTYPE),
            values,
            s=14.0,
            color=METRIC_COLORS["qfim"],
            alpha=0.55,
            edgecolors="black",
            linewidths=0.20,
            rasterized=True,
        )
    ticks = _eigenvalue_index_ticks(eigs_plot.shape[1])
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(tick)) for tick in ticks])
    ax.set_xlim(0.5, eigs_plot.shape[1] + 0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("QFIM eigenvalue")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(
        float(QFIM_TRACE_EIGENVALUE_THRESHOLD),
        color="C3",
        linestyle="--",
        linewidth=1.0,
        label=rf"rank threshold $\lambda_i={QFIM_TRACE_THRESHOLD_TEX}$",
    )
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=False)


def _save_qfim_eigs_by_index_colored_by_layer(
    eigs_by_layer: dict,
    layers,
    *,
    title: str,
    outpath: str,
) -> None:
    """Overlay raw random-point QFIM spectra, coloring samples by layer."""
    valid_layers = [int(L) for L in layers if eigs_by_layer.get(int(L)) is not None]
    if not valid_layers:
        return
    max_n_params = max(eigs_by_layer[L].shape[1] for L in valid_layers)
    cmap = matplotlib.colormaps.get_cmap("viridis")
    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    handles = []
    for layer_index, L in enumerate(valid_layers):
        eigs = np.asarray(eigs_by_layer[L], dtype=NP_REAL_DTYPE)
        if eigs.ndim != 2:
            raise ValueError(f"Random-point QFIM spectra for L={L} must be 2D data.")
        color = cmap(layer_index / max(len(valid_layers) - 1, 1))
        handles.append(Patch(facecolor=color, edgecolor=color, alpha=0.35, label=f"L={L}"))
        eigs_plot = np.where(
            np.isfinite(eigs) & (eigs > 0.0),
            eigs,
            NP_REAL_DTYPE(cfg.QFIM_EIG_PLOT_EPS),
        )
        for index in range(eigs_plot.shape[1]):
            values = eigs_plot[:, index]
            ax.scatter(
                np.full(values.shape, index + 1, dtype=NP_REAL_DTYPE),
                values,
                s=10.0,
                color=color,
                alpha=0.50,
                edgecolors="black",
                linewidths=0.15,
                rasterized=True,
            )
    ticks = _eigenvalue_index_ticks(max_n_params)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(tick)) for tick in ticks])
    ax.set_xlim(0.5, max_n_params + 0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("QFIM eigenvalue")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    handles.append(
        ax.axhline(
            float(QFIM_TRACE_EIGENVALUE_THRESHOLD),
            color="C3",
            linestyle="--",
            linewidth=1.0,
            label=rf"rank threshold $\lambda_i={QFIM_TRACE_THRESHOLD_TEX}$",
        )
    )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=True)


def _finite_mean_sem(values, *, axis=None):
    """Return finite-sample mean, SEM, and count along an optional axis."""
    array = np.asarray(values, dtype=NP_REAL_DTYPE)
    valid = np.isfinite(array)
    counts = np.sum(valid, axis=axis)
    sums = np.sum(np.where(valid, array, 0.0), axis=axis, dtype=NP_REAL_DTYPE)
    means = np.divide(
        sums,
        counts,
        out=np.full(np.shape(sums), np.nan, dtype=NP_REAL_DTYPE),
        where=counts > 0,
    )
    if axis is None:
        finite = array[valid]
        sem = (
            NP_REAL_DTYPE(0.0)
            if finite.size <= 1
            else NP_REAL_DTYPE(np.std(finite, ddof=1) / np.sqrt(finite.size))
        )
        return NP_REAL_DTYPE(means), sem, int(finite.size)

    centered = np.where(valid, array - np.expand_dims(means, axis=axis), np.nan)
    squared = np.nansum(centered**2, axis=axis)
    variance = np.divide(
        squared,
        counts - 1,
        out=np.zeros_like(squared, dtype=NP_REAL_DTYPE),
        where=counts > 1,
    )
    sems = np.divide(
        np.sqrt(variance),
        np.sqrt(counts),
        out=np.zeros_like(variance, dtype=NP_REAL_DTYPE),
        where=counts > 1,
    )
    return means, sems, counts


def _validated_scalar_text(value, *, name: str) -> str:
    """Return one scalar metadata string."""
    raw = np.asarray(value)
    if raw.size != 1:
        raise ValueError(f"{name} must be one scalar string.")
    return str(raw.reshape(-1)[0])


def _validated_nonnegative_integer_scalar(value, *, name: str) -> int:
    """Return one nonnegative integer scalar without lossy coercion."""
    raw = np.asarray(value)
    if (
        raw.size != 1
        or not np.issubdtype(raw.dtype, np.number)
        or np.iscomplexobj(raw)
    ):
        raise TypeError(f"{name} must be one real numeric scalar.")
    scalar = float(raw.reshape(-1)[0])
    if not np.isfinite(scalar) or scalar < 0.0 or scalar != np.rint(scalar):
        raise ValueError(f"{name} must be one finite nonnegative integer.")
    return int(scalar)


def _load_random_hessian_result(layers) -> None:
    """Load and validate the minimal random-point Hessian archive."""
    path = os.path.join(
        upqc.hessian_results_dir,
        "hessian_random_points.npz",
    )
    result = _load_required_result(path, require_h_param=True)
    description = "random-point Hessian archive"
    required_metadata = (
        "schema_version",
        "analysis_kind",
        "ansatz",
        "h_param",
        "layers",
        "num_hessian_samples",
        "hessian_sample_seed_base",
        "hessian_rank_threshold",
        "hessian_rank_definition",
        "hessian_condition_number_definition",
        "num_params_per_layer",
        "analysis_batch_size",
    )
    missing = [key for key in required_metadata if key not in result]
    if missing:
        raise KeyError(
            f"{description} is missing: " + ", ".join(missing)
        )

    schema_version = _validated_positive_integer_scalar(
        result["schema_version"],
        name=f"{description} schema_version",
    )
    if schema_version != HESSIAN_RANDOM_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Hessian schema version {schema_version}; expected "
            f"{HESSIAN_RANDOM_SCHEMA_VERSION}."
        )
    if _validated_scalar_text(
        result["analysis_kind"], name=f"{description} analysis_kind"
    ) != "random_points":
        raise ValueError(f"{description} analysis_kind must be 'random_points'.")
    if _validated_scalar_text(
        result["ansatz"], name=f"{description} ansatz"
    ) != "unitary_pqc":
        raise ValueError(f"{description} ansatz must be 'unitary_pqc'.")

    archive_layers = _validated_qfim_layers(
        result,
        description=description,
    )
    _require_matching_integer_sequence(
        archive_layers,
        layers,
        actual_name=f"{description} layers",
        expected_name="random-point QFIM layers",
    )
    num_samples = _validated_positive_integer_scalar(
        result["num_hessian_samples"],
        name=f"{description} num_hessian_samples",
    )
    if num_samples != int(upqc.NUM_QFIM_SAMPLES):
        raise ValueError(
            f"{description} num_hessian_samples {num_samples} does not match "
            f"the QFIM random-point count {int(upqc.NUM_QFIM_SAMPLES)}."
        )
    seed_base = _validated_nonnegative_integer_scalar(
        result["hessian_sample_seed_base"],
        name=f"{description} hessian_sample_seed_base",
    )
    if seed_base != int(upqc.QFIM_SAMPLE_SEED_BASE):
        raise ValueError(
            f"{description} seed {seed_base} does not match the QFIM seed "
            f"{int(upqc.QFIM_SAMPLE_SEED_BASE)}."
        )

    threshold_raw = np.asarray(result["hessian_rank_threshold"])
    if (
        threshold_raw.size != 1
        or not np.issubdtype(threshold_raw.dtype, np.number)
        or np.iscomplexobj(threshold_raw)
    ):
        raise TypeError(f"{description} rank threshold must be one real scalar.")
    threshold = float(threshold_raw.reshape(-1)[0])
    if (
        not np.isfinite(threshold)
        or NP_REAL_DTYPE(threshold) != QFIM_TRACE_EIGENVALUE_THRESHOLD
    ):
        raise ValueError(
            f"{description} rank threshold {threshold!r} does not match "
            f"{float(QFIM_TRACE_EIGENVALUE_THRESHOLD)!r}."
        )
    if _validated_scalar_text(
        result["hessian_rank_definition"],
        name=f"{description} hessian_rank_definition",
    ) != HESSIAN_RANK_DEFINITION:
        raise ValueError(f"{description} has an incompatible rank definition.")
    if _validated_scalar_text(
        result["hessian_condition_number_definition"],
        name=f"{description} hessian_condition_number_definition",
    ) != HESSIAN_CONDITION_NUMBER_DEFINITION:
        raise ValueError(
            f"{description} has an incompatible condition-number definition."
        )

    params_per_layer = _validated_positive_integer_scalar(
        result["num_params_per_layer"],
        name=f"{description} num_params_per_layer",
    )
    if params_per_layer != int(upqc.num_params_per_layer):
        raise ValueError(
            f"{description} num_params_per_layer {params_per_layer} does not "
            f"match {int(upqc.num_params_per_layer)}."
        )
    _validated_positive_integer_scalar(
        result["analysis_batch_size"],
        name=f"{description} analysis_batch_size",
    )

    expected_data_keys = {
        key
        for L in archive_layers
        for key in (f"L{L}_rank", f"L{L}_condition_number")
    }
    actual_layer_keys = {key for key in result if key.startswith("L")}
    if actual_layer_keys != expected_data_keys:
        unexpected = sorted(actual_layer_keys - expected_data_keys)
        missing_data = sorted(expected_data_keys - actual_layer_keys)
        details = []
        if missing_data:
            details.append("missing " + ", ".join(missing_data))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise KeyError(f"{description} layer data mismatch: " + "; ".join(details))

    rank_by_layer = {}
    condition_by_layer = {}
    for L in archive_layers:
        rank_key = f"L{L}_rank"
        condition_key = f"L{L}_condition_number"
        raw_ranks = np.asarray(result[rank_key])
        if (
            raw_ranks.shape != (num_samples,)
            or not np.issubdtype(raw_ranks.dtype, np.number)
            or np.iscomplexobj(raw_ranks)
            or not np.all(np.isfinite(raw_ranks))
            or not np.all(raw_ranks == np.rint(raw_ranks))
        ):
            raise ValueError(f"Invalid Hessian rank array {rank_key}.")
        ranks = raw_ranks.astype(NP_INT_DTYPE)
        if np.any(ranks < 0) or np.any(ranks > params_per_layer * int(L)):
            raise ValueError(f"Out-of-range Hessian ranks in {rank_key}.")

        raw_conditions = np.asarray(result[condition_key])
        if (
            raw_conditions.shape != (num_samples,)
            or not np.issubdtype(raw_conditions.dtype, np.number)
            or np.iscomplexobj(raw_conditions)
        ):
            raise ValueError(
                f"Invalid Hessian condition-number array {condition_key}."
            )
        conditions = np.asarray(raw_conditions, dtype=NP_REAL_DTYPE)
        if np.any(np.isinf(conditions)):
            raise ValueError(f"Infinite Hessian condition number in {condition_key}.")
        finite_conditions = np.isfinite(conditions)
        if np.any(conditions[finite_conditions] < 1.0 - 1e-12):
            raise ValueError(f"Hessian condition number below one in {condition_key}.")
        if not np.array_equal(finite_conditions, ranks > 0):
            raise ValueError(
                f"Hessian rank/condition definedness mismatch at L={L}."
            )
        rank_by_layer[L] = ranks
        condition_by_layer[L] = conditions

    upqc.hessian_rank_by_layer = rank_by_layer
    upqc.hessian_condition_by_layer = condition_by_layer
    upqc.HESSIAN_RANK_THRESHOLD = NP_REAL_DTYPE(threshold)


def _plot_random_hessian_summary(
    values_by_layer: dict,
    layers,
    *,
    ylabel: str,
    title: str,
    outpath: str,
    integer_y_axis: bool,
) -> None:
    """Plot random-point maximum, mean +/- SEM, and minimum by layer."""
    rows = []
    for L in layers:
        samples = np.asarray(
            values_by_layer[int(L)],
            dtype=NP_REAL_DTYPE,
        ).reshape(-1)
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            rows.append((int(L), np.nan, np.nan, np.nan, np.nan))
            continue
        mean, sem, _ = _finite_mean_sem(samples)
        rows.append(
            (
                int(L),
                float(np.max(samples)),
                float(mean),
                float(sem),
                float(np.min(samples)),
            )
        )
    finite = np.asarray(
        [np.all(np.isfinite(row[1:])) for row in rows],
        dtype=bool,
    )
    if not np.any(finite):
        raise ValueError(f"No finite Hessian statistics are available for {title}.")
    x_all = np.asarray([row[0] for row in rows], dtype=NP_REAL_DTYPE)
    x = x_all[finite]
    maxima = np.asarray([row[1] for row in rows], dtype=NP_REAL_DTYPE)[finite]
    means = np.asarray([row[2] for row in rows], dtype=NP_REAL_DTYPE)[finite]
    sems = np.asarray([row[3] for row in rows], dtype=NP_REAL_DTYPE)[finite]
    minima = np.asarray([row[4] for row in rows], dtype=NP_REAL_DTYPE)[finite]

    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    ax.plot(
        x,
        maxima,
        marker="^",
        linestyle="--",
        linewidth=1.2,
        color="C3",
        label="Maximum",
    )
    ax.errorbar(
        x,
        means,
        yerr=sems,
        marker="o",
        linestyle="-",
        linewidth=1.5,
        capsize=3.0,
        elinewidth=0.9,
        color=METRIC_COLORS["hessian"],
        label=r"Mean $\pm$ SEM",
        zorder=3,
    )
    ax.plot(
        x,
        minima,
        marker="v",
        linestyle="--",
        linewidth=1.2,
        color="C2",
        label="Minimum",
    )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_all)
    ax.set_xticklabels([str(row[0]) for row in rows])
    if integer_y_axis:
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=False)


def _plot_qfim_trace_max_mean_sem_by_layer(
    trace_by_layer: dict,
    layers,
    *,
    keep_label: str,
    num_samples: int,
    outpath: str,
) -> None:
    """Plot random-point trace maximum and mean with SEM against layers."""
    rows = []
    for L in layers:
        values = np.asarray(trace_by_layer.get(int(L)), dtype=NP_REAL_DTYPE).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        mean, sem, _ = _finite_mean_sem(values)
        rows.append((int(L), float(np.max(values)), float(mean), float(sem)))
    if not rows:
        return
    x = np.asarray([row[0] for row in rows], dtype=NP_REAL_DTYPE)

    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    ax.plot(
        x,
        [row[1] for row in rows],
        marker="o",
        linewidth=1.4,
        color="C0",
        label="Maximum QFIM trace",
    )
    ax.errorbar(
        x,
        [row[2] for row in rows],
        yerr=[row[3] for row in rows],
        marker="s",
        linewidth=1.4,
        capsize=4.0,
        elinewidth=1.0,
        color="C1",
        label=r"Mean QFIM trace $\pm$ SEM",
    )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(QFIM_TRACE_YLABEL)
    ax.set_title(
        rf"QFIM trace maximum and mean $\pm$ SEM at {int(num_samples)} "
        rf"random points ({keep_label}, $\lambda_i \geq {QFIM_TRACE_THRESHOLD_TEX}$)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(row[0]) for row in rows])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=False)


def _plot_qfim_trace_history_mean_sem(
    trace_by_layer: dict,
    layers,
    sample_iters,
    *,
    keep_label: str,
    outpath: str,
) -> None:
    """Plot layer-colored run mean and SEM of Trace against iterations."""
    x = np.asarray(sample_iters, dtype=NP_REAL_DTYPE)
    valid_layers = [int(L) for L in layers if trace_by_layer.get(int(L)) is not None]
    if not valid_layers:
        return
    cmap = matplotlib.colormaps.get_cmap("viridis")
    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    plotted = False
    for layer_index, L in enumerate(valid_layers):
        traces = np.asarray(trace_by_layer[L], dtype=NP_REAL_DTYPE)
        if traces.ndim != 2 or traces.shape[1] != x.size:
            raise ValueError(
                f"QFIM trace history for L={L} must have shape "
                f"(num_runs, {x.size}), got {traces.shape}."
            )
        means, sems, counts = _finite_mean_sem(traces, axis=0)
        finite = np.isfinite(means) & (counts > 0)
        if not np.any(finite):
            continue
        plotted = True
        color = cmap(layer_index / max(len(valid_layers) - 1, 1))
        ax.errorbar(
            x[finite],
            means[finite],
            yerr=sems[finite],
            marker="o",
            linestyle="-",
            linewidth=1.2,
            markersize=4.5,
            capsize=3.0,
            elinewidth=0.8,
            color=color,
            label=f"L={L}",
        )
    if not plotted:
        plt.close(plt.gcf())
        return
    ax.set_xlabel("Iterations")
    ax.set_ylabel(QFIM_TRACE_MEAN_YLABEL)
    ax.set_title(
        rf"Mean QFIM trace along optimization path ({keep_label}, "
        rf"$\lambda_i \geq {QFIM_TRACE_THRESHOLD_TEX}$)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(value)) for value in x], rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=True)


def _plot_qfim_trace_flat_mean_sem_by_layer(
    trace_by_layer: dict,
    layers,
    *,
    keep_label: str,
    outpath: str,
) -> None:
    """Flatten run/time samples and plot their mean with SEM by layer."""
    rows = []
    for L in layers:
        values = np.asarray(trace_by_layer.get(int(L)), dtype=NP_REAL_DTYPE).reshape(-1)
        mean, sem, count = _finite_mean_sem(values)
        if count > 0:
            rows.append((int(L), float(mean), float(sem)))
    if not rows:
        return
    x = np.asarray([row[0] for row in rows], dtype=NP_REAL_DTYPE)
    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    ax.errorbar(
        x,
        [row[1] for row in rows],
        yerr=[row[2] for row in rows],
        marker="o",
        linestyle="-",
        linewidth=1.2,
        markersize=6.0,
        capsize=4.0,
        elinewidth=1.0,
        color=METRIC_COLORS["qfim"],
        label=r"Mean QFIM trace $\pm$ SEM",
    )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(QFIM_TRACE_MEAN_YLABEL)
    ax.set_title(
        rf"QFIM trace mean $\pm$ SEM vs Layers along optimization path "
        rf"({keep_label}, $\lambda_i \geq {QFIM_TRACE_THRESHOLD_TEX}$)"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(row[0]) for row in rows])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=False)


def _plot_spectral_count_by_layer(
    eigs_by_layer: dict,
    *,
    title: str,
    ylabel: str,
    outpath: str,
    use_absolute_values: bool = False,
) -> None:
    """Plot mean component counts above each threshold against layer count."""
    layers = [int(L) for L in sorted(eigs_by_layer) if eigs_by_layer[L] is not None]
    thresholds = tuple(float(t) for t in cfg.QFIM_PATH_EIGCOUNT_THRESHOLDS)
    if not layers or not thresholds:
        return

    upqc.new_prx_figure(width="double")
    fig, ax = plt.gcf(), plt.gca()
    cmap = matplotlib.colormaps.get_cmap("viridis")
    for idx, threshold in enumerate(thresholds):
        means, sems = [], []
        for L in layers:
            eigs = np.asarray(eigs_by_layer[L], dtype=NP_REAL_DTYPE)
            if eigs.ndim < 2:
                raise ValueError("Spectral arrays need sample and eigenvalue axes.")
            eigs = eigs.reshape(-1, eigs.shape[-1])
            if use_absolute_values:
                eigs = np.abs(eigs)
            counts = np.sum(eigs >= threshold, axis=1).astype(NP_REAL_DTYPE)
            means.append(np.mean(counts))
            sems.append(0.0 if counts.size < 2 else np.std(counts, ddof=1) / np.sqrt(counts.size))
        ax.errorbar(
            layers, means, yerr=sems, marker="o",
            linestyle=STATISTIC_LINESTYLES["mean"], linewidth=1.2,
            capsize=3.0, color=cmap(idx / max(len(thresholds) - 1, 1)),
            label=rf"threshold $\geq {threshold:g}$",
        )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=True)


def _load_unitary_vqe_results(result: Optional[dict] = None) -> dict:
    path = os.path.join(
        upqc.energy_results_dir,
        "vqe_optimization_results.npz",
    )
    if result is None:
        result = _load_required_result(path, require_h_param=True)
    else:
        _validate_result_h_param(
            result,
            path,
            expected_h_param=upqc.h_param,
            required=True,
        )

    upqc.layer_list = [
        int(L)
        for L in np.asarray(result["layers"], dtype=NP_INT_DTYPE)
    ]
    upqc.sample_iters = np.asarray(result["sample_iters"], dtype=NP_INT_DTYPE)
    upqc.sample_iter_set = set(int(t) for t in upqc.sample_iters.tolist())
    upqc.steps = int(np.asarray(result["steps"]).item())
    upqc.num_runs = int(np.asarray(result["num_runs"]).item())
    upqc.tolerance = float(np.asarray(result["tolerance"]).item())
    upqc.lr = float(np.asarray(result["learning_rate"]).item())
    upqc.smallest_eigval = float(np.asarray(result["smallest_eigval"]).item())
    upqc.cmap = matplotlib.colormaps.get_cmap("viridis")

    upqc.final_stats = {
        "layer": np.asarray(result["final_stats_layer"], dtype=NP_INT_DTYPE).tolist(),
        "success_rate": np.asarray(
            result["final_stats_success_rate"],
            dtype=NP_REAL_DTYPE,
        ).tolist(),
        "mean_energy": np.asarray(
            result["final_stats_mean_energy"],
            dtype=NP_REAL_DTYPE,
        ).tolist(),
        "std_energy": np.asarray(
            result["final_stats_std_energy"],
            dtype=NP_REAL_DTYPE,
        ).tolist(),
    }

    upqc.theta_history = {}
    upqc.best_theta_by_layer = {}
    upqc.final_theta_wrapped_rmsdist_by_layer = {}
    upqc.energy_traces_by_layer = {}
    upqc.grad_norm_traces_by_layer = {}
    upqc.theta_sample_traces_by_layer = {}
    upqc.energy_mean_history = {}
    upqc.energy_std_history = {}
    upqc.success_rates_history = {}

    for L in upqc.layer_list:
        upqc.theta_history[L] = np.asarray(
            result[f"L{L}_theta_history"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.best_theta_by_layer[L] = np.asarray(
            result[f"L{L}_best_theta"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.final_theta_wrapped_rmsdist_by_layer[L] = np.asarray(
            result[f"L{L}_final_theta_wrapped_rmsdist"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.energy_traces_by_layer[L] = np.asarray(
            result[f"L{L}_energy_traces"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.grad_norm_traces_by_layer[L] = np.asarray(
            result[f"L{L}_grad_norm_traces"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.theta_sample_traces_by_layer[L] = np.asarray(
            result[f"L{L}_theta_samples"],
            dtype=NP_REAL_DTYPE,
        )
        energy_data = upqc.energy_traces_by_layer[L]
        upqc.energy_mean_history[L] = np.mean(energy_data, axis=0)
        upqc.energy_std_history[L] = np.std(energy_data, axis=0)
        diffs = np.abs(energy_data - upqc.smallest_eigval)
        upqc.success_rates_history[L] = np.mean(diffs <= upqc.tolerance, axis=0)

    return result


def _load_random_qfim_results() -> None:
    qfim_path = os.path.join(upqc.qfim_results_dir, "qfim_random_points.npz")
    qfim_result = _load_required_result(qfim_path)
    legacy_description = "legacy combined random-point QFIM archive"
    layers = _validated_qfim_layers(
        qfim_result,
        description=legacy_description,
    )

    upqc.qfim_layer_list = layers
    if "num_qfim_samples" not in qfim_result:
        raise KeyError(f"{legacy_description} is missing 'num_qfim_samples'.")
    upqc.NUM_QFIM_SAMPLES = _validated_positive_integer_scalar(
        qfim_result["num_qfim_samples"],
        name=f"{legacy_description} num_qfim_samples",
    )
    upqc.QFIM_SAMPLE_SEED_BASE = int(
        np.asarray(qfim_result["qfim_sample_seed_base"]).item()
    )
    upqc.RED_JVP_CHUNK = int(np.asarray(qfim_result["red_jvp_chunk"]).item())
    upqc.PURE_QFIM_LAYER_THRESHOLD = int(
        np.asarray(qfim_result["pure_qfim_layer_threshold"]).item()
    )

    upqc.qfim_random_thetas_by_layer = {}

    for L in layers:
        theta_key = f"L{L}_theta"
        if theta_key not in qfim_result:
            raise KeyError(f"{legacy_description} is missing {theta_key!r}.")
        raw_theta = np.asarray(qfim_result[theta_key])
        if (
            not np.issubdtype(raw_theta.dtype, np.number)
            or np.iscomplexobj(raw_theta)
        ):
            raise TypeError(f"{legacy_description} {theta_key} must be real numeric data.")
        theta = np.asarray(raw_theta, dtype=NP_REAL_DTYPE)
        expected_theta_shape = (
            upqc.NUM_QFIM_SAMPLES,
            int(upqc.num_params_per_layer) * int(L),
        )
        if theta.shape != expected_theta_shape or not np.all(np.isfinite(theta)):
            raise ValueError(
                f"{legacy_description} {theta_key} must be finite with shape "
                f"{expected_theta_shape}, got {theta.shape}."
            )
        upqc.qfim_random_thetas_by_layer[L] = theta

    canonical_eigs_by_keep = {}
    for keep_key in ("keep0123", "keep01234"):
        canonical_path = os.path.join(
            upqc.qfim_results_dir,
            f"qfim_random_points_{keep_key}.npz",
        )
        canonical_result = _load_required_result(
            canonical_path,
            require_h_param=True,
        )
        description = f"canonical random-point QFIM archive {keep_key}"
        canonical_layers = _validated_qfim_layers(
            canonical_result,
            description=description,
        )
        _require_matching_integer_sequence(
            canonical_layers,
            layers,
            actual_name=f"{description} layers",
            expected_name=f"{legacy_description} layers",
        )
        if "num_qfim_samples" not in canonical_result:
            raise KeyError(f"{description} is missing 'num_qfim_samples'.")
        canonical_num_samples = _validated_positive_integer_scalar(
            canonical_result["num_qfim_samples"],
            name=f"{description} num_qfim_samples",
        )
        if canonical_num_samples != upqc.NUM_QFIM_SAMPLES:
            raise ValueError(
                f"{description} num_qfim_samples {canonical_num_samples} does "
                f"not match {upqc.NUM_QFIM_SAMPLES}."
            )
        _validate_raw_qfim_archive_metadata(
            canonical_result,
            description=description,
            expected_keep_key=keep_key,
            expected_analysis_kind="random_points",
        )
        canonical_eigs_by_keep[keep_key] = _validated_random_qfim_eigs(
            canonical_result,
            canonical_layers,
            num_samples=canonical_num_samples,
            description=description,
        )

    upqc.qfim_eigs_reduced_by_layer = canonical_eigs_by_keep["keep0123"]
    upqc.qfim_eigs_pure_by_layer = canonical_eigs_by_keep["keep01234"]
    upqc.qfim_trace_reduced_by_layer = {
        L: qfim_trace_at_or_above_rank_threshold(eigs)
        for L, eigs in upqc.qfim_eigs_reduced_by_layer.items()
    }
    upqc.qfim_trace_pure_by_layer = {
        L: qfim_trace_at_or_above_rank_threshold(eigs)
        for L, eigs in upqc.qfim_eigs_pure_by_layer.items()
    }

    hs_path = os.path.join(
        upqc.hs_results_dir,
        "hs_random_points_reduced_0123.npz",
    )
    hs_result = _load_required_result(hs_path)

    upqc.hs_eigs_reduced_by_layer = {}

    for L in layers:
        upqc.hs_eigs_reduced_by_layer[L] = np.asarray(
            hs_result[f"L{L}_eigs_desc"],
            dtype=NP_REAL_DTYPE,
        )

    hs_pure_result = _load_required_result(
        os.path.join(upqc.hs_results_dir, "hs_random_points_pure_full.npz")
    )
    upqc.hs_eigs_pure_by_layer = {
        L: np.asarray(hs_pure_result[f"L{L}_eigs_desc"], dtype=NP_REAL_DTYPE)
        for L in layers
    }

    _load_random_hessian_result(layers)


def _plot_random_qfim_results() -> None:
    hs_eigs_reduced_0123_dir = upqc.hs_eigs_reduced_0123_dir
    qfim_trace_dir = os.path.join(upqc.qfim_fig_dir, "trace")

    os.makedirs(hs_eigs_reduced_0123_dir, exist_ok=True)
    os.makedirs(qfim_trace_dir, exist_ok=True)

    qfim_random_specs = (
        (
            "keep0123",
            "Reduced keep=(0,1,2,3)",
            upqc.qfim_eigs_reduced_by_layer,
            upqc.qfim_eigs_reduced_0123_dir,
            "reduced_0123",
        ),
        (
            "keep01234",
            "Pure full-state keep=(0,1,2,3,4)",
            upqc.qfim_eigs_pure_by_layer,
            upqc.qfim_eigs_pure_dir,
            "pure_full",
        ),
    )

    for L in upqc.qfim_layer_list:
        for keep_key, keep_label, eigs_by_layer, eigs_dir, file_label in qfim_random_specs:
            _save_qfim_eigs_by_index(
                eigs_by_layer[L],
                title=(
                    f"QFIM eigenvalues at {upqc.NUM_QFIM_SAMPLES} random "
                    f"points (L={L}, {keep_label})"
                ),
                outpath=os.path.join(
                    eigs_dir,
                    f"L{L}_{file_label}.pdf",
                ),
            )
        upqc.plot_style.save_eigenvalue_histograms_by_trial(
            upqc.hs_eigs_reduced_by_layer[L],
            outdir=os.path.join(
                hs_eigs_reduced_0123_dir,
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
            color=METRIC_COLORS["hs"],
        )
        upqc.plot_style.save_eigenvalue_histograms_by_trial(
            upqc.hs_eigs_pure_by_layer[L],
            outdir=os.path.join(
                upqc.hs_eigs_dir, "pure_full", "histograms", "random_points", f"L{L}"
            ),
            matrix_tag="unitary_pqc_hs_gram",
            matrix_label="HS tangent Gram",
            num_layers=L,
            context_tag="random",
            context_label="random point",
            condition_tag="pure_full",
            condition_label="pure full state",
            color=METRIC_COLORS["hs"],
        )

    for keep_key, keep_label, eigs_by_layer, _, _ in qfim_random_specs:
        _save_qfim_eigs_by_index_colored_by_layer(
            eigs_by_layer,
            upqc.qfim_layer_list,
            title=(
                f"QFIM eigenvalues at {upqc.NUM_QFIM_SAMPLES} random points "
                f"({keep_label})"
            ),
            outpath=os.path.join(
                upqc.qfim_eigs_dir,
                f"qfim_eigs_by_index_layers_{keep_key}.pdf",
            ),
        )

    trace_specs = (
        (
            "keep0123",
            "Reduced keep=(0,1,2,3)",
            upqc.qfim_trace_reduced_by_layer,
        ),
        (
            "keep01234",
            "Pure full-state keep=(0,1,2,3,4)",
            upqc.qfim_trace_pure_by_layer,
        ),
    )
    for keep_key, keep_label, trace_by_layer in trace_specs:
        _plot_qfim_trace_max_mean_sem_by_layer(
            trace_by_layer,
            upqc.qfim_layer_list,
            keep_label=keep_label,
            num_samples=upqc.NUM_QFIM_SAMPLES,
            outpath=os.path.join(
                qfim_trace_dir,
                "qfim_trace_max_mean_sem_random_points_"
                f"{QFIM_TRACE_THRESHOLD_FILE_TAG}_{keep_key}.pdf",
            ),
        )

    threshold_tex = _qfim_threshold_tex(float(upqc.HESSIAN_RANK_THRESHOLD))
    _plot_random_hessian_summary(
        upqc.hessian_rank_by_layer,
        upqc.qfim_layer_list,
        ylabel=rf"Hessian rank ($|\lambda_i| \geq {threshold_tex}$)",
        title=(
            f"Hessian rank at {upqc.NUM_QFIM_SAMPLES} random parameter points"
        ),
        outpath=os.path.join(
            upqc.hessian_fig_dir,
            "hessian_rank_random_points.pdf",
        ),
        integer_y_axis=True,
    )
    _plot_random_hessian_summary(
        upqc.hessian_condition_by_layer,
        upqc.qfim_layer_list,
        ylabel=(
            rf"Thresholded Hessian condition number "
            rf"($|\lambda_i| \geq {threshold_tex}$)"
        ),
        title=(
            "Hessian condition number at "
            f"{upqc.NUM_QFIM_SAMPLES} random parameter points"
        ),
        outpath=os.path.join(
            upqc.hessian_fig_dir,
            "hessian_condition_number_random_points.pdf",
        ),
        integer_y_axis=False,
    )

    spectral_random_summaries = (
        (upqc.qfim_eigs_reduced_by_layer, upqc.qfim_eigs_reduced_0123_dir,
         "Reduced QFIM", "qfim_reduced", False),
        (upqc.qfim_eigs_pure_by_layer, upqc.qfim_eigs_pure_dir,
         "Pure-state QFIM", "qfim_pure", False),
        (upqc.hs_eigs_reduced_by_layer, upqc.hs_eigs_reduced_0123_dir,
         "HS tangent Gram", "hs", False),
        (upqc.hs_eigs_pure_by_layer, os.path.join(upqc.hs_eigs_dir, "pure_full"),
         "Pure-full HS tangent Gram", "hs_pure", False),
    )
    for eigs, directory, label, tag, use_abs in spectral_random_summaries:
        if not eigs or not any(value is not None for value in eigs.values()):
            continue
        _plot_spectral_count_by_layer(
            eigs,
            title=f"{label} eigenvalue count at random points by threshold",
            ylabel=f"Mean {label} eigenvalue count",
            outpath=os.path.join(directory, f"{tag}_eigcount_threshold_overlay_random_points.pdf"),
            use_absolute_values=use_abs,
        )


def _load_optimization_path_results() -> None:
    vqe_layers = list(upqc.layer_list)
    vqe_sample_iters = np.asarray(upqc.sample_iters, dtype=NP_INT_DTYPE)
    canonical_eigs_history_by_keep = {}
    canonical_layers_reference = None
    canonical_sample_iters_reference = None

    for keep_key in ("keep0123", "keep01234"):
        eigs_path = os.path.join(
            upqc.qfim_results_dir,
            f"qfim_eigs_history_optimization_path_{keep_key}.npz",
        )
        eigs_result = _load_required_result(eigs_path, require_h_param=True)
        description = f"canonical optimization-path QFIM archive {keep_key}"
        archive_layers = _validated_qfim_layers(
            eigs_result,
            description=description,
        )
        archive_sample_iters = _validated_qfim_sample_iters(
            eigs_result,
            description=description,
        )
        _validate_raw_qfim_archive_metadata(
            eigs_result,
            description=description,
            expected_keep_key=keep_key,
            expected_analysis_kind="optimization_path",
        )
        if "num_runs" not in eigs_result:
            raise KeyError(f"{description} is missing 'num_runs'.")
        archive_num_runs = _validated_positive_integer_scalar(
            eigs_result["num_runs"],
            name=f"{description} num_runs",
        )
        if archive_num_runs != int(upqc.num_runs):
            raise ValueError(
                f"{description} num_runs {archive_num_runs} does not match "
                f"VQE num_runs {int(upqc.num_runs)}."
            )

        if canonical_layers_reference is None:
            canonical_layers_reference = archive_layers
            canonical_sample_iters_reference = archive_sample_iters
            _require_matching_integer_sequence(
                archive_layers,
                vqe_layers,
                actual_name=f"{description} layers",
                expected_name="VQE layers",
            )
            _require_matching_integer_sequence(
                archive_sample_iters,
                vqe_sample_iters,
                actual_name=f"{description} sample_iters",
                expected_name="VQE sample_iters",
            )
        else:
            _require_matching_integer_sequence(
                archive_layers,
                canonical_layers_reference,
                actual_name=f"{description} layers",
                expected_name="keep0123 QFIM layers",
            )
            _require_matching_integer_sequence(
                archive_sample_iters,
                canonical_sample_iters_reference,
                actual_name=f"{description} sample_iters",
                expected_name="keep0123 QFIM sample_iters",
            )

        canonical_eigs_history_by_keep[keep_key] = _validated_qfim_eigs_history(
            eigs_result,
            archive_layers,
            archive_sample_iters,
            num_runs=archive_num_runs,
            description=description,
        )

    layers = list(canonical_layers_reference)
    upqc.layer_list = layers
    upqc.sample_iters = np.asarray(
        canonical_sample_iters_reference,
        dtype=NP_INT_DTYPE,
    )
    upqc.qfim_eigs_history_by_layer = canonical_eigs_history_by_keep["keep0123"]
    upqc.qfim_eigs_history_pure_by_layer = canonical_eigs_history_by_keep[
        "keep01234"
    ]
    upqc.qfim_trace_history_by_layer = {
        L: qfim_trace_at_or_above_rank_threshold(eigs)
        for L, eigs in upqc.qfim_eigs_history_by_layer.items()
    }
    upqc.qfim_trace_history_pure_by_layer = {
        L: qfim_trace_at_or_above_rank_threshold(eigs)
        for L, eigs in upqc.qfim_eigs_history_pure_by_layer.items()
    }

    hs_rank_path = os.path.join(
        upqc.hs_results_dir,
        "hs_rank_history_optimization_path_reduced_0123.npz",
    )
    hs_result = _load_required_result(hs_rank_path)
    upqc.hs_eigs_history_by_layer = {
        L: np.asarray(hs_result[f"L{L}_eigs"], dtype=NP_REAL_DTYPE)
        for L in layers
    }
    hs_pure_result = _load_required_result(
        os.path.join(
            upqc.hs_results_dir,
            "hs_rank_history_optimization_path_pure_full.npz",
        )
    )
    upqc.hs_eigs_history_pure_by_layer = {
        L: np.asarray(hs_pure_result[f"L{L}_eigs"], dtype=NP_REAL_DTYPE)
        for L in layers
    }

def _plot_optimization_path_results() -> None:
    qfim_trace_dir = os.path.join(upqc.qfim_fig_dir, "trace")
    os.makedirs(qfim_trace_dir, exist_ok=True)
    spectral_path_summaries = (
        (upqc.qfim_eigs_history_by_layer, upqc.qfim_eigs_dir, "QFIM", "qfim", False),
        (upqc.qfim_eigs_history_pure_by_layer, os.path.join(upqc.qfim_eigs_dir, "pure_full"),
         "Pure-full QFIM", "qfim_pure", False),
        (upqc.hs_eigs_history_by_layer, upqc.hs_eigs_dir, "HS tangent Gram", "hs", False),
        (upqc.hs_eigs_history_pure_by_layer, os.path.join(upqc.hs_eigs_dir, "pure_full"),
         "Pure-full HS tangent Gram", "hs_pure", False),
    )
    for eigs, directory, label, tag, use_abs in spectral_path_summaries:
        if not eigs:
            continue
        _plot_spectral_count_by_layer(
            eigs,
            title=f"{label} eigenvalue count along optimization path by threshold",
            ylabel=f"Mean {label} eigenvalue count",
            outpath=os.path.join(
                directory,
                f"{tag}_eigcount_threshold_overlay_optimization_path_by_layer.pdf",
            ),
            use_absolute_values=use_abs,
        )

    qfim_trace_history_specs = (
        (
            "keep0123",
            "Reduced keep=(0,1,2,3)",
            upqc.qfim_trace_history_by_layer,
        ),
        (
            "keep01234",
            "Pure full-state keep=(0,1,2,3,4)",
            upqc.qfim_trace_history_pure_by_layer,
        ),
    )
    for keep_key, keep_label, trace_by_layer in qfim_trace_history_specs:
        _plot_qfim_trace_history_mean_sem(
            trace_by_layer,
            upqc.layer_list,
            upqc.sample_iters,
            keep_label=keep_label,
            outpath=os.path.join(
                qfim_trace_dir,
                "qfim_trace_mean_sem_optimization_path_by_iteration_"
                f"{QFIM_TRACE_THRESHOLD_FILE_TAG}_{keep_key}.pdf",
            ),
        )
        _plot_qfim_trace_flat_mean_sem_by_layer(
            trace_by_layer,
            upqc.layer_list,
            keep_label=keep_label,
            outpath=os.path.join(
                qfim_trace_dir,
                "qfim_trace_flattened_mean_sem_optimization_path_by_layer_"
                f"{QFIM_TRACE_THRESHOLD_FILE_TAG}_{keep_key}.pdf",
            ),
        )

def run_unitary_pqc_visualization(
    *,
    h_param: Optional[float] = None,
) -> dict:
    selected_h_param = _finite_float(
        str(_SELECTED_H_PARAM if h_param is None else h_param)
    )

    # Validate the selected h archive before configuration creates its output
    # directory tree. This prevents a mistyped h from producing empty folders.
    selected_save_dir = upqc._unitary_pqc_save_dir(selected_h_param)
    vqe_result_path = os.path.join(
        selected_save_dir,
        "numerical_results",
        "energy",
        "vqe_optimization_results.npz",
    )
    vqe_result = _load_required_result(
        vqe_result_path,
        expected_h_param=selected_h_param,
        require_h_param=True,
    )

    upqc.configure_unitary_pqc_overparam(h_value=selected_h_param)
    _load_unitary_vqe_results(vqe_result)
    upqc.plot_vqe_optimization_results()
    # Re-render the shared final-error filenames with the DPQC layout and add
    # its log-scale, threshold-detail, and multiple-tolerance figures.
    _plot_vqe_ground_truth_error_results()

    _load_random_qfim_results()
    _plot_random_qfim_results()

    _load_optimization_path_results()
    _plot_optimization_path_results()

    return upqc.collect_unitary_pqc_result()


if __name__ == "__main__":
    visualization_result = run_unitary_pqc_visualization(
        h_param=_CLI_ARGS.h_param,
    )
    print(
        "Visualized Hamiltonian parameter h: "
        f"{visualization_result['h_param']}"
    )
    print(f"Saved figures to: {visualization_result['save_dir']}")
