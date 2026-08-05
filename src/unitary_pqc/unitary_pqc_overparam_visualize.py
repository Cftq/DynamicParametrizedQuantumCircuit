#!/usr/bin/env python
# coding: utf-8
"""Visualize saved Unitary-PQC numerical results.

Run unitary_pqc_overparam_compute.py first. This script loads saved .npz
results under figs/unitary_pqc/h_<h_param>/numerical_results and generates
numerical figures without recomputing VQE or QFIM quantities. Circuit drawings
are handled independently by unitary_pqc_overparam_draw_circuits.py.

Example::

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

import unitary_pqc_overparam_compute as upqc
from dpqc_overparam_common import load_npz_result as _load_npz_result_unchecked


NP_REAL_DTYPE = np.float64
NP_INT_DTYPE = np.int64
ENERGY_ERROR_PLOT_EPS = NP_REAL_DTYPE(1e-12)
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
    "ortk": "#009E73",
    "hessian": "#CC79A7",
    "energy": "#E69F00",
}
STATISTIC_LINESTYLES = {"min": ":", "mean": "-", "max": "--"}


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
    _plot_final_energy_error_violin(
        final_absolute_errors_by_layer,
        layers,
        outpath=os.path.join(energy_dir, "final_energy_error_logscale.pdf"),
        log_scale=True,
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


def _qfim_rank_min_mean_max_sem_by_layer(
    rank_by_layer: dict,
    layers,
):
    """Return finite random-point QFIM rank statistics for each layer."""
    valid_items = []
    for L in layers:
        ranks_L = rank_by_layer.get(int(L))
        if ranks_L is None:
            continue

        ranks = np.asarray(ranks_L, dtype=NP_REAL_DTYPE).reshape(-1)
        ranks = ranks[np.isfinite(ranks)]
        if ranks.size > 0:
            valid_items.append((int(L), ranks))

    if not valid_items:
        empty = np.asarray([], dtype=NP_REAL_DTYPE)
        return empty, empty, empty, empty, empty, []

    valid_layers = [L for L, _ in valid_items]
    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)
    min_ranks = np.asarray(
        [np.min(ranks) for _, ranks in valid_items],
        dtype=NP_REAL_DTYPE,
    )
    mean_ranks = np.asarray(
        [np.mean(ranks) for _, ranks in valid_items],
        dtype=NP_REAL_DTYPE,
    )
    max_ranks = np.asarray(
        [np.max(ranks) for _, ranks in valid_items],
        dtype=NP_REAL_DTYPE,
    )
    sem_ranks = np.asarray(
        [
            0.0
            if ranks.size < 2
            else np.std(ranks, ddof=1) / np.sqrt(ranks.size)
            for _, ranks in valid_items
        ],
        dtype=NP_REAL_DTYPE,
    )
    return x, min_ranks, mean_ranks, max_ranks, sem_ranks, valid_layers


def _plot_qfim_rank_min_mean_max_by_layer(
    rank_by_layer: dict,
    layers,
    *,
    title: str,
    outpath: str,
    color: str = METRIC_COLORS["qfim"],
) -> None:
    """Plot random-point QFIM rank minimum, mean +/- SEM, and maximum."""
    (
        x,
        min_ranks,
        mean_ranks,
        max_ranks,
        sem_ranks,
        valid_layers,
    ) = _qfim_rank_min_mean_max_sem_by_layer(rank_by_layer, layers)
    if x.size == 0:
        return

    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    ax.plot(
        x,
        min_ranks,
        marker="v",
        linestyle=STATISTIC_LINESTYLES["min"],
        linewidth=1.3,
        markersize=5.0,
        color=color,
        label="Minimum effective rank",
    )
    ax.errorbar(
        x,
        mean_ranks,
        yerr=sem_ranks,
        marker="o",
        linestyle=STATISTIC_LINESTYLES["mean"],
        linewidth=1.3,
        markersize=5.0,
        capsize=3.0,
        elinewidth=0.9,
        color=color,
        label=r"Mean effective rank $\pm$ SEM",
    )
    ax.plot(
        x,
        max_ranks,
        marker="^",
        linestyle=STATISTIC_LINESTYLES["max"],
        linewidth=1.3,
        markersize=5.0,
        color=color,
        label="Maximum effective rank",
    )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(r"QFIM effective rank $(\lambda_k > 10^{-12})$")
    upqc.set_prx_title(title, ax=ax)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in valid_layers])
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=False)


def _plot_qfim_rank_history_max_by_layer(
    rank_history_by_layer: dict,
    layers,
    sample_iters,
    *,
    title: str,
    outpath: str,
    ylabel: str = r"Maximum QFIM effective rank $(\lambda_k > 10^{-12})$",
    cmap=None,
) -> None:
    """Plot the finite run-maximum QFIM rank at each sampled iteration."""
    valid_layers = [
        int(L)
        for L in layers
        if rank_history_by_layer.get(int(L)) is not None
    ]
    if not valid_layers:
        return

    x = upqc._qfim_history_plot_iterations(sample_iters)
    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap
    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    num_layers = len(valid_layers)

    for layer_idx, L in enumerate(valid_layers):
        ranks = np.asarray(rank_history_by_layer[L], dtype=NP_REAL_DTYPE)
        if ranks.ndim != 2:
            raise ValueError(
                "Each rank history array must be 2D: "
                "(num_runs, num_sample_iters)."
            )
        if ranks.shape[1] != x.size and ranks.shape[0] == x.size:
            ranks = ranks.T
        if ranks.shape[1] != x.size:
            raise ValueError(
                f"Shape mismatch for L={L}: ranks.shape={ranks.shape}, "
                f"len(sample_iters)={x.size}."
            )

        finite = np.isfinite(ranks)
        counts = np.sum(finite, axis=0)
        max_ranks = np.max(np.where(finite, ranks, -np.inf), axis=0)
        max_ranks = np.where(counts > 0, max_ranks, np.nan)
        finite_mask = np.isfinite(max_ranks) & (counts > 0)
        if not np.any(finite_mask):
            continue

        color = cmap(layer_idx / max(num_layers - 1, 1))
        ax.plot(
            x[finite_mask],
            max_ranks[finite_mask],
            marker="o",
            linestyle=STATISTIC_LINESTYLES["max"],
            linewidth=1.2,
            markersize=4.5,
            color=color,
            label=f"L={L}",
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel(ylabel)
    upqc.set_prx_title(title, ax=ax)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in x], rotation=45, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
    )
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(
        outpath,
        outside_legend=True,
        legend_space_frac=0.26,
    )


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
    upqc.grad_sample_traces_by_layer = {}
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
        upqc.grad_sample_traces_by_layer[L] = np.asarray(
            result[f"L{L}_grad_samples"],
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
    layers = [int(L) for L in np.asarray(qfim_result["layers"], dtype=NP_INT_DTYPE)]

    upqc.layer_list = layers
    upqc.NUM_QFIM_SAMPLES = int(
        np.asarray(qfim_result["num_qfim_samples"]).item()
    )
    upqc.QFIM_SAMPLE_SEED_BASE = int(
        np.asarray(qfim_result["qfim_sample_seed_base"]).item()
    )
    upqc.RED_JVP_CHUNK = int(np.asarray(qfim_result["red_jvp_chunk"]).item())
    upqc.PURE_QFIM_LAYER_THRESHOLD = int(
        np.asarray(qfim_result["pure_qfim_layer_threshold"]).item()
    )

    upqc.qfim_random_thetas_by_layer = {}
    upqc.qfim_rank_reduced_by_layer = {}
    upqc.qfim_eigs_reduced_by_layer = {}
    upqc.qfim_thresh_reduced_by_layer = {}
    upqc.qfim_rank_pure_by_layer = {}
    upqc.qfim_eigs_pure_by_layer = {}
    upqc.qfim_thresh_pure_by_layer = {}

    for L in layers:
        upqc.qfim_random_thetas_by_layer[L] = np.asarray(
            qfim_result[f"L{L}_theta"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.qfim_rank_reduced_by_layer[L] = np.asarray(
            qfim_result[f"L{L}_rank_reduced"],
            dtype=NP_INT_DTYPE,
        )
        upqc.qfim_eigs_reduced_by_layer[L] = np.asarray(
            qfim_result[f"L{L}_eigs_reduced_desc"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.qfim_thresh_reduced_by_layer[L] = np.asarray(
            qfim_result[f"L{L}_rank_threshold_reduced"],
            dtype=NP_REAL_DTYPE,
        )

        pure_rank_key = f"L{L}_rank_pure"
        if pure_rank_key in qfim_result:
            upqc.qfim_rank_pure_by_layer[L] = np.asarray(
                qfim_result[pure_rank_key],
                dtype=NP_INT_DTYPE,
            )
            upqc.qfim_eigs_pure_by_layer[L] = np.asarray(
                qfim_result[f"L{L}_eigs_pure_desc"],
                dtype=NP_REAL_DTYPE,
            )
            upqc.qfim_thresh_pure_by_layer[L] = np.asarray(
                qfim_result[f"L{L}_rank_threshold_pure"],
                dtype=NP_REAL_DTYPE,
            )
        else:
            upqc.qfim_rank_pure_by_layer[L] = None
            upqc.qfim_eigs_pure_by_layer[L] = None
            upqc.qfim_thresh_pure_by_layer[L] = None

    hs_path = os.path.join(
        upqc.hs_results_dir,
        "hs_random_points_reduced_0123.npz",
    )
    hs_result = _load_required_result(hs_path)

    upqc.hs_rank_reduced_by_layer = {}
    upqc.hs_eigs_reduced_by_layer = {}
    upqc.hs_thresh_reduced_by_layer = {}

    for L in layers:
        upqc.hs_rank_reduced_by_layer[L] = np.asarray(
            hs_result[f"L{L}_rank"],
            dtype=NP_INT_DTYPE,
        )
        upqc.hs_eigs_reduced_by_layer[L] = np.asarray(
            hs_result[f"L{L}_eigs_desc"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.hs_thresh_reduced_by_layer[L] = np.asarray(
            hs_result[f"L{L}_rank_threshold"],
            dtype=NP_REAL_DTYPE,
        )

    hs_pure_result = _load_required_result(
        os.path.join(upqc.hs_results_dir, "hs_random_points_pure_full.npz")
    )
    upqc.hs_rank_pure_by_layer = {
        L: np.asarray(hs_pure_result[f"L{L}_rank"], dtype=NP_INT_DTYPE)
        for L in layers
    }
    upqc.hs_eigs_pure_by_layer = {
        L: np.asarray(hs_pure_result[f"L{L}_eigs_desc"], dtype=NP_REAL_DTYPE)
        for L in layers
    }
    upqc.hs_thresh_pure_by_layer = {
        L: np.asarray(hs_pure_result[f"L{L}_rank_threshold"], dtype=NP_REAL_DTYPE)
        for L in layers
    }

    ortk_path = os.path.join(
        upqc.ortk_results_dir,
        "ortk_random_points.npz",
    )
    ortk_result = _load_required_result(ortk_path)
    upqc.ORTK_RANK_THRESHOLD = float(
        np.asarray(ortk_result["ortk_rank_threshold"]).item()
    )
    upqc.ORTK_PARTICIPATION_EPS = float(
        np.asarray(ortk_result["ortk_participation_eps"]).item()
    )
    upqc.ortk_rank_by_layer = {}
    upqc.ortk_effective_rank_by_layer = {}
    upqc.ortk_eigs_by_layer = {}
    upqc.ortk_trace_by_layer = {}

    for L in layers:
        upqc.ortk_rank_by_layer[L] = np.asarray(
            ortk_result[f"L{L}_rank"],
            dtype=NP_INT_DTYPE,
        )
        upqc.ortk_effective_rank_by_layer[L] = np.asarray(
            ortk_result[f"L{L}_effective_rank"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.ortk_eigs_by_layer[L] = np.asarray(
            ortk_result[f"L{L}_eigs_desc"],
            dtype=NP_REAL_DTYPE,
        )
        upqc.ortk_trace_by_layer[L] = np.asarray(
            ortk_result[f"L{L}_trace"],
            dtype=NP_REAL_DTYPE,
        )

    hessian_path = os.path.join(
        upqc.hessian_results_dir,
        "hessian_random_points.npz",
    )
    upqc.hessian_rank_by_layer = {}
    upqc.hessian_eigs_by_layer = {}
    upqc.hessian_thresh_by_layer = {}
    upqc.hessian_trace_by_layer = {}
    upqc.hessian_abs_eigsum_by_layer = {}

    if os.path.exists(hessian_path):
        hessian_result = _load_required_result(hessian_path)

        for L in layers:
            upqc.hessian_rank_by_layer[L] = np.asarray(
                hessian_result[f"L{L}_rank"],
                dtype=NP_INT_DTYPE,
            )
            upqc.hessian_eigs_by_layer[L] = np.asarray(
                hessian_result[f"L{L}_eigs_desc"],
                dtype=NP_REAL_DTYPE,
            )
            upqc.hessian_thresh_by_layer[L] = np.asarray(
                hessian_result[f"L{L}_rank_threshold"],
                dtype=NP_REAL_DTYPE,
            )
            upqc.hessian_trace_by_layer[L] = np.asarray(
                hessian_result[f"L{L}_trace"],
                dtype=NP_REAL_DTYPE,
            )
            upqc.hessian_abs_eigsum_by_layer[L] = np.asarray(
                hessian_result[f"L{L}_abs_eigsum"],
                dtype=NP_REAL_DTYPE,
            )


def _plot_random_qfim_results() -> None:
    qfim_eigs_pure_dir = upqc.qfim_eigs_pure_dir
    qfim_eigs_reduced_0123_dir = upqc.qfim_eigs_reduced_0123_dir
    hs_eigs_reduced_0123_dir = upqc.hs_eigs_reduced_0123_dir
    ortk_eigs_dir = upqc.ortk_eigs_dir
    hessian_eigs_dir = upqc.hessian_eigs_dir
    qfim_rank_random_dir = upqc.qfim_rank_random_dir
    hs_rank_random_dir = upqc.hs_rank_random_dir
    ortk_rank_random_dir = upqc.ortk_rank_random_dir
    ortk_effective_rank_random_dir = upqc.ortk_effective_rank_random_dir
    hessian_rank_random_dir = upqc.hessian_rank_random_dir

    os.makedirs(qfim_eigs_pure_dir, exist_ok=True)
    os.makedirs(qfim_eigs_reduced_0123_dir, exist_ok=True)
    os.makedirs(hs_eigs_reduced_0123_dir, exist_ok=True)
    os.makedirs(ortk_eigs_dir, exist_ok=True)
    os.makedirs(hessian_eigs_dir, exist_ok=True)
    os.makedirs(qfim_rank_random_dir, exist_ok=True)
    os.makedirs(hs_rank_random_dir, exist_ok=True)
    os.makedirs(ortk_rank_random_dir, exist_ok=True)
    os.makedirs(ortk_effective_rank_random_dir, exist_ok=True)
    os.makedirs(hessian_rank_random_dir, exist_ok=True)

    for L in upqc.layer_list:
        upqc.plot_style.save_eigenvalue_histograms_by_trial(
            upqc.qfim_eigs_reduced_by_layer[L],
            outdir=os.path.join(
                qfim_eigs_reduced_0123_dir,
                "histograms",
                "random_points",
                f"L{L}",
            ),
            matrix_tag="unitary_pqc_qfim",
            matrix_label="QFIM",
            num_layers=L,
            context_tag="random",
            context_label="random point",
            condition_tag="reduced0123",
            condition_label="reduced keep=(0,1,2,3)",
            color=METRIC_COLORS["qfim"],
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
        if upqc.hessian_eigs_by_layer.get(L) is not None:
            upqc._save_signed_eigs_scatterplot_by_index(
                upqc.hessian_eigs_by_layer[L],
                title=rf"Energy Hessian eigenvalues at {upqc.NUM_QFIM_SAMPLES} random points (L={L})",
                outpath=os.path.join(hessian_eigs_dir, f"L{L}.pdf"),
                rank_thresholds=upqc.hessian_thresh_by_layer[L],
                ylabel="Energy Hessian eigenvalue",
            )
            upqc.plot_style.save_eigenvalue_histograms_by_trial(
                upqc.hessian_eigs_by_layer[L],
                outdir=os.path.join(
                    hessian_eigs_dir,
                    "histograms",
                    "random_points",
                    f"L{L}",
                ),
                matrix_tag="unitary_pqc_energy_hessian",
                matrix_label="Energy Hessian",
                num_layers=L,
                context_tag="random",
                context_label="random point",
                color=METRIC_COLORS["hessian"],
            )
        if upqc.qfim_eigs_pure_by_layer[L] is not None:
            upqc.plot_style.save_eigenvalue_histograms_by_trial(
                upqc.qfim_eigs_pure_by_layer[L],
                outdir=os.path.join(
                    qfim_eigs_pure_dir,
                    "histograms",
                    "random_points",
                    f"L{L}",
                ),
                matrix_tag="unitary_pqc_qfim",
                matrix_label="QFIM",
                num_layers=L,
                context_tag="random",
                context_label="random point",
                condition_tag="pure_full",
                condition_label="pure full state",
                color=METRIC_COLORS["qfim"],
            )

    _plot_qfim_rank_min_mean_max_by_layer(
        upqc.qfim_rank_pure_by_layer,
        upqc.layer_list,
        title=(
            "Pure full-state QFIM effective rank minimum, mean, and maximum "
            f"at {upqc.NUM_QFIM_SAMPLES} random points"
        ),
        outpath=os.path.join(
            qfim_rank_random_dir,
            "qfim_rank_min_mean_max_random_points_pure_full.pdf",
        ),
    )
    _plot_qfim_rank_min_mean_max_by_layer(
        upqc.qfim_rank_reduced_by_layer,
        upqc.layer_list,
        title=(
            "QFIM effective rank minimum, mean, and maximum "
            f"at {upqc.NUM_QFIM_SAMPLES} random points (keep=(0,1,2,3))"
        ),
        outpath=os.path.join(
            qfim_rank_random_dir,
            "qfim_rank_min_mean_max_random_points_reduced_0123.pdf",
        ),
    )

    # Keep the historical maximum-only files for downstream compatibility.
    upqc.plot_qfim_rank_max_by_layer(
        upqc.qfim_rank_pure_by_layer,
        upqc.layer_list,
        color=METRIC_COLORS["qfim"],
        title=rf"Maximum pure full-state QFIM rank at {upqc.NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum QFIM effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(qfim_rank_random_dir, "qfim_rank_max_random_points_pure_full.pdf"),
        marker="s",
        lw=1.0,
    )
    upqc.plot_qfim_rank_max_by_layer(
        upqc.qfim_rank_reduced_by_layer,
        upqc.layer_list,
        color=METRIC_COLORS["qfim"],
        title=rf"Maximum QFIM rank at {upqc.NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum QFIM effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(qfim_rank_random_dir, "qfim_rank_max_random_points_reduced_0123.pdf"),
        marker="o",
        lw=1.0,
    )
    upqc.plot_qfim_rank_max_by_layer(
        upqc.hs_rank_reduced_by_layer,
        upqc.layer_list,
        color=METRIC_COLORS["hs"],
        title=rf"Maximum HS tangent Gram rank at {upqc.NUM_QFIM_SAMPLES} random points",
        ylabel=r"Maximum HS effective rank $(\lambda_k > 10^{-12})$",
        outpath=os.path.join(hs_rank_random_dir, "hs_rank_max_random_points_reduced_0123.pdf"),
        marker="D",
        lw=1.0,
    )
    upqc.plot_qfim_rank_max_by_layer(
        upqc.hs_rank_pure_by_layer,
        upqc.layer_list,
        color=METRIC_COLORS["hs"],
        title=rf"Maximum pure-full HS tangent Gram rank at {upqc.NUM_QFIM_SAMPLES} random points",
        ylabel="Maximum pure-full HS effective rank",
        outpath=os.path.join(hs_rank_random_dir, "hs_rank_max_random_points_pure_full.pdf"),
        marker="D",
        lw=1.0,
    )

    if upqc.hessian_rank_by_layer:
        upqc.plot_qfim_rank_max_by_layer(
            upqc.hessian_rank_by_layer,
            upqc.layer_list,
            color=METRIC_COLORS["hessian"],
            title=rf"Maximum energy Hessian rank at {upqc.NUM_QFIM_SAMPLES} random points",
            ylabel=r"Maximum Hessian rank $(|\eta_k| > 10^{-12})$",
            outpath=os.path.join(hessian_rank_random_dir, "hessian_rank_max_random_points.pdf"),
            marker="P",
            lw=1.0,
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
        (upqc.ortk_eigs_by_layer, upqc.ortk_eigs_dir,
         "ORTK", "ortk", False),
        (upqc.ortk_eigs_by_layer, os.path.join(upqc.ortk_eigs_dir, "pure_full"),
         "Pure-full ORTK (identical to reduced)", "ortk_pure", False),
        (upqc.hessian_eigs_by_layer, upqc.hessian_eigs_dir,
         "Absolute Hessian", "hessian_abs", True),
        (upqc.hessian_eigs_by_layer, os.path.join(upqc.hessian_eigs_dir, "pure_full"),
         "Pure-full absolute Hessian (identical to reduced)", "hessian_abs_pure", True),
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
    qfim_rank_path = os.path.join(
        upqc.qfim_results_dir,
        "qfim_rank_history_optimization_path_reduced_0123.npz",
    )
    qfim_result = _load_required_result(qfim_rank_path)
    layers = [int(L) for L in np.asarray(qfim_result["layers"], dtype=NP_INT_DTYPE)]
    upqc.sample_iters = np.asarray(qfim_result["sample_iters"], dtype=NP_INT_DTYPE)
    upqc.layer_list = layers
    upqc.qfim_rank_history_by_layer = {
        L: np.asarray(qfim_result[f"L{L}_rank"], dtype=NP_REAL_DTYPE)
        for L in layers
    }
    upqc.qfim_eigs_history_by_layer = {
        L: np.asarray(qfim_result[f"L{L}_eigs"], dtype=NP_REAL_DTYPE)
        for L in layers
    }
    qfim_pure_result = _load_required_result(
        os.path.join(
            upqc.qfim_results_dir,
            "qfim_rank_history_optimization_path_pure_full.npz",
        )
    )
    upqc.qfim_rank_history_pure_by_layer = {
        L: np.asarray(qfim_pure_result[f"L{L}_rank"], dtype=NP_REAL_DTYPE)
        for L in layers
    }
    upqc.qfim_eigs_history_pure_by_layer = {
        L: np.asarray(qfim_pure_result[f"L{L}_eigs"], dtype=NP_REAL_DTYPE)
        for L in layers
    }

    hs_rank_path = os.path.join(
        upqc.hs_results_dir,
        "hs_rank_history_optimization_path_reduced_0123.npz",
    )
    hs_result = _load_required_result(hs_rank_path)
    upqc.hs_rank_history_by_layer = {
        L: np.asarray(hs_result[f"L{L}_rank"], dtype=NP_REAL_DTYPE)
        for L in layers
    }
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
    upqc.hs_rank_history_pure_by_layer = {
        L: np.asarray(hs_pure_result[f"L{L}_rank"], dtype=NP_REAL_DTYPE)
        for L in layers
    }
    upqc.hs_eigs_history_pure_by_layer = {
        L: np.asarray(hs_pure_result[f"L{L}_eigs"], dtype=NP_REAL_DTYPE)
        for L in layers
    }

    ortk_rank_result = _load_required_result(
        os.path.join(
            upqc.ortk_results_dir,
            "ortk_rank_history_optimization_path.npz",
        )
    )
    ortk_effective_rank_result = _load_required_result(
        os.path.join(
            upqc.ortk_results_dir,
            "ortk_effective_rank_history_optimization_path.npz",
        )
    )
    ortk_eigs_result = _load_required_result(
        os.path.join(
            upqc.ortk_results_dir,
            "ortk_eigs_history_optimization_path.npz",
        )
    )
    ortk_trace_result = _load_required_result(
        os.path.join(
            upqc.ortk_results_dir,
            "ortk_trace_history_optimization_path.npz",
        )
    )
    upqc.ortk_rank_history_by_layer = {
        L: np.asarray(ortk_rank_result[f"L{L}"], dtype=NP_REAL_DTYPE)
        for L in layers
    }
    upqc.ortk_effective_rank_history_by_layer = {
        L: np.asarray(
            ortk_effective_rank_result[f"L{L}"],
            dtype=NP_REAL_DTYPE,
        )
        for L in layers
    }
    upqc.ortk_eigs_history_by_layer = {
        L: np.asarray(ortk_eigs_result[f"L{L}"], dtype=NP_REAL_DTYPE)
        for L in layers
    }
    upqc.ortk_trace_history_by_layer = {
        L: np.asarray(ortk_trace_result[f"L{L}"], dtype=NP_REAL_DTYPE)
        for L in layers
    }

    hessian_rank_path = os.path.join(
        upqc.hessian_results_dir,
        "hessian_rank_history_optimization_path.npz",
    )
    upqc.hessian_rank_history_by_layer = {}
    upqc.hessian_eigs_history_by_layer = {}
    upqc.hessian_thresh_history_by_layer = {}

    if os.path.exists(hessian_rank_path):
        hessian_result = _load_required_result(hessian_rank_path)
        upqc.hessian_rank_history_by_layer = {
            L: np.asarray(hessian_result[f"L{L}_rank"], dtype=NP_REAL_DTYPE)
            for L in layers
        }
        upqc.hessian_eigs_history_by_layer = {
            L: np.asarray(hessian_result[f"L{L}_eigs"], dtype=NP_REAL_DTYPE)
            for L in layers
        }
        upqc.hessian_thresh_history_by_layer = {
            L: np.asarray(
                hessian_result[f"L{L}_rank_threshold"],
                dtype=NP_REAL_DTYPE,
            )
            for L in layers
        }


def _plot_optimization_path_results() -> None:
    qfim_rank_optimization_path_max_dir = os.path.join(
        upqc.qfim_rank_optimization_path_dir,
        "max",
    )
    os.makedirs(qfim_rank_optimization_path_max_dir, exist_ok=True)

    def save_optimization_path_eigenvalue_histograms(
        eigs_history_by_layer: dict,
        *,
        outdir: str,
        matrix_tag: str,
        matrix_label: str,
        color: str,
        condition_tag: Optional[str] = None,
        condition_label: Optional[str] = None,
    ) -> None:
        target_iterations = np.asarray(upqc.sample_iters, dtype=NP_INT_DTYPE)

        for L in upqc.layer_list:
            eigs_L = eigs_history_by_layer.get(L)
            if eigs_L is None:
                continue

            eigs_L = np.asarray(eigs_L, dtype=NP_REAL_DTYPE)
            if eigs_L.ndim != 3:
                raise ValueError(
                    "Each optimization-path eigenvalue array must have shape "
                    "(num_runs, num_sample_iters, num_params)."
                )
            if (
                eigs_L.shape[1] != target_iterations.size
                and eigs_L.shape[0] == target_iterations.size
            ):
                eigs_L = np.transpose(eigs_L, (1, 0, 2))
            if eigs_L.shape[1] != target_iterations.size:
                raise ValueError(
                    f"Shape mismatch for L={L}: eigs_L.shape={eigs_L.shape}, "
                    f"len(sample_iters)={target_iterations.size}."
                )

            for time_idx, iteration in enumerate(target_iterations):
                upqc.plot_style.save_eigenvalue_histogram_across_trials(
                    eigs_L[:, time_idx, :],
                    outdir=os.path.join(outdir, "histograms", f"L{L}"),
                    matrix_tag=matrix_tag,
                    matrix_label=matrix_label,
                    num_layers=L,
                    context_tag="opt_path",
                    context_label="optimization path",
                    iteration=int(iteration),
                    condition_tag=condition_tag,
                    condition_label=condition_label,
                    color=color,
                )

    save_optimization_path_eigenvalue_histograms(
        upqc.qfim_eigs_history_by_layer,
        outdir=os.path.join(
            upqc.qfim_eigs_dir,
            "optimization_path_keep0123",
        ),
        matrix_tag="unitary_pqc_qfim",
        matrix_label="QFIM",
        color=METRIC_COLORS["qfim"],
        condition_tag="reduced0123",
        condition_label="reduced keep=(0,1,2,3)",
    )
    save_optimization_path_eigenvalue_histograms(
        upqc.qfim_eigs_history_pure_by_layer,
        outdir=os.path.join(upqc.qfim_eigs_dir, "optimization_path_pure_full"),
        matrix_tag="unitary_pqc_qfim",
        matrix_label="QFIM",
        color=METRIC_COLORS["qfim"],
        condition_tag="pure_full",
        condition_label="pure full state",
    )
    save_optimization_path_eigenvalue_histograms(
        upqc.hs_eigs_history_by_layer,
        outdir=os.path.join(
            upqc.hs_eigs_dir,
            "optimization_path_keep0123",
        ),
        matrix_tag="unitary_pqc_hs_gram",
        matrix_label="HS tangent Gram",
        color=METRIC_COLORS["hs"],
        condition_tag="reduced0123",
        condition_label="reduced keep=(0,1,2,3)",
    )
    save_optimization_path_eigenvalue_histograms(
        upqc.hs_eigs_history_pure_by_layer,
        outdir=os.path.join(upqc.hs_eigs_dir, "optimization_path_pure_full"),
        matrix_tag="unitary_pqc_hs_gram",
        matrix_label="HS tangent Gram",
        color=METRIC_COLORS["hs"],
        condition_tag="pure_full",
        condition_label="pure full state",
    )
    save_optimization_path_eigenvalue_histograms(
        upqc.ortk_eigs_history_by_layer,
        outdir=os.path.join(
            upqc.ortk_eigs_dir,
            "optimization_path",
        ),
        matrix_tag="unitary_pqc_ortk",
        matrix_label="Observable-Relevant Tangent Kernel",
        color=METRIC_COLORS["ortk"],
    )
    if upqc.hessian_eigs_history_by_layer:
        save_optimization_path_eigenvalue_histograms(
            upqc.hessian_eigs_history_by_layer,
            outdir=os.path.join(
                upqc.hessian_eigs_dir,
                "optimization_path",
            ),
            matrix_tag="unitary_pqc_energy_hessian",
            matrix_label="Energy Hessian",
            color=METRIC_COLORS["hessian"],
        )

    spectral_path_summaries = (
        (upqc.qfim_eigs_history_by_layer, upqc.qfim_eigs_dir, "QFIM", "qfim", False),
        (upqc.qfim_eigs_history_pure_by_layer, os.path.join(upqc.qfim_eigs_dir, "pure_full"),
         "Pure-full QFIM", "qfim_pure", False),
        (upqc.hs_eigs_history_by_layer, upqc.hs_eigs_dir, "HS tangent Gram", "hs", False),
        (upqc.hs_eigs_history_pure_by_layer, os.path.join(upqc.hs_eigs_dir, "pure_full"),
         "Pure-full HS tangent Gram", "hs_pure", False),
        (upqc.ortk_eigs_history_by_layer, upqc.ortk_eigs_dir, "ORTK", "ortk", False),
        (upqc.ortk_eigs_history_by_layer, os.path.join(upqc.ortk_eigs_dir, "pure_full"),
         "Pure-full ORTK (identical to reduced)", "ortk_pure", False),
        (upqc.hessian_eigs_history_by_layer, upqc.hessian_eigs_dir,
         "Absolute Hessian", "hessian_abs", True),
        (upqc.hessian_eigs_history_by_layer, os.path.join(upqc.hessian_eigs_dir, "pure_full"),
         "Pure-full absolute Hessian (identical to reduced)", "hessian_abs_pure", True),
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

    for metric_tag, metric_label, histories, base_dir in (
        ("qfim", "Pure-full QFIM", upqc.qfim_rank_history_pure_by_layer,
         os.path.join(upqc.qfim_rank_optimization_path_dir, "pure_full")),
        ("hs", "Pure-full HS tangent Gram", upqc.hs_rank_history_pure_by_layer,
         os.path.join(upqc.hs_rank_optimization_path_dir, "pure_full")),
    ):
        os.makedirs(base_dir, exist_ok=True)
        upqc.plot_qfim_rank_history_mean_by_layer(
            histories,
            upqc.layer_list,
            upqc.sample_iters,
            title=f"Mean {metric_label} effective rank along optimization path",
            outpath=os.path.join(base_dir, f"{metric_tag}_rank_mean_history_optimization_path_pure_full.pdf"),
            ylabel=f"Mean {metric_label} effective rank",
            cmap=upqc.cmap,
        )
        upqc.plot_qfim_rank_history_min_by_layer(
            histories,
            upqc.layer_list,
            upqc.sample_iters,
            title=f"Minimum {metric_label} effective rank along optimization path",
            outpath=os.path.join(base_dir, f"{metric_tag}_rank_min_history_optimization_path_pure_full.pdf"),
            ylabel=f"Minimum {metric_label} effective rank",
            cmap=upqc.cmap,
        )
        if metric_tag == "qfim":
            _plot_qfim_rank_history_max_by_layer(
                histories,
                upqc.layer_list,
                upqc.sample_iters,
                title=(
                    "Maximum pure-full QFIM effective rank along "
                    "optimization path"
                ),
                outpath=os.path.join(
                    base_dir,
                    "qfim_rank_max_history_optimization_path_pure_full.pdf",
                ),
                cmap=upqc.cmap,
            )

    upqc.plot_qfim_rank_history_mean_by_layer(
        upqc.qfim_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Mean QFIM effective rank along optimization path (keep=(0,1,2,3))",
        outpath=os.path.join(
            upqc.qfim_rank_optimization_path_mean_dir,
            "qfim_rank_mean_history_optimization_path_reduced_0123.pdf",
        ),
        cmap=upqc.cmap,
    )
    upqc.plot_qfim_rank_history_min_by_layer(
        upqc.qfim_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Minimum QFIM effective rank along optimization path (keep=(0,1,2,3))",
        outpath=os.path.join(
            upqc.qfim_rank_optimization_path_min_dir,
            "qfim_rank_min_history_optimization_path_reduced_0123.pdf",
        ),
        cmap=upqc.cmap,
    )
    _plot_qfim_rank_history_max_by_layer(
        upqc.qfim_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Maximum QFIM effective rank along optimization path (keep=(0,1,2,3))",
        outpath=os.path.join(
            qfim_rank_optimization_path_max_dir,
            "qfim_rank_max_history_optimization_path_reduced_0123.pdf",
        ),
        cmap=upqc.cmap,
    )
    upqc.plot_qfim_rank_history_mean_by_layer(
        upqc.hs_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Mean HS tangent Gram effective rank along optimization path (keep=(0,1,2,3))",
        outpath=os.path.join(
            upqc.hs_rank_optimization_path_mean_dir,
            "hs_rank_mean_history_optimization_path_reduced_0123.pdf",
        ),
        ylabel=r"Mean HS effective rank $(\lambda_k > 10^{-12})$",
        cmap=upqc.cmap,
    )
    upqc.plot_qfim_rank_history_min_by_layer(
        upqc.hs_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Minimum HS tangent Gram effective rank along optimization path (keep=(0,1,2,3))",
        outpath=os.path.join(
            upqc.hs_rank_optimization_path_min_dir,
            "hs_rank_min_history_optimization_path_reduced_0123.pdf",
        ),
        ylabel=r"Minimum HS effective rank $(\lambda_k > 10^{-12})$",
        cmap=upqc.cmap,
    )

    upqc.plot_qfim_rank_history_mean_by_layer(
        upqc.ortk_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Mean Observable-Relevant Tangent Kernel rank along optimization path",
        outpath=os.path.join(
            upqc.ortk_rank_optimization_path_mean_dir,
            "ortk_rank_mean_history_optimization_path.pdf",
        ),
        ylabel="Mean ORTK rank",
        cmap=upqc.cmap,
    )
    upqc.plot_qfim_rank_history_min_by_layer(
        upqc.ortk_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Minimum Observable-Relevant Tangent Kernel rank along optimization path",
        outpath=os.path.join(
            upqc.ortk_rank_optimization_path_min_dir,
            "ortk_rank_min_history_optimization_path.pdf",
        ),
        ylabel="Minimum ORTK rank",
        cmap=upqc.cmap,
        integer_y_axis=True,
    )
    upqc.plot_qfim_rank_history_mean_by_layer(
        upqc.ortk_effective_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title=(
            "Mean Observable-Relevant Tangent Kernel participation effective "
            "rank along optimization path"
        ),
        outpath=os.path.join(
            upqc.ortk_effective_rank_optimization_path_mean_dir,
            "ortk_effective_rank_mean_history_optimization_path.pdf",
        ),
        ylabel="Mean ORTK participation effective rank",
        cmap=upqc.cmap,
    )
    upqc.plot_qfim_rank_history_min_by_layer(
        upqc.ortk_effective_rank_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title=(
            "Minimum Observable-Relevant Tangent Kernel participation "
            "effective rank along optimization path"
        ),
        outpath=os.path.join(
            upqc.ortk_effective_rank_optimization_path_min_dir,
            "ortk_effective_rank_min_history_optimization_path.pdf",
        ),
        ylabel="Minimum ORTK participation effective rank",
        cmap=upqc.cmap,
        integer_y_axis=False,
    )
    upqc.plot_qfim_rank_history_mean_by_layer(
        upqc.ortk_trace_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Mean Observable-Relevant Tangent Kernel trace along optimization path",
        outpath=os.path.join(
            upqc.ortk_trace_optimization_path_dir,
            "ortk_trace_mean_history_optimization_path.pdf",
        ),
        ylabel="Mean ORTK trace",
        cmap=upqc.cmap,
    )
    upqc.plot_qfim_rank_history_min_by_layer(
        upqc.ortk_trace_history_by_layer,
        upqc.layer_list,
        upqc.sample_iters,
        title="Minimum Observable-Relevant Tangent Kernel trace along optimization path",
        outpath=os.path.join(
            upqc.ortk_trace_optimization_path_dir,
            "ortk_trace_min_history_optimization_path.pdf",
        ),
        ylabel="Minimum ORTK trace",
        cmap=upqc.cmap,
        integer_y_axis=False,
    )

    if upqc.hessian_rank_history_by_layer:
        upqc.plot_qfim_rank_history_mean_by_layer(
            upqc.hessian_rank_history_by_layer,
            upqc.layer_list,
            upqc.sample_iters,
            title="Mean energy Hessian rank along optimization path",
            outpath=os.path.join(
                upqc.hessian_rank_optimization_path_mean_dir,
                "hessian_rank_mean_history_optimization_path.pdf",
            ),
            ylabel=r"Mean Hessian rank $(|\eta_k| > 10^{-12})$",
            cmap=upqc.cmap,
        )
        upqc.plot_qfim_rank_history_min_by_layer(
            upqc.hessian_rank_history_by_layer,
            upqc.layer_list,
            upqc.sample_iters,
            title="Minimum energy Hessian rank along optimization path",
            outpath=os.path.join(
                upqc.hessian_rank_optimization_path_min_dir,
                "hessian_rank_min_history_optimization_path.pdf",
            ),
            ylabel=r"Minimum Hessian rank $(|\eta_k| > 10^{-12})$",
            cmap=upqc.cmap,
        )

        hessian_path_dir = os.path.join(
            upqc.hessian_eigs_dir,
            "optimization_path",
        )
        os.makedirs(hessian_path_dir, exist_ok=True)

        target_iterations = np.asarray(upqc.sample_iters, dtype=NP_INT_DTYPE)

        for L in upqc.layer_list:
            eigs_L = upqc.hessian_eigs_history_by_layer.get(L)
            if eigs_L is None:
                continue

            eigs_L = np.asarray(eigs_L, dtype=NP_REAL_DTYPE)
            if eigs_L.ndim != 3:
                continue

            if (
                eigs_L.shape[1] != target_iterations.size
                and eigs_L.shape[0] == target_iterations.size
            ):
                eigs_L = np.transpose(eigs_L, (1, 0, 2))

            if eigs_L.shape[1] != target_iterations.size:
                continue

            layer_dir = os.path.join(hessian_path_dir, f"L{L}")
            os.makedirs(layer_dir, exist_ok=True)
            thresholds_L = upqc.hessian_thresh_history_by_layer.get(L)
            thresholds_L_arr = None
            if thresholds_L is not None:
                thresholds_L_arr = np.asarray(thresholds_L, dtype=NP_REAL_DTYPE)
                if (
                    thresholds_L_arr.ndim == 2
                    and thresholds_L_arr.shape[1] != target_iterations.size
                    and thresholds_L_arr.shape[0] == target_iterations.size
                ):
                    thresholds_L_arr = thresholds_L_arr.T
                if (
                    thresholds_L_arr.ndim != 2
                    or thresholds_L_arr.shape[1] != target_iterations.size
                ):
                    thresholds_L_arr = None

            for time_idx, iteration in enumerate(target_iterations):
                iteration_int = int(iteration)
                upqc._save_signed_eigs_scatterplot_by_index(
                    eigs_L[:, time_idx, :],
                    title=(
                        rf"Energy Hessian eigenvalues along optimization path "
                        rf"(L={L}, iteration={iteration_int})"
                    ),
                    outpath=os.path.join(
                        layer_dir,
                        f"iter{iteration_int:06d}.pdf",
                    ),
                    rank_thresholds=(
                        None
                        if thresholds_L_arr is None
                        else thresholds_L_arr[:, time_idx]
                    ),
                    ylabel="Energy Hessian eigenvalue",
                )

    # ORTK and energy Hessian are invariant under tracing out qubit 4 for the
    # current system-only observables.  Emit explicitly labelled pure-full
    # views so both representations have a complete, symmetric figure tree.
    equivalent_histories = (
        ("ortk_rank", "ORTK rank", upqc.ortk_rank_history_by_layer, True),
        ("ortk_effective_rank", "ORTK participation effective rank",
         upqc.ortk_effective_rank_history_by_layer, False),
        ("ortk_trace", "ORTK trace", upqc.ortk_trace_history_by_layer, False),
        ("hessian_rank", "Energy Hessian rank",
         upqc.hessian_rank_history_by_layer, True),
    )
    for tag, label, histories, integer_axis in equivalent_histories:
        if not histories:
            continue
        output_dir = os.path.join(upqc.figures_dir, tag.split("_")[0], "pure_full")
        os.makedirs(output_dir, exist_ok=True)
        upqc.plot_qfim_rank_history_mean_by_layer(
            histories, upqc.layer_list, upqc.sample_iters,
            title=f"Mean pure-full {label} along optimization path",
            outpath=os.path.join(output_dir, f"{tag}_mean_optimization_path_pure_full.pdf"),
            ylabel=f"Mean {label}", cmap=upqc.cmap,
        )
        upqc.plot_qfim_rank_history_min_by_layer(
            histories, upqc.layer_list, upqc.sample_iters,
            title=f"Minimum pure-full {label} along optimization path",
            outpath=os.path.join(output_dir, f"{tag}_min_optimization_path_pure_full.pdf"),
            ylabel=f"Minimum {label}", cmap=upqc.cmap,
            integer_y_axis=integer_axis,
        )


def _plot_qfim_grad_alignment_results() -> None:
    if cfg.QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS is None:
        target_iterations = tuple(int(t) for t in upqc.sample_iters)
    else:
        target_iterations = tuple(
            int(t) for t in cfg.QFIM_GRAD_ALIGNMENT_TARGET_ITERATIONS
        )

    for L in upqc.layer_list:
        layer_data_dir = os.path.join(upqc.qfim_grad_align_results_dir, f"L{L}")
        layer_plot_dir = os.path.join(upqc.qfim_grad_align_dir, f"L{L}")
        os.makedirs(layer_plot_dir, exist_ok=True)

        for iteration in target_iterations:
            iter_tag = f"iter{iteration:06d}"
            data_path = os.path.join(
                layer_data_dir,
                f"qfim_grad_alignment_scatter_data_L{L}_{iter_tag}.npz",
            )
            if not os.path.exists(data_path):
                continue

            table = _load_required_result(data_path)
            upqc.plot_qfim_grad_alignment_table(
                table,
                title=(
                    rf"QFIM eigenvalue vs gradient weight, "
                    rf"L={L}, iteration {iteration}"
                ),
                outpath=os.path.join(
                    layer_plot_dir,
                    f"qfim_grad_weight_scatter_L{L}_{iter_tag}.pdf",
                ),
                log_x=cfg.LOG_X_QFIM_GRAD_ALIGNMENT,
                log_y=cfg.LOG_Y_QFIM_GRAD_ALIGNMENT,
                color_by=None,
                point_size=14.0,
                alpha=0.45,
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
    _plot_qfim_grad_alignment_results()

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
