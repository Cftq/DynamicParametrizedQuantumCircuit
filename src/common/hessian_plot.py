"""QFIM-style diagnostics for saved energy Hessians, without JAX imports.

Magnitude spectra define participation rank, entropy and threshold counts.
The signed trace and signed spectrum are kept explicit because energy
Hessians can be indefinite. All statistics are over random parameter points.
"""

from pathlib import Path

import numpy as np

from hessian_statistics import finite_sample_statistics, hessian_spectral_statistics


def _threshold_tex(threshold):
    mantissa, exponent = f"{float(threshold):.12e}".split("e")
    mantissa, exponent = float(mantissa), int(exponent)
    if exponent == 0:
        return f"{mantissa:g}"
    if mantissa == 1.0:
        return rf"10^{{{exponent}}}"
    return rf"{mantissa:g}\times 10^{{{exponent}}}"


def _layer_axis(ax, layers):
    ax.set_xlabel("Number of Layers")
    ax.set_xticks(layers)
    # Dense early layers need angled labels at the shared publication width.
    ax.set_xticklabels([str(layer) for layer in layers], rotation=60, ha="right")
    ax.grid(True, axis="y", alpha=0.3)


def plot_hessian_summary(
    values_by_layer, layers, *, ylabel, title, outpath,
    lower_bound_zero=False, integer_ticks=False, empty_message=None,
):
    """Save layerwise mean +/- sample SEM with explicit extrema curves."""
    from matplotlib.ticker import MaxNLocator
    from plot import new_fig_ax, save_fig

    layers = [int(layer) for layer in layers if int(layer) in values_by_layer]
    stats = np.asarray([finite_sample_statistics(values_by_layer[L]) for L in layers])
    if not layers:
        raise ValueError("No Hessian layers are available for plotting.")
    mean, sem, minimum, maximum, count = stats.T
    if not np.any(count > 0) and empty_message is None:
        raise ValueError(f"No finite Hessian statistics are available for {title}.")
    fig, ax = new_fig_ax(outside_legend=False)
    if np.any(count > 0):
        ax.errorbar(
            layers, mean, yerr=sem, fmt="o-", color="C0", capsize=3,
            label=r"Mean $\pm$ SEM", zorder=3,
        )
        ax.plot(layers, minimum, "v--", color="C2", label="Minimum")
        ax.plot(layers, maximum, "^--", color="C3", label="Maximum")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, empty_message, ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
    _layer_axis(ax, layers)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if lower_bound_zero:
        ax.set_ylim(bottom=0.0)
    if integer_ticks and np.any(count > 0):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, ax, str(outpath), outside_legend=False)
    return outpath


def _save_eigenvalue_counts(metrics_by_layer, layers, thresholds, outpath):
    from matplotlib import colormaps
    from matplotlib.ticker import MaxNLocator
    from plot import new_fig_ax, save_fig

    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.24)
    cmap = colormaps.get_cmap("viridis")
    for index, threshold in enumerate(thresholds):
        stats = np.asarray([
            finite_sample_statistics(metrics_by_layer[L]["eigcounts"][:, index])
            for L in layers
        ])
        ax.errorbar(
            layers, stats[:, 0], yerr=stats[:, 1], fmt="o-", capsize=3,
            color=cmap(index / max(len(thresholds) - 1, 1)),
            label=rf"$|\lambda_i| \geq {_threshold_tex(threshold)}$",
        )
    _layer_axis(ax, layers)
    ax.set_ylabel("Mean Hessian eigenvalue count")
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, ax, str(outpath), outside_legend=True, legend_space_frac=0.24)


def _save_spectrum(eigenvalues, *, layer, threshold, signed, outpath):
    from plot import new_fig_ax, save_fig

    spectrum = np.sort(eigenvalues if signed else np.abs(eigenvalues), axis=1)[:, ::-1]
    dimension = spectrum.shape[1]
    positions = np.broadcast_to(np.arange(1, dimension + 1), spectrum.shape)
    fig, ax = new_fig_ax(outside_legend=False)
    if signed:
        # Symlog retains negative and exactly zero eigenvalues without clipping.
        ax.scatter(
            positions.ravel(), spectrum.ravel(), s=10, alpha=0.5,
            color="C0", edgecolors="black", linewidths=0.15, rasterized=True,
        )
        ax.set_yscale("symlog", linthresh=threshold, linscale=2.0)
        ax.axhline(0.0, color="0.5", linewidth=0.5)
        ax.set_ylabel("Signed Hessian eigenvalue")
    else:
        # Zero is not representable on a log axis. Mark it at an explicit floor.
        floor = min(threshold, 1e-16)
        zero = spectrum == 0.0
        ax.scatter(
            positions[~zero], spectrum[~zero], s=10, alpha=0.5,
            color="C0", edgecolors="black", linewidths=0.15, rasterized=True,
        )
        if np.any(zero):
            ax.scatter(
                positions[zero], np.full(np.count_nonzero(zero), floor),
                s=10, alpha=0.5, color="C3", marker="v", rasterized=True,
                label=rf"Zero (shown at ${_threshold_tex(floor)}$)",
            )
            ax.legend(loc="best", fontsize=9)
        ax.set_yscale("log")
        ax.set_ylabel(r"Absolute Hessian eigenvalue $|\lambda_i|$")
    ticks = np.unique(np.rint(np.linspace(1, dimension, min(9, dimension))).astype(int))
    ax.set_xticks(ticks)
    ax.set_xlim(0.5, dimension + 0.5)
    ax.set_xlabel("Eigenvalue index (descending order)")
    ax.set_title(f"Hessian spectrum at L={layer}, {spectrum.shape[0]} random points")
    ax.grid(True, which="both", alpha=0.3)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, ax, str(outpath), outside_legend=False)


def save_hessian_statistic_figures(result, *, figures_dir, count_thresholds, h_param):
    """Save QFIM counterpart PDFs and auditable per-sample/statistical arrays.

    ``result`` is the validated matrix archive from ``hessian_results``. Its
    eigenvalues are reused, so plotting never repeats the eigendecomposition.
    Layer and sample counts come from the archive, not compute configuration.
    """
    layers = [int(layer) for layer in result["layers"]]
    threshold = result["threshold"]
    thresholds = tuple(float(value) for value in count_thresholds)
    figures_dir = Path(figures_dir)
    metrics_by_layer = {}
    for layer in layers:
        metrics = hessian_spectral_statistics(
            result["eigenvalues_by_layer"][layer],
            threshold=threshold, count_thresholds=thresholds,
        )
        metrics["abs_entry_sum"] = np.sum(np.abs(result["hessian_by_layer"][layer]), axis=(1, 2))
        metrics_by_layer[layer] = metrics

    active_label = rf"$|\lambda_i| \geq {_threshold_tex(threshold)}$"
    specs = {
        "participation_rank": (
            "effective_rank/hessian_participation_rank_random_points.pdf",
            "Hessian participation effective rank\n"
            rf"$(\sum |\lambda_i|)^2/\sum\lambda_i^2$, $|\lambda_i|>{_threshold_tex(threshold)}$",
            True,
        ),
        "absolute_trace": (
            "hessian_trace/hessian_absolute_trace_random_points.pdf",
            "Hessian absolute spectral sum\n" + rf"$\sum |\lambda_i|$, {active_label}",
            True,
        ),
        "trace": (
            "hessian_trace/hessian_signed_trace_random_points.pdf",
            r"Signed Hessian trace $\mathrm{Tr}(H)=\sum_i\lambda_i$",
            False,
        ),
        "shannon_entropy": (
            "shannon_entropy/hessian_shannon_entropy_random_points.pdf",
            "Hessian spectral Shannon entropy (nats)\n" + active_label,
            True,
        ),
        "abs_entry_sum": (
            "hessian_abs_entry_sum_mean_errorbar_random_points.pdf",
            r"Hessian absolute-entry sum $\sum_{ij}|H_{ij}|$",
            True,
        ),
    }
    figure_paths = {}
    for key, (filename, label, nonnegative) in specs.items():
        figure_paths[key] = plot_hessian_summary(
            {L: metrics_by_layer[L][key] for L in layers}, layers,
            ylabel=label, title=f"Hessian {key} at {result['num_samples']} random points",
            outpath=figures_dir / filename, lower_bound_zero=nonnegative,
        )
    count_path = figures_dir / "hessian_eigcount/random_points/hessian_eigcount_threshold_overlay_random_points.pdf"
    _save_eigenvalue_counts(metrics_by_layer, layers, thresholds, count_path)
    figure_paths["eigcounts"] = count_path
    for layer in layers:
        for signed, kind in ((False, "absolute"), (True, "signed")):
            outpath = figures_dir / f"hessian_eigs/{kind}/L{layer}_{kind}.pdf"
            _save_spectrum(
                result["eigenvalues_by_layer"][layer], layer=layer,
                threshold=threshold, signed=signed, outpath=outpath,
            )
            figure_paths[f"L{layer}_{kind}"] = outpath

    arrays = {
        "schema_version": np.asarray(1),
        "source_archive": np.asarray(str(result["path"])),
        "output_family": np.asarray(result["output_family"]),
        "h_param": np.asarray(h_param),
        "layers": np.asarray(layers, dtype=np.int64),
        "num_hessian_samples": np.asarray(result["num_samples"]),
        "rank_threshold": np.asarray(threshold),
        "eigcount_thresholds": np.asarray(thresholds),
        "participation_definition": np.asarray("(sum(abs(lambda[abs(lambda)>threshold])))^2 / sum(lambda[abs(lambda)>threshold]^2)"),
        "absolute_trace_definition": np.asarray("sum(abs(lambda[abs(lambda)>=threshold]))"),
        "trace_definition": np.asarray("full signed sum(lambda), without threshold"),
        "active_signed_trace_definition": np.asarray("sum(lambda[abs(lambda)>=threshold])"),
        "entropy_definition": np.asarray("-sum(p*log(p)), p proportional to abs(lambda) at abs(lambda)>=threshold; nats"),
        "zero_active_spectrum_policy": np.asarray("participation rank, entropy and absolute trace are zero"),
        "sem_definition": np.asarray("sample standard deviation (ddof=1) / sqrt(finite sample count); singleton SEM=0"),
    }
    scalar_keys = [key for key in metrics_by_layer[layers[0]] if key != "eigcounts"]
    for layer, metrics in metrics_by_layer.items():
        for key, values in metrics.items():
            arrays[f"L{layer}_{key}"] = values
    for key in scalar_keys:
        stats = np.asarray([finite_sample_statistics(metrics_by_layer[L][key]) for L in layers])
        for name, values in zip(("mean", "sem", "minimum", "maximum", "finite_sample_count"), stats.T):
            arrays[f"{key}_{name}"] = values
    data_path = figures_dir / "hessian_statistics_random_points.npz"
    np.savez_compressed(data_path, **arrays)
    return {"figure_paths": figure_paths, "data_path": data_path, "metrics_by_layer": metrics_by_layer}
