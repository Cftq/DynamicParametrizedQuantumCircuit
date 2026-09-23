"""Mean log det(I + kappa F) from archived random-point QFIM spectra.

Numerical routines require only NumPy. No quantum circuit, differentiation, or
training code is imported. All saved positive eigenvalues contribute, without
rank thresholding or parameter-dimension normalization. Logs are natural.
"""

from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np


DEFAULT_KAPPA = 1.0


def _positive_integer(value, name):
    array = np.asarray(value)
    if (
        array.ndim != 0 or not np.issubdtype(array.dtype, np.integer)
        or int(array) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(array)


def _finite_scalar(value, name):
    array = np.asarray(value)
    if (
        array.ndim != 0 or not np.issubdtype(array.dtype, np.number)
        or np.iscomplexobj(array) or not np.isfinite(array)
    ):
        raise ValueError(f"{name} must be a finite real scalar.")
    return float(array)


def _spectra_array(value, layer):
    array = np.asarray(value)
    if (
        array.ndim != 2 or min(array.shape) == 0
        or not np.issubdtype(array.dtype, np.number)
        or np.iscomplexobj(array) or not np.all(np.isfinite(array))
    ):
        raise ValueError(
            f"Layer {layer}: eigenvalues must be a nonempty finite real "
            "array with shape (samples, parameters)."
        )
    with np.errstate(over="ignore"):
        array = array.astype(np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Layer {layer}: eigenvalues must be representable as finite float64 values.")
    return array


def _metadata_matches(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return False
    if np.issubdtype(expected.dtype, np.floating):
        return (
            np.issubdtype(actual.dtype, np.number)
            and not np.iscomplexobj(actual)
            and np.all(np.isfinite(actual))
            and np.allclose(actual, expected, rtol=1e-12, atol=1e-14)
        )
    if expected.dtype.kind == "b" and actual.dtype.kind != "b":
        return False
    if expected.dtype.kind in "iu" and actual.dtype.kind not in "iu":
        return False
    return np.array_equal(actual, expected)


def load_random_qfim_spectra(
    path, *, expected_h_param=None, parameters_per_layer=None,
    required_metadata=None,
):
    """Read spectra and metadata without loading archived parameter vectors.

    Legacy DPQC archives have no explicit masking flag and are accepted: their
    producer stores all positive eigenvalues after clipping negative roundoff.
    Archives explicitly declaring rank-threshold masking are rejected, since
    discarded small eigenvalues cannot be recovered by visualization.
    ``required_metadata`` lets a caller require its canonical model schema.
    """
    path = Path(path)
    expected_params = (
        None if parameters_per_layer is None
        else _positive_integer(parameters_per_layer, "parameters_per_layer")
    )
    with np.load(path, allow_pickle=False) as archive:
        metadata = {}
        for key in archive.files:
            if re.match(r"^L\d+_", key):
                continue
            value = archive[key]
            metadata[key] = value.item() if value.ndim == 0 else value.copy()
        for key, expected in (required_metadata or {}).items():
            if key not in metadata or not _metadata_matches(metadata[key], expected):
                raise ValueError(f"{path}: required metadata {key!r} does not match {expected!r}.")
        if "analysis_kind" in metadata and not _metadata_matches(metadata["analysis_kind"], "random_points"):
            raise ValueError(f"{path}: analysis_kind must be random_points.")
        if "eigenvalues_threshold_masked" in metadata:
            if not _metadata_matches(metadata["eigenvalues_threshold_masked"], False):
                raise ValueError(f"{path}: threshold-masked eigenvalues cannot determine the requested log determinant.")
        if "eigenvalue_order" in metadata and not _metadata_matches(metadata["eigenvalue_order"], "descending"):
            raise ValueError(f"{path}: eigenvalue_order must be descending.")
        if "h_param" in metadata:
            saved_h = _finite_scalar(metadata["h_param"], "h_param")
        elif expected_h_param is not None:
            raise ValueError(f"{path}: missing h_param metadata.")
        if expected_h_param is not None:
            expected_h = _finite_scalar(expected_h_param, "expected_h_param")
            if not math.isclose(saved_h, expected_h, rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError(f"{path}: h_param does not match {expected_h!r}.")
        if "num_params_per_layer" in metadata:
            saved_params = _positive_integer(metadata["num_params_per_layer"], "num_params_per_layer")
            if expected_params is not None and saved_params != expected_params:
                raise ValueError(f"{path}: num_params_per_layer does not match {expected_params}.")
            expected_params = saved_params
        expected_samples = (
            _positive_integer(metadata["num_qfim_samples"], "num_qfim_samples")
            if "num_qfim_samples" in metadata else None
        )
        layers = np.asarray(metadata.get("layers", []))
        if (
            layers.ndim != 1 or layers.size == 0
            or not np.issubdtype(layers.dtype, np.integer)
            or np.any(layers <= 0) or np.unique(layers).size != layers.size
        ):
            raise ValueError(f"{path}: layers must be a nonempty vector of unique positive integers.")
        eigenvalues_by_layer = {}
        for layer in sorted(int(value) for value in layers):
            key = f"L{layer}_eigs_desc"
            if key not in archive.files:
                raise ValueError(f"{path}: missing saved spectrum {key}.")
            values = _spectra_array(archive[key], layer)
            if expected_samples is None:
                expected_samples = values.shape[0]
            if values.shape[0] != expected_samples:
                raise ValueError(f"{path}: layer {layer} sample count does not match {expected_samples}.")
            if expected_params is not None and values.shape[1] != expected_params * layer:
                raise ValueError(f"{path}: layer {layer} parameter count does not match {expected_params * layer}.")
            eigenvalues_by_layer[layer] = values
    return {
        "eigenvalues_by_layer": eigenvalues_by_layer,
        "metadata": metadata,
        "source_path": path,
    }


def compute_qfim_logdet(eigenvalues_by_layer, *, kappa=DEFAULT_KAPPA):
    """Return the empirical mean of sample log determinants at each layer.

    For each sample use sum_i log1p(kappa * lambda_i), then average over
    samples. Tiny negative values are clipped only within
    64 * n_parameters * float64_epsilon * max_i(abs(lambda_i)) for that
    sample; materially negative spectra are rejected. No positive value is
    discarded. SEM is the sample standard deviation / sqrt(n), zero for n=1.
    """
    kappa = _finite_scalar(kappa, "kappa")
    if kappa <= 0.0:
        raise ValueError("kappa must be finite and positive.")
    if not eigenvalues_by_layer:
        raise ValueError("Saved random-point QFIM spectra are required.")
    for layer in eigenvalues_by_layer:
        _positive_integer(layer, "Layer key")
    layers = np.asarray(sorted(eigenvalues_by_layer), dtype=np.int64)
    logdet_by_layer = {}
    tolerances_by_layer = {}
    statistics = []
    sample_counts = []
    parameter_counts = []
    clipped_counts = []
    for layer in layers:
        values = _spectra_array(eigenvalues_by_layer[int(layer)], layer)
        tolerance = (
            64.0 * np.finfo(np.float64).eps * values.shape[1]
            * np.max(np.abs(values), axis=1)
        )
        if np.any(values < -tolerance[:, None]):
            raise ValueError(f"Layer {layer}: eigenvalues contain materially negative values beyond roundoff tolerance.")
        clipped_counts.append(int(np.count_nonzero(values < 0.0)))
        values = np.maximum(values, 0.0)
        with np.errstate(over="ignore", under="ignore"):
            products = kappa * values
        overflow = np.isinf(products)
        with np.errstate(over="ignore"):
            contributions = np.log1p(products)
        if np.any(overflow):
            contributions[overflow] = np.logaddexp(
                0.0, math.log(kappa) + np.log(values[overflow]),
            )
        samples = np.sum(contributions, axis=1)
        if not np.all(np.isfinite(samples)):
            raise ValueError(f"Layer {layer}: log determinants are non-finite.")
        std = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
        statistics.append((np.mean(samples), std, std / math.sqrt(samples.size), np.min(samples), np.max(samples)))
        logdet_by_layer[int(layer)] = samples
        tolerances_by_layer[int(layer)] = tolerance
        sample_counts.append(samples.size)
        parameter_counts.append(values.shape[1])
    return {
        "kappa": kappa,
        "layers": layers,
        "logdet_by_layer": logdet_by_layer,
        "negative_roundoff_tolerance_by_layer": tolerances_by_layer,
        "num_samples_by_layer": np.asarray(sample_counts, dtype=np.int64),
        "num_parameters_by_layer": np.asarray(parameter_counts, dtype=np.int64),
        "num_clipped_negative_by_layer": np.asarray(clipped_counts, dtype=np.int64),
        **dict(zip(("mean", "std", "sem", "min", "max"), np.asarray(statistics).T)),
    }


def save_qfim_logdet_outputs(
    result, output_dir, *, keep_key, title=None, source_path=None, metadata=None,
):
    """Save the layer-vs-mean (+/- SEM) PDF and pickle-free statistics NPZ.

    Input provenance metadata is stored with a ``source_`` prefix. The source
    spectra themselves remain in the original archive; all sample log
    determinants and sample-wise negative-roundoff tolerances are saved here.
    """
    if __package__:
        from .plot import new_fig_ax, save_fig
    else:
        from plot import new_fig_ax, save_fig

    if not isinstance(keep_key, str) or not re.fullmatch(r"[A-Za-z0-9_]+", keep_key):
        raise ValueError("keep_key must contain only letters, digits, and underscores.")
    kappa = float(result["kappa"])
    kappa_token = str(kappa)
    if kappa_token.endswith(".0"):
        kappa_token = kappa_token[:-2]
    basename = f"qfim_logdet_random_points_{keep_key}_kappa_{kappa_token}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"{basename}.pdf"
    statistics_path = output_dir / f"{basename}.npz"
    arrays = {
        key: np.asarray(value) for key, value in result.items()
        if key not in ("logdet_by_layer", "negative_roundoff_tolerance_by_layer")
    }
    arrays.update({
        "keep_key": np.asarray(keep_key),
        "metric": np.asarray("mean_theta[log det(I + kappa F(theta))]"),
        "estimator": np.asarray("mean of sample sums of log1p(kappa * eigenvalue)"),
        "log_base": np.asarray("e"),
        "dimension_normalization": np.asarray("none"),
        "positive_eigenvalue_threshold": np.asarray(0.0),
        "negative_eigenvalue_policy": np.asarray("clip only sample-wise roundoff: 64 * parameter_count * float64_epsilon * max_abs_eigenvalue"),
        "sample_source": np.asarray("saved random-point QFIM spectra; no QFIM or training recomputation"),
    })
    if title is not None:
        arrays["title"] = np.asarray(title)
    if source_path is not None:
        arrays["source_path"] = np.asarray(str(source_path))
    for layer in result["layers"]:
        arrays[f"L{layer}_logdet"] = result["logdet_by_layer"][int(layer)]
        arrays[f"L{layer}_negative_roundoff_tolerance"] = result["negative_roundoff_tolerance_by_layer"][int(layer)]
    for key, value in (metadata or {}).items():
        if not isinstance(key, str) or f"source_{key}" in arrays:
            raise ValueError(f"Invalid or reserved metadata key: {key!r}.")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"Metadata {key!r} must not require pickle storage.")
        arrays[f"source_{key}"] = array
    np.savez_compressed(statistics_path, **arrays)

    layers = result["layers"]
    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.30)
    ax.errorbar(
        layers, result["mean"], yerr=result["sem"], fmt="o-", color="C0",
        capsize=2.5, label="Mean $\\pm$ SEM\n" + rf"$\kappa={kappa:g}$",
    )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(r"$\overline{\mathcal{L}}_{\kappa}^{(L)}$")
    indices = np.unique(np.linspace(0, len(layers) - 1, min(len(layers), 7)).round().astype(int))
    ax.set_xticks(layers[indices])
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    if title is not None:
        ax.set_title(title)
    save_fig(fig, ax, str(figure_path), outside_legend=True, legend_space_frac=0.30)
    return {"figure_path": figure_path, "statistics_path": statistics_path}
