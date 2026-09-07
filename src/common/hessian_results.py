"""Load saved DPQC Hessians and derive threshold-dependent plot quantities.

Schema 2 stores signed Hessian matrices and their parameter points; changing
the analysis threshold does not require running the quantum circuit again.
Schema 1 contains only summaries and can be used at its saved threshold.
This module deliberately depends only on NumPy, not on JAX or TensorCircuit.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


RESULT_NAME = "hessian_random_points.npz"
_PARAMETERS_PER_LAYER = {"dpqc": 14, "dpqc_reset": 12}
_MODEL_IDS = {
    "dpqc": "dpqc_dynamic_channel",
    "dpqc_reset": "dpqc_reset_fixed_rx_pi",
}
_COMMON_METADATA = {
    "schema_version", "h_param", "layers", "num_hessian_samples",
    "hessian_sample_seed_base", "parameter_distribution",
    "parameters_per_layer", "hessian_method", "hvp_chunk_size",
}
_LEGACY_METADATA = {
    "hessian_rank_threshold", "hessian_rank_definition",
    "hessian_condition_number_definition",
}


def _scalar(archive, key):
    value = np.asarray(archive[key])
    if value.shape != ():
        raise ValueError(f"Hessian archive {key} must be scalar.")
    return value.item()


def _integer_scalar(archive, key, minimum=1):
    raw = _scalar(archive, key)
    if isinstance(raw, (bool, complex, str)) or not np.isfinite(raw):
        raise ValueError(f"Hessian archive {key} must be an integer.")
    value = int(raw)
    if value != raw or value < minimum:
        raise ValueError(f"Hessian archive {key} must be an integer >= {minimum}.")
    return value


def _layers(values):
    raw = np.asarray(values)
    if (
        raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.number)
        or np.iscomplexobj(raw) or not np.all(np.isfinite(raw))
        or np.any(raw <= 0) or np.any(raw != np.rint(raw))
        or np.unique(raw).size != raw.size
    ):
        raise ValueError("Hessian layers must be unique positive integers.")
    return sorted(int(layer) for layer in raw)


def _real_array(archive, key, shape):
    if key not in archive:
        raise KeyError(f"Hessian archive is missing: {key}.")
    values = np.asarray(archive[key])
    if (
        values.shape != shape or not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values) or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"Invalid Hessian array {key}; expected finite real shape {shape}.")
    return values.astype(np.float64, copy=False)


def load_random_hessian_result(
    results_dir: Path,
    *,
    expected_h_param: float,
    requested_layers=None,
    expected_output_family: str = "dpqc",
    rank_threshold: float = 1e-12,
):
    """Validate an archive and derive rank/condition from its signed matrices.

    Rank counts ``abs(eigenvalue) >= rank_threshold``. Condition is the ratio
    of the largest to smallest active absolute eigenvalue, or NaN if none are
    active. The returned matrices, theta points, and signed eigenvalues are
    available for additional plots; these dictionaries are empty for schema 1.
    Only requested layers are loaded and diagonalized.
    """
    threshold = float(rank_threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("Hessian rank threshold must be finite and positive.")
    family = str(expected_output_family)
    if family not in _PARAMETERS_PER_LAYER:
        raise ValueError(f"Unsupported Hessian output family: {family!r}.")
    path = Path(results_dir) / RESULT_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Random-point Hessian result was not found: {path}. "
            "Run with --hessian-only or --with-hessian and without "
            "--reuse-hessian-results to compute it first."
        )

    rank_by_layer, condition_by_layer = {}, {}
    hessian_by_layer, theta_by_layer, eigenvalues_by_layer = {}, {}, {}
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(_COMMON_METADATA.difference(archive.files))
        if missing:
            raise KeyError(f"Hessian archive {path} is missing: {', '.join(missing)}.")
        schema = _integer_scalar(archive, "schema_version")
        if schema not in (1, 2):
            raise ValueError(f"Unsupported Hessian schema version {schema}; expected 1 or 2.")
        archived_h = float(_scalar(archive, "h_param"))
        if not math.isclose(archived_h, float(expected_h_param), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Hessian archive h_param mismatch: {archived_h} != {expected_h_param}.")
        parameters_per_layer = _integer_scalar(archive, "parameters_per_layer")
        if parameters_per_layer != _PARAMETERS_PER_LAYER[family]:
            raise ValueError(f"Unexpected Hessian parameter count per layer for {family}.")
        for key, expected in (("output_family", family), ("model_id", _MODEL_IDS[family])):
            if key in archive:
                if str(_scalar(archive, key)) != expected:
                    raise ValueError(f"Hessian archive {key} mismatch; expected {expected!r}.")
            elif family != "dpqc" or schema == 2:
                raise KeyError(f"Hessian archive is missing: {key}.")
        num_samples = _integer_scalar(archive, "num_hessian_samples")
        _integer_scalar(archive, "hessian_sample_seed_base", minimum=0)
        _integer_scalar(archive, "hvp_chunk_size")
        available_layers = _layers(archive["layers"])
        layers = available_layers if requested_layers is None else _layers(requested_layers)
        missing_layers = sorted(set(layers).difference(available_layers))
        if missing_layers:
            raise KeyError(f"Requested Hessian layers are absent from the archive: {missing_layers}.")

        if schema == 1:
            missing = sorted(_LEGACY_METADATA.difference(archive.files))
            if missing:
                raise KeyError(f"Legacy Hessian archive is missing: {', '.join(missing)}.")
            archived_threshold = float(_scalar(archive, "hessian_rank_threshold"))
            if archived_threshold != threshold:
                raise ValueError(
                    f"Legacy Hessian archive contains only rank/condition summaries at "
                    f"threshold {archived_threshold:g}, not Hessian matrices. "
                    "Recompute the Hessian stage with the updated compute script "
                    "before changing --hessian-rank-threshold."
                )

        for layer in layers:
            dimension = parameters_per_layer * layer
            if schema == 2:
                matrices = _real_array(archive, f"L{layer}_hessian", (num_samples, dimension, dimension))
                theta = _real_array(archive, f"L{layer}_theta", (num_samples, dimension))
                if not np.allclose(matrices, matrices.swapaxes(-1, -2), rtol=1e-10, atol=1e-14):
                    raise ValueError(f"Hessian matrices must be symmetric at L={layer}.")
                eigenvalues = np.linalg.eigvalsh(matrices)
                if not np.all(np.isfinite(eigenvalues)):
                    raise FloatingPointError(f"Non-finite Hessian eigenvalues at L={layer}.")
                absolute = np.abs(eigenvalues)
                active = absolute >= threshold
                ranks = np.count_nonzero(active, axis=1)
                largest = np.max(np.where(active, absolute, 0.0), axis=1)
                smallest = np.min(np.where(active, absolute, np.inf), axis=1)
                conditions = np.full(num_samples, np.nan)
                np.divide(largest, smallest, out=conditions, where=ranks > 0)
                hessian_by_layer[layer] = matrices
                theta_by_layer[layer] = theta
                eigenvalues_by_layer[layer] = eigenvalues
            else:
                ranks = _real_array(archive, f"L{layer}_rank", (num_samples,))
                if np.any(ranks != np.rint(ranks)) or np.any(ranks < 0) or np.any(ranks > dimension):
                    raise ValueError(f"Invalid Hessian rank array L{layer}_rank.")
                ranks = ranks.astype(np.int64)
                condition_key = f"L{layer}_condition_number"
                if condition_key not in archive:
                    raise KeyError(f"Hessian archive is missing: {condition_key}.")
                conditions = np.asarray(archive[condition_key])
                if conditions.shape != (num_samples,) or np.iscomplexobj(conditions):
                    raise ValueError(f"Invalid Hessian condition-number array {condition_key}.")
                conditions = conditions.astype(np.float64)
                if (
                    np.any(np.isinf(conditions))
                    or np.any(conditions[np.isfinite(conditions)] < 1.0 - 1e-12)
                    or not np.array_equal(np.isfinite(conditions), ranks > 0)
                ):
                    raise ValueError(f"Hessian rank/condition mismatch at L={layer}.")
            rank_by_layer[layer] = ranks
            condition_by_layer[layer] = conditions

    return {
        "path": path, "schema_version": schema, "output_family": family,
        "layers": layers, "num_samples": num_samples, "threshold": threshold,
        "rank_by_layer": rank_by_layer, "condition_by_layer": condition_by_layer,
        "hessian_by_layer": hessian_by_layer, "theta_by_layer": theta_by_layer,
        "eigenvalues_by_layer": eigenvalues_by_layer,
    }
