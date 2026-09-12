"""Read outcome-1 unitary Hessians without initializing the quantum simulator.

Schema 1 originally held rank/condition summaries and now optionally includes
the raw signed matrices. Those matrices allow spectral plots and changing the
analysis threshold without repeating any circuit calculations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .hessian_curvature import load_optional_hessian_matrices
except ImportError:  # The visualization scripts put src/common on sys.path.
    from hessian_curvature import load_optional_hessian_matrices


RESULT_NAME = "hessian_random_points.npz"
OUTPUT_FAMILY = "unitary_pqc_measured_1"
PARAMETERS_PER_LAYER = 14
HESSIAN_RANK_DEFINITION = "count(abs(eigenvalue) >= hessian_rank_threshold)"
HESSIAN_CONDITION_NUMBER_DEFINITION = (
    "max(abs(active eigenvalue)) / min(abs(active eigenvalue)); NaN if rank == 0"
)
_REQUIRED_METADATA = {
    "schema_version", "analysis_kind", "ansatz", "measurement_outcome",
    "h_param", "layers", "num_hessian_samples", "hessian_sample_seed_base",
    "hessian_rank_threshold", "hessian_rank_definition",
    "hessian_condition_number_definition", "num_params_per_layer",
    "analysis_batch_size",
}


def _scalar(values, name):
    value = np.asarray(values)
    if value.shape != ():
        raise ValueError(f"Hessian {name} must be scalar.")
    return value


def _finite_real(values, name, *, positive=False):
    value = _scalar(values, name)
    if (
        not np.issubdtype(value.dtype, np.number)
        or np.iscomplexobj(value) or not np.isfinite(value)
    ):
        raise ValueError(f"Hessian {name} must be a finite real number.")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"Hessian {name} must be finite and positive.")
    return result


def _integer(values, name, *, minimum=1, require_integer_dtype=False):
    value = _scalar(values, name)
    if require_integer_dtype and not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"Hessian {name} must use an integer dtype.")
    number = _finite_real(value, name)
    result = int(value.item())
    if result != number or result < minimum:
        raise ValueError(f"Hessian {name} must be an integer >= {minimum}.")
    return result


def _layers(values, name="layers"):
    values = np.asarray(values)
    if (
        values.ndim != 1 or values.size == 0
        or not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values) or not np.all(np.isfinite(values))
        or np.any(values <= 0) or np.any(values != np.rint(values))
        or np.unique(values).size != values.size
    ):
        raise ValueError(f"Hessian {name} must be unique positive integers.")
    return sorted(int(layer) for layer in values)


def _saved_summaries(archive, layers, num_samples):
    """Validate summaries together, at the archive's original threshold."""
    rank_by_layer, condition_by_layer = {}, {}
    for layer in layers:
        rank_key, condition_key = f"L{layer}_rank", f"L{layer}_condition_number"
        missing = [key for key in (rank_key, condition_key) if key not in archive]
        if missing:
            raise KeyError(f"Hessian archive is missing: {', '.join(missing)}.")
        ranks = np.asarray(archive[rank_key])
        if (
            ranks.shape != (num_samples,)
            or not np.issubdtype(ranks.dtype, np.number)
            or np.iscomplexobj(ranks) or not np.all(np.isfinite(ranks))
            or np.any(ranks != np.rint(ranks)) or np.any(ranks < 0)
            or np.any(ranks > PARAMETERS_PER_LAYER * layer)
        ):
            raise ValueError(f"Invalid Hessian rank samples in {rank_key}.")
        ranks = ranks.astype(np.int64)
        conditions = np.asarray(archive[condition_key])
        if (
            conditions.shape != (num_samples,)
            or not np.issubdtype(conditions.dtype, np.number)
            or np.iscomplexobj(conditions) or np.any(np.isinf(conditions))
        ):
            raise ValueError(f"Invalid Hessian condition-number samples in {condition_key}.")
        conditions = conditions.astype(np.float64, copy=False)
        finite = np.isfinite(conditions)
        if not np.array_equal(finite, ranks > 0):
            raise ValueError(
                f"Hessian condition number must be finite iff rank is positive at L={layer}."
            )
        if np.any(conditions[finite] < 1.0 - 1e-12):
            raise ValueError(f"Hessian condition numbers below one in {condition_key}.")
        rank_by_layer[layer] = ranks
        condition_by_layer[layer] = conditions
    return rank_by_layer, condition_by_layer


def load_measured_unitary_hessian_result(
    results_dir,
    *,
    expected_h_param,
    expected_layers=None,
    expected_num_samples=None,
    expected_seed_base=None,
    rank_threshold=1e-12,
):
    """Load only measured-outcome-1 unitary random-point Hessian results.

    Optional expected layers/sample count/seed verify correspondence with a
    loaded QFIM archive. Layer order is normalized, but the layer sets must
    match exactly. Raw matrices are diagonalized once per layer; their signed
    eigenvalues are retained for reuse by all figures. Rank counts absolute
    eigenvalues >= the requested threshold; condition is the largest/smallest
    active magnitude, or NaN for rank zero. Legacy summary-only archives can
    be loaded only at their saved threshold and return empty matrix/spectrum
    mappings. The saved summaries are validated before deriving new metrics.
    """
    threshold = _finite_real(rank_threshold, "rank_threshold", positive=True)
    expected_h = _finite_real(expected_h_param, "expected_h_param")
    path = Path(results_dir) / RESULT_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Measured-outcome-1 Unitary-PQC Hessian result was not found: {path}. "
            "Run unitary_pqc_measured_1_overparam_hessian.py "
            f"--h-param {expected_h:g} before visualizing."
        )

    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(_REQUIRED_METADATA.difference(archive.files))
        if missing:
            raise KeyError(f"Hessian archive {path} is missing: {', '.join(missing)}.")
        schema = _integer(archive["schema_version"], "schema_version")
        if schema != 1:
            raise ValueError(f"Unsupported Hessian schema_version {schema}; expected 1.")
        for key, expected in (
            ("analysis_kind", "random_points"),
            ("ansatz", OUTPUT_FAMILY),
            ("hessian_rank_definition", HESSIAN_RANK_DEFINITION),
            ("hessian_condition_number_definition", HESSIAN_CONDITION_NUMBER_DEFINITION),
        ):
            actual = _scalar(archive[key], key).item()
            if actual != expected:
                raise ValueError(f"Hessian archive {key} mismatch: {actual!r} != {expected!r}.")
        # Additive family metadata must not contradict the required ansatz.
        if "output_family" in archive:
            if _scalar(archive["output_family"], "output_family").item() != OUTPUT_FAMILY:
                raise ValueError("Hessian archive output_family mismatch.")
        outcome = _integer(
            archive["measurement_outcome"], "measurement_outcome",
            minimum=0, require_integer_dtype=True,
        )
        if outcome != 1:
            raise ValueError("Hessian measurement_outcome must be 1.")
        parameters = _integer(
            archive["num_params_per_layer"], "num_params_per_layer",
            require_integer_dtype=True,
        )
        if parameters != PARAMETERS_PER_LAYER:
            raise ValueError(f"Hessian num_params_per_layer must be {PARAMETERS_PER_LAYER}.")
        archived_h = _finite_real(archive["h_param"], "h_param")
        if archived_h != expected_h:
            raise ValueError(f"Hessian archive h_param mismatch: {archived_h} != {expected_h}.")
        layers = _layers(archive["layers"])
        if expected_layers is not None and layers != _layers(expected_layers, "expected_layers"):
            raise ValueError("Hessian layers do not match the random-point QFIM layers.")
        num_samples = _integer(archive["num_hessian_samples"], "num_hessian_samples")
        seed_base = _integer(archive["hessian_sample_seed_base"], "hessian_sample_seed_base", minimum=0)
        _integer(archive["analysis_batch_size"], "analysis_batch_size")
        for actual, expected, name, minimum in (
            (num_samples, expected_num_samples, "sample count", 1),
            (seed_base, expected_seed_base, "seed base", 0),
        ):
            if expected is not None and actual != _integer(expected, f"expected {name}", minimum=minimum):
                raise ValueError(f"Hessian and QFIM random-point {name}s differ: {actual} != {expected}.")
        saved_threshold = _finite_real(
            archive["hessian_rank_threshold"], "hessian_rank_threshold", positive=True,
        )
        rank_by_layer, condition_by_layer = _saved_summaries(archive, layers, num_samples)
        hessian_by_layer = load_optional_hessian_matrices(
            archive, layers, num_samples, PARAMETERS_PER_LAYER,
        )

    eigenvalues_by_layer = {}
    if not hessian_by_layer and threshold != saved_threshold:
        raise ValueError(
            "Legacy Hessian archive contains only rank/condition summaries at "
            f"threshold {saved_threshold:g}. Recompute with "
            "unitary_pqc_measured_1_overparam_hessian.py to save raw matrices "
            "before changing the analysis threshold."
        )
    for layer, matrices in hessian_by_layer.items():
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
        rank_by_layer[layer] = ranks
        condition_by_layer[layer] = conditions
        eigenvalues_by_layer[layer] = eigenvalues

    return {
        "path": path, "output_family": OUTPUT_FAMILY, "layers": layers,
        "num_samples": num_samples, "seed_base": seed_base, "threshold": threshold,
        "rank_by_layer": rank_by_layer, "condition_by_layer": condition_by_layer,
        "hessian_by_layer": hessian_by_layer, "eigenvalues_by_layer": eigenvalues_by_layer,
    }
