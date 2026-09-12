"""NumPy-only spectral statistics for plots of saved Hessian eigenvalues.

Hessian eigenvalues remain signed for traces. Rank, participation rank and
entropy use their absolute values so that negative curvature contributes.
"""

from __future__ import annotations

import numpy as np


DEFAULT_COUNT_THRESHOLDS = (10.0, 5.0, 1.0, 0.1, 0.01, 0.001, 0.0001)


def _real_array(values, name):
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numbers.")
    with np.errstate(over="ignore", invalid="ignore"):
        return array.astype(np.float64, copy=False)


def _positive_thresholds(values, *, scalar):
    array = _real_array(values, "Thresholds")
    if (
        array.ndim != (0 if scalar else 1)
        or array.size == 0
        or not np.all(np.isfinite(array))
        or np.any(array <= 0.0)
    ):
        kind = "a scalar" if scalar else "a nonempty one-dimensional array"
        raise ValueError(f"Thresholds must be {kind} of finite positive values.")
    return array


def _scaled_rows(values):
    scale = np.max(np.abs(values), axis=1)
    scaled = np.divide(
        values, scale[:, None], out=np.zeros_like(values),
        where=scale[:, None] > 0.0,
    )
    return scaled, scale


def _row_sum(values):
    # Scaling also avoids an intermediate overflow for cancelling signed sums.
    scaled, scale = _scaled_rows(values)
    with np.errstate(over="ignore"):
        return np.sum(scaled, axis=1) * scale


def hessian_spectral_statistics(
    eigenvalues,
    *,
    threshold=1e-12,
    count_thresholds=DEFAULT_COUNT_THRESHOLDS,
):
    """Return per-sample statistics for a finite real ``(samples, P)`` array.

    ``rank``, ``active_signed_trace``, ``absolute_trace`` and
    ``shannon_entropy`` include eigenvalues with ``abs(lambda) >= threshold``.
    ``participation_rank`` uses the strict ``abs(lambda) > threshold`` mask
    used by the QFIM participation calculation, with formula
    ``(sum abs(lambda))**2 / sum abs(lambda)**2``. ``trace`` always includes
    the full signed spectrum, including eigenvalues below the threshold.

    Entropy uses normalized absolute eigenvalues and natural logarithms.
    Dimensionless statistics are computed after scaling to avoid overflow or
    underflow of squared eigenvalues or their normalization sums. Empty active
    spectra give zero. A trace whose magnitude exceeds float64 range can be
    infinite, even when all input eigenvalues are finite.

    Every result has shape ``(samples,)``, except ``eigcounts`` whose shape is
    ``(samples, len(count_thresholds))``. Counts use inclusive absolute-value
    thresholds in the supplied order. All thresholds must be finite positive.
    """
    values = _real_array(eigenvalues, "Eigenvalues")
    if (
        values.ndim != 2 or 0 in values.shape
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Eigenvalues must be finite real arrays of shape (samples, P).")
    threshold = _positive_thresholds(threshold, scalar=True).item()
    count_thresholds = _positive_thresholds(count_thresholds, scalar=False)

    absolute = np.abs(values)
    active = absolute >= threshold
    weights = np.where(active, absolute, 0.0)
    scaled_weights, _ = _scaled_rows(weights)
    weight_sum = np.sum(scaled_weights, axis=1)
    probabilities = np.divide(
        scaled_weights, weight_sum[:, None], out=np.zeros_like(weights),
        where=weight_sum[:, None] > 0.0,
    )
    log_probabilities = np.zeros_like(probabilities)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0.0)
    entropy = -np.sum(probabilities * log_probabilities, axis=1)

    participation_weights = np.where(absolute > threshold, absolute, 0.0)
    scaled_participation, _ = _scaled_rows(participation_weights)
    participation_sum = np.sum(scaled_participation, axis=1)
    participation_square_sum = np.sum(scaled_participation**2, axis=1)
    participation_rank = np.divide(
        participation_sum**2, participation_square_sum,
        out=np.zeros_like(participation_sum), where=participation_square_sum > 0.0,
    )

    return {
        "rank": np.count_nonzero(active, axis=1),
        "participation_rank": participation_rank,
        "trace": _row_sum(values),
        "active_signed_trace": _row_sum(np.where(active, values, 0.0)),
        "absolute_trace": _row_sum(weights),
        "shannon_entropy": entropy,
        "eigcounts": np.column_stack([
            np.count_nonzero(absolute >= value, axis=1)
            for value in count_thresholds
        ]),
    }


def finite_sample_statistics(samples):
    """Return ``(mean, SEM, minimum, maximum, count)`` of finite samples.

    Real input arrays are flattened. SEM uses the sample standard deviation
    (``ddof=1``) and is zero for a singleton. With no finite observations the
    four statistics are NaN and the count is zero. Scaling before computing
    moments prevents overflow and underflow for extreme finite samples.
    """
    values = _real_array(samples, "Samples").ravel()
    values = values[np.isfinite(values)]
    count = int(values.size)
    if count == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    scale = np.max(np.abs(values))
    scaled = values / scale if scale > 0.0 else np.zeros_like(values)
    mean = float(np.mean(scaled) * scale)
    sem = float((np.std(scaled, ddof=1) / np.sqrt(count)) * scale) if count > 1 else 0.0
    return mean, sem, float(np.min(values)), float(np.max(values)), count
