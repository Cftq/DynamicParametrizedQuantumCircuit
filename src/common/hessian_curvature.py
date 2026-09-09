"""Energy-width-normalized curvature diagnostics from saved Hessian matrices.

All four metrics use the entire signed spectrum, without a rank threshold.
The numerical routines require only NumPy; Matplotlib is imported for plotting.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


METRIC_LABELS = {
    "diagonal_square_sum": r"Diagonal curvature square sum $D$",
    "curvature_square_sum": r"Total curvature square sum $S$",
    "curvature_effective_rank": r"Curvature effective rank $r_{\mathrm{curv}}$",
    "negative_curvature_fraction": r"Negative curvature fraction $\nu_{-}$",
}


def _positive_width(value):
    width = float(value)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("Hamiltonian energy width must be finite and positive.")
    return width


def _hessian_array(values):
    matrices = np.asarray(values)
    if (
        matrices.ndim != 3 or matrices.shape[0] == 0
        or matrices.shape[1] == 0 or matrices.shape[1] != matrices.shape[2]
        or not np.issubdtype(matrices.dtype, np.number)
        or np.iscomplexobj(matrices) or not np.all(np.isfinite(matrices))
    ):
        raise ValueError("Hessians must be finite real arrays of shape (samples, P, P).")
    matrices = matrices.astype(np.float64, copy=False)
    # Check symmetry relative to each matrix's own magnitude. This also checks
    # tiny matrices, rather than accepting them under an absolute tolerance.
    scale = np.max(np.abs(matrices), axis=(1, 2))
    scaled = matrices / np.where(scale > 0.0, scale, 1.0)[:, None, None]
    if not np.allclose(scaled, scaled.swapaxes(-1, -2), rtol=1e-10, atol=1e-12):
        raise ValueError("Hessian matrices must be symmetric.")
    return matrices


def load_optional_hessian_matrices(result, layers, num_samples, parameters_per_layer):
    """Read additive raw-matrix fields in older unitary archive schemas.

    An archive containing only old rank/condition summaries returns an empty
    mapping. A partially populated or malformed matrix archive is an error.
    Family, h, and sampling metadata are checked by each existing caller.
    """
    layers = [int(layer) for layer in layers]
    keys = [f"L{layer}_hessian" for layer in layers]
    present = [key in result for key in keys]
    if not any(present):
        return {}
    if not all(present):
        missing = [key for key, exists in zip(keys, present) if not exists]
        raise KeyError(f"Hessian archive is missing raw matrices: {missing}.")
    output = {}
    for layer, key in zip(layers, keys):
        dimension = int(parameters_per_layer) * layer
        matrices = _hessian_array(result[key])
        if matrices.shape != (int(num_samples), dimension, dimension):
            raise ValueError(f"Hessian matrix shape mismatch in {key}.")
        theta_key = f"L{layer}_theta"
        if theta_key in result:
            theta = np.asarray(result[theta_key])
            if (
                theta.shape != (int(num_samples), dimension)
                or not np.issubdtype(theta.dtype, np.number)
                or np.iscomplexobj(theta) or not np.all(np.isfinite(theta))
            ):
                raise ValueError(f"Invalid saved Hessian parameters in {theta_key}.")
        output[layer] = matrices
    return output


def energy_width_from_hamiltonian(hamiltonian_matrix):
    """Return lambda_max(H) - lambda_min(H), accepting H_S or H_S tensor I."""
    matrix = np.asarray(hamiltonian_matrix)
    if (
        matrix.ndim != 2 or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
        or not np.issubdtype(matrix.dtype, np.number)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix, matrix.conj().T, rtol=1e-12, atol=1e-14)
    ):
        raise ValueError("Hamiltonian must be a finite Hermitian square matrix.")
    eigenvalues = np.linalg.eigvalsh(matrix)
    return _positive_width(eigenvalues[-1] - eigenvalues[0])


def hamiltonian_energy_width(h_param):
    """Width of the project's four-qubit H_S(h), without importing JAX.

    This mirrors common/hamiltonian.py::hamiltonian_terms. The dense spectrum
    is used for arbitrary finite h, without assuming h lies in [0, 1].
    """
    h_param = float(h_param)
    if not math.isfinite(h_param):
        raise ValueError("h_param must be finite.")
    identity = np.eye(2)
    x = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])

    def pauli_product(operators):
        matrix = np.ones((1, 1))
        for wire in range(4):
            matrix = np.kron(matrix, operators.get(wire, identity))
        return matrix

    matrix = np.zeros((16, 16))
    for u, v in ((0, 1), (0, 2), (1, 3), (2, 3)):
        matrix -= (1.0 - h_param) * pauli_product({u: z, v: z})
    matrix -= (1.0 - h_param) * pauli_product(dict.fromkeys(range(4), x))
    for wire in range(4):
        matrix -= h_param * pauli_product({wire: z})
    return energy_width_from_hamiltonian(matrix)


def curvature_metrics(hessian_samples, energy_width, *, eigenvalues=None):
    """Compute D, S, r_curv, and nu_minus separately for each saved point.

    D and S use the matrix divided by Omega. Scale-free spectral ratios are
    evaluated after rescaling, avoiding underflow in fourth powers. Exactly
    zero matrices give D=S=0 and r_curv=nu_minus=NaN; tiny nonzero matrices are
    not thresholded away. Optional eigenvalues are the full signed RAW spectra.
    """
    matrices = _hessian_array(hessian_samples)
    width = _positive_width(energy_width)
    normalized = matrices / width
    if not np.all(np.isfinite(normalized)):
        raise FloatingPointError("Normalized Hessians exceed float64 range.")
    diagonal_square_sum = np.sum(
        np.square(np.diagonal(normalized, axis1=1, axis2=2)), axis=1,
    )
    curvature_square_sum = np.sum(np.square(normalized), axis=(1, 2))

    scales = np.max(np.abs(matrices), axis=(1, 2))
    nonzero = scales > 0.0
    safe_scales = np.where(nonzero, scales, 1.0)
    if eigenvalues is None:
        spectrum = np.linalg.eigvalsh(matrices / safe_scales[:, None, None])
    else:
        spectrum = np.asarray(eigenvalues)
        if (
            spectrum.shape != matrices.shape[:2]
            or not np.issubdtype(spectrum.dtype, np.number)
            or np.iscomplexobj(spectrum) or not np.all(np.isfinite(spectrum))
        ):
            raise ValueError("Raw Hessian eigenvalues must have shape (samples, P).")
        spectrum = spectrum / safe_scales[:, None]
    if not np.all(np.isfinite(spectrum)):
        raise FloatingPointError("Non-finite Hessian spectrum.")
    squared = np.square(spectrum)
    second_moment = np.sum(squared, axis=1)
    fourth_moment = np.sum(np.square(squared), axis=1)
    effective_rank = np.full(matrices.shape[0], np.nan)
    negative_fraction = np.full(matrices.shape[0], np.nan)
    if np.any(nonzero & (second_moment == 0.0)):
        raise ValueError("Nonzero Hessian has an empty supplied spectrum.")
    np.divide(
        np.square(second_moment), fourth_moment,
        out=effective_rank, where=nonzero,
    )
    np.divide(
        np.sum(np.where(spectrum < 0.0, squared, 0.0), axis=1),
        second_moment, out=negative_fraction, where=nonzero,
    )
    return {
        "diagonal_square_sum": diagonal_square_sum,
        "curvature_square_sum": curvature_square_sum,
        "curvature_effective_rank": effective_rank,
        "negative_curvature_fraction": negative_fraction,
    }


def _statistics(values):
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    sem = np.std(finite, ddof=1) / np.sqrt(finite.size) if finite.size > 1 else 0.0
    return (np.mean(finite), sem, np.min(finite), np.max(finite), finite.size)


def save_hessian_curvature_figures(
    hessians_by_layer,
    *,
    h_param,
    figures_dir,
    hamiltonian_matrix=None,
    eigenvalues_by_layer=None,
):
    """Save four layerwise mean +/- SEM/min/max PDFs and their sampled values.

    The input mapping determines P independently for each model and layer.
    Undefined zero-curvature ratios are excluded from statistics; an empty
    layer remains a gap, and an entirely undefined figure is annotated.
    """
    import matplotlib.pyplot as plt

    if not hessians_by_layer:
        raise ValueError("Raw Hessian matrices are required for curvature figures.")
    layers = sorted(hessians_by_layer)
    if any(not isinstance(layer, (int, np.integer)) or layer < 1 for layer in layers):
        raise ValueError("Hessian layer keys must be positive integers.")
    h_param = float(h_param)
    if not math.isfinite(h_param):
        raise ValueError("h_param must be finite.")
    width = (
        hamiltonian_energy_width(h_param)
        if hamiltonian_matrix is None
        else energy_width_from_hamiltonian(hamiltonian_matrix)
    )
    metrics_by_layer = {
        layer: curvature_metrics(
            hessians_by_layer[layer], width,
            eigenvalues=(
                None if eigenvalues_by_layer is None
                else eigenvalues_by_layer[layer]
            ),
        )
        for layer in layers
    }
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "h_param": np.asarray(h_param),
        "energy_width": np.asarray(width),
        "layers": np.asarray(layers, dtype=np.int64),
        "parameter_counts": np.asarray([np.shape(hessians_by_layer[L])[1] for L in layers]),
        "sample_counts": np.asarray([np.shape(hessians_by_layer[L])[0] for L in layers]),
        "normalization": np.asarray("raw energy Hessian / (lambda_max(H_S) - lambda_min(H_S))"),
        "spectrum_policy": np.asarray("full signed spectrum; no eigenvalue threshold"),
        "zero_curvature_policy": np.asarray("D=S=0; r_curv and nu_minus are NaN"),
    }
    figure_paths = {}
    for key, label in METRIC_LABELS.items():
        stats = np.asarray([_statistics(metrics_by_layer[L][key]) for L in layers])
        mean, sem, minimum, maximum, count = stats.T
        for layer in layers:
            arrays[f"L{layer}_{key}"] = metrics_by_layer[layer][key]
        for suffix, values in zip(
            ("mean", "sem", "minimum", "maximum", "finite_sample_count"), stats.T,
        ):
            arrays[f"{key}_{suffix}"] = values

        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        if np.any(count > 0):
            ax.plot(layers, maximum, "^--", color="C3", label="Maximum", lw=1.2)
            ax.errorbar(
                layers, mean, yerr=sem, fmt="o-", color="C0",
                capsize=3, label=r"Mean $\pm$ SEM", zorder=3,
            )
            ax.plot(layers, minimum, "v--", color="C2", label="Minimum", lw=1.2)
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "Undefined: all sampled Hessians are zero",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Number of Layers")
        ax.set_ylabel(label)
        ax.set_xticks(layers)
        ax.set_title(rf"Energy-width-normalized Hessian ($h={h_param:g}$, $\Omega={width:.5g}$)")
        if key == "negative_curvature_fraction":
            ax.set_ylim(-0.03, 1.03)
        elif key == "curvature_effective_rank":
            ax.set_ylim(bottom=0.0)
        else:
            ax.set_ylim(bottom=0.0)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        destination = figures_dir / f"hessian_{key}_random_points.pdf"
        fig.savefig(destination, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        figure_paths[key] = destination

    data_path = figures_dir / "hessian_curvature_random_points.npz"
    np.savez_compressed(data_path, **arrays)
    print(f"Saved four normalized Hessian curvature figures to: {figures_dir}", flush=True)
    return {
        "energy_width": width, "metrics_by_layer": metrics_by_layer,
        "figure_paths": figure_paths, "data_path": data_path,
    }
