#!/usr/bin/env python
# coding: utf-8
"""Visualize saved measurement-outcome-1 Unitary-PQC numerical results.

Train once with ``unitary_pqc_measured_1_overparam_vqe.py`` when needed, then
calculate QFIM/HS and Hessian with their separate programs (or the default
``unitary_pqc_measured_1_overparam_compute.py`` analysis stage). This script loads
saved .npz results under
``figs/unitary_pqc_measured_1/u3_cartan/h_<h_param>/numerical_results`` and generates
numerical figures without recomputing VQE or QFIM quantities. Circuit drawings
are handled independently by
``unitary_pqc_measured_1_overparam_draw_circuits.py``.
QFIM eigenvalue, Trace, and spectral-Shannon-entropy figures use the canonical
unmasked spectra. Trace sums finite eigenvalues satisfying the inclusive fixed
rank cutoff; entropy normalizes that active spectrum and uses the natural log.
Saved Hessian matrices also yield energy-width-normalized diagonal and total
squared curvature, curvature effective rank, and negative-curvature fraction.
These four quantities use all eigenvalues, without an effective-rank cutoff.
Hessian output also includes the same statistics as reset-DPQC: participation
rank, absolute spectral sum, signed trace, spectral Shannon entropy, matrix
absolute-entry sum, threshold-count overlays, and signed/absolute per-layer
spectra. Scalar summaries show mean +/- SEM, minimum, and maximum. Magnitude
spectra define participation rank (strict cutoff) and entropy (inclusive
cutoff); the signed trace uses all eigenvalues. Sampled values and definitions
are saved in ``figures/hessian/hessian_statistics_random_points.npz``.
Use ``--hessian-only`` to render a saved Hessian archive independently of
VQE/QFIM results, without importing JAX, Optax, or the compute program.
Use ``--gap-normalized-only`` to beeswarm-plot saved final (E-E0)/Delta and
plot success probabilities at epsilon = 1e-5 through 1e-10 using only the
VQE energy archive.
These figures are also included in normal visualization. No training or QFIM
is recomputed; the gap is to the first excitation above the ground subspace.
Use ``--qfim-logdet-only`` to plot mean log det(I + kappa F) from saved
random-point QFIM spectra only. All eigenvalues are used without a rank cutoff,
and ``--qfim-logdet-kappa`` sets the positive scale (default: 1).
Random-point QFIM rank figures count all eigenvalues at or above the selected
threshold (default: QFIM_EFFECTIVE_RANK_THRESHOLD) and show mean +/- SEM,
minimum, and maximum. Use ``--qfim-rank-only`` to render them independently
of VQE, HS, Hessian, and optimization-path results from the saved raw spectra.

    python src/unitary_pqc/unitary_pqc_measured_1_overparam_visualize.py --h-param 0.1
    python src/unitary_pqc/unitary_pqc_measured_1_overparam_visualize.py --h-param 0.1 --hessian-only
    python src/unitary_pqc/unitary_pqc_measured_1_overparam_visualize.py --h-param 0.1 --gap-normalized-only
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Sequence

_MODULE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _MODULE_DIR.parent
_COMMON_DIR = _SRC_DIR / "common"
for _path in (_MODULE_DIR, _COMMON_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


import config_overparam as cfg
from unitary_pqc_measured_1_model import ANSATZ_NAME, NUM_PARAMS_PER_LAYER, OUTPUT_VARIANT


def _finite_float(value: str) -> float:
    """Parse one finite floating-point command-line value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError("value must be a finite number") from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be a finite number")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Visualize saved measurement-outcome-1 Unitary-PQC results for "
            "one Hamiltonian parameter h."
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
    parser.add_argument(
        "--convergence-tolerance",
        dest="convergence_tolerances",
        action="append",
        type=_positive_float,
        default=None,
        metavar="DELTA",
        help=(
            "Positive absolute-energy tolerance used for first-passage "
            "convergence figures. Repeat the option for multiple values "
            "(default: 1.0)."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--hessian-only", action="store_true",
        help="Render only saved random-point Hessian statistics; no recomputation.",
    )
    mode.add_argument(
        "--gap-normalized-only", action="store_true",
        help=(
            "Render final (E-E0)/gap and strict success probabilities from "
            "saved VQE energies only; no training, QFIM, or Hessian required."
        ),
    )
    mode.add_argument(
        "--qfim-logdet-only", action="store_true",
        help="Render mean log det(I + kappa F) from saved random QFIM spectra only.",
    )
    mode.add_argument(
        "--qfim-rank-only", action="store_true",
        help="Render QFIM rank mean/SEM/min/max from saved random QFIM spectra only.",
    )
    parser.add_argument(
        "--qfim-rank-threshold", type=_positive_float,
        default=float(cfg.QFIM_EFFECTIVE_RANK_THRESHOLD),
        help="Inclusive QFIM rank cutoff (default: QFIM_EFFECTIVE_RANK_THRESHOLD).",
    )
    parser.add_argument(
        "--qfim-rank-results-dir", type=Path, default=None,
        help="Directory containing canonical random QFIM archives for --qfim-rank-only.",
    )
    parser.add_argument(
        "--qfim-rank-figures-dir", type=Path, default=None,
        help="Optional PDF/statistics output directory for --qfim-rank-only.",
    )
    parser.add_argument(
        "--qfim-logdet-kappa", type=_positive_float, default=1.0,
        help="Positive finite kappa for the QFIM log-determinant (default: 1).",
    )
    parser.add_argument(
        "--qfim-logdet-results-dir", type=Path, default=None,
        help="Directory containing random-point QFIM archives for --qfim-logdet-only.",
    )
    parser.add_argument(
        "--qfim-logdet-figures-dir", type=Path, default=None,
        help="Optional PDF/statistics output directory for --qfim-logdet-only.",
    )
    parser.add_argument(
        "--gap-results-dir", type=Path, default=None,
        help="Optional VQE energy-results directory for --gap-normalized-only.",
    )
    parser.add_argument(
        "--gap-figures-dir", type=Path, default=None,
        help="Optional figure output directory for --gap-normalized-only.",
    )
    parser.add_argument(
        "--hessian-results-dir", type=Path, default=None,
        help="Optional numerical-results directory for --hessian-only.",
    )
    parser.add_argument(
        "--hessian-figures-dir", type=Path, default=None,
        help="Optional figure output directory for --hessian-only.",
    )
    args = parser.parse_args(argv)
    if not args.hessian_only and (
        args.hessian_results_dir is not None or args.hessian_figures_dir is not None
    ):
        parser.error("Hessian directory overrides require --hessian-only.")
    if not args.gap_normalized_only and (
        args.gap_results_dir is not None or args.gap_figures_dir is not None
    ):
        parser.error("Gap directory overrides require --gap-normalized-only.")
    if not args.qfim_logdet_only and (
        args.qfim_logdet_results_dir is not None
        or args.qfim_logdet_figures_dir is not None
    ):
        parser.error("QFIM logdet directory overrides require --qfim-logdet-only.")
    if not args.qfim_rank_only and (
        args.qfim_rank_results_dir is not None
        or args.qfim_rank_figures_dir is not None
    ):
        parser.error("QFIM rank directory overrides require --qfim-rank-only.")
    return args


def _render_unitary_hessian_result(
    result, *, h_param, figures_dir, hamiltonian_matrix=None,
):
    """Render a validated unitary archive with the reset-DPQC definitions."""
    from hessian_plot import (
        _threshold_tex, plot_hessian_summary, save_hessian_statistic_figures,
    )
    from hessian_curvature import save_hessian_curvature_figures

    figures_dir = Path(figures_dir)
    threshold_tex = _threshold_tex(result["threshold"])
    figure_paths = {}
    for key, label, integer_ticks in (
        ("rank", "Hessian rank", True),
        ("condition", "Hessian condition number", False),
    ):
        filename = "rank" if key == "rank" else "condition_number"
        label_separator = " " if integer_ticks else "\n"
        figure_paths[key] = plot_hessian_summary(
            result[f"{key}_by_layer"], result["layers"],
            ylabel=rf"{label}{label_separator}($|\lambda_i| \geq {threshold_tex}$)",
            title=f"{label} at {result['num_samples']} random parameter points",
            outpath=figures_dir / f"hessian_{filename}_random_points.pdf",
            lower_bound_zero=integer_ticks, integer_ticks=integer_ticks,
            empty_message=(
                None if integer_ticks else
                "Condition number undefined\n(no active Hessian eigenvalues)"
            ),
        )
    statistics, curvature = None, None
    if result["hessian_by_layer"]:
        statistics = save_hessian_statistic_figures(
            result, figures_dir=figures_dir,
            count_thresholds=cfg.QFIM_PATH_EIGCOUNT_THRESHOLDS, h_param=h_param,
        )
        curvature = save_hessian_curvature_figures(
            result["hessian_by_layer"], h_param=h_param, figures_dir=figures_dir,
            hamiltonian_matrix=hamiltonian_matrix,
            eigenvalues_by_layer=result["eigenvalues_by_layer"],
        )
        figure_paths.update(statistics["figure_paths"])
        figure_paths.update(curvature["figure_paths"])
    else:
        warnings.warn(
            "The saved Hessian archive contains only rank/condition summaries; "
            "spectrum-dependent statistics and curvature figures require raw "
            "matrices. Refresh the archive with "
            "unitary_pqc_measured_1_overparam_hessian.py "
            f"--h-param {h_param}.",
            RuntimeWarning, stacklevel=2,
        )
    return {"figure_paths": figure_paths, "statistics": statistics, "curvature": curvature}


def run_unitary_hessian_visualization(
    *, h_param=None, results_dir=None, figures_dir=None,
):
    """Render saved outcome-1 Hessians independently of other result archives."""
    from unitary_hessian_results import load_measured_unitary_hessian_result

    selected_h = _finite_float(str(cfg.H_PARAM if h_param is None else h_param))
    save_dir = (
        _SRC_DIR.parent / "figs" / ANSATZ_NAME / OUTPUT_VARIANT / f"h_{selected_h}"
    )
    results_dir = (
        save_dir / "numerical_results" / "hessian" if results_dir is None
        else Path(results_dir).expanduser().resolve()
    )
    figures_dir = (
        save_dir / "figures" / "hessian" if figures_dir is None
        else Path(figures_dir).expanduser().resolve()
    )
    result = load_measured_unitary_hessian_result(
        results_dir, expected_h_param=selected_h,
        rank_threshold=float(cfg.QFIM_EFFECTIVE_RANK_THRESHOLD),
    )
    output = _render_unitary_hessian_result(
        result, h_param=selected_h, figures_dir=figures_dir,
    )
    output.update(h_param=selected_h, save_dir=save_dir, hessian_fig_dir=figures_dir)
    return output


def run_unitary_gap_normalized_visualization(
    *, h_param=None, results_dir=None, figures_dir=None,
):
    """Plot archived outcome-1 energies without importing a quantum runtime."""
    import numpy as np
    from gap_normalized_energy import (
        save_gap_normalized_energy_outputs, spectral_gap_from_hamiltonian,
    )
    from hessian_curvature import hamiltonian_matrix_numpy
    from unitary_pqc_measured_1_energy_results import load_measured_unitary_final_energies

    selected_h = _finite_float(str(cfg.H_PARAM if h_param is None else h_param))
    save_dir = (
        _SRC_DIR.parent / "figs" / ANSATZ_NAME / OUTPUT_VARIANT / f"h_{selected_h}"
    )
    results_dir = (
        save_dir / "numerical_results" / "energy" if results_dir is None
        else Path(results_dir).expanduser().resolve()
    )
    figures_dir = (
        save_dir / "figures" / "energy" if figures_dir is None
        else Path(figures_dir).expanduser().resolve()
    )
    energies, saved_ground, metadata = load_measured_unitary_final_energies(
        results_dir / "vqe_optimization_results.npz", expected_h_param=selected_h,
    )
    matrix = hamiltonian_matrix_numpy(selected_h)
    spectrum = spectral_gap_from_hamiltonian(matrix)
    ground_tolerance = (
        128 * np.finfo(np.float64).eps * max(1.0, np.linalg.norm(matrix, ord=2))
    )
    if abs(saved_ground - spectrum["ground_energy"]) > ground_tolerance:
        raise ValueError("Saved VQE ground energy does not match the selected Hamiltonian.")
    metadata.update(output_family=ANSATZ_NAME, saved_ground_energy=saved_ground)
    result = save_gap_normalized_energy_outputs(
        energies,
        h_param=selected_h,
        hamiltonian_matrix=matrix,
        figures_dir=figures_dir,
        statistics_outpath=results_dir / "gap_normalized_energy_statistics.npz",
        metadata=metadata,
    )
    result.update(save_dir=save_dir, energy_fig_dir=figures_dir)
    print(f"Spectral gap above the ground subspace: {result['spectral_gap']:.12g}")
    return result


def _load_unitary_random_qfim_spectra(*, h_param, results_dir):
    """Validate both canonical outcome-1 spectra without quantum dependencies."""
    import numpy as np
    from qfim_logdet import load_random_qfim_spectra

    loaded = {}
    for keep_key, keep_wires, representation in (
        ("keep0123", (0, 1, 2, 3), "reduced_mixed"),
        ("keep01234", (0, 1, 2, 3, 4), "pure_full"),
    ):
        loaded[keep_key] = load_random_qfim_spectra(
            Path(results_dir) / f"qfim_random_points_{keep_key}.npz",
            expected_h_param=h_param,
            parameters_per_layer=NUM_PARAMS_PER_LAYER,
            required_metadata={
                "schema_version": 1,
                "ansatz": ANSATZ_NAME,
                "measurement_outcome": 1,
                "num_total_qubits": 5,
                "num_params_per_layer": NUM_PARAMS_PER_LAYER,
                "analysis_kind": "random_points",
                "keep_key": keep_key,
                "keep_wires": np.asarray(keep_wires, dtype=np.int64),
                "traced_wires": np.asarray(
                    [wire for wire in range(5) if wire not in keep_wires], dtype=np.int64,
                ),
                "representation": representation,
                "qfim_definition": "SLD_QFIM",
                "eigenvalue_order": "descending",
                "eigenvalues_threshold_masked": False,
            },
        )
    return loaded


def run_unitary_qfim_logdet_visualization(
    *, h_param=None, results_dir=None, figures_dir=None, kappa=1.0,
):
    """Average saved sample log determinants without importing a quantum runtime.

    ``results_dir`` contains the canonical random-point QFIM archives directly.
    Layers and sample counts come from those archives, independent of current
    analysis configuration. No optimization-path samples or rank cutoff enter
    this expectation over the archived parameter samples.
    """
    from qfim_logdet import compute_qfim_logdet, save_qfim_logdet_outputs

    selected_h = _finite_float(str(cfg.H_PARAM if h_param is None else h_param))
    kappa = _positive_float(str(kappa))
    save_dir = (
        _SRC_DIR.parent / "figs" / ANSATZ_NAME / OUTPUT_VARIANT / f"h_{selected_h}"
    )
    results_dir = (
        save_dir / "numerical_results" / "qfim" if results_dir is None
        else Path(results_dir).expanduser().resolve()
    )
    figures_dir = (
        save_dir / "figures" / "qfim" / "logdet" if figures_dir is None
        else Path(figures_dir).expanduser().resolve()
    )
    archives = _load_unitary_random_qfim_spectra(
        h_param=selected_h, results_dir=results_dir,
    )
    loaded = {
        keep_key: (
            archive, compute_qfim_logdet(archive["eigenvalues_by_layer"], kappa=kappa),
        )
        for keep_key, archive in archives.items()
    }

    # Validate both states before writing any files.
    outputs = {}
    for keep_key, (archive, statistics) in loaded.items():
        metadata = dict(archive["metadata"])
        metadata["output_family"] = ANSATZ_NAME
        if "sampling_distribution" not in metadata:
            metadata["sampling_distribution"] = "independent_uniform[-pi,pi)"
            metadata["sampling_distribution_source"] = (
                "unitary_pqc_measured_1_overparam_qfim.py random-point sampler"
            )
        outputs[keep_key] = {
            "statistics": statistics,
            **save_qfim_logdet_outputs(
                statistics, figures_dir, keep_key=keep_key,
                title=f"Measured 1: {keep_key}, h={selected_h:g}",
                source_path=archive["source_path"], metadata=metadata,
            ),
        }
    print(f"Saved random-point QFIM logdet figures to: {figures_dir}")
    return {
        "h_param": selected_h, "kappa": kappa, "save_dir": save_dir,
        "qfim_logdet_fig_dir": figures_dir, "outputs": outputs,
    }


def run_unitary_qfim_rank_visualization(
    *, h_param=None, results_dir=None, figures_dir=None, rank_threshold=None,
):
    """Plot DPQC-style inclusive threshold rank from saved outcome-1 spectra."""
    from qfim_rank import compute_qfim_rank, save_qfim_rank_outputs

    selected_h = _finite_float(str(cfg.H_PARAM if h_param is None else h_param))
    threshold = _positive_float(str(
        cfg.QFIM_EFFECTIVE_RANK_THRESHOLD if rank_threshold is None else rank_threshold
    ))
    save_dir = (
        _SRC_DIR.parent / "figs" / ANSATZ_NAME / OUTPUT_VARIANT / f"h_{selected_h}"
    )
    results_dir = (
        save_dir / "numerical_results" / "qfim" if results_dir is None
        else Path(results_dir).expanduser().resolve()
    )
    figures_dir = (
        save_dir / "figures" / "qfim" / "rank" / "random_points"
        if figures_dir is None else Path(figures_dir).expanduser().resolve()
    )
    archives = _load_unitary_random_qfim_spectra(
        h_param=selected_h, results_dir=results_dir,
    )
    # Validate/compute both kept states before creating output files.
    statistics_by_keep = {
        keep_key: compute_qfim_rank(archive["eigenvalues_by_layer"], threshold=threshold)
        for keep_key, archive in archives.items()
    }
    outputs = {}
    for keep_key, archive in archives.items():
        statistics = statistics_by_keep[keep_key]
        metadata = dict(archive["metadata"])
        metadata["output_family"] = ANSATZ_NAME
        outputs[keep_key] = {
            "statistics": statistics,
            **save_qfim_rank_outputs(
                statistics, figures_dir, keep_key=keep_key,
                title=f"Measured 1 QFIM rank: {keep_key}, h={selected_h:g}",
                source_path=archive["source_path"], metadata=metadata,
            ),
        }
    print(f"Saved random-point QFIM rank figures to: {figures_dir}")
    return {
        "h_param": selected_h, "rank_threshold": threshold, "save_dir": save_dir,
        "qfim_rank_fig_dir": figures_dir, "outputs": outputs,
    }


if __name__ == "__main__":
    _CLI_ARGS = _parse_cli_args()
else:
    _CLI_ARGS = argparse.Namespace(
        h_param=float(cfg.H_PARAM),
        convergence_tolerances=None,
        qfim_logdet_kappa=1.0,
        qfim_rank_threshold=float(cfg.QFIM_EFFECTIVE_RANK_THRESHOLD),
    )

_SELECTED_H_PARAM = float(_CLI_ARGS.h_param)
if not math.isfinite(_SELECTED_H_PARAM):
    raise ValueError("h_param must be a finite number.")

# Exit before importing the compute module or any quantum/optimizer runtime.
if __name__ == "__main__" and _CLI_ARGS.qfim_rank_only:
    os.environ.setdefault("MPLBACKEND", "Agg")
    run_unitary_qfim_rank_visualization(
        h_param=_CLI_ARGS.h_param, results_dir=_CLI_ARGS.qfim_rank_results_dir,
        figures_dir=_CLI_ARGS.qfim_rank_figures_dir,
        rank_threshold=_CLI_ARGS.qfim_rank_threshold,
    )
    raise SystemExit(0)


if __name__ == "__main__" and _CLI_ARGS.qfim_logdet_only:
    os.environ.setdefault("MPLBACKEND", "Agg")
    run_unitary_qfim_logdet_visualization(
        h_param=_CLI_ARGS.h_param, results_dir=_CLI_ARGS.qfim_logdet_results_dir,
        figures_dir=_CLI_ARGS.qfim_logdet_figures_dir,
        kappa=_CLI_ARGS.qfim_logdet_kappa,
    )
    raise SystemExit(0)


if __name__ == "__main__" and _CLI_ARGS.gap_normalized_only:
    os.environ.setdefault("MPLBACKEND", "Agg")
    run_unitary_gap_normalized_visualization(
        h_param=_CLI_ARGS.h_param, results_dir=_CLI_ARGS.gap_results_dir,
        figures_dir=_CLI_ARGS.gap_figures_dir,
    )
    raise SystemExit(0)


if __name__ == "__main__" and _CLI_ARGS.hessian_only:
    os.environ.setdefault("MPLBACKEND", "Agg")
    _hessian_output = run_unitary_hessian_visualization(
        h_param=_CLI_ARGS.h_param, results_dir=_CLI_ARGS.hessian_results_dir,
        figures_dir=_CLI_ARGS.hessian_figures_dir,
    )
    print(f"Saved Hessian figures to: {_hessian_output['hessian_fig_dir']}")
    raise SystemExit(0)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from convergence_time import generate_convergence_time_outputs

if __package__:
    from . import unitary_pqc_measured_1_overparam_common as upqc
else:
    import unitary_pqc_measured_1_overparam_common as upqc


NP_REAL_DTYPE = np.float64
NP_INT_DTYPE = np.int64
ENERGY_ERROR_PLOT_EPS = NP_REAL_DTYPE(1e-12)
FINAL_ENERGY_ERROR_DETAIL_THRESHOLD = NP_REAL_DTYPE(
    getattr(cfg, "FINAL_ENERGY_ERROR_DETAIL_THRESHOLD", 6e-1)
)
SUCCESS_PROBABILITY_FIGURE_THRESHOLDS = np.asarray(
    # Plot-only tolerances, evaluated from saved final energies.
    (1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10),
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

QFIM_TRACE_EIGENVALUE_THRESHOLD = NP_REAL_DTYPE(
    cfg.QFIM_EFFECTIVE_RANK_THRESHOLD
)
QFIM_EIGENVALUE_PLOT_EPS = NP_REAL_DTYPE(cfg.QFIM_EIG_PLOT_EPS)
QFIM_KEEP_KEYS = ("keep0123", "keep01234")
QFIM_KEEP_LABELS = {
    "keep0123": "reduced keep=(0,1,2,3)",
    "keep01234": "pure full state keep=(0,1,2,3,4)",
}
HESSIAN_RANK_THRESHOLD = NP_REAL_DTYPE(upqc.QFIM_EFFECTIVE_RANK_THRESHOLD)


def _load_npz_result_unchecked(inpath: str) -> dict:
    """Load one numerical archive without importing simulator packages."""
    with np.load(inpath, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


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


def _validate_result_variant(
    result: dict,
    result_path,
    *,
    required: bool = False,
) -> None:
    """Ensure variant-tagged archives belong to measurement outcome 1."""
    keys = ("ansatz", "measurement_outcome", "num_params_per_layer")
    present = tuple(key for key in keys if key in result)
    if not present:
        if required:
            raise KeyError(
                "The required archive does not contain Unitary-PQC variant "
                f"metadata: {Path(result_path).resolve()}"
            )
        return
    missing = tuple(key for key in keys if key not in result)
    if missing:
        raise KeyError(
            "The archive contains incomplete Unitary-PQC variant metadata "
            f"({', '.join(missing)} missing): {Path(result_path).resolve()}"
        )

    ansatz = np.asarray(result["ansatz"])
    outcome = np.asarray(result["measurement_outcome"])
    params_per_layer = np.asarray(result["num_params_per_layer"])
    if ansatz.size != 1 or str(ansatz.reshape(-1)[0]) != upqc.ANSATZ_NAME:
        raise ValueError(
            "Saved ansatz does not match this visualizer: "
            f"{ansatz!r} != {upqc.ANSATZ_NAME!r} in "
            f"{Path(result_path).resolve()}"
        )
    if (
        outcome.size != 1
        or not np.issubdtype(outcome.dtype, np.integer)
        or int(outcome.reshape(-1)[0]) != upqc.MEASUREMENT_OUTCOME
    ):
        raise ValueError(
            "Saved measurement_outcome does not match this visualizer in "
            f"{Path(result_path).resolve()}"
        )
    if (
        params_per_layer.size != 1
        or not np.issubdtype(params_per_layer.dtype, np.integer)
        or int(params_per_layer.reshape(-1)[0]) != upqc.num_params_per_layer
    ):
        raise ValueError(
            "Saved num_params_per_layer does not match this visualizer in "
            f"{Path(result_path).resolve()}"
        )


def _load_required_result(
    path: str,
    *,
    expected_h_param: Optional[float] = None,
    require_h_param: bool = False,
    require_variant: bool = False,
) -> dict:
    """Load a compute-stage result or explain how to generate it."""
    result_path = Path(path).resolve()
    selected_h_param = float(
        upqc.h_param if expected_h_param is None else expected_h_param
    )
    if not result_path.is_file():
        stage = {
            "energy": "vqe", "qfim": "qfim", "hs": "qfim", "hessian": "hessian",
        }.get(result_path.parent.name, "compute")
        compute_script = _MODULE_DIR / f"unitary_pqc_measured_1_overparam_{stage}.py"
        raise FileNotFoundError(
            "Required Unitary-PQC numerical result is missing:\n"
            f"  {result_path}\n"
            "Generate this saved result with its dedicated program before "
            "visualizing (existing training results can be reused):\n"
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
    _validate_result_variant(
        result,
        result_path,
        required=require_variant,
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
    if np.isclose(threshold, 1.0, rtol=1e-12, atol=0.0):
        return r"$\delta=1.0$"

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
    only_threshold: Optional[float] = None,
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

    if only_threshold is not None:
        requested_threshold = float(only_threshold)
        matching = np.flatnonzero(
            np.isclose(
                thresholds,
                requested_threshold,
                rtol=1e-12,
                atol=0.0,
            )
        )
        if matching.size != 1:
            raise ValueError(
                "Expected exactly one success-probability threshold matching "
                f"{requested_threshold:g}; found {matching.size}."
            )
        thresholds = thresholds[matching]
        probabilities = probabilities[:, matching]

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
    if thresholds.size == 1:
        title = (
            "Success probability at "
            f"{_success_probability_threshold_label(thresholds[0])} "
            f"({num_trials} independent trials)"
        )
    else:
        title = (
            "Success probability at multiple accuracy levels "
            f"({num_trials} independent trials)"
        )
    upqc.set_prx_title(title, ax=ax)
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
    # Keep the separate delta=1 figure independent of the six tighter
    # tolerances used by success_probability_multiple_tolerances.pdf.
    delta_one_statistics = _multiple_tolerance_success_statistics(
        final_energies_by_layer,
        layers,
        ground_energy=ground_energy,
        num_trials=num_trials,
        thresholds=(1.0,),
    )
    _plot_success_probability_multiple_tolerances(
        delta_one_statistics,
        outpath=os.path.join(
            energy_dir,
            "success_probability_delta_1.pdf",
        ),
        only_threshold=1.0,
    )
    return success_statistics


def _qfim_threshold_tex(threshold: float) -> str:
    threshold = float(threshold)
    if threshold <= 0.0:
        return f"{threshold:g}"
    exponent = int(np.floor(np.log10(threshold)))
    mantissa = threshold / (10.0 ** exponent)
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
QFIM_SHANNON_ENTROPY_YLABEL = "QFIM spectral Shannon entropy (nats)"


def _required_archive_scalar(result: dict, key: str, result_path):
    if key not in result:
        raise KeyError(f"Missing {key!r} in {Path(result_path).resolve()}")
    value = np.asarray(result[key])
    if value.size != 1:
        raise ValueError(
            f"{key!r} must be scalar in {Path(result_path).resolve()}, "
            f"got shape {value.shape}."
        )
    return value.reshape(-1)[0]


def _validated_archive_layers(result: dict, result_path) -> list[int]:
    if "layers" not in result:
        raise KeyError(f"Missing 'layers' in {Path(result_path).resolve()}")
    raw_layers = np.asarray(result["layers"])
    if (
        raw_layers.ndim != 1
        or raw_layers.size == 0
        or not np.issubdtype(raw_layers.dtype, np.integer)
    ):
        raise ValueError(
            "QFIM archive layers must be a non-empty integer vector in "
            f"{Path(result_path).resolve()}, got {raw_layers.shape} / "
            f"{raw_layers.dtype}."
        )
    layers = [int(L) for L in raw_layers.tolist()]
    if any(L <= 0 for L in layers) or len(set(layers)) != len(layers):
        raise ValueError(
            "QFIM archive layers must be positive and unique in "
            f"{Path(result_path).resolve()}: {layers}."
        )
    return layers


def _validated_real_qfim_array(value, *, key: str, result_path) -> np.ndarray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise TypeError(
            f"{key!r} must be real numeric data in "
            f"{Path(result_path).resolve()}, got dtype={raw.dtype}."
        )
    return np.asarray(raw, dtype=NP_REAL_DTYPE)


def _validate_canonical_qfim_metadata(
    result: dict,
    result_path,
    *,
    keep_key: str,
    analysis_kind: str,
) -> tuple[list[int], int]:
    archived_keep = str(
        _required_archive_scalar(result, "keep_key", result_path)
    )
    if archived_keep != keep_key:
        raise ValueError(
            f"QFIM keep_key mismatch in {Path(result_path).resolve()}: "
            f"{archived_keep!r} != {keep_key!r}."
        )
    archived_kind = str(
        _required_archive_scalar(result, "analysis_kind", result_path)
    )
    if archived_kind != analysis_kind:
        raise ValueError(
            f"QFIM analysis_kind mismatch in {Path(result_path).resolve()}: "
            f"{archived_kind!r} != {analysis_kind!r}."
        )
    eigenvalue_order = str(
        _required_archive_scalar(result, "eigenvalue_order", result_path)
    )
    if eigenvalue_order != "descending":
        raise ValueError(
            "Canonical QFIM eigenvalues must be stored in descending order "
            f"in {Path(result_path).resolve()}."
        )
    threshold_masked = bool(
        _required_archive_scalar(
            result,
            "eigenvalues_threshold_masked",
            result_path,
        )
    )
    if threshold_masked:
        raise ValueError(
            "Canonical raw QFIM eigenvalues are required, but the archive "
            f"is threshold-masked: {Path(result_path).resolve()}."
        )
    archived_threshold = float(
        _required_archive_scalar(
            result,
            "qfim_effective_rank_threshold",
            result_path,
        )
    )
    if not math.isclose(
        archived_threshold,
        float(QFIM_TRACE_EIGENVALUE_THRESHOLD),
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError(
            "QFIM rank-threshold mismatch in "
            f"{Path(result_path).resolve()}: {archived_threshold} != "
            f"{float(QFIM_TRACE_EIGENVALUE_THRESHOLD)}."
        )
    params_per_layer = int(
        _required_archive_scalar(
            result,
            "num_params_per_layer",
            result_path,
        )
    )
    if params_per_layer <= 0 or params_per_layer != int(upqc.num_params_per_layer):
        raise ValueError(
            "QFIM num_params_per_layer mismatch in "
            f"{Path(result_path).resolve()}: {params_per_layer} != "
            f"{int(upqc.num_params_per_layer)}."
        )
    return _validated_archive_layers(result, result_path), params_per_layer


def _qfim_trace_at_or_above_rank_threshold(
    eigenvalues: np.ndarray,
) -> np.ndarray:
    """Threshold-sum a raw spectrum, returning NaN unless it is all finite."""
    eigs = _validated_real_qfim_array(
        eigenvalues,
        key="QFIM eigenvalues",
        result_path="<in-memory>",
    )
    if eigs.ndim == 0:
        raise ValueError("QFIM eigenvalues must have an eigenvalue axis.")
    finite_spectrum = np.all(np.isfinite(eigs), axis=-1)
    selected = eigs >= QFIM_TRACE_EIGENVALUE_THRESHOLD
    trace = np.sum(
        np.where(selected, eigs, NP_REAL_DTYPE(0.0)),
        axis=-1,
        dtype=NP_REAL_DTYPE,
    )
    return np.where(
        finite_spectrum,
        trace,
        NP_REAL_DTYPE(np.nan),
    )


def _qfim_spectral_shannon_entropy(
    eigenvalues: np.ndarray,
) -> np.ndarray:
    """Return active-spectrum ``-sum(p log p)`` in nats.

    The active eigenvalues are normalized by their sum. A zero active trace
    has entropy zero, while a non-finite input spectrum yields NaN.
    """
    eigs = _validated_real_qfim_array(
        eigenvalues,
        key="QFIM eigenvalues",
        result_path="<in-memory>",
    )
    if eigs.ndim == 0:
        raise ValueError("QFIM eigenvalues must have an eigenvalue axis.")

    finite_eigenvalues = np.isfinite(eigs)
    finite_spectrum = np.all(finite_eigenvalues, axis=-1)
    active_eigs = np.where(
        finite_eigenvalues & (eigs >= QFIM_TRACE_EIGENVALUE_THRESHOLD),
        eigs,
        NP_REAL_DTYPE(0.0),
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        trace = np.sum(active_eigs, axis=-1, dtype=NP_REAL_DTYPE)
        positive_trace = np.isfinite(trace) & (trace > NP_REAL_DTYPE(0.0))
        probabilities = np.divide(
            active_eigs,
            trace[..., None],
            out=np.zeros_like(active_eigs, dtype=NP_REAL_DTYPE),
            where=positive_trace[..., None],
        )
        positive_probability = probabilities > NP_REAL_DTYPE(0.0)
        entropy = -np.sum(
            np.where(
                positive_probability,
                probabilities
                * np.log(
                    np.where(
                        positive_probability,
                        probabilities,
                        NP_REAL_DTYPE(1.0),
                    )
                ),
                NP_REAL_DTYPE(0.0),
            ),
            axis=-1,
            dtype=NP_REAL_DTYPE,
        )
    finite_spectrum = finite_spectrum & np.isfinite(trace)
    entropy = np.maximum(entropy, NP_REAL_DTYPE(0.0))
    entropy = np.where(positive_trace, entropy, NP_REAL_DTYPE(0.0))
    return np.where(
        finite_spectrum,
        entropy,
        NP_REAL_DTYPE(np.nan),
    )


def _load_canonical_random_qfim_eigenvalues(
    keep_key: str,
    *,
    expected_layers,
    expected_num_samples: int,
) -> dict[int, np.ndarray]:
    result_path = os.path.join(
        upqc.qfim_results_dir,
        f"qfim_random_points_{keep_key}.npz",
    )
    result = _load_required_result(
        result_path,
        require_h_param=True,
        require_variant=True,
    )
    layers, params_per_layer = _validate_canonical_qfim_metadata(
        result,
        result_path,
        keep_key=keep_key,
        analysis_kind="random_points",
    )
    expected_layers = [int(L) for L in expected_layers]
    if layers != expected_layers:
        raise ValueError(
            f"Canonical random-point QFIM layers differ for {keep_key}: "
            f"{layers} != {expected_layers}."
        )
    num_samples = int(
        _required_archive_scalar(result, "num_qfim_samples", result_path)
    )
    if num_samples != int(expected_num_samples):
        raise ValueError(
            f"Canonical random-point QFIM sample count differs for "
            f"{keep_key}: {num_samples} != {int(expected_num_samples)}."
        )

    eigs_by_layer = {}
    for L in layers:
        key = f"L{L}_eigs_desc"
        if key not in result:
            raise KeyError(f"Missing {key!r} in {Path(result_path).resolve()}")
        eigs = _validated_real_qfim_array(
            result[key],
            key=key,
            result_path=result_path,
        )
        expected_shape = (num_samples, params_per_layer * L)
        if eigs.ndim != 2 or eigs.shape != expected_shape:
            raise ValueError(
                f"Canonical random-point QFIM shape mismatch for {key}: "
                f"expected {expected_shape}, got {eigs.shape}."
            )
        if not np.all(np.isfinite(eigs)):
            raise ValueError(
                f"Canonical random-point QFIM eigenvalues are not all "
                f"finite for {key} in {Path(result_path).resolve()}."
            )
        eigs_by_layer[L] = eigs
    return eigs_by_layer


def _load_canonical_qfim_eigenvalue_history(
    keep_key: str,
    *,
    expected_layers,
    expected_sample_iters,
    expected_num_runs: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    result_path = os.path.join(
        upqc.qfim_results_dir,
        f"qfim_eigs_history_optimization_path_{keep_key}.npz",
    )
    result = _load_required_result(
        result_path,
        require_h_param=True,
        require_variant=True,
    )
    layers, params_per_layer = _validate_canonical_qfim_metadata(
        result,
        result_path,
        keep_key=keep_key,
        analysis_kind="optimization_path",
    )
    expected_layers = [int(L) for L in expected_layers]
    if layers != expected_layers:
        raise ValueError(
            f"Canonical optimization-path QFIM layers differ for {keep_key}: "
            f"{layers} != {expected_layers}."
        )

    if "sample_iters" not in result:
        raise KeyError(f"Missing 'sample_iters' in {Path(result_path).resolve()}")
    raw_sample_iters = np.asarray(result["sample_iters"])
    if raw_sample_iters.ndim != 1 or not np.issubdtype(
        raw_sample_iters.dtype,
        np.integer,
    ):
        raise ValueError(
            "Canonical QFIM sample_iters must be a one-dimensional integer "
            f"array in {Path(result_path).resolve()}."
        )
    sample_iters = np.asarray(raw_sample_iters, dtype=NP_INT_DTYPE)
    expected_sample_iters = np.asarray(
        expected_sample_iters,
        dtype=NP_INT_DTYPE,
    )
    if not np.array_equal(sample_iters, expected_sample_iters):
        raise ValueError(
            f"Canonical optimization-path sample_iters differ for {keep_key}: "
            f"{sample_iters.tolist()} != {expected_sample_iters.tolist()}."
        )
    num_runs = int(_required_archive_scalar(result, "num_runs", result_path))
    if num_runs != int(expected_num_runs):
        raise ValueError(
            f"Canonical optimization-path run count differs for {keep_key}: "
            f"{num_runs} != {int(expected_num_runs)}."
        )

    eigs_by_layer = {}
    for L in layers:
        key = f"L{L}"
        if key not in result:
            raise KeyError(f"Missing {key!r} in {Path(result_path).resolve()}")
        eigs = _validated_real_qfim_array(
            result[key],
            key=key,
            result_path=result_path,
        )
        expected_shape = (
            num_runs,
            sample_iters.size,
            params_per_layer * L,
        )
        if eigs.ndim != 3 or eigs.shape != expected_shape:
            raise ValueError(
                f"Canonical optimization-path QFIM shape mismatch for {key}: "
                f"expected {expected_shape}, got {eigs.shape}."
            )
        if not np.all(np.isfinite(eigs)):
            raise ValueError(
                f"Canonical optimization-path QFIM eigenvalues are not all "
                f"finite for {key} in {Path(result_path).resolve()}."
            )
        eigs_by_layer[L] = eigs
    return eigs_by_layer, sample_iters


def _eigenvalue_index_ticks(n_params: int, max_ticks: int = 11) -> np.ndarray:
    n_params = int(n_params)
    if n_params <= 0:
        return np.asarray([], dtype=NP_INT_DTYPE)
    if n_params <= int(max_ticks):
        return np.arange(1, n_params + 1, dtype=NP_INT_DTYPE)
    ticks = np.rint(
        np.linspace(1, n_params, num=int(max_ticks))
    ).astype(NP_INT_DTYPE)
    ticks[0], ticks[-1] = 1, n_params
    return np.unique(ticks)


def _qfim_eigenvalues_for_log_plot(eigs: np.ndarray) -> np.ndarray:
    eigs = np.asarray(eigs, dtype=NP_REAL_DTYPE)
    return np.where(
        np.isfinite(eigs) & (eigs > 0.0),
        eigs,
        QFIM_EIGENVALUE_PLOT_EPS,
    )


def _plot_qfim_eigenvalues_by_index(
    eigs: np.ndarray,
    *,
    title: str,
    outpath: str,
    color=METRIC_COLORS["qfim"],
) -> None:
    eigs = np.asarray(eigs, dtype=NP_REAL_DTYPE)
    if eigs.ndim != 2 or eigs.shape[0] == 0 or eigs.shape[1] == 0:
        raise ValueError(
            "Random-point QFIM eigenvalues must have shape "
            "(num_samples, num_params>0)."
        )
    eigs_plot = _qfim_eigenvalues_for_log_plot(eigs)
    num_params = int(eigs.shape[-1])
    indices = np.broadcast_to(
        np.arange(1, num_params + 1, dtype=NP_REAL_DTYPE),
        eigs.shape,
    )

    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    ax.scatter(
        indices.reshape(-1),
        eigs_plot.reshape(-1),
        s=12.0,
        color=color,
        alpha=0.50,
        edgecolors="black",
        linewidths=0.15,
        rasterized=True,
    )
    ax.axhline(
        float(QFIM_TRACE_EIGENVALUE_THRESHOLD),
        color="C3",
        linestyle="--",
        linewidth=1.0,
        label=rf"rank threshold $\lambda_i={QFIM_TRACE_THRESHOLD_TEX}$",
    )
    ticks = _eigenvalue_index_ticks(num_params)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(tick)) for tick in ticks])
    ax.set_xlim(0.5, num_params + 0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("QFIM eigenvalue")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=False)


def _plot_qfim_eigenvalues_by_index_all_layers(
    eigs_by_layer: dict,
    layers,
    *,
    title: str,
    outpath: str,
) -> None:
    valid_layers = [
        int(L) for L in layers if eigs_by_layer.get(int(L)) is not None
    ]
    if not valid_layers:
        return
    max_num_params = max(
        int(np.asarray(eigs_by_layer[L]).shape[-1]) for L in valid_layers
    )
    cmap = matplotlib.colormaps.get_cmap("viridis")
    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    for layer_index, L in enumerate(valid_layers):
        eigs = np.asarray(eigs_by_layer[L], dtype=NP_REAL_DTYPE)
        if eigs.ndim != 2 or eigs.shape[-1] == 0:
            raise ValueError(
                f"Random-point QFIM eigenvalues for L={L} must be 2D."
            )
        eigs_plot = _qfim_eigenvalues_for_log_plot(eigs)
        num_params = int(eigs.shape[-1])
        indices = np.broadcast_to(
            np.arange(1, num_params + 1, dtype=NP_REAL_DTYPE),
            eigs.shape,
        )
        color = cmap(layer_index / max(len(valid_layers) - 1, 1))
        ax.scatter(
            indices.reshape(-1),
            eigs_plot.reshape(-1),
            s=9.0,
            color=color,
            alpha=0.42,
            edgecolors="black",
            linewidths=0.12,
            rasterized=True,
            label=f"L={L}",
        )
    ax.axhline(
        float(QFIM_TRACE_EIGENVALUE_THRESHOLD),
        color="C3",
        linestyle="--",
        linewidth=1.0,
        label=rf"rank threshold $\lambda_i={QFIM_TRACE_THRESHOLD_TEX}$",
    )
    ticks = _eigenvalue_index_ticks(max_num_params)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(tick)) for tick in ticks])
    ax.set_xlim(0.5, max_num_params + 0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("QFIM eigenvalue")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=True)


def _finite_mean_sem(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=NP_REAL_DTYPE).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    sem = (
        0.0
        if values.size < 2
        else float(np.std(values, ddof=1) / np.sqrt(values.size))
    )
    return mean, sem


def _finite_mean_sem_by_column(values) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=NP_REAL_DTYPE)
    if values.ndim != 2:
        raise ValueError("QFIM scalar histories must be two-dimensional.")
    means = np.full(values.shape[1], np.nan, dtype=NP_REAL_DTYPE)
    sems = np.full(values.shape[1], np.nan, dtype=NP_REAL_DTYPE)
    for column in range(values.shape[1]):
        means[column], sems[column] = _finite_mean_sem(values[:, column])
    return means, sems


def _plot_random_qfim_trace_by_layer(
    trace_by_layer: dict,
    layers,
    *,
    keep_label: str,
    num_samples: int,
    outpath: str,
) -> None:
    valid_layers, maxima, means, sems, minima = [], [], [], [], []
    for L in layers:
        values = np.asarray(trace_by_layer[int(L)], dtype=NP_REAL_DTYPE)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue
        mean, sem = _finite_mean_sem(finite_values)
        valid_layers.append(int(L))
        maxima.append(float(np.max(finite_values)))
        means.append(mean)
        sems.append(sem)
        minima.append(float(np.min(finite_values)))
    if not valid_layers:
        return

    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)
    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    ax.errorbar(
        x,
        means,
        yerr=sems,
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=5.5,
        capsize=3.0,
        elinewidth=0.9,
        color=METRIC_COLORS["qfim"],
        label=r"Mean $\pm$ SEM",
        zorder=3,
    )
    ax.plot(
        x,
        minima,
        marker="v",
        linestyle="--",
        linewidth=1.2,
        markersize=5.0,
        color="C2",
        label="Minimum",
    )
    ax.plot(
        x,
        maxima,
        marker="^",
        linestyle="--",
        linewidth=1.2,
        markersize=5.0,
        color="C3",
        label="Maximum",
    )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(QFIM_TRACE_YLABEL)
    ax.set_title(
        f"QFIM trace at {int(num_samples)} random parameter points "
        rf"($\lambda_i \geq {QFIM_TRACE_THRESHOLD_TEX}$, {keep_label})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in valid_layers])
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=False)


def _plot_random_qfim_shannon_entropy_by_layer(
    entropy_by_layer: dict,
    layers,
    *,
    keep_label: str,
    num_samples: int,
    outpath: str,
) -> None:
    """Plot random-point entropy mean with SEM and extrema by layer."""
    valid_layers, maxima, means, sems, minima = [], [], [], [], []
    for L in layers:
        values = np.asarray(entropy_by_layer[int(L)], dtype=NP_REAL_DTYPE)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue
        mean, sem = _finite_mean_sem(finite_values)
        valid_layers.append(int(L))
        maxima.append(float(np.max(finite_values)))
        means.append(mean)
        sems.append(sem)
        minima.append(float(np.min(finite_values)))
    if not valid_layers:
        return

    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)
    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    ax.errorbar(
        x,
        means,
        yerr=sems,
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=5.5,
        capsize=3.0,
        elinewidth=0.9,
        color=METRIC_COLORS["qfim"],
        label=r"Mean $\pm$ SEM",
        zorder=3,
    )
    ax.plot(
        x,
        minima,
        marker="v",
        linestyle="--",
        linewidth=1.2,
        markersize=5.0,
        color="C2",
        label="Minimum",
    )
    ax.plot(
        x,
        maxima,
        marker="^",
        linestyle="--",
        linewidth=1.2,
        markersize=5.0,
        color="C3",
        label="Maximum",
    )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(QFIM_SHANNON_ENTROPY_YLABEL)
    ax.set_title(
        "QFIM spectral Shannon entropy at "
        f"{int(num_samples)} random parameter points "
        rf"($\lambda_i \geq {QFIM_TRACE_THRESHOLD_TEX}$, {keep_label})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in valid_layers])
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=False)


def _plot_qfim_trace_history_by_iteration(
    trace_by_layer: dict,
    layers,
    sample_iters,
    *,
    keep_label: str,
    outpath: str,
) -> None:
    sample_iters = np.asarray(sample_iters, dtype=NP_INT_DTYPE)
    cmap = matplotlib.colormaps.get_cmap("viridis")
    valid_layers = [
        int(L) for L in layers if trace_by_layer.get(int(L)) is not None
    ]
    if not valid_layers:
        return
    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    plotted = False
    for layer_index, L in enumerate(valid_layers):
        values = np.asarray(trace_by_layer[L], dtype=NP_REAL_DTYPE)
        if values.ndim != 2 or values.shape[1] != sample_iters.size:
            raise ValueError(
                f"QFIM Trace history shape mismatch for L={L}: "
                f"{values.shape} vs {sample_iters.size} iterations."
            )
        means, sems = _finite_mean_sem_by_column(values)
        finite = np.isfinite(means)
        if not np.any(finite):
            continue
        plotted = True
        color = cmap(layer_index / max(len(valid_layers) - 1, 1))
        ax.errorbar(
            sample_iters[finite],
            means[finite],
            yerr=sems[finite],
            marker="o",
            linewidth=1.2,
            markersize=4.2,
            capsize=2.5,
            color=color,
            label=f"L={L}",
        )
    if not plotted:
        plt.close(plt.gcf())
        return
    ax.set_xlabel("Iterations")
    ax.set_ylabel(rf"Mean {QFIM_TRACE_YLABEL}")
    ax.set_title(
        "Mean QFIM trace along the optimization path "
        rf"($\lambda_i \geq {QFIM_TRACE_THRESHOLD_TEX}$, {keep_label})"
    )
    ax.set_xticks(sample_iters)
    ax.set_xticklabels(
        [str(int(iteration)) for iteration in sample_iters],
        rotation=45,
        ha="right",
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=True)


def _plot_qfim_shannon_entropy_history_by_iteration(
    entropy_by_layer: dict,
    layers,
    sample_iters,
    *,
    keep_label: str,
    outpath: str,
) -> None:
    """Plot run-mean entropy with SEM along each optimization path."""
    sample_iters = np.asarray(sample_iters, dtype=NP_INT_DTYPE)
    cmap = matplotlib.colormaps.get_cmap("viridis")
    valid_layers = [
        int(L) for L in layers if entropy_by_layer.get(int(L)) is not None
    ]
    if not valid_layers:
        return
    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    plotted = False
    for layer_index, L in enumerate(valid_layers):
        values = np.asarray(entropy_by_layer[L], dtype=NP_REAL_DTYPE)
        if values.ndim != 2 or values.shape[1] != sample_iters.size:
            raise ValueError(
                f"QFIM Shannon-entropy history shape mismatch for L={L}: "
                f"{values.shape} vs {sample_iters.size} iterations."
            )
        means, sems = _finite_mean_sem_by_column(values)
        finite = np.isfinite(means)
        if not np.any(finite):
            continue
        plotted = True
        color = cmap(layer_index / max(len(valid_layers) - 1, 1))
        ax.errorbar(
            sample_iters[finite],
            means[finite],
            yerr=sems[finite],
            marker="o",
            linewidth=1.2,
            markersize=4.2,
            capsize=2.5,
            color=color,
            label=f"L={L}",
        )
    if not plotted:
        plt.close(plt.gcf())
        return
    ax.set_xlabel("Iterations")
    ax.set_ylabel(f"Mean {QFIM_SHANNON_ENTROPY_YLABEL}")
    ax.set_title(
        "Mean QFIM spectral Shannon entropy along the optimization path "
        rf"($\lambda_i \geq {QFIM_TRACE_THRESHOLD_TEX}$, {keep_label})"
    )
    ax.set_xticks(sample_iters)
    ax.set_xticklabels(
        [str(int(iteration)) for iteration in sample_iters],
        rotation=45,
        ha="right",
    )
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=True)


def _plot_flattened_qfim_trace_by_layer(
    trace_by_layer: dict,
    layers,
    *,
    keep_label: str,
    outpath: str,
) -> None:
    valid_layers, means, sems = [], [], []
    for L in layers:
        values = trace_by_layer.get(int(L))
        if values is None:
            continue
        mean, sem = _finite_mean_sem(values)
        if not np.isfinite(mean):
            continue
        valid_layers.append(int(L))
        means.append(mean)
        sems.append(sem)
    if not valid_layers:
        return
    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)
    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    ax.errorbar(
        x,
        means,
        yerr=sems,
        marker="o",
        linewidth=1.3,
        capsize=3.0,
        color=METRIC_COLORS["qfim"],
        label=r"Flattened mean $\pm$ SEM",
    )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(rf"Mean {QFIM_TRACE_YLABEL}")
    ax.set_title(
        "Flattened optimization-path QFIM trace by layer "
        rf"($\lambda_i \geq {QFIM_TRACE_THRESHOLD_TEX}$, {keep_label})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in valid_layers])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    upqc.save_current_figure(outpath, outside_legend=False)


def _plot_flattened_qfim_shannon_entropy_by_layer(
    entropy_by_layer: dict,
    layers,
    *,
    keep_label: str,
    outpath: str,
) -> None:
    """Plot entropy aggregated over runs and sampled iterations by layer."""
    valid_layers, means, sems = [], [], []
    for L in layers:
        values = entropy_by_layer.get(int(L))
        if values is None:
            continue
        mean, sem = _finite_mean_sem(values)
        if not np.isfinite(mean):
            continue
        valid_layers.append(int(L))
        means.append(mean)
        sems.append(sem)
    if not valid_layers:
        return
    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)
    upqc.new_prx_figure(width="double")
    ax = plt.gca()
    ax.errorbar(
        x,
        means,
        yerr=sems,
        marker="o",
        linewidth=1.3,
        capsize=3.0,
        color=METRIC_COLORS["qfim"],
        label=r"Flattened mean $\pm$ SEM",
    )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(f"Mean {QFIM_SHANNON_ENTROPY_YLABEL}")
    ax.set_title(
        "Flattened optimization-path QFIM spectral Shannon entropy by layer "
        rf"($\lambda_i \geq {QFIM_TRACE_THRESHOLD_TEX}$, {keep_label})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in valid_layers])
    ax.set_ylim(bottom=0.0)
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
        result = _load_required_result(
            path,
            require_h_param=True,
            require_variant=True,
        )
    else:
        _validate_result_h_param(
            result,
            path,
            expected_h_param=upqc.h_param,
            required=True,
        )
        _validate_result_variant(result, path, required=True)

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


def _load_random_hessian_results(
    *,
    expected_layers,
    expected_num_samples: int,
) -> None:
    """Validate the outcome-1 archive and reuse its signed eigendecomposition."""
    from unitary_hessian_results import load_measured_unitary_hessian_result

    result = load_measured_unitary_hessian_result(
        upqc.hessian_results_dir,
        expected_h_param=upqc.h_param,
        expected_layers=expected_layers,
        expected_num_samples=expected_num_samples,
        expected_seed_base=upqc.QFIM_SAMPLE_SEED_BASE,
        rank_threshold=float(HESSIAN_RANK_THRESHOLD),
    )
    upqc.hessian_result = result
    upqc.hessian_rank_by_layer = result["rank_by_layer"]
    upqc.hessian_condition_by_layer = result["condition_by_layer"]
    upqc.hessian_by_layer = result["hessian_by_layer"]
    upqc.HESSIAN_RANK_THRESHOLD = result["threshold"]


def _plot_random_hessian_results() -> None:
    """Render the same Hessian statistics and spectra as reset-DPQC."""
    output = _render_unitary_hessian_result(
        upqc.hessian_result, h_param=upqc.h_param,
        figures_dir=upqc.hessian_fig_dir, hamiltonian_matrix=upqc.H_matrix,
    )
    upqc.hessian_statistics_result = output["statistics"]
    upqc.hessian_curvature_result = output["curvature"]


def _load_random_qfim_results() -> None:
    qfim_path = os.path.join(upqc.qfim_results_dir, "qfim_random_points.npz")
    qfim_result = _load_required_result(qfim_path)
    layers = [int(L) for L in np.asarray(qfim_result["layers"], dtype=NP_INT_DTYPE)]

    upqc.qfim_layer_list = layers
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

    # Keep the historical combined archive only for metadata and theta
    # samples.  Its spectra are threshold-masked and therefore cannot support
    # the inclusive Trace definition at the exact threshold boundary.
    upqc.qfim_random_thetas_by_layer = {}

    for L in layers:
        upqc.qfim_random_thetas_by_layer[L] = np.asarray(
            qfim_result[f"L{L}_theta"],
            dtype=NP_REAL_DTYPE,
        )

    upqc.qfim_eigs_by_keep = {
        keep_key: _load_canonical_random_qfim_eigenvalues(
            keep_key,
            expected_layers=layers,
            expected_num_samples=upqc.NUM_QFIM_SAMPLES,
        )
        for keep_key in QFIM_KEEP_KEYS
    }
    upqc.qfim_eigs_reduced_by_layer = upqc.qfim_eigs_by_keep["keep0123"]
    upqc.qfim_eigs_pure_by_layer = upqc.qfim_eigs_by_keep["keep01234"]
    upqc.qfim_trace_random_by_keep = {
        keep_key: {
            L: _qfim_trace_at_or_above_rank_threshold(eigs)
            for L, eigs in eigs_by_layer.items()
        }
        for keep_key, eigs_by_layer in upqc.qfim_eigs_by_keep.items()
    }
    upqc.qfim_shannon_entropy_random_by_keep = {
        keep_key: {
            L: _qfim_spectral_shannon_entropy(eigs)
            for L, eigs in eigs_by_layer.items()
        }
        for keep_key, eigs_by_layer in upqc.qfim_eigs_by_keep.items()
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

    _load_random_hessian_results(
        expected_layers=layers,
        expected_num_samples=upqc.NUM_QFIM_SAMPLES,
    )


def _plot_random_qfim_results(*, rank_threshold=None) -> None:
    upqc.qfim_rank_result = run_unitary_qfim_rank_visualization(
        h_param=upqc.h_param, results_dir=upqc.qfim_results_dir,
        figures_dir=Path(upqc.qfim_fig_dir) / "rank" / "random_points",
        rank_threshold=rank_threshold,
    )
    hs_eigs_reduced_0123_dir = upqc.hs_eigs_reduced_0123_dir
    qfim_trace_dir = os.path.join(upqc.qfim_fig_dir, "trace")
    qfim_shannon_entropy_dir = os.path.join(
        upqc.qfim_fig_dir,
        "shannon_entropy",
    )

    os.makedirs(hs_eigs_reduced_0123_dir, exist_ok=True)
    os.makedirs(upqc.qfim_eigs_reduced_0123_dir, exist_ok=True)
    os.makedirs(upqc.qfim_eigs_pure_dir, exist_ok=True)
    os.makedirs(qfim_trace_dir, exist_ok=True)
    os.makedirs(qfim_shannon_entropy_dir, exist_ok=True)

    qfim_keep_specs = (
        (
            "keep0123",
            upqc.qfim_eigs_reduced_0123_dir,
            "reduced_0123",
        ),
        (
            "keep01234",
            upqc.qfim_eigs_pure_dir,
            "pure_full",
        ),
    )
    for keep_key, eigs_dir, per_layer_tag in qfim_keep_specs:
        eigs_by_layer = upqc.qfim_eigs_by_keep[keep_key]
        keep_label = QFIM_KEEP_LABELS[keep_key]
        for L in upqc.qfim_layer_list:
            _plot_qfim_eigenvalues_by_index(
                eigs_by_layer[L],
                title=(
                    f"QFIM eigenvalues at {upqc.NUM_QFIM_SAMPLES} random "
                    f"points (L={L}, {keep_label})"
                ),
                outpath=os.path.join(
                    eigs_dir,
                    f"L{L}_{per_layer_tag}.pdf",
                ),
            )
        _plot_qfim_eigenvalues_by_index_all_layers(
            eigs_by_layer,
            upqc.qfim_layer_list,
            title=(
                f"QFIM eigenvalues at {upqc.NUM_QFIM_SAMPLES} random "
                f"points ({keep_label})"
            ),
            outpath=os.path.join(
                upqc.qfim_eigs_dir,
                f"qfim_eigs_by_index_layers_{keep_key}.pdf",
            ),
        )
        _plot_random_qfim_trace_by_layer(
            upqc.qfim_trace_random_by_keep[keep_key],
            upqc.qfim_layer_list,
            keep_label=keep_label,
            num_samples=upqc.NUM_QFIM_SAMPLES,
            outpath=os.path.join(
                qfim_trace_dir,
                (
                    "qfim_trace_max_mean_sem_random_points_"
                    f"{QFIM_TRACE_THRESHOLD_FILE_TAG}_{keep_key}.pdf"
                ),
            ),
        )
        _plot_random_qfim_shannon_entropy_by_layer(
            upqc.qfim_shannon_entropy_random_by_keep[keep_key],
            upqc.qfim_layer_list,
            keep_label=keep_label,
            num_samples=upqc.NUM_QFIM_SAMPLES,
            outpath=os.path.join(
                qfim_shannon_entropy_dir,
                (
                    "qfim_shannon_entropy_max_mean_sem_random_points_"
                    f"{QFIM_TRACE_THRESHOLD_FILE_TAG}_{keep_key}.pdf"
                ),
            ),
        )

    for L in upqc.qfim_layer_list:
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

    _plot_random_hessian_results()


def _load_optimization_path_results() -> None:
    layers = [int(L) for L in upqc.layer_list]
    sample_iters = np.asarray(upqc.sample_iters, dtype=NP_INT_DTYPE)
    upqc.qfim_eigs_history_by_keep = {}
    canonical_sample_iters = {}
    for keep_key in QFIM_KEEP_KEYS:
        (
            upqc.qfim_eigs_history_by_keep[keep_key],
            canonical_sample_iters[keep_key],
        ) = _load_canonical_qfim_eigenvalue_history(
            keep_key,
            expected_layers=layers,
            expected_sample_iters=sample_iters,
            expected_num_runs=upqc.num_runs,
        )
    if not np.array_equal(
        canonical_sample_iters["keep0123"],
        canonical_sample_iters["keep01234"],
    ):
        raise ValueError(
            "Canonical QFIM optimization-path archives have inconsistent "
            "sample_iters."
        )
    upqc.sample_iters = canonical_sample_iters["keep0123"]
    upqc.layer_list = layers
    upqc.qfim_eigs_history_by_layer = (
        upqc.qfim_eigs_history_by_keep["keep0123"]
    )
    upqc.qfim_eigs_history_pure_by_layer = (
        upqc.qfim_eigs_history_by_keep["keep01234"]
    )
    upqc.qfim_trace_history_by_keep = {
        keep_key: {
            L: _qfim_trace_at_or_above_rank_threshold(eigs)
            for L, eigs in eigs_by_layer.items()
        }
        for keep_key, eigs_by_layer in upqc.qfim_eigs_history_by_keep.items()
    }
    upqc.qfim_shannon_entropy_history_by_keep = {
        keep_key: {
            L: _qfim_spectral_shannon_entropy(eigs)
            for L, eigs in eigs_by_layer.items()
        }
        for keep_key, eigs_by_layer in upqc.qfim_eigs_history_by_keep.items()
    }

    hs_eigs_path = os.path.join(
        upqc.hs_results_dir,
        "hs_eigs_history_optimization_path_reduced_0123.npz",
    )
    hs_result = _load_required_result(hs_eigs_path)
    upqc.hs_eigs_history_by_layer = {
        L: np.asarray(hs_result[f"L{L}_eigs"], dtype=NP_REAL_DTYPE)
        for L in layers
    }
    hs_pure_result = _load_required_result(
        os.path.join(
            upqc.hs_results_dir,
            "hs_eigs_history_optimization_path_pure_full.npz",
        )
    )
    upqc.hs_eigs_history_pure_by_layer = {
        L: np.asarray(hs_pure_result[f"L{L}_eigs"], dtype=NP_REAL_DTYPE)
        for L in layers
    }

def _plot_optimization_path_results() -> None:
    qfim_trace_dir = os.path.join(upqc.qfim_fig_dir, "trace")
    qfim_shannon_entropy_dir = os.path.join(
        upqc.qfim_fig_dir,
        "shannon_entropy",
    )
    os.makedirs(qfim_trace_dir, exist_ok=True)
    os.makedirs(qfim_shannon_entropy_dir, exist_ok=True)
    for keep_key in QFIM_KEEP_KEYS:
        keep_label = QFIM_KEEP_LABELS[keep_key]
        trace_by_layer = upqc.qfim_trace_history_by_keep[keep_key]
        entropy_by_layer = upqc.qfim_shannon_entropy_history_by_keep[keep_key]
        _plot_qfim_trace_history_by_iteration(
            trace_by_layer,
            upqc.layer_list,
            upqc.sample_iters,
            keep_label=keep_label,
            outpath=os.path.join(
                qfim_trace_dir,
                (
                    "qfim_trace_mean_sem_optimization_path_by_iteration_"
                    f"{QFIM_TRACE_THRESHOLD_FILE_TAG}_{keep_key}.pdf"
                ),
            ),
        )
        _plot_flattened_qfim_trace_by_layer(
            trace_by_layer,
            upqc.layer_list,
            keep_label=keep_label,
            outpath=os.path.join(
                qfim_trace_dir,
                (
                    "qfim_trace_flattened_mean_sem_optimization_path_by_layer_"
                    f"{QFIM_TRACE_THRESHOLD_FILE_TAG}_{keep_key}.pdf"
                ),
            ),
        )
        _plot_qfim_shannon_entropy_history_by_iteration(
            entropy_by_layer,
            upqc.layer_list,
            upqc.sample_iters,
            keep_label=keep_label,
            outpath=os.path.join(
                qfim_shannon_entropy_dir,
                (
                    "qfim_shannon_entropy_mean_sem_optimization_path_"
                    "by_iteration_"
                    f"{QFIM_TRACE_THRESHOLD_FILE_TAG}_{keep_key}.pdf"
                ),
            ),
        )
        _plot_flattened_qfim_shannon_entropy_by_layer(
            entropy_by_layer,
            upqc.layer_list,
            keep_label=keep_label,
            outpath=os.path.join(
                qfim_shannon_entropy_dir,
                (
                    "qfim_shannon_entropy_flattened_mean_sem_"
                    "optimization_path_by_layer_"
                    f"{QFIM_TRACE_THRESHOLD_FILE_TAG}_{keep_key}.pdf"
                ),
            ),
        )

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



def run_unitary_pqc_visualization(
    *,
    h_param: Optional[float] = None,
    convergence_tolerances: Optional[Sequence[float]] = None,
    qfim_logdet_kappa: float = 1.0,
    qfim_rank_threshold: Optional[float] = None,
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
        require_variant=True,
    )

    upqc.configure_unitary_pqc_overparam(h_value=selected_h_param)
    _load_unitary_vqe_results(vqe_result)
    gap_normalized_energy = run_unitary_gap_normalized_visualization(
        h_param=selected_h_param, results_dir=upqc.energy_results_dir,
        figures_dir=upqc.energy_fig_dir,
    )
    generate_convergence_time_outputs(
        upqc.energy_traces_by_layer,
        upqc.layer_list,
        ground_energy=upqc.smallest_eigval,
        tolerances=convergence_tolerances,
        num_runs=upqc.num_runs,
        optimizer_steps=upqc.steps,
        figure_dir=upqc.energy_fig_dir,
        statistics_outpath=os.path.join(
            upqc.energy_results_dir,
            "vqe_convergence_time_statistics.npz",
        ),
        metadata={
            "h_param": selected_h_param,
            "architecture": upqc.ANSATZ_NAME,
            "measurement_outcome": upqc.MEASUREMENT_OUTCOME,
            "source_archive": os.path.basename(vqe_result_path),
        },
    )
    upqc.plot_vqe_optimization_results()
    # Re-render the shared final-error filenames with the DPQC layout and add
    # its log-scale, threshold-detail, and multiple-tolerance figures.
    _plot_vqe_ground_truth_error_results()

    _load_random_qfim_results()
    _plot_random_qfim_results(rank_threshold=qfim_rank_threshold)
    qfim_logdet = run_unitary_qfim_logdet_visualization(
        h_param=selected_h_param, results_dir=upqc.qfim_results_dir,
        kappa=qfim_logdet_kappa,
    )

    _load_optimization_path_results()
    _plot_optimization_path_results()

    result = upqc.collect_unitary_pqc_result()
    result["gap_normalized_energy"] = gap_normalized_energy
    result["qfim_logdet"] = qfim_logdet
    result["qfim_rank"] = upqc.qfim_rank_result
    return result


if __name__ == "__main__":
    visualization_result = run_unitary_pqc_visualization(
        h_param=_CLI_ARGS.h_param,
        convergence_tolerances=_CLI_ARGS.convergence_tolerances,
        qfim_logdet_kappa=_CLI_ARGS.qfim_logdet_kappa,
        qfim_rank_threshold=_CLI_ARGS.qfim_rank_threshold,
    )
    print(
        "Visualized Hamiltonian parameter h: "
        f"{visualization_result['h_param']}"
    )
    print(f"Saved figures to: {visualization_result['save_dir']}")
