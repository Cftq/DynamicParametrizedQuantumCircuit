#!/usr/bin/env python
# coding: utf-8
"""Visualize saved DPQC overparameterization numerical results.

Run DPQC_overparam_vqe.py followed by DPQC_overparam_qfim.py to create the
.npz files under figs/dpqc/h_<h_param>/numerical_results. This script loads
those saved results and generates figures without recomputing VQE/QFIM.
QFIM trace figures sum only eigenvalues at or above the configured QFIM rank
threshold (1e-12 by default).

Example::

    python src/dpqc/DPQC_overparam_visualize.py --h-param 0.1
    python src/dpqc/DPQC_overparam_visualize.py --h-param 0.1 --output-family dpqc_reset
    python src/dpqc/DPQC_overparam_visualize.py --h-param 0.1 --hessian-only
"""


import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple


# Support direct execution from any working directory.  Shared configuration,
# plotting helpers, and numerical utilities live in ``src/common`` rather than
# beside this visualization entry point.
_MODULE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _MODULE_DIR.parent
_COMMON_DIR = _SRC_DIR / "common"
for _path in (_MODULE_DIR, _COMMON_DIR):
    _path_string = str(_path)
    if _path_string not in sys.path:
        sys.path.insert(0, _path_string)


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


def _positive_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a comma-separated integer list"
        ) from exc
    if not parsed or any(not item.strip() for item in value.split(",")):
        raise argparse.ArgumentTypeError(
            "value must be a non-empty comma-separated integer list"
        )
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("indices must not be duplicated")
    return parsed


def _parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Visualize saved DPQC results for one Hamiltonian parameter h. "
            "Run this command from the same project directory used by the "
            "DPQC compute programs."
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
    parser.add_argument(
        "--output-family",
        choices=("dpqc", "dpqc_reset"),
        default="dpqc",
        help=(
            "Result family below ./figs to load and visualize "
            "(default: dpqc). Use dpqc_reset for the fixed-Rx(pi) "
            "reset model."
        ),
    )
    parser.add_argument(
        "--skip-optimization-path-qfim",
        action="store_true",
        help=(
            "Do not load or render QFIM diagnostics evaluated along the VQE "
            "optimization path. Random-point QFIM figures are still rendered."
        ),
    )
    parser.add_argument(
        "--skip-qfim-eigs-by-index-layers",
        action="store_true",
        help=(
            "Do not render the QFIM eigenvalue-by-index figures that overlay "
            "all layer counts. Per-layer eigenvalue figures are still rendered."
        ),
    )
    parser.add_argument(
        "--skip-qfim-trace-figures",
        action="store_true",
        help="Do not render any QFIM trace figures.",
    )
    hessian_mode = parser.add_mutually_exclusive_group()
    hessian_mode.add_argument(
        "--hessian-only",
        action="store_true",
        help=(
            "Compute/load random-point Hessians and render the rank and "
            "condition-number figures only. "
            "This path does not require TensorCircuit."
        ),
    )
    hessian_mode.add_argument(
        "--with-hessian",
        action="store_true",
        help="Run the Hessian workflow after the existing energy/QFIM figures.",
    )
    parser.add_argument(
        "--reuse-hessian-results",
        action="store_true",
        help="Render the existing random-point Hessian result without recomputing.",
    )
    parser.add_argument(
        "--hessian-results-dir",
        type=Path,
        default=None,
        help="Optional Hessian numerical-results directory.",
    )
    parser.add_argument(
        "--hessian-figures-dir",
        type=Path,
        default=None,
        help="Optional Hessian figure output directory.",
    )
    parser.add_argument(
        "--hessian-layers",
        type=_comma_separated_ints,
        default=None,
        help="Comma-separated layers to analyze (default: QFIM layer schedule).",
    )
    parser.add_argument(
        "--hessian-num-samples",
        type=_positive_int,
        default=int(cfg.NUM_QFIM_SAMPLES),
        help=(
            "Number of random parameter points per layer "
            f"(default: {int(cfg.NUM_QFIM_SAMPLES)})."
        ),
    )
    parser.add_argument(
        "--hessian-seed-base",
        type=_nonnegative_int,
        default=int(cfg.QFIM_SAMPLE_SEED_BASE),
        help=(
            "Base PRNG seed; layer L uses seed base + L "
            f"(default: {int(cfg.QFIM_SAMPLE_SEED_BASE)})."
        ),
    )
    parser.add_argument(
        "--hessian-hvp-chunk-size",
        type=_positive_int,
        default=8,
        help="Number of Hessian-vector products per compiled batch (default: 8).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    _CLI_ARGS = _parse_cli_args()
else:
    _CLI_ARGS = argparse.Namespace(
        h_param=float(cfg.H_PARAM),
        convergence_tolerances=None,
        output_family="dpqc",
        skip_optimization_path_qfim=False,
        skip_qfim_eigs_by_index_layers=False,
        skip_qfim_trace_figures=False,
    )

h_param = float(_CLI_ARGS.h_param)
if not math.isfinite(h_param):
    raise ValueError("h_param must be a finite number.")

output_family = str(_CLI_ARGS.output_family)
if output_family not in ("dpqc", "dpqc_reset"):
    raise ValueError(f"Unsupported output family: {output_family!r}.")
INCLUDE_OPTIMIZATION_PATH_QFIM = not bool(
    getattr(_CLI_ARGS, "skip_optimization_path_qfim", False)
)
INCLUDE_QFIM_EIGS_BY_INDEX_LAYERS = not bool(
    getattr(_CLI_ARGS, "skip_qfim_eigs_by_index_layers", False)
)
INCLUDE_QFIM_TRACE_FIGURES = not bool(
    getattr(_CLI_ARGS, "skip_qfim_trace_figures", False)
)
if output_family == "dpqc_reset" and (
    getattr(_CLI_ARGS, "hessian_only", False)
    or getattr(_CLI_ARGS, "with_hessian", False)
):
    raise ValueError(
        "Reset-DPQC Hessian visualization is not supported by the "
        "14L-parameter dynamic-channel Hessian workflow. Run without "
        "--hessian-only or "
        "--with-hessian."
    )


# ============================================================
# Random-parameter Hessian rank and condition-number analysis
# ============================================================

_HESSIAN_RANDOM_RESULT_NAME = "hessian_random_points.npz"
_HESSIAN_RANDOM_SCHEMA_VERSION = 1
_HESSIAN_PARAMETERS_PER_LAYER = 14
_HESSIAN_RANK_THRESHOLD = float(cfg.QFIM_EFFECTIVE_RANK_THRESHOLD)


def _hessian_h_tag(value: float) -> str:
    return f"{float(value):.12g}"


def _resolve_hessian_paths(args):
    """Resolve the random-point Hessian result and two-figure output dirs."""
    h_tags = list(
        dict.fromkeys((str(float(args.h_param)), _hessian_h_tag(args.h_param)))
    )
    h_roots = [
        Path.cwd() / "figs" / str(args.output_family) / f"h_{tag}"
        for tag in h_tags
    ]
    h_root = next((path for path in h_roots if path.is_dir()), h_roots[0])
    results_dir = (
        h_root / "numerical_results" / "hessian"
        if args.hessian_results_dir is None
        else Path(args.hessian_results_dir).expanduser().resolve()
    )
    figures_dir = (
        h_root / "hessian_figures"
        if args.hessian_figures_dir is None
        else Path(args.hessian_figures_dir).expanduser().resolve()
    )
    return results_dir, figures_dir


def _load_random_hessian_result(
    results_dir: Path,
    *,
    expected_h_param: float,
    requested_layers=None,
):
    """Load and validate the minimal random-point Hessian archive."""
    import numpy as hnp

    path = Path(results_dir) / _HESSIAN_RANDOM_RESULT_NAME
    if not path.is_file():
        raise FileNotFoundError(
            "Random-point Hessian result was not found: "
            f"{path}. Run without --reuse-hessian-results first."
        )

    required_metadata = {
        "schema_version",
        "h_param",
        "layers",
        "num_hessian_samples",
        "hessian_sample_seed_base",
        "hessian_rank_threshold",
        "hessian_rank_definition",
        "hessian_condition_number_definition",
        "parameter_distribution",
        "parameters_per_layer",
        "hessian_method",
        "hvp_chunk_size",
    }
    with hnp.load(path, allow_pickle=False) as archive:
        missing = sorted(required_metadata.difference(archive.files))
        if missing:
            raise KeyError(
                f"Random-point Hessian archive {path} is missing: "
                + ", ".join(missing)
            )
        data = {key: archive[key] for key in archive.files}

    schema_version = int(hnp.asarray(data["schema_version"]).item())
    if schema_version != _HESSIAN_RANDOM_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Hessian schema version {schema_version}; "
            f"expected {_HESSIAN_RANDOM_SCHEMA_VERSION}."
        )

    archived_h = float(hnp.asarray(data["h_param"]).item())
    if not math.isclose(
        archived_h,
        float(expected_h_param),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Hessian archive h_param mismatch: {archived_h} != "
            f"{expected_h_param}."
        )

    threshold = float(hnp.asarray(data["hessian_rank_threshold"]).item())
    if not math.isclose(
        threshold,
        _HESSIAN_RANK_THRESHOLD,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            f"Hessian rank threshold mismatch: {threshold} != "
            f"{_HESSIAN_RANK_THRESHOLD}."
        )

    parameters_per_layer = int(
        hnp.asarray(data["parameters_per_layer"]).item()
    )
    if parameters_per_layer != _HESSIAN_PARAMETERS_PER_LAYER:
        raise ValueError(
            "Unexpected Hessian parameter count per layer: "
            f"{parameters_per_layer}."
        )

    num_samples = int(hnp.asarray(data["num_hessian_samples"]).item())
    if num_samples <= 0:
        raise ValueError("num_hessian_samples must be positive.")

    archived_layers = hnp.asarray(data["layers"], dtype=int).reshape(-1)
    if (
        archived_layers.size == 0
        or hnp.any(archived_layers <= 0)
        or hnp.unique(archived_layers).size != archived_layers.size
    ):
        raise ValueError("Hessian archive layers must be unique positive integers.")
    available_layers = [int(layer) for layer in archived_layers]

    if requested_layers is None:
        layers = available_layers
    else:
        layers = [int(layer) for layer in requested_layers]
        missing_layers = [layer for layer in layers if layer not in available_layers]
        if missing_layers:
            raise KeyError(
                "Requested Hessian layers are absent from the archive: "
                + ", ".join(str(layer) for layer in missing_layers)
            )

    rank_by_layer = {}
    condition_by_layer = {}
    for layer in layers:
        rank_key = f"L{layer}_rank"
        condition_key = f"L{layer}_condition_number"
        missing_keys = [
            key for key in (rank_key, condition_key) if key not in data
        ]
        if missing_keys:
            raise KeyError(
                f"Hessian archive {path} is missing: " + ", ".join(missing_keys)
            )

        raw_ranks = hnp.asarray(data[rank_key])
        if (
            raw_ranks.shape != (num_samples,)
            or hnp.iscomplexobj(raw_ranks)
            or not hnp.all(hnp.isfinite(raw_ranks))
            or not hnp.all(raw_ranks == hnp.rint(raw_ranks))
        ):
            raise ValueError(f"Invalid Hessian rank array {rank_key}.")
        ranks = raw_ranks.astype(hnp.int64)
        if hnp.any(ranks < 0) or hnp.any(
            ranks > parameters_per_layer * int(layer)
        ):
            raise ValueError(f"Out-of-range Hessian ranks in {rank_key}.")

        conditions = hnp.asarray(data[condition_key], dtype=float)
        if conditions.shape != (num_samples,) or hnp.any(hnp.isinf(conditions)):
            raise ValueError(
                f"Invalid Hessian condition-number array {condition_key}."
            )
        finite_conditions = hnp.isfinite(conditions)
        if hnp.any(conditions[finite_conditions] < 1.0 - 1e-12):
            raise ValueError(
                f"Hessian condition numbers below one in {condition_key}."
            )
        if not hnp.array_equal(finite_conditions, ranks > 0):
            raise ValueError(
                f"Hessian rank/condition definedness mismatch at L={layer}."
            )

        rank_by_layer[layer] = ranks
        condition_by_layer[layer] = conditions

    return {
        "path": path,
        "layers": layers,
        "num_samples": num_samples,
        "threshold": threshold,
        "rank_by_layer": rank_by_layer,
        "condition_by_layer": condition_by_layer,
    }


def _finite_hessian_statistics(values):
    """Return maximum, mean, SEM, and minimum over finite random samples."""
    import numpy as hnp

    samples = hnp.asarray(values, dtype=float).reshape(-1)
    samples = samples[hnp.isfinite(samples)]
    if samples.size == 0:
        return (hnp.nan, hnp.nan, hnp.nan, hnp.nan)
    sem = (
        float(hnp.std(samples, ddof=1) / hnp.sqrt(samples.size))
        if samples.size > 1
        else 0.0
    )
    return (
        float(hnp.max(samples)),
        float(hnp.mean(samples)),
        sem,
        float(hnp.min(samples)),
    )


def _hessian_threshold_tex(threshold: float) -> str:
    exponent = int(math.floor(math.log10(float(threshold))))
    mantissa = float(threshold) / (10.0**exponent)
    if math.isclose(mantissa, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return rf"10^{{{exponent}}}"
    return rf"{mantissa:g}\times 10^{{{exponent}}}"


def _plot_random_hessian_summary(
    values_by_layer: dict,
    layers,
    *,
    ylabel: str,
    title: str,
    outpath: Path,
    lower_bound_zero: bool,
) -> Path:
    """Plot layerwise maximum, mean +/- SEM, and minimum in one figure."""
    import matplotlib.pyplot as plt
    import numpy as hnp

    valid_layers = [int(layer) for layer in layers if int(layer) in values_by_layer]
    statistics = [
        _finite_hessian_statistics(values_by_layer[layer])
        for layer in valid_layers
    ]
    x = hnp.asarray(valid_layers, dtype=float)
    maxima = hnp.asarray([item[0] for item in statistics], dtype=float)
    means = hnp.asarray([item[1] for item in statistics], dtype=float)
    sems = hnp.asarray([item[2] for item in statistics], dtype=float)
    minima = hnp.asarray([item[3] for item in statistics], dtype=float)
    finite = (
        hnp.isfinite(maxima)
        & hnp.isfinite(means)
        & hnp.isfinite(sems)
        & hnp.isfinite(minima)
    )
    if not hnp.any(finite):
        raise ValueError(f"No finite Hessian statistics are available for {title}.")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(
        x[finite],
        maxima[finite],
        marker="^",
        linestyle="--",
        linewidth=1.2,
        markersize=5.0,
        color="C3",
        label="Maximum",
    )
    ax.errorbar(
        x[finite],
        means[finite],
        yerr=sems[finite],
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=5.5,
        capsize=3.0,
        elinewidth=0.9,
        color="C0",
        label=r"Mean $\pm$ SEM",
        zorder=3,
    )
    ax.plot(
        x[finite],
        minima[finite],
        marker="v",
        linestyle="--",
        linewidth=1.2,
        markersize=5.0,
        color="C2",
        label="Minimum",
    )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(layer) for layer in valid_layers])
    if lower_bound_zero:
        ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    fig.tight_layout()

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return outpath


def visualize_hessian_results(
    results_dir: Path,
    figures_dir: Path,
    *,
    expected_h_param: float,
    layers=None,
):
    """Render only the random-point Hessian rank and condition figures."""
    result = _load_random_hessian_result(
        results_dir,
        expected_h_param=expected_h_param,
        requested_layers=layers,
    )
    threshold_tex = _hessian_threshold_tex(result["threshold"])
    figures_dir = Path(figures_dir)
    rank_path = _plot_random_hessian_summary(
        result["rank_by_layer"],
        result["layers"],
        ylabel=rf"Hessian rank ($|\lambda_i| \geq {threshold_tex}$)",
        title=(
            f"Hessian rank at {result['num_samples']} random parameter points"
        ),
        outpath=figures_dir / "hessian_rank_random_points.pdf",
        lower_bound_zero=True,
    )
    condition_path = _plot_random_hessian_summary(
        result["condition_by_layer"],
        result["layers"],
        ylabel=(
            rf"Thresholded Hessian condition number "
            rf"($|\lambda_i| \geq {threshold_tex}$)"
        ),
        title=(
            "Hessian condition number at "
            f"{result['num_samples']} random parameter points"
        ),
        outpath=figures_dir / "hessian_condition_number_random_points.pdf",
        lower_bound_zero=False,
    )
    print(
        "Hessian visualization: saved rank and condition-number figures to "
        f"{figures_dir}",
        flush=True,
    )
    return rank_path, condition_path


def run_hessian_workflow(args):
    """Compute/reuse random-point Hessian summaries and render exactly two PDFs."""
    results_dir, figures_dir = _resolve_hessian_paths(args)
    if not args.reuse_hessian_results:
        from DPQC_overparam_hessian import run_hessian_analysis

        run_hessian_analysis(
            output_dir=results_dir,
            h_param=float(args.h_param),
            layers=args.hessian_layers,
            num_samples=int(args.hessian_num_samples),
            seed_base=int(args.hessian_seed_base),
            hvp_chunk_size=int(args.hessian_hvp_chunk_size),
        )
    return visualize_hessian_results(
        results_dir,
        figures_dir,
        expected_h_param=float(args.h_param),
        layers=args.hessian_layers,
    )


if __name__ == "__main__" and _CLI_ARGS.hessian_only:
    run_hessian_workflow(_CLI_ARGS)
    raise SystemExit(0)


# ------------------------------------------------------------
# IMPORTANT: env vars should be set BEFORE importing jax
# ------------------------------------------------------------
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.ticker as mticker
import numpy as np
import tensorcircuit as tc
from matplotlib.patches import Patch
from convergence_time import generate_convergence_time_outputs
from plot import (
    new_fig_ax,
    save_fig,
    style_axes_for_prx,
)
from tqdm.auto import tqdm

jax.config.update("jax_enable_x64", True)

tc.set_backend("jax")
tc.set_dtype("complex128")

COMPLEX_DTYPE = jnp.complex128
NP_REAL_DTYPE = np.float64
NP_COMPLEX_DTYPE = np.complex128
NP_INT_DTYPE = np.int64

from dpqc_overparam_common import (
    _thr_tag,
    build_H_matrix_jax,
    build_layer_list,
    dpqc_vqe_optimizer_display_name,
    hamiltonian_terms,
    load_npz_result as _load_npz_result_unchecked,
    normalize_dpqc_vqe_optimizer_name,
    rho_zero_state,
)


def _validate_result_h_param(
    result: dict,
    result_path,
    *,
    required: bool = False,
) -> None:
    """Ensure an archive belongs to the h value selected on the CLI."""
    if "h_param" not in result:
        if required:
            raise KeyError(
                "The required DPQC archive does not contain h_param metadata: "
                f"{Path(result_path).resolve()}"
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
    if NP_REAL_DTYPE(archived_h_param) != NP_REAL_DTYPE(h_param):
        raise ValueError(
            "Saved DPQC h_param does not match --h-param: "
            f"{archived_h_param} != {h_param} in "
            f"{Path(result_path).resolve()}"
        )


def load_npz_result(result_path) -> dict:
    """Load one h-scoped archive and validate h metadata when available."""
    result = _load_npz_result_unchecked(str(result_path))
    _validate_result_h_param(result, result_path, required=False)
    return result

# ============================================================
# Shared constants / helpers
# ============================================================
num_system_qubits = 5
tolerance = cfg.TOLERANCE
steps = cfg.STEPS
num_runs = int(cfg.NUM_RUNS)
if num_runs <= 0:
    raise ValueError("cfg.NUM_RUNS must be a positive integer.")
lr = cfg.LEARNING_RATE

# Optimization-history sampling cadence used by history plots.
eps = 1e-12
sample_every = cfg.SAMPLE_EVERY

# Upper cutoff for the zoomed final-energy-error distribution.  The h=0.1
# results have a dense lowest-error cloud below 0.6; changing this value
# regenerates the detailed figure with a different cutoff.
FINAL_ENERGY_ERROR_DETAIL_THRESHOLD = 6e-1

# Optimization-history sampling points used for history plots.  Iteration 1
# is used instead of iteration 0 so the optimized-path parameters are sampled at
#   1, 1000, 2000, ..., 10000.
sample_iters = np.asarray(
    [1] + list(range(sample_every, steps + 1, sample_every)),
    dtype=NP_INT_DTYPE,
)

sample_iters = sample_iters[
    (sample_iters >= 1) & (sample_iters <= steps)
]

sample_iters = np.unique(sample_iters).astype(NP_INT_DTYPE)
sample_iter_set = set(int(t) for t in sample_iters.tolist())

NUM_BLOCKS = 4
PARAMS_PER_BLOCK = 3
EXTRA_PARAMS_PER_LAYER = 2
n_param_per_layer = NUM_BLOCKS * PARAMS_PER_BLOCK + EXTRA_PARAMS_PER_LAYER

TOP, LEFT, RIGHT, BOTTOM, ANC_CENTER, FRESH_ANCILLA = 0, 1, 2, 3, 4, 5

LAYER_PAIRS = (
    (LEFT, BOTTOM),
    (RIGHT, BOTTOM),
    (TOP, RIGHT),
    (TOP, ANC_CENTER),
)

RED4_COLOR = "blue"

# Reduced-system QFIM identifiers used in filenames and figure titles.
# Define these near the top so later cells/sections cannot hit NameError.
keep_key = "keep0123"
keep_label = "Reduced (0,1,2,3)"
keep_key_5 = "keep01234"
keep_label_5 = "Reduced (0,1,2,3,4)"


def make_layer_legend_handles(layer_list, cmap, *, alpha=0.25):
    n = max(len(layer_list), 1)

    return [
        Patch(
            facecolor=cmap(idx / n),
            edgecolor=cmap(idx / n),
            alpha=alpha,
            label=f"L{L}",
        )
        for idx, L in enumerate(layer_list)
    ]


def violin_width_from_positions(positions, *, scale=0.60, default=0.75):
    positions = np.asarray(positions, dtype=NP_REAL_DTYPE)

    return (
        scale * np.min(np.diff(np.sort(positions)))
        if positions.size > 1
        else default
    )


def style_violin(vp, color, *, alpha=0.20, lw=1.0, line_color=None):
    for body in vp["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(alpha)
        body.set_linewidth(lw)

    line_color = color if line_color is None else line_color

    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        artist = vp.get(key)
        if artist is not None:
            artist.set_color(line_color)
            artist.set_linewidth(lw)


def style_violin_bodies(vp, colors, *, alpha=0.20, lw=1.0, line_color=None):
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(alpha)
        body.set_linewidth(lw)

    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        artist = vp.get(key)
        if artist is not None:
            if line_color is not None:
                artist.set_color(line_color)
            artist.set_linewidth(lw)


def _beeswarm_offsets_1d(
    y_for_layout: np.ndarray,
    *,
    max_width: float = 0.32,
    nbins: Optional[int] = None,
) -> np.ndarray:
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
        edges = np.linspace(ymin, ymax, nbins + 1)
        bin_ids = np.digitize(y, edges[1:-1], right=False).astype(NP_INT_DTYPE)

    offsets = np.zeros(n, dtype=NP_REAL_DTYPE)

    for b in np.unique(bin_ids):
        inds = np.where(bin_ids == b)[0]
        m = int(inds.size)

        if m <= 1:
            offsets[inds] = 0.0
            continue

        inds = inds[np.argsort(y[inds])]

        order = np.zeros(m, dtype=NP_REAL_DTYPE)

        for k in range(1, m):
            amp = (k + 1) // 2
            sign = 1.0 if k % 2 == 1 else -1.0
            order[k] = sign * amp

        max_abs = np.max(np.abs(order))

        if max_abs > 0.0:
            order = order / max_abs * max_width

        offsets[inds] = order

    return offsets


def plot_beeswarm_by_layer(
    datasets,
    layer_list,
    *,
    cmap,
    ylabel,
    title,
    outpath,
    point_size: float = 18.0,
    alpha: float = 0.65,
    edge_lw: float = 0.25,
    max_width: float = 0.32,
    log_scale: bool = False,
    eps: float = 1e-12,
):
    positions = np.asarray(layer_list, dtype=NP_REAL_DTYPE)

    fig, ax = new_fig_ax(outside_legend=False)

    for idx, (L, values) in enumerate(zip(layer_list, datasets)):
        color = cmap(idx / len(layer_list))

        y = np.asarray(values, dtype=NP_REAL_DTYPE).reshape(-1)

        # A thresholded detail view can legitimately leave a layer empty.
        # Skip non-finite/empty inputs before computing layout bins or a
        # median, while keeping every layer position on the x-axis.
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue

        if log_scale:
            y_plot = np.maximum(y, eps)
            y_for_layout = np.log10(y_plot)
        else:
            y_plot = y
            y_for_layout = y

        offsets = _beeswarm_offsets_1d(
            y_for_layout,
            max_width=max_width,
        )

        x = np.asarray(L, dtype=NP_REAL_DTYPE) + offsets

        ax.scatter(
            x,
            y_plot,
            s=point_size,
            alpha=alpha,
            color=color,
            edgecolors="black",
            linewidths=edge_lw,
            zorder=3,
        )

        median_y = np.median(y_plot)

        ax.hlines(
            median_y,
            xmin=float(L) - max_width,
            xmax=float(L) + max_width,
            color="black",
            linewidth=1.0,
            zorder=4,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(L) for L in layer_list])
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.3)
    else:
        ax.grid(True, axis="y", alpha=0.3)

    save_fig(fig, ax, outpath, outside_legend=False)


def plot_history_violin(
    data_by_layer,
    layer_list,
    *,
    sample_iters,
    x_groups,
    offset_span,
    violin_width,
    cmap,
    ylabel,
    title,
    outpath,
    transform,
    log_scale: bool = True,
):
    fig, ax = new_fig_ax(outside_legend=True)
    n = len(layer_list)
    handles = []

    for idx, L in enumerate(layer_list):
        color = cmap(idx / n)
        handles.append(
            Patch(
                facecolor=color,
                edgecolor=color,
                alpha=0.25,
                label=f"L{L}",
            )
        )

        offset = (idx - (n - 1) / 2) * (offset_span / n)
        positions = x_groups + offset

        runs = np.asarray(data_by_layer[L], dtype=NP_REAL_DTYPE)
        datasets = [transform(runs[:, t]) for t in sample_iters]

        vp = ax.violinplot(
            datasets,
            positions=positions,
            widths=violin_width,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        style_violin(vp, color, alpha=0.20, lw=1.0)

    ax.set_xlabel("Iterations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_groups)
    ax.set_xticklabels([str(t) for t in sample_iters], rotation=45, ha="right")

    if log_scale:
        ax.set_yscale("log")

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

    save_fig(fig, ax, outpath, outside_legend=True)


def plot_violin_by_layer(
    datasets,
    layer_list,
    *,
    cmap,
    ylabel,
    title,
    outpath,
    alpha=0.20,
    lw=1.0,
    line_color=None,
    show_legend: bool = True,
    log_scale: bool = False,
):
    positions = np.asarray(layer_list, dtype=NP_REAL_DTYPE)
    colors = [cmap(idx / len(layer_list)) for idx in range(len(layer_list))]

    fig, ax = new_fig_ax(outside_legend=show_legend)

    vp = ax.violinplot(
        datasets,
        positions=positions,
        widths=violin_width_from_positions(positions),
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    style_violin_bodies(vp, colors, alpha=alpha, lw=lw, line_color=line_color)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(L) for L in layer_list])
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.3)
    else:
        ax.grid(True, axis="y", alpha=0.3)

    if show_legend:
        ax.legend(
            handles=make_layer_legend_handles(layer_list, cmap),
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

    save_fig(fig, ax, outpath, outside_legend=show_legend)


def plot_single_line_by_layer(
    x,
    y,
    *,
    ylabel,
    title,
    outpath,
    label,
    ylim=None,
):
    fig, ax = new_fig_ax(outside_legend=False)

    ax.plot(x, y, marker="o", linestyle="-", label=label)
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.3)

    save_fig(fig, ax, outpath, outside_legend=False)


SUCCESS_PROBABILITY_FIGURE_THRESHOLDS = np.asarray(
    cfg.SUCCESS_PROBABILITY_FIGURE_THRESHOLDS,
    dtype=NP_REAL_DTYPE,
)


def _warn_skip_success_probability_figure(message: str) -> None:
    warnings.warn(
        "Skipping multiple-accuracy success-probability figure: "
        f"{message}",
        RuntimeWarning,
        stacklevel=2,
    )


def _success_probability_result_for_figure(result: dict) -> dict:
    """Recompute plot summaries at the figure-specific thresholds."""
    thresholds = np.asarray(
        SUCCESS_PROBABILITY_FIGURE_THRESHOLDS,
        dtype=NP_REAL_DTYPE,
    )
    if thresholds.ndim != 1 or thresholds.size == 0:
        raise ValueError(
            "SUCCESS_PROBABILITY_FIGURE_THRESHOLDS must be a non-empty "
            "one-dimensional sequence"
        )
    if not np.all(np.isfinite(thresholds)) or np.any(thresholds <= 0.0):
        raise ValueError(
            "SUCCESS_PROBABILITY_FIGURE_THRESHOLDS must contain only "
            "positive finite values"
        )
    if np.any(np.diff(thresholds) >= 0.0):
        raise ValueError(
            "SUCCESS_PROBABILITY_FIGURE_THRESHOLDS must be strictly decreasing"
        )

    if "layers" not in result or "num_trials" not in result:
        raise ValueError("archive is missing 'layers' or 'num_trials'")

    layers = np.asarray(result["layers"])
    if layers.ndim != 1 or layers.size == 0:
        raise ValueError("'layers' must be a non-empty one-dimensional array")

    try:
        num_trials_array = np.asarray(
            result["num_trials"],
            dtype=NP_REAL_DTYPE,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("'num_trials' must be a numeric scalar") from exc
    if (
        num_trials_array.size != 1
        or not np.isfinite(num_trials_array.reshape(-1)[0])
        or num_trials_array.reshape(-1)[0]
        != np.rint(num_trials_array.reshape(-1)[0])
    ):
        raise ValueError("'num_trials' must be one finite integer scalar")
    num_trials = int(num_trials_array.reshape(-1)[0])
    if num_trials <= 0:
        raise ValueError("'num_trials' must be positive")

    if "final_energy_errors" in result:
        try:
            final_energy_errors = np.asarray(
                result["final_energy_errors"],
                dtype=NP_REAL_DTYPE,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "'final_energy_errors' must be a numeric array"
            ) from exc
    else:
        if "final_energies" not in result:
            raise ValueError(
                "archive is missing both 'final_energy_errors' and "
                "'final_energies'"
            )
        ground_energy_key = (
            "ground_energy"
            if "ground_energy" in result
            else "smallest_eigval"
        )
        if ground_energy_key not in result:
            raise ValueError(
                "archive with 'final_energies' is missing 'ground_energy' "
                "and 'smallest_eigval'"
            )
        try:
            final_energies = np.asarray(
                result["final_energies"],
                dtype=NP_REAL_DTYPE,
            )
            ground_energy_array = np.asarray(
                result[ground_energy_key],
                dtype=NP_REAL_DTYPE,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "'final_energies' and the ground energy must be numeric"
            ) from exc
        if ground_energy_array.size != 1 or not np.isfinite(
            ground_energy_array.reshape(-1)[0]
        ):
            raise ValueError("ground energy must be one finite scalar")
        final_energy_errors = np.maximum(
            final_energies - ground_energy_array.reshape(-1)[0],
            0.0,
        )

    expected_error_shape = (layers.size, num_trials)
    if final_energy_errors.shape != expected_error_shape:
        raise ValueError(
            f"'final_energy_errors' has shape {final_energy_errors.shape}; "
            f"expected {expected_error_shape}"
        )
    if not np.all(np.isfinite(final_energy_errors)):
        raise ValueError("'final_energy_errors' contains a non-finite value")
    if np.any(final_energy_errors < 0.0):
        raise ValueError("'final_energy_errors' contains a negative value")

    success_indicators = (
        final_energy_errors[:, :, None] <= thresholds[None, None, :]
    ).astype(NP_INT_DTYPE)
    success_counts = np.sum(
        success_indicators,
        axis=1,
        dtype=NP_INT_DTYPE,
    )
    success_probabilities = (
        success_counts.astype(NP_REAL_DTYPE) / NP_REAL_DTYPE(num_trials)
    )

    figure_result = dict(result)
    figure_result.update(
        thresholds=thresholds,
        tolerances=thresholds,
        success_indicators=success_indicators,
        success_counts=success_counts,
        success_probabilities=success_probabilities,
    )
    return figure_result


def _validated_success_probability_data(result: dict):
    """Validate and normalize the dedicated multiple-accuracy archive."""
    required_keys = {
        "layers",
        "num_trials",
        "success_counts",
        "success_probabilities",
    }
    missing_keys = sorted(required_keys.difference(result))
    if missing_keys:
        raise ValueError(
            "archive is missing required key(s): "
            + ", ".join(repr(key) for key in missing_keys)
        )

    if "thresholds" not in result and "tolerances" not in result:
        raise ValueError(
            "archive is missing both 'thresholds' and 'tolerances'"
        )

    try:
        layers_float = np.asarray(
            result["layers"],
            dtype=NP_REAL_DTYPE,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("'layers' must be a numeric one-dimensional array") from exc

    if layers_float.ndim != 1 or layers_float.size == 0:
        raise ValueError("'layers' must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(layers_float)):
        raise ValueError("'layers' contains a non-finite value")
    if not np.all(layers_float == np.rint(layers_float)):
        raise ValueError("'layers' must contain only integer values")

    try:
        layers = layers_float.astype(NP_INT_DTYPE)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("'layers' contains a value outside the integer range") from exc

    if np.any(layers <= 0):
        raise ValueError("'layers' must contain only positive integers")
    if np.unique(layers).size != layers.size:
        raise ValueError("'layers' contains duplicate entries")

    threshold_key = "thresholds" if "thresholds" in result else "tolerances"
    try:
        thresholds = np.asarray(
            result[threshold_key],
            dtype=NP_REAL_DTYPE,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"'{threshold_key}' must be a numeric one-dimensional array"
        ) from exc

    if thresholds.ndim != 1:
        raise ValueError(f"'{threshold_key}' must be one-dimensional")
    if (
        thresholds.size != SUCCESS_PROBABILITY_FIGURE_THRESHOLDS.size
        or not np.all(np.isfinite(thresholds))
        or np.any(thresholds <= 0.0)
    ):
        raise ValueError(
            f"'{threshold_key}' must contain the "
            f"{SUCCESS_PROBABILITY_FIGURE_THRESHOLDS.size} configured positive "
            "finite accuracy levels"
        )

    if "thresholds" in result and "tolerances" in result:
        try:
            tolerance_alias = np.asarray(
                result["tolerances"],
                dtype=NP_REAL_DTYPE,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "'tolerances' must be a numeric one-dimensional array"
            ) from exc

        if (
            tolerance_alias.shape != np.asarray(result["thresholds"]).shape
            or not np.allclose(
                tolerance_alias,
                np.asarray(result["thresholds"], dtype=NP_REAL_DTYPE),
                rtol=1e-12,
                atol=0.0,
            )
        ):
            raise ValueError(
                "'thresholds' and its 'tolerances' alias are inconsistent"
            )

    threshold_order = []
    for expected_threshold in SUCCESS_PROBABILITY_FIGURE_THRESHOLDS:
        matching = np.flatnonzero(
            np.isclose(
                thresholds,
                expected_threshold,
                rtol=1e-12,
                atol=0.0,
            )
        )
        if matching.size != 1:
            raise ValueError(
                f"'{threshold_key}' must contain exactly one "
                f"{expected_threshold:.8g} entry"
            )
        threshold_order.append(int(matching[0]))

    try:
        num_trials_array = np.asarray(
            result["num_trials"],
            dtype=NP_REAL_DTYPE,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("'num_trials' must be a numeric scalar") from exc

    if (
        num_trials_array.size != 1
        or not np.isfinite(num_trials_array.reshape(-1)[0])
        or num_trials_array.reshape(-1)[0]
        != np.rint(num_trials_array.reshape(-1)[0])
    ):
        raise ValueError("'num_trials' must be one finite integer scalar")
    num_trials = int(num_trials_array.reshape(-1)[0])

    expected_shape = (layers.size, thresholds.size)
    try:
        success_probabilities = np.asarray(
            result["success_probabilities"],
            dtype=NP_REAL_DTYPE,
        )
        success_counts_float = np.asarray(
            result["success_counts"],
            dtype=NP_REAL_DTYPE,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "'success_probabilities' and 'success_counts' must be numeric arrays"
        ) from exc

    if success_probabilities.shape != expected_shape:
        raise ValueError(
            "'success_probabilities' has shape "
            f"{success_probabilities.shape}; expected {expected_shape}"
        )
    if success_counts_float.shape != expected_shape:
        raise ValueError(
            f"'success_counts' has shape {success_counts_float.shape}; "
            f"expected {expected_shape}"
        )
    if not np.all(np.isfinite(success_probabilities)):
        raise ValueError("'success_probabilities' contains a non-finite value")

    probability_slack = 1e-12
    if np.any(success_probabilities < -probability_slack) or np.any(
        success_probabilities > 1.0 + probability_slack
    ):
        raise ValueError(
            "'success_probabilities' contains a value outside [0, 1]"
        )
    success_probabilities = np.clip(success_probabilities, 0.0, 1.0)

    if (
        not np.all(np.isfinite(success_counts_float))
        or not np.all(success_counts_float == np.rint(success_counts_float))
    ):
        raise ValueError("'success_counts' must contain only finite integers")
    if np.any(success_counts_float < 0) or np.any(
        success_counts_float > num_trials
    ):
        raise ValueError(
            "'success_counts' contains a value outside [0, num_trials]"
        )
    success_counts = success_counts_float.astype(NP_INT_DTYPE)

    if not np.allclose(
        success_probabilities,
        success_counts.astype(NP_REAL_DTYPE) / num_trials,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "'success_probabilities' is inconsistent with "
            "'success_counts / num_trials'"
        )

    threshold_order = np.asarray(threshold_order, dtype=NP_INT_DTYPE)
    thresholds = thresholds[threshold_order]
    success_probabilities = success_probabilities[:, threshold_order]
    success_counts = success_counts[:, threshold_order]

    # A smaller tolerance cannot have more successes than a larger one.
    if np.any(np.diff(success_counts, axis=1) > 0):
        raise ValueError(
            "'success_counts' is not monotone across decreasing tolerances"
        )

    layer_order = np.argsort(layers, kind="stable")
    return (
        layers[layer_order],
        thresholds,
        success_probabilities[layer_order],
        success_counts[layer_order],
        num_trials,
    )


def _success_probability_threshold_label(threshold: float) -> str:
    """Format the requested decade-spanning tolerances exactly."""
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


def plot_success_probability_multiple_tolerances(
    layers,
    thresholds,
    success_probabilities,
    *,
    num_trials: int,
    outpath: str,
    only_threshold: Optional[float] = None,
) -> None:
    """Plot one empirical VQE success-probability curve per tolerance."""
    layers = np.asarray(layers, dtype=NP_INT_DTYPE)
    thresholds = np.asarray(thresholds, dtype=NP_REAL_DTYPE)
    success_probabilities = np.asarray(
        success_probabilities,
        dtype=NP_REAL_DTYPE,
    )
    expected_shape = (layers.size, thresholds.size)
    if success_probabilities.shape != expected_shape:
        raise ValueError(
            "Success-probability array has shape "
            f"{success_probabilities.shape}; expected {expected_shape}."
        )

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
        success_probabilities = success_probabilities[:, matching]

    colors = matplotlib.colormaps.get_cmap("viridis")(
        np.linspace(0.08, 0.92, thresholds.size)
    )
    markers = ("o", "s", "^", "D", "P", "v", "X")
    linestyles = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))

    fig, ax = new_fig_ax(outside_legend=True)

    threshold_plot_order = np.argsort(thresholds, kind="stable")
    for plot_index, threshold_index in enumerate(threshold_plot_order):
        threshold = thresholds[threshold_index]
        ax.plot(
            layers,
            success_probabilities[:, threshold_index],
            color=colors[plot_index],
            marker=markers[plot_index % len(markers)],
            linestyle=linestyles[plot_index % len(linestyles)],
            linewidth=1.35,
            markersize=4.5,
            label=_success_probability_threshold_label(threshold),
            zorder=3,
        )

    ax.set_xlabel(r"Number of Layers $L$")
    ax.set_ylabel(
        r"Empirical success probability $\widehat{S}_L(\delta)$"
    )
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
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.set_xticklabels([str(int(layer)) for layer in layers])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        title=rf"$R={int(num_trials)}$ independent trials",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    save_fig(fig, ax, outpath, outside_legend=True)


def render_success_probability_multiple_tolerances_figure(
    *,
    result_path: str,
    outpath: str,
    delta_one_outpath: Optional[str] = None,
) -> bool:
    """Load, validate, and render the optional success-probability figures."""
    if not os.path.exists(result_path):
        _warn_skip_success_probability_figure(
            f"result file was not found: {result_path}"
        )
        return False

    try:
        result = load_npz_result(result_path)
    except (OSError, ValueError, KeyError) as exc:
        _warn_skip_success_probability_figure(
            f"could not load result file {result_path}: {exc}"
        )
        return False

    recompute_error = None
    try:
        result_for_figure = _success_probability_result_for_figure(result)
    except (TypeError, ValueError, OverflowError) as exc:
        # Backward compatibility for summary-only archives that already use
        # the configured figure thresholds.
        recompute_error = exc
        result_for_figure = result

    try:
        (
            layers,
            thresholds,
            success_probabilities,
            _success_counts,
            num_trials,
        ) = _validated_success_probability_data(result_for_figure)
    except (TypeError, ValueError, OverflowError) as exc:
        recompute_detail = (
            ""
            if recompute_error is None
            else f"; could not recompute from final energies: {recompute_error}"
        )
        _warn_skip_success_probability_figure(
            f"invalid archive {result_path}: {exc}{recompute_detail}"
        )
        return False

    plot_success_probability_multiple_tolerances(
        layers,
        thresholds,
        success_probabilities,
        num_trials=num_trials,
        outpath=outpath,
    )
    if delta_one_outpath is not None:
        plot_success_probability_multiple_tolerances(
            layers,
            thresholds,
            success_probabilities,
            num_trials=num_trials,
            outpath=delta_one_outpath,
            only_threshold=1.0,
        )
    return True


# ==============================
# Circuit blocks TensorCircuit
# ==============================
# ==============================
# TC -> Qiskit drawing
# ==============================
PARAMETER_FREE_GATE_LABELS = {
    "rz": r"$R_z$",
    "rx": r"$R_x$",
    "ry": r"$R_y$",
    "rxx": r"$R_{xx}$",
    "crx": r"$CR_x$",
    "cry": r"$CR_y$",
    "crz": r"$CR_z$",
    "cx": r"$CX$",
    "x": r"$X$",
    "z": r"$Z$",
    "h": r"$H$",
}


# ==============================
# Hamiltonian & ground truth
# ==============================
H_terms = tuple(hamiltonian_terms(h_param))

H_matrix = build_H_matrix_jax(H_terms, num_system_qubits)

smallest_eigval = float(
    np.linalg.eigvalsh(np.array(H_matrix, dtype=NP_COMPLEX_DTYPE)).min().real
)


# ==============================
# Parameter wrapping
# ==============================
# ============================================================
# Sequential trace-out machinery
# ============================================================
I2 = jnp.eye(2, dtype=COMPLEX_DTYPE)
X2 = jnp.array([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)
P0 = jnp.array([[1, 0], [0, 0]], dtype=COMPLEX_DTYPE)
P1 = jnp.array([[0, 0], [0, 1]], dtype=COMPLEX_DTYPE)


_U_CX = jnp.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=COMPLEX_DTYPE,
)


_RHO_KEEP_INIT = rho_zero_state(num_system_qubits, dtype=COMPLEX_DTYPE)


# ==============================
# Optimization loop per layer
# ==============================
success_rates_history = {}

final_stats = {
    "layer": [],
    "success_rate": [],
    "mean_energy": [],
    "std_energy": [],
}

# ------------------------------------------------------------
# Layer schedules
# ------------------------------------------------------------
vqe_dense_until_layer = cfg.VQE_DENSE_UNTIL_LAYER
vqe_max_layer = cfg.VQE_MAX_LAYER
vqe_sparse_step = cfg.VQE_SPARSE_STEP

vqe_layer_list = build_layer_list(
    vqe_max_layer,
    vqe_dense_until_layer,
    vqe_sparse_step,
)

qfim_dense_until_layer = cfg.DPQC_QFIM_DENSE_UNTIL_LAYER
qfim_max_layer = cfg.DPQC_QFIM_MAX_LAYER
qfim_sparse_step = cfg.DPQC_QFIM_SPARSE_STEP

qfim_layer_list = build_layer_list(
    qfim_max_layer,
    qfim_dense_until_layer,
    qfim_sparse_step,
)

if not vqe_layer_list:
    raise ValueError(
        "vqe_layer_list is empty. Check vqe_max_layer, "
        "vqe_dense_until_layer, and vqe_sparse_step."
    )

if not qfim_layer_list:
    raise ValueError(
        "qfim_layer_list is empty. Check DPQC_QFIM_MAX_LAYER, "
        "DPQC_QFIM_DENSE_UNTIL_LAYER, and DPQC_QFIM_SPARSE_STEP."
    )

save_dir = f"./figs/{output_family}/h_{h_param}"

if output_family == "dpqc_reset":
    reset_metadata_path = os.path.join(
        save_dir,
        "reset_model_metadata.json",
    )
    if not os.path.isfile(reset_metadata_path):
        raise FileNotFoundError(
            "Reset-DPQC metadata was not found: "
            f"{reset_metadata_path}"
        )
    with open(reset_metadata_path, "r", encoding="utf-8") as metadata_file:
        reset_metadata = json.load(metadata_file)
    if reset_metadata.get("model_id") != "dpqc_reset_fixed_rx_pi":
        raise ValueError(
            "Reset-DPQC metadata has an incompatible model_id: "
            f"{reset_metadata.get('model_id')!r}."
        )
    metadata_h_param = float(reset_metadata.get("h_param", math.nan))
    if not math.isclose(
        metadata_h_param,
        h_param,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "Reset-DPQC metadata h_param does not match --h-param: "
            f"{metadata_h_param} != {h_param}."
        )
    fixed_rx_angle = float(
        reset_metadata.get("fixed_feed_forward_rx_angle", math.nan)
    )
    if not math.isclose(
        fixed_rx_angle,
        math.pi,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "Reset-DPQC metadata does not specify fixed Rx(pi)."
        )

energy_fig_dir = os.path.join(save_dir, "energy_figures")
qfim_fig_dir = os.path.join(save_dir, "qfim_figures")
qfim_trace_dir = os.path.join(qfim_fig_dir, "qfim_trace")
qfim_eigs_dir = os.path.join(qfim_fig_dir, "qfim_eigs")
qfim_eigs_dir_red4 = os.path.join(qfim_eigs_dir, "reduced_keep_0123")
qfim_eigs_dir_red5 = os.path.join(qfim_eigs_dir, "reduced_keep_01234")
qfim_eigcount_dir = os.path.join(qfim_fig_dir, "qfim_eigcount")
qfim_eigcount_random_dir = os.path.join(qfim_eigcount_dir, "random_points")
qfim_eigcount_optimization_path_dir = os.path.join(
    qfim_eigcount_dir,
    "optimization_path",
)
qfim_eigcount_optimization_path_mean_dir = os.path.join(
    qfim_eigcount_optimization_path_dir,
    "mean",
)
qfim_eigcount_optimization_path_min_dir = os.path.join(
    qfim_eigcount_optimization_path_dir,
    "min",
)
qfim_effective_rank_dir = os.path.join(qfim_fig_dir, "effective_rank")
circuit_dir = os.path.join(save_dir, "optimized_circuits")
numerical_results_dir = os.path.join(save_dir, "numerical_results")
energy_results_dir = os.path.join(numerical_results_dir, "energy")
qfim_results_dir = os.path.join(numerical_results_dir, "qfim")

def _load_layer_arrays_from_npz(
    result: dict,
    layers,
    suffix: Optional[str] = None,
    *,
    dtype=None,
) -> dict:
    arrays_by_layer = {}

    for L in layers:
        L_int = int(L)
        key = f"L{L_int}" if suffix is None else f"L{L_int}_{suffix}"

        if key not in result:
            continue

        arr = np.asarray(result[key])
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        arrays_by_layer[L_int] = arr

    return arrays_by_layer


cmap = matplotlib.colormaps.get_cmap("viridis")


# ============================================================
# Load VQE numerical results and generate energy figures
# ============================================================
vqe_optimization_result_path = os.path.join(
    energy_results_dir,
    "vqe_optimization_histories.npz",
)

vqe_optimization_results = load_npz_result(vqe_optimization_result_path)
_validate_result_h_param(
    vqe_optimization_results,
    vqe_optimization_result_path,
    required=True,
)

if "optimizer_name" in vqe_optimization_results:
    archived_optimizer = np.asarray(
        vqe_optimization_results["optimizer_name"]
    )
    if archived_optimizer.shape != ():
        raise ValueError(
            "optimizer_name must be a scalar string in the VQE archive: "
            f"{vqe_optimization_result_path}"
        )
    archived_optimizer_value = archived_optimizer.item()
    if isinstance(archived_optimizer_value, bytes):
        archived_optimizer_value = archived_optimizer_value.decode("utf-8")
    vqe_optimizer_name = normalize_dpqc_vqe_optimizer_name(
        archived_optimizer_value
    )
else:
    # Every archive produced before optimizer provenance was added used Adam.
    vqe_optimizer_name = "adam"

vqe_optimizer_label = dpqc_vqe_optimizer_display_name(vqe_optimizer_name)
vqe_optimizer_title_suffix = f" ({vqe_optimizer_label})"
print(f"Loaded VQE optimizer: {vqe_optimizer_label}")

# Create output directories only after confirming that the selected h has a
# valid VQE archive.  Raw numerical archives are treated as inputs; the only
# numerical file written below is the derived convergence-time statistics.
_output_dirs = (
    save_dir,
    energy_fig_dir,
    qfim_fig_dir,
    qfim_eigs_dir,
    qfim_eigs_dir_red4,
    qfim_eigs_dir_red5,
    qfim_eigcount_dir,
    qfim_eigcount_random_dir,
    qfim_effective_rank_dir,
    circuit_dir,
)
if INCLUDE_QFIM_TRACE_FIGURES:
    _output_dirs += (qfim_trace_dir,)
if INCLUDE_OPTIMIZATION_PATH_QFIM:
    _output_dirs += (
        qfim_eigcount_optimization_path_dir,
        qfim_eigcount_optimization_path_mean_dir,
        qfim_eigcount_optimization_path_min_dir,
    )
for _output_dir in _output_dirs:
    os.makedirs(_output_dir, exist_ok=True)

vqe_layer_list = [
    int(L)
    for L in np.asarray(vqe_optimization_results["vqe_layers"], dtype=NP_INT_DTYPE)
]
qfim_layer_list = [
    int(L)
    for L in np.asarray(vqe_optimization_results["qfim_layers"], dtype=NP_INT_DTYPE)
]
sample_iters = np.asarray(vqe_optimization_results["sample_iters"], dtype=NP_INT_DTYPE)
steps = int(np.asarray(vqe_optimization_results["steps"]).item())
num_runs = int(np.asarray(vqe_optimization_results["num_runs"]).item())
smallest_eigval = NP_REAL_DTYPE(
    np.asarray(vqe_optimization_results["smallest_eigval"]).item()
)

energy_traces_by_layer = _load_layer_arrays_from_npz(
    vqe_optimization_results,
    vqe_layer_list,
    "energy_traces",
    dtype=NP_REAL_DTYPE,
)
grad_norm_traces_by_layer = _load_layer_arrays_from_npz(
    vqe_optimization_results,
    vqe_layer_list,
    "grad_norm_traces",
    dtype=NP_REAL_DTYPE,
)
theta_history = _load_layer_arrays_from_npz(
    vqe_optimization_results,
    vqe_layer_list,
    "theta_final",
    dtype=NP_REAL_DTYPE,
)
success_rates_history = _load_layer_arrays_from_npz(
    vqe_optimization_results,
    vqe_layer_list,
    "success_rates",
    dtype=NP_REAL_DTYPE,
)
final_theta_periodic_only_rmsdist_by_layer = _load_layer_arrays_from_npz(
    vqe_optimization_results,
    vqe_layer_list,
    "theta_rmsdist",
    dtype=NP_REAL_DTYPE,
)
final_stats = {
    "layer": np.asarray(vqe_optimization_results["final_stats_layer"], dtype=NP_INT_DTYPE),
    "success_rate": np.asarray(vqe_optimization_results["final_stats_success_rate"], dtype=NP_REAL_DTYPE),
    "mean_energy": np.asarray(vqe_optimization_results["final_stats_mean_energy"], dtype=NP_REAL_DTYPE),
    "std_energy": np.asarray(vqe_optimization_results["final_stats_std_energy"], dtype=NP_REAL_DTYPE),
}

convergence_time_statistics = generate_convergence_time_outputs(
    energy_traces_by_layer,
    vqe_layer_list,
    ground_energy=smallest_eigval,
    tolerances=getattr(_CLI_ARGS, "convergence_tolerances", None),
    num_runs=num_runs,
    optimizer_steps=steps,
    figure_dir=energy_fig_dir,
    statistics_outpath=os.path.join(
        energy_results_dir,
        "vqe_convergence_time_statistics.npz",
    ),
    metadata={
        "h_param": h_param,
        "architecture": output_family,
        "optimizer_name": vqe_optimizer_name,
        "source_archive": os.path.basename(vqe_optimization_result_path),
    },
)


# ==============================
# Plotting & saving figures
# ==============================
os.makedirs(save_dir, exist_ok=True)
os.makedirs(energy_fig_dir, exist_ok=True)
os.makedirs(qfim_fig_dir, exist_ok=True)

x_groups = np.arange(len(sample_iters), dtype=NP_REAL_DTYPE)
num_vqe_layers = len(vqe_layer_list)
offset_span = 0.75
box_width = 0.85 * (offset_span / num_vqe_layers)

plot_history_violin(
    energy_traces_by_layer,
    vqe_layer_list,
    sample_iters=sample_iters,
    x_groups=x_groups,
    offset_span=offset_span,
    violin_width=box_width,
    cmap=cmap,
    ylabel="Absolute energy error",
    title=f"Loss history over {num_runs} runs{vqe_optimizer_title_suffix}",
    outpath=os.path.join(energy_fig_dir, "loss_history.pdf"),
    transform=lambda x: np.abs(x - smallest_eigval) + eps,
)

plot_history_violin(
    grad_norm_traces_by_layer,
    vqe_layer_list,
    sample_iters=sample_iters,
    x_groups=x_groups,
    offset_span=offset_span,
    violin_width=box_width,
    cmap=cmap,
    ylabel="Gradient norm",
    title=(
        f"Gradient-norm history over {num_runs} runs"
        f"{vqe_optimizer_title_suffix}"
    ),
    outpath=os.path.join(energy_fig_dir, "grad_norm_history.pdf"),
    transform=lambda x: x + eps,
)

final_energy_error_by_layer = [
    np.abs(
        np.asarray(
            energy_traces_by_layer[L][:, -1],
            dtype=NP_REAL_DTYPE,
        )
        - smallest_eigval
    )
    for L in vqe_layer_list
]

plot_violin_by_layer(
    final_energy_error_by_layer,
    vqe_layer_list,
    cmap=cmap,
    ylabel="Final energy error",
    title=f"Final energy-error distributions{vqe_optimizer_title_suffix}",
    outpath=os.path.join(energy_fig_dir, "final_energy_error.pdf"),
    show_legend=False,
    log_scale=False,
)

plot_beeswarm_by_layer(
    final_energy_error_by_layer,
    vqe_layer_list,
    cmap=cmap,
    ylabel="Final energy error",
    title=f"Final energy-error distributions{vqe_optimizer_title_suffix}",
    outpath=os.path.join(energy_fig_dir, "final_energy_error_beeswarm.pdf"),
    point_size=18.0,
    alpha=0.65,
    max_width=0.32,
    log_scale=False,
)

plot_beeswarm_by_layer(
    [
        np.maximum(err, eps)
        for err in final_energy_error_by_layer
    ],
    vqe_layer_list,
    cmap=cmap,
    ylabel="Final energy error",
    title=f"Final energy-error distributions{vqe_optimizer_title_suffix}",
    outpath=os.path.join(energy_fig_dir, "final_energy_error_beeswarm_logscale.pdf"),
    point_size=18.0,
    alpha=0.65,
    max_width=0.32,
    log_scale=True,
    eps=eps,
)

final_energy_error_detail_threshold = float(
    FINAL_ENERGY_ERROR_DETAIL_THRESHOLD
)

if (
    not np.isfinite(final_energy_error_detail_threshold)
    or final_energy_error_detail_threshold <= 0.0
):
    raise ValueError(
        "FINAL_ENERGY_ERROR_DETAIL_THRESHOLD must be finite and positive."
    )

final_energy_error_detail_by_layer = [
    err[
        np.isfinite(err)
        & (err <= final_energy_error_detail_threshold)
    ]
    for err in final_energy_error_by_layer
]

if not any(values.size for values in final_energy_error_detail_by_layer):
    warnings.warn(
        "No final energy errors satisfy "
        f"error <= {final_energy_error_detail_threshold:g}; "
        "skipping the detailed beeswarm figure.",
        RuntimeWarning,
        stacklevel=1,
    )
else:
    plot_beeswarm_by_layer(
        final_energy_error_detail_by_layer,
        vqe_layer_list,
        cmap=cmap,
        ylabel=(
            "Final energy error "
            rf"($\Delta E \leq {final_energy_error_detail_threshold:g}$)"
        ),
        title=(
            "Detailed final energy-error distributions "
            rf"($\Delta E \leq {final_energy_error_detail_threshold:g}$)"
            f"{vqe_optimizer_title_suffix}"
        ),
        outpath=os.path.join(
            energy_fig_dir,
            (
                "final_energy_error_beeswarm_below_"
                f"{_thr_tag(final_energy_error_detail_threshold)}.pdf"
            ),
        ),
        point_size=18.0,
        alpha=0.65,
        max_width=0.32,
        log_scale=False,
    )

plot_single_line_by_layer(
    np.array(final_stats["layer"], dtype=NP_REAL_DTYPE),
    np.array(final_stats["success_rate"], dtype=NP_REAL_DTYPE),
    ylabel="Success Rate",
    title=(
        f"Success Rate over {num_runs} runs (Tol={tolerance})"
        f"{vqe_optimizer_title_suffix}"
    ),
    outpath=os.path.join(energy_fig_dir, "success_rate.pdf"),
    label="Success Rate",
    ylim=(-0.02, 1.02),
)

render_success_probability_multiple_tolerances_figure(
    result_path=os.path.join(
        energy_results_dir,
        "vqe_success_probability_multiple_tolerances.npz",
    ),
    outpath=os.path.join(
        energy_fig_dir,
        "success_probability_multiple_tolerances.pdf",
    ),
    delta_one_outpath=os.path.join(
        energy_fig_dir,
        "success_probability_delta_1.pdf",
    ),
)

# ============================================================
# Load random-parameter QFIM results and generate QFIM figures
# ============================================================
# QFIM eigenvalue, trace, and eigenvalue-count plots for both retained subsystems.
# ============================================================
QFIM_EIG_PLOT_EPS = cfg.QFIM_EIG_PLOT_EPS
NUM_QFIM_SAMPLES = cfg.NUM_QFIM_SAMPLES
QFIM_TRACE_EIGENVALUE_THRESHOLD = NP_REAL_DTYPE(
    cfg.QFIM_EFFECTIVE_RANK_THRESHOLD
)

QFIM_PATH_EIGCOUNT_THRESHOLDS = tuple(
    float(t) for t in cfg.QFIM_PATH_EIGCOUNT_THRESHOLDS
)


def qfim_trace_at_or_above_rank_threshold(eigenvalues: np.ndarray) -> np.ndarray:
    """Sum eigenvalues at/above the cutoff; preserve invalid spectra as NaN."""
    eigs = np.asarray(eigenvalues, dtype=NP_REAL_DTYPE)
    if eigs.ndim == 0:
        raise ValueError("QFIM eigenvalues must have an eigenvalue axis.")

    trace = np.sum(
        np.where(
            eigs >= QFIM_TRACE_EIGENVALUE_THRESHOLD,
            eigs,
            NP_REAL_DTYPE(0.0),
        ),
        axis=-1,
        dtype=NP_REAL_DTYPE,
    )
    finite_spectrum = np.all(np.isfinite(eigs), axis=-1)
    return np.where(finite_spectrum, trace, NP_REAL_DTYPE(np.nan))


def eigenvalue_index_ticks(n_params: int, *, max_ticks: int = 11) -> np.ndarray:
    n_params = int(n_params)

    if n_params <= 0:
        return np.asarray([], dtype=NP_INT_DTYPE)

    max_ticks = max(2, int(max_ticks))

    if n_params <= max_ticks:
        return np.arange(1, n_params + 1, dtype=NP_INT_DTYPE)

    ticks = np.rint(
        np.linspace(1, n_params, num=max_ticks)
    ).astype(NP_INT_DTYPE)

    ticks = np.unique(ticks)
    ticks[0] = 1
    ticks[-1] = n_params

    if ticks.size >= 3:
        gaps = np.diff(ticks).astype(NP_REAL_DTYPE)
        typical_gap = float(np.median(gaps))

        if typical_gap > 0.0 and float(gaps[-1]) < 0.60 * typical_gap:
            ticks = np.delete(ticks, -2)

    return ticks


def save_qfim_eigs_by_index(
    eigs_sorted_desc: np.ndarray,
    *,
    title: str,
    outpath: str,
    eps: float = QFIM_EIG_PLOT_EPS,
    point_size: float = 14.0,
    alpha: float = 0.55,
) -> None:
    eigs_raw = np.asarray(eigs_sorted_desc, dtype=NP_REAL_DTYPE)

    if eigs_raw.ndim == 1:
        eigs_raw = eigs_raw[None, :]

    eigs_plot = eigs_raw.copy()
    eigs_plot[eigs_plot <= 0.0] = eps

    n_params = int(eigs_plot.shape[1])

    fig, ax = new_fig_ax(outside_legend=False)

    positions = np.arange(1, n_params + 1, dtype=NP_REAL_DTYPE)

    for i, x0 in enumerate(positions):
        y_plot = eigs_plot[:, i]
        ax.scatter(
            np.full_like(y_plot, x0, dtype=NP_REAL_DTYPE),
            y_plot,
            s=point_size,
            color="C0",
            alpha=alpha,
            edgecolors="black",
            linewidths=0.20,
            rasterized=True,
        )

    ticks = eigenvalue_index_ticks(n_params, max_ticks=11)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(0.5, n_params + 0.5)
    ax.set_yscale("log")

    ymin = float(np.min(eigs_plot))
    ymax = float(np.max(eigs_plot))
    ymin = max(eps, ymin)

    if ymax > ymin:
        ax.set_ylim(max(eps, ymin / 1.5), ymax * 1.5)

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("QFIM eigenvalue")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    save_fig(fig, ax, outpath, outside_legend=False)


def save_qfim_eigs_by_index_colored_by_layer(
    eigs_sorted_desc_by_layer: dict,
    layer_list,
    *,
    title: str,
    outpath: str,
    eps: float = QFIM_EIG_PLOT_EPS,
    cmap=None,
    alpha: float = 0.50,
    point_size: float = 10.0,
) -> None:
    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap

    layers = [
        int(L)
        for L in layer_list
        if eigs_sorted_desc_by_layer.get(L) is not None
    ]

    if not layers:
        return

    max_n_params = max(
        int(
            np.asarray(
                eigs_sorted_desc_by_layer[L],
                dtype=NP_REAL_DTYPE,
            ).shape[1]
        )
        for L in layers
    )

    fig, ax = new_fig_ax(outside_legend=True)

    handles = []

    for idx, L in enumerate(layers):
        color = cmap(idx / len(layers))

        handles.append(
            Patch(
                facecolor=color,
                edgecolor=color,
                alpha=0.25,
                label=f"L{L}",
            )
        )

        eigs = np.asarray(
            eigs_sorted_desc_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )

        for i in range(max_n_params):
            if i < eigs.shape[1]:
                v = eigs[:, i].astype(float)
                v[v <= 0.0] = eps
                x = np.full_like(v, i + 1, dtype=NP_REAL_DTYPE)
                ax.scatter(
                    x,
                    v,
                    s=point_size,
                    color=color,
                    alpha=alpha,
                    edgecolors="black",
                    linewidths=0.15,
                    rasterized=True,
                )

    ticks = eigenvalue_index_ticks(max_n_params, max_ticks=11)

    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(0.5, max_n_params + 0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("QFIM eigenvalue")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.85,
    )

    save_fig(fig, ax, outpath, outside_legend=True)


qfim_random_points_result_path = os.path.join(
    qfim_results_dir,
    f"qfim_random_points_{keep_key}.npz",
)

qfim_random_points_results = load_npz_result(qfim_random_points_result_path)
qfim_layer_list = [
    int(L)
    for L in np.asarray(qfim_random_points_results["layers"], dtype=NP_INT_DTYPE)
]
qfim_eigs_reduced_0123_by_layer = _load_layer_arrays_from_npz(
    qfim_random_points_results,
    qfim_layer_list,
    "eigs_desc",
    dtype=NP_REAL_DTYPE,
)
qfim_thresholded_trace_reduced_0123_by_layer = {
    int(L): qfim_trace_at_or_above_rank_threshold(eigs)
    for L, eigs in qfim_eigs_reduced_0123_by_layer.items()
}
qfim_abs_entry_sum_reduced_0123_by_layer = _load_layer_arrays_from_npz(
    qfim_random_points_results,
    qfim_layer_list,
    "abs_entry_sum",
    dtype=NP_REAL_DTYPE,
)

for L in qfim_layer_list:
    save_qfim_eigs_by_index(
        qfim_eigs_reduced_0123_by_layer[L],
        title=rf"QFIM eigenvalues at {NUM_QFIM_SAMPLES} random points (L={L})",
        outpath=os.path.join(qfim_eigs_dir_red4, f"L{L}_reduced_0123.pdf"),
    )


if INCLUDE_QFIM_EIGS_BY_INDEX_LAYERS:
    save_qfim_eigs_by_index_colored_by_layer(
        qfim_eigs_reduced_0123_by_layer,
        qfim_layer_list,
        title=rf"QFIM eigenvalues at {NUM_QFIM_SAMPLES} random points",
        outpath=os.path.join(
            qfim_eigs_dir,
            "qfim_eigs_by_index_layers_reduced_keep_0123.pdf",
        ),
        cmap=cmap,
    )


def _qfim_threshold_tex_for_label(threshold: float) -> str:
    threshold = float(threshold)

    if threshold <= 0.0:
        return f"{threshold:g}"

    exp = int(np.floor(np.log10(threshold)))
    mant = threshold / (10.0 ** exp)

    if np.isclose(mant, 1.0):
        return rf"10^{{{exp}}}"

    return rf"{mant:g}\times 10^{{{exp}}}"


QFIM_TRACE_THRESHOLD_TEX = _qfim_threshold_tex_for_label(
    QFIM_TRACE_EIGENVALUE_THRESHOLD
)
QFIM_TRACE_YLABEL = (
    rf"QFIM trace ($\lambda_i \geq {QFIM_TRACE_THRESHOLD_TEX}$)"
)
QFIM_TRACE_MEAN_YLABEL = (
    rf"Mean QFIM trace ($\lambda_i \geq {QFIM_TRACE_THRESHOLD_TEX}$)"
)
QFIM_TRACE_SUM_LABEL = (
    rf"$\sum_{{\lambda_k \geq {QFIM_TRACE_THRESHOLD_TEX}}}\lambda_k(F)$"
)


def qfim_random_eigcount_by_threshold_by_layer(
    eigs_by_layer: dict,
    layers,
    thresholds,
) -> dict:
    eigcount_by_threshold = {}

    for threshold in thresholds:
        threshold = float(threshold)
        count_by_layer = {}

        for L in layers:
            L_int = int(L)
            if eigs_by_layer.get(L_int) is None:
                continue

            eigs_L = np.asarray(eigs_by_layer[L_int], dtype=NP_REAL_DTYPE)

            if eigs_L.ndim != 2:
                raise ValueError(
                    "Each random-point QFIM eigenvalue array must be 2D: "
                    "(num_samples, num_params)."
                )

            count_by_layer[L_int] = np.sum(
                eigs_L >= threshold,
                axis=1,
            ).astype(NP_REAL_DTYPE)

        eigcount_by_threshold[threshold] = count_by_layer

    return eigcount_by_threshold


def plot_qfim_random_eigcount_threshold_overlay(
    eigs_by_layer: dict,
    layers,
    thresholds,
    *,
    title: str,
    outpath: str,
    cmap=None,
):
    thresholds = tuple(float(thr) for thr in thresholds)

    if not thresholds:
        return

    eigcount_by_threshold = qfim_random_eigcount_by_threshold_by_layer(
        eigs_by_layer,
        layers,
        thresholds,
    )

    valid_layers = [
        int(L)
        for L in layers
        if any(
            eigcount_by_threshold[threshold].get(int(L)) is not None
            for threshold in thresholds
        )
    ]

    if not valid_layers:
        return

    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)
    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap

    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.24)
    n_thresholds = len(thresholds)

    for threshold_idx, threshold in enumerate(thresholds):
        means = []
        sems = []
        layers_for_threshold = []

        for L in valid_layers:
            counts = eigcount_by_threshold[threshold].get(L)

            if counts is None:
                continue

            counts = np.asarray(counts, dtype=NP_REAL_DTYPE).reshape(-1)
            counts = counts[np.isfinite(counts)]

            if counts.size == 0:
                continue

            layers_for_threshold.append(L)
            means.append(NP_REAL_DTYPE(np.mean(counts)))
            sems.append(
                NP_REAL_DTYPE(0.0)
                if counts.size <= 1
                else NP_REAL_DTYPE(np.std(counts, ddof=1) / np.sqrt(counts.size))
            )

        if not layers_for_threshold:
            continue

        x_thr = np.asarray(layers_for_threshold, dtype=NP_REAL_DTYPE)
        color = cmap(threshold_idx / max(n_thresholds - 1, 1))
        label = rf"$\lambda_i \geq {_qfim_threshold_tex_for_label(threshold)}$"

        ax.errorbar(
            x_thr,
            np.asarray(means, dtype=NP_REAL_DTYPE),
            yerr=np.asarray(sems, dtype=NP_REAL_DTYPE),
            marker="o",
            linestyle="-",
            linewidth=1.2,
            markersize=4.8,
            capsize=3.0,
            elinewidth=0.8,
            color=color,
            label=label,
        )

    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Mean QFIM eigenvalue count")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in valid_layers])
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
    )

    save_fig(
        fig,
        ax,
        outpath,
        outside_legend=True,
        legend_space_frac=0.24,
    )


plot_qfim_random_eigcount_threshold_overlay(
    qfim_eigs_reduced_0123_by_layer,
    qfim_layer_list,
    QFIM_PATH_EIGCOUNT_THRESHOLDS,
    title=rf"QFIM eigenvalue count at random points by threshold ({keep_label})",
    outpath=os.path.join(
        qfim_eigcount_random_dir,
        f"qfim_eigcount_threshold_overlay_random_points_{keep_key}.pdf",
    ),
    cmap=cmap,
)


# ============================================================
# Maximum QFIM trace + mean ± SEM by layer
#   The plotted value is the sum of QFIM eigenvalues at or above the rank
#   threshold at each random parameter point.
#   reduced keep=(0,1,2,3) only
# ============================================================
def _qfim_metric_max_mean_sem_xy(metric_by_layer: dict, layers):
    valid_items = []

    for L in layers:
        values_L = metric_by_layer.get(L)

        if values_L is None:
            continue

        values_arr = np.asarray(values_L, dtype=NP_REAL_DTYPE).reshape(-1)
        values_arr = values_arr[np.isfinite(values_arr)]

        if values_arr.size == 0:
            continue

        valid_items.append((int(L), values_arr))

    if not valid_items:
        return (
            np.asarray([], dtype=NP_REAL_DTYPE),
            np.asarray([], dtype=NP_REAL_DTYPE),
            np.asarray([], dtype=NP_REAL_DTYPE),
            np.asarray([], dtype=NP_REAL_DTYPE),
            [],
        )

    valid_layers = [L for L, _ in valid_items]
    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)

    max_values = np.asarray(
        [np.max(values_arr) for _, values_arr in valid_items],
        dtype=NP_REAL_DTYPE,
    )

    mean_values = np.asarray(
        [np.mean(values_arr) for _, values_arr in valid_items],
        dtype=NP_REAL_DTYPE,
    )

    sem_values = np.asarray(
        [
            NP_REAL_DTYPE(0.0)
            if values_arr.size <= 1
            else NP_REAL_DTYPE(np.std(values_arr, ddof=1) / np.sqrt(values_arr.size))
            for _, values_arr in valid_items
        ],
        dtype=NP_REAL_DTYPE,
    )

    return x, max_values, mean_values, sem_values, valid_layers


def plot_qfim_trace_max_mean_sem_by_layer(
    trace_by_layer: dict,
    layers,
    *,
    title: str,
    outpath: str,
    color_max="C0",
    color_mean="C1",
    marker_max: str = "o",
    marker_mean: str = "s",
    lw: float = 1.4,
    log_scale: bool = False,
):
    (
        x,
        max_values,
        mean_values,
        sem_values,
        valid_layers,
    ) = _qfim_metric_max_mean_sem_xy(trace_by_layer, layers)

    if x.size == 0:
        return

    fig, ax = new_fig_ax(outside_legend=False)

    ax.plot(
        x,
        max_values,
        marker=marker_max,
        linestyle="-",
        linewidth=lw,
        markersize=6.0,
        color=color_max,
        label="Maximum QFIM trace",
    )

    ax.errorbar(
        x,
        mean_values,
        yerr=sem_values,
        marker=marker_mean,
        linestyle="-",
        linewidth=lw,
        markersize=5.0,
        capsize=4.0,
        elinewidth=1.0,
        color=color_mean,
        label=r"Mean QFIM trace $\pm$ SEM",
    )

    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(QFIM_TRACE_YLABEL)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in valid_layers])

    if log_scale:
        ax.set_yscale("log")

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    save_fig(fig, ax, outpath, outside_legend=False)


if INCLUDE_QFIM_TRACE_FIGURES:
    plot_qfim_trace_max_mean_sem_by_layer(
        qfim_thresholded_trace_reduced_0123_by_layer,
        qfim_layer_list,
        title=(
            rf"QFIM trace maximum and mean $\pm$ SEM at "
            rf"{NUM_QFIM_SAMPLES} random points"
        ),
        outpath=os.path.join(
            qfim_trace_dir,
            "qfim_trace_max_mean_sem_random_points_reduced_0123.pdf",
        ),
        color_max="C0",
        color_mean="C1",
        marker_max="o",
        marker_mean="s",
        lw=1.4,
        log_scale=False,
    )


# ============================================================

# ============================================================
# Shared statistics for optimization-path traces
# ============================================================
def _finite_mean_sem_over_runs_by_time(values_2d: np.ndarray):
    values = np.asarray(values_2d, dtype=NP_REAL_DTYPE)

    if values.ndim == 1:
        values = values[None, :]

    valid = np.isfinite(values)
    counts = np.sum(valid, axis=0).astype(NP_REAL_DTYPE)
    sums = np.nansum(np.where(valid, values, 0.0), axis=0)

    means = np.divide(
        sums,
        counts,
        out=np.full(values.shape[1], np.nan, dtype=NP_REAL_DTYPE),
        where=counts > 0,
    )

    centered = np.where(valid, values - means[None, :], np.nan)
    sq = np.nansum(centered**2, axis=0)

    stds = np.sqrt(
        np.divide(
            sq,
            counts - 1.0,
            out=np.zeros_like(sq, dtype=NP_REAL_DTYPE),
            where=counts > 1,
        )
    )

    sems = np.divide(
        stds,
        np.sqrt(counts),
        out=np.zeros_like(stds, dtype=NP_REAL_DTYPE),
        where=counts > 1,
    )

    return means, sems, counts


# ============================================================

# ============================================================
# Load optimization-path QFIM results and generate path figures
# ============================================================
def plot_qfim_trace_history_mean_by_layer(
    trace_history_by_layer: dict,
    layers,
    sample_iters,
    *,
    title: str,
    outpath: str,
    ylabel: str = QFIM_TRACE_MEAN_YLABEL,
    metric_name: str = "QFIM trace",
    cmap=None,
    log_scale: bool = False,
):
    valid_layers = [
        int(L)
        for L in layers
        if trace_history_by_layer.get(int(L)) is not None
    ]

    if not valid_layers:
        return

    x = np.asarray(sample_iters, dtype=NP_REAL_DTYPE)
    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap

    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.22)
    n_layers = len(valid_layers)

    for layer_idx, L in enumerate(valid_layers):
        traces = np.asarray(trace_history_by_layer[L], dtype=NP_REAL_DTYPE)

        if traces.ndim != 2:
            raise ValueError(
                f"Each {metric_name} history array must be 2D: "
                "(num_runs, num_sample_iters)."
            )

        if traces.shape[1] != x.size and traces.shape[0] == x.size:
            traces = traces.T

        if traces.shape[1] != x.size:
            raise ValueError(
                f"Shape mismatch for L={L}: "
                f"traces.shape={traces.shape}, len(sample_iters)={x.size}."
            )

        means, sems, counts = _finite_mean_sem_over_runs_by_time(traces)
        finite_mask = np.isfinite(means) & (counts > 0)

        if not np.any(finite_mask):
            continue

        color = cmap(layer_idx / max(n_layers - 1, 1))

        ax.errorbar(
            x[finite_mask],
            means[finite_mask],
            yerr=sems[finite_mask],
            marker="o",
            linestyle="-",
            linewidth=1.2,
            markersize=4.5,
            capsize=3.0,
            elinewidth=0.8,
            color=color,
            label=f"L={L}",
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in x], rotation=45, ha="right")

    if log_scale:
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.3)
    else:
        ax.grid(True, axis="y", alpha=0.3)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
    )

    save_fig(
        fig,
        ax,
        outpath,
        outside_legend=True,
        legend_space_frac=0.22,
    )


qfim_eigs_history_optimization_path_by_layer = {}
qfim_trace_history_optimization_path_by_layer = {}
qfim_trace_history_layer_list = []

if INCLUDE_OPTIMIZATION_PATH_QFIM:
    qfim_eigs_history_result_path = os.path.join(
        qfim_results_dir,
        f"qfim_eigs_history_optimization_path_{keep_key}.npz",
    )
    qfim_eigs_history_results = load_npz_result(qfim_eigs_history_result_path)
    qfim_eigs_history_optimization_path_by_layer = _load_layer_arrays_from_npz(
        qfim_eigs_history_results,
        vqe_layer_list,
        suffix=None,
        dtype=NP_REAL_DTYPE,
    )

    qfim_trace_history_layer_list = list(vqe_layer_list)
    qfim_trace_history_optimization_path_by_layer = {
        int(L): qfim_trace_at_or_above_rank_threshold(eigs)
        for L, eigs in qfim_eigs_history_optimization_path_by_layer.items()
    }

    if INCLUDE_QFIM_TRACE_FIGURES:
        plot_qfim_trace_history_mean_by_layer(
            qfim_trace_history_optimization_path_by_layer,
            qfim_trace_history_layer_list,
            sample_iters,
            title=rf"Mean QFIM trace along optimization path ({keep_label})",
            outpath=os.path.join(
                qfim_trace_dir,
                f"qfim_trace_mean_history_optimization_path_{keep_key}.pdf",
            ),
            cmap=cmap,
            log_scale=False,
        )


def _sample_mean_sem(samples: np.ndarray) -> Tuple[NP_REAL_DTYPE, NP_REAL_DTYPE]:
    samples = np.asarray(samples, dtype=NP_REAL_DTYPE).reshape(-1)
    n = int(samples.size)

    if n == 0:
        return NP_REAL_DTYPE(0.0), NP_REAL_DTYPE(0.0)

    mean = NP_REAL_DTYPE(np.mean(samples))

    if n <= 1:
        sem = NP_REAL_DTYPE(0.0)
    else:
        sem = NP_REAL_DTYPE(np.std(samples, ddof=1) / np.sqrt(n))

    return mean, sem


def _mean_sem_arrays_from_by_layer(stats_by_layer: dict, layers):
    valid_layers = [
        int(L)
        for L in layers
        if stats_by_layer.get(L) is not None
    ]

    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)

    means = np.asarray(
        [stats_by_layer[L]["mean"] for L in valid_layers],
        dtype=NP_REAL_DTYPE,
    )

    sems = np.asarray(
        [stats_by_layer[L]["sem"] for L in valid_layers],
        dtype=NP_REAL_DTYPE,
    )

    return x, means, sems



def _threshold_tex(threshold: float) -> str:
    threshold = float(threshold)

    if threshold <= 0.0:
        return f"{threshold:g}"

    exp = int(np.floor(np.log10(threshold)))
    mant = threshold / (10.0 ** exp)

    if np.isclose(mant, 1.0):
        return rf"10^{{{exp}}}"

    return rf"{mant:g}\times 10^{{{exp}}}"


def qfim_eigcount_history_by_layer(
    eigs_history_by_layer: dict,
    layers,
    sample_iters_for_labels,
    *,
    threshold: float,
) -> dict:
    sample_iters_arr = np.asarray(sample_iters_for_labels, dtype=NP_INT_DTYPE)
    threshold = float(threshold)
    count_history_by_layer = {}

    for L in layers:
        L_int = int(L)
        if eigs_history_by_layer.get(L_int) is None:
            continue

        eigs_L = np.asarray(eigs_history_by_layer[L_int], dtype=NP_REAL_DTYPE)

        if eigs_L.ndim != 3:
            raise ValueError(
                "Each QFIM eigenvalue history array must be 3D: "
                "(num_runs, num_sample_iters, num_params)."
            )

        if eigs_L.shape[1] != sample_iters_arr.size and eigs_L.shape[0] == sample_iters_arr.size:
            eigs_L = np.transpose(eigs_L, (1, 0, 2))

        if eigs_L.shape[1] != sample_iters_arr.size:
            raise ValueError(
                f"Shape mismatch for L={L_int}: "
                f"eigs_L.shape={eigs_L.shape}, len(sample_iters)={sample_iters_arr.size}."
            )

        count_history_by_layer[L_int] = np.sum(
            eigs_L >= threshold,
            axis=2,
        ).astype(NP_REAL_DTYPE)

    return count_history_by_layer


def plot_qfim_eigcount_history_mean_by_layer(
    eigcount_history_by_layer: dict,
    layers,
    sample_iters_for_labels,
    *,
    threshold: float,
    title: str,
    outpath: str,
    cmap=None,
):
    valid_layers = [
        int(L)
        for L in layers
        if eigcount_history_by_layer.get(int(L)) is not None
    ]

    if not valid_layers:
        return

    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap
    threshold_label = _threshold_tex(threshold)
    x = np.asarray(sample_iters_for_labels, dtype=NP_REAL_DTYPE)
    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.22)
    n_layers = len(valid_layers)

    for layer_idx, L in enumerate(valid_layers):
        counts_L = np.asarray(
            eigcount_history_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )

        if counts_L.ndim != 2:
            raise ValueError(
                "Each QFIM eig-count history array must be 2D: "
                "(num_runs, num_sample_iters)."
            )

        if counts_L.shape[1] != x.size:
            raise ValueError(
                f"Shape mismatch for L={L}: "
                f"counts_L.shape={counts_L.shape}, len(sample_iters)={x.size}."
            )

        valid = np.isfinite(counts_L)
        n_valid = np.sum(valid, axis=0)
        sums = np.sum(np.where(valid, counts_L, 0.0), axis=0)
        means = np.divide(
            sums,
            n_valid,
            out=np.full(x.shape, np.nan, dtype=NP_REAL_DTYPE),
            where=n_valid > 0,
        )

        centered = np.where(valid, counts_L - means[None, :], np.nan)
        sq = np.nansum(centered**2, axis=0)
        stds = np.sqrt(
            np.divide(
                sq,
                n_valid - 1.0,
                out=np.zeros_like(sq, dtype=NP_REAL_DTYPE),
                where=n_valid > 1,
            )
        )
        sems = np.divide(
            stds,
            np.sqrt(n_valid),
            out=np.zeros_like(stds, dtype=NP_REAL_DTYPE),
            where=n_valid > 1,
        )

        finite_mask = np.isfinite(means) & (n_valid > 0)
        if not np.any(finite_mask):
            continue

        color = cmap(layer_idx / max(n_layers - 1, 1))
        ax.plot(
            x[finite_mask],
            means[finite_mask],
            marker="o",
            linestyle="-",
            linewidth=1.2,
            markersize=4.5,
            color=color,
            label=f"L={L}",
        )
        ax.fill_between(
            x[finite_mask],
            means[finite_mask] - sems[finite_mask],
            means[finite_mask] + sems[finite_mask],
            color=color,
            alpha=0.16,
            linewidth=0.0,
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel(rf"Mean QFIM eigenvalue count ($\lambda_i \geq {threshold_label}$)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in x], rotation=45, ha="right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
    )

    save_fig(
        fig,
        ax,
        outpath,
        outside_legend=True,
        legend_space_frac=0.22,
    )


def plot_qfim_eigcount_history_min_by_layer(
    eigcount_history_by_layer: dict,
    layers,
    sample_iters_for_labels,
    *,
    threshold: float,
    title: str,
    outpath: str,
    cmap=None,
):
    valid_layers = [
        int(L)
        for L in layers
        if eigcount_history_by_layer.get(int(L)) is not None
    ]

    if not valid_layers:
        return

    cmap = matplotlib.colormaps.get_cmap("viridis") if cmap is None else cmap
    threshold_label = _threshold_tex(threshold)
    x = np.asarray(sample_iters_for_labels, dtype=NP_REAL_DTYPE)
    fig, ax = new_fig_ax(outside_legend=True, legend_space_frac=0.22)
    n_layers = len(valid_layers)

    for layer_idx, L in enumerate(valid_layers):
        counts_L = np.asarray(
            eigcount_history_by_layer[L],
            dtype=NP_REAL_DTYPE,
        )

        if counts_L.ndim != 2:
            raise ValueError(
                "Each QFIM eig-count history array must be 2D: "
                "(num_runs, num_sample_iters)."
            )

        if counts_L.shape[1] != x.size:
            raise ValueError(
                f"Shape mismatch for L={L}: "
                f"counts_L.shape={counts_L.shape}, len(sample_iters)={x.size}."
            )

        valid = np.isfinite(counts_L)
        n_valid = np.sum(valid, axis=0)
        counts_for_min = np.where(valid, counts_L, np.inf)
        min_counts = np.min(counts_for_min, axis=0)
        min_counts = np.where(n_valid > 0, min_counts, np.nan)
        finite_mask = np.isfinite(min_counts) & (n_valid > 0)

        if not np.any(finite_mask):
            continue

        color = cmap(layer_idx / max(n_layers - 1, 1))
        ax.plot(
            x[finite_mask],
            min_counts[finite_mask],
            marker="o",
            linestyle="-",
            linewidth=1.2,
            markersize=4.5,
            color=color,
            label=f"L={L}",
        )

    ax.set_xlabel("Iterations")
    ax.set_ylabel(rf"Minimum QFIM eigenvalue count ($\lambda_i \geq {threshold_label}$)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in x], rotation=45, ha="right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
    )

    save_fig(
        fig,
        ax,
        outpath,
        outside_legend=True,
        legend_space_frac=0.22,
    )


qfim_eigcount_sample_iters = np.asarray(sample_iters, dtype=NP_INT_DTYPE)
qfim_eigcount_time_indices = np.arange(
    qfim_eigcount_sample_iters.size,
    dtype=NP_INT_DTYPE,
)

for threshold in (
    QFIM_PATH_EIGCOUNT_THRESHOLDS
    if INCLUDE_OPTIMIZATION_PATH_QFIM
    else ()
):
    eigcount_history_all_by_layer = qfim_eigcount_history_by_layer(
        qfim_eigs_history_optimization_path_by_layer,
        vqe_layer_list,
        sample_iters,
        threshold=threshold,
    )
    eigcount_history_by_layer = {
        int(L): np.asarray(counts, dtype=NP_REAL_DTYPE)[
            :,
            qfim_eigcount_time_indices,
        ]
        for L, counts in eigcount_history_all_by_layer.items()
    }
    threshold_tag = _thr_tag(threshold)
    threshold_label = _threshold_tex(threshold)

    plot_qfim_eigcount_history_mean_by_layer(
        eigcount_history_by_layer,
        vqe_layer_list,
        qfim_eigcount_sample_iters,
        threshold=threshold,
        title=(
            rf"Mean QFIM eigenvalue count along optimization path "
            rf"($\lambda_i \geq {threshold_label}$, {keep_label})"
        ),
        outpath=os.path.join(
            qfim_eigcount_optimization_path_mean_dir,
            f"qfim_eigcount_mean_history_optimization_path_thr_{threshold_tag}_{keep_key}.pdf",
        ),
        cmap=cmap,
    )

    plot_qfim_eigcount_history_min_by_layer(
        eigcount_history_by_layer,
        vqe_layer_list,
        qfim_eigcount_sample_iters,
        threshold=threshold,
        title=(
            rf"Minimum QFIM eigenvalue count along optimization path "
            rf"($\lambda_i \geq {threshold_label}$, {keep_label})"
        ),
        outpath=os.path.join(
            qfim_eigcount_optimization_path_min_dir,
            f"qfim_eigcount_min_history_optimization_path_thr_{threshold_tag}_{keep_key}.pdf",
        ),
        cmap=cmap,
    )


# ============================================================
# Mean ± SEM scalar QFIM diagnostics
#   1. Sum of QFIM eigenvalues at or above the rank threshold
#   2. Sum of absolute values of QFIM matrix entries
#   reduced keep=(0,1,2,3) only
# ============================================================
# The trace summary below uses optimization-path samples. The absolute-entry
# summary remains based on the random-point QFIM data saved above.
def _metric_mean_sem_by_layer(metric_by_layer: dict, layers) -> dict:
    stats_by_layer = {}

    for L in layers:
        if metric_by_layer.get(L) is None:
            continue

        mean_L, sem_L = _sample_mean_sem(metric_by_layer[L])

        stats_by_layer[L] = {
            "mean": mean_L,
            "sem": sem_L,
            "n": NP_INT_DTYPE(np.asarray(metric_by_layer[L]).size),
        }

    return stats_by_layer


def plot_metric_mean_sem_by_layer(
    metric_by_layer: dict,
    layers,
    *,
    ylabel: str,
    title: str,
    outpath: str,
    label: str,
    color="C0",
    marker: str = "o",
    log_scale: bool = False,
):
    stats_by_layer = _metric_mean_sem_by_layer(metric_by_layer, layers)

    x, means, sems = _mean_sem_arrays_from_by_layer(
        stats_by_layer,
        layers,
    )

    if x.size == 0:
        return

    fig, ax = new_fig_ax(outside_legend=False)

    ax.errorbar(
        x,
        means,
        yerr=sems,
        marker=marker,
        linestyle="-",
        linewidth=1.2,
        markersize=6.0,
        capsize=4.0,
        elinewidth=1.0,
        color=color,
        label=label,
    )

    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(L)) for L in x])

    if log_scale:
        ax.set_yscale("log")

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    save_fig(fig, ax, outpath, outside_legend=False)


if INCLUDE_OPTIMIZATION_PATH_QFIM and INCLUDE_QFIM_TRACE_FIGURES:
    plot_metric_mean_sem_by_layer(
        qfim_trace_history_optimization_path_by_layer,
        qfim_trace_history_layer_list,
        ylabel=QFIM_TRACE_MEAN_YLABEL,
        title=(
            rf"QFIM trace mean $\pm$ SEM vs Layers along optimization path "
            rf"({keep_label})"
        ),
        outpath=os.path.join(
            qfim_trace_dir,
            f"qfim_trace_mean_errorbar_optimization_path_{keep_key}.pdf",
        ),
        label=QFIM_TRACE_SUM_LABEL,
        color="C0",
        marker="o",
        log_scale=False,
    )

plot_metric_mean_sem_by_layer(
    qfim_abs_entry_sum_reduced_0123_by_layer,
    qfim_layer_list,
    ylabel="Mean elementwise absolute sum",
    title=rf"QFIM elementwise-absolute-sum mean $\pm$ SEM vs Layers ({keep_label}) at {NUM_QFIM_SAMPLES} random points",
    outpath=os.path.join(
        qfim_fig_dir,
        f"qfim_abs_entry_sum_mean_errorbar_{keep_key}.pdf",
    ),
    label=r"$\sum_{i,j} |F_{ij}|$",
    color="C1",
    marker="s",
    log_scale=False,
)

# ============================================================
# Optional keep=(0,1,2,3,4) QFIM figures
# ============================================================
def _warn_skip_new_figure(message: str) -> None:
    warnings.warn(
        f"Skipping optional QFIM figure: {message}",
        RuntimeWarning,
        stacklevel=2,
    )


def _load_optional_npz_result(path: str, *, description: str):
    if not os.path.exists(path):
        _warn_skip_new_figure(
            f"{description} result file was not found: {path}"
        )
        return None

    try:
        return load_npz_result(path)
    except (OSError, ValueError, KeyError) as exc:
        _warn_skip_new_figure(
            f"could not load {description} result file {path}: {exc}"
        )
        return None


def _summary_layers_or_none(result: dict, *, description: str):
    if "layers" not in result:
        _warn_skip_new_figure(f"{description} archive has no 'layers' key")
        return None

    layers = [
        int(L)
        for L in np.asarray(result["layers"], dtype=NP_INT_DTYPE).reshape(-1)
    ]

    if not layers:
        _warn_skip_new_figure(f"{description} archive has an empty layer list")
        return None

    return layers


def _summary_sample_iters_or_none(result: dict, *, description: str):
    if "sample_iters" not in result:
        _warn_skip_new_figure(
            f"{description} archive has no 'sample_iters' key"
        )
        return None

    sample_iters_result = np.asarray(
        result["sample_iters"],
        dtype=NP_INT_DTYPE,
    ).reshape(-1)

    if sample_iters_result.size == 0:
        _warn_skip_new_figure(
            f"{description} archive has an empty 'sample_iters' array"
        )
        return None

    return sample_iters_result


_QFIM_PARTICIPATION_RANK_LABEL = (
    r"Participation effective rank "
    r"$r_{\mathrm{eff}}="
    r"(\sum_{\lambda_i>10^{-12}}\lambda_i)^2/"
    r"\sum_{\lambda_i>10^{-12}}\lambda_i^2$"
)


def _has_expected_participation_rank_threshold(
    result: dict,
    *,
    description: str,
) -> bool:
    """Reject legacy participation-rank archives without active-spectrum metadata."""
    metadata_key = "participation_effective_rank_threshold"
    recompute_command = (
        "DPQC_overparam_reset_compute.py --stage qfim"
        if output_family == "dpqc_reset"
        else "DPQC_overparam_qfim.py"
    )
    if metadata_key not in result:
        _warn_skip_new_figure(
            f"{description} archive predates thresholded participation ranks; "
            f"rerun {recompute_command}"
        )
        return False

    try:
        threshold = float(np.asarray(result[metadata_key]).item())
    except (TypeError, ValueError) as exc:
        _warn_skip_new_figure(
            f"{description} archive has invalid {metadata_key!r}: {exc}"
        )
        return False

    expected = float(cfg.QFIM_EFFECTIVE_RANK_THRESHOLD)
    if not np.isfinite(threshold) or not np.isclose(
        threshold,
        expected,
        rtol=0.0,
        atol=0.0,
    ):
        _warn_skip_new_figure(
            f"{description} archive uses participation-rank threshold "
            f"{threshold!r}, expected {expected!r}; rerun {recompute_command}"
        )
        return False

    return True


def _finite_sample_effective_rank_statistics(samples):
    """Return mean, minimum, maximum, and SEM for finite sample values."""
    samples = np.asarray(samples, dtype=NP_REAL_DTYPE).reshape(-1)
    samples = samples[np.isfinite(samples)]
    sample_count = int(samples.size)

    if sample_count == 0:
        nan = NP_REAL_DTYPE(np.nan)
        return nan, nan, nan, nan

    mean = NP_REAL_DTYPE(np.mean(samples))
    minimum = NP_REAL_DTYPE(np.min(samples))
    maximum = NP_REAL_DTYPE(np.max(samples))
    sem = (
        NP_REAL_DTYPE(np.std(samples, ddof=1) / np.sqrt(sample_count))
        if sample_count > 1
        else NP_REAL_DTYPE(0.0)
    )
    return mean, minimum, maximum, sem


def plot_qfim_participation_effective_rank_by_layer(
    participation_rank_by_layer: dict,
    layers,
    *,
    state_label: str,
    outpath: str,
) -> bool:
    """Plot layerwise mean, extrema, and SEM of participation ranks."""
    valid_layers = [
        int(L)
        for L in layers
        if int(L) in participation_rank_by_layer
    ]

    if not valid_layers:
        _warn_skip_new_figure(
            "random-point effective-rank summary has no layer containing "
            "'participation_rank'"
        )
        return False

    x = np.asarray(valid_layers, dtype=NP_REAL_DTYPE)
    statistics = [
        _finite_sample_effective_rank_statistics(participation_rank_by_layer[L])
        for L in valid_layers
    ]
    means = np.asarray([item[0] for item in statistics], dtype=NP_REAL_DTYPE)
    minima = np.asarray([item[1] for item in statistics], dtype=NP_REAL_DTYPE)
    maxima = np.asarray([item[2] for item in statistics], dtype=NP_REAL_DTYPE)
    sems = np.asarray([item[3] for item in statistics], dtype=NP_REAL_DTYPE)
    finite = (
        np.isfinite(means)
        & np.isfinite(minima)
        & np.isfinite(maxima)
        & np.isfinite(sems)
    )

    if not np.any(finite):
        _warn_skip_new_figure(
            "random-point participation-effective-rank summary has no "
            "finite layer statistics"
        )
        return False

    fig, ax = new_fig_ax(outside_legend=False)
    ax.errorbar(
        x[finite],
        means[finite],
        yerr=sems[finite],
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=5.5,
        capsize=3.0,
        elinewidth=0.9,
        color="C0",
        label=r"Mean $\pm$ SEM",
        zorder=3,
    )
    ax.plot(
        x[finite],
        minima[finite],
        marker="v",
        linestyle="--",
        linewidth=1.2,
        markersize=5.0,
        color="C2",
        label="Minimum",
    )
    ax.plot(
        x[finite],
        maxima[finite],
        marker="^",
        linestyle="--",
        linewidth=1.2,
        markersize=5.0,
        color="C3",
        label="Maximum",
    )

    ax.set_xlabel("Number of Layers")
    ax.set_ylabel(_QFIM_PARTICIPATION_RANK_LABEL)
    ax.set_title(
        f"QFIM participation effective rank over random parameter points "
        f"({state_label})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in valid_layers])
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    save_fig(fig, ax, outpath, outside_legend=False)
    return True


def _valid_participation_rank_arrays(
    result: dict,
    layers,
    *,
    expected_ndim: int,
    description: str,
) -> dict:
    """Load finite, nonnegative participation-rank arrays by layer."""
    loaded = _load_layer_arrays_from_npz(
        result,
        layers,
        "participation_rank",
        dtype=NP_REAL_DTYPE,
    )
    valid = {}

    for L in layers:
        L_int = int(L)
        values = loaded.get(L_int)
        if values is None:
            _warn_skip_new_figure(
                f"{description} archive has no L{L_int}_participation_rank key"
            )
            continue

        values = np.asarray(values, dtype=NP_REAL_DTYPE)
        if values.ndim != int(expected_ndim):
            _warn_skip_new_figure(
                f"{description} participation rank for L={L_int} must be "
                f"{expected_ndim}D; received shape {values.shape}"
            )
            continue

        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            _warn_skip_new_figure(
                f"{description} participation rank for L={L_int} has no "
                "finite value"
            )
            continue
        if np.any(finite_values < 0.0):
            _warn_skip_new_figure(
                f"{description} participation rank for L={L_int} contains "
                "a negative value"
            )
            continue

        valid[L_int] = values

    return valid


def render_qfim_participation_effective_rank_figures(
    result_key: str,
    state_label: str,
) -> None:
    """Render layerwise participation-rank statistics at random points."""
    random_description = (
        f"{result_key} random-point participation-effective-rank summary"
    )
    random_summary_path = os.path.join(
        qfim_results_dir,
        f"qfim_effective_rank_random_points_{result_key}.npz",
    )
    random_summary = _load_optional_npz_result(
        random_summary_path,
        description=random_description,
    )

    if random_summary is not None and _has_expected_participation_rank_threshold(
        random_summary,
        description=random_description,
    ):
        random_layers = _summary_layers_or_none(
            random_summary,
            description=random_description,
        )
        if random_layers is not None:
            participation_rank_by_layer = _valid_participation_rank_arrays(
                random_summary,
                random_layers,
                expected_ndim=1,
                description=random_description,
            )
            plot_qfim_participation_effective_rank_by_layer(
                participation_rank_by_layer,
                random_layers,
                state_label=state_label,
                outpath=os.path.join(
                    qfim_effective_rank_dir,
                    f"qfim_participation_rank_random_points_{result_key}.pdf",
                ),
            )


def render_qfim_keep01234_core_figures() -> None:
    """Render the keep=(0,1,2,3,4) counterparts of the core QFIM figures."""
    random_result_path = os.path.join(
        qfim_results_dir,
        f"qfim_random_points_{keep_key_5}.npz",
    )
    random_result = _load_optional_npz_result(
        random_result_path,
        description=(
            "keep01234 random-point QFIM result; rerun "
            "DPQC_overparam_qfim.py for complete keep01234 figures"
        ),
    )
    if random_result is not None:
        random_layers = _summary_layers_or_none(
            random_result,
            description="keep01234 random-point QFIM result",
        )

        if random_layers is not None:
            num_random_samples = int(
                np.asarray(
                    random_result.get("num_qfim_samples", NUM_QFIM_SAMPLES),
                    dtype=NP_INT_DTYPE,
                ).reshape(-1)[0]
            )
            eigs_by_layer = _load_layer_arrays_from_npz(
                random_result,
                random_layers,
                "eigs_desc",
                dtype=NP_REAL_DTYPE,
            )
            trace_by_layer = {
                int(L): qfim_trace_at_or_above_rank_threshold(eigs)
                for L, eigs in eigs_by_layer.items()
            }
            abs_entry_sum_by_layer = _load_layer_arrays_from_npz(
                random_result,
                random_layers,
                "abs_entry_sum",
                dtype=NP_REAL_DTYPE,
            )
            for L in random_layers:
                if int(L) not in eigs_by_layer:
                    continue
                save_qfim_eigs_by_index(
                    eigs_by_layer[int(L)],
                    title=(
                        rf"QFIM eigenvalues at {num_random_samples} random "
                        rf"points (L={int(L)}, {keep_label_5})"
                    ),
                    outpath=os.path.join(
                        qfim_eigs_dir_red5,
                        f"L{int(L)}_reduced_01234.pdf",
                    ),
                )

            if INCLUDE_QFIM_EIGS_BY_INDEX_LAYERS:
                save_qfim_eigs_by_index_colored_by_layer(
                    eigs_by_layer,
                    random_layers,
                    title=(
                        rf"QFIM eigenvalues at {num_random_samples} random "
                        rf"points ({keep_label_5})"
                    ),
                    outpath=os.path.join(
                        qfim_eigs_dir,
                        "qfim_eigs_by_index_layers_reduced_keep_01234.pdf",
                    ),
                    cmap=cmap,
                )

            plot_qfim_random_eigcount_threshold_overlay(
                eigs_by_layer,
                random_layers,
                QFIM_PATH_EIGCOUNT_THRESHOLDS,
                title=(
                    "QFIM eigenvalue count at random points by threshold "
                    f"({keep_label_5})"
                ),
                outpath=os.path.join(
                    qfim_eigcount_random_dir,
                    (
                        "qfim_eigcount_threshold_overlay_random_points_"
                        f"{keep_key_5}.pdf"
                    ),
                ),
                cmap=cmap,
            )

            if INCLUDE_QFIM_TRACE_FIGURES:
                plot_qfim_trace_max_mean_sem_by_layer(
                    trace_by_layer,
                    random_layers,
                    title=(
                        rf"QFIM trace maximum and mean $\pm$ SEM at "
                        rf"{num_random_samples} random points ({keep_label_5})"
                    ),
                    outpath=os.path.join(
                        qfim_trace_dir,
                        (
                            "qfim_trace_max_mean_sem_random_points_"
                            "reduced_01234.pdf"
                        ),
                    ),
                    color_max="C0",
                    color_mean="C1",
                    marker_max="o",
                    marker_mean="s",
                    lw=1.4,
                    log_scale=False,
                )

            plot_metric_mean_sem_by_layer(
                abs_entry_sum_by_layer,
                random_layers,
                ylabel="Mean elementwise absolute sum",
                title=(
                    rf"QFIM elementwise-absolute-sum mean $\pm$ SEM vs "
                    rf"Layers ({keep_label_5}) at {num_random_samples} "
                    "random points"
                ),
                outpath=os.path.join(
                    qfim_fig_dir,
                    f"qfim_abs_entry_sum_mean_errorbar_{keep_key_5}.pdf",
                ),
                label=r"$\sum_{i,j} |F_{ij}|$",
                color="C1",
                marker="s",
                log_scale=False,
            )
    if not INCLUDE_OPTIMIZATION_PATH_QFIM:
        return

    eigs_history_path = os.path.join(
        qfim_results_dir,
        f"qfim_eigs_history_optimization_path_{keep_key_5}.npz",
    )
    eigs_history_result = _load_optional_npz_result(
        eigs_history_path,
        description=(
            "keep01234 optimization-path QFIM-eigenvalue result; rerun "
            "DPQC_overparam_qfim.py for spectrum-dependent figures"
        ),
    )

    eigs_history_layers = None
    eigs_history_sample_iters = None
    eigs_history_by_layer = {}

    if eigs_history_result is not None:
        eigs_history_layers = _summary_layers_or_none(
            eigs_history_result,
            description="keep01234 optimization-path QFIM-eigenvalue result",
        )
        eigs_history_sample_iters = _summary_sample_iters_or_none(
            eigs_history_result,
            description="keep01234 optimization-path QFIM-eigenvalue result",
        )

        if (
            eigs_history_layers is not None
            and eigs_history_sample_iters is not None
        ):
            eigs_history_by_layer = _load_layer_arrays_from_npz(
                eigs_history_result,
                eigs_history_layers,
                suffix=None,
                dtype=NP_REAL_DTYPE,
            )
            for threshold in QFIM_PATH_EIGCOUNT_THRESHOLDS:
                eigcount_history_by_layer = qfim_eigcount_history_by_layer(
                    eigs_history_by_layer,
                    eigs_history_layers,
                    eigs_history_sample_iters,
                    threshold=threshold,
                )
                threshold_tag = _thr_tag(threshold)
                threshold_label = _threshold_tex(threshold)

                plot_qfim_eigcount_history_mean_by_layer(
                    eigcount_history_by_layer,
                    eigs_history_layers,
                    eigs_history_sample_iters,
                    threshold=threshold,
                    title=(
                        "Mean QFIM eigenvalue count along optimization path "
                        rf"($\lambda_i \geq {threshold_label}$, "
                        f"{keep_label_5})"
                    ),
                    outpath=os.path.join(
                        qfim_eigcount_optimization_path_mean_dir,
                        (
                            "qfim_eigcount_mean_history_optimization_path_"
                            f"thr_{threshold_tag}_{keep_key_5}.pdf"
                        ),
                    ),
                    cmap=cmap,
                )
                plot_qfim_eigcount_history_min_by_layer(
                    eigcount_history_by_layer,
                    eigs_history_layers,
                    eigs_history_sample_iters,
                    threshold=threshold,
                    title=(
                        "Minimum QFIM eigenvalue count along optimization "
                        rf"path ($\lambda_i \geq {threshold_label}$, "
                        f"{keep_label_5})"
                    ),
                    outpath=os.path.join(
                        qfim_eigcount_optimization_path_min_dir,
                        (
                            "qfim_eigcount_min_history_optimization_path_"
                            f"thr_{threshold_tag}_{keep_key_5}.pdf"
                        ),
                    ),
                    cmap=cmap,
                )

    trace_history_layers = eigs_history_layers
    trace_history_sample_iters = eigs_history_sample_iters
    trace_history_by_layer = {
        int(L): qfim_trace_at_or_above_rank_threshold(eigs)
        for L, eigs in eigs_history_by_layer.items()
    }

    if (
        INCLUDE_QFIM_TRACE_FIGURES
        and trace_history_layers is not None
        and trace_history_sample_iters is not None
    ):
        plot_qfim_trace_history_mean_by_layer(
            trace_history_by_layer,
            trace_history_layers,
            trace_history_sample_iters,
            title=(
                "Mean QFIM trace along optimization path "
                f"({keep_label_5})"
            ),
            outpath=os.path.join(
                qfim_trace_dir,
                f"qfim_trace_mean_history_optimization_path_{keep_key_5}.pdf",
            ),
            cmap=cmap,
            log_scale=False,
        )
        plot_metric_mean_sem_by_layer(
            trace_history_by_layer,
            trace_history_layers,
            ylabel=QFIM_TRACE_MEAN_YLABEL,
            title=(
                rf"QFIM trace mean $\pm$ SEM vs Layers along optimization "
                rf"path ({keep_label_5})"
            ),
            outpath=os.path.join(
                qfim_trace_dir,
                (
                    "qfim_trace_mean_errorbar_optimization_path_"
                    f"{keep_key_5}.pdf"
                ),
            ),
            label=QFIM_TRACE_SUM_LABEL,
            color="C0",
            marker="o",
            log_scale=False,
        )


render_qfim_participation_effective_rank_figures(keep_key, keep_label)
render_qfim_participation_effective_rank_figures(keep_key_5, keep_label_5)
render_qfim_keep01234_core_figures()


if __name__ == "__main__" and _CLI_ARGS.with_hessian:
    run_hessian_workflow(_CLI_ARGS)

print(f"Visualized Hamiltonian parameter h: {h_param}")
print(f"Saved figures to: {save_dir}")
