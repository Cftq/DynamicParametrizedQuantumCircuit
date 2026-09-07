#!/usr/bin/env python
# coding: utf-8
"""Visualize saved fixed-Rx(pi) reset-DPQC results.

Run ``DPQC_overparam_reset_compute.py`` first from the project directory that
should contain the ``figs`` output tree.  This entry point then renders the
saved VQE and random-point QFIM results below
``figs/dpqc_reset/h_<h_param>`` without recomputing either quantity.  The
optional Hessian workflow computes or reuses random-point Hessian rank and
active-spectrum condition-number samples and renders their layerwise extrema
and mean with SEM.
The shared QFIM figures include the participation effective rank
``(sum(lambda[lambda > 1e-12]))**2 / sum(lambda[lambda > 1e-12]**2)``
saved by the compute stage, plus Trace figures computed as
``sum(lambda[lambda >= 1e-12])`` and spectral Shannon entropy figures computed
as ``-sum(p * log(p))`` in nats from those same saved eigenvalues.  Here the
probabilities normalize the active spectrum ``lambda >= 1e-12``; a zero active
trace gives entropy zero.  Threshold QFIM rank is computed directly from the
saved spectra as ``count(lambda >= 1e-12)``.  For each retained subsystem, the
random-parameter figures use the number of layers on the x-axis.  The
effective-rank, threshold-rank, Trace, and Shannon-entropy figures show their
means with SEM error bars together with explicit minimum and maximum curves.
Optimization-path effective rank is neither computed nor rendered.  Random-
point threshold rank is rendered only for the saved random parameter points.

The plotting implementation is shared with ``DPQC_overparam_visualize.py``.
It is launched in a separate Python process with the result family fixed to
``dpqc_reset``.  Keeping the model selection here non-configurable prevents a
reset visualization command from silently reading the original DPQC archive.
The shared plotter also validates ``reset_model_metadata.json`` before it
loads the numerical results.  VQE figures are labeled with the optimizer saved
in the numerical archive; legacy archives without that field are labeled Adam.

Examples::

    python src/dpqc/DPQC_overparam_reset_visualize.py
    python src/dpqc/DPQC_overparam_reset_visualize.py --h-param 0.1
    python src/dpqc/DPQC_overparam_reset_visualize.py --h-param 0.1 --hessian-only

In a Jupyter notebook::

    !python src/dpqc/DPQC_overparam_reset_visualize.py --h-param 0.1
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


_MODULE_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _MODULE_DIR.parent / "common"
_BASE_VISUALIZER = _MODULE_DIR / "DPQC_overparam_visualize.py"
_OUTPUT_FAMILY = "dpqc_reset"


def _finite_float(value: str) -> float:
    """Parse one finite floating-point command-line value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            "value must be a finite number"
        ) from exc
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be a finite number")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _finite_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            "value must be a nonnegative integer"
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    tokens = value.split(",")
    if not tokens or any(not token.strip() for token in tokens):
        raise argparse.ArgumentTypeError(
            "value must be a non-empty comma-separated integer list"
        )
    try:
        parsed = tuple(int(token.strip()) for token in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a comma-separated integer list"
        ) from exc
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("layers must be positive integers")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("layers must not be duplicated")
    return parsed


def _default_h_param() -> float:
    """Read the Hamiltonian default used by the reset compute program."""
    common_dir = str(_COMMON_DIR)
    if common_dir not in sys.path:
        sys.path.insert(0, common_dir)

    import config_overparam as cfg

    return float(cfg.H_PARAM)


def _default_hessian_settings() -> tuple[int, int]:
    """Return the random-point sample count and seed used by reset QFIM."""
    common_dir = str(_COMMON_DIR)
    if common_dir not in sys.path:
        sys.path.insert(0, common_dir)

    import config_overparam as cfg

    return int(cfg.NUM_QFIM_SAMPLES), int(cfg.QFIM_SAMPLE_SEED_BASE)


def _parse_cli_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    default_hessian_samples, default_hessian_seed = _default_hessian_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Visualize saved fixed-Rx(pi) reset-DPQC VQE and random-point "
            "QFIM results and optionally compute/reuse random-point Hessian "
            "rank and condition-number results."
        )
    )
    parser.add_argument(
        "--h-param",
        type=_finite_float,
        default=_default_h_param(),
        help=(
            "Hamiltonian parameter h whose reset-DPQC results are loaded "
            "(default: H_PARAM from config_overparam.py)."
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
    hessian_mode = parser.add_mutually_exclusive_group()
    hessian_mode.add_argument(
        "--hessian-only",
        action="store_true",
        help=(
            "Compute/load random-point reset-DPQC Hessians and render only "
            "the rank and condition-number figures."
        ),
    )
    hessian_mode.add_argument(
        "--with-hessian",
        action="store_true",
        help=(
            "Run the reset-DPQC Hessian workflow after the existing "
            "energy/QFIM figures."
        ),
    )
    parser.add_argument(
        "--reuse-hessian-results",
        action="store_true",
        help=(
            "Render the existing reset-DPQC Hessian result without "
            "recomputing."
        ),
    )
    parser.add_argument(
        "--hessian-results-dir",
        type=Path,
        default=None,
        help="Optional reset-DPQC Hessian numerical-results directory.",
    )
    parser.add_argument(
        "--hessian-figures-dir",
        type=Path,
        default=None,
        help="Optional reset-DPQC Hessian figure output directory.",
    )
    parser.add_argument(
        "--hessian-layers",
        type=_comma_separated_ints,
        default=None,
        help=(
            "Comma-separated layers to analyze "
            "(default: QFIM layer schedule)."
        ),
    )
    parser.add_argument(
        "--hessian-num-samples",
        type=_positive_int,
        default=default_hessian_samples,
        help=(
            "Number of random parameter points per layer "
            f"(default: {default_hessian_samples})."
        ),
    )
    parser.add_argument(
        "--hessian-seed-base",
        type=_nonnegative_int,
        default=default_hessian_seed,
        help=(
            "Base PRNG seed; layer L uses seed base + L "
            f"(default: {default_hessian_seed})."
        ),
    )
    parser.add_argument(
        "--hessian-hvp-chunk-size",
        type=_positive_int,
        default=8,
        help=(
            "Number of Hessian-vector products per compiled batch "
            "(default: 8)."
        ),
    )
    return parser.parse_args(argv)


def _build_visualizer_command(
    h_param: float,
    convergence_tolerances: Sequence[float] | None = None,
    *,
    hessian_only: bool = False,
    with_hessian: bool = False,
    reuse_hessian_results: bool = False,
    hessian_results_dir: Path | None = None,
    hessian_figures_dir: Path | None = None,
    hessian_layers: Sequence[int] | None = None,
    hessian_num_samples: int | None = None,
    hessian_seed_base: int | None = None,
    hessian_hvp_chunk_size: int | None = None,
) -> tuple[str, ...]:
    """Return the fixed-family child-process command."""
    h_param = float(h_param)
    if not math.isfinite(h_param):
        raise ValueError("h_param must be finite.")
    if not _BASE_VISUALIZER.is_file():
        raise FileNotFoundError(
            f"Base DPQC visualizer was not found: {_BASE_VISUALIZER}"
        )
    if hessian_only and with_hessian:
        raise ValueError("hessian_only and with_hessian are mutually exclusive.")
    command = [
        sys.executable,
        str(_BASE_VISUALIZER),
        "--h-param",
        repr(h_param),
        "--output-family",
        _OUTPUT_FAMILY,
        "--skip-optimization-path-qfim",
        "--skip-qfim-eigs-by-index-layers",
    ]
    tolerance_values = (
        () if convergence_tolerances is None else convergence_tolerances
    )
    for tolerance in tolerance_values:
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("convergence tolerances must be finite and positive")
        command.extend(("--convergence-tolerance", repr(tolerance)))

    hessian_requested = bool(hessian_only or with_hessian)
    if hessian_only:
        command.append("--hessian-only")
    elif with_hessian:
        command.append("--with-hessian")
    if hessian_requested and reuse_hessian_results:
        command.append("--reuse-hessian-results")
    if hessian_requested and hessian_results_dir is not None:
        command.extend(("--hessian-results-dir", str(hessian_results_dir)))
    if hessian_requested and hessian_figures_dir is not None:
        command.extend(("--hessian-figures-dir", str(hessian_figures_dir)))
    if hessian_requested and hessian_layers is not None:
        normalized_layers = tuple(int(layer) for layer in hessian_layers)
        if (
            not normalized_layers
            or any(layer <= 0 for layer in normalized_layers)
            or len(set(normalized_layers)) != len(normalized_layers)
        ):
            raise ValueError("hessian_layers must be unique positive integers.")
        command.extend(
            (
                "--hessian-layers",
                ",".join(str(layer) for layer in normalized_layers),
            )
        )
    if hessian_requested and hessian_num_samples is not None:
        command.extend(
            (
                "--hessian-num-samples",
                str(_positive_int(str(hessian_num_samples))),
            )
        )
    if hessian_requested and hessian_seed_base is not None:
        command.extend(
            (
                "--hessian-seed-base",
                str(_nonnegative_int(str(hessian_seed_base))),
            )
        )
    if hessian_requested and hessian_hvp_chunk_size is not None:
        command.extend(
            (
                "--hessian-hvp-chunk-size",
                str(_positive_int(str(hessian_hvp_chunk_size))),
            )
        )
    return tuple(command)


def _run_visualizer(
    h_param: float,
    convergence_tolerances: Sequence[float] | None = None,
    **hessian_options,
) -> int:
    """Run the canonical plotter and propagate its process status."""
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        _build_visualizer_command(
            h_param,
            convergence_tolerances,
            **hessian_options,
        ),
        check=False,
        env=environment,
        shell=False,
    )
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli_args(argv)
    return _run_visualizer(
        args.h_param,
        args.convergence_tolerances,
        hessian_only=args.hessian_only,
        with_hessian=args.with_hessian,
        reuse_hessian_results=args.reuse_hessian_results,
        hessian_results_dir=args.hessian_results_dir,
        hessian_figures_dir=args.hessian_figures_dir,
        hessian_layers=args.hessian_layers,
        hessian_num_samples=args.hessian_num_samples,
        hessian_seed_base=args.hessian_seed_base,
        hessian_hvp_chunk_size=args.hessian_hvp_chunk_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
