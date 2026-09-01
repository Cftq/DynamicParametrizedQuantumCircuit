#!/usr/bin/env python
# coding: utf-8
"""Visualize saved fixed-Rx(pi) reset-DPQC results.

Run ``DPQC_overparam_reset_compute.py`` first from the project directory that
should contain the ``figs`` output tree.  This entry point then renders the
saved VQE and random-point QFIM results below
``figs/dpqc_reset/h_<h_param>`` without recomputing either quantity.
The shared QFIM figures include the participation effective rank
``(sum(lambda[lambda > 1e-12]))**2 / sum(lambda[lambda > 1e-12]**2)``
saved by the compute stage, plus Trace figures computed as
``sum(lambda[lambda >= 1e-12])`` from the saved eigenvalues.  For each retained
subsystem, the random-parameter figures use the number of layers on the x-axis.
The effective-rank figure shows its mean with SEM error bars together with
explicit minimum and maximum curves.
Threshold rank and optimization-iteration effective-rank figures are not
rendered.

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


def _default_h_param() -> float:
    """Read the Hamiltonian default used by the reset compute program."""
    common_dir = str(_COMMON_DIR)
    if common_dir not in sys.path:
        sys.path.insert(0, common_dir)

    import config_overparam as cfg

    return float(cfg.H_PARAM)


def _parse_cli_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize saved fixed-Rx(pi) reset-DPQC VQE and random-point "
            "QFIM results without recomputation, including layerwise "
            "participation-effective-rank and thresholded-Trace summaries."
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
    return parser.parse_args(argv)


def _build_visualizer_command(
    h_param: float,
    convergence_tolerances: Sequence[float] | None = None,
) -> tuple[str, ...]:
    """Return the fixed-family child-process command."""
    h_param = float(h_param)
    if not math.isfinite(h_param):
        raise ValueError("h_param must be finite.")
    if not _BASE_VISUALIZER.is_file():
        raise FileNotFoundError(
            f"Base DPQC visualizer was not found: {_BASE_VISUALIZER}"
        )
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
    return tuple(command)


def _run_visualizer(
    h_param: float,
    convergence_tolerances: Sequence[float] | None = None,
) -> int:
    """Run the canonical plotter and propagate its process status."""
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        _build_visualizer_command(h_param, convergence_tolerances),
        check=False,
        env=environment,
        shell=False,
    )
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli_args(argv)
    return _run_visualizer(args.h_param, args.convergence_tolerances)


if __name__ == "__main__":
    raise SystemExit(main())
