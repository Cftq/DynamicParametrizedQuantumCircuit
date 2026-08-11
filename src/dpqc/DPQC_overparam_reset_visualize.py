#!/usr/bin/env python
# coding: utf-8
"""Visualize saved shared-Rz, fixed-Rx(pi) reset-DPQC results.

Run ``DPQC_overparam_reset_compute.py`` first from the project directory that
should contain the ``figs`` output tree.  This entry point then renders the
saved VQE and QFIM results below
``figs/dpqc_reset/h_<h_param>`` without recomputing either quantity.

The plotting implementation is shared with ``DPQC_overparam_visualize.py``.
It is launched in a separate Python process with the result family fixed to
``dpqc_reset``.  Keeping the model selection here non-configurable prevents a
reset visualization command from silently reading the original DPQC archive.
The shared plotter also validates ``reset_model_metadata.json`` before it
loads the numerical results.

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
            "Visualize saved shared-Rz, fixed-Rx(pi) reset-DPQC VQE and "
            "QFIM results without recomputation."
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
    return parser.parse_args(argv)


def _build_visualizer_command(h_param: float) -> tuple[str, ...]:
    """Return the fixed-family child-process command."""
    h_param = float(h_param)
    if not math.isfinite(h_param):
        raise ValueError("h_param must be finite.")
    if not _BASE_VISUALIZER.is_file():
        raise FileNotFoundError(
            f"Base DPQC visualizer was not found: {_BASE_VISUALIZER}"
        )
    return (
        sys.executable,
        str(_BASE_VISUALIZER),
        "--h-param",
        repr(h_param),
        "--output-family",
        _OUTPUT_FAMILY,
    )


def _run_visualizer(h_param: float) -> int:
    """Run the canonical plotter and propagate its process status."""
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        _build_visualizer_command(h_param),
        check=False,
        env=environment,
        shell=False,
    )
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli_args(argv)
    return _run_visualizer(args.h_param)


if __name__ == "__main__":
    raise SystemExit(main())
