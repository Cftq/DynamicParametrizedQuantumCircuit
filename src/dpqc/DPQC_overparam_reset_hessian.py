#!/usr/bin/env python
# coding: utf-8
"""Compute reset-DPQC energy Hessians without VQE or QFIM execution.

This entry point fixes the shared Hessian program's model to dpqc_reset. It
saves signed matrices and random parameters under
figs/dpqc_reset/h_<h>/numerical_results/hessian by default. Neither trained
parameters nor a VQE archive are required. Plot these saved matrices with
DPQC_overparam_reset_visualize.py --hessian-only --reuse-hessian-results.

    python src/dpqc/DPQC_overparam_reset_hessian.py --h-param 0.1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import dpqc_reset_model as _model


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def main(argv=None) -> int:
    default_h, _ = _model._default_config_values()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h-param", type=_model._finite_float, default=default_h)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--layers", default=None, help="Comma-separated positive layers.")
    parser.add_argument("--num-samples", type=_model._positive_int, default=None)
    parser.add_argument("--seed-base", type=_nonnegative_int, default=None)
    parser.add_argument("--hvp-chunk-size", type=_model._positive_int, default=None)
    args = parser.parse_args(argv)
    command = [
        sys.executable, str(_MODULE_DIR / "DPQC_overparam_hessian.py"),
        "--output-family", _model.OUTPUT_FAMILY, "--h-param", str(args.h_param),
    ]
    for name in ("output_dir", "layers", "num_samples", "seed_base", "hvp_chunk_size"):
        value = getattr(args, name)
        if value is not None:
            command.extend(("--" + name.replace("_", "-"), str(value)))
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
