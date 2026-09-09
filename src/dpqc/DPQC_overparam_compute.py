#!/usr/bin/env python
# coding: utf-8
"""Run the DPQC VQE, QFIM, and random-point energy Hessian programs.

The default ``all`` stage runs VQE, QFIM, and Hessian analysis in separate
processes, stopping if a stage fails. The Hessian stage uses the DPQC QFIM
layer schedule, ``NUM_QFIM_SAMPLES``, and ``QFIM_SAMPLE_SEED_BASE`` from
``config_overparam.py``. It saves full signed energy Hessians and their random
parameter points below the current working directory at
``figs/dpqc/h_<h>/numerical_results/hessian/hessian_random_points.npz``.
For depth L, ``L{L}_hessian`` has shape ``(samples, 14*L, 14*L)`` and
``L{L}_theta`` has shape ``(samples, 14*L)``.

``DPQC_overparam_visualize.py --h-param <h>`` reads these saved results.
Use ``--stage hessian`` to compute only Hessians when VQE/QFIM are already
complete. Each stage can also be run through its own split program.

Examples::

    python src/dpqc/DPQC_overparam_compute.py --h-param 0.1
    python src/dpqc/DPQC_overparam_compute.py --h-param 0.1 --stage hessian
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path


_MODULE_DIR = Path(__file__).resolve().parent


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be a finite number")
    return parsed


def _default_config_values():
    common_dir = _MODULE_DIR.parent / "common"
    common_dir_string = str(common_dir)
    if common_dir_string not in sys.path:
        sys.path.insert(0, common_dir_string)

    import config_overparam as cfg

    return float(cfg.H_PARAM), int(getattr(cfg, "VQE_BATCH_SIZE", 5))


def _parse_cli_args(argv=None):
    default_h_param, default_vqe_batch_size = _default_config_values()
    parser = argparse.ArgumentParser(
        description=(
            "Run DPQC VQE, QFIM, and random-point Hessian analysis in "
            "separate processes and save their numerical results."
        )
    )
    parser.add_argument(
        "--h-param",
        type=_finite_float,
        default=default_h_param,
        help="Hamiltonian parameter H_PARAM.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "vqe", "qfim", "hessian"),
        default="all",
        help=(
            "all: run VQE, then QFIM, then random-point Hessian analysis; "
            "vqe/qfim/hessian: run only the selected program"
        ),
    )
    parser.add_argument(
        "--vqe-batch-size",
        type=_positive_int,
        default=default_vqe_batch_size,
        help="Number of independent VQE trials evaluated by each vmap call.",
    )
    return parser.parse_args(argv)


def _run_script(script_name: str, *arguments: str) -> int:
    completed = subprocess.run(
        [sys.executable, str(_MODULE_DIR / script_name), *arguments],
        check=False,
    )
    return int(completed.returncode)


def main(argv=None) -> int:
    args = _parse_cli_args(argv)
    h_arguments = ("--h-param", str(args.h_param))

    if args.stage in ("all", "vqe"):
        return_code = _run_script(
            "DPQC_overparam_vqe.py",
            *h_arguments,
            "--vqe-batch-size",
            str(args.vqe_batch_size),
        )
        if return_code:
            return return_code

    if args.stage in ("all", "qfim"):
        return_code = _run_script("DPQC_overparam_qfim.py", *h_arguments)
        if return_code:
            return return_code

    if args.stage in ("all", "hessian"):
        return _run_script(
            "DPQC_overparam_hessian.py",
            *h_arguments,
            "--output-family",
            "dpqc",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
