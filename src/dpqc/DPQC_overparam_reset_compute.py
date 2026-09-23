#!/usr/bin/env python
# coding: utf-8
"""Launch independent reset-DPQC numerical programs.

The default ``analysis`` stage runs only random-point QFIM and Hessian
calculations. It never runs VQE training and needs no trained parameters.
Training lives in DPQC_overparam_reset_vqe.py; circuit/model definitions live
in dpqc_reset_model.py. Each pair block applies independent RY-RZ-RY rotations
on both wires, then RXX-RYY-RZZ, then independent RY-RZ-RY rotations on both
wires (15 angles per block, 60 per layer). The fixed Rx(pi) reset is unchanged.
Archives from the former 12*L circuit are incompatible with this model.
Current results use figs/dpqc_reset/u3_cartan/h_<h>/; the former model's
figs/dpqc_reset/h_<h>/ results are kept separately.

Each stage has its own entry point:

    python src/dpqc/DPQC_overparam_reset_vqe.py --h-param 0.1
    python src/dpqc/DPQC_overparam_reset_qfim.py --h-param 0.1
    python src/dpqc/DPQC_overparam_reset_hessian.py --h-param 0.1

For both analyses without training:

    python src/dpqc/DPQC_overparam_reset_compute.py --h-param 0.1

Legacy explicit --stage vqe/qfim/hessian commands still work. Use --stage all
only when training followed by both analyses is intended. Unlike the former
default, omitting --stage now selects analysis, not all. Each selected program
runs in a fresh process and a failure prevents later stages from running.

For saved-data visualization (no training or analysis recomputation):

    python src/dpqc/DPQC_overparam_reset_visualize.py --h-param 0.1
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
import dpqc_backend
import dpqc_wsl

# Preserve the documented circuit-builder API of this former combined file.
MODEL_ID = _model.MODEL_ID
OUTPUT_FAMILY = _model.OUTPUT_FAMILY
build_reset_circuit = _model.build_reset_circuit
num_trainable_parameters = _model.num_trainable_parameters
parameter_names = _model.parameter_names


def _default_config_values():
    return _model._default_config_values()


def _parse_cli_args(argv=None):
    default_h, default_batch = _default_config_values()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h-param", type=_model._finite_float, default=default_h)
    parser.add_argument(
        "--stage", choices=("analysis", "all", "vqe", "qfim", "hessian"),
        default="analysis",
        help="analysis (default): QFIM + Hessian; all: explicitly include VQE training.",
    )
    parser.add_argument(
        "--vqe-batch-size", type=_model._positive_int, default=default_batch,
        help="Training batch size; used only with --stage vqe/all.",
    )
    dpqc_backend.add_device_argument(parser)
    return parser.parse_args(argv)


def _launch_stage_subprocess(stage, args) -> int:
    if stage not in ("vqe", "qfim", "hessian"):
        raise ValueError(f"Unsupported numerical stage: {stage!r}")
    command = [
        sys.executable, str(_MODULE_DIR / f"DPQC_overparam_reset_{stage}.py"),
        "--h-param", str(args.h_param),
        "--device", dpqc_backend.resolve_stage_device(args.device, stage),
    ]
    if stage == "vqe":
        command.extend(("--vqe-batch-size", str(args.vqe_batch_size)))
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def main(argv=None) -> int:
    args = _parse_cli_args(argv)
    routed_status = dpqc_wsl.maybe_relaunch_in_wsl(
        __file__, argv, args.device, preserve_auto=True,
    )
    if routed_status is not None:
        return routed_status
    if args.stage == "all":
        stages = ("vqe", "qfim", "hessian")
    elif args.stage == "analysis":
        stages = ("qfim", "hessian")
    else:
        stages = (args.stage,)
    for stage in stages:
        return_code = _launch_stage_subprocess(stage, args)
        if return_code:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
