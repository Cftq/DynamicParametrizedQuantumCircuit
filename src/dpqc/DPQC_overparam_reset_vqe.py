#!/usr/bin/env python
# coding: utf-8
"""Train the fixed-Rx(pi) reset-DPQC and save its VQE parameter histories.

This is the training-only entry point. It does not compute QFIMs or Hessians.
Run DPQC_overparam_reset_qfim.py and DPQC_overparam_reset_hessian.py separately
for random-point analyses, or DPQC_overparam_reset_visualize.py to plot saved
results. Those commands never invoke this training program.

    python src/dpqc/DPQC_overparam_reset_vqe.py --h-param 0.1 --vqe-batch-size 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import dpqc_reset_model as _model
from dpqc_backend import add_device_argument
from dpqc_wsl import maybe_relaunch_in_wsl


def run_vqe(*, h_param=None, vqe_batch_size=None, device=None) -> int:
    """Execute only optimization and persist the existing reset archive format."""
    default_h, default_batch = _model._default_config_values()
    h_param = _model._finite_float(str(default_h if h_param is None else h_param))
    batch_size = _model._positive_int(
        str(default_batch if vqe_batch_size is None else vqe_batch_size)
    )
    module = _model._load_base_stage_module(
        "vqe", h_param=h_param, vqe_batch_size=batch_size, device=device,
    )
    _model._install_reset_model(module)
    save_dir = _model._configure_reset_output_paths(module)
    module.run_vqe()
    metadata_path = _model._write_model_metadata(save_dir, h_param)
    print(f"Saved reset-DPQC VQE results to: {module.energy_results_dir}")
    print(f"Saved reset-DPQC model metadata to: {metadata_path}")
    return 0


def main(argv=None) -> int:
    default_h, default_batch = _model._default_config_values()
    parser = argparse.ArgumentParser(description=__doc__)
    add_device_argument(parser)
    parser.add_argument("--h-param", type=_model._finite_float, default=default_h)
    parser.add_argument(
        "--vqe-batch-size", type=_model._positive_int, default=default_batch,
        help="Independent optimization trials per compiled batch.",
    )
    args = parser.parse_args(argv)
    wsl_returncode = maybe_relaunch_in_wsl(
        __file__, sys.argv[1:] if argv is None else argv, args.device,
    )
    if wsl_returncode is not None:
        return wsl_returncode
    return run_vqe(
        h_param=args.h_param, vqe_batch_size=args.vqe_batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
