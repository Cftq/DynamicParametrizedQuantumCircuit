#!/usr/bin/env python
# coding: utf-8
"""Compute random-point reset-DPQC QFIMs independently of VQE training.

Layer/sample/seed settings come from config_overparam.py. Only the QFIM stage
is imported; no VQE archive or trained parameters are needed. Existing energy
histories are preserved. Model metadata and QFIM archives are saved under
figs/dpqc_reset/u3_cartan/h_<h>/, separately from the former Rz/Rxx model.
The shared circuit uses 60 independent angles per layer: local RY-RZ-RY,
pair RXX-RYY-RZZ, and local RY-RZ-RY in each block. The reset is unchanged.
Saved results from the former 12-angle layer must be recomputed separately.

    python src/dpqc/DPQC_overparam_reset_qfim.py --h-param 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import dpqc_reset_model as _model
from dpqc_backend import add_device_argument, resolve_stage_device
from dpqc_wsl import maybe_relaunch_in_wsl


def run_qfim(*, h_param=None, device=None) -> int:
    """Compute random QFIM samples without loading or running the VQE stage."""
    device = resolve_stage_device(device, "qfim")
    default_h, _ = _model._default_config_values()
    h_param = _model._finite_float(str(default_h if h_param is None else h_param))
    module = _model._load_base_stage_module("qfim", h_param=h_param, device=device)
    _model._install_reset_model(module)
    save_dir = _model._configure_reset_output_paths(module)
    _model._ensure_model_metadata(save_dir, h_param)
    module.run_qfim(include_optimization_path=False)
    print(f"Saved reset-DPQC random-point QFIM results to: {module.qfim_results_dir}")
    return 0


def main(argv=None) -> int:
    default_h, _ = _model._default_config_values()
    parser = argparse.ArgumentParser(description=__doc__)
    add_device_argument(parser)
    parser.add_argument("--h-param", type=_model._finite_float, default=default_h)
    args = parser.parse_args(argv)
    args.device = resolve_stage_device(args.device, "qfim")
    wsl_returncode = maybe_relaunch_in_wsl(
        __file__, sys.argv[1:] if argv is None else argv, args.device,
    )
    if wsl_returncode is not None:
        return wsl_returncode
    return run_qfim(h_param=args.h_param, device=args.device)


if __name__ == "__main__":
    raise SystemExit(main())
