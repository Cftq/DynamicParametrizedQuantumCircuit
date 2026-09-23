#!/usr/bin/env python
# coding: utf-8
"""Lightweight command-line options for independent Cartan stages."""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path

_COMMON_DIR = Path(__file__).resolve().parent.parent / "common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

import config_overparam as cfg
import dpqc_backend
import dpqc_wsl


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _coerce_finite_h_param(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("h_param must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError("h_param must be a finite number")
    return parsed


def _finite_float(value: str) -> float:
    try:
        return _coerce_finite_h_param(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _resolve_h_param(value) -> float:
    return _coerce_finite_h_param(cfg.H_PARAM if value is None else value)


def _add_stage_options(parser, *, training=False, analysis=False, qfim=False):
    dpqc_backend.add_device_argument(parser)
    parser.add_argument(
        "--h-param", type=_finite_float, default=None,
        help="Hamiltonian parameter h (default: config_overparam.H_PARAM).",
    )
    if training:
        parser.add_argument(
            "--vqe-batch-size", type=_positive_int,
            default=int(getattr(cfg, "VQE_BATCH_SIZE", 5)),
            help="Number of independent VQE trials per batch; training only.",
        )
    if analysis:
        parser.add_argument(
            "--analysis-batch-size", type=_positive_int,
            default=int(getattr(cfg, "ANALYSIS_BATCH_SIZE", 5)),
            help="Number of independent analysis parameter points per batch.",
        )
    if qfim:
        parser.add_argument(
            "--random-only", action="store_true",
            help=("Compute random-point QFIM/HS only, without loading VQE results. "
                  "Otherwise reuse saved VQE samples for optimization-path QFIM/HS."),
        )


def parse_stage_args(stage: str, argv=None):
    descriptions = {
        "vqe": "Train the Cartan Unitary-PQC and save optimization results only.",
        "qfim": "Compute Cartan QFIM/HS without training or Hessian calculation.",
        "hessian": "Compute Cartan Hessians independently of VQE and QFIM.",
    }
    if stage not in descriptions:
        raise ValueError(f"Unsupported numerical stage: {stage!r}")
    parser = argparse.ArgumentParser(description=descriptions[stage])
    _add_stage_options(
        parser, training=stage == "vqe", analysis=stage != "vqe",
        qfim=stage == "qfim",
    )
    args = parser.parse_args(argv)
    args.h_param = _resolve_h_param(args.h_param)
    return args


def parse_compute_args(argv=None):
    parser = argparse.ArgumentParser(
        description=("Launch independent Cartan Unitary-PQC programs. "
                     "The default analysis stage runs QFIM/HS and Hessian only; "
                     "VQE training requires --stage vqe or --stage all."),
    )
    _add_stage_options(parser, training=True, analysis=True, qfim=True)
    parser.add_argument(
        "--stage", choices=("analysis", "all", "vqe", "qfim", "hessian"),
        default="analysis",
        help="analysis (default): QFIM/HS then Hessian; all: include VQE training.",
    )
    args = parser.parse_args(argv)
    args.h_param = _resolve_h_param(args.h_param)
    return args


def maybe_relaunch_stage_in_wsl(stage, args, script_path, argv=None):
    """Resolve the stage device before handing Windows execution to WSL."""
    args.device = dpqc_backend.resolve_stage_device(args.device, stage)
    return dpqc_wsl.maybe_relaunch_in_wsl(script_path, argv, args.device)


def load_stage_common(stage, device=None):
    """Select a backend before importing circuit constants or numerical code.

    A cached common module has already initialized JAX. Validate that backend
    too, so a later CPU/GPU request cannot silently use the previous device.
    Stages using different devices must run in separate Python processes.
    """
    selected = dpqc_backend.resolve_stage_device(device, stage)
    dpqc_backend.configure_jax_backend(selected)
    name = "unitary_pqc_overparam_common"
    qualified_name = f"{__package__}.{name}" if __package__ else name
    if qualified_name in sys.modules:
        dpqc_backend.initialize_jax_backend()
    return importlib.import_module(qualified_name)
