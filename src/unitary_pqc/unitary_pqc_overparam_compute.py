#!/usr/bin/env python
# coding: utf-8
"""Launch independent 60-parameter Cartan Unitary-PQC programs.

The default ``analysis`` stage computes QFIM/HS and Hessian without training.
QFIM reuses saved VQE parameter samples for optimization-path analysis;
``--random-only`` skips that path and needs no VQE archive. Hessians are
always independent of saved VQE/QFIM results. Each stage has its own file:

    python src/unitary_pqc/unitary_pqc_overparam_vqe.py --h-param 0.1
    python src/unitary_pqc/unitary_pqc_overparam_qfim.py --h-param 0.1
    python src/unitary_pqc/unitary_pqc_overparam_hessian.py --h-param 0.1

Unlike the former default, training now requires explicit --stage vqe/all.
With --device auto (the default), VQE uses the available GPU and QFIM/HS
and Hessian run on CPU. Windows uses the same configured WSL environment
as DPQC. Explicit --device cpu/gpu applies to every selected stage.
Each layer has four Cartan blocks and 60 angles, with no feed-forward.
Results use figs/unitary_pqc/u3_cartan/h_<h>/, separate from legacy results.
Historical numerical functions and result reads are available through lazy
delegates. Custom scripts that assign shared state should import the common
module directly. Importing this launcher does not initialize a numerical
runtime or run any computation.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import unitary_pqc_overparam_cli as _cli
import dpqc_backend
import dpqc_wsl

_parse_cli_args = _cli.parse_compute_args
_positive_int = _cli._positive_int
_coerce_finite_h_param = _cli._coerce_finite_h_param
_finite_float = _cli._finite_float
_resolve_h_param = _cli._resolve_h_param


def _stage_module(stage):
    name = f"unitary_pqc_overparam_{stage}"
    return importlib.import_module(f".{name}", __package__) if __package__ else importlib.import_module(name)


def _launch_stage_subprocess(stage, args) -> int:
    if stage not in ("vqe", "qfim", "hessian"):
        raise ValueError(f"Unsupported subprocess stage: {stage!r}")
    command = [
        sys.executable,
        str(_MODULE_DIR / f"unitary_pqc_overparam_{stage}.py"),
        "--h-param", str(args.h_param),
        "--device", dpqc_backend.resolve_stage_device(args.device, stage),
    ]
    if stage == "vqe":
        command.extend(("--vqe-batch-size", str(args.vqe_batch_size)))
    else:
        command.extend(("--analysis-batch-size", str(args.analysis_batch_size)))
    if stage == "qfim" and args.random_only:
        command.append("--random-only")
    return int(subprocess.run(command, check=False).returncode)


def _run_split_cli_pipeline(args) -> int:
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


def main(argv=None) -> int:
    args = _parse_cli_args(argv)
    routed_status = dpqc_wsl.maybe_relaunch_in_wsl(
        __file__, argv, args.device, preserve_auto=True,
    )
    if routed_status is not None:
        return routed_status
    return _run_split_cli_pipeline(args)


def run_unitary_pqc_overparam(
    *, h_param=None, vqe_batch_size=None, analysis_batch_size=None,
) -> dict:
    """Legacy in-process CPU API: train, then run all numerical analyses.

    For analysis without training, call the separate qfim/hessian stage
    functions or use the default command-line launcher instead. Use the CLI
    with --stage all for GPU training followed by CPU analyses in separate
    processes; this compatibility API retains its original CPU execution.
    """
    # Fail before training if the caller already initialized a GPU backend.
    cli = _stage_module("cli")
    cli.load_stage_common("vqe", "cpu")
    _stage_module("vqe").run_unitary_pqc_vqe_stage(
        h_param=h_param, vqe_batch_size=vqe_batch_size, device="cpu",
    )
    qfim_result = _stage_module("qfim").run_unitary_pqc_qfim_stage(
        h_param=h_param, analysis_batch_size=analysis_batch_size, device="cpu",
    )
    _stage_module("hessian").run_unitary_pqc_hessian_stage(
        h_param=h_param, analysis_batch_size=analysis_batch_size, device="cpu",
    )
    result = _stage_module("common").collect_unitary_pqc_result()
    result["analysis_batch_size"] = qfim_result["analysis_batch_size"]
    return result


_STAGE_EXPORTS = {
    "vqe": {
        "_vqe_sample_slot_by_iteration", "make_vqe_batch_runner",
        "_pad_vqe_theta_batch", "_resolve_vqe_batch_size",
        "run_vqe_optimization", "run_unitary_pqc_vqe_stage",
    },
    "qfim": {
        "_make_psd_analysis_batch_runner", "_make_psd_eigenvalue_batch_runner",
        "make_pure_qfim_matrix_fn_for_layer",
        "make_reduced_qfim_matrix_fn_for_layer_sequential",
        "make_reduced_hs_matrix_fn_for_layer_sequential",
        "make_pure_full_hs_matrix_fn_for_layer",
        "make_reduced_qfim_rank_fn_for_layer",
        "make_qfim_eigvals_fn_for_layer", "make_qfim_rank_fn_for_layer",
        "make_hs_eigvals_fn_for_layer", "make_hs_rank_fn_for_layer",
        "make_qfim_analysis_batch_runner", "make_hs_analysis_batch_runner",
        "compute_qfim_eigenvalue_history_by_layer",
        "compute_hs_eigenvalue_history_by_layer",
        "run_random_qfim_analysis", "run_optimization_path_qfim_analysis",
        "run_unitary_pqc_qfim_stage",
    },
    "hessian": {
        "make_energy_hessian_fn_for_layer",
        "make_energy_hessian_eigvals_fn_for_layer",
        "hessian_rank_and_condition_from_eigvals",
        "make_hessian_analysis_batch_runner", "run_random_hessian_analysis",
        "run_unitary_pqc_hessian_stage",
    },
}


def __getattr__(name):
    # Special metadata probes must not initialize the numerical runtime.
    if name.startswith("__"):
        raise AttributeError(name)
    for stage, names in _STAGE_EXPORTS.items():
        if name in names:
            return getattr(_stage_module(stage), name)
    return getattr(_stage_module("common"), name)


if __name__ == "__main__":
    raise SystemExit(main())
