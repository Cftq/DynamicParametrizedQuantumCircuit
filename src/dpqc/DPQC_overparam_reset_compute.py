#!/usr/bin/env python
# coding: utf-8
"""Run the fixed-Rx(pi) reset-DPQC numerical pipeline.

This entry point reuses the established VQE and random-point QFIM workflows from
``DPQC_overparam_vqe.py`` and ``DPQC_overparam_qfim.py``, while replacing
their circuit model with

    CX(center -> fresh),
    CRX(fresh -> center, pi).

Each layer keeps its own 12 unitary parameters.  The feed-forward ``Rx(pi)``
angle is fixed and is not trainable, so a depth-L circuit has ``12 * L``
trainable parameters.  The numerical state update uses the equivalent reset
channel directly.  A Qiskit builder is exposed by
:func:`build_reset_circuit` so the requested gate sequence can also be
inspected explicitly.

Results are isolated below ``figs/dpqc_reset`` and never share archives with
the original 14-parameters-per-layer DPQC model.

The parameter-update rule is selected with ``DPQC_VQE_OPTIMIZER`` in
``config_overparam.py``.  Use ``"adam"`` (the default) or
``"gradient_descent"`` for deterministic full-batch gradient descent.

Examples::

    python DPQC_overparam_reset_compute.py --stage all
    python DPQC_overparam_reset_compute.py --stage vqe --vqe-batch-size 20
    python DPQC_overparam_reset_compute.py --stage qfim --h-param 0.10
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence


_MODULE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _MODULE_DIR.parent
_COMMON_DIR = _SRC_DIR / "common"
for _path in (_MODULE_DIR, _COMMON_DIR):
    _path_string = str(_path)
    if _path_string not in sys.path:
        sys.path.insert(0, _path_string)


MODEL_ID = "dpqc_reset_fixed_rx_pi"
OUTPUT_FAMILY = "dpqc_reset"
NUM_SYSTEM_QUBITS = 5
NUM_BLOCKS = 4
PARAMS_PER_BLOCK = 3
UNITARY_PARAMS_PER_LAYER = NUM_BLOCKS * PARAMS_PER_BLOCK
NUM_TRAINABLE_FEED_FORWARD_PARAMS = 0
FIXED_FEED_FORWARD_RX_ANGLE = math.pi

TOP, LEFT, RIGHT, BOTTOM, ANC_CENTER = 0, 1, 2, 3, 4
LAYER_PAIRS = (
    (LEFT, BOTTOM),
    (RIGHT, BOTTOM),
    (TOP, RIGHT),
    (TOP, ANC_CENTER),
)


def num_trainable_parameters(n_layer: int) -> int:
    """Return the parameter count ``12 * n_layer``."""
    n_layer = int(n_layer)
    if n_layer < 1:
        raise ValueError("n_layer must be >= 1.")
    return UNITARY_PARAMS_PER_LAYER * n_layer


def parameter_names(n_layer: int) -> tuple[str, ...]:
    """Return the layer-major unitary parameter names."""
    n_layer = int(n_layer)
    num_trainable_parameters(n_layer)
    names: list[str] = []
    for layer_index in range(1, n_layer + 1):
        for block_index in range(NUM_BLOCKS):
            names.extend(
                (
                    f"L{layer_index}_B{block_index}_Rz_q0",
                    f"L{layer_index}_B{block_index}_Rz_q1",
                    f"L{layer_index}_B{block_index}_Rxx",
                )
            )
    return tuple(names)


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


def _default_config_values() -> tuple[float, int]:
    import config_overparam as cfg

    return float(cfg.H_PARAM), int(getattr(cfg, "VQE_BATCH_SIZE", 5))


def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_h_param, default_vqe_batch_size = _default_config_values()
    parser = argparse.ArgumentParser(
        description=(
            "Run reset-DPQC with a fixed Rx(pi) feed-forward rotation."
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
        choices=("all", "vqe", "qfim"),
        default="all",
        help=(
            "all: run VQE and then random-point QFIM in separate processes; "
            "vqe/qfim: run only the selected stage (the QFIM stage excludes "
            "optimization-path diagnostics)"
        ),
    )
    parser.add_argument(
        "--vqe-batch-size",
        type=_positive_int,
        default=default_vqe_batch_size,
        help="Number of independent VQE trials evaluated by each vmap call.",
    )
    return parser.parse_args(argv)


def build_reset_circuit(n_layer: int, theta: Iterable[float] | None = None):
    """Build the requested Stinespring circuit with one fresh wire per layer.

    Parameters
    ----------
    n_layer:
        Positive circuit depth.
    theta:
        Optional numerical vector in the order returned by
        :func:`parameter_names`.  If omitted, all 12L unitary angles are
        symbolic Qiskit parameters.  Rx(pi) is always numeric and is never
        included in ``circuit.parameters``.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    n_layer = int(n_layer)
    expected = num_trainable_parameters(n_layer)

    if theta is None:
        unitary_values = [
            Parameter(name) for name in parameter_names(n_layer)
        ]
    else:
        import numpy as np

        values = np.asarray(tuple(theta), dtype=np.float64)
        if values.shape != (expected,):
            raise ValueError(
                f"theta must have shape ({expected},), got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("theta must contain only finite values.")
        unitary_values = values.tolist()

    circuit = QuantumCircuit(NUM_SYSTEM_QUBITS + n_layer, name=MODEL_ID)
    value_index = 0
    for layer_index in range(n_layer):
        for q0, q1 in LAYER_PAIRS:
            circuit.rz(unitary_values[value_index], q0)
            circuit.rz(unitary_values[value_index + 1], q1)
            circuit.rxx(unitary_values[value_index + 2], q0, q1)
            value_index += PARAMS_PER_BLOCK

        fresh_wire = NUM_SYSTEM_QUBITS + layer_index
        circuit.cx(ANC_CENTER, fresh_wire)
        circuit.crx(FIXED_FEED_FORWARD_RX_ANGLE, fresh_wire, ANC_CENTER)

    if value_index != UNITARY_PARAMS_PER_LAYER * n_layer:
        raise AssertionError("Internal unitary-parameter routing mismatch.")
    return circuit


def _prepare_base_config(h_param: float, vqe_batch_size: int) -> None:
    """Set values read by an imported split-stage module."""
    import config_overparam as cfg

    cfg.H_PARAM = float(h_param)
    cfg.VQE_BATCH_SIZE = int(vqe_batch_size)


def _load_base_stage_module(
    stage: str,
    *,
    h_param: float,
    vqe_batch_size: int,
) -> ModuleType:
    _prepare_base_config(h_param, vqe_batch_size)
    module_name = (
        "DPQC_overparam_vqe" if stage == "vqe" else "DPQC_overparam_qfim"
    )
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def _install_reset_model(module: ModuleType) -> None:
    """Install the fixed-Rx(pi) reset state model into one stage module."""
    jnp = module.jnp

    module.NUM_BLOCKS = NUM_BLOCKS
    module.PARAMS_PER_BLOCK = PARAMS_PER_BLOCK
    module.UNITARY_PARAMS_PER_LAYER = UNITARY_PARAMS_PER_LAYER
    module.NUM_TRAINABLE_FEED_FORWARD_PARAMS = (
        NUM_TRAINABLE_FEED_FORWARD_PARAMS
    )
    module.EXTRA_PARAMS_PER_LAYER = 0
    module.n_param_per_layer = UNITARY_PARAMS_PER_LAYER
    module.FIXED_FEED_FORWARD_RX_ANGLE = FIXED_FEED_FORWARD_RX_ANGLE
    module.num_trainable_parameters = num_trainable_parameters

    def _validate_theta_shape(theta, n_layer: int) -> int:
        expected = num_trainable_parameters(n_layer)
        if theta.ndim != 1 or theta.shape[0] != expected:
            raise ValueError(
                f"theta must have shape ({expected},) for L={n_layer}, "
                f"got {theta.shape}."
            )
        return expected

    def wrap_theta_periodic_only(theta, n_layer: int):
        theta = jnp.asarray(theta, dtype=module.REAL_DTYPE)
        _validate_theta_shape(theta, n_layer)
        return module.wrap_to_pi(theta)

    def theta_difference_periodic_only(theta_a, theta_b, n_layer: int):
        theta_a = jnp.asarray(theta_a, dtype=module.REAL_DTYPE)
        theta_b = jnp.asarray(theta_b, dtype=module.REAL_DTYPE)
        _validate_theta_shape(theta_a, n_layer)
        _validate_theta_shape(theta_b, n_layer)
        return module.wrap_to_pi(theta_a - theta_b)

    def rms_theta_distance_periodic_only(theta_a, theta_b, n_layer: int):
        difference = theta_difference_periodic_only(
            theta_a,
            theta_b,
            n_layer=n_layer,
        )
        return jnp.sqrt(jnp.mean(difference**2))

    def _apply_kept_blocks(
        rho,
        layer_theta,
        *,
        k: int = NUM_SYSTEM_QUBITS,
    ):
        if layer_theta.shape != (UNITARY_PARAMS_PER_LAYER,):
            raise ValueError(
                "Each unitary layer must contain exactly "
                f"{UNITARY_PARAMS_PER_LAYER} parameters, got "
                f"{layer_theta.shape}."
            )
        blocks = jnp.reshape(
            layer_theta,
            (NUM_BLOCKS, PARAMS_PER_BLOCK),
        )
        for (q0, q1), block_params in zip(LAYER_PAIRS, blocks):
            rho = module.apply_unitary_on_rho(
                rho,
                module.U_rz(block_params[0]),
                (q0,),
                k,
            )
            rho = module.apply_unitary_on_rho(
                rho,
                module.U_rz(block_params[1]),
                (q1,),
                k,
            )
            rho = module.apply_unitary_on_rho(
                rho,
                module.U_rxx(block_params[2]),
                (q0, q1),
                k,
            )
        return rho

    def _apply_dynamic_delay_kraus(rho):
        """Apply the exact reset channel induced by fixed CRx(pi)."""
        expected_shape = (2**NUM_SYSTEM_QUBITS, 2**NUM_SYSTEM_QUBITS)
        if rho.shape != expected_shape:
            raise ValueError(
                f"Expected rho shape {expected_shape}, got {rho.shape}."
            )
        if ANC_CENTER != NUM_SYSTEM_QUBITS - 1:
            raise ValueError("The direct reset channel requires center last.")

        dim_rest = 2 ** (NUM_SYSTEM_QUBITS - 1)
        rho_blocks = jnp.reshape(rho, (dim_rest, 2, dim_rest, 2))
        rho00 = rho_blocks[:, 0, :, 0]
        rho11 = rho_blocks[:, 1, :, 1]
        rho_rest = rho00 + rho11
        rho_next = jnp.einsum(
            "rs,ab->rasb",
            rho_rest,
            module._RHO_QUBIT_ZERO,
        )
        p1 = jnp.real(jnp.trace(rho11))
        return jnp.reshape(rho_next, expected_shape), p1

    def _split_theta(theta, n_layer: int):
        theta = jnp.asarray(theta, dtype=module.REAL_DTYPE)
        _validate_theta_shape(theta, n_layer)
        return jnp.reshape(
            theta,
            (int(n_layer), UNITARY_PARAMS_PER_LAYER),
        )

    def _rho5_after_layer(rho, layer_theta):
        rho = _apply_kept_blocks(rho, layer_theta)
        return _apply_dynamic_delay_kraus(rho)

    def rho_keep_sequential_dpqc(theta, n_layer: int):
        unitary_layers = _split_theta(theta, n_layer)

        def one_layer(rho, layer_theta):
            rho_next, _ = _rho5_after_layer(rho, layer_theta)
            return rho_next, None

        rho_final, _ = module.jax.lax.scan(
            one_layer,
            module._RHO_KEEP_INIT,
            unitary_layers,
        )
        return module._hermitian(rho_final)

    def ancilla_p1_sequential_dpqc(theta, n_layer: int):
        unitary_layers = _split_theta(theta, n_layer)

        def one_layer(rho, layer_theta):
            rho_next, p1 = _rho5_after_layer(rho, layer_theta)
            return rho_next, p1

        _, p1_vector = module.jax.lax.scan(
            one_layer,
            module._RHO_KEEP_INIT,
            unitary_layers,
        )
        return p1_vector

    module.wrap_theta_periodic_only = wrap_theta_periodic_only
    module.theta_difference_periodic_only = theta_difference_periodic_only
    module.rms_theta_distance_periodic_only = rms_theta_distance_periodic_only
    module._apply_kept_blocks = _apply_kept_blocks
    module._apply_dynamic_delay_kraus = _apply_dynamic_delay_kraus
    module._rho5_after_layer = _rho5_after_layer
    module.rho_keep_sequential_dpqc = rho_keep_sequential_dpqc
    module.ancilla_p1_sequential_dpqc = ancilla_p1_sequential_dpqc


def _configure_reset_output_paths(module: ModuleType) -> Path:
    """Redirect every split-stage result to the isolated reset directory."""
    save_dir = Path.cwd() / "figs" / OUTPUT_FAMILY / f"h_{module.h_param}"
    energy_fig_dir = save_dir / "energy_figures"
    qfim_fig_dir = save_dir / "qfim_figures"
    numerical_results_dir = save_dir / "numerical_results"
    energy_results_dir = numerical_results_dir / "energy"
    qfim_results_dir = numerical_results_dir / "qfim"

    for directory in (
        save_dir,
        energy_fig_dir,
        qfim_fig_dir,
        numerical_results_dir,
        energy_results_dir,
        qfim_results_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    module.save_dir = str(save_dir)
    module.energy_fig_dir = str(energy_fig_dir)
    module.qfim_fig_dir = str(qfim_fig_dir)
    module.numerical_results_dir = str(numerical_results_dir)
    module.energy_results_dir = str(energy_results_dir)
    module.qfim_results_dir = str(qfim_results_dir)
    module.vqe_optimization_result_path = str(
        energy_results_dir / "vqe_optimization_histories.npz"
    )
    return save_dir


def _model_metadata(h_param: float) -> dict[str, object]:
    return {
        "schema_version": 2,
        "model_id": MODEL_ID,
        "h_param": float(h_param),
        "unitary_parameters_per_layer": UNITARY_PARAMS_PER_LAYER,
        "trainable_feed_forward_parameters": (
            NUM_TRAINABLE_FEED_FORWARD_PARAMS
        ),
        "total_parameter_formula": "12 * L",
        "fixed_feed_forward_rx_angle": FIXED_FEED_FORWARD_RX_ANGLE,
        "feed_forward_gate_sequence": (
            "CX(center->fresh); CRX(fresh->center,pi)"
        ),
        "feed_forward_rz_removed": True,
        "numerical_channel": (
            "trace_center(rho) tensor |0><0|_center"
        ),
        "physical_rms_distance_uses_all_parameters": True,
    }


def _metadata_path(save_dir: Path) -> Path:
    return save_dir / "reset_model_metadata.json"


def _write_model_metadata(save_dir: Path, h_param: float) -> Path:
    destination = _metadata_path(save_dir)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(_model_metadata(h_param), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def _validate_model_metadata(save_dir: Path, h_param: float) -> None:
    archive = (
        save_dir
        / "numerical_results"
        / "energy"
        / "vqe_optimization_histories.npz"
    )
    metadata_path = _metadata_path(save_dir)
    if not archive.is_file():
        raise FileNotFoundError(
            "Reset-DPQC VQE archive was not found. Run --stage vqe first: "
            f"{archive}"
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Reset-DPQC model metadata was not found; refusing to read an "
            f"unidentified archive: {metadata_path}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = _model_metadata(h_param)
    for key in (
        "schema_version",
        "model_id",
        "unitary_parameters_per_layer",
        "trainable_feed_forward_parameters",
        "total_parameter_formula",
        "feed_forward_gate_sequence",
        "feed_forward_rz_removed",
    ):
        if metadata.get(key) != expected[key]:
            raise ValueError(
                f"Incompatible reset archive metadata for {key!r}: "
                f"{metadata.get(key)!r} != {expected[key]!r}."
            )
    if not math.isclose(
        float(metadata.get("h_param", math.nan)),
        float(h_param),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Saved reset archive has a different h_param.")
    if not math.isclose(
        float(metadata.get("fixed_feed_forward_rx_angle", math.nan)),
        FIXED_FEED_FORWARD_RX_ANGLE,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Saved reset archive does not use fixed Rx(pi).")


def _run_numerical_stage(stage: str, args: argparse.Namespace) -> int:
    module = _load_base_stage_module(
        stage,
        h_param=args.h_param,
        vqe_batch_size=args.vqe_batch_size,
    )
    _install_reset_model(module)
    save_dir = _configure_reset_output_paths(module)

    if stage == "vqe":
        module.run_vqe()
        metadata_path = _write_model_metadata(save_dir, args.h_param)
        print(f"Saved reset-DPQC VQE results to: {module.energy_results_dir}")
        print(f"Saved reset-DPQC model metadata to: {metadata_path}")
        return 0

    if stage == "qfim":
        _validate_model_metadata(save_dir, args.h_param)
        module.run_qfim(include_optimization_path=False)
        print(
            "Saved reset-DPQC random-point QFIM results to: "
            f"{module.qfim_results_dir}"
        )
        return 0

    raise ValueError(f"Unsupported numerical stage: {stage!r}")


def _launch_stage_subprocess(stage: str, args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--h-param",
        str(args.h_param),
    ]
    if stage == "vqe":
        command.extend(("--vqe-batch-size", str(args.vqe_batch_size)))
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli_args(argv)
    if args.stage == "all":
        print(
            "Running reset-DPQC with P(L)=12L and fixed Rx(pi).",
            flush=True,
        )
        return_code = _launch_stage_subprocess("vqe", args)
        if return_code:
            return return_code
        return _launch_stage_subprocess("qfim", args)
    return _run_numerical_stage(args.stage, args)


if __name__ == "__main__":
    raise SystemExit(main())
