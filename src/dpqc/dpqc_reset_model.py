#!/usr/bin/env python
# coding: utf-8
"""Shared fixed-Rx(pi) reset-DPQC model, output paths, and archive metadata.

Importing this module does not load or run VQE, QFIM, or Hessian calculations.
Numerical dependencies are imported only when a stage explicitly requests them.
The reset channel, 12 parameters per layer, and metadata schema are shared by
the independent training and analysis entry points.
"""


from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable


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


def _prepare_base_config(
    h_param: float,
    vqe_batch_size: int | None = None,
) -> None:
    """Set values read by an imported split-stage module."""
    import config_overparam as cfg

    cfg.H_PARAM = float(h_param)
    if vqe_batch_size is not None:
        cfg.VQE_BATCH_SIZE = int(vqe_batch_size)


def _load_base_stage_module(
    stage: str,
    *,
    h_param: float,
    vqe_batch_size: int | None = None,
    device: str | None = None,
) -> ModuleType:
    if stage not in ("vqe", "qfim"):
        raise ValueError(f"Unsupported base numerical stage: {stage!r}.")
    _prepare_base_config(h_param, vqe_batch_size)
    from dpqc_backend import configure_jax_backend, resolve_stage_device

    configure_jax_backend(resolve_stage_device(device, stage))
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
            if getattr(module, "USE_ELEMENTWISE_DENSITY_KERNELS", False):
                rho = module.apply_rz_density(rho, block_params[0], q0, k)
                rho = module.apply_rz_density(rho, block_params[1], q1, k)
                rho = module.apply_rxx_density(rho, block_params[2], (q0, q1), k)
                continue
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


def _validate_model_metadata(
    save_dir: Path,
    h_param: float,
    *,
    require_vqe_archive: bool = False,
) -> None:
    """Validate model identity, optionally requiring saved training results."""
    archive = (
        save_dir
        / "numerical_results"
        / "energy"
        / "vqe_optimization_histories.npz"
    )
    metadata_path = _metadata_path(save_dir)
    if require_vqe_archive and not archive.is_file():
        raise FileNotFoundError(
            "Reset-DPQC VQE archive was not found. "
            "Run DPQC_overparam_reset_vqe.py first: "
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


def _ensure_model_metadata(save_dir: Path, h_param: float) -> Path:
    """Reuse valid metadata or identify a new model directory safely.

    Random-point analysis does not require training results. Existing VQE
    archives without metadata remain unidentified and must not be stamped as
    compatible simply because a new analysis is requested.
    """
    destination = _metadata_path(save_dir)
    if destination.exists():
        _validate_model_metadata(save_dir, h_param)
        return destination

    archive = (
        save_dir
        / "numerical_results"
        / "energy"
        / "vqe_optimization_histories.npz"
    )
    if archive.exists():
        raise FileNotFoundError(
            "Reset-DPQC model metadata was not found; refusing to identify "
            f"existing VQE results without their metadata: {archive}. "
            f"Restore the matching {destination.name} before continuing."
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    return _write_model_metadata(save_dir, h_param)
