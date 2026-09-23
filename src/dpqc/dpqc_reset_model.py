#!/usr/bin/env python
# coding: utf-8
"""Shared fixed-Rx(pi) reset-DPQC model, output paths, and archive metadata.

Importing this module does not load or run VQE, QFIM, or Hessian calculations.
Numerical dependencies are imported only when a stage explicitly requests them.
The reset channel, 60 parameters per layer, and metadata schema are shared by
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


MODEL_ID = "dpqc_reset_u3_cartan_fixed_rx_pi"
OUTPUT_FAMILY = "dpqc_reset"
OUTPUT_VARIANT = "u3_cartan"
NUM_SYSTEM_QUBITS = 5
NUM_BLOCKS = 4
BLOCK_PARAMETER_NAMES = (
    "pre_Ry1_q0", "pre_Rz_q0", "pre_Ry2_q0",
    "pre_Ry1_q1", "pre_Rz_q1", "pre_Ry2_q1",
    "Rxx", "Ryy", "Rzz",
    "post_Ry1_q0", "post_Rz_q0", "post_Ry2_q0",
    "post_Ry1_q1", "post_Rz_q1", "post_Ry2_q1",
)
PARAMS_PER_BLOCK = len(BLOCK_PARAMETER_NAMES)
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
    """Return the parameter count ``60 * n_layer`` (no shared angles)."""
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
                f"L{layer_index}_B{block_index}_{name}"
                for name in BLOCK_PARAMETER_NAMES
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
        :func:`parameter_names`.  If omitted, all 60L unitary angles are
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
            block = unitary_values[value_index:value_index + PARAMS_PER_BLOCK]
            for wire, start in ((q0, 0), (q1, 3)):
                circuit.ry(block[start], wire)
                circuit.rz(block[start + 1], wire)
                circuit.ry(block[start + 2], wire)
            circuit.rxx(block[6], q0, q1)
            circuit.ryy(block[7], q0, q1)
            circuit.rzz(block[8], q0, q1)
            for wire, start in ((q0, 9), (q1, 12)):
                circuit.ry(block[start], wire)
                circuit.rz(block[start + 1], wire)
                circuit.ry(block[start + 2], wire)
            value_index += PARAMS_PER_BLOCK

        fresh_wire = NUM_SYSTEM_QUBITS + layer_index
        circuit.cx(ANC_CENTER, fresh_wire)
        circuit.crx(FIXED_FEED_FORWARD_RX_ANGLE, fresh_wire, ANC_CENTER)

    if value_index != UNITARY_PARAMS_PER_LAYER * n_layer:
        raise AssertionError("Internal unitary-parameter routing mismatch.")
    return circuit


def unitary_block_matrix(block, jnp):
    """Return the 15-angle pair unitary in first-wire/second-wire basis order.

    ``U3`` here means RY(a), RZ(b), RY(c) in application order, not the
    library U3 gate. Both wires and both sides use independent triples.
    The array module is supplied by the caller to keep imports lightweight.
    """
    if block.shape != (PARAMS_PER_BLOCK,):
        raise ValueError(f"A unitary block requires {PARAMS_PER_BLOCK} angles.")

    def ry(theta):
        c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
        return jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)

    def rz(theta):
        return jnp.diag(jnp.exp(jnp.array([-0.5j, 0.5j]) * theta))

    def u3(start):
        return ry(block[start + 2]) @ rz(block[start + 1]) @ ry(block[start])

    x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
    y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
    z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
    identity = jnp.eye(4, dtype=jnp.complex128)
    unitary = jnp.kron(u3(0), u3(3))
    for theta, pauli in zip(block[6:9], (x, y, z)):
        rotation = (
            jnp.cos(theta / 2) * identity
            - 1j * jnp.sin(theta / 2) * jnp.kron(pauli, pauli)
        )
        unitary = rotation @ unitary
    return jnp.kron(u3(9), u3(12)) @ unitary


def apply_unitary_layer(
    rho, layer_theta, *, jnp, apply_unitary, k: int = NUM_SYSTEM_QUBITS,
):
    """Apply the four independent pair blocks, without the reset channel."""
    if layer_theta.shape != (UNITARY_PARAMS_PER_LAYER,):
        raise ValueError(
            f"Each unitary layer requires {UNITARY_PARAMS_PER_LAYER} angles, "
            f"got {layer_theta.shape}."
        )
    blocks = jnp.reshape(layer_theta, (NUM_BLOCKS, PARAMS_PER_BLOCK))
    for wires, block in zip(LAYER_PAIRS, blocks):
        rho = apply_unitary(rho, unitary_block_matrix(block, jnp), wires, k)
    return rho


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
        if not getattr(module, "USE_ELEMENTWISE_DENSITY_KERNELS", False):
            return apply_unitary_layer(
                rho, layer_theta, jnp=jnp,
                apply_unitary=module.apply_unitary_on_rho, k=k,
            )

        from dpqc_density_kernels import (
            apply_ry_density, apply_ryy_density, apply_rzz_density,
        )

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
            for wire, start in ((q0, 0), (q1, 3)):
                rho = apply_ry_density(rho, block_params[start], wire, k)
                rho = module.apply_rz_density(rho, block_params[start + 1], wire, k)
                rho = apply_ry_density(rho, block_params[start + 2], wire, k)
            rho = module.apply_rxx_density(rho, block_params[6], (q0, q1), k)
            rho = apply_ryy_density(rho, block_params[7], (q0, q1), k)
            rho = apply_rzz_density(rho, block_params[8], (q0, q1), k)
            for wire, start in ((q0, 9), (q1, 12)):
                rho = apply_ry_density(rho, block_params[start], wire, k)
                rho = module.apply_rz_density(rho, block_params[start + 1], wire, k)
                rho = apply_ry_density(rho, block_params[start + 2], wire, k)
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


def reset_output_dir(h_param: float, *, root: Path | None = None) -> Path:
    """Return the current circuit's result directory without reading old data.

    The former Rz/Rxx model used ``figs/dpqc_reset/h_<h>``. Keep that tree
    intact and use a distinct directory for the 60-angle U3-Cartan model.
    Numerical stages and visualization must share this path definition.
    """
    h_param = float(h_param)
    if not math.isfinite(h_param):
        raise ValueError("h_param must be finite.")
    root = Path.cwd() if root is None else Path(root)
    return root / "figs" / OUTPUT_FAMILY / OUTPUT_VARIANT / f"h_{h_param}"


def _configure_reset_output_paths(module: ModuleType) -> Path:
    """Redirect every split-stage result to the current circuit's directory."""
    save_dir = reset_output_dir(module.h_param)
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
        "schema_version": 3,
        "model_id": MODEL_ID,
        "h_param": float(h_param),
        "unitary_parameters_per_layer": UNITARY_PARAMS_PER_LAYER,
        "trainable_feed_forward_parameters": (
            NUM_TRAINABLE_FEED_FORWARD_PARAMS
        ),
        "total_parameter_formula": "60 * L",
        "unitary_block_parameter_order": list(BLOCK_PARAMETER_NAMES),
        "layer_pairs": [list(pair) for pair in LAYER_PAIRS],
        "parameters_shared": False,
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
        "unitary_block_parameter_order",
        "layer_pairs",
        "parameters_shared",
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

    Random-point analysis does not require training results. Existing numerical
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
    unidentified = next((save_dir / "numerical_results").rglob("*.npz"), None)
    if unidentified is not None:
        raise FileNotFoundError(
            "Reset-DPQC model metadata was not found; refusing to identify "
            f"existing numerical results without their metadata: {unidentified}. "
            f"Restore the matching {destination.name} before continuing."
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    return _write_model_metadata(save_dir, h_param)
