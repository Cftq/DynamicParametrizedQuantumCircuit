#!/usr/bin/env python
# coding: utf-8
"""Draw optimized measurement-outcome-0 Unitary-PQC circuits.

Run ``unitary_pqc_measured_0_overparam_compute.py`` first. This script reads
the saved
``vqe_optimization_results.npz`` archive and writes circuit figures without
rerunning VQE, QFIM, or any other numerical calculation.

Examples::

    python src/unitary_pqc/unitary_pqc_measured_0_overparam_draw_circuits.py
    python src/unitary_pqc/unitary_pqc_measured_0_overparam_draw_circuits.py --layers 1 4 8
    python src/unitary_pqc/unitary_pqc_measured_0_overparam_draw_circuits.py --layers 8 --fold -1
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from copy import copy as shallow_copy
from pathlib import Path


_MODULE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _MODULE_DIR.parent
_COMMON_DIR = _SRC_DIR / "common"
_PROJECT_ROOT = _SRC_DIR.parent
for _path in (_MODULE_DIR, _COMMON_DIR):
    _path_string = str(_path)
    if _path_string not in sys.path:
        sys.path.insert(0, _path_string)


import config_overparam as cfg


# Configure noninteractive drawing before importing Matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from plot import CIRCUIT_SAVE_PDF, CIRCUIT_SAVE_PNG, SAVE_DPI
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.visualization import circuit_drawer


NP_REAL_DTYPE = np.float64
DEFAULT_DRAW_DPI = min(int(SAVE_DPI), 100)

NUM_QUBITS = 5
NUM_BLOCKS = 4
PARAMS_PER_BLOCK = 3
N_PARAM_PER_LAYER = NUM_BLOCKS * PARAMS_PER_BLOCK
ANCILLA_QUBIT = 4
MEASUREMENT_OUTCOME = 0
ANSATZ_NAME = "unitary_pqc_measured_0"
LAYER_PAIRS = (
    (2, 3),
    (0, 2),
    (1, 3),
    (0, ANCILLA_QUBIT),
)

PARAMETER_FREE_GATE_LABELS = {
    "rz": r"$R_z$",
    "rx": r"$R_x$",
    "ry": r"$R_y$",
    "rxx": r"$R_{xx}$",
    "crx": r"$CR_x$",
    "cry": r"$CR_y$",
    "crz": r"$CR_z$",
    "cx": r"$CX$",
    "x": r"$X$",
    "z": r"$Z$",
    "h": r"$H$",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _circuit_fold(value: str) -> int:
    parsed = int(value)
    if parsed != -1 and parsed <= 0:
        raise argparse.ArgumentTypeError("fold must be -1 or a positive integer")
    return parsed


def _default_result_root() -> Path:
    return _PROJECT_ROOT / "figs" / ANSATZ_NAME / f"h_{cfg.H_PARAM}"


def _validate_archive_variant(archive, archive_path: Path) -> None:
    """Reject archives produced by a different measurement branch."""
    required = ("ansatz", "measurement_outcome", "num_params_per_layer")
    missing = tuple(key for key in required if key not in archive.files)
    if missing:
        raise KeyError(
            "VQE archive is missing variant metadata "
            f"({', '.join(missing)}): {archive_path}"
        )

    ansatz = np.asarray(archive["ansatz"])
    outcome = np.asarray(archive["measurement_outcome"])
    params_per_layer = np.asarray(archive["num_params_per_layer"])
    if ansatz.size != 1 or str(ansatz.reshape(-1)[0]) != ANSATZ_NAME:
        raise ValueError(
            f"VQE archive ansatz does not match {ANSATZ_NAME!r}: {archive_path}"
        )
    if (
        outcome.size != 1
        or not np.issubdtype(outcome.dtype, np.integer)
        or int(outcome.reshape(-1)[0]) != MEASUREMENT_OUTCOME
    ):
        raise ValueError(
            "VQE archive measurement_outcome does not match this drawer: "
            f"{archive_path}"
        )
    if (
        params_per_layer.size != 1
        or not np.issubdtype(params_per_layer.dtype, np.integer)
        or int(params_per_layer.reshape(-1)[0]) != N_PARAM_PER_LAYER
    ):
        raise ValueError(
            "VQE archive num_params_per_layer does not match this drawer: "
            f"{archive_path}"
        )


def _parse_cli_args(argv=None):
    result_root = _default_result_root()
    parser = argparse.ArgumentParser(
        description=(
            "Draw optimized measurement-outcome-0 Unitary-PQC circuits from "
            "saved VQE results; "
            "no numerical calculation is performed."
        )
    )
    parser.add_argument(
        "--input",
        "--archive",
        dest="input_path",
        type=Path,
        default=(
            result_root
            / "numerical_results"
            / "energy"
            / "vqe_optimization_results.npz"
        ),
        help="Path to vqe_optimization_results.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=result_root / "optimized_circuits",
        help="Directory in which optimized_circuit_L*.png is written.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=_positive_int,
        default=None,
        metavar="L",
        help="Layers to draw. By default, all layers stored in the archive.",
    )
    parser.add_argument(
        "--dpi",
        type=_positive_int,
        default=DEFAULT_DRAW_DPI,
        help=(
            f"PNG resolution (default: {DEFAULT_DRAW_DPI}). Use "
            f"--dpi {SAVE_DPI} for the configured high-resolution output."
        ),
    )
    parser.add_argument(
        "--fold",
        type=_circuit_fold,
        default=20,
        help=(
            "Maximum gates per row (default: 20). Use -1 only when an "
            "unfolded layout is required."
        ),
    )
    parser.add_argument(
        "--show-params",
        action="store_true",
        help="Show optimized numerical gate parameters in the circuit figure.",
    )
    return parser.parse_args(argv)


def qg_layer(
    circuit: QuantumCircuit,
    q0: int,
    q1: int,
    params: np.ndarray,
) -> None:
    """Append one Unitary-PQC two-qubit block."""
    circuit.rz(float(params[0]), q0)
    circuit.rz(float(params[1]), q1)
    circuit.rxx(float(params[2]), q0, q1)


def create_unitary_pqc(theta: np.ndarray, num_layers: int) -> QuantumCircuit:
    """Rebuild the optimized five-qubit circuit for drawing."""
    theta_array = np.asarray(theta, dtype=NP_REAL_DTYPE)
    expected_parameters = N_PARAM_PER_LAYER * int(num_layers)
    if theta_array.ndim != 1 or theta_array.size != expected_parameters:
        raise ValueError(
            f"theta must have shape ({expected_parameters},) for L={num_layers}; "
            f"got {theta_array.shape}."
        )

    circuit = QuantumCircuit(NUM_QUBITS)
    theta_layers = theta_array.reshape(num_layers, NUM_BLOCKS, PARAMS_PER_BLOCK)
    for layer_theta in theta_layers:
        for (q0, q1), params in zip(LAYER_PAIRS, layer_theta):
            qg_layer(circuit, q0, q1, params)

    return circuit


def make_parameter_free_qiskit_for_drawing(qc: QuantumCircuit) -> QuantumCircuit:
    """Copy a circuit while suppressing numerical parameter text."""
    if qc.num_clbits > 0:
        qc_draw = QuantumCircuit(qc.num_qubits, qc.num_clbits, name=qc.name)
    else:
        qc_draw = QuantumCircuit(qc.num_qubits, name=qc.name)
    qc_draw.global_phase = 0.0
    hidden_parameter = Parameter("")

    for instruction in qc.data:
        if hasattr(instruction, "operation"):
            operation = instruction.operation
            qargs = instruction.qubits
            cargs = instruction.clbits
        else:
            operation, qargs, cargs = instruction

        if hasattr(operation, "to_mutable"):
            operation_for_draw = operation.to_mutable()
        else:
            try:
                operation_for_draw = operation.copy()
            except Exception:
                operation_for_draw = shallow_copy(operation)

        gate_name = str(getattr(operation_for_draw, "name", "")).lower()
        if gate_name in PARAMETER_FREE_GATE_LABELS:
            operation_for_draw.label = PARAMETER_FREE_GATE_LABELS[gate_name]
        elif getattr(operation_for_draw, "params", None):
            operation_for_draw.label = rf"${gate_name}$"

        # Qiskit's Matplotlib drawer renders numeric params even with a custom
        # label. Empty symbolic parameters preserve native gate shapes while
        # hiding only the optimized values.
        if getattr(operation_for_draw, "params", None):
            operation_for_draw.params = [
                hidden_parameter for _ in operation_for_draw.params
            ]

        q_indices = [qc.find_bit(qubit).index for qubit in qargs]
        c_indices = [qc.find_bit(clbit).index for clbit in cargs]
        qc_draw.append(operation_for_draw, q_indices, c_indices)

    return qc_draw


def save_circuit_figure(
    qc: QuantumCircuit,
    outpath: Path,
    *,
    dpi: int,
    fold: int,
    hide_params: bool,
) -> tuple[Path, ...]:
    """Draw and save one Qiskit circuit according to the shared plot config."""
    if hide_params:
        qc_for_draw = make_parameter_free_qiskit_for_drawing(qc)
        drawer_style = {"displaytext": PARAMETER_FREE_GATE_LABELS}
    else:
        qc_for_draw = qc
        drawer_style = None

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    root = outpath.with_suffix("")
    saved_paths = []
    fig = circuit_drawer(
        qc_for_draw,
        output="mpl",
        fold=fold,
        style=drawer_style,
    )
    try:
        if CIRCUIT_SAVE_PNG:
            estimated_rgba_bytes = float(np.prod(fig.get_size_inches() * dpi) * 4)
            if estimated_rgba_bytes >= 2.0 * 1024**3:
                warnings.warn(
                    "The requested PNG may require approximately "
                    f"{estimated_rgba_bytes / 1024**3:.1f} GiB of raw image "
                    "memory. Reduce --dpi or select fewer --layers if needed.",
                    UserWarning,
                    stacklevel=2,
                )
            png_path = root.with_suffix(".png")
            fig.savefig(
                png_path,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.02,
            )
            saved_paths.append(png_path)
        if CIRCUIT_SAVE_PDF:
            pdf_path = root.with_suffix(".pdf")
            fig.savefig(
                pdf_path,
                bbox_inches="tight",
                pad_inches=0.02,
            )
            saved_paths.append(pdf_path)
    finally:
        plt.close(fig)

    return tuple(saved_paths)


def load_best_theta_by_layer(
    archive_path: Path,
    requested_layers=None,
) -> dict[int, dict[str, object]]:
    """Load and validate the best optimized parameters saved by compute."""
    archive_path = Path(archive_path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"VQE archive was not found: {archive_path}")

    with np.load(archive_path, allow_pickle=False) as archive:
        _validate_archive_variant(archive, archive_path)
        if "layers" not in archive.files:
            raise KeyError(f"Missing key 'layers' in {archive_path}")

        layer_array = np.asarray(archive["layers"], dtype=np.int64)
        if layer_array.ndim != 1 or layer_array.size == 0:
            raise ValueError("layers must be a nonempty one-dimensional array.")
        available_layers = tuple(int(layer) for layer in layer_array.tolist())
        if any(layer <= 0 for layer in available_layers):
            raise ValueError("layers must contain only positive integers.")
        if len(set(available_layers)) != len(available_layers):
            raise ValueError("layers must not contain duplicates.")

        if requested_layers is None:
            selected_layers = available_layers
        else:
            selected_layers = tuple(int(layer) for layer in requested_layers)
            if len(set(selected_layers)) != len(selected_layers):
                raise ValueError("--layers must not contain duplicates.")
            missing_layers = [
                layer for layer in selected_layers if layer not in available_layers
            ]
            if missing_layers:
                raise ValueError(
                    f"Requested layers are absent from the archive: {missing_layers}. "
                    f"Available layers: {list(available_layers)}"
                )

        archived_num_runs = None
        if "num_runs" in archive.files:
            num_runs_array = np.asarray(archive["num_runs"])
            if num_runs_array.size != 1:
                raise ValueError("num_runs must be scalar when present.")
            archived_num_runs = int(num_runs_array.reshape(()))
            if archived_num_runs <= 0:
                raise ValueError("num_runs must be positive when present.")

        best_by_layer = {}
        for layer in selected_layers:
            theta_key = f"L{layer}_best_theta"
            if theta_key not in archive.files:
                raise KeyError(f"Missing key '{theta_key}' in {archive_path}")

            best_theta = np.asarray(archive[theta_key], dtype=NP_REAL_DTYPE)
            expected_parameters = N_PARAM_PER_LAYER * layer
            if best_theta.ndim != 1 or best_theta.size != expected_parameters:
                raise ValueError(
                    f"{theta_key} must have shape ({expected_parameters},); "
                    f"got {best_theta.shape}."
                )
            if not np.all(np.isfinite(best_theta)):
                raise FloatingPointError(
                    f"{theta_key} contains non-finite parameter values."
                )

            best_run = None
            best_energy = None
            energy_key = f"L{layer}_energy_traces"
            theta_history_key = f"L{layer}_theta_history"
            if energy_key in archive.files:
                energy_traces = np.asarray(
                    archive[energy_key],
                    dtype=NP_REAL_DTYPE,
                )
                if energy_traces.ndim != 2 or energy_traces.shape[1] == 0:
                    raise ValueError(
                        f"{energy_key} must have shape (runs, steps); "
                        f"got {energy_traces.shape}."
                    )
                if energy_traces.shape[0] == 0:
                    raise ValueError(f"No VQE runs are stored for L={layer}.")
                if (
                    archived_num_runs is not None
                    and energy_traces.shape[0] != archived_num_runs
                ):
                    raise ValueError(
                        f"Run-count mismatch for L={layer}: "
                        f"num_runs={archived_num_runs}, {energy_key} contains "
                        f"{energy_traces.shape[0]} runs."
                    )
                final_energies = energy_traces[:, -1]
                finite_run_indices = np.flatnonzero(np.isfinite(final_energies))
                if finite_run_indices.size == 0:
                    raise FloatingPointError(
                        f"No finite final VQE energy is stored for L={layer}."
                    )
                local_best = int(np.argmin(final_energies[finite_run_indices]))
                best_run = int(finite_run_indices[local_best])
                best_energy = float(final_energies[best_run])

                if theta_history_key in archive.files:
                    theta_history = np.asarray(
                        archive[theta_history_key],
                        dtype=NP_REAL_DTYPE,
                    )
                    expected_shape = (
                        energy_traces.shape[0],
                        expected_parameters,
                    )
                    if theta_history.shape != expected_shape:
                        raise ValueError(
                            f"{theta_history_key} must have shape "
                            f"{expected_shape}; got {theta_history.shape}."
                        )
                    if not np.array_equal(theta_history[best_run], best_theta):
                        warnings.warn(
                            f"{theta_key} differs from the lowest-energy "
                            f"entry in {theta_history_key}; drawing the "
                            "explicitly saved best parameters.",
                            UserWarning,
                            stacklevel=2,
                        )
                        best_run = None
                        best_energy = None

            best_by_layer[layer] = {
                "run": best_run,
                "energy": best_energy,
                "theta": best_theta.copy(),
            }

    return best_by_layer


def main(argv=None) -> int:
    args = _parse_cli_args(argv)
    best_by_layer = load_best_theta_by_layer(
        args.input_path,
        requested_layers=args.layers,
    )

    if args.fold == -1 and any(layer >= 20 for layer in best_by_layer):
        warnings.warn(
            "Unfolded high-layer circuits can require several gigabytes of "
            "image memory. Use --fold 20 (or another positive value) if needed.",
            UserWarning,
            stacklevel=2,
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer, best in best_by_layer.items():
        qc = create_unitary_pqc(
            np.asarray(best["theta"], dtype=NP_REAL_DTYPE),
            num_layers=layer,
        )
        saved_paths = save_circuit_figure(
            qc,
            output_dir / f"optimized_circuit_L{layer}.png",
            dpi=args.dpi,
            fold=args.fold,
            hide_params=not args.show_params,
        )
        saved_text = ", ".join(str(path) for path in saved_paths)
        if best["run"] is None or best["energy"] is None:
            selection_text = "loaded saved best parameters"
        else:
            selection_text = (
                f"loaded saved best parameters from run {best['run']} "
                f"(final energy={best['energy']:.16g})"
            )
        print(f"L={layer}: {selection_text}; saved {saved_text}")

    print(f"Saved {len(best_by_layer)} optimized circuit(s) to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
