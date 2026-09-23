#!/usr/bin/env python
# coding: utf-8
"""Read saved U3-Cartan VQE energies without a quantum runtime.

Only metadata and energy histories are loaded. In the current saver, each
energy sample precedes an optimizer update, so the last stored sample is
not necessarily the energy at the final saved parameter vector. This
module never evaluates parameter vectors or substitutes a minimum energy.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np

if __package__:
    from .unitary_pqc_model import (
        ANSATZ_NAME,
        NUM_PARAMS_PER_LAYER,
        OUTPUT_VARIANT,
    )
else:
    from unitary_pqc_model import (
        ANSATZ_NAME,
        NUM_PARAMS_PER_LAYER,
        OUTPUT_VARIANT,
    )


def _finite_real_scalar(value, *, name: str) -> float:
    value = np.asarray(value)
    if (
        value.shape != ()
        or value.dtype.kind not in "iuf"
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be one finite real scalar.")
    result = float(value.item())
    if not np.isfinite(result):
        raise ValueError(f"{name} must be representable as a finite float64.")
    return result


def _positive_integer_scalar(value, *, name: str) -> int:
    value = np.asarray(value)
    if value.shape != () or value.dtype.kind not in "iu" or value <= 0:
        raise ValueError(f"{name} must be a positive integer scalar.")
    return int(value.item())


def load_unitary_final_energies(
    archive_path: str | PathLike[str],
    *,
    expected_h_param: float,
) -> tuple[dict[int, np.ndarray], float, dict[str, object]]:
    """Return each run's last stored energy, saved ground energy, and provenance.

    The archive must belong to the 60-parameter-per-layer U3-Cartan model
    at ``expected_h_param``. Histories with ``steps`` samples (the current
    pre-update saver) and ``steps + 1`` samples are supported, provided all
    layers have the same history length. Parameter and QFIM arrays are
    deliberately left unopened, including any legacy object arrays.

    Missing or incompatible archives raise an error; no training, fallback
    archive search, directory creation, or file writes are performed.
    """
    archive_path = Path(archive_path)
    expected_h = _finite_real_scalar(expected_h_param, name="expected_h_param")
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Saved Unitary PQC VQE energy histories were not found: {archive_path}"
        )

    with np.load(archive_path, allow_pickle=False) as archive:
        required_keys = (
            "ansatz",
            "num_params_per_layer",
            "h_param",
            "layers",
            "num_runs",
            "steps",
            "smallest_eigval",
        )
        missing = [key for key in required_keys if key not in archive]
        if missing:
            raise KeyError(
                "VQE archive is missing required metadata: " + ", ".join(missing)
            )

        ansatz = np.asarray(archive["ansatz"])
        if ansatz.shape != () or ansatz.dtype.kind not in "US":
            raise ValueError("VQE ansatz must be a scalar string.")
        if str(ansatz.item()) != ANSATZ_NAME:
            raise ValueError(
                f"VQE ansatz mismatch: {ansatz.item()!r} != {ANSATZ_NAME!r}."
            )

        parameter_count = _positive_integer_scalar(
            archive["num_params_per_layer"], name="VQE num_params_per_layer"
        )
        if parameter_count != NUM_PARAMS_PER_LAYER:
            raise ValueError(
                "VQE num_params_per_layer mismatch: "
                f"{parameter_count} != {NUM_PARAMS_PER_LAYER} "
                f"for {OUTPUT_VARIANT}."
            )

        archived_h = _finite_real_scalar(archive["h_param"], name="VQE h_param")
        if archived_h != expected_h:
            raise ValueError(f"VQE h_param mismatch: {archived_h} != {expected_h}.")
        saved_ground = _finite_real_scalar(
            archive["smallest_eigval"], name="VQE smallest_eigval"
        )
        runs = _positive_integer_scalar(archive["num_runs"], name="VQE num_runs")
        steps = _positive_integer_scalar(archive["steps"], name="VQE steps")
        layers = np.asarray(archive["layers"])
        if (
            layers.ndim != 1
            or layers.size == 0
            or layers.dtype.kind not in "iu"
            or np.any(layers <= 0)
            or np.unique(layers).size != layers.size
        ):
            raise ValueError("VQE layers must contain unique positive integers.")

        final_energies = {}
        history_length = None
        for raw_layer in layers:
            layer = int(raw_layer)
            key = f"L{layer}_energy_traces"
            if key not in archive:
                raise KeyError(f"VQE archive is missing required array: {key}")
            values = np.asarray(archive[key])
            if (
                values.ndim != 2
                or values.shape[0] != runs
                or values.shape[1] not in (steps, steps + 1)
                or values.dtype.kind not in "iuf"
            ):
                raise ValueError(
                    f"VQE {key} must be a real array with {runs} runs and "
                    f"{steps} or {steps + 1} stored energies per run."
                )
            if history_length is not None and values.shape[1] != history_length:
                raise ValueError("VQE energy history lengths must match across layers.")
            history_length = values.shape[1]
            final_values = np.array(values[:, -1], dtype=np.float64, copy=True)
            if not np.all(np.isfinite(final_values)):
                raise ValueError(f"VQE {key} final energy samples must be finite.")
            final_energies[layer] = final_values

    return final_energies, saved_ground, {
        "source_archive": str(archive_path.resolve()),
        "source_energy_definition": "last saved energy-trace sample per run",
        "optimizer_steps": steps,
        "num_stored_energy_samples": history_length,
        "ansatz": ANSATZ_NAME,
        "num_params_per_layer": parameter_count,
        "output_variant": OUTPUT_VARIANT,
    }
