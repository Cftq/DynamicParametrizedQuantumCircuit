"""Select the JAX device before importing the numerical DPQC stages.

This module deliberately uses only the standard library.  Entry points can
parse ``--help`` without importing JAX or probing a CUDA driver.  An explicit
GPU request never falls back to CPU; native Windows needs CUDA-enabled JAX in
WSL2, while Linux can use a CUDA-enabled Python environment directly.
"""

import argparse
import importlib
import operator
import os


DEVICE_CHOICES = ("auto", "cpu", "gpu")


def _device_name(device=None):
    value = os.environ.get("DPQC_DEVICE", "auto") if device is None else device
    if value not in DEVICE_CHOICES:
        raise ValueError(
            f"Unknown DPQC device {value!r}; choose auto, cpu, or gpu."
        )
    return value


def add_device_argument(parser: argparse.ArgumentParser):
    """Add the common device option to a lightweight entry-point parser."""
    return parser.add_argument(
        "--device",
        choices=DEVICE_CHOICES,
        default=_device_name(),
        help=(
            "JAX device: auto uses GPU when available for VQE and CPU for "
            "QFIM/Hessian; cpu/gpu explicitly select one device for all "
            "requested stages (default: DPQC_DEVICE environment variable, "
            "otherwise auto)."
        ),
    )


def resolve_stage_device(device, stage):
    """Keep analysis on CPU by default while retaining automatic VQE devices.

    Resolve this before WSL routing or JAX initialization. Explicit device
    choices (including DPQC_DEVICE) override the per-stage automatic policy.
    Each numerical stage must run in a fresh process when devices differ.
    """
    if stage not in ("vqe", "qfim", "hessian"):
        raise ValueError(f"Unsupported numerical stage: {stage!r}.")
    chosen = _device_name(device)
    if chosen == "auto" and stage in ("qfim", "hessian"):
        return "cpu"
    return chosen


def configure_jax_backend(device=None):
    """Set process configuration before importing JAX, returning the choice.

    ``auto`` respects any externally supplied JAX platform configuration.
    Explicit choices replace platform restrictions inherited by subprocesses.
    CUDA memory grows as needed unless the user supplies another setting.
    Changing devices after JAX has initialized requires a fresh process;
    :func:`initialize_jax_backend` detects a conflicting actual backend.
    """
    chosen = _device_name(device)
    os.environ["DPQC_DEVICE"] = chosen
    if chosen == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    elif chosen == "gpu":
        # Listing only CUDA prohibits JAX's usual fallback to a CPU backend.
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
    if chosen in ("auto", "gpu"):
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return chosen


def initialize_jax_backend():
    """Initialize JAX, validate the requested device, and report real devices.

    Call immediately after ``import jax``, before constructing Hamiltonians,
    parameters, optimizer state, or compiled computations.  This keeps all
    those arrays on the selected default device, including Adam state.
    """
    chosen = _device_name()
    try:
        jax = importlib.import_module("jax")
        platform = jax.default_backend()
        devices = jax.local_devices()
    except Exception as exc:
        if chosen == "gpu":
            raise RuntimeError(
                "DPQC requested --device gpu, but JAX could not initialize "
                "CUDA. Use a CUDA-enabled JAX environment with a supported "
                "NVIDIA driver (WSL2 on Windows). Training has not started. "
                f"Original error: {exc}"
            ) from exc
        raise RuntimeError(
            f"DPQC could not initialize the {chosen!r} JAX backend: {exc}"
        ) from exc

    expected = "gpu" if chosen == "gpu" else "cpu" if chosen == "cpu" else None
    if expected is not None and platform != expected:
        raise RuntimeError(
            f"DPQC requested --device {chosen}, but JAX's active backend is "
            f"{platform!r}. Select the device before importing JAX and run "
            "in a fresh Python process. An explicit GPU request does not "
            "fall back to CPU."
        )
    if not devices or any(device.platform != platform for device in devices):
        raise RuntimeError(
            f"JAX reported no consistent local devices for backend {platform!r}."
        )

    descriptions = ", ".join(
        f"{getattr(device, 'device_kind', str(device))} "
        f"(id={getattr(device, 'id', '?')})"
        for device in devices
    )
    print(f"[DPQC] JAX backend: {platform}; devices: {descriptions}", flush=True)
    return devices[0]


def effective_vqe_batch_size(requested, num_runs):
    """Limit the compiled trial batch to real trials, avoiding wasted padding."""
    try:
        requested_int = operator.index(requested)
        runs_int = operator.index(num_runs)
    except TypeError as exc:
        raise ValueError("VQE batch size and number of runs must be positive integers.") from exc
    if isinstance(requested, bool) or isinstance(num_runs, bool):
        raise ValueError("VQE batch size and number of runs must be positive integers.")
    if requested_int <= 0 or runs_int <= 0:
        raise ValueError("VQE batch size and number of runs must be positive integers.")
    return min(requested_int, runs_int)
