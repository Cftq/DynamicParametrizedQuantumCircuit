"""Route Windows DPQC commands to the project's configured WSL environment.

The ignored project-local .dpqc-gpu-wsl.json is written by GPU setup after
CUDA validation. Command arguments are passed as an argv list, without a shell.
Linux execution and unconfigured automatic CPU execution remain native.
"""

from __future__ import annotations

import json
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CONFIG = _PROJECT_ROOT / ".dpqc-gpu-wsl.json"


def _linux_path(value: str) -> str:
    """Convert a rooted drive path; leave flags and Linux paths unchanged."""
    if re.match(r"^[A-Za-z]:[\\/]", value):
        path = PureWindowsPath(value)
        return "/mnt/" + path.drive[0].lower() + "/" + "/".join(path.parts[1:])
    return value


def _forward_arguments(argv, device, *, preserve_auto=False):
    forwarded = []
    iterator = iter(argv)
    for argument in iterator:
        if argument == "--device":
            next(iterator, None)
            continue
        if argument.startswith("--device="):
            continue
        if argument.startswith("--") and "=" in argument:
            flag, value = argument.split("=", 1)
            forwarded.append(flag + "=" + _linux_path(value))
        else:
            forwarded.append(_linux_path(argument))
    # VQE auto selects the verified GPU. Multi-stage launchers preserve auto
    # so their child stages can select CPU for QFIM/Hessian independently.
    selected = "gpu" if device == "auto" and not preserve_auto else device
    return forwarded + ["--device", selected]


def maybe_relaunch_in_wsl(script_path, argv, device, *, preserve_auto=False):
    """Return a WSL child's status, or None when native execution is selected.

CPU comparisons also use the configured environment, so they use identical
JAX/Optax versions. DPQC_USE_WSL=0 explicitly selects native Windows Python.
"""
    if sys.platform != "win32" or os.environ.get("DPQC_USE_WSL") == "0":
        return None
    if not RUNTIME_CONFIG.is_file():
        if device == "gpu":
            raise RuntimeError(
                "Native Windows JAX cannot use CUDA. Configure the WSL GPU "
                "environment with src/common/setup_dpqc_gpu_wsl.py first."
            )
        return None
    config = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    distribution = config.get("distribution")
    python = config.get("python")
    if not isinstance(distribution, str) or not distribution.strip():
        raise ValueError(f"Invalid WSL distribution in {RUNTIME_CONFIG}")
    if not isinstance(python, str) or not python.startswith("/"):
        raise ValueError(f"Invalid Linux Python path in {RUNTIME_CONFIG}")
    arguments = list(sys.argv[1:] if argv is None else argv)
    environment = [
        name + "=" + os.environ[name]
        for name in (
            "DPQC_DENSITY_KERNEL", "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
        )
        if name in os.environ
    ]
    command = [
        "wsl.exe", "--distribution", distribution,
        "--cd", str(Path.cwd()), "--exec",
        *(["env", *environment] if environment else []), python,
        _linux_path(str(Path(script_path).resolve())),
        *_forward_arguments(arguments, device, preserve_auto=preserve_auto),
    ]
    selected = "gpu" if device == "auto" and not preserve_auto else device
    print(f"DPQC runtime: WSL2 {distribution}; requested device: {selected}", flush=True)
    return int(subprocess.run(command, check=False).returncode)
