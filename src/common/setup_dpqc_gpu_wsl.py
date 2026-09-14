#!/usr/bin/env python
"""Install an isolated DPQC CUDA environment and configure Windows launchers.

Run inside the chosen WSL2 distribution, from the project directory:
    python3 src/common/setup_dpqc_gpu_wsl.py
Existing Python environments and production numerical results are untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=Path.home() / ".venvs" / "dpqc-gpu")
    parser.add_argument("--distribution", default=os.environ.get("WSL_DISTRO_NAME"))
    parser.add_argument("--skip-install", action="store_true",
                        help="Only validate and register an already-installed environment.")
    args = parser.parse_args(argv)
    if sys.platform != "linux" or not args.distribution:
        parser.error("Run this setup inside WSL2 (or provide its distribution name).")
    root = Path(__file__).resolve().parents[2]
    environment = args.venv.expanduser().resolve()
    python = environment / "bin" / "python"
    if not args.skip_install:
        if not python.is_file():
            subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        subprocess.run([
            str(python), "-m", "pip", "install", "-r",
            str(root / "src" / "common" / "requirements-dpqc-gpu.txt"),
        ], check=True)
    validation = (
        "import os; os.environ['JAX_PLATFORMS']='cuda'; "
        "os.environ['JAX_ENABLE_X64']='1'; "
        "os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false'); "
        "import jax, jax.numpy as jnp, optax, tensorcircuit; "
        "jax.config.update('jax_enable_x64', True); "
        "assert jax.default_backend() == 'gpu'; "
        "x=jnp.ones((32,32),dtype=jnp.float64); "
        "y=(x@x).block_until_ready(); assert y.dtype == jnp.float64; "
        "print('Validated CUDA devices:', jax.devices())"
    )
    subprocess.run([str(python), "-c", validation], check=True)
    config = root / ".dpqc-gpu-wsl.json"
    config.write_text(json.dumps({
        "distribution": args.distribution, "python": str(python),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Configured Windows DPQC GPU launchers: {config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
