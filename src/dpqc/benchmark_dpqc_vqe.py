#!/usr/bin/env python
"""Benchmark the actual DPQC Adam loop without changing production results.

CPU and GPU runs use identical float64 initial points. Compilation, warm
execution and host transfer are timed separately, with device synchronization.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

_MODULE_DIR = Path(__file__).resolve().parent
for _path in (_MODULE_DIR, _MODULE_DIR.parent / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def main(argv=None):
    if os.environ.get("DPQC_BENCHMARK_TRACE") == "1":
        import faulthandler
        faulthandler.dump_traceback_later(30, repeat=True)
    from dpqc_backend import add_device_argument, configure_jax_backend, effective_vqe_batch_size
    from dpqc_wsl import maybe_relaunch_in_wsl

    parser = argparse.ArgumentParser(description=__doc__)
    add_device_argument(parser)
    parser.add_argument("--family", choices=("dpqc", "dpqc_reset"), required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--h-param", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/dpqc_gpu_benchmark"))
    args = parser.parse_args(argv)
    if min(*args.layers, args.steps, args.trials, args.repeats) <= 0:
        parser.error("Layers, steps, trials and repeats must be positive.")
    routed = maybe_relaunch_in_wsl(__file__, argv, args.device)
    if routed is not None:
        return routed
    configure_jax_backend(args.device)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Stage imports create empty output folders relative to CWD. Isolate them.
    work = output / "runtime" / f"{args.family}_{args.device}"
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)

    import config_overparam as cfg
    cfg.NUM_RUNS = args.trials
    cfg.VQE_BATCH_SIZE = args.trials
    cfg.STEPS = args.steps
    cfg.H_PARAM = args.h_param
    cfg.DPQC_VQE_OPTIMIZER = "adam"
    if args.family == "dpqc_reset":
        import dpqc_reset_model as model
        stage = model._load_base_stage_module(
            "vqe", h_param=args.h_param, vqe_batch_size=args.trials, device=args.device,
        )
        model._install_reset_model(stage)
    else:
        import DPQC_overparam_vqe as stage
    print("Numerical stage initialized.", flush=True)
    import jax
    import numpy as np
    optimizer = stage.build_dpqc_vqe_optimizer("adam", cfg.LEARNING_RATE)
    summary = {
        "family": args.family, "backend": jax.default_backend(),
        "devices": [d.device_kind for d in jax.devices()], "jax_version": jax.__version__,
        "steps": args.steps, "trials": args.trials, "h_param": args.h_param,
        "precision": "float64/complex128", "optimizer": "adam",
        "density_kernel": stage.DENSITY_KERNEL,
        "learning_rate": cfg.LEARNING_RATE, "initial_seed": 2026,
        "production_results_modified": False, "layers": [],
    }
    for layer in args.layers:
        theta_host = np.random.default_rng(2026 + layer).uniform(
            -np.pi, np.pi, (args.trials, stage.n_param_per_layer * layer),
        ).astype(np.float64)
        theta = jax.device_put(theta_host)
        theta.block_until_ready()
        runner = stage.make_vqe_batch_runner(
            layer, num_steps=args.steps,
            sample_iterations=np.unique([1, args.steps]).astype(np.int64),
            optimizer=optimizer,
        )
        start = time.perf_counter()
        compiled = runner.lower(theta).compile()
        compilation = time.perf_counter() - start
        print(f"L{layer}: compilation {compilation:.3f}s; warming up...", flush=True)
        jax.block_until_ready(compiled(theta))
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            result = jax.block_until_ready(compiled(theta))
            times.append(time.perf_counter() - start)
        start = time.perf_counter()
        host = jax.device_get(result)
        transfer = time.perf_counter() - start
        for values in host:
            if values.dtype != np.float64 or not np.all(np.isfinite(values)):
                raise AssertionError("Adam output must remain finite float64.")
        name = f"{args.family}_{jax.default_backend()}_L{layer}_S{args.steps}_R{args.trials}"
        np.savez_compressed(output / f"{name}.npz", theta_initial=theta_host,
                            theta_final=host[0], energy_trace=host[1],
                            grad_norm_trace=host[2], theta_samples=host[3])
        row = {
            "layer": layer, "compile_seconds": compilation,
            "execution_seconds": times, "median_seconds": float(np.median(times)),
            "host_transfer_seconds": transfer,
            "effective_batch_size": effective_vqe_batch_size(args.trials, args.trials),
            "final_energy_mean": float(np.mean(host[1][:, -1])),
        }
        summary["layers"].append(row)
        print(json.dumps(row), flush=True)
        del compiled, runner, result
        jax.clear_caches()
    path = output / f"{args.family}_{jax.default_backend()}_S{args.steps}_R{args.trials}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved benchmark: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
