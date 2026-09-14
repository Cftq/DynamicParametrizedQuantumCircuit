"""Check GPU stage setup without importing JAX or running quantum jobs."""

from contextlib import redirect_stdout
import importlib
import io
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


_MODULE_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _MODULE_DIR.parent / "common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
_BACKEND = importlib.import_module("dpqc_backend")


class _StopBeforeNumerics(Exception):
    """End a setup probe before any circuit, optimizer or array is created."""


def _fake_backend():
    backend = ModuleType("dpqc_backend")
    backend.add_device_argument = _BACKEND.add_device_argument
    backend.resolve_stage_device = _BACKEND.resolve_stage_device
    backend.configure_jax_backend = Mock()
    backend.initialize_jax_backend = Mock()
    backend.effective_vqe_batch_size = _BACKEND.effective_vqe_batch_size
    return backend


class DPQCGPUStageSetupTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"DPQC_DEVICE": "auto"}))

    def test_device_configuration_and_initialization_precede_numerical_work(self):
        for stage in ("vqe", "qfim", "hessian"):
            for device in ("auto", "cpu", "gpu"):
                with self.subTest(stage=stage, device=device):
                    events = []
                    backend = _fake_backend()
                    backend.configure_jax_backend.side_effect = (
                        lambda selected: events.append(("configure", selected))
                    )

                    def stop_after_initialization():
                        events.append(("initialize",))
                        raise _StopBeforeNumerics

                    backend.initialize_jax_backend.side_effect = stop_after_initialization
                    wsl = ModuleType("dpqc_wsl")
                    wsl.maybe_relaunch_in_wsl = Mock(
                        side_effect=lambda *args: events.append(("route", args[2]))
                    )
                    script = _MODULE_DIR / f"DPQC_overparam_{stage}.py"
                    with patch.dict(sys.modules, {
                        "dpqc_backend": backend,
                        "dpqc_wsl": wsl,
                        "jax": ModuleType("jax"),
                    }), patch.object(sys, "argv", [str(script), "--device", device]):
                        with self.assertRaises(_StopBeforeNumerics):
                            runpy.run_path(str(script), run_name="__main__")
                    expected = "cpu" if device == "auto" and stage != "vqe" else device
                    self.assertEqual(events, [
                        ("route", expected), ("configure", expected), ("initialize",),
                    ])

    def test_omitted_device_uses_stage_default_or_environment_before_wsl(self):
        for inherited in ("auto", "cpu", "gpu"):
            for stage in ("vqe", "qfim", "hessian"):
                with self.subTest(inherited=inherited, stage=stage):
                    backend = _fake_backend()
                    wsl = ModuleType("dpqc_wsl")
                    wsl.maybe_relaunch_in_wsl = Mock(return_value=0)
                    script = _MODULE_DIR / f"DPQC_overparam_{stage}.py"
                    with patch.dict(sys.modules, {
                        "dpqc_backend": backend, "dpqc_wsl": wsl,
                    }), patch.dict(os.environ, {"DPQC_DEVICE": inherited}), patch.object(
                        sys, "argv", [str(script)],
                    ), self.assertRaises(SystemExit) as result:
                        runpy.run_path(str(script), run_name="__main__")
                    self.assertEqual(result.exception.code, 0)
                    expected = "cpu" if inherited == "auto" and stage != "vqe" else inherited
                    self.assertEqual(wsl.maybe_relaunch_in_wsl.call_args.args[2], expected)
                    backend.configure_jax_backend.assert_not_called()
                    backend.initialize_jax_backend.assert_not_called()

    def test_imported_analysis_defaults_to_cpu_before_backend_initialization(self):
        for stage in ("qfim", "hessian"):
            for inherited, expected in (("auto", "cpu"), ("gpu", "gpu")):
                with self.subTest(stage=stage, inherited=inherited):
                    backend = _fake_backend()
                    backend.initialize_jax_backend.side_effect = _StopBeforeNumerics
                    wsl = ModuleType("dpqc_wsl")
                    wsl.maybe_relaunch_in_wsl = Mock()
                    script = _MODULE_DIR / f"DPQC_overparam_{stage}.py"
                    with patch.dict(sys.modules, {
                        "dpqc_backend": backend, "dpqc_wsl": wsl,
                        "jax": ModuleType("jax"),
                    }), patch.dict(os.environ, {"DPQC_DEVICE": inherited}):
                        with self.assertRaises(_StopBeforeNumerics):
                            runpy.run_path(str(script), run_name="import_probe")
                    backend.configure_jax_backend.assert_called_once_with(expected)
                    backend.initialize_jax_backend.assert_called_once_with()
                    wsl.maybe_relaunch_in_wsl.assert_not_called()

    def test_wsl_route_returns_before_initializing_or_computing(self):
        for stage in ("vqe", "qfim", "hessian"):
            with self.subTest(stage=stage):
                backend = _fake_backend()
                wsl = ModuleType("dpqc_wsl")
                wsl.maybe_relaunch_in_wsl = Mock(return_value=21)
                script = _MODULE_DIR / f"DPQC_overparam_{stage}.py"
                with patch.dict(sys.modules, {
                    "dpqc_backend": backend, "dpqc_wsl": wsl,
                }), patch.object(sys, "argv", [str(script), "--device", "gpu"]):
                    with self.assertRaises(SystemExit) as result:
                        runpy.run_path(str(script), run_name="__main__")
                self.assertEqual(result.exception.code, 21)
                backend.configure_jax_backend.assert_not_called()
                backend.initialize_jax_backend.assert_not_called()

    def test_vqe_setup_clamps_compiled_batch_to_actual_trials(self):
        # Import fakes are enough to reach the real stage's batch setup. Stop
        # there: no optimizer or circuit function is allowed to execute.
        for requested, runs, expected in ((128, 3, 3), (2, 3, 2), (128, 1, 1)):
            with self.subTest(requested=requested, runs=runs):
                backend = _fake_backend()
                observed = []

                def capture_effective_batch(requested_count, trial_count):
                    effective = _BACKEND.effective_vqe_batch_size(
                        requested_count, trial_count
                    )
                    observed.append((requested_count, trial_count, effective))
                    raise _StopBeforeNumerics

                backend.effective_vqe_batch_size = capture_effective_batch
                config = ModuleType("config_overparam")
                config.H_PARAM = 0.1
                config.VQE_BATCH_SIZE = 5
                config.TOLERANCE = 1e-8
                config.STEPS = 2
                config.NUM_RUNS = runs
                jax = ModuleType("jax")
                jax.config = SimpleNamespace(update=Mock())
                numpy = ModuleType("numpy")
                numpy.float64, numpy.complex128, numpy.int64 = float, complex, int
                jax.numpy = numpy
                tensorcircuit = ModuleType("tensorcircuit")
                tensorcircuit.set_backend = Mock()
                tensorcircuit.set_dtype = Mock()
                tqdm = ModuleType("tqdm.auto")
                tqdm.tqdm = Mock()
                common = ModuleType("dpqc_overparam_common")
                for name in (
                    "_thr_tag", "build_dpqc_vqe_optimizer", "build_H_matrix_jax",
                    "build_layer_list", "dpqc_vqe_optimizer_display_name",
                    "hamiltonian_terms", "load_npz_result",
                    "normalize_dpqc_vqe_optimizer_name", "rho_zero_state",
                    "threshold_psd_eigvals_for_rank",
                ):
                    setattr(common, name, Mock(side_effect=AssertionError(
                        "Numerical helper must not run in setup probe: " + name
                    )))
                wsl = ModuleType("dpqc_wsl")
                wsl.maybe_relaunch_in_wsl = Mock(return_value=None)
                script = _MODULE_DIR / "DPQC_overparam_vqe.py"
                with patch.dict(sys.modules, {
                    "dpqc_backend": backend, "dpqc_wsl": wsl,
                    "config_overparam": config, "jax": jax, "jax.numpy": numpy,
                    "numpy": numpy, "optax": ModuleType("optax"),
                    "tensorcircuit": tensorcircuit, "tqdm.auto": tqdm,
                    "dpqc_overparam_common": common,
                }), patch.object(sys, "argv", [
                    str(script), "--device", "gpu", "--vqe-batch-size", str(requested),
                ]), redirect_stdout(io.StringIO()):
                    with self.assertRaises(_StopBeforeNumerics):
                        runpy.run_path(str(script), run_name="__main__")
                self.assertEqual(observed, [(requested, runs, expected)])
                backend.configure_jax_backend.assert_called_once_with("gpu")
                backend.initialize_jax_backend.assert_called_once_with()

    def test_every_stage_help_lists_device_without_loading_numerical_modules(self):
        probe = r'''
import importlib.abc
from pathlib import Path
import runpy
import sys

blocked = {"jax", "jaxlib", "numpy", "scipy", "matplotlib", "qiskit",
           "optax", "tensorcircuit"}
class RejectNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError("Unexpected numerical import: " + fullname)
sys.meta_path.insert(0, RejectNumericalImports())
script = Path(sys.argv[1])
sys.argv = [str(script), "--help"]
try:
    runpy.run_path(str(script), run_name="__main__")
except SystemExit as result:
    assert result.code == 0, result.code
else:
    raise AssertionError("--help did not exit before numerical work")
'''
        with tempfile.TemporaryDirectory() as directory:
            for prefix in ("DPQC_overparam", "DPQC_overparam_reset"):
                for stage in ("compute", "vqe", "qfim", "hessian"):
                    with self.subTest(prefix=prefix, stage=stage):
                        script = _MODULE_DIR / f"{prefix}_{stage}.py"
                        result = subprocess.run(
                            [sys.executable, "-c", probe, str(script)],
                            cwd=directory, capture_output=True, text=True,
                            check=False, timeout=30,
                        )
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn("--device", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
