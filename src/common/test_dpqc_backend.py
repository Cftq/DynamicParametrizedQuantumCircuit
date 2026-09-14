"""Device selection tests without importing JAX, CUDA, or the VQE program."""

import argparse
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


_PATH = Path(__file__).with_name("dpqc_backend.py")
_SPEC = importlib.util.spec_from_file_location("dpqc_backend", _PATH)
backend = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(backend)


class DPQCBackendTests(unittest.TestCase):
    def setUp(self):
        self.environ = mock.patch.dict(os.environ, {}, clear=True)
        self.environ.start()
        self.addCleanup(self.environ.stop)

    def _fake_jax(self, platform):
        kind = "NVIDIA GeForce RTX 3080" if platform == "gpu" else "cpu"
        device = SimpleNamespace(platform=platform, device_kind=kind, id=0)
        return SimpleNamespace(
            default_backend=mock.Mock(return_value=platform),
            local_devices=mock.Mock(return_value=[device]),
        ), device

    def test_default_argument_is_auto_and_environment_can_select_gpu(self):
        parser = argparse.ArgumentParser()
        backend.add_device_argument(parser)
        self.assertEqual(parser.parse_args([]).device, "auto")
        os.environ["DPQC_DEVICE"] = "gpu"
        parser = argparse.ArgumentParser()
        backend.add_device_argument(parser)
        self.assertEqual(parser.parse_args([]).device, "gpu")
        self.assertEqual(parser.parse_args(["--device", "cpu"]).device, "cpu")

    def test_invalid_choice_rejected_before_numerics(self):
        with self.assertRaisesRegex(ValueError, "choose auto, cpu, or gpu"):
            backend.configure_jax_backend("cuda")

    def test_stage_defaults_keep_training_auto_and_analysis_on_cpu(self):
        for stage, expected in (("vqe", "auto"), ("qfim", "cpu"), ("hessian", "cpu")):
            for selection in (None, "auto"):
                with self.subTest(stage=stage, selection=selection):
                    self.assertEqual(backend.resolve_stage_device(selection, stage), expected)
        self.assertEqual(dict(os.environ), {}, "Resolving a stage must not initialize or pin JAX")

    def test_explicit_stage_devices_override_environment(self):
        for inherited in ("auto", "cpu", "gpu"):
            os.environ["DPQC_DEVICE"] = inherited
            for stage in ("vqe", "qfim", "hessian"):
                for selection in ("cpu", "gpu"):
                    with self.subTest(inherited=inherited, stage=stage, selection=selection):
                        self.assertEqual(backend.resolve_stage_device(selection, stage), selection)
                expected_auto = "auto" if stage == "vqe" else "cpu"
                self.assertEqual(backend.resolve_stage_device("auto", stage), expected_auto)

    def test_omitted_stage_device_uses_environment_override(self):
        for selection in ("cpu", "gpu"):
            os.environ["DPQC_DEVICE"] = selection
            for stage in ("vqe", "qfim", "hessian"):
                with self.subTest(stage=stage, selection=selection):
                    self.assertEqual(backend.resolve_stage_device(None, stage), selection)

    def test_invalid_stage_or_stage_device_is_rejected_without_environment_changes(self):
        for device, stage in (("cuda", "qfim"), ("auto", "analysis"), ("auto", "missing")):
            with self.subTest(device=device, stage=stage), self.assertRaises(ValueError):
                backend.resolve_stage_device(device, stage)
        self.assertEqual(dict(os.environ), {})

    def test_cpu_replaces_inherited_gpu_restriction(self):
        os.environ.update(JAX_PLATFORMS="cuda", JAX_PLATFORM_NAME="gpu")
        self.assertEqual(backend.configure_jax_backend("cpu"), "cpu")
        self.assertEqual(os.environ["DPQC_DEVICE"], "cpu")
        self.assertEqual(os.environ["JAX_PLATFORMS"], "cpu")
        self.assertEqual(os.environ["JAX_PLATFORM_NAME"], "cpu")

    def test_gpu_requires_cuda_and_memory_grows_without_reserving_ten_gb(self):
        os.environ.update(JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu")
        backend.configure_jax_backend("gpu")
        self.assertEqual(os.environ["DPQC_DEVICE"], "gpu")
        self.assertEqual(os.environ["JAX_PLATFORMS"], "cuda")
        self.assertEqual(os.environ["JAX_PLATFORM_NAME"], "gpu")
        self.assertEqual(os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")

    def test_gpu_preserves_explicit_memory_setting(self):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        backend.configure_jax_backend("gpu")
        self.assertEqual(os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"], "true")

    def test_auto_uses_unrestricted_jax_default(self):
        self.assertEqual(backend.configure_jax_backend(), "auto")
        self.assertNotIn("JAX_PLATFORMS", os.environ)
        self.assertNotIn("JAX_PLATFORM_NAME", os.environ)

    def test_auto_preserves_external_platform_configuration(self):
        os.environ.update(JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu")
        backend.configure_jax_backend("auto")
        self.assertEqual(os.environ["JAX_PLATFORMS"], "cpu")
        self.assertEqual(os.environ["JAX_PLATFORM_NAME"], "cpu")

    def test_cpu_initialization_reports_device(self):
        backend.configure_jax_backend("cpu")
        fake_jax, device = self._fake_jax("cpu")
        output = io.StringIO()
        with mock.patch.dict(sys.modules, {"jax": fake_jax}), contextlib.redirect_stdout(output):
            self.assertIs(backend.initialize_jax_backend(), device)
        self.assertIn("JAX backend: cpu", output.getvalue())

    def test_gpu_initialization_reports_real_model(self):
        backend.configure_jax_backend("gpu")
        fake_jax, device = self._fake_jax("gpu")
        output = io.StringIO()
        with mock.patch.dict(sys.modules, {"jax": fake_jax}), contextlib.redirect_stdout(output):
            self.assertIs(backend.initialize_jax_backend(), device)
        self.assertIn("NVIDIA GeForce RTX 3080", output.getvalue())

    def test_explicit_gpu_never_silently_uses_initialized_cpu(self):
        backend.configure_jax_backend("gpu")
        fake_jax, _ = self._fake_jax("cpu")
        with mock.patch.dict(sys.modules, {"jax": fake_jax}):
            with self.assertRaisesRegex(RuntimeError, "fresh Python process"):
                backend.initialize_jax_backend()

    def test_explicit_cpu_rejects_initialized_gpu(self):
        backend.configure_jax_backend("cpu")
        fake_jax, _ = self._fake_jax("gpu")
        with mock.patch.dict(sys.modules, {"jax": fake_jax}):
            with self.assertRaisesRegex(RuntimeError, "active backend is 'gpu'"):
                backend.initialize_jax_backend()

    def test_unavailable_cuda_reports_actionable_error(self):
        backend.configure_jax_backend("gpu")
        fake_jax, _ = self._fake_jax("gpu")
        fake_jax.default_backend.side_effect = RuntimeError("no CUDA plugin")
        with mock.patch.dict(sys.modules, {"jax": fake_jax}):
            with self.assertRaisesRegex(RuntimeError, "WSL2 on Windows") as raised:
                backend.initialize_jax_backend()
        self.assertIn("no CUDA plugin", str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_auto_accepts_cpu_or_gpu_jax_default(self):
        backend.configure_jax_backend("auto")
        for platform in ("cpu", "gpu"):
            fake_jax, device = self._fake_jax(platform)
            with self.subTest(platform=platform):
                with mock.patch.dict(sys.modules, {"jax": fake_jax}), contextlib.redirect_stdout(io.StringIO()):
                    self.assertIs(backend.initialize_jax_backend(), device)

    def test_empty_device_list_fails_before_training(self):
        fake_jax, _ = self._fake_jax("cpu")
        fake_jax.local_devices.return_value = []
        with mock.patch.dict(sys.modules, {"jax": fake_jax}):
            with self.assertRaisesRegex(RuntimeError, "no consistent local devices"):
                backend.initialize_jax_backend()

    def test_batch_clamp_avoids_twenty_trials_when_only_ten_requested(self):
        self.assertEqual(backend.effective_vqe_batch_size(20, 10), 10)
        self.assertEqual(backend.effective_vqe_batch_size(5, 10), 5)
        self.assertEqual(backend.effective_vqe_batch_size(10, 10), 10)

    def test_batch_inputs_must_be_positive_integers(self):
        for pair in ((0, 10), (-1, 10), (10, 0), (10, -1), (True, 10), (10, False), (1.5, 10)):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                backend.effective_vqe_batch_size(*pair)

    def test_helper_import_and_help_do_not_import_jax(self):
        code = """
import argparse
import importlib.util
import sys
class RejectJAX:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('jax', 'numpy', 'optax', 'tensorcircuit'):
            raise AssertionError('unexpected numerical import: ' + fullname)
sys.meta_path.insert(0, RejectJAX())
spec = importlib.util.spec_from_file_location('dpqc_backend', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parser = argparse.ArgumentParser()
module.add_device_argument(parser)
assert parser.parse_args([]).device == 'auto'
module.configure_jax_backend('gpu')
print(parser.format_help())
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(_PATH)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--device", result.stdout)


if __name__ == "__main__":
    unittest.main()
