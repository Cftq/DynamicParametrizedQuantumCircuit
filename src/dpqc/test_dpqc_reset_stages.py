"""Lightweight regression tests for independent reset-DPQC stages.

Run with ``python -m unittest discover -s src/dpqc -p test_dpqc_reset_stages.py``.
Numerical stage modules are replaced by fakes; no quantum dependencies or
training jobs are needed. All archive fixtures use temporary directories.
"""

from contextlib import redirect_stderr
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

_LAUNCHER = importlib.import_module("DPQC_overparam_reset_compute")
_MODEL = importlib.import_module("dpqc_reset_model")
_VQE = importlib.import_module("DPQC_overparam_reset_vqe")
_QFIM = importlib.import_module("DPQC_overparam_reset_qfim")
_HESSIAN = importlib.import_module("DPQC_overparam_reset_hessian")
_VISUALIZE = importlib.import_module("DPQC_overparam_reset_visualize")


class ResetLauncherTests(unittest.TestCase):
    def setUp(self):
        config_patch = patch.object(
            _LAUNCHER, "_default_config_values", return_value=(0.25, 17)
        )
        config_patch.start()
        self.addCleanup(config_patch.stop)

    def assert_stage_commands(self, run, stages, *, h_param="0.1", batch="3"):
        self.assertEqual(run.call_count, len(stages))
        for invocation, stage in zip(run.call_args_list, stages):
            command = invocation.args[0]
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(
                Path(command[1]),
                _MODULE_DIR / f"DPQC_overparam_reset_{stage}.py",
            )
            expected_options = ["--h-param", h_param]
            if stage == "vqe":
                expected_options.extend(["--vqe-batch-size", batch])
            self.assertEqual(command[2:], expected_options)
            self.assertEqual(invocation.kwargs, {"check": False})

    def test_default_runs_analysis_without_training(self):
        with patch.object(
            _LAUNCHER.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
        ) as run:
            result = _LAUNCHER.main([])
        self.assertEqual(result, 0)
        self.assert_stage_commands(run, ("qfim", "hessian"), h_param="0.25", batch="17")

    def test_explicit_stage_selections_are_isolated(self):
        selections = {
            "analysis": ("qfim", "hessian"),
            "all": ("vqe", "qfim", "hessian"),
            "vqe": ("vqe",),
            "qfim": ("qfim",),
            "hessian": ("hessian",),
        }
        for selection, stages in selections.items():
            with self.subTest(stage=selection), patch.object(
                _LAUNCHER.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run:
                result = _LAUNCHER.main(
                    ["--stage", selection, "--h-param", "0.1", "--vqe-batch-size", "3"]
                )
                self.assertEqual(result, 0)
                self.assert_stage_commands(run, stages)

    def test_pipeline_stops_on_first_failure_and_preserves_exit_code(self):
        for selection, stages in (
            ("analysis", ("qfim", "hessian")),
            ("all", ("vqe", "qfim", "hessian")),
        ):
            for failed_index, failed_stage in enumerate(stages):
                status = 7 + failed_index
                completions = [
                    subprocess.CompletedProcess([], 0) for _ in range(failed_index)
                ] + [subprocess.CompletedProcess([], status)]
                with self.subTest(selection=selection, failure=failed_stage), patch.object(
                    _LAUNCHER.subprocess, "run", side_effect=completions
                ) as run:
                    result = _LAUNCHER.main(
                        ["--stage", selection, "--h-param", "0.1", "--vqe-batch-size", "3"]
                    )
                    self.assertEqual(result, status)
                    self.assert_stage_commands(run, stages[: failed_index + 1])

    def test_individual_stage_failure_is_propagated(self):
        for stage in ("vqe", "qfim", "hessian"):
            with self.subTest(stage=stage), patch.object(
                _LAUNCHER.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 23),
            ) as run:
                result = _LAUNCHER.main(
                    ["--stage", stage, "--h-param", "0.1", "--vqe-batch-size", "3"]
                )
                self.assertEqual(result, 23)
                self.assert_stage_commands(run, (stage,))


class ResetNumericalStageIsolationTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        original_cwd = Path.cwd()
        os.chdir(temp_dir.name)
        self.addCleanup(os.chdir, original_cwd)
        self.root = Path(temp_dir.name)
        self.save_dir = self.root / "figs" / "dpqc_reset" / "h_0.1"
        self.energy_archive = (
            self.save_dir / "numerical_results" / "energy" / "vqe_optimization_histories.npz"
        )
        self.metadata_path = self.save_dir / "reset_model_metadata.json"

    @staticmethod
    def make_stage(stage):
        module = ModuleType(f"fake_reset_{stage}")
        module.h_param = 0.1
        # Deliberately omit the opposite stage's entry point. Calling it is a
        # failure, even when training would otherwise be hidden inside a fake.
        setattr(module, f"run_{stage}", Mock())
        return module

    def run_fake_qfim(self):
        module = self.make_stage("qfim")
        with patch.object(
            _MODEL, "_load_base_stage_module", return_value=module
        ) as load, patch.object(_MODEL, "_install_reset_model") as install:
            _QFIM.run_qfim(h_param=0.1)
        self.assertEqual(load.call_args.args, ("qfim",))
        self.assertEqual(load.call_args.kwargs["h_param"], 0.1)
        install.assert_called_once_with(module)
        module.run_qfim.assert_called_once_with(include_optimization_path=False)
        return module

    def test_qfim_runs_without_training_archive_and_identifies_reset_model(self):
        self.assertFalse(self.energy_archive.exists())
        self.run_fake_qfim()
        self.assertFalse(self.energy_archive.exists())
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["model_id"], "dpqc_reset_fixed_rx_pi")
        self.assertEqual(metadata["h_param"], 0.1)
        self.assertEqual(metadata["total_parameter_formula"], "12 * L")

    def test_repeated_qfim_preserves_saved_training_and_metadata(self):
        self.energy_archive.parent.mkdir(parents=True)
        training_bytes = b"existing training results must not be regenerated"
        self.energy_archive.write_bytes(training_bytes)
        _MODEL._write_model_metadata(self.save_dir, 0.1)
        metadata_before = self.metadata_path.read_bytes()

        self.run_fake_qfim()
        self.run_fake_qfim()

        self.assertEqual(self.energy_archive.read_bytes(), training_bytes)
        self.assertEqual(self.metadata_path.read_bytes(), metadata_before)

    def test_incompatible_metadata_prevents_analysis_and_is_not_overwritten(self):
        self.save_dir.mkdir(parents=True)
        _MODEL._write_model_metadata(self.save_dir, 0.2)
        metadata_before = self.metadata_path.read_bytes()
        module = self.make_stage("qfim")
        with patch.object(
            _MODEL, "_load_base_stage_module", return_value=module
        ), patch.object(_MODEL, "_install_reset_model"), self.assertRaises(ValueError):
            _QFIM.run_qfim(h_param=0.1)
        module.run_qfim.assert_not_called()
        self.assertEqual(self.metadata_path.read_bytes(), metadata_before)

    def test_unidentified_training_archive_is_not_stamped_as_compatible(self):
        self.energy_archive.parent.mkdir(parents=True)
        self.energy_archive.write_bytes(b"unidentified training archive")
        module = self.make_stage("qfim")
        with patch.object(
            _MODEL, "_load_base_stage_module", return_value=module
        ), patch.object(_MODEL, "_install_reset_model"), self.assertRaises(FileNotFoundError):
            _QFIM.run_qfim(h_param=0.1)
        module.run_qfim.assert_not_called()
        self.assertFalse(self.metadata_path.exists())

    def test_vqe_entry_point_runs_only_training_and_persists_metadata(self):
        module = self.make_stage("vqe")
        with patch.object(
            _MODEL, "_load_base_stage_module", return_value=module
        ) as load, patch.object(_MODEL, "_install_reset_model") as install:
            _VQE.run_vqe(h_param=0.1, vqe_batch_size=3)
        load.assert_called_once_with("vqe", h_param=0.1, vqe_batch_size=3)
        install.assert_called_once_with(module)
        module.run_vqe.assert_called_once_with()
        self.assertTrue(self.metadata_path.is_file())


class ResetHessianStageTests(unittest.TestCase):
    def test_hessian_uses_reset_family_and_forwards_options_and_exit_status(self):
        options = [
            "--h-param", "0.1", "--layers", "1,4", "--num-samples", "2",
            "--seed-base", "17", "--hvp-chunk-size", "3",
            "--output-dir", "an output directory",
        ]
        with patch.object(
            _HESSIAN.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 19),
        ) as run:
            result = _HESSIAN.main(options)
        self.assertEqual(result, 19)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[:2], [sys.executable, str(_MODULE_DIR / "DPQC_overparam_hessian.py")])
        forwarded = command[2:]
        family_index = forwarded.index("--output-family")
        self.assertEqual(forwarded[family_index + 1], "dpqc_reset")
        del forwarded[family_index:family_index + 2]
        self.assertEqual(len(forwarded), len(options))
        self.assertEqual(
            dict(zip(forwarded[::2], forwarded[1::2])),
            dict(zip(options[::2], options[1::2])),
        )
        self.assertEqual(run.call_args.kwargs, {"check": False})

    def test_hessian_rejects_overriding_reset_output_family(self):
        for options in (["--output-family", "dpqc"], ["--output-family=dpqc"]):
            with self.subTest(options=options), patch.object(
                _HESSIAN.subprocess, "run"
            ) as run, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                _HESSIAN.main(options)
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()


class ResetSavedVisualizationTests(unittest.TestCase):
    def test_default_visualization_reuses_saved_hessians(self):
        command = _VISUALIZE._build_visualizer_command(0.1)
        self.assertEqual(
            command[:2],
            (sys.executable, str(_MODULE_DIR / "DPQC_overparam_visualize.py")),
        )
        self.assertIn("--with-hessian", command)
        self.assertIn("--reuse-hessian-results", command)
        self.assertIn("--skip-optimization-path-qfim", command)
        self.assertEqual(command[command.index("--output-family") + 1], "dpqc_reset")

    def test_plain_visualizer_launches_only_plotter_and_propagates_failure(self):
        with patch.object(
            _VISUALIZE.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 11),
        ) as run:
            result = _VISUALIZE.main(["--h-param", "0.1"])
        self.assertEqual(result, 11)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(Path(command[1]), _MODULE_DIR / "DPQC_overparam_visualize.py")
        self.assertIn("--with-hessian", command)
        self.assertIn("--reuse-hessian-results", command)
        self.assertFalse(run.call_args.kwargs["shell"])


class ResetLightweightEntryPointTests(unittest.TestCase):
    def test_import_and_help_do_not_load_numerical_dependencies_or_stages(self):
        # A clean subprocess catches eager imports that would be hidden by
        # another test having already cached a numerical module in sys.modules.
        probe = r'''
import importlib.abc
from pathlib import Path
import runpy
import sys

blocked = {
    "jax", "jaxlib", "numpy", "scipy", "matplotlib", "qiskit", "optax",
    "DPQC_overparam_vqe", "DPQC_overparam_qfim", "DPQC_overparam_hessian",
}

class RejectNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError("Unexpected numerical import: " + fullname)

sys.meta_path.insert(0, RejectNumericalImports())
module_dir = Path(sys.argv[1])
for name in (
    "dpqc_reset_model", "DPQC_overparam_reset_compute",
    "DPQC_overparam_reset_vqe", "DPQC_overparam_reset_qfim",
    "DPQC_overparam_reset_hessian", "DPQC_overparam_reset_visualize",
):
    script = module_dir / (name + ".py")
    runpy.run_path(str(script), run_name="smoke_" + name)
    if name != "dpqc_reset_model":
        sys.argv = [str(script), "--help"]
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            if exc.code != 0:
                raise
        else:
            raise AssertionError(name + " --help did not exit")
print("RESET_LIGHTWEIGHT_OK")
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", probe, str(_MODULE_DIR)],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("RESET_LIGHTWEIGHT_OK", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
