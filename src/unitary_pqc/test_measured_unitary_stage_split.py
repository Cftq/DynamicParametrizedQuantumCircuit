"""Regression tests for independently runnable outcome-1 numerical stages.

Run with ``python -m unittest discover -s src/unitary_pqc -p
test_measured_unitary_stage_split.py``. Subprocess launches and numerical
workers are isolated so these tests never train a quantum circuit.
"""

from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


_MODULE_DIR = Path(__file__).resolve().parent
_PREFIX = "unitary_pqc_measured_1_overparam_"
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))


class MeasuredUnitaryLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = importlib.import_module(_PREFIX + "compute")

    def invoke(self, *extra, returncodes=None):
        arguments = [
            "--h-param", "0.3", "--vqe-batch-size", "7",
            "--analysis-batch-size", "2", *extra,
        ]
        completions = (
            None if returncodes is None else
            [subprocess.CompletedProcess([], code) for code in returncodes]
        )
        with patch.object(
            self.launcher.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0),
            side_effect=completions,
        ) as run, redirect_stdout(io.StringIO()):
            result = self.launcher.main(arguments)
        return result, run

    def assert_commands(self, run, stages, *, random_only=False):
        self.assertEqual(run.call_count, len(stages))
        for invocation, stage in zip(run.call_args_list, stages):
            command = invocation.args[0]
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(
                Path(command[1]), _MODULE_DIR / f"{_PREFIX}{stage}.py"
            )
            options = command[2:]
            expected = ["--h-param", "0.3"]
            expected.extend(
                ["--vqe-batch-size", "7"] if stage == "vqe" else
                ["--analysis-batch-size", "2"]
            )
            if stage == "qfim" and random_only:
                expected.append("--random-only")
            self.assertEqual(options, expected)
            self.assertEqual(invocation.kwargs, {"check": False})

    def test_default_launches_only_qfim_and_hessian(self):
        result, run = self.invoke()
        self.assertEqual(result, 0)
        self.assert_commands(run, ("qfim", "hessian"))

    def test_selected_stages_and_explicit_full_pipeline(self):
        selections = {
            "analysis": ("qfim", "hessian"),
            "all": ("vqe", "qfim", "hessian"),
            "vqe": ("vqe",), "qfim": ("qfim",), "hessian": ("hessian",),
        }
        for stage, workers in selections.items():
            with self.subTest(stage=stage):
                result, run = self.invoke("--stage", stage)
                self.assertEqual(result, 0)
                self.assert_commands(run, workers)

    def test_random_only_is_forwarded_only_to_qfim(self):
        result, run = self.invoke("--stage", "analysis", "--random-only")
        self.assertEqual(result, 0)
        self.assert_commands(run, ("qfim", "hessian"), random_only=True)

    def test_failure_stops_pipeline_without_starting_later_workers(self):
        for selection, stages in (
            ("analysis", ("qfim", "hessian")),
            ("all", ("vqe", "qfim", "hessian")),
        ):
            for index, failed_stage in enumerate(stages):
                with self.subTest(selection=selection, failure=failed_stage):
                    returncodes = [0] * index + [17 + index]
                    result, run = self.invoke(
                        "--stage", selection, returncodes=returncodes,
                    )
                    self.assertEqual(result, 17 + index)
                    self.assert_commands(run, stages[:index + 1])

    def test_individual_worker_failure_is_preserved(self):
        for stage in ("vqe", "qfim", "hessian"):
            with self.subTest(stage=stage):
                result, run = self.invoke("--stage", stage, returncodes=[23])
                self.assertEqual(result, 23)
                self.assert_commands(run, (stage,))

    def test_invalid_arguments_never_start_a_worker(self):
        for arguments in (
            ["--h-param", "nan"], ["--h-param", "inf"],
            ["--vqe-batch-size", "0"], ["--analysis-batch-size", "-1"],
            ["--stage", "training"],
        ):
            with self.subTest(arguments=arguments), patch.object(
                self.launcher.subprocess, "run"
            ) as run, redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ) as raised:
                self.launcher.main(arguments)
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()


class MeasuredUnitaryAnalysisIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qfim = importlib.import_module(_PREFIX + "qfim")
        cls.hessian = importlib.import_module(_PREFIX + "hessian")
        cls.vqe = importlib.import_module(_PREFIX + "vqe")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.energy_archive = self.root / "energy" / "vqe_optimization_results.npz"
        self.common = ModuleType(_PREFIX + "common")
        self.common.os = os
        self.common.configure_unitary_pqc_overparam = Mock()
        self.common._resolve_analysis_batch_size = Mock(return_value=2)
        self.common.load_unitary_vqe_samples = Mock(
            return_value=str(self.energy_archive)
        )
        self.common.collect_unitary_pqc_result = Mock(return_value={})
        self.common.energy_results_dir = str(self.energy_archive.parent)
        self.common.hessian_results_dir = str(self.root / "hessian")
        common_patch = patch.dict(sys.modules, {_PREFIX + "common": self.common})
        common_patch.start()
        self.addCleanup(common_patch.stop)

    def run_qfim(self, *, include_optimization_path):
        with patch.object(self.qfim, "run_random_qfim_analysis") as random, patch.object(
            self.qfim, "run_optimization_path_qfim_analysis"
        ) as path, redirect_stdout(io.StringIO()):
            result = self.qfim.run_unitary_pqc_qfim_stage(
                h_param=0.3, analysis_batch_size=2,
                include_optimization_path=include_optimization_path,
            )
        return result, random, path

    def test_random_qfim_needs_no_training_archive(self):
        self.common.load_unitary_vqe_samples.side_effect = AssertionError(
            "Random QFIM must not load a training archive"
        )
        result, random, path = self.run_qfim(include_optimization_path=False)
        self.common.configure_unitary_pqc_overparam.assert_called_once_with(h_value=0.3)
        random.assert_called_once_with(make_plots=False, analysis_batch_size=2)
        path.assert_not_called()
        self.common.load_unitary_vqe_samples.assert_not_called()
        self.assertEqual(result["analysis_batch_size"], 2)
        self.assertFalse(self.energy_archive.exists())

    def test_qfim_cli_random_only_skips_saved_path(self):
        self.common.collect_unitary_pqc_result.return_value = {
            "qfim_results_dir": str(self.root / "qfim")
        }
        self.common.load_unitary_vqe_samples.side_effect = AssertionError(
            "--random-only must not load or generate training results"
        )
        with patch.object(self.qfim, "run_random_qfim_analysis") as random, patch.object(
            self.qfim, "run_optimization_path_qfim_analysis"
        ) as path, redirect_stdout(io.StringIO()):
            status = self.qfim.main([
                "--h-param", "0.3", "--analysis-batch-size", "2", "--random-only",
            ])
        self.assertEqual(status, 0)
        random.assert_called_once_with(make_plots=False, analysis_batch_size=2)
        path.assert_not_called()
        self.common.load_unitary_vqe_samples.assert_not_called()
        self.assertFalse(self.energy_archive.exists())

    def test_saved_path_qfim_reuses_training_archive_without_writing_it(self):
        self.energy_archive.parent.mkdir()
        saved_training = b"previously trained parameter and energy histories"
        self.energy_archive.write_bytes(saved_training)
        for _ in range(2):
            result, random, path = self.run_qfim(include_optimization_path=True)
            random.assert_called_once_with(make_plots=False, analysis_batch_size=2)
            path.assert_called_once_with(analysis_batch_size=2)
            self.assertEqual(result["vqe_input_path"], str(self.energy_archive))
            self.assertEqual(self.energy_archive.read_bytes(), saved_training)
        self.assertEqual(self.common.load_unitary_vqe_samples.call_count, 2)

    def test_missing_saved_vqe_fails_without_starting_any_computation(self):
        self.common.load_unitary_vqe_samples.side_effect = FileNotFoundError(
            str(self.energy_archive)
        )
        with patch.object(self.qfim, "run_random_qfim_analysis") as random, patch.object(
            self.qfim, "run_optimization_path_qfim_analysis"
        ) as path, self.assertRaises(FileNotFoundError):
            self.qfim.run_unitary_pqc_qfim_stage(
                h_param=0.3, analysis_batch_size=2,
            )
        random.assert_not_called()
        path.assert_not_called()
        self.assertFalse(self.energy_archive.exists())

    def test_hessian_stage_needs_no_training_or_qfim(self):
        self.common.load_unitary_vqe_samples.side_effect = AssertionError(
            "Hessian must not load a training archive"
        )
        with patch.object(self.hessian, "run_random_hessian_analysis") as hessian, patch.object(
            self.qfim, "run_random_qfim_analysis",
            side_effect=AssertionError("Hessian must not run QFIM"),
        ), patch.object(
            self.vqe, "run_vqe_optimization",
            side_effect=AssertionError("Hessian must not train"),
        ), redirect_stdout(io.StringIO()):
            self.hessian.run_unitary_pqc_hessian_stage(
                h_param=0.3, analysis_batch_size=2,
            )
        self.common.configure_unitary_pqc_overparam.assert_called_once_with(h_value=0.3)
        hessian.assert_called_once_with(analysis_batch_size=2)
        self.common.load_unitary_vqe_samples.assert_not_called()
        self.assertFalse(self.energy_archive.exists())

    def test_vqe_stage_runs_training_without_running_analysis(self):
        with patch.object(self.vqe, "run_vqe_optimization") as train, patch.object(
            self.qfim, "run_random_qfim_analysis",
            side_effect=AssertionError("VQE-only must not run QFIM"),
        ), patch.object(
            self.hessian, "run_random_hessian_analysis",
            side_effect=AssertionError("VQE-only must not run Hessian"),
        ):
            archive_path = self.vqe.run_unitary_pqc_vqe_stage(
                h_param=0.3, vqe_batch_size=7,
            )
        self.common.configure_unitary_pqc_overparam.assert_called_once_with(h_value=0.3)
        train.assert_called_once_with(vqe_batch_size=7)
        self.assertEqual(Path(archive_path), self.energy_archive)
        self.common.load_unitary_vqe_samples.assert_not_called()


class MeasuredUnitaryLightweightEntryPointTests(unittest.TestCase):
    def test_import_and_help_do_not_load_numerical_dependencies_or_create_files(self):
        # Run in a fresh interpreter so earlier cached imports cannot hide an
        # accidental dependency on JAX or the VQE-only optimizer.
        probe = r'''
import importlib.abc
from pathlib import Path
import runpy
import sys

blocked = {
    "jax", "jaxlib", "numpy", "scipy", "matplotlib", "qiskit", "optax",
    "tensorcircuit", "unitary_pqc_measured_1_overparam_common",
}

class RejectNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked or fullname.rsplit(".", 1)[-1] in blocked:
            raise AssertionError("Unexpected numerical import: " + fullname)

sys.meta_path.insert(0, RejectNumericalImports())
module_dir = Path(sys.argv[1])
for suffix in ("compute", "vqe", "qfim", "hessian"):
    name = "unitary_pqc_measured_1_overparam_" + suffix
    script = module_dir / (name + ".py")
    runpy.run_path(str(script), run_name="smoke_" + name)
    sys.argv = [str(script), "--help"]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code != 0:
            raise
    else:
        raise AssertionError(name + " --help did not exit")

# Exercise the python -m form too: it follows package-relative imports.
sys.path.insert(0, str(module_dir.parent.parent))
for suffix in ("compute", "vqe", "qfim", "hessian"):
    name = "src.unitary_pqc.unitary_pqc_measured_1_overparam_" + suffix
    sys.argv = [name, "--help"]
    try:
        runpy.run_module(name, run_name="__main__")
    except SystemExit as exc:
        if exc.code != 0:
            raise
    else:
        raise AssertionError(name + " --help did not exit")
print("MEASURED_UNITARY_LIGHTWEIGHT_OK")
'''
        with tempfile.TemporaryDirectory() as directory:
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                [sys.executable, "-c", probe, str(_MODULE_DIR)],
                cwd=directory, env=environment, capture_output=True,
                text=True, encoding="utf-8", check=False, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("MEASURED_UNITARY_LIGHTWEIGHT_OK", result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])


class MeasuredUnitaryPackageImportTests(unittest.TestCase):
    def test_package_facade_and_stages_share_the_same_common_state(self):
        # Package imports must not silently create a second top-level common
        # module: that would split the Hamiltonian, sampled parameters, and
        # saved-result state between the facade, plotting, and worker stages.
        project_root = _MODULE_DIR.parent.parent
        with patch.object(sys, "path", [str(project_root), *sys.path]):
            package_name = "src.unitary_pqc"
            package = importlib.import_module(package_name)
            modules = {
                stage: importlib.import_module(f"{package_name}.{_PREFIX}{stage}")
                for stage in ("compute", "vqe", "qfim", "hessian")
            }

        state = {}
        common_name = _PREFIX + "common"
        common = ModuleType(f"{package_name}.{common_name}")
        common.shared_result_state = state
        common.configure_unitary_pqc_overparam = Mock()
        common._resolve_analysis_batch_size = Mock(return_value=2)
        common.collect_unitary_pqc_result = Mock(return_value={"state": state})
        common.load_unitary_vqe_samples = Mock(
            side_effect=AssertionError("These package analysis calls need no VQE archive")
        )
        common.energy_results_dir = "package-only-energy-results"
        common.os = os
        facade = modules["compute"]

        def record_qfim(**kwargs):
            state["qfim"] = kwargs["analysis_batch_size"]

        def record_hessian(**kwargs):
            state["hessian"] = kwargs["analysis_batch_size"]
            return "package-only-hessian-results.npz"

        def record_training(**kwargs):
            # This substitutes for the real training loop; no optimizer or
            # circuit execution is involved in the package-state regression.
            state["vqe"] = kwargs["vqe_batch_size"]

        with patch.dict(sys.modules, {
            f"{package_name}.{common_name}": common,
            common_name: None,
        }), patch.object(package, common_name, common, create=True), patch.object(
            modules["qfim"], "run_random_qfim_analysis", side_effect=record_qfim,
        ), patch.object(
            modules["qfim"], "run_optimization_path_qfim_analysis",
            side_effect=AssertionError("Random-only QFIM must skip the saved path"),
        ), patch.object(
            modules["hessian"], "run_random_hessian_analysis", side_effect=record_hessian,
        ), patch.object(
            modules["vqe"], "run_vqe_optimization", side_effect=record_training,
        ), redirect_stdout(io.StringIO()):
            self.assertIs(facade.shared_result_state, state)
            for stage in ("qfim", "hessian", "vqe"):
                function_name = f"run_unitary_pqc_{stage}_stage"
                self.assertIs(
                    getattr(facade, function_name),
                    getattr(modules[stage], function_name),
                )
            qfim_result = facade.run_unitary_pqc_qfim_stage(
                h_param=0.3, analysis_batch_size=2, include_optimization_path=False,
            )
            hessian_result = facade.run_unitary_pqc_hessian_stage(
                h_param=0.3, analysis_batch_size=2,
            )
            vqe_archive = facade.run_unitary_pqc_vqe_stage(
                h_param=0.3, vqe_batch_size=7,
            )
            self.assertIs(qfim_result["state"], state)
            self.assertIs(hessian_result["state"], state)
            self.assertIs(facade.shared_result_state, state)
            self.assertIsNone(sys.modules[common_name])

        self.assertEqual(state, {"qfim": 2, "hessian": 2, "vqe": 7})
        self.assertEqual(
            Path(vqe_archive), Path(common.energy_results_dir) / "vqe_optimization_results.npz",
        )
        self.assertEqual(common.configure_unitary_pqc_overparam.call_count, 3)
        common.load_unitary_vqe_samples.assert_not_called()


if __name__ == "__main__":
    unittest.main()
