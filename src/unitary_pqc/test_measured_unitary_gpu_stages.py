"""Verify outcome-1 device routing without loading JAX or running circuits."""

from contextlib import redirect_stdout
import importlib
import io
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


_MODULE_DIR = Path(__file__).resolve().parent
_PREFIX = "unitary_pqc_measured_1_overparam_"
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
_CLI = importlib.import_module(_PREFIX + "cli")
_BACKEND = importlib.import_module("dpqc_backend")
_WSL = importlib.import_module("dpqc_wsl")


def _selected_option(command, option):
    return command[command.index(option) + 1]


class MeasuredUnitaryDeviceRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = importlib.import_module(_PREFIX + "compute")
        cls.stages = {
            stage: importlib.import_module(_PREFIX + stage)
            for stage in ("vqe", "qfim", "hessian")
        }

    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"DPQC_DEVICE": "auto"}))

    def test_launcher_preserves_auto_for_training_and_cpu_for_analyses(self):
        with patch.object(
            _WSL, "maybe_relaunch_in_wsl", return_value=None,
        ) as route, patch.object(
            self.launcher.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            self.assertEqual(self.launcher.main(["--stage", "all"]), 0)
        self.assertEqual(
            [_selected_option(call.args[0], "--device") for call in run.call_args_list],
            ["auto", "cpu", "cpu"],
        )
        self.assertEqual(route.call_args.args[2], "auto")
        self.assertEqual(route.call_args.kwargs, {"preserve_auto": True})

    def test_launcher_honors_environment_and_explicit_device_overrides(self):
        for inherited in ("auto", "cpu", "gpu"):
            for explicit in (None, "auto", "cpu", "gpu"):
                with self.subTest(inherited=inherited, explicit=explicit):
                    arguments = ["--stage", "all"]
                    if explicit is not None:
                        arguments.extend(("--device", explicit))
                    selected = inherited if explicit is None else explicit
                    expected = (
                        ["auto", "cpu", "cpu"] if selected == "auto" else
                        [selected] * 3
                    )
                    with patch.dict(os.environ, {"DPQC_DEVICE": inherited}), patch.object(
                        _WSL, "maybe_relaunch_in_wsl", return_value=None,
                    ), patch.object(
                        self.launcher.subprocess, "run",
                        return_value=subprocess.CompletedProcess([], 0),
                    ) as run:
                        self.assertEqual(self.launcher.main(arguments), 0)
                    self.assertEqual(
                        [_selected_option(call.args[0], "--device")
                         for call in run.call_args_list],
                        expected,
                    )

    def test_launcher_wsl_result_prevents_native_worker_launch(self):
        with patch.object(
            _WSL, "maybe_relaunch_in_wsl", return_value=29,
        ) as route, patch.object(self.launcher.subprocess, "run") as run:
            self.assertEqual(self.launcher.main(["--stage", "all"]), 29)
        run.assert_not_called()
        self.assertEqual(route.call_args.kwargs, {"preserve_auto": True})

    def test_stage_default_and_override_are_resolved_before_wsl_and_work(self):
        for inherited in ("auto", "cpu", "gpu"):
            for explicit in (None, "auto", "cpu", "gpu"):
                for stage, module in self.stages.items():
                    with self.subTest(inherited=inherited, explicit=explicit, stage=stage):
                        arguments = ["--h-param", "0.3"]
                        if explicit is not None:
                            arguments.extend(("--device", explicit))
                        selected = inherited if explicit is None else explicit
                        expected = "cpu" if selected == "auto" and stage != "vqe" else selected
                        result = "vqe.npz" if stage == "vqe" else {
                            f"{stage}_results_dir": stage,
                        }
                        events = []

                        def route_call(*args, **kwargs):
                            events.append(("route", args[2]))

                        def worker_call(**kwargs):
                            events.append(("work", kwargs["device"]))
                            return result

                        with patch.dict(os.environ, {"DPQC_DEVICE": inherited}), patch.object(
                            _WSL, "maybe_relaunch_in_wsl", side_effect=route_call,
                        ), patch.object(
                            module, f"run_unitary_pqc_{stage}_stage", side_effect=worker_call,
                        ), redirect_stdout(io.StringIO()):
                            self.assertEqual(module.main(arguments), 0)
                        self.assertEqual(events, [("route", expected), ("work", expected)])

    def test_stage_wsl_exit_never_initializes_or_runs_numerical_worker(self):
        for stage, module in self.stages.items():
            with self.subTest(stage=stage), patch.object(
                _WSL, "maybe_relaunch_in_wsl", return_value=21,
            ), patch.object(
                module, f"run_unitary_pqc_{stage}_stage",
            ) as work, patch.object(_BACKEND, "configure_jax_backend") as configure, patch.object(
                _BACKEND, "initialize_jax_backend",
            ) as initialize:
                self.assertEqual(module.main(["--device", "gpu"]), 21)
            work.assert_not_called()
            configure.assert_not_called()
            initialize.assert_not_called()

    def test_python_stage_api_selects_device_before_any_numerical_work(self):
        for stage, module in self.stages.items():
            worker_name = {
                "vqe": "run_vqe_optimization", "qfim": "run_random_qfim_analysis",
                "hessian": "run_random_hessian_analysis",
            }[stage]
            for selected in (None, "gpu", "cpu"):
                with self.subTest(stage=stage, selected=selected), patch.object(
                    _CLI, "load_stage_common", side_effect=RuntimeError("backend preflight"),
                ) as load_common, patch.object(module, worker_name) as worker:
                    with self.assertRaisesRegex(RuntimeError, "backend preflight"):
                        getattr(module, f"run_unitary_pqc_{stage}_stage")(
                            h_param=0.3, device=selected,
                        )
                load_common.assert_called_once_with(stage, selected)
                worker.assert_not_called()


class MeasuredUnitaryBackendInitializationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"DPQC_DEVICE": "auto"}))

    def test_first_common_import_observes_selected_backend_before_loading_arrays(self):
        common_name = _PREFIX + "common"
        for stage, device, expected in (
            ("vqe", "gpu", "gpu"), ("qfim", "auto", "cpu"),
            ("hessian", "auto", "cpu"), ("qfim", "gpu", "gpu"),
        ):
            with self.subTest(stage=stage, device=device):
                common = ModuleType(common_name)
                events = []
                original_import = importlib.import_module

                def import_common(name, package=None):
                    if name == common_name:
                        events.append(("import", os.environ["DPQC_DEVICE"]))
                        self.assertEqual(os.environ["JAX_PLATFORM_NAME"], expected)
                        self.assertEqual(os.environ["JAX_PLATFORMS"], "cuda" if expected == "gpu" else "cpu")
                        return common
                    return original_import(name, package)

                with patch.dict(sys.modules, {}), patch.object(
                    _CLI.importlib, "import_module", side_effect=import_common,
                ), patch.object(_BACKEND, "initialize_jax_backend") as initialize:
                    sys.modules.pop(common_name, None)
                    self.assertIs(_CLI.load_stage_common(stage, device), common)
                self.assertEqual(events, [("import", expected)])
                initialize.assert_not_called()

    def test_cached_common_rejects_switching_an_initialized_backend(self):
        common_name = _PREFIX + "common"
        for active, requested in (("cpu", "gpu"), ("gpu", "cpu")):
            with self.subTest(active=active, requested=requested):
                fake_jax = ModuleType("jax")
                fake_jax.default_backend = Mock(return_value=active)
                fake_jax.local_devices = Mock(return_value=[SimpleNamespace(
                    platform=active, device_kind=active, id=0,
                )])
                with patch.dict(sys.modules, {
                    "jax": fake_jax, common_name: ModuleType(common_name),
                }), self.assertRaisesRegex(RuntimeError, "fresh Python process"):
                    _CLI.load_stage_common("vqe", requested)
                fake_jax.default_backend.assert_called_once_with()

    def test_common_initializes_backend_before_any_array_or_circuit_helpers(self):
        class StopBeforeArrays(Exception):
            pass

        matplotlib = ModuleType("matplotlib")
        matplotlib.pyplot = ModuleType("matplotlib.pyplot")
        matplotlib.ticker = ModuleType("matplotlib.ticker")
        patches = ModuleType("matplotlib.patches")
        patches.Patch = Mock()
        tqdm = ModuleType("tqdm.auto")
        tqdm.tqdm = Mock()
        modules = {
            "jax": ModuleType("jax"), "numpy": ModuleType("numpy"),
            "matplotlib": matplotlib, "matplotlib.pyplot": matplotlib.pyplot,
            "matplotlib.ticker": matplotlib.ticker, "matplotlib.patches": patches,
            "tqdm.auto": tqdm,
            "hamiltonian": None, "qfim": None,
        }

        def stop_after_initialization():
            self.assertEqual(os.environ["DPQC_DEVICE"], "gpu")
            self.assertEqual(os.environ["JAX_PLATFORM_NAME"], "gpu")
            raise StopBeforeArrays

        _BACKEND.configure_jax_backend("gpu")
        with patch.dict(sys.modules, modules), patch.object(
            _BACKEND, "initialize_jax_backend", side_effect=stop_after_initialization,
        ) as initialize, self.assertRaises(StopBeforeArrays):
            runpy.run_path(str(_MODULE_DIR / f"{_PREFIX}common.py"), run_name="bootstrap_probe")
        initialize.assert_called_once_with()


class MeasuredUnitaryLegacyPipelineTests(unittest.TestCase):
    def setUp(self):
        self.launcher = importlib.import_module(_PREFIX + "compute")
        self.events = []
        self.modules = {}
        for stage in ("vqe", "qfim", "hessian"):
            module = ModuleType(_PREFIX + stage)

            def run_stage(*, _stage=stage, **kwargs):
                self.events.append((_stage, kwargs["device"]))
                return {"analysis_batch_size": kwargs.get("analysis_batch_size")}

            setattr(module, f"run_unitary_pqc_{stage}_stage", Mock(side_effect=run_stage))
            self.modules[stage] = module
        common = ModuleType(_PREFIX + "common")
        common.collect_unitary_pqc_result = Mock(return_value={"complete": True})
        self.modules["common"] = common
        self.modules["cli"] = _CLI
        self.enterContext(patch.object(
            self.launcher, "_stage_module", side_effect=self.modules.__getitem__,
        ))

    def test_legacy_in_process_pipeline_preflights_cpu_before_training(self):
        def preflight(stage, device):
            self.events.append(("preflight", stage, device))
            return self.modules["common"]

        with patch.dict(os.environ, {"DPQC_DEVICE": "gpu"}), patch.object(
            _CLI, "load_stage_common", side_effect=preflight,
        ):
            result = self.launcher.run_unitary_pqc_overparam(
                h_param=0.3, vqe_batch_size=7, analysis_batch_size=2,
            )
        self.assertEqual(self.events, [
            ("preflight", "vqe", "cpu"), ("vqe", "cpu"),
            ("qfim", "cpu"), ("hessian", "cpu"),
        ])
        self.assertEqual(result, {"complete": True, "analysis_batch_size": 2})

    def test_legacy_backend_conflict_fails_before_any_training_or_analysis(self):
        with patch.object(
            _CLI, "load_stage_common", side_effect=RuntimeError("cached GPU backend"),
        ), self.assertRaisesRegex(RuntimeError, "cached GPU backend"):
            self.launcher.run_unitary_pqc_overparam(h_param=0.3)
        for stage in ("vqe", "qfim", "hessian"):
            getattr(self.modules[stage], f"run_unitary_pqc_{stage}_stage").assert_not_called()
        self.modules["common"].collect_unitary_pqc_result.assert_not_called()


if __name__ == "__main__":
    unittest.main()
