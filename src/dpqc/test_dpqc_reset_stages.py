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
        environment_patch = patch.dict(os.environ, {"DPQC_DEVICE": "auto"})
        environment_patch.start()
        self.addCleanup(environment_patch.stop)
        config_patch = patch.object(
            _LAUNCHER, "_default_config_values", return_value=(0.25, 17)
        )
        config_patch.start()
        self.addCleanup(config_patch.stop)
        route_patch = patch.object(
            _LAUNCHER.dpqc_wsl, "maybe_relaunch_in_wsl", return_value=None
        )
        route_patch.start()
        self.addCleanup(route_patch.stop)

    def assert_stage_commands(
        self, run, stages, *, h_param="0.1", batch="3", device="auto"
    ):
        self.assertEqual(run.call_count, len(stages))
        for invocation, stage in zip(run.call_args_list, stages):
            command = invocation.args[0]
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(
                Path(command[1]),
                _MODULE_DIR / f"DPQC_overparam_reset_{stage}.py",
            )
            stage_device = "cpu" if device == "auto" and stage != "vqe" else device
            expected_options = ["--h-param", h_param, "--device", stage_device]
            if stage == "vqe":
                expected_options.extend(["--vqe-batch-size", batch])
            self.assertEqual(len(command[2:]), len(expected_options))
            self.assertEqual(
                dict(zip(command[2::2], command[3::2])),
                dict(zip(expected_options[::2], expected_options[1::2])),
            )
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

    def test_gpu_selection_reaches_all_children_without_changing_stage_order(self):
        with patch.object(
            _LAUNCHER.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            result = _LAUNCHER.main([
                "--stage", "all", "--h-param", "0.1", "--vqe-batch-size", "3",
                "--device", "gpu",
            ])
        self.assertEqual(result, 0)
        self.assert_stage_commands(run, ("vqe", "qfim", "hessian"), device="gpu")

    def test_gpu_analysis_default_does_not_add_training(self):
        with patch.object(
            _LAUNCHER.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            result = _LAUNCHER.main(["--h-param", "0.1", "--device", "gpu"])
        self.assertEqual(result, 0)
        self.assert_stage_commands(run, ("qfim", "hessian"), device="gpu")

    def test_cpu_selection_is_forwarded_to_selected_stage(self):
        with patch.object(
            _LAUNCHER.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            result = _LAUNCHER.main([
                "--stage", "qfim", "--h-param", "0.1", "--device", "cpu",
            ])
        self.assertEqual(result, 0)
        self.assert_stage_commands(run, ("qfim",), device="cpu")

    def test_environment_device_and_explicit_auto_override_reach_separate_stages(self):
        for inherited in ("cpu", "gpu"):
            for arguments, expected in (([], inherited), (["--device", "auto"], "auto")):
                with self.subTest(inherited=inherited, arguments=arguments), patch.dict(
                    os.environ, {"DPQC_DEVICE": inherited}
                ), patch.object(
                    _LAUNCHER.subprocess, "run",
                    return_value=subprocess.CompletedProcess([], 0),
                ) as run:
                    result = _LAUNCHER.main([
                        "--stage", "all", "--h-param", "0.1", "--vqe-batch-size", "3",
                        *arguments,
                    ])
                    self.assertEqual(result, 0)
                    self.assert_stage_commands(run, ("vqe", "qfim", "hessian"), device=expected)

    def test_default_wsl_route_preserves_auto_before_per_stage_dispatch(self):
        arguments = ["--stage", "all"]
        with patch.object(_LAUNCHER.subprocess, "run") as run, patch.object(
            _LAUNCHER.dpqc_wsl, "maybe_relaunch_in_wsl", return_value=0,
        ) as route:
            self.assertEqual(_LAUNCHER.main(arguments), 0)
        self.assertEqual(route.call_args.args[1:], (arguments, "auto"))
        self.assertEqual(route.call_args.kwargs, {"preserve_auto": True})
        run.assert_not_called()

    def test_wsl_route_preserves_status_without_local_launch(self):
        arguments = ["--stage", "qfim", "--h-param", "0.1", "--device", "gpu"]
        with patch.object(_LAUNCHER.subprocess, "run") as run, patch.object(
            _LAUNCHER.dpqc_wsl, "maybe_relaunch_in_wsl", return_value=17
        ) as route:
            result = _LAUNCHER.main(arguments)
        self.assertEqual(result, 17)
        route.assert_called_once()
        self.assertEqual(
            Path(route.call_args.args[0]), _MODULE_DIR / "DPQC_overparam_reset_compute.py"
        )
        self.assertEqual(route.call_args.args[1:], (arguments, "gpu"))
        self.assertEqual(route.call_args.kwargs, {"preserve_auto": True})
        run.assert_not_called()

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
        self.enterContext(patch.dict(os.environ, {"DPQC_DEVICE": "auto"}))
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        original_cwd = Path.cwd()
        os.chdir(temp_dir.name)
        self.addCleanup(os.chdir, original_cwd)
        self.root = Path(temp_dir.name)
        self.save_dir = self.root / "figs" / "dpqc_reset" / "u3_cartan" / "h_0.1"
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
        self.assertEqual(load.call_args.kwargs["device"], "cpu")
        install.assert_called_once_with(module)
        module.run_qfim.assert_called_once_with(include_optimization_path=False)
        return module

    def test_qfim_runs_without_training_archive_and_identifies_reset_model(self):
        self.assertFalse(self.energy_archive.exists())
        self.run_fake_qfim()
        self.assertFalse(self.energy_archive.exists())
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["model_id"], "dpqc_reset_u3_cartan_fixed_rx_pi")
        self.assertEqual(metadata["h_param"], 0.1)
        self.assertEqual(metadata["schema_version"], 3)
        self.assertEqual(metadata["total_parameter_formula"], "60 * L")

    def test_qfim_and_vqe_use_versioned_directory_and_preserve_legacy_results(self):
        legacy_dir = self.root / "figs" / "dpqc_reset" / "h_0.1"
        legacy_metadata = _MODEL._model_metadata(0.1)
        legacy_metadata.update(
            schema_version=2,
            model_id="dpqc_reset_fixed_rx_pi",
            unitary_parameters_per_layer=12,
            total_parameter_formula="12 * L",
        )
        legacy_files = {
            legacy_dir / "reset_model_metadata.json": json.dumps(legacy_metadata).encode(),
            legacy_dir / "numerical_results" / "energy" / "vqe_optimization_histories.npz": b"old training results",
            legacy_dir / "numerical_results" / "qfim" / "qfim_random_points.npz": b"old QFIM results",
            legacy_dir / "numerical_results" / "hessian" / "hessian_random_points.npz": b"old Hessian results",
        }
        for path, contents in legacy_files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)

        for stage, entry in (("qfim", _QFIM.run_qfim), ("vqe", _VQE.run_vqe)):
            module = self.make_stage(stage)
            options = {"h_param": 0.1}
            if stage == "vqe":
                options["vqe_batch_size"] = 3
            with self.subTest(stage=stage), patch.object(
                _MODEL, "_load_base_stage_module", return_value=module
            ), patch.object(_MODEL, "_install_reset_model"):
                entry(**options)
                getattr(module, f"run_{stage}").assert_called_once()
                self.assertEqual(Path(module.save_dir), self.save_dir)
                self.assertEqual(Path(module.vqe_optimization_result_path), self.energy_archive)
                self.assertEqual(
                    Path(module.qfim_results_dir), self.save_dir / "numerical_results" / "qfim",
                )
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(metadata["schema_version"], 3)
                self.assertEqual(metadata["model_id"], "dpqc_reset_u3_cartan_fixed_rx_pi")
                self.assertEqual(
                    {path: path.read_bytes() for path in legacy_dir.rglob("*") if path.is_file()},
                    legacy_files,
                )

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
        load.assert_called_once_with("vqe", h_param=0.1, vqe_batch_size=3, device=None)
        install.assert_called_once_with(module)
        module.run_vqe.assert_called_once_with()
        self.assertTrue(self.metadata_path.is_file())

    def test_versioned_directory_rejects_incompatible_metadata_before_either_stage(self):
        _MODEL._ensure_model_metadata(self.save_dir, 0.1)
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            schema_version=2,
            model_id="dpqc_reset_fixed_rx_pi",
            unitary_parameters_per_layer=12,
            total_parameter_formula="12 * L",
        )
        self.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        before = self.metadata_path.read_bytes()
        for stage, entry in (("qfim", _QFIM.run_qfim), ("vqe", _VQE.run_vqe)):
            module = self.make_stage(stage)
            with self.subTest(stage=stage), patch.object(
                _MODEL, "_load_base_stage_module", return_value=module
            ), patch.object(_MODEL, "_install_reset_model"), self.assertRaisesRegex(
                ValueError, "Incompatible reset archive metadata",
            ):
                entry(h_param=0.1)
            getattr(module, f"run_{stage}").assert_not_called()
            self.assertEqual(self.metadata_path.read_bytes(), before)


    def test_explicit_device_is_forwarded_to_only_the_selected_base_stage(self):
        for stage, entry in (("vqe", _VQE.run_vqe), ("qfim", _QFIM.run_qfim)):
            module = self.make_stage(stage)
            options = {"h_param": 0.1, "device": "gpu"}
            if stage == "vqe":
                options["vqe_batch_size"] = 3
            with self.subTest(stage=stage), patch.object(
                _MODEL, "_load_base_stage_module", return_value=module
            ) as load, patch.object(_MODEL, "_install_reset_model"):
                entry(**options)
            load.assert_called_once_with(stage, **options)
            if stage == "qfim":
                module.run_qfim.assert_called_once_with(include_optimization_path=False)
            else:
                module.run_vqe.assert_called_once_with()


class ResetHessianStageTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"DPQC_DEVICE": "auto"}))
        route_patch = patch.object(
            _HESSIAN, "maybe_relaunch_in_wsl", return_value=None
        )
        route_patch.start()
        self.addCleanup(route_patch.stop)

    def test_default_hessian_forwards_cpu_to_shared_stage(self):
        with patch.object(
            _HESSIAN.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            self.assertEqual(_HESSIAN.main(["--h-param", "0.1"]), 0)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--device") + 1], "cpu")
        self.assertEqual(Path(command[1]), _MODULE_DIR / "DPQC_overparam_hessian.py")

    def test_reset_analysis_resolves_default_and_overrides_before_wsl_routing(self):
        for module in (_QFIM, _HESSIAN):
            for inherited in ("auto", "gpu"):
                for arguments, expected in (
                    ([], "cpu" if inherited == "auto" else inherited),
                    (["--device", "auto"], "cpu"),
                    (["--device", "gpu"], "gpu"),
                    (["--device", "cpu"], "cpu"),
                ):
                    with self.subTest(module=module.__name__, inherited=inherited, arguments=arguments), patch.dict(
                        os.environ, {"DPQC_DEVICE": inherited}
                    ), patch.object(
                        module, "maybe_relaunch_in_wsl", return_value=27,
                    ) as route, patch.object(_MODEL, "_load_base_stage_module") as load, patch.object(
                        _HESSIAN.subprocess, "run",
                    ) as run:
                        self.assertEqual(module.main(arguments), 27)
                    self.assertEqual(route.call_args.args[2], expected)
                    load.assert_not_called()
                    run.assert_not_called()

    def test_hessian_uses_reset_family_and_forwards_options_and_exit_status(self):
        options = [
            "--h-param", "0.1", "--layers", "1,4", "--num-samples", "2",
            "--seed-base", "17", "--hvp-chunk-size", "3",
            "--output-dir", "an output directory", "--device", "gpu",
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
