"""Regression tests for the ordinary DPQC computation launcher.

Run with ``python -m unittest discover -s src/dpqc -p test_dpqc_compute_launcher.py``.
All numerical subprocesses are mocked.
"""

from contextlib import redirect_stderr
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


_MODULE_PATH = Path(__file__).with_name("DPQC_overparam_compute.py")
_SPEC = importlib.util.spec_from_file_location("dpqc_compute_launcher", _MODULE_PATH)
_LAUNCHER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LAUNCHER)


class DPQCComputeLauncherTests(unittest.TestCase):
    STAGES = ("vqe", "qfim", "hessian")
    ANALYSIS_STAGES = ("qfim", "hessian")

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

    def assert_stage_calls(
        self, run, stages, *, h_param="0.1", batch_size="3", device="auto"
    ):
        self.assertEqual(run.call_count, len(stages))
        for invocation, stage in zip(run.call_args_list, stages):
            self.assertEqual(invocation.kwargs, {"check": False})
            self.assertEqual(len(invocation.args), 1)
            command = invocation.args[0]
            self.assertEqual(command[0], _LAUNCHER.sys.executable)
            self.assertEqual(
                Path(command[1]),
                _MODULE_PATH.with_name(f"DPQC_overparam_{stage}.py"),
            )
            options = command[2:]
            self.assertEqual(len(options) % 2, 0)
            parsed_options = dict(zip(options[::2], options[1::2]))
            self.assertEqual(len(parsed_options) * 2, len(options))
            stage_device = "cpu" if device == "auto" and stage != "vqe" else device
            expected = {"--h-param": h_param, "--device": stage_device}
            if stage == "vqe":
                expected["--vqe-batch-size"] = batch_size
            elif stage == "hessian":
                expected["--output-family"] = "dpqc"
            self.assertEqual(parsed_options, expected)

    def test_default_analysis_runs_qfim_and_hessian_without_training(self):
        with patch.object(
            _LAUNCHER.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            result = _LAUNCHER.main(["--h-param", "0.1", "--vqe-batch-size", "3"])
        self.assertEqual(result, 0)
        self.assert_stage_calls(run, self.ANALYSIS_STAGES)

    def test_no_arguments_forward_configuration_defaults_without_training(self):
        with patch.object(
            _LAUNCHER.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            result = _LAUNCHER.main([])
        self.assertEqual(result, 0)
        self.assert_stage_calls(run, self.ANALYSIS_STAGES, h_param="0.25")

    def test_explicit_stage_selection_runs_only_requested_stages(self):
        for selection in ("analysis", "all", *self.STAGES):
            with self.subTest(stage=selection), patch.object(
                _LAUNCHER.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run:
                result = _LAUNCHER.main(
                    ["--stage", selection, "--h-param", "0.1", "--vqe-batch-size", "3"]
                )
                if selection == "all":
                    expected_stages = self.STAGES
                elif selection == "analysis":
                    expected_stages = self.ANALYSIS_STAGES
                else:
                    expected_stages = (selection,)
                self.assertEqual(result, 0)
                self.assert_stage_calls(run, expected_stages)

    def test_all_stops_at_each_failed_stage_and_returns_its_exit_code(self):
        for failed_index, failed_stage in enumerate(self.STAGES):
            exit_code = 7 + failed_index
            completions = [
                subprocess.CompletedProcess([], 0) for _ in range(failed_index)
            ] + [subprocess.CompletedProcess([], exit_code)]
            with self.subTest(stage=failed_stage), patch.object(
                _LAUNCHER.subprocess, "run", side_effect=completions
            ) as run:
                result = _LAUNCHER.main(
                    ["--stage", "all", "--h-param", "0.1", "--vqe-batch-size", "3"]
                )
                self.assertEqual(result, exit_code)
                self.assert_stage_calls(run, self.STAGES[: failed_index + 1])

    def test_analysis_failure_stops_pipeline_without_attempting_training(self):
        for failed_index, failed_stage in enumerate(self.ANALYSIS_STAGES):
            exit_code = 11 + failed_index
            completions = [
                subprocess.CompletedProcess([], 0) for _ in range(failed_index)
            ] + [subprocess.CompletedProcess([], exit_code)]
            with self.subTest(stage=failed_stage), patch.object(
                _LAUNCHER.subprocess, "run", side_effect=completions
            ) as run:
                result = _LAUNCHER.main(["--h-param", "0.1"])
                self.assertEqual(result, exit_code)
                self.assert_stage_calls(
                    run, self.ANALYSIS_STAGES[: failed_index + 1]
                )

    def test_explicit_device_is_forwarded_to_every_selected_stage(self):
        for device in ("cpu", "gpu"):
            with self.subTest(device=device), patch.object(
                _LAUNCHER.subprocess, "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run:
                result = _LAUNCHER.main([
                    "--stage", "all", "--h-param", "0.1",
                    "--vqe-batch-size", "3", "--device", device,
                ])
                self.assertEqual(result, 0)
                self.assert_stage_calls(run, self.STAGES, device=device)

    def test_gpu_default_analysis_still_excludes_training(self):
        with patch.object(
            _LAUNCHER.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            result = _LAUNCHER.main(["--h-param", "0.1", "--device", "gpu"])
        self.assertEqual(result, 0)
        self.assert_stage_calls(run, self.ANALYSIS_STAGES, device="gpu")

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
                    self.assert_stage_calls(run, self.STAGES, device=expected)

    def test_default_wsl_route_preserves_auto_before_per_stage_dispatch(self):
        arguments = ["--stage", "all"]
        with patch.object(_LAUNCHER.subprocess, "run") as run, patch.object(
            _LAUNCHER.dpqc_wsl, "maybe_relaunch_in_wsl", return_value=0,
        ) as route:
            self.assertEqual(_LAUNCHER.main(arguments), 0)
        self.assertEqual(route.call_args.args[1:], (arguments, "auto"))
        self.assertEqual(route.call_args.kwargs, {"preserve_auto": True})
        run.assert_not_called()

    def test_invalid_device_fails_before_routing_or_launching(self):
        with patch.object(_LAUNCHER.subprocess, "run") as run, patch.object(
            _LAUNCHER.dpqc_wsl, "maybe_relaunch_in_wsl"
        ) as route, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            _LAUNCHER.main(["--device", "tpu"])
        self.assertEqual(error.exception.code, 2)
        route.assert_not_called()
        run.assert_not_called()

    def test_wsl_routing_returns_status_without_local_stage_launch(self):
        arguments = ["--device", "gpu", "--stage", "hessian", "--h-param", "0.1"]
        with patch.object(_LAUNCHER.subprocess, "run") as run, patch.object(
            _LAUNCHER.dpqc_wsl, "maybe_relaunch_in_wsl", return_value=19
        ) as route:
            result = _LAUNCHER.main(arguments)
        self.assertEqual(result, 19)
        route.assert_called_once()
        self.assertEqual(Path(route.call_args.args[0]), _MODULE_PATH)
        self.assertEqual(route.call_args.args[1:], (arguments, "gpu"))
        self.assertEqual(route.call_args.kwargs, {"preserve_auto": True})
        run.assert_not_called()

    def test_selected_stage_failure_propagates_exit_code(self):
        for stage in self.STAGES:
            with self.subTest(stage=stage), patch.object(
                _LAUNCHER.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 23),
            ) as run:
                result = _LAUNCHER.main(
                    ["--stage", stage, "--h-param", "0.1", "--vqe-batch-size", "3"]
                )
                self.assertEqual(result, 23)
                self.assert_stage_calls(run, (stage,))


if __name__ == "__main__":
    unittest.main()
