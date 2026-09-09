"""Regression tests for the ordinary DPQC computation launcher.

Run with ``python -m unittest discover -s src/dpqc -p test_dpqc_compute_launcher.py``.
All numerical subprocesses are mocked.
"""

import importlib.util
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

    def setUp(self):
        config_patch = patch.object(
            _LAUNCHER, "_default_config_values", return_value=(0.25, 17)
        )
        config_patch.start()
        self.addCleanup(config_patch.stop)

    def assert_stage_calls(self, run, stages, *, h_param="0.1", batch_size="3"):
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
            expected = {"--h-param": h_param}
            if stage == "vqe":
                expected["--vqe-batch-size"] = batch_size
            elif stage == "hessian":
                expected["--output-family"] = "dpqc"
            self.assertEqual(parsed_options, expected)

    def test_default_all_runs_vqe_qfim_and_dpqc_hessian_in_order(self):
        with patch.object(
            _LAUNCHER.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            result = _LAUNCHER.main(["--h-param", "0.1", "--vqe-batch-size", "3"])
        self.assertEqual(result, 0)
        self.assert_stage_calls(run, self.STAGES)

    def test_no_arguments_forward_configuration_defaults_to_all_stages(self):
        with patch.object(
            _LAUNCHER.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            result = _LAUNCHER.main([])
        self.assertEqual(result, 0)
        self.assert_stage_calls(run, self.STAGES, h_param="0.25", batch_size="17")

    def test_explicit_stage_selection_runs_only_requested_stages(self):
        for selection in ("all", *self.STAGES):
            with self.subTest(stage=selection), patch.object(
                _LAUNCHER.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run:
                result = _LAUNCHER.main(
                    ["--stage", selection, "--h-param", "0.1", "--vqe-batch-size", "3"]
                )
                expected_stages = self.STAGES if selection == "all" else (selection,)
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
                    ["--h-param", "0.1", "--vqe-batch-size", "3"]
                )
                self.assertEqual(result, exit_code)
                self.assert_stage_calls(run, self.STAGES[: failed_index + 1])

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
