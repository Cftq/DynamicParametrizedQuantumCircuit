"""Test Windows-to-WSL routing without starting WSL or another process."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


_SPEC = importlib.util.spec_from_file_location(
    "dpqc_wsl", Path(__file__).with_name("dpqc_wsl.py"),
)
runtime = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runtime)


class DPQCWSLRoutingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = Path(self.directory.name) / ".dpqc-gpu-wsl.json"
        self.script = Path(self.directory.name) / "日本語 folder" / "VQE example.py"
        self.child = mock.Mock(returncode=0)
        self.runner = self.enterContext(mock.patch.object(
            runtime.subprocess, "run", return_value=self.child,
        ))
        self.enterContext(mock.patch.object(runtime, "RUNTIME_CONFIG", self.config))
        self.enterContext(mock.patch.object(runtime.sys, "platform", "win32"))
        self.enterContext(mock.patch.dict(os.environ, {"DPQC_USE_WSL": "1"}))
        for name in (
            "DPQC_DENSITY_KERNEL", "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
        ):
            os.environ.pop(name, None)
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def _write_config(self, **overrides):
        config = {
            "distribution": "Ubuntu GPU",
            "python": "/home/研究 user/dpqc venv/bin/python",
        }
        config.update(overrides)
        self.config.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    def _command(self):
        self.runner.assert_called_once()
        self.assertEqual(self.runner.call_args.kwargs, {"check": False})
        command, = self.runner.call_args.args
        self.assertIsInstance(command, list)
        return command

    def test_unconfigured_auto_keeps_native_execution(self):
        self.assertIsNone(runtime.maybe_relaunch_in_wsl(self.script, [], "auto"))
        self.runner.assert_not_called()

    def test_unconfigured_cpu_keeps_native_execution(self):
        self.assertIsNone(runtime.maybe_relaunch_in_wsl(self.script, [], "cpu"))
        self.runner.assert_not_called()

    def test_unconfigured_explicit_gpu_reports_setup_command(self):
        with self.assertRaisesRegex(RuntimeError, "setup_dpqc_gpu_wsl.py"):
            runtime.maybe_relaunch_in_wsl(self.script, [], "gpu")
        self.runner.assert_not_called()

    def test_linux_uses_native_execution_even_when_configured(self):
        self._write_config()
        with mock.patch.object(runtime.sys, "platform", "linux"):
            self.assertIsNone(runtime.maybe_relaunch_in_wsl(self.script, [], "gpu"))
        self.runner.assert_not_called()

    def test_environment_bypass_keeps_native_execution(self):
        self._write_config()
        with mock.patch.dict(os.environ, {"DPQC_USE_WSL": "0"}):
            self.assertIsNone(runtime.maybe_relaunch_in_wsl(self.script, [], "gpu"))
        self.runner.assert_not_called()

    def test_configured_auto_selects_gpu_and_preserves_structured_unicode_paths(self):
        self._write_config()
        args = ["--h-param", "0.1", "--label", "実験 1; $(echo untouched)"]
        self.assertEqual(runtime.maybe_relaunch_in_wsl(self.script, args, "auto"), 0)
        command = self._command()
        self.assertEqual(command[:3], ["wsl.exe", "--distribution", "Ubuntu GPU"])
        self.assertEqual(command[3:5], ["--cd", str(Path.cwd())])
        self.assertEqual(command[5:7], ["--exec", "/home/研究 user/dpqc venv/bin/python"])
        self.assertEqual(command[7], runtime._linux_path(str(self.script.resolve())))
        self.assertEqual(command[8:], args + ["--device", "gpu"])
        self.assertEqual(args[-1], "実験 1; $(echo untouched)")

    def test_cpu_comparison_uses_same_configured_python(self):
        self._write_config()
        runtime.maybe_relaunch_in_wsl(self.script, ["--device", "cpu"], "cpu")
        command = self._command()
        self.assertEqual(command[6], "/home/研究 user/dpqc venv/bin/python")
        self.assertEqual(command[8:], ["--device", "cpu"])

    def test_launcher_preserves_auto_across_wsl_for_later_stage_selection(self):
        self._write_config()
        arguments = ["--stage", "all", "--device=auto", "--h-param", "0.1"]
        self.assertEqual(runtime.maybe_relaunch_in_wsl(
            self.script, arguments, "auto", preserve_auto=True,
        ), 0)
        self.assertEqual(self._command()[8:], [
            "--stage", "all", "--h-param", "0.1", "--device", "auto",
        ])

    def test_auto_preservation_does_not_change_explicit_devices_or_duplicate_options(self):
        for device in ("auto", "cpu", "gpu"):
            with self.subTest(device=device):
                self.assertEqual(runtime._forward_arguments(
                    ["--device=gpu", "--stage", "all", "--device", "cpu"],
                    device, preserve_auto=True,
                ), ["--stage", "all", "--device", device])

    def test_split_and_equals_device_options_are_replaced_once(self):
        args = [
            "--device=cpu", "--layers", "1", "--device", "auto",
            "--num-samples=2", "--device=gpu",
        ]
        self.assertEqual(
            runtime._forward_arguments(args, "gpu"),
            ["--layers", "1", "--num-samples=2", "--device", "gpu"],
        )

    def test_windows_output_directories_are_converted_in_both_argument_forms(self):
        args = [
            "--output-dir", r"C:\Users\研究 者\実験 results",
            r"--cache-dir=D:\data folder\解析",
        ]
        self.assertEqual(
            runtime._forward_arguments(args, "auto"),
            [
                "--output-dir", "/mnt/c/Users/研究 者/実験 results",
                "--cache-dir=/mnt/d/data folder/解析", "--device", "gpu",
            ],
        )

    def test_linux_and_relative_paths_and_nonpath_values_remain_unchanged(self):
        args = [
            "--output-dir=/home/研究 user/results", "relative path/結果",
            "--label=x=y", "--h-param", "-0.1",
        ]
        self.assertEqual(runtime._forward_arguments(args, "cpu"), args + ["--device", "cpu"])

    def test_slash_windows_paths_and_drive_root_convert(self):
        self.assertEqual(runtime._linux_path("C:/Users/研究 者/result"), "/mnt/c/Users/研究 者/result")
        self.assertEqual(runtime._linux_path("D:\\"), "/mnt/d/")

    def test_none_argv_uses_current_cli_without_program_name(self):
        self._write_config()
        with mock.patch.object(sys, "argv", ["launcher.py", "--h-param=0.2", "--device=auto"]):
            runtime.maybe_relaunch_in_wsl(self.script, None, "auto")
        self.assertEqual(self._command()[8:], ["--h-param=0.2", "--device", "gpu"])

    def test_child_failure_exit_code_is_preserved(self):
        self._write_config()
        self.child.returncode = 23
        self.assertEqual(runtime.maybe_relaunch_in_wsl(self.script, [], "gpu"), 23)
        self._command()

    def test_selected_environment_values_are_forwarded_as_literal_argv(self):
        self._write_config()
        values = {
            "DPQC_DENSITY_KERNEL": "generic; $(printf ignored)",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.65",
            "EXAMPLE_UNLISTED_ENVIRONMENT": "must remain outside WSL argv",
        }
        with mock.patch.dict(os.environ, values):
            runtime.maybe_relaunch_in_wsl(self.script, ["--h-param", "0.1"], "gpu")
        command = self._command()
        self.assertEqual(command[5:10], [
            "--exec", "env", "DPQC_DENSITY_KERNEL=generic; $(printf ignored)",
            "XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION=0.65",
        ])
        self.assertEqual(command[10], "/home/研究 user/dpqc venv/bin/python")
        self.assertEqual(command[12:], ["--h-param", "0.1", "--device", "gpu"])
        self.assertFalse(any("EXAMPLE_UNLISTED_ENVIRONMENT" in argument for argument in command))
        self.assertNotIn(values["EXAMPLE_UNLISTED_ENVIRONMENT"], command)

    def test_unlisted_environment_does_not_insert_env_wrapper(self):
        self._write_config()
        with mock.patch.dict(os.environ, {"EXAMPLE_UNLISTED_ENVIRONMENT": "unrelated value"}):
            runtime.maybe_relaunch_in_wsl(self.script, [], "cpu")
        command = self._command()
        self.assertEqual(command[5:7], ["--exec", "/home/研究 user/dpqc venv/bin/python"])
        self.assertNotIn("env", command)

    def test_invalid_distribution_fails_before_subprocess(self):
        for value in (None, 12, "", "  "):
            with self.subTest(distribution=value):
                self._write_config(distribution=value)
                with self.assertRaisesRegex(ValueError, "Invalid WSL distribution"):
                    runtime.maybe_relaunch_in_wsl(self.script, [], "gpu")
        self.runner.assert_not_called()

    def test_invalid_python_path_fails_before_subprocess(self):
        for value in (None, 12, "", "python", r"C:\python.exe"):
            with self.subTest(python=value):
                self._write_config(python=value)
                with self.assertRaisesRegex(ValueError, "Invalid Linux Python path"):
                    runtime.maybe_relaunch_in_wsl(self.script, [], "gpu")
        self.runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
