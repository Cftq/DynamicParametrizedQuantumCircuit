"""Ensure the reset saved-energy mode never enables Hessian calculation.

Run with ``python -m unittest discover -s src/dpqc -p test_reset_gap_normalized_visualization.py``.
Only command construction is exercised; numerical jobs are not launched.
"""

from contextlib import redirect_stderr
import importlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
_VISUALIZE = importlib.import_module("DPQC_overparam_reset_visualize")


class ResetGapNormalizedVisualizationTests(unittest.TestCase):
    def test_gap_mode_forwards_fixed_family_without_hessian_options(self):
        command = _VISUALIZE._build_visualizer_command(
            0.1,
            gap_normalized_only=True,
            reuse_hessian_results=True,
            hessian_results_dir=Path("unused results"),
            hessian_figures_dir=Path("unused figures"),
            hessian_layers=(1, 2),
            hessian_rank_threshold=1e-9,
            hessian_num_samples=7,
            hessian_seed_base=11,
            hessian_hvp_chunk_size=3,
        )
        self.assertEqual(
            command[:2],
            (sys.executable, str(_MODULE_DIR / "DPQC_overparam_visualize.py")),
        )
        self.assertIn("--gap-normalized-only", command)
        self.assertEqual(command[command.index("--h-param") + 1], "0.1")
        self.assertEqual(command[command.index("--output-family") + 1], "dpqc_reset")
        self.assertFalse(any("hessian" in option for option in command[2:]))

    def test_plain_visualization_still_reuses_saved_hessians(self):
        command = _VISUALIZE._build_visualizer_command(0.1)
        self.assertNotIn("--gap-normalized-only", command)
        self.assertIn("--with-hessian", command)
        self.assertIn("--reuse-hessian-results", command)

    def test_explicit_hessian_modes_keep_existing_calculation_behavior(self):
        for option in ("hessian_only", "with_hessian"):
            with self.subTest(option=option):
                command = _VISUALIZE._build_visualizer_command(0.1, **{option: True})
                self.assertIn("--" + option.replace("_", "-"), command)
                self.assertNotIn("--reuse-hessian-results", command)
                self.assertNotIn("--gap-normalized-only", command)

    def test_gap_mode_launches_only_plotter_and_propagates_status(self):
        with patch.object(
            _VISUALIZE.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 13),
        ) as run:
            status = _VISUALIZE.main(["--h-param", "0.1", "--gap-normalized-only"])
        self.assertEqual(status, 13)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(Path(command[1]), _MODULE_DIR / "DPQC_overparam_visualize.py")
        self.assertIn("--gap-normalized-only", command)
        self.assertFalse(any("hessian" in option for option in command[2:]))
        self.assertEqual(run.call_args.kwargs["env"]["MPLBACKEND"], "Agg")
        self.assertFalse(run.call_args.kwargs["shell"])

    def test_cli_rejects_gap_and_hessian_modes_before_launch(self):
        for hessian_option in ("--hessian-only", "--with-hessian"):
            with self.subTest(option=hessian_option), patch.object(
                _VISUALIZE.subprocess, "run"
            ) as run, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                _VISUALIZE.main(["--gap-normalized-only", hessian_option])
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()

    def test_command_builder_also_rejects_conflicting_modes(self):
        for option in ("hessian_only", "with_hessian"):
            with self.subTest(option=option), self.assertRaisesRegex(
                ValueError, "mutually exclusive"
            ):
                _VISUALIZE._build_visualizer_command(
                    0.1, gap_normalized_only=True, **{option: True}
                )


if __name__ == "__main__":
    unittest.main()
