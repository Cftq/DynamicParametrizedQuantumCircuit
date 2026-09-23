"""Saved-spectrum CLI regressions with quantum runtimes explicitly unavailable."""

from contextlib import redirect_stderr
import hashlib
import importlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
import dpqc_reset_model as reset_model

_RESET_VISUALIZER = importlib.import_module("DPQC_overparam_reset_visualize")
_VISUALIZER = _MODULE_DIR / "DPQC_overparam_visualize.py"


def _fixture(parameters_per_layer, keep_key):
    fields = {
        "h_param": 0.1,
        "num_qfim_samples": 3,
        "qfim_sample_seed_base": 7,
        "qfim_effective_rank_threshold": 1e-12,
        "layers": np.array([7, 2]),
        # The log-determinant reader must not access unrelated parameter data.
        "L7_theta": np.array([object()], dtype=object),
    }
    for layer in (2, 7):
        eigs = np.zeros((3, parameters_per_layer * layer))
        eigs[:, :3] = np.array([[3., 1., 1e-14], [2., .5, 1e-15], [0., 0., 0.]]) * layer
        fields[f"L{layer}_eigs_desc"] = eigs
    # Preserve compatibility with legacy DPQC archives lacking keep metadata.
    if keep_key == "keep01234":
        fields["keep_wires"] = np.arange(5)
    return fields


def _write_fixture(root, family):
    h_root = (
        reset_model.reset_output_dir(0.1, root=root) if family == "dpqc_reset"
        else root / "figs" / family / "h_0.1"
    )
    results_dir = h_root / "numerical_results" / "qfim"
    results_dir.mkdir(parents=True)
    if family == "dpqc_reset":
        reset_model._write_model_metadata(h_root, 0.1)
    parameters_per_layer = 60 if family == "dpqc_reset" else 14
    for keep_key in ("keep0123", "keep01234"):
        np.savez(results_dir / f"qfim_random_points_{keep_key}.npz", **_fixture(parameters_per_layer, keep_key))
    return h_root


def _run_cli(root, family, *options):
    script = r'''
import runpy, sys
for name in ("jax", "tensorcircuit", "optax", "DPQC_overparam_vqe",
             "DPQC_overparam_qfim", "DPQC_overparam_hessian",
             "DPQC_overparam_reset_vqe", "DPQC_overparam_reset_qfim",
             "DPQC_overparam_reset_hessian"):
    sys.modules[name] = None
entrypoint, family, *options = sys.argv[1:]
sys.argv = [entrypoint, "--h-param", "0.1", "--output-family", family,
            "--qfim-logdet-only", *options]
runpy.run_path(entrypoint, run_name="__main__")
'''
    return subprocess.run(
        [sys.executable, "-c", script, str(_VISUALIZER), family, *options],
        cwd=root, capture_output=True, text=True, check=False,
        env={**os.environ, "MPLBACKEND": "Agg"}, timeout=60,
    )


class QFIMLogdetVisualizationTests(unittest.TestCase):
    def test_both_families_render_saved_spectra_without_energy_or_quantum_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for family in ("dpqc", "dpqc_reset"):
                with self.subTest(family=family):
                    h_root = _write_fixture(root, family)
                    input_paths = tuple((h_root / "numerical_results" / "qfim").glob("*.npz"))
                    digests = {path: hashlib.sha256(path.read_bytes()).digest() for path in input_paths}
                    completed = _run_cli(root, family, "--qfim-logdet-kappa", "2")
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    self.assertFalse((h_root / "energy_figures").exists())
                    self.assertFalse((h_root / "hessian_figures").exists())
                    self.assertFalse((h_root / "numerical_results" / "energy").exists())
                    self.assertEqual({path: hashlib.sha256(path.read_bytes()).digest() for path in input_paths}, digests)
                    figures = h_root / "figures" / "qfim" / "logdet"
                    self.assertEqual(len(list(figures.glob("*.pdf"))), 2)
                    statistics = list(figures.glob("*.npz"))
                    self.assertEqual(len(statistics), 2)
                    for path in statistics:
                        with np.load(path, allow_pickle=False) as result:
                            np.testing.assert_array_equal(result["layers"], [2, 7])
                            self.assertEqual(float(result["kappa"]), 2.)
                            self.assertEqual(str(result["source_output_family"]), family)
                            expected = [np.log1p(2 * layer * np.array([[3., 1., 1e-14], [2., .5, 1e-15], [0., 0., 0.]])).sum(axis=1).mean() for layer in (2, 7)]
                            np.testing.assert_allclose(result["mean"], expected, rtol=1e-14)

    def test_malformed_second_archive_is_rejected_before_any_plot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            h_root = _write_fixture(root, "dpqc")
            invalid = _fixture(14, "keep01234")
            invalid["L2_eigs_desc"] = np.ones((3, 12 * 2))
            np.savez(h_root / "numerical_results" / "qfim" / "qfim_random_points_keep01234.npz", **invalid)
            completed = _run_cli(root, "dpqc")
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((h_root / "figures").exists())

    def test_reset_requires_current_model_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            h_root = _write_fixture(Path(temporary), "dpqc_reset")
            (h_root / "reset_model_metadata.json").unlink()
            completed = _run_cli(temporary, "dpqc_reset")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("reset_model_metadata.json", completed.stderr)
            self.assertFalse((h_root / "figures").exists())

    def test_reset_mode_forwards_kappa_without_hessian_options(self):
        for only_mode in (False, True):
            command = _RESET_VISUALIZER._build_visualizer_command(0.1, qfim_logdet_only=only_mode, qfim_logdet_kappa=2.5)
            self.assertEqual(command[command.index("--qfim-logdet-kappa") + 1], "2.5")
            if only_mode:
                self.assertIn("--qfim-logdet-only", command)
                self.assertFalse(any("hessian" in option for option in command[2:]))
            else:
                self.assertIn("--with-hessian", command)
                self.assertIn("--reuse-hessian-results", command)

    def test_base_cli_rejects_conflicting_modes_and_invalid_kappa(self):
        with tempfile.TemporaryDirectory() as temporary:
            for options in (["--gap-normalized-only"], ["--hessian-only"], ["--with-hessian"], ["--qfim-logdet-kappa", "nan"], ["--qfim-logdet-kappa", "0"]):
                with self.subTest(options=options):
                    completed = _run_cli(temporary, "dpqc", *options)
                    self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
                    self.assertFalse((Path(temporary) / "figs").exists())

    def test_reset_cli_dispatch_and_invalid_arguments(self):
        with patch.object(_RESET_VISUALIZER.subprocess, "run", return_value=subprocess.CompletedProcess([], 13)) as run:
            status = _RESET_VISUALIZER.main(["--qfim-logdet-only", "--qfim-logdet-kappa", "3"])
        self.assertEqual(status, 13)
        command = run.call_args.args[0]
        self.assertIn("--qfim-logdet-only", command)
        self.assertEqual(command[command.index("--qfim-logdet-kappa") + 1], "3.0")
        for options in (["--gap-normalized-only"], ["--hessian-only"], ["--with-hessian"], ["--qfim-logdet-kappa", "nan"], ["--qfim-logdet-kappa", "0"], ["--qfim-logdet-kappa", "-1"]):
            with self.subTest(options=options), patch.object(_RESET_VISUALIZER.subprocess, "run") as run, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                _RESET_VISUALIZER.main(["--qfim-logdet-only", *options])
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()
        for mode in ("gap_normalized_only", "hessian_only", "with_hessian"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "mutually exclusive"):
                _RESET_VISUALIZER._build_visualizer_command(0.1, qfim_logdet_only=True, **{mode: True})


if __name__ == "__main__":
    unittest.main()
