"""Measured 1 QFIM-rank plots from synthetic saved spectra only."""

import ast
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


_MODULE_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _MODULE_DIR.parent / "common"
_VISUALIZER = _MODULE_DIR / "unitary_pqc_measured_1_overparam_visualize.py"
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from test_measured_unitary_logdet_visualization import (
    _fixture as _canonical_fixture,
    _lightweight_namespace,
)

_THRESHOLD = 1e-12


def _fixture(keep_key):
    fields = _canonical_fixture(keep_key)
    below = np.nextafter(_THRESHOLD, 0.0)
    above = np.nextafter(_THRESHOLD, np.inf)
    spectra = {
        1: [[2., _THRESHOLD, below, 0., 0.],
            [above, _THRESHOLD, _THRESHOLD, below, 0.],
            [0., 0., 0., 0., 0.]],
        3: [[4., 2., 1., _THRESHOLD, below],
            [3 * _THRESHOLD, 2 * _THRESHOLD, _THRESHOLD, below, 0.],
            [below, 0., 0., 0., 0.]],
    }
    for layer, values in spectra.items():
        eigs = np.zeros((3, 62 * layer))
        eigs[:, :5] = values
        fields[f"L{layer}_eigs_desc"] = eigs
    # This intentionally disagrees with the requested threshold. The plot
    # must count the saved raw eigenvalues using its own chosen threshold.
    fields["qfim_effective_rank_threshold"] = 100.0
    return fields


def _write_archives(directory, *, mutate=None):
    directory.mkdir()
    for keep_key in ("keep0123", "keep01234"):
        fixture = _fixture(keep_key)
        if mutate is not None:
            mutate(keep_key, fixture)
        np.savez(directory / f"qfim_random_points_{keep_key}.npz", **fixture)


class MeasuredUnitaryRankVisualizationTests(unittest.TestCase):
    def test_rank_only_cli_uses_inclusive_cutoff_and_saved_counts_without_quantum_runtime(self):
        launcher = r'''
import runpy, sys
for name in ("jax", "tensorcircuit", "optax",
             "unitary_pqc_measured_1_overparam_common",
             "unitary_pqc_measured_1_overparam_compute",
             "unitary_pqc_measured_1_overparam_qfim",
             "unitary_pqc_measured_1_overparam_vqe",
             "unitary_pqc_measured_1_overparam_hessian",
             "unitary_pqc_measured_1_overparam_hs"):
    sys.modules[name] = None
sys.path.insert(0, sys.argv[1])
import config_overparam as cfg
cfg.NUM_QFIM_SAMPLES = 999
cfg.QFIM_EFFECTIVE_RANK_THRESHOLD = 50.0
cfg.UNITARY_PQC_MAX_LAYER = 1
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, outputs = root / "saved qfim", root / "rank figures"
            _write_archives(inputs)
            originals = {path: path.read_bytes() for path in inputs.iterdir()}
            completed = subprocess.run(
                [sys.executable, "-c", launcher, str(_COMMON_DIR), str(_VISUALIZER),
                 "--h-param", "0.1", "--qfim-rank-only", "--qfim-rank-threshold", "1e-12",
                 "--qfim-rank-results-dir", str(inputs),
                 "--qfim-rank-figures-dir", str(outputs)],
                cwd=root, env={**os.environ, "MPLBACKEND": "Agg"},
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(len(list(outputs.glob("*.pdf"))), 2)
            self.assertEqual(len(list(outputs.glob("*.npz"))), 2)
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)
            for keep_key in ("keep0123", "keep01234"):
                stem = f"qfim_rank_mean_sem_min_max_random_points_ge_1e-12_{keep_key}"
                self.assertTrue((outputs / f"{stem}.pdf").read_bytes().startswith(b"%PDF-"))
                with np.load(outputs / f"{stem}.npz", allow_pickle=False) as archive:
                    np.testing.assert_array_equal(archive["layers"], [1, 3])
                    np.testing.assert_array_equal(archive["num_samples_by_layer"], [3, 3])
                    np.testing.assert_array_equal(archive["num_parameters_by_layer"], [62, 186])
                    np.testing.assert_array_equal(archive["L1_rank"], [2, 3, 0])
                    np.testing.assert_array_equal(archive["L3_rank"], [4, 3, 0])
                    np.testing.assert_allclose(archive["mean"], [5/3, 7/3])
                    np.testing.assert_allclose(archive["sem"], np.sqrt([7, 13]) / 3)
                    np.testing.assert_array_equal(archive["min"], [0, 0])
                    np.testing.assert_array_equal(archive["max"], [3, 4])
                    self.assertEqual(float(archive["threshold"]), _THRESHOLD)
                    self.assertEqual(str(archive["rank_criterion"]), "eigenvalue >= threshold (inclusive)")
                    self.assertEqual(int(archive["source_measurement_outcome"]), 1)
            self.assertFalse(any(root.rglob("*vqe*")))
            self.assertFalse(any(root.rglob("*hessian*")))
            self.assertEqual(len(list(inputs.iterdir())), 2)

    def test_custom_threshold_recounts_raw_spectra_instead_of_archived_rank(self):
        function = _lightweight_namespace()["run_unitary_qfim_rank_visualization"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def add_obsolete_ranks(keep_key, fixture):
                fixture["L1_rank"] = np.full(3, 999)
                fixture["L3_rank"] = np.full(3, 999)
            _write_archives(root / "inputs", mutate=add_obsolete_ranks)
            result = function(h_param=0.1, results_dir=root / "inputs",
                              figures_dir=root / "outputs", rank_threshold=2.0)
            self.assertEqual(Path(result["qfim_rank_fig_dir"]), root / "outputs")
            for keep_key in ("keep0123", "keep01234"):
                output = result["outputs"][keep_key]
                statistics = output["statistics"]
                np.testing.assert_array_equal(statistics["rank_by_layer"][1], [1, 0, 0])
                np.testing.assert_array_equal(statistics["rank_by_layer"][3], [2, 0, 0])
                self.assertEqual(float(statistics["threshold"]), 2.0)
                self.assertEqual(Path(output["figure_path"]).name,
                                 f"qfim_rank_mean_sem_min_max_random_points_ge_2e00_{keep_key}.pdf")

    def test_incompatible_second_archive_fails_before_creating_outputs(self):
        function = _lightweight_namespace()["run_unitary_qfim_rank_visualization"]
        cases = {
            "num_params_per_layer": 14, "measurement_outcome": 0,
            "ansatz": "dpqc", "h_param": 0.5,
            "analysis_kind": "optimization_path", "eigenvalues_threshold_masked": True,
            "keep_wires": np.asarray([0, 1, 2, 3]), "qfim_definition": "HS",
            "representation": "reduced_mixed", "L3_eigs_desc": np.zeros((3, 14 * 3)),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for key, value in cases.items():
                inputs, outputs = root / key, root / (key + "_outputs")
                def mutate(keep_key, fixture):
                    if keep_key == "keep01234":
                        fixture[key] = value
                _write_archives(inputs, mutate=mutate)
                with self.subTest(key=key), self.assertRaises((ValueError, KeyError)):
                    function(h_param=0.1, results_dir=inputs, figures_dir=outputs)
                self.assertFalse(outputs.exists())
            inputs, outputs = root / "missing_second", root / "missing_outputs"
            _write_archives(inputs)
            (inputs / "qfim_random_points_keep01234.npz").unlink()
            with self.assertRaises(FileNotFoundError):
                function(h_param=0.1, results_dir=inputs, figures_dir=outputs)
            self.assertFalse(outputs.exists())

    def test_rank_cli_constraints_and_config_default(self):
        namespace = _lightweight_namespace()
        parse = namespace["_parse_cli_args"]
        self.assertEqual(parse(["--qfim-rank-only"]).qfim_rank_threshold,
                         float(namespace["cfg"].QFIM_EFFECTIVE_RANK_THRESHOLD))
        for arguments in (
            ["--qfim-rank-only", "--hessian-only"],
            ["--qfim-rank-only", "--gap-normalized-only"],
            ["--qfim-rank-only", "--qfim-logdet-only"],
            ["--qfim-rank-results-dir", "unused"],
            ["--qfim-rank-figures-dir", "unused"],
            ["--qfim-rank-only", "--gap-results-dir", "unused"],
            ["--qfim-rank-only", "--qfim-logdet-results-dir", "unused"],
            *(["--qfim-rank-threshold", value] for value in ("0", "-1", "nan", "inf")),
        ):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                parse(arguments)
            self.assertEqual(raised.exception.code, 2)

    def test_standard_visualization_reaches_saved_rank_workflow(self):
        # Follow local helper calls so normal integration can remain in the
        # random-QFIM plotting helper, without requiring a direct main call.
        tree = ast.parse(_VISUALIZER.read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        pending, reachable = ["run_unitary_pqc_visualization"], set()
        while pending:
            name = pending.pop()
            if name in reachable:
                continue
            reachable.add(name)
            pending.extend(
                node.func.id for node in ast.walk(functions[name])
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in functions and node.func.id not in reachable
            )
        self.assertIn("run_unitary_qfim_rank_visualization", reachable)


if __name__ == "__main__":
    unittest.main()
