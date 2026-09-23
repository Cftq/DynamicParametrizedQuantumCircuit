"""Saved-spectrum visualization tests; no circuit or optimizer is executed."""

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


def _lightweight_namespace():
    source = _VISUALIZER.read_text(encoding="utf-8")
    prefix = source.split('\nif __name__ == "__main__":\n    _CLI_ARGS', 1)[0]
    namespace = {"__name__": "saved_qfim_visualizer", "__file__": str(_VISUALIZER)}
    exec(compile(prefix, str(_VISUALIZER), "exec"), namespace)
    return namespace


def _fixture(keep_key):
    keep = [0, 1, 2, 3] if keep_key == "keep0123" else [0, 1, 2, 3, 4]
    result = {
        "schema_version": 1, "ansatz": "unitary_pqc_measured_1",
        "measurement_outcome": 1, "num_total_qubits": 5,
        "num_params_per_layer": 62, "h_param": 0.1,
        "analysis_kind": "random_points", "keep_key": keep_key,
        "keep_wires": np.asarray(keep, dtype=np.int64),
        "traced_wires": np.asarray([i for i in range(5) if i not in keep], dtype=np.int64),
        "representation": "reduced_mixed" if len(keep) == 4 else "pure_full",
        "qfim_definition": "SLD_QFIM", "eigenvalue_order": "descending",
        "eigenvalues_threshold_masked": False,
        "num_qfim_samples": 3, "qfim_sample_seed_base": 17,
        # This cutoff must not change logdet; small positive eigenvalues count.
        "qfim_effective_rank_threshold": 100.0,
        "layers": np.asarray([3, 1]),
    }
    for layer in (1, 3):
        eigs = np.zeros((3, 62 * layer))
        eigs[:, 0] = [0.0, layer, 3 * layer]
        eigs[0, 0] = 1e-15
        result[f"L{layer}_eigs_desc"] = eigs
        # Prove that the reader never loads parameter arrays.
        result[f"L{layer}_theta"] = np.asarray([object()], dtype=object)
    return result


def _write_archives(directory, *, mutate=None):
    directory.mkdir()
    fixtures = {}
    for keep_key in ("keep0123", "keep01234"):
        fixture = _fixture(keep_key)
        if mutate is not None:
            mutate(keep_key, fixture)
        np.savez(directory / f"qfim_random_points_{keep_key}.npz", **fixture)
        fixtures[keep_key] = fixture
    return fixtures


class MeasuredUnitaryLogdetTests(unittest.TestCase):
    def test_only_cli_without_quantum_dependencies_uses_saved_sample_counts(self):
        launcher = r'''
import runpy, sys
for name in ("jax", "tensorcircuit", "optax",
             "unitary_pqc_measured_1_overparam_common",
             "unitary_pqc_measured_1_overparam_compute",
             "unitary_pqc_measured_1_overparam_qfim",
             "unitary_pqc_measured_1_overparam_vqe"):
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
            inputs, outputs = root / "saved qfim", root / "figures"
            fixtures = _write_archives(inputs)
            originals = {path: path.read_bytes() for path in inputs.iterdir()}
            completed = subprocess.run(
                [sys.executable, "-c", launcher, str(_COMMON_DIR), str(_VISUALIZER),
                 "--h-param", "0.1", "--qfim-logdet-only",
                 "--qfim-logdet-results-dir", str(inputs),
                 "--qfim-logdet-figures-dir", str(outputs)],
                cwd=root, env={**os.environ, "MPLBACKEND": "Agg"},
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(len(list(outputs.glob("*.pdf"))), 2)
            self.assertEqual(len(list(outputs.glob("*.npz"))), 2)
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)
            for keep_key, fixture in fixtures.items():
                stem = f"qfim_logdet_random_points_{keep_key}_kappa_1"
                self.assertTrue((outputs / f"{stem}.pdf").read_bytes().startswith(b"%PDF-"))
                with np.load(outputs / f"{stem}.npz", allow_pickle=False) as archive:
                    np.testing.assert_array_equal(archive["layers"], [1, 3])
                    np.testing.assert_array_equal(archive["num_samples_by_layer"], [3, 3])
                    self.assertEqual(float(archive["kappa"]), 1.0)
                    expected = [np.log1p(fixture[f"L{layer}_eigs_desc"]).sum(axis=1).mean()
                                for layer in (1, 3)]
                    np.testing.assert_allclose(archive["mean"], expected)
            self.assertFalse(any(root.rglob("*vqe*")))

    def test_custom_kappa_keeps_small_eigenvalues(self):
        function = _lightweight_namespace()["run_unitary_qfim_logdet_visualization"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = _write_archives(root / "qfim")
            result = function(h_param=0.1, results_dir=root / "qfim",
                              figures_dir=root / "figures", kappa=1e15)
            for keep_key, fixture in fixtures.items():
                statistics = result["outputs"][keep_key]["statistics"]
                for layer in (1, 3):
                    np.testing.assert_allclose(
                        statistics["logdet_by_layer"][layer],
                        np.log1p(1e15 * fixture[f"L{layer}_eigs_desc"]).sum(axis=1),
                    )
                    self.assertAlmostEqual(statistics["logdet_by_layer"][layer][0], np.log(2))

    def test_incompatible_second_archive_fails_before_outputs(self):
        function = _lightweight_namespace()["run_unitary_qfim_logdet_visualization"]
        cases = {
            "num_params_per_layer": 14, "measurement_outcome": 0,
            "ansatz": "dpqc", "h_param": 0.5,
            "analysis_kind": "optimization_path", "eigenvalues_threshold_masked": True,
            "keep_wires": np.asarray([0, 1, 2, 3]), "qfim_definition": "HS",
            "representation": "reduced_mixed",
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
            with self.assertRaises(FileNotFoundError):
                function(h_param=0.1, results_dir=root / "missing", figures_dir=root / "absent")
            self.assertFalse((root / "absent").exists())

    def test_cli_constraints_and_standard_integration(self):
        parse = _lightweight_namespace()["_parse_cli_args"]
        self.assertEqual(parse(["--qfim-logdet-only"]).qfim_logdet_kappa, 1.0)
        for arguments in (
            ["--qfim-logdet-only", "--hessian-only"],
            ["--qfim-logdet-only", "--gap-normalized-only"],
            ["--qfim-logdet-results-dir", "unused"],
            ["--qfim-logdet-figures-dir", "unused"],
            *(["--qfim-logdet-kappa", value] for value in ("0", "-1", "nan", "inf")),
        ):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                parse(arguments)
            self.assertEqual(raised.exception.code, 2)
        tree = ast.parse(_VISUALIZER.read_text(encoding="utf-8"))
        standard = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "run_unitary_pqc_visualization")
        calls = [node for node in ast.walk(standard) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)
                 and node.func.id == "run_unitary_qfim_logdet_visualization"]
        self.assertEqual(len(calls), 1)
        self.assertIn("kappa", [keyword.arg for keyword in calls[0].keywords])


if __name__ == "__main__":
    unittest.main()
