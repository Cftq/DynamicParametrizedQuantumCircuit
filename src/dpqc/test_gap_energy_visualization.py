"""Saved-energy CLI regressions; quantum runtimes are explicitly blocked."""
import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

_MODULE_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _MODULE_DIR.parent / "common"
for directory in (_MODULE_DIR, _COMMON_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from hessian_curvature import hamiltonian_matrix_numpy
import dpqc_reset_model as reset_model

_VISUALIZER = _MODULE_DIR / "DPQC_overparam_visualize.py"
_EXPECTED_THRESHOLDS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)


def _fixture():
    eigenvalues = np.linalg.eigvalsh(hamiltonian_matrix_numpy(0.1))
    ground, gap = eigenvalues[0], eigenvalues[1] - eigenvalues[0]
    return {
        "h_param": 0.1, "smallest_eigval": ground,
        "num_runs": 3, "steps": 2, "vqe_layers": np.array([7, 2]),
        "L7_energy_traces": ground + gap * np.array([[1, 0.5, 0], [1, 0.5, 5e-8], [1, 0.5, 5e-7]]),
        "L2_energy_traces": ground + gap * np.array([[1, 0.5, 5e-4], [1, 0.5, 2e-3], [1, 0.5, 5e-9]]),
        "optimizer_name": "adam",
        # Object storage would fail if a loader tried to read unrelated arrays.
        "L7_theta_samples": np.array([object()], dtype=object),
    }


def _run_cli(root, family, *, synthetic_gap=False):
    script = r'''
import runpy, sys
for name in ("jax", "tensorcircuit", "optax", "DPQC_overparam_vqe",
             "DPQC_overparam_qfim", "DPQC_overparam_hessian",
             "DPQC_overparam_reset_vqe", "DPQC_overparam_reset_qfim",
             "DPQC_overparam_reset_hessian"):
    sys.modules[name] = None
entrypoint, family, synthetic_gap = sys.argv[1:]
if synthetic_gap == "1":
    from pathlib import Path
    sys.path.insert(0, str(Path(entrypoint).resolve().parent.parent / "common"))
    import numpy as np
    import hessian_curvature
    hessian_curvature.hamiltonian_matrix_numpy = lambda h: np.diag([0.0, 2.0])
sys.argv = [entrypoint, "--h-param", "0.1", "--output-family", family,
            "--gap-normalized-only"]
runpy.run_path(entrypoint, run_name="__main__")
'''
    return subprocess.run(
        [sys.executable, "-c", script, str(_VISUALIZER), family, str(int(synthetic_gap))],
        cwd=root, capture_output=True, text=True, check=False,
        env={**os.environ, "MPLBACKEND": "Agg"}, timeout=60,
    )


class GapEnergyVisualizationTests(unittest.TestCase):
    def test_both_families_render_only_saved_energy_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for family in ("dpqc", "dpqc_reset"):
                with self.subTest(family=family):
                    h_root = (
                        reset_model.reset_output_dir(0.1, root=root) if family == "dpqc_reset"
                        else root / "figs" / family / "h_0.1"
                    )
                    energy_dir = h_root / "numerical_results" / "energy"
                    energy_dir.mkdir(parents=True)
                    archive = energy_dir / "vqe_optimization_histories.npz"
                    np.savez(archive, **_fixture())
                    digest = hashlib.sha256(archive.read_bytes()).digest()
                    if family == "dpqc_reset":
                        reset_model._write_model_metadata(h_root, 0.1)
                    completed = _run_cli(root, family)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    self.assertEqual(hashlib.sha256(archive.read_bytes()).digest(), digest)
                    self.assertFalse((h_root / "qfim_figures").exists())
                    self.assertFalse((h_root / "hessian_figures").exists())
                    figures = h_root / "energy_figures"
                    for name in (
                        "final_gap_normalized_energy_error.pdf",
                        "gap_normalized_success_probability.pdf",
                        "success_probability_multiple_tolerances_gap_normalized.pdf",
                    ):
                        self.assertTrue((figures / name).is_file())
                    with np.load(energy_dir / "gap_normalized_energy_statistics.npz", allow_pickle=False) as data:
                        np.testing.assert_array_equal(data["layers"], [2, 7])
                        np.testing.assert_array_equal(data["num_runs_by_layer"], [3, 3])
                        np.testing.assert_array_equal(data["thresholds"], _EXPECTED_THRESHOLDS)
                        np.testing.assert_allclose(data["success_probabilities"], [
                            [2/3, 1/3, 1/3, 1/3, 1/3, 1/3, 0, 0],
                            [1, 1, 1, 1, 2/3, 1/3, 1/3, 1/3],
                        ])
                        self.assertIn("strict inequality", str(data["success_criterion"]))
                        self.assertEqual(str(data["output_family"]), family)
                        self.assertEqual(str(data["source_energy_definition"]), "last saved energy-trace sample per run")

    def test_eight_cutoffs_use_strict_gap_normalized_final_energy_for_both_families(self):
        # A zero ground energy and power-of-two gap keep equality exact in
        # float64, so this detects <= as well as missing gap normalization.
        final_energies = np.asarray([
            value
            for threshold in _EXPECTED_THRESHOLDS
            for value in (
                np.nextafter(2 * threshold, -np.inf),
                2 * threshold,
                np.nextafter(2 * threshold, np.inf),
            )
        ])
        runs = len(final_energies)
        fixture = {
            "h_param": 0.1, "smallest_eigval": 0.0,
            "num_runs": runs, "steps": 2, "vqe_layers": np.array([7, 2]),
            "L7_energy_traces": np.column_stack((np.full(runs, 2.0), np.zeros(runs), final_energies)),
            "L2_energy_traces": np.column_stack((np.full(runs, 2.0), np.zeros(runs), np.ones(runs))),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for family in ("dpqc", "dpqc_reset"):
                with self.subTest(family=family):
                    h_root = (
                        reset_model.reset_output_dir(0.1, root=root) if family == "dpqc_reset"
                        else root / "figs" / family / "h_0.1"
                    )
                    energy_dir = h_root / "numerical_results" / "energy"
                    energy_dir.mkdir(parents=True)
                    np.savez(energy_dir / "vqe_optimization_histories.npz", **fixture)
                    if family == "dpqc_reset":
                        reset_model._write_model_metadata(h_root, 0.1)
                    completed = _run_cli(root, family, synthetic_gap=True)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    with np.load(energy_dir / "gap_normalized_energy_statistics.npz", allow_pickle=False) as data:
                        self.assertEqual(float(data["spectral_gap"]), 2.0)
                        np.testing.assert_array_equal(data["thresholds"], _EXPECTED_THRESHOLDS)
                        np.testing.assert_array_equal(data["L7_normalized_errors"], final_energies / 2)
                        np.testing.assert_array_equal(data["success_probabilities"][0], np.zeros(8))
                        np.testing.assert_allclose(data["success_probabilities"][1], np.asarray([22, 19, 16, 13, 10, 7, 4, 1]) / 24)

    def test_inconsistent_ground_energy_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            h_root = Path(temporary) / "figs" / "dpqc" / "h_0.1"
            energy_dir = h_root / "numerical_results" / "energy"
            energy_dir.mkdir(parents=True)
            data = _fixture()
            data["smallest_eigval"] += 0.1
            np.savez(energy_dir / "vqe_optimization_histories.npz", **data)
            completed = _run_cli(temporary, "dpqc")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("ground energy does not match", completed.stderr)
            self.assertFalse((h_root / "energy_figures").exists())

    def test_archive_loader_rejects_wrong_h_layers_counts_and_shapes(self):
        # Load just the IO function; importing the legacy plotter normally
        # renders its full workflow as a module-level operation.
        tree = ast.parse(_VISUALIZER.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_load_gap_normalized_vqe_archive"]
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_VISUALIZER), "exec"), namespace)
        loader = namespace["_load_gap_normalized_vqe_archive"]
        invalid = {
            "h_param": [0.5, np.nan, [0.1]],
            "smallest_eigval": [np.nan, 1j],
            "vqe_layers": [[2, 2], [2.0, 7.0], [0, 7], []],
            "num_runs": [0, 3.0, 4],
            "steps": [0, 2.0, 4],
            "L7_energy_traces": [np.zeros((2, 3)), np.zeros((3, 2), dtype=complex), np.zeros((3, 2))],
        }
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "energies.npz"
            # Older archives omit the final post-update energy, so their
            # history length equals steps; both conventions remain usable.
            legacy = _fixture()
            for layer in (2, 7):
                legacy[f"L{layer}_energy_traces"] = legacy[f"L{layer}_energy_traces"][:, [0, -1]]
            np.savez(archive, **legacy)
            energies, _, metadata = loader(archive, expected_h_param=0.1)
            self.assertEqual(metadata["num_stored_energy_samples"], 2)
            np.testing.assert_array_equal(energies[7], legacy["L7_energy_traces"][:, -1])
            for key, values in invalid.items():
                for value in values:
                    data = _fixture()
                    data[key] = value
                    np.savez(archive, **data)
                    with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                        loader(archive, expected_h_param=0.1)


if __name__ == "__main__":
    unittest.main()
