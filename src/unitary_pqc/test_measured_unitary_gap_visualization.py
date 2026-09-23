"""Saved-energy CLI checks; no training or quantum analysis is executed.

Run with ``python -m unittest discover -s src/unitary_pqc -p
test_measured_unitary_gap_visualization.py``. All archives are synthetic and
all output is written into temporary directories.
"""

import ast
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
for directory in (_MODULE_DIR, _COMMON_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from hessian_curvature import hamiltonian_matrix_numpy

_EXPECTED_THRESHOLDS = (1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
_NUM_RUNS = 2 * len(_EXPECTED_THRESHOLDS)


def _archive_fixture():
    """Known final-energy samples straddling each requested accuracy cutoff."""
    eigenvalues = np.linalg.eigvalsh(hamiltonian_matrix_numpy(0.1))
    ground, gap = eigenvalues[0], eigenvalues[1] - eigenvalues[0]
    boundary_values = []
    for threshold in _EXPECTED_THRESHOLDS:
        boundary = ground + gap * threshold
        boundary_values.extend((
            np.nextafter(boundary, -np.inf),
            np.nextafter(boundary, np.inf),
        ))
    layer2 = ground + gap * np.r_[0.0, 5e-11, np.arange(1, _NUM_RUNS - 1) * 1e-3]
    # Earlier samples include a better energy: success must use the last
    # saved sample, not the minimum anywhere in the optimization history.
    return {
        "ansatz": "unitary_pqc_measured_1",
        "measurement_outcome": 1,
        "num_params_per_layer": 62,
        "h_param": 0.1,
        "smallest_eigval": ground,
        "num_runs": _NUM_RUNS,
        "steps": 3,
        "layers": np.asarray([7, 2]),
        "L7_energy_traces": np.column_stack((
            np.full(_NUM_RUNS, ground + gap), np.full(_NUM_RUNS, ground), boundary_values,
        )),
        "L2_energy_traces": np.column_stack((
            np.full(_NUM_RUNS, ground + gap), np.full(_NUM_RUNS, ground), layer2,
        )),
        # Reading unrelated parameter storage would fail with allow_pickle=False.
        "L7_theta_samples": np.asarray([object()], dtype=object),
        "L2_theta_history": np.asarray([object()], dtype=object),
    }


def _run_cli(temporary, arguments, *, synthetic_gap=False):
    launcher = r'''
import runpy, sys
for name in (
    "jax", "optax", "tensorcircuit",
    "unitary_pqc_measured_1_overparam_common",
    "unitary_pqc_measured_1_overparam_compute",
    "unitary_pqc_measured_1_overparam_vqe",
    "unitary_pqc_measured_1_overparam_qfim",
    "unitary_pqc_measured_1_overparam_hessian",
):
    sys.modules[name] = None
sys.path.insert(0, sys.argv[1])
synthetic_gap = sys.argv.pop(2)
if synthetic_gap == "1":
    import numpy as np
    import hessian_curvature
    hessian_curvature.hamiltonian_matrix_numpy = lambda h: np.diag([0.0, 2.0])
import config_overparam as cfg
cfg.NUM_RUNS = 999
cfg.STEPS = 999
cfg.UNITARY_PQC_MAX_LAYER = 1
cfg.UNITARY_PQC_DENSE_UNTIL_LAYER = 1
cfg.NUM_QFIM_SAMPLES = 999
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
    return subprocess.run(
        [sys.executable, "-c", launcher, str(_COMMON_DIR), str(int(synthetic_gap)), str(_VISUALIZER),
         "--h-param", "0.1", *map(str, arguments)],
        cwd=temporary,
        env={**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", timeout=60,
        check=False,
    )


def _gap_arguments(results_dir, figures_dir):
    return [
        "--gap-normalized-only", "--gap-results-dir", results_dir,
        "--gap-figures-dir", figures_dir,
    ]


class MeasuredUnitaryGapVisualizationTests(unittest.TestCase):
    def test_cli_renders_saved_final_energies_without_quantum_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "saved energy"
            figures_dir = Path(temporary) / "energy figures"
            results_dir.mkdir()
            archive_path = results_dir / "vqe_optimization_results.npz"
            fixture = _archive_fixture()
            np.savez_compressed(archive_path, **fixture)
            original = archive_path.read_bytes()

            completed = _run_cli(temporary, _gap_arguments(results_dir, figures_dir))
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(archive_path.read_bytes(), original)
            self.assertEqual(
                {path.name for path in figures_dir.rglob("*.pdf")},
                {
                    "final_gap_normalized_energy_error.pdf",
                    "gap_normalized_success_probability.pdf",
                    "success_probability_multiple_tolerances_gap_normalized.pdf",
                },
            )
            for path in figures_dir.glob("*.pdf"):
                self.assertTrue(path.read_bytes().startswith(b"%PDF-"), str(path))
            self.assertFalse(any(Path(temporary).rglob("*qfim*")))
            self.assertFalse(any(Path(temporary).rglob("*hessian*")))
            with np.load(
                results_dir / "gap_normalized_energy_statistics.npz", allow_pickle=False,
            ) as archive:
                np.testing.assert_array_equal(archive["layers"], [2, 7])
                np.testing.assert_array_equal(archive["num_runs_by_layer"], [_NUM_RUNS, _NUM_RUNS])
                np.testing.assert_array_equal(archive["thresholds"], _EXPECTED_THRESHOLDS)
                np.testing.assert_allclose(
                    archive["success_probabilities"],
                    [[1/6] * 6, [11/12, 9/12, 7/12, 5/12, 3/12, 1/12]],
                )
                self.assertEqual(str(archive["output_family"]), "unitary_pqc_measured_1")
                self.assertEqual(int(archive["measurement_outcome"]), 1)
                self.assertEqual(int(archive["num_params_per_layer"]), 62)
                self.assertEqual(
                    str(archive["source_energy_definition"]),
                    "last saved energy-trace sample per run",
                )
                self.assertIn("strict inequality", str(archive["success_criterion"]))
                self.assertAlmostEqual(float(archive["ground_energy"]), fixture["smallest_eigval"])
                for layer in (2, 7):
                    np.testing.assert_array_equal(
                        archive[f"L{layer}_final_energies"],
                        fixture[f"L{layer}_energy_traces"][:, -1],
                    )
                self.assertEqual(float(archive["L2_normalized_errors"][0]), 0.0)

    def test_six_cutoffs_use_strict_gap_normalized_final_energy(self):
        # The mock Hamiltonian has E0=0 and gap=2; equality remains exact in
        # binary floating point after energy-gap division.
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
        fixture = _archive_fixture()
        fixture.update(
            smallest_eigval=0.0,
            num_runs=runs,
            L7_energy_traces=np.column_stack((np.full(runs, 2.0), np.zeros(runs), final_energies)),
            L2_energy_traces=np.column_stack((np.full(runs, 2.0), np.zeros(runs), np.ones(runs))),
        )
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "results"
            figures_dir = Path(temporary) / "figures"
            results_dir.mkdir()
            np.savez(results_dir / "vqe_optimization_results.npz", **fixture)
            completed = _run_cli(temporary, _gap_arguments(results_dir, figures_dir), synthetic_gap=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            with np.load(results_dir / "gap_normalized_energy_statistics.npz", allow_pickle=False) as data:
                self.assertEqual(float(data["spectral_gap"]), 2.0)
                np.testing.assert_array_equal(data["thresholds"], _EXPECTED_THRESHOLDS)
                np.testing.assert_array_equal(data["L7_normalized_errors"], final_energies / 2)
                np.testing.assert_array_equal(data["success_probabilities"][0], np.zeros(6))
                np.testing.assert_allclose(data["success_probabilities"][1], np.asarray([16, 13, 10, 7, 4, 1]) / 18)

    def test_loader_uses_archive_counts_and_never_reads_parameter_storage(self):
        from unitary_pqc_measured_1_energy_results import load_measured_unitary_final_energies

        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "energy.npz"
            for steps in (2, 3):
                with self.subTest(steps=steps):
                    fixture = _archive_fixture()
                    fixture["steps"] = steps
                    np.savez(archive_path, **fixture)
                    original = archive_path.read_bytes()
                    energies, ground, metadata = load_measured_unitary_final_energies(
                        archive_path, expected_h_param=0.1,
                    )
                    self.assertEqual(set(energies), {2, 7})
                    self.assertEqual(ground, fixture["smallest_eigval"])
                    self.assertEqual(metadata["num_stored_energy_samples"], 3)
                    self.assertEqual(metadata["num_params_per_layer"], 62)
                    for layer in (2, 7):
                        np.testing.assert_array_equal(
                            energies[layer], fixture[f"L{layer}_energy_traces"][:, -1],
                        )
                    self.assertEqual(archive_path.read_bytes(), original)

    def test_loader_rejects_wrong_model_metadata_counts_shapes_and_final_samples(self):
        from unitary_pqc_measured_1_energy_results import load_measured_unitary_final_energies

        invalid = {
            "ansatz": ["dpqc_reset", ["unitary_pqc_measured_1"]],
            "measurement_outcome": [0, 1.0, True, [1]],
            "num_params_per_layer": [14, 62.0, [62]],
            "h_param": [0.5, np.nan, np.inf, [0.1], 0.1 + 1j],
            "smallest_eigval": [np.nan, np.inf, [0.0], 1j],
            "layers": [[2, 2], [2.0, 7.0], [0, 7], [], [[2, 7]], [True]],
            "num_runs": [0, float(_NUM_RUNS), True, _NUM_RUNS - 1, [_NUM_RUNS]],
            "steps": [0, 3.0, True, 4, [3]],
            "L7_energy_traces": [
                np.zeros((_NUM_RUNS - 1, 3)), np.zeros((_NUM_RUNS, 1)), np.zeros((_NUM_RUNS, 3, 1)),
                np.zeros((_NUM_RUNS, 3), dtype=complex), np.full((_NUM_RUNS, 3), np.nan),
                np.full((_NUM_RUNS, 3), np.inf), np.full((_NUM_RUNS, 3), "invalid"),
                np.zeros((_NUM_RUNS, 3), dtype=bool),
                # Each width is independently valid, but must agree by layer.
                np.zeros((_NUM_RUNS, 4)),
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "energy.npz"
            for key, values in invalid.items():
                for index, value in enumerate(values):
                    fixture = _archive_fixture()
                    fixture[key] = value
                    np.savez(archive_path, **fixture)
                    with self.subTest(key=key, case=index), self.assertRaises(ValueError):
                        load_measured_unitary_final_energies(archive_path, expected_h_param=0.1)
            for key in ("ansatz", "smallest_eigval", "layers", "L7_energy_traces"):
                fixture = _archive_fixture()
                del fixture[key]
                np.savez(archive_path, **fixture)
                with self.subTest(missing=key), self.assertRaises((ValueError, KeyError)):
                    load_measured_unitary_final_energies(archive_path, expected_h_param=0.1)

    def test_invalid_or_missing_archives_fail_before_creating_outputs(self):
        cases = {
            "old_model": ("num_params_per_layer", 14),
            "wrong_outcome": ("measurement_outcome", 0),
            "wrong_ansatz": ("ansatz", "unitary_pqc_measured_0"),
            "wrong_h": ("h_param", 0.5),
            "wrong_ground": ("smallest_eigval", 0.0),
            "wrong_shape": ("L7_energy_traces", np.zeros((_NUM_RUNS - 1, 3))),
            "nonfinite_final": ("L7_energy_traces", np.full((_NUM_RUNS, 3), np.nan)),
            "subground_final": ("L7_energy_traces", np.full((_NUM_RUNS, 3), -100.0)),
            "missing_archive": None,
        }
        with tempfile.TemporaryDirectory() as temporary:
            for name, change in cases.items():
                with self.subTest(case=name):
                    results_dir = Path(temporary) / name / "results"
                    figures_dir = Path(temporary) / name / "figures"
                    results_dir.mkdir(parents=True)
                    archive_path = results_dir / "vqe_optimization_results.npz"
                    if change is not None:
                        fixture = _archive_fixture()
                        fixture[change[0]] = change[1]
                        np.savez(archive_path, **fixture)
                    completed = _run_cli(temporary, _gap_arguments(results_dir, figures_dir))
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertFalse(figures_dir.exists(), completed.stdout + completed.stderr)
                    self.assertFalse((results_dir / "gap_normalized_energy_statistics.npz").exists())

    def test_cli_rejects_incompatible_modes_and_directory_overrides(self):
        cases = (
            ["--gap-normalized-only", "--hessian-only"],
            ["--gap-results-dir", "unused"],
            ["--gap-figures-dir", "unused"],
            ["--gap-normalized-only", "--hessian-results-dir", "unused"],
        )
        with tempfile.TemporaryDirectory() as temporary:
            for arguments in cases:
                with self.subTest(arguments=arguments):
                    completed = _run_cli(temporary, arguments)
                    self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
                    self.assertIn("error:", completed.stderr)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_standard_visualization_also_calls_saved_energy_metrics(self):
        tree = ast.parse(_VISUALIZER.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_unitary_pqc_visualization"
        )
        calls = {
            node.func.id for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("run_unitary_gap_normalized_visualization", calls)


if __name__ == "__main__":
    unittest.main()
