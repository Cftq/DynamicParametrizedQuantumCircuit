"""CLI checks for measured-unitary plots made only from saved Hessians.

Run with ``python -m unittest discover -s src/unitary_pqc -p
test_measured_unitary_hessian_visualization.py``. Files are temporary and
quantum/optimizer imports are blocked in the visualization subprocess.
"""

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


def _archive_fixture(*, include_matrices=True):
    """Schema-1 outcome-1 archive with two analytically known signed spectra."""
    data = {
        "schema_version": np.asarray(1),
        "analysis_kind": np.asarray("random_points"),
        "ansatz": np.asarray("unitary_pqc_measured_1"),
        "measurement_outcome": np.asarray(1),
        "h_param": np.asarray(0.1),
        "layers": np.asarray([3, 1]),
        "num_hessian_samples": np.asarray(2),
        "hessian_sample_seed_base": np.asarray(1234),
        "hessian_rank_threshold": np.asarray(1e-12),
        "hessian_rank_definition": np.asarray(
            "count(abs(eigenvalue) >= hessian_rank_threshold)"
        ),
        "hessian_condition_number_definition": np.asarray(
            "max(abs(active eigenvalue)) / min(abs(active eigenvalue)); NaN if rank == 0"
        ),
        "num_params_per_layer": np.asarray(62),
        "analysis_batch_size": np.asarray(8),
        "hessian_matrix_definition": np.asarray("d2 E(theta) / dtheta_i dtheta_j"),
    }
    for layer in (3, 1):
        dimension = 62 * layer
        data[f"L{layer}_rank"] = np.asarray([2, 2])
        data[f"L{layer}_condition_number"] = np.asarray([1.0, 2.0])
        if include_matrices:
            matrices = np.zeros((2, dimension, dimension))
            matrices[0, :2, :2] = [[0.0, 3.0], [3.0, 0.0]]
            matrices[1, :2, :2] = [[-2.0, 0.0], [0.0, 1.0]]
            data[f"L{layer}_hessian"] = matrices
            data[f"L{layer}_theta"] = np.zeros((2, dimension))
    return data


def _run_hessian_cli(temporary, results_dir, figures_dir):
    # Pin compute-only configuration to conflicting values: rendering must use
    # the archive's sample count and seed even if computation settings change.
    launch = (
        "import runpy, sys; "
        "sys.modules['jax'] = None; "
        "sys.modules['optax'] = None; "
        "sys.modules['tensorcircuit'] = None; "
        "sys.path.insert(0, sys.argv[1]); "
        "import config_overparam as cfg; "
        "cfg.NUM_QFIM_SAMPLES = 999; "
        "cfg.UNITARY_PQC_QFIM_SAMPLE_SEED_BASE = 987654; "
        "sys.argv = sys.argv[2:]; "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [
            sys.executable, "-c", launch, str(_COMMON_DIR), str(_VISUALIZER),
            "--h-param", "0.1", "--hessian-only",
            "--hessian-results-dir", str(results_dir),
            "--hessian-figures-dir", str(figures_dir),
        ],
        cwd=temporary, env=environment, capture_output=True,
        text=True, encoding="utf-8", timeout=60, check=False,
    )


class MeasuredUnitaryHessianVisualizationTests(unittest.TestCase):
    def test_saved_matrices_render_statistics_without_quantum_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "saved Hessians"
            figures_dir = Path(temporary) / "Hessian figures"
            results_dir.mkdir()
            archive_path = results_dir / "hessian_random_points.npz"
            np.savez_compressed(archive_path, **_archive_fixture())
            original_archive = archive_path.read_bytes()

            completed = _run_hessian_cli(temporary, results_dir, figures_dir)
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            self.assertEqual(archive_path.read_bytes(), original_archive)
            # Two rank/condition summaries, six QFIM-style summaries, four
            # normalized-curvature summaries, and two signed/absolute pairs.
            pdf_paths = list(figures_dir.rglob("*.pdf"))
            self.assertEqual(len(pdf_paths), 16)
            for path in pdf_paths:
                self.assertTrue(path.read_bytes().startswith(b"%PDF-"), str(path))
            for kind in ("absolute", "signed"):
                self.assertEqual(
                    {path.name for path in (figures_dir / "hessian_eigs" / kind).glob("*.pdf")},
                    {f"L1_{kind}.pdf", f"L3_{kind}.pdf"},
                )
            with np.load(
                figures_dir / "hessian_statistics_random_points.npz",
                allow_pickle=False,
            ) as archive:
                np.testing.assert_array_equal(archive["layers"], [1, 3])
                self.assertEqual(int(archive["num_hessian_samples"]), 2)
                self.assertEqual(float(archive["rank_threshold"]), 1e-12)
                self.assertEqual(str(archive["output_family"]), "unitary_pqc_measured_1")
                for layer in (1, 3):
                    expected = {
                        "rank": [2, 2],
                        "trace": [0.0, -1.0],
                        "absolute_trace": [6.0, 3.0],
                        "abs_entry_sum": [6.0, 3.0],
                        "participation_rank": [2.0, 1.8],
                        "shannon_entropy": [np.log(2), np.log(3) - 2 * np.log(2) / 3],
                    }
                    for key, values in expected.items():
                        with self.subTest(layer=layer, metric=key):
                            np.testing.assert_allclose(archive[f"L{layer}_{key}"], values)
                np.testing.assert_allclose(archive["absolute_trace_mean"], [4.5, 4.5])
                np.testing.assert_allclose(archive["absolute_trace_sem"], [1.5, 1.5])
                np.testing.assert_allclose(archive["trace_mean"], [-0.5, -0.5])

    def test_legacy_summaries_render_two_figures_and_explain_missing_spectra(self):
        with tempfile.TemporaryDirectory() as temporary:
            results_dir = Path(temporary) / "saved"
            figures_dir = Path(temporary) / "figures"
            results_dir.mkdir()
            archive_path = results_dir / "hessian_random_points.npz"
            np.savez_compressed(
                archive_path, **_archive_fixture(include_matrices=False)
            )
            original_archive = archive_path.read_bytes()

            completed = _run_hessian_cli(temporary, results_dir, figures_dir)
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            self.assertEqual(archive_path.read_bytes(), original_archive)
            self.assertIn("only rank/condition summaries", completed.stderr)
            self.assertIn("require raw matrices", completed.stderr)
            self.assertEqual(
                {path.name for path in figures_dir.rglob("*.pdf")},
                {"hessian_rank_random_points.pdf", "hessian_condition_number_random_points.pdf"},
            )
            self.assertFalse(
                (figures_dir / "hessian_statistics_random_points.npz").exists()
            )


if __name__ == "__main__":
    unittest.main()
