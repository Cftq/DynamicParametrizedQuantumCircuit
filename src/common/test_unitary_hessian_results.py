"""Analytic coverage for the NumPy-only measured unitary Hessian loader."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    from . import unitary_hessian_results as loader
except ImportError:
    import unitary_hessian_results as loader


def archive_fixture(layers=(1,), *, matrices=True):
    arrays = {
        "schema_version": 1,
        "analysis_kind": "random_points",
        "ansatz": "unitary_pqc_measured_1",
        "measurement_outcome": 1,
        "h_param": 0.1,
        "layers": np.asarray(layers),
        "num_hessian_samples": 3,
        "hessian_sample_seed_base": 123,
        "hessian_rank_threshold": 1e-12,
        "hessian_rank_definition": "count(abs(eigenvalue) >= hessian_rank_threshold)",
        "hessian_condition_number_definition": (
            "max(abs(active eigenvalue)) / min(abs(active eigenvalue)); NaN if rank == 0"
        ),
        "num_params_per_layer": 14,
        "analysis_batch_size": 2,
    }
    for layer in layers:
        arrays[f"L{layer}_rank"] = np.asarray([2, 2, 0])
        arrays[f"L{layer}_condition_number"] = np.asarray([1.5, 3.0, np.nan])
        if matrices:
            values = np.zeros((3, 14 * layer, 14 * layer))
            values[0, 0, 0], values[0, 1, 1] = -3.0, 2.0
            values[1, :2, :2] = [[1.0, 2.0], [2.0, 1.0]]
            arrays[f"L{layer}_hessian"] = values
            arrays[f"L{layer}_theta"] = np.zeros((3, 14 * layer))
    return arrays


class MeasuredUnitaryHessianResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def load(self, arrays=None, **kwargs):
        if arrays is not None:
            np.savez(self.directory / loader.RESULT_NAME, **arrays)
        return loader.load_measured_unitary_hessian_result(
            self.directory, expected_h_param=0.1, **kwargs,
        )

    def test_signed_offdiagonal_and_zero_spectra(self):
        actual = self.load(archive_fixture())
        self.assertEqual(actual["output_family"], "unitary_pqc_measured_1")
        self.assertEqual(actual["layers"], [1])
        self.assertEqual(actual["num_samples"], 3)
        self.assertEqual(actual["seed_base"], 123)
        np.testing.assert_array_equal(actual["rank_by_layer"][1], [2, 2, 0])
        np.testing.assert_allclose(actual["condition_by_layer"][1], [1.5, 3, np.nan])
        np.testing.assert_allclose(actual["eigenvalues_by_layer"][1][:, 0], [-3, -1, 0])
        np.testing.assert_allclose(actual["eigenvalues_by_layer"][1][:, -1], [2, 3, 0])

    def test_raw_matrices_allow_changed_inclusive_absolute_threshold(self):
        actual = self.load(archive_fixture(), rank_threshold=3.0)
        self.assertEqual(actual["threshold"], 3.0)
        np.testing.assert_array_equal(actual["rank_by_layer"][1], [1, 1, 0])
        np.testing.assert_allclose(actual["condition_by_layer"][1], [1, 1, np.nan])
        actual = self.load(rank_threshold=4.0)
        np.testing.assert_array_equal(actual["rank_by_layer"][1], [0, 0, 0])
        self.assertTrue(np.isnan(actual["condition_by_layer"][1]).all())

    def test_each_layer_is_diagonalized_only_once_and_layers_are_sorted(self):
        with patch.object(loader.np.linalg, "eigvalsh", wraps=np.linalg.eigvalsh) as diagonalize:
            actual = self.load(archive_fixture((2, 1)))
        self.assertEqual(diagonalize.call_count, 2)
        self.assertEqual(actual["layers"], [1, 2])
        self.assertEqual(list(actual["eigenvalues_by_layer"]), [1, 2])

    def test_legacy_summaries_are_preserved_without_diagonalization(self):
        with patch.object(loader.np.linalg, "eigvalsh") as diagonalize:
            actual = self.load(archive_fixture(matrices=False))
        diagonalize.assert_not_called()
        self.assertEqual(actual["hessian_by_layer"], {})
        self.assertEqual(actual["eigenvalues_by_layer"], {})
        np.testing.assert_array_equal(actual["rank_by_layer"][1], [2, 2, 0])
        np.testing.assert_allclose(actual["condition_by_layer"][1], [1.5, 3, np.nan])
        with self.assertRaisesRegex(ValueError, "Legacy Hessian"):
            self.load(rank_threshold=1e-8)

    def test_optional_qfim_metadata_must_match(self):
        actual = self.load(
            archive_fixture((2, 1)), expected_layers=[1, 2],
            expected_num_samples=3, expected_seed_base=123,
        )
        self.assertEqual(actual["layers"], [1, 2])
        for kwargs in (
            {"expected_layers": [1]}, {"expected_layers": [1, 3]},
            {"expected_num_samples": 4}, {"expected_seed_base": 124},
            {"expected_num_samples": 3.5}, {"expected_seed_base": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.load(**kwargs)

    def test_wrong_model_outcome_h_and_archive_definitions_are_rejected(self):
        invalid = {
            "ansatz": ["unitary_pqc", "unitary_pqc_measured_0", "dpqc"],
            "measurement_outcome": [0, 2, 1.0, True],
            "num_params_per_layer": [12, 14.0, 14.5],
            "h_param": [0.2, np.nan, np.inf, "0.1", 0.1 + 0j],
            "schema_version": [0, 2, 1.5],
            "analysis_kind": ["optimized_points"],
            "hessian_rank_definition": ["count(eigenvalue >= threshold)"],
            "hessian_condition_number_definition": ["max/min"],
            "output_family": ["dpqc_reset", "unitary_pqc"],
        }
        for key, values in invalid.items():
            for value in values:
                arrays = archive_fixture()
                arrays[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.load(arrays)

    def test_required_metadata_must_be_present_and_scalar(self):
        for key in loader._REQUIRED_METADATA:
            arrays = archive_fixture()
            del arrays[key]
            with self.subTest(missing=key), self.assertRaises(KeyError):
                self.load(arrays)
            if key != "layers":
                arrays = archive_fixture()
                arrays[key] = np.asarray([arrays[key]])
                with self.subTest(nonscalar=key), self.assertRaises(ValueError):
                    self.load(arrays)

    def test_invalid_sampling_and_threshold_metadata_are_rejected(self):
        invalid = {
            "num_hessian_samples": [0, -1, 3.2, np.nan, "3", True],
            "analysis_batch_size": [0, -1, 1.5, np.inf, "2", True],
            "hessian_sample_seed_base": [-1, 0.5, np.inf, "123", False],
            "hessian_rank_threshold": [0, -1, np.nan, np.inf, "1e-12", 1j],
        }
        for key, values in invalid.items():
            for value in values:
                arrays = archive_fixture()
                arrays[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.load(arrays)

    def test_layers_must_be_unique_positive_integers(self):
        for values in ([], [0], [-1], [1, 1], [1.5], [np.nan], [np.inf], [[1]], [1j], ["1"], [True]):
            arrays = archive_fixture()
            arrays["layers"] = values
            with self.subTest(layers=values), self.assertRaises(ValueError):
                self.load(arrays)

    def test_invalid_requested_threshold_is_rejected(self):
        for value in (0, -1, np.nan, np.inf, [1.0], "1.0", 1j, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load(archive_fixture(), rank_threshold=value)

    def test_saved_summaries_are_validated_even_when_recomputed(self):
        invalid = {
            "L1_rank": [[2, 2], [15, 2, 0], [-1, 2, 0], [1.5, 2, 0], [np.nan, 2, 0], [2j, 2, 0]],
            "L1_condition_number": [[1.5, 3], [0.5, 3, np.nan], [np.inf, 3, np.nan],
                                     [np.nan, 3, np.nan], [1.5, 3, 1], [1.5j, 3, np.nan],
                                     ["1.5", "3", "nan"]],
        }
        for key, values in invalid.items():
            for value in values:
                arrays = archive_fixture()
                arrays[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.load(arrays, rank_threshold=4.0)
        for key in ("L1_rank", "L1_condition_number"):
            arrays = archive_fixture()
            del arrays[key]
            with self.subTest(missing=key), self.assertRaises(KeyError):
                self.load(arrays)

    def test_partial_and_malformed_raw_matrices_are_rejected(self):
        arrays = archive_fixture((1, 2))
        del arrays["L2_hessian"]
        with self.assertRaisesRegex(KeyError, "raw matrices"):
            self.load(arrays)
        for values in (
            np.zeros((2, 14, 14)), np.zeros((3, 13, 13)),
            np.zeros((3, 14, 14), dtype=complex), np.full((3, 14, 14), np.nan),
        ):
            arrays = archive_fixture()
            arrays["L1_hessian"] = values
            with self.subTest(shape=values.shape, dtype=values.dtype), self.assertRaises(ValueError):
                self.load(arrays)
        arrays = archive_fixture()
        arrays["L1_hessian"][0, 0, 1] = 1.0
        with self.assertRaisesRegex(ValueError, "symmetric"):
            self.load(arrays)

    def test_optional_saved_parameter_points_are_validated(self):
        for values in (np.zeros((3, 13)), np.full((3, 14), np.inf), np.zeros((3, 14), dtype=complex)):
            arrays = archive_fixture()
            arrays["L1_theta"] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.load(arrays)
        arrays = archive_fixture()
        del arrays["L1_theta"]
        self.assertIn(1, self.load(arrays)["hessian_by_layer"])

    def test_missing_exact_path_does_not_fall_back_to_other_family(self):
        self.load(archive_fixture())
        missing = self.directory / "other_family"
        with self.assertRaises(FileNotFoundError) as error:
            loader.load_measured_unitary_hessian_result(missing, expected_h_param=0.1)
        self.assertIn(str(missing / loader.RESULT_NAME), str(error.exception))
        self.assertIn(
            "unitary_pqc_measured_1_overparam_hessian.py --h-param 0.1",
            str(error.exception),
        )

    def test_import_does_not_load_quantum_or_plotting_runtimes(self):
        script = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import unitary_hessian_results; "
            "assert not ({'jax', 'optax', 'tensorcircuit', 'matplotlib'} & set(sys.modules))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).parent)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
