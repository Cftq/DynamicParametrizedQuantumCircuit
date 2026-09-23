"""Saved-spectrum log-determinant checks; no QFIM/training computations."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.qfim_logdet import (
    DEFAULT_KAPPA,
    compute_qfim_logdet,
    load_random_qfim_spectra,
    save_qfim_logdet_outputs,
)


class QfimLogdetTests(unittest.TestCase):
    def test_agrees_with_independent_matrix_determinants(self):
        matrices = [
            np.asarray([[2.0, 1.0], [1.0, 2.0]]),
            np.asarray([[4.0, 0.0], [0.0, 0.0]]),
        ]
        spectra = np.asarray([np.linalg.eigvalsh(matrix) for matrix in matrices])
        expected = [np.linalg.slogdet(np.eye(2) + 0.3 * matrix)[1] for matrix in matrices]
        actual = compute_qfim_logdet({2: spectra}, kappa=0.3)
        np.testing.assert_allclose(actual["logdet_by_layer"][2], expected)
        self.assertAlmostEqual(actual["mean"][0], np.mean(expected))
        self.assertAlmostEqual(actual["std"][0], np.std(expected, ddof=1))
        self.assertAlmostEqual(actual["sem"][0], np.std(expected, ddof=1) / np.sqrt(2))

    def test_mean_is_taken_after_logdet_not_before(self):
        actual = compute_qfim_logdet({1: [[0.0], [8.0]]})
        self.assertAlmostEqual(actual["mean"][0], np.log(3.0))
        self.assertNotAlmostEqual(actual["mean"][0], np.log1p(4.0))

    def test_all_positive_values_contribute_without_rank_threshold(self):
        actual = compute_qfim_logdet({1: [[1e-15, 0.0]]}, kappa=1e15)
        self.assertAlmostEqual(actual["mean"][0], np.log(2.0))
        tiny = compute_qfim_logdet({1: [[1e-20]]})
        self.assertEqual(tiny["mean"][0], np.log1p(1e-20))

    def test_large_product_remains_finite(self):
        actual = compute_qfim_logdet({1: [[1e308, 1e-308, 0.0]]}, kappa=1e308)
        expected = 2 * np.log(1e308) + np.log(2.0)
        self.assertTrue(np.isfinite(actual["mean"][0]))
        self.assertAlmostEqual(actual["mean"][0], expected, places=11)

    def test_layer_order_ragged_dimensions_and_sample_counts(self):
        actual = compute_qfim_logdet({3: [[1.0] * 6], 1: [[0.0, 0.0], [1.0, 1.0]]})
        np.testing.assert_array_equal(actual["layers"], [1, 3])
        np.testing.assert_array_equal(actual["num_parameters_by_layer"], [2, 6])
        np.testing.assert_array_equal(actual["num_samples_by_layer"], [2, 1])
        np.testing.assert_allclose(actual["mean"], [np.log(2.0), 6 * np.log(2.0)])
        np.testing.assert_allclose(actual["sem"], [np.log(2.0), 0.0])
        self.assertEqual(actual["kappa"], DEFAULT_KAPPA)

    def test_negative_roundoff_clipping_is_sample_scaled_and_nonmutating(self):
        original = np.asarray([[1.0, -1e-15], [1e-20, -1e-35]])
        copy = original.copy()
        actual = compute_qfim_logdet({1: original})
        np.testing.assert_array_equal(original, copy)
        np.testing.assert_allclose(actual["logdet_by_layer"][1], np.log1p([1.0, 1e-20]))
        np.testing.assert_array_equal(actual["num_clipped_negative_by_layer"], [2])
        tolerance = actual["negative_roundoff_tolerance_by_layer"][1]
        self.assertAlmostEqual(tolerance[1] / tolerance[0], 1e-20)
        for spectra in ([[1.0, -1e-6]], [[-1e-30]], [[1e-20, -1e-22]]):
            with self.subTest(spectra=spectra), self.assertRaisesRegex(ValueError, "negative"):
                compute_qfim_logdet({1: spectra})

    def test_invalid_inputs_are_rejected(self):
        for kappa in (0.0, -1.0, np.nan, np.inf, 1j, True, [1.0], "1"):
            with self.subTest(kappa=kappa), self.assertRaises(ValueError):
                compute_qfim_logdet({1: [[1.0]]}, kappa=kappa)
        for spectra in ([], [1.0], [[np.nan]], [[np.inf]], [[1j]], [["1"]], np.empty((0, 2))):
            with self.subTest(spectra=spectra), self.assertRaises(ValueError):
                compute_qfim_logdet({1: spectra})
        for value in ({}, {0: [[1.0]]}, {1.0: [[1.0]]}, {True: [[1.0]]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compute_qfim_logdet(value)


class SavedQfimReaderTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = Path(self.tempdir.name) / "qfim_random_points_keep0123.npz"

    def write_archive(self, **overrides):
        arrays = {
            "h_param": 0.1,
            "layers": [3, 1],
            "num_qfim_samples": 2,
            "qfim_effective_rank_threshold": 1e-12,
            "L1_eigs_desc": np.asarray([[1.0, 0.0], [0.5, 1e-15]]),
            "L3_eigs_desc": np.asarray([[1.0] * 6, [0.0] * 6]),
            # Loading this object array under allow_pickle=False would fail.
            "L1_theta": np.asarray([{"must_not_be_loaded": True}], dtype=object),
        }
        arrays.update(overrides)
        np.savez(self.path, **arrays)

    def test_legacy_schema_loads_spectra_without_touching_theta(self):
        self.write_archive()
        actual = load_random_qfim_spectra(self.path, expected_h_param=0.1, parameters_per_layer=2)
        self.assertEqual(list(actual["eigenvalues_by_layer"]), [1, 3])
        self.assertEqual(actual["eigenvalues_by_layer"][1][1, 1], 1e-15)
        self.assertNotIn("L1_theta", actual["metadata"])
        self.assertEqual(actual["source_path"], self.path)

    def test_canonical_required_metadata_handles_arrays_and_scalar_types(self):
        expected = {
            "analysis_kind": "random_points", "num_params_per_layer": 2,
            "eigenvalues_threshold_masked": False, "measurement_outcome": 1,
            "eigenvalue_order": "descending", "keep_wires": [0, 1, 2, 3],
            "h_param": 0.1,
        }
        self.write_archive(**expected)
        actual = load_random_qfim_spectra(self.path, required_metadata=expected)
        np.testing.assert_array_equal(actual["metadata"]["keep_wires"], [0, 1, 2, 3])
        for override in ({"measurement_outcome": 0}, {"keep_wires": [0, 1, 2]}, {"num_params_per_layer": 14}):
            self.write_archive(**{**expected, **override})
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "required metadata"):
                load_random_qfim_spectra(self.path, required_metadata=expected)

    def test_malformed_archives_are_rejected(self):
        overrides = (
            {"h_param": 0.2}, {"h_param": np.nan},
            {"layers": [1, 1]}, {"layers": [0, 1]}, {"layers": [1.0, 3.0]},
            {"layers": [1, 2]}, {"num_qfim_samples": 3}, {"num_qfim_samples": 0},
            {"num_params_per_layer": 3}, {"L1_eigs_desc": np.ones((2, 3))},
            {"L1_eigs_desc": np.ones((1, 2))}, {"L1_eigs_desc": [[np.nan, 0], [0, 0]]},
            {"analysis_kind": "optimization_path"}, {"eigenvalues_threshold_masked": True},
        )
        for override in overrides:
            self.write_archive(**override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_random_qfim_spectra(self.path, expected_h_param=0.1, parameters_per_layer=2)

    def test_missing_required_metadata_is_rejected(self):
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "required metadata"):
            load_random_qfim_spectra(self.path, required_metadata={"ansatz": "measured_1"})


class LogdetOutputTests(unittest.TestCase):
    def test_statistics_roundtrip_provenance_and_unique_kappa_filename(self):
        result = compute_qfim_logdet({1: [[0.0, 1.0], [1.0, 2.0]], 3: [[0.0] * 6]})
        with tempfile.TemporaryDirectory() as directory:
            paths = save_qfim_logdet_outputs(
                result, directory, keep_key="keep0123", source_path="saved.npz",
                metadata={"layers": [1, 3], "qfim_sample_seed_base": 9},
            )
            self.assertEqual(paths["figure_path"].name, "qfim_logdet_random_points_keep0123_kappa_1.pdf")
            self.assertTrue(paths["figure_path"].is_file())
            with np.load(paths["statistics_path"], allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["layers"], [1, 3])
                np.testing.assert_allclose(archive["L1_logdet"], result["logdet_by_layer"][1])
                np.testing.assert_array_equal(archive["source_layers"], [1, 3])
                self.assertEqual(archive["source_qfim_sample_seed_base"].item(), 9)
                self.assertEqual(archive["source_path"].item(), "saved.npz")
                self.assertEqual(archive["log_base"].item(), "e")
                self.assertEqual(archive["dimension_normalization"].item(), "none")
            other = save_qfim_logdet_outputs(
                compute_qfim_logdet({1: [[1.0]]}, kappa=np.nextafter(1.0, 2.0)),
                directory, keep_key="keep0123",
            )
            self.assertNotEqual(paths["figure_path"], other["figure_path"])


if __name__ == "__main__":
    unittest.main()
