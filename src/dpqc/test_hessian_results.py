"""NumPy-only regression tests for saved Hessian analysis.

Run with ``python -m unittest discover -s src/dpqc -p test_hessian_results.py``.
Fixtures live in temporary directories; importing this module does not import
the plotting entry point or initialize JAX.
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "common" / "hessian_results.py"
_SPEC = importlib.util.spec_from_file_location("hessian_results", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
load_random_hessian_result = _MODULE.load_random_hessian_result


class HessianResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.results_dir = Path(self.temp_dir.name)

    @staticmethod
    def matrix_fixture(*, layers=(1,), num_samples=2):
        data = {
            "schema_version": np.asarray(2),
            "h_param": np.asarray(0.1),
            "layers": np.asarray(layers, dtype=np.int64),
            "num_hessian_samples": np.asarray(num_samples),
            "hessian_sample_seed_base": np.asarray(1234),
            "parameter_distribution": np.asarray(
                "jax.random.uniform[-pi, pi), float64, "
                "PRNGKey(hessian_sample_seed_base + layer)"
            ),
            "parameters_per_layer": np.asarray(12),
            "hessian_method": np.asarray("chunked_forward_over_reverse_hvp"),
            "hvp_chunk_size": np.asarray(8),
            "output_family": np.asarray("dpqc_reset"),
            "model_id": np.asarray("dpqc_reset_fixed_rx_pi"),
        }
        for layer in layers:
            size = 12 * layer
            data[f"L{layer}_hessian"] = np.zeros(
                (num_samples, size, size), dtype=np.float64
            )
            data[f"L{layer}_theta"] = np.zeros(
                (num_samples, size), dtype=np.float64
            )
        return data

    def save(self, data):
        np.savez_compressed(self.results_dir / "hessian_random_points.npz", **data)

    def load(self, **kwargs):
        arguments = {
            "expected_h_param": 0.1,
            "expected_output_family": "dpqc_reset",
        }
        arguments.update(kwargs)
        return load_random_hessian_result(self.results_dir, **arguments)

    def test_signed_spectrum_and_inclusive_absolute_threshold(self):
        data = self.matrix_fixture()
        spectrum = np.asarray([-4.0, 2.0, 1e-12, -5e-13] + [0.0] * 8)
        data["L1_hessian"][0] = np.diag(spectrum)
        # A non-diagonal indefinite block catches accidental diagonal-only use.
        data["L1_hessian"][1, :2, :2] = [[0.0, 3.0], [3.0, 0.0]]
        self.save(data)

        result = self.load()

        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["num_samples"], 2)
        self.assertEqual(result["threshold"], 1e-12)
        np.testing.assert_array_equal(result["rank_by_layer"][1], [3, 2])
        np.testing.assert_allclose(result["condition_by_layer"][1], [4e12, 1.0])
        np.testing.assert_allclose(result["eigenvalues_by_layer"][1][0], np.sort(spectrum))
        np.testing.assert_allclose(
            result["eigenvalues_by_layer"][1][1], [-3.0] + [0.0] * 10 + [3.0]
        )

    def test_same_saved_matrices_support_different_analysis_thresholds(self):
        data = self.matrix_fixture(num_samples=1)
        data["L1_hessian"][0] = np.diag([-8.0, 2.0, 0.5] + [0.0] * 9)
        self.save(data)

        low = self.load(rank_threshold=0.5)
        high = self.load(rank_threshold=2.0)

        np.testing.assert_array_equal(low["rank_by_layer"][1], [3])
        np.testing.assert_allclose(low["condition_by_layer"][1], [16.0])
        np.testing.assert_array_equal(high["rank_by_layer"][1], [2])
        np.testing.assert_allclose(high["condition_by_layer"][1], [4.0])
        np.testing.assert_array_equal(
            low["eigenvalues_by_layer"][1], high["eigenvalues_by_layer"][1]
        )

    def test_zero_and_entirely_inactive_spectra(self):
        data = self.matrix_fixture()
        data["L1_hessian"][1] = np.diag([-5e-13, 5e-13] + [0.0] * 10)
        self.save(data)

        result = self.load()

        np.testing.assert_array_equal(result["rank_by_layer"][1], [0, 0])
        self.assertTrue(np.isnan(result["condition_by_layer"][1]).all())

    def test_legacy_summaries_load_but_threshold_changes_require_recompute(self):
        data = self.matrix_fixture()
        del data["L1_hessian"]
        del data["L1_theta"]
        data.update(
            schema_version=np.asarray(1),
            hessian_rank_threshold=np.asarray(1e-12),
            hessian_rank_definition=np.asarray("count(abs(eigenvalues) >= threshold)"),
            hessian_condition_number_definition=np.asarray(
                "max(abs(active_eigenvalues)) / min(abs(active_eigenvalues))"
            ),
            L1_rank=np.asarray([2, 0]),
            L1_condition_number=np.asarray([4.0, np.nan]),
        )
        self.save(data)

        result = self.load()

        self.assertEqual(result["schema_version"], 1)
        np.testing.assert_array_equal(result["rank_by_layer"][1], [2, 0])
        np.testing.assert_allclose(result["condition_by_layer"][1], [4.0, np.nan])
        with self.assertRaisesRegex(ValueError, r"[Rr]ecompute|--hessian-only"):
            self.load(rank_threshold=1e-6)

    def test_reject_invalid_hessian_arrays(self):
        invalid_arrays = {
            "wrong shape": np.zeros((2, 11, 11), dtype=np.float64),
            "nonfinite": np.full((2, 12, 12), np.nan, dtype=np.float64),
            "complex": np.zeros((2, 12, 12), dtype=np.complex128),
        }
        asymmetric = np.zeros((2, 12, 12), dtype=np.float64)
        asymmetric[0, 0, 1] = 1.0
        invalid_arrays["asymmetric"] = asymmetric
        for label, invalid in invalid_arrays.items():
            with self.subTest(label=label):
                data = self.matrix_fixture()
                data["L1_hessian"] = invalid
                self.save(data)
                with self.assertRaises(ValueError):
                    self.load()

    def test_reject_invalid_saved_parameters(self):
        invalid_arrays = (
            np.zeros((2, 11), dtype=np.float64),
            np.full((2, 12), np.inf, dtype=np.float64),
            np.zeros((2, 12), dtype=np.complex128),
        )
        for invalid in invalid_arrays:
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                data = self.matrix_fixture()
                data["L1_theta"] = invalid
                self.save(data)
                with self.assertRaises(ValueError):
                    self.load()

    def test_reject_incorrect_model_and_hamiltonian(self):
        self.save(self.matrix_fixture())
        with self.assertRaises(ValueError):
            self.load(expected_h_param=0.2)
        with self.assertRaises(ValueError):
            self.load(expected_output_family="dpqc")
        data = self.matrix_fixture()
        data["model_id"] = np.asarray("dpqc_dynamic_channel")
        self.save(data)
        with self.assertRaises(ValueError):
            self.load()

    def test_requested_layers_are_selected_and_sorted_for_plotting(self):
        self.save(self.matrix_fixture(layers=(3, 1, 2)))

        result = self.load(requested_layers=[3, 1])

        self.assertEqual(list(result["layers"]), [1, 3])
        self.assertEqual(set(result["rank_by_layer"]), {1, 3})
        self.assertEqual(result["eigenvalues_by_layer"][3].shape, (2, 36))
        with self.assertRaises(KeyError):
            self.load(requested_layers=[4])


if __name__ == "__main__":
    unittest.main()
