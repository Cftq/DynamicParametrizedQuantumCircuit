"""Regression tests for curvature diagnostics derived from saved Hessians.

Run with ``python -m unittest discover -s src/common -p test_hessian_curvature.py``.
The analytic fixtures do not initialize the quantum simulator or use JAX.
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


_MODULE_PATH = Path(__file__).with_name("hessian_curvature.py")
_SPEC = importlib.util.spec_from_file_location("hessian_curvature", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
curvature_metrics = _MODULE.curvature_metrics
hamiltonian_energy_width = _MODULE.hamiltonian_energy_width

METRIC_KEYS = (
    "diagonal_square_sum",
    "curvature_square_sum",
    "curvature_effective_rank",
    "negative_curvature_fraction",
)


def independent_hamiltonian(h_param):
    """Build the experiment's Hamiltonian directly in the bit basis."""
    result = np.zeros((16, 16), dtype=np.float64)
    for basis_index in range(16):
        z_values = [1 - 2 * ((basis_index >> (3 - wire)) & 1) for wire in range(4)]
        bonds = sum(z_values[i] * z_values[j] for i, j in ((0, 1), (0, 2), (1, 3), (2, 3)))
        result[basis_index, basis_index] = -(1 - h_param) * bonds - h_param * sum(z_values)
        result[basis_index, basis_index ^ 15] = -(1 - h_param)
    return result


class HessianCurvatureTests(unittest.TestCase):
    def test_analytic_indefinite_and_offdiagonal_hessians(self):
        # The second Hessian has spectrum (-1, 0, 3): off-diagonal
        # entries contribute to S but must not contribute to D.
        matrices = np.asarray([
            np.diag([-3.0, 2.0, 0.0]),
            [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ])
        actual = curvature_metrics(matrices, 2.0)
        expected = {
            "diagonal_square_sum": [13.0 / 4, 2.0 / 4],
            "curvature_square_sum": [13.0 / 4, 10.0 / 4],
            "curvature_effective_rank": [169.0 / 97, 50.0 / 41],
            "negative_curvature_fraction": [9.0 / 13, 1.0 / 10],
        }
        for key in METRIC_KEYS:
            with self.subTest(metric=key):
                self.assertEqual(np.asarray(actual[key]).shape, (2,))
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-13)

    def test_zero_matrices_have_nan_ratio_metrics(self):
        actual = curvature_metrics(np.zeros((2, 3, 3)), 5.0)
        for key in METRIC_KEYS[:2]:
            np.testing.assert_array_equal(actual[key], [0.0, 0.0])
        for key in METRIC_KEYS[2:]:
            self.assertTrue(np.isnan(actual[key]).all(), key)

    def test_small_nonzero_curvature_is_not_thresholded(self):
        # Squaring these eigenvalues is safe, but directly computing
        # their fourth powers underflows. The dimensionless rank is 25/17.
        tiny_matrix = 1e-100 * np.diag([-2.0, 1.0, 0.0])
        actual = curvature_metrics(tiny_matrix[None, :, :], 2.0)
        for key in METRIC_KEYS[:2]:
            np.testing.assert_allclose(actual[key], [1.25e-200], rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(actual["curvature_effective_rank"], [25.0 / 17])
        np.testing.assert_allclose(actual["negative_curvature_fraction"], [0.8])

    def test_large_curvature_does_not_overflow_dimensionless_ratios(self):
        matrix = 1e100 * np.diag([-2.0, 1.0])
        actual = curvature_metrics(matrix[None, :, :], 2.0)
        for key in METRIC_KEYS[:2]:
            np.testing.assert_allclose(actual[key], [1.25e200], rtol=1e-13)
        np.testing.assert_allclose(actual["curvature_effective_rank"], [25.0 / 17])
        np.testing.assert_allclose(actual["negative_curvature_fraction"], [0.8])

    def test_joint_hessian_and_energy_width_scaling_is_invariant(self):
        matrices = np.asarray([[[1.0, 2.0], [2.0, 1.0]]])
        reference = curvature_metrics(matrices, 7.0)
        for scale in (1e-100, 3.0, 1e100):
            actual = curvature_metrics(scale * matrices, scale * 7.0)
            for key in METRIC_KEYS:
                with self.subTest(scale=scale, metric=key):
                    np.testing.assert_allclose(actual[key], reference[key], rtol=1e-13)

    def test_hessian_and_width_rescaling_have_expected_effect(self):
        matrices = np.asarray([np.diag([-2.0, 1.0, 0.0])])
        reference = curvature_metrics(matrices, 2.0)
        for hessian_scale, width_scale in ((3.0, 1.0), (1.0, 4.0)):
            actual = curvature_metrics(hessian_scale * matrices, width_scale * 2.0)
            for key in METRIC_KEYS:
                factor = (hessian_scale / width_scale) ** 2 if key in METRIC_KEYS[:2] else 1.0
                with self.subTest(hessian_scale=hessian_scale, width_scale=width_scale, metric=key):
                    np.testing.assert_allclose(actual[key], factor * reference[key])

    def test_sign_reversal_complements_negative_curvature_fraction(self):
        matrices = np.asarray([np.diag([-2.0, 1.0, 0.0])])
        positive = curvature_metrics(matrices, 1.0)
        negative = curvature_metrics(-matrices, 1.0)
        for key in METRIC_KEYS[:3]:
            np.testing.assert_allclose(positive[key], negative[key])
        np.testing.assert_allclose(
            positive["negative_curvature_fraction"] + negative["negative_curvature_fraction"],
            [1.0],
        )

    def test_equal_magnitude_spectrum_and_definite_curvatures(self):
        matrices = np.asarray([
            np.diag([2.0, -2.0, 2.0]),
            np.diag([2.0, 1.0, 0.5]),
            np.diag([-2.0, -1.0, -0.5]),
            np.diag([0.0, -2.0, 0.0]),
        ])
        actual = curvature_metrics(matrices, 1.0)
        np.testing.assert_allclose(actual["curvature_effective_rank"][[0, 3]], [3.0, 1.0])
        np.testing.assert_allclose(actual["negative_curvature_fraction"], [1.0 / 3, 0.0, 1.0, 1.0])

    def test_metric_bounds_under_change_of_parameter_basis(self):
        rng = np.random.default_rng(842)
        orthogonal, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        matrix = orthogonal @ np.diag([-3.0, 2.0, 1.0, 0.0, 0.0, 0.0]) @ orthogonal.T
        actual = curvature_metrics(matrix[None, :, :], 9.0)
        diagonal, square_sum, rank, negative = (float(actual[key][0]) for key in METRIC_KEYS)
        self.assertGreaterEqual(diagonal, 0.0)
        self.assertLessEqual(diagonal, square_sum + 1e-14)
        self.assertGreaterEqual(rank, 1.0 - 1e-14)
        self.assertLessEqual(rank, 3.0 + 1e-14)
        self.assertGreaterEqual(negative, 0.0)
        self.assertLessEqual(negative, 1.0)
        np.testing.assert_allclose([square_sum, rank, negative], [14.0 / 81, 2.0, 9.0 / 14])

    def test_invalid_hessians_are_rejected(self):
        invalid = {
            "missing sample axis": np.eye(2),
            "rectangular": np.zeros((1, 2, 3)),
            "nonfinite nan": np.full((1, 2, 2), np.nan),
            "nonfinite infinity": np.full((1, 2, 2), np.inf),
            "complex": np.eye(2, dtype=complex)[None, :, :],
            "asymmetric": np.asarray([[[1.0, 2.0], [0.0, 1.0]]]),
        }
        for label, matrices in invalid.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                curvature_metrics(matrices, 1.0)

    def test_invalid_energy_widths_are_rejected(self):
        for width in (0.0, -1.0, np.nan, np.inf, -np.inf):
            with self.subTest(width=width), self.assertRaises(ValueError):
                curvature_metrics(np.eye(2)[None, :, :], width)

    def test_energy_width_matches_independent_bit_basis_hamiltonian(self):
        for h_param in (0.0, 0.1, 0.5, 1.0):
            with self.subTest(h_param=h_param):
                spectrum = np.linalg.eigvalsh(independent_hamiltonian(h_param))
                expected = spectrum[-1] - spectrum[0]
                self.assertAlmostEqual(hamiltonian_energy_width(h_param), expected, places=12)
        self.assertAlmostEqual(hamiltonian_energy_width(0.0), 10.0, places=12)
        self.assertAlmostEqual(hamiltonian_energy_width(1.0), 8.0, places=12)

    def test_reusing_signed_eigenvalues_preserves_metrics(self):
        matrices = np.asarray([
            [[1.0, 2.0], [2.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ])
        reference = curvature_metrics(matrices, 7.0)
        cached = curvature_metrics(matrices, 7.0, eigenvalues=np.linalg.eigvalsh(matrices))
        for key in METRIC_KEYS:
            with self.subTest(metric=key):
                np.testing.assert_allclose(cached[key], reference[key], equal_nan=True)

    def test_plotting_writes_four_pdfs_and_per_sample_metrics(self):
        hessians = {
            3: np.asarray([np.diag([-3.0, 2.0, 1.0, 0.0]), np.zeros((4, 4))]),
            1: np.asarray([[[1.0, 2.0], [2.0, 1.0]], np.zeros((2, 2))]),
        }
        # Supplying a matrix must determine normalization, including when
        # it differs from the default Hamiltonian at the given h parameter.
        hamiltonian = np.diag([-3.0, 1.0, 4.0])
        with tempfile.TemporaryDirectory() as temp_dir:
            output = _MODULE.save_hessian_curvature_figures(
                hessians,
                h_param=0.1,
                figures_dir=Path(temp_dir),
                hamiltonian_matrix=hamiltonian,
            )
            self.assertEqual(output["energy_width"], 7.0)
            self.assertEqual(set(output["figure_paths"]), set(METRIC_KEYS))
            self.assertEqual(len(list(Path(temp_dir).glob("*.pdf"))), 4)
            for key, path in output["figure_paths"].items():
                with self.subTest(metric=key):
                    path = Path(path)
                    self.assertEqual(path.name, f"hessian_{key}_random_points.pdf")
                    self.assertTrue(path.read_bytes().startswith(b"%PDF-"))
                    self.assertGreater(path.stat().st_size, 1000)
            self.assertEqual(Path(output["data_path"]).name, "hessian_curvature_random_points.npz")
            with np.load(output["data_path"], allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["layers"], [1, 3])
                self.assertEqual(float(archive["h_param"]), 0.1)
                self.assertEqual(float(archive["energy_width"]), 7.0)
                for layer, matrices in hessians.items():
                    reference = curvature_metrics(matrices, 7.0)
                    for key in METRIC_KEYS:
                        with self.subTest(layer=layer, metric=key):
                            np.testing.assert_allclose(
                                archive[f"L{layer}_{key}"], reference[key], equal_nan=True
                            )
                            np.testing.assert_allclose(
                                output["metrics_by_layer"][layer][key], reference[key], equal_nan=True
                            )


if __name__ == "__main__":
    unittest.main()
