"""Analytic tests for NumPy Hessian statistics; no simulator is initialized.

Run with ``python -m unittest discover -s src/common -p test_hessian_statistics.py``.
"""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "hessian_statistics", Path(__file__).with_name("hessian_statistics.py"),
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
hessian_spectral_statistics = _MODULE.hessian_spectral_statistics
finite_sample_statistics = _MODULE.finite_sample_statistics


class HessianSpectralStatisticsTests(unittest.TestCase):
    def test_signed_and_offdiagonal_negative_curvature_is_included(self):
        matrices = np.array([
            np.diag([-3.0, 2.0, 0.0]),
            [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        ])
        actual = hessian_spectral_statistics(
            np.linalg.eigvalsh(matrices), count_thresholds=(3.0, 2.0, 1.0),
        )
        expected = {
            "rank": [2, 2],
            "participation_rank": [25.0 / 13.0, 16.0 / 10.0],
            "trace": [-1.0, 2.0],
            "active_signed_trace": [-1.0, 2.0],
            "absolute_trace": [5.0, 4.0],
            "shannon_entropy": [
                -np.dot([0.6, 0.4], np.log([0.6, 0.4])),
                -np.dot([0.75, 0.25], np.log([0.75, 0.25])),
            ],
            "eigcounts": [[1, 2, 2], [1, 1, 2]],
        }
        self.assertEqual(set(actual), set(expected))
        for name, values in expected.items():
            with self.subTest(metric=name):
                np.testing.assert_allclose(actual[name], values, rtol=1e-14)

    def test_inclusive_boundary_and_strict_participation_boundary(self):
        threshold = 1e-3
        actual = hessian_spectral_statistics(
            [[threshold, -2 * threshold, -threshold, 0.5 * threshold]],
            threshold=threshold, count_thresholds=(threshold, 2 * threshold, 4 * threshold),
        )
        np.testing.assert_array_equal(actual["rank"], [3])
        np.testing.assert_array_equal(actual["participation_rank"], [1])
        np.testing.assert_allclose(actual["trace"], [-1.5 * threshold])
        np.testing.assert_allclose(actual["active_signed_trace"], [-2 * threshold])
        np.testing.assert_allclose(actual["absolute_trace"], [4 * threshold])
        np.testing.assert_allclose(actual["shannon_entropy"], [1.5 * np.log(2)])
        np.testing.assert_array_equal(actual["eigcounts"], [[3, 1, 0]])

    def test_zero_and_inactive_spectra(self):
        actual = hessian_spectral_statistics([[0.0, 0.0], [1e-13, -2e-13]])
        for name, values in actual.items():
            with self.subTest(metric=name):
                if name == "trace":
                    np.testing.assert_allclose(values, [0.0, -1e-13], atol=0.0)
                else:
                    np.testing.assert_array_equal(values, np.zeros_like(values))

    def test_all_negative_spectrum_and_single_active_eigenvalue(self):
        actual = hessian_spectral_statistics([[-3.0, -2.0], [0.0, -5.0]])
        np.testing.assert_allclose(actual["trace"], [-5.0, -5.0])
        np.testing.assert_allclose(actual["absolute_trace"], [5.0, 5.0])
        np.testing.assert_allclose(actual["participation_rank"], [25.0 / 13.0, 1.0])
        self.assertEqual(actual["shannon_entropy"][1], 0.0)

    def test_dimensionless_metrics_are_stable_at_tiny_and_huge_scales(self):
        reference = hessian_spectral_statistics(
            [[-3.0, 2.0, 0.0]], threshold=0.01, count_thresholds=(1.0, 3.0),
        )
        for scale in (1e-300, 1e300):
            with self.subTest(scale=scale), np.errstate(all="raise"):
                actual = hessian_spectral_statistics(
                    scale * np.array([[-3.0, 2.0, 0.0]]),
                    threshold=0.01 * scale, count_thresholds=(scale, 3.0 * scale),
                )
                for name in ("rank", "participation_rank", "shannon_entropy", "eigcounts"):
                    np.testing.assert_allclose(actual[name], reference[name], rtol=1e-14)
                for name in ("trace", "active_signed_trace", "absolute_trace"):
                    np.testing.assert_allclose(actual[name] / scale, reference[name], rtol=1e-14)

    def test_ratios_and_signed_trace_survive_overflowing_absolute_trace(self):
        with np.errstate(all="raise"):
            actual = hessian_spectral_statistics([[1e308, 1e308, -1e308, -1e308]])
        np.testing.assert_array_equal(actual["trace"], [0.0])
        np.testing.assert_array_equal(actual["active_signed_trace"], [0.0])
        np.testing.assert_allclose(actual["participation_rank"], [4.0])
        np.testing.assert_allclose(actual["shannon_entropy"], [np.log(4.0)])
        self.assertTrue(np.isposinf(actual["absolute_trace"][0]))

    def test_default_count_thresholds_match_qfim_figure_thresholds(self):
        actual = hessian_spectral_statistics([[10.0, -5.0, 1.0, -0.1, 0.01, -0.001, 0.0001]])
        np.testing.assert_array_equal(actual["eigcounts"], [[1, 2, 3, 4, 5, 6, 7]])

    def test_invalid_eigenvalue_arrays_are_rejected(self):
        invalid = (
            [], [[]], np.empty((0, 2)), [1.0, 2.0], [[[1.0]]],
            [[np.nan]], [[np.inf]], [[-np.inf]], [[1.0j]], [["1.0"]], [[True]],
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                hessian_spectral_statistics(values)

    def test_invalid_thresholds_are_rejected(self):
        for threshold in (0, -1, np.nan, np.inf, [1.0], 1.0j, "1.0", True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                hessian_spectral_statistics([[1.0]], threshold=threshold)
        for thresholds in ([], 1.0, [[1.0]], [0.0], [-1.0], [np.nan], [np.inf], [1.0j], ["1.0"]):
            with self.subTest(count_thresholds=thresholds), self.assertRaises(ValueError):
                hessian_spectral_statistics([[1.0]], count_thresholds=thresholds)


class FiniteSampleStatisticsTests(unittest.TestCase):
    def test_only_finite_samples_contribute_to_sample_sem(self):
        mean, sem, minimum, maximum, count = finite_sample_statistics(
            [[1.0, 2.0, 3.0], [np.nan, np.inf, -np.inf]],
        )
        self.assertEqual(count, 3)
        np.testing.assert_allclose([mean, sem, minimum, maximum], [2.0, 1 / np.sqrt(3), 1.0, 3.0])

    def test_empty_and_all_nonfinite_samples(self):
        for samples in ([], [np.nan, np.inf, -np.inf]):
            with self.subTest(samples=samples):
                actual = finite_sample_statistics(samples)
                self.assertEqual(actual[-1], 0)
                self.assertTrue(np.isnan(actual[:4]).all())

    def test_singleton_constant_and_zero_samples(self):
        for samples, expected in (([np.nan, -2.0], -2.0), ([3.0] * 4, 3.0), ([0.0] * 3, 0.0)):
            with self.subTest(samples=samples):
                actual = finite_sample_statistics(samples)
                np.testing.assert_array_equal(actual[:4], [expected, 0.0, expected, expected])
                self.assertEqual(actual[-1], np.count_nonzero(np.isfinite(samples)))

    def test_extreme_finite_samples_have_stable_moments(self):
        for scale in (1e-300, 1e308):
            with self.subTest(scale=scale), np.errstate(all="raise"):
                mean, sem, minimum, maximum, count = finite_sample_statistics([-scale, scale])
                self.assertEqual(count, 2)
                self.assertEqual(mean, 0.0)
                np.testing.assert_allclose(sem / scale, 1.0, rtol=1e-14)
                self.assertEqual(minimum, -scale)
                self.assertEqual(maximum, scale)

    def test_nonreal_samples_are_rejected(self):
        for samples in ([1.0j], ["1.0"], [True]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                finite_sample_statistics(samples)


if __name__ == "__main__":
    unittest.main()
