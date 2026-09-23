"""Fixed-threshold rank checks using saved/analytic spectra only."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.qfim_rank import (
    DEFAULT_THRESHOLD,
    compute_qfim_rank,
    save_qfim_rank_outputs,
)


class QfimRankTests(unittest.TestCase):
    def test_fixed_cutoff_includes_exact_equality(self):
        cutoff = DEFAULT_THRESHOLD
        values = np.asarray([
            [np.nextafter(cutoff, 0.0), cutoff, np.nextafter(cutoff, np.inf), 0.0, -1e-15],
            [cutoff] * 5,
            [0.0] * 5,
        ])
        original = values.copy()
        actual = compute_qfim_rank({1: values})
        np.testing.assert_array_equal(actual["rank_by_layer"][1], [2, 5, 0])
        np.testing.assert_array_equal(values, original)
        self.assertEqual(actual["rank_by_layer"][1].dtype, np.dtype(np.int64))
        self.assertEqual(actual["threshold"], 1e-12)

    def test_fixed_cutoff_is_not_relative_to_largest_eigenvalue(self):
        actual = compute_qfim_rank({1: [[1e100, 1e-12, 0.0], [1e-12, 1e-12, 0.0]]})
        np.testing.assert_array_equal(actual["rank_by_layer"][1], [2, 2])
        custom = compute_qfim_rank({1: [[1.0, 2.0, 3.0, -5.0]]}, threshold=2.0)
        np.testing.assert_array_equal(custom["rank_by_layer"][1], [2])

    def test_order_variable_dimensions_counts_and_statistics(self):
        actual = compute_qfim_rank({
            3: [[0.0] * 6],
            1: [[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        })
        np.testing.assert_array_equal(actual["layers"], [1, 3])
        np.testing.assert_array_equal(actual["num_samples_by_layer"], [3, 1])
        np.testing.assert_array_equal(actual["num_parameters_by_layer"], [2, 6])
        np.testing.assert_allclose(actual["mean"], [1.0, 0.0])
        np.testing.assert_allclose(actual["std"], [1.0, 0.0])
        np.testing.assert_allclose(actual["sem"], [1 / np.sqrt(3), 0.0])
        np.testing.assert_array_equal(actual["min"], [0, 0])
        np.testing.assert_array_equal(actual["max"], [2, 0])

    def test_positive_subnormal_threshold_is_supported(self):
        tiny = np.nextafter(0.0, 1.0)
        actual = compute_qfim_rank({1: [[0.0, tiny]]}, threshold=tiny)
        np.testing.assert_array_equal(actual["rank_by_layer"][1], [1])

    def test_invalid_thresholds_are_rejected(self):
        for value in (0.0, -1.0, np.nan, np.inf, -np.inf, True, "1e-12", 1j, [1e-12]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compute_qfim_rank({1: [[1.0]]}, threshold=value)

    def test_invalid_spectra_are_rejected_without_dropping_samples(self):
        for values in (
            [], [1.0], np.empty((0, 2)), np.empty((2, 0)),
            [[np.nan, 1.0]], [[np.inf, 1.0]], [[-np.inf, 1.0]],
            [[1.0, 2.0], [np.nan, 0.0]], [[1j]], [["1.0"]], [[True]],
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                compute_qfim_rank({1: values})
        for mapping in ({}, {0: [[1.0]]}, {-1: [[1.0]]}, {1.0: [[1.0]]}, {True: [[1.0]]}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                compute_qfim_rank(mapping)

    def test_numerical_import_requires_no_quantum_or_plotting_libraries(self):
        code = """
import sys
sys.path.insert(0, sys.argv[1])
blocked = {'jax', 'tensorcircuit', 'tensorflow', 'matplotlib', 'optax'}
from common.qfim_rank import compute_qfim_rank
assert compute_qfim_rank({1: [[0.0, 1e-12]]})['rank_by_layer'][1].tolist() == [1]
assert blocked.isdisjoint(sys.modules), blocked.intersection(sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", code, str(SRC_DIR)], capture_output=True,
            text=True, check=False, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


class QfimRankOutputTests(unittest.TestCase):
    def test_pdf_style_filename_and_pickle_free_statistics(self):
        from common.plot import save_fig as real_save_fig
        from matplotlib.ticker import MaxNLocator

        result = compute_qfim_rank({1: [[1.0, 0.0], [1.0, 1.0]], 3: [[0.0] * 6]})
        captured = {}

        def capture(fig, ax, path, **kwargs):
            captured["bottom"] = ax.get_ylim()[0]
            captured["ticks"] = ax.get_xticks().copy()
            captured["legend"] = ax.get_legend_handles_labels()[1]
            captured["integer_locator"] = isinstance(ax.yaxis.get_major_locator(), MaxNLocator)
            return real_save_fig(fig, ax, path, **kwargs)

        with tempfile.TemporaryDirectory() as directory, mock.patch("common.plot.save_fig", side_effect=capture):
            paths = save_qfim_rank_outputs(
                result, directory, keep_key="keep0123", source_path="saved_spectra.npz",
                metadata={"layers": [1, 3], "qfim_sample_seed_base": 5, "threshold": 99.0},
            )
            self.assertEqual(paths["figure_path"].name, "qfim_rank_mean_sem_min_max_random_points_ge_1e-12_keep0123.pdf")
            self.assertTrue(paths["figure_path"].read_bytes().startswith(b"%PDF-"))
            with np.load(paths["statistics_path"], allow_pickle=False) as archive:
                for key in archive.files:
                    self.assertFalse(archive[key].dtype.hasobject, key)
                np.testing.assert_array_equal(archive["L1_rank"], [1, 2])
                self.assertEqual(archive["L1_rank"].dtype, np.dtype(np.int64))
                np.testing.assert_array_equal(archive["source_layers"], [1, 3])
                self.assertEqual(archive["threshold"].item(), 1e-12)
                self.assertEqual(archive["source_threshold"].item(), 99.0)
                self.assertEqual(archive["source_qfim_sample_seed_base"].item(), 5)
                self.assertIn("inclusive", archive["rank_criterion"].item())
                self.assertEqual(archive["source_path"].item(), "saved_spectra.npz")
        self.assertEqual(captured["bottom"], 0.0)
        np.testing.assert_array_equal(captured["ticks"], [1, 3])
        self.assertEqual(set(captured["legend"]), {r"Mean $\pm$ SEM", "Minimum", "Maximum"})
        self.assertTrue(captured["integer_locator"])

    def test_custom_cutoffs_keep_distinct_filenames_and_zero_rank_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for threshold in (2.0, 2.01, np.nextafter(0.0, 1.0)):
                result = compute_qfim_rank({1: [[0.0, -1.0]]}, threshold=threshold)
                paths.append(save_qfim_rank_outputs(result, directory, keep_key="keep01234")["figure_path"])
            self.assertIn("ge_2e00_", paths[0].name)
            self.assertEqual(len(set(paths)), 3)
            self.assertTrue(all(path.is_file() for path in paths))

    def test_invalid_metadata_and_unsafe_keep_names_are_rejected(self):
        result = compute_qfim_rank({1: [[1.0]]})
        with tempfile.TemporaryDirectory() as directory:
            for metadata in ({"nested": {"value": 1}}, {1: "bad key"}):
                with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                    save_qfim_rank_outputs(result, directory, keep_key="keep0123", metadata=metadata)
            for key in ("../keep0123", "", 3):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    save_qfim_rank_outputs(result, directory, keep_key=key)


if __name__ == "__main__":
    unittest.main()
