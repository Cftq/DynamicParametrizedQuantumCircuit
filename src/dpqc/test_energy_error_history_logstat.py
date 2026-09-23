"""Check saved-energy log-statistics without importing quantum runtimes.

Run with ``python -m unittest discover -s src/dpqc -p test_energy_error_history_logstat.py``.
"""

import ast
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_MODULE_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _MODULE_DIR.parent / "common"
for _directory in (_MODULE_DIR, _COMMON_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import plot


def _band_limits(collection, iterations):
    """Read actual rendered band boundaries, independent of path ordering."""
    vertices = np.concatenate([path.vertices for path in collection.get_paths()])
    values = [vertices[vertices[:, 0] == iteration, 1] for iteration in iterations]
    return np.array([value.min() for value in values]), np.array([
        value.max() for value in values
    ])


class EnergyErrorHistoryLogstatTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_curves_and_bands_reproduce_population_energy_moments(self):
        ground = -4.0
        errors = np.array([[1.0, 3.0, 5.0], [3.0, 7.0, 9.0]])
        with patch.object(plot, "save_fig") as save:
            plot.plot_energy_error_history_logstat(
                {3: ground + errors, 1: ground + 2.0 * errors},
                [3, 1],
                ground_energy=ground,
                outpath="unused.pdf",
            )

        save.assert_called_once()
        fig, ax, outpath = save.call_args.args
        self.assertIs(ax.figure, fig)
        self.assertEqual(outpath, "unused.pdf")
        self.assertTrue(save.call_args.kwargs["outside_legend"])
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual([line.get_label() for line in ax.lines], ["L3", "L1"])
        self.assertEqual(len(ax.collections), 2)

        for multiplier, line, band in zip((1.0, 2.0), ax.lines, ax.collections):
            iterations = np.arange(3)
            np.testing.assert_array_equal(line.get_xdata(), iterations)
            center = line.get_ydata()
            low, high = _band_limits(band, iterations)
            # Recover the arithmetic moments implied by the plotted lognormal
            # center and one-log-standard-deviation band. This distinguishes
            # the unitary convention from direct statistics of log(errors).
            sigma = (np.log(high) - np.log(low)) / 2.0
            implied_mean = center * np.exp(sigma**2 / 2.0)
            implied_variance = np.expm1(sigma**2) * implied_mean**2
            np.testing.assert_allclose(np.sqrt(low * high), center, rtol=1e-13)
            np.testing.assert_allclose(
                implied_mean, multiplier * np.array([2.0, 5.0, 7.0]) + 1e-12,
                rtol=1e-13,
            )
            # Population variance is [1, 4, 4]; sample variance would double it.
            np.testing.assert_allclose(
                implied_variance, multiplier**2 * np.array([1.0, 4.0, 4.0]),
                rtol=1e-12,
            )
            self.assertFalse(np.allclose(center, multiplier * np.sqrt(errors.prod(axis=0))))

    def test_zero_error_and_single_run_have_finite_collapsed_bands(self):
        ground = -2.0
        with patch.object(plot, "save_fig") as save:
            plot.plot_energy_error_history_logstat(
                {2: np.array([[ground, ground + 2.0, ground - 3.0]])},
                [2],
                ground_energy=ground,
                outpath="unused.pdf",
            )
        _, ax, _ = save.call_args.args
        center = ax.lines[0].get_ydata()
        low, high = _band_limits(ax.collections[0], np.arange(3))
        expected = np.array([1e-12, 2.0 + 1e-12, 3.0 + 1e-12])
        np.testing.assert_allclose(center, expected, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(low, expected, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(high, expected, rtol=1e-13, atol=0.0)
        self.assertTrue(np.isfinite(center).all())
        self.assertTrue((center > 0.0).all())

    def test_helper_saves_pdf_from_energy_histories(self):
        with tempfile.TemporaryDirectory() as temporary:
            outpath = Path(temporary) / "energy_figures" / "energy_error_history_logstat.pdf"
            plot.plot_energy_error_history_logstat(
                {1: np.array([[0.0, -0.5, -1.0], [0.5, 0.0, -0.8]])},
                [1],
                ground_energy=-1.0,
                title="Saved-energy history",
                outpath=str(outpath),
            )
            self.assertTrue(outpath.is_file())
            self.assertTrue(outpath.read_bytes().startswith(b"%PDF-"))
            self.assertGreater(outpath.stat().st_size, 100)

    def test_dpqc_regular_workflow_calls_shared_plotter_and_reset_forwards(self):
        visualizer = _MODULE_DIR / "DPQC_overparam_visualize.py"
        tree = ast.parse(visualizer.read_text(encoding="utf-8"))
        shared_imports = [
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "plot"
            for alias in node.names
        ]
        self.assertIn("plot_energy_error_history_logstat", shared_imports)
        calls = [
            node.value
            for node in tree.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "plot_energy_error_history_logstat"
        ]
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual([argument.id for argument in call.args], [
            "energy_traces_by_layer", "vqe_layer_list",
        ])
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        self.assertEqual(keywords["ground_energy"].id, "smallest_eigval")
        self.assertIn("energy_error_history_logstat.pdf", [
            node.value for node in ast.walk(keywords["outpath"])
            if isinstance(node, ast.Constant)
        ])

        reset = importlib.import_module("DPQC_overparam_reset_visualize")
        command = reset._build_visualizer_command(0.1)
        self.assertEqual(Path(command[1]), visualizer)
        self.assertEqual(command[command.index("--output-family") + 1], "dpqc_reset")
        self.assertNotIn("--gap-normalized-only", command)
        self.assertNotIn("--qfim-logdet-only", command)
        self.assertNotIn("--hessian-only", command)


if __name__ == "__main__":
    unittest.main()
