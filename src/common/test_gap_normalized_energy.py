"""Saved-energy accuracy checks; no training or QFIM computations are run.

Run with ``python -m unittest discover -s src/common -p test_gap_normalized_energy.py``.
"""

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

from common.gap_normalized_energy import (
    DEFAULT_THRESHOLDS,
    _beeswarm_x_positions,
    _swarm_half_widths,
    compute_gap_normalized_energy,
    save_gap_normalized_energy_outputs,
    spectral_gap_from_hamiltonian,
)
from common.hessian_curvature import hamiltonian_matrix_numpy


def independent_hamiltonian(h_param):
    """Project Hamiltonian constructed independently in the bit basis."""
    matrix = np.zeros((16, 16))
    for index in range(16):
        z_values = [1 - 2 * ((index >> (3 - wire)) & 1) for wire in range(4)]
        zz_sum = sum(z_values[i] * z_values[j] for i, j in ((0, 1), (0, 2), (1, 3), (2, 3)))
        matrix[index, index] = -(1 - h_param) * zz_sum - h_param * sum(z_values)
        matrix[index, index ^ 15] = -(1 - h_param)
    return matrix


class SpectralGapTests(unittest.TestCase):
    def test_gap_skips_entire_degenerate_ground_subspace(self):
        actual = spectral_gap_from_hamiltonian(np.diag([-3.0, -3.0, -1.0, 4.0]))
        self.assertEqual(actual["ground_energy"], -3.0)
        self.assertEqual(actual["first_excited_energy"], -1.0)
        self.assertEqual(actual["spectral_gap"], 2.0)
        self.assertEqual(actual["ground_degeneracy"], 2)
        self.assertGreater(actual["degeneracy_tolerance"], 0.0)

    def test_spectator_ancilla_preserves_gap_and_doubles_degeneracy(self):
        system = np.diag([-3.0, -3.0, -1.0, 4.0])
        reference = spectral_gap_from_hamiltonian(system)
        actual = spectral_gap_from_hamiltonian(np.kron(system, np.eye(2)))
        for key in ("ground_energy", "first_excited_energy", "spectral_gap"):
            self.assertEqual(actual[key], reference[key])
        self.assertEqual(actual["ground_degeneracy"], 4)

    def test_gap_accepts_complex_hermitian_and_float32_inputs(self):
        for matrix in (
            np.asarray([[0.0, -1j], [1j, 0.0]]),
            np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        ):
            with self.subTest(dtype=matrix.dtype):
                actual = spectral_gap_from_hamiltonian(matrix)
                self.assertEqual(actual["spectral_gap"], 2.0)
                self.assertEqual(actual["ground_energy"], -1.0)

    def test_near_degeneracy_uses_numerical_resolution(self):
        matrix = np.diag([-2.0, -2.0 + np.finfo(float).eps, 0.0, 3.0])
        actual = spectral_gap_from_hamiltonian(matrix)
        self.assertEqual(actual["ground_degeneracy"], 2)
        self.assertEqual(actual["spectral_gap"], 2.0)

    def test_invalid_hamiltonians_and_absent_gap_are_rejected(self):
        matrices = (
            np.asarray(1.0), np.zeros((0, 0)), np.zeros((2, 3)),
            np.diag([0.0, np.nan]), np.diag([0.0, np.inf]),
            np.asarray([[0.0, 1.0], [0.0, 1.0]]),
            np.asarray([[0.0, 1j], [1j, 1.0]]),
            np.asarray([["zero", "one"], ["one", "zero"]]),
            np.zeros((2, 2)), 3.0 * np.eye(4), np.asarray([[2.0]]),
        )
        for index, matrix in enumerate(matrices):
            with self.subTest(index=index), self.assertRaises(ValueError):
                spectral_gap_from_hamiltonian(matrix)

    def test_project_hamiltonian_matches_independent_bit_basis(self):
        for h_param in (-0.25, 0.0, 0.1, 0.5, 1.0, 1.25):
            with self.subTest(h_param=h_param):
                reference = independent_hamiltonian(h_param)
                np.testing.assert_allclose(hamiltonian_matrix_numpy(h_param), reference, atol=1e-14)
                spectrum = np.linalg.eigvalsh(reference)
                expected_gap = spectrum[np.flatnonzero(spectrum - spectrum[0] > 1e-10)[0]] - spectrum[0]
                actual = compute_gap_normalized_energy({1: [spectrum[0]]}, h_param=h_param)
                self.assertAlmostEqual(actual["ground_energy"], spectrum[0], places=12)
                self.assertAlmostEqual(actual["spectral_gap"], expected_gap, places=12)
        for h_param in (0.0, 1.0):
            actual = spectral_gap_from_hamiltonian(hamiltonian_matrix_numpy(h_param))
            self.assertAlmostEqual(actual["spectral_gap"], 2.0, places=12)


class GapNormalizedEnergyTests(unittest.TestCase):
    def compute(self, energies, **kwargs):
        arguments = {"h_param": 0.1, "hamiltonian_matrix": np.diag([-2.0, 0.0, 4.0])}
        arguments.update(kwargs)
        return compute_gap_normalized_energy(energies, **arguments)

    def test_layer_order_samples_summaries_and_variable_run_counts(self):
        actual = self.compute({3: [-2.0, 0.0], 1: [-2.0, -1.0, 0.0, 4.0]})
        np.testing.assert_array_equal(actual["layers"], [1, 3])
        np.testing.assert_array_equal(actual["num_runs_by_layer"], [4, 2])
        np.testing.assert_allclose(actual["normalized_errors_by_layer"][1], [0.0, 0.5, 1.0, 3.0])
        np.testing.assert_allclose(actual["normalized_errors_by_layer"][3], [0.0, 1.0])
        np.testing.assert_allclose(actual["mean"], [1.125, 0.5])
        np.testing.assert_allclose(actual["median"], [0.75, 0.5])
        np.testing.assert_allclose(actual["min"], [0.0, 0.0])
        np.testing.assert_allclose(actual["max"], [3.0, 1.0])
        np.testing.assert_allclose(actual["sem"], [np.std([0.0, 0.5, 1.0, 3.0], ddof=1) / 2, 0.5])
        np.testing.assert_allclose(actual["success_probabilities"], [[0.25] * 6, [0.5] * 6])
        np.testing.assert_array_equal(actual["thresholds"], [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])

    def test_success_criterion_is_strict_at_threshold_equality(self):
        # Powers of two are exact: subtraction/division cannot move equality.
        actual = self.compute({1: [-2.0, -1.5, -1.0, 0.0]}, thresholds=(0.25, 0.5, 1.0))
        np.testing.assert_array_equal(actual["normalized_errors_by_layer"][1], [0.0, 0.25, 0.5, 1.0])
        np.testing.assert_array_equal(actual["success_probabilities"], [[0.25, 0.5, 0.75]])

    def test_default_thresholds_count_all_trials_in_denominator(self):
        actual = self.compute(
            {1: [0.0, 5e-9, 5e-8, 5e-7, 2.0]},
            hamiltonian_matrix=np.diag([0.0, 1.0, 2.0]),
        )
        self.assertEqual(tuple(actual["thresholds"]), DEFAULT_THRESHOLDS)
        np.testing.assert_allclose(actual["success_probabilities"], [[4 / 5, 4 / 5, 3 / 5, 2 / 5, 1 / 5, 1 / 5]])

    def test_single_run_has_zero_sem(self):
        actual = self.compute({np.int64(2): [-1.0]})
        np.testing.assert_array_equal(actual["sem"], [0.0])
        np.testing.assert_array_equal(actual["num_runs_by_layer"], [1])

    def test_roundoff_negative_error_clips_only_normalized_samples(self):
        below_ground = np.nextafter(-2.0, -np.inf)
        energies = np.asarray([below_ground, -2.0, -1.0])
        actual = self.compute({1: energies})
        np.testing.assert_array_equal(actual["normalized_errors_by_layer"][1], [0.0, 0.0, 0.5])
        np.testing.assert_array_equal(actual["final_energies_by_layer"][1], energies)
        np.testing.assert_array_equal(actual["num_clipped_negative_by_layer"], [1])
        self.assertLess(abs(below_ground + 2.0), actual["negative_roundoff_tolerance"])

    def test_materially_subground_saved_energy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "below the ground energy"):
            self.compute({1: [-2.0 - 1e-8, -1.0]})

    def test_joint_positive_rescaling_and_energy_shift_preserve_accuracy(self):
        matrix = np.diag([-2.0, 0.0, 4.0])
        energies = np.asarray([-2.0, -1.75, -0.5, 4.0])
        reference = self.compute({1: energies}, thresholds=(0.25, 1.0))
        for scale, shift in ((1e-100, 0.0), (1e100, 0.0), (4.0, 7.0), (0.5, -5.0)):
            with self.subTest(scale=scale, shift=shift):
                actual = self.compute(
                    {1: scale * energies + shift},
                    hamiltonian_matrix=scale * matrix + shift * np.eye(3),
                    thresholds=(0.25, 1.0),
                )
                np.testing.assert_allclose(actual["normalized_errors_by_layer"][1], reference["normalized_errors_by_layer"][1])
                np.testing.assert_array_equal(actual["success_probabilities"], reference["success_probabilities"])

    def test_mixed_state_excited_population_obeys_gap_bound(self):
        # Ground states have total population .3; excitation population .7
        # is bounded by the gap-normalized energy 1.2.
        matrix = np.diag([-2.0, -2.0, 0.0, 2.0])
        population = np.asarray([0.1, 0.2, 0.2, 0.5])
        energy = float(np.dot(population, np.diag(matrix)))
        actual = self.compute({1: [energy]}, hamiltonian_matrix=matrix)
        normalized = actual["normalized_errors_by_layer"][1][0]
        self.assertAlmostEqual(normalized, 1.2)
        self.assertLessEqual(1 - population[:2].sum(), normalized)

    def test_empty_or_invalid_layers_are_rejected(self):
        for energies in ({}, {0: [0.0]}, {-1: [0.0]}, {1.0: [0.0]}, {"1": [0.0]}, {True: [0.0]}):
            with self.subTest(energies=energies), self.assertRaises(ValueError):
                self.compute(energies)

    def test_invalid_energy_arrays_are_rejected(self):
        for values in ([], 0.0, [[0.0]], [np.nan], [np.inf], [-np.inf], [1j], ["0"], [True]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.compute({1: values})

    def test_invalid_thresholds_are_rejected(self):
        for thresholds in ([], 1e-6, [[1e-6]], [0.0], [-1.0], [np.nan], [np.inf], [1j], ["1e-6"]):
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                self.compute({1: [0.0]}, thresholds=thresholds)

    def test_nonfinite_h_is_rejected_even_with_explicit_hamiltonian(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(h_param=value), self.assertRaises(ValueError):
                self.compute({1: [0.0]}, h_param=value)

    def test_numerical_module_import_does_not_load_quantum_or_plotting_stack(self):
        code = """
import sys
sys.path.insert(0, sys.argv[1])
from common.gap_normalized_energy import compute_gap_normalized_energy
compute_gap_normalized_energy({1: [-4.0]}, h_param=1.0)
blocked = {'jax', 'tensorcircuit', 'tensorflow', 'matplotlib', 'common.hamiltonian'}
assert blocked.isdisjoint(sys.modules), blocked.intersection(sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", code, str(SRC_DIR)],
            capture_output=True, text=True, check=False, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


class GapNormalizedBeeswarmTests(unittest.TestCase):
    def test_display_distance_packing_is_deterministic_across_symlog_regions(self):
        from common.plot import new_fig_ax
        import matplotlib.pyplot as plt

        fig, ax = new_fig_ax()
        self.addCleanup(plt.close, fig)
        ax.set_yscale("symlog", linthresh=1e-9, linscale=0.5)
        ax.set_xlim(-9.0, 11.0)
        ax.set_ylim(-1e-9, 1e4)
        fig.canvas.draw()
        values = np.asarray([0.0, 0.0, 1e-14, 1e-7, 1e-6, 1.0, 1.0, 1e3])
        unchanged = values.copy()
        x_positions = _beeswarm_x_positions(ax, 1.0, values, half_width=9.0)
        repeated = _beeswarm_x_positions(ax, 1.0, values, half_width=9.0)
        np.testing.assert_array_equal(x_positions, repeated)
        np.testing.assert_array_equal(values, unchanged)
        pixels = ax.transData.transform(np.column_stack((x_positions, values)))
        distances = np.linalg.norm(pixels[:, None, :] - pixels[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        diameter = np.sqrt(10.0) * fig.dpi / 72.0 * 1.1
        self.assertGreaterEqual(float(np.min(distances)), diameter * (1 - 1e-9))

    def test_one_hundred_repeated_zeros_are_preserved_within_layer_cell(self):
        from common.plot import new_fig_ax
        import matplotlib.pyplot as plt

        fig, ax = new_fig_ax()
        self.addCleanup(plt.close, fig)
        ax.set_yscale("symlog", linthresh=1e-11, linscale=0.5)
        ax.set_xlim(0.5, 16.5)
        ax.set_ylim(-1e-11, 10.0)
        fig.canvas.draw()
        values = np.zeros(100)
        x_positions = _beeswarm_x_positions(ax, 2.0, values, half_width=0.36)
        self.assertEqual(x_positions.size, 100)
        self.assertEqual(np.unique(x_positions).size, 100)
        self.assertLessEqual(float(np.max(np.abs(x_positions - 2.0))), 0.36 + 1e-12)
        self.assertGreater(np.ptp(x_positions), 0.5)
        np.testing.assert_array_equal(values, np.zeros(100))
        np.testing.assert_array_equal(
            x_positions, _beeswarm_x_positions(ax, 2.0, values, half_width=0.36),
        )
        np.testing.assert_allclose(_swarm_half_widths([1, 2, 4, 8]), [0.36, 0.36, 0.72, 1.44])


class GapNormalizedOutputTests(unittest.TestCase):
    def test_beeswarm_success_alias_and_pickle_free_statistics_are_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            output = save_gap_normalized_energy_outputs(
                {3: [-2.0, -1.5], 1: [-2.0, 0.0, 4.0]},
                h_param=0.1, figures_dir=directory / "figures",
                statistics_outpath=directory / "statistics" / "metrics.npz",
                hamiltonian_matrix=np.diag([-2.0, 0.0, 4.0]),
                metadata={"model_family": "test_saved_energies"},
            )
            self.assertEqual(set(output["figure_paths"]), {
                "normalized_energy_error", "success_probability", "success_probability_multiple_tolerances",
            })
            self.assertEqual(len(list((directory / "figures").glob("*.pdf"))), 3)
            self.assertEqual(
                output["figure_paths"]["success_probability_multiple_tolerances"].name,
                "success_probability_multiple_tolerances_gap_normalized.pdf",
            )
            self.assertEqual(
                output["figure_paths"]["success_probability"].read_bytes(),
                output["figure_paths"]["success_probability_multiple_tolerances"].read_bytes(),
            )
            for path in output["figure_paths"].values():
                self.assertTrue(Path(path).read_bytes().startswith(b"%PDF-"))
                self.assertGreater(Path(path).stat().st_size, 1000)
            with np.load(output["data_path"], allow_pickle=False) as archive:
                for key in archive.files:
                    self.assertFalse(archive[key].dtype.hasobject, key)
                np.testing.assert_array_equal(archive["layers"], [1, 3])
                np.testing.assert_array_equal(archive["num_runs_by_layer"], [3, 2])
                np.testing.assert_allclose(archive["success_probabilities"], [[1 / 3] * 6, [1 / 2] * 6])
                np.testing.assert_array_equal(archive["L1_final_energies"], [-2.0, 0.0, 4.0])
                np.testing.assert_array_equal(archive["L3_normalized_errors"], [0.0, 0.25])
                self.assertEqual(float(archive["spectral_gap"]), 2.0)
                self.assertEqual(str(archive["model_family"]), "test_saved_energies")
                self.assertIn("strict", str(archive["success_criterion"]))
                self.assertIn("beeswarm", str(archive["error_plot_style"]))
                self.assertIn("crowded ties may overlap", str(archive["beeswarm_overflow_policy"]))
                self.assertEqual(archive["L1_beeswarm_x"].size, 3)

    def test_plot_preserves_all_y_values_and_keeps_zero_markers_inside_axes(self):
        from common.plot import save_fig as real_save_fig

        energies = {1: np.zeros(100), 2: np.asarray([0.0, 1e-12, 1e-9, 1e-5, 1.0])}
        captured = {}

        def capture_figure(fig, ax, outpath, **kwargs):
            if Path(outpath).name == "final_gap_normalized_energy_error.pdf":
                captured["offsets"] = [collection.get_offsets().copy() for collection in ax.collections]
                captured["labels"] = [line.get_label() for line in ax.lines]
                zero_pixel_y = ax.transData.transform((1.0, 0.0))[1]
                captured["zero_margin"] = zero_pixel_y - ax.bbox.y0
                captured["marker_radius"] = np.sqrt(10.0) * fig.dpi / 144.0
            return real_save_fig(fig, ax, outpath, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch("common.plot.save_fig", side_effect=capture_figure):
            save_gap_normalized_energy_outputs(
                energies, h_param=0.1, figures_dir=temp_dir,
                thresholds=(1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10),
                hamiltonian_matrix=np.diag([0.0, 1.0]),
            )
        self.assertEqual(len(captured["offsets"]), 2)
        for offsets, layer in zip(captured["offsets"], (1, 2)):
            np.testing.assert_array_equal(offsets[:, 1], energies[layer])
            self.assertEqual(offsets.shape[0], len(energies[layer]))
        self.assertIn("Median", captured["labels"])
        self.assertFalse(any("Mean" in label for label in captured["labels"]))
        self.assertGreater(captured["zero_margin"], captured["marker_radius"])

    def test_reserved_or_object_metadata_cannot_corrupt_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for metadata in ({"spectral_gap": 99}, {"extra": {"nested": 1}}, {1: "invalid key"}):
                with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                    save_gap_normalized_energy_outputs(
                        {1: [-2.0]}, h_param=0.1, figures_dir=temp_dir,
                        hamiltonian_matrix=np.diag([-2.0, 0.0]), metadata=metadata,
                    )


if __name__ == "__main__":
    unittest.main()
