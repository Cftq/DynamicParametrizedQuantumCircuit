"""CLI regression checks for plotting saved Hessians without quantum imports.

Run with ``python -m unittest discover -s src/dpqc -p test_hessian_visualization.py``.
All generated archives and figures live in a temporary directory.
"""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


_MODULE_DIR = Path(__file__).resolve().parent
_VISUALIZER = _MODULE_DIR / "DPQC_overparam_visualize.py"
_RESET_VISUALIZER = _MODULE_DIR / "DPQC_overparam_reset_visualize.py"


def _archive_fixture(output_family):
    """Two signed analytic Hessians, on an intentionally non-configured schedule."""
    parameters_per_layer, model_id = {
        "dpqc": (14, "dpqc_dynamic_channel"),
        "dpqc_reset": (12, "dpqc_reset_fixed_rx_pi"),
    }[output_family]
    data = {
        "schema_version": np.asarray(2),
        "h_param": np.asarray(0.1),
        "layers": np.asarray([3, 1]),
        "num_hessian_samples": np.asarray(2),
        "hessian_sample_seed_base": np.asarray(1234),
        "parameter_distribution": np.asarray("analytic test fixture"),
        "parameters_per_layer": np.asarray(parameters_per_layer),
        "hessian_method": np.asarray("analytic test fixture"),
        "hvp_chunk_size": np.asarray(8),
        "output_family": np.asarray(output_family),
        "model_id": np.asarray(model_id),
    }
    for layer in (3, 1):
        dimension = parameters_per_layer * layer
        matrices = np.zeros((2, dimension, dimension))
        matrices[0, :2, :2] = [[0.0, 3.0], [3.0, 0.0]]
        matrices[1, :2, :2] = [[-2.0, 0.0], [0.0, 1.0]]
        data[f"L{layer}_hessian"] = matrices
        data[f"L{layer}_theta"] = np.zeros((2, dimension))
    return data


class HessianVisualizationTests(unittest.TestCase):
    def test_reuse_cli_avoids_quantum_imports_and_preserves_archive_sampling(self):
        for output_family in ("dpqc", "dpqc_reset"):
            with self.subTest(output_family=output_family):
                self._assert_reuse_cli(output_family)

    def _assert_reuse_cli(self, output_family):
        with tempfile.TemporaryDirectory() as temporary:
            if output_family == "dpqc":
                # Exercise the original model's default family and paths.
                h_root = Path(temporary) / "figs" / "dpqc" / "h_0.1"
                results_dir = h_root / "numerical_results" / "hessian"
                figures_dir = h_root / "hessian_figures"
                path_arguments = []
            else:
                results_dir = Path(temporary) / "saved"
                figures_dir = Path(temporary) / "figures"
                path_arguments = [
                    "--output-family", output_family,
                    "--hessian-results-dir", str(results_dir),
                    "--hessian-figures-dir", str(figures_dir),
                ]
            results_dir.mkdir(parents=True)
            archive_path = results_dir / "hessian_random_points.npz"
            np.savez_compressed(archive_path, **_archive_fixture(output_family))
            original_archive = archive_path.read_bytes()
            # A fresh interpreter tests the real __main__ early-exit path.
            # Any accidental quantum import raises ModuleNotFoundError.
            launch = (
                "import runpy, sys; "
                "sys.modules['jax'] = None; "
                "sys.modules['tensorcircuit'] = None; "
                "sys.argv = sys.argv[1:]; "
                "runpy.run_path(sys.argv[0], run_name='__main__')"
            )
            environment = os.environ.copy()
            environment["MPLBACKEND"] = "Agg"
            environment["PYTHONIOENCODING"] = "utf-8"
            completed = subprocess.run(
                [
                    sys.executable, "-c", launch, str(_VISUALIZER),
                    "--h-param", "0.1",
                    "--hessian-only", "--reuse-hessian-results",
                    *path_arguments,
                    "--hessian-rank-threshold", "1.0",
                    # A compute-only option must not replace archived counts.
                    "--hessian-num-samples", "999",
                ],
                cwd=temporary, env=environment, capture_output=True,
                text=True, timeout=60, check=False,
            )
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )
            self.assertEqual(archive_path.read_bytes(), original_archive)
            # Six existing summaries, six new summaries, and two spectra/layer.
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
                self.assertEqual(float(archive["rank_threshold"]), 1.0)
                self.assertEqual(str(archive["output_family"]), output_family)
                for layer in (1, 3):
                    expected = {
                        "rank": [2, 2],
                        "trace": [0.0, -1.0],
                        "absolute_trace": [6.0, 3.0],
                        "abs_entry_sum": [6.0, 3.0],
                        # |lambda| = 1 contributes to rank/entropy, but the
                        # participation calculation uses a strict cutoff.
                        "participation_rank": [2.0, 1.0],
                        "shannon_entropy": [np.log(2), np.log(3) - 2 * np.log(2) / 3],
                    }
                    for key, values in expected.items():
                        with self.subTest(layer=layer, metric=key):
                            np.testing.assert_allclose(archive[f"L{layer}_{key}"], values)
                np.testing.assert_allclose(archive["absolute_trace_mean"], [4.5, 4.5])
                np.testing.assert_allclose(archive["absolute_trace_sem"], [1.5, 1.5])

    def test_reset_wrapper_forwards_reuse_and_explicit_directories(self):
        spec = importlib.util.spec_from_file_location("reset_visualizer_test", _RESET_VISUALIZER)
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        command = wrapper._build_visualizer_command(
            0.1, hessian_only=True, reuse_hessian_results=True,
            hessian_results_dir=Path("saved Hessians"),
            hessian_figures_dir=Path("Hessian figures"),
            hessian_rank_threshold=1.0,
        )
        self.assertEqual(command[:2], (sys.executable, str(_VISUALIZER)))
        self.assertIn("--hessian-only", command)
        self.assertIn("--reuse-hessian-results", command)
        for option, expected in (
            ("--output-family", "dpqc_reset"),
            ("--hessian-results-dir", "saved Hessians"),
            ("--hessian-figures-dir", "Hessian figures"),
            ("--hessian-rank-threshold", "1.0"),
        ):
            self.assertEqual(command[command.index(option) + 1], expected)


if __name__ == "__main__":
    unittest.main()
