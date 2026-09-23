"""Small real VQE -> saved QFIM/HS -> Hessian -> visualization integration.

Run in the configured numerical environment; all outputs use a temporary root.
No production configuration or saved experiment is changed.
"""

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


_HERE = Path(__file__).resolve().parent
for directory in (_HERE, _HERE.parent / "common"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

_MISSING = [
    name for name in ("jax", "optax", "numpy", "matplotlib")
    if importlib.util.find_spec(name) is None
]


@unittest.skipIf(bool(_MISSING), "Missing dependencies: " + ", ".join(_MISSING))
class UnitaryCartanPipelineTests(unittest.TestCase):
    def test_small_pipeline_and_saved_result_visualization(self):
        import numpy as np
        import config_overparam as cfg
        import plot as plot_style
        import unitary_pqc_overparam_common as common
        import unitary_pqc_overparam_vqe as vqe
        import unitary_pqc_overparam_qfim as qfim
        import unitary_pqc_overparam_hessian as hessian
        import unitary_pqc_overparam_visualize as visualize
        from unitary_pqc_energy_results import load_unitary_final_energies
        from unitary_pqc_hessian_results import load_unitary_hessian_result

        settings = {
            "NUM_RUNS": 2, "STEPS": 3, "SAMPLE_EVERY": 2,
            "NUM_QFIM_SAMPLES": 2,
            "UNITARY_PQC_MAX_LAYER": 1,
            "UNITARY_PQC_DENSE_UNTIL_LAYER": 1,
            "UNITARY_PQC_QFIM_MAX_LAYER": 1,
            "UNITARY_PQC_QFIM_DENSE_UNTIL_LAYER": 1,
        }
        with tempfile.TemporaryDirectory(prefix="unitary-cartan-") as temporary:
            root = Path(temporary)
            with (
                patch.multiple(cfg, **settings),
                patch.object(common, "_PROJECT_ROOT", root),
                patch.object(visualize, "_SRC_DIR", root / "src"),
                patch.object(plot_style, "SAVE_DPI", 60),
                patch.object(plot_style, "NUMERICAL_SAVE_PNG", True),
                patch.object(plot_style, "NUMERICAL_SAVE_PDF", False),
            ):
                energy_path = Path(vqe.run_unitary_pqc_vqe_stage(
                    h_param=0.3, vqe_batch_size=2, device="cpu",
                ))
                before = hashlib.sha256(energy_path.read_bytes()).digest()
                with np.load(energy_path, allow_pickle=False) as archive:
                    self.assertEqual(archive["num_params_per_layer"].item(), 60)
                    self.assertEqual(archive["ansatz"].item(), "unitary_pqc")
                    self.assertNotIn("measurement_outcome", archive.files)
                    self.assertEqual(archive["L1_best_theta"].shape, (60,))
                    self.assertEqual(archive["L1_energy_traces"].shape, (2, 3))

                qfim_result = qfim.run_unitary_pqc_qfim_stage(
                    h_param=0.3, analysis_batch_size=1,
                    include_optimization_path=True, device="cpu",
                )
                hessian_result = hessian.run_unitary_pqc_hessian_stage(
                    h_param=0.3, analysis_batch_size=1, device="cpu",
                )
                qfim_root = Path(qfim_result["qfim_results_dir"])
                with np.load(qfim_root / "qfim_random_points_keep0123.npz") as archive:
                    self.assertEqual(archive["num_params_per_layer"].item(), 60)
                    self.assertEqual(archive["L1_theta"].shape, (2, 60))
                    random_thetas = archive["L1_theta"].copy()
                with np.load(hessian_result["hessian_result_path"]) as archive:
                    np.testing.assert_array_equal(archive["L1_theta"], random_thetas)
                    self.assertEqual(archive["L1_hessian"].shape, (2, 60, 60))

                energies, _, metadata = load_unitary_final_energies(
                    energy_path, expected_h_param=0.3,
                )
                self.assertEqual(energies[1].shape, (2,))
                self.assertEqual(metadata["num_params_per_layer"], 60)
                loaded_hessian = load_unitary_hessian_result(
                    hessian_result["hessian_results_dir"], expected_h_param=0.3,
                )
                self.assertEqual(loaded_hessian["layers"], [1])

                # Plot saved archives only. Any accidental numerical rerun fails.
                with (
                    patch.object(vqe, "run_vqe_optimization", side_effect=AssertionError("retraining")),
                    patch.object(qfim, "run_random_qfim_analysis", side_effect=AssertionError("QFIM rerun")),
                    patch.object(hessian, "run_random_hessian_analysis", side_effect=AssertionError("Hessian rerun")),
                ):
                    visualize.run_unitary_pqc_visualization(h_param=0.3)
                    visualize.run_unitary_hessian_visualization(
                        h_param=0.3,
                        results_dir=hessian_result["hessian_results_dir"],
                        figures_dir=Path(common.hessian_fig_dir),
                    )
                self.assertEqual(hashlib.sha256(energy_path.read_bytes()).digest(), before)
                self.assertTrue(list(root.rglob("*.pdf")) + list(root.rglob("*.png")))
                self.assertTrue((Path(common.energy_results_dir) / "gap_normalized_energy_statistics.npz").is_file())
        common.jax.clear_caches()


if __name__ == "__main__":
    unittest.main()
