"""Regressions for the independent, 60-angle U3-Cartan unitary workflow.

Numerical references use explicit Qiskit gates. Archive and launcher checks
write only to temporary directories and never run the training pipeline.
"""

from contextlib import redirect_stdout
import importlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


_MODULE_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _MODULE_DIR.parent / "common"
for _directory in (_MODULE_DIR, _COMMON_DIR):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))
_PREFIX = "unitary_pqc_overparam_"
_MISSING_NUMERICAL = [
    name for name in ("jax", "qiskit", "matplotlib", "tqdm")
    if importlib.util.find_spec(name) is None
]


@unittest.skipIf(bool(_MISSING_NUMERICAL), "Missing dependencies: " + ", ".join(_MISSING_NUMERICAL))
class UnitaryCartanNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        cls.jax, cls.jnp = jax, jnp
        cls.QuantumCircuit, cls.Statevector = QuantumCircuit, Statevector
        cls.model = importlib.import_module("unitary_pqc_model")
        cls.common = importlib.import_module(_PREFIX + "common")
        jax.config.update("jax_enable_x64", True)
        with patch.object(cls.common, "_ensure_unitary_result_dirs"):
            cls.common.configure_unitary_pqc_overparam(h_value=0.3)

    @classmethod
    def tearDownClass(cls):
        cls.jax.clear_caches()

    def reference_state(self, theta, layers):
        circuit = self.QuantumCircuit(5)
        # The numerical model makes wire zero the most significant bit.
        for layer in range(layers):
            for block, (q0, q1) in enumerate(((1, 3), (2, 3), (0, 2), (0, 4))):
                p = theta[60 * layer + 15 * block:60 * layer + 15 * (block + 1)]
                a, b = 4 - q0, 4 - q1
                circuit.ry(p[0], a)
                circuit.rz(p[1], a)
                circuit.ry(p[2], a)
                circuit.ry(p[3], b)
                circuit.rz(p[4], b)
                circuit.ry(p[5], b)
                circuit.rxx(p[6], a, b)
                circuit.ryy(p[7], a, b)
                circuit.rzz(p[8], a, b)
                circuit.ry(p[9], a)
                circuit.rz(p[10], a)
                circuit.ry(p[11], a)
                circuit.ry(p[12], b)
                circuit.rz(p[13], b)
                circuit.ry(p[14], b)
        return np.asarray(self.Statevector.from_instruction(circuit).data)

    def reference_energy(self, theta):
        state = self.reference_state(np.asarray(theta), 1)
        return np.vdot(state, np.asarray(self.common.H_matrix) @ state).real

    def test_one_and_two_layer_states_match_independent_circuits(self):
        self.assertEqual(self.model.NUM_PARAMS_PER_LAYER, 60)
        self.assertEqual(self.model.ANSATZ_NAME, "unitary_pqc")
        self.assertEqual(self.model.OUTPUT_VARIANT, "u3_cartan")
        for layers in (1, 2):
            with self.subTest(layers=layers):
                theta = np.random.default_rng(811 + layers).uniform(-2.0, 2.0, 60 * layers)
                expected = self.reference_state(theta, layers)
                actual = np.asarray(self.common.statevector_sequential_unitary_pqc(theta, layers))
                np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)
                np.testing.assert_allclose(np.vdot(actual, actual), 1.0, atol=2e-13)
                bipartite = expected.reshape(16, 2)
                reduced = np.asarray(self.common.rho_keep_sequential_unitary_pqc(theta, layers))
                np.testing.assert_allclose(reduced, bipartite @ bipartite.conj().T, atol=2e-13)

    def test_measured_variant_has_independent_configuration_and_result_state(self):
        measured = importlib.import_module("unitary_pqc_measured_1_overparam_common")
        original_h = measured.h_param
        try:
            with patch.object(measured, "_ensure_unitary_result_dirs"):
                measured.configure_unitary_pqc_overparam(h_value=0.2)
            measured_hamiltonian = measured.H_matrix
            measured_save_dir = measured.save_dir
            with patch.object(self.common, "_ensure_unitary_result_dirs"):
                self.common.configure_unitary_pqc_overparam(h_value=0.3)
            self.assertEqual(measured.h_param, 0.2)
            self.assertIs(measured.H_matrix, measured_hamiltonian)
            self.assertEqual(measured.save_dir, measured_save_dir)
            self.assertEqual(measured.num_params_per_layer, 62)
            self.assertEqual(self.common.num_params_per_layer, 60)
            self.assertNotEqual(Path(measured.save_dir), Path(self.common.save_dir))
            self.assertEqual(Path(self.common.save_dir).parts[-3:-1], ("unitary_pqc", "u3_cartan"))
            for name in ("theta_history", "best_theta_by_layer", "energy_traces_by_layer"):
                self.assertIsNot(getattr(self.common, name), getattr(measured, name))
        finally:
            with patch.object(measured, "_ensure_unitary_result_dirs"):
                measured.configure_unitary_pqc_overparam(h_value=original_h)

    def test_energy_derivatives_and_qfim_use_60_angles(self):
        jax, jnp = self.jax, self.jnp
        rng = np.random.default_rng(1805)
        theta = jnp.asarray(rng.uniform(-1.4, 1.4, 60))
        energy = self.common.make_energy_fn_for_layer(1)
        value, gradient = jax.jit(jax.value_and_grad(energy))(theta)
        self.assertEqual(gradient.shape, (60,))
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())
        np.testing.assert_allclose(value, self.reference_energy(theta), atol=2e-13)
        direction = rng.normal(size=60)
        direction /= np.linalg.norm(direction)
        step = 1e-5
        finite_gradient = (
            self.reference_energy(theta + step * direction)
            - self.reference_energy(theta - step * direction)
        ) / (2 * step)
        np.testing.assert_allclose(direction @ np.asarray(gradient), finite_gradient, atol=2e-10, rtol=2e-6)

        state = lambda angles: self.common.statevector_sequential_unitary_pqc(angles, 1)
        rho = lambda angles: self.common.rho_keep_sequential_unitary_pqc(angles, 1)
        qfim_functions = (
            self.common.make_pure_state_qfim_fn(state),
            self.common.make_mixed_state_qfim_fn(rho, eig_sum_eps=1e-12, jvp_chunk=16),
        )
        for kind, qfim in zip(("pure", "reduced"), qfim_functions):
            with self.subTest(qfim=kind):
                matrix = np.asarray(qfim(theta))
                self.assertEqual(matrix.shape, (60, 60))
                self.assertTrue(np.isfinite(matrix).all())
                np.testing.assert_allclose(matrix, matrix.T, atol=2e-12)
                self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -2e-10)

        hessian_stage = importlib.import_module(_PREFIX + "hessian")
        hessian = np.asarray(hessian_stage.make_energy_hessian_fn_for_layer(1)(theta))
        self.assertEqual(hessian.shape, (60, 60))
        self.assertTrue(np.isfinite(hessian).all())
        np.testing.assert_allclose(hessian, hessian.T, atol=2e-12)
        step = 2e-4
        finite_curvature = (
            self.reference_energy(theta + step * direction)
            - 2 * self.reference_energy(theta)
            + self.reference_energy(theta - step * direction)
        ) / step**2
        np.testing.assert_allclose(direction @ hessian @ direction, finite_curvature, atol=2e-7, rtol=2e-6)


class UnitaryCartanArchiveAndDrawingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drawer = importlib.import_module(_PREFIX + "draw_circuits")
        cls.energy_reader = importlib.import_module("unitary_pqc_energy_results")
        cls.hessian_reader = importlib.import_module("unitary_pqc_hessian_results")

    def energy_fixture(self):
        return {
            "ansatz": "unitary_pqc", "output_variant": "u3_cartan",
            "num_params_per_layer": 60, "h_param": 0.3,
            "layers": np.asarray([1]), "num_runs": 2, "steps": 2,
            "smallest_eigval": -3.0,
            "L1_energy_traces": np.asarray([[-2.0, -1.0], [-1.5, -0.5]]),
            "L1_best_theta": np.linspace(-1.0, 1.0, 60),
        }

    def test_drawing_contains_all_cartan_blocks_without_feed_forward(self):
        theta = np.arange(120, dtype=float) / 100
        circuit = self.drawer.create_unitary_pqc(theta, 2)
        self.assertEqual(circuit.num_qubits, 5)
        self.assertEqual(circuit.num_clbits, 0)
        self.assertEqual(len(circuit.data), 120)
        self.assertEqual(dict(circuit.count_ops()), {"ry": 64, "rz": 32, "rxx": 8, "ryy": 8, "rzz": 8})
        for layer in range(2):
            operations = circuit.data[60 * layer:60 * (layer + 1)]
            np.testing.assert_allclose(
                [float(item.operation.params[0]) for item in operations],
                theta[60 * layer:60 * (layer + 1)],
            )
            for index, pair in enumerate(((1, 3), (2, 3), (0, 2), (0, 4))):
                for instruction in operations[15 * index + 6:15 * index + 9]:
                    self.assertEqual(tuple(circuit.find_bit(q).index for q in instruction.qubits), pair)
        root = self.drawer._default_result_root()
        self.assertEqual(root.parts[-3:-1], ("unitary_pqc", "u3_cartan"))

    def test_energy_and_drawing_read_60_angle_archives_and_reject_other_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vqe_optimization_results.npz"
            fixture = self.energy_fixture()
            np.savez(path, **fixture)
            original = path.read_bytes()
            final, ground, metadata = self.energy_reader.load_unitary_final_energies(path, expected_h_param=0.3)
            np.testing.assert_array_equal(final[1], [-1.0, -0.5])
            self.assertEqual(ground, -3.0)
            self.assertEqual(metadata["ansatz"], "unitary_pqc")
            self.assertEqual(metadata["num_params_per_layer"], 60)
            self.assertNotIn("measurement_outcome", metadata)
            loaded = self.drawer.load_best_theta_by_layer(path, [1])
            np.testing.assert_array_equal(loaded[1]["theta"], fixture["L1_best_theta"])
            self.assertEqual(path.read_bytes(), original)
            for changes in (
                {"ansatz": "unitary_pqc_measured_1", "measurement_outcome": 1},
                {"num_params_per_layer": 62, "L1_best_theta": np.zeros(62)},
                {"num_params_per_layer": 12, "L1_best_theta": np.zeros(12)},
            ):
                with self.subTest(changes=tuple(changes)):
                    np.savez(path, **{**fixture, **changes})
                    with self.assertRaises(ValueError):
                        self.energy_reader.load_unitary_final_energies(path, expected_h_param=0.3)
                    with self.assertRaises(ValueError):
                        self.drawer.load_best_theta_by_layer(path, [1])

    def test_hessian_reader_isolates_60_angle_family(self):
        reader = self.hessian_reader
        fixture = {
            "schema_version": 1, "analysis_kind": "random_points",
            "ansatz": "unitary_pqc", "output_variant": "u3_cartan", "h_param": 0.3,
            "layers": np.asarray([1]), "num_hessian_samples": 1,
            "hessian_sample_seed_base": 8, "hessian_rank_threshold": 1e-12,
            "hessian_rank_definition": reader.HESSIAN_RANK_DEFINITION,
            "hessian_condition_number_definition": reader.HESSIAN_CONDITION_NUMBER_DEFINITION,
            "num_params_per_layer": 60, "analysis_batch_size": 1,
            "L1_rank": np.asarray([2]), "L1_condition_number": np.asarray([2.0]),
            "L1_hessian": np.diag([2.0, -1.0] + [0.0] * 58)[None, :, :],
            "L1_theta": np.zeros((1, 60)),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "hessian_random_points.npz"
            np.savez(path, **fixture)
            original = path.read_bytes()
            result = reader.load_unitary_hessian_result(temporary, expected_h_param=0.3)
            self.assertEqual(result["output_family"], "unitary_pqc")
            self.assertEqual(result["hessian_by_layer"][1].shape, (1, 60, 60))
            np.testing.assert_array_equal(result["rank_by_layer"][1], [2])
            np.testing.assert_allclose(result["condition_by_layer"][1], [2.0])
            self.assertEqual(path.read_bytes(), original)
            for changes in ({"ansatz": "unitary_pqc_measured_1"}, {"num_params_per_layer": 62}):
                with self.subTest(changes=changes):
                    np.savez(path, **{**fixture, **changes})
                    with self.assertRaises(ValueError):
                        reader.load_unitary_hessian_result(temporary, expected_h_param=0.3)


class UnitaryCartanEntryPointTests(unittest.TestCase):
    def test_launcher_selects_new_stages_and_preserves_child_failure(self):
        launcher = importlib.import_module(_PREFIX + "compute")
        for selected, expected in (("analysis", ("qfim", "hessian")), ("all", ("vqe", "qfim", "hessian"))):
            with self.subTest(stage=selected), patch.dict(os.environ, {"DPQC_DEVICE": "cpu"}), patch.object(
                launcher.dpqc_wsl, "maybe_relaunch_in_wsl", return_value=None,
            ), patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run, redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main(["--stage", selected, "--random-only"]), 0)
            self.assertEqual([Path(call.args[0][1]).name for call in run.call_args_list],
                             [f"{_PREFIX}{stage}.py" for stage in expected])
            for call, stage in zip(run.call_args_list, expected):
                self.assertEqual("--random-only" in call.args[0], stage == "qfim")
        with patch.object(launcher.dpqc_wsl, "maybe_relaunch_in_wsl", return_value=None), patch.object(
            launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 19),
        ) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.main(["--stage", "all", "--device", "cpu"]), 19)
        self.assertEqual(run.call_count, 1)

    def test_stage_help_is_lightweight_in_script_and_package_forms(self):
        probe = r'''
import importlib.abc, pathlib, runpy, sys
blocked = {"jax", "jaxlib", "numpy", "scipy", "matplotlib", "qiskit", "optax", "tensorcircuit", "unitary_pqc_overparam_common"}
class RejectNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked or fullname.rsplit(".", 1)[-1] in blocked:
            raise AssertionError("Unexpected numerical import: " + fullname)
sys.meta_path.insert(0, RejectNumericalImports())
module_dir = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(module_dir.parent.parent))
for stage in ("compute", "vqe", "qfim", "hessian"):
    name = "unitary_pqc_overparam_" + stage
    script = module_dir / (name + ".py")
    runpy.run_path(str(script), run_name="import_probe_" + stage)
    for package in (False, True):
        sys.argv = [str(script), "--help"]
        try:
            if package:
                runpy.run_module("src.unitary_pqc." + name, run_name="__main__")
            else:
                runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 0, exc.code
        else:
            raise AssertionError("Help did not exit")
print("UNITARY_CARTAN_LIGHTWEIGHT_OK")
'''
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [sys.executable, "-c", probe, str(_MODULE_DIR)], cwd=temporary,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"}, capture_output=True,
                text=True, encoding="utf-8", timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("UNITARY_CARTAN_LIGHTWEIGHT_OK", result.stdout)
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
