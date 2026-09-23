"""Numerical U3-Cartan regressions against independent Qiskit circuits.

These tests exercise small circuit states and derivatives without training,
drawing circuits, or saving analysis archives. TensorCircuit is optional.
"""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


_MODULE_DIR = Path(__file__).resolve().parent
for directory in (_MODULE_DIR, _MODULE_DIR.parent / "common"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


_MISSING = [
    name for name in ("numpy", "jax", "qiskit", "matplotlib", "tqdm")
    if importlib.util.find_spec(name) is None
]


@unittest.skipIf(bool(_MISSING), "Missing numerical dependencies: " + ", ".join(_MISSING))
class MeasuredUnitaryCartanNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator, Statevector

        import unitary_pqc_measured_1_model as model
        import unitary_pqc_measured_1_overparam_common as common

        jax.config.update("jax_enable_x64", True)
        cls.jax, cls.jnp, cls.np = jax, jnp, np
        cls.QuantumCircuit = QuantumCircuit
        cls.Operator, cls.Statevector = Operator, Statevector
        cls.model, cls.common = model, common
        with patch.object(common, "_ensure_unitary_result_dirs"):
            common.configure_unitary_pqc_overparam(h_value=0.3)

    @classmethod
    def tearDownClass(cls):
        cls.jax.clear_caches()

    @staticmethod
    def _append_reference_block(circuit, angles, a, b):
        """Use explicit Qiskit gates, independent of production builders."""
        circuit.ry(angles[0], a)
        circuit.rz(angles[1], a)
        circuit.ry(angles[2], a)
        circuit.ry(angles[3], b)
        circuit.rz(angles[4], b)
        circuit.ry(angles[5], b)
        circuit.rxx(angles[6], a, b)
        circuit.ryy(angles[7], a, b)
        circuit.rzz(angles[8], a, b)
        circuit.ry(angles[9], a)
        circuit.rz(angles[10], a)
        circuit.ry(angles[11], a)
        circuit.ry(angles[12], b)
        circuit.rz(angles[13], b)
        circuit.ry(angles[14], b)

    def _reference_state(self, theta, layers):
        # Numerical wire 0 is the most significant bit; Qiskit wire 0 is
        # the least significant bit. Reverse wire labels to match bases.
        theta = self.np.asarray(theta)
        circuit = self.QuantumCircuit(5)
        for layer in range(layers):
            for block, (q0, q1) in enumerate(((1, 3), (2, 3), (0, 2), (0, 4))):
                offset = 62 * layer + 15 * block
                self._append_reference_block(
                    circuit, theta[offset:offset + 15], 4 - q0, 4 - q1,
                )
            varphi, phi = theta[62 * layer + 60:62 * (layer + 1)]
            circuit.rz(varphi, 0)
            circuit.rx(2 * phi, 0)
            circuit.rz(varphi, 0)
        return self.np.asarray(self.Statevector.from_instruction(circuit).data)

    def _reference_energy(self, theta, layers):
        state = self._reference_state(theta, layers)
        return self.np.vdot(state, self.np.asarray(self.common.H_matrix) @ state).real

    def test_pair_unitary_matches_independent_gate_sequence(self):
        np = self.np
        angles = np.random.default_rng(521).uniform(-2.0, 2.0, 15)
        circuit = self.QuantumCircuit(2)
        self._append_reference_block(circuit, angles, 1, 0)
        expected = np.asarray(self.Operator(circuit).data)
        actual = np.asarray(self.model.unitary_block_matrix(self.jnp.asarray(angles), self.jnp))
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)
        np.testing.assert_allclose(actual.conj().T @ actual, np.eye(4), atol=2e-13)
        np.testing.assert_allclose(np.linalg.det(actual), 1.0, atol=2e-13)

    def test_one_and_two_layer_states_match_independent_circuits(self):
        from unitary_hessian_results import PARAMETERS_PER_LAYER

        np = self.np
        self.assertEqual(self.model.NUM_PARAMS_PER_LAYER, PARAMETERS_PER_LAYER)
        self.assertEqual(self.common.num_params_per_layer, 62)
        for layers in (1, 2):
            theta = np.random.default_rng(702 + layers).uniform(-2.0, 2.0, 62 * layers)
            expected = self._reference_state(theta, layers)
            actual = np.asarray(self.common.statevector_sequential_unitary_pqc(theta, layers))
            with self.subTest(layers=layers):
                np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)
                np.testing.assert_allclose(np.vdot(actual, actual), 1.0, atol=2e-13)
                reference_matrix = expected.reshape(16, 2)
                expected_reduced = reference_matrix @ reference_matrix.conj().T
                actual_reduced = np.asarray(self.common.rho_keep_sequential_unitary_pqc(theta, layers))
                np.testing.assert_allclose(actual_reduced, expected_reduced, atol=2e-13, rtol=2e-12)

    def test_state_jacobian_matches_finite_differences_for_new_angles(self):
        jax, jnp, np = self.jax, self.jnp, self.np
        theta = np.random.default_rng(819).uniform(-1.5, 1.5, 62)
        state = lambda angles: self.common.statevector_sequential_unitary_pqc(angles, 1)
        jacobian = np.asarray(jax.jit(jax.jacfwd(state))(jnp.asarray(theta)))
        self.assertEqual(jacobian.shape, (32, 62))
        self.assertTrue(np.isfinite(jacobian).all())
        step = 1e-5
        # Cover every block position, YY/ZZ, both post-rotation triples,
        # and the shared Rz angle and doubled Rx feed-forward angle.
        indices = (*range(15), 22, 23, 29, 37, 38, 44, *range(51, 62))
        for index in indices:
            plus, minus = theta.copy(), theta.copy()
            plus[index] += step
            minus[index] -= step
            expected = (self._reference_state(plus, 1) - self._reference_state(minus, 1)) / (2 * step)
            with self.subTest(index=index):
                np.testing.assert_allclose(jacobian[:, index], expected, atol=2e-10, rtol=2e-8)

    def test_energy_gradient_qfim_and_hessian_support_62_angles(self):
        import unitary_pqc_measured_1_overparam_hessian as hessian_stage

        jax, jnp, np = self.jax, self.jnp, self.np
        rng = np.random.default_rng(934)
        theta = jnp.asarray(rng.uniform(-1.3, 1.3, 62))
        energy = self.common.make_energy_fn_for_layer(1)
        value, gradient = jax.jit(jax.value_and_grad(energy))(theta)
        self.assertEqual(gradient.shape, (62,))
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())
        np.testing.assert_allclose(value, self._reference_energy(np.asarray(theta), 1), atol=2e-13)

        direction = rng.normal(size=62)
        direction /= np.linalg.norm(direction)
        step = 1e-5
        directional_gradient = (
            self._reference_energy(theta + step * direction, 1)
            - self._reference_energy(theta - step * direction, 1)
        ) / (2 * step)
        np.testing.assert_allclose(direction @ np.asarray(gradient), directional_gradient, atol=2e-10, rtol=2e-8)

        state = lambda angles: self.common.statevector_sequential_unitary_pqc(angles, 1)
        rho = lambda angles: self.common.rho_keep_sequential_unitary_pqc(angles, 1)
        qfim_functions = (
            self.common.make_pure_state_qfim_fn(state),
            self.common.make_mixed_state_qfim_fn(rho, eig_sum_eps=1e-12, jvp_chunk=16),
        )
        for kind, qfim in zip(("pure", "reduced"), qfim_functions):
            matrix = np.asarray(qfim(theta))
            with self.subTest(qfim=kind):
                self.assertEqual(matrix.shape, (62, 62))
                self.assertTrue(np.isfinite(matrix).all())
                np.testing.assert_allclose(matrix, matrix.T, atol=2e-12)
                self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -2e-10)

        matrix = np.asarray(hessian_stage.make_energy_hessian_fn_for_layer(1)(theta))
        self.assertEqual(matrix.shape, (62, 62))
        self.assertTrue(np.isfinite(matrix).all())
        np.testing.assert_allclose(matrix, matrix.T, atol=2e-12)
        step = 2e-4
        directional_curvature = (
            self._reference_energy(theta + step * direction, 1)
            - 2 * self._reference_energy(theta, 1)
            + self._reference_energy(theta - step * direction, 1)
        ) / step**2
        np.testing.assert_allclose(direction @ matrix @ direction, directional_curvature, atol=2e-7, rtol=2e-6)

    @unittest.skipUnless(importlib.util.find_spec("tensorcircuit"), "Optional TensorCircuit is not installed")
    def test_tensorcircuit_builder_matches_statevector(self):
        np = self.np
        for layers in (1, 2):
            theta = np.random.default_rng(1105 + layers).uniform(-2.0, 2.0, 62 * layers)
            circuit = self.common.create_unitary_pqc(self.jnp.asarray(theta), layers, 5)
            with self.subTest(layers=layers):
                np.testing.assert_allclose(
                    np.asarray(circuit.state()), self._reference_state(theta, layers),
                    atol=2e-13, rtol=2e-12,
                )


if __name__ == "__main__":
    unittest.main()
