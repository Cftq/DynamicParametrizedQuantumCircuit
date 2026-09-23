"""Numerical reset-circuit regressions against independently built Qiskit gates.

These tests require NumPy, JAX and Qiskit, but do not import a training stage,
need Optax/TensorCircuit, run optimization, or create result archives.
"""

from collections import Counter
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


_MODULE_DIR = Path(__file__).resolve().parent
for directory in (_MODULE_DIR, _MODULE_DIR.parent / "common"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import dpqc_reset_model as model


_MISSING = [
    name for name in ("numpy", "jax", "qiskit")
    if importlib.util.find_spec(name) is None
]


@unittest.skipIf(bool(_MISSING), "Missing numerical dependencies: " + ", ".join(_MISSING))
class ResetGateNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

        jax.config.update("jax_enable_x64", True)
        cls.jax, cls.jnp, cls.np = jax, jnp, np
        cls.QuantumCircuit = QuantumCircuit
        cls.DensityMatrix, cls.Statevector = DensityMatrix, Statevector
        cls.partial_trace = staticmethod(partial_trace)

    @classmethod
    def tearDownClass(cls):
        cls.jax.clear_caches()

    def _stage(self, *, elementwise=False):
        jax, jnp, np = self.jax, self.jnp, self.np

        def dense_conjugation(rho, unitary, wires, k):
            # Embed the small gate in the full Hilbert space independently of
            # the production tensor-axis permutation/contraction kernel.
            basis = np.arange(2**k)
            bits = (basis[:, None] >> np.arange(k - 1, -1, -1)) & 1
            selected = bits[:, wires] @ (1 << np.arange(len(wires) - 1, -1, -1))
            other = tuple(wire for wire in range(k) if wire not in wires)
            mask = np.all(bits[:, None, other] == bits[None, :, other], axis=-1)
            expanded = unitary[selected[:, None], selected[None, :]] * jnp.asarray(mask)
            return expanded @ rho @ expanded.conj().T

        initial = jnp.zeros((32, 32), dtype=jnp.complex128).at[0, 0].set(1.0)
        module = SimpleNamespace(
            jax=jax,
            jnp=jnp,
            REAL_DTYPE=jnp.float64,
            USE_ELEMENTWISE_DENSITY_KERNELS=elementwise,
            apply_unitary_on_rho=dense_conjugation,
            _RHO_KEEP_INIT=initial,
            _RHO_QUBIT_ZERO=jnp.array([[1, 0], [0, 0]], dtype=jnp.complex128),
            _hermitian=lambda rho: (rho + rho.conj().T) / 2,
            wrap_to_pi=lambda theta: (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi,
        )
        if elementwise:
            from dpqc_density_kernels import apply_rz_density, apply_rxx_density

            module.apply_rz_density = apply_rz_density
            module.apply_rxx_density = apply_rxx_density
        model._install_reset_model(module)
        return module

    def _reference(self, theta, layers):
        """Apply explicit Qiskit gates and an actual reset instruction.

        Physical wire q maps to Qiskit wire 4-q to match the numerical model's
        most-significant-bit-first density-matrix basis.
        """
        rho = self.DensityMatrix.from_label("00000")
        probabilities = []
        for layer in range(layers):
            circuit = self.QuantumCircuit(5)
            for block, (q0, q1) in enumerate(((1, 3), (2, 3), (0, 2), (0, 4))):
                angles = theta[60 * layer + 15 * block:60 * layer + 15 * (block + 1)]
                a, b = 4 - q0, 4 - q1
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
            rho = rho.evolve(circuit)
            probabilities.append(rho.probabilities([0])[1])
            reset = self.QuantumCircuit(5)
            reset.reset(0)
            rho = rho.evolve(reset)
        return self.np.asarray(rho.data), self.np.asarray(probabilities)

    def test_symbolic_gates_have_independent_parameters_and_fixed_reset(self):
        circuit = model.build_reset_circuit(2)
        self.assertEqual(circuit.num_qubits, 7)
        self.assertEqual(len(circuit.parameters), 120)
        self.assertEqual(circuit.count_ops(), Counter(ry=64, rz=32, rxx=8, ryy=8, rzz=8, cx=2, crx=2))
        occurrences = Counter()
        for instruction in circuit.data:
            for angle in instruction.operation.params:
                if hasattr(angle, "parameters"):
                    occurrences.update(angle.parameters)
        self.assertEqual(len(occurrences), 120)
        self.assertTrue(all(count == 1 for count in occurrences.values()))
        self.assertEqual({angle.name for angle in occurrences}, set(model.parameter_names(2)))
        for layer in range(2):
            cx, crx = circuit.data[layer * 62 + 60:layer * 62 + 62]
            self.assertEqual(cx.operation.name, "cx")
            self.assertEqual(crx.operation.name, "crx")
            self.assertEqual(crx.operation.params, [self.np.pi])
            self.assertEqual([circuit.find_bit(q).index for q in cx.qubits], [4, 5 + layer])
            self.assertEqual([circuit.find_bit(q).index for q in crx.qubits], [5 + layer, 4])

    def test_density_and_dilation_match_independent_reset_reference(self):
        np = self.np
        for layers in (1, 2):
            theta = np.random.default_rng(847 + layers).uniform(-2.0, 2.0, 60 * layers)
            expected, probabilities = self._reference(theta, layers)
            for elementwise in (False, True):
                with self.subTest(layers=layers, elementwise=elementwise):
                    module = self._stage(elementwise=elementwise)
                    actual = np.asarray(module.rho_keep_sequential_dpqc(theta, layers))
                    actual_p1 = np.asarray(module.ancilla_p1_sequential_dpqc(theta, layers))
                    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)
                    np.testing.assert_allclose(actual_p1, probabilities, atol=2e-13)
                    np.testing.assert_allclose(actual, actual.conj().T, atol=2e-13)
                    np.testing.assert_allclose(np.trace(actual), 1.0, atol=2e-13)
                    self.assertGreaterEqual(np.linalg.eigvalsh(actual).min(), -2e-13)
                    np.testing.assert_allclose(actual[1::2, :], 0.0, atol=2e-13)
                    np.testing.assert_allclose(actual[:, 1::2], 0.0, atol=2e-13)

            dilation = self.Statevector.from_instruction(model.build_reset_circuit(layers, theta))
            reduced = np.asarray(self.partial_trace(dilation, list(range(5, 5 + layers))).data)
            reverse = np.array([int(f"{value:05b}"[::-1], 2) for value in range(32)])
            reduced = reduced[np.ix_(reverse, reverse)]
            np.testing.assert_allclose(reduced, expected, atol=2e-13, rtol=2e-12)

    def test_jit_state_jacobian_matches_independent_finite_differences(self):
        jax, jnp, np = self.jax, self.jnp, self.np
        module = self._stage()
        theta = np.random.default_rng(937).uniform(-1.5, 1.5, 60)
        state = lambda angles: module.rho_keep_sequential_dpqc(angles, 1)
        actual = np.asarray(jax.jit(jax.jacfwd(state))(jnp.asarray(theta)))
        self.assertEqual(actual.shape, (32, 32, 60))
        self.assertTrue(np.isfinite(actual).all())
        step = 1e-5
        for index in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 37, 59):
            plus, minus = theta.copy(), theta.copy()
            plus[index] += step
            minus[index] -= step
            expected = (self._reference(plus, 1)[0] - self._reference(minus, 1)[0]) / (2 * step)
            with self.subTest(index=index):
                np.testing.assert_allclose(actual[:, :, index], expected, atol=2e-10, rtol=2e-8)
        elementwise = self._stage(elementwise=True)
        other = np.asarray(jax.jit(jax.jacfwd(
            lambda angles: elementwise.rho_keep_sequential_dpqc(angles, 1)
        ))(jnp.asarray(theta)))
        np.testing.assert_allclose(other, actual, atol=2e-13, rtol=2e-12)

    def test_old_or_shared_parameter_vectors_are_rejected(self):
        for layers, count in ((1, 12), (2, 24), (1, 9), (2, 60)):
            theta = self.np.zeros(count)
            with self.subTest(layers=layers, count=count):
                with self.assertRaisesRegex(ValueError, "theta must have shape"):
                    model.build_reset_circuit(layers, theta)
                with self.assertRaisesRegex(ValueError, "theta must have shape"):
                    self._stage().rho_keep_sequential_dpqc(theta, layers)

    def test_hessian_state_and_directional_curvature_match_reference(self):
        import DPQC_overparam_hessian as hessian

        jax, jnp, np = self.jax, self.jnp, self.np
        rng = np.random.default_rng(624)
        for layers in (1, 2):
            theta = rng.uniform(-1.3, 1.3, 60 * layers)
            expected, _ = self._reference(theta, layers)
            actual = np.asarray(hessian.full_state(theta, layers, output_family="dpqc_reset"))
            with self.subTest(layers=layers):
                np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-12)

        theta = rng.uniform(-1.3, 1.3, 60)
        direction = rng.normal(size=60)
        direction /= np.linalg.norm(direction)
        energy = hessian.make_energy_function(1, 0.1, output_family="dpqc_reset")
        hvp = hessian.make_hessian_vector_chunk_function(energy)
        actual = np.asarray(hvp(jnp.asarray(theta), jnp.asarray(direction[None, :])))[0]
        self.assertEqual(actual.shape, (60,))
        self.assertTrue(np.isfinite(actual).all())
        gradient = jax.jit(jax.grad(energy))
        step = 1e-5
        expected = (
            np.asarray(gradient(theta + step * direction))
            - np.asarray(gradient(theta - step * direction))
        ) / (2 * step)
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-7)

        hamiltonian = np.asarray(hessian.hamiltonian4(0.1))

        def reference_energy(angles):
            density, _ = self._reference(angles, 1)
            reduced = np.trace(density.reshape(16, 2, 16, 2), axis1=1, axis2=3)
            return np.trace(hamiltonian @ reduced).real

        np.testing.assert_allclose(float(energy(theta)), reference_energy(theta), atol=2e-13)
        step = 2e-4
        curvature = (
            reference_energy(theta + step * direction)
            - 2 * reference_energy(theta)
            + reference_energy(theta - step * direction)
        ) / step**2
        np.testing.assert_allclose(direction @ actual, curvature, atol=2e-7, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
