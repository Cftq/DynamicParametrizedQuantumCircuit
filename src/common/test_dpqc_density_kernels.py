"""Compare analytic density kernels against independent full unitaries."""

import importlib.util
import itertools
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_SPEC = importlib.util.spec_from_file_location(
    "dpqc_density_kernels", Path(__file__).with_name("dpqc_density_kernels.py"),
)
kernels = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kernels)


def _pauli_product(num_qubits, wires, pauli):
    result = np.ones((1, 1), dtype=np.complex128)
    for wire in range(num_qubits):
        result = np.kron(result, pauli if wire in wires else np.eye(2))
    return result


def _density(num_qubits, seed=74):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(2**num_qubits,) * 2) + 1j * rng.normal(size=(2**num_qubits,) * 2)
    rho = matrix @ matrix.conj().T
    return rho / np.trace(rho)


def _direct_numpy(rho, theta, generator):
    unitary = np.cos(theta / 2) * np.eye(rho.shape[-1]) - 1j * np.sin(theta / 2) * generator
    return unitary @ rho @ unitary.conj().T


class DPQCDensityKernelTests(unittest.TestCase):
    def test_rz_matches_numpy_for_each_wire_and_half_angle_convention(self):
        for num_qubits in (2, 3):
            rho = _density(num_qubits)
            for wire, theta in itertools.product(range(num_qubits), (0.0, np.pi, -0.37, 2.4)):
                generator = _pauli_product(num_qubits, (wire,), np.diag([1.0, -1.0]))
                with self.subTest(num_qubits=num_qubits, wire=wire, theta=theta):
                    actual = kernels.apply_rz_density(rho, theta, wire, num_qubits)
                    np.testing.assert_allclose(actual, _direct_numpy(rho, theta, generator), atol=2e-15, rtol=2e-14)
                    self.assertEqual(actual.dtype, jnp.complex128)

    def test_rxx_matches_numpy_for_neighbor_and_nonadjacent_wire_pairs(self):
        x = np.array([[0.0, 1.0], [1.0, 0.0]])
        for num_qubits in (2, 3):
            rho = _density(num_qubits)
            for wires, theta in itertools.product(itertools.permutations(range(num_qubits), 2), (0.0, np.pi, -0.37, 2.4)):
                generator = _pauli_product(num_qubits, wires, x)
                with self.subTest(num_qubits=num_qubits, wires=wires, theta=theta):
                    actual = kernels.apply_rxx_density(rho, theta, wires, num_qubits)
                    np.testing.assert_allclose(actual, _direct_numpy(rho, theta, generator), atol=2e-15, rtol=2e-14)
                    self.assertEqual(actual.dtype, jnp.complex128)

    def test_rxx_interference_term_has_correct_sign_for_nonhermitian_input(self):
        rho = np.arange(16).reshape(4, 4).astype(np.complex128) * (0.07 + 0.03j)
        generator = _pauli_product(2, (0, 1), np.array([[0, 1], [1, 0]]))
        actual = kernels.apply_rxx_density(rho, 0.73, (0, 1), 2)
        np.testing.assert_allclose(actual, _direct_numpy(rho, 0.73, generator), atol=2e-15, rtol=2e-14)

    def test_jit_gradients_and_second_derivatives_match_full_matrix_reference(self):
        rho = jnp.asarray(_density(3))
        observable = jnp.asarray(_density(3, seed=92))
        cases = (
            (kernels.apply_rz_density, 1, np.diag([1, -1]), (1,)),
            (kernels.apply_rxx_density, (0, 2), np.array([[0, 1], [1, 0]]), (0, 2)),
        )
        for kernel, wires, pauli, product_wires in cases:
            generator = jnp.asarray(_pauli_product(3, product_wires, pauli))

            def objective(theta):
                return jnp.real(jnp.trace(observable @ kernel(rho, theta, wires, 3)))

            def reference(theta):
                unitary = jnp.cos(theta / 2) * jnp.eye(8) - 1j * jnp.sin(theta / 2) * generator
                return jnp.real(jnp.trace(observable @ unitary @ rho @ unitary.conj().T))

            with self.subTest(kernel=kernel.__name__):
                actual = jax.jit(jax.value_and_grad(objective))(jnp.float64(0.73))
                expected = jax.jit(jax.value_and_grad(reference))(jnp.float64(0.73))
                np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-13)
                actual_hessian = jax.jit(jax.grad(jax.grad(objective)))(jnp.float64(0.73))
                expected_hessian = jax.jit(jax.grad(jax.grad(reference)))(jnp.float64(0.73))
                np.testing.assert_allclose(actual_hessian, expected_hessian, atol=2e-15, rtol=2e-13)

    def test_vmap_trial_angles_and_leading_density_batches(self):
        rhos = np.stack([_density(2, seed=seed) for seed in (74, 75, 76)])
        angles = jnp.array([0.1, -0.7, 2.1], dtype=jnp.float64)
        cases = ((kernels.apply_rz_density, 1), (kernels.apply_rxx_density, (0, 1)))
        for kernel, wires in cases:
            with self.subTest(kernel=kernel.__name__):
                actual = jax.jit(jax.vmap(lambda rho, theta: kernel(rho, theta, wires, 2)))(rhos, angles)
                expected = np.stack([kernel(rho, theta, wires, 2) for rho, theta in zip(rhos, angles)])
                np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-14)
                batch = kernel(rhos, 0.4, wires, 2)
                expected_batch = np.stack([kernel(rho, 0.4, wires, 2) for rho in rhos])
                np.testing.assert_allclose(batch, expected_batch, atol=2e-15, rtol=2e-14)

    def test_density_physical_invariants_and_five_qubit_default(self):
        rho = _density(5)
        for actual in (
            kernels.apply_rz_density(rho, 0.81, 4),
            kernels.apply_rxx_density(rho, 0.81, (0, 4)),
        ):
            np.testing.assert_allclose(actual, actual.conj().T, atol=2e-15)
            np.testing.assert_allclose(np.trace(actual), 1.0, atol=2e-15)
            np.testing.assert_allclose(np.linalg.eigvalsh(actual), np.linalg.eigvalsh(rho), atol=2e-15)

    def test_invalid_shapes_and_wire_indices_are_rejected(self):
        rho = _density(2)
        for wire in (-1, 2):
            with self.subTest(wire=wire), self.assertRaises(ValueError):
                kernels.apply_rz_density(rho, 0.1, wire, 2)
        for wires in ((0, 0), (0,), (0, 1, 2), (0, 2)):
            with self.subTest(wires=wires), self.assertRaises(ValueError):
                kernels.apply_rxx_density(rho, 0.1, wires, 2)
        with self.assertRaises(ValueError):
            kernels.apply_rz_density(rho, 0.1, 0, 3)


if __name__ == "__main__":
    unittest.main()
