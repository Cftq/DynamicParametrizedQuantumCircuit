"""Compare generic and elementwise kernels in the actual DPQC VQE circuits.

Only two fixed parameter trials and three Adam updates are evaluated. The
training/archive writer is never called, and imports happen in a temporary
working directory. Run in the project environment containing TensorCircuit
and Optax; a lightweight environment reports an explicit dependency skip.
"""

import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


_MODULE_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _MODULE_DIR.parent / "common"
_DEPENDENCIES = ("jax", "numpy", "optax", "tensorcircuit")
_MISSING = [name for name in _DEPENDENCIES if importlib.util.find_spec(name) is None]


@unittest.skipIf(
    bool(_MISSING),
    "Density integration requires JAX, NumPy, Optax and TensorCircuit; missing: "
    + ", ".join(_MISSING),
)
class DPQCDensityIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dpqc_density_integration_")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        previous_cwd = Path.cwd()
        os.chdir(self.directory)
        self.addCleanup(os.chdir, previous_cwd)

        path_patch = patch.object(
            sys, "path", [str(_MODULE_DIR), str(_COMMON_DIR), *sys.path]
        )
        path_patch.start()
        self.addCleanup(path_patch.stop)
        environment_patch = patch.dict(os.environ, {"DPQC_DENSITY_KERNEL": "generic"})
        environment_patch.start()
        self.addCleanup(environment_patch.stop)

        cfg = importlib.import_module("config_overparam")
        config_patch = patch.multiple(
            cfg, H_PARAM=0.1, NUM_RUNS=2, STEPS=3, SAMPLE_EVERY=1,
            VQE_BATCH_SIZE=2, VQE_MAX_LAYER=2, VQE_DENSE_UNTIL_LAYER=2,
        )
        config_patch.start()
        self.addCleanup(config_patch.stop)

        # A distinct module object keeps the reset installation and mutable
        # kernel branch separate from other imports/tests in this process.
        spec = importlib.util.spec_from_file_location(
            "_density_integration_vqe", _MODULE_DIR / "DPQC_overparam_vqe.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.jax = self.module.jax
        self.jnp = self.module.jnp
        self.np = self.module.np
        self.addCleanup(self.jax.clear_caches)
        self.optax = importlib.import_module("optax")

    def _evaluate_branch(self, theta, layers, elementwise):
        module, jax, jnp, np = self.module, self.jax, self.jnp, self.np
        module.USE_ELEMENTWISE_DENSITY_KERNELS = elementwise
        # Branch selection is a Python decision at trace time. Clearing all
        # caches and creating fresh wrappers prevents reusing the other branch.
        jax.clear_caches()

        def state_one(parameters):
            return module.rho_keep_sequential_dpqc(parameters, n_layer=layers)

        def energy_one(parameters):
            return module.energy_from_rho_keep(state_one(parameters))

        with patch.object(module, "apply_rz_density", wraps=module.apply_rz_density) as rz, patch.object(
            module, "apply_rxx_density", wraps=module.apply_rxx_density
        ) as rxx, patch.object(
            module, "apply_unitary_on_rho", wraps=module.apply_unitary_on_rho
        ) as generic:
            states = jax.jit(jax.vmap(state_one))(theta)
            state_jacobians = jax.jit(jax.vmap(jax.jacfwd(state_one)))(theta)
            energies, gradients = jax.jit(jax.vmap(jax.value_and_grad(energy_one)))(theta)
            # Use the production scan/vmap implementation and the same Adam
            # state/update rules that a full VQE job would use.
            runner = module.make_vqe_batch_runner(
                layers,
                num_steps=3,
                sample_iterations=np.arange(4, dtype=np.int64),
                optimizer=self.optax.adam(learning_rate=0.015),
            )
            optimizer_outputs = runner(jnp.asarray(theta, dtype=jnp.float64))
            result = jax.device_get((
                states, state_jacobians, energies, gradients, optimizer_outputs,
            ))

            if elementwise:
                self.assertGreater(rz.call_count, 0)
                self.assertGreater(rxx.call_count, 0)
                generic.assert_not_called()
            else:
                rz.assert_not_called()
                rxx.assert_not_called()
                self.assertGreater(generic.call_count, 0)

        return result

    def _compare_family(self, *, reset):
        np, jnp = self.np, self.jnp
        if reset:
            model = importlib.import_module("dpqc_reset_model")
            model._install_reset_model(self.module)
            self.assertEqual(self.module.n_param_per_layer, 12)
        else:
            self.assertEqual(self.module.n_param_per_layer, 14)

        for layers in (1, 2):
            with self.subTest(layers=layers):
                rng = np.random.default_rng(510 + layers + 100 * int(reset))
                theta = jnp.asarray(
                    rng.uniform(-1.1, 1.1, size=(2, self.module.n_param_per_layer * layers)),
                    dtype=jnp.float64,
                )
                generic = self._evaluate_branch(theta, layers, False)
                elementwise = self._evaluate_branch(theta, layers, True)
                states, state_jacobians, energies, gradients, optimized = elementwise
                self.assertEqual(states.shape, (2, 32, 32))
                self.assertEqual(states.dtype, np.complex128)
                self.assertEqual(state_jacobians.shape, (2, 32, 32, theta.shape[1]))
                self.assertEqual(gradients.shape, theta.shape)
                for actual, expected in zip(elementwise[:4], generic[:4]):
                    self.assertTrue(np.all(np.isfinite(actual)))
                    np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=3e-12)
                np.testing.assert_allclose(
                    states, states.conj().swapaxes(-1, -2), atol=2e-13
                )
                np.testing.assert_allclose(np.trace(states, axis1=-2, axis2=-1), 1.0, atol=2e-13)
                self.assertGreaterEqual(float(np.linalg.eigvalsh(states).min()), -2e-12)

                theta_final, energy_trace, gradient_trace, theta_samples = optimized
                self.assertEqual(theta_final.shape, theta.shape)
                self.assertEqual(energy_trace.shape, (2, 4))
                self.assertEqual(gradient_trace.shape, (2, 4))
                self.assertEqual(theta_samples.shape, (2, 4, theta.shape[1]))
                np.testing.assert_allclose(energy_trace[:, 0], energies, atol=3e-12)
                np.testing.assert_allclose(theta_samples[:, 0], theta, atol=0.0)
                for index, (actual, expected) in enumerate(zip(optimized, generic[4])):
                    self.assertTrue(np.all(np.isfinite(actual)))
                    # Adam divides nearly-zero gradients by its small epsilon;
                    # angle comparisons allow the resulting roundoff, while
                    # energy and gradient histories keep tighter tolerances.
                    atol = 3e-9 if index in (0, 3) else 3e-11
                    np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=atol)

        self.assertEqual(list(self.directory.rglob("*.npz")), [])

    def test_dpqc_full_states_derivatives_and_adam_match(self):
        self._compare_family(reset=False)

    def test_reset_full_states_derivatives_and_adam_match(self):
        self._compare_family(reset=True)


if __name__ == "__main__":
    unittest.main()
