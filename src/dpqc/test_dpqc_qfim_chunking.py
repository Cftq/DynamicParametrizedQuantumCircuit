"""Check chunked QFIM matrices against full-Jacobian SLD calculations.

The production factories are nested inside ``run_qfim``. Extract their AST
definitions into the actual stage's numerical namespace so these tests never
call its archive writer, load training histories, or run VQE. Imports happen
inside a temporary working directory because the stage creates output folders.
"""

import ast
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
_STAGE_PATH = _MODULE_DIR / "DPQC_overparam_qfim.py"
_DEPENDENCIES = ("jax", "numpy", "tensorcircuit")
_MISSING = [name for name in _DEPENDENCIES if importlib.util.find_spec(name) is None]
_JOINT_FACTORY = "make_joint_reduced_qfim_data_fn_for_layer_sequential"
_MIXED_FACTORY = "make_mixed_state_qfim_matrix_fn_for_layer_sequential"


@unittest.skipIf(
    bool(_MISSING),
    "QFIM chunking tests require JAX, NumPy and TensorCircuit; missing: "
    + ", ".join(_MISSING),
)
class DPQCQFIMChunkingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dpqc_qfim_chunking_")
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

        # An isolated module object prevents the reset model from changing
        # an ordinary-DPQC stage imported by another test in this process.
        spec = importlib.util.spec_from_file_location(
            "_qfim_chunking_stage", _STAGE_PATH
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.jax = self.module.jax
        self.jnp = self.module.jnp
        self.np = self.module.np
        self.addCleanup(self.jax.clear_caches)

        for name in ("run_qfim", "_load_saved_vqe_samples", "save_npz_result"):
            forbidden = patch.object(
                self.module,
                name,
                side_effect=AssertionError("Numerical tests must not run archive stages"),
            )
            forbidden.start()
            self.addCleanup(forbidden.stop)

    def _production_factory(self, factory_name):
        tree = ast.parse(_STAGE_PATH.read_text(encoding="utf-8"))
        run_definition = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_qfim"
        )
        definition = next(
            node for node in run_definition.body
            if isinstance(node, ast.FunctionDef) and node.name == factory_name
        )
        namespace = dict(vars(self.module))
        cfg = self.module.cfg
        namespace.update(
            RED_JVP_CHUNK=cfg.RED_JVP_CHUNK,
            EIG_SUM_EPS=cfg.EIG_SUM_EPS,
            QFIM_EFFECTIVE_RANK_THRESHOLD=cfg.QFIM_EFFECTIVE_RANK_THRESHOLD,
        )
        extracted = ast.Module(body=[definition], type_ignores=[])
        exec(compile(extracted, str(_STAGE_PATH), "exec"), namespace)
        return namespace[factory_name]

    def _reference_sld_qfim(self, rho, derivatives):
        """Solve the SLD equation and contract Tr(rho L_i L_j) with NumPy.

        The production path forms weighted derivative features and their Gram
        matrix. This reference instead reconstructs each SLD operator in the
        original state basis from a full forward-mode state Jacobian.
        """
        np = self.np
        rho = 0.5 * (rho + rho.conj().T)
        derivatives = 0.5 * (
            derivatives + derivatives.conj().swapaxes(-2, -1)
        )
        eigenvalues, vectors = np.linalg.eigh(rho)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        pair_sums = eigenvalues[:, None] + eigenvalues[None, :]
        coefficients = np.zeros_like(pair_sums)
        np.divide(
            2.0,
            pair_sums,
            out=coefficients,
            where=pair_sums > self.module.cfg.EIG_SUM_EPS,
        )
        vectors_dagger = vectors.conj().T
        derivatives_eigenbasis = vectors_dagger @ derivatives @ vectors
        sld = vectors @ (derivatives_eigenbasis * coefficients) @ vectors_dagger
        fisher = np.real(np.einsum("ab,ibc,jca->ij", rho, sld, sld, optimize=True))
        rank = np.count_nonzero(
            eigenvalues > self.module.cfg.QFIM_EFFECTIVE_RANK_THRESHOLD
        )
        return 0.5 * (fisher + fisher.T), rank

    def _compare_family(self, *, reset):
        module, jax, jnp, np = self.module, self.jax, self.jnp, self.np
        if reset:
            model = importlib.import_module("dpqc_reset_model")
            model._install_reset_model(module)

        layers = 2
        n_params = module.n_param_per_layer * layers
        self.assertEqual(n_params, 24 if reset else 28)
        theta = jnp.asarray(
            np.random.default_rng(1941 + int(reset)).uniform(
                -1.3, 1.3, size=n_params
            ),
            dtype=jnp.float64,
        )

        def state(parameters):
            return module.rho_keep_sequential_dpqc(parameters, n_layer=layers)

        rho5, jacobian = jax.device_get(jax.jit(
            lambda parameters: (state(parameters), jax.jacfwd(state)(parameters))
        )(theta))
        derivatives5 = np.moveaxis(jacobian, -1, 0)
        # Independent NumPy partial trace over the final qubit.
        rho4 = np.trace(rho5.reshape(16, 2, 16, 2), axis1=1, axis2=3)
        derivatives4 = np.trace(
            derivatives5.reshape(n_params, 16, 2, 16, 2), axis1=2, axis2=4
        )
        reference4, rank4 = self._reference_sld_qfim(rho4, derivatives4)
        reference5, rank5 = self._reference_sld_qfim(rho5, derivatives5)
        factory = self._production_factory(_JOINT_FACTORY)

        # 16 leaves a partial final block in both models. An oversized
        # requested chunk must also retain exactly the valid parameter rows.
        for chunk in (1, 16, n_params + 5):
            with self.subTest(reset=reset, chunk=chunk):
                runner = jax.jit(factory(layers, jvp_chunk=chunk))
                fisher4, fisher5, actual_rank4, actual_rank5 = jax.device_get(
                    runner(theta)
                )
                self.assertEqual(int(actual_rank4), rank4)
                self.assertEqual(int(actual_rank5), rank5)
                for fisher, expected in (
                    (fisher4, reference4), (fisher5, reference5)
                ):
                    self.assertEqual(fisher.shape, (n_params, n_params))
                    self.assertEqual(fisher.dtype, np.dtype(np.float64))
                    self.assertTrue(np.all(np.isfinite(fisher)))
                    np.testing.assert_allclose(fisher, expected, rtol=5e-9, atol=2e-10)
                    np.testing.assert_allclose(fisher, fisher.T, rtol=0.0, atol=2e-13)
                    self.assertGreaterEqual(float(np.linalg.eigvalsh(fisher).min()), -2e-10)

                if chunk == 16:
                    # The compatibility single-state factory uses the same
                    # chunk mechanism but has its own numerical code path.
                    single_factory = self._production_factory(_MIXED_FACTORY)
                    for n_keep, reduction, expected in (
                        (4, module.rho4_from_rho5, fisher4),
                        (5, lambda rho: rho, fisher5),
                    ):
                        with self.subTest(legacy_n_keep=n_keep):
                            single_runner = jax.jit(single_factory(
                                layers,
                                rho_from_rho5_fn=reduction,
                                n_keep=n_keep,
                                jvp_chunk=chunk,
                            ))
                            actual = np.asarray(jax.device_get(single_runner(theta)))
                            np.testing.assert_allclose(
                                actual, expected, rtol=5e-9, atol=2e-10
                            )

        self.assertEqual(list(self.directory.rglob("*.npz")), [])

    def test_dpqc_chunk_sizes_match_full_jacobian_sld(self):
        self._compare_family(reset=False)

    def test_reset_chunk_sizes_match_full_jacobian_sld(self):
        self._compare_family(reset=True)


if __name__ == "__main__":
    unittest.main()
