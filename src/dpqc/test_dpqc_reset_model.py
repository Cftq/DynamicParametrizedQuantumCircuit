"""Tests for reset model identity and safe analysis without VQE dependencies."""

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


_MODULE_PATH = Path(__file__).with_name("dpqc_reset_model.py")
_SPEC = importlib.util.spec_from_file_location("reset_model_test", _MODULE_PATH)
_MODEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODEL)


class ResetModelTests(unittest.TestCase):
    def test_model_and_parameter_order_are_preserved(self):
        self.assertEqual(_MODEL.MODEL_ID, "dpqc_reset_fixed_rx_pi")
        self.assertEqual(_MODEL.OUTPUT_FAMILY, "dpqc_reset")
        self.assertEqual(_MODEL.NUM_TRAINABLE_FEED_FORWARD_PARAMS, 0)
        self.assertEqual(_MODEL.FIXED_FEED_FORWARD_RX_ANGLE, math.pi)
        self.assertEqual(_MODEL.num_trainable_parameters(2), 24)
        names = _MODEL.parameter_names(2)
        self.assertEqual(len(names), 24)
        self.assertEqual(len(set(names)), 24)
        self.assertEqual(
            names[:3],
            ("L1_B0_Rz_q0", "L1_B0_Rz_q1", "L1_B0_Rxx"),
        )
        self.assertEqual(names[-1], "L2_B3_Rxx")
        with self.assertRaises(ValueError):
            _MODEL.num_trainable_parameters(0)

    def test_import_does_not_load_numerical_or_training_modules(self):
        probe = "\n".join(
            [
                "import importlib.util, sys",
                "spec = importlib.util.spec_from_file_location('probe', sys.argv[1])",
                "module = importlib.util.module_from_spec(spec)",
                "spec.loader.exec_module(module)",
                "forbidden = ('jax', 'numpy', 'optax', 'tensorcircuit', "
                "'DPQC_overparam_vqe', 'DPQC_overparam_qfim', 'DPQC_overparam_hessian')",
                "assert not any(name == prefix or name.startswith(prefix + '.') "
                "for name in sys.modules for prefix in forbidden)",
            ]
        )
        completed = subprocess.run(
            [sys.executable, "-S", "-c", probe, str(_MODULE_PATH)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_analysis_config_does_not_change_training_batch_size(self):
        config = SimpleNamespace(H_PARAM=0.9, VQE_BATCH_SIZE=29)
        with patch.dict(sys.modules, {"config_overparam": config}):
            _MODEL._prepare_base_config(0.1)
            self.assertEqual(config.H_PARAM, 0.1)
            self.assertEqual(config.VQE_BATCH_SIZE, 29)
            _MODEL._prepare_base_config(0.5, 3)
            self.assertEqual(config.H_PARAM, 0.5)
            self.assertEqual(config.VQE_BATCH_SIZE, 3)

    def test_invalid_stage_is_rejected_before_loading_any_module(self):
        with patch.object(_MODEL, "_prepare_base_config") as configure, patch.object(
            _MODEL.importlib, "import_module"
        ) as import_module:
            with self.assertRaises(ValueError):
                _MODEL._load_base_stage_module("hessian", h_param=0.1)
        configure.assert_not_called()
        import_module.assert_not_called()

    def test_qfim_loads_only_qfim_and_does_not_set_training_batch_size(self):
        config = SimpleNamespace(H_PARAM=0.9, VQE_BATCH_SIZE=29)
        stage_module = SimpleNamespace()
        with patch.dict(sys.modules, {"config_overparam": config}), patch.object(
            _MODEL.importlib, "import_module", return_value=stage_module
        ) as import_module, patch.object(
            _MODEL.importlib, "reload", return_value=stage_module
        ) as reload_module:
            already_imported = "DPQC_overparam_qfim" in sys.modules
            actual = _MODEL._load_base_stage_module("qfim", h_param=0.1)
        self.assertIs(actual, stage_module)
        self.assertEqual(config.H_PARAM, 0.1)
        self.assertEqual(config.VQE_BATCH_SIZE, 29)
        if already_imported:
            reload_module.assert_called_once()
            import_module.assert_not_called()
        else:
            import_module.assert_called_once_with("DPQC_overparam_qfim")
            reload_module.assert_not_called()


class ResetMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="dpqc_reset_metadata_")
        self.addCleanup(temporary.cleanup)
        self.save_dir = Path(temporary.name) / "h_0.1"
        self.archive = (
            self.save_dir
            / "numerical_results"
            / "energy"
            / "vqe_optimization_histories.npz"
        )

    def test_fresh_analysis_directory_gets_metadata_without_vqe(self):
        path = _MODEL._ensure_model_metadata(self.save_dir, 0.1)
        self.assertTrue(path.is_file())
        self.assertFalse(self.archive.exists())
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8")),
            _MODEL._model_metadata(0.1),
        )
        _MODEL._validate_model_metadata(self.save_dir, 0.1)
        with self.assertRaisesRegex(FileNotFoundError, "reset_vqe.py"):
            _MODEL._validate_model_metadata(
                self.save_dir, 0.1, require_vqe_archive=True
            )

    def test_compatible_metadata_is_reused_without_rewriting(self):
        path = _MODEL._ensure_model_metadata(self.save_dir, 0.1)
        before = path.read_bytes()
        with patch.object(_MODEL, "_write_model_metadata") as write:
            self.assertEqual(
                _MODEL._ensure_model_metadata(self.save_dir, 0.1), path
            )
        write.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    def test_incompatible_existing_metadata_is_not_overwritten(self):
        path = _MODEL._ensure_model_metadata(self.save_dir, 0.1)
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["unitary_parameters_per_layer"] = 14
        path.write_text(json.dumps(metadata), encoding="utf-8")
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "unitary_parameters_per_layer"):
            _MODEL._ensure_model_metadata(self.save_dir, 0.1)
        self.assertEqual(path.read_bytes(), before)

    def test_different_hamiltonian_metadata_is_rejected(self):
        path = _MODEL._ensure_model_metadata(self.save_dir, 0.1)
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "h_param"):
            _MODEL._ensure_model_metadata(self.save_dir, 0.5)
        self.assertEqual(path.read_bytes(), before)

    def test_unidentified_vqe_archive_is_not_stamped_by_analysis(self):
        self.archive.parent.mkdir(parents=True)
        self.archive.write_bytes(b"existing training archive")
        with self.assertRaisesRegex(FileNotFoundError, "refusing to identify"):
            _MODEL._ensure_model_metadata(self.save_dir, 0.1)
        self.assertFalse(_MODEL._metadata_path(self.save_dir).exists())
        self.assertEqual(self.archive.read_bytes(), b"existing training archive")

    def test_explicit_vqe_requirement_accepts_archive_with_valid_metadata(self):
        _MODEL._ensure_model_metadata(self.save_dir, 0.1)
        self.archive.parent.mkdir(parents=True)
        self.archive.write_bytes(b"existing training archive")
        _MODEL._validate_model_metadata(
            self.save_dir, 0.1, require_vqe_archive=True
        )


if __name__ == "__main__":
    unittest.main()
