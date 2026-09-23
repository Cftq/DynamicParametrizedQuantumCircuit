"""Archive and drawing regression checks for the measured-1 U3-Cartan circuit."""

import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))


class MeasuredUnitaryDrawingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drawer = importlib.import_module(
            "unitary_pqc_measured_1_overparam_draw_circuits"
        )

    def test_layer_has_all_cartan_blocks_and_unchanged_feed_forward(self):
        theta = np.arange(124, dtype=np.float64) / 10.0
        circuit = self.drawer.create_unitary_pqc(theta, num_layers=2)
        self.assertEqual(circuit.count_ops()["rxx"], 8)
        self.assertEqual(circuit.count_ops()["ryy"], 8)
        self.assertEqual(circuit.count_ops()["rzz"], 8)
        self.assertEqual(circuit.num_qubits, 5)
        self.assertEqual(circuit.num_clbits, 0)
        for layer_index in range(2):
            offset = layer_index * 63
            layer_operations = circuit.data[offset:offset + 63]
            for block_index, pair in enumerate(self.drawer.LAYER_PAIRS):
                block = layer_operations[15 * block_index:15 * (block_index + 1)]
                for instruction in block[6:9]:
                    wires = tuple(
                        circuit.find_bit(qubit).index
                        for qubit in instruction.qubits
                    )
                    self.assertEqual(wires, pair)
            values = [float(item.operation.params[0]) for item in layer_operations]
            np.testing.assert_array_equal(
                values[:60], theta[layer_index * 62:layer_index * 62 + 60]
            )
            varphi, phi = theta[layer_index * 62 + 60:layer_index * 62 + 62]
            np.testing.assert_array_equal(values[-3:], [varphi, 2 * phi, varphi])

    def test_hiding_parameters_preserves_topology_and_original_values(self):
        circuit = self.drawer.create_unitary_pqc(np.linspace(0.1, 1.1, 62), 1)
        numeric_values = [tuple(item.operation.params) for item in circuit.data]
        hidden = self.drawer.make_parameter_free_qiskit_for_drawing(circuit)
        self.assertEqual(hidden.count_ops(), circuit.count_ops())
        self.assertEqual(len(hidden.data), len(circuit.data))
        for original, drawing in zip(circuit.data, hidden.data):
            self.assertEqual(original.operation.name, drawing.operation.name)
            self.assertEqual(
                [circuit.find_bit(qubit).index for qubit in original.qubits],
                [hidden.find_bit(qubit).index for qubit in drawing.qubits],
            )
            self.assertEqual(drawing.operation.params[0].name, "")
            if drawing.operation.name in ("ryy", "rzz"):
                self.assertEqual(
                    drawing.operation.label,
                    self.drawer.PARAMETER_FREE_GATE_LABELS[drawing.operation.name],
                )
        self.assertEqual(
            numeric_values, [tuple(item.operation.params) for item in circuit.data]
        )

    def test_loads_62_parameter_archives_and_rejects_old_14_parameter_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "vqe_optimization_results.npz"
            for parameters_per_layer in (14, 62):
                with self.subTest(parameters_per_layer=parameters_per_layer):
                    theta = np.linspace(0.0, 1.0, parameters_per_layer)
                    np.savez(
                        archive_path,
                        ansatz=self.drawer.ANSATZ_NAME,
                        measurement_outcome=1,
                        num_params_per_layer=parameters_per_layer,
                        layers=np.array([1]),
                        L1_best_theta=theta,
                    )
                    if parameters_per_layer == 14:
                        with self.assertRaisesRegex(ValueError, "num_params_per_layer"):
                            self.drawer.load_best_theta_by_layer(archive_path)
                    else:
                        loaded = self.drawer.load_best_theta_by_layer(archive_path)
                        np.testing.assert_array_equal(loaded[1]["theta"], theta)

    def test_default_paths_include_u3_cartan_variant(self):
        args = self.drawer._parse_cli_args([])
        self.assertEqual(args.input_path.parents[3].name, "u3_cartan")
        self.assertEqual(args.output_dir.parent.parent.name, "u3_cartan")


if __name__ == "__main__":
    unittest.main()
