"""Unit tests for Hardware Topology & Device Auto-Resolution Subsystem."""

import unittest
import os
import torch

from irene_brain.device import (
    probe_system_gpus,
    resolve_optimal_device,
    is_directml_available,
    get_directml_device,
    get_device_telemetry,
    configure_cpu_threading,
)


class DeviceResolutionTests(unittest.TestCase):
    """Test device discovery, fallback, and telemetry."""

    def test_cpu_fallback_resolution(self):
        """Verify device resolution returns a valid torch.device and backend label."""
        dev, backend = resolve_optimal_device(preference="cpu")
        self.assertEqual(dev.type, "cpu")
        self.assertEqual(backend, "cpu")

    def test_explicit_preference_handling(self):
        """Verify fallback when requested accelerator is absent."""
        dev, backend = resolve_optimal_device(preference="dml")
        if not is_directml_available():
            self.assertEqual(dev.type, "cpu")
            self.assertEqual(backend, "cpu_fallback")
        else:
            self.assertEqual(backend, "directml")

    def test_telemetry_fields(self):
        """Verify telemetry dictionary contains expected diagnostic properties."""
        telemetry = get_device_telemetry()
        self.assertIn("active_device", telemetry)
        self.assertIn("backend", telemetry)
        self.assertIn("cpu_threads", telemetry)
        self.assertIn("has_amd_radeon", telemetry)
        self.assertIsInstance(telemetry["cpu_threads"], int)

    def test_cpu_threading_configuration(self):
        """Verify CPU threading configuration is applied."""
        orig_threads = torch.get_num_threads()
        try:
            applied = configure_cpu_threading(2)
            expected = 1 if os.environ.get("PSEUDO_BRAIN_CPU_ONLY") == "1" else 2
            self.assertEqual(applied, expected)
            self.assertEqual(torch.get_num_threads(), expected)
        finally:
            configure_cpu_threading(orig_threads)

    def test_system_gpu_probing(self):
        """Verify probe_system_gpus executes safely without throwing."""
        info = probe_system_gpus()
        self.assertIn("os", info)
        self.assertIn("controllers", info)
        self.assertIsInstance(info["controllers"], list)

    def test_directml_best_device_selection(self):
        """Verify resolve_optimal_device auto-resolves to dedicated DirectML GPU when available."""
        if is_directml_available():
            dev, backend = resolve_optimal_device()
            self.assertEqual(backend, "directml")
            self.assertEqual(dev.type, "privateuseone")
            # If AMD Radeon RX 9070 XT is present, verify it is selected (index 1)
            telemetry = get_device_telemetry()
            if any("9070" in d.get("name", "") for d in telemetry.get("directml_devices", [])):
                self.assertIn("9070", telemetry.get("directml_device_name", ""))
                self.assertEqual(telemetry.get("directml_device_index"), 1)

    def test_directml_indexed_preference(self):
        """Verify explicit index preference parsing (e.g., dml:0, dml:1)."""
        if is_directml_available():
            dev0, backend0 = resolve_optimal_device(preference="dml:0")
            self.assertEqual(backend0, "directml")
            self.assertEqual(str(dev0), "privateuseone:0")

            telemetry = get_device_telemetry()
            if telemetry.get("directml_device_count", 0) >= 2:
                dev1, backend1 = resolve_optimal_device(preference="dml:1")
                self.assertEqual(backend1, "directml")
                self.assertEqual(str(dev1), "privateuseone:1")

    def test_directml_tensor_computation(self):
        """Verify tensor allocation, matrix multiplication, and host sync on DirectML device."""
        if is_directml_available():
            dev, backend = resolve_optimal_device()
            self.assertEqual(backend, "directml")
            a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=dev)
            b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=dev)
            c = a @ b
            c_cpu = c.cpu()
            expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
            self.assertTrue(torch.allclose(c_cpu, expected))

    def test_directml_telemetry_fields(self):
        """Verify telemetry contains DirectML hardware metadata when DirectML is available."""
        if is_directml_available():
            telemetry = get_device_telemetry()
            self.assertTrue(telemetry["directml_available"])
            self.assertIn("directml_device_count", telemetry)
            self.assertIn("directml_devices", telemetry)
            self.assertIn("directml_device_index", telemetry)
            self.assertIn("directml_device_name", telemetry)
            self.assertGreaterEqual(telemetry["directml_device_count"], 1)


if __name__ == "__main__":
    unittest.main()
