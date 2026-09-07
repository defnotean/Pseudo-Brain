"""Unit tests for Hardware Topology & Device Auto-Resolution Subsystem."""

import unittest
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
            self.assertEqual(applied, 2)
            self.assertEqual(torch.get_num_threads(), 2)
        finally:
            configure_cpu_threading(orig_threads)

    def test_system_gpu_probing(self):
        """Verify probe_system_gpus executes safely without throwing."""
        info = probe_system_gpus()
        self.assertIn("os", info)
        self.assertIn("controllers", info)
        self.assertIsInstance(info["controllers"], list)


if __name__ == "__main__":
    unittest.main()
