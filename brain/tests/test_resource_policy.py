from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from irene_brain.runtime.policy import Capability, ResourceDenied, ResourcePolicy


class ResourcePolicyTests(unittest.TestCase):
    def test_default_policy_denies_every_expansive_capability(self) -> None:
        policy = ResourcePolicy.play_safe()
        for capability in Capability:
            with self.subTest(capability=capability):
                self.assertFalse(policy.allows(capability))
                with self.assertRaises(ResourceDenied):
                    policy.require(capability)

    def test_phase0_configuration_is_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        policy = ResourcePolicy.from_toml(
            root / "configs" / "experiment" / "phase0-smoke.toml"
        )

        self.assertEqual(policy, ResourcePolicy.play_safe())
        self.assertEqual(policy.process_environment()["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(policy.process_environment()["OMP_NUM_THREADS"], "1")

    def test_video_recording_requires_capture_and_artifact_capabilities(self) -> None:
        config = """
[experiment]
device = "cpu"
allow_capture = false

[runtime]
record_video = true
write_artifacts = false
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.toml"
            path.write_text(config, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires allow_capture"):
                ResourcePolicy.from_toml(path)


if __name__ == "__main__":
    unittest.main()
