from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from irene_brain.experiment import ExperimentManifest
from irene_brain.runtime.policy import ResourceDenied, ResourcePolicy


def make_manifest() -> ExperimentManifest:
    return ExperimentManifest(
        run_id="phase0-test",
        schema_version=1,
        created_at="2026-08-16T12:00:00.000000Z",
        code_hash="a" * 64,
        data_hash="b" * 64,
        environment_hash="c" * 64,
        checkpoint_hash=None,
        seed=7,
        config_hash="d" * 64,
        hardware={"cpu": "test", "logical_threads": 1},
        runtime={"python": "3.11", "mode": "deterministic"},
        precision="integer",
        allow_gpu=False,
        allow_capture=False,
        allow_hid_output=False,
        allow_background_threads=False,
        allow_network=False,
        allow_subprocess=False,
        allow_artifact_write=False,
        cpu_threads=1,
        status="completed",
        metrics={"replay_steps": 1000, "signed_zero": -0.0},
    )


class ExperimentManifestTests(unittest.TestCase):
    def test_canonical_round_trip_and_hash(self) -> None:
        manifest = make_manifest()
        restored = ExperimentManifest.from_json(manifest.canonical_json())

        self.assertEqual(restored, manifest)
        self.assertEqual(restored.manifest_hash, manifest.manifest_hash)
        self.assertEqual(restored.metrics["signed_zero"], 0.0)
        self.assertNotIn("-0.0", manifest.canonical_json())
        self.assertEqual(restored.resource_policy, ResourcePolicy.play_safe())

    def test_atomic_save_requires_explicit_write_capability(self) -> None:
        manifest = make_manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            with self.assertRaises(ResourceDenied):
                manifest.save(path, policy=ResourcePolicy.play_safe())
            self.assertFalse(path.exists())

            writable = ResourcePolicy(allow_artifact_write=True)
            manifest.save(path, policy=writable)
            loaded = ExperimentManifest.load(path)

        self.assertEqual(loaded, manifest)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentManifest.from_json('{"run_id":"one","run_id":"two"}')


if __name__ == "__main__":
    unittest.main()
