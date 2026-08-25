"""Contracts for the V2.1 result/checkpoint verifier."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import torch

from verify_v21_predictive_trajectory_artifact import ARMS, file_sha256, verify


def _arm(name: str, checkpoint: dict[str, str]) -> dict[str, object]:
    predictive = name != "decision_only_zero_pe"
    return {
        "status": "completed",
        "init_param_digest": {"sha256": "same"},
        "max_abs_belief": 0.5,
        "max_abs_thought": 2.0,
        "initial_validation": {
            "components": {"next_latent": 1.0, "reward": 1.0, "hazard": 1.0},
        },
        "final_validation": {
            "components": {
                "next_latent": 0.5 if predictive else 1.0,
                "reward": 0.5 if predictive else 1.0,
                "hazard": 0.5 if predictive else 1.0,
            },
            "metrics": {
                "turn_accuracy": 0.5,
                "turn_samples": 20.0,
                "hazard_positives": 12.0,
                "hazard_positive_probability": 0.6,
                "hazard_negative_probability": 0.4,
            },
        },
        "checkpoint": checkpoint,
    }


class V21ArtifactVerifierContract(unittest.TestCase):
    def _artifact(self, directory: str) -> str:
        artifact = os.path.join(directory, "result.json")
        checkpoint_dir = artifact + ".checkpoints"
        os.makedirs(checkpoint_dir)
        arms = {}
        for name in sorted(ARMS):
            path = os.path.join(checkpoint_dir, f"{name}.pt")
            torch.save(
                {
                    "schema_version": 1,
                    "arm": name,
                    "seed": 42,
                    "dataset_manifest": "mini-manifest",
                    "model_state_dict": {"weight": torch.tensor([1.0])},
                },
                path,
            )
            arms[name] = _arm(
                name,
                {"path": f"/workspace/run/{name}.pt", "sha256": file_sha256(path)},
            )
        with open(artifact, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "preregistration": "2026-08-24-core-v2-predictive-trajectory-smoke-prereg",
                    "mode": "mini_integration",
                    "dataset_manifest": "mini-manifest",
                    "arms": arms,
                },
                handle,
                allow_nan=False,
            )
        return artifact

    def test_verifies_all_checkpoint_hashes_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = verify(self._artifact(directory), None, allow_mini=True)
        self.assertTrue(result["verified"])
        self.assertEqual(set(result["checkpoint_sha256"]), ARMS)

    def test_rejects_checkpoint_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = self._artifact(directory)
            path = os.path.join(
                artifact + ".checkpoints", "predictive_normal_pe.pt"
            )
            with open(path, "ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify(artifact, None, allow_mini=True)


if __name__ == "__main__":
    unittest.main()
