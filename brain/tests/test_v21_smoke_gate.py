"""Regression contracts for the Stage V2.1 smoke gate."""
from __future__ import annotations

import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from stage_v21_predictive_trajectory_smoke import smoke_gate


def _arm() -> dict[str, object]:
    return {
        "status": "completed",
        "init_param_digest": {"l2": 4.0, "n": 10, "sum": 2.0},
        "max_abs_belief": 0.5,
        "max_abs_thought": 10.0,
        "initial_validation": {
            "components": {"next_latent": 1.0, "reward": 1.0, "hazard": 1.0},
        },
        "final_validation": {
            "components": {"next_latent": 0.8, "reward": 0.8, "hazard": 0.8},
            "metrics": {
                "turn_accuracy": 0.5,
                "turn_samples": 20,
                "hazard_positives": 12,
                "hazard_positive_probability": 0.30,
                "hazard_negative_probability": 0.20,
            },
        },
    }


class V21SmokeGateContract(unittest.TestCase):
    def test_structured_matching_digests_pass(self) -> None:
        arms = {
            "decision_only_zero_pe": _arm(),
            "predictive_zero_pe": _arm(),
            "predictive_normal_pe": _arm(),
        }

        self.assertTrue(smoke_gate(arms))

    def test_digest_mismatch_fails_without_crashing(self) -> None:
        arms = {
            "decision_only_zero_pe": _arm(),
            "predictive_zero_pe": _arm(),
            "predictive_normal_pe": deepcopy(_arm()),
        }
        arms["predictive_normal_pe"]["init_param_digest"]["sum"] = 3.0

        self.assertFalse(smoke_gate(arms))


if __name__ == "__main__":
    unittest.main()
