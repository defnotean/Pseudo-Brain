"""Unit tests for Temporal Persistence and Matching Inertia Diagnostics."""

import unittest
import torch

from irene_brain.evaluation.temporal_persistence_diagnostics import (
    evaluate_temporal_persistence,
)
from irene_brain.model.thought_mediated_model import build_resource_matched_thought_model


class TestTemporalPersistence(unittest.TestCase):
    def test_evaluate_persistence(self) -> None:
        model = build_resource_matched_thought_model(4)
        rep0 = evaluate_temporal_persistence(model, inertia_weight=0.0, num_episodes=2, episode_length=5)
        rep1 = evaluate_temporal_persistence(model, inertia_weight=0.5, num_episodes=2, episode_length=5)
        self.assertGreaterEqual(rep0.branch_identity_retention_rate_pct, 0.0)
        self.assertGreaterEqual(rep1.branch_identity_retention_rate_pct, 0.0)
        self.assertGreaterEqual(rep1.mean_hypothesis_lifetime_ticks, 1.0)


if __name__ == "__main__":
    unittest.main()
