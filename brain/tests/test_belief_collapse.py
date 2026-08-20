"""Unit tests for Belief Collapse and Value of Information (VoI) Diagnostics."""

import unittest
import torch

from irene_brain.evaluation.belief_collapse_diagnostics import (
    evaluate_belief_collapse_and_voi,
)
from irene_brain.model.thought_mediated_model import build_resource_matched_thought_model


class TestBeliefCollapse(unittest.TestCase):
    def test_evaluate_belief_collapse(self) -> None:
        model = build_resource_matched_thought_model(4)
        rep = evaluate_belief_collapse_and_voi(model, num_episodes=5)
        self.assertEqual(rep.num_episodes_evaluated, 5)
        self.assertGreaterEqual(rep.pre_observation_entropy, 0.0)
        self.assertGreaterEqual(rep.post_observation_entropy, 0.0)
        self.assertGreaterEqual(rep.probe_wait_pct, 0.0)
        self.assertLessEqual(rep.probe_wait_pct, 100.0)


if __name__ == "__main__":
    unittest.main()
