"""Unit tests for Counterfactual Thought Transplant diagnostic suite."""

import unittest
import torch

from irene_brain.environments.phase2_suite import Phase2TaskEnvironment, make_family_suite, TaskFamily
from irene_brain.evaluation.counterfactual_transplant_diagnostics import (
    collect_conflicting_state_pairs,
    evaluate_counterfactual_thought_transplant,
)
from irene_brain.model.thought_mediated_model import build_resource_matched_thought_model


class TestCounterfactualThoughtTransplant(unittest.TestCase):
    def test_collect_conflicting_pairs(self) -> None:
        env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_B_PURSUIT_EVASION)[0])
        pairs = collect_conflicting_state_pairs(env, num_pairs=5, base_seed=8000)
        self.assertGreater(len(pairs), 0)
        for pair in pairs:
            self.assertNotEqual(pair.recipient_optimal_action, pair.donor_optimal_action)
            self.assertEqual(pair.recipient_rgb.shape, (1, 3, 32, 32))
            self.assertEqual(pair.donor_rgb.shape, (1, 3, 32, 32))

    def test_evaluate_transplant(self) -> None:
        model = build_resource_matched_thought_model(4)
        env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_B_PURSUIT_EVASION)[0])
        report = evaluate_counterfactual_thought_transplant(model, env, num_pairs=3)
        self.assertGreaterEqual(report.num_pairs_evaluated, 1)
        self.assertGreaterEqual(report.donor_action_adoption_rate_pct, 0.0)
        self.assertLessEqual(report.donor_action_adoption_rate_pct, 100.0)
        self.assertGreaterEqual(report.semantic_fidelity_score, 0.0)
        self.assertLessEqual(report.semantic_fidelity_score, 1.0)


if __name__ == "__main__":
    unittest.main()
