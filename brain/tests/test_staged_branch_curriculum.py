"""Unit tests for Staged Multi-Branch Curriculum and Multi-Hypothesis Loss."""

from __future__ import annotations

import unittest

import torch

from irene_brain.environments.phase2_suite import Phase2TaskEnvironment, TaskFamily, make_family_suite
from irene_brain.model.thought_mediated_actuator import ThoughtMediatedActuator
from irene_brain.training.staged_branch_curriculum import (
    MultiHypothesisBranchLoss,
    generate_comprehensive_branch_bundle,
)


class TestStagedBranchCurriculum(unittest.TestCase):
    def test_comprehensive_branch_bundle_generation(self) -> None:
        env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_B_PURSUIT_EVASION)[0])
        obs = env.reset(42)

        bundle = generate_comprehensive_branch_bundle(env, obs)

        # 2 WAIT/probe branches + 4 committing actions x 2 ghost outcomes
        self.assertEqual(tuple(bundle.action_indices.shape), (10,))
        self.assertEqual(tuple(bundle.displacements.shape), (10, 2))
        self.assertEqual(tuple(bundle.hazard_probs.shape), (10, 1))
        self.assertEqual(tuple(bundle.rewards.shape), (10, 1))
        self.assertEqual(tuple(bundle.utilities.shape), (10, 1))
        self.assertEqual(tuple(bundle.branch_probabilities.shape), (10, 1))
        self.assertEqual(tuple(bundle.expected_action_utilities.shape), (5, 1))

        self.assertTrue((bundle.hazard_probs >= 0.0).all() and (bundle.hazard_probs <= 1.0).all())
        self.assertTrue(torch.isfinite(bundle.utilities).all())
        self.assertTrue(torch.isfinite(bundle.expected_action_utilities).all())
        self.assertTrue(torch.isclose(bundle.branch_probabilities[:2].sum(), torch.tensor(1.0), atol=1e-5))

    def test_multi_hypothesis_branch_loss_forward_and_backward(self) -> None:
        batch = 2
        k_slots = 8
        core_width = 32
        num_buttons = 296

        actuator = ThoughtMediatedActuator(core_width=core_width, num_buttons=num_buttons)
        loss_fn = MultiHypothesisBranchLoss()

        sensors = torch.randn(batch, 4, core_width)
        thoughts = torch.randn(batch, k_slots, core_width, requires_grad=True)

        out = actuator(sensors=sensors, thoughts=thoughts)

        env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_B_PURSUIT_EVASION)[0])
        obs = env.reset(42)
        bundles = [generate_comprehensive_branch_bundle(env, obs) for _ in range(batch)]

        loss, metrics = loss_fn(out.proposals, bundles)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("branch_total_loss", metrics)

        loss.backward()
        self.assertIsNotNone(thoughts.grad)
        self.assertTrue(torch.isfinite(thoughts.grad).all())


if __name__ == "__main__":
    unittest.main()
