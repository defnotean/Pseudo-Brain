"""Tests for ranked-K unmatched suppression, multiplicity Q(a), and collapse guard."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from irene_brain.environments.phase2_suite import Phase2TaskEnvironment, TaskFamily, make_family_suite
from irene_brain.model.consequence_thought_actuator import (
    ConsequenceProposal,
    ConsequenceProposalAggregator,
    ConsequenceThoughtActuator,
)
from irene_brain.training.staged_branch_curriculum import (
    MultiHypothesisBranchLoss,
    generate_comprehensive_branch_bundle,
)


def _proposal(
    *,
    action_logits: torch.Tensor,
    utility: torch.Tensor,
    branch_prob: torch.Tensor | None = None,
    active_mask: torch.Tensor | None = None,
) -> ConsequenceProposal:
    action_probs = F.softmax(action_logits, dim=-1)
    batch, k_slots, _ = action_logits.shape
    if branch_prob is None:
        branch_prob = torch.ones(batch, k_slots, 1)
    if active_mask is None:
        active_mask = torch.ones(batch, k_slots)
    return ConsequenceProposal(
        action_logits=action_logits,
        action_probs=action_probs,
        displacement=torch.zeros(batch, k_slots, 2),
        hazard_prob=torch.zeros(batch, k_slots, 1),
        reward_estimate=utility.clone(),
        confidence=torch.full((batch, k_slots, 1), 0.5),
        consequence_utility=utility,
        active_mask=active_mask,
        branch_probability=branch_prob,
    )


class TestRankedK8UnmatchedSuppress(unittest.TestCase):
    def test_multiplicity_q_prefers_specialist_over_duplicate_majority(self) -> None:
        aggregator = ConsequenceProposalAggregator(temperature=0.10, num_buttons=296)
        action_logits = torch.full((1, 9, 5), -10.0)
        action_logits[0, 0, 2] = 10.0
        action_logits[0, 1:, 4] = 10.0
        utility = torch.ones(1, 9, 1)
        utility[0, 0, 0] = 1.05
        intent, _weights, dist = aggregator(_proposal(action_logits=action_logits, utility=utility))
        self.assertEqual(int(dist.argmax(-1).item()), 2)
        self.assertTrue(torch.isfinite(intent).all())

    def test_permutation_of_slots_leaves_action_dist_unchanged(self) -> None:
        torch.manual_seed(0)
        actuator = ConsequenceThoughtActuator(core_width=16)
        actuator.eval()
        sensors = torch.randn(4, 4, 16)
        thoughts = torch.randn(4, 8, 3, 16) + 0.4
        with torch.no_grad():
            baseline = actuator(sensors=sensors, thoughts=thoughts)
            permuted = actuator(sensors=sensors, thoughts=thoughts[:, torch.randperm(8)])
        self.assertTrue(torch.allclose(baseline.action_dist, permuted.action_dist, atol=1e-5))
        self.assertEqual(
            float((baseline.action_dist.argmax(-1) != permuted.action_dist.argmax(-1)).float().mean()),
            0.0,
        )

    def test_register_swap_changes_action_when_registers_differ(self) -> None:
        torch.manual_seed(3)
        actuator = ConsequenceThoughtActuator(core_width=16)
        actuator.eval()
        sensors = torch.randn(8, 4, 16)
        thoughts = torch.randn(8, 8, 3, 16)
        thoughts[:, :, 0] = thoughts[:, :, 0] + 1.25
        thoughts[:, :, 1] = thoughts[:, :, 1] - 1.25
        swapped = thoughts.clone()
        swapped[:, :, 0], swapped[:, :, 1] = swapped[:, :, 1].clone(), swapped[:, :, 0].clone()
        with torch.no_grad():
            baseline = actuator(sensors=sensors, thoughts=thoughts)
            scrambled = actuator(sensors=sensors, thoughts=swapped)
        changed = float((baseline.action_dist.argmax(-1) != scrambled.action_dist.argmax(-1)).float().mean())
        l1 = float((baseline.action_dist - scrambled.action_dist).abs().mean())
        self.assertTrue(changed > 0.0 or l1 > 1e-4)

    def test_unmatched_slots_are_penalized_and_loss_backprops(self) -> None:
        torch.manual_seed(4)
        loss_fn = MultiHypothesisBranchLoss()
        env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_B_PURSUIT_EVASION)[0])
        obs = env.reset(42)
        bundles = [generate_comprehensive_branch_bundle(env, obs)]
        k_slots = 16
        action_logits = torch.zeros(1, k_slots, 5, requires_grad=True)
        utility = torch.zeros(1, k_slots, 1)
        branch_prob = torch.ones(1, k_slots, 1)
        proposals = _proposal(action_logits=action_logits, utility=utility, branch_prob=branch_prob)
        thoughts = torch.randn(1, k_slots, 3, 8, requires_grad=True)
        thoughts = thoughts.clone()
        thoughts[:, :, 0] = thoughts[:, :, 1]
        loss, metrics = loss_fn(proposals, bundles, thoughts=thoughts)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["unmatched_suppress"], 0.0)
        self.assertIn("collapse_guard", metrics)
        loss.backward()
        self.assertIsNotNone(action_logits.grad)
        self.assertTrue(torch.isfinite(action_logits.grad).all())

    def test_belief_tokens_do_not_change_main_intent(self) -> None:
        torch.manual_seed(5)
        actuator = ConsequenceThoughtActuator(core_width=16)
        thoughts = torch.randn(2, 8, 3, 16) + 0.5
        with torch.no_grad():
            a = actuator(sensors=torch.randn(2, 4, 16), thoughts=thoughts)
            b = actuator(sensors=torch.randn(2, 4, 16) * 4.0, thoughts=thoughts)
        self.assertTrue(torch.equal(a.main_action_intent, b.main_action_intent))


if __name__ == "__main__":
    unittest.main()
