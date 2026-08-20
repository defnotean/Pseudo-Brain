"""Unit tests for Phase 2 Design 2 modules: branch set matching and thought scrambling."""

from __future__ import annotations

import unittest
import torch

from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.thought_scrambler import (
    apply_capacity_slot_mask,
    permute_whole_slots,
    scramble_cognitive_bindings,
)
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.training.branch_set_objective import (
    BranchOutcome,
    MultiFutureBranchObjective,
)


class Phase2Design2Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_slot_dropout_and_branch_matching(self) -> None:
        """Verify slot dropout and Hungarian branch loss computation."""
        objective = MultiFutureBranchObjective(
            thoughtlets=8,
            core_width=16,
            slot_dropout_prob=0.25,
        )
        thoughts = torch.randn(2, 8, 2, 16)
        masked_thoughts, active_mask = objective.apply_slot_dropout(thoughts, training=True)
        self.assertEqual(masked_thoughts.shape, thoughts.shape)
        self.assertEqual(active_mask.shape, (2, 8))
        self.assertGreaterEqual(int(active_mask.sum(dim=1).min()), 4)

        branches = [
            [
                BranchOutcome(action_index=1, hazard_prob=0.9, reward=-1.0, dx=0.0, dy=-1.0),
                BranchOutcome(action_index=2, hazard_prob=0.1, reward=1.0, dx=-1.0, dy=0.0),
            ],
            [
                BranchOutcome(action_index=3, hazard_prob=0.0, reward=2.0, dx=0.0, dy=1.0),
            ],
        ]
        loss, metrics = objective.compute_branch_loss(thoughts, branches)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("branch_total_loss", metrics)
        self.assertIn("branch_hazard_err", metrics)

    def test_thought_scrambler_and_capacity_sweep(self) -> None:
        """Verify permutation equivariance helper, cognitive scrambling, and capacity masking."""
        config = ThoughtFieldConfig(
            thoughtlets=8,
            cognitive_cycles=2,
            core_width=16,
            attention_heads=2,
            sensor_tokens=4,
            belief_tokens=2,
            working_memory_tokens=2,
            goal_context_tokens=4,
            registers_per_thoughtlet=2,
            brain_cell_blocks=1,
            routed_neighbors=1,
        )
        model = IreneBrainModel(config=config, input_resolution=(16, 16))
        state = model.initial_state(1)

        # 1. Whole-slot permutation
        perm = [7, 6, 5, 4, 3, 2, 1, 0]
        permuted = permute_whole_slots(state, perm)
        self.assertEqual(permuted.thoughts.shape, state.thoughts.shape)
        self.assertTrue(torch.equal(permuted.thoughts[:, 0], state.thoughts[:, 7]))

        # 2. Cognitive scrambling
        scrambled = scramble_cognitive_bindings(state)
        self.assertEqual(scrambled.thoughts.shape, state.thoughts.shape)
        self.assertFalse(torch.equal(scrambled.thoughts, state.thoughts))

        # 3. Capacity slot masking
        for active in (1, 4, 8):
            masked = apply_capacity_slot_mask(state, active)
            self.assertEqual(masked.thoughts.shape, state.thoughts.shape)
            if active < 8:
                self.assertTrue(torch.equal(masked.thoughts[:, active:], torch.zeros_like(masked.thoughts[:, active:])))


if __name__ == "__main__":
    unittest.main()
