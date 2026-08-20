"""Test for Multiplicity-Proof Action-Conditional Expected Utility Aggregator."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from irene_brain.evaluation.multiplicity_aggregator import (
        aggregate_action_conditional_expected_utilities,
    )


@unittest.skipUnless(torch is not None, "PyTorch is not installed in play-safe runtime")
class TestMultiplicityProofAggregator(unittest.TestCase):
    def test_duplicate_immunity(self) -> None:
        batch, k_slots, num_buttons = 1, 9, 296
        button_logits = torch.full((batch, k_slots, num_buttons), -10.0)
        button_logits[0, 0, 4] = +10.0
        button_logits[0, 1:, 7] = +10.0

        utility = torch.zeros((batch, k_slots, 1))
        utility[0, 0, 0] = 2.0
        utility[0, 1:, 0] = 1.0

        prob = torch.ones((batch, k_slots, 1))
        mask = torch.ones((batch, k_slots))

        q = aggregate_action_conditional_expected_utilities(button_logits, utility, prob, mask)
        self.assertAlmostEqual(q[0, 2].item(), 2.0, places=2)
        self.assertAlmostEqual(q[0, 4].item(), 1.0, places=2)
        self.assertGreater(q[0, 2].item(), q[0, 4].item())


if __name__ == "__main__":
    unittest.main()
