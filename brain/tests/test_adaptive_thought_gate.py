"""Unit tests for adaptive_thought_gate.py."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from irene_brain.model.adaptive_thought_gate import (
        AdaptiveHaltingController,
        AdaptiveThoughtUpdateGate,
        PredictiveFutureTrajectoryHead,
    )


@unittest.skipUnless(torch is not None, "PyTorch required")
class AdaptiveThoughtGateTests(unittest.TestCase):
    def test_adaptive_thought_update_gate(self) -> None:
        B, T, R, W = 2, 4, 2, 32
        gate = AdaptiveThoughtUpdateGate(width=W)

        thoughts = torch.randn(B, T, R, W)
        seeds = torch.randn(B, T, R, W)
        sensors = torch.randn(B, 8, W)
        belief = torch.randn(B, 4, W)

        refreshed, alphas = gate(
            thoughts=thoughts,
            seeds=seeds,
            sensors=sensors,
            belief=belief,
        )

        self.assertEqual(refreshed.shape, (B, T, R, W))
        self.assertEqual(alphas.shape, (B, T))
        self.assertTrue((alphas >= 0.0).all() and (alphas <= 1.0).all())

    def test_predictive_future_trajectory_head(self) -> None:
        B, T, R, W = 2, 4, 2, 32
        head = PredictiveFutureTrajectoryHead(width=W, horizons=(1, 3, 5))
        thoughts = torch.randn(B, T, R, W)

        preds = head(thoughts)
        self.assertIn("predicted_displacement", preds)
        self.assertIn("predicted_ghost_proximity", preds)
        self.assertIn("predicted_escape_margin", preds)

        self.assertEqual(preds["predicted_displacement"].shape, (B, 3, 2))
        self.assertEqual(preds["predicted_ghost_proximity"].shape, (B, 3, 1))
        self.assertEqual(preds["predicted_escape_margin"].shape, (B, 3, 1))

    def test_adaptive_halting_controller(self) -> None:
        B, T, R, W = 2, 4, 2, 32
        controller = AdaptiveHaltingController(width=W, max_cycles=4, threshold=0.5)
        thoughts = torch.randn(B, T, R, W)

        # Cycle 0 (first cycle)
        prob, should_halt = controller(thoughts, cycle_index=0)
        self.assertEqual(prob.shape, (B, 1))
        self.assertIsInstance(should_halt, bool)

        # Cycle 3 (max cycle should always halt)
        _, should_halt_max = controller(thoughts, cycle_index=3)
        self.assertTrue(should_halt_max)


if __name__ == "__main__":
    unittest.main()
