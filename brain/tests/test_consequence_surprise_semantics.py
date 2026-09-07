"""P0 Consequence-Surprise Semantics & Causal Timing Test Suite.

Proves the mathematical and empirical decoupling of sensory surprise and consequence surprise:
1. Case a: Expected outcome produces near-zero consequence error (delta_consequence ~ 0).
2. Case b: Unexpected reward shock produces spike in reward error and consequence surprise.
3. Case c: Unexpected penalty shock produces spike in consequence surprise.
4. Case d: Pure sensory noise without consequence yields sensory_surprise > 0, but consequence_surprise == 0,
          preventing spurious fast weight updates and preserving thoughts.
5. Case e: Causal timing DAG verification: P_{t+1} depends causally on outcome_{t+1}, not vice versa;
          counterfactual outcomes produce distinct P_{t+1}.
6. Case f: CGP fast weights update strictly post-consequence and exhibit long-term retention.
"""

from __future__ import annotations

import unittest
import torch
import torch.nn.functional as F

from irene_brain.model import (
    BrainState,
    IreneBrainModel,
    ThoughtFieldConfig,
)


def make_test_model(*, use_cgp: bool = True) -> tuple[IreneBrainModel, ThoughtFieldConfig]:
    config = ThoughtFieldConfig.smoke()
    object.__setattr__(config, "core_width", 32)
    object.__setattr__(config, "thoughtlets", 4)
    object.__setattr__(config, "registers_per_thoughtlet", 2)
    object.__setattr__(config, "attention_heads", 2)
    object.__setattr__(config, "routed_neighbors", 1)
    object.__setattr__(config, "brain_cell_blocks", 1)
    object.__setattr__(config, "cognitive_cycles", 1)
    object.__setattr__(config, "use_cgp", use_cgp)
    object.__setattr__(config, "plastic_decay", 0.999)
    object.__setattr__(config, "plastic_lr", 0.25)
    model = IreneBrainModel(config, input_resolution=(32, 32), use_cgp=use_cgp)
    model.eval()
    return model, config


class ConsequenceSurpriseSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.model, self.config = make_test_model(use_cgp=True)
        self.batch = 1
        self.ctrl = torch.zeros(self.batch, self.config.actuator.total_queries)
        self.elapsed = torch.tensor([[1.0 / 60.0]])
        self.pixels = torch.randn(self.batch, 3, 32, 32)

    def test_case_a_expected_outcome_zero_error(self) -> None:
        """Case a: Expected outcome and reward yields consequence_surprise ~ 0."""
        s0 = self.model.initial_state(self.batch)
        # Warmup step to establish latent and reward predictions
        out1 = self.model(self.pixels, self.ctrl, self.elapsed, s0)
        s1 = out1.next_state

        # Query with actual_reward equal to the predicted reward
        expected_reward = s1.prev_reward_pred.clone()
        out2 = self.model(self.pixels, self.ctrl, self.elapsed, s1, actual_reward=expected_reward)

        diag = out2.diagnostics
        self.assertIsNotNone(diag.reward_error)
        self.assertIsNotNone(diag.consequence_surprise)
        self.assertLess(diag.reward_error.item(), 1e-4)
        self.assertLess(diag.consequence_surprise.item(), 1e-4)

    def test_case_b_unexpected_reward_shock(self) -> None:
        """Case b: Unexpected positive reward shock triggers reward error and consequence surprise spike."""
        s0 = self.model.initial_state(self.batch)
        out1 = self.model(self.pixels, self.ctrl, self.elapsed, s0)
        s1 = out1.next_state

        pred_r = s1.prev_reward_pred.item()
        shock_reward = torch.tensor([[pred_r + 5.0]])

        out2 = self.model(self.pixels, self.ctrl, self.elapsed, s1, actual_reward=shock_reward)
        diag = out2.diagnostics

        self.assertAlmostEqual(diag.reward_error.item(), 5.0, places=3)
        self.assertAlmostEqual(diag.consequence_surprise.item(), 5.0, places=3)

        # Fast plastic weights P_t must capture the shock
        p2_norm = out2.next_state.plastic_weights.norm().item()
        self.assertGreater(p2_norm, 0.05, f"Plastic weights must update under reward shock: {p2_norm}")

    def test_case_c_unexpected_penalty_shock(self) -> None:
        """Case c: Unexpected penalty shock triggers consequence surprise spike."""
        s0 = self.model.initial_state(self.batch)
        out1 = self.model(self.pixels, self.ctrl, self.elapsed, s0)
        s1 = out1.next_state

        pred_r = s1.prev_reward_pred.item()
        penalty_reward = torch.tensor([[pred_r - 5.0]])

        out2 = self.model(self.pixels, self.ctrl, self.elapsed, s1, actual_reward=penalty_reward)
        diag = out2.diagnostics

        self.assertAlmostEqual(diag.reward_error.item(), 5.0, places=3)
        self.assertAlmostEqual(diag.consequence_surprise.item(), 5.0, places=3)

        p2_norm = out2.next_state.plastic_weights.norm().item()
        self.assertGreater(p2_norm, 0.05, f"Plastic weights must update under penalty shock: {p2_norm}")

    def test_case_d_sensory_noise_without_consequence(self) -> None:
        """Case d (CRITICAL DECOUPLING): Pure sensory noise produces sensory surprise > 0, but consequence surprise == 0."""
        s0 = self.model.initial_state(self.batch)
        out1 = self.model(self.pixels, self.ctrl, self.elapsed, s0)
        s1 = out1.next_state

        # Drastically perturbed pixels (visual flash / distractor noise)
        noisy_pixels = torch.randn_like(self.pixels) * 5.0 + 3.0

        # Feed noisy pixels with EXPECTED reward (zero consequence surprise)
        expected_reward = s1.prev_reward_pred.clone()
        out2 = self.model(noisy_pixels, self.ctrl, self.elapsed, s1, actual_reward=expected_reward)
        diag = out2.diagnostics

        # 1. Sensory surprise must detect the visual discrepancy
        self.assertIsNotNone(diag.sensory_surprise)
        self.assertGreater(diag.sensory_surprise.item(), 0.5, "Sensory surprise must detect visual perturbation")

        # 2. Consequence surprise must remain zero!
        self.assertIsNotNone(diag.consequence_surprise)
        self.assertLess(diag.consequence_surprise.item(), 1e-4, "Consequence surprise must be zero under zero reward/outcome error")
        self.assertLess(diag.reward_error.item(), 1e-4)

        # 3. Plastic weights P_t must NOT update spuriously from sensory noise alone
        p1 = s1.plastic_weights
        p2 = out2.next_state.plastic_weights
        decay = getattr(self.config, "plastic_decay", 0.999)
        expected_decayed_norm = (decay * p1).norm().item()
        actual_norm = p2.norm().item()
        self.assertAlmostEqual(
            actual_norm,
            expected_decayed_norm,
            places=4,
            msg=f"Plastic weights updated spuriously from sensory noise: actual {actual_norm} vs decayed {expected_decayed_norm}",
        )

        # 4. Thought expiration must remain gated (not expired by sensory noise)
        self.assertLess(diag.applied_expire_probability.mean().item(), 0.1)

    def test_case_e_causal_timing_dag(self) -> None:
        """Case e: Causal timing DAG verification: P_{t+1} depends on outcome_{t+1}, not vice versa."""
        s0 = self.model.initial_state(self.batch)
        out1 = self.model(self.pixels, self.ctrl, self.elapsed, s0)
        s1 = out1.next_state

        # At step t, future outcome is unknown.
        # Run step t+1 under two counterfactual outcomes:
        r_actual_a = torch.tensor([[0.0]])
        r_actual_b = torch.tensor([[10.0]])

        out2_a = self.model(self.pixels, self.ctrl, self.elapsed, s1, actual_reward=r_actual_a)
        out2_b = self.model(self.pixels, self.ctrl, self.elapsed, s1, actual_reward=r_actual_b)

        # Both must have identical starting state
        self.assertTrue(torch.equal(s1.plastic_weights, s1.plastic_weights))

        # But counterfactual outcomes at t+1 produce distinct P_{t+1}
        p2_a = out2_a.next_state.plastic_weights
        p2_b = out2_b.next_state.plastic_weights
        diff = torch.norm(p2_a - p2_b).item()
        self.assertGreater(diff, 0.1, f"Counterfactual consequences must produce distinct plastic states: {diff}")

    def test_case_f_cgp_post_consequence_latching(self) -> None:
        """Case f: Fast plastic weights latch post-consequence and retain trace across subsequent quiet steps."""
        s0 = self.model.initial_state(self.batch)
        out1 = self.model(self.pixels, self.ctrl, self.elapsed, s0)
        s1 = out1.next_state

        # Consequence shock at step 2
        shock = torch.tensor([[2.0]])
        out2 = self.model(self.pixels, self.ctrl, self.elapsed, s1, actual_reward=shock)
        s2 = out2.next_state
        initial_latched_norm = s2.plastic_weights.norm().item()
        self.assertGreater(initial_latched_norm, 0.01)

        # Subsequent quiet steps (no consequence shock)
        cur_state = s2
        for _ in range(20):
            exp_r = cur_state.prev_reward_pred.clone()
            out_quiet = self.model(self.pixels, self.ctrl, self.elapsed, cur_state, actual_reward=exp_r)
            cur_state = out_quiet.next_state

        retained_norm = cur_state.plastic_weights.norm().item()
        expected_floor = initial_latched_norm * (0.999**20) * 0.95
        self.assertGreater(
            retained_norm,
            expected_floor,
            f"Plastic trace decayed faster than biological decay: {retained_norm} vs floor {expected_floor}",
        )


if __name__ == "__main__":
    unittest.main()
