"""Stage 0: Mechanical Causal Plumbing Verification Tests for Thought-Mediated Actuator."""

from __future__ import annotations

import unittest

import torch

from irene_brain.model.thought_mediated_actuator import ThoughtMediatedActuator


class TestThoughtMediatedActuator(unittest.TestCase):
    def test_stage0_all_thoughts_zero_produces_strictly_neutral_main_action(self) -> None:
        """When all thoughts are zero (K=0), main action intent must be EXACTLY zero."""
        batch = 2
        thoughtlets = 32
        core_width = 32
        num_buttons = 296

        actuator = ThoughtMediatedActuator(core_width=core_width, num_buttons=num_buttons, max_reflex_delta=0.10)

        sensors = torch.randn(batch, 4, core_width)
        zero_thoughts = torch.zeros(batch, thoughtlets, 4, core_width)

        out = actuator(sensors=sensors, thoughts=zero_thoughts)

        self.assertTrue(
            torch.equal(out.main_action_intent, torch.zeros(batch, num_buttons)),
            "Main action intent was not strictly zero when all thoughts were empty!",
        )
        self.assertLessEqual(float(out.reflex_correction.abs().max()), 0.10001)
        self.assertTrue(torch.allclose(out.button_logits, out.reflex_correction, atol=1e-6))

    def test_stage0_one_thought_active_determines_action(self) -> None:
        """When only one thought is active (K=1), its proposal alone determines main action."""
        batch = 1
        thoughtlets = 8
        core_width = 32
        num_buttons = 296

        actuator = ThoughtMediatedActuator(core_width=core_width, num_buttons=num_buttons)

        sensors = torch.randn(batch, 4, core_width)
        thoughts = torch.zeros(batch, thoughtlets, core_width)
        thoughts[:, 3, :] = torch.randn(batch, core_width) + 1.0

        out = actuator(sensors=sensors, thoughts=thoughts)

        self.assertTrue(torch.isclose(out.aggregation_weights[0, 3], torch.tensor(1.0), atol=1e-5))
        for slot in range(thoughtlets):
            if slot != 3:
                self.assertTrue(torch.isclose(out.aggregation_weights[0, slot], torch.tensor(0.0), atol=1e-5))

        slot3_proposal = out.proposals.button_logits[0, 3]
        self.assertTrue(torch.allclose(out.main_action_intent[0], slot3_proposal, atol=1e-5))

    def test_stage0_slot_permutation_invariance(self) -> None:
        """Permuting whole thought slots leaves main action intent mathematically identical."""
        batch = 1
        thoughtlets = 16
        core_width = 32
        num_buttons = 296

        actuator = ThoughtMediatedActuator(core_width=core_width, num_buttons=num_buttons)

        sensors = torch.randn(batch, 4, core_width)
        thoughts = torch.randn(batch, thoughtlets, core_width) + 0.5

        out1 = actuator(sensors=sensors, thoughts=thoughts)

        perm = torch.randperm(thoughtlets)
        permuted_thoughts = thoughts[:, perm, :]

        out2 = actuator(sensors=sensors, thoughts=permuted_thoughts)

        self.assertTrue(torch.allclose(out1.main_action_intent, out2.main_action_intent, atol=1e-5))
        self.assertTrue(torch.allclose(out1.button_logits, out2.button_logits, atol=1e-5))

    def test_stage0_thought_removal_zeros_its_contribution(self) -> None:
        """Explicitly masking / removing a thought slot zeroes its contribution exactly."""
        batch = 1
        thoughtlets = 4
        core_width = 32
        num_buttons = 296

        actuator = ThoughtMediatedActuator(core_width=core_width, num_buttons=num_buttons)

        sensors = torch.randn(batch, 4, core_width)
        thoughts = torch.randn(batch, thoughtlets, core_width) + 1.0

        active_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])

        out = actuator(sensors=sensors, thoughts=thoughts, active_mask=active_mask)

        self.assertTrue(torch.isclose(out.aggregation_weights[0, 2], torch.tensor(0.0), atol=1e-5))
        self.assertTrue(torch.equal(out.proposals.button_logits[0, 2], torch.zeros(num_buttons)))

    def test_stage0_sensors_change_does_not_affect_main_action_intent(self) -> None:
        """Changing sensory / belief tokens does not alter main action intent (no bypass)."""
        batch = 1
        thoughtlets = 8
        core_width = 32
        num_buttons = 296

        actuator = ThoughtMediatedActuator(core_width=core_width, num_buttons=num_buttons)

        sensors_1 = torch.randn(batch, 4, core_width)
        sensors_2 = torch.randn(batch, 4, core_width) * 10.0 + 5.0
        frozen_thoughts = torch.randn(batch, thoughtlets, core_width) + 0.5

        out1 = actuator(sensors=sensors_1, thoughts=frozen_thoughts)
        out2 = actuator(sensors=sensors_2, thoughts=frozen_thoughts)

        self.assertTrue(torch.equal(out1.main_action_intent, out2.main_action_intent))
        self.assertLessEqual(float(out1.reflex_correction.abs().max()), 0.10001)
        self.assertLessEqual(float(out2.reflex_correction.abs().max()), 0.10001)

    def test_stage0_scrambled_thoughts_change_main_action_intent(self) -> None:
        """Scrambling thought tokens changes main action intent."""
        batch = 1
        thoughtlets = 8
        core_width = 32
        num_buttons = 296

        actuator = ThoughtMediatedActuator(core_width=core_width, num_buttons=num_buttons)

        sensors = torch.randn(batch, 4, core_width)
        thoughts_a = torch.randn(batch, thoughtlets, core_width) + 1.0
        thoughts_b = torch.randn(batch, thoughtlets, core_width) - 1.0

        out_a = actuator(sensors=sensors, thoughts=thoughts_a)
        out_b = actuator(sensors=sensors, thoughts=thoughts_b)

        self.assertFalse(torch.allclose(out_a.main_action_intent, out_b.main_action_intent, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
