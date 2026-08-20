"""Stage 0: Mechanical Causal Plumbing Verification Tests for Thought-Mediated Actuator."""

from __future__ import annotations

import torch
import torch.nn as nn

from irene_brain.model.thought_mediated_actuator import (
    BoundedReflexHead,
    PermutationInvariantProposalAggregator,
    SharedPerThoughtletProposalHead,
    ThoughtMediatedActuator,
)


def test_stage0_all_thoughts_zero_produces_strictly_neutral_main_action() -> None:
    """Test 1: When all thoughts are zero (K=0), main action intent must be EXACTLY zero."""
    batch = 2
    thoughtlets = 32
    core_width = 32
    num_buttons = 296

    actuator = ThoughtMediatedActuator(core_width=core_width, num_buttons=num_buttons, max_reflex_delta=0.10)

    sensors = torch.randn(batch, 4, core_width)
    zero_thoughts = torch.zeros(batch, thoughtlets, 4, core_width)

    out = actuator(sensors=sensors, thoughts=zero_thoughts)

    # Inactive/zero thoughts must produce EXACT zero main action intent
    assert torch.equal(out.main_action_intent, torch.zeros(batch, num_buttons)), (
        "Main action intent was not strictly zero when all thoughts were empty!"
    )
    # The only sensory contribution must be within the bounded reflex [-0.10, +0.10]
    assert out.reflex_correction.abs().max() <= 0.10001
    assert torch.allclose(out.button_logits, out.reflex_correction, atol=1e-6)


def test_stage0_one_thought_active_determines_action() -> None:
    """Test 2: When only one thought is active (K=1), its proposal alone determines main action."""
    batch = 1
    thoughtlets = 8
    core_width = 32
    num_buttons = 296

    actuator = ThoughtMediatedActuator(core_width=core_width, num_buttons=num_buttons)

    sensors = torch.randn(batch, 4, core_width)
    thoughts = torch.zeros(batch, thoughtlets, core_width)
    # Activate only slot 3
    thoughts[:, 3, :] = torch.randn(batch, core_width) + 1.0

    out = actuator(sensors=sensors, thoughts=thoughts)

    # Slot 3 must have weight 1.0, and all other slots must have weight 0.0
    assert torch.isclose(out.aggregation_weights[0, 3], torch.tensor(1.0), atol=1e-5)
    for k in range(thoughtlets):
        if k != 3:
            assert torch.isclose(out.aggregation_weights[0, k], torch.tensor(0.0), atol=1e-5)

    # Main action intent should match slot 3's proposal exactly
    slot3_proposal = out.proposals.button_logits[0, 3]
    assert torch.allclose(out.main_action_intent[0], slot3_proposal, atol=1e-5)


def test_stage0_slot_permutation_invariance() -> None:
    """Test 3: Permuting whole thought slots leaves main action intent mathematically identical."""
    batch = 1
    thoughtlets = 16
    core_width = 32
    num_buttons = 296

    actuator = ThoughtMediatedActuator(core_width=core_width, num_buttons=num_buttons)

    sensors = torch.randn(batch, 4, core_width)
    thoughts = torch.randn(batch, thoughtlets, core_width) + 0.5

    out1 = actuator(sensors=sensors, thoughts=thoughts)

    # Randomly permute slots
    perm = torch.randperm(thoughtlets)
    permuted_thoughts = thoughts[:, perm, :]

    out2 = actuator(sensors=sensors, thoughts=permuted_thoughts)

    # Main action intent and final button logits must be invariant to slot ordering
    assert torch.allclose(out1.main_action_intent, out2.main_action_intent, atol=1e-5)
    assert torch.allclose(out1.button_logits, out2.button_logits, atol=1e-5)


def test_stage0_thought_removal_zeros_its_contribution() -> None:
    """Test 4: Explicitly masking / removing a thought slot zeroes its contribution exactly."""
    batch = 1
    thoughtlets = 4
    core_width = 32
    num_buttons = 296

    actuator = ThoughtMediatedActuator(core_width=core_width, num_buttons=num_buttons)

    sensors = torch.randn(batch, 4, core_width)
    thoughts = torch.randn(batch, thoughtlets, core_width) + 1.0

    # Mask slot 2
    active_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])

    out = actuator(sensors=sensors, thoughts=thoughts, active_mask=active_mask)

    # Slot 2 must have weight 0.0 and proposal 0.0
    assert torch.isclose(out.aggregation_weights[0, 2], torch.tensor(0.0), atol=1e-5)
    assert torch.equal(out.proposals.button_logits[0, 2], torch.zeros(num_buttons))


def test_stage0_sensors_change_does_not_affect_main_action_intent() -> None:
    """Test 5: Changing sensory / belief tokens does not alter main action intent (no bypass)."""
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

    # Main action intent must be 100% identical despite huge sensor shift
    assert torch.equal(out1.main_action_intent, out2.main_action_intent)

    # Reflex is allowed to differ, but must remain bounded inside [-0.10, +0.10]
    assert out1.reflex_correction.abs().max() <= 0.10001
    assert out2.reflex_correction.abs().max() <= 0.10001


def test_stage0_scrambled_thoughts_change_main_action_intent() -> None:
    """Test 6: Scrambling thought tokens changes main action intent."""
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

    # Different thought representations must produce different main action intents
    assert not torch.allclose(out_a.main_action_intent, out_b.main_action_intent, atol=1e-3)
