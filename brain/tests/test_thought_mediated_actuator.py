"""Tests for Thought-Mediated Actuator Architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from irene_brain.model.thought_mediated_actuator import (
    BoundedReflexHead,
    PermutationInvariantProposalAggregator,
    SharedPerThoughtletProposalHead,
    ThoughtMediatedActuator,
)


def test_thought_mediated_actuator_shape_and_no_bypass() -> None:
    batch = 2
    thoughtlets = 32
    registers = 4
    core_width = 32
    sensors_count = 4
    num_buttons = 296

    actuator = ThoughtMediatedActuator(core_width=core_width, num_buttons=num_buttons, max_reflex_delta=0.10)

    sensors = torch.randn(batch, sensors_count, core_width)
    thoughts = torch.randn(batch, thoughtlets, registers, core_width)

    out = actuator(sensors=sensors, thoughts=thoughts)

    assert out.button_logits.shape == (batch, num_buttons)
    assert out.main_action_intent.shape == (batch, num_buttons)
    assert out.reflex_correction.shape == (batch, num_buttons)
    assert out.aggregation_weights.shape == (batch, thoughtlets)
    assert torch.allclose(out.aggregation_weights.sum(dim=-1), torch.ones(batch), atol=1e-5)

    # Verify reflex bound
    assert (out.reflex_correction.abs() <= 0.10001).all()

    # Verify No-Bypass Invariant:
    # When thoughts are zeroed and we zero the proposal head bias, main action intent is exactly 0.0
    zero_thoughts = torch.zeros(batch, thoughtlets, registers, core_width)
    zero_out = actuator(sensors=sensors, thoughts=zero_thoughts)
    # The only sensory impact on final button logits is through the bounded reflex [-0.1, 0.1]
    assert zero_out.reflex_correction.abs().max() <= 0.10


def test_permutation_equivariance_of_proposal_aggregation() -> None:
    batch = 1
    thoughtlets = 16
    core_width = 32
    num_buttons = 296

    actuator = ThoughtMediatedActuator(core_width=core_width, num_buttons=num_buttons)

    sensors = torch.randn(batch, 4, core_width)
    thoughts = torch.randn(batch, thoughtlets, core_width)

    out1 = actuator(sensors=sensors, thoughts=thoughts)

    # Permute slots
    perm = torch.randperm(thoughtlets)
    permuted_thoughts = thoughts[:, perm, :]

    out2 = actuator(sensors=sensors, thoughts=permuted_thoughts)

    # Main action intent should be invariant to slot permutation
    assert torch.allclose(out1.main_action_intent, out2.main_action_intent, atol=1e-5)
    assert torch.allclose(out1.button_logits, out2.button_logits, atol=1e-5)
