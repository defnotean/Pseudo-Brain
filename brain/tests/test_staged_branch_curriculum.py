"""Unit tests for Staged Multi-Branch Curriculum and Multi-Hypothesis Loss."""

from __future__ import annotations

import torch
import torch.nn as nn

from irene_brain.environments.phase2_suite import Phase2TaskEnvironment, TaskFamily, make_family_suite
from irene_brain.model.thought_mediated_actuator import ThoughtMediatedActuator
from irene_brain.training.staged_branch_curriculum import (
    ComprehensiveBranchBundle,
    MultiHypothesisBranchLoss,
    generate_comprehensive_branch_bundle,
)


def test_comprehensive_branch_bundle_generation() -> None:
    env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_B_PURSUIT_EVASION)[0])
    obs = env.reset(42)

    bundle = generate_comprehensive_branch_bundle(env, obs)

    assert bundle.action_indices.shape == (5,)
    assert bundle.displacements.shape == (5, 2)
    assert bundle.hazard_probs.shape == (5, 1)
    assert bundle.rewards.shape == (5, 1)
    assert bundle.utilities.shape == (5, 1)

    # Invariants
    assert (bundle.hazard_probs >= 0.0).all() and (bundle.hazard_probs <= 1.0).all()
    assert torch.isfinite(bundle.utilities).all()


def test_multi_hypothesis_branch_loss_forward_and_backward() -> None:
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

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert "branch_total_loss" in metrics

    loss.backward()
    assert thoughts.grad is not None
    assert torch.isfinite(thoughts.grad).all()
