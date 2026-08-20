"""Quantitative Diagnostic for Hypothesis Persistence and Temporal Matching Inertia.

Measures how consistently thoughtlets maintain persistent hypothesis trajectories across
consecutive frames without unnecessary reassignment or identity fluttering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ..environments.phase2_suite import Phase2TaskEnvironment, make_family_suite, TaskFamily
from ..training.staged_branch_curriculum import generate_multi_step_trajectory_tree, MultiHypothesisBranchLoss
from ..types import GenericControl, HidKey


@dataclass(frozen=True, slots=True)
class TemporalPersistenceReport:
    """Quantitative report on thoughtlet hypothesis persistence across consecutive frames."""
    inertia_weight: float
    branch_identity_retention_rate_pct: float  # % slots continuing same root action across t -> t+1
    slot_reassignment_rate_pct: float          # % slots switching root action
    mean_hypothesis_lifetime_ticks: float      # average run length in ticks a slot tracks a hypothesis
    unnecessary_switch_rate_pct: float         # % switches where previous hypothesis error was still low (<= 0.5)
    mean_matching_cost: float                  # average Hungarian matching cost


def evaluate_temporal_persistence(
    model: nn.Module,
    *,
    inertia_weight: float = 0.5,
    num_episodes: int = 5,
    episode_length: int = 20,
    base_seed: int = 12000,
    device: torch.device = torch.device("cpu"),
) -> TemporalPersistenceReport:
    """Evaluate hypothesis continuity and lifetime across consecutive rollout steps."""
    model.eval()
    loss_fn = MultiHypothesisBranchLoss(inertia_weight=inertia_weight).to(device)

    all_families = [TaskFamily.FAMILY_B_PURSUIT_EVASION, TaskFamily.FAMILY_C_JUNCTIONS]
    dir_actions = [int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)]

    retentions = []
    reassignments = []
    lifetimes = []
    unnecessary_switches = []
    matching_costs = []

    ctrl_tensor = torch.zeros((1, 307), device=device)
    dt_tensor = torch.tensor([0.016667], device=device)

    with torch.no_grad():
        for ep in range(num_episodes):
            fam = all_families[ep % len(all_families)]
            env = Phase2TaskEnvironment(make_family_suite(fam)[0])
            obs = env.reset(base_seed + ep * 100)

            state = model.initial_state(1)
            if isinstance(state, Tensor):
                state = state.to(device)

            k_slots = getattr(model, "config", None).thoughtlets if hasattr(model, "config") else 32
            current_slot_actions = np.full(k_slots, -1, dtype=np.int64)
            slot_lifetimes = np.zeros(k_slots, dtype=np.int64)

            # Clear loss matcher history
            loss_fn._prev_assignments.clear()

            for tick in range(episode_length):
                raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
                rgb_tensor = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                if rgb_tensor.shape[-1] != 32:
                    rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")

                out = model(rgb_tensor, ctrl_tensor, dt_tensor, state=state)
                state = out.next_state

                bundle = generate_multi_step_trajectory_tree(env, obs, horizon=2, device=device)
                _loss, metrics = loss_fn(out.action.proposals, [bundle])

                matching_costs.append(float(_loss.item()))

                # Extract current matched assignment
                assign = loss_fn._prev_assignments.get(0, None)
                if assign is not None and len(assign) == k_slots and k_slots > 1:
                    for slot_idx in range(k_slots):
                        branch_col = assign[slot_idx]
                        if branch_col < bundle.action_indices.shape[0]:
                            act_idx = int(bundle.action_indices[branch_col].item())

                            if current_slot_actions[slot_idx] != -1:
                                if act_idx == current_slot_actions[slot_idx]:
                                    retentions.append(1.0)
                                    reassignments.append(0.0)
                                    slot_lifetimes[slot_idx] += 1
                                else:
                                    retentions.append(0.0)
                                    reassignments.append(1.0)
                                    lifetimes.append(slot_lifetimes[slot_idx])
                                    slot_lifetimes[slot_idx] = 1

                                    # Was switch unnecessary? (Check if prev branch error was low)
                                    prev_col = branch_col
                                    disp_err = float(torch.norm(out.action.proposals.displacement[0, slot_idx] - bundle.displacements[prev_col]).item())
                                    unnecessary_switches.append(1.0 if disp_err <= 0.5 else 0.0)
                            else:
                                slot_lifetimes[slot_idx] = 1

                            current_slot_actions[slot_idx] = act_idx

                # Advance environment with exploratory or model action
                act_key = dir_actions[tick % len(dir_actions)]
                step_out = env.step(GenericControl(mouse_dx=0.0, mouse_dy=0.0, keys_down=(act_key,)))
                if step_out.terminated or step_out.truncated:
                    break
                obs = step_out.observation

            for sl in slot_lifetimes:
                if sl > 0:
                    lifetimes.append(sl)

    retention_rate = float(np.mean(retentions) * 100.0) if retentions else 0.0
    reassign_rate = float(np.mean(reassignments) * 100.0) if reassignments else 0.0
    mean_lifetime = float(np.mean(lifetimes)) if lifetimes else 1.0
    unnec_rate = float(np.mean(unnecessary_switches) * 100.0) if unnecessary_switches else 0.0
    mean_cost = float(np.mean(matching_costs)) if matching_costs else 0.0

    return TemporalPersistenceReport(
        inertia_weight=inertia_weight,
        branch_identity_retention_rate_pct=retention_rate,
        slot_reassignment_rate_pct=reassign_rate,
        mean_hypothesis_lifetime_ticks=mean_lifetime,
        unnecessary_switch_rate_pct=unnec_rate,
        mean_matching_cost=mean_cost,
    )
