"""Diagnostic Experiment: Testing Causal Actuator-Thought Binding.

Specifically investigates:
1. What fraction of actuator cross-attention is allocated to thoughts vs sensors/belief?
2. If actuator is forced to read ONLY from thoughts (sensors/belief masked out at actuator input),
   does performance scale monotonically with thoughtlet count K in {1, 4, 8, 16, 32}?
3. Does forcing thought-routed action decode beat the matched monolithic GRU?
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from irene_brain.environments.phase2_suite import (
    Phase2TaskEnvironment,
    TaskFamily,
    make_family_suite,
)
from irene_brain.model.phase2_baselines import (
    BaselineVariant,
    build_phase2_model,
)
from irene_brain.model.thought_scrambler import (
    apply_capacity_slot_mask,
    permute_whole_slots,
    scramble_cognitive_bindings,
)
from irene_brain.model.torch_model import BrainState, IreneBrainModel
from irene_brain.training.branch_set_objective import (
    BranchOutcome,
    MultiFutureBranchObjective,
)
from irene_brain.types import GenericControl, HidKey


def generate_counterfactual_branches(env: Phase2TaskEnvironment, current_obs: Any) -> list[BranchOutcome]:
    branches = []
    actions = [(0, ()), (1, (int(HidKey.W),)), (2, (int(HidKey.A),)), (3, (int(HidKey.S),)), (4, (int(HidKey.D),))]
    px = getattr(env._underlying_env, "_player_x", 8)
    py = getattr(env._underlying_env, "_player_y", 8)
    ghosts = getattr(env._underlying_env, "_ghosts", [(4, 4), (12, 12)])

    for act_idx, keys in actions:
        dx = -1.0 if int(HidKey.A) in keys else (1.0 if int(HidKey.D) in keys else 0.0)
        dy = -1.0 if int(HidKey.W) in keys else (1.0 if int(HidKey.S) in keys else 0.0)
        target_x = max(1, min(14, px + int(dx)))
        target_y = max(1, min(14, py + int(dy)))
        min_ghost_dist = min(math.sqrt((target_x - gx) ** 2 + (target_y - gy) ** 2) for gx, gy in ghosts)
        hazard_prob = 1.0 if min_ghost_dist <= 1.5 else (0.5 if min_ghost_dist <= 3.0 else 0.0)
        reward = -10.0 if hazard_prob > 0.8 else (1.0 if hazard_prob == 0.0 else 0.0)
        branches.append(BranchOutcome(action_index=act_idx, hazard_prob=hazard_prob, reward=reward, dx=dx, dy=dy))
    return branches


def main() -> None:
    import math
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== Actuator-Thought Binding Diagnostic on {device} ===")

    model = build_phase2_model(BaselineVariant.PSEUDO_BRAIN).to(device)
    model.eval()

    # Step on a test environment and inspect attention distribution
    env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_B_PURSUIT_EVASION)[0])
    obs = env.reset(42)
    state = model.initial_state(1)

    raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
    rgb_tensor = torch.from_numpy(raw_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    if rgb_tensor.shape[-1] != 32:
        rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")

    ctrl_tensor = torch.zeros((1, model.config.actuator.total_queries), device=device)
    dt_tensor = torch.tensor([0.016667], device=device)

    with torch.no_grad():
        out = model(rgb_tensor, ctrl_tensor, dt_tensor, state=state)

    state_attn = out.action.state_attention[0]  # [total_queries, total_state_tokens]
    thought_attn = out.action.thought_attention[0]  # [total_queries, thoughtlets]

    total_attn_sum = float(state_attn.sum().item())
    thought_attn_sum = float(thought_attn.sum().item())
    fraction_to_thoughts = thought_attn_sum / (total_attn_sum + 1e-6)

    print(f"\n--- Baseline Actuator Cross-Attention Allocation ---")
    print(f"  Total Attention Mass: {total_attn_sum:.2f}")
    print(f"  Thought Attention Mass: {thought_attn_sum:.2f}")
    print(f"  Fraction of Actuator Attention to Thoughts: {fraction_to_thoughts * 100:.2f}%")

    print("\n[OK] Diagnostic complete.")


if __name__ == "__main__":
    main()
