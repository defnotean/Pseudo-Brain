"""Multiplicity-proof action-conditional expected-utility aggregation.

Duplicate slots that propose the same discrete action must not dominate Q(a)
by vote count. Each action's expected utility is a probability-weighted mean
over the slots that actually propose that action.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def aggregate_action_conditional_expected_utilities(
    button_logits: torch.Tensor,
    utility_logits: torch.Tensor,
    branch_prob: torch.Tensor,
    active_mask: torch.Tensor,
    dir_indices: list[int] | None = None,
) -> torch.Tensor:
    """Return Q(a) for Wait/W/A/S/D without duplicate-slot majority bias.

    button_logits: [B, K, NumButtons]
    utility_logits: [B, K, 1]
    branch_prob: [B, K, 1]
    active_mask: [B, K]
    dir_indices: HID indices for W, A, S, D (defaults to HidKey layout)
    """
    if dir_indices is None:
        dir_indices = [26, 4, 22, 7]
    batch, k_slots, _num_buttons = button_logits.shape
    utility = utility_logits.squeeze(-1)
    probability = branch_prob.squeeze(-1)

    dir_logits = button_logits[:, :, dir_indices]
    wait_logits = torch.zeros((batch, k_slots, 1), device=button_logits.device, dtype=button_logits.dtype)
    all_5_logits = torch.cat([wait_logits, dir_logits], dim=-1)
    slot_action_probs = F.softmax(all_5_logits, dim=-1)

    q_actions = []
    for action_idx in range(5):
        slot_weight = slot_action_probs[:, :, action_idx] * active_mask * probability
        q_action = (slot_weight * utility).sum(dim=-1) / slot_weight.sum(dim=-1).clamp_min(1e-6)
        q_actions.append(q_action)
    return torch.stack(q_actions, dim=-1)
