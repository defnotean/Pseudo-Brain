"""Thought-Mediated Actuator Architecture: Zero-Shortcut Causal Routing.

Key Principles:
1. No Thought Activation = EXACTLY ZERO Action Contribution.
   - When a thought slot is empty or masked (active_mask_k = 0), its proposal logits,
     hazard prediction, reward estimate, and displacement are multiplied by 0.0.
   - Its utility logit is masked to -1e9 so it receives 0 attention weight.
2. Neutral Baseline on Complete Inactivity:
   - When ALL slots are inactive (K=0), the main action intent is strictly zero: [0, 0, ..., 0].
   - No proposal bias can ever leak into the main action intent when thoughts are empty.
3. Permutation Equivariance:
   - Shared proposal head on all K slots; proposal aggregation is permutation-invariant.
4. Bounded Reflex Path:
   - Raw sensory tokens map to [-delta_max, +delta_max] delta via tanh for physical stabilization only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ..types import HidKey
from .spec import ActuatorQuerySpec


@dataclass(frozen=True, slots=True)
class ThoughtProposal:
    """Action and future prediction proposal emitted by a single thoughtlet."""
    button_logits: Tensor       # [B, K, num_buttons]
    hazard_prob: Tensor         # [B, K, 1] in [0, 1]
    reward_estimate: Tensor     # [B, K, 1]
    displacement: Tensor        # [B, K, 2] (dx, dy)
    utility_logits: Tensor      # [B, K, 1] unnormalized aggregation weight
    active_mask: Tensor         # [B, K] in {0, 1}


@dataclass(frozen=True, slots=True)
class ThoughtMediatedActionOutput:
    """Complete output of the Thought-Mediated Actuator."""
    control: Tensor
    button_logits: Tensor
    proposals: ThoughtProposal
    aggregation_weights: Tensor  # [B, K]
    reflex_correction: Tensor    # [B, num_buttons] in [-0.1, +0.1]
    main_action_intent: Tensor   # [B, num_buttons]


class SharedPerThoughtletProposalHead(nn.Module):
    """Shared neural projection mapping each individual thoughtlet to an action & future proposal."""

    def __init__(self, *, core_width: int, num_buttons: int = 296) -> None:
        super().__init__()
        self.core_width = core_width
        self.num_buttons = num_buttons

        self.trunk = nn.Sequential(
            nn.Linear(core_width, core_width),
            nn.SiLU(),
            nn.Linear(core_width, core_width),
            nn.SiLU(),
        )
        self.action_head = nn.Linear(core_width, num_buttons)
        self.hazard_head = nn.Linear(core_width, 1)
        self.reward_head = nn.Linear(core_width, 1)
        self.displacement_head = nn.Linear(core_width, 2)
        self.utility_head = nn.Linear(core_width, 1)

    def forward(self, thoughts: Tensor, active_mask: Tensor | None = None) -> ThoughtProposal:
        """Forward pass over unpooled thoughts with strict active-gating.

        thoughts: [B, K, Registers, Width] or [B, K, Width]
        active_mask: [B, K] in {0, 1}
        """
        if thoughts.ndim == 4:
            slot_tokens = thoughts.mean(dim=2)  # [B, K, Width]
        else:
            slot_tokens = thoughts

        b, k, w = slot_tokens.shape

        if active_mask is None:
            # Structurally detect active thoughts by norm > 1e-4
            thought_norms = torch.norm(slot_tokens, dim=-1)  # [B, K]
            active_mask = (thought_norms > 1e-4).float()
        else:
            active_mask = active_mask.float()

        mask_3d = active_mask.unsqueeze(-1)  # [B, K, 1]

        # Pass through trunk and enforce exact zero gating
        raw_features = self.trunk(slot_tokens)
        gated_features = raw_features * mask_3d

        # Compute heads with strict mask multiplication (dead slots emit exactly 0.0)
        button_logits = self.action_head(gated_features) * mask_3d
        hazard_prob = torch.sigmoid(self.hazard_head(gated_features)) * mask_3d
        reward_estimate = self.reward_head(gated_features) * mask_3d
        displacement = self.displacement_head(gated_features) * mask_3d

        # Utility logit receives -1e9 for inactive slots so softmax ignores them completely
        raw_utility = self.utility_head(gated_features)
        utility_logits = raw_utility * mask_3d + (1.0 - mask_3d) * (-1e9)

        return ThoughtProposal(
            button_logits=button_logits,
            hazard_prob=hazard_prob,
            reward_estimate=reward_estimate,
            displacement=displacement,
            utility_logits=utility_logits,
            active_mask=active_mask,
        )


class PermutationInvariantProposalAggregator(nn.Module):
    """Aggregates K thoughtlet proposals into main action intent using utility softmax over active slots."""

    def __init__(self, *, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, proposals: ThoughtProposal) -> tuple[Tensor, Tensor]:
        """Aggregate proposals into main action intent.

        If all slots are inactive (K=0), returns strictly neutral zero action intent: [0, 0, ..., 0].
        Returns: (main_action_intent: [B, num_buttons], weights: [B, K])
        """
        active_mask = proposals.active_mask  # [B, K]
        num_active = active_mask.sum(dim=-1, keepdim=True)  # [B, 1]

        # Softmax over masked utility logits
        raw_weights = F.softmax(proposals.utility_logits.squeeze(-1) / self.temperature, dim=-1)  # [B, K]
        # Zero out weights for inactive slots
        active_weights = raw_weights * active_mask

        # Re-normalize over active slots if any are active
        weight_sum = active_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        normalized_weights = active_weights / weight_sum  # [B, K]

        # If batch element has 0 active slots, weight is strictly 0.0
        has_active = (num_active > 0).float()
        final_weights = normalized_weights * has_active  # [B, K]

        # Weighted sum: [B, K, 1] * [B, K, num_buttons] -> sum over K -> [B, num_buttons]
        main_intent = (final_weights.unsqueeze(-1) * proposals.button_logits).sum(dim=1)

        # Strict guarantee: if 0 active slots, main_intent is exactly 0.0
        main_intent = main_intent * has_active

        return main_intent, final_weights


class BoundedReflexHead(nn.Module):
    """Small linear reflex path for physical stabilization only (bounded to [-delta_max, +delta_max])."""

    def __init__(self, *, core_width: int, num_buttons: int = 296, max_delta: float = 0.10) -> None:
        super().__init__()
        self.max_delta = max_delta
        self.linear = nn.Linear(core_width, num_buttons)

    def forward(self, sensors: Tensor) -> Tensor:
        """Compute bounded reflex delta from sensory tokens.

        sensors: [B, S, Width]
        """
        sensor_summary = sensors.mean(dim=1)  # [B, Width]
        raw_delta = self.linear(sensor_summary)
        return self.max_delta * torch.tanh(raw_delta)


class ThoughtMediatedActuator(nn.Module):
    """Complete Thought-Mediated Actuator without belief-to-action bypass."""

    def __init__(
        self,
        *,
        core_width: int,
        num_buttons: int = 296,
        max_reflex_delta: float = 0.10,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.core_width = core_width
        self.num_buttons = num_buttons
        self.proposal_head = SharedPerThoughtletProposalHead(core_width=core_width, num_buttons=num_buttons)
        self.aggregator = PermutationInvariantProposalAggregator(temperature=temperature)
        self.reflex_head = BoundedReflexHead(core_width=core_width, num_buttons=num_buttons, max_delta=max_reflex_delta)

    def forward(
        self,
        *,
        sensors: Tensor,
        thoughts: Tensor,
        active_mask: Tensor | None = None,
    ) -> ThoughtMediatedActionOutput:
        """Decode control through strictly gated thought proposals and bounded reflex."""
        proposals = self.proposal_head(thoughts, active_mask=active_mask)
        main_intent, weights = self.aggregator(proposals)
        reflex = self.reflex_head(sensors)

        final_button_logits = main_intent + reflex

        # Control vector wire compatibility
        control = final_button_logits.clone()

        return ThoughtMediatedActionOutput(
            control=control,
            button_logits=final_button_logits,
            proposals=proposals,
            aggregation_weights=weights,
            reflex_correction=reflex,
            main_action_intent=main_intent,
        )
