"""Consequence-Driven Thought-Mediated Actuator Architecture.

Core Architectural Shift:
1. Thoughtlets Predict Action-Conditioned Consequences:
   - Each thoughtlet outputs:
     * action_condition (5-way categorical distribution: Wait, Up, Left, Down, Right)
     * displacement (dx, dy)
     * hazard_prob in [0, 1]
     * reward_estimate in R
     * confidence in [0, 1]
2. Outcome-Derived Utility:
   - The utility of each thoughtlet is NOT an arbitrary free parameter.
   - It is directly derived from predicted consequences:
     u_k = reward_k - 3.0 * hazard_k + 0.5 * log(confidence_k + 1e-4)
3. Consequence-Driven Action Selection:
   - The main action intent is formed by aggregating action conditions weighted by consequence utility:
     Action_Logits = sum_k softmax(u_k) * Action_Condition_k
   - Single slot (K=1) can represent only 1 action hypothesis.
   - Multiple slots (K=4..32) can represent all competing alternatives simultaneously.
4. Exchangeable Dynamic Slot Binding:
   - Action conditions dynamically travel with thoughts; no slot is hardcoded to a specific direction.
5. Strict Zero-Activation Gating:
   - Inactive thoughts have active_mask = 0, contributing exactly 0.0 to action and consequence outputs.
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


@dataclass(frozen=True, slots=True)
class ConsequenceProposal:
    """Action-conditioned future prediction emitted by a single thoughtlet."""
    action_logits: Tensor      # [B, K, 5] (0: Wait, 1: Up/W, 2: Left/A, 3: Down/S, 4: Right/D)
    action_probs: Tensor       # [B, K, 5]
    displacement: Tensor       # [B, K, 2] (dx, dy)
    hazard_prob: Tensor        # [B, K, 1] in [0, 1]
    reward_estimate: Tensor    # [B, K, 1]
    confidence: Tensor         # [B, K, 1] in [0, 1]
    consequence_utility: Tensor # [B, K, 1] derived utility score
    active_mask: Tensor        # [B, K] in {0, 1}


@dataclass(frozen=True, slots=True)
class ConsequenceActionOutput:
    """Complete output of the Consequence-Driven Actuator."""
    control: Tensor
    button_logits: Tensor
    proposals: ConsequenceProposal
    aggregation_weights: Tensor  # [B, K]
    reflex_correction: Tensor    # [B, num_buttons] in [-0.10, +0.10]
    main_action_intent: Tensor   # [B, num_buttons]


class SharedConsequenceProposalHead(nn.Module):
    """Shared neural projection mapping each thoughtlet to an action-conditioned consequence hypothesis."""

    def __init__(self, *, core_width: int, num_actions: int = 5) -> None:
        super().__init__()
        self.core_width = core_width
        self.num_actions = num_actions

        self.trunk = nn.Sequential(
            nn.Linear(core_width, core_width),
            nn.SiLU(),
            nn.Linear(core_width, core_width),
            nn.SiLU(),
        )
        self.action_head = nn.Linear(core_width, num_actions)
        self.displacement_head = nn.Linear(core_width, 2)
        self.hazard_head = nn.Linear(core_width, 1)
        self.reward_head = nn.Linear(core_width, 1)
        self.confidence_head = nn.Linear(core_width, 1)

    def forward(self, thoughts: Tensor, active_mask: Tensor | None = None) -> ConsequenceProposal:
        """Forward pass over unpooled thoughts.

        thoughts: [B, K, Registers, Width] or [B, K, Width]
        active_mask: [B, K] in {0, 1}
        """
        if thoughts.ndim == 4:
            slot_tokens = thoughts.mean(dim=2)  # [B, K, Width]
        else:
            slot_tokens = thoughts

        b, k, w = slot_tokens.shape

        if active_mask is None:
            thought_norms = torch.norm(slot_tokens, dim=-1)  # [B, K]
            active_mask = (thought_norms > 1e-4).float()
        else:
            active_mask = active_mask.float()

        mask_3d = active_mask.unsqueeze(-1)  # [B, K, 1]

        raw_features = self.trunk(slot_tokens)
        gated_features = raw_features * mask_3d

        # Heads with strict zero gating for inactive thoughts
        raw_act_logits = self.action_head(gated_features)
        action_logits = raw_act_logits * mask_3d + (1.0 - mask_3d) * (-1e9)
        action_probs = F.softmax(action_logits, dim=-1) * mask_3d

        displacement = self.displacement_head(gated_features) * mask_3d
        hazard_prob = torch.sigmoid(self.hazard_head(gated_features)) * mask_3d
        reward_estimate = self.reward_head(gated_features) * mask_3d
        confidence = torch.sigmoid(self.confidence_head(gated_features)) * mask_3d

        # Derive consequence utility: U = Reward - 3.0 * Hazard + 0.5 * log(conf + 1e-4)
        raw_util = reward_estimate - 3.0 * hazard_prob + 0.5 * torch.log(confidence.clamp_min(1e-4))
        consequence_utility = raw_util * mask_3d + (1.0 - mask_3d) * (-1e9)

        return ConsequenceProposal(
            action_logits=action_logits,
            action_probs=action_probs,
            displacement=displacement,
            hazard_prob=hazard_prob,
            reward_estimate=reward_estimate,
            confidence=confidence,
            consequence_utility=consequence_utility,
            active_mask=active_mask,
        )


class ConsequenceProposalAggregator(nn.Module):
    """Compares predicted future consequences across K thoughtlets and selects the winning action."""

    def __init__(self, *, temperature: float = 1.0, num_buttons: int = 296) -> None:
        super().__init__()
        self.temperature = temperature
        self.num_buttons = num_buttons

    def forward(self, proposals: ConsequenceProposal) -> tuple[Tensor, Tensor]:
        """Aggregate action conditions weighted by consequence utilities into 296-channel button logits.

        Returns: (main_action_intent: [B, num_buttons], aggregation_weights: [B, K])
        """
        active_mask = proposals.active_mask  # [B, K]
        num_active = active_mask.sum(dim=-1, keepdim=True)  # [B, 1]
        device = active_mask.device
        batch_size, k_slots = active_mask.shape

        # Softmax over active consequence utilities
        raw_weights = F.softmax(proposals.consequence_utility.squeeze(-1) / self.temperature, dim=-1)  # [B, K]
        active_weights = raw_weights * active_mask
        weight_sum = active_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        normalized_weights = active_weights / weight_sum  # [B, K]

        has_active = (num_active > 0).float()
        final_weights = normalized_weights * has_active  # [B, K]

        # Aggregate 5-way action distributions: [B, K, 1] * [B, K, 5] -> [B, 5]
        action_dist = (final_weights.unsqueeze(-1) * proposals.action_probs).sum(dim=1)  # [B, 5]

        # Project 5-way actions to 296-channel HID button wire layout
        # 0: Wait, 1: W, 2: A, 3: S, 4: D
        main_intent = torch.zeros((batch_size, self.num_buttons), device=device)
        main_intent[:, int(HidKey.W)] = action_dist[:, 1] * 5.0
        main_intent[:, int(HidKey.A)] = action_dist[:, 2] * 5.0
        main_intent[:, int(HidKey.S)] = action_dist[:, 3] * 5.0
        main_intent[:, int(HidKey.D)] = action_dist[:, 4] * 5.0

        main_intent = main_intent * has_active

        return main_intent, final_weights


class ConsequenceThoughtActuator(nn.Module):
    """Complete Consequence-Driven Actuator without bypass."""

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
        self.proposal_head = SharedConsequenceProposalHead(core_width=core_width)
        self.aggregator = ConsequenceProposalAggregator(temperature=temperature, num_buttons=num_buttons)
        self.reflex_linear = nn.Linear(core_width, num_buttons)
        self.max_reflex_delta = max_reflex_delta

    def forward(
        self,
        *,
        sensors: Tensor,
        thoughts: Tensor,
        active_mask: Tensor | None = None,
    ) -> ConsequenceActionOutput:
        proposals = self.proposal_head(thoughts, active_mask=active_mask)
        main_intent, weights = self.aggregator(proposals)

        # Bounded reflex
        sensor_summary = sensors.mean(dim=1)
        reflex = self.max_reflex_delta * torch.tanh(self.reflex_linear(sensor_summary))

        final_button_logits = main_intent + reflex
        control = final_button_logits.clone()

        return ConsequenceActionOutput(
            control=control,
            button_logits=final_button_logits,
            proposals=proposals,
            aggregation_weights=weights,
            reflex_correction=reflex,
            main_action_intent=main_intent,
        )
