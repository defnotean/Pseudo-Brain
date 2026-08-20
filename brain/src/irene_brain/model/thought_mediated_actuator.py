"""Thought-Mediated Actuator Architecture: Zero-Shortcut Causal Routing & Stochastic VoI.

Key Principles:
1. No Thought Activation = EXACTLY ZERO Action Contribution.
2. Grouped Expected Utility & Value of Information (VoI):
   - Each thoughtlet carries: (action, displacement, reward, hazard, branch_prob, confidence).
   - Committing actions evaluate expected utility over branch probabilities: Q(a) = Sum_j p_j * U(T_j).
   - Probing actions (WAIT) evaluate Value of Information: Q(WAIT) = c_wait + gamma * E_obs[max_a' Q(a' | obs)].
3. Permutation Equivariance & Bounded Reflex.
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
    button_logits: Tensor           # [B, K, num_buttons]
    hazard_prob: Tensor             # [B, K, 1] in [0, 1]
    reward_estimate: Tensor         # [B, K, 1]
    displacement: Tensor            # [B, K, 2] (dx, dy)
    utility_logits: Tensor          # [B, K, 1] unnormalized aggregation weight
    active_mask: Tensor             # [B, K] in {0, 1}
    branch_probability: Tensor      # [B, K, 1] in [0, 1] (aleatoric branch probability)
    epistemic_confidence: Tensor    # [B, K, 1] in [0, 1] (model certainty)


@dataclass(frozen=True, slots=True)
class ThoughtMediatedActionOutput:
    """Complete output of the Thought-Mediated Actuator."""
    control: Tensor
    button_logits: Tensor
    proposals: ThoughtProposal
    aggregation_weights: Tensor     # [B, K]
    reflex_correction: Tensor       # [B, num_buttons] in [-0.1, +0.1]
    main_action_intent: Tensor      # [B, num_buttons]


class SharedPerThoughtletProposalHead(nn.Module):
    """Shared neural projection mapping each individual thoughtlet to action, future, prob & VoI."""

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
        self.branch_prob_head = nn.Linear(core_width, 1)
        self.confidence_head = nn.Linear(core_width, 1)

    def forward(self, thoughts: Tensor, active_mask: Tensor | None = None) -> ThoughtProposal:
        """Forward pass over unpooled thoughts with strict active-gating."""
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

        button_logits = self.action_head(gated_features) * mask_3d
        hazard_prob = torch.sigmoid(self.hazard_head(gated_features)) * mask_3d
        reward_estimate = self.reward_head(gated_features) * mask_3d
        displacement = self.displacement_head(gated_features) * mask_3d
        branch_probability = torch.sigmoid(self.branch_prob_head(gated_features)) * mask_3d
        epistemic_confidence = torch.sigmoid(self.confidence_head(gated_features)) * mask_3d

        raw_utility = self.utility_head(gated_features)
        utility_logits = raw_utility * mask_3d + (1.0 - mask_3d) * (-1e9)

        return ThoughtProposal(
            button_logits=button_logits,
            hazard_prob=hazard_prob,
            reward_estimate=reward_estimate,
            displacement=displacement,
            utility_logits=utility_logits,
            active_mask=active_mask,
            branch_probability=branch_probability,
            epistemic_confidence=epistemic_confidence,
        )


class PermutationInvariantProposalAggregator(nn.Module):
    """Aggregates K thoughtlet proposals into main action intent using probability, confidence, and utility softmax."""

    def __init__(self, *, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        proposals: ThoughtProposal,
        *,
        scramble_prob: bool = False,
        scramble_binding: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Aggregate proposals into main action intent."""
        active_mask = proposals.active_mask  # [B, K]
        num_active = active_mask.sum(dim=-1, keepdim=True)  # [B, 1]
        has_active = (num_active > 0).float()

        u = proposals.utility_logits.squeeze(-1)  # [B, K]
        p = getattr(proposals, "branch_probability", torch.ones_like(proposals.utility_logits)).squeeze(-1)  # [B, K]
        conf = getattr(proposals, "epistemic_confidence", torch.ones_like(proposals.utility_logits)).squeeze(-1)  # [B, K]

        if (scramble_prob or scramble_binding) and p.shape[-1] > 1:
            perm = torch.randperm(p.shape[-1], device=p.device)
            p = p[:, perm]
            if scramble_binding:
                conf = conf[:, perm]

        raw_weights = F.softmax(u / self.temperature, dim=-1)  # [B, K]
        effective_weights = raw_weights * active_mask * p.clamp_min(0.01) * conf.clamp_min(0.01)

        weight_sum = effective_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        final_weights = (effective_weights / weight_sum) * has_active  # [B, K]

        main_intent = (final_weights.unsqueeze(-1) * proposals.button_logits).sum(dim=1)
        main_intent = main_intent * has_active

        return main_intent, final_weights


class BoundedReflexHead(nn.Module):
    """Small linear reflex path for physical stabilization only (bounded to [-delta_max, +delta_max])."""

    def __init__(self, *, core_width: int, num_buttons: int = 296, max_delta: float = 0.10) -> None:
        super().__init__()
        self.max_delta = max_delta
        self.linear = nn.Linear(core_width, num_buttons)

    def forward(self, sensors: Tensor) -> Tensor:
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
        scramble_prob: bool = False,
        scramble_binding: bool = False,
    ) -> ThoughtMediatedActionOutput:
        """Decode control through strictly gated thought proposals and bounded reflex."""
        proposals = self.proposal_head(thoughts, active_mask=active_mask)
        main_intent, weights = self.aggregator(
            proposals,
            scramble_prob=scramble_prob,
            scramble_binding=scramble_binding,
        )
        reflex = self.reflex_head(sensors)

        final_button_logits = main_intent + reflex
        control = final_button_logits.clone()

        return ThoughtMediatedActionOutput(
            control=control,
            button_logits=final_button_logits,
            proposals=proposals,
            aggregation_weights=weights,
            reflex_correction=reflex,
            main_action_intent=main_intent,
        )
