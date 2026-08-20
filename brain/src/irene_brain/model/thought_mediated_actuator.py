"""Thought-Mediated Actuator Architecture: Eliminating the Belief Bypass.

Architectural Flow:
1. Sensors -> Belief -> Thought Field (K slots).
2. Shared Per-Thoughtlet Proposal Head:
   - Each thoughtlet outputs: candidate action logits, future hazard, reward estimate, and utility score.
3. Permutation-Invariant Proposal Aggregator:
   - Softmax-weighted mixture over K proposals based on utility scores.
   - Main action intent is decoded EXCLUSIVELY from thought proposals (no direct connection from belief).
4. Bounded Reflex Path:
   - Maps raw sensors to bounded [-0.10, +0.10] physical correction delta for short-timescale stabilization.
5. Final Control = Main Action Intent + Bounded Reflex Correction.
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

    def forward(self, thoughts: Tensor) -> ThoughtProposal:
        """Forward pass over unpooled thoughts.

        thoughts: [B, K, Registers, Width] or [B, K, Width]
        """
        if thoughts.ndim == 4:
            slot_tokens = thoughts.mean(dim=2)  # [B, K, Width]
        else:
            slot_tokens = thoughts

        features = self.trunk(slot_tokens)  # [B, K, Width]

        return ThoughtProposal(
            button_logits=self.action_head(features),
            hazard_prob=torch.sigmoid(self.hazard_head(features)),
            reward_estimate=self.reward_head(features),
            displacement=self.displacement_head(features),
            utility_logits=self.utility_head(features),
        )


class PermutationInvariantProposalAggregator(nn.Module):
    """Aggregates K thoughtlet proposals into main action intent using utility softmax."""

    def __init__(self, *, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, proposals: ThoughtProposal) -> tuple[Tensor, Tensor]:
        """Aggregate proposals into main action intent.

        Returns: (main_action_intent: [B, num_buttons], weights: [B, K])
        """
        weights = F.softmax(proposals.utility_logits.squeeze(-1) / self.temperature, dim=-1)  # [B, K]
        # Weighted sum: [B, K, 1] * [B, K, num_buttons] -> sum over K -> [B, num_buttons]
        main_intent = (weights.unsqueeze(-1) * proposals.button_logits).sum(dim=1)
        return main_intent, weights


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
    ) -> ThoughtMediatedActionOutput:
        """Decode control through thought proposals and bounded reflex."""
        proposals = self.proposal_head(thoughts)
        main_intent, weights = self.aggregator(proposals)
        reflex = self.reflex_head(sensors)

        final_button_logits = main_intent + reflex

        # Dummy control layout for wire compatibility
        control = final_button_logits.clone()

        return ThoughtMediatedActionOutput(
            control=control,
            button_logits=final_button_logits,
            proposals=proposals,
            aggregation_weights=weights,
            reflex_correction=reflex,
            main_action_intent=main_intent,
        )
