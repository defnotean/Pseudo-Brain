"""Thought-Mediated Brain Model and Proposal GRU Baseline.

Implements:
1. ThoughtMediatedBrainModel:
   - Full thought-mediated information flow: Sensors -> Belief -> Thoughts -> Per-Thoughtlet Proposals -> Main Action Intent (+ Bounded Reflex).
   - No direct connection from belief to main action.
2. ProposalGRUBaseline:
   - Monolithic GRU baseline equipped with the identical proposal-style action head (critical control baseline).
3. Resource-Matched Scaling Factory:
   - Constructs matched models across K in {1, 4, 8, 16, 32} with hidden width compensation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .spec import ThoughtFieldConfig
from .thought_mediated_actuator import ThoughtMediatedActionOutput, ThoughtMediatedActuator
from .torch_model import BrainState, IreneBrainModel
from ..types import GenericControl


@dataclass(frozen=True, slots=True)
class ThoughtMediatedModelOutput:
    action: ThoughtMediatedActionOutput
    next_state: BrainState


class ThoughtMediatedBrainModel(nn.Module):
    """Pseudo-Brain Model with Thought-Mediated Action Flow (No Belief Bypass)."""

    def __init__(self, config: ThoughtFieldConfig) -> None:
        super().__init__()
        self.config = config
        self.base_brain = IreneBrainModel(config)
        self.actuator = ThoughtMediatedActuator(
            core_width=config.core_width,
            num_buttons=config.actuator.keyboard_keys + config.actuator.mouse_buttons,
            max_reflex_delta=0.10,
        )

    def initial_state(self, batch_size: int = 1) -> BrainState:
        return self.base_brain.initial_state(batch_size)

    def forward(
        self,
        rgb: Tensor,
        control: Tensor,
        delta_time: Tensor,
        *,
        state: BrainState | None = None,
        max_cycles: int | None = None,
        active_slots: int | None = None,
    ) -> ThoughtMediatedModelOutput:
        batch = rgb.shape[0]
        if state is None:
            state = self.initial_state(batch)

        # Run sensory, memory, and cognitive cycle transitions through base brain
        base_out = self.base_brain(rgb, control, delta_time, state=state, max_cycles=max_cycles)
        next_state = base_out.next_state

        thoughts = next_state.thoughts  # [B, K, Registers, Width]

        # Capacity masking if active_slots is constrained
        if active_slots is not None and active_slots < self.config.thoughtlets:
            mask = torch.zeros_like(thoughts)
            mask[:, :active_slots] = 1.0
            thoughts = thoughts * mask

        # Decode action EXCLUSIVELY through thought proposals + bounded sensory reflex
        sensors = next_state.belief  # [B, S, Width]
        action_out = self.actuator(sensors=sensors, thoughts=thoughts)

        return ThoughtMediatedModelOutput(
            action=action_out,
            next_state=next_state,
        )


class ProposalGRUBaseline(nn.Module):
    """Monolithic GRU Recurrent Baseline with Identical Proposal Aggregator."""

    def __init__(self, *, hidden_dim: int = 180, num_proposals: int = 1, num_buttons: int = 296) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_proposals = num_proposals
        self.num_buttons = num_buttons

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.encoder = nn.Linear(32 * 8 * 8, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.actuator = ThoughtMediatedActuator(
            core_width=hidden_dim,
            num_buttons=num_buttons,
            max_reflex_delta=0.10,
        )

    def initial_state(self, batch_size: int = 1) -> Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def forward(
        self,
        rgb: Tensor,
        control: Tensor,
        delta_time: Tensor,
        *,
        state: Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        batch = rgb.shape[0]
        if state is None:
            state = self.initial_state(batch).to(rgb.device)

        conv_feats = self.conv(rgb)
        enc_feats = self.encoder(conv_feats)
        next_h = self.gru(enc_feats, state)

        # Reshape monolithic hidden state as proposal slots: [B, 1, 1, HiddenDim]
        pseudo_thoughts = next_h.unsqueeze(1).unsqueeze(1)
        pseudo_sensors = enc_feats.unsqueeze(1)

        action_out = self.actuator(sensors=pseudo_sensors, thoughts=pseudo_thoughts)

        @dataclass(frozen=True, slots=True)
        class GRUOutput:
            action: Any
            next_state: Tensor

        return GRUOutput(action=action_out, next_state=next_h)


def build_resource_matched_thought_model(k_slots: int) -> ThoughtMediatedBrainModel:
    """Build resource-matched thought model with compensating width and valid topology constraints."""
    width_map = {
        1: 180,
        4: 92,
        8: 64,
        16: 48,
        32: 32,
    }
    width = width_map.get(k_slots, 32)
    routed_neighbors = 0 if k_slots == 1 else (1 if k_slots == 4 else 2)
    config = ThoughtFieldConfig(
        thoughtlets=k_slots,
        core_width=width,
        attention_heads=4,
        routed_neighbors=routed_neighbors,
        cognitive_cycles=3,
    )
    return ThoughtMediatedBrainModel(config)
