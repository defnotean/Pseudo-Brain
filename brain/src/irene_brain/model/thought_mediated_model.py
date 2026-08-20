"""Thought-Mediated Brain Model and Proposal GRU Baseline with Consequence-Driven Actuators.

Implements:
1. ThoughtMediatedBrainModel:
   - Full thought-mediated information flow: Sensors -> Belief -> Thoughts -> Action-Conditioned Consequence Proposals -> Outcome-Driven Utility Selection -> Main Action Intent (+ Bounded Reflex).
   - No direct connection from belief to main action.
2. ProposalGRUBaseline:
   - Monolithic GRU baseline equipped with the identical consequence-proposal action head (critical control baseline).
3. Resource-Matched Scaling Factory:
   - Constructs matched models across K in {1, 4, 8, 16, 32} with hidden width compensation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .consequence_thought_actuator import ConsequenceActionOutput, ConsequenceThoughtActuator
from .spec import ThoughtFieldConfig
from .torch_model import BrainState, IreneBrainModel
from ..types import GenericControl


@dataclass(frozen=True, slots=True)
class ThoughtMediatedModelOutput:
    action: ConsequenceActionOutput
    next_state: BrainState


class ThoughtMediatedBrainModel(nn.Module):
    """Pseudo-Brain Model with Consequence-Driven Thought-Mediated Flow (No Belief Bypass)."""

    def __init__(self, config: ThoughtFieldConfig) -> None:
        super().__init__()
        self.config = config
        self.base_brain = IreneBrainModel(config)
        self.actuator = ConsequenceThoughtActuator(
            core_width=config.core_width,
            num_buttons=config.actuator.keyboard_keys + config.actuator.mouse_buttons,
            max_reflex_delta=0.02,
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
        thought_noise: Tensor | None = None,
        thought_intervention: str | None = None,
        donor_thoughts: Tensor | None = None,
        slot_permutation: Sequence[int] | None = None,
        scramble_prob: bool = False,
        scramble_binding: bool = False,
        **kwargs: Any,
    ) -> ThoughtMediatedModelOutput:
        batch = rgb.shape[0]
        if state is None:
            state = self.initial_state(batch)

        # Run sensory, memory, and cognitive cycle transitions through base brain
        base_out = self.base_brain(
            rgb,
            control,
            delta_time,
            state=state,
            max_cycles=max_cycles,
            thought_noise=thought_noise,
        )
        next_state = base_out.next_state
        thoughts = next_state.thoughts

        # Capacity masking if active_slots is constrained
        active_mask = None
        if active_slots is not None:
            active_mask = torch.zeros((batch, self.config.thoughtlets), device=rgb.device)
            if active_slots > 0:
                active_mask[:, :min(active_slots, self.config.thoughtlets)] = 1.0
            thoughts = thoughts * active_mask.unsqueeze(-1).unsqueeze(-1)

        # Progressive Causal Thought Interventions (applied directly before actuator decoding)
        if thought_intervention == "permute" and slot_permutation is not None:
            thoughts = thoughts[:, slot_permutation]
        elif thought_intervention == "register_swap":
            if thoughts.ndim == 4 and thoughts.shape[2] >= 2:
                thoughts = thoughts.clone()
                thoughts[:, :, 0], thoughts[:, :, 1] = thoughts[:, :, 1].clone(), thoughts[:, :, 0].clone()
        elif thought_intervention == "stale" and donor_thoughts is not None:
            thoughts = donor_thoughts.to(thoughts.device)
            active_mask = torch.ones((batch, self.config.thoughtlets), device=rgb.device)
        elif thought_intervention == "donor" and donor_thoughts is not None:
            thoughts = donor_thoughts.to(thoughts.device)
            active_mask = torch.ones((batch, self.config.thoughtlets), device=rgb.device)
        elif thought_intervention == "gaussian":
            noise_scale = thoughts.std().clamp_min(1e-2).item()
            thoughts = torch.randn_like(thoughts) * noise_scale
            active_mask = torch.ones((batch, self.config.thoughtlets), device=rgb.device)
        elif thought_intervention == "zero_active":
            # Semantic Zero Content BUT Active Pathway Forced ON
            thoughts = torch.zeros_like(thoughts)
            active_mask = torch.ones((batch, self.config.thoughtlets), device=rgb.device)
        elif thought_intervention in ("zero", "zero_knockout") or active_slots == 0:
            # Complete Pathway Knockout
            thoughts = torch.zeros_like(thoughts)
            active_mask = torch.zeros((batch, self.config.thoughtlets), device=rgb.device)

        # Decode action EXCLUSIVELY through consequence proposals + bounded sensory reflex
        sensors = next_state.belief  # [B, S, Width]
        final_scramble_prob = scramble_prob or (thought_intervention == "scramble_prob")
        final_scramble_binding = scramble_binding or (thought_intervention == "scramble_binding")
        action_out = self.actuator(
            sensors=sensors,
            thoughts=thoughts,
            active_mask=active_mask,
            scramble_prob=final_scramble_prob,
            scramble_binding=final_scramble_binding,
        )

        return ThoughtMediatedModelOutput(
            action=action_out,
            next_state=next_state,
        )


class ProposalGRUBaseline(nn.Module):
    """Monolithic GRU Recurrent Baseline with Identical Consequence-Proposal Aggregator."""

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
        self.proposal_expander = nn.Linear(hidden_dim, num_proposals * hidden_dim)
        self.actuator = ConsequenceThoughtActuator(
            core_width=hidden_dim,
            num_buttons=num_buttons,
            max_reflex_delta=0.02,
            temperature=0.10,
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

        # Monolithic GRU decodes full M=21 branch hypotheses from its single compressed vector
        expanded_slots = self.proposal_expander(next_h).view(batch, self.num_proposals, 1, self.hidden_dim)
        pseudo_sensors = enc_feats.unsqueeze(1)

        scramble_prob = kwargs.get("scramble_prob", False)
        scramble_binding = kwargs.get("scramble_binding", False)
        action_out = self.actuator(
            sensors=pseudo_sensors,
            thoughts=expanded_slots,
            scramble_prob=scramble_prob,
            scramble_binding=scramble_binding,
        )

        @dataclass(frozen=True, slots=True)
        class GRUOutput:
            action: Any
            next_state: Tensor

        return GRUOutput(action=action_out, next_state=next_h)


# Historical compensating widths for the default 819,894-parameter envelope.
# Search is reserved for non-default envelopes so CPU tests stay cheap.
_RESOURCE_MATCHED_WIDTH_SEED: dict[int, int] = {
    1: 180,
    4: 92,
    5: 80,
    8: 64,
    16: 48,
    32: 32,
}


@lru_cache(maxsize=32)
def find_resource_matched_width(k_slots: int, target_params: int = 819_894, tolerance: float = 0.05) -> int:
    """Find core_width W (divisible by 4) matching the target parameter envelope."""
    seed = _RESOURCE_MATCHED_WIDTH_SEED.get(k_slots, 60)
    if target_params == 819_894 and k_slots in _RESOURCE_MATCHED_WIDTH_SEED:
        return seed

    best_w = seed
    best_diff = float("inf")
    routed_neighbors = 0 if k_slots == 1 else (1 if k_slots in (4, 5) else 2)
    lo = max(16, seed - 48)
    hi = min(256, seed + 48)
    for w in range(lo, hi + 1, 4):
        cfg = ThoughtFieldConfig(
            thoughtlets=k_slots,
            core_width=w,
            attention_heads=4,
            routed_neighbors=routed_neighbors,
            cognitive_cycles=3,
        )
        try:
            m = ThoughtMediatedBrainModel(cfg)
            p = sum(param.numel() for param in m.parameters())
            diff = abs(p - target_params)
            if diff < best_diff:
                best_diff = diff
                best_w = w
            if diff <= tolerance * target_params:
                return w
        except Exception:
            continue
    return best_w


def build_resource_matched_thought_model(k_slots: int, target_params: int = 819_894) -> ThoughtMediatedBrainModel:
    """Build resource-matched thought model with compensating width for K."""
    width = find_resource_matched_width(k_slots, target_params=target_params)
    routed_neighbors = 0 if k_slots == 1 else (1 if k_slots in (4, 5) else 2)
    config = ThoughtFieldConfig(
        thoughtlets=k_slots,
        core_width=width,
        attention_heads=4,
        routed_neighbors=routed_neighbors,
        cognitive_cycles=3,
    )
    return ThoughtMediatedBrainModel(config)

