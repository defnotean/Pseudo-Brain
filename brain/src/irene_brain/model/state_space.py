"""Selective State Space Model (SSM / S4 style) recurrent baseline."""

from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .brain_cell import ContinuousTimeBlend, ResidualCrossAttention, RoutingDiagnostics
from .spec import ThoughtFieldConfig
from .torch_model import IreneBrainModel


class StateSpaceRecurrentBlock(nn.Module):
    """Selective linear state-space recurrent block (SSM/S4 style control).

    Replaces multi-thought routing with a continuous-time selective state space model:
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C * h_t + D * x_t
    where A_bar = exp(-softplus(Delta) * exp(A_log)) and B_bar = Delta * B(x_t).
    """

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.width = width
        self.belief_attention = ResidualCrossAttention(width=width, heads=heads)
        self.memory_attention = ResidualCrossAttention(width=width, heads=heads)
        self.belief_blend = ContinuousTimeBlend(width)
        self.thought_blend = ContinuousTimeBlend(width)
        self.memory_blend = ContinuousTimeBlend(width)
        self.context_projection = nn.Sequential(
            nn.Linear(width * 6, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )
        self.a_log = nn.Parameter(torch.randn(width))
        self.delta_proj = nn.Linear(width, width)
        self.b_proj = nn.Linear(width, width)
        self.c_proj = nn.Linear(width, width)
        self.d_weight = nn.Parameter(torch.ones(width))
        self.out_proj = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.LayerNorm(width),
        )

    def forward(
        self,
        *,
        belief: Tensor,
        working_memory: Tensor,
        thoughts: Tensor,
        sensors: Tensor,
        action_time_tokens: Tensor,
        goal_context: Tensor,
        retrieved_memory: Tensor,
        elapsed_seconds: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, RoutingDiagnostics]:
        batch, thoughtlets, registers, width = thoughts.shape
        if thoughtlets != 1 or registers != 1 or width != self.width:
            raise ValueError(
                f"SSM recurrent state must have shape [batch, 1, 1, {self.width}], got {thoughts.shape}"
            )

        belief_context = torch.cat(
            (sensors, action_time_tokens, belief, working_memory),
            dim=1,
        )
        proposed_belief = self.belief_attention(belief, belief_context)
        belief = self.belief_blend(belief, proposed_belief, elapsed_seconds)

        previous = thoughts[:, 0, 0]
        pooled_context = torch.cat(
            (
                sensors.mean(dim=1),
                belief.mean(dim=1),
                action_time_tokens.mean(dim=1),
                goal_context.mean(dim=1),
                retrieved_memory.mean(dim=(1, 2)),
                working_memory.mean(dim=1),
            ),
            dim=-1,
        )
        x = self.context_projection(pooled_context)
        delta = F.softplus(self.delta_proj(x))
        a_decay = torch.exp(-delta * F.softplus(self.a_log))
        b = self.b_proj(x)
        proposed_h = a_decay * previous + delta * b
        y = self.c_proj(proposed_h) + self.d_weight * x
        proposed = self.out_proj(y)

        thoughts = self.thought_blend(
            previous.unsqueeze(1),
            proposed.unsqueeze(1),
            elapsed_seconds,
        ).reshape(batch, 1, 1, width)

        memory_context = torch.cat(
            (belief, thoughts.flatten(1, 2), retrieved_memory.flatten(1, 2)),
            dim=1,
        )
        proposed_memory = self.memory_attention(working_memory, memory_context)
        working_memory = self.memory_blend(
            working_memory,
            proposed_memory,
            elapsed_seconds,
        )
        empty_indices = torch.empty(
            batch,
            1,
            0,
            device=thoughts.device,
            dtype=torch.long,
        )
        empty_weights = thoughts.new_empty((batch, 1, 0))
        return (
            belief,
            working_memory,
            thoughts,
            RoutingDiagnostics(empty_indices, empty_weights),
        )


class StateSpaceRecurrentCell(nn.Module):
    """A tied stack of selective state space blocks with the BrainCell call contract."""

    def __init__(self, *, width: int, heads: int, blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            StateSpaceRecurrentBlock(width=width, heads=heads)
            for _ in range(blocks)
        )

    def forward(
        self,
        *,
        belief: Tensor,
        working_memory: Tensor,
        thoughts: Tensor,
        sensors: Tensor,
        action_time_tokens: Tensor,
        goal_context: Tensor,
        retrieved_memory: Tensor,
        elapsed_seconds: Tensor,
        allow_routing: bool,
        allow_workspace_writes: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[RoutingDiagnostics, ...]]:
        del allow_routing, allow_workspace_writes
        routing: list[RoutingDiagnostics] = []
        for block in self.blocks:
            belief, working_memory, thoughts, block_routing = block(
                belief=belief,
                working_memory=working_memory,
                thoughts=thoughts,
                sensors=sensors,
                action_time_tokens=action_time_tokens,
                goal_context=goal_context,
                retrieved_memory=retrieved_memory,
                elapsed_seconds=elapsed_seconds,
            )
            routing.append(block_routing)
        if not routing:
            raise RuntimeError("StateSpaceRecurrentCell must contain at least one block")
        return belief, working_memory, thoughts, tuple(routing)


class StateSpaceRecurrentBaseline(IreneBrainModel):
    """Selective State Space Model (SSM / S4 style) baseline (Baseline 4)."""

    architecture_variant_id = "irene.state_space_ssm.selective_latent.v1"

    def _build_brain_cell(self, *, width: int) -> nn.Module:
        return StateSpaceRecurrentCell(
            width=width,
            heads=self.config.attention_heads,
            blocks=self.config.brain_cell_blocks,
        )
