"""Tied recurrent cell and sparse thoughtlet communication."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class RoutingDiagnostics:
    indices: Tensor
    weights: Tensor


class ContinuousTimeBlend(nn.Module):
    """Learn a per-feature update rate while respecting real elapsed time."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.rate_unconstrained = nn.Parameter(torch.full((width,), 2.0))

    def forward(self, previous: Tensor, proposed: Tensor, elapsed_seconds: Tensor) -> Tensor:
        rate = F.softplus(self.rate_unconstrained).to(dtype=previous.dtype)
        gate = -torch.expm1(-rate * elapsed_seconds)
        while gate.ndim < previous.ndim:
            gate = gate.unsqueeze(1)
        return previous + gate * (proposed - previous)


class ResidualCrossAttention(nn.Module):
    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.SiLU(),
            nn.Linear(width * 4, width),
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        hidden = self.attention_norm(query + attended)
        return self.output_norm(hidden + self.feed_forward(hidden))


class StructuredBrainBlock(nn.Module):
    """One block shared by every slot and reused on every cognitive cycle."""

    def __init__(self, *, width: int, heads: int, routed_neighbors: int) -> None:
        super().__init__()
        self.width = width
        self.routed_neighbors = routed_neighbors
        self.belief_attention = ResidualCrossAttention(width=width, heads=heads)
        self.thought_attention = ResidualCrossAttention(width=width, heads=heads)
        self.memory_attention = ResidualCrossAttention(width=width, heads=heads)
        self.belief_blend = ContinuousTimeBlend(width)
        self.thought_blend = ContinuousTimeBlend(width)
        self.memory_blend = ContinuousTimeBlend(width)
        self.route_query = nn.Linear(width, width, bias=False)
        self.route_key = nn.Linear(width, width, bias=False)
        self.route_value = nn.Linear(width, width, bias=False)
        self.utility = nn.Linear(width, 1)

    def _route(
        self,
        summaries: Tensor,
        *,
        allow_routing: bool,
    ) -> tuple[Tensor, RoutingDiagnostics]:
        batch, thoughtlets, width = summaries.shape
        if not allow_routing:
            empty_indices = torch.empty(
                batch,
                thoughtlets,
                0,
                device=summaries.device,
                dtype=torch.long,
            )
            empty_weights = summaries.new_empty((batch, thoughtlets, 0))
            return torch.zeros_like(summaries), RoutingDiagnostics(empty_indices, empty_weights)

        query = self.route_query(summaries)
        key = self.route_key(summaries)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(width)
        diagonal = torch.eye(thoughtlets, device=summaries.device, dtype=torch.bool)
        scores = scores.masked_fill(diagonal.unsqueeze(0), torch.finfo(scores.dtype).min)
        values, indices = torch.topk(scores, k=self.routed_neighbors, dim=-1)
        weights = torch.softmax(values, dim=-1)

        projected = self.route_value(summaries)
        candidates = projected.unsqueeze(1).expand(batch, thoughtlets, thoughtlets, width)
        selected = torch.gather(
            candidates,
            2,
            indices.unsqueeze(-1).expand(batch, thoughtlets, self.routed_neighbors, width),
        )
        message = torch.sum(selected * weights.unsqueeze(-1), dim=2)
        return message, RoutingDiagnostics(indices, weights)

    @staticmethod
    def _repeat_per_thoughtlet(tokens: Tensor, thoughtlets: int) -> Tensor:
        batch, count, width = tokens.shape
        return (
            tokens.unsqueeze(1)
            .expand(batch, thoughtlets, count, width)
            .reshape(batch * thoughtlets, count, width)
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
    ) -> tuple[Tensor, Tensor, Tensor, RoutingDiagnostics]:
        batch, thoughtlets, registers, width = thoughts.shape

        belief_context = torch.cat(
            (sensors, action_time_tokens, belief, working_memory),
            dim=1,
        )
        proposed_belief = self.belief_attention(belief, belief_context)
        belief = self.belief_blend(belief, proposed_belief, elapsed_seconds)

        summaries = thoughts.mean(dim=2)
        routed_message, routing = self._route(summaries, allow_routing=allow_routing)
        thought_query = thoughts.reshape(batch * thoughtlets, registers, width)
        thought_context = torch.cat(
            (
                thought_query,
                self._repeat_per_thoughtlet(sensors, thoughtlets),
                self._repeat_per_thoughtlet(belief, thoughtlets),
                self._repeat_per_thoughtlet(goal_context, thoughtlets),
                retrieved_memory.reshape(
                    batch * thoughtlets,
                    retrieved_memory.shape[2],
                    width,
                ),
                routed_message.reshape(batch * thoughtlets, 1, width),
            ),
            dim=1,
        )
        proposed_thoughts = self.thought_attention(thought_query, thought_context).reshape_as(
            thoughts
        )
        thought_elapsed = elapsed_seconds.repeat_interleave(thoughtlets, dim=0)
        thoughts = self.thought_blend(
            thought_query,
            proposed_thoughts.reshape(batch * thoughtlets, registers, width),
            thought_elapsed,
        ).reshape_as(thoughts)

        if allow_workspace_writes:
            summaries = thoughts.mean(dim=2)
            utility = self.utility(summaries).squeeze(-1)
            write_count = min(4, thoughtlets)
            selected_utility, write_indices = torch.topk(
                utility,
                k=write_count,
                dim=-1,
            )
            selected_writes = torch.gather(
                summaries,
                1,
                write_indices.unsqueeze(-1).expand(batch, write_count, width),
            )
            # Selection remains hard and sparse, while the selected utility values
            # remain on the differentiable path into working memory.  Previously
            # utility affected only integer indices, leaving the scorer without a
            # gradient from the memory write it was supposed to control.
            write_weights = torch.softmax(selected_utility.float(), dim=-1).to(
                dtype=selected_writes.dtype
            )
            # Collapse the sparse candidates into one direction-changing mixture.
            # Scaling separate tokens would be almost erased by the context
            # LayerNorm inside memory_attention, leaving only epsilon-sized utility
            # gradients.
            writes = torch.sum(
                selected_writes * write_weights.unsqueeze(-1),
                dim=1,
                keepdim=True,
            )
            memory_context = torch.cat(
                (belief, writes, retrieved_memory.flatten(1, 2)),
                dim=1,
            )
        else:
            # The isolation baseline must not leak one slot into another through
            # the shared working-memory write path. Belief and external retrieval
            # remain available because they do not contain current-cycle slot writes.
            memory_context = torch.cat(
                (belief, retrieved_memory.flatten(1, 2)),
                dim=1,
            )
        proposed_memory = self.memory_attention(working_memory, memory_context)
        working_memory = self.memory_blend(
            working_memory,
            proposed_memory,
            elapsed_seconds,
        )
        return belief, working_memory, thoughts, routing


class BrainCell(nn.Module):
    """A stack of blocks; this one instance is tied across cognitive cycles."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        routed_neighbors: int,
        blocks: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            StructuredBrainBlock(
                width=width,
                heads=heads,
                routed_neighbors=routed_neighbors,
            )
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
                allow_routing=allow_routing,
                allow_workspace_writes=allow_workspace_writes,
            )
            routing.append(block_routing)
        if not routing:
            raise RuntimeError("BrainCell must contain at least one block")
        return belief, working_memory, thoughts, tuple(routing)


class MonolithicRecurrentBlock(nn.Module):
    """Conventional pooled recurrent latent used only as a control model.

    The block keeps the candidate's belief and working-memory interfaces, but
    replaces factorized thought slots and sparse routing with one GRU latent.
    Pooling is intentional: it is the monolithic bottleneck being compared.
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
        self.recurrent = nn.GRUCell(width, width)

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
                "monolithic recurrent state must have shape [batch, 1, 1, width]"
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
        recurrent_input = self.context_projection(pooled_context)
        proposed = self.recurrent(recurrent_input, previous)
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


class MonolithicRecurrentCell(nn.Module):
    """A tied stack of pooled GRU blocks with the BrainCell call contract."""

    def __init__(self, *, width: int, heads: int, blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            MonolithicRecurrentBlock(width=width, heads=heads)
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
            raise RuntimeError("MonolithicRecurrentCell must contain at least one block")
        return belief, working_memory, thoughts, tuple(routing)
