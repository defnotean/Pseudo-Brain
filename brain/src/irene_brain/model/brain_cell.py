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

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        routed_neighbors: int,
        dense_routing: bool = False,
    ) -> None:
        super().__init__()
        self.width = width
        self.routed_neighbors = routed_neighbors
        self.dense_routing = bool(dense_routing)
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
        projected = self.route_value(summaries)
        if self.dense_routing:
            # Unrestricted all-to-all communication: every other slot is a
            # routing target with its softmax weight, no sparse top-k.
            base = torch.arange(thoughtlets, device=summaries.device)
            off_diagonal = base.unsqueeze(0).expand(thoughtlets, thoughtlets)
            off_diagonal = off_diagonal[~diagonal].reshape(thoughtlets, thoughtlets - 1)
            indices = off_diagonal.unsqueeze(0).expand(batch, thoughtlets, thoughtlets - 1)
            weights = torch.gather(torch.softmax(scores, dim=-1), 2, indices)
            selected = torch.gather(
                projected.unsqueeze(1).expand(batch, thoughtlets, thoughtlets, width),
                2,
                indices.unsqueeze(-1).expand(batch, thoughtlets, thoughtlets - 1, width),
            )
            message = torch.sum(selected * weights.unsqueeze(-1), dim=2)
            return message, RoutingDiagnostics(indices, weights)

        values, indices = torch.topk(scores, k=self.routed_neighbors, dim=-1)
        weights = torch.softmax(values, dim=-1)

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
        dense_routing: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            StructuredBrainBlock(
                width=width,
                heads=heads,
                routed_neighbors=routed_neighbors,
                dense_routing=dense_routing,
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


class EnsembleMemberBlock(nn.Module):
    """One lean slot-update block owned by a single ensemble member.

    Used only by the matched-cost independent-ensemble control. The block
    carries no routing, belief-maintenance, or workspace-write parameters:
    members never communicate, and belief/working memory are maintained by
    the model's shared ingest pathway instead of per-block attention. Its
    thought context mirrors the reference block minus the routed-message
    token, so a member slot sees its own registers, sensors, belief, goal
    context, and retrieved memory — nothing from any other member.
    """

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        self.width = width
        self.thought_attention = ResidualCrossAttention(width=width, heads=heads)
        self.thought_blend = ContinuousTimeBlend(width)

    def forward(
        self,
        *,
        belief: Tensor,
        thoughts: Tensor,
        sensors: Tensor,
        goal_context: Tensor,
        retrieved_memory: Tensor,
        elapsed_seconds: Tensor,
    ) -> Tensor:
        batch, thoughtlets, registers, width = thoughts.shape
        if width != self.width:
            raise ValueError("ensemble member thought width mismatch")
        thought_query = thoughts.reshape(batch * thoughtlets, registers, width)
        thought_context = torch.cat(
            (
                thought_query,
                StructuredBrainBlock._repeat_per_thoughtlet(sensors, thoughtlets),
                StructuredBrainBlock._repeat_per_thoughtlet(belief, thoughtlets),
                StructuredBrainBlock._repeat_per_thoughtlet(goal_context, thoughtlets),
                retrieved_memory.reshape(
                    batch * thoughtlets,
                    retrieved_memory.shape[2],
                    width,
                ),
            ),
            dim=1,
        )
        proposed_thoughts = self.thought_attention(thought_query, thought_context)
        thought_elapsed = elapsed_seconds.repeat_interleave(thoughtlets, dim=0)
        return self.thought_blend(
            thought_query,
            proposed_thoughts,
            thought_elapsed,
        ).reshape_as(thoughts)


class EnsembleBrainCell(nn.Module):
    """Untied independent member stacks over disjoint thought-slot chunks.

    Each of the ``members`` stacks owns ``thoughtlets // members`` slots and
    its own untied block parameters; slots in different members never share
    weights or messages. Weights remain tied across cognitive cycles within
    a member, exactly like the reference ties its cell across cycles. Belief
    and working memory pass through unchanged. One empty routing diagnostic
    per block layer preserves the ``BrainCell`` result arity.
    """

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        blocks: int,
        members: int,
        thoughtlets: int,
    ) -> None:
        super().__init__()
        if isinstance(members, bool) or not isinstance(members, int) or members < 2:
            raise ValueError("members must be an integer of at least 2")
        if thoughtlets % members != 0:
            raise ValueError("thoughtlets must divide evenly across members")
        if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
            raise ValueError("blocks must be a positive integer")
        self.members = members
        self.member_stacks = nn.ModuleList(
            nn.ModuleList(
                EnsembleMemberBlock(width=width, heads=heads) for _ in range(blocks)
            )
            for _ in range(members)
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
        del action_time_tokens, allow_routing, allow_workspace_writes
        if thoughts.shape[1] % self.members != 0:
            raise ValueError("thought slots must divide evenly across members")
        thought_chunks = thoughts.chunk(self.members, dim=1)
        memory_chunks = retrieved_memory.chunk(self.members, dim=1)
        updated: list[Tensor] = []
        for member_stack, thought_chunk, memory_chunk in zip(
            self.member_stacks, thought_chunks, memory_chunks
        ):
            current = thought_chunk
            for block in member_stack:
                current = block(
                    belief=belief,
                    thoughts=current,
                    sensors=sensors,
                    goal_context=goal_context,
                    retrieved_memory=memory_chunk,
                    elapsed_seconds=elapsed_seconds,
                )
            updated.append(current)
        thoughts = torch.cat(updated, dim=1)
        batch, thoughtlets = thoughts.shape[0], thoughts.shape[1]
        empty_indices = torch.empty(
            batch,
            thoughtlets,
            0,
            device=thoughts.device,
            dtype=torch.long,
        )
        empty_weights = thoughts.new_empty((batch, thoughtlets, 0))
        blocks = len(self.member_stacks[0])
        return (
            belief,
            working_memory,
            thoughts,
            tuple(
                RoutingDiagnostics(empty_indices, empty_weights) for _ in range(blocks)
            ),
        )


class TransformerCarryCell(nn.Module):
    """A standard Transformer encoder with one recurrent carry token.

    Used only by the recurrent-transformer control (PLAN section 28 item
    5). One token set — belief, working memory, the carry, sensors,
    action/time, goal context, and a pooled retrieved-memory token — passes
    through untied-per-layer encoder blocks; belief, working memory, and
    the carry are re-read from their own output positions through
    continuous-time blends. The carry is the only slot-shaped state, so the
    config must hold exactly one thoughtlet with one register. One empty
    routing diagnostic per layer preserves the ``BrainCell`` result arity.
    """

    def __init__(self, *, width: int, heads: int, blocks: int) -> None:
        super().__init__()
        if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1:
            raise ValueError("blocks must be a positive integer")
        self.width = width
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=width,
                nhead=heads,
                dim_feedforward=width * 4,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(blocks)
        )
        self.belief_blend = ContinuousTimeBlend(width)
        self.memory_blend = ContinuousTimeBlend(width)
        self.carry_blend = ContinuousTimeBlend(width)

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
        batch, thoughtlets, registers, width = thoughts.shape
        if thoughtlets != 1 or registers != 1 or width != self.width:
            raise ValueError(
                "transformer carry state must have shape [batch, 1, 1, width]"
            )
        retrieved_token = retrieved_memory.mean(dim=(1, 2)).unsqueeze(1)
        tokens = torch.cat(
            (
                belief,
                working_memory,
                thoughts.reshape(batch, 1, width),
                sensors,
                action_time_tokens,
                goal_context,
                retrieved_token,
            ),
            dim=1,
        )
        for layer in self.layers:
            tokens = layer(tokens)
        belief_count = belief.shape[1]
        memory_count = working_memory.shape[1]
        belief = self.belief_blend(
            belief, tokens[:, :belief_count], elapsed_seconds
        )
        working_memory = self.memory_blend(
            working_memory,
            tokens[:, belief_count : belief_count + memory_count],
            elapsed_seconds,
        )
        carry_token = tokens[:, belief_count + memory_count]
        thoughts = self.carry_blend(
            thoughts.reshape(batch, 1, width),
            carry_token.unsqueeze(1),
            elapsed_seconds,
        ).reshape(batch, 1, 1, width)
        empty_indices = torch.empty(
            batch, 1, 0, device=thoughts.device, dtype=torch.long
        )
        empty_weights = thoughts.new_empty((batch, 1, 0))
        return (
            belief,
            working_memory,
            thoughts,
            tuple(
                RoutingDiagnostics(empty_indices, empty_weights)
                for _ in range(len(self.layers))
            ),
        )


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
