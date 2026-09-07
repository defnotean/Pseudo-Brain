"""Tied recurrent cell and sparse thoughtlet communication."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


from .sparse_thought_router import RoutingDiagnostics, SparseThoughtRouter


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
        self.router = SparseThoughtRouter(
            width=width,
            routed_neighbors=routed_neighbors,
            dense_routing=dense_routing,
        )
        self.utility = nn.Linear(width, 1)

    @property
    def route_query(self) -> nn.Linear:
        return self.router.route_query

    @property
    def route_key(self) -> nn.Linear:
        return self.router.route_key

    @property
    def route_value(self) -> nn.Linear:
        return self.router.route_value

    def _route(
        self,
        summaries: Tensor,
        *,
        allow_routing: bool,
    ) -> tuple[Tensor, RoutingDiagnostics]:
        return self.router(summaries, allow_routing=allow_routing)

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


class BrainCellOutput(tuple):
    """4-tuple subclass (belief, working_memory, thoughts, routing) carrying episodic plasticity attributes.

    Provides 100% backward compatibility with existing (belief, working_memory, thoughts, routing)
    unpackings while exposing .plastic_weights and .P_t for Consequence-Gated Plasticity.
    """

    plastic_weights: Tensor | None
    P_t: Tensor | None

    def __new__(
        cls,
        belief: Tensor,
        working_memory: Tensor,
        thoughts: Tensor,
        routing: tuple[RoutingDiagnostics, ...],
        plastic_weights: Tensor | None = None,
    ) -> BrainCellOutput:
        inst = super().__new__(cls, (belief, working_memory, thoughts, routing))
        inst.plastic_weights = plastic_weights
        inst.P_t = plastic_weights
        return inst


class SurpriseEncoder(nn.Module):
    """Embeds scalar prediction error distance into a compact surprise representation."""

    def __init__(self, out_dim: int = 16) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )
        with torch.no_grad():
            nn.init.uniform_(self.fc[0].weight, 0.2, 0.6)
            nn.init.zeros_(self.fc[0].bias)
            nn.init.uniform_(self.fc[2].weight, 0.2, 0.6)
            nn.init.zeros_(self.fc[2].bias)

    def forward(self, error: Tensor) -> Tensor:
        if error.dim() == 1:
            error = error.unsqueeze(-1)
        return self.fc(error)


class FastPlasticityModule(nn.Module):
    """Fast episodic state adaptation (P_t) for Irene BrainCell.

    Updates online during gameplay without backprop:
        P_{t+1} = gamma * P_t + eta * gate * tanh(W [state, surprise])
    """

    def __init__(
        self,
        *,
        state_dim: int,
        plastic_dim: int,
        surprise_dim: int = 16,
        decay: float = 0.999,
        lr: float = 0.25,
    ) -> None:
        super().__init__()
        self.decay = decay
        self.lr = lr
        self.state_dim = state_dim
        self.plastic_dim = plastic_dim
        self.modulator = nn.Linear(state_dim + surprise_dim, plastic_dim)
        self.surprise_gate = nn.Sequential(
            nn.Linear(surprise_dim, 1),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            nn.init.uniform_(self.surprise_gate[0].weight, 0.2, 0.5)
            nn.init.constant_(self.surprise_gate[0].bias, -2.0)
        self.scale = nn.Parameter(torch.tensor(1.5))

    def init_trace(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.zeros(batch_size, self.plastic_dim, device=device, dtype=dtype)

    def update(
        self,
        P_t: Tensor,
        state: Tensor,
        surprise: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mag = torch.tanh(torch.norm(surprise, dim=-1, keepdim=True))
        gate = self.surprise_gate(surprise) * mag
        delta = gate * torch.tanh(self.modulator(torch.cat([state, surprise], dim=-1)))
        P_next = self.decay * P_t + self.lr * delta
        return P_next, delta, gate


class FactorizedLowRankProjection(nn.Module):
    """Factorized low-rank linear projection (W -> r -> proj_dim or in_dim -> r -> out_dim).

    Factorizes a high-dimensional linear projection into two low-rank stages:
        down: in_features -> rank (no bias)
        up:   rank -> out_features (with bias)

    Reduces parameter complexity from O(in_dim * out_dim) to O(rank * (in_dim + out_dim)),
    preventing latency explosion during core parameter scaling (proj_dim >= 512, 1024, 2816).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {rank}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
        if self.up.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.up.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.up.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}"


class BrainCellCore(nn.Module):
    """Recurrent BrainCell core with optional factorized low-rank projections and conditional recurrence."""

    def __init__(
        self,
        input_size: int = 512,
        thought_size: int = 48,
        rank: int | None = None,
        proj_dim: int | None = None,
        tier: str | None = None,
        num_deep_layers: int = 3,
    ) -> None:
        super().__init__()
        self.tier = tier

        # Tier-specific architectural scaling laws
        if tier == "tier2":
            # Law 1 (Slot Width Clamping): W=64, K=128 slots (32 KB state memory)
            thought_size = 64
            # Law 2 (Factorized Deep Projections): W=64 -> rank r=32 -> proj_dim=4096 (~50M capacity)
            rank = 32
            proj_dim = 4096
            input_size = 4096 if input_size in (48, 256, 512) else input_size

        self.input_size = input_size
        self.thought_size = thought_size
        self.rank = rank
        self.proj_dim = proj_dim
        self.num_deep_layers = num_deep_layers

        # Law 2: Deep projection parameter core delivering Tier 2 ~50M capacity
        if tier == "tier2" and proj_dim is not None and proj_dim > 0:
            deep_layers: list[nn.Module] = []
            for _ in range(num_deep_layers):
                deep_layers.append(nn.Linear(proj_dim, proj_dim))
                deep_layers.append(nn.GELU())
            self.deep_proj: nn.Module | None = nn.Sequential(*deep_layers)
        else:
            self.deep_proj = None

        # Input projections (proj_dim -> thought_size): dense or factorized low-rank
        if rank is not None and rank > 0:
            self.W_ir: nn.Module = FactorizedLowRankProjection(input_size, thought_size, rank=rank)
            self.W_iz: nn.Module = FactorizedLowRankProjection(input_size, thought_size, rank=rank)
            self.W_in: nn.Module = FactorizedLowRankProjection(input_size, thought_size, rank=rank)
        else:
            self.W_ir = nn.Linear(input_size, thought_size)
            self.W_iz = nn.Linear(input_size, thought_size)
            self.W_in = nn.Linear(input_size, thought_size)

        # Recurrent hidden transitions (thought_size -> thought_size)
        self.W_hr = nn.Linear(thought_size, thought_size)
        self.W_hz = nn.Linear(thought_size, thought_size)
        self.W_hn = nn.Linear(thought_size, thought_size)

        # Telemetry & FLOP tracking
        self.eval_count: int = 0
        self.total_flops: int = 0

    def state_bytes(self, K: int = 128, bytes_per_element: int = 4) -> int:
        """Calculate recurrent state memory footprint in bytes.

        Law 1 Contract: W=64, K=128 slots -> 128 * 64 * 4 = 32,768 bytes (32 KB).
        """
        return K * self.thought_size * bytes_per_element

    def count_parameters(self) -> dict[str, int]:
        """Count total and trainable parameters of BrainCellCore."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def flops_per_slot(self) -> int:
        """Calculate the theoretical FLOPs evaluated per active slot."""
        flops_deep = 0
        if self.deep_proj is not None and self.proj_dim is not None:
            flops_deep = self.num_deep_layers * 2 * (self.proj_dim * self.proj_dim)

        if self.rank is not None and self.rank > 0:
            flops_input = 3 * 2 * (self.input_size * self.rank + self.rank * self.thought_size)
        else:
            flops_input = 3 * 2 * (self.input_size * self.thought_size)
        flops_hidden = 3 * 2 * (self.thought_size * self.thought_size)
        flops_pointwise = (
            3 * self.thought_size  # additions for pre-activations
            + 8 * self.thought_size  # 2 sigmoids (~4 flops each)
            + 4 * self.thought_size  # tanh (~4 flops)
            + 4 * self.thought_size  # blend: (1-z)*n + z*thought
        )
        return flops_deep + flops_input + flops_hidden + flops_pointwise

    def reset_telemetry(self) -> None:
        """Reset evaluation slot count and FLOP tracking telemetry."""
        self.eval_count = 0
        self.total_flops = 0

    def forward(self, thought: Tensor, x: Tensor) -> Tensor:
        """Dense recurrent evaluation for arbitrary batch shape of thought and x.

        Args:
            thought: [..., thought_size]
            x: [..., input_size]
        """
        batch_slots = thought.numel() // self.thought_size
        self.eval_count += batch_slots
        self.total_flops += batch_slots * self.flops_per_slot()

        x_eff = self.deep_proj(x) if self.deep_proj is not None else x
        r = torch.sigmoid(self.W_ir(x_eff) + self.W_hr(thought))
        z = torch.sigmoid(self.W_iz(x_eff) + self.W_hz(thought))
        n = torch.tanh(self.W_in(x_eff) + r * self.W_hn(thought))
        return (1.0 - z) * n + z * thought

    def forward_conditional(
        self,
        thoughts: Tensor,
        x: Tensor,
        salience: Tensor,
        epsilon_dormant: float = 0.05,
    ) -> tuple[Tensor, Tensor]:
        """Event-driven sparse slot ticking (conditional recurrence).

        Bypasses BrainCellCore evaluation when Cognitive Input Gate salience s_k < epsilon_dormant.
        Dormant slots remain strictly unchanged with ZERO FLOPs evaluated.
        Only active slots evaluate the heavy BrainCellCore matrix multiplications.

        Args:
            thoughts: [B, K, thought_size] or [N, thought_size] slot thought states
            x: [B, K, input_size], [B, input_size], or [N, input_size] input features
            salience: [B, K, 1], [B, K], or [N, 1], [N] CIG salience scores s_k
            epsilon_dormant: dormancy threshold below which slots are bypassed

        Returns:
            updated_thoughts: updated thoughts tensor with dormant slots unchanged
            active_mask: boolean tensor indicating which slots ticked
        """
        if thoughts.dim() == 2:
            N, W = thoughts.shape
            s_val = salience.squeeze(-1) if salience.dim() == 2 else salience
            active_mask = (s_val >= epsilon_dormant)
            num_active = int(active_mask.sum().item())

            if num_active == 0:
                return thoughts, active_mask

            if num_active == N:
                new_t = self.forward(thoughts, x)
                s_blend = salience if salience.dim() == 2 else salience.unsqueeze(-1)
                updated = (1.0 - s_blend) * thoughts + s_blend * new_t
                return updated, active_mask

            idx = torch.nonzero(active_mask, as_tuple=True)[0]
            t_active = thoughts[idx]
            x_active = x[idx]
            new_t_active = self.forward(t_active, x_active)
            s_blend = salience if salience.dim() == 2 else salience.unsqueeze(-1)
            s_active = s_blend[idx]
            updated_active = (1.0 - s_active) * t_active + s_active * new_t_active

            next_thoughts = thoughts.clone()
            next_thoughts[idx] = updated_active
            return next_thoughts, active_mask

        # 3D tensor: [B, K, W]
        B, K, W = thoughts.shape
        s_val = salience.squeeze(-1) if salience.dim() == 3 else salience
        active_mask = (s_val >= epsilon_dormant)
        num_active = int(active_mask.sum().item())

        if num_active == 0:
            return thoughts, active_mask

        x_exp = x.unsqueeze(1).expand(-1, K, -1) if x.dim() == 2 else x

        if num_active == B * K:
            t_flat = thoughts.reshape(B * K, W)
            x_flat = x_exp.reshape(B * K, -1)
            new_t = self.forward(t_flat, x_flat).reshape(B, K, W)
            s_blend = salience if salience.dim() == 3 else salience.unsqueeze(-1)
            updated = (1.0 - s_blend) * thoughts + s_blend * new_t
            return updated, active_mask

        b_idx, k_idx = torch.nonzero(active_mask, as_tuple=True)
        t_active = thoughts[b_idx, k_idx]
        x_active = x_exp[b_idx, k_idx]
        new_t_active = self.forward(t_active, x_active)

        s_blend = salience if salience.dim() == 3 else salience.unsqueeze(-1)
        s_active = s_blend[b_idx, k_idx]
        updated_active = (1.0 - s_active) * t_active + s_active * new_t_active

        next_thoughts = thoughts.clone()
        next_thoughts[b_idx, k_idx] = updated_active
        return next_thoughts, active_mask


class BrainCell(nn.Module):
    """A stack of blocks; this one instance is tied across cognitive cycles."""

    def __init__(
        self,
        *,
        width: int = 64,
        heads: int = 8,
        routed_neighbors: int = 4,
        blocks: int = 2,
        dense_routing: bool = False,
        use_cgp: bool = False,
        plastic_decay: float = 0.999,
        plastic_lr: float = 0.25,
        plastic_dim: int | None = None,
        tier: str | None = None,
    ) -> None:
        super().__init__()
        self.tier = tier
        if tier == "tier2":
            width = 64
            heads = 8
        self.width = width
        self.use_cgp = bool(use_cgp)
        self.blocks = nn.ModuleList(
            StructuredBrainBlock(
                width=width,
                heads=heads,
                routed_neighbors=routed_neighbors,
                dense_routing=dense_routing,
            )
            for _ in range(blocks)
        )
        if self.use_cgp:
            p_dim = plastic_dim if plastic_dim is not None else width
            self.plastic_dim = p_dim
            self.surprise_encoder = SurpriseEncoder(out_dim=16)
            self.plasticity = FastPlasticityModule(
                state_dim=width,
                plastic_dim=p_dim,
                surprise_dim=16,
                decay=plastic_decay,
                lr=plastic_lr,
            )
            self.cognitive_gate = nn.Sequential(
                nn.Linear(width * 2 + 16, width),
                nn.Sigmoid(),
            )
            with torch.no_grad():
                if hasattr(self.cognitive_gate[0], "bias") and self.cognitive_gate[0].bias is not None:
                    self.cognitive_gate[0].bias.fill_(1.0)
            self.plastic_proj = nn.Linear(p_dim, width)
            with torch.no_grad():
                nn.init.zeros_(self.plastic_proj.weight)
                nn.init.zeros_(self.plastic_proj.bias)
        else:
            self.plastic_dim = None
            self.surprise_encoder = None
            self.plasticity = None
            self.cognitive_gate = None
            self.plastic_proj = None

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
        plastic_weights: Tensor | None = None,
        surprise: Tensor | None = None,
    ) -> BrainCellOutput:
        routing: list[RoutingDiagnostics] = []
        cur_belief = belief
        cur_memory = working_memory
        cur_thoughts = thoughts

        for block in self.blocks:
            cur_belief, cur_memory, cur_thoughts, block_routing = block(
                belief=cur_belief,
                working_memory=cur_memory,
                thoughts=cur_thoughts,
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

        next_plastic_weights = plastic_weights
        if (
            self.use_cgp
            and self.plasticity is not None
            and self.cognitive_gate is not None
            and self.plastic_proj is not None
            and self.surprise_encoder is not None
        ):
            B = thoughts.shape[0]
            dev = thoughts.device
            dtype = thoughts.dtype
            if plastic_weights is None:
                plastic_weights = self.plasticity.init_trace(B, dev, dtype)
            if surprise is None:
                surprise = thoughts.new_zeros(B, 1)
            surprise_emb = self.surprise_encoder(surprise)

            # Cognitive input gating: prevent blank/corridor diffusion
            thought_summary = cur_thoughts.mean(dim=(1, 2))  # [B, width]
            sensor_summary = sensors.mean(dim=1)  # [B, width]
            gate_in = torch.cat([thought_summary, sensor_summary, surprise_emb], dim=-1)
            raw_salience = self.cognitive_gate(gate_in).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, width]
            surprise_scale = torch.tanh(torch.norm(surprise, dim=-1, keepdim=True)).unsqueeze(1).unsqueeze(2)
            salience = raw_salience * surprise_scale
            cur_thoughts = (1.0 - salience) * thoughts + salience * cur_thoughts

            # Fast episodic plasticity update
            state_feat = cur_thoughts.mean(dim=(1, 2))
            next_plastic_weights, delta, _ = self.plasticity.update(plastic_weights, state_feat, surprise_emb)

            # Integrate episodic plasticity delta into working memory and thoughts upon consequence surprise
            plastic_injection = self.plastic_proj(delta)
            cur_thoughts = cur_thoughts + 0.1 * self.plasticity.scale * plastic_injection.unsqueeze(1).unsqueeze(2)
            cur_memory = cur_memory + 0.1 * self.plasticity.scale * plastic_injection.unsqueeze(1)

        return BrainCellOutput(
            cur_belief,
            cur_memory,
            cur_thoughts,
            tuple(routing),
            next_plastic_weights,
        )


class PlasticBrainCell(BrainCell):
    """Dedicated Consequence-Gated Plasticity BrainCell with use_cgp=True."""

    def __init__(
        self,
        *,
        width: int,
        heads: int,
        routed_neighbors: int,
        blocks: int,
        dense_routing: bool = False,
        plastic_decay: float = 0.999,
        plastic_lr: float = 0.25,
        plastic_dim: int | None = None,
    ) -> None:
        super().__init__(
            width=width,
            heads=heads,
            routed_neighbors=routed_neighbors,
            blocks=blocks,
            dense_routing=dense_routing,
            use_cgp=True,
            plastic_decay=plastic_decay,
            plastic_lr=plastic_lr,
            plastic_dim=plastic_dim,
        )


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
        **kwargs: object,
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
        **kwargs: object,
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
        **kwargs: object,
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


__all__ = [
    "BrainCell",
    "BrainCellCore",
    "BrainCellOutput",
    "ContinuousTimeBlend",
    "EnsembleBrainCell",
    "EnsembleMemberBlock",
    "FactorizedLowRankProjection",
    "FastPlasticityModule",
    "MonolithicRecurrentBlock",
    "MonolithicRecurrentCell",
    "PlasticBrainCell",
    "ResidualCrossAttention",
    "RoutingDiagnostics",
    "StructuredBrainBlock",
    "SurpriseEncoder",
    "TransformerCarryCell",
]
