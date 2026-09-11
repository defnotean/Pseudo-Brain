"""Unified Multi-Modal, Tool-Calling, Game-Playing & Coding Pseudo-Brain Architecture.

Unifies:
1. Sensory Perception: ConvEncoder for visual game frames + 32k/64k BPE for language/code.
2. Dual-Mode Recurrence:
   - Training: O(log T) Parallel Associative Prefix Scan across sequences.
   - Streaming: O(1) single-token step (<1 ms latency, 60 Hz real-time compliant).
3. Memory Architecture: Two-Tier Hierarchical State (4.0 KB Fast Working Cache + Consolidated Episodic Bank).
4. Deep Stability: Recurrent RMSNorm [0.9, 1.1] + Cayley skew-symmetric spectral bounds rho(W) <= 1.0.
5. Decoupled Readout Heads:
   - Language & Code Head (text, python syntax, tool calls)
   - Action Policy Head (discrete arcade/POMDP actions: 0=Idle, 1=W, 2=A, 3=S, 4=D)
   - Consequence Value Head (reward/outcome error for CGSL online adaptation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from irene_brain.model.brain_cell import FactorizedLowRankProjection
from irene_brain.model.sparse_thought_router import SparseThoughtRouter
from irene_brain.model.torch_model import deterministic_thought_identity_codes
from irene_brain.parallel.parallel_scan import parallel_scan
from irene_brain.parallel.triton_scan import triton_scan
from irene_brain.memory.hierarchical_state import (
    HierarchicalCognitiveState,
    HierarchicalMemoryConfig,
    FastWorkingMemoryTier,
    ConsolidatedEpisodicTier,
    CrossTierConsolidationGate,
    CrossTierRetrievalModule,
)
from irene_brain.stability.recurrent_norm import RecurrentRMSNorm
from irene_brain.stability.spectral_norm import CayleyLinear
from irene_brain.stability.deep_recurrent_stack import DeepRecurrentStack
from irene_brain.semantic.multimodal_model import ConvEncoder


@dataclass
class UnifiedCognitiveState:
    """Persistent cognitive state across multi-modal game, dialog, and coding turns."""
    hierarchical_state: HierarchicalCognitiveState
    active_thread: Tensor          # [B] integer thread index
    P_t: Optional[Tensor] = None   # [B, K, V] synaptic plasticity matrix if enabled
    last_action: Optional[Tensor] = None # [B] last executed action index
    ptr_prev_alpha: Optional[Tensor] = None # [B, N] previous pointer attention distribution
    ptr_prev_gamma: Optional[Tensor] = None # [B, 1] previous pointer copy gate


class DeepHighwayResidual(nn.Module):
    """Pre-Norm Residual Highway stack delivering stable 1B parameter capacity.

    Guarantees robust gradient flow across 24-32 layers by combining
    RMSNorm with identity skip connections: x_{l+1} = x_l + GELU(Linear(RMSNorm(x_l))).
    """

    def __init__(self, dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                RecurrentRMSNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
            ) for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class UnifiedPseudoBrain(nn.Module):
    """Production Unified Pseudo-Brain Model for Games, Conversation, Tools, and Coding."""

    def __init__(
        self,
        vocab_size: int = 32000,
        tier: str = "tier2",
        K_fast: Optional[int] = None,
        W_fast: Optional[int] = None,
        K_episodic: Optional[int] = None,
        W_episodic: Optional[int] = None,
        embed_dim: Optional[int] = None,
        proj_dim: Optional[int] = None,
        rank: Optional[int] = None,
        num_deep_layers: Optional[int] = None,
        n_actions: int = 5,
        in_channels: int = 3,
        visual_dim: int = 192,
        num_visual_tokens: int = 4,
        use_cgp: bool = True,
        use_routing: bool = True,
        spectral_mode: str = "cayley",
        use_funnel: Optional[bool] = None,
        use_readout_norm: bool = False,
        use_token_skip: bool = False,
        use_gated_token_skip: bool = False,
        use_pointer_copy: bool = False,
        compensated_state: bool = False,
        pointer_mode: str = "sequential",
        retention_profile: str = "legacy",
    ):
        super().__init__()
        if compensated_state and use_routing:
            raise ValueError("compensated_state=True cannot be combined with use_routing=True")
        self.vocab_size = vocab_size
        self.tier = tier

        if use_funnel is None:
            self.use_funnel = tier in ("tier2_35m", "35m", "tier3", "tier3_1b", "1b", "tier5_1b")
        else:
            self.use_funnel = bool(use_funnel)

        # 1. Establish tier presets as defaults
        if tier in ("tier2_35m", "35m"):
            tier_defaults = {
                "K_fast": 16,
                "W_fast": 64,
                "K_episodic": 64,
                "W_episodic": 128,
                "embed_dim": 384,
                "proj_dim": 4096,
                "rank": 32,
                "num_deep_layers": 4,
            }
        elif tier in ("tier3_1b", "1b", "tier5_1b"):
            tier_defaults = {
                "K_fast": 16,
                "W_fast": 64,
                "K_episodic": 128,
                "W_episodic": 128,
                "embed_dim": 1536,
                "proj_dim": 6144,
                "rank": 64,
                "num_deep_layers": 24,
            }
        elif tier in ("tier0", "micro", "131k"):
            tier_defaults = {
                "K_fast": 8,
                "W_fast": 32,
                "K_episodic": 16,
                "W_episodic": 64,
                "embed_dim": 64,
                "proj_dim": 128,
                "rank": 16,
                "num_deep_layers": 2,
            }
        elif tier in ("tier1", "embedded", "10m"):
            tier_defaults = {
                "K_fast": 16,
                "W_fast": 64,
                "K_episodic": 32,
                "W_episodic": 64,
                "embed_dim": 128,
                "proj_dim": 512,
                "rank": 32,
                "num_deep_layers": 3,
            }
        elif tier in ("tier3", "agentic", "100m", "300m"):
            tier_defaults = {
                "K_fast": 16,
                "W_fast": 64,
                "K_episodic": 128,
                "W_episodic": 128,
                "embed_dim": 512,
                "proj_dim": 2048,
                "rank": 32,
                "num_deep_layers": 12,
            }
        else: # tier2 / default
            tier_defaults = {
                "K_fast": 16,
                "W_fast": 64,
                "K_episodic": 64,
                "W_episodic": 128,
                "embed_dim": 256,
                "proj_dim": 1024,
                "rank": 32,
                "num_deep_layers": 4,
            }

        # 2. Caller-supplied explicit values override tier presets cleanly
        self.K_fast = K_fast if K_fast is not None else tier_defaults["K_fast"]
        self.W_fast = W_fast if W_fast is not None else tier_defaults["W_fast"]
        self.K_episodic = K_episodic if K_episodic is not None else tier_defaults["K_episodic"]
        self.W_episodic = W_episodic if W_episodic is not None else tier_defaults["W_episodic"]
        self.embed_dim = embed_dim if embed_dim is not None else tier_defaults["embed_dim"]
        self.proj_dim = proj_dim if proj_dim is not None else tier_defaults["proj_dim"]
        self.rank = rank if rank is not None else tier_defaults["rank"]
        self.num_deep_layers = num_deep_layers if num_deep_layers is not None else tier_defaults["num_deep_layers"]
        self.n_actions = n_actions

        K_fast = self.K_fast
        W_fast = self.W_fast
        K_episodic = self.K_episodic
        W_episodic = self.W_episodic
        embed_dim = self.embed_dim
        proj_dim = self.proj_dim
        rank = self.rank
        num_deep_layers = self.num_deep_layers
        self.visual_dim = visual_dim
        self.use_cgp = use_cgp
        self.use_routing = use_routing

        # 1. Sensory Ingestion
        # Language / Code Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lang_proj = nn.Linear(embed_dim, proj_dim)

        # Visual Game Frame ConvEncoder
        self.visual_encoder = ConvEncoder(
            in_channels=in_channels,
            feature_dim=visual_dim,
            num_entity_tokens=num_visual_tokens,
        )
        self.visual_proj = nn.Linear(visual_dim, proj_dim)

        # Action Feedback Embedding (past action conditioning for closed-loop games)
        self.action_embed = nn.Embedding(n_actions + 1, embed_dim)
        self.action_proj = nn.Linear(embed_dim, proj_dim)

        # 2. Hierarchical Memory Configuration (Law 1 Compliant)
        self.memory_cfg = HierarchicalMemoryConfig(
            tier=tier,
            K_fast=K_fast,
            W_fast=W_fast,
            K_episodic=K_episodic,
            W_episodic=W_episodic,
            consolidation_rate=0.1,
            retrieval_heads=4,
        )
        self.consolidation_gate = CrossTierConsolidationGate(
            W_fast=W_fast,
            W_episodic=W_episodic,
            threshold=0.2,
        )
        self.retrieval_module = CrossTierRetrievalModule(
            W_fast=W_fast,
            W_episodic=W_episodic,
            num_heads=4,
            top_k=4,
        )

        # 3. Deep Recurrent Core with Cayley Spectral Radius <= 1.0 & RMSNorm
        self.recurrent_stack = DeepRecurrentStack(
            num_layers=num_deep_layers,
            thought_size=W_fast,
            input_size=proj_dim,
            rank=rank,
            K=K_fast,
            spectral_mode=spectral_mode,
            max_spectral_radius=1.0,
            use_highway=True,
            slot_identity_weight=0.05,
        )

        # Law 2: Deep projection parameter core delivering 1B capacity via Pre-Norm Residuals
        if tier in ("tier3_1b", "1b", "tier3", "tier5_1b") and proj_dim is not None and proj_dim >= 4096:
            self.deep_proj: Optional[nn.Module] = DeepHighwayResidual(proj_dim, num_deep_layers)
        else:
            self.deep_proj = None

        # Progressive Bottleneck Funnel (proj_dim -> intermediate_dim -> W_fast)
        # Prevents state entanglement when proj_dim is scaled (e.g. 4096 or 6144)
        if self.use_funnel and proj_dim > 512:
            self.intermediate_dim = 512
            self.funnel_down = nn.Sequential(
                RecurrentRMSNorm(proj_dim),
                nn.Linear(proj_dim, self.intermediate_dim),
                nn.SiLU(),
                RecurrentRMSNorm(self.intermediate_dim),
            )
            funnel_in = self.intermediate_dim
        else:
            self.intermediate_dim = proj_dim
            self.funnel_down = None
            funnel_in = proj_dim

        # Parallel Associative Scan Input Gates for O(log T) parallel training
        effective_proj_rank = max(rank, W_fast) if self.use_funnel else rank
        self.W_iz_parallel = FactorizedLowRankProjection(funnel_in, W_fast, rank=effective_proj_rank)
        self.W_in_parallel = FactorizedLowRankProjection(funnel_in, W_fast, rank=effective_proj_rank)

        # 4. Cognitive Input Gating (CIG)
        self.cig_gate = nn.Sequential(
            nn.Linear(proj_dim + W_fast, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.cig_gate[0].bias, 1.0)

        # 5. Sparse Thought Router for cross-slot communication
        if self.use_routing and self.K_fast > 1:
            self.router = SparseThoughtRouter(
                width=W_fast,
                routed_neighbors=min(2, max(1, K_fast - 1)),
                dense_routing=False,
            )
        else:
            self.router = None

        # 6. Decoupled Readout Heads
        head_hidden = max(64, min(proj_dim // 2, 512))

        # (a) Language & Python Code Generation Head
        self.use_readout_norm = use_readout_norm
        self.readout_norm = RecurrentRMSNorm(W_fast) if use_readout_norm else nn.Identity()
        self.slot_head = nn.Sequential(
            nn.Linear(W_fast, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, vocab_size),
        )

        # (b) Local Token Skip Connection (Contextually Gated)
        # logits = slot_head(RMSNorm(h_t)) + beta_t * W_skip · embedding(previous_token)
        # beta_t = sigmoid(W_beta · [contextualized_slot; embedding(previous_token)])
        self.use_token_skip = use_token_skip
        self.use_gated_token_skip = use_gated_token_skip
        if use_token_skip:
            self.token_skip = nn.Linear(embed_dim, vocab_size, bias=False)
            nn.init.zeros_(self.token_skip.weight)
            if use_gated_token_skip:
                self.skip_gate = nn.Sequential(
                    nn.Linear(W_fast + embed_dim, 64),
                    nn.GELU(),
                    nn.Linear(64, 1),
                )
                nn.init.constant_(self.skip_gate[-1].bias, 2.0)
            else:
                self.skip_gate = None
        else:
            self.token_skip = None
            self.skip_gate = None

        # (c) In-Context Pointer-Copy Head
        # Allows recurrent state to attend back to observed prompt spans for verbatim symbolic copying
        if pointer_mode not in ("sequential", "parallel_predecessor"):
            raise ValueError(f"Unknown pointer mode: {pointer_mode}")
        self.pointer_mode = pointer_mode
        if retention_profile not in ("legacy", "multiscale"):
            raise ValueError(f"Unknown retention profile: {retention_profile}")
        self.retention_profile = retention_profile
        floors = torch.zeros(self.W_fast, dtype=torch.float64)
        floors[self.W_fast // 2:3 * self.W_fast // 4] = 0.99
        floors[3 * self.W_fast // 4:] = 0.999
        self.register_buffer("retention_floors", floors, persistent=False)
        self.use_pointer_copy = use_pointer_copy
        if use_pointer_copy:
            ptr_dim = 64
            self.ptr_q = nn.Linear(W_fast + embed_dim, ptr_dim)
            self.ptr_k = nn.Linear(embed_dim, ptr_dim)
            self.ptr_gate = nn.Linear(W_fast + embed_dim, 1)
            self.ptr_scale = 25.0
            self.ptr_seq_boost = nn.Parameter(torch.tensor(15.0))
            nn.init.constant_(self.ptr_gate.bias, -2.0)
        else:
            self.ptr_q = None
            self.ptr_k = None
            self.ptr_gate = None
            self.ptr_scale = 25.0
            self.ptr_seq_boost = None

        # (d) Game / POMDP Action Policy Head (discrete 0..n_actions-1)
        self.action_head = nn.Sequential(
            nn.Linear(W_fast, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, n_actions),
        )

        # (c) Consequence / Value Prediction Head (r_hat for CGSL online adaptation)
        self.value_head = nn.Sequential(
            nn.Linear(W_fast, head_hidden // 2),
            nn.GELU(),
            nn.Linear(head_hidden // 2, 1),
        )

        # (d) Tool-Calling Decision Head (P(call_tool) in [0, 1])
        self.tool_gate = nn.Sequential(
            nn.Linear(W_fast, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 7. Slot Orthogonality Anchors
        self.register_buffer(
            "slot_identities",
            deterministic_thought_identity_codes(thoughtlets=K_fast, width=W_fast),
            persistent=False,
        )
        self.register_buffer(
            "episodic_identities",
            deterministic_thought_identity_codes(thoughtlets=K_episodic, width=W_episodic),
            persistent=False,
        )
        self.compensated_state = False
        if compensated_state:
            self.enable_compensated_state()

    @property
    def logical_slots(self) -> int:
        return self.K_fast // 2 if self.compensated_state else self.K_fast

    def enable_compensated_state(self) -> None:
        """Use paired float32 slots and float64 arithmetic without growing fast state.

        This explicit candidate mode trades half the logical slots for numerical
        residuals. Weights/temporary arithmetic use double precision, not a KV cache.
        """
        if self.K_fast % 2 or self.K_fast < 8:
            raise ValueError("Compensated state requires an even number of at least eight physical slots")
        self.compensated_state = True
        self.double()

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def init_state(self, batch_size: int, device: torch.device) -> UnifiedCognitiveState:
        """Initialize orthogonal two-tier cognitive state."""
        fast_init = self.slot_identities.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)
        if self.compensated_state:
            fast_init = fast_init.float()
            fast_init[:, self.logical_slots:] = 0
        ep_init = self.episodic_identities.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)

        h_state = HierarchicalCognitiveState(
            working_thoughts=fast_init,
            episodic_thoughts=ep_init,
            working_prev=fast_init.clone(),
            episodic_ages=torch.full((batch_size, self.K_episodic), 100.0, device=device),
            step_count=torch.zeros(batch_size, dtype=torch.long, device=device),
            P_t=None,
            active_thread=torch.zeros(batch_size, dtype=torch.long, device=device),
            working_salience=torch.ones(batch_size, self.K_fast, device=device),
            config=self.memory_cfg,
        )
        active_thread = torch.zeros(batch_size, dtype=torch.long, device=device)
        return UnifiedCognitiveState(
            hierarchical_state=h_state,
            active_thread=active_thread,
            P_t=None,
            last_action=torch.zeros(batch_size, dtype=torch.long, device=device),
            ptr_prev_alpha=None,
            ptr_prev_gamma=None,
        )

    def encode_sensory(
        self,
        token_ids: Optional[Tensor] = None,
        pixels: Optional[Tensor] = None,
        past_action: Optional[Tensor] = None,
    ) -> Tensor:
        """Projects multi-modal sensory inputs into unified proj_dim representation."""
        proj_parts: List[Tensor] = []

        if token_ids is not None:
            emb = self.embedding(token_ids)
            proj_parts.append(self.lang_proj(emb))

        if pixels is not None:
            vis_feat = self.visual_encoder(pixels, as_entities=False)
            proj_parts.append(self.visual_proj(vis_feat))

        if past_action is not None:
            act_emb = self.action_embed(past_action)
            proj_parts.append(self.action_proj(act_emb))

        if not proj_parts:
            raise ValueError("At least one sensory input (tokens, pixels, action) must be provided.")

        if len(proj_parts) == 1:
            sensory = proj_parts[0]
        else:
            sensory = torch.stack(proj_parts, dim=0).sum(dim=0)

        if self.deep_proj is not None:
            sensory = self.deep_proj(sensory)
        return sensory

    def step(
        self,
        sensory_input: Tensor,
        state: UnifiedCognitiveState,
        thread_id: Optional[Tensor] = None,
        allow_routing: bool = False,
        salience: Optional[Tensor] = None,
        token_id: Optional[Tensor] = None,
        token_embed: Optional[Tensor] = None,
        prompt_tokens: Optional[Tensor] = None,
        read_language: bool = True,
    ) -> Tuple[Dict[str, Tensor], UnifiedCognitiveState]:
        """Streaming update with optional omission of unused language readout.

        Pointer-active steps still compute the readout because the legacy
        pointer carries attention state. Cost does not grow with generated history.
        """
        B = sensory_input.shape[0]
        dev = sensory_input.device

        if thread_id is not None:
            if self.compensated_state and bool(((thread_id < 0) | (thread_id >= self.logical_slots)).any()):
                raise ValueError("Compensated mode has eight logical slots; residual slots cannot be addressed")
            active_tid = torch.clamp(thread_id, 0, self.logical_slots - 1)
        else:
            active_tid = state.active_thread

        working = state.hierarchical_state.working_thoughts.clone()
        batch_idx = torch.arange(B, device=dev)
        active_slot = working[batch_idx, active_tid]
        if self.compensated_state:
            if allow_routing:
                raise ValueError("Compensated-state routing is not implemented")
            active_slot = active_slot.double() + working[batch_idx, active_tid + self.logical_slots].double()

        # 1. Cognitive Input Gating (default salience = 1.0 for active streaming tokens)
        if salience is not None:
            g_t = salience if isinstance(salience, Tensor) else torch.full((B, 1), float(salience), device=dev)
        else:
            g_t = torch.ones((B, 1), device=dev)

        # 2. Associative Recurrent Step (Exact O(1) Streaming Counterpart of Parallel Scan)
        # Note: sensory_input was already projected through deep_proj in encode_sensory
        u_sensory = self.funnel_down(sensory_input) if (self.use_funnel and self.funnel_down is not None) else sensory_input
        a_t = self._retention_gates(u_sensory)
        b_t = (1.0 - a_t) * torch.tanh(self.W_in_parallel(u_sensory))
        updated_slot = a_t * active_slot + b_t

        gated_slot = (1.0 - g_t) * active_slot + g_t * updated_slot
        working[batch_idx, active_tid] = gated_slot.to(working.dtype)
        if self.compensated_state:
            working[batch_idx, active_tid + self.logical_slots] = (
                gated_slot - working[batch_idx, active_tid].double()
            ).float()

        # 3. Sparse Thought Routing
        if self.use_routing and self.router is not None and allow_routing:
            routed_working, _ = self.router(working)
            working = 0.9 * working + 0.1 * routed_working

        # 4. Cross-Tier Consolidation & Episodic Retrieval
        salience_3d = g_t.unsqueeze(1).expand(-1, self.K_fast, 1)
        memory_input = working
        if self.compensated_state:
            # Episodic operations use reconstructed logical vectors, not residual slots.
            memory_input = working[:, :self.logical_slots].double() + working[:, self.logical_slots:].double()
            salience_3d = salience_3d[:, :self.logical_slots].double()
        ep_thoughts, ep_ages, *_ = self.consolidation_gate(
            memory_input,
            state.hierarchical_state.episodic_thoughts,
            state.hierarchical_state.episodic_ages,
            salience=salience_3d,
        )

        augmented_slot, retrieved_ctx, *_ = self.retrieval_module(
            memory_input[batch_idx, active_tid].unsqueeze(1),
            ep_thoughts,
        )
        if allow_routing:
            if self.compensated_state:
                contextualized_slot = (
                    working[batch_idx, active_tid].double()
                    + working[batch_idx, active_tid + self.logical_slots].double()
                    + retrieved_ctx.squeeze(1).double()
                )
            else:
                contextualized_slot = augmented_slot.squeeze(1)
        else:
            contextualized_slot = working[batch_idx, active_tid]
            if self.compensated_state:
                contextualized_slot = (
                    contextualized_slot.double()
                    + working[batch_idx, active_tid + self.logical_slots].double()
                )

        # 5. Decoupled Readout Predictions
        language_logits = (self.slot_head(self.readout_norm(contextualized_slot))
                           if read_language or prompt_tokens is not None else None)
        skip_beta = None
        if language_logits is not None and self.use_token_skip and self.token_skip is not None:
            if token_embed is None and token_id is not None:
                token_embed = self.embedding(token_id)
            if token_embed is not None:
                skip_logits = self.token_skip(token_embed)
                if hasattr(self, "skip_gate") and self.skip_gate is not None:
                    gate_in = torch.cat([contextualized_slot, token_embed], dim=-1)
                    skip_beta = torch.sigmoid(self.skip_gate(gate_in))
                    language_logits = language_logits + skip_beta * skip_logits
                else:
                    scale = getattr(self, "token_skip_scale", 1.0)
                    language_logits = language_logits + scale * skip_logits

        # 5b. In-Context Sequential Pointer-Copy Readout
        ptr_gamma = None
        ptr_attn = None
        if self.use_pointer_copy and self.ptr_q is not None and prompt_tokens is not None:
            if token_embed is None and token_id is not None:
                token_embed = self.embedding(token_id)
            if token_embed is not None:
                gate_in = torch.cat([contextualized_slot, token_embed], dim=-1)
                ptr_gamma = torch.sigmoid(self.ptr_gate(gate_in))
                q_ptr = self.ptr_q(gate_in)
                prompt_embeds = self.embedding(prompt_tokens)
                k_ptr = self.ptr_k(prompt_embeds)
                ptr_dim = q_ptr.shape[-1]
                scores = torch.bmm(q_ptr.unsqueeze(1), k_ptr.transpose(1, 2)).squeeze(1) / (ptr_dim ** 0.5)
                pad_mask = (prompt_tokens == 0)
                scores = scores.masked_fill(pad_mask, -1e9)

                # Sequential shift prior from previous pointer attention
                if self.pointer_mode == "sequential" and state.ptr_prev_alpha is not None and state.ptr_prev_gamma is not None and self.ptr_seq_boost is not None:
                    prev_a = state.ptr_prev_alpha
                    cur_n = prompt_tokens.shape[1]
                    if prev_a.shape[1] != cur_n:
                        padded_prev_a = torch.zeros(B, cur_n, device=dev)
                        min_n = min(prev_a.shape[1], cur_n)
                        padded_prev_a[:, :min_n] = prev_a[:, :min_n]
                        prev_a = padded_prev_a
                    shift_prior = torch.zeros_like(prev_a)
                    shift_prior[:, 1:] = prev_a[:, :-1]
                    scores = scores + self.ptr_seq_boost * state.ptr_prev_gamma * shift_prior

                    # Monotonic Causal Lower-Bound Mask: prevent backward attention jumps during active copying
                    prev_max_idx = prev_a.argmax(dim=-1, keepdim=True)
                    pos_indices = torch.arange(cur_n, device=dev).unsqueeze(0).expand(B, -1)
                    backward_mask = (pos_indices < prev_max_idx) & (state.ptr_prev_gamma > 0.3)
                    scores = scores.masked_fill(backward_mask, -1e9)

                if self.pointer_mode == "parallel_predecessor":
                    ptr_attn = self._parallel_pointer_attention(scores, token_id, prompt_tokens)
                else:
                    ptr_attn = torch.softmax(scores, dim=-1)
                ptr_boost = ptr_attn * (ptr_gamma * self.ptr_scale)
                language_logits = language_logits.scatter_add(dim=-1, index=prompt_tokens, src=ptr_boost)

        action_logits = self.action_head(contextualized_slot)
        predicted_value = self.value_head(contextualized_slot)
        tool_call_prob = self.tool_gate(contextualized_slot)

        next_step_count = state.hierarchical_state.step_count + 1
        p_t = state.P_t if state.P_t is not None else state.hierarchical_state.P_t

        next_hierarchical = HierarchicalCognitiveState(
            working_thoughts=working,
            episodic_thoughts=ep_thoughts,
            working_prev=state.hierarchical_state.working_thoughts.clone(),
            episodic_ages=ep_ages,
            step_count=next_step_count,
            P_t=p_t,
            active_thread=active_tid,
            working_salience=g_t.expand(-1, self.K_fast),
            config=self.memory_cfg,
        )

        next_state = UnifiedCognitiveState(
            hierarchical_state=next_hierarchical,
            active_thread=active_tid,
            P_t=p_t,
            last_action=action_logits.argmax(dim=-1),
            ptr_prev_alpha=ptr_attn.detach() if ptr_attn is not None and self.pointer_mode == "sequential" else None,
            ptr_prev_gamma=ptr_gamma.detach() if ptr_gamma is not None and self.pointer_mode == "sequential" else None,
        )

        outputs = {
            "logits": language_logits,
            "action_logits": action_logits,
            "predicted_value": predicted_value,
            "tool_prob": tool_call_prob,
            "skip_beta": skip_beta,
            "ptr_gamma": ptr_gamma,
            "ptr_attn": ptr_attn,
        }
        return outputs, next_state

    def _retention_gates(self, sensory):
        gates = torch.sigmoid(self.W_iz_parallel(sensory))
        if self.retention_profile == "multiscale":
            floors = self.retention_floors.to(dtype=gates.dtype)
            gates = floors + (1.0 - floors) * gates
        return gates

    def _parallel_pointer_attention(self, scores, token_ids, prompt_tokens):
        """Batched copy continuation from the current input and immutable prompt.

        No previous attention or generated-token history is read. Each time row
        is independent after the recurrent scan. Duplicate predecessors remain
        alternatives for the content query, rather than receiving an oracle index.
        """
        if token_ids is None:
            raise ValueError("parallel_predecessor requires current token IDs")
        single_step = scores.ndim == 2
        scores = scores.unsqueeze(1) if single_step else scores
        token_ids = token_ids.unsqueeze(1) if single_step else token_ids
        predecessor = torch.zeros_like(scores, dtype=torch.bool)
        predecessor[..., 1:] = (
            token_ids.unsqueeze(-1) == prompt_tokens[:, None, :-1]
        ) & (prompt_tokens[:, None, :-1] != 0)
        scores = scores + self.ptr_seq_boost * predecessor.to(scores.dtype)
        valid = (prompt_tokens != 0).unsqueeze(1)
        # Finite minimum avoids NaNs for an entirely padded prompt. Multiplying
        # by valid then makes its pointer contribution exactly zero.
        attention = torch.softmax(scores.masked_fill(~valid, torch.finfo(scores.dtype).min), dim=-1)
        attention = attention * valid
        return attention.squeeze(1) if single_step else attention

    def forward_sequence_parallel(
        self,
        token_seq: Optional[Tensor] = None,
        pixel_seq: Optional[Tensor] = None,
        action_seq: Optional[Tensor] = None,
        reset_mask: Optional[Tensor] = None,
        prompt_tokens: Optional[Tensor] = None,
        pointer_mask: Optional[Tensor] = None,
        language_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """O(log T) Parallel Associative Scan Sequence Evaluation for High-Throughput GPU Training."""
        B = None
        T = None
        dev = None
        sensory_components = []

        tok_emb = None
        if token_seq is not None:
            B, T = token_seq.shape
            dev = token_seq.device
            tok_emb = self.embedding(token_seq)
            sensory_components.append(self.lang_proj(tok_emb))

        if pixel_seq is not None:
            B, T, C, H, W = pixel_seq.shape
            dev = pixel_seq.device
            flat_pixels = pixel_seq.view(B * T, C, H, W)
            flat_feats = self.visual_encoder(flat_pixels, as_entities=False)
            sensory_components.append(self.visual_proj(flat_feats).view(B, T, -1))

        if action_seq is not None:
            B, T = action_seq.shape
            dev = action_seq.device
            sensory_components.append(self.action_proj(self.action_embed(action_seq)))

        if not sensory_components:
            raise ValueError("Must provide at least one sequence input.")

        X_seq = torch.stack(sensory_components, dim=0).sum(dim=0)
        if self.deep_proj is not None:
            X_seq = self.deep_proj(X_seq)

        u_seq = self.funnel_down(X_seq) if (self.use_funnel and self.funnel_down is not None) else X_seq
        A_gates = self._retention_gates(u_seq)
        B_cands = (1.0 - A_gates) * torch.tanh(self.W_in_parallel(u_seq))

        h_init = self.slot_identities[0].unsqueeze(0).expand(B, -1)

        if reset_mask is not None:
            mask_3d = reset_mask.unsqueeze(-1).to(dtype=A_gates.dtype)  # [B, T, 1]
            h_init_3d = h_init.unsqueeze(1).expand(-1, T, -1)  # [B, T, W]
            # Fold h_init into B_cands at boundary positions BEFORE zeroing A_gates
            B_cands = B_cands + mask_3d * (A_gates * h_init_3d)
            # Sever recurrent state propagation across document boundaries
            A_gates = A_gates * (1.0 - mask_3d)
            H_scanned = triton_scan(A_gates, B_cands, h_init=h_init)
        else:
            H_scanned = triton_scan(A_gates, B_cands, h_init=h_init)

        # Training can request logits only at supervised positions. Recurrent
        # transitions still consume the complete trajectory, including failures.
        readout_h = H_scanned if language_mask is None else H_scanned[language_mask]
        readout_emb = tok_emb if language_mask is None or tok_emb is None else tok_emb[language_mask]
        language_logits = self.slot_head(self.readout_norm(readout_h))
        beta = None
        if self.use_token_skip and tok_emb is not None and self.token_skip is not None:
            skip_logits = self.token_skip(readout_emb)
            if hasattr(self, "skip_gate") and self.skip_gate is not None:
                gate_in = torch.cat([readout_h, readout_emb], dim=-1)
                beta = torch.sigmoid(self.skip_gate(gate_in))
                language_logits = language_logits + beta * skip_logits
            else:
                scale = getattr(self, "token_skip_scale", 1.0)
                language_logits = language_logits + scale * skip_logits

        # In-Context Sequential Pointer-Copy Readout
        ptr_gamma = None
        ptr_attn = None
        if self.use_pointer_copy and self.ptr_q is not None and prompt_tokens is not None and tok_emb is not None:
            gate_in = torch.cat([H_scanned, tok_emb], dim=-1)
            ptr_gamma = torch.sigmoid(self.ptr_gate(gate_in))  # [B, T, 1]
            q_ptr = self.ptr_q(gate_in)  # [B, T, ptr_dim]
            prompt_embeds = self.embedding(prompt_tokens)  # [B, N, embed_dim]
            k_ptr = self.ptr_k(prompt_embeds)  # [B, N, ptr_dim]
            ptr_dim = q_ptr.shape[-1]
            content_scores = torch.einsum("btd,bnd->btn", q_ptr, k_ptr) / (ptr_dim ** 0.5)  # [B, T, N]
            pad_mask = (prompt_tokens == 0).unsqueeze(1).expand(-1, content_scores.shape[1], -1)
            content_scores = content_scores.masked_fill(pad_mask, -1e9)

            if self.pointer_mode == "parallel_predecessor":
                ptr_attn = self._parallel_pointer_attention(content_scores, token_seq, prompt_tokens)
                if pointer_mask is not None:
                    ptr_attn = ptr_attn * pointer_mask.unsqueeze(-1)
            else:
                # Legacy nonlinear attention recurrence (serial over time).
                alphas = []
                cur_n = prompt_tokens.shape[1]
                alpha_prev = torch.zeros(B, cur_n, device=content_scores.device)
                gamma_prev = torch.zeros(B, 1, device=content_scores.device)
                pos_indices = torch.arange(cur_n, device=content_scores.device).unsqueeze(0).expand(B, -1)
                boost = self.ptr_seq_boost if self.ptr_seq_boost is not None else 15.0
                for t in range(T):
                    if pointer_mask is not None:
                        # Observations are recurrent inputs, never pointer sources or history.
                        keep = pointer_mask[:, t].unsqueeze(-1)
                        alpha_prev = alpha_prev * keep
                        gamma_prev = gamma_prev * keep
                    shift_prior = torch.zeros_like(alpha_prev)
                    shift_prior[:, 1:] = alpha_prev[:, :-1]
                    s_t = content_scores[:, t] + boost * gamma_prev * shift_prior

                    # Monotonic Causal Lower-Bound Mask: prevent backward attention jumps during active copying
                    prev_max_idx = alpha_prev.argmax(dim=-1, keepdim=True)
                    backward_mask = (pos_indices < prev_max_idx) & (gamma_prev > 0.3)
                    s_t = s_t.masked_fill(backward_mask, -1e9)

                    alpha_t = torch.softmax(s_t, dim=-1)
                    if pointer_mask is not None:
                        alpha_t = alpha_t * keep
                    alphas.append(alpha_t)
                    alpha_prev = alpha_t
                    gamma_prev = ptr_gamma[:, t]
                    if pointer_mask is not None:
                        gamma_prev = gamma_prev * keep
                ptr_attn = torch.stack(alphas, dim=1)  # [B, T, N]
            ptr_boost = ptr_attn * (ptr_gamma * self.ptr_scale)  # [B, T, N]
            expanded_prompt_tokens = prompt_tokens.unsqueeze(1).expand(-1, T, -1)  # [B, T, N]
            if language_mask is not None:
                expanded_prompt_tokens = expanded_prompt_tokens[language_mask]
                ptr_boost = ptr_boost[language_mask]
            language_logits = language_logits.scatter_add(dim=-1, index=expanded_prompt_tokens, src=ptr_boost)

        action_logits = self.action_head(H_scanned)
        values = self.value_head(H_scanned).squeeze(-1)

        return {
            "logits": language_logits,
            "action_logits": action_logits,
            "values": values,
            "H_scanned": H_scanned,
            "A_gates": A_gates,
            "skip_beta": beta,
            "ptr_gamma": ptr_gamma,
            "ptr_attn": ptr_attn,
        }

    def forward_sequence_sequential(
        self,
        token_seq: Optional[Tensor] = None,
        pixel_seq: Optional[Tensor] = None,
        action_seq: Optional[Tensor] = None,
        reset_mask: Optional[Tensor] = None,
        prompt_tokens: Optional[Tensor] = None,
        pointer_mask: Optional[Tensor] = None,
        language_mask: Optional[Tensor] = None,
        allow_routing: bool = False,
        thread_seq: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Exact sequential unrolled reference training and evaluation path.

        Executes the exact same step-by-step state transitions, slot routing (when
        allow_routing=True), cross-tier episodic retrieval, and pointer mechanics
        as streaming inference model.step().
        """
        B = None
        T = None
        dev = None

        if token_seq is not None:
            B, T = token_seq.shape
            dev = token_seq.device
        elif pixel_seq is not None:
            B, T = pixel_seq.shape[:2]
            dev = pixel_seq.device
        elif action_seq is not None:
            B, T = action_seq.shape
            dev = action_seq.device
        else:
            raise ValueError("Must provide at least one of token_seq, pixel_seq, or action_seq.")

        state = self.init_state(B, dev)
        logits_list = []
        action_list = []
        values_list = []

        for t in range(T):
            sensory = self.encode_sensory(
                token_ids=token_seq[:, t] if token_seq is not None else None,
                pixels=pixel_seq[:, t] if pixel_seq is not None else None,
                past_action=action_seq[:, t] if action_seq is not None else None,
            )
            if reset_mask is not None:
                reset = reset_mask[:, t].bool()
                if reset.any():
                    initial = self.init_state(B, dev)
                    reset_3d = reset[:, None, None]
                    reset_2d = reset[:, None]
                    state.hierarchical_state.working_thoughts = torch.where(
                        reset_3d, initial.hierarchical_state.working_thoughts,
                        state.hierarchical_state.working_thoughts,
                    )
                    state.hierarchical_state.episodic_thoughts = torch.where(
                        reset_3d, initial.hierarchical_state.episodic_thoughts,
                        state.hierarchical_state.episodic_thoughts,
                    )
                    state.hierarchical_state.episodic_ages = torch.where(
                        reset_2d, initial.hierarchical_state.episodic_ages,
                        state.hierarchical_state.episodic_ages,
                    )
                    state.hierarchical_state.step_count = torch.where(
                        reset, initial.hierarchical_state.step_count,
                        state.hierarchical_state.step_count,
                    )
                    if state.ptr_prev_alpha is not None:
                        state.ptr_prev_alpha = torch.where(reset_2d, torch.zeros_like(state.ptr_prev_alpha), state.ptr_prev_alpha)
                        state.ptr_prev_gamma = torch.where(reset_2d, torch.zeros_like(state.ptr_prev_gamma), state.ptr_prev_gamma)

            if pointer_mask is not None and state.ptr_prev_alpha is not None:
                keep = pointer_mask[:, t].unsqueeze(-1)
                state.ptr_prev_alpha = state.ptr_prev_alpha * keep
                state.ptr_prev_gamma = state.ptr_prev_gamma * keep

            tid = thread_seq[:, t] if thread_seq is not None else None
            tok_t = token_seq[:, t] if token_seq is not None else None

            outputs, state = self.step(
                sensory,
                state,
                thread_id=tid,
                allow_routing=allow_routing,
                token_id=tok_t,
                prompt_tokens=prompt_tokens,
            )

            if pointer_mask is not None and outputs["ptr_attn"] is not None and prompt_tokens is not None:
                keep = pointer_mask[:, t].unsqueeze(-1)
                # Remove pointer contribution on observation/padding positions.
                boost = outputs["ptr_attn"] * (outputs["ptr_gamma"] * self.ptr_scale)
                outputs["logits"] = outputs["logits"].scatter_add(
                    -1, prompt_tokens, -boost * (1 - keep),
                )
                if state.ptr_prev_alpha is not None:
                    state.ptr_prev_alpha = state.ptr_prev_alpha * keep
                    state.ptr_prev_gamma = state.ptr_prev_gamma * keep

            logits_list.append(outputs["logits"])
            action_list.append(outputs["action_logits"])
            values_list.append(outputs["predicted_value"].squeeze(-1))

        stacked_logits = torch.stack(logits_list, dim=1)
        readout_logits = stacked_logits if language_mask is None else stacked_logits[language_mask]

        return {
            "logits": readout_logits,
            "action_logits": torch.stack(action_list, dim=1),
            "values": torch.stack(values_list, dim=1),
        }

    def forward(
        self,
        token_seq: Optional[Tensor] = None,
        pixel_seq: Optional[Tensor] = None,
        action_seq: Optional[Tensor] = None,
        parallel: bool = True,
        reset_mask: Optional[Tensor] = None,
        prompt_tokens: Optional[Tensor] = None,
        pointer_mask: Optional[Tensor] = None,
        language_mask: Optional[Tensor] = None,
        allow_routing: bool = False,
        thread_seq: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Unified forward pass."""
        if parallel and (token_seq is not None or pixel_seq is not None):
            return self.forward_sequence_parallel(
                token_seq=token_seq,
                pixel_seq=pixel_seq,
                action_seq=action_seq,
                reset_mask=reset_mask,
                prompt_tokens=prompt_tokens,
                pointer_mask=pointer_mask,
                language_mask=language_mask,
            )

        return self.forward_sequence_sequential(
            token_seq=token_seq,
            pixel_seq=pixel_seq,
            action_seq=action_seq,
            reset_mask=reset_mask,
            prompt_tokens=prompt_tokens,
            pointer_mask=pointer_mask,
            language_mask=language_mask,
            allow_routing=allow_routing,
            thread_seq=thread_seq,
        )


def make_unified_model(
    tier: str = "tier2",
    vocab_size: int = 32000,
    **kwargs: Any,
) -> UnifiedPseudoBrain:
    """Factory function for Unified Pseudo-Brain architecture across tiers."""
    if kwargs.get("compensated_state", False):
        if "use_routing" not in kwargs:
            kwargs["use_routing"] = False
        elif kwargs["use_routing"]:
            raise ValueError("compensated_state=True cannot be combined with use_routing=True")

    tier_norm = tier.lower().replace("-", "_")
    if tier_norm in ("tier0", "micro", "131k"):
        return UnifiedPseudoBrain(
            vocab_size=vocab_size,
            tier="tier0",
            K_fast=8,
            W_fast=32,
            K_episodic=16,
            W_episodic=64,
            embed_dim=64,
            proj_dim=128,
            rank=16,
            num_deep_layers=2,
            **kwargs,
        )
    elif tier_norm in ("tier1", "embedded", "10m", "tier1_3m", "3m"):
        return UnifiedPseudoBrain(
            vocab_size=vocab_size,
            tier="tier1",
            K_fast=16,
            W_fast=64,
            K_episodic=32,
            W_episodic=64,
            embed_dim=128,
            proj_dim=512,
            rank=32,
            num_deep_layers=3,
            **kwargs,
        )
    elif tier_norm in ("tier2_35m", "35m"):
        return UnifiedPseudoBrain(
            vocab_size=vocab_size,
            tier="tier2_35m",
            K_fast=16,
            W_fast=64,
            K_episodic=64,
            W_episodic=128,
            embed_dim=kwargs.pop("embed_dim", 384),
            proj_dim=kwargs.pop("proj_dim", 4096),
            rank=kwargs.pop("rank", 32),
            num_deep_layers=kwargs.pop("num_deep_layers", 4),
            **kwargs,
        )
    elif tier_norm in ("tier2", "conversational", "50m"):
        return UnifiedPseudoBrain(
            vocab_size=vocab_size,
            tier="tier2",
            K_fast=16,
            W_fast=64,
            K_episodic=64,
            W_episodic=128,
            embed_dim=256,
            proj_dim=1024,
            rank=32,
            num_deep_layers=4,
            **kwargs,
        )
    elif tier_norm in ("tier3", "agentic", "100m", "300m"):
        return UnifiedPseudoBrain(
            vocab_size=vocab_size,
            tier="tier3",
            K_fast=16,
            W_fast=64,
            K_episodic=128,
            W_episodic=128,
            embed_dim=512,
            proj_dim=2048,
            rank=32,
            num_deep_layers=12,
            **kwargs,
        )
    elif tier_norm in ("tier3_1b", "tier5", "tier5_1b", "1b", "general"):
        return UnifiedPseudoBrain(
            vocab_size=vocab_size,
            tier="tier3_1b",
            **kwargs,
        )
    else:
        valid_presets = ["tier0", "tier1", "tier2", "tier2_35m", "tier3", "tier3_1b"]
        raise ValueError(f"Unknown tier: '{tier}'. Declared presets are: {valid_presets}")
