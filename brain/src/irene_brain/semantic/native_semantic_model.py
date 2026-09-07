"""Native Semantic Language Model Architecture based on Pseudo-Brain.

Processes and generates language sequences directly through recurrent cognitive threads,
Cognitive Input Gating (CIG), BrainCellCore, and Consequence-Gated Synaptic Latching (CGSL),
without using external LLMs or Transformers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from irene_brain.model.brain_cell import BrainCellCore
from irene_brain.model.sparse_thought_router import SparseThoughtRouter
from irene_brain.model.torch_model import deterministic_thought_identity_codes


@dataclass
class SemanticCognitiveState:
    """Persistent cognitive state across streaming steps."""
    thoughts: torch.Tensor  # [B, K, W]
    P_t: torch.Tensor  # [B, K, V]
    prev_thoughts: torch.Tensor  # [B, K, W]
    active_thread: torch.Tensor  # [B] integer thread index


class NativeSemanticPseudoBrain(nn.Module):
    """Pseudo-Brain Native Semantic Cognitive Architecture."""

    def __init__(
        self,
        vocab_size: int = 344,
        K: int = 16,
        thought_size: int = 32,
        embed_dim: int = 64,
        proj_dim: int = 128,
        routed_neighbors: int = 2,
        plastic_decay: float = 0.9999,
        plastic_lr: float = 0.25,
        use_cgp: bool = True,
        use_routing: bool = True,
        tier: Optional[str] = None,
        rank: Optional[int] = None,
        num_deep_layers: int = 3,
        conditional_recurrence: bool = False,
        epsilon_dormant: float = 0.05,
    ):
        super().__init__()
        self.tier = tier
        if tier == "tier2":
            # Apply Law 1 (Bounded W=64) & Law 2 (Rank r=32, Proj_dim=4096)
            thought_size = 64
            rank = 32 if rank is None else rank
            proj_dim = 4096 if proj_dim == 128 else proj_dim
            embed_dim = 128 if embed_dim == 64 else embed_dim

        self.vocab_size = vocab_size
        self.K = K
        self.thought_size = thought_size
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.rank = rank
        self.num_deep_layers = num_deep_layers
        self.conditional_recurrence = conditional_recurrence
        self.epsilon_dormant = epsilon_dormant
        self.plastic_decay = plastic_decay
        self.plastic_lr = plastic_lr
        self.use_cgp = use_cgp
        self.use_routing = use_routing

        # 1. Semantic Token Embedding & Sensory Projection
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, proj_dim)

        # 2. Shared Recurrent Core across K thought slots
        self.brain_cell = BrainCellCore(
            input_size=proj_dim,
            thought_size=thought_size,
            rank=rank,
            proj_dim=proj_dim,
            tier=tier,
            num_deep_layers=num_deep_layers,
        )

        # 3. Cognitive Input Gating with T=0.5 sharpening
        self.cig_gate = nn.Sequential(
            nn.Linear(proj_dim + thought_size, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.cig_gate[0].bias, 1.0)

        # 4. Selective Routing (for cross-thread communication)
        if self.use_routing and self.K > 1:
            self.router = SparseThoughtRouter(
                width=thought_size,
                routed_neighbors=min(routed_neighbors, max(1, K - 1)),
                dense_routing=False,
            )
        else:
            self.router = None

        # 5. Consequence-Gated Synaptic Latching (CGSL)
        if self.use_cgp:
            self.plastic_modulator = nn.Linear(thought_size + 16, vocab_size)
            self.surprise_gate = nn.Sequential(
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )
            nn.init.constant_(self.surprise_gate[0].bias, -1.0)
            self.plastic_scale = nn.Parameter(torch.tensor(1.0))
            self.surprise_encoder = nn.Sequential(
                nn.Linear(1, 16),
                nn.LayerNorm(16),
                nn.GELU(),
                nn.Linear(16, 16),
            )
        else:
            self.plastic_modulator = None
            self.surprise_gate = None
            self.plastic_scale = None
            self.surprise_encoder = None

        # 6. Thread-Targeted Semantic Readout Head
        head_hidden = max(64, min(proj_dim // 2, 512))
        self.slot_head = nn.Sequential(
            nn.Linear(thought_size, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, vocab_size),
        )

        # 7. Slot Orthogonality Codes
        self.register_buffer(
            "slot_identities",
            deterministic_thought_identity_codes(thoughtlets=K, width=thought_size),
            persistent=False,
        )

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def state_bytes(self, bytes_per_element: int = 4) -> int:
        """Law 1 Recurrent State Footprint: K * W * 4 bytes."""
        return self.K * self.thought_size * bytes_per_element

    def init_state(self, batch_size: int, device: torch.device) -> SemanticCognitiveState:
        """Initialize zero/identity state for streaming."""
        if self.slot_identities.shape[0] != self.K:
            slots = deterministic_thought_identity_codes(thoughtlets=self.K, width=self.thought_size).to(device)
        else:
            slots = self.slot_identities.to(device=device)

        thoughts = slots.unsqueeze(0).expand(batch_size, -1, -1).clone()
        P_t = torch.zeros(batch_size, self.K, self.vocab_size, device=device)
        active_thread = torch.zeros(batch_size, dtype=torch.long, device=device)
        return SemanticCognitiveState(
            thoughts=thoughts,
            P_t=P_t,
            prev_thoughts=thoughts.clone(),
            active_thread=active_thread,
        )

    def step(
        self,
        token_ids: torch.Tensor,  # [B]
        state: SemanticCognitiveState,
        thread_ids: Optional[torch.Tensor] = None,  # [B] optional explicit thread addressing
        allow_routing: bool = False,
    ) -> Tuple[torch.Tensor, SemanticCognitiveState]:
        """Single streaming token cognitive update step. Returns (logits [B, V], next_state)."""
        B = token_ids.shape[0]
        dev = token_ids.device

        # Update active thread if explicit or if token is thread token [THREAD:i] (tokens 9 .. 9+K-1)
        if thread_ids is not None:
            state.active_thread = thread_ids.clamp(0, self.K - 1)
        else:
            # Check if token is thread marker (base_thread_offset is 9 in tokenizer)
            is_thread_tok = (token_ids >= 9) & (token_ids < 9 + self.K)
            if is_thread_tok.any():
                new_tids = (token_ids - 9).clamp(0, self.K - 1)
                state.active_thread = torch.where(is_thread_tok, new_tids, state.active_thread)

        # 1. Embed & Project
        emb = self.embedding(token_ids)  # [B, embed_dim]
        x_proj = self.proj(emb)  # [B, proj_dim]

        # 2. Expand across K slots
        x_exp = x_proj.unsqueeze(1).expand(-1, self.K, -1)  # [B, K, proj_dim]

        # 3. Cognitive Input Gating with T=0.5 Sharpening
        gate_in = torch.cat([x_exp, state.thoughts], dim=-1)
        raw_gate = self.cig_gate(gate_in)
        clamped_gate = raw_gate.clamp(1e-4, 1.0 - 1e-4)
        logit_gate = torch.log(clamped_gate / (1.0 - clamped_gate))
        salience = torch.sigmoid(logit_gate / 0.5)

        # 4. Recurrent Core Update (with optional event-driven conditional recurrence)
        if self.conditional_recurrence:
            active_mask = (salience.squeeze(-1) >= self.epsilon_dormant).any(dim=0)
            if not active_mask.any():
                new_t = state.thoughts
            elif active_mask.all():
                t_flat = state.thoughts.reshape(B * self.K, self.thought_size)
                x_flat = x_exp.reshape(B * self.K, -1)
                new_t = self.brain_cell(t_flat, x_flat).reshape(B, self.K, self.thought_size)
            else:
                active_indices = torch.where(active_mask)[0]
                n_act = len(active_indices)
                t_active = state.thoughts[:, active_indices, :]
                x_active = x_exp[:, active_indices, :]
                t_flat = t_active.reshape(B * n_act, self.thought_size)
                x_flat = x_active.reshape(B * n_act, -1)
                new_act = self.brain_cell(t_flat, x_flat).reshape(B, n_act, self.thought_size)
                new_t = state.thoughts.clone()
                new_t[:, active_indices, :] = new_act
        else:
            t_flat = state.thoughts.reshape(B * self.K, self.thought_size)
            x_flat = x_exp.reshape(B * self.K, -1)
            new_t = self.brain_cell(t_flat, x_flat).reshape(B, self.K, self.thought_size)

        # 5. Routing between slots
        if self.use_routing and self.router is not None and allow_routing:
            messages, _ = self.router(new_t)
            t_candidate = new_t + messages
        else:
            t_candidate = new_t

        # 6. Gated State Update
        next_thoughts = (1.0 - salience) * state.thoughts + salience * t_candidate

        # 7. Fast Synaptic Latching (CGSL)
        if self.use_cgp and self.surprise_encoder is not None:
            delta_slot = torch.norm(next_thoughts - state.prev_thoughts, dim=-1, keepdim=True)
            prev_thoughts_next = next_thoughts.detach()
            surprise_emb = self.surprise_encoder(delta_slot)
            p_gate = self.surprise_gate(surprise_emb)
            delta_P = p_gate * torch.tanh(self.plastic_modulator(torch.cat([next_thoughts, surprise_emb], dim=-1)))
            next_P_t = self.plastic_decay * state.P_t + self.plastic_lr * delta_P
        else:
            prev_thoughts_next = next_thoughts
            next_P_t = state.P_t

        # 8. Thread-Targeted Readout from active thread
        batch_idx = torch.arange(B, device=dev)
        clamped_active = state.active_thread.clamp(0, self.K - 1)
        queried_slot = next_thoughts[batch_idx, clamped_active]  # [B, W]
        slot_logits = self.slot_head(queried_slot)  # [B, V]

        if self.use_cgp and self.plastic_scale is not None:
            queried_P = next_P_t[batch_idx, clamped_active]  # [B, V]
            total_logits = slot_logits + self.plastic_scale * queried_P
        else:
            total_logits = slot_logits

        next_state = SemanticCognitiveState(
            thoughts=next_thoughts,
            P_t=next_P_t,
            prev_thoughts=prev_thoughts_next,
            active_thread=state.active_thread,
        )
        return total_logits, next_state

    def forward(
        self,
        token_seq: torch.Tensor,  # [B, T]
        thread_seq: Optional[torch.Tensor] = None,  # [B, T] optional per-step thread ids
        allow_routing: bool = False,
    ) -> torch.Tensor:
        """Sequential unroll over sequence. Returns logits [B, T, V]."""
        B, T = token_seq.shape
        dev = token_seq.device
        state = self.init_state(B, dev)
        logits_list: List[torch.Tensor] = []

        for t in range(T):
            tid = thread_seq[:, t] if thread_seq is not None else None
            logits_t, state = self.step(token_seq[:, t], state, thread_ids=tid, allow_routing=allow_routing)
            logits_list.append(logits_t)

        return torch.stack(logits_list, dim=1)


# ==============================================================================
# Competitor Baseline Models (Matched Parameter Budget)
# ==============================================================================

class MonolithicGRULanguageModel(nn.Module):
    """Parameter-matched Monolithic GRU language baseline."""

    def __init__(
        self,
        vocab_size: int = 344,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def step(self, token_ids: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(token_ids).unsqueeze(1)
        x = self.proj(emb)
        out, h_next = self.gru(x, h)
        logits = self.head(out.squeeze(1))
        return logits, h_next

    def forward(self, token_seq: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        emb = self.embedding(token_seq)
        x = self.proj(emb)
        out, _ = self.gru(x)
        return self.head(out)


class ModernSSMLanguageModel(nn.Module):
    """Parameter-matched Modern Diagonal State-Space Model (GLRU / S4D-class)."""

    def __init__(
        self,
        vocab_size: int = 344,
        embed_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.gate_proj = nn.Linear(embed_dim, hidden_dim)
        self.nu = nn.Parameter(torch.randn(hidden_dim) * 0.1 + 2.0)  # Log decay timescale
        self.head = nn.Linear(hidden_dim, vocab_size)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(self, token_ids: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(token_ids)
        u = self.input_proj(emb)
        gate = torch.sigmoid(self.gate_proj(emb))
        decay = torch.sigmoid(self.nu)
        h_next = decay * h + (1.0 - decay) * u
        logits = self.head(h_next * gate)
        return logits, h_next

    def forward(self, token_seq: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        B, T = token_seq.shape
        dev = token_seq.device
        h = self.init_state(B, dev)
        logits_list = []
        for t in range(T):
            l_t, h = self.step(token_seq[:, t], h)
            logits_list.append(l_t)
        return torch.stack(logits_list, dim=1)


class ReactiveLanguageModel(nn.Module):
    """Reactive feedforward language baseline (MLP with zero recurrent state)."""

    def __init__(
        self,
        vocab_size: int = 344,
        embed_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def init_state(self, batch_size: int, device: torch.device) -> None:
        return None

    def step(self, token_ids: torch.Tensor, state: Any = None) -> Tuple[torch.Tensor, None]:
        emb = self.embedding(token_ids)
        logits = self.mlp(emb)
        return logits, None

    def forward(self, token_seq: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        emb = self.embedding(token_seq)
        return self.mlp(emb)


def make_semantic_model(
    model_type: str,
    vocab_size: int = 344,
    K: int = 16,
    thought_size: int = 32,
    embed_dim: int = 64,
    proj_dim: int = 128,
    **kwargs: Any,
) -> nn.Module:
    """Factory creating parameter-calibrated semantic models."""
    if model_type == "pseudo_brain":
        return NativeSemanticPseudoBrain(
            vocab_size=vocab_size,
            K=K,
            thought_size=thought_size,
            embed_dim=embed_dim,
            proj_dim=proj_dim,
            use_cgp=True,
            use_routing=True,
            **kwargs,
        )
    elif model_type in ("pseudo_brain_tier2", "tier2"):
        return NativeSemanticPseudoBrain(
            vocab_size=vocab_size,
            K=kwargs.get("K", K),
            thought_size=64,
            tier="tier2",
            proj_dim=kwargs.get("proj_dim", 4096),
            rank=kwargs.get("rank", 32),
            num_deep_layers=kwargs.get("num_deep_layers", 3),
            use_cgp=True,
            use_routing=True,
            conditional_recurrence=kwargs.get("conditional_recurrence", True),
        )
    elif model_type == "pseudo_brain_no_cgp":
        return NativeSemanticPseudoBrain(
            vocab_size=vocab_size,
            K=K,
            thought_size=thought_size,
            embed_dim=embed_dim,
            proj_dim=proj_dim,
            use_cgp=False,
            use_routing=True,
            **kwargs,
        )
    elif model_type == "pseudo_brain_no_threads":
        # Monolithic single-slot Pseudo-Brain: K=1, thought_size = K * thought_size
        return NativeSemanticPseudoBrain(
            vocab_size=vocab_size,
            K=1,
            thought_size=thought_size * 4,
            embed_dim=embed_dim,
            proj_dim=proj_dim,
            use_cgp=True,
            use_routing=False,
            **kwargs,
        )
    elif model_type == "gru":
        return MonolithicGRULanguageModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=256,
        )
    elif model_type == "ssm":
        return ModernSSMLanguageModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=256,
        )
    elif model_type == "reactive":
        return ReactiveLanguageModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=256,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
