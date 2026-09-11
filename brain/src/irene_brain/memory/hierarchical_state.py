"""Hierarchical Memory Architecture for Pseudo-Brain.

Breaks the 4.0 KB Information Bottleneck (Law 1) to enable 1B-parameter
long-context reasoning over multi-page documents, codebases, and extended dialogues.

Two-Tier Cognitive Memory Hierarchy:
1. Fast Working Memory Tier:
   - K_fast = 16 slots, W_fast = 64 floats (maintains strict 4.0 KB ultra-fast
     60 Hz operational cache for active sentence generation).
2. Consolidated Episodic Memory Tier:
   - K_episodic = 64 to 128 slots, W_episodic = 128 floats (32 KB to 64 KB
     consolidated store for document-level and cross-dialogue long-term retention).
3. Cross-Tier Gated Consolidation:
   - BrainCell state transitions slowly write salient patterns from working slots
     into episodic slots without token replay buffers.
   - Fast working slots query episodic slots via cross-attention or gated routing.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from irene_brain.model.brain_cell import BrainCellCore
from irene_brain.model.sparse_thought_router import SparseThoughtRouter
from irene_brain.model.torch_model import deterministic_thought_identity_codes


@dataclass
class HierarchicalMemoryConfig:
    """Configuration specification for Hierarchical Memory Architecture.

    Supports scalable parameter tiers from Micro (Tier 0) to General (Tier 5 / 1B).
    """

    tier: str = "tier2"
    K_fast: int = 16
    W_fast: int = 64
    K_episodic: int = 64
    W_episodic: int = 128
    consolidation_rate: float = 0.1
    consolidation_threshold: float = 0.2
    retrieval_heads: int = 4
    retrieval_top_k: int = 4
    use_gated_routing: bool = True
    age_decay: float = 0.9995

    @classmethod
    def for_tier(cls, tier: str, **kwargs: Any) -> HierarchicalMemoryConfig:
        """Create tier-specific configuration scaled with parameter budgets."""
        normalized = tier.lower().replace("-", "_")
        if normalized in ("tier0", "micro", "131k"):
            cfg = cls(
                tier="tier0",
                K_fast=8,
                W_fast=32,
                K_episodic=16,
                W_episodic=64,
                retrieval_heads=2,
                retrieval_top_k=2,
            )
        elif normalized in ("tier1", "embedded", "10m"):
            cfg = cls(
                tier="tier1",
                K_fast=16,
                W_fast=64,
                K_episodic=32,
                W_episodic=64,
                retrieval_heads=4,
                retrieval_top_k=4,
            )
        elif normalized in ("tier2", "tier2_35m", "35m", "on_device", "50m"):
            cfg = cls(
                tier="tier2",
                K_fast=16,
                W_fast=64,
                K_episodic=64,
                W_episodic=128,
                retrieval_heads=4,
                retrieval_top_k=4,
            )
        elif normalized in ("tier3", "agentic", "100m", "300m"):
            cfg = cls(
                tier="tier3",
                K_fast=16,
                W_fast=64,
                K_episodic=128,
                W_episodic=128,
                retrieval_heads=4,
                retrieval_top_k=8,
            )
        elif normalized in ("tier5", "tier5_1b", "1b", "general"):
            cfg = cls(
                tier="tier5_1b",
                K_fast=16,
                W_fast=64,
                K_episodic=256,
                W_episodic=128,
                retrieval_heads=8,
                retrieval_top_k=16,
            )
        else:
            cfg = cls(tier=tier)

        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def fast_state_bytes(self, bytes_per_element: int = 4) -> int:
        """Exact memory footprint of the Fast Working Memory tier in bytes.

        Law 1 Contract: K_fast=16, W_fast=64 -> 16 * 64 * 4 = 4,096 bytes (4.0 KB).
        """
        return self.K_fast * self.W_fast * bytes_per_element

    def episodic_state_bytes(self, bytes_per_element: int = 4) -> int:
        """Exact memory footprint of the Consolidated Episodic Memory tier in bytes."""
        return self.K_episodic * self.W_episodic * bytes_per_element

    def total_state_bytes(self, bytes_per_element: int = 4) -> int:
        """Total hierarchical recurrent state memory footprint in bytes."""
        return self.fast_state_bytes(bytes_per_element) + self.episodic_state_bytes(bytes_per_element)


class HierarchicalCognitiveState:
    """Persistent cognitive state across streaming steps for Hierarchical Memory."""

    working_memory: Tensor  # [B, K_fast, W_fast]
    episodic_memory: Tensor  # [B, K_episodic, W_episodic]
    episodic_ages: Tensor  # [B, K_episodic]
    step_count: Tensor  # [B]
    P_t: Optional[Tensor] = None  # [B, K_fast, V] or [B, K_fast, W_fast] fast synaptic trace
    prev_working: Optional[Tensor] = None  # [B, K_fast, W_fast]
    active_thread: Optional[Tensor] = None  # [B]
    working_salience: Optional[Tensor] = None
    config: Optional[Any] = None

    def __init__(
        self,
        working_memory: Optional[Tensor] = None,
        episodic_memory: Optional[Tensor] = None,
        episodic_ages: Optional[Tensor] = None,
        step_count: Optional[Tensor] = None,
        P_t: Optional[Tensor] = None,
        prev_working: Optional[Tensor] = None,
        active_thread: Optional[Tensor] = None,
        *,
        working_thoughts: Optional[Tensor] = None,
        episodic_thoughts: Optional[Tensor] = None,
        working_prev: Optional[Tensor] = None,
        working_salience: Optional[Tensor] = None,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        wm = working_memory if working_memory is not None else working_thoughts
        if wm is None:
            raise ValueError("working_memory or working_thoughts must be provided.")
        self.working_memory = wm

        em = episodic_memory if episodic_memory is not None else episodic_thoughts
        if em is None:
            raise ValueError("episodic_memory or episodic_thoughts must be provided.")
        self.episodic_memory = em

        self.episodic_ages = episodic_ages if episodic_ages is not None else torch.zeros(wm.shape[0], em.shape[1], device=wm.device)
        self.step_count = step_count if step_count is not None else torch.zeros(wm.shape[0], dtype=torch.long, device=wm.device)
        self.P_t = P_t
        self.prev_working = prev_working if prev_working is not None else (working_prev if working_prev is not None else wm.clone())
        self.active_thread = active_thread if active_thread is not None else torch.zeros(wm.shape[0], dtype=torch.long, device=wm.device)
        self.working_salience = working_salience
        self.config = config

    @property
    def working_thoughts(self) -> Tensor:
        return self.working_memory

    @working_thoughts.setter
    def working_thoughts(self, val: Tensor) -> None:
        self.working_memory = val

    @property
    def episodic_thoughts(self) -> Tensor:
        return self.episodic_memory

    @episodic_thoughts.setter
    def episodic_thoughts(self, val: Tensor) -> None:
        self.episodic_memory = val

    @property
    def working_prev(self) -> Optional[Tensor]:
        return self.prev_working

    @working_prev.setter
    def working_prev(self, val: Optional[Tensor]) -> None:
        self.prev_working = val

    def clone(self) -> HierarchicalCognitiveState:
        return HierarchicalCognitiveState(
            working_memory=self.working_memory.clone(),
            episodic_memory=self.episodic_memory.clone(),
            episodic_ages=self.episodic_ages.clone(),
            step_count=self.step_count.clone(),
            P_t=self.P_t.clone() if self.P_t is not None else None,
            prev_working=self.prev_working.clone() if self.prev_working is not None else None,
            active_thread=self.active_thread.clone() if self.active_thread is not None else None,
            working_salience=self.working_salience.clone() if self.working_salience is not None else None,
            config=self.config,
        )

    def detach(self) -> HierarchicalCognitiveState:
        return HierarchicalCognitiveState(
            working_memory=self.working_memory.detach(),
            episodic_memory=self.episodic_memory.detach(),
            episodic_ages=self.episodic_ages.detach(),
            step_count=self.step_count.detach(),
            P_t=self.P_t.detach() if self.P_t is not None else None,
            prev_working=self.prev_working.detach() if self.prev_working is not None else None,
            active_thread=self.active_thread.detach() if self.active_thread is not None else None,
            working_salience=self.working_salience.detach() if self.working_salience is not None else None,
            config=self.config,
        )

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> HierarchicalCognitiveState:
        return HierarchicalCognitiveState(
            working_memory=self.working_memory.to(device=device, dtype=dtype),
            episodic_memory=self.episodic_memory.to(device=device, dtype=dtype),
            episodic_ages=self.episodic_ages.to(device=device, dtype=dtype),
            step_count=self.step_count.to(device=device),
            P_t=self.P_t.to(device=device, dtype=dtype) if self.P_t is not None else None,
            prev_working=self.prev_working.to(device=device, dtype=dtype) if self.prev_working is not None else None,
            active_thread=self.active_thread.to(device=device) if self.active_thread is not None else None,
            working_salience=self.working_salience.to(device=device, dtype=dtype) if self.working_salience is not None else None,
            config=self.config,
        )

    def fast_state_bytes(self) -> int:
        return self.working_memory.numel() * self.working_memory.element_size() // max(1, self.working_memory.shape[0])

    def episodic_state_bytes(self) -> int:
        return self.episodic_memory.numel() * self.episodic_memory.element_size() // max(1, self.episodic_memory.shape[0])

    def total_state_bytes(self) -> int:
        return self.fast_state_bytes() + self.episodic_state_bytes()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state tensors into a clean Python dictionary."""
        return {
            "working_memory": self.working_memory.detach().cpu(),
            "episodic_memory": self.episodic_memory.detach().cpu(),
            "episodic_ages": self.episodic_ages.detach().cpu(),
            "step_count": self.step_count.detach().cpu(),
            "P_t": self.P_t.detach().cpu() if self.P_t is not None else None,
            "prev_working": self.prev_working.detach().cpu() if self.prev_working is not None else None,
            "active_thread": self.active_thread.detach().cpu() if self.active_thread is not None else None,
            "working_salience": self.working_salience.detach().cpu() if self.working_salience is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: Optional[torch.device] = None) -> HierarchicalCognitiveState:
        """Deserialize state from dictionary."""
        dev = device if device is not None else torch.device("cpu")
        return cls(
            working_memory=data["working_memory"].to(dev),
            episodic_memory=data["episodic_memory"].to(dev),
            episodic_ages=data["episodic_ages"].to(dev),
            step_count=data["step_count"].to(dev),
            P_t=data["P_t"].to(dev) if data.get("P_t") is not None else None,
            prev_working=data["prev_working"].to(dev) if data.get("prev_working") is not None else None,
            active_thread=data["active_thread"].to(dev) if data.get("active_thread") is not None else None,
            working_salience=data["working_salience"].to(dev) if data.get("working_salience") is not None else None,
        )

    def to_bytes(self) -> bytes:
        """Serialize state into binary format."""
        buf = io.BytesIO()
        torch.save(self.to_dict(), buf)
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, b: bytes, device: Optional[torch.device] = None) -> HierarchicalCognitiveState:
        """Deserialize state from binary bytes."""
        buf = io.BytesIO(b)
        data = torch.load(buf, weights_only=False)
        return cls.from_dict(data, device=device)


class CrossTierGatedConsolidator(nn.Module):
    """Consolidation module writing salient patterns from Working into Episodic slots.

    Functions without token replay buffers:
    1. Salience gating: Identifies operational patterns in working memory that represent
       salient, consequence-bearing facts or enduring context.
    2. Competitive episodic slot allocation: Matches working patterns against episodic slots using
       semantic key affinity and recency/vacancy bias with straight-through sparse addressing.
    3. Slow continuous-time consolidation: Blends representations slowly into episodic slots
       to avoid catastrophic overwrite while cementing long-term attractors.
    """

    def __init__(
        self,
        *,
        W_fast: int = 64,
        W_episodic: int = 128,
        K_episodic: int = 64,
        consolidation_rate: float = 0.1,
        consolidation_threshold: float = 0.2,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.W_fast = W_fast
        self.W_episodic = W_episodic
        self.K_episodic = K_episodic
        self.consolidation_rate = consolidation_rate
        self.consolidation_threshold = threshold if threshold is not None else consolidation_threshold

        # Write projection: maps working slot features into episodic representation
        self.write_proj = nn.Linear(W_fast, W_episodic, bias=False)

        # Write key projection: maps working pattern to key space
        self.write_key = nn.Linear(W_fast, W_episodic, bias=False)

        # Episodic key projection
        self.episodic_key = nn.Linear(W_episodic, W_episodic, bias=False)

        # Learned write gate / salience scorer
        self.salience_gate = nn.Sequential(
            nn.Linear(W_fast, W_fast // 2),
            nn.SiLU(),
            nn.Linear(W_fast // 2, 1),
            nn.Sigmoid(),
        )

        # Scale parameter for consolidation affinity
        self.scale = 1.0 / math.sqrt(W_episodic)

        # Initialise projections with identity in the shared coordinates
        with torch.no_grad():
            d = min(W_fast, W_episodic)
            self.write_proj.weight.zero_()
            self.write_proj.weight[:d, :d] = torch.eye(d)
            self.write_key.weight.zero_()
            self.write_key.weight[:d, :d] = torch.eye(d)
            self.episodic_key.weight.zero_()
            self.episodic_key.weight[:d, :d] = torch.eye(d)

    def forward(
        self,
        working_memory: Tensor,  # [B, K_fast, W_fast]
        episodic_memory: Tensor,  # [B, K_episodic, W_episodic]
        episodic_ages: Tensor,  # [B, K_episodic]
        salience: Optional[Tensor] = None,  # [B, K_fast, 1] optional external salience
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Consolidate salient working memory patterns into episodic memory.

        Returns:
            updated_episodic_memory: [B, K_episodic, W_episodic]
            updated_episodic_ages: [B, K_episodic]
            slot_write_gates: [B, K_episodic, 1]
        """
        B, K_f, W_f = working_memory.shape
        _, K_e, W_e = episodic_memory.shape

        if salience is None:
            raw_salience = self.salience_gate(working_memory)  # [B, K_f, 1]
        else:
            raw_salience = salience

        # Salience thresholding
        active_salience = torch.relu(raw_salience - self.consolidation_threshold)  # [B, K_f, 1]
        active_mask = (active_salience.squeeze(-1) > 0.0)  # [B, K_f]

        # Fast path if no working slots exceed threshold
        if not active_mask.any():
            updated_ages = episodic_ages + 1.0
            zero_gates = torch.zeros(B, K_e, 1, device=working_memory.device, dtype=working_memory.dtype)
            return episodic_memory, updated_ages, zero_gates

        # Project working slots into episodic value and key space
        write_vals = self.write_proj(working_memory)  # [B, K_f, W_e]
        write_keys = self.write_key(working_memory)  # [B, K_f, W_e]
        epi_keys = self.episodic_key(episodic_memory)  # [B, K_e, W_e]

        # Normalized cosine affinity for scale-invariant addressing
        w_keys_norm = F.normalize(write_keys, dim=-1)
        e_keys_norm = F.normalize(epi_keys, dim=-1)
        affinity = torch.bmm(w_keys_norm, e_keys_norm.transpose(1, 2))  # [B, K_f, K_e]

        # Vacancy & age bias: older slots receive higher allocation priority
        age_bonus = torch.tanh(episodic_ages / 20.0).unsqueeze(1)  # [B, 1, K_e]
        scores = affinity + 0.5 * age_bonus  # [B, K_f, K_e]

        # Competitive Top-1 Slot Assignment with straight-through gradient
        best_slots = torch.argmax(scores, dim=-1)  # [B, K_f]
        one_hot = F.one_hot(best_slots, num_classes=K_e).to(dtype=scores.dtype)  # [B, K_f, K_e]
        soft = F.softmax(scores, dim=-1)
        write_dist = (one_hot - soft).detach() + soft  # [B, K_f, K_e]

        # Mask out inactive working slots: [B, K_f, K_e]
        mask_expanded = active_mask.unsqueeze(-1).to(dtype=write_dist.dtype)
        gated_dist = write_dist * mask_expanded  # [B, K_f, K_e]

        # Aggregate writes per episodic slot: [B, K_e, K_f] x [B, K_f, W_e] -> [B, K_e, W_e]
        write_weights = gated_dist.transpose(1, 2)  # [B, K_e, K_f]
        salient_writes = active_salience * write_vals  # [B, K_f, W_e]
        incoming_write = torch.bmm(write_weights, salient_writes)  # [B, K_e, W_e]

        # Slot write gate: [B, K_e, 1]
        slot_write_gate = torch.bmm(write_weights, active_salience).clamp(0.0, 1.0)

        # Normalize incoming writes by total weight if multiple slots targeted same episodic slot
        sum_weights = write_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, K_e, 1]
        incoming_norm = incoming_write / sum_weights

        # Dual-timescale update: instantaneous allocation for vacant slots (age > 20),
        # slow continuous consolidation for occupied slots
        is_vacant = (episodic_ages > 20.0).unsqueeze(-1)
        slow_alpha = self.consolidation_rate * slot_write_gate
        alpha = torch.where(is_vacant & (slot_write_gate > 0.0), torch.ones_like(slot_write_gate), slow_alpha)
        updated_episodic = (1.0 - alpha) * episodic_memory + alpha * incoming_norm

        # Reset age on written slots, advance on unwritten slots
        write_mask = slot_write_gate.squeeze(-1)  # [B, K_e]
        updated_ages = (1.0 - write_mask) * (episodic_ages + 1.0)

        return updated_episodic, updated_ages, slot_write_gate


class CrossTierGatedRetriever(nn.Module):
    """Retrieval module querying Consolidated Episodic Memory into Fast Working Memory.

    Allows fast working slots to selectively attend to long-past facts without
    storing them continuously in the 4.0 KB fast operational cache.
    """

    def __init__(
        self,
        *,
        W_fast: int = 64,
        W_episodic: int = 128,
        heads: int = 4,
        num_heads: Optional[int] = None,
        top_k: int = 4,
        use_gated_routing: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.W_fast = W_fast
        self.W_episodic = W_episodic
        self.heads = num_heads if num_heads is not None else heads
        self.top_k = top_k
        self.use_gated_routing = use_gated_routing

        attn_dim = W_fast
        self.attn_dim = attn_dim
        self.q_proj = nn.Linear(W_fast, attn_dim, bias=False)
        self.k_proj = nn.Linear(W_episodic, attn_dim, bias=False)
        self.v_proj = nn.Linear(W_episodic, W_fast, bias=False)

        # Read gate controlling how much retrieved context modulates working memory
        self.read_gate = nn.Sequential(
            nn.Linear(W_fast * 2, W_fast),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.read_gate[0].bias, 0.0)

        with torch.no_grad():
            d = min(W_fast, W_episodic)
            self.q_proj.weight.zero_()
            self.q_proj.weight[:d, :d] = torch.eye(d)
            self.k_proj.weight.zero_()
            self.k_proj.weight[:d, :d] = torch.eye(d)
            self.v_proj.weight.zero_()
            self.v_proj.weight[:d, :d] = torch.eye(d)

    def forward(
        self,
        working_memory: Tensor,  # [B, K_fast, W_fast]
        episodic_memory: Tensor,  # [B, K_episodic, W_episodic]
        query_override: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Retrieve relevant context from episodic memory to augment working memory.

        Returns:
            augmented_working: [B, K_fast, W_fast]
            retrieved_context: [B, K_fast, W_fast]
            retrieval_weights: [B, K_fast, K_episodic]
        """
        B, K_f, W_f = working_memory.shape
        _, K_e, W_e = episodic_memory.shape

        q_in = query_override if query_override is not None else working_memory
        Q = self.q_proj(q_in)  # [B, K_f, attn_dim]
        K = self.k_proj(episodic_memory)  # [B, K_e, attn_dim]
        V = self.v_proj(episodic_memory)  # [B, K_e, W_f]

        # Scaled cosine-matching attention scores
        Q_norm = F.normalize(Q, dim=-1)
        K_norm = F.normalize(K, dim=-1)
        scores = torch.bmm(Q_norm, K_norm.transpose(1, 2)) / 0.1  # [B, K_f, K_e]

        if self.use_gated_routing and self.top_k < K_e:
            topk_vals, topk_idx = torch.topk(scores, k=self.top_k, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)  # [B, K_f, top_k]

            # Gather values: [B, K_f, top_k, W_f]
            V_expanded = V.unsqueeze(1).expand(-1, K_f, -1, -1)
            idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, W_f)
            gathered_V = torch.gather(V_expanded, 2, idx_expanded)
            retrieved = (topk_weights.unsqueeze(-1) * gathered_V).sum(dim=2)  # [B, K_f, W_f]

            weights = torch.zeros_like(scores)
            weights.scatter_(2, topk_idx, topk_weights)
        else:
            weights = F.softmax(scores, dim=-1)
            retrieved = torch.bmm(weights, V)  # [B, K_f, W_f]

        # Dynamic read gating
        gate_input = torch.cat([working_memory, retrieved], dim=-1)
        gate = self.read_gate(gate_input)  # [B, K_f, W_f]

        augmented = working_memory + gate * retrieved
        return augmented, retrieved, weights


class HierarchicalMemorySystem(nn.Module):
    """Integrated Hierarchical Memory Architecture coordinator.

    Combines:
    - Fast Working Memory Tier (Law 1 4.0 KB compliant)
    - Consolidated Episodic Memory Tier (32 KB - 128 KB document retention)
    - Cross-Tier Gated Consolidation (hidden pattern slow transfer)
    - Cross-Tier Gated Retrieval (cross-attention factual recall)
    """

    def __init__(self, config: Optional[HierarchicalMemoryConfig] = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            config = HierarchicalMemoryConfig.for_tier("tier2", **kwargs)
        self.config = config

        self.consolidator = CrossTierGatedConsolidator(
            W_fast=config.W_fast,
            W_episodic=config.W_episodic,
            K_episodic=config.K_episodic,
            consolidation_rate=config.consolidation_rate,
            consolidation_threshold=config.consolidation_threshold,
        )

        self.retriever = CrossTierGatedRetriever(
            W_fast=config.W_fast,
            W_episodic=config.W_episodic,
            heads=config.retrieval_heads,
            top_k=config.retrieval_top_k,
            use_gated_routing=config.use_gated_routing,
        )

        # Deterministic identity codes for working slots
        self.register_buffer(
            "working_identities",
            deterministic_thought_identity_codes(thoughtlets=config.K_fast, width=config.W_fast),
            persistent=False,
        )

        # Deterministic identity codes for episodic slots
        self.register_buffer(
            "episodic_identities",
            deterministic_thought_identity_codes(thoughtlets=config.K_episodic, width=config.W_episodic),
            persistent=False,
        )

    def initial_state(self, batch_size: int, device: torch.device) -> HierarchicalCognitiveState:
        """Create fresh initialized hierarchical state."""
        w_id = self.working_identities.to(device=device)
        e_id = self.episodic_identities.to(device=device)

        working = w_id.unsqueeze(0).expand(batch_size, -1, -1).clone()
        episodic = e_id.unsqueeze(0).expand(batch_size, -1, -1).clone()
        ages = torch.full((batch_size, self.config.K_episodic), 100.0, device=device)
        step_count = torch.zeros(batch_size, dtype=torch.long, device=device)

        return HierarchicalCognitiveState(
            working_memory=working,
            episodic_memory=episodic,
            episodic_ages=ages,
            step_count=step_count,
            prev_working=working.clone(),
            active_thread=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def retrieve(
        self,
        state: HierarchicalCognitiveState,
        query: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Retrieve long-term facts from episodic memory into working context."""
        return self.retriever(
            working_memory=state.working_memory,
            episodic_memory=state.episodic_memory,
            query_override=query,
        )

    def consolidate(
        self,
        state: HierarchicalCognitiveState,
        salience: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Consolidate salient working thoughts into episodic memory."""
        return self.consolidator(
            working_memory=state.working_memory,
            episodic_memory=state.episodic_memory,
            episodic_ages=state.episodic_ages,
            salience=salience,
        )

    def step(
        self,
        working_thoughts: Tensor,  # [B, K_fast, W_fast]
        state: HierarchicalCognitiveState,
        salience: Optional[Tensor] = None,
    ) -> Tuple[Tensor, HierarchicalCognitiveState, Dict[str, Any]]:
        """Single hierarchical memory step.

        1. Consolidates salient patterns from working thoughts into episodic store.
        2. Retrieves relevant episodic context to augment working thoughts.
        3. Updates step count and state traces.
        """
        # 1. Consolidate into episodic memory
        new_epi, new_ages, write_gates = self.consolidator(
            working_memory=working_thoughts,
            episodic_memory=state.episodic_memory,
            episodic_ages=state.episodic_ages,
            salience=salience,
        )

        # 2. Retrieve episodic context
        augmented_working, retrieved_context, retrieval_weights = self.retriever(
            working_memory=working_thoughts,
            episodic_memory=new_epi,
        )

        next_state = HierarchicalCognitiveState(
            working_memory=augmented_working,
            episodic_memory=new_epi,
            episodic_ages=new_ages,
            step_count=state.step_count + 1,
            P_t=state.P_t,
            prev_working=working_thoughts.detach(),
            active_thread=state.active_thread,
        )

        telemetry = {
            "write_gates": write_gates,
            "retrieval_weights": retrieval_weights,
            "retrieved_context": retrieved_context,
            "mean_write_gate": write_gates.mean().detach(),
            "active_episodic_slots": (new_ages < 50.0).float().sum(dim=-1).mean().detach(),
        }
        return augmented_working, next_state, telemetry

    def footprint_report(self) -> Dict[str, Any]:
        """Report exact state footprints, parameter counts, and cache compliance."""
        fast_bytes = self.config.fast_state_bytes()
        epi_bytes = self.config.episodic_state_bytes()
        total_bytes = self.config.total_state_bytes()

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "tier": self.config.tier,
            "K_fast": self.config.K_fast,
            "W_fast": self.config.W_fast,
            "fast_state_bytes": fast_bytes,
            "fast_state_kb": fast_bytes / 1024.0,
            "l1_cache_compliant": fast_bytes <= 4096,
            "K_episodic": self.config.K_episodic,
            "W_episodic": self.config.W_episodic,
            "episodic_state_bytes": epi_bytes,
            "episodic_state_kb": epi_bytes / 1024.0,
            "total_state_bytes": total_bytes,
            "total_state_kb": total_bytes / 1024.0,
            "memory_expansion_ratio": total_bytes / fast_bytes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }


class HierarchicalSemanticPseudoBrain(nn.Module):
    """End-to-End Pseudo-Brain Language & Reasoning Model with Hierarchical Memory.

    Solves the 4.0 KB Information Bottleneck:
    - Maintains strict 4.0 KB fast operational state during 60 Hz token processing.
    - Consolidates episodic knowledge across 500+ tokens without attractor collapse.
    """

    def __init__(
        self,
        vocab_size: int = 344,
        config: Optional[HierarchicalMemoryConfig] = None,
        embed_dim: int = 64,
        proj_dim: int = 128,
        tier: Optional[str] = None,
        rank: Optional[int] = None,
        num_deep_layers: int = 3,
        use_cgp: bool = True,
        use_routing: bool = True,
        routed_neighbors: int = 2,
    ) -> None:
        super().__init__()
        if config is None:
            config = HierarchicalMemoryConfig.for_tier(tier if tier is not None else "tier2")
        self.config = config
        self.vocab_size = vocab_size

        if config.tier in ("1b", "tier5_1b") and proj_dim == 128:
            proj_dim = 6144
            embed_dim = 256
            rank = 32 if rank is None else rank
            num_deep_layers = 24 if num_deep_layers == 3 else num_deep_layers
        elif config.tier == "tier2" and proj_dim == 128:
            proj_dim = 4096
            embed_dim = 128
            rank = 32 if rank is None else rank

        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.use_cgp = use_cgp
        self.use_routing = use_routing

        # 1. Semantic Embedding & Projection
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, proj_dim)

        # 2. Shared Recurrent Core across K_fast working slots
        cell_tier = config.tier if (
            (config.tier == "tier2" and proj_dim == 4096)
            or (config.tier in ("1b", "tier5_1b", "tier3") and proj_dim == 6144)
        ) else None
        self.brain_cell = BrainCellCore(
            input_size=proj_dim,
            thought_size=config.W_fast,
            rank=rank,
            proj_dim=proj_dim,
            tier=cell_tier,
            num_deep_layers=num_deep_layers,
        )

        # 3. Cognitive Input Gating (CIG)
        self.cig_gate = nn.Sequential(
            nn.Linear(proj_dim + config.W_fast, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.cig_gate[0].bias, 1.0)

        # 4. Sparse Thought Router for working memory
        if use_routing and config.K_fast > 1:
            self.router = SparseThoughtRouter(
                width=config.W_fast,
                routed_neighbors=min(routed_neighbors, max(1, config.K_fast - 1)),
                dense_routing=False,
            )
        else:
            self.router = None

        # 5. Hierarchical Memory Coordinator
        self.memory_system = HierarchicalMemorySystem(config=config)

        # 6. Readout Head
        head_hidden = max(64, min(proj_dim // 2, 512))
        self.slot_head = nn.Sequential(
            nn.Linear(config.W_fast, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, vocab_size),
        )

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def initial_state(self, batch_size: int, device: torch.device) -> HierarchicalCognitiveState:
        return self.memory_system.initial_state(batch_size, device)

    def step(
        self,
        token_ids: Tensor,  # [B]
        state: HierarchicalCognitiveState,
        thread_ids: Optional[Tensor] = None,
        allow_routing: bool = False,
    ) -> Tuple[Tensor, HierarchicalCognitiveState]:
        """Single streaming token step with hierarchical memory integration."""
        B = token_ids.shape[0]
        dev = token_ids.device

        # Update active thread
        if thread_ids is not None:
            state.active_thread = thread_ids.clamp(0, self.config.K_fast - 1)
        else:
            is_thread_tok = (token_ids >= 9) & (token_ids < 9 + self.config.K_fast)
            if is_thread_tok.any():
                new_tids = (token_ids - 9).clamp(0, self.config.K_fast - 1)
                state.active_thread = torch.where(is_thread_tok, new_tids, state.active_thread)

        # 1. Embed & Project
        emb = self.embedding(token_ids)  # [B, embed_dim]
        x_proj = self.proj(emb)  # [B, proj_dim]

        # 2. Expand across K_fast working slots
        x_exp = x_proj.unsqueeze(1).expand(-1, self.config.K_fast, -1)  # [B, K_fast, proj_dim]

        # 3. Retrieve long-term context from episodic memory
        retrieved_working, _, _ = self.memory_system.retriever(
            working_memory=state.working_memory,
            episodic_memory=state.episodic_memory,
        )

        # 4. Cognitive Input Gating with T=0.5 sharpening
        gate_in = torch.cat([x_exp, retrieved_working], dim=-1)
        raw_gate = self.cig_gate(gate_in)
        clamped_gate = raw_gate.clamp(1e-4, 1.0 - 1e-4)
        logit_gate = torch.log(clamped_gate / (1.0 - clamped_gate))
        salience = torch.sigmoid(logit_gate / 0.5)

        # 5. Fast Recurrent Core Update on working slots
        t_flat = retrieved_working.reshape(B * self.config.K_fast, self.config.W_fast)
        x_flat = x_exp.reshape(B * self.config.K_fast, -1)
        new_t = self.brain_cell(t_flat, x_flat).reshape(B, self.config.K_fast, self.config.W_fast)

        # 6. Routing across working slots
        if self.use_routing and self.router is not None and allow_routing:
            messages, _ = self.router(new_t)
            t_candidate = new_t + messages
        else:
            t_candidate = new_t

        # 7. Gated State Update for Working Memory
        next_working = (1.0 - salience) * retrieved_working + salience * t_candidate

        # 8. Cross-Tier Gated Consolidation into Episodic Memory
        new_episodic, new_ages, _ = self.memory_system.consolidator(
            working_memory=next_working,
            episodic_memory=state.episodic_memory,
            episodic_ages=state.episodic_ages,
            salience=salience,
        )

        # 9. Readout from active thread working slot
        batch_idx = torch.arange(B, device=dev)
        clamped_active = state.active_thread.clamp(0, self.config.K_fast - 1)
        queried_slot = next_working[batch_idx, clamped_active]  # [B, W_fast]
        logits = self.slot_head(queried_slot)  # [B, V]

        next_state = HierarchicalCognitiveState(
            working_memory=next_working,
            episodic_memory=new_episodic,
            episodic_ages=new_ages,
            step_count=state.step_count + 1,
            P_t=state.P_t,
            prev_working=next_working.detach(),
            active_thread=state.active_thread,
        )

        return logits, next_state

    def forward(
        self,
        token_seq: Tensor,  # [B, T]
        thread_seq: Optional[Tensor] = None,
        allow_routing: bool = False,
    ) -> Tensor:
        """Sequential unroll over sequence. Returns logits [B, T, V]."""
        B, T = token_seq.shape
        dev = token_seq.device
        state = self.initial_state(B, dev)
        logits_list = []

        for t in range(T):
            tid = thread_seq[:, t] if thread_seq is not None else None
            logits_t, state = self.step(token_seq[:, t], state, thread_ids=tid, allow_routing=allow_routing)
            logits_list.append(logits_t)

        return torch.stack(logits_list, dim=1)


# Aliases for cross-module compatibility
CrossTierConsolidationGate = CrossTierGatedConsolidator
CrossTierRetrievalModule = CrossTierGatedRetriever
FastWorkingMemoryTier = HierarchicalMemoryConfig
ConsolidatedEpisodicTier = HierarchicalMemoryConfig

__all__ = [
    "CrossTierGatedConsolidator",
    "CrossTierGatedRetriever",
    "CrossTierConsolidationGate",
    "CrossTierRetrievalModule",
    "FastWorkingMemoryTier",
    "ConsolidatedEpisodicTier",
    "HierarchicalCognitiveState",
    "HierarchicalMemoryConfig",
    "HierarchicalMemorySystem",
    "HierarchicalSemanticPseudoBrain",
]

