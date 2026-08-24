"""Episodic Memory V2 — bounded differentiable session-state memory.

Capacity 256, retrieval via learned similarity.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import CoreV2Config
from .state import EpisodicMemoryV2 as EpisodicMemoryState, EpisodicMemoryEntry


class EpisodicMemoryV2(nn.Module):
    """Learnable episodic memory with write policy and retrieval."""

    def __init__(self, config: CoreV2Config):
        super().__init__()
        self.config = config
        W = config.width
        self.capacity = config.episodic_capacity
        self.retrieval_k = config.retrieval_k

        # Write priority network
        self.write_net = nn.Sequential(
            nn.Linear(W * 4 + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Query projection for retrieval
        self.query_proj = nn.Linear(W * 3, W)

        # Memory context projection
        self.memory_out = nn.Linear(W, W)

    def compute_write_priority(
        self,
        surprise: torch.Tensor,          # [B, 1]
        prediction_error: torch.Tensor,  # [B, W] or [B, K, W]
        reward: torch.Tensor,            # [B, 1]
        hazard: torch.Tensor,            # [B, 1]
        belief: torch.Tensor,            # [B, W]
        session_latent: torch.Tensor,    # [B, W]
    ) -> torch.Tensor:
        """Compute learned write priority [B, 1]."""
        B = belief.shape[0]
        device = belief.device

        # Flatten prediction error if needed
        if prediction_error.dim() == 3:
            pe = prediction_error.mean(dim=1)
        else:
            pe = prediction_error

        x = torch.cat([
            belief,
            session_latent,
            pe,
            surprise,
            reward,
            hazard,
        ], dim=-1)

        return self.write_net(x)

    def should_write(
        self,
        priority: torch.Tensor,
        sparsity_penalty: float = 0.1,
    ) -> torch.Tensor:
        """Bernoulli write decision with sparsity regularization."""
        # During training: straight-through estimator
        # During eval: threshold
        if self.training:
            # Sample with Gumbel-Softmax / straight-through
            u = torch.rand_like(priority)
            logits = torch.log(priority + 1e-8) - torch.log(1 - priority + 1e-8)
            gate = torch.sigmoid(logits + torch.log(u + 1e-8) - torch.log(1 - u + 1e-8))
            # Sparsity loss will be computed externally
            return gate
        else:
            return (priority > 0.5).float()

    def write_entry(
        self,
        memory_state: EpisodicMemoryState,
        key: torch.Tensor,           # [B, W]
        value: torch.Tensor,         # [B, W]
        action: torch.Tensor,        # [B]
        predicted_consequence: dict,
        observed_consequence: dict,
        prediction_error: dict,
        reward: torch.Tensor,
        hazard: torch.Tensor,
        confidence: torch.Tensor,
        trial_index: int,
        priority: torch.Tensor,
    ):
        """Write entries to memory state.

        Uses learned priority for replacement policy when full.
        """
        B = key.shape[0]
        for b in range(B):
            # Determine if we should write this batch item
            write_prob = priority[b].item() if priority.dim() > 0 else priority.item()
            if write_prob < 0.5:
                continue

            entry = EpisodicMemoryEntry(
                key=key[b].detach(),
                value=value[b].detach(),
                action=action[b].detach(),
                predicted_consequence=predicted_consequence,
                observed_consequence=observed_consequence,
                prediction_error=prediction_error,
                reward=reward[b].detach(),
                hazard=hazard[b].detach(),
                confidence=confidence[b].detach(),
                trial_index=trial_index,
                timestamp=len(memory_state.entries),
            )

            if len(memory_state.entries) < self.capacity:
                memory_state.entries.append(entry)
            else:
                # Replace lowest priority (simple FIFO for now)
                idx = memory_state.write_pointer % self.capacity
                memory_state.entries[idx] = entry
                memory_state.write_pointer += 1

    def retrieve(
        self,
        memory_state: EpisodicMemoryState,
        query: torch.Tensor,          # [B, W]
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """Retrieve top-k memories, return context [B, k, W]."""
        if not memory_state.entries:
            B = query.shape[0]
            return torch.zeros(B, self.retrieval_k, self.config.width,
                               device=query.device, dtype=query.dtype)

        k = min(k or self.retrieval_k, len(memory_state.entries))
        B = query.shape[0]
        device = query.device

        # Project query
        q = self.query_proj(query)  # [B, W]

        # Stack keys
        keys = torch.stack([e.key.to(device) for e in memory_state.entries])  # [N, W]
        values = torch.stack([e.value.to(device) for e in memory_state.entries])  # [N, W]

        # Cosine similarity
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(keys, dim=-1)
        sim = q_norm @ k_norm.T  # [B, N]

        # Top-k
        _, topk_idx = sim.topk(k, dim=-1)

        # Gather values
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, k)
        retrieved = values[topk_idx]  # [B, k, W]

        # Project output
        return self.memory_out(retrieved)

    def retrieve_with_weights(
        self,
        memory_state: EpisodicMemoryState,
        query: torch.Tensor,
        k: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve with attention weights for diagnostics."""
        if not memory_state.entries:
            B = query.shape[0]
            return (torch.zeros(B, self.retrieval_k, self.config.width,
                                device=query.device, dtype=query.dtype),
                    torch.zeros(B, self.retrieval_k, device=query.device))

        k = min(k or self.retrieval_k, len(memory_state.entries))
        B = query.shape[0]
        device = query.device

        q = self.query_proj(query)
        keys = torch.stack([e.key.to(device) for e in memory_state.entries])
        values = torch.stack([e.value.to(device) for e in memory_state.entries])

        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(keys, dim=-1)
        sim = q_norm @ k_norm.T
        weights = F.softmax(sim, dim=-1)

        _, topk_idx = weights.topk(k, dim=-1)
        topk_weights = torch.gather(weights, 1, topk_idx)

        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, k)
        retrieved = values[topk_idx]

        return self.memory_out(retrieved), topk_weights