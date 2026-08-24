"""Thought Field V2 — K thoughtlets with vectorized thought-to-thought communication.

Permutation equivariant: whole-slot permutation leaves outputs unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import CoreV2Config
from .brain_cell import BrainCellV2
from .state import PredictionErrorState


class ThoughtFieldV2(nn.Module):
    """K exchangeable thoughtlets with shared-weight updates and attention."""

    def __init__(self, config: CoreV2Config):
        super().__init__()
        self.config = config
        self.K = config.thoughtlets
        self.C = config.cycles
        W = config.width

        self.brain_cell = BrainCellV2(config)

        # Vectorized attention for thought-to-thought communication
        self.attention = nn.MultiheadAttention(
            embed_dim=W,
            num_heads=config.attention_heads,
            batch_first=True,
        )

        # Thought initialization
        self.seed_proj = nn.Linear(W * 3, W)  # belief + session + noise

    def init_thoughts(
        self,
        belief: torch.Tensor,           # [B, W]
        session_latent: torch.Tensor,   # [B, W]
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Initialize K thoughtlets from belief + session + learned seed distribution."""
        B, W = belief.shape
        device = belief.device

        if noise is None:
            noise = torch.randn(B, self.K, W, device=device)

        # Project combined context + noise per slot
        ctx = torch.cat([
            belief.unsqueeze(1).expand(-1, self.K, -1),
            session_latent.unsqueeze(1).expand(-1, self.K, -1),
            noise,
        ], dim=-1)  # [B, K, 3W]

        thoughts = self.seed_proj(ctx)  # [B, K, W]
        return thoughts

    def forward(
        self,
        thoughts: torch.Tensor,                    # [B, K, W]
        belief: torch.Tensor,                      # [B, W]
        prediction_error: Optional[PredictionErrorState] = None,
        retrieved_memory: Optional[torch.Tensor] = None,  # [B, K, W] or [B, W]
        session_latent: Optional[torch.Tensor] = None,    # [B, W]
    ) -> torch.Tensor:
        """Run C cycles of BrainCell update + attention."""
        B, K, W = thoughts.shape

        for _ in range(self.C):
            # A. Private shared-weight BrainCell update per thoughtlet
            thoughts = self.brain_cell(
                thoughts, belief, prediction_error, retrieved_memory, session_latent
            )

            # B. Vectorized thought-to-thought communication
            # Self-attention over K slots — permutation equivariant
            attn_out, _ = self.attention(thoughts, thoughts, thoughts)
            thoughts = thoughts + attn_out  # residual

        return thoughts