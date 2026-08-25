"""BrainCell V2 — shared-weight thoughtlet update.

Receives: old_thought, belief, prediction_error, retrieved_memory, session_context
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import CoreV2Config
from .state import PredictionErrorState


class BrainCellV2(nn.Module):
    """Single thoughtlet update — weights shared across all K slots and C cycles."""

    def __init__(self, config: CoreV2Config):
        super().__init__()
        self.config = config
        W = config.width
        hidden = config.braincell_hidden

        # Input: thought + belief + pe + memory + session (each W) + has_pe flag
        in_dim = W * 5 + 1
        self.in_proj = nn.Linear(in_dim, hidden)
        self.gate_preserve = nn.Linear(hidden, W)
        self.gate_accept = nn.Linear(hidden, W)
        self.gate_revise = nn.Linear(hidden, W)
        self.gate_retain = nn.Linear(hidden, W)
        self.gate_react = nn.Linear(hidden, W)
        self.out_proj = nn.Linear(hidden, W)

        # Fast plasticity hook (disabled by default)
        self.fast_plasticity_enabled = config.fast_plasticity_enabled
        if self.fast_plasticity_enabled:
            rank = config.fast_plasticity_rank
            self.fast_U = nn.Parameter(torch.zeros(W, rank))
            self.fast_V = nn.Parameter(torch.zeros(rank, W))
            nn.init.xavier_uniform_(self.fast_U)
            nn.init.zeros_(self.fast_V)

    def forward(
        self,
        thought: torch.Tensor,                    # [B, W] or [B, K, W]
        belief: torch.Tensor,                     # [B, W]
        prediction_error: Optional[PredictionErrorState] = None,
        retrieved_memory: Optional[torch.Tensor] = None,  # [B, W] or [B, K, W]
        session_latent: Optional[torch.Tensor] = None,    # [B, W]
    ) -> torch.Tensor:
        """Single BrainCell update step."""
        # Handle [B, K, W] by flattening batch*K, then restore
        has_k = thought.dim() == 3
        if has_k:
            B, K, W = thought.shape
            thought_flat = thought.reshape(B * K, W)
            belief = belief.unsqueeze(1).expand(-1, K, -1).reshape(B * K, W)
            if retrieved_memory is not None:
                if retrieved_memory.dim() == 3:
                    retrieved_memory = retrieved_memory.reshape(B * K, W)
                else:
                    retrieved_memory = retrieved_memory.unsqueeze(1).expand(-1, K, -1).reshape(B * K, W)
            else:
                retrieved_memory = torch.zeros(B * K, W, device=thought.device)
            if session_latent is not None:
                session_latent = session_latent.unsqueeze(1).expand(-1, K, -1).reshape(B * K, W)
            else:
                session_latent = torch.zeros(B * K, W, device=thought.device)

            # Prediction error summary
            if prediction_error is not None and prediction_error.latent_error is not None:
                pe = (
                    prediction_error.cognitive_error
                    if prediction_error.cognitive_error is not None
                    else prediction_error.latent_error
                )
                if pe.dim() == 3:
                    pe = pe.reshape(B * K, W)
                elif pe.dim() == 2 and pe.shape[0] == B:
                    # Belief-level error (e.g. first tick, no pending prediction):
                    # broadcast the same error summary to every slot
                    pe = pe.unsqueeze(1).expand(-1, K, -1).reshape(B * K, W)
                else:
                    pe = torch.zeros(B * K, W, device=thought.device)
                has_pe = torch.ones(B * K, 1, device=thought.device)
            else:
                pe = torch.zeros(B * K, W, device=thought.device)
                has_pe = torch.zeros(B * K, 1, device=thought.device)
        else:
            B, W = thought.shape
            thought_flat = thought
            if retrieved_memory is None:
                retrieved_memory = torch.zeros(B, W, device=thought.device)
            if session_latent is None:
                session_latent = torch.zeros(B, W, device=thought.device)
            if prediction_error is not None and prediction_error.latent_error is not None:
                pe = (
                    prediction_error.cognitive_error
                    if prediction_error.cognitive_error is not None
                    else prediction_error.latent_error
                )
                if pe.dim() == 3:
                    pe = pe.mean(dim=1)
                has_pe = torch.ones(B, 1, device=thought.device)
            else:
                pe = torch.zeros(B, W, device=thought.device)
                has_pe = torch.zeros(B, 1, device=thought.device)

        # Concatenate inputs
        x = torch.cat([
            thought_flat,           # W
            belief,                 # W
            pe,                     # W
            retrieved_memory,       # W
            session_latent,         # W
            has_pe,                 # 1
        ], dim=-1)  # [B*K, 5*W + 1] or [B, 5*W + 1]

        h = F.relu(self.in_proj(x))

        # Gated residual update components
        g_preserve = torch.sigmoid(self.gate_preserve(h))
        g_accept = torch.sigmoid(self.gate_accept(h))
        g_revise = torch.sigmoid(self.gate_revise(h))
        g_retain = torch.sigmoid(self.gate_retain(h))
        g_react = torch.sigmoid(self.gate_react(h))

        # Candidate update
        candidate = self.out_proj(h)

        if self.config.braincell_dynamics == "legacy_additive_v0":
            # Historical V2.0/V2.0b path.  Its independently added gates can
            # produce up to roughly 2*thought + 2*candidate per cycle and is
            # retained only for exact failure reproduction.
            new_thought = (
                g_preserve * thought_flat
                + g_accept * candidate
                + g_revise * (-thought_flat)
                + g_retain * thought_flat
                + g_react * candidate
            )
        else:
            # Three semantic proposals compete through one normalized gate.
            # The convex mixture prevents vote-like coefficient inflation;
            # per-sample layer normalization bounds recurrent state scale at
            # every cognitive cycle without batch-dependent statistics.
            gate_scores = torch.stack([
                g_preserve + g_retain,
                g_accept + g_react,
                g_revise,
            ], dim=-1)
            mixture = F.softmax(gate_scores, dim=-1)
            proposals = torch.stack([
                thought_flat,
                torch.tanh(candidate),
                torch.tanh(candidate - thought_flat),
            ], dim=-1)
            new_thought = (mixture * proposals).sum(dim=-1)
            new_thought = F.layer_norm(new_thought, (W,), eps=1e-5)

        # Fast plasticity residual (disabled by default)
        if self.fast_plasticity_enabled and self.training:
            fast_delta = thought_flat @ self.fast_V.T @ self.fast_U.T
            new_thought = new_thought + 0.01 * fast_delta

        # Restore [B, K, W] if needed
        if has_k:
            new_thought = new_thought.reshape(B, K, W)

        return new_thought
