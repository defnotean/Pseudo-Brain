"""Belief Updater V2 — persistent world belief.

b_t = BeliefUpdater(b_{t-1}, z_t, e_t, m_t, s_t, a_{t-1})
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import CoreV2Config
from .state import PredictionErrorState


class BeliefUpdaterV2(nn.Module):
    """Gated recurrent update for world belief.

    belief: world-state context, NOT action selector.
    """

    def __init__(self, config: CoreV2Config):
        super().__init__()
        self.config = config
        W = config.width
        hidden = config.belief_hidden

        # Inputs: old_belief + observation + pe + memory + session (each W) + action_onehot(5) + has_error(1)
        in_dim = W * 5 + 6
        self.in_proj = nn.Linear(in_dim, hidden)
        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, W),
        )

    def forward(
        self,
        old_belief: torch.Tensor,              # [B, W]
        observation: torch.Tensor,              # [B, W] (encoded z_t)
        prediction_error: Optional[PredictionErrorState] = None,
        retrieved_memory: Optional[torch.Tensor] = None,  # [B, W] or [B, K, W]
        session_latent: Optional[torch.Tensor] = None,    # [B, W]
        prev_action: Optional[torch.Tensor] = None,       # [B] or [B, 5] (one-hot)
    ) -> torch.Tensor:
        B, W = old_belief.shape
        device = old_belief.device

        # Flatten memory if [B, K, W]
        if retrieved_memory is not None and retrieved_memory.dim() == 3:
            retrieved_memory = retrieved_memory.mean(dim=1)  # [B, W]
        elif retrieved_memory is None:
            retrieved_memory = torch.zeros(B, W, device=device)

        if session_latent is None:
            session_latent = torch.zeros(B, W, device=device)

        # Prediction error summary
        if prediction_error is not None and prediction_error.latent_error is not None:
            pe = prediction_error.latent_error
            if pe.dim() == 3:
                pe = pe.mean(dim=1)  # [B, W]
            has_error = torch.ones(B, 1, device=device)
        else:
            pe = torch.zeros(B, W, device=device)
            has_error = torch.zeros(B, 1, device=device)

        # Previous action one-hot
        if prev_action is not None:
            if prev_action.dim() == 1:
                action_oh = F.one_hot(prev_action.long(), num_classes=5).float()
            else:
                action_oh = prev_action
        else:
            action_oh = torch.zeros(B, 5, device=device)

        # Concatenate all inputs
        x = torch.cat([
            old_belief,           # W
            observation,          # W
            pe,                   # W
            retrieved_memory,     # W
            session_latent,       # W
            action_oh,            # 5
            has_error,            # 1
        ], dim=-1)  # [B, 4*W + 6]

        h = F.relu(self.in_proj(x))
        g = self.gate(h)
        delta = self.update(h)
        new_belief = old_belief + g * delta

        return new_belief