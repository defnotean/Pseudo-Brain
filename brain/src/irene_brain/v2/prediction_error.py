"""Prediction Error V2 — computed BEFORE new cognition at each step.

Compares pending predictions against reality, produces error state.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import CoreV2Config
from .state import PredictionErrorState, PendingPrediction


class ErrorEncoder(nn.Module):
    """Encodes raw latent difference into structured error representation."""
    def __init__(self, config: CoreV2Config):
        super().__init__()
        self.config = config
        W = config.width
        hidden = config.error_encoder_dim
        self.net = nn.Sequential(
            nn.Linear(W, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return self.net(delta)


class PredictionErrorV2(nn.Module):
    """Computes prediction error from pending predictions vs reality.

    Called at step t with:
    - pending prediction from t-1
    - actual observation at t (encoded)
    - actual reward/hazard from t-1 action
    """

    def __init__(self, config: CoreV2Config):
        super().__init__()
        self.config = config
        self.error_encoder = ErrorEncoder(config)
        # Surprise predictor: fixed 4 features = |latent|, |reward|, |hazard|,
        # |confidence| error magnitudes (zero when the component is absent).
        self.surprise_net = nn.Sequential(
            nn.Linear(4, config.surprise_dim),
            nn.ReLU(),
            nn.Linear(config.surprise_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pending: Optional[PendingPrediction],
        actual_latent: torch.Tensor,          # [B, W] or [B, K, W]
        actual_reward: Optional[torch.Tensor] = None,    # [B, 1] or [B, K, 1]
        actual_hazard: Optional[torch.Tensor] = None,    # [B, 1] or [B, K, 1]
        intervention: Optional[str] = None,  # "normal", "zero", "scrambled", "stale", "oracle"
    ) -> PredictionErrorState:
        """Compute prediction error.

        If pending is None (first step), returns zero errors.
        """
        B = actual_latent.shape[0]
        device = actual_latent.device
        W = actual_latent.shape[-1]

        if pending is None or pending.predicted_next_latent is None:
            return PredictionErrorState(
                latent_error=torch.zeros(B, W, device=device),
                reward_error=torch.zeros(B, 1, device=device) if actual_reward is not None else None,
                hazard_error=torch.zeros(B, 1, device=device) if actual_hazard is not None else None,
                confidence_error=None,
                surprise=torch.zeros(B, 1, device=device),
            )

        # Ensure shapes match: predicted [B, K, W] vs actual [B, W] -> expand actual
        pred_latent = pending.predicted_next_latent
        if pred_latent.dim() == 3 and actual_latent.dim() == 2:
            # [B, K, W] vs [B, W] -> expand actual to [B, K, W]
            actual_latent = actual_latent.unsqueeze(1).expand(-1, pred_latent.shape[1], -1)
        elif pred_latent.dim() == 2 and actual_latent.dim() == 2:
            pass  # both [B, W]

        # Latent error: stop-gradient on target
        latent_delta = pred_latent - actual_latent.detach()
        latent_error = self.error_encoder(latent_delta.flatten(0, 1)).reshape(latent_delta.shape[:2] + (-1,))

        # Reward error
        reward_error = None
        if pending.predicted_reward is not None and actual_reward is not None:
            pred_r = pending.predicted_reward
            if pred_r.dim() == 3 and actual_reward.dim() == 2:
                actual_reward = actual_reward.unsqueeze(1).expand(-1, pred_r.shape[1], -1)
            reward_error = actual_reward - pred_r.detach()

        # Hazard error
        hazard_error = None
        if pending.predicted_hazard is not None and actual_hazard is not None:
            pred_h = pending.predicted_hazard
            if pred_h.dim() == 3 and actual_hazard.dim() == 2:
                actual_hazard = actual_hazard.unsqueeze(1).expand(-1, pred_h.shape[1], -1)
            hazard_error = actual_hazard - pred_h.detach()

        # Confidence error (if we have predicted confidence and some measure of correctness)
        confidence_error = None
        if pending.predicted_confidence is not None:
            # We don't have ground-truth "correctness" here; skip or use 0
            confidence_error = torch.zeros_like(pending.predicted_confidence)

        # Surprise: fixed 4-feature vector of error magnitudes (0 when absent)
        def mag(t, ref):
            # reduce to [N, 1] magnitude matching latent_error's leading dims
            if t is None:
                return torch.zeros_like(ref)
            m = t.abs()
            while m.dim() > ref.dim():
                m = m.mean(dim=-1, keepdim=True) if m.shape[-1] != 1 else m
            if m.dim() == ref.dim() and m.shape != ref.shape:
                m = m.mean(dim=-1, keepdim=True)
            return m

        ref = latent_error.abs().mean(dim=-1, keepdim=True)
        feats = torch.cat([
            ref,
            mag(reward_error, ref),
            mag(hazard_error, ref),
            mag(confidence_error, ref),
        ], dim=-1)
        surprise = self.surprise_net(feats)

        # Reshape surprise to match latent_error batch dims
        if surprise.dim() == 3 and latent_error.dim() == 3:
            pass  # both [B, K, 1]
        elif surprise.dim() == 2 and latent_error.dim() == 3:
            surprise = surprise.unsqueeze(1).expand(-1, latent_error.shape[1], -1)

        state = PredictionErrorState(
            latent_error=latent_error,
            reward_error=reward_error,
            hazard_error=hazard_error,
            confidence_error=confidence_error,
            surprise=surprise,
        )

        if intervention and intervention != "normal":
            state = state.with_intervention(intervention)

        return state