"""World Model V2 — predicts next latent given action.

Used for prediction error computation and planning.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CoreV2Config
from .consequence_model import ConsequenceHypothesis


class WorldModelV2(nn.Module):
    """Action-conditioned next-latent prediction.

    For each consequence hypothesis, predicts the next latent state.
    """

    def __init__(self, config: CoreV2Config):
        super().__init__()
        self.config = config
        W = config.width
        A = config.actions
        hidden = config.world_model_hidden

        # Action embedding
        self.action_embed = nn.Embedding(A, W)

        # Transition network
        self.transition = nn.Sequential(
            nn.Linear(W * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, W),
        )

        # EMA target encoder (for stable targets)
        self.target_encoder = None
        self.ema_decay = config.ema_decay

    def set_target_encoder(self, encoder: nn.Module):
        """Set the EMA target encoder (typically the SensorEncoder)."""
        import copy
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def update_target_encoder(self, source_encoder: nn.Module):
        """EMA update of target encoder."""
        if self.target_encoder is None:
            self.set_target_encoder(source_encoder)
        else:
            for tp, sp in zip(self.target_encoder.parameters(), source_encoder.parameters()):
                tp.data.mul_(self.ema_decay).add_(sp.data, alpha=1 - self.ema_decay)

    def forward(
        self,
        hypotheses: ConsequenceHypothesis,
        current_latent: torch.Tensor,   # [B, W] or [B, K, W]
    ) -> torch.Tensor:
        """Predict next latent for each hypothesis.

        Returns: [B, K, W] predicted next latents.
        """
        B, K, W = hypotheses.predicted_next_latent.shape
        device = hypotheses.predicted_next_latent.device

        # Expand current latent if [B, W]
        if current_latent.dim() == 2:
            current_latent = current_latent.unsqueeze(1).expand(-1, K, -1)

        # Get action probabilities and sample / expect
        action_probs = hypotheses.action_probabilities  # [B, K, A]

        # Expected action embedding
        action_emb = action_probs @ self.action_embed.weight  # [B, K, W]

        # Concatenate current latent + expected action embedding
        trans_input = torch.cat([current_latent, action_emb], dim=-1)  # [B, K, 2W]
        trans_input = trans_input.reshape(B * K, 2 * W)

        predicted_next = self.transition(trans_input).reshape(B, K, W)

        return predicted_next

    def compute_target(
        self,
        next_observation: torch.Tensor,  # [B, C, H, W] raw observation
    ) -> torch.Tensor:
        """Compute stop-gradient target encoding for next observation."""
        if self.target_encoder is None:
            raise RuntimeError("Target encoder not set. Call set_target_encoder() first.")
        with torch.no_grad():
            target = self.target_encoder(next_observation)
        return target