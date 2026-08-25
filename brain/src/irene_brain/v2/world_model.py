"""World Model V2 — predicts next latent given action.

Used for prediction error computation and planning.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CoreV2Config
from .consequence_model import ConsequenceHypothesis
from .outcome_model import ActionOutcomeTable, VectorizedOutcomeModelV2


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

        if config.outcome_architecture == "all_action_table_v1":
            self.action_embed = None
            self.transition = None
            self.outcome_model = VectorizedOutcomeModelV2(config)
        else:
            # Historical per-slot-shaped transition path.
            self.action_embed = nn.Embedding(A, W)
            self.transition = nn.Sequential(
                nn.Linear(W * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, W),
            )
            self.outcome_model = None

        # EMA target encoder (for stable targets)
        self.target_encoder = None
        self.ema_decay = config.ema_decay

    def set_target_encoder(self, encoder: nn.Module):
        """Set the EMA target encoder (typically the SensorEncoder)."""
        import copy
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.target_encoder.eval()

    def train(self, mode: bool = True):
        """Set train mode while keeping the EMA target strictly evaluation-only."""
        super().train(mode)
        if self.target_encoder is not None:
            self.target_encoder.eval()
        return self

    def update_target_encoder(self, source_encoder: nn.Module):
        """EMA update of target encoder."""
        if self.target_encoder is None:
            self.set_target_encoder(source_encoder)
        else:
            for tp, sp in zip(self.target_encoder.parameters(), source_encoder.parameters()):
                tp.data.mul_(self.ema_decay).add_(sp.data, alpha=1 - self.ema_decay)
            for target_buffer, source_buffer in zip(
                self.target_encoder.buffers(),
                source_encoder.buffers(),
            ):
                if target_buffer.is_floating_point():
                    target_buffer.data.mul_(self.ema_decay).add_(
                        source_buffer.data,
                        alpha=1 - self.ema_decay,
                    )
                else:
                    target_buffer.data.copy_(source_buffer.data)
            self.target_encoder.eval()

    def forward(
        self,
        hypotheses: ConsequenceHypothesis,
        current_latent: torch.Tensor,   # [B, W] or [B, K, W]
        action: torch.Tensor | None = None,  # factual [B] / [B, A], else per-slot proposal
    ) -> torch.Tensor:
        """Predict next latent for each hypothesis.

        Returns: [B, K, W] predicted next latents.
        """
        if self.transition is None or self.action_embed is None:
            raise RuntimeError(
                "legacy world-model forward is unavailable under all_action_table_v1"
            )
        B, K, W = hypotheses.predicted_next_latent.shape
        device = hypotheses.predicted_next_latent.device

        # Expand current latent if [B, W]
        if current_latent.dim() == 2:
            current_latent = current_latent.unsqueeze(1).expand(-1, K, -1)

        if action is None:
            # Counterfactual branch mode: each thoughtlet conditions its own
            # proposal on its expected action embedding.
            action_emb = hypotheses.action_probabilities @ self.action_embed.weight
        elif action.dim() == 1:
            # Factual teacher/deployment mode: the action that physically
            # drives this transition conditions every slot's prediction.
            action_emb = self.action_embed(action.long()).unsqueeze(1).expand(-1, K, -1)
        elif action.dim() == 2 and action.shape[-1] == self.config.actions:
            action_emb = (action @ self.action_embed.weight).unsqueeze(1).expand(-1, K, -1)
        else:
            raise ValueError("world-model action must have shape [B] or [B, actions]")

        # Concatenate current latent + expected action embedding
        trans_input = torch.cat([current_latent, action_emb], dim=-1)  # [B, K, 2W]
        trans_input = trans_input.reshape(B * K, 2 * W)

        predicted_next = self.transition(trans_input).reshape(B, K, W)

        return predicted_next

    def forward_all(self, context: torch.Tensor) -> ActionOutcomeTable:
        """Predict one explicitly labelled outcome row for every action."""

        if self.outcome_model is None:
            raise RuntimeError("forward_all requires all_action_table_v1")
        return self.outcome_model(context)

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
