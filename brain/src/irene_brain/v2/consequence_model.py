"""Consequence Model V2 — every thoughtlet emits a consequence hypothesis.

Separates existence_probability from branch_probability.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from .config import CoreV2Config


@dataclass
class ConsequenceHypothesis:
    """One consequence hypothesis from a thoughtlet."""
    action_logits: torch.Tensor              # [B, K, A]
    predicted_next_latent: torch.Tensor      # [B, K, W]
    predicted_reward: torch.Tensor           # [B, K, 1]
    predicted_hazard: torch.Tensor           # [B, K, 1]
    confidence: torch.Tensor                 # [B, K, 1]
    existence_logit: torch.Tensor            # [B, K, 1]
    branch_logit: torch.Tensor               # [B, K, 1]

    @property
    def existence(self) -> torch.Tensor:
        return torch.sigmoid(self.existence_logit)

    @property
    def branch_probability(self) -> torch.Tensor:
        """Softmax over meaningful slots (existence-weighted).

        For training stability, we use a temperature-scaled softmax
        masked by existence > 0.5.
        """
        # This is computed in the aggregator
        return torch.zeros_like(self.existence)

    @property
    def action_probabilities(self) -> torch.Tensor:
        return F.softmax(self.action_logits, dim=-1)


class ConsequenceModelV2(nn.Module):
    """Maps thoughtlets to consequence hypotheses."""

    def __init__(self, config: CoreV2Config):
        super().__init__()
        self.config = config
        W = config.width
        K = config.thoughtlets
        A = config.actions
        hidden = config.consequence_hidden

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(W, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Heads
        self.action_head = nn.Linear(hidden, A)
        self.next_latent_head = nn.Linear(hidden, W)
        self.reward_head = nn.Linear(hidden, 1)
        self.hazard_head = nn.Linear(hidden, 1)
        self.confidence_head = nn.Linear(hidden, 1)
        self.existence_head = nn.Linear(hidden, 1)
        self.branch_head = nn.Linear(hidden, 1)

    def forward(self, thoughts: torch.Tensor) -> ConsequenceHypothesis:
        """thoughts: [B, K, W] -> consequence hypotheses."""
        B, K, W = thoughts.shape

        # Flatten for shared trunk
        flat = thoughts.reshape(B * K, W)
        h = self.trunk(flat)

        # Heads
        action_logits = self.action_head(h).reshape(B, K, -1)
        predicted_next_latent = self.next_latent_head(h).reshape(B, K, -1)
        predicted_reward = self.reward_head(h).reshape(B, K, 1)
        predicted_hazard = self.hazard_head(h).reshape(B, K, 1)
        confidence = torch.sigmoid(self.confidence_head(h)).reshape(B, K, 1)
        existence_logit = self.existence_head(h).reshape(B, K, 1)
        branch_logit = self.branch_head(h).reshape(B, K, 1)

        return ConsequenceHypothesis(
            action_logits=action_logits,
            predicted_next_latent=predicted_next_latent,
            predicted_reward=predicted_reward,
            predicted_hazard=predicted_hazard,
            confidence=confidence,
            existence_logit=existence_logit,
            branch_logit=branch_logit,
        )


def compute_branch_probabilities(
    hypotheses: ConsequenceHypothesis,
    temperature: float = 1.0,
    existence_threshold: float = 0.5,
) -> torch.Tensor:
    """Compute branch probabilities from existence and branch logits.

    Returns: [B, K, 1] branch probabilities that sum to 1 over meaningful slots.

    Meaningful = existence > threshold. If none meaningful, uniform over all.
    """
    existence = hypotheses.existence  # [B, K, 1]
    branch_logit = hypotheses.branch_logit  # [B, K, 1]

    # Mask by existence
    meaningful = (existence > existence_threshold).float()
    meaningful_sum = meaningful.sum(dim=1, keepdim=True)

    # If no meaningful slots, treat all as meaningful
    meaningful = torch.where(meaningful_sum > 0, meaningful, torch.ones_like(meaningful))

    # Masked softmax
    masked_logits = branch_logit + (meaningful - 1) * 1e9
    branch_probs = F.softmax(masked_logits / temperature, dim=1)

    return branch_probs