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

        if config.outcome_action_conditioning == "factual_or_proposal_v1":
            self.outcome_action_embed = nn.Embedding(A, W)
            self.outcome_trunk = nn.Sequential(
                nn.Linear(hidden + W, hidden),
                nn.ReLU(),
            )
        else:
            self.outcome_action_embed = None
            self.outcome_trunk = None

        # Heads
        self.action_head = nn.Linear(hidden, A)
        self.next_latent_head = nn.Linear(hidden, W)
        self.reward_head = nn.Linear(hidden, 1)
        self.hazard_head = nn.Linear(hidden, 1)
        self.confidence_head = nn.Linear(hidden, 1)
        self.existence_head = nn.Linear(hidden, 1)
        self.branch_head = nn.Linear(hidden, 1)

    def forward(
        self,
        thoughts: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> ConsequenceHypothesis:
        """thoughts: [B, K, W] -> consequence hypotheses."""
        B, K, W = thoughts.shape

        # Flatten for shared trunk
        flat = thoughts.reshape(B * K, W)
        h = self.trunk(flat)

        # The action proposal itself is always computed from thought state.
        action_logits = self.action_head(h).reshape(B, K, -1)

        # Outcome heads can remain historically thought-only or explicitly
        # condition on either a factual applied action (training/pending
        # prediction) or each slot's soft proposal (counterfactual decision).
        outcome_h = h
        if self.outcome_action_embed is not None and self.outcome_trunk is not None:
            if action is None:
                action_emb = (
                    F.softmax(action_logits, dim=-1)
                    @ self.outcome_action_embed.weight
                )
            elif action.dim() == 1:
                action_emb = self.outcome_action_embed(action.long()).unsqueeze(1)
                action_emb = action_emb.expand(-1, K, -1)
            elif action.dim() == 2 and action.shape[-1] == self.config.actions:
                action_emb = (action @ self.outcome_action_embed.weight).unsqueeze(1)
                action_emb = action_emb.expand(-1, K, -1)
            else:
                raise ValueError(
                    "outcome action must have shape [B] or [B, actions]"
                )
            outcome_h = self.outcome_trunk(
                torch.cat([h.reshape(B, K, -1), action_emb], dim=-1).reshape(
                    B * K, -1
                )
            )
        predicted_next_latent = self.next_latent_head(outcome_h).reshape(B, K, -1)
        predicted_reward = self.reward_head(outcome_h).reshape(B, K, 1)
        predicted_hazard = self.hazard_head(outcome_h).reshape(B, K, 1)
        if self.config.hazard_parameterization == "probability_sigmoid_v1":
            predicted_hazard = torch.sigmoid(predicted_hazard)
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
