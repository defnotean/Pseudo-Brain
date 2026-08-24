"""Multiplicity-Proof Action Aggregator V2.

Weighted MEAN, not raw vote count. Duplicate hypotheses cannot win by occupying more slots.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from .config import CoreV2Config
from .consequence_model import ConsequenceHypothesis, compute_branch_probabilities


@dataclass
class AggregatedDecision:
    """Final deployed cognitive decision."""
    action_dist: torch.Tensor              # [B, A] — DEPLOYED decision
    action_values: torch.Tensor            # [B, A] — Q-values
    thought_action_probs: torch.Tensor     # [B, K, A] — per-thoughtlet probs
    thought_existence: torch.Tensor        # [B, K, 1]
    thought_branch_prob: torch.Tensor      # [B, K, 1]
    thought_utility: torch.Tensor          # [B, K, 1]
    mass_per_action: torch.Tensor          # [B, A] — existence * branch * action_prob


class MultiplicityProofAggregator(nn.Module):
    """Aggregates K consequence hypotheses into deployed action_dist.

    For each action a:
        mass_k,a = existence_k * branch_probability_k * action_probability_k,a

    Q(a) = sum_k mass_k,a * U_k / (sum_k mass_k,a + eps)

    This is a weighted MEAN. Duplicate hypotheses cannot win by vote stuffing.
    """

    def __init__(self, config: CoreV2Config):
        super().__init__()
        self.config = config
        self.temperature = config.action_temperature
        self.reward_coeff = config.utility_reward_coeff
        self.hazard_coeff = config.utility_hazard_coeff
        self.eps = 1e-8

    def forward(
        self,
        hypotheses: ConsequenceHypothesis,
    ) -> AggregatedDecision:
        """Aggregate hypotheses to final deployed decision."""
        B, K, A = hypotheses.action_logits.shape
        device = hypotheses.action_logits.device

        # Per-thoughtlet action probabilities
        action_probs = hypotheses.action_probabilities  # [B, K, A]

        # Existence: "is this a meaningful hypothesis?"
        existence = hypotheses.existence  # [B, K, 1]

        # Branch probability: "given meaningful, how plausible?"
        branch_probs = compute_branch_probabilities(hypotheses)  # [B, K, 1]

        # Utility: predicted_reward - 3 * predicted_hazard
        utility = (
            self.reward_coeff * hypotheses.predicted_reward
            - self.hazard_coeff * hypotheses.predicted_hazard
        )  # [B, K, 1]

        # Mass per thoughtlet per action: existence * branch_prob * action_prob
        mass = existence * branch_probs * action_probs  # [B, K, A]

        # Q-values: weighted mean of utility per action
        mass_sum = mass.sum(dim=1, keepdim=True) + self.eps  # [B, 1, A]
        weighted_utility = (mass * utility).sum(dim=1, keepdim=True)  # [B, 1, A]
        Q = weighted_utility / mass_sum  # [B, 1, A]
        Q = Q.squeeze(1)  # [B, A]

        # Deployed action distribution
        action_dist = F.softmax(Q / self.temperature, dim=-1)  # [B, A]

        return AggregatedDecision(
            action_dist=action_dist,
            action_values=Q,
            thought_action_probs=action_probs,
            thought_existence=existence,
            thought_branch_prob=branch_probs,
            thought_utility=utility,
            mass_per_action=mass.sum(dim=1),  # [B, A]
        )