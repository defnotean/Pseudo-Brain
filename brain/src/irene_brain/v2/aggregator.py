"""Permutation-invariant action aggregation for Core V2.

Stage V2.0 exposed a zero-gradient defect in the original scalar-utility
quotient.  The legacy path is retained for exact failure reproduction.  The
active path aggregates action-specific evidence directly with a set mean.
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
    """Aggregates K exchangeable hypotheses into the deployed action.

    ``direct_mean_logits_v1``:
        Q(a) = mean_k action_logit(k, a)

    This is a smooth Deep-Sets-style invariant with a direct supervised path
    from deployed cross entropy to every slot.  It is bounded with respect to
    K, though (like any ordinary multiset mean) duplicating only a subset can
    still change the result.

    Historical ``legacy_scalar_utility_v0``:
        mass_k,a = existence_k * branch_probability_k * action_probability_k,a
        Q(a) = sum_k mass_k,a * U_k / (sum_k mass_k,a + eps)

    The historical path can have identically zero action values and gradients
    when utilities are zero.  It must not be selected by new training runs.
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

        if self.config.decision_aggregation == "legacy_scalar_utility_v0":
            # Exact Stage V2.0 failure path, preserved for reproduction.
            mass_sum = mass.sum(dim=1, keepdim=True) + self.eps
            weighted_utility = (mass * utility).sum(dim=1, keepdim=True)
            Q = (weighted_utility / mass_sum).squeeze(1)
            mass_per_action = mass.sum(dim=1)
        else:
            # Direct action-specific evidence.  Consequence semantics remain
            # diagnostics until they receive grounded auxiliary targets.
            Q = hypotheses.action_logits.mean(dim=1)
            mass_per_action = action_probs.mean(dim=1)

        # Deployed action distribution
        action_dist = F.softmax(Q / self.temperature, dim=-1)  # [B, A]

        return AggregatedDecision(
            action_dist=action_dist,
            action_values=Q,
            thought_action_probs=action_probs,
            thought_existence=existence,
            thought_branch_prob=branch_probs,
            thought_utility=utility,
            mass_per_action=mass_per_action,
        )
