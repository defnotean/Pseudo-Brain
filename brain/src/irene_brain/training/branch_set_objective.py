"""Phase 2 Design 2: Multi-Future Branch Supervision & Causal Slot Dropout.

Core Mechanisms:
1. Unordered Multi-Future Branch Matching (Hungarian/Bipartite Set Matching):
   - Supervise K=32 thoughtlets against sets of counterfactual branch outcomes.
   - Permits permutation invariance: thoughtlets self-organize into complementary
     hypotheses (e.g. hazard evasion vs open corridor vs reward path) without fixed semantic indices.
2. Slot Dropout:
   - Randomly masks subsets of thought slots during training (p_drop in [0.10, 0.25])
     to prevent diffuse co-dependence and force distributed, complementary utility.
3. Causal Marginal Utility Supervision:
   - Evaluates the marginal contribution of individual thoughtlets, penalizing zero-impact redundancy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass(frozen=True, slots=True)
class BranchOutcome:
    """Ground-truth future outcome for a candidate branch."""
    action_index: int
    hazard_prob: float
    reward: float
    dx: float
    dy: float


class MultiFutureBranchObjective(nn.Module):
    """Permutation-invariant multi-future branch matching objective with slot dropout."""

    def __init__(
        self,
        *,
        thoughtlets: int = 32,
        core_width: int = 32,
        slot_dropout_prob: float = 0.15,
        hazard_weight: float = 2.0,
        reward_weight: float = 1.0,
        displacement_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.thoughtlets = thoughtlets
        self.core_width = core_width
        self.slot_dropout_prob = slot_dropout_prob
        self.hazard_weight = hazard_weight
        self.reward_weight = reward_weight
        self.displacement_weight = displacement_weight

        # Branch prediction projection from each thoughtlet: [dx, dy, reward, hazard_logit]
        self.branch_head = nn.Sequential(
            nn.Linear(core_width, core_width),
            nn.SiLU(),
            nn.Linear(core_width, 4),  # [dx, dy, reward, hazard_logit]
        )

    def apply_slot_dropout(self, thoughts: Tensor, training: bool = True) -> tuple[Tensor, Tensor]:
        """Apply Bernoulli slot dropout across thoughtlets during training.

        thoughts: [B, K, Registers, Width]
        Returns: (masked_thoughts, active_mask: [B, K])
        """
        if not training or self.slot_dropout_prob <= 0.0:
            active_mask = torch.ones(
                (thoughts.shape[0], thoughts.shape[1]),
                device=thoughts.device,
                dtype=torch.bool,
            )
            return thoughts, active_mask

        # Keep at least 4 slots active per sample
        batch_size, k, registers, width = thoughts.shape
        keep_prob = 1.0 - self.slot_dropout_prob
        rand = torch.rand((batch_size, k), device=thoughts.device)
        active_mask = rand < keep_prob

        # Guard: ensure at least 4 active slots per batch item
        for b in range(batch_size):
            if active_mask[b].sum() < 4:
                top_indices = torch.topk(rand[b], k=4, largest=False).indices
                active_mask[b, top_indices] = True

        mask_expanded = active_mask.unsqueeze(-1).unsqueeze(-1).expand_as(thoughts).float()
        masked_thoughts = thoughts * mask_expanded
        return masked_thoughts, active_mask

    def compute_branch_loss(
        self,
        thoughts: Tensor,  # [B, K, Registers, Width]
        ground_truth_branches: Sequence[Sequence[BranchOutcome]],  # per batch item
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute Hungarian-matched branch coverage loss."""
        batch_size, k, _registers, _width = thoughts.shape
        slot_summaries = thoughts.mean(dim=2)  # [B, K, Width]
        predictions = self.branch_head(slot_summaries)  # [B, K, 4] -> [dx, dy, r, hazard_logit]

        total_loss = torch.tensor(0.0, device=thoughts.device)
        matched_hazard_err = 0.0
        matched_reward_err = 0.0
        matched_disp_err = 0.0
        match_count = 0

        for b in range(batch_size):
            branches = ground_truth_branches[b] if b < len(ground_truth_branches) else []
            if not branches:
                continue

            num_branches = len(branches)
            target_tensors = torch.zeros((num_branches, 4), device=thoughts.device)
            for m, br in enumerate(branches):
                target_tensors[m, 0] = br.dx
                target_tensors[m, 1] = br.dy
                target_tensors[m, 2] = br.reward
                target_tensors[m, 3] = br.hazard_prob

            pred_b = predictions[b]  # [K, 4]
            # Pairwise cost matrix: [K, M]
            disp_diff = pred_b[:, :2].unsqueeze(1) - target_tensors[:, :2].unsqueeze(0)  # [K, M, 2]
            disp_cost = (disp_diff ** 2).sum(dim=-1)  # [K, M]

            reward_diff = pred_b[:, 2].unsqueeze(1) - target_tensors[:, 2].unsqueeze(0)  # [K, M]
            reward_cost = reward_diff ** 2

            pred_haz_prob = torch.sigmoid(pred_b[:, 3]).unsqueeze(1)  # [K, 1]
            target_haz = target_tensors[:, 3].unsqueeze(0)  # [1, M]
            hazard_cost = F.binary_cross_entropy(
                pred_haz_prob.expand(-1, num_branches),
                target_haz.expand(k, -1),
                reduction="none",
            )

            cost_matrix = (
                self.displacement_weight * disp_cost
                + self.reward_weight * reward_cost
                + self.hazard_weight * hazard_cost
            )  # [K, M]

            # Bipartite matching
            if _HAS_SCIPY:
                cost_np = cost_matrix.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_np)
            else:
                # Greedy min fallback
                min_slots = torch.argmin(cost_matrix, dim=0)
                row_ind = min_slots.cpu().numpy()
                col_ind = np.arange(num_branches)

            sample_loss = cost_matrix[row_ind, col_ind].mean()
            total_loss = total_loss + sample_loss

            with torch.no_grad():
                matched_disp_err += float(disp_cost[row_ind, col_ind].mean().item())
                matched_reward_err += float(reward_cost[row_ind, col_ind].mean().item())
                matched_hazard_err += float(hazard_cost[row_ind, col_ind].mean().item())
                match_count += 1

        if match_count > 0:
            total_loss = total_loss / match_count
            metrics = {
                "branch_total_loss": float(total_loss.item()),
                "branch_disp_err": matched_disp_err / match_count,
                "branch_reward_err": matched_reward_err / match_count,
                "branch_hazard_err": matched_hazard_err / match_count,
            }
        else:
            metrics = {
                "branch_total_loss": 0.0,
                "branch_disp_err": 0.0,
                "branch_reward_err": 0.0,
                "branch_hazard_err": 0.0,
            }

        return total_loss, metrics

    def apply_belief_dropout(self, belief: Tensor, p: float = 0.3, training: bool = True) -> Tensor:
        """Randomly mask out belief state to force actuator cross-attention to rely on thoughts."""
        if not training or p <= 0.0 or random.random() >= p:
            return belief
        return torch.zeros_like(belief)

    def compute_marginal_utility_loss(
        self,
        action_logits_with_thoughts: Tensor,
        action_logits_without_thoughts: Tensor,
        target_buttons: Tensor,
        margin: float = 0.2,
    ) -> Tensor:
        """Compute causal marginal utility loss.

        Penalizes representations where removing thoughtlet tokens causes no degradation
        to action classification accuracy.
        """
        loss_with = F.binary_cross_entropy_with_logits(action_logits_with_thoughts, target_buttons)
        loss_without = F.binary_cross_entropy_with_logits(action_logits_without_thoughts, target_buttons)
        # We want loss_with + margin <= loss_without
        marginal_penalty = F.relu(loss_with - loss_without + margin)
        return marginal_penalty
