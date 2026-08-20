"""Staged Multi-Branch Curriculum & Multi-Hypothesis Supervision Engine.

Core Objectives:
1. Multi-Branch Counterfactual Generation:
   - For every state, generate M alternative action futures (e.g., Left, Right, Up, Down, Wait).
   - Each branch includes: displacement vector (dx, dy), hazard probability, reward, and optimal action.
2. Hungarian Minimum Bipartite Set Matching:
   - Assigns ground-truth branch outcomes to the K thoughtlets permutation-invariantly.
   - For K=1: Model can represent only 1 branch.
   - For K=4..32: Model represents multiple simultaneous alternative futures.
3. Decision-Relevant Utility Supervision:
   - Supervise the utility logits u_k such that high-value, low-hazard branches receive higher utility,
     directly driving the permutation-invariant aggregator to select the winning action intent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ..environments.phase2_suite import Phase2TaskEnvironment
from ..types import HidKey

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass(frozen=True, slots=True)
class ComprehensiveBranchBundle:
    """Rich set of alternative future branches from a single decision state."""
    action_indices: Tensor    # [M] (0: None, 1: W, 2: A, 3: S, 4: D)
    displacements: Tensor     # [M, 2] (dx, dy)
    hazard_probs: Tensor      # [M, 1]
    rewards: Tensor           # [M, 1]
    utilities: Tensor         # [M, 1] ground truth value/preference score


def generate_comprehensive_branch_bundle(
    env: Phase2TaskEnvironment,
    current_obs: Any,
    device: torch.device = torch.device("cpu"),
) -> ComprehensiveBranchBundle:
    """Generate exhaustive set of M=5 action branch outcomes from current state."""
    underlying = env._underlying_env
    px = getattr(underlying, "_player_x", 8)
    py = getattr(underlying, "_player_y", 8)
    ghosts = getattr(underlying, "_ghosts", [(4, 4), (12, 12)])
    pellets = getattr(underlying, "_pellets", None)

    # 5 discrete action options: Wait, W (Up), A (Left), S (Down), D (Right)
    actions = [
        (0, 0.0, 0.0),    # Wait
        (1, 0.0, -1.0),   # W: Up
        (2, -1.0, 0.0),   # A: Left
        (3, 0.0, 1.0),    # S: Down
        (4, 1.0, 0.0),    # D: Right
    ]

    act_indices = []
    disps = []
    hazards = []
    rewards = []
    utilities = []

    maze_walls = getattr(underlying, "_maze", set())
    has_pellet_fn = getattr(underlying, "_has_pellet", lambda x, y: False)

    for act_idx, dx, dy in actions:
        if dx == 0 and dy == 0:
            tx, ty = px, py
            is_wall = False
        else:
            tx = px + int(dx)
            ty = py + int(dy)
            is_wall = (tx, ty) in maze_walls or tx < 0 or tx >= 16 or ty < 0 or ty >= 16

        # Ghost hazard calculation
        min_ghost_dist = min(math.sqrt((tx - gx) ** 2 + (ty - gy) ** 2) for gx, gy in ghosts)
        if min_ghost_dist <= 1.2:
            haz = 1.0
        elif min_ghost_dist <= 2.5:
            haz = 0.6
        elif min_ghost_dist <= 4.0:
            haz = 0.2
        else:
            haz = 0.0

        if is_wall:
            rew = -2.0
            haz = max(haz, 0.4)
            tx, ty = px, py
        elif act_idx == 0:
            rew = -0.1
        elif has_pellet_fn(tx, ty):
            rew = 1.0
        else:
            rew = 0.1

        # If collision with hazard, severe penalty
        if haz >= 0.8:
            rew = -10.0

        # Branch utility score: U = Reward - 3.0 * Hazard
        util = rew - 3.0 * haz

        act_indices.append(act_idx)
        disps.append([float(tx - px), float(ty - py)])
        hazards.append([haz])
        rewards.append([rew])
        utilities.append([util])

    return ComprehensiveBranchBundle(
        action_indices=torch.tensor(act_indices, dtype=torch.long, device=device),
        displacements=torch.tensor(disps, dtype=torch.float32, device=device),
        hazard_probs=torch.tensor(hazards, dtype=torch.float32, device=device),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        utilities=torch.tensor(utilities, dtype=torch.float32, device=device),
    )


def generate_multi_step_trajectory_tree(
    env: Phase2TaskEnvironment,
    current_obs: Any,
    horizon: int = 2,
    device: torch.device = torch.device("cpu"),
) -> ComprehensiveBranchBundle:
    """Generate multi-step trajectory tree of candidate futures (1-step and 2-step paths)."""
    # Base 5 immediate branches
    base_bundle = generate_comprehensive_branch_bundle(env, current_obs, device=device)
    if horizon <= 1:
        return base_bundle

    underlying = env._underlying_env
    px = getattr(underlying, "_player_x", 8)
    py = getattr(underlying, "_player_y", 8)
    ghosts = getattr(underlying, "_ghosts", [(4, 4), (12, 12)])
    maze_walls = getattr(underlying, "_maze", set())
    has_pellet_fn = getattr(underlying, "_has_pellet", lambda x, y: False)

    act_indices = list(base_bundle.action_indices.cpu().numpy())
    disps = list(base_bundle.displacements.cpu().numpy())
    hazards = list(base_bundle.hazard_probs.cpu().numpy())
    rewards = list(base_bundle.rewards.cpu().numpy())
    utilities = list(base_bundle.utilities.cpu().numpy())

    # 4 directional 2-step extensions: (Up-Up, Up-Right, Left-Left, etc.)
    step1_actions = [(1, 0.0, -1.0), (2, -1.0, 0.0), (3, 0.0, 1.0), (4, 1.0, 0.0)]
    step2_actions = [(1, 0.0, -1.0), (2, -1.0, 0.0), (3, 0.0, 1.0), (4, 1.0, 0.0)]

    for a1_idx, dx1, dy1 in step1_actions:
        p1_x = px + int(dx1)
        p1_y = py + int(dy1)
        is_wall1 = (p1_x, p1_y) in maze_walls or p1_x < 0 or p1_x >= 16 or p1_y < 0 or p1_y >= 16
        if is_wall1:
            p1_x, p1_y = px, py

        for a2_idx, dx2, dy2 in step2_actions:
            # Avoid immediate reversal
            if (a1_idx == 1 and a2_idx == 3) or (a1_idx == 3 and a2_idx == 1):
                continue
            if (a1_idx == 2 and a2_idx == 4) or (a1_idx == 4 and a2_idx == 2):
                continue

            p2_x = p1_x + int(dx2)
            p2_y = p1_y + int(dy2)
            is_wall2 = (p2_x, p2_y) in maze_walls or p2_x < 0 or p2_x >= 16 or p2_y < 0 or p2_y >= 16
            if is_wall2:
                p2_x, p2_y = p1_x, p1_y

            cum_dx = float(p2_x - px)
            cum_dy = float(p2_y - py)

            min_g_dist = min(math.sqrt((p2_x - gx) ** 2 + (p2_y - gy) ** 2) for gx, gy in ghosts)
            haz = 1.0 if min_g_dist <= 1.5 else (0.5 if min_g_dist <= 3.0 else 0.0)
            if is_wall1 or is_wall2:
                rew = -2.0
            elif has_pellet_fn(p2_x, p2_y):
                rew = 1.5
            else:
                rew = 0.2

            if haz >= 0.8:
                rew = -15.0

            util = rew - 3.0 * haz

            act_indices.append(a1_idx)  # Root action of this trajectory
            disps.append([cum_dx, cum_dy])
            hazards.append([haz])
            rewards.append([rew])
            utilities.append([util])

    return ComprehensiveBranchBundle(
        action_indices=torch.tensor(act_indices, dtype=torch.long, device=device),
        displacements=torch.tensor(disps, dtype=torch.float32, device=device),
        hazard_probs=torch.tensor(hazards, dtype=torch.float32, device=device),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        utilities=torch.tensor(utilities, dtype=torch.float32, device=device),
    )


class MultiHypothesisBranchLoss(nn.Module):
    """Loss module supervising K thoughtlets on sets of M alternative futures via Hungarian matching."""

    def __init__(
        self,
        *,
        displacement_weight: float = 1.0,
        hazard_weight: float = 2.0,
        reward_weight: float = 1.0,
        utility_weight: float = 1.5,
        action_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.displacement_weight = displacement_weight
        self.hazard_weight = hazard_weight
        self.reward_weight = reward_weight
        self.utility_weight = utility_weight
        self.action_weight = action_weight

    def forward(
        self,
        proposals: Any,  # ThoughtProposal from ThoughtMediatedActuator
        bundles: Sequence[ComprehensiveBranchBundle],
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute set-matching loss between K thoughtlet proposals and M ground-truth branches.

        proposals.displacement: [B, K, 2]
        proposals.hazard_prob: [B, K, 1]
        proposals.reward_estimate: [B, K, 1]
        proposals.utility_logits: [B, K, 1]
        proposals.button_logits: [B, K, num_buttons]
        """
        batch_size = proposals.displacement.shape[0]
        k_slots = proposals.displacement.shape[1]
        device = proposals.displacement.device

        total_loss = torch.tensor(0.0, device=device)
        matched_disp_err = 0.0
        matched_haz_err = 0.0
        matched_rew_err = 0.0

        for b in range(batch_size):
            bundle = bundles[b]
            num_branches = bundle.displacements.shape[0]  # M

            pred_disp = proposals.displacement[b]       # [K, 2]
            pred_haz = proposals.hazard_prob[b]         # [K, 1]
            pred_rew = proposals.reward_estimate[b]     # [K, 1]
            pred_util = getattr(proposals, "consequence_utility", getattr(proposals, "utility_logits", None))[b]
            pred_act = getattr(proposals, "action_logits", getattr(proposals, "button_logits", None))[b]

            # Compute pairwise cost matrix between K proposals and M branches
            disp_cost = torch.cdist(pred_disp, bundle.displacements)  # [K, M]
            haz_cost = F.binary_cross_entropy(
                pred_haz.expand(-1, num_branches),
                bundle.hazard_probs.squeeze(-1).unsqueeze(0).expand(k_slots, -1),
                reduction="none",
            )  # [K, M]
            rew_cost = F.mse_loss(
                pred_rew.expand(-1, num_branches),
                bundle.rewards.squeeze(-1).unsqueeze(0).expand(k_slots, -1),
                reduction="none",
            )  # [K, M]

            cost_matrix = (
                self.displacement_weight * disp_cost
                + self.hazard_weight * haz_cost
                + self.reward_weight * rew_cost
            )  # [K, M]

            # Hungarian Bipartite Assignment
            if k_slots == 1:
                # Monolithic 1-slot model must be trained on the optimal lookahead branch
                opt_col = int(torch.argmax(bundle.utilities[:min(5, num_branches)]).item())
                row_ind = np.array([0])
                col_ind = np.array([opt_col])
            elif _HAS_SCIPY:
                cost_np = cost_matrix.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_np)
            else:
                row_ind = torch.argmin(cost_matrix, dim=0).cpu().numpy()
                col_ind = np.arange(num_branches)

            # Accumulate matched branch predictions
            matched_pred_loss = cost_matrix[row_ind, col_ind].mean()

            # Target action loss for matched slots
            if hasattr(proposals, "action_logits"):
                matched_act_loss = F.cross_entropy(
                    proposals.action_logits[b, row_ind],
                    bundle.action_indices[col_ind],
                )
            else:
                target_buttons = torch.zeros((len(row_ind), pred_act.shape[-1]), device=device)
                for idx, col in enumerate(col_ind):
                    act_idx = int(bundle.action_indices[col].item())
                    if act_idx == 1:
                        target_buttons[idx, int(HidKey.W)] = 1.0
                    elif act_idx == 2:
                        target_buttons[idx, int(HidKey.A)] = 1.0
                    elif act_idx == 3:
                        target_buttons[idx, int(HidKey.S)] = 1.0
                    elif act_idx == 4:
                        target_buttons[idx, int(HidKey.D)] = 1.0
                matched_act_loss = F.binary_cross_entropy_with_logits(pred_act[row_ind], target_buttons)

            # Confidence loss: active matched slots -> 1.0, unmatched slots -> 0.05
            if hasattr(proposals, "confidence"):
                conf_pred = proposals.confidence[b].squeeze(-1)  # [K]
                conf_target = torch.full_like(conf_pred, 0.05)
                conf_target[row_ind] = 0.95
                conf_loss = F.binary_cross_entropy(conf_pred, conf_target)
            else:
                conf_loss = torch.tensor(0.0, device=device)

            # Direct Pairwise Ranking Loss
            rank_loss = torch.tensor(0.0, device=device)
            if len(row_ind) >= 2 and pred_util is not None:
                gt_u = bundle.utilities[col_ind].squeeze(-1)     # [num_matched]
                pred_u = pred_util[row_ind].squeeze(-1)          # [num_matched]
                # Compare all pairs (i, j)
                diff_gt = gt_u.unsqueeze(1) - gt_u.unsqueeze(0)   # [N, N]
                diff_pred = pred_u.unsqueeze(1) - pred_u.unsqueeze(0) # [N, N]
                margin = 0.2
                # Target: if gt_i > gt_j + margin, pred_i should be > pred_j + margin
                pair_mask = (diff_gt > margin).float()
                if pair_mask.sum() > 0:
                    hinge = F.relu(margin - diff_pred) * pair_mask
                    rank_loss = hinge.sum() / pair_mask.sum().clamp_min(1.0)

            sample_loss = (
                matched_pred_loss
                + self.action_weight * matched_act_loss
                + 0.5 * conf_loss
                + self.utility_weight * rank_loss
            )
            total_loss = total_loss + sample_loss

            with torch.no_grad():
                matched_disp_err += float(disp_cost[row_ind, col_ind].mean().item())
                matched_haz_err += float(haz_cost[row_ind, col_ind].mean().item())
                matched_rew_err += float(rew_cost[row_ind, col_ind].mean().item())

        total_loss = total_loss / max(1, batch_size)
        metrics = {
            "branch_total_loss": float(total_loss.item()),
            "matched_disp_err": matched_disp_err / max(1, batch_size),
            "matched_haz_err": matched_haz_err / max(1, batch_size),
            "matched_rew_err": matched_rew_err / max(1, batch_size),
        }
        return total_loss, metrics
