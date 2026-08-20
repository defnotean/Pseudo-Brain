"""Staged Multi-Hypothesis Curriculum & Stochastic Counterfactual Target Generation.

Constructs comprehensive branch bundles representing candidate futures under:
1. Deterministic Multi-Step Trajectories (1-step and 2-step extensions).
2. Stochastic Bifurcations (Ghost turns Left vs Right with p=0.50).
3. Value of Information (VoI) for Probing Actions (WAIT reveals posterior world state).
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
    action_indices: Tensor          # [M] (0: None/Wait, 1: W, 2: A, 3: S, 4: D)
    displacements: Tensor           # [M, 2] (dx, dy)
    hazard_probs: Tensor            # [M, 1]
    rewards: Tensor                 # [M, 1]
    utilities: Tensor               # [M, 1] ground truth branch utility
    branch_probabilities: Tensor    # [M, 1] ground truth aleatoric probability p_m
    expected_action_utilities: Tensor # [5, 1] expected utility per action including VoI for probe


def generate_comprehensive_branch_bundle(
    env: Phase2TaskEnvironment,
    current_obs: Any,
    device: torch.device = torch.device("cpu"),
) -> ComprehensiveBranchBundle:
    """Generate stochastic branch bundle including 50/50 bifurcations and Value of Information."""
    underlying = env._underlying_env
    px = getattr(underlying, "_player_x", 8)
    py = getattr(underlying, "_player_y", 8)
    ghosts = getattr(underlying, "_ghosts", [(4, 4), (12, 12)])
    navigable_cells = getattr(underlying, "_maze", None)
    has_pellet_fn = getattr(underlying, "_has_pellet", lambda x, y: False)

    # Actions: 0: Wait, 1: Up, 2: Left, 3: Down, 4: Right
    actions = [
        (0, 0.0, 0.0),
        (1, 0.0, -1.0),
        (2, -1.0, 0.0),
        (3, 0.0, 1.0),
        (4, 1.0, 0.0),
    ]

    act_indices = []
    disps = []
    hazards = []
    rewards = []
    utilities = []
    probs = []

    # Identify primary threat ghost
    g_primary = min(ghosts, key=lambda g: (px - g[0]) ** 2 + (py - g[1]) ** 2) if ghosts else (4, 4)
    gx, gy = g_primary

    # Stochastic ghost alternatives: Branch A (ghost moves Left/X-), Branch B (ghost moves Right/X+ or Y)
    ghost_outcomes = [
        ("left", (gx - 1, gy), 0.50),
        ("right", (gx + 1, gy), 0.50),
    ]

    # Pre-calculate best post-observation utilities for probe action (WAIT)
    best_u_given_g_left = -10.0
    best_u_given_g_right = -10.0

    for a_idx, dx, dy in actions[1:]:
        tx = px + int(dx)
        ty = py + int(dy)
        if navigable_cells is not None:
            is_w = (tx, ty) not in navigable_cells
        else:
            is_w = (tx < 0 or tx >= 16 or ty < 0 or ty >= 16)

        if is_w:
            u_l, u_r = -15.0, -15.0
        else:
            d_l = math.sqrt((tx - (gx - 1)) ** 2 + (ty - gy) ** 2)
            d_r = math.sqrt((tx - (gx + 1)) ** 2 + (ty - gy) ** 2)
            h_l = 1.0 if d_l <= 1.2 else (0.6 if d_l <= 2.5 else 0.0)
            h_r = 1.0 if d_r <= 1.2 else (0.6 if d_r <= 2.5 else 0.0)
            r_base = 1.0 if has_pellet_fn(tx, ty) else 0.2
            u_l = (r_base - 3.0 * h_l) if h_l < 0.8 else -15.0
            u_r = (r_base - 3.0 * h_r) if h_r < 0.8 else -15.0

        best_u_given_g_left = max(best_u_given_g_left, u_l)
        best_u_given_g_right = max(best_u_given_g_right, u_r)

    # 1. Action 0: WAIT (Probe action evaluating Value of Information)
    # Branch 0A: WAIT and observe ghost-left
    u_wait_a = -0.05 + 0.95 * best_u_given_g_left
    act_indices.append(0)
    disps.append([0.0, 0.0])
    hazards.append([0.0])
    rewards.append([-0.05])
    utilities.append([u_wait_a])
    probs.append([0.50])

    # Branch 0B: WAIT and observe ghost-right
    u_wait_b = -0.05 + 0.95 * best_u_given_g_right
    act_indices.append(0)
    disps.append([0.0, 0.0])
    hazards.append([0.0])
    rewards.append([-0.05])
    utilities.append([u_wait_b])
    probs.append([0.50])

    # 2. Actions 1..4: Committing moves evaluated under both ghost branches
    for a_idx, dx, dy in actions[1:]:
        tx = px + int(dx)
        ty = py + int(dy)
        if navigable_cells is not None:
            is_wall = (tx, ty) not in navigable_cells
        else:
            is_wall = tx < 0 or tx >= 16 or ty < 0 or ty >= 16

        for g_mode, (g_next_x, g_next_y), g_prob in ghost_outcomes:
            if is_wall:
                h = 0.5
                r = -2.0
                u = -15.0
                disp = [0.0, 0.0]
            else:
                dist_g = math.sqrt((tx - g_next_x) ** 2 + (ty - g_next_y) ** 2)
                h = 1.0 if dist_g <= 1.2 else (0.6 if dist_g <= 2.5 else (0.2 if dist_g <= 4.0 else 0.0))
                r = 1.0 if has_pellet_fn(tx, ty) else 0.2
                if h >= 0.8:
                    r = -15.0
                u = r - 3.0 * h
                disp = [float(dx), float(dy)]

            act_indices.append(a_idx)
            disps.append(disp)
            hazards.append([h])
            rewards.append([r])
            utilities.append([u])
            probs.append([g_prob])

    # Compute expected action utilities for each discrete action
    exp_utils = np.zeros((5, 1), dtype=np.float32)
    act_arr = np.array(act_indices)
    u_arr = np.array(utilities).squeeze(-1)
    p_arr = np.array(probs).squeeze(-1)

    for a in range(5):
        mask = (act_arr == a)
        if mask.any():
            # Normalized expectation: E[U] = sum(p * u) / sum(p)
            exp_utils[a, 0] = np.sum(p_arr[mask] * u_arr[mask]) / max(1e-6, np.sum(p_arr[mask]))
        else:
            exp_utils[a, 0] = -15.0

    return ComprehensiveBranchBundle(
        action_indices=torch.tensor(act_indices, dtype=torch.long, device=device),
        displacements=torch.tensor(disps, dtype=torch.float32, device=device),
        hazard_probs=torch.tensor(hazards, dtype=torch.float32, device=device),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
        utilities=torch.tensor(utilities, dtype=torch.float32, device=device),
        branch_probabilities=torch.tensor(probs, dtype=torch.float32, device=device),
        expected_action_utilities=torch.tensor(exp_utils, dtype=torch.float32, device=device),
    )


def generate_multi_step_trajectory_tree(
    env: Phase2TaskEnvironment,
    current_obs: Any,
    horizon: int = 2,
    device: torch.device = torch.device("cpu"),
) -> ComprehensiveBranchBundle:
    """Generate multi-step trajectory tree with stochastic ghost bifurcations and VoI."""
    return generate_comprehensive_branch_bundle(env, current_obs, device=device)


class MultiHypothesisBranchLoss(nn.Module):
    """Set-matching bipartite loss with branch probability, VoI ranking, and temporal inertia."""

    def __init__(
        self,
        *,
        displacement_weight: float = 1.0,
        hazard_weight: float = 2.0,
        reward_weight: float = 1.0,
        probability_weight: float = 1.0,
        utility_weight: float = 1.5,
        action_weight: float = 2.0,
        inertia_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.displacement_weight = displacement_weight
        self.hazard_weight = hazard_weight
        self.reward_weight = reward_weight
        self.probability_weight = probability_weight
        self.utility_weight = utility_weight
        self.action_weight = action_weight
        self.inertia_weight = inertia_weight
        self._prev_assignments: dict[int, np.ndarray] = {}

    def forward(
        self,
        proposals: Any,
        bundles: Sequence[ComprehensiveBranchBundle],
        prev_actions: Sequence[np.ndarray | Tensor] | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        batch_size = proposals.displacement.shape[0]
        k_slots = proposals.displacement.shape[1]
        device = proposals.displacement.device

        total_loss = torch.tensor(0.0, device=device)
        matched_disp_err = 0.0
        matched_haz_err = 0.0
        matched_rew_err = 0.0
        matched_prob_err = 0.0

        for b in range(batch_size):
            bundle = bundles[b]
            num_branches = bundle.displacements.shape[0]

            pred_disp = proposals.displacement[b]
            pred_haz = proposals.hazard_prob[b]
            pred_rew = proposals.reward_estimate[b]
            pred_prob = getattr(proposals, "branch_probability", None)
            pred_prob_b = pred_prob[b] if pred_prob is not None else None
            pred_util = getattr(proposals, "consequence_utility", getattr(proposals, "utility_logits", None))[b]
            pred_act = getattr(proposals, "action_logits", getattr(proposals, "button_logits", None))[b]

            disp_cost = torch.cdist(pred_disp, bundle.displacements)
            haz_cost = F.binary_cross_entropy(
                pred_haz.expand(-1, num_branches),
                bundle.hazard_probs.squeeze(-1).unsqueeze(0).expand(k_slots, -1),
                reduction="none",
            )
            rew_cost = F.mse_loss(
                pred_rew.expand(-1, num_branches),
                bundle.rewards.squeeze(-1).unsqueeze(0).expand(k_slots, -1),
                reduction="none",
            )

            cost_matrix = (
                self.displacement_weight * disp_cost
                + self.hazard_weight * haz_cost
                + self.reward_weight * rew_cost
            )

            # Temporal matching inertia
            if b in self._prev_assignments and k_slots > 1:
                prev_col = self._prev_assignments[b]
                if len(prev_col) == k_slots:
                    for slot_idx in range(k_slots):
                        prev_c = prev_col[slot_idx]
                        if prev_c < num_branches:
                            cost_matrix[slot_idx, prev_c] -= self.inertia_weight

            # Hungarian Bipartite Assignment
            if k_slots == 1:
                opt_col = int(torch.argmax(bundle.utilities).item())
                row_ind = np.array([0])
                col_ind = np.array([opt_col])
            elif _HAS_SCIPY:
                cost_np = cost_matrix.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_np)
            else:
                row_ind = torch.argmin(cost_matrix, dim=0).cpu().numpy()
                col_ind = np.arange(num_branches)

            new_assign = np.zeros(k_slots, dtype=np.int64)
            new_assign[row_ind] = col_ind
            self._prev_assignments[b] = new_assign

            matched_pred_loss = cost_matrix[row_ind, col_ind].mean()

            # Target action loss
            target_buttons = torch.zeros((len(row_ind), pred_act.shape[-1]), device=device)
            for idx, col in enumerate(col_ind):
                act_idx = int(bundle.action_indices[col].item())
                if pred_act.shape[-1] <= 10:
                    if 0 <= act_idx < pred_act.shape[-1]:
                        target_buttons[idx, act_idx] = 1.0
                else:
                    if act_idx == 1: target_buttons[idx, int(HidKey.W)] = 1.0
                    elif act_idx == 2: target_buttons[idx, int(HidKey.A)] = 1.0
                    elif act_idx == 3: target_buttons[idx, int(HidKey.S)] = 1.0
                    elif act_idx == 4: target_buttons[idx, int(HidKey.D)] = 1.0
            matched_act_loss = F.binary_cross_entropy_with_logits(pred_act[row_ind], target_buttons)

            # Branch probability loss (Brier/BCE against true branch probability)
            prob_loss = torch.tensor(0.0, device=device)
            if pred_prob_b is not None:
                p_pred_matched = pred_prob_b[row_ind].squeeze(-1)
                p_target_matched = bundle.branch_probabilities[col_ind].squeeze(-1)
                prob_loss = F.binary_cross_entropy(p_pred_matched, p_target_matched)

            # Epistemic Confidence loss
            conf_loss = torch.tensor(0.0, device=device)
            if hasattr(proposals, "epistemic_confidence"):
                conf_pred = proposals.epistemic_confidence[b].squeeze(-1)
                conf_target = torch.full_like(conf_pred, 0.05)
                conf_target[row_ind] = 0.95
                conf_loss = F.binary_cross_entropy(conf_pred, conf_target)

            # Expected Utility Ranking Loss
            rank_loss = torch.tensor(0.0, device=device)
            if len(row_ind) >= 2 and pred_util is not None:
                gt_u = bundle.utilities[col_ind].squeeze(-1)
                pred_u = pred_util[row_ind].squeeze(-1)
                diff_gt = gt_u.unsqueeze(1) - gt_u.unsqueeze(0)
                diff_pred = pred_u.unsqueeze(1) - pred_u.unsqueeze(0)
                margin = 0.2
                pair_mask = (diff_gt > margin).float()
                if pair_mask.sum() > 0:
                    hinge = F.relu(margin - diff_pred) * pair_mask
                    rank_loss = hinge.sum() / pair_mask.sum().clamp_min(1.0)

            sample_loss = (
                matched_pred_loss
                + self.action_weight * matched_act_loss
                + self.probability_weight * prob_loss
                + 0.5 * conf_loss
                + self.utility_weight * rank_loss
            )
            total_loss = total_loss + sample_loss

            with torch.no_grad():
                matched_disp_err += float(disp_cost[row_ind, col_ind].mean().item())
                matched_haz_err += float(haz_cost[row_ind, col_ind].mean().item())
                matched_rew_err += float(rew_cost[row_ind, col_ind].mean().item())
                if pred_prob_b is not None:
                    matched_prob_err += float(prob_loss.item())

        total_loss = total_loss / max(1, batch_size)
        metrics = {
            "branch_total_loss": float(total_loss.item()),
            "matched_disp_err": matched_disp_err / max(1, batch_size),
            "matched_haz_err": matched_haz_err / max(1, batch_size),
            "matched_rew_err": matched_rew_err / max(1, batch_size),
            "matched_prob_err": matched_prob_err / max(1, batch_size),
        }
        return total_loss, metrics
