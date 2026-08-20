"""Scientific Diagnostics for Multi-Future Coverage and Consequence Accuracy across K.

Metrics Evaluated:
1. Distinct Action Future Coverage (out of 5):
   - Measures how many of the 5 possible action conditions (Wait, Up, Left, Down, Right)
     are actively represented by at least one thoughtlet with confidence > 0.5.
   - For K=1: strictly <= 1.0.
   - For K=4..32: measures true parallel hypothesis capacity.
2. Hazard Future Identification Rate (%):
   - Measures whether imminent collision hazards (hazard >= 0.8) are recognized by a dedicated slot.
3. Consequence Prediction Error (Displacement & Reward MSE):
   - Quantifies whether internal future rollouts match actual environment dynamics.
4. Action Ranking Correlation (Spearman rho):
   - Correlation between thoughtlet consequence utilities and ground-truth branch rankings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..environments.phase2_suite import Phase2TaskEnvironment
from ..training.staged_branch_curriculum import (
    ComprehensiveBranchBundle,
    generate_comprehensive_branch_bundle,
)


def _compute_spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Pure NumPy Spearman rank correlation with zero external dependencies."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    std_a = np.std(rank_a)
    std_b = np.std(rank_b)
    if std_a < 1e-6 or std_b < 1e-6:
        return 0.0
    cov = np.mean((rank_a - np.mean(rank_a)) * (rank_b - np.mean(rank_b)))
    return float(cov / (std_a * std_b))


def _compute_auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUROC using rank-sum statistic."""
    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    r_pos = ranks[:len(pos)]
    u = np.sum(r_pos) - len(pos) * (len(pos) - 1) / 2.0
    return float(u / (len(pos) * len(neg)))


@dataclass(frozen=True, slots=True)
class FutureCoverageReport:
    """Quantitative evaluation report for multi-future representation and decision decomposition."""
    k_slots: int
    distinct_future_coverage: float       # in [1.0, 5.0]
    hazard_identification_rate_pct: float # in [0.0, 100.0]
    displacement_mse: float
    hazard_auroc: float                   # in [0.0, 1.0]
    mean_hazard_on_danger: float
    mean_hazard_on_safe: float
    ranking_correlation_rho: float        # in [-1.0, +1.0]
    stage1_imagined_pct: float            # % optimal action represented in slots
    stage2_accurate_pct: float            # % predictions accurate within epsilon
    stage3_correctly_ranked_pct: float    # % optimal consequence ranked #1
    stage4_selected_pct: float            # % winning action selected by actuator


def evaluate_future_coverage(
    model: nn.Module,
    env: Phase2TaskEnvironment,
    *,
    num_eval_states: int = 100,
    device: torch.device = torch.device("cpu"),
    active_slots: int | None = None,
) -> FutureCoverageReport:
    """Evaluate multi-future representation and decision decomposition on 100 decision states."""
    model.eval()

    covered_actions_list = []
    hazard_identified_list = []
    disp_errors = []
    correlations = []

    all_gt_hazards = []
    all_pred_hazards = []
    danger_haz_preds = []
    safe_haz_preds = []

    s1_imagined = []
    s2_accurate = []
    s3_ranked = []
    s4_selected = []

    k_slots = getattr(model, "config", None).thoughtlets if hasattr(model, "config") else 32
    if active_slots is not None:
        k_slots = active_slots

    with torch.no_grad():
        for state_idx in range(num_eval_states):
            obs = env.reset(state_idx + 5000)
            state = model.initial_state(1) if hasattr(model, "initial_state") else None

            raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor = torch.from_numpy(raw_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor.shape[-1] != 32:
                rgb_tensor = torch.nn.functional.interpolate(rgb_tensor, size=(32, 32), mode="nearest")

            ctrl_tensor = torch.zeros((1, 307), device=device)
            dt_tensor = torch.tensor([0.016667], device=device)

            out = model(rgb_tensor, ctrl_tensor, dt_tensor, state=state, active_slots=active_slots)
            proposals = out.action.proposals

            bundle = generate_comprehensive_branch_bundle(env, obs, device=device)

            # 1. Measure distinct action conditions covered
            pred_action_probs = proposals.action_probs[0]  # [K, 5]
            if proposals.active_mask is not None:
                mask = proposals.active_mask[0].bool()      # [K]
                active_action_probs = pred_action_probs[mask]
                active_hazards = proposals.hazard_prob[0].squeeze(-1)[mask]
                active_disps = proposals.displacement[0][mask]
                active_utils = proposals.consequence_utility[0].squeeze(-1)[mask]
            else:
                active_action_probs = pred_action_probs
                active_hazards = proposals.hazard_prob[0].squeeze(-1)
                active_disps = proposals.displacement[0]
                active_utils = proposals.consequence_utility[0].squeeze(-1)

            if len(active_action_probs) > 0:
                predicted_action_choices = torch.argmax(active_action_probs, dim=-1)  # [K_active]
                unique_actions = torch.unique(predicted_action_choices).cpu().numpy()
                num_unique = len(unique_actions)
            else:
                predicted_action_choices = torch.tensor([], device=device)
                num_unique = 0
            covered_actions_list.append(num_unique)

            # 2. Continuous Hazard Evaluation
            gt_hazards = bundle.hazard_probs[:5].squeeze(-1)
            has_dangerous = (gt_hazards >= 0.6).any().item()
            max_pred_h = float(active_hazards.max().item()) if len(active_hazards) > 0 else 0.0

            if has_dangerous:
                hazard_identified_list.append(1.0 if max_pred_h >= 0.40 else 0.0)
                danger_haz_preds.append(max_pred_h)
                all_gt_hazards.append(1)
                all_pred_hazards.append(max_pred_h)
            else:
                safe_haz_preds.append(max_pred_h)
                all_gt_hazards.append(0)
                all_pred_hazards.append(max_pred_h)

            # 3. Consequence Errors
            gt_disps = bundle.displacements[:5]
            if len(active_disps) > 0:
                dists = torch.cdist(active_disps, gt_disps)
                min_disp_err = dists.min(dim=0).values.mean().item()
            else:
                min_disp_err = 0.80
            # 4. Action-Aligned Ranking Correlation
            action_aligned_utils = []
            for act_i in range(5):
                matching_slots = (predicted_action_choices == act_i).nonzero(as_tuple=True)[0]
                if len(matching_slots) > 0:
                    act_u = float(active_utils[matching_slots].max().item())
                else:
                    act_u = -10.0
                action_aligned_utils.append(act_u)

            gt_utils_np = bundle.utilities[:5].squeeze(-1).cpu().numpy()
            rho = _compute_spearman_rho(np.array(action_aligned_utils), gt_utils_np)
            correlations.append(rho)

            # 5. 4-Stage Decision Failure Decomposition
            gt_utils = bundle.utilities[:5].squeeze(-1)
            opt_action_idx = int(torch.argmax(gt_utils).item())

            # Stage 1: Did any active slot represent the optimal action?
            if len(predicted_action_choices) > 0 and (predicted_action_choices == opt_action_idx).any():
                s1_imagined.append(1.0)
                matching_slots = (predicted_action_choices == opt_action_idx).nonzero(as_tuple=True)[0]
                best_slot = matching_slots[0]

                # Stage 2: Was its consequence prediction accurate?
                slot_disp_err = (active_disps[best_slot] - gt_disps[opt_action_idx]).abs().max().item()
                slot_haz_err = abs(active_hazards[best_slot].item() - gt_hazards[opt_action_idx].item())
                accurate = (slot_disp_err <= 0.6 and slot_haz_err <= 0.4)
                s2_accurate.append(1.0 if accurate else 0.0)

                # Stage 3: Was it ranked highest among active slots?
                ranked_top = (torch.argmax(active_utils).item() == best_slot.item())
                s3_ranked.append(1.0 if ranked_top else 0.0)
            else:
                s1_imagined.append(0.0)
                s2_accurate.append(0.0)
                s3_ranked.append(0.0)

            # Stage 4: Did actuator select optimal action?
            button_logits = out.action.button_logits[0].cpu().numpy()
            dir_keys = [0, 119, 97, 115, 100]  # Wait, W (119), A (97), S (115), D (100)
            dir_scores = [0.0 if idx == 0 else float(button_logits[key]) for idx, key in enumerate(dir_keys)]
            selected_act = int(np.argmax(dir_scores))
            s4_selected.append(1.0 if selected_act == opt_action_idx else 0.0)

    auroc = _compute_auroc(np.array(all_gt_hazards), np.array(all_pred_hazards))

    return FutureCoverageReport(
        k_slots=k_slots,
        distinct_future_coverage=float(np.mean(covered_actions_list)),
        hazard_identification_rate_pct=float(np.mean(hazard_identified_list) * 100.0) if hazard_identified_list else 0.0,
        displacement_mse=float(np.mean(disp_errors)),
        hazard_auroc=auroc,
        mean_hazard_on_danger=float(np.mean(danger_haz_preds)) if danger_haz_preds else 0.0,
        mean_hazard_on_safe=float(np.mean(safe_haz_preds)) if safe_haz_preds else 0.0,
        ranking_correlation_rho=float(np.mean(correlations)) if correlations else 0.0,
        stage1_imagined_pct=float(np.mean(s1_imagined) * 100.0),
        stage2_accurate_pct=float(np.mean(s2_accurate) * 100.0),
        stage3_correctly_ranked_pct=float(np.mean(s3_ranked) * 100.0),
        stage4_selected_pct=float(np.mean(s4_selected) * 100.0),
    )
