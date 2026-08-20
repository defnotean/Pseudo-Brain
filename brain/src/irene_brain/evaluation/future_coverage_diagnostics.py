"""Diagnostic module for evaluating multi-future representation, consequence accuracy,
and full 5-stage decision decomposition across thoughtlet configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..environments.phase2_suite import Phase2TaskEnvironment
from ..types import HidKey
from ..training.staged_branch_curriculum import (
    ComprehensiveBranchBundle,
    generate_comprehensive_branch_bundle,
    generate_multi_step_trajectory_tree,
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
    """Quantitative evaluation report for multi-future representation and granular decision decomposition."""
    k_slots: int
    distinct_future_coverage: float           # in [1.0, 5.0]
    hazard_identification_rate_pct: float     # in [0.0, 100.0]
    displacement_mse: float
    hazard_auroc: float                       # in [0.0, 1.0]
    mean_hazard_on_danger: float
    mean_hazard_on_safe: float
    ranking_correlation_rho: float            # in [-1.0, +1.0]
    optimal_action_top1_acc_pct: float        # % states where predicted argmax utility == optimal action
    reflex_action_flip_rate_pct: float        # % states where reflex changes the discrete action choice
    stage1_imagined_pct: float                # % optimal action represented in slots
    stage2a_displacement_pct: float           # % displacement prediction accurate
    stage2b_reward_pct: float                 # % reward prediction accurate
    stage2c_hazard_pct: float                 # % hazard prediction accurate
    stage2d_confidence_pct: float             # % confidence calibrated >= 0.5
    stage2_accurate_pct: float                # % all consequences simultaneously accurate
    stage3_correctly_ranked_pct: float        # % optimal consequence ranked #1 by derived utility
    stage4_selected_pct: float                # % winning action selected by actuator
    duplicate_slot_distribution: dict[str, float] # mean count of slots assigned to each action


def evaluate_future_coverage(
    model: nn.Module,
    env: Phase2TaskEnvironment,
    *,
    num_eval_states: int = 100,
    device: torch.device = torch.device("cpu"),
    active_slots: int | None = None,
) -> FutureCoverageReport:
    """Evaluate multi-future representation and granular decision decomposition on 100 decision states."""
    model.eval()

    covered_actions_list = []
    hazard_identified_list = []
    disp_errors = []
    correlations = []

    all_gt_hazards = []
    all_pred_hazards = []
    danger_haz_preds = []
    safe_haz_preds = []

    top1_ranked_list = []
    reflex_flips = []
    s1_imagined = []
    s2a_disp = []
    s2b_rew = []
    s2c_haz = []
    s2d_conf = []
    s2_accurate = []
    s3_ranked = []
    s4_selected = []
    action_slot_counts: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: [], 4: []}

    k_slots = getattr(model, "config", None).thoughtlets if hasattr(model, "config") else 32
    if active_slots is not None:
        k_slots = active_slots

    with torch.no_grad():
        for state_idx in range(num_eval_states):
            obs = env.reset(state_idx + 5000)
            state = model.initial_state(1) if hasattr(model, "initial_state") else None
            if isinstance(state, Tensor):
                state = state.to(device)

            raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor.shape[-1] != 32:
                rgb_tensor = torch.nn.functional.interpolate(rgb_tensor, size=(32, 32), mode="nearest")

            ctrl_tensor = torch.zeros((1, 307), device=device)
            dt_tensor = torch.tensor([0.016667], device=device)

            kwargs = {}
            if active_slots is not None:
                kwargs["active_slots"] = active_slots

            out = model(rgb_tensor, ctrl_tensor, dt_tensor, state=state, **kwargs)
            proposals = out.action.proposals

            bundle = generate_multi_step_trajectory_tree(env, obs, horizon=2, device=device)

            pred_disps = proposals.displacement[0]       # [K, 2]
            pred_hazards = proposals.hazard_prob[0]       # [K, 1]
            pred_rewards = proposals.reward_estimate[0]   # [K, 1]
            pred_utils = getattr(proposals, "consequence_utility", getattr(proposals, "utility_logits", None))[0]  # [K, 1]
            pred_confs = proposals.confidence[0].squeeze(-1) if hasattr(proposals, "confidence") else torch.ones(k_slots, device=device)

            active_mask = (pred_confs > 0.1) if k_slots > 0 else torch.zeros(0, dtype=torch.bool, device=device)
            if active_slots == 0:
                active_mask = torch.zeros(k_slots, dtype=torch.bool, device=device)

            if active_mask.any():
                active_disps = pred_disps[active_mask]
                active_hazards = pred_hazards[active_mask]
                active_rewards = pred_rewards[active_mask]
                active_utils = pred_utils[active_mask]
                active_confs = pred_confs[active_mask]

                if hasattr(proposals, "action_logits"):
                    predicted_action_choices = torch.argmax(proposals.action_logits[0][active_mask], dim=-1)
                else:
                    dir_keys = [0, int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)]
                    b_logits = proposals.button_logits[0][active_mask]
                    key_scores = torch.stack([b_logits[:, k] if k != 0 else torch.zeros(len(b_logits), device=device) for k in dir_keys], dim=-1)
                    predicted_action_choices = torch.argmax(key_scores, dim=-1)
            else:
                active_disps = torch.zeros((0, 2), device=device)
                active_hazards = torch.zeros((0, 1), device=device)
                active_rewards = torch.zeros((0, 1), device=device)
                active_utils = torch.zeros((0, 1), device=device)
                active_confs = torch.zeros(0, device=device)
                predicted_action_choices = torch.zeros(0, dtype=torch.long, device=device)

            # Record slot distribution per action
            for act_i in range(5):
                n_slots = int((predicted_action_choices == act_i).sum().item()) if len(predicted_action_choices) > 0 else 0
                action_slot_counts[act_i].append(n_slots)

            unique_actions = set(predicted_action_choices.cpu().numpy().tolist()) if len(predicted_action_choices) > 0 else set()
            covered_actions_list.append(len(unique_actions))

            gt_hazards = bundle.hazard_probs[:5].squeeze(-1)
            gt_rewards = bundle.rewards[:5].squeeze(-1)
            gt_disps = bundle.displacements[:5]

            has_danger = (gt_hazards > 0.5).any()
            if has_danger and len(active_hazards) > 0:
                identified = (active_hazards.max().item() > 0.5)
                hazard_identified_list.append(1.0 if identified else 0.0)
            elif not has_danger:
                hazard_identified_list.append(1.0)

            for h_gt, h_pred in zip(gt_hazards.cpu().numpy(), (active_hazards[:5].squeeze(-1).cpu().numpy() if len(active_hazards) >= 5 else np.zeros(5))):
                all_gt_hazards.append(1 if h_gt > 0.5 else 0)
                all_pred_hazards.append(float(h_pred))
                if h_gt > 0.5:
                    danger_haz_preds.append(float(h_pred))
                else:
                    safe_haz_preds.append(float(h_pred))

            if len(active_disps) > 0:
                disp_err = torch.cdist(active_disps, gt_disps).min().item()
                disp_errors.append(disp_err)
            else:
                disp_errors.append(2.0)

            # Align utilities to 5 discrete actions
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

            # 5. Optimal Action Top-1 Accuracy & Granular Decision Decomposition
            gt_utils = bundle.utilities[:5].squeeze(-1)
            opt_action_idx = int(torch.argmax(gt_utils).item())
            pred_top1_action = int(np.argmax(action_aligned_utils))
            top1_ranked_list.append(1.0 if pred_top1_action == opt_action_idx else 0.0)

            # Stage 1: Did any active slot represent the optimal action?
            if len(predicted_action_choices) > 0 and (predicted_action_choices == opt_action_idx).any():
                s1_imagined.append(1.0)
                matching_slots = (predicted_action_choices == opt_action_idx).nonzero(as_tuple=True)[0]
                best_slot = matching_slots[0]

                # Stage 2a: Displacement accurate within 1.0?
                slot_disp_err = (active_disps[best_slot] - gt_disps[opt_action_idx]).abs().max().item()
                disp_ok = (slot_disp_err <= 1.0)
                s2a_disp.append(1.0 if disp_ok else 0.0)

                # Stage 2b: Reward accurate within 0.3?
                slot_rew_err = abs(active_rewards[best_slot].item() - gt_rewards[opt_action_idx].item())
                rew_ok = (slot_rew_err <= 0.3)
                s2b_rew.append(1.0 if rew_ok else 0.0)

                # Stage 2c: Hazard accurate within 0.25?
                slot_haz_err = abs(active_hazards[best_slot].item() - gt_hazards[opt_action_idx].item())
                haz_ok = (slot_haz_err <= 0.25)
                s2c_haz.append(1.0 if haz_ok else 0.0)

                # Stage 2d: Confidence calibrated >= 0.5?
                conf_ok = (active_confs[best_slot].item() >= 0.50)
                s2d_conf.append(1.0 if conf_ok else 0.0)

                # Composite Stage 2:
                s2_accurate.append(1.0 if (disp_ok and rew_ok and haz_ok) else 0.0)

                # Stage 3: Ranked #1 among slots?
                ranked_top = (pred_top1_action == opt_action_idx)
                s3_ranked.append(1.0 if ranked_top else 0.0)
            else:
                s1_imagined.append(0.0)
                s2a_disp.append(0.0)
                s2b_rew.append(0.0)
                s2c_haz.append(0.0)
                s2d_conf.append(0.0)
                s2_accurate.append(0.0)
                s3_ranked.append(0.0)

            # Stage 4: Did actuator select optimal action?
            button_logits = out.action.button_logits[0].cpu().numpy()
            dir_keys = [0, int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)]
            dir_scores = [0.0 if idx == 0 else float(button_logits[key]) for idx, key in enumerate(dir_keys)]
            selected_act = int(np.argmax(dir_scores))
            s4_selected.append(1.0 if selected_act == opt_action_idx else 0.0)

            # Reflex Flip Rate Check
            main_logits = out.action.main_action_intent[0].cpu().numpy()
            main_dir_scores = [0.0 if idx == 0 else float(main_logits[key]) for idx, key in enumerate(dir_keys)]
            main_act = int(np.argmax(main_dir_scores))
            reflex_flips.append(1.0 if main_act != selected_act else 0.0)

    auroc = _compute_auroc(np.array(all_gt_hazards), np.array(all_pred_hazards))
    action_names = {0: "Wait", 1: "Up", 2: "Left", 3: "Down", 4: "Right"}
    slot_dist = {action_names[k]: float(np.mean(v)) for k, v in action_slot_counts.items()}

    return FutureCoverageReport(
        k_slots=k_slots,
        distinct_future_coverage=float(np.mean(covered_actions_list)),
        hazard_identification_rate_pct=float(np.mean(hazard_identified_list) * 100.0) if hazard_identified_list else 0.0,
        displacement_mse=float(np.mean(disp_errors)),
        hazard_auroc=auroc,
        mean_hazard_on_danger=float(np.mean(danger_haz_preds)) if danger_haz_preds else 0.0,
        mean_hazard_on_safe=float(np.mean(safe_haz_preds)) if safe_haz_preds else 0.0,
        ranking_correlation_rho=float(np.mean(correlations)) if correlations else 0.0,
        optimal_action_top1_acc_pct=float(np.mean(top1_ranked_list) * 100.0),
        reflex_action_flip_rate_pct=float(np.mean(reflex_flips) * 100.0),
        stage1_imagined_pct=float(np.mean(s1_imagined) * 100.0),
        stage2a_displacement_pct=float(np.mean(s2a_disp) * 100.0),
        stage2b_reward_pct=float(np.mean(s2b_rew) * 100.0),
        stage2c_hazard_pct=float(np.mean(s2c_haz) * 100.0),
        stage2d_confidence_pct=float(np.mean(s2d_conf) * 100.0),
        stage2_accurate_pct=float(np.mean(s2_accurate) * 100.0),
        stage3_correctly_ranked_pct=float(np.mean(s3_ranked) * 100.0),
        stage4_selected_pct=float(np.mean(s4_selected) * 100.0),
        duplicate_slot_distribution=slot_dist,
    )
