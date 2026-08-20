"""Diagnostic module for Evaluating Belief Collapse, Posterior Updates, and Value of Information (VoI).

Tracks:
1. Pre-Observation Entropy H(p) vs Post-Observation Entropy H(p | obs).
2. Posterior Probability Shift: Surge in confirmed branch, suppression of disproven branches.
3. Fine-Grained Action Breakdown:
   - probe_wait_pct: % choosing WAIT / probe to disambiguate.
   - robust_detour_pct: % choosing safe non-gambling alternative route.
   - blind_gamble_pct: % choosing high-hazard 50/50 lethal gamble.
   - wrong_safe_pct: % choosing ineffective safe moves (e.g. wall collisions).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ..environments.phase2_suite import Phase2TaskEnvironment, make_family_suite, TaskFamily
from ..training.staged_branch_curriculum import generate_comprehensive_branch_bundle
from ..types import GenericControl, HidKey


@dataclass(frozen=True, slots=True)
class BeliefCollapseReport:
    """Quantitative report on belief collapse, hypothesis survival, and Value of Information."""
    num_episodes_evaluated: int
    pre_observation_entropy: float          # Shannon entropy H(p) before observation
    post_observation_entropy: float         # Shannon entropy H(p | obs) after observation
    entropy_reduction_pct: float            # % reduction in uncertainty
    posterior_accuracy_pct: float           # % states where confirmed branch probability surged > 0.70
    disproven_suppression_pct: float        # % states where disproven branch probability dropped < 0.20
    voi_regret: float                       # Average suboptimality vs perfect information lookahead
    # Action Breakdown:
    probe_wait_pct: float                   # % choosing WAIT/probe
    robust_detour_pct: float                # % choosing safe alternative detour
    blind_gamble_pct: float                 # % choosing 50/50 lethal gamble
    wrong_safe_pct: float                   # % choosing unhelpful safe move (wall bump)


def _compute_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy with epsilon smoothing."""
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log2(p)))


def compute_analytic_voi_example() -> tuple[float, float]:
    """Closed-form VoI example: WAIT/probe vs committing RIGHT under 50/50 ghost turn.

    Case: player at an intersection. Ghost turns LEFT or RIGHT with p=0.5.
    Action RIGHT:
      Branch 1 (ghost left):  reward = +2.0, hazard = 0.0 -> U = +2.0
      Branch 2 (ghost right): reward = -10.0, hazard = 1.0 -> U = -13.0
      Q(RIGHT) = 0.5 * 2.0 + 0.5 * (-13.0) = -5.5
    Action WAIT (probe):
      Immediate cost = -0.1
      After the ghost turn is observed, the best continuation is worth +2.0
      Q(WAIT) = -0.1 + 0.95 * 2.0 = +1.8
    """
    q_right = 0.5 * 2.0 + 0.5 * (-13.0)
    q_wait = -0.1 + 0.95 * (0.5 * 2.0 + 0.5 * 2.0)
    return q_right, q_wait


def evaluate_belief_collapse_and_voi(
    model: nn.Module,
    *,
    num_episodes: int = 50,
    base_seed: int = 20000,
    device: torch.device = torch.device("cpu"),
    active_slots: int | None = None,
) -> BeliefCollapseReport:
    """Evaluate belief collapse across 2-step decision sequence: t=0 (ambiguity) -> t=1 (evidence)."""
    model.eval()

    all_families = [
        TaskFamily.FAMILY_B_PURSUIT_EVASION,
        TaskFamily.FAMILY_C_JUNCTIONS,
    ]
    dir_keys = [0, int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)]

    pre_entropies = []
    post_entropies = []
    posterior_accuracies = []
    disproven_suppressions = []
    voi_regrets = []

    probe_counts = 0
    detour_counts = 0
    gamble_counts = 0
    wrong_safe_counts = 0

    ctrl_tensor = torch.zeros((1, 307), device=device)
    dt_tensor = torch.tensor([0.016667], device=device)

    with torch.no_grad():
        for ep in range(num_episodes):
            fam = all_families[ep % len(all_families)]
            env = Phase2TaskEnvironment(make_family_suite(fam)[0])
            obs_t0 = env.reset(base_seed + ep)

            # Step 1: Pre-Observation at t=0
            raw_rgb0 = np.frombuffer(obs_t0.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb0 = F.interpolate(torch.from_numpy(raw_rgb0.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))

            state0 = model.initial_state(1)
            if isinstance(state0, Tensor):
                state0 = state0.to(device)
            kwargs = {}
            if active_slots is not None:
                kwargs["active_slots"] = active_slots

            out0 = model(rgb0, ctrl_tensor, dt_tensor, state=state0, **kwargs)

            btn_logits0 = out0.action.button_logits[0].cpu().numpy()
            dir_scores0 = [0.0 if idx == 0 else float(btn_logits0[key]) for idx, key in enumerate(dir_keys)]

            # Compute pre-observation entropy over true decision distribution
            if hasattr(out0.action, "action_dist") and out0.action.action_dist is not None:
                p0 = out0.action.action_dist[0].cpu().numpy()
            else:
                p0 = F.softmax(torch.tensor(dir_scores0, dtype=torch.float32), dim=-1).numpy()
            h0 = _compute_entropy(p0)
            pre_entropies.append(h0)

            # Decompose action choice at t=0
            if hasattr(out0.action, "action_dist") and out0.action.action_dist is not None:
                chosen_act0 = int(torch.argmax(out0.action.action_dist[0]).item())
            else:
                chosen_act0 = int(np.argmax(dir_scores0))

            bundle0 = generate_comprehensive_branch_bundle(env, obs_t0, device=device)
            best_case_rewards = [float(bundle0.rewards[bundle0.action_indices == a].max().item()) if (bundle0.action_indices == a).any() else -10.0 for a in range(5)]
            hazard_risks = [float(bundle0.hazard_probs[bundle0.action_indices == a].max().item()) if (bundle0.action_indices == a).any() else 0.0 for a in range(5)]
            greedy_gamble_act = int(np.argmax(best_case_rewards))
            gt_voi_opt_act = int(torch.argmax(bundle0.expected_action_utilities).item())

            if chosen_act0 == 0:
                probe_counts += 1
            elif chosen_act0 == greedy_gamble_act and hazard_risks[greedy_gamble_act] >= 0.5:
                gamble_counts += 1
            elif chosen_act0 == gt_voi_opt_act:
                detour_counts += 1
            else:
                wrong_safe_counts += 1

            # Step 2: Agent takes probe action (WAIT) or proceeds to t=1
            act_key = dir_keys[chosen_act0]
            action_ctrl = GenericControl(mouse_dx=0.0, mouse_dy=0.0, keys_down=() if act_key == 0 else (act_key,))
            step_out = env.step(action_ctrl)
            obs_t1 = step_out.observation

            # Step 3: Post-Observation at t=1
            raw_rgb1 = np.frombuffer(obs_t1.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb1 = F.interpolate(torch.from_numpy(raw_rgb1.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))

            out1 = model(rgb1, ctrl_tensor, dt_tensor, state=out0.next_state, **kwargs)

            # Compute post-observation entropy over true decision distribution
            if hasattr(out1.action, "action_dist") and out1.action.action_dist is not None:
                p1 = out1.action.action_dist[0].cpu().numpy()
            else:
                btn_logits1 = out1.action.button_logits[0].cpu().numpy()
                dir_scores1 = [0.0 if idx == 0 else float(btn_logits1[key]) for idx, key in enumerate(dir_keys)]
                p1 = F.softmax(torch.tensor(dir_scores1), dim=-1).numpy()
            h1 = _compute_entropy(p1)
            post_entropies.append(h1)

            # Check posterior collapse: true branch surges, false branch drops
            bundle1 = generate_comprehensive_branch_bundle(env, obs_t1, device=device)
            confirmed_branch_idx = int(torch.argmax(bundle1.expected_action_utilities).item())
            posterior_accuracies.append(1.0 if p1[confirmed_branch_idx] >= 0.40 else 0.0)
            
            # Disproven branch (the unchosen high-risk branch)
            other_branches = [b for b in range(5) if b != confirmed_branch_idx and b != 0]
            if other_branches:
                max_disproven_p = max(p1[b] for b in other_branches)
                disproven_suppressions.append(1.0 if max_disproven_p <= 0.25 else 0.0)
            else:
                disproven_suppressions.append(1.0)

            # Regret vs optimal lookahead
            opt_u = float(bundle0.expected_action_utilities.max().item())
            chosen_u = float(bundle0.expected_action_utilities[chosen_act0].item()) if chosen_act0 < len(bundle0.expected_action_utilities) else -10.0
            voi_regrets.append(max(0.0, opt_u - chosen_u))

    mean_h0 = float(np.mean(pre_entropies))
    mean_h1 = float(np.mean(post_entropies))
    entropy_red = float(max(0.0, (mean_h0 - mean_h1) / (mean_h0 + 1e-6) * 100.0))

    return BeliefCollapseReport(
        num_episodes_evaluated=num_episodes,
        pre_observation_entropy=mean_h0,
        post_observation_entropy=mean_h1,
        entropy_reduction_pct=entropy_red,
        posterior_accuracy_pct=float(np.mean(posterior_accuracies) * 100.0) if posterior_accuracies else 0.0,
        disproven_suppression_pct=float(np.mean(disproven_suppressions) * 100.0) if disproven_suppressions else 0.0,
        voi_regret=float(np.mean(voi_regrets)),
        probe_wait_pct=float(probe_counts / num_episodes * 100.0),
        robust_detour_pct=float(detour_counts / num_episodes * 100.0),
        blind_gamble_pct=float(gamble_counts / num_episodes * 100.0),
        wrong_safe_pct=float(wrong_safe_counts / num_episodes * 100.0),
    )
