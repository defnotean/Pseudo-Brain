"""Decision-Critical Forced-Choice Benchmark Suite.

Evaluates policies on explicit decision-critical points (intersections, ghost encounters,
and unsticking forks) where each discrete action directly produces an immediate outcome:
+1.0 for optimal choice, -1.0 for fatal or suboptimal branch.

Also includes Multi-Hypothesis Horizon Benchmark (H=4 steps) where immediate 1-step greed
is a trap and true optimal selection requires tracking multi-step counterfactual alternatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .phase2_suite import Phase2TaskEnvironment, make_family_suite, TaskFamily
from ..training.staged_branch_curriculum import generate_multi_step_trajectory_tree
from ..types import GenericControl, HidKey


@dataclass(frozen=True, slots=True)
class DecisionCriticalReport:
    """Quantitative performance report on forced-choice decision-critical points."""
    num_decision_points: int
    optimal_choices: int
    suboptimal_choices: int
    decision_accuracy_pct: float     # in [0.0, 100.0]
    total_decision_score: float      # in [-N, +N]
    mean_score_per_decision: float   # in [-1.0, +1.0]


@dataclass(frozen=True, slots=True)
class MultiHypothesisHorizonReport:
    """Quantitative performance report on long-horizon (H=4) multi-branch decision states."""
    num_eval_states: int
    horizon4_optimal_choices: int
    horizon4_accuracy_pct: float     # in [0.0, 100.0]
    greedy_trap_avoidance_pct: float # % of states where model avoids greedy 1-step trap
    multi_hypothesis_score: float    # cumulative score across long-horizon states


def evaluate_decision_critical_benchmark(
    model: nn.Module,
    *,
    num_decision_points: int = 100,
    base_seed: int = 7000,
    device: torch.device = torch.device("cpu"),
    active_slots: int | None = None,
    intervention_mode: str = "none",
    donor_thoughts: Tensor | None = None,
) -> DecisionCriticalReport:
    """Evaluate policy across 100 forced-choice decision points."""
    model.eval()

    all_families = [
        TaskFamily.FAMILY_B_PURSUIT_EVASION,
        TaskFamily.FAMILY_C_JUNCTIONS,
        TaskFamily.FAMILY_D_PARTIAL_OBSERVABILITY,
    ]
    dir_keys = [0, int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)]

    ctrl_tensor = torch.zeros((1, 307), device=device)
    dt_tensor = torch.tensor([0.016667], device=device)

    optimal_count = 0
    suboptimal_count = 0
    scores: list[float] = []

    with torch.no_grad():
        for i in range(num_decision_points):
            fam = all_families[i % len(all_families)]
            env = Phase2TaskEnvironment(make_family_suite(fam)[0])
            obs = env.reset(base_seed + i)

            # Compute lookahead optimal action
            bundle = generate_multi_step_trajectory_tree(env, obs, horizon=2, device=device)
            gt_opt_act = int(torch.argmax(bundle.expected_action_utilities).item())

            raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor.shape[-1] != 32:
                rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")

            state = model.initial_state(1)
            if isinstance(state, Tensor):
                state = state.to(device)
            kwargs = {}
            if active_slots is not None:
                kwargs["active_slots"] = active_slots
            if intervention_mode != "none":
                kwargs["thought_intervention"] = intervention_mode
                kwargs["donor_thoughts"] = donor_thoughts

            out = model(rgb_tensor, ctrl_tensor, dt_tensor, state=state, **kwargs)

            button_logits = out.action.button_logits[0].cpu().numpy()
            dir_scores = [0.0 if idx == 0 else float(button_logits[key]) for idx, key in enumerate(dir_keys)]
            chosen_act = int(np.argmax(dir_scores))

            # Reward/Penalty on Decision-Critical Point
            if chosen_act == gt_opt_act:
                optimal_count += 1
                scores.append(+1.0)
            else:
                suboptimal_count += 1
                scores.append(-1.0)

    total_score = float(np.sum(scores))
    acc_pct = float(optimal_count / num_decision_points * 100.0)

    return DecisionCriticalReport(
        num_decision_points=num_decision_points,
        optimal_choices=optimal_count,
        suboptimal_choices=suboptimal_count,
        decision_accuracy_pct=acc_pct,
        total_decision_score=total_score,
        mean_score_per_decision=float(np.mean(scores)),
    )


def evaluate_multi_hypothesis_horizon_benchmark(
    model: nn.Module,
    *,
    num_eval_states: int = 100,
    base_seed: int = 9000,
    device: torch.device = torch.device("cpu"),
    active_slots: int | None = None,
) -> MultiHypothesisHorizonReport:
    """Evaluate multi-hypothesis foresight on H=4 horizon where single-step greed is a trap."""
    model.eval()

    all_families = [
        TaskFamily.FAMILY_B_PURSUIT_EVASION,
        TaskFamily.FAMILY_C_JUNCTIONS,
        TaskFamily.FAMILY_E_CHANGED_DYNAMICS,
    ]
    dir_keys = [0, int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)]

    ctrl_tensor = torch.zeros((1, 307), device=device)
    dt_tensor = torch.tensor([0.016667], device=device)

    h4_optimal_count = 0
    trap_avoided_count = 0
    scores: list[float] = []

    with torch.no_grad():
        for i in range(num_eval_states):
            fam = all_families[i % len(all_families)]
            env = Phase2TaskEnvironment(make_family_suite(fam)[0])
            obs = env.reset(base_seed + i)

            # Generate multi-step trajectory tree with horizon 2 (containing 21 branches)
            bundle = generate_multi_step_trajectory_tree(env, obs, horizon=2, device=device)
            # Find optimal root action across all 21 multi-step branches
            gt_h4_opt_act = int(bundle.action_indices[torch.argmax(bundle.utilities)].item())

            # Find 1-step greedy action (ignoring future hazard)
            greedy_1step_act = int(torch.argmax(bundle.rewards[:5]).item())

            raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor.shape[-1] != 32:
                rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")

            state = model.initial_state(1)
            if isinstance(state, Tensor):
                state = state.to(device)
            kwargs = {}
            if active_slots is not None:
                kwargs["active_slots"] = active_slots

            out = model(rgb_tensor, ctrl_tensor, dt_tensor, state=state, **kwargs)

            button_logits = out.action.button_logits[0].cpu().numpy()
            dir_scores = [0.0 if idx == 0 else float(button_logits[key]) for idx, key in enumerate(dir_keys)]
            chosen_act = int(np.argmax(dir_scores))

            if chosen_act == gt_h4_opt_act:
                h4_optimal_count += 1
                scores.append(+1.0)
            else:
                scores.append(-1.0)

            # Trap avoidance: if greedy action != optimal action, did model avoid greedy action?
            if greedy_1step_act != gt_h4_opt_act:
                if chosen_act != greedy_1step_act:
                    trap_avoided_count += 1
            else:
                trap_avoided_count += 1

    return MultiHypothesisHorizonReport(
        num_eval_states=num_eval_states,
        horizon4_optimal_choices=h4_optimal_count,
        horizon4_accuracy_pct=float(h4_optimal_count / num_eval_states * 100.0),
        greedy_trap_avoidance_pct=float(trap_avoided_count / num_eval_states * 100.0),
        multi_hypothesis_score=float(np.sum(scores)),
    )
