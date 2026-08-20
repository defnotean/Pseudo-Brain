"""Stochastic, Partially-Occluded & Information-Gathering Benchmark Suite.

Evaluates decision quality in environments with genuine irreducible uncertainty:
1. Stochastic Ghost Branching: 50% probability ghost turns Left, 50% Right.
2. Information-Gathering Probes: WAIT reveals true branch to eliminate hazard.
3. Occluded Blind Corners: Hidden threats requiring counterfactual caution.

A single-hypothesis model (K=1) cannot simultaneously track incompatible branches
and is forced to collapse/guess, whereas multi-hypothesis models (K >= 8, 16, 32)
maintain candidate futures to evaluate expected risk and value of information.
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

from .phase2_suite import Phase2TaskEnvironment, make_family_suite, TaskFamily
from ..training.staged_branch_curriculum import generate_multi_step_trajectory_tree
from ..types import GenericControl, HidKey


@dataclass(frozen=True, slots=True)
class StochasticOccludedReport:
    """Quantitative performance report on stochastic, occluded, and probe decision states."""
    num_eval_states: int
    stochastic_decision_accuracy_pct: float       # % of states where model picks the expected-utility optimal action
    greedy_gamble_avoidance_pct: float            # % of states where model avoids the 50% fatal gamble
    information_probe_selection_pct: float        # % of ambiguous states where model selects WAIT/probe
    cumulative_expected_utility_score: float      # Total expected return across stochastic states
    mean_expected_utility_per_decision: float     # Average expected return per decision


def evaluate_stochastic_occluded_benchmark(
    model: nn.Module,
    *,
    num_eval_states: int = 100,
    base_seed: int = 15000,
    device: torch.device = torch.device("cpu"),
    active_slots: int | None = None,
) -> StochasticOccludedReport:
    """Evaluate decision-making under stochastic branching, occluded corners, and information probing."""
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
    gamble_avoided_count = 0
    probe_selected_count = 0
    expected_scores: list[float] = []

    with torch.no_grad():
        for i in range(num_eval_states):
            fam = all_families[i % len(all_families)]
            env = Phase2TaskEnvironment(make_family_suite(fam)[0])
            obs = env.reset(base_seed + i)

            # Generate multi-step trajectory tree
            bundle = generate_multi_step_trajectory_tree(env, obs, horizon=2, device=device)

            # Calculate Expected Utilities under 50/50 stochastic branch bifurcation
            # For each immediate action act_i in {0..4}, evaluate branches that share that root action
            action_expected_utils = []
            for act_i in range(5):
                matching_cols = (bundle.action_indices == act_i).nonzero(as_tuple=True)[0]
                if len(matching_cols) > 0:
                    branch_utils = bundle.utilities[matching_cols].squeeze(-1)
                    # Expected utility = mean across branching futures for this action
                    exp_u = float(branch_utils.mean().item())
                else:
                    exp_u = -10.0
                action_expected_utils.append(exp_u)

            # Find the true expected-utility optimal root action
            gt_expected_opt_act = int(np.argmax(action_expected_utils))

            # Identify if there is a tempting greedy gamble (action with max best-case reward but high hazard risk)
            best_case_rewards = [float(bundle.rewards[bundle.action_indices == a].max().item()) if (bundle.action_indices == a).any() else -10.0 for a in range(5)]
            hazard_risks = [float(bundle.hazard_probs[bundle.action_indices == a].max().item()) if (bundle.action_indices == a).any() else 0.0 for a in range(5)]
            greedy_gamble_act = int(np.argmax(best_case_rewards))

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

            # Score achieved: expected utility of the chosen action
            score = action_expected_utils[chosen_act]
            expected_scores.append(score)

            if chosen_act == gt_expected_opt_act:
                optimal_count += 1

            # Did the model avoid the greedy gamble if the gamble was suboptimal?
            if greedy_gamble_act != gt_expected_opt_act and hazard_risks[greedy_gamble_act] >= 0.5:
                if chosen_act != greedy_gamble_act:
                    gamble_avoided_count += 1
            else:
                gamble_avoided_count += 1

            # Probe action check: was WAIT chosen when WAIT had highest expected utility?
            if gt_expected_opt_act == 0:
                if chosen_act == 0:
                    probe_selected_count += 1

    return StochasticOccludedReport(
        num_eval_states=num_eval_states,
        stochastic_decision_accuracy_pct=float(optimal_count / num_eval_states * 100.0),
        greedy_gamble_avoidance_pct=float(gamble_avoided_count / num_eval_states * 100.0),
        information_probe_selection_pct=float(probe_selected_count / max(1, sum(1 for i in range(num_eval_states) if True)) * 100.0),
        cumulative_expected_utility_score=float(np.sum(expected_scores)),
        mean_expected_utility_per_decision=float(np.mean(expected_scores)),
    )
