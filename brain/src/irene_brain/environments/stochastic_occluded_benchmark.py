"""Stochastic, Partially-Occluded & Information-Gathering Benchmark Suite.

Evaluates decision quality in environments with genuine irreducible uncertainty:
1. Stochastic Ghost Branching: 50% probability ghost turns Left, 50% Right.
2. Sequential Probe Resolution: WAIT at t=0 reveals ghost turn; agent must adapt at t=1.
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
from ..training.staged_branch_curriculum import generate_multi_step_trajectory_tree, generate_comprehensive_branch_bundle
from ..types import GenericControl, HidKey


@dataclass(frozen=True, slots=True)
class StochasticOccludedReport:
    """Quantitative performance report on stochastic, occluded, and probe decision states."""
    num_eval_states: int
    stochastic_decision_accuracy_pct: float       # % of states where model picks expected-utility optimal action
    greedy_gamble_avoidance_pct: float            # % of states where model avoids the 50% fatal gamble
    information_probe_selection_pct: float        # % of ambiguous states where model selects WAIT/probe
    cumulative_expected_utility_score: float      # Total expected return across stochastic states
    mean_expected_utility_per_decision: float     # Average expected return per decision


@dataclass(frozen=True, slots=True)
class SequentialProbeResolutionReport:
    """Quantitative performance report on multi-tick ambiguous probe-then-resolve episodes."""
    num_episodes: int
    probe_at_t0_rate_pct: float                   # % episodes where agent probes (WAIT) at t=0
    posterior_resolution_accuracy_pct: float      # % episodes where agent chooses correct corridor at t=1
    blind_gamble_death_rate_pct: float            # % episodes where agent guessed blindly and died
    mean_episode_return: float                    # Average closed-loop score achieved
    probe_value_realized_pct: float               # % of maximum information value captured


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
    probe_opportunity_count = 0
    expected_scores: list[float] = []

    with torch.no_grad():
        for i in range(num_eval_states):
            fam = all_families[i % len(all_families)]
            env = Phase2TaskEnvironment(make_family_suite(fam)[0])
            obs = env.reset(base_seed + i)

            bundle = generate_comprehensive_branch_bundle(env, obs, device=device)
            exp_utils = bundle.expected_action_utilities.squeeze(-1).cpu().numpy()
            gt_expected_opt_act = int(np.argmax(exp_utils))

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

            score = exp_utils[chosen_act]
            expected_scores.append(score)

            if chosen_act == gt_expected_opt_act:
                optimal_count += 1

            if greedy_gamble_act != gt_expected_opt_act and hazard_risks[greedy_gamble_act] >= 0.5:
                if chosen_act != greedy_gamble_act:
                    gamble_avoided_count += 1
            else:
                gamble_avoided_count += 1

            if gt_expected_opt_act == 0:
                probe_opportunity_count += 1
                if chosen_act == 0:
                    probe_selected_count += 1

    return StochasticOccludedReport(
        num_eval_states=num_eval_states,
        stochastic_decision_accuracy_pct=float(optimal_count / num_eval_states * 100.0),
        greedy_gamble_avoidance_pct=float(gamble_avoided_count / num_eval_states * 100.0),
        information_probe_selection_pct=float(probe_selected_count / max(1, probe_opportunity_count) * 100.0),
        cumulative_expected_utility_score=float(np.sum(expected_scores)),
        mean_expected_utility_per_decision=float(np.mean(expected_scores)),
    )


def evaluate_sequential_probe_resolution(
    model: nn.Module,
    *,
    num_episodes: int = 50,
    base_seed: int = 25000,
    device: torch.device = torch.device("cpu"),
    active_slots: int | None = None,
) -> SequentialProbeResolutionReport:
    """Evaluate full closed-loop 2-step decision sequence: Ambiguous t=0 (Probe) -> Revealed t=1 (Resolve)."""
    model.eval()
    dir_keys = [0, int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)]

    ctrl_tensor = torch.zeros((1, 307), device=device)
    dt_tensor = torch.tensor([0.016667], device=device)

    probed_at_t0_count = 0
    resolved_at_t1_count = 0
    blind_death_count = 0
    episode_returns = []

    with torch.no_grad():
        for ep in range(num_episodes):
            env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_B_PURSUIT_EVASION)[0])
            obs_0 = env.reset(base_seed + ep)

            # Randomize ghost coin-flip: 50% ghost moves Left, 50% ghost moves Right
            rng = random.Random(base_seed + ep * 7)
            ghost_turns_left = rng.random() < 0.50

            # Step 1: Decision at t=0 (Ambiguous State)
            raw_rgb0 = np.frombuffer(obs_0.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
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
            chosen_act0 = int(np.argmax(dir_scores0))

            bundle0 = generate_comprehensive_branch_bundle(env, obs_0, device=device)

            if chosen_act0 == 0:
                # Agent wisely chose to WAIT (Probe)
                probed_at_t0_count += 1
                # Step environment with WAIT
                step_out = env.step(GenericControl(mouse_dx=0.0, mouse_dy=0.0, keys_down=()))
                obs_1 = step_out.observation

                # Step 2: Decision at t=1 (Ghost direction is now revealed in observation!)
                raw_rgb1 = np.frombuffer(obs_1.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
                rgb1 = F.interpolate(torch.from_numpy(raw_rgb1.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))

                out1 = model(rgb1, ctrl_tensor, dt_tensor, state=out0.next_state, **kwargs)
                btn_logits1 = out1.action.button_logits[0].cpu().numpy()
                dir_scores1 = [0.0 if idx == 0 else float(btn_logits1[key]) for idx, key in enumerate(dir_keys)]
                chosen_act1 = int(np.argmax(dir_scores1))

                # Correct escape determined by actual observed state at t=1
                bundle1 = generate_comprehensive_branch_bundle(env, obs_1, device=device)
                correct_act1 = int(torch.argmax(bundle1.expected_action_utilities).item())
                if chosen_act1 == correct_act1:
                    resolved_at_t1_count += 1
                    ep_return = +10.0
                else:
                    ep_return = -5.0
            else:
                # Agent rushed blindly at t=0
                act_hazards = bundle0.hazard_probs[bundle0.action_indices == chosen_act0]
                is_high_hazard = float(act_hazards.max().item()) >= 0.50 if len(act_hazards) > 0 else False
                if is_high_hazard:
                    blind_death_count += 1
                    ep_return = -15.0
                else:
                    ep_return = +2.0  # Safe alternative

            episode_returns.append(ep_return)

    mean_ret = float(np.mean(episode_returns))
    max_possible_ret = +10.0
    voi_realized = max(0.0, min(100.0, (mean_ret - (-6.5)) / (max_possible_ret - (-6.5)) * 100.0))

    return SequentialProbeResolutionReport(
        num_episodes=num_episodes,
        probe_at_t0_rate_pct=float(probed_at_t0_count / num_episodes * 100.0),
        posterior_resolution_accuracy_pct=float(resolved_at_t1_count / max(1, probed_at_t0_count) * 100.0),
        blind_gamble_death_rate_pct=float(blind_death_count / num_episodes * 100.0),
        mean_episode_return=mean_ret,
        probe_value_realized_pct=voi_realized,
    )
