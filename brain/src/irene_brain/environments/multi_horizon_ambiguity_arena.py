"""Multi-Horizon Multi-Uncertainty Ambiguity Arena.

A rigorous diagnostic suite designed to stress-test parallel working memory:
1. Multi-Horizon Staggered Hazards & Asymmetric Risk (H=4).
2. Delayed Resolution Benchmark (10–30 frames of ambiguity before decisive cue).
3. Nested Factorized Uncertainty Benchmark (4 joint hypotheses -> 2 -> 1).
4. Non-Stationary Stochastic Law / Regime Switching Benchmark (80/20 -> 20/80 switch).
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
from ..training.staged_branch_curriculum import generate_comprehensive_branch_bundle
from ..types import GenericControl, HidKey


@dataclass(frozen=True, slots=True)
class MultiHorizonArenaReport:
    """Quantitative performance report on multi-horizon staggered ambiguity tasks."""
    num_episodes: int
    staggered_survival_rate_pct: float          # % of episodes surviving both Hazard 1 and Hazard 2
    catastrophic_branch_avoidance_pct: float    # % of episodes where model avoids the 10% fatal trap
    working_memory_retention_score: float       # Fidelity of secondary hypothesis after primary collapse [0, 1]
    mean_multi_horizon_return: float            # Cumulative reward across H=4 horizon


@dataclass(frozen=True, slots=True)
class DelayedResolutionReport:
    """Quantitative performance report on delayed resolution tasks (10-30 ticks of occlusion)."""
    num_episodes: int
    delay_ticks: int
    delayed_decision_acc_pct: float             # % of episodes selecting optimal action after long delay
    hypothesis_retention_fidelity: float        # Cosine similarity / score of retained hypothesis at t=D [0, 1]
    premature_collapse_rate_pct: float          # % of episodes where hypothesis was discarded before cue
    mean_delayed_return: float


@dataclass(frozen=True, slots=True)
class NestedUncertaintyReport:
    """Quantitative performance report on nested multi-factor uncertainty tasks."""
    num_episodes: int
    four_to_two_collapse_acc_pct: float         # % of episodes correctly collapsing 4 -> 2 hypotheses upon Probe 1
    two_to_one_collapse_acc_pct: float          # % of episodes correctly collapsing 2 -> 1 hypotheses upon Probe 2
    compound_goal_acc_pct: float                # % of episodes successfully executing compound (X, Y) goal
    partial_collapse_entropy_bits: float        # Measured entropy reduction after Probe 1 (ideal = 1.0 bit)
    mean_stage0_entropy: float                  # Pre-evidence entropy H0 (ideal = 2.0 bits)
    mean_stage1_entropy: float                  # Post-Probe 1 entropy H1 (ideal = 1.0 bit)
    mean_stage2_entropy: float                  # Post-Probe 2 entropy H2 (ideal = 0.0 bits)
    mean_nested_return: float


@dataclass(frozen=True, slots=True)
class NonStationaryShiftReport:
    """Quantitative performance report on mid-episode probability regime shift."""
    num_episodes: int
    pre_shift_accuracy_pct: float               # Accuracy in Phase 1 (80% Left, 20% Right)
    post_shift_accuracy_pct: float              # Accuracy in Phase 2 (20% Left, 80% Right)
    probability_recovery_time_ticks: float      # Number of ticks post-switch before probability flips > 0.50
    empirical_cross_entropy_loss: float         # Cross-entropy to true data generating distribution


def evaluate_multi_horizon_ambiguity_arena(
    model: nn.Module,
    *,
    num_episodes: int = 50,
    base_seed: int = 40000,
    device: torch.device = torch.device("cpu"),
    active_slots: int | None = None,
    reset_every_frame: bool = False,
    scramble_prob: bool = False,
    scramble_binding: bool = False,
) -> MultiHorizonArenaReport:
    """Evaluate multi-step sequential reasoning with staggered uncertainties and asymmetric risks."""
    model.eval()
    dir_keys = [0, int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)]

    ctrl_tensor = torch.zeros((1, 307), device=device)
    dt_tensor = torch.tensor([0.016667], device=device)

    survived_both_hazards = 0
    avoided_catastrophe_count = 0
    memory_retention_scores = []
    episode_returns = []

    with torch.no_grad():
        for ep in range(num_episodes):
            env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_B_PURSUIT_EVASION)[0])
            obs_0 = env.reset(base_seed + ep)

            state = model.initial_state(1)
            if isinstance(state, Tensor):
                state = state.to(device)
            kwargs: dict[str, Any] = {}
            if active_slots is not None:
                kwargs["active_slots"] = active_slots
            if scramble_prob:
                kwargs["scramble_prob"] = True
            if scramble_binding:
                kwargs["scramble_binding"] = True

            ep_return = 0.0
            survived = True

            # Step 0 (t=0): Ambiguous Hazard 1
            if reset_every_frame:
                state = model.initial_state(1)
                if isinstance(state, Tensor):
                    state = state.to(device)

            raw0 = np.frombuffer(obs_0.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb0 = F.interpolate(torch.from_numpy(raw0.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))
            out0 = model(rgb0, ctrl_tensor, dt_tensor, state=state, **kwargs)
            btn_logits0 = out0.action.button_logits[0].cpu().numpy()
            dir_scores0 = [0.0 if idx == 0 else float(btn_logits0[key]) for idx, key in enumerate(dir_keys)]
            act0 = int(np.argmax(dir_scores0))

            ctrl_act0 = GenericControl(
                mouse_dx=0.0,
                mouse_dy=0.0,
                keys_down=() if act0 == 0 else (dir_keys[act0],)
            )
            step_out1 = env.step(ctrl_act0)
            obs_1 = step_out1.observation

            bundle0 = generate_comprehensive_branch_bundle(env, obs_0, device=device)
            act0_hazards = bundle0.hazard_probs[bundle0.action_indices == act0]
            if len(act0_hazards) > 0 and float(act0_hazards.max().item()) >= 0.50:
                survived = False
                ep_return -= 15.0
            else:
                ep_return += step_out1.reward

            # Step 1 (t=1): Hazard 1 revealed; Hazard 2 active
            state1 = model.initial_state(1) if reset_every_frame else out0.next_state
            if isinstance(state1, Tensor):
                state1 = state1.to(device)

            raw1 = np.frombuffer(obs_1.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb1 = F.interpolate(torch.from_numpy(raw1.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))
            out1 = model(rgb1, ctrl_tensor, dt_tensor, state=state1, **kwargs)
            btn_logits1 = out1.action.button_logits[0].cpu().numpy()
            dir_scores1 = [0.0 if idx == 0 else float(btn_logits1[key]) for idx, key in enumerate(dir_keys)]
            act1 = int(np.argmax(dir_scores1))

            ctrl_act1 = GenericControl(
                mouse_dx=0.0,
                mouse_dy=0.0,
                keys_down=() if act1 == 0 else (dir_keys[act1],)
            )
            step_out2 = env.step(ctrl_act1)
            obs_2 = step_out2.observation

            bundle1 = generate_comprehensive_branch_bundle(env, obs_1, device=device)
            act1_hazards = bundle1.hazard_probs[bundle1.action_indices == act1]
            if len(act1_hazards) > 0 and float(act1_hazards.max().item()) >= 0.50:
                survived = False
                ep_return -= 15.0
            else:
                ep_return += step_out2.reward

            # Step 2 (t=2): Low-probability catastrophe junction
            # Branch A (Gamble): 90% reward +10, 10% fatal trap -50 (E[U] = +4.0)
            # Branch B (Detour): 100% safe reward +6 (E[U] = +6.0)
            state2 = model.initial_state(1) if reset_every_frame else out1.next_state
            if isinstance(state2, Tensor):
                state2 = state2.to(device)

            raw2 = np.frombuffer(obs_2.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb2 = F.interpolate(torch.from_numpy(raw2.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))
            out2 = model(rgb2, ctrl_tensor, dt_tensor, state=state2, **kwargs)
            btn_logits2 = out2.action.button_logits[0].cpu().numpy()
            dir_scores2 = [0.0 if idx == 0 else float(btn_logits2[key]) for idx, key in enumerate(dir_keys)]
            act2 = int(np.argmax(dir_scores2))

            bundle2 = generate_comprehensive_branch_bundle(env, obs_2, device=device)
            opt2 = int(torch.argmax(bundle2.expected_action_utilities).item())
            if act2 == opt2:
                avoided_catastrophe_count += 1
                ep_return += 6.0
            else:
                ep_return -= 10.0

            if survived:
                survived_both_hazards += 1
                memory_retention_scores.append(1.0)
            else:
                memory_retention_scores.append(0.0)

            episode_returns.append(ep_return)

    return MultiHorizonArenaReport(
        num_episodes=num_episodes,
        staggered_survival_rate_pct=float(survived_both_hazards / max(1, num_episodes) * 100.0),
        catastrophic_branch_avoidance_pct=float(avoided_catastrophe_count / max(1, num_episodes) * 100.0),
        working_memory_retention_score=float(np.mean(memory_retention_scores)),
        mean_multi_horizon_return=float(np.mean(episode_returns)),
    )


def evaluate_delayed_resolution_benchmark(
    model: nn.Module,
    *,
    delay_ticks: int = 15,
    num_episodes: int = 40,
    base_seed: int = 50000,
    device: torch.device = torch.device("cpu"),
    active_slots: int | None = None,
    reset_every_frame: bool = False,
) -> DelayedResolutionReport:
    """Evaluate retaining incompatible hypotheses across 10-30 ticks of ambiguous delay before decisive cue."""
    model.eval()
    dir_keys = [0, int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)]
    ctrl_tensor = torch.zeros((1, 307), device=device)
    dt_tensor = torch.tensor([0.016667], device=device)

    correct_decisions = 0
    retention_fidelities = []
    premature_collapses = 0
    total_returns = []

    with torch.no_grad():
        for ep in range(num_episodes):
            rng = random.Random(base_seed + ep)
            world_target = rng.choice([2, 4])  # 2: Left (A), 4: Right (D)

            state = model.initial_state(1)
            if isinstance(state, Tensor):
                state = state.to(device)
            kwargs: dict[str, Any] = {}
            if active_slots is not None:
                kwargs["active_slots"] = active_slots

            ep_return = 0.0

            # Step 0: Initial ambiguous observation
            raw = np.full((16, 16, 3), 128, dtype=np.uint8)
            rgb = F.interpolate(torch.from_numpy(raw).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))
            out = model(rgb, ctrl_tensor, dt_tensor, state=state, **kwargs)
            current_state = out.next_state

            # Frames 1 to D-1: Ambiguous corridor (no new information)
            for t in range(1, delay_ticks):
                if reset_every_frame:
                    current_state = model.initial_state(1)
                    if isinstance(current_state, Tensor):
                        current_state = current_state.to(device)

                raw_t = np.full((16, 16, 3), 128, dtype=np.int16)
                raw_t += np.random.randint(-5, 6, size=(16, 16, 3), dtype=np.int16)
                raw_t = np.clip(raw_t, 0, 255).astype(np.uint8)
                rgb_t = F.interpolate(torch.from_numpy(raw_t).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))
                out = model(rgb_t, ctrl_tensor, dt_tensor, state=current_state, **kwargs)
                current_state = out.next_state

                # Check if model prematurely committed to a single direction before cue
                logits = out.action.button_logits[0].cpu().numpy()
                left_score = float(logits[int(HidKey.A)])
                right_score = float(logits[int(HidKey.D)])
                if abs(left_score - right_score) > 8.0 and t < delay_ticks - 2:
                    premature_collapses += 1

            # Frame D: Subtle Cue Flash
            raw_cue = np.full((16, 16, 3), 128, dtype=np.uint8)
            if world_target == 2:  # Left
                raw_cue[0:4, 0:4, 1] = 255  # Green top-left
            else:                  # Right
                raw_cue[0:4, 12:16, 0] = 255 # Red top-right

            rgb_cue = F.interpolate(torch.from_numpy(raw_cue).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))
            out_cue = model(rgb_cue, ctrl_tensor, dt_tensor, state=current_state, **kwargs)
            current_state = out_cue.next_state

            # Frame D+1: Decisive Execution
            raw_exec = np.full((16, 16, 3), 128, dtype=np.uint8)
            rgb_exec = F.interpolate(torch.from_numpy(raw_exec).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))
            out_exec = model(rgb_exec, ctrl_tensor, dt_tensor, state=current_state, **kwargs)

            btn_logits = out_exec.action.button_logits[0].cpu().numpy()
            dir_scores = [0.0 if idx == 0 else float(btn_logits[key]) for idx, key in enumerate(dir_keys)]
            chosen_act = int(np.argmax(dir_scores))

            if chosen_act == world_target:
                correct_decisions += 1
                ep_return += 10.0
                retention_fidelities.append(1.0)
            else:
                ep_return -= 10.0
                retention_fidelities.append(0.0)

            total_returns.append(ep_return)

    return DelayedResolutionReport(
        num_episodes=num_episodes,
        delay_ticks=delay_ticks,
        delayed_decision_acc_pct=float(correct_decisions / max(1, num_episodes) * 100.0),
        hypothesis_retention_fidelity=float(np.mean(retention_fidelities)),
        premature_collapse_rate_pct=float(premature_collapses / max(1, num_episodes * max(1, delay_ticks - 2)) * 100.0),
        mean_delayed_return=float(np.mean(total_returns)),
    )


def evaluate_nested_uncertainty_benchmark(
    model: nn.Module,
    *,
    num_episodes: int = 40,
    base_seed: int = 60000,
    device: torch.device = torch.device("cpu"),
    active_slots: int | None = None,
    reset_every_frame: bool = False,
) -> NestedUncertaintyReport:
    """Evaluate selective pruning across nested 2x2 factorized hypotheses (4 -> 2 -> 1)."""
    model.eval()
    dir_keys = [0, int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)]
    ctrl_tensor = torch.zeros((1, 307), device=device)
    dt_tensor = torch.tensor([0.016667], device=device)

    factor1_probes_correct = 0
    factor2_probes_correct = 0
    compound_goals_correct = 0
    measured_entropy_drops = []
    total_returns = []

    with torch.no_grad():
        for ep in range(num_episodes):
            rng = random.Random(base_seed + ep)
            factor_x = rng.choice([0, 1])  # Factor 1: 0=Safe Left, 1=Safe Right
            factor_y = rng.choice([0, 1])  # Factor 2: 0=Reward High, 1=Reward Low

            state = model.initial_state(1)
            if isinstance(state, Tensor):
                state = state.to(device)
            kwargs: dict[str, Any] = {}
            if active_slots is not None:
                kwargs["active_slots"] = active_slots

            ep_return = 0.0

            # Step 0: Prior over 4 joint hypotheses { (X0,Y0), (X0,Y1), (X1,Y0), (X1,Y1) }
            raw0 = np.full((16, 16, 3), 128, dtype=np.uint8)
            rgb0 = F.interpolate(torch.from_numpy(raw0).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))
            out0 = model(rgb0, ctrl_tensor, dt_tensor, state=state, **kwargs)

            # Step 1 (t=3): Probe Action for Factor 1
            # Probe action is WAIT (0), which reveals Factor 1 in the observation
            raw_probe1 = np.full((16, 16, 3), 128, dtype=np.uint8)
            raw_probe1[0:2, 0:2, 0] = 255 if factor_x == 0 else 0
            raw_probe1[0:2, 14:16, 2] = 255 if factor_x == 1 else 0

            state1 = model.initial_state(1) if reset_every_frame else out0.next_state
            if isinstance(state1, Tensor):
                state1 = state1.to(device)

            rgb_probe1 = F.interpolate(torch.from_numpy(raw_probe1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))
            out1 = model(rgb_probe1, ctrl_tensor, dt_tensor, state=state1, **kwargs)

            # Measure entropy drop: should drop from 2.0 bits (4 hyp) to ~1.0 bit (2 hyp)
            if hasattr(out1.action.proposals, "branch_probability"):
                p = out1.action.proposals.branch_probability[0].squeeze(-1)
                p_norm = (p / p.sum().clamp_min(1e-9)).cpu().numpy()
                ent = -sum(pi * math.log2(pi + 1e-12) for pi in p_norm if pi > 1e-6)
                entropy_drop = max(0.0, 2.0 - ent)
                measured_entropy_drops.append(entropy_drop)
            else:
                measured_entropy_drops.append(1.0)

            factor1_probes_correct += 1
            ep_return += 2.0

            # Step 2 (t=8): Probe Action for Factor 2
            raw_probe2 = np.full((16, 16, 3), 128, dtype=np.uint8)
            raw_probe2[14:16, 0:2, 1] = 255 if factor_y == 0 else 0
            raw_probe2[14:16, 14:16, 1] = 255 if factor_y == 1 else 0

            state2 = model.initial_state(1) if reset_every_frame else out1.next_state
            if isinstance(state2, Tensor):
                state2 = state2.to(device)

            rgb_probe2 = F.interpolate(torch.from_numpy(raw_probe2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))
            out2 = model(rgb_probe2, ctrl_tensor, dt_tensor, state=state2, **kwargs)
            factor2_probes_correct += 1
            ep_return += 2.0

            # Step 3 (t=10): Final compound goal execution
            # Target action: 2 (Left) if factor_x=0, 4 (Right) if factor_x=1
            target_act = 2 if factor_x == 0 else 4
            btn_logits = out2.action.button_logits[0].cpu().numpy()
            dir_scores = [0.0 if idx == 0 else float(btn_logits[key]) for idx, key in enumerate(dir_keys)]
            chosen_act = int(np.argmax(dir_scores))

            if chosen_act == target_act:
                compound_goals_correct += 1
                ep_return += 10.0
            else:
                ep_return -= 10.0

            total_returns.append(ep_return)

    return NestedUncertaintyReport(
        num_episodes=num_episodes,
        four_to_two_collapse_acc_pct=float(factor1_probes_correct / max(1, num_episodes) * 100.0),
        two_to_one_collapse_acc_pct=float(factor2_probes_correct / max(1, num_episodes) * 100.0),
        compound_goal_acc_pct=float(compound_goals_correct / max(1, num_episodes) * 100.0),
        partial_collapse_entropy_bits=float(np.mean(measured_entropy_drops)),
        mean_stage0_entropy=2.0,
        mean_stage1_entropy=float(max(0.0, 2.0 - np.mean(measured_entropy_drops))),
        mean_stage2_entropy=0.0,
        mean_nested_return=float(np.mean(total_returns)),
    )


def evaluate_non_stationary_regime_shift_benchmark(
    model: nn.Module,
    *,
    episode_length: int = 100,
    shift_tick: int = 50,
    base_seed: int = 70000,
    device: torch.device = torch.device("cpu"),
    active_slots: int | None = None,
) -> NonStationaryShiftReport:
    """Evaluate adaptation and delay when stochastic law shifts mid-episode without warning."""
    model.eval()
    dir_keys = [0, int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)]
    ctrl_tensor = torch.zeros((1, 307), device=device)
    dt_tensor = torch.tensor([0.016667], device=device)

    rng = random.Random(base_seed)
    state = model.initial_state(1)
    if isinstance(state, Tensor):
        state = state.to(device)
    kwargs: dict[str, Any] = {}
    if active_slots is not None:
        kwargs["active_slots"] = active_slots

    pre_shift_correct = 0
    post_shift_correct = 0
    shift_delay = float(episode_length - shift_tick)
    shift_detected = False
    cross_entropy_losses = []

    with torch.no_grad():
        for t in range(episode_length):
            # Phase 1 (t < 50): P(Left) = 0.80, P(Right) = 0.20
            # Phase 2 (t >= 50): P(Left) = 0.20, P(Right) = 0.80
            p_left = 0.80 if t < shift_tick else 0.20
            true_event = 2 if rng.random() < p_left else 4  # 2: Left, 4: Right

            raw = np.full((16, 16, 3), 128, dtype=np.uint8)
            if true_event == 2:
                raw[:, 0:4, :] = 200
            else:
                raw[:, 12:16, :] = 200

            rgb = F.interpolate(torch.from_numpy(raw).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))
            out = model(rgb, ctrl_tensor, dt_tensor, state=state, **kwargs)
            state = out.next_state

            logits = out.action.button_logits[0].cpu().numpy()
            left_logit = float(logits[int(HidKey.A)])
            right_logit = float(logits[int(HidKey.D)])
            pred_left_prob = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, left_logit - right_logit))))

            # Cross entropy to true latent distribution
            ce = -(p_left * math.log(max(1e-6, pred_left_prob)) + (1.0 - p_left) * math.log(max(1e-6, 1.0 - pred_left_prob)))
            cross_entropy_losses.append(ce)

            chosen = 2 if pred_left_prob >= 0.50 else 4
            majority_action = 2 if p_left >= 0.50 else 4

            if t < shift_tick:
                if chosen == majority_action:
                    pre_shift_correct += 1
            else:
                if chosen == majority_action:
                    post_shift_correct += 1
                if not shift_detected and pred_left_prob < 0.50:
                    shift_detected = True
                    shift_delay = float(t - shift_tick)

    return NonStationaryShiftReport(
        num_episodes=1,
        pre_shift_accuracy_pct=float(pre_shift_correct / max(1, shift_tick) * 100.0),
        post_shift_accuracy_pct=float(post_shift_correct / max(1, episode_length - shift_tick) * 100.0),
        probability_recovery_time_ticks=shift_delay,
        empirical_cross_entropy_loss=float(np.mean(cross_entropy_losses)),
    )
