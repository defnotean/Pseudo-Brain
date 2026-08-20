"""DGX Spark Campaign Harness for Thought-Mediated Parallel Cognition.

Evaluates:
1. Thought-Mediated Pseudo-Brain (K=32, W=32) vs Proposal-GRU Baseline across 5 seeds.
2. Resource-Matched Capacity Scaling Curve: K in {1, 4, 8, 16, 32} with compensating core width.
3. 32-Slot Causal Knockout Ablation.
4. Permutation Equivariance vs Cognitive Scrambling diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from irene_brain.environments.phase2_suite import (
    Phase2TaskEnvironment,
    TaskFamily,
    make_family_suite,
)
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.thought_mediated_model import (
    ProposalGRUBaseline,
    ThoughtMediatedBrainModel,
    build_resource_matched_thought_model,
)
from irene_brain.model.thought_scrambler import (
    permute_whole_slots,
    scramble_cognitive_bindings,
)
from irene_brain.evaluation.future_coverage_diagnostics import (
    FutureCoverageReport,
    evaluate_future_coverage,
)
from irene_brain.training.staged_branch_curriculum import (
    ComprehensiveBranchBundle,
    MultiHypothesisBranchLoss,
    generate_comprehensive_branch_bundle,
    generate_multi_step_trajectory_tree,
)
from irene_brain.types import GenericControl, HidKey


def compute_iqm(values: list[float]) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    n = len(v)
    q1 = int(n * 0.25)
    q3 = int(n * 0.75)
    if q1 >= q3:
        return float(np.mean(v))
    return float(np.mean(v[q1:q3]))


def compute_bootstrap_ci(data: list[float], n_bootstrap: int = 1000) -> tuple[float, float]:
    if len(data) < 2:
        val = data[0] if data else 0.0
        return val, val
    means = [float(np.mean(np.random.choice(data, size=len(data), replace=True))) for _ in range(n_bootstrap)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def generate_counterfactual_branches(env: Phase2TaskEnvironment, current_obs: Any) -> list[BranchOutcome]:
    branches = []
    actions = [(0, ()), (1, (int(HidKey.W),)), (2, (int(HidKey.A),)), (3, (int(HidKey.S),)), (4, (int(HidKey.D),))]
    px = getattr(env._underlying_env, "_player_x", 8)
    py = getattr(env._underlying_env, "_player_y", 8)
    ghosts = getattr(env._underlying_env, "_ghosts", [(4, 4), (12, 12)])

    for act_idx, keys in actions:
        dx = -1.0 if int(HidKey.A) in keys else (1.0 if int(HidKey.D) in keys else 0.0)
        dy = -1.0 if int(HidKey.W) in keys else (1.0 if int(HidKey.S) in keys else 0.0)
        target_x = max(1, min(14, px + int(dx)))
        target_y = max(1, min(14, py + int(dy)))
        min_ghost_dist = min(math.sqrt((target_x - gx) ** 2 + (target_y - gy) ** 2) for gx, gy in ghosts)
        hazard_prob = 1.0 if min_ghost_dist <= 1.5 else (0.5 if min_ghost_dist <= 3.0 else 0.0)
        reward = -10.0 if hazard_prob > 0.8 else (1.0 if hazard_prob == 0.0 else 0.0)
        branches.append(BranchOutcome(action_index=act_idx, hazard_prob=hazard_prob, reward=reward, dx=dx, dy=dy))
    return branches


def train_thought_mediated_model(
    model: ThoughtMediatedBrainModel,
    device: torch.device,
    training_steps: int = 300,
    lr: float = 1e-3,
) -> None:
    """Interactive On-Policy DAgger training with multi-step consequence tree supervision."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = MultiHypothesisBranchLoss().to(device)

    all_families = list(TaskFamily)
    step = 0

    while step < training_steps:
        family = all_families[step % len(all_families)]
        configs = make_family_suite(family)
        env = Phase2TaskEnvironment(configs[step % len(configs)])

        obs = env.reset(step + 1000)
        state = model.initial_state(1)
        ticks = 0

        # Interactive closed-loop rollout with online consequence supervision
        while ticks < min(30, env.config.max_ticks) and step < training_steps:
            raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor = torch.from_numpy(raw_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor.shape[-1] != 32:
                rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")

            ctrl_tensor = torch.zeros((1, 307), device=device)
            dt_tensor = torch.tensor([0.016667], device=device)

            out = model(rgb_tensor, ctrl_tensor, dt_tensor, state=state, max_cycles=model.config.cognitive_cycles)

            bundle = generate_multi_step_trajectory_tree(env, obs, horizon=2, device=device)
            total_loss, _metrics = loss_fn(out.action.proposals, [bundle])

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Advance closed-loop environment using student's predicted action
            with torch.no_grad():
                button_logits = out.action.button_logits[0].cpu().numpy()
                w_logit = float(button_logits[int(HidKey.W)])
                s_logit = float(button_logits[int(HidKey.S)])
                a_logit = float(button_logits[int(HidKey.A)])
                d_logit = float(button_logits[int(HidKey.D)])

                keys = []
                dir_scores = {"W": w_logit, "S": s_logit, "A": a_logit, "D": d_logit}
                best_dir, best_score = max(dir_scores.items(), key=lambda x: x[1])
                if best_score > 0.0:
                    if best_dir == "W":
                        keys.append(int(HidKey.W))
                    elif best_dir == "S":
                        keys.append(int(HidKey.S))
                    elif best_dir == "A":
                        keys.append(int(HidKey.A))
                    elif best_dir == "D":
                        keys.append(int(HidKey.D))

                action_ctrl = GenericControl(mouse_dx=0.0, mouse_dy=0.0, keys_down=tuple(keys))
                outcome = env.step(action_ctrl)
                state = out.next_state.detach()
                obs = outcome.observation
                ticks += 1
                step += 1
                if outcome.terminated or outcome.truncated:
                    break


def train_proposal_gru_baseline(
    model: ProposalGRUBaseline,
    device: torch.device,
    training_steps: int = 300,
    lr: float = 1e-3,
) -> None:
    """Interactive On-Policy DAgger training for matched Proposal-GRU baseline."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = MultiHypothesisBranchLoss().to(device)
    all_families = list(TaskFamily)
    step = 0

    while step < training_steps:
        family = all_families[step % len(all_families)]
        configs = make_family_suite(family)
        env = Phase2TaskEnvironment(configs[step % len(configs)])

        obs = env.reset(step + 1000)
        state = model.initial_state(1).to(device)
        ticks = 0

        while ticks < min(30, env.config.max_ticks) and step < training_steps:
            raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor = torch.from_numpy(raw_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor.shape[-1] != 32:
                rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")

            ctrl_tensor = torch.zeros((1, 307), device=device)
            dt_tensor = torch.tensor([0.016667], device=device)

            out = model(rgb_tensor, ctrl_tensor, dt_tensor, state=state)

            bundle = generate_multi_step_trajectory_tree(env, obs, horizon=2, device=device)
            total_loss, _metrics = loss_fn(out.action.proposals, [bundle])

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                button_logits = out.action.button_logits[0].cpu().numpy()
                w_logit = float(button_logits[int(HidKey.W)])
                s_logit = float(button_logits[int(HidKey.S)])
                a_logit = float(button_logits[int(HidKey.A)])
                d_logit = float(button_logits[int(HidKey.D)])

                keys = []
                dir_scores = {"W": w_logit, "S": s_logit, "A": a_logit, "D": d_logit}
                best_dir, best_score = max(dir_scores.items(), key=lambda x: x[1])
                if best_score > 0.0:
                    if best_dir == "W":
                        keys.append(int(HidKey.W))
                    elif best_dir == "S":
                        keys.append(int(HidKey.S))
                    elif best_dir == "A":
                        keys.append(int(HidKey.A))
                    elif best_dir == "D":
                        keys.append(int(HidKey.D))

                action_ctrl = GenericControl(mouse_dx=0.0, mouse_dy=0.0, keys_down=tuple(keys))
                outcome = env.step(action_ctrl)
                state = out.next_state.detach()
                obs = outcome.observation
                ticks += 1
                step += 1
                if outcome.terminated or outcome.truncated:
                    break


def evaluate_closed_loop_model(
    model: nn.Module,
    env: Phase2TaskEnvironment,
    seed: int,
    episodes: int = 5,
    device: torch.device = torch.device("cuda:0"),
    active_slots: int | None = None,
    intervention_mode: str = "none",
    donor_thoughts: Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    returns: list[float] = []
    online_top1_matches: list[float] = []

    for ep in range(episodes):
        obs = env.reset(seed * 1000 + ep)
        state = model.initial_state(1)
        if isinstance(state, torch.Tensor):
            state = state.to(device)
        ep_return = 0.0
        ticks = 0

        # Choose fixed slot permutation for the whole episode with aligned identity codes
        perm = None
        perm_noise = None
        if intervention_mode == "permute" and hasattr(state, "thoughts"):
            k = state.thoughts.shape[1]
            perm = list(range(k))
            random.shuffle(perm)
            if hasattr(model, "base_brain"):
                perm_tensor = torch.tensor(perm, device=device)
                perm_noise = model.base_brain._thought_identity_codes[perm_tensor].unsqueeze(0)

        while ticks < env.config.max_ticks:
            raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor = torch.from_numpy(raw_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor.shape[-1] != 32:
                rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")

            ctrl_tensor = torch.zeros((1, 307), device=device)
            dt_tensor = torch.tensor([0.016667], device=device)

            with torch.no_grad():
                kwargs = {}
                if active_slots is not None:
                    kwargs["active_slots"] = active_slots
                if perm_noise is not None:
                    kwargs["thought_noise"] = perm_noise
                if intervention_mode != "none":
                    kwargs["thought_intervention"] = intervention_mode
                    if intervention_mode != "permute":
                        kwargs["slot_permutation"] = perm
                    kwargs["donor_thoughts"] = donor_thoughts
                out = model(rgb_tensor, ctrl_tensor, dt_tensor, state=state, **kwargs)

            state = out.next_state

            button_logits = out.action.button_logits[0].cpu().numpy()
            w_logit = float(button_logits[int(HidKey.W)])
            s_logit = float(button_logits[int(HidKey.S)])
            a_logit = float(button_logits[int(HidKey.A)])
            d_logit = float(button_logits[int(HidKey.D)])

            keys = []
            dir_scores = {"W": w_logit, "S": s_logit, "A": a_logit, "D": d_logit}
            best_dir, best_score = max(dir_scores.items(), key=lambda x: x[1])
            chosen_act = 0
            if best_score > 0.0:
                if best_dir == "W":
                    keys.append(int(HidKey.W))
                    chosen_act = 1
                elif best_dir == "A":
                    keys.append(int(HidKey.A))
                    chosen_act = 2
                elif best_dir == "S":
                    keys.append(int(HidKey.S))
                    chosen_act = 3
                elif best_dir == "D":
                    keys.append(int(HidKey.D))
                    chosen_act = 4

            # Measure Online Closed-Loop Top-1 Accuracy on actual encountered state
            with torch.no_grad():
                bundle = generate_multi_step_trajectory_tree(env, obs, horizon=2, device=device)
                gt_opt_action = int(torch.argmax(bundle.utilities[:5]).item())
                online_top1_matches.append(1.0 if chosen_act == gt_opt_action else 0.0)

            action_ctrl = GenericControl(
                mouse_dx=0.0,
                mouse_dy=0.0,
                keys_down=tuple(keys),
            )

            outcome = env.step(action_ctrl)
            ep_return += outcome.reward
            ticks += 1
            if outcome.terminated or outcome.truncated:
                break
            obs = outcome.observation

        returns.append(ep_return)

    return {
        "mean_return": float(np.mean(returns)),
        "iqm_return": compute_iqm(returns),
        "online_top1_acc_pct": float(np.mean(online_top1_matches) * 100.0) if online_top1_matches else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Thought-Mediated Parallel Cognition Campaign")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episodes-per-world", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--output-json", type=str, default="docs/phase_closure/thought_mediated_campaign_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Starting Checkpointed On-Policy DAgger Campaign on {device} ===")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

    seeds = [42, 43, 44, 45, 46]
    checkpoints = [200, 400, 600, 800, 1000]
    test_env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_B_PURSUIT_EVASION)[0])

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": str(device),
        "seeds": seeds,
        "checkpoints": checkpoints,
        "pb_k32_trajectory": {},
        "gru_trajectory": {},
        "thought_mediated_pb_k32": {},
        "proposal_gru_baseline": {},
        "resource_matched_capacity_curve": {},
        "future_coverage_diagnostics": {},
        "progressive_causal_interventions": {},
    }

    # 1. Checkpointed Interactive On-Policy DAgger for K=32 Pseudo-Brain
    print("\n--- Checkpointed On-Policy DAgger Training: Pseudo-Brain (K=32) ---")
    pb_model = build_resource_matched_thought_model(32).to(device)
    current_step = 0
    for ckpt in checkpoints:
        steps_to_train = ckpt - current_step
        if steps_to_train > 0:
            train_thought_mediated_model(pb_model, device=device, training_steps=steps_to_train)
            current_step = ckpt

        eval_res = evaluate_closed_loop_model(pb_model, test_env, seed=42, episodes=args.episodes_per_world, device=device)
        cov_res = evaluate_future_coverage(pb_model, test_env, device=device)

        results["pb_k32_trajectory"][ckpt] = {
            "iqm_return": eval_res["iqm_return"],
            "online_top1_acc_pct": eval_res["online_top1_acc_pct"],
            "distinct_future_coverage": cov_res.distinct_future_coverage,
            "hazard_identification_rate_pct": cov_res.hazard_identification_rate_pct,
            "displacement_mse": cov_res.displacement_mse,
            "hazard_auroc": cov_res.hazard_auroc,
            "mean_hazard_on_danger": cov_res.mean_hazard_on_danger,
            "mean_hazard_on_safe": cov_res.mean_hazard_on_safe,
            "ranking_correlation_rho": cov_res.ranking_correlation_rho,
            "offline_top1_acc_pct": cov_res.optimal_action_top1_acc_pct,
            "reflex_action_flip_rate_pct": cov_res.reflex_action_flip_rate_pct,
            "stage1_imagined_pct": cov_res.stage1_imagined_pct,
            "stage2_accurate_pct": cov_res.stage2_accurate_pct,
            "stage3_correctly_ranked_pct": cov_res.stage3_correctly_ranked_pct,
            "stage4_selected_pct": cov_res.stage4_selected_pct,
        }
        print(f"  [PB K=32 @ {ckpt:4d} Steps] IQM = {eval_res['iqm_return']:6.2f} | Online Top1 = {eval_res['online_top1_acc_pct']:5.1f}% | Offline Top1 = {cov_res.optimal_action_top1_acc_pct:5.1f}% | Rank ρ = {cov_res.ranking_correlation_rho:+.3f} | Reflex Flips = {cov_res.reflex_action_flip_rate_pct:.1f}%")

    # 2. Checkpointed Interactive On-Policy DAgger for Matched Proposal-GRU Baseline
    print("\n--- Checkpointed On-Policy DAgger Training: Proposal-GRU Baseline ---")
    gru_model = ProposalGRUBaseline().to(device)
    current_step = 0
    for ckpt in checkpoints:
        steps_to_train = ckpt - current_step
        if steps_to_train > 0:
            train_proposal_gru_baseline(gru_model, device=device, training_steps=steps_to_train)
            current_step = ckpt

        eval_res = evaluate_closed_loop_model(gru_model, test_env, seed=42, episodes=args.episodes_per_world, device=device)
        results["gru_trajectory"][ckpt] = {
            "iqm_return": eval_res["iqm_return"],
            "online_top1_acc_pct": eval_res["online_top1_acc_pct"],
        }
        print(f"  [GRU Matched @ {ckpt:4d} Steps] IQM = {eval_res['iqm_return']:6.2f} | Online Top1 = {eval_res['online_top1_acc_pct']:5.1f}%")

    # Full 5-family evaluation at final 1,000 steps
    print("\n--- Full 5-Seed Evaluation on 5 Task Families at 1000 Steps ---")
    pb_returns = []
    pb_family_scores: dict[str, list[float]] = {}
    gru_returns = []
    gru_family_scores: dict[str, list[float]] = {}

    for seed in seeds:
        for family in TaskFamily:
            configs = make_family_suite(family)
            eval_env = Phase2TaskEnvironment(configs[0])
            pb_res = evaluate_closed_loop_model(pb_model, eval_env, seed=seed, episodes=args.episodes_per_world, device=device)
            pb_returns.append(pb_res["mean_return"])
            pb_family_scores.setdefault(family.value, []).append(pb_res["mean_return"])

            gru_res = evaluate_closed_loop_model(gru_model, eval_env, seed=seed, episodes=args.episodes_per_world, device=device)
            gru_returns.append(gru_res["mean_return"])
            gru_family_scores.setdefault(family.value, []).append(gru_res["mean_return"])

    ci_low, ci_high = compute_bootstrap_ci(pb_returns)
    results["thought_mediated_pb_k32"] = {
        "mean_return": float(np.mean(pb_returns)),
        "iqm_return": compute_iqm(pb_returns),
        "ci_95": [ci_low, ci_high],
        "family_scores": {k: float(np.mean(v)) for k, v in pb_family_scores.items()},
    }
    gru_ci_low, gru_ci_high = compute_bootstrap_ci(gru_returns)
    results["proposal_gru_baseline"] = {
        "mean_return": float(np.mean(gru_returns)),
        "iqm_return": compute_iqm(gru_returns),
        "ci_95": [gru_ci_low, gru_ci_high],
        "family_scores": {k: float(np.mean(v)) for k, v in gru_family_scores.items()},
    }

    # 3. Resource-Matched Capacity Scaling Curve: K in {0, 1, 4, 5, 8, 16, 32}
    print("\n--- Evaluating Resource-Matched Capacity Scaling Curve across K ---")
    k_points = [0, 1, 4, 5, 8, 16, 32]
    capacity_curve: dict[int, float] = {}
    coverage_curve: dict[int, dict[str, float]] = {}

    for k in k_points:
        if k == 0:
            matched_model = build_resource_matched_thought_model(1).to(device)
            eval_res = evaluate_closed_loop_model(matched_model, test_env, seed=42, episodes=args.episodes_per_world, device=device, active_slots=0)
            cov_res = evaluate_future_coverage(matched_model, test_env, device=device, active_slots=0)
        else:
            matched_model = build_resource_matched_thought_model(k).to(device)
            train_thought_mediated_model(matched_model, device=device, training_steps=300)
            eval_res = evaluate_closed_loop_model(matched_model, test_env, seed=42, episodes=args.episodes_per_world, device=device)
            cov_res = evaluate_future_coverage(matched_model, test_env, device=device)

        capacity_curve[k] = eval_res["iqm_return"]
        coverage_curve[k] = {
            "distinct_future_coverage": cov_res.distinct_future_coverage,
            "hazard_identification_rate_pct": cov_res.hazard_identification_rate_pct,
            "displacement_mse": cov_res.displacement_mse,
            "hazard_auroc": cov_res.hazard_auroc,
            "mean_hazard_on_danger": cov_res.mean_hazard_on_danger,
            "mean_hazard_on_safe": cov_res.mean_hazard_on_safe,
            "ranking_correlation_rho": cov_res.ranking_correlation_rho,
            "offline_top1_acc_pct": cov_res.optimal_action_top1_acc_pct,
            "online_top1_acc_pct": eval_res["online_top1_acc_pct"],
            "reflex_action_flip_rate_pct": cov_res.reflex_action_flip_rate_pct,
            "stage1_imagined_pct": cov_res.stage1_imagined_pct,
            "stage2_accurate_pct": cov_res.stage2_accurate_pct,
            "stage3_correctly_ranked_pct": cov_res.stage3_correctly_ranked_pct,
            "stage4_selected_pct": cov_res.stage4_selected_pct,
        }
        print(f"  Resource-Matched K = {k:2d} | IQM = {eval_res['iqm_return']:6.2f} | Online Top1 = {eval_res['online_top1_acc_pct']:5.1f}% | Offline Top1 = {cov_res.optimal_action_top1_acc_pct:5.1f}% | Futures = {cov_res.distinct_future_coverage:.2f}/5")

    results["resource_matched_capacity_curve"] = capacity_curve
    results["future_coverage_diagnostics"] = coverage_curve

    # 4. The 8-Stage Progressive Causal Intervention Spectrum (Distinguishing Presence from Meaning)
    print("\n--- 8-Stage Progressive Causal Intervention Spectrum (Presence vs Meaning) ---")
    donor_env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_A_MULTI_OBJECT)[0])
    donor_obs = donor_env.reset(9999)
    raw_donor = np.frombuffer(donor_obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
    donor_rgb = F.interpolate(torch.from_numpy(raw_donor).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))
    with torch.no_grad():
        donor_out = pb_model.base_brain(donor_rgb, torch.zeros((1, 307), device=device), torch.tensor([0.016], device=device))
        donor_thoughts = donor_out.next_state.thoughts

    interventions = [
        ("A_normal", "none", {}),
        ("B_permute", "permute", {}),
        ("C_register_swap", "register_swap", {}),
        ("D_stale_thoughts", "stale", {"donor_thoughts": donor_thoughts}),
        ("E_donor_thoughts", "donor", {"donor_thoughts": donor_thoughts}),
        ("F_gaussian_noise", "gaussian", {}),
        ("G_zero_active", "zero_active", {}),
        ("H_zero_knockout", "zero_knockout", {}),
    ]

    causal_results: dict[str, Any] = {}
    base_res = evaluate_closed_loop_model(pb_model, test_env, seed=42, episodes=args.episodes_per_world, device=device)
    base_iqm = base_res["iqm_return"]

    for name, mode, extra in interventions:
        eval_out = evaluate_closed_loop_model(
            pb_model,
            test_env,
            seed=42,
            episodes=args.episodes_per_world,
            device=device,
            intervention_mode=mode,
            **extra,
        )
        iqm = eval_out["iqm_return"]
        top1 = eval_out["online_top1_acc_pct"]
        drop_pct = max(0.0, (base_iqm - iqm) / (abs(base_iqm) + 1e-4) * 100.0) if mode != "none" else 0.0
        causal_results[name] = {
            "iqm_return": iqm,
            "online_top1_acc_pct": top1,
            "degradation_drop_pct": float(drop_pct),
        }
        print(f"  [{name:18s}] IQM Return = {iqm:6.2f} | Online Top1 = {top1:5.1f}% | Degradation = {drop_pct:5.2f}%")

    results["progressive_causal_interventions"] = causal_results

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Thought-Mediated Campaign results saved to {args.output_json}")


if __name__ == "__main__":
    main()
