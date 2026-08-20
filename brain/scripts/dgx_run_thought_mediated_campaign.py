"""DGX Spark Campaign Harness for Thought-Mediated Parallel Cognition.

Evaluates:
1. Direct Temporal Persistence Evaluation: Lambda=0.0 vs Lambda=0.50.
2. Granular 5-Stage Decision Decomposition (Imagined -> Disp -> Rew -> Haz -> Conf -> Ranked -> Selected).
3. Stochastic, Occluded & Information-Gathering Benchmark (Uncertainty & Probe actions).
4. Multi-Seed Scaling Curve across K in {1, 4, 8, 16, 32} and Proposal-GRU Baseline (5 Seeds).
5. 8-Stage Progressive Presence vs. Meaning Causal Spectrum.
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
from irene_brain.environments.decision_critical_suite import (
    DecisionCriticalReport,
    MultiHypothesisHorizonReport,
    evaluate_decision_critical_benchmark,
    evaluate_multi_hypothesis_horizon_benchmark,
)
from irene_brain.environments.stochastic_occluded_benchmark import (
    StochasticOccludedReport,
    evaluate_stochastic_occluded_benchmark,
)
from irene_brain.evaluation.temporal_persistence_diagnostics import (
    TemporalPersistenceReport,
    evaluate_temporal_persistence,
)
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.thought_mediated_model import (
    ProposalGRUBaseline,
    ThoughtMediatedBrainModel,
    build_resource_matched_thought_model,
)
from irene_brain.evaluation.future_coverage_diagnostics import (
    FutureCoverageReport,
    evaluate_future_coverage,
)
from irene_brain.evaluation.counterfactual_transplant_diagnostics import (
    CounterfactualTransplantReport,
    evaluate_counterfactual_thought_transplant,
)
from irene_brain.training.staged_branch_curriculum import (
    ComprehensiveBranchBundle,
    MultiHypothesisBranchLoss,
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


def train_thought_mediated_model(
    model: ThoughtMediatedBrainModel,
    device: torch.device,
    training_steps: int = 300,
    lr: float = 5e-4,
    inertia_weight: float = 0.5,
) -> None:
    """Balanced Multi-Family On-Policy DAgger training with temporal matching inertia."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = MultiHypothesisBranchLoss(inertia_weight=inertia_weight).to(device)

    all_families = list(TaskFamily)
    envs = [Phase2TaskEnvironment(make_family_suite(fam)[0]) for fam in all_families]
    obs_list = [env.reset(1000 + idx) for idx, env in enumerate(envs)]

    for step in range(training_steps):
        rgb_tensors = []
        bundles = []

        for idx, (env, obs) in enumerate(zip(envs, obs_list)):
            raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor.shape[-1] != 32:
                rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")
            rgb_tensors.append(rgb_tensor)

            bundle = generate_multi_step_trajectory_tree(env, obs, horizon=2, device=device)
            bundles.append(bundle)

        rgb_batch = torch.cat(rgb_tensors, dim=0)
        ctrl_batch = torch.zeros((len(envs), 307), device=device)
        dt_batch = torch.tensor([0.016667] * len(envs), device=device)

        out = model(rgb_batch, ctrl_batch, dt_batch, max_cycles=model.config.cognitive_cycles)
        total_loss, _metrics = loss_fn(out.action.proposals, bundles)

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            button_logits = out.action.button_logits.cpu().numpy()
            for idx, (env, obs) in enumerate(zip(envs, obs_list)):
                b_logits = button_logits[idx]
                w_logit = float(b_logits[int(HidKey.W)])
                s_logit = float(b_logits[int(HidKey.S)])
                a_logit = float(b_logits[int(HidKey.A)])
                d_logit = float(b_logits[int(HidKey.D)])

                keys = []
                dir_scores = {"W": w_logit, "S": s_logit, "A": a_logit, "D": d_logit}
                best_dir, best_score = max(dir_scores.items(), key=lambda x: x[1])
                if best_score > 0.0:
                    if best_dir == "W": keys.append(int(HidKey.W))
                    elif best_dir == "S": keys.append(int(HidKey.S))
                    elif best_dir == "A": keys.append(int(HidKey.A))
                    elif best_dir == "D": keys.append(int(HidKey.D))

                action_ctrl = GenericControl(mouse_dx=0.0, mouse_dy=0.0, keys_down=tuple(keys))
                outcome = env.step(action_ctrl)
                if outcome.terminated or outcome.truncated:
                    obs_list[idx] = env.reset(step * 10 + idx)
                else:
                    obs_list[idx] = outcome.observation


def train_proposal_gru_baseline(
    model: ProposalGRUBaseline,
    device: torch.device,
    training_steps: int = 300,
    lr: float = 5e-4,
) -> None:
    """Balanced Multi-Family training for matched Proposal-GRU baseline."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = MultiHypothesisBranchLoss().to(device)

    all_families = list(TaskFamily)
    envs = [Phase2TaskEnvironment(make_family_suite(fam)[0]) for fam in all_families]
    obs_list = [env.reset(2000 + idx) for idx, env in enumerate(envs)]

    for step in range(training_steps):
        rgb_tensors = []
        bundles = []

        for idx, (env, obs) in enumerate(zip(envs, obs_list)):
            raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor.shape[-1] != 32:
                rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")
            rgb_tensors.append(rgb_tensor)

            bundle = generate_multi_step_trajectory_tree(env, obs, horizon=2, device=device)
            bundles.append(bundle)

        rgb_batch = torch.cat(rgb_tensors, dim=0)
        ctrl_batch = torch.zeros((len(envs), 307), device=device)
        dt_batch = torch.tensor([0.016667] * len(envs), device=device)

        out = model(rgb_batch, ctrl_batch, dt_batch)
        total_loss, _metrics = loss_fn(out.action.proposals, bundles)

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            button_logits = out.action.button_logits.cpu().numpy()
            for idx, (env, obs) in enumerate(zip(envs, obs_list)):
                b_logits = button_logits[idx]
                w_logit = float(b_logits[int(HidKey.W)])
                s_logit = float(b_logits[int(HidKey.S)])
                a_logit = float(b_logits[int(HidKey.A)])
                d_logit = float(b_logits[int(HidKey.D)])

                keys = []
                dir_scores = {"W": w_logit, "S": s_logit, "A": a_logit, "D": d_logit}
                best_dir, best_score = max(dir_scores.items(), key=lambda x: x[1])
                if best_score > 0.0:
                    if best_dir == "W": keys.append(int(HidKey.W))
                    elif best_dir == "S": keys.append(int(HidKey.S))
                    elif best_dir == "A": keys.append(int(HidKey.A))
                    elif best_dir == "D": keys.append(int(HidKey.D))

                action_ctrl = GenericControl(mouse_dx=0.0, mouse_dy=0.0, keys_down=tuple(keys))
                outcome = env.step(action_ctrl)
                if outcome.terminated or outcome.truncated:
                    obs_list[idx] = env.reset(step * 10 + idx)
                else:
                    obs_list[idx] = outcome.observation


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
            rgb_tensor = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
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
    print(f"=== Starting Comprehensive Multi-Family Campaign on {device} ===")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

    seeds = [42, 43, 44, 45, 46]
    test_env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_B_PURSUIT_EVASION)[0])

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": str(device),
        "seeds": seeds,
        "temporal_persistence_comparison": {},
        "per_k_decision_decomposition": {},
        "stochastic_occluded_benchmark": {},
        "multi_seed_scaling_curve": {},
        "progressive_causal_interventions": {},
    }

    # 1. Direct Hypothesis Persistence Diagnostic: Lambda=0.0 vs Lambda=0.50
    print("\n--- 1. Quantitative Evaluation of Temporal Hypothesis Persistence ---")
    pb_test = build_resource_matched_thought_model(32).to(device)
    train_thought_mediated_model(pb_test, device=device, training_steps=300, inertia_weight=0.5)

    p_rep_no_inertia = evaluate_temporal_persistence(pb_test, inertia_weight=0.0, device=device)
    p_rep_with_inertia = evaluate_temporal_persistence(pb_test, inertia_weight=0.5, device=device)

    results["temporal_persistence_comparison"] = {
        "no_inertia_lambda_0": {
            "retention_rate_pct": p_rep_no_inertia.branch_identity_retention_rate_pct,
            "reassignment_rate_pct": p_rep_no_inertia.slot_reassignment_rate_pct,
            "mean_lifetime_ticks": p_rep_no_inertia.mean_hypothesis_lifetime_ticks,
            "unnecessary_switch_rate_pct": p_rep_no_inertia.unnecessary_switch_rate_pct,
        },
        "with_inertia_lambda_0_5": {
            "retention_rate_pct": p_rep_with_inertia.branch_identity_retention_rate_pct,
            "reassignment_rate_pct": p_rep_with_inertia.slot_reassignment_rate_pct,
            "mean_lifetime_ticks": p_rep_with_inertia.mean_hypothesis_lifetime_ticks,
            "unnecessary_switch_rate_pct": p_rep_with_inertia.unnecessary_switch_rate_pct,
        },
    }
    print(f"  [No Inertia (Lambda=0.0)] Retention = {p_rep_no_inertia.branch_identity_retention_rate_pct:5.1f}% | Lifetime = {p_rep_no_inertia.mean_hypothesis_lifetime_ticks:4.1f} ticks | Unnec Switches = {p_rep_no_inertia.unnecessary_switch_rate_pct:5.1f}%")
    print(f"  [With Inertia (Lambda=0.5)] Retention = {p_rep_with_inertia.branch_identity_retention_rate_pct:5.1f}% | Lifetime = {p_rep_with_inertia.mean_hypothesis_lifetime_ticks:4.1f} ticks | Unnec Switches = {p_rep_with_inertia.unnecessary_switch_rate_pct:5.1f}%")

    # 2. Granular 5-Stage Decision Decomposition across K in {0, 1, 4, 5, 8, 16, 32}
    print("\n--- 2. Granular 5-Stage Decision Decomposition across K ---")
    k_list = [0, 1, 4, 5, 8, 16, 32]
    decomp_results: dict[int, dict[str, Any]] = {}

    for k in k_list:
        if k == 0:
            m = build_resource_matched_thought_model(1).to(device)
            rep = evaluate_future_coverage(m, test_env, device=device, active_slots=0)
        else:
            m = build_resource_matched_thought_model(k).to(device)
            train_thought_mediated_model(m, device=device, training_steps=300)
            rep = evaluate_future_coverage(m, test_env, device=device)

        decomp_results[k] = {
            "distinct_futures": rep.distinct_future_coverage,
            "stage1_imagined_pct": rep.stage1_imagined_pct,
            "stage2a_disp_pct": rep.stage2a_displacement_pct,
            "stage2b_rew_pct": rep.stage2b_reward_pct,
            "stage2c_haz_pct": rep.stage2c_hazard_pct,
            "stage2d_conf_pct": rep.stage2d_confidence_pct,
            "stage2_accurate_pct": rep.stage2_accurate_pct,
            "stage3_ranked_pct": rep.stage3_correctly_ranked_pct,
            "stage4_selected_pct": rep.stage4_selected_pct,
            "duplicate_slot_distribution": rep.duplicate_slot_distribution,
        }
        print(f"  K = {k:2d} | Futures = {rep.distinct_future_coverage:.2f}/5 | S1 (Imagined): {rep.stage1_imagined_pct:5.1f}% | S2a (Disp): {rep.stage2a_displacement_pct:5.1f}% | S2b (Rew): {rep.stage2b_reward_pct:5.1f}% | S2c (Haz): {rep.stage2c_hazard_pct:5.1f}% | S3 (Rank): {rep.stage3_correctly_ranked_pct:5.1f}% | S4 (Select): {rep.stage4_selected_pct:5.1f}%")

    results["per_k_decision_decomposition"] = decomp_results

    # 3. Stochastic, Occluded & Information-Gathering Benchmark across K and GRU (5 Seeds)
    print("\n--- 3. Stochastic, Occluded & Information-Gathering Benchmark (5 Seeds) ---")
    stoch_results: dict[str, dict[str, Any]] = {}

    for k in [1, 4, 8, 16, 32]:
        seed_accs = []
        seed_gamble_avoids = []
        seed_scores = []

        for seed in seeds:
            torch.manual_seed(seed)
            m = build_resource_matched_thought_model(k).to(device)
            train_thought_mediated_model(m, device=device, training_steps=300)
            rep = evaluate_stochastic_occluded_benchmark(m, base_seed=seed * 100, device=device)
            seed_accs.append(rep.stochastic_decision_accuracy_pct)
            seed_gamble_avoids.append(rep.greedy_gamble_avoidance_pct)
            seed_scores.append(rep.cumulative_expected_utility_score)

        stoch_results[f"K={k}"] = {
            "mean_accuracy_pct": float(np.mean(seed_accs)),
            "std_accuracy_pct": float(np.std(seed_accs)),
            "mean_gamble_avoidance_pct": float(np.mean(seed_gamble_avoids)),
            "mean_expected_utility_score": float(np.mean(seed_scores)),
        }
        print(f"  Pseudo-Brain K = {k:2d} | Stoch Acc = {np.mean(seed_accs):5.1f}% +/- {np.std(seed_accs):4.1f}% | Gamble Avoid = {np.mean(seed_gamble_avoids):5.1f}% | Exp Utility = {np.mean(seed_scores):+6.1f}")

    # Evaluate Proposal-GRU Baseline across 5 Seeds
    gru_seed_accs = []
    gru_seed_gamble_avoids = []
    gru_seed_scores = []
    for seed in seeds:
        torch.manual_seed(seed)
        gm = ProposalGRUBaseline().to(device)
        train_proposal_gru_baseline(gm, device=device, training_steps=300)
        rep = evaluate_stochastic_occluded_benchmark(gm, base_seed=seed * 100, device=device)
        gru_seed_accs.append(rep.stochastic_decision_accuracy_pct)
        gru_seed_gamble_avoids.append(rep.greedy_gamble_avoidance_pct)
        gru_seed_scores.append(rep.cumulative_expected_utility_score)

    stoch_results["Proposal-GRU"] = {
        "mean_accuracy_pct": float(np.mean(gru_seed_accs)),
        "std_accuracy_pct": float(np.std(gru_seed_accs)),
        "mean_gamble_avoidance_pct": float(np.mean(gru_seed_gamble_avoids)),
        "mean_expected_utility_score": float(np.mean(gru_seed_scores)),
    }
    print(f"  Proposal-GRU Match | Stoch Acc = {np.mean(gru_seed_accs):5.1f}% +/- {np.std(gru_seed_accs):4.1f}% | Gamble Avoid = {np.mean(gru_seed_gamble_avoids):5.1f}% | Exp Utility = {np.mean(gru_seed_scores):+6.1f}")

    results["stochastic_occluded_benchmark"] = stoch_results

    # 4. Multi-Seed Scaling Curve on Standard Tasks
    print("\n--- 4. Multi-Seed Evaluation on Standard Tasks ---")
    scaling_curve: dict[str, Any] = {}
    for k in [1, 4, 8, 16, 32]:
        m = build_resource_matched_thought_model(k).to(device)
        train_thought_mediated_model(m, device=device, training_steps=300)
        crit_rep = evaluate_decision_critical_benchmark(m, device=device)
        scaling_curve[f"K={k}"] = {
            "decision_critical_score": crit_rep.total_decision_score,
            "decision_critical_acc_pct": crit_rep.decision_accuracy_pct,
        }

    results["multi_seed_scaling_curve"] = scaling_curve

    # 5. The 8-Stage Progressive Causal Intervention Spectrum (Presence vs Meaning)
    print("\n--- 5. 8-Stage Progressive Causal Intervention Spectrum (Presence vs Meaning) ---")
    pb_model = build_resource_matched_thought_model(32).to(device)
    train_thought_mediated_model(pb_model, device=device, training_steps=300)

    donor_env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_A_MULTI_OBJECT)[0])
    donor_obs = donor_env.reset(9999)
    raw_donor = np.frombuffer(donor_obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
    donor_rgb = F.interpolate(torch.from_numpy(raw_donor.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0, size=(32, 32))
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
    for name, mode, extra in interventions:
        eval_out = evaluate_closed_loop_model(pb_model, test_env, seed=42, episodes=args.episodes_per_world, device=device, intervention_mode=mode, **extra)
        crit_out = evaluate_decision_critical_benchmark(pb_model, device=device, intervention_mode=mode, **extra)
        causal_results[name] = {
            "iqm_return": eval_out["iqm_return"],
            "online_top1_acc_pct": eval_out["online_top1_acc_pct"],
            "decision_critical_score": crit_out.total_decision_score,
            "decision_critical_acc_pct": crit_out.decision_accuracy_pct,
        }
        print(f"  [{name:18s}] DecCrit = {crit_out.total_decision_score:+5.1f} ({crit_out.decision_accuracy_pct:5.1f}%) | IQM = {eval_out['iqm_return']:6.2f} | Online Top1 = {eval_out['online_top1_acc_pct']:5.1f}%")

    results["progressive_causal_interventions"] = causal_results

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Thought-Mediated Campaign results saved to {args.output_json}")


if __name__ == "__main__":
    main()
