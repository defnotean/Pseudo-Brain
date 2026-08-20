"""DGX Spark Campaign Harness for Thought-Mediated Parallel Cognition.

Evaluates:
1. Direct Hypothesis Persistence Diagnostic: Lambda=0.0 vs Lambda=0.50.
2. Granular 5-Stage Decision Decomposition (Imagined -> Disp -> Rew -> Haz -> Conf -> Ranked -> Selected).
3. Belief Collapse & Value of Information (VoI) Diagnostics (Pre/Post Entropy, Posterior Updates, Action Breakdown).
4. Multi-Seed Stochastic & Sequential Probe Resolution Benchmark across K in {1, 4, 8, 16, 32} vs Proposal-GRU Baseline (5 Seeds).
5. 8-Stage Progressive Presence vs. Meaning Causal Spectrum.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
import random
import sys
import time
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

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
    SequentialProbeResolutionReport,
    evaluate_stochastic_occluded_benchmark,
    evaluate_sequential_probe_resolution,
)
from irene_brain.environments.multi_horizon_ambiguity_arena import (
    MultiHorizonArenaReport,
    DelayedResolutionReport,
    NestedUncertaintyReport,
    NonStationaryShiftReport,
    evaluate_multi_horizon_ambiguity_arena,
    evaluate_delayed_resolution_benchmark,
    evaluate_nested_uncertainty_benchmark,
    evaluate_non_stationary_regime_shift_benchmark,
)
from irene_brain.evaluation.temporal_persistence_diagnostics import (
    TemporalPersistenceReport,
    evaluate_temporal_persistence,
)
from irene_brain.evaluation.belief_collapse_diagnostics import (
    BeliefCollapseReport,
    evaluate_belief_collapse_and_voi,
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
    generate_comprehensive_branch_bundle,
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
    training_steps: int = 400,
    lr: float = 5e-4,
    inertia_weight: float = 0.5,
) -> None:
    """Balanced Multi-Family 2-step recurrent unroll training with VoI loss."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = MultiHypothesisBranchLoss(inertia_weight=inertia_weight).to(device)

    all_families = list(TaskFamily)
    envs = [Phase2TaskEnvironment(make_family_suite(fam)[0]) for fam in all_families]
    obs_list = [env.reset(1000 + idx) for idx, env in enumerate(envs)]

    for step in range(training_steps):
        # Step 0: Initial Observation
        rgb_tensors0 = []
        bundles0 = []

        for idx, (env, obs) in enumerate(zip(envs, obs_list)):
            raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor.shape[-1] != 32:
                rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")
            rgb_tensors0.append(rgb_tensor)

            bundle = generate_comprehensive_branch_bundle(env, obs, device=device)
            bundles0.append(bundle)

        rgb_batch0 = torch.cat(rgb_tensors0, dim=0)
        ctrl_batch0 = torch.zeros((len(envs), 307), device=device)
        dt_batch0 = torch.tensor([0.016667] * len(envs), device=device)

        out0 = model(rgb_batch0, ctrl_batch0, dt_batch0, max_cycles=model.config.cognitive_cycles)
        loss0, _ = loss_fn(out0.action.proposals, bundles0)

        # Step 1: Environment step based on policy intent to advance 1 tick
        rgb_tensors1 = []
        bundles1 = []
        button_logits0 = out0.action.button_logits.detach().cpu().numpy()

        for idx, (env, obs) in enumerate(zip(envs, obs_list)):
            b_logits = button_logits0[idx]
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
                obs_next = env.reset(step * 10 + idx)
            else:
                obs_next = outcome.observation
            obs_list[idx] = obs_next

            raw_rgb1 = np.frombuffer(obs_next.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor1 = torch.from_numpy(raw_rgb1.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor1.shape[-1] != 32:
                rgb_tensor1 = F.interpolate(rgb_tensor1, size=(32, 32), mode="nearest")
            rgb_tensors1.append(rgb_tensor1)

            bundle1 = generate_comprehensive_branch_bundle(env, obs_next, device=device)
            bundles1.append(bundle1)

        rgb_batch1 = torch.cat(rgb_tensors1, dim=0)
        out1 = model(rgb_batch1, ctrl_batch0, dt_batch0, state=out0.next_state, max_cycles=model.config.cognitive_cycles)
        loss1, _ = loss_fn(out1.action.proposals, bundles1)

        total_loss = loss0 + loss1
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def train_proposal_gru_baseline(
    model: ProposalGRUBaseline,
    device: torch.device,
    training_steps: int = 400,
    lr: float = 5e-4,
) -> None:
    """Balanced Multi-Family 2-step recurrent unroll training for Proposal-GRU baseline."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = MultiHypothesisBranchLoss().to(device)

    all_families = list(TaskFamily)
    envs = [Phase2TaskEnvironment(make_family_suite(fam)[0]) for fam in all_families]
    obs_list = [env.reset(2000 + idx) for idx, env in enumerate(envs)]

    for step in range(training_steps):
        # Step 0
        rgb_tensors0 = []
        bundles0 = []

        for idx, (env, obs) in enumerate(zip(envs, obs_list)):
            raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor.shape[-1] != 32:
                rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")
            rgb_tensors0.append(rgb_tensor)

            bundle = generate_comprehensive_branch_bundle(env, obs, device=device)
            bundles0.append(bundle)

        rgb_batch0 = torch.cat(rgb_tensors0, dim=0)
        ctrl_batch0 = torch.zeros((len(envs), 307), device=device)
        dt_batch0 = torch.tensor([0.016667] * len(envs), device=device)

        out0 = model(rgb_batch0, ctrl_batch0, dt_batch0)
        loss0, _ = loss_fn(out0.action.proposals, bundles0)

        # Step 1
        rgb_tensors1 = []
        bundles1 = []
        button_logits0 = out0.action.button_logits.detach().cpu().numpy()

        for idx, (env, obs) in enumerate(zip(envs, obs_list)):
            b_logits = button_logits0[idx]
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
                obs_next = env.reset(step * 10 + idx)
            else:
                obs_next = outcome.observation
            obs_list[idx] = obs_next

            raw_rgb1 = np.frombuffer(obs_next.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor1 = torch.from_numpy(raw_rgb1.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor1.shape[-1] != 32:
                rgb_tensor1 = F.interpolate(rgb_tensor1, size=(32, 32), mode="nearest")
            rgb_tensors1.append(rgb_tensor1)

            bundle1 = generate_comprehensive_branch_bundle(env, obs_next, device=device)
            bundles1.append(bundle1)

        rgb_batch1 = torch.cat(rgb_tensors1, dim=0)
        out1 = model(rgb_batch1, ctrl_batch0, dt_batch0, state=out0.next_state)
        loss1, _ = loss_fn(out1.action.proposals, bundles1)

        total_loss = loss0 + loss1
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def evaluate_closed_loop_model(
    model: nn.Module,
    env: Phase2TaskEnvironment,
    seed: int,
    episodes: int = 5,
    device: torch.device = torch.device("cpu"),
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
                bundle = generate_comprehensive_branch_bundle(env, obs, device=device)
                gt_opt_action = int(torch.argmax(bundle.expected_action_utilities).item())
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


def run_cpu_diagnostic_smoke() -> dict[str, Any]:
    """Tiny CPU-only Phase 2.5 diagnostic path. Does not train and does not use GPU."""
    device = torch.device("cpu")
    print("=== Phase 2.5 CPU diagnostic smoke (no training, no GPU) ===")
    model = build_resource_matched_thought_model(4)
    gru = ProposalGRUBaseline()

    belief = evaluate_belief_collapse_and_voi(model, num_episodes=4, device=device)
    arena = evaluate_multi_horizon_ambiguity_arena(model, num_episodes=3, device=device)
    delayed = evaluate_delayed_resolution_benchmark(model, delay_ticks=6, num_episodes=3, device=device)
    nested = evaluate_nested_uncertainty_benchmark(model, num_episodes=3, device=device)
    shift = evaluate_non_stationary_regime_shift_benchmark(model, episode_length=12, shift_tick=6, device=device)
    probe = evaluate_sequential_probe_resolution(model, num_episodes=4, device=device)
    gru_arena = evaluate_multi_horizon_ambiguity_arena(gru, num_episodes=2, device=device)

    results = {
        "device": "cpu",
        "belief_collapse": {
            "pre_observation_entropy": belief.pre_observation_entropy,
            "post_observation_entropy": belief.post_observation_entropy,
            "voi_regret": belief.voi_regret,
            "probe_wait_pct": belief.probe_wait_pct,
        },
        "multi_horizon": {
            "staggered_survival_rate_pct": arena.staggered_survival_rate_pct,
            "mean_multi_horizon_return": arena.mean_multi_horizon_return,
        },
        "delayed_resolution": {
            "delayed_decision_acc_pct": delayed.delayed_decision_acc_pct,
            "premature_collapse_rate_pct": delayed.premature_collapse_rate_pct,
        },
        "nested_uncertainty": {
            "compound_goal_acc_pct": nested.compound_goal_acc_pct,
            "partial_collapse_entropy_bits": nested.partial_collapse_entropy_bits,
        },
        "non_stationary_shift": {
            "pre_shift_accuracy_pct": shift.pre_shift_accuracy_pct,
            "post_shift_accuracy_pct": shift.post_shift_accuracy_pct,
        },
        "sequential_probe": {
            "probe_at_t0_rate_pct": probe.probe_at_t0_rate_pct,
            "mean_episode_return": probe.mean_episode_return,
        },
        "gru_multi_horizon_return": gru_arena.mean_multi_horizon_return,
    }
    print(
        f"  Belief H {belief.pre_observation_entropy:.2f}->{belief.post_observation_entropy:.2f} | "
        f"VoI regret {belief.voi_regret:.2f} | Arena return {arena.mean_multi_horizon_return:+.2f} | "
        f"Delayed acc {delayed.delayed_decision_acc_pct:.1f}%"
    )
    print("[OK] CPU diagnostic smoke finished. Full K-sweep training remains Spark-only.")
    return results


def main() -> None:
    hidden_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-1"
    default_device = "cpu" if hidden_cuda or not torch.cuda.is_available() else "cuda:0"
    parser = argparse.ArgumentParser(description="Thought-Mediated Parallel Cognition Campaign")
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--episodes-per-world", type=int, default=5)
    parser.add_argument("--train-steps", type=int, default=400)
    parser.add_argument("--output-json", type=str, default="docs/phase_closure/thought_mediated_campaign_results.json")
    parser.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="Tiny CPU diagnostic evals only. No training, no GPU, not the Spark campaign.",
    )
    args = parser.parse_args()

    if args.cpu_smoke:
        results = run_cpu_diagnostic_smoke()
        default_json = "docs/phase_closure/thought_mediated_campaign_results.json"
        if args.output_json and args.output_json != default_json:
            out_dir = os.path.dirname(args.output_json)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output_json, "w") as handle:
                json.dump(results, handle, indent=2)
            print(f"[OK] CPU diagnostic smoke saved to {args.output_json}")
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        print("This is the Spark campaign harness. Do not launch it from the Windows workstation.")
    print(f"=== Starting Comprehensive Multi-Family Campaign on {device} ===")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

    # Print Resource Matching Table
    print("\n" + "=" * 90)
    print(f"{'Model Architecture':24s} | {'Params':10s} | {'Core Width':10s} | {'K':4s} | {'Decoder Outputs':20s} | {'Approx FLOPs':12s}")
    print("-" * 90)
    for k in [1, 4, 8, 16, 32]:
        m = build_resource_matched_thought_model(k).to(device)
        total_p = sum(p.numel() for p in m.parameters())
        w = m.config.core_width
        dec_out = f"{k}x(5+4+2) branch"
        flops = f"~{total_p * 2 / 1e6:.2f} M"
        print(f"{f'Pseudo-Brain K={k}':24s} | {total_p:10,d} | {w:10d} | {k:4d} | {dec_out:20s} | {flops:12s}")

    gru_m = ProposalGRUBaseline().to(device)
    gru_p = sum(p.numel() for p in gru_m.parameters())
    print(f"{'Proposal-GRU Baseline':24s} | {gru_p:10,d} | {gru_m.hidden_dim:10d} | {gru_m.num_proposals:4d} | {f'{gru_m.num_proposals} branch heads':20s} | {f'~{gru_p * 2 / 1e6:.2f} M':12s}")
    print("=" * 90 + "\n")

    seeds = [42, 43, 44, 45, 46]
    test_env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_B_PURSUIT_EVASION)[0])

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": str(device),
        "seeds": seeds,
        "temporal_persistence_comparison": {},
        "per_k_decision_decomposition": {},
        "belief_collapse_diagnostics": {},
        "stochastic_occluded_benchmark": {},
        "sequential_probe_resolution_benchmark": {},
        "progressive_causal_interventions": {},
    }

    # 1. Direct Hypothesis Persistence Diagnostic: Lambda=0.0 vs Lambda=0.50
    print("\n--- 1. Quantitative Evaluation of Temporal Hypothesis Persistence ---")
    pb_test = build_resource_matched_thought_model(32).to(device)
    train_thought_mediated_model(pb_test, device=device, training_steps=args.train_steps, inertia_weight=0.5)

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
            train_thought_mediated_model(m, device=device, training_steps=args.train_steps)
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

    # 3. Belief Collapse & Value of Information (VoI) Diagnostics
    print("\n--- 3. Belief Collapse & Value of Information Diagnostics ---")
    belief_results: dict[str, Any] = {}
    for k in [1, 8, 32]:
        m = build_resource_matched_thought_model(k).to(device)
        train_thought_mediated_model(m, device=device, training_steps=args.train_steps)
        b_rep = evaluate_belief_collapse_and_voi(m, num_episodes=30, device=device)
        belief_results[f"K={k}"] = {
            "pre_obs_entropy": b_rep.pre_observation_entropy,
            "post_obs_entropy": b_rep.post_observation_entropy,
            "entropy_reduction_pct": b_rep.entropy_reduction_pct,
            "posterior_accuracy_pct": b_rep.posterior_accuracy_pct,
            "disproven_suppression_pct": b_rep.disproven_suppression_pct,
            "probe_wait_pct": b_rep.probe_wait_pct,
            "robust_detour_pct": b_rep.robust_detour_pct,
            "blind_gamble_pct": b_rep.blind_gamble_pct,
            "wrong_safe_pct": b_rep.wrong_safe_pct,
            "voi_regret": b_rep.voi_regret,
        }
        print(f"  PB K = {k:2d} | Pre H: {b_rep.pre_observation_entropy:.2f} -> Post H: {b_rep.post_observation_entropy:.2f} (Red: {b_rep.entropy_reduction_pct:4.1f}%) | Probe/WAIT: {b_rep.probe_wait_pct:4.1f}% | Gamble: {b_rep.blind_gamble_pct:4.1f}% | Detour: {b_rep.robust_detour_pct:4.1f}% | Wrong Safe: {b_rep.wrong_safe_pct:4.1f}%")

    results["belief_collapse_diagnostics"] = belief_results

    # 4. Multi-Seed Stochastic Benchmark across K and GRU (5 Seeds)
    print("\n--- 4. Stochastic, Occluded & Information-Gathering Benchmark (5 Seeds) ---")
    stoch_results: dict[str, dict[str, Any]] = {}

    for k in [1, 4, 8, 16, 32]:
        seed_accs = []
        seed_gamble_avoids = []
        seed_scores = []

        for seed in seeds:
            torch.manual_seed(seed)
            m = build_resource_matched_thought_model(k).to(device)
            train_thought_mediated_model(m, device=device, training_steps=args.train_steps)
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
        train_proposal_gru_baseline(gm, device=device, training_steps=args.train_steps)
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

    # 5. Sequential Probe Resolution Benchmark (2-Step Ambiguity Arena across 5 Seeds)
    print("\n--- 5. Sequential Probe Resolution Benchmark (Ambiguity -> Probe -> Resolve) ---")
    seq_probe_results: dict[str, dict[str, Any]] = {}

    for k in [1, 4, 8, 16, 32]:
        probe_t0_list = []
        resolve_t1_list = []
        blind_death_list = []
        ret_list = []
        voi_realized_list = []

        for seed in seeds:
            torch.manual_seed(seed)
            m = build_resource_matched_thought_model(k).to(device)
            train_thought_mediated_model(m, device=device, training_steps=args.train_steps)
            rep = evaluate_sequential_probe_resolution(m, num_episodes=40, base_seed=seed * 200, device=device)
            probe_t0_list.append(rep.probe_at_t0_rate_pct)
            resolve_t1_list.append(rep.posterior_resolution_accuracy_pct)
            blind_death_list.append(rep.blind_gamble_death_rate_pct)
            ret_list.append(rep.mean_episode_return)
            voi_realized_list.append(rep.probe_value_realized_pct)

        seq_probe_results[f"K={k}"] = {
            "mean_probe_t0_pct": float(np.mean(probe_t0_list)),
            "mean_resolve_t1_pct": float(np.mean(resolve_t1_list)),
            "mean_blind_death_pct": float(np.mean(blind_death_list)),
            "mean_return": float(np.mean(ret_list)),
            "mean_voi_realized_pct": float(np.mean(voi_realized_list)),
        }
        print(f"  Pseudo-Brain K = {k:2d} | Probe@t0: {np.mean(probe_t0_list):5.1f}% | Resolve@t1: {np.mean(resolve_t1_list):5.1f}% | Blind Deaths: {np.mean(blind_death_list):5.1f}% | Return: {np.mean(ret_list):+5.2f} | VoI Captured: {np.mean(voi_realized_list):5.1f}%")

    # GRU Sequential Probe Benchmark
    gru_probe_t0 = []
    gru_resolve_t1 = []
    gru_blind_death = []
    gru_rets = []
    gru_vois = []
    for seed in seeds:
        torch.manual_seed(seed)
        gm = ProposalGRUBaseline().to(device)
        train_proposal_gru_baseline(gm, device=device, training_steps=args.train_steps)
        rep = evaluate_sequential_probe_resolution(gm, num_episodes=40, base_seed=seed * 200, device=device)
        gru_probe_t0.append(rep.probe_at_t0_rate_pct)
        gru_resolve_t1.append(rep.posterior_resolution_accuracy_pct)
        gru_blind_death.append(rep.blind_gamble_death_rate_pct)
        gru_rets.append(rep.mean_episode_return)
        gru_vois.append(rep.probe_value_realized_pct)

    seq_probe_results["Proposal-GRU"] = {
        "mean_probe_t0_pct": float(np.mean(gru_probe_t0)),
        "mean_resolve_t1_pct": float(np.mean(gru_resolve_t1)),
        "mean_blind_death_pct": float(np.mean(gru_blind_death)),
        "mean_return": float(np.mean(gru_rets)),
        "mean_voi_realized_pct": float(np.mean(gru_vois)),
    }
    print(f"  Proposal-GRU Match | Probe@t0: {np.mean(gru_probe_t0):5.1f}% | Resolve@t1: {np.mean(gru_resolve_t1):5.1f}% | Blind Deaths: {np.mean(gru_blind_death):5.1f}% | Return: {np.mean(gru_rets):+5.2f} | VoI Captured: {np.mean(gru_vois):5.1f}%")

    results["sequential_probe_resolution_benchmark"] = seq_probe_results

    # 6. The 8-Stage Progressive Causal Intervention Spectrum (Presence vs Meaning)
    print("\n--- 6. 8-Stage Progressive Causal Intervention Spectrum (Presence vs Meaning) ---")
    pb_model = build_resource_matched_thought_model(32).to(device)
    train_thought_mediated_model(pb_model, device=device, training_steps=args.train_steps)

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

    # 7. Multi-Horizon Multi-Uncertainty Ambiguity Arena (H=4 Staggered Hazards & Asymmetric Risk)
    print("\n--- 7. Multi-Horizon Multi-Uncertainty Ambiguity Arena (H=4) (3 Seeds) ---")
    multi_horizon_results: dict[str, Any] = {}
    seeds = [40000, 40001, 40002]

    for k in [1, 4, 8, 16, 32]:
        surv_list = []
        avoid_list = []
        wm_list = []
        ret_list = []
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            m = build_resource_matched_thought_model(k).to(device)
            train_thought_mediated_model(m, device=device, training_steps=args.train_steps)
            mh_rep = evaluate_multi_horizon_ambiguity_arena(m, num_episodes=30, base_seed=s, device=device)
            surv_list.append(mh_rep.staggered_survival_rate_pct)
            avoid_list.append(mh_rep.catastrophic_branch_avoidance_pct)
            wm_list.append(mh_rep.working_memory_retention_score)
            ret_list.append(mh_rep.mean_multi_horizon_return)

        mean_ret = float(np.mean(ret_list))
        ci_lo, ci_hi = compute_bootstrap_ci(ret_list)
        multi_horizon_results[f"Pseudo-Brain K={k}"] = {
            "staggered_survival_rate_pct": float(np.mean(surv_list)),
            "catastrophic_branch_avoidance_pct": float(np.mean(avoid_list)),
            "working_memory_retention_score": float(np.mean(wm_list)),
            "mean_multi_horizon_return": mean_ret,
            "bootstrap_ci_95": [ci_lo, ci_hi],
        }
        print(f"  Pseudo-Brain K = {k:2d} | Staggered Surv: {np.mean(surv_list):5.1f}% | Trap Avoid: {np.mean(avoid_list):5.1f}% | Return: {mean_ret:+5.2f} (95% CI: [{ci_lo:+5.2f}, {ci_hi:+5.2f}])")

    gru_surv = []
    gru_avoid = []
    gru_wm = []
    gru_ret = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        gru_m = ProposalGRUBaseline().to(device)
        train_proposal_gru_baseline(gru_m, device=device, training_steps=args.train_steps)
        gru_mh_rep = evaluate_multi_horizon_ambiguity_arena(gru_m, num_episodes=30, base_seed=s, device=device)
        gru_surv.append(gru_mh_rep.staggered_survival_rate_pct)
        gru_avoid.append(gru_mh_rep.catastrophic_branch_avoidance_pct)
        gru_wm.append(gru_mh_rep.working_memory_retention_score)
        gru_ret.append(gru_mh_rep.mean_multi_horizon_return)

    mean_gru_ret = float(np.mean(gru_ret))
    gru_ci_lo, gru_ci_hi = compute_bootstrap_ci(gru_ret)
    multi_horizon_results["Proposal-GRU Match"] = {
        "staggered_survival_rate_pct": float(np.mean(gru_surv)),
        "catastrophic_branch_avoidance_pct": float(np.mean(gru_avoid)),
        "working_memory_retention_score": float(np.mean(gru_wm)),
        "mean_multi_horizon_return": mean_gru_ret,
        "bootstrap_ci_95": [gru_ci_lo, gru_ci_hi],
    }
    print(f"  Proposal-GRU Match | Staggered Surv: {np.mean(gru_surv):5.1f}% | Trap Avoid: {np.mean(gru_avoid):5.1f}% | Return: {mean_gru_ret:+5.2f} (95% CI: [{gru_ci_lo:+5.2f}, {gru_ci_hi:+5.2f}])")
    results["multi_horizon_ambiguity_arena"] = multi_horizon_results

    # 8. Hostile Causal Ablation Controls (Normal vs Reset vs Scramble)
    print("\n--- 8. Hostile Causal Ablation Controls (K=8 vs Proposal-GRU) ---")
    hostile_results: dict[str, Any] = {}
    test_pb8 = build_resource_matched_thought_model(8).to(device)
    train_thought_mediated_model(test_pb8, device=device, training_steps=args.train_steps)

    rep_normal = evaluate_delayed_resolution_benchmark(test_pb8, delay_ticks=15, num_episodes=30, device=device, reset_every_frame=False)
    rep_reset = evaluate_delayed_resolution_benchmark(test_pb8, delay_ticks=15, num_episodes=30, device=device, reset_every_frame=True)

    hostile_results["PB_K8_Normal"] = {"delayed_acc_pct": rep_normal.delayed_decision_acc_pct, "fidelity": rep_normal.hypothesis_retention_fidelity, "mean_return": rep_normal.mean_delayed_return}
    hostile_results["PB_K8_Reset_Thoughts"] = {"delayed_acc_pct": rep_reset.delayed_decision_acc_pct, "fidelity": rep_reset.hypothesis_retention_fidelity, "mean_return": rep_reset.mean_delayed_return}
    print(f"  [PB K=8 Normal        ] Delayed Acc = {rep_normal.delayed_decision_acc_pct:5.1f}% | Retention Fidelity = {rep_normal.hypothesis_retention_fidelity:.2f} | Return = {rep_normal.mean_delayed_return:+5.2f}")
    print(f"  [PB K=8 Reset Thoughts] Delayed Acc = {rep_reset.delayed_decision_acc_pct:5.1f}% | Retention Fidelity = {rep_reset.hypothesis_retention_fidelity:.2f} | Return = {rep_reset.mean_delayed_return:+5.2f}")

    gru_hostile = ProposalGRUBaseline().to(device)
    train_proposal_gru_baseline(gru_hostile, device=device, training_steps=args.train_steps)
    gru_rep_norm = evaluate_delayed_resolution_benchmark(gru_hostile, delay_ticks=15, num_episodes=30, device=device, reset_every_frame=False)
    gru_rep_reset = evaluate_delayed_resolution_benchmark(gru_hostile, delay_ticks=15, num_episodes=30, device=device, reset_every_frame=True)
    hostile_results["GRU_Normal"] = {"delayed_acc_pct": gru_rep_norm.delayed_decision_acc_pct, "fidelity": gru_rep_norm.hypothesis_retention_fidelity, "mean_return": gru_rep_norm.mean_delayed_return}
    hostile_results["GRU_Reset_Hidden"] = {"delayed_acc_pct": gru_rep_reset.delayed_decision_acc_pct, "fidelity": gru_rep_reset.hypothesis_retention_fidelity, "mean_return": gru_rep_reset.mean_delayed_return}
    print(f"  [GRU Normal           ] Delayed Acc = {gru_rep_norm.delayed_decision_acc_pct:5.1f}% | Retention Fidelity = {gru_rep_norm.hypothesis_retention_fidelity:.2f} | Return = {gru_rep_norm.mean_delayed_return:+5.2f}")
    print(f"  [GRU Reset Hidden     ] Delayed Acc = {gru_rep_reset.delayed_decision_acc_pct:5.1f}% | Retention Fidelity = {gru_rep_reset.hypothesis_retention_fidelity:.2f} | Return = {gru_rep_reset.mean_delayed_return:+5.2f}")

    results["hostile_causal_controls"] = hostile_results

    # 9. Delayed Resolution Benchmark (10-30 Ticks of Uninformative Occlusion)
    print("\n--- 9. Delayed Resolution Benchmark (D=15 Ticks) ---")
    delayed_results: dict[str, Any] = {}
    for k in [1, 4, 8, 16, 32]:
        m = build_resource_matched_thought_model(k).to(device)
        train_thought_mediated_model(m, device=device, training_steps=args.train_steps)
        d_rep = evaluate_delayed_resolution_benchmark(m, delay_ticks=15, num_episodes=30, device=device)
        delayed_results[f"Pseudo-Brain K={k}"] = {
            "delayed_decision_acc_pct": d_rep.delayed_decision_acc_pct,
            "hypothesis_retention_fidelity": d_rep.hypothesis_retention_fidelity,
            "premature_collapse_rate_pct": d_rep.premature_collapse_rate_pct,
            "mean_delayed_return": d_rep.mean_delayed_return,
        }
        print(f"  Pseudo-Brain K = {k:2d} | Delayed Acc: {d_rep.delayed_decision_acc_pct:5.1f}% | Retention: {d_rep.hypothesis_retention_fidelity:.2f} | Premature Collapse: {d_rep.premature_collapse_rate_pct:5.1f}% | Return: {d_rep.mean_delayed_return:+5.2f}")

    gru_d = ProposalGRUBaseline().to(device)
    train_proposal_gru_baseline(gru_d, device=device, training_steps=args.train_steps)
    gru_d_rep = evaluate_delayed_resolution_benchmark(gru_d, delay_ticks=15, num_episodes=30, device=device)
    delayed_results["Proposal-GRU Match"] = {
        "delayed_decision_acc_pct": gru_d_rep.delayed_decision_acc_pct,
        "hypothesis_retention_fidelity": gru_d_rep.hypothesis_retention_fidelity,
        "premature_collapse_rate_pct": gru_d_rep.premature_collapse_rate_pct,
        "mean_delayed_return": gru_d_rep.mean_delayed_return,
    }
    print(f"  Proposal-GRU Match | Delayed Acc: {gru_d_rep.delayed_decision_acc_pct:5.1f}% | Retention: {gru_d_rep.hypothesis_retention_fidelity:.2f} | Premature Collapse: {gru_d_rep.premature_collapse_rate_pct:5.1f}% | Return: {gru_d_rep.mean_delayed_return:+5.2f}")
    results["delayed_resolution_benchmark"] = delayed_results

    # 10. Nested Factorized Uncertainty Benchmark (4 Joint Hypotheses -> 2 -> 1)
    print("\n--- 10. Nested Factorized Uncertainty Benchmark (4 -> 2 -> 1) ---")
    nested_results: dict[str, Any] = {}
    for k in [1, 4, 8, 16, 32]:
        m = build_resource_matched_thought_model(k).to(device)
        train_thought_mediated_model(m, device=device, training_steps=args.train_steps)
        n_rep = evaluate_nested_uncertainty_benchmark(m, num_episodes=30, device=device)
        nested_results[f"Pseudo-Brain K={k}"] = {
            "four_to_two_collapse_acc_pct": n_rep.four_to_two_collapse_acc_pct,
            "two_to_one_collapse_acc_pct": n_rep.two_to_one_collapse_acc_pct,
            "partial_collapse_entropy_bits": n_rep.partial_collapse_entropy_bits,
            "compound_goal_acc_pct": n_rep.compound_goal_acc_pct,
            "mean_nested_return": n_rep.mean_nested_return,
        }
        print(f"  Pseudo-Brain K = {k:2d} | 4->2 Acc: {n_rep.four_to_two_collapse_acc_pct:5.1f}% | 2->1 Acc: {n_rep.two_to_one_collapse_acc_pct:5.1f}% | Entropy Drop: {n_rep.partial_collapse_entropy_bits:.2f} bits | Goal Acc: {n_rep.compound_goal_acc_pct:5.1f}% | Return: {n_rep.mean_nested_return:+5.2f}")

    gru_n = ProposalGRUBaseline().to(device)
    train_proposal_gru_baseline(gru_n, device=device, training_steps=args.train_steps)
    gru_n_rep = evaluate_nested_uncertainty_benchmark(gru_n, num_episodes=30, device=device)
    nested_results["Proposal-GRU Match"] = {
        "four_to_two_collapse_acc_pct": gru_n_rep.four_to_two_collapse_acc_pct,
        "two_to_one_collapse_acc_pct": gru_n_rep.two_to_one_collapse_acc_pct,
        "partial_collapse_entropy_bits": gru_n_rep.partial_collapse_entropy_bits,
        "compound_goal_acc_pct": gru_n_rep.compound_goal_acc_pct,
        "mean_nested_return": gru_n_rep.mean_nested_return,
    }
    print(f"  Proposal-GRU Match | 4->2 Acc: {gru_n_rep.four_to_two_collapse_acc_pct:5.1f}% | 2->1 Acc: {gru_n_rep.two_to_one_collapse_acc_pct:5.1f}% | Entropy Drop: {gru_n_rep.partial_collapse_entropy_bits:.2f} bits | Goal Acc: {gru_n_rep.compound_goal_acc_pct:5.1f}% | Return: {gru_n_rep.mean_nested_return:+5.2f}")
    results["nested_uncertainty_benchmark"] = nested_results

    # 11. Non-Stationary Stochastic Law / Regime Shift Benchmark (80/20 -> 20/80 switch)
    print("\n--- 11. Non-Stationary Stochastic Law Shift Benchmark (80/20 -> 20/80) ---")
    shift_results: dict[str, Any] = {}
    for k in [1, 4, 8, 16, 32]:
        m = build_resource_matched_thought_model(k).to(device)
        train_thought_mediated_model(m, device=device, training_steps=args.train_steps)
        s_rep = evaluate_non_stationary_regime_shift_benchmark(m, episode_length=100, shift_tick=50, device=device)
        shift_results[f"Pseudo-Brain K={k}"] = {
            "pre_shift_accuracy_pct": s_rep.pre_shift_accuracy_pct,
            "post_shift_accuracy_pct": s_rep.post_shift_accuracy_pct,
            "probability_recovery_time_ticks": s_rep.probability_recovery_time_ticks,
            "empirical_cross_entropy_loss": s_rep.empirical_cross_entropy_loss,
        }
        print(f"  Pseudo-Brain K = {k:2d} | Pre Acc: {s_rep.pre_shift_accuracy_pct:5.1f}% | Post Acc: {s_rep.post_shift_accuracy_pct:5.1f}% | Recovery Time: {s_rep.probability_recovery_time_ticks:4.1f} ticks | CE Loss: {s_rep.empirical_cross_entropy_loss:.3f}")

    gru_s = ProposalGRUBaseline().to(device)
    train_proposal_gru_baseline(gru_s, device=device, training_steps=args.train_steps)
    gru_s_rep = evaluate_non_stationary_regime_shift_benchmark(gru_s, episode_length=100, shift_tick=50, device=device)
    shift_results["Proposal-GRU Match"] = {
        "pre_shift_accuracy_pct": gru_s_rep.pre_shift_accuracy_pct,
        "post_shift_accuracy_pct": gru_s_rep.post_shift_accuracy_pct,
        "probability_recovery_time_ticks": gru_s_rep.probability_recovery_time_ticks,
        "empirical_cross_entropy_loss": gru_s_rep.empirical_cross_entropy_loss,
    }
    print(f"  Proposal-GRU Match | Pre Acc: {gru_s_rep.pre_shift_accuracy_pct:5.1f}% | Post Acc: {gru_s_rep.post_shift_accuracy_pct:5.1f}% | Recovery Time: {gru_s_rep.probability_recovery_time_ticks:4.1f} ticks | CE Loss: {gru_s_rep.empirical_cross_entropy_loss:.3f}")
    results["non_stationary_shift_benchmark"] = shift_results

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Thought-Mediated Campaign results saved to {args.output_json}")


if __name__ == "__main__":
    main()
