"""Master benchmark evaluation script for Rapid Online Adaptation & Hidden Rule Learning.

Evaluates 6 model conditions:
1. Reactive Baseline
2. Standard GRU (recurrent memory, no plasticity)
3. Thoughtlet Baseline (recurrent thoughtlets, no plasticity)
4. Thoughtlet + Fast Plasticity (Full: surprise-driven P_t)
5. Thoughtlet + Fast Plasticity (Ablation: surprise ablated, e_t = 0)
6. Thoughtlet + Fast Plasticity (Ablation: plasticity disabled, P_t = 0)

Across multi-seed test sessions:
- Block 1 (Trials 1..10): Rule A (Initial learning & retention)
- Block 2 (Trials 11..20): Rule B (Rule Reversal 1)
- Block 3 (Trials 21..30): Rule A (Rule Reversal 2 / Catastrophic interference)

Generates:
- online_adaptation_results.json
- ONLINE_ADAPTATION_REPORT.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.types import GenericControl, HidKey
from online_adaptation.hidden_rule_env import HiddenRuleEnv, Rule
from online_adaptation.models import (
    ReactiveModel,
    PredictiveGRUModel,
    PredictiveThoughtletModel,
    ACTION_CLASSES,
)

N_FRAMES = 4

CTRL_MAP = {
    0: GenericControl(),
    1: GenericControl(keys_down=(int(HidKey.W),)),
    2: GenericControl(keys_down=(int(HidKey.A),)),
    3: GenericControl(keys_down=(int(HidKey.S),)),
    4: GenericControl(keys_down=(int(HidKey.D),)),
}


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_single_session(
    model: torch.nn.Module,
    model_type: str,
    env: HiddenRuleEnv,
    seed: int,
    device: torch.device,
    use_plasticity: bool = False,
    ablate_surprise: bool = False,
) -> Dict:
    """Evaluates a model over a full 30-trial session."""
    obs = env.reset(seed=seed)

    frame_raw = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
    frame_history = [frame_raw] * N_FRAMES

    recurrent_state = None
    P_t = None
    prev_action = 0
    surprise = torch.zeros(1, 1, device=device)

    # Plasticity telemetry
    p_norms = []
    delta_p_norms = []
    surprises = []

    # Configure model runtime flags
    if hasattr(model, "use_plasticity"):
        model.use_plasticity = use_plasticity
    if hasattr(model, "ablate_surprise"):
        model.ablate_surprise = ablate_surprise

    done = False
    step_count = 0
    t0 = time.perf_counter()

    # Pre-encode initial observation to eliminate redundant encoding per tick
    z_t = None
    if model_type != "reactive":
        init_stacked = np.stack(frame_history[-N_FRAMES:], axis=0).transpose(0, 3, 1, 2)
        init_tensor = torch.from_numpy(init_stacked).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            z_t = model.encode_observation(init_tensor)

    while not done:
        step_count += 1
        prev_act_tensor = torch.tensor([prev_action], dtype=torch.long, device=device)

        with torch.no_grad():
            if model_type == "reactive":
                stacked = np.stack(frame_history[-N_FRAMES:], axis=0).transpose(0, 3, 1, 2)
                frames_tensor = torch.from_numpy(stacked).float().unsqueeze(0).to(device) / 255.0
                logits = model(frames_tensor)
                action = int(logits.argmax(dim=-1).item())
            else:
                if model_type == "gru":
                    logits, recurrent_state, e_t, P_t, delta_P = model.forward_step(
                        z_t=z_t,
                        prev_action=prev_act_tensor,
                        surprise_t=surprise,
                        h=recurrent_state,
                        P_t=P_t,
                    )
                else:
                    logits, recurrent_state, e_t, P_t, delta_P = model.forward_step(
                        z_t=z_t,
                        prev_action=prev_act_tensor,
                        surprise_t=surprise,
                        thoughts=recurrent_state,
                        P_t=P_t,
                    )
                action = int(logits.argmax(dim=-1).item())

                # Predict next latent
                z_hat = model.predict_next_latent(recurrent_state, torch.tensor([action], device=device))

                if P_t is not None:
                    p_norms.append(float(torch.norm(P_t).item()))
                if delta_P is not None:
                    delta_p_norms.append(float(torch.norm(delta_P).item()))

        # Step environment
        ctrl = CTRL_MAP.get(action, CTRL_MAP[0])
        outcome = env.step(ctrl)

        # Update frame history
        next_raw = np.frombuffer(outcome.observation.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
        frame_history.append(next_raw)

        if model_type != "reactive":
            next_stacked = np.stack(frame_history[-N_FRAMES:], axis=0).transpose(0, 3, 1, 2)
            next_tensor = torch.from_numpy(next_stacked).float().unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                z_true_next = model.encode_observation(next_tensor)
                err = torch.norm(z_hat - z_true_next, dim=-1, keepdim=True)
                surprise = err
                surprises.append(float(err.item()))
                z_t = z_true_next  # Recycle: avoids re-encoding identical frame next tick

        prev_action = action
        if outcome.terminated or outcome.truncated:
            done = True

    elapsed = time.perf_counter() - t0
    latency_ms = (elapsed / max(step_count, 1)) * 1000.0

    return {
        "trial_outcomes": env.trial_outcomes,
        "total_steps": step_count,
        "latency_ms": latency_ms,
        "mean_p_norm": float(np.mean(p_norms)) if p_norms else 0.0,
        "mean_delta_p": float(np.mean(delta_p_norms)) if delta_p_norms else 0.0,
        "mean_surprise": float(np.mean(surprises)) if surprises else 0.0,
    }


def evaluate_sessions_batched(
    model: torch.nn.Module,
    model_type: str,
    seeds: List[int],
    device: torch.device,
    rule_schedule: Optional[List[Tuple[Rule, int]]] = None,
    use_plasticity: bool = False,
    ablate_surprise: bool = False,
) -> List[Dict]:
    """Evaluates multiple test sessions concurrently with batched model inference.
    Replaces N serial sessions with a single batched runner (batch_size = len(seeds)).
    """
    N = len(seeds)
    if N == 0:
        return []

    envs = [HiddenRuleEnv(rule_schedule=rule_schedule) for _ in range(N)]
    obs_list = [envs[i].reset(seed=seeds[i]) for i in range(N)]
    frame_histories = [[np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)] * N_FRAMES for obs in obs_list]

    if hasattr(model, "use_plasticity"):
        model.use_plasticity = use_plasticity
    if hasattr(model, "ablate_surprise"):
        model.ablate_surprise = ablate_surprise

    recurrent_state = None
    P_t = None
    prev_actions = torch.zeros(N, dtype=torch.long, device=device)
    surprise = torch.zeros(N, 1, device=device)

    # Pre-encode initial observation
    if model_type != "reactive":
        stacked_init = np.stack([
            np.stack(h[-N_FRAMES:], axis=0).transpose(0, 3, 1, 2)
            for h in frame_histories
        ], axis=0)
        frames_tensor = torch.from_numpy(stacked_init).float().to(device) / 255.0
        with torch.no_grad():
            z_t = model.encode_observation(frames_tensor)
    else:
        z_t = None

    dones = [False] * N
    step_counts = [0] * N
    latencies = [[] for _ in range(N)]
    p_norms = [[] for _ in range(N)]
    delta_p_norms = [[] for _ in range(N)]
    surprises = [[] for _ in range(N)]

    while not all(dones):
        t0 = time.perf_counter()

        with torch.no_grad():
            if model_type == "reactive":
                stacked = np.stack([
                    np.stack(h[-N_FRAMES:], axis=0).transpose(0, 3, 1, 2)
                    for h in frame_histories
                ], axis=0)
                frames_tensor = torch.from_numpy(stacked).float().to(device) / 255.0
                logits = model(frames_tensor)
                actions = logits.argmax(dim=-1).cpu().tolist()
            else:
                if model_type == "gru":
                    logits, recurrent_state, e_t, P_t, delta_P = model.forward_step(
                        z_t=z_t,
                        prev_action=prev_actions,
                        surprise_t=surprise,
                        h=recurrent_state,
                        P_t=P_t,
                    )
                else:
                    logits, recurrent_state, e_t, P_t, delta_P = model.forward_step(
                        z_t=z_t,
                        prev_action=prev_actions,
                        surprise_t=surprise,
                        thoughts=recurrent_state,
                        P_t=P_t,
                    )
                actions = logits.argmax(dim=-1).cpu().tolist()
                actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
                z_hat = model.predict_next_latent(recurrent_state, actions_tensor)

                if P_t is not None:
                    p_norm_vals = torch.norm(P_t, dim=-1).cpu().tolist()
                    for i in range(N):
                        if not dones[i]:
                            p_norms[i].append(p_norm_vals[i])
                if delta_P is not None:
                    dp_norm_vals = torch.norm(delta_P, dim=-1).cpu().tolist()
                    for i in range(N):
                        if not dones[i]:
                            delta_p_norms[i].append(dp_norm_vals[i])

        # Step environments
        for i in range(N):
            if not dones[i]:
                step_counts[i] += 1
                ctrl = CTRL_MAP.get(actions[i], CTRL_MAP[0])
                outcome = envs[i].step(ctrl)
                next_raw = np.frombuffer(outcome.observation.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
                frame_histories[i].append(next_raw)
                if outcome.terminated or outcome.truncated:
                    dones[i] = True

        # Next observation encoding & surprise
        if model_type != "reactive":
            next_stacked = np.stack([
                np.stack(h[-N_FRAMES:], axis=0).transpose(0, 3, 1, 2)
                for h in frame_histories
            ], axis=0)
            next_tensor = torch.from_numpy(next_stacked).float().to(device) / 255.0
            with torch.no_grad():
                z_true_next = model.encode_observation(next_tensor)
                err = torch.norm(z_hat - z_true_next, dim=-1, keepdim=True)
                surprise = err
                z_t = z_true_next

            surp_vals = err.squeeze(-1).cpu().tolist()
            for i in range(N):
                if not dones[i]:
                    surprises[i].append(surp_vals[i])

        prev_actions = torch.tensor(actions, dtype=torch.long, device=device)

        step_elapsed = time.perf_counter() - t0
        for i in range(N):
            if not dones[i]:
                latencies[i].append(step_elapsed * 1000.0)

    results = []
    for i in range(N):
        results.append({
            "trial_outcomes": envs[i].trial_outcomes,
            "total_steps": step_counts[i],
            "latency_ms": float(np.mean(latencies[i])) if latencies[i] else 0.0,
            "mean_p_norm": float(np.mean(p_norms[i])) if p_norms[i] else 0.0,
            "mean_delta_p": float(np.mean(delta_p_norms[i])) if delta_p_norms[i] else 0.0,
            "mean_surprise": float(np.mean(surprises[i])) if surprises[i] else 0.0,
        })
    return results


def run_benchmark(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    seeds: List[int] = [42, 142, 242],
    sessions_per_seed: int = 20,
    device_str: str = "cpu",
) -> Dict:
    ckpt_path = Path(checkpoint_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    # Standard evaluation schedule: 10 Rule A -> 10 Rule B -> 10 Rule A (30 trials total)
    test_schedule = [
        (Rule.RULE_A, 10),
        (Rule.RULE_B, 10),
        (Rule.RULE_A, 10),
    ]

    plastic_ckpt = "plastic_thoughtlet_seed_42.pt" if (ckpt_path / "plastic_thoughtlet_seed_42.pt").exists() else "thoughtlet_seed_42.pt"

    # Models and configurations to benchmark
    # Format: (condition_name, model_type, ckpt_name, use_plasticity, ablate_surprise)
    conditions = [
        ("Reactive Baseline", "reactive", "reactive_seed_42.pt", False, False),
        ("Standard GRU", "gru", "gru_seed_42.pt", False, False),
        ("Thoughtlet Baseline", "thoughtlet", "thoughtlet_seed_42.pt", False, False),
        ("Plastic Thoughtlet (Full)", "thoughtlet", plastic_ckpt, True, False),
        ("Plastic Thoughtlet (No Surprise)", "thoughtlet", plastic_ckpt, True, True),
        ("Plastic Thoughtlet (Disabled)", "thoughtlet", plastic_ckpt, False, False),
    ]

    all_results = {}

    print("=" * 80)
    print("PSEUDO-BRAIN: RAPID ONLINE ADAPTATION & HIDDEN RULE BENCHMARK")
    print(f"Seeds: {seeds} | Sessions per seed: {sessions_per_seed} (Total: {len(seeds)*sessions_per_seed} sessions/model)")
    print(f"Schedule: 10 Rule A -> 10 Rule B -> 10 Rule A (30 trials/session)")
    print("=" * 80)

    for cond_name, m_type, ckpt_name, use_plast, ablate_surp in conditions:
        print(f"\n>>> Evaluating Condition: {cond_name.upper()}")
        ckpt_file = ckpt_path / ckpt_name
        if not ckpt_file.exists():
            print(f"Warning: Checkpoint {ckpt_file} not found, skipping...")
            continue

        ckpt_data = torch.load(ckpt_file, map_location=device)
        if m_type == "reactive":
            model = ReactiveModel().to(device)
        elif m_type == "gru":
            model = PredictiveGRUModel(use_plasticity=use_plast, ablate_surprise=ablate_surp).to(device)
        elif m_type == "thoughtlet":
            model = PredictiveThoughtletModel(use_plasticity=use_plast, ablate_surprise=ablate_surp).to(device)

        model.load_state_dict(ckpt_data["model_state_dict"])
        model.eval()

        n_params = count_parameters(model)

        # Storage across all sessions
        trial_correct_matrix = np.zeros((len(seeds) * sessions_per_seed, 30))
        latencies = []
        p_norms = []
        delta_p_norms = []
        surprises = []
        adaptation_trials_b2 = []
        adaptation_trials_b3 = []

        sess_idx = 0
        for s in seeds:
            for ep in range(sessions_per_seed):
                sess_seed = s * 1000 + ep
                env = HiddenRuleEnv(rule_schedule=test_schedule)
                sess_res = evaluate_single_session(
                    model=model,
                    model_type=m_type,
                    env=env,
                    seed=sess_seed,
                    device=device,
                    use_plasticity=use_plast,
                    ablate_surprise=ablate_surp,
                )

                outcomes = sess_res["trial_outcomes"]
                for t_idx, item in enumerate(outcomes):
                    if t_idx < 30:
                        trial_correct_matrix[sess_idx, t_idx] = 1.0 if item["correct"] else 0.0

                # Compute trials to adaptation for Block 2 (Trials 10..19)
                # First trial in Block 2 where choice is correct
                b2_correct = [i for i in range(10, 20) if i < len(outcomes) and outcomes[i]["correct"]]
                if b2_correct:
                    adaptation_trials_b2.append(b2_correct[0] - 10 + 1)
                else:
                    adaptation_trials_b2.append(10)  # max trials in block

                # Block 3 adaptation (Trials 20..29)
                b3_correct = [i for i in range(20, 30) if i < len(outcomes) and outcomes[i]["correct"]]
                if b3_correct:
                    adaptation_trials_b3.append(b3_correct[0] - 20 + 1)
                else:
                    adaptation_trials_b3.append(10)

                latencies.append(sess_res["latency_ms"])
                p_norms.append(sess_res["mean_p_norm"])
                delta_p_norms.append(sess_res["mean_delta_p"])
                surprises.append(sess_res["mean_surprise"])

                sess_idx += 1

        # Aggregate metrics
        trial_acc_mean = np.mean(trial_correct_matrix, axis=0) * 100.0
        trial_acc_std = np.std(trial_correct_matrix, axis=0) * 100.0

        # Key milestone accuracies:
        # Trial 1 (0 exp), Trial 2 (1 exp), Trial 3 (2 exp), Trial 5 (4 exp), Trial 10 (retention)
        # Trial 11 (reversal), Trial 12 (1 exp), Trial 15 (4 exp), Trial 20 (retention)
        # Trial 21 (reversal 2), Trial 22 (1 exp), Trial 30 (retention)
        summary = {
            "parameters": n_params,
            "latency_ms": float(np.mean(latencies)),
            "mean_p_norm": float(np.mean(p_norms)),
            "mean_delta_p": float(np.mean(delta_p_norms)),
            "mean_surprise": float(np.mean(surprises)),
            "trials_to_adaptation_b2": float(np.mean(adaptation_trials_b2)),
            "trials_to_adaptation_b3": float(np.mean(adaptation_trials_b3)),
            "trial_accuracy_mean": [float(x) for x in trial_acc_mean],
            "trial_accuracy_std": [float(x) for x in trial_acc_std],
            "milestones": {
                "t1_0exp": float(trial_acc_mean[0]),
                "t2_1exp": float(trial_acc_mean[1]),
                "t3_2exp": float(trial_acc_mean[2]),
                "t5_4exp": float(trial_acc_mean[4]),
                "t10_retention": float(trial_acc_mean[9]),
                "t11_reversal1": float(trial_acc_mean[10]),
                "t12_rev_1exp": float(trial_acc_mean[11]),
                "t15_rev_4exp": float(trial_acc_mean[14]),
                "t20_retention_b2": float(trial_acc_mean[19]),
                "t21_reversal2": float(trial_acc_mean[20]),
                "t22_rev2_1exp": float(trial_acc_mean[21]),
                "t30_retention_b3": float(trial_acc_mean[29]),
            },
        }

        all_results[cond_name] = summary
        print(f"  Params: {n_params:,} | Latency: {summary['latency_ms']:.2f} ms")
        print(f"  Block 1: T1={summary['milestones']['t1_0exp']:.1f}% -> T2={summary['milestones']['t2_1exp']:.1f}% -> T10={summary['milestones']['t10_retention']:.1f}%")
        print(f"  Block 2 (Reversal): T11={summary['milestones']['t11_reversal1']:.1f}% -> T12={summary['milestones']['t12_rev_1exp']:.1f}% -> T20={summary['milestones']['t20_retention_b2']:.1f}% (Trials to adapt: {summary['trials_to_adaptation_b2']:.1f})")
        print(f"  Block 3 (Reversal 2): T21={summary['milestones']['t21_reversal2']:.1f}% -> T22={summary['milestones']['t22_rev2_1exp']:.1f}% -> T30={summary['milestones']['t30_retention_b3']:.1f}% (Trials to adapt: {summary['trials_to_adaptation_b3']:.1f})")

    # Save JSON results
    json_path = out_path / "online_adaptation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw results to {json_path}")

    # Generate dedicated Markdown Report
    report_path = out_path / "ONLINE_ADAPTATION_REPORT.md"
    generate_markdown_report(all_results, report_path)
    print(f"Generated comprehensive report at {report_path}")

    return all_results


def generate_markdown_report(results: Dict, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Online Adaptation & Hidden Rule Benchmark Report\n\n")
        f.write("## 1. Executive Summary & Experimental Question\n\n")
        f.write("> **Fundamental Question**: *Does the fast plasticity mechanism ($P_t$) actually learn a new association from experience during an episode, use that learned information later without backpropagation, and adapt when the hidden rule reverses?*\n\n")

        f.write("### Evaluation Conditions:\n")
        f.write("1. **Reactive Baseline**: Feedforward policy, no working memory, no plasticity.\n")
        f.write("2. **Standard GRU**: 384-d recurrent working memory, no plasticity.\n")
        f.write("3. **Thoughtlet Baseline**: 32 parallel thoughtlets with multi-head attention, no plasticity.\n")
        f.write("4. **Plastic Thoughtlet (Full)**: Thoughtlets with fast associative state $P_t$ driven by prediction error $e_t$.\n")
        f.write("5. **Plastic Thoughtlet (No Surprise)**: Ablation with surprise input zeroed ($e_t = 0$).\n")
        f.write("6. **Plastic Thoughtlet (Disabled)**: Identical checkpoint with $P_t = 0$.\n\n")

        f.write("## 2. Multi-Trial Adaptation Matrix\n\n")
        f.write("| Condition | Params | Latency | Trial 1 (0 exp) | Trial 2 (1 exp) | Trial 3 (2 exp) | Trial 10 (Retain) | T11 (Reversal) | T12 (Adapt 1) | T20 (Retain B) | Adapt Latency (B2) |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

        for name, data in results.items():
            m = data["milestones"]
            f.write(
                f"| **{name}** | {data['parameters']:,} | {data['latency_ms']:.2f} ms | "
                f"{m['t1_0exp']:.1f}% | {m['t2_1exp']:.1f}% | {m['t3_2exp']:.1f}% | {m['t10_retention']:.1f}% | "
                f"{m['t11_reversal1']:.1f}% | {m['t12_rev_1exp']:.1f}% | {m['t20_retention_b2']:.1f}% | "
                f"{data['trials_to_adaptation_b2']:.1f} trials |\n"
            )

        f.write("\n## 3. Learning Curves (Performance vs. Experiences)\n\n")
        f.write("```text\n")
        f.write("Trial:          T1 (0)   T2 (1)   T3 (2)   T5 (4)   T10 (Ret)  |  T11 (Rev)  T12 (1)   T15 (4)   T20 (Ret)\n")
        f.write("Rule:           [---------------- Rule A ----------------]  |  [---------------- Rule B ----------------]\n")
        f.write("-" * 95 + "\n")
        for name, data in results.items():
            m = data["milestones"]
            f.write(f"{name:<28} {m['t1_0exp']:5.1f}%  {m['t2_1exp']:5.1f}%  {m['t3_2exp']:5.1f}%  {m['t5_4exp']:5.1f}%  {m['t10_retention']:5.1f}%   |  {m['t11_reversal1']:5.1f}%  {m['t12_rev_1exp']:5.1f}%  {m['t15_rev_4exp']:5.1f}%  {m['t20_retention_b2']:5.1f}%\n")
        f.write("```\n\n")

        f.write("## 4. Scientific Findings & Mechanistic Diagnosis\n\n")
        f.write("### A. Memory vs. Plasticity Analysis\n")
        f.write("- Does recurrence alone support rapid online rule retention?\n")
        f.write("- Does fast plasticity ($P_t$) modulate decision policy directly from feedback?\n\n")
        f.write("### B. Critical Surprise Ablation (Condition 4 vs. Condition 5)\n")
        f.write("- Evaluating whether prediction surprise is necessary for rapid behavioral adaptation.\n\n")
        f.write("### C. Rule Reversal & Catastrophic Interference (Condition 4)\n")
        f.write("- Ability to unlearn obsolete associations and switch policies repeatedly.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="brain/runs/online_adaptation/checkpoints")
    parser.add_argument("--output_dir", type=str, default="brain/runs/online_adaptation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 142, 242])
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    run_benchmark(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        sessions_per_seed=args.sessions,
        device_str=args.device,
    )
