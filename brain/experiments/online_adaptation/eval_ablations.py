"""Mechanistic Ablation Benchmark for Rapid Online Adaptation.

Evaluates 4 core ablation conditions across 10 held-out test sessions:
1. Full Plastic Thoughtlet: both recurrent state h_t and fast plasticity P_t active
2. Plasticity-only: reset recurrent thoughts h_t = 0 across trial boundaries, keeping only P_t
3. Surprise-Ablated: zero out surprise e_t = 0, eliminating prediction error signal
4. Plasticity-Disabled: ablate P_t = 0, relying solely on recurrent thoughtlets

Also evaluates the Pure Thoughtlet Baseline (thoughtlet_seed_42.pt) as a trained non-plastic control.

Saves:
- runs/online_adaptation_colab_v2/ablation_results.json
- runs/online_adaptation_colab_v2/ablation_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Path setup
_SCRIPT_DIR = Path(__file__).resolve().parent
_BRAIN_DIR = _SCRIPT_DIR.parents[1]  # brain
_PROJECT_ROOT = _BRAIN_DIR.parent    # Pseudo-Brain repository root

sys.path.insert(0, str(_BRAIN_DIR / "src"))
sys.path.insert(0, str(_BRAIN_DIR / "experiments"))

from irene_brain.types import GenericControl, HidKey
from online_adaptation.hidden_rule_env import HiddenRuleEnv, Rule
from online_adaptation.models import (
    PredictiveThoughtletModel,
    ACTION_CLASSES,
    N_FRAMES,
)

CTRL_MAP = {
    0: GenericControl(),
    1: GenericControl(keys_down=(int(HidKey.W),)),
    2: GenericControl(keys_down=(int(HidKey.A),)),
    3: GenericControl(keys_down=(int(HidKey.S),)),
    4: GenericControl(keys_down=(int(HidKey.D),)),
}


def evaluate_ablation_session(
    model: PredictiveThoughtletModel,
    env: HiddenRuleEnv,
    seed: int,
    device: torch.device,
    use_plasticity: bool,
    ablate_surprise: bool,
    reset_recurrent_on_trial: bool,
) -> Dict[str, Any]:
    """Runs a single 30-trial session under specific mechanistic ablation flags."""
    obs = env.reset(seed=seed)

    frame_raw = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
    frame_history = [frame_raw] * N_FRAMES

    recurrent_state = None
    P_t = None
    prev_action = 0
    surprise = torch.zeros(1, 1, device=device)
    last_trial_idx = 0

    p_norms: List[float] = []
    delta_p_norms: List[float] = []
    surprises: List[float] = []

    model.use_plasticity = use_plasticity
    model.ablate_surprise = ablate_surprise

    done = False
    step_count = 0
    t0 = time.perf_counter()

    while not done:
        step_count += 1

        # Mechanistic intervention: reset recurrent thoughtlets at trial boundaries
        if reset_recurrent_on_trial and env.current_trial != last_trial_idx:
            recurrent_state = None
            last_trial_idx = env.current_trial

        stacked = np.stack(frame_history[-N_FRAMES:], axis=0).transpose(0, 3, 1, 2)
        frames_tensor = torch.from_numpy(stacked).float().unsqueeze(0).to(device) / 255.0
        prev_act_tensor = torch.tensor([prev_action], dtype=torch.long, device=device)

        with torch.no_grad():
            z_t = model.encode_observation(frames_tensor)

            # Surprise ablation handling: enforce e_t = 0 when ablated
            eff_surprise = torch.zeros_like(surprise) if ablate_surprise else surprise

            logits, recurrent_state, e_t, P_t, delta_P = model.forward_step(
                z_t=z_t,
                prev_action=prev_act_tensor,
                surprise_t=eff_surprise,
                thoughts=recurrent_state,
                P_t=P_t,
            )

            action = int(logits.argmax(dim=-1).item())
            z_hat = model.predict_next_latent(recurrent_state, torch.tensor([action], device=device))

            if P_t is not None:
                p_norms.append(float(torch.norm(P_t).item()))
            if delta_P is not None:
                delta_p_norms.append(float(torch.norm(delta_P).item()))

        ctrl = CTRL_MAP.get(action, CTRL_MAP[0])
        outcome = env.step(ctrl)

        next_raw = np.frombuffer(outcome.observation.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
        frame_history.append(next_raw)

        with torch.no_grad():
            next_stacked = np.stack(frame_history[-N_FRAMES:], axis=0).transpose(0, 3, 1, 2)
            next_tensor = torch.from_numpy(next_stacked).float().unsqueeze(0).to(device) / 255.0
            z_true_next = model.encode_observation(next_tensor)
            err = torch.norm(z_hat - z_true_next, dim=-1, keepdim=True)
            surprise = err
            surprises.append(float(err.item()))

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


def run_ablation_study(
    checkpoint_dir: Path,
    output_dir: Path,
    n_sessions: int = 10,
    base_seed: int = 42,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    plastic_ckpt_file = checkpoint_dir / "plastic_thoughtlet_seed_42.pt"
    thoughtlet_ckpt_file = checkpoint_dir / "thoughtlet_seed_42.pt"

    if not plastic_ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint {plastic_ckpt_file} not found!")

    plastic_ckpt = torch.load(plastic_ckpt_file, map_location=device, weights_only=False)
    has_thoughtlet_ckpt = thoughtlet_ckpt_file.exists()
    if has_thoughtlet_ckpt:
        thoughtlet_ckpt = torch.load(thoughtlet_ckpt_file, map_location=device, weights_only=False)

    # 3-block schedule: 10 Rule A -> 10 Rule B -> 10 Rule A
    test_schedule = [(Rule.RULE_A, 10), (Rule.RULE_B, 10), (Rule.RULE_A, 10)]

    conditions = [
        {
            "id": "full_plastic",
            "name": "1. Full Plastic Thoughtlet",
            "ckpt_data": plastic_ckpt,
            "ckpt_file": plastic_ckpt_file.name,
            "use_plasticity": True,
            "ablate_surprise": False,
            "reset_recurrent_on_trial": False,
            "description": "Both recurrent thoughts h_t and fast plasticity P_t active across trials",
        },
        {
            "id": "plasticity_only",
            "name": "2. Plasticity-only",
            "ckpt_data": plastic_ckpt,
            "ckpt_file": plastic_ckpt_file.name,
            "use_plasticity": True,
            "ablate_surprise": False,
            "reset_recurrent_on_trial": True,
            "description": "Recurrent thoughts reset (h_t = 0) at trial boundaries; memory carried solely by P_t",
        },
        {
            "id": "surprise_ablated",
            "name": "3. Surprise-Ablated",
            "ckpt_data": plastic_ckpt,
            "ckpt_file": plastic_ckpt_file.name,
            "use_plasticity": True,
            "ablate_surprise": True,
            "reset_recurrent_on_trial": False,
            "description": "Surprise signal zeroed (e_t = 0); plasticity receives no prediction error feedback",
        },
        {
            "id": "plasticity_disabled",
            "name": "4. Plasticity-Disabled",
            "ckpt_data": plastic_ckpt,
            "ckpt_file": plastic_ckpt_file.name,
            "use_plasticity": False,
            "ablate_surprise": False,
            "reset_recurrent_on_trial": False,
            "description": "Plastic trace disabled (P_t = 0); relies solely on recurrent thoughtlet state",
        },
    ]

    if has_thoughtlet_ckpt:
        conditions.append({
            "id": "thoughtlet_baseline",
            "name": "Control: Pure Thoughtlet Baseline",
            "ckpt_data": thoughtlet_ckpt,
            "ckpt_file": thoughtlet_ckpt_file.name,
            "use_plasticity": False,
            "ablate_surprise": False,
            "reset_recurrent_on_trial": False,
            "description": "Model trained without plasticity module (thoughtlet_seed_42.pt)",
        })

    print("=" * 85)
    print("PSEUDO-BRAIN: RAPID ONLINE ADAPTATION MECHANISTIC ABLATION BENCHMARK")
    print(f"Schedule: 10 Rule A -> 10 Rule B -> 10 Rule A (30 trials/session)")
    print(f"Sessions: {n_sessions} held-out test sessions (seeds: {base_seed * 1000}..{base_seed * 1000 + n_sessions - 1})")
    print(f"Device: {device}")
    print("=" * 85)

    all_results = {}
    csv_rows = []

    for cond in conditions:
        print(f"\nEvaluating Condition: {cond['name']} ...")
        model = PredictiveThoughtletModel(
            use_plasticity=cond["use_plasticity"],
            ablate_surprise=cond["ablate_surprise"],
        ).to(device)
        model.load_state_dict(cond["ckpt_data"]["model_state_dict"])
        model.eval()

        trial_matrix = np.zeros((n_sessions, 30))
        latencies = []
        p_norms = []
        delta_p_norms = []
        surprises = []
        adapt_trials_b2 = []
        adapt_trials_b3 = []

        for s_idx in range(n_sessions):
            sess_seed = base_seed * 1000 + s_idx
            env = HiddenRuleEnv(rule_schedule=test_schedule)
            sess_res = evaluate_ablation_session(
                model=model,
                env=env,
                seed=sess_seed,
                device=device,
                use_plasticity=cond["use_plasticity"],
                ablate_surprise=cond["ablate_surprise"],
                reset_recurrent_on_trial=cond["reset_recurrent_on_trial"],
            )

            outcomes = sess_res["trial_outcomes"]
            for t_idx, item in enumerate(outcomes):
                if t_idx < 30:
                    trial_matrix[s_idx, t_idx] = 1.0 if item["correct"] else 0.0

            # Compute trials to adaptation for Block 2 (Trials 11..20, 0-indexed 10..19)
            b2_correct = [i for i in range(10, 20) if i < len(outcomes) and outcomes[i]["correct"]]
            adapt_trials_b2.append((b2_correct[0] - 10 + 1) if b2_correct else 10)

            # Compute trials to adaptation for Block 3 (Trials 21..30, 0-indexed 20..29)
            b3_correct = [i for i in range(20, 30) if i < len(outcomes) and outcomes[i]["correct"]]
            adapt_trials_b3.append((b3_correct[0] - 20 + 1) if b3_correct else 10)

            latencies.append(sess_res["latency_ms"])
            p_norms.append(sess_res["mean_p_norm"])
            delta_p_norms.append(sess_res["mean_delta_p"])
            surprises.append(sess_res["mean_surprise"])

        mean_acc_per_trial = np.mean(trial_matrix, axis=0) * 100.0
        std_acc_per_trial = np.std(trial_matrix, axis=0) * 100.0

        b1_acc = float(np.mean(mean_acc_per_trial[0:10]))
        b2_acc = float(np.mean(mean_acc_per_trial[10:20]))
        b3_acc = float(np.mean(mean_acc_per_trial[20:30]))
        overall_acc = float(np.mean(mean_acc_per_trial))

        t1_acc = float(mean_acc_per_trial[0])
        t2_acc = float(mean_acc_per_trial[1])
        t10_acc = float(mean_acc_per_trial[9])
        t11_acc = float(mean_acc_per_trial[10])
        t12_acc = float(mean_acc_per_trial[11])
        t20_acc = float(mean_acc_per_trial[19])
        t21_acc = float(mean_acc_per_trial[20])
        t22_acc = float(mean_acc_per_trial[21])
        t30_acc = float(mean_acc_per_trial[29])

        mean_p = float(np.mean(p_norms))
        mean_dp = float(np.mean(delta_p_norms))
        mean_surp = float(np.mean(surprises))
        mean_lat = float(np.mean(latencies))
        mean_adapt_b2 = float(np.mean(adapt_trials_b2))
        mean_adapt_b3 = float(np.mean(adapt_trials_b3))

        cond_result = {
            "name": cond["name"],
            "id": cond["id"],
            "checkpoint": cond["ckpt_file"],
            "description": cond["description"],
            "use_plasticity": cond["use_plasticity"],
            "ablate_surprise": cond["ablate_surprise"],
            "reset_recurrent_on_trial": cond["reset_recurrent_on_trial"],
            "block_accuracies": {
                "block1_rule_a": b1_acc,
                "block2_reversal_rule_b": b2_acc,
                "block3_reversal_rule_a": b3_acc,
                "overall_accuracy": overall_acc,
            },
            "milestones": {
                "t1_0exp": t1_acc,
                "t2_1exp": t2_acc,
                "t10_retention": t10_acc,
                "t11_reversal1": t11_acc,
                "t12_rev_1exp": t12_acc,
                "t20_retention_b2": t20_acc,
                "t21_reversal2": t21_acc,
                "t22_rev2_1exp": t22_acc,
                "t30_retention_b3": t30_acc,
            },
            "adaptation_latency": {
                "trials_to_adaptation_b2": mean_adapt_b2,
                "trials_to_adaptation_b3": mean_adapt_b3,
            },
            "telemetry": {
                "mean_p_norm": mean_p,
                "mean_delta_p": mean_dp,
                "mean_surprise": mean_surp,
                "mean_latency_ms": mean_lat,
            },
            "per_trial_accuracy_mean": [float(x) for x in mean_acc_per_trial],
            "per_trial_accuracy_std": [float(x) for x in std_acc_per_trial],
        }

        all_results[cond["id"]] = cond_result

        csv_rows.append({
            "condition": cond["name"],
            "checkpoint": cond["ckpt_file"],
            "block1_acc": f"{b1_acc:.1f}",
            "block2_reversal_acc": f"{b2_acc:.1f}",
            "block3_reversal_acc": f"{b3_acc:.1f}",
            "overall_acc": f"{overall_acc:.1f}",
            "t1_acc": f"{t1_acc:.1f}",
            "t2_acc": f"{t2_acc:.1f}",
            "t10_acc": f"{t10_acc:.1f}",
            "t11_reversal_acc": f"{t11_acc:.1f}",
            "t12_adapt_acc": f"{t12_acc:.1f}",
            "t20_acc": f"{t20_acc:.1f}",
            "t21_reversal_acc": f"{t21_acc:.1f}",
            "t22_adapt_acc": f"{t22_acc:.1f}",
            "t30_acc": f"{t30_acc:.1f}",
            "adapt_trials_b2": f"{mean_adapt_b2:.1f}",
            "adapt_trials_b3": f"{mean_adapt_b3:.1f}",
            "mean_p_norm": f"{mean_p:.4f}",
            "mean_latency_ms": f"{mean_lat:.2f}",
        })

        print(f"  Block 1 (Rule A):      {b1_acc:.1f}% (T1={t1_acc:.0f}%, T2={t2_acc:.0f}%, T10={t10_acc:.0f}%)")
        print(f"  Block 2 (Reversal B):  {b2_acc:.1f}% (T11={t11_acc:.0f}%, T12={t12_acc:.0f}%, T20={t20_acc:.0f}%) [Adapt: {mean_adapt_b2:.1f} tr]")
        print(f"  Block 3 (Reversal A):  {b3_acc:.1f}% (T21={t21_acc:.0f}%, T22={t22_acc:.0f}%, T30={t30_acc:.0f}%) [Adapt: {mean_adapt_b3:.1f} tr]")
        print(f"  Telemetry: ||P_t||={mean_p:.4f} | Latency={mean_lat:.2f} ms")

    # Save JSON results
    json_path = output_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved JSON] {json_path}")

    # Save CSV results
    csv_path = output_dir / "ablation_results.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[Saved CSV]  {csv_path}")

    # Print Formatted Markdown Summary Table
    print("\n" + "=" * 85)
    print("FORMATTED MARKDOWN SUMMARY TABLE")
    print("=" * 85 + "\n")
    table_md = format_markdown_table(csv_rows)
    print(table_md)

    return all_results


def format_markdown_table(rows: List[Dict[str, str]]) -> str:
    lines = [
        "| Condition | Block 1 (Rule A) | Block 2 (Rev B) | Block 3 (Rev A) | Overall Acc | T11 (Rev) | T12 (Adapt) | T21 (Rev) | T22 (Adapt) | Mean ||P_t|| | Latency |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        lines.append(
            f"| **{r['condition']}** | {r['block1_acc']}% | {r['block2_reversal_acc']}% | {r['block3_reversal_acc']}% | {r['overall_acc']}% | {r['t11_reversal_acc']}% | {r['t12_adapt_acc']}% | {r['t21_reversal_acc']}% | {r['t22_adapt_acc']}% | {r['mean_p_norm']} | {r['mean_latency_ms']} ms |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate mechanistic ablations for Rapid Online Adaptation.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="runs/online_adaptation_colab_v2",
        help="Directory containing trained model checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/online_adaptation_colab_v2",
        help="Directory to write ablation_results.json and ablation_results.csv.",
    )
    parser.add_argument(
        "--n_sessions",
        type=int,
        default=10,
        help="Number of test sessions to evaluate per condition (default: 10).",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed for test sessions (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Evaluation device (default: cpu).",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)

    run_ablation_study(
        checkpoint_dir=ckpt_dir,
        output_dir=out_dir,
        n_sessions=args.n_sessions,
        base_seed=args.base_seed,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
