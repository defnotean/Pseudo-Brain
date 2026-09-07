"""Level 7 Capability Benchmark: Continual Multi-Reversal Evaluation.

Evaluates models across a 4-block continual reversal schedule:
[(Rule.RULE_A, 10), (Rule.RULE_B, 10), (Rule.RULE_A, 10), (Rule.RULE_B, 10)] (40 trials total).

Tests:
1. Reversal 1 (Rule A -> Rule B): T12 adaptation & T20 retention.
2. Reversal 2 (Rule B -> Rule A): T22 adaptation back to A & T30 retention.
3. Reversal 3 (Rule A -> Rule B): T32 adaptation again to B & T40 retention.
4. Fast plasticity trace dynamics (||P_t|| norm across continual reversals).
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

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]

sys.path.insert(0, str(_REPO_ROOT / "brain/src"))
sys.path.insert(0, str(_REPO_ROOT / "brain/experiments"))

from irene_brain.types import GenericControl, HidKey
from online_adaptation.hidden_rule_env import HiddenRuleEnv, Rule
from online_adaptation.models import (
    PredictiveGRUModel,
    PredictiveThoughtletModel,
    PredictiveCGPThoughtletModel,
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


def evaluate_continual_session(
    model: Any,
    model_type: str,
    env: HiddenRuleEnv,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    obs = env.reset(seed)
    model.eval()

    initial_pixels = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
    frame_history = [initial_pixels.copy() for _ in range(N_FRAMES)]

    recurrent_state = None
    P_t = torch.zeros(1, ACTION_CLASSES, device=device)
    prev_action = 0
    surprise = torch.zeros(1, 1, device=device)

    done = False
    p_norms = []

    while not done:
        stacked = np.stack(frame_history[-N_FRAMES:], axis=0).transpose(0, 3, 1, 2)
        frames_tensor = torch.from_numpy(stacked).float().unsqueeze(0).to(device) / 255.0
        prev_act_tensor = torch.tensor([prev_action], dtype=torch.long, device=device)

        with torch.no_grad():
            z_t = model.encode_observation(frames_tensor)
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
            z_hat = model.predict_next_latent(recurrent_state, torch.tensor([action], device=device))

        ctrl = CTRL_MAP.get(action, CTRL_MAP[0])
        outcome = env.step(ctrl)

        next_raw = np.frombuffer(outcome.observation.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
        frame_history.append(next_raw)

        next_stacked = np.stack(frame_history[-N_FRAMES:], axis=0).transpose(0, 3, 1, 2)
        next_tensor = torch.from_numpy(next_stacked).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            if hasattr(model, "predict_next_consequence") and getattr(model, "use_cgp", False):
                r_hat = model.predict_next_consequence(recurrent_state, torch.tensor([action], device=device))
                r_true = torch.tensor([outcome.reward], dtype=torch.float32, device=device)
                surprise = torch.abs(r_hat - r_true).unsqueeze(-1)
            else:
                z_true_next = model.encode_observation(next_tensor)
                surprise = torch.norm(z_hat - z_true_next, dim=-1, keepdim=True)

        prev_action = action
        p_norms.append(float(P_t.norm().item()))

        if outcome.terminated or outcome.truncated:
            done = True

    return {
        "trial_outcomes": env.trial_outcomes,
        "mean_p_norm": float(np.mean(p_norms)) if p_norms else 0.0,
        "max_p_norm": float(np.max(p_norms)) if p_norms else 0.0,
    }


def run_benchmark(n_sessions: int = 10, device_str: str = "cpu"):
    device = torch.device(device_str)
    print("=" * 85)
    print("LEVEL 7 CAPABILITY BENCHMARK: CONTINUAL 4-BLOCK MULTI-REVERSAL")
    print(f"Device: {device} | Sessions per condition: {n_sessions}")
    print("Schedule: Block 1 (A, 10) -> Block 2 (B, 10) -> Block 3 (A, 10) -> Block 4 (B, 10)")
    print("=" * 85)

    schedule = [
        (Rule.RULE_A, 10),
        (Rule.RULE_B, 10),
        (Rule.RULE_A, 10),
        (Rule.RULE_B, 10),
    ]

    checkpoints_to_eval = [
        # CGP Thoughtlet (distractor-trained)
        ("CGP Thoughtlet (Seed 42)", "cgp_thoughtlet", _REPO_ROOT / "runs/online_adaptation_cgp/checkpoints/cgp_thoughtlet_seed_42.pt"),
        ("CGP Thoughtlet (Seed 142)", "cgp_thoughtlet", _REPO_ROOT / "runs/online_adaptation_cgp/checkpoints/cgp_thoughtlet_seed_142.pt"),
        ("CGP Thoughtlet (Seed 242)", "cgp_thoughtlet", _REPO_ROOT / "runs/online_adaptation_cgp/checkpoints/cgp_thoughtlet_seed_242.pt"),
        # Baseline Plastic Thoughtlet (distractor-trained)
        ("Baseline Plastic (Seed 42)", "plastic_thoughtlet", _REPO_ROOT / "runs/online_adaptation_cgp/checkpoints/plastic_thoughtlet_seed_42.pt"),
        ("Baseline Plastic (Seed 142)", "plastic_thoughtlet", _REPO_ROOT / "runs/online_adaptation_cgp/checkpoints/plastic_thoughtlet_seed_142.pt"),
        ("Baseline Plastic (Seed 242)", "plastic_thoughtlet", _REPO_ROOT / "runs/online_adaptation_cgp/checkpoints/plastic_thoughtlet_seed_242.pt"),
        # Clean trained models from v2
        ("Original Plastic (v2 Seed 42)", "plastic_thoughtlet", _REPO_ROOT / "runs/online_adaptation_colab_v2/plastic_thoughtlet_seed_42.pt"),
        ("Pure Thoughtlet (v2 Seed 42)", "thoughtlet", _REPO_ROOT / "runs/online_adaptation_colab_v2/thoughtlet_seed_42.pt"),
        ("GRU Baseline (v2 Seed 42)", "gru", _REPO_ROOT / "runs/online_adaptation_colab_v2/gru_seed_42.pt"),
    ]

    all_results = {}

    for name, m_type, ckpt_path in checkpoints_to_eval:
        if not ckpt_path.exists():
            print(f"Skipping {name}: {ckpt_path} not found.")
            continue

        print(f"\n>>> Evaluating {name}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        if m_type == "cgp_thoughtlet":
            model = PredictiveCGPThoughtletModel().to(device)
        elif m_type == "plastic_thoughtlet":
            model = PredictiveThoughtletModel(use_plasticity=True).to(device)
        elif m_type == "thoughtlet":
            model = PredictiveThoughtletModel(use_plasticity=False).to(device)
        elif m_type == "gru":
            model = PredictiveGRUModel().to(device)
        else:
            raise ValueError(f"Unknown model type: {m_type}")

        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        t10_list, t12_list, t20_list, t22_list, t30_list, t32_list, t40_list = [], [], [], [], [], [], []
        total_acc_list = []
        p_norm_list = []

        for ep in range(n_sessions):
            env = HiddenRuleEnv(rule_schedule=schedule)
            sess_res = evaluate_continual_session(
                model=model,
                model_type=m_type,
                env=env,
                seed=42000 + ep * 7,
                device=device,
            )
            outcomes = sess_res["trial_outcomes"]
            correct_flags = [1.0 if t["correct"] else 0.0 for t in outcomes]

            if len(outcomes) >= 40:
                t10_list.append(correct_flags[9])    # End Block 1 (A)
                t12_list.append(correct_flags[11])   # Rev 1 Adapt (B)
                t20_list.append(correct_flags[19])   # End Block 2 (B)
                t22_list.append(correct_flags[21])   # Rev 2 Adapt (A)
                t30_list.append(correct_flags[29])   # End Block 3 (A)
                t32_list.append(correct_flags[31])   # Rev 3 Adapt (B)
                t40_list.append(correct_flags[39])   # End Block 4 (B)
                total_acc_list.append(np.mean(correct_flags) * 100.0)
                p_norm_list.append(sess_res["mean_p_norm"])

        res_summary = {
            "name": name,
            "model_type": m_type,
            "t10_retention_A": float(np.mean(t10_list) * 100.0),
            "t12_rev1_adapt_B": float(np.mean(t12_list) * 100.0),
            "t20_retention_B": float(np.mean(t20_list) * 100.0),
            "t22_rev2_adapt_A": float(np.mean(t22_list) * 100.0),
            "t30_retention_A": float(np.mean(t30_list) * 100.0),
            "t32_rev3_adapt_B": float(np.mean(t32_list) * 100.0),
            "t40_retention_B": float(np.mean(t40_list) * 100.0),
            "overall_session_acc": float(np.mean(total_acc_list)),
            "mean_p_norm": float(np.mean(p_norm_list)),
        }
        all_results[name] = res_summary

        print(
            f"  Rev 1 (->B): {res_summary['t12_rev1_adapt_B']:5.1f}% | "
            f"Rev 2 (->A): {res_summary['t22_rev2_adapt_A']:5.1f}% | "
            f"Rev 3 (->B): {res_summary['t32_rev3_adapt_B']:5.1f}% | "
            f"Overall Acc: {res_summary['overall_session_acc']:5.1f}% | "
            f"||P||: {res_summary['mean_p_norm']:.3f}"
        )

    # Save results
    out_dir = _REPO_ROOT / "runs/online_adaptation_cgp"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "multi_reversal_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print markdown table
    print("\n" + "=" * 95)
    print("CONTINUAL 4-BLOCK MULTI-REVERSAL BENCHMARK RESULTS")
    print("=" * 95)
    print("| Model / Checkpoint | Rev 1 Adapt ($T_{12}$) | Block 2 Ret ($T_{20}$) | Rev 2 Adapt ($T_{22}$) | Block 3 Ret ($T_{30}$) | Rev 3 Adapt ($T_{32}$) | Overall Acc |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for name, r in all_results.items():
        print(
            f"| **{name}** | {r['t12_rev1_adapt_B']:5.1f}% | {r['t20_retention_B']:5.1f}% | "
            f"{r['t22_rev2_adapt_A']:5.1f}% | {r['t30_retention_A']:5.1f}% | "
            f"{r['t32_rev3_adapt_B']:5.1f}% | **{r['overall_session_acc']:5.1f}%** |"
        )
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run_benchmark(n_sessions=args.sessions, device_str=args.device)
