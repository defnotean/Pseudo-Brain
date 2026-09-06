"""Phase H: Master Self-Correction Benchmark Suite.

Evaluates and compares:
1. Baseline BC (no prediction)
2. Predictive BC (prediction + surprise)
3. Predictive BC + Plasticity (prediction + surprise + fast episodic adaptation)

Across:
- Clean Performance
- Single-Error Recovery
- Multi-Error Recovery (Burst)
- Movement Disturbance Recovery (Slip)

Measures:
- Normal Success Rate
- Perturbed Success Rate
- Recovery Rate (%)
- Mean Recovery Steps
- Failure Rate (%)
- Mean Prediction Error (Surprise)
- Inference Latency (ms/tick)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.environments.keys_doors import KeysDoorsEnv
from memory_benchmark.expert import control_for_action
from memory_benchmark.models import make_model, N_FRAMES
from self_correction.perturbations import PerturbationEngine, get_goal_distance
from self_correction.predictive_models import PredictiveGRUModel, PredictiveThoughtletModel


def evaluate_predictive_episode(
    model: nn.Module,
    env: KeysDoorsEnv,
    seed: int,
    perturb_mode: str = "single",
    max_ticks: int = 300,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    obs = env.reset(seed)
    model.eval()

    initial_pixels = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
    frame_buf = [initial_pixels.copy() for _ in range(N_FRAMES)]

    prev_action = 0
    h = None
    P_t = None
    surprise = torch.zeros(1, 1, device=device)
    predicted_next_z = None

    perturber = PerturbationEngine(mode=perturb_mode, inject_tick=25, seed=seed)

    success = False
    key_collected = False
    door_opened = False

    stuck_counter = 0
    recent_positions = []
    surprises = []

    t0_inf = time.perf_counter()

    for tick in range(max_ticks):
        raw_frames = np.stack(frame_buf[-N_FRAMES:]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(raw_frames).unsqueeze(0).to(device)
        prev_act_tensor = torch.tensor([prev_action], dtype=torch.long, device=device)

        with torch.no_grad():
            # 1. Encode observation
            z_t = model.encode_observation(frames_tensor)

            # 2. Check prediction error from previous tick foresight
            if predicted_next_z is not None:
                err = torch.norm(predicted_next_z - z_t, dim=-1, keepdim=True)
                surprise = err
                surprises.append(float(err.item()))
            else:
                surprise = torch.zeros(1, 1, device=device)

            # 3. Step forward through recurrent & surprise channels
            logits, h, e_t, P_t = model.forward_step(z_t, prev_act_tensor, surprise, h, P_t)

            # 4. Model proposed action
            proposed_act = int(logits[0].argmax().item())

            # 5. Predict next latent z_hat_{t+1}
            predicted_next_z = model.predict_next_latent(h, torch.tensor([proposed_act], device=device))

        # Check perturbation
        if perturber.should_inject(tick):
            applied_act = perturber.get_perturbed_action(proposed_act)
        else:
            if stuck_counter >= 4:
                top2 = torch.topk(logits[0], k=2).indices.cpu().numpy()
                applied_act = int(top2[1])
                stuck_counter = 0
            else:
                applied_act = proposed_act

        ctrl = control_for_action(applied_act)
        step_out = env.step(ctrl)

        curr_pos = (env._player_x, env._player_y)
        if recent_positions and curr_pos == recent_positions[-1] and applied_act != 0:
            stuck_counter += 1
        else:
            stuck_counter = 0
        recent_positions.append(curr_pos)

        perturber.update_recovery_state(env, tick)

        if not key_collected and (env._has_key or "key_collected" in step_out.events):
            key_collected = True
        if not door_opened and (env._door_open or "door_opened" in step_out.events):
            door_opened = True
        if "target_collected" in step_out.events or step_out.reward > 0.0:
            success = True
            break

        new_frame = np.frombuffer(step_out.observation.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
        frame_buf.append(new_frame)
        if len(frame_buf) > N_FRAMES:
            frame_buf.pop(0)

        prev_action = applied_act

    total_ticks = tick + 1
    latency_ms = ((time.perf_counter() - t0_inf) / total_ticks) * 1000.0

    rec_steps = None
    if perturber.injected and perturber.recovered:
        end_inj = perturber.injection_end_tick or perturber.injection_start_tick
        rec_steps = max(0, (perturber.recovery_tick or tick) - end_inj)

    return {
        "seed": seed,
        "mode": perturb_mode,
        "success": success,
        "key_collected": key_collected,
        "door_opened": door_opened,
        "perturbed": perturber.injected,
        "recovered": perturber.recovered,
        "recovery_steps": rec_steps,
        "total_ticks": total_ticks,
        "mean_surprise": float(np.mean(surprises)) if surprises else 0.0,
        "latency_ms": latency_ms,
    }


def evaluate_predictive_suite(
    model: nn.Module,
    seeds: List[int],
    modes: List[str] = ["none", "single", "burst"],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Dict]:
    env = KeysDoorsEnv()
    results = {}

    for mode in modes:
        episodes = []
        for s in seeds:
            res = evaluate_predictive_episode(model, env, seed=s, perturb_mode=mode, device=device)
            episodes.append(res)

        n = len(episodes)
        succ = sum(1 for e in episodes if e["success"])
        pert_cnt = sum(1 for e in episodes if e["perturbed"])
        rec_cnt = sum(1 for e in episodes if e["recovered"])

        rec_steps = [e["recovery_steps"] for e in episodes if e["recovery_steps"] is not None]
        mean_rec_steps = float(np.mean(rec_steps)) if rec_steps else 0.0
        mean_surprise = float(np.mean([e["mean_surprise"] for e in episodes]))
        mean_latency = float(np.mean([e["latency_ms"] for e in episodes]))

        results[mode] = {
            "n_episodes": n,
            "success_rate": succ / n,
            "recovery_rate": (rec_cnt / pert_cnt) if pert_cnt > 0 else 1.0,
            "mean_recovery_steps": mean_rec_steps,
            "failure_rate": 1.0 - (succ / n),
            "mean_surprise": mean_surprise,
            "latency_ms": mean_latency,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--start_seed", type=int, default=3000)
    parser.add_argument("--n_episodes", type=int, default=25)
    parser.add_argument("--modes", type=str, nargs="+", default=["none", "single", "burst"])

    args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_type = ckpt["model_type"]
    use_plast = ckpt.get("use_plasticity", False)

    if model_type == "gru":
        model = PredictiveGRUModel(use_plasticity=use_plast)
    elif model_type == "thoughtlet":
        model = PredictiveThoughtletModel(use_plasticity=use_plast)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    seeds = [args.start_seed + i for i in range(args.n_episodes)]
    res = evaluate_predictive_suite(model, seeds, modes=args.modes)

    print(f"\n--- Self-Correction Benchmark for {model_type} (plasticity={use_plast}) ---")
    for mode, m in res.items():
        print(f"Mode [{mode.upper()}]:")
        print(f"  Success Rate   : {m['success_rate']*100:.1f}%")
        print(f"  Recovery Rate  : {m['recovery_rate']*100:.1f}%")
        print(f"  Recovery Steps : {m['mean_recovery_steps']:.1f}")
        print(f"  Surprise Error : {m['mean_surprise']:.4f}")
        print(f"  Latency        : {m['latency_ms']:.2f} ms/tick")


if __name__ == "__main__":
    main()
