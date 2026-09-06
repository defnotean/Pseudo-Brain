"""Phase A: Diagnostic Evaluation of Policy Robustness & Recovery.

Deliberately injects mistakes into trained policies to evaluate whether they can
recover or whether they collapse into fatal deadlocks.

Records:
- recovery_rate: % of perturbed episodes reaching the target
- mean_recovery_steps: average ticks to regain course after perturbation
- failure_rate_after_perturbation: % of perturbed runs ending in timeout/deadlock
- goal_success_after_perturbation: target collection rate under perturbations vs clean
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.environments.keys_doors import KeysDoorsEnv
from memory_benchmark.expert import KeysDoorsExpert, control_for_action
from memory_benchmark.models import make_model, N_FRAMES
from self_correction.perturbations import PerturbationEngine, get_goal_distance


def run_diagnostic_episode(
    model: nn.Module,
    env: KeysDoorsEnv,
    seed: int,
    perturb_mode: str = "single",  # "none", "single", "burst", "slip"
    max_ticks: int = 300,
    device: torch.device = torch.device("cpu"),
    record_trace: bool = False,
) -> Dict:
    obs = env.reset(seed)
    expert = KeysDoorsExpert(env)
    model.eval()

    initial_pixels = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
    frame_buf = [initial_pixels.copy() for _ in range(N_FRAMES)]

    prev_action = 0
    h = None

    perturber = PerturbationEngine(mode=perturb_mode, inject_tick=25, seed=seed)

    trace = []
    success = False
    key_collected = False
    door_opened = False

    stuck_counter = 0
    recent_positions = []

    for tick in range(max_ticks):
        raw_frames = np.stack(frame_buf[-N_FRAMES:]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(raw_frames).unsqueeze(0).to(device)
        prev_act_tensor = torch.tensor([prev_action], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, h = model(frames_tensor, prev_act_tensor, h)

        # Model's proposed action
        proposed_action = int(logits[0].argmax().item())
        expert_action = expert.get_action()  # For analysis only

        # Perturbation injection
        if perturber.should_inject(tick):
            applied_action = perturber.get_perturbed_action(proposed_action)
            is_perturbed = True
        else:
            # Anti-stuck heuristic if hovering in place for > 4 steps
            if stuck_counter >= 4:
                top2 = torch.topk(logits[0], k=2).indices.cpu().numpy()
                applied_action = int(top2[1])
                stuck_counter = 0
            else:
                applied_action = proposed_action
            is_perturbed = False

        ctrl = control_for_action(applied_action)
        step_out = env.step(ctrl)

        curr_pos = (env._player_x, env._player_y)
        if recent_positions and curr_pos == recent_positions[-1] and applied_action != 0:
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

        if record_trace:
            trace.append({
                "tick": tick,
                "proposed_action": proposed_action,
                "applied_action": applied_action,
                "expert_action": expert_action,
                "is_perturbed": is_perturbed,
                "recovered": perturber.recovered,
                "distance_to_goal": get_goal_distance(env),
                "has_key": env._has_key,
                "door_open": env._door_open,
            })

        if success:
            break

        new_frame = np.frombuffer(step_out.observation.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
        frame_buf.append(new_frame)
        if len(frame_buf) > N_FRAMES:
            frame_buf.pop(0)

        prev_action = applied_action

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
        "total_ticks": tick + 1,
        "trace": trace if record_trace else None,
    }


def run_diagnostic_suite(
    model: nn.Module,
    seeds: List[int],
    modes: List[str] = ["none", "single", "burst"],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Dict]:
    env = KeysDoorsEnv()
    suite_results = {}

    for mode in modes:
        mode_episodes = []
        for s in seeds:
            res = run_diagnostic_episode(model, env, seed=s, perturb_mode=mode, device=device)
            mode_episodes.append(res)

        n = len(mode_episodes)
        succ = sum(1 for e in mode_episodes if e["success"])
        key_cnt = sum(1 for e in mode_episodes if e["key_collected"])
        door_cnt = sum(1 for e in mode_episodes if e["door_opened"])
        pert_cnt = sum(1 for e in mode_episodes if e["perturbed"])
        rec_cnt = sum(1 for e in mode_episodes if e["recovered"])

        rec_steps = [e["recovery_steps"] for e in mode_episodes if e["recovery_steps"] is not None]
        mean_rec_steps = float(np.mean(rec_steps)) if rec_steps else 0.0

        suite_results[mode] = {
            "n_episodes": n,
            "success_rate": succ / n,
            "key_rate": key_cnt / n,
            "door_rate": door_cnt / n,
            "perturbed_count": pert_cnt,
            "recovered_count": rec_cnt,
            "recovery_rate": (rec_cnt / pert_cnt) if pert_cnt > 0 else 1.0,
            "mean_recovery_steps": mean_rec_steps,
            "failure_rate": 1.0 - (succ / n),
        }

    return suite_results


def main():
    parser = argparse.ArgumentParser(description="Diagnostic evaluation of policy recovery")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--start_seed", type=int, default=3000)
    parser.add_argument("--n_episodes", type=int, default=25)
    parser.add_argument("--modes", type=str, nargs="+", default=["none", "single", "burst"])

    args = parser.parse_args()
    data = torch.load(args.checkpoint, map_location="cpu")
    model_type = data["model_type"]
    model = make_model(model_type)
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    seeds = [args.start_seed + i for i in range(args.n_episodes)]
    print(f"Running Diagnostic Suite for {model_type} across modes: {args.modes}")
    results = run_diagnostic_suite(model, seeds, modes=args.modes)

    print("\n--- Diagnostic Results ---")
    for mode, metrics in results.items():
        print(f"Mode [{mode.upper()}]:")
        print(f"  Goal Success Rate    : {metrics['success_rate']*100:.1f}%")
        print(f"  Recovery Rate        : {metrics['recovery_rate']*100:.1f}%")
        print(f"  Mean Recovery Steps  : {metrics['mean_recovery_steps']:.1f}")
        print(f"  Failure Rate         : {metrics['failure_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
