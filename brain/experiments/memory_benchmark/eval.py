"""Closed-loop evaluation script for KeysDoors Memory Benchmark.

Tests whether models have learned sequential memory:
1. Key Collection Rate
2. Door Opening Rate (requires remembering key possession)
3. Target Completion Rate
4. Key-to-Door Conditional Rate: P(door_opened | key_collected)
   -> A reactive model that cannot remember holding the key will wander or regress to key location.
   -> A recurrent model with working memory proceeds to unlock the door.
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

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.environments.keys_doors import KeysDoorsEnv
from memory_benchmark.expert import control_for_action
from memory_benchmark.models import make_model, N_FRAMES

def run_closed_loop_episode(
    model: nn.Module,
    env: KeysDoorsEnv,
    seed: int,
    max_ticks: int = 300,
    device: torch.device = torch.device("cpu"),
    temperature: float = 0.0,
    anti_stuck: int = 4,
) -> Dict:
    obs = env.reset(seed)
    model.eval()

    # Initial frame buffer: left-padded with 4 copies of first frame
    initial_pixels = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
    frame_buf = [initial_pixels.copy() for _ in range(N_FRAMES)]

    prev_action = 0
    h = None

    key_collected = False
    door_opened = False
    target_collected = False
    ticks_to_key = None
    ticks_to_door = None
    ticks_to_target = None

    recent_positions = []
    stuck_counter = 0

    for tick in range(max_ticks):
        # Format frames tensor: (1, N_FRAMES, 3, 16, 16) float32 [0, 1]
        raw_frames = np.stack(frame_buf[-N_FRAMES:]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(raw_frames).unsqueeze(0).to(device)
        prev_act_tensor = torch.tensor([prev_action], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, h = model(frames_tensor, prev_act_tensor, h)

        if temperature > 0.0:
            probs = torch.softmax(logits[0] / temperature, dim=-1).cpu().numpy()
            act = int(np.random.choice(len(probs), p=probs))
        else:
            # Argmax with anti-stuck heuristic if hovering in place
            if stuck_counter >= anti_stuck:
                # Take second best action to break wall collision lock
                top2 = torch.topk(logits[0], k=2).indices.cpu().numpy()
                act = int(top2[1])
                stuck_counter = 0
            else:
                act = int(logits[0].argmax().item())

        ctrl = control_for_action(act)
        step_out = env.step(ctrl)
        curr_pos = (env._player_x, env._player_y)

        # Track stuck / wall collisions
        if recent_positions and curr_pos == recent_positions[-1] and act != 0:
            stuck_counter += 1
        else:
            stuck_counter = 0
        recent_positions.append(curr_pos)

        # Update events
        if not key_collected and (env._has_key or "key_collected" in step_out.events):
            key_collected = True
            ticks_to_key = tick

        if not door_opened and (env._door_open or "door_opened" in step_out.events):
            door_opened = True
            ticks_to_door = tick

        if "target_collected" in step_out.events or step_out.reward > 0.0:
            target_collected = True
            ticks_to_target = tick
            break

        # Append new frame
        new_frame = np.frombuffer(step_out.observation.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
        frame_buf.append(new_frame)
        if len(frame_buf) > N_FRAMES:
            frame_buf.pop(0)

        prev_action = act

    return {
        "seed": seed,
        "key_collected": key_collected,
        "door_opened": door_opened,
        "target_collected": target_collected,
        "ticks_to_key": ticks_to_key,
        "ticks_to_door": ticks_to_door,
        "ticks_to_target": ticks_to_target,
        "total_ticks": tick + 1,
    }


def evaluate_model_on_split(
    model: nn.Module,
    start_seed: int = 3000,
    n_episodes: int = 25,
    max_ticks: int = 300,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    env = KeysDoorsEnv()
    episodes = []

    for i in range(n_episodes):
        seed = start_seed + i
        res = run_closed_loop_episode(model, env, seed=seed, max_ticks=max_ticks, device=device)
        episodes.append(res)

    key_count = sum(1 for e in episodes if e["key_collected"])
    door_count = sum(1 for e in episodes if e["door_opened"])
    target_count = sum(1 for e in episodes if e["target_collected"])

    # Key-to-door conditional rate
    key_to_door_rate = (door_count / key_count) if key_count > 0 else 0.0

    return {
        "n_episodes": n_episodes,
        "key_rate": key_count / n_episodes,
        "door_rate": door_count / n_episodes,
        "success_rate": target_count / n_episodes,
        "key_to_door_rate": key_to_door_rate,
        "key_count": key_count,
        "door_count": door_count,
        "target_count": target_count,
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on closed-loop KeysDoors")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--start_seed", type=int, default=3000)
    parser.add_argument("--n_episodes", type=int, default=25)
    parser.add_argument("--max_ticks", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    ckpt_path = Path(args.checkpoint)
    data = torch.load(ckpt_path, map_location="cpu")

    model_type = data["model_type"]
    model = make_model(model_type)
    model.load_state_dict(data["model_state_dict"])
    device = torch.device(args.device)
    model.to(device)

    print(f"Evaluating {model_type} (seed={data.get('seed')}) on {args.n_episodes} TEST episodes...")
    results = evaluate_model_on_split(
        model,
        start_seed=args.start_seed,
        n_episodes=args.n_episodes,
        max_ticks=args.max_ticks,
        device=device,
    )

    print(f"--- Results for {model_type} ---")
    print(f"Key Collection Rate : {results['key_rate'] * 100:.1f}% ({results['key_count']}/{results['n_episodes']})")
    print(f"Door Opening Rate   : {results['door_rate'] * 100:.1f}% ({results['door_count']}/{results['n_episodes']})")
    print(f"Target Success Rate : {results['success_rate'] * 100:.1f}% ({results['target_count']}/{results['n_episodes']})")
    print(f"Key -> Door Conv.   : {results['key_to_door_rate'] * 100:.1f}%")


if __name__ == "__main__":
    main()
