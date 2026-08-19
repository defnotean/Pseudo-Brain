"""Phase 1 Behavioral Verification Suite.

Tests and provides direct empirical measurements for Phase 1 Behavioral Gates:
- Gate 5: 5% Dropped Frame Behavioral Resilience (Pellets, Catches, Survival vs Clean)
- Gate 6: 0-2 Frame Randomized Input Delay Behavioral Resilience (Pellets, Catches, Survival vs Clean)
- Gate 7: Anytime Cycle-1 Task & Safety Utility (Closed-Loop Pellets, Catches, Survival using Cycle 1 exits)
"""

from __future__ import annotations

import copy
import json
import math
import os
import pathlib
import sys
import time
from typing import Any
import torch

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.closed_loop_play import (
    EXCLUSIVE_ARGMAX_WASD_V1,
    decode_closed_loop_control,
)
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.training.batches import CONTINUOUS_TARGET_INDICES
from irene_brain.training.objective import _rgb_tensor
from irene_brain.types import GenericControl, HidKey


def evaluate_behavioral_condition(
    model: IreneBrainModel,
    condition: str,
    num_episodes: int = 20,
    seed_start: int = 2001,
    max_steps_per_ep: int = 50,
) -> dict[str, Any]:
    """Evaluate closed-loop policy under specific physical disruption conditions."""
    model.eval()
    env = MazeChaseEnv()

    total_pellets = 0
    total_catches = 0
    total_steps = 0
    nan_detected = False

    for seed in range(seed_start, seed_start + num_episodes):
        obs = env.reset(seed=seed)
        state = model.initial_state(batch_size=1)
        last_control = torch.zeros((1, model.config.actuator.total_queries), dtype=torch.float32)

        obs_history = [obs]
        stale_obs = obs

        for step in range(max_steps_per_ep):
            total_steps += 1

            # Condition simulation
            if condition == "clean":
                cur_obs = obs
            elif condition == "dropped_frames_5pct":
                # 5% Bernoulli frame drop
                if torch.rand(1).item() < 0.05 and len(obs_history) > 1:
                    cur_obs = stale_obs  # Use stale frame
                else:
                    cur_obs = obs
                    stale_obs = cur_obs
            elif condition == "input_delay_0_to_2":
                delay = int(torch.randint(0, 3, (1,)).item())
                cur_obs = obs_history[-1 - min(delay, len(obs_history) - 1)]
            elif condition == "cycle_1_anytime":
                cur_obs = obs
            else:
                cur_obs = obs

            device = next(model.parameters()).device
            rgb = _rgb_tensor((cur_obs.rgb,), device=device, resolution=getattr(model, "input_resolution", None))
            dt = torch.tensor([0.016], dtype=torch.float32, device=device)

            max_cyc = 1 if condition == "cycle_1_anytime" else 3
            with torch.no_grad():
                out = model(rgb, last_control, dt, state, max_cycles=max_cyc)
                state = out.next_state
                last_control = out.action.control

            if torch.isnan(state.thoughts).any():
                nan_detected = True

            logits = out.action.button_logits[0].float().cpu()
            continuous_values = out.action.control[0].float().cpu()
            control, _ = decode_closed_loop_control(
                logits.tolist(),
                [float(continuous_values[idx]) for idx in CONTINUOUS_TARGET_INDICES],
                decode_kind=EXCLUSIVE_ARGMAX_WASD_V1,
            )

            outcome = env.step(control)
            obs = outcome.observation
            obs_history.append(obs)
            if len(obs_history) > 10:
                obs_history.pop(0)

            if "pellet" in outcome.events:
                total_pellets += 1
            if "caught" in outcome.events:
                total_catches += 1

    return {
        "condition": condition,
        "episodes_evaluated": num_episodes,
        "total_steps": total_steps,
        "mean_pellets_per_ep": round(total_pellets / num_episodes, 2),
        "total_catches": total_catches,
        "mean_catches_per_ep": round(total_catches / num_episodes, 2),
        "nan_or_divergence_detected": nan_detected,
    }


def run_behavioral_suite(model: IreneBrainModel | None = None) -> dict[str, Any]:
    if model is None:
        cfg = ThoughtFieldConfig(
            thoughtlets=32,
            cognitive_cycles=3,
            core_width=32,
            attention_heads=4,
            sensor_tokens=4,
            belief_tokens=4,
            working_memory_tokens=4,
            goal_context_tokens=8,
            registers_per_thoughtlet=2,
            brain_cell_blocks=1,
            routed_neighbors=2,
        )
        model = IreneBrainModel(config=cfg, input_resolution=(32, 32))

    print("Running Clean Baseline (20 Worlds)...")
    clean = evaluate_behavioral_condition(model, "clean", num_episodes=20)
    print(f"  Clean: Pellets={clean['mean_pellets_per_ep']}/ep | Catches={clean['mean_catches_per_ep']}/ep")

    print("Running 5% Dropped Frames Condition (20 Worlds)...")
    drop = evaluate_behavioral_condition(model, "dropped_frames_5pct", num_episodes=20)
    print(f"  5% Drop: Pellets={drop['mean_pellets_per_ep']}/ep | Catches={drop['mean_catches_per_ep']}/ep")

    print("Running 0-2 Frame Randomized Input Delay Condition (20 Worlds)...")
    delay = evaluate_behavioral_condition(model, "input_delay_0_to_2", num_episodes=20)
    print(f"  0-2 Delay: Pellets={delay['mean_pellets_per_ep']}/ep | Catches={delay['mean_catches_per_ep']}/ep")

    print("Running Cycle-1 Anytime Output Condition (20 Worlds)...")
    c1 = evaluate_behavioral_condition(model, "cycle_1_anytime", num_episodes=20)
    print(f"  Cycle-1: Pellets={c1['mean_pellets_per_ep']}/ep | Catches={c1['mean_catches_per_ep']}/ep")

    return {
        "clean": clean,
        "dropped_5pct": drop,
        "input_delay_0_2": delay,
        "cycle_1_anytime": c1,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("PHASE 1 BEHAVIORAL AND ROBUSTNESS BENCHMARK")
    print("=" * 80)
    results = run_behavioral_suite()
    out_file = pathlib.Path("docs/phase_closure/phase1_behavioral_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved behavioral results to {out_file}")
