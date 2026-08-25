"""Bounded single-stream latency diagnostic for a Core V2 checkpoint."""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Config, CoreV2Model
from irene_brain.v2.maze_policy import CoreV2MazePolicy


def percentile(sorted_values: list[float], probability: float) -> float:
    index = min(len(sorted_values) - 1, int(probability * len(sorted_values)))
    return sorted_values[index]


def summarize(durations: list[float]) -> dict[str, float | int]:
    durations.sort()
    return {
        "iterations": len(durations),
        "mean_ms": round(statistics.mean(durations), 4),
        "p50_ms": round(percentile(durations, 0.50), 4),
        "p95_ms": round(percentile(durations, 0.95), 4),
        "p99_ms": round(percentile(durations, 0.99), 4),
        "max_ms": round(max(durations), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CoreV2Model(
        CoreV2Config(**payload["config"]),
        CONFIG_B_PREDICTIVE,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    observation = torch.rand(1, 3, 32, 32)
    previous_action = torch.zeros(1, dtype=torch.long)
    state = model.init_state(1, torch.device("cpu"))
    forward_durations = []
    with torch.no_grad():
        for iteration in range(args.warmup + args.iterations):
            start = time.perf_counter_ns()
            _, state = model(observation, state, prev_action=previous_action)
            duration_ms = (time.perf_counter_ns() - start) / 1e6
            if iteration >= args.warmup:
                forward_durations.append(duration_ms)

    policy = CoreV2MazePolicy(model)
    policy.reset(20260825)
    environment = MazeChaseEnv(
        ghost_count=5,
        ghost_period=1,
        ghost_rule="direct",
        max_ticks=args.warmup + args.iterations,
    )
    live_observation = environment.reset(20260825)
    policy_durations = []
    environment_durations = []
    loop_durations = []
    for iteration in range(args.warmup + args.iterations):
        loop_start = time.perf_counter_ns()
        policy_start = loop_start
        control = policy.act(live_observation)
        policy_finished = time.perf_counter_ns()
        outcome = environment.step(control)
        loop_finished = time.perf_counter_ns()
        live_observation = outcome.observation
        if iteration >= args.warmup:
            policy_durations.append((policy_finished - policy_start) / 1e6)
            environment_durations.append((loop_finished - policy_finished) / 1e6)
            loop_durations.append((loop_finished - loop_start) / 1e6)

    print({
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "scope": {
            "included": "pixels-to-control plus in-process simulated environment step",
            "excluded": "screen capture, serialization/network, physical HID, display effect",
        },
        "model_forward": summarize(forward_durations),
        "policy_act": summarize(policy_durations),
        "environment_step": summarize(environment_durations),
        "in_process_loop": summarize(loop_durations),
    })


if __name__ == "__main__":
    main()
