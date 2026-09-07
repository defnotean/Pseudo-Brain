"""
Micro-benchmark quantifying wall-clock planning latency and evaluated step complexity:
Static Exhaustive Lookahead vs. Dynamic Uncertainty-Gated Beam Search across horizons H in [2, 3, 4, 5].
"""

import time
import torch
from irene_brain.model import LatentLookaheadPlanner, ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel
from tests.test_lookahead_planner import tiny_config

def run_benchmark():
    config = tiny_config()
    model = IreneBrainModel(config)
    state = model.initial_state(1)
    sensors = torch.randn(1, config.sensor_tokens, config.core_width)

    horizons = [2, 3, 4, 5]
    repeats = 10

    print("=" * 80)
    print(f"{'Horizon H':<10} | {'Static Latency (ms)':<20} | {'Dynamic Latency (ms)':<20} | {'Speedup':<10} | {'Steps (Exh vs Dyn)':<20}")
    print("=" * 80)

    for h in horizons:
        planner_static = LatentLookaheadPlanner(model=model, config=config, horizon=h, dynamic_pruning=False)
        planner_dyn = LatentLookaheadPlanner(model=model, config=config, horizon=h, beam_width=4, dynamic_pruning=True)

        # Warmup
        planner_static.plan(state=state, sensors=sensors, horizon=h)
        planner_dyn.plan(state=state, sensors=sensors, horizon=h)

        # Measure static
        t0 = time.perf_counter()
        for _ in range(repeats):
            res_static = planner_static.plan(state=state, sensors=sensors, horizon=h)
        t_static = (time.perf_counter() - t0) / repeats * 1000.0

        # Measure dynamic
        t0 = time.perf_counter()
        for _ in range(repeats):
            res_dyn = planner_dyn.plan(state=state, sensors=sensors, horizon=h)
        t_dyn = (time.perf_counter() - t0) / repeats * 1000.0

        steps_static = sum(len(b.rollout_steps) for b in res_static.all_branches)
        steps_dyn = sum(len(b.rollout_steps) for b in res_dyn.all_branches)
        speedup = t_static / max(1e-5, t_dyn)

        print(f"{h:<10} | {t_static:<20.2f} | {t_dyn:<20.2f} | {speedup:<10.2f}x | {steps_static} vs {steps_dyn:<10}")

    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
