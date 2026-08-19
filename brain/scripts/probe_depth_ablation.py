"""Probe script to investigate cognitive depth scaling and C=2 anomaly."""

from __future__ import annotations

import os
import sys
import torch

from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.spatial_cognitive_diagnostics import (
    run_cognitive_depth_ablation,
    run_instrumented_diagnostic_episode,
)
from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
from irene_brain.model.lookahead_planner import LatentLookaheadPlanner
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel


def main() -> None:
    torch.set_num_threads(1)
    ckpt_path = "artifacts/checkpoints/best_champion_model.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt["model_config"]
        model = IreneBrainModel(config)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("Using random model")
        config = ThoughtFieldConfig.smoke()
        model = IreneBrainModel(config)

    print("\n--- Running Cognitive Depth Ablation Probe ---")
    res = run_cognitive_depth_ablation(
        model=model,
        seeds=(1702, 1703, 1704),
        cycles_list=(1, 2, 2, 2, 3, 4, 6),
        max_ticks=60,
    )
    for c, r in res.items():
        print(f"Cycles={c}: Pellets={r['mean_pellets']:.2f}, Catches={r['mean_catches']:.2f}, EffRank={r['mean_effective_rank']:.2f}, Latency={r['latency_ms_per_step']:.2f}ms")


if __name__ == "__main__":
    main()
