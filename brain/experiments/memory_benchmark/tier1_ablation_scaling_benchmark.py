"""Tier 1 Cognitive Capacity Source-of-Scaling Ablation Benchmark.

Dissects the causal drivers behind capacity scaling when moving from Tier 0 (130k) to Tier 1 (~10M):
- Ablation A (Width Scaling): W in [24, 48, 96, 192, 384, 832] with fixed proj_dim=512.
- Ablation B (BrainCell Projection Scaling): proj_dim in [256, 512, 1024, 2048, 2816] with fixed W=48.
- Ablation C (Readout Head Capacity): Linear vs 2-layer MLP vs 3-layer MLP.
- Ablation D (Fast Plasticity): Synaptic Latching (P_t) enabled vs ablated (P_t = 0) at 10M scale.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from memory_benchmark.multi_threaded_cognitive_process_benchmark import (
    MultiThreadedCognitiveProcessEnv,
    build_mtcp_batch,
    train_mtcp_model,
    evaluate_mtcp_model,
    count_params,
    MTCPPseudoBrainModel,
)


def run_width_sweep(
    K: int = 64,
    widths: List[int] = [24, 48, 96, 192, 384, 832],
    proj_dim: int = 512,
    num_seeds: int = 10,
    train_steps: int = 100,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate capacity scaling across slot width W with fixed projection dimension."""
    device = torch.device(device_str)
    env = MultiThreadedCognitiveProcessEnv(num_threads=K, num_values=8)
    results = {}

    print(f"\n--- ABLATION A: WIDTH SCALING AT K={K} (proj_dim={proj_dim}) ---")
    for w in widths:
        model = MTCPPseudoBrainModel(input_dim=64, K=K, thought_size=w, proj_dim=proj_dim).to(device)
        p_count = count_params(model)["total"]
        train_mtcp_model(model, env, num_steps=train_steps, device=device)
        metrics = evaluate_mtcp_model(model, env, num_seeds=num_seeds, device=device)

        rec = metrics["recovery_accuracy"] / 100.0
        iso = (100.0 - metrics["cross_talk_error"]) / 100.0
        k_eff = K * rec * iso
        metrics["K_eff"] = k_eff
        metrics["params"] = p_count
        metrics["W"] = w
        results[f"W_{w}"] = metrics
        print(f"  W={w:3d} | Params: {p_count:8,d} | K_eff: {k_eff:5.2f} ({k_eff/K*100:4.1f}%) | Recov: {metrics['recovery_accuracy']:5.1f}% | Lat: {metrics['latency_ms_mean']:5.2f}ms")

    return results


def run_proj_sweep(
    K: int = 64,
    projs: List[int] = [256, 512, 1024, 2048, 2816],
    W: int = 48,
    num_seeds: int = 10,
    train_steps: int = 100,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate capacity scaling across BrainCell projection dimension with fixed W."""
    device = torch.device(device_str)
    env = MultiThreadedCognitiveProcessEnv(num_threads=K, num_values=8)
    results = {}

    print(f"\n--- ABLATION B: BRAINCELL PROJECTION SCALING AT K={K} (W={W}) ---")
    for p_dim in projs:
        model = MTCPPseudoBrainModel(input_dim=64, K=K, thought_size=W, proj_dim=p_dim).to(device)
        p_count = count_params(model)["total"]
        train_mtcp_model(model, env, num_steps=train_steps, device=device)
        metrics = evaluate_mtcp_model(model, env, num_seeds=num_seeds, device=device)

        rec = metrics["recovery_accuracy"] / 100.0
        iso = (100.0 - metrics["cross_talk_error"]) / 100.0
        k_eff = K * rec * iso
        metrics["K_eff"] = k_eff
        metrics["params"] = p_count
        metrics["proj_dim"] = p_dim
        results[f"proj_{p_dim}"] = metrics
        print(f"  proj={p_dim:4d} | Params: {p_count:8,d} | K_eff: {k_eff:5.2f} ({k_eff/K*100:4.1f}%) | Recov: {metrics['recovery_accuracy']:5.1f}% | Lat: {metrics['latency_ms_mean']:5.2f}ms")

    return results


def run_all_tier1_ablations(
    K: int = 64,
    device_str: str = "cpu",
    output_dir: Path = Path("brain/docs/runs"),
) -> Dict[str, Any]:
    """Run full battery of source-of-scaling ablations."""
    all_results = {
        "K": K,
        "width_sweep": run_width_sweep(K=K, device_str=device_str),
        "proj_sweep": run_proj_sweep(K=K, device_str=device_str),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "2026-09-07-tier1-scaling-ablations.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved ablation telemetry to {out_json}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="brain/docs/runs")
    args = parser.parse_args()

    run_all_tier1_ablations(K=args.threads, device_str=args.device, output_dir=Path(args.output_dir))
