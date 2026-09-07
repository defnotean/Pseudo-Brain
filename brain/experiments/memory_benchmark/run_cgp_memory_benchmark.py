"""Multi-seed Benchmark for Keys and Doors Sequential POMDP.

Compares:
1. Predictive CGP Thoughtlet (Consequence-Gated Plasticity + Cognitive Gating + Milestone Latching)
2. Vanilla Thoughtlet (K=32 parallel recurrent thoughtlets)
3. Standard GRU (384-d recurrent state)

Evaluated across multiple seeds (default: 42, 142, 242) on 25 held-out test episodes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from memory_benchmark.train import train_model, resolve_device
from memory_benchmark.eval import evaluate_model_on_split
from memory_benchmark.models import make_model, count_parameters


def run_cgp_benchmark(
    models: List[str] = ["cgp_thoughtlet", "thoughtlet", "gru"],
    seeds: List[int] = [42, 142, 242],
    steps: int = 2000,
    batch_size: int = 32,
    test_episodes: int = 25,
    device_str: str = "auto",
    corpus_dir: str = "brain/datasets/keys_doors_corpus_v1",
    output_dir: str = "brain/runs/memory_benchmark_cgp",
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device_str)

    print("=" * 80)
    print("KEYS AND DOORS SEQUENTIAL POMDP BENCHMARK")
    print(f"Models       : {models}")
    print(f"Seeds        : {seeds}")
    print(f"Steps        : {steps} | Batch Size: {batch_size}")
    print(f"Test Episodes: {test_episodes} (seeds 3000..{3000 + test_episodes - 1})")
    print(f"Device       : {device}")
    print("=" * 80)

    raw_results = {}

    for m in models:
        raw_results[m] = {}
        for s in seeds:
            print(f"\n[{m.upper()}] >>> Training Seed {s} ({steps} steps)...")
            t0 = time.time()
            model, train_info = train_model(
                model_type=m,
                corpus_dir=corpus_dir,
                seed=s,
                total_steps=steps,
                batch_size=batch_size,
                device_str=device_str,
                output_dir=ckpt_dir,
            )
            train_time = time.time() - t0

            print(f"[{m.upper()}] >>> Evaluating Seed {s} on {test_episodes} closed-loop test episodes...")
            eval_device = torch.device("cpu")
            model.to(eval_device)
            eval_res = evaluate_model_on_split(
                model=model,
                start_seed=3000,
                n_episodes=test_episodes,
                max_ticks=300,
                device=eval_device,
            )

            param_info = count_parameters(model)
            val_loss = train_info["history"][-1]["val_loss"] if train_info.get("history") else 0.0
            val_acc = train_info["history"][-1]["val_acc"] if train_info.get("history") else 0.0

            seed_data = {
                "seed": s,
                "train_time_s": train_time,
                "parameters": param_info["total"],
                "val_loss": val_loss,
                "val_acc": val_acc,
                "key_rate": eval_res["key_rate"],
                "door_rate": eval_res["door_rate"],
                "success_rate": eval_res["success_rate"],
                "key_to_door_rate": eval_res["key_to_door_rate"],
                "key_count": eval_res["key_count"],
                "door_count": eval_res["door_count"],
                "target_count": eval_res["target_count"],
            }
            raw_results[m][s] = seed_data

            print(
                f"  [{m} seed {s}] Key: {eval_res['key_rate']*100:.1f}% | "
                f"Door: {eval_res['door_rate']*100:.1f}% | "
                f"Success: {eval_res['success_rate']*100:.1f}% | "
                f"Key->Door: {eval_res['key_to_door_rate']*100:.1f}%"
            )

    # Compute aggregate statistics (mean +- std)
    summary_stats = {}
    for m in models:
        m_seeds = raw_results[m]
        key_rates = [m_seeds[s]["key_rate"] * 100 for s in seeds]
        door_rates = [m_seeds[s]["door_rate"] * 100 for s in seeds]
        succ_rates = [m_seeds[s]["success_rate"] * 100 for s in seeds]
        k2d_rates = [m_seeds[s]["key_to_door_rate"] * 100 for s in seeds]
        val_accs = [m_seeds[s]["val_acc"] * 100 for s in seeds]
        params = list(m_seeds.values())[0]["parameters"]
        t_time = np.mean([m_seeds[s]["train_time_s"] for s in seeds])

        summary_stats[m] = {
            "parameters": params,
            "mean_train_time_s": float(t_time),
            "key_rate_mean": float(np.mean(key_rates)),
            "key_rate_std": float(np.std(key_rates)),
            "door_rate_mean": float(np.mean(door_rates)),
            "door_rate_std": float(np.std(door_rates)),
            "success_rate_mean": float(np.mean(succ_rates)),
            "success_rate_std": float(np.std(succ_rates)),
            "k2d_rate_mean": float(np.mean(k2d_rates)),
            "k2d_rate_std": float(np.std(k2d_rates)),
            "val_acc_mean": float(np.mean(val_accs)),
            "val_acc_std": float(np.std(val_accs)),
        }

    full_output = {
        "metadata": {
            "models": models,
            "seeds": seeds,
            "steps": steps,
            "batch_size": batch_size,
            "test_episodes": test_episodes,
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": summary_stats,
        "per_seed": raw_results,
    }

    # Save JSON
    json_path = out_path / "memory_benchmark_cgp_results.json"
    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nSaved full benchmark results to {json_path}")

    # Print Summary Markdown Table
    print("\n" + "=" * 80)
    print("### BENCHMARK SUMMARY TABLE (Multi-Seed Mean +- Std)")
    print("=" * 80)
    print("| Model | Parameters | Val Acc | Key Rate | Door Rate | Key->Door Conv | Task Success Rate |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for m in models:
        s = summary_stats[m]
        p_str = f"{s['parameters']:,}"
        va_str = f"{s['val_acc_mean']:.1f} +- {s['val_acc_std']:.1f}%"
        k_str = f"{s['key_rate_mean']:.1f} +- {s['key_rate_std']:.1f}%"
        d_str = f"{s['door_rate_mean']:.1f} +- {s['door_rate_std']:.1f}%"
        k2d_str = f"{s['k2d_rate_mean']:.1f} +- {s['k2d_rate_std']:.1f}%"
        succ_str = f"**{s['success_rate_mean']:.1f} +- {s['success_rate_std']:.1f}%**"
        print(f"| `{m}` | {p_str} | {va_str} | {k_str} | {d_str} | {k2d_str} | {succ_str} |")
    print("=" * 80 + "\n")

    return full_output


def main():
    parser = argparse.ArgumentParser(description="Run Keys and Doors CGP Benchmark")
    parser.add_argument("--models", nargs="+", default=["cgp_thoughtlet", "thoughtlet", "gru"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 142, 242])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_episodes", type=int, default=25)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--corpus_dir", type=str, default="brain/datasets/keys_doors_corpus_v1")
    parser.add_argument("--output_dir", type=str, default="brain/runs/memory_benchmark_cgp")

    args = parser.parse_args()
    run_cgp_benchmark(
        models=args.models,
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        test_episodes=args.test_episodes,
        device_str=args.device,
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
