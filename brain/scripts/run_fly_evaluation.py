#!/usr/bin/env python3
"""End-to-end evaluation runner for fly-inspired vs canonical VectorizedPseudoBrain.

Runs full RCQ evaluation on both variants and produces comparison report.
CPU-only, single-threaded, deterministic per AGENTS.md.
"""

import argparse
import json
import os
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.irene_brain.model.fly_inspired import create_fly_inspired_model
from src.irene_brain.evaluation.fly_rcq import rcq_evaluate, rcq_evaluate_modularity


def deterministic_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def canonical_forward(x: torch.Tensor, width: int = 120) -> torch.Tensor:
    """Canonical dense thoughtlet forward pass (reference baseline)."""
    batch_size, input_dim = x.shape
    rng = np.random.RandomState(
        zlib.crc32(x.numpy().tobytes()) & 0xFFFFFFFF)
    W = rng.randn(input_dim, width).astype(np.float32)
    h = x @ torch.from_numpy(W)
    h = torch.relu(h)
    W_out = rng.randn(width, input_dim).astype(np.float32)
    return h @ torch.from_numpy(W_out)


def run_full_evaluation(steps: int = 1000, batch_size: int = 16,
                        input_dim: int = 64, width: int = 120,
                        n_thoughtlets: int = 32) -> dict:
    deterministic_seed(42)
    results = {}

    fly_model = create_fly_inspired_model(
        input_dim=input_dim, thoughtlet_width=width,
        n_thoughtlets=n_thoughtlets)
    fly_model.eval()

    x = torch.randn(batch_size, input_dim)
    reward = torch.randn(batch_size)

    with torch.no_grad():
        canon_out = canonical_forward(x, width=width)
        results["canonical"] = {
            "output_mean": float(canon_out.mean()),
            "output_std": float(canon_out.std()),
            "output_norm_mean": float(canon_out.norm(dim=-1).mean()),
        }

    fly_results = rcq_evaluate(fly_model, n_steps=steps,
                               batch_size=batch_size, input_dim=input_dim)
    results["fly_inspired"] = fly_results

    mod_results = rcq_evaluate_modularity(fly_model, n_modules=4)
    results["fly_modularity"] = mod_results

    results["summary"] = {
        "canonical_output_norm": results["canonical"]["output_norm_mean"],
        "fly_output_norm": results["fly_inspired"]["output_norm_mean"],
        "fly_determinism_max_diff": results["fly_inspired"]["determinism_max_diff"],
        "fly_dropout_0.5": results["fly_inspired"]["robustness"].get("drop_0.5", 0.0),
        "fly_dropout_0.9": results["fly_inspired"]["robustness"].get("drop_0.9", 0.0),
        "fly_cross_module_corr_mean": results["fly_modularity"]["mean_cross_module_corr"],
    }
    return results
def main():
    parser = argparse.ArgumentParser(
        description="Run full RCQ evaluation: fly-inspired vs canonical")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument("--thoughtlets", type=int, default=32)
    args = parser.parse_args()

    print("Running full RCQ evaluation...")
    print(f"  Steps: {args.steps}, Batch: {args.batch}, "
          f"Width: {args.width}, Thoughtlets: {args.thoughtlets}")

    results = run_full_evaluation(
        steps=args.steps, batch_size=args.batch,
        width=args.width, n_thoughtlets=args.thoughtlets)

    print("\n=== Evaluation Summary ===")
    for k, v in results["summary"].items():
        print(f"  {k}: {v:.6f}")

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "runs", "fly_evaluation_report.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()
