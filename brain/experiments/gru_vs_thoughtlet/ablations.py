"""Ablation studies for thoughtlet architecture.

Tests:
1. Persistence (persistent vs reset every frame)
2. Thoughtlet count (K sweep: 1, 2, 4, 8, 16, 32)
3. Interaction (attention vs no attention)
4. Shared weights (shared BrainCell vs independent per-slot)
5. Thinking cycles (C sweep: 1, 2, 4)
"""
from __future__ import annotations

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

BRAIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BRAIN_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (
    ConvEncoder, BrainCell, ThoughtletModel, ThoughtletNoPersistence,
    ThoughtletNoAttention, IndependentPerSlot, ThoughtletK1,
    count_parameters, make_model,
    N_FRAMES, ACTION_CLASSES, FEATURE_DIM, prepare_input, _prev_onehot,
)
from train import CorpusSequenceDataset, collate_sequences, evaluate, get_device, train_model
from evaluate import FAMILY_CONFIGS, run_episode
from irene_brain.environments.maze_chase import MazeChaseEnv

INPUT_SIZE = FEATURE_DIM + ACTION_CLASSES  # 768 + 5 = 773


# --- Ablation model variants (canonical definitions live in models.py) ---
# make_ablation_model delegates to the factory so ablations train under the
# IDENTICAL recipe (sequence windows, scheduled sampling, label smoothing,
# padded buffers) as base models. The old separate train_ablation loop and
# duplicate class definitions were removed to kill the recipe confound.


def make_ablation_model(variant: str, K: int = 32, thought_size: int = 12, cycles: int = 1, **kwargs):
    """Create ablation model variant (delegates to models.make_model)."""
    if variant in ("no_persistence", "no_attention", "independent_slots",
                   "k1", "k2", "k4", "k8", "k16", "k32"):
        return make_model(variant, K=K, thought_size=thought_size, cycles=cycles, **kwargs)
    raise ValueError(f"Unknown ablation variant: {variant}")


# --- Training for ablations ---
# NOTE: ablations train through train.train_model (identical recipe to base:
# sequence windows, scheduled sampling, label smoothing, padded buffers).
# The old separate train_ablation loop was deleted to kill the recipe confound.

def train_ablation(
    variant: str,
    corpus_dir: str,
    n_steps: int = 3000,
    batch_size: int = 16,
    seq_len: int = 32,
    lr: float = 5e-4,
    seed: int = 42,
    device: str = "auto",
    save_dir: str = None,
    **model_kwargs,
) -> Dict:
    """Train a single ablation variant via the unified train_model recipe."""
    result = train_model(
        model_type=variant,
        corpus_dir=corpus_dir,
        n_steps=n_steps,
        batch_size=batch_size,
        seq_len=seq_len,
        lr=lr,
        seed=seed,
        save_dir=save_dir or str(BRAIN_ROOT / "runs" / "gru_vs_thoughtlet" / "ablations"),
        eval_every=min(500, n_steps),
        device=device,
        scheduled_sampling=True,
        model_kwargs=model_kwargs or None,
    )
    result["variant"] = variant
    return result


# --- Closed-loop evaluation for ablations ---

def eval_ablation_closed_loop(model, model_type: str, variant: str, seed: int,
                               n_seeds: int = 2, device: str = "cpu", temperature: float = 1.0) -> Dict:
    """Evaluate one ablation variant in closed-loop.

    temperature=1.0: argmax deadlocks all models against walls (0.00 pellets for every
    model), so sampling is required for a discriminative benchmark.
    """
    dev = get_device(device)
    if temperature > 0:
        torch.manual_seed(seed)
    results = {}

    for family_name, config in FAMILY_CONFIGS.items():
        family_results = []
        for ep_seed in range(n_seeds):
            env = MazeChaseEnv(**config)
            ep_result = run_episode(
                model=model, env=env, seed=seed * 1000 + ep_seed,
                device=dev, model_type="thoughtlet", max_ticks=config["max_ticks"],
                temperature=temperature,
            )
            family_results.append(ep_result)

        pellet_fractions = [r["pellets_eaten"] / max(r["total_pellets"], 1) for r in family_results]
        results[family_name] = {
            "pellet_fraction": float(np.mean(pellet_fractions)),
            "survival": float(np.mean([1 if r["survived"] else 0 for r in family_results])),
            "mean_latency_ms": float(np.mean([r["mean_latency_ms"] for r in family_results])),
        }

    avg_pf = np.mean([results[f]["pellet_fraction"] for f in FAMILY_CONFIGS])
    results["overall"] = {"mean_pellet_fraction": float(avg_pf)}
    return results


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Thoughtlet ablation studies")
    parser.add_argument("--corpus", default=str(BRAIN_ROOT / "datasets" / "embodied-corpus-v1"))
    parser.add_argument("--save-dir", default=str(BRAIN_ROOT / "runs" / "gru_vs_thoughtlet" / "ablations"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 142, 242])
    parser.add_argument("--device", default="auto", help="cpu, cuda, dml, dml:1, or auto")
    parser.add_argument("--variants", nargs="+", default=None, help="subset of variants to run")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define ablations (k32 = base thoughtlet config; base checkpoints reused, not retrained)
    ablations = {
        "no_persistence": {"K": 32, "thought_size": 12, "cycles": 1},
        "k1": {},
        "k2": {},
        "k4": {},
        "k8": {},
        "k16": {},
        "no_attention": {"K": 32, "thought_size": 12, "cycles": 1},
        "independent_slots": {"K": 32, "thought_size": 12, "cycles": 1},
    }

    all_results = {}

    if args.variants:
        ablations = {k: v for k, v in ablations.items() if k in args.variants}

    for variant, kwargs in ablations.items():
        print(f"\n=== Ablation: {variant} ===")
        if not args.skip_training:
            for seed in args.seeds:
                ckpt_path = save_dir / f"seed_{seed}_{variant}.pt"
                if ckpt_path.exists():
                    print(f"  seed={seed}: checkpoint exists, skipping")
                    continue
                print(f"  --- seed={seed} ---")
                train_ablation(
                    variant=variant,
                    corpus_dir=args.corpus,
                    n_steps=args.steps,
                    batch_size=args.batch_size,
                    seed=seed,
                    device=args.device,
                    save_dir=str(save_dir),
                    **kwargs,
                )
        # Eval across seeds (honest metrics computed separately; quick pellet check here)
        print(f"  {variant} done.")
        all_results[variant] = {"seeds": args.seeds}

    summary_path = save_dir / "ablation_runs.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
