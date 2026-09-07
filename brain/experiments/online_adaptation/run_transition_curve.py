"""Dense Transition Curve Experiment: B in [16, 24, 32, 48, 64, 96].

Tests the hypothesis of an optimization phase transition:
Does adaptation reliability exhibit a sharp threshold between B=16 and B=64?

Normalizes total training exposure to 24,000 sequence samples (1,200 trajectory-equivalents):
- B=16: 1,500 steps (BEST_KNOWN baseline)
- B=24: 1,000 steps
- B=32: 750 steps
- B=48: 500 steps
- B=64: 375 steps
- B=96: 250 steps

Across seeds [42, 142, 242] using the proven accelerated pipeline (compiled + bfloat16 AMP).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from brain.experiments.run import run_single_experiment


def run_transition_sweep(
    output_dir: Path,
    data_dir: Path,
    batch_sizes: List[int] = [16, 24, 32, 48, 64, 96],
    seeds: List[int] = [42, 142, 242],
    device_str: str = "cuda",
    total_samples: int = 24000,
) -> List[Dict]:
    all_results = []
    print("=" * 80)
    print("DENSE OPTIMIZATION PHASE TRANSITION SWEEP: B in [16, 24, 32, 48, 64, 96]")
    print(f"Total sample budget normalized to {total_samples} sequences (1,200 trajectory-equivalents)")
    print("=" * 80)

    for b in batch_sizes:
        steps = total_samples // b
        b_dir = output_dir / f"sweep_b{b}"
        b_dir.mkdir(parents=True, exist_ok=True)

        for s in seeds:
            print(f"\n>>> Sweep: B={b:2d} | Steps={steps:4d} | Seed={s:3d}")
            res = run_single_experiment(
                experiment="online_adaptation",
                model_name="plastic_thoughtlet",
                seed=s,
                steps=steps,
                batch_size=b,
                device_str=device_str,
                output_dir=b_dir,
                data_dir=data_dir,
                compile_model=True,
                use_amp=True,
                total_samples=total_samples,
                grad_accum_steps=1,
                use_legacy_forward=False,
            )
            res["batch_size"] = b
            res["steps"] = steps
            all_results.append(res)
            write_transition_csv(all_results, output_dir / "transition_results.csv")

    return all_results


def write_transition_csv(results: List[Dict], out_csv: Path):
    if not results:
        return
    fieldnames = [
        "model",
        "seed",
        "batch_size",
        "steps",
        "grad_accum_steps",
        "effective_batch_size",
        "train_time_s",
        "total_time_s",
    ]
    all_train_keys = set()
    all_eval_keys = set()
    for r in results:
        all_train_keys.update(r.get("train_metrics", {}).keys())
        all_eval_keys.update(r.get("eval", {}).keys())
    train_keys = sorted(list(all_train_keys))
    eval_keys = sorted(list(all_eval_keys))
    fieldnames.extend(train_keys)
    fieldnames.extend(eval_keys)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in results:
            row = [
                r.get("model", "plastic_thoughtlet"),
                r.get("seed", 42),
                r.get("batch_size", 16),
                r.get("steps", 1500),
                r.get("grad_accum_steps", 1),
                r.get("effective_batch_size", r.get("batch_size", 16)),
                f"{r.get('train_time_s', 0.0):.1f}",
                f"{r.get('total_time_s', 0.0):.1f}",
            ]
            tr = r.get("train_metrics", {})
            for k in train_keys:
                val = tr.get(k, "")
                row.append(f"{val:.4f}" if isinstance(val, float) else str(val))
            ev = r.get("eval", {})
            for k in eval_keys:
                val = ev.get(k, "")
                row.append(f"{val:.3f}" if isinstance(val, float) else str(val))
            writer.writerow(row)
    print(f"[SUMMARY] Updated {out_csv.resolve()} ({len(results)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[16, 24, 32, 48, 64, 96])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 142, 242])
    parser.add_argument("--output_dir", type=str, default="/content/runs_transition_curve")
    parser.add_argument("--data_dir", type=str, default="/content/Pseudo-Brain/brain/datasets")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--total_samples", type=int, default=24000)

    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_transition_sweep(
        output_dir=out_dir,
        data_dir=data_dir,
        batch_sizes=args.batch_sizes,
        seeds=args.seeds,
        device_str=args.device,
        total_samples=args.total_samples,
    )
    print("\n[COMPLETE] Dense transition sweep finished!")
