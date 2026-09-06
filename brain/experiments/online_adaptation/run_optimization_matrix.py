"""Master Experiment Orchestrator: Optimization Variable Isolation & Plasticity Scaling Curve.

Executes:
1. Phase 1: 2x2 Controlled Matrix (Original vs Accelerated Pipeline x B=16 vs B=64)
2. Phase 2: Gradient Accumulation (B=16 acc=1 vs B=16 acc=4 vs B=64 acc=1)
3. Phase 3: The Plasticity Scaling Curve (B in [4, 8, 16, 32, 64, 128])

All runs are normalized to 24,000 sequence samples (1,200 trajectory-equivalents of exposure).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from brain.experiments.run import run_single_experiment


def run_phase1_2x2_matrix(
    output_dir: Path,
    data_dir: Path,
    seeds: List[int] = [42, 142, 242],
    device_str: str = "cuda",
) -> List[Dict]:
    """Runs the 2x2 matrix on plastic_thoughtlet:
    Cell A: B=16, legacy_forward=True, compile=False, amp=False (1500 steps)
    Cell B: B=64, legacy_forward=True, compile=False, amp=False (375 steps)
    Cell C: B=16, legacy_forward=False, compile=True, amp=True (1500 steps)
    Cell D: B=64, legacy_forward=False, compile=True, amp=True (375 steps)
    """
    cells = [
        {"cell": "A_orig_b16", "batch_size": 16, "steps": 1500, "legacy": True, "compile": False, "amp": False},
        {"cell": "B_orig_b64", "batch_size": 64, "steps": 375, "legacy": True, "compile": False, "amp": False},
        {"cell": "C_accel_b16", "batch_size": 16, "steps": 1500, "legacy": False, "compile": True, "amp": True},
        {"cell": "D_accel_b64", "batch_size": 64, "steps": 375, "legacy": False, "compile": True, "amp": True},
    ]

    all_results = []
    print("=" * 80)
    print("PHASE 1: 2x2 CONTROLLED OPTIMIZATION MATRIX (PLASTIC THOUGHTLET)")
    print("Total sample budget normalized to 24,000 sequences (1,200 trajectory-equivalents)")
    print("=" * 80)

    for cfg in cells:
        c_name = cfg["cell"]
        b = cfg["batch_size"]
        steps = cfg["steps"]
        legacy = cfg["legacy"]
        comp = cfg["compile"]
        amp = cfg["amp"]

        cell_dir = output_dir / f"phase1_{c_name}"
        cell_dir.mkdir(parents=True, exist_ok=True)

        for s in seeds:
            print(f"\n>>> Running [{c_name}] Seed {s} | B={b} | Steps={steps} | Legacy={legacy} | Compile={comp} | AMP={amp}")
            res = run_single_experiment(
                experiment="online_adaptation",
                model_name="plastic_thoughtlet",
                seed=s,
                steps=steps,
                batch_size=b,
                device_str=device_str,
                output_dir=cell_dir,
                data_dir=data_dir,
                compile_model=comp,
                use_amp=amp,
                total_samples=24000,
                grad_accum_steps=1,
                use_legacy_forward=legacy,
            )
            res["cell"] = c_name
            all_results.append(res)

    return all_results


def run_phase2_gradient_accumulation(
    output_dir: Path,
    data_dir: Path,
    seeds: List[int] = [42, 142, 242],
    device_str: str = "cuda",
) -> List[Dict]:
    """Runs the gradient accumulation test:
    1. B=16, accum=1 (1500 steps, 1500 updates)
    2. B=16, accum=4 (1500 steps, 375 updates, effective B=64)
    3. B=64, accum=1 (375 steps, 375 updates, effective B=64)
    """
    configs = [
        {"name": "b16_accum1", "batch_size": 16, "accum": 1, "steps": 1500},
        {"name": "b16_accum4", "batch_size": 16, "accum": 4, "steps": 1500},
        {"name": "b64_accum1", "batch_size": 64, "accum": 1, "steps": 375},
    ]

    all_results = []
    print("=" * 80)
    print("PHASE 2: GRADIENT ACCUMULATION VS PHYSICAL BATCH SIZE")
    print("=" * 80)

    for cfg in configs:
        name = cfg["name"]
        b = cfg["batch_size"]
        accum = cfg["accum"]
        steps = cfg["steps"]

        cfg_dir = output_dir / f"phase2_{name}"
        cfg_dir.mkdir(parents=True, exist_ok=True)

        for s in seeds:
            print(f"\n>>> Running [{name}] Seed {s} | Physical B={b} | Accum={accum} (Eff B={b*accum}) | Steps={steps}")
            res = run_single_experiment(
                experiment="online_adaptation",
                model_name="plastic_thoughtlet",
                seed=s,
                steps=steps,
                batch_size=b,
                device_str=device_str,
                output_dir=cfg_dir,
                data_dir=data_dir,
                compile_model=True,
                use_amp=True,
                total_samples=24000,
                grad_accum_steps=accum,
                use_legacy_forward=False,
            )
            res["accum_config"] = name
            all_results.append(res)

    return all_results


def run_phase3_scaling_curve(
    output_dir: Path,
    data_dir: Path,
    batch_sizes: List[int] = [4, 8, 16, 32, 64, 128],
    seeds: List[int] = [42, 142, 242],
    device_str: str = "cuda",
) -> List[Dict]:
    """Runs the Plasticity Scaling Curve across B in [4, 8, 16, 32, 64, 128],
    normalizing steps = 24000 // B.
    """
    all_results = []
    print("=" * 80)
    print("PHASE 3: PLASTICITY SCALING CURVE (B in [4, 8, 16, 32, 64, 128])")
    print("=" * 80)

    for b in batch_sizes:
        steps = 24000 // b
        b_dir = output_dir / f"phase3_b{b}"
        b_dir.mkdir(parents=True, exist_ok=True)

        for s in seeds:
            print(f"\n>>> Scaling Curve: B={b:3d} | Steps={steps:4d} | Seed={s}")
            res = run_single_experiment(
                experiment="online_adaptation",
                model_name="plastic_thoughtlet",
                seed=s,
                steps=steps,
                batch_size=b,
                device_str=device_str,
                output_dir=b_dir,
                data_dir=data_dir,
                compile_model=(b >= 16),
                use_amp=True,
                total_samples=24000,
                grad_accum_steps=1,
                use_legacy_forward=False,
            )
            all_results.append(res)

    return all_results


def write_summary_csv(results: List[Dict], out_csv: Path):
    if not results:
        return
    fieldnames = [
        "model",
        "seed",
        "cell",
        "accum_config",
        "batch_size",
        "grad_accum_steps",
        "effective_batch_size",
        "train_time_s",
        "total_time_s",
    ]
    train_keys = sorted(list(results[0].get("train_metrics", {}).keys()))
    eval_keys = sorted(list(results[0].get("eval", {}).keys()))
    fieldnames.extend(train_keys)
    fieldnames.extend(eval_keys)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for r in results:
            row = [
                r.get("model", "plastic_thoughtlet"),
                r.get("seed", 42),
                r.get("cell", ""),
                r.get("accum_config", ""),
                r.get("batch_size", 16),
                r.get("grad_accum_steps", 1),
                r.get("effective_batch_size", 16),
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
    print(f"\n[SUMMARY] Saved consolidated results to {out_csv.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all", choices=["1", "2", "3", "all", "smoke"])
    parser.add_argument("--output_dir", type=str, default="/content/runs_optimization_matrix")
    parser.add_argument("--data_dir", type=str, default="brain/datasets")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 142, 242])

    args = parser.parse_args()
    out_path = Path(args.output_dir)
    data_path = Path(args.data_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []

    if args.phase in ["1", "all"]:
        r1 = run_phase1_2x2_matrix(out_path, data_path, seeds=args.seeds, device_str=args.device)
        results.extend(r1)
        write_summary_csv(results, out_path / "matrix_results.csv")

    if args.phase in ["2", "all"]:
        r2 = run_phase2_gradient_accumulation(out_path, data_path, seeds=args.seeds, device_str=args.device)
        results.extend(r2)
        write_summary_csv(results, out_path / "matrix_results.csv")

    if args.phase in ["3", "all"]:
        r3 = run_phase3_scaling_curve(out_path, data_path, seeds=args.seeds, device_str=args.device)
        results.extend(r3)
        write_summary_csv(results, out_path / "matrix_results.csv")

    if args.phase == "smoke":
        print("Running quick qualification smoke test (2 steps)...")
        r_smoke = run_single_experiment(
            experiment="online_adaptation",
            model_name="plastic_thoughtlet",
            seed=42,
            steps=2,
            batch_size=16,
            device_str=args.device,
            output_dir=out_path / "smoke_test",
            data_dir=data_path,
            compile_model=False,
            use_amp=False,
            total_samples=32,
            grad_accum_steps=2,
            use_legacy_forward=True,
        )
        print("Smoke test completed:", r_smoke["eval"])
