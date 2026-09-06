"""Unified Experiment & Batch Launcher for Pseudo-Brain.

Designed for turnkey, unattended remote execution on Google Colab, DGX Spark, and local workstations.

Usage:
  # Single run:
  python -m brain.experiments.run --experiment online_adaptation --model thoughtlet --seed 42 --steps 1500

  # Multi-model / multi-seed batch:
  python -m brain.experiments.run --experiment online_adaptation --models reactive gru thoughtlet --seeds 42 142 242 --steps 1500

  # Resuming an interrupted run:
  python -m brain.experiments.run --experiment online_adaptation --model thoughtlet --seed 42 --resume

  # Target Google Drive directly:
  python -m brain.experiments.run --experiment online_adaptation --output_dir /content/drive/MyDrive/PseudoBrain/runs
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_BRAIN_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BRAIN_DIR / "src"))
sys.path.insert(0, str(_BRAIN_DIR / "experiments"))
sys.path.insert(0, str(_BRAIN_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from irene_brain.device import resolve_device, get_hardware_summary, format_hardware_summary
from telemetry import TelemetryLogger


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def ensure_dataset(experiment: str, data_root: Path) -> Path:
    """Verifies that the required corpus exists; generates it automatically if missing."""
    data_root.mkdir(parents=True, exist_ok=True)

    if experiment == "online_adaptation":
        corpus_path = data_root / "hidden_rule_corpus_v1"
        if not (corpus_path / "train").exists() or len(list((corpus_path / "train").glob("*.npz"))) < 10:
            print(f"[DATASET] Generating missing corpus for {experiment} at {corpus_path}...")
            from online_adaptation.dataset import generate_corpus
            generate_corpus(corpus_path, n_train=80, n_dev=20, seed=1000)
        return corpus_path

    elif experiment in ("memory_benchmark", "self_correction"):
        corpus_path = data_root / "keys_doors_corpus_v1"
        if not (corpus_path / "train").exists() or len(list((corpus_path / "train").glob("*.npz"))) < 10:
            print(f"[DATASET] Generating missing corpus for {experiment} at {corpus_path}...")
            from memory_benchmark.generate_corpus import generate_corpus
            generate_corpus(corpus_path, n_train=80, n_dev=20, n_test=25, seed=1000)
        return corpus_path

    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def run_single_experiment(
    experiment: str,
    model_name: str,
    seed: int,
    steps: int,
    batch_size: int,
    device_str: str,
    output_dir: Path,
    data_dir: Path,
    resume: bool = False,
    auto_eval: bool = True,
    webhook_url: Optional[str] = None,
    use_wandb: bool = False,
) -> Dict[str, Any]:
    """Executes training and evaluation for a single (model, seed) pair."""
    exp_dir = output_dir / experiment / f"{model_name}_seed_{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    meta_file = exp_dir / "run_meta.json"
    eval_file = exp_dir / "eval_results.json"

    # Check if already completed
    if not resume and eval_file.exists():
        print(f"[{model_name.upper()} | Seed {seed}] Found completed eval_results.json. Skipping.")
        with open(eval_file, "r", encoding="utf-8") as f:
            return json.load(f)

    corpus_path = ensure_dataset(experiment, data_dir)
    hw_info = get_hardware_summary()
    commit_hash = get_git_commit()

    meta = {
        "experiment": experiment,
        "model": model_name,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "device": device_str,
        "git_commit": commit_hash,
        "hardware": hw_info,
        "start_time": time.time(),
        "status": "running",
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Initialize live telemetry engine
    telemetry = TelemetryLogger(
        run_dir=exp_dir,
        experiment=experiment,
        model_name=model_name,
        seed=seed,
        total_steps=steps,
        webhook_url=webhook_url,
        use_wandb=use_wandb,
    )

    t0 = time.time()

    try:
        # 1. Training Phase
        if experiment == "online_adaptation":
            from online_adaptation.train import train_model

            # Map model name
            actual_model = "thoughtlet" if "thoughtlet" in model_name else model_name
            model, train_info = train_model(
                model_type=actual_model,
                corpus_dir=corpus_path,
                seed=seed,
                total_steps=steps,
                batch_size=batch_size,
                device_str=device_str,
                output_dir=exp_dir,
                save_every=min(250, steps // 2 if steps > 2 else steps),
                resume=resume,
                telemetry=telemetry,
            )

        elif experiment == "self_correction":
            from self_correction.train_predictive import train_predictive_model

            use_plast = "plastic" in model_name
            actual_model = "thoughtlet" if "thoughtlet" in model_name else "gru"
            model, train_info = train_predictive_model(
                model_type=actual_model,
                corpus_dir=corpus_path,
                seed=seed,
                total_steps=steps,
                batch_size=batch_size,
                use_plasticity=use_plast,
                device_str=device_str,
                output_dir=exp_dir,
                save_every=min(250, steps // 2 if steps > 2 else steps),
                resume=resume,
                telemetry=telemetry,
            )

        elif experiment == "memory_benchmark":
            from memory_benchmark.train import train_model

            actual_model = model_name
            model, train_info = train_model(
                model_type=actual_model,
                corpus_dir=corpus_path,
                seed=seed,
                total_steps=steps,
                batch_size=batch_size,
                device_str=device_str,
                output_dir=exp_dir,
                save_every=min(250, steps // 2 if steps > 2 else steps),
                resume=resume,
                telemetry=telemetry,
            )

        train_elapsed = time.time() - t0

        # 2. Evaluation Phase
        eval_results = {}
        if auto_eval:
            print(f"[{model_name.upper()} | Seed {seed}] Running automated evaluation...")
            device = resolve_device(device_str)

            if experiment == "online_adaptation":
                from online_adaptation.run_online_adaptation_benchmark import evaluate_single_session
                from online_adaptation.hidden_rule_env import HiddenRuleEnv, Rule

                test_schedule = [(Rule.RULE_A, 10), (Rule.RULE_B, 10), (Rule.RULE_A, 10)]
                use_plast = "plastic" in model_name
                ablate_surp = "no_surprise" in model_name

                accuracies = []
                latencies = []
                for s_idx in range(10):  # 10 test sessions for quick single-run eval
                    env = HiddenRuleEnv(rule_schedule=test_schedule)
                    sess = evaluate_single_session(
                        model=model,
                        model_type=actual_model,
                        env=env,
                        seed=seed * 1000 + s_idx,
                        device=device,
                        use_plasticity=use_plast,
                        ablate_surprise=ablate_surp,
                    )
                    latencies.append(sess["latency_ms"])
                    correct_vec = [1.0 if t["correct"] else 0.0 for t in sess["trial_outcomes"]]
                    accuracies.append(correct_vec)

                acc_matrix = np.array(accuracies)
                mean_acc = np.mean(acc_matrix, axis=0) * 100.0
                eval_results = {
                    "t1_acc": float(mean_acc[0]),
                    "t2_acc": float(mean_acc[1]),
                    "t10_acc": float(mean_acc[9]),
                    "t11_acc": float(mean_acc[10]),
                    "t12_acc": float(mean_acc[11]),
                    "t20_acc": float(mean_acc[19]),
                    "t30_acc": float(mean_acc[29]),
                    "mean_latency_ms": float(np.mean(latencies)),
                }

            elif experiment == "memory_benchmark":
                from memory_benchmark.eval import evaluate_model_on_split
                eval_res = evaluate_model_on_split(
                    model=model,
                    start_seed=3000,
                    n_episodes=15,
                    max_ticks=300,
                    device=device,
                )
                eval_results = {
                    "key_rate": eval_res["key_rate"],
                    "door_rate": eval_res["door_rate"],
                    "success_rate": eval_res["success_rate"],
                    "key_to_door_rate": eval_res["key_to_door_rate"],
                }

            elif experiment == "self_correction":
                from self_correction.diagnostic_eval import evaluate_perturbation_diagnostics
                from self_correction.perturbations import SingleStepPerturbation
                diag = evaluate_perturbation_diagnostics(
                    model=model,
                    perturbation=SingleStepPerturbation(),
                    n_episodes=15,
                    start_seed=3000,
                    device=device,
                )
                eval_results = {
                    "recovery_rate": diag["recovery_rate"],
                    "mean_recovery_steps": diag["mean_recovery_steps"],
                    "failure_rate": diag["failure_rate"],
                }

    except Exception as e:
        telemetry.log_failure(str(e))
        meta["status"] = "failed"
        meta["error"] = str(e)
        meta["end_time"] = time.time()
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        raise e

    total_elapsed = time.time() - t0

    # Save final results and update meta
    meta["status"] = "completed"
    meta["end_time"] = time.time()
    meta["total_elapsed_s"] = total_elapsed
    meta["train_elapsed_s"] = train_elapsed
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    final_payload = {
        "model": model_name,
        "seed": seed,
        "train_time_s": train_elapsed,
        "total_time_s": total_elapsed,
        "eval": eval_results,
    }
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    # Telemetry completion alert
    telemetry.log_completion(eval_results=eval_results)

    print(f"[{model_name.upper()} | Seed {seed}] Completed in {total_elapsed:.1f}s. Results saved to {eval_file}")
    return final_payload


def run_batch(
    experiment: str,
    models: List[str],
    seeds: List[int],
    steps: int,
    batch_size: int,
    device_str: str,
    output_dir: Path,
    data_dir: Path,
    resume: bool = False,
    auto_eval: bool = True,
    webhook_url: Optional[str] = None,
    use_wandb: bool = False,
):
    """Iterates through all (model, seed) combinations, handling errors and aggregating results."""
    print("=" * 70)
    print("PSEUDO-BRAIN UNIFIED EXPERIMENT BATCH")
    print(f"Experiment: {experiment}")
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")
    print(f"Steps: {steps} | Device: {device_str}")
    print(f"Output Directory: {output_dir.resolve()}")
    if webhook_url:
        print("Telemetry: Webhook notifications active")
    if use_wandb:
        print("Telemetry: Weights & Biases logging active")
    print("=" * 70)

    results = []
    failures = []

    for m in models:
        for s in seeds:
            print(f"\n>>> Starting Run: Model={m} | Seed={s}")
            try:
                res = run_single_experiment(
                    experiment=experiment,
                    model_name=m,
                    seed=s,
                    steps=steps,
                    batch_size=batch_size,
                    device_str=device_str,
                    output_dir=output_dir,
                    data_dir=data_dir,
                    resume=resume,
                    auto_eval=auto_eval,
                    webhook_url=webhook_url,
                    use_wandb=use_wandb,
                )
                results.append(res)
            except Exception as e:
                print(f"[ERROR] Run failed for {m} (seed {s}): {e}")
                traceback.print_exc()
                failures.append({"model": m, "seed": s, "error": str(e)})

    # Write aggregate results
    agg_dir = output_dir / experiment
    agg_dir.mkdir(parents=True, exist_ok=True)

    agg_json = agg_dir / "aggregate_results.json"
    with open(agg_json, "w", encoding="utf-8") as f:
        json.dump({"completed": results, "failures": failures}, f, indent=2)

    # Write summary CSV
    agg_csv = agg_dir / "aggregate_results.csv"
    if results:
        fieldnames = ["model", "seed", "train_time_s", "total_time_s"]
        eval_keys = sorted(list(results[0].get("eval", {}).keys()))
        fieldnames.extend(eval_keys)

        with open(agg_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for r in results:
                row = [r["model"], r["seed"], f"{r['train_time_s']:.1f}", f"{r['total_time_s']:.1f}"]
                ev = r.get("eval", {})
                for k in eval_keys:
                    val = ev.get(k, "")
                    row.append(f"{val:.3f}" if isinstance(val, float) else str(val))
                writer.writerow(row)

    print("\n" + "=" * 70)
    print("BATCH SUMMARY")
    print(f"Completed Runs: {len(results)}")
    print(f"Failed Runs:    {len(failures)}")
    print(f"Aggregate JSON: {agg_json.resolve()}")
    if results:
        print(f"Aggregate CSV:  {agg_csv.resolve()}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Pseudo-Brain Unified Experiment Launcher")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["online_adaptation", "memory_benchmark", "self_correction"],
        help="Experiment suite to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model to train (e.g. reactive, gru, thoughtlet, plastic_thoughtlet)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of models to benchmark in batch",
    )
    parser.add_argument("--seed", type=int, default=42, help="Single seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="List of seeds for batch")
    parser.add_argument("--steps", type=int, default=1500, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Minibatch size")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="brain/runs",
        help="Root output directory (can be /content/drive/MyDrive/PseudoBrain/runs on Colab)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="brain/datasets",
        help="Root dataset directory",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--no_eval", action="store_true", help="Skip automatic evaluation")
    parser.add_argument("--webhook", type=str, default=None, help="Webhook URL for Discord/Slack alerts")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases telemetry")

    args = parser.parse_args()

    # Determine models list
    if args.models:
        model_list = args.models
    elif args.model:
        model_list = [args.model]
    else:
        # Defaults per experiment
        if args.experiment == "online_adaptation":
            model_list = ["reactive", "gru", "thoughtlet", "plastic_thoughtlet"]
        elif args.experiment == "memory_benchmark":
            model_list = ["reactive", "gru", "thoughtlet"]
        elif args.experiment == "self_correction":
            model_list = ["gru", "thoughtlet", "plastic_thoughtlet"]
        else:
            model_list = ["thoughtlet"]

    # Determine seeds list
    if args.seeds:
        seed_list = args.seeds
    else:
        seed_list = [args.seed]

    run_batch(
        experiment=args.experiment,
        models=model_list,
        seeds=seed_list,
        steps=args.steps,
        batch_size=args.batch_size,
        device_str=args.device,
        output_dir=Path(args.output_dir),
        data_dir=Path(args.data_dir),
        resume=args.resume,
        auto_eval=not args.no_eval,
        webhook_url=args.webhook,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
