"""Master Runner: Consequence-Gated Plasticity (CGP) & Cognitive Input Gating Benchmark.

Runs:
1. Generation of distractor-aware corpus (/content/corpus_distractor_v1) with D in [0, 2, 5, 8].
2. Multi-seed training of:
   - cgp_thoughtlet (Consequence-Gated Plasticity + Cognitive Input Gating)
   - plastic_thoughtlet (Baseline Plastic Thoughtlet on distractor data)
   Normalized to B=32, 750 steps (24,000 sequence samples) with bfloat16 AMP and compile.
3. Multi-seed honest evaluation across delay horizons D in [0, 10, 25, 50, 100] without any oracle cheats.
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "brain/src"))
sys.path.insert(0, str(_REPO_ROOT / "brain/experiments"))

from online_adaptation.dataset import generate_corpus
from online_adaptation.train import train_model
from online_adaptation.distractor_benchmark import (
    DistractorHiddenRuleEnv,
    Rule,
    evaluate_session_under_distractor,
)
from online_adaptation.models import (
    PredictiveThoughtletModel,
    PredictiveCGPThoughtletModel,
)


def run_experiment(
    corpus_dir: Path,
    output_dir: Path,
    seeds: List[int] = [42, 142, 242],
    models: List[str] = ["cgp_thoughtlet", "plastic_thoughtlet"],
    batch_size: int = 32,
    total_samples: int = 24000,
    delays: List[int] = [0, 10, 25, 50, 100],
    eval_sessions: int = 10,
    device_str: str = "cuda",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    total_steps = total_samples // batch_size  # 750 steps at B=32

    print("=" * 85, flush=True)
    print("CONSEQUENCE-GATED PLASTICITY (CGP) & COGNITIVE GATING BENCHMARK", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    print(f"Corpus: {corpus_dir}", flush=True)
    print(f"Models: {models} | Seeds: {seeds} | Batch Size: {batch_size} | Steps: {total_steps}", flush=True)
    print(f"Evaluation Delays D: {delays} | Sessions per seed: {eval_sessions}", flush=True)
    print("=" * 85, flush=True)

    # 1. Generate distractor-aware corpus if not present
    train_dir = corpus_dir / "train"
    if not train_dir.exists() or len(list(train_dir.glob("*.npz"))) < 80:
        print(f"\n[CORPUS] Generating distractor-aware corpus into {corpus_dir}...", flush=True)
        generate_corpus(
            output_dir=corpus_dir,
            n_train=80,
            n_dev=20,
            seed=1000,
            distractor_delays=[0, 2, 5, 8],
            distractor_noise_level=35.0,
        )
    else:
        print(f"\n[CORPUS] Using existing corpus at {corpus_dir} ({len(list(train_dir.glob('*.npz')))} files).", flush=True)

    # 2. Train models
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for m_type in models:
        for seed in seeds:
            ckpt_path = ckpt_dir / f"{m_type}_seed_{seed}.pt"
            if ckpt_path.exists():
                print(f"[TRAIN] Checkpoint {ckpt_path.name} already exists. Skipping training.", flush=True)
                continue

            print(f"\n[TRAIN] Training {m_type} (seed={seed}, batch_size={batch_size}, steps={total_steps})...", flush=True)
            t0 = time.time()
            train_model(
                model_type=m_type,
                corpus_dir=corpus_dir,
                seed=seed,
                total_steps=total_steps,
                batch_size=batch_size,
                grad_accum_steps=1,
                lr=5e-4,
                pred_weight=0.5,
                device_str=device_str,
                output_dir=ckpt_dir,
                compile_model=True,
                use_amp=True,
            )
            print(f"[TRAIN] Finished {m_type}_seed_{seed} in {time.time() - t0:.1f}s", flush=True)

    # 3. Honest Multi-Seed Distractor Evaluation
    print("\n" + "=" * 85, flush=True)
    print("HONEST DISTRACTOR EVALUATION (ZERO ORACLE CHEATS)", flush=True)
    print("=" * 85, flush=True)

    test_schedule = [(Rule.RULE_A, 10), (Rule.RULE_B, 10)]
    all_eval_results: Dict[str, Dict] = {}

    for m_type in models:
        print(f"\n>>> Evaluating {m_type.upper()}...", flush=True)
        all_eval_results[m_type] = {}

        for D in delays:
            t10_all = []
            t11_all = []
            t12_all = []

            for seed in seeds:
                ckpt_path = ckpt_dir / f"{m_type}_seed_{seed}.pt"
                ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)

                if m_type == "cgp_thoughtlet":
                    model = PredictiveCGPThoughtletModel().to(device)
                else:
                    model = PredictiveThoughtletModel(use_plasticity=True).to(device)

                model.load_state_dict(ckpt_data["model_state_dict"], strict=False)
                model.eval()

                for ep in range(eval_sessions):
                    sess_seed = seed * 1000 + ep
                    env = DistractorHiddenRuleEnv(
                        rule_schedule=test_schedule,
                        distractor_delay=D,
                        distractor_noise_level=35.0,
                    )
                    res = evaluate_session_under_distractor(
                        model=model,
                        model_type=m_type,
                        env=env,
                        seed=sess_seed,
                        device=device,
                        use_plasticity=True,
                    )
                    outcomes = res["trial_outcomes"]
                    t10_all.append(1.0 if (len(outcomes) > 9 and outcomes[9]["correct"]) else 0.0)
                    t11_all.append(1.0 if (len(outcomes) > 10 and outcomes[10]["correct"]) else 0.0)
                    t12_all.append(1.0 if (len(outcomes) > 11 and outcomes[11]["correct"]) else 0.0)

            m_t10 = float(np.mean(t10_all) * 100.0)
            s_t10 = float(np.std(t10_all) * 100.0)
            m_t11 = float(np.mean(t11_all) * 100.0)
            m_t12 = float(np.mean(t12_all) * 100.0)
            s_t12 = float(np.std(t12_all) * 100.0)

            all_eval_results[m_type][str(D)] = {
                "delay": D,
                "t10_mean": m_t10,
                "t10_std": s_t10,
                "t11_mean": m_t11,
                "t12_mean": m_t12,
                "t12_std": s_t12,
            }
            print(f"  Delay D = {D:3d} | Rule A (T10): {m_t10:5.1f}% ± {s_t10:4.1f}% | Reversal (T11): {m_t11:5.1f}% | Adaptation (T12): {m_t12:5.1f}% ± {s_t12:4.1f}%", flush=True)

    # Save summary JSON
    results_file = output_dir / "cgp_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_eval_results, f, indent=2)
    print(f"\nSaved benchmark results to {results_file}", flush=True)

    # Print Comparison Markdown Table
    print("\n" + "=" * 85, flush=True)
    print("COMPARATIVE TABLE: RULE A RETENTION (T10) & ADAPTATION (T12) VS DELAY D", flush=True)
    print("=" * 85, flush=True)
    print("| Delay $D$ | CGP Thoughtlet $T_{10}$ | CGP Thoughtlet $T_{12}$ | Baseline Plastic $T_{10}$ | Baseline Plastic $T_{12}$ |", flush=True)
    print("|---:|:---:|:---:|:---:|:---:|", flush=True)
    for D in delays:
        cgp = all_eval_results.get("cgp_thoughtlet", {}).get(str(D), {})
        base = all_eval_results.get("plastic_thoughtlet", {}).get(str(D), {})
        cgp_t10 = f"{cgp.get('t10_mean', 0):.1f}% ± {cgp.get('t10_std', 0):.1f}%"
        cgp_t12 = f"{cgp.get('t12_mean', 0):.1f}% ± {cgp.get('t12_std', 0):.1f}%"
        base_t10 = f"{base.get('t10_mean', 0):.1f}% ± {base.get('t10_std', 0):.1f}%"
        base_t12 = f"{base.get('t12_mean', 0):.1f}% ± {base.get('t12_std', 0):.1f}%"
        print(f"| $D = {D}$ | **{cgp_t10}** | **{cgp_t12}** | {base_t10} | {base_t12} |", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, default="/content/corpus_distractor_v1")
    parser.add_argument("--output_dir", type=str, default="/content/runs_cgp")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 142, 242])
    parser.add_argument("--models", type=str, nargs="+", default=["cgp_thoughtlet", "plastic_thoughtlet"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_samples", type=int, default=24000)
    parser.add_argument("--delays", type=int, nargs="+", default=[0, 10, 25, 50, 100])
    parser.add_argument("--eval_sessions", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_experiment(
        corpus_dir=Path(args.corpus_dir),
        output_dir=Path(args.output_dir),
        seeds=args.seeds,
        models=args.models,
        batch_size=args.batch_size,
        total_samples=args.total_samples,
        delays=args.delays,
        eval_sessions=args.eval_sessions,
        device_str=args.device,
    )
