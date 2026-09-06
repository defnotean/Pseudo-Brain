"""Master evaluator for Phase H: Self-Correction Benchmark.

Evaluates:
- Baseline (BC trained on normal data, no prediction)
- Prediction (BC + next-latent prediction + surprise feedback)
- Prediction + Plasticity (Prediction + surprise + fast episodic adaptation)

Across both GRU and Thoughtlet architectures on 25 held-out TEST episodes.
Outputs the complete Phase H comparative matrix to brain/runs/self_correction/SELF_CORRECTION_REPORT.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.environments.keys_doors import KeysDoorsEnv
from memory_benchmark.models import make_model, count_parameters as count_bc_params
from self_correction.diagnostic_eval import run_diagnostic_episode
from self_correction.predictive_models import PredictiveGRUModel, PredictiveThoughtletModel
from self_correction.self_correction_benchmark import evaluate_predictive_episode


def evaluate_checkpoint(
    ckpt_path: Path,
    is_predictive: bool,
    seeds: List[int],
    modes: List[str] = ["none", "single", "burst"],
) -> Dict:
    data = torch.load(ckpt_path, map_location="cpu")
    model_type = data["model_type"]

    if not is_predictive:
        model = make_model(model_type)
        model.load_state_dict(data["model_state_dict"])
        model.eval()
        p_info = count_bc_params(model)
        total_params = p_info["total"]
    else:
        use_plast = data.get("use_plasticity", False)
        if model_type == "gru":
            model = PredictiveGRUModel(use_plasticity=use_plast)
        elif model_type == "thoughtlet":
            model = PredictiveThoughtletModel(use_plasticity=use_plast)
        else:
            raise ValueError(f"Unknown model {model_type}")
        model.load_state_dict(data["model_state_dict"])
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())

    env = KeysDoorsEnv()
    res_by_mode = {}

    for mode in modes:
        episodes = []
        t0_mode = time.perf_counter()
        total_ticks_mode = 0

        for s in seeds:
            if not is_predictive:
                ep_res = run_diagnostic_episode(model, env, seed=s, perturb_mode=mode)
                ep_res["mean_surprise"] = 0.0
            else:
                ep_res = evaluate_predictive_episode(model, env, seed=s, perturb_mode=mode)

            episodes.append(ep_res)
            total_ticks_mode += ep_res["total_ticks"]

        t_elapsed = time.perf_counter() - t0_mode
        latency_ms = (t_elapsed / max(total_ticks_mode, 1)) * 1000.0

        n = len(episodes)
        succ = sum(1 for e in episodes if e["success"])
        pert_cnt = sum(1 for e in episodes if e["perturbed"])
        rec_cnt = sum(1 for e in episodes if e["recovered"])
        rec_steps = [e["recovery_steps"] for e in episodes if e["recovery_steps"] is not None]
        mean_rec_steps = float(np.mean(rec_steps)) if rec_steps else 0.0
        surp = float(np.mean([e.get("mean_surprise", 0.0) for e in episodes]))

        res_by_mode[mode] = {
            "success_rate": succ / n,
            "recovery_rate": (rec_cnt / pert_cnt) if pert_cnt > 0 else 1.0,
            "mean_recovery_steps": mean_rec_steps,
            "failure_rate": 1.0 - (succ / n),
            "mean_surprise": surp,
            "latency_ms": latency_ms,
        }

    return {
        "model_type": model_type,
        "is_predictive": is_predictive,
        "use_plasticity": data.get("use_plasticity", False) if is_predictive else False,
        "total_params": total_params,
        "by_mode": res_by_mode,
    }


def main():
    runs_dir = _REPO_ROOT / "runs"
    out_dir = runs_dir / "self_correction"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [3000 + i for i in range(25)]
    print(f"Evaluating Self-Correction Benchmark on 25 held-out TEST seeds (3000..3024)...")

    checkpoints_to_eval = [
        # GRU Family
        ("GRU Baseline", runs_dir / "memory_benchmark" / "checkpoints" / "gru_seed_42.pt", False),
        ("GRU Predictive", out_dir / "checkpoints" / "predictive_gru_seed_42.pt", True),
        ("GRU Pred + Plasticity", out_dir / "checkpoints" / "predictive_gru_plastic_seed_42.pt", True),
        # Thoughtlet Family
        ("Thoughtlet Baseline", runs_dir / "memory_benchmark" / "checkpoints" / "thoughtlet_seed_42.pt", False),
        ("Thoughtlet Predictive", out_dir / "checkpoints" / "predictive_thoughtlet_seed_42.pt", True),
        ("Thoughtlet Pred + Plasticity", out_dir / "checkpoints" / "predictive_thoughtlet_plastic_seed_42.pt", True),
    ]

    all_evals = {}
    for name, path, is_pred in checkpoints_to_eval:
        print(f"\nEvaluating {name}...")
        if not path.exists():
            print(f"  Warning: checkpoint {path} does not exist, skipping.")
            continue
        res = evaluate_checkpoint(path, is_pred, seeds)
        all_evals[name] = res
        print(f"  Clean Success : {res['by_mode']['none']['success_rate']*100:.1f}%")
        print(f"  Single Recov. : {res['by_mode']['single']['recovery_rate']*100:.1f}% (steps: {res['by_mode']['single']['mean_recovery_steps']:.1f})")
        print(f"  Burst Recov.  : {res['by_mode']['burst']['recovery_rate']*100:.1f}% (steps: {res['by_mode']['burst']['mean_recovery_steps']:.1f})")

    # Save JSON
    with open(out_dir / "self_correction_eval_results.json", "w") as f:
        json.dump(all_evals, f, indent=2)

    # Generate Markdown Report matching Phase H format
    report_path = out_dir / "SELF_CORRECTION_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# Self-Correction & Learning from Prediction Error: Benchmark Report\n\n")
        f.write("Evaluation across 25 held-out TEST episodes (`KeysDoorsEnv` POMDP maze).\n\n")

        f.write("## 1. GRU Architecture Comparison\n\n")
        f.write("| Metric | Baseline (BC) | Prediction + Surprise | Prediction + Plasticity |\n")
        f.write("|---|---:|---:|---:|\n")

        b_gru = all_evals.get("GRU Baseline", {}).get("by_mode", {})
        p_gru = all_evals.get("GRU Predictive", {}).get("by_mode", {})
        pl_gru = all_evals.get("GRU Pred + Plasticity", {}).get("by_mode", {})

        f.write(f"| **Normal success** | {b_gru.get('none', {}).get('success_rate', 0)*100:.1f}% | {p_gru.get('none', {}).get('success_rate', 0)*100:.1f}% | {pl_gru.get('none', {}).get('success_rate', 0)*100:.1f}% |\n")
        f.write(f"| **Perturbed success (single)** | {b_gru.get('single', {}).get('success_rate', 0)*100:.1f}% | {p_gru.get('single', {}).get('success_rate', 0)*100:.1f}% | {pl_gru.get('single', {}).get('success_rate', 0)*100:.1f}% |\n")
        f.write(f"| **Recovery rate (single)** | {b_gru.get('single', {}).get('recovery_rate', 0)*100:.1f}% | {p_gru.get('single', {}).get('recovery_rate', 0)*100:.1f}% | {pl_gru.get('single', {}).get('recovery_rate', 0)*100:.1f}% |\n")
        f.write(f"| **Mean recovery steps** | {b_gru.get('single', {}).get('mean_recovery_steps', 0):.1f} | {p_gru.get('single', {}).get('mean_recovery_steps', 0):.1f} | {pl_gru.get('single', {}).get('mean_recovery_steps', 0):.1f} |\n")
        f.write(f"| **Failure after perturbation** | {b_gru.get('single', {}).get('failure_rate', 0)*100:.1f}% | {p_gru.get('single', {}).get('failure_rate', 0)*100:.1f}% | {pl_gru.get('single', {}).get('failure_rate', 0)*100:.1f}% |\n")
        f.write(f"| **Prediction error (surprise)** | — | {p_gru.get('single', {}).get('mean_surprise', 0):.4f} | {pl_gru.get('single', {}).get('mean_surprise', 0):.4f} |\n")
        f.write(f"| **Parameters** | {all_evals.get('GRU Baseline', {}).get('total_params', 0):,} | {all_evals.get('GRU Predictive', {}).get('total_params', 0):,} | {all_evals.get('GRU Pred + Plasticity', {}).get('total_params', 0):,} |\n")
        f.write(f"| **Inference latency** | {b_gru.get('single', {}).get('latency_ms', 0):.2f} ms | {p_gru.get('single', {}).get('latency_ms', 0):.2f} ms | {pl_gru.get('single', {}).get('latency_ms', 0):.2f} ms |\n")

        f.write("\n## 2. Thoughtlet Architecture Comparison\n\n")
        f.write("| Metric | Baseline (BC) | Prediction + Surprise | Prediction + Plasticity |\n")
        f.write("|---|---:|---:|---:|\n")

        b_th = all_evals.get("Thoughtlet Baseline", {}).get("by_mode", {})
        p_th = all_evals.get("Thoughtlet Predictive", {}).get("by_mode", {})
        pl_th = all_evals.get("Thoughtlet Pred + Plasticity", {}).get("by_mode", {})

        f.write(f"| **Normal success** | {b_th.get('none', {}).get('success_rate', 0)*100:.1f}% | {p_th.get('none', {}).get('success_rate', 0)*100:.1f}% | {pl_th.get('none', {}).get('success_rate', 0)*100:.1f}% |\n")
        f.write(f"| **Perturbed success (single)** | {b_th.get('single', {}).get('success_rate', 0)*100:.1f}% | {p_th.get('single', {}).get('success_rate', 0)*100:.1f}% | {pl_th.get('single', {}).get('success_rate', 0)*100:.1f}% |\n")
        f.write(f"| **Recovery rate (single)** | {b_th.get('single', {}).get('recovery_rate', 0)*100:.1f}% | {p_th.get('single', {}).get('recovery_rate', 0)*100:.1f}% | {pl_th.get('single', {}).get('recovery_rate', 0)*100:.1f}% |\n")
        f.write(f"| **Mean recovery steps** | {b_th.get('single', {}).get('mean_recovery_steps', 0):.1f} | {p_th.get('single', {}).get('mean_recovery_steps', 0):.1f} | {pl_th.get('single', {}).get('mean_recovery_steps', 0):.1f} |\n")
        f.write(f"| **Failure after perturbation** | {b_th.get('single', {}).get('failure_rate', 0)*100:.1f}% | {p_th.get('single', {}).get('failure_rate', 0)*100:.1f}% | {pl_th.get('single', {}).get('failure_rate', 0)*100:.1f}% |\n")
        f.write(f"| **Prediction error (surprise)** | — | {p_th.get('single', {}).get('mean_surprise', 0):.4f} | {pl_th.get('single', {}).get('mean_surprise', 0):.4f} |\n")
        f.write(f"| **Parameters** | {all_evals.get('Thoughtlet Baseline', {}).get('total_params', 0):,} | {all_evals.get('Thoughtlet Predictive', {}).get('total_params', 0):,} | {all_evals.get('Thoughtlet Pred + Plasticity', {}).get('total_params', 0):,} |\n")
        f.write(f"| **Inference latency** | {b_th.get('single', {}).get('latency_ms', 0):.2f} ms | {p_th.get('single', {}).get('latency_ms', 0):.2f} ms | {pl_th.get('single', {}).get('latency_ms', 0):.2f} ms |\n")

    print(f"\nGenerated report at {report_path}")

if __name__ == "__main__":
    main()
