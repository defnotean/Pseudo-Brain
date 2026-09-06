"""Analyze GRU vs Thoughtlet results and generate report."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

BRAIN_ROOT = Path(__file__).resolve().parents[2]


def load_training_summary(run_dir: str) -> dict:
    path = Path(run_dir) / "training_summary.json"
    with open(path) as f:
        return json.load(f)


def load_eval_results(run_dir: str) -> dict:
    path = Path(run_dir) / "eval_results.json"
    with open(path) as f:
        return json.load(f)


def load_ablation_results(run_dir: str) -> dict:
    path = Path(run_dir) / "ablations" / "ablation_summary.json"
    with open(path) as f:
        return json.load(f)


def print_training_comparison(summary: dict):
    print("\n" + "=" * 70)
    print("TRAINING RESULTS (Teacher-Forced)")
    print("=" * 70)
    print(f"{'Model':12s} {'Seed':>5s} {'Eval Loss':>10s} {'Eval Acc':>10s} {'Params':>8s}")
    print("-" * 70)
    for key in sorted(summary.keys()):
        v = summary[key]
        model_type, seed_str = key.rsplit("_seed", 1)
        print(f"{model_type:12s} {seed_str:>5s} {v['final_eval_loss']:10.4f} {v['final_eval_acc']:10.3f} {v['param_info']['total']:>8d}")

    # Averages
    print("-" * 70)
    model_types = list(set(k.rsplit("_seed", 1)[0] for k in summary.keys()))
    for mt in sorted(model_types):
        mt_results = {k: v for k, v in summary.items() if k.startswith(mt)}
        if mt_results:
            avg_loss = np.mean([v["final_eval_loss"] for v in mt_results.values()])
            avg_acc = np.mean([v["final_eval_acc"] for v in mt_results.values()])
            avg_params = np.mean([v["param_info"]["total"] for v in mt_results.values()])
            print(f"{mt+' avg':12s} {'':>5s} {avg_loss:10.4f} {avg_acc:10.3f} {avg_params:>8.0f}")


def print_eval_comparison(eval_results: dict):
    print("\n" + "=" * 70)
    print("CLOSED-LOOP EVALUATION (Autonomous)")
    print("=" * 70)
    print(f"{'Model':12s} {'Seed':>5s} {'Pellet':>8s} {'Surv':>6s} {'Latency':>8s}")
    print("-" * 70)

    families = ["F0_calm", "F1_standard", "F2_pressured", "F3_fast", "F4_open_slow"]
    model_types = list(set(k.rsplit("_seed", 1)[0] for k in eval_results.keys() if k != "overall"))

    for mt in sorted(model_types):
        for seed_str in ["42", "142", "242", "342"]:
            key = f"{mt}_seed{seed_str}"
            if key in eval_results:
                r = eval_results[key]
                print(f"{mt:12s} {seed_str:>5s} {r['overall']['mean_pellet_fraction']:8.3f} {r['overall']['mean_survival']:6.2f} {r['overall']['mean_latency_ms']:8.2f}")
        # Per-family breakdown
        print()
        for family in families:
            vals = []
            for seed_str in ["42", "142", "242", "342"]:
                key = f"{mt}_seed{seed_str}"
                if key in eval_results and family in eval_results[key]:
                    vals.append(eval_results[key][family]["pellet_fraction"])
            if vals:
                print(f"  {family:20s} {' ':>5s} {np.mean(vals):8.3f} +/- {np.std(vals):.3f}")
        print()

    # Grand averages
    print("-" * 70)
    for mt in sorted(model_types):
        mt_results = {k: v for k, v in eval_results.items() if k.startswith(mt) and k != "overall"}
        if mt_results:
            avg_pf = np.mean([v["overall"]["mean_pellet_fraction"] for v in mt_results.values()])
            avg_surv = np.mean([v["overall"]["mean_survival"] for v in mt_results.values()])
            avg_lat = np.mean([v["overall"]["mean_latency_ms"] for v in mt_results.values()])
            print(f"{mt+' avg':12s} {'':>5s} {avg_pf:8.3f} {avg_surv:6.2f} {avg_lat:8.2f}")


def print_ablation_results(ablation_results: dict):
    print("\n" + "=" * 70)
    print("ABLATION RESULTS (Thoughtlet variants)")
    print("=" * 70)
    print(f"{'Ablation':25s} {'Params':>8s} {'Pellet Frac':>12s}")
    print("-" * 70)
    for variant in ablation_results:
        r = ablation_results[variant]
        pf = r["closed_loop"]["overall"]["mean_pellet_fraction"]
        print(f"{variant:25s} {r['param_info']['total']:>8d} {pf:12.3f}")


def generate_report(summary, eval_results, ablation_results=None):
    """Print full analysis report."""
    print_training_comparison(summary)
    print_eval_comparison(eval_results)
    if ablation_results:
        print_ablation_results(ablation_results)

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Compare models at matched state
    model_types = list(set(k.rsplit("_seed", 1)[0] for k in eval_results.keys() if k != "overall"))
    averages = {}
    for mt in model_types:
        mt_results = {k: v for k, v in eval_results.items() if k.startswith(mt) and k != "overall"}
        if mt_results:
            averages[mt] = {
                "pellet": np.mean([v["overall"]["mean_pellet_fraction"] for v in mt_results.values()]),
                "survival": np.mean([v["overall"]["mean_survival"] for v in mt_results.values()]),
                "latency": np.mean([v["overall"]["mean_latency_ms"] for v in mt_results.values()]),
            }

    if "gru" in averages and "thoughtlet" in averages:
        g = averages["gru"]
        t = averages["thoughtlet"]
        print(f"\nGRU vs Thoughtlet (384-d matched state):")
        print(f"  Pellet fraction:  GRU={g['pellet']:.3f}  Thoughtlet={t['pellet']:.3f}  (diff={t['pellet']-g['pellet']:+.3f})")
        print(f"  Survival:         GRU={g['survival']:.2f}   Thoughtlet={t['survival']:.2f}")
        print(f"  Latency:          GRU={g['latency']:.2f}ms   Thoughtlet={t['latency']:.2f}ms")

    if ablation_results:
        print(f"\nAblation insights:")
        variants = list(ablation_results.keys())
        if "k32" in ablation_results and "no_persistence" in ablation_results:
            pf_k32 = ablation_results["k32"]["closed_loop"]["overall"]["mean_pellet_fraction"]
            pf_nop = ablation_results["no_persistence"]["closed_loop"]["overall"]["mean_pellet_fraction"]
            print(f"  Persistence:      persistent={pf_k32:.3f}  reset={pf_nop:.3f}  (diff={pf_k32-pf_nop:+.3f})")
        if "k32" in ablation_results and "no_attention" in ablation_results:
            pf_k32 = ablation_results["k32"]["closed_loop"]["overall"]["mean_pellet_fraction"]
            pf_noattn = ablation_results["no_attention"]["closed_loop"]["overall"]["mean_pellet_fraction"]
            print(f"  Attention:        with={pf_k32:.3f}  without={pf_noattn:.3f}  (diff={pf_k32-pf_noattn:+.3f})")
        if "k32" in ablation_results and "independent_slots" in ablation_results:
            pf_k32 = ablation_results["k32"]["closed_loop"]["overall"]["mean_pellet_fraction"]
            pf_ind = ablation_results["independent_slots"]["closed_loop"]["overall"]["mean_pellet_fraction"]
            print(f"  Shared weights:   shared={pf_k32:.3f}  independent={pf_ind:.3f}  (diff={pf_k32-pf_ind:+.3f})")

        # K sweep
        print(f"\n  K sweep (thoughtlet count):")
        for k in [1, 2, 4, 8, 16, 32]:
            variant = f"k{k}"
            if variant in ablation_results:
                pf = ablation_results[variant]["closed_loop"]["overall"]["mean_pellet_fraction"]
                params = ablation_results[variant]["param_info"]["total"]
                print(f"    K={k:2d}  pellet={pf:.3f}  params={params:,d}")


def main():
    run_dir = str(BRAIN_ROOT / "runs" / "gru_vs_thoughtlet")
    summary = load_training_summary(run_dir)
    eval_results = load_eval_results(run_dir)

    ablation_results = None
    ablation_path = Path(run_dir) / "ablations" / "ablation_summary.json"
    if ablation_path.exists():
        ablation_results = load_ablation_results(run_dir)

    generate_report(summary, eval_results, ablation_results)


if __name__ == "__main__":
    main()
