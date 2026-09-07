"""Visualization and Analysis of Dense Batch Phase Transition: B in [16, 24, 32, 48, 64, 96]."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    matplotlib.use("Agg")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def print_ascii_bar(label: str, val: float, max_val: float = 100.0, width: int = 30):
    norm = max(0.0, min(1.0, val / max_val))
    bar_len = int(round(norm * width))
    bar = "█" * bar_len + "░" * (width - bar_len)
    print(f"  {label:18s} | {bar} | {val:5.1f}%")


def analyze_transition_csv(csv_path: Path, output_plot_path: Path):
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if not reader:
        print("No rows found in CSV.")
        return

    print("=" * 90)
    print("DENSE OPTIMIZATION PHASE TRANSITION ANALYSIS (B in [16..96])")
    print("=" * 90)

    # Group by batch_size
    by_b = {}
    for r in reader:
        b = int(r["batch_size"])
        by_b.setdefault(b, []).append(r)

    print(f"{'Batch':6s} | {'Steps':6s} | {'Seed 42':8s} | {'Seed 142':8s} | {'Seed 242':8s} | {'Mean T12 Acc':15s} | {'Eval Surp Var':14s} | {'Gate':8s}")
    print("-" * 90)

    stats = {}
    for b in sorted(by_b.keys()):
        rows = by_b[b]
        steps = rows[0]["steps"]
        seed_map = {int(r["seed"]): float(r["t12_acc"]) for r in rows}
        s42 = f"{seed_map.get(42, 0.0):.0f}%" if 42 in seed_map else "-"
        s142 = f"{seed_map.get(142, 0.0):.0f}%" if 142 in seed_map else "-"
        s242 = f"{seed_map.get(242, 0.0):.0f}%" if 242 in seed_map else "-"
        t12s = [float(r["t12_acc"]) for r in rows]
        m_t12, s_t12 = np.mean(t12s), np.std(t12s)
        esv = np.mean([float(r.get("var_surprise", 0.0)) for r in rows])
        gate = np.mean([float(r.get("mean_gate", 0.0)) for r in rows])
        dp = np.mean([float(r.get("mean_delta_p", 0.0)) for r in rows])
        p_norm = np.mean([float(r.get("mean_p_norm", 0.0)) for r in rows])
        print(f"B={b:<4d} | {steps:<6s} | {s42:<8s} | {s142:<8s} | {s242:<8s} | {m_t12:5.1f}% ± {s_t12:4.1f}% | {esv:<14.4f} | {gate:<8.4f}")
        stats[b] = {
            "m_t12": m_t12, "s_t12": s_t12, "esv": esv, "gate": gate, "dp": dp, "p_norm": p_norm
        }

    print("\nAdaptation Reliability Phase Diagram:")
    for b in sorted(stats.keys()):
        print_ascii_bar(f"B={b:<3d}", stats[b]["m_t12"])

    if HAS_MPL and stats:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
        bs = sorted(stats.keys())
        accs = [stats[b]["m_t12"] for b in bs]
        stds = [stats[b]["s_t12"] for b in bs]
        esvs = [stats[b]["esv"] for b in bs]
        dps = [stats[b]["dp"] for b in bs]

        # 1. Adaptation Accuracy
        axes[0].plot(bs, accs, "o-", color="#1f77b4", linewidth=2.5, markersize=8)
        axes[0].fill_between(bs, np.array(accs) - np.array(stds), np.array(accs) + np.array(stds), color="#1f77b4", alpha=0.2)
        axes[0].set_xlabel("Batch Size (B)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Adaptation Accuracy T12 (%)", fontsize=12, fontweight="bold")
        axes[0].set_title("Optimization Phase Transition: Adaptation Reliability", fontsize=12, fontweight="bold")
        axes[0].set_xticks(bs)
        axes[0].set_ylim(-5, 105)
        axes[0].grid(True, linestyle="--", alpha=0.6)

        # 2. Surprise Variance
        axes[1].plot(bs, esvs, "s-", color="#d62728", linewidth=2.5, markersize=8)
        axes[1].set_xlabel("Batch Size (B)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Surprise Variance Var(e_t)", fontsize=12, fontweight="bold")
        axes[1].set_title("Surprise Variance Collapse vs Batch Size", fontsize=12, fontweight="bold")
        axes[1].set_xticks(bs)
        axes[1].grid(True, linestyle="--", alpha=0.6)

        # 3. Delta P Norm
        axes[2].plot(bs, dps, "^-", color="#2ca02c", linewidth=2.5, markersize=8)
        axes[2].set_xlabel("Batch Size (B)", fontsize=12, fontweight="bold")
        axes[2].set_ylabel("Plasticity Delta Norm ||ΔP_t||", fontsize=12, fontweight="bold")
        axes[2].set_title("Synaptic Update Magnitude vs Batch Size", fontsize=12, fontweight="bold")
        axes[2].set_xticks(bs)
        axes[2].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        fig.savefig(output_plot_path)
        print(f"\n[FIGURE] Saved transition phase curve to {output_plot_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default="transition_phase_curve.png")
    args = parser.parse_args()
    analyze_transition_csv(Path(args.csv), Path(args.out))
