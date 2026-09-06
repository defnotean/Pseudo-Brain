"""Visualization and Analysis script for Optimization Variable Isolation & Plasticity Scaling Curve.

Generates:
1. ASCII tables and plots for stdout
2. Matplotlib PNG figure: `plasticity_scaling_curve.png`
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_results_csv(csv_path: Path) -> List[Dict]:
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def print_ascii_bar(label: str, val: float, max_val: float = 100.0, width: int = 30):
    norm = max(0.0, min(1.0, val / max_val))
    bar_len = int(round(norm * width))
    bar = "█" * bar_len + "░" * (width - bar_len)
    print(f"  {label:18s} | {bar} | {val:5.1f}%")


def analyze_and_plot(csv_path: Path, output_plot_path: Path):
    rows = load_results_csv(csv_path)
    if not rows:
        print(f"No results found in {csv_path}")
        return

    print("=" * 80)
    print("PLASTICITY OPTIMIZATION & SCALING ANALYSIS")
    print(f"Source: {csv_path.resolve()}")
    print("=" * 80)

    # 1. Phase 1: 2x2 Matrix Analysis
    phase1_rows = [r for r in rows if r.get("cell")]
    if phase1_rows:
        print("\n--- PHASE 1: 2x2 CONTROLLED MATRIX ---")
        print(f"{'Cell / Config':32s} | {'T11 (Reversal)':16s} | {'T12 (Adaptation)':16s} | {'Surp Var':10s}")
        print("-" * 80)
        cells = sorted(list(set(r["cell"] for r in phase1_rows)))
        for c in cells:
            c_rows = [r for r in phase1_rows if r["cell"] == c]
            t11s = [float(r["t11_acc"]) for r in c_rows]
            t12s = [float(r["t12_acc"]) for r in c_rows]
            surp_vars = [float(r.get("var_surprise", 0.0)) for r in c_rows]
            m11, s11 = np.mean(t11s), np.std(t11s)
            m12, s12 = np.mean(t12s), np.std(t12s)
            m_sv = np.mean(surp_vars)
            print(f"{c:32s} | {m11:5.1f}% ± {s11:4.1f}%   | {m12:5.1f}% ± {s12:4.1f}%   | {m_sv:8.4f}")

    # 2. Phase 2: Gradient Accumulation Analysis
    phase2_rows = [r for r in rows if r.get("accum_config")]
    if phase2_rows:
        print("\n--- PHASE 2: GRADIENT ACCUMULATION VS PHYSICAL BATCH ---")
        print(f"{'Config':32s} | {'Phys B':6s} | {'Eff B':6s} | {'T12 (Adapt)':16s} | {'Surp Var':10s}")
        print("-" * 80)
        cfgs = sorted(list(set(r["accum_config"] for r in phase2_rows)))
        for cfg in cfgs:
            c_rows = [r for r in phase2_rows if r["accum_config"] == cfg]
            b_phys = c_rows[0].get("batch_size", "")
            b_eff = c_rows[0].get("effective_batch_size", "")
            t12s = [float(r["t12_acc"]) for r in c_rows]
            surp_vars = [float(r.get("var_surprise", 0.0)) for r in c_rows]
            m12, s12 = np.mean(t12s), np.std(t12s)
            m_sv = np.mean(surp_vars)
            print(f"{cfg:32s} | {b_phys:6s} | {b_eff:6s} | {m12:5.1f}% ± {s12:4.1f}%   | {m_sv:8.4f}")

    # 3. Phase 3: Scaling Curve Analysis
    # Filter rows that have batch_size and are part of scaling
    b_ladder = [4, 8, 16, 32, 64, 128]
    scaling_data = {}
    for b in b_ladder:
        b_rows = [r for r in rows if int(r.get("batch_size", 0)) == b and int(r.get("grad_accum_steps", 1)) == 1 and not r.get("cell")]
        if not b_rows:
            # Fallback: check all rows with this batch_size
            b_rows = [r for r in rows if int(r.get("batch_size", 0)) == b and int(r.get("grad_accum_steps", 1)) == 1]
        if b_rows:
            t12s = [float(r["t12_acc"]) for r in b_rows]
            t10s = [float(r["t10_acc"]) for r in b_rows]
            t20s = [float(r["t20_acc"]) for r in b_rows]
            surp_vars = [float(r.get("var_surprise", 0.0)) for r in b_rows]
            surp_means = [float(r.get("mean_surprise", 0.0)) for r in b_rows]
            gates = [float(r.get("mean_gate", 0.0)) for r in b_rows]
            grad_norms = [float(r.get("mean_grad_norm", 0.0)) for r in b_rows]
            scaling_data[b] = {
                "t12_mean": float(np.mean(t12s)),
                "t12_std": float(np.std(t12s)),
                "t10_mean": float(np.mean(t10s)),
                "t20_mean": float(np.mean(t20s)),
                "surp_var": float(np.mean(surp_vars)),
                "surp_mean": float(np.mean(surp_means)),
                "gate_mean": float(np.mean(gates)),
                "grad_norm": float(np.mean(grad_norms)),
                "n_runs": len(b_rows),
            }

    if scaling_data:
        print("\n--- PHASE 3: PLASTICITY SCALING CURVE (B in [4..128]) ---")
        print("Adaptation Accuracy (T12) vs Batch Size:")
        for b, st in sorted(scaling_data.items()):
            print_ascii_bar(f"B={b:3d} (n={st['n_runs']})", st["t12_mean"])

        print("\nSurprise Variance vs Batch Size:")
        for b, st in sorted(scaling_data.items()):
            print(f"  B={b:3d} | Var(e_t): {st['surp_var']:.5f} | Gate: {st['gate_mean']:.4f} | Grad Norm: {st['grad_norm']:.4f}")

    # Generate Matplotlib Figure if available
    if HAS_MPL and scaling_data:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
        
        # Subplot 1: Batch Size vs Adaptation Accuracy
        bs = sorted(scaling_data.keys())
        accs = [scaling_data[b]["t12_mean"] for b in bs]
        stds = [scaling_data[b]["t12_std"] for b in bs]
        axes[0].plot(bs, accs, "o-", color="#1f77b4", linewidth=2.5, markersize=8, label="Plastic Thoughtlet (T12 Adaptation)")
        axes[0].fill_between(bs, np.array(accs) - np.array(stds), np.array(accs) + np.array(stds), color="#1f77b4", alpha=0.2)
        axes[0].set_xscale("log", base=2)
        axes[0].set_xticks(bs)
        axes[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axes[0].set_xlabel("Batch Size (B)", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Adaptation Accuracy (%)", fontsize=12, fontweight="bold")
        axes[0].set_title("Plasticity Scaling Curve: Adaptation vs Batch Size", fontsize=13, fontweight="bold")
        axes[0].set_ylim(-5, 105)
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend(loc="lower left", fontsize=10)

        # Subplot 2: Batch Size vs Surprise Variance
        svars = [scaling_data[b]["surp_var"] for b in bs]
        axes[1].plot(bs, svars, "s-", color="#d62728", linewidth=2.5, markersize=8, label="Surprise Variance Var(e_t)")
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks(bs)
        axes[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axes[1].set_xlabel("Batch Size (B)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Surprise Variance", fontsize=12, fontweight="bold")
        axes[1].set_title("Surprise Dynamics: Error Variance vs Batch Size", fontsize=13, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend(loc="upper right", fontsize=10)

        plt.tight_layout()
        fig.savefig(output_plot_path)
        print(f"\n[FIGURE] Saved scaling curve figure to: {output_plot_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to matrix_results.csv")
    parser.add_argument("--out", type=str, default="plasticity_scaling_curve.png", help="Path to save figure")
    args = parser.parse_args()
    analyze_and_plot(Path(args.csv), Path(args.out))
