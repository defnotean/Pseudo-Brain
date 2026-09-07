"""
Plotting and comparative visualization script for Memory Benchmark CGP experiments.
Generates publication-quality figures comparing CGP Thoughtlet, Vanilla Thoughtlet, and GRU baselines.
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def generate_benchmark_plots(
    results_json_path: str = "brain/runs/memory_benchmark_cgp/memory_benchmark_cgp_results.json",
    output_png_path: str = "brain/runs/memory_benchmark_cgp/memory_benchmark_comparison.png",
):
    path = Path(results_json_path)
    if not path.exists():
        print(f"Error: {path} does not exist.")
        return

    with open(path, "r") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    per_seed = data.get("per_seed", {})

    models = ["cgp_thoughtlet", "thoughtlet", "gru"]
    model_labels = ["CGP Thoughtlet\n(Ours)", "Vanilla Thoughtlet\n(Recurrent)", "GRU Baseline\n(High Capacity)"]
    palette = ["#2b5c8f", "#d95f02", "#7570b3"]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plt.subplots_adjust(wspace=0.3)

    # -------------------------------------------------------------
    # Panel 1: Key-to-Door Conversion & Milestone Rates
    # -------------------------------------------------------------
    ax1 = axes[0]
    metrics = ["Key Rate", "Door Rate", "Key -> Door", "Success Rate"]
    x = np.arange(len(metrics))
    width = 0.25

    for i, (m, label, col) in enumerate(zip(models, model_labels, palette)):
        s = summary.get(m, {})
        means = [
            s.get("key_rate_mean", 0.0),
            s.get("door_rate_mean", 0.0),
            s.get("k2d_rate_mean", 0.0),
            s.get("success_rate_mean", 0.0),
        ]
        stds = [
            s.get("key_rate_std", 0.0),
            s.get("door_rate_std", 0.0),
            s.get("k2d_rate_std", 0.0),
            s.get("success_rate_std", 0.0),
        ]
        bars = ax1.bar(
            x + (i - 1) * width,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=label.split("\n")[0],
            color=col,
            alpha=0.88,
            edgecolor="black",
            linewidth=0.8,
        )
        for bar, mean in zip(bars, means):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                max(2.0, mean + 2.0),
                f"{mean:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#222222",
            )

    ax1.set_title("Closed-Loop Milestone Rates (N=25 held-out, 3 seeds)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Rate (%)", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.legend(frameon=True, fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Parameter Efficiency vs Key-to-Door Conversion
    # -------------------------------------------------------------
    ax2 = axes[1]
    for m, label, col in zip(models, model_labels, palette):
        s = summary.get(m, {})
        params = s.get("parameters", 0) / 1000.0  # in thousands
        k2d_mean = s.get("k2d_rate_mean", 0.0)
        k2d_std = s.get("k2d_rate_std", 0.0)

        ax2.errorbar(
            params,
            k2d_mean,
            yerr=k2d_std,
            fmt="o",
            color=col,
            markersize=10,
            capsize=5,
            elinewidth=1.5,
            label=label.split("\n")[0],
        )
        short_label = label.split("\n")[0]
        ax2.annotate(
            f"{short_label}\n({params:.0f}k params: {k2d_mean:.1f}%)",
            xy=(params, k2d_mean),
            xytext=(10, -5 if m == "cgp_thoughtlet" else (10 if m == "thoughtlet" else -20)),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=col),
        )

    ax2.set_xscale("log")
    ax2.set_xlim(40, 3000)
    ax2.set_ylim(30, 90)
    ax2.set_xlabel("Model Parameters (k, log scale)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Key -> Door Conversion Rate (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Parameter Efficiency & Corridor Memory Latching", fontsize=11, fontweight="bold")

    # -------------------------------------------------------------
    # Panel 3: Per-Seed Stability of Corridor Retention
    # -------------------------------------------------------------
    ax3 = axes[2]
    seeds = [42, 142, 242]
    x_seeds = np.arange(len(seeds))

    for i, (m, label, col) in enumerate(zip(models, model_labels, palette)):
        seed_k2d = []
        for s in seeds:
            s_data = per_seed.get(m, {}).get(str(s), {})
            seed_k2d.append(s_data.get("key_to_door_rate", 0.0) * 100.0)

        ax3.plot(
            x_seeds,
            seed_k2d,
            marker="s" if i == 0 else ("^" if i == 1 else "o"),
            linewidth=2.0,
            markersize=8,
            label=label.split("\n")[0],
            color=col,
        )

    ax3.set_xticks(x_seeds)
    ax3.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=10, fontweight="bold")
    ax3.set_ylabel("Key -> Door Retention Rate (%)", fontsize=10, fontweight="bold")
    ax3.set_title("Per-Seed Corridor Memory Stability", fontsize=11, fontweight="bold")
    ax3.set_ylim(20, 100)
    ax3.legend(frameon=True, fontsize=9)

    plt.suptitle("Level 12 POMDP (Keys & Doors): Consequence-Gated Plasticity Benchmark", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_file = Path(output_png_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved successfully to {out_file.resolve()}")

if __name__ == "__main__":
    generate_benchmark_plots()
