"""Cognitive Scaling Ladder Benchmark (131k -> 500k -> 2M -> 10M).

Systematically benchmarks the empirical scaling trajectories of:
1. CGP Thoughtlets (Pseudo-Brain shared recurrent core with CIG & synaptic latching)
2. Monolithic GRU (parameter-matched multi-layer gated recurrent unit)
3. Modern Diagonal SSM / GLRU (parameter-matched diagonal gated linear recurrent unit)

Across 4 parameter tiers:
- Tier 1: ~131k parameters
- Tier 2: ~500k parameters
- Tier 3: ~2.0M parameters
- Tier 4: ~8.0M parameters

Across 3 critical cognitive stress conditions:
- Condition A: Standard MTLD (M=4 variables, L=16 delay)
- Condition B: Long-Horizon Temporal Stress (M=4 variables, L=128 delay)
- Condition C: Massive Concurrent Load (M=32 variables, L=32 delay)

Evaluates:
- Overall Query Retention Accuracy (%)
- Untouched Variable Retention Accuracy (%)
- Slot Cross-Talk Error Rate (%)
- Inference Latency per Step (ms)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.model.sparse_thought_router import SparseThoughtRouter
from irene_brain.model.torch_model import deterministic_thought_identity_codes
from memory_benchmark.multi_threaded_latent_dependency_benchmark import (
    MultiThreadedLatentDependencyEnv,
    MTLDPhases,
    build_batch,
    BrainCellCore,
    count_params,
)


# ==============================================================================
# 1. Parameter-Matched Model Architectures
# ==============================================================================

class ScalingCGPModel(nn.Module):
    """CGP Thoughtlet model parameterized for exact tier scaling."""

    def __init__(
        self,
        input_dim: int = 64,
        K: int = 4,
        thought_size: int = 12,
        proj_dim: int = 192,
        num_values: int = 8,
        plastic_decay: float = 0.9999,
        plastic_lr: float = 0.25,
    ):
        super().__init__()
        self.K = K
        self.thought_size = thought_size
        self.num_values = num_values
        self.plastic_decay = plastic_decay
        self.plastic_lr = plastic_lr

        self.proj = nn.Linear(input_dim, proj_dim)
        self.brain_cell = BrainCellCore(input_size=proj_dim, thought_size=thought_size)

        self.cig_gate = nn.Sequential(
            nn.Linear(proj_dim + thought_size, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.cig_gate[0].bias, 1.0)

        self.plastic_modulator = nn.Linear(K * thought_size + 16, num_values)
        self.surprise_gate = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.surprise_gate[0].bias, -1.0)
        self.plastic_scale = nn.Parameter(torch.tensor(1.2))

        self.surprise_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 16),
        )

        self.register_buffer(
            "slot_identities",
            deterministic_thought_identity_codes(thoughtlets=K, width=thought_size),
            persistent=False,
        )

        self.head = nn.Sequential(
            nn.Linear(K * thought_size, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, num_values),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_seq.shape
        dev = x_seq.device
        x_proj = self.proj(x_seq)

        # Ensure slot identities match current K dynamically
        if self.slot_identities.shape[0] != self.K:
            slots = deterministic_thought_identity_codes(thoughtlets=self.K, width=self.thought_size).to(dev)
        else:
            slots = self.slot_identities.to(device=dev, dtype=x_seq.dtype)

        thoughts = slots.unsqueeze(0).expand(B, -1, -1).clone()
        P_t = torch.zeros(B, self.num_values, device=dev)
        prev_flat = torch.zeros(B, self.K * self.thought_size, device=dev)
        logits_list = []

        for t in range(T):
            x_in = x_proj[:, t]
            x_exp = x_in.unsqueeze(1).expand(-1, self.K, -1)

            # CIG Gate with sharpened temperature 0.5
            gate_in = torch.cat([x_exp, thoughts], dim=-1)
            raw_gate = self.cig_gate(gate_in)
            salience = torch.sigmoid((torch.logit(raw_gate.clamp(1e-6, 1 - 1e-6))) / 0.5)

            # BrainCell recurrent core
            t_flat = thoughts.reshape(B * self.K, self.thought_size)
            x_flat = x_exp.reshape(B * self.K, -1)
            new_t = self.brain_cell(t_flat, x_flat).reshape(B, self.K, self.thought_size)

            # Disconnected slot update (eliminates inter-slot cross-talk noise)
            thoughts = (1.0 - salience) * thoughts + salience * new_t

            # Surprise & Synaptic Latch Update
            flat_thoughts = thoughts.reshape(B, -1)
            state_delta = torch.norm(flat_thoughts - prev_flat, dim=-1, keepdim=True)
            prev_flat = flat_thoughts.detach()
            surprise_emb = self.surprise_encoder(state_delta)

            gate = self.surprise_gate(surprise_emb)
            delta_P = gate * torch.tanh(self.plastic_modulator(torch.cat([flat_thoughts, surprise_emb], dim=-1)))
            P_t = self.plastic_decay * P_t + self.plastic_lr * delta_P

            # Readout
            base_logits = self.head(flat_thoughts)
            logits_list.append(base_logits + self.plastic_scale * P_t)

        return torch.stack(logits_list, dim=1)


class ScalingGRUModel(nn.Module):
    """Monolithic GRU scaled to parameter budget tiers."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_values: int = 8,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_values),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x_seq)
        return self.head(out)


class ScalingGLRUModel(nn.Module):
    """Modern Gated Linear Recurrent Unit (Diagonal State Space Model)."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_values: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.in_proj = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.gate_proj = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.decay_param = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim) * 0.1 - 2.0)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_values),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_seq.shape
        cur_x = x_seq

        for layer in range(self.num_layers):
            in_p = self.in_proj[layer](cur_x)
            gate_p = torch.sigmoid(self.gate_proj[layer](cur_x))
            alpha = torch.sigmoid(self.decay_param[layer]).unsqueeze(0).unsqueeze(0)  # [1, 1, H]

            # Diagonal recurrence: h_t = alpha * h_{t-1} + (1 - alpha) * (in_p * gate_p)
            h_t = torch.zeros(B, self.hidden_dim, device=x_seq.device)
            h_list = []
            for t in range(T):
                inp_t = in_p[:, t] * gate_p[:, t]
                a_t = alpha[:, 0]
                h_t = a_t * h_t + (1.0 - a_t) * inp_t
                h_list.append(h_t)
            h_seq = torch.stack(h_list, dim=1)
            cur_x = cur_x + self.out_proj[layer](h_seq) if cur_x.shape[-1] == self.hidden_dim else self.out_proj[layer](h_seq)

        return self.head(cur_x)


# ==============================================================================
# 2. Scaling Tier Factory
# ==============================================================================

def create_model_for_tier(
    arch: str,
    tier_name: str,
    input_dim: int = 64,
    num_values: int = 8,
    K: int = 4,
) -> nn.Module:
    """Instantiate model matched to specific parameter tier."""
    # Target parameter tiers:
    # 131k: ~130k
    # 500k: ~500k
    # 2M:   ~2.0M
    # 8M:   ~8.0M

    if arch == "cgp":
        configs = {
            "131k": {"thought_size": 12, "proj_dim": 192},
            "500k": {"thought_size": 24, "proj_dim": 384},
            "2M":   {"thought_size": 48, "proj_dim": 768},
            "8M":   {"thought_size": 96, "proj_dim": 1536},
        }
        cfg = configs[tier_name]
        return ScalingCGPModel(
            input_dim=input_dim,
            K=K,
            thought_size=cfg["thought_size"],
            proj_dim=cfg["proj_dim"],
            num_values=num_values,
            plastic_decay=0.9999,
        )

    elif arch == "gru":
        configs = {
            "131k": {"hidden_dim": 128, "num_layers": 2},
            "500k": {"hidden_dim": 256, "num_layers": 2},
            "2M":   {"hidden_dim": 512, "num_layers": 2},
            "8M":   {"hidden_dim": 1024, "num_layers": 2},
        }
        cfg = configs[tier_name]
        return ScalingGRUModel(
            input_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            num_values=num_values,
        )

    elif arch == "glru":
        configs = {
            "131k": {"hidden_dim": 144, "num_layers": 3},
            "500k": {"hidden_dim": 288, "num_layers": 3},
            "2M":   {"hidden_dim": 576, "num_layers": 3},
            "8M":   {"hidden_dim": 1152, "num_layers": 3},
        }
        cfg = configs[tier_name]
        return ScalingGLRUModel(
            input_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            num_values=num_values,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ==============================================================================
# 3. Training and Evaluation Harness
# ==============================================================================

def train_scaling_model(
    model: nn.Module,
    env: MultiThreadedLatentDependencyEnv,
    num_steps: int = 120,
    batch_size: int = 32,
    lr: float = 2e-3,
    device: torch.device = torch.device("cpu"),
) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for step_idx in range(num_steps):
        obs_batch, targets_batch, _ = build_batch(env, batch_size, seed=3000 + step_idx)
        obs_batch = obs_batch.to(device)
        targets_batch = targets_batch.to(device)

        optimizer.zero_grad()
        logits = model(obs_batch)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), targets_batch.reshape(B * T), ignore_index=-100)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def evaluate_scaling_model(
    model: nn.Module,
    env: MultiThreadedLatentDependencyEnv,
    num_seeds: int = 15,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    model.eval()
    all_accuracies = []
    untouched_accuracies = []
    updated_accuracies = []
    latencies_ms = []

    for s_idx in range(num_seeds):
        obs_batch, targets_batch, episodes = build_batch(env, batch_size=16, seed=8000 + s_idx)
        obs_batch = obs_batch.to(device)

        t0 = time.perf_counter_ns()
        with torch.no_grad():
            logits = model(obs_batch)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / (1e6 * 16))

        preds = logits.argmax(dim=-1).cpu().numpy()
        targets = targets_batch.numpy()

        for b in range(16):
            ep = episodes[b]
            for t, step in enumerate(ep):
                if step.is_query:
                    is_correct = (preds[b, t] == targets[b, t])
                    all_accuracies.append(1.0 if is_correct else 0.0)
                    if step.is_untouched_var:
                        untouched_accuracies.append(1.0 if is_correct else 0.0)
                    elif step.is_updated_var:
                        updated_accuracies.append(1.0 if is_correct else 0.0)

    mean_acc = float(np.mean(all_accuracies)) * 100.0 if all_accuracies else 0.0
    mean_untouched = float(np.mean(untouched_accuracies)) * 100.0 if untouched_accuracies else 0.0
    mean_updated = float(np.mean(updated_accuracies)) * 100.0 if updated_accuracies else 0.0

    return {
        "overall_retention_accuracy": mean_acc,
        "untouched_retention_accuracy": mean_untouched,
        "cross_talk_error_rate": max(0.0, 100.0 - mean_untouched),
        "updated_retention_accuracy": mean_updated,
        "latency_ms_mean": float(np.mean(latencies_ms)),
    }


# ==============================================================================
# 4. Master Scaling Ladder Experiment
# ==============================================================================

def run_scaling_ladder(
    num_seeds: int = 15,
    train_steps: int = 120,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    device = torch.device(device_str)
    tiers = ["131k", "500k", "2M", "8M"]
    architectures = [
        ("cgp", "Pseudo-Brain CGP"),
        ("gru", "Monolithic GRU"),
        ("glru", "Modern Diagonal SSM (GLRU)"),
    ]

    # Cognitive Stress Conditions
    conditions = [
        ("baseline", "Condition A: Baseline (M=4, L=16)", {"num_variables": 4, "delay_1": 16, "delay_2": 16}),
        ("long_delay", "Condition B: Temporal Stress (M=4, L=128)", {"num_variables": 4, "delay_1": 128, "delay_2": 128}),
        ("massive_load", "Condition C: Massive Load (M=32, L=32)", {"num_variables": 32, "delay_1": 32, "delay_2": 32}),
    ]

    results: Dict[str, Any] = {
        "tiers": tiers,
        "architectures": [a[0] for a in architectures],
        "grid": {},
    }

    for t_name in tiers:
        results["grid"][t_name] = {}
        print("\n" + "=" * 95)
        print(f"PARAMETER TIER: {t_name}")
        print("=" * 95)

        for arch_key, arch_label in architectures:
            results["grid"][t_name][arch_key] = {}
            print(f"\n--- Architecture: {arch_label} ({t_name}) ---")

            # Condition-specific training & evaluation
            for cond_key, cond_label, cond_params in conditions:
                K_slots = cond_params["num_variables"]
                model = create_model_for_tier(arch_key, t_name, K=K_slots).to(device)
                p_count = count_params(model)["total"]

                # Train on baseline env of that M
                train_env = MultiThreadedLatentDependencyEnv(
                    num_variables=K_slots,
                    delay_1=16,
                    delay_2=16,
                )
                train_scaling_model(model, train_env, num_steps=train_steps, device=device)

                eval_env = MultiThreadedLatentDependencyEnv(**cond_params)
                metrics = evaluate_scaling_model(model, eval_env, num_seeds=num_seeds, device=device)
                metrics["params"] = p_count
                metrics["arch_label"] = arch_label
                results["grid"][t_name][arch_key][cond_key] = metrics

                print(f"  {cond_label:<45s} | Params: {p_count/1e3:6.1f}k | Ret: {metrics['overall_retention_accuracy']:5.1f}% | Untouched: {metrics['untouched_retention_accuracy']:5.1f}% | Lat: {metrics['latency_ms_mean']:.2f} ms")

    return results


# ==============================================================================
# 5. Report Generation
# ==============================================================================

def generate_scaling_report(results: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "2026-09-07-cognitive-scaling-ladder.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw scaling telemetry to {json_path}")

    md_path = output_dir / "2026-09-07-cognitive-scaling-ladder.md"
    lines = [
        "# Cognitive Scaling Ladder Benchmark Report (131k -> 500k -> 2M -> 8M)",
        "**Date:** 2026-09-07  ",
        "**Status:** `[MEASURED]` Multi-scale empirical comparison between Pseudo-Brain CGP, Monolithic GRU, and Diagonal SSM (GLRU).  \n",
        "## 1. Executive Summary",
        "This experiment tracks whether the architectural advantage of Pseudo-Brain's shared recurrent core, CIG gating, and synaptic latching ($P_t$) scales monotonically across parameter tiers (131k to 8M) and stress regimes:",
        "- **Condition A**: Baseline MTLD ($M=4, L=16$)",
        "- **Condition B**: Temporal Delay Stress ($M=4, L=128$)",
        "- **Condition C**: Massive Concurrent Load ($M=32, L=32$)\n",
        "## 2. Scaling Trajectories across Parameter Tiers\n",
    ]

    for cond_key, cond_title in [
        ("baseline", "Condition A: Baseline (M=4, L=16)"),
        ("long_delay", "Condition B: Temporal Stress (M=4, L=128)"),
        ("massive_load", "Condition C: Massive Load (M=32, L=32)"),
    ]:
        lines.extend([
            f"### {cond_title}",
            "| Tier | CGP Accuracy | GRU Accuracy | GLRU (SSM) Accuracy | CGP Margin vs GRU | CGP Latency |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for t_name in results["tiers"]:
            cgp_m = results["grid"][t_name]["cgp"][cond_key]
            gru_m = results["grid"][t_name]["gru"][cond_key]
            glru_m = results["grid"][t_name]["glru"][cond_key]

            cgp_acc = cgp_m["overall_retention_accuracy"]
            gru_acc = gru_m["overall_retention_accuracy"]
            glru_acc = glru_m["overall_retention_accuracy"]
            margin = cgp_acc - gru_acc
            lat = cgp_m["latency_ms_mean"]

            lines.append(
                f"| **{t_name}** | **{cgp_acc:.1f}%** | {gru_acc:.1f}% | {glru_acc:.1f}% | **+{margin:.1f}%** | {lat:.2f} ms |"
            )
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Generated Markdown report at {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cognitive Scaling Ladder Benchmark")
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="brain/docs/runs")
    args = parser.parse_args()

    results = run_scaling_ladder(
        num_seeds=args.num_seeds,
        train_steps=args.train_steps,
        device_str=args.device,
    )
    generate_scaling_report(results, output_dir=Path(args.output_dir))
