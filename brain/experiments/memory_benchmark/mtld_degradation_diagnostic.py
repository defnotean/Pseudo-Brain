"""Multi-Threaded Latent Dependency Degradation Causal Diagnostic Suite.

Systematically investigates the 4 causal hypotheses behind the delay degradation failure curve
(where retention drops from 38.3% to 19.1% across L=16 -> 128) and the Disconnected vs. Routed crossover:

1. Hypothesis 1 (Passive Latch Decay):
   Sweep plastic_decay lambda in [0.99, 0.999, 0.9999, 1.000].
   Does setting lambda = 1.0 (zero passive decay) halt corridor retention erosion?

2. Hypothesis 2 (CIG Gate Bleed / Noise Accumulation):
   Audit mean gate salience g_bar_delay on noise tokens.
   Does gate sharpening (temperature scaling) eliminate slow memory overwrite?

3. Hypothesis 3 (Slot Width Capacity):
   Sweep width W in [12, 24, 48, 64].
   Does higher-dimensional hyperspherical volume prevent subspace collapse?

4. Hypothesis 4 (Routing Modality & The Long-Horizon Crossover):
   Compare Disconnected (A=0), Static Top-4, Dense All-to-All, and
   Dependency-Gated Dynamic Routing (A_ij = Softmax(QK^T) if energy > tau else 0).
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
# 1. Diagnostic CGP Model with Modular Routing, Width, and Latch Controls
# ==============================================================================

class DiagnosticCGPModel(nn.Module):
    """Configurable CGP Thoughtlet model for causal failure mode decomposition."""

    def __init__(
        self,
        input_dim: int = 64,
        K: int = 32,
        thought_size: int = 12,
        num_values: int = 8,
        plastic_decay: float = 0.999,
        plastic_lr: float = 0.25,
        routing_mode: str = "sparse_topk",  # 'disconnected', 'sparse_topk', 'dense', 'dependency_gated'
        routed_neighbors: int = 4,
        gate_temperature: float = 1.0,
        dependency_threshold: float = 0.5,
    ):
        super().__init__()
        self.K = K
        self.thought_size = thought_size
        self.num_values = num_values
        self.plastic_decay = plastic_decay
        self.plastic_lr = plastic_lr
        self.routing_mode = routing_mode
        self.routed_neighbors = routed_neighbors
        self.gate_temperature = gate_temperature
        self.dependency_threshold = dependency_threshold

        proj_dim = max(128, min(512, 16 * thought_size))
        self.proj = nn.Linear(input_dim, proj_dim)
        self.brain_cell = BrainCellCore(input_size=proj_dim, thought_size=thought_size)

        # Routers
        if routing_mode == "sparse_topk":
            self.router = SparseThoughtRouter(
                width=thought_size,
                routed_neighbors=routed_neighbors,
                dense_routing=False,
            )
        elif routing_mode == "dense":
            self.router = nn.MultiheadAttention(embed_dim=thought_size, num_heads=max(1, thought_size // 6), batch_first=True)
            self.attn_proj = nn.Linear(thought_size, thought_size)
        elif routing_mode == "dependency_gated":
            # Gated router with query/key projections and thresholded energy
            self.q_proj = nn.Linear(thought_size, thought_size, bias=False)
            self.k_proj = nn.Linear(thought_size, thought_size, bias=False)
            self.v_proj = nn.Linear(thought_size, thought_size, bias=False)
            self.out_proj = nn.Linear(thought_size, thought_size)

        # Cognitive Input Gate (CIG)
        self.cig_gate = nn.Sequential(
            nn.Linear(proj_dim + thought_size, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.cig_gate[0].bias, 1.0)

        # Synaptic Latching (CGSL)
        self.plastic_modulator = nn.Linear(K * thought_size + 16, num_values)
        self.surprise_gate = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.surprise_gate[0].bias, -1.0)
        self.plastic_scale = nn.Parameter(torch.tensor(1.2))

        # Surprise Encoder
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

    def forward(
        self,
        x_seq: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T, _ = x_seq.shape
        dev = x_seq.device
        x_proj = self.proj(x_seq)

        thoughts = self.slot_identities.to(device=dev, dtype=x_seq.dtype).unsqueeze(0).expand(B, -1, -1).clone()
        P_t = torch.zeros(B, self.num_values, device=dev)
        prev_flat = torch.zeros(B, self.K * self.thought_size, device=dev)
        logits_list = []

        delay_salience_accum = []
        routing_energy_accum = []

        for t in range(T):
            x_in = x_proj[:, t]
            x_exp = x_in.unsqueeze(1).expand(-1, self.K, -1)

            # CIG Gate with optional temperature scaling
            gate_in = torch.cat([x_exp, thoughts], dim=-1)
            raw_gate = self.cig_gate(gate_in)
            if self.gate_temperature != 1.0:
                salience = torch.sigmoid((torch.logit(raw_gate.clamp(1e-6, 1 - 1e-6))) / self.gate_temperature)
            else:
                salience = raw_gate

            # BrainCell update
            t_flat = thoughts.reshape(B * self.K, self.thought_size)
            x_flat = x_exp.reshape(B * self.K, -1)
            new_t = self.brain_cell(t_flat, x_flat).reshape(B, self.K, self.thought_size)

            # Routing Mechanics
            if self.routing_mode == "disconnected":
                t_candidate = new_t

            elif self.routing_mode == "sparse_topk":
                messages, _ = self.router(new_t)
                t_candidate = new_t + messages

            elif self.routing_mode == "dense":
                attn_out, _ = self.router(new_t, new_t, new_t)
                t_candidate = new_t + self.attn_proj(attn_out)

            elif self.routing_mode == "dependency_gated":
                # Compute QK^T energy
                Q = self.q_proj(new_t)
                K_mat = self.k_proj(new_t)
                V_mat = self.v_proj(new_t)

                # Scaled dot-product: [B, K, K]
                scale = 1.0 / math.sqrt(self.thought_size)
                scores = torch.bmm(Q, K_mat.transpose(1, 2)) * scale

                # Self-exclusion: diagonal = -inf
                eye_mask = torch.eye(self.K, device=dev, dtype=torch.bool).unsqueeze(0)
                scores = scores.masked_fill(eye_mask, -1e9)

                # Measure maximum mutual cross-affinity per slot
                max_energy = scores.max(dim=-1, keepdim=True).values  # [B, K, 1]
                routing_active = (torch.sigmoid(max_energy) > self.dependency_threshold).float()
                routing_energy_accum.append(float(torch.sigmoid(max_energy).mean()))

                attn_weights = F.softmax(scores, dim=-1)
                routed_context = torch.bmm(attn_weights, V_mat)
                gated_messages = routing_active * self.out_proj(routed_context)
                t_candidate = new_t + gated_messages

            # Gated state update
            thoughts = (1.0 - salience) * thoughts + salience * t_candidate

            # Telemetry for gate salience
            delay_salience_accum.append(float(salience.mean()))

            # Surprise / Consequence estimation
            flat_thoughts = thoughts.reshape(B, -1)
            state_delta = torch.norm(flat_thoughts - prev_flat, dim=-1, keepdim=True)
            prev_flat = flat_thoughts.detach()
            surprise_emb = self.surprise_encoder(state_delta)

            # Plastic Latch Update: P_t
            gate = self.surprise_gate(surprise_emb)
            delta_P = gate * torch.tanh(self.plastic_modulator(torch.cat([flat_thoughts, surprise_emb], dim=-1)))
            P_t = self.plastic_decay * P_t + self.plastic_lr * delta_P

            # Readout Head
            base_logits = self.head(flat_thoughts)
            logits_list.append(base_logits + self.plastic_scale * P_t)

        logits = torch.stack(logits_list, dim=1)
        diagnostics = {
            "mean_salience": float(np.mean(delay_salience_accum)) if delay_salience_accum else 0.0,
            "mean_routing_energy": float(np.mean(routing_energy_accum)) if routing_energy_accum else 0.0,
        }
        return logits, diagnostics


# ==============================================================================
# 2. Training and Evaluation Harness
# ==============================================================================

def train_diagnostic_model(
    model: DiagnosticCGPModel,
    env: MultiThreadedLatentDependencyEnv,
    num_steps: int = 120,
    batch_size: int = 32,
    lr: float = 2e-3,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Train diagnostic model on MTLD query tokens."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for step_idx in range(num_steps):
        obs_batch, targets_batch, _ = build_batch(env, batch_size, seed=2000 + step_idx)
        obs_batch = obs_batch.to(device)
        targets_batch = targets_batch.to(device)

        optimizer.zero_grad()
        logits, _ = model(obs_batch)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), targets_batch.reshape(B * T), ignore_index=-100)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def evaluate_diagnostic_model(
    model: DiagnosticCGPModel,
    env: MultiThreadedLatentDependencyEnv,
    num_seeds: int = 15,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Evaluate diagnostic model across seeds and collect telemetry."""
    model.eval()
    all_accuracies = []
    untouched_accuracies = []
    updated_accuracies = []
    latencies_ms = []
    salience_records = []

    for s_idx in range(num_seeds):
        obs_batch, targets_batch, episodes = build_batch(env, batch_size=16, seed=7000 + s_idx)
        obs_batch = obs_batch.to(device)

        t0 = time.perf_counter_ns()
        with torch.no_grad():
            logits, diag = model(obs_batch, return_diagnostics=True)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / (1e6 * 16))
        salience_records.append(diag["mean_salience"])

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
        "mean_salience": float(np.mean(salience_records)),
    }


# ==============================================================================
# 3. Master Diagnostic Experiment Suite
# ==============================================================================

def run_degradation_diagnostics(
    num_seeds: int = 15,
    train_steps: int = 120,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Execute all 4 degradation diagnostic experiments."""
    device = torch.device(device_str)
    canonical_m = 4
    delay_horizons = [16, 32, 64, 128]

    results: Dict[str, Any] = {
        "experiment_1_latch_decay": {},
        "experiment_2_gate_bleed": {},
        "experiment_3_slot_width": {},
        "experiment_4_routing_crossover": {},
    }

    train_env = MultiThreadedLatentDependencyEnv(num_variables=canonical_m, delay_1=16, delay_2=16)

    # --------------------------------------------------------------------------
    # Experiment 1: Latch Persistence Sweep (lambda in [0.99, 0.999, 0.9999, 1.0])
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("EXPERIMENT 1: Passive Synaptic Latch Decay Sweep (Testing Lambda in [0.99, 0.999, 0.9999, 1.0])")
    print("=" * 90)
    decay_values = [0.99, 0.999, 0.9999, 1.0]

    for d_val in decay_values:
        key = f"lambda_{d_val}"
        results["experiment_1_latch_decay"][key] = {}
        model = DiagnosticCGPModel(
            thought_size=12,
            plastic_decay=d_val,
            routing_mode="sparse_topk",
        ).to(device)
        train_diagnostic_model(model, train_env, num_steps=train_steps, device=device)

        for L in delay_horizons:
            eval_env = MultiThreadedLatentDependencyEnv(num_variables=canonical_m, delay_1=L, delay_2=L)
            metrics = evaluate_diagnostic_model(model, eval_env, num_seeds=num_seeds, device=device)
            results["experiment_1_latch_decay"][key][L] = metrics
            print(f"Decay lambda={d_val:<6} | Delay L={L:3d} | Overall Acc: {metrics['overall_retention_accuracy']:5.1f}% | Untouched: {metrics['untouched_retention_accuracy']:5.1f}%")

    # --------------------------------------------------------------------------
    # Experiment 2: CIG Gate Bleed & Sharpening
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("EXPERIMENT 2: CIG Gate Sharpening & Leakage Audit (Temperature in [1.0, 0.5, 0.2])")
    print("=" * 90)
    temp_values = [1.0, 0.5, 0.2]

    for t_val in temp_values:
        key = f"temp_{t_val}"
        results["experiment_2_gate_bleed"][key] = {}
        model = DiagnosticCGPModel(
            thought_size=12,
            plastic_decay=1.0,
            gate_temperature=t_val,
            routing_mode="sparse_topk",
        ).to(device)
        train_diagnostic_model(model, train_env, num_steps=train_steps, device=device)

        for L in [16, 64, 128]:
            eval_env = MultiThreadedLatentDependencyEnv(num_variables=canonical_m, delay_1=L, delay_2=L)
            metrics = evaluate_diagnostic_model(model, eval_env, num_seeds=num_seeds, device=device)
            results["experiment_2_gate_bleed"][key][L] = metrics
            print(f"Gate Temp T={t_val:<4} | Delay L={L:3d} | Overall Acc: {metrics['overall_retention_accuracy']:5.1f}% | Gate Salience: {metrics['mean_salience']:.4f}")

    # --------------------------------------------------------------------------
    # Experiment 3: Slot Width Capacity Scaling (W in [12, 24, 48, 64])
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("EXPERIMENT 3: Slot Width Capacity Scaling (Testing W in [12, 24, 48, 64])")
    print("=" * 90)
    widths = [12, 24, 48, 64]

    for w_val in widths:
        key = f"width_{w_val}"
        results["experiment_3_slot_width"][key] = {}
        model = DiagnosticCGPModel(
            thought_size=w_val,
            plastic_decay=1.0,
            routing_mode="sparse_topk",
        ).to(device)
        p_count = count_params(model)["total"]
        train_diagnostic_model(model, train_env, num_steps=train_steps, device=device)

        for L in delay_horizons:
            eval_env = MultiThreadedLatentDependencyEnv(num_variables=canonical_m, delay_1=L, delay_2=L)
            metrics = evaluate_diagnostic_model(model, eval_env, num_seeds=num_seeds, device=device)
            metrics["params"] = p_count
            results["experiment_3_slot_width"][key][L] = metrics
            print(f"Slot Width W={w_val:2d} ({p_count/1e3:5.1f}k params) | Delay L={L:3d} | Overall Acc: {metrics['overall_retention_accuracy']:5.1f}% | Latency: {metrics['latency_ms_mean']:.2f} ms")

    # --------------------------------------------------------------------------
    # Experiment 4: Routing Modality & The Long-Horizon Crossover
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("EXPERIMENT 4: Routing Modality & Crossover (Disconnected vs Top-4 vs Dense vs Dependency-Gated)")
    print("=" * 90)
    routing_modes = [
        ("disconnected", "Disconnected (A=0)"),
        ("sparse_topk", "Static Top-4 Routing"),
        ("dense", "Dense All-to-All Attention"),
        ("dependency_gated", "Dependency-Gated Dynamic Routing"),
    ]

    for mode_key, mode_name in routing_modes:
        results["experiment_4_routing_crossover"][mode_key] = {}
        model = DiagnosticCGPModel(
            thought_size=12,
            plastic_decay=1.0,
            routing_mode=mode_key,
        ).to(device)
        train_diagnostic_model(model, train_env, num_steps=train_steps, device=device)

        for L in delay_horizons:
            eval_env = MultiThreadedLatentDependencyEnv(num_variables=canonical_m, delay_1=L, delay_2=L)
            metrics = evaluate_diagnostic_model(model, eval_env, num_seeds=num_seeds, device=device)
            metrics["name"] = mode_name
            results["experiment_4_routing_crossover"][mode_key][L] = metrics
            print(f"{mode_name:32s} | Delay L={L:3d} | Overall Acc: {metrics['overall_retention_accuracy']:5.1f}% | Untouched: {metrics['untouched_retention_accuracy']:5.1f}% | Latency: {metrics['latency_ms_mean']:.2f} ms")

    return results


# ==============================================================================
# 4. Telemetry and Report Generation
# ==============================================================================

def generate_diagnostic_report(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate structured JSON telemetry and Markdown report for diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "2026-09-07-mtld-degradation-causal-diagnostic.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw JSON telemetry to {json_path}")

    md_path = output_dir / "2026-09-07-mtld-degradation-causal-diagnostic.md"
    lines = [
        "# MTLD Degradation Causal Diagnostic Report",
        "**Date:** 2026-09-07  ",
        "**Status:** `[MEASURED]` Causal decomposition of the delay degradation failure curve and routing crossover.  \n",
        "## 1. Executive Summary",
        r"Following the discovery of the delay failure curve in MTLD-Extreme (where retention decays from 38.3% to 19.1% across $L=16 \to 128$), this diagnostic systematically audited the 4 root-cause hypotheses:",
        r"1. **Hypothesis 1 (Passive Latch Decay)**: Tested whether passive decay in $P_t = \lambda P_{t-1} + \eta \Delta P$ erodes memory over 256 delay ticks.",
        r"2. **Hypothesis 2 (CIG Gate Bleed)**: Audited mean salience $\bar{g}_{\text{delay}}$ on uninformative sensory noise.",
        r"3. **Hypothesis 3 (Slot Width / Dimensional Capacity)**: Swept slot width $W \in [12, 24, 48, 64]$ to determine if hyperspherical subspace collapse causes long-horizon forgetting.",
        r"4. **Hypothesis 4 (Routing Modality & Crossover)**: Audited the Disconnected vs. Routed crossover and evaluated Dependency-Gated Dynamic Routing." + "\n",
        "## 2. Experiment 1: Latch Persistence Sweep (Lambda in [0.99, 0.999, 0.9999, 1.0])",
        "| Lambda Decay | Delay L=16 | Delay L=32 | Delay L=64 | Delay L=128 | Retention Drop (16->128) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for key, l_dict in results["experiment_1_latch_decay"].items():
        acc16 = l_dict[16]["overall_retention_accuracy"]
        acc32 = l_dict[32]["overall_retention_accuracy"]
        acc64 = l_dict[64]["overall_retention_accuracy"]
        acc128 = l_dict[128]["overall_retention_accuracy"]
        drop = acc16 - acc128
        lines.append(f"| **{key}** | {acc16:.1f}% | {acc32:.1f}% | {acc64:.1f}% | **{acc128:.1f}%** | -{drop:.1f}% |")

    lines.extend([
        "\n## 3. Experiment 2: CIG Gate Bleed & Sharpening",
        "| Gate Temperature | Delay L=16 Acc | Delay L=64 Acc | Delay L=128 Acc | Mean Gate Salience |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])
    for key, l_dict in results["experiment_2_gate_bleed"].items():
        acc16 = l_dict[16]["overall_retention_accuracy"]
        acc64 = l_dict[64]["overall_retention_accuracy"]
        acc128 = l_dict[128]["overall_retention_accuracy"]
        sal = l_dict[16]["mean_salience"]
        lines.append(f"| **{key}** | {acc16:.1f}% | {acc64:.1f}% | **{acc128:.1f}%** | {sal:.4f} |")

    lines.extend([
        "\n## 4. Experiment 3: Slot Width Capacity Scaling (W in [12, 24, 48, 64])",
        "| Slot Width | Params | Delay L=16 | Delay L=32 | Delay L=64 | Delay L=128 | Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for key, l_dict in results["experiment_3_slot_width"].items():
        p_c = l_dict[16]["params"]
        acc16 = l_dict[16]["overall_retention_accuracy"]
        acc32 = l_dict[32]["overall_retention_accuracy"]
        acc64 = l_dict[64]["overall_retention_accuracy"]
        acc128 = l_dict[128]["overall_retention_accuracy"]
        lat = l_dict[16]["latency_ms_mean"]
        lines.append(f"| **{key}** | {p_c:,d} | {acc16:.1f}% | {acc32:.1f}% | {acc64:.1f}% | **{acc128:.1f}%** | {lat:.2f} ms |")

    lines.extend([
        "\n## 5. Experiment 4: Routing Modality & The Long-Horizon Crossover",
        "| Routing Strategy | Delay L=16 | Delay L=32 | Delay L=64 | Delay L=128 | Latency (L=16) | Latency (L=128) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for key, l_dict in results["experiment_4_routing_crossover"].items():
        name = l_dict[16]["name"]
        acc16 = l_dict[16]["overall_retention_accuracy"]
        acc32 = l_dict[32]["overall_retention_accuracy"]
        acc64 = l_dict[64]["overall_retention_accuracy"]
        acc128 = l_dict[128]["overall_retention_accuracy"]
        lat16 = l_dict[16]["latency_ms_mean"]
        lat128 = l_dict[128]["latency_ms_mean"]
        lines.append(f"| **{name}** | **{acc16:.1f}%** | {acc32:.1f}% | {acc64:.1f}% | **{acc128:.1f}%** | {lat16:.2f} ms | {lat128:.2f} ms |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Generated Markdown report at {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTLD Degradation Causal Diagnostic Suite")
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="brain/docs/runs")
    args = parser.parse_args()

    results = run_degradation_diagnostics(
        num_seeds=args.num_seeds,
        train_steps=args.train_steps,
        device_str=args.device,
    )
    generate_diagnostic_report(results, output_dir=Path(args.output_dir))
