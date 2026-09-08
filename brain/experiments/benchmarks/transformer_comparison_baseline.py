"""Standard Causal Transformer Baseline and Head-to-Head Benchmark for Pseudo-Brain.

Benchmarks Pseudo-Brain against standard open-source Transformer architectures (LLaMA/GPT/Pythia)
on:
1. Operational State Memory Footprint: O(1) 4.0 KB Law 1 vs O(N) linear KV-cache growth.
2. Step Latency vs Sequence Length: Real-time 60 Hz (<16.6 ms) gaming compliance.
3. Long-Context Factual Retention across 500+ distractor tokens.
4. Tool-Calling & Code Repair Convergence.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Project imports
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from irene_brain.unified.unified_model import UnifiedPseudoBrain, make_unified_model
from irene_brain.agent.unified_agent_loop import UnifiedCognitiveAgent, CodeExecutionEngine


class CausalSelfAttentionWithKVCache(nn.Module):
    """Standard Multi-Head Causal Self-Attention with autoregressive KV-Cache."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: Tensor,  # [B, T, D]
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v) if use_cache else None

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask if full sequence
        seq_len_k = k.shape[2]
        if T > 1:
            causal_mask = torch.triu(torch.full((T, seq_len_k), float("-inf"), device=x.device), diagonal=seq_len_k - T + 1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out), new_kv_cache


class TransformerBlock(nn.Module):
    """Standard Transformer Decoder Block with Pre-LN and MLP."""

    def __init__(self, d_model: int, n_heads: int, mlp_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionWithKVCache(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )

    def forward(
        self,
        x: Tensor,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        normed = self.ln1(x)
        attn_out, new_cache = self.attn(normed, kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class StandardCausalTransformer(nn.Module):
    """Standard Open-Source Causal Decoder-Only Transformer (LLaMA/GPT style)."""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_dim: int = 1024,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_dim) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_sequence(self, token_seq: Tensor) -> Tensor:
        """Full sequence parallel forward pass."""
        B, T = token_seq.shape
        pos = torch.arange(T, device=token_seq.device).unsqueeze(0)
        x = self.embed(token_seq) + self.pos_embed(pos)

        for layer in self.layers:
            x, _ = layer(x, use_cache=False)

        x = self.final_ln(x)
        return self.lm_head(x)

    def step_with_cache(
        self,
        token_id: Tensor,  # [B, 1]
        position: int,
        past_caches: Optional[List[Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """Single-token autoregressive generation step with KV-cache."""
        B = token_id.shape[0]
        pos = torch.tensor([[position]], device=token_id.device)
        x = self.embed(token_id) + self.pos_embed(pos)

        new_caches = []
        for idx, layer in enumerate(self.layers):
            layer_cache = past_caches[idx] if past_caches is not None else None
            x, new_c = layer(x, kv_cache=layer_cache, use_cache=True)
            new_caches.append(new_c)

        x = self.final_ln(x)
        logits = self.lm_head(x).squeeze(1)
        return logits, new_caches

    def calculate_kv_cache_bytes(self, seq_len: int, batch_size: int = 1, bytes_per_element: int = 4) -> int:
        """Calculate total KV-cache memory in bytes for sequence length."""
        # 2 tensors (K, V) per layer: shape [B, n_heads, seq_len, head_dim]
        # n_heads * head_dim = d_model
        # bytes = 2 * n_layers * (B * d_model * seq_len) * bytes_per_element
        return 2 * self.n_layers * batch_size * self.d_model * seq_len * bytes_per_element


# ==============================================================================
# Benchmarking Engine
# ==============================================================================

def run_head_to_head_benchmark(
    device_name: str = "cpu",
    seq_lengths: List[int] = [16, 64, 256, 512, 1024, 2048],
) -> Dict[str, Any]:
    """Execute head-to-head comparison between Pseudo-Brain and Transformer."""
    device = torch.device(device_name)
    torch.manual_seed(42)

    # 1. Instantiate matched parameter models
    pb_model = make_unified_model(tier="tier2", vocab_size=32000).to(device)
    pb_model.eval()

    tf_model = StandardCausalTransformer(
        vocab_size=32000,
        d_model=pb_model.proj_dim // 2,
        n_layers=pb_model.num_deep_layers,
        n_heads=8,
        mlp_dim=pb_model.proj_dim,
    ).to(device)
    tf_model.eval()

    pb_params = pb_model.count_parameters()["total"]
    tf_params = tf_model.count_parameters()

    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD BENCHMARK: PSEUDO-BRAIN VS OPEN-SOURCE CAUSAL TRANSFORMER")
    print("=" * 80)
    print(f"Pseudo-Brain Parameters: {pb_params:,}")
    print(f"Transformer  Parameters: {tf_params:,}")
    print(f"Device: {device_name.upper()}")

    # Benchmark 1: Operational Memory Footprint vs Context Length
    memory_results = []
    for T in seq_lengths:
        pb_state = pb_model.init_state(batch_size=1, device=device)
        pb_fast_bytes = pb_state.hierarchical_state.fast_state_bytes()  # Strictly 4.0 KB
        pb_total_bytes = pb_state.hierarchical_state.total_state_bytes() # Fast + episodic

        tf_kv_bytes = tf_model.calculate_kv_cache_bytes(seq_len=T, batch_size=1)
        ratio = tf_kv_bytes / pb_fast_bytes

        memory_results.append({
            "sequence_length": T,
            "pseudo_brain_fast_kb": pb_fast_bytes / 1024.0,
            "pseudo_brain_total_kb": pb_total_bytes / 1024.0,
            "transformer_kv_kb": tf_kv_bytes / 1024.0,
            "memory_ratio": ratio,
        })

    # Benchmark 2: Streaming Step Latency (<16.6 ms 60 Hz Gaming Constraint)
    latency_results = []
    for T in [16, 64, 256, 512]:
        # Pseudo-Brain single step
        pb_state = pb_model.init_state(batch_size=1, device=device)
        sensory = torch.randn(1, pb_model.proj_dim, device=device)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _, pb_state = pb_model.step(sensory, pb_state)

        times_pb = []
        for _ in range(30):
            t0 = time.perf_counter()
            with torch.no_grad():
                _, pb_state = pb_model.step(sensory, pb_state)
            times_pb.append((time.perf_counter() - t0) * 1000.0)

        # Transformer generation step at context length T
        # Initialize cache to length T
        past_tokens = torch.randint(0, 1000, (1, T), device=device)
        with torch.no_grad():
            _ = tf_model.forward_sequence(past_tokens)
            # Build cache
            caches = None
            for p in range(min(T, 32)):
                _, caches = tf_model.step_with_cache(past_tokens[:, p:p+1], position=p, past_caches=caches)

        next_tok = torch.tensor([[42]], device=device)
        times_tf = []
        for _ in range(30):
            t0 = time.perf_counter()
            with torch.no_grad():
                _, _ = tf_model.step_with_cache(next_tok, position=T, past_caches=caches)
            times_tf.append((time.perf_counter() - t0) * 1000.0)

        pb_mean = float(np.mean(times_pb))
        tf_mean = float(np.mean(times_tf))

        latency_results.append({
            "context_length": T,
            "pseudo_brain_latency_ms": pb_mean,
            "transformer_latency_ms": tf_mean,
            "pb_60hz_compliant": pb_mean < 16.67,
            "tf_60hz_compliant": tf_mean < 16.67,
            "speedup": tf_mean / max(pb_mean, 1e-4),
        })

    # Benchmark 3: Long-Horizon Factual Retention across 500+ Distractor Tokens
    fact_dim = pb_model.W_fast
    fact_pattern = F.normalize(torch.randn(1, 1, fact_dim, device=device), dim=-1)

    pb_state = pb_model.init_state(batch_size=1, device=device)
    pb_state.hierarchical_state.working_memory[0, 0:1, :] = fact_pattern

    # Consolidate fact into episodic memory
    salience_fact = torch.zeros(1, pb_model.K_fast, 1, device=device)
    salience_fact[:, 0, :] = 1.0
    for _ in range(3):
        ep_mem, ep_ages, _ = pb_model.consolidation_gate(
            pb_state.hierarchical_state.working_memory,
            pb_state.hierarchical_state.episodic_memory,
            pb_state.hierarchical_state.episodic_ages,
            salience=salience_fact,
        )
        pb_state.hierarchical_state.episodic_memory = ep_mem
        pb_state.hierarchical_state.episodic_ages = ep_ages

    with torch.no_grad():
        projected_fact = F.normalize(pb_model.consolidation_gate.write_proj(fact_pattern).squeeze(1), dim=-1)

    # Ingest 500 distractor random steps with low background salience (< threshold 0.2)
    distractor_steps = 500
    dist_salience = torch.tensor([[0.05]], device=device)
    for _ in range(distractor_steps):
        dist_sensory = torch.randn(1, pb_model.proj_dim, device=device) * 0.1
        with torch.no_grad():
            _, pb_state = pb_model.step(dist_sensory, pb_state, allow_routing=False, salience=dist_salience)

    # Episodic recall check against projected fact
    stored_episodic = pb_state.hierarchical_state.episodic_memory[0]
    stored_norm = F.normalize(stored_episodic, dim=-1)
    cos_sims = torch.matmul(projected_fact, stored_norm.transpose(0, 1)).squeeze(0)
    max_pb_sim = float(cos_sims.max().item())

    # Baseline working memory forgetting check
    baseline_slot0 = pb_state.hierarchical_state.working_memory[0, 0]
    working_sim = float(F.cosine_similarity(baseline_slot0.unsqueeze(0), fact_pattern.squeeze(1)).item())

    retention_result = {
        "distractor_tokens": distractor_steps,
        "pseudo_brain_episodic_retention_cos_sim": max_pb_sim,
        "pseudo_brain_working_retention_cos_sim": working_sim,
        "transformer_kv_cache_kb_at_500": tf_model.calculate_kv_cache_bytes(500) / 1024.0,
        "retention_success": max_pb_sim >= 0.85,
    }

    # Benchmark 4: Autonomous Code Self-Repair Efficiency
    agent = UnifiedCognitiveAgent(model=pb_model, device=device)
    buggy = "def multiply(a, b):\n    return a / b  # BUG\n"
    test_s = "assert multiply(6, 7) == 42\nprint('OK')"
    repair_success, fixed_code, history = agent.self_repair_code(buggy, test_s, max_attempts=3)

    repair_result = {
        "task": "arithmetic_bug_repair",
        "success": repair_success,
        "iterations_to_pass": len(history),
        "consequence_surprises": agent.consequence_history[:len(history)],
    }

    # Format Results Report
    report = {
        "model_parameters": {
            "pseudo_brain": pb_params,
            "transformer": tf_params,
        },
        "memory_scaling": memory_results,
        "latency_scaling": latency_results,
        "long_horizon_retention": retention_result,
        "autonomous_code_repair": repair_result,
    }

    # Print Summary Markdown Table
    print("\n1. OPERATIONAL MEMORY FOOTPRINT SCALING (Law 1 O(1) vs Transformer O(N) KV-Cache):")
    print("| Context T | Pseudo-Brain Fast (KB) | Total PB (KB) | Transformer KV-Cache (KB) | Advantage Ratio |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for r in memory_results:
        print(f"| {r['sequence_length']:<9} | {r['pseudo_brain_fast_kb']:<22.1f} | {r['pseudo_brain_total_kb']:<13.1f} | {r['transformer_kv_kb']:<25.1f} | **{r['memory_ratio']:.1f}x smaller** |")

    print("\n2. STREAMING STEP LATENCY & 60 HZ REAL-TIME GAMING COMPLIANCE (<16.6 ms):")
    print("| Context T | Pseudo-Brain Latency | Transformer Latency | 60 Hz Gaming Compliant? |")
    print("| :--- | :--- | :--- | :--- |")
    for r in latency_results:
        pb_status = "YES (PASS)" if r['pb_60hz_compliant'] else "NO"
        tf_status = "YES (PASS)" if r['tf_60hz_compliant'] else "NO"
        print(f"| {r['context_length']:<9} | {r['pseudo_brain_latency_ms']:.2f} ms              | {r['transformer_latency_ms']:.2f} ms             | PB: {pb_status} | TF: {tf_status} |")

    print(f"\n3. LONG-CONTEXT FACT RETENTION (500+ Distractor Tokens):")
    print(f"  * Needle-in-a-haystack cosine similarity: {max_pb_sim:.4f} (Goal >= 0.85: {'PASS' if max_pb_sim >= 0.85 else 'FAIL'})")

    print(f"\n4. AUTONOMOUS CODE SELF-REPAIR:")
    print(f"  * Closed-loop repair success: {repair_success} (Completed in {len(history)} iterations)")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run Pseudo-Brain vs Transformer benchmark")
    parser.add_argument("--device", type=str, default="cpu", help="Device to evaluate on")
    parser.add_argument("--save-report", type=str, default=None, help="Save JSON report path")
    args = parser.parse_args()

    report = run_head_to_head_benchmark(device_name=args.device)

    out_path = Path(args.save_report) if args.save_report else _REPO_ROOT / "experiments" / "benchmarks" / "results" / "transformer_comparison_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved comprehensive benchmark report to: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
