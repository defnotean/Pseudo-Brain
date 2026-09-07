"""Language Cognitive Benchmark Harness for Native Semantic Pseudo-Brain.

Evaluates semantic sequence processing, multi-threaded conversations, preemption recovery,
thread isolation, cross-thread dependencies, and latency under the 16.67ms (60 Hz) CPU budget.
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

from irene_brain.semantic.native_semantic_model import (
    NativeSemanticPseudoBrain,
    make_semantic_model,
)
from irene_brain.semantic.streaming_engine import StreamingCognitiveSession
from irene_brain.semantic.tokenizer import SemanticTokenizer
from semantic_benchmark.curriculum_datasets import (
    LanguageCurriculumGenerator,
    SemanticEpisode,
    build_curriculum_batch,
)


def train_semantic_model(
    model: nn.Module,
    generator: LanguageCurriculumGenerator,
    num_steps: int = 250,
    batch_size: int = 16,
    lr: float = 3e-3,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Train semantic model on the multi-threaded curriculum."""
    dev = device or torch.device("cpu")
    model.to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    losses: List[float] = []
    for step in range(num_steps):
        tokens, targets, threads, _ = build_curriculum_batch(generator, batch_size=batch_size, seed=step * 101)
        tokens = tokens.to(dev)
        targets = targets.to(dev)
        threads = threads.to(dev)

        optimizer.zero_grad()
        logits = model(tokens, thread_seq=threads, allow_routing=True)  # [B, T, V]

        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.item()))

    return losses


TASK_NAME_MAP = {
    "stage_a_copy": "stage_a_copy",
    "stage_a_associative_binding": "stage_a_binding",
    "stage_a_binding": "stage_a_binding",
    "stage_b_delayed_recall": "stage_b_recall",
    "stage_b_recall": "stage_b_recall",
    "stage_b_interleaved_conversations": "stage_b_interleaved",
    "stage_b_interleaved": "stage_b_interleaved",
    "stage_b_preemption": "stage_b_preemption",
    "stage_b_cross_thread_dependency": "stage_b_dependency",
    "stage_b_dependency": "stage_b_dependency",
}


def evaluate_semantic_model(
    model: nn.Module,
    generator: LanguageCurriculumGenerator,
    num_eval_episodes: int = 100,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Exhaustive evaluation across individual task metrics and failure attribution."""
    dev = device or torch.device("cpu")
    model.to(dev)
    model.eval()

    task_results: Dict[str, List[float]] = {
        "stage_a_copy": [],
        "stage_a_binding": [],
        "stage_b_recall": [],
        "stage_b_interleaved": [],
        "stage_b_preemption": [],
        "stage_b_dependency": [],
    }
    latencies_ms: List[float] = []

    eval_batches = int(math.ceil(num_eval_episodes / batch_size))
    total_tokens_tested = 0
    total_tokens_correct = 0

    with torch.no_grad():
        for b_idx in range(eval_batches):
            tokens, targets, threads, episodes = build_curriculum_batch(
                generator,
                batch_size=batch_size,
                seed=9000 + b_idx * 17,
            )
            tokens = tokens.to(dev)
            targets = targets.to(dev)
            threads = threads.to(dev)

            t0 = time.perf_counter()
            logits = model(tokens, thread_seq=threads, allow_routing=True)
            dt_ms = (time.perf_counter() - t0) * 1000.0 / tokens.shape[1]  # Per-step latency
            latencies_ms.append(dt_ms)

            preds = logits.argmax(dim=-1)  # [B, T]

            for b in range(len(episodes)):
                ep = episodes[b]
                mask = targets[b] != -100
                if mask.sum() == 0:
                    continue

                t_tgt = targets[b][mask]
                t_pred = preds[b][mask]
                matches = (t_pred == t_tgt).float()
                tok_acc = float(matches.mean().item()) * 100.0
                seq_acc = 100.0 if matches.min().item() > 0.5 else 0.0

                total_tokens_tested += int(mask.sum().item())
                total_tokens_correct += int(matches.sum().item())

                # Track by task with normalized mapping
                norm_task = TASK_NAME_MAP.get(ep.task_name, ep.task_name)
                if norm_task in task_results:
                    task_results[norm_task].append(seq_acc)

    overall_token_acc = (total_tokens_correct / max(1, total_tokens_tested)) * 100.0
    preemption_rec = float(np.mean(task_results["stage_b_preemption"])) if task_results["stage_b_preemption"] else 0.0
    isolation_acc = float(np.mean(task_results["stage_b_interleaved"])) if task_results["stage_b_interleaved"] else 0.0
    dependency_acc = float(np.mean(task_results["stage_b_dependency"])) if task_results["stage_b_dependency"] else 0.0
    delayed_recall = float(np.mean(task_results["stage_b_recall"])) if task_results["stage_b_recall"] else 0.0
    copy_acc = float(np.mean(task_results["stage_a_copy"])) if task_results["stage_a_copy"] else 0.0
    binding_acc = float(np.mean(task_results["stage_a_binding"])) if task_results["stage_a_binding"] else 0.0

    # Benchmark per-token update latency via streaming single-token steps (B=1)
    streaming_lats_ms: List[float] = []
    with torch.no_grad():
        s_state = model.init_state(1, dev) if hasattr(model, "init_state") else None
        dummy_tok = torch.tensor([15], dtype=torch.long, device=dev)
        dummy_th = torch.tensor([0], dtype=torch.long, device=dev)
        # Warmup
        for _ in range(15):
            if isinstance(model, NativeSemanticPseudoBrain):
                _, s_state = model.step(dummy_tok, s_state, thread_ids=dummy_th, allow_routing=True)
            elif hasattr(model, "step"):
                _, s_state = model.step(dummy_tok, s_state)
        # Benchmark 100 streaming token iterations
        for _ in range(100):
            t0 = time.perf_counter()
            if isinstance(model, NativeSemanticPseudoBrain):
                _, s_state = model.step(dummy_tok, s_state, thread_ids=dummy_th, allow_routing=True)
            elif hasattr(model, "step"):
                _, s_state = model.step(dummy_tok, s_state)
            streaming_lats_ms.append((time.perf_counter() - t0) * 1000.0)

    lats = np.array(streaming_lats_ms) if streaming_lats_ms else (np.array(latencies_ms) if latencies_ms else np.array([0.0]))

    # K_eff for multi-threaded models: K * Recovery * Isolation
    K_val = getattr(model, "K", 1)
    K_eff = K_val * (preemption_rec / 100.0) * (isolation_acc / 100.0)

    # State size in bytes
    param_count = sum(p.numel() for p in model.parameters())
    if hasattr(model, "K") and hasattr(model, "thought_size"):
        # Pseudo-Brain state: K * W * 4 bytes + K * V * 4 bytes (P_t)
        recurrent_state_bytes = model.K * model.thought_size * 4 + model.K * model.vocab_size * 4
    elif hasattr(model, "hidden_dim"):
        # GRU/SSM state
        recurrent_state_bytes = model.hidden_dim * 4
    else:
        recurrent_state_bytes = 0

    return {
        "parameters": param_count,
        "token_accuracy": round(overall_token_acc, 2),
        "copy_accuracy": round(copy_acc, 2),
        "binding_accuracy": round(binding_acc, 2),
        "delayed_recall_accuracy": round(delayed_recall, 2),
        "preemption_recovery_accuracy": round(preemption_rec, 2),
        "thread_isolation_accuracy": round(isolation_acc, 2),
        "cross_thread_dependency_accuracy": round(dependency_acc, 2),
        "effective_threads_Keff": round(K_eff, 2),
        "latency_mean_ms": round(float(np.mean(lats)), 3),
        "latency_p50_ms": round(float(np.percentile(lats, 50)), 3),
        "latency_p90_ms": round(float(np.percentile(lats, 90)), 3),
        "latency_p99_ms": round(float(np.percentile(lats, 99)), 3),
        "recurrent_state_bytes": recurrent_state_bytes,
        "under_16_67ms": bool(np.all(lats <= 16.67)),
    }


def run_language_cognitive_benchmark(
    architectures: Optional[List[str]] = None,
    K: int = 16,
    num_train_steps: int = 240,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Execute complete architecture benchmark across language cognition tasks."""
    if architectures is None:
        architectures = [
            "pseudo_brain",
            "pseudo_brain_no_cgp",
            "pseudo_brain_no_threads",
            "gru",
            "ssm",
            "reactive",
        ]

    device = torch.device(device_str)
    tokenizer = SemanticTokenizer(max_threads=K)
    generator = LanguageCurriculumGenerator(tokenizer=tokenizer, max_threads=K)

    results: Dict[str, Any] = {
        "config": {
            "K": K,
            "vocab_size": tokenizer.vocab_size,
            "train_steps": num_train_steps,
            "device": device_str,
        },
        "architectures": {},
    }

    print("\n" + "=" * 95)
    print(f"PSEUDO-BRAIN NATIVE SEMANTIC COGNITIVE BENCHMARK (K={K} THREADS)")
    print("=" * 95)

    for arch in architectures:
        print(f"\n[Evaluating Architecture: {arch}]")
        model = make_semantic_model(
            arch,
            vocab_size=tokenizer.vocab_size,
            K=K,
            thought_size=32,
            embed_dim=64,
            proj_dim=128,
        )
        p_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {p_count:,}")

        t0 = time.perf_counter()
        losses = train_semantic_model(model, generator, num_steps=num_train_steps, device=device)
        train_time = time.perf_counter() - t0
        print(f"  Training complete in {train_time:.2f}s (Final Loss: {losses[-1]:.4f})")

        metrics = evaluate_semantic_model(model, generator, num_eval_episodes=100, device=device)
        metrics["final_train_loss"] = round(losses[-1], 4)
        metrics["train_time_sec"] = round(train_time, 2)
        results["architectures"][arch] = metrics

        print(f"  Token Acc: {metrics['token_accuracy']}% | Recovery: {metrics['preemption_recovery_accuracy']}% | Isolation: {metrics['thread_isolation_accuracy']}% | K_eff: {metrics['effective_threads_Keff']}")
        print(f"  Latency: {metrics['latency_mean_ms']:.3f} ms (p90: {metrics['latency_p90_ms']:.3f} ms, 60Hz: {metrics['under_16_67ms']})")

    return results


def run_scaling_sweep(
    K_values: Optional[List[int]] = None,
    W_values: Optional[List[int]] = None,
    num_train_steps: int = 80,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    """Execute Concurrency and Width Scaling Sweep for Native Semantic Pseudo-Brain."""
    if K_values is None:
        K_values = [2, 4, 8, 16, 32, 64]
    if W_values is None:
        W_values = [12, 24, 48]

    device = torch.device(device_str)
    results: Dict[str, Any] = {
        "sweep_config": {
            "K_values": K_values,
            "W_values": W_values,
            "train_steps": num_train_steps,
            "device": device_str,
        },
        "grid": {},
    }

    print("\n" + "=" * 95)
    print(f"PSEUDO-BRAIN CONCURRENCY & WIDTH SCALING SWEEP (K in {K_values}, W in {W_values})")
    print("=" * 95)

    for K in K_values:
        tokenizer = SemanticTokenizer(max_threads=K)
        generator = LanguageCurriculumGenerator(tokenizer=tokenizer, max_threads=K)

        for W in W_values:
            key = f"K{K}_W{W}"
            print(f"\n--- Running Sweep Configuration: {key} (K={K}, W={W}) ---")
            model = make_semantic_model(
                "pseudo_brain",
                vocab_size=tokenizer.vocab_size,
                K=K,
                thought_size=W,
                embed_dim=64,
                proj_dim=128,
            )
            p_count = sum(p.numel() for p in model.parameters())

            t0 = time.perf_counter()
            losses = train_semantic_model(model, generator, num_steps=num_train_steps, device=device)
            train_time = time.perf_counter() - t0

            metrics = evaluate_semantic_model(model, generator, num_eval_episodes=80, device=device)
            metrics["K"] = K
            metrics["W"] = W
            metrics["final_train_loss"] = round(losses[-1], 4)
            metrics["train_time_sec"] = round(train_time, 2)
            results["grid"][key] = metrics

            print(f"  K_eff: {metrics['effective_threads_Keff']:.2f} | Recovery: {metrics['preemption_recovery_accuracy']}% | Isolation: {metrics['thread_isolation_accuracy']}% | Dependency: {metrics['cross_thread_dependency_accuracy']}%")
            print(f"  Latency mean: {metrics['latency_mean_ms']:.3f} ms (p90: {metrics['latency_p90_ms']:.3f} ms, 60Hz: {metrics['under_16_67ms']})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--sweep", action="store_true", help="Run K and W scaling sweep")
    parser.add_argument("--sweep-output", type=str, default=None)
    args = parser.parse_args()

    if args.sweep:
        sweep_data = run_scaling_sweep(
            num_train_steps=args.steps,
            device_str=args.device,
        )
        if args.sweep_output:
            out_p = Path(args.sweep_output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(sweep_data, f, indent=2)
            print(f"\nSaved scaling sweep results to {out_p}")
    else:
        benchmark_data = run_language_cognitive_benchmark(
            K=args.threads,
            num_train_steps=args.steps,
            device_str=args.device,
        )

        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(benchmark_data, f, indent=2)
            print(f"\nSaved benchmark results to {out_p}")
