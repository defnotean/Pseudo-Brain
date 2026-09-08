"""Benchmark Suite: Sequential Unrolling vs Parallel Associative Scan in Pseudo-Brain.

Breaks the Sequential Recurrence Wall for scaling Pseudo-Brain from 36M to 1B parameters.

Evaluates:
  1. Pure Recurrent Scan Kernel:
     - Sequential Loop (Python for t in range(T):)
     - Work-Efficient Parallel Associative Scan (O(T) work, O(log T) depth)
     - Hillis-Steele Parallel Associative Scan (O(T log T) work, O(log T) depth)
  2. Fused Recurrent Cell (Forward and Forward+Backward):
     - Sequential Unrolling
     - Parallel Associative Scan
Across:
  - Batch sizes: 8, 16
  - Sequence lengths: 64, 128, 512, 1024
  - Hidden / thoughtlet dimensions: 64 (Tier 2 Law 1 contract)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

# Ensure project root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from irene_brain.parallel.parallel_scan import (
    parallel_scan,
    sequential_scan,
)
from irene_brain.parallel.fused_cell import FusedBrainCellCore


def benchmark_kernel(
    batch_sizes: List[int] = [8, 16],
    seq_lens: List[int] = [64, 128, 512, 1024],
    hidden_dim: int = 64,
    warmup_iters: int = 5,
    bench_iters: int = 15,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Benchmarks pure linear recurrence kernels: Sequential vs Parallel Associative Scan."""
    dev = torch.device(device)
    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 90)
    print(" 1. PURE RECURRENT KERNEL BENCHMARK (h_t = a_t * h_{t-1} + b_t)")
    print(f" Device: {device} | Warmup: {warmup_iters} | Iterations: {bench_iters} | Dim: {hidden_dim}")
    print("=" * 90)

    for B in batch_sizes:
        for T in seq_lens:
            torch.manual_seed(42)
            # Create gates and candidate inputs
            a = (torch.rand(B, T, hidden_dim, device=dev) * 0.85 + 0.1)
            b = torch.randn(B, T, hidden_dim, device=dev)
            h0 = torch.randn(B, hidden_dim, device=dev)

            # Accuracy verification
            out_seq = sequential_scan(a, b, h0=h0, dim=1)
            out_we = parallel_scan(a, b, h0=h0, dim=1, method="work_efficient")
            out_hs = parallel_scan(a, b, h0=h0, dim=1, method="hillis_steele")

            max_err_we = (out_seq - out_we).abs().max().item()
            max_err_hs = (out_seq - out_hs).abs().max().item()
            mean_err_we = (out_seq - out_we).abs().mean().item()

            # --- Forward Latency ---
            # Warmup
            for _ in range(warmup_iters):
                _ = sequential_scan(a, b, h0=h0, dim=1)
                _ = parallel_scan(a, b, h0=h0, dim=1, method="work_efficient")
                _ = parallel_scan(a, b, h0=h0, dim=1, method="hillis_steele")

            # Sequential Forward
            t0 = time.perf_counter()
            for _ in range(bench_iters):
                _ = sequential_scan(a, b, h0=h0, dim=1)
            t_seq_fwd = (time.perf_counter() - t0) / bench_iters * 1000.0

            # Work-Efficient Parallel Forward
            t0 = time.perf_counter()
            for _ in range(bench_iters):
                _ = parallel_scan(a, b, h0=h0, dim=1, method="work_efficient")
            t_we_fwd = (time.perf_counter() - t0) / bench_iters * 1000.0

            # Hillis-Steele Parallel Forward
            t0 = time.perf_counter()
            for _ in range(bench_iters):
                _ = parallel_scan(a, b, h0=h0, dim=1, method="hillis_steele")
            t_hs_fwd = (time.perf_counter() - t0) / bench_iters * 1000.0

            # --- Forward + Backward Latency ---
            a_seq = a.clone().requires_grad_(True)
            b_seq = b.clone().requires_grad_(True)
            a_we = a.clone().requires_grad_(True)
            b_we = b.clone().requires_grad_(True)
            a_hs = a.clone().requires_grad_(True)
            b_hs = b.clone().requires_grad_(True)

            # Warmup Fwd+Bwd
            for _ in range(warmup_iters):
                sequential_scan(a_seq, b_seq, h0=h0, dim=1).sum().backward()
                a_seq.grad = None; b_seq.grad = None
                parallel_scan(a_we, b_we, h0=h0, dim=1, method="work_efficient").sum().backward()
                a_we.grad = None; b_we.grad = None

            # Sequential Fwd+Bwd
            t0 = time.perf_counter()
            for _ in range(bench_iters):
                a_seq.grad = None; b_seq.grad = None
                loss = sequential_scan(a_seq, b_seq, h0=h0, dim=1).sum()
                loss.backward()
            t_seq_bwd = (time.perf_counter() - t0) / bench_iters * 1000.0

            # Work-Efficient Parallel Fwd+Bwd
            t0 = time.perf_counter()
            for _ in range(bench_iters):
                a_we.grad = None; b_we.grad = None
                loss = parallel_scan(a_we, b_we, h0=h0, dim=1, method="work_efficient").sum()
                loss.backward()
            t_we_bwd = (time.perf_counter() - t0) / bench_iters * 1000.0

            # Hillis-Steele Parallel Fwd+Bwd
            t0 = time.perf_counter()
            for _ in range(bench_iters):
                a_hs.grad = None; b_hs.grad = None
                loss = parallel_scan(a_hs, b_hs, h0=h0, dim=1, method="hillis_steele").sum()
                loss.backward()
            t_hs_bwd = (time.perf_counter() - t0) / bench_iters * 1000.0

            speedup_fwd = t_seq_fwd / max(t_we_fwd, 1e-6)
            speedup_bwd = t_seq_bwd / max(t_we_bwd, 1e-6)

            entry = {
                "batch_size": B,
                "seq_len": T,
                "hidden_dim": hidden_dim,
                "seq_fwd_ms": t_seq_fwd,
                "we_fwd_ms": t_we_fwd,
                "hs_fwd_ms": t_hs_fwd,
                "speedup_fwd": speedup_fwd,
                "seq_bwd_ms": t_seq_bwd,
                "we_bwd_ms": t_we_bwd,
                "hs_bwd_ms": t_hs_bwd,
                "speedup_bwd": speedup_bwd,
                "max_abs_err": max_err_we,
                "mean_abs_err": mean_err_we,
            }
            results.append(entry)

            print(
                f"B={B:2d}, T={T:4d} | Fwd: Seq={t_seq_fwd:6.2f}ms, ParWE={t_we_fwd:5.2f}ms ({speedup_fwd:5.2f}x) | "
                f"Fwd+Bwd: Seq={t_seq_bwd:6.2f}ms, ParWE={t_we_bwd:5.2f}ms ({speedup_bwd:5.2f}x) | MaxErr={max_err_we:.2e}"
            )

    return results


def benchmark_fused_cell(
    batch_sizes: List[int] = [8, 16],
    seq_lens: List[int] = [64, 128, 512, 1024],
    input_dim: int = 128,
    thought_dim: int = 64,
    warmup_iters: int = 3,
    bench_iters: int = 10,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Benchmarks full FusedBrainCellCore module: Sequential unroll vs Parallel scan."""
    dev = torch.device(device)
    results: List[Dict[str, Any]] = []

    print("\n" + "=" * 90)
    print(" 2. FUSED CELL MODULE BENCHMARK (FeedForward + Recurrence)")
    print(f" Device: {device} | InputDim: {input_dim} | ThoughtDim: {thought_dim} (Tier 2 Law 1)")
    print("=" * 90)

    cell = FusedBrainCellCore(input_size=input_dim, thought_size=thought_dim).to(dev)

    for B in batch_sizes:
        for T in seq_lens:
            torch.manual_seed(42)
            x = torch.randn(B, T, input_dim, device=dev)
            h0 = torch.randn(B, thought_dim, device=dev)

            # Accuracy check
            out_seq = cell.forward_sequential(x, h0=h0)
            out_par = cell(x, h0=h0, method="work_efficient")
            max_err = (out_seq - out_par).abs().max().item()

            # Warmup
            for _ in range(warmup_iters):
                _ = cell.forward_sequential(x, h0=h0)
                _ = cell(x, h0=h0, method="work_efficient")

            # Sequential Forward
            t0 = time.perf_counter()
            for _ in range(bench_iters):
                _ = cell.forward_sequential(x, h0=h0)
            t_seq_fwd = (time.perf_counter() - t0) / bench_iters * 1000.0

            # Parallel Forward
            t0 = time.perf_counter()
            for _ in range(bench_iters):
                _ = cell(x, h0=h0, method="work_efficient")
            t_par_fwd = (time.perf_counter() - t0) / bench_iters * 1000.0

            # Forward + Backward
            x_seq = x.clone().requires_grad_(True)
            x_par = x.clone().requires_grad_(True)

            # Sequential Fwd+Bwd
            t0 = time.perf_counter()
            for _ in range(bench_iters):
                cell.zero_grad()
                loss = cell.forward_sequential(x_seq, h0=h0).sum()
                loss.backward()
            t_seq_bwd = (time.perf_counter() - t0) / bench_iters * 1000.0

            # Parallel Fwd+Bwd
            t0 = time.perf_counter()
            for _ in range(bench_iters):
                cell.zero_grad()
                loss = cell(x_par, h0=h0, method="work_efficient").sum()
                loss.backward()
            t_par_bwd = (time.perf_counter() - t0) / bench_iters * 1000.0

            speedup_fwd = t_seq_fwd / max(t_par_fwd, 1e-6)
            speedup_bwd = t_seq_bwd / max(t_par_bwd, 1e-6)

            entry = {
                "batch_size": B,
                "seq_len": T,
                "input_dim": input_dim,
                "thought_dim": thought_dim,
                "seq_fwd_ms": t_seq_fwd,
                "par_fwd_ms": t_par_fwd,
                "speedup_fwd": speedup_fwd,
                "seq_bwd_ms": t_seq_bwd,
                "par_bwd_ms": t_par_bwd,
                "speedup_bwd": speedup_bwd,
                "max_abs_err": max_err,
            }
            results.append(entry)

            print(
                f"B={B:2d}, T={T:4d} | Fwd: Seq={t_seq_fwd:6.2f}ms, Par={t_par_fwd:5.2f}ms ({speedup_fwd:5.2f}x) | "
                f"Fwd+Bwd: Seq={t_seq_bwd:6.2f}ms, Par={t_par_bwd:5.2f}ms ({speedup_bwd:5.2f}x) | MaxErr={max_err:.2e}"
            )

    return results


def print_markdown_summary(kernel_results: List[Dict[str, Any]], cell_results: List[Dict[str, Any]]) -> str:
    """Formats benchmark results as Markdown tables."""
    lines: List[str] = []
    lines.append("# Pseudo-Brain Parallel Associative Scan Benchmark Report")
    lines.append("\n### 1. Pure Recurrent Scan Kernel Benchmark")
    lines.append("| Batch | SeqLen (T) | Sequential Fwd (ms) | Parallel Fwd (ms) | Fwd Speedup | Sequential Fwd+Bwd (ms) | Parallel Fwd+Bwd (ms) | Fwd+Bwd Speedup | Max Abs Error |")
    lines.append("|:-----:|:----------:|:-------------------:|:-----------------:|:-----------:|:-----------------------:|:---------------------:|:---------------:|:-------------:|")
    for r in kernel_results:
        lines.append(
            f"| {r['batch_size']:5d} | {r['seq_len']:10d} | {r['seq_fwd_ms']:19.2f} | {r['we_fwd_ms']:17.2f} | {r['speedup_fwd']:10.2f}x | {r['seq_bwd_ms']:23.2f} | {r['we_bwd_ms']:21.2f} | {r['speedup_bwd']:14.2f}x | {r['max_abs_err']:13.2e} |"
        )

    lines.append("\n### 2. Fused Recurrent Cell Module Benchmark")
    lines.append("| Batch | SeqLen (T) | Sequential Fwd (ms) | Parallel Fwd (ms) | Fwd Speedup | Sequential Fwd+Bwd (ms) | Parallel Fwd+Bwd (ms) | Fwd+Bwd Speedup | Max Abs Error |")
    lines.append("|:-----:|:----------:|:-------------------:|:-----------------:|:-----------:|:-----------------------:|:---------------------:|:---------------:|:-------------:|")
    for r in cell_results:
        lines.append(
            f"| {r['batch_size']:5d} | {r['seq_len']:10d} | {r['seq_fwd_ms']:19.2f} | {r['par_fwd_ms']:17.2f} | {r['speedup_fwd']:10.2f}x | {r['seq_bwd_ms']:23.2f} | {r['par_bwd_ms']:21.2f} | {r['speedup_bwd']:14.2f}x | {r['max_abs_err']:13.2e} |"
        )

    report_str = "\n".join(lines)
    print("\n" + report_str)
    return report_str


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel Associative Scan Benchmark for Pseudo-Brain")
    parser.add_argument("--device", type=str, default="cpu", help="Device to test (cpu, cuda, etc.)")
    parser.add_argument("--save", type=str, default="experiments/bench_parallel_scan_results.json", help="Output file path")
    args = parser.parse_args()

    kernel_res = benchmark_kernel(
        batch_sizes=[8, 16],
        seq_lens=[64, 128, 512, 1024],
        device=args.device,
    )

    cell_res = benchmark_fused_cell(
        batch_sizes=[8, 16],
        seq_lens=[64, 128, 512, 1024],
        device=args.device,
    )

    md_report = print_markdown_summary(kernel_res, cell_res)

    # Save results to json
    save_path = Path(PROJECT_ROOT) / args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "kernel_benchmarks": kernel_res,
        "cell_benchmarks": cell_res,
        "markdown_report": md_report,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved benchmark telemetry to {save_path}")


if __name__ == "__main__":
    main()
