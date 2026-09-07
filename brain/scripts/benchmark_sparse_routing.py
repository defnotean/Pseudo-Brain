"""Benchmark forward-pass latency of dense vs sparse top-k routing on CPU.

Phase 2.6 Workstream E:
Measures latency as thought capacity scales across K in [16, 32, 64].
Verifies that latency remains well within the real-time budget (<= 1.5 ms at K=64).
"""

from __future__ import annotations

import statistics
import time
import torch

from irene_brain.model.sparse_thought_router import SparseThoughtRouter
from irene_brain.model.brain_cell import BrainCell


def benchmark_router(
    router: SparseThoughtRouter,
    summaries: torch.Tensor,
    *,
    warmup: int = 50,
    repeats: int = 300,
) -> dict[str, float]:
    """Benchmark a router instance with warmup and latency percentiles."""
    with torch.no_grad():
        for _ in range(warmup):
            router(summaries)

        latencies: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            router(summaries)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies.sort()
    mean_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)
    p90_lat = latencies[int(len(latencies) * 0.90)]
    p99_lat = latencies[int(len(latencies) * 0.99)]
    return {
        "mean": mean_lat,
        "p50": median_lat,
        "p90": p90_lat,
        "p99": p99_lat,
    }


def benchmark_brain_cell(
    cell: BrainCell,
    batch_size: int,
    thoughtlets: int,
    width: int,
    *,
    warmup: int = 20,
    repeats: int = 100,
) -> dict[str, float]:
    """Benchmark an end-to-end tied BrainCell recurrent forward pass."""
    registers = 3
    sensors = torch.randn(batch_size, 16, width)
    action_time = torch.randn(batch_size, 4, width)
    belief = torch.randn(batch_size, 16, width)
    working_memory = torch.randn(batch_size, 8, width)
    thoughts = torch.randn(batch_size, thoughtlets, registers, width)
    goal_context = torch.randn(batch_size, 2, width)
    retrieved = torch.randn(batch_size, thoughtlets, 2, width)
    elapsed = torch.tensor([[1.0 / 60.0]])

    with torch.no_grad():
        for _ in range(warmup):
            cell(
                belief=belief,
                working_memory=working_memory,
                thoughts=thoughts,
                sensors=sensors,
                action_time_tokens=action_time,
                goal_context=goal_context,
                retrieved_memory=retrieved,
                elapsed_seconds=elapsed,
                allow_routing=True,
            )

        latencies: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            cell(
                belief=belief,
                working_memory=working_memory,
                thoughts=thoughts,
                sensors=sensors,
                action_time_tokens=action_time,
                goal_context=goal_context,
                retrieved_memory=retrieved,
                elapsed_seconds=elapsed,
                allow_routing=True,
            )
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    latencies.sort()
    return {
        "mean": statistics.mean(latencies),
        "p50": statistics.median(latencies),
        "p90": latencies[int(len(latencies) * 0.90)],
        "p99": latencies[int(len(latencies) * 0.99)],
    }


def run_benchmarks() -> dict[str, object]:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    k_values = [16, 32, 64]
    width = 384  # Full canonical core width
    batch_size = 1

    print("=" * 95)
    print("Pseudo-Brain Sparse Thoughtlet Routing Scaling Benchmark (Phase 2.6 Workstream E)")
    print(f"Platform: CPU (Single-thread deterministic) | Core Width W={width} | Batch Size B={batch_size}")
    print("=" * 95)
    print(
        f"{'K':<5} | {'Mode':<18} | {'Mean (ms)':<10} | {'p50 (ms)':<10} | {'p99 (ms)':<10} | "
        f"{'Speedup':<9} | {'Interference Reduction':<24}"
    )
    print("-" * 95)

    results: dict[str, object] = {"router": {}, "cell": {}}

    for K in k_values:
        summaries = torch.randn(batch_size, K, width)

        # 1. Dense routing (all K-1 peers)
        router_dense = SparseThoughtRouter(width=width, dense_routing=True)
        stats_dense = benchmark_router(router_dense, summaries)

        print(
            f"{K:<5} | {'Dense (all-to-all)':<18} | {stats_dense['mean']:<10.4f} | "
            f"{stats_dense['p50']:<10.4f} | {stats_dense['p99']:<10.4f} | "
            f"{'1.00x':<9} | {'0.0% (all peers mix)':<24}"
        )

        results["router"][f"K{K}_dense"] = stats_dense

        # 2. Sparse Top-k (k=4)
        router_k4 = SparseThoughtRouter(width=width, routed_neighbors=4)
        stats_k4 = benchmark_router(router_k4, summaries)
        speedup_k4 = stats_dense["mean"] / stats_k4["mean"]
        reduc_k4 = 100.0 * (1.0 - min(4, K - 1) / (K - 1))
        print(
            f"{K:<5} | {'Sparse (k=4)':<18} | {stats_k4['mean']:<10.4f} | "
            f"{stats_k4['p50']:<10.4f} | {stats_k4['p99']:<10.4f} | "
            f"{speedup_k4:<9.2f}x | {reduc_k4:<5.1f}% uninfluenced peers"
        )
        results["router"][f"K{K}_k4"] = stats_k4

        # 3. Sparse Top-k (k=2)
        router_k2 = SparseThoughtRouter(width=width, routed_neighbors=2)
        stats_k2 = benchmark_router(router_k2, summaries)
        speedup_k2 = stats_dense["mean"] / stats_k2["mean"]
        reduc_k2 = 100.0 * (1.0 - min(2, K - 1) / (K - 1))
        print(
            f"{K:<5} | {'Sparse (k=2)':<18} | {stats_k2['mean']:<10.4f} | "
            f"{stats_k2['p50']:<10.4f} | {stats_k2['p99']:<10.4f} | "
            f"{speedup_k2:<9.2f}x | {reduc_k2:<5.1f}% uninfluenced peers"
        )
        results["router"][f"K{K}_k2"] = stats_k2
        print("-" * 95)

    print("\n" + "=" * 95)
    print("End-to-End BrainCell (Single Block) Latency Scaling:")
    print("=" * 95)
    print(f"{'K':<5} | {'Routing Mode':<18} | {'Mean (ms)':<10} | {'p50 (ms)':<10} | {'p99 (ms)':<10} | {'Budget <= 1.5ms':<15}")
    print("-" * 95)

    for K in k_values:
        cell_dense = BrainCell(width=width, heads=8, routed_neighbors=2, blocks=1, dense_routing=True)
        cell_sparse = BrainCell(width=width, heads=8, routed_neighbors=2, blocks=1, dense_routing=False)

        bc_dense = benchmark_brain_cell(cell_dense, batch_size, K, width)
        bc_sparse = benchmark_brain_cell(cell_sparse, batch_size, K, width)

        status_dense = "PASS" if bc_dense["mean"] <= 1.5 else "EXCEEDED"
        status_sparse = "PASS" if bc_sparse["mean"] <= 1.5 else "EXCEEDED"

        print(f"{K:<5} | {'Dense Block':<18} | {bc_dense['mean']:<10.4f} | {bc_dense['p50']:<10.4f} | {bc_dense['p99']:<10.4f} | {status_dense:<15}")
        print(f"{K:<5} | {'Sparse Block (k=2)':<18} | {bc_sparse['mean']:<10.4f} | {bc_sparse['p50']:<10.4f} | {bc_sparse['p99']:<10.4f} | {status_sparse:<15}")
        print("-" * 95)

        results["cell"][f"K{K}_dense"] = bc_dense
        results["cell"][f"K{K}_sparse"] = bc_sparse

    # Verify real-time budget contract:
    k64_sparse_router = results["router"]["K64_k2"]["mean"]
    print(f"\nReal-Time Contract Verification:")
    print(f"- SparseTopK Router Latency at K=64 (W=384, k=2): {k64_sparse_router:.4f} ms")
    if k64_sparse_router <= 1.5:
        print(f"[PASS] Latency is comfortably below the 1.5 ms real-time ceiling (margin: {1.5 - k64_sparse_router:.4f} ms)")
    else:
        print(f"[FAIL] Latency exceeded 1.5 ms budget!")

    return results


if __name__ == "__main__":
    run_benchmarks()
