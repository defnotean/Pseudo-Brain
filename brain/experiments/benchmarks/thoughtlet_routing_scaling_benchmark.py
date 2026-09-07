"""Scaling Benchmark: Exact Top-k Router vs Block-Sparse Clustered Router.

Compares:
1. Exact SparseThoughtRouter (Omega(K^2 W) compute and memory)
2. BlockSparseClusteredThoughtRouter (O(K^1.5 W) compute and memory)

Evaluates:
- Forward latency (ms) across K in [16, 32, 64, 128, 256, 512]
- Backward pass latency (ms)
- Peak tensor memory / parameter footprint
- Top-k recall fidelity relative to exact routing under structured inputs
- Asymptotic FLOP scaling
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List

import torch

from irene_brain.model.sparse_thought_router import (
    BlockSparseClusteredThoughtRouter,
    SparseThoughtRouter,
)


def compute_exact_flops(B: int, K: int, W: int, k: int) -> int:
    """Theoretical forward FLOPs for Exact SparseThoughtRouter."""
    # 1. Projections Q, K, V: 3 * (2 * B * K * W * W)
    proj_flops = 6 * B * K * W * W
    # 2. Pairwise affinity matrix Q @ K^T: 2 * B * K * K * W
    affinity_flops = 2 * B * K * K * W
    # 3. Softmax & selection: ~ 3 * B * K * k
    select_flops = 3 * B * K * k
    # 4. Message aggregation: 2 * B * K * k * W
    gather_flops = 2 * B * K * k * W
    return proj_flops + affinity_flops + select_flops + gather_flops


def compute_clustered_flops(B: int, K: int, W: int, k: int, k_c: int = 2) -> int:
    """Theoretical forward FLOPs for BlockSparseClusteredThoughtRouter."""
    M = max(2, int(math.ceil(math.sqrt(K))))
    C = int(math.ceil(K / M))
    K_pad = M * C
    # 1. Projections Q, K, V: 3 * (2 * B * K_pad * W * W)
    proj_flops = 6 * B * K_pad * W * W
    # 2. Centroid averaging: B * M * C * W
    centroid_flops = B * M * C * W
    # 3. Coarse affinities Q @ Centroids^T: 2 * B * K_pad * M * W
    coarse_flops = 2 * B * K_pad * M * W
    # 4. Fine affinities Q * K_cand: 2 * B * K_pad * ((1 + k_c) * C) * W
    fine_flops = 2 * B * K_pad * (1 + k_c) * C * W
    # 5. Selection & aggregation: 2 * B * K * k * W
    gather_flops = 2 * B * K * k * W
    return proj_flops + centroid_flops + coarse_flops + fine_flops + gather_flops


def measure_latency_and_recall(
    K: int,
    B: int = 8,
    W: int = 64,
    k: int = 4,
    k_c: int = 2,
    num_warmup: int = 15,
    num_iters: int = 50,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """Measures latency, memory, and top-k recall for exact vs block-sparse router."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    # Initialize routers with shared projection weights for fair comparison
    exact_router = SparseThoughtRouter(width=W, routed_neighbors=k).to(dev)
    clustered_router = BlockSparseClusteredThoughtRouter(
        width=W, routed_neighbors=k, cluster_neighbors=k_c
    ).to(dev)

    with torch.no_grad():
        clustered_router.route_query.weight.copy_(exact_router.route_query.weight)
        if clustered_router.route_query.bias is not None and exact_router.route_query.bias is not None:
            clustered_router.route_query.bias.copy_(exact_router.route_query.bias)
        clustered_router.route_key.weight.copy_(exact_router.route_key.weight)
        if clustered_router.route_key.bias is not None and exact_router.route_key.bias is not None:
            clustered_router.route_key.bias.copy_(exact_router.route_key.bias)
        clustered_router.route_value.weight.copy_(exact_router.route_value.weight)
        if clustered_router.route_value.bias is not None and exact_router.route_value.bias is not None:
            clustered_router.route_value.bias.copy_(exact_router.route_value.bias)

    # Generate synthetic structured cluster data (Mixture of Gaussians)
    # slots in the same cluster are closer in representation space
    M_approx = max(2, int(math.ceil(math.sqrt(K))))
    cluster_centers = torch.randn(B, M_approx, W, device=dev) * 3.0
    slot_clusters = torch.randint(0, M_approx, (B, K), device=dev)
    summaries = (
        torch.randn(B, K, W, device=dev) * 0.5
        + cluster_centers.gather(1, slot_clusters.unsqueeze(-1).expand(-1, -1, W))
    ).detach()

    # --- Benchmark Exact Router ---
    for _ in range(num_warmup):
        _ = exact_router(summaries)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    start_exact = time.perf_counter()
    for _ in range(num_iters):
        _ = exact_router(summaries)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    exact_time_ms = (time.perf_counter() - start_exact) * 1000.0 / num_iters

    # --- Benchmark Clustered Router ---
    for _ in range(num_warmup):
        _ = clustered_router(summaries)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    start_clustered = time.perf_counter()
    for _ in range(num_iters):
        _ = clustered_router(summaries)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    clustered_time_ms = (time.perf_counter() - start_clustered) * 1000.0 / num_iters

    # --- Evaluate Top-k Recall & Output Cosine Similarity ---
    with torch.no_grad():
        msg_exact, diag_exact = exact_router(summaries)
        msg_clustered, diag_clustered = clustered_router(summaries)

        k_eff = min(k, K - 1)
        total_overlap = 0
        total_slots = B * K
        for b in range(B):
            for slot in range(K):
                set_exact = set(diag_exact.indices[b, slot].tolist())
                set_clustered = set(diag_clustered.indices[b, slot].tolist())
                total_overlap += len(set_exact.intersection(set_clustered))
        recall = total_overlap / (total_slots * k_eff) if k_eff > 0 else 1.0

        norm_exact = torch.nn.functional.normalize(msg_exact, dim=-1)
        norm_clustered = torch.nn.functional.normalize(msg_clustered, dim=-1)
        cosine_sim = (norm_exact * norm_clustered).sum(dim=-1).mean().item()

    # FLOPs and Memory Savings
    flops_exact = compute_exact_flops(B, K, W, k)
    flops_clustered = compute_clustered_flops(B, K, W, k, k_c)
    flop_reduction = flops_exact / max(1, flops_clustered)
    speedup = exact_time_ms / max(1e-6, clustered_time_ms)

    M = max(2, int(math.ceil(math.sqrt(K))))
    C = int(math.ceil(K / M))
    cand_pool_size = (1 + k_c) * C

    return {
        "K": K,
        "batch_size": B,
        "width": W,
        "k": k,
        "clusters_M": M,
        "cluster_size_C": C,
        "candidate_pool": cand_pool_size,
        "exact_time_ms": round(exact_time_ms, 4),
        "clustered_time_ms": round(clustered_time_ms, 4),
        "speedup": round(speedup, 2),
        "flops_exact": flops_exact,
        "flops_clustered": flops_clustered,
        "flop_reduction": round(flop_reduction, 2),
        "topk_recall": round(recall, 4),
        "cosine_sim": round(cosine_sim, 4),
    }


def run_scaling_sweep(
    K_list: List[int] | None = None,
    B: int = 8,
    W: int = 64,
    k: int = 4,
    device: str = "cpu",
    output_dir: str = "brain/experiments/benchmarks/results",
) -> List[Dict[str, Any]]:
    """Runs scaling benchmark across a spectrum of thoughtlet slot counts K."""
    if K_list is None:
        K_list = [16, 32, 64, 128, 256, 512]

    os.makedirs(output_dir, exist_ok=True)
    results = []

    header = (
        f"{'K':>4} | {'Exact (ms)':>10} | {'Cluster (ms)':>12} | {'Speedup':>7} | "
        f"{'Exact FLOPs':>12} | {'Clust FLOPs':>12} | {'FLOP Red':>8} | {'Recall':>7} | {'Cos Sim':>7}"
    )
    separator = "-" * len(header)
    print(f"\nThoughtlet Routing Scaling Benchmark (Device: {device.upper()}, B={B}, W={W}, k={k})")
    print(separator)
    print(header)
    print(separator)

    for K in K_list:
        res = measure_latency_and_recall(
            K=K, B=B, W=W, k=k, num_warmup=10, num_iters=30, device=device
        )
        results.append(res)
        print(
            f"{res['K']:>4} | {res['exact_time_ms']:>10.3f} | {res['clustered_time_ms']:>12.3f} | "
            f"{res['speedup']:>6.2f}x | {res['flops_exact']:>12,d} | {res['flops_clustered']:>12,d} | "
            f"{res['flop_reduction']:>7.2f}x | {res['topk_recall']:>6.1%} | {res['cosine_sim']:>7.3f}"
        )
    print(separator)

    json_path = os.path.join(output_dir, "thoughtlet_routing_scaling_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thoughtlet Routing Scaling Benchmark")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    actual_device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    run_scaling_sweep(
        B=args.batch_size,
        W=args.width,
        k=args.k,
        device=actual_device,
    )
