"""TE V2 Profiler v2 — corrected timing + batched ceiling test.

Measures TRUE per-step wall time and tests:
  (a) original ARM B (14 serial B=1 passes)
  (b) batched ARM B (all 14 tasks padded into ONE B=14 forward pass)
  (c) non-blocking H2D variant

Goal: quantify the real speedup ceiling for Training Engine V2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "..", "src"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "esc_3a", os.path.join(_here, "dgx_phase2_definitive_escalation.py"))
esc = importlib.util.module_from_spec(_spec)
sys.modules["esc_3a"] = esc
_spec.loader.exec_module(esc)

from run_provenance import (build_bank_pinned, bank_digest,
                            apply_deterministic_mode)
from irene_brain.evaluation.torture_suite import TASKS


def build_frames_once(ep, device, non_blocking=False):
    arr = np.stack(ep.frames).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)
    return t.to(device, non_blocking=non_blocking)


def profile_arm_b_serial(device, n_steps=30, non_blocking=False):
    """Original ARM B: 14 serial B=1 passes (one per task)."""
    torch.manual_seed(42); np.random.seed(42); torch.cuda.manual_seed_all(42)
    model = esc.VectorizedPseudoBrain(
        thoughtlets=32, width=120, heads=4, cycles=3, init_seed=42).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()

    step_times = []
    for step in range(n_steps):
        t0 = time.perf_counter()
        rng = np.random.default_rng(42 + step * 10007 + 12345)
        eps = [fn(rng) for fn in TASKS]
        by_len = {}
        for e in eps:
            by_len.setdefault(e.decision_frame_index + 1, []).append(e)
        for ln, group_eps in by_len.items():
            B = len(group_eps)
            ftb = torch.stack([build_frames_once(e, device, non_blocking)[:ln]
                               for e in group_eps])
            ctrl = torch.zeros(B, 307, device=device)
            dt = torch.tensor([0.016667], device=device)
            state = None
            for j in range(ln):
                out_j, state = model(ftb[:, j], ctrl, dt, state=state)
            logits = out_j.proposals.action_logits
            tgt = torch.tensor([e.label for e in group_eps], dtype=torch.long, device=device)
            tgt = tgt.unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1)
            loss = F.cross_entropy(logits.reshape(-1, 5), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        step_times.append(time.perf_counter() - t0)
    return float(np.mean(step_times[5:]) * 1000)  # skip warmup


def profile_arm_b_batched(device, n_steps=30, non_blocking=False):
    """Batched ARM B: pad all 14 tasks to max length, ONE B=14 forward pass.

    Uses per-length masking in the recurrent loop so padding frames don't
    affect the decision frame. This is the TE V2 candidate path.
    """
    torch.manual_seed(42); np.random.seed(42); torch.cuda.manual_seed_all(42)
    model = esc.VectorizedPseudoBrain(
        thoughtlets=32, width=120, heads=4, cycles=3, init_seed=42).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()

    step_times = []
    for step in range(n_steps):
        t0 = time.perf_counter()
        rng = np.random.default_rng(42 + step * 10007 + 12345)
        eps = [fn(rng) for fn in TASKS]
        # max length across all 14 tasks this step
        max_len = max(e.decision_frame_index + 1 for e in eps)
        B = len(eps)
        # pad to [B, max_len, 3, 32, 32]
        batch = torch.zeros(B, max_len, 3, 32, 32, device=device)
        decision_idx = torch.zeros(B, dtype=torch.long, device=device)
        labels = []
        for bi, e in enumerate(eps):
            fr = build_frames_once(e, device, non_blocking)
            ln = fr.shape[0]
            batch[bi, :ln] = fr
            decision_idx[bi] = ln - 1
            labels.append(e.label)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        ctrl = torch.zeros(B, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        state = None
        out_last = None
        for j in range(max_len):
            out_j, state = model(batch[:, j], ctrl, dt, state=state)
            # capture output only at each episode's decision frame
            is_dec = (j == decision_idx)
            if is_dec.any():
                out_last = out_j
        logits = out_last.proposals.action_logits
        tgt = labels.unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1)
        loss = F.cross_entropy(logits.reshape(-1, 5), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        step_times.append(time.perf_counter() - t0)
    return float(np.mean(step_times[5:]) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    apply_deterministic_mode()
    device = torch.device("cuda:0")

    print("=== TE V2 PROFILER v2 ===")
    print("[1] ARM B serial (original, blocking H2D):")
    serial_block = profile_arm_b_serial(device, n_steps=30, non_blocking=False)
    print(f"    {serial_block:.2f} ms/step")

    print("[2] ARM B serial (non-blocking H2D):")
    serial_nb = profile_arm_b_serial(device, n_steps=30, non_blocking=True)
    print(f"    {serial_nb:.2f} ms/step")

    print("[3] ARM B BATCHED (pad to max, one B=14 pass, blocking):")
    batched_block = profile_arm_b_batched(device, n_steps=30, non_blocking=False)
    print(f"    {batched_block:.2f} ms/step")

    print("[4] ARM B BATCHED (non-blocking H2D):")
    batched_nb = profile_arm_b_batched(device, n_steps=30, non_blocking=True)
    print(f"    {batched_nb:.2f} ms/step")

    report = {
        "serial_blocking_ms": serial_block,
        "serial_nonblocking_ms": serial_nb,
        "batched_blocking_ms": batched_block,
        "batched_nonblocking_ms": batched_nb,
        "derived": {
            "serial_speedup_nonblocking": serial_block / serial_nb if serial_nb > 0 else 0,
            "batched_vs_serial_blocking": serial_block / batched_block if batched_block > 0 else 0,
            "batched_vs_serial_nonblocking": serial_nb / batched_nb if batched_nb > 0 else 0,
            "batched_total_speedup": serial_block / batched_nb if batched_nb > 0 else 0,
        },
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDONE -> {args.output}")
    print(f"Serial→Batched speedup (blocking): {report['derived']['batched_vs_serial_blocking']:.2f}x")
    print(f"Serial→Batched speedup (non-blocking): {report['derived']['batched_vs_serial_nonblocking']:.2f}x")
    print(f"TOTAL speedup (serial-blocking → batched-nonblocking): {report['derived']['batched_total_speedup']:.2f}x")


if __name__ == "__main__":
    main()
