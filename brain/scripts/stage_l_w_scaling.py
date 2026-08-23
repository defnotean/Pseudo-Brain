"""Stage L: controlled W-axis scaling sweep (Phase 2.7).

Frozen Core V1 architecture; ONLY width varies (120/240/480), K=32 C=3 fixed.
Batched engine (equivalence-passed), 3 seeds per config, locked eval protocol.
Question: what does learned capacity (W) buy on the torture suite?
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "esc_base", os.path.join(_here, "..", "..", "dgx_phase2_definitive_escalation.py"))
esc = importlib.util.module_from_spec(_spec)
sys.modules["esc_scale"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import TASKS  # noqa: E402

SEEDS = [42, 142, 242]
WIDTHS = [120, 240, 480]


def frames_tensor(ep, device):
    return torch.from_numpy(np.stack(ep.frames).astype(np.float32) / 255.0) \
        .permute(0, 3, 1, 2).to(device)


def build_bank(seed, eps_per_task=24):
    bank = []
    for fn in TASKS:
        for i in range(eps_per_task):
            r = np.random.default_rng(seed + hash(fn.__name__) % 99991 + i)
            bank.append(fn(r))
    return bank


def train_batched(model, device, steps, seed, B=16):
    """Approved batched engine (length-bucketed, final-frame loss)."""
    bank = build_bank(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()
    by_len = {}
    for idx, ep in enumerate(bank):
        by_len.setdefault(ep.decision_frame_index + 1, []).append(idx)
    order = []
    for ln in sorted(by_len):
        members = by_len[ln]
        for k in range(0, len(members), B):
            group = members[k:k+B]
            while len(group) < B:
                group.append(members[(k + len(group)) % len(members)])
            order.append((ln, group))
    step = 0
    while step < steps:
        for ln, group in order:
            if step >= steps:
                break
            eps = [bank[i] for i in group]
            ftb = torch.stack([
                torch.stack([frames_tensor(e, device)[j] for j in range(ln)])
                for e in eps])
            ctrl = torch.zeros(len(eps), 307, device=device)
            dt = torch.tensor([0.016667], device=device)
            state = None
            for j in range(ln):
                out_j, state = model(ftb[:, j], ctrl, dt, state=state)
            logits = out_j.proposals.action_logits
            tgt = torch.tensor([e.label for e in eps], dtype=torch.long,
                               device=device).unsqueeze(1).expand(-1, logits.shape[1])
            loss = F.cross_entropy(logits.reshape(-1, 5), tgt.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1


@torch.no_grad()
def eval_lift(model, device, episodes=30, base_seed=20260822):
    model.eval()
    lifts = []
    for fn in TASKS:
        ok = 0
        chance = float(getattr(fn(np.random.default_rng(0)), "chance", 0.0))
        for i in range(episodes):
            r = np.random.default_rng(base_seed + i * 7919)
            ep = fn(r)
            ft = frames_tensor(ep, device)
            ctrl = torch.zeros(1, 307, device=device)
            dt = torch.tensor([0.016667], device=device)
            state = None
            act = 0
            for j in range(ep.decision_frame_index + 1):
                out, state = model(ft[j:j+1], ctrl, dt, state=state)
                act = int(torch.argmax(out.action_dist[0]).item())
            ok += int(act == ep.label)
        lifts.append(ok / episodes - chance)
    model.train()
    return float(np.mean(lifts))


@torch.no_grad()
def latency_p50(model, device, n=30):
    model.eval()
    ft = torch.rand(1, 3, 32, 32, device=device)
    ctrl = torch.zeros(1, 307, device=device)
    dt = torch.tensor([0.016667], device=device)
    for _ in range(8):
        model(ft, ctrl, dt)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(ft, ctrl, dt)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    model.train()
    return round(float(np.percentile(ts, 50)), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    print("=== STAGE L: W-AXIS SCALING SWEEP ===")

    payload = {}
    for w in WIDTHS:
        cfg = {"seeds": {}}
        t_cfg = time.perf_counter()
        for seed in SEEDS:
            model = esc.VectorizedPseudoBrain(
                thoughtlets=32, width=w, heads=4, cycles=3,
                init_seed=seed).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            t0 = time.perf_counter()
            train_batched(model, device, args.steps, seed)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            lift = eval_lift(model, device)
            lat = latency_p50(model, device)
            vram = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            torch.cuda.reset_peak_memory_stats()
            cfg["seeds"][str(seed)] = {
                "lift": round(lift, 4), "params": n_params,
                "train_wall_s": round(wall, 1),
                "latency_p50_ms": lat, "peak_vram_mb": vram}
            print(f"W={w} s{seed}: lift={lift:+.4f} params={n_params} "
                  f"wall={wall:.0f}s lat={lat}ms")
        lifts = [v["lift"] for v in cfg["seeds"].values()]
        cfg["mean_lift"] = round(float(np.mean(lifts)), 4)
        cfg["std_lift"] = round(float(np.std(lifts)), 4)
        cfg["params"] = cfg["seeds"][str(SEEDS[0])]["params"]
        cfg["latency_p50_ms"] = cfg["seeds"][str(SEEDS[0])]["latency_p50_ms"]
        payload[f"W{w}"] = cfg
        print(f"W={w} SUMMARY: mean={cfg['mean_lift']} ±{cfg['std_lift']} "
              f"({time.perf_counter()-t_cfg:.0f}s total)")

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
