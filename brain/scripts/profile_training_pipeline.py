"""Stage H: profile frozen Core V1 training on GB10.

Profiles the canonical multitask trainer: frame-by-frame recurrent feeding,
6000 steps, K=32 W=120 C=3. Measures wall-clock composition:
forward / backward / optimizer / data-prep / sync overhead.
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
sys.modules["esc_prof"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import TASKS  # noqa: E402


def frames_tensor(ep, device):
    return torch.from_numpy(np.stack(ep.frames).astype(np.float32) / 255.0) \
        .permute(0, 3, 1, 2).to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    print("=== TRAINING PIPELINE PROFILE (GB10) ===")

    model = esc.VectorizedPseudoBrain(thoughtlets=32, width=120,
                                      heads=4, cycles=3, init_seed=42).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    bank = []
    for fn in TASKS:
        for i in range(24):
            r = np.random.default_rng(42 + hash(fn.__name__) % 99991 + i)
            bank.append(fn(r))

    timings = {"env_frame_access": 0.0, "tensor_prep": 0.0,
               "forward": 0.0, "loss": 0.0, "backward": 0.0,
               "optimizer": 0.0, "total": 0.0}
    frames_per_step = []

    model.train()
    t_all0 = time.perf_counter()
    for step in range(args.steps):
        t_s = time.perf_counter()

        t0 = time.perf_counter()
        ep = bank[step % len(bank)]
        n_frames = ep.decision_frame_index + 1
        frames_per_step.append(n_frames)
        timings["env_frame_access"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        ft = frames_tensor(ep, device)
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        timings["tensor_prep"] += time.perf_counter() - t0

        state = None
        out = None
        t0 = time.perf_counter()
        for j in range(n_frames):
            out, state = model(ft[j:j+1], ctrl, dt, state=state)
        torch.cuda.synchronize()
        timings["forward"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        logits = out.proposals.action_logits
        target = torch.full((1, logits.shape[1]), ep.label,
                            dtype=torch.long, device=device)
        loss = F.cross_entropy(logits.view(-1, 5), target.view(-1))
        timings["loss"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.cuda.synchronize()
        timings["backward"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        opt.step()
        torch.cuda.synchronize()
        timings["optimizer"] += time.perf_counter() - t0

        timings["total"] += time.perf_counter() - t_s

    wall = time.perf_counter() - t_all0
    total_steps = sum(frames_per_step)
    report = {
        "steps": args.steps,
        "wall_s": round(wall, 2),
        "ms_per_trainstep": round(timings["total"] / args.steps * 1000, 2),
        "frames_per_trainstep_mean": round(float(np.mean(frames_per_step)), 2),
        "frames_per_sec": round(total_steps / wall, 1),
        "breakdown_pct": {k: round(v / timings["total"] * 100, 1)
                          for k, v in timings.items() if k != "total"},
        "transitions_per_sec": round(total_steps / wall * 8, 1),  # 25 tasks x ~24 eps scale note
    }
    print(json.dumps(report, indent=2))
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
