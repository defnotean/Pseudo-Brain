"""Stage K: batched-engine equivalence test (preregistered).

Corrected batched trainer:
- per-episode state carry, final-decision-frame loss only
- padded members masked out of the loss
- B episodes processed per forward wave

Compared against canonical solo trainer, 3 seeds, torture lift.
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
sys.modules["esc_equiv"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import TASKS  # noqa: E402

SEEDS = [42, 142, 242]


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


def train_solo(model, device, steps, seed):
    """Canonical reference recipe."""
    bank = build_bank(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()
    for step in range(steps):
        ep = bank[step % len(bank)]
        ft = frames_tensor(ep, device)
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        state = None
        out = None
        for j in range(ep.decision_frame_index + 1):
            out, state = model(ft[j:j+1], ctrl, dt, state=state)
        logits = out.proposals.action_logits
        target = torch.full((1, logits.shape[1]), ep.label, dtype=torch.long,
                            device=device)
        loss = F.cross_entropy(logits.view(-1, 5), target.view(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def train_batched(model, device, steps, seed, B=16):
    """Batched engine: B episodes in parallel; each member carries its own
    recurrent state; loss on final decision frame only; padded frames are run
    through the forward (state updates on them — same as solo running blank
    frames? NO: solo never sees padding) — so instead we group episodes of
    EQUAL length to avoid padding entirely. Length-bucketed batching."""
    bank = build_bank(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()
    # bucket episode indices by length
    by_len = {}
    for idx, ep in enumerate(bank):
        by_len.setdefault(ep.decision_frame_index + 1, []).append(idx)
    order = []
    for ln in sorted(by_len):
        members = by_len[ln]
        for k in range(0, len(members), B):
            group = members[k:k+B]
            while len(group) < B and len(bank) > 0:
                # top up short groups with repeats from same length bucket
                group.append(members[(k + len(group)) % len(members)])
            order.append((ln, group))

    step = 0
    while step < steps:
        for ln, group in order:
            if step >= steps:
                break
            eps = [bank[i] for i in group]
            ftb = torch.stack([frames_tensor(e, device)[0] for e in eps]) \
                if False else None
            # frames: all episodes here have exactly `ln` frames
            ftb = torch.stack([
                torch.stack([frames_tensor(e, device)[j]
                             for j in range(ln)]) for e in eps])  # [B,T,C,H,W]
            ctrl = torch.zeros(B, 307, device=device)
            dt = torch.tensor([0.016667], device=device)
            state = None
            logits_final = None
            for j in range(ln):
                out_j, state = model(ftb[:, j], ctrl, dt, state=state)
            logits_final = out_j.proposals.action_logits     # [B,K,5]
            tgt = torch.tensor([e.label for e in eps], dtype=torch.long,
                               device=device).unsqueeze(1).expand(B, logits_final.shape[1])
            loss = F.cross_entropy(logits_final.reshape(-1, 5), tgt.reshape(-1))
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    print(f"=== BATCHED ENGINE EQUIVALENCE TEST (B={args.batch}) ===")

    payload = {"arms": {}}
    for arm in ("solo", "batched"):
        seeds_out = {}
        t_arm = time.perf_counter()
        for seed in SEEDS:
            model = esc.VectorizedPseudoBrain(thoughtlets=32, width=120,
                                              heads=4, cycles=3,
                                              init_seed=seed).to(device)
            t0 = time.perf_counter()
            if arm == "solo":
                train_solo(model, device, args.steps, seed)
            else:
                train_batched(model, device, args.steps, seed, B=args.batch)
            torch.cuda.synchronize()
            train_wall = time.perf_counter() - t0
            lift = eval_lift(model, device)
            seeds_out[str(seed)] = {"lift": round(lift, 4),
                                    "train_wall_s": round(train_wall, 2)}
            print(f"{arm} s{seed}: lift={lift:+.4f} wall={train_wall:.1f}s")
        payload["arms"][arm] = {
            "seeds": seeds_out,
            "mean_lift": round(float(np.mean([v["lift"] for v in seeds_out.values()])), 4),
            "total_wall_s": round(time.perf_counter() - t_arm, 2)}

    deltas = [abs(payload["arms"]["solo"]["seeds"][s]["lift"]
                  - payload["arms"]["batched"]["seeds"][s]["lift"])
              for s in map(str, SEEDS)]
    n_close = sum(1 for d in deltas if d <= 0.03)
    mean_delta = float(np.mean(deltas))
    verdict = {"per_seed_deltas": [round(d, 4) for d in deltas],
               "seeds_within_003": n_close,
               "mean_abs_delta": round(mean_delta, 4),
               "PASS_2of3_and_mean_le_002": bool(n_close >= 2 and mean_delta <= 0.02)}
    speedup = payload["arms"]["solo"]["total_wall_s"] / \
        max(payload["arms"]["batched"]["total_wall_s"], 1e-9)
    verdict["wall_speedup_x"] = round(speedup, 2)
    payload["VERDICT"] = verdict
    print(json.dumps(verdict, indent=2))
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
