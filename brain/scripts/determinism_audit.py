"""Stage D1: determinism + scaling-stability audit (corrected banks).

Preregistered design (frozen before run), following the owner's directive:
architecture frozen; find the FIRST point where supposedly identical runs
diverge; separate execution nondeterminism from optimization instability.

Stages:
  A. BANK PIN CHECK: crc32-offset banks identical across processes
     (digest of episode labels+lengths+frame checksums).
  B. SAME-PROCESS REPEAT x3 (W480, seed 142): init digest -> first-batch
     digest -> early loss trajectory -> grad norm -> final digest -> lift.
     Identical => training is bitwise-reproducible in-process.
  C. DETERMINISTIC-MODE REPEAT: torch.use_deterministic_algorithms(True),
     CUBLAS_WORKSPACE_CONFIG=:4096:8, no TF32; same protocol as B.
  D. CROSS-PROCESS REPEAT: two independent container processes, W480 s142,
     normal mode — does divergence reappear when banks are pinned?
  E. CORRECTED VARIANCE READ: W120/W240/W480 x 4 seeds {42,142,242,342},
     pinned banks, one process — the first honest seed-variance curve.

Digests: float64 sum + L2 norm of all params (order-stable); loss
trajectory recorded at steps {0, 10, 50, 100, 500, 1000}.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import zlib

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "esc_d1", os.path.join(_here, "..", "..", "dgx_phase2_definitive_escalation.py"))
esc = importlib.util.module_from_spec(_spec)
sys.modules["esc_d1"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import TASKS  # noqa: E402


def task_offset(fn) -> int:
    """STABLE across processes (the salted-hash fix)."""
    return zlib.crc32(fn.__name__.encode()) % 99991


def frames_tensor(ep, device):
    return torch.from_numpy(np.stack(ep.frames).astype(np.float32) / 255.0) \
        .permute(0, 3, 1, 2).to(device)


def build_bank(seed, eps_per_task=24):
    bank = []
    for fn in TASKS:
        for i in range(eps_per_task):
            r = np.random.default_rng(seed + task_offset(fn) + i)
            bank.append(fn(r))
    return bank


def bank_digest(bank) -> str:
    h = hashlib.sha256()
    for ep in bank:
        h.update(str(ep.label).encode())
        h.update(str(ep.decision_frame_index).encode())
        fr = np.stack(ep.frames)
        h.update(np.asarray(fr, dtype=np.uint8).tobytes())
    return h.hexdigest()[:16]


def param_digest(model):
    """Order-stable float64 stats over all params."""
    s = 0.0
    sq = 0.0
    n = 0
    with torch.no_grad():
        for p in model.parameters():
            pd = p.detach().to(torch.float64)
            s += float(pd.sum())
            sq += float((pd ** 2).sum())
            n += p.numel()
    return {"sum": round(s, 6), "l2": round(sq ** 0.5, 6), "n": n}


def train_with_probes(model, device, steps, seed, B=16, probe_steps=None,
                      collect_grads=False):
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
            group = members[k:k + B]
            while len(group) < B:
                group.append(members[(k + len(group)) % len(members)])
            order.append((ln, group))
    probe_steps = probe_steps or {0, 10, 50, 100, 500, 1000}
    losses, grads = {}, {}
    step = 0
    first_batch_digest = None
    while step < steps:
        for ln, group in order:
            if step >= steps:
                break
            eps = [bank[i] for i in group]
            ftb = torch.stack([
                torch.stack([frames_tensor(e, device)[j] for j in range(ln)])
                for e in eps])
            if step == 0:
                h = hashlib.sha256()
                h.update(ftb.cpu().numpy().tobytes())
                first_batch_digest = h.hexdigest()[:16]
            ctrl = torch.zeros(len(eps), 307, device=device)
            dt = torch.tensor([0.016667], device=device)
            state = None
            for j in range(ln):
                out_j, state = model(ftb[:, j], ctrl, dt, state=state)
            logits = out_j.proposals.action_logits
            tgt = torch.tensor([e.label for e in eps], dtype=torch.long,
                               device=device).unsqueeze(1).expand(-1, logits.shape[1])
            loss = F.cross_entropy(logits.reshape(-1, 5), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            if collect_grads:
                gsq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        gsq += float(p.grad.to(torch.float64).pow(2).sum())
                grads[step] = round(gsq ** 0.5, 6)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step in probe_steps:
                losses[step] = round(float(loss.item()), 6)
            step += 1
    return {"losses": losses, "grad_norms": grads,
            "first_batch_digest": first_batch_digest}


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
                out, state = model(ft[j:j + 1], ctrl, dt, state=state)
                act = int(torch.argmax(out.action_dist[0]).item())
            ok += int(act == ep.label)
        lifts.append(ok / episodes - chance)
    model.train()
    return float(np.mean(lifts))


def make_model(w, seed, device):
    return esc.VectorizedPseudoBrain(
        thoughtlets=32, width=w, heads=4, cycles=3, init_seed=seed).to(device)


def full_run(w, seed, device, steps, deterministic=False):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.cuda.manual_seed_all(seed)
    model = make_model(w, seed, device)
    rec = {}
    rec["init"] = param_digest(model)
    out = train_with_probes(model, device, steps, seed, collect_grads=True)
    rec.update(out)
    rec["final"] = param_digest(model)
    rec["lift"] = round(eval_lift(model, device), 4)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--stage", required=True,
                    choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    payload = {"stage": args.stage, "steps": args.steps}

    if args.stage == "A":
        b1 = build_bank(42)
        payload["bank_digest_s42"] = bank_digest(b1)
        print("BANK s42:", payload["bank_digest_s42"])

    elif args.stage in ("B", "C"):
        if args.stage == "C":
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        runs = []
        for rep in range(3):
            r = full_run(480, 142, device, args.steps)
            runs.append(r)
            print(f"[{args.stage}] rep{rep}: lift={r['lift']} "
                  f"init={r['init']['l2']} final={r['final']['l2']} "
                  f"fb={r['first_batch_digest']} "
                  f"L@100={r['losses'].get(100)}")
        payload["runs"] = runs
        payload["identical_final"] = len({json.dumps(r["final"], sort_keys=True)
                                          for r in runs}) == 1
        payload["identical_lift"] = len({r["lift"] for r in runs}) == 1

    elif args.stage == "E":
        for w in (120, 240, 480):
            cfg = {}
            for seed in (42, 142, 242, 342):
                r = full_run(w, seed, device, args.steps)
                cfg[str(seed)] = r["lift"]
                print(f"E W{w} s{seed}: lift={r['lift']}")
            lifts = list(cfg.values())
            payload[f"W{w}"] = {
                "seeds": cfg,
                "mean": round(float(np.mean(lifts)), 4),
                "sigma": round(float(np.std(lifts)), 4)}
            print(f"E W{w} SUMMARY: mean={payload[f'W{w}']['mean']} "
                  f"±{payload[f'W{w}']['sigma']}")

    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
