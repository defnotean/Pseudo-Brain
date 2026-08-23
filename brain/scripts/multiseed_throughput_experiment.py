"""Stage I: multi-seed batched training — throughput experiment.

Trains M independent VectorizedPseudoBrains SIMULTANEOUSLY by batching along
the model dimension: each owns its own slot states; weights are NOT shared —
we use grouped conv/fused-bmm style batching via torch.vmap-style ensembling
through functional_call + stacked params, OR simpler: since the core is
tiny, run M models as a batched module via functorch ensembling.

Scientific equivalence requirement: each member must produce the SAME loss
trajectory as solo training given the same seed and data order.
"""
from __future__ import annotations

import argparse
import copy
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
sys.modules["esc_mbatch"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import TASKS  # noqa: E402


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


def make_params(M, base_factory, device):
    """Stacked parameters for M independently-initialized models."""
    states = []
    for m in range(M):
        model = base_factory(m)
        states.append({k: v.detach().clone().to(device)
                       for k, v in model.state_dict().items()})
    keys = list(states[0].keys())
    stacked = {k: torch.stack([s[k] for s in states]) for k in keys}
    return stacked, keys


def batched_forward(model_template, stacked_params, rgb_b, ctrl_b, dt_b,
                    h_states):
    """Run M models on batched inputs via functional_call per-member loop.
    Even a Python loop over members amortizes kernel launches vs separate
    processes; true vectorization comes from vmap where ops permit."""
    from torch.func import functional_call, stack_module_state
    outs, new_states = [], []
    M = len(h_states)
    for m in range(M):
        params = {k: stacked_params[k][m] for k in stacked_params}
        out_m, h_m = functional_call(
            model_template, (params, {}), (rgb_b[m], ctrl_b, dt_b),
            dict(state=h_states[m]))
        outs.append(out_m.proposals.action_logits)
        new_states.append(h_m)
    return torch.stack(outs), new_states


def solo_baseline(model, opt, bank, device, steps):
    """Reference: current sequential training speed."""
    model.train()
    t0 = time.perf_counter()
    done = 0
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
        done += ep.decision_frame_index + 1
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return wall, done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    print("=== MULTI-SEED BATCHING THROUGHPUT EXPERIMENT ===")

    def factory(seed_offset):
        torch.manual_seed(42 + seed_offset * 1000)
        return esc.VectorizedPseudoBrain(thoughtlets=32, width=120,
                                         heads=4, cycles=3, init_seed=42)

    bank = build_bank(42)

    # --- baseline: one model solo ---
    model = factory(0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    wall_solo, frames_solo = solo_baseline(model, opt, bank, device, args.steps)
    print(f"SOLO: {wall_solo:.2f}s for {args.steps} steps "
          f"({wall_solo/args.steps*1000:.1f} ms/step)")

    # --- multi-seed: measure kernel-amortization potential ---
    # The honest cheap test: same model architecture, batched INPUT frames.
    # Since each frame is [1,3,32,32], running B episodes at once tests how
    # much of the 80ms is fixed launch overhead vs real compute.
    results = {"solo_wall_s": round(wall_solo, 2),
               "solo_ms_per_step": round(wall_solo/args.steps*1000, 1)}
    for B in (4, 8, 16):
        model_b = factory(0).to(device)
        opt_b = torch.optim.AdamW(model_b.parameters(), lr=5e-4, weight_decay=1e-4)
        model_b.train()
        t0 = time.perf_counter()
        steps_done = 0
        while steps_done < args.steps:
            eps = [bank[(steps_done + k) % len(bank)] for k in range(B)]
            max_len = max(e.decision_frame_index + 1 for e in eps)
            ftb = []
            for e in eps:
                ft = frames_tensor(e, device)
                padded = torch.zeros(max_len, *ft.shape[1:], device=device)
                padded[:ft.shape[0]] = ft
                ftb.append(padded)
            ftb = torch.stack(ftb)                      # [B,T,C,H,W]
            ctrl = torch.zeros(1, 307, device=device).expand(B, -1).contiguous()
            dt = torch.tensor([0.016667], device=device)
            # batch through conv+encoder by folding time into batch
            state = None
            losses = []
            for j in range(max_len):
                out_j, state = model_b(ftb[:, j], ctrl, dt, state=state)
                logits = out_j.proposals.action_logits      # [B,K,5]
                tgt = torch.tensor([eps[b].label for b in range(B)],
                                   dtype=torch.long, device=device)
                tgt = tgt.unsqueeze(1).expand(B, logits.shape[1])
                losses.append(F.cross_entropy(logits.reshape(-1, 5),
                                              tgt.reshape(-1)))
            loss = torch.stack(losses).mean()
            opt_b.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model_b.parameters(), 1.0); opt_b.step()
            steps_done += 1
        torch.cuda.synchronize()
        wall_b = time.perf_counter() - t0
        results[f"batch{B}_wall_s"] = round(wall_b, 2)
        results[f"batch{B}_ms_per_trainstep"] = round(wall_b/args.steps*1000, 1)
        results[f"batch{B}_speedup_vs_solo_per_model"] = round(
            wall_solo / wall_b, 2)
        print(f"BATCH-{B}: {wall_b:.2f}s ({wall_b/args.steps*1000:.1f} ms/step) "
              f"-> {wall_solo/wall_b:.2f}x aggregate throughput")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
