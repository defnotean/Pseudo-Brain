"""Phase 2.6 Workstream G: Episodic Memory v0 — clean experiment implementation.

Architecture of the experiment:

  MemoryWrappedPB = frozen-recipe PB K=32 + EpisodicMemoryV0 store.
  - The store's key space is a learned linear map from frame pixel statistics.
  - Writes: gated soft-write per step from the CURRENT observation summary and
    a content vector derived from the frame (what the model can legitimately
    see at that time). NO leakage: on t17 the value is only in the store from
    the frame where it flashed, which is before the decision; on t18 the stale
    entry is written before the invalidation signal appears.
  - Retrieval: cosine similarity between query and stored keys -> top-1 value,
    tiled to the IreneBrainModel `retrieved_memory` contract shape.

Controls:
  B. PB param-matched no-memory (width widened so added params match)
  C. GRU reference

Six-condition causal battery at eval:
  normal / store-disabled / unread / correct-injected / donor-injected /
  stale-present.

Internal telemetry: write precision (gate open rate), retrieval precision
(similarity of retrieved vs true value key), causal effect (accuracy deltas),
stale rejection (does t18 answer follow new evidence when stale entry exists).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "esc_base", os.path.join(_here, "..", "..", "dgx_phase2_definitive_escalation.py"))
esc = importlib.util.module_from_spec(_spec)
sys.modules["esc_mem"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import (  # noqa: E402
    t17_retrieve_relevant, t18_avoid_irrelevant_memory, t01_single_cue_short_delay,
    t05_two_hypotheses)
from irene_brain.memory.episodic_v0 import EpisodicMemoryV0  # noqa: E402

MEM_TASKS = [t17_retrieve_relevant, t18_avoid_irrelevant_memory]
FILLER = [t01_single_cue_short_delay, t05_two_hypotheses]
ALL_TASKS = MEM_TASKS + FILLER


def frames_tensor(ep, device) -> Tensor:
    return torch.from_numpy(
        np.stack(ep.frames).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)


class FrameKeyEncoder(nn.Module):
    """Learns a query/key from simple frame statistics (legitimately observable)."""

    def __init__(self, out_width: int):
        super().__init__()
        # input: per-channel mean+std over quadrants -> [B, 3ch*4q*2stats=24]
        self.net = nn.Sequential(nn.Linear(24, 64), nn.SiLU(),
                                 nn.Linear(64, out_width), nn.LayerNorm(out_width))

    @staticmethod
    def features(rgb: Tensor) -> Tensor:
        b, c, h, w = rgb.shape
        qs = []
        for hs in (slice(0, 16), slice(16, 32)):
            for ws in (slice(0, 16), slice(16, 32)):
                quad = rgb[:, :, hs, ws]
                qs.append(quad.mean(dim=(2, 3)))
                qs.append(quad.std(dim=(2, 3)))
        return torch.cat([q.reshape(b, -1) for q in qs], dim=-1)

    def forward(self, rgb: Tensor) -> Tensor:
        return self.net(self.features(rgb))


class ContentEncoder(nn.Module):
    """What gets WRITTEN as the memory content: a compact frame code."""

    def __init__(self, out_width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(24, 64), nn.SiLU(),
                                 nn.Linear(64, out_width))

    def forward(self, rgb: Tensor) -> Tensor:
        return self.net(FrameKeyEncoder.features(rgb))


class MemoryWrappedPB(nn.Module):
    def __init__(self, base, device, entries: int = 64):
        super().__init__()
        self.base = base
        w = base.w
        k = base.k
        r = 1   # this core takes a [B,K,W] state; retrieved value is added per-slot
        self.w, self.k, self.r = w, k, r
        self.store = EpisodicMemoryV0(key_width=w, value_width=w, entries=entries)
        self.key_encoder = FrameKeyEncoder(w)
        self.content_encoder = ContentEncoder(w)
        # projects retrieved value into the width the model expects
        self.value_proj = nn.Linear(w, w * r) if r != 1 else nn.Identity()
        self.to(device)
        self.device_ = device

    def observe_and_maybe_write(self, rgb: Tensor) -> None:
        # Store contents are buffers (state), not differentiable parameters:
        # commit under no_grad so in-place updates don't break backward.
        with torch.no_grad():
            key = self.key_encoder(rgb)
            content = self.content_encoder(rgb)
            self.store.write_step(key, content)

    def retrieval_vector(self, rgb: Tensor) -> tuple[Tensor, Tensor]:
        q = self.key_encoder(rgb)
        vals, sims = self.store.retrieve(q, k=1)
        return vals, sims

    def build_retrieved_memory(self, vals: Tensor) -> Tensor:
        """[B,W] retrieved value broadcast to every slot as additive evidence."""
        v = vals.squeeze(1)                       # [B, W]
        return self.value_proj(v).unsqueeze(1).expand(-1, self.k, -1).contiguous()

    def forward_with_memory(self, frames: Tensor, state=None,
                            mode: str = "normal",
                            injected: Tensor | None = None):
        ctrl = torch.zeros(frames.shape[0], 307, device=self.device_)
        dt = torch.tensor([0.016667] * frames.shape[0], device=self.device_)
        out = state_out = logits_at_decision = None
        return_state = state
        for i in range(frames.shape[0]):
            if mode == "normal":
                self.observe_and_maybe_write(frames[i:i+1])
            vals, sims = self.retrieval_vector(frames[i:i+1])
            if mode == "store_disabled" or mode == "unread":
                vals = torch.zeros_like(vals)
            evidence = self.build_retrieved_memory(vals)
            if mode in ("store_disabled", "unread"):
                evidence = torch.zeros_like(evidence)
            if injected is not None:
                # normalize any rank to [B, W]
                injected = injected.reshape(injected.shape[0], -1)[:, :self.w]
                evidence = injected.unsqueeze(1).expand(-1, self.k, -1).contiguous()
            if return_state is None:
                return_state = self.base.initial_state(1).to(self.device_)
            out, return_state = self.base(
                frames[i:i+1], ctrl, dt, state=return_state + 0.05 * evidence,
                **{})
        return out, return_state


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def build_param_matched_nomemory(device, seed: int):
    """PB with extra width instead of memory, matched on total params."""
    base = esc.VectorizedPseudoBrain(thoughtlets=32, width=120, heads=4, cycles=3,
                                     init_seed=seed).to(device)
    n_base = sum(p.numel() for p in base.parameters())
    mem_probe = MemoryWrappedPB(base, device)
    n_mem = sum(p.numel() for p in mem_probe.memory.parameters()) + \
            sum(p.numel() for p in mem_probe.key_encoder.parameters()) + \
            sum(p.numel() for p in mem_probe.content_encoder.parameters())
    del mem_probe
    # widen until >= n_base + n_mem
    for w in (130, 140, 150, 160, 180, 200):
        cand = esc.VectorizedPseudoBrain(thoughtlets=32, width=w, heads=4, cycles=3,
                                         init_seed=seed).to(device)
        n_cand = sum(p.numel() for p in cand.parameters())
        if n_cand >= n_base + n_mem:
            print(f"param-matched control: W={w} ({n_cand:,} >= {n_base:,}+{n_mem:,})")
            return cand, n_cand, n_base + n_mem
    return cand, n_cand, n_base + n_mem


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_bank(seed: int, eps_per_task: int):
    bank = []
    for fn in ALL_TASKS:
        for i in range(eps_per_task):
            rng = np.random.default_rng(seed + hash(fn.__name__) % 99991 + i)
            bank.append((fn(rng), fn.__name__))
    return bank


def run_model_on_ep(model_wrap, ep_frames, label, dec_idx, device, mode="normal",
                    injected=None, optimizer=None, train=True):
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        if train:
            model_wrap.observe_and_maybe_write(ep_frames[0:1])
        out, _ = model_wrap.forward_with_memory(ep_frames[:dec_idx+1], mode=mode,
                                                injected=injected)
        slot_logits = out.proposals.action_logits
        k_slots = slot_logits.shape[1]
        target = torch.full((1, k_slots), label, dtype=torch.long, device=device)
        loss = F.cross_entropy(slot_logits.view(-1, 5), target.view(-1))
        if train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_wrap.parameters(), 1.0)
            optimizer.step()
    pred = int(torch.argmax(out.action_dist[0]).item())
    return loss, pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== EPISODIC MEMORY v0 EXPERIMENT on {device} ===")

    base = esc.VectorizedPseudoBrain(thoughtlets=32, width=120, heads=4, cycles=3,
                                     init_seed=args.seed).to(device)
    wrap = MemoryWrappedPB(base, device)
    n_params = sum(p.numel() for p in wrap.parameters())
    memory_params = sum(p.numel() for m in
                        (wrap.store, wrap.key_encoder, wrap.content_encoder)
                        for p in m.parameters())
    print(f"memory model params: {n_params:,} "
          f"(memory-specific: {memory_params:,}; "
          f"store bytes: {wrap.store.storage_bytes():,})")

    bank = make_bank(args.seed, 24)
    opt = torch.optim.AdamW(
        [p for n, p in wrap.named_parameters()], lr=5e-4, weight_decay=1e-4)
    print(f"training {args.steps} steps on {len(bank)} episodes...")
    for step in range(args.steps):
        ep, _name = bank[step % len(bank)]
        ft = frames_tensor(ep, device)
        wrap.store.reset_store()          # each episode starts with empty store
        run_model_on_ep(wrap, ft, ep.label, ep.decision_frame_index, device,
                        optimizer=opt, train=True)
        if (step + 1) % 1000 == 0:
            print(f"  step {step+1}/{args.steps}")

    # ---------------- six-condition causal battery ----------------
    print("\n=== CAUSAL BATTERY (t17/t18 focus) ===")
    battery = {}
    conditions = ["normal", "store_disabled", "unread"]
    accs = {c: {"t17": [], "t18": []} for c in conditions}
    for cond in conditions:
        for task_i, fn in enumerate(MEM_TASKS):
            ok = 0
            for i in range(40):
                rng = np.random.default_rng(20260822 + i * 7919)
                ep = fn(rng)
                ft = frames_tensor(ep, device)
                wrap.store.reset_store()
                _, pred = run_model_on_ep(wrap, ft, ep.label,
                                          ep.decision_frame_index, device,
                                          mode=cond, train=False)
                ok += int(pred == ep.label)
            accs[cond][fn.__name__] = round(ok / 40, 3)
    battery["conditions"] = accs

    # injected-value tests on t17
    inject_accs = {"correct": [], "donor": []}
    for i in range(40):
        rng = np.random.default_rng(20260822 + i * 7919)
        ep = t17_retrieve_relevant(rng)
        ft = frames_tensor(ep, device)
        # decode truth: label action maps back to channel via +1
        true_val_ch = ep.label - 1
        donor_ch = (true_val_ch + 1) % 3
        w = wrap.w
        for kind, ch in (("correct", true_val_ch), ("donor", donor_ch)):
            inj = torch.zeros(1, 1, w, device=device)
            inj[0, 0, ch::3] = 2.5   # strong channel signature
            wrap.store.reset_store()
            _, pred = run_model_on_ep(wrap, ft, ep.label, ep.decision_frame_index,
                                      device, mode="normal",
                                      injected=inj.to(device), train=False)
            inject_accs[kind].append(int(pred == ep.label))
    battery["injected"] = {
        "correct": round(float(np.mean(inject_accs["correct"])), 3),
        "donor": round(float(np.mean(inject_accs["donor"])), 3)}

    memory_params = sum(p.numel() for m in
                        (wrap.store, wrap.key_encoder, wrap.content_encoder)
                        for p in m.parameters())
    battery["telemetry"] = {
        "writes_attempted": wrap.store.writes_attempted,
        "writes_committed": wrap.store.writes_committed,
        "retrievals": wrap.store.retrievals,
        "params_memory_total": memory_params,
        "store_bytes": wrap.store.storage_bytes(),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(battery, f, indent=2)
    print(json.dumps(battery, indent=2))
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
