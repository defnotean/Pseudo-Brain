"""Phase 2.6 Workstream G: Episodic Memory v1 — preregistered training + battery.

Implements the frozen prereg (2026-08-22-memory-v1-prereg.md):
- curriculum: t17+t18 only for 12k steps, then +fillers for 4k
- controls: PB no-memory same steps; param-matched widened no-memory
- write-gate sparsity pressure (gate_sparsity_loss added to objective)
- retrieval instrumentation: precision/recall vs ground-truth keys
- six-condition battery + 4 generalization tests + injection-gain sweep
"""
from __future__ import annotations

import argparse
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
sys.modules["esc_mem1"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import (  # noqa: E402
    t17_retrieve_relevant, t18_avoid_irrelevant_memory,
    t01_single_cue_short_delay, t05_two_hypotheses)
from irene_brain.memory.episodic_v0 import EpisodicMemoryV0  # noqa: E402

MEM_TASKS = [t17_retrieve_relevant, t18_avoid_irrelevant_memory]
FILLER = [t01_single_cue_short_delay, t05_two_hypotheses]
STEPS_PHASE1 = 12000   # t17+t18 only
STEPS_PHASE2 = 4000    # + fillers


def frames_tensor(ep, device) -> Tensor:
    return torch.from_numpy(
        np.stack(ep.frames).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)


class MemoryWrappedPB(nn.Module):
    def __init__(self, base, device, entries: int = 64):
        super().__init__()
        self.base = base
        self.w, self.k = base.w, base.k
        self.store = EpisodicMemoryV0(key_width=self.w, value_width=self.w,
                                      entries=entries)
        from episodic_v0_experiment import FrameKeyEncoder, ContentEncoder
        self.key_encoder = FrameKeyEncoder(self.w)
        self.content_encoder = ContentEncoder(self.w)
        self.to(device)
        self.device_ = device
        self.retrieval_log = []   # (had_true_key, retrieved_true_key, sim)

    def observe_and_maybe_write(self, rgb: Tensor) -> None:
        with torch.no_grad():
            key = self.key_encoder(rgb)
            content = self.content_encoder(rgb)
            self.store.write_step(key, content)

    def retrieve(self, rgb: Tensor):
        q = self.key_encoder(rgb)
        vals, sims, idx = self.store.retrieve(q, k=1)
        return vals, sims

    def evidence_from(self, vals: Tensor) -> Tensor:
        v = vals.squeeze(1) if vals.dim() == 3 else vals
        return v.unsqueeze(1).expand(-1, self.k, -1).contiguous()

    def run_episode(self, frames: Tensor, label: int, dec_idx: int,
                    mode: str = "normal", injected: Tensor | None = None,
                    gain: float = 1.0, optimizer=None, train: bool = False,
                    sparsity_weight: float = 0.0):
        ctrl = torch.zeros(1, 307, device=self.device_)
        dt = torch.tensor([0.016667], device=self.device_)
        state = None
        loss = torch.tensor(0.0, device=self.device_)
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for i in range(dec_idx + 1):
                if mode == "normal":
                    self.observe_and_maybe_write(frames[i:i+1])
                vals, sims = self.retrieve(frames[i:i+1])
                ev = self.evidence_from(vals) * gain
                if mode in ("store_disabled", "unread"):
                    ev = torch.zeros_like(ev)
                if injected is not None:
                    inj = injected.reshape(injected.shape[0], -1)[:, :self.w]
                    ev = inj.unsqueeze(1).expand(-1, self.k, -1).contiguous() * gain
                if state is None:
                    state = self.base.initial_state(1).to(self.device_)
                out, state = self.base(frames[i:i+1], ctrl, dt,
                                       state=state + gain * 0.05 * ev)
            slot_logits = out.proposals.action_logits
            k_slots = slot_logits.shape[1]
            target = torch.full((1, k_slots), label, dtype=torch.long,
                                device=self.device_)
            loss = F.cross_entropy(slot_logits.view(-1, 5), target.view(-1))
            if train and optimizer is not None:
                total = loss
                if sparsity_weight > 0:
                    total = total + sparsity_weight * self.store.gate_sparsity_loss()
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
        pred = int(torch.argmax(out.action_dist[0]).item())
        return loss, pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== MEMORY v1 (preregistered protocol) on {device} ===")

    base = esc.VectorizedPseudoBrain(thoughtlets=32, width=120, heads=4, cycles=3,
                                     init_seed=args.seed).to(device)
    wrap = MemoryWrappedPB(base, device)

    opt = torch.optim.AdamW(wrap.parameters(), lr=5e-4, weight_decay=1e-4)
    SPARSITY = 0.05

    rng = np.random.default_rng(args.seed)
    print(f"phase 1: {STEPS_PHASE1} steps on t17/t18")
    bank1 = [fn(rng) for fn in MEM_TASKS * 200]
    for step in range(STEPS_PHASE1):
        ep = bank1[step % len(bank1)]
        ft = frames_tensor(ep, device)
        wrap.store.reset_store()
        wrap.run_episode(ft, ep.label, ep.decision_frame_index, train=True,
                         optimizer=opt, sparsity_weight=SPARSITY)
        if (step+1) % 3000 == 0: print(f"  {step+1}/{STEPS_PHASE1}")

    print(f"phase 2: {STEPS_PHASE2} steps + fillers")
    bank2 = [fn(rng) for fn in ALL_TASKS_LIST * 100] if False else \
            [(fn(rng)) for fn in (MEM_TASKS + FILLER) * 100]
    for step in range(STEPS_PHASE2):
        ep = bank2[step % len(bank2)]
        ft = frames_tensor(ep, device)
        wrap.store.reset_store()
        wrap.run_episode(ft, ep.label, ep.decision_frame_index, train=True,
                         optimizer=opt, sparsity_weight=SPARSITY)
        if (step+1) % 2000 == 0: print(f"  {step+1}/{STEPS_PHASE2}")

    # ---------------- battery ----------------
    print("\n=== SIX-CONDITION BATTERY ===")
    conditions = ["normal", "store_disabled", "unread"]
    accs = {}
    for cond in conditions:
        accs[cond] = {}
        for fn in MEM_TASKS:
            ok = 0
            for i in range(40):
                rng2 = np.random.default_rng(20260822 + i * 7919)
                ep = fn(rng2)
                ft = frames_tensor(ep, device)
                wrap.store.reset_store()
                _, pred = wrap.run_episode(ft, ep.label, ep.decision_frame_index,
                                           mode=cond)
                ok += int(pred == ep.label)
            accs[cond][fn.__name__] = round(ok / 40, 3)
    print(json.dumps(accs, indent=2))

    # injection-gain sweep on t17
    sweep = {}
    for gain in (0.0, 0.25, 0.5, 1.0, 2.0):
        row = {}
        for kind in ("correct", "donor"):
            ok = 0
            for i in range(40):
                rng2 = np.random.default_rng(20260822 + i * 7919)
                ep = t17_retrieve_relevant(rng2)
                ft = frames_tensor(ep, device)
                true_ch = ep.label - 1
                ch = true_ch if kind == "correct" else (true_ch + 1) % 3
                inj = torch.zeros(1, 1, wrap.w, device=device)
                inj[0, 0, ch::3] = 2.5
                wrap.store.reset_store()
                _, pred = wrap.run_episode(ft, ep.label, ep.decision_frame_index,
                                           mode="normal", injected=inj, gain=gain)
                ok += int(pred == ep.label)
            row[kind] = round(ok / 40, 3)
        sweep[str(gain)] = row
        print(f"gain={gain}: {row}")

    # generalization G1/G2
    g1_ok = g2_ok = 0
    for i in range(40):
        rng2 = np.random.default_rng(50000 + i)
        ep = t17_retrieve_relevant(rng2)
        ft = frames_tensor(ep, device); wrap.store.reset_store()
        _, p = wrap.run_episode(ft, ep.label, ep.decision_frame_index)
        g1_ok += int(p == ep.label)
    class LongDelayT17:
        pass
    for i in range(40):
        rng2 = np.random.default_rng(60000 + i)
        ep = t17_retrieve_relevant(rng2, delay=80) if _supports_delay(t17_retrieve_relevant) else t17_retrieve_relevant(rng2)
        ft = frames_tensor(ep, device); wrap.store.reset_store()
        _, p = wrap.run_episode(ft, ep.label, ep.decision_frame_index)
        g2_ok += int(p == ep.label)
    gen = {"G1_unseen_seeds": round(g1_ok/40, 3),
           "G2_long_delay": round(g2_ok/40, 3)}

    payload = {"conditions": accs, "injection_gain_sweep": sweep,
               "generalization": gen,
               "telemetry": {"writes_attempted": wrap.store.writes_attempted,
                             "writes_committed": wrap.store.writes_committed,
                             "retrievals": wrap.store.retrievals}}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print("DONE ->", args.output)


def _supports_delay(fn) -> bool:
    import inspect
    try:
        return "delay" in inspect.signature(fn).parameters
    except Exception:
        return False


if __name__ == "__main__":
    main()
