"""Phase 2.6 Dynamic-K v1: isolation + instrumentation + closure attempt.

Four arms (same recipe, seeds, steps):
  A1 fixed        : VectorizedPseudoBrain, no gate (reference)
  A2 soft_modulate: gate present, NO sparsity pressure (modulation-only control)
  A3 dynamic      : gate + modest compute cost + temperature annealing
                    (real chance to close)

Heavy gate telemetry per arm/task; escalation phase-binned effective-K at eval.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
sys.modules["esc_dk1"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import TASKS  # noqa: E402


class AnnealedGate(nn.Module):
    """Content-addressed slot gate with annealable temperature and optional
    compute-cost pressure. Cost is a MEAN penalty (not target-K): the network
    pays for wakefulness and must earn it via performance."""

    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width // 2), nn.SiLU(),
                                 nn.Linear(width // 2, 1))
        self.tau = 1.0          # annealed 2.0 -> 0.5 over training
        self.cost_weight = 0.0  # annealed 0 -> 0.03

    def set_schedule(self, frac: float):
        """frac in [0,1] progress through training."""
        self.tau = 2.0 - 1.5 * frac          # 2.0 .. 0.5
        self.cost_weight = 0.03 * frac       # ramp cost in slowly

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.net(h)
        if self.training:
            # straight-through-ish soft gate with temperature
            gate = torch.sigmoid(logits / self.tau)
        else:
            gate = torch.sigmoid(logits / self.tau)
        mean_gate = gate.mean()
        aux = self.cost_weight * mean_gate   # linear compute cost
        return gate, aux

    def stats(self, gate: Tensor) -> dict:
        g = gate.detach().flatten()
        return {"mean": round(float(g.mean()), 4),
                "min": round(float(g.min()), 4),
                "max": round(float(g.max()), 4),
                "p10": round(float(torch.quantile(g, 0.10)), 4),
                "p50": round(float(torch.quantile(g, 0.50)), 4),
                "p90": round(float(torch.quantile(g, 0.90)), 4),
                "soft_K": round(float(g.sum()), 2),
                "n>0.1": int((g > 0.1).sum()),
                "n>0.5": int((g > 0.5).sum()),
                "n>0.9": int((g > 0.9).sum())}


class GatedPB(nn.Module):
    def __init__(self, base, device, gated: bool = True):
        super().__init__()
        self.base = base
        self.gate = AnnealedGate(base.w) if gated else None
        self.to(device); self.device_ = device
        self.last_stats = None

    def forward(self, rgb, ctrl, dt, state=None):
        b = rgb.shape[0]
        if state is None:
            state = self.base.initial_state(b).to(self.device_)
        sens = self.base.conv(rgb)
        sens_expanded = sens.unsqueeze(1).expand(b, self.base.k, self.base.w)
        h_flat = state.reshape(b * self.base.k, self.base.w)
        gru_in = torch.cat((sens_expanded, state), dim=-1).reshape(
            b * self.base.k, self.base.w * 2)
        h_next = self.base.slot_gru(gru_in, h_flat).reshape(b, self.base.k, self.base.w)
        h_next = self.base.slot_norm(h_next)
        h_mixed = h_next
        aux = torch.zeros((), device=self.device_)
        for c in range(1, self.base.cycles):
            q = self.base.q_proj(h_mixed).reshape(b, self.base.k, self.base.heads,
                                                  self.base.w // self.base.heads).transpose(1, 2)
            k_ = self.base.k_proj(h_mixed).reshape(b, self.base.k, self.base.heads,
                                                   self.base.w // self.base.heads).transpose(1, 2)
            v = self.base.v_proj(h_mixed).reshape(b, self.base.k, self.base.heads,
                                                  self.base.w // self.base.heads).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k_, v)
            attn = attn.transpose(1, 2).reshape(b, self.base.k, self.base.w)
            h_mixed = self.base.attn_norm(h_mixed + self.base.attn_out(attn))
            if self.gate is not None:
                gate, a = self.gate(h_mixed)
                aux = aux + a
                h_mixed = gate * h_mixed + (1 - gate) * (0.9 * state + 0.1 * state)
                if not self.training:
                    self.last_stats = self.gate.stats(gate)
        out = self.base.actuator(sensors=sens.unsqueeze(1), thoughts=h_mixed)
        return out, h_mixed, aux


def frames_tensor(ep, device):
    return torch.from_numpy(np.stack(ep.frames).astype(np.float32) / 255.0)\
        .permute(0, 3, 1, 2).to(device)


def train_arm(wrap, device, steps, seed, eps_per_task=24):
    rng = np.random.default_rng(seed)
    bank = []
    from irene_brain.evaluation.torture_suite import TASKS as T
    for fn in T:
        for i in range(eps_per_task):
            r = np.random.default_rng(seed + hash(fn.__name__) % 99991 + i)
            bank.append(fn(r))
    opt = torch.optim.AdamW(wrap.parameters(), lr=5e-4, weight_decay=1e-4)
    wrap.train()
    for step in range(steps):
        if wrap.gate is not None:
            wrap.gate.set_schedule(step / max(1, steps - 1))
        ep = bank[step % len(bank)]
        ft = frames_tensor(ep, device)
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        state = None
        loss_sum = 0.0
        for i in range(ep.decision_frame_index + 1):
            out, state, aux = wrap(ft[i:i+1], ctrl, dt, state=state)
        slot_logits = out.proposals.action_logits
        target = torch.full((1, slot_logits.shape[1]), ep.label,
                            dtype=torch.long, device=device)
        loss = F.cross_entropy(slot_logits.view(-1, 5), target.view(-1)) + 0.1 * aux
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(wrap.parameters(), 1.0); opt.step()
        if (step+1) % 2000 == 0:
            print(f"  step {step+1}/{steps} loss={loss.item():.4f}")


@torch.no_grad()
def eval_torture(wrap, device, episodes=30, base_seed=20260822):
    wrap.eval()
    results = {}
    for fn in TASKS:
        ok = 0; stat_acc = []
        for i in range(episodes):
            r = np.random.default_rng(base_seed + i * 7919)
            ep = fn(r)
            ft = frames_tensor(ep, device)
            ctrl = torch.zeros(1, 307, device=device)
            dt = torch.tensor([0.016667], device=device)
            state = None; act = 0
            for j in range(ep.decision_frame_index + 1):
                out, state, _aux = wrap(ft[j:j+1], ctrl, dt, state=state)
                act = int(torch.argmax(out.action_dist[0]).item())
                if wrap.gate is not None and wrap.last_stats:
                    stat_acc.append(wrap.last_stats["soft_K"])
            ok += int(act == ep.label)
        entry = {"acc": round(ok/episodes, 3),
                 "chance": round(float(getattr(ep, "chance", 0.0)), 4)}
        if stat_acc:
            entry["eff_K"] = round(float(np.mean(stat_acc)), 2)
        results[fn.__name__] = entry
    return results


@torch.no_grad()
def eval_escalation_phases(model_wrap, device, episodes=40):
    """Phase-binned effective-K on the escalation task WITHOUT training here —
    uses the model as-is (post torture-training). Phases: S0 8-world, S1 4,
    S2 2, S3 resolved."""
    wrap = model_wrap
    wrap.eval()
    bins = {0: [], 15: [], 30: [], 45: []}
    for ep_i in range(episodes):
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        state = None
        haz_a = bool((ep_i % 2 == 0)); haz_b = bool(((ep_i//2) % 2 == 0))
        rgb = torch.randn(1, 3, 32, 32, device=device) * 0.1 + 0.5
        for t in range(60):
            if t == 15:
                if haz_a: rgb[:, 0, :16, :16] += 0.8
                else: rgb[:, 0, 16:] += 0.8
            elif t == 30:
                if haz_b: rgb[:, 1, :16, :16] += 0.8
                else: rgb[:, 1, 16:] += 0.8
            elif t == 45:
                rgb[:, 2, :, :] += 0.8
            out, state, _a = wrap(rgb, ctrl, dt, state=state)
            if t in bins and wrap.gate is not None and wrap.last_stats:
                bins[t].append(wrap.last_stats["soft_K"])
    return {"S0_8worlds": round(float(np.mean(bins[0])), 2),
            "S1_4worlds": round(float(np.mean(bins[15])), 2),
            "S2_2worlds": round(float(np.mean(bins[30])), 2),
            "S3_resolved": round(float(np.mean(bins[45])), 2)} if wrap.gate else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== DYNAMIC-K v1 (isolation + instrumentation) on {device} ===")

    arms = {}
    b1 = esc.VectorizedPseudoBrain(thoughtlets=32, width=120, heads=4, cycles=3,
                                   init_seed=args.seed).to(device)
    arms["fixed"] = GatedPB(b1, device, gated=False)
    b2 = esc.VectorizedPseudoBrain(thoughtlets=32, width=120, heads=4, cycles=3,
                                   init_seed=args.seed).to(device)
    arms["soft_modulate"] = GatedPB(b2, device, gated=True)
    arms["soft_modulate"].gate.cost_weight = 0.0   # control: no pressure
    b3 = esc.VectorizedPseudoBrain(thoughtlets=32, width=120, heads=4, cycles=3,
                                   init_seed=args.seed).to(device)
    arms["dynamic"] = GatedPB(b3, device, gated=True)  # scheduled cost+tau

    payload = {}
    for name, wrap in arms.items():
        print(f"\n--- arm {name} ---")
        train_arm(wrap, device, args.steps, args.seed)
        res = eval_torture(wrap, device)
        lift = float(np.mean([v["acc"] - v.get("chance", 0.0) for v in res.values()]))
        above = sum(1 for v in res.values() if v["acc"] - v.get("chance", 0.0) > 0.02)
        entry = {"per_task": res, "mean_lift": round(lift, 4),
                 "tasks_above": above}
        ph = eval_escalation_phases(wrap, device)
        if ph: entry["escalation_phase_effK"] = ph
        # latency probe
        import time
        wrap.eval()
        rgb = torch.randn(1, 3, 32, 32, device=device)
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        st = None
        for _ in range(10): out, st, _ = wrap(rgb, ctrl, dt, state=st)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(200): out, st, _ = wrap(rgb, ctrl, dt, state=st)
        torch.cuda.synchronize()
        entry["latency_ms_per_frame_p50approx"] = \
            round((time.perf_counter() - t0) / 200 * 1000, 4)
        payload[name] = entry
        print(f"{name}: lift={lift:+.4f} ({above}/25) latency={entry['latency_ms_per_frame_p50approx']}ms")
        if ph: print(f"  phase effK: {ph}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
