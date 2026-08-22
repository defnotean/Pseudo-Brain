"""Phase 2.6 Workstream C/D: Dynamic-K / lifecycle v0.

Minimal lifecycle mechanism, permutation-safe by construction:

  - A learned activity gate g_k in [0,1] per slot, computed FROM the slot's own
    state (content-addressed, not index-addressed).
  - Active slots: full update + full vote weight in aggregation.
  - Sleeping slots: state decays toward their identity code; vote weight ~0.
  - A load-balancing pressure (differentiable) prevents total collapse to zero
    active slots without hardcoding a target count.

Experiment: does dynamic-K improve on fixed K=32 on
  (a) torture multitask sample efficiency (PB's known weakness),
  (b) collapse resistance on the escalation task (Phase 2's strength),
without hurting either?

Arms (all same recipe/steps/seeds):
  D1: PB K=32 fixed (reference = locked baseline recipe)
  D2: PB K=32 + dynamic gate (candidate)
Plus telemetry: active-K distribution over training, per-task active-K.
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
sys.modules["esc_dk"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import TASKS  # noqa: E402


class DynamicGate(nn.Module):
    """Content-addressed per-slot activity gate with load balancing."""

    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width // 2), nn.SiLU(),
            nn.Linear(width // 2, 1))

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """h: [B,K,W]. Returns (gate [B,K,1], aux_loss)."""
        logits = self.net(h)                       # [B,K,1]
        gate = torch.sigmoid(logits)
        # Load-balance: mean gate should not collapse to 0. Penalize deviation
        # from a soft floor via hinge on batch-mean.
        mean_gate = gate.mean()
        aux = torch.relu(0.15 - mean_gate).square()   # keep >= ~15% awake
        return gate, aux


class DynamicWrappedPB(nn.Module):
    """Wraps VectorizedPseudoBrain with a lifecycle gate on slot updates."""

    def __init__(self, base, device):
        super().__init__()
        self.base = base
        self.gate = DynamicGate(base.w)
        self.to(device)
        self.device_ = device
        self.active_k_history: list[float] = []

    def forward(self, rgb, ctrl, dt, state=None, **kw):
        b = rgb.shape[0]
        if state is None:
            state = self.base.initial_state(b).to(self.device_)
        sens = self.base.conv(rgb)
        sens_expanded = sens.unsqueeze(1).expand(b, self.base.k, self.base.w)
        gate, aux = self.gate(state)
        self.active_k_history.append(float((gate > 0.5).float().sum().item()) / b)
        h_flat = state.reshape(b * self.base.k, self.base.w)
        # sleeping slots decay toward identity codes instead of full update
        identities = self.base.slot_identities.expand(b, -1, -1).reshape(
            b * self.base.k, self.base.w)
        gru_in = torch.cat((sens_expanded, state), dim=-1).reshape(
            b * self.base.k, self.base.w * 2)
        h_next = self.base.slot_gru(gru_in, h_flat).reshape(b, self.base.k, self.base.w)
        h_next = self.base.slot_norm(h_next)
        h_mixed = gate * h_next + (1 - gate) * (
            0.9 * state + 0.1 * identities.view_as(state))
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
            gate, aux2 = self.gate(h_mixed)
            aux = aux + aux2
        out = self.base.actuator(sensors=sens.unsqueeze(1), thoughts=h_mixed,
                                 **{kk: vv for kk, vv in kw.items()})
        self.last_aux = aux
        return out, h_mixed


def frames_tensor(ep, device):
    return torch.from_numpy(
        np.stack(ep.frames).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)


def train_arm(wrap, device, tasks, steps: int, seed: int, eps_per_task: int = 24):
    rng = np.random.default_rng(seed)
    bank = []
    for fn in tasks:
        for i in range(eps_per_task):
            r = np.random.default_rng(seed + hash(fn.__name__) % 99991 + i)
            bank.append(fn(r))
    opt = torch.optim.AdamW(wrap.parameters(), lr=5e-4, weight_decay=1e-4)
    wrap.train()
    for step in range(steps):
        ep = bank[step % len(bank)]
        ft = frames_tensor(ep, device)
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=ft.device)
        state = None
        for i in range(ep.decision_frame_index + 1):
            out, state = wrap(ft[i:i+1], ctrl, dt, state=state)
        slot_logits = out.proposals.action_logits
        target = torch.full((1, slot_logits.shape[1]), ep.label,
                            dtype=torch.long, device=device)
        loss = F.cross_entropy(slot_logits.view(-1, 5), target.view(-1))
        if hasattr(wrap, "last_aux"):
            loss = loss + 0.1 * wrap.last_aux
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(wrap.parameters(), 1.0); opt.step()
        if (step+1) % 2000 == 0: print(f"  step {step+1}/{steps} loss={loss.item():.4f}")


@torch.no_grad()
def eval_torture(wrap, device, episodes=30, base_seed=20260822):
    wrap.eval()
    results = {}
    for fn in TASKS:
        ok = waits = 0
        for i in range(episodes):
            r = np.random.default_rng(base_seed + i * 7919)
            ep = fn(r)
            ft = frames_tensor(ep, device)
            ctrl = torch.zeros(1, 307, device=device)
            dt = torch.tensor([0.016667], device=device)
            state = None; act = 0
            for j in range(ep.decision_frame_index + 1):
                out, state = wrap(ft[j:j+1], ctrl, dt, state=state)
                act = int(torch.argmax(out.action_dist[0]).item())
            waits += int(act == 0)
            if ep.label != 0 and act == ep.label: ok += 1
        results[fn.__name__] = {"acc": round(ok/episodes, 3),
                                "chance": round(ep.chance, 4)}
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== DYNAMIC-K v0 EXPERIMENT on {device}, steps={args.steps} ===")

    arms = {}
    base_fixed = esc.VectorizedPseudoBrain(thoughtlets=32, width=120, heads=4,
                                           cycles=3, init_seed=args.seed).to(device)
    arms["fixed_K32"] = base_fixed
    dyn_base = esc.VectorizedPseudoBrain(thoughtlets=32, width=120, heads=4,
                                         cycles=3, init_seed=args.seed).to(device)
    arms["dynamic_gate"] = DynamicWrappedPB(dyn_base, device)

    payload = {}
    for name, wrap in arms.items():
        print(f"\n--- training arm {name} ---")
        train_arm(wrap, device, TASKS, args.steps, args.seed)
        res = eval_torture(wrap, device)
        lift = float(np.mean([v["acc"] - v["chance"] for v in res.values()]))
        n_above = sum(1 for v in res.values() if v["acc"] - v["chance"] > 0.02)
        payload[name] = {"per_task": res, "mean_lift": round(lift, 4),
                         "tasks_above": n_above}
        if isinstance(wrap, DynamicWrappedPB):
            ak = wrap.active_k_history[-500:]
            payload[name]["active_K_last500"] = round(float(np.mean(ak)), 2)
            print(f"  active-K (last 500 fwd): {np.mean(ak):.2f}")
        print(f"  {name}: mean_lift={lift:+.4f} ({n_above}/25 tasks)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
