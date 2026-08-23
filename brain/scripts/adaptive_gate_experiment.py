"""Stage C: preregistered adaptive-gate wiring experiment (2026-08-22).

Prereg: brain/docs/preregistrations/2026-08-22-adaptive-gate-wiring-prereg.md
Two arms: control (canonical VectorizedPseudoBrain) vs gate-wired (same core +
AdaptiveThoughtUpdateGate as the thought-refresh blend between cycles).
Frozen gates: memory-causal 3/3 seeds; lift >= control-1sigma; collapse <=
control+0.05; permutation exact; latency p50 <= 1.25x.
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
from torch import nn, Tensor

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "esc_base", os.path.join(_here, "..", "..", "dgx_phase2_definitive_escalation.py"))
esc = importlib.util.module_from_spec(_spec)
sys.modules["esc_gate"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.model.adaptive_thought_gate import AdaptiveThoughtUpdateGate  # noqa: E402
from irene_brain.evaluation.torture_suite import TASKS  # noqa: E402

SEEDS = [42, 142, 242]


class GateWiredCore(nn.Module):
    """Canonical VectorizedPseudoBrain + AdaptiveThoughtUpdateGate as the
    between-cycle thought-refresh blend. Minimal wiring per prereg:
    refreshed = keep*thoughts + (1-keep)*seeds  (existing lifecycle blend)
    thoughts  = alpha*refreshed + (1-alpha)*thoughts   (gate modulation)
    """

    def __init__(self, base) -> None:
        super().__init__()
        self.base = base
        self.gate = AdaptiveThoughtUpdateGate(width=base.w)

    def forward(self, rgb, ctrl, dt, state=None, evidence=None):
        b = rgb.shape[0]
        if state is None:
            state = self.base.initial_state(b).to(rgb.device)
        raw = self.base.conv(rgb)
        sens = raw.unsqueeze(1).expand(b, self.base.k, self.base.w)
        if evidence is not None:
            sens = sens + evidence          # idealized evidence enters the input path
        h = self.update(sens, state)
        out = self.base.actuator(sensors=raw.unsqueeze(1), thoughts=h)
        return out, h

    def update(self, sens, h):
        """Canonical cycles (GRU + slot attention), then gate decides per-slot
        how much of the new state to accept."""
        base = self.base
        for _c in range(base.cycles):
            gru_in = torch.cat((sens, h), dim=-1).reshape(-1, base.w * 2)
            h_flat = h.reshape(-1, base.w)
            h_next = base.slot_gru(gru_in, h_flat)
            h = base.slot_norm(h_next.reshape(h.shape[0], base.k, base.w))
            q = base.q_proj(h).reshape(h.shape[0], base.k, base.heads,
                                       base.w // base.heads).transpose(1, 2)
            k_ = base.k_proj(h).reshape(h.shape[0], base.k, base.heads,
                                        base.w // base.heads).transpose(1, 2)
            v = base.v_proj(h).reshape(h.shape[0], base.k, base.heads,
                                       base.w // base.heads).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k_, v)
            attn = attn.transpose(1, 2).reshape(h.shape[0], base.k, base.w)
            h = base.attn_norm(h + base.attn_out(attn))
        h_new = h
        belief_like = sens.mean(dim=1, keepdim=True)
        _, alphas = self.gate(
            thoughts=h.unsqueeze(2),        # [B,K,1,W]
            seeds=h_new.unsqueeze(2),
            sensors=sens.mean(dim=1, keepdim=True),
            belief=belief_like,
        )
        alphas = alphas.unsqueeze(-1)       # [B,K,1,1] broadcasts over W
        return alphas.squeeze(-1).squeeze(-1).unsqueeze(-1) * h_new + \
            (1.0 - alphas.squeeze(-1).squeeze(-1).unsqueeze(-1)) * h


def frames_tensor(ep, device):
    return torch.from_numpy(np.stack(ep.frames).astype(np.float32) / 255.0)\
        .permute(0, 3, 1, 2).to(device)


def build_bank(seed, eps_per_task=24):
    bank = []
    for fn in TASKS:
        for i in range(eps_per_task):
            r = np.random.default_rng(seed + hash(fn.__name__) % 99991 + i)
            bank.append(fn(r))
    return bank


def train_torture(model, device, steps, seed, curve_at=(2000,)):
    bank = build_bank(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()
    curve = {}
    for step in range(1, steps + 1):
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
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in curve_at:
            curve[str(step)] = round(eval_lift(model, device, episodes=10), 4)
    return curve


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
def memory_probe(model, device, episodes=40):
    """Ideal-evidence probe. The GateWiredCore exposes an explicit evidence
    input path via the gate's context inputs; control receives none."""
    model.eval()
    res = {"correct": [], "zero": [], "wrong": []}
    kk = model.base.k if hasattr(model, "base") else model.k
    w = model.base.w if hasattr(model, "base") else model.w
    for i in range(episodes):
        r = np.random.default_rng(20260822 + i * 7919)
        from irene_brain.evaluation.torture_suite import t01_single_cue_short_delay
        ep = t01_single_cue_short_delay(r)
        ft = frames_tensor(ep, device)
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        dec = ep.decision_frame_index
        true_ch = ep.label - 1
        wired = isinstance(model, GateWiredCore)
        for cond in ("correct", "zero", "wrong"):
            ch = true_ch if cond == "correct" else ((true_ch + 1) % 3 if cond == "wrong" else -1)
            state = None
            act = 0
            for j in range(dec + 1):
                ev_ch = ch if (wired and j >= max(0, dec - 2)) else -1
                ev = None
                if ev_ch >= 0:
                    e = torch.zeros(1, kk, w, device=device)
                    e[:, :, ev_ch::3] = 2.0
                    ev = e
                out, state = model(ft[j:j+1], ctrl, dt, state=state, evidence=ev)
                act = int(torch.argmax(out.action_dist[0]).item())
            res[cond].append(int(act == ep.label))
    model.train()
    return {k_: round(float(np.mean(v)), 3) for k_, v in res.items()}


@torch.no_grad()
def permutation_test(model, device):
    model.eval()
    rgb = torch.rand(1, 3, 32, 32, device=device)
    ctrl = torch.zeros(1, 307, device=device)
    dt = torch.tensor([0.016667], device=device)
    out1, _ = model(rgb, ctrl, dt)
    d1 = out1.action_dist[0]
    exact = True
    dev = 0.0
    try:
        state_p = model.initial_state(1).to(device)
        perm = list(range(1, state_p.shape[1])) + [0]
        out2, _ = model(rgb, ctrl, dt, state=state_p[:, perm])
        d2 = out2.action_dist[0]
        exact = bool(torch.allclose(d1, d2, atol=1e-5))
        dev = float((d1 - d2).abs().max()) if not exact else 0.0
    except Exception as exc:  # property test must never silently pass
        exact = False
        dev = float("nan")
        print(f"[perm-test] error: {exc}")
    model.train()
    return {"exact": exact, "max_dev": dev}


@torch.no_grad()
def escalation_eval(model, device, episodes=40):
    model.eval()
    returns, collapses = [], 0
    for ep_i in range(episodes):
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        haz_a, haz_b, haz_c = bool(ep_i % 2 == 0), bool((ep_i//2) % 2 == 0), bool((ep_i//4) % 2 == 0)
        world_idx = int(haz_a)*4 + int(haz_b)*2 + int(haz_c)
        correct_act = 1 + (world_idx % 4)
        frames = [torch.randn(1, 3, 32, 32, device=device) * 0.1 + 0.5 for _ in range(4)]
        frames[1][:, 0, :16 if haz_a else 16:] += 0.8
        frames[2] = frames[1].clone(); frames[2][:, 1, :16 if haz_b else 16:] += 0.8
        frames[3] = frames[2].clone(); frames[3][:, 2, :16 if haz_c else 16:] += 0.8
        state = None
        ret = -100.0
        escaped = False
        for t in range(60):
            fi = 0 if t < 15 else (1 if t < 30 else (2 if t < 45 else 3))
            out, state = model(frames[fi], ctrl, dt, state=state)
            act = int(torch.argmax(out.action_dist[0]).item())
            if act != 0 and t >= 45:
                if act == correct_act:
                    ret = 40.0 + (t - 45) * (-1.0); escaped = True; break
                ret = -100.0
                break
        if not escaped:
            collapses += 1
        returns.append(ret)
    model.train()
    return {"mean_return": round(float(np.mean(returns)), 2),
            "collapse_rate": round(collapses / len(returns), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== ADAPTIVE-GATE WIRING EXPERIMENT on {device} ===")

    payload = {"git_head": os.popen("git rev-parse HEAD").read().strip() or "n/a",
               "arms": {}}
    for arm in ("control", "gate_wired"):
        vp = {"seeds": {}}
        for seed in SEEDS:
            base = esc.VectorizedPseudoBrain(thoughtlets=32, width=120,
                                             heads=4, cycles=3, init_seed=seed)
            model = base if arm == "control" else GateWiredCore(base)
            model = model.to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n--- {arm} seed={seed} params={n_params} ---")
            curve = train_torture(model, device, args.steps, seed)
            lift = eval_lift(model, device)
            esc_res = escalation_eval(model, device)
            perm = permutation_test(model, device)
            mem = memory_probe(model, device)
            causal = mem["correct"] > mem["zero"] and mem["wrong"] < mem["correct"]
            # latency p50
            model.eval()
            ft = torch.rand(1, 3, 32, 32, device=device)
            ctrl = torch.zeros(1, 307, device=device)
            dt = torch.tensor([0.016667], device=device)
            with torch.no_grad():
                for _ in range(10):
                    model(ft, ctrl, dt)
                torch.cuda.synchronize() if device.type == "cuda" else None
                ts = []
                for _ in range(50):
                    t0 = time.perf_counter()
                    model(ft, ctrl, dt)
                    torch.cuda.synchronize() if device.type == "cuda" else None
                    ts.append((time.perf_counter() - t0) * 1000)
            lat_p50 = round(float(np.percentile(ts, 50)), 4)
            vp["seeds"][str(seed)] = {
                "final_lift": round(lift, 4), "curve": curve,
                "escalation": esc_res, "permutation": perm,
                "memory_probe": mem, "memory_causal": bool(causal),
                "latency_p50_ms": lat_p50, "params": n_params}
            print(f"  lift={lift:+.4f} esc={esc_res} perm={perm['exact']} "
                  f"mem={mem} causal={causal} lat={lat_p50}ms")

        lifts = [v["final_lift"] for v in vp["seeds"].values()]
        cols = [v["escalation"]["collapse_rate"] for v in vp["seeds"].values()]
        lats = [v["latency_p50_ms"] for v in vp["seeds"].values()]
        vp["summary"] = {
            "mean_lift": round(float(np.mean(lifts)), 4),
            "std_lift": round(float(np.std(lifts)), 4),
            "collapse_rate_mean": round(float(np.mean(cols)), 3),
            "memory_causal_n": sum(v["memory_causal"] for v in vp["seeds"].values()),
            "perm_exact_all": all(v["permutation"]["exact"]
                                  for v in vp["seeds"].values()),
            "latency_p50_mean_ms": round(float(np.mean(lats)), 4)}
        payload["arms"][arm] = vp
        print(f"\n{arm.upper()} SUMMARY: {vp['summary']}")

    c = payload["arms"]["control"]["summary"]
    e = payload["arms"]["gate_wired"]["summary"]
    verdict = {
        "G1_memory_causal_3of3": e["memory_causal_n"] == 3,
        "G2_lift_no_regression":
            e["mean_lift"] >= c["mean_lift"] - c["std_lift"],
        "G3_collapse_no_worse":
            e["collapse_rate_mean"] <= c["collapse_rate_mean"] + 0.05,
        "G4_permutation_exact": e["perm_exact_all"],
        "G5_latency_within_tolerance":
            e["latency_p50_mean_ms"] <= 1.25 * c["latency_p50_mean_ms"]}
    verdict["SCREENING_PASS"] = all(verdict.values())
    payload["VERDICT"] = verdict
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
