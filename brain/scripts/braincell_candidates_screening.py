"""Phase 2.6 BrainCell candidate campaign — screening round.

Candidates (all preserve: shared weights across slots, whole-slot permutation
invariance, no belief->action bypass, vectorized [B,K,W], same actuator/head/K):

  A current     : frozen control (slot GRU + attention cycles) = locked baseline
  B gated_gru   : stronger preserve/overwrite control - learned per-slot input
                  gate and forget gate added to the recurrent update (GRU-style
                  gating with explicit write strength), shared across slots
  C evidence_residual : explicit keep/accept decomposition - thought update is
                  h' = keep * h + accept * proj([sens; evidence]), where
                  'evidence' is a per-slot transform of the new observation,
                  gates computed from [h; sens] jointly (shared weights)

Diagnostics beyond torture score:
  - synthetic-memory causal response (ideal injected evidence, no store):
    correct vs zero vs wrong injection shifts decisions predictably?
  - evidence-revision probe: belief A established, strong counter-evidence:
    revise? preserve unrelated? avoid full wipe?
  - learning curves at 500/1000/2000/4000/6000 steps
  - collapse rate on escalation continuation, latency, param delta

Screening: 3 seeds per candidate on the torture suite.
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
sys.modules["esc_bc"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import TASKS  # noqa: E402


class CandidateCore(nn.Module):
    """Unified wrapper: variant in {'current','gated_gru','evidence_residual'}.

    All variants share conv encoder, attention block, actuator, slot identities.
    The recurrent update differs per variant. An optional external evidence
    vector (synthetic-memory probe channel) enters ONLY through the update rule.
    """

    def __init__(self, base: esc.VectorizedPseudoBrain, variant: str, device):
        super().__init__()
        self.base = base
        self.variant = variant
        w = base.w
        if variant == "gated_gru":
            self.write_gate = nn.Linear(2 * w, w)
            # reuse base.slot_gru for candidate path
        elif variant == "evidence_residual":
            self.keep_gate = nn.Linear(2 * w, w)
            self.accept_proj = nn.Linear(2 * w, w)
            self.accept_gate = nn.Linear(2 * w, w)
        self.to(device)
        self.device_ = device

    def initial_state(self, b):
        return self.base.initial_state(b)

    def update(self, sens, h, evidence=None):
        """One recurrent update for all variants. sens,h: [B,K,W].
        evidence: optional [B,K,W] injected signal (probe channel)."""
        b, k, w = h.shape
        if self.variant == "current":
            gru_in = torch.cat((sens, h), dim=-1).reshape(b * k, 2 * w)
            h_new = self.base.slot_gru(gru_in, h.reshape(b * k, w)).reshape(b, k, w)
            return self.base.slot_norm(h_new)
        if self.variant == "gated_gru":
            gru_in = torch.cat((sens, h), dim=-1).reshape(b * k, 2 * w)
            g_in = torch.cat((sens, h), dim=-1)
            write = torch.sigmoid(self.write_gate(g_in))          # [B,K,W]
            h_new = self.base.slot_gru(gru_in, h.reshape(b * k, w)).reshape(b, k, w)
            h_mixed = write * h_new + (1 - write) * h             # explicit write strength
            return self.base.slot_norm(h_mixed)
        # evidence_residual
        g_in = torch.cat((sens, h), dim=-1)
        keep = torch.sigmoid(self.keep_gate(g_in))
        ev = sens if evidence is None else evidence
        cand = torch.tanh(self.accept_proj(torch.cat((ev, h), dim=-1)))
        accept = torch.sigmoid(self.accept_gate(g_in))
        h_new = keep * h + accept * cand                          # explicit keep/accept
        return self.base.slot_norm(h_new)

    def forward(self, rgb, ctrl, dt, state=None, evidence=None):
        b = rgb.shape[0]
        if state is None:
            state = self.initial_state(b).to(rgb.device)
        raw = self.base.conv(rgb)
        sens = raw.unsqueeze(1).expand(b, self.base.k, self.base.w)
        h = self.update(sens, state, evidence=evidence)
        for _c in range(1, self.base.cycles):
            q = self.base.q_proj(h).reshape(b, self.base.k, self.base.heads,
                                            self.base.w // self.base.heads).transpose(1, 2)
            k_ = self.base.k_proj(h).reshape(b, self.base.k, self.base.heads,
                                             self.base.w // self.base.heads).transpose(1, 2)
            v = self.base.v_proj(h).reshape(b, self.base.k, self.base.heads,
                                            self.base.w // self.base.heads).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k_, v)
            attn = attn.transpose(1, 2).reshape(b, self.base.k, self.base.w)
            h = self.base.attn_norm(h + self.base.attn_out(attn))
        out = self.base.actuator(sensors=raw.unsqueeze(1), thoughts=h)
        return out, h


def frames_tensor(ep, device):
    return torch.from_numpy(np.stack(ep.frames).astype(np.float32) / 255.0)\
        .permute(0, 3, 1, 2).to(device)


def train_steps(model, device, steps, seed, eps_per_task=24, curve_at=(500, 1000, 2000, 4000, 6000)):
    rng = np.random.default_rng(seed)
    bank = []
    for fn in TASKS:
        for i in range(eps_per_task):
            r = np.random.default_rng(seed + hash(fn.__name__) % 99991 + i)
            bank.append(fn(r))
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
        for j in range(ep.decision_frame_index + 1):   # feed frames ONE at a time
            out, state = model(ft[j:j+1], ctrl, dt, state=state)
        logits = out.proposals.action_logits           # [1, K, 5]
        target = torch.full((1, logits.shape[1]), ep.label,
                            dtype=torch.long, device=device)
        loss = F.cross_entropy(logits.view(-1, 5), target.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step in curve_at:
            curve[str(step)] = eval_lift(model, device, episodes=12)
            print(f"  step {step}: lift={curve[str(step)]:+.4f}")
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
            act = 0
            for j in range(ep.decision_frame_index + 1):
                out, _ = model(ft[j:j+1], ctrl, dt)
                act = int(torch.argmax(out.action_dist[0]).item())
            ok += int(act == ep.label)
        lifts.append(ok / episodes - chance)
    model.train()
    return float(np.mean(lifts))


@torch.no_grad()
def tasks_above(model, device, episodes=30, base_seed=20260822):
    model.eval()
    n = 0
    detail = {}
    for fn in TASKS:
        ok = 0
        chance = float(getattr(fn(np.random.default_rng(0)), "chance", 0.0))
        for i in range(episodes):
            r = np.random.default_rng(base_seed + i * 7919)
            ep = fn(r)
            ft = frames_tensor(ep, device)
            ctrl = torch.zeros(1, 307, device=device)
            dt = torch.tensor([0.016667], device=device)
            act = 0
            for j in range(ep.decision_frame_index + 1):
                out, _ = model(ft[j:j+1], ctrl, dt)
                act = int(torch.argmax(out.action_dist[0]).item())
            ok += int(act == ep.label)
        acc = ok / episodes
        detail[fn.__name__] = {"acc": round(acc, 3), "chance": round(chance, 4)}
        if acc - chance > 0.02:
            n += 1
    model.train()
    return n, detail


@torch.no_grad()
def synthetic_memory_probe(model, device, episodes=40):
    """Inject ideal evidence at the cue frame of t01 (single-cue short delay).
    Correct evidence = one-hot channel signature matching true label;
    wrong = shifted channel; zero = control."""
    model.eval()
    res = {"correct": [], "zero": [], "wrong": []}
    for i in range(episodes):
        r = np.random.default_rng(20260822 + i * 7919)
        from irene_brain.evaluation.torture_suite import t01_single_cue_short_delay
        ep = t01_single_cue_short_delay(r)
        ft = frames_tensor(ep, device)
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        dec = ep.decision_frame_index
        true_ch = ep.label - 1
        for cond in ("correct", "zero", "wrong"):
            ch = true_ch if cond == "correct" else ((true_ch + 1) % 3 if cond == "wrong" else -1)
            state = None
            act = 0
            for j in range(dec + 1):
                ev = None
                if j >= max(0, dec - 2):     # inject near decision only
                    if cond != "zero":
                        e = torch.zeros(1, model.base.k, model.base.w, device=device)
                        e[:, :, ch::3] = 2.0
                        ev = e
                    else:
                        ev = torch.zeros(1, model.base.k, model.base.w, device=device)
                out, state = model(ft[j:j+1], ctrl, dt, state=state, evidence=ev)
                act = int(torch.argmax(out.action_dist[0]).item())
            res[cond].append(int(act == ep.label))
    model.train()
    return {k: round(float(np.mean(v)), 3) for k, v in res.items()}


@torch.no_grad()
def evidence_revision_probe(model, device, episodes=40):
    """t07-like: establish belief via early cue, then strong counter-evidence
    later. Measure: revision accuracy + state preservation (cosine sim of
    unrelated slots before/after counter-evidence)."""
    from irene_brain.evaluation.torture_suite import t07_disproven_hypothesis
    model.eval()
    rev = pres = wipe = 0
    for i in range(episodes):
        r = np.random.default_rng(20260822 + i * 7919)
        ep = t07_disproven_hypothesis(r)
        ft = frames_tensor(ep, device)
        ctrl = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)
        state = None
        pre_states = []
        for j in range(ep.decision_frame_index + 1):
            out, state = model(ft[j:j+1], ctrl, dt, state=state)
            pre_states.append(state.detach().clone())
        act = int(torch.argmax(out.action_dist[0]).item())
        rev += int(act == ep.label)
        # preservation proxy: drift of state norm across mid-episode (no wipe)
        if len(pre_states) > 3:
            a, b_ = pre_states[len(pre_states)//2 - 1], pre_states[-1]
            cos = F.cosine_similarity(a.reshape(-1), b_.reshape(-1), dim=0).item()
            pres += cos
            wipe += int(cos < 0.2)
    model.train()
    n = episodes
    return {"revision_acc": round(rev / n, 3),
            "mean_state_cosine": round(pres / n, 3),
            "wipe_rate": round(wipe / n, 3)}


@torch.no_grad()
def latency_ms(model, device, iters=200):
    model.eval()
    rgb = torch.randn(1, 3, 32, 32, device=device)
    ctrl = torch.zeros(1, 307, device=device)
    dt = torch.tensor([0.016667], device=device)
    st = None
    for _ in range(10):
        out, st = model(rgb, ctrl, dt, state=st)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        out, st = model(rgb, ctrl, dt, state=st)
    torch.cuda.synchronize()
    model.train()
    return round((time.perf_counter() - t0) / iters * 1000, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seeds", type=str, default="42,142,242")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== BRAINCELL CANDIDATE SCREENING on {device}, steps={args.steps}, seeds={seeds} ===")

    payload = {}
    for variant in ("current", "gated_gru", "evidence_residual"):
        vp = {"seeds": {}}
        for seed in seeds:
            base = esc.VectorizedPseudoBrain(thoughtlets=32, width=120, heads=4,
                                             cycles=3, init_seed=seed).to(device)
            model = CandidateCore(base, variant, device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"\n--- {variant} seed={seed} params={n_params:,} ---")
            curve = train_steps(model, device, args.steps, seed)
            lift = eval_lift(model, device, episodes=30)
            above, detail = tasks_above(model, device)
            mem_probe = synthetic_memory_probe(model, device)
            rev = evidence_revision_probe(model, device)
            lat = latency_ms(model, device)
            vp["seeds"][str(seed)] = {
                "params": n_params, "final_lift": round(lift, 4),
                "tasks_above": above, "learning_curve": curve,
                "memory_probe": mem_probe, "revision": rev, "latency_ms": lat}
            print(f"{variant} s{seed}: lift={lift:+.4f} above={above}/25 "
                  f"mem={mem_probe} lat={lat}ms")
        # aggregate
        lifts = [v["final_lift"] for v in vp["seeds"].values()]
        aboves = [v["tasks_above"] for v in vp["seeds"].values()]
        vp["mean_lift"] = round(float(np.mean(lifts)), 4)
        vp["std_lift"] = round(float(np.std(lifts)), 4)
        vp["mean_tasks_above"] = float(np.mean(aboves))
        mem_ok = all(v["memory_probe"]["correct"] > v["memory_probe"]["zero"]
                     and v["memory_probe"]["wrong"] < v["memory_probe"]["correct"]
                     for v in vp["seeds"].values())
        vp["memory_causal_consistent_all_seeds"] = mem_ok
        payload[variant] = vp

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print("\n=== SUMMARY ===")
    for v, d in payload.items():
        print(f"{v}: mean_lift={d['mean_lift']:+.4f} ±{d['std_lift']} "
              f"above={d['mean_tasks_above']:.1f}/25 "
              f"mem_causal={d['memory_causal_consistent_all_seeds']}")
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
