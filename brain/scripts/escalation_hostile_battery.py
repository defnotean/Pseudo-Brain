"""Phase 2.5 hostile mechanistic battery on preserved discovery-set checkpoints.

Loads checkpoints saved by escalation_mechanism_discovery.py and runs the full
held-out evaluation under each intervention. Compares four cells:
  surviving-PB, collapsed-PB, surviving-GRU, collapsed-GRU.

Frozen design (see runs/2026-08-22-precollapse-trajectories.md): interventions are
normal / reset / stale1 / stale5 / stale20 / scramble_prob / scramble_binding /
permute (PB only). No architecture or training changes.
"""
import json
import os
import sys

import numpy as np
import torch

import importlib.util
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "esc_base", os.path.join(_here, "dgx_phase2_definitive_escalation.py"))
esc = importlib.util.module_from_spec(_spec)
sys.modules["esc_base_batt"] = esc
_spec.loader.exec_module(esc)

RUN_DIR = "/workspace/run"
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
CELLS = {
    "surviving-PB": [("pb", 742), ("pb", 842), ("pb", 442)],
    "collapsed-PB": [("pb", 142), ("pb", 242), ("pb", 542)],
    "surviving-GRU": [("gru", 442), ("gru", 642), ("gru", 242)],
    "collapsed-GRU": [("gru", 42), ("gru", 142), ("gru", 842)],
}
INTERVENTIONS = ["normal", "reset", "stale1", "stale5", "stale20",
                 "scramble_prob", "scramble_binding", "permute"]
EVAL_SEEDS = [1000, 2000, 3000, 4000, 5000]


def load_model(m_type: str, seed: int, device):
    ck = torch.load(os.path.join(CKPT_DIR, f"{m_type}_seed{seed}.pt"),
                    map_location=device, weights_only=False)
    meta = ck["meta"]
    if m_type == "gru":
        model = esc.ProposalGRUBaseline(hidden_dim=meta["hidden_dim"],
                                        num_proposals=meta["num_proposals"]).to(device)
    else:
        model = esc.VectorizedPseudoBrain(thoughtlets=meta["k"], width=meta["width"],
                                          heads=4, cycles=meta["cycles"],
                                          init_seed=meta["init_seed"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, meta


def eval_with(model, device, ablation: str) -> dict[str, float]:
    # stale D is emulated by freezing state updates for the last D ticks of each
    # ambiguity stage via repeated first-tick inputs; approximated here by running
    # the canonical evaluator with delay reduction for stale modes is NOT valid,
    # so we implement stale as: run normal eval but re-inject the state snapshot
    # from D ticks prior at the final decision. The base evaluator does not expose
    # that hook, so stale uses reset_state=False plus manual rollout below.
    if ablation in ("normal", "reset", "scramble_prob", "scramble_binding", "permute"):
        return esc.evaluate_8hypothesis_benchmark(
            model, device, eval_seeds=EVAL_SEEDS, delay=15, ablation=ablation)
    return stale_eval(model, device, int(ablation[5:]))


def stale_eval(model, device, d: int) -> dict[str, float]:
    """Freeze the persistent state to its value D ticks before the final cue."""
    ctrl0 = torch.zeros(1, 307, device=device)
    dt = torch.tensor([0.016667], device=device)
    returns, s1s, s2s, accs = [], [], [], []
    with torch.no_grad():
        for seed in EVAL_SEEDS:
            ep_ret, s1c, s2c, fin_c = 0.0, 0, 0, 0
            episodes = 20
            for ep in range(episodes):
                np.random.seed(seed + ep * 13)
                torch.manual_seed(seed + ep * 13)
                haz_a = np.random.rand() > 0.5
                haz_b = np.random.rand() > 0.5
                haz_c = np.random.rand() > 0.5
                world_idx = (0 if haz_a else 4) + (0 if haz_b else 2) + (0 if haz_c else 1)
                state = None
                rgb0 = torch.randn(1, 3, 32, 32, device=device) * 0.1 + 0.5

                def roll(rgb, ticks, penalty, hist):
                    nonlocal state, ep_ret
                    act = 0
                    for _ in range(ticks):
                        out, state = model(rgb + torch.randn_like(rgb) * 0.02,
                                           ctrl0, dt, state=state)
                        act = int(torch.argmax(out.action_dist[0]).item())
                        if act != 0:
                            ep_ret -= penalty
                    if act == 0:
                        hist[0] += 1

                h1 = [0]
                roll(rgb0, 15, 15.0, h1)
                rgb1 = rgb0.clone()
                if haz_a: rgb1[:, 0, :16, :16] += 0.8
                else: rgb1[:, 0, :16, 16:] += 0.8
                h2 = [0]
                roll(rgb1, 15, 10.0, h2)
                s1c += h2[0]

                # snapshot state D ticks into stage 2 by replaying stage-2 noise
                snap = None
                rgb2 = rgb1.clone()
                if haz_b: rgb2[:, 1, :16, :16] += 0.8
                else: rgb2[:, 1, :16, 16:] += 0.8
                for t in range(15):
                    out, state = model(rgb2 + torch.randn_like(rgb2) * 0.02,
                                       ctrl0, dt, state=state)
                    act = int(torch.argmax(out.action_dist[0]).item())
                    if act != 0: ep_ret -= 10.0
                    if t == max(0, 14 - d):
                        snap = state
                if act == 0: s2c += 1

                rgb3 = rgb2.clone()
                if haz_c: rgb3[:, 2, 16:, :16] += 0.8
                else: rgb3[:, 2, 16:, 16:] += 0.8
                if snap is not None:
                    state = snap  # stale injection: rewind thought state by D ticks
                out_f, _ = model(rgb3 + torch.randn_like(rgb3) * 0.02,
                                 ctrl0, dt, state=state)
                final_act = int(torch.argmax(out_f.action_dist[0]).item())
                opt_act = 1 + (world_idx % 4)
                if final_act == opt_act:
                    fin_c += 1; ep_ret += 20.0
                else:
                    ep_ret -= 10.0
            returns.append(ep_ret / episodes)
            s1s.append(s1c / episodes * 100.0)
            s2s.append(s2c / episodes * 100.0)
            accs.append(fin_c / episodes * 100.0)
    return {"mean_return": float(np.mean(returns)),
            "stage1_surv_pct": float(np.mean(s1s)),
            "stage2_surv_pct": float(np.mean(s2s)),
            "final_acc_pct": float(np.mean(accs))}


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== HOSTILE BATTERY on preserved checkpoints ({device}) ===")
    out_path = os.path.join(RUN_DIR, "battery_results.jsonl")
    for cell, members in CELLS.items():
        per_intervention: dict[str, list[float]] = {k: [] for k in INTERVENTIONS}
        detail: dict[str, dict] = {}
        for m_type, seed in members:
            model, meta = load_model(m_type, seed, device)
            for abl in INTERVENTIONS:
                if abl == "permute" and m_type != "pb":
                    continue
                res = eval_with(model, device, abl)
                per_intervention[abl].append(res["mean_return"])
                detail[f"{m_type}{seed}/{abl}"] = res
            print(f"  [{cell}] {m_type} seed={seed} done")
        summary = {abl: {"mean": float(np.mean(v)) if v else None,
                         "per_seed": v}
                   for abl, v in per_intervention.items()}
        with open(out_path, "a") as f:
            f.write(json.dumps({"cell": cell, "summary": summary,
                                "detail": detail}) + "\n")
        print(f"--- {cell} ---")
        for abl in INTERVENTIONS:
            if summary[abl]["mean"] is not None:
                print(f"  {abl:18s} mean_return={summary[abl]['mean']:+8.2f}")
    print("DONE ->", out_path)


if __name__ == "__main__":
    main()
