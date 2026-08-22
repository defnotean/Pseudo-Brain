"""Phase 2.6 torture-suite runner: scores a model on all 25 cognition tasks.

Usage (inside GPU container, from /workspace/repo):
  python3 -B brain/scripts/run_torture_suite.py \
      --model {pb_k32|gru} [--checkpoint PATH | --fresh-seed N] \
      --episodes 40 --output /workspace/out/torture_<name>.json

Model wrappers: the two Phase 2.5 campaign architectures (VectorizedPseudoBrain
K=32 W=120 and ProposalGRUBaseline H=112), loaded either fresh (fixed init seed)
or from a saved checkpoint.

Protocol per task: N episodes with seeds derived from a fixed base seed; the
model consumes frames one at a time; at `decision_frame_index` its argmax action
is scored against the episode label. WAIT (0) counts as wrong except on t19/t21
where it is recorded separately as a calibration statistic.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import torch

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "esc_base", os.path.join(_here, "..", "..", "dgx_phase2_definitive_escalation.py"))
esc = importlib.util.module_from_spec(_spec)
sys.modules["esc_torture"] = esc
_spec.loader.exec_module(esc)

sys.path.insert(0, os.path.join(_here, "..", "src"))
from irene_brain.evaluation.torture_suite import TASKS  # noqa: E402


def build_model(kind: str, seed: int, device, checkpoint: str | None):
    if kind == "pb_k32":
        model = esc.VectorizedPseudoBrain(
            thoughtlets=32, width=120, heads=4, cycles=3, init_seed=seed).to(device)
    elif kind == "gru":
        model = esc.ProposalGRUBaseline(hidden_dim=112, num_proposals=21).to(device)
    else:
        raise ValueError(kind)
    if checkpoint:
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        # warm-up for lazy rank-dependent layers before loading
        with torch.no_grad():
            rgb0 = torch.randn(1, 3, 32, 32, device=device)
            model(rgb0, torch.zeros(1, 307, device=device),
                  torch.tensor([0.016667], device=device), state=None)
        model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    model.eval()
    return model


@torch.no_grad()
def score_model(model, device, episodes_per_task: int, base_seed: int) -> dict:
    ctrl = torch.zeros(1, 307, device=device)
    dt = torch.tensor([0.016667], device=device)
    results = {}
    for task_fn in TASKS:
        correct = 0
        waits_at_decision = 0
        for ep_i in range(episodes_per_task):
            rng = np.random.default_rng(base_seed + ep_i * 7919)
            ep = task_fn(rng)
            state = None
            act = 0
            for i, frame in enumerate(ep.frames):
                rgb = torch.from_numpy(frame.astype(np.float32) / 255.0)
                rgb = rgb.permute(2, 0, 1).unsqueeze(0).to(device)
                out, state = model(rgb, ctrl, dt, state=state)
                act = int(torch.argmax(out.action_dist[0]).item())
                if i == ep.decision_frame_index:
                    break
            if act == 0:
                waits_at_decision += 1
            if ep.label != 0 and act == ep.label:
                correct += 1
        acc = correct / episodes_per_task
        results[ep.task_name] = {
            "accuracy": round(acc, 4),
            "chance": round(ep.chance, 4),
            "lift": round(acc - ep.chance, 4),
            "wait_rate_at_decision": round(waits_at_decision / episodes_per_task, 3),
        }
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["pb_k32", "gru"])
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--fresh-seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=20260822)
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.fresh_seed, device, args.checkpoint)
    print(f"=== TORTURE SUITE: {args.model} ckpt={args.checkpoint or 'fresh'} "
          f"episodes={args.episodes} on {device} ===")
    results = score_model(model, device, args.episodes, args.base_seed)

    total_lift = np.mean([v["lift"] for v in results.values()])
    n_above = sum(1 for v in results.values() if v["lift"] > 0.02)
    payload = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "fresh_seed": args.fresh_seed,
        "episodes_per_task": args.episodes,
        "per_task": results,
        "mean_lift_over_chance": round(float(total_lift), 4),
        "tasks_above_chance_2pct": n_above,
        "n_tasks": len(results),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"{'task':28s} {'acc':>7s} {'chance':>7s} {'lift':>7s} {'wait%':>6s}")
    for name, v in results.items():
        flag = " +" if v["lift"] > 0.02 else "  "
        print(f"{name:28s} {v['accuracy']:7.3f} {v['chance']:7.3f} "
              f"{v['lift']:+7.3f}{flag} {v['wait_rate_at_decision']*100:5.1f}%")
    print(f"\nmean lift over chance: {payload['mean_lift_over_chance']:+.4f} "
          f"({n_above}/{len(results)} tasks >+2%)")
    print("DONE ->", args.output)


if __name__ == "__main__":
    main()
