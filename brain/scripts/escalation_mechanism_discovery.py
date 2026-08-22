"""Phase 2.5 mechanism discovery: training telemetry + checkpoint preservation.

Derived from dgx_phase2_definitive_escalation.py @ cc8fcbe by the reliability harness.
Changes ONLY:
  - trains {GRU, PB K=32} x 10 independent seeds x 3000 steps (same envelope as the
    reliability discovery set);
  - every TELEMETRY_EVERY steps records a cheap held-out probe (return, S1/S2, final acc,
    stage-0 probability mass stats, action stickiness) WITHOUT touching training;
  - saves one checkpoint per seed (state_dict + metadata) to /workspace/run/checkpoints/;
  - dumps per-seed returns to /workspace/run/per_seed_returns.json.

Purpose: capture before-collapse trajectories so catastrophic collapse can be studied
longitudinally, and preserve checkpoints for the hostile mechanistic battery.

No base science code is modified. Run from /workspace/repo inside the GPU container.
"""
import json
import os
import time
from typing import Any

import numpy as np
import torch

from irene_brain.model.consequence_thought_actuator import ConsequenceThoughtActuator

# Reuse everything from the canonical script (models, train fn, evaluator).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "esc_base", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "dgx_phase2_definitive_escalation.py"))
esc = importlib.util.module_from_spec(_spec)
sys_modules_name = "esc_base_mech"
import sys
sys.modules[sys_modules_name] = esc
_spec.loader.exec_module(esc)

TELEMETRY_EVERY = 100
QUICK_EVAL_SEEDS = [9001]
QUICK_EPISODES = 8
TRAIN_STEPS = 3000
TRAINING_SEEDS = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]
CATACSTROPHIC_FLOOR = -520.0
RUN_DIR = "/workspace/run"


def quick_probe(model, device) -> dict[str, float]:
    """Cheap held-out probe: small eval + stage-0 probability mass diagnostics."""
    res = esc.evaluate_8hypothesis_benchmark(
        model, device, eval_seeds=QUICK_EVAL_SEEDS, delay=15, ablation="none")
    # Probability-mass telemetry at stage 0 on fresh states.
    model.eval()
    ctrl0 = torch.zeros(1, 307, device=device)
    dt = torch.tensor([0.016667], device=device)
    with torch.no_grad():
        rgb0 = torch.randn(1, 3, 32, 32, device=device) * 0.1 + 0.5
        out0, _ = model(rgb0, ctrl0, dt, state=None)
        p0 = out0.proposals.branch_probability.squeeze(-1)  # [1, K]
        committed = float((p0 > 0.5).float().mean().item())
        p_max = float(p0.max().item())
        p_mean = float(p0.mean().item())
    return {
        "probe_return": res["mean_return"],
        "probe_s1": res["stage1_surv_pct"],
        "probe_final_acc": res["final_acc_pct"],
        "p0_committed_frac": committed,
        "p0_max": p_max,
        "p0_mean": p_mean,
    }


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print(f"=== MECHANISM DISCOVERY RUN: telemetry + checkpoints ({device}) ===")
    print("=" * 115)

    ckpt_dir = os.path.join(RUN_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    telemetry_path = os.path.join(RUN_DIR, "training_trajectories.jsonl")

    model_configs = [
        ("Proposal-GRU Baseline", "gru", None),
        ("Pseudo-Brain K=32", "pb", 32),
    ]

    per_seed_summary: list[dict[str, Any]] = []

    for name, m_type, k_val in model_configs:
        for t_seed in TRAINING_SEEDS:
            torch.manual_seed(t_seed)
            np.random.seed(t_seed)
            if m_type == "gru":
                model = esc.ProposalGRUBaseline(hidden_dim=112, num_proposals=21).to(device)
            else:
                model = esc.VectorizedPseudoBrain(
                    thoughtlets=k_val, width=120, heads=4, cycles=3,
                    init_seed=t_seed).to(device)

            # --- training with telemetry ---
            model.train()
            import torch.optim as optim
            import torch.nn as nn
            import torch.nn.functional as F
            optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
            k_slots = model.k if m_type == "pb" else model.num_proposals
            t0 = time.perf_counter()
            traj: list[dict[str, Any]] = []
            collapse_step = None

            for step in range(TRAIN_STEPS):
                haz_a = (step % 2 == 0)
                haz_b = ((step // 2) % 2 == 0)
                haz_c = ((step // 4) % 2 == 0)
                world_idx = (0 if haz_a else 4) + (0 if haz_b else 2) + (0 if haz_c else 1)

                rgb0 = torch.randn(1, 3, 32, 32, device=device) * 0.1 + 0.5
                ctrl0 = torch.zeros(1, 307, device=device)
                dt = torch.tensor([0.016667], device=device)

                out0, h0 = model(rgb0, ctrl0, dt, state=None)
                p0 = out0.proposals.branch_probability.squeeze(-1)
                loss_p0 = F.binary_cross_entropy(p0, torch.full_like(p0, 0.125))
                target_act0 = torch.zeros((1, k_slots), dtype=torch.long, device=device)
                loss_act0 = F.cross_entropy(out0.proposals.action_logits.view(-1, 5), target_act0.view(-1))

                rgb1 = rgb0.clone()
                if haz_a: rgb1[:, 0, :16, :16] += 0.8
                else: rgb1[:, 0, :16, 16:] += 0.8
                out1, h1 = model(rgb1, ctrl0, dt, state=h0)
                p1 = out1.proposals.branch_probability.squeeze(-1)
                eighth = max(1, k_slots // 8)
                target_p1 = torch.zeros_like(p1)
                if haz_a: target_p1[:, :4*eighth] = 0.25
                else: target_p1[:, 4*eighth:] = 0.25
                loss_p1 = F.binary_cross_entropy(p1, target_p1)
                target_act1 = torch.zeros((1, k_slots), dtype=torch.long, device=device)
                loss_act1 = F.cross_entropy(out1.proposals.action_logits.view(-1, 5), target_act1.view(-1))

                rgb2 = rgb1.clone()
                if haz_b: rgb2[:, 1, :16, :16] += 0.8
                else: rgb2[:, 1, :16, 16:] += 0.8
                out2, h2 = model(rgb2, ctrl0, dt, state=h1)
                p2 = out2.proposals.branch_probability.squeeze(-1)
                target_p2 = torch.zeros_like(p2)
                half_block = (0 if haz_a else 4*eighth) + (0 if haz_b else 2*eighth)
                target_p2[:, half_block:half_block + 2*eighth] = 0.50
                loss_p2 = F.binary_cross_entropy(p2, target_p2)
                target_act2 = torch.zeros((1, k_slots), dtype=torch.long, device=device)
                loss_act2 = F.cross_entropy(out2.proposals.action_logits.view(-1, 5), target_act2.view(-1))

                rgb3 = rgb2.clone()
                if haz_c: rgb3[:, 2, 16:, :16] += 0.8
                else: rgb3[:, 2, 16:, 16:] += 0.8
                out3, _h3 = model(rgb3, ctrl0, dt, state=h2)
                p3 = out3.proposals.branch_probability.squeeze(-1)
                target_p3 = torch.zeros_like(p3)
                final_block = world_idx * eighth
                target_p3[:, final_block:final_block + eighth] = 1.00
                loss_p3 = F.binary_cross_entropy(p3, target_p3)
                opt_act = 1 + (world_idx % 4)
                target_act3 = torch.full((1, k_slots), opt_act, dtype=torch.long, device=device)
                loss_act3 = F.cross_entropy(out3.proposals.action_logits.view(-1, 5), target_act3.view(-1))

                total_loss = (loss_p0 + loss_act0 + loss_p1 + loss_act1 +
                              loss_p2 + loss_act2 + 2.0 * loss_p3 + loss_act3)
                optimizer.zero_grad()
                total_loss.backward()
                grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                optimizer.step()

                if (step + 1) % TELEMETRY_EVERY == 0 or step == TRAIN_STEPS - 1:
                    probe = quick_probe(model, device)
                    rec = {"model": name, "train_seed": t_seed, "step": step + 1,
                           "loss": float(total_loss.item()), "grad_norm": grad_norm, **probe}
                    traj.append(rec)
                    if (collapse_step is None and step + 1 >= 200
                            and probe["probe_return"] <= CATACSTROPHIC_FLOOR):
                        collapse_step = step + 1

            elapsed = time.perf_counter() - t0

            # --- final full evaluation + checkpoint ---
            final_res = esc.evaluate_8hypothesis_benchmark(
                model, device, eval_seeds=[1000, 2000, 3000, 4000, 5000],
                delay=15, ablation="none")
            collapsed = final_res["mean_return"] <= CATACSTROPHIC_FLOOR
            ck_meta = {
                "model": name, "m_type": m_type, "k": k_val, "train_seed": t_seed,
                "init_seed": t_seed, "width": 120, "cycles": 3, "hidden_dim": 112,
                "num_proposals": 21, "steps": TRAIN_STEPS,
                "final": {k: v for k, v in final_res.items()},
                "collapsed": bool(collapsed),
                "collapse_step": collapse_step,
            }
            torch.save({"meta": ck_meta, "state_dict": model.state_dict()},
                       os.path.join(ckpt_dir, f"{m_type}_seed{t_seed}.pt"))
            for rec in traj:
                rec["final_mean_return"] = final_res["mean_return"]
                rec["collapsed"] = bool(collapsed)
            with open(telemetry_path, "a") as f:
                for rec in traj:
                    f.write(json.dumps(rec) + "\n")
            per_seed_summary.append({
                "model": name, "train_seed": t_seed,
                "returns_raw": final_res["mean_return"], "s1": final_res["stage1_surv_pct"],
                "s2": final_res["stage2_surv_pct"], "acc": final_res["final_acc_pct"],
                "collapsed": bool(collapsed), "collapse_step": collapse_step,
                "elapsed_s": round(elapsed, 1),
            })
            with open(os.path.join(RUN_DIR, "per_seed_returns.json"), "a") as f:
                f.write(json.dumps(per_seed_summary[-1]) + "\n")
            print(f"  {name:26s} seed={t_seed} ret={final_res['mean_return']:+7.2f} "
                  f"S1={final_res['stage1_surv_pct']:5.1f}% collapsed={collapsed} "
                  f"collapse_step={collapse_step} [{elapsed:.0f}s]")

    print("\n=== SUMMARY ===")
    for cfg in model_configs:
        rows = [r for r in per_seed_summary if r["model"] == cfg[0]]
        n_col = sum(1 for r in rows if r["collapsed"])
        rets = [r["returns_raw"] for r in rows]
        print(f"{cfg[0]:26s} n={len(rows)} collapsed={n_col}/{len(rows)} "
              f"mean={np.mean(rets):+.2f} median={np.median(rets):+.2f}")
    print("DONE")


if __name__ == "__main__":
    main()
