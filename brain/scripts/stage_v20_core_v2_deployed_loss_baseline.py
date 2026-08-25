"""Stage V2.0: Core V2 deployed-loss baseline vs Core V1 reference (preregistered).

Prereg: brain/docs/preregistrations/2026-08-24-core-v2-deployed-loss-baseline-prereg.md
Frozen gates: alpha Δ>=+0.05 & σ<=0.06 | beta |Δ|<=0.02 | gamma Δ<=-0.03 | else AMBIGUOUS.
Arms: V2-A (CONFIG_A_DECISION_ONLY), V2-C (CONFIG_C_FULL). Deployed decision loss only.
Bank/eval protocol identical to Stage 3b (bank b3bb5fc33fd5f605, eval seed 20260822+i*7919).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "..", "src"))

from run_provenance import (build_bank_pinned, bank_digest, param_digest,
                            apply_deterministic_mode, provenance)
from irene_brain.evaluation.torture_suite import TASKS
from irene_brain.v2 import (
    CoreV2Model, CoreV2Config,
    CONFIG_A_DECISION_ONLY, CONFIG_C_FULL,
)
from irene_brain.v2.losses import deployed_decision_loss

SEEDS = [42, 142, 242, 342]
TOTAL_STEPS = 6000
REFERENCE_R = -0.2130  # Core V1 FIXED @6k, same bank, n=4


def frames_tensor(ep, device):
    return torch.from_numpy(np.stack(ep.frames).astype(np.float32) / 255.0) \
        .permute(0, 3, 1, 2).to(device)


def build_order(bank, B=16):
    by_len = {}
    for idx, ep in enumerate(bank):
        by_len.setdefault(ep.decision_frame_index + 1, []).append(idx)
    order = []
    for ln in sorted(by_len):
        members = by_len[ln]
        for k in range(0, len(members), B):
            group = members[k:k + B]
            while len(group) < B:
                group.append(members[(k + len(group)) % len(members)])
            order.append((ln, group))
    return order


@torch.no_grad()
def fixed_probe_loss(model, bank, order, device, probe_seed=424242,
                     class_weights=None, all_batches=False):
    """Deployed loss on fixed data without perturbing training RNG."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    try:
        torch.manual_seed(probe_seed)
        torch.cuda.manual_seed_all(probe_seed)
        model.eval()
        numerator = torch.zeros((), device=device)
        denominator = torch.zeros((), device=device)
        groups = order if all_batches else order[:1]
        for length, group in groups:
            episodes = [bank[index] for index in group]
            frames = torch.stack([
                torch.stack([frames_tensor(episode, device)[tick]
                             for tick in range(length)])
                for episode in episodes
            ])
            labels = torch.tensor(
                [episode.label for episode in episodes],
                dtype=torch.long,
                device=device,
            )
            state = model.init_state(len(episodes), device)
            output = None
            for tick in range(length):
                output, state = model(frames[:, tick], state)
            log_probs = F.log_softmax(output.decision.action_values, dim=-1)
            losses = F.nll_loss(log_probs, labels, reduction="none")
            sample_weights = (
                class_weights[labels] if class_weights is not None
                else torch.ones_like(losses)
            )
            numerator += (losses * sample_weights).sum()
            denominator += sample_weights.sum()
        return float((numerator / denominator).item())
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.train(was_training)


@torch.no_grad()
def eval_full_v2(model, device, episodes=30, base_seed=20260822):
    """Identical protocol to Stage 3b eval_full: argmax on DEPLOYED action_dist.

    Returns mean_lift plus non-gated diagnostics (slot agreement, mass entropy).
    """
    model.eval()
    lifts = []
    slot_agree = []
    mass_ent = []
    action_histogram = np.zeros(5, dtype=np.int64)
    confusion = np.zeros((5, 5), dtype=np.int64)
    for fn in TASKS:
        ok = 0
        chance = float(getattr(fn(np.random.default_rng(0)), "chance", 0.0))
        for i in range(episodes):
            r = np.random.default_rng(base_seed + i * 7919)
            ep = fn(r)
            ft = frames_tensor(ep, device)
            state = model.init_state(1, device)
            act = 0
            for j in range(ep.decision_frame_index + 1):
                out, state = model(ft[j:j + 1], state)
                act = int(torch.argmax(out.decision.action_dist[0]).item())
            ok += int(act == ep.label)
            action_histogram[act] += 1
            confusion[int(ep.label), act] += 1

            # diagnostics at decision frame
            dec = out.decision
            slot_argmax = dec.thought_action_probs[0].argmax(dim=-1)  # [K]
            slot_agree.append(float((slot_argmax == act).float().mean().item()))
            p = dec.mass_per_action[0].clamp_min(0)
            if p.sum() > 0:
                p = p / p.sum()
                mass_ent.append(float(-(p * p.clamp_min(1e-9).log()).sum().item()))
        lifts.append(ok / episodes - chance)
    model.train()
    lifts_a = np.array(lifts)
    class_totals = confusion.sum(axis=1)
    recalls = np.divide(
        np.diag(confusion), class_totals,
        out=np.zeros(5, dtype=np.float64), where=class_totals > 0,
    )
    return {
        "mean_lift": round(float(lifts_a.mean()), 4),
        "above_2pct": int((lifts_a > 0.02).sum()),
        "per_faculty": [round(float(x), 4) for x in lifts],
        "slot_agreement": round(float(np.mean(slot_agree)), 4) if slot_agree else None,
        "mass_entropy": round(float(np.mean(mass_ent)), 4) if mass_ent else None,
        "action_histogram": action_histogram.tolist(),
        "unique_actions": int(np.count_nonzero(action_histogram)),
        "confusion_matrix": confusion.tolist(),
        "per_class_recall": [round(float(value), 4) for value in recalls],
        "balanced_accuracy": round(float(recalls[class_totals > 0].mean()), 4),
    }


def run_v2_arm(flags, arm_name, seed, device, bdig, *, config=None,
               total_steps=TOTAL_STEPS, class_weights=None,
               full_bank_probe=False,
               class_weight_normalization="batch_weight_sum_v0"):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.cuda.manual_seed_all(seed)
    config = config or CoreV2Config(
        decision_aggregation="legacy_scalar_utility_v0")
    model = CoreV2Model(config=config, flags=flags).to(device)
    if class_weights is not None:
        class_weights = torch.as_tensor(
            class_weights, dtype=torch.float32, device=device)
        if class_weights.shape != (config.actions,):
            raise ValueError("class_weights must have one value per action")
    init_dig = param_digest(model)

    bank = build_bank_pinned(42, TASKS)  # fixed bank seed, same as all Stage 3 runs
    assert bank_digest(bank) == bdig, "bank digest mismatch"
    order = build_order(bank)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()

    rec = {"arm": arm_name, "seed": seed,
           "decision_aggregation": config.decision_aggregation,
           "braincell_dynamics": config.braincell_dynamics,
           "belief_dynamics": config.belief_dynamics,
           "status": "completed",
           "class_weights": class_weights.detach().cpu().tolist()
           if class_weights is not None else None,
           "class_weight_normalization": class_weight_normalization,
           "provenance": provenance(model=model, train_seed=seed,
                                    eval_seed=20260822, bank_digest=bdig,
                                    deterministic=True)}
    rec["provenance"]["init_param_digest"] = init_dig
    rec["initial_probe_loss"] = round(
        fixed_probe_loss(
            model, bank, order, device,
            class_weights=class_weights, all_batches=full_bank_probe,
        ), 5)
    rec["probe_scope"] = "full_padded_bank" if full_bank_probe else "first_batch"

    loss_curve = []
    initial_loss = None
    final_loss = None
    max_abs_belief = 0.0
    max_abs_thought = 0.0
    step = 0
    t0 = time.perf_counter()
    while step < total_steps:
        for ln, group in order:
            if step >= total_steps:
                break
            eps = [bank[i] for i in group]
            ftb = torch.stack([
                torch.stack([frames_tensor(e, device)[j] for j in range(ln)])
                for e in eps])
            labels = torch.tensor([e.label for e in eps], dtype=torch.long,
                                  device=device)
            # Fresh trial state per sequence batch; single supervised pass to
            # the decision frame (identical exposure to V1's harness).
            state = model.init_state(len(eps), device)
            out_j = None
            for j in range(ln):
                out_j, state = model(ftb[:, j], state)
                max_abs_belief = max(
                    max_abs_belief,
                    float(out_j.belief.detach().abs().max().item()),
                )
                max_abs_thought = max(
                    max_abs_thought,
                    float(out_j.thoughts.detach().abs().max().item()),
                )
            loss = deployed_decision_loss(
                out_j, labels, class_weights,
                weight_normalization=class_weight_normalization,
            )
            loss_value = float(loss.item())
            if not np.isfinite(loss_value):
                raise FloatingPointError(
                    f"non-finite deployed loss at step {step}")
            if initial_loss is None:
                initial_loss = loss_value
            opt.zero_grad()
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            if not np.isfinite(grad_norm):
                raise FloatingPointError(
                    f"non-finite gradient norm at step {step}")
            opt.step()
            for parameter_name, parameter in model.named_parameters():
                if not bool(torch.isfinite(parameter).all().item()):
                    raise FloatingPointError(
                        f"non-finite parameter {parameter_name} at step {step}")
            if step % 100 == 0:
                loss_curve.append({"step": step, "loss": round(loss_value, 5)})
            final_loss = loss_value
            step += 1
    if final_loss is not None and (
            not loss_curve or loss_curve[-1]["step"] != total_steps - 1):
        loss_curve.append({"step": total_steps - 1,
                           "loss": round(final_loss, 5)})
    wall = time.perf_counter() - t0
    res = eval_full_v2(model, device)
    rec["final_eval"] = res
    rec["final_probe_loss"] = round(
        fixed_probe_loss(
            model, bank, order, device,
            class_weights=class_weights, all_batches=full_bank_probe,
        ), 5)
    rec["initial_loss"] = round(initial_loss, 5) if initial_loss is not None else None
    rec["final_loss"] = round(final_loss, 5) if final_loss is not None else None
    rec["max_abs_belief"] = round(max_abs_belief, 5)
    rec["max_abs_thought"] = round(max_abs_thought, 5)
    rec["loss_curve"] = loss_curve
    rec["wall_s"] = round(wall, 1)
    return rec


def classify(mean_lift, sigma):
    delta = mean_lift - REFERENCE_R
    if delta >= 0.05 and sigma <= 0.06:
        return "ALPHA_contract_fix_works", round(delta, 4)
    if abs(delta) <= 0.02:
        return "BETA_parity_intake_bottleneck", round(delta, 4)
    if delta <= -0.03:
        return "GAMMA_regression_stop", round(delta, 4)
    return "AMBIGUOUS", round(delta, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="override seed list (for smoke testing only)")
    args = ap.parse_args()

    apply_deterministic_mode()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bank_check = build_bank_pinned(42, TASKS)
    bdig = bank_digest(bank_check)
    assert bdig == "b3bb5fc33fd5f605", f"bank digest drift: {bdig}"

    seeds = args.seeds if args.seeds else SEEDS
    legacy_config = CoreV2Config(
        decision_aggregation="legacy_scalar_utility_v0")
    results = {"prereg": "2026-08-24-core-v2-deployed-loss-baseline",
               "reference_R": REFERENCE_R, "bank_digest": bdig,
               "seeds": seeds, "arms": {}}

    for flags, name in [(CONFIG_A_DECISION_ONLY, "V2A_decision_only"),
                        (CONFIG_C_FULL, "V2C_full")]:
        recs = []
        for s in seeds:
            print(f"[{name}] seed {s} ...", flush=True)
            recs.append(run_v2_arm(
                flags, name, s, device, bdig, config=legacy_config))
        lifts = [r["final_eval"]["mean_lift"] for r in recs]
        ml, sig = float(np.mean(lifts)), float(np.std(lifts))
        cls, delta = classify(ml, sig)
        results["arms"][name] = {
            "runs": recs, "mean_lift": round(ml, 4),
            "sigma": round(sig, 4), "delta_vs_V1": delta,
            "classification": cls,
        }
        print(f"[{name}] mean={ml:.4f} σ={sig:.4f} Δ={delta:+.4f} -> {cls}", flush=True)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"DONE -> {args.output}")


if __name__ == "__main__":
    main()
