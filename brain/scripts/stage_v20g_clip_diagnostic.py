"""Stage V2.0g diagnostic: quantify class weighting before and after clipping."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from irene_brain.evaluation.torture_suite import TASKS
from irene_brain.v2 import CONFIG_A_DECISION_ONLY, CoreV2Config, CoreV2Model
from irene_brain.v2.losses import deployed_decision_loss
from run_provenance import apply_deterministic_mode, bank_digest, build_bank_pinned
from stage_v20_core_v2_deployed_loss_baseline import build_order, frames_tensor
from stage_v20e_global_macro_policy import BANK_DIGEST, LOSS_NORMALIZATION, macro_weights


def gradient_norm(model):
    squared = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += parameter.grad.detach().square().sum()
    return float(torch.sqrt(squared).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    apply_deterministic_mode()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bank = build_bank_pinned(42, TASKS)
    digest = bank_digest(bank)
    if digest != BANK_DIGEST:
        raise RuntimeError(f"bank digest drift: {digest}")
    order = build_order(bank)
    weights = torch.tensor(macro_weights(), dtype=torch.float32, device=device)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    config = CoreV2Config(
        decision_aggregation="direct_mean_logits_v1",
        braincell_dynamics="normalized_mixture_v1",
        belief_dynamics="convex_gated_v1",
    )
    model = CoreV2Model(config=config, flags=CONFIG_A_DECISION_ONLY).to(device)
    model.train()
    records = []
    preclip_mass = np.zeros(5, dtype=np.float64)
    postclip_mass = np.zeros(5, dtype=np.float64)
    weight_aware_mass = np.zeros(5, dtype=np.float64)
    for position, (length, group) in enumerate(order):
        episodes = [bank[index] for index in group]
        frames = torch.stack([
            torch.stack([frames_tensor(episode, device)[tick]
                         for tick in range(length)])
            for episode in episodes
        ])
        labels = torch.tensor(
            [episode.label for episode in episodes], dtype=torch.long, device=device)
        torch.manual_seed(100_000 + position)
        torch.cuda.manual_seed_all(100_000 + position)
        state = model.init_state(len(episodes), device)
        output = None
        for tick in range(length):
            output, state = model(frames[:, tick], state)
        unweighted_loss = F.nll_loss(
            F.log_softmax(output.decision.action_values, dim=-1), labels)
        weighted_loss = deployed_decision_loss(
            output, labels, weights, weight_normalization=LOSS_NORMALIZATION)

        model.zero_grad(set_to_none=True)
        unweighted_loss.backward(retain_graph=True)
        unweighted_norm = gradient_norm(model)
        model.zero_grad(set_to_none=True)
        weighted_loss.backward()
        weighted_norm = gradient_norm(model)
        if not math.isfinite(unweighted_norm) or not math.isfinite(weighted_norm):
            raise FloatingPointError(f"non-finite gradient norm at batch {position}")

        fixed_clip_factor = min(1.0, 1.0 / max(weighted_norm, 1e-30))
        mean_selected_weight = float(weights[labels].mean().item())
        aware_threshold = mean_selected_weight
        aware_clip_factor = min(1.0, aware_threshold / max(weighted_norm, 1e-30))
        counts = Counter(int(label) for label in labels.cpu().tolist())
        for action, count in counts.items():
            coefficient = count * float(weights[action].item()) / len(group)
            preclip_mass[action] += coefficient
            postclip_mass[action] += coefficient * fixed_clip_factor
            weight_aware_mass[action] += coefficient * aware_clip_factor
        records.append({
            "position": position,
            "length": length,
            "class_counts": [counts[action] for action in range(5)],
            "unweighted_loss": round(float(unweighted_loss.item()), 6),
            "weighted_loss": round(float(weighted_loss.item()), 6),
            "unweighted_gradient_norm": round(unweighted_norm, 6),
            "weighted_gradient_norm": round(weighted_norm, 6),
            "fixed_clip_factor": round(fixed_clip_factor, 6),
            "mean_selected_weight": round(mean_selected_weight, 6),
            "weight_aware_threshold": round(aware_threshold, 6),
            "weight_aware_clip_factor": round(aware_clip_factor, 6),
        })

    def shares(values):
        return [round(float(value / values.sum()), 6) for value in values]

    class_zero_batches = [record for record in records if record["class_counts"][0]]
    result = {
        "schema_version": 1,
        "diagnostic": "macro_weight_gradient_clipping",
        "claim_status": "diagnostic_only_not_preregistered",
        "bank_digest": digest,
        "class_weights": weights.cpu().tolist(),
        "clip_norm": 1.0,
        "preclip_nominal_class_shares": shares(preclip_mass),
        "postclip_scalar_approximation_class_shares": shares(postclip_mass),
        "weight_aware_scalar_approximation_class_shares": shares(weight_aware_mass),
        "batches_clipped_fixed": sum(record["fixed_clip_factor"] < 1.0 for record in records),
        "batches_clipped_weight_aware": sum(
            record["weight_aware_clip_factor"] < 1.0 for record in records),
        "class_zero_batches": class_zero_batches,
        "batches": records,
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(json.dumps({
        "preclip": result["preclip_nominal_class_shares"],
        "fixed_postclip": result["postclip_scalar_approximation_class_shares"],
        "weight_aware_postclip": result["weight_aware_scalar_approximation_class_shares"],
        "fixed_clipped": result["batches_clipped_fixed"],
        "aware_clipped": result["batches_clipped_weight_aware"],
        "class_zero_batches": class_zero_batches,
    }, indent=2), flush=True)
    print(f"DONE -> {output}", flush=True)


if __name__ == "__main__":
    main()
