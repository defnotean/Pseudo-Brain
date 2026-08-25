"""Stage V2.0f diagnostic: fixed batch order versus deterministic reshuffling."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from irene_brain.evaluation.torture_suite import TASKS
from irene_brain.v2 import CONFIG_A_DECISION_ONLY, CoreV2Config, CoreV2Model
from irene_brain.v2.losses import deployed_decision_loss
from run_provenance import apply_deterministic_mode, bank_digest, build_bank_pinned
from stage_v20_core_v2_deployed_loss_baseline import (
    build_order, eval_full_v2, frames_tensor, param_digest, provenance,
)
from stage_v20e_global_macro_policy import BANK_DIGEST, LOSS_NORMALIZATION, macro_weights


CYCLES = 8
ORDER_SEED = 20260824


@torch.no_grad()
def probe_bank(model, bank, order, device, weights, seed=424242):
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model.eval()
        weighted_numerator = torch.zeros((), device=device)
        weighted_denominator = torch.zeros((), device=device)
        class_nll = torch.zeros(5, device=device)
        class_count = torch.zeros(5, device=device)
        confusion = torch.zeros(5, 5, dtype=torch.int64, device=device)
        for length, group in order:
            episodes = [bank[index] for index in group]
            frames = torch.stack([
                torch.stack([frames_tensor(episode, device)[tick]
                             for tick in range(length)])
                for episode in episodes
            ])
            labels = torch.tensor(
                [episode.label for episode in episodes], dtype=torch.long, device=device)
            state = model.init_state(len(episodes), device)
            output = None
            for tick in range(length):
                output, state = model(frames[:, tick], state)
            log_probs = F.log_softmax(output.decision.action_values, dim=-1)
            losses = F.nll_loss(log_probs, labels, reduction="none")
            sample_weights = weights[labels]
            weighted_numerator += (losses * sample_weights).sum()
            weighted_denominator += sample_weights.sum()
            class_nll.scatter_add_(0, labels, losses)
            class_count.scatter_add_(0, labels, torch.ones_like(losses))
            predictions = output.decision.action_values.argmax(dim=-1)
            flat = labels * 5 + predictions
            confusion += torch.bincount(flat, minlength=25).reshape(5, 5)
        recalls = confusion.diag().float() / confusion.sum(dim=1).clamp_min(1)
        return {
            "macro_loss": round(float((weighted_numerator / weighted_denominator).item()), 5),
            "per_class_nll": [round(float(value), 5) for value in (class_nll / class_count)],
            "per_class_recall": [round(float(value), 4) for value in recalls],
            "action_histogram": confusion.sum(dim=0).cpu().tolist(),
        }
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        model.train(was_training)


def run_arm(schedule_name, device, bank, order, digest, weights):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = CoreV2Config(
        decision_aggregation="direct_mean_logits_v1",
        braincell_dynamics="normalized_mixture_v1",
        belief_dynamics="convex_gated_v1",
    )
    model = CoreV2Model(config=config, flags=CONFIG_A_DECISION_ONLY).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    model.train()
    record = {
        "schedule": schedule_name,
        "seed": seed,
        "order_seed": ORDER_SEED if schedule_name == "reshuffle_each_cycle_v1" else None,
        "cycles": CYCLES,
        "batch_exposures": CYCLES * len(order),
        "init_param_digest": param_digest(model),
        "provenance": provenance(
            model=model, train_seed=seed, eval_seed=20260822,
            bank_digest=digest, deterministic=True),
        "initial_probe": probe_bank(model, bank, order, device, weights),
        "cycle_probes": [],
        "rare_block_probes": [],
    }
    start = time.perf_counter()
    step = 0
    max_abs_belief = 0.0
    max_abs_thought = 0.0
    for cycle in range(CYCLES):
        positions = list(range(len(order)))
        if schedule_name == "reshuffle_each_cycle_v1":
            random.Random(ORDER_SEED + cycle).shuffle(positions)
        rare_positions = [
            position for position, order_index in enumerate(positions)
            if any(int(bank[index].label) == 0 for index in order[order_index][1])
        ]
        last_rare_position = max(rare_positions)
        for position, order_index in enumerate(positions):
            length, group = order[order_index]
            episodes = [bank[index] for index in group]
            frames = torch.stack([
                torch.stack([frames_tensor(episode, device)[tick]
                             for tick in range(length)])
                for episode in episodes
            ])
            labels = torch.tensor(
                [episode.label for episode in episodes], dtype=torch.long, device=device)
            state = model.init_state(len(episodes), device)
            output = None
            for tick in range(length):
                output, state = model(frames[:, tick], state)
                max_abs_belief = max(
                    max_abs_belief, float(output.belief.detach().abs().max().item()))
                max_abs_thought = max(
                    max_abs_thought, float(output.thoughts.detach().abs().max().item()))
            loss = deployed_decision_loss(
                output, labels, weights,
                weight_normalization=LOSS_NORMALIZATION,
            )
            loss_value = float(loss.item())
            if not np.isfinite(loss_value):
                raise FloatingPointError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
            if not np.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient at step {step}")
            optimizer.step()
            step += 1
            if position == last_rare_position:
                record["rare_block_probes"].append({
                    "cycle": cycle + 1,
                    "step": step,
                    "position": position,
                    **probe_bank(model, bank, order, device, weights),
                })
        record["cycle_probes"].append({
            "cycle": cycle + 1,
            "step": step,
            **probe_bank(model, bank, order, device, weights),
        })
    record["final_eval"] = eval_full_v2(model, device)
    record["max_abs_belief"] = round(max_abs_belief, 5)
    record["max_abs_thought"] = round(max_abs_thought, 5)
    record["wall_s"] = round(time.perf_counter() - start, 1)
    return record


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
    result = {
        "schema_version": 1,
        "diagnostic": "fixed_order_vs_deterministic_reshuffling",
        "claim_status": "diagnostic_only_not_preregistered",
        "bank_digest": digest,
        "class_weights": weights.cpu().tolist(),
        "loss_normalization": LOSS_NORMALIZATION,
        "arms": {},
    }
    for schedule in ("fixed_order_v0", "reshuffle_each_cycle_v1"):
        print(f"[{schedule}] {CYCLES} complete cycles ...", flush=True)
        result["arms"][schedule] = run_arm(
            schedule, device, bank, order, digest, weights)
        print(
            f"[{schedule}] final macro loss "
            f"{result['arms'][schedule]['cycle_probes'][-1]['macro_loss']}",
            flush=True,
        )
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(f"DONE -> {output}", flush=True)


if __name__ == "__main__":
    main()
