"""Stage V2.0j: preregistered full-cycle accumulated macro policy."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch import nn

from irene_brain.evaluation.torture_suite import TASKS
from irene_brain.v2 import CONFIG_A_DECISION_ONLY, CONFIG_C_FULL, CoreV2Config, CoreV2Model
from irene_brain.v2.losses import deployed_decision_loss
from run_provenance import apply_deterministic_mode, bank_digest, build_bank_pinned
from stage_v20_core_v2_deployed_loss_baseline import (
    REFERENCE_R, SEEDS, build_order, classify, eval_full_v2, frames_tensor,
    param_digest, provenance,
)
from stage_v20e_global_macro_policy import BANK_DIGEST, LOSS_NORMALIZATION, macro_weights
from stage_v20f_order_diagnostic import probe_bank


SMOKE_CYCLES = 32
FULL_CYCLES = 128
SMOKE_PROBE_EVERY = 4
FULL_PROBE_EVERY = 16
LEARNING_RATE = 1e-4
ORDER_SEED = 20260824


def write_atomic(path, value):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_arm(flags, arm_name, seed, cycles, probe_every, device, bank, order,
            digest, weights):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.cuda.manual_seed_all(seed)
    config = CoreV2Config(
        decision_aggregation="direct_mean_logits_v1",
        braincell_dynamics="normalized_mixture_v1",
        belief_dynamics="convex_gated_v1",
    )
    model = CoreV2Model(config=config, flags=flags).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    model.train()
    record = {
        "arm": arm_name,
        "seed": seed,
        "status": "completed",
        "cycles": cycles,
        "batch_exposures": cycles * len(order),
        "optimizer_steps": cycles,
        "learning_rate": LEARNING_RATE,
        "clip_scope": "complete_47_batch_cycle_v1",
        "clip_norm": 1.0,
        "class_weight_normalization": LOSS_NORMALIZATION,
        "class_weights": weights.cpu().tolist(),
        "decision_aggregation": config.decision_aggregation,
        "braincell_dynamics": config.braincell_dynamics,
        "belief_dynamics": config.belief_dynamics,
        "init_param_digest": param_digest(model),
        "provenance": provenance(
            model=model, train_seed=seed, eval_seed=20260822,
            bank_digest=digest, deterministic=True),
        "initial_probe": probe_bank(model, bank, order, device, weights),
        "cycle_telemetry": [],
        "probes": [],
    }
    max_abs_belief = 0.0
    max_abs_thought = 0.0
    start = time.perf_counter()
    for cycle in range(cycles):
        positions = list(range(len(order)))
        random.Random(ORDER_SEED + seed * 1009 + cycle).shuffle(positions)
        optimizer.zero_grad(set_to_none=True)
        mean_training_loss = 0.0
        for order_index in positions:
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
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"non-finite loss at {arm_name} seed {seed} cycle {cycle}")
            mean_training_loss += loss_value / len(order)
            (loss / len(order)).backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(
                f"non-finite gradient at {arm_name} seed {seed} cycle {cycle}")
        optimizer.step()
        for parameter_name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter).all().item()):
                raise FloatingPointError(
                    f"non-finite parameter {parameter_name} at cycle {cycle}")
        record["cycle_telemetry"].append({
            "cycle": cycle + 1,
            "mean_training_loss": round(mean_training_loss, 5),
            "preclip_gradient_norm": round(gradient_norm, 5),
        })
        if (cycle + 1) % probe_every == 0:
            probe = {
                "cycle": cycle + 1,
                "batch_exposures": (cycle + 1) * len(order),
                **probe_bank(model, bank, order, device, weights),
            }
            record["probes"].append(probe)
            print(
                f"[{arm_name} seed={seed}] cycle={cycle + 1:03d} "
                f"macro={probe['macro_loss']:.5f} "
                f"recall={probe['per_class_recall']}",
                flush=True,
            )
    record["final_probe"] = record["probes"][-1]
    record["final_eval"] = eval_full_v2(model, device)
    record["max_abs_belief"] = round(max_abs_belief, 5)
    record["max_abs_thought"] = round(max_abs_thought, 5)
    record["wall_s"] = round(time.perf_counter() - start, 1)
    return record


def summarize(records, expected_runs=None):
    completed = [record for record in records if record.get("status") == "completed"]
    if len(completed) != len(records):
        return {"status": "training_failure", "runs": records}
    lifts = [record["final_eval"]["mean_lift"] for record in records]
    mean_lift = float(np.mean(lifts))
    sigma = float(np.std(lifts))
    classification, delta = classify(mean_lift, sigma)
    return {
        "status": (
            "completed"
            if expected_runs is None or len(records) == expected_runs
            else "in_progress"
        ),
        "runs": records,
        "mean_lift": round(mean_lift, 4),
        "sigma": round(sigma, 4),
        "delta_vs_V1": delta,
        "classification": classification,
    }


def smoke_passed(arms):
    for arm in arms.values():
        if arm.get("status") != "completed" or len(arm["runs"]) != 1:
            return False
        record = arm["runs"][0]
        initial = record["initial_probe"]["macro_loss"]
        final = record["final_probe"]["macro_loss"]
        training_actions = sum(
            count > 0 for count in record["final_probe"]["action_histogram"])
        evaluation = record["final_eval"]
        if not (
            np.isfinite(initial)
            and np.isfinite(final)
            and final < 1.45
            and final <= 0.90 * initial
            and training_actions == 5
            and evaluation["unique_actions"] >= 3
            and evaluation["balanced_accuracy"] >= 0.25
            and record["max_abs_belief"] <= 1.00001
            and record["max_abs_thought"] <= 16.0
        ):
            return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    apply_deterministic_mode()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bank = build_bank_pinned(42, TASKS)
    digest = bank_digest(bank)
    if digest != BANK_DIGEST:
        raise RuntimeError(f"bank digest drift: {digest}")
    order = build_order(bank)
    weights = torch.tensor(macro_weights(), dtype=torch.float32, device=device)
    cycles = SMOKE_CYCLES if args.smoke else FULL_CYCLES
    probe_every = SMOKE_PROBE_EVERY if args.smoke else FULL_PROBE_EVERY
    seeds = [42] if args.smoke else SEEDS
    result = {
        "schema_version": 1,
        "preregistration": "2026-08-24-core-v2-accumulated-macro-policy-prereg",
        "mode": "smoke" if args.smoke else "confirmatory",
        "reference_R": REFERENCE_R,
        "bank_digest": digest,
        "cycles": cycles,
        "batch_exposures_per_run": cycles * len(order),
        "probe_every_cycles": probe_every,
        "seeds": seeds,
        "learning_rate": LEARNING_RATE,
        "clip_scope": "complete_47_batch_cycle_v1",
        "loss_normalization": LOSS_NORMALIZATION,
        "class_weights": weights.cpu().tolist(),
        "arms": {},
    }
    output = os.path.abspath(args.output)
    for flags, name in (
        (CONFIG_A_DECISION_ONLY, "V2A_accumulated_macro"),
        (CONFIG_C_FULL, "V2C_accumulated_macro"),
    ):
        records = []
        for seed in seeds:
            print(f"[{name}] seed={seed} cycles={cycles}", flush=True)
            try:
                records.append(run_arm(
                    flags, name, seed, cycles, probe_every,
                    device, bank, order, digest, weights))
            except FloatingPointError as error:
                records.append({
                    "arm": name,
                    "seed": seed,
                    "status": "nonfinite",
                    "error": str(error),
                })
            result["arms"][name] = summarize(records, expected_runs=len(seeds))
            write_atomic(output, result)
        result["arms"][name] = summarize(records, expected_runs=len(seeds))
    if args.smoke:
        result["smoke_gate"] = {
            "maximum_final_macro_probe_loss": 1.45,
            "minimum_relative_macro_probe_reduction": 0.10,
            "required_training_actions": 5,
            "minimum_heldout_actions": 3,
            "minimum_heldout_balanced_accuracy": 0.25,
            "maximum_absolute_belief": 1.00001,
            "maximum_absolute_thought": 16.0,
            "passed": smoke_passed(result["arms"]),
        }
    write_atomic(output, result)
    print(f"DONE -> {output}", flush=True)
    failed = any(arm["status"] != "completed" for arm in result["arms"].values())
    if failed or (args.smoke and not result["smoke_gate"]["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
