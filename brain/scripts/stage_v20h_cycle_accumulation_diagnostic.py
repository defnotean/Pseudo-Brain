"""Stage V2.0h diagnostic: clip the accumulated full-cycle macro gradient once."""
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
from irene_brain.v2 import CONFIG_A_DECISION_ONLY, CoreV2Config, CoreV2Model
from irene_brain.v2.losses import deployed_decision_loss
from run_provenance import apply_deterministic_mode, bank_digest, build_bank_pinned
from stage_v20_core_v2_deployed_loss_baseline import (
    build_order, eval_full_v2, frames_tensor, param_digest, provenance,
)
from stage_v20e_global_macro_policy import BANK_DIGEST, LOSS_NORMALIZATION, macro_weights
from stage_v20f_order_diagnostic import probe_bank


CYCLES = 16
ORDER_SEED = 20260824


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
    result = {
        "schema_version": 1,
        "diagnostic": "full_cycle_macro_gradient_accumulation",
        "claim_status": "diagnostic_only_not_preregistered",
        "bank_digest": digest,
        "cycles": CYCLES,
        "batch_exposures": CYCLES * len(order),
        "optimizer_steps": CYCLES,
        "clip_norm": 1.0,
        "loss_normalization": LOSS_NORMALIZATION,
        "class_weights": weights.cpu().tolist(),
        "init_param_digest": param_digest(model),
        "provenance": provenance(
            model=model, train_seed=seed, eval_seed=20260822,
            bank_digest=digest, deterministic=True),
        "initial_probe": probe_bank(model, bank, order, device, weights),
        "cycle_probes": [],
    }
    max_abs_belief = 0.0
    max_abs_thought = 0.0
    start = time.perf_counter()
    for cycle in range(CYCLES):
        positions = list(range(len(order)))
        random.Random(ORDER_SEED + cycle).shuffle(positions)
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
                    f"non-finite loss at cycle {cycle} batch {order_index}")
            mean_training_loss += loss_value / len(order)
            (loss / len(order)).backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite cycle gradient at cycle {cycle}")
        optimizer.step()
        for parameter_name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter).all().item()):
                raise FloatingPointError(
                    f"non-finite parameter {parameter_name} at cycle {cycle}")
        result["cycle_probes"].append({
            "cycle": cycle + 1,
            "batch_exposures": (cycle + 1) * len(order),
            "mean_training_loss": round(mean_training_loss, 5),
            "preclip_gradient_norm": round(gradient_norm, 5),
            **probe_bank(model, bank, order, device, weights),
        })
        probe = result["cycle_probes"][-1]
        print(
            f"cycle={cycle + 1:02d} macro={probe['macro_loss']:.5f} "
            f"recall={probe['per_class_recall']}",
            flush=True,
        )
    result["final_eval"] = eval_full_v2(model, device)
    result["max_abs_belief"] = round(max_abs_belief, 5)
    result["max_abs_thought"] = round(max_abs_thought, 5)
    result["wall_s"] = round(time.perf_counter() - start, 1)
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
