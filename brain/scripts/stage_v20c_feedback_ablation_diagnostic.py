"""Fail-fast diagnosis for the Stage V2.0b V2-C non-finite failure.

This is diagnostic, not a confirmatory experiment.  It separates the feature
chain and reports the first non-finite output, gradient, or parameter instead
of allowing hundreds of NaN updates to continue.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from irene_brain.evaluation.torture_suite import TASKS
from irene_brain.v2 import CoreV2Config, CoreV2Model, FeatureFlags
from irene_brain.v2.losses import deployed_decision_loss
from run_provenance import apply_deterministic_mode, bank_digest, build_bank_pinned
from stage_v20_core_v2_deployed_loss_baseline import build_order, frames_tensor


BANK_DIGEST = "b3bb5fc33fd5f605"

ARMS = {
    "consequence_only": FeatureFlags(
        consequence_learning=True,
    ),
    "world_no_feedback": FeatureFlags(
        consequence_learning=True,
        world_model_learning=True,
    ),
    "error_feedback": FeatureFlags(
        consequence_learning=True,
        world_model_learning=True,
        prediction_error_feedback=True,
    ),
    "full": FeatureFlags(
        consequence_learning=True,
        world_model_learning=True,
        prediction_error_feedback=True,
        episodic_memory=True,
        session_adaptation=True,
        meta_learning=True,
    ),
}


def _stats(tensor: torch.Tensor | None) -> dict | None:
    if tensor is None:
        return None
    detached = tensor.detach()
    finite = bool(torch.isfinite(detached).all().item())
    return {
        "finite": finite,
        "max_abs": float(detached.abs().max().item()) if finite else None,
    }


def _first_bad_named(named_tensors) -> str | None:
    for name, tensor in named_tensors:
        if tensor is not None and not bool(torch.isfinite(tensor).all().item()):
            return name
    return None


def diagnose(arm: str, steps: int, seed: int, device: torch.device,
             dynamics: str, belief_dynamics: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.cuda.manual_seed_all(seed)

    config = CoreV2Config(
        decision_aggregation="direct_mean_logits_v1",
        braincell_dynamics=dynamics,
        belief_dynamics=belief_dynamics,
    )
    model = CoreV2Model(config=config, flags=ARMS[arm]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    bank = build_bank_pinned(42, TASKS)
    if bank_digest(bank) != BANK_DIGEST:
        raise RuntimeError("bank digest drift")
    order = build_order(bank)

    result = {
        "arm": arm,
        "seed": seed,
        "braincell_dynamics": dynamics,
        "belief_dynamics": belief_dynamics,
        "requested_steps": steps,
        "status": "stable",
        "first_failure": None,
        "trace": [],
    }
    started = time.perf_counter()
    step = 0
    while step < steps:
        for length, group in order:
            if step >= steps:
                break
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
            failure = None
            for tick in range(length):
                output, state = model(frames[:, tick], state)
                component_tensors = [
                    ("latent", output.latent),
                    ("belief", output.belief),
                    ("thoughts", output.thoughts),
                    ("action_logits", output.hypotheses.action_logits),
                    ("action_values", output.decision.action_values),
                    ("action_dist", output.decision.action_dist),
                    ("prediction_error", output.prediction_error.latent_error
                     if output.prediction_error is not None else None),
                    ("pending_prediction", output.pending_prediction.predicted_next_latent
                     if output.pending_prediction is not None else None),
                ]
                bad_component = _first_bad_named(component_tensors)
                if bad_component is not None:
                    failure = {
                        "phase": "forward",
                        "name": bad_component,
                        "tick": tick,
                    }
                    break
            if failure is None:
                loss = deployed_decision_loss(output, labels)
                if not bool(torch.isfinite(loss).item()):
                    failure = {"phase": "loss", "name": "deployed_loss"}
            else:
                loss = torch.tensor(float("nan"), device=device)

            if failure is None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                bad_gradient = _first_bad_named(
                    (name, parameter.grad) for name, parameter in model.named_parameters()
                )
                if bad_gradient is not None:
                    failure = {"phase": "backward", "name": bad_gradient}

            grad_norm = None
            if failure is None:
                grad_norm_tensor = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_norm = float(grad_norm_tensor.item())
                if not math.isfinite(grad_norm):
                    failure = {"phase": "gradient_norm", "name": "global"}

            if failure is None:
                optimizer.step()
                bad_parameter = _first_bad_named(model.named_parameters())
                if bad_parameter is not None:
                    failure = {"phase": "optimizer", "name": bad_parameter}

            if step % 10 == 0 or failure is not None or step == steps - 1:
                result["trace"].append({
                    "step": step,
                    "length": length,
                    "loss": float(loss.detach().item()) if torch.isfinite(loss) else None,
                    "grad_norm": grad_norm if grad_norm is not None and math.isfinite(grad_norm) else None,
                    "belief": _stats(output.belief) if output is not None else None,
                    "thoughts": _stats(output.thoughts) if output is not None else None,
                    "action_logits": _stats(output.hypotheses.action_logits)
                    if output is not None else None,
                    "prediction_error": _stats(output.prediction_error.latent_error)
                    if output is not None and output.prediction_error is not None else None,
                })

            if failure is not None:
                failure["step"] = step
                failure["length"] = length
                result["status"] = "nonfinite"
                result["first_failure"] = failure
                result["completed_steps"] = step
                result["wall_s"] = round(time.perf_counter() - started, 2)
                return result
            step += 1

    result["completed_steps"] = step
    result["wall_s"] = round(time.perf_counter() - started, 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dynamics",
        choices=("legacy_additive_v0", "normalized_mixture_v1"),
        default="legacy_additive_v0",
    )
    parser.add_argument(
        "--belief-dynamics",
        choices=("legacy_residual_v0", "convex_gated_v1"),
        default="legacy_residual_v0",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 400:
        raise ValueError("diagnostic steps must be in [1, 400]")

    apply_deterministic_mode()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = diagnose(
        args.arm, args.steps, args.seed, device,
        args.dynamics, args.belief_dynamics,
    )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    print(payload, end="", flush=True)
    if args.output:
        output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)


if __name__ == "__main__":
    main()
