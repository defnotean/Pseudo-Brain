"""Stage V2.1: bounded predictive trajectory smoke on maze chase."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
import os
import time

import torch
from torch import nn

from irene_brain.training.batches import MazeChaseBatchConfig, MazeChaseBatchSource
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Config, CoreV2Model
from irene_brain.v2.trajectory_objective import V2TrajectoryObjective
from run_provenance import apply_deterministic_mode
from stage_v20_core_v2_deployed_loss_baseline import param_digest, provenance


DATASET_MANIFEST = "7c7c12e29011f5bd5d7436fc5aa31c42ccf983d8abd876f18991af4926f3f72b"
TRAIN_SEED = 42
EVAL_SEED = 20260825
EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
CLIP_NORM = 1.0


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


def save_checkpoint(model, config, arm_name, source, directory):
    os.makedirs(directory, exist_ok=True)
    path = os.path.abspath(os.path.join(directory, f"{arm_name}.pt"))
    temporary = path + ".tmp"
    torch.save(
        {
            "schema_version": 1,
            "arm": arm_name,
            "seed": TRAIN_SEED,
            "dataset_manifest": source.manifest_sha256,
            "config": asdict(config),
            "flags": asdict(CONFIG_B_PREDICTIVE),
            "model_state_dict": model.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path, "sha256": digest.hexdigest()}


def dataset_config(mini=False):
    if mini:
        return MazeChaseBatchConfig(
            train_sequences=2,
            validation_sequences=1,
            test_sequences=1,
            sequence_length=3,
            burn_in_steps=1,
            seed_offset=8_388_608,
            ghost_count=5,
            ghost_period=1,
            ghost_rule="direct",
        )
    return MazeChaseBatchConfig(
        train_sequences=96,
        validation_sequences=32,
        test_sequences=32,
        sequence_length=16,
        burn_in_steps=4,
        seed_offset=8_388_608,
        ghost_count=5,
        ghost_period=1,
        ghost_rule="direct",
    )


def model_config(predictive):
    return CoreV2Config(
        decision_aggregation="direct_mean_logits_v1",
        braincell_dynamics="normalized_mixture_v1",
        belief_dynamics="convex_gated_v1",
        hazard_parameterization="probability_sigmoid_v1",
        next_weight=0.5 if predictive else 0.0,
        reward_weight=0.25 if predictive else 0.0,
        hazard_weight=0.25 if predictive else 0.0,
    )


def preserve_rng():
    return (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def restore_rng(state):
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


@torch.no_grad()
def evaluate(objective, source, device, split="validation"):
    saved_rng = preserve_rng()
    torch.manual_seed(EVAL_SEED)
    torch.cuda.manual_seed_all(EVAL_SEED)
    objective.eval()
    total_samples = 0
    loss_sum = 0.0
    component_sums = {}
    metric_sums = {}
    turn_correct = 0.0
    turn_samples = 0.0
    hazard_positives = 0.0
    hazard_negatives = 0.0
    hazard_positive_probability_sum = 0.0
    hazard_negative_probability_sum = 0.0
    max_abs_belief = 0.0
    max_abs_thought = 0.0
    batch_size = min(BATCH_SIZE, source.config.validation_sequences)
    for batch in source.iter_batches(
        split=split,
        epoch=0,
        start_batch=0,
        batch_size=batch_size,
    ):
        result = objective(batch)
        samples = result.samples
        total_samples += samples
        loss_sum += float(result.loss) * samples
        for name, value in result.components.items():
            component_sums[name] = component_sums.get(name, 0.0) + float(value) * samples
        for name in ("action_accuracy", "reward_mae", "hazard_brier", "next_latent_cosine"):
            metric_sums[name] = metric_sums.get(name, 0.0) + float(result.metrics[name]) * samples
        local_turn_samples = float(result.metrics["turn_samples"])
        turn_samples += local_turn_samples
        turn_correct += float(result.metrics["turn_accuracy"]) * local_turn_samples
        local_hazard_positives = float(result.metrics["hazard_positives"])
        local_hazard_negatives = float(result.metrics["hazard_negatives"])
        hazard_positives += local_hazard_positives
        hazard_negatives += local_hazard_negatives
        hazard_positive_probability_sum += (
            float(result.metrics["hazard_positive_probability"])
            * local_hazard_positives
        )
        hazard_negative_probability_sum += (
            float(result.metrics["hazard_negative_probability"])
            * local_hazard_negatives
        )
        max_abs_belief = max(max_abs_belief, float(result.metrics["max_abs_belief"]))
        max_abs_thought = max(max_abs_thought, float(result.metrics["max_abs_thought"]))
    restore_rng(saved_rng)
    objective.train()
    return {
        "samples": total_samples,
        "loss": round(loss_sum / total_samples, 6),
        "components": {
            name: round(value / total_samples, 6)
            for name, value in sorted(component_sums.items())
        },
        "metrics": {
            **{
                name: round(value / total_samples, 6)
                for name, value in sorted(metric_sums.items())
            },
            "turn_accuracy": round(turn_correct / max(turn_samples, 1.0), 6),
            "turn_samples": int(turn_samples),
            "hazard_positives": int(hazard_positives),
            "hazard_negatives": int(hazard_negatives),
            "hazard_positive_probability": round(
                hazard_positive_probability_sum / max(hazard_positives, 1.0),
                6,
            ),
            "hazard_negative_probability": round(
                hazard_negative_probability_sum / max(hazard_negatives, 1.0),
                6,
            ),
            "max_abs_belief": round(max_abs_belief, 6),
            "max_abs_thought": round(max_abs_thought, 6),
        },
    }


def run_arm(name, predictive, intervention, source, device, epochs, checkpoint_dir):
    torch.manual_seed(TRAIN_SEED)
    torch.cuda.manual_seed_all(TRAIN_SEED)
    config = model_config(predictive)
    model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE).to(device)
    objective = V2TrajectoryObjective(
        model,
        flags=CONFIG_B_PREDICTIVE,
        prediction_error_intervention=intervention,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=1e-4)
    initial_digest = param_digest(model)
    initial_validation = evaluate(objective, source, device)
    torch.manual_seed(TRAIN_SEED + 10_000)
    torch.cuda.manual_seed_all(TRAIN_SEED + 10_000)
    telemetry = []
    start = time.perf_counter()
    max_abs_belief = 0.0
    max_abs_thought = 0.0
    optimizer_steps = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        max_preclip_norm = 0.0
        for batch in source.iter_batches(
            split="train",
            epoch=epoch,
            start_batch=0,
            batch_size=min(BATCH_SIZE, source.config.train_sequences),
        ):
            result = objective(batch)
            loss_value = float(result.loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"non-finite loss in {name} epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            gradient_norm = float(nn.utils.clip_grad_norm_(trainable, CLIP_NORM))
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient in {name} epoch {epoch}")
            optimizer.step()
            objective.update_target_encoder()
            optimizer_steps += 1
            epoch_loss += loss_value * result.samples
            epoch_samples += result.samples
            max_preclip_norm = max(max_preclip_norm, gradient_norm)
            max_abs_belief = max(max_abs_belief, float(result.metrics["max_abs_belief"]))
            max_abs_thought = max(max_abs_thought, float(result.metrics["max_abs_thought"]))
        validation = evaluate(objective, source, device)
        telemetry.append({
            "epoch": epoch + 1,
            "training_loss": round(epoch_loss / epoch_samples, 6),
            "max_preclip_gradient_norm": round(max_preclip_norm, 6),
            "validation": validation,
        })
        print(
            f"[{name}] epoch={epoch + 1} train={epoch_loss / epoch_samples:.5f} "
            f"val={validation['loss']:.5f} turn={validation['metrics']['turn_accuracy']:.4f} "
            f"next={validation['components']['next_latent']:.5f}",
            flush=True,
        )
    for parameter_name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all().item()):
            raise FloatingPointError(f"non-finite parameter {parameter_name} in {name}")
    checkpoint = save_checkpoint(model, config, name, source, checkpoint_dir)
    return {
        "status": "completed",
        "arm": name,
        "predictive_losses": predictive,
        "prediction_error_intervention": intervention,
        "seed": TRAIN_SEED,
        "init_param_digest": initial_digest,
        "optimizer_steps": optimizer_steps,
        "initial_validation": initial_validation,
        "final_validation": telemetry[-1]["validation"],
        "telemetry": telemetry,
        "max_abs_belief": round(max_abs_belief, 6),
        "max_abs_thought": round(max_abs_thought, 6),
        "wall_s": round(time.perf_counter() - start, 1),
        "checkpoint": checkpoint,
        "provenance": provenance(
            model=model,
            train_seed=TRAIN_SEED,
            eval_seed=EVAL_SEED,
            bank_digest=source.manifest_sha256,
            deterministic=True,
        ),
    }


def smoke_gate(arms):
    if set(arms) != {"decision_only_zero_pe", "predictive_zero_pe", "predictive_normal_pe"}:
        return False
    if any(arm.get("status") != "completed" for arm in arms.values()):
        return False
    init_digests = {
        json.dumps(arm["init_param_digest"], sort_keys=True, separators=(",", ":"))
        for arm in arms.values()
    }
    if len(init_digests) != 1:
        return False
    for arm in arms.values():
        final = arm["final_validation"]
        if not (
            final["metrics"]["turn_samples"] > 0
            and final["metrics"]["hazard_positives"] >= 10
            and arm["max_abs_belief"] <= 1.00001
            and arm["max_abs_thought"] <= 16.0
        ):
            return False
    for name in ("predictive_zero_pe", "predictive_normal_pe"):
        arm = arms[name]
        initial = arm["initial_validation"]["components"]
        final = arm["final_validation"]["components"]
        if not (
            final["next_latent"] <= 0.85 * initial["next_latent"]
            and final["reward"] <= 0.85 * initial["reward"]
            and final["hazard"] <= 0.85 * initial["hazard"]
            and (
                arm["final_validation"]["metrics"]["hazard_positive_probability"]
                >= arm["final_validation"]["metrics"]["hazard_negative_probability"]
                + 0.02
            )
        ):
            return False
    control_turn = arms["decision_only_zero_pe"]["final_validation"]["metrics"]["turn_accuracy"]
    feedback_turn = arms["predictive_normal_pe"]["final_validation"]["metrics"]["turn_accuracy"]
    return feedback_turn >= control_turn - 0.10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--mini", action="store_true")
    args = parser.parse_args()
    apply_deterministic_mode()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source = MazeChaseBatchSource(dataset_config(mini=args.mini))
    if not args.mini and source.manifest_sha256 != DATASET_MANIFEST:
        raise RuntimeError(f"dataset manifest drift: {source.manifest_sha256}")
    epochs = 1 if args.mini else EPOCHS
    result = {
        "schema_version": 1,
        "preregistration": "2026-08-24-core-v2-predictive-trajectory-smoke-prereg",
        "mode": "mini_integration" if args.mini else "smoke",
        "device": str(device),
        "dataset_manifest": source.manifest_sha256,
        "epochs": epochs,
        "batch_size": min(BATCH_SIZE, source.config.train_sequences),
        "learning_rate": LEARNING_RATE,
        "clip_norm": CLIP_NORM,
        "arms": {},
    }
    output = os.path.abspath(args.output)
    checkpoint_dir = output + ".checkpoints"
    for name, predictive, intervention in (
        ("decision_only_zero_pe", False, "zero"),
        ("predictive_zero_pe", True, "zero"),
        ("predictive_normal_pe", True, "normal"),
    ):
        try:
            result["arms"][name] = run_arm(
                name,
                predictive,
                intervention,
                source,
                device,
                epochs,
                checkpoint_dir,
            )
        except FloatingPointError as error:
            result["arms"][name] = {
                "status": "nonfinite",
                "error": str(error),
            }
        write_atomic(output, result)
    if not args.mini:
        result["smoke_gate"] = {
            "minimum_validation_hazard_positives": 10,
            "maximum_predictive_loss_ratio": 0.85,
            "minimum_hazard_probability_margin": 0.02,
            "maximum_turn_accuracy_regression": 0.10,
            "maximum_absolute_belief": 1.00001,
            "maximum_absolute_thought": 16.0,
            "passed": smoke_gate(result["arms"]),
        }
    write_atomic(output, result)
    print(f"DONE -> {output}", flush=True)
    failed = any(arm.get("status") != "completed" for arm in result["arms"].values())
    if failed or (not args.mini and not result["smoke_gate"]["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
