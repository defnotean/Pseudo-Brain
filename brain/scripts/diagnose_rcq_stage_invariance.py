"""Bounded probes for the schema-3 frozen-stage invariance failure.

This is diagnosis only. It does not train the RCQ-v2 reference, does not write
campaign receipts, and does not construct sealed TEST ranges. Use it after a
staging canary fails ``action_outputs_sha256`` identity.

Local workstation: CPU float32, CUDA hidden.
Spark: mount the immutable release read-only and this file separately.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys


def _insert_source(source_root: Path) -> None:
    resolved = source_root.resolve()
    if not (resolved / "irene_brain" / "__init__.py").is_file():
        raise SystemExit(f"source root is not an irene_brain package: {resolved}")
    sys.path.insert(0, str(resolved))


def _snapshot_delta(reference: dict[str, object], current: dict[str, object]) -> dict[str, object]:
    keys = (
        "batch_count",
        "sequence_count",
        "timestep_count",
        "non_value_state_sha256",
        "action_outputs_sha256",
        "recurrent_states_sha256",
    )
    mismatched = [name for name in keys if reference.get(name) != current.get(name)]
    return {
        "matched": not mismatched,
        "mismatched": mismatched,
        "reference": {name: reference.get(name) for name in keys},
        "current": {name: current.get(name) for name in keys},
    }


def _build_system(config):
    from irene_brain.training.factory import build_smoke_model
    from irene_brain.training.objective import ThoughtFieldObjective
    from irene_brain.training.torch_system import TorchTrainingSystem

    objective_config = config.objective
    objective = ThoughtFieldObjective(
        build_smoke_model(config),
        action_loss_kind=objective_config.action_loss_kind,
        button_support_control_indices=objective_config.button_support_control_indices,
        button_support_weight=objective_config.button_support_weight,
        button_background_weight=objective_config.button_background_weight,
        button_background_tail_mix=objective_config.button_background_tail_mix,
        button_background_tail_temperature=objective_config.button_background_tail_temperature,
        continuous_action_weight=objective_config.continuous_action_weight,
        action_weight=objective_config.action_weight,
        value_weight=objective_config.value_weight,
        world_weight=objective_config.world_weight,
        diversity_weight=objective_config.diversity_weight,
    )
    return TorchTrainingSystem(objective, config)


def _batches(config):
    from irene_brain.training.batches import MovingShapesBatchSource

    source = MovingShapesBatchSource(config.dataset)
    train = next(
        source.iter_batches(
            split="train",
            epoch=0,
            start_batch=0,
            batch_size=config.optimization.batch_size,
            max_batches=1,
        )
    )
    validation = tuple(
        source.iter_batches(
            split="validation",
            epoch=0,
            start_batch=0,
            batch_size=config.optimization.batch_size,
            max_batches=source.batches_per_epoch(
                split="validation",
                batch_size=config.optimization.batch_size,
            ),
        )
    )
    if not validation:
        raise SystemExit("validation split produced no invariance batches")
    return train, validation


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose schema-3 frozen-stage action invariance without a campaign run."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--mode", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--allow-tf32", action="store_true")
    args = parser.parse_args(argv[1:])

    _insert_source(args.source_root)

    from irene_brain.training.config import PrecisionConfig, load_training_config

    registered = load_training_config(args.config)
    config = replace(
        registered,
        precision=PrecisionConfig(
            device=args.device,
            mode=args.mode,
            allow_tf32=bool(args.allow_tf32),
        ),
        resources=replace(
            registered.resources,
            allow_gpu=args.device == "cuda",
            cpu_threads=1 if args.device == "cpu" else registered.resources.cpu_threads,
        ),
    )
    system = _build_system(config)
    train, validation = _batches(config)
    system.train_optimizer_step((train,))

    first = system._capture_invariance_snapshot(validation)
    second = system._capture_invariance_snapshot(validation)
    probes = {"double_capture": _snapshot_delta(first, second)}

    weight_system = _build_system(config)
    weight_train, weight_validation = _batches(config)
    weight_system.train_optimizer_step((weight_train,))
    before_weights = weight_system._capture_invariance_snapshot(weight_validation)
    stage = config.stages[1]
    weight_system.objective.set_loss_weights(
        action_weight=stage.action_weight,
        value_weight=stage.value_weight,
        world_weight=stage.world_weight,
        diversity_weight=stage.diversity_weight,
    )
    after_weights = weight_system._capture_invariance_snapshot(weight_validation)
    probes["loss_weights"] = _snapshot_delta(before_weights, after_weights)

    freeze_system = _build_system(config)
    freeze_train, freeze_validation = _batches(config)
    freeze_system.train_optimizer_step((freeze_train,))
    before_freeze = freeze_system._capture_invariance_snapshot(freeze_validation)
    active = {
        "model.value_per_thought.bias",
        "model.value_per_thought.weight",
    }
    for name, parameter in freeze_system.objective.named_parameters():
        parameter.requires_grad_(name in active)
    after_freeze = freeze_system._capture_invariance_snapshot(freeze_validation)
    probes["freeze_mask"] = _snapshot_delta(before_freeze, after_freeze)

    import torch

    def _capture_detached(system, batches):
        saved = [(parameter, parameter.requires_grad) for parameter in system.objective.parameters()]
        try:
            for parameter, _flag in saved:
                parameter.requires_grad_(False)
            with torch.inference_mode():
                return system._capture_invariance_snapshot(batches)
        finally:
            for parameter, flag in saved:
                parameter.requires_grad_(flag)

    detached_system = _build_system(config)
    detached_train, detached_validation = _batches(config)
    detached_system.train_optimizer_step((detached_train,))
    before_detached = _capture_detached(detached_system, detached_validation)
    for name, parameter in detached_system.objective.named_parameters():
        parameter.requires_grad_(name in active)
    after_detached = _capture_detached(detached_system, detached_validation)
    probes["freeze_mask_inference_mode"] = _snapshot_delta(before_detached, after_detached)

    transition_system = _build_system(config)
    transition_train, transition_validation = _batches(config)
    transition_system.train_optimizer_step((transition_train,))
    try:
        transition_system.transition_to_stage(
            1,
            invariance_batches=transition_validation,
            entry_gate_report_sha256="0" * 64,
            entry_gate_passed=True,
            entry_gate_step=config.stages[0].end_optimizer_step,
        )
        probes["transition"] = {
            "matched": True,
            "mismatched": [],
            "error": None,
        }
    except RuntimeError as error:
        probes["transition"] = {
            "matched": False,
            "mismatched": [str(error)],
            "error": str(error),
        }

    report = {
        "diagnosis_only": True,
        "device": args.device,
        "mode": args.mode,
        "allow_tf32": bool(args.allow_tf32),
        "config": str(args.config),
        "probes": probes,
    }
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
