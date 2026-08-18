"""Foreground command-line entry point for local or DGX Spark training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import random
from typing import Callable, Sequence

from ..project_paths import resolve_workspace_path
from .batches import dataset_batch_source
from .checkpoint import source_tree_sha256
from .config import TrainingConfig, load_training_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Pseudo-Brain thought-field model")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="canary-only foreground interruption at an optimizer boundary",
    )
    parser.add_argument(
        "--evaluate-only",
        choices=("validation", "test"),
        help="load --resume and evaluate without optimizer steps",
    )
    return parser


def _environment_value(canonical: str, legacy: str) -> str | None:
    canonical_present = canonical in os.environ
    legacy_present = legacy in os.environ
    canonical_value = os.environ.get(canonical)
    legacy_value = os.environ.get(legacy)
    if canonical_present and legacy_present and canonical_value != legacy_value:
        raise ValueError(f"conflicting {canonical} and legacy {legacy} values")
    return canonical_value if canonical_present else legacy_value


def _bounded_config(config: TrainingConfig) -> TrainingConfig:
    raw = _environment_value("PSEUDO_BRAIN_MAX_STEPS", "IRENE_BRAIN_MAX_STEPS")
    if raw is None:
        return config
    try:
        maximum = int(raw)
    except ValueError as error:
        raise ValueError("PSEUDO_BRAIN_MAX_STEPS must be a positive integer") from error
    if maximum < 1:
        raise ValueError("PSEUDO_BRAIN_MAX_STEPS must be a positive integer")
    if config.schema_version == 3 and maximum != config.run.max_optimizer_steps:
        raise ValueError(
            "PSEUDO_BRAIN_MAX_STEPS cannot rewrite a hashed schema-3 stage plan"
        )
    bounded = min(maximum, config.run.max_optimizer_steps)
    optimization = replace(
        config.optimization,
        warmup_steps=min(config.optimization.warmup_steps, bounded),
    )
    return replace(
        config,
        run=replace(config.run, max_optimizer_steps=bounded),
        optimization=optimization,
    )


def _run_directory(argument: Path | None) -> Path:
    if argument is not None:
        return argument
    configured = _environment_value("PSEUDO_BRAIN_RUN_DIR", "IRENE_BRAIN_RUN_DIR")
    if not configured:
        raise ValueError("set PSEUDO_BRAIN_RUN_DIR or pass --run-dir")
    return Path(configured)


def _resume_path(argument: Path | None) -> Path | None:
    if argument is not None:
        return argument
    configured = _environment_value("PSEUDO_BRAIN_RESUME", "IRENE_BRAIN_RESUME")
    return None if not configured else Path(configured)


def _model_factory(path: str) -> Callable[[TrainingConfig], object]:
    module_name, attribute = path.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError(f"configured model factory is not callable: {path}")
    return factory


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config_path = resolve_workspace_path(arguments.config)
    config = _bounded_config(load_training_config(config_path))
    run_dir = resolve_workspace_path(_run_directory(arguments.run_dir))
    resume = _resume_path(arguments.resume)
    if resume is not None:
        resume = resolve_workspace_path(resume)
    expected_digest = arguments.checkpoint_sha256 or _environment_value(
        "PSEUDO_BRAIN_CHECKPOINT_SHA256",
        "IRENE_BRAIN_CHECKPOINT_SHA256",
    )
    if arguments.stop_after_step is not None and config.run.name != (
        "dgx-rcq-v2-staging-canary"
    ):
        raise ValueError(
            "--stop-after-step is reserved for the registered staging canary"
        )
    if arguments.evaluate_only and arguments.stop_after_step is not None:
        raise ValueError("--evaluate-only cannot be combined with --stop-after-step")

    # Apply resource limits before the configured factory imports PyTorch.
    for name, value in config.resource_policy.process_environment().items():
        os.environ[name] = value
    if config.precision.device == "cuda" and config.determinism.enabled:
        cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cublas_workspace is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif cublas_workspace not in {":4096:8", ":16:8"}:
            raise ValueError(
                "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
            )
    print(
        json.dumps(
            {
                "config": str(config_path),
                "config_sha256": config.config_sha256,
                "device": config.precision.device,
                "precision": config.precision.mode,
                "run_dir": str(run_dir),
                "max_optimizer_steps": config.run.max_optimizer_steps,
                "training_stages": len(config.stages),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # Accelerator-dependent imports occur only after config validation and the
    # explicit foreground invocation of this module.
    from .objective import ThoughtFieldObjective
    from .torch_system import TorchTrainingSystem
    from .trainer import Trainer
    import torch

    # Model construction consumes the torch RNG, so seed before invoking the
    # configured factory. TorchTrainingSystem reseeds once more afterward to
    # make the training-time RNG stream independent of initialization draws.
    random.seed(config.run.seed)
    torch.manual_seed(config.run.seed)
    if config.precision.device == "cuda":
        torch.cuda.manual_seed_all(config.run.seed)
    model = _model_factory(config.run.model_factory)(config)
    # Variants in their own preregistered recipe family (the B2 world-model
    # actor) declare a fail-closed objective hook on the model; every other
    # variant keeps the shared slot-suite objective.
    objective_class_path = getattr(model, "training_objective_class_path", None)
    if objective_class_path is None:
        objective_class = ThoughtFieldObjective
    else:
        if not isinstance(objective_class_path, str):
            raise TypeError("training_objective_class_path must be a string")
        objective_class = _model_factory(objective_class_path)
    objective = objective_class(
        model,
        action_loss_kind=config.objective.action_loss_kind,
        button_support_control_indices=(
            config.objective.button_support_control_indices
        ),
        button_support_weight=config.objective.button_support_weight,
        button_background_weight=config.objective.button_background_weight,
        button_background_tail_mix=config.objective.button_background_tail_mix,
        button_background_tail_temperature=(
            config.objective.button_background_tail_temperature
        ),
        continuous_action_weight=config.objective.continuous_action_weight,
        action_weight=config.objective.action_weight,
        value_weight=config.objective.value_weight,
        world_weight=config.objective.world_weight,
        diversity_weight=config.objective.diversity_weight,
        deadzone_hinge_weight=config.objective.continuous_deadzone_hinge_weight,
        deadzone_hinge_margin=config.objective.continuous_deadzone_hinge_margin,
        opposite_pair_weight=config.objective.opposite_key_pair_weight,
    )
    system = TorchTrainingSystem(objective, config)
    source = dataset_batch_source(config.dataset)
    package_root = Path(__file__).resolve().parents[1]
    trainer = Trainer(
        config,
        system,
        source,
        run_dir,
        code_sha256=source_tree_sha256(package_root),
    )
    if arguments.evaluate_only:
        if resume is None:
            raise ValueError("--evaluate-only requires --resume or PSEUDO_BRAIN_RESUME")
        if config.schema_version == 3 and arguments.evaluate_only == "test":
            raise ValueError(
                "schema-3 RCQ test data may be opened only by the once-only final evaluator"
            )
        result = trainer.evaluate_checkpoint(
            resume,
            split=arguments.evaluate_only,
            expected_checkpoint_sha256=expected_digest,
        )
        print(
            json.dumps(
                {
                    "status": "evaluated",
                    "split": arguments.evaluate_only,
                    "loss": result.loss,
                    "metrics": dict(result.metrics),
                    "samples": result.samples,
                },
                allow_nan=False,
                sort_keys=True,
            ),
            flush=True,
        )
        if config.dataset.kind == "maze_chase":
            from .play_gate import write_maze_chase_play_gate

            write_maze_chase_play_gate(
                system.objective.model,
                run_dir,
                decode_kind=config.objective.play_decode_kind,
            )
        return 0

    summary = trainer.run(
        resume_from=resume,
        expected_checkpoint_sha256=expected_digest,
        stop_after_step=arguments.stop_after_step,
    )
    print(
        json.dumps(
            {
                "status": "completed" if summary.completed else "stopped",
                "optimizer_step": summary.cursor.optimizer_step,
                "epoch": summary.cursor.epoch,
                "checkpoint": str(summary.last_checkpoint),
                "checkpoint_sha256": summary.checkpoint_sha256,
                "metrics": str(summary.metrics_path),
                "stop_reason": summary.stop_reason,
                "development_gate": (
                    None
                    if summary.development_gate_path is None
                    else str(summary.development_gate_path)
                ),
                "invariance_report": (
                    None
                    if summary.invariance_report_path is None
                    else str(summary.invariance_report_path)
                ),
                "final_development": (
                    None
                    if summary.final_development_path is None
                    else str(summary.final_development_path)
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if config.dataset.kind == "maze_chase":
        from .play_gate import write_maze_chase_play_gate

        write_maze_chase_play_gate(
            system.objective.model,
            run_dir,
            decode_kind=config.objective.play_decode_kind,
        )
    return 0 if summary.completed or summary.stop_reason == "operational_stop" else 1


if __name__ == "__main__":
    raise SystemExit(main())
