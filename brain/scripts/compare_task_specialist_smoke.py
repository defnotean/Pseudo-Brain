"""Bounded CPU smoke for B3: generalist versus per-world specialists.

Exploratory, CPU-only, single-threaded *verification probe* — not a
training run and not a qualified result. Implements the preregistered B3
control (PLAN §28 item 14 / §29 unchanged-generalist vs game-specific
fine-tune) on the existing lazy moving_shapes and maze_chase datasets:

1. Train the smoke-scale reference thought field on the mixed-world
   generalist source.
2. Freeze that checkpoint as the unchanged generalist.
3. Fine-tune two copies — one on moving_shapes, one on maze_chase.
4. Score all three on both worlds' held-out validation sequences.

Campaign-scale materialization stays DGX-window work. ``DatasetConfig.kind``
is unchanged. Schema 2 / constant_after_warmup so every numbered step is a
real optimizer update (see the 2026-08-18 smoke-probe schedule erratum).

Usage (from the repository root, play-safe Python):

    python brain/scripts/compare_task_specialist_smoke.py
    python brain/scripts/compare_task_specialist_smoke.py --generalist-steps 16 --specialist-steps 8
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

BRAIN_ROOT = Path(__file__).resolve().parents[1]
SRC = BRAIN_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SEED = 20260818
TRAIN_SEQUENCES = 4
VALIDATION_SEQUENCES = 2
SEQUENCE_LENGTH = 8
BURN_IN_STEPS = 2
DEFAULT_GENERALIST_STEPS = 16
DEFAULT_SPECIALIST_STEPS = 8
DEFAULT_OUTPUT_DIR = (
    BRAIN_ROOT / "docs" / "runs" / "artifacts" / "task-specialist-smoke"
)


def _moving_config() -> object:
    from irene_brain.training.config import DatasetConfig

    return DatasetConfig(
        kind="moving_shapes",
        train_sequences=TRAIN_SEQUENCES,
        validation_sequences=VALIDATION_SEQUENCES,
        test_sequences=2,
        sequence_length=SEQUENCE_LENGTH,
        burn_in_steps=BURN_IN_STEPS,
        seed_offset=0,
        hazard_count=3,
        tick_period_ns=16_666_667,
        discount=0.99,
    )


def _maze_config() -> object:
    from irene_brain.training.batches import MazeChaseBatchConfig

    return MazeChaseBatchConfig(
        train_sequences=TRAIN_SEQUENCES,
        validation_sequences=VALIDATION_SEQUENCES,
        test_sequences=2,
        sequence_length=SEQUENCE_LENGTH,
        burn_in_steps=BURN_IN_STEPS,
    )


def _training_config():
    from irene_brain.training.config import (
        DeterminismConfig,
        LoggingConfig,
        OptimizationConfig,
        PrecisionConfig,
        ResourceConfig,
        RunConfig,
        TrainingConfig,
    )

    return TrainingConfig(
        schema_version=2,
        run=RunConfig(
            name="task-specialist-smoke",
            seed=SEED,
            model_factory="irene_brain.model.torch_model:IreneBrainModel",
            max_optimizer_steps=2,
        ),
        dataset=_moving_config(),
        optimization=OptimizationConfig(
            batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_gradient_norm=1.0,
            warmup_steps=0,
            scheduler_kind="constant_after_warmup",
        ),
        precision=PrecisionConfig(device="cpu", mode="float32", allow_tf32=False),
        determinism=DeterminismConfig(enabled=True, num_workers=0, compile_model=False),
        logging=LoggingConfig(
            log_every_steps=1,
            evaluate_every_steps=1,
            validation_batches=1,
            checkpoint_every_steps=1,
            keep_last_checkpoints=1,
        ),
        resources=ResourceConfig(
            allow_gpu=False,
            allow_capture=False,
            allow_hid_output=False,
            allow_background_threads=False,
            allow_network=False,
            allow_subprocess=False,
            write_artifacts=True,
            cpu_threads=1,
        ),
    )


def _model():
    from irene_brain.model.spec import ThoughtFieldConfig
    from irene_brain.model.torch_model import IreneBrainModel

    config = replace(
        ThoughtFieldConfig.smoke(),
        core_width=16,
        sensor_tokens=4,
        belief_tokens=2,
        working_memory_tokens=1,
        thoughtlets=4,
        registers_per_thoughtlet=3,
        goal_context_tokens=1,
        cognitive_cycles=2,
        brain_cell_blocks=1,
        attention_heads=2,
        routed_neighbors=1,
        episodic_memory_entries=8,
        retrieved_entries_per_thoughtlet=1,
    )
    return IreneBrainModel(config, input_resolution=(8, 8), plan_steps=2)


def _train(system, source, *, steps: int, sequences: int) -> list[float]:
    losses: list[float] = []
    for step in range(steps):
        epoch, index = divmod(step, sequences)
        batch = next(
            source.iter_batches(
                split="train",
                epoch=epoch,
                start_batch=index,
                batch_size=1,
                max_batches=1,
            )
        )
        losses.append(system.train_optimizer_step((batch,)).loss)
    return losses


def _evaluate(system, source, *, batches: int) -> dict[str, float]:
    rows = []
    for index in range(batches):
        batch = next(
            source.iter_batches(
                split="validation",
                epoch=0,
                start_batch=index,
                batch_size=1,
                max_batches=1,
            )
        )
        rows.append(system.evaluate_batch(batch).metrics)
    keys = rows[0].keys()
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def _snapshot(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generalist-steps", type=int, default=DEFAULT_GENERALIST_STEPS)
    parser.add_argument("--specialist-steps", type=int, default=DEFAULT_SPECIALIST_STEPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    if arguments.generalist_steps < 1 or arguments.specialist_steps < 1:
        raise SystemExit("step counts must be positive")

    import torch

    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    from irene_brain.training.batches import (
        MazeChaseBatchSource,
        MixedWorldBatchConfig,
        MixedWorldBatchSource,
        MovingShapesBatchSource,
    )
    from irene_brain.training.objective import ThoughtFieldObjective
    from irene_brain.training.torch_system import TorchTrainingSystem

    moving_config = _moving_config()
    maze_config = _maze_config()
    mixed = MixedWorldBatchSource(
        MixedWorldBatchConfig(moving_shapes=moving_config, maze_chase=maze_config)
    )
    shapes = MovingShapesBatchSource(moving_config)
    maze = MazeChaseBatchSource(maze_config)
    mixed_train_sequences = TRAIN_SEQUENCES * 2

    generalist_model = _model()
    generalist = TorchTrainingSystem(
        ThoughtFieldObjective(generalist_model), _training_config()
    )
    generalist_losses = _train(
        generalist,
        mixed,
        steps=arguments.generalist_steps,
        sequences=mixed_train_sequences,
    )
    snapshot = _snapshot(generalist_model)

    def _specialist(source, *, steps: int):
        model = _model()
        model.load_state_dict(snapshot, strict=True)
        system = TorchTrainingSystem(ThoughtFieldObjective(model), _training_config())
        losses = _train(system, source, steps=steps, sequences=TRAIN_SEQUENCES)
        return system, losses

    shapes_system, shapes_losses = _specialist(
        shapes, steps=arguments.specialist_steps
    )
    maze_system, maze_losses = _specialist(maze, steps=arguments.specialist_steps)

    agents = {
        "generalist": generalist,
        "moving_shapes_specialist": shapes_system,
        "maze_chase_specialist": maze_system,
    }
    worlds = {"moving_shapes": shapes, "maze_chase": maze}
    table = {}
    for agent_name, system in agents.items():
        table[agent_name] = {
            world_name: {
                key: value
                for key, value in _evaluate(
                    system, source, batches=VALIDATION_SEQUENCES
                ).items()
                if key in {"action_loss", "movement_exact_match", "world_loss"}
            }
            for world_name, source in worlds.items()
        }

    payload = {
        "seed": SEED,
        "schema_version": 2,
        "scheduler_kind": "constant_after_warmup",
        "generalist_steps": arguments.generalist_steps,
        "specialist_steps": arguments.specialist_steps,
        "sequence_length": SEQUENCE_LENGTH,
        "burn_in_steps": BURN_IN_STEPS,
        "mixed_manifest_sha256": mixed.manifest_sha256,
        "train_loss": {
            "generalist_first": generalist_losses[0],
            "generalist_last": generalist_losses[-1],
            "moving_shapes_specialist_last": shapes_losses[-1],
            "maze_chase_specialist_last": maze_losses[-1],
        },
        "validation": table,
    }
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = arguments.output_dir / "irene.task_specialist.v1.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
