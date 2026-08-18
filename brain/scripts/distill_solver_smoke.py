"""Bounded CPU smoke distillation: thought-field on solver demonstrations.

This is an exploratory, CPU-only, single-threaded *verification probe* —
not a training run and not a qualified result. It distills the smoke-scale
thought-field model on solver-teacher demonstrations for one of the four
solver worlds (keys_doors, junction, occlusion, pursuit) for 256
optimizer steps and reports four pinned measurements:

- train loss, first vs last 32 steps;
- validation teacher-agreement (action loss and movement exact-match on
  held-out validation sequences) before and after;
- closed-loop play on the world's canonical slot (seeds 5/9, 240 ticks)
  before and after.

Usage (from the repository root, play-safe Python):

    python brain/scripts/distill_solver_smoke.py [world]

``world`` defaults to ``occlusion``, the ladder's memory-skill world,
where the teacher's advantage over reactive policies comes entirely from
persistent state — the skill the thought-state thesis predicts the model
architecture should be able to imitate.

The 2026-08-18 occlusion reference outcome (bit-identical across two
runs): train loss 0.7623 -> 0.4306, validation action loss
1.0133 -> 0.4316, movement exact-match 0.0000 -> 0.3065, closed-loop
reward/collisions 0/0 -> -17/17. The imitation gradient is real and far
stronger than on maze_chase (0.31 vs 0.008 exact-match), but the
untrained model's zero-collision stillness was worth more than
half-learned movement: play-level transfer — above all hazard
*avoidance*, the memory-dependent part — needs real scale, the DGX
distillation campaign, not more CPU steps.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

BRAIN_ROOT = Path(__file__).resolve().parents[1]
SRC = BRAIN_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SEED = 20260818
OPTIMIZER_STEPS = 256
TRAIN_SEQUENCES = 16
VALIDATION_SEQUENCES = 4
SEQUENCE_LENGTH = 32
BURN_IN_STEPS = 1
PLAY_SEEDS = (5, 9)
PLAY_TICKS = 240


def main() -> None:
    import torch

    torch.set_num_threads(1)

    from irene_brain.data import (
        SOLVER_WORLD_NAMES,
        solver_environment_factory,
    )
    from irene_brain.evaluation.closed_loop_play import (
        ClosedLoopPlayConfig,
        evaluate_closed_loop_play,
    )
    from irene_brain.model.spec import ThoughtFieldConfig
    from irene_brain.model.torch_model import IreneBrainModel
    from irene_brain.training.batches import (
        SolverBatchConfig,
        SolverBatchSource,
    )
    from irene_brain.training.config import (
        DatasetConfig,
        DeterminismConfig,
        LoggingConfig,
        OptimizationConfig,
        PrecisionConfig,
        ResourceConfig,
        RunConfig,
        TrainingConfig,
    )
    from irene_brain.training.objective import ThoughtFieldObjective
    from irene_brain.training.torch_system import TorchTrainingSystem

    world = sys.argv[1] if len(sys.argv) > 1 else "occlusion"
    if world not in SOLVER_WORLD_NAMES:
        raise SystemExit(
            f"world must be one of {sorted(SOLVER_WORLD_NAMES)}, got {world!r}"
        )
    print(f"world: {world}")

    config = TrainingConfig(
        schema_version=1,
        run=RunConfig(
            name=f"solver-{world}-distill-probe",
            seed=SEED,
            model_factory="irene_brain.model.torch_model:IreneBrainModel",
            max_optimizer_steps=OPTIMIZER_STEPS,
        ),
        dataset=DatasetConfig(
            kind="moving_shapes",
            train_sequences=4,
            validation_sequences=2,
            test_sequences=2,
            sequence_length=2,
            burn_in_steps=1,
            seed_offset=0,
            hazard_count=1,
            tick_period_ns=33_333_333,
            discount=0.99,
        ),
        optimization=OptimizationConfig(
            batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_gradient_norm=1.0,
            warmup_steps=0,
        ),
        precision=PrecisionConfig(device="cpu", mode="float32", allow_tf32=False),
        determinism=DeterminismConfig(
            enabled=True, num_workers=0, compile_model=False
        ),
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

    torch.manual_seed(SEED)
    model_config = replace(
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
    model = IreneBrainModel(model_config)
    system = TorchTrainingSystem(ThoughtFieldObjective(model), config)
    source = SolverBatchSource(
        SolverBatchConfig(
            train_sequences=TRAIN_SEQUENCES,
            validation_sequences=VALIDATION_SEQUENCES,
            test_sequences=2,
            sequence_length=SEQUENCE_LENGTH,
            burn_in_steps=BURN_IN_STEPS,
            world=world,
        )
    )
    play = ClosedLoopPlayConfig(episode_seeds=PLAY_SEEDS, max_ticks=PLAY_TICKS)
    canonical_factory = solver_environment_factory(world)

    def env_factory() -> object:
        return canonical_factory(PLAY_TICKS, 16_666_667)

    def play_row() -> tuple[float, int, int]:
        report = evaluate_closed_loop_play(
            model,
            config=play,
            model_description="solver distillation probe",
            environment_factory=env_factory,
        )
        totals = report.to_dict()["totals"]
        return (
            totals["reward_sum"],
            totals["collisions"],
            totals["decisions_rejected"],
        )

    def validation_agreement() -> tuple[float, float]:
        rows = []
        for batch in source.iter_batches(
            split="validation",
            epoch=0,
            start_batch=0,
            batch_size=1,
            max_batches=VALIDATION_SEQUENCES,
        ):
            result = system.evaluate_batch(batch)
            rows.append(
                (
                    result.metrics["action_loss"],
                    result.metrics["movement_exact_match"],
                )
            )
        return (
            sum(row[0] for row in rows) / len(rows),
            sum(row[1] for row in rows) / len(rows),
        )

    print("play before:", play_row())
    print("validation before: action_loss=%.4f movement_exact_match=%.4f" % validation_agreement())
    losses: list[float] = []
    for step in range(OPTIMIZER_STEPS):
        epoch, index = divmod(step, TRAIN_SEQUENCES)
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
    first = sum(losses[:32]) / 32
    last = sum(losses[-32:]) / 32
    print(f"train loss: first32={first:.4f} last32={last:.4f}")
    print("validation after:  action_loss=%.4f movement_exact_match=%.4f" % validation_agreement())
    print("play after:", play_row())


if __name__ == "__main__":
    main()
