"""Zero-shot transfer battery: a moving_shapes-trained smoke model on the ladder.

Exploratory, CPU-only, single-threaded *verification probe* — not a
training run and not a qualified result. A smoke-scale reference
thought-field model is trained on moving_shapes for 128 optimizer steps
(the campaign window shape: sequence_length 8, burn-in 2), then played
zero-shot — no fine-tuning — on all six canonical ladder worlds. Three
non-privileged diagnostic baselines (no-op, random, scripted reactive
chaser) are evaluated under the identical play configuration, so every
model row is interpretable against the same zero-points.

``--variant world_model_actor`` swaps the trained agent for the B2
latent world-model actor (monolithic trunk at the parameter-matched smoke
width 18) trained through its declared ``LatentRolloutObjective`` hook —
the same resolution discipline as train.py — and writes its rows to a
per-variant subdirectory so the pinned reference rows stay untouched.

Per-world JSON rows are written incrementally, so a partial run can be
resumed. This quantifies the transfer gap named in the 2026-08-18
strategic review; the qualified transfer study remains DGX-scale.

Usage (from the repository root, play-safe Python):

    python brain/scripts/transfer_gap_smoke.py [--steps 128] [--worlds pursuit,occlusion]
    python brain/scripts/transfer_gap_smoke.py --variant world_model_actor
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
TRAIN_SEQUENCES = 16
SEQUENCE_LENGTH = 8
BURN_IN_STEPS = 2
PLAY_SEEDS = (5, 9)
PLAY_TICKS = 240
DEFAULT_OUTPUT_DIR = BRAIN_ROOT / "docs" / "runs" / "artifacts" / "transfer-gap-smoke"


def _training_config():
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

    # 2026-08-18 erratum fix: schema_version 1 forces the legacy cosine
    # schedule, which decays the LR multiplier to exactly 0.0 at
    # max_optimizer_steps (2) — every numbered step past step 1 applied no
    # update. Schema 2 with constant_after_warmup (the campaign baseline
    # regime) makes every numbered step a real optimizer update.
    return TrainingConfig(
        schema_version=2,
        run=RunConfig(
            name="transfer-gap-smoke",
            seed=SEED,
            model_factory="irene_brain.model.torch_model:IreneBrainModel",
            max_optimizer_steps=2,
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


def _slot_smoke_config():
    from irene_brain.model.spec import ThoughtFieldConfig

    return replace(
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


def _build_trained_model(steps: int, *, variant: str = "routed"):
    import importlib

    import torch

    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    from irene_brain.model.torch_model import IreneBrainModel
    from irene_brain.model.world_model_actor import LatentWorldModelActor
    from irene_brain.training.batches import MovingShapesBatchSource
    from irene_brain.training.config import DatasetConfig
    from irene_brain.training.objective import ThoughtFieldObjective
    from irene_brain.training.torch_system import TorchTrainingSystem

    if variant == "routed":
        model = IreneBrainModel(
            _slot_smoke_config(), input_resolution=(8, 8), plan_steps=2
        )
    elif variant == "world_model_actor":
        model = LatentWorldModelActor(
            replace(
                _slot_smoke_config(),
                core_width=18,
                thoughtlets=1,
                registers_per_thoughtlet=1,
                routed_neighbors=0,
            ),
            input_resolution=(8, 8),
            plan_steps=2,
        )
    else:
        raise ValueError(f"unknown transfer-battery variant: {variant}")
    # Own-recipe-family variants declare their objective fail-closed on the
    # model; resolve it with the same importlib discipline as train.py.
    objective_class_path = getattr(model, "training_objective_class_path", None)
    if objective_class_path is None:
        objective_class = ThoughtFieldObjective
    else:
        module_name, attribute = objective_class_path.split(":", 1)
        objective_class = getattr(importlib.import_module(module_name), attribute)
    system = TorchTrainingSystem(objective_class(model), _training_config())
    source = MovingShapesBatchSource(
        DatasetConfig(
            kind="moving_shapes",
            train_sequences=TRAIN_SEQUENCES,
            validation_sequences=2,
            test_sequences=2,
            sequence_length=SEQUENCE_LENGTH,
            burn_in_steps=BURN_IN_STEPS,
            seed_offset=0,
            hazard_count=3,
            tick_period_ns=16_666_667,
            discount=0.99,
        )
    )
    losses: list[float] = []
    for step in range(steps):
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
    window = min(16, steps)
    train_loss_last = sum(losses[-window:]) / window
    return system.objective.model, train_loss_last


def _play_rows(agent, world_slots, config, description: str) -> list[dict[str, object]]:
    from irene_brain.evaluation.closed_loop_play import (
        ClosedLoopPlayReport,
        evaluate_closed_loop_play,
        run_policy_closed_loop_episode,
    )

    rows = []
    for slot in world_slots:
        if hasattr(agent, "training"):
            report = evaluate_closed_loop_play(
                agent,
                config=config,
                model_description=description,
                environment_factory=lambda slot=slot: slot.factory(config),
            )
        else:
            episodes = tuple(
                run_policy_closed_loop_episode(
                    agent,
                    seed=seed,
                    config=config,
                    environment_factory=lambda slot=slot: slot.factory(config),
                )
                for seed in config.episode_seeds
            )
            report = ClosedLoopPlayReport(
                schema_version=1,
                config=config,
                model_description=f"diagnostic policy {description} on {slot.identity}",
                episodes=episodes,
            )
        totals = report.to_dict()["totals"]
        rows.append(
            {
                "world": slot.identity,
                "reward_sum": totals["reward_sum"],
                "targets_collected": totals["targets_collected"],
                "collisions": totals["collisions"],
                "decisions_rejected": totals["decisions_rejected"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--worlds", type=str, default=None)
    parser.add_argument(
        "--variant",
        type=str,
        default="routed",
        choices=("routed", "world_model_actor"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()
    if arguments.steps < 1:
        raise SystemExit("--steps must be positive")
    output_dir = arguments.output_dir
    if output_dir is None:
        output_dir = (
            DEFAULT_OUTPUT_DIR
            if arguments.variant == "routed"
            else DEFAULT_OUTPUT_DIR / arguments.variant
        )

    from irene_brain.evaluation.cross_world_matrix import default_world_slots
    from irene_brain.evaluation.closed_loop_play import ClosedLoopPlayConfig
    from irene_brain.evaluation.diagnostic_policies import (
        NoOpPolicy,
        RandomMovementPolicy,
        ScriptedTargetChasePolicy,
    )

    config = ClosedLoopPlayConfig(
        episode_seeds=PLAY_SEEDS,
        max_ticks=PLAY_TICKS,
        hazard_count=3,
    )
    slots = list(default_world_slots()[:6])
    if arguments.worlds is not None:
        wanted = {item.strip() for item in arguments.worlds.split(",") if item.strip()}
        slots = [
            slot
            for slot in slots
            if slot.identity in wanted
            or slot.identity.removeprefix("world.").removesuffix(".v1") in wanted
        ]
        if not slots:
            raise SystemExit(f"no worlds matched {sorted(wanted)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model, train_loss_last = _build_trained_model(arguments.steps, variant=arguments.variant)
    agents = (
        ("trained_smoke_model", model),
        ("diagnostic.noop.v1", NoOpPolicy()),
        ("diagnostic.random_movement.v1", RandomMovementPolicy()),
        ("diagnostic.scripted_chase.v1", ScriptedTargetChasePolicy()),
    )
    for slot in slots:
        world_rows = []
        for description, agent in agents:
            rows = _play_rows(agent, [slot], config, description)
            world_rows.append({"agent": description, **rows[0]})
            print(f"{slot.identity} {description}: {rows[0]}")
        payload = {
            "seed": SEED,
            "variant": arguments.variant,
            "optimizer_steps": arguments.steps,
            "train_loss_last16": train_loss_last,
            "play_config": {
                "episode_seeds": list(PLAY_SEEDS),
                "max_ticks": PLAY_TICKS,
                "hazard_count": 3,
            },
            "rows": world_rows,
        }
        (output_dir / f"{slot.identity}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
