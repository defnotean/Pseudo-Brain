"""Interactive DAgger & Multi-Scenario Curriculum Training Runner for Pseudo-Brain.

Executes closed-loop DAgger distillation combining:
1. Multi-scenario recovery data (wall unsticking, junctions, evasion, babbling, standard navigation).
2. On-policy student rollouts with interactive expert lookahead planner queries.
3. Multi-thought BrainCell backpropagation through time.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys
import time

try:
    import torch
except ModuleNotFoundError:
    torch = None


def main() -> int:
    if torch is None:
        print("Error: PyTorch is required to run curriculum DAgger training.", file=sys.stderr)
        return 1

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    from irene_brain.data.curriculum_dataset import CurriculumDataset, CurriculumDatasetConfig
    from irene_brain.model.spec import ThoughtFieldConfig
    from irene_brain.model.torch_model import IreneBrainModel
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
    from irene_brain.training.dagger_distill import DAggerConfig, DAggerDistiller
    from irene_brain.training.objective import ThoughtFieldObjective
    from irene_brain.training.torch_system import TorchTrainingSystem

    parser = argparse.ArgumentParser(description="Pseudo-Brain Curriculum DAgger Distillation Runner")
    parser.add_argument("--iterations", type=int, default=4, help="Number of DAgger iterations")
    parser.add_argument("--episodes-per-iter", type=int, default=2, help="Episodes to collect per iteration")
    parser.add_argument("--updates-per-iter", type=int, default=8, help="Optimizer updates per iteration")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--initial-beta", type=float, default=0.8, help="Initial expert probability beta")
    parser.add_argument("--beta-decay", type=float, default=0.6, help="Beta decay factor per iteration")
    parser.add_argument("--seed-curriculum", action="store_true", default=True, help="Seed buffer with curriculum recovery data")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write metrics JSON")
    args = parser.parse_args()

    print("=" * 75)
    print("   PSEUDO-BRAIN MULTI-SCENARIO CURRICULUM & DAGGER DISTILLATION")
    print("=" * 75)
    print(f"Iterations: {args.iterations} | Episodes/Iter: {args.episodes_per_iter} | Updates/Iter: {args.updates_per_iter}")
    print(f"Batch Size: {args.batch_size} | Learning Rate: {args.learning_rate} | Beta: {args.initial_beta} -> decay {args.beta_decay}")
    print("-" * 75)

    # 1. Initialize Model
    base_config = ThoughtFieldConfig.smoke()
    model_config = replace(
        base_config,
        core_width=32,
        thoughtlets=4,
        cognitive_cycles=2,
        actuator=replace(base_config.actuator, continuous_squash="deadzone_tanh"),
    )
    model = IreneBrainModel(model_config)

    # 2. Setup Training System
    training_config = TrainingConfig(
        schema_version=1,
        run=RunConfig(
            name="dagger-curriculum-runner",
            seed=20260818,
            model_factory="irene_brain.model.torch_model:IreneBrainModel",
            max_optimizer_steps=args.iterations * args.updates_per_iter + 100,
        ),
        dataset=DatasetConfig(
            kind="curriculum",
            curriculum_scenario="mixed",
            train_sequences=64,
            validation_sequences=8,
            test_sequences=8,
            sequence_length=8,
            burn_in_steps=2,
            seed_offset=0,
            hazard_count=2,
            tick_period_ns=16_666_667,
            discount=0.99,
        ),
        optimization=OptimizationConfig(
            batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=args.learning_rate,
            weight_decay=1e-4,
            max_gradient_norm=1.0,
            warmup_steps=0,
        ),
        precision=PrecisionConfig(
            device="cpu",
            mode="float32",
            allow_tf32=False,
        ),
        determinism=DeterminismConfig(
            enabled=True,
            num_workers=0,
            compile_model=False,
        ),
        logging=LoggingConfig(
            log_every_steps=1,
            evaluate_every_steps=1,
            validation_batches=1,
            checkpoint_every_steps=10,
            keep_last_checkpoints=2,
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

    objective = ThoughtFieldObjective(model)
    system = TorchTrainingSystem(objective, training_config)

    # 3. Setup DAgger Config & Distiller
    dagger_config = DAggerConfig(
        iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iter,
        max_ticks_per_episode=60,
        initial_beta=args.initial_beta,
        beta_decay=args.beta_decay,
        sequence_length=8,
        burn_in_steps=2,
        batch_size=args.batch_size,
        updates_per_iteration=args.updates_per_iter,
        ghost_count=2,
        ghost_period=2,
        extra_loops=8,
    )

    distiller = DAggerDistiller(
        config=dagger_config,
        student_model=model,
        training_system=system,
    )

    # 4. Seed with initial multi-scenario curriculum transitions if requested
    if args.seed_curriculum:
        curriculum_cfg = CurriculumDatasetConfig(
            sequence_length=8,
            sequence_count=32,
            scenario="mixed",
        )
        curriculum_dataset = CurriculumDataset(curriculum_cfg)
        for i in range(min(16, len(curriculum_dataset))):
            distiller.buffer.add_sequence(curriculum_dataset[i])
        print(f"Pre-seeded aggregation buffer with {len(distiller.buffer)} curriculum recovery sequences.")

    # 5. Execute DAgger iterations
    history = []
    start_time = time.perf_counter()
    for it in range(args.iterations):
        it_start = time.perf_counter()
        result = distiller.run_dagger_iteration(iteration=it)
        it_elapsed = time.perf_counter() - it_start

        record = {
            "iteration": result.iteration + 1,
            "beta": result.beta,
            "episodes_collected": result.episodes_collected,
            "transitions_collected": result.transitions_collected,
            "sequences_in_buffer": result.sequences_in_buffer,
            "mean_train_loss": result.mean_train_loss,
            "training_metrics": dict(result.training_metrics),
            "rollout_metrics": dict(result.rollout_metrics),
            "elapsed_seconds": it_elapsed,
        }
        history.append(record)

        print(
            f"Iter [{result.iteration + 1}/{args.iterations}] | "
            f"Buffer: {result.sequences_in_buffer:3d} seqs (+{result.transitions_collected} tr) | "
            f"Beta: {result.beta:.3f} | "
            f"Train Loss: {result.mean_train_loss:.4f} | "
            f"Time: {it_elapsed:.2f}s"
        )

    total_time = time.perf_counter() - start_time
    print("-" * 75)
    print(f"Curriculum DAgger Complete in {total_time:.2f}s | Final Buffer: {len(distiller.buffer)} sequences.")
    print("=" * 75)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"history": history, "total_seconds": total_time}, f, indent=2)
        print(f"Metrics saved to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
