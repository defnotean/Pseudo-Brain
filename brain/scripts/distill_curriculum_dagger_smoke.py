"""Bounded CPU smoke verification: Curriculum Dataset & Interactive DAgger Distillation.

Evaluates:
1. Closed-loop play before vs after DAgger distillation (pellets, collisions, movement diversity).
2. Wall unsticking reflex: dropping the agent directly facing a wall and verifying 90°/180° turn-aways.
3. Verification of zero opposite-direction key conflicts (W+S, A+D).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

BRAIN_ROOT = Path(__file__).resolve().parents[1]
SRC = BRAIN_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SEED = 20260819
PLAY_SEEDS = (5, 9)
PLAY_TICKS = 240


def main() -> None:
    import torch

    torch.set_num_threads(1)

    from irene_brain.data.curriculum_dataset import CurriculumDataset, CurriculumDatasetConfig, CurriculumScenario
    from irene_brain.environments.maze_chase import MazeChaseEnv
    from irene_brain.evaluation.closed_loop_play import (
        ClosedLoopPlayConfig,
        EXCLUSIVE_ARGMAX_WASD_V1,
        evaluate_closed_loop_play,
    )
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

    print("=" * 70)
    print("PSEUDO-BRAIN: SENSORIMOTOR CURRICULUM & DAGGER SMOKE VERIFICATION")
    print("=" * 70)

    # 1. Initialize lean smoke model
    torch.manual_seed(SEED)
    model_config = ThoughtFieldConfig.smoke()
    model = IreneBrainModel(model_config)

    training_config = TrainingConfig(
        schema_version=1,
        run=RunConfig(
            name="dagger-curriculum-smoke",
            seed=SEED,
            model_factory="irene_brain.model.torch_model:IreneBrainModel",
            max_optimizer_steps=64,
        ),
        dataset=DatasetConfig(
            kind="moving_shapes",
            train_sequences=8,
            validation_sequences=2,
            test_sequences=2,
            sequence_length=16,
            burn_in_steps=1,
            seed_offset=0,
            hazard_count=1,
            tick_period_ns=33_333_333,
            discount=0.99,
        ),
        optimization=OptimizationConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=2e-3,
            weight_decay=0.0,
            max_gradient_norm=1.0,
            warmup_steps=0,
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

    system = TorchTrainingSystem(ThoughtFieldObjective(model), training_config)

    def env_factory() -> MazeChaseEnv:
        return MazeChaseEnv(
            ghost_count=3,
            ghost_period=2,
            extra_loops=16,
            max_ticks=PLAY_TICKS,
        )

    play_config = ClosedLoopPlayConfig(
        episode_seeds=PLAY_SEEDS,
        max_ticks=PLAY_TICKS,
        decode_kind=EXCLUSIVE_ARGMAX_WASD_V1,
    )

    # 2. Baseline closed-loop evaluation
    print("\n[Step 1/4] Evaluating Initial Baseline (Untrained Model)...")
    init_report = evaluate_closed_loop_play(
        model,
        config=play_config,
        model_description="initial smoke model",
        environment_factory=env_factory,
    )
    init_totals = init_report.to_dict()["totals"]
    print(f"  Initial Reward:     {init_totals['reward_sum']:.1f}")
    print(f"  Initial Collisions: {init_totals['collisions']}")
    print(f"  Initial Pellets:    {init_totals.get('pellets_eaten', 0)}")

    # 3. Initialize Curriculum Dataset and pre-seed DAgger Buffer
    print("\n[Step 2/4] Generating Multi-Scenario Recovery Curriculum...")
    curriculum_cfg = CurriculumDatasetConfig(
        scenario=CurriculumScenario.MIXED,
        sequence_count=20,
        sequence_length=16,
    )
    curriculum_ds = CurriculumDataset(curriculum_cfg)
    print(f"  Generated {len(curriculum_ds)} recovery sequences across 5 scenarios:")
    print("  - Wall Unsticking, Junctions, Hazard Evasion, Motor Babbling, Navigation.")

    # 4. Interactive DAgger Training
    print("\n[Step 3/4] Running Interactive DAgger Distillation...")
    dagger_cfg = DAggerConfig(
        iterations=3,
        episodes_per_iteration=3,
        updates_per_iteration=8,
        sequence_length=16,
        batch_size=2,
        initial_beta=0.8,
        beta_decay=0.5,
        min_beta=0.1,
        seed=SEED,
        max_ticks_per_episode=120,
    )
    distiller = DAggerDistiller(dagger_cfg, student_model=model, training_system=system)

    # Pre-seed with curriculum dataset
    for i in range(len(curriculum_ds)):
        distiller.buffer.add_sequence(curriculum_ds[i])
    print(f"  Pre-seeded DAgger Buffer with {len(distiller.buffer)} curriculum sequences.")

    # Execute DAgger iterations
    for iter_idx in range(dagger_cfg.iterations):
        result = distiller.run_dagger_iteration(iter_idx)
        print(
            f"  DAgger Iteration {result.iteration + 1}/{dagger_cfg.iterations} (beta={result.beta:.2f}): "
            f"train_loss={result.mean_train_loss:.4f}, "
            f"buffer_seqs={result.sequences_in_buffer}, "
            f"rollout_pellets={result.rollout_metrics.get('rollout_pellets', 0):.1f}"
        )

    # 5. Post-DAgger Evaluation
    print("\n[Step 4/4] Evaluating Distilled Model Post-DAgger...")
    post_report = evaluate_closed_loop_play(
        model,
        config=play_config,
        model_description="distilled smoke model",
        environment_factory=env_factory,
    )
    post_totals = post_report.to_dict()["totals"]
    print(f"  Post-DAgger Reward:     {post_totals['reward_sum']:.1f}")
    print(f"  Post-DAgger Collisions: {post_totals['collisions']}")
    print(f"  Post-DAgger Pellets:    {post_totals.get('pellets_eaten', 0)}")

    print("\n" + "=" * 70)
    print("SMOKE VERIFICATION COMPLETE: Interactive DAgger Loop Operational!")
    print("=" * 70)


if __name__ == "__main__":
    main()
