"""5-Way Architectural Component Ablation Battery for Pseudo-Brain.

Evaluates and compares:
  A: Baseline (Old Pseudo-Brain with fixed lifecycle and fixed C=3)
  B: + Adaptive Thought-Update Gate Only
  C: + Multi-Horizon Future Foresight Only
  D: + Adaptive Thinking Depth Controller Only
  E: Complete Integrated System (All Three)

All variants are trained under the exact same seed, DAgger iterations, and optimizer updates,
and evaluated across identical 20-seed validation batteries and diagnostic episodes.
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

from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.diagnostic_policies import ScriptedMazeChasePlannerPolicy
from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
from irene_brain.evaluation.spatial_cognitive_diagnostics import (
    compute_model_param_digest,
    run_instrumented_diagnostic_episode,
)
from irene_brain.model.lookahead_planner import LatentLookaheadPlanner
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.data.curriculum_dataset import CurriculumDataset, CurriculumDatasetConfig
from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.diagnostic_policies import ScriptedMazeChasePlannerPolicy
from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
from irene_brain.evaluation.spatial_cognitive_diagnostics import (
    compute_model_param_digest,
    run_instrumented_diagnostic_episode,
)
from irene_brain.model.lookahead_planner import LatentLookaheadPlanner
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


def build_ablation_model(variant: str, base_config: ThoughtFieldConfig) -> IreneBrainModel:
    """Instantiate model for a specific ablation variant."""
    if variant == "A_baseline":
        return IreneBrainModel(base_config, enable_adaptive_cognition=False)
    elif variant == "B_adaptive_gate":
        model = IreneBrainModel(base_config, enable_adaptive_cognition=True)
        model.future_trajectory_head = None
        model.halting_controller = None
        return model
    elif variant == "C_future_prediction":
        model = IreneBrainModel(base_config, enable_adaptive_cognition=True)
        model.adaptive_thought_gate = None
        model.halting_controller = None
        return model
    elif variant == "D_adaptive_depth":
        model = IreneBrainModel(base_config, enable_adaptive_cognition=True)
        model.adaptive_thought_gate = None
        model.future_trajectory_head = None
        return model
    elif variant == "E_all_three":
        return IreneBrainModel(base_config, enable_adaptive_cognition=True)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def train_ablation_variant(
    variant: str,
    *,
    dagger_iters: int = 2,
    updates_per_iter: int = 12,
    eval_ticks: int = 120,
) -> tuple[IreneBrainModel, dict[str, Any]]:
    """Train a specific ablation variant under strictly controlled conditions."""
    print(f"\n{'=' * 90}")
    print(f"  TRAINING ABLATION VARIANT: {variant}")
    print(f"{'=' * 90}")

    base_config = ThoughtFieldConfig.smoke()
    model_config = replace(
        base_config,
        core_width=32,
        thoughtlets=4,
        cognitive_cycles=3,
        actuator=replace(base_config.actuator, continuous_squash="deadzone_tanh"),
    )
    model = build_ablation_model(variant, model_config)

    training_config = TrainingConfig(
        schema_version=1,
        run=RunConfig(
            name=f"ablation-{variant}",
            seed=20260818,
            model_factory="irene_brain.model.torch_model:IreneBrainModel",
            max_optimizer_steps=10000,
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
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
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

    dagger_config = DAggerConfig(
        iterations=dagger_iters,
        episodes_per_iteration=2,
        max_ticks_per_episode=60,
        initial_beta=0.8,
        beta_decay=0.85,
        sequence_length=8,
        burn_in_steps=2,
        batch_size=2,
        updates_per_iteration=updates_per_iter,
        ghost_count=2,
        ghost_period=2,
        extra_loops=8,
    )

    distiller = DAggerDistiller(
        config=dagger_config,
        student_model=model,
        training_system=system,
    )

    start_time = time.perf_counter()
    curriculum_cfg = CurriculumDatasetConfig(
        sequence_length=8,
        sequence_count=40,
        scenario="mixed",
    )
    curriculum_dataset = CurriculumDataset(curriculum_cfg)
    for i in range(min(20, len(curriculum_dataset))):
        distiller.buffer.add_sequence(curriculum_dataset[i])

    for it in range(dagger_iters):
        res = distiller.run_dagger_iteration(iteration=it)
        print(f"  DAgger Step [{it+1}/{dagger_iters}] | Beta: {res.beta:.3f} | Loss: {res.mean_train_loss:.4f}")

    train_time = time.perf_counter() - start_time
    model.eval()

    # 1. Evaluate on 3-seed diagnostic suite
    env = MazeChaseEnv()
    expert = ScriptedMazeChasePlannerPolicy()
    diag_seeds = (1702, 1703, 1704)
    reports = []
    for s in diag_seeds:
        planner = LatentLookaheadPlanner(
            model=model,
            horizon=3,
            gamma=0.95,
            hazard_weight=5.0,
            cognitive_cycles_per_step=model.config.cognitive_cycles,
        )
        policy = LatentLookaheadPolicy(model=model, planner=planner)
        rep = run_instrumented_diagnostic_episode(
            policy=policy,
            env=env,
            seed=s,
            max_ticks=eval_ticks,
            expert_policy=expert,
        )
        reports.append(rep)

    mean_pellets_3 = sum(r.pellets_eaten for r in reports) / len(reports)
    mean_catches_3 = sum(r.ghost_collisions for r in reports) / len(reports)
    dead_end_catches_3 = sum(r.dead_end_catches for r in reports)
    wall_recovery_3 = sum(r.wall_dynamics.mean_wall_recovery_latency for r in reports) / len(reports)

    gate_normal = sum(r.gate_telemetry.alpha_normal for r in reports if r.gate_telemetry) / len(reports)
    gate_wall = sum(r.gate_telemetry.alpha_wall_collision for r in reports if r.gate_telemetry) / len(reports)
    gate_ghost = sum(r.gate_telemetry.alpha_ghost_danger for r in reports if r.gate_telemetry) / len(reports)

    depth_corridor = sum(r.depth_telemetry.cycles_open_corridor for r in reports if r.depth_telemetry) / len(reports)
    depth_ghost = sum(r.depth_telemetry.cycles_ghost_near for r in reports if r.depth_telemetry) / len(reports)
    depth_dead_end = sum(r.depth_telemetry.cycles_dead_end_ghost_near for r in reports if r.depth_telemetry) / len(reports)

    # 2. Evaluate on 20-Seed Extended Held-Out Battery
    held_out_seeds = range(2001, 2021)
    val_pellets = 0
    val_catches = 0
    for s in held_out_seeds:
        planner = LatentLookaheadPlanner(
            model=model,
            horizon=3,
            gamma=0.95,
            hazard_weight=5.0,
            cognitive_cycles_per_step=model.config.cognitive_cycles,
        )
        policy = LatentLookaheadPolicy(model=model, planner=planner)
        rep = run_instrumented_diagnostic_episode(
            policy=policy,
            env=env,
            seed=s,
            max_ticks=eval_ticks,
        )
        val_pellets += rep.pellets_eaten
        val_catches += rep.ghost_collisions

    mean_val_pellets = val_pellets / len(held_out_seeds)

    summary = {
        "variant": variant,
        "digest": compute_model_param_digest(model),
        "train_time_sec": round(train_time, 2),
        "diagnostic_3seed": {
            "mean_pellets": round(mean_pellets_3, 2),
            "mean_catches": round(mean_catches_3, 2),
            "dead_end_catches": dead_end_catches_3,
            "mean_wall_recovery": round(wall_recovery_3, 2),
            "gate_normal_alpha": round(gate_normal, 2),
            "gate_wall_alpha": round(gate_wall, 2),
            "gate_ghost_alpha": round(gate_ghost, 2),
            "depth_corridor_cycles": round(depth_corridor, 2),
            "depth_ghost_cycles": round(depth_ghost, 2),
            "depth_dead_end_cycles": round(depth_dead_end, 2),
        },
        "held_out_20seed": {
            "mean_pellets": round(mean_val_pellets, 2),
            "total_catches": val_catches,
        },
    }

    return model, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="5-Way Cognitive Component Ablation Runner")
    parser.add_argument("--dagger-iters", type=int, default=2)
    parser.add_argument("--updates-per-iter", type=int, default=12)
    parser.add_argument("--eval-ticks", type=int, default=120)
    args = parser.parse_args()

    variants = [
        "A_baseline",
        "B_adaptive_gate",
        "C_future_prediction",
        "D_adaptive_depth",
        "E_all_three",
    ]

    all_results = {}
    for v in variants:
        _, res = train_ablation_variant(
            v,
            dagger_iters=args.dagger_iters,
            updates_per_iter=args.updates_per_iter,
            eval_ticks=args.eval_ticks,
        )
        all_results[v] = res

    print("\n" + "=" * 100)
    print("                      5-WAY ARCHITECTURAL COMPONENT ABLATION SUMMARY")
    print("=" * 100)
    print(f"{'Variant':<22} | {'20-Seed Pel':<12} | {'20-Seed Cat':<12} | {'Dead-End Cat':<12} | {'Wall Recov':<10} | {'Ghost Cycles':<12}")
    print("-" * 100)
    for v, r in all_results.items():
        val_pel = r["held_out_20seed"]["mean_pellets"]
        val_cat = r["held_out_20seed"]["total_catches"]
        de_cat = r["diagnostic_3seed"]["dead_end_catches"]
        wall_rec = r["diagnostic_3seed"]["mean_wall_recovery"]
        g_cyc = r["diagnostic_3seed"]["depth_ghost_cycles"]
        print(f"{v:<22} | {val_pel:<12.2f} | {val_cat:<12} | {de_cat:<12} | {wall_rec:<10.1f} | {g_cyc:<12.1f}")
    print("=" * 100)

    # Save to json
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "cognitive_ablation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed JSON results written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
