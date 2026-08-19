"""Controlled Empirical Experiment: Action-Conditioned Counterfactual Foresight vs Baselines.

Compares:
1. Variant A_baseline: Standard Baseline
2. Variant C_unconditioned_foresight: Passive Multi-Horizon Future Prediction
3. Variant F_counterfactual_foresight: Action-Conditioned Counterfactual Foresight (evaluating & comparing 5 alternative futures)
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import pathlib
import sys
import time
from typing import Any

try:
    import torch
except ModuleNotFoundError:
    torch = None

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
from irene_brain.training.batches import control_to_vector
from irene_brain.training.cognitive_losses import CognitiveAuxiliaryLoss
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
from irene_brain.training.objective import ThoughtFieldObjective, _rgb_tensor
from irene_brain.training.torch_system import TorchTrainingSystem
from irene_brain.types import HidKey


def train_and_eval_variant(
    variant_name: str,
    *,
    use_counterfactual: bool,
    use_passive_foresight: bool,
    dagger_iters: int = 2,
    updates_per_iter: int = 12,
    eval_ticks: int = 120,
) -> dict[str, Any]:
    print(f"\n{'=' * 90}")
    print(f"  EXPERIMENT VARIANT: {variant_name}")
    print(f"{'=' * 90}")
    t0 = time.perf_counter()

    config = ThoughtFieldConfig(
        thoughtlets=4,
        registers_per_thoughtlet=2,
        core_width=32,
        attention_heads=4,
        sensor_tokens=4,
        belief_tokens=4,
        working_memory_tokens=4,
        goal_context_tokens=8,
        cognitive_cycles=3,
    )

    model = IreneBrainModel(
        config=config,
        enable_adaptive_cognition=(use_counterfactual or use_passive_foresight),
    )

    # If only passive foresight is requested, disable counterfactual head
    if use_passive_foresight and not use_counterfactual:
        model.counterfactual_foresight_head = None
    elif use_counterfactual and not use_passive_foresight:
        model.future_trajectory_head = None

    training_config = TrainingConfig(
        schema_version=1,
        run=RunConfig(
            name=f"exp-{variant_name}",
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

    # Pre-seed buffer
    curriculum_cfg = CurriculumDatasetConfig(
        sequence_length=8,
        sequence_count=40,
        scenario="mixed",
    )
    curriculum_dataset = CurriculumDataset(curriculum_cfg)
    for i in range(min(20, len(curriculum_dataset))):
        distiller.buffer.add_sequence(curriculum_dataset[i])

    # Supervised cognitive auxiliary training function
    cog_loss_fn = CognitiveAuxiliaryLoss(
        future_weight=1.0,
        gate_surprise_weight=1.0,
        halting_weight=0.5,
        counterfactual_weight=1.0,
    )
    optimizer = system.optimizer if hasattr(system, "optimizer") else None

    for it in range(dagger_iters):
        res = distiller.run_dagger_iteration(iteration=it)
        
        # Supervise auxiliary heads if present
        if (use_counterfactual or use_passive_foresight) and optimizer is not None and len(distiller.buffer) > 0:
            model.train()
            for _ in range(updates_per_iter):
                # Sample batch from buffer
                batch = distiller.buffer.sample_batch(batch_size=2, burn_in_steps=2)
                for seq in batch.sequences:
                    if len(seq.transitions) < 6:
                        continue
                    # Get ground truth transitions
                    t0_obs = seq.transitions[2]
                    device = next(model.parameters()).device
                    rgb_0 = _rgb_tensor(
                        (t0_obs.observation.rgb,),
                        device=device,
                        resolution=getattr(model, "input_resolution", None),
                    )
                    prev_c = torch.tensor([control_to_vector(t0_obs.observation.previous_control)], dtype=torch.float32, device=device)
                    dt = torch.tensor([0.016], dtype=torch.float32, device=device)
                    st = model.initial_state(batch_size=1)
                    
                    # Forward pass
                    out = model(rgb_0, prev_c, dt, st)
                    
                    # Determine taken action
                    act_vec = control_to_vector(t0_obs.action_target)
                    wasd = [act_vec[int(k)] for k in (HidKey.W, HidKey.A, HidKey.S, HidKey.D)]
                    if any(w > 0.5 for w in wasd):
                        act_idx = 1 + wasd.index(max(wasd))
                    else:
                        act_idx = 0
                    
                    # Check collisions in future steps t+1, t+3, t+5
                    haz_flags = []
                    for h_off in (1, 3, 5):
                        idx = min(len(seq.transitions) - 1, 2 + h_off)
                        fut_t = seq.transitions[idx]
                        is_haz = 1.0 if any(ev in ("collision", "caught") for ev in fut_t.event_targets) else 0.0
                        haz_flags.append([is_haz])
                    
                    actual_collisions = torch.tensor([haz_flags], dtype=torch.float32) # [1, 3, 1]
                    executed_actions = torch.tensor([act_idx], dtype=torch.long) # [1]
                    future_disp_targets = torch.zeros((1, 3, 2), dtype=torch.float32) # [1, 3, 2]
                    
                    is_hazard = any(ev in ("collision", "caught") for ev in t0_obs.event_targets) or (actual_collisions[0, 0, 0].item() > 0.5)
                    hazard_mask = torch.tensor([is_hazard], dtype=torch.bool)
                    
                    cog_out = cog_loss_fn(
                        diagnostics=out.diagnostics,
                        future_disp_targets=future_disp_targets,
                        executed_actions=executed_actions,
                        actual_collisions=actual_collisions,
                        hazard_mask=hazard_mask,
                    )
                    
                    if cog_out.total_loss.requires_grad:
                        optimizer.zero_grad()
                        cog_out.total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
        print(f"  DAgger Step [{it+1}/{dagger_iters}] | Beta: {res.beta:.3f} | Loss: {res.mean_train_loss:.4f}")

    train_sec = time.perf_counter() - t0
    model.eval()

    # 1. 3-Seed Diagnostic Suite
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

    # 2. 20-Seed Held-Out Validation Battery
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
            expert_policy=expert,
        )
        val_pellets += rep.pellets_eaten
        val_catches += rep.ghost_collisions

    mean_held_pellets = val_pellets / len(held_out_seeds)
    total_held_catches = val_catches

    digest = compute_model_param_digest(model)

    return {
        "variant": variant_name,
        "digest": digest,
        "train_time_sec": round(train_sec, 2),
        "diagnostic_3seed": {
            "mean_pellets": round(mean_pellets_3, 2),
            "mean_catches": round(mean_catches_3, 2),
            "dead_end_catches": dead_end_catches_3,
            "mean_wall_recovery": round(wall_recovery_3, 2),
            "gate_normal_alpha": round(gate_normal, 2),
            "gate_wall_alpha": round(gate_wall, 2),
            "gate_ghost_alpha": round(gate_ghost, 2),
        },
        "held_out_20seed": {
            "mean_pellets": round(mean_held_pellets, 2),
            "total_catches": total_held_catches,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dagger-iters", type=int, default=2)
    parser.add_argument("--updates-per-iter", type=int, default=12)
    parser.add_argument("--eval-ticks", type=int, default=120)
    args = parser.parse_args()

    results = {}

    # Variant A: Baseline
    results["A_baseline"] = train_and_eval_variant(
        "A_baseline",
        use_counterfactual=False,
        use_passive_foresight=False,
        dagger_iters=args.dagger_iters,
        updates_per_iter=args.updates_per_iter,
        eval_ticks=args.eval_ticks,
    )

    # Variant C: Unconditioned Passive Foresight
    results["C_unconditioned_foresight"] = train_and_eval_variant(
        "C_unconditioned_foresight",
        use_counterfactual=False,
        use_passive_foresight=True,
        dagger_iters=args.dagger_iters,
        updates_per_iter=args.updates_per_iter,
        eval_ticks=args.eval_ticks,
    )

    # Variant F: Action-Conditioned Counterfactual Foresight
    results["F_counterfactual_foresight"] = train_and_eval_variant(
        "F_counterfactual_foresight",
        use_counterfactual=True,
        use_passive_foresight=False,
        dagger_iters=args.dagger_iters,
        updates_per_iter=args.updates_per_iter,
        eval_ticks=args.eval_ticks,
    )

    print("\n" + "=" * 100)
    print("           COUNTERFACTUAL FORESIGHT vs BASELINES COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Variant':<28} | {'20-Seed Pel':<12} | {'20-Seed Cat':<12} | {'Dead-End Cat':<12} | {'Wall Recov':<10}")
    print("-" * 100)
    for k, v in results.items():
        print(
            f"{k:<28} | "
            f"{v['held_out_20seed']['mean_pellets']:<12.2f} | "
            f"{v['held_out_20seed']['total_catches']:<12d} | "
            f"{v['diagnostic_3seed']['dead_end_catches']:<12d} | "
            f"{v['diagnostic_3seed']['mean_wall_recovery']:<10.1f}"
        )
    print("=" * 100)

    out_path = pathlib.Path(__file__).parent.parent / "artifacts" / "counterfactual_experiment_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Detailed JSON results written to {out_path.resolve()}")


if __name__ == "__main__":
    main()
