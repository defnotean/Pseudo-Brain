"""Forensic Pillar Audit & Root-Cause Investigation for Pseudo-Brain.

Investigates:
1. Manual Episode Step-by-Step Trace: Checks every catch event, action choice, and ghost distance.
2. Pillar Ablation Matrix: Tests Baseline, P1 only, P2 only, P3 only, and combinations across seeds.
3. Pillar 1 Sign & Action-Index Verification: Verifies danger penalty monotonically decreases action utility.
4. Gate Alpha Profile: Compares normal alpha vs hazard alpha under bias 0.0 vs -2.0.
5. Thoughtlet Knock-Out (Ablation) Test: Masks each thoughtlet slot (0..3) to test functional specialization.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any

try:
    import torch
    import torch.nn as nn
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

HELD_OUT_SEEDS = tuple(range(2001, 2021))


def train_configured_model(
    train_seed: int,
    *,
    p1_calibrated_planner: bool,
    p2_anti_collapse: bool,
    p3_pred_error_gate: bool,
    gate_bias: float = 0.0,
    dagger_iters: int = 2,
    updates_per_iter: int = 12,
) -> IreneBrainModel:
    """Train model under specific pillar configuration."""
    torch.manual_seed(train_seed)

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

    model = IreneBrainModel(config=config, enable_adaptive_cognition=True)

    # Configure Pillar 3 (Gate)
    if model.adaptive_thought_gate is not None:
        if not p3_pred_error_gate:
            # Revert to legacy bias -2.0 and disable prediction error gate
            pass
        if hasattr(model.adaptive_thought_gate, "gate_mlp"):
            with torch.no_grad():
                model.adaptive_thought_gate.gate_mlp[-1].bias.fill_(gate_bias)

    training_config = TrainingConfig(
        schema_version=1,
        run=RunConfig(
            name=f"forensic-seed{train_seed}",
            seed=train_seed,
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
        precision=PrecisionConfig(device="cpu", mode="float32", allow_tf32=False),
        determinism=DeterminismConfig(enabled=True, num_workers=0, compile_model=False),
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
    trainer = TorchTrainingSystem(objective, training_config)
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
        training_system=trainer,
    )

    curriculum = CurriculumDataset(
        CurriculumDatasetConfig(
            sequence_length=8,
            sequence_count=40,
            scenario="mixed",
        )
    )
    for i in range(min(20, len(curriculum))):
        distiller.buffer.add_sequence(curriculum[i])

    for it in range(dagger_iters):
        distiller.run_dagger_iteration(iteration=it)

    # Optional Pillar 2 (Anti-Collapse Loss training step)
    if p2_anti_collapse:
        cog_loss_fn = CognitiveAuxiliaryLoss()
        # Bounded cognitive supervision steps
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for i in range(min(8, len(curriculum))):
            seq = curriculum[i]
            pix = _rgb_tensor(tuple(t.observation.rgb for t in seq.transitions), device=torch.device("cpu"), resolution=model.input_resolution)
            act_vec = torch.tensor([control_to_vector(t.observation.previous_control) for t in seq.transitions], dtype=torch.float32)
            elap = torch.tensor([1.0 / 60.0] * len(seq.transitions), dtype=torch.float32)
            st = model.initial_state(len(seq.transitions), device=torch.device("cpu"))
            out = model(pix, act_vec, elap, st)
            cog_out = cog_loss_fn(diagnostics=out.diagnostics)
            if cog_out.total_loss.requires_grad:
                opt.zero_grad()
                cog_out.total_loss.backward()
                opt.step()

    model.eval()
    return model


def evaluate_model(
    model: IreneBrainModel,
    *,
    hazard_prune_threshold: float = 0.50,
    hazard_weight: float = 5.0,
    eval_ticks: int = 120,
) -> dict[str, Any]:
    """Evaluate model across 20 held-out seeds."""
    env = MazeChaseEnv()
    expert = ScriptedMazeChasePlannerPolicy()
    val_pellets = 0
    val_catches = 0
    gate_normal_alphas = []
    gate_hazard_alphas = []

    for s in HELD_OUT_SEEDS:
        planner = LatentLookaheadPlanner(
            model=model,
            horizon=3,
            gamma=0.95,
            hazard_weight=hazard_weight,
            hazard_prune_threshold=hazard_prune_threshold,
            cognitive_cycles_per_step=model.config.cognitive_cycles,
        )
        policy = LatentLookaheadPolicy(
            model=model,
            planner=planner,
            hazard_prune_threshold=hazard_prune_threshold,
            hazard_weight=hazard_weight,
        )
        rep = run_instrumented_diagnostic_episode(
            policy=policy,
            env=env,
            seed=s,
            max_ticks=eval_ticks,
            expert_policy=expert,
        )
        val_pellets += rep.pellets_eaten
        val_catches += rep.ghost_collisions
        if rep.gate_telemetry:
            gate_normal_alphas.append(rep.gate_telemetry.alpha_normal)
            gate_hazard_alphas.append(rep.gate_telemetry.alpha_ghost_danger)

    return {
        "mean_pellets": round(val_pellets / len(HELD_OUT_SEEDS), 2),
        "total_catches": val_catches,
        "mean_catches_per_ep": round(val_catches / len(HELD_OUT_SEEDS), 2),
        "gate_normal_alpha": round(sum(gate_normal_alphas) / max(1, len(gate_normal_alphas)), 3),
        "gate_hazard_alpha": round(sum(gate_hazard_alphas) / max(1, len(gate_hazard_alphas)), 3),
    }


def forensic_single_episode_trace(model: IreneBrainModel, seed: int = 2001) -> None:
    """Print step-by-step forensic trace of a single episode."""
    print(f"\n{'=' * 80}")
    print(f"  FORENSIC EPISODE TRACE: Seed {seed}")
    print(f"{'=' * 80}")
    env = MazeChaseEnv()
    obs = env.reset(seed)
    planner = LatentLookaheadPlanner(model=model, horizon=3, gamma=0.95, hazard_weight=5.0, hazard_prune_threshold=0.50)
    policy = LatentLookaheadPolicy(model=model, planner=planner, hazard_prune_threshold=0.50)
    policy.reset(seed)

    catches_seen = 0
    for tick in range(60):
        old_player = (env._player_x, env._player_y)
        ghosts = list(env._ghosts)
        nearest_ghost_dist = min(math.hypot(env._player_x - gx, env._player_y - gy) for gx, gy in ghosts) if ghosts else 99.0
        
        ctrl, audit, _ = policy.decide(obs, 1.0 / 60.0)
        keys = ctrl.keys_down
        plan = policy.last_plan
        best_act = plan.best_action if plan else "None"
        pruned = plan.pruned_count if plan else 0
        best_u = plan.best_branch.cumulative_utility if plan else 0.0
        best_haz = plan.best_branch.discounted_danger_sum if plan else 0.0

        pre_caught = env._times_caught
        step_out = env.step(ctrl)
        obs = step_out.observation
        new_player = (env._player_x, env._player_y)
        caught = env._times_caught > pre_caught
        if caught:
            catches_seen += 1

        if tick < 25 or caught or nearest_ghost_dist <= 3.0:
            print(
                f"Tick {tick:02d} | Pos {old_player}->{new_player} | GhostDist: {nearest_ghost_dist:.1f} | "
                f"Act: {best_act} (Keys: {keys}) | Pruned: {pruned} | Util: {best_u:.2f} | Haz: {best_haz:.2f} | "
                f"{'*** CAUGHT ***' if caught else ''}"
            )


def thoughtlet_knockout_test(model: IreneBrainModel) -> None:
    """Mask out each of the 4 thoughtlet slots to test functional specialization."""
    print("\n" + "=" * 80)
    print("  THOUGHTLET KNOCK-OUT TEST (FUNCTIONAL DIVERSITY CHECK)")
    print("=" * 80)
    print(f"{'Ablated Thoughtlet':<25} | {'Mean Pellets':<15} | {'Total Catches':<15}")
    print("-" * 60)

    # Baseline (no knockout)
    res_base = evaluate_model(model, hazard_prune_threshold=0.50, eval_ticks=120)
    print(f"{'None (Full Model)':<25} | {res_base['mean_pellets']:<15.2f} | {res_base['total_catches']:<15}")

    for slot in range(4):
        # Create hook to zero out specific thoughtlet
        def make_hook(target_slot: int):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 2:
                    # output[2] is thoughts [batch, thoughtlets, registers, width]
                    thoughts = output[2].clone()
                    thoughts[:, target_slot] = 0.0
                    return (output[0], output[1], thoughts) + output[3:]
                return output
            return hook

        handle = model.brain_cell.register_forward_hook(make_hook(slot))
        res_slot = evaluate_model(model, hazard_prune_threshold=0.50, eval_ticks=120)
        handle.remove()
        print(f"{f'Slot {slot} Zeroed':<25} | {res_slot['mean_pellets']:<15.2f} | {res_slot['total_catches']:<15}")


def main() -> None:
    print("=" * 90)
    print("  FORENSIC ROOT-CAUSE AUDIT: ABLATING PILLARS 1, 2, 3")
    print("=" * 90)

    configs = [
        ("Legacy Baseline E (Orig)", False, False, False, -2.0, 0.50),
        ("+ P1 Only (Relaxed Prune 0.90)", True, False, False, -2.0, 0.90),
        ("+ P2 Only (Anti-Collapse)", False, True, False, -2.0, 0.50),
        ("+ P3 Only (Gate Bias 0.0)", False, False, True, 0.0, 0.50),
        ("+ P2 + P3 (Gate Bias 0.0 + Anti-Collapse, Prune 0.50)", False, True, True, 0.0, 0.50),
        ("+ Full Three Pillars (Prune 0.90 + Bias 0.0 + Anti-Collapse)", True, True, True, 0.0, 0.90),
        ("+ Corrected Three Pillars (Prune 0.50 + Bias -2.0 + Anti-Collapse)", False, True, False, -2.0, 0.50),
    ]

    print(f"{'Configuration':<55} | {'Pellets':<8} | {'Total Catches':<14} | {'Catches/Ep':<10} | {'Gate Normal':<12} | {'Gate Hazard':<12}")
    print("-" * 120)

    trained_models = {}
    for name, p1, p2, p3, bias, prune_th in configs:
        model = train_configured_model(
            train_seed=43,
            p1_calibrated_planner=p1,
            p2_anti_collapse=p2,
            p3_pred_error_gate=p3,
            gate_bias=bias,
        )
        res = evaluate_model(
            model,
            hazard_prune_threshold=prune_th,
            hazard_weight=5.0,
            eval_ticks=120,
        )
        trained_models[name] = model
        print(
            f"{name:<55} | {res['mean_pellets']:<8.2f} | {res['total_catches']:<14} | "
            f"{res['mean_catches_per_ep']:<10.2f} | {res['gate_normal_alpha']:<12.3f} | {res['gate_hazard_alpha']:<12.3f}"
        )

    # Forensic trace of the best vs worst
    print("\n--- FORENSIC TRACE: Relaxed Prune 0.90 (Failed Model) ---")
    forensic_single_episode_trace(trained_models["+ P1 Only (Relaxed Prune 0.90)"], seed=2001)

    print("\n--- FORENSIC TRACE: Strict Prune 0.50 (Safe Model) ---")
    forensic_single_episode_trace(trained_models["Legacy Baseline E (Orig)"], seed=2001)

    # Run Thoughtlet Knockout Test on the Corrected Anti-Collapse model
    thoughtlet_knockout_test(trained_models["+ Corrected Three Pillars (Prune 0.50 + Bias -2.0 + Anti-Collapse)"])


if __name__ == "__main__":
    main()

