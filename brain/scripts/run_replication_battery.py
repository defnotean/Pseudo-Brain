from __future__ import annotations
import argparse
from dataclasses import replace
import json
import math
import os
import pathlib
import sys
import time
from typing import Any
import torch

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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

TRAIN_SEEDS = (42, 43, 44, 45, 46)
HELD_OUT_SEEDS = tuple(range(2001, 2021))


def train_and_eval_instance(
    variant_name: str,
    train_seed: int,
    *,
    use_counterfactual: bool,
    use_topological_goal: bool,
    dagger_iters: int = 2,
    updates_per_iter: int = 12,
    eval_ticks: int = 120,
) -> dict[str, Any]:
    torch.manual_seed(train_seed)
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
        enable_adaptive_cognition=True,
    )

    if not use_topological_goal:
        model.topological_goal_head = None
    if not use_counterfactual:
        model.counterfactual_foresight_head = None

    training_config = TrainingConfig(
        schema_version=1,
        run=RunConfig(
            name=f"rep-{variant_name}-seed{train_seed}",
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

    curriculum_cfg = CurriculumDatasetConfig(
        sequence_length=8,
        sequence_count=40,
        scenario="mixed",
    )
    curriculum_dataset = CurriculumDataset(curriculum_cfg)
    for i in range(min(20, len(curriculum_dataset))):
        distiller.buffer.add_sequence(curriculum_dataset[i])

    cog_loss_fn = CognitiveAuxiliaryLoss(
        future_weight=1.0,
        gate_surprise_weight=1.0,
        halting_weight=0.5,
        counterfactual_weight=1.0 if use_counterfactual else 0.0,
        topological_goal_weight=1.0 if use_topological_goal else 0.0,
    )
    optimizer = system.optimizer if hasattr(system, "optimizer") else None

    final_loss = 0.0
    for it in range(dagger_iters):
        res = distiller.run_dagger_iteration(iteration=it)
        final_loss = res.mean_train_loss
        
        if optimizer is not None and len(distiller.buffer) > 0:
            model.train()
            for _ in range(updates_per_iter):
                batch = distiller.buffer.sample_batch(batch_size=2, burn_in_steps=2)
                for seq in batch.sequences:
                    if len(seq.transitions) < 6:
                        continue
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
                    
                    out = model(rgb_0, prev_c, dt, st)
                    
                    act_vec = control_to_vector(t0_obs.action_target)
                    wasd = [act_vec[int(k)] for k in (HidKey.W, HidKey.A, HidKey.S, HidKey.D)]
                    if any(w > 0.5 for w in wasd):
                        act_idx = 1 + wasd.index(max(wasd))
                    else:
                        act_idx = 0
                    
                    haz_flags = []
                    for h_off in (1, 3, 5):
                        idx = min(len(seq.transitions) - 1, 2 + h_off)
                        fut_t = seq.transitions[idx]
                        is_haz = 1.0 if any(ev in ("collision", "caught") for ev in fut_t.event_targets) else 0.0
                        haz_flags.append([is_haz])
                    
                    actual_collisions = torch.tensor([haz_flags], dtype=torch.float32)
                    executed_actions = torch.tensor([act_idx], dtype=torch.long)
                    future_disp_targets = torch.zeros((1, 3, 2), dtype=torch.float32)
                    
                    vec_targets = torch.zeros((1, 2), dtype=torch.float32)
                    if act_idx == 1:
                        vec_targets[0, 1] = -1.0
                    elif act_idx == 2:
                        vec_targets[0, 0] = -1.0
                    elif act_idx == 3:
                        vec_targets[0, 1] = 1.0
                    elif act_idx == 4:
                        vec_targets[0, 0] = 1.0

                    is_hazard = any(ev in ("collision", "caught") for ev in t0_obs.event_targets) or (actual_collisions[0, 0, 0].item() > 0.5)
                    hazard_mask = torch.tensor([is_hazard], dtype=torch.bool)
                    
                    cog_out = cog_loss_fn(
                        diagnostics=out.diagnostics,
                        future_disp_targets=future_disp_targets,
                        executed_actions=executed_actions,
                        actual_collisions=actual_collisions,
                        hazard_mask=hazard_mask,
                        topological_goal_targets=vec_targets if use_topological_goal else None,
                        topological_exit_targets=executed_actions if use_topological_goal else None,
                    )
                    
                    if cog_out.total_loss.requires_grad:
                        optimizer.zero_grad()
                        cog_out.total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

    train_sec = time.perf_counter() - t0

    # Evaluate across 20 held-out validation seeds
    env = MazeChaseEnv()
    expert = ScriptedMazeChasePlannerPolicy()
    val_pellets = 0
    val_catches = 0
    for s in HELD_OUT_SEEDS:
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

    mean_held_pellets = val_pellets / len(HELD_OUT_SEEDS)
    total_held_catches = val_catches
    digest = compute_model_param_digest(model)

    return {
        "variant": variant_name,
        "train_seed": train_seed,
        "digest": digest,
        "final_loss": round(float(final_loss), 4),
        "train_sec": round(train_sec, 2),
        "mean_pellets": round(mean_held_pellets, 2),
        "total_catches": total_held_catches,
    }


def stats(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dagger-iters", type=int, default=2)
    parser.add_argument("--updates-per-iter", type=int, default=12)
    parser.add_argument("--eval-ticks", type=int, default=120)
    args = parser.parse_args()

    variant_specs = [
        ("E_champion", False, False),
        ("F_counterfactual", True, False),
        ("G_topological", True, True),
    ]

    all_results = {}

    print("=" * 90)
    print("  MULTI-SEED REPLICATION BATTERY (5 Training Seeds x 20 Held-Out Validation Seeds)")
    print("=" * 90)

    for vid, use_cf, use_topo in variant_specs:
        print(f"\n>>> Running Variant: {vid}")
        runs = []
        for s in TRAIN_SEEDS:
            rec = train_and_eval_instance(
                variant_name=vid,
                train_seed=s,
                use_counterfactual=use_cf,
                use_topological_goal=use_topo,
                dagger_iters=args.dagger_iters,
                updates_per_iter=args.updates_per_iter,
                eval_ticks=args.eval_ticks,
            )
            runs.append(rec)
            print(
                f"  Seed {s:2d} | Loss: {rec['final_loss']:.4f} | Pellets: {rec['mean_pellets']:.2f} | Catches: {rec['total_catches']:3d} | Digest: {rec['digest']}"
            )
        all_results[vid] = runs

    print("\n" + "=" * 90)
    print("  STATISTICAL REPLICATION SUMMARY (5 Training Seeds x 20 Held-Out Seeds = 100 Episodes/Variant)")
    print("=" * 90)
    print(f"{'Variant':<22} | {'Mean Pellets':<18} | {'Total Catches':<18} | {'Final Loss':<15}")
    print("-" * 90)

    summary = {}
    for vid, _, _ in variant_specs:
        pellets = [r["mean_pellets"] for r in all_results[vid]]
        catches = [r["total_catches"] for r in all_results[vid]]
        losses = [r["final_loss"] for r in all_results[vid]]

        p_mean, p_std = stats(pellets)
        c_mean, c_std = stats(catches)
        l_mean, l_std = stats(losses)

        summary[vid] = {
            "mean_pellets": round(p_mean, 2),
            "std_pellets": round(p_std, 2),
            "mean_catches": round(c_mean, 1),
            "std_catches": round(c_std, 1),
            "mean_loss": round(l_mean, 4),
            "std_loss": round(l_std, 4),
            "runs": all_results[vid],
        }

        print(
            f"{vid:<22} | {p_mean:.2f} +/- {p_std:.2f}         | {c_mean:.1f} +/- {c_std:.1f}         | {l_mean:.4f} +/- {l_std:.4f}"
        )

    print("=" * 90)

    out_path = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "replication_battery_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDetailed replication results saved to {out_path}")


if __name__ == "__main__":
    main()
