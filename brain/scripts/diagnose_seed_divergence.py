"""Diagnose internal telemetry differences between good and reckless training seeds in Pseudo-Brain.

Compares Good Seed (43) vs Bad/Reckless Seed (46) on identical held-out
procedural worlds to determine why identical architectures develop radically different
safety policies.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import pathlib
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.data.curriculum_dataset import CurriculumDataset, CurriculumDatasetConfig
from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.diagnostic_policies import ScriptedMazeChasePlannerPolicy
from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
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


@dataclass
class TelemetrySummary:
    seed: int
    pellets: float
    catches: int
    danger_steps: int
    open_steps: int
    danger_recall: float
    danger_precision: float
    mean_hazard_cycles: float
    mean_open_cycles: float
    mean_hazard_gate_alpha: float
    mean_open_gate_alpha: float
    mean_thought_effective_rank: float
    mean_thought_slot_similarity: float
    mean_action_entropy_danger: float
    mean_action_entropy_open: float
    evasion_rate: float
    suicide_rate: float
    wall_collision_rate: float


def train_model_on_seed(train_seed: int, dagger_iters: int = 2, updates_per_iter: int = 12) -> IreneBrainModel:
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
    model.topological_goal_head = None
    model.counterfactual_foresight_head = None

    training_config = TrainingConfig(
        schema_version=1,
        run=RunConfig(
            name=f"diag-seed{train_seed}",
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
        counterfactual_weight=0.0,
        topological_goal_weight=0.0,
    )
    optimizer = system.optimizer if hasattr(system, "optimizer") else None

    for it in range(dagger_iters):
        res = distiller.run_dagger_iteration(iteration=it)
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
                        topological_goal_targets=None,
                        topological_exit_targets=None,
                    )
                    
                    if cog_out.total_loss.requires_grad:
                        optimizer.zero_grad()
                        cog_out.total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

    model.eval()
    return model


def compute_thoughtlet_effective_rank(thoughtlets: torch.Tensor) -> float:
    if thoughtlets.dim() == 3:
        thoughtlets = thoughtlets[0]
    u, s, v = torch.linalg.svd(thoughtlets.float(), full_matrices=False)
    s_norm = s / (s.sum() + 1e-8)
    entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-8))
    return float(torch.exp(entropy).item())


def compute_thoughtlet_slot_similarity(thoughtlets: torch.Tensor) -> float:
    if thoughtlets.dim() == 3:
        thoughtlets = thoughtlets[0]
    K = thoughtlets.shape[0]
    normed = F.normalize(thoughtlets.float(), p=2, dim=-1)
    sim_matrix = torch.matmul(normed, normed.T)
    mask = ~torch.eye(K, dtype=torch.bool, device=thoughtlets.device)
    return float(sim_matrix[mask].mean().item())


def run_instrumented_telemetry(
    model: IreneBrainModel,
    eval_seeds: tuple[int, ...],
    ticks_per_eval: int = 120,
) -> TelemetrySummary:
    env = MazeChaseEnv()
    
    total_pellets = 0.0
    total_catches = 0
    danger_steps = 0
    open_steps = 0
    
    true_pos_danger = 0
    false_pos_danger = 0
    false_neg_danger = 0
    
    hazard_cycles_list = []
    open_cycles_list = []
    
    hazard_alpha_list = []
    open_alpha_list = []
    
    effective_ranks = []
    slot_sims = []
    
    action_entropy_danger = []
    action_entropy_open = []
    
    evasion_actions = 0
    suicide_actions = 0
    wall_collisions = 0
    total_decision_steps = 0

    wasd_to_dir = {
        1: (0, -1), # W (up)
        2: (-1, 0), # A (left)
        3: (0, 1),  # S (down)
        4: (1, 0),  # D (right)
    }

    planner = LatentLookaheadPlanner(
        model=model,
        horizon=3,
        gamma=0.95,
        hazard_weight=5.0,
        cognitive_cycles_per_step=model.config.cognitive_cycles,
    )
    policy = LatentLookaheadPolicy(model=model, planner=planner)

    for ev_seed in eval_seeds:
        obs = env.reset(seed=ev_seed)
        policy.reset(episode_seed=ev_seed)
        state = model.initial_state(batch_size=1)
        prev_pos = (env._player_x, env._player_y)
        
        for tick in range(ticks_per_eval):
            total_decision_steps += 1
            device = next(model.parameters()).device
            rgb = _rgb_tensor((obs.rgb,), device=device, resolution=getattr(model, "input_resolution", None))
            prev_ctrl_vec = torch.tensor([control_to_vector(obs.previous_control)], dtype=torch.float32, device=device)
            dt = torch.tensor([0.016], dtype=torch.float32, device=device)
            
            with torch.no_grad():
                out = model(rgb, prev_ctrl_vec, dt, state)
                state = out.next_state
            
            px, py = env._player_x, env._player_y
            ghost_dists = [math.hypot(px - gx, py - gy) for (gx, gy) in env._ghosts]
            min_ghost_dist = min(ghost_dists) if ghost_dists else 999.0
            nearest_ghost_idx = ghost_dists.index(min_ghost_dist) if ghost_dists else -1
            
            is_danger = min_ghost_dist <= 2.5
            
            # Extract hazard prediction from world predictions / diagnostics
            diag = out.diagnostics
            pred_danger_prob = 0.0
            if hasattr(out, "world") and out.world is not None:
                if hasattr(out.world, "occurrence_logits") and out.world.occurrence_logits is not None:
                    pred_danger_prob = max(pred_danger_prob, float(torch.sigmoid(out.world.occurrence_logits).max().item()))
                if hasattr(out.world, "urgency") and out.world.urgency is not None:
                    pred_danger_prob = max(pred_danger_prob, float(torch.sigmoid(out.world.urgency).max().item()))
            if diag.future_trajectory_predictions is not None:
                fut = diag.future_trajectory_predictions
                if "collision_logits" in fut:
                    pred_danger_prob = max(pred_danger_prob, float(torch.sigmoid(fut["collision_logits"]).max().item()))
                elif "future_hazards" in fut:
                    pred_danger_prob = max(pred_danger_prob, float(torch.sigmoid(fut["future_hazards"]).max().item()))
            elif diag.cycles_completed > 2:
                pred_danger_prob = 1.0

            if is_danger:
                danger_steps += 1
                if pred_danger_prob > 0.5:
                    true_pos_danger += 1
                else:
                    false_neg_danger += 1
            else:
                open_steps += 1
                if pred_danger_prob > 0.5:
                    false_pos_danger += 1

            cycles = float(diag.cycles_completed)
            if is_danger:
                hazard_cycles_list.append(cycles)
            else:
                open_cycles_list.append(cycles)

            alpha = 1.0
            if diag.thought_update_gates is not None:
                alpha = float(diag.thought_update_gates.mean().item())
            if is_danger:
                hazard_alpha_list.append(alpha)
            else:
                open_alpha_list.append(alpha)

            st = state
            if hasattr(st, "thoughts") and st.thoughts is not None:
                th = st.thoughts
                if th.dim() >= 3:
                    th_flat = th.reshape(th.shape[0], th.shape[1], -1)[0]
                else:
                    th_flat = th
                effective_ranks.append(compute_thoughtlet_effective_rank(th_flat))
                slot_sims.append(compute_thoughtlet_slot_similarity(th_flat))

            button_logits = out.action.button_logits[0]
            wasd_keys = [int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)]
            wasd_logits = torch.tensor([0.0] + [button_logits[k].item() for k in wasd_keys], device=device)
            probs = F.softmax(wasd_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
            
            if is_danger:
                action_entropy_danger.append(entropy)
            else:
                action_entropy_open.append(entropy)

            control = policy.act(obs)
            is_w = int(HidKey.W) in control.keys_down
            is_a = int(HidKey.A) in control.keys_down
            is_s = int(HidKey.S) in control.keys_down
            is_d = int(HidKey.D) in control.keys_down
            chosen_dx = (1 if is_d else 0) - (1 if is_a else 0)
            chosen_dy = (1 if is_s else 0) - (1 if is_w else 0)

            if is_danger and nearest_ghost_idx >= 0:
                gx, gy = env._ghosts[nearest_ghost_idx]
                v_gx, v_gy = gx - px, gy - py
                dot = chosen_dx * v_gx + chosen_dy * v_gy
                if dot < 0:
                    evasion_actions += 1
                elif dot > 0:
                    suicide_actions += 1

            outcome = env.step(control)
            obs = outcome.observation
            
            if (env._player_x, env._player_y) == prev_pos and (chosen_dx != 0 or chosen_dy != 0):
                wall_collisions += 1
            prev_pos = (env._player_x, env._player_y)

        total_pellets += env._pellets_eaten
        total_catches += env._times_caught

    danger_recall = (true_pos_danger / max(1, danger_steps)) * 100.0
    danger_precision = (true_pos_danger / max(1, true_pos_danger + false_pos_danger)) * 100.0
    
    return TelemetrySummary(
        seed=-1,
        pellets=total_pellets / len(eval_seeds),
        catches=total_catches,
        danger_steps=danger_steps,
        open_steps=open_steps,
        danger_recall=danger_recall,
        danger_precision=danger_precision,
        mean_hazard_cycles=float(torch.tensor(hazard_cycles_list).mean().item()) if hazard_cycles_list else 3.0,
        mean_open_cycles=float(torch.tensor(open_cycles_list).mean().item()) if open_cycles_list else 3.0,
        mean_hazard_gate_alpha=float(torch.tensor(hazard_alpha_list).mean().item()) if hazard_alpha_list else 1.0,
        mean_open_gate_alpha=float(torch.tensor(open_alpha_list).mean().item()) if open_alpha_list else 1.0,
        mean_thought_effective_rank=float(torch.tensor(effective_ranks).mean().item()) if effective_ranks else 1.0,
        mean_thought_slot_similarity=float(torch.tensor(slot_sims).mean().item()) if slot_sims else 0.0,
        mean_action_entropy_danger=float(torch.tensor(action_entropy_danger).mean().item()) if action_entropy_danger else 0.0,
        mean_action_entropy_open=float(torch.tensor(action_entropy_open).mean().item()) if action_entropy_open else 0.0,
        evasion_rate=(evasion_actions / max(1, danger_steps)) * 100.0,
        suicide_rate=(suicide_actions / max(1, danger_steps)) * 100.0,
        wall_collision_rate=(wall_collisions / max(1, total_decision_steps)) * 100.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose internal telemetry across training seeds")
    parser.add_argument("--good-seed", type=int, default=43, help="Training seed for good model (default 43)")
    parser.add_argument("--bad-seed", type=int, default=46, help="Training seed for reckless model (default 46)")
    parser.add_argument("--eval-seeds-count", type=int, default=20, help="Number of held-out evaluation seeds")
    parser.add_argument("--ticks-per-eval", type=int, default=120, help="Decision steps per evaluation episode")
    args = parser.parse_args()

    eval_seeds = tuple(range(2001, 2001 + args.eval_seeds_count))

    print("=" * 80)
    print(f"  INTERNAL TELEMETRY DIAGNOSIS: SEED {args.good_seed} (GOOD) vs SEED {args.bad_seed} (RECKLESS)")
    print("=" * 80)

    print(f"\n[1/4] Training Good Model on Seed {args.good_seed}...")
    good_model = train_model_on_seed(args.good_seed)

    print(f"\n[2/4] Training Reckless Model on Seed {args.bad_seed}...")
    bad_model = train_model_on_seed(args.bad_seed)

    print(f"\n[3/4] Instrumenting Good Model (Seed {args.good_seed}) across {len(eval_seeds)} held-out worlds...")
    good_telemetry = run_instrumented_telemetry(good_model, eval_seeds, args.ticks_per_eval)
    good_telemetry.seed = args.good_seed

    print(f"\n[4/4] Instrumenting Reckless Model (Seed {args.bad_seed}) across {len(eval_seeds)} held-out worlds...")
    bad_telemetry = run_instrumented_telemetry(bad_model, eval_seeds, args.ticks_per_eval)
    bad_telemetry.seed = args.bad_seed

    print("\n" + "=" * 80)
    print("  INTERNAL TELEMETRY COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Metric':<38} | {'Good (Seed ' + str(args.good_seed) + ')':<18} | {'Reckless (Seed ' + str(args.bad_seed) + ')':<18} | {'Diagnosis / Finding':<20}")
    print("-" * 102)
    print(f"{'Mean Pellets Collected':<38} | {good_telemetry.pellets:<18.2f} | {bad_telemetry.pellets:<18.2f} | {'Higher is better'}")
    print(f"{'Total Ghost Catches':<38} | {good_telemetry.catches:<18} | {bad_telemetry.catches:<18} | {'Lower is safer'}")
    print(f"{'Danger Prediction Recall (%)':<38} | {good_telemetry.danger_recall:<18.1f} | {bad_telemetry.danger_recall:<18.1f} | {'Hazard detection rate'}")
    print(f"{'Danger Prediction Precision (%)':<38} | {good_telemetry.danger_precision:<18.1f} | {bad_telemetry.danger_precision:<18.1f} | {'Hazard precision'}")
    print(f"{'Hazard Cycles Allocated (C_hazard)':<38} | {good_telemetry.mean_hazard_cycles:<18.2f} | {bad_telemetry.mean_hazard_cycles:<18.2f} | {'Extra compute under threat'}")
    print(f"{'Open-Space Cycles (C_open)':<38} | {good_telemetry.mean_open_cycles:<18.2f} | {bad_telemetry.mean_open_cycles:<18.2f} | {'Baseline compute'}")
    print(f"{'Surprise Gate alpha (Hazard)':<38} | {good_telemetry.mean_hazard_gate_alpha:<18.3f} | {bad_telemetry.mean_hazard_gate_alpha:<18.3f} | {'Reconsideration intensity'}")
    print(f"{'Thoughtlet Effective Rank (1-4)':<38} | {good_telemetry.mean_thought_effective_rank:<18.2f} | {bad_telemetry.mean_thought_effective_rank:<18.2f} | {'Slot dimensional diversity'}")
    print(f"{'Thoughtlet Pairwise Cosine Sim':<38} | {good_telemetry.mean_thought_slot_similarity:<18.3f} | {bad_telemetry.mean_thought_slot_similarity:<18.3f} | {'Lower = specialized'}")
    print(f"{'Action Entropy under Danger':<38} | {good_telemetry.mean_action_entropy_danger:<18.3f} | {bad_telemetry.mean_action_entropy_danger:<18.3f} | {'Decision hesitation/spread'}")
    print(f"{'Active Evasion Rate (%)':<38} | {good_telemetry.evasion_rate:<18.1f} | {bad_telemetry.evasion_rate:<18.1f} | {'Moving AWAY from ghost'}")
    print(f"{'Suicide / Toward Ghost Rate (%)':<38} | {good_telemetry.suicide_rate:<18.1f} | {bad_telemetry.suicide_rate:<18.1f} | {'Moving INTO ghost'}")
    print(f"{'Wall Collision Rate (%)':<38} | {good_telemetry.wall_collision_rate:<18.1f} | {bad_telemetry.wall_collision_rate:<18.1f} | {'Wall bumping'}")
    print("=" * 102)

    out_path = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "seed_divergence_telemetry.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "good_seed": good_telemetry.__dict__,
        "bad_seed": bad_telemetry.__dict__,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved diagnosis telemetry report to: {out_path}")


if __name__ == "__main__":
    main()
