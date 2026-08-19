"""Diagnose internal telemetry differences between good and reckless training seeds in Pseudo-Brain.

Compares Good Seed (e.g. 43) vs Bad/Reckless Seed (e.g. 46) on identical held-out
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
        ),
        optimization=OptimizationConfig(
            batch_size=8,
            learning_rate=0.001,
            warmup_steps=4,
            weight_decay=0.01,
            gradient_clip_norm=1.0,
            seed=train_seed,
        ),
        precision=PrecisionConfig(mode="fp32"),
        resource=ResourceConfig(max_rss_mb=4096, max_threads=1, allow_gpu=False),
        determinism=DeterminismConfig(enforce_single_thread=True, seed=train_seed),
        logging=LoggingConfig(log_interval_steps=10),
    )

    system = TorchTrainingSystem(training_config)
    cognitive_aux = CognitiveAuxiliaryLoss(
        prediction_weight=0.25,
        hazard_weight=0.25,
        depth_weight=0.05,
    )
    curriculum = CurriculumDataset(
        CurriculumDatasetConfig(
            scenario="mixed",
            train_episodes=16,
            val_episodes=4,
            test_episodes=4,
            episode_length=8,
            seed=train_seed,
        )
    )
    dagger = DAggerDistiller(
        DAggerConfig(
            dagger_iterations=dagger_iters,
            rollouts_per_iter=4,
            rollout_horizon=8,
            updates_per_iter=updates_per_iter,
            beta_decay=0.6,
            batch_size=8,
            seed=train_seed,
        )
    )

    expert_planner = ScriptedMazeChasePlannerPolicy(lookahead_steps=6)
    env = MazeChaseEnv()

    for it in range(dagger.config.dagger_iterations):
        dagger.run_dagger_iteration(
            student_model=model,
            env=env,
            expert_planner=expert_planner,
            training_system=system,
            dataset=curriculum,
            cognitive_loss_fn=cognitive_aux,
            iteration_index=it,
        )
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
        HidKey.KEY_W: (0, -1),
        HidKey.KEY_S: (0, 1),
        HidKey.KEY_A: (-1, 0),
        HidKey.KEY_D: (1, 0),
    }

    for ev_seed in eval_seeds:
        obs = env.reset(seed=ev_seed)
        state = model.initial_state(batch_size=1)
        prev_pos = (obs.player_pos[0], obs.player_pos[1])
        
        for tick in range(ticks_per_eval):
            total_decision_steps += 1
            obs_tensor = _rgb_tensor([obs.screen_rgb])
            prev_ctrl = control_to_vector(obs.last_applied_control)
            
            with torch.no_grad():
                out = model(
                    pixels=obs_tensor,
                    previous_control=prev_ctrl.unsqueeze(0) if prev_ctrl.dim() == 1 else prev_ctrl,
                    state=state,
                )
            state = out.next_state

            px, py = obs.player_pos
            ghost_dists = [math.hypot(px - gx, py - gy) for (gx, gy) in obs.ghost_positions]
            min_ghost_dist = min(ghost_dists) if ghost_dists else 999.0
            nearest_ghost_idx = ghost_dists.index(min_ghost_dist) if ghost_dists else -1
            
            is_danger = min_ghost_dist <= 2.5
            
            pred_danger_prob = float(torch.sigmoid(out.action_logits[:, 0]).item())
            if hasattr(model, "hazard_head") and model.hazard_head is not None:
                pred_danger_prob = float(torch.sigmoid(model.hazard_head(state.thoughtlets)).mean().item())
            elif hasattr(out, "hazard_pred") and out.hazard_pred is not None:
                pred_danger_prob = float(torch.sigmoid(out.hazard_pred).mean().item())
            else:
                pred_danger_prob = 1.0 if (out.cognitive_cycles_executed is not None and out.cognitive_cycles_executed.float().mean().item() > 2.0) else 0.0

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

            cycles = float(out.cognitive_cycles_executed.float().mean().item()) if out.cognitive_cycles_executed is not None else 3.0
            if is_danger:
                hazard_cycles_list.append(cycles)
            else:
                open_cycles_list.append(cycles)

            alpha = float(state.thought_gate_alpha.mean().item()) if hasattr(state, "thought_gate_alpha") and state.thought_gate_alpha is not None else 1.0
            if is_danger:
                hazard_alpha_list.append(alpha)
            else:
                open_alpha_list.append(alpha)

            if state.thoughtlets is not None:
                effective_ranks.append(compute_thoughtlet_effective_rank(state.thoughtlets))
                slot_sims.append(compute_thoughtlet_slot_similarity(state.thoughtlets))

            action_logits = out.action_logits
            wasd_indices = [HidKey.KEY_W.value, HidKey.KEY_A.value, HidKey.KEY_S.value, HidKey.KEY_D.value]
            wasd_logits = action_logits[0, wasd_indices]
            wasd_probs = F.softmax(wasd_logits, dim=-1)
            entropy = -torch.sum(wasd_probs * torch.log(wasd_probs + 1e-8)).item()
            
            if is_danger:
                action_entropy_danger.append(entropy)
            else:
                action_entropy_open.append(entropy)

            chosen_key_idx = wasd_indices[torch.argmax(wasd_probs).item()]
            chosen_key = HidKey(chosen_key_idx)
            chosen_dx, chosen_dy = wasd_to_dir.get(chosen_key, (0, 0))

            if is_danger and nearest_ghost_idx >= 0:
                gx, gy = obs.ghost_positions[nearest_ghost_idx]
                v_gx, v_gy = gx - px, gy - py
                dot = chosen_dx * v_gx + chosen_dy * v_gy
                if dot < 0:
                    evasion_actions += 1
                elif dot > 0:
                    suicide_actions += 1

            control = obs.last_applied_control
            control = control.with_key_pressed(chosen_key)
            obs = env.step(control)
            
            if (obs.player_pos[0], obs.player_pos[1]) == prev_pos and (chosen_dx != 0 or chosen_dy != 0):
                wall_collisions += 1
            prev_pos = (obs.player_pos[0], obs.player_pos[1])

        total_pellets += obs.pellets_eaten
        total_catches += obs.ghost_catches

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
    print(f"{'Surprise Gate α (Hazard)':<38} | {good_telemetry.mean_hazard_gate_alpha:<18.3f} | {bad_telemetry.mean_hazard_gate_alpha:<18.3f} | {'Reconsideration intensity'}")
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
