"""Mistake-Unrolled Multi-Step Latent Foresight Training Battery.

Trains Pseudo-Brain by recursively unrolling its own predicted intermediate thoughtlet states:
z_t -> \hat{z}_{t+1} -> \hat{z}_{t+2} -> \hat{z}_{t+3}
supervising both latent state consistency (MSE) and hazard foresight at every imagined depth.

Evaluates:
- Latent State MSE by horizon (MSE_1, MSE_2, MSE_3)
- Hazard Discrimination AUROC by horizon (H=1, H=2, H=3) on 50 evaluation worlds (100+ choice points)
- Closed-loop Catches for Direct vs H=1 vs H=2 vs H=3 across Seeds 42-46.
"""

from __future__ import annotations

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

from irene_brain.data.curriculum_dataset import (
    CurriculumDataset,
    CurriculumDatasetConfig,
)
from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.closed_loop_play import (
    EXCLUSIVE_ARGMAX_WASD_V1,
    decode_closed_loop_control,
)
from irene_brain.evaluation.diagnostic_policies import _parse_maze_chase_frame
from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.training.batches import CONTINUOUS_TARGET_INDICES, control_to_vector
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
from irene_brain.types import GenericControl, HidKey

from train_multistep_stable_foresight_battery import (
    _build_action_control,
    audit_per_horizon_discrimination,
)

TRAIN_SEEDS = (42, 43, 44, 45, 46)


def train_mistake_unrolled_seed(
    seed: int,
    num_dagger_iters: int = 4,
    updates_per_iter: int = 24,
) -> IreneBrainModel:
    """Train Pseudo-Brain with deep 3-step autoregressive unrolling on its own imagined states."""
    torch.manual_seed(seed)
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
    device = torch.device("cpu")
    model.to(device)
    model.train()

    training_config = TrainingConfig(
        schema_version=1,
        run=RunConfig(
            name=f"mistake-unroll-seed{seed}",
            seed=seed,
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
            learning_rate=1.5e-3,
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
    system = TorchTrainingSystem(objective, training_config)
    dagger_config = DAggerConfig(
        iterations=num_dagger_iters,
        episodes_per_iteration=3,
        max_ticks_per_episode=80,
        initial_beta=0.8,
        beta_decay=0.80,
        sequence_length=8,
        burn_in_steps=2,
        batch_size=2,
        updates_per_iteration=updates_per_iter,
        ghost_count=2,
        ghost_period=2,
        extra_loops=8,
    )
    distiller = DAggerDistiller(config=dagger_config, student_model=model, training_system=system)

    curriculum_cfg_haz = CurriculumDatasetConfig(sequence_length=8, sequence_count=50, scenario="hazard_evasion")
    curriculum_dataset_haz = CurriculumDataset(curriculum_cfg_haz)
    for i in range(min(30, len(curriculum_dataset_haz))):
        distiller.buffer.add_sequence(curriculum_dataset_haz[i])

    curriculum_cfg_mixed = CurriculumDatasetConfig(sequence_length=8, sequence_count=50, scenario="mixed")
    curriculum_dataset_mixed = CurriculumDataset(curriculum_cfg_mixed)
    for i in range(min(30, len(curriculum_dataset_mixed))):
        distiller.buffer.add_sequence(curriculum_dataset_mixed[i])

    optimizer = system.optimizer if hasattr(system, "optimizer") else None

    cog_loss_fn = CognitiveAuxiliaryLoss(
        future_weight=1.0,
        gate_surprise_weight=1.0,
        halting_weight=0.5,
        counterfactual_weight=3.0,
        topological_goal_weight=1.0,
        diversity_weight=0.2,
        hazard_positive_weight=8.0,
    )

    dests = [(0, 0), (0, -1), (-1, 0), (0, 1), (1, 0)]

    for it in range(num_dagger_iters):
        distiller.run_dagger_iteration(iteration=it)

        for u in range(updates_per_iter):
            if len(distiller.buffer) == 0:
                break
            batch = distiller.buffer.sample_batch(batch_size=2, burn_in_steps=2)
            for seq in batch.sequences:
                if len(seq.transitions) < 7:
                    continue

            # Forward pass over full sequence to extract ground-truth internal states
            state = model.initial_state(batch_size=1)
            real_states = []
            for step_i in range(6):
                tr = seq.transitions[step_i]
                rgb_t = _rgb_tensor((tr.observation.rgb,), device=device, resolution=getattr(model, "input_resolution", None))
                prev_c_t = torch.tensor([control_to_vector(tr.observation.previous_control)], dtype=torch.float32, device=device)
                dt = torch.tensor([0.016], dtype=torch.float32, device=device)
                out = model(rgb_t, prev_c_t, dt, state)
                real_states.append(out)
                state = out.next_state

            # Focus on transition t=2 with deep 3-step unrolling (predicting t=3, t=4, t=5)
            t0_tr = seq.transitions[2]
            t0_out = real_states[2]

            cur_thoughts = t0_out.next_state.thoughts           # [1, 4, 32]
            cur_belief = t0_out.next_state.belief               # [1, 32]
            cur_memory = t0_out.next_state.working_memory       # [1, 4, 32]
            cur_sensors = model.pixel_encoder(_rgb_tensor((t0_tr.observation.rgb,), device=device, resolution=getattr(model, "input_resolution", None)))
            goal_ctx = t0_out.next_state.goal_context           # [1, 8, 32]
            ret_mem = torch.zeros((1, config.thoughtlets, config.retrieved_entries_per_thoughtlet, config.core_width), dtype=torch.float32, device=device)

            walkable, pellets, player, ghosts = _parse_maze_chase_frame(
                t0_tr.observation.rgb,
                caller="diagnostic.scripted_maze_chase_planner.v1",
            )

            act_vec = control_to_vector(t0_tr.action_target)
            wasd = [act_vec[int(k)] for k in (HidKey.W, HidKey.A, HidKey.S, HidKey.D)]
            expert_act = 1 + wasd.index(max(wasd)) if any(w > 0.5 for w in wasd) else 0

            # --- 1. COUNTERFACTUAL FORESIGHT SUPERVISION ---
            cf_preds = model.counterfactual_foresight_head.forward_all_actions(cur_thoughts)
            branch_losses = []
            for a_i in range(5):
                branch_out = cf_preds[a_i]
                haz_tgt = torch.zeros_like(branch_out.hazard_probability)
                esc_tgt = torch.zeros_like(branch_out.predicted_escape_margin)
                disp_tgt = torch.zeros_like(branch_out.predicted_displacement)

                for h_idx, step_mult in enumerate([1, 2, 3]):
                    if player is not None and ghosts:
                        cand_x = player[0] + dests[a_i][0] * step_mult
                        cand_y = player[1] + dests[a_i][1] * step_mult
                        min_g_dist = min(abs(cand_x - gx) + abs(cand_y - gy) for gx, gy in ghosts)
                        if min_g_dist <= 1:
                            h_val = 1.0
                        elif min_g_dist == 2:
                            h_val = 0.70
                        elif min_g_dist == 3:
                            h_val = 0.25
                        else:
                            h_val = 0.0
                        haz_tgt[0, h_idx, 0] = h_val
                        esc_tgt[0, h_idx, 0] = float(min_g_dist)
                        disp_tgt[0, h_idx, 0] = dests[a_i][0] * step_mult
                        disp_tgt[0, h_idx, 1] = dests[a_i][1] * step_mult
                    else:
                        esc_tgt.fill_(5.0)

                pred_haz = branch_out.hazard_probability.clamp(1e-6, 1.0 - 1e-6)
                weight = torch.where(haz_tgt > 0.2, torch.tensor(8.0, device=device), torch.tensor(1.0, device=device))
                haz_loss = -(weight * haz_tgt * torch.log(pred_haz) + (1.0 - haz_tgt) * torch.log(1.0 - pred_haz)).mean()
                disp_loss = F.smooth_l1_loss(branch_out.predicted_displacement, disp_tgt)
                esc_loss = F.smooth_l1_loss(branch_out.predicted_escape_margin, esc_tgt)
                branch_losses.append(haz_loss + disp_loss + 0.5 * esc_loss)

            cf_loss = torch.stack(branch_losses).mean()

            # --- 2. DEEP 3-STEP AUTOREGRESSIVE MISTAKE UNROLLING ---
            # Helper to create action-time tokens
            def _get_act_time_tokens(action_idx):
                act_v = torch.tensor([control_to_vector(_build_action_control(action_idx))], dtype=torch.float32, device=device)
                action_token = model.control_encoder(act_v).view(1, 1, -1)
                time_input = torch.log1p(torch.tensor([[0.016]], dtype=torch.float32, device=device) * 1000.0)
                time_token = model.time_encoder(time_input).view(1, 1, -1)
                return torch.cat((action_token, time_token), dim=1)

            # Step 1: z_0 -> \hat{z}_1
            act_tokens_1 = _get_act_time_tokens(expert_act)
            unrolled_b1, unrolled_m1, hat_z_1, _ = model.brain_cell(
                belief=cur_belief,
                working_memory=cur_memory,
                thoughts=cur_thoughts,
                sensors=cur_sensors,
                action_time_tokens=act_tokens_1,
                goal_context=goal_ctx,
                retrieved_memory=ret_mem,
                elapsed_seconds=torch.tensor([[0.016]], device=device),
                allow_routing=True,
                allow_workspace_writes=True,
            )
            real_z_1 = real_states[3].next_state.thoughts
            loss_latent_1 = F.mse_loss(hat_z_1, real_z_1)

            # Step 2: \hat{z}_1 -> \hat{z}_2 (unrolled from predicted state!)
            act_next_1 = 0
            if len(seq.transitions) > 3:
                act_vec_1 = control_to_vector(seq.transitions[3].action_target)
                wasd_1 = [act_vec_1[int(k)] for k in (HidKey.W, HidKey.A, HidKey.S, HidKey.D)]
                act_next_1 = 1 + wasd_1.index(max(wasd_1)) if any(w > 0.5 for w in wasd_1) else 0

            act_tokens_2 = _get_act_time_tokens(act_next_1)
            unrolled_b2, unrolled_m2, hat_z_2, _ = model.brain_cell(
                belief=unrolled_b1,
                working_memory=unrolled_m1,
                thoughts=hat_z_1,
                sensors=cur_sensors,
                action_time_tokens=act_tokens_2,
                goal_context=goal_ctx,
                retrieved_memory=ret_mem,
                elapsed_seconds=torch.tensor([[0.016]], device=device),
                allow_routing=True,
                allow_workspace_writes=True,
            )
            real_z_2 = real_states[4].next_state.thoughts
            loss_latent_2 = F.mse_loss(hat_z_2, real_z_2)

            # Step 3: \hat{z}_2 -> \hat{z}_3 (unrolled from 2nd predicted state!)
            act_next_2 = 0
            if len(seq.transitions) > 4:
                act_vec_2 = control_to_vector(seq.transitions[4].action_target)
                wasd_2 = [act_vec_2[int(k)] for k in (HidKey.W, HidKey.A, HidKey.S, HidKey.D)]
                act_next_2 = 1 + wasd_2.index(max(wasd_2)) if any(w > 0.5 for w in wasd_2) else 0
            act_tokens_3 = _get_act_time_tokens(act_next_2)
            unrolled_b3, unrolled_m3, hat_z_3, _ = model.brain_cell(
                belief=unrolled_b2,
                working_memory=unrolled_m2,
                thoughts=hat_z_2,
                sensors=cur_sensors,
                action_time_tokens=act_tokens_3,
                goal_context=goal_ctx,
                retrieved_memory=ret_mem,
                elapsed_seconds=torch.tensor([[0.016]], device=device),
                allow_routing=True,
                allow_workspace_writes=True,
            )
            real_z_3 = real_states[5].next_state.thoughts
            loss_latent_3 = F.mse_loss(hat_z_3, real_z_3)

            # Multi-step latent consistency loss
            deep_latent_loss = loss_latent_1 + loss_latent_2 + loss_latent_3

            # --- 3. COGNITIVE & TOPOLOGICAL LOSS ---
            vec_targets = torch.zeros((1, 2), dtype=torch.float32, device=device)
            if expert_act == 1: vec_targets[0, 1] = -1.0
            elif expert_act == 2: vec_targets[0, 0] = -1.0
            elif expert_act == 3: vec_targets[0, 1] = 1.0
            elif expert_act == 4: vec_targets[0, 0] = 1.0

            is_hazard = (player is not None and ghosts and min(abs(player[0] - gx) + abs(player[1] - gy) for gx, gy in ghosts) <= 2)
            hazard_mask = torch.tensor([is_hazard], dtype=torch.bool, device=device)

            cog_out = cog_loss_fn(
                diagnostics=t0_out.diagnostics,
                hazard_mask=hazard_mask,
                topological_goal_targets=vec_targets,
                topological_exit_targets=torch.tensor([expert_act], dtype=torch.long, device=device),
            )

            total_loss = cog_out.total_loss + 3.0 * cf_loss + 2.0 * deep_latent_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

    return model


def evaluate_policy_at_horizon(model: IreneBrainModel, horizon: int | None, num_eval_seeds: int = 20, seed_start: int = 2001) -> dict[str, Any]:
    model.eval()
    if horizon is not None:
        policy = LatentLookaheadPolicy(model=model, horizon=horizon, hazard_weight=8.0, hazard_prune_threshold=0.80)
    else:
        policy = None

    env = MazeChaseEnv()
    total_pellets = 0.0
    total_catches = 0

    for seed in range(seed_start, seed_start + num_eval_seeds):
        obs = env.reset(seed=seed)
        state = model.initial_state(batch_size=1)
        last_control = torch.zeros((1, 307), dtype=torch.float32)

        if policy is not None:
            policy.reset(seed)

        ep_pellets = 0
        ep_catches = 0

        for step in range(50):
            device = next(model.parameters()).device
            rgb = _rgb_tensor((obs.rgb,), device=device, resolution=getattr(model, "input_resolution", None))
            dt = torch.tensor([0.016], dtype=torch.float32, device=device)

            if policy is not None:
                control = policy.act(obs)
            else:
                with torch.no_grad():
                    out = model(rgb, last_control, dt, state)
                    state = out.next_state
                    last_control = out.action.control
                logits = out.action.button_logits[0].float().cpu()
                continuous_values = out.action.control[0].float().cpu()
                control, _ = decode_closed_loop_control(
                    logits.tolist(),
                    [float(continuous_values[idx]) for idx in CONTINUOUS_TARGET_INDICES],
                    decode_kind=EXCLUSIVE_ARGMAX_WASD_V1,
                )

            outcome = env.step(control)
            obs = outcome.observation
            if "pellet" in outcome.events:
                ep_pellets += 1
            if "caught" in outcome.events:
                ep_catches += 1

        total_pellets += ep_pellets
        total_catches += ep_catches

    return {
        "mean_pellets_per_ep": round(total_pellets / num_eval_seeds, 2),
        "total_catches": total_catches,
        "mean_catches_per_ep": round(total_catches / num_eval_seeds, 2),
    }


def main() -> None:
    print("=" * 80)
    print("PSEUDO-BRAIN: MISTAKE-UNROLLED MULTI-STEP LATENT FORESIGHT BATTERY")
    print("Autoregressive Deep Rollout Training across Horizons H=1, 2, 3 (Seeds 42-46)")
    print("=" * 80)

    battery_results = {}

    for seed in TRAIN_SEEDS:
        print(f"\n--- Training Mistake-Unrolled Seed {seed} ---")
        t0 = time.perf_counter()
        model = train_mistake_unrolled_seed(seed, num_dagger_iters=4, updates_per_iter=24)
        t1 = time.perf_counter()
        print(f"Training completed in {t1-t0:.1f}s")

        print(f"Auditing Large-Scale Per-Horizon Diagnostics (50 Worlds: Seeds 2001-2050)...")
        diag = audit_per_horizon_discrimination(model, num_eval_seeds=50, seed_start=2001)
        hb = diag["horizon_breakdown"]
        print(f"  Choice-Points: {diag['choice_points_evaluated']} | Safe Pick Rate: {diag['choice_point_safe_pick_rate']}%")
        print(f"  H=1 AUROC: {hb['H1']['pairwise_ranking_auroc']}% (Margin: {hb['H1']['discrimination_margin']:+.4f})")
        print(f"  H=2 AUROC: {hb['H2']['pairwise_ranking_auroc']}% (Margin: {hb['H2']['discrimination_margin']:+.4f})")
        print(f"  H=3 AUROC: {hb['H3']['pairwise_ranking_auroc']}% (Margin: {hb['H3']['discrimination_margin']:+.4f})")

        print(f"Evaluating Closed-Loop Policy across Horizon Depths (20 Evaluation Worlds)...")
        direct_res = evaluate_policy_at_horizon(model, horizon=None, num_eval_seeds=20, seed_start=2001)
        h1_res = evaluate_policy_at_horizon(model, horizon=1, num_eval_seeds=20, seed_start=2001)
        h2_res = evaluate_policy_at_horizon(model, horizon=2, num_eval_seeds=20, seed_start=2001)
        h3_res = evaluate_policy_at_horizon(model, horizon=3, num_eval_seeds=20, seed_start=2001)

        print(f"  Direct : {direct_res['total_catches']} catches")
        print(f"  H=1    : {h1_res['total_catches']} catches")
        print(f"  H=2    : {h2_res['total_catches']} catches")
        print(f"  H=3    : {h3_res['total_catches']} catches")

        battery_results[str(seed)] = {
            "diagnostics_50_worlds": diag,
            "direct": direct_res,
            "h1": h1_res,
            "h2": h2_res,
            "h3": h3_res,
        }

    out_file = pathlib.Path("docs/runs/2026-08-19-mistake-unrolled-foresight-battery.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(battery_results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY OF MISTAKE-UNROLLED MULTI-STEP BATTERY (SEEDS 42-46)")
    print("=" * 80)
    print(f"{'Seed':<6} | {'Direct':<8} | {'H=1':<8} | {'H=2':<8} | {'H=3':<8} | {'H1 AUROC':<10} | {'H2 AUROC':<10} | {'H3 AUROC':<10}")
    print("-" * 80)
    for seed, r in battery_results.items():
        d_cat = r['direct']['total_catches']
        h1_cat = r['h1']['total_catches']
        h2_cat = r['h2']['total_catches']
        h3_cat = r['h3']['total_catches']
        h1_a = r['diagnostics_50_worlds']['horizon_breakdown']['H1']['pairwise_ranking_auroc']
        h2_a = r['diagnostics_50_worlds']['horizon_breakdown']['H2']['pairwise_ranking_auroc']
        h3_a = r['diagnostics_50_worlds']['horizon_breakdown']['H3']['pairwise_ranking_auroc']
        print(f"{seed:<6} | {d_cat:<8} | {h1_cat:<8} | {h2_cat:<8} | {h3_cat:<8} | {h1_a:<10.1f} | {h2_a:<10.1f} | {h3_a:<10.1f}")
    print(f"\nSaved results to {out_file}")


if __name__ == "__main__":
    main()
