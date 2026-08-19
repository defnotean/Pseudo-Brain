"""Train and verify Calibrated Counterfactual Foresight across all 5 seeds (42-46).

Implements:
1. Full 5-branch counterfactual supervision (simulating candidate branches a in {None, W, A, S, D})
2. Balanced hazard weighting (w_pos = 12.0)
3. Rigorous Choice-Point Discrimination Audit:
   - Evaluated strictly at true hazard choice-points (states with >= 1 lethal action and >= 1 safe action)
   - Pairwise ranking AUROC (did pred_danger(a_lethal) > pred_danger(a_safe)?)
   - Choice-point safe pick rate (did argmin_a pred_danger(a) pick a truly safe action?)
4. Closed-loop 20-World Evaluation (Direct Controller vs Calibrated Lookahead Planner) across all 5 seeds.
"""

from __future__ import annotations

import copy
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
from irene_brain.evaluation.closed_loop_play import (
    EXCLUSIVE_ARGMAX_WASD_V1,
    decode_closed_loop_control,
)
from irene_brain.evaluation.diagnostic_policies import ScriptedMazeChasePlannerPolicy
from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
from irene_brain.model.lookahead_planner import LatentLookaheadPlanner
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.training.batches import (
    CONTINUOUS_TARGET_INDICES,
    control_to_vector,
)
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

TRAIN_SEEDS = (42, 43, 44, 45, 46)


def _build_action_control(action_idx: int) -> GenericControl:
    if action_idx == 1:
        return GenericControl(keys_down=(HidKey.W,))
    elif action_idx == 2:
        return GenericControl(keys_down=(HidKey.A,))
    elif action_idx == 3:
        return GenericControl(keys_down=(HidKey.S,))
    elif action_idx == 4:
        return GenericControl(keys_down=(HidKey.D,))
    return GenericControl.neutral()


from irene_brain.evaluation.diagnostic_policies import _parse_maze_chase_frame


def train_calibrated_seed(train_seed: int, num_dagger_iters: int = 3) -> IreneBrainModel:
    """Train IreneBrainModel with geometrically grounded 5-branch counterfactual hazard supervision."""
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

    training_config = TrainingConfig(
        schema_version=1,
        run=RunConfig(
            name=f"calibrated-seed{train_seed}",
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
    system = TorchTrainingSystem(objective, training_config)
    dagger_config = DAggerConfig(
        iterations=num_dagger_iters,
        episodes_per_iteration=3,
        max_ticks_per_episode=80,
        initial_beta=0.8,
        beta_decay=0.85,
        sequence_length=8,
        burn_in_steps=2,
        batch_size=2,
        updates_per_iteration=16,
        ghost_count=2,
        ghost_period=2,
        extra_loops=8,
    )
    distiller = DAggerDistiller(config=dagger_config, student_model=model, training_system=system)

    # Pre-populate buffer with both mixed curriculum and focused hazard evasion
    curriculum_cfg_mixed = CurriculumDatasetConfig(sequence_length=8, sequence_count=30, scenario="mixed")
    curriculum_dataset_mixed = CurriculumDataset(curriculum_cfg_mixed)
    for i in range(min(15, len(curriculum_dataset_mixed))):
        distiller.buffer.add_sequence(curriculum_dataset_mixed[i])

    curriculum_cfg_haz = CurriculumDatasetConfig(sequence_length=8, sequence_count=30, scenario="hazard_evasion")
    curriculum_dataset_haz = CurriculumDataset(curriculum_cfg_haz)
    for i in range(min(15, len(curriculum_dataset_haz))):
        distiller.buffer.add_sequence(curriculum_dataset_haz[i])

    optimizer = system.optimizer if hasattr(system, "optimizer") else None

    # Loss with positive hazard reweighting
    cog_loss_fn = CognitiveAuxiliaryLoss(
        future_weight=1.0,
        gate_surprise_weight=1.0,
        halting_weight=0.5,
        counterfactual_weight=2.0,
        topological_goal_weight=1.0,
        diversity_weight=0.2,
        hazard_positive_weight=8.0,
    )

    for it in range(num_dagger_iters):
        distiller.run_dagger_iteration(iteration=it)
        if optimizer is not None and len(distiller.buffer) > 0:
            model.train()
            for _ in range(16):
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
                    expert_act = 1 + wasd.index(max(wasd)) if any(w > 0.5 for w in wasd) else 0

                    # Parse spatial frame for grounded counterfactual supervision
                    walkable, pellets, player, ghosts = _parse_maze_chase_frame(
                        t0_obs.observation.rgb,
                        caller="diagnostic.scripted_maze_chase_planner.v1",
                    )

                    # Evaluate all 5 candidate branches from current state
                    cf_preds = model.counterfactual_foresight_head.forward_all_actions(out.next_state.thoughts)

                    branch_losses = []
                    dests = [(0, 0), (0, -1), (-1, 0), (0, 1), (1, 0)]

                    for a_i in range(5):
                        branch_out = cf_preds[a_i]
                        # Target displacement for action a_i
                        disp_tgt = torch.zeros((1, 3, 2), dtype=torch.float32, device=device)
                        disp_tgt[:, :, 0] = dests[a_i][0]
                        disp_tgt[:, :, 1] = dests[a_i][1]

                        # Ground-truth geometric hazard calculation
                        if player is not None and ghosts:
                            px, py = player
                            cand_x = px + dests[a_i][0]
                            cand_y = py + dests[a_i][1]
                            min_g_dist = min(abs(cand_x - gx) + abs(cand_y - gy) for gx, gy in ghosts)
                            if min_g_dist <= 1:
                                haz_val = 1.0
                            elif min_g_dist == 2:
                                haz_val = 0.70
                            elif min_g_dist == 3:
                                haz_val = 0.30
                            else:
                                haz_val = 0.0
                            esc_val = min_g_dist
                        else:
                            haz_val = 0.0
                            esc_val = 5.0

                        haz_tgt = torch.tensor([[[haz_val], [haz_val * 0.8], [haz_val * 0.6]]], dtype=torch.float32, device=device)
                        esc_tgt = torch.tensor([[[esc_val], [esc_val], [esc_val]]], dtype=torch.float32, device=device)

                        # Weighted BCE on hazard
                        target_haz = haz_tgt.clamp(0.0, 1.0)
                        pred_haz = branch_out.hazard_probability.clamp(1e-6, 1.0 - 1e-6)
                        weight = torch.where(target_haz > 0.2, torch.tensor(8.0, device=device), torch.tensor(1.0, device=device))
                        haz_loss = -(weight * target_haz * torch.log(pred_haz) + (1.0 - target_haz) * torch.log(1.0 - pred_haz)).mean()

                        disp_loss = F.smooth_l1_loss(branch_out.predicted_displacement, disp_tgt)
                        esc_loss = F.smooth_l1_loss(branch_out.predicted_escape_margin, esc_tgt)
                        branch_losses.append(haz_loss + disp_loss + 0.5 * esc_loss)

                    total_cf_loss = torch.stack(branch_losses).mean()

                    # Goal vector target
                    vec_targets = torch.zeros((1, 2), dtype=torch.float32, device=device)
                    if expert_act == 1: vec_targets[0, 1] = -1.0
                    elif expert_act == 2: vec_targets[0, 0] = -1.0
                    elif expert_act == 3: vec_targets[0, 1] = 1.0
                    elif expert_act == 4: vec_targets[0, 0] = 1.0

                    is_hazard = (player is not None and ghosts and min(abs(player[0] - gx) + abs(player[1] - gy) for gx, gy in ghosts) <= 2)
                    hazard_mask = torch.tensor([is_hazard], dtype=torch.bool, device=device)

                    cog_out = cog_loss_fn(
                        diagnostics=out.diagnostics,
                        hazard_mask=hazard_mask,
                        topological_goal_targets=vec_targets,
                        topological_exit_targets=torch.tensor([expert_act], dtype=torch.long, device=device),
                    )

                    loss = cog_out.total_loss + 2.0 * total_cf_loss
                    if loss.requires_grad:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

    return model


def audit_choice_point_discrimination(model: IreneBrainModel, num_eval_seeds: int = 20, seed_start: int = 2001) -> dict[str, Any]:
    """Rigorously evaluate counterfactual hazard discrimination specifically at choice points."""
    model.eval()
    head = model.counterfactual_foresight_head

    total_choice_points = 0
    choice_point_safe_picks = 0
    pairwise_pairs = 0
    pairwise_correct = 0

    choice_point_d_lethal_list = []
    choice_point_d_safe_list = []

    brier_sum = 0.0
    total_preds = 0

    env = MazeChaseEnv()

    for seed in range(seed_start, seed_start + num_eval_seeds):
        obs = env.reset(seed=seed)
        state = model.initial_state(batch_size=1)
        last_control = torch.zeros((1, 307), dtype=torch.float32)

        for step in range(25):
            device = next(model.parameters()).device
            rgb = _rgb_tensor((obs.rgb,), device=device, resolution=getattr(model, "input_resolution", None))
            dt = torch.tensor([0.016], dtype=torch.float32, device=device)

            with torch.no_grad():
                out = model(rgb, last_control, dt, state)
                state = out.next_state
                last_control = out.action.control
                cf_preds = head.forward_all_actions(state.thoughts)

            action_dangers_pred = []
            action_dangers_true = []

            for a_idx in range(5):
                pred_haz_prob = float(cf_preds[a_idx].hazard_probability[0, 0, 0].item())
                action_dangers_pred.append(pred_haz_prob)

                sim_env = copy.deepcopy(env)
                act_ctrl = _build_action_control(a_idx)
                is_coll = 0.0
                for _ in range(5):
                    outcome = sim_env.step(act_ctrl)
                    if sim_env._times_caught > env._times_caught:
                        is_coll = 1.0
                        break
                action_dangers_true.append(is_coll)

                brier_sum += (pred_haz_prob - is_coll) ** 2
                total_preds += 1

            # Check if this state is a TRUE hazard choice-point
            has_danger = any(d > 0.5 for d in action_dangers_true)
            has_safe = any(d < 0.5 for d in action_dangers_true)

            if has_danger and has_safe:
                total_choice_points += 1
                best_pred_a = int(torch.tensor(action_dangers_pred).argmin().item())
                if action_dangers_true[best_pred_a] < 0.5:
                    choice_point_safe_picks += 1

                for a_d in range(5):
                    if action_dangers_true[a_d] > 0.5:
                        choice_point_d_lethal_list.append(action_dangers_pred[a_d])
                        for a_s in range(5):
                            if action_dangers_true[a_s] < 0.5:
                                pairwise_pairs += 1
                                if action_dangers_pred[a_d] > action_dangers_pred[a_s]:
                                    pairwise_correct += 1
                    else:
                        choice_point_d_safe_list.append(action_dangers_pred[a_d])

            logits = out.action.button_logits[0].float().cpu()
            continuous_values = out.action.control[0].float().cpu()
            control, _ = decode_closed_loop_control(
                logits.tolist(),
                [float(continuous_values[idx]) for idx in CONTINUOUS_TARGET_INDICES],
                decode_kind=EXCLUSIVE_ARGMAX_WASD_V1,
            )
            outcome = env.step(control)
            obs = outcome.observation

    mean_d_lethal = sum(choice_point_d_lethal_list) / len(choice_point_d_lethal_list) if choice_point_d_lethal_list else 0.0
    mean_d_safe = sum(choice_point_d_safe_list) / len(choice_point_d_safe_list) if choice_point_d_safe_list else 0.0
    choice_point_margin = mean_d_lethal - mean_d_safe

    choice_point_safe_acc = (choice_point_safe_picks / total_choice_points * 100.0) if total_choice_points > 0 else 0.0
    pairwise_auroc = (pairwise_correct / pairwise_pairs * 100.0) if pairwise_pairs > 0 else 0.0
    brier = (brier_sum / total_preds) if total_preds > 0 else 0.0

    return {
        "brier_score": round(brier, 4),
        "hazard_choice_points_evaluated": total_choice_points,
        "mean_danger_lethal_actions": round(mean_d_lethal, 4),
        "mean_danger_safe_actions": round(mean_d_safe, 4),
        "choice_point_discrimination_margin": round(choice_point_margin, 4),
        "choice_point_safe_pick_rate": round(choice_point_safe_acc, 1),
        "pairwise_ranking_auroc": round(pairwise_auroc, 1),
    }


def evaluate_closed_loop_policy(model: IreneBrainModel, use_planner: bool, num_eval_seeds: int = 20, seed_start: int = 2001) -> dict[str, Any]:
    """Run closed-loop evaluation over held-out worlds."""
    model.eval()
    if use_planner:
        policy = LatentLookaheadPolicy(
            model=model,
            horizon=3,
            hazard_weight=8.0,
            hazard_prune_threshold=0.80,
        )
    else:
        policy = None

    env = MazeChaseEnv()
    total_pellets = 0.0
    total_catches = 0
    total_survival = 0

    for seed in range(seed_start, seed_start + num_eval_seeds):
        obs = env.reset(seed=seed)
        state = model.initial_state(batch_size=1)
        last_control = torch.zeros((1, 307), dtype=torch.float32)

        if use_planner:
            policy.reset(seed)

        ep_pellets = 0
        ep_catches = 0
        ep_steps = 0

        for step in range(50):
            ep_steps += 1
            device = next(model.parameters()).device
            rgb = _rgb_tensor((obs.rgb,), device=device, resolution=getattr(model, "input_resolution", None))
            dt = torch.tensor([0.016], dtype=torch.float32, device=device)

            if use_planner:
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
        total_survival += ep_steps

    return {
        "mean_pellets_per_ep": round(total_pellets / num_eval_seeds, 2),
        "total_catches": total_catches,
        "mean_catches_per_ep": round(total_catches / num_eval_seeds, 2),
        "mean_survival_steps": round(total_survival / num_eval_seeds, 2),
    }


def main() -> None:
    print("=" * 80)
    print("PSEUDO-BRAIN: CALIBRATED FORESIGHT MULTI-SEED BATTERY (SEEDS 42-46)")
    print("Standard: True Choice-Point Hazard Separation & Closed-Loop Replication")
    print("=" * 80)

    results = {}

    for seed in TRAIN_SEEDS:
        print(f"\n--- Training Calibrated Seed {seed} ---")
        t0 = time.perf_counter()
        model = train_calibrated_seed(seed, num_dagger_iters=3)
        t1 = time.perf_counter()
        print(f"Training completed in {t1-t0:.1f}s")

        print(f"Auditing Choice-Point Discrimination for Seed {seed}...")
        diag = audit_choice_point_discrimination(model, num_eval_seeds=20, seed_start=2001)
        print(f"  Choice-Points: {diag['hazard_choice_points_evaluated']} | Separation Margin: {diag['choice_point_discrimination_margin']:+.4f}")
        print(f"  Choice-Point Safe Pick Rate: {diag['choice_point_safe_pick_rate']}% | Pairwise AUROC: {diag['pairwise_ranking_auroc']}% | Brier: {diag['brier_score']}")

        print(f"Evaluating Direct Controller for Seed {seed}...")
        direct_eval = evaluate_closed_loop_policy(model, use_planner=False, num_eval_seeds=20, seed_start=2001)
        print(f"  Direct: Pellets={direct_eval['mean_pellets_per_ep']}, Catches={direct_eval['total_catches']} ({direct_eval['mean_catches_per_ep']}/ep)")

        print(f"Evaluating Calibrated Grounded Planner for Seed {seed}...")
        planner_eval = evaluate_closed_loop_policy(model, use_planner=True, num_eval_seeds=20, seed_start=2001)
        print(f"  Planner: Pellets={planner_eval['mean_pellets_per_ep']}, Catches={planner_eval['total_catches']} ({planner_eval['mean_catches_per_ep']}/ep)")

        results[str(seed)] = {
            "diagnostics": diag,
            "direct_eval": direct_eval,
            "planner_eval": planner_eval,
        }

    out_file = pathlib.Path("docs/runs/2026-08-19-calibrated-foresight-battery.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY OF 5-SEED CALIBRATED BATTERY")
    print("=" * 80)
    for seed, r in results.items():
        d_cat = r['direct_eval']['total_catches']
        p_cat = r['planner_eval']['total_catches']
        red_pct = ((d_cat - p_cat) / d_cat * 100.0) if d_cat > 0 else 0.0
        print(f"Seed {seed}: Direct Catches = {d_cat:3d} -> Planner Catches = {p_cat:3d} ({red_pct:+.1f}% reduction) | Margin = {r['diagnostics']['choice_point_discrimination_margin']:+.4f} | AUROC = {r['diagnostics']['pairwise_ranking_auroc']:.1f}%")
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
