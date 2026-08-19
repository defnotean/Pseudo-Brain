"""Audit Counterfactual Foresight Head Convergence across Seeds 42-46.

Quantifies:
1. Hazard prediction precision, recall, accuracy, F1
2. Brier calibration score (mean squared error of predicted probability)
3. Action-ranking accuracy (safest predicted action vs ground-truth safest action)
4. Escape-margin MAE across horizons (1, 3, 5 ticks)
5. Comparison matrix across Seeds 42, 43, 44, 45, 46
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
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.training.batches import (
    BUTTON_TARGET_INDICES,
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


def train_seed_model(train_seed: int) -> IreneBrainModel:
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
            name=f"audit-seed{train_seed}",
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
        iterations=2,
        episodes_per_iteration=2,
        max_ticks_per_episode=60,
        initial_beta=0.8,
        beta_decay=0.85,
        sequence_length=8,
        burn_in_steps=2,
        batch_size=2,
        updates_per_iteration=12,
        ghost_count=2,
        ghost_period=2,
        extra_loops=8,
    )
    distiller = DAggerDistiller(config=dagger_config, student_model=model, training_system=system)
    curriculum_cfg = CurriculumDatasetConfig(sequence_length=8, sequence_count=40, scenario="mixed")
    curriculum_dataset = CurriculumDataset(curriculum_cfg)
    for i in range(min(20, len(curriculum_dataset))):
        distiller.buffer.add_sequence(curriculum_dataset[i])

    cog_loss_fn = CognitiveAuxiliaryLoss(
        future_weight=1.0,
        gate_surprise_weight=1.0,
        halting_weight=0.5,
        counterfactual_weight=1.0,
        topological_goal_weight=1.0,
        diversity_weight=0.1,
    )
    optimizer = system.optimizer if hasattr(system, "optimizer") else None

    for it in range(2):
        distiller.run_dagger_iteration(iteration=it)
        if optimizer is not None and len(distiller.buffer) > 0:
            model.train()
            for _ in range(12):
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
                    act_idx = 1 + wasd.index(max(wasd)) if any(w > 0.5 for w in wasd) else 0

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
                        topological_goal_targets=vec_targets,
                        topological_exit_targets=executed_actions,
                    )
                    if cog_out.total_loss.requires_grad:
                        optimizer.zero_grad()
                        cog_out.total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
    return model


def _build_action_control(act_idx: int) -> GenericControl:
    if act_idx == 1:
        return GenericControl(keys_down=(HidKey.W,))
    elif act_idx == 2:
        return GenericControl(keys_down=(HidKey.A,))
    elif act_idx == 3:
        return GenericControl(keys_down=(HidKey.S,))
    elif act_idx == 4:
        return GenericControl(keys_down=(HidKey.D,))
    return GenericControl()


def evaluate_foresight_on_validation_scenarios(
    model: IreneBrainModel,
    num_eval_seeds: int = 20,
    seed_start: int = 2001,
) -> dict[str, Any]:
    model.eval()
    head = model.counterfactual_foresight_head

    total_predictions = 0
    brier_sum = 0.0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    ranking_correct = 0
    ranking_total = 0
    escape_mae_sum = 0.0

    horizon_brier = [0.0, 0.0, 0.0]
    horizon_acc = [0.0, 0.0, 0.0]
    horizon_total = 0

    env = MazeChaseEnv()

    # Positive vs Negative Probability Separation (Discrimination Margin)
    pos_probs = []
    neg_probs = []

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
            action_min_dists = []

            for a_idx in range(5):
                pred_haz_prob = float(cf_preds[a_idx].hazard_probability[0, 0, 0].item())
                pred_esc_marg = float(cf_preds[a_idx].predicted_escape_margin[0, 0, 0].item())
                action_dangers_pred.append(pred_haz_prob)

                sim_env = copy.deepcopy(env)
                act_ctrl = _build_action_control(a_idx)
                true_hazards = [0.0, 0.0, 0.0]
                min_ghost_dist = 999.0

                for k in range(5):
                    outcome = sim_env.step(act_ctrl)
                    px, py = sim_env._player_x, sim_env._player_y
                    for gx, gy in sim_env._ghosts:
                        dist = abs(px - gx) + abs(py - gy)
                        if dist < min_ghost_dist:
                            min_ghost_dist = dist

                    is_coll = 1.0 if sim_env._times_caught > env._times_caught else 0.0
                    if k == 0:
                        true_hazards[0] = is_coll
                    elif k == 2:
                        true_hazards[1] = is_coll
                    elif k == 4:
                        true_hazards[2] = is_coll

                action_dangers_true.append(true_hazards[0])
                action_min_dists.append(min_ghost_dist)

                y_true = true_hazards[0]
                y_pred = pred_haz_prob
                if y_true > 0.5:
                    pos_probs.append(y_pred)
                else:
                    neg_probs.append(y_pred)

                # Multi-horizon tracking
                if cf_preds is not None and a_idx in cf_preds:
                    haz_t = cf_preds[a_idx].hazard_probability[0].squeeze(-1) # [3]
                    for h_i in range(3):
                        h_p = float(haz_t[h_i].item())
                        h_y = float(true_hazards[h_i])
                        horizon_brier[h_i] += (h_p - h_y) ** 2
                        horizon_acc[h_i] += 1.0 if ((h_p >= 0.5) == (h_y > 0.5)) else 0.0
                    horizon_total += 3

                brier_sum += (y_pred - y_true) ** 2
                total_predictions += 1

                if y_pred >= 0.50:
                    if y_true > 0.5:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if y_true > 0.5:
                        fn += 1
                    else:
                        tn += 1

                escape_mae_sum += abs(pred_esc_marg - min_ghost_dist)

            best_pred_a = int(torch.tensor(action_dangers_pred).argmin().item())
            if action_dangers_true[best_pred_a] < 0.5:
                ranking_correct += 1
            ranking_total += 1

            logits = out.action.button_logits[0].float().cpu()
            continuous_values = out.action.control[0].float().cpu()
            control, _ = decode_closed_loop_control(
                logits.tolist(),
                [float(continuous_values[idx]) for idx in CONTINUOUS_TARGET_INDICES],
                decode_kind=EXCLUSIVE_ARGMAX_WASD_V1,
            )
            outcome = env.step(control)
            obs = outcome.observation

    # Separation Margin & Calibration
    mean_pos_prob = sum(pos_probs) / len(pos_probs) if pos_probs else 0.0
    mean_neg_prob = sum(neg_probs) / len(neg_probs) if neg_probs else 0.0
    discrimination_margin = mean_pos_prob - mean_neg_prob

    # Optimal Threshold Sweep
    best_opt_thresh = 0.50
    best_opt_f1 = 0.0
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        t_tp = sum(1 for p in pos_probs if p >= thresh)
        t_fp = sum(1 for p in neg_probs if p >= thresh)
        t_fn = sum(1 for p in pos_probs if p < thresh)
        t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0.0
        t_rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0.0
        t_f1 = (2 * t_prec * t_rec / (t_prec + t_rec)) if (t_prec + t_rec) > 0 else 0.0
        if t_f1 > best_opt_f1:
            best_opt_f1 = t_f1
            best_opt_thresh = thresh

    # Compute Final Summary Metrics
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / total_predictions if total_predictions > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    brier_score = brier_sum / total_predictions if total_predictions > 0 else 0.0
    ranking_acc = ranking_correct / ranking_total if ranking_total > 0 else 0.0
    escape_mae = escape_mae_sum / total_predictions if total_predictions > 0 else 0.0

    h_briers = [round(hb / (horizon_total / 3), 4) for hb in horizon_brier]
    h_accs = [round(ha / (horizon_total / 3) * 100, 1) for ha in horizon_acc]

    return {
        "brier_score": round(brier_score, 4),
        "mean_pos_hazard_prob": round(mean_pos_prob, 4),
        "mean_neg_hazard_prob": round(mean_neg_prob, 4),
        "discrimination_margin": round(discrimination_margin, 4),
        "optimal_threshold": best_opt_thresh,
        "optimal_f1": round(best_opt_f1, 4),
        "recall": round(recall * 100, 1),
        "precision": round(precision * 100, 1),
        "accuracy": round(accuracy * 100, 1),
        "f1": round(f1, 4),
        "safest_action_accuracy": round(ranking_acc * 100, 1),
        "escape_margin_mae": round(escape_mae, 2),
        "hazard_counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total_predictions},
    }


def main() -> None:
    print("=" * 80)
    print("COUNTERFACTUAL FORESIGHT HEAD AUDIT ACROSS SEEDS 42-46")
    print("Evaluation Protocol: 20 Held-Out Worlds x 25 Steps x 5 Candidate Actions (2500 Predictions / Seed)")
    print("=" * 80)

    results = {}
    for seed in TRAIN_SEEDS:
        print(f"\nEvaluating Seed {seed}...")
        model = train_seed_model(seed)
        diag = evaluate_foresight_on_validation_scenarios(model, num_eval_seeds=20, seed_start=2001)
        results[str(seed)] = diag

        print(f"  Brier Score (Calibration):     {diag['brier_score']:.4f}")
        print(f"  Mean Pos Hazard Prob:           {diag['mean_pos_hazard_prob']:.4f}")
        print(f"  Mean Neg Hazard Prob:           {diag['mean_neg_hazard_prob']:.4f}")
        print(f"  Discrimination Margin:          {diag['discrimination_margin']:+.4f}")
        print(f"  Optimal Threshold / F1:         tau={diag['optimal_threshold']} (F1={diag['optimal_f1']:.4f})")
        print(f"  Safest Action Survival Rate:    {diag['safest_action_accuracy']:.1f}%")
        print(f"  Escape Margin MAE:              {diag['escape_margin_mae']:.2f}")

    out_file = pathlib.Path("docs/runs/2026-08-19-foresight-head-audit.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
