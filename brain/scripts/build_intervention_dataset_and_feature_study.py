"""Intervention Value Dataset Builder & Cognitive Signal Analysis.

Evaluates candidate intervention points across held-out worlds using exact
simulator snapshot/restore forks to determine which internal signals reliably
predict when internal foresight improves vs degrades decision outcomes.

Metrics audited:
- Helpful, Harmful, Neutral counts & rates
- Net Intervention Value = Helpful - Harmful
- Intervention Precision = Helpful / (Helpful + Harmful)  [N/A when 0 interventions]
- Coverage = Interventions / Eligible Decision Points
- Feature Correlations with Helpful vs Harmful outcomes
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
SCRIPTS = pathlib.Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.closed_loop_play import (
    EXCLUSIVE_ARGMAX_WASD_V1,
    decode_closed_loop_control,
)
from irene_brain.evaluation.diagnostic_policies import _parse_maze_chase_frame
from irene_brain.model.intent import DirectionalAction
from irene_brain.model.lookahead_planner import (
    LatentLookaheadPlanner,
    generate_directional_candidate_sequences,
)
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.training.batches import CONTINUOUS_TARGET_INDICES
from irene_brain.training.objective import _rgb_tensor
from irene_brain.types import GenericControl, HidKey


TRAIN_SEEDS = [42, 43, 44, 45, 46]


def control_to_act_index(ctrl: GenericControl) -> int:
    if HidKey.W in ctrl.keys_down: return 1
    elif HidKey.A in ctrl.keys_down: return 2
    elif HidKey.S in ctrl.keys_down: return 3
    elif HidKey.D in ctrl.keys_down: return 4
    return 0


def act_index_to_control(act_idx: int) -> GenericControl:
    key_map = {1: (HidKey.W,), 2: (HidKey.A,), 3: (HidKey.S,), 4: (HidKey.D,)}
    return GenericControl(keys_down=key_map.get(act_idx, ()))


def build_intervention_dataset(
    eval_seeds: list[int] = list(range(2001, 2051)),
    lookahead_eval_steps: int = 3,
) -> dict[str, Any]:
    print("=" * 80)
    print("PSEUDO-BRAIN INTERVENTION VALUE DATASET & COGNITIVE FEATURE STUDY")
    print(f"Sampling decision points across Seeds 42-46 (50 Worlds, Forward Fork Depth={lookahead_eval_steps})")
    print("=" * 80)

    dataset_records = []
    summary_by_seed = {}

    for seed in TRAIN_SEEDS:
        ckpt_path = pathlib.Path(f"artifacts/checkpoints/mistake_unrolled_seed_{seed}.pt")
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = IreneBrainModel(ckpt["config"], enable_adaptive_cognition=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        cf_head = getattr(model, "counterfactual_foresight_head", None)
        topo_head = getattr(model, "topological_goal_head", None)
        planner_h2 = LatentLookaheadPlanner(model=model, horizon=2, gamma=0.95, hazard_weight=8.0)

        total_steps = 0
        eligible_decision_points = 0
        helpful_count = 0
        neutral_count = 0
        harmful_count = 0

        env = MazeChaseEnv()

        for w_seed in eval_seeds:
            obs = env.reset(seed=w_seed)
            state = model.initial_state(1)
            last_ctrl = torch.zeros((1, 307), dtype=torch.float32)

            for step in range(50):
                total_steps += 1
                rgb = _rgb_tensor((obs.rgb,), device="cpu", resolution=getattr(model, "input_resolution", None))
                dt = torch.tensor([0.016], dtype=torch.float32)

                with torch.no_grad():
                    out = model(rgb, last_ctrl, dt, state)
                    state = out.next_state
                    last_ctrl = out.action.control

                logits = out.action.button_logits[0].float().cpu()
                continuous_values = out.action.control[0].float().cpu()
                d_ctrl, _ = decode_closed_loop_control(
                    logits.tolist(),
                    [float(continuous_values[idx]) for idx in CONTINUOUS_TARGET_INDICES],
                    decode_kind=EXCLUSIVE_ARGMAX_WASD_V1,
                )
                d_act = control_to_act_index(d_ctrl)

                # Compute Direct Policy Entropy and Margin
                probs = F.softmax(logits, dim=-1).numpy()
                action_entropy = -float(np.sum(probs * np.log(probs + 1e-8)))
                sorted_probs = np.sort(probs)
                policy_margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) > 1 else 1.0

                # Extract Thoughtlet Dispersion (Mean Pairwise Distance)
                thoughts = out.next_state.thoughts[0].flatten(start_dim=1)  # [K, R*d]
                norm_thoughts = F.normalize(thoughts, dim=-1)
                sim_matrix = torch.mm(norm_thoughts, norm_thoughts.t())
                K = thoughts.shape[0]
                triu_indices = torch.triu_indices(K, K, offset=1)
                pairwise_sims = sim_matrix[triu_indices[0], triu_indices[1]].cpu().numpy()
                thought_dispersion = float(1.0 - np.mean(pairwise_sims))

                # Spatial parsing for ground truth reference
                walkable, pellets, player, ghosts = _parse_maze_chase_frame(
                    obs.rgb, caller="diagnostic.scripted_maze_chase_planner.v1"
                )
                min_g_dist = 999
                if player is not None and ghosts:
                    min_g_dist = min(abs(player[0] - gx) + abs(player[1] - gy) for gx, gy in ghosts)

                # Candidate Foresight Evaluation
                if cf_head is not None:
                    cf_preds = cf_head.forward_all_actions(out.next_state.thoughts)
                    hazards_h1 = {a_i: float(cf_preds[a_i].hazard_probability[0, 0, 0].item()) for a_i in range(5)}
                    hazards_h2 = {a_i: float(cf_preds[a_i].hazard_probability[0, min(1, cf_preds[a_i].hazard_probability.shape[1] - 1), 0].item()) for a_i in range(5)}
                    escape_margins = {a_i: float(cf_preds[a_i].predicted_escape_margin[0, 0, 0].item()) for a_i in range(5)}

                    min_haz_h1 = min(hazards_h1.values())
                    direct_haz = hazards_h1.get(d_act, 0.0)
                    contrast_margin = direct_haz - min_haz_h1
                    best_safe_act = min(hazards_h1, key=hazards_h1.get)
                    hazard_reduction = direct_haz - hazards_h1.get(best_safe_act, 0.0)

                    # Check if this point is an eligible decision point (where planner and direct disagree on best move)
                    if best_safe_act != d_act or contrast_margin > 0.001:
                        eligible_decision_points += 1

                        # Exact Simulator Counterfactual Fork
                        snap = env.snapshot()

                        # Fork A: Direct Action Forward
                        env.restore(snap)
                        direct_caught = False
                        direct_pellets = 0
                        out_d = env.step(act_index_to_control(d_act))
                        if "caught" in out_d.events: direct_caught = True
                        if "pellet" in out_d.events: direct_pellets += 1
                        for _ in range(lookahead_eval_steps - 1):
                            if out_d.terminated or out_d.truncated: break
                            out_d = env.step(GenericControl())
                            if "caught" in out_d.events: direct_caught = True
                            if "pellet" in out_d.events: direct_pellets += 1

                        # Fork B: Planner Action Forward
                        env.restore(snap)
                        planner_caught = False
                        planner_pellets = 0
                        out_p = env.step(act_index_to_control(best_safe_act))
                        if "caught" in out_p.events: planner_caught = True
                        if "pellet" in out_p.events: planner_pellets += 1
                        for _ in range(lookahead_eval_steps - 1):
                            if out_p.terminated or out_p.truncated: break
                            out_p = env.step(GenericControl())
                            if "caught" in out_p.events: planner_caught = True
                            if "pellet" in out_p.events: planner_pellets += 1

                        env.restore(snap)

                        if direct_caught and not planner_caught:
                            label = "HELPFUL"
                            label_val = 1
                            helpful_count += 1
                        elif planner_caught and not direct_caught:
                            label = "HARMFUL"
                            label_val = -1
                            harmful_count += 1
                        else:
                            label = "NEUTRAL"
                            label_val = 0
                            neutral_count += 1

                        record = {
                            "seed": seed,
                            "world_seed": w_seed,
                            "step": step,
                            "direct_action": d_act,
                            "planner_action": best_safe_act,
                            "label": label,
                            "label_val": label_val,
                            # Internal Features
                            "contrast_margin": round(float(contrast_margin), 5),
                            "direct_hazard": round(float(direct_haz), 5),
                            "planner_hazard": round(float(hazards_h1[best_safe_act]), 5),
                            "hazard_reduction": round(float(hazard_reduction), 5),
                            "h1_vs_h2_diff": round(float(abs(hazards_h1[best_safe_act] - hazards_h2[best_safe_act])), 5),
                            "predicted_escape_margin": round(float(escape_margins.get(best_safe_act, 0.0)), 4),
                            "thought_dispersion": round(float(thought_dispersion), 4),
                            "policy_entropy": round(float(action_entropy), 4),
                            "policy_margin": round(float(policy_margin), 4),
                            "ghost_distance": min_g_dist,
                        }
                        dataset_records.append(record)

                outcome = env.step(act_index_to_control(d_act))
                obs = outcome.observation
                if outcome.terminated or outcome.truncated:
                    break

        total_interventions = helpful_count + harmful_count + neutral_count
        precision = (
            round((helpful_count / (helpful_count + harmful_count)) * 100.0, 2)
            if (helpful_count + harmful_count) > 0
            else None
        )
        coverage = (
            round((total_interventions / eligible_decision_points) * 100.0, 2)
            if eligible_decision_points > 0
            else 0.0
        )

        summary_by_seed[seed] = {
            "total_steps": total_steps,
            "eligible_decision_points": eligible_decision_points,
            "total_interventions": total_interventions,
            "helpful": helpful_count,
            "neutral": neutral_count,
            "harmful": harmful_count,
            "helpful_rate_pct": round((helpful_count / total_interventions) * 100.0, 2) if total_interventions > 0 else 0.0,
            "harmful_rate_pct": round((harmful_count / total_interventions) * 100.0, 2) if total_interventions > 0 else 0.0,
            "net_intervention_value": helpful_count - harmful_count,
            "intervention_precision_pct": precision,
            "coverage_pct": coverage,
        }

        print(f"\n--- SEED {seed} INTERVENTION SUMMARY ---")
        print(f"Eligible Points: {eligible_decision_points} | Interventions: {total_interventions} (Coverage: {coverage}%)")
        print(f"Breakdown      : HELPFUL={helpful_count} | NEUTRAL={neutral_count} | HARMFUL={harmful_count}")
        print(f"Precision      : {precision if precision is not None else 'N/A (0 active decisions)'}% | Net Value = {helpful_count - harmful_count:+d}")

    # Compute Feature Correlations with Positive Intervention Value
    feature_keys = [
        "contrast_margin",
        "direct_hazard",
        "hazard_reduction",
        "h1_vs_h2_diff",
        "predicted_escape_margin",
        "thought_dispersion",
        "policy_entropy",
        "policy_margin",
        "ghost_distance",
    ]

    labels = np.array([r["label_val"] for r in dataset_records])
    feature_correlations = {}
    for fk in feature_keys:
        fvals = np.array([r[fk] for r in dataset_records])
        if np.std(fvals) > 1e-8 and np.std(labels) > 1e-8:
            corr = float(np.corrcoef(fvals, labels)[0, 1])
        else:
            corr = 0.0
        feature_correlations[fk] = round(corr, 4)

    print("\n" + "=" * 80)
    print("STATISTICAL FEATURE CORRELATIONS WITH HELPFUL INTERVENTIONS (+1=Helpful, -1=Harmful):")
    for fk, corr in sorted(feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {fk:<28}: Pearson r = {corr:+.4f}")
    print("=" * 80)

    output = {
        "dataset_size": len(dataset_records),
        "seed_summaries": summary_by_seed,
        "feature_correlations": feature_correlations,
        "sample_records": dataset_records[:15],
    }

    out_path = pathlib.Path("docs/runs/2026-08-19-intervention-feature-correlations.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved full intervention dataset and correlation analysis to {out_path}")
    return output


if __name__ == "__main__":
    build_intervention_dataset()
