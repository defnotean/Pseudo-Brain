"""Empirical audit of predicted hazard distributions on safe vs true-danger states
across Seeds 42-46. Computes mean, p50, p95, Brier score, and relative margin.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import numpy as np
import torch

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
SCRIPTS = pathlib.Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.diagnostic_policies import _parse_maze_chase_frame
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.training.objective import _rgb_tensor
from irene_brain.types import GenericControl, HidKey


TRAIN_SEEDS = [42, 43, 44, 45, 46]


def audit_hazard_distributions(eval_seeds: list[int] = list(range(2001, 2051))):
    print("=" * 80)
    print("PSEUDO-BRAIN HAZARD DISTRIBUTION & CALIBRATION AUDIT")
    print("Auditing Safe vs True-Danger Predicted Probabilities Across Seeds 42-46")
    print("=" * 80)

    results = {}
    dests = [(0, 0), (0, -1), (-1, 0), (0, 1), (1, 0)]  # None, W, A, S, D

    for seed in TRAIN_SEEDS:
        ckpt_path = pathlib.Path(f"artifacts/checkpoints/mistake_unrolled_seed_{seed}.pt")
        if not ckpt_path.exists():
            print(f"Checkpoint for seed {seed} not found at {ckpt_path}, skipping.")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = IreneBrainModel(ckpt["config"], enable_adaptive_cognition=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        cf_head = getattr(model, "counterfactual_foresight_head", None)
        if cf_head is None:
            print(f"Seed {seed} does not have counterfactual foresight head.")
            continue

        safe_hazards = []
        danger_hazards = []
        relative_margins = []  # d(danger) - d(safe_min)

        env = MazeChaseEnv()

        for w_seed in eval_seeds:
            obs = env.reset(seed=w_seed)
            state = model.initial_state(1)
            last_ctrl = torch.zeros((1, 307), dtype=torch.float32)

            for step in range(40):
                rgb = _rgb_tensor((obs.rgb,), device="cpu", resolution=getattr(model, "input_resolution", None))
                dt = torch.tensor([0.016], dtype=torch.float32)

                with torch.no_grad():
                    out = model(rgb, last_ctrl, dt, state)
                    state = out.next_state
                    last_ctrl = out.action.control
                    cf_preds = cf_head.forward_all_actions(out.next_state.thoughts)

                walkable, pellets, player, ghosts = _parse_maze_chase_frame(
                    obs.rgb,
                    caller="diagnostic.scripted_maze_chase_planner.v1",
                )
                if player is None:
                    continue
                px, py = player

                # Identify ground-truth danger for each action
                step_pred_hazards = []
                step_true_danger = []

                for a_i, (dx, dy) in enumerate(dests):
                    nx, ny = px + dx, py + dy
                    is_danger = False
                    # Ghost collision or proximity within 1 step
                    for gx, gy in ghosts:
                        if (nx == gx and ny == gy) or (abs(nx - gx) + abs(ny - gy) <= 1):
                            is_danger = True
                            break

                    pred_h = float(cf_preds[a_i].hazard_probability[0, 0, 0].item())
                    step_pred_hazards.append(pred_h)
                    step_true_danger.append(is_danger)

                    if is_danger:
                        danger_hazards.append(pred_h)
                    else:
                        safe_hazards.append(pred_h)

                # If there is at least one danger and one safe action at this state
                has_danger = any(step_true_danger)
                has_safe = any(not d for d in step_true_danger)
                if has_danger and has_safe:
                    dang_preds = [step_pred_hazards[i] for i in range(5) if step_true_danger[i]]
                    safe_preds = [step_pred_hazards[i] for i in range(5) if not step_true_danger[i]]
                    margin = np.mean(dang_preds) - np.min(safe_preds)
                    relative_margins.append(margin)

                # Step environment with neutral or safe move
                outcome = env.step(GenericControl())
                obs = outcome.observation
                if outcome.terminated or outcome.truncated:
                    break

        safe_arr = np.array(safe_hazards)
        danger_arr = np.array(danger_hazards)
        margin_arr = np.array(relative_margins)

        # Brier score (mean squared error of probabilities vs binary outcome)
        all_preds = np.concatenate([safe_arr, danger_arr])
        all_targets = np.concatenate([np.zeros_like(safe_arr), np.ones_like(danger_arr)])
        brier_score = np.mean((all_preds - all_targets) ** 2)

        stats = {
            "safe_states": {
                "count": len(safe_arr),
                "mean": round(float(np.mean(safe_arr)), 4),
                "p25": round(float(np.percentile(safe_arr, 25)), 4),
                "p50": round(float(np.percentile(safe_arr, 50)), 4),
                "p75": round(float(np.percentile(safe_arr, 75)), 4),
                "p90": round(float(np.percentile(safe_arr, 90)), 4),
                "p95": round(float(np.percentile(safe_arr, 95)), 4),
                "max": round(float(np.max(safe_arr)), 4),
            },
            "danger_states": {
                "count": len(danger_arr),
                "mean": round(float(np.mean(danger_arr)), 4),
                "p25": round(float(np.percentile(danger_arr, 25)), 4),
                "p50": round(float(np.percentile(danger_arr, 50)), 4),
                "p75": round(float(np.percentile(danger_arr, 75)), 4),
                "p90": round(float(np.percentile(danger_arr, 90)), 4),
                "p95": round(float(np.percentile(danger_arr, 95)), 4),
                "max": round(float(np.max(danger_arr)), 4),
            },
            "relative_margin": {
                "mean": round(float(np.mean(margin_arr)), 4),
                "p50": round(float(np.percentile(margin_arr, 50)), 4),
                "positive_rate": round(float(np.mean(margin_arr > 0)) * 100, 2),
            },
            "brier_score": round(float(brier_score), 4),
        }

        print(f"\n--- SEED {seed} HAZARD STATS ---")
        print(f"Safe States   (N={len(safe_arr)}): Mean={stats['safe_states']['mean']} | p50={stats['safe_states']['p50']} | p95={stats['safe_states']['p95']}")
        print(f"Danger States (N={len(danger_arr)}): Mean={stats['danger_states']['mean']} | p50={stats['danger_states']['p50']} | p95={stats['danger_states']['p95']}")
        print(f"Relative Contrast Margin: Mean={stats['relative_margin']['mean']:+.4f} | Positive Rate={stats['relative_margin']['positive_rate']}%")
        print(f"Brier Calibration Score: {stats['brier_score']:.4f}")

        results[seed] = stats

    out_path = pathlib.Path("docs/runs/2026-08-19-hazard-calibration-audit.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full calibration audit to {out_path}")


if __name__ == "__main__":
    audit_hazard_distributions()
