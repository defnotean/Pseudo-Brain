"""Horizon Depth Ablation Study for Pseudo-Brain.

Evaluates identical trained models across Seeds 42-46 with:
1. Direct Policy (No planning)
2. Planner H=1 (1-step lookahead)
3. Planner H=2 (2-step lookahead)
4. Planner H=3 (3-step lookahead)
5. Epistemic Horizon Gating (Confidence-weighted / H<=2)

Also runs on 50 held-out evaluation worlds (seeds 2001-2050) to evaluate 100+ genuine hazard choice points.
"""

from __future__ import annotations

import copy
import json
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

from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.closed_loop_play import (
    EXCLUSIVE_ARGMAX_WASD_V1,
    decode_closed_loop_control,
)
from irene_brain.evaluation.diagnostic_policies import _parse_maze_chase_frame
from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.training.batches import (
    CONTINUOUS_TARGET_INDICES,
    control_to_vector,
)
from irene_brain.training.objective import _rgb_tensor
from irene_brain.types import GenericControl, HidKey

from train_multistep_stable_foresight_battery import (
    _build_action_control,
    audit_per_horizon_discrimination,
    train_multistep_stable_seed,
)

TRAIN_SEEDS = (42, 43, 44, 45, 46)


def evaluate_policy_at_horizon(
    model: IreneBrainModel,
    horizon: int | None,
    num_eval_seeds: int = 20,
    seed_start: int = 2001,
) -> dict[str, Any]:
    """Evaluate closed-loop policy with a specific lookahead horizon (or None for direct)."""
    model.eval()
    if horizon is not None:
        policy = LatentLookaheadPolicy(
            model=model,
            horizon=horizon,
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

        if policy is not None:
            policy.reset(seed)

        ep_pellets = 0
        ep_catches = 0
        ep_steps = 0

        for step in range(50):
            ep_steps += 1
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
        total_survival += ep_steps

    return {
        "mean_pellets_per_ep": round(total_pellets / num_eval_seeds, 2),
        "total_catches": total_catches,
        "mean_catches_per_ep": round(total_catches / num_eval_seeds, 2),
        "mean_survival_steps": round(total_survival / num_eval_seeds, 2),
    }




def main() -> None:
    print("=" * 80)
    print("PSEUDO-BRAIN: HORIZON DEPTH ABLATION STUDY (H=1, H=2, H=3)")
    print("Direct Policy vs H=1 vs H=2 vs H=3 on Identical Models Across 5 Seeds")
    print("=" * 80)

    ablation_results = {}

    for seed in TRAIN_SEEDS:
        print(f"\n=======================================================")
        print(f"TRAINING SEED {seed}")
        print(f"=======================================================")
        t0 = time.perf_counter()
        model = train_multistep_stable_seed(seed, num_dagger_iters=4, updates_per_iter=24)
        t1 = time.perf_counter()
        print(f"Training completed in {t1-t0:.1f}s")

        print(f"Auditing Large-Scale Choice Points (50 Worlds: Seeds 2001-2050)...")
        diag = audit_per_horizon_discrimination(model, num_eval_seeds=50, seed_start=2001)
        hb = diag["horizon_breakdown"]
        print(f"  Total Choice Points: {diag['choice_points_evaluated']} | Safe Pick Rate: {diag['choice_point_safe_pick_rate']}%")
        print(f"  H=1 AUROC: {hb['H1']['pairwise_ranking_auroc']}% (Margin: {hb['H1']['discrimination_margin']:+.4f})")
        print(f"  H=2 AUROC: {hb['H2']['pairwise_ranking_auroc']}% (Margin: {hb['H2']['discrimination_margin']:+.4f})")
        print(f"  H=3 AUROC: {hb['H3']['pairwise_ranking_auroc']}% (Margin: {hb['H3']['discrimination_margin']:+.4f})")

        print(f"\nRunning Horizon Depth Ablation on 20 Held-Out Evaluation Worlds...")
        direct_res = evaluate_policy_at_horizon(model, horizon=None, num_eval_seeds=20, seed_start=2001)
        h1_res = evaluate_policy_at_horizon(model, horizon=1, num_eval_seeds=20, seed_start=2001)
        h2_res = evaluate_policy_at_horizon(model, horizon=2, num_eval_seeds=20, seed_start=2001)
        h3_res = evaluate_policy_at_horizon(model, horizon=3, num_eval_seeds=20, seed_start=2001)

        print(f"  Direct Controller : {direct_res['total_catches']} catches ({direct_res['mean_catches_per_ep']}/ep)")
        print(f"  Planner H=1 (1-step): {h1_res['total_catches']} catches ({h1_res['mean_catches_per_ep']}/ep)")
        print(f"  Planner H=2 (2-step): {h2_res['total_catches']} catches ({h2_res['mean_catches_per_ep']}/ep)")
        print(f"  Planner H=3 (3-step): {h3_res['total_catches']} catches ({h3_res['mean_catches_per_ep']}/ep)")

        ablation_results[str(seed)] = {
            "diagnostics_50_worlds": diag,
            "direct": direct_res,
            "h1": h1_res,
            "h2": h2_res,
            "h3": h3_res,
        }

    out_file = pathlib.Path("docs/runs/2026-08-19-horizon-depth-ablation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(ablation_results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY OF HORIZON DEPTH ABLATION (SEEDS 42-46)")
    print("=" * 80)
    print(f"{'Seed':<6} | {'Direct':<8} | {'H=1':<8} | {'H=2':<8} | {'H=3':<8} | {'H1 AUROC':<10} | {'H2 AUROC':<10} | {'H3 AUROC':<10}")
    print("-" * 80)
    for seed, r in ablation_results.items():
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
