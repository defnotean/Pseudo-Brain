"""Parametric sweep over relative hazard contrast and opportunity routing thresholds
to identify the optimal calibration where planner interventions are rare, safe, and improve pellets/catches.
"""

from __future__ import annotations

import json
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
from irene_brain.model.spec import ThoughtFieldConfig
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


def evaluate_threshold_combination(
    models: dict[int, IreneBrainModel],
    delta_danger: float,
    delta_opportunity: float,
    eval_seeds: list[int] = list(range(2001, 2021)),
) -> dict[str, Any]:
    env = MazeChaseEnv()
    seed_catches = []
    seed_pellets = []
    seed_interventions = []

    for seed, model in models.items():
        planner_h2 = LatentLookaheadPlanner(model=model, horizon=2, gamma=0.95, hazard_weight=8.0)
        cf_head = getattr(model, "counterfactual_foresight_head", None)
        topo_head = getattr(model, "topological_goal_head", None)

        total_c = 0
        total_p = 0
        total_int = 0

        for w in eval_seeds:
            obs = env.reset(seed=w)
            state = model.initial_state(1)
            last_ctrl = torch.zeros((1, 307), dtype=torch.float32)

            for step in range(50):
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
                chosen_act = d_act

                if cf_head is not None:
                    cf_preds = cf_head.forward_all_actions(out.next_state.thoughts)
                    hazards = {a_i: float(cf_preds[a_i].hazard_probability[0, 0, 0].item()) for a_i in range(5)}
                    min_haz = min(hazards.values())
                    direct_haz = hazards.get(d_act, 0.0)
                    contrast_margin = direct_haz - min_haz

                    # Safety Intervention Check
                    if contrast_margin >= delta_danger:
                        total_int += 1
                        chosen_act = min(hazards, key=hazards.get)
                    elif delta_opportunity > 0.0 and topo_head is not None:
                        # Opportunity check: If direct action is safe, check topological goal routing bonus
                        topo_pred = topo_head(out.next_state.thoughts)
                        j_logits = topo_pred.junction_exit_logits[0].tolist()
                        best_topo_act = int(np.argmax(j_logits[1:])) + 1
                        if best_topo_act != d_act and (j_logits[best_topo_act] - j_logits[d_act] >= delta_opportunity):
                            if hazards.get(best_topo_act, 0.0) <= min_haz + 0.01:
                                total_int += 1
                                chosen_act = best_topo_act

                key_map = {1: (HidKey.W,), 2: (HidKey.A,), 3: (HidKey.S,), 4: (HidKey.D,)}
                ctrl = GenericControl(keys_down=key_map.get(chosen_act, ()))
                outcome = env.step(ctrl)
                obs = outcome.observation
                if "pellet" in outcome.events: total_p += 1
                if "caught" in outcome.events: total_c += 1

        seed_catches.append(total_c)
        seed_pellets.append(total_p)
        seed_interventions.append(total_int)

    return {
        "delta_danger": delta_danger,
        "delta_opportunity": delta_opportunity,
        "mean_catches": round(float(np.mean(seed_catches)), 2),
        "mean_pellets": round(float(np.mean(seed_pellets)), 2),
        "mean_interventions": round(float(np.mean(seed_interventions)), 1),
        "seed_catches": seed_catches,
        "seed_interventions": seed_interventions,
    }


def run_sweep():
    print("=" * 80)
    print("PARAMETRIC SWEEP: RELATIVE HAZARD CONTRAST & OPPORTUNITY GATING")
    print("=" * 80)

    # Load all models once into memory
    models = {}
    for seed in TRAIN_SEEDS:
        ckpt_path = pathlib.Path(f"artifacts/checkpoints/mistake_unrolled_seed_{seed}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = IreneBrainModel(ckpt["config"], enable_adaptive_cognition=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models[seed] = model

    danger_thresholds = [0.005, 0.01, 0.015, 0.02, 0.03]
    opportunity_thresholds = [0.0, 1.0, 2.0]

    sweep_results = []

    for d_dang in danger_thresholds:
        for d_opp in opportunity_thresholds:
            res = evaluate_threshold_combination(models, d_dang, d_opp)
            print(f"Danger delta={d_dang:0.3f} | Opp delta={d_opp:0.1f} -> Catches={res['mean_catches']} | Interventions={res['mean_interventions']}/1000 (Per-Seed: {res['seed_interventions']})")
            sweep_results.append(res)

    out_path = pathlib.Path("docs/runs/2026-08-19-gating-sweep-results.json")
    with open(out_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved sweep results to {out_path}")


if __name__ == "__main__":
    run_sweep()
