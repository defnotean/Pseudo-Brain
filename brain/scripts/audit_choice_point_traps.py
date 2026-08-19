"""Choice-Point Trap & Tactical Evasion Audit.

Evaluates high-stakes choice points (intersections with pursuing ghosts within 2-3 cells)
to test whether 2-step lookahead planning actively prevents corner traps and outperforms the reactive policy.
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


def run_tactical_choice_point_audit(
    eval_seeds: list[int] = list(range(3001, 3051)),
    lookahead_eval_steps: int = 4,
):
    print("=" * 80)
    print("PSEUDO-BRAIN TACTICAL CHOICE-POINT & CORNER TRAP AUDIT")
    print("Testing 2-Step Lookahead Planning on High-Stakes Ghost Convergence Points")
    print("=" * 80)

    results = {}
    dests = [(0, 0), (0, -1), (-1, 0), (0, 1), (1, 0)]  # None, W, A, S, D

    for seed in TRAIN_SEEDS:
        ckpt_path = pathlib.Path(f"artifacts/checkpoints/mistake_unrolled_seed_{seed}.pt")
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = IreneBrainModel(ckpt["config"], enable_adaptive_cognition=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        cf_head = getattr(model, "counterfactual_foresight_head", None)
        planner_h2 = LatentLookaheadPlanner(model=model, horizon=2, gamma=0.95, hazard_weight=8.0)

        choice_points_evaluated = 0
        planner_interventions = 0
        helpful_saves = 0
        harmful_errors = 0
        neutral_cases = 0

        env = MazeChaseEnv()

        for w_seed in eval_seeds:
            obs = env.reset(seed=w_seed)
            state = model.initial_state(1)
            last_ctrl = torch.zeros((1, 307), dtype=torch.float32)

            for step in range(60):
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

                # Check if this is a high-hazard tactical choice point
                walkable, pellets, player, ghosts = _parse_maze_chase_frame(
                    obs.rgb, caller="diagnostic.scripted_maze_chase_planner.v1"
                )
                is_tactical = False
                if player is not None and ghosts:
                    min_g_dist = min(abs(player[0] - gx) + abs(player[1] - gy) for gx, gy in ghosts)
                    if min_g_dist <= 3:
                        is_tactical = True

                chosen_act = d_act

                if is_tactical and cf_head is not None:
                    choice_points_evaluated += 1
                    cf_preds = cf_head.forward_all_actions(out.next_state.thoughts)
                    hazards = {a_i: float(cf_preds[a_i].hazard_probability[0, 0, 0].item()) for a_i in range(5)}
                    min_haz = min(hazards.values())
                    direct_haz = hazards.get(d_act, 0.0)
                    contrast_margin = direct_haz - min_haz

                    # Evaluate 2-step lookahead branch utility
                    sensors = model.pixel_encoder(rgb)
                    cand_seqs = generate_directional_candidate_sequences(2, include_none=False)
                    branch_res = planner_h2.evaluate_candidate_branches(
                        state=out.next_state,
                        sensors=sensors,
                        candidate_sequences=cand_seqs,
                    )
                    valid_branches = [b for b in branch_res if not b.is_pruned]
                    best_planner_act = int(max(valid_branches, key=lambda b: b.cumulative_utility).action_sequence[0]) if valid_branches else min(hazards, key=hazards.get)

                    if best_planner_act != d_act:
                        planner_interventions += 1
                        # Counterfactual Fork Test
                        snap = env.snapshot()

                        # Fork A: Direct Action Forward 4 Steps
                        env.restore(snap)
                        direct_caught = False
                        out_d = env.step(act_index_to_control(d_act))
                        if "caught" in out_d.events: direct_caught = True
                        for _ in range(lookahead_eval_steps - 1):
                            if out_d.terminated or out_d.truncated: break
                            out_d = env.step(GenericControl())
                            if "caught" in out_d.events: direct_caught = True

                        # Fork B: Planner Action Forward 4 Steps
                        env.restore(snap)
                        planner_caught = False
                        out_p = env.step(act_index_to_control(best_planner_act))
                        if "caught" in out_p.events: planner_caught = True
                        for _ in range(lookahead_eval_steps - 1):
                            if out_p.terminated or out_p.truncated: break
                            out_p = env.step(GenericControl())
                            if "caught" in out_p.events: planner_caught = True

                        env.restore(snap)

                        if direct_caught and not planner_caught:
                            helpful_saves += 1
                        elif planner_caught and not direct_caught:
                            harmful_errors += 1
                        else:
                            neutral_cases += 1

                        chosen_act = best_planner_act

                outcome = env.step(act_index_to_control(chosen_act))
                obs = outcome.observation
                if outcome.terminated or outcome.truncated:
                    break

        ivr = (helpful_saves / (helpful_saves + harmful_errors) * 100.0) if (helpful_saves + harmful_errors) > 0 else 100.0
        seed_res = {
            "tactical_choice_points": choice_points_evaluated,
            "interventions": planner_interventions,
            "helpful_saves": helpful_saves,
            "harmful_errors": harmful_errors,
            "neutral_cases": neutral_cases,
            "intervention_value_ratio_pct": round(ivr, 2),
            "net_save_value": helpful_saves - harmful_errors,
        }

        print(f"\n--- SEED {seed} TACTICAL CHOICE-POINT RESULTS ---")
        print(f"Choice Points Evaluated: {choice_points_evaluated} | Interventions: {planner_interventions}")
        print(f"Outcome Breakdown      : HELPFUL SAVES={helpful_saves} | HARMFUL ERRORS={harmful_errors} | NEUTRAL={neutral_cases}")
        print(f"Intervention Quality   : IVR = {seed_res['intervention_value_ratio_pct']}% | Net Saves = {seed_res['net_save_value']:+d}")

        results[seed] = seed_res

    out_path = pathlib.Path("docs/runs/2026-08-19-tactical-choice-point-audit.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved tactical choice-point results to {out_path}")


if __name__ == "__main__":
    run_tactical_choice_point_audit()
