"""High-Precision Cognitive Veto & Intervention Quality Battery.

Implements the multi-signal cognitive veto rule:
Intervene ONLY when:
1. Direct Policy is uncertain (PolicyEntropy >= theta_entropy).
2. Direct Action is noticeably hazardous (Contrast Margin Delta >= delta_contrast).
3. Mental Foresight is consistent (H1 vs H2 drift <= epsilon_drift).

Evaluates across Seeds 42-46 on 50 held-out worlds (Seeds 2001-2050).
"""

from __future__ import annotations

import json
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


def run_high_precision_veto_battery(
    eval_seeds: list[int] = list(range(2001, 2051)),
    entropy_threshold: float = 0.40,
    contrast_threshold: float = 0.010,
    max_drift_threshold: float = 0.040,
    lookahead_eval_steps: int = 3,
):
    print("=" * 80)
    print("PSEUDO-BRAIN HIGH-PRECISION COGNITIVE VETO BATTERY")
    print(f"Entropy >= {entropy_threshold} | Contrast >= {contrast_threshold} | Drift <= {max_drift_threshold}")
    print("=" * 80)

    results = {}

    for seed in TRAIN_SEEDS:
        ckpt_path = pathlib.Path(f"artifacts/checkpoints/mistake_unrolled_seed_{seed}.pt")
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = IreneBrainModel(ckpt["config"], enable_adaptive_cognition=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        cf_head = getattr(model, "counterfactual_foresight_head", None)

        total_steps = 0
        eligible_choice_points = 0
        interventions = 0
        helpful_saves = 0
        neutral_cases = 0
        harmful_errors = 0

        direct_catches = 0
        veto_catches = 0

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

                # Compute Policy Entropy
                probs = F.softmax(logits, dim=-1).numpy()
                action_entropy = -float(np.sum(probs * np.log(probs + 1e-8)))

                chosen_act = d_act
                intervened = False

                if cf_head is not None:
                    cf_preds = cf_head.forward_all_actions(out.next_state.thoughts)
                    hazards_h1 = {a_i: float(cf_preds[a_i].hazard_probability[0, 0, 0].item()) for a_i in range(5)}
                    hazards_h2 = {a_i: float(cf_preds[a_i].hazard_probability[0, min(1, cf_preds[a_i].hazard_probability.shape[1] - 1), 0].item()) for a_i in range(5)}

                    min_haz_h1 = min(hazards_h1.values())
                    direct_haz = hazards_h1.get(d_act, 0.0)
                    contrast_margin = direct_haz - min_haz_h1
                    best_safe_act = min(hazards_h1, key=hazards_h1.get)
                    drift = abs(hazards_h1[best_safe_act] - hazards_h2[best_safe_act])

                    if best_safe_act != d_act:
                        eligible_choice_points += 1

                        # Multi-Signal Cognitive Veto Condition:
                        # 1. Direct policy must be somewhat uncertain (entropy >= threshold)
                        # 2. Hazard contrast must be real (contrast >= threshold)
                        # 3. Foresight drift must be low (drift <= threshold)
                        if (
                            action_entropy >= entropy_threshold
                            and contrast_margin >= contrast_threshold
                            and drift <= max_drift_threshold
                        ):
                            intervened = True
                            interventions += 1
                            chosen_act = best_safe_act

                            # Counterfactual Fork Test
                            snap = env.snapshot()

                            # Fork A: Direct Action Forward
                            env.restore(snap)
                            direct_caught = False
                            out_d = env.step(act_index_to_control(d_act))
                            if "caught" in out_d.events: direct_caught = True
                            for _ in range(lookahead_eval_steps - 1):
                                if out_d.terminated or out_d.truncated: break
                                out_d = env.step(GenericControl())
                                if "caught" in out_d.events: direct_caught = True

                            # Fork B: Planner Action Forward
                            env.restore(snap)
                            planner_caught = False
                            out_p = env.step(act_index_to_control(best_safe_act))
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

                outcome = env.step(act_index_to_control(chosen_act))
                obs = outcome.observation
                if "caught" in outcome.events:
                    veto_catches += 1
                if outcome.terminated or outcome.truncated:
                    break

        precision = (
            round((helpful_saves / (helpful_saves + harmful_errors)) * 100.0, 2)
            if (helpful_saves + harmful_errors) > 0
            else None
        )
        coverage = (
            round((interventions / eligible_choice_points) * 100.0, 2)
            if eligible_choice_points > 0
            else 0.0
        )

        results[seed] = {
            "total_steps": total_steps,
            "eligible_choice_points": eligible_choice_points,
            "interventions": interventions,
            "coverage_pct": coverage,
            "helpful_saves": helpful_saves,
            "neutral_cases": neutral_cases,
            "harmful_errors": harmful_errors,
            "intervention_precision_pct": precision,
            "net_save_value": helpful_saves - harmful_errors,
            "veto_catches": veto_catches,
        }

        print(f"\n--- SEED {seed} HIGH-PRECISION VETO RESULTS ---")
        print(f"Total Steps: {total_steps} | Eligible Points: {eligible_choice_points} | Interventions: {interventions} ({coverage}%)")
        print(f"Breakdown  : HELPFUL SAVES={helpful_saves} | NEUTRAL={neutral_cases} | HARMFUL ERRORS={harmful_errors}")
        print(f"Precision  : {precision if precision is not None else 'N/A (0 active decisions)'}% | Net Saves = {helpful_saves - harmful_errors:+d}")

    out_path = pathlib.Path("docs/runs/2026-08-19-high-precision-veto-results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved high precision veto results to {out_path}")
    return results


if __name__ == "__main__":
    run_high_precision_veto_battery()
