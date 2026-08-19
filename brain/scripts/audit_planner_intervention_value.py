"""Planner Intervention Value (PIV) Forensic Audit.

Measures the ground-truth counterfactual quality of every single planner intervention:
Whenever the planner overrides the direct policy:
1. Snapshots the exact simulator state.
2. Rolls out the Direct Policy action forward 3 steps in a counterfactual fork.
3. Rolls out the Planner action forward 3 steps in the factual fork.
4. Classifies each intervention as HELPFUL, NEUTRAL, or HARMFUL.
5. Computes the Intervention Value Ratio (IVR = Helpful / (Helpful + Harmful)) and Net Value Score.
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


def evaluate_intervention_value_audit(
    eval_seeds: list[int] = list(range(2001, 2051)),
    delta_danger: float = 0.005,
    lookahead_eval_steps: int = 3,
):
    print("=" * 80)
    print("PSEUDO-BRAIN PLANNER INTERVENTION VALUE (PIV) FORENSIC AUDIT")
    print(f"Auditing Counterfactual vs Factual Outcomes across Seeds 42-46 (50 Worlds, delta={delta_danger})")
    print("=" * 80)

    audit_summary = {}

    for seed in TRAIN_SEEDS:
        ckpt_path = pathlib.Path(f"artifacts/checkpoints/mistake_unrolled_seed_{seed}.pt")
        if not ckpt_path.exists():
            print(f"Checkpoint for seed {seed} not found, skipping.")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = IreneBrainModel(ckpt["config"], enable_adaptive_cognition=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        cf_head = getattr(model, "counterfactual_foresight_head", None)
        planner_h2 = LatentLookaheadPlanner(model=model, horizon=2, gamma=0.95, hazard_weight=8.0)

        total_steps = 0
        total_interventions = 0
        helpful_count = 0
        neutral_count = 0
        harmful_count = 0

        interventions_log = []
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
                chosen_act = d_act
                intervened = False
                contrast_margin = 0.0

                if cf_head is not None:
                    cf_preds = cf_head.forward_all_actions(out.next_state.thoughts)
                    hazards = {a_i: float(cf_preds[a_i].hazard_probability[0, 0, 0].item()) for a_i in range(5)}
                    min_haz = min(hazards.values())
                    direct_haz = hazards.get(d_act, 0.0)
                    contrast_margin = direct_haz - min_haz

                    if contrast_margin >= delta_danger:
                        intervened = True
                        best_safe_act = min(hazards, key=hazards.get)
                        chosen_act = best_safe_act

                if intervened:
                    total_interventions += 1
                    # Counterfactual Simulation Audit:
                    # 1. Snapshot simulator state at this exact decision point
                    snap = env.snapshot()

                    # 2. Branch A: Direct Action Counterfactual Rollout
                    env.restore(snap)
                    direct_caught = False
                    direct_pellets = 0
                    # Step direct action first
                    out_d = env.step(act_index_to_control(d_act))
                    if "caught" in out_d.events: direct_caught = True
                    if "pellet" in out_d.events: direct_pellets += 1
                    # Step 2 more ticks with neutral/continuation
                    for _ in range(lookahead_eval_steps - 1):
                        if out_d.terminated or out_d.truncated: break
                        out_d = env.step(GenericControl())
                        if "caught" in out_d.events: direct_caught = True
                        if "pellet" in out_d.events: direct_pellets += 1

                    # 3. Branch B: Planner Action Factual Rollout
                    env.restore(snap)
                    planner_caught = False
                    planner_pellets = 0
                    out_p = env.step(act_index_to_control(chosen_act))
                    if "caught" in out_p.events: planner_caught = True
                    if "pellet" in out_p.events: planner_pellets += 1
                    for _ in range(lookahead_eval_steps - 1):
                        if out_p.terminated or out_p.truncated: break
                        out_p = env.step(GenericControl())
                        if "caught" in out_p.events: planner_caught = True
                        if "pellet" in out_p.events: planner_pellets += 1

                    # 4. Classify Outcome
                    if direct_caught and not planner_caught:
                        # Planner saved the agent from a catch!
                        classification = "HELPFUL"
                        helpful_count += 1
                    elif planner_caught and not direct_caught:
                        # Planner caused an unnecessary catch
                        classification = "HARMFUL"
                        harmful_count += 1
                    elif planner_pellets > direct_pellets and not planner_caught:
                        # Planner collected more pellets safely
                        classification = "HELPFUL"
                        helpful_count += 1
                    elif direct_pellets > planner_pellets and not direct_caught:
                        # Direct collected more pellets
                        classification = "HARMFUL"
                        harmful_count += 1
                    else:
                        classification = "NEUTRAL"
                        neutral_count += 1

                    interventions_log.append({
                        "world_seed": w_seed,
                        "step": step,
                        "direct_action": d_act,
                        "planner_action": chosen_act,
                        "contrast_margin": round(float(contrast_margin), 4),
                        "direct_caught": direct_caught,
                        "planner_caught": planner_caught,
                        "direct_pellets": direct_pellets,
                        "planner_pellets": planner_pellets,
                        "classification": classification,
                    })

                    # Restore simulator to factual planner state and continue main rollout
                    env.restore(snap)

                # Main factual progression
                outcome = env.step(act_index_to_control(chosen_act))
                obs = outcome.observation
                if outcome.terminated or outcome.truncated:
                    break

        ivr = (helpful_count / (helpful_count + harmful_count) * 100.0) if (helpful_count + harmful_count) > 0 else 100.0
        net_value = helpful_count - harmful_count

        seed_summary = {
            "total_steps": total_steps,
            "total_interventions": total_interventions,
            "intervention_rate_pct": round((total_interventions / total_steps) * 100.0, 2),
            "helpful": helpful_count,
            "neutral": neutral_count,
            "harmful": harmful_count,
            "intervention_value_ratio_pct": round(ivr, 2),
            "net_value": net_value,
            "sample_log": interventions_log[:10],
        }

        print(f"\n--- SEED {seed} INTERVENTION VALUE AUDIT ---")
        print(f"Total Steps: {total_steps} | Interventions: {total_interventions} ({seed_summary['intervention_rate_pct']}%)")
        print(f"Breakdown  : HELPFUL={helpful_count} | NEUTRAL={neutral_count} | HARMFUL={harmful_count}")
        print(f"Quality    : Intervention Value Ratio (IVR) = {seed_summary['intervention_value_ratio_pct']}% | Net Value = {net_value:+d}")

        audit_summary[seed] = seed_summary

    out_path = pathlib.Path("docs/runs/2026-08-19-planner-intervention-value-audit.json")
    with open(out_path, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nSaved full intervention value audit to {out_path}")


if __name__ == "__main__":
    evaluate_intervention_value_audit()
