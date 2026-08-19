"""Evaluation and audit of Relative-Contrast Gated Foresight and Accumulated Multi-Step Planning (H=1 vs H=2) across Seeds 42-46.
"""

from __future__ import annotations

import hashlib
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
from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
from irene_brain.model.intent import DirectionalAction
from irene_brain.model.lookahead_planner import (
    LatentLookaheadPlanner,
    directional_to_control_vector,
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


def run_relative_gating_battery(
    eval_seeds: list[int] = list(range(2001, 2021)),
    contrast_threshold: float = 0.03,
):
    print("=" * 80)
    print("PSEUDO-BRAIN RELATIVE-CONTRAST GATED FORESIGHT BATTERY")
    print(f"Auditing Direct vs H=1 vs H=2 with Relative Contrast Threshold delta={contrast_threshold}")
    print("=" * 80)

    summary_results = {}

    for seed in TRAIN_SEEDS:
        ckpt_path = pathlib.Path(f"artifacts/checkpoints/mistake_unrolled_seed_{seed}.pt")
        if not ckpt_path.exists():
            print(f"Checkpoint for seed {seed} not found, skipping.")
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = IreneBrainModel(ckpt["config"], enable_adaptive_cognition=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        controllers = ["direct", "rel_gated_h1", "rel_gated_h2"]
        world_actions = {c: {} for c in controllers}
        world_catches = {c: 0 for c in controllers}
        world_pellets = {c: 0 for c in controllers}
        interventions = {c: 0 for c in controllers}

        env = MazeChaseEnv()

        # 1. Direct Controller
        for w in eval_seeds:
            obs = env.reset(seed=w)
            state = model.initial_state(1)
            last_ctrl = torch.zeros((1, 307), dtype=torch.float32)
            d_acts = []
            ep_c, ep_p = 0, 0

            for step in range(50):
                rgb = _rgb_tensor((obs.rgb,), device="cpu", resolution=getattr(model, "input_resolution", None))
                dt = torch.tensor([0.016], dtype=torch.float32)
                with torch.no_grad():
                    out = model(rgb, last_ctrl, dt, state)
                    state = out.next_state
                    last_ctrl = out.action.control
                logits = out.action.button_logits[0].float().cpu()
                continuous_values = out.action.control[0].float().cpu()
                control, _ = decode_closed_loop_control(
                    logits.tolist(),
                    [float(continuous_values[idx]) for idx in CONTINUOUS_TARGET_INDICES],
                    decode_kind=EXCLUSIVE_ARGMAX_WASD_V1,
                )
                act_idx = control_to_act_index(control)
                d_acts.append(act_idx)

                outcome = env.step(control)
                obs = outcome.observation
                if "pellet" in outcome.events: ep_p += 1
                if "caught" in outcome.events: ep_c += 1

            world_actions["direct"][w] = d_acts
            world_catches["direct"] += ep_c
            world_pellets["direct"] += ep_p

        # 2. Relative-Contrast Gated H=1
        for w in eval_seeds:
            obs = env.reset(seed=w)
            state = model.initial_state(1)
            last_ctrl = torch.zeros((1, 307), dtype=torch.float32)
            acts = []
            ep_c, ep_p = 0, 0
            ep_interventions = 0

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

                # Evaluate Relative Hazard Contrast
                cf_head = getattr(model, "counterfactual_foresight_head", None)
                chosen_act = d_act

                if cf_head is not None:
                    cf_preds = cf_head.forward_all_actions(out.next_state.thoughts)
                    hazards = {a_i: float(cf_preds[a_i].hazard_probability[0, 0, 0].item()) for a_i in range(5)}
                    min_haz = min(hazards.values())
                    direct_haz = hazards.get(d_act, 0.0)
                    contrast_margin = direct_haz - min_haz

                    if contrast_margin >= contrast_threshold:
                        # Direct action is significantly more dangerous than safest option!
                        ep_interventions += 1
                        # Choose action with lowest hazard
                        best_safe_act = min(hazards, key=hazards.get)
                        chosen_act = best_safe_act

                acts.append(chosen_act)
                key_map = {1: (HidKey.W,), 2: (HidKey.A,), 3: (HidKey.S,), 4: (HidKey.D,)}
                ctrl = GenericControl(keys_down=key_map.get(chosen_act, ()))
                outcome = env.step(ctrl)
                obs = outcome.observation
                if "pellet" in outcome.events: ep_p += 1
                if "caught" in outcome.events: ep_c += 1

            world_actions["rel_gated_h1"][w] = acts
            world_catches["rel_gated_h1"] += ep_c
            world_pellets["rel_gated_h1"] += ep_p
            interventions["rel_gated_h1"] += ep_interventions

        # 3. Relative-Contrast Gated H=2 (Accumulated 2-Step Latent Foresight)
        planner_h2 = LatentLookaheadPlanner(model=model, horizon=2, gamma=0.95, hazard_weight=8.0)
        for w in eval_seeds:
            obs = env.reset(seed=w)
            state = model.initial_state(1)
            last_ctrl = torch.zeros((1, 307), dtype=torch.float32)
            acts = []
            ep_c, ep_p = 0, 0
            ep_interventions = 0

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

                # Check 1-step hazard contrast
                cf_head = getattr(model, "counterfactual_foresight_head", None)
                chosen_act = d_act

                if cf_head is not None:
                    cf_preds = cf_head.forward_all_actions(out.next_state.thoughts)
                    hazards = {a_i: float(cf_preds[a_i].hazard_probability[0, 0, 0].item()) for a_i in range(5)}
                    min_haz = min(hazards.values())
                    direct_haz = hazards.get(d_act, 0.0)
                    contrast_margin = direct_haz - min_haz

                    if contrast_margin >= contrast_threshold:
                        ep_interventions += 1
                        # Evaluate 2-step candidate sequences to find best branch
                        sensors = model.pixel_encoder(rgb)
                        cand_seqs = generate_directional_candidate_sequences(2, include_none=False)
                        branch_res = planner_h2.evaluate_candidate_branches(
                            state=out.next_state,
                            sensors=sensors,
                            candidate_sequences=cand_seqs,
                        )
                        # Best non-pruned branch
                        valid_branches = [b for b in branch_res if not b.is_pruned]
                        if valid_branches:
                            best_b = max(valid_branches, key=lambda b: b.cumulative_utility)
                            chosen_act = int(best_b.action_sequence[0])
                        else:
                            chosen_act = min(hazards, key=hazards.get)

                acts.append(chosen_act)
                key_map = {1: (HidKey.W,), 2: (HidKey.A,), 3: (HidKey.S,), 4: (HidKey.D,)}
                ctrl = GenericControl(keys_down=key_map.get(chosen_act, ()))
                outcome = env.step(ctrl)
                obs = outcome.observation
                if "pellet" in outcome.events: ep_p += 1
                if "caught" in outcome.events: ep_c += 1

            world_actions["rel_gated_h2"][w] = acts
            world_catches["rel_gated_h2"] += ep_c
            world_pellets["rel_gated_h2"] += ep_p
            interventions["rel_gated_h2"] += ep_interventions

        # Compute Disagreement Metrics & Trajectory Hashes
        total_steps = sum(len(world_actions["direct"][w]) for w in eval_seeds)
        traj_hashes = {
            c: hashlib.sha256(json.dumps(world_actions[c]).encode()).hexdigest()[:12]
            for c in controllers
        }

        diff_d_h1 = sum(sum(1 for i in range(len(world_actions["direct"][w])) if world_actions["direct"][w][i] != world_actions["rel_gated_h1"][w][i]) for w in eval_seeds)
        diff_d_h2 = sum(sum(1 for i in range(len(world_actions["direct"][w])) if world_actions["direct"][w][i] != world_actions["rel_gated_h2"][w][i]) for w in eval_seeds)
        diff_h1_h2 = sum(sum(1 for i in range(len(world_actions["rel_gated_h1"][w])) if world_actions["rel_gated_h1"][w][i] != world_actions["rel_gated_h2"][w][i]) for w in eval_seeds)

        print(f"\n--- SEED {seed} RELATIVE GATING RESULTS ---")
        print(f"Catches      : Direct={world_catches['direct']} | Gated_H1={world_catches['rel_gated_h1']} | Gated_H2={world_catches['rel_gated_h2']}")
        print(f"Interventions: Gated_H1={interventions['rel_gated_h1']}/{total_steps} ({(interventions['rel_gated_h1']/total_steps)*100:.1f}%) | Gated_H2={interventions['rel_gated_h2']}/{total_steps} ({(interventions['rel_gated_h2']/total_steps)*100:.1f}%)")
        print(f"Hashes       : Direct={traj_hashes['direct']} | Gated_H1={traj_hashes['rel_gated_h1']} | Gated_H2={traj_hashes['rel_gated_h2']}")
        print(f"Disagreement : Direct vs H1={(diff_d_h1/total_steps)*100:.1f}% | Direct vs H2={(diff_d_h2/total_steps)*100:.1f}% | H1 vs H2={(diff_h1_h2/total_steps)*100:.1f}%")

        summary_results[seed] = {
            "catches": world_catches,
            "pellets": world_pellets,
            "interventions": interventions,
            "trajectory_hashes": traj_hashes,
            "disagreements": {
                "direct_vs_h1_pct": round((diff_d_h1 / total_steps) * 100.0, 2),
                "direct_vs_h2_pct": round((diff_d_h2 / total_steps) * 100.0, 2),
                "h1_vs_h2_pct": round((diff_h1_h2 / total_steps) * 100.0, 2),
            },
        }

    out_path = pathlib.Path("docs/runs/2026-08-19-relative-gating-battery.json")
    with open(out_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nSaved relative gating battery results to {out_path}")


if __name__ == "__main__":
    run_relative_gating_battery()
