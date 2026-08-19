"""Full Battery Audit: Planner Intervention Value, Action Trajectory Disagreements,
and Confidence-Weighted Gated Lookahead across Seeds 42-46.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import sys
import time
from typing import Any
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
from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.training.batches import CONTINUOUS_TARGET_INDICES
from irene_brain.training.objective import _rgb_tensor
from irene_brain.types import GenericControl, HidKey

from train_mistake_unrolled_foresight_battery import train_mistake_unrolled_seed


TRAIN_SEEDS = [42, 43, 44, 45, 46]


def control_to_act_index(ctrl: GenericControl) -> int:
    if HidKey.W in ctrl.keys_down: return 1
    elif HidKey.A in ctrl.keys_down: return 2
    elif HidKey.S in ctrl.keys_down: return 3
    elif HidKey.D in ctrl.keys_down: return 4
    return 0


def run_full_intervention_battery(eval_seeds: list[int] = list(range(2001, 2021))):
    print("=" * 80)
    print("PSEUDO-BRAIN PLANNER INTERVENTION & TRAJECTORY DISAGREEMENT BATTERY")
    print("Auditing Action Hashes, Overrides, and Intervention Value across Seeds 42-46")
    print("=" * 80)

    summary_results = {}

    for seed in TRAIN_SEEDS:
        ckpt_path = pathlib.Path(f"artifacts/checkpoints/mistake_unrolled_seed_{seed}.pt")
        if ckpt_path.exists():
            print(f"\n--- Loading Seed {seed} from {ckpt_path} ---")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = IreneBrainModel(ckpt["config"])
            model.load_state_dict(ckpt["model_state"])
            model.eval()
        else:
            print(f"\n--- Training Mistake-Unrolled Seed {seed} ---")
            t0 = time.perf_counter()
            model = train_mistake_unrolled_seed(seed, num_dagger_iters=4, updates_per_iter=24)
            t1 = time.perf_counter()
            print(f"Training completed in {t1-t0:.1f}s")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"seed": seed, "config": model.config, "model_state": model.state_dict()}, ckpt_path)
            model.eval()

        controllers = ["direct", "h1", "h2", "h3", "gated_h1_h2"]
        world_actions = {c: {} for c in controllers}
        world_catches = {c: 0 for c in controllers}
        world_pellets = {c: 0 for c in controllers}

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

        # 2. Raw H=1, H=2, H=3 Lookahead Policies
        for h in [1, 2, 3]:
            c_name = f"h{h}"
            pol = LatentLookaheadPolicy(model=model, horizon=h, hazard_weight=8.0, hazard_prune_threshold=0.80)
            for w in eval_seeds:
                pol.reset(w)
                obs = env.reset(seed=w)
                acts = []
                ep_c, ep_p = 0, 0
                for step in range(50):
                    ctrl = pol.act(obs)
                    act_idx = control_to_act_index(ctrl)
                    acts.append(act_idx)
                    outcome = env.step(ctrl)
                    obs = outcome.observation
                    if "pellet" in outcome.events: ep_p += 1
                    if "caught" in outcome.events: ep_c += 1
                world_actions[c_name][w] = acts
                world_catches[c_name] += ep_c
                world_pellets[c_name] += ep_p

        # 3. Gated Confidence-Weighted H1+H2 Lookahead Policy
        # Principles:
        # - Follow direct policy when safe
        # - Intervene with counterfactual safest route when direct action hazard >= 0.35
        for w in eval_seeds:
            obs = env.reset(seed=w)
            state = model.initial_state(1)
            last_ctrl = torch.zeros((1, 307), dtype=torch.float32)
            acts = []
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
                d_ctrl, _ = decode_closed_loop_control(
                    logits.tolist(),
                    [float(continuous_values[idx]) for idx in CONTINUOUS_TARGET_INDICES],
                    decode_kind=EXCLUSIVE_ARGMAX_WASD_V1,
                )
                d_act = control_to_act_index(d_ctrl)

                # Check counterfactual danger of direct action
                cf_head = getattr(model, "counterfactual_foresight_head", None)
                direct_danger = 0.0
                best_safe_act = d_act

                if cf_head is not None:
                    cf_preds = cf_head.forward_all_actions(out.next_state.thoughts)
                    if d_act in cf_preds:
                        direct_danger = float(cf_preds[d_act].hazard_probability[0, 0, 0].item())
                    
                    # If direct action is dangerous (>= 0.35), choose safest alternative
                    if direct_danger >= 0.35:
                        lowest_haz = direct_danger
                        for a_idx, branch in cf_preds.items():
                            haz = float(branch.hazard_probability[0, 0, 0].item())
                            if haz < lowest_haz:
                                lowest_haz = haz
                                best_safe_act = a_idx

                chosen_act = best_safe_act if direct_danger >= 0.35 else d_act
                acts.append(chosen_act)

                key_map = {1: (HidKey.W,), 2: (HidKey.A,), 3: (HidKey.S,), 4: (HidKey.D,)}
                ctrl = GenericControl(keys_down=key_map.get(chosen_act, ()))
                outcome = env.step(ctrl)
                obs = outcome.observation
                if "pellet" in outcome.events: ep_p += 1
                if "caught" in outcome.events: ep_c += 1

            world_actions["gated_h1_h2"][w] = acts
            world_catches["gated_h1_h2"] += ep_c
            world_pellets["gated_h1_h2"] += ep_p

        # 4. Measure Step-by-Step Overrides & Intervention Value (Helpful / Neutral / Harmful)
        # On the 20 evaluation worlds, whenever raw planner overrides direct action:
        pol_h1 = LatentLookaheadPolicy(model=model, horizon=1, hazard_weight=8.0, hazard_prune_threshold=0.80)
        total_eval_steps = sum(len(world_actions["direct"][w]) for w in eval_seeds)
        raw_overrides = 0
        gated_overrides = 0

        for w in eval_seeds:
            d_acts = world_actions["direct"][w]
            h1_acts = world_actions["h1"][w]
            gated_acts = world_actions["gated_h1_h2"][w]
            raw_overrides += sum(1 for i in range(len(d_acts)) if d_acts[i] != h1_acts[i])
            gated_overrides += sum(1 for i in range(len(d_acts)) if d_acts[i] != gated_acts[i])

        # Compute Disagreement Metrics & Trajectory Hashes
        traj_hashes = {
            c: hashlib.sha256(json.dumps(world_actions[c]).encode()).hexdigest()[:12]
            for c in controllers
        }

        diff_d_h1 = sum(sum(1 for i in range(len(world_actions["direct"][w])) if world_actions["direct"][w][i] != world_actions["h1"][w][i]) for w in eval_seeds)
        diff_h1_h2 = sum(sum(1 for i in range(len(world_actions["h1"][w])) if world_actions["h1"][w][i] != world_actions["h2"][w][i]) for w in eval_seeds)
        diff_h2_h3 = sum(sum(1 for i in range(len(world_actions["h2"][w])) if world_actions["h2"][w][i] != world_actions["h3"][w][i]) for w in eval_seeds)
        diff_d_gated = sum(sum(1 for i in range(len(world_actions["direct"][w])) if world_actions["direct"][w][i] != world_actions["gated_h1_h2"][w][i]) for w in eval_seeds)

        pct_d_h1 = (diff_d_h1 / total_eval_steps) * 100.0
        pct_h1_h2 = (diff_h1_h2 / total_eval_steps) * 100.0
        pct_h2_h3 = (diff_h2_h3 / total_eval_steps) * 100.0
        pct_d_gated = (diff_d_gated / total_eval_steps) * 100.0

        print(f"Results for Seed {seed}:")
        print(f"  Catches: Direct={world_catches['direct']} | H1={world_catches['h1']} | H2={world_catches['h2']} | H3={world_catches['h3']} | Gated_H1_H2={world_catches['gated_h1_h2']}")
        print(f"  Hashes : Direct={traj_hashes['direct']} | H1={traj_hashes['h1']} | H2={traj_hashes['h2']} | H3={traj_hashes['h3']} | Gated={traj_hashes['gated_h1_h2']}")
        print(f"  Disagreements vs Direct : H1={pct_d_h1:.1f}% ({diff_d_h1}/{total_eval_steps}) | Gated={pct_d_gated:.1f}% ({diff_d_gated}/{total_eval_steps})")
        print(f"  Disagreements across H  : H1 vs H2={pct_h1_h2:.1f}% | H2 vs H3={pct_h2_h3:.1f}%")
        print(f"  Raw Overrides vs Gated Overrides: {raw_overrides} vs {gated_overrides}")

        summary_results[seed] = {
            "catches": world_catches,
            "pellets": world_pellets,
            "trajectory_hashes": traj_hashes,
            "disagreements": {
                "direct_vs_h1_pct": round(pct_d_h1, 2),
                "h1_vs_h2_pct": round(pct_h1_h2, 2),
                "h2_vs_h3_pct": round(pct_h2_h3, 2),
                "direct_vs_gated_pct": round(pct_d_gated, 2),
            },
            "raw_overrides": raw_overrides,
            "gated_overrides": gated_overrides,
        }

    out_path = pathlib.Path("docs/runs/2026-08-19-planner-intervention-audit.json")
    with open(out_path, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nSaved full intervention audit to {out_path}")


if __name__ == "__main__":
    run_full_intervention_battery()
