"""Cognitive Veto Parameter Sweep: Pareto Frontier of Precision vs Coverage.

Sweeps over:
- entropy_threshold in [0.05, 0.15, 0.25]
- contrast_threshold in [0.003, 0.005, 0.008]
- drift_threshold in [0.02, 0.05, 0.10]
Evaluates on Seeds 44 and 46 across 20 worlds (1,000 steps).
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
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.training.batches import CONTINUOUS_TARGET_INDICES
from irene_brain.training.objective import _rgb_tensor
from irene_brain.types import GenericControl, HidKey


def control_to_act_index(ctrl: GenericControl) -> int:
    if HidKey.W in ctrl.keys_down: return 1
    elif HidKey.A in ctrl.keys_down: return 2
    elif HidKey.S in ctrl.keys_down: return 3
    elif HidKey.D in ctrl.keys_down: return 4
    return 0


def act_index_to_control(act_idx: int) -> GenericControl:
    key_map = {1: (HidKey.W,), 2: (HidKey.A,), 3: (HidKey.S,), 4: (HidKey.D,)}
    return GenericControl(keys_down=key_map.get(act_idx, ()))


def sweep_veto_parameters(eval_seeds: list[int] = list(range(2001, 2021))):
    print("=" * 80)
    print("COGNITIVE VETO PARETO FRONTIER SWEEP (Seeds 44 & 46)")
    print("=" * 80)

    models = {}
    for seed in [44, 46]:
        ckpt_path = pathlib.Path(f"artifacts/checkpoints/mistake_unrolled_seed_{seed}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = IreneBrainModel(ckpt["config"], enable_adaptive_cognition=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models[seed] = model

    entropy_grid = [0.05, 0.15, 0.25]
    contrast_grid = [0.003, 0.005, 0.008]
    drift_grid = [0.03, 0.06]

    sweep_records = []

    for ent_th in entropy_grid:
        for cont_th in contrast_grid:
            for drift_th in drift_grid:
                total_int = 0
                total_helpful = 0
                total_harmful = 0
                total_neutral = 0

                env = MazeChaseEnv()

                for seed, model in models.items():
                    cf_head = getattr(model, "counterfactual_foresight_head", None)
                    if cf_head is None:
                        continue

                    for w_seed in eval_seeds:
                        obs = env.reset(seed=w_seed)
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

                            probs = F.softmax(logits, dim=-1).numpy()
                            action_entropy = -float(np.sum(probs * np.log(probs + 1e-8)))

                            cf_preds = cf_head.forward_all_actions(out.next_state.thoughts)
                            hazards_h1 = {a_i: float(cf_preds[a_i].hazard_probability[0, 0, 0].item()) for a_i in range(5)}
                            hazards_h2 = {a_i: float(cf_preds[a_i].hazard_probability[0, min(1, cf_preds[a_i].hazard_probability.shape[1] - 1), 0].item()) for a_i in range(5)}

                            min_haz_h1 = min(hazards_h1.values())
                            direct_haz = hazards_h1.get(d_act, 0.0)
                            contrast_margin = direct_haz - min_haz_h1
                            best_safe_act = min(hazards_h1, key=hazards_h1.get)
                            drift = abs(hazards_h1[best_safe_act] - hazards_h2[best_safe_act])

                            chosen_act = d_act

                            if (
                                best_safe_act != d_act
                                and action_entropy >= ent_th
                                and contrast_margin >= cont_th
                                and drift <= drift_th
                            ):
                                total_int += 1
                                snap = env.snapshot()

                                # Fork A: Direct
                                env.restore(snap)
                                direct_caught = False
                                out_d = env.step(act_index_to_control(d_act))
                                if "caught" in out_d.events: direct_caught = True
                                for _ in range(2):
                                    if out_d.terminated or out_d.truncated: break
                                    out_d = env.step(GenericControl())
                                    if "caught" in out_d.events: direct_caught = True

                                # Fork B: Planner
                                env.restore(snap)
                                planner_caught = False
                                out_p = env.step(act_index_to_control(best_safe_act))
                                if "caught" in out_p.events: planner_caught = True
                                for _ in range(2):
                                    if out_p.terminated or out_p.truncated: break
                                    out_p = env.step(GenericControl())
                                    if "caught" in out_p.events: planner_caught = True

                                env.restore(snap)

                                if direct_caught and not planner_caught:
                                    total_helpful += 1
                                elif planner_caught and not direct_caught:
                                    total_harmful += 1
                                else:
                                    total_neutral += 1

                                chosen_act = best_safe_act

                            outcome = env.step(act_index_to_control(chosen_act))
                            obs = outcome.observation
                            if outcome.terminated or outcome.truncated:
                                break

                active_decisions = total_helpful + total_harmful
                precision = round((total_helpful / active_decisions) * 100.0, 2) if active_decisions > 0 else None
                net_val = total_helpful - total_harmful

                rec = {
                    "entropy_th": ent_th,
                    "contrast_th": cont_th,
                    "drift_th": drift_th,
                    "interventions": total_int,
                    "helpful": total_helpful,
                    "neutral": total_neutral,
                    "harmful": total_harmful,
                    "precision_pct": precision,
                    "net_value": net_val,
                }
                sweep_records.append(rec)
                print(f"Ent={ent_th:0.2f} | Cont={cont_th:0.003f} | Drift={drift_th:0.02f} -> Int={total_int:3d} | Helpful={total_helpful:2d} | Neutral={total_neutral:3d} | Harmful={total_harmful:2d} | Precision={precision if precision is not None else 'N/A'}% | Net={net_val:+d}")

    out_path = pathlib.Path("docs/runs/2026-08-19-veto-pareto-sweep.json")
    with open(out_path, "w") as f:
        json.dump(sweep_records, f, indent=2)
    print(f"\nSaved Pareto sweep to {out_path}")


if __name__ == "__main__":
    sweep_veto_parameters()
