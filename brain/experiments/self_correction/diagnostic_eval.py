"""Phase A & F: Diagnostic Evaluation of Policy Robustness, Recovery & Fast Plasticity.

Deliberately injects mistakes into trained policies to evaluate whether they can
recover or whether they collapse into fatal deadlocks.

Records:
- recovery_rate: % of perturbed episodes reaching the target
- mean_recovery_steps: average ticks to regain course after perturbation
- failure_rate_after_perturbation: % of perturbed runs ending in timeout/deadlock
- goal_success_after_perturbation: target collection rate under perturbations vs clean
- mean_plasticity_norm: average ||P_t|| norm during closed-loop closed evaluation
- max_plasticity_norm: peak ||P_t|| norm observed during evaluation
- mean_surprise: average prediction error ||z_hat_t - z_t|| experienced
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.environments.keys_doors import KeysDoorsEnv
from memory_benchmark.expert import KeysDoorsExpert, control_for_action
from memory_benchmark.models import make_model, N_FRAMES
from self_correction.perturbations import (
    PerturbationEngine,
    get_goal_distance,
    SingleStepPerturbation,
    BurstPerturbation,
    SlipPerturbation,
)
from self_correction.models import (
    PredictiveGRUModel,
    PredictiveThoughtletModel,
)


def run_diagnostic_episode(
    model: nn.Module,
    env: KeysDoorsEnv,
    seed: int,
    perturb_mode: str = "single",  # "none", "single", "burst", "slip"
    max_ticks: int = 300,
    device: torch.device = torch.device("cpu"),
    record_trace: bool = False,
    use_plasticity: bool = True,
) -> Dict:
    obs = env.reset(seed)
    expert = KeysDoorsExpert(env)
    model.eval()

    if hasattr(model, "use_plasticity"):
        model.use_plasticity = use_plasticity

    initial_pixels = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
    frame_buf = [initial_pixels.copy() for _ in range(N_FRAMES)]

    prev_action = 0
    recurrent_state = None
    P_t = None
    predicted_next_z = None
    surprise = torch.zeros(1, 1, device=device)

    perturber = PerturbationEngine(mode=perturb_mode, inject_tick=25, seed=seed)

    trace = []
    success = False
    key_collected = False
    door_opened = False

    stuck_counter = 0
    recent_positions = []

    p_norms: List[float] = []
    delta_p_norms: List[float] = []
    surprises: List[float] = []

    is_predictive = hasattr(model, "encode_observation") and hasattr(model, "forward_step")

    for tick in range(max_ticks):
        raw_frames = np.stack(frame_buf[-N_FRAMES:]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(raw_frames).unsqueeze(0).to(device)
        prev_act_tensor = torch.tensor([prev_action], dtype=torch.long, device=device)

        with torch.no_grad():
            if is_predictive:
                z_t = model.encode_observation(frames_tensor)

                # Compute surprise from previous step foresight error ||z_hat_t - z_t||
                if predicted_next_z is not None:
                    if hasattr(model, "compute_surprise"):
                        surprise = model.compute_surprise(predicted_next_z, z_t)
                    else:
                        surprise = torch.norm(predicted_next_z - z_t, dim=-1, keepdim=True)
                    surprises.append(float(surprise.item()))
                else:
                    surprise = torch.zeros(1, 1, device=device)

                is_thoughtlet = hasattr(model, "brain_cell") or hasattr(model, "K")
                if is_thoughtlet:
                    step_res = model.forward_step(
                        z_t=z_t,
                        prev_action=prev_act_tensor,
                        surprise_t=surprise,
                        thoughts=recurrent_state,
                        P_t=P_t,
                        return_delta=True,
                    )
                else:
                    step_res = model.forward_step(
                        z_t=z_t,
                        prev_action=prev_act_tensor,
                        surprise_t=surprise,
                        h=recurrent_state,
                        P_t=P_t,
                        return_delta=True,
                    )

                logits = step_res[0]
                recurrent_state = step_res[1]
                e_t = step_res[2]
                P_t = step_res[3]
                delta_P = step_res[4] if len(step_res) > 4 else None

                proposed_action = int(logits[0].argmax().item())

                # Predict next latent state foresight
                predicted_next_z = model.predict_next_latent(
                    recurrent_state, torch.tensor([proposed_action], device=device)
                )

                # Track plasticity norms
                if P_t is not None:
                    p_norms.append(float(torch.norm(P_t).item()))
                if delta_P is not None:
                    delta_p_norms.append(float(torch.norm(delta_P).item()))

            else:
                logits, recurrent_state = model(frames_tensor, prev_act_tensor, recurrent_state)
                proposed_action = int(logits[0].argmax().item())
                delta_P = None

        expert_action = expert.get_action()  # For analysis only

        # Perturbation injection
        if perturber.should_inject(tick):
            applied_action = perturber.get_perturbed_action(proposed_action)
            is_perturbed = True
        else:
            # Anti-stuck heuristic if hovering in place for > 4 steps
            if stuck_counter >= 4:
                top2 = torch.topk(logits[0], k=2).indices.cpu().numpy()
                applied_action = int(top2[1])
                stuck_counter = 0
            else:
                applied_action = proposed_action
            is_perturbed = False

        ctrl = control_for_action(applied_action)
        step_out = env.step(ctrl)

        curr_pos = (env._player_x, env._player_y)
        if recent_positions and curr_pos == recent_positions[-1] and applied_action != 0:
            stuck_counter += 1
        else:
            stuck_counter = 0
        recent_positions.append(curr_pos)

        perturber.update_recovery_state(env, tick)

        if not key_collected and (env._has_key or "key_collected" in step_out.events):
            key_collected = True
        if not door_opened and (env._door_open or "door_opened" in step_out.events):
            door_opened = True
        if "target_collected" in step_out.events or step_out.reward > 0.0:
            success = True

        if record_trace:
            trace.append({
                "tick": tick,
                "proposed_action": proposed_action,
                "applied_action": applied_action,
                "expert_action": expert_action,
                "is_perturbed": is_perturbed,
                "recovered": perturber.recovered,
                "distance_to_goal": get_goal_distance(env),
                "has_key": env._has_key,
                "door_open": env._door_open,
                "p_norm": float(torch.norm(P_t).item()) if P_t is not None else 0.0,
                "delta_p_norm": float(torch.norm(delta_P).item()) if delta_P is not None else 0.0,
                "surprise": float(surprise.item()) if surprise is not None else 0.0,
            })

        if success:
            break

        new_frame = np.frombuffer(step_out.observation.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
        frame_buf.append(new_frame)
        if len(frame_buf) > N_FRAMES:
            frame_buf.pop(0)

        prev_action = applied_action

    rec_steps = None
    if perturber.injected and perturber.recovered:
        end_inj = perturber.injection_end_tick or perturber.injection_start_tick
        rec_steps = max(0, (perturber.recovery_tick or tick) - end_inj)

    return {
        "seed": seed,
        "mode": perturb_mode,
        "success": success,
        "key_collected": key_collected,
        "door_opened": door_opened,
        "perturbed": perturber.injected,
        "recovered": perturber.recovered,
        "recovery_steps": rec_steps,
        "total_ticks": tick + 1,
        "mean_p_norm": float(np.mean(p_norms)) if p_norms else 0.0,
        "max_p_norm": float(np.max(p_norms)) if p_norms else 0.0,
        "mean_delta_p_norm": float(np.mean(delta_p_norms)) if delta_p_norms else 0.0,
        "mean_surprise": float(np.mean(surprises)) if surprises else 0.0,
        "trace": trace if record_trace else None,
    }


def run_diagnostic_suite(
    model: nn.Module,
    seeds: List[int],
    modes: List[str] = ["none", "single", "burst"],
    device: torch.device = torch.device("cpu"),
    use_plasticity: bool = True,
) -> Dict[str, Dict]:
    env = KeysDoorsEnv()
    suite_results = {}

    for mode in modes:
        mode_episodes = []
        for s in seeds:
            res = run_diagnostic_episode(
                model, env, seed=s, perturb_mode=mode, device=device, use_plasticity=use_plasticity
            )
            mode_episodes.append(res)

        n = len(mode_episodes)
        succ = sum(1 for e in mode_episodes if e["success"])
        key_cnt = sum(1 for e in mode_episodes if e["key_collected"])
        door_cnt = sum(1 for e in mode_episodes if e["door_opened"])
        pert_cnt = sum(1 for e in mode_episodes if e["perturbed"])
        rec_cnt = sum(1 for e in mode_episodes if e["recovered"])

        rec_steps = [e["recovery_steps"] for e in mode_episodes if e["recovery_steps"] is not None]
        mean_rec_steps = float(np.mean(rec_steps)) if rec_steps else 0.0

        p_norms = [e["mean_p_norm"] for e in mode_episodes if e["mean_p_norm"] > 0.0]
        mean_p_norm = float(np.mean(p_norms)) if p_norms else 0.0
        max_p_norm = float(np.max([e["max_p_norm"] for e in mode_episodes])) if mode_episodes else 0.0
        dp_norms = [e["mean_delta_p_norm"] for e in mode_episodes if e["mean_delta_p_norm"] > 0.0]
        mean_delta_p_norm = float(np.mean(dp_norms)) if dp_norms else 0.0
        surp_vals = [e["mean_surprise"] for e in mode_episodes if e["mean_surprise"] > 0.0]
        mean_surp = float(np.mean(surp_vals)) if surp_vals else 0.0

        suite_results[mode] = {
            "n_episodes": n,
            "success_rate": succ / n,
            "key_rate": key_cnt / n,
            "door_rate": door_cnt / n,
            "perturbed_count": pert_cnt,
            "recovered_count": rec_cnt,
            "recovery_rate": (rec_cnt / pert_cnt) if pert_cnt > 0 else 1.0,
            "mean_recovery_steps": mean_rec_steps,
            "failure_rate": 1.0 - (succ / n),
            "mean_plasticity_norm": mean_p_norm,
            "max_plasticity_norm": max_p_norm,
            "mean_delta_p_norm": mean_delta_p_norm,
            "mean_surprise": mean_surp,
        }

    return suite_results


def evaluate_perturbation_diagnostics(
    model: nn.Module,
    perturbation: Any = None,
    n_episodes: int = 15,
    start_seed: int = 3000,
    device: torch.device = torch.device("cpu"),
    use_plasticity: bool = True,
) -> Dict:
    """Evaluates perturbation diagnostics for a single perturbation mode."""
    mode = "single"
    if perturbation is not None:
        if isinstance(perturbation, str):
            mode = perturbation
        elif hasattr(perturbation, "mode"):
            mode = perturbation.mode

    seeds = [start_seed + i for i in range(n_episodes)]
    suite = run_diagnostic_suite(
        model=model,
        seeds=seeds,
        modes=[mode],
        device=device,
        use_plasticity=use_plasticity,
    )
    return suite[mode]


def main():
    parser = argparse.ArgumentParser(description="Diagnostic evaluation of policy recovery & plasticity")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--start_seed", type=int, default=3000)
    parser.add_argument("--n_episodes", type=int, default=25)
    parser.add_argument("--modes", type=str, nargs="+", default=["none", "single", "burst"])
    parser.add_argument("--use_plasticity", action="store_true", default=True)
    parser.add_argument("--no_plasticity", action="store_false", dest="use_plasticity")

    args = parser.parse_args()
    data = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_type = data.get("model_type", "thoughtlet")

    if "is_predictive" in data or "latent_proj.weight" in data.get("model_state_dict", {}):
        if "gru" in model_type:
            model = PredictiveGRUModel(use_plasticity=args.use_plasticity)
        else:
            model = PredictiveThoughtletModel(use_plasticity=args.use_plasticity)
    else:
        try:
            model = make_model(model_type)
        except Exception:
            model = PredictiveThoughtletModel(use_plasticity=args.use_plasticity)

    model.load_state_dict(data["model_state_dict"])
    model.eval()

    seeds = [args.start_seed + i for i in range(args.n_episodes)]
    print(f"Running Diagnostic Suite for {model_type} (plasticity={args.use_plasticity}) across modes: {args.modes}")
    results = run_diagnostic_suite(model, seeds, modes=args.modes, use_plasticity=args.use_plasticity)

    print("\n--- Diagnostic Results ---")
    for mode, metrics in results.items():
        print(f"Mode [{mode.upper()}]:")
        print(f"  Goal Success Rate    : {metrics['success_rate']*100:.1f}%")
        print(f"  Recovery Rate        : {metrics['recovery_rate']*100:.1f}%")
        print(f"  Mean Recovery Steps  : {metrics['mean_recovery_steps']:.1f}")
        print(f"  Failure Rate         : {metrics['failure_rate']*100:.1f}%")
        print(f"  Mean Plasticity Norm : {metrics.get('mean_plasticity_norm', 0.0):.4f}")
        print(f"  Max Plasticity Norm  : {metrics.get('max_plasticity_norm', 0.0):.4f}")
        print(f"  Mean Surprise        : {metrics.get('mean_surprise', 0.0):.4f}")


if __name__ == "__main__":
    main()
