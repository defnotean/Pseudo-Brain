"""Named follow-up: ranked K=8 unmatched-suppress vs matched Proposal-GRU.

Not a rerun of dgx-gate6-matched-gru-v1 (K=32, FAIL). Caps K at 8 (Gate 2 last
non-catastrophic point), uses unmatched-slot suppression + multiplicity Q(a)
+ collapse/register-diversity losses already in MultiHypothesisBranchLoss.

Smoke (default off): seeds 43+45, 60 steps, 3 episodes. Full Gate 6 envelope
only when the operator passes --full.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

for _candidate in (
    os.environ.get("IRENE_BRAIN_SRC", ""),
    "/workspace/repo/brain/src",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"),
):
    if _candidate and os.path.isdir(_candidate):
        sys.path.insert(0, os.path.abspath(_candidate))
        break

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from irene_brain.environments.phase2_suite import (
    Phase2TaskEnvironment,
    TaskFamily,
    make_family_suite,
)
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.thought_mediated_model import (
    ProposalGRUBaseline,
    ThoughtMediatedBrainModel,
)
from irene_brain.training.staged_branch_curriculum import (
    MultiHypothesisBranchLoss,
    generate_multi_step_trajectory_tree,
)
from irene_brain.types import GenericControl, HidKey

PROBE_ID = "ranked-k8-unmatched-suppress-v1"
SEEDS_FULL = [42, 43, 44, 45, 46]
SEEDS_SMOKE = [43, 45]
GATE6_MARGIN = 0.10
KNOCKOUT_SANITY_MARGIN = 0.30
PROBE_K = 8
GRU_HIDDEN = 180
TRAIN_STEPS_FULL = 300
TRAIN_STEPS_SMOKE = 60
EPISODES_FULL = 5
EPISODES_SMOKE = 3
LR = 5e-4
PARAM_MATCH_TOLERANCE = 0.06


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def thought_config(width: int) -> ThoughtFieldConfig:
    return ThoughtFieldConfig(
        thoughtlets=PROBE_K,
        core_width=width,
        attention_heads=4,
        routed_neighbors=2,
        cognitive_cycles=3,
    )


def find_param_matched_k8_width(target_params: int) -> int:
    best_w = 64
    best_diff = float("inf")
    for width in range(16, 320, 4):
        try:
            model = ThoughtMediatedBrainModel(thought_config(width))
        except Exception:
            continue
        diff = abs(count_params(model) - target_params)
        if diff < best_diff:
            best_diff = diff
            best_w = width
    model = ThoughtMediatedBrainModel(thought_config(best_w))
    rel = abs(count_params(model) - target_params) / max(target_params, 1)
    if rel > PARAM_MATCH_TOLERANCE:
        raise RuntimeError(
            f"K=8 width search could not match GRU envelope {target_params}: "
            f"best W={best_w} params={count_params(model)} rel={rel:.3f}"
        )
    return best_w


def compute_iqm(values: list[float]) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    n = len(v)
    q1 = int(n * 0.25)
    q3 = int(n * 0.75)
    if q1 >= q3:
        return float(np.mean(v))
    return float(np.mean(v[q1:q3]))


def relative_advantage(challenger: float, baseline: float) -> float:
    denom = abs(baseline)
    if denom < 1e-8:
        return 0.0 if abs(challenger) < 1e-8 else float("inf")
    return float((challenger - baseline) / denom)


def train_thought_mediated_model(
    model: ThoughtMediatedBrainModel,
    device: torch.device,
    training_steps: int,
    lr: float = LR,
) -> None:
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = MultiHypothesisBranchLoss(inertia_weight=0.5).to(device)
    all_families = list(TaskFamily)
    envs = [Phase2TaskEnvironment(make_family_suite(fam)[0]) for fam in all_families]
    obs_list = [env.reset(1000 + idx) for idx, env in enumerate(envs)]

    for step in range(training_steps):
        rgb_tensors = []
        bundles = []
        for idx, (env, obs) in enumerate(zip(envs, obs_list)):
            raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor.shape[-1] != 32:
                rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")
            rgb_tensors.append(rgb_tensor)
            bundles.append(generate_multi_step_trajectory_tree(env, obs, horizon=2, device=device))

        rgb_batch = torch.cat(rgb_tensors, dim=0)
        ctrl_batch = torch.zeros((len(envs), 307), device=device)
        dt_batch = torch.tensor([0.016667] * len(envs), device=device)
        out = model(rgb_batch, ctrl_batch, dt_batch, max_cycles=model.config.cognitive_cycles)
        total_loss, _metrics = loss_fn(
            out.action.proposals,
            bundles,
            thoughts=out.next_state.thoughts,
        )
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            button_logits = out.action.button_logits.cpu().numpy()
            for idx, (env, obs) in enumerate(zip(envs, obs_list)):
                b_logits = button_logits[idx]
                dir_scores = {
                    "W": float(b_logits[int(HidKey.W)]),
                    "S": float(b_logits[int(HidKey.S)]),
                    "A": float(b_logits[int(HidKey.A)]),
                    "D": float(b_logits[int(HidKey.D)]),
                }
                best_dir, best_score = max(dir_scores.items(), key=lambda x: x[1])
                keys: list[int] = []
                if best_score > 0.0:
                    keys.append(int(getattr(HidKey, best_dir)))
                outcome = env.step(GenericControl(mouse_dx=0.0, mouse_dy=0.0, keys_down=tuple(keys)))
                obs_list[idx] = env.reset(step * 10 + idx) if outcome.terminated or outcome.truncated else outcome.observation


def train_proposal_gru_baseline(
    model: ProposalGRUBaseline,
    device: torch.device,
    training_steps: int,
    lr: float = LR,
) -> None:
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = MultiHypothesisBranchLoss().to(device)
    all_families = list(TaskFamily)
    envs = [Phase2TaskEnvironment(make_family_suite(fam)[0]) for fam in all_families]
    obs_list = [env.reset(2000 + idx) for idx, env in enumerate(envs)]

    for step in range(training_steps):
        rgb_tensors = []
        bundles = []
        for idx, (env, obs) in enumerate(zip(envs, obs_list)):
            raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor.shape[-1] != 32:
                rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")
            rgb_tensors.append(rgb_tensor)
            bundles.append(generate_multi_step_trajectory_tree(env, obs, horizon=2, device=device))

        rgb_batch = torch.cat(rgb_tensors, dim=0)
        ctrl_batch = torch.zeros((len(envs), 307), device=device)
        dt_batch = torch.tensor([0.016667] * len(envs), device=device)
        out = model(rgb_batch, ctrl_batch, dt_batch)
        total_loss, _metrics = loss_fn(out.action.proposals, bundles)
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            button_logits = out.action.button_logits.cpu().numpy()
            for idx, (env, obs) in enumerate(zip(envs, obs_list)):
                b_logits = button_logits[idx]
                dir_scores = {
                    "W": float(b_logits[int(HidKey.W)]),
                    "S": float(b_logits[int(HidKey.S)]),
                    "A": float(b_logits[int(HidKey.A)]),
                    "D": float(b_logits[int(HidKey.D)]),
                }
                best_dir, best_score = max(dir_scores.items(), key=lambda x: x[1])
                keys: list[int] = []
                if best_score > 0.0:
                    keys.append(int(getattr(HidKey, best_dir)))
                outcome = env.step(GenericControl(mouse_dx=0.0, mouse_dy=0.0, keys_down=tuple(keys)))
                obs_list[idx] = env.reset(step * 10 + idx) if outcome.terminated or outcome.truncated else outcome.observation


def evaluate_closed_loop_model(
    model: nn.Module,
    env: Phase2TaskEnvironment,
    *,
    seed: int,
    episodes: int,
    device: torch.device,
    intervention_mode: str = "none",
    collect_histogram: bool = False,
) -> dict[str, Any]:
    returns: list[float] = []
    histogram = {"idle": 0, "W": 0, "A": 0, "S": 0, "D": 0}
    for ep in range(episodes):
        obs = env.reset(seed * 100 + ep)
        state = model.initial_state(1)
        if hasattr(state, "to"):
            state = state.to(device)
        elif torch.is_tensor(state):
            state = state.to(device)
        ep_return = 0.0
        ticks = 0
        while ticks < 256:
            raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
            rgb_tensor = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
            if rgb_tensor.shape[-1] != 32:
                rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")
            ctrl_tensor = torch.zeros((1, 307), device=device)
            dt_tensor = torch.tensor([0.016667], device=device)
            kwargs: dict[str, Any] = {}
            if intervention_mode != "none":
                kwargs["thought_intervention"] = intervention_mode
            with torch.no_grad():
                out = model(rgb_tensor, ctrl_tensor, dt_tensor, state=state, **kwargs)
            state = out.next_state
            button_logits = out.action.button_logits[0].cpu().numpy()
            dir_scores = {
                "W": float(button_logits[int(HidKey.W)]),
                "S": float(button_logits[int(HidKey.S)]),
                "A": float(button_logits[int(HidKey.A)]),
                "D": float(button_logits[int(HidKey.D)]),
            }
            best_dir, best_score = max(dir_scores.items(), key=lambda x: x[1])
            keys: list[int] = []
            if best_score > 0.0:
                keys.append(int(getattr(HidKey, best_dir)))
                if collect_histogram:
                    histogram[best_dir] += 1
            elif collect_histogram:
                histogram["idle"] += 1
            outcome = env.step(GenericControl(mouse_dx=0.0, mouse_dy=0.0, keys_down=tuple(keys)))
            ep_return += outcome.reward
            ticks += 1
            if outcome.terminated or outcome.truncated:
                break
            obs = outcome.observation
        returns.append(ep_return)
    payload: dict[str, Any] = {
        "mean_return": float(np.mean(returns)),
        "iqm_return": compute_iqm(returns),
        "returns": [float(x) for x in returns],
    }
    if collect_histogram:
        payload["action_histogram"] = histogram
    return payload


def evaluate_all_families(
    model: nn.Module,
    seed: int,
    episodes: int,
    device: torch.device,
    intervention_mode: str = "none",
) -> dict[str, Any]:
    per_family: dict[str, Any] = {}
    pooled: list[float] = []
    for family in TaskFamily:
        env = Phase2TaskEnvironment(make_family_suite(family)[0])
        out = evaluate_closed_loop_model(
            model, env, seed=seed, episodes=episodes, device=device, intervention_mode=intervention_mode
        )
        per_family[family.name] = {"iqm_return": out["iqm_return"], "mean_return": out["mean_return"]}
        pooled.extend(out["returns"])
    return {
        "per_family": per_family,
        "pooled_iqm_return": compute_iqm(pooled),
        "pooled_mean_return": float(np.mean(pooled)),
        "pooled_returns": pooled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ranked-k8-unmatched-suppress-v1 vs matched Proposal-GRU")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--smoke", action="store_true", help="seeds 43+45, 60 steps, 3 episodes")
    parser.add_argument("--full", action="store_true", help="Gate 6 envelope: 5 seeds, 300 steps, 5 episodes")
    parser.add_argument("--output-json", type=str, default="ranked_k8_results.json")
    args = parser.parse_args()
    if args.smoke and args.full:
        raise SystemExit("pass only one of --smoke or --full")
    if not args.smoke and not args.full:
        args.smoke = True

    seeds = SEEDS_SMOKE if args.smoke else SEEDS_FULL
    train_steps = TRAIN_STEPS_SMOKE if args.smoke else TRAIN_STEPS_FULL
    episodes = EPISODES_SMOKE if args.smoke else EPISODES_FULL
    mode = "smoke" if args.smoke else "full"

    device = torch.device(args.device)
    print(f"=== {PROBE_ID} ({mode}) vs matched Proposal-GRU ===")
    print(f"device={device} torch={torch.__version__} cuda={torch.cuda.is_available()}")
    print(f"train_steps={train_steps} seeds={seeds} episodes_per_world={episodes} K={PROBE_K}")

    gru_probe = ProposalGRUBaseline(hidden_dim=GRU_HIDDEN)
    gru_params = count_params(gru_probe)
    matched_width = find_param_matched_k8_width(gru_params)
    pb_probe = ThoughtMediatedBrainModel(thought_config(matched_width))
    pb_params = count_params(pb_probe)
    param_delta_pct = 100.0 * (pb_params - gru_params) / max(gru_params, 1)
    print(
        f"params PB K={PROBE_K} W={pb_probe.config.core_width} C={pb_probe.config.cognitive_cycles} "
        f"= {pb_params:,} | GRU H={GRU_HIDDEN} = {gru_params:,} | delta={param_delta_pct:+.2f}%"
    )

    pb_returns: list[float] = []
    gru_returns: list[float] = []
    pb_by_seed: dict[str, Any] = {}
    gru_by_seed: dict[str, Any] = {}
    knockout_reports: dict[str, Any] = {}

    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pb = ThoughtMediatedBrainModel(thought_config(matched_width)).to(device)
        train_thought_mediated_model(pb, device=device, training_steps=train_steps)
        pb_eval = evaluate_all_families(pb, seed=seed, episodes=episodes, device=device)
        pb_by_seed[str(seed)] = pb_eval
        pb_returns.extend(pb_eval["pooled_returns"])
        print(f"  PB  pooled IQM={pb_eval['pooled_iqm_return']:.3f} mean={pb_eval['pooled_mean_return']:.3f}")

        family_b = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_B_PURSUIT_EVASION)[0])
        normal_b = evaluate_closed_loop_model(
            pb, family_b, seed=seed, episodes=episodes, device=device, collect_histogram=True
        )
        knockout_b = evaluate_closed_loop_model(
            pb,
            family_b,
            seed=seed,
            episodes=episodes,
            device=device,
            intervention_mode="zero_knockout",
            collect_histogram=True,
        )
        if abs(normal_b["iqm_return"]) < 1e-8:
            knockout_degradation = 0.0
        else:
            knockout_degradation = float((normal_b["iqm_return"] - knockout_b["iqm_return"]) / abs(normal_b["iqm_return"]))
        knockout_reports[str(seed)] = {
            "family_b_normal_iqm": normal_b["iqm_return"],
            "family_b_knockout_iqm": knockout_b["iqm_return"],
            "knockout_degradation": knockout_degradation,
            "family_b_action_histogram": normal_b.get("action_histogram"),
            "family_b_knockout_histogram": knockout_b.get("action_histogram"),
        }
        print(
            f"  knockout Family B IQM normal={normal_b['iqm_return']:.3f} "
            f"zero={knockout_b['iqm_return']:.3f} degradation={100.0 * knockout_degradation:.1f}% "
            f"hist={normal_b.get('action_histogram')}"
        )

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        gru = ProposalGRUBaseline(hidden_dim=GRU_HIDDEN).to(device)
        train_proposal_gru_baseline(gru, device=device, training_steps=train_steps)
        gru_eval = evaluate_all_families(gru, seed=seed, episodes=episodes, device=device)
        gru_by_seed[str(seed)] = gru_eval
        gru_returns.extend(gru_eval["pooled_returns"])
        print(f"  GRU pooled IQM={gru_eval['pooled_iqm_return']:.3f} mean={gru_eval['pooled_mean_return']:.3f}")
        del pb, gru
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pb_iqm = compute_iqm(pb_returns)
    gru_iqm = compute_iqm(gru_returns)
    pb_mean = float(np.mean(pb_returns))
    gru_mean = float(np.mean(gru_returns))
    adv_iqm = relative_advantage(pb_iqm, gru_iqm)
    adv_mean = relative_advantage(pb_mean, gru_mean)
    knockout_degradations = [float(v["knockout_degradation"]) for v in knockout_reports.values()]
    mean_knockout_degradation = float(np.mean(knockout_degradations))
    knockout_ok = mean_knockout_degradation >= KNOCKOUT_SANITY_MARGIN
    gru_wins_or_ties = adv_iqm <= 0.0
    smoke_ahead = bool(adv_iqm > 0.0)
    if gru_wins_or_ties:
        verdict = "FAIL"
        claim = False
        reason = "Proposal-GRU ties or wins IQM; do not claim thought-mediated architecture superiority"
    elif not knockout_ok:
        verdict = "FAIL"
        claim = False
        reason = "thought knockout did not collapse Family B return by >=30%; mediation sanity failed"
    elif args.smoke:
        verdict = "SMOKE"
        claim = False
        reason = (
            f"smoke IQM advantage {100.0 * adv_iqm:.2f}% (PB ahead={smoke_ahead}). "
            "Not a Gate 6 pass. Launch --full only if PB is ahead."
        )
    elif adv_iqm >= GATE6_MARGIN:
        verdict = "PASS"
        claim = True
        reason = f"IQM advantage {100.0 * adv_iqm:.2f}% >= 10% with knockout sanity"
    else:
        verdict = "FAIL"
        claim = False
        reason = f"IQM advantage {100.0 * adv_iqm:.2f}% is below the preregistered 10% margin"

    results = {
        "probe_id": PROBE_ID,
        "mode": mode,
        "parent_fail_probe_id": "dgx-gate6-matched-gru-v1",
        "registration_id": "PREREG-PHASE2-REVISED-THOUGHT-MEDIATED-V1",
        "gate_id": "Gate 6 follow-up" if not args.smoke else "Gate 6 smoke",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "seeds": seeds,
        "train_steps": train_steps,
        "episodes_per_world": episodes,
        "lr": LR,
        "pair": {
            "thought_mediated": {
                "k": PROBE_K,
                "core_width": int(pb_probe.config.core_width),
                "cognitive_cycles": int(pb_probe.config.cognitive_cycles),
                "routed_neighbors": int(pb_probe.config.routed_neighbors),
                "parameters": pb_params,
            },
            "proposal_gru": {
                "hidden_dim": GRU_HIDDEN,
                "num_proposals": 1,
                "parameters": gru_params,
            },
            "parameter_delta_pct": param_delta_pct,
        },
        "pb_iqm_return": pb_iqm,
        "gru_iqm_return": gru_iqm,
        "pb_mean_return": pb_mean,
        "gru_mean_return": gru_mean,
        "relative_iqm_advantage": adv_iqm,
        "relative_mean_advantage": adv_mean,
        "gate6_margin": GATE6_MARGIN,
        "knockout_sanity_margin": KNOCKOUT_SANITY_MARGIN,
        "mean_knockout_degradation": mean_knockout_degradation,
        "knockout_sanity_passed": knockout_ok,
        "gru_wins_or_ties": gru_wins_or_ties,
        "smoke_pb_ahead": smoke_ahead,
        "gate6_status": verdict,
        "architecture_superiority_claimed": claim,
        "reason": reason,
        "pb_by_seed": pb_by_seed,
        "gru_by_seed": gru_by_seed,
        "knockout_by_seed": knockout_reports,
        "frozen_parent_fail": {
            "pb_iqm": -27.758,
            "gru_iqm": -25.694,
            "relative_pct": -8.04,
        },
    }

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\n=== {PROBE_ID} {mode} verdict ===")
    print(f"PB IQM={pb_iqm:.4f}  GRU IQM={gru_iqm:.4f}  advantage={100.0 * adv_iqm:.2f}%")
    print(f"PB mean={pb_mean:.4f} GRU mean={gru_mean:.4f} advantage={100.0 * adv_mean:.2f}%")
    print(f"knockout degradation={100.0 * mean_knockout_degradation:.1f}% sanity={knockout_ok}")
    print(f"STATUS={verdict} claim_superiority={claim}")
    print(reason)
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
