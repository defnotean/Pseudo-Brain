"""Closed-loop evaluation for GRU vs Thoughtlet models.

Runs each trained model autonomously in MazeChaseEnv across all 5 families.
"""
from __future__ import annotations

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

BRAIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BRAIN_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import make_model, N_FRAMES

from train import get_device

from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.environments.pacman_harness import control_for_class, FAMILIES


def _registered_family_configs():
    """Frozen registered mechanics (pacman_harness.FAMILIES) — same distribution
    the corpus expert generated. Catch = respawn, episodes run to clear/cap."""
    return {fam.name: fam.env_kwargs() for fam in FAMILIES}


FAMILY_CONFIGS = _registered_family_configs()


def _rgb_to_np(rgb_frame) -> np.ndarray:
    """Convert RgbFrame to (H, W, 3) float32 [0,1]."""
    arr = np.frombuffer(rgb_frame.pixels, dtype=np.uint8).reshape(
        rgb_frame.height, rgb_frame.width, 3
    )
    return arr.astype(np.float32) / 255.0


@torch.no_grad()
def run_episode(
    model,
    env: MazeChaseEnv,
    seed: int,
    device: torch.device,
    model_type: str = "reactive",
    max_ticks: int = 2000,
    temperature: float = 0.0,
    anti_stuck: int = 0,
) -> Dict:
    """anti_stuck=N: if the player frame is unchanged for N consecutive ticks while
    repeating the same action (wall deadlock — never occurs in expert corpus), ban
    that action for one step and take the 2nd-best. 0 = disabled."""
    obs = env.reset(seed=seed)

    # Initialize frame buffer — repeated first frame (matches training padding for t < N_FRAMES-1)
    first_frame = _rgb_to_np(obs.rgb)
    frames_buf = [first_frame] * N_FRAMES

    prev_action = 0
    h = None
    total_reward = 0.0
    pellets_eaten = 0
    catches = 0
    catch_ticks = []
    pellet_ticks = []
    nan_ticks = 0
    terminated = False
    action_counts = [0] * 5
    latencies = []
    stuck_count = 0
    last_frame = first_frame
    recoveries = 0
    banned_until = {}  # action -> tick index when ban expires

    for tick in range(max_ticks):
        # Build frame tensor: [1, N_FRAMES, 3, 16, 16]
        frames_np = np.stack(frames_buf)  # (N_FRAMES, 16, 16, 3)
        frames_t = torch.from_numpy(frames_np.transpose(0, 3, 1, 2)).unsqueeze(0).to(device)

        prev_action_t = torch.tensor([prev_action], dtype=torch.long, device=device)

        t0 = time.perf_counter()
        if model_type == "reactive":
            logits, h = model(frames_t, prev_action_t, None)
        elif model_type == "gru":
            if h is None and hasattr(model, "init_hidden"):
                h = model.init_hidden(1, device)
            logits, h = model(frames_t, prev_action_t, h)
        else:
            if h is None and hasattr(model, "init_thoughts"):
                h = model.init_thoughts(1, device)
            logits, h = model(frames_t, prev_action_t, h)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        if temperature > 0:
            if not torch.isfinite(logits).all():
                nan_ticks += 1
                logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            probs = torch.softmax(logits / temperature, dim=-1).squeeze()
            s = probs.sum()
            if not torch.isfinite(s) or s.item() <= 0:
                probs = torch.ones_like(probs) / probs.numel()
            else:
                probs = probs / s
            action = torch.multinomial(probs, 1).item()
        else:
            order = torch.argsort(logits, dim=-1, descending=True).squeeze().tolist()
            if isinstance(order, int):
                order = [order]
            if anti_stuck > 0:
                # Drop expired bans, pick best non-banned action
                for a in list(banned_until.keys()):
                    if banned_until[a] <= tick:
                        del banned_until[a]
                action = next((a for a in order if a not in banned_until), order[0])
                if stuck_count >= anti_stuck:
                    # Wall deadlock: ban the stuck action for a sustained window
                    # so the agent turns away instead of flipping back next tick
                    if action not in banned_until:
                        banned_until[action] = tick + anti_stuck * 2
                        recoveries += 1
                    stuck_count = 0
                    action = next((a for a in order if a not in banned_until), order[0])
            else:
                action = order[0]
        action_counts[action] += 1

        control = control_for_class(action)
        step_out = env.step(control)

        total_reward += step_out.reward
        if "pellet_eaten" in step_out.events:
            pellets_eaten += 1
            pellet_ticks.append(tick)
        repeated = (action == prev_action)
        prev_action = action

        # Deadlock detection: player not moving while repeating same action.
        # Threshold (not exact equality): ghost flicker changes ~1.5 px-sum per
        # tick, real player movement changes far more. Measured: stuck = 0.0
        # with 1.51 ghost flicker every 4th tick.
        new_frame = _rgb_to_np(step_out.observation.rgb)
        if anti_stuck > 0:
            frame_diff = float(np.abs(new_frame - last_frame).sum())
            if repeated and frame_diff < 5.0:
                stuck_count += 1
            else:
                stuck_count = 0
            last_frame = new_frame

        # Update frame buffer
        frames_buf.append(_rgb_to_np(step_out.observation.rgb))
        if len(frames_buf) > N_FRAMES:
            frames_buf = frames_buf[-N_FRAMES:]

        if "caught" in step_out.events:
            # Corpus semantics: catch = respawn, episode continues to clear/cap
            catches += 1
            catch_ticks.append(tick)
        if "cleared" in step_out.events:
            terminated = True
        if terminated:
            break

    total_pellets = env._pellets_remaining + pellets_eaten if hasattr(env, '_pellets_remaining') else pellets_eaten

    # Recovery after disruption: mean pellets eaten within 200 ticks after each catch.
    # High = policy re-orients and resumes foraging post-respawn; 0 with catches = lost.
    post_catch = []
    for ct in catch_ticks:
        post_catch.append(sum(1 for pt in pellet_ticks if ct < pt <= ct + 200))
    mean_recovery = float(np.mean(post_catch)) if post_catch else 0.0

    return {
        "total_reward": float(total_reward),
        "total_steps": tick + 1,
        "nan_ticks": nan_ticks,
        "recoveries": recoveries,
        "catches": catches,
        "survived": catches == 0,
        "post_catch_pellets_200": mean_recovery,
        "pellets_eaten": pellets_eaten,
        "total_pellets": total_pellets,
        "action_counts": action_counts,
        "action_entropy": _entropy(action_counts),
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
    }


def _entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * np.log2(p) for p in probs if p > 0)


def evaluate_model(model_path, model_type, seed, n_seeds_per_family=4, device="auto", temperature=0.0, anti_stuck=0,
                   model_kwargs=None):
    dev = get_device(device)
    if temperature > 0:
        torch.manual_seed(seed)  # reproducible sampling per model-seed
    # Load on CPU then move to target device — direct map_location to DML (privateuseone) is not supported by torch.load
    # weights_only=False: trusted source (our own checkpoints); DML-saved tensors need full unpickler
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = make_model(model_type, **(model_kwargs or {})).to(dev)
    # strict=False: BC checkpoints predate value_head (zero-init, policy-identical)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    results = {}
    for family_name, config in FAMILY_CONFIGS.items():
        family_results = []
        for ep_seed in range(n_seeds_per_family):
            env = MazeChaseEnv(**config)
            ep_result = run_episode(
                model=model, env=env, seed=seed * 1000 + ep_seed,
                device=dev, model_type=model_type, max_ticks=config["max_ticks"],
                temperature=temperature, anti_stuck=anti_stuck,
            )
            family_results.append(ep_result)

        pellet_fractions = [r["pellets_eaten"] / max(r["total_pellets"], 1) for r in family_results]
        tot_pel = sum(r["pellets_eaten"] for r in family_results)
        tot_ticks = sum(r["total_steps"] for r in family_results)
        results[family_name] = {
            "pellet_fraction_mean": float(np.mean(pellet_fractions)),
            "pellet_fraction_std": float(np.std(pellet_fractions)),
            "pellets_per_1k": float(tot_pel / max(tot_ticks, 1) * 1000),
            "catches_per_1k": float(sum(r["catches"] for r in family_results) / max(tot_ticks, 1) * 1000),
            "survival_rate": float(np.mean([1 if r["survived"] else 0 for r in family_results])),
            "mean_catches": float(np.mean([r["catches"] for r in family_results])),
            "mean_recovery": float(np.mean([r["post_catch_pellets_200"] for r in family_results])),
            "mean_recoveries": float(np.mean([r["recoveries"] for r in family_results])),
            "mean_nan_ticks": float(np.mean([r["nan_ticks"] for r in family_results])),
            "mean_reward": float(np.mean([r["total_reward"] for r in family_results])),
            "mean_steps": float(np.mean([r["total_steps"] for r in family_results])),
            "action_entropy": float(np.mean([r["action_entropy"] for r in family_results])),
            "mean_latency_ms": float(np.mean([r["mean_latency_ms"] for r in family_results])),
            "p99_latency_ms": float(np.mean([r["p99_latency_ms"] for r in family_results])),
        }

    all_pf = [results[f]["pellet_fraction_mean"] for f in FAMILY_CONFIGS]
    results["overall"] = {
        "mean_pellet_fraction": float(np.mean(all_pf)),
        "mean_latency_ms": float(np.mean([results[f]["mean_latency_ms"] for f in FAMILY_CONFIGS])),
        "mean_survival": float(np.mean([results[f]["survival_rate"] for f in FAMILY_CONFIGS])),
        "mean_catches": float(np.mean([results[f]["mean_catches"] for f in FAMILY_CONFIGS])),
        "mean_pellets_per_1k": float(np.mean([results[f]["pellets_per_1k"] for f in FAMILY_CONFIGS])),
        "mean_catches_per_1k": float(np.mean([results[f]["catches_per_1k"] for f in FAMILY_CONFIGS])),
        "mean_recovery": float(np.mean([results[f]["mean_recovery"] for f in FAMILY_CONFIGS])),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="GRU vs Thoughtlet closed-loop evaluation")
    parser.add_argument("--run-dir", default=str(BRAIN_ROOT / "runs" / "gru_vs_thoughtlet"))
    parser.add_argument("--models", nargs="+", default=["reactive", "gru", "thoughtlet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 142, 242, 342])
    parser.add_argument("--episodes-per-family", type=int, default=4)
    parser.add_argument("--device", default="auto", help="cpu, cuda, dml, or auto")
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature; 0 = argmax")
    parser.add_argument("--anti-stuck", type=int, default=0, help="wall-deadlock recovery window (ticks); 0 = disabled")
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--thought-size", type=int, default=None)
    parser.add_argument("--out", default="eval_results.json", help="output filename in run dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    all_results = {}

    for model_type in args.models:
        for seed in args.seeds:
            model_path = run_dir / f"seed_{seed}_{model_type}.pt"
            if not model_path.exists():
                print(f"  SKIP {model_type} seed={seed}: no checkpoint")
                continue
            print(f"  Evaluating {model_type} seed={seed}...")
            mk = {}
            if args.cycles is not None:
                mk["cycles"] = args.cycles
            if args.K is not None:
                mk["K"] = args.K
            if args.thought_size is not None:
                mk["thought_size"] = args.thought_size
            results = evaluate_model(
                str(model_path), model_type, seed,
                args.episodes_per_family, args.device,
                temperature=args.temperature, anti_stuck=args.anti_stuck,
                model_kwargs=mk,
            )
            key = f"{model_type}_seed{seed}"
            all_results[key] = results
            o = results["overall"]
            print(f"    pellet_frac={o['mean_pellet_fraction']:.3f}  survival={o['mean_survival']:.2f}  latency={o['mean_latency_ms']:.2f}ms")

    eval_path = run_dir / args.out
    with open(eval_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"{'Model':12s} {'Seed':>5s} {'Pellet':>8s} {'Surv':>6s} {'Latency':>8s}")
    print(f"{'-'*70}")
    for key in sorted(all_results.keys()):
        if key == "overall":
            continue
        r = all_results[key]
        model_type, seed_str = key.rsplit("_seed", 1)
        print(f"{model_type:12s} {seed_str:>5s} {r['overall']['mean_pellet_fraction']:8.3f} {r['overall']['mean_survival']:6.2f} {r['overall']['mean_latency_ms']:8.2f}")

    print(f"{'-'*70}")
    for model_type in args.models:
        mr = {k: v for k, v in all_results.items() if k.startswith(model_type)}
        if mr:
            avg_pf = np.mean([v["overall"]["mean_pellet_fraction"] for v in mr.values()])
            avg_surv = np.mean([v["overall"]["mean_survival"] for v in mr.values()])
            avg_lat = np.mean([v["overall"]["mean_latency_ms"] for v in mr.values()])
            print(f"{model_type+' avg':12s} {'':>5s} {avg_pf:8.3f} {avg_surv:6.2f} {avg_lat:8.2f}")

    print(f"\nResults saved to {eval_path}")


if __name__ == "__main__":
    main()
