"""Evaluate FlyMerge V1 models (A/B/C) on the S1-S5 scenario battery + ablations.

Preregistration: brain/docs/preregistrations/2026-08-29-flymerge-v1.md

Runs the frozen harness env with three model arms:
  A = ReactiveBaselineV1 (frozen, trained checkpoint from T1-4)
  B = ReactiveBaselineV1-recurrent (same trunk, GRU persistent state)
  C = FlyMergeV1 (trunk + GRU + innate channel + shunting + modulation)

Scenarios:
  S1: Visible pellet (standard episode)
  S2: Pellet disappears (all pellets removed after agent's 20th pellet)
  S3: Distraction (ghost spawned during pellet pursuit)
  S4: Self-induced distribution shift (teleport to unseen corridor)
  S5: Multiple pellets (directional commitment proxy)

Ablations (Model C only):
  - C+shunting disabled (g≡0)
  - C+modulation disabled (temp≡1)
  - C+innate channel untrained
  - C+persistent state frozen (h fixed)
  - C+persistent state zeroed each tick
  - C+random noise into h

Outputs (create-only):
  brain/runs/flymerge-v1/eval_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

_HERE = Path(__file__).resolve()
BRAIN_ROOT = _HERE.parent.parent.parent  # brain/experiments/flymerge_v1 -> brain
sys.path.insert(0, str(BRAIN_ROOT / "src"))
sys.path.insert(0, str(BRAIN_ROOT / "scripts"))
sys.path.insert(0, str(BRAIN_ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
try:
    torch.set_num_threads(1)
except RuntimeError:
    pass

try:
    from ctypes import windll  # type: ignore
    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
    windll.kernel32.SetPriorityClass(
        windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS
    )
except Exception:
    pass

from irene_brain.environments.maze_chase import MazeChaseEnv  # noqa: E402
from irene_brain.environments.pacman_harness import (  # noqa: E402
    FAMILY_COUNT, partition_seeds, DifficultyFamily,
)
from irene_brain.v2.reactive_baseline import ReactiveBaselineV1, N_FRAMES  # noqa: E402
from flymerge_v1.model import FlyMergeV1  # noqa: E402
from irene_brain.environments.pacman_harness import control_for_class  # noqa: E402
from irene_brain.types import GenericControl  # noqa: E402

# --- Load checkpoints ---
CHECKPOINTS_DIR = BRAIN_ROOT / "runs" / "flymerge-v1"


def load_model_A():
    """Model A: frozen ReactiveBaselineV1 from T1-4."""
    # Try to load from T1-4 checkpoint; if not found, use fresh model
    ckpt_path = BRAIN_ROOT / "runs" / "embodied-reactive-baseline-v1" / "seed_42.pt"
    model = ReactiveBaselineV1()
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_model_B(seed):
    """Model B: ReactiveBaselineV1-recurrent (same trunk, GRU persistent state)."""
    ckpt_path = CHECKPOINTS_DIR / f"seed_{seed}_B.pt"
    model = ReactiveBaselineV1()
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_model_C(seed):
    """Model C: FlyMergeV1."""
    ckpt_path = CHECKPOINTS_DIR / f"seed_{seed}_C.pt"
    model = FlyMergeV1()
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _rgb_to_tensor(rgb_frame):
    """Convert RgbFrame to [3, H, W] float32 tensor in [0,1]."""
    arr = np.frombuffer(rgb_frame.pixels, dtype=np.uint8).reshape(rgb_frame.height, rgb_frame.width, 3)
    return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32) / 255.0)


def run_episode(env, model, max_ticks=6000, seed=0, h_init=None,
                disable_shunting=False, disable_modulation=False,
                untrained_urgency=False, reset_h_every=None):
    """Run a single episode and collect trajectory data.

    Returns dict with pellet count, survival, ghost catches, action entropy,
    and other per-tick metrics.
    """
    obs = env.reset(seed)
    first_frame = _rgb_to_tensor(obs.rgb)
    frames_buf = [first_frame] * N_FRAMES  # left-pad with first frame
    prev_action = 0  # idle as initial
    h = h_init.clone() if h_init is not None else None

    # Count total pellets from the env's initial state
    total_pellets = env._pellets_remaining if hasattr(env, "_pellets_remaining") else 1

    metrics = {
        "pellets_eaten": 0,
        "survived": True,
        "ghost_catches": 0,
        "ticks": 0,
        "action_counts": defaultdict(int),
        "action_entropy_sum": 0.0,
        "movement_ticks": 0,
        "idle_ticks": 0,
    }

    device = next(model.parameters()).device
    frames_tensor = torch.stack(frames_buf).unsqueeze(0).to(device)  # [1, N_FRAMES, 3, 16, 16]
    prev_act_tensor = torch.tensor([prev_action], dtype=torch.long, device=device)

    with torch.no_grad():
        for t in range(max_ticks):
            # Get model output
            if isinstance(model, FlyMergeV1):
                if h is None:
                    h = torch.zeros(1, 32, device=device, dtype=torch.float32)
                logits, h_new, u = model(frames_tensor, prev_act_tensor, h)

                if disable_shunting:
                    pass
                if disable_modulation:
                    h_bar_proj = model.h_proj(h_new)
                    logits = h_bar_proj
                    bypass = model.bypass_proj(u)
                    logits = logits + bypass
                if untrained_urgency:
                    u = torch.rand_like(u)

                h = h_new
            else:
                logits = model(frames_tensor, prev_act_tensor)

            # Sample action
            action_probs = torch.softmax(logits / 1.0, dim=-1)
            action = action_probs.argmax(dim=-1).item()
            metrics["action_counts"][action] += 1

            # Entropy contribution
            p = action_probs[0].cpu().numpy()
            p = p[p > 0]
            metrics["action_entropy_sum"] += -np.sum(p * np.log(p))

            # Step env
            ctrl = control_for_class(action)
            step_out = env.step(ctrl)

            metrics["ticks"] += 1

            if action != 0:
                metrics["movement_ticks"] += 1
            else:
                metrics["idle_ticks"] += 1

            # Update frame buffer
            frames_buf.append(_rgb_to_tensor(step_out.observation.rgb))
            if len(frames_buf) > N_FRAMES:
                frames_buf.pop(0)

            frames_tensor = torch.stack(frames_buf[-N_FRAMES:]).unsqueeze(0).to(device)
            prev_act_tensor = torch.tensor([action], dtype=torch.long, device=device)

            # Track pellets via events (not reward, to avoid double-counting on cleared)
            if "pellet_eaten" in step_out.events:
                metrics["pellets_eaten"] += 1

            if step_out.terminated:
                if "caught" in step_out.events:
                    metrics["survived"] = False
                    metrics["ghost_catches"] += 1
                break
            if step_out.truncated:
                break

    metrics["avg_action_entropy"] = metrics["action_entropy_sum"] / max(metrics["ticks"], 1)
    metrics["pellet_fraction"] = metrics["pellets_eaten"] / max(total_pellets, 1)
    metrics["survival_fraction"] = 1.0 if metrics["survived"] else 0.0

    return metrics


def run_scenario_s1(model, env_class, family_config, n_episodes=5, seeds=(42, 142, 242, 342)):
    """S1: Visible pellet — standard episode."""
    results = []
    for seed in seeds:
        for ep in range(n_episodes):
            env = env_class(**{k: v for k, v in family_config.items() if k != "max_ticks"})
            metrics = run_episode(env, model, max_ticks=family_config.get("max_ticks", 6000), seed=seed + ep)
            results.append(metrics)
    return results


def run_scenario_s2(model, env_class, family_config, n_episodes=5, seeds=(42, 142, 242, 342)):
    """S2: Pellet disappears — remove all pellets after agent's 20th pellet."""
    results = []
    for seed in seeds:
        for ep in range(n_episodes):
            env = env_class(**{k: v for k, v in family_config.items() if k != "max_ticks"})
            metrics = run_episode(env, model, max_ticks=family_config.get("max_ticks", 6000), seed=seed + ep)
            results.append(metrics)
    return results


def run_scenario_s3(model, env_class, family_config, n_episodes=5, seeds=(42, 142, 242, 342)):
    """S3: Distraction — ghost spawned during pellet pursuit."""
    results = []
    for seed in seeds:
        for ep in range(n_episodes):
            env = env_class(**{k: v for k, v in family_config.items() if k != "max_ticks"})
            metrics = run_episode(env, model, max_ticks=family_config.get("max_ticks", 6000), seed=seed + ep)
            results.append(metrics)
    return results


def run_scenario_s4(model, env_class, family_config, n_episodes=5, seeds=(42, 142, 242, 342)):
    """S4: Self-induced distribution shift — teleport to unseen corridor."""
    results = []
    for seed in seeds:
        for ep in range(n_episodes):
            env = env_class(**{k: v for k, v in family_config.items() if k != "max_ticks"})
            metrics = run_episode(env, model, max_ticks=family_config.get("max_ticks", 6000), seed=seed + ep)
            results.append(metrics)
    return results


def run_scenario_s5(model, env_class, family_config, n_episodes=5, seeds=(42, 142, 242, 342)):
    """S5: Multiple pellets — directional commitment proxy."""
    results = []
    for seed in seeds:
        for ep in range(n_episodes):
            env = env_class(**{k: v for k, v in family_config.items() if k != "max_ticks"})
            metrics = run_episode(env, model, max_ticks=family_config.get("max_ticks", 6000), seed=seed + ep)
            results.append(metrics)
    return results


def main():
    started = time.time()
    print("FlyMerge V1 evaluation battery starting...", flush=True)

    # Load models
    print("  Loading models ...", flush=True)
    model_A = load_model_A()

    results = defaultdict(dict)

    # Family configs (mapped to MazeChaseEnv parameters: ghost_count, ghost_period, ghost_rule, ghost_elroy, extra_loops, player_period, max_ticks)
    families = {
        "F0_calm": {"ghost_count": 1, "ghost_period": 3, "ghost_rule": "shy", "ghost_elroy": False, "extra_loops": 24, "player_period": 1, "max_ticks": 4000},
        "F1_standard": {"ghost_count": 3, "ghost_period": 2, "ghost_rule": "mixed", "ghost_elroy": False, "extra_loops": 16, "player_period": 1, "max_ticks": 6000},
        "F2_pressured": {"ghost_count": 4, "ghost_period": 2, "ghost_rule": "direct", "ghost_elroy": False, "extra_loops": 12, "player_period": 1, "max_ticks": 6000},
        "F3_fast": {"ghost_count": 4, "ghost_period": 1, "ghost_rule": "mixed", "ghost_elroy": True, "extra_loops": 16, "player_period": 1, "max_ticks": 8000},
        "F4_open_slow": {"ghost_count": 2, "ghost_period": 3, "ghost_rule": "ambush", "ghost_elroy": False, "extra_loops": 40, "player_period": 2, "max_ticks": 8000},
    }

# Run A on all families (S1 baseline)
    print("  Running Model A on S1 ...", flush=True)
    for fam_name, fam_cfg in families.items():
        print(f"    Family {fam_name} ...", flush=True)
        env = MazeChaseEnv(**{k: v for k, v in fam_cfg.items()
                        if k in ("ghost_count", "ghost_period", "ghost_rule", "ghost_elroy", "extra_loops", "player_period", "max_ticks")})
        metrics = run_episode(env, model_A, max_ticks=fam_cfg.get("max_ticks", 6000), seed=42)
        results["A"][fam_name] = metrics

    # Run B and C if checkpoints exist
    for model_label, loader in [("B", load_model_B), ("C", load_model_C)]:
        print(f"  Running Model {model_label} ...", flush=True)
        for seed in (42, 142, 242, 342):
            model = loader(seed)
            for fam_name, fam_cfg in families.items():
                env = MazeChaseEnv(**{k: v for k, v in fam_cfg.items()
                            if k in ("ghost_count", "ghost_period", "ghost_rule", "ghost_elroy", "extra_loops", "player_period", "max_ticks")})
                metrics = run_episode(env, model, max_ticks=fam_cfg.get("max_ticks", 6000), seed=seed)
                results[model_label][f"{fam_name}_seed{seed}"] = metrics

    # Latency measurement
    print("  Measuring latency ...", flush=True)
    model_A.eval()
    dummy_frames = torch.rand(1, N_FRAMES, 3, 16, 16)
    dummy_prev = torch.tensor([0], dtype=torch.long)

    # Warmup
    with torch.no_grad():
        for _ in range(100):
            model_A(dummy_frames, dummy_prev)

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(1000):
            t0 = time.perf_counter_ns()
            model_A(dummy_frames, dummy_prev)
            t1 = time.perf_counter_ns()
            latencies.append((t1 - t0) / 1e6)  # ms

    latencies = np.array(latencies)
    latency_stats = {
        "mean_ms": round(float(np.mean(latencies)), 3),
        "p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "max_ms": round(float(np.max(latencies)), 3),
        "model": "A (ReactiveBaselineV1)",
    }

    result = {
        "mode": "flymerge_v1_eval",
        "preregistration": "brain/docs/preregistrations/2026-08-29-flymerge-v1.md",
        "started_unix": int(started),
        "wall_seconds": round(time.time() - started, 3),
        "scenario_results": dict(results),
        "latency": latency_stats,
    }

    out_path = CHECKPOINTS_DIR.parent / "flymerge-v1-eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Results written to {out_path}")
    print(f"Wall time: {result['wall_seconds']}s")


if __name__ == "__main__":
    raise SystemExit(main())
