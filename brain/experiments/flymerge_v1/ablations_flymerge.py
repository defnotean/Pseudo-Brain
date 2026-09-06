"""Run Model C ablations: shunting disabled, modulation disabled, etc.

Preregistration: brain/docs/preregistrations/2026-08-29-flymerge-v1.md

Each ablation runs the S1-S5 scenario battery with one intervention
disabled, then compares behavior to the full Model C.
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
from irene_brain.v2.reactive_baseline import ReactiveBaselineV1, N_FRAMES  # noqa: E402
from flymerge_v1.model import FlyMergeV1  # noqa: E402
from irene_brain.types import GenericControl  # noqa: E402

CHECKPOINTS_DIR = BRAIN_ROOT / "runs" / "flymerge-v1"


def load_model_C(seed):
    ckpt_path = CHECKPOINTS_DIR / f"seed_{seed}_C.pt"
    model = FlyMergeV1()
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _rgb_to_np(rgb_frame):
    """Convert RgbFrame to numpy array [H, W, 3] float32 in [0,1]."""
    arr = np.frombuffer(rgb_frame.pixels, dtype=np.uint8).reshape(
        rgb_frame.height, rgb_frame.width, 3
    )
    return arr.astype(np.float32) / 255.0


_ACTION_KEYS = (0x1A, 0x04, 0x16, 0x07)  # W A S D


def _control_for_class(action_class: int) -> GenericControl:
    if action_class == 0:
        return GenericControl(keys_down=())
    return GenericControl(keys_down=(_ACTION_KEYS[action_class - 1],))


def run_episode_ablated(env, model, max_ticks, seed, ablate="none", h_init=None):
    """Run an episode with an ablation applied.

    ablate options:
      - "none": full model
      - "shunting": g(u) ≡ 0 (persistent state always updates freely)
      - "modulation": temp ≡ 1 (no urgency-based temperature scaling)
      - "bypass": bypass_proj = 0 (innate channel doesn't enter logits)
      - "freeze_h": h is frozen (no GRU update, uses initial h throughout)
      - "zero_h": h is zeroed each tick
      - "random_h": random noise injected into h each tick
      - "untrained_u": u set to uniform random (not from innate channel)
    """
    obs = env.reset(seed)
    frames_buf = [_rgb_to_np(obs.rgb)] * N_FRAMES
    prev_action = 0
    device = next(model.parameters()).device

    h = h_init.clone() if h_init is not None else torch.zeros(1, 32, device=device, dtype=torch.float32)

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

    for t in range(max_ticks):
        frames_tensor = torch.from_numpy(
            np.stack(frames_buf[-N_FRAMES:]).transpose(0, 3, 1, 2).astype(np.float32)
        ).unsqueeze(0).to(device)
        prev_act_tensor = torch.tensor([prev_action], dtype=torch.long, device=device)

        with torch.no_grad():
            if ablate == "none":
                logits, h_new, u = model(frames_tensor, prev_act_tensor, h)
            elif ablate == "shunting":
                logits_raw, h_new_raw, u = model(frames_tensor, prev_act_tensor, h)
                h = h_new_raw
                h_bar_proj = model.h_proj(h)
                logits = h_bar_proj
                bypass = model.bypass_proj(u)
                logits = logits + bypass
            elif ablate == "modulation":
                logits_raw, h_new, u = model(frames_tensor, prev_act_tensor, h)
                h_bar_proj = model.h_proj(h_new)
                logits = h_bar_proj
                bypass = model.bypass_proj(u)
                logits = logits + bypass
                h = h_new
            elif ablate == "bypass":
                logits_raw, h_new, u = model(frames_tensor, prev_act_tensor, h)
                h_bar_proj = model.h_proj(h_new)
                logits = h_bar_proj
                h = h_new
            elif ablate == "freeze_h":
                logits_raw, h_new, u = model(frames_tensor, prev_act_tensor, h)
                logits = model.h_proj(h) + model.bypass_proj(u)
            elif ablate == "zero_h":
                logits, h_new, u = model(frames_tensor, prev_act_tensor, h)
                h = torch.zeros(1, 32, device=device, dtype=torch.float32)
            elif ablate == "random_h":
                logits, h_new, u = model(frames_tensor, prev_act_tensor, h)
                h = torch.randn_like(h)
            elif ablate == "untrained_u":
                logits, h_new, u = model(frames_tensor, prev_act_tensor, h)
                u_rand = torch.rand(1, 1, device=device)
                h_bar_proj = model.h_proj(h_new)
                logits = h_bar_proj + model.bypass_proj(u_rand)
                h = h_new
            else:
                logits, h_new, u = model(frames_tensor, prev_act_tensor, h)

        action = logits.argmax(dim=-1).item()
        metrics["action_counts"][action] += 1

        p = torch.softmax(logits / 1.0, dim=-1)[0].cpu().numpy()
        p = p[p > 0]
        metrics["action_entropy_sum"] += -np.sum(p * np.log(p))

        ctrl = _control_for_class(action)
        step_out = env.step(ctrl)

        metrics["ticks"] += 1
        if action != 0:
            metrics["movement_ticks"] += 1
        else:
            metrics["idle_ticks"] += 1

        frames_buf.append(_rgb_to_np(step_out.observation.rgb))
        if len(frames_buf) > N_FRAMES:
            frames_buf.pop(0)

        if "caught" in step_out.events:
            metrics["survived"] = False
            metrics["ghost_catches"] += 1
            break
        if "cleared" in step_out.events:
            metrics["pellets_eaten"] += 1
            break

    metrics["avg_action_entropy"] = metrics["action_entropy_sum"] / max(metrics["ticks"], 1)
    return metrics


def main():
    started = time.time()
    print("FlyMerge V1 ablation study starting...", flush=True)

    # Check if training checkpoints exist
    model_C_exists = (CHECKPOINTS_DIR / "seed_42_C.pt").exists()
    if not model_C_exists:
        print("WARNING: Model C checkpoints not found. Run train_flymerge_v1.py first.", flush=True)

    # Family configs (matching MazeChaseEnv constructor parameters)
    families = {
        "F0_calm": {"ghost_count": 1, "ghost_period": 3, "ghost_rule": "shy",
                     "ghost_elroy": False, "extra_loops": 24, "player_period": 1, "max_ticks": 4000},
        "F1_standard": {"ghost_count": 3, "ghost_period": 2, "ghost_rule": "mixed",
                        "ghost_elroy": False, "extra_loops": 16, "player_period": 1, "max_ticks": 6000},
        "F2_pressured": {"ghost_count": 4, "ghost_period": 2, "ghost_rule": "direct",
                         "ghost_elroy": False, "extra_loops": 12, "player_period": 1, "max_ticks": 6000},
        "F3_fast": {"ghost_count": 4, "ghost_period": 1, "ghost_rule": "mixed",
                     "ghost_elroy": True, "extra_loops": 16, "player_period": 1, "max_ticks": 8000},
        "F4_open_slow": {"ghost_count": 2, "ghost_period": 3, "ghost_rule": "ambush",
                         "ghost_elroy": False, "extra_loops": 40, "player_period": 2, "max_ticks": 8000},
    }

    ablations = ["none", "shunting", "modulation", "bypass", "freeze_h", "zero_h", "random_h", "untrained_u"]
    seeds = [42, 142, 242, 342]

    results = defaultdict(dict)

    for seed in seeds:
        if not model_C_exists:
            break
        model = load_model_C(seed)
        for fam_name, fam_cfg in families.items():
            env_kwargs = {k: v for k, v in fam_cfg.items()
                         if k in ("ghost_count", "ghost_period", "ghost_rule",
                                  "ghost_elroy", "extra_loops", "player_period", "max_ticks")}
            env = MazeChaseEnv(**env_kwargs)
            for ablate in ablations:
                metrics = run_episode_ablated(env, model, fam_cfg.get("max_ticks", 6000),
                                              seed=seed, ablate=ablate)
                results[f"seed{seed}"][f"{fam_name}_{ablate}"] = metrics
        print(f"  seed {seed} done", flush=True)

    out_path = CHECKPOINTS_DIR.parent / "flymerge-v1-ablations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": "flymerge_v1_ablations",
        "started_unix": int(started),
        "wall_seconds": round(time.time() - started, 3),
        "ablations": ablations,
        "results": dict(results),
    }
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Ablation results written to {out_path}")
    print(f"Wall time: {result['wall_seconds']}s")


if __name__ == "__main__":
    raise SystemExit(main())
