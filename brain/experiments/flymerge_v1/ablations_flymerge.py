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


def run_episode_ablated(env, model, max_ticks, ablate="none", h_init=None):
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
    obs = env.reset()
    frames_buf = [obs.frame] * N_FRAMES
    prev_action = 0
    device = next(model.parameters()).device

    B = 1
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
            np.stack(frames_buf[-N_FRAMES:]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        ).unsqueeze(0).to(device)
        prev_act_tensor = torch.tensor([prev_action], dtype=torch.long, device=device)

        with torch.no_grad():
            if ablate == "none":
                logits, h_new, u = model(frames_tensor, prev_act_tensor, h)
            elif ablate == "shunting":
                # Compute with h_new but don't apply shunting: use h_new directly
                logits_raw, h_new_raw, u = model(frames_tensor, prev_act_tensor, h)
                # Override: bypass shunting means h = h_new (g=0, so h' = GRU(h,f))
                h = h_new_raw
                # Recompute logits without shunting effect
                h_bar_proj = model.h_proj(h)
                logits = h_bar_proj
                bypass = model.bypass_proj(u)
                logits = logits + bypass
            elif ablate == "modulation":
                logits_raw, h_new, u = model(frames_tensor, prev_act_tensor, h)
                # Temperature modulation disabled: use temp=1
                h_bar_proj = model.h_proj(h_new)
                logits = h_bar_proj
                bypass = model.bypass_proj(u)
                logits = logits + bypass
                h = h_new
            elif ablate == "bypass":
                logits_raw, h_new, u = model(frames_tensor, prev_act_tensor, h)
                # Zero out the bypass contribution
                h_bar_proj = model.h_proj(h_new)
                logits = h_bar_proj  # no bypass term
                h = h_new
            elif ablate == "freeze_h":
                # h stays frozen at its initial value throughout
                logits_raw, h_new, u = model(frames_tensor, prev_act_tensor, h)
                h = h  # don't update h
                logits = model.h_proj(h) + model.bypass_proj(u)
            elif ablate == "zero_h":
                logits, h_new, u = model(frames_tensor, prev_act_tensor, h)
                h = torch.zeros(1, 32, device=device, dtype=torch.float32)  # zero each tick
            elif ablate == "random_h":
                logits, h_new, u = model(frames_tensor, prev_act_tensor, h)
                h = torch.randn_like(h)  # random noise each tick
            elif ablate == "untrained_u":
                logits, h_new, u = model(frames_tensor, prev_act_tensor, h)
                # Replace u with random (not from innate channel)
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

        control_map = {0: GenericControl(NOOP=True), 1: GenericControl(w=1),
                       2: GenericControl(a=1), 3: GenericControl(s=1),
                       4: GenericControl(d=1)}
        ctrl = control_map.get(action, GenericControl(NOOP=True))
        step_out = env.step(ctrl)

        metrics["ticks"] += 1
        if action != 0:
            metrics["movement_ticks"] += 1
        else:
            metrics["idle_ticks"] += 1

        if step_out.frame is not None:
            frames_buf.append(step_out.frame)
            if len(frames_buf) > N_FRAMES:
                frames_buf.pop(0)

        if step_out.caught:
            metrics["survived"] = False
            metrics["ghost_catches"] += 1
            break
        if step_out.cleared:
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

    # Family configs
    families = {
        "F0_calm": {"n_ghosts": 1, "ghost_rule": "shy", "max_ticks": 4000,
                     "player_speed": 1, "ghost_speed": 1},
        "F1_standard": {"n_ghosts": 3, "ghost_rule": "mixed", "max_ticks": 6000,
                        "player_speed": 1, "ghost_speed": 1},
        "F2_pressured": {"n_ghosts": 4, "ghost_rule": "direct", "max_ticks": 6000,
                         "player_speed": 1, "ghost_speed": 1},
        "F3_fast": {"n_ghosts": 4, "ghost_rule": "mixed", "max_ticks": 8000,
                     "player_speed": 1, "ghost_speed": 1, "ghost_elroy": True},
        "F4_open_slow": {"n_ghosts": 2, "ghost_rule": "ambush", "max_ticks": 8000,
                         "player_speed": 1, "ghost_speed": 2},
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
                         if k in ("n_ghosts", "ghost_rule", "player_speed", "ghost_speed", "ghost_elroy")}
            env = MazeChaseEnv(grid_size=16, seed=seed, **env_kwargs)
            for ablate in ablations:
                metrics = run_episode_ablated(env, model, fam_cfg.get("max_ticks", 6000), ablate=ablate)
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
