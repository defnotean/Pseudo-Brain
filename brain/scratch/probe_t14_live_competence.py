"""T1-4 follow-up DIAGNOSTIC PROBE (read-only; NOT a qualification).

Question it answers: does the trained ReactiveBaselineV1 actually PLAY the
Pac-Man-like env closed-loop — does it collect pellets and survive like the
teacher — rather than merely classifying stored frames?

Method: for a fixed set of (family, seed) episodes, run BOTH the pixel-only
expert (the corpus teacher) and the trained seed-42 model policy on the
IDENTICAL env, and report the same M_P / pellet-fraction the corpus generator
uses. Model M_P vs teacher M_P is the honest competence read.

This opens no DEV/CAL/TEST partition, trains nothing, and publishes nothing:
it is a diagnostic probe (same discipline as the L-series). It re-uses the
frozen corpus-generation env contract so the numbers are comparable.

Usage (from brain/):
  py -3.11 scratch/probe_t14_live_competence.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

_HERE = Path(__file__).resolve()
BRAIN_ROOT = _HERE.parent.parent
sys.path.insert(0, str(BRAIN_ROOT / "src"))
sys.path.insert(0, str(BRAIN_ROOT / "scripts"))

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
try:  # below-normal priority: gaming-polite
    from ctypes import windll  # type: ignore
    windll.kernel32.SetPriorityClass(
        windll.kernel32.GetCurrentProcess(), 0x00004000)
except Exception:
    pass

from irene_brain.environments.maze_chase import MazeChaseEnv  # noqa: E402
from irene_brain.environments.pacman_harness import FAMILIES, partition_seeds  # noqa: E402
from irene_brain.evaluation.diagnostic_policies import ScriptedMazeChasePlannerPolicy  # noqa: E402
from irene_brain.types import GenericControl  # noqa: E402
from irene_brain.v2.trajectory_objective import control_action_class  # noqa: E402
from irene_brain.v2.reactive_baseline import N_FRAMES, ReactiveBaselineV1  # noqa: E402

EP_PER_FAMILY = 5
CKPT = BRAIN_ROOT / "runs" / "embodied-reactive-baseline-v1" / "seed_42.pt"
# action class -> HidKey (0 idle, 1 W, 2 A, 3 S, 4 D); each a TUPLE
CLASS_TO_KEY = {0: (), 1: (0x1A,), 2: (0x04,), 3: (0x16,), 4: (0x07,)}


def _planner(fam):
    return ScriptedMazeChasePlannerPolicy(
        ghost_period=fam.ghost_period, player_period=fam.player_period,
        ghost_elroy=fam.ghost_elroy, candidate_pellets=6, horizon=24,
        input_delay_ticks=0)


def total_pellets(env) -> int:
    n = 0
    for (x, y) in env._maze:  # noqa: SLF001 (evidence only)
        if env._has_pellet(x, y):  # noqa: SLF001
            n += 1
    return n


def run_teacher(fam_idx, seed):
    fam = FAMILIES[fam_idx]
    env = MazeChaseEnv(**fam.env_kwargs())
    env.reset(seed)
    plan = _planner(fam); plan.reset(seed)
    tp = total_pellets(env)
    pellets = 0; tick = 0
    while True:
        obs = env.current_observation
        ctrl = plan.act(obs)
        out = env.step(ctrl)
        if "pellet_eaten" in out.events:
            pellets += 1
        tick += 1
        if out.terminated or out.truncated:
            break
    m_p = min(1.0, max(0.0, 0.5 * (pellets / max(1, tp)) + 0.5 * (tick / fam.max_ticks)))
    return m_p, pellets, tick, tp


def run_model(model, fam_idx, seed):
    fam = FAMILIES[fam_idx]
    env = MazeChaseEnv(**fam.env_kwargs())
    env.reset(seed)
    tp = total_pellets(env)
    pellets = 0; tick = 0
    frames = []  # rolling last-4 window of obs.rgb.pixels bytes
    first_px = None
    prev_action = 0
    while True:
        obs = env.current_observation
        px = obs.rgb.pixels
        if first_px is None:
            first_px = px
        frames.append(px)
        if len(frames) > N_FRAMES:
            frames.pop(0)
        # right-aligned last-4; left-pad with the episode's reset frame
        # (the first rendered frame), per the frozen training convention.
        if len(frames) < N_FRAMES:
            win = [first_px] * (N_FRAMES - len(frames)) + list(frames)
        else:
            win = frames
        win = [np.frombuffer(w, dtype=np.uint8).reshape(16, 16, 3) for w in win]
        ft = torch.from_numpy(np.stack(win).transpose(0, 3, 1, 2).astype(np.float32) / 255.0)[None]
        pa = torch.tensor([prev_action], dtype=torch.long)
        with torch.no_grad():
            cls = int(model(ft, pa).argmax(dim=1).item())
        ctrl = GenericControl(keys_down=CLASS_TO_KEY[cls])
        out = env.step(ctrl)
        if "pellet_eaten" in out.events:
            pellets += 1
        tick += 1
        prev_action = cls
        if out.terminated or out.truncated:
            break
    m_p = min(1.0, max(0.0, 0.5 * (pellets / max(1, tp)) + 0.5 * (tick / fam.max_ticks)))
    return m_p, pellets, tick, tp


def main() -> int:
    t0 = time.time()
    sd = torch.load(CKPT)["model_state_dict"]
    model = ReactiveBaselineV1(seed=int(__import__("zlib").crc32(b"reactive.42")))
    model.load_state_dict(sd)
    model.eval()
    print(f"loaded {CKPT.name} ({model.num_parameters} params); "
          f"{time.time()-t0:.1f}s")

    rows = []
    for fam_idx in range(len(FAMILIES)):
        seeds = partition_seeds(fam_idx, "TRAIN")[:EP_PER_FAMILY]
        for seed in seeds:
            tm, tp_p, tp_t, tp_tot = run_teacher(fam_idx, seed)
            mm, mp_p, mp_t, _ = run_model(model, fam_idx, seed)
            rows.append((fam_idx, seed, tm, mm, tp_p, mp_p, tp_t, mp_t, tp_tot))
            print(f"F{fam_idx} seed {seed}: teacher M_P={tm:.3f} ({tp_p}/{tp_tot} pel, {tp_t}t) | "
                  f"model M_P={mm:.3f} ({mp_p}/{tp_tot} pel, {mp_t}t)", flush=True)

    tm = np.mean([r[2] for r in rows]); mm = np.mean([r[3] for r in rows])
    tpf = np.mean([r[4] / r[8] for r in rows]); mpf = np.mean([r[5] / r[8] for r in rows])
    tt = np.mean([r[6] for r in rows]); mt = np.mean([r[7] for r in rows])
    print("\n=== CLOSED-LOOP COMPETENCE (seed-42 model vs teacher, read-only) ===")
    print(f"teacher: mean M_P={tm:.3f}  mean pellet-frac={tpf:.3f}  mean survival={tt:.0f}t")
    print(f"model  : mean M_P={mm:.3f}  mean pellet-frac={mpf:.3f}  mean survival={mt:.0f}t")
    print(f"ratio (model/teacher)  M_P={mm/max(tm,1e-9):.3f}  pellet={mpf/max(tpf,1e-9):.3f}  "
          f"survival={mt/max(tt,1e-9):.3f}")
    print(f"(wall {time.time()-t0:.1f}s; n={len(rows)} episodes; diagnostic only, no publish)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
