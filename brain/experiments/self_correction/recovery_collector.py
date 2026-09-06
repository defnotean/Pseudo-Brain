"""Phase D: Recovery demonstration collector.

Rolls out a policy in closed loop, injects controlled mistakes/perturbations,
and calls the expert/oracle pathfinder from the resulting off-course state
to record true recovery trajectories back to the goal.

Mixes clean expert demonstrations with recovery transitions at a configurable
ratio (default: 75% clean / 25% recovery).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.environments.keys_doors import KeysDoorsEnv
from memory_benchmark.expert import KeysDoorsExpert, control_for_action, ACTION_IDLE
from memory_benchmark.models import make_model, N_FRAMES
from self_correction.perturbations import PerturbationEngine


def collect_recovery_episode(
    env: KeysDoorsEnv,
    seed: int,
    policy_model: Optional[object] = None,
    perturb_tick: int = 20,
    max_ticks: int = 300,
) -> Tuple[List[Dict], bool]:
    obs = env.reset(seed)
    expert = KeysDoorsExpert(env)

    transitions = []
    prev_action = ACTION_IDLE
    perturber = PerturbationEngine(mode="single", inject_tick=perturb_tick, seed=seed)

    success = False

    for tick in range(max_ticks):
        raw_rgb = obs.rgb.pixels
        player_x, player_y = env._player_x, env._player_y
        has_key = env._has_key
        door_open = env._door_open

        # Stage 1: before perturbation, follow expert
        # Stage 2: at perturb_tick, inject wrong action
        # Stage 3: after perturbation, expert takes over from the error state!
        if perturber.should_inject(tick):
            expert_opt = expert.get_action()
            action_to_apply = perturber.get_perturbed_action(expert_opt)
            is_recovery_transition = False
        else:
            action_to_apply = expert.get_action()
            is_recovery_transition = perturber.injected and not perturber.recovered

        perturber.update_recovery_state(env, tick)

        transitions.append({
            "rgb": raw_rgb,
            "action": action_to_apply,
            "prev_action": prev_action,
            "has_key": has_key,
            "door_open": door_open,
            "is_recovery": is_recovery_transition,
        })

        ctrl = control_for_action(action_to_apply)
        step_out = env.step(ctrl)
        obs = step_out.observation
        prev_action = action_to_apply

        if step_out.reward > 0.0 or "target_collected" in step_out.events:
            success = True
            break

    return transitions, success


def build_recovery_corpus(
    output_dir: str | Path,
    clean_corpus_dir: str | Path,
    n_recovery_episodes: int = 35,
    clean_ratio: float = 0.75,
    start_seed: int = 5000,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_out = out_path / "train"
    train_out.mkdir(parents=True, exist_ok=True)

    env = KeysDoorsEnv()

    # 1. Collect recovery episodes
    print(f"Collecting {n_recovery_episodes} recovery demonstration episodes...")
    recovery_count = 0
    for idx in range(n_recovery_episodes):
        seed = start_seed + idx
        perturb_tick = int(np.random.randint(15, 35))
        transitions, succ = collect_recovery_episode(env, seed, perturb_tick=perturb_tick)
        if not succ:
            continue

        T = len(transitions)
        frames = np.zeros((T, 16, 16, 3), dtype=np.uint8)
        actions = np.zeros((T,), dtype=np.int64)
        prev_actions = np.zeros((T,), dtype=np.int64)
        has_key = np.zeros((T,), dtype=np.uint8)
        door_open = np.zeros((T,), dtype=np.uint8)

        for t, tr in enumerate(transitions):
            frames[t] = np.frombuffer(tr["rgb"], dtype=np.uint8).reshape(16, 16, 3)
            actions[t] = tr["action"]
            prev_actions[t] = tr["prev_action"]
            has_key[t] = tr["has_key"]
            door_open[t] = tr["door_open"]

        save_file = train_out / f"recovery_ep_{idx:04d}_seed_{seed}.npz"
        np.savez_compressed(
            save_file,
            frames=frames,
            actions=actions,
            prev_actions=prev_actions,
            has_key=has_key,
            door_open=door_open,
        )
        recovery_count += 1

    print(f"Saved {recovery_count} recovery episodes.")

    # 2. Link/Copy clean episodes to match the clean_ratio
    clean_src = Path(clean_corpus_dir) / "train"
    clean_files = sorted(list(clean_src.glob("*.npz")))
    n_clean_target = int(recovery_count * (clean_ratio / (1.0 - clean_ratio)))
    n_clean_to_take = min(len(clean_files), n_clean_target)

    print(f"Mixing in {n_clean_to_take} clean demonstration episodes (ratio {clean_ratio*100:.0f}% clean / {(1-clean_ratio)*100:.0f}% recovery)...")
    for i in range(n_clean_to_take):
        src_file = clean_files[i]
        dst_file = train_out / f"clean_{src_file.name}"
        if not dst_file.exists():
            import shutil
            shutil.copy2(src_file, dst_file)

    # 3. Copy DEV and TEST splits verbatim from clean corpus
    for split in ["dev", "test"]:
        s_src = Path(clean_corpus_dir) / split
        s_dst = out_path / split
        s_dst.mkdir(parents=True, exist_ok=True)
        for f in s_src.glob("*.npz"):
            import shutil
            shutil.copy2(f, s_dst / f.name)

    total_train = len(list(train_out.glob("*.npz")))
    print(f"Mixed recovery dataset complete: {total_train} train episodes in {train_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="brain/datasets/keys_doors_recovery_corpus_v1")
    parser.add_argument("--clean_corpus_dir", type=str, default="brain/datasets/keys_doors_corpus_v1")
    parser.add_argument("--n_recovery", type=int, default=35)
    parser.add_argument("--clean_ratio", type=float, default=0.75)

    args = parser.parse_args()
    build_recovery_corpus(
        output_dir=args.output_dir,
        clean_corpus_dir=args.clean_corpus_dir,
        n_recovery_episodes=args.n_recovery,
        clean_ratio=args.clean_ratio,
    )
