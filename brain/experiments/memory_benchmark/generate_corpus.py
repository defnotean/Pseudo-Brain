"""Corpus generator for KeysDoorsEnv.

Generates ground-truth expert demonstrations partitioned into TRAIN, DEV, and TEST.
Saves episodes as compressed numpy archives containing:
- frames: (T, 16, 16, 3) uint8 RGB
- actions: (T,) int64
- prev_actions: (T,) int64
- has_key: (T,) uint8 (diagnostic)
- door_open: (T,) uint8 (diagnostic)
"""
from __future__ import annotations

import os
import sys
import numpy as np
from pathlib import Path

# Add package paths
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.environments.keys_doors import KeysDoorsEnv
from memory_benchmark.expert import KeysDoorsExpert

def generate_partition(name: str, start_seed: int, n_episodes: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    env = KeysDoorsEnv()
    expert = KeysDoorsExpert(env)

    total_transitions = 0
    episodes_saved = 0

    print(f"Generating {name} split: {n_episodes} episodes starting at seed {start_seed}...")
    for idx in range(n_episodes):
        seed = start_seed + idx
        transitions, success = expert.run_episode(seed=seed, max_ticks=400)
        if not success:
            print(f"  Warning: episode seed {seed} did not succeed, skipping.")
            continue

        T = len(transitions)
        frames = np.zeros((T, 16, 16, 3), dtype=np.uint8)
        actions = np.zeros((T,), dtype=np.int64)
        prev_actions = np.zeros((T,), dtype=np.int64)
        has_key = np.zeros((T,), dtype=np.uint8)
        door_open = np.zeros((T,), dtype=np.uint8)

        for t, trans in enumerate(transitions):
            # Parse raw bytes (16*16*3)
            raw_rgb = trans["rgb"]
            frames[t] = np.frombuffer(raw_rgb, dtype=np.uint8).reshape(16, 16, 3)
            actions[t] = trans["action"]
            prev_actions[t] = trans["prev_action"]
            has_key[t] = trans["has_key"]
            door_open[t] = trans["door_open"]

        save_path = out_dir / f"ep_{idx:04d}_seed_{seed}.npz"
        np.savez_compressed(
            save_path,
            frames=frames,
            actions=actions,
            prev_actions=prev_actions,
            has_key=has_key,
            door_open=door_open,
        )
        total_transitions += T
        episodes_saved += 1

    print(f"  Done {name}: {episodes_saved} episodes, {total_transitions} transitions.")
    return episodes_saved, total_transitions


def main():
    base_dir = _REPO_ROOT / "datasets" / "keys_doors_corpus_v1"
    print(f"Writing corpus to {base_dir}")

    generate_partition("TRAIN", start_seed=1000, n_episodes=100, out_dir=base_dir / "train")
    generate_partition("DEV", start_seed=2000, n_episodes=20, out_dir=base_dir / "dev")
    generate_partition("TEST", start_seed=3000, n_episodes=20, out_dir=base_dir / "test")
    print("Corpus generation complete!")


if __name__ == "__main__":
    main()
