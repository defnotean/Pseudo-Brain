"""Multi-trial demonstration corpus and dataset loader for HiddenRuleEnv.

Generates and loads multi-trial demonstration sessions demonstrating:
1. Trial 1 exploration (50/50 prior, discovering the active rule via feedback).
2. Exploitation in subsequent trials within the block.
3. Adaptation upon rule reversal (surprise feedback -> behavioral switch).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.types import GenericControl, HidKey
from online_adaptation.hidden_rule_env import HiddenRuleEnv, Rule

N_FRAMES = 4

# Map action integers to controls
ACTION_IDLE = 0
ACTION_UP = 1    # W
ACTION_LEFT = 2  # A
ACTION_DOWN = 3  # S
ACTION_RIGHT = 4 # D

CTRL_MAP = {
    ACTION_IDLE: GenericControl(),
    ACTION_UP: GenericControl(keys_down=(int(HidKey.W),)),
    ACTION_LEFT: GenericControl(keys_down=(int(HidKey.A),)),
    ACTION_DOWN: GenericControl(keys_down=(int(HidKey.S),)),
    ACTION_RIGHT: GenericControl(keys_down=(int(HidKey.D),)),
}


def generate_single_session_trajectory(
    env: HiddenRuleEnv,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Rolls out an expert/explorative demonstrator across a multi-trial session."""
    np.random.seed(seed)
    obs = env.reset(seed=seed)

    all_frames = []
    all_actions = []
    all_prev_actions = []
    all_rewards = []
    all_trials = []
    all_rules = []

    # Frame buffer to stack N_FRAMES
    frame_raw = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
    frame_history = [frame_raw] * N_FRAMES

    prev_action = ACTION_IDLE
    known_rule: Optional[Rule] = None
    last_trial_idx = -1

    done = False
    while not done:
        curr_trial = env.current_trial
        if curr_trial != last_trial_idx:
            # New trial started
            last_trial_idx = curr_trial
            # If the current block's rule changed without the agent knowing yet,
            # we keep the previous known_rule until we encounter feedback!

        # Decide action based on position and known_rule
        px, py = env._player_x, env._player_y

        if py > 7:
            # In vertical corridor: must move UP to junction
            action = ACTION_UP
        elif py == 7 and px == 7:
            # At decision junction!
            if known_rule is None:
                # Exploratory trial: randomly explore Left or Right (50/50)
                action = ACTION_LEFT if np.random.rand() < 0.5 else ACTION_RIGHT
            elif known_rule == Rule.RULE_A:
                action = ACTION_LEFT
            elif known_rule == Rule.RULE_B:
                action = ACTION_RIGHT
            else:
                action = ACTION_LEFT
        elif px < 7 and px > 3:
            # Heading LEFT towards door
            action = ACTION_LEFT
        elif px > 7 and px < 11:
            # Heading RIGHT towards door
            action = ACTION_RIGHT
        else:
            action = ACTION_IDLE

        # Record observation
        all_frames.append(frame_raw)
        all_actions.append(action)
        all_prev_actions.append(prev_action)
        all_trials.append(curr_trial)
        all_rules.append(1 if env.current_rule == Rule.RULE_A else 2)

        # Step environment
        ctrl = CTRL_MAP[action]
        outcome = env.step(ctrl)

        all_rewards.append(outcome.reward)

        # Update known rule from feedback
        if "trial_success" in outcome.events:
            # The action we took was correct for this block!
            if px <= 3 or action == ACTION_LEFT:
                known_rule = Rule.RULE_A
            else:
                known_rule = Rule.RULE_B
        elif "trial_failure" in outcome.events:
            # The action we took was wrong! So the rule must be the opposite!
            if px <= 3 or action == ACTION_LEFT:
                known_rule = Rule.RULE_B
            else:
                known_rule = Rule.RULE_A

        prev_action = action
        frame_raw = np.frombuffer(outcome.observation.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)

        if outcome.terminated or outcome.truncated:
            done = True

    return {
        "frames": np.array(all_frames, dtype=np.uint8),            # [T, 16, 16, 3]
        "actions": np.array(all_actions, dtype=np.int64),          # [T]
        "prev_actions": np.array(all_prev_actions, dtype=np.int64),# [T]
        "rewards": np.array(all_rewards, dtype=np.float32),        # [T]
        "trials": np.array(all_trials, dtype=np.int32),            # [T]
        "rules": np.array(all_rules, dtype=np.int32),              # [T]
    }


def generate_corpus(
    output_dir: str | Path,
    n_train: int = 80,
    n_dev: int = 20,
    seed: int = 1000,
):
    out_path = Path(output_dir)
    train_dir = out_path / "train"
    dev_dir = out_path / "dev"
    train_dir.mkdir(parents=True, exist_ok=True)
    dev_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_train} train and {n_dev} dev sessions into {out_path}...")

    # Varied schedules for training to prevent overfitting to a single block length
    schedules = [
        [(Rule.RULE_A, 8), (Rule.RULE_B, 8)],
        [(Rule.RULE_B, 8), (Rule.RULE_A, 8)],
        [(Rule.RULE_A, 5), (Rule.RULE_B, 5), (Rule.RULE_A, 5)],
        [(Rule.RULE_B, 5), (Rule.RULE_A, 5), (Rule.RULE_B, 5)],
        [(Rule.RULE_A, 10)],
        [(Rule.RULE_B, 10)],
    ]

    # Generate Train
    for i in range(n_train):
        sched = schedules[i % len(schedules)]
        env = HiddenRuleEnv(rule_schedule=sched)
        data = generate_single_session_trajectory(env, seed=seed + i)
        np.savez_compressed(train_dir / f"session_{i:04d}.npz", **data)

    # Generate Dev
    for i in range(n_dev):
        sched = schedules[i % len(schedules)]
        env = HiddenRuleEnv(rule_schedule=sched)
        data = generate_single_session_trajectory(env, seed=seed + 5000 + i)
        np.savez_compressed(dev_dir / f"session_{i:04d}.npz", **data)

    print(f"Dataset generated: {len(list(train_dir.glob('*.npz')))} train, {len(list(dev_dir.glob('*.npz')))} dev.")


class HiddenRuleSequenceDataset(Dataset):
    """Yields consecutive (T, N_FRAMES, 3, 16, 16) observation windows."""

    def __init__(self, split_dir: str | Path, seq_len: int = 32):
        self.split_dir = Path(split_dir)
        self.seq_len = seq_len
        self.files = sorted(list(self.split_dir.glob("*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.split_dir}")

        self.frames = []
        self.actions = []
        self.prev_actions = []
        self.rewards = []
        self.trials = []

        for f in self.files:
            data = np.load(f)
            self.frames.append(data["frames"])
            self.actions.append(data["actions"])
            self.prev_actions.append(data["prev_actions"])
            self.rewards.append(data["rewards"])
            self.trials.append(data["trials"])

        self.indices: List[Tuple[int, int]] = []
        for ep_idx, (fr, tr) in enumerate(zip(self.frames, self.trials)):
            T = fr.shape[0]
            max_start = T - self.seq_len
            if max_start < 0:
                continue
            # Anchor sequence windows at trial boundaries to capture causal feedback -> decision transitions
            trial_starts = set([0] + [t for t in range(1, T) if tr[t] != tr[t - 1]])
            for t_start in trial_starts:
                if t_start <= max_start:
                    self.indices.append((ep_idx, t_start))
            for t in range(0, max_start + 1, 4):
                if t not in trial_starts:
                    self.indices.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t0 = self.indices[idx]
        fr = self.frames[ep_idx]
        act = self.actions[ep_idx]
        pact = self.prev_actions[ep_idx]
        rew = self.rewards[ep_idx]

        T = self.seq_len
        frames_win = []
        for dt in range(T):
            t_curr = t0 + dt
            stacked = []
            for k in range(N_FRAMES):
                past_t = max(0, t_curr - (N_FRAMES - 1 - k))
                stacked.append(fr[past_t])
            # [N_FRAMES, 16, 16, 3] -> transpose to [N_FRAMES, 3, 16, 16]
            arr = np.stack(stacked, axis=0).transpose(0, 3, 1, 2)
            frames_win.append(arr)

        frames_tensor = torch.from_numpy(np.stack(frames_win, axis=0)).float() / 255.0
        actions_tensor = torch.from_numpy(act[t0 : t0 + T].copy()).long()
        prev_actions_tensor = torch.from_numpy(pact[t0 : t0 + T].copy()).long()
        rewards_tensor = torch.from_numpy(rew[t0 : t0 + T].copy()).float()

        return {
            "frames": frames_tensor,           # [T, N_FRAMES, 3, 16, 16]
            "actions": actions_tensor,         # [T]
            "prev_actions": prev_actions_tensor,# [T]
            "rewards": rewards_tensor,         # [T]
        }


def collate_sequences(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "frames": torch.stack([b["frames"] for b in batch], dim=0),
        "actions": torch.stack([b["actions"] for b in batch], dim=0),
        "prev_actions": torch.stack([b["prev_actions"] for b in batch], dim=0),
        "rewards": torch.stack([b["rewards"] for b in batch], dim=0),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="brain/datasets/hidden_rule_corpus_v1")
    parser.add_argument("--n_train", type=int, default=80)
    parser.add_argument("--n_dev", type=int, default=20)
    args = parser.parse_args()

    generate_corpus(args.output_dir, n_train=args.n_train, n_dev=args.n_dev)
