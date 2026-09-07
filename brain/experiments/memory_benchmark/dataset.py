"""Sequence dataset loader for KeysDoors corpus.

Loads .npz files containing:
- frames: (T, 16, 16, 3) uint8 RGB
- actions: (T,) int64
- prev_actions: (T,) int64
- has_key: (T,) uint8
- door_open: (T,) uint8

Returns fixed-length sequence windows (default seq_len=32) formatted for:
- frames: (T, N_FRAMES, 3, 16, 16) float32 [0, 1]
- actions: (T,) int64 ground truth action labels
- prev_actions: (T,) int64 ground truth previous actions
- has_key: (T,) uint8 diagnostic state
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

N_FRAMES = 4

class KeysDoorsSequenceDataset(Dataset):
    """Yields consecutive (T, N_FRAMES, 3, 16, 16) observation windows."""

    def __init__(self, split_dir: str | Path, seq_len: int = 32):
        self.split_dir = Path(split_dir)
        self.seq_len = seq_len
        self.files = sorted(list(self.split_dir.glob("*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No .npz episodes found in {self.split_dir}")

        self.frames = []
        self.actions = []
        self.prev_actions = []
        self.has_key = []
        self.door_open = []

        # Preload all episodes in memory (they are tiny, ~100 episodes is a few MBs)
        for f in self.files:
            data = np.load(f)
            self.frames.append(data["frames"])          # (T, 16, 16, 3)
            self.actions.append(data["actions"])        # (T,)
            self.prev_actions.append(data["prev_actions"])
            self.has_key.append(data["has_key"])
            self.door_open.append(data["door_open"])

        # Construct indices (episode_idx, start_t)
        self.indices: List[Tuple[int, int]] = []
        for ep_idx, fr in enumerate(self.frames):
            T = fr.shape[0]
            max_start = T - self.seq_len
            if max_start < 0:
                continue
            for t in range(0, max_start + 1):
                self.indices.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t0 = self.indices[idx]
        fr = self.frames[ep_idx]
        act = self.actions[ep_idx]
        pact = self.prev_actions[ep_idx]
        hk = self.has_key[ep_idx]
        do = self.door_open[ep_idx]

        T = self.seq_len
        # Stack N_FRAMES prior frames for each timestep
        frames_win = []
        for dt in range(T):
            t = t0 + dt
            win = []
            for i in range(N_FRAMES):
                ti = max(t - (N_FRAMES - 1 - i), 0)
                win.append(fr[ti])
            frames_win.append(np.stack(win))  # (N_FRAMES, 16, 16, 3)

        # Transpose to (T, N_FRAMES, 3, 16, 16) float32 in [0, 1]
        frames = np.stack(frames_win).transpose(0, 1, 4, 2, 3).astype(np.float32) / 255.0

        return {
            "frames": torch.from_numpy(frames),
            "actions": torch.from_numpy(act[t0 : t0 + T].astype(np.int64)),
            "prev_actions": torch.from_numpy(pact[t0 : t0 + T].astype(np.int64)),
            "has_key": torch.from_numpy(hk[t0 : t0 + T].astype(np.uint8)),
            "door_open": torch.from_numpy(do[t0 : t0 + T].astype(np.uint8)),
        }


def collate_sequences(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "frames": torch.stack([b["frames"] for b in batch]),          # (B, T, N_FRAMES, 3, 16, 16)
        "actions": torch.stack([b["actions"] for b in batch]),        # (B, T)
        "prev_actions": torch.stack([b["prev_actions"] for b in batch]),
        "has_key": torch.stack([b["has_key"] for b in batch]),
        "door_open": torch.stack([b["door_open"] for b in batch]),
    }
