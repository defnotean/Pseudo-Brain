"""ReactiveBaselineV1 — the frozen T1-4 reactive comparison arm.

Preregistration: ``brain/docs/preregistrations/2026-08-28-pacman-reactive-
baseline-v1.md``.

A pixel-driven, NON-RECURRENT model: a small CNN over a stack of the last
``N`` rendered frames plus a one-hot previous-action context, to a 5-class
(idle/W/A/S/D) action head. It is the "reactive" Q2/Q3 comparison arm: same
TRAIN corpus and same objective as the Core V2 candidate, but WITHOUT the
recurrent thoughtlet field, world model, belief, or episodic memory. It is
intended to be a *genuine* baseline (never weakened): it gets the full
4-frame temporal context (the strongest value in the registered 1-4 range)
and the candidate's training budget.

Sensory boundary (matches the candidate's deployed maze policy): the model
reads only rendered pixels (the last ``N`` frames, each 16x16, nearest-
upsampled to 32x32) and the previous applied action one-hot. It reads no
clocks, frame IDs, reward, hazard, or privileged state (constitution C8).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FRAMES = 4  # frozen: the strongest value in the registered 1-4 range
ACTION_CLASSES = 5  # idle/W/A/S/D
UPSAMPLE = 32


class ReactiveBaselineV1(nn.Module):
    """Small CNN/MLP reactive policy (T1-4). See module docstring."""

    def __init__(self, *, n_frames: int = N_FRAMES, seed: int = 0) -> None:
        super().__init__()
        self.n_frames = n_frames
        in_ch = 3 * n_frames
        self.conv1 = nn.Conv2d(in_ch, 24, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        flat = 48 * 4 * 4
        self.fc1 = nn.Linear(flat + ACTION_CLASSES, 128)
        self.fc2 = nn.Linear(128, ACTION_CLASSES)
        self._seed = seed
        self._reinit(seed)

    def _reinit(self, seed: int) -> None:
        g = torch.Generator()
        g.manual_seed(int(seed))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu", generator=g)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, generator=g)
                nn.init.zeros_(m.bias)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, frames: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        """``frames``: [B, n_frames, 3, 16, 16] in [0,1]. ``prev_action``: [B]
        int (0..4). Returns 5 action logits [B, 5]."""
        B = frames.size(0)
        x = frames.reshape(B, self.n_frames * 3, frames.size(-2), frames.size(-1))
        if x.size(-2) != UPSAMPLE:
            x = F.interpolate(x, size=(UPSAMPLE, UPSAMPLE), mode="nearest")
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.reshape(B, -1)
        prev = F.one_hot(prev_action.clamp(0, ACTION_CLASSES - 1).to(torch.long),
                         ACTION_CLASSES).to(x.dtype)
        x = torch.cat([x, prev], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


__all__ = ["ReactiveBaselineV1", "N_FRAMES", "ACTION_CLASSES", "UPSAMPLE"]
