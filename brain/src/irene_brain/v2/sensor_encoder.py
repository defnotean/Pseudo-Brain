"""Sensor Encoder V2 — vectorized visual encoder.

Supports [B,C,H,W] and [B,T,C,H,W] -> [B,T,W] batch encoding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CoreV2Config


class SensorEncoderV2(nn.Module):
    """Efficient CNN encoder for 32x32 RGB observations.

    Vectorized over time: [B,T,C,H,W] -> [B*T,C,H,W] -> CNN -> [B,T,W]
    """

    def __init__(self, config: CoreV2Config):
        super().__init__()
        self.config = config
        W = config.width
        C = config.encoder_channels

        # For 32x32 input with stride 2:
        # Conv1: 32x32 -> 16x16 (kernel 3, stride 2, padding 1)
        # Pool: 16x16 -> 8x8
        # Conv2: 8x8 -> 4x4
        # Conv3: 4x4 -> 2x2
        # Flatten: 2*2*C = 4C -> Linear -> W

        self.conv1 = nn.Conv2d(3, C, kernel_size=config.encoder_kernel,
                               stride=config.encoder_stride, padding=1)
        self.bn1 = nn.BatchNorm2d(C)
        self.pool = nn.MaxPool2d(2, 2) if config.encoder_pool else nn.Identity()
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(C)
        self.conv3 = nn.Conv2d(C, C, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(C)

        # Compute flattened size
        self._flattened_size = C * 2 * 2  # 32 -> 16 -> 8 -> 4 -> 2
        self.fc = nn.Linear(self._flattened_size, W)

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observation(s) to latent.

        Args:
            x: [B, C, H, W] or [B, T, C, H, W] in [0,1] range
        Returns:
            [B, W] or [B, T, W]
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            time_dim = True
        else:
            time_dim = False

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.flatten(1)
        x = self.fc(x)

        if time_dim:
            x = x.reshape(B, T, -1)
        return x