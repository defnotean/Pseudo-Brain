"""Spatial pixel tokenization for the trainable thought-field model."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class PixelEncoder(nn.Module):
    """A small CNN that preserves a grid of tokens instead of global pooling."""

    def __init__(self, *, width: int, sensor_tokens: int) -> None:
        super().__init__()
        grid = math.isqrt(sensor_tokens)
        if grid * grid != sensor_tokens:
            raise ValueError("sensor_tokens must be a perfect square for PixelEncoder")
        if width < 16:
            raise ValueError("width must be at least 16")

        first = max(16, width // 4)
        second = max(32, width // 2)
        self.sensor_tokens = sensor_tokens
        self.grid_size = grid
        self.stem = nn.Sequential(
            nn.Conv2d(3, first, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(1, first),
            nn.SiLU(),
            nn.Conv2d(first, second, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, second),
            nn.SiLU(),
            nn.Conv2d(second, width, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, width),
            nn.SiLU(),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.position = nn.Parameter(torch.empty(1, sensor_tokens, width))
        self.output_norm = nn.LayerNorm(width)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, pixels: Tensor) -> Tensor:
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("pixels must have shape [batch, 3, height, width]")
        if not pixels.is_floating_point():
            raise ValueError("pixels must be a floating-point tensor")
        if pixels.shape[-2] < self.grid_size or pixels.shape[-1] < self.grid_size:
            raise ValueError("pixel dimensions are smaller than the sensor token grid")

        features = self.stem(pixels)
        # AdaptiveAvgPool2d backward is intentionally rejected by strict CUDA
        # determinism. Exact-grid configurations need no resampling, so keep
        # the mathematically identical identity path differentiable instead.
        target_grid = (self.grid_size, self.grid_size)
        if tuple(features.shape[-2:]) != target_grid:
            features = self.spatial_pool(features)
        tokens = features.flatten(2).transpose(1, 2)
        return self.output_norm(tokens + self.position.to(dtype=tokens.dtype))
