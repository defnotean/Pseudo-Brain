"""FlyMerge V1 model — fast innate channel + persistent selection state + modulatory scalar.

See brain/docs/preregistrations/2026-08-29-flymerge-v1.md for the
preregistered architecture and experimental protocol.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from irene_brain.v2.reactive_baseline import ReactiveBaselineV1, N_FRAMES, ACTION_CLASSES


class FlyMergeV1(nn.Module):
    """Model C: ReactiveBaselineV1 trunk + GRU persistent state + innate channel.

    Architecture (prereg §2):
    - Trunk: shared with ReactiveBaselineV1 (same conv layers, same param budget)
    - GRU persistent state h in R^32, updated every tick
    - Innate channel I: 1x1 conv over trunk features -> scalar urgency u in [0,1]
    - Shunting inhibition: h' = (1-g(u))*GRU(h,f) + g(u)*h, g(u)=sigmoid(a*u+b)
    - Modulatory scalar: urgency scales logit temperature (gain modulation)
    - Bypass logit: innate channel's action contribution enters logits additively
    """

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__()
        self.reactive_trunk = ReactiveBaselineV1(seed=seed)
        # Freeze trunk params; only the new heads and GRU are trained
        for p in self.reactive_trunk.parameters():
            p.requires_grad_(False)

        # GRU persistent state: h in R^32
        self.gru = nn.GRUCell(
            input_size=self.reactive_trunk.fc2.out_features + ACTION_CLASSES,
            hidden_size=32,
        )
        self.h_proj = nn.Linear(32, self.reactive_trunk.fc2.out_features)

        # Innate channel: 1x1 conv over trunk features (48*4*4=768 flattened)
        # Maps to a single urgency scalar u in [0,1]
        self.innate_channel = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Shunting gate parameters (learned scalars a, b in g(u)=sigmoid(a*u+b))
        self.gate_a = nn.Parameter(torch.tensor(1.0))
        self.gate_b = nn.Parameter(torch.tensor(0.0))

        # Temperature modulation: urgency scales the logit temperature
        self.temp_scale = nn.Parameter(torch.tensor(1.0))

        # Bypass logit: innate channel contribution enters logits additively
        # The innate channel produces a scalar; we project it to 5 action logits
        self.bypass_proj = nn.Linear(1, ACTION_CLASSES)

        self._seed = seed

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        frames: torch.Tensor,
        prev_action: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            frames: [B, N_FRAMES, 3, 16, 16] in [0,1]
            prev_action: [B] long (0..4)
            h: [B, 32] GRU hidden state, or None (zero-init)

        Returns:
            logits: [B, ACTION_CLASSES]
            h_new: [B, 32] updated GRU state
            u: [B, 1] urgency scalar
        """
        B = frames.size(0)

        # --- Trunk features ---
        x = frames.reshape(B, N_FRAMES * 3, frames.size(-2), frames.size(-1))
        if x.size(-2) != 32:
            x = F.interpolate(x, size=(32, 32), mode="nearest")
        # Run through trunk conv layers (same as ReactiveBaselineV1)
        x = F.relu(self.reactive_trunk.bn1(self.reactive_trunk.conv1(x)))
        x = self.reactive_trunk.pool(x)
        x = F.relu(self.reactive_trunk.bn2(self.reactive_trunk.conv2(x)))
        x = F.relu(self.reactive_trunk.bn3(self.reactive_trunk.conv3(x)))
        # x is now [B, 48, 4, 4]

        trunk_flat = x.reshape(B, -1)  # [B, 768]

        # --- Trunk output (frozen, no grad) ---
        with torch.no_grad():
            prev_onehot = F.one_hot(
                prev_action.clamp(0, ACTION_CLASSES - 1).to(torch.long),
                ACTION_CLASSES,
            ).to(trunk_flat.dtype)
            trunk_out = self.reactive_trunk.fc2(
                F.relu(self.reactive_trunk.fc1(
                    torch.cat([trunk_flat, prev_onehot], dim=1)
                ))
            )  # [B, 5]

        # --- Innate channel (urgency) ---
        u = self.innate_channel(x)  # [B, 1]

        # --- GRU persistent state ---
        if h is None:
            h = torch.zeros(B, 32, device=frames.device, dtype=trunk_flat.dtype)
        gru_input = torch.cat([trunk_out, prev_onehot], dim=1)  # [B, 5+5=10]
        h_new = self.gru(gru_input, h)  # [B, 32]

        # --- Shunting inhibition ---
        g = torch.sigmoid(self.gate_a * u + self.gate_b)  # [B, 1]
        h_bar = (1 - g) * h_new + g * h  # freeze persistent state under high urgency

        # --- Modulatory scalar: urgency scales logit temperature ---
        temperature = 1.0 + (self.temp_scale - 1.0) * u  # modulated by urgency

        # --- Compute logits from modulated persistent state ---
        h_bar_proj = self.h_proj(h_bar)  # [B, 5]
        logits = h_bar_proj / temperature.clamp(min=0.01)

        # --- Bypass logit (innate channel action contribution) ---
        bypass = self.bypass_proj(u)  # [B, 5]
        logits = logits + bypass

        return logits, h_new, u