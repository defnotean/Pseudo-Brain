"""Latent goal and intent conditioning for the Pseudo-Brain model.

Equips Pseudo-Brain with:
1. High-level intent representation (`[UNSTICK]`, `[EVADE]`, `[NAVIGATE]`, `[EXPLORE]`, `[NEUTRAL]`)
   that maps to the goal context tensor (tokens x core_width).
2. Supervised and discrete intent encoding during training.
3. Latent intent prediction from recurrent thought-field and belief representations
   for zero-shot test inference without cheating.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..types import HidKey


class IntentKind(IntEnum):
    """Canonical high-level intent categories for latent conditioning."""

    UNSTICK = 0
    EVADE = 1
    NAVIGATE = 2
    EXPLORE = 3
    NEUTRAL = 4

    @classmethod
    def from_string(cls, name: str) -> IntentKind:
        """Parse an intent name with or without bracket notation (e.g. '[UNSTICK]' or 'unstick')."""
        clean = name.strip().upper().removeprefix("[").removesuffix("]")
        try:
            return cls[clean]
        except KeyError:
            allowed = ", ".join(f"[{e.name}]" for e in cls)
            raise ValueError(f"unknown intent name: {name!r}; expected one of {allowed}")

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Return canonical bracketed names for all intents."""
        return tuple(f"[{e.name}]" for e in cls)

    @classmethod
    def count(cls) -> int:
        """Total number of canonical intents."""
        return len(cls)


NUM_INTENTS = len(IntentKind)


@dataclass(frozen=True, slots=True)
class LatentIntentPrediction:
    """Output contract for latent intent prediction from the thought field."""

    logits: Tensor
    probabilities: Tensor
    predicted_intent: Tensor
    attention_weights: Tensor | None = None


class IntentEncoder(nn.Module):
    """Maps discrete or continuous intent representations to goal_context tokens.

    Shape contract:
    - Input: Discrete intent IDs [batch] / [batch, 1] or soft intent vectors [batch, num_intents]
    - Output: goal_context tokens of shape [batch, tokens, width]
    """

    def __init__(
        self,
        *,
        width: int,
        tokens: int = 8,
        num_intents: int = NUM_INTENTS,
    ) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("width must be a positive integer")
        if tokens < 1:
            raise ValueError("tokens must be a positive integer")
        if num_intents < 1:
            raise ValueError("num_intents must be a positive integer")
        self.width = width
        self.tokens = tokens
        self.num_intents = num_intents
        self.intent_embeddings = nn.Parameter(
            torch.empty(num_intents, tokens, width)
        )
        self.norm = nn.LayerNorm(width)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.intent_embeddings, std=0.02)

    def encode_discrete(self, intent_ids: Tensor) -> Tensor:
        """Encode discrete integer intent IDs to goal context tokens."""
        if intent_ids.ndim == 0:
            intent_ids = intent_ids.unsqueeze(0)
        elif intent_ids.ndim == 2 and intent_ids.shape[1] == 1:
            intent_ids = intent_ids.squeeze(1)
        if intent_ids.ndim != 1:
            raise ValueError(
                f"discrete intent tensor must have shape [batch] or [batch, 1], got {tuple(intent_ids.shape)}"
            )
        if (intent_ids < 0).any() or (intent_ids >= self.num_intents).any():
            raise ValueError(f"intent IDs must be in range [0, {self.num_intents - 1}]")

        raw = self.intent_embeddings[intent_ids]
        return self.norm(raw)

    def encode_continuous(self, intent_vectors: Tensor) -> Tensor:
        """Encode continuous / soft intent vectors or probabilities to goal context tokens."""
        if intent_vectors.ndim == 3 and intent_vectors.shape[1] == 1:
            intent_vectors = intent_vectors.squeeze(1)
        elif intent_vectors.ndim == 1 and intent_vectors.shape[0] == self.num_intents:
            intent_vectors = intent_vectors.unsqueeze(0)
        if intent_vectors.ndim != 2 or intent_vectors.shape[1] != self.num_intents:
            raise ValueError(
                f"continuous intent tensor must have shape [batch, {self.num_intents}], got {tuple(intent_vectors.shape)}"
            )
        intent_vectors = intent_vectors.to(
            device=self.intent_embeddings.device,
            dtype=self.intent_embeddings.dtype,
        )
        raw = torch.einsum("bi,itd->btd", intent_vectors, self.intent_embeddings)
        return self.norm(raw)

    def forward(
        self,
        intents: Tensor | int | IntentKind | Sequence[int | IntentKind],
    ) -> Tensor:
        """Encode intents into goal_context tokens of shape [batch, tokens, width]."""
        if isinstance(intents, (int, IntentKind)):
            tensor_ids = torch.tensor(
                [int(intents)],
                device=self.intent_embeddings.device,
                dtype=torch.long,
            )
            return self.encode_discrete(tensor_ids)
        if isinstance(intents, (list, tuple)):
            tensor_ids = torch.tensor(
                [int(x) for x in intents],
                device=self.intent_embeddings.device,
                dtype=torch.long,
            )
            return self.encode_discrete(tensor_ids)
        if not isinstance(intents, Tensor):
            raise TypeError(
                f"unsupported intent input type: {type(intents).__name__}; expected Tensor, int, IntentKind, or Sequence"
            )

        if not intents.is_floating_point():
            return self.encode_discrete(intents.to(dtype=torch.long))
        return self.encode_continuous(intents)


class LatentIntentHead(nn.Module):
    """Predicts intent distribution from recurrent thought-field and belief representations.

    Uses cross-attention with a learned intent query attending to the unpooled
    cognitive state (thought registers, belief tokens, and optional sensors).
    """

    def __init__(
        self,
        *,
        width: int,
        num_intents: int = NUM_INTENTS,
        heads: int = 4,
        hidden_width: int | None = None,
    ) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("width must be a positive integer")
        if num_intents < 1:
            raise ValueError("num_intents must be a positive integer")
        if heads < 1:
            raise ValueError("heads must be a positive integer")
        if width % heads != 0:
            raise ValueError(f"width ({width}) must be divisible by heads ({heads})")

        self.width = width
        self.num_intents = num_intents
        self.heads = heads
        hidden = hidden_width if hidden_width is not None else width

        self.intent_query = nn.Parameter(torch.empty(1, 1, width))
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_intents),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.intent_query, std=0.02)

    def forward(
        self,
        thoughts: Tensor,
        belief: Tensor | None = None,
        *,
        sensors: Tensor | None = None,
        working_memory: Tensor | None = None,
    ) -> LatentIntentPrediction:
        """Extract and classify latent intent from the cognitive state.

        Args:
            thoughts: Thought field tensor [batch, thoughtlets, registers, width] or [batch, thoughtlets, width].
            belief: Optional belief state tensor [batch, belief_tokens, width].
            sensors: Optional sensory tokens [batch, sensor_tokens, width].
            working_memory: Optional working memory tokens [batch, memory_tokens, width].

        Returns:
            LatentIntentPrediction containing logits, probabilities, predicted_intent, and attention weights.
        """
        if thoughts.ndim == 4:
            batch, thoughtlets, registers, width = thoughts.shape
            flat_thoughts = thoughts.reshape(batch, thoughtlets * registers, width)
        elif thoughts.ndim == 3:
            batch, thoughtlets, width = thoughts.shape
            flat_thoughts = thoughts
        else:
            raise ValueError(f"thoughts must have 3 or 4 dimensions, got shape {tuple(thoughts.shape)}")

        if width != self.width:
            raise ValueError(f"expected tensor width {self.width}, got {width}")

        context_parts = [flat_thoughts]
        if belief is not None:
            if belief.ndim != 3 or belief.shape[0] != batch or belief.shape[2] != width:
                raise ValueError(
                    f"belief must have shape [{batch}, belief_tokens, {width}], got {tuple(belief.shape)}"
                )
            context_parts.append(belief)
        if sensors is not None:
            if sensors.ndim != 3 or sensors.shape[0] != batch or sensors.shape[2] != width:
                raise ValueError(
                    f"sensors must have shape [{batch}, sensor_tokens, {width}], got {tuple(sensors.shape)}"
                )
            context_parts.append(sensors)
        if working_memory is not None:
            if (
                working_memory.ndim != 3
                or working_memory.shape[0] != batch
                or working_memory.shape[2] != width
            ):
                raise ValueError(
                    f"working_memory must have shape [{batch}, memory_tokens, {width}], got {tuple(working_memory.shape)}"
                )
            context_parts.append(working_memory)

        context = torch.cat(context_parts, dim=1)
        query = self.intent_query.expand(batch, -1, -1).to(
            device=thoughts.device,
            dtype=thoughts.dtype,
        )

        attn_out, attn_weights = self.cross_attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=True,
            average_attn_weights=True,
        )
        features = attn_out.squeeze(1)
        logits = self.classifier(features)
        probabilities = F.softmax(logits, dim=-1)
        predicted_intent = logits.argmax(dim=-1)

        return LatentIntentPrediction(
            logits=logits,
            probabilities=probabilities,
            predicted_intent=predicted_intent,
            attention_weights=attn_weights,
        )

    def predict_intent(
        self,
        thoughts: Tensor,
        belief: Tensor | None = None,
        *,
        sensors: Tensor | None = None,
        working_memory: Tensor | None = None,
    ) -> Tensor:
        """Return categorical intent IDs [batch] in [0, num_intents - 1]."""
        return self.forward(
            thoughts,
            belief,
            sensors=sensors,
            working_memory=working_memory,
        ).predicted_intent

    def predict_probabilities(
        self,
        thoughts: Tensor,
        belief: Tensor | None = None,
        *,
        sensors: Tensor | None = None,
        working_memory: Tensor | None = None,
    ) -> Tensor:
        """Return normalized intent probability distribution [batch, num_intents]."""
        return self.forward(
            thoughts,
            belief,
            sensors=sensors,
            working_memory=working_memory,
        ).probabilities


class LatentIntentPredictor(nn.Module):
    """End-to-end latent intent module for zero-shot goal context generation.

    Routes internal thought-field and belief representations into goal_context
    tokens without requiring external teacher intent labels.
    """

    def __init__(
        self,
        *,
        width: int,
        tokens: int = 8,
        num_intents: int = NUM_INTENTS,
        heads: int = 4,
        hard_routing: bool = False,
    ) -> None:
        super().__init__()
        self.head = LatentIntentHead(
            width=width,
            num_intents=num_intents,
            heads=heads,
        )
        self.encoder = IntentEncoder(
            width=width,
            tokens=tokens,
            num_intents=num_intents,
        )
        self.hard_routing = hard_routing

    def forward(
        self,
        thoughts: Tensor,
        belief: Tensor | None = None,
        *,
        sensors: Tensor | None = None,
        working_memory: Tensor | None = None,
        hard: bool | None = None,
    ) -> tuple[Tensor, LatentIntentPrediction]:
        """Predict intent and generate goal_context tokens.

        Returns:
            Tuple of (goal_context [batch, tokens, width], prediction LatentIntentPrediction).
        """
        prediction = self.head(
            thoughts=thoughts,
            belief=belief,
            sensors=sensors,
            working_memory=working_memory,
        )
        use_hard = self.hard_routing if hard is None else hard
        if use_hard:
            goal_context = self.encoder(prediction.predicted_intent)
        else:
            goal_context = self.encoder(prediction.probabilities)
        return goal_context, prediction


class DirectionalAction(IntEnum):
    """Mutually-exclusive directional action options."""

    NONE = 0
    W = 1
    A = 2
    S = 3
    D = 4

    @property
    def hid_key(self) -> int | None:
        mapping = {
            DirectionalAction.NONE: None,
            DirectionalAction.W: int(HidKey.W),
            DirectionalAction.A: int(HidKey.A),
            DirectionalAction.S: int(HidKey.S),
            DirectionalAction.D: int(HidKey.D),
        }
        return mapping[self]


@dataclass(frozen=True, slots=True)
class DirectionalActionPrediction:
    """Output contract for the 5-way categorical directional action head."""

    logits: Tensor
    probabilities: Tensor
    action: Tensor
    wasd_binary: Tensor


class DirectionalActionHead(nn.Module):
    """5-way categorical projection for WASD movement.

    Mechanically eliminates simultaneous opposite-direction conflicts
    (W+S, A+D) by construction through categorical selection over
    {None, W, A, S, D}.
    """

    def __init__(
        self,
        *,
        width: int,
        hidden_width: int | None = None,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("width must be a positive integer")
        if num_classes != 5:
            raise ValueError("DirectionalActionHead requires exactly 5 classes (None, W, A, S, D)")
        self.width = width
        self.num_classes = num_classes
        hidden = hidden_width if hidden_width is not None else width
        self.projection = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, features: Tensor) -> DirectionalActionPrediction:
        """Compute directional logits, probabilities, categorical decision, and WASD binary tensor.

        Args:
            features: Input tensor [batch, width] or [batch, seq, width].
                      If 3D, feature mean pooling across queries/tokens is performed.
        """
        if features.ndim == 3:
            pooled = features.mean(dim=1)
        elif features.ndim == 2:
            pooled = features
        else:
            raise ValueError(
                f"features must have shape [batch, width] or [batch, seq, width], got {tuple(features.shape)}"
            )

        if pooled.shape[-1] != self.width:
            raise ValueError(f"expected feature width {self.width}, got {pooled.shape[-1]}")

        logits = self.projection(pooled)
        probabilities = F.softmax(logits, dim=-1)
        action = logits.argmax(dim=-1)
        wasd_binary = self.to_wasd_binary(logits)

        return DirectionalActionPrediction(
            logits=logits,
            probabilities=probabilities,
            action=action,
            wasd_binary=wasd_binary,
        )

    def to_wasd_binary(self, logits: Tensor) -> Tensor:
        """Convert 5-way logits to a 4-way [W, A, S, D] binary indicator tensor.

        Guaranteed invariants:
        1. sum(wasd_binary, dim=-1) <= 1
        2. W and S can NEVER both be 1 (wasd_binary[:, 0] & wasd_binary[:, 2] is always 0)
        3. A and D can NEVER both be 1 (wasd_binary[:, 1] & wasd_binary[:, 3] is always 0)
        """
        if logits.ndim != 2 or logits.shape[1] != 5:
            raise ValueError(f"logits must have shape [batch, 5], got {tuple(logits.shape)}")
        action = logits.argmax(dim=-1)
        batch = logits.shape[0]
        wasd = torch.zeros(batch, 4, dtype=logits.dtype, device=logits.device)
        mask_w = action == int(DirectionalAction.W)
        mask_a = action == int(DirectionalAction.A)
        mask_s = action == int(DirectionalAction.S)
        mask_d = action == int(DirectionalAction.D)
        wasd[mask_w, 0] = 1.0
        wasd[mask_a, 1] = 1.0
        wasd[mask_s, 2] = 1.0
        wasd[mask_d, 3] = 1.0
        return wasd

    def to_keyboard_tensor(self, logits: Tensor, total_keys: int = 256) -> Tensor:
        """Convert 5-way logits to full 256-key binary keyboard tensor."""
        if total_keys < 256:
            raise ValueError("total_keys must be at least 256")
        batch = logits.shape[0]
        wasd = self.to_wasd_binary(logits)
        keyboard = torch.zeros(batch, total_keys, dtype=logits.dtype, device=logits.device)
        keyboard[:, int(HidKey.W)] = wasd[:, 0]
        keyboard[:, int(HidKey.A)] = wasd[:, 1]
        keyboard[:, int(HidKey.S)] = wasd[:, 2]
        keyboard[:, int(HidKey.D)] = wasd[:, 3]
        return keyboard

    def sample_wasd(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> Tensor:
        """Sample 4-way [W, A, S, D] actions via Gumbel-Softmax on the 5-way logits."""
        if logits.ndim != 2 or logits.shape[1] != 5:
            raise ValueError(f"logits must have shape [batch, 5], got {tuple(logits.shape)}")
        sampled_5way = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        return sampled_5way[:, 1:5]


__all__ = [
    "DirectionalAction",
    "DirectionalActionHead",
    "DirectionalActionPrediction",
    "IntentEncoder",
    "IntentKind",
    "LatentIntentHead",
    "LatentIntentPrediction",
    "LatentIntentPredictor",
    "NUM_INTENTS",
]

