"""Structural Action Grouping module for Pseudo-Brain.

Formalizes structured action spaces over the canonical 307-channel generic HID
wire layout:
- Group A: Movement Direction (5-way or 9-way categorical Softmax: None, W, A, S, D, diagonals)
- Group B: Stance & Modifier Keys (Shift, Space, Ctrl, Alt, C, Tab, etc.) via Bernoulli probabilities
- Group C: Continuous Aim / Look Coordinates (mouse_dx, mouse_dy) with tanh bounding & deadzones
- Group D: Discrete Action Triggers (LMB, RMB, MMB, E, R, F, Q, etc.) via Bernoulli probabilities

Includes:
- StructuredActuatorHead: Modular neural readout head projecting unpooled thought-field
  states into structured action groups and recombining them deterministically into the
  canonical 307-channel GenericControl wire format.
- StructuredActionLoss: Multi-group loss combining categorical cross-entropy on movement,
  Bernoulli BCE on modifiers and triggers, and smooth L1/MSE on continuous axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..types import GenericControl, HidKey

CONTROL_VECTOR_SIZE = 307
KEYBOARD_CHANNELS = 256
MOUSE_BUTTONS_START = 256
MOUSE_BUTTONS_COUNT = 8
MOUSE_AXIS_DX = 264
MOUSE_AXIS_DY = 265
SCROLL_AXIS = 266
GAMEPAD_BUTTONS_START = 267
GAMEPAD_BUTTONS_COUNT = 32
GAMEPAD_AXES_START = 299
GAMEPAD_AXES_COUNT = 8

CONTINUOUS_DEADZONE_LIMIT = 0.05
# SQUASH_LIMIT is exactly representable in float32/bfloat16 (1.5 * 2**-5)
SQUASH_LIMIT = 0.046875

DEFAULT_MOVEMENT_KEYS: tuple[int, ...] = (
    int(HidKey.W),  # 26
    int(HidKey.A),  # 4
    int(HidKey.S),  # 22
    int(HidKey.D),  # 7
)

DEFAULT_MODIFIER_KEYS: tuple[int, ...] = (
    225,  # Left Shift
    int(HidKey.SPACE),  # 44 / Space
    224,  # Left Control
    226,  # Left Alt
    6,    # C (Crouch / Prone toggle)
    43,   # Tab
)

DEFAULT_TRIGGER_KEYS: tuple[int, ...] = (
    8,   # E (Interact / Use)
    21,  # R (Reload)
    9,   # F (Flashlight / Melee)
    20,  # Q (Special / Ability)
    10,  # G (Grenade)
    30,  # 1 (Weapon 1)
    31,  # 2 (Weapon 2)
    32,  # 3 (Weapon 3)
)

DEFAULT_TRIGGER_MOUSE_BUTTONS: tuple[int, ...] = (
    0,  # LMB (Primary Fire)
    1,  # RMB (Secondary / ADS)
    2,  # MMB (Tertiary / Ping)
)


class MovementMode(str, Enum):
    """Supported movement discretization schemes."""

    FIVE_WAY = "5way"
    NINE_WAY = "9way"


class MovementDirection5(IntEnum):
    """5-way categorical movement direction (Group A)."""

    NONE = 0
    W = 1
    A = 2
    S = 3
    D = 4

    @property
    def keys(self) -> tuple[int, ...]:
        mapping = {
            MovementDirection5.NONE: (),
            MovementDirection5.W: (int(HidKey.W),),
            MovementDirection5.A: (int(HidKey.A),),
            MovementDirection5.S: (int(HidKey.S),),
            MovementDirection5.D: (int(HidKey.D),),
        }
        return mapping[self]

    @property
    def name_lower(self) -> str:
        return self.name.lower()

    @classmethod
    def from_keys(
        cls,
        keys: Sequence[int],
        movement_keys: tuple[int, ...] = DEFAULT_MOVEMENT_KEYS,
    ) -> MovementDirection5:
        """Map active keys to 5-way movement direction."""
        w, a, s, d = movement_keys
        k_set = set(keys)
        has_w = w in k_set
        has_a = a in k_set
        has_s = s in k_set
        has_d = d in k_set

        if has_w and not (has_s or has_a or has_d):
            return cls.W
        if has_a and not (has_d or has_w or has_s):
            return cls.A
        if has_s and not (has_w or has_a or has_d):
            return cls.S
        if has_d and not (has_a or has_w or has_s):
            return cls.D
        return cls.NONE


class MovementDirection9(IntEnum):
    """9-way categorical movement direction with diagonal support (Group A)."""

    NONE = 0
    W = 1
    A = 2
    S = 3
    D = 4
    WA = 5  # North-West / Up-Left
    WD = 6  # North-East / Up-Right
    SA = 7  # South-West / Down-Left
    SD = 8  # South-East / Down-Right

    @property
    def keys(self) -> tuple[int, ...]:
        mapping = {
            MovementDirection9.NONE: (),
            MovementDirection9.W: (int(HidKey.W),),
            MovementDirection9.A: (int(HidKey.A),),
            MovementDirection9.S: (int(HidKey.S),),
            MovementDirection9.D: (int(HidKey.D),),
            MovementDirection9.WA: (int(HidKey.W), int(HidKey.A)),
            MovementDirection9.WD: (int(HidKey.W), int(HidKey.D)),
            MovementDirection9.SA: (int(HidKey.S), int(HidKey.A)),
            MovementDirection9.SD: (int(HidKey.S), int(HidKey.D)),
        }
        return mapping[self]

    @property
    def name_lower(self) -> str:
        return self.name.lower()

    @classmethod
    def from_keys(
        cls,
        keys: Sequence[int],
        movement_keys: tuple[int, ...] = DEFAULT_MOVEMENT_KEYS,
    ) -> MovementDirection9:
        """Map active keys to 9-way movement direction."""
        w, a, s, d = movement_keys
        k_set = set(keys)
        has_w = w in k_set
        has_a = a in k_set
        has_s = s in k_set
        has_d = d in k_set

        if has_w and has_a and not (has_s or has_d):
            return cls.WA
        if has_w and has_d and not (has_s or has_a):
            return cls.WD
        if has_s and has_a and not (has_w or has_d):
            return cls.SA
        if has_s and has_d and not (has_w or has_a):
            return cls.SD
        if has_w and not (has_s or has_a or has_d):
            return cls.W
        if has_a and not (has_d or has_w or has_s):
            return cls.A
        if has_s and not (has_w or has_a or has_d):
            return cls.S
        if has_d and not (has_a or has_w or has_s):
            return cls.D
        return cls.NONE


@dataclass(frozen=True, slots=True)
class StructuredActionSpec:
    """Specification of structured action groups mapped over the 307-channel wire layout."""

    width: int
    movement_mode: MovementMode | str = MovementMode.FIVE_WAY
    movement_keys: tuple[int, ...] = DEFAULT_MOVEMENT_KEYS
    modifier_keys: tuple[int, ...] = DEFAULT_MODIFIER_KEYS
    trigger_keys: tuple[int, ...] = DEFAULT_TRIGGER_KEYS
    trigger_mouse_buttons: tuple[int, ...] = DEFAULT_TRIGGER_MOUSE_BUTTONS
    include_scroll: bool = True
    include_gamepad_axes: bool = False
    continuous_deadzone: float = CONTINUOUS_DEADZONE_LIMIT
    continuous_squash: str = "deadzone_tanh"
    max_mouse_aim: float = 1.0
    scale_mouse_aim: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.width, int) or self.width < 1:
            raise ValueError("width must be a positive integer")

        mode = self.movement_mode
        if isinstance(mode, str):
            if mode not in {m.value for m in MovementMode}:
                raise ValueError(f"unknown movement_mode: {mode!r}; expected '5way' or '9way'")
            object.__setattr__(self, "movement_mode", MovementMode(mode))
        elif not isinstance(mode, MovementMode):
            raise ValueError("movement_mode must be MovementMode or str")

        if len(self.movement_keys) != 4:
            raise ValueError("movement_keys must have exactly 4 keys (W, A, S, D)")
        for key in self.movement_keys:
            if not isinstance(key, int) or key < 0 or key >= KEYBOARD_CHANNELS:
                raise ValueError(f"movement key {key} must be an integer in [0, 255]")
        if len(set(self.movement_keys)) != 4:
            raise ValueError("movement_keys must be unique")

        for key in self.modifier_keys:
            if not isinstance(key, int) or key < 0 or key >= KEYBOARD_CHANNELS:
                raise ValueError(f"modifier key {key} must be an integer in [0, 255]")
        if len(set(self.modifier_keys)) != len(self.modifier_keys):
            raise ValueError("modifier_keys must be unique")

        for key in self.trigger_keys:
            if not isinstance(key, int) or key < 0 or key >= KEYBOARD_CHANNELS:
                raise ValueError(f"trigger key {key} must be an integer in [0, 255]")
        if len(set(self.trigger_keys)) != len(self.trigger_keys):
            raise ValueError("trigger_keys must be unique")

        for btn in self.trigger_mouse_buttons:
            if not isinstance(btn, int) or btn < 0 or btn >= MOUSE_BUTTONS_COUNT:
                raise ValueError(f"mouse button {btn} must be an integer in [0, 7]")
        if len(set(self.trigger_mouse_buttons)) != len(self.trigger_mouse_buttons):
            raise ValueError("trigger_mouse_buttons must be unique")

        # Disjoint keyboard key sets
        mov_set = set(self.movement_keys)
        mod_set = set(self.modifier_keys)
        trig_set = set(self.trigger_keys)
        if mov_set & mod_set:
            raise ValueError(f"movement and modifier keys overlap: {mov_set & mod_set}")
        if mov_set & trig_set:
            raise ValueError(f"movement and trigger keys overlap: {mov_set & trig_set}")
        if mod_set & trig_set:
            raise ValueError(f"modifier and trigger keys overlap: {mod_set & trig_set}")

        if self.continuous_squash not in {"none", "tanh", "deadzone_tanh", "hard", "smooth"}:
            raise ValueError(
                f"unsupported continuous_squash: {self.continuous_squash!r}; "
                "expected 'none', 'tanh', 'deadzone_tanh', 'hard', or 'smooth'"
            )
        if self.continuous_deadzone < 0.0:
            raise ValueError("continuous_deadzone cannot be negative")
        if self.max_mouse_aim <= 0.0:
            raise ValueError("max_mouse_aim must be positive")
        if self.scale_mouse_aim <= 0.0:
            raise ValueError("scale_mouse_aim must be positive")

    @property
    def num_movement_classes(self) -> int:
        return 5 if self.movement_mode == MovementMode.FIVE_WAY else 9

    @property
    def num_modifiers(self) -> int:
        return len(self.modifier_keys)

    @property
    def num_triggers(self) -> int:
        return len(self.trigger_keys) + len(self.trigger_mouse_buttons)

    @property
    def num_continuous(self) -> int:
        count = 2  # mouse_dx, mouse_dy
        if self.include_scroll:
            count += 1
        if self.include_gamepad_axes:
            count += 8
        return count


@dataclass(frozen=True, slots=True)
class StructuredActionPrediction:
    """Output container for structured action groups and reconstructed wire control."""

    movement_logits: Tensor
    movement_probs: Tensor
    movement_action: Tensor
    modifier_logits: Tensor
    modifier_probs: Tensor
    trigger_logits: Tensor
    trigger_probs: Tensor
    mouse_aim: Tensor
    scroll: Tensor | None
    gamepad_axes: Tensor | None
    control_vector: Tensor
    query_features: Tensor | None = None
    state_attention: Tensor | None = None
    thought_attention: Tensor | None = None


class StructuredActuatorHead(nn.Module):
    """Modular neural readout head for structured action prediction and wire recombination."""

    def __init__(
        self,
        *,
        width: int,
        spec: StructuredActionSpec | None = None,
        heads: int = 4,
        hidden_width: int | None = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.spec = spec if spec is not None else StructuredActionSpec(width=width)
        self.heads = heads
        hidden = hidden_width if hidden_width is not None else width

        # 4 specialized group queries: Movement, Modifier, Aim, Trigger
        self.group_query_names = ("movement", "modifier", "aim", "trigger")
        self.num_group_queries = len(self.group_query_names)
        self.group_queries = nn.Parameter(torch.empty(1, self.num_group_queries, width))

        # Unpooled cross-attention readout
        self.state_norm = nn.LayerNorm(width)
        self.query_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.coordination = nn.MultiheadAttention(width, heads, batch_first=True)
        self.coordination_norm = nn.LayerNorm(width)

        # Group A: Movement Direction (Categorical)
        self.movement_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.spec.num_movement_classes),
        )

        # Group B: Stance & Modifier Keys (Bernoulli)
        self.modifier_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.spec.num_modifiers),
        )

        # Group C: Continuous Aim / Look Coordinates
        self.aim_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.spec.num_continuous),
        )

        # Group D: Discrete Action Triggers (Bernoulli)
        self.trigger_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.spec.num_triggers),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.group_queries, std=0.02)

    def forward(
        self,
        features: Tensor | None = None,
        *,
        sensors: Tensor | None = None,
        belief: Tensor | None = None,
        thoughts: Tensor | None = None,
        working_memory: Tensor | None = None,
        retrieved_memory: Tensor | None = None,
        goal_context: Tensor | None = None,
        threshold: float = 0.5,
    ) -> StructuredActionPrediction:
        """Forward pass projecting unpooled state tokens into structured action groups."""
        state_tokens: list[Tensor] = []
        thought_start = 0
        thought_count = 0
        thoughtlets = 0
        registers = 0

        if sensors is not None:
            state_tokens.append(sensors)
        if belief is not None:
            state_tokens.append(belief)
        if thoughts is not None:
            thought_start = sum(t.shape[1] for t in state_tokens)
            thoughtlets = thoughts.shape[1]
            registers = thoughts.shape[2]
            thought_count = thoughtlets * registers
            state_tokens.append(thoughts.flatten(1, 2))
        if working_memory is not None:
            state_tokens.append(working_memory)
        if retrieved_memory is not None:
            state_tokens.append(retrieved_memory.flatten(1, 2))
        if goal_context is not None:
            state_tokens.append(goal_context)

        state_attention: Tensor | None = None
        thought_attention: Tensor | None = None
        group_feats: Tensor | None = None

        if state_tokens:
            complete_state = torch.cat(state_tokens, dim=1)
            batch = complete_state.shape[0]
            queries = self.group_queries.expand(batch, -1, -1).to(dtype=complete_state.dtype)
            attended, state_attention = self.cross_attention(
                self.query_norm(queries),
                self.state_norm(complete_state),
                self.state_norm(complete_state),
                need_weights=True,
                average_attn_weights=True,
            )
            coordinated, _ = self.coordination(attended, attended, attended, need_weights=False)
            group_feats = self.coordination_norm(attended + coordinated)

            feat_movement = group_feats[:, 0]
            feat_modifier = group_feats[:, 1]
            feat_aim = group_feats[:, 2]
            feat_trigger = group_feats[:, 3]

            if thought_count > 0 and state_attention is not None:
                th_attn = state_attention[
                    :, :, thought_start : thought_start + thought_count
                ].reshape(batch, self.num_group_queries, thoughtlets, registers)
                thought_attention = th_attn.sum(dim=-1)

        elif features is not None:
            if features.ndim == 3:
                complete_state = features
                batch = complete_state.shape[0]
                queries = self.group_queries.expand(batch, -1, -1).to(dtype=complete_state.dtype)
                attended, state_attention = self.cross_attention(
                    self.query_norm(queries),
                    self.state_norm(complete_state),
                    self.state_norm(complete_state),
                    need_weights=True,
                    average_attn_weights=True,
                )
                coordinated, _ = self.coordination(attended, attended, attended, need_weights=False)
                group_feats = self.coordination_norm(attended + coordinated)

                feat_movement = group_feats[:, 0]
                feat_modifier = group_feats[:, 1]
                feat_aim = group_feats[:, 2]
                feat_trigger = group_feats[:, 3]
            elif features.ndim == 2:
                batch = features.shape[0]
                feat_movement = features
                feat_modifier = features
                feat_aim = features
                feat_trigger = features
                group_feats = None
            else:
                raise ValueError(
                    f"features must have shape [batch, width] or [batch, seq, width], got {tuple(features.shape)}"
                )
        else:
            raise ValueError(
                "StructuredActuatorHead requires unpooled state components or features tensor"
            )

        # Projections
        movement_logits = self.movement_head(feat_movement)
        movement_probs = F.softmax(movement_logits, dim=-1)
        movement_action = movement_logits.argmax(dim=-1)

        modifier_logits = self.modifier_head(feat_modifier)
        modifier_probs = torch.sigmoid(modifier_logits)

        trigger_logits = self.trigger_head(feat_trigger)
        trigger_probs = torch.sigmoid(trigger_logits)

        raw_continuous = self.aim_head(feat_aim)
        mouse_aim, scroll, gamepad_axes = self._process_continuous(raw_continuous)

        control_vector = self.reconstruct_control_vector(
            movement_action=movement_action,
            modifier_probs=modifier_probs,
            trigger_probs=trigger_probs,
            mouse_aim=mouse_aim,
            scroll=scroll,
            gamepad_axes=gamepad_axes,
            threshold=threshold,
        )

        return StructuredActionPrediction(
            movement_logits=movement_logits,
            movement_probs=movement_probs,
            movement_action=movement_action,
            modifier_logits=modifier_logits,
            modifier_probs=modifier_probs,
            trigger_logits=trigger_logits,
            trigger_probs=trigger_probs,
            mouse_aim=mouse_aim,
            scroll=scroll,
            gamepad_axes=gamepad_axes,
            control_vector=control_vector,
            query_features=group_feats,
            state_attention=state_attention,
            thought_attention=thought_attention,
        )

    def _process_continuous(
        self, raw: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Apply bounding and deadzone processing to continuous outputs."""
        squash = self.spec.continuous_squash
        raw_aim = raw[:, 0:2]

        if squash == "deadzone_tanh":
            mouse_aim = SQUASH_LIMIT * torch.tanh(raw_aim / SQUASH_LIMIT)
        elif squash == "tanh":
            mouse_aim = self.spec.max_mouse_aim * torch.tanh(
                raw_aim / self.spec.scale_mouse_aim
            )
        elif squash == "hard":
            t_aim = self.spec.max_mouse_aim * torch.tanh(
                raw_aim / self.spec.scale_mouse_aim
            )
            mask = t_aim.abs() <= self.spec.continuous_deadzone
            mouse_aim = t_aim.masked_fill(mask, 0.0)
        elif squash == "smooth":
            t_aim = self.spec.max_mouse_aim * torch.tanh(
                raw_aim / self.spec.scale_mouse_aim
            )
            excess = (t_aim.abs() - self.spec.continuous_deadzone).clamp_min(0.0)
            mouse_aim = torch.sign(t_aim) * excess
        else:  # none
            mouse_aim = raw_aim

        scroll: Tensor | None = None
        gamepad_axes: Tensor | None = None
        idx = 2

        if self.spec.include_scroll:
            raw_scroll = raw[:, idx : idx + 1]
            if squash == "deadzone_tanh":
                scroll = SQUASH_LIMIT * torch.tanh(raw_scroll / SQUASH_LIMIT)
            else:
                scroll = torch.tanh(raw_scroll)
            idx += 1

        if self.spec.include_gamepad_axes:
            raw_gp = raw[:, idx : idx + 8]
            if squash == "deadzone_tanh":
                gamepad_axes = SQUASH_LIMIT * torch.tanh(raw_gp / SQUASH_LIMIT)
            else:
                gamepad_axes = torch.tanh(raw_gp)

        return mouse_aim, scroll, gamepad_axes

    def reconstruct_control_vector(
        self,
        movement_action: Tensor,
        modifier_probs: Tensor,
        trigger_probs: Tensor,
        mouse_aim: Tensor,
        scroll: Tensor | None = None,
        gamepad_axes: Tensor | None = None,
        threshold: float = 0.5,
    ) -> Tensor:
        """Reconstruct the canonical 307-channel wire control tensor deterministically."""
        batch = movement_action.shape[0]
        device = movement_action.device
        dtype = mouse_aim.dtype
        ctrl = torch.zeros(batch, CONTROL_VECTOR_SIZE, dtype=dtype, device=device)

        # 1. Group A: Movement Direction
        w_key, a_key, s_key, d_key = self.spec.movement_keys
        if self.spec.movement_mode == MovementMode.FIVE_WAY:
            ctrl[movement_action == int(MovementDirection5.W), w_key] = 1.0
            ctrl[movement_action == int(MovementDirection5.A), a_key] = 1.0
            ctrl[movement_action == int(MovementDirection5.S), s_key] = 1.0
            ctrl[movement_action == int(MovementDirection5.D), d_key] = 1.0
        else:  # 9-way
            ctrl[movement_action == int(MovementDirection9.W), w_key] = 1.0
            ctrl[movement_action == int(MovementDirection9.A), a_key] = 1.0
            ctrl[movement_action == int(MovementDirection9.S), s_key] = 1.0
            ctrl[movement_action == int(MovementDirection9.D), d_key] = 1.0

            mask_wa = movement_action == int(MovementDirection9.WA)
            ctrl[mask_wa, w_key] = 1.0
            ctrl[mask_wa, a_key] = 1.0

            mask_wd = movement_action == int(MovementDirection9.WD)
            ctrl[mask_wd, w_key] = 1.0
            ctrl[mask_wd, d_key] = 1.0

            mask_sa = movement_action == int(MovementDirection9.SA)
            ctrl[mask_sa, s_key] = 1.0
            ctrl[mask_sa, a_key] = 1.0

            mask_sd = movement_action == int(MovementDirection9.SD)
            ctrl[mask_sd, s_key] = 1.0
            ctrl[mask_sd, d_key] = 1.0

        # 2. Group B: Stance & Modifier Keys
        for idx, key in enumerate(self.spec.modifier_keys):
            active = modifier_probs[:, idx] > threshold
            ctrl[active, key] = 1.0

        # 3. Group D: Discrete Action Triggers
        num_trig_keys = len(self.spec.trigger_keys)
        for idx, key in enumerate(self.spec.trigger_keys):
            active = trigger_probs[:, idx] > threshold
            ctrl[active, key] = 1.0

        for btn_idx, btn in enumerate(self.spec.trigger_mouse_buttons):
            active = trigger_probs[:, num_trig_keys + btn_idx] > threshold
            ctrl[active, MOUSE_BUTTONS_START + btn] = 1.0

        # 4. Group C: Continuous Coordinates
        ctrl[:, MOUSE_AXIS_DX] = mouse_aim[:, 0]
        ctrl[:, MOUSE_AXIS_DY] = mouse_aim[:, 1]
        if scroll is not None:
            ctrl[:, SCROLL_AXIS] = scroll[:, 0]
        if gamepad_axes is not None:
            ctrl[:, GAMEPAD_AXES_START : GAMEPAD_AXES_START + 8] = gamepad_axes

        return ctrl

    def to_generic_controls(
        self,
        prediction: StructuredActionPrediction,
        threshold: float = 0.5,
        impulse_sequence: int = 1,
    ) -> tuple[GenericControl, ...]:
        """Convert batch predictions into high-level GenericControl instances."""
        ctrl_vecs = prediction.control_vector.detach()
        batch = ctrl_vecs.shape[0]
        results: list[GenericControl] = []

        for i in range(batch):
            row = ctrl_vecs[i]
            # Keyboard keys [0, 255]
            keys_down = tuple(
                k for k in range(KEYBOARD_CHANNELS) if float(row[k]) > threshold
            )
            # Mouse buttons [256, 263]
            mouse_buttons = tuple(
                b
                for b in range(MOUSE_BUTTONS_COUNT)
                if float(row[MOUSE_BUTTONS_START + b]) > threshold
            )
            # Mouse deltas & wheel
            mouse_dx = float(row[MOUSE_AXIS_DX])
            mouse_dy = float(row[MOUSE_AXIS_DY])
            mouse_wheel = float(row[SCROLL_AXIS])

            # Gamepad buttons [267, 298]
            gamepad_buttons = tuple(
                gb
                for gb in range(GAMEPAD_BUTTONS_COUNT)
                if float(row[GAMEPAD_BUTTONS_START + gb]) > threshold
            )
            # Gamepad axes [299, 306]
            if self.spec.include_gamepad_axes and prediction.gamepad_axes is not None:
                gamepad_axes = tuple(
                    float(row[GAMEPAD_AXES_START + ax])
                    for ax in range(GAMEPAD_AXES_COUNT)
                )
            else:
                gamepad_axes = ()

            # Impulse sequence rule: nonzero relative mouse requires sequence > 0
            seq = impulse_sequence
            if (mouse_dx != 0.0 or mouse_dy != 0.0 or mouse_wheel != 0.0) and seq == 0:
                seq = 1

            results.append(
                GenericControl(
                    keys_down=keys_down,
                    mouse_dx=mouse_dx,
                    mouse_dy=mouse_dy,
                    mouse_buttons=mouse_buttons,
                    mouse_wheel=mouse_wheel,
                    gamepad_axes=gamepad_axes,
                    gamepad_buttons=gamepad_buttons,
                    impulse_sequence=seq,
                )
            )
        return tuple(results)

    def to_generic_control(
        self,
        prediction: StructuredActionPrediction,
        index: int = 0,
        threshold: float = 0.5,
        impulse_sequence: int = 1,
    ) -> GenericControl:
        """Convert a single prediction from the batch into a GenericControl."""
        controls = self.to_generic_controls(
            prediction, threshold=threshold, impulse_sequence=impulse_sequence
        )
        return controls[index]

    def sample_actions(
        self,
        prediction: StructuredActionPrediction,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> dict[str, Tensor]:
        """Sample actions using Gumbel-Softmax for movement and Bernoulli for discrete triggers."""
        # Movement sampling (guarantees mutual exclusion)
        movement_sample = F.gumbel_softmax(
            prediction.movement_logits, tau=temperature, hard=hard
        )
        movement_action = movement_sample.argmax(dim=-1)

        # Modifier Bernoulli sampling
        mod_probs = prediction.modifier_probs
        mod_sample = (torch.rand_like(mod_probs) < mod_probs).float()

        # Trigger Bernoulli sampling
        trig_probs = prediction.trigger_probs
        trig_sample = (torch.rand_like(trig_probs) < trig_probs).float()

        return {
            "movement_action": movement_action,
            "movement_sample": movement_sample,
            "modifier_sample": mod_sample,
            "trigger_sample": trig_sample,
            "mouse_aim": prediction.mouse_aim,
        }


@dataclass(frozen=True, slots=True)
class StructuredLossOutput:
    """Output container for structured action loss computation."""

    loss: Tensor
    metrics: Mapping[str, Tensor]
    samples: int


class StructuredActionLoss(nn.Module):
    """Multi-group action loss combining categorical CE, Bernoulli BCE, and Continuous MSE/L1."""

    def __init__(
        self,
        spec: StructuredActionSpec | None = None,
        *,
        movement_weight: float = 1.0,
        modifier_weight: float = 1.0,
        trigger_weight: float = 1.0,
        continuous_weight: float = 0.5,
        continuous_loss_fn: str = "smooth_l1",
        deadzone_hinge_weight: float = 0.0,
        deadzone_hinge_margin: float = 0.04,
    ) -> None:
        super().__init__()
        self.spec = spec if spec is not None else StructuredActionSpec(width=16)
        if movement_weight < 0.0:
            raise ValueError("movement_weight cannot be negative")
        if modifier_weight < 0.0:
            raise ValueError("modifier_weight cannot be negative")
        if trigger_weight < 0.0:
            raise ValueError("trigger_weight cannot be negative")
        if continuous_weight < 0.0:
            raise ValueError("continuous_weight cannot be negative")
        if continuous_loss_fn not in {"smooth_l1", "mse", "l1"}:
            raise ValueError(f"unsupported continuous_loss_fn: {continuous_loss_fn!r}")
        if deadzone_hinge_weight < 0.0:
            raise ValueError("deadzone_hinge_weight cannot be negative")
        if not (0.0 <= deadzone_hinge_margin < CONTINUOUS_DEADZONE_LIMIT):
            raise ValueError(
                f"deadzone_hinge_margin must be in [0.0, {CONTINUOUS_DEADZONE_LIMIT})"
            )

        self.movement_weight = float(movement_weight)
        self.modifier_weight = float(modifier_weight)
        self.trigger_weight = float(trigger_weight)
        self.continuous_weight = float(continuous_weight)
        self.continuous_loss_fn = continuous_loss_fn
        self.deadzone_hinge_weight = float(deadzone_hinge_weight)
        self.deadzone_hinge_margin = float(deadzone_hinge_margin)

    def extract_targets(
        self, target_vectors: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
        """Extract structured group target tensors from canonical 307-channel target vectors."""
        if target_vectors.ndim != 2 or target_vectors.shape[1] != CONTROL_VECTOR_SIZE:
            raise ValueError(
                f"target_vectors must have shape [batch, {CONTROL_VECTOR_SIZE}], got {tuple(target_vectors.shape)}"
            )

        batch = target_vectors.shape[0]
        device = target_vectors.device
        w_k, a_k, s_k, d_k = self.spec.movement_keys

        w = target_vectors[:, w_k] > 0.5
        a = target_vectors[:, a_k] > 0.5
        s = target_vectors[:, s_k] > 0.5
        d = target_vectors[:, d_k] > 0.5

        movement_target = torch.zeros(batch, dtype=torch.long, device=device)
        if self.spec.movement_mode == MovementMode.FIVE_WAY:
            movement_target[w & ~s & ~a & ~d] = int(MovementDirection5.W)
            movement_target[a & ~d & ~w & ~s] = int(MovementDirection5.A)
            movement_target[s & ~w & ~a & ~d] = int(MovementDirection5.S)
            movement_target[d & ~a & ~w & ~s] = int(MovementDirection5.D)
        else:  # 9-way
            movement_target[w & ~s & ~a & ~d] = int(MovementDirection9.W)
            movement_target[a & ~d & ~w & ~s] = int(MovementDirection9.A)
            movement_target[s & ~w & ~a & ~d] = int(MovementDirection9.S)
            movement_target[d & ~a & ~w & ~s] = int(MovementDirection9.D)
            movement_target[w & a & ~s & ~d] = int(MovementDirection9.WA)
            movement_target[w & d & ~s & ~a] = int(MovementDirection9.WD)
            movement_target[s & a & ~w & ~d] = int(MovementDirection9.SA)
            movement_target[s & d & ~w & ~a] = int(MovementDirection9.SD)

        # Modifiers
        modifier_target = target_vectors[:, list(self.spec.modifier_keys)]

        # Triggers
        trig_key_targets = target_vectors[:, list(self.spec.trigger_keys)]
        trig_mouse_targets = target_vectors[
            :, [MOUSE_BUTTONS_START + b for b in self.spec.trigger_mouse_buttons]
        ]
        trigger_target = torch.cat((trig_key_targets, trig_mouse_targets), dim=1)

        # Continuous
        aim_target = target_vectors[:, [MOUSE_AXIS_DX, MOUSE_AXIS_DY]]
        scroll_target = (
            target_vectors[:, SCROLL_AXIS : SCROLL_AXIS + 1]
            if self.spec.include_scroll
            else None
        )
        gamepad_axes_target = (
            target_vectors[:, GAMEPAD_AXES_START : GAMEPAD_AXES_START + 8]
            if self.spec.include_gamepad_axes
            else None
        )

        return (
            movement_target,
            modifier_target,
            trigger_target,
            aim_target,
            scroll_target,
            gamepad_axes_target,
        )

    def forward(
        self,
        prediction: StructuredActionPrediction,
        target: Tensor | GenericControl | Sequence[GenericControl],
    ) -> StructuredLossOutput:
        """Compute combined loss across all structured action groups."""
        if isinstance(target, GenericControl):
            from ..training.batches import control_to_vector
            target_vectors = prediction.movement_logits.new_tensor(
                [control_to_vector(target)]
            )
        elif isinstance(target, (list, tuple)) and all(
            isinstance(x, GenericControl) for x in target
        ):
            from ..training.batches import control_to_vector
            target_vectors = prediction.movement_logits.new_tensor(
                [control_to_vector(x) for x in target]
            )
        elif isinstance(target, Tensor):
            target_vectors = target
        else:
            raise ValueError("target must be a Tensor, GenericControl, or Sequence[GenericControl]")

        (
            movement_target,
            modifier_target,
            trigger_target,
            aim_target,
            scroll_target,
            gamepad_axes_target,
        ) = self.extract_targets(target_vectors)

        batch = target_vectors.shape[0]

        # 1. Group A: Movement Categorical Loss
        movement_loss = F.cross_entropy(prediction.movement_logits, movement_target)
        movement_acc = (prediction.movement_action == movement_target).float().mean()

        # 2. Group B: Stance & Modifier Bernoulli Loss
        modifier_loss = F.binary_cross_entropy_with_logits(
            prediction.modifier_logits, modifier_target
        )

        # 3. Group D: Discrete Triggers Bernoulli Loss
        trigger_loss = F.binary_cross_entropy_with_logits(
            prediction.trigger_logits, trigger_target
        )

        # 4. Group C: Continuous Loss
        if self.continuous_loss_fn == "smooth_l1":
            aim_loss = F.smooth_l1_loss(prediction.mouse_aim, aim_target)
        elif self.continuous_loss_fn == "mse":
            aim_loss = F.mse_loss(prediction.mouse_aim, aim_target)
        else:  # l1
            aim_loss = F.l1_loss(prediction.mouse_aim, aim_target)

        continuous_loss = aim_loss
        if (
            self.spec.include_scroll
            and prediction.scroll is not None
            and scroll_target is not None
        ):
            if self.continuous_loss_fn == "smooth_l1":
                scroll_loss = F.smooth_l1_loss(prediction.scroll, scroll_target)
            else:
                scroll_loss = F.mse_loss(prediction.scroll, scroll_target)
            continuous_loss = continuous_loss + scroll_loss

        if (
            self.spec.include_gamepad_axes
            and prediction.gamepad_axes is not None
            and gamepad_axes_target is not None
        ):
            if self.continuous_loss_fn == "smooth_l1":
                gp_loss = F.smooth_l1_loss(prediction.gamepad_axes, gamepad_axes_target)
            else:
                gp_loss = F.mse_loss(prediction.gamepad_axes, gamepad_axes_target)
            continuous_loss = continuous_loss + gp_loss

        # 5. Optional Deadzone Hinge Loss
        deadzone_hinge_loss = prediction.movement_logits.new_zeros(())
        if self.deadzone_hinge_weight > 0.0:
            zero_aim_mask = (aim_target == 0.0).all(dim=-1)
            if zero_aim_mask.any():
                quiescent_aim = prediction.mouse_aim[zero_aim_mask]
                hinge = (quiescent_aim.abs() - self.deadzone_hinge_margin).clamp_min(0.0)
                deadzone_hinge_loss = hinge.mean()

        total_loss = (
            self.movement_weight * movement_loss
            + self.modifier_weight * modifier_loss
            + self.trigger_weight * trigger_loss
            + self.continuous_weight * continuous_loss
            + self.deadzone_hinge_weight * deadzone_hinge_loss
        )

        metrics = {
            "loss": total_loss.detach(),
            "movement_loss": movement_loss.detach(),
            "movement_accuracy": movement_acc.detach(),
            "modifier_loss": modifier_loss.detach(),
            "trigger_loss": trigger_loss.detach(),
            "continuous_loss": continuous_loss.detach(),
            "deadzone_hinge_loss": deadzone_hinge_loss.detach(),
        }

        return StructuredLossOutput(loss=total_loss, metrics=metrics, samples=batch)


__all__ = [
    "DEFAULT_MODIFIER_KEYS",
    "DEFAULT_MOVEMENT_KEYS",
    "DEFAULT_TRIGGER_KEYS",
    "DEFAULT_TRIGGER_MOUSE_BUTTONS",
    "MovementDirection5",
    "MovementDirection9",
    "MovementMode",
    "StructuredActionLoss",
    "StructuredActionPrediction",
    "StructuredActionSpec",
    "StructuredActuatorHead",
    "StructuredLossOutput",
]
