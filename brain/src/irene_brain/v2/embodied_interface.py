"""EmbodiedInterfaceV1 — the versioned common model-output record.

Realtime-embodied-qualification v3 (2026-08-25), Section 2, freezes one
common model-output record that every candidate must emit on every tick,
independently of the environment it is running in.  Core V2 as trained is
**not** a Q0 candidate: its single five-class idle/W/A/S/D decision head has
no look, cursor, button, or hotbar outputs, and nothing in the tree today
exposes the common record.  This module closes exactly that gap, and nothing
else:

- :class:`EmbodiedInterfaceRecord` is the frozen per-tick record schema
  (Section 2 field table, values frozen verbatim);
- :mod:`irene_brain.v2.embodied_adapter` holds the stateless, format-only
  translation of one record into ordinary key/mouse events, with a
  documented allow-list of actuated keys;
- :class:`EmbodiedInterfaceV1` composes a frozen Core V2 core with
  supervised heads that emit the record every tick from the core's
  head-input context: the current frame's encoded latent concatenated
  with the recurrent post-belief world state.  The heads are the *only*
  new parameters; the core is always stop-gradient and byte-preserved
  (v3 Sections 3 and 8.1).

Causal-integrity rules encoded here (v3 Section 2, Section 8.1):

- No environment token, task label, reward, hazard, coordinate, map, or
  history input may reach the heads.  ``emit``/``emit_batch`` accept only
  (frame, core state, previous factual action id) and nothing else.
- A command lasts exactly one tick; adapters are stateless and carry no
  memory between calls, so holding a key requires re-emitting the command
  every tick.
- Any richer action vocabulary, added field, or changed value set is a
  new interface version requiring a new preregistration — this module is
  frozen as v1 and its identity string pins that.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

import torch

from .core import CoreV2Model

# --- Frozen schema constants (v3 Section 2, verbatim) ----------------------

LOCOMOTION = ("NOOP", "FORWARD", "BACKWARD", "LEFT", "RIGHT")
LOOK_YAW_DEG = (-30, -15, -5, 0, 5, 15, 30)
LOOK_PITCH_DEG = (-30, -15, -5, 0, 5, 15, 30)
CURSOR_DX_PX = (-64, -16, 0, 16, 64)
CURSOR_DY_PX = (-64, -16, 0, 16, 64)
BUTTON_NAMES = ("jump", "crouch", "sprint", "attack", "use", "inventory")
HOTBAR = ("HOLD", "1", "2", "3", "4", "5", "6", "7", "8", "9")

LOCOMOTION_INDEX = {name: i for i, name in enumerate(LOCOMOTION)}
YAW_INDEX = {d: i for i, d in enumerate(LOOK_YAW_DEG)}
PITCH_INDEX = {d: i for i, d in enumerate(LOOK_PITCH_DEG)}
CURSOR_X_INDEX = {p: i for i, p in enumerate(CURSOR_DX_PX)}
CURSOR_Y_INDEX = {p: i for i, p in enumerate(CURSOR_DY_PX)}
HOTBAR_INDEX = {s: i for i, s in enumerate(HOTBAR)}
BUTTON_INDEX = {name: i for i, name in enumerate(BUTTON_NAMES)}

RECORD_FIELDS = (
    "locomotion",
    "look_yaw_deg",
    "look_pitch_deg",
    "cursor_dx_px",
    "cursor_dy_px",
    "buttons",
    "hotbar",
)


def _require_one_of(value: object, allowed: tuple, name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class EmbodiedInterfaceRecord:
    """One tick of the common model-output record (v3 Section 2)."""

    locomotion: str = "NOOP"
    look_yaw_deg: int = 0
    look_pitch_deg: int = 0
    cursor_dx_px: int = 0
    cursor_dy_px: int = 0
    buttons: tuple[bool, ...] = (False,) * len(BUTTON_NAMES)
    hotbar: str = "HOLD"

    def __post_init__(self) -> None:
        _require_one_of(self.locomotion, LOCOMOTION, "locomotion")
        _require_one_of(self.look_yaw_deg, LOOK_YAW_DEG, "look_yaw_deg")
        _require_one_of(self.look_pitch_deg, LOOK_PITCH_DEG, "look_pitch_deg")
        _require_one_of(self.cursor_dx_px, CURSOR_DX_PX, "cursor_dx_px")
        _require_one_of(self.cursor_dy_px, CURSOR_DY_PX, "cursor_dy_px")
        if not isinstance(self.buttons, tuple) or len(self.buttons) != len(
            BUTTON_NAMES
        ):
            raise ValueError(
                f"buttons must be a {len(BUTTON_NAMES)}-tuple of booleans"
            )
        for value in self.buttons:
            _require_bool(value, "buttons item")
        _require_one_of(self.hotbar, HOTBAR, "hotbar")

    @classmethod
    def neutral(cls) -> "EmbodiedInterfaceRecord":
        """The all-default record: no movement, centered view, no cursor
        move, no buttons, HOLD hotbar."""
        return cls()

    @property
    def is_neutral(self) -> bool:
        return self == EmbodiedInterfaceRecord.neutral()

    def to_dict(self) -> dict[str, object]:
        return {
            "buttons": [bool(b) for b in self.buttons],
            "cursor_dx_px": self.cursor_dx_px,
            "cursor_dy_px": self.cursor_dy_px,
            "hotbar": self.hotbar,
            "locomotion": self.locomotion,
            "look_pitch_deg": self.look_pitch_deg,
            "look_yaw_deg": self.look_yaw_deg,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EmbodiedInterfaceRecord":
        """Strict wire deserializer: exact key set, no repairs."""
        if not isinstance(value, Mapping):
            raise ValueError("record must be an object")
        unknown = set(value) - set(RECORD_FIELDS)
        if unknown:
            raise ValueError(f"unknown record fields: {sorted(unknown)}")
        missing = set(RECORD_FIELDS) - set(value)
        if missing:
            raise ValueError(f"missing record fields: {sorted(missing)}")
        buttons = value["buttons"]
        if not isinstance(buttons, (list, tuple)) or len(buttons) != len(
            BUTTON_NAMES
        ):
            raise ValueError("buttons must be a 6-item list/tuple")
        return cls(
            locomotion=str(value["locomotion"]),
            look_yaw_deg=int(value["look_yaw_deg"]),
            look_pitch_deg=int(value["look_pitch_deg"]),
            cursor_dx_px=int(value["cursor_dx_px"]),
            cursor_dy_px=int(value["cursor_dy_px"]),
            buttons=tuple(bool(b) for b in buttons),
            hotbar=str(value["hotbar"]),
        )

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


# --- Stateless format-only adapters (v3 Section 2, paragraph 3) -----------
#
# An adapter may only resize/normalize pixels (the pixel side is owned by
# the caller's frame convention), serialize the common record, translate it
# to ordinary key/mouse events, and ignore fields the environment physically
# lacks.  It must not observe rewards, hazards, privileged state, maps,
# coordinates, or task labels; choose an action; persist state; or carry any
# memory between ticks.  ``translate_record`` below is that translation,
# frozen and hashed into the interface identity.

LOCOMOTION_KEYS: Mapping[str, int] = {
    "NOOP": -1,  # no key; the sentinel is not a HID usage
    "FORWARD": 0x1A,  # W
    "BACKWARD": 0x16,  # S
    "LEFT": 0x04,  # A
    "RIGHT": 0x07,  # D
}
BUTTON_KEYS: Mapping[str, int] = {
    "jump": 0x2C,  # SPACE
    "crouch": 0x52,  # UP
    "sprint": 0x50,  # LEFT arrow
    "attack": 0x4F,  # RIGHT arrow
    "use": 0x51,  # DOWN
    "inventory": -1,  # not actuated by V1 (no HID key in the frozen allow-list)
}
# Keys the V1 frozen adapter may ever actuate.  Anything outside this set is
# a new interface version.
ACTUATED_KEY_SET: frozenset[int] = frozenset(
    {v for v in LOCOMOTION_KEYS.values() if v >= 0}
    | {v for v in BUTTON_KEYS.values() if v >= 0}
)


def translate_record(record: EmbodiedInterfaceRecord) -> tuple[int, ...]:
    """Stateless, format-only translation of one tick record to actuated
    HID keys (sorted, deduplicated).

    Frozen mapping conventions (part of the interface identity):

    - locomotion: W/S/A/D keys per ``LOCOMOTION_KEYS``; NOOP presses none.
    - look: yaw degrees become a horizontal mouse impulse
      (``mouse_dx = look_yaw_deg``), pitch degrees become a vertical mouse
      impulse (``mouse_dy = -look_pitch_deg``); positive yaw drags the view
      right, negative pitch drags the view up.  The environment consumes
      the impulses through the ordinary HID path; the adapter keeps no
      state between calls.
    - cursor: ``mouse_dx += cursor_dx_px``, ``mouse_dy += cursor_dy_px``.
    - buttons: the mapped keys from ``BUTTON_KEYS`` (``inventory`` is
      ignored by V1).
    - hotbar: no key in the V1 frozen allow-list; ignored (the record still
      carries it, for environments that do actuate digits).

    The mapping is a pure function of the record: calling it twice with the
    same record always yields the same output, and it reads nothing else.
    """
    keys: set[int] = set()
    locomotion_key = LOCOMOTION_KEYS[record.locomotion]
    if locomotion_key >= 0:
        keys.add(locomotion_key)
    for name, key in BUTTON_KEYS.items():
        if record.buttons[BUTTON_INDEX[name]] and key >= 0:
            keys.add(key)
    return tuple(sorted(keys))


def mouse_deltas_for_record(record: EmbodiedInterfaceRecord) -> tuple[int, int]:
    """The (dx, dy) pixel impulse implied by one record (see
    ``translate_record`` conventions)."""
    dx = record.look_yaw_deg + record.cursor_dx_px
    dy = -record.look_pitch_deg + record.cursor_dy_px
    return dx, dy


# --- The interface module --------------------------------------------------


class EmbodiedInterfaceV1:
    """Versioned interface: frozen Core V2 core + record-emitting heads.

    The heads are a ``nn.ParameterDict`` mapping the core's head-input
    context ``[B, 2W]`` — the current frame's encoded latent concatenated
    with the recurrent belief — to per-field logits, then a frozen decode
    maps logits onto the Section-2 value sets.  The core is always
    stop-gradient (``belief.detach()`` / ``latent.detach()``); these heads
    are the only parameters the qualification candidate adds, and a
    byte-preserved core is part of the candidate's identity.

    Why ``latent + belief`` and not belief alone: the frozen Core V2 belief
    is a slow world-state tracker — on fast-changing observation streams its
    per-dimension variance collapses toward zero, so belief-only heads
    cannot carry the current frame's fast sensorimotor signal (measured:
    belief readout ~chance on the Q0.6 synthetic pixel->record task, while
    the current-frame latent readout is near-perfect).  Including the
    frame's encoded latent is fully consistent with the v3 causal-integrity
    contract: the interface already receives the frame, the latent is
    derived from it by the frozen core encoder, and no privileged
    environment/reward/hazard/coordinate information reaches the heads.
    """

    IDENTITY = "irene.brain.embodied_interface.v1"
    VERSION = 1

    def __init__(
        self,
        core: CoreV2Model,
        heads: torch.nn.ParameterDict | None = None,
        init_seed: int = 0,
    ) -> None:
        if not isinstance(core, CoreV2Model):
            raise ValueError("core must be a CoreV2Model")
        if isinstance(init_seed, bool) or not isinstance(init_seed, int):
            raise ValueError("init_seed must be an integer")
        self.core = core
        width = core.config.width
        in_dim = 2 * width  # cat([latent, belief])
        if heads is None:
            heads = torch.nn.ParameterDict()
            generator = torch.Generator(device="cpu")
            # Process-stable seed (CRC32, not the salted builtin hash): the
            # same seed in any process derives the same initial heads.
            generator.manual_seed(zlib_crc32(
                f"{self.IDENTITY}:{init_seed}".encode("utf-8")
            ))
            for field_name in RECORD_FIELDS:
                size = {
                    "locomotion": len(LOCOMOTION),
                    "look_yaw_deg": len(LOOK_YAW_DEG),
                    "look_pitch_deg": len(LOOK_PITCH_DEG),
                    "cursor_dx_px": len(CURSOR_DX_PX),
                    "cursor_dy_px": len(CURSOR_DY_PX),
                    "buttons": len(BUTTON_NAMES),
                    "hotbar": len(HOTBAR),
                }[field_name]
                value = torch.zeros(in_dim, size, device="cpu")
                value.add_(
                    torch.randn(
                        (in_dim, size), generator=generator, device="cpu"
                    ).mul_(0.01)
                )
                heads[field_name] = torch.nn.Parameter(value)
        else:
            if not isinstance(heads, torch.nn.ParameterDict):
                raise ValueError("heads must be an nn.ParameterDict")
            missing = [f for f in RECORD_FIELDS if f not in heads]
            if missing:
                raise ValueError(f"heads missing fields: {missing}")
            for field_name in RECORD_FIELDS:
                if heads[field_name].shape[0] != in_dim:
                    raise ValueError(
                        f"heads[{field_name}] must have in-dim {in_dim} "
                        f"(cat of latent and belief), got "
                        f"{heads[field_name].shape[0]}"
                    )
        self.heads = heads

    def heads_sha256(self) -> str:
        """Process-stable SHA-256 of the head parameters (canonical JSON,
        sorted keys; CRC32-seeded init, so two processes constructing the
        same interface from the same seed hash identically)."""
        payload = {
            field: heads.flatten().tolist()
            for field, heads in self.heads.items()
        }
        blob = json.dumps(payload, sort_keys=True, allow_nan=False)
        return sha256(blob.encode("utf-8")).hexdigest()

    @torch.no_grad()
    def emit(self, frame, state, prev_action: int | None = 0) -> tuple[EmbodiedInterfaceRecord, object]:
        """One core forward + head decode for a single frame.

        ``frame`` is a ``RgbFrame`` (32x32 RGB); ``state`` is the live
        ``CoreV2State``; ``prev_action`` is the factual previous action
        class (0..4, or ``None`` for models that do not condition on it).
        Returns the common record for this tick and the core's post-tick
        state.  The heads are stateless: they add no state of their own,
        so the returned state is exactly the core's.
        """
        from ..training.objective import _rgb_tensor

        device = next(self.core.parameters()).device
        pixels = _rgb_tensor((frame,), device=device, resolution=(32, 32))
        prev = (
            None
            if prev_action is None
            else torch.tensor([prev_action], dtype=torch.long, device=device)
        )
        output, state = self.core(pixels, state, prev_action=prev)
        context = self.head_context(output)
        records = self._decode_from_context(context)
        return records[0], state

    def head_context(self, output) -> torch.Tensor:
        """The head-input context for one tick: ``[latent, belief]``.

        Both halves are stop-gradient by construction: the latent is the
        frozen encoder's output on the current frame and the belief is the
        core's recurrent world state; the heads read them but can never
        push gradients back into the core.
        """
        latent = output.latent.detach()
        belief = output.belief.detach()
        return torch.cat([latent, belief], dim=-1)

    @torch.no_grad()
    def emit_batch(
        self, frames, state, prev_actions=None
    ) -> tuple[list[EmbodiedInterfaceRecord], object]:
        """Vectorized over a batch of frames (the K-dimension-batching
        discipline applies to the thoughtlets inside the core; here the
        batch dimension is the ordinary model batch)."""
        from ..training.objective import _rgb_tensor

        device = next(self.core.parameters()).device
        pixels = _rgb_tensor(frames, device=device, resolution=(32, 32))
        prev = (
            None
            if prev_actions is None
            else torch.as_tensor(prev_actions, dtype=torch.long, device=device)
        )
        output, state = self.core(pixels, state, prev_action=prev)
        context = self.head_context(output)
        records = self._decode_from_context(context)
        return records, state

    def _decode_from_context(
        self, context: torch.Tensor
    ) -> list[EmbodiedInterfaceRecord]:
        B = context.shape[0]
        logits = {
            field: context @ self.heads[field] for field in RECORD_FIELDS
        }
        records = []
        for b in range(B):
            loc = int(logits["locomotion"][b].argmax())
            yaw = LOOK_YAW_DEG[int(logits["look_yaw_deg"][b].argmax())]
            pitch = LOOK_PITCH_DEG[int(logits["look_pitch_deg"][b].argmax())]
            cx = CURSOR_DX_PX[int(logits["cursor_dx_px"][b].argmax())]
            cy = CURSOR_DY_PX[int(logits["cursor_dy_px"][b].argmax())]
            buttons = tuple(
                bool(logits["buttons"][b][i] > 0.0)
                for i in range(len(BUTTON_NAMES))
            )
            hotbar = HOTBAR[int(logits["hotbar"][b].argmax())]
            records.append(
                EmbodiedInterfaceRecord(
                    locomotion=LOCOMOTION[loc],
                    look_yaw_deg=yaw,
                    look_pitch_deg=pitch,
                    cursor_dx_px=cx,
                    cursor_dy_px=cy,
                    buttons=buttons,
                    hotbar=hotbar,
                )
            )
        return records


def zlib_crc32(text: bytes) -> int:
    """Process-stable 32-bit hash (stdlib zlib; unlike the builtin ``hash``
    it is NOT salted by ``PYTHONHASHSEED``).  Used for seed derivation so
    that interface construction is deterministic across processes."""
    import zlib

    return zlib.crc32(text) & 0xFFFFFFFF


__all__ = [
    "LOCOMOTION",
    "LOOK_YAW_DEG",
    "LOOK_PITCH_DEG",
    "CURSOR_DX_PX",
    "CURSOR_DY_PX",
    "BUTTON_NAMES",
    "HOTBAR",
    "LOCOMOTION_INDEX",
    "YAW_INDEX",
    "PITCH_INDEX",
    "CURSOR_X_INDEX",
    "CURSOR_Y_INDEX",
    "HOTBAR_INDEX",
    "BUTTON_INDEX",
    "RECORD_FIELDS",
    "EmbodiedInterfaceRecord",
    "LOCOMOTION_KEYS",
    "BUTTON_KEYS",
    "ACTUATED_KEY_SET",
    "translate_record",
    "mouse_deltas_for_record",
    "EmbodiedInterfaceV1",
    "zlib_crc32",
]
