"""Generic, game-independent sensor and actuator contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from hashlib import sha256
from math import isfinite
from struct import pack
from typing import Any, Mapping


UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1


MODEL_OBSERVATION_FIELDS = frozenset(
    {
        "frame_id",
        "elapsed_ns",
        "observation_age_ns",
        "rgb",
        "previous_control",
        "audio_pcm_s16le",
        "text_inputs",
        "dropped_frames",
    }
)
"""The exact top-level inputs accepted by the shared model boundary."""


class HidKey(IntEnum):
    """USB HID keyboard usage IDs used by the deterministic environments."""

    A = 0x04
    D = 0x07
    S = 0x16
    W = 0x1A
    SPACE = 0x2C
    RIGHT = 0x4F
    LEFT = 0x50
    DOWN = 0x51
    UP = 0x52


def _canonical_int_tuple(
    values: tuple[int, ...],
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple")
    canonical = tuple(
        sorted(set(_require_int(value, name=f"{name} item") for value in values))
    )
    if any(value < minimum or value > maximum for value in canonical):
        raise ValueError(f"{name} values must be between {minimum} and {maximum}")
    if len(canonical) > UINT32_MAX:
        raise ValueError(f"{name} cannot contain more than {UINT32_MAX} values")
    return canonical


def _reject_unknown_fields(value: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown fields: {names}")


def _require_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_uint(value: Any, *, name: str, maximum: int = UINT64_MAX) -> int:
    result = _require_int(value, name=name)
    if result < 0 or result > maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}")
    return result


def _require_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _length_prefixed(value: bytes) -> bytes:
    if len(value) > UINT64_MAX:
        raise ValueError("byte sequence is too large for the uint64 wire length")
    return pack(">Q", len(value)) + value


def _float64(value: float) -> bytes:
    normalized = 0.0 if value == 0.0 else value
    return pack(">d", normalized)


@dataclass(frozen=True, slots=True)
class GenericControl:
    """A game-agnostic snapshot of ordinary human input devices.

    Keys, buttons, and gamepad axes are persistent states. Mouse deltas and the
    wheel are exactly-once relative impulses identified by impulse_sequence.
    Direct construction canonicalizes the semantically unordered key/button
    sets by sorting and de-duplicating them. Serialized input is stricter:
    :meth:`from_dict` rejects duplicates instead of repairing an invalid wire
    record.

    Gamepad axes are either omitted (the compact neutral representation) or a
    complete vector of exactly eight normalized axes. This keeps one stable
    actuator dimension without making neutral controls verbose.

    For example, a game adapter may allow HID key W, but the model is never
    told that W means move forward.
    """

    keys_down: tuple[int, ...] = ()
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    mouse_buttons: tuple[int, ...] = ()
    mouse_wheel: float = 0.0
    gamepad_axes: tuple[float, ...] = ()
    gamepad_buttons: tuple[int, ...] = ()
    impulse_sequence: int = 0
    intended_hold_ns: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "keys_down",
            _canonical_int_tuple(
                self.keys_down,
                name="keys_down",
                minimum=0,
                maximum=255,
            ),
        )
        object.__setattr__(
            self,
            "mouse_buttons",
            _canonical_int_tuple(
                self.mouse_buttons,
                name="mouse_buttons",
                minimum=0,
                maximum=7,
            ),
        )
        object.__setattr__(
            self,
            "gamepad_buttons",
            _canonical_int_tuple(
                self.gamepad_buttons,
                name="gamepad_buttons",
                minimum=0,
                maximum=31,
            ),
        )
        mouse_dx = _require_number(self.mouse_dx, name="mouse_dx")
        mouse_dy = _require_number(self.mouse_dy, name="mouse_dy")
        mouse_wheel = _require_number(self.mouse_wheel, name="mouse_wheel")
        if not isinstance(self.gamepad_axes, tuple):
            raise ValueError("gamepad_axes must be a tuple")
        gamepad_axes = tuple(
            _require_number(value, name="gamepad_axes item") for value in self.gamepad_axes
        )
        object.__setattr__(self, "mouse_dx", mouse_dx)
        object.__setattr__(self, "mouse_dy", mouse_dy)
        object.__setattr__(self, "mouse_wheel", mouse_wheel)
        object.__setattr__(self, "gamepad_axes", gamepad_axes)
        if len(gamepad_axes) not in (0, 8):
            raise ValueError("gamepad_axes must be empty or contain exactly 8 axes")
        if any(value < -1.0 or value > 1.0 for value in gamepad_axes):
            raise ValueError("gamepad axes must be normalized to [-1, 1]")
        _require_uint(self.impulse_sequence, name="impulse_sequence")
        if (
            mouse_dx != 0.0 or mouse_dy != 0.0 or mouse_wheel != 0.0
        ) and self.impulse_sequence == 0:
            raise ValueError("relative mouse/wheel impulses require a nonzero impulse_sequence")
        _require_uint(self.intended_hold_ns, name="intended_hold_ns")

    @classmethod
    def neutral(cls) -> GenericControl:
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "keys_down": list(self.keys_down),
            "mouse_dx": self.mouse_dx,
            "mouse_dy": self.mouse_dy,
            "mouse_buttons": list(self.mouse_buttons),
            "mouse_wheel": self.mouse_wheel,
            "gamepad_axes": list(self.gamepad_axes),
            "gamepad_buttons": list(self.gamepad_buttons),
            "impulse_sequence": self.impulse_sequence,
            "intended_hold_ns": self.intended_hold_ns,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GenericControl:
        if not isinstance(value, Mapping):
            raise ValueError("control must be an object")
        allowed = frozenset(
            {
                "keys_down",
                "mouse_dx",
                "mouse_dy",
                "mouse_buttons",
                "mouse_wheel",
                "gamepad_axes",
                "gamepad_buttons",
                "impulse_sequence",
                "intended_hold_ns",
            }
        )
        _reject_unknown_fields(value, allowed)

        def integer_tuple(name: str) -> tuple[int, ...]:
            raw = value.get(name, ())
            if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple)):
                raise ValueError(f"{name} must be a list or tuple")
            result = tuple(_require_int(item, name=f"{name} item") for item in raw)
            if len(result) != len(set(result)):
                raise ValueError(f"{name} cannot contain duplicates")
            return result

        raw_axes = value.get("gamepad_axes", ())
        if isinstance(raw_axes, (str, bytes)) or not isinstance(raw_axes, (list, tuple)):
            raise ValueError("gamepad_axes must be a list or tuple")
        return cls(
            keys_down=integer_tuple("keys_down"),
            mouse_dx=_require_number(value.get("mouse_dx", 0.0), name="mouse_dx"),
            mouse_dy=_require_number(value.get("mouse_dy", 0.0), name="mouse_dy"),
            mouse_buttons=integer_tuple("mouse_buttons"),
            mouse_wheel=_require_number(value.get("mouse_wheel", 0.0), name="mouse_wheel"),
            gamepad_axes=tuple(
                _require_number(item, name="gamepad_axes item") for item in raw_axes
            ),
            gamepad_buttons=integer_tuple("gamepad_buttons"),
            impulse_sequence=_require_int(
                value.get("impulse_sequence", 0),
                name="impulse_sequence",
            ),
            intended_hold_ns=_require_int(
                value.get("intended_hold_ns", 0),
                name="intended_hold_ns",
            ),
        )

    def canonical_bytes(self) -> bytes:
        """Versioned, process-independent encoding for replay identity."""

        result = bytearray(b"IRCTL\x01")
        for values in (self.keys_down, self.mouse_buttons, self.gamepad_buttons):
            result.extend(pack(">I", len(values)))
            for value in values:
                result.extend(pack(">I", value))
        result.extend(_float64(self.mouse_dx))
        result.extend(_float64(self.mouse_dy))
        result.extend(_float64(self.mouse_wheel))
        result.extend(pack(">I", len(self.gamepad_axes)))
        for value in self.gamepad_axes:
            result.extend(_float64(float(value)))
        result.extend(pack(">Q", self.impulse_sequence))
        result.extend(pack(">Q", self.intended_hold_ns))
        return bytes(result)

    @property
    def has_relative_impulse(self) -> bool:
        return self.mouse_dx != 0.0 or self.mouse_dy != 0.0 or self.mouse_wheel != 0.0

    def without_relative_impulse(self) -> GenericControl:
        """Keep persistent device state while consuming relative input once."""

        if not self.has_relative_impulse:
            return self
        return GenericControl(
            keys_down=self.keys_down,
            mouse_buttons=self.mouse_buttons,
            gamepad_axes=self.gamepad_axes,
            gamepad_buttons=self.gamepad_buttons,
            impulse_sequence=self.impulse_sequence,
            intended_hold_ns=self.intended_hold_ns,
        )


@dataclass(frozen=True, slots=True)
class RgbFrame:
    """Packed row-major 8-bit RGB frame."""

    width: int
    height: int
    pixels: bytes = field(repr=False)

    def __post_init__(self) -> None:
        width = _require_uint(self.width, name="width", maximum=UINT32_MAX)
        height = _require_uint(self.height, name="height", maximum=UINT32_MAX)
        if width == 0 or height == 0:
            raise ValueError("frame dimensions must be positive")
        if not isinstance(self.pixels, bytes):
            raise ValueError("pixels must be immutable bytes")
        expected = width * height * 3
        if expected > UINT64_MAX:
            raise ValueError("RGB payload is too large for the uint64 wire length")
        if len(self.pixels) != expected:
            raise ValueError(f"expected {expected} RGB bytes, received {len(self.pixels)}")

    @property
    def sha256(self) -> str:
        digest = sha256()
        digest.update(b"IRRGB\x01")
        digest.update(pack(">I", self.width))
        digest.update(pack(">I", self.height))
        digest.update(b"RGB8")
        digest.update(_length_prefixed(self.pixels))
        return digest.hexdigest()


class TextProvenance(IntEnum):
    """Allowed sources for model-visible text."""

    USER_INSTRUCTION = 1
    VISIBLE_UI = 2
    AUDIO_TRANSCRIPT = 3


@dataclass(frozen=True, slots=True)
class TextInput:
    provenance: TextProvenance
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if not isinstance(self.provenance, TextProvenance):
            raise ValueError("text provenance must be an allowed TextProvenance")
        if len(self.text.encode("utf-8")) > UINT64_MAX:
            raise ValueError("text is too large for the uint64 wire length")


@dataclass(frozen=True, slots=True)
class ModelObservation:
    """Canonical immutable model input for both live and recorded paths.

    This boundary contains resolved sensory bytes, never storage references or
    an absolute capture clock. Timing is relative to the current episode and
    to observation readiness, so replay and live inference share one contract.
    """

    frame_id: int
    elapsed_ns: int
    observation_age_ns: int
    rgb: RgbFrame
    previous_control: GenericControl
    audio_pcm_s16le: bytes | None = field(default=None, repr=False)
    text_inputs: tuple[TextInput, ...] = ()
    dropped_frames: int = 0

    def __post_init__(self) -> None:
        for name in ("frame_id", "elapsed_ns", "observation_age_ns", "dropped_frames"):
            _require_uint(getattr(self, name), name=name)
        if not isinstance(self.rgb, RgbFrame):
            raise ValueError("rgb must be an RgbFrame")
        if not isinstance(self.previous_control, GenericControl):
            raise ValueError("previous_control must be a GenericControl")
        if self.audio_pcm_s16le is not None and not isinstance(self.audio_pcm_s16le, bytes):
            raise ValueError("audio_pcm_s16le must be bytes or None")
        if self.audio_pcm_s16le is not None and len(self.audio_pcm_s16le) > UINT64_MAX:
            raise ValueError("audio_pcm_s16le is too large for the uint64 wire length")
        if not isinstance(self.text_inputs, tuple) or any(
            not isinstance(value, TextInput) for value in self.text_inputs
        ):
            raise ValueError("text_inputs must be a tuple of TextInput values")
        if len(self.text_inputs) > UINT32_MAX:
            raise ValueError("text_inputs is too large for the uint32 wire count")

    @property
    def content_hash(self) -> str:
        digest = sha256()
        digest.update(b"IRMOD\x01")
        digest.update(pack(">Q", self.frame_id))
        digest.update(pack(">Q", self.elapsed_ns))
        digest.update(pack(">Q", self.observation_age_ns))
        digest.update(bytes.fromhex(self.rgb.sha256))
        digest.update(_length_prefixed(self.previous_control.canonical_bytes()))
        if self.audio_pcm_s16le is None:
            digest.update(b"\x00")
        else:
            digest.update(b"\x01")
            digest.update(_length_prefixed(self.audio_pcm_s16le))
        digest.update(pack(">I", len(self.text_inputs)))
        for text_input in self.text_inputs:
            digest.update(pack(">B", int(text_input.provenance)))
            digest.update(_length_prefixed(text_input.text.encode("utf-8")))
        digest.update(pack(">Q", self.dropped_frames))
        return digest.hexdigest()


if frozenset(ModelObservation.__dataclass_fields__) != MODEL_OBSERVATION_FIELDS:
    raise RuntimeError("ModelObservation fields differ from the canonical allowlist")


@dataclass(frozen=True, slots=True)
class Observation:
    """Raw timestamped sensor envelope produced by a live environment."""

    frame_id: int
    capture_tick: int
    elapsed_ns: int
    rgb: RgbFrame
    previous_control: GenericControl
    audio_pcm_s16le: bytes | None = field(default=None, repr=False)
    text_inputs: tuple[TextInput, ...] = ()

    def __post_init__(self) -> None:
        for name in ("frame_id", "capture_tick", "elapsed_ns"):
            _require_uint(getattr(self, name), name=name)
        if not isinstance(self.rgb, RgbFrame):
            raise ValueError("rgb must be an RgbFrame")
        if not isinstance(self.previous_control, GenericControl):
            raise ValueError("previous_control must be a GenericControl")
        if self.audio_pcm_s16le is not None and not isinstance(self.audio_pcm_s16le, bytes):
            raise ValueError("audio_pcm_s16le must be bytes or None")
        if self.audio_pcm_s16le is not None and len(self.audio_pcm_s16le) > UINT64_MAX:
            raise ValueError("audio_pcm_s16le is too large for the uint64 wire length")
        if not isinstance(self.text_inputs, tuple) or any(
            not isinstance(value, TextInput) for value in self.text_inputs
        ):
            raise ValueError("text_inputs must be a tuple of TextInput values")
        if len(self.text_inputs) > UINT32_MAX:
            raise ValueError("text_inputs is too large for the uint32 wire count")

    def to_model_observation(
        self,
        *,
        observation_age_ns: int,
        dropped_frames: int = 0,
    ) -> ModelObservation:
        """Drop the absolute clock and produce the only legal model boundary."""

        return ModelObservation(
            frame_id=self.frame_id,
            elapsed_ns=self.elapsed_ns,
            observation_age_ns=_require_uint(
                observation_age_ns,
                name="observation_age_ns",
            ),
            rgb=self.rgb,
            previous_control=self.previous_control,
            audio_pcm_s16le=self.audio_pcm_s16le,
            text_inputs=self.text_inputs,
            dropped_frames=_require_uint(dropped_frames, name="dropped_frames"),
        )

    @property
    def content_hash(self) -> str:
        digest = sha256()
        digest.update(b"IROBS\x01")
        digest.update(pack(">Q", self.frame_id))
        digest.update(pack(">Q", self.capture_tick))
        digest.update(pack(">Q", self.elapsed_ns))
        digest.update(bytes.fromhex(self.rgb.sha256))
        digest.update(_length_prefixed(self.previous_control.canonical_bytes()))
        if self.audio_pcm_s16le is None:
            digest.update(b"\x00")
        else:
            digest.update(b"\x01")
            digest.update(_length_prefixed(self.audio_pcm_s16le))
        digest.update(pack(">I", len(self.text_inputs)))
        for text_input in self.text_inputs:
            digest.update(pack(">B", int(text_input.provenance)))
            digest.update(_length_prefixed(text_input.text.encode("utf-8")))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ActionEnvelope:
    """A versioned action proposal with explicit deadline and provenance."""

    action_sequence: int
    source_frame_id: int
    model_state_version: int
    created_qpc: int
    ready_qpc: int
    submit_deadline_qpc: int
    expires_qpc: int
    control: GenericControl

    def __post_init__(self) -> None:
        for name in (
            "action_sequence",
            "source_frame_id",
            "model_state_version",
            "created_qpc",
            "ready_qpc",
            "submit_deadline_qpc",
            "expires_qpc",
        ):
            _require_uint(getattr(self, name), name=name)
        if self.ready_qpc < self.created_qpc:
            raise ValueError("ready_qpc cannot precede created_qpc")
        if self.submit_deadline_qpc < self.ready_qpc:
            raise ValueError("submit deadline cannot precede ready_qpc")
        if self.expires_qpc < self.submit_deadline_qpc:
            raise ValueError("expiry cannot precede the submit deadline")
        if not isinstance(self.control, GenericControl):
            raise ValueError("control must be a GenericControl")


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """Result returned by a branchable environment step."""

    observation: Observation
    requested_control: GenericControl
    applied_control: GenericControl
    reward: float
    events: tuple[str, ...] = ()
    terminated: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "reward", _require_number(self.reward, name="reward"))
        if not isinstance(self.observation, Observation):
            raise ValueError("observation must be an Observation")
        if not isinstance(self.requested_control, GenericControl):
            raise ValueError("requested_control must be a GenericControl")
        if not isinstance(self.applied_control, GenericControl):
            raise ValueError("applied_control must be a GenericControl")
        if not isinstance(self.events, tuple) or any(
            not isinstance(event, str) or not event for event in self.events
        ):
            raise ValueError("events must be a tuple of non-empty strings")
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise ValueError("terminal flags must be booleans")
