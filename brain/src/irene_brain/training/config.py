"""Strict, immutable configuration for offline thought-field training.

The parser intentionally accepts a small fixed TOML surface.  Silent fallback
from a requested accelerator or precision would make experiment records
misleading, so incompatible settings fail before PyTorch is imported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping
import tomllib

from ..runtime.policy import ResourcePolicy
from .schedules import COSINE_AFTER_WARMUP, SCHEDULER_KINDS


_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_FACTORY = re.compile(
    r"[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*\Z"
)
_PARAMETER_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\Z")

# Recipe fields introduced after the first RCQ-v2 registration. They are
# optional in TOML and omitted from the canonical JSON while they hold these
# defaults, so every historical configuration hash stays byte-identical. Any
# non-default value becomes part of the configuration identity, which is how a
# future qualification binds its redesigned recipe.
OBJECTIVE_OPTIONAL_DEFAULTS: dict[str, object] = {
    "continuous_deadzone_hinge_weight": 0.0,
    "continuous_deadzone_hinge_margin": 0.04,
    "opposite_key_pair_weight": 0.0,
    "continuous_output_squash": "none",
    # Frozen RCQ / moving-shapes closed-loop decode. Omitted from canonical
    # JSON at this default so every historical configuration hash stays
    # byte-identical. exclusive_argmax_wasd_v1 is maze_chase-only.
    "play_decode_kind": "independent_logit_gt_zero_v1",
}

# Maze-chase episode-window sampling. Omitted from canonical JSON at 0 so
# every historical moving_shapes and spawn-only maze_chase configuration
# hash stays byte-identical. A positive value is a new identity.
DATASET_OPTIONAL_DEFAULTS: dict[str, object] = {
    "episode_horizon": 0,
    "window_sampling": "uniform",
}

# Play scoring during training. Omitted from canonical JSON at these defaults
# so every historical configuration hash stays byte-identical. A positive
# play_eval_every_steps is a new identity and maze_chase-only.
LOGGING_OPTIONAL_DEFAULTS: dict[str, object] = {
    "play_eval_every_steps": 0,
    "play_early_stop_kind": "none",
}


def _table(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a TOML table")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    *,
    name: str,
    required: frozenset[str],
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError(f"{name}: {'; '.join(details)}")


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _number(value: object, *, name: str, minimum: float = 0.0) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not (result >= minimum and result < float("inf")):
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return 0.0 if result == 0.0 else result


def _boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str
    seed: int
    model_factory: str
    max_optimizer_steps: int

    def __post_init__(self) -> None:
        if _SAFE_NAME.fullmatch(_string(self.name, name="run.name")) is None:
            raise ValueError("run.name contains non-portable characters")
        _integer(self.seed, name="run.seed", minimum=0)
        if self.seed > (2**63 - 1):
            raise ValueError("run.seed cannot exceed 2^63 - 1")
        if _FACTORY.fullmatch(
            _string(self.model_factory, name="run.model_factory")
        ) is None:
            raise ValueError("run.model_factory must use module.path:callable syntax")
        _integer(
            self.max_optimizer_steps,
            name="run.max_optimizer_steps",
            minimum=1,
        )


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    kind: str
    train_sequences: int
    validation_sequences: int
    test_sequences: int
    sequence_length: int
    burn_in_steps: int
    seed_offset: int
    hazard_count: int
    tick_period_ns: int
    discount: float
    episode_horizon: int = 0
    window_sampling: str = "uniform"

    def __post_init__(self) -> None:
        kind = _string(self.kind, name="dataset.kind")
        if kind not in {"moving_shapes", "maze_chase"}:
            raise ValueError(
                "dataset.kind currently supports only moving_shapes or maze_chase"
            )
        object.__setattr__(self, "kind", kind)
        _integer(self.train_sequences, name="dataset.train_sequences", minimum=1)
        _integer(
            self.validation_sequences,
            name="dataset.validation_sequences",
            minimum=1,
        )
        _integer(self.test_sequences, name="dataset.test_sequences", minimum=1)
        _integer(self.sequence_length, name="dataset.sequence_length", minimum=2)
        _integer(self.burn_in_steps, name="dataset.burn_in_steps", minimum=0)
        _integer(self.seed_offset, name="dataset.seed_offset", minimum=0)
        _integer(self.hazard_count, name="dataset.hazard_count", minimum=1)
        _integer(self.tick_period_ns, name="dataset.tick_period_ns", minimum=1)
        discount = _number(self.discount, name="dataset.discount", minimum=0.0)
        if discount > 1.0:
            raise ValueError("dataset.discount must be at most 1")
        if self.burn_in_steps >= self.sequence_length:
            raise ValueError("dataset.burn_in_steps must be smaller than sequence_length")
        namespace_size = 1 << 62
        if self.sequence_length > (2**32 - 1):
            raise ValueError("dataset.sequence_length cannot exceed 2^32 - 1")
        if self.hazard_count > (16 * 16 - 2):
            raise ValueError("dataset.hazard_count does not fit in MovingShapes")
        if self.tick_period_ns > (2**64 - 1):
            raise ValueError("dataset.tick_period_ns cannot exceed uint64")
        if self.seed_offset >= namespace_size:
            raise ValueError("dataset.seed_offset exceeds its split namespace")
        if self.seed_offset + max(
            self.train_sequences,
            self.validation_sequences,
            self.test_sequences,
        ) > namespace_size:
            raise ValueError("dataset sequence range exceeds its split namespace")
        horizon = _integer(self.episode_horizon, name="dataset.episode_horizon")
        if horizon != 0 and horizon <= self.sequence_length:
            raise ValueError(
                "dataset.episode_horizon must be 0 or greater than sequence_length"
            )
        if kind == "moving_shapes" and horizon != 0:
            raise ValueError(
                "dataset.episode_horizon is only valid for maze_chase"
            )
        sampling = _string(self.window_sampling, name="dataset.window_sampling")
        if sampling not in {"uniform", "tiled"}:
            raise ValueError("dataset.window_sampling must be uniform or tiled")
        if sampling == "tiled":
            if kind != "maze_chase":
                raise ValueError(
                    "dataset.window_sampling tiled is only valid for maze_chase"
                )
            if horizon == 0:
                raise ValueError(
                    "dataset.window_sampling tiled requires episode_horizon > 0"
                )
            if horizon % self.sequence_length != 0:
                raise ValueError(
                    "dataset.window_sampling tiled requires episode_horizon "
                    "divisible by sequence_length"
                )
        object.__setattr__(self, "episode_horizon", horizon)
        object.__setattr__(self, "window_sampling", sampling)


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_gradient_norm: float
    warmup_steps: int
    scheduler_kind: str = COSINE_AFTER_WARMUP

    def __post_init__(self) -> None:
        _integer(self.batch_size, name="optimization.batch_size", minimum=1)
        _integer(
            self.gradient_accumulation_steps,
            name="optimization.gradient_accumulation_steps",
            minimum=1,
        )
        if _number(
            self.learning_rate,
            name="optimization.learning_rate",
            minimum=0.0,
        ) == 0.0:
            raise ValueError("optimization.learning_rate must be positive")
        _number(self.weight_decay, name="optimization.weight_decay", minimum=0.0)
        if _number(
            self.max_gradient_norm,
            name="optimization.max_gradient_norm",
            minimum=0.0,
        ) == 0.0:
            raise ValueError("optimization.max_gradient_norm must be positive")
        _integer(self.warmup_steps, name="optimization.warmup_steps", minimum=0)
        scheduler_kind = _string(
            self.scheduler_kind,
            name="optimization.scheduler_kind",
        )
        if scheduler_kind not in SCHEDULER_KINDS:
            supported = ", ".join(sorted(SCHEDULER_KINDS))
            raise ValueError(
                f"optimization.scheduler_kind must be one of: {supported}"
            )


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    action_loss_kind: str = "support_aware_calibrated_v1"
    button_support_control_indices: tuple[int, ...] = (4, 7, 22, 26)
    button_support_weight: float = 0.8
    button_background_weight: float = 0.2
    button_background_tail_mix: float = 0.9
    button_background_tail_temperature: float = 0.1
    continuous_action_weight: float = 0.25
    action_weight: float = 1.0
    value_weight: float = 0.1
    world_weight: float = 0.1
    diversity_weight: float = 0.05
    continuous_deadzone_hinge_weight: float = 0.0
    continuous_deadzone_hinge_margin: float = 0.04
    opposite_key_pair_weight: float = 0.0
    continuous_output_squash: str = "none"
    play_decode_kind: str = "independent_logit_gt_zero_v1"

    def __post_init__(self) -> None:
        action_loss_kind = _string(
            self.action_loss_kind,
            name="objective.action_loss_kind",
        )
        if action_loss_kind not in {
            "sparse_hard_negative_v1",
            "support_aware_calibrated_v1",
            "exclusive_wasd_softmax_v1",
            "exclusive_wasd_softmax_turn_weighted_v1",
        }:
            raise ValueError(
                "objective.action_loss_kind must be sparse_hard_negative_v1, "
                "support_aware_calibrated_v1, exclusive_wasd_softmax_v1, or "
                "exclusive_wasd_softmax_turn_weighted_v1"
            )
        object.__setattr__(self, "action_loss_kind", action_loss_kind)

        raw_indices = self.button_support_control_indices
        if not isinstance(raw_indices, (list, tuple)) or not raw_indices:
            raise ValueError(
                "objective.button_support_control_indices must be a non-empty array"
            )
        indices = tuple(
            _integer(
                value,
                name=f"objective.button_support_control_indices[{index}]",
                minimum=0,
            )
            for index, value in enumerate(raw_indices)
        )
        if indices != tuple(sorted(set(indices))):
            raise ValueError(
                "objective.button_support_control_indices must be sorted and unique"
            )
        if any(not (value < 264 or 267 <= value < 299) for value in indices):
            raise ValueError(
                "objective.button_support_control_indices must select button channels"
            )
        object.__setattr__(self, "button_support_control_indices", indices)

        support_weight = _number(
            self.button_support_weight,
            name="objective.button_support_weight",
            minimum=0.0,
        )
        background_weight = _number(
            self.button_background_weight,
            name="objective.button_background_weight",
            minimum=0.0,
        )
        if abs((support_weight + background_weight) - 1.0) > 1e-12:
            raise ValueError(
                "objective button support and background weights must sum to 1"
            )
        object.__setattr__(self, "button_support_weight", support_weight)
        object.__setattr__(self, "button_background_weight", background_weight)

        tail_mix = _number(
            self.button_background_tail_mix,
            name="objective.button_background_tail_mix",
            minimum=0.0,
        )
        if tail_mix > 1.0:
            raise ValueError("objective.button_background_tail_mix must be at most 1")
        object.__setattr__(self, "button_background_tail_mix", tail_mix)
        tail_temperature = _number(
            self.button_background_tail_temperature,
            name="objective.button_background_tail_temperature",
            minimum=0.0,
        )
        if tail_temperature == 0.0:
            raise ValueError(
                "objective.button_background_tail_temperature must be positive"
            )
        object.__setattr__(
            self,
            "button_background_tail_temperature",
            tail_temperature,
        )
        object.__setattr__(
            self,
            "continuous_action_weight",
            _number(
                self.continuous_action_weight,
                name="objective.continuous_action_weight",
                minimum=0.0,
            ),
        )
        for name in (
            "action_weight",
            "value_weight",
            "world_weight",
            "diversity_weight",
        ):
            normalized = _number(
                getattr(self, name),
                name=f"objective.{name}",
                minimum=0.0,
            )
            object.__setattr__(self, name, normalized)
        object.__setattr__(
            self,
            "continuous_deadzone_hinge_weight",
            _number(
                self.continuous_deadzone_hinge_weight,
                name="objective.continuous_deadzone_hinge_weight",
                minimum=0.0,
            ),
        )
        hinge_margin = _number(
            self.continuous_deadzone_hinge_margin,
            name="objective.continuous_deadzone_hinge_margin",
            minimum=0.0,
        )
        if hinge_margin >= 0.05:
            raise ValueError(
                "objective.continuous_deadzone_hinge_margin must stay inside the "
                "0.05 deadzone"
            )
        object.__setattr__(self, "continuous_deadzone_hinge_margin", hinge_margin)
        object.__setattr__(
            self,
            "opposite_key_pair_weight",
            _number(
                self.opposite_key_pair_weight,
                name="objective.opposite_key_pair_weight",
                minimum=0.0,
            ),
        )
        squash = _string(
            self.continuous_output_squash,
            name="objective.continuous_output_squash",
        )
        if squash not in {"none", "deadzone_tanh"}:
            raise ValueError(
                "objective.continuous_output_squash must be none or deadzone_tanh"
            )
        object.__setattr__(self, "continuous_output_squash", squash)
        play_decode = _string(
            self.play_decode_kind,
            name="objective.play_decode_kind",
        )
        if play_decode not in {
            "independent_logit_gt_zero_v1",
            "exclusive_argmax_wasd_v1",
        }:
            raise ValueError(
                "objective.play_decode_kind must be independent_logit_gt_zero_v1 "
                "or exclusive_argmax_wasd_v1"
            )
        object.__setattr__(self, "play_decode_kind", play_decode)


@dataclass(frozen=True, slots=True)
class PrecisionConfig:
    device: str
    mode: str
    allow_tf32: bool

    def __post_init__(self) -> None:
        device = _string(self.device, name="precision.device").casefold()
        if device not in {"cpu", "cuda"}:
            raise ValueError("precision.device must be cpu or cuda")
        mode = _string(self.mode, name="precision.mode").casefold()
        if mode not in {"float32", "bfloat16", "float16"}:
            raise ValueError("precision.mode must be float32, bfloat16, or float16")
        if mode == "float16" and device != "cuda":
            raise ValueError("float16 training requires CUDA")
        _boolean(self.allow_tf32, name="precision.allow_tf32")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True, slots=True)
class DeterminismConfig:
    enabled: bool
    num_workers: int
    compile_model: bool

    def __post_init__(self) -> None:
        enabled = _boolean(self.enabled, name="determinism.enabled")
        workers = _integer(
            self.num_workers,
            name="determinism.num_workers",
            minimum=0,
        )
        compiled = _boolean(self.compile_model, name="determinism.compile_model")
        if enabled and workers != 0:
            raise ValueError("exact-resume mode requires determinism.num_workers=0")
        if enabled and compiled:
            raise ValueError("exact-resume mode requires determinism.compile_model=false")


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    log_every_steps: int
    evaluate_every_steps: int
    validation_batches: int
    checkpoint_every_steps: int
    keep_last_checkpoints: int
    play_eval_every_steps: int = 0
    play_early_stop_kind: str = "none"

    def __post_init__(self) -> None:
        for name in (
            "log_every_steps",
            "evaluate_every_steps",
            "validation_batches",
            "checkpoint_every_steps",
            "keep_last_checkpoints",
        ):
            _integer(getattr(self, name), name=f"logging.{name}", minimum=1)
        play_every = _integer(
            self.play_eval_every_steps,
            name="logging.play_eval_every_steps",
            minimum=0,
        )
        object.__setattr__(self, "play_eval_every_steps", play_every)
        kind = _string(
            self.play_early_stop_kind,
            name="logging.play_early_stop_kind",
        )
        if kind not in {"none", "play_peak_v1"}:
            raise ValueError(
                "logging.play_early_stop_kind must be none or play_peak_v1"
            )
        object.__setattr__(self, "play_early_stop_kind", kind)
        if kind == "play_peak_v1" and play_every < 1:
            raise ValueError(
                "logging.play_early_stop_kind play_peak_v1 requires "
                "logging.play_eval_every_steps >= 1"
            )


@dataclass(frozen=True, slots=True)
class ResourceConfig:
    allow_gpu: bool
    allow_capture: bool
    allow_hid_output: bool
    allow_background_threads: bool
    allow_network: bool
    allow_subprocess: bool
    write_artifacts: bool
    cpu_threads: int

    def __post_init__(self) -> None:
        for name in (
            "allow_gpu",
            "allow_capture",
            "allow_hid_output",
            "allow_background_threads",
            "allow_network",
            "allow_subprocess",
            "write_artifacts",
        ):
            _boolean(getattr(self, name), name=f"resources.{name}")
        _integer(self.cpu_threads, name="resources.cpu_threads", minimum=1)

    @property
    def policy(self) -> ResourcePolicy:
        return ResourcePolicy(
            allow_gpu=self.allow_gpu,
            allow_capture=self.allow_capture,
            allow_hid_output=self.allow_hid_output,
            allow_background_threads=self.allow_background_threads,
            allow_network=self.allow_network,
            allow_subprocess=self.allow_subprocess,
            allow_artifact_write=self.write_artifacts,
            cpu_threads=self.cpu_threads,
        )


@dataclass(frozen=True, slots=True)
class TrainingStageConfig:
    """One fully registered optimizer phase in a schema-3 run."""

    index: int
    name: str
    start_optimizer_step: int
    end_optimizer_step: int
    trainable_parameters: tuple[str, ...]
    reset_optimizer: bool
    learning_rate: float
    weight_decay: float
    max_gradient_norm: float
    warmup_steps: int
    scheduler_kind: str
    action_weight: float
    value_weight: float
    world_weight: float
    diversity_weight: float
    transition_gate: str
    completion_gate: str
    invariance_audit: str

    def __post_init__(self) -> None:
        _integer(self.index, name="stages.index", minimum=0)
        if _SAFE_NAME.fullmatch(_string(self.name, name="stages.name")) is None:
            raise ValueError("stages.name contains non-portable characters")
        start = _integer(
            self.start_optimizer_step,
            name="stages.start_optimizer_step",
            minimum=0,
        )
        end = _integer(
            self.end_optimizer_step,
            name="stages.end_optimizer_step",
            minimum=1,
        )
        if end <= start:
            raise ValueError("stages.end_optimizer_step must exceed its start")

        raw_parameters = self.trainable_parameters
        if not isinstance(raw_parameters, (list, tuple)) or not raw_parameters:
            raise ValueError("stages.trainable_parameters must be a non-empty array")
        parameters = tuple(
            _string(value, name=f"stages.trainable_parameters[{index}]")
            for index, value in enumerate(raw_parameters)
        )
        if parameters == ("*",):
            pass
        elif "*" in parameters:
            raise ValueError(
                "stages.trainable_parameters cannot mix '*' with explicit names"
            )
        elif parameters != tuple(sorted(set(parameters))):
            raise ValueError(
                "stages.trainable_parameters must be sorted and unique"
            )
        elif any(_PARAMETER_NAME.fullmatch(name) is None for name in parameters):
            raise ValueError("stages.trainable_parameters contains an invalid name")
        object.__setattr__(self, "trainable_parameters", parameters)

        _boolean(self.reset_optimizer, name="stages.reset_optimizer")
        learning_rate = _number(
            self.learning_rate,
            name="stages.learning_rate",
            minimum=0.0,
        )
        if learning_rate == 0.0:
            raise ValueError("stages.learning_rate must be positive")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(
            self,
            "weight_decay",
            _number(self.weight_decay, name="stages.weight_decay", minimum=0.0),
        )
        maximum_norm = _number(
            self.max_gradient_norm,
            name="stages.max_gradient_norm",
            minimum=0.0,
        )
        if maximum_norm == 0.0:
            raise ValueError("stages.max_gradient_norm must be positive")
        object.__setattr__(self, "max_gradient_norm", maximum_norm)
        warmup = _integer(self.warmup_steps, name="stages.warmup_steps", minimum=0)
        if warmup > (end - start):
            raise ValueError("stages.warmup_steps cannot exceed the stage length")
        scheduler_kind = _string(
            self.scheduler_kind,
            name="stages.scheduler_kind",
        )
        if scheduler_kind not in SCHEDULER_KINDS:
            supported = ", ".join(sorted(SCHEDULER_KINDS))
            raise ValueError(f"stages.scheduler_kind must be one of: {supported}")
        for name in (
            "action_weight",
            "value_weight",
            "world_weight",
            "diversity_weight",
        ):
            object.__setattr__(
                self,
                name,
                _number(getattr(self, name), name=f"stages.{name}", minimum=0.0),
            )
        if not any(
            getattr(self, name) > 0.0
            for name in (
                "action_weight",
                "value_weight",
                "world_weight",
                "diversity_weight",
            )
        ):
            raise ValueError("each stage must enable at least one objective weight")
        transition_gate = _string(
            self.transition_gate,
            name="stages.transition_gate",
        )
        if transition_gate not in {"none", "rcq_v2_development_v1"}:
            raise ValueError(
                "stages.transition_gate has an unsupported registered gate ID"
            )
        completion_gate = _string(
            self.completion_gate,
            name="stages.completion_gate",
        )
        if completion_gate not in {"none", "rcq_v2_value_development_v1"}:
            raise ValueError(
                "stages.completion_gate has an unsupported registered gate ID"
            )
        invariance_audit = _string(
            self.invariance_audit,
            name="stages.invariance_audit",
        )
        if invariance_audit not in {"none", "nonvalue_action_state_v1"}:
            raise ValueError(
                "stages.invariance_audit must be none or nonvalue_action_state_v1"
            )

    @property
    def optimizer_steps(self) -> int:
        return self.end_optimizer_step - self.start_optimizer_step


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    schema_version: int
    run: RunConfig
    dataset: DatasetConfig
    optimization: OptimizationConfig
    precision: PrecisionConfig
    determinism: DeterminismConfig
    logging: LoggingConfig
    resources: ResourceConfig
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    stages: tuple[TrainingStageConfig, ...] = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in {1, 2, 3}:
            raise ValueError("schema_version must be integer 1, 2, or 3")
        for name, expected in (
            ("run", RunConfig),
            ("dataset", DatasetConfig),
            ("optimization", OptimizationConfig),
            ("precision", PrecisionConfig),
            ("determinism", DeterminismConfig),
            ("logging", LoggingConfig),
            ("resources", ResourceConfig),
            ("objective", ObjectiveConfig),
        ):
            if not isinstance(getattr(self, name), expected):
                raise ValueError(f"{name} must be a {expected.__name__}")
        if self.precision.device != "cpu" and not self.resources.allow_gpu:
            raise ValueError("CUDA training requires resources.allow_gpu=true")
        if self.resources.allow_capture or self.resources.allow_hid_output:
            raise ValueError("offline training forbids capture and HID output")
        if self.resources.allow_network or self.resources.allow_subprocess:
            raise ValueError("the trainer itself forbids network and subprocess access")
        if not self.resources.write_artifacts:
            raise ValueError("training requires explicit artifact-write permission")
        if self.optimization.warmup_steps > self.run.max_optimizer_steps:
            raise ValueError("warmup_steps cannot exceed max_optimizer_steps")
        if (
            self.objective.play_decode_kind != "independent_logit_gt_zero_v1"
            and self.dataset.kind != "maze_chase"
        ):
            raise ValueError(
                "objective.play_decode_kind other than independent_logit_gt_zero_v1 "
                "is only valid for maze_chase"
            )
        if self.logging.play_eval_every_steps > 0:
            if self.dataset.kind != "maze_chase":
                raise ValueError(
                    "logging.play_eval_every_steps is only valid for maze_chase"
                )
            if self.schema_version == 3:
                raise ValueError(
                    "schema-3 configs cannot declare play evaluation during training"
                )
        if self.logging.play_early_stop_kind != "none" and self.schema_version == 3:
            raise ValueError(
                "schema-3 configs cannot declare play-peak early-stop"
            )
        if self.objective.action_loss_kind in {
            "exclusive_wasd_softmax_v1",
            "exclusive_wasd_softmax_turn_weighted_v1",
        }:
            if self.dataset.kind != "maze_chase":
                raise ValueError(
                    "objective.action_loss_kind exclusive WASD softmax "
                    "is only valid for maze_chase"
                )
            if self.objective.play_decode_kind != "exclusive_argmax_wasd_v1":
                raise ValueError(
                    "objective.action_loss_kind exclusive WASD softmax "
                    "requires exclusive_argmax_wasd_v1 play decode"
                )
        if (
            self.schema_version == 1
            and self.optimization.scheduler_kind != COSINE_AFTER_WARMUP
        ):
            raise ValueError("schema_version 1 supports only the legacy cosine schedule")
        raw_stages = self.stages
        if not isinstance(raw_stages, (list, tuple)):
            raise ValueError("stages must be an array of training stages")
        stages = tuple(raw_stages)
        if any(not isinstance(stage, TrainingStageConfig) for stage in stages):
            raise ValueError("stages must contain only TrainingStageConfig values")
        object.__setattr__(self, "stages", stages)
        if self.schema_version < 3:
            if stages:
                raise ValueError("schema versions 1 and 2 cannot declare stages")
            return
        # Schema 3 makes semantically equivalent numeric spellings share one
        # canonical identity. Historical schema-1/2 objects deliberately retain
        # their byte-for-byte canonicalization behavior.
        object.__setattr__(
            self,
            "dataset",
            replace(
                self.dataset,
                discount=(
                    0.0
                    if float(self.dataset.discount) == 0.0
                    else float(self.dataset.discount)
                ),
            ),
        )
        object.__setattr__(
            self,
            "optimization",
            replace(
                self.optimization,
                learning_rate=float(self.optimization.learning_rate),
                weight_decay=(
                    0.0
                    if float(self.optimization.weight_decay) == 0.0
                    else float(self.optimization.weight_decay)
                ),
                max_gradient_norm=float(self.optimization.max_gradient_norm),
            ),
        )
        if not stages:
            raise ValueError("schema_version 3 requires at least one stage")
        if not stages[0].reset_optimizer:
            raise ValueError("the first schema-3 stage must initialize its optimizer")
        expected_start = 0
        for expected_index, stage in enumerate(stages):
            if stage.index != expected_index:
                raise ValueError("schema-3 stage indices must be contiguous from zero")
            if stage.start_optimizer_step != expected_start:
                raise ValueError("schema-3 optimizer stage ranges must be contiguous")
            expected_start = stage.end_optimizer_step
            if not stage.reset_optimizer:
                raise ValueError(
                    "schema-3 currently requires an explicit optimizer reset per stage"
                )
            if expected_index == len(stages) - 1 and stage.transition_gate != "none":
                raise ValueError("the final stage cannot declare a transition gate")
            if expected_index < len(stages) - 1 and stage.completion_gate != "none":
                raise ValueError("only the final stage can declare a completion gate")
            if stage.end_optimizer_step % self.logging.checkpoint_every_steps != 0:
                raise ValueError(
                    "each schema-3 stage boundary must align with checkpoint cadence"
                )
            if (
                stage.transition_gate != "none" or stage.completion_gate != "none"
            ) and stage.end_optimizer_step % self.logging.evaluate_every_steps != 0:
                raise ValueError(
                    "each schema-3 gate must align with evaluation cadence"
                )
        if expected_start != self.run.max_optimizer_steps:
            raise ValueError("the final stage must end at run.max_optimizer_steps")
        first = stages[0]
        first_optimization = (
            first.learning_rate,
            first.weight_decay,
            first.max_gradient_norm,
            first.warmup_steps,
            first.scheduler_kind,
        )
        configured_optimization = (
            self.optimization.learning_rate,
            self.optimization.weight_decay,
            self.optimization.max_gradient_norm,
            self.optimization.warmup_steps,
            self.optimization.scheduler_kind,
        )
        if first_optimization != configured_optimization:
            raise ValueError(
                "schema-3 top-level optimization must exactly match stage zero"
            )
        first_objective = (
            first.action_weight,
            first.value_weight,
            first.world_weight,
            first.diversity_weight,
        )
        configured_objective = (
            self.objective.action_weight,
            self.objective.value_weight,
            self.objective.world_weight,
            self.objective.diversity_weight,
        )
        if first_objective != configured_objective:
            raise ValueError(
                "schema-3 top-level objective weights must exactly match stage zero"
            )
        if any(
            stage.transition_gate == "rcq_v2_development_v1"
            or stage.completion_gate == "rcq_v2_value_development_v1"
            for stage in stages
        ):
            self._validate_rcq_v2_gate_geometry()

    def _validate_rcq_v2_gate_geometry(self) -> None:
        """Bind the registered RCQ gate ID to its frozen data geometry."""

        expected_dataset = {
            "kind": "moving_shapes",
            "train_sequences": 8192,
            "validation_sequences": 256,
            "test_sequences": 512,
            "sequence_length": 8,
            "burn_in_steps": 2,
            "seed_offset": 1_048_576,
            "hazard_count": 3,
            "tick_period_ns": 16_666_667,
            "discount": 0.99,
        }
        observed_dataset = {
            name: getattr(self.dataset, name) for name in expected_dataset
        }
        if observed_dataset != expected_dataset:
            raise ValueError(
                "rcq_v2_development_v1 requires the frozen RCQ-v2 dataset geometry"
            )
        if (
            self.run.seed != 1702
            or self.run.model_factory
            != "irene_brain.training.factory:build_thesis_model"
            or self.run.max_optimizer_steps != 2048
        ):
            raise ValueError(
                "rcq_v2_development_v1 requires the frozen RCQ-v2 run identity"
            )
        if (
            self.optimization.batch_size != 1
            or self.optimization.gradient_accumulation_steps != 8
            or self.logging.evaluate_every_steps != 256
            or self.logging.validation_batches != 256
            or self.logging.checkpoint_every_steps != 256
        ):
            raise ValueError(
                "rcq_v2_development_v1 requires its registered batch and cadence geometry"
            )
        decisions = (
            self.logging.validation_batches
            * self.optimization.batch_size
            * (self.dataset.sequence_length - self.dataset.burn_in_steps)
        )
        if decisions != 1_536:
            raise ValueError(
                "rcq_v2_development_v1 requires exactly 1,536 dev decisions"
            )
        if (
            len(self.stages) != 2
            or self.stages[0].start_optimizer_step != 0
            or self.stages[0].end_optimizer_step != 1536
            or self.stages[0].transition_gate != "rcq_v2_development_v1"
            or self.stages[0].completion_gate != "none"
            or self.stages[1].start_optimizer_step != 1536
            or self.stages[1].end_optimizer_step != 2048
            or self.stages[1].transition_gate != "none"
            or self.stages[1].completion_gate
            != "rcq_v2_value_development_v1"
        ):
            raise ValueError(
                "rcq_v2_development_v1 requires the registered 1536+512 stage boundaries"
            )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.schema_version == 1:
            # Schema 1 predates configurable schedules. Keep its canonical JSON
            # and checkpoint identity stable while resolving it to cosine at runtime.
            del result["optimization"]["scheduler_kind"]
        if self.schema_version < 3:
            # Staging was added in schema 3. Preserve every historical schema-1/2
            # canonical hash byte-for-byte.
            del result["stages"]
        # Keep the in-memory configuration immutable while exposing a canonical
        # JSON-compatible representation for hashing and run manifests.
        result["objective"]["button_support_control_indices"] = list(
            self.objective.button_support_control_indices
        )
        for key, default in OBJECTIVE_OPTIONAL_DEFAULTS.items():
            if result["objective"].get(key) == default:
                del result["objective"][key]
        for key, default in DATASET_OPTIONAL_DEFAULTS.items():
            if result["dataset"].get(key) == default:
                del result["dataset"][key]
        for key, default in LOGGING_OPTIONAL_DEFAULTS.items():
            if result["logging"].get(key) == default:
                del result["logging"][key]
        return result

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
    def config_sha256(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def resource_policy(self) -> ResourcePolicy:
        return self.resources.policy

    @classmethod
    def from_toml(cls, path: str | Path) -> TrainingConfig:
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        root = _table(raw, name="root")
        _exact_fields(
            root,
            name="root",
            required=frozenset(
                {
                    "schema_version",
                    "run",
                    "dataset",
                    "optimization",
                    "precision",
                    "determinism",
                    "logging",
                    "resources",
                    "objective",
                    *({"stages"} if raw.get("schema_version") == 3 else set()),
                }
            ),
        )
        schema_version = root["schema_version"]
        if type(schema_version) is not int or schema_version not in {1, 2, 3}:
            raise ValueError("schema_version must be integer 1, 2, or 3")
        tables: dict[str, Mapping[str, Any]] = {
            name: _table(root[name], name=name)
            for name in (
                "run",
                "dataset",
                "optimization",
                "precision",
                "determinism",
                "logging",
                "resources",
                "objective",
            )
        }
        expected = {
            "run": frozenset(
                {"name", "seed", "model_factory", "max_optimizer_steps"}
            ),
            "dataset": frozenset(
                {
                    "kind",
                    "train_sequences",
                    "validation_sequences",
                    "test_sequences",
                    "sequence_length",
                    "burn_in_steps",
                    "seed_offset",
                    "hazard_count",
                    "tick_period_ns",
                    "discount",
                }
            ),
            "optimization": frozenset(
                {
                    "batch_size",
                    "gradient_accumulation_steps",
                    "learning_rate",
                    "weight_decay",
                    "max_gradient_norm",
                    "warmup_steps",
                    *({"scheduler_kind"} if schema_version >= 2 else set()),
                }
            ),
            "objective": frozenset(
                {
                    "action_loss_kind",
                    "button_support_control_indices",
                    "button_support_weight",
                    "button_background_weight",
                    "button_background_tail_mix",
                    "button_background_tail_temperature",
                    "continuous_action_weight",
                    "action_weight",
                    "value_weight",
                    "world_weight",
                    "diversity_weight",
                }
            ),
            "precision": frozenset({"device", "mode", "allow_tf32"}),
            "determinism": frozenset({"enabled", "num_workers", "compile_model"}),
            "logging": frozenset(
                {
                    "log_every_steps",
                    "evaluate_every_steps",
                    "validation_batches",
                    "checkpoint_every_steps",
                    "keep_last_checkpoints",
                }
            ),
            "resources": frozenset(
                {
                    "allow_gpu",
                    "allow_capture",
                    "allow_hid_output",
                    "allow_background_threads",
                    "allow_network",
                    "allow_subprocess",
                    "write_artifacts",
                    "cpu_threads",
                }
            ),
        }
        for name, fields in expected.items():
            if name == "objective":
                optional = frozenset(OBJECTIVE_OPTIONAL_DEFAULTS)
                unknown = sorted(set(tables[name]) - set(fields) - set(optional))
                missing = sorted(set(fields) - set(tables[name]))
                if unknown or missing:
                    details: list[str] = []
                    if missing:
                        details.append(f"missing fields: {', '.join(missing)}")
                    if unknown:
                        details.append(f"unknown fields: {', '.join(unknown)}")
                    raise ValueError(f"objective: {'; '.join(details)}")
                tables[name] = {
                    **OBJECTIVE_OPTIONAL_DEFAULTS,
                    **tables[name],
                }
                continue
            if name == "dataset":
                optional = frozenset(DATASET_OPTIONAL_DEFAULTS)
                unknown = sorted(set(tables[name]) - set(fields) - set(optional))
                missing = sorted(set(fields) - set(tables[name]))
                if unknown or missing:
                    details: list[str] = []
                    if missing:
                        details.append(f"missing fields: {', '.join(missing)}")
                    if unknown:
                        details.append(f"unknown fields: {', '.join(unknown)}")
                    raise ValueError(f"dataset: {'; '.join(details)}")
                tables[name] = {
                    **DATASET_OPTIONAL_DEFAULTS,
                    **tables[name],
                }
                continue
            if name == "logging":
                optional = frozenset(LOGGING_OPTIONAL_DEFAULTS)
                unknown = sorted(set(tables[name]) - set(fields) - set(optional))
                missing = sorted(set(fields) - set(tables[name]))
                if unknown or missing:
                    details: list[str] = []
                    if missing:
                        details.append(f"missing fields: {', '.join(missing)}")
                    if unknown:
                        details.append(f"unknown fields: {', '.join(unknown)}")
                    raise ValueError(f"logging: {'; '.join(details)}")
                tables[name] = {
                    **LOGGING_OPTIONAL_DEFAULTS,
                    **tables[name],
                }
                continue
            _exact_fields(tables[name], name=name, required=fields)
        stages: tuple[TrainingStageConfig, ...] = ()
        if schema_version == 3:
            raw_stages = root["stages"]
            if not isinstance(raw_stages, list) or not raw_stages:
                raise ValueError("stages must be a non-empty TOML array of tables")
            stage_fields = frozenset(
                {
                    "index",
                    "name",
                    "start_optimizer_step",
                    "end_optimizer_step",
                    "trainable_parameters",
                    "reset_optimizer",
                    "learning_rate",
                    "weight_decay",
                    "max_gradient_norm",
                    "warmup_steps",
                    "scheduler_kind",
                    "action_weight",
                    "value_weight",
                    "world_weight",
                    "diversity_weight",
                    "transition_gate",
                    "completion_gate",
                    "invariance_audit",
                }
            )
            parsed_stages: list[TrainingStageConfig] = []
            for index, raw_stage in enumerate(raw_stages):
                stage_table = _table(raw_stage, name=f"stages[{index}]")
                _exact_fields(
                    stage_table,
                    name=f"stages[{index}]",
                    required=stage_fields,
                )
                parsed_stages.append(TrainingStageConfig(**stage_table))
            stages = tuple(parsed_stages)
        return cls(
            schema_version=root["schema_version"],
            run=RunConfig(**tables["run"]),
            dataset=DatasetConfig(**tables["dataset"]),
            optimization=OptimizationConfig(**tables["optimization"]),
            precision=PrecisionConfig(**tables["precision"]),
            determinism=DeterminismConfig(**tables["determinism"]),
            logging=LoggingConfig(**tables["logging"]),
            resources=ResourceConfig(**tables["resources"]),
            objective=ObjectiveConfig(**tables["objective"]),
            stages=stages,
        )


def load_training_config(path: str | Path) -> TrainingConfig:
    return TrainingConfig.from_toml(path)


__all__ = [
    "DatasetConfig",
    "DeterminismConfig",
    "LoggingConfig",
    "ObjectiveConfig",
    "OptimizationConfig",
    "PrecisionConfig",
    "ResourceConfig",
    "RunConfig",
    "TrainingConfig",
    "TrainingStageConfig",
    "DATASET_OPTIONAL_DEFAULTS",
    "LOGGING_OPTIONAL_DEFAULTS",
    "OBJECTIVE_OPTIONAL_DEFAULTS",
    "load_training_config",
]
