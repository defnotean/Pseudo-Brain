"""Fail-closed V2.1 live-play qualification against a five-action random floor.

This module deliberately separates the qualification protocol from model
loading and from namespace construction.  It never opens a dataset and it does
not know how to train a model.  Production accepts only the exact checkpoint-
backed maze policy and exact in-repository world.  An underscore-prefixed test
seam can inject a runner, but its artifacts are permanently non-production.

The model boundary is intentionally narrower than ``Observation``.  The
controller receives only an immutable :class:`RgbFrame` and the action that the
adapter itself selected on the preceding decision.  Environment seeds,
timestamps, rewards, events, simulator state, and ``previous_control`` never
cross that boundary.  The controller is reset to one registered constant seed
for every episode, so the episode seed cannot become a side channel.

Evidence is paired by episode seed.  Reward advantage is model minus random;
catch advantage is random minus model.  Both must have a strictly positive
one-sided 95% lower bound under a deterministic paired cluster bootstrap.  All
model and baseline controls must also be legal, submitted, and scheduled on an
exact 60 Hz manual-clock cadence.  This is explicitly simulated timing
evidence; capture, transport, and physical HID latency remain separate gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from struct import pack
from typing import BinaryIO, Callable, Hashable, Mapping, Protocol, Sequence

from ..environments.maze_chase import MazeChaseEnv
from ..types import GenericControl, HidKey, Observation, RgbFrame, StepOutcome
from .closed_loop_play import (
    ClosedLoopEpisodeReport,
    ClosedLoopPlayConfig,
    DecisionTiming,
    _run_episode_core,
    control_audit_stats,
)


SCHEMA_VERSION = 1
TICK_PERIOD_NS_60HZ = 16_666_667
ACTION_SEMANTICS = ("idle", "W", "A", "S", "D")
ACTION_IDS = tuple(range(len(ACTION_SEMANTICS)))
_ACTION_KEYS = (
    None,
    int(HidKey.W),
    int(HidKey.A),
    int(HidKey.S),
    int(HidKey.D),
)
_ACTION_MASKS = (0, 1, 2, 4, 8)
_UINT64_MASK = (1 << 64) - 1
_TRACE_DOMAIN = b"IRENE-V21-LIVE-TRACE\x01"
_STEP_TRACE_DOMAIN = b"IRENE-V21-LIVE-STEP-TRACE\x01"
EXACT_MAZE_ENVIRONMENT_ID = "maze_chase.exact_live.v1"
DEVELOPMENT_CHECKPOINT_ENVELOPE = "development_v1"
POST_DGX_CHECKPOINT_ENVELOPE = "post_dgx_v1"


def exact_maze_environment_config(max_ticks: int) -> dict[str, object]:
    """Return the only world configuration accepted by production play."""

    _require_plain_int(max_ticks, name="max_ticks", minimum=1)
    return {
        "environment_id": EXACT_MAZE_ENVIRONMENT_ID,
        "extra_loops": 16,
        "ghost_count": 5,
        "ghost_elroy": False,
        "ghost_period": 1,
        "ghost_rule": "direct",
        "input_delay_ticks": 0,
        "max_ticks": max_ticks,
        "player_period": 1,
        "sticky_direction": False,
        "tick_period_ns": TICK_PERIOD_NS_60HZ,
    }


def exact_maze_environment_config_sha256(max_ticks: int) -> str:
    return sha256(
        _canonical_json(exact_maze_environment_config(max_ticks)).encode("utf-8")
    ).hexdigest()


class LivePlayQualificationError(RuntimeError):
    """The live-play protocol or one of its evidence inputs is invalid."""


class LivePlayQualificationFailed(LivePlayQualificationError):
    """All evidence was written, but one or more registered gates failed."""

    def __init__(self, report: "LivePlayQualificationReport") -> None:
        super().__init__("live-play qualification failed")
        self.report = report


class RgbActionController(Protocol):
    """Narrow model-facing contract used by :class:`StrictRgbOnlyPolicy`."""

    def reset(self, seed: int) -> None:
        """Reset recurrent state and stochastic state to the registered seed."""

    def act_rgb(self, rgb: RgbFrame, previous_action_id: int) -> int:
        """Return one semantic action ID from ``ACTION_IDS``."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _require_plain_int(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LivePlayQualificationError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _require_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LivePlayQualificationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise LivePlayQualificationError(f"{name} must be a finite number")
    return result


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise LivePlayQualificationError(f"{name} must be nonempty text")
    return value


def _require_digest(value: object, *, name: str) -> str:
    result = _require_text(value, name=name)
    if len(result) != 64 or result != result.lower():
        raise LivePlayQualificationError(f"{name} must be a lowercase SHA-256")
    try:
        bytes.fromhex(result)
    except ValueError as error:
        raise LivePlayQualificationError(
            f"{name} must be a lowercase SHA-256"
        ) from error
    return result


def _action_control(action_id: int) -> GenericControl:
    _require_plain_int(action_id, name="action_id")
    if action_id not in ACTION_IDS:
        raise LivePlayQualificationError(
            "action_id must be exactly one of idle/W/A/S/D (0/1/2/3/4)"
        )
    key = _ACTION_KEYS[action_id]
    return GenericControl() if key is None else GenericControl(keys_down=(key,))


class _TraceRecorder:
    __slots__ = ("_action_histogram", "_step_count", "_trace")

    def _reset_trace(self) -> None:
        self._trace = sha256(_TRACE_DOMAIN)
        self._step_count = 0
        self._action_histogram = [0] * len(ACTION_IDS)

    def _record(self, rgb: RgbFrame, prior_action: int, action_id: int) -> None:
        self._trace.update(pack(">Q", self._step_count))
        self._trace.update(bytes.fromhex(rgb.sha256))
        self._trace.update(pack(">BB", prior_action, action_id))
        self._step_count += 1
        self._action_histogram[action_id] += 1

    @property
    def trajectory_input_action_sha256(self) -> str:
        return self._trace.hexdigest()

    @property
    def action_count(self) -> int:
        return self._step_count

    @property
    def action_histogram(self) -> tuple[int, ...]:
        return tuple(self._action_histogram)


class StrictRgbOnlyPolicy(_TraceRecorder):
    """Adapt a controller without exposing the environment observation.

    ``reset(episode_seed)`` intentionally does not forward ``episode_seed``.
    The controller sees ``model_reset_seed`` on every reset.  Its only decision
    inputs are the immutable RGB frame and this adapter's previous semantic
    action ID.  The first prior action is idle.
    """

    __slots__ = ("_controller", "_model_reset_seed", "_previous_action")
    identity = "irene.v21.strict_rgb_internal_prior.v1"
    uses_privileged_state = False

    def __init__(self, controller: RgbActionController, *, model_reset_seed: int) -> None:
        if not callable(getattr(controller, "reset", None)) or not callable(
            getattr(controller, "act_rgb", None)
        ):
            raise LivePlayQualificationError(
                "controller must define reset(seed) and act_rgb(rgb, previous_action_id)"
            )
        self._controller = controller
        self._model_reset_seed = _require_plain_int(
            model_reset_seed, name="model_reset_seed"
        )
        self._previous_action = 0
        self._reset_trace()

    def reset(self, episode_seed: int) -> None:
        _require_plain_int(episode_seed, name="episode_seed")
        self._previous_action = 0
        self._reset_trace()
        self._controller.reset(self._model_reset_seed)

    def act(self, observation: Observation) -> GenericControl:
        if not isinstance(observation, Observation):
            raise LivePlayQualificationError("policy input must be an Observation")
        prior_action = self._previous_action
        action_id = self._controller.act_rgb(observation.rgb, prior_action)
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise LivePlayQualificationError("controller action must be an integer")
        control = _action_control(action_id)
        self._record(observation.rgb, prior_action, action_id)
        self._previous_action = action_id
        return control


class _SplitMix64:
    """Small fixed PRNG with rejection-sampled uniform bounded draws."""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = seed & _UINT64_MASK

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & _UINT64_MASK
        value = self._state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
        return (value ^ (value >> 31)) & _UINT64_MASK

    def randbelow(self, upper_bound: int) -> int:
        _require_plain_int(upper_bound, name="upper_bound", minimum=1)
        limit = (1 << 64) - ((1 << 64) % upper_bound)
        while True:
            candidate = self.next_u64()
            if candidate < limit:
                return candidate % upper_bound


class SeededFiveActionRandomPolicy(_TraceRecorder):
    """Registered random floor: uniform over exactly idle/W/A/S/D."""

    __slots__ = ("_campaign_seed", "_previous_action", "_rng")
    identity = "diagnostic.uniform_idle_wasd.v1"
    uses_privileged_state = False

    def __init__(self, *, campaign_seed: int) -> None:
        self._campaign_seed = _require_plain_int(
            campaign_seed, name="campaign_seed"
        )
        self._rng = _SplitMix64(0)
        self._previous_action = 0
        self._reset_trace()

    def reset(self, episode_seed: int) -> None:
        episode = _require_plain_int(episode_seed, name="episode_seed")
        self._rng = _SplitMix64(
            self._campaign_seed
            ^ episode
            ^ 0x4952454E455F524E
        )
        self._previous_action = 0
        self._reset_trace()

    def act(self, observation: Observation) -> GenericControl:
        if not isinstance(observation, Observation):
            raise LivePlayQualificationError("policy input must be an Observation")
        action_id = self._rng.randbelow(len(ACTION_IDS))
        self._record(observation.rgb, self._previous_action, action_id)
        self._previous_action = action_id
        return _action_control(action_id)


@dataclass(frozen=True, slots=True)
class LivePlayProvenance:
    qualification_id: str
    model_identity: str
    checkpoint_envelope: str
    checkpoint_sha256: str
    model_state_sha256: str
    model_config_sha256: str
    feature_flags_sha256: str
    source_bundle_sha256: str
    evaluator_bundle_sha256: str
    environment_source_sha256: str
    preregistration_sha256: str
    cohort_binding_sha256: str
    post_dgx_release_body_sha256: str | None
    post_dgx_release_file_sha256: str | None

    def __post_init__(self) -> None:
        _require_text(self.qualification_id, name="qualification_id")
        _require_text(self.model_identity, name="model_identity")
        if self.checkpoint_envelope not in {
            DEVELOPMENT_CHECKPOINT_ENVELOPE,
            POST_DGX_CHECKPOINT_ENVELOPE,
        }:
            raise LivePlayQualificationError("checkpoint envelope is unsupported")
        for name in (
            "checkpoint_sha256",
            "model_state_sha256",
            "model_config_sha256",
            "feature_flags_sha256",
            "source_bundle_sha256",
            "evaluator_bundle_sha256",
            "environment_source_sha256",
            "preregistration_sha256",
            "cohort_binding_sha256",
        ):
            _require_digest(getattr(self, name), name=name)
        release_digests = (
            self.post_dgx_release_body_sha256,
            self.post_dgx_release_file_sha256,
        )
        if self.checkpoint_envelope == DEVELOPMENT_CHECKPOINT_ENVELOPE:
            if any(value is not None for value in release_digests):
                raise LivePlayQualificationError(
                    "development provenance cannot carry a post-DGX release"
                )
        else:
            for name, value in zip(
                (
                    "post_dgx_release_body_sha256",
                    "post_dgx_release_file_sha256",
                ),
                release_digests,
            ):
                _require_digest(value, name=name)

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_envelope": self.checkpoint_envelope,
            "checkpoint_sha256": self.checkpoint_sha256,
            "cohort_binding_sha256": self.cohort_binding_sha256,
            "environment_source_sha256": self.environment_source_sha256,
            "evaluator_bundle_sha256": self.evaluator_bundle_sha256,
            "model_config_sha256": self.model_config_sha256,
            "feature_flags_sha256": self.feature_flags_sha256,
            "model_state_sha256": self.model_state_sha256,
            "model_identity": self.model_identity,
            "post_dgx_release_body_sha256": self.post_dgx_release_body_sha256,
            "post_dgx_release_file_sha256": self.post_dgx_release_file_sha256,
            "preregistration_sha256": self.preregistration_sha256,
            "qualification_id": self.qualification_id,
            "source_bundle_sha256": self.source_bundle_sha256,
        }


@dataclass(frozen=True, slots=True)
class LivePlayQualificationConfig:
    episode_seeds: tuple[int, ...]
    cluster_ids: tuple[str, ...]
    max_ticks: int
    model_reset_seed: int
    random_baseline_seed: int
    bootstrap_seed: int
    random_baseline_streams: int = 8
    bootstrap_resamples: int = 10_000
    confidence_level: float = 0.95
    minimum_reward_advantage: float = 0.0
    minimum_catch_advantage: float = 0.0
    deterministic_repeats: int = 2
    tick_period_ns: int = TICK_PERIOD_NS_60HZ
    decision_interval_ns: int = TICK_PERIOD_NS_60HZ
    inference_latency_ns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.episode_seeds, tuple) or not self.episode_seeds:
            raise LivePlayQualificationError("episode_seeds must be a nonempty tuple")
        if len(set(self.episode_seeds)) != len(self.episode_seeds):
            raise LivePlayQualificationError("episode_seeds must be unique")
        for seed in self.episode_seeds:
            _require_plain_int(seed, name="episode seed")
        if not isinstance(self.cluster_ids, tuple) or len(self.cluster_ids) != len(
            self.episode_seeds
        ):
            raise LivePlayQualificationError(
                "cluster_ids must be a tuple aligned one-to-one with episode_seeds"
            )
        if any(not isinstance(value, str) or not value for value in self.cluster_ids):
            raise LivePlayQualificationError("cluster_ids must contain nonempty strings")
        distinct_clusters = set(self.cluster_ids)
        if len(distinct_clusters) < 2:
            raise LivePlayQualificationError(
                "paired clustered bootstrap requires at least two clusters"
            )
        _require_plain_int(self.max_ticks, name="max_ticks", minimum=1)
        _require_plain_int(self.model_reset_seed, name="model_reset_seed")
        _require_plain_int(self.random_baseline_seed, name="random_baseline_seed")
        _require_plain_int(self.bootstrap_seed, name="bootstrap_seed")
        if self.random_baseline_streams != 8:
            raise LivePlayQualificationError(
                "live-play qualification requires eight random streams per episode"
            )
        _require_plain_int(
            self.bootstrap_resamples, name="bootstrap_resamples", minimum=1
        )
        confidence = _require_finite(self.confidence_level, name="confidence_level")
        if not 0.5 < confidence < 1.0:
            raise LivePlayQualificationError(
                "confidence_level must be strictly between 0.5 and 1.0"
            )
        _require_finite(
            self.minimum_reward_advantage, name="minimum_reward_advantage"
        )
        _require_finite(
            self.minimum_catch_advantage, name="minimum_catch_advantage"
        )
        if self.deterministic_repeats != 2:
            raise LivePlayQualificationError(
                "live-play qualification requires exactly two deterministic runs"
            )
        if self.tick_period_ns != TICK_PERIOD_NS_60HZ:
            raise LivePlayQualificationError(
                "live-play qualification requires the registered 60 Hz tick period"
            )
        if self.decision_interval_ns != TICK_PERIOD_NS_60HZ:
            raise LivePlayQualificationError(
                "live-play qualification requires one decision every 60 Hz tick"
            )
        _require_plain_int(self.inference_latency_ns, name="inference_latency_ns")
        if self.inference_latency_ns > TICK_PERIOD_NS_60HZ:
            raise LivePlayQualificationError(
                "declared inference latency exceeds the 60 Hz frame deadline"
            )

    def closed_loop_config(self) -> ClosedLoopPlayConfig:
        return ClosedLoopPlayConfig(
            episode_seeds=self.episode_seeds,
            max_ticks=self.max_ticks,
            tick_period_ns=self.tick_period_ns,
            decision_interval_ns=self.decision_interval_ns,
            inference_latency_ns=self.inference_latency_ns,
            decode_kind="exclusive_argmax_wasd_v1",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "action_ids": list(ACTION_IDS),
            "action_semantics": list(ACTION_SEMANTICS),
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "cluster_ids": list(self.cluster_ids),
            "confidence_level": self.confidence_level,
            "decision_interval_ns": self.decision_interval_ns,
            "deterministic_repeats": self.deterministic_repeats,
            "environment": exact_maze_environment_config(self.max_ticks),
            "episode_seeds": list(self.episode_seeds),
            "inference_latency_ns": self.inference_latency_ns,
            "max_ticks": self.max_ticks,
            "measurement_kind": "simulated_control_schedule_integrity_only",
            "minimum_catch_advantage": self.minimum_catch_advantage,
            "minimum_reward_advantage": self.minimum_reward_advantage,
            "model_reset_seed": self.model_reset_seed,
            "random_baseline_distribution": "uniform",
            "random_baseline_seed": self.random_baseline_seed,
            "random_baseline_streams": self.random_baseline_streams,
            "tick_period_ns": self.tick_period_ns,
        }


def live_play_workload_sha256(config: LivePlayQualificationConfig) -> str:
    """Bind the exact cohort, world, controls, schedule, and statistical gates."""

    if not isinstance(config, LivePlayQualificationConfig):
        raise LivePlayQualificationError(
            "config must be a LivePlayQualificationConfig"
        )
    return sha256(
        _canonical_json(
            {
                "config": config.to_dict(),
                "model_policy": StrictRgbOnlyPolicy.identity,
                "random_policy": SeededFiveActionRandomPolicy.identity,
                "schema": "irene.v21.live_play_workload.v1",
            }
        ).encode("utf-8")
    ).hexdigest()


def live_play_cohort_binding_sha256(receipt: Mapping[str, object]) -> str:
    if not isinstance(receipt, Mapping):
        raise LivePlayQualificationError("cohort receipt must be a mapping")
    digest = sha256(b"IRV21ILIVECOHORT\x01")
    digest.update(_canonical_json(dict(receipt)).encode("utf-8"))
    return digest.hexdigest()


def verify_live_play_cohort_receipt(
    receipt: Mapping[str, object],
    *,
    config: LivePlayQualificationConfig,
    provenance: LivePlayProvenance,
) -> None:
    """Verify runner-produced sealed cohort evidence before any play occurs."""

    if not isinstance(receipt, Mapping):
        raise LivePlayQualificationError("cohort receipt must be a mapping")
    namespace = receipt.get("namespace")
    if not isinstance(namespace, str) or not namespace or "TEST" in namespace.upper():
        raise LivePlayQualificationError("cohort namespace is missing or touches TEST")
    if receipt.get("sealed") is not True or receipt.get("test_accessed") is not False:
        raise LivePlayQualificationError("cohort receipt must be sealed with TEST unopened")
    if receipt.get("episode_seeds") != list(config.episode_seeds):
        raise LivePlayQualificationError("cohort seeds differ from live-play registration")
    if receipt.get("cluster_ids") != list(config.cluster_ids):
        raise LivePlayQualificationError("cohort clusters differ from live-play registration")
    if receipt.get("max_ticks") != config.max_ticks:
        raise LivePlayQualificationError("cohort max_ticks differs from live-play registration")
    if receipt.get("random_baseline_streams") != config.random_baseline_streams:
        raise LivePlayQualificationError(
            "cohort random stream count differs from live-play registration"
        )
    if receipt.get("preregistration_sha256") != provenance.preregistration_sha256:
        raise LivePlayQualificationError(
            "cohort receipt preregistration differs from live-play provenance"
        )
    if live_play_cohort_binding_sha256(receipt) != provenance.cohort_binding_sha256:
        raise LivePlayQualificationError("cohort binding digest mismatch")


@dataclass(frozen=True, slots=True)
class WallClockLatencyReceipt:
    """Externally measured, provenance-bound latency required for final pass.

    Simulated manual-clock timing can prove schedule/control integrity only.
    This distinct receipt binds a real wall-clock benchmark to the same
    checkpoint, source bundle, and workload.  Its fixed sample counts and
    thresholds match the V2.1i preregistration.
    """

    checkpoint_envelope: str
    checkpoint_sha256: str
    model_state_sha256: str
    feature_flags_sha256: str
    source_bundle_sha256: str
    workload_sha256: str
    benchmark_workload_sha256: str
    validated_artifact_receipt_sha256: str
    runtime_manifest_sha256: str
    device_manifest_sha256: str
    post_dgx_release_body_sha256: str
    post_dgx_release_file_sha256: str
    repetitions: int
    warmup_per_repetition: int
    measured_per_repetition: int
    model_p99_ns: int
    loop_p99_ns: int
    loop_p999_ns: int
    deadline_miss_rate: float
    passed: bool

    def __post_init__(self) -> None:
        if self.checkpoint_envelope != POST_DGX_CHECKPOINT_ENVELOPE:
            raise LivePlayQualificationError(
                "live-play wall-clock receipt requires a post-DGX checkpoint"
            )
        for name in (
            "checkpoint_sha256",
            "model_state_sha256",
            "feature_flags_sha256",
            "source_bundle_sha256",
            "workload_sha256",
            "benchmark_workload_sha256",
            "validated_artifact_receipt_sha256",
            "runtime_manifest_sha256",
            "device_manifest_sha256",
            "post_dgx_release_body_sha256",
            "post_dgx_release_file_sha256",
        ):
            _require_digest(getattr(self, name), name=name)
        if self.repetitions != 3:
            raise LivePlayQualificationError(
                "wall-clock receipt requires exactly three repetitions"
            )
        if self.warmup_per_repetition != 100:
            raise LivePlayQualificationError(
                "wall-clock receipt requires 100 warmups per repetition"
            )
        if self.measured_per_repetition != 5_000:
            raise LivePlayQualificationError(
                "wall-clock receipt requires 5,000 measured ticks per repetition"
            )
        for name in ("model_p99_ns", "loop_p99_ns", "loop_p999_ns"):
            _require_plain_int(getattr(self, name), name=name)
        miss_rate = _require_finite(
            self.deadline_miss_rate, name="deadline_miss_rate"
        )
        if not 0.0 <= miss_rate <= 1.0:
            raise LivePlayQualificationError("deadline_miss_rate must be in [0, 1]")
        derived_pass = (
            self.model_p99_ns <= 8_000_000
            and self.loop_p99_ns <= 12_000_000
            and self.loop_p999_ns <= TICK_PERIOD_NS_60HZ
            and self.deadline_miss_rate < 0.001
        )
        if not isinstance(self.passed, bool) or self.passed is not derived_pass:
            raise LivePlayQualificationError(
                "wall-clock receipt passed flag differs from derived fixed gates"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "checkpoint_envelope": self.checkpoint_envelope,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_state_sha256": self.model_state_sha256,
            "feature_flags_sha256": self.feature_flags_sha256,
            "benchmark_workload_sha256": self.benchmark_workload_sha256,
            "deadline_miss_rate": self.deadline_miss_rate,
            "device_manifest_sha256": self.device_manifest_sha256,
            "loop_p999_ns": self.loop_p999_ns,
            "loop_p99_ns": self.loop_p99_ns,
            "measured_per_repetition": self.measured_per_repetition,
            "model_p99_ns": self.model_p99_ns,
            "passed": self.passed,
            "post_dgx_release_body_sha256": self.post_dgx_release_body_sha256,
            "post_dgx_release_file_sha256": self.post_dgx_release_file_sha256,
            "repetitions": self.repetitions,
            "runtime_manifest_sha256": self.runtime_manifest_sha256,
            "source_bundle_sha256": self.source_bundle_sha256,
            "validated_artifact_receipt_sha256": (
                self.validated_artifact_receipt_sha256
            ),
            "warmup_per_repetition": self.warmup_per_repetition,
            "workload_sha256": self.workload_sha256,
        }

    @property
    def sha256(self) -> str:
        return sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


def verify_wall_clock_latency_receipt(
    receipt: WallClockLatencyReceipt,
    *,
    config: LivePlayQualificationConfig,
    provenance: LivePlayProvenance,
) -> None:
    if not isinstance(receipt, WallClockLatencyReceipt):
        raise LivePlayQualificationError(
            "a separately measured WallClockLatencyReceipt is required"
        )
    expected_workload = live_play_workload_sha256(config)
    if provenance.checkpoint_envelope != POST_DGX_CHECKPOINT_ENVELOPE:
        raise LivePlayQualificationError(
            "production live-play provenance requires a post-DGX checkpoint"
        )
    if receipt.checkpoint_envelope != provenance.checkpoint_envelope:
        raise LivePlayQualificationError(
            "wall-clock checkpoint envelope differs from live-play provenance"
        )
    if receipt.checkpoint_sha256 != provenance.checkpoint_sha256:
        raise LivePlayQualificationError(
            "wall-clock receipt checkpoint differs from live-play provenance"
        )
    if receipt.model_state_sha256 != provenance.model_state_sha256:
        raise LivePlayQualificationError(
            "wall-clock receipt model state differs from live-play provenance"
        )
    if receipt.feature_flags_sha256 != provenance.feature_flags_sha256:
        raise LivePlayQualificationError(
            "wall-clock receipt feature flags differ from live-play provenance"
        )
    if receipt.source_bundle_sha256 != provenance.source_bundle_sha256:
        raise LivePlayQualificationError(
            "wall-clock receipt source bundle differs from live-play provenance"
        )
    if (
        receipt.post_dgx_release_body_sha256
        != provenance.post_dgx_release_body_sha256
    ):
        raise LivePlayQualificationError(
            "wall-clock post-DGX release body differs from live-play provenance"
        )
    if (
        receipt.post_dgx_release_file_sha256
        != provenance.post_dgx_release_file_sha256
    ):
        raise LivePlayQualificationError(
            "wall-clock post-DGX release file differs from live-play provenance"
        )
    if receipt.workload_sha256 != expected_workload:
        raise LivePlayQualificationError(
            "wall-clock receipt workload differs from registered live-play workload"
        )
    from .v21_wallclock_latency import wallclock_workload_sha256

    if receipt.benchmark_workload_sha256 != wallclock_workload_sha256():
        raise LivePlayQualificationError(
            "wall-clock benchmark workload digest differs from its frozen utility"
        )
    if not receipt.passed:
        raise LivePlayQualificationError("wall-clock latency receipt did not pass")


def wall_clock_latency_receipt_from_artifact(
    artifact: Mapping[str, object],
    *,
    config: LivePlayQualificationConfig,
    provenance: LivePlayProvenance,
) -> WallClockLatencyReceipt:
    """Validate the wall-clock utility artifact and adapt its immutable receipt."""

    from .v21_wallclock_latency import (
        validate_wallclock_artifact,
        wallclock_workload_sha256,
    )

    validated = validate_wallclock_artifact(
        artifact,
        expected_checkpoint_sha256=provenance.checkpoint_sha256,
        expected_source_bundle_sha256=provenance.source_bundle_sha256,
        expected_preregistration_sha256=provenance.preregistration_sha256,
        required_checkpoint_envelope=POST_DGX_CHECKPOINT_ENVELOPE,
        expected_post_dgx_release_receipt_sha256=(
            provenance.post_dgx_release_file_sha256
        ),
    )
    repetitions = validated.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != 3:
        raise LivePlayQualificationError(
            "validated wall-clock receipt lacks three repetitions"
        )
    if any(not isinstance(value, Mapping) for value in repetitions):
        raise LivePlayQualificationError(
            "validated wall-clock repetition evidence is malformed"
        )
    typed = [value for value in repetitions if isinstance(value, Mapping)]
    runtime = validated.get("runtime")
    if not isinstance(runtime, Mapping):
        raise LivePlayQualificationError("validated wall-clock runtime is missing")
    receipt_digest = artifact.get("receipt_sha256")
    receipt = WallClockLatencyReceipt(
        checkpoint_envelope=_require_text(
            validated.get("checkpoint_envelope"),
            name="wall-clock checkpoint_envelope",
        ),
        checkpoint_sha256=provenance.checkpoint_sha256,
        model_state_sha256=_require_digest(
            validated.get("model_state_sha256"),
            name="wall-clock model_state_sha256",
        ),
        feature_flags_sha256=_require_digest(
            validated.get("feature_flags_sha256"),
            name="wall-clock feature_flags_sha256",
        ),
        source_bundle_sha256=provenance.source_bundle_sha256,
        workload_sha256=live_play_workload_sha256(config),
        benchmark_workload_sha256=wallclock_workload_sha256(),
        validated_artifact_receipt_sha256=_require_digest(
            receipt_digest, name="wall-clock artifact receipt sha256"
        ),
        runtime_manifest_sha256=sha256(
            _canonical_json(dict(runtime)).encode("utf-8")
        ).hexdigest(),
        device_manifest_sha256=sha256(
            _canonical_json(
                {
                    "device": runtime.get("device"),
                    "torch_interop_threads": runtime.get("torch_interop_threads"),
                    "torch_threads": runtime.get("torch_threads"),
                }
            ).encode("utf-8")
        ).hexdigest(),
        post_dgx_release_body_sha256=_require_digest(
            validated.get("post_dgx_release_body_sha256"),
            name="wall-clock post_dgx_release_body_sha256",
        ),
        post_dgx_release_file_sha256=_require_digest(
            validated.get("post_dgx_release_file_sha256"),
            name="wall-clock post_dgx_release_file_sha256",
        ),
        repetitions=3,
        warmup_per_repetition=100,
        measured_per_repetition=5_000,
        model_p99_ns=max(int(value["model_forward_p99_ns"]) for value in typed),
        loop_p99_ns=max(int(value["loop_p99_ns"]) for value in typed),
        loop_p999_ns=max(int(value["loop_p999_ns"]) for value in typed),
        deadline_miss_rate=max(
            float(value["deadline_miss_rate"]) for value in typed
        ),
        passed=bool(validated.get("overall_passed")),
    )
    verify_wall_clock_latency_receipt(
        receipt,
        config=config,
        provenance=provenance,
    )
    return receipt


@dataclass(frozen=True, slots=True)
class LiveEpisodeExecution:
    report: ClosedLoopEpisodeReport
    timings: tuple[DecisionTiming, ...]
    environment_id: str
    environment_config_sha256: str
    step_trajectory_sha256: str
    trajectory_step_count: int


@dataclass(frozen=True, slots=True)
class EpisodeTimingEvidence:
    measurement_kind: str
    tick_period_ns: int
    attempts: int
    submitted: int
    deadline_misses: int
    schedule_mismatches: int
    maximum_inference_ns: int
    p99_inference_ns: int
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "attempts": self.attempts,
            "deadline_misses": self.deadline_misses,
            "maximum_inference_ns": self.maximum_inference_ns,
            "measurement_kind": self.measurement_kind,
            "p99_inference_ns": self.p99_inference_ns,
            "passed": self.passed,
            "schedule_mismatches": self.schedule_mismatches,
            "submitted": self.submitted,
            "tick_period_ns": self.tick_period_ns,
        }


@dataclass(frozen=True, slots=True)
class LiveEpisodeEvidence:
    agent: str
    episode_seed: int
    cluster_id: str
    baseline_stream: int | None
    reward: float
    pellets: int
    catches: int
    cleared: bool
    ticks_advanced: int
    controls_total: int
    controls_valid: int
    action_histogram: tuple[int, ...]
    control_validity_passed: bool
    timing: EpisodeTimingEvidence
    environment_id: str
    environment_config_sha256: str
    input_action_sha256: str
    step_trajectory_sha256: str
    trajectory_step_count: int
    trajectory_sha256: str
    repeat_trajectory_sha256: str
    deterministic_repeat_match: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "action_histogram": list(self.action_histogram),
            "agent": self.agent,
            "baseline_stream": self.baseline_stream,
            "catches": self.catches,
            "cleared": self.cleared,
            "cluster_id": self.cluster_id,
            "control_validity_passed": self.control_validity_passed,
            "controls_total": self.controls_total,
            "controls_valid": self.controls_valid,
            "episode_seed": self.episode_seed,
            "environment_config_sha256": self.environment_config_sha256,
            "environment_id": self.environment_id,
            "input_action_sha256": self.input_action_sha256,
            "pellets": self.pellets,
            "reward": self.reward,
            "ticks_advanced": self.ticks_advanced,
            "timing": self.timing.to_dict(),
            "step_trajectory_sha256": self.step_trajectory_sha256,
            "trajectory_step_count": self.trajectory_step_count,
            "trajectory_sha256": self.trajectory_sha256,
            "repeat_trajectory_sha256": self.repeat_trajectory_sha256,
            "deterministic_repeat_match": self.deterministic_repeat_match,
        }


@dataclass(frozen=True, slots=True)
class PairedClusterBootstrap:
    direction: str
    resamples: int
    seed: int
    confidence_level: float
    pair_count: int
    cluster_count: int
    observed_mean_advantage: float
    lower_bound: float
    replicate_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_count": self.cluster_count,
            "confidence_level": self.confidence_level,
            "direction": self.direction,
            "lower_bound": self.lower_bound,
            "observed_mean_advantage": self.observed_mean_advantage,
            "pair_count": self.pair_count,
            "replicate_sha256": self.replicate_sha256,
            "resamples": self.resamples,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    passed: bool
    observed: object
    operator: str
    threshold: object

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "observed": self.observed,
            "operator": self.operator,
            "passed": self.passed,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class LivePlayQualificationReport:
    schema_version: int
    evidence_scope: str
    provenance: LivePlayProvenance
    config: LivePlayQualificationConfig
    cohort_receipt: Mapping[str, object]
    model_episodes: tuple[LiveEpisodeEvidence, ...]
    baseline_episodes: tuple[LiveEpisodeEvidence, ...]
    reward_bootstrap: PairedClusterBootstrap
    catch_bootstrap: PairedClusterBootstrap
    wall_clock_latency_receipt: WallClockLatencyReceipt
    gates: tuple[GateResult, ...]

    @property
    def gates_passed(self) -> bool:
        return bool(self.gates) and all(gate.passed for gate in self.gates)

    @property
    def passed(self) -> bool:
        return self.evidence_scope == "production" and self.gates_passed

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_episodes": [entry.to_dict() for entry in self.baseline_episodes],
            "catch_bootstrap": self.catch_bootstrap.to_dict(),
            "cohort_receipt": dict(self.cohort_receipt),
            "config": self.config.to_dict(),
            "evidence_scope": self.evidence_scope,
            "gates": [gate.to_dict() for gate in self.gates],
            "gates_passed": self.gates_passed,
            "model_episodes": [entry.to_dict() for entry in self.model_episodes],
            "passed": self.passed,
            "provenance": self.provenance.to_dict(),
            "reward_bootstrap": self.reward_bootstrap.to_dict(),
            "schema_version": self.schema_version,
            "wall_clock_latency_receipt": self.wall_clock_latency_receipt.to_dict(),
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


EpisodeRunner = Callable[
    [object, int, ClosedLoopPlayConfig],
    LiveEpisodeExecution,
]


class _ExactMazeStepTrace:
    """Exact maze wrapper hashing full transition evidence at every boundary."""

    __slots__ = ("_environment", "_step_count", "_trace")

    def __init__(self, environment: MazeChaseEnv) -> None:
        if type(environment) is not MazeChaseEnv:
            raise LivePlayQualificationError("production world must be MazeChaseEnv")
        self._environment = environment
        self._trace = sha256(_STEP_TRACE_DOMAIN)
        self._step_count = 0

    @property
    def tick_period_ns(self) -> int:
        return self._environment.tick_period_ns

    @property
    def current_observation(self) -> Observation:
        return self._environment.current_observation

    def reset(self, seed: int) -> Observation:
        observation = self._environment.reset(seed)
        self._trace = sha256(_STEP_TRACE_DOMAIN)
        self._trace.update(pack(">Q", seed))
        self._step_count = 0
        return observation

    def step(self, control: GenericControl) -> StepOutcome:
        before = self._environment.current_observation
        outcome = self._environment.step(control)
        self._trace.update(pack(">Q", self._step_count))
        self._trace.update(bytes.fromhex(before.rgb.sha256))
        self._trace.update(bytes.fromhex(sha256(control.canonical_bytes()).hexdigest()))
        self._trace.update(
            bytes.fromhex(sha256(outcome.requested_control.canonical_bytes()).hexdigest())
        )
        self._trace.update(
            bytes.fromhex(sha256(outcome.applied_control.canonical_bytes()).hexdigest())
        )
        self._trace.update(pack(">d", float(outcome.reward)))
        self._trace.update(pack(">I", len(outcome.events)))
        for event in outcome.events:
            encoded = event.encode("utf-8")
            self._trace.update(pack(">I", len(encoded)))
            self._trace.update(encoded)
        self._trace.update(pack(">??", outcome.terminated, outcome.truncated))
        self._trace.update(bytes.fromhex(outcome.observation.rgb.sha256))
        self._step_count += 1
        return outcome

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def sha256(self) -> str:
        return self._trace.hexdigest()


def _exact_maze_factory(max_ticks: int) -> _ExactMazeStepTrace:
    environment = MazeChaseEnv(
        ghost_count=5,
        ghost_period=1,
        player_period=1,
        extra_loops=16,
        ghost_rule="direct",
        ghost_elroy=False,
        input_delay_ticks=0,
        sticky_direction=False,
        tick_period_ns=TICK_PERIOD_NS_60HZ,
        max_ticks=max_ticks,
    )
    expected = exact_maze_environment_config(max_ticks)
    actual = {
        "environment_id": EXACT_MAZE_ENVIRONMENT_ID,
        "extra_loops": environment._extra_loops,
        "ghost_count": environment._ghost_count,
        "ghost_elroy": environment._ghost_elroy,
        "ghost_period": environment._ghost_period,
        "ghost_rule": environment._ghost_rule,
        "input_delay_ticks": environment._input_delay_ticks,
        "max_ticks": environment._max_ticks,
        "player_period": environment._player_period,
        "sticky_direction": environment._sticky_direction,
        "tick_period_ns": environment._tick_period_ns,
    }
    if actual != expected:
        raise LivePlayQualificationError(
            "constructed maze does not match the registered exact configuration"
        )
    return _ExactMazeStepTrace(environment)


def run_in_repo_live_episode(
    policy: object,
    seed: int,
    config: ClosedLoopPlayConfig,
) -> LiveEpisodeExecution:
    """Run the exact registered maze and retain timing plus full step traces."""

    for attribute in ("reset", "act"):
        if not callable(getattr(policy, attribute, None)):
            raise LivePlayQualificationError(f"policy must define {attribute}()")
    policy.reset(seed)
    timings: list[DecisionTiming] = []
    environments: list[_ExactMazeStepTrace] = []

    def environment_factory() -> _ExactMazeStepTrace:
        environment = _exact_maze_factory(config.max_ticks)
        environments.append(environment)
        return environment

    def decide(
        observation: Observation, elapsed_seconds: float
    ) -> tuple[GenericControl, Mapping[str, int | float], None]:
        del elapsed_seconds
        control = policy.act(observation)
        if not isinstance(control, GenericControl):
            raise LivePlayQualificationError("policy must return GenericControl")
        return control, control_audit_stats(control), None

    report = _run_episode_core(
        seed=seed,
        config=config,
        decide=decide,
        environment_factory=environment_factory,
        attempt_sink=timings.append,
    )
    if len(environments) != 1:
        raise LivePlayQualificationError("exact maze factory invocation count differs from one")
    traced = environments[0]
    return LiveEpisodeExecution(
        report=report,
        timings=tuple(timings),
        environment_id=EXACT_MAZE_ENVIRONMENT_ID,
        environment_config_sha256=exact_maze_environment_config_sha256(
            config.max_ticks
        ),
        step_trajectory_sha256=traced.sha256,
        trajectory_step_count=traced.step_count,
    )


def _nearest_rank(values: Sequence[int], probability: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _timing_evidence(
    execution: LiveEpisodeExecution,
    *,
    config: LivePlayQualificationConfig,
) -> EpisodeTimingEvidence:
    timings = execution.timings
    schedule_mismatches = 0
    if timings:
        origin = timings[0].scheduled_ns
        for index, timing in enumerate(timings):
            if timing.action_sequence != index + 1:
                schedule_mismatches += 1
            if timing.scheduled_ns != origin + index * TICK_PERIOD_NS_60HZ:
                schedule_mismatches += 1
    inference = [
        timing.inference_finished_ns - timing.scheduled_ns for timing in timings
    ]
    if any(value < 0 for value in inference):
        raise LivePlayQualificationError("timing evidence runs backwards")
    submitted = sum(int(timing.submitted) for timing in timings)
    misses = sum(
        int(
            (not timing.submitted)
            or duration > TICK_PERIOD_NS_60HZ
            or duration != config.inference_latency_ns
        )
        for timing, duration in zip(timings, inference)
    )
    report = execution.report
    passed = bool(timings) and all(
        (
            schedule_mismatches == 0,
            misses == 0,
            submitted == len(timings),
            report.decisions_submitted == len(timings),
            report.decisions_rejected == 0,
            report.observations_dropped == 0,
            all(
                timing.failure_reason is None
                for timing in timings
                if timing.submitted
            ),
            config.tick_period_ns == TICK_PERIOD_NS_60HZ,
            config.decision_interval_ns == TICK_PERIOD_NS_60HZ,
        )
    )
    return EpisodeTimingEvidence(
        measurement_kind="simulated_control_schedule_integrity_only",
        tick_period_ns=TICK_PERIOD_NS_60HZ,
        attempts=len(timings),
        submitted=submitted,
        deadline_misses=misses,
        schedule_mismatches=schedule_mismatches,
        maximum_inference_ns=max(inference, default=0),
        p99_inference_ns=_nearest_rank(inference, 0.99),
        passed=passed,
    )


def _episode_evidence(
    *,
    agent: str,
    seed: int,
    cluster_id: Hashable,
    baseline_stream: int | None,
    execution: LiveEpisodeExecution,
    policy: _TraceRecorder,
    config: LivePlayQualificationConfig,
) -> LiveEpisodeEvidence:
    report = execution.report
    if report.episode_seed != seed:
        raise LivePlayQualificationError("episode runner returned the wrong seed")
    expected_environment_config = exact_maze_environment_config_sha256(
        config.max_ticks
    )
    if execution.environment_id != EXACT_MAZE_ENVIRONMENT_ID:
        raise LivePlayQualificationError(
            "episode runner did not execute the registered exact maze environment"
        )
    if execution.environment_config_sha256 != expected_environment_config:
        raise LivePlayQualificationError(
            "episode environment configuration differs from the registered exact maze"
        )
    timing = _timing_evidence(execution, config=config)
    action_histogram = policy.action_histogram
    controls_total = policy.action_count
    histogram_well_formed = all(
        isinstance(mask, int)
        and not isinstance(mask, bool)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        for mask, count in report.movement_mask_histogram
    )
    masks = {mask for mask, count in report.movement_mask_histogram if count > 0}
    valid_controls = (
        histogram_well_formed
        and len(action_histogram) == len(ACTION_IDS)
        and sum(action_histogram) == controls_total
        and sum(count for _mask, count in report.movement_mask_histogram)
        == controls_total
        and controls_total == len(execution.timings)
        and controls_total == report.ticks_advanced
        and execution.trajectory_step_count == report.ticks_advanced
        and masks.issubset(set(_ACTION_MASKS))
        and report.opposite_conflicts == 0
        and report.non_movement_key_activations == 0
        and report.continuous_outside_deadzone == 0
        and report.continuous_max_abs == 0.0
        and 0 < report.ticks_advanced <= config.max_ticks
    )
    mask_histogram = dict(report.movement_mask_histogram)
    semantic_histogram_matches = all(
        mask_histogram.get(mask, 0) == action_histogram[action_id]
        for action_id, mask in enumerate(_ACTION_MASKS)
    ) and len(mask_histogram) == len(
        [mask for mask in _ACTION_MASKS if mask_histogram.get(mask, 0) > 0]
    )
    valid_controls = valid_controls and semantic_histogram_matches
    cleared = report.ticks_advanced < config.max_ticks
    payload = {
        "action_histogram": list(action_histogram),
        "agent": agent,
        "baseline_stream": baseline_stream,
        "catches": report.collisions,
        "cleared": cleared,
        "cluster_id": str(cluster_id),
        "episode_seed": seed,
        "environment_config_sha256": execution.environment_config_sha256,
        "environment_id": execution.environment_id,
        "input_action_sha256": policy.trajectory_input_action_sha256,
        "step_trajectory_sha256": execution.step_trajectory_sha256,
        "trajectory_step_count": execution.trajectory_step_count,
        "pellets": report.pellets_eaten,
        "reward": report.reward_sum,
        "timing": timing.to_dict(),
        "ticks_advanced": report.ticks_advanced,
    }
    return LiveEpisodeEvidence(
        agent=agent,
        episode_seed=seed,
        cluster_id=str(cluster_id),
        baseline_stream=baseline_stream,
        reward=_require_finite(report.reward_sum, name="episode reward"),
        pellets=_require_plain_int(report.pellets_eaten, name="pellets"),
        catches=_require_plain_int(report.collisions, name="catches"),
        cleared=cleared,
        ticks_advanced=_require_plain_int(
            report.ticks_advanced, name="ticks_advanced", minimum=1
        ),
        controls_total=controls_total,
        controls_valid=controls_total if valid_controls else 0,
        action_histogram=action_histogram,
        control_validity_passed=valid_controls,
        timing=timing,
        environment_id=execution.environment_id,
        environment_config_sha256=execution.environment_config_sha256,
        input_action_sha256=policy.trajectory_input_action_sha256,
        step_trajectory_sha256=execution.step_trajectory_sha256,
        trajectory_step_count=execution.trajectory_step_count,
        trajectory_sha256=sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
        repeat_trajectory_sha256="",
        deterministic_repeat_match=False,
    )


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_clustered_bootstrap(
    advantages: Sequence[float],
    cluster_ids: Sequence[Hashable],
    *,
    direction: str,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> PairedClusterBootstrap:
    """Resample whole clusters while preserving each model/baseline pair."""

    if not advantages or len(advantages) != len(cluster_ids):
        raise LivePlayQualificationError(
            "bootstrap advantages and cluster_ids must be aligned and nonempty"
        )
    values = tuple(
        _require_finite(value, name="paired advantage") for value in advantages
    )
    _require_plain_int(resamples, name="resamples", minimum=1)
    _require_plain_int(seed, name="seed")
    confidence = _require_finite(confidence_level, name="confidence_level")
    if not 0.5 < confidence < 1.0:
        raise LivePlayQualificationError(
            "confidence_level must be strictly between 0.5 and 1.0"
        )
    positions: dict[Hashable, int] = {}
    groups: list[list[int]] = []
    for index, cluster_id in enumerate(cluster_ids):
        try:
            group = positions.get(cluster_id)
        except TypeError as error:
            raise LivePlayQualificationError("cluster_ids must be hashable") from error
        if group is None:
            group = len(groups)
            positions[cluster_id] = group
            groups.append([])
        groups[group].append(index)
    if len(groups) < 2:
        raise LivePlayQualificationError(
            "paired clustered bootstrap requires at least two clusters"
        )

    rng = _SplitMix64(seed)
    replicates: list[float] = []
    digest = sha256(b"IRENE-V21-PAIRED-CLUSTER-BOOTSTRAP\x01")
    for _ in range(resamples):
        sampled: list[float] = []
        for _cluster_draw in range(len(groups)):
            sampled.extend(values[index] for index in groups[rng.randbelow(len(groups))])
        replicate = sum(sampled) / len(sampled)
        replicates.append(replicate)
        digest.update(pack(">d", replicate))
    return PairedClusterBootstrap(
        direction=_require_text(direction, name="direction"),
        resamples=resamples,
        seed=seed,
        confidence_level=confidence,
        pair_count=len(values),
        cluster_count=len(groups),
        observed_mean_advantage=sum(values) / len(values),
        lower_bound=_linear_quantile(replicates, 1.0 - confidence),
        replicate_sha256=digest.hexdigest(),
    )


def _build_report(
    *,
    provenance: LivePlayProvenance,
    evidence_scope: str,
    config: LivePlayQualificationConfig,
    cohort_receipt: Mapping[str, object],
    model_episodes: tuple[LiveEpisodeEvidence, ...],
    baseline_episodes: tuple[LiveEpisodeEvidence, ...],
    wall_clock_latency_receipt: WallClockLatencyReceipt,
) -> LivePlayQualificationReport:
    if len(model_episodes) != len(config.episode_seeds):
        raise LivePlayQualificationError("one model result is required per episode seed")
    expected_baselines = len(config.episode_seeds) * config.random_baseline_streams
    if len(baseline_episodes) != expected_baselines:
        raise LivePlayQualificationError(
            "eight random-stream results are required per episode seed"
        )
    if tuple(entry.episode_seed for entry in model_episodes) != config.episode_seeds:
        raise LivePlayQualificationError("model episode ordering differs from registration")
    expected_baseline_order = tuple(
        (seed, stream)
        for seed in config.episode_seeds
        for stream in range(config.random_baseline_streams)
    )
    if tuple(
        (entry.episode_seed, entry.baseline_stream) for entry in baseline_episodes
    ) != expected_baseline_order:
        raise LivePlayQualificationError(
            "baseline episode ordering differs from registration"
        )
    baseline_groups = tuple(
        baseline_episodes[
            index * config.random_baseline_streams :
            (index + 1) * config.random_baseline_streams
        ]
        for index in range(len(config.episode_seeds))
    )
    baseline_mean_rewards = tuple(
        sum(entry.reward for entry in group) / len(group) for group in baseline_groups
    )
    baseline_mean_catches = tuple(
        sum(entry.catches for entry in group) / len(group) for group in baseline_groups
    )
    reward_advantages = tuple(
        model.reward - baseline_reward
        for model, baseline_reward in zip(model_episodes, baseline_mean_rewards)
    )
    catch_advantages = tuple(
        baseline_catches - model.catches
        for model, baseline_catches in zip(model_episodes, baseline_mean_catches)
    )
    reward_bootstrap = paired_clustered_bootstrap(
        reward_advantages,
        config.cluster_ids,
        direction="model_reward_minus_random_reward",
        resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
        confidence_level=config.confidence_level,
    )
    catch_bootstrap = paired_clustered_bootstrap(
        catch_advantages,
        config.cluster_ids,
        direction="random_catches_minus_model_catches",
        resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed ^ 0x4341544348,
        confidence_level=config.confidence_level,
    )
    model_controls = all(entry.control_validity_passed for entry in model_episodes)
    baseline_controls = all(
        entry.control_validity_passed for entry in baseline_episodes
    )
    timing_passed = all(
        entry.timing.passed for entry in model_episodes + baseline_episodes
    )
    repeat_passed = all(
        entry.deterministic_repeat_match
        for entry in model_episodes + baseline_episodes
    )
    model_pellets = sum(entry.pellets for entry in model_episodes)
    verify_wall_clock_latency_receipt(
        wall_clock_latency_receipt,
        config=config,
        provenance=provenance,
    )
    verify_live_play_cohort_receipt(
        cohort_receipt,
        config=config,
        provenance=provenance,
    )
    gates = (
        GateResult(
            name="reward_paired_cluster_bootstrap",
            passed=(
                reward_bootstrap.lower_bound > config.minimum_reward_advantage
            ),
            observed=reward_bootstrap.lower_bound,
            operator=">",
            threshold=config.minimum_reward_advantage,
        ),
        GateResult(
            name="catch_paired_cluster_bootstrap",
            passed=(
                catch_bootstrap.lower_bound > config.minimum_catch_advantage
            ),
            observed=catch_bootstrap.lower_bound,
            operator=">",
            threshold=config.minimum_catch_advantage,
        ),
        GateResult(
            name="model_controls_valid",
            passed=model_controls,
            observed=sum(entry.controls_valid for entry in model_episodes),
            operator="==",
            threshold=sum(entry.controls_total for entry in model_episodes),
        ),
        GateResult(
            name="random_controls_valid",
            passed=baseline_controls,
            observed=sum(entry.controls_valid for entry in baseline_episodes),
            operator="==",
            threshold=sum(entry.controls_total for entry in baseline_episodes),
        ),
        GateResult(
            name="model_pellets_earned",
            passed=model_pellets > 0,
            observed=model_pellets,
            operator=">",
            threshold=0,
        ),
        GateResult(
            name="deterministic_repeat_trajectory_equality",
            passed=repeat_passed,
            observed=sum(
                int(entry.deterministic_repeat_match)
                for entry in model_episodes + baseline_episodes
            ),
            operator="==",
            threshold=len(model_episodes) + len(baseline_episodes),
        ),
        GateResult(
            name="simulated_control_schedule_integrity",
            passed=timing_passed,
            observed=sum(
                entry.timing.deadline_misses
                for entry in model_episodes + baseline_episodes
            ),
            operator="==",
            threshold=0,
        ),
        GateResult(
            name="separate_bound_wall_clock_latency_receipt",
            passed=wall_clock_latency_receipt.passed,
            observed=wall_clock_latency_receipt.sha256,
            operator="verified_pass",
            threshold=True,
        ),
    )
    return LivePlayQualificationReport(
        schema_version=SCHEMA_VERSION,
        evidence_scope=evidence_scope,
        provenance=provenance,
        config=config,
        cohort_receipt=dict(cohort_receipt),
        model_episodes=model_episodes,
        baseline_episodes=baseline_episodes,
        reward_bootstrap=reward_bootstrap,
        catch_bootstrap=catch_bootstrap,
        wall_clock_latency_receipt=wall_clock_latency_receipt,
        gates=gates,
    )


def _write_open_artifact(handle: BinaryIO, payload: object) -> None:
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())


def _core_v2_state_sha256(model: object) -> str:
    state_dict = model.state_dict()
    if not isinstance(state_dict, Mapping):
        raise LivePlayQualificationError("controller model state_dict is invalid")
    digest = sha256(b"IRV21ISTATE\x01")
    for name, value in sorted(state_dict.items()):
        if not isinstance(name, str) or not hasattr(value, "detach"):
            raise LivePlayQualificationError("controller state_dict is malformed")
        tensor = value.detach().to(device="cpu").contiguous()
        encoded_name = name.encode("utf-8")
        encoded_dtype = str(tensor.dtype).encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(encoded_dtype).to_bytes(2, "big"))
        digest.update(encoded_dtype)
        digest.update(len(tensor.shape).to_bytes(2, "big"))
        for dimension in tensor.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _core_v2_config_sha256(model: object) -> str:
    try:
        payload = asdict(model.config)
    except (AttributeError, TypeError) as error:
        raise LivePlayQualificationError("controller model config is invalid") from error
    digest = sha256(b"IRV21IMODELCONFIG\x01")
    digest.update(_canonical_json(payload).encode("utf-8"))
    return digest.hexdigest()


def _core_v2_feature_flags_sha256(model: object) -> str:
    try:
        payload = asdict(model.flags)
    except (AttributeError, TypeError) as error:
        raise LivePlayQualificationError("controller feature flags are invalid") from error
    digest = sha256(b"IRV21IFEATUREFLAGS\x01")
    digest.update(_canonical_json(payload).encode("utf-8"))
    return digest.hexdigest()


def _verify_production_checkpoint_lineage(provenance: LivePlayProvenance) -> None:
    if provenance.checkpoint_envelope != POST_DGX_CHECKPOINT_ENVELOPE:
        raise LivePlayQualificationError(
            "production live-play qualification requires a post-DGX checkpoint"
        )
    _require_digest(
        provenance.post_dgx_release_body_sha256,
        name="post_dgx_release_body_sha256",
    )
    _require_digest(
        provenance.post_dgx_release_file_sha256,
        name="post_dgx_release_file_sha256",
    )


def _verify_exact_production_controller(
    controller: object,
    provenance: LivePlayProvenance,
    config: LivePlayQualificationConfig,
) -> None:
    from ..v2.maze_policy import OutcomeAwareCoreV2MazePolicy

    if type(controller) is not OutcomeAwareCoreV2MazePolicy:
        raise LivePlayQualificationError(
            "production qualification requires exact OutcomeAwareCoreV2MazePolicy"
        )
    if provenance.model_identity != OutcomeAwareCoreV2MazePolicy.identity:
        raise LivePlayQualificationError(
            "provenance model identity differs from the exact production policy"
        )
    if config.model_reset_seed != controller.model_seed:
        raise LivePlayQualificationError(
            "registered model reset seed differs from the exact production policy"
        )
    if _core_v2_state_sha256(controller.model) != provenance.model_state_sha256:
        raise LivePlayQualificationError(
            "controller model state differs from checkpoint provenance"
        )
    if _core_v2_config_sha256(controller.model) != provenance.model_config_sha256:
        raise LivePlayQualificationError(
            "controller model config differs from checkpoint provenance"
        )
    if (
        _core_v2_feature_flags_sha256(controller.model)
        != provenance.feature_flags_sha256
    ):
        raise LivePlayQualificationError(
            "controller feature flags differ from checkpoint provenance"
        )


def _verify_exact_production_live_config(
    config: LivePlayQualificationConfig,
) -> None:
    expected_seeds = tuple((3 << 62) | index for index in range(64))
    expected_clusters = tuple(
        f"play-qual-environment-{index:02d}" for index in range(64)
    )
    if config.episode_seeds != expected_seeds:
        raise LivePlayQualificationError(
            "production PLAY-QUAL requires the exact ordered 64-seed top-bit-11 cohort"
        )
    if config.cluster_ids != expected_clusters:
        raise LivePlayQualificationError(
            "production PLAY-QUAL requires one registered cluster per environment seed"
        )
    if config.max_ticks != 600:
        raise LivePlayQualificationError("production PLAY-QUAL requires 600 ticks")
    if config.random_baseline_seed != 2_126_202_681:
        raise LivePlayQualificationError(
            "production PLAY-QUAL random baseline seed drifted"
        )
    if config.bootstrap_seed != 2_126_202_699:
        raise LivePlayQualificationError("production PLAY-QUAL bootstrap seed drifted")
    if config.bootstrap_resamples != 10_000 or config.confidence_level != 0.95:
        raise LivePlayQualificationError(
            "production PLAY-QUAL bootstrap contract drifted"
        )
    if (
        config.minimum_reward_advantage != 0.0
        or config.minimum_catch_advantage != 0.0
    ):
        raise LivePlayQualificationError(
            "production PLAY-QUAL advantage thresholds drifted"
        )


def qualify_live_play_create_only(
    controller: RgbActionController,
    *,
    config: LivePlayQualificationConfig,
    provenance: LivePlayProvenance,
    cohort_receipt: Mapping[str, object],
    wall_clock_artifact: Mapping[str, object],
    output_path: str | Path,
) -> LivePlayQualificationReport:
    """Production surface: exact MazeChase only, with no injectable runner.

    The production entrypoint must compute and verify ``provenance`` from the
    checkpoint, source bundles, preregistration, and sealed cohort before
    calling this function.  Arbitrary runners are available only through the
    underscore-prefixed test helper below and are not exported.
    """

    _verify_production_checkpoint_lineage(provenance)
    _verify_exact_production_live_config(config)
    _verify_exact_production_controller(controller, provenance, config)
    receipt = wall_clock_latency_receipt_from_artifact(
        wall_clock_artifact,
        config=config,
        provenance=provenance,
    )
    return _qualify_live_play_create_only(
        controller,
        config=config,
        provenance=provenance,
        cohort_receipt=cohort_receipt,
        wall_clock_latency_receipt=receipt,
        output_path=output_path,
        episode_runner=None,
        test_only=False,
    )


def _qualify_live_play_create_only_for_tests(
    controller: RgbActionController,
    *,
    config: LivePlayQualificationConfig,
    provenance: LivePlayProvenance,
    cohort_receipt: Mapping[str, object],
    wall_clock_latency_receipt: WallClockLatencyReceipt,
    output_path: str | Path,
    episode_runner: EpisodeRunner,
) -> LivePlayQualificationReport:
    """Internal mock seam.  Never call this for a production artifact."""

    if not callable(episode_runner):
        raise LivePlayQualificationError("episode_runner must be callable")
    return _qualify_live_play_create_only(
        controller,
        config=config,
        provenance=provenance,
        cohort_receipt=cohort_receipt,
        wall_clock_latency_receipt=wall_clock_latency_receipt,
        output_path=output_path,
        episode_runner=episode_runner,
        test_only=True,
    )


def _qualify_live_play_create_only(
    controller: RgbActionController,
    *,
    config: LivePlayQualificationConfig,
    provenance: LivePlayProvenance,
    cohort_receipt: Mapping[str, object],
    wall_clock_latency_receipt: WallClockLatencyReceipt,
    output_path: str | Path,
    episode_runner: EpisodeRunner | None,
    test_only: bool,
) -> LivePlayQualificationReport:
    """Run paired qualification and publish exactly one create-only artifact.

    The output file is atomically claimed before the first policy reset.  A
    collision therefore prevents gameplay rather than overwriting evidence.
    Any exception after the claim writes a canonical failure receipt and is
    re-raised.  A completed report with a failed gate is also written and then
    raised as :class:`LivePlayQualificationFailed`.
    """

    if not isinstance(config, LivePlayQualificationConfig):
        raise LivePlayQualificationError(
            "config must be a LivePlayQualificationConfig"
        )
    if not isinstance(provenance, LivePlayProvenance):
        raise LivePlayQualificationError("provenance must be LivePlayProvenance")
    destination = Path(output_path)
    if not destination.parent.is_dir():
        raise LivePlayQualificationError("artifact parent directory must already exist")
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError as error:
        raise LivePlayQualificationError(
            f"refusing to overwrite existing artifact: {destination}"
        ) from error

    with os.fdopen(descriptor, "wb") as handle:
        try:
            if not test_only:
                _verify_production_checkpoint_lineage(provenance)
                _verify_exact_production_live_config(config)
                _verify_exact_production_controller(controller, provenance, config)
            verify_wall_clock_latency_receipt(
                wall_clock_latency_receipt,
                config=config,
                provenance=provenance,
            )
            verify_live_play_cohort_receipt(
                cohort_receipt,
                config=config,
                provenance=provenance,
            )
            runner = episode_runner or (
                lambda policy, seed, closed_config: run_in_repo_live_episode(
                    policy, seed, closed_config
                )
            )
            closed_config = config.closed_loop_config()
            model_policy = StrictRgbOnlyPolicy(
                controller, model_reset_seed=config.model_reset_seed
            )
            model_episodes: list[LiveEpisodeEvidence] = []
            baseline_episodes: list[LiveEpisodeEvidence] = []

            def repeated_evidence(
                *,
                policy: _TraceRecorder,
                agent: str,
                seed: int,
                cluster_id: str,
                baseline_stream: int | None,
            ) -> LiveEpisodeEvidence:
                rows: list[LiveEpisodeEvidence] = []
                for _repeat in range(config.deterministic_repeats):
                    execution = runner(policy, seed, closed_config)
                    if not isinstance(execution, LiveEpisodeExecution):
                        raise LivePlayQualificationError(
                            "episode_runner must return LiveEpisodeExecution"
                        )
                    rows.append(
                        _episode_evidence(
                            agent=agent,
                            seed=seed,
                            cluster_id=cluster_id,
                            baseline_stream=baseline_stream,
                            execution=execution,
                            policy=policy,
                            config=config,
                        )
                    )
                first, second = rows
                matched = (
                    first.trajectory_sha256 == second.trajectory_sha256
                    and first.input_action_sha256 == second.input_action_sha256
                    and first.step_trajectory_sha256 == second.step_trajectory_sha256
                )
                return replace(
                    first,
                    repeat_trajectory_sha256=second.trajectory_sha256,
                    deterministic_repeat_match=matched,
                )

            for seed, cluster_id in zip(config.episode_seeds, config.cluster_ids):
                model_episodes.append(
                    repeated_evidence(
                        policy=model_policy,
                        agent="model",
                        seed=seed,
                        cluster_id=cluster_id,
                        baseline_stream=None,
                    )
                )
                for stream in range(config.random_baseline_streams):
                    random_policy = SeededFiveActionRandomPolicy(
                        campaign_seed=(
                            config.random_baseline_seed
                            ^ ((stream + 1) * 0x9E3779B97F4A7C15)
                        )
                    )
                    baseline_episodes.append(
                        repeated_evidence(
                            policy=random_policy,
                            agent="random",
                            seed=seed,
                            cluster_id=cluster_id,
                            baseline_stream=stream,
                        )
                    )
            report = _build_report(
                provenance=provenance,
                evidence_scope="test_only_mocked" if test_only else "production",
                config=config,
                cohort_receipt=cohort_receipt,
                model_episodes=tuple(model_episodes),
                baseline_episodes=tuple(baseline_episodes),
                wall_clock_latency_receipt=wall_clock_latency_receipt,
            )
            artifact = {
                "report": report.to_dict(),
                "report_sha256": report.sha256,
                "schema_version": SCHEMA_VERSION,
                "production_eligible": not test_only,
                "status": (
                    "test_only"
                    if test_only
                    else ("passed" if report.passed else "failed")
                ),
            }
            _write_open_artifact(handle, artifact)
            if not report.gates_passed:
                raise LivePlayQualificationFailed(report)
            return report
        except LivePlayQualificationFailed:
            raise
        except Exception as error:
            failure = {
                "config": config.to_dict(),
                "failure": {
                    "message": str(error),
                    "type": type(error).__name__,
                },
                "provenance": provenance.to_dict(),
                "cohort_receipt": (
                    dict(cohort_receipt)
                    if isinstance(cohort_receipt, Mapping)
                    else None
                ),
                "wall_clock_latency_receipt_sha256": (
                    wall_clock_latency_receipt.sha256
                    if isinstance(
                        wall_clock_latency_receipt, WallClockLatencyReceipt
                    )
                    else None
                ),
                "schema_version": SCHEMA_VERSION,
                "production_eligible": not test_only,
                "status": "test_only" if test_only else "failed",
            }
            _write_open_artifact(handle, failure)
            raise LivePlayQualificationError(
                "live-play qualification aborted; failure evidence was written"
            ) from error


__all__ = [
    "ACTION_IDS",
    "ACTION_SEMANTICS",
    "EXACT_MAZE_ENVIRONMENT_ID",
    "EpisodeTimingEvidence",
    "GateResult",
    "LiveEpisodeEvidence",
    "LiveEpisodeExecution",
    "LivePlayProvenance",
    "LivePlayQualificationConfig",
    "LivePlayQualificationError",
    "LivePlayQualificationFailed",
    "LivePlayQualificationReport",
    "PairedClusterBootstrap",
    "POST_DGX_CHECKPOINT_ENVELOPE",
    "RgbActionController",
    "SCHEMA_VERSION",
    "SeededFiveActionRandomPolicy",
    "StrictRgbOnlyPolicy",
    "TICK_PERIOD_NS_60HZ",
    "WallClockLatencyReceipt",
    "exact_maze_environment_config",
    "exact_maze_environment_config_sha256",
    "live_play_cohort_binding_sha256",
    "live_play_workload_sha256",
    "paired_clustered_bootstrap",
    "qualify_live_play_create_only",
    "run_in_repo_live_episode",
    "verify_live_play_cohort_receipt",
    "verify_wall_clock_latency_receipt",
    "wall_clock_latency_receipt_from_artifact",
]
