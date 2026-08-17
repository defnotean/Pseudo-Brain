"""Deterministic foreground trainer with exact optimizer-boundary resume."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import fsum
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from ..runtime.policy import Capability
from .checkpoint import (
    TrainerCursor,
    file_sha256,
    load_checkpoint,
    save_checkpoint,
)
from .config import TrainingConfig
from .metrics import JsonlMetricWriter, MetricRecord
from .protocol import (
    DeterministicBatchSource,
    StagedTrainingSystem,
    TrainingStepResult,
    TrainingSystem,
)


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    cursor: TrainerCursor
    last_checkpoint: Path
    checkpoint_sha256: str
    metrics_path: Path
    completed: bool = True
    stop_reason: str | None = None
    development_gate_path: Path | None = None
    invariance_report_path: Path | None = None
    final_development_path: Path | None = None


class Trainer:
    """Orchestrate one shared model in the calling process.

    The class creates no threads, processes, network clients, capture devices,
    or HID outputs. Calling :meth:`run` is the only operation that starts work.
    """

    def __init__(
        self,
        config: TrainingConfig,
        system: TrainingSystem,
        batch_source: DeterministicBatchSource,
        run_dir: str | os.PathLike[str],
        *,
        code_sha256: str,
    ) -> None:
        if not isinstance(config, TrainingConfig):
            raise ValueError("config must be a TrainingConfig")
        if not isinstance(system, TrainingSystem):
            raise ValueError("system does not satisfy TrainingSystem")
        if not isinstance(batch_source, DeterministicBatchSource):
            raise ValueError("batch_source does not satisfy DeterministicBatchSource")
        if len(code_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in code_sha256
        ):
            raise ValueError("code_sha256 must be a lowercase SHA-256 string")
        config.resource_policy.require(Capability.ARTIFACT_WRITE)
        self.config = config
        self.system = system
        self.batch_source = batch_source
        self.run_dir = Path(run_dir)
        self.code_sha256 = code_sha256
        self.cursor = TrainerCursor()
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self._metrics: JsonlMetricWriter | None = None
        self._invariance_batches_cache: tuple[object, ...] | None = None
        self._development_gate_path: Path | None = None
        self._development_gate_sha256: str | None = None
        self._invariance_report_path: Path | None = None
        self._final_development_path: Path | None = None
        self._resume_metric_tail: list[bytes] = []
        self._recorded_metric_keys: set[tuple[str, int]] = set()

    def run(
        self,
        *,
        resume_from: str | os.PathLike[str] | None = None,
        expected_checkpoint_sha256: str | None = None,
        stop_after_step: int | None = None,
    ) -> TrainingSummary:
        resuming = resume_from is not None
        self._prepare_run(resuming=resuming)
        if resuming:
            assert resume_from is not None
            self._resume(
                resume_from,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
            )
            if self.cursor.optimizer_step >= self.config.run.max_optimizer_steps:
                raise ValueError(
                    "resume checkpoint has reached or exceeded the configured "
                    "training budget"
                )
        elif self.cursor.optimizer_step > self.config.run.max_optimizer_steps:
            raise ValueError("checkpoint is beyond the configured training budget")
        self._assert_staged_cursor()
        self._assert_stage_for_next_update()
        if stop_after_step is not None:
            if (
                self.config.run.name != "dgx-rcq-v2-staging-canary"
                or self.config.schema_version != 3
                or type(stop_after_step) is not int
                or not self.cursor.optimizer_step < stop_after_step
                < self.config.run.max_optimizer_steps
            ):
                raise ValueError(
                    "stop_after_step is canary-only and must be an integer strictly "
                    "between the cursor and configured budget"
                )

        last_checkpoint: Path | None = None
        last_digest: str | None = None
        stop_reason: str | None = None
        while self.cursor.optimizer_step < self.config.run.max_optimizer_steps:
            microbatches, advanced = self._next_microbatches(
                self.cursor,
                self.config.optimization.gradient_accumulation_steps,
            )
            result = self.system.train_optimizer_step(microbatches)
            self.cursor = TrainerCursor(
                epoch=advanced.epoch,
                next_batch=advanced.next_batch,
                optimizer_step=self.cursor.optimizer_step + 1,
            )
            step = self.cursor.optimizer_step
            if step % self.config.logging.log_every_steps == 0 or step == 1:
                self._record("train", result)
            validation_result: TrainingStepResult | None = None
            if step % self.config.logging.evaluate_every_steps == 0:
                validation_result = self.evaluate(split="validation")
                self._record("validation", validation_result)
            boundary_stop = self._transition_stage_if_due(validation_result)
            if boundary_stop is not None:
                self._validate_invariance_if_active()
                last_checkpoint, last_digest = self._checkpoint(prune=not resuming)
                stop_reason = boundary_stop
                break
            if step % self.config.logging.checkpoint_every_steps == 0:
                self._validate_invariance_if_active()
                last_checkpoint, last_digest = self._checkpoint(prune=not resuming)
            if stop_after_step is not None and step == stop_after_step:
                if (
                    last_checkpoint is None
                    or int(last_checkpoint.stem.split("-")[-1]) != step
                ):
                    self._validate_invariance_if_active()
                    last_checkpoint, last_digest = self._checkpoint(
                        prune=not resuming
                    )
                stop_reason = "operational_stop"
                break
            self._assert_staged_cursor()
            self._assert_stage_for_next_update()

        if last_checkpoint is None or int(last_checkpoint.stem.split("-")[-1]) != (
            self.cursor.optimizer_step
        ):
            self._validate_invariance_if_active()
            last_checkpoint, last_digest = self._checkpoint(prune=not resuming)
        assert last_checkpoint is not None and last_digest is not None
        return TrainingSummary(
            cursor=self.cursor,
            last_checkpoint=last_checkpoint,
            checkpoint_sha256=last_digest,
            metrics_path=self.metrics_path,
            completed=(
                stop_reason is None
                and self.cursor.optimizer_step == self.config.run.max_optimizer_steps
            ),
            stop_reason=stop_reason,
            development_gate_path=self._development_gate_path,
            invariance_report_path=self._invariance_report_path,
            final_development_path=self._final_development_path,
        )

    def evaluate(self, *, split: str, max_batches: int | None = None) -> TrainingStepResult:
        if split not in {"validation", "test"}:
            raise ValueError("evaluation split must be validation or test")
        if self.config.schema_version == 3 and split == "test":
            raise ValueError(
                "schema-3 test data may be opened only by the once-only final evaluator"
            )
        limit = (
            self.config.logging.validation_batches
            if max_batches is None
            else max_batches
        )
        if type(limit) is not int or limit < 1:
            raise ValueError("max_batches must be a positive integer")
        results = tuple(
            self.system.evaluate_batch(batch)
            for batch in self.batch_source.iter_batches(
                split=split,
                epoch=0,
                start_batch=0,
                batch_size=self.config.optimization.batch_size,
                max_batches=min(
                    limit,
                    self.batch_source.batches_per_epoch(
                        split=split,
                        batch_size=self.config.optimization.batch_size,
                    ),
                ),
            )
        )
        if not results:
            raise RuntimeError(f"{split} produced no evaluation batches")
        samples = sum(result.samples for result in results)
        # Weight before multiplying so a finite, very large per-batch failure
        # metric does not overflow merely because it represents many samples.
        loss = fsum(
            result.loss * (result.samples / samples) for result in results
        )
        names = set.intersection(*(set(result.metrics) for result in results))
        metrics = {
            name: fsum(
                result.metrics[name] * (result.samples / samples)
                for result in results
            )
            for name in sorted(names)
        }
        return TrainingStepResult(loss=loss, metrics=metrics, samples=samples)

    def evaluate_checkpoint(
        self,
        checkpoint: str | os.PathLike[str],
        *,
        split: str = "test",
        max_batches: int | None = None,
        expected_checkpoint_sha256: str | None = None,
    ) -> TrainingStepResult:
        """Load one exact checkpoint, evaluate it, and append its metrics."""

        self._prepare_run(resuming=True)
        self._resume(
            checkpoint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        result = self.evaluate(split=split, max_batches=max_batches)
        self._record(split, result)
        return result

    def _prepare_run(self, *, resuming: bool) -> None:
        self.config.resource_policy.require(Capability.ARTIFACT_WRITE)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        resolved = self.run_dir / "resolved-config.json"
        encoded = (self.config.canonical_json + "\n").encode("utf-8")
        if resolved.exists():
            if resolved.read_bytes() != encoded:
                raise ValueError("run directory contains a different resolved configuration")
        else:
            _write_atomic(resolved, encoded)
        if not resuming and self.metrics_path.exists() and self.metrics_path.stat().st_size:
            raise FileExistsError(
                "run directory already contains metrics; pass an explicit checkpoint to resume"
            )
        self._metrics = JsonlMetricWriter(
            self.metrics_path,
            policy=self.config.resource_policy,
        )
        if self.metrics_path.exists() and self.metrics_path.stat().st_size:
            lines = _canonical_metric_lines(self.metrics_path.read_bytes())
            self._recorded_metric_keys = {
                (
                    json.loads(line[:-1].decode("utf-8"))["split"],
                    json.loads(line[:-1].decode("utf-8"))["step"],
                )
                for line in lines
            }
        else:
            self._recorded_metric_keys = set()

    def _next_microbatches(
        self,
        cursor: TrainerCursor,
        count: int,
    ) -> tuple[tuple[object, ...], TrainerCursor]:
        epoch = cursor.epoch
        next_batch = cursor.next_batch
        batches: list[object] = []
        batch_size = self.config.optimization.batch_size
        while len(batches) < count:
            total = self.batch_source.batches_per_epoch(
                split="train",
                batch_size=batch_size,
            )
            if next_batch >= total:
                epoch += 1
                next_batch = 0
                continue
            iterator = self.batch_source.iter_batches(
                split="train",
                epoch=epoch,
                start_batch=next_batch,
                batch_size=batch_size,
                max_batches=1,
            )
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError("batch source ended before its declared batch count") from error
            batches.append(batch)
            next_batch += 1
            if next_batch == total:
                epoch += 1
                next_batch = 0
        return tuple(batches), TrainerCursor(
            epoch=epoch,
            next_batch=next_batch,
            optimizer_step=cursor.optimizer_step,
        )

    def _record(self, split: str, result: TrainingStepResult) -> None:
        if self._metrics is None:
            raise RuntimeError("trainer has not prepared its run directory")
        metrics = dict(result.metrics)
        metrics["loss"] = result.loss
        metrics["samples"] = float(result.samples)
        if self.config.schema_version == 3:
            metrics["stage_index"] = float(self._staged_system().active_stage_index)
        record = MetricRecord(
            step=self.cursor.optimizer_step,
            epoch=self.cursor.epoch,
            split=split,
            metrics=metrics,
        )
        encoded = (record.canonical_json + "\n").encode("utf-8")
        key = (record.split, record.step)
        if self._resume_metric_tail:
            expected = self._resume_metric_tail.pop(0)
            if encoded != expected:
                raise ValueError(
                    "replayed metric record differs from the durable post-checkpoint tail"
                )
            return
        if key in self._recorded_metric_keys:
            raise ValueError("metric split/step is already durably recorded")
        self._metrics.append(record)
        self._recorded_metric_keys.add(key)

    def _checkpoint(self, *, prune: bool = True) -> tuple[Path, str]:
        self._assert_staged_cursor()
        stage_state = (
            self._staged_system().stage_checkpoint_state(
                global_optimizer_step=self.cursor.optimizer_step
            )
            if self.config.schema_version == 3
            else None
        )
        trainer_state = self._metrics_prefix_state() if stage_state is not None else None
        path = self.checkpoint_dir / f"step-{self.cursor.optimizer_step:08d}.pt"
        digest = save_checkpoint(
            path,
            cursor=self.cursor,
            system_state=self.system.checkpoint_state(),
            rng_state=self.system.capture_rng_state(),
            config_sha256=self.config.config_sha256,
            data_sha256=self.batch_source.manifest_sha256,
            code_sha256=self.code_sha256,
            runtime_fingerprint=self.system.runtime_fingerprint,
            policy=self.config.resource_policy,
            stage_state=stage_state,
            trainer_state=trainer_state,
        )
        pointer = {
            "schema_version": 1,
            "checkpoint": path.name,
            "checkpoint_sha256": digest,
            "optimizer_step": self.cursor.optimizer_step,
        }
        encoded = (
            json.dumps(pointer, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        _write_atomic(self.checkpoint_dir / "latest.json", encoded)
        if prune:
            self._prune_checkpoints()
        return path, digest

    def _prune_checkpoints(self) -> None:
        keep = self.config.logging.keep_last_checkpoints
        candidates = sorted(self.checkpoint_dir.glob("step-????????.pt"))
        for obsolete in candidates[:-keep]:
            if obsolete.parent.resolve() != self.checkpoint_dir.resolve():
                raise RuntimeError("checkpoint retention target escaped the run directory")
            obsolete.unlink()

    def _resume(
        self,
        path: str | os.PathLike[str],
        *,
        expected_checkpoint_sha256: str | None,
    ) -> None:
        checkpoint_path = Path(path)
        pointer_optimizer_step: int | None = None
        if expected_checkpoint_sha256 is None:
            pointer_path = checkpoint_path.parent / "latest.json"
            if not pointer_path.is_file():
                raise ValueError(
                    "resume requires an expected checkpoint SHA-256 or latest.json sidecar"
                )
            try:
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
                raise ValueError("checkpoint latest.json is not valid JSON") from error
            required = {
                "schema_version",
                "checkpoint",
                "checkpoint_sha256",
                "optimizer_step",
            }
            if not isinstance(pointer, dict) or set(pointer) != required:
                raise ValueError("checkpoint latest.json has incompatible fields")
            if type(pointer["schema_version"]) is not int or pointer["schema_version"] != 1:
                raise ValueError("checkpoint latest.json schema_version must be integer 1")
            if type(pointer["optimizer_step"]) is not int or pointer["optimizer_step"] < 0:
                raise ValueError("checkpoint latest.json optimizer_step must be nonnegative")
            pointer_optimizer_step = pointer["optimizer_step"]
            expected_name = f"step-{pointer_optimizer_step:08d}.pt"
            if (
                type(pointer["checkpoint"]) is not str
                or pointer["checkpoint"] != checkpoint_path.name
                or pointer["checkpoint"] != expected_name
            ):
                raise ValueError("checkpoint latest.json does not identify the requested file")
            pointer_digest = pointer["checkpoint_sha256"]
            if (
                type(pointer_digest) is not str
                or len(pointer_digest) != 64
                or any(character not in "0123456789abcdef" for character in pointer_digest)
            ):
                raise ValueError("checkpoint latest.json SHA-256 is malformed")
            expected_checkpoint_sha256 = pointer["checkpoint_sha256"]
        loaded = load_checkpoint(
            checkpoint_path,
            expected_config_sha256=self.config.config_sha256,
            expected_data_sha256=self.batch_source.manifest_sha256,
            expected_code_sha256=self.code_sha256,
            expected_runtime_fingerprint=self.system.runtime_fingerprint,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        if (
            pointer_optimizer_step is not None
            and pointer_optimizer_step != loaded.cursor.optimizer_step
        ):
            raise ValueError("checkpoint latest.json step disagrees with loaded cursor")
        gate_metric_bindings: list[tuple[int, Mapping[str, object], str]] = []
        if self.config.schema_version == 3:
            if loaded.stage_state is None:
                raise ValueError("schema-3 resume requires checkpoint stage state")
            if loaded.trainer_state is None:
                raise ValueError("schema-3 resume requires checkpoint trainer state")
            stage_state = loaded.stage_state
            stage_index = stage_state.get("stage_index")
            local_step = stage_state.get("stage_local_optimizer_step")
            if (
                type(stage_index) is not int
                or not 0 <= stage_index < len(self.config.stages)
                or type(local_step) is not int
            ):
                raise ValueError("checkpoint stage cursor fields must be integers")
            stage = self.config.stages[stage_index]
            if stage.start_optimizer_step + local_step != loaded.cursor.optimizer_step:
                raise ValueError("checkpoint global and stage-local cursors disagree")
            self._staged_system().prepare_stage_for_resume(stage_state)
            entry_report = None
            entry_gate_id = stage_state.get("entry_gate_id")
            if entry_gate_id != "none":
                entry_step = stage_state.get("entry_gate_step")
                entry_digest = stage_state.get("entry_gate_report_sha256")
                if type(entry_step) is not int or type(entry_digest) is not str:
                    raise ValueError("checkpoint entry-gate identity is malformed")
                gate_path = self.run_dir / (
                    f"development-gate-step-{entry_step:08d}.json"
                )
                if not gate_path.is_file() or file_sha256(gate_path) != entry_digest:
                    raise ValueError(
                        "staged resume requires its exact passed development-gate artifact"
                    )
                entry_report = self._load_entry_development_report(gate_path)
                entry_payload = _load_canonical_registered_json(
                    gate_path,
                    name="checkpoint entry-gate artifact",
                )
                entry_metrics = entry_payload.get("metrics")
                if not isinstance(entry_metrics, Mapping):
                    raise ValueError("checkpoint entry-gate metrics are missing")
                gate_metric_bindings.append(
                    (entry_step, entry_metrics, "entry development gate")
                )
                if entry_gate_id != "rcq_v2_development_v1":
                    raise ValueError("checkpoint entry-gate artifact has the wrong gate ID")
                self._development_gate_path = gate_path
                self._development_gate_sha256 = entry_digest
            completion_gate_id = stage_state.get("completion_gate_id")
            completion_passed = stage_state.get("completion_gate_passed")
            if completion_gate_id != "none" and completion_passed is not None:
                completion_step = stage_state.get("completion_gate_step")
                completion_digest = stage_state.get(
                    "completion_gate_report_sha256"
                )
                if (
                    type(completion_gate_id) is not str
                    or type(completion_passed) is not bool
                    or type(completion_step) is not int
                    or type(completion_digest) is not str
                ):
                    raise ValueError("checkpoint completion-gate identity is malformed")
                final_path = self.run_dir / (
                    f"final-development-step-{completion_step:08d}.json"
                )
                if (
                    not final_path.is_file()
                    or file_sha256(final_path) != completion_digest
                ):
                    raise ValueError(
                        "staged resume requires its exact final development artifact"
                    )
                final_payload = _load_canonical_registered_json(
                    final_path,
                    name="checkpoint completion-gate artifact",
                )
                if (
                    type(final_payload.get("schema_version")) is not int
                    or final_payload.get("schema_version") != 1
                    or final_payload.get("gate") != completion_gate_id
                    or final_payload.get("passed") is not completion_passed
                ):
                    raise ValueError(
                        "checkpoint completion-gate artifact disagrees with stage state"
                    )
                if completion_gate_id == "rcq_v2_value_development_v1":
                    if entry_report is None:
                        raise ValueError(
                            "value completion artifact lacks its passed entry report"
                        )
                    from ..evaluation.rcq_v2 import (
                        evaluate_rcq_v2_value_development,
                    )

                    reconstructed_completion = evaluate_rcq_v2_value_development(
                        TrainingStepResult(
                            loss=0.0,
                            metrics=final_payload["metrics"],
                            samples=1_536,
                        ),
                        entry_report=entry_report,
                        entry_report_sha256=stage_state[
                            "entry_gate_report_sha256"
                        ],
                    )
                    if final_path.read_bytes() != (
                        reconstructed_completion.canonical_json + "\n"
                    ).encode("utf-8"):
                        raise ValueError(
                            "checkpoint value completion artifact cannot be reconstructed"
                        )
                    completion_metrics = final_payload.get("metrics")
                    if not isinstance(completion_metrics, Mapping):
                        raise ValueError("checkpoint completion-gate metrics are missing")
                    gate_metric_bindings.append(
                        (
                            completion_step,
                            completion_metrics,
                            "value completion gate",
                        )
                    )
                else:
                    raise ValueError("checkpoint completion gate ID is unsupported")
                self._final_development_path = final_path
        elif loaded.stage_state is not None:
            raise ValueError("legacy configurations refuse staged checkpoints")
        elif loaded.trainer_state is not None:
            raise ValueError("legacy configurations refuse staged trainer state")
        total = self.batch_source.batches_per_epoch(
            split="train",
            batch_size=self.config.optimization.batch_size,
        )
        if loaded.cursor.next_batch >= total and loaded.cursor.next_batch != 0:
            raise ValueError("checkpoint sampler cursor exceeds this dataset epoch")
        self.system.restore_checkpoint_state(loaded.system_state)
        self.cursor = loaded.cursor
        self._assert_staged_cursor()
        if loaded.trainer_state is not None:
            self._restore_metrics_state(loaded.trainer_state)
            for gate_step, gate_metrics, gate_name in gate_metric_bindings:
                self._assert_gate_metrics_match_durable_validation(
                    loaded.trainer_state,
                    gate_step=gate_step,
                    gate_metrics=gate_metrics,
                    gate_name=gate_name,
                )
        # Restore RNG last, after module/optimizer construction and state moves.
        self.system.restore_rng_state(loaded.rng_state)
        if self.config.schema_version == 3:
            self._validate_invariance_if_active()

    def _staged_system(self) -> StagedTrainingSystem:
        if not isinstance(self.system, StagedTrainingSystem):
            raise TypeError("schema-3 training requires a StagedTrainingSystem")
        return self.system

    def _assert_staged_cursor(self) -> None:
        if self.config.schema_version != 3:
            return
        batches_per_epoch = self.batch_source.batches_per_epoch(
            split="train",
            batch_size=self.config.optimization.batch_size,
        )
        consumed = (
            self.cursor.optimizer_step
            * self.config.optimization.gradient_accumulation_steps
        )
        expected_epoch, expected_batch = divmod(consumed, batches_per_epoch)
        if (self.cursor.epoch, self.cursor.next_batch) != (
            expected_epoch,
            expected_batch,
        ):
            raise ValueError(
                "schema-3 sampler cursor does not equal optimizer-step exposure"
            )

    def _assert_stage_for_next_update(self) -> None:
        if self.config.schema_version != 3:
            return
        step = self.cursor.optimizer_step
        if step == self.config.run.max_optimizer_steps:
            expected = len(self.config.stages) - 1
        else:
            expected = next(
                stage.index
                for stage in self.config.stages
                if stage.start_optimizer_step <= step < stage.end_optimizer_step
            )
        if self._staged_system().active_stage_index != expected:
            raise ValueError("active optimizer stage does not own the next update")

    def _invariance_batches(self) -> tuple[object, ...]:
        if self._invariance_batches_cache is None:
            total = self.batch_source.batches_per_epoch(
                split="validation",
                batch_size=self.config.optimization.batch_size,
            )
            batches = tuple(
                self.batch_source.iter_batches(
                    split="validation",
                    epoch=0,
                    start_batch=0,
                    batch_size=self.config.optimization.batch_size,
                    max_batches=total,
                )
            )
            if len(batches) != total or not batches:
                raise RuntimeError("invariance audit did not materialize the full dev slice")
            self._invariance_batches_cache = batches
        return self._invariance_batches_cache

    def _transition_stage_if_due(
        self,
        validation_result: TrainingStepResult | None,
    ) -> str | None:
        # Keep the dependency-free gate importable on its own: importing
        # training.protocol first executes training.__init__, which exports
        # Trainer and would otherwise form rcq_v2 -> protocol -> Trainer -> rcq_v2.
        from ..evaluation.rcq_v2 import (
            evaluate_rcq_v2_development,
            evaluate_rcq_v2_value_development,
        )

        if self.config.schema_version != 3:
            return None
        system = self._staged_system()
        stage = self.config.stages[system.active_stage_index]
        if self.cursor.optimizer_step != stage.end_optimizer_step:
            return None
        if stage.index == len(self.config.stages) - 1:
            if stage.completion_gate == "none":
                return None
            if validation_result is None:
                raise RuntimeError(
                    "the final schema-3 boundary requires same-step development evaluation"
                )
            if stage.completion_gate != "rcq_v2_value_development_v1":
                raise RuntimeError("unsupported registered completion gate")
            entry_path = self._development_gate_path or self.run_dir / (
                "development-gate-step-00001536.json"
            )
            entry_report = self._load_entry_development_report(entry_path)
            entry_digest = file_sha256(entry_path)
            if (
                self._development_gate_sha256 is None
                or entry_digest != self._development_gate_sha256
            ):
                raise RuntimeError(
                    "value completion entry artifact differs from transition receipt"
                )
            report = evaluate_rcq_v2_value_development(
                validation_result,
                entry_report=entry_report,
                entry_report_sha256=entry_digest,
            )
            path = self.run_dir / (
                f"final-development-step-{self.cursor.optimizer_step:08d}.json"
            )
            _write_registered(
                path,
                (report.canonical_json + "\n").encode("utf-8"),
            )
            self._final_development_path = path
            self._staged_system().bind_completion_gate(
                report_sha256=file_sha256(path),
                passed=report.passed,
                global_optimizer_step=self.cursor.optimizer_step,
            )
            if not report.passed:
                return "final_development_gate_failed"
            return None
        if stage.transition_gate == "rcq_v2_development_v1":
            if validation_result is None:
                raise RuntimeError(
                    "the registered stage transition requires same-step validation"
                )
            report = evaluate_rcq_v2_development(validation_result)
            path = self.run_dir / f"development-gate-step-{self.cursor.optimizer_step:08d}.json"
            gate_bytes = (report.canonical_json + "\n").encode("utf-8")
            _write_registered(path, gate_bytes)
            self._development_gate_path = path
            if not report.passed:
                return "development_gate_failed"
            gate_digest = file_sha256(path)
            self._development_gate_sha256 = gate_digest
        elif stage.transition_gate != "none":
            raise RuntimeError("unsupported registered stage transition gate")
        else:
            gate_digest = "0" * 64
        next_stage = self.config.stages[stage.index + 1]
        invariance_batches = (
            self._invariance_batches()
            if next_stage.invariance_audit != "none"
            else ()
        )
        transition_state = system.transition_to_stage(
            next_stage.index,
            invariance_batches=invariance_batches,
            entry_gate_report_sha256=gate_digest,
            entry_gate_passed=True,
            entry_gate_step=self.cursor.optimizer_step,
        )
        transition_path = self.run_dir / f"stage-transition-{next_stage.index:02d}.json"
        payload = {
            "schema_version": 1,
            "config_sha256": self.config.config_sha256,
            "global_optimizer_step": self.cursor.optimizer_step,
            "from_stage_index": stage.index,
            "to_stage_index": next_stage.index,
            "stage_state": dict(transition_state),
        }
        encoded = (
            json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        _write_registered(transition_path, encoded)
        return None

    def _load_entry_development_report(self, path: Path) -> object:
        from ..evaluation.rcq_v2 import evaluate_rcq_v2_development

        payload = _load_canonical_registered_json(
            path,
            name="entry development-gate artifact",
        )
        if not isinstance(payload.get("metrics"), Mapping):
            raise ValueError("entry development-gate metrics are missing")
        report = evaluate_rcq_v2_development(
            TrainingStepResult(
                loss=0.0,
                metrics=payload["metrics"],
                samples=1_536,
            )
        )
        if not report.passed or path.read_bytes() != (
            report.canonical_json + "\n"
        ).encode("utf-8"):
            raise ValueError("entry development-gate artifact cannot be reconstructed")
        return report

    def _validate_invariance_if_active(self) -> None:
        if self.config.schema_version != 3:
            return
        system = self._staged_system()
        stage = self.config.stages[system.active_stage_index]
        if stage.invariance_audit == "none":
            return
        report = system.validate_stage_invariance(
            invariance_batches=self._invariance_batches()
        )
        path = self.run_dir / (
            f"invariance-step-{self.cursor.optimizer_step:08d}.json"
        )
        payload = {
            "schema_version": 1,
            "config_sha256": self.config.config_sha256,
            "global_optimizer_step": self.cursor.optimizer_step,
            "report": dict(report),
        }
        encoded = (
            json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        _write_registered(path, encoded)
        self._invariance_report_path = path

    def _metrics_prefix_state(self) -> dict[str, object]:
        if self._resume_metric_tail:
            raise ValueError("cannot checkpoint before the durable metric tail is replayed")
        try:
            encoded = self.metrics_path.read_bytes()
        except OSError as error:
            raise ValueError("cannot bind the durable metrics prefix") from error
        records = _canonical_metric_lines(encoded)
        if not records:
            raise ValueError("a staged checkpoint requires a non-empty metrics prefix")
        observed_records = tuple(_metric_identity(line) for line in records)
        expected_records = self._expected_metric_records(self.cursor.optimizer_step)
        if observed_records != expected_records:
            raise ValueError(
                "metrics prefix does not exactly cover ordered cadence/epochs through cursor"
            )
        return {
            "schema_version": 1,
            "metrics_byte_length": len(encoded),
            "metrics_record_count": len(records),
            "metrics_sha256": sha256(encoded).hexdigest(),
        }

    def _restore_metrics_state(self, state: Mapping[str, object]) -> None:
        required = {
            "schema_version",
            "metrics_byte_length",
            "metrics_record_count",
            "metrics_sha256",
        }
        if not isinstance(state, Mapping) or set(state) != required:
            raise ValueError("checkpoint metrics state has incompatible fields")
        if type(state["schema_version"]) is not int or state["schema_version"] != 1:
            raise ValueError("checkpoint metrics schema_version must be integer 1")
        byte_length = state["metrics_byte_length"]
        record_count = state["metrics_record_count"]
        digest = state["metrics_sha256"]
        if type(byte_length) is not int or byte_length < 1:
            raise ValueError("checkpoint metrics byte length must be positive")
        if type(record_count) is not int or record_count < 1:
            raise ValueError("checkpoint metrics record count must be positive")
        if type(digest) is not str or len(digest) != 64:
            raise ValueError("checkpoint metrics SHA-256 is malformed")
        try:
            encoded = self.metrics_path.read_bytes()
        except OSError as error:
            raise ValueError("staged resume requires its durable metrics file") from error
        if len(encoded) < byte_length:
            raise ValueError("metrics file is shorter than the checkpoint-bound prefix")
        prefix = encoded[:byte_length]
        _canonical_metric_lines(encoded)
        if sha256(prefix).hexdigest() != digest:
            raise ValueError("metrics prefix differs from the checkpoint-bound SHA-256")
        prefix_lines = _canonical_metric_lines(prefix)
        if len(prefix_lines) != record_count:
            raise ValueError("metrics prefix record count differs from the checkpoint")
        observed_prefix_records = tuple(
            _metric_identity(line) for line in prefix_lines
        )
        expected_prefix_records = self._expected_metric_records(
            self.cursor.optimizer_step
        )
        if observed_prefix_records != expected_prefix_records:
            raise ValueError(
                "checkpoint metrics prefix disagrees with cursor, order, or epochs"
            )
        tail = encoded[byte_length:]
        tail_lines = _canonical_metric_lines(tail) if tail else ()
        remaining_records = self._expected_metric_records(
            self.config.run.max_optimizer_steps
        )[len(expected_prefix_records) :]
        observed_tail_records = tuple(_metric_identity(line) for line in tail_lines)
        if observed_tail_records != remaining_records[: len(observed_tail_records)]:
            raise ValueError(
                "durable metric tail is not an exact ordered cadence continuation"
            )
        self._resume_metric_tail = list(tail_lines)

    def _assert_gate_metrics_match_durable_validation(
        self,
        trainer_state: Mapping[str, object],
        *,
        gate_step: int,
        gate_metrics: Mapping[str, object],
        gate_name: str,
    ) -> None:
        """Bind a gate report to the checkpoint-owned same-step validation record."""

        byte_length = trainer_state.get("metrics_byte_length")
        if type(byte_length) is not int or byte_length < 1:
            raise ValueError("checkpoint gate binding lacks a metrics prefix boundary")
        encoded = self.metrics_path.read_bytes()
        prefix = encoded[:byte_length]
        expected_digest = trainer_state.get("metrics_sha256")
        if (
            len(prefix) != byte_length
            or type(expected_digest) is not str
            or sha256(prefix).hexdigest() != expected_digest
        ):
            raise ValueError("checkpoint gate binding metrics prefix changed")
        records = _canonical_metric_lines(prefix)
        matches: list[dict[str, object]] = []
        for line in records:
            payload = json.loads(line[:-1].decode("utf-8"))
            if payload.get("split") == "validation" and payload.get("step") == gate_step:
                matches.append(payload)
        if len(matches) != 1:
            raise ValueError(f"{gate_name} lacks one checkpoint-bound validation record")
        observed = matches[0].get("metrics")
        if not isinstance(observed, dict):
            raise ValueError(f"{gate_name} validation metrics are malformed")
        metadata = {"loss", "samples", "stage_index"}
        if set(observed) != set(gate_metrics) | metadata:
            raise ValueError(f"{gate_name} metrics differ from durable validation keys")
        for name, value in gate_metrics.items():
            if type(observed[name]) is not type(value) or observed[name] != value:
                raise ValueError(
                    f"{gate_name} metric {name!r} differs from durable validation"
                )
        total_loss = gate_metrics.get("total_loss")
        validation_loss = observed["loss"]
        if (
            type(total_loss) is not float
            or type(validation_loss) is not float
            or validation_loss.hex() != total_loss.hex()
        ):
            raise ValueError(f"{gate_name} loss differs from durable validation")
        if observed["samples"] != 1_536.0:
            raise ValueError(f"{gate_name} durable validation sample count changed")
        owner = next(
            (
                stage
                for stage in self.config.stages
                if stage.start_optimizer_step < gate_step <= stage.end_optimizer_step
            ),
            None,
        )
        if owner is None or observed["stage_index"] != float(owner.index):
            raise ValueError(f"{gate_name} durable validation stage changed")

    def _expected_metric_keys(self, optimizer_step: int) -> set[tuple[str, int]]:
        return {
            (split, step)
            for split, step, _epoch in self._expected_metric_records(optimizer_step)
        }

    def _expected_metric_records(
        self,
        optimizer_step: int,
    ) -> tuple[tuple[str, int, int], ...]:
        if type(optimizer_step) is not int or optimizer_step < 1:
            raise ValueError("staged metric cadence requires a positive cursor")
        batches_per_epoch = self.batch_source.batches_per_epoch(
            split="train",
            batch_size=self.config.optimization.batch_size,
        )
        records: list[tuple[str, int, int]] = []
        for step in range(1, optimizer_step + 1):
            epoch = (
                step * self.config.optimization.gradient_accumulation_steps
            ) // batches_per_epoch
            if step == 1 or step % self.config.logging.log_every_steps == 0:
                records.append(("train", step, epoch))
            if step % self.config.logging.evaluate_every_steps == 0:
                records.append(("validation", step, epoch))
        return tuple(records)


def _write_atomic(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_registered(path: Path, content: bytes) -> None:
    """Publish one immutable registration-derived artifact, or verify a replay."""

    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"registered artifact already differs: {path.name}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        if path.read_bytes() != content:
            raise ValueError(f"registered artifact already differs: {path.name}")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _canonical_metric_lines(encoded: bytes) -> tuple[bytes, ...]:
    if not isinstance(encoded, bytes):
        raise ValueError("metrics content must be bytes")
    if not encoded:
        return ()
    if not encoded.endswith(b"\n"):
        raise ValueError("metrics content must end at a complete newline record")
    lines = tuple(encoded.splitlines(keepends=True))
    seen: set[tuple[str, int]] = set()
    prior_step = -1
    for line in lines:
        if not line.endswith(b"\n") or line in {b"\n", b"\r\n"}:
            raise ValueError("metrics content contains an incomplete or blank record")
        try:
            raw = json.loads(line[:-1].decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("metrics content contains invalid JSON") from error
        required = {"schema_version", "step", "epoch", "split", "metrics"}
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("metrics content has an incompatible record schema")
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
            raise ValueError("metrics record schema_version must be integer 1")
        try:
            record = MetricRecord(
                step=raw["step"],
                epoch=raw["epoch"],
                split=raw["split"],
                metrics=raw["metrics"],
            )
        except (TypeError, ValueError) as error:
            raise ValueError("metrics content contains an invalid record") from error
        canonical = (record.canonical_json + "\n").encode("utf-8")
        if line != canonical:
            raise ValueError("metrics record is not byte-canonical")
        if record.step < prior_step:
            raise ValueError("metrics records are not step-monotonic")
        key = (record.split, record.step)
        if key in seen:
            raise ValueError("metrics content contains a duplicate split/step")
        seen.add(key)
        prior_step = record.step
    return lines


def _load_canonical_registered_json(path: Path, *, name: str) -> dict[str, object]:
    try:
        encoded = path.read_bytes()
        if not encoded.endswith(b"\n"):
            raise ValueError(f"{name} is not newline terminated")
        payload = json.loads(
            encoded[:-1].decode("utf-8", errors="strict"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
        if type(payload) is not dict:
            raise ValueError(f"{name} must contain one JSON object")
        canonical = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot validate {name}") from error
    if encoded != canonical:
        raise ValueError(f"{name} is not byte-canonical")
    return payload


def _metric_identity(line: bytes) -> tuple[str, int, int]:
    """Return the already-canonical record's ordered cadence identity."""

    payload = json.loads(line[:-1].decode("utf-8"))
    split = payload["split"]
    step = payload["step"]
    epoch = payload["epoch"]
    if type(split) is not str or type(step) is not int or type(epoch) is not int:
        raise ValueError("metric record identity has incompatible types")
    return split, step, epoch


__all__ = ["Trainer", "TrainingSummary"]
