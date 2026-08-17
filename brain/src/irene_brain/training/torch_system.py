"""PyTorch optimizer/mixed-precision adapter for the generic trainer."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, fields
from hashlib import sha256
import json
from math import isfinite
import os
import platform
import random
from types import MappingProxyType
from typing import ContextManager, Mapping, Sequence

import torch
from torch import Tensor, nn

from ..runtime.policy import Capability
from .batches import CONTROL_LAYOUT_ID, TrajectoryBatch, control_to_vector
from .config import TrainingConfig, TrainingStageConfig
from .objective import (
    LossOutput,
    _rgb_tensor,
    deterministic_eval_thought_noise,
)
from .protocol import TrainingStepResult
from .schedules import learning_rate_multiplier


class TorchTrainingSystem:
    """Own exactly one objective/model, optimizer, scheduler, and scaler."""

    def __init__(self, objective: nn.Module, config: TrainingConfig) -> None:
        if not isinstance(objective, nn.Module):
            raise ValueError("objective must be a torch.nn.Module")
        if not isinstance(config, TrainingConfig):
            raise ValueError("config must be a TrainingConfig")
        self.config = config
        self.device = torch.device(config.precision.device)
        if self.device.type == "cuda":
            config.resource_policy.require(Capability.GPU)
            if not torch.cuda.is_available():
                raise RuntimeError("configuration requires CUDA, but torch.cuda is unavailable")
        self._configure_determinism()
        self._seed_everything(config.run.seed, use_cuda=self.device.type == "cuda")
        self.objective = objective.to(self.device)
        self._active_stage_index = 0
        self._optimizer_parameter_names: tuple[str, ...] = ()
        self._optimizer_parameters: tuple[nn.Parameter, ...] = ()
        self._invariance_reference: dict[str, object] | None = None
        self._invariance_current: dict[str, object] | None = None
        self._resume_stage_local_step: int | None = None
        self._expected_initialized_optimizer_parameter_names: tuple[str, ...] | None = None
        self._entry_gate_state: dict[str, object] = {
            "entry_gate_id": "none",
            "entry_gate_report_sha256": "0" * 64,
            "entry_gate_passed": True,
            "entry_gate_step": 0,
        }
        self._completion_gate_state: dict[str, object] = {
            "completion_gate_id": (
                config.stages[0].completion_gate
                if config.schema_version == 3
                else "none"
            ),
            "completion_gate_report_sha256": "0" * 64,
            "completion_gate_passed": None,
            "completion_gate_step": 0,
        }
        self._validate_all_stage_parameter_masks()
        self._configure_optimizer(
            config.stages[0] if config.schema_version == 3 else None
        )
        self._runtime_fingerprint = MappingProxyType(self._build_runtime_fingerprint())

    @property
    def active_stage_index(self) -> int:
        return self._active_stage_index

    @property
    def optimizer_parameter_names(self) -> tuple[str, ...]:
        return self._optimizer_parameter_names

    def _active_stage(self) -> TrainingStageConfig | None:
        if self.config.schema_version < 3:
            return None
        return self.config.stages[self._active_stage_index]

    def _configure_optimizer(self, stage: TrainingStageConfig | None) -> None:
        self._expected_initialized_optimizer_parameter_names = None
        named = tuple(self.objective.named_parameters())
        available = {name: parameter for name, parameter in named}
        if stage is None or stage.trainable_parameters == ("*",):
            active_names = tuple(sorted(available))
        else:
            missing = sorted(set(stage.trainable_parameters) - set(available))
            if missing:
                raise ValueError(
                    "staged trainable parameters do not exist: " + ", ".join(missing)
                )
            active_names = tuple(stage.trainable_parameters)
        active_set = set(active_names)
        for name, parameter in named:
            parameter.requires_grad_(name in active_set)
        parameters = tuple(available[name] for name in active_names)
        if not parameters:
            raise ValueError("an optimizer stage must own at least one parameter")

        if stage is None:
            learning_rate = self.config.optimization.learning_rate
            weight_decay = self.config.optimization.weight_decay
        else:
            setter = getattr(self.objective, "set_loss_weights", None)
            if not callable(setter):
                raise TypeError("schema-3 objectives must implement set_loss_weights")
            setter(
                action_weight=stage.action_weight,
                value_weight=stage.value_weight,
                world_weight=stage.world_weight,
                diversity_weight=stage.diversity_weight,
            )
            learning_rate = stage.learning_rate
            weight_decay = stage.weight_decay
        self._optimizer_parameter_names = active_names
        self._optimizer_parameters = parameters
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._learning_rate_multiplier,
        )
        self.scaler = _make_grad_scaler(
            enabled=self.config.precision.mode == "float16",
        )
        optimizer_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != {id(parameter) for parameter in parameters}:
            raise RuntimeError("optimizer parameter identities do not match the freeze mask")
        if stage is not None and stage.invariance_audit == "nonvalue_action_state_v1":
            required = (
                "model.value_per_thought.bias",
                "model.value_per_thought.weight",
            )
            if active_names != required:
                raise ValueError(
                    "the nonvalue invariance stage must optimize exactly "
                    "model.value_per_thought weight and bias"
                )
        group = self.optimizer.param_groups[0]
        self._optimizer_group_contract = {
            name: value
            for name, value in group.items()
            if name not in {"params", "lr"}
        }
        self._optimizer_defaults_contract = dict(self.optimizer.defaults)
        initial_scheduler_state = self.scheduler.state_dict()
        self._scheduler_state_keys = frozenset(initial_scheduler_state)
        self._scheduler_static_contract = {
            name: value
            for name, value in initial_scheduler_state.items()
            if name not in {"base_lrs", "last_epoch", "_step_count", "_last_lr"}
        }
        initial_scaler_state = self.scaler.state_dict()
        self._scaler_state_keys = frozenset(initial_scaler_state)
        self._scaler_static_contract = {
            name: value
            for name, value in initial_scaler_state.items()
            if name not in {"scale", "_growth_tracker"}
        }
        self._validate_optimizer_contract(local_step=0)
        self._validate_scaler_contract()

    def _validate_all_stage_parameter_masks(self) -> None:
        if self.config.schema_version != 3:
            return
        available = {name for name, _parameter in self.objective.named_parameters()}
        for stage in self.config.stages:
            if stage.trainable_parameters == ("*",):
                resolved = tuple(sorted(available))
            else:
                missing = sorted(set(stage.trainable_parameters) - available)
                if missing:
                    raise ValueError(
                        f"stage {stage.index} trainable parameters do not exist: "
                        + ", ".join(missing)
                    )
                resolved = tuple(stage.trainable_parameters)
            if stage.invariance_audit == "nonvalue_action_state_v1" and resolved != (
                "model.value_per_thought.bias",
                "model.value_per_thought.weight",
            ):
                raise ValueError(
                    "nonvalue_action_state_v1 must resolve exactly the value head"
                )

    def _configure_determinism(self) -> None:
        deterministic = self.config.determinism.enabled
        torch.use_deterministic_algorithms(deterministic)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = deterministic
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = self.config.precision.allow_tf32
            if deterministic and self.device.type == "cuda":
                if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                    torch.backends.cuda.enable_flash_sdp(False)
                if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                if hasattr(torch.backends.cuda, "enable_math_sdp"):
                    torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = self.config.precision.allow_tf32
        torch.set_num_threads(self.config.resources.cpu_threads)

    @staticmethod
    def _seed_everything(seed: int, *, use_cuda: bool) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed_all(seed)

    def _learning_rate_multiplier(self, step: int) -> float:
        stage = self._active_stage()
        return learning_rate_multiplier(
            scheduler_kind=(
                self.config.optimization.scheduler_kind
                if stage is None
                else stage.scheduler_kind
            ),
            warmup_steps=(
                self.config.optimization.warmup_steps
                if stage is None
                else stage.warmup_steps
            ),
            max_optimizer_steps=(
                self.config.run.max_optimizer_steps
                if stage is None
                else stage.optimizer_steps
            ),
            step=step,
        )

    @property
    def runtime_fingerprint(self) -> Mapping[str, str | int | bool]:
        return self._runtime_fingerprint

    def _build_runtime_fingerprint(self) -> dict[str, str | int | bool]:
        model = getattr(self.objective, "model", self.objective)
        model_config = getattr(model, "config", None)
        config_value = vars(model_config) if hasattr(model_config, "__dict__") else repr(model_config)
        model_config_hash = sha256(
            json.dumps(config_value, default=repr, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cuda_version = torch.version.cuda or "none"
        device_name = (
            torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else "cpu"
        )
        pixel_encoder = getattr(model, "pixel_encoder", None)
        spatial_pool = getattr(pixel_encoder, "spatial_pool", None)
        cudnn_version = torch.backends.cudnn.version()
        compute_capability = (
            ".".join(str(part) for part in torch.cuda.get_device_capability(self.device))
            if self.device.type == "cuda"
            else "none"
        )
        return {
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "cuda_version": cuda_version,
            "cudnn_version": "none" if cudnn_version is None else int(cudnn_version),
            "compute_capability": compute_capability,
            "device_type": self.device.type,
            "device_name": device_name,
            "precision": self.config.precision.mode,
            "allow_tf32": self.config.precision.allow_tf32,
            "deterministic_algorithms": self.config.determinism.enabled,
            "compile_model": self.config.determinism.compile_model,
            "num_workers": self.config.determinism.num_workers,
            "world_size": 1,
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG", "unset"
            ),
            "sdpa_policy": (
                "math_only"
                if self.device.type == "cuda" and self.config.determinism.enabled
                else "runtime_default"
            ),
            "control_layout": CONTROL_LAYOUT_ID,
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "pixel_pool_class": (
                f"{type(spatial_pool).__module__}.{type(spatial_pool).__qualname__}"
            ),
            "model_config_sha256": model_config_hash,
        }

    def _autocast(self) -> ContextManager[object]:
        mode = self.config.precision.mode
        if mode == "float32":
            return nullcontext()
        dtype = torch.bfloat16 if mode == "bfloat16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def train_optimizer_step(
        self,
        microbatches: Sequence[object],
    ) -> TrainingStepResult:
        expected = self.config.optimization.gradient_accumulation_steps
        if not isinstance(microbatches, Sequence) or len(microbatches) != expected:
            raise ValueError(
                f"one optimizer step requires exactly {expected} microbatches"
            )
        self.objective.train(True)
        self.optimizer.zero_grad(set_to_none=True)
        samples_per_batch = tuple(_sample_count(batch) for batch in microbatches)
        total_samples = sum(samples_per_batch)
        metric_tensors: dict[str, Tensor] = {}
        mean_loss_tensor: Tensor | None = None

        for batch, sample_count in zip(microbatches, samples_per_batch):
            with self._autocast():
                output = _loss_output(self.objective(batch))
                loss = output.loss
            if loss.ndim != 0 or not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError("training loss must be a finite scalar")
            weight = sample_count / total_samples
            self.scaler.scale(loss * weight).backward()
            detached_loss = loss.detach().float()
            mean_loss_tensor = (
                detached_loss * weight
                if mean_loss_tensor is None
                else mean_loss_tensor + detached_loss * weight
            )
            for name, value in output.metrics.items():
                detached = value.detach().float() * weight
                metric_tensors[name] = metric_tensors.get(name, 0.0) + detached

        self.scaler.unscale_(self.optimizer)
        stage = self._active_stage()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self._optimizer_parameters,
            (
                self.config.optimization.max_gradient_norm
                if stage is None
                else stage.max_gradient_norm
            ),
            error_if_nonfinite=True,
        )
        applied_learning_rate = float(self.optimizer.param_groups[0]["lr"])
        prior_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scaler.is_enabled() and self.scaler.get_scale() < prior_scale:
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("FP16 gradient overflow skipped the optimizer step")
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        assert mean_loss_tensor is not None
        metric_tensors["gradient_norm"] = gradient_norm.detach().float()
        metric_tensors["learning_rate"] = mean_loss_tensor.new_tensor(
            applied_learning_rate
        )
        names = sorted(metric_tensors)
        packed = torch.stack((mean_loss_tensor, *(metric_tensors[name] for name in names)))
        if not bool(torch.isfinite(packed).all()):
            raise FloatingPointError("one or more training metrics became non-finite")
        host_values = packed.cpu().tolist()
        mean_loss = float(host_values[0])
        metrics = {
            name: float(value) for name, value in zip(names, host_values[1:])
        }
        # Learning rate is configuration telemetry, not a model tensor. Preserve
        # the exact Python value used by the optimizer instead of float32-rounding it.
        metrics["learning_rate"] = applied_learning_rate
        if stage is not None:
            metrics["stage_index"] = float(stage.index)
            metrics["stage_local_optimizer_step"] = float(
                self.scheduler.last_epoch
            )
        return TrainingStepResult(
            loss=mean_loss,
            metrics=metrics,
            samples=total_samples,
        )

    def evaluate_batch(self, batch: object) -> TrainingStepResult:
        self.objective.train(False)
        with torch.no_grad(), self._autocast():
            output = _loss_output(self.objective(batch))
        loss = float(output.loss.detach().float().cpu())
        metrics = {
            name: float(value.detach().float().cpu())
            for name, value in output.metrics.items()
        }
        return TrainingStepResult(loss=loss, metrics=metrics, samples=output.samples)

    def checkpoint_state(self) -> Mapping[str, object]:
        return {
            "objective": self.objective.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def restore_checkpoint_state(self, state: Mapping[str, object]) -> None:
        expected = {"objective", "optimizer", "scheduler", "scaler"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("checkpoint system state has incompatible fields")
        scaler_payload = state["scaler"]
        optimizer_payload = state["optimizer"]
        scheduler_payload = state["scheduler"]
        if not isinstance(optimizer_payload, Mapping) or set(optimizer_payload) != {
            "state",
            "param_groups",
        }:
            raise ValueError("checkpoint optimizer state has incompatible fields")
        if (
            not isinstance(scheduler_payload, Mapping)
            or frozenset(scheduler_payload) != self._scheduler_state_keys
        ):
            raise ValueError("checkpoint scheduler state has incompatible fields")
        if not isinstance(scaler_payload, Mapping):
            raise ValueError("checkpoint gradient-scaler state must be a mapping")
        if not self.scaler.is_enabled() and scaler_payload:
            raise ValueError("a disabled gradient scaler must have empty state")
        self.optimizer.load_state_dict(optimizer_payload)
        self.scheduler.load_state_dict(scheduler_payload)
        self.scaler.load_state_dict(scaler_payload)
        self._validate_scaler_contract()
        if self.config.schema_version == 3:
            expected_local_step = self._resume_stage_local_step
            if expected_local_step is None:
                raise ValueError("staged checkpoint restore was not prepared")
            if self.scheduler.last_epoch != expected_local_step:
                raise ValueError(
                    "checkpoint scheduler epoch disagrees with its stage-local cursor"
                )
            self._validate_optimizer_contract(local_step=expected_local_step)
        # Admit model weights only after all independently supplied training
        # machinery agrees with the registered stage and local cursor.
        self.objective.load_state_dict(state["objective"], strict=True)
        if self.config.schema_version == 3:
            if self._invariance_reference is not None:
                observed = _nonvalue_state_sha256(self.objective)
                expected = self._invariance_reference["non_value_state_sha256"]
                if observed != expected:
                    raise ValueError(
                        "checkpoint objective violates the frozen non-value reference"
                    )
            self._expected_initialized_optimizer_parameter_names = None
            self._resume_stage_local_step = None

    def restore_model_for_evaluation(
        self,
        system_state: Mapping[str, object],
        stage_state: Mapping[str, object],
    ) -> None:
        """Restore and validate one terminal schema-3 checkpoint for evaluation.

        Optimizer, scheduler, and scaler state remain unused by inference, but
        they are restored and checked against the stage-local cursor before the
        objective is admitted to the trusted final evaluator.
        """

        expected = {"objective", "optimizer", "scheduler", "scaler"}
        if not isinstance(system_state, Mapping) or set(system_state) != expected:
            raise ValueError("checkpoint system state has incompatible fields")
        normalized = _validated_stage_state(stage_state, config=self.config)
        stage_index = normalized["stage_index"]
        local_step = normalized["stage_local_optimizer_step"]
        if type(stage_index) is not int or stage_index != len(self.config.stages) - 1:
            raise ValueError("final evaluation requires the terminal training stage")
        stage = self.config.stages[stage_index]
        if type(local_step) is not int or local_step != stage.optimizer_steps:
            raise ValueError("final evaluation requires a completed terminal stage")
        if stage.completion_gate != "none":
            if normalized["completion_gate_passed"] is not True:
                raise ValueError("final evaluation requires a passed completion gate")
            if normalized["completion_gate_step"] != stage.end_optimizer_step:
                raise ValueError("completion gate is bound to the wrong optimizer step")

        self._active_stage_index = stage_index
        self._configure_optimizer(stage)
        if tuple(normalized["optimizer_parameter_names"]) != (
            self._optimizer_parameter_names
        ):
            raise ValueError(
                "final checkpoint optimizer parameters differ from the freeze mask"
            )
        if tuple(normalized["initialized_optimizer_parameter_names"]) != (
            self._optimizer_parameter_names
        ):
            raise ValueError(
                "final checkpoint must initialize AdamW state for both value-head parameters"
            )
        self.prepare_stage_for_resume(normalized)
        self.restore_checkpoint_state(system_state)
        # The trusted evaluator needs only the admitted objective. Rebuild fresh
        # stage-local training machinery so loaded Adam/scheduler/scaler state
        # cannot accidentally become a continuation path during evaluation.
        self._configure_optimizer(stage)
        self.objective.train(False)

    def transition_to_stage(
        self,
        stage_index: int,
        *,
        invariance_batches: Sequence[object],
        entry_gate_report_sha256: str,
        entry_gate_passed: bool,
        entry_gate_step: int,
    ) -> Mapping[str, object]:
        if self.config.schema_version != 3:
            raise ValueError("only schema-3 systems can transition optimizer stages")
        if type(stage_index) is not int or stage_index != self._active_stage_index + 1:
            raise ValueError("staged training transitions must advance exactly one stage")
        stage = self.config.stages[stage_index]
        prior_stage = self.config.stages[stage_index - 1]
        if not stage.reset_optimizer:
            raise ValueError("this staged trainer requires an explicit optimizer reset")
        if prior_stage.transition_gate != "none":
            _sha256_string(
                entry_gate_report_sha256,
                name="entry_gate_report_sha256",
            )
            if type(entry_gate_passed) is not bool or not entry_gate_passed:
                raise ValueError("a gated stage transition requires a passed gate")
            if type(entry_gate_step) is not int or entry_gate_step != prior_stage.end_optimizer_step:
                raise ValueError("entry gate step must equal the prior stage boundary")
            self._entry_gate_state = {
                "entry_gate_id": prior_stage.transition_gate,
                "entry_gate_report_sha256": entry_gate_report_sha256,
                "entry_gate_passed": True,
                "entry_gate_step": entry_gate_step,
            }
        else:
            if (
                entry_gate_report_sha256 != "0" * 64
                or entry_gate_passed is not True
                or entry_gate_step != prior_stage.end_optimizer_step
            ):
                raise ValueError("ungated transition metadata is inconsistent")
            self._entry_gate_state = {
                "entry_gate_id": "none",
                "entry_gate_report_sha256": "0" * 64,
                "entry_gate_passed": True,
                "entry_gate_step": entry_gate_step,
            }
        reference = (
            self._capture_invariance_snapshot(invariance_batches)
            if stage.invariance_audit == "nonvalue_action_state_v1"
            else None
        )
        self._active_stage_index = stage_index
        self._configure_optimizer(stage)
        self._completion_gate_state = {
            "completion_gate_id": stage.completion_gate,
            "completion_gate_report_sha256": "0" * 64,
            "completion_gate_passed": None,
            "completion_gate_step": 0,
        }
        if reference is not None:
            current = self._capture_invariance_snapshot(invariance_batches)
            _require_identical_invariance(reference, current)
            self._invariance_reference = reference
            self._invariance_current = current
        else:
            self._invariance_reference = None
            self._invariance_current = None
        return self.stage_checkpoint_state(
            global_optimizer_step=stage.start_optimizer_step
        )

    def prepare_stage_for_resume(self, stage_state: Mapping[str, object]) -> None:
        normalized = _validated_stage_state(stage_state, config=self.config)
        stage_index = normalized["stage_index"]
        assert isinstance(stage_index, int)
        self._active_stage_index = stage_index
        self._configure_optimizer(self.config.stages[stage_index])
        checkpoint_names = tuple(normalized["optimizer_parameter_names"])
        if self._optimizer_parameter_names != checkpoint_names:
            raise ValueError(
                "checkpoint optimizer parameters do not match the resolved freeze mask"
            )
        reference = normalized["invariance_reference"]
        current = normalized["invariance_current"]
        self._invariance_reference = (
            None if reference is None else dict(reference)  # type: ignore[arg-type]
        )
        self._invariance_current = (
            None if current is None else dict(current)  # type: ignore[arg-type]
        )
        self._entry_gate_state = {
            name: normalized[name]
            for name in (
                "entry_gate_id",
                "entry_gate_report_sha256",
                "entry_gate_passed",
                "entry_gate_step",
            )
        }
        self._completion_gate_state = {
            name: normalized[name]
            for name in (
                "completion_gate_id",
                "completion_gate_report_sha256",
                "completion_gate_passed",
                "completion_gate_step",
            )
        }
        local_step = normalized["stage_local_optimizer_step"]
        assert isinstance(local_step, int)
        self._resume_stage_local_step = local_step
        initialized_names = normalized["initialized_optimizer_parameter_names"]
        assert isinstance(initialized_names, list)
        self._expected_initialized_optimizer_parameter_names = tuple(
            initialized_names
        )

    def stage_checkpoint_state(
        self,
        *,
        global_optimizer_step: int,
    ) -> Mapping[str, object]:
        if self.config.schema_version != 3:
            raise ValueError("only schema-3 systems expose staged checkpoint state")
        if type(global_optimizer_step) is not int:
            raise ValueError("global_optimizer_step must be an integer")
        stage = self.config.stages[self._active_stage_index]
        if not (
            stage.start_optimizer_step
            <= global_optimizer_step
            <= stage.end_optimizer_step
        ):
            raise ValueError("global optimizer step is outside the active stage")
        local_step = global_optimizer_step - stage.start_optimizer_step
        if self.scheduler.last_epoch != local_step:
            raise ValueError(
                "scheduler epoch disagrees with the stage-local optimizer cursor"
            )
        self._validate_optimizer_contract(local_step=local_step)
        if self._invariance_reference is not None:
            current_nonvalue = _nonvalue_state_sha256(self.objective)
            if current_nonvalue != self._invariance_reference["non_value_state_sha256"]:
                raise RuntimeError("a non-value model tensor changed during the frozen stage")
            if self._invariance_current is not None:
                self._invariance_current["non_value_state_sha256"] = current_nonvalue
        initialized_optimizer_names = self._initialized_optimizer_parameter_names()
        state = {
            "schema_version": 1,
            "stage_index": stage.index,
            "stage_name": stage.name,
            "stage_start_optimizer_step": stage.start_optimizer_step,
            "stage_end_optimizer_step": stage.end_optimizer_step,
            "stage_local_optimizer_step": local_step,
            "stage_config_sha256": _stage_config_sha256(stage),
            "trainable_parameter_names": list(stage.trainable_parameters),
            "optimizer_parameter_names": list(self._optimizer_parameter_names),
            "optimizer_parameter_names_sha256": _names_sha256(
                self._optimizer_parameter_names
            ),
            "initialized_optimizer_parameter_names": list(initialized_optimizer_names),
            "initialized_optimizer_parameter_names_sha256": _names_sha256(
                initialized_optimizer_names
            ),
            "optimizer_reset": stage.reset_optimizer,
            "invariance_audit": stage.invariance_audit,
            "invariance_reference": self._invariance_reference,
            "invariance_current": self._invariance_current,
            **self._entry_gate_state,
            **self._completion_gate_state,
        }
        return _validated_stage_state(state, config=self.config)

    def _validate_optimizer_contract(self, *, local_step: int) -> None:
        """Fail closed if loaded optimizer/scheduler semantics drifted."""

        if type(local_step) is not int or local_step < 0:
            raise ValueError("optimizer contract requires a nonnegative local step")
        stage = self._active_stage()
        stage_steps = (
            self.config.run.max_optimizer_steps
            if stage is None
            else stage.optimizer_steps
        )
        if local_step > stage_steps:
            raise ValueError("optimizer local step exceeds its registered stage")
        if len(self.optimizer.param_groups) != 1:
            raise ValueError("checkpoint optimizer must contain exactly one parameter group")
        group = self.optimizer.param_groups[0]
        if set(group) != set(self._optimizer_group_contract) | {"params", "lr"}:
            raise ValueError("checkpoint optimizer parameter-group fields changed")
        if not _type_sensitive_equal(
            {name: group[name] for name in self._optimizer_group_contract},
            self._optimizer_group_contract,
        ):
            raise ValueError("checkpoint optimizer parameter-group options changed")
        if not _type_sensitive_equal(
            self.optimizer.defaults,
            self._optimizer_defaults_contract,
        ):
            raise ValueError("checkpoint AdamW defaults changed")
        parameters = tuple(group["params"])
        parameter_ids = tuple(id(parameter) for parameter in parameters)
        expected_parameter_ids = tuple(
            id(parameter) for parameter in self._optimizer_parameters
        )
        if parameter_ids != expected_parameter_ids:
            raise ValueError("checkpoint optimizer parameter order or identities changed")
        if len(set(parameter_ids)) != len(parameters):
            raise ValueError("checkpoint optimizer contains duplicate parameters")
        optimizer_state_ids = {id(parameter) for parameter in self.optimizer.state}
        if any(parameter_id not in set(parameter_ids) for parameter_id in optimizer_state_ids):
            raise ValueError("checkpoint optimizer state owns an unregistered parameter")
        if local_step == 0 and self.optimizer.state:
            raise ValueError("a reset stage at local step zero must have empty AdamW state")
        if local_step > 0:
            if not optimizer_state_ids:
                raise ValueError(
                    "checkpoint AdamW state must initialize at least one parameter"
                )
            for parameter in parameters:
                if id(parameter) not in optimizer_state_ids:
                    continue
                parameter_state = self.optimizer.state.get(parameter)
                if not isinstance(parameter_state, Mapping) or set(parameter_state) != {
                    "step",
                    "exp_avg",
                    "exp_avg_sq",
                }:
                    raise ValueError("checkpoint AdamW parameter state has changed fields")
                step_tensor = parameter_state["step"]
                if (
                    not isinstance(step_tensor, Tensor)
                    or step_tensor.numel() != 1
                    or step_tensor.dtype != torch.float32
                    or not bool(torch.isfinite(step_tensor).all())
                    or float(step_tensor.detach().cpu().item()) != float(local_step)
                ):
                    raise ValueError(
                        "checkpoint AdamW parameter step disagrees with local cursor"
                    )
                for moment_name in ("exp_avg", "exp_avg_sq"):
                    moment = parameter_state[moment_name]
                    if (
                        not isinstance(moment, Tensor)
                        or moment.shape != parameter.shape
                        or moment.dtype != parameter.dtype
                        or moment.device != parameter.device
                        or not bool(torch.isfinite(moment).all())
                        or (
                            moment_name == "exp_avg_sq"
                            and bool((moment < 0).any())
                        )
                    ):
                        raise ValueError(
                            f"checkpoint AdamW {moment_name} is incompatible with parameter"
                        )
        initialized_names = self._initialized_optimizer_parameter_names()
        if (
            self._expected_initialized_optimizer_parameter_names is not None
            and initialized_names
            != self._expected_initialized_optimizer_parameter_names
        ):
            raise ValueError(
                "checkpoint initialized AdamW parameter set differs from stage receipt"
            )

        base_learning_rate = (
            self.config.optimization.learning_rate
            if stage is None
            else stage.learning_rate
        )
        expected_learning_rate = base_learning_rate * self._learning_rate_multiplier(
            local_step
        )
        if type(group["lr"]) is not float or group["lr"] != expected_learning_rate:
            raise ValueError("checkpoint optimizer learning rate disagrees with schedule")
        scheduler_state = self.scheduler.state_dict()
        if frozenset(scheduler_state) != self._scheduler_state_keys:
            raise ValueError("checkpoint scheduler fields changed")
        if not _type_sensitive_equal(
            scheduler_state.get("base_lrs"), [base_learning_rate]
        ):
            raise ValueError("checkpoint scheduler base learning rate changed")
        if type(scheduler_state.get("last_epoch")) is not int or (
            scheduler_state["last_epoch"] != local_step
        ):
            raise ValueError("checkpoint scheduler epoch disagrees with local step")
        if type(scheduler_state.get("_step_count")) is not int or (
            scheduler_state["_step_count"] != local_step + 1
        ):
            raise ValueError("checkpoint scheduler step count disagrees with local step")
        if not _type_sensitive_equal(
            scheduler_state.get("_last_lr"), [expected_learning_rate]
        ):
            raise ValueError("checkpoint scheduler current learning rate changed")
        observed_static = {
            name: value
            for name, value in scheduler_state.items()
            if name not in {"base_lrs", "last_epoch", "_step_count", "_last_lr"}
        }
        if not _type_sensitive_equal(
            observed_static,
            self._scheduler_static_contract,
        ):
            raise ValueError("checkpoint scheduler static state changed")

    def _validate_scaler_contract(self) -> None:
        state = self.scaler.state_dict()
        if frozenset(state) != self._scaler_state_keys:
            raise ValueError("checkpoint gradient-scaler fields changed")
        if not self.scaler.is_enabled():
            if state:
                raise ValueError("a disabled gradient scaler must have empty state")
            return
        expected = {
            "scale",
            "growth_factor",
            "backoff_factor",
            "growth_interval",
            "_growth_tracker",
        }
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("checkpoint gradient-scaler fields changed")
        observed_static = {
            name: value
            for name, value in state.items()
            if name not in {"scale", "_growth_tracker"}
        }
        if not _type_sensitive_equal(observed_static, self._scaler_static_contract):
            raise ValueError("checkpoint gradient-scaler static semantics changed")
        scale = state["scale"]
        if type(scale) is not float or not isfinite(scale) or scale <= 0.0:
            raise ValueError("checkpoint gradient-scaler scale is invalid")
        if type(state["_growth_tracker"]) is not int or state["_growth_tracker"] < 0:
            raise ValueError("checkpoint gradient-scaler growth tracker is invalid")

    def _initialized_optimizer_parameter_names(self) -> tuple[str, ...]:
        initialized_ids = {id(parameter) for parameter in self.optimizer.state}
        return tuple(
            name
            for name, parameter in zip(
                self._optimizer_parameter_names,
                self._optimizer_parameters,
            )
            if id(parameter) in initialized_ids
        )

    def validate_stage_invariance(
        self,
        *,
        invariance_batches: Sequence[object],
    ) -> Mapping[str, object]:
        stage = self._active_stage()
        if stage is None or stage.invariance_audit == "none":
            return {
                "schema_version": 1,
                "stage_index": self._active_stage_index,
                "audit": "none",
                "passed": True,
            }
        if self._invariance_reference is None:
            raise RuntimeError("the frozen stage lacks its invariance reference")
        current = self._capture_invariance_snapshot(invariance_batches)
        _require_identical_invariance(self._invariance_reference, current)
        self._invariance_current = current
        return {
            "schema_version": 1,
            "stage_index": self._active_stage_index,
            "audit": stage.invariance_audit,
            "passed": True,
            "reference": dict(self._invariance_reference),
            "current": dict(current),
        }

    def bind_completion_gate(
        self,
        *,
        report_sha256: str,
        passed: bool,
        global_optimizer_step: int,
    ) -> None:
        stage = self._active_stage()
        if stage is None or stage.completion_gate == "none":
            raise ValueError("the active stage has no registered completion gate")
        _sha256_string(report_sha256, name="completion gate report sha256")
        if report_sha256 == "0" * 64:
            raise ValueError("completion gate report SHA-256 cannot be the null digest")
        if type(passed) is not bool:
            raise ValueError("completion gate status must be a boolean")
        if (
            type(global_optimizer_step) is not int
            or global_optimizer_step != stage.end_optimizer_step
        ):
            raise ValueError("completion gate must bind the exact final stage boundary")
        self._completion_gate_state = {
            "completion_gate_id": stage.completion_gate,
            "completion_gate_report_sha256": report_sha256,
            "completion_gate_passed": passed,
            "completion_gate_step": global_optimizer_step,
        }

    def _capture_invariance_snapshot(
        self,
        batches: Sequence[object],
    ) -> dict[str, object]:
        if not isinstance(batches, Sequence) or not batches:
            raise ValueError("invariance audit requires at least one deterministic batch")
        model = getattr(self.objective, "model", None)
        if not isinstance(model, nn.Module):
            raise TypeError("invariance audit requires objective.model")
        action_digest = sha256(b"IRENERCQACTION\x01")
        state_digest = sha256(b"IRENERCQSTATE\x01")
        batch_count = 0
        sequence_count = 0
        timestep_count = 0
        was_training = self.objective.training
        saved_requires_grad = tuple(
            (parameter, bool(parameter.requires_grad))
            for parameter in self.objective.parameters()
        )
        self.objective.train(False)
        try:
            for parameter, _flag in saved_requires_grad:
                parameter.requires_grad_(False)
            with torch.inference_mode():
                for raw_batch in batches:
                    if not isinstance(raw_batch, TrajectoryBatch):
                        raise ValueError(
                            "invariance audit requires TrajectoryBatch inputs"
                        )
                    batch = raw_batch
                    parameter = next(model.parameters())
                    state = None
                    thought_noise = deterministic_eval_thought_noise(
                        thoughtlets=model.config.thoughtlets,
                        width=model.config.core_width,
                        batch_size=batch.batch_size,
                        device=parameter.device,
                    )
                    for time_index in range(batch.sequence_length):
                        transitions = tuple(
                            sequence.transitions[time_index]
                            for sequence in batch.sequences
                        )
                        pixels = _rgb_tensor(
                            tuple(
                                transition.observation.rgb
                                for transition in transitions
                            ),
                            device=parameter.device,
                            resolution=getattr(model, "input_resolution", None),
                        )
                        previous_control = torch.tensor(
                            [
                                control_to_vector(
                                    transition.observation.previous_control
                                )
                                for transition in transitions
                            ],
                            dtype=torch.float32,
                            device=parameter.device,
                        )
                        delta_seconds = torch.tensor(
                            [
                                0.0
                                if time_index == 0
                                else (
                                    transition.observation.elapsed_ns
                                    - batch.sequences[batch_index]
                                    .transitions[time_index - 1]
                                    .observation.elapsed_ns
                                )
                                / 1_000_000_000.0
                                for batch_index, transition in enumerate(transitions)
                            ],
                            dtype=torch.float32,
                            device=parameter.device,
                        )
                        output = model(
                            pixels,
                            previous_control,
                            delta_seconds,
                            state,
                            thought_noise=thought_noise,
                        )
                        state = output.next_state
                        for exit_index, action in enumerate(output.anytime_actions):
                            for descriptor in fields(action):
                                _update_tensor_hash(
                                    action_digest,
                                    f"{batch_count}:{time_index}:{exit_index}:"
                                    f"{descriptor.name}",
                                    getattr(action, descriptor.name),
                                )
                        for descriptor in fields(state):
                            _update_tensor_hash(
                                state_digest,
                                f"{batch_count}:{time_index}:{descriptor.name}",
                                getattr(state, descriptor.name),
                            )
                        timestep_count += batch.batch_size
                    batch_count += 1
                    sequence_count += batch.batch_size
        finally:
            for parameter, flag in saved_requires_grad:
                parameter.requires_grad_(flag)
            self.objective.train(was_training)
        return {
            "schema_version": 1,
            # Captures run in eval + inference_mode with requires_grad cleared
            # so CUDA kernel selection cannot follow the live freeze mask. The
            # audit still uses the registered runtime TF32 policy and claims
            # same-runtime bit identity, not an IEEE-float32 numerical rerun.
            "evaluation_precision": "same_runtime_no_autocast_v1",
            "batch_count": batch_count,
            "sequence_count": sequence_count,
            "timestep_count": timestep_count,
            "non_value_state_sha256": _nonvalue_state_sha256(self.objective),
            "action_outputs_sha256": action_digest.hexdigest(),
            "recurrent_states_sha256": state_digest.hexdigest(),
        }

    def capture_rng_state(self) -> Mapping[str, object]:
        return {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
        }

    def restore_rng_state(self, state: Mapping[str, object]) -> None:
        expected = {"python", "torch_cpu", "torch_cuda"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("checkpoint RNG state has incompatible fields")
        random.setstate(state["python"])
        torch.set_rng_state(state["torch_cpu"])
        cuda_states = state["torch_cuda"]
        if self.device.type == "cuda":
            if not isinstance(cuda_states, list) or len(cuda_states) != torch.cuda.device_count():
                raise ValueError("checkpoint CUDA RNG state count is incompatible")
            torch.cuda.set_rng_state_all(cuda_states)
        elif cuda_states:
            raise ValueError("CPU resume refuses a checkpoint containing CUDA RNG state")


def _make_grad_scaler(*, enabled: bool) -> object:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _stage_config_sha256(stage: TrainingStageConfig) -> str:
    payload = asdict(stage)
    payload["trainable_parameters"] = list(stage.trainable_parameters)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(b"IRENESTAGE\x01" + encoded).hexdigest()


def _names_sha256(names: Sequence[str]) -> str:
    encoded = json.dumps(
        list(names),
        ensure_ascii=True,
        sort_keys=False,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(b"IRENEPARAMS\x01" + encoded).hexdigest()


def _sha256_string(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


def _type_sensitive_equal(left: object, right: object) -> bool:
    """Compare JSON-like optimizer metadata without bool/int coercion."""

    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        return set(left) == set(right) and all(
            _type_sensitive_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        assert isinstance(right, (list, tuple))
        return len(left) == len(right) and all(
            _type_sensitive_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def _snapshot(value: object, *, name: str) -> dict[str, object]:
    required = {
        "schema_version",
        "evaluation_precision",
        "batch_count",
        "sequence_count",
        "timestep_count",
        "non_value_state_sha256",
        "action_outputs_sha256",
        "recurrent_states_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{name} has incompatible fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError(f"{name}.schema_version must be integer 1")
    if value["evaluation_precision"] != "same_runtime_no_autocast_v1":
        raise ValueError(f"{name} must use same_runtime_no_autocast_v1")
    for field_name in ("batch_count", "sequence_count", "timestep_count"):
        field_value = value[field_name]
        if type(field_value) is not int or field_value < 1:
            raise ValueError(f"{name}.{field_name} must be a positive integer")
    for field_name in (
        "non_value_state_sha256",
        "action_outputs_sha256",
        "recurrent_states_sha256",
    ):
        _sha256_string(value[field_name], name=f"{name}.{field_name}")
    return dict(value)


def _validated_stage_state(
    value: object,
    *,
    config: TrainingConfig,
) -> dict[str, object]:
    required = {
        "schema_version",
        "stage_index",
        "stage_name",
        "stage_start_optimizer_step",
        "stage_end_optimizer_step",
        "stage_local_optimizer_step",
        "stage_config_sha256",
        "trainable_parameter_names",
        "optimizer_parameter_names",
        "optimizer_parameter_names_sha256",
        "initialized_optimizer_parameter_names",
        "initialized_optimizer_parameter_names_sha256",
        "optimizer_reset",
        "invariance_audit",
        "invariance_reference",
        "invariance_current",
        "entry_gate_id",
        "entry_gate_report_sha256",
        "entry_gate_passed",
        "entry_gate_step",
        "completion_gate_id",
        "completion_gate_report_sha256",
        "completion_gate_passed",
        "completion_gate_step",
    }
    if config.schema_version != 3:
        raise ValueError("stage state requires a schema-3 configuration")
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("checkpoint stage state has incompatible fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("checkpoint stage schema_version must be integer 1")
    stage_index = value["stage_index"]
    if type(stage_index) is not int or not 0 <= stage_index < len(config.stages):
        raise ValueError("checkpoint stage_index is outside the configured stages")
    stage = config.stages[stage_index]
    comparisons = (
        ("stage_name", value["stage_name"], stage.name),
        (
            "stage_start_optimizer_step",
            value["stage_start_optimizer_step"],
            stage.start_optimizer_step,
        ),
        (
            "stage_end_optimizer_step",
            value["stage_end_optimizer_step"],
            stage.end_optimizer_step,
        ),
        ("stage_config_sha256", value["stage_config_sha256"], _stage_config_sha256(stage)),
        ("trainable_parameter_names", value["trainable_parameter_names"], list(stage.trainable_parameters)),
        ("optimizer_reset", value["optimizer_reset"], stage.reset_optimizer),
        ("invariance_audit", value["invariance_audit"], stage.invariance_audit),
    )
    for name, observed, expected in comparisons:
        if type(observed) is not type(expected) or observed != expected:
            raise ValueError(f"checkpoint {name} differs from the registered stage")
    local_step = value["stage_local_optimizer_step"]
    if type(local_step) is not int or not 0 <= local_step <= stage.optimizer_steps:
        raise ValueError("checkpoint stage-local optimizer step is outside its stage")
    raw_names = value["optimizer_parameter_names"]
    if not isinstance(raw_names, list) or not raw_names or any(
        type(name) is not str or not name for name in raw_names
    ):
        raise ValueError("checkpoint optimizer parameter names are invalid")
    names = tuple(raw_names)
    if names != tuple(sorted(set(names))):
        raise ValueError("checkpoint optimizer parameter names must be sorted and unique")
    names_hash = _sha256_string(
        value["optimizer_parameter_names_sha256"],
        name="optimizer_parameter_names_sha256",
    )
    if names_hash != _names_sha256(names):
        raise ValueError("checkpoint optimizer parameter-name hash is inconsistent")
    raw_initialized_names = value["initialized_optimizer_parameter_names"]
    if not isinstance(raw_initialized_names, list) or any(
        type(name) is not str or not name for name in raw_initialized_names
    ):
        raise ValueError("checkpoint initialized optimizer parameter names are invalid")
    initialized_names = tuple(raw_initialized_names)
    if initialized_names != tuple(
        name for name in names if name in set(initialized_names)
    ):
        raise ValueError(
            "checkpoint initialized optimizer names must be an ordered subset"
        )
    initialized_names_hash = _sha256_string(
        value["initialized_optimizer_parameter_names_sha256"],
        name="initialized_optimizer_parameter_names_sha256",
    )
    if initialized_names_hash != _names_sha256(initialized_names):
        raise ValueError(
            "checkpoint initialized optimizer parameter-name hash is inconsistent"
        )
    if local_step == 0 and initialized_names:
        raise ValueError("a reset stage at local step zero cannot carry AdamW state")
    if local_step > 0 and not initialized_names:
        raise ValueError("a progressed stage must carry initialized AdamW state")
    reference = value["invariance_reference"]
    current = value["invariance_current"]
    if stage.invariance_audit == "none":
        if reference is not None or current is not None:
            raise ValueError("an unaudited stage cannot carry invariance snapshots")
        normalized_reference = None
        normalized_current = None
    else:
        normalized_reference = _snapshot(
            reference,
            name="invariance_reference",
        )
        normalized_current = _snapshot(current, name="invariance_current")
        try:
            _require_identical_invariance(
                normalized_reference,
                normalized_current,
            )
        except RuntimeError as error:
            raise ValueError("checkpoint records a failed invariance audit") from error
    entry_gate_id = value["entry_gate_id"]
    entry_gate_sha = value["entry_gate_report_sha256"]
    entry_gate_passed = value["entry_gate_passed"]
    entry_gate_step = value["entry_gate_step"]
    if type(entry_gate_id) is not str or not entry_gate_id:
        raise ValueError("checkpoint entry_gate_id must be a non-empty string")
    _sha256_string(entry_gate_sha, name="entry_gate_report_sha256")
    if type(entry_gate_passed) is not bool:
        raise ValueError("checkpoint entry_gate_passed must be a boolean")
    if type(entry_gate_step) is not int or entry_gate_step < 0:
        raise ValueError("checkpoint entry_gate_step must be a nonnegative integer")
    if stage.index == 0:
        expected_gate = ("none", "0" * 64, True, 0)
    else:
        prior_stage = config.stages[stage.index - 1]
        expected_gate = (
            prior_stage.transition_gate,
            entry_gate_sha,
            True,
            prior_stage.end_optimizer_step,
        )
        if prior_stage.transition_gate != "none" and entry_gate_sha == "0" * 64:
            raise ValueError("a gated stage checkpoint lacks its gate receipt digest")
        if prior_stage.transition_gate == "none" and entry_gate_sha != "0" * 64:
            raise ValueError("an ungated stage checkpoint has an unexpected gate digest")
    observed_gate = (entry_gate_id, entry_gate_sha, entry_gate_passed, entry_gate_step)
    if observed_gate != expected_gate:
        raise ValueError("checkpoint entry-gate provenance is inconsistent")
    completion_gate_id = value["completion_gate_id"]
    completion_gate_sha = value["completion_gate_report_sha256"]
    completion_gate_passed = value["completion_gate_passed"]
    completion_gate_step = value["completion_gate_step"]
    if type(completion_gate_id) is not str or completion_gate_id != stage.completion_gate:
        raise ValueError("checkpoint completion-gate ID differs from the stage")
    _sha256_string(
        completion_gate_sha,
        name="completion_gate_report_sha256",
    )
    if type(completion_gate_step) is not int or completion_gate_step < 0:
        raise ValueError("checkpoint completion_gate_step must be nonnegative")
    if stage.completion_gate == "none" or local_step < stage.optimizer_steps:
        expected_completion = (stage.completion_gate, "0" * 64, None, 0)
    else:
        if type(completion_gate_passed) is not bool:
            raise ValueError("terminal completion gate status must be a boolean")
        if completion_gate_sha == "0" * 64:
            raise ValueError("terminal checkpoint lacks its completion-gate digest")
        expected_completion = (
            stage.completion_gate,
            completion_gate_sha,
            completion_gate_passed,
            stage.end_optimizer_step,
        )
    observed_completion = (
        completion_gate_id,
        completion_gate_sha,
        completion_gate_passed,
        completion_gate_step,
    )
    if observed_completion != expected_completion:
        raise ValueError("checkpoint completion-gate provenance is inconsistent")
    return {
        **dict(value),
        "optimizer_parameter_names": list(names),
        "initialized_optimizer_parameter_names": list(initialized_names),
        "invariance_reference": normalized_reference,
        "invariance_current": normalized_current,
    }


def _update_tensor_hash(digest: object, name: str, tensor: object) -> None:
    if not isinstance(digest, type(sha256())) or not isinstance(tensor, Tensor):
        raise TypeError("tensor hashing requires a SHA-256 digest and Tensor")
    contiguous = tensor.detach().cpu().contiguous()
    name_bytes = name.encode("utf-8")
    dtype_bytes = str(contiguous.dtype).encode("ascii")
    digest.update(len(name_bytes).to_bytes(4, "big"))
    digest.update(name_bytes)
    digest.update(len(dtype_bytes).to_bytes(2, "big"))
    digest.update(dtype_bytes)
    digest.update(len(contiguous.shape).to_bytes(2, "big"))
    for dimension in contiguous.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    raw = contiguous.view(torch.uint8).numpy().tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def _nonvalue_state_sha256(objective: nn.Module) -> str:
    model = getattr(objective, "model", None)
    if not isinstance(model, nn.Module):
        raise TypeError("non-value state hashing requires objective.model")
    state = objective.state_dict()
    excluded = {
        "model.value_per_thought.weight",
        "model.value_per_thought.bias",
    }
    if not excluded <= set(state):
        raise ValueError("model lacks the registered value_per_thought weight and bias")
    digest = sha256(b"IRENENONVALUE\x01")
    for name in sorted(set(state) - excluded):
        _update_tensor_hash(digest, name, state[name])
    return digest.hexdigest()


def _require_identical_invariance(
    reference: Mapping[str, object],
    current: Mapping[str, object],
) -> None:
    expected = _snapshot(reference, name="invariance_reference")
    observed = _snapshot(current, name="invariance_current")
    for name in (
        "batch_count",
        "sequence_count",
        "timestep_count",
        "non_value_state_sha256",
        "action_outputs_sha256",
        "recurrent_states_sha256",
    ):
        if observed[name] != expected[name]:
            raise RuntimeError(f"frozen-stage invariance failed for {name}")


def _loss_output(value: object) -> LossOutput:
    if not isinstance(value, LossOutput):
        raise TypeError("objective must return LossOutput")
    return value


def _sample_count(batch: object) -> int:
    value = getattr(batch, "sample_count", None)
    if type(value) is not int or value < 1:
        raise ValueError("each microbatch must expose a positive sample_count")
    return value


__all__ = ["TorchTrainingSystem"]
