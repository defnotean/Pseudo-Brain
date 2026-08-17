"""Explicit model factories referenced by immutable training configurations."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

from torch import nn

from ..model.baselines import (
    MonolithicRecurrentBaseline,
    NoCommunicationSlotBaseline,
    ParameterMatchedMonolithicBaseline,
)
from ..model.spec import ThoughtFieldConfig
from ..model.torch_model import IreneBrainModel
from .config import TrainingConfig


ModelT = TypeVar("ModelT", bound=IreneBrainModel)


# Selected by exact allocated-parameter enumeration in the architecture
# manifest. Six is the largest valid head count below the reference's eight;
# MultiheadAttention's projection parameter count is head-invariant.
PARAMETER_MATCHED_MONOLITH_WIDTH = 396
PARAMETER_MATCHED_MONOLITH_HEADS = 6


def _require_config(config: TrainingConfig) -> None:
    if not isinstance(config, TrainingConfig):
        raise ValueError("config must be a TrainingConfig")


def _identity_pixel_grid(model: ModelT) -> ModelT:
    # 64x64 maps exactly to the thesis 8x8 sensor grid.
    model.pixel_encoder.spatial_pool = nn.Identity()
    return model


def _monolithic_config(*, width: int, attention_heads: int = 8) -> ThoughtFieldConfig:
    reference = ThoughtFieldConfig.thesis_mvp()
    return replace(
        reference,
        core_width=width,
        attention_heads=attention_heads,
        thoughtlets=1,
        registers_per_thoughtlet=1,
        routed_neighbors=0,
        # Keep the candidate's aggregate 32 * 2 retrieval-token bandwidth.
        retrieved_entries_per_thoughtlet=(
            reference.thoughtlets * reference.retrieved_entries_per_thoughtlet
        ),
    )


def build_smoke_model(config: TrainingConfig) -> IreneBrainModel:
    _require_config(config)
    model = IreneBrainModel(
        ThoughtFieldConfig.smoke(),
        input_resolution=(32, 32),
        plan_steps=2,
    )
    # Three stride-2 convolutions map 32x32 exactly to the configured 4x4
    # sensor grid. Avoid CUDA AdaptiveAvgPool2d backward, which strict
    # deterministic PyTorch deliberately rejects even when it is an identity.
    model.pixel_encoder.spatial_pool = nn.Identity()
    return model


def build_thesis_model(config: TrainingConfig) -> IreneBrainModel:
    _require_config(config)
    model_config = ThoughtFieldConfig.thesis_mvp()
    if config.objective.continuous_output_squash == "deadzone_tanh":
        model_config = replace(
            model_config,
            actuator=replace(
                model_config.actuator,
                continuous_squash="deadzone_tanh",
            ),
        )
    model = IreneBrainModel(
        model_config,
        input_resolution=(64, 64),
        plan_steps=3,
    )
    return _identity_pixel_grid(model)


def build_thesis_no_communication_model(
    config: TrainingConfig,
) -> NoCommunicationSlotBaseline:
    _require_config(config)
    model = NoCommunicationSlotBaseline(
        ThoughtFieldConfig.thesis_mvp(),
        input_resolution=(64, 64),
        plan_steps=3,
    )
    return _identity_pixel_grid(model)


def build_thesis_monolithic_model(
    config: TrainingConfig,
) -> MonolithicRecurrentBaseline:
    _require_config(config)
    model = MonolithicRecurrentBaseline(
        _monolithic_config(width=ThoughtFieldConfig.thesis_mvp().core_width),
        input_resolution=(64, 64),
        plan_steps=3,
    )
    return _identity_pixel_grid(model)


def build_thesis_parameter_matched_monolithic_model(
    config: TrainingConfig,
) -> ParameterMatchedMonolithicBaseline:
    _require_config(config)
    model = ParameterMatchedMonolithicBaseline(
        _monolithic_config(
            width=PARAMETER_MATCHED_MONOLITH_WIDTH,
            attention_heads=PARAMETER_MATCHED_MONOLITH_HEADS,
        ),
        input_resolution=(64, 64),
        plan_steps=3,
    )
    return _identity_pixel_grid(model)


__all__ = [
    "PARAMETER_MATCHED_MONOLITH_WIDTH",
    "PARAMETER_MATCHED_MONOLITH_HEADS",
    "build_smoke_model",
    "build_thesis_model",
    "build_thesis_monolithic_model",
    "build_thesis_no_communication_model",
    "build_thesis_parameter_matched_monolithic_model",
]
