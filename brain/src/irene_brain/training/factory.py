"""Explicit model factories referenced by immutable training configurations."""

from __future__ import annotations

from dataclasses import replace
from typing import TypeVar

from torch import nn

from ..model.baselines import (
    DenseCommunicationSlotBaseline,
    FixedMultiHorizonSlotBaseline,
    MatchedEnsembleBaseline,
    MonolithicRecurrentBaseline,
    NoCommunicationSlotBaseline,
    ParameterMatchedMonolithicBaseline,
    ReactiveSlotBaseline,
    RecurrentTransformerBaseline,
    ResetStateSlotBaseline,
    SerialDepthSlotBaseline,
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

# Selected by exact allocated-parameter enumeration over member counts and
# widths: four untied members at width 352 land 0.72% under the reference's
# trainable budget, the nearest allocation inside the 1% tolerance.
MATCHED_ENSEMBLE_WIDTH = 352

# Selected by exact allocated-parameter enumeration: the carry-token
# Transformer control at width 568 lands 0.22% under the reference's
# trainable budget, inside the 1% tolerance.
RECURRENT_TRANSFORMER_WIDTH = 568


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
    model_config = ThoughtFieldConfig.smoke()
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


def build_thesis_reset_slots_model(
    config: TrainingConfig,
) -> ResetStateSlotBaseline:
    _require_config(config)
    model = ResetStateSlotBaseline(
        ThoughtFieldConfig.thesis_mvp(),
        input_resolution=(64, 64),
        plan_steps=3,
    )
    return _identity_pixel_grid(model)


def build_thesis_fixed_multi_horizon_model(
    config: TrainingConfig,
) -> FixedMultiHorizonSlotBaseline:
    _require_config(config)
    model = FixedMultiHorizonSlotBaseline(
        ThoughtFieldConfig.thesis_mvp(),
        input_resolution=(64, 64),
        plan_steps=3,
    )
    return _identity_pixel_grid(model)


def build_thesis_dense_routing_model(
    config: TrainingConfig,
) -> DenseCommunicationSlotBaseline:
    _require_config(config)
    model = DenseCommunicationSlotBaseline(
        ThoughtFieldConfig.thesis_mvp(),
        input_resolution=(64, 64),
        plan_steps=3,
    )
    return _identity_pixel_grid(model)


def build_thesis_reactive_model(
    config: TrainingConfig,
) -> ReactiveSlotBaseline:
    _require_config(config)
    model = ReactiveSlotBaseline(
        ThoughtFieldConfig.thesis_mvp(),
        input_resolution=(64, 64),
        plan_steps=3,
    )
    return _identity_pixel_grid(model)


def build_thesis_serial_depth_model(
    config: TrainingConfig,
) -> SerialDepthSlotBaseline:
    _require_config(config)
    reference = ThoughtFieldConfig.thesis_mvp()
    # Twelve untied serial blocks replace four tied blocks over three cycles:
    # the same block-application count with no weight tying across depth.
    model_config = replace(
        reference,
        cognitive_cycles=1,
        brain_cell_blocks=reference.cognitive_cycles * reference.brain_cell_blocks,
    )
    model = SerialDepthSlotBaseline(
        model_config,
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


def build_thesis_matched_ensemble_model(
    config: TrainingConfig,
) -> MatchedEnsembleBaseline:
    _require_config(config)
    model_config = replace(
        ThoughtFieldConfig.thesis_mvp(),
        core_width=MATCHED_ENSEMBLE_WIDTH,
    )
    model = MatchedEnsembleBaseline(
        model_config,
        input_resolution=(64, 64),
        plan_steps=3,
    )
    return _identity_pixel_grid(model)


def build_thesis_recurrent_transformer_model(
    config: TrainingConfig,
) -> RecurrentTransformerBaseline:
    _require_config(config)
    model = RecurrentTransformerBaseline(
        _monolithic_config(width=RECURRENT_TRANSFORMER_WIDTH),
        input_resolution=(64, 64),
        plan_steps=3,
    )
    return _identity_pixel_grid(model)


__all__ = [
    "MATCHED_ENSEMBLE_WIDTH",
    "PARAMETER_MATCHED_MONOLITH_WIDTH",
    "PARAMETER_MATCHED_MONOLITH_HEADS",
    "RECURRENT_TRANSFORMER_WIDTH",
    "build_smoke_model",
    "build_thesis_model",
    "build_thesis_dense_routing_model",
    "build_thesis_fixed_multi_horizon_model",
    "build_thesis_matched_ensemble_model",
    "build_thesis_monolithic_model",
    "build_thesis_no_communication_model",
    "build_thesis_parameter_matched_monolithic_model",
    "build_thesis_reactive_model",
    "build_thesis_recurrent_transformer_model",
    "build_thesis_reset_slots_model",
    "build_thesis_serial_depth_model",
]
