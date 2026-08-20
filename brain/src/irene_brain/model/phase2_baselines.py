"""Phase 2 Canonical Baselines Suite.

Provides unified construction and matching factories for the 8 mandatory baselines:
1. Baseline 1 — Reactive Controller (no recurrent memory)
2. Baseline 2 — GRU Recurrent Controller (CNN-GRU)
3. Baseline 3 — Recurrent Transformer (carry-token transformer)
4. Baseline 4 — State-Space Recurrent Controller (SSM/S4 style)
5. Baseline 5 — Wider Monolithic Recurrent State (capacity-matched GRU)
6. Baseline 6 — Deeper Serial Model (matched sequential FLOPs)
7. Baseline 7 — Fixed Multi-Horizon Predictor (static horizon partition)
8. Baseline 8 — Conventional Latent World-Model Actor (recurrent world model + rollout)
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum, unique
from typing import Any, Callable

import torch
from torch import nn

from .baselines import (
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
from .spec import ThoughtFieldConfig
from .state_space import StateSpaceRecurrentBaseline
from .torch_model import IreneBrainModel
from .world_model_actor import LatentWorldModelActor


@unique
class BaselineVariant(str, Enum):
    PSEUDO_BRAIN = "pseudo_brain_reference"
    B1_REACTIVE = "baseline_1_reactive"
    B2_GRU = "baseline_2_gru_recurrent"
    B3_RECURRENT_TRANSFORMER = "baseline_3_recurrent_transformer"
    B4_STATE_SPACE_SSM = "baseline_4_state_space_ssm"
    B5_WIDER_MONOLITH = "baseline_5_wider_monolith"
    B6_SERIAL_DEPTH = "baseline_6_serial_depth"
    B7_FIXED_MULTI_HORIZON = "baseline_7_fixed_multi_horizon"
    B8_WORLD_MODEL_ACTOR = "baseline_8_world_model_actor"


def build_phase2_model(
    variant: BaselineVariant,
    base_config: ThoughtFieldConfig | None = None,
    input_resolution: tuple[int, int] = (32, 32),
) -> IreneBrainModel:
    """Instantiate a model for the given Phase 2 variant under matched configuration."""
    if base_config is None:
        base_config = ThoughtFieldConfig(
            thoughtlets=32,
            cognitive_cycles=3,
            core_width=32,
            attention_heads=4,
            sensor_tokens=4,
            belief_tokens=4,
            working_memory_tokens=4,
            goal_context_tokens=8,
            registers_per_thoughtlet=2,
            brain_cell_blocks=1,
            routed_neighbors=2,
        )

    if variant == BaselineVariant.PSEUDO_BRAIN:
        return IreneBrainModel(config=base_config, input_resolution=input_resolution)

    elif variant == BaselineVariant.B1_REACTIVE:
        return ReactiveSlotBaseline(config=base_config, input_resolution=input_resolution)

    elif variant == BaselineVariant.B2_GRU:
        cfg = replace(base_config, thoughtlets=1, registers_per_thoughtlet=1, routed_neighbors=0)
        return MonolithicRecurrentBaseline(config=cfg, input_resolution=input_resolution)

    elif variant == BaselineVariant.B3_RECURRENT_TRANSFORMER:
        cfg = replace(base_config, thoughtlets=1, registers_per_thoughtlet=1, routed_neighbors=0)
        return RecurrentTransformerBaseline(config=cfg, input_resolution=input_resolution)

    elif variant == BaselineVariant.B4_STATE_SPACE_SSM:
        cfg = replace(base_config, thoughtlets=1, registers_per_thoughtlet=1, routed_neighbors=0)
        return StateSpaceRecurrentBaseline(config=cfg, input_resolution=input_resolution)

    elif variant == BaselineVariant.B5_WIDER_MONOLITH:
        # Widened single recurrent state to match total persistent state capacity
        widened_width = int(base_config.core_width * 2.0)
        cfg = replace(base_config, thoughtlets=1, registers_per_thoughtlet=1, routed_neighbors=0, core_width=widened_width)
        return ParameterMatchedMonolithicBaseline(config=cfg, input_resolution=input_resolution)

    elif variant == BaselineVariant.B6_SERIAL_DEPTH:
        return SerialDepthSlotBaseline(config=base_config, input_resolution=input_resolution)

    elif variant == BaselineVariant.B7_FIXED_MULTI_HORIZON:
        return FixedMultiHorizonSlotBaseline(config=base_config, input_resolution=input_resolution)

    elif variant == BaselineVariant.B8_WORLD_MODEL_ACTOR:
        cfg = replace(base_config, thoughtlets=1, registers_per_thoughtlet=1, routed_neighbors=0)
        return LatentWorldModelActor(config=cfg, input_resolution=input_resolution)

    else:
        raise ValueError(f"Unknown Phase 2 baseline variant: {variant}")
