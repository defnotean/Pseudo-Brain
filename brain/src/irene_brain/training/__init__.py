"""Offline training contracts with accelerator imports kept opt-in."""

from .batches import (
    BUTTON_TARGET_INDICES,
    CONTINUOUS_TARGET_INDICES,
    CONTROL_LAYOUT_ID,
    CONTROL_VECTOR_SIZE,
    MovingShapesBatchSource,
    TrajectoryBatch,
    control_to_vector,
)
from .checkpoint import (
    LoadedCheckpoint,
    TrainerCursor,
    file_sha256,
    load_checkpoint,
    save_checkpoint,
    source_tree_sha256,
)
from .config import (
    DatasetConfig,
    DeterminismConfig,
    LoggingConfig,
    OptimizationConfig,
    PrecisionConfig,
    ResourceConfig,
    RunConfig,
    TrainingConfig,
    TrainingStageConfig,
    load_training_config,
)
from .metrics import JsonlMetricWriter, MetricRecord
from .protocol import (
    DeterministicBatchSource,
    StagedTrainingSystem,
    TrainingStepResult,
    TrainingSystem,
)
from .trainer import Trainer, TrainingSummary

__all__ = [
    "BUTTON_TARGET_INDICES",
    "CONTINUOUS_TARGET_INDICES",
    "CONTROL_LAYOUT_ID",
    "CONTROL_VECTOR_SIZE",
    "DatasetConfig",
    "DeterminismConfig",
    "DeterministicBatchSource",
    "JsonlMetricWriter",
    "LoadedCheckpoint",
    "LoggingConfig",
    "MetricRecord",
    "MovingShapesBatchSource",
    "OptimizationConfig",
    "PrecisionConfig",
    "ResourceConfig",
    "RunConfig",
    "StagedTrainingSystem",
    "Trainer",
    "TrainerCursor",
    "TrainingConfig",
    "TrainingStageConfig",
    "TrainingStepResult",
    "TrainingSummary",
    "TrainingSystem",
    "TrajectoryBatch",
    "control_to_vector",
    "file_sha256",
    "load_checkpoint",
    "load_training_config",
    "save_checkpoint",
    "source_tree_sha256",
]
