"""Canonical lifetime records, deterministic replay, and branch rollouts."""

from .branching import evaluate_branches
from .curriculum_dataset import (
    CURRICULUM_SCENARIOS,
    CurriculumDataset,
    CurriculumDatasetConfig,
    CurriculumScenario,
    CurriculumSequence,
    CurriculumSequenceDataset,
    CurriculumTransition,
    curriculum_dataset_manifest_sha256,
)
from .maze_chase_dataset import (
    MazeChaseCounterfactualSequence,
    MazeChaseCounterfactualTarget,
    MazeChaseCounterfactualTransition,
    MazeChaseDatasetConfig,
    MazeChaseSequence,
    MazeChaseSequenceDataset,
    MazeChaseTransition,
    maze_chase_dataset_manifest_sha256,
)
from .moving_shapes_dataset import (
    DatasetSplit,
    MovingShapesDatasetConfig,
    MovingShapesSequence,
    MovingShapesSequenceDataset,
    MovingShapesTransition,
    dataset_manifest_sha256,
    split_episode_seed,
    split_for_episode_seed,
)
from .records import BranchRecord, LifetimeRecord, SensorStorageLocator, StepRecord
from .replay import ReplayMismatch, ReplayStep, ReplayTrace, record_trace, verify_trace
from .solver_dataset import (
    SOLVER_WORLD_NAMES,
    SolverDatasetConfig,
    SolverSequence,
    SolverSequenceDataset,
    SolverTransition,
    solver_dataset_manifest_sha256,
    solver_environment_factory,
)
from .splits import (
    SplitIntegrityError,
    SplitLeakage,
    audit_split_integrity,
    require_split_integrity,
)
# Lazy export of streaming loader to keep Phase 0 modules lightweight and dependency-free
def __getattr__(name: str):
    if name in {
        "MultiTaskMixtureStream",
        "PackedSequenceBlock",
        "SequencePacker",
        "StreamingBatch",
        "StreamingConfig",
        "StreamingMultiTaskDataset",
        "TaskType",
        "TaskWeights",
        "collate_streaming_batch",
        "create_streaming_dataloader",
    }:
        from . import streaming_loader
        return getattr(streaming_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



__all__ = [
    "BranchRecord",
    "CURRICULUM_SCENARIOS",
    "CurriculumDataset",
    "CurriculumDatasetConfig",
    "CurriculumScenario",
    "CurriculumSequence",
    "CurriculumSequenceDataset",
    "CurriculumTransition",
    "DatasetSplit",
    "LifetimeRecord",
    "MazeChaseDatasetConfig",
    "MazeChaseCounterfactualSequence",
    "MazeChaseCounterfactualTarget",
    "MazeChaseCounterfactualTransition",
    "MazeChaseSequence",
    "MazeChaseSequenceDataset",
    "MazeChaseTransition",
    "MovingShapesDatasetConfig",
    "MovingShapesSequence",
    "MovingShapesSequenceDataset",
    "MovingShapesTransition",
    "ReplayMismatch",
    "ReplayStep",
    "ReplayTrace",
    "SOLVER_WORLD_NAMES",
    "SensorStorageLocator",
    "SolverDatasetConfig",
    "SolverSequence",
    "SolverSequenceDataset",
    "SolverTransition",
    "SplitIntegrityError",
    "SplitLeakage",
    "StepRecord",
    "audit_split_integrity",
    "curriculum_dataset_manifest_sha256",
    "dataset_manifest_sha256",
    "evaluate_branches",
    "maze_chase_dataset_manifest_sha256",
    "record_trace",
    "require_split_integrity",
    "solver_dataset_manifest_sha256",
    "solver_environment_factory",
    "split_episode_seed",
    "split_for_episode_seed",
    "MultiTaskMixtureStream",
    "PackedSequenceBlock",
    "SequencePacker",
    "StreamingBatch",
    "StreamingConfig",
    "StreamingMultiTaskDataset",
    "TaskType",
    "TaskWeights",
    "collate_streaming_batch",
    "create_streaming_dataloader",
    "verify_trace",
]
