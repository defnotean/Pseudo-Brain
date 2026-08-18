"""Canonical lifetime records, deterministic replay, and branch rollouts."""

from .branching import evaluate_branches
from .maze_chase_dataset import (
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
)
from .splits import (
    SplitIntegrityError,
    SplitLeakage,
    audit_split_integrity,
    require_split_integrity,
)

__all__ = [
    "BranchRecord",
    "DatasetSplit",
    "LifetimeRecord",
    "MazeChaseDatasetConfig",
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
    "dataset_manifest_sha256",
    "evaluate_branches",
    "maze_chase_dataset_manifest_sha256",
    "record_trace",
    "require_split_integrity",
    "solver_dataset_manifest_sha256",
    "split_episode_seed",
    "split_for_episode_seed",
    "verify_trace",
]
