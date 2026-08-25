"""Development-only V2.1i all-action training and calibration runner.

This runner deliberately cannot open TEST or CPU-QUAL.  It jointly trains one
current Core V2 model on the fixed TRAIN-FIT episode range, fits the immutable
per-action affine hazard calibrator on the disjoint TRAIN-CAL range, bakes that
transform into the live outcome model, and evaluates the frozen candidate on
DEV.  DEV is diagnostic evidence only and cannot publish a qualified model.
"""
from __future__ import annotations

import os

# Set these before importing torch.  This diagnostic must not compete with a
# foreground game or accidentally initialize a GPU context.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from irene_brain.data import (
    DatasetSplit,
    MazeChaseCounterfactualSequence,
    MazeChaseDatasetConfig,
    MazeChaseSequenceDataset,
)
from irene_brain.evaluation.latent_noncollapse_metrics import (
    LatentNonCollapseThresholds,
    evaluate_latent_noncollapse,
)
from irene_brain.evaluation.v21_qualification_metrics import (
    all_action_probability_metrics,
    binary_probability_metrics,
    clustered_improvement_bootstrap,
    evaluate_cross_episode_derangements,
    evaluate_train_fitted_baselines,
    fit_train_baselines,
    gather_factual_rows,
)
from irene_brain.training.batches import AllActionTrajectoryBatchV1
from irene_brain.training.objective import _rgb_tensor
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Config, CoreV2Model
from irene_brain.v2.hazard_calibration import (
    PerActionAffineHazardCalibrator,
    TrainCalibrationProvenance,
    fit_train_only_per_action_affine,
)
from irene_brain.v2.trajectory_objective import (
    AllActionV2TrajectoryObjective,
    control_action_class,
    transition_hazard,
)
from run_provenance import apply_deterministic_mode


SCHEMA_VERSION = 1
MODE = "v21i_development_only"
TRAIN_FIT_OFFSET = 16_777_216
TRAIN_FIT_EPISODES = 128
TRAIN_CAL_OFFSET = 16_777_344
TRAIN_CAL_EPISODES = 64
DEV_OFFSET = 25_165_824
DEV_EPISODES = 64
SEQUENCE_LENGTH = 16
BURN_IN_STEPS = 4
ACTION_COUNT = 5
BATCH_SIZE = 8
TRAINING_EPOCHS = 3
LEARNING_RATE = 1.0e-4
WEIGHT_DECAY = 1.0e-4
CLIP_NORM = 1.0
HAZARD_REFINEMENT_PASSES = 8
HAZARD_REFINEMENT_ROOT_BATCH_SIZE = 24
HAZARD_REFINEMENT_LEARNING_RATE = 1.0e-3
HAZARD_REFINEMENT_WEIGHT_DECAY = 1.0e-4
HAZARD_REFINEMENT_CLIP_NORM = 1.0
CALIBRATION_MODE = "bias_only"
CALIBRATION_MODES = (CALIBRATION_MODE,)
CALIBRATION_SCHEMA = "irene.hazard.per_action_affine.v1"
CALIBRATION_FORMULA = (
    "calibrated_logit[action] = scale[action] * raw_logit + bias[action]"
)
PARTITION_ALGORITHM = (
    "fixed_contiguous_episode_ranges_v1:"
    "TRAIN-FIT=train[16777216,16777344);"
    "TRAIN-CAL=train[16777344,16777408);"
    "DEV=validation[25165824,25165888)"
)
DERANGEMENT_REPETITIONS = 20
DERANGEMENT_SEED = 21_100_021
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 21_100_043
ECE_BINS = 10
REQUIRED_MODEL_SEEDS = (44, 45, 46, 47, 48)
MAX_BASELINE_BCE_RATIO = 0.98
MAX_BASELINE_BRIER_RATIO = 0.95
MINIMUM_AGGREGATE_ROC_AUC = 0.65
MINIMUM_BASELINE_ROC_AUC_GAIN = 0.10
MINIMUM_PER_ACTION_ROC_AUC = 0.60
MINIMUM_PER_ACTION_PR_PREVALENCE_GAIN = 0.05
MAXIMUM_AGGREGATE_ECE = 0.05
MAXIMUM_PER_ACTION_ECE = 0.075
MAXIMUM_ABSOLUTE_PER_ACTION_BIAS = 0.05
MINIMUM_SHUFFLED_BCE_RATIO = 1.05
MINIMUM_SHUFFLED_AGGREGATE_AUC_DROP = 0.05
MINIMUM_SHUFFLED_PER_ACTION_AUC_DROP = 0.03

LATENT_THRESHOLDS = LatentNonCollapseThresholds(
    minimum_samples=128,
    minimum_nonzero_variance_fraction=0.50,
    minimum_covariance_effective_rank=4.0,
    minimum_mean_pairwise_cosine_distance=1.0e-3,
)

_SOURCE_BUNDLE_PIPELINE_FILES = (
    "brain/scripts/run_provenance.py",
    "brain/scripts/v21i_development_runner.py",
)


@dataclass(frozen=True)
class PartitionContract:
    name: str
    dataset_config: MazeChaseDatasetConfig
    burn_in_steps: int = BURN_IN_STEPS

    def __post_init__(self) -> None:
        if self.name not in {"TRAIN-FIT", "TRAIN-CAL", "DEV"}:
            raise ValueError("unsupported V2.1i development partition")
        expected_split = (
            DatasetSplit.VALIDATION if self.name == "DEV" else DatasetSplit.TRAIN
        )
        if self.dataset_config.split is not expected_split:
            raise ValueError(f"{self.name} has the wrong dataset split")
        if self.dataset_config.counterfactual_targets != "all_actions_v1":
            raise ValueError("every V2.1i partition requires all_actions_v1")
        if self.dataset_config.behavior_policy != "balanced_intervention_v1":
            raise ValueError(
                "every V2.1i partition requires balanced_intervention_v1"
            )
        if self.dataset_config.behavior_intervention_rate != 0.5:
            raise ValueError("every V2.1i partition requires intervention rate 0.5")
        if self.burn_in_steps < 0 or self.burn_in_steps >= self.dataset_config.sequence_length:
            raise ValueError("invalid development burn-in")


class PartitionSource:
    """Single-split lazy source; no unused TEST dataset is ever constructed."""

    def __init__(
        self,
        contract: PartitionContract,
        *,
        dataset_factory: Callable[[MazeChaseDatasetConfig], MazeChaseSequenceDataset]
        = MazeChaseSequenceDataset,
    ) -> None:
        self.contract = contract
        self.dataset = dataset_factory(contract.dataset_config)

    @property
    def manifest_sha256(self) -> str:
        return self.dataset.manifest_sha256

    def iter_all_action_batches(
        self,
        *,
        epoch: int,
        batch_size: int,
    ) -> Iterator[AllActionTrajectoryBatchV1]:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        split = self.contract.dataset_config.split
        indices = self.dataset.epoch_indices(
            epoch=epoch,
            shuffle=split is DatasetSplit.TRAIN,
        )
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            sequences = tuple(self.dataset[index] for index in selected)
            if any(
                not isinstance(sequence, MazeChaseCounterfactualSequence)
                for sequence in sequences
            ):
                raise RuntimeError("all-action source emitted a legacy sequence")
            yield AllActionTrajectoryBatchV1(
                split=split.value,
                burn_in_steps=self.contract.burn_in_steps,
                sequences=sequences,
            )


@dataclass(frozen=True)
class CollectedEvidence:
    raw_hazard_logits: np.ndarray
    hazard_probabilities: np.ndarray
    hazard_targets: np.ndarray
    factual_actions: np.ndarray
    episode_group_ids: tuple[str, ...]
    root_state_ids: tuple[str, ...]
    beliefs: np.ndarray
    next_latent_losses: np.ndarray
    reward_losses: np.ndarray
    sequence_group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        roots = self.hazard_targets.shape[0]
        expected_table = (roots, ACTION_COUNT)
        if self.raw_hazard_logits.shape != expected_table:
            raise ValueError("raw hazard logit table has the wrong shape")
        if self.hazard_probabilities.shape != expected_table:
            raise ValueError("hazard probability table has the wrong shape")
        if self.factual_actions.shape != (roots,):
            raise ValueError("factual action vector has the wrong shape")
        if self.beliefs.ndim != 2 or self.beliefs.shape[0] != roots:
            raise ValueError("belief evidence has the wrong shape")
        if self.next_latent_losses.shape != expected_table:
            raise ValueError("per-action next-latent losses have the wrong shape")
        if self.reward_losses.shape != expected_table:
            raise ValueError("per-action reward losses have the wrong shape")
        if len(self.episode_group_ids) != roots or len(self.root_state_ids) != roots:
            raise ValueError("root provenance does not cover every evidence row")
        if not self.sequence_group_ids:
            raise ValueError("sequence provenance may not be empty")


def development_partition_contracts() -> dict[str, PartitionContract]:
    """Return the only dataset ranges this runner is authorized to use."""

    common: dict[str, object] = {
        "sequence_length": SEQUENCE_LENGTH,
        "ghost_count": 5,
        "ghost_period": 1,
        "ghost_rule": "direct",
        "input_delay_ticks": 0,
        "sticky_direction": False,
        "episode_horizon": 0,
        "behavior_policy": "balanced_intervention_v1",
        "behavior_intervention_rate": 0.5,
        "counterfactual_targets": "all_actions_v1",
    }
    return {
        "TRAIN-FIT": PartitionContract(
            "TRAIN-FIT",
            MazeChaseDatasetConfig(
                split=DatasetSplit.TRAIN,
                sequence_count=TRAIN_FIT_EPISODES,
                seed_offset=TRAIN_FIT_OFFSET,
                **common,
            ),
        ),
        "TRAIN-CAL": PartitionContract(
            "TRAIN-CAL",
            MazeChaseDatasetConfig(
                split=DatasetSplit.TRAIN,
                sequence_count=TRAIN_CAL_EPISODES,
                seed_offset=TRAIN_CAL_OFFSET,
                **common,
            ),
        ),
        "DEV": PartitionContract(
            "DEV",
            MazeChaseDatasetConfig(
                split=DatasetSplit.VALIDATION,
                sequence_count=DEV_EPISODES,
                seed_offset=DEV_OFFSET,
                **common,
            ),
        ),
    }


def model_config() -> CoreV2Config:
    """The current V2.1 dedicated-hazard all-action model, without scaling."""

    return CoreV2Config(
        decision_aggregation="direct_mean_logits_v1",
        braincell_dynamics="normalized_mixture_v1",
        belief_dynamics="convex_gated_v1",
        hazard_parameterization="probability_sigmoid_v1",
        latent_comparison="cosine_distance_v1",
        reward_comparison="raw_mse_v0",
        outcome_action_conditioning="thought_only_v0",
        outcome_architecture="all_action_table_v1",
        hazard_outcome_path="dedicated_stopgrad_v1",
        reward_prediction="symlog_twohot_v1",
        prediction_error_fusion="latent_outcome_surprise_v1",
        next_weight=0.5,
        reward_weight=0.25,
        hazard_weight=1.0,
    )


def _source_bundle_files(project_root: Path) -> tuple[str, ...]:
    package_root = project_root / "brain" / "src" / "irene_brain"
    package_files = (
        path.relative_to(project_root).as_posix()
        for path in package_root.rglob("*.py")
        if path.is_file()
    )
    return tuple(sorted({*_SOURCE_BUNDLE_PIPELINE_FILES, *package_files}))


def _source_bundle(project_root: Path) -> dict[str, object]:
    digest = sha256(b"IRV21IDEVSOURCE\x01")
    files: dict[str, str] = {}
    for relative in _source_bundle_files(project_root):
        content = (project_root / relative).read_bytes()
        file_sha = sha256(content).hexdigest()
        files[relative] = file_sha
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(file_sha))
    return {"schema_version": 1, "sha256": digest.hexdigest(), "files": files}


def _state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    digest = sha256(b"IRV21ISTATE\x01")
    for name, value in sorted(state_dict.items()):
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


def audit_model_finite(model: CoreV2Model) -> dict[str, object]:
    """Fail before checkpoint publication if any parameter or buffer is non-finite."""

    parameter_names = tuple(name for name, _ in model.named_parameters())
    buffer_names = tuple(name for name, _ in model.named_buffers())
    for kind, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in values:
            if not bool(torch.isfinite(value.detach()).all()):
                raise FloatingPointError(f"non-finite model {kind}: {name}")
    return {
        "all_parameters_finite": True,
        "all_buffers_finite": True,
        "parameter_tensor_count": len(parameter_names),
        "buffer_tensor_count": len(buffer_names),
        "state_sha256": _state_dict_sha256(model.state_dict()),
    }


def _assert_publication_targets_available(
    output: Path,
    checkpoint: Path,
    uncalibrated_checkpoint: Path,
) -> None:
    targets = tuple(path.resolve() for path in (output, checkpoint, uncalibrated_checkpoint))
    if len(set(targets)) != len(targets):
        raise ValueError("output and checkpoint paths must all be distinct")
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to replace an existing V2.1i artifact")


def _publish_json_create_only(path: Path, payload: Mapping[str, object]) -> str:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256(encoded).hexdigest()


def _publish_checkpoint_create_only(path: Path, payload: Mapping[str, object]) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256(path.read_bytes()).hexdigest()


def _model_inputs(
    transitions: Sequence[object],
    *,
    prior_transitions: Sequence[object] | None,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None]:
    pixels = _rgb_tensor(
        tuple(transition.observation.rgb for transition in transitions),
        device=device,
        resolution=(32, 32),
    )
    applied = torch.tensor(
        [control_action_class(transition.applied_control) for transition in transitions],
        dtype=torch.long,
        device=device,
    )
    previous = torch.tensor(
        [
            control_action_class(transition.observation.previous_control)
            for transition in transitions
        ],
        dtype=torch.long,
        device=device,
    )
    if prior_transitions is None:
        return pixels, applied, previous, None, None
    prior_reward = torch.tensor(
        [[transition.reward_target] for transition in prior_transitions],
        dtype=torch.float32,
        device=device,
    )
    prior_hazard = torch.tensor(
        [[transition_hazard(transition.event_targets)] for transition in prior_transitions],
        dtype=torch.float32,
        device=device,
    )
    return pixels, applied, previous, prior_reward, prior_hazard


def _canonicalize_table(values: Tensor, action_ids: Tensor) -> np.ndarray:
    if values.dim() == 3 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.shape != action_ids.shape:
        raise ValueError("action table values and IDs have different shapes")
    canonical = torch.empty_like(values)
    canonical.scatter_(1, action_ids, values)
    return canonical.detach().to(device="cpu", dtype=torch.float64).numpy()


@torch.no_grad()
def collect_evidence(
    model: CoreV2Model,
    source: PartitionSource,
    *,
    batch_size: int = BATCH_SIZE,
) -> CollectedEvidence:
    """Collect action tables while preserving exact root and episode identity."""

    was_training = model.training
    model.eval()
    device = torch.device("cpu")
    raw_tables: list[np.ndarray] = []
    probability_tables: list[np.ndarray] = []
    target_tables: list[np.ndarray] = []
    factual_actions: list[int] = []
    episode_groups: list[str] = []
    root_ids: list[str] = []
    beliefs: list[np.ndarray] = []
    next_latent_losses: list[np.ndarray] = []
    reward_losses: list[np.ndarray] = []
    sequence_groups: list[str] = []

    for batch in source.iter_all_action_batches(epoch=0, batch_size=batch_size):
        state = model.init_state(batch.batch_size, device)
        for sequence in batch.sequences:
            sequence_groups.append(
                f"{source.contract.name}:episode:{sequence.episode_seed}"
            )
        for tick in range(batch.sequence_length):
            transitions = tuple(sequence.transitions[tick] for sequence in batch.sequences)
            prior = (
                None
                if tick == 0
                else tuple(sequence.transitions[tick - 1] for sequence in batch.sequences)
            )
            pixels, applied, previous, prior_reward, prior_hazard = _model_inputs(
                transitions,
                prior_transitions=prior,
                device=device,
            )
            output, state = model(
                pixels,
                state,
                prev_action=previous,
                actual_reward=prior_reward,
                actual_hazard=prior_hazard,
                world_model_action=applied,
                intervention_pe="normal",
            )
            if tick < batch.burn_in_steps:
                continue
            table = output.outcome_table
            if (
                table is None
                or table.raw_hazard_logits is None
                or table.predicted_hazard_logits is None
            ):
                raise RuntimeError("V2.1i requires raw and calibrated all-action logits")
            raw_tables.append(_canonicalize_table(table.raw_hazard_logits, table.action_ids))
            probability_tables.append(
                _canonicalize_table(table.predicted_hazard, table.action_ids)
            )
            canonical_targets = np.asarray(
                [
                    [
                        transition_hazard(target.event_targets)
                        for target in transition.counterfactual_targets
                    ]
                    for transition in transitions
                ],
                dtype=np.float64,
            )
            if canonical_targets.shape != (batch.batch_size, ACTION_COUNT):
                raise RuntimeError("counterfactual target width drifted")
            target_tables.append(canonical_targets)
            canonical_reward = torch.tensor(
                [
                    [target.reward_target for target in transition.counterfactual_targets]
                    for transition in transitions
                ],
                dtype=torch.float32,
                device=device,
            )
            branch_pixels = _rgb_tensor(
                tuple(
                    target.next_observation_target.rgb
                    for transition in transitions
                    for target in transition.counterfactual_targets
                ),
                device=device,
                resolution=(32, 32),
            )
            if model.world_model is None or model.world_model.outcome_model is None:
                raise RuntimeError("V2.1i evidence requires the world outcome model")
            canonical_next = model.world_model.compute_target(branch_pixels).reshape(
                batch.batch_size,
                ACTION_COUNT,
                -1,
            )
            aligned_next = torch.gather(
                canonical_next,
                1,
                table.action_ids.unsqueeze(-1).expand(-1, -1, canonical_next.shape[-1]),
            )
            table_next_loss = 1.0 - torch.nn.functional.cosine_similarity(
                table.predicted_next_latent,
                aligned_next,
                dim=-1,
            )
            next_latent_losses.append(
                _canonicalize_table(table_next_loss, table.action_ids)
            )
            reward_logits = table.predicted_reward_logits
            reward_codec = model.world_model.outcome_model.reward_distribution
            if reward_logits is None or reward_codec is None:
                raise RuntimeError("V2.1i evidence requires distributional reward logits")
            aligned_reward = torch.gather(canonical_reward, 1, table.action_ids)
            table_reward_loss = reward_codec.loss(
                reward_logits,
                aligned_reward,
                reduction="none",
            )
            reward_losses.append(_canonicalize_table(table_reward_loss, table.action_ids))
            factual_actions.extend(int(value) for value in applied.to("cpu").tolist())
            beliefs.append(output.belief.detach().to("cpu", dtype=torch.float64).numpy())
            for sequence, transition in zip(batch.sequences, transitions, strict=True):
                episode_groups.append(
                    f"{source.contract.name}:episode:{sequence.episode_seed}"
                )
                root_ids.append(transition.root_state_sha256)

    if was_training:
        model.train()
    evidence = CollectedEvidence(
        raw_hazard_logits=np.concatenate(raw_tables, axis=0),
        hazard_probabilities=np.concatenate(probability_tables, axis=0),
        hazard_targets=np.concatenate(target_tables, axis=0),
        factual_actions=np.asarray(factual_actions, dtype=np.int64),
        episode_group_ids=tuple(episode_groups),
        root_state_ids=tuple(root_ids),
        beliefs=np.concatenate(beliefs, axis=0),
        next_latent_losses=np.concatenate(next_latent_losses, axis=0),
        reward_losses=np.concatenate(reward_losses, axis=0),
        sequence_group_ids=tuple(sequence_groups),
    )
    expected_roots = (
        source.contract.dataset_config.sequence_count
        * (source.contract.dataset_config.sequence_length - source.contract.burn_in_steps)
    )
    if evidence.hazard_targets.shape[0] != expected_roots:
        raise RuntimeError("development evidence root count drifted")
    if len(set(evidence.root_state_ids)) != len(evidence.root_state_ids):
        raise RuntimeError("development evidence contains duplicate factual roots")
    return evidence


def train_joint_model(
    model: CoreV2Model,
    source: PartitionSource,
    *,
    epochs: int,
) -> dict[str, object]:
    """Joint train with deterministic episode permutations and no replacement."""

    objective = AllActionV2TrajectoryObjective(
        model,
        flags=CONFIG_B_PREDICTIVE,
        prediction_error_intervention="normal",
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    telemetry: list[dict[str, object]] = []
    optimizer_steps = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        loss_sum = 0.0
        sample_count = 0
        component_sums: dict[str, float] = {}
        maximum_gradient_norm = 0.0
        for batch in source.iter_all_action_batches(epoch=epoch, batch_size=BATCH_SIZE):
            result = objective(batch)
            loss_value = float(result.loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"non-finite joint loss in epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            gradient_norm = float(nn.utils.clip_grad_norm_(trainable, CLIP_NORM))
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient in epoch {epoch + 1}")
            optimizer.step()
            objective.update_target_encoder()
            optimizer_steps += 1
            loss_sum += loss_value * result.samples
            sample_count += result.samples
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
            for name, value in result.components.items():
                component_sums[name] = component_sums.get(name, 0.0) + (
                    float(value.detach()) * result.samples
                )
        telemetry.append(
            {
                "epoch": epoch + 1,
                "samples": sample_count,
                "loss": loss_sum / sample_count,
                "components": {
                    name: value / sample_count
                    for name, value in sorted(component_sums.items())
                },
                "maximum_preclip_gradient_norm": maximum_gradient_norm,
            }
        )
        print(
            f"[V2.1i seed train] epoch={epoch + 1} "
            f"loss={loss_sum / sample_count:.6f}",
            flush=True,
        )
    return {
        "source_partition": source.contract.name,
        "optimizer": "AdamW",
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "batch_size": BATCH_SIZE,
        "sequence_count": source.contract.dataset_config.sequence_count,
        "sequence_length": source.contract.dataset_config.sequence_length,
        "burn_in_steps": source.contract.burn_in_steps,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "clip_norm": CLIP_NORM,
        "prediction_error_intervention": "normal",
        "target_encoder_update": "after_every_optimizer_step",
        "sampling": "deterministic_episode_permutation_without_replacement",
        "wall_seconds": time.perf_counter() - started,
        "telemetry": telemetry,
    }


def refine_hazard_path(
    model: CoreV2Model,
    training: CollectedEvidence,
    *,
    seed: int,
) -> dict[str, object]:
    """Fit only the nonlinear hazard path on exhaustive whole TRAIN-FIT roots."""

    if model.world_model is None or model.world_model.outcome_model is None:
        raise RuntimeError("hazard refinement requires the all-action outcome model")
    outcome_model = model.world_model.outcome_model
    prefixes = (
        "hazard_state_trunk.",
        "hazard_action_embedding.",
        "hazard_outcome_trunk.",
        "hazard_head.",
    )
    model.requires_grad_(False)
    selected: list[nn.Parameter] = []
    selected_names: list[str] = []
    for name, parameter in outcome_model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True)
            selected.append(parameter)
            selected_names.append(f"world_model.outcome_model.{name}")
    if not selected:
        raise RuntimeError("hazard refinement selected no parameters")

    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        selected,
        lr=HAZARD_REFINEMENT_LEARNING_RATE,
        weight_decay=HAZARD_REFINEMENT_WEIGHT_DECAY,
    )
    contexts = torch.from_numpy(training.beliefs).to(dtype=torch.float32)
    targets = torch.from_numpy(training.hazard_targets).to(dtype=torch.float32)
    root_count = contexts.shape[0]
    expected_steps_per_pass = math.ceil(
        root_count / HAZARD_REFINEMENT_ROOT_BATCH_SIZE
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pass_records: list[dict[str, object]] = []
    optimizer_steps = 0
    outcome_model.train()
    for pass_index in range(HAZARD_REFINEMENT_PASSES):
        permutation = torch.randperm(root_count, generator=generator)
        if int(permutation.unique().numel()) != root_count:
            raise RuntimeError("hazard refinement root permutation contains replacement")
        loss_sum = 0.0
        sample_count = 0
        maximum_gradient_norm = 0.0
        for start in range(0, root_count, HAZARD_REFINEMENT_ROOT_BATCH_SIZE):
            indices = permutation[start : start + HAZARD_REFINEMENT_ROOT_BATCH_SIZE]
            table = outcome_model(contexts[indices])
            if table.raw_hazard_logits is None:
                raise RuntimeError("hazard refinement requires raw hazard logits")
            logits = table.raw_hazard_logits.squeeze(-1)
            aligned_targets = torch.gather(targets[indices], 1, table.action_ids)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                aligned_targets,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite exhaustive hazard-refinement loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = float(
                nn.utils.clip_grad_norm_(selected, HAZARD_REFINEMENT_CLIP_NORM)
            )
            if not math.isfinite(gradient_norm):
                raise FloatingPointError("non-finite hazard-refinement gradient")
            optimizer.step()
            optimizer_steps += 1
            roots = len(indices)
            loss_sum += float(loss.detach()) * roots
            sample_count += roots
            maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        pass_records.append(
            {
                "pass": pass_index + 1,
                "roots": sample_count,
                "optimizer_steps": expected_steps_per_pass,
                "mean_unweighted_all_action_bce": loss_sum / sample_count,
                "maximum_preclip_gradient_norm": maximum_gradient_norm,
                "root_permutation_sha256": sha256(
                    permutation.to(dtype=torch.int64).numpy().astype(
                        "<i8",
                        copy=False,
                    ).tobytes(order="C")
                ).hexdigest(),
            }
        )
    model.requires_grad_(False)
    after = model.state_dict()
    changed = tuple(
        name for name in sorted(before) if not torch.equal(before[name], after[name])
    )
    if not changed or not set(changed).issubset(set(selected_names)):
        raise RuntimeError("hazard refinement changed a non-hazard-path tensor")
    expected_steps = expected_steps_per_pass * HAZARD_REFINEMENT_PASSES
    if optimizer_steps != expected_steps:
        raise RuntimeError("hazard refinement optimizer-step count drifted")
    return {
        "enabled": True,
        "source_partition": "TRAIN-FIT",
        "dev_used_for_stopping": False,
        "belief_context": "frozen_joint_trained_recurrent_belief",
        "passes": HAZARD_REFINEMENT_PASSES,
        "root_batch_size": HAZARD_REFINEMENT_ROOT_BATCH_SIZE,
        "root_count": root_count,
        "action_branches_per_root": ACTION_COUNT,
        "optimizer_steps_per_pass": expected_steps_per_pass,
        "optimizer_steps": optimizer_steps,
        "sampling": "deterministic_complete_root_permutation_without_replacement",
        "learning_rate": HAZARD_REFINEMENT_LEARNING_RATE,
        "weight_decay": HAZARD_REFINEMENT_WEIGHT_DECAY,
        "clip_norm": HAZARD_REFINEMENT_CLIP_NORM,
        "loss": "unweighted_binary_cross_entropy_with_logits_all_five_actions",
        "trainable_parameter_count": sum(parameter.numel() for parameter in selected),
        "trainable_parameter_names": sorted(selected_names),
        "changed_tensor_names": list(changed),
        "all_other_tensors_byte_equal": True,
        "state_sha256_before": _state_dict_sha256(before),
        "state_sha256_after": _state_dict_sha256(after),
        "pass_records": pass_records,
    }


@torch.no_grad()
def evaluate_joint_objective(
    model: CoreV2Model,
    source: PartitionSource,
) -> dict[str, object]:
    objective = AllActionV2TrajectoryObjective(
        model,
        flags=CONFIG_B_PREDICTIVE,
        prediction_error_intervention="normal",
    )
    objective.eval()
    loss_sum = 0.0
    sample_count = 0
    component_sums: dict[str, float] = {}
    metric_sums: dict[str, float] = {}
    for batch in source.iter_all_action_batches(epoch=0, batch_size=BATCH_SIZE):
        result = objective(batch)
        sample_count += result.samples
        loss_sum += float(result.loss) * result.samples
        for name, value in result.components.items():
            component_sums[name] = component_sums.get(name, 0.0) + (
                float(value) * result.samples
            )
        for name in ("action_accuracy", "reward_mae", "hazard_brier", "next_latent_cosine"):
            metric_sums[name] = metric_sums.get(name, 0.0) + (
                float(result.metrics[name]) * result.samples
            )
    return {
        "samples": sample_count,
        "loss": loss_sum / sample_count,
        "components": {
            name: value / sample_count for name, value in sorted(component_sums.items())
        },
        "metrics": {
            name: value / sample_count for name, value in sorted(metric_sums.items())
        },
    }


def fit_calibrator(
    calibration: CollectedEvidence,
    training: CollectedEvidence,
    *,
    upstream_checkpoint_sha256: str,
    dataset_manifest_sha256: str,
    source_bundle_sha256: str,
    fit_mode: str = CALIBRATION_MODE,
) -> PerActionAffineHazardCalibrator:
    action_ids = np.broadcast_to(
        np.arange(ACTION_COUNT, dtype=np.int64),
        calibration.hazard_targets.shape,
    )
    calibration_groups = tuple(
        group
        for group in calibration.episode_group_ids
        for _ in range(ACTION_COUNT)
    )
    provenance = TrainCalibrationProvenance(
        source_namespace="maze_chase.v21i.train-only.v1",
        source_split="TRAIN",
        source_partition="TRAIN-CAL",
        calibration_group_ids=calibration_groups,
        upstream_model_fit_group_ids=tuple(sorted(set(training.sequence_group_ids))),
        upstream_checkpoint_sha256=upstream_checkpoint_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        source_bundle_sha256=source_bundle_sha256,
        partition_algorithm=PARTITION_ALGORITHM,
    )
    calibrator = fit_train_only_per_action_affine(
        torch.from_numpy(calibration.raw_hazard_logits.reshape(-1)),
        torch.from_numpy(calibration.hazard_targets.reshape(-1)),
        torch.from_numpy(action_ids.reshape(-1)),
        provenance=provenance,
        action_count=ACTION_COUNT,
        fit_mode=fit_mode,
    )
    if not calibrator.accepted:
        raise RuntimeError(
            "TRAIN-CAL calibration did not strictly improve BCE; "
            "refusing to bake or publish an identity calibration"
        )
    return calibrator


def _canonical_group_digest(values: Sequence[str]) -> str:
    encoded = json.dumps(
        sorted(set(values)),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def calibration_expected_bindings(
    calibration: CollectedEvidence,
    training: CollectedEvidence,
    *,
    upstream_checkpoint_sha256: str,
    dataset_manifest_sha256: str,
    source_bundle_sha256: str,
) -> dict[str, object]:
    calibration_groups = tuple(
        group
        for group in calibration.episode_group_ids
        for _ in range(ACTION_COUNT)
    )
    upstream_groups = tuple(sorted(set(training.sequence_group_ids)))
    return {
        "source_namespace": "maze_chase.v21i.train-only.v1",
        "source_split": "TRAIN",
        "source_partition": "TRAIN-CAL",
        "calibration_group_sha256": _canonical_group_digest(calibration_groups),
        "upstream_model_fit_group_sha256": _canonical_group_digest(upstream_groups),
        "upstream_checkpoint_sha256": upstream_checkpoint_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_bundle_sha256": source_bundle_sha256,
        "partition_algorithm": PARTITION_ALGORITHM,
        "calibration_groups": len(set(calibration_groups)),
        "observations": calibration.hazard_targets.size,
        "per_action_observations": calibration.hazard_targets.shape[0],
    }


def install_calibrator(
    model: CoreV2Model,
    calibrator: PerActionAffineHazardCalibrator,
) -> dict[str, object]:
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    world_model = model.world_model
    if world_model is None or world_model.outcome_model is None:
        raise RuntimeError("V2.1i model has no all-action outcome model")
    world_model.outcome_model.install_hazard_calibration(calibrator)
    after = model.state_dict()
    changed = tuple(
        name for name in sorted(before) if not torch.equal(before[name], after[name])
    )
    allowed = {
        "world_model.outcome_model.hazard_calibration_scale",
        "world_model.outcome_model.hazard_calibration_bias",
    }
    if not set(changed).issubset(allowed):
        raise RuntimeError("hazard calibration changed a non-calibration tensor")
    return {
        "allowed_tensor_names": sorted(allowed),
        "changed_tensor_names": list(changed),
        "all_other_tensors_byte_equal": True,
        "state_sha256_before": _state_dict_sha256(before),
        "state_sha256_after": _state_dict_sha256(after),
    }


def _factual_action_support(
    targets: np.ndarray,
    factual_actions: np.ndarray,
    *,
    partition: str,
) -> dict[str, object]:
    target_values = np.asarray(targets, dtype=np.float64)
    action_values = np.asarray(factual_actions, dtype=np.int64)
    if target_values.ndim != 1 or action_values.shape != target_values.shape:
        raise ValueError(f"{partition} factual targets/actions have the wrong shape")
    if not np.isin(target_values, (0.0, 1.0)).all():
        raise ValueError(f"{partition} factual hazard targets must be binary")
    if not np.isin(action_values, np.arange(ACTION_COUNT)).all():
        raise ValueError(f"{partition} factual actions are outside the semantic range")
    per_action: list[dict[str, object]] = []
    for action_id in range(ACTION_COUNT):
        action_targets = target_values[action_values == action_id]
        positives = int(np.count_nonzero(action_targets == 1.0))
        observations = int(action_targets.size)
        negatives = observations - positives
        per_action.append(
            {
                "action_id": action_id,
                "observations": observations,
                "positives": positives,
                "negatives": negatives,
                "has_both_hazard_classes": positives > 0 and negatives > 0,
            }
        )
    return {
        "partition": partition,
        "observations": int(target_values.size),
        "every_action_has_both_hazard_classes": all(
            record["has_both_hazard_classes"] is True for record in per_action
        ),
        "per_action": per_action,
    }


def factual_baseline_eligibility_report(
    training_factual_targets: np.ndarray,
    training_factual_actions: np.ndarray,
    development_factual_targets: np.ndarray,
    development_factual_actions: np.ndarray,
    baseline_factual_metrics: Mapping[str, object],
) -> dict[str, object]:
    """Record whether factual-baseline comparisons are statistically eligible."""

    training_support = _factual_action_support(
        training_factual_targets,
        training_factual_actions,
        partition="TRAIN-FIT",
    )
    development_support = _factual_action_support(
        development_factual_targets,
        development_factual_actions,
        partition="DEV",
    )
    baseline_bce = float(baseline_factual_metrics["bce"])
    baseline_brier = float(baseline_factual_metrics["brier"])
    baseline_finite = math.isfinite(baseline_bce) and math.isfinite(baseline_brier)
    baseline_positive = baseline_bce > 0.0 and baseline_brier > 0.0
    passed = (
        training_support["every_action_has_both_hazard_classes"] is True
        and development_support["every_action_has_both_hazard_classes"] is True
        and baseline_finite
        and baseline_positive
    )
    return {
        "schema_version": 1,
        "action_ids": list(range(ACTION_COUNT)),
        "minimum_per_class_per_action": 1,
        "partitions": {
            "TRAIN-FIT": training_support,
            "DEV": development_support,
        },
        "baseline_factual": {
            "bce": baseline_bce,
            "brier": baseline_brier,
            "finite": baseline_finite,
            "strictly_positive": baseline_positive,
        },
        "passed": passed,
    }


def evaluate_development_metrics(
    training: CollectedEvidence,
    calibration: CollectedEvidence,
    development: CollectedEvidence,
    calibrator: PerActionAffineHazardCalibrator,
    *,
    uncalibrated_development: CollectedEvidence | None = None,
) -> dict[str, object]:
    raw_development = (
        development
        if uncalibrated_development is None
        else uncalibrated_development
    )
    action_ids = tuple(range(ACTION_COUNT))
    training_factual_targets, _ = gather_factual_rows(
        training.hazard_targets,
        torch.sigmoid(torch.from_numpy(training.raw_hazard_logits)).numpy(),
        training.factual_actions,
        action_ids=action_ids,
    )
    baselines = fit_train_baselines(
        training.hazard_targets,
        training.factual_actions,
        training_factual_targets,
        action_ids=action_ids,
    )
    dev_factual_targets, dev_factual_probabilities = gather_factual_rows(
        development.hazard_targets,
        development.hazard_probabilities,
        development.factual_actions,
        action_ids=action_ids,
    )
    raw_dev_probabilities = torch.sigmoid(
        torch.from_numpy(raw_development.raw_hazard_logits)
    ).numpy()
    raw_factual_targets, raw_factual_probabilities = gather_factual_rows(
        raw_development.hazard_targets,
        raw_dev_probabilities,
        raw_development.factual_actions,
        action_ids=action_ids,
    )
    baseline_metrics = evaluate_train_fitted_baselines(
        baselines,
        development.hazard_targets,
        development.factual_actions,
        dev_factual_targets,
        ece_bins=ECE_BINS,
    )
    baseline_metrics_dict = baseline_metrics.as_dict()
    factual_baseline_eligibility = factual_baseline_eligibility_report(
        training_factual_targets,
        training.factual_actions,
        dev_factual_targets,
        development.factual_actions,
        baseline_metrics_dict["factual"],
    )
    baseline_table = baselines.all_action.probability_table(
        development.hazard_targets.shape[0]
    )
    factual_baseline = baselines.factual.probabilities_for_actions(
        development.factual_actions
    )
    calibration_action_ids = torch.arange(ACTION_COUNT, dtype=torch.long).expand(
        calibration.raw_hazard_logits.shape[0], -1
    )
    transformed_calibration = calibrator.transform_all_actions(
        torch.from_numpy(calibration.raw_hazard_logits),
        calibration_action_ids,
    )
    calibration_probabilities = torch.sigmoid(transformed_calibration).numpy()
    return {
        "calibration_fit_population": {
            "raw": all_action_probability_metrics(
                calibration.hazard_targets,
                torch.sigmoid(torch.from_numpy(calibration.raw_hazard_logits)).numpy(),
                action_ids=action_ids,
                ece_bins=ECE_BINS,
            ).as_dict(),
            "calibrated": all_action_probability_metrics(
                calibration.hazard_targets,
                calibration_probabilities,
                action_ids=action_ids,
                ece_bins=ECE_BINS,
            ).as_dict(),
        },
        "dev": {
            "raw_all_action": all_action_probability_metrics(
                raw_development.hazard_targets,
                raw_dev_probabilities,
                action_ids=action_ids,
                ece_bins=ECE_BINS,
            ).as_dict(),
            "calibrated_all_action": all_action_probability_metrics(
                development.hazard_targets,
                development.hazard_probabilities,
                action_ids=action_ids,
                ece_bins=ECE_BINS,
            ).as_dict(),
            "raw_factual": binary_probability_metrics(
                raw_factual_targets,
                raw_factual_probabilities,
                ece_bins=ECE_BINS,
            ).as_dict(),
            "calibrated_factual": binary_probability_metrics(
                dev_factual_targets,
                dev_factual_probabilities,
                ece_bins=ECE_BINS,
            ).as_dict(),
            "train_fitted_baselines": baselines.as_dict(),
            "baseline_metrics": baseline_metrics_dict,
            "factual_baseline_eligibility": factual_baseline_eligibility,
            "causal_cross_episode_derangements": evaluate_cross_episode_derangements(
                development.hazard_targets,
                development.hazard_probabilities,
                development.episode_group_ids,
                action_ids=action_ids,
                repetitions=DERANGEMENT_REPETITIONS,
                seed=DERANGEMENT_SEED,
                ece_bins=ECE_BINS,
            ).as_dict(include_repetitions=True),
            "all_action_clustered_bootstrap": clustered_improvement_bootstrap(
                development.hazard_targets,
                development.hazard_probabilities,
                baseline_table,
                development.episode_group_ids,
                resamples=BOOTSTRAP_RESAMPLES,
                seed=BOOTSTRAP_SEED,
            ).as_dict(),
            "factual_clustered_bootstrap": clustered_improvement_bootstrap(
                dev_factual_targets,
                dev_factual_probabilities,
                factual_baseline,
                development.episode_group_ids,
                resamples=BOOTSTRAP_RESAMPLES,
                seed=BOOTSTRAP_SEED + 1,
            ).as_dict(),
            "belief_noncollapse": evaluate_latent_noncollapse(
                development.beliefs,
                thresholds=LATENT_THRESHOLDS,
            ).as_dict(),
            "raw_semantic_hazard_logits": {
                "retained": True,
                "action_ids": list(action_ids),
                "shape": list(raw_development.raw_hazard_logits.shape),
                "sha256": sha256(
                    raw_development.raw_hazard_logits.astype(
                        "<f8",
                        copy=False,
                    ).tobytes(order="C")
                ).hexdigest(),
            },
        },
    }


def assert_evidence_partitions_disjoint(
    evidence: Mapping[str, CollectedEvidence],
) -> dict[str, object]:
    """Prove that no factual root appears in more than one partition."""

    names = tuple(sorted(evidence))
    overlaps: dict[str, int] = {}
    for left_index, left in enumerate(names):
        left_roots = set(evidence[left].root_state_ids)
        for right in names[left_index + 1 :]:
            overlap = left_roots.intersection(evidence[right].root_state_ids)
            overlaps[f"{left}::{right}"] = len(overlap)
            if overlap:
                raise RuntimeError(
                    f"development partitions share factual root {min(overlap)}"
                )
    return {
        "partition_names": list(names),
        "pairwise_root_overlap_counts": overlaps,
        "all_factual_roots_disjoint": True,
    }


def assert_dev_replay_identity(
    initial: CollectedEvidence,
    parent: CollectedEvidence,
    installed: CollectedEvidence,
) -> dict[str, object]:
    """Require every model state to see the same ordered DEV roots and labels."""

    for candidate in (parent, installed):
        if candidate.root_state_ids != initial.root_state_ids:
            raise RuntimeError("DEV root order/content changed between model states")
        if not np.array_equal(candidate.hazard_targets, initial.hazard_targets):
            raise RuntimeError("DEV hazard labels changed between model states")
        if not np.array_equal(candidate.factual_actions, initial.factual_actions):
            raise RuntimeError("DEV factual actions changed between model states")
    return {
        "ordered_roots_equal": True,
        "hazard_labels_equal": True,
        "factual_actions_equal": True,
        "root_count": len(initial.root_state_ids),
        "ordered_root_sha256": sha256(
            "".join(initial.root_state_ids).encode("ascii")
        ).hexdigest(),
    }


def installed_model_preservation_report(
    uncalibrated: Mapping[str, object],
    calibrated: Mapping[str, object],
) -> dict[str, object]:
    """Compare non-hazard end-to-end behavior before and after calibration."""

    before_components = uncalibrated["components"]
    after_components = calibrated["components"]
    before_metrics = uncalibrated["metrics"]
    after_metrics = calibrated["metrics"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            before_components,
            after_components,
            before_metrics,
            after_metrics,
        )
    ):
        raise TypeError("joint-objective preservation inputs are malformed")

    component_ratios: dict[str, float] = {}
    component_checks: dict[str, bool] = {}
    for name in ("decision", "next_latent", "reward"):
        before = _finite_number(before_components[name])
        after = _finite_number(after_components[name])
        ratio = after / before if before > 0.0 else (1.0 if after == 0.0 else math.inf)
        component_ratios[name] = ratio
        component_checks[name] = ratio <= 1.01
    before_cosine = _finite_number(before_metrics["next_latent_cosine"])
    after_cosine = _finite_number(after_metrics["next_latent_cosine"])
    cosine_drop = before_cosine - after_cosine
    before_reward_mae = _finite_number(before_metrics["reward_mae"])
    after_reward_mae = _finite_number(after_metrics["reward_mae"])
    reward_mae_ratio = (
        after_reward_mae / before_reward_mae
        if before_reward_mae > 0.0
        else (1.0 if after_reward_mae == 0.0 else math.inf)
    )
    checks = {
        "decision_loss_ratio_at_most_1_01": component_checks["decision"],
        "next_loss_ratio_at_most_1_01": component_checks["next_latent"],
        "reward_loss_ratio_at_most_1_01": component_checks["reward"],
        "next_cosine_drop_at_most_0_01": cosine_drop <= 0.01,
        "reward_mae_ratio_at_most_1_02": reward_mae_ratio <= 1.02,
    }
    return {
        "component_loss_ratios": component_ratios,
        "next_latent_cosine_drop": cosine_drop,
        "reward_mae_ratio": reward_mae_ratio,
        "checks": checks,
        "passed": all(checks.values()),
    }


def learning_from_initialization_report(
    initial_objective: Mapping[str, object],
    final_objective: Mapping[str, object],
    initial_evidence: CollectedEvidence,
    final_evidence: CollectedEvidence,
) -> dict[str, object]:
    """Apply the frozen aggregate and per-action learning-ratio gates."""

    initial_components = initial_objective["components"]
    final_components = final_objective["components"]
    if not isinstance(initial_components, Mapping) or not isinstance(
        final_components,
        Mapping,
    ):
        raise TypeError("initial/final objective components must be mappings")
    aggregate_limits = {"hazard": 0.90, "next_latent": 0.60, "reward": 0.80}
    aggregate_ratios = {
        name: _finite_number(final_components[name])
        / _finite_number(initial_components[name])
        for name in aggregate_limits
    }
    aggregate_checks = {
        name: aggregate_ratios[name] <= limit
        for name, limit in aggregate_limits.items()
    }
    initial_next = np.mean(initial_evidence.next_latent_losses, axis=0)
    final_next = np.mean(final_evidence.next_latent_losses, axis=0)
    initial_reward = np.mean(initial_evidence.reward_losses, axis=0)
    final_reward = np.mean(final_evidence.reward_losses, axis=0)
    if np.any(initial_next <= 0.0) or np.any(initial_reward <= 0.0):
        raise ValueError("initial per-action predictive losses must be positive")
    per_action_next_ratios = final_next / initial_next
    per_action_reward_ratios = final_reward / initial_reward
    checks = {
        "hazard_loss_ratio_at_most_0_90": aggregate_checks["hazard"],
        "next_loss_ratio_at_most_0_60": aggregate_checks["next_latent"],
        "reward_loss_ratio_at_most_0_80": aggregate_checks["reward"],
        "every_action_next_loss_ratio_at_most_0_90": bool(
            np.all(per_action_next_ratios <= 0.90)
        ),
        "every_action_reward_loss_ratio_at_most_0_90": bool(
            np.all(per_action_reward_ratios <= 0.90)
        ),
    }
    return {
        "aggregate_loss_ratios": aggregate_ratios,
        "per_action_next_loss_ratios": per_action_next_ratios.tolist(),
        "per_action_reward_loss_ratios": per_action_reward_ratios.tolist(),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("gate metric must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("gate metric must be finite")
    return result


def _joint_training_schedule_is_exact(record: Mapping[str, object]) -> bool:
    telemetry = record["telemetry"]
    expected_samples = TRAIN_FIT_EPISODES * (SEQUENCE_LENGTH - BURN_IN_STEPS)
    if not isinstance(telemetry, list):
        return False
    telemetry_valid = len(telemetry) == TRAINING_EPOCHS
    for epoch, entry in enumerate(telemetry, start=1):
        if not isinstance(entry, Mapping):
            return False
        components = entry.get("components")
        telemetry_valid = telemetry_valid and (
            entry.get("epoch") == epoch
            and entry.get("samples") == expected_samples
            and _finite_number(entry.get("loss")) >= 0.0
            and _finite_number(entry.get("maximum_preclip_gradient_norm")) >= 0.0
            and isinstance(components, Mapping)
            and bool(components)
            and all(_finite_number(value) >= 0.0 for value in components.values())
        )
    return (
        record.get("source_partition") == "TRAIN-FIT"
        and record.get("optimizer") == "AdamW"
        and record.get("epochs") == TRAINING_EPOCHS
        and record.get("optimizer_steps") == 48
        and record.get("batch_size") == BATCH_SIZE
        and record.get("sequence_count") == TRAIN_FIT_EPISODES
        and record.get("sequence_length") == SEQUENCE_LENGTH
        and record.get("burn_in_steps") == BURN_IN_STEPS
        and _finite_number(record.get("learning_rate")) == LEARNING_RATE
        and _finite_number(record.get("weight_decay")) == WEIGHT_DECAY
        and _finite_number(record.get("clip_norm")) == CLIP_NORM
        and record.get("prediction_error_intervention") == "normal"
        and record.get("target_encoder_update") == "after_every_optimizer_step"
        and record.get("sampling")
        == "deterministic_episode_permutation_without_replacement"
        and _finite_number(record.get("wall_seconds")) >= 0.0
        and telemetry_valid
    )


def v21i_development_gate(
    calibration_parameters: Mapping[str, object],
    metrics: Mapping[str, object],
    installation_audit: Mapping[str, object],
    expected_calibration_bindings: Mapping[str, object],
) -> dict[str, object]:
    """Apply every preregistered per-seed gate, failing closed on missing data."""

    check_names = (
        "calibration_strictly_improved_train_cal_bce",
        "calibration_full_schema_and_fit_records",
        "calibration_bias_only",
        "calibration_provenance_complete",
        "calibrated_dev_bce_not_worse_than_parent",
        "joint_training_exact_schedule",
        "dev_bce_beats_train_prior",
        "dev_brier_beats_train_prior",
        "dev_aggregate_roc_auc",
        "dev_aggregate_roc_auc_gain",
        "dev_every_action_roc_auc",
        "dev_every_action_pr_auc",
        "dev_every_action_nonnegative_brier_skill",
        "dev_aggregate_ece",
        "dev_every_action_ece",
        "dev_every_action_calibration_bias",
        "factual_baseline_eligible_and_supported",
        "dev_factual_brier_beats_train_prior",
        "twenty_cross_episode_derangements",
        "shuffle_bce_degradation",
        "shuffle_aggregate_auc_drop",
        "shuffle_every_action_auc_drop",
        "all_action_bootstrap_bce_lower_bound",
        "all_action_bootstrap_brier_lower_bound",
        "factual_bootstrap_bce_lower_bound",
        "factual_bootstrap_brier_lower_bound",
        "belief_noncollapse",
        "episode_roots_disjoint",
        "dev_replay_identity",
        "dev_evaluation_state_immutable",
        "hazard_refinement_exact_train_fit_schedule",
        "raw_semantic_hazard_logits_retained",
        "learned_hazard_from_initialization",
        "learned_next_from_initialization",
        "learned_reward_from_initialization",
        "learned_next_every_action_from_initialization",
        "learned_reward_every_action_from_initialization",
        "installed_decision_loss_preserved",
        "installed_next_loss_preserved",
        "installed_reward_loss_preserved",
        "installed_next_cosine_preserved",
        "installed_reward_mae_preserved",
        "calibration_buffer_only_diff",
        "all_model_parameters_and_buffers_finite",
    )
    checks = {name: False for name in check_names}
    errors: list[str] = []
    try:
        fit_bce = calibration_parameters["fit_bce"]
        fitted_on = calibration_parameters["fitted_on"]
        if not isinstance(fit_bce, Mapping) or not isinstance(fitted_on, Mapping):
            raise TypeError("calibration fit/provenance must be mappings")
        scales = calibration_parameters["scale"]
        biases = calibration_parameters["bias"]
        per_action_fit = calibration_parameters["per_action"]
        if (
            not isinstance(scales, list)
            or not isinstance(biases, list)
            or not isinstance(per_action_fit, list)
            or len(scales) != ACTION_COUNT
            or len(biases) != ACTION_COUNT
            or len(per_action_fit) != ACTION_COUNT
        ):
            raise ValueError("calibrator must contain exactly five action records")
        cal_before = _finite_number(fit_bce["before"])
        cal_after = _finite_number(fit_bce["after"])
        finite_scales = [_finite_number(value) for value in scales]
        finite_biases = [_finite_number(value) for value in biases]
        per_action_records_valid = True
        per_action_observation_sum = 0
        per_action_before_weighted = 0.0
        per_action_after_weighted = 0.0
        expected_per_action_observations = expected_calibration_bindings.get(
            "per_action_observations"
        )
        for action_id, record in enumerate(per_action_fit):
            if not isinstance(record, Mapping):
                raise TypeError("calibrator per-action fit record must be a mapping")
            observations = record.get("observations")
            positives = record.get("positives")
            negatives = record.get("negatives")
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (observations, positives, negatives)
            ):
                raise TypeError("calibrator per-action counts must be integers")
            before = _finite_number(record["before_bce"])
            after = _finite_number(record["after_bce"])
            accepted = record.get("accepted")
            record_valid = (
                record.get("action_id") == action_id
                and observations == expected_per_action_observations
                and positives >= 2
                and negatives >= 2
                and positives + negatives == observations
                and before >= 0.0
                and after >= 0.0
                and isinstance(accepted, bool)
                and accepted == (after < before)
                and (accepted or after == before)
            )
            per_action_records_valid = per_action_records_valid and record_valid
            per_action_observation_sum += observations
            per_action_before_weighted += before * observations
            per_action_after_weighted += after * observations
        fitted_observations = fitted_on.get("observations")
        if isinstance(fitted_observations, bool) or not isinstance(
            fitted_observations,
            int,
        ):
            raise TypeError("calibration observations must be an integer")
        if fitted_observations <= 0:
            raise ValueError("calibration observations must be positive")
        weighted_before = per_action_before_weighted / fitted_observations
        weighted_after = per_action_after_weighted / fitted_observations
        checks["calibration_full_schema_and_fit_records"] = (
            calibration_parameters.get("schema") == CALIBRATION_SCHEMA
            and calibration_parameters.get("formula") == CALIBRATION_FORMULA
            and calibration_parameters.get("action_count") == ACTION_COUNT
            and len(finite_biases) == ACTION_COUNT
            and all(scale > 0.0 for scale in finite_scales)
            and per_action_records_valid
            and per_action_observation_sum == fitted_observations
            and math.isclose(weighted_before, cal_before, rel_tol=1.0e-9, abs_tol=1.0e-12)
            and math.isclose(weighted_after, cal_after, rel_tol=1.0e-9, abs_tol=1.0e-12)
        )
        checks["calibration_strictly_improved_train_cal_bce"] = (
            calibration_parameters.get("accepted") is True and cal_after < cal_before
        )
        checks["calibration_bias_only"] = (
            calibration_parameters.get("fit_mode") == CALIBRATION_MODE
            and all(scale == 1.0 for scale in finite_scales)
        )
        binding_keys = (
            "source_namespace",
            "source_split",
            "source_partition",
            "calibration_group_sha256",
            "upstream_model_fit_group_sha256",
            "upstream_checkpoint_sha256",
            "dataset_manifest_sha256",
            "source_bundle_sha256",
            "partition_algorithm",
            "calibration_groups",
            "observations",
        )
        checks["calibration_provenance_complete"] = (
            all(fitted_on.get(key) == expected_calibration_bindings.get(key) for key in binding_keys)
            and all(
                _is_sha256(fitted_on.get(key))
                for key in (
                    "calibration_group_sha256",
                    "upstream_model_fit_group_sha256",
                    "upstream_checkpoint_sha256",
                    "dataset_manifest_sha256",
                    "source_bundle_sha256",
                )
            )
            and fitted_on.get("source_namespace") == "maze_chase.v21i.train-only.v1"
            and fitted_on.get("source_split") == "TRAIN"
            and fitted_on.get("source_partition") == "TRAIN-CAL"
            and fitted_on.get("partition_algorithm") == PARTITION_ALGORITHM
            and fitted_on.get("calibration_groups") == TRAIN_CAL_EPISODES
            and fitted_on.get("observations")
            == TRAIN_CAL_EPISODES * (SEQUENCE_LENGTH - BURN_IN_STEPS) * ACTION_COUNT
        )

        dev = metrics["dev"]
        if not isinstance(dev, Mapping):
            raise TypeError("DEV metrics must be a mapping")
        model = dev["calibrated_all_action"]
        baseline_report = dev["baseline_metrics"]
        if not isinstance(model, Mapping) or not isinstance(baseline_report, Mapping):
            raise TypeError("model/baseline metrics must be mappings")
        aggregate = model["aggregate"]
        per_action = model["per_action"]
        baseline_all = baseline_report["all_action"]
        baseline_factual = baseline_report["factual"]
        baseline_aggregate = baseline_all["aggregate"]
        baseline_per_action = baseline_all["per_action"]
        if (
            not isinstance(aggregate, Mapping)
            or not isinstance(per_action, list)
            or not isinstance(baseline_aggregate, Mapping)
            or not isinstance(baseline_per_action, list)
        ):
            raise TypeError("aggregate/per-action metrics have the wrong shape")
        if len(per_action) != ACTION_COUNT or len(baseline_per_action) != ACTION_COUNT:
            raise ValueError("gate requires exactly five semantic actions")
        if [entry["action_id"] for entry in per_action] != list(range(ACTION_COUNT)):
            raise ValueError("model action metrics are not in canonical order")
        if [entry["action_id"] for entry in baseline_per_action] != list(
            range(ACTION_COUNT)
        ):
            raise ValueError("baseline action metrics are not in canonical order")

        eligibility = dev["factual_baseline_eligibility"]
        if not isinstance(eligibility, Mapping):
            raise TypeError("factual baseline eligibility must be a mapping")
        eligibility_partitions = eligibility["partitions"]
        eligibility_baseline = eligibility["baseline_factual"]
        if not isinstance(eligibility_partitions, Mapping) or not isinstance(
            eligibility_baseline,
            Mapping,
        ):
            raise TypeError("factual baseline support/baseline must be mappings")
        support_valid = True
        expected_partition_observations = {
            "TRAIN-FIT": TRAIN_FIT_EPISODES * (SEQUENCE_LENGTH - BURN_IN_STEPS),
            "DEV": DEV_EPISODES * (SEQUENCE_LENGTH - BURN_IN_STEPS),
        }
        for partition, expected_observations in expected_partition_observations.items():
            support = eligibility_partitions[partition]
            if not isinstance(support, Mapping):
                raise TypeError("factual action support must be a mapping")
            action_support = support["per_action"]
            if not isinstance(action_support, list) or len(action_support) != ACTION_COUNT:
                raise ValueError("factual action support must cover all five actions")
            partition_observation_sum = 0
            for action_id, record in enumerate(action_support):
                if not isinstance(record, Mapping):
                    raise TypeError("per-action factual support must be a mapping")
                counts = (
                    record.get("observations"),
                    record.get("positives"),
                    record.get("negatives"),
                )
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in counts
                ):
                    raise TypeError("factual support counts must be integers")
                observations, positives, negatives = counts
                record_valid = (
                    record.get("action_id") == action_id
                    and observations > 0
                    and positives >= 1
                    and negatives >= 1
                    and positives + negatives == observations
                    and record.get("has_both_hazard_classes") is True
                )
                support_valid = support_valid and record_valid
                partition_observation_sum += observations
            support_valid = support_valid and (
                support.get("partition") == partition
                and support.get("observations") == expected_observations
                and partition_observation_sum == expected_observations
                and support.get("every_action_has_both_hazard_classes") is True
            )
        eligibility_bce = _finite_number(eligibility_baseline["bce"])
        eligibility_brier = _finite_number(eligibility_baseline["brier"])
        reported_factual_bce = _finite_number(baseline_factual["bce"])
        reported_factual_brier = _finite_number(baseline_factual["brier"])
        checks["factual_baseline_eligible_and_supported"] = (
            eligibility.get("schema_version") == 1
            and eligibility.get("action_ids") == list(range(ACTION_COUNT))
            and eligibility.get("minimum_per_class_per_action") == 1
            and set(eligibility_partitions) == {"TRAIN-FIT", "DEV"}
            and support_valid
            and eligibility_baseline.get("finite") is True
            and eligibility_baseline.get("strictly_positive") is True
            and eligibility_bce > 0.0
            and eligibility_brier > 0.0
            and math.isclose(
                eligibility_bce,
                reported_factual_bce,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
            and math.isclose(
                eligibility_brier,
                reported_factual_brier,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
            and eligibility.get("passed") is True
        )

        model_bce = _finite_number(aggregate["bce"])
        model_brier = _finite_number(aggregate["brier"])
        model_auc = _finite_number(aggregate["roc_auc"])
        parent_bce = _finite_number(dev["raw_all_action"]["aggregate"]["bce"])
        baseline_bce = _finite_number(baseline_aggregate["bce"])
        baseline_brier = _finite_number(baseline_aggregate["brier"])
        baseline_auc = _finite_number(baseline_aggregate["roc_auc"])
        checks["calibrated_dev_bce_not_worse_than_parent"] = model_bce <= parent_bce
        joint = dev["joint_training"]
        if not isinstance(joint, Mapping):
            raise TypeError("joint training schedule must be a mapping")
        checks["joint_training_exact_schedule"] = _joint_training_schedule_is_exact(
            joint
        )
        checks["dev_bce_beats_train_prior"] = (
            model_bce <= MAX_BASELINE_BCE_RATIO * baseline_bce
        )
        checks["dev_brier_beats_train_prior"] = (
            model_brier <= MAX_BASELINE_BRIER_RATIO * baseline_brier
        )
        checks["dev_aggregate_roc_auc"] = model_auc >= MINIMUM_AGGREGATE_ROC_AUC
        checks["dev_aggregate_roc_auc_gain"] = (
            model_auc >= baseline_auc + MINIMUM_BASELINE_ROC_AUC_GAIN
        )
        action_metrics = [entry["metrics"] for entry in per_action]
        baseline_action_metrics = [entry["metrics"] for entry in baseline_per_action]
        checks["dev_every_action_roc_auc"] = all(
            _finite_number(entry["roc_auc"]) >= MINIMUM_PER_ACTION_ROC_AUC
            for entry in action_metrics
        )
        checks["dev_every_action_pr_auc"] = all(
            _finite_number(entry["pr_auc"])
            >= _finite_number(entry["prevalence"])
            + MINIMUM_PER_ACTION_PR_PREVALENCE_GAIN
            for entry in action_metrics
        )
        checks["dev_every_action_nonnegative_brier_skill"] = all(
            _finite_number(entry["brier"]) <= _finite_number(baseline["brier"])
            for entry, baseline in zip(
                action_metrics,
                baseline_action_metrics,
                strict=True,
            )
        )
        checks["dev_aggregate_ece"] = (
            _finite_number(aggregate["ece_equal_mass"]) <= MAXIMUM_AGGREGATE_ECE
        )
        checks["dev_every_action_ece"] = all(
            _finite_number(entry["ece_equal_mass"]) <= MAXIMUM_PER_ACTION_ECE
            for entry in action_metrics
        )
        checks["dev_every_action_calibration_bias"] = all(
            abs(_finite_number(entry["calibration_bias"]))
            <= MAXIMUM_ABSOLUTE_PER_ACTION_BIAS
            for entry in action_metrics
        )
        checks["dev_factual_brier_beats_train_prior"] = (
            _finite_number(dev["calibrated_factual"]["brier"])
            < _finite_number(baseline_report["factual"]["brier"])
        )

        shuffle = dev["causal_cross_episode_derangements"]
        if not isinstance(shuffle, Mapping):
            raise TypeError("shuffle report must be a mapping")
        shuffle_actions = shuffle["per_action"]
        if not isinstance(shuffle_actions, list) or len(shuffle_actions) != ACTION_COUNT:
            raise ValueError("shuffle report must cover every action")
        checks["twenty_cross_episode_derangements"] = (
            shuffle.get("repetitions") == DERANGEMENT_REPETITIONS
            and isinstance(shuffle.get("shuffled"), list)
            and len(shuffle["shuffled"]) == DERANGEMENT_REPETITIONS
        )
        checks["shuffle_bce_degradation"] = (
            _finite_number(shuffle["median_shuffled_bce"])
            >= MINIMUM_SHUFFLED_BCE_RATIO * model_bce
        )
        checks["shuffle_aggregate_auc_drop"] = (
            _finite_number(shuffle["aggregate_roc_auc_drop"])
            >= MINIMUM_SHUFFLED_AGGREGATE_AUC_DROP
        )
        checks["shuffle_every_action_auc_drop"] = all(
            _finite_number(entry["roc_auc_drop"])
            >= MINIMUM_SHUFFLED_PER_ACTION_AUC_DROP
            for entry in shuffle_actions
        )

        all_bootstrap = dev["all_action_clustered_bootstrap"]
        factual_bootstrap = dev["factual_clustered_bootstrap"]
        checks["all_action_bootstrap_bce_lower_bound"] = (
            _finite_number(all_bootstrap["bce_improvement_lower_bound"]) > 0.0
        )
        checks["all_action_bootstrap_brier_lower_bound"] = (
            _finite_number(all_bootstrap["brier_improvement_lower_bound"]) > 0.0
        )
        checks["factual_bootstrap_bce_lower_bound"] = (
            _finite_number(factual_bootstrap["bce_improvement_lower_bound"]) > 0.0
        )
        checks["factual_bootstrap_brier_lower_bound"] = (
            _finite_number(factual_bootstrap["brier_improvement_lower_bound"]) > 0.0
        )
        checks["belief_noncollapse"] = dev["belief_noncollapse"].get("passed") is True
        checks["episode_roots_disjoint"] = (
            dev["partition_disjointness"].get("all_factual_roots_disjoint") is True
        )
        replay_identity = dev["replay_identity"]
        checks["dev_replay_identity"] = (
            replay_identity.get("ordered_roots_equal") is True
            and replay_identity.get("hazard_labels_equal") is True
            and replay_identity.get("factual_actions_equal") is True
            and _is_sha256(replay_identity.get("ordered_root_sha256"))
        )
        state_immutability = dev["state_immutability"]
        checks["dev_evaluation_state_immutable"] = (
            state_immutability.get("initialized_reference_unchanged") is True
            and state_immutability.get("uncalibrated_parent_unchanged") is True
            and state_immutability.get("installed_model_unchanged") is True
        )
        refinement = dev["hazard_refinement"]
        checks["hazard_refinement_exact_train_fit_schedule"] = (
            refinement.get("enabled") is True
            and refinement.get("source_partition") == "TRAIN-FIT"
            and refinement.get("dev_used_for_stopping") is False
            and refinement.get("passes") == HAZARD_REFINEMENT_PASSES
            and refinement.get("root_batch_size")
            == HAZARD_REFINEMENT_ROOT_BATCH_SIZE
            and refinement.get("root_count")
            == TRAIN_FIT_EPISODES * (SEQUENCE_LENGTH - BURN_IN_STEPS)
            and refinement.get("optimizer_steps_per_pass") == 64
            and refinement.get("optimizer_steps") == 512
            and refinement.get("trainable_parameter_count") == 44_161
            and refinement.get("all_other_tensors_byte_equal") is True
            and refinement.get("sampling")
            == "deterministic_complete_root_permutation_without_replacement"
        )
        raw_logits = dev["raw_semantic_hazard_logits"]
        checks["raw_semantic_hazard_logits_retained"] = (
            raw_logits.get("retained") is True
            and raw_logits.get("action_ids") == list(range(ACTION_COUNT))
            and isinstance(raw_logits.get("shape"), list)
            and len(raw_logits["shape"]) == 2
            and raw_logits["shape"][1] == ACTION_COUNT
            and _is_sha256(raw_logits.get("sha256"))
        )
        learning = dev["learning_from_initialization"]
        learning_checks = learning["checks"]
        checks["learned_hazard_from_initialization"] = (
            learning_checks.get("hazard_loss_ratio_at_most_0_90") is True
        )
        checks["learned_next_from_initialization"] = (
            learning_checks.get("next_loss_ratio_at_most_0_60") is True
        )
        checks["learned_reward_from_initialization"] = (
            learning_checks.get("reward_loss_ratio_at_most_0_80") is True
        )
        checks["learned_next_every_action_from_initialization"] = (
            learning_checks.get("every_action_next_loss_ratio_at_most_0_90") is True
        )
        checks["learned_reward_every_action_from_initialization"] = (
            learning_checks.get("every_action_reward_loss_ratio_at_most_0_90") is True
        )
        preservation = dev["installed_model_preservation"]
        preservation_checks = preservation["checks"]
        checks["installed_decision_loss_preserved"] = (
            preservation_checks.get("decision_loss_ratio_at_most_1_01") is True
        )
        checks["installed_next_loss_preserved"] = (
            preservation_checks.get("next_loss_ratio_at_most_1_01") is True
        )
        checks["installed_reward_loss_preserved"] = (
            preservation_checks.get("reward_loss_ratio_at_most_1_01") is True
        )
        checks["installed_next_cosine_preserved"] = (
            preservation_checks.get("next_cosine_drop_at_most_0_01") is True
        )
        checks["installed_reward_mae_preserved"] = (
            preservation_checks.get("reward_mae_ratio_at_most_1_02") is True
        )

        allowed = {
            "world_model.outcome_model.hazard_calibration_scale",
            "world_model.outcome_model.hazard_calibration_bias",
        }
        changed = installation_audit["changed_tensor_names"]
        checks["calibration_buffer_only_diff"] = (
            installation_audit.get("all_other_tensors_byte_equal") is True
            and isinstance(changed, list)
            and bool(changed)
            and set(changed).issubset(allowed)
            and installation_audit.get("allowed_tensor_names") == sorted(allowed)
            and installation_audit.get("state_sha256_before")
            != installation_audit.get("state_sha256_after")
        )
        finite_state = dev["finite_state_audit"]
        checks["all_model_parameters_and_buffers_finite"] = (
            finite_state.get("all_parameters_finite") is True
            and finite_state.get("all_buffers_finite") is True
            and isinstance(finite_state.get("parameter_tensor_count"), int)
            and finite_state["parameter_tensor_count"] > 0
            and isinstance(finite_state.get("buffer_tensor_count"), int)
            and finite_state["buffer_tensor_count"] > 0
            and _is_sha256(finite_state.get("state_sha256"))
            and finite_state.get("state_sha256")
            == installation_audit.get("state_sha256_after")
        )
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")

    return {
        "schema_version": 1,
        "gate": "v21i_development_per_seed_v1",
        "thresholds": {
            "maximum_baseline_bce_ratio": MAX_BASELINE_BCE_RATIO,
            "maximum_baseline_brier_ratio": MAX_BASELINE_BRIER_RATIO,
            "minimum_aggregate_roc_auc": MINIMUM_AGGREGATE_ROC_AUC,
            "minimum_baseline_roc_auc_gain": MINIMUM_BASELINE_ROC_AUC_GAIN,
            "minimum_per_action_roc_auc": MINIMUM_PER_ACTION_ROC_AUC,
            "minimum_per_action_pr_prevalence_gain": (
                MINIMUM_PER_ACTION_PR_PREVALENCE_GAIN
            ),
            "maximum_aggregate_ece": MAXIMUM_AGGREGATE_ECE,
            "maximum_per_action_ece": MAXIMUM_PER_ACTION_ECE,
            "maximum_absolute_per_action_bias": MAXIMUM_ABSOLUTE_PER_ACTION_BIAS,
            "minimum_factual_class_count_per_action": 1,
            "minimum_factual_baseline_bce_exclusive": 0.0,
            "minimum_factual_baseline_brier_exclusive": 0.0,
            "minimum_shuffled_bce_ratio": MINIMUM_SHUFFLED_BCE_RATIO,
            "minimum_shuffled_aggregate_auc_drop": (
                MINIMUM_SHUFFLED_AGGREGATE_AUC_DROP
            ),
            "minimum_shuffled_per_action_auc_drop": (
                MINIMUM_SHUFFLED_PER_ACTION_AUC_DROP
            ),
            "bootstrap_lower_bounds_strictly_greater_than": 0.0,
            "calibration_fit_mode": CALIBRATION_MODE,
            "maximum_calibrated_to_parent_dev_bce_ratio": 1.0,
            "joint_optimizer_steps": 48,
            "maximum_initial_hazard_loss_ratio": 0.90,
            "maximum_initial_next_loss_ratio": 0.60,
            "maximum_initial_reward_loss_ratio": 0.80,
            "maximum_per_action_initial_next_loss_ratio": 0.90,
            "maximum_per_action_initial_reward_loss_ratio": 0.90,
            "maximum_installed_component_loss_ratio": 1.01,
            "maximum_installed_next_cosine_drop": 0.01,
            "maximum_installed_reward_mae_ratio": 1.02,
        },
        "checks": checks,
        "errors": errors,
        "passed": not errors and all(checks.values()),
    }


def v21i_five_seed_cohort_gate(
    per_seed_gates: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    """Require exactly the preregistered five seeds and a pass from every one."""

    supplied = set(per_seed_gates)
    required = set(REQUIRED_MODEL_SEEDS)
    seed_passes = {
        seed: (
            seed in per_seed_gates
            and isinstance(per_seed_gates[seed], Mapping)
            and per_seed_gates[seed].get("passed") is True
        )
        for seed in REQUIRED_MODEL_SEEDS
    }
    exact_cohort = supplied == required
    return {
        "schema_version": 1,
        "gate": "v21i_five_seed_cohort_v1",
        "required_model_seeds": list(REQUIRED_MODEL_SEEDS),
        "required_passes": len(REQUIRED_MODEL_SEEDS),
        "exact_cohort": exact_cohort,
        "per_seed_passed": {str(seed): passed for seed, passed in seed_passes.items()},
        "passed": exact_cohort and all(seed_passes.values()),
    }


def _partition_record(source: PartitionSource) -> dict[str, object]:
    return {
        "name": source.contract.name,
        "selected_split": source.contract.dataset_config.split.value,
        "burn_in_steps": source.contract.burn_in_steps,
        "dataset_manifest_sha256": source.manifest_sha256,
        "dataset_manifest": source.contract.dataset_config.manifest_dict(),
    }


def _run_impl(
    *,
    output: Path,
    checkpoint: Path,
    uncalibrated_checkpoint: Path,
    model_seed: int,
    epochs: int,
    calibration_mode: str,
) -> None:
    if epochs != TRAINING_EPOCHS:
        raise ValueError("V2.1i requires the pinned 48-step/3-epoch schedule")
    if calibration_mode != CALIBRATION_MODE:
        raise ValueError("V2.1i requires the preregistered bias-only calibration")
    apply_deterministic_mode()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(model_seed)
    device = torch.device("cpu")

    contracts = development_partition_contracts()
    sources = {name: PartitionSource(contract) for name, contract in contracts.items()}
    config = model_config()
    project_root = Path(__file__).resolve().parents[2]
    source_bundle = _source_bundle(project_root)
    model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE).to(device)
    initialized_state_sha = _state_dict_sha256(model.state_dict())
    joint_training_record = train_joint_model(
        model,
        sources["TRAIN-FIT"],
        epochs=epochs,
    )
    if not _joint_training_schedule_is_exact(joint_training_record):
        raise RuntimeError("joint training record drifted from the frozen schedule")
    joint_trained_state_sha = _state_dict_sha256(model.state_dict())
    refinement_cache = collect_evidence(model, sources["TRAIN-FIT"])
    if _state_dict_sha256(model.state_dict()) != joint_trained_state_sha:
        raise RuntimeError("TRAIN-FIT cache collection mutated the joint model")
    hazard_refinement_record = refine_hazard_path(
        model,
        refinement_cache,
        seed=model_seed + 20_000,
    )
    trained_uncalibrated_state_sha = _state_dict_sha256(model.state_dict())
    training_record = {
        "joint_training": joint_training_record,
        "hazard_refinement": hazard_refinement_record,
    }

    # Regenerate evidence from the live refined recurrence.  The refinement
    # cache never comes from TRAIN-CAL, DEV, CPU-QUAL, or TEST.
    training_evidence = collect_evidence(model, sources["TRAIN-FIT"])
    calibration_evidence = collect_evidence(model, sources["TRAIN-CAL"])
    if _state_dict_sha256(model.state_dict()) != trained_uncalibrated_state_sha:
        raise RuntimeError("evidence collection mutated the refined parent state")
    uncalibrated_finite_state_audit = audit_model_finite(model)
    if uncalibrated_finite_state_audit["state_sha256"] != trained_uncalibrated_state_sha:
        raise RuntimeError("uncalibrated finite-state audit observed state drift")

    uncalibrated_payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "checkpoint_role": "upstream_joint_and_hazard_refined_uncalibrated_parent",
        "classification": "development_candidate_not_qualified",
        "device": "cpu",
        "threads": 1,
        "model_seed": model_seed,
        "config": asdict(config),
        "flags": asdict(CONFIG_B_PREDICTIVE),
        "model_state_dict": model.state_dict(),
        "dataset_partitions": {
            name: _partition_record(source) for name, source in sources.items()
        },
        "source_bundle": source_bundle,
        "training": training_record,
        "finite_state_audit": uncalibrated_finite_state_audit,
        "state_sha256": trained_uncalibrated_state_sha,
    }
    uncalibrated_checkpoint_sha = _publish_checkpoint_create_only(
        uncalibrated_checkpoint,
        uncalibrated_payload,
    )
    expected_calibration_bindings = calibration_expected_bindings(
        calibration_evidence,
        training_evidence,
        upstream_checkpoint_sha256=uncalibrated_checkpoint_sha,
        dataset_manifest_sha256=sources["TRAIN-CAL"].manifest_sha256,
        source_bundle_sha256=str(source_bundle["sha256"]),
    )

    calibrator = fit_calibrator(
        calibration_evidence,
        training_evidence,
        upstream_checkpoint_sha256=uncalibrated_checkpoint_sha,
        dataset_manifest_sha256=sources["TRAIN-CAL"].manifest_sha256,
        source_bundle_sha256=str(source_bundle["sha256"]),
        fit_mode=calibration_mode,
    )

    # Reconstruct the exact pre-optimizer state only after training/calibration
    # are frozen.  This supplies a true initialization comparison without DEV
    # ever entering an optimizer or calibration decision.
    torch.manual_seed(model_seed)
    initial_reference = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE).to(device)
    if _state_dict_sha256(initial_reference.state_dict()) != initialized_state_sha:
        raise RuntimeError("could not reconstruct the exact initialized model")
    initial_dev_evidence = collect_evidence(initial_reference, sources["DEV"])
    initial_dev_objective = evaluate_joint_objective(initial_reference, sources["DEV"])
    if _state_dict_sha256(initial_reference.state_dict()) != initialized_state_sha:
        raise RuntimeError("initial DEV evaluation mutated the reference model")

    # Measure the exact trained parent before installing calibration, then
    # prove both the DEV pass and the installation itself leave it immutable.
    parent_dev_evidence = collect_evidence(model, sources["DEV"])
    parent_dev_objective = evaluate_joint_objective(model, sources["DEV"])
    if _state_dict_sha256(model.state_dict()) != trained_uncalibrated_state_sha:
        raise RuntimeError("uncalibrated DEV evaluation mutated the parent model")
    calibration_install = install_calibrator(model, calibrator)
    installed_state_sha = _state_dict_sha256(model.state_dict())
    development_evidence = collect_evidence(model, sources["DEV"])
    dev_objective = evaluate_joint_objective(model, sources["DEV"])
    if _state_dict_sha256(model.state_dict()) != installed_state_sha:
        raise RuntimeError("calibrated DEV evaluation mutated the installed model")
    partition_disjointness = assert_evidence_partitions_disjoint(
        {
            "TRAIN-FIT": training_evidence,
            "TRAIN-CAL": calibration_evidence,
            "DEV": development_evidence,
        }
    )
    dev_replay_identity = assert_dev_replay_identity(
        initial_dev_evidence,
        parent_dev_evidence,
        development_evidence,
    )
    metrics = evaluate_development_metrics(
        training_evidence,
        calibration_evidence,
        development_evidence,
        calibrator,
        uncalibrated_development=parent_dev_evidence,
    )
    metrics["dev"]["installed_model_preservation"] = (
        installed_model_preservation_report(parent_dev_objective, dev_objective)
    )
    metrics["dev"]["learning_from_initialization"] = (
        learning_from_initialization_report(
            initial_dev_objective,
            dev_objective,
            initial_dev_evidence,
            development_evidence,
        )
    )
    state_immutability = {
        "initialized_reference_unchanged": True,
        "uncalibrated_parent_unchanged": True,
        "installed_model_unchanged": True,
    }
    metrics["dev"]["partition_disjointness"] = partition_disjointness
    metrics["dev"]["replay_identity"] = dev_replay_identity
    metrics["dev"]["state_immutability"] = state_immutability
    metrics["dev"]["hazard_refinement"] = hazard_refinement_record
    metrics["dev"]["joint_training"] = joint_training_record
    final_finite_state_audit = audit_model_finite(model)
    if final_finite_state_audit["state_sha256"] != installed_state_sha:
        raise RuntimeError("final finite-state audit observed state drift")
    metrics["dev"]["finite_state_audit"] = final_finite_state_audit
    current_source_bundle = _source_bundle(project_root)
    if current_source_bundle != source_bundle:
        raise RuntimeError("source bundle changed during the development run")
    development_gate = v21i_development_gate(
        calibrator.export_parameters(),
        metrics,
        calibration_install,
        expected_calibration_bindings,
    )
    checkpoint_payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "classification": "development_candidate_not_qualified",
        "device": "cpu",
        "threads": 1,
        "model_seed": model_seed,
        "config": asdict(config),
        "flags": asdict(CONFIG_B_PREDICTIVE),
        "model_state_dict": model.state_dict(),
        "hazard_calibration": calibrator.export_parameters(),
        "calibration_mode": calibration_mode,
        "upstream_uncalibrated_checkpoint": {
            "path": str(uncalibrated_checkpoint.resolve()),
            "sha256": uncalibrated_checkpoint_sha,
            "state_sha256": trained_uncalibrated_state_sha,
        },
        "dataset_partitions": {
            name: _partition_record(source) for name, source in sources.items()
        },
        "source_bundle": source_bundle,
        "training": training_record,
        "calibration_installation_audit": calibration_install,
        "finite_state_audit": final_finite_state_audit,
        "expected_calibration_bindings": expected_calibration_bindings,
        "development_gate": development_gate,
        "state_sha256": _state_dict_sha256(model.state_dict()),
    }
    checkpoint_record: dict[str, object] = {
        "path": str(checkpoint.resolve()),
        "published": False,
        "reason": "development gate failed",
    }
    if development_gate["passed"] is True:
        checkpoint_sha = _publish_checkpoint_create_only(checkpoint, checkpoint_payload)
        checkpoint_record = {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_sha,
            "published": True,
        }
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "classification": "development_candidate_not_qualified",
        "qualification_claimed": False,
        "namespace_policy": {
            "opened": ["TRAIN-FIT", "TRAIN-CAL", "DEV"],
            "test_split_opened": False,
            "cpu_qual_opened": False,
            "dev_used_for_training_or_calibration": False,
            "dev_used_for_development_gate": True,
        },
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "model_seed": model_seed,
        "epochs": epochs,
        "config": asdict(config),
        "dataset_partitions": {
            name: _partition_record(source) for name, source in sources.items()
        },
        "source_bundle": source_bundle,
        "training": training_record,
        "calibration": {
            "mode": calibration_mode,
            "sampling": "all_train_cal_roots_and_actions_once_no_replacement",
            "parameters": calibrator.export_parameters(),
            "installation_audit": calibration_install,
        },
        "state_sha256": {
            "initialized": initialized_state_sha,
            "joint_trained_before_hazard_refinement": joint_trained_state_sha,
            "hazard_refined_uncalibrated_parent": trained_uncalibrated_state_sha,
            "final_calibrated": _state_dict_sha256(model.state_dict()),
        },
        "evidence_counts": {
            "train_fit_roots": training_evidence.hazard_targets.shape[0],
            "train_cal_roots": calibration_evidence.hazard_targets.shape[0],
            "dev_roots": development_evidence.hazard_targets.shape[0],
            "dev_episodes": len(set(development_evidence.episode_group_ids)),
        },
        "partition_disjointness": partition_disjointness,
        "dev_replay_identity": dev_replay_identity,
        "dev_state_immutability": state_immutability,
        "initial_dev_joint_objective": initial_dev_objective,
        "uncalibrated_parent_dev_joint_objective": parent_dev_objective,
        "dev_joint_objective": dev_objective,
        "metrics": metrics,
        "expected_calibration_bindings": expected_calibration_bindings,
        "finite_state_audit": final_finite_state_audit,
        "development_gate": development_gate,
        "checkpoint": checkpoint_record,
        "upstream_uncalibrated_checkpoint": {
            "path": str(uncalibrated_checkpoint.resolve()),
            "sha256": uncalibrated_checkpoint_sha,
            "state_sha256": trained_uncalibrated_state_sha,
        },
    }
    _publish_json_create_only(output, result)
    if development_gate["passed"] is not True:
        print(f"FAILED development gate -> {output.resolve()}", flush=True)
        raise SystemExit(1)
    print(f"DONE development-only gate passed -> {output.resolve()}", flush=True)


def _failure_path_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "published": False}
    return {
        "path": str(resolved),
        "published": True,
        "sha256": sha256(resolved.read_bytes()).hexdigest(),
    }


def _run_with_failure_receipt(
    *,
    output: Path,
    checkpoint: Path,
    uncalibrated_checkpoint: Path,
    operation: Callable[[], None],
) -> None:
    """Persist canonical evidence for every post-preflight failure, then re-raise."""

    try:
        operation()
    except BaseException as error:
        if not output.exists():
            project_root = Path(__file__).resolve().parents[2]
            try:
                source_bundle: dict[str, object] | None = _source_bundle(project_root)
            except Exception:
                source_bundle = None
            failure: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "mode": MODE,
                "classification": "development_run_failed",
                "qualification_claimed": False,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "source_bundle_at_failure": source_bundle,
                "checkpoint": _failure_path_record(checkpoint),
                "upstream_uncalibrated_checkpoint": _failure_path_record(
                    uncalibrated_checkpoint
                ),
            }
            _publish_json_create_only(output, failure)
        raise


def run(
    *,
    output: Path,
    checkpoint: Path,
    uncalibrated_checkpoint: Path,
    model_seed: int,
    epochs: int,
    calibration_mode: str,
) -> None:
    def preflight_and_run() -> None:
        _assert_publication_targets_available(
            output,
            checkpoint,
            uncalibrated_checkpoint,
        )
        _run_impl(
            output=output,
            checkpoint=checkpoint,
            uncalibrated_checkpoint=uncalibrated_checkpoint,
            model_seed=model_seed,
            epochs=epochs,
            calibration_mode=calibration_mode,
        )

    _run_with_failure_receipt(
        output=output,
        checkpoint=checkpoint,
        uncalibrated_checkpoint=uncalibrated_checkpoint,
        operation=preflight_and_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--uncalibrated-checkpoint")
    parser.add_argument("--model-seed", type=int, default=44)
    parser.add_argument(
        "--epochs",
        type=int,
        choices=(TRAINING_EPOCHS,),
        default=TRAINING_EPOCHS,
        help="Pinned to 48 optimizer steps: 128 episodes / batch 8 * 3 epochs.",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=CALIBRATION_MODES,
        default=CALIBRATION_MODE,
        help="Pinned explicitly per run; bias_only is the preregistered first attempt.",
    )
    args = parser.parse_args()
    if args.model_seed < 0:
        parser.error("--model-seed must be nonnegative")
    output = Path(args.output).expanduser().resolve()
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else Path(f"{output}.checkpoint.pt")
    )
    uncalibrated_checkpoint = (
        Path(args.uncalibrated_checkpoint).expanduser().resolve()
        if args.uncalibrated_checkpoint
        else Path(f"{output}.uncalibrated.pt")
    )
    run(
        output=output,
        checkpoint=checkpoint,
        uncalibrated_checkpoint=uncalibrated_checkpoint,
        model_seed=args.model_seed,
        epochs=args.epochs,
        calibration_mode=args.calibration_mode,
    )


if __name__ == "__main__":
    main()
