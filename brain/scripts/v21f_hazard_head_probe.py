"""Bounded CPU screen for state-conditioned hazard learning.

This diagnostic never trains the checkpointed Core V2 model and never opens
the maze-chase TEST split.  It causally replays the registered 50% behavior-
intervention TRAIN and VALIDATION trajectories, caches frozen belief states,
and trains only a fresh ``VectorizedOutcomeModelV2`` hazard path.  Every
reported validation loss and calibration metric is unweighted, including for
the optional train-prevalence-weighted optimization arm.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

from irene_brain.training.batches import MazeChaseBatchSource
from irene_brain.training.objective import _rgb_tensor
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Config, CoreV2Model
from irene_brain.v2.config import FeatureFlags
from irene_brain.v2.outcome_model import VectorizedOutcomeModelV2
from irene_brain.v2.trajectory_objective import (
    control_action_class,
    transition_hazard,
)
from run_provenance import apply_deterministic_mode
from stage_v21_predictive_trajectory_smoke import EVAL_SEED, dataset_config
from v21_cpu_heldout_learning_diagnostic import _source_bundle


BEHAVIOR_POLICY = "balanced_intervention_v1"
BEHAVIOR_INTERVENTION_RATE = 0.5
CACHE_BATCH_SIZE = 8
TRAINING_ARMS = ("unweighted", "train_prevalence_pos_weight")
SCREEN_MINIMUM_LEARNING_FRACTION = 0.10
SCREEN_MINIMUM_ACTION_BASELINE_FRACTION = 0.01
SCREEN_MINIMUM_SHUFFLE_PENALTY_FRACTION = 0.01
SCREEN_MINIMUM_AUC_MARGIN = 0.01


@dataclass(frozen=True)
class CachedHazardData:
    context: Tensor
    action: Tensor
    hazard: Tensor

    def __post_init__(self) -> None:
        count = int(self.context.shape[0])
        if self.context.dim() != 2:
            raise ValueError("cached context must have shape [N, W]")
        if self.action.shape != (count,):
            raise ValueError("cached action must have shape [N]")
        if self.hazard.shape != (count,):
            raise ValueError("cached hazard must have shape [N]")
        if not bool(self.hazard.eq(0.0).logical_or(self.hazard.eq(1.0)).all()):
            raise ValueError("cached hazard labels must be binary")

    @property
    def samples(self) -> int:
        return int(self.context.shape[0])


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to replace existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard link publishes the fsynced payload atomically
        # and fails if another process created the requested artifact first.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_checkpoint_create_only(
    path: Path,
    payload: dict[str, object],
) -> str:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to replace checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_file(path)


def _assert_publication_targets_available(
    output: Path,
    calibrated_checkpoint: Path | None,
) -> None:
    """Refuse occupied or aliased publication targets before expensive work."""

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing artifact: {output}")
    if calibrated_checkpoint is None:
        return
    calibrated_checkpoint = calibrated_checkpoint.resolve()
    if calibrated_checkpoint == output:
        raise ValueError("output and calibrated checkpoint paths must be distinct")
    if calibrated_checkpoint.exists():
        raise FileExistsError(
            f"refusing to replace checkpoint: {calibrated_checkpoint}"
        )


def _require_current_parent_source_bundle(
    checkpoint: dict[str, object],
    project_root: Path,
) -> dict[str, object]:
    current = _source_bundle(project_root.resolve())
    if checkpoint.get("source_bundle") != current:
        raise ValueError(
            "parent checkpoint source bundle does not match current pipeline source"
        )
    return current


def _checkpoint_model(
    checkpoint: dict[str, object],
) -> tuple[CoreV2Model, CoreV2Config]:
    raw_config = checkpoint.get("config")
    raw_flags = checkpoint.get("flags")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint lacks a CoreV2 config")
    if not isinstance(raw_flags, dict):
        raise ValueError("checkpoint lacks CoreV2 feature flags")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint lacks model_state_dict")
    config = CoreV2Config(**raw_config)
    flags = FeatureFlags(**raw_flags)
    if flags != CONFIG_B_PREDICTIVE:
        raise ValueError("hazard probe requires a CONFIG_B_PREDICTIVE checkpoint")
    model = CoreV2Model(config=config, flags=flags)
    model.load_state_dict(state_dict, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model, config


def _registered_source(
    checkpoint: dict[str, object],
    *,
    target_mode: str,
) -> MazeChaseBatchSource:
    config = dataset_config(mini=False)
    if target_mode == "factual_intervention_v1":
        config = replace(
            config,
            behavior_policy=BEHAVIOR_POLICY,
            behavior_intervention_rate=BEHAVIOR_INTERVENTION_RATE,
        )
    elif target_mode == "all_actions_v1":
        config = replace(config, counterfactual_targets="all_actions_v1")
    else:
        raise ValueError(f"unknown hazard target mode: {target_mode}")
    source = MazeChaseBatchSource(config)
    registered_manifest = checkpoint.get("dataset_manifest")
    if not isinstance(registered_manifest, str) or len(registered_manifest) != 64:
        raise ValueError("checkpoint lacks a registered dataset manifest")
    if source.manifest_sha256 != registered_manifest:
        raise ValueError(
            "checkpoint dataset does not match the 50% intervention source: "
            f"{registered_manifest} != {source.manifest_sha256}"
        )
    return source


@torch.no_grad()
def _cache_split(
    core: CoreV2Model,
    source: MazeChaseBatchSource,
    *,
    split: str,
    rng_seed: int,
    target_mode: str = "factual_intervention_v1",
) -> CachedHazardData:
    if split not in {"train", "validation"}:
        raise ValueError("hazard probe may cache only TRAIN or VALIDATION")
    torch.manual_seed(rng_seed)
    contexts: list[Tensor] = []
    actions: list[Tensor] = []
    hazards: list[Tensor] = []
    device = torch.device("cpu")
    iterator = (
        source.iter_all_action_batches
        if target_mode == "all_actions_v1"
        else source.iter_batches
    )
    for batch in iterator(
        split=split,
        epoch=0,
        start_batch=0,
        batch_size=CACHE_BATCH_SIZE,
    ):
        state = core.init_state(batch.batch_size, device)
        for tick in range(batch.sequence_length):
            transitions = tuple(
                sequence.transitions[tick] for sequence in batch.sequences
            )
            pixels = _rgb_tensor(
                tuple(transition.observation.rgb for transition in transitions),
                device=device,
                resolution=(32, 32),
            )
            applied = torch.tensor(
                [
                    control_action_class(transition.applied_control)
                    for transition in transitions
                ],
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
            prior_reward = None
            prior_hazard = None
            if tick:
                prior = tuple(
                    sequence.transitions[tick - 1] for sequence in batch.sequences
                )
                prior_reward = torch.tensor(
                    [[transition.reward_target] for transition in prior],
                    dtype=torch.float32,
                    device=device,
                )
                prior_hazard = torch.tensor(
                    [
                        [transition_hazard(transition.event_targets)]
                        for transition in prior
                    ],
                    dtype=torch.float32,
                    device=device,
                )
            output, state = core(
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
            belief = output.belief.detach().cpu().clone()
            if target_mode == "all_actions_v1":
                action_count = core.config.actions
                if any(
                    not hasattr(transition, "counterfactual_targets")
                    or len(transition.counterfactual_targets) != action_count
                    for transition in transitions
                ):
                    raise RuntimeError("all-action batch lacks exhaustive targets")
                contexts.append(
                    belief.unsqueeze(1)
                    .expand(-1, action_count, -1)
                    .reshape(-1, belief.shape[-1])
                )
                actions.append(
                    torch.arange(action_count, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(batch.batch_size, -1)
                    .reshape(-1)
                )
                hazards.append(
                    torch.tensor(
                        [
                            transition_hazard(target.event_targets)
                            for transition in transitions
                            for target in transition.counterfactual_targets
                        ],
                        dtype=torch.float32,
                    )
                )
            else:
                contexts.append(belief)
                actions.append(applied.detach().cpu().clone())
                hazards.append(
                    torch.tensor(
                        [
                            transition_hazard(transition.event_targets)
                            for transition in transitions
                        ],
                        dtype=torch.float32,
                    )
                )
    if not contexts:
        raise RuntimeError(f"{split} cache is empty")
    return CachedHazardData(
        context=torch.cat(contexts, dim=0).contiguous(),
        action=torch.cat(actions, dim=0).contiguous(),
        hazard=torch.cat(hazards, dim=0).contiguous(),
    )


def _hazard_parameters(
    model: VectorizedOutcomeModelV2,
    *,
    training_scope: str = "all_hazard_path",
) -> list[nn.Parameter]:
    if training_scope == "all_hazard_path":
        prefixes = (
            (
                "hazard_state_trunk.",
                "hazard_action_embedding.",
                "hazard_outcome_trunk.",
                "hazard_head.",
            )
            if model.config.hazard_outcome_path == "dedicated_stopgrad_v1"
            else (
                "state_trunk.",
                "action_embedding.",
                "outcome_trunk.",
                "hazard_head.",
            )
        )
    elif training_scope == "head_only":
        prefixes = ("hazard_head.",)
    else:
        raise ValueError(f"unknown hazard training scope: {training_scope}")
    selected: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith(prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    if not selected:
        raise RuntimeError("hazard probe selected no trainable parameters")
    return selected


def _factual_probability(
    model: VectorizedOutcomeModelV2,
    context: Tensor,
    action: Tensor,
) -> Tensor:
    return model(context).gather(action).predicted_hazard.squeeze(-1)


def _binary_cross_entropy(
    probability: Tensor,
    target: Tensor,
    *,
    positive_weight: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be mean, sum, or none")
    epsilon = torch.finfo(probability.dtype).eps
    probability = probability.clamp(min=epsilon, max=1.0 - epsilon)
    losses = -(
        positive_weight * target * probability.log()
        + (1.0 - target) * (1.0 - probability).log()
    )
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    return losses


def _roc_auc(probability: Tensor, target: Tensor) -> float | None:
    positive = probability[target.eq(1.0)]
    negative = probability[target.eq(0.0)]
    if not positive.numel() or not negative.numel():
        return None
    comparisons = positive.unsqueeze(1) - negative.unsqueeze(0)
    auc = comparisons.gt(0.0).to(torch.float64).mean()
    auc += 0.5 * comparisons.eq(0.0).to(torch.float64).mean()
    return float(auc)


def _pr_auc(probability: Tensor, target: Tensor) -> float | None:
    positives = int(target.sum())
    if positives == 0:
        return None
    order = torch.argsort(probability, descending=True, stable=True)
    score = probability[order]
    label = target[order].to(torch.float64)
    true_positive = label.cumsum(dim=0)
    false_positive = (1.0 - label).cumsum(dim=0)
    threshold_end = torch.ones(score.shape[0], dtype=torch.bool)
    threshold_end[:-1] = score[:-1].ne(score[1:])
    true_positive = true_positive[threshold_end]
    false_positive = false_positive[threshold_end]
    recall = true_positive / positives
    precision = true_positive / (true_positive + false_positive)
    previous_recall = torch.cat(
        (torch.zeros(1, dtype=recall.dtype), recall[:-1])
    )
    return float(((recall - previous_recall) * precision).sum())


def _rounded_metric(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _metrics(probability: Tensor, data: CachedHazardData) -> dict[str, object]:
    probability = probability.detach().cpu().to(torch.float32)
    target = data.hazard
    positive = target.eq(1.0)
    negative = target.eq(0.0)
    positives = int(positive.sum())
    negatives = int(negative.sum())
    per_action: dict[str, object] = {}
    for action_id in sorted(int(value) for value in data.action.unique()):
        mask = data.action.eq(action_id)
        action_probability = probability[mask]
        action_target = target[mask]
        action_positives = int(action_target.sum())
        action_negatives = int(action_target.numel() - action_positives)
        action_positive_mask = action_target.eq(1.0)
        action_negative_mask = action_target.eq(0.0)
        per_action[str(action_id)] = {
            "samples": int(mask.sum()),
            "positives": action_positives,
            "negatives": action_negatives,
            "prevalence": round(float(action_target.mean()), 6),
            "unweighted_bce": round(
                float(_binary_cross_entropy(action_probability, action_target)),
                6,
            ),
            "brier": round(
                float((action_probability - action_target).square().mean()),
                6,
            ),
            "roc_auc": _rounded_metric(
                _roc_auc(action_probability, action_target)
            ),
            "pr_auc": _rounded_metric(_pr_auc(action_probability, action_target)),
            "positive_probability_mean": (
                round(float(action_probability[action_positive_mask].mean()), 6)
                if action_positives
                else None
            ),
            "negative_probability_mean": (
                round(float(action_probability[action_negative_mask].mean()), 6)
                if action_negatives
                else None
            ),
        }
    return {
        "samples": data.samples,
        "positives": positives,
        "negatives": negatives,
        "prevalence": round(float(target.mean()), 6),
        "unweighted_bce": round(
            float(_binary_cross_entropy(probability, target)),
            6,
        ),
        "brier": round(float((probability - target).square().mean()), 6),
        "roc_auc": _rounded_metric(_roc_auc(probability, target)),
        "pr_auc": _rounded_metric(_pr_auc(probability, target)),
        "positive_probability_mean": (
            round(float(probability[positive].mean()), 6) if positives else None
        ),
        "negative_probability_mean": (
            round(float(probability[negative].mean()), 6) if negatives else None
        ),
        "prediction_mean": round(float(probability.mean()), 6),
        "prediction_min": round(float(probability.min()), 6),
        "prediction_max": round(float(probability.max()), 6),
        "per_action": per_action,
    }


@torch.no_grad()
def _evaluate(
    model: VectorizedOutcomeModelV2,
    data: CachedHazardData,
    *,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    predictions: list[Tensor] = []
    for start in range(0, data.samples, batch_size):
        stop = min(data.samples, start + batch_size)
        predictions.append(
            _factual_probability(
                model,
                data.context[start:stop],
                data.action[start:stop],
            ).cpu()
        )
    return _metrics(torch.cat(predictions), data)


@torch.no_grad()
def _action_only_baseline(
    train: CachedHazardData,
    validation: CachedHazardData,
    *,
    actions: int,
) -> dict[str, object]:
    action_probability = torch.empty(actions, dtype=torch.float32)
    fit_support: dict[str, object] = {}
    for action_id in range(actions):
        mask = train.action.eq(action_id)
        if not bool(mask.any()):
            raise RuntimeError(f"TRAIN has no samples for action {action_id}")
        targets = train.hazard[mask]
        positives = int(targets.sum())
        negatives = int(targets.numel() - positives)
        if positives == 0 or negatives == 0:
            raise RuntimeError(
                f"TRAIN action {action_id} lacks both hazard outcomes"
            )
        action_probability[action_id] = targets.mean()
        fit_support[str(action_id)] = {
            "samples": int(mask.sum()),
            "positives": positives,
            "negatives": negatives,
            "fitted_probability": round(float(targets.mean()), 6),
        }
    result = _metrics(action_probability[validation.action], validation)
    result.update(
        {
            "fit_split": "train",
            "evaluated_split": "validation",
            "fit_support": fit_support,
        }
    )
    return result


def _action_stratified_shuffle(data: CachedHazardData) -> CachedHazardData:
    shuffled = data.context.clone()
    for action_id in sorted(int(value) for value in data.action.unique()):
        indices = data.action.eq(action_id).nonzero(as_tuple=False).flatten()
        if indices.numel() > 1:
            shuffled[indices] = data.context[indices.roll(1)]
    return CachedHazardData(
        context=shuffled,
        action=data.action,
        hazard=data.hazard,
    )


def _positive_weight(data: CachedHazardData) -> float:
    positives = float(data.hazard.sum())
    negatives = float(data.samples - positives)
    if positives <= 0.0 or negatives <= 0.0:
        raise RuntimeError("TRAIN must contain both hazard classes")
    return negatives / positives


def _train_hazard_path(
    model: VectorizedOutcomeModelV2,
    data: CachedHazardData,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    positive_weight: float,
    training_scope: str = "all_hazard_path",
) -> tuple[list[dict[str, object]], int]:
    trainable = _hazard_parameters(model, training_scope=training_scope)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    trace: list[dict[str, object]] = []
    cadence = max(1, steps // 8)
    model.train()
    for step in range(1, steps + 1):
        indices = torch.randint(
            data.samples,
            (min(batch_size, data.samples),),
            generator=generator,
        )
        target = data.hazard[indices]
        probability = _factual_probability(
            model,
            data.context[indices],
            data.action[indices],
        )
        loss = _binary_cross_entropy(
            probability,
            target,
            positive_weight=positive_weight,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite hazard loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"non-finite hazard gradient at step {step}")
        optimizer.step()
        if step == 1 or step == steps or step % cadence == 0:
            trace.append(
                {
                    "step": step,
                    "minibatch_training_bce": round(float(loss.detach()), 6),
                    "training_positive_weight": round(positive_weight, 6),
                    "preclip_gradient_norm": round(float(gradient_norm), 6),
                }
            )
    return trace, sum(parameter.numel() for parameter in trainable)


def _finite_parameters(parameters: Iterable[nn.Parameter]) -> None:
    for parameter in parameters:
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError("hazard probe produced a non-finite parameter")


def _require_metric(metrics: dict[str, object], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, (float, int)):
        raise RuntimeError(f"hazard screen requires numeric metric {name}")
    return float(value)


def _screen(
    initial: dict[str, object],
    final: dict[str, object],
    action_only: dict[str, object],
    shuffled: dict[str, object],
) -> dict[str, object]:
    initial_bce = _require_metric(initial, "unweighted_bce")
    final_bce = _require_metric(final, "unweighted_bce")
    baseline_bce = _require_metric(action_only, "unweighted_bce")
    shuffled_bce = _require_metric(shuffled, "unweighted_bce")
    final_auc = _require_metric(final, "roc_auc")
    baseline_auc = _require_metric(action_only, "roc_auc")
    shuffled_auc = _require_metric(shuffled, "roc_auc")
    checks = {
        "learned_from_initialization": final_bce
        <= initial_bce * (1.0 - SCREEN_MINIMUM_LEARNING_FRACTION),
        "beats_train_fitted_action_only_baseline": final_bce
        <= baseline_bce * (1.0 - SCREEN_MINIMUM_ACTION_BASELINE_FRACTION),
        "state_shuffle_hurts_heldout_bce": shuffled_bce
        >= final_bce * (1.0 + SCREEN_MINIMUM_SHUFFLE_PENALTY_FRACTION),
        "roc_auc_beats_action_only_baseline": final_auc
        >= baseline_auc + SCREEN_MINIMUM_AUC_MARGIN,
        "state_shuffle_hurts_roc_auc": shuffled_auc
        <= final_auc - SCREEN_MINIMUM_AUC_MARGIN,
    }
    return {
        "thresholds": {
            "minimum_learning_fraction": SCREEN_MINIMUM_LEARNING_FRACTION,
            "minimum_bce_improvement_over_action_only_fraction": (
                SCREEN_MINIMUM_ACTION_BASELINE_FRACTION
            ),
            "minimum_state_shuffle_bce_penalty_fraction": (
                SCREEN_MINIMUM_SHUFFLE_PENALTY_FRACTION
            ),
            "minimum_roc_auc_margin": SCREEN_MINIMUM_AUC_MARGIN,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--target-mode",
        choices=("factual_intervention_v1", "all_actions_v1"),
        default="factual_intervention_v1",
    )
    parser.add_argument(
        "--training-scope",
        choices=("all_hazard_path", "head_only"),
        default="all_hazard_path",
    )
    parser.add_argument(
        "--initialization",
        choices=("fresh", "checkpoint_outcome"),
        default="fresh",
    )
    parser.add_argument("--calibrated-checkpoint")
    parser.add_argument(
        "--training-arms",
        nargs="+",
        choices=TRAINING_ARMS,
        default=list(TRAINING_ARMS),
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= 4096:
        parser.error("--steps must be in [1, 4096]")
    if not 1 <= args.batch_size <= 1024:
        parser.error("--batch-size must be in [1, 1024]")
    if not math.isfinite(args.learning_rate) or not 0.0 < args.learning_rate <= 1.0:
        parser.error("--learning-rate must be finite and in (0, 1]")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be in [0, 2**63)")
    if len(set(args.training_arms)) != len(args.training_arms):
        parser.error("--training-arms must not contain duplicates")
    if args.calibrated_checkpoint and (
        args.training_arms != ["unweighted"]
        or args.initialization != "checkpoint_outcome"
        or args.training_scope != "all_hazard_path"
    ):
        parser.error(
            "--calibrated-checkpoint requires the unweighted arm, "
            "checkpoint_outcome initialization, and all_hazard_path scope"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    apply_deterministic_mode()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    calibrated_checkpoint_path = (
        Path(args.calibrated_checkpoint).expanduser().resolve()
        if args.calibrated_checkpoint
        else None
    )
    _assert_publication_targets_available(
        output_path,
        calibrated_checkpoint_path,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a mapping")
    if calibrated_checkpoint_path is not None:
        _require_current_parent_source_bundle(
            checkpoint,
            Path(__file__).resolve().parents[2],
        )
    core, checkpoint_config = _checkpoint_model(checkpoint)
    source = _registered_source(checkpoint, target_mode=args.target_mode)
    train = _cache_split(
        core,
        source,
        split="train",
        rng_seed=args.seed + 1,
        target_mode=args.target_mode,
    )
    validation = _cache_split(
        core,
        source,
        split="validation",
        rng_seed=EVAL_SEED,
        target_mode=args.target_mode,
    )

    hazard_config = replace(
        checkpoint_config,
        outcome_architecture="all_action_table_v1",
        outcome_action_conditioning="thought_only_v0",
        hazard_parameterization="probability_sigmoid_v1",
    )
    torch.manual_seed(args.seed)
    template = VectorizedOutcomeModelV2(hazard_config)
    if args.initialization == "checkpoint_outcome":
        checkpoint_outcome = core.world_model.outcome_model
        if checkpoint_outcome is None:
            raise ValueError("checkpoint has no all-action outcome model")
        template.load_state_dict(checkpoint_outcome.state_dict(), strict=True)
    template_state = {
        name: value.detach().clone() for name, value in template.state_dict().items()
    }
    action_only = _action_only_baseline(
        train,
        validation,
        actions=hazard_config.actions,
    )
    train_positive_weight = _positive_weight(train)
    shuffled_validation_data = _action_stratified_shuffle(validation)
    arm_results: dict[str, object] = {}
    calibrated_outcome: VectorizedOutcomeModelV2 | None = None
    for arm_name in args.training_arms:
        model = VectorizedOutcomeModelV2(hazard_config)
        model.load_state_dict(template_state, strict=True)
        initial_validation = _evaluate(
            model,
            validation,
            batch_size=args.batch_size,
        )
        positive_weight = (
            1.0
            if arm_name == "unweighted"
            else train_positive_weight
        )
        trace, trainable_parameters = _train_hazard_path(
            model,
            train,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed + 2,
            positive_weight=positive_weight,
            training_scope=args.training_scope,
        )
        _finite_parameters(model.parameters())
        final_validation = _evaluate(
            model,
            validation,
            batch_size=args.batch_size,
        )
        shuffled_validation = _evaluate(
            model,
            shuffled_validation_data,
            batch_size=args.batch_size,
        )
        arm_results[arm_name] = {
            "training_positive_weight": round(positive_weight, 6),
            "evaluation_loss_weighting": "unweighted",
            "trainable_parameters": trainable_parameters,
            "optimization_trace": trace,
            "initial_validation": initial_validation,
            "final_validation": final_validation,
            "action_stratified_state_shuffle": shuffled_validation,
            "screen": _screen(
                initial_validation,
                final_validation,
                action_only,
                shuffled_validation,
            ),
        }
        if arm_name == "unweighted":
            calibrated_outcome = model

    calibration_screen = arm_results.get("unweighted", {}).get("screen")
    publication_refused = bool(
        args.calibrated_checkpoint
        and (
            not isinstance(calibration_screen, dict)
            or calibration_screen.get("passed") is not True
        )
    )
    calibrated_checkpoint_record: dict[str, object] | None = None
    if args.calibrated_checkpoint and not publication_refused:
        if calibrated_outcome is None:
            raise RuntimeError("calibrated outcome model is unavailable")
        if checkpoint_config.hazard_outcome_path != "dedicated_stopgrad_v1":
            raise ValueError(
                "checkpoint publication requires dedicated_stopgrad_v1"
            )
        before = {
            name: value.detach().cpu().clone()
            for name, value in core.state_dict().items()
        }
        core.world_model.outcome_model.load_state_dict(
            calibrated_outcome.state_dict(),
            strict=True,
        )
        after = {
            name: value.detach().cpu().clone()
            for name, value in core.state_dict().items()
        }
        allowed_prefixes = (
            "world_model.outcome_model.hazard_state_trunk.",
            "world_model.outcome_model.hazard_action_embedding.",
            "world_model.outcome_model.hazard_outcome_trunk.",
            "world_model.outcome_model.hazard_head.",
        )
        changed = tuple(
            name for name in before if not torch.equal(before[name], after[name])
        )
        if not changed or any(
            not name.startswith(allowed_prefixes) for name in changed
        ):
            raise RuntimeError(
                "hazard calibration changed a non-isolated checkpoint tensor"
            )
        calibrated_payload = dict(checkpoint)
        calibrated_payload["model_state_dict"] = after
        calibrated_payload["hazard_calibration"] = {
            "schema_version": 1,
            "parent_checkpoint_sha256": _sha256_file(checkpoint_path),
            "dataset_manifest_sha256": source.manifest_sha256,
            "target_mode": args.target_mode,
            "training_scope": args.training_scope,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "changed_state_keys": list(changed),
            "non_hazard_tensors_byte_equal": True,
            "calibration_screen_passed": True,
            "source_bundle_sha256": (
                checkpoint.get("source_bundle", {}).get("sha256")
                if isinstance(checkpoint.get("source_bundle"), dict)
                else None
            ),
        }
        if calibrated_checkpoint_path is None:
            raise RuntimeError("calibrated checkpoint path is unavailable")
        calibrated_path = calibrated_checkpoint_path
        calibrated_sha = _save_checkpoint_create_only(
            calibrated_path,
            calibrated_payload,
        )
        calibrated_checkpoint_record = {
            "status": "published",
            "path": str(calibrated_path),
            "sha256": calibrated_sha,
            "parent_sha256": _sha256_file(checkpoint_path),
            "changed_state_keys": list(changed),
            "non_hazard_tensors_byte_equal": True,
        }
    elif publication_refused:
        calibrated_checkpoint_record = {
            "status": "refused_screen_failed",
            "requested_path": str(
                calibrated_checkpoint_path
            ),
            "screen": calibration_screen,
        }

    payload: dict[str, object] = {
        "schema_version": 1,
        "mode": "v21f_hazard_head_probe",
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "test_split_opened": False,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "arm": checkpoint.get("arm"),
            "seed": checkpoint.get("seed"),
        },
        "dataset": {
            "manifest_sha256": source.manifest_sha256,
            "behavior_policy": source.config.behavior_policy,
            "behavior_intervention_rate": source.config.behavior_intervention_rate,
            "target_mode": args.target_mode,
            "train_samples": train.samples,
            "validation_samples": validation.samples,
            "train_positives": int(train.hazard.sum()),
            "validation_positives": int(validation.hazard.sum()),
            "source_config": asdict(source.config),
        },
        "hazard_model": {
            "architecture": "VectorizedOutcomeModelV2",
            "context": "frozen_checkpoint_belief",
            "action_conditioning": "all_action_table_factual_gather",
            "prediction": hazard_config.hazard_parameterization,
            "checkpoint_core_trainable_parameters": 0,
        },
        "optimization": {
            "steps_per_arm": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "gradient_clip_norm": 1.0,
            "training_arms": args.training_arms,
            "training_scope": args.training_scope,
            "initialization": args.initialization,
            "train_prevalence_positive_weight": round(train_positive_weight, 6),
        },
        "action_only_baseline": action_only,
        "arms": arm_results,
        "calibrated_checkpoint": calibrated_checkpoint_record,
    }
    _write_atomic(output_path, payload)
    if publication_refused:
        print(f"FAIL -> {output_path}", flush=True)
        raise SystemExit(1)
    print(f"DONE -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
