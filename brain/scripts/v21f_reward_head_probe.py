"""Bounded CPU screen for state-conditioned distributional reward learning.

This diagnostic never trains the checkpointed Core V2 model and never opens
the maze-chase TEST split.  It causally replays the registered 50% behavior-
intervention TRAIN and VALIDATION trajectories, caches frozen belief states,
and trains only a fresh ``VectorizedOutcomeModelV2`` reward path.
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


BEHAVIOR_POLICY = "balanced_intervention_v1"
BEHAVIOR_INTERVENTION_RATE = 0.5
CACHE_BATCH_SIZE = 8
SCREEN_MINIMUM_LEARNING_FRACTION = 0.10
SCREEN_MINIMUM_ACTION_BASELINE_FRACTION = 0.01
SCREEN_MINIMUM_SHUFFLE_PENALTY_FRACTION = 0.01


@dataclass(frozen=True)
class CachedRewardData:
    context: Tensor
    action: Tensor
    reward: Tensor

    def __post_init__(self) -> None:
        count = int(self.context.shape[0])
        if self.context.dim() != 2:
            raise ValueError("cached context must have shape [N, W]")
        if self.action.shape != (count,):
            raise ValueError("cached action must have shape [N]")
        if self.reward.shape != (count,):
            raise ValueError("cached reward must have shape [N]")

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
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_model(checkpoint: dict[str, object]) -> tuple[CoreV2Model, CoreV2Config]:
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
        raise ValueError("reward probe requires a CONFIG_B_PREDICTIVE checkpoint")
    model = CoreV2Model(config=config, flags=flags)
    model.load_state_dict(state_dict, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model, config


def _intervention_source(checkpoint: dict[str, object]) -> MazeChaseBatchSource:
    config = replace(
        dataset_config(mini=False),
        behavior_policy=BEHAVIOR_POLICY,
        behavior_intervention_rate=BEHAVIOR_INTERVENTION_RATE,
    )
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
) -> CachedRewardData:
    if split not in {"train", "validation"}:
        raise ValueError("reward probe may cache only TRAIN or VALIDATION")
    torch.manual_seed(rng_seed)
    contexts: list[Tensor] = []
    actions: list[Tensor] = []
    rewards: list[Tensor] = []
    device = torch.device("cpu")
    for batch in source.iter_batches(
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
            contexts.append(output.belief.detach().cpu().clone())
            actions.append(applied.detach().cpu().clone())
            rewards.append(
                torch.tensor(
                    [transition.reward_target for transition in transitions],
                    dtype=torch.float32,
                )
            )
    if not contexts:
        raise RuntimeError(f"{split} cache is empty")
    return CachedRewardData(
        context=torch.cat(contexts, dim=0).contiguous(),
        action=torch.cat(actions, dim=0).contiguous(),
        reward=torch.cat(rewards, dim=0).contiguous(),
    )


def _reward_parameters(model: VectorizedOutcomeModelV2) -> list[nn.Parameter]:
    prefixes = (
        "state_trunk.",
        "action_embedding.",
        "outcome_trunk.",
        "reward_head.",
    )
    selected: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith(prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append(parameter)
    if not selected:
        raise RuntimeError("reward probe selected no trainable parameters")
    return selected


def _factual_logits(
    model: VectorizedOutcomeModelV2,
    context: Tensor,
    action: Tensor,
) -> Tensor:
    factual = model(context).gather(action)
    if factual.predicted_reward_logits is None:
        raise RuntimeError("reward probe requires distributional reward logits")
    return factual.predicted_reward_logits


@torch.no_grad()
def _evaluate(
    model: VectorizedOutcomeModelV2,
    data: CachedRewardData,
    *,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    distribution = model.reward_distribution
    if distribution is None:
        raise RuntimeError("reward probe requires a two-hot reward distribution")
    ce_sum = 0.0
    mae_sum = 0.0
    predictions: list[Tensor] = []
    for start in range(0, data.samples, batch_size):
        stop = min(data.samples, start + batch_size)
        reward = data.reward[start:stop]
        logits = _factual_logits(
            model,
            data.context[start:stop],
            data.action[start:stop],
        )
        ce = distribution.loss(logits, reward, reduction="sum")
        predicted = distribution.decode(logits)
        ce_sum += float(ce)
        mae_sum += float((predicted - reward).abs().sum())
        predictions.append(predicted.cpu())
    prediction = torch.cat(predictions)
    per_action: dict[str, object] = {}
    for action_id in range(model.config.actions):
        mask = data.action.eq(action_id)
        if not bool(mask.any()):
            continue
        per_action[str(action_id)] = {
            "samples": int(mask.sum()),
            "target_mean": round(float(data.reward[mask].mean()), 6),
            "prediction_mean": round(float(prediction[mask].mean()), 6),
        }
    return {
        "samples": data.samples,
        "twohot_cross_entropy": round(ce_sum / data.samples, 6),
        "raw_reward_mae": round(mae_sum / data.samples, 6),
        "target_mean": round(float(data.reward.mean()), 6),
        "prediction_mean": round(float(prediction.mean()), 6),
        "prediction_min": round(float(prediction.min()), 6),
        "prediction_max": round(float(prediction.max()), 6),
        "per_action": per_action,
    }


@torch.no_grad()
def _action_only_baseline(
    model: VectorizedOutcomeModelV2,
    train: CachedRewardData,
    validation: CachedRewardData,
) -> dict[str, object]:
    distribution = model.reward_distribution
    if distribution is None:
        raise RuntimeError("reward probe requires a two-hot reward distribution")
    train_targets = distribution.targets(train.reward)
    probabilities = []
    for action_id in range(model.config.actions):
        mask = train.action.eq(action_id)
        if not bool(mask.any()):
            raise RuntimeError(f"TRAIN has no samples for action {action_id}")
        probability = train_targets[mask].mean(dim=0).clamp_min(1e-12)
        probabilities.append(probability / probability.sum())
    action_probabilities = torch.stack(probabilities)
    validation_probability = action_probabilities[validation.action]
    validation_target = distribution.targets(validation.reward)
    cross_entropy = -(
        validation_target * validation_probability.log()
    ).sum(dim=-1)
    raw_bins = distribution.raw_bins.to(dtype=validation_probability.dtype)
    prediction = (validation_probability * raw_bins).sum(dim=-1)
    return {
        "fit_split": "train",
        "evaluated_split": "validation",
        "samples": validation.samples,
        "twohot_cross_entropy": round(float(cross_entropy.mean()), 6),
        "raw_reward_mae": round(
            float((prediction - validation.reward).abs().mean()),
            6,
        ),
        "prediction_mean": round(float(prediction.mean()), 6),
    }


def _action_stratified_shuffle(
    data: CachedRewardData,
) -> CachedRewardData:
    shuffled = data.context.clone()
    for action_id in sorted(int(value) for value in data.action.unique()):
        indices = data.action.eq(action_id).nonzero(as_tuple=False).flatten()
        if indices.numel() > 1:
            shuffled[indices] = data.context[indices.roll(1)]
    return CachedRewardData(
        context=shuffled,
        action=data.action,
        reward=data.reward,
    )


def _train_reward_path(
    model: VectorizedOutcomeModelV2,
    data: CachedRewardData,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[dict[str, object]], int]:
    trainable = _reward_parameters(model)
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
        reward = data.reward[indices]
        logits = _factual_logits(model, data.context[indices], data.action[indices])
        distribution = model.reward_distribution
        if distribution is None:
            raise RuntimeError("reward probe requires a two-hot reward distribution")
        loss = distribution.loss(logits, reward)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite reward loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"non-finite reward gradient at step {step}")
        optimizer.step()
        if step == 1 or step == steps or step % cadence == 0:
            trace.append(
                {
                    "step": step,
                    "minibatch_twohot_cross_entropy": round(float(loss.detach()), 6),
                    "preclip_gradient_norm": round(float(gradient_norm), 6),
                }
            )
    return trace, sum(parameter.numel() for parameter in trainable)


def _finite_parameters(parameters: Iterable[nn.Parameter]) -> None:
    for parameter in parameters:
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError("reward probe produced a non-finite parameter")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    if not 1 <= args.steps <= 4096:
        parser.error("--steps must be in [1, 4096]")
    if not 1 <= args.batch_size <= 1024:
        parser.error("--batch-size must be in [1, 1024]")
    if not math.isfinite(args.learning_rate) or not 0.0 < args.learning_rate <= 1.0:
        parser.error("--learning-rate must be finite and in (0, 1]")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must be in [0, 2**63)")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    apply_deterministic_mode()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a mapping")
    core, checkpoint_config = _checkpoint_model(checkpoint)
    source = _intervention_source(checkpoint)
    train = _cache_split(
        core,
        source,
        split="train",
        rng_seed=args.seed + 1,
    )
    validation = _cache_split(
        core,
        source,
        split="validation",
        rng_seed=EVAL_SEED,
    )

    reward_config = replace(
        checkpoint_config,
        outcome_architecture="all_action_table_v1",
        outcome_action_conditioning="thought_only_v0",
        reward_prediction="symlog_twohot_v1",
    )
    torch.manual_seed(args.seed)
    reward_model = VectorizedOutcomeModelV2(reward_config)
    initial_validation = _evaluate(
        reward_model,
        validation,
        batch_size=args.batch_size,
    )
    action_only = _action_only_baseline(reward_model, train, validation)
    trace, trainable_parameters = _train_reward_path(
        reward_model,
        train,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed + 2,
    )
    _finite_parameters(reward_model.parameters())
    final_validation = _evaluate(
        reward_model,
        validation,
        batch_size=args.batch_size,
    )
    shuffled_validation = _evaluate(
        reward_model,
        _action_stratified_shuffle(validation),
        batch_size=args.batch_size,
    )

    initial_ce = float(initial_validation["twohot_cross_entropy"])
    final_ce = float(final_validation["twohot_cross_entropy"])
    baseline_ce = float(action_only["twohot_cross_entropy"])
    shuffled_ce = float(shuffled_validation["twohot_cross_entropy"])
    checks = {
        "learned_from_initialization": final_ce
        <= initial_ce * (1.0 - SCREEN_MINIMUM_LEARNING_FRACTION),
        "beats_train_fitted_action_only_baseline": final_ce
        <= baseline_ce * (1.0 - SCREEN_MINIMUM_ACTION_BASELINE_FRACTION),
        "state_shuffle_hurts_heldout_prediction": shuffled_ce
        >= final_ce * (1.0 + SCREEN_MINIMUM_SHUFFLE_PENALTY_FRACTION),
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "mode": "v21f_reward_head_probe",
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
            "behavior_policy": BEHAVIOR_POLICY,
            "behavior_intervention_rate": BEHAVIOR_INTERVENTION_RATE,
            "train_samples": train.samples,
            "validation_samples": validation.samples,
            "source_config": asdict(source.config),
        },
        "reward_model": {
            "architecture": "VectorizedOutcomeModelV2",
            "context": "frozen_checkpoint_belief",
            "prediction": reward_config.reward_prediction,
            "bins": reward_config.reward_bins,
            "symlog_min": reward_config.reward_symlog_min,
            "symlog_max": reward_config.reward_symlog_max,
            "trainable_parameters": trainable_parameters,
            "checkpoint_core_trainable_parameters": 0,
        },
        "optimization": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "gradient_clip_norm": 1.0,
            "trace": trace,
        },
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "action_only_baseline": action_only,
        "action_stratified_state_shuffle": shuffled_validation,
        "screen": {
            "thresholds": {
                "minimum_learning_fraction": SCREEN_MINIMUM_LEARNING_FRACTION,
                "minimum_ce_improvement_over_action_only_fraction": (
                    SCREEN_MINIMUM_ACTION_BASELINE_FRACTION
                ),
                "minimum_state_shuffle_ce_penalty_fraction": (
                    SCREEN_MINIMUM_SHUFFLE_PENALTY_FRACTION
                ),
            },
            "checks": checks,
            "passed": all(checks.values()),
        },
    }
    _write_atomic(output_path, payload)
    print(f"DONE -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
