"""Frozen-feature probes that isolate V2.1 hazard prediction bottlenecks."""
from __future__ import annotations

import argparse
import json

import torch
from torch import nn
import torch.nn.functional as F

from irene_brain.training.batches import MazeChaseBatchSource
from irene_brain.training.objective import _rgb_tensor
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Config, CoreV2Model
from irene_brain.v2.trajectory_objective import control_action_class, transition_hazard
from stage_v21_predictive_trajectory_smoke import DATASET_MANIFEST, dataset_config


@torch.no_grad()
def features(model: CoreV2Model, source: MazeChaseBatchSource, split: str):
    latents = []
    actions = []
    labels = []
    batch_size = getattr(source.config, split + "_sequences")
    for batch in source.iter_batches(
        split=split,
        epoch=0,
        start_batch=0,
        batch_size=batch_size,
    ):
        transitions = [
            transition
            for sequence in batch.sequences
            for transition in sequence.transitions[batch.burn_in_steps :]
        ]
        pixels = _rgb_tensor(
            tuple(transition.observation.rgb for transition in transitions),
            device=torch.device("cpu"),
            resolution=(32, 32),
        )
        latents.append(model.encoder(pixels))
        actions.append(
            F.one_hot(
                torch.tensor(
                    [control_action_class(t.applied_control) for t in transitions],
                    dtype=torch.long,
                ),
                num_classes=model.config.actions,
            ).float()
        )
        labels.append(
            torch.tensor(
                [[transition_hazard(t.event_targets)] for t in transitions],
                dtype=torch.float32,
            )
        )
    return torch.cat(latents), torch.cat(actions), torch.cat(labels)


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean())


def run_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    *,
    balanced: bool,
) -> dict[str, float | int]:
    torch.manual_seed(20260825)
    head = nn.Linear(train_x.shape[-1], 1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.02, weight_decay=1e-4)
    positives = float(train_y.sum())
    negatives = float(train_y.numel() - train_y.sum())
    positive_weight = negatives / positives if balanced else 1.0
    for _ in range(300):
        logits = head(train_x)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            train_y,
            pos_weight=torch.tensor([positive_weight]),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = head(validation_x)
        probabilities = torch.sigmoid(logits).flatten()
        flat_labels = validation_y.flatten()
        positive = probabilities[flat_labels == 1]
        negative = probabilities[flat_labels == 0]
        return {
            "train_positive_weight": round(positive_weight, 6),
            "validation_auc": round(auc(probabilities, flat_labels), 6),
            "validation_bce": round(
                float(F.binary_cross_entropy(probabilities, flat_labels)), 6
            ),
            "validation_positive_probability": round(float(positive.mean()), 6),
            "validation_negative_probability": round(float(negative.mean()), 6),
            "validation_probability_margin": round(
                float(positive.mean() - negative.mean()), 6
            ),
        }


def standardize(train: torch.Tensor, validation: torch.Tensor):
    mean = train.mean(dim=0, keepdim=True)
    scale = train.std(dim=0, keepdim=True).clamp_min(1e-5)
    return (train - mean) / scale, (validation - mean) / scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = CoreV2Model(CoreV2Config(**payload["config"]), CONFIG_B_PREDICTIVE)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    source = MazeChaseBatchSource(dataset_config(mini=False))
    if source.manifest_sha256 != DATASET_MANIFEST:
        raise RuntimeError("dataset manifest drift")
    train_latent, train_action, train_y = features(model, source, "train")
    validation_latent, validation_action, validation_y = features(
        model, source, "validation"
    )
    train_latent, validation_latent = standardize(train_latent, validation_latent)
    arms = {
        "latent_unweighted": (train_latent, validation_latent, False),
        "latent_action_unweighted": (
            torch.cat([train_latent, train_action], dim=-1),
            torch.cat([validation_latent, validation_action], dim=-1),
            False,
        ),
        "latent_action_balanced": (
            torch.cat([train_latent, train_action], dim=-1),
            torch.cat([validation_latent, validation_action], dim=-1),
            True,
        ),
        "action_only_balanced": (train_action, validation_action, True),
    }
    result = {
        "schema_version": 1,
        "dataset_manifest": source.manifest_sha256,
        "test_split_opened": False,
        "train_samples": int(train_y.numel()),
        "train_positives": int(train_y.sum()),
        "validation_samples": int(validation_y.numel()),
        "validation_positives": int(validation_y.sum()),
        "arms": {
            name: run_probe(train_x, train_y, validation_x, validation_y, balanced=balanced)
            for name, (train_x, validation_x, balanced) in arms.items()
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
