"""Measure whether V2.1 next-latent MSE is dominated by representation scale drift."""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from irene_brain.training.batches import MazeChaseBatchSource
from irene_brain.training.objective import _rgb_tensor
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Model
from irene_brain.v2.trajectory_objective import control_action_class, transition_hazard
from stage_v21_predictive_trajectory_smoke import (
    BATCH_SIZE,
    DATASET_MANIFEST,
    TRAIN_SEED,
    dataset_config,
    model_config,
)


@torch.no_grad()
def measure(model: CoreV2Model, source: MazeChaseBatchSource) -> dict[str, float | int]:
    model.eval()
    target_norms = []
    prediction_norms = []
    raw_mses = []
    normalized_mses = []
    cosines = []
    samples = 0
    for batch in source.iter_batches(
        split="validation",
        epoch=0,
        start_batch=0,
        batch_size=BATCH_SIZE,
    ):
        state = model.init_state(batch.batch_size, torch.device("cpu"))
        for tick in range(batch.sequence_length):
            transitions = tuple(sequence.transitions[tick] for sequence in batch.sequences)
            pixels = _rgb_tensor(
                tuple(transition.observation.rgb for transition in transitions),
                device=torch.device("cpu"),
                resolution=(32, 32),
            )
            next_pixels = _rgb_tensor(
                tuple(transition.next_observation_target.rgb for transition in transitions),
                device=torch.device("cpu"),
                resolution=(32, 32),
            )
            applied_actions = torch.tensor(
                [control_action_class(transition.applied_control) for transition in transitions],
                dtype=torch.long,
            )
            previous_actions = torch.tensor(
                [control_action_class(transition.observation.previous_control) for transition in transitions],
                dtype=torch.long,
            )
            prior_reward = None
            prior_hazard = None
            if tick:
                prior = tuple(sequence.transitions[tick - 1] for sequence in batch.sequences)
                prior_reward = torch.tensor(
                    [[transition.reward_target] for transition in prior], dtype=torch.float32
                )
                prior_hazard = torch.tensor(
                    [[transition_hazard(transition.event_targets)] for transition in prior],
                    dtype=torch.float32,
                )
            output, state = model(
                pixels,
                state,
                prev_action=previous_actions,
                actual_reward=prior_reward,
                actual_hazard=prior_hazard,
                world_model_action=applied_actions,
            )
            if tick < batch.burn_in_steps:
                continue
            target = model.world_model.compute_target(next_pixels)
            prediction = output.pending_prediction.predicted_next_latent.mean(dim=1)
            target_norms.extend(target.norm(dim=-1).tolist())
            prediction_norms.extend(prediction.norm(dim=-1).tolist())
            raw_mses.append(F.mse_loss(prediction, target).item())
            normalized_mses.append(
                F.mse_loss(F.normalize(prediction, dim=-1), F.normalize(target, dim=-1)).item()
            )
            cosines.append(F.cosine_similarity(prediction, target, dim=-1).mean().item())
            samples += batch.batch_size
    return {
        "samples": samples,
        "target_norm_mean": round(sum(target_norms) / len(target_norms), 6),
        "prediction_norm_mean": round(sum(prediction_norms) / len(prediction_norms), 6),
        "raw_mse_mean": round(sum(raw_mses) / len(raw_mses), 6),
        "normalized_mse_mean": round(sum(normalized_mses) / len(normalized_mses), 6),
        "cosine_mean": round(sum(cosines) / len(cosines), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = MazeChaseBatchSource(dataset_config(mini=False))
    if source.manifest_sha256 != DATASET_MANIFEST:
        raise RuntimeError("dataset manifest drift")
    torch.manual_seed(TRAIN_SEED)
    initial = CoreV2Model(model_config(True), CONFIG_B_PREDICTIVE)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    trained = CoreV2Model(model_config(True), CONFIG_B_PREDICTIVE)
    trained.load_state_dict(payload["model_state_dict"])
    print(
        json.dumps(
            {"initial": measure(initial, source), "trained": measure(trained, source)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
