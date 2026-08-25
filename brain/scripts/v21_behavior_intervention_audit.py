"""CPU-only coverage audit for V2.1 causal behavior interventions.

This diagnostic opens TRAIN and VALIDATION only. It measures whether every
applied action has both hazard and non-hazard outcomes after burn-in, which is
the minimum evidence needed before treating the hazard head as action-aware.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os

import torch

from irene_brain.training.batches import MazeChaseBatchConfig, MazeChaseBatchSource
from irene_brain.v2.trajectory_objective import control_action_class


def audit_split(source: MazeChaseBatchSource, split: str) -> dict[str, object]:
    contingency: Counter[tuple[int, int]] = Counter()
    teacher_actions: Counter[int] = Counter()
    applied_actions: Counter[int] = Counter()
    disagreements = 0
    samples = 0
    for batch in source.iter_batches(
        split=split,
        epoch=0,
        start_batch=0,
        batch_size=8,
    ):
        for sequence in batch.sequences:
            for transition in sequence.transitions[batch.burn_in_steps :]:
                applied = control_action_class(transition.applied_control)
                teacher = control_action_class(transition.action_target)
                hazard = int("caught" in transition.event_targets)
                contingency[applied, hazard] += 1
                teacher_actions[teacher] += 1
                applied_actions[applied] += 1
                disagreements += int(applied != teacher)
                samples += 1
    support = {
        str(action): {
            "safe": contingency[action, 0],
            "hazard": contingency[action, 1],
        }
        for action in range(5)
    }
    return {
        "samples": samples,
        "behavior_teacher_disagreements": disagreements,
        "behavior_teacher_disagreement_rate": disagreements / samples,
        "teacher_action_counts": {
            str(action): teacher_actions[action] for action in range(5)
        },
        "applied_action_counts": {
            str(action): applied_actions[action] for action in range(5)
        },
        "applied_action_hazard_support": support,
        "all_actions_have_both_outcomes": all(
            contingency[action, 0] > 0 and contingency[action, 1] > 0
            for action in range(5)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, action="append", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    results = []
    for rate in args.rate:
        config = MazeChaseBatchConfig(
            train_sequences=96,
            validation_sequences=32,
            test_sequences=32,
            sequence_length=16,
            burn_in_steps=4,
            seed_offset=8_388_608,
            ghost_count=5,
            ghost_period=1,
            ghost_rule="direct",
            behavior_policy="balanced_intervention_v1",
            behavior_intervention_rate=rate,
        )
        source = MazeChaseBatchSource(config)
        results.append(
            {
                "rate": rate,
                "dataset_manifest": source.manifest_sha256,
                "train": audit_split(source, "train"),
                "validation": audit_split(source, "validation"),
                "test_split_opened": False,
            }
        )
    payload = {"schema_version": 1, "results": results}
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = os.path.abspath(args.output)
        temporary = output + ".tmp"
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
