"""Bounded CPU audit of candidate V2.1 maze-chase hazard coverage."""
from __future__ import annotations

from irene_brain.training.batches import MazeChaseBatchConfig, MazeChaseBatchSource


def caught_count(source: MazeChaseBatchSource, split: str) -> int:
    return sum(
        "caught" in transition.event_targets
        for batch in source.iter_batches(
            split=split,
            epoch=0,
            start_batch=0,
            batch_size=8,
        )
        for sequence in batch.sequences
        for transition in sequence.transitions[batch.burn_in_steps :]
    )


def main() -> None:
    for ghost_count in (3, 5):
        for ghost_period in (1, 2):
            for ghost_rule in ("direct", "ambush", "mixed"):
                config = MazeChaseBatchConfig(
                    train_sequences=96,
                    validation_sequences=32,
                    test_sequences=32,
                    sequence_length=16,
                    burn_in_steps=4,
                    seed_offset=8_388_608,
                    ghost_count=ghost_count,
                    ghost_period=ghost_period,
                    ghost_rule=ghost_rule,
                )
                source = MazeChaseBatchSource(config)
                print(
                    ghost_count,
                    ghost_period,
                    ghost_rule,
                    caught_count(source, "train"),
                    caught_count(source, "validation"),
                    source.manifest_sha256[:12],
                )


if __name__ == "__main__":
    main()
