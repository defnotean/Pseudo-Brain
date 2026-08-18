"""Bounded CPU smoke comparison across the matched baseline suite.

Exploratory, CPU-only, single-threaded *verification probe* — not a
training run and not a qualified result. Every registered architecture
variant is rebuilt at smoke scale with identical seeds, trained on the
same deterministic moving_shapes batches for a bounded number of
optimizer steps, and scored on the same held-out validation sequences.
The campaign window shape (sequence_length 8, burn-in 2) is used so the
B1 multi-horizon world loss is exercised by every variant.

Per-variant JSON rows are written incrementally, so a partial run can be
resumed by re-invoking with the remaining variant ids. Variants in their
own recipe family (the B2 world-model actor) are trained through the
objective their model declares on its fail-closed
``training_objective_class_path`` hook — the same resolution discipline
as train.py — so the probe never silently swaps objectives. This probe
exists to give the matched-baseline decision an early, cheap signal —
the qualified comparison remains the DGX campaign.

Usage (from the repository root, play-safe Python):

    python brain/scripts/compare_baselines_smoke.py --steps 64
    python brain/scripts/compare_baselines_smoke.py --steps 64 --variants irene.thought_field.routed.v1
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

BRAIN_ROOT = Path(__file__).resolve().parents[1]
SRC = BRAIN_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SEED = 20260818
TRAIN_SEQUENCES = 16
VALIDATION_SEQUENCES = 4
SEQUENCE_LENGTH = 8
BURN_IN_STEPS = 2
DEFAULT_OUTPUT_DIR = BRAIN_ROOT / "docs" / "runs" / "artifacts" / "baseline-smoke-compare"


def _slot_config():
    from irene_brain.model.spec import ThoughtFieldConfig

    return replace(
        ThoughtFieldConfig.smoke(),
        core_width=16,
        sensor_tokens=4,
        belief_tokens=2,
        working_memory_tokens=1,
        thoughtlets=4,
        registers_per_thoughtlet=3,
        goal_context_tokens=1,
        cognitive_cycles=2,
        brain_cell_blocks=1,
        attention_heads=2,
        routed_neighbors=1,
        episodic_memory_entries=8,
        retrieved_entries_per_thoughtlet=1,
    )


def _monolithic_config(*, width: int = 16):
    return replace(
        _slot_config(),
        core_width=width,
        thoughtlets=1,
        registers_per_thoughtlet=1,
        routed_neighbors=0,
    )


def _build_variant(variant_id: str):
    from irene_brain.model.baselines import (
        DenseCommunicationSlotBaseline,
        FixedMultiHorizonSlotBaseline,
        MatchedEnsembleBaseline,
        MonolithicRecurrentBaseline,
        NoCommunicationSlotBaseline,
        ParameterMatchedMonolithicBaseline,
        ReactiveSlotBaseline,
        RecurrentTransformerBaseline,
        ResetStateSlotBaseline,
        SerialDepthSlotBaseline,
    )
    from irene_brain.model.torch_model import IreneBrainModel
    from irene_brain.model.world_model_actor import LatentWorldModelActor

    slot = _slot_config()
    builders = {
        "irene.thought_field.routed.v1": lambda: IreneBrainModel(
            slot, input_resolution=(8, 8), plan_steps=2
        ),
        "irene.thought_field.isolated_slots.v1": lambda: NoCommunicationSlotBaseline(
            slot, input_resolution=(8, 8), plan_steps=2
        ),
        "irene.thought_field.reset_slots.v1": lambda: ResetStateSlotBaseline(
            slot, input_resolution=(8, 8), plan_steps=2
        ),
        "irene.thought_field.dense_routing.v1": lambda: DenseCommunicationSlotBaseline(
            slot, input_resolution=(8, 8), plan_steps=2
        ),
        "irene.thought_field.reactive.v1": lambda: ReactiveSlotBaseline(
            slot, input_resolution=(8, 8), plan_steps=2
        ),
        "irene.thought_field.serial_depth.v1": lambda: SerialDepthSlotBaseline(
            replace(slot, cognitive_cycles=1, brain_cell_blocks=2),
            input_resolution=(8, 8),
            plan_steps=2,
        ),
        "irene.thought_field.independent_ensemble.v1": lambda: MatchedEnsembleBaseline(
            slot, input_resolution=(8, 8), plan_steps=2
        ),
        "irene.thought_field.fixed_multi_horizon.v1": lambda: FixedMultiHorizonSlotBaseline(
            slot, input_resolution=(8, 8), plan_steps=2
        ),
        "irene.monolithic_gru.same_width.v1": lambda: MonolithicRecurrentBaseline(
            _monolithic_config(), input_resolution=(8, 8), plan_steps=2
        ),
        "irene.monolithic_gru.parameter_matched.v1": lambda: ParameterMatchedMonolithicBaseline(
            _monolithic_config(width=18), input_resolution=(8, 8), plan_steps=2
        ),
        "irene.recurrent_transformer.carry_token.v1": lambda: RecurrentTransformerBaseline(
            _monolithic_config(), input_resolution=(8, 8), plan_steps=2
        ),
        "irene.world_model_actor.gru_latent.v1": lambda: LatentWorldModelActor(
            _monolithic_config(width=18), input_resolution=(8, 8), plan_steps=2
        ),
    }
    if variant_id not in builders:
        raise ValueError(f"unknown variant id: {variant_id}")
    return builders[variant_id]()


def _training_config():
    from irene_brain.training.config import (
        DatasetConfig,
        DeterminismConfig,
        LoggingConfig,
        OptimizationConfig,
        PrecisionConfig,
        ResourceConfig,
        RunConfig,
        TrainingConfig,
    )

    return TrainingConfig(
        schema_version=1,
        run=RunConfig(
            name="baseline-smoke-compare",
            seed=SEED,
            model_factory="irene_brain.model.torch_model:IreneBrainModel",
            max_optimizer_steps=2,
        ),
        dataset=DatasetConfig(
            kind="moving_shapes",
            train_sequences=4,
            validation_sequences=2,
            test_sequences=2,
            sequence_length=2,
            burn_in_steps=1,
            seed_offset=0,
            hazard_count=1,
            tick_period_ns=33_333_333,
            discount=0.99,
        ),
        optimization=OptimizationConfig(
            batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_gradient_norm=1.0,
            warmup_steps=0,
        ),
        precision=PrecisionConfig(device="cpu", mode="float32", allow_tf32=False),
        determinism=DeterminismConfig(enabled=True, num_workers=0, compile_model=False),
        logging=LoggingConfig(
            log_every_steps=1,
            evaluate_every_steps=1,
            validation_batches=1,
            checkpoint_every_steps=1,
            keep_last_checkpoints=1,
        ),
        resources=ResourceConfig(
            allow_gpu=False,
            allow_capture=False,
            allow_hid_output=False,
            allow_background_threads=False,
            allow_network=False,
            allow_subprocess=False,
            write_artifacts=True,
            cpu_threads=1,
        ),
    )


def run_variant(variant_id: str, *, steps: int) -> dict[str, object]:
    import importlib

    import torch

    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    from irene_brain.training.batches import MovingShapesBatchSource
    from irene_brain.training.config import DatasetConfig
    from irene_brain.training.objective import ThoughtFieldObjective
    from irene_brain.training.torch_system import TorchTrainingSystem

    model = _build_variant(variant_id)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Own-recipe-family variants declare their objective fail-closed on the
    # model; resolve it with the same importlib discipline as train.py.
    objective_class_path = getattr(model, "training_objective_class_path", None)
    if objective_class_path is None:
        objective_class = ThoughtFieldObjective
        objective_name = "irene_brain.training.objective:ThoughtFieldObjective"
    else:
        module_name, attribute = objective_class_path.split(":", 1)
        objective_class = getattr(importlib.import_module(module_name), attribute)
        objective_name = objective_class_path
    system = TorchTrainingSystem(objective_class(model), _training_config())
    source = MovingShapesBatchSource(
        DatasetConfig(
            kind="moving_shapes",
            train_sequences=TRAIN_SEQUENCES,
            validation_sequences=VALIDATION_SEQUENCES,
            test_sequences=2,
            sequence_length=SEQUENCE_LENGTH,
            burn_in_steps=BURN_IN_STEPS,
            seed_offset=0,
            hazard_count=3,
            tick_period_ns=16_666_667,
            discount=0.99,
        )
    )

    def validation_metrics() -> dict[str, float]:
        rows = []
        for batch in source.iter_batches(
            split="validation",
            epoch=0,
            start_batch=0,
            batch_size=1,
            max_batches=VALIDATION_SEQUENCES,
        ):
            rows.append(system.evaluate_batch(batch).metrics)
        keys = rows[0].keys()
        return {
            key: sum(float(row[key]) for row in rows) / len(rows) for key in keys
        }

    before = validation_metrics()
    losses: list[float] = []
    for step in range(steps):
        epoch, index = divmod(step, TRAIN_SEQUENCES)
        batch = next(
            source.iter_batches(
                split="train",
                epoch=epoch,
                start_batch=index,
                batch_size=1,
                max_batches=1,
            )
        )
        losses.append(system.train_optimizer_step((batch,)).loss)
    after = validation_metrics()
    window = min(16, steps)
    return {
        "variant_id": variant_id,
        "seed": SEED,
        "optimizer_steps": steps,
        "trainable_parameters": trainable,
        "training_objective": objective_name,
        "sequence_length": SEQUENCE_LENGTH,
        "burn_in_steps": BURN_IN_STEPS,
        "train_loss_first": sum(losses[:window]) / window,
        "train_loss_last": sum(losses[-window:]) / window,
        "val_action_loss_before": before["action_loss"],
        "val_action_loss_after": after["action_loss"],
        "val_movement_exact_match_before": before["movement_exact_match"],
        "val_movement_exact_match_after": after["movement_exact_match"],
        "val_world_loss_after": after["world_loss"],
        "val_thought_pairwise_cosine_mean_after": after[
            "thought_pairwise_cosine_mean"
        ],
        "val_thought_duplicate_pair_fraction_after": after[
            "thought_duplicate_pair_fraction"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--variants", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arguments = parser.parse_args()
    if arguments.steps < 1:
        raise SystemExit("--steps must be positive")

    from irene_brain.model.baselines import (
        DENSE_COMMUNICATION_IDENTITY,
        FIXED_MULTI_HORIZON_IDENTITY,
        MATCHED_ENSEMBLE_IDENTITY,
        MONOLITHIC_IDENTITY,
        NO_COMMUNICATION_IDENTITY,
        PARAMETER_MATCHED_MONOLITHIC_IDENTITY,
        REACTIVE_IDENTITY,
        REFERENCE_IDENTITY,
        RECURRENT_TRANSFORMER_IDENTITY,
        RESET_STATE_IDENTITY,
        SERIAL_DEPTH_IDENTITY,
    )
    from irene_brain.model.world_model_actor import WORLD_MODEL_ACTOR_IDENTITY

    all_variants = [
        REFERENCE_IDENTITY.variant_id,
        NO_COMMUNICATION_IDENTITY.variant_id,
        RESET_STATE_IDENTITY.variant_id,
        DENSE_COMMUNICATION_IDENTITY.variant_id,
        REACTIVE_IDENTITY.variant_id,
        SERIAL_DEPTH_IDENTITY.variant_id,
        MATCHED_ENSEMBLE_IDENTITY.variant_id,
        FIXED_MULTI_HORIZON_IDENTITY.variant_id,
        MONOLITHIC_IDENTITY.variant_id,
        PARAMETER_MATCHED_MONOLITHIC_IDENTITY.variant_id,
        RECURRENT_TRANSFORMER_IDENTITY.variant_id,
        WORLD_MODEL_ACTOR_IDENTITY.variant_id,
    ]
    selected = (
        all_variants
        if arguments.variants is None
        else [item.strip() for item in arguments.variants.split(",") if item.strip()]
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for variant_id in selected:
        row_path = arguments.output_dir / f"{variant_id}.json"
        row = run_variant(variant_id, steps=arguments.steps)
        row_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        print(
            f"{variant_id}: val_action_loss {row['val_action_loss_before']:.4f}"
            f" -> {row['val_action_loss_after']:.4f},"
            f" movement_exact {row['val_movement_exact_match_before']:.4f}"
            f" -> {row['val_movement_exact_match_after']:.4f},"
            f" pairwise_cos {row['val_thought_pairwise_cosine_mean_after']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
