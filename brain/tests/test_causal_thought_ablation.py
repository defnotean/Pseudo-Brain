from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest

from irene_brain.evaluation.causal_thought_ablation import (
    CausalThoughtAblationReport,
    CausalThoughtEvaluationInputError,
    CheckpointIdentity,
    DecisionCounts,
    EXPECTED_CHANGED_SAMPLES,
    EXPECTED_SAMPLES,
    PairedCorrectnessCounts,
    REGISTERED_A_CONFIG_FILE_SHA256,
    REGISTERED_A_CONFIG_SHA256,
    REGISTERED_A_RECIPE_ID,
    REGISTERED_B_CONFIG_FILE_SHA256,
    REGISTERED_B_CONFIG_SHA256,
    REGISTERED_B_RECIPE_ID,
    REGISTERED_CONFIG_FILE_SHA256,
    REGISTERED_CONFIG_SHA256,
    REGISTERED_DATA_SHA256,
    REGISTERED_RECIPES,
    _movement_prediction,
    _preregistered_checks,
    _resolve_release_inputs,
    _validation_slice,
    evaluate_causal_thought_ablation,
)
from irene_brain.training.batches import MovingShapesBatchSource
from irene_brain.training.config import load_training_config
from irene_brain.training.checkpoint import file_sha256, source_tree_sha256


BRAIN_ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
except ImportError:  # pragma: no cover - the deterministic non-ML suite still imports
    torch = None


def _identity(recipe_id: str = REGISTERED_A_RECIPE_ID) -> CheckpointIdentity:
    recipe = REGISTERED_RECIPES[recipe_id]
    return CheckpointIdentity(
        checkpoint_sha256="a" * 64,
        config_sha256=recipe.config_sha256,
        config_file_sha256=recipe.config_file_sha256,
        data_sha256=recipe.data_sha256,
        training_code_sha256="b" * 64,
        evaluator_sha256="c" * 64,
        optimizer_step=500,
        epoch=0,
        next_batch=4000,
        runtime_fingerprint={"runtime": "test", "world_size": 1},
        training_source_root="/immutable/release/brain/src/irene_brain",
    )


def _boundary_report(
    recipe_id: str = REGISTERED_A_RECIPE_ID,
) -> CausalThoughtAblationReport:
    conditions = {
        "full": DecisionCounts(80, 13),
        "zeroed": DecisionCounts(75, 10),
        "batch_shuffled": DecisionCounts(75, 10),
    }
    pairs = {
        name: PairedCorrectnessCounts(5, 0, 3, 0)
        for name in ("zeroed", "batch_shuffled")
    }
    changed_samples = [2] * 10 + [1] * 6
    overall_full_only = {0, 1, 2, 11, 12}
    changed_full_only = {0, 1, 2}
    per_sequence = []
    for index in range(16):
        full_changed = 2 if index < 6 else (1 if index == 6 else 0)
        altered_changed = full_changed - (index in changed_full_only)
        full_exact = 5
        altered_exact = full_exact - (index in overall_full_only)
        pair_wins = int(index in overall_full_only)
        changed_pair_wins = int(index in changed_full_only)
        per_sequence.append(
            {
                "sequence_index": index,
                "decision_count": 6,
                "changed_decision_count": changed_samples[index],
                "conditions": {
                    "full": {
                        "movement_exact_count": full_exact,
                        "changed_movement_exact_count": full_changed,
                    },
                    "zeroed": {
                        "movement_exact_count": altered_exact,
                        "changed_movement_exact_count": altered_changed,
                    },
                    "batch_shuffled": {
                        "movement_exact_count": altered_exact,
                        "changed_movement_exact_count": altered_changed,
                    },
                },
                "effects": {
                    name: {
                        "decisions_changed_from_full_count": pair_wins,
                        "paired_correctness": {
                            "full_correct_intervention_wrong": pair_wins,
                            "full_wrong_intervention_correct": 0,
                            "changed_full_correct_intervention_wrong": (
                                changed_pair_wins
                            ),
                            "changed_full_wrong_intervention_correct": 0,
                        },
                    }
                    for name in ("zeroed", "batch_shuffled")
                },
            }
        )
    return CausalThoughtAblationReport(
        identity=_identity(recipe_id),
        validation_slice_sha256="d" * 64,
        conditions=conditions,
        decisions_changed_from_full={"zeroed": 5, "batch_shuffled": 5},
        paired_correctness=pairs,
        per_sequence_outcomes=tuple(per_sequence),
    )


class CausalThoughtReportTests(unittest.TestCase):
    def test_registered_boundary_passes_with_integer_effect_sizes(self) -> None:
        report = _boundary_report()
        payload = report.to_dict()

        self.assertTrue(report.passed)
        self.assertTrue(report.causal_decision_effect_observed)
        self.assertEqual(payload["validation_slice"]["decision_count"], 96)
        self.assertEqual(
            payload["validation_slice"][
                "batch_shuffled_donor_sequence_index_by_receiver"
            ],
            [*range(1, 16), 0],
        )
        self.assertEqual(
            payload["effects"]["zeroed"]["paired_correctness"][
                "full_correct_intervention_wrong"
            ],
            5,
        )
        self.assertEqual(len(payload["per_sequence_outcomes"]), 16)
        self.assertEqual(payload["identity"]["registered_recipe"], REGISTERED_A_RECIPE_ID)
        self.assertEqual(
            payload["registered_recipe"]["recipe_id"],
            REGISTERED_A_RECIPE_ID,
        )

    def test_constant_lr_b_identity_and_report_are_registered_exactly(self) -> None:
        identity = _identity(REGISTERED_B_RECIPE_ID)
        report = _boundary_report(REGISTERED_B_RECIPE_ID)
        recipe_payload = report.to_dict()["registered_recipe"]

        self.assertEqual(identity.registered_recipe.recipe_id, REGISTERED_B_RECIPE_ID)
        self.assertEqual(recipe_payload["recipe_id"], REGISTERED_B_RECIPE_ID)
        self.assertEqual(recipe_payload["config_schema_version"], 2)
        self.assertEqual(recipe_payload["scheduler_kind"], "constant_after_warmup")
        self.assertEqual(recipe_payload["optimizer_step"], 500)
        self.assertEqual(recipe_payload["epoch"], 0)
        self.assertEqual(recipe_payload["next_batch"], 4000)

    def test_absolute_and_both_relative_controls_are_required(self) -> None:
        report = _boundary_report()
        base = dict(report.conditions)
        failures = {
            "full absolute": {
                "full": DecisionCounts(79, 13),
                "zeroed": DecisionCounts(74, 10),
                "batch_shuffled": DecisionCounts(74, 10),
            },
            "changed absolute": {
                "full": DecisionCounts(80, 12),
                "zeroed": DecisionCounts(75, 9),
                "batch_shuffled": DecisionCounts(75, 9),
            },
            "zero effect": {**base, "zeroed": DecisionCounts(76, 10)},
            "shuffle effect": {
                **base,
                "batch_shuffled": DecisionCounts(76, 10),
            },
        }
        for name, conditions in failures.items():
            with self.subTest(name=name):
                self.assertFalse(
                    all(check["passed"] for check in _preregistered_checks(conditions))
                )

    def test_impossible_counts_and_wrong_checkpoint_boundary_fail_closed(self) -> None:
        with self.assertRaises(CausalThoughtEvaluationInputError):
            DecisionCounts(EXPECTED_SAMPLES + 1, 0)
        with self.assertRaises(CausalThoughtEvaluationInputError):
            DecisionCounts(1, EXPECTED_CHANGED_SAMPLES + 1)
        with self.assertRaises(CausalThoughtEvaluationInputError):
            CheckpointIdentity(
                checkpoint_sha256="a" * 64,
                config_sha256=REGISTERED_CONFIG_SHA256,
                config_file_sha256=REGISTERED_CONFIG_FILE_SHA256,
                data_sha256=REGISTERED_DATA_SHA256,
                training_code_sha256="b" * 64,
                evaluator_sha256="c" * 64,
                optimizer_step=499,
                epoch=0,
                next_batch=3992,
                runtime_fingerprint={"runtime": "test"},
                training_source_root="/release/source",
            )

    def test_only_exact_registered_a_or_b_identity_tuple_is_accepted(self) -> None:
        self.assertEqual(
            tuple(REGISTERED_RECIPES),
            (REGISTERED_A_RECIPE_ID, REGISTERED_B_RECIPE_ID),
        )
        self.assertEqual(REGISTERED_CONFIG_SHA256, REGISTERED_A_CONFIG_SHA256)
        self.assertEqual(
            REGISTERED_CONFIG_FILE_SHA256,
            REGISTERED_A_CONFIG_FILE_SHA256,
        )

        invalid_hash_tuples = (
            (
                REGISTERED_A_CONFIG_SHA256,
                REGISTERED_B_CONFIG_FILE_SHA256,
                REGISTERED_DATA_SHA256,
            ),
            (
                REGISTERED_B_CONFIG_SHA256,
                REGISTERED_A_CONFIG_FILE_SHA256,
                REGISTERED_DATA_SHA256,
            ),
            (
                REGISTERED_A_CONFIG_SHA256,
                REGISTERED_A_CONFIG_FILE_SHA256,
                "0" * 64,
            ),
            ("0" * 64, "1" * 64, REGISTERED_DATA_SHA256),
        )
        for config_hash, raw_hash, data_hash in invalid_hash_tuples:
            with self.subTest(
                config_hash=config_hash,
                raw_hash=raw_hash,
                data_hash=data_hash,
            ):
                with self.assertRaises(CausalThoughtEvaluationInputError):
                    CheckpointIdentity(
                        checkpoint_sha256="a" * 64,
                        config_sha256=config_hash,
                        config_file_sha256=raw_hash,
                        data_sha256=data_hash,
                        training_code_sha256="b" * 64,
                        evaluator_sha256="c" * 64,
                        optimizer_step=500,
                        epoch=0,
                        next_batch=4000,
                        runtime_fingerprint={"runtime": "test"},
                        training_source_root="/release/source",
                    )

        with self.assertRaises(TypeError):
            REGISTERED_RECIPES["unregistered"] = REGISTERED_RECIPES[  # type: ignore[index]
                REGISTERED_A_RECIPE_ID
            ]

    def test_impossible_per_sequence_paired_ledger_is_rejected_and_frozen(self) -> None:
        report = _boundary_report()
        with self.assertRaises(TypeError):
            report.per_sequence_outcomes[0]["conditions"]["full"][  # type: ignore[index]
                "movement_exact_count"
            ] = 0

        records = report.to_dict()["per_sequence_outcomes"]
        records[0]["effects"]["zeroed"]["paired_correctness"][  # type: ignore[index]
            "full_correct_intervention_wrong"
        ] = 2
        records[1]["effects"]["zeroed"]["paired_correctness"][  # type: ignore[index]
            "full_correct_intervention_wrong"
        ] = 0
        with self.assertRaises(CausalThoughtEvaluationInputError):
            CausalThoughtAblationReport(
                identity=report.identity,
                validation_slice_sha256=report.validation_slice_sha256,
                conditions=report.conditions,
                decisions_changed_from_full=report.decisions_changed_from_full,
                paired_correctness=report.paired_correctness,
                per_sequence_outcomes=tuple(records),
            )


class FixedValidationSliceTests(unittest.TestCase):
    def test_both_recipes_reconstruct_the_exact_registered_validation_slice(self) -> None:
        slice_digests = set()
        for recipe_id, recipe in REGISTERED_RECIPES.items():
            with self.subTest(recipe_id=recipe_id):
                config_path = (
                    BRAIN_ROOT / "configs" / "training" / recipe.config_filename
                )
                self.assertEqual(file_sha256(config_path), recipe.config_file_sha256)
                config = load_training_config(config_path)
                source = MovingShapesBatchSource(config.dataset)
                batch, digest = _validation_slice(source, config)

                self.assertEqual(config.config_sha256, recipe.config_sha256)
                self.assertEqual(config.schema_version, recipe.config_schema_version)
                self.assertEqual(config.run.name, recipe.run_name)
                self.assertEqual(
                    config.optimization.scheduler_kind,
                    recipe.scheduler_kind,
                )
                self.assertEqual(source.manifest_sha256, recipe.data_sha256)
                self.assertEqual(batch.sample_count, EXPECTED_SAMPLES)
                self.assertEqual(
                    tuple(sequence.sequence_index for sequence in batch.sequences),
                    tuple(range(16)),
                )
                self.assertEqual(len(digest), 64)
                slice_digests.add(digest)

        self.assertEqual(len(slice_digests), 1)

    def test_release_binding_requires_the_actually_imported_source_tree(self) -> None:
        release_root = BRAIN_ROOT.parent
        source_root = BRAIN_ROOT / "src" / "irene_brain"
        config_path = (
            BRAIN_ROOT
            / "configs"
            / "training"
            / "dgx-stagea-continuation-gate.toml"
        )
        digest = source_tree_sha256(source_root)

        resolved_release, resolved_source, resolved_config = _resolve_release_inputs(
            training_release_root=release_root,
            config_path=config_path,
            expected_training_code_sha256=digest,
        )

        self.assertEqual(resolved_release, release_root.resolve())
        self.assertEqual(resolved_source, source_root.resolve())
        self.assertEqual(resolved_config, config_path.resolve())
        with self.assertRaises(CausalThoughtEvaluationInputError):
            _resolve_release_inputs(
                training_release_root=release_root,
                config_path=config_path,
                expected_training_code_sha256="0" * 64,
            )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CausalThoughtExecutionTests(unittest.TestCase):
    def test_nonfinite_movement_logits_fail_closed(self) -> None:
        logits = torch.zeros(1, 296)
        logits[0, 4] = float("nan")
        with self.assertRaises(CausalThoughtEvaluationInputError):
            _movement_prediction(SimpleNamespace(button_logits=logits))

    def test_full_trajectory_advances_once_and_interventions_only_replay_actuator(self) -> None:
        config = load_training_config(
            BRAIN_ROOT
            / "configs"
            / "training"
            / "dgx-stagea-continuation-gate.toml"
        )
        source = MovingShapesBatchSource(config.dataset)
        batch, digest = _validation_slice(source, config)
        model = _FakeModel()

        report = evaluate_causal_thought_ablation(
            model=model,
            validation_batch=batch,
            validation_slice_sha256=digest,
            identity=_identity(),
        )

        self.assertEqual(model.forward_calls, 16 * 8)
        self.assertEqual(model.actuator.calls, (16 * 8 * 2) + (16 * 6 * 3))
        self.assertEqual(report.identity, _identity())
        self.assertEqual(
            sum(
                record["changed_decision_count"]
                for record in report.per_sequence_outcomes
            ),
            EXPECTED_CHANGED_SAMPLES,
        )


if torch is not None:

    @dataclass(frozen=True)
    class _FakeState:
        thoughts: object
        step: int

        def detach(self) -> "_FakeState":
            return _FakeState(self.thoughts.detach(), self.step)


    class _FakeActuator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(
            self,
            *,
            sensors: object,
            belief: object,
            thoughts: object,
            working_memory: object,
            retrieved_memory: object,
            goal_context: object,
        ) -> object:
            del belief, working_memory, retrieved_memory, goal_context
            self.calls += 1
            signal = thoughts.float().mean(dim=(1, 2, 3))
            sensor_signal = sensors.float().mean(dim=(1, 2))
            logits = torch.full(
                (thoughts.shape[0], 296),
                -1.0,
                dtype=thoughts.dtype,
                device=thoughts.device,
            )
            logits[:, 4] = signal
            logits[:, 7] = -signal
            logits[:, 22] = sensor_signal - 0.5
            return SimpleNamespace(button_logits=logits)


    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                thoughtlets=2,
                core_width=1,
                cognitive_cycles=1,
            )
            self.input_resolution = (8, 8)
            self.actuator = _FakeActuator()
            self.forward_calls = 0

        def forward(
            self,
            pixels: object,
            previous_control: object,
            elapsed_seconds: object,
            state: object = None,
            *,
            thought_noise: object = None,
        ) -> object:
            del previous_control, elapsed_seconds, thought_noise
            self.forward_calls += 1
            prior_step = 0 if state is None else state.step
            flattened = pixels.float().flatten(1)
            weights = torch.arange(
                1,
                flattened.shape[1] + 1,
                dtype=flattened.dtype,
                device=flattened.device,
            )
            signature = torch.sin((flattened * weights).sum(dim=1) + prior_step)
            thoughts = torch.stack((signature, -signature), dim=1).reshape(1, 2, 1, 1)
            sensors = pixels.float().mean(dim=(2, 3)).mean(dim=1).reshape(1, 1, 1)
            belief = torch.zeros_like(sensors)
            working_memory = torch.zeros_like(sensors)
            retrieved_memory = torch.zeros(1, 2, 1, 1)
            goal_context = torch.zeros_like(sensors)
            common = {
                "sensors": sensors,
                "belief": belief,
                "working_memory": working_memory,
                "retrieved_memory": retrieved_memory,
                "goal_context": goal_context,
            }
            self.actuator(thoughts=thoughts * 0.5, **common)
            action = self.actuator(thoughts=thoughts, **common)
            next_state = _FakeState(thoughts=thoughts, step=prior_step + 1)
            return SimpleNamespace(action=action, next_state=next_state)


if __name__ == "__main__":
    unittest.main()
