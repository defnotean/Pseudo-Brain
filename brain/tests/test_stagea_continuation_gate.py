from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from irene_brain.evaluation.stagea_continuation_gate import (
    EXPECTED_RECORD_ORDER,
    GATE_SCOPE,
    StageAGateInputError,
    evaluate_stagea_continuation_gate,
    main,
)
from irene_brain.training.batches import MovingShapesBatchSource
from irene_brain.training.config import load_training_config
from irene_brain.training.schedules import learning_rate_multiplier


BRAIN_ROOT = Path(__file__).resolve().parents[1]


def _passing_final_metrics() -> dict[str, float]:
    return {
        "loss": 0.2,
        "samples": 96.0,
        "movement_target_active_count": 137.0 / 96.0,
        "movement_changed_samples_per_sample": 26.0 / 96.0,
        "previous_control_movement_exact_match": 70.0 / 96.0,
        "movement_exact_match": 80.0 / 96.0,
        "movement_changed_exact_matches_per_sample": 13.0 / 96.0,
        "positive_key_recall": 173.0 / 192.0,
        "movement_false_positive_count": 12.0 / 96.0,
        "movement_predicted_active_count": 137.0 / 96.0,
        "movement_opposite_conflict_rate": 0.0,
        "non_movement_key_false_positive_count": 0.0,
        "thought_summary_rank_proxy": 0.125,
        "thought_full_register_rank_proxy": 0.125,
        "movement_query_effective_slot_count": 4.0,
        "diversity_loss": 7.0 / 31.0,
        "value_loss": 0.358,
        "world_loss": 0.01,
    }


def _records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for split, step in EXPECTED_RECORD_ORDER:
        metrics = _passing_final_metrics()
        metrics["samples"] = 48.0 if split == "train" else 96.0
        metrics["world_loss"] = 0.5 if step == 1 else 0.01
        records.append(
            {
                "schema_version": 1,
                "step": step,
                "epoch": 0,
                "split": split,
                "metrics": metrics,
            }
        )
    return records


def _write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


class StageAContinuationRecipeTests(unittest.TestCase):
    def test_registered_a_bytes_canonical_hash_and_cosine_resolution_are_unchanged(
        self,
    ) -> None:
        path = (
            BRAIN_ROOT
            / "configs"
            / "training"
            / "dgx-stagea-continuation-gate.toml"
        )
        gate = load_training_config(path)

        self.assertEqual(
            sha256(path.read_bytes()).hexdigest(),
            "47ef30ba199f0b83b5719caa938e3eb77dfd1695dd5e10cda615ab503dc4a162",
        )
        self.assertEqual(
            gate.config_sha256,
            "d6f8c8ba3caaaab4643c430ff75747c0025f866363ddfbbb2114fc80969c8fcd",
        )
        self.assertEqual(gate.schema_version, 1)
        self.assertEqual(gate.optimization.scheduler_kind, "cosine_after_warmup")
        self.assertNotIn("scheduler_kind", gate.to_dict()["optimization"])

    def test_flat_lr_b_has_only_the_intended_semantic_differences(self) -> None:
        config_dir = BRAIN_ROOT / "configs" / "training"
        gate = load_training_config(config_dir / "dgx-stagea-continuation-gate.toml")
        flat = load_training_config(config_dir / "dgx-stagea-continuation-gate-b.toml")

        self.assertEqual(flat.schema_version, 2)
        self.assertEqual(flat.run.name, "dgx-stagea-continuation-gate-b")
        self.assertEqual(flat.optimization.scheduler_kind, "constant_after_warmup")
        self.assertEqual(replace(flat.run, name=gate.run.name), gate.run)
        self.assertEqual(
            replace(
                flat.optimization,
                scheduler_kind=gate.optimization.scheduler_kind,
            ),
            gate.optimization,
        )
        self.assertEqual(flat.dataset, gate.dataset)
        self.assertEqual(flat.objective, gate.objective)
        self.assertEqual(flat.precision, gate.precision)
        self.assertEqual(flat.determinism, gate.determinism)
        self.assertEqual(flat.logging, gate.logging)
        self.assertEqual(flat.resources, gate.resources)
        self.assertEqual(
            flat.run.max_optimizer_steps
            * flat.optimization.gradient_accumulation_steps,
            gate.run.max_optimizer_steps
            * gate.optimization.gradient_accumulation_steps,
        )
        self.assertEqual(
            flat.run.max_optimizer_steps
            * flat.optimization.gradient_accumulation_steps,
            4000,
        )
        self.assertIn(
            '"scheduler_kind":"constant_after_warmup"',
            flat.canonical_json,
        )
        self.assertNotEqual(flat.config_sha256, gate.config_sha256)
        cosine_flat = replace(
            flat,
            optimization=replace(
                flat.optimization,
                scheduler_kind="cosine_after_warmup",
            ),
        )
        self.assertNotEqual(cosine_flat.config_sha256, flat.config_sha256)

    def test_flat_lr_b_has_exact_milestones_and_exposure(self) -> None:
        flat = load_training_config(
            BRAIN_ROOT
            / "configs"
            / "training"
            / "dgx-stagea-continuation-gate-b.toml"
        )

        def learning_rate_for_update(update: int) -> float:
            return flat.optimization.learning_rate * learning_rate_multiplier(
                scheduler_kind=flat.optimization.scheduler_kind,
                warmup_steps=flat.optimization.warmup_steps,
                max_optimizer_steps=flat.run.max_optimizer_steps,
                step=update - 1,
            )

        expected = {
            1: 0.000005,
            10: 0.00005,
            19: 0.000095,
            20: 0.0001,
            21: 0.0001,
            100: 0.0001,
            200: 0.0001,
            300: 0.0001,
            400: 0.0001,
            500: 0.0001,
        }
        for update, expected_learning_rate in expected.items():
            self.assertAlmostEqual(
                learning_rate_for_update(update),
                expected_learning_rate,
                places=15,
            )

        multipliers = [
            learning_rate_multiplier(
                scheduler_kind=flat.optimization.scheduler_kind,
                warmup_steps=flat.optimization.warmup_steps,
                max_optimizer_steps=flat.run.max_optimizer_steps,
                step=step,
            )
            for step in range(flat.run.max_optimizer_steps)
        ]
        self.assertAlmostEqual(sum(multipliers[:20]), 10.5, places=12)
        self.assertEqual(multipliers[20:], [1.0] * 480)
        self.assertAlmostEqual(sum(multipliers), 490.5, places=12)
        self.assertAlmostEqual(
            sum(
                flat.optimization.learning_rate * multiplier
                for multiplier in multipliers
            ),
            0.04905,
            places=15,
        )

    def test_registered_a_retains_the_exact_cosine_schedule(self) -> None:
        gate = load_training_config(
            BRAIN_ROOT
            / "configs"
            / "training"
            / "dgx-stagea-continuation-gate.toml"
        )

        expected_multipliers = {
            1: 0.05,
            20: 1.0,
            21: 1.0,
            100: 0.9346396619725719,
            200: 0.6943609850761978,
            300: 0.373754211492421,
            400: 0.10532397890342504,
            500: 0.000010709167935385455,
        }
        for update, expected_multiplier in expected_multipliers.items():
            self.assertAlmostEqual(
                learning_rate_multiplier(
                    scheduler_kind=gate.optimization.scheduler_kind,
                    warmup_steps=gate.optimization.warmup_steps,
                    max_optimizer_steps=gate.run.max_optimizer_steps,
                    step=update - 1,
                ),
                expected_multiplier,
                places=15,
            )
        self.assertAlmostEqual(
            sum(
                learning_rate_multiplier(
                    scheduler_kind=gate.optimization.scheduler_kind,
                    warmup_steps=gate.optimization.warmup_steps,
                    max_optimizer_steps=gate.run.max_optimizer_steps,
                    step=step,
                )
                for step in range(gate.run.max_optimizer_steps)
            ),
            251.0,
            places=12,
        )

    def test_schema_two_requires_a_valid_explicit_scheduler(self) -> None:
        source = (
            BRAIN_ROOT
            / "configs"
            / "training"
            / "dgx-stagea-continuation-gate-b.toml"
        ).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-scheduler.toml"
            missing.write_text(
                source.replace('scheduler_kind = "constant_after_warmup"\n', ""),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing fields: scheduler_kind"):
                load_training_config(missing)

            invalid = Path(directory) / "invalid-scheduler.toml"
            invalid.write_text(
                source.replace("constant_after_warmup", "future_scheduler"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "scheduler_kind must be one of"):
                load_training_config(invalid)

            legacy_explicit = Path(directory) / "legacy-explicit-scheduler.toml"
            legacy_explicit.write_text(
                (
                    BRAIN_ROOT
                    / "configs"
                    / "training"
                    / "dgx-stagea-continuation-gate.toml"
                )
                .read_text(encoding="utf-8")
                .replace(
                    "warmup_steps = 20",
                    'warmup_steps = 20\nscheduler_kind = "cosine_after_warmup"',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown fields: scheduler_kind"):
                load_training_config(legacy_explicit)

        flat = load_training_config(
            BRAIN_ROOT
            / "configs"
            / "training"
            / "dgx-stagea-continuation-gate-b.toml"
        )
        with self.assertRaisesRegex(ValueError, "schema_version 1 supports only"):
            replace(flat, schema_version=1)

    def test_recipe_has_phase1_data_objective_and_resource_parity(self) -> None:
        phase1 = load_training_config(
            BRAIN_ROOT / "configs" / "training" / "phase1-bootstrap.toml"
        )
        gate = load_training_config(
            BRAIN_ROOT
            / "configs"
            / "training"
            / "dgx-stagea-continuation-gate.toml"
        )

        self.assertEqual(gate.run.name, "dgx-stagea-continuation-gate")
        self.assertEqual(gate.run.seed, phase1.run.seed)
        self.assertEqual(gate.run.model_factory, phase1.run.model_factory)
        self.assertEqual(gate.run.max_optimizer_steps, 500)
        self.assertEqual(gate.dataset, phase1.dataset)
        self.assertEqual(gate.objective, phase1.objective)
        self.assertEqual(gate.precision, phase1.precision)
        self.assertEqual(gate.determinism, phase1.determinism)
        self.assertEqual(gate.resources, phase1.resources)
        self.assertEqual(gate.optimization.batch_size, phase1.optimization.batch_size)
        self.assertEqual(
            gate.optimization.gradient_accumulation_steps,
            phase1.optimization.gradient_accumulation_steps,
        )
        self.assertEqual(gate.optimization.learning_rate, phase1.optimization.learning_rate)
        self.assertEqual(gate.optimization.weight_decay, phase1.optimization.weight_decay)
        self.assertEqual(
            gate.optimization.max_gradient_norm,
            phase1.optimization.max_gradient_norm,
        )
        self.assertEqual(gate.optimization.warmup_steps, 20)
        self.assertEqual(gate.logging.log_every_steps, 100)
        self.assertEqual(gate.logging.evaluate_every_steps, 100)
        self.assertEqual(gate.logging.validation_batches, 16)
        self.assertEqual(gate.logging.checkpoint_every_steps, 100)
        self.assertEqual(
            gate.logging.keep_last_checkpoints,
            phase1.logging.keep_last_checkpoints,
        )

    def test_recipe_itself_reproduces_schedule_and_slice_invariants(self) -> None:
        gate = load_training_config(
            BRAIN_ROOT
            / "configs"
            / "training"
            / "dgx-stagea-continuation-gate.toml"
        )
        source = MovingShapesBatchSource(gate.dataset)
        consumed_train_batches = (
            gate.run.max_optimizer_steps
            * gate.optimization.gradient_accumulation_steps
        )
        self.assertLess(
            consumed_train_batches,
            source.batches_per_epoch(
                split="train",
                batch_size=gate.optimization.batch_size,
            ),
        )
        derived_order: list[tuple[str, int]] = [("train", 1)]
        for step in range(1, gate.run.max_optimizer_steps + 1):
            if step == 1:
                continue
            if step % gate.logging.log_every_steps == 0:
                derived_order.append(("train", step))
            if step % gate.logging.evaluate_every_steps == 0:
                derived_order.append(("validation", step))
        self.assertEqual(tuple(derived_order), EXPECTED_RECORD_ORDER)

        movement_keys = {4, 7, 22, 26}
        samples = 0
        target_active = 0
        changed = 0
        previous_exact = 0
        batches = source.iter_batches(
            split="validation",
            epoch=0,
            start_batch=0,
            batch_size=gate.optimization.batch_size,
            max_batches=gate.logging.validation_batches,
        )
        for batch in batches:
            for sequence in batch.sequences:
                for transition in sequence.transitions[batch.burn_in_steps :]:
                    target = set(transition.action_target.keys_down) & movement_keys
                    previous = (
                        set(transition.observation.previous_control.keys_down)
                        & movement_keys
                    )
                    samples += 1
                    target_active += len(target)
                    changed += target != previous
                    previous_exact += target == previous
        self.assertEqual(samples, 96)
        self.assertEqual(target_active, 137)
        self.assertEqual(changed, 26)
        self.assertEqual(previous_exact, 70)


class StageAContinuationEvaluatorTests(unittest.TestCase):
    def test_boundary_fixture_passes_and_input_is_not_modified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            _write_records(path, _records())
            before = path.read_bytes()

            report = evaluate_stagea_continuation_gate(path)

            self.assertTrue(report.passed)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(report.to_dict()["scope"], GATE_SCOPE)
            self.assertIn("not evidence", report.to_dict()["world_sanity"]["interpretation"])
            self.assertEqual(report.world_sanity["train_step_1_world_loss"], 0.5)
            self.assertEqual(report.world_sanity["train_step_500_world_loss"], 0.01)

    def test_every_scientific_threshold_fails_closed(self) -> None:
        failures = {
            "movement_exact_match": {"movement_exact_match": 79.0 / 96.0},
            "movement_changed_exact_matches_per_sample": {
                "movement_changed_exact_matches_per_sample": 12.0 / 96.0,
            },
            "positive_key_recall": {"positive_key_recall": 172.0 / 192.0},
            "movement_false_positive_count": {
                "movement_false_positive_count": 13.0 / 96.0,
            },
            "movement_predicted_active_count": {
                "movement_predicted_active_count": 122.0 / 96.0,
                "positive_key_recall": 165.0 / 192.0,
            },
            "movement_opposite_conflict_rate": {
                "movement_opposite_conflict_rate": 1.0 / 96.0,
            },
            "non_movement_key_false_positive_count": {
                "non_movement_key_false_positive_count": 1.0 / 96.0,
            },
            "thought_summary_rank_proxy": {"thought_summary_rank_proxy": 0.124},
            "thought_full_register_rank_proxy": {
                "thought_full_register_rank_proxy": 0.124,
            },
            "movement_query_effective_slot_count": {
                "movement_query_effective_slot_count": 3.99,
            },
            "diversity_loss": {"diversity_loss": (7.0 / 31.0) + 0.001},
            "value_loss": {"value_loss": 0.35806523},
        }
        for metric, replacements in failures.items():
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as directory:
                records = _records()
                records[-1]["metrics"].update(replacements)  # type: ignore[union-attr]
                path = Path(directory) / "metrics.jsonl"
                _write_records(path, records)
                self.assertFalse(evaluate_stagea_continuation_gate(path).passed)

    def test_fractional_or_cross_metric_impossible_counts_are_rejected(self) -> None:
        impossible = {
            "fractional count": {"movement_predicted_active_count": 137.5 / 96.0},
            "true positives exceed targets": {
                "movement_predicted_active_count": 150.0 / 96.0,
            },
            "exact unchanged exceeds unchanged samples": {
                "movement_exact_match": 96.0 / 96.0,
            },
            "recall off half-count lattice": {
                "positive_key_recall": 0.90,
            },
            "recall contradicts implied true positives": {
                "positive_key_recall": 192.0 / 192.0,
            },
            "exact decisions contradict false positives": {
                "movement_exact_match": 96.0 / 96.0,
                "movement_changed_exact_matches_per_sample": 26.0 / 96.0,
                "positive_key_recall": 192.0 / 192.0,
                "movement_predicted_active_count": 149.0 / 96.0,
                "movement_false_positive_count": 12.0 / 96.0,
            },
            "errors exceed nonexact decision capacity": {
                "movement_exact_match": 95.0 / 96.0,
                "movement_changed_exact_matches_per_sample": 25.0 / 96.0,
                "positive_key_recall": 190.0 / 192.0,
                "movement_predicted_active_count": 138.0 / 96.0,
                "movement_false_positive_count": 3.0 / 96.0,
            },
        }
        for name, replacements in impossible.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                records = _records()
                records[-1]["metrics"].update(replacements)  # type: ignore[union-attr]
                path = Path(directory) / "metrics.jsonl"
                _write_records(path, records)
                with self.assertRaises(StageAGateInputError):
                    evaluate_stagea_continuation_gate(path)

    def test_final_slice_invariants_are_checked(self) -> None:
        failures = {
            "samples": 95.0,
            "movement_target_active_count": 136.0 / 96.0,
            "movement_changed_samples_per_sample": 25.0 / 96.0,
            "previous_control_movement_exact_match": 69.0 / 96.0,
        }
        for metric, value in failures.items():
            with self.subTest(metric=metric), tempfile.TemporaryDirectory() as directory:
                records = _records()
                records[-1]["metrics"][metric] = value  # type: ignore[index]
                path = Path(directory) / "metrics.jsonl"
                _write_records(path, records)
                with self.assertRaises(StageAGateInputError):
                    evaluate_stagea_continuation_gate(path)

    def test_world_sanity_is_relative_and_not_multiworld_certification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            records = _records()
            records[-1]["metrics"]["world_loss"] = 0.6  # type: ignore[index]
            _write_records(path, records)
            self.assertFalse(evaluate_stagea_continuation_gate(path).passed)

            records = _records()
            records[-2]["metrics"]["world_loss"] = 0.005  # type: ignore[index]
            _write_records(path, records)
            report = evaluate_stagea_continuation_gate(path)
            self.assertFalse(report.passed)
            self.assertIn("not evidence", report.to_dict()["world_sanity"]["interpretation"])

    def test_missing_duplicate_wrong_schema_split_and_step_are_rejected(self) -> None:
        mutations = {
            "missing": lambda records: records.pop(),
            "duplicate": lambda records: records.append(records[-1]),
            "schema": lambda records: records[0].update(schema_version=2),
            "split": lambda records: records[0].update(split="test"),
            "step": lambda records: records[0].update(step=2),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                records = _records()
                mutate(records)
                path = Path(directory) / "metrics.jsonl"
                _write_records(path, records)
                with self.assertRaises(StageAGateInputError):
                    evaluate_stagea_continuation_gate(path)

    def test_nonfinite_missing_metric_and_duplicate_json_key_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            nonfinite = _records()
            nonfinite[-1]["metrics"]["value_loss"] = math.inf  # type: ignore[index]
            nonfinite_path = root / "nonfinite.jsonl"
            nonfinite_path.write_text(
                "".join(json.dumps(record) + "\n" for record in nonfinite),
                encoding="utf-8",
            )
            with self.assertRaises(StageAGateInputError):
                evaluate_stagea_continuation_gate(nonfinite_path)

            overflow = _records()
            overflow[-1]["metrics"]["value_loss"] = 10**400  # type: ignore[index]
            overflow_path = root / "overflow.jsonl"
            overflow_path.write_text(
                "".join(json.dumps(record) + "\n" for record in overflow),
                encoding="utf-8",
            )
            with self.assertRaises(StageAGateInputError):
                evaluate_stagea_continuation_gate(overflow_path)

            huge_integer_path = root / "huge-integer.jsonl"
            huge_integer_path.write_text(
                '{"schema_version":1,"step":1,"epoch":0,"split":"train",'
                '"metrics":{"world_loss":'
                + ("9" * 5000)
                + "}}\n",
                encoding="utf-8",
            )
            with self.assertRaises(StageAGateInputError):
                evaluate_stagea_continuation_gate(huge_integer_path)

            missing = _records()
            del missing[-1]["metrics"]["value_loss"]  # type: ignore[index]
            missing_path = root / "missing.jsonl"
            _write_records(missing_path, missing)
            with self.assertRaises(StageAGateInputError):
                evaluate_stagea_continuation_gate(missing_path)

            duplicate_key_path = root / "duplicate-key.jsonl"
            valid_lines = [
                json.dumps(record, sort_keys=True, separators=(",", ":"))
                for record in _records()
            ]
            valid_lines[0] = valid_lines[0].replace(
                '"schema_version":1',
                '"schema_version":1,"schema_version":1',
                1,
            )
            duplicate_key_path.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
            with self.assertRaises(StageAGateInputError):
                evaluate_stagea_continuation_gate(duplicate_key_path)

    def test_cli_exit_codes_are_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            _write_records(path, _records())
            output = io.StringIO()
            with redirect_stdout(output):
                code = main([str(path)])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(output.getvalue())["passed"])

            failing = _records()
            failing[-1]["metrics"]["movement_exact_match"] = 79.0 / 96.0  # type: ignore[index]
            _write_records(path, failing)
            output = io.StringIO()
            with redirect_stdout(output):
                code = main([str(path)])
            self.assertEqual(code, 1)
            self.assertFalse(json.loads(output.getvalue())["passed"])

            path.write_text("{}\n", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = main([str(path)])
            self.assertEqual(code, 2)
            self.assertFalse(json.loads(output.getvalue())["passed"])


if __name__ == "__main__":
    unittest.main()
