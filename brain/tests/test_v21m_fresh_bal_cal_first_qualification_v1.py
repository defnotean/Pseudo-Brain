"""Focused contracts for PB21M fresh BAL-UPMIX CAL-first qualification."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F


BRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN_ROOT / "scripts"), str(BRAIN_ROOT / "src")]

SCRIPT = BRAIN_ROOT / "scripts" / "v21m_fresh_bal_cal_first_qualification_v1.py"
spec = importlib.util.spec_from_file_location("pb21m_under_test", SCRIPT)
assert spec is not None and spec.loader is not None
pb21m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pb21m
spec.loader.exec_module(pb21m)


def _clone(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in arrays.items()}


def _tape(label: str, roots: int = 2):
    return pb21m.pb21j.FreshLiveTape(
        beliefs=np.zeros((roots, pb21m.pb21j.WIDTH), dtype=np.float32),
        updater_inputs=np.zeros(
            (roots, pb21m.strict.CAPACITY_STATE_WIDTH), dtype=np.float32
        ),
        hazard_targets=np.asarray(
            [[(row + action) % 2 for action in range(pb21m.ACTION_COUNT)] for row in range(roots)],
            dtype=np.float32,
        ),
        parent_raw_logits=np.zeros((roots, pb21m.ACTION_COUNT), dtype=np.float32),
        factual_actions=np.arange(roots, dtype=np.int64) % pb21m.ACTION_COUNT,
        episode_group_ids=tuple(f"{label}:episode:{row}" for row in range(roots)),
        root_state_ids=tuple(f"{label}:root:{row}" for row in range(roots)),
        prior_assignment_count=np.zeros(roots, dtype=np.int64),
        prior_disagreement_count=np.zeros(roots, dtype=np.int64),
    )


def _clustered_tape(label: str, *, roots: int, roots_per_episode: int, offset: int):
    row = np.arange(roots, dtype=np.int64)
    action = np.arange(pb21m.ACTION_COUNT, dtype=np.int64)
    targets = ((row[:, None] + 2 * action[None, :] + offset) % 4 < 2).astype(np.float32)
    updater_inputs = np.zeros(
        (roots, pb21m.strict.CAPACITY_STATE_WIDTH), dtype=np.float32
    )
    updater_inputs[:, 365] = 1.0
    return pb21m.pb21j.FreshLiveTape(
        beliefs=np.zeros((roots, pb21m.pb21j.WIDTH), dtype=np.float32),
        updater_inputs=updater_inputs,
        hazard_targets=targets,
        parent_raw_logits=np.zeros((roots, pb21m.ACTION_COUNT), dtype=np.float32),
        factual_actions=((row + offset) % pb21m.ACTION_COUNT).astype(np.int64),
        episode_group_ids=tuple(
            f"{label}:episode:{index // roots_per_episode}" for index in range(roots)
        ),
        root_state_ids=tuple(f"{label}:root:{index}" for index in range(roots)),
        prior_assignment_count=(row % 3).astype(np.int64),
        prior_disagreement_count=(row % 2).astype(np.int64),
    )


def _mini_manifests() -> dict[str, str]:
    return {spec.label: f"{index + 1:064x}" for index, spec in enumerate(pb21m.PARTITION_SPECS)}


def _mini_cal_arrays(bundle: str) -> dict[str, np.ndarray]:
    raw = np.arange(2 * 9 * 2 * 2 * 5, dtype=np.float64).reshape(2, 9, 2, 2, 5) / 37.0
    scales = np.empty((2, 9, 2, 5), dtype=np.float64)
    biases = np.empty_like(scales)
    scales[:, :, 0] = 1.25
    scales[:, :, 1] = 0.75
    biases[:, :, 0] = -0.20
    biases[:, :, 1] = 0.30
    expected_a = raw[:, :, 0] * scales[:, :, 1, None, :] + biases[:, :, 1, None, :]
    expected_b = raw[:, :, 1] * scales[:, :, 0, None, :] + biases[:, :, 0, None, :]
    oof = np.concatenate((expected_a, expected_b), axis=2)
    crossfit_roles = tuple(role for _ in range(2 * 9) for role in ("A", "B"))
    final_roles = ("A+B",) * (2 * 9)
    manifests = _mini_manifests()
    crossfit_manifests: list[str] = []
    final_manifests: list[str] = []
    for _scorer in pb21m.SCORERS:
        for cell_index, _cell in enumerate(pb21m.CELLS):
            cohort = cell_index // len(pb21m.INIT_SEEDS)
            left = manifests[pb21m.CAL_A_COHORTS[cohort]]
            right = manifests[pb21m.CAL_B_COHORTS[cohort]]
            crossfit_manifests.extend((left, right))
            final_manifests.append(pb21m.combined_cal_manifest_sha256(left, right))
    crossfit_namespaces = tuple(
        f"maze_chase.pb21m.train-only.{role}.v1" for role in crossfit_roles
    )
    final_namespaces = tuple(
        f"maze_chase.pb21m.train-only.{role}.v1" for role in final_roles
    )
    crossfit_algorithms = tuple(
        f"{pb21m.PARTITION_ALGORITHM};calibration_role={role}"
        for role in crossfit_roles
    )
    final_algorithms = tuple(
        f"{pb21m.PARTITION_ALGORITHM};calibration_role={role}"
        for role in final_roles
    )
    arrays = {
        "scorer_ids": pb21m._byte_strings(pb21m.SCORERS, 12),
        "cell_ids": pb21m._byte_strings(pb21m.CELLS, 16),
        "cal_targets": np.zeros((3, 4, 5), dtype=np.float32),
        "cal_actions": np.zeros((3, 4), dtype=np.int64),
        "cal_raw_logits": raw,
        "crossfit_fit_fold_index": np.asarray((0, 1), dtype=np.int16),
        "crossfit_eval_fold_index": np.asarray((1, 0), dtype=np.int16),
        "oof_row_eval_fold_index": np.repeat(np.asarray((0, 1), dtype=np.int16), 2),
        "crossfit_scales": scales,
        "crossfit_biases": biases,
        "crossfit_accepted": np.ones((2, 9, 2), dtype=np.bool_),
        "final_accepted": np.ones((2, 9), dtype=np.bool_),
        "oof_calibrated_logits": oof,
        "oof_calibrated_probabilities": 1.0 / (1.0 + np.exp(-oof)),
        "bootstrap_indices": np.zeros((3, 3, 2, 2), dtype=np.uint16),
        "episode_derangements": np.ones((2, 3, 2, 2), dtype=np.uint16),
        "calibration_source_namespace": pb21m._byte_strings(crossfit_namespaces, 64),
        "calibration_source_split": pb21m._byte_strings(("TRAIN",) * 36, 16),
        "calibration_source_partition": pb21m._byte_strings(("TRAIN-CAL",) * 36, 16),
        "calibration_dataset_manifest_sha256": pb21m._byte_strings(crossfit_manifests, 64),
        "calibration_source_bundle_sha256": pb21m._byte_strings((bundle,) * 36, 64),
        "calibration_partition_algorithm": pb21m._byte_strings(crossfit_algorithms, 512),
        "final_calibration_source_namespace": pb21m._byte_strings(final_namespaces, 64),
        "final_calibration_source_split": pb21m._byte_strings(("TRAIN",) * 18, 16),
        "final_calibration_source_partition": pb21m._byte_strings(("TRAIN-CAL",) * 18, 16),
        "final_calibration_dataset_manifest_sha256": pb21m._byte_strings(final_manifests, 64),
        "final_calibration_source_bundle_sha256": pb21m._byte_strings((bundle,) * 18, 64),
        "final_calibration_partition_algorithm": pb21m._byte_strings(final_algorithms, 512),
    }
    return arrays


class PB21MFreshBALCALFirstTests(unittest.TestCase):
    def test_frozen_constants_ranges_and_collision_registry(self) -> None:
        self.assertEqual(pb21m.SCORERS, ("BASE", "BAL-UPMIX"))
        self.assertEqual(pb21m.FIT_COHORTS, ("F0", "F1", "F2"))
        self.assertEqual(pb21m.INIT_SEEDS, (91_042, 92_042, 93_042))
        self.assertEqual(pb21m.PERMUTATION_SEED, 94_042)
        self.assertEqual(pb21m.BOOTSTRAP_SEED, 95_042)
        self.assertEqual(pb21m.DERANGEMENT_SEED, 96_042)
        self.assertEqual(pb21m.TOTAL_HEADS, 18)
        self.assertEqual(pb21m.TOTAL_OPTIMIZER_STEPS, 73_728)
        observed = [
            (item.label, item.split.value, item.seed_offset, item.seed_offset + item.episodes)
            for item in pb21m.PARTITION_SPECS
        ]
        self.assertEqual(
            observed,
            [
                ("F0", "train", 167_772_160, 167_772_416),
                ("C0A", "train", 167_772_416, 167_772_672),
                ("C0B", "train", 167_772_672, 167_772_928),
                ("F1", "train", 167_772_928, 167_773_184),
                ("C1A", "train", 167_773_184, 167_773_440),
                ("C1B", "train", 167_773_440, 167_773_696),
                ("F2", "train", 167_773_696, 167_773_952),
                ("C2A", "train", 167_773_952, 167_774_208),
                ("C2B", "train", 167_774_208, 167_774_464),
                ("DEV", "validation", 184_549_376, 184_550_144),
            ],
        )
        audit = pb21m.assert_no_range_collisions()
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["collisions"], [])
        self.assertEqual(len(audit["records"]), 34)
        self.assertEqual(
            set(pb21m.partition_manifests()), {spec.label for spec in pb21m.PARTITION_SPECS}
        )
        colliding = [
            {"dataset_family": "maze_chase", "effective_start": 10,
             "effective_stop_exclusive": 20, "owner": "left"},
            {"dataset_family": "maze_chase", "effective_start": 19,
             "effective_stop_exclusive": 30, "owner": "right"},
        ]
        with self.assertRaisesRegex(RuntimeError, "collide"):
            pb21m.assert_no_range_collisions(colliding)

    def test_bal_objective_is_exact_per_action_balanced_factual_mix(self) -> None:
        actions = np.concatenate(
            tuple(np.full(count, action) for action, count in enumerate((400, 500, 600, 700, 872)))
        ).astype(np.int64)
        weights = pb21m.pb21l.consumed_factual_balance_weights(actions)
        counts = np.bincount(actions, minlength=5)
        self.assertEqual(weights.dtype, np.dtype("<f4"))
        for action in range(5):
            expected = np.float32(pb21m.ROOTS_PER_FIT / (5 * counts[action]))
            self.assertTrue(np.all(weights[actions == action] == expected))
        logits = torch.linspace(-1.5, 1.5, pb21m.ROOTS_PER_FIT * 5).reshape(-1, 5)
        targets = ((torch.arange(pb21m.ROOTS_PER_FIT)[:, None] + torch.arange(5)) % 3 == 0).float()
        action_tensor = torch.from_numpy(actions)
        loss, report = pb21m.pb21l.upmix_loss(
            logits, targets, action_tensor, scorer=pb21m.SCORER_BAL,
            balanced_weights=torch.from_numpy(weights),
        )
        all_loss = F.binary_cross_entropy_with_logits(logits, targets)
        selected = F.binary_cross_entropy_with_logits(
            logits[torch.arange(len(logits)), action_tensor],
            targets[torch.arange(len(logits)), action_tensor], reduction="none",
        )
        factual = torch.stack([selected[action_tensor == action].mean() for action in range(5)]).mean()
        self.assertTrue(torch.allclose(loss, 0.5 * all_loss + 0.5 * factual, atol=2e-7))
        self.assertAlmostEqual(report["factual_component_BCE"], float(factual), places=6)

    def test_real_calibrator_preflight_is_full_deterministic_and_non_lifecycle(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = pb21m.real_calibrator_packaging_preflight(root)
            second = pb21m.real_calibrator_packaging_preflight(root)
            self.assertTrue(first["passed"])
            self.assertTrue(first["uses_real_fitted_PerActionAffineHazardCalibrator"])
            self.assertTrue(first["builder_writer_allow_pickle_false_reload_refit_evaluator_traversed"])
            self.assertTrue(first["canonical_lifecycle_paths_untouched"])
            self.assertFalse(first["reserved_partition_constructed"])
            self.assertEqual(first["evidence_sha256"], second["evidence_sha256"])
            self.assertEqual(first["array_manifest_sha256"], second["array_manifest_sha256"])
            for role in ("attempt", "cal_evidence", "cal_decision", "dev_open", "dev_evidence", "result"):
                self.assertFalse(pb21m.canonical_paths(root)[role].exists())

    def test_production_cal_builder_round_trip_real_refit_and_evaluator(self) -> None:
        roots = 100
        roots_per_episode = 5
        episodes = roots // roots_per_episode
        bootstrap_resamples = 8
        derangement_repetitions = 4
        source_bundle = "7" * 64
        manifests = _mini_manifests()
        bootstrap = np.random.default_rng(12_345).integers(
            0, episodes,
            size=(bootstrap_resamples, len(pb21m.FIT_COHORTS), 2, episodes),
            dtype=np.uint16,
        )
        derangements = np.empty(
            (derangement_repetitions, len(pb21m.FIT_COHORTS), 2, episodes),
            dtype=np.uint16,
        )
        identity = np.arange(episodes, dtype=np.uint16)
        for repetition in range(derangement_repetitions):
            for cohort in range(len(pb21m.FIT_COHORTS)):
                for fold in range(2):
                    shift = 1 + (repetition + cohort + fold) % (episodes - 1)
                    derangements[repetition, cohort, fold] = np.roll(identity, shift)

        fit_tapes = {
            label: _clustered_tape(
                label, roots=roots, roots_per_episode=roots_per_episode, offset=index
            )
            for index, label in enumerate(pb21m.FIT_COHORTS)
        }
        cal_labels = (*pb21m.CAL_A_COHORTS, *pb21m.CAL_B_COHORTS)
        cal_tapes = {
            label: _clustered_tape(
                label, roots=roots, roots_per_episode=roots_per_episode, offset=10 + index
            )
            for index, label in enumerate(cal_labels)
        }
        heads: dict[str, dict[str, torch.nn.Module]] = {}
        training: dict[str, dict[str, object]] = {}
        pruning: dict[str, dict[str, object]] = {}
        raw = np.empty((2, 9, 2, roots, 5), dtype=np.float64)
        crossfit = []
        final = []
        oof_logits = np.empty((2, 9, 2 * roots, 5), dtype=np.float64)
        oof_probabilities = np.empty_like(oof_logits)

        with ExitStack() as stack:
            stack.enter_context(patch.multiple(
                pb21m,
                ROOTS_PER_EPISODE=roots_per_episode,
                FIT_EPISODES=episodes,
                CAL_FOLD_EPISODES=episodes,
                ROOTS_PER_FIT=roots,
                ROOTS_PER_CAL_FOLD=roots,
                ROOTS_PER_CAL=2 * roots,
                PASSES=2,
                ROOT_BATCH_SIZE=20,
                STEPS_PER_HEAD=10,
                BOOTSTRAP_RESAMPLES=bootstrap_resamples,
                DERANGEMENT_REPETITIONS=derangement_repetitions,
            ))
            stack.enter_context(patch.object(pb21m.pb21l, "ROOTS_PER_FIT", roots))
            stack.enter_context(patch.object(pb21m, "partition_manifests", return_value=manifests))
            stack.enter_context(patch.object(pb21m, "cal_bootstrap_indices", return_value=bootstrap))
            stack.enter_context(patch.object(pb21m, "cal_episode_derangements", return_value=derangements))
            permutations = pb21m.deterministic_permutations()
            permutation_hashes = tuple(
                pb21m._array_sha256(value.numpy().astype(np.int64, copy=False))
                for value in permutations
            )
            for cell_index, cell in enumerate(pb21m.CELLS):
                heads[cell] = {}
                training[cell] = {}
                pruning[cell] = {}
                initial_state_sha = pb21m._state_sha256(
                    pb21m.initialized_head(int(cell.rsplit("-", 1)[1]))
                )
                for scorer_index, scorer in enumerate(pb21m.SCORERS):
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(1000 + 10 * cell_index + scorer_index)
                        head = pb21m.pb21j.PrunedHazardHead(input_width=1)
                    heads[cell][scorer] = head
                    state_sha = pb21m._state_sha256(head)
                    training[cell][scorer] = {
                        "initial_state_sha256": initial_state_sha,
                        "final_state_sha256": state_sha,
                        "optimizer_steps": 10,
                        "pass_records": [
                            {
                                "pass": pass_index + 1,
                                "mean_all_action_BCE": 0.8 - 0.1 * pass_index,
                                "mean_factual_component_BCE": 0.7 - 0.1 * pass_index,
                                "mean_objective": 0.75 - 0.1 * pass_index,
                                "maximum_preclip_gradient_norm": 1.0,
                                "permutation_sha256": permutation_hashes[pass_index],
                            }
                            for pass_index in range(2)
                        ],
                    }
                    mapping = {
                        "arm": pb21m.ARM_NZ,
                        "active_columns": list(pb21m.pb21j.ACTIVE_COLUMNS[pb21m.ARM_NZ]),
                        "parameter_count": pb21m.pb21j.PRUNED_PARAMETER_COUNTS[pb21m.ARM_NZ],
                        "pruned_state_sha256": state_sha,
                    }
                    pruning[cell][scorer] = {
                        "mapping": mapping,
                        "FIT_inference_transfer": {
                            "padded_logit_sha256": state_sha,
                            "pruned_logit_sha256": state_sha,
                            "maximum_absolute_difference": 0.0,
                        },
                    }
            for scorer_index, scorer in enumerate(pb21m.SCORERS):
                for cell_index, cell in enumerate(pb21m.CELLS):
                    cohort = cell_index // len(pb21m.INIT_SEEDS)
                    fit_label = pb21m.FIT_COHORTS[cohort]
                    tape_a = cal_tapes[pb21m.CAL_A_COHORTS[cohort]]
                    tape_b = cal_tapes[pb21m.CAL_B_COHORTS[cohort]]
                    for fold_index, tape in enumerate((tape_a, tape_b)):
                        sign = 2.0 * tape.hazard_targets.astype(np.float64) - 1.0
                        row_offset = np.linspace(-0.15, 0.15, roots)[:, None]
                        raw[scorer_index, cell_index, fold_index] = (
                            sign * (0.8 + 0.1 * scorer_index) + row_offset
                            + 0.001 * cell_index
                        )
                    fitted_a = pb21m._fit_calibrator(
                        raw[scorer_index, cell_index, 0], tape_a, fit_tapes[fit_label],
                        heads[cell][scorer],
                        manifest_sha256=manifests[pb21m.CAL_A_COHORTS[cohort]],
                        source_bundle_sha256=source_bundle, calibration_role="A",
                    )
                    fitted_b = pb21m._fit_calibrator(
                        raw[scorer_index, cell_index, 1], tape_b, fit_tapes[fit_label],
                        heads[cell][scorer],
                        manifest_sha256=manifests[pb21m.CAL_B_COHORTS[cohort]],
                        source_bundle_sha256=source_bundle, calibration_role="B",
                    )
                    crossfit.extend((fitted_a, fitted_b))
                    logits_a, probabilities_a = pb21m._apply_calibrator(
                        fitted_b, raw[scorer_index, cell_index, 0]
                    )
                    logits_b, probabilities_b = pb21m._apply_calibrator(
                        fitted_a, raw[scorer_index, cell_index, 1]
                    )
                    oof_logits[scorer_index, cell_index] = np.concatenate((logits_a, logits_b))
                    oof_probabilities[scorer_index, cell_index] = np.concatenate(
                        (probabilities_a, probabilities_b)
                    )
                    combined = pb21m.pb21j.FreshLiveTape(
                        beliefs=np.concatenate((tape_a.beliefs, tape_b.beliefs)),
                        updater_inputs=np.concatenate((tape_a.updater_inputs, tape_b.updater_inputs)),
                        hazard_targets=np.concatenate((tape_a.hazard_targets, tape_b.hazard_targets)),
                        parent_raw_logits=np.concatenate((tape_a.parent_raw_logits, tape_b.parent_raw_logits)),
                        factual_actions=np.concatenate((tape_a.factual_actions, tape_b.factual_actions)),
                        episode_group_ids=(*tape_a.episode_group_ids, *tape_b.episode_group_ids),
                        root_state_ids=(*tape_a.root_state_ids, *tape_b.root_state_ids),
                        prior_assignment_count=np.concatenate((tape_a.prior_assignment_count, tape_b.prior_assignment_count)),
                        prior_disagreement_count=np.concatenate((tape_a.prior_disagreement_count, tape_b.prior_disagreement_count)),
                    )
                    final.append(pb21m._fit_calibrator(
                        np.concatenate((raw[scorer_index, cell_index, 0], raw[scorer_index, cell_index, 1])),
                        combined, fit_tapes[fit_label], heads[cell][scorer],
                        manifest_sha256=pb21m.combined_cal_manifest_sha256(
                            manifests[pb21m.CAL_A_COHORTS[cohort]],
                            manifests[pb21m.CAL_B_COHORTS[cohort]],
                        ),
                        source_bundle_sha256=source_bundle, calibration_role="A+B",
                    ))
            self.assertEqual(len(crossfit), 36)
            self.assertEqual(len(final), 18)
            self.assertTrue(all(type(value) is pb21m.PerActionAffineHazardCalibrator for value in (*crossfit, *final)))
            self.assertTrue(all(not hasattr(value, "provenance") for value in (*crossfit, *final)))
            arrays = pb21m.build_cal_evidence_arrays(
                fit_tapes=fit_tapes, cal_tapes=cal_tapes, training=training,
                pruning=pruning, raw=raw, crossfit=crossfit, final=final,
                oof_logits=oof_logits, oof_probabilities=oof_probabilities,
            )
            with TemporaryDirectory() as directory:
                root = Path(directory)
                before = {role: path.exists() for role, path in pb21m.canonical_paths(root).items()}
                evidence = root / "synthetic-production-cal.npz"
                reduced_contract = {
                    name: {"shape": list(value.shape), "dtype": value.dtype.str}
                    for name, value in arrays.items()
                }
                with patch.object(
                    pb21m, "expected_cal_evidence_contract", return_value=reduced_contract
                ):
                    publication = pb21m.publish_evidence_create_only(
                        evidence, arrays, stage="CAL", attempt_sha256="8" * 64
                    )
                    del arrays
                    authoritative = pb21m.reload_evidence(
                        evidence, publication, stage="CAL"
                    )
                pb21m.validate_cal_evidence(
                    authoritative, source_bundle_sha256=source_bundle
                )
                self.assertEqual(set(authoritative), pb21m.CAL_EVIDENCE_KEYS)
                changed = _clone(authoritative)
                changed["fit_ids"][0] = b"XX"
                with self.assertRaisesRegex(ValueError, "canonical axes"):
                    pb21m.validate_cal_evidence(
                        changed, source_bundle_sha256=source_bundle
                    )
                changed = _clone(authoritative)
                changed["factual_action_counts"][0, 0] += 1
                with self.assertRaisesRegex(ValueError, "factual action counts"):
                    pb21m.validate_cal_evidence(
                        changed, source_bundle_sha256=source_bundle
                    )
                changed = _clone(authoritative)
                changed["fit_group_ids"][0, 5] = changed["fit_group_ids"][0, 0]
                with self.assertRaisesRegex(ValueError, "episode-group geometry"):
                    pb21m.validate_cal_evidence(
                        changed, source_bundle_sha256=source_bundle
                    )
                changed = _clone(authoritative)
                namespaces = list(pb21m._decode_strings(
                    changed["calibration_source_namespace"]
                ))
                namespaces[0] = "maze_chase.pb21m.train-only.B.v1"
                changed["calibration_source_namespace"] = pb21m._byte_strings(
                    namespaces, 64
                )
                with self.assertRaisesRegex(ValueError, "role order"):
                    pb21m.validate_cal_evidence(
                        changed, source_bundle_sha256=source_bundle
                    )
                changed = _clone(authoritative)
                changed["oof_calibrated_logits"][0, 0, 0, 0] += 0.25
                changed["oof_calibrated_probabilities"] = 1.0 / (
                    1.0 + np.exp(-changed["oof_calibrated_logits"])
                )
                with self.assertRaisesRegex(ValueError, "fit-half|direction"):
                    pb21m.validate_cal_evidence(
                        changed, source_bundle_sha256=source_bundle
                    )
                refit = pb21m.deterministic_refit_from_reloaded_cal(
                    authoritative, manifests=manifests,
                    source_bundle_sha256=source_bundle,
                )
                decision = pb21m.evaluate_stage(authoritative, stage="CAL")
                self.assertEqual(refit["crossfit_refits"], 36)
                self.assertEqual(refit["final_refits"], 18)
                self.assertTrue(refit["passed"])
                self.assertEqual(decision["stage"], "CAL")
                json.dumps(decision, allow_nan=False)
                after = {role: path.exists() for role, path in pb21m.canonical_paths(root).items()}
                self.assertEqual(after, before)

    def test_cal_schema_excludes_every_dev_field(self) -> None:
        dev_only = {
            "cal_evidence_sha256", "cal_decision_sha256", "dev_open_sha256",
            "dev_targets", "dev_actions", "dev_root_ids", "dev_cluster_ordinal",
            "dev_cluster_ids", "dev_raw_logits", "dev_raw_probabilities",
            "dev_calibrated_logits", "dev_calibrated_probabilities",
        }
        self.assertTrue(pb21m.CAL_EVIDENCE_KEYS.isdisjoint(dev_only))
        self.assertTrue(dev_only.issubset(pb21m.DEV_EVIDENCE_KEYS))
        self.assertNotEqual(
            pb21m._key_set_sha256(pb21m.CAL_EVIDENCE_KEYS, "CAL"),
            pb21m._key_set_sha256(pb21m.DEV_EVIDENCE_KEYS, "DEV"),
        )

    def test_crossfit_direction_axes_encode_opposite_fold_evaluation(self) -> None:
        bundle = "b" * 64
        arrays = _mini_cal_arrays(bundle)
        self.assertEqual(arrays["crossfit_fit_fold_index"].tolist(), [0, 1])
        self.assertEqual(arrays["crossfit_eval_fold_index"].tolist(), [1, 0])
        self.assertEqual(arrays["oof_row_eval_fold_index"].tolist(), [0, 0, 1, 1])
        raw = arrays["cal_raw_logits"]
        scales = arrays["crossfit_scales"]
        biases = arrays["crossfit_biases"]
        expected = np.concatenate((
            raw[:, :, 0] * scales[:, :, 1, None, :] + biases[:, :, 1, None, :],
            raw[:, :, 1] * scales[:, :, 0, None, :] + biases[:, :, 0, None, :],
        ), axis=2)
        self.assertTrue(np.array_equal(expected, arrays["oof_calibrated_logits"]))

    def test_dev_evidence_binds_cal_decision_open_and_final_calibrator(self) -> None:
        cal_sha, decision_sha, open_sha = "a" * 64, "d" * 64, "e" * 64
        cal_arrays = {
            "scorer_ids": pb21m._byte_strings(pb21m.SCORERS, 12),
            "cell_ids": pb21m._byte_strings(pb21m.CELLS, 16),
            "cell_fit_index": np.asarray((0, 0, 0, 1, 1, 1, 2, 2, 2), dtype=np.int16),
            "cell_init_seed": np.asarray(tuple(pb21m.INIT_SEEDS) * 3, dtype=np.int32),
            "fit_targets": np.zeros((3, 2, 5), dtype=np.float32),
            "fit_actions": np.zeros((3, 2), dtype=np.int64),
            "fit_root_ids": np.stack([
                pb21m._byte_strings((f"FIT:{index}:root",), 64) for index in range(3)
            ]),
            "fit_group_ids": np.stack([
                pb21m._byte_strings((f"FIT:{index}:group",), 64) for index in range(3)
            ]),
            "cal_root_ids": np.stack([
                pb21m._byte_strings((f"CAL:{index}:root",), 64) for index in range(3)
            ]),
            "cal_cluster_ids": np.stack([
                pb21m._byte_strings((f"CAL:{index}:group",), 64) for index in range(3)
            ]),
            "final_scales": np.ones((2, 9, 5), dtype=np.float64),
            "final_biases": np.zeros((2, 9, 5), dtype=np.float64),
            "final_accepted": np.ones((2, 9), dtype=np.bool_),
        }
        raw = np.linspace(-1, 1, 2 * 9 * 2 * 5).reshape(2, 9, 2, 5)
        logits = raw.copy()
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        bootstrap = np.zeros((3, 2), dtype=np.uint16)
        derangements = np.asarray([[1, 0], [1, 0]], dtype=np.uint16)
        with patch.multiple(
            pb21m, ROOTS_PER_EPISODE=1, DEV_ROOTS=2, DEV_EPISODES=2,
            BOOTSTRAP_RESAMPLES=3, DERANGEMENT_REPETITIONS=2,
        ), patch.object(pb21m, "partition_manifests", return_value=_mini_manifests()), patch.object(
            pb21m, "dev_bootstrap_indices", return_value=bootstrap
        ), patch.object(
            pb21m, "dev_episode_derangements", return_value=derangements
        ), patch.object(
            pb21m, "validate_cal_decision_receipt", return_value={
                "sha256": decision_sha,
                "payload": {"passed": True, "decision": {"passed": True},
                            "decision_sha256": pb21m._decision_sha256({"passed": True})},
            }
        ), patch.object(
            pb21m, "validate_dev_open_receipt", return_value={"sha256": open_sha}
        ), patch.object(
            pb21m, "validate_canonical_cal_evidence_file",
            return_value={"sha256": cal_sha},
        ), patch.object(
            pb21m, "evaluate_stage", return_value={"passed": True}
        ):
            arrays = pb21m.build_dev_evidence_arrays(
                cal_arrays=cal_arrays, cal_evidence_sha256=cal_sha,
                cal_decision_sha256=decision_sha, dev_open_sha256=open_sha,
                dev_tape=_tape("DEV"), raw_logits=raw,
                calibrated_logits=logits, calibrated_probabilities=probabilities,
                latency_measurements_ms=np.zeros((2, 9, 256), dtype=np.float64),
            )
            self.assertEqual(set(arrays), pb21m.DEV_EVIDENCE_KEYS)
            pb21m.validate_dev_evidence(
                arrays, cal_arrays=cal_arrays, cal_evidence_sha256=cal_sha,
                project_root=Path("unused"), attempt_sha256="1" * 64,
            )
            mismatched_decision = {"passed": True, "gate": "different"}
            with patch.object(pb21m, "validate_cal_decision_receipt", return_value={
                "sha256": decision_sha,
                "payload": {"passed": True, "decision": mismatched_decision,
                            "decision_sha256": pb21m._decision_sha256(mismatched_decision)},
            }), self.assertRaisesRegex(ValueError, "exact passing authoritative CAL decision"):
                pb21m.validate_dev_evidence(
                    arrays, cal_arrays=cal_arrays, cal_evidence_sha256=cal_sha,
                    project_root=Path("unused"), attempt_sha256="1" * 64,
                )
            for mutate, message in (
                (lambda value: value["dev_targets"].__setitem__((0, 0), 0.5), "exactly binary"),
                (lambda value: value["dev_actions"].__setitem__(0, pb21m.ACTION_COUNT), "semantic action"),
                (lambda value: value["dev_cluster_ids"].__setitem__(1, value["dev_cluster_ids"][0]),
                 "cluster geometry"),
            ):
                changed = _clone(arrays); mutate(changed)
                with self.assertRaisesRegex(ValueError, message):
                    pb21m.validate_dev_evidence(
                        changed, cal_arrays=cal_arrays, cal_evidence_sha256=cal_sha,
                        project_root=Path("unused"), attempt_sha256="1" * 64,
                    )
            for name, expected, message in (
                ("cal_evidence_sha256", "f" * 64, "CAL evidence"),
                ("cal_decision_sha256", "f" * 64, "CAL decision"),
                ("dev_open_sha256", "f" * 64, "DEV-open"),
            ):
                changed = _clone(arrays)
                changed[name] = np.asarray(expected.encode("ascii"), dtype="S64")
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                    pb21m.validate_dev_evidence(
                        changed, cal_arrays=cal_arrays, cal_evidence_sha256=cal_sha,
                        project_root=Path("unused"), attempt_sha256="1" * 64,
                    )
            changed = _clone(arrays)
            changed["final_scales"][0, 0, 0] += 1
            with self.assertRaisesRegex(ValueError, "cross-link"):
                pb21m.validate_dev_evidence(
                    changed, cal_arrays=cal_arrays, cal_evidence_sha256=cal_sha,
                    project_root=Path("unused"), attempt_sha256="1" * 64,
                )

    def test_receipts_are_create_only_and_dev_open_requires_passing_cal(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            decision = {"passed": True, "gate": "synthetic"}
            receipt = pb21m.publish_cal_decision(
                root, attempt_sha256="1" * 64, cal_evidence_sha256="2" * 64,
                decision=decision, resource_snapshot={"passed": True},
            )
            self.assertEqual(receipt["payload"]["decision_sha256"], pb21m._decision_sha256(decision))
            opened = pb21m.publish_dev_open(
                root, attempt_sha256="1" * 64, cal_evidence_sha256="2" * 64,
                resource_snapshot={"passed": True},
            )
            self.assertEqual(opened["payload"]["cal_decision_sha256"], receipt["sha256"])
            with self.assertRaises(FileExistsError):
                pb21m.publish_dev_open(
                    root, attempt_sha256="1" * 64, cal_evidence_sha256="2" * 64,
                    resource_snapshot={"passed": True},
                )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            failed = pb21m.publish_cal_decision(
                root, attempt_sha256="1" * 64, cal_evidence_sha256="2" * 64,
                decision={"passed": False}, resource_snapshot={"passed": True},
            )
            with self.assertRaisesRegex(RuntimeError, "passing sealed CAL"):
                pb21m.publish_dev_open(
                    root, attempt_sha256="1" * 64, cal_evidence_sha256="2" * 64,
                    resource_snapshot={"passed": True},
                )
            self.assertFalse(pb21m.canonical_paths(root)["dev_open"].exists())
        with TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = pb21m.publish_cal_decision(
                root, attempt_sha256="1" * 64, cal_evidence_sha256="2" * 64,
                decision={"passed": True, "gate": "original"},
                resource_snapshot={"passed": True},
            )
            path = Path(receipt["path"])
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["decision"]["gate"] = "tampered"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reload/hash/lineage"):
                pb21m.validate_cal_decision_receipt(
                    root, attempt_sha256="1" * 64, cal_evidence_sha256="2" * 64
                )

    def test_dev_validation_rehashes_canonical_cal_evidence_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = pb21m.canonical_paths(root)["cal_evidence"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"sealed-CAL-evidence")
            digest = pb21m._sha256_file(path)
            record = pb21m.validate_canonical_cal_evidence_file(
                root, expected_sha256=digest
            )
            self.assertEqual(record["sha256"], digest)
            path.write_bytes(b"tampered-CAL-evidence")
            with self.assertRaisesRegex(ValueError, "hash drifted"):
                pb21m.validate_canonical_cal_evidence_file(
                    root, expected_sha256=digest
                )

    def test_evidence_writer_is_deterministic_and_create_only(self) -> None:
        arrays = {name: np.asarray(index, dtype=np.int64)
                  for index, name in enumerate(sorted(pb21m.PREFLIGHT_EVIDENCE_KEYS))}
        with TemporaryDirectory() as directory:
            left = Path(directory) / "left.npz"
            right = Path(directory) / "right.npz"
            one = pb21m.publish_evidence_create_only(
                left, arrays, stage="PREFLIGHT", attempt_sha256="1" * 64
            )
            two = pb21m.publish_evidence_create_only(
                right, arrays, stage="PREFLIGHT", attempt_sha256="1" * 64
            )
            self.assertEqual(one["sha256"], two["sha256"])
            reloaded = pb21m.reload_evidence(left, one, stage="PREFLIGHT")
            self.assertEqual(set(reloaded), pb21m.PREFLIGHT_EVIDENCE_KEYS)
            with self.assertRaises(FileExistsError):
                pb21m.publish_evidence_create_only(
                    left, arrays, stage="PREFLIGHT", attempt_sha256="1" * 64
                )

    def _exercise_cal_branch(self, passed: bool) -> list[tuple[str, ...]]:
        calls: list[tuple[str, ...]] = []
        source_bundle = {"sha256": "s" * 64}
        registration = {
            "payload": {"source_bundle": source_bundle, "scientific_parent": {}},
            "sha256": "r" * 64,
        }
        attempt = {"sha256": "a" * 64}
        parent = torch.nn.Linear(1, 1)

        def sources(labels):
            labels = tuple(labels)
            calls.append(labels)
            return {label: label for label in labels}

        def collect(_model, source, *, partition_label):
            self.assertEqual(source, partition_label)
            return _tape(partition_label)

        fake_calibrator = SimpleNamespace(
            scales=np.ones(5, dtype=np.float64), biases=np.zeros(5, dtype=np.float64),
            accepted=True,
        )
        cal_arrays = {
            "final_scales": np.ones((2, 9, 5), dtype=np.float64),
            "final_biases": np.zeros((2, 9, 5), dtype=np.float64),
        }
        cal_publication = {"sha256": "c" * 64, "byte_length": 1}
        dev_publication = {"sha256": "v" * 64, "byte_length": 1}
        cal_receipt = {"sha256": "d" * 64, "payload": {"passed": passed}}
        dev_open = {"sha256": "o" * 64, "payload": {"passed": True}}

        with ExitStack() as stack:
            stack.enter_context(patch.object(pb21m, "apply_deterministic_mode"))
            stack.enter_context(patch.object(pb21m.torch, "set_num_threads"))
            stack.enter_context(patch.object(pb21m.torch, "set_num_interop_threads"))
            stack.enter_context(patch.object(pb21m.torch, "get_num_threads", return_value=1))
            stack.enter_context(patch.object(pb21m, "_source_bundle", return_value=source_bundle))
            stack.enter_context(patch.object(pb21m, "resource_guard", side_effect=lambda _path, *, phase: {"phase": phase, "passed": True}))
            stack.enter_context(patch.object(pb21m, "partition_manifests", return_value=_mini_manifests()))
            stack.enter_context(patch.object(pb21m, "pretraining_support_report", return_value={"passed": True}))
            stack.enter_context(patch.object(pb21m, "partition_sources", side_effect=sources))
            stack.enter_context(patch.object(pb21m, "collect_fresh_live_tape", side_effect=collect))
            stack.enter_context(patch.object(pb21m, "deterministic_permutations", return_value=()))
            stack.enter_context(patch.object(pb21m, "initialized_head", side_effect=lambda _seed: torch.nn.Linear(1, 1)))
            stack.enter_context(patch.object(pb21m, "_state_sha256", return_value="h" * 64))
            stack.enter_context(patch.object(pb21m, "_features", side_effect=lambda tape: np.zeros((len(tape.hazard_targets), 1), dtype=np.float32)))
            stack.enter_context(patch.object(pb21m.pb21l, "train_upmix_head", return_value={"initial_state_sha256": "i" * 64, "final_state_sha256": "f" * 64, "optimizer_steps": 1}))
            stack.enter_context(patch.object(pb21m.pb21j, "prune_head", side_effect=lambda head, _arm: (head, {"pruned_state_sha256": "p" * 64})))
            stack.enter_context(patch.object(pb21m.pb21j, "pruning_equivalence_audit", return_value={"passed": True}))
            stack.enter_context(patch.object(pb21m.pb21j, "compact_features", side_effect=lambda features, _arm: features))
            stack.enter_context(patch.object(pb21m.pb21j, "raw_logit_table", side_effect=lambda _head, features, batch_size: np.zeros((len(features), 5), dtype=np.float64)))
            stack.enter_context(patch.object(pb21m, "_fit_calibrator", return_value=fake_calibrator))
            stack.enter_context(patch.object(pb21m, "build_cal_evidence_arrays", return_value={"cal": np.asarray(1)}))
            stack.enter_context(patch.object(pb21m, "publish_evidence_create_only", side_effect=lambda _path, _arrays, *, stage, attempt_sha256: cal_publication if stage == "CAL" else dev_publication))
            stack.enter_context(patch.object(pb21m, "reload_evidence", side_effect=lambda _path, _publication, *, stage: cal_arrays if stage == "CAL" else {"dev": np.asarray(1), "latency_measurements_ms": np.zeros((2, 9, 256))}))
            stack.enter_context(patch.object(pb21m, "validate_cal_evidence"))
            stack.enter_context(patch.object(pb21m, "deterministic_refit_from_reloaded_cal", return_value={"passed": True}))
            stack.enter_context(patch.object(pb21m, "evaluate_stage", side_effect=lambda _arrays, *, stage: {"passed": passed if stage == "CAL" else True}))
            stack.enter_context(patch.object(pb21m, "publish_cal_decision", return_value=cal_receipt))
            stack.enter_context(patch.object(pb21m, "publish_dev_open", return_value=dev_open))
            stack.enter_context(patch.object(pb21m, "build_dev_evidence_arrays", return_value={"dev": np.asarray(1)}))
            stack.enter_context(patch.object(pb21m, "validate_dev_evidence"))
            stack.enter_context(patch.object(pb21m, "latency_probe", return_value=np.zeros((2, 9, 256))))
            stack.enter_context(patch.object(pb21m, "latency_report", return_value={"passed": True}))
            stack.enter_context(patch.object(pb21m, "_publish_result"))
            stack.enter_context(patch.multiple(pb21m, ROOTS_PER_CAL_FOLD=2, ROOTS_PER_CAL=4, DEV_ROOTS=2))
            pb21m._run_impl(
                output=Path("unused-result.json"), registration=registration, attempt=attempt,
                preloaded_parent=(parent, {}, {}),
            )
        return calls

    def test_cal_failure_never_constructs_dev_and_pass_constructs_it_once(self) -> None:
        failed_calls = self._exercise_cal_branch(False)
        self.assertFalse(any("DEV" in labels for labels in failed_calls))
        passed_calls = self._exercise_cal_branch(True)
        self.assertEqual(sum(labels == ("DEV",) for labels in passed_calls), 1)

    def test_full_bootstrap_geometry_without_allocating_full_matrices(self) -> None:
        requested: list[tuple[int, ...]] = []

        class ShapeOnlyRNG:
            def integers(self, low, high, *, size, dtype):
                self_bounds = (low, high, np.dtype(dtype))
                self.bounds = self_bounds
                requested.append(tuple(size))
                return np.broadcast_to(np.zeros(1, dtype=dtype), size)

        rngs = [ShapeOnlyRNG(), ShapeOnlyRNG()]
        with patch.object(pb21m.np.random, "default_rng", side_effect=rngs):
            cal = pb21m.cal_bootstrap_indices()
            dev = pb21m.dev_bootstrap_indices()
        self.assertEqual(cal.shape, (50_000, 3, 2, 256))
        self.assertEqual(dev.shape, (50_000, 768))
        self.assertEqual(cal.dtype, np.dtype(np.uint16))
        self.assertEqual(dev.dtype, np.dtype(np.uint16))
        self.assertEqual(cal.nbytes, 153_600_000)
        self.assertEqual(dev.nbytes, 76_800_000)
        self.assertFalse(cal.flags.owndata)
        self.assertFalse(dev.flags.owndata)
        self.assertEqual(requested, [(50_000, 3, 2, 256), (50_000, 768)])

    def test_full_derangement_geometry_is_deterministic_and_fixed_point_free(self) -> None:
        cal = pb21m.cal_episode_derangements()
        dev = pb21m.dev_episode_derangements()
        self.assertEqual(cal.shape, (20, 3, 2, 256))
        self.assertEqual(dev.shape, (20, 768))
        self.assertEqual(cal.dtype, np.dtype(np.uint16))
        self.assertEqual(dev.dtype, np.dtype(np.uint16))
        self.assertTrue(np.array_equal(cal, pb21m.cal_episode_derangements()))
        self.assertTrue(np.array_equal(dev, pb21m.dev_episode_derangements()))
        identity_cal = np.arange(256, dtype=np.uint16)
        identity_dev = np.arange(768, dtype=np.uint16)
        self.assertFalse(np.any(cal == identity_cal))
        self.assertFalse(np.any(dev == identity_dev))
        self.assertTrue(np.array_equal(np.sort(cal, axis=-1), np.broadcast_to(identity_cal, cal.shape)))
        self.assertTrue(np.array_equal(np.sort(dev, axis=-1), np.broadcast_to(identity_dev, dev.shape)))
        flattened = pb21m._flatten_cal_derangements(cal, cohort=1)
        self.assertEqual(flattened.shape, (20, 512))
        self.assertTrue(np.all(flattened[:, :256] < 256))
        self.assertTrue(np.all(flattened[:, 256:] >= 256))

    def test_pretraining_geometry_rejects_root_alias_and_malformed_groups(self) -> None:
        fit = {
            label: _clustered_tape(label, roots=100, roots_per_episode=5, offset=index)
            for index, label in enumerate(pb21m.FIT_COHORTS)
        }
        labels = (*pb21m.CAL_A_COHORTS, *pb21m.CAL_B_COHORTS)
        cal = {
            label: _clustered_tape(label, roots=100, roots_per_episode=5, offset=10 + index)
            for index, label in enumerate(labels)
        }
        with patch.multiple(
            pb21m, ROOTS_PER_EPISODE=5, ROOTS_PER_FIT=100, FIT_EPISODES=20,
            ROOTS_PER_CAL_FOLD=100, CAL_FOLD_EPISODES=20,
        ), patch.object(pb21m.pb21l, "ROOTS_PER_FIT", 100):
            good = pb21m.pretraining_support_report(fit, cal)
            self.assertTrue(good["passed"])
            roots = list(fit["F0"].root_state_ids); roots[1] = roots[0]
            aliased_fit = dict(fit); aliased_fit["F0"] = replace(fit["F0"], root_state_ids=tuple(roots))
            aliased = pb21m.pretraining_support_report(aliased_fit, cal)
            self.assertFalse(aliased["FIT"]["F0"]["geometry"]["unique_root_identity_geometry"])
            groups = list(cal["C0A"].episode_group_ids); groups[5] = groups[0]
            malformed_cal = dict(cal); malformed_cal["C0A"] = replace(
                cal["C0A"], episode_group_ids=tuple(groups)
            )
            malformed = pb21m.pretraining_support_report(fit, malformed_cal)
            self.assertFalse(malformed["CAL_folds"]["C0A"]["geometry"]["episode_group_geometry"])
            targets = fit["F0"].hazard_targets.copy(); targets[:, 0] = 0.0
            degenerate_fit = dict(fit); degenerate_fit["F0"] = replace(
                fit["F0"], hazard_targets=targets
            )
            degenerate = pb21m.pretraining_support_report(degenerate_fit, cal)
            self.assertFalse(degenerate["FIT"]["F0"]["passed"])
            self.assertTrue(any(
                not value["passed"]
                for value in degenerate["FIT"]["F0"]["domain_priors"].values()
            ))

    @unittest.skipUnless(sys.platform == "win32", "Windows ABI contract")
    def test_unmocked_windows_resource_guard_uses_pointer_width_safe_handles(self) -> None:
        with TemporaryDirectory() as directory:
            code = (
                "import importlib.util,json,pathlib,sys,torch;"
                f"p=pathlib.Path({str(SCRIPT)!r});"
                "sys.path[:0]=[str(p.parent),str(p.parents[1]/'src')];"
                "s=importlib.util.spec_from_file_location('pb21m_guard_subprocess',p);"
                "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
                "m.apply_deterministic_mode();torch.set_num_threads(1);torch.set_num_interop_threads(1);"
                f"print(json.dumps(m.resource_guard(pathlib.Path({directory!r})/'evidence.npz',phase='unmocked_windows_ABI')))"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=30
            )
        snapshot = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertGreater(snapshot["process_working_set_bytes"], 0)
        self.assertGreater(snapshot["available_physical_RAM_bytes"], 0)
        self.assertEqual(snapshot["torch_threads"], 1)
        self.assertEqual(snapshot["torch_interop_threads"], 1)
        self.assertTrue(snapshot["passed"])


if __name__ == "__main__":
    unittest.main()
