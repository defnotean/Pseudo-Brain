"""Focused contracts for strict-live V2.1i component localization v1."""
from __future__ import annotations

from hashlib import sha256
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import v21i_strict_live_component_localization_v1 as probe


def _tape(
    *, roots: int = 8, roots_per_episode: int = 2
) -> probe.strict.StrictLiveFeatureTape:
    rng = np.random.default_rng(117)
    updater = rng.normal(size=(roots, probe.CAPACITY_STATE_WIDTH))
    updater[:, 360:365] = np.eye(5)[np.arange(roots) % 5]
    updater[:, 365] = 1.0
    targets = np.asarray(
        [[(root + action) % 2 for action in range(5)] for root in range(roots)],
        dtype=np.float64,
    )
    return probe.strict.StrictLiveFeatureTape(
        beliefs=rng.normal(size=(roots, probe.WIDTH)),
        updater_inputs=updater,
        hazard_targets=targets,
        parent_raw_logits=rng.normal(size=(roots, 5)),
        episode_group_ids=tuple(
            f"DEV:episode:{root // roots_per_episode}" for root in range(roots)
        ),
        root_state_ids=tuple(
            sha256(f"component-root:{root}".encode()).hexdigest()
            for root in range(roots)
        ),
        history=probe.strict.InterventionHistory(
            prior_assignment_count=np.arange(roots, dtype=np.int64),
            prior_disagreement_count=np.arange(roots, dtype=np.int64),
        ),
    )


def _gate_map(*eligible: str, updater_pass: bool = True, u0_pass: bool = True):
    result = {arm: {"passed": arm in eligible} for arm in probe.ARMS}
    result[probe.ARM_U]["passed"] = updater_pass
    result[probe.ARM_U0]["passed"] = u0_pass
    return result


def _bootstrap_map(*eligible: str):
    comparisons = {}
    for arm in probe.REDUCED_ARMS:
        passes = arm in eligible
        comparisons[arm] = {
            "minus_N": {
                "simultaneous_one_sided_lower": 0.01 if passes else -0.01
            },
            "minus_U0": {
                "simultaneous_one_sided_lower": -0.01 if passes else -0.04
            },
            "minus_U": {
                "simultaneous_one_sided_lower": -0.01 if passes else -0.04
            },
        }
    return {"comparisons": comparisons}


class StrictLiveComponentLocalizationTests(unittest.TestCase):
    def test_exact_eleven_arms_and_eight_reduced_arms_are_frozen(self) -> None:
        self.assertEqual(
            probe.ARMS,
            (
                "N", "U", "U0", "B", "Z", "E", "A", "BZ", "BE", "ZE", "BZE"
            ),
        )
        self.assertEqual(len(probe.REDUCED_ARMS), 8)
        self.assertEqual(
            probe.FIXED_TIE_ORDER,
            ("Z", "E", "B", "ZE", "BZ", "BE", "BZE"),
        )
        self.assertNotIn(probe.ARM_A, probe.PRODUCTION_ARMS)

    def test_feature_slots_match_exact_registered_layout(self) -> None:
        tape = _tape()
        zero = np.zeros_like(tape.updater_inputs)
        expected = {
            probe.ARM_N: zero.copy(),
            probe.ARM_U: tape.updater_inputs.copy(),
            probe.ARM_U0: tape.updater_inputs.copy(),
            probe.ARM_B: zero.copy(),
            probe.ARM_Z: zero.copy(),
            probe.ARM_E: zero.copy(),
            probe.ARM_A: zero.copy(),
            probe.ARM_BZ: zero.copy(),
            probe.ARM_BE: zero.copy(),
            probe.ARM_ZE: zero.copy(),
            probe.ARM_BZE: zero.copy(),
        }
        expected[probe.ARM_N][:, :120] = tape.beliefs
        expected[probe.ARM_U0][:, 365] = 0.0
        expected[probe.ARM_B][:, :120] = tape.updater_inputs[:, :120]
        expected[probe.ARM_Z][:, 120:240] = tape.updater_inputs[:, 120:240]
        expected[probe.ARM_E][:, 240:360] = tape.updater_inputs[:, 240:360]
        expected[probe.ARM_A][:, 360:365] = tape.updater_inputs[:, 360:365]
        expected[probe.ARM_BZ][:, :240] = tape.updater_inputs[:, :240]
        expected[probe.ARM_BE][:, :120] = tape.updater_inputs[:, :120]
        expected[probe.ARM_BE][:, 240:360] = tape.updater_inputs[:, 240:360]
        expected[probe.ARM_ZE][:, 120:360] = tape.updater_inputs[:, 120:360]
        expected[probe.ARM_BZE][:, :360] = tape.updater_inputs[:, :360]
        for arm in probe.ARMS:
            actual = probe.component_features(tape, arm)
            np.testing.assert_array_equal(actual, expected[arm].astype(np.float32))
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(actual.flags.c_contiguous)
            self.assertTrue(np.isfinite(actual).all())
            if arm != probe.ARM_U:
                np.testing.assert_array_equal(actual[:, 365], 0.0)
                self.assertFalse(np.signbit(actual[:, 365]).any())

    def test_pending_must_be_constant_one_and_is_not_localized(self) -> None:
        tape = _tape()
        tape.updater_inputs[0, 365] = 0.0
        with self.assertRaisesRegex(RuntimeError, "P must be exactly one"):
            probe.component_features(tape, probe.ARM_B)
        self.assertFalse(any(arm == "P" or "P" in arm for arm in probe.REDUCED_ARMS))

    def test_u_and_u0_projection_never_alias_or_mutate_float32_tape(self) -> None:
        tape = _tape()
        tape.updater_inputs[:] = tape.updater_inputs.astype(np.float32)
        float32_tape = probe.strict.StrictLiveFeatureTape(
            beliefs=tape.beliefs.astype(np.float32),
            updater_inputs=tape.updater_inputs.astype(np.float32),
            hazard_targets=tape.hazard_targets,
            parent_raw_logits=tape.parent_raw_logits,
            episode_group_ids=tape.episode_group_ids,
            root_state_ids=tape.root_state_ids,
            history=tape.history,
        )
        before = float32_tape.updater_inputs.copy()
        full = probe.component_features(float32_tape, probe.ARM_U)
        pending_zero = probe.component_features(float32_tape, probe.ARM_U0)
        np.testing.assert_array_equal(float32_tape.updater_inputs, before)
        np.testing.assert_array_equal(full, before)
        np.testing.assert_array_equal(pending_zero[:, :365], before[:, :365])
        np.testing.assert_array_equal(pending_zero[:, 365], 0.0)
        self.assertFalse(np.shares_memory(full, float32_tape.updater_inputs))
        self.assertFalse(np.shares_memory(pending_zero, float32_tape.updater_inputs))

    def test_targets_and_provenance_never_change_features(self) -> None:
        first = _tape()
        second = probe.strict.StrictLiveFeatureTape(
            beliefs=first.beliefs.copy(),
            updater_inputs=first.updater_inputs.copy(),
            hazard_targets=1.0 - first.hazard_targets,
            parent_raw_logits=first.parent_raw_logits + 100.0,
            episode_group_ids=tuple(reversed(first.episode_group_ids)),
            root_state_ids=tuple(reversed(first.root_state_ids)),
            history=probe.strict.InterventionHistory(
                prior_assignment_count=first.history.prior_assignment_count + 50,
                prior_disagreement_count=first.history.prior_disagreement_count + 50,
            ),
        )
        for arm in probe.ARMS:
            np.testing.assert_array_equal(
                probe.component_features(first, arm),
                probe.component_features(second, arm),
            )

    def test_all_heads_are_capacity_and_byte_initialization_matched(self) -> None:
        heads = probe.initialized_heads()
        self.assertEqual(set(heads), set(probe.ARMS))
        states = list(heads.values())
        for head in states:
            self.assertEqual(
                sum(parameter.numel() for parameter in head.parameters()),
                262_801,
            )
        for left, right in zip(states[:-1], states[1:], strict=True):
            for left_value, right_value in zip(
                left.state_dict().values(), right.state_dict().values(), strict=True
            ):
                self.assertTrue(torch.equal(left_value, right_value))
        self.assertEqual(
            probe._state_dict_sha256(states[0].state_dict()),
            probe.EXPECTED_INITIAL_STATE_SHA256,
        )

    def test_schedule_has_exact_fit_budget_and_no_sealed_split(self) -> None:
        schedule = probe.schedule_record()
        self.assertEqual(schedule["passes_per_arm"], 32)
        self.assertEqual(schedule["roots_per_pass"], 1536)
        self.assertEqual(schedule["root_batch_size"], 24)
        self.assertEqual(schedule["steps_per_pass"], 64)
        self.assertEqual(schedule["optimizer_steps_per_arm"], 2048)
        self.assertEqual(schedule["learning_rate"], 1.0e-3)
        self.assertEqual(schedule["weight_decay"], 1.0e-4)
        self.assertEqual(schedule["clip_norm"], 1.0)
        self.assertTrue(schedule["fit_all_arms_before_any_DEV_prediction_or_metric"])
        self.assertFalse(schedule["train_cal_constructed"])
        self.assertFalse(schedule["cpu_qual_opened"])
        self.assertFalse(schedule["test_split_opened"])
        self.assertFalse(schedule["early_stopping"])
        self.assertFalse(schedule["retry_allowed"])
        self.assertEqual(
            schedule["bootstrap"]["expected_DEV_geometry"],
            {
                "episode_clusters": 64,
                "roots_per_episode_cluster": 12,
                "counterfactual_action_branches_per_root": 5,
            },
        )

    def test_run_source_fits_every_arm_before_any_probability_endpoint(self) -> None:
        source = inspect.getsource(probe._run_impl)
        training_loop = source.index("for arm in ARMS:")
        training_call = source.index("strict.train_probe_head", training_loop)
        first_probability = source.index("strict.probability_table")
        self.assertLess(training_call, first_probability)
        self.assertIn("No DEV metric or prediction", source)

    def test_control_constants_are_exactly_pinned(self) -> None:
        self.assertEqual(
            probe.EXPECTED_CONTROL[probe.ARM_N]["final_state_sha256"],
            "dd3aa2792773c3ecb8ae266d0ac3dc9927cd76ead35bd2c33e4d653fe3ea13c1",
        )
        self.assertEqual(
            probe.EXPECTED_CONTROL[probe.ARM_U]["DEV_probability_sha256"],
            "6f4a3e8578b4294d5a48adf9b33644cd91b88bf0e8923b9b676ee635eeac5647",
        )
        self.assertEqual(
            probe.EXPECTED_CONTROL[probe.ARM_N]["DEV_aggregate_roc_auc"],
            0.6040297854619737,
        )
        self.assertEqual(
            probe.EXPECTED_CONTROL[probe.ARM_U]["DEV_aggregate_roc_auc"],
            0.7321055105872505,
        )
        self.assertEqual(probe.EXPECTED_PRIOR_DEV_AGGREGATE_ROC_AUC, 0.5380051490468308)

    def test_control_failure_forbids_interpretation_and_selection(self) -> None:
        controls = {"passed": False}
        result = probe.select_component(
            _gate_map(probe.ARM_Z), _bootstrap_map(probe.ARM_Z), controls
        )
        self.assertEqual(result["diagnosis"], "control_reproduction_failure")
        self.assertFalse(result["scientific_interpretation_allowed"])
        self.assertIsNone(result["selected_arm"])

    def test_eight_arm_bonferroni_iut_and_margin_are_exact(self) -> None:
        contract = probe.selection_contract()
        self.assertEqual(contract["inference"]["reduced_arm_count"], 8)
        self.assertEqual(contract["inference"]["one_sided_lower_quantile"], 0.00625)
        self.assertEqual(contract["inference"]["noninferiority_margin"], 0.025)
        self.assertFalse(contract["inference"]["confirmatory_nominal_FWER_claimed"])
        self.assertTrue(contract["inference"]["fresh_DEV_required_for_confirmation"])
        self.assertAlmostEqual(
            contract["inference"]["approximate_retention_of_known_U_minus_N_gain"],
            0.805,
            places=3,
        )
        self.assertIn("slot-confounded", contract["native_slot_initialization_caveat"])
        self.assertIn("not_unique_causality", contract["claim_scope"])
        self.assertIn("multiple_fixed_initializations", contract["fresh_production_form_requirements"])
        self.assertIn("incremental explicit A onehot", contract["prediction_error_action_caveat"])
        result = probe.select_component(
            _gate_map(probe.ARM_Z),
            _bootstrap_map(probe.ARM_Z),
            {"passed": True},
        )
        self.assertTrue(result["eligibility"][probe.ARM_Z]["eligible"])
        self.assertEqual(result["selected_arm"], probe.ARM_Z)
        self.assertFalse(result["architecture_change_supported"])
        self.assertTrue(result["next_production_form_probe_nominated"])
        self.assertFalse(result["unique_component_causality_claimed"])

    def test_u0_is_required_and_exact_u_remains_a_direct_ni_comparator(self) -> None:
        result = probe.select_component(
            _gate_map(probe.ARM_Z, u0_pass=False),
            _bootstrap_map(probe.ARM_Z),
            {"passed": True},
        )
        self.assertEqual(
            result["diagnosis"], "pending_intercept_optimization_confounded"
        )
        self.assertFalse(result["scientific_interpretation_allowed"])
        self.assertIsNone(result["selected_arm"])
        source = inspect.getsource(probe.select_component)
        self.assertIn('comparisons[arm]["minus_U0"]', source)
        self.assertIn('comparisons[arm]["minus_U"]', source)

    def test_all_three_iut_limbs_use_the_same_bound_without_alpha_split(self) -> None:
        bootstrap = _bootstrap_map(probe.ARM_Z)
        bootstrap["comparisons"][probe.ARM_Z]["minus_U"][
            "simultaneous_one_sided_lower"
        ] = -0.026
        result = probe.select_component(
            _gate_map(probe.ARM_Z), bootstrap, {"passed": True}
        )
        self.assertFalse(result["eligibility"][probe.ARM_Z]["eligible"])
        contract = probe.selection_contract()
        self.assertTrue(
            contract["inference"][
                "three_IUT_limbs_share_same_quantile_without_within_arm_alpha_split"
            ]
        )

    def test_selection_prefers_single_then_fixed_order_not_observed_auc(self) -> None:
        eligible = (probe.ARM_B, probe.ARM_Z, probe.ARM_ZE, probe.ARM_BZE)
        result = probe.select_component(
            _gate_map(*eligible), _bootstrap_map(*eligible), {"passed": True}
        )
        self.assertEqual(result["selected_arm"], probe.ARM_Z)
        self.assertFalse(result["observed_auc_used_to_break_ties"])
        pair_only = (probe.ARM_BZ, probe.ARM_BE, probe.ARM_ZE, probe.ARM_BZE)
        pair_result = probe.select_component(
            _gate_map(*pair_only), _bootstrap_map(*pair_only), {"passed": True}
        )
        self.assertEqual(pair_result["selected_arm"], probe.ARM_ZE)

    def test_action_only_and_updater_only_interpretations_are_fixed(self) -> None:
        action = probe.select_component(
            _gate_map(probe.ARM_A),
            _bootstrap_map(probe.ARM_A),
            {"passed": True},
        )
        self.assertEqual(action["diagnosis"], "behavior_history_shortcut_supported")
        self.assertIsNone(action["selected_arm"])
        updater = probe.select_component(
            _gate_map(), _bootstrap_map(), {"passed": True}
        )
        self.assertEqual(updater["diagnosis"], "multi_component_dependence_unresolved")
        self.assertIsNone(updater["selected_arm"])

    def test_dense_triple_can_localize_without_action_or_pending(self) -> None:
        result = probe.select_component(
            _gate_map(probe.ARM_BZE),
            _bootstrap_map(probe.ARM_BZE),
            {"passed": True},
        )
        self.assertEqual(result["selected_arm"], probe.ARM_BZE)
        self.assertEqual(
            result["diagnosis"],
            "component_set_nominated_for_production_form_probe",
        )
        self.assertIn("explicit_prior_action_onehot", result["BZE_minus_U0_scope"])

    def test_shared_bootstrap_indices_and_digest_are_deterministic(self) -> None:
        tape = _tape(roots=768, roots_per_episode=12)
        rng = np.random.default_rng(93)
        probabilities = {
            arm: rng.uniform(0.05, 0.95, size=tape.hazard_targets.shape)
            for arm in probe.ARMS
        }
        first = probe.shared_bootstrap_report(
            tape.hazard_targets,
            probabilities,
            tape.episode_group_ids,
            resamples=64,
            seed=19,
        )
        second = probe.shared_bootstrap_report(
            tape.hazard_targets,
            probabilities,
            tape.episode_group_ids,
            resamples=64,
            seed=19,
        )
        self.assertEqual(
            first["shared_sampled_group_index_sha256"],
            second["shared_sampled_group_index_sha256"],
        )
        self.assertEqual(first["comparisons"], second["comparisons"])
        self.assertEqual(first["one_sided_lower_quantile"], 0.00625)
        self.assertEqual(first["quantile_method"], "linear")
        self.assertEqual(first["delta_orientation"], "left_minus_right")
        self.assertEqual(first["group_count"], 64)
        self.assertEqual(first["roots_per_group"], 12)
        self.assertEqual(first["branches_per_root"], 5)
        self.assertIn("intersection_union", first["multiplicity_method"])
        self.assertIn("not_confirmatory", first["inference_status"])
        self.assertIn("U_minus_U0_descriptive", first)
        self.assertFalse(
            first["U_minus_U0_descriptive"]["used_as_eligibility_gate"]
        )

    def test_bootstrap_rejects_nonregistered_dev_geometry(self) -> None:
        tape = _tape(roots=8)
        probabilities = {
            arm: np.full(tape.hazard_targets.shape, 0.5, dtype=np.float64)
            for arm in probe.ARMS
        }
        with self.assertRaisesRegex(ValueError, "exactly 64 episode clusters"):
            probe.shared_bootstrap_report(
                tape.hazard_targets,
                probabilities,
                tape.episode_group_ids,
                resamples=4,
                seed=1,
            )

    def test_strict_result_and_source_hashes_are_pinned(self) -> None:
        self.assertEqual(
            probe.EXACT_STRICT_RESULT_SHA256,
            "2ab123d5d2d42a1d14327bcd76710aa93538564f8897c8c2431a1133127bd2aa",
        )
        self.assertEqual(
            probe.EXACT_STRICT_REGISTRATION_SHA256,
            "1577d518ae138250f495f875cb0160193adb170a821dddb2798ec9530941864c",
        )
        self.assertEqual(len(probe.EXACT_STRICT_SCRIPT_SHA256), 64)
        self.assertEqual(len(probe.EXACT_STRICT_DOC_SHA256), 64)
        self.assertEqual(len(probe.EXACT_STRICT_TEST_SHA256), 64)

    def test_registration_is_create_only_and_detects_source_drift(self) -> None:
        synthetic_bundle = {
            "schema_version": 1,
            "sha256": "1" * 64,
            "strict_live_dependencies": {},
            "diagnostic_files": {},
        }
        strict_record = {
            "artifact": probe.STRICT_RESULT,
            "sha256": probe.EXACT_STRICT_RESULT_SHA256,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "upstream_result": root / "upstream.json",
                "registration": root / "registration.json",
                "result": root / "result.json",
            }
            registration = paths["registration"]
            with patch.object(probe, "canonical_paths", return_value=paths):
                with patch.object(probe, "_diagnostic_source_bundle", return_value=synthetic_bundle):
                    with patch.object(probe, "validate_strict_result", return_value=strict_record):
                        probe.register(registration)
                        record = probe.validate_registration(registration)
                        self.assertEqual(record["sha256"], probe._sha256_file(registration))
                        with self.assertRaises(FileExistsError):
                            probe.register(registration)
            drifted = dict(synthetic_bundle)
            drifted["sha256"] = "2" * 64
            with patch.object(probe, "canonical_paths", return_value=paths):
                with patch.object(probe, "_diagnostic_source_bundle", return_value=drifted):
                    with patch.object(probe, "validate_strict_result", return_value=strict_record):
                        with self.assertRaisesRegex(ValueError, "drifted"):
                            probe.validate_registration(registration)

    def test_canonical_paths_make_registration_and_result_single_use(self) -> None:
        identity = probe.run_identity()
        self.assertFalse(identity["retry_allowed"])
        self.assertFalse(identity["alternate_paths_allowed"])
        self.assertEqual(identity["registration"], probe.CANONICAL_REGISTRATION)
        self.assertEqual(identity["result"], probe.CANONICAL_RESULT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "upstream_result": root / "upstream.json",
                "registration": root / "registration.json",
                "result": root / "result.json",
            }
            wrong_registration = root / "alternate-registration.json"
            with patch.object(probe, "canonical_paths", return_value=paths):
                with self.assertRaisesRegex(ValueError, "single canonical path"):
                    probe.register(wrong_registration)
                with self.assertRaisesRegex(ValueError, "single canonical path"):
                    probe.run(
                        upstream_result=paths["upstream_result"],
                        output=root / "alternate-result.json",
                        registration=paths["registration"],
                    )
            self.assertFalse(wrong_registration.exists())
            self.assertFalse((root / "alternate-result.json").exists())

    def test_complete_strict_live_source_bundle_is_recomputed(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        exact = probe.strict._diagnostic_source_bundle(project_root)
        drifted = dict(exact)
        drifted["sha256"] = "0" * 64
        with patch.object(
            probe.strict, "_diagnostic_source_bundle", return_value=drifted
        ):
            with self.assertRaisesRegex(ValueError, "current complete strict-live"):
                probe.validate_strict_result(project_root)

    def test_control_failure_stops_before_reduced_metrics_or_bootstrap(self) -> None:
        source = inspect.getsource(probe._run_impl)
        self.assertIn('sources["TRAIN-FIT"].manifest_sha256', source)
        self.assertIn('sources["DEV"].manifest_sha256', source)
        guard = source.index('if controls["passed"] is not True')
        reduced_metrics = source.index("if arm not in {ARM_N, ARM_U}")
        bootstrap = source.index("shared_bootstrap_report")
        self.assertLess(guard, reduced_metrics)
        self.assertLess(guard, bootstrap)
        self.assertIn("exact N/U control reproduction failed", source)
        parent_guard = source.index("if parent_after != parent_before")
        publication = source.index("_publish_json_create_only")
        self.assertLess(parent_guard, publication)

    def test_failure_receipt_cannot_publish_or_emit_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {
                "upstream_result": root / "upstream.json",
                "registration": root / "registration.json",
                "result": root / "failure.json",
            }
            output = paths["result"]
            registration = paths["registration"]
            registration.write_text("{}", encoding="utf-8")
            record = {
                "path": str(registration),
                "sha256": sha256(b"{}").hexdigest(),
                "mode": probe.REGISTRATION_MODE,
                "diagnostic_source_bundle": {},
                "strict_live_v2": {},
            }
            with patch.object(probe, "canonical_paths", return_value=paths):
                with patch.object(probe, "validate_registration", return_value=record):
                    with patch.object(probe, "_run_impl", side_effect=RuntimeError("synthetic")):
                        with self.assertRaisesRegex(RuntimeError, "synthetic"):
                            probe.run(
                                upstream_result=paths["upstream_result"],
                                output=output,
                                registration=registration,
                            )
                        with self.assertRaises(FileExistsError):
                            probe.run(
                                upstream_result=paths["upstream_result"],
                                output=output,
                                registration=registration,
                            )
            payload = probe.json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["classification"], "diagnostic_run_failed_not_candidate")
        self.assertFalse(payload["qualification_claimed"])
        self.assertFalse(payload["candidate_publication_allowed"])
        self.assertFalse(payload["candidate_checkpoint"]["published"])
        self.assertFalse(payload["candidate_checkpoint"]["checkpoint_emitted"])
        self.assertFalse(
            payload["strict_sensory_and_namespace_scope"]["cpu_qual_opened"]
        )


if __name__ == "__main__":
    unittest.main()
