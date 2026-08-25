"""Focused contracts for the strict-live V2.1i representation/head probe."""
from __future__ import annotations

from hashlib import sha256
import inspect
from pathlib import Path
from struct import pack
import sys
import tempfile
from types import SimpleNamespace
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

import v21i_strict_live_representation_probe_v1 as probe

from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Model
from irene_brain.types import RgbFrame
from irene_brain.v2.maze_policy import OutcomeAwareCoreV2MazePolicy
from irene_brain.v2.outcome_model import ActionOutcomeTable


def _tape(*, roots: int = 6, history_offset: int = 0) -> probe.StrictLiveFeatureTape:
    rng = np.random.default_rng(41)
    targets = np.asarray(
        [[(root + action) % 2 for action in range(probe.ACTION_COUNT)] for root in range(roots)],
        dtype=np.float64,
    )
    return probe.StrictLiveFeatureTape(
        beliefs=rng.normal(size=(roots, probe.model_config().width)),
        updater_inputs=rng.normal(size=(roots, probe.CAPACITY_STATE_WIDTH)),
        hazard_targets=targets,
        parent_raw_logits=rng.normal(size=(roots, probe.ACTION_COUNT)),
        episode_group_ids=tuple(f"DEV:episode:{root // 2}" for root in range(roots)),
        root_state_ids=tuple(sha256(f"root:{root}".encode()).hexdigest() for root in range(roots)),
        history=probe.InterventionHistory(
            prior_assignment_count=np.arange(roots, dtype=np.int64) + history_offset,
            prior_disagreement_count=np.arange(roots, dtype=np.int64) + history_offset + 2,
        ),
    )


class StrictLiveRepresentationProbeTests(unittest.TestCase):
    def test_revision2_binds_exact_failed_v1_artifact_and_registration(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        record = probe.failed_v1_record(project_root)
        self.assertEqual(record["sha256"], probe.EXACT_FAILED_V1_ARTIFACT_SHA256)
        self.assertEqual(
            record["registration"]["sha256"],
            probe.EXACT_FAILED_V1_REGISTRATION_SHA256,
        )
        self.assertEqual(record["cause"], "cpu_gemm_batch_shape_rounding")
        self.assertEqual(record["diagnosis"]["actual_failed_guard_batch_size"], 1536)
        self.assertEqual(
            record["diagnosis"]["independent_reproduction_batch_size"],
            257,
        )
        self.assertEqual(
            record["diagnosis"][
                "independent_matched_batch_size_one_mismatch_count"
            ],
            0,
        )

    def test_local_partition_factory_constructs_only_exact_train_fit_and_dev(self) -> None:
        contracts = probe.strict_live_partition_contracts()
        self.assertEqual(set(contracts), {"TRAIN-FIT", "DEV"})
        self.assertEqual(
            contracts["TRAIN-FIT"].dataset_config.seed_offset, probe.TRAIN_FIT_OFFSET
        )
        self.assertEqual(contracts["DEV"].dataset_config.seed_offset, probe.DEV_OFFSET)
        constructed: list[str] = []

        def source_factory(contract):
            constructed.append(contract.name)
            return SimpleNamespace(contract=contract)

        sources = probe.strict_live_partition_sources(source_factory)
        self.assertEqual(set(sources), {"TRAIN-FIT", "DEV"})
        self.assertEqual(constructed, ["TRAIN-FIT", "DEV"])
        self.assertNotIn("TRAIN-CAL", constructed)
        self.assertNotIn("CPU-QUAL", constructed)
        self.assertNotIn("TEST", constructed)

    def test_schedule_is_exact_and_has_no_train_cal_or_selection(self) -> None:
        schedule = probe.schedule_record()
        self.assertEqual(schedule["train_partition"], "TRAIN-FIT")
        self.assertEqual(schedule["evaluation_partitions"], ["TRAIN-FIT", "DEV"])
        self.assertFalse(schedule["train_cal_constructed"])
        self.assertFalse(schedule["dev_used_for_stopping"])
        self.assertFalse(schedule["dev_used_for_model_selection"])
        self.assertFalse(schedule["calibration_fit_or_install"])
        self.assertEqual(schedule["strict_live_replay_batch_size"], 1)
        self.assertEqual(
            schedule["model_initialization_seed"],
            probe.OUTCOME_AWARE_MODEL_SEED_V1,
        )
        self.assertTrue(schedule["first_tick_rng_restarted_per_episode"])
        self.assertEqual(schedule["implementation_revision"], 2)
        self.assertEqual(schedule["parent_logit_equivalence_batch_size"], 1)
        self.assertEqual(schedule["parent_logit_equivalence"], "byte_exact_no_tolerance")
        self.assertEqual(
            schedule["v1_guard_correction_cause"],
            "cpu_gemm_batch_shape_rounding",
        )
        self.assertEqual(schedule["steps_per_pass"], 64)
        self.assertEqual(schedule["arms"][probe.ARM_O_WARM]["added_passes"], 24)
        self.assertEqual(
            schedule["arms"][probe.ARM_O_WARM]["nominal_total_passes_across_signatures"],
            32,
        )
        self.assertEqual(
            schedule["arms"][probe.ARM_O_WARM]["strict_live_tape_passes"], 24
        )
        self.assertEqual(schedule["arms"][probe.ARM_O_WARM]["optimizer_steps"], 1536)
        self.assertIn(
            "not 32 matched-distribution passes",
            schedule["arms"][probe.ARM_O_WARM]["cross_signature_caveat"],
        )
        for arm in (probe.ARM_O_SCRATCH, probe.ARM_C_BELIEF, probe.ARM_C_UPDATER):
            self.assertEqual(schedule["arms"][arm]["passes"], 32)
            self.assertEqual(schedule["arms"][arm]["optimizer_steps"], 2048)

    def test_training_function_cannot_accept_dev_or_stopping_inputs(self) -> None:
        parameters = inspect.signature(probe.train_probe_head).parameters
        self.assertEqual(
            tuple(parameters),
            ("head", "features", "targets", "permutations"),
        )
        source = inspect.getsource(probe.train_probe_head)
        self.assertNotIn("DEV", source)
        self.assertNotIn("validation", source.lower())
        self.assertIn('"early_stopping": False', source)
        self.assertIn('"model_selection": False', source)

    def test_collector_uses_strict_live_signature_and_private_prior(self) -> None:
        collector = inspect.getsource(probe.collect_strict_live_tape)
        reference = inspect.getsource(probe.strict_live_reference_step)
        self.assertIn("actual_reward=None", reference)
        self.assertIn("actual_hazard=None", reference)
        self.assertIn("prev_action=prior_action", reference)
        self.assertIn("strict_live_reference_step", collector)
        self.assertIn("_install_applied_pending", collector)
        self.assertNotIn("previous_control", collector + reference)
        # Features are formed before the current applied action becomes history.
        self.assertLess(
            collector.index("strict_live_reference_step"),
            collector.index("prior_action = applied"),
        )
        self.assertLess(
            collector.index("targets.append"),
            collector.index("_install_applied_pending"),
        )

    def test_multi_tick_reference_matches_live_hazard_path_and_selected_pending(self) -> None:
        """Pin hazard-path equivalence, not thoughts or deployed decision output."""

        torch.manual_seed(731)
        live_model = CoreV2Model(
            config=probe.model_config(),
            flags=CONFIG_B_PREDICTIVE,
        ).eval()
        reference_model = CoreV2Model(
            config=probe.model_config(),
            flags=CONFIG_B_PREDICTIVE,
        ).eval()
        reference_model.load_state_dict(live_model.state_dict())
        policy = OutcomeAwareCoreV2MazePolicy(live_model)
        policy.reset(9_999_991)
        reference_state = reference_model.init_state(1, torch.device("cpu"))
        logged_actions = (4, 2, 1)
        frames = tuple(
            RgbFrame(
                width=32,
                height=32,
                pixels=bytes(
                    ((pixel * 17 + tick * 29) % 256)
                    for pixel in range(32 * 32 * 3)
                ),
            )
            for tick in range(len(logged_actions))
        )
        live_steps: list[probe.StrictLiveStepSnapshot] = []
        original_forward = live_model.forward

        def record_live_forward(observation, state, *args, **kwargs):
            self.assertIsNone(kwargs.get("actual_reward"))
            self.assertIsNone(kwargs.get("actual_hazard"))
            old_belief = state.fast.belief.detach().clone()
            has_pending = torch.full(
                (1, 1),
                float(state.fast.pending_prediction is not None),
                dtype=torch.float32,
            )
            prior = kwargs["prev_action"]
            output, new_state = original_forward(observation, state, *args, **kwargs)
            cognitive_error = output.prediction_error.cognitive_error
            self.assertIsNotNone(cognitive_error)
            if cognitive_error.dim() == 3:
                cognitive_error = cognitive_error.mean(dim=1)
            updater = torch.cat(
                (
                    old_belief,
                    output.latent.detach(),
                    cognitive_error.detach(),
                    torch.nn.functional.one_hot(
                        prior,
                        num_classes=probe.ACTION_COUNT,
                    ).to(torch.float32),
                    has_pending,
                ),
                dim=-1,
            )
            table = output.outcome_table
            self.assertIsNotNone(table)
            live_steps.append(
                probe.StrictLiveStepSnapshot(
                    output=output,
                    state=new_state,
                    updater_input=updater,
                    canonical_raw_hazard_logits=probe._canonicalize_table(
                        table.raw_hazard_logits,
                        table.action_ids,
                    ),
                )
            )
            return output, new_state

        previous_action = 0
        with patch.object(live_model, "forward", side_effect=record_live_forward):
            with patch.object(
                policy,
                "_select_semantic_action",
                side_effect=logged_actions,
            ) as select_action:
                for tick, (frame, applied_action) in enumerate(
                    zip(frames, logged_actions, strict=True)
                ):
                    actual_action = policy.act_rgb(frame, previous_action)
                    self.assertEqual(actual_action, applied_action)
                    reference_step = probe.strict_live_reference_step(
                        reference_model,
                        (frame,),
                        reference_state,
                        torch.tensor([previous_action], dtype=torch.long),
                        first_tick=tick == 0,
                    )
                    reference_state = reference_step.state
                    live_step = live_steps[tick]
                    torch.testing.assert_close(
                        live_step.output.belief,
                        reference_step.output.belief,
                        rtol=0.0,
                        atol=0.0,
                    )
                    torch.testing.assert_close(
                        live_step.updater_input,
                        reference_step.updater_input,
                        rtol=0.0,
                        atol=0.0,
                    )
                    np.testing.assert_array_equal(
                        live_step.canonical_raw_hazard_logits,
                        reference_step.canonical_raw_hazard_logits,
                    )

                    probe._install_applied_pending(
                        reference_state,
                        reference_step.output.outcome_table,
                        torch.tensor([applied_action], dtype=torch.long),
                    )
                    live_pending = policy._state.fast.pending_prediction
                    reference_pending = reference_state.fast.pending_prediction
                    for field in (
                        "predicted_next_latent",
                        "predicted_reward",
                        "predicted_reward_logits",
                        "predicted_hazard",
                        "predicted_confidence",
                        "predicted_branch_logit",
                    ):
                        live_value = getattr(live_pending, field)
                        reference_value = getattr(reference_pending, field)
                        if live_value is None or reference_value is None:
                            self.assertIs(live_value, reference_value)
                        else:
                            torch.testing.assert_close(
                                live_value,
                                reference_value,
                                rtol=0.0,
                                atol=0.0,
                            )
                    self.assertEqual(policy._previous_action, applied_action)
                    previous_action = applied_action
        self.assertEqual(select_action.call_count, len(logged_actions))

    def test_intervention_prefix_counts_are_strictly_prior_only(self) -> None:
        values = [True, False, True, True, False]
        np.testing.assert_array_equal(
            probe.prior_only_history_counts(values),
            np.asarray([0, 1, 1, 2, 3], dtype=np.int64),
        )
        with self.assertRaises(TypeError):
            probe.prior_only_history_counts([True, 1])

    def test_assignment_reconstruction_matches_independent_dataset_digest(self) -> None:
        manifest = probe.EXACT_TRAIN_FIT_MANIFEST_SHA256
        for episode_seed in (16_777_216, 16_777_223):
            for tick in range(8):
                digest = sha256(
                    b"IRMCBEHAVIOR\x01"
                    + bytes.fromhex(manifest)
                    + pack(">Q", episode_seed)
                    + pack(">Q", tick)
                ).digest()
                expected = int.from_bytes(digest[:8], "big") < (1 << 63)
                self.assertEqual(
                    probe.behavior_assignment(
                        manifest,
                        episode_seed=episode_seed,
                        tick=tick,
                    ),
                    expected,
                )

    def test_history_and_provenance_never_enter_model_features(self) -> None:
        first = _tape(history_offset=0)
        second = probe.StrictLiveFeatureTape(
            beliefs=first.beliefs.copy(),
            updater_inputs=first.updater_inputs.copy(),
            hazard_targets=1.0 - first.hazard_targets,
            parent_raw_logits=first.parent_raw_logits + 10.0,
            episode_group_ids=tuple(reversed(first.episode_group_ids)),
            root_state_ids=tuple(reversed(first.root_state_ids)),
            history=probe.InterventionHistory(
                prior_assignment_count=first.history.prior_assignment_count + 100,
                prior_disagreement_count=first.history.prior_disagreement_count + 100,
            ),
        )
        for arm in probe.ARMS:
            np.testing.assert_array_equal(
                first.model_features(arm), second.model_features(arm)
            )
        belief_capacity = first.model_features(probe.ARM_C_BELIEF)
        np.testing.assert_array_equal(
            belief_capacity[:, : probe.model_config().width], first.beliefs
        )
        self.assertTrue(np.all(belief_capacity[:, probe.model_config().width :] == 0.0))
        np.testing.assert_array_equal(
            first.model_features(probe.ARM_C_UPDATER), first.updater_inputs
        )

    def test_current_probe_exactly_matches_parent_raw_hazard_topology(self) -> None:
        torch.manual_seed(7)
        model = CoreV2Model(config=probe.model_config(), flags=CONFIG_B_PREDICTIVE)
        head = probe.CurrentHazardProbe()
        head.load_parent(model)
        context = torch.randn(5, probe.model_config().width)
        expected = model.world_model.outcome_model(context).raw_hazard_logits.squeeze(-1)
        actual = head(context)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            sum(parameter.numel() for parameter in head.parameters()),
            probe.CURRENT_HEAD_PARAMETERS,
        )
        identity = probe.current_hazard_weight_copy_identity(head, model)
        self.assertTrue(identity["all_tensors_byte_equal"])
        self.assertTrue(identity["state_sha256_equal"])
        with torch.no_grad():
            head.head.bias.add_(1.0)
        drifted = probe.current_hazard_weight_copy_identity(head, model)
        self.assertFalse(drifted["all_tensors_byte_equal"])
        self.assertIn("head.bias", drifted["mismatched_tensor_names"])

    def test_exact_parent_guard_matches_batch_shape_not_loose_tolerance(self) -> None:
        torch.manual_seed(17)
        model = CoreV2Model(config=probe.model_config(), flags=CONFIG_B_PREDICTIVE).eval()
        head = probe.CurrentHazardProbe().eval()
        head.load_parent(model)
        features = torch.randn(257, probe.model_config().width).to(torch.float64).numpy()
        parent_rows: list[np.ndarray] = []
        with torch.no_grad():
            for index in range(len(features)):
                table = model.world_model.outcome_model(
                    torch.from_numpy(features[index : index + 1]).to(torch.float32)
                )
                parent_rows.append(
                    probe._canonicalize_table(table.raw_hazard_logits, table.action_ids)
                )
        parent = np.concatenate(parent_rows, axis=0)
        matched = probe.raw_logit_table(head, features, batch_size=1)
        large_batch = probe.raw_logit_table(head, features, batch_size=len(features))
        np.testing.assert_array_equal(matched, parent)
        mismatch_count = int(np.count_nonzero(large_batch != parent))
        self.assertGreater(mismatch_count, 0)
        self.assertLessEqual(float(np.max(np.abs(large_batch - parent))), 1.0e-5)
        self.assertFalse(np.array_equal(large_batch, parent))

    def test_capacity_arms_are_parameter_and_initialization_matched(self) -> None:
        left = probe.initialized_probe(probe.CapacityHazardProbe)
        right = probe.initialized_probe(probe.CapacityHazardProbe)
        self.assertEqual(
            sum(parameter.numel() for parameter in left.parameters()),
            probe.CAPACITY_HEAD_PARAMETERS,
        )
        for left_value, right_value in zip(
            left.state_dict().values(), right.state_dict().values(), strict=True
        ):
            self.assertTrue(torch.equal(left_value, right_value))

    def test_applied_pending_uses_semantic_gather_under_permutation(self) -> None:
        ids = torch.tensor([[4, 2, 0, 3, 1]], dtype=torch.long)
        semantic = torch.arange(5, dtype=torch.float32)
        column_values = semantic[ids]
        table = ActionOutcomeTable(
            action_ids=ids,
            predicted_next_latent=column_values.unsqueeze(-1).repeat(1, 1, 3),
            predicted_reward=column_values.unsqueeze(-1) + 10.0,
            predicted_reward_logits=None,
            predicted_hazard=column_values.unsqueeze(-1) / 4.0,
            raw_hazard_logits=column_values.unsqueeze(-1),
            predicted_hazard_logits=column_values.unsqueeze(-1),
        )
        state = SimpleNamespace(fast=SimpleNamespace(pending_prediction=None))
        probe._install_applied_pending(state, table, torch.tensor([2]))
        self.assertEqual(state.fast.pending_prediction.predicted_reward.item(), 12.0)
        self.assertEqual(state.fast.pending_prediction.predicted_hazard.item(), 0.5)

    def test_permutations_are_complete_and_fixed(self) -> None:
        left = probe.deterministic_permutations(17, passes=3)
        right = probe.deterministic_permutations(17, passes=3)
        self.assertEqual(len(left), 3)
        for left_value, right_value in zip(left, right, strict=True):
            self.assertTrue(torch.equal(left_value, right_value))
            self.assertEqual(set(left_value.tolist()), set(range(17)))

    def test_ordered_digest_preserves_order_and_multiplicity(self) -> None:
        original = ("episode:1", "episode:1", "episode:2")
        reordered = ("episode:1", "episode:2", "episode:1")
        deduplicated = ("episode:1", "episode:2")
        self.assertNotEqual(
            probe._ordered_sequence_digest(original),
            probe._ordered_sequence_digest(reordered),
        )
        self.assertNotEqual(
            probe._ordered_sequence_digest(original),
            probe._ordered_sequence_digest(deduplicated),
        )

    def test_ranking_gate_uses_only_frozen_thresholds(self) -> None:
        report = {
            "aggregate": {"roc_auc": 0.66},
            "per_action": [
                {"action_id": action, "metrics": {"roc_auc": 0.61}}
                for action in range(5)
            ],
        }
        result = probe.ranking_gate(report, prior_aggregate_auc=0.55)
        self.assertTrue(result["passed"])
        self.assertEqual(result["thresholds"]["minimum_aggregate_roc_auc"], 0.65)
        self.assertEqual(result["thresholds"]["minimum_baseline_roc_auc_gain"], 0.10)
        self.assertEqual(result["thresholds"]["minimum_per_action_roc_auc"], 0.60)

    def test_parent_and_train_fitted_prior_report_both_fixed_partitions(self) -> None:
        train_tape = _tape(roots=6)
        dev_tape = _tape(roots=6, history_offset=20)
        train_tape.hazard_targets[:] = (
            np.arange(6)[:, None] < np.arange(1, probe.ACTION_COUNT + 1)[None, :]
        )
        dev_tape.hazard_targets[:] = 1.0 - train_tape.hazard_targets
        reports, parent_probabilities, prior_probabilities = (
            probe.fixed_baseline_reports(train_tape, dev_tape)
        )
        self.assertEqual(
            set(reports["parent_strict_live"]),
            {"TRAIN-FIT", "DEV"},
        )
        prior_report = reports["train_fitted_action_prior"]
        self.assertEqual(prior_report["fit_partition"], "TRAIN-FIT")
        self.assertEqual(
            set(prior_report["partitions"]),
            {"TRAIN-FIT", "DEV"},
        )
        expected_prior = train_tape.hazard_targets.mean(axis=0)
        np.testing.assert_array_equal(
            np.asarray(prior_report["per_action_probability"]),
            expected_prior,
        )
        for partition, tape in (("TRAIN-FIT", train_tape), ("DEV", dev_tape)):
            self.assertEqual(parent_probabilities[partition].shape, tape.hazard_targets.shape)
            np.testing.assert_array_equal(
                prior_probabilities[partition],
                np.broadcast_to(expected_prior, tape.hazard_targets.shape),
            )

    def test_c_updater_interpretation_is_limited_to_current_tick_accessibility(self) -> None:
        gates = {arm: {"passed": False} for arm in probe.ARMS}
        gates[probe.ARM_C_UPDATER]["passed"] = True
        comparisons = {
            "O-warm_minus_parent": {"lower_95": -0.1},
            "O-scratch_minus_parent": {"lower_95": -0.1},
            "C-belief_minus_O-warm": {"lower_95": -0.1},
            "C-belief_minus_O-scratch": {"lower_95": -0.1},
            "C-updater_minus_C-belief": {"lower_95": 0.01},
        }
        report = probe.interpretation_report(gates, comparisons)
        self.assertEqual(
            report["diagnosis"],
            "current_tick_belief_update_compression_or_accessibility_supported",
        )
        self.assertTrue(report["current_tick_belief_update_compression_supported"])
        self.assertFalse(report["longer_history_information_loss_ruled_out"])
        self.assertIn("old_belief", report["c_updater_scope"])

        gates[probe.ARM_C_UPDATER]["passed"] = False
        inconclusive = probe.interpretation_report(gates, comparisons)
        self.assertEqual(
            inconclusive["diagnosis"],
            "inconclusive_across_upstream_sensory_or_longer_history_compression",
        )
        self.assertIn(
            "longer-history",
            inconclusive["c_updater_failure_interpretation"],
        )

    def test_paired_bootstrap_is_episode_clustered_and_fixed(self) -> None:
        targets = np.asarray(
            [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 0, 1, 0, 1]],
            dtype=np.float64,
        )
        better = np.where(targets == 1.0, 0.9, 0.1)
        worse = 1.0 - better
        groups = ("episode:0", "episode:0", "episode:1", "episode:1")
        result = probe.paired_episode_auc_delta(
            targets,
            better,
            worse,
            groups,
            resamples=64,
            seed=9,
        )
        self.assertGreater(result["lower_95"], 0.0)
        self.assertEqual(result["clustering"], "episode_paired_with_replacement")

    def test_intervention_strata_are_explicitly_non_feature_non_gate(self) -> None:
        tape = _tape()
        probabilities = np.where(tape.hazard_targets == 1.0, 0.8, 0.2)
        result = probe.intervention_stratification(
            tape.hazard_targets,
            probabilities,
            tape.history.prior_assignment_count,
        )
        self.assertFalse(result["used_as_feature"])
        self.assertFalse(result["used_as_gate"])
        self.assertEqual(result["causal_scope"], "strictly prior ticks; current tick excluded")
        for stratum in result["strata"]:
            self.assertFalse(stratum["used_as_feature"])
            self.assertFalse(stratum["used_as_gate"])

    def test_failure_receipt_cannot_publish_or_emit_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "failure.json"
            registration = Path(temporary) / "registration.json"
            registration.write_text("{}", encoding="utf-8")
            record = {
                "path": str(registration),
                "sha256": sha256(b"{}").hexdigest(),
                "mode": probe.REGISTRATION_MODE,
                "implementation_revision": probe.IMPLEMENTATION_REVISION,
                "supersedes_failed_v1": {
                    "sha256": probe.EXACT_FAILED_V1_ARTIFACT_SHA256,
                    "cause": probe.FAILED_V1_CAUSE,
                },
                "diagnostic_source_bundle": {},
                "scope": probe.diagnostic_scope(),
            }
            with patch.object(probe, "validate_registration", return_value=record):
                with patch.object(probe, "_run_impl", side_effect=RuntimeError("synthetic")):
                    with self.assertRaisesRegex(RuntimeError, "synthetic"):
                        probe.run(
                            upstream_result=Path("missing.json"),
                            output=output,
                            registration=registration,
                        )
            payload = probe.json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["classification"], "diagnostic_run_failed_not_candidate")
        self.assertEqual(payload["implementation_revision"], 2)
        self.assertEqual(payload["mode"], "v21i_strict_live_representation_probe_v2")
        self.assertFalse(payload["qualification_claimed"])
        self.assertFalse(payload["candidate_publication_allowed"])
        self.assertFalse(payload["candidate_checkpoint"]["published"])
        self.assertFalse(payload["candidate_checkpoint"]["publication_allowed"])
        self.assertFalse(payload["candidate_checkpoint"]["checkpoint_emitted"])
        self.assertEqual(
            payload["strict_sensory_and_namespace_scope"], probe.diagnostic_scope()
        )
        self.assertEqual(payload["registration"]["path"], str(registration.resolve()))
        self.assertEqual(
            payload["supersedes_failed_v1"]["sha256"],
            probe.EXACT_FAILED_V1_ARTIFACT_SHA256,
        )
        self.assertEqual(
            payload["revision_correction"]["cause"],
            "cpu_gemm_batch_shape_rounding",
        )

    def test_create_only_registration_binds_files_and_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in probe._DIAGNOSTIC_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"synthetic:{relative}\n", encoding="utf-8")
            registration = root / "registration.json"
            production = {
                "schema_version": 1,
                "sha256": probe.EXACT_V3_SOURCE_BUNDLE_SHA256,
                "files": {},
            }
            failed_v1 = {
                "artifact": probe.FAILED_V1_ARTIFACT,
                "sha256": probe.EXACT_FAILED_V1_ARTIFACT_SHA256,
                "registration": {
                    "artifact": probe.FAILED_V1_REGISTRATION,
                    "sha256": probe.EXACT_FAILED_V1_REGISTRATION_SHA256,
                },
                "cause": probe.FAILED_V1_CAUSE,
                "diagnosis": dict(probe.FAILED_V1_DIAGNOSIS),
            }
            with patch.object(probe, "_source_bundle", return_value=production):
                with patch.object(probe, "failed_v1_record", return_value=failed_v1):
                    probe.register(registration, project_root=root)
                    record = probe.validate_registration(registration, project_root=root)
                    self.assertEqual(record["sha256"], probe._sha256_file(registration))
                    self.assertEqual(record["implementation_revision"], 2)
                    self.assertEqual(record["supersedes_failed_v1"], failed_v1)
                    with self.assertRaises(FileExistsError):
                        probe.register(registration, project_root=root)
                    (root / probe._TEST_FILE).write_text("drift\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "drifted"):
                        probe.validate_registration(registration, project_root=root)


if __name__ == "__main__":
    unittest.main()
