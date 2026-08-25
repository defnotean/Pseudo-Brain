"""Synthetic contract tests for the production-only V2.1i PLAY-QUAL runner."""
from __future__ import annotations

import copy
from dataclasses import asdict
from hashlib import sha256
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import v21i_live_qualification_runner as runner
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Model


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _training_runtime() -> dict[str, object]:
    return {
        "accelerator": "nvidia_dgx_spark",
        "authorization_receipt_sha256": "8" * 64,
        "bounded_schedule_sha256": "9" * 64,
        "run_artifact_sha256": "a" * 64,
    }


def _release_receipt(
    *,
    checkpoint_sha256: str,
    state_sha256: str,
    model_config_sha256: str,
    feature_flags_sha256: str,
    hazard_calibration_sha256: str,
    source_bundle_sha256: str,
    model_seed: int,
) -> runner.PostDgxReleaseReceipt:
    runtime = _training_runtime()
    body = {
        "authorization": {
            "cpu_wallclock_allowed": True,
            "play_qualification_allowed": True,
            "test_split_access_allowed": False,
            "training_allowed": False,
        },
        "checkpoint": {
            "checkpoint_sha256": checkpoint_sha256,
            "feature_flags_sha256": feature_flags_sha256,
            "hazard_calibration_sha256": hazard_calibration_sha256,
            "model_config_sha256": model_config_sha256,
            "model_seed": model_seed,
            "source_bundle_sha256": source_bundle_sha256,
            "state_sha256": state_sha256,
        },
        "classification": "post_dgx_cpu_qual_release_v1",
        "cpu_qualification": {
            "all_candidates_passed": True,
            "evaluation_artifact_sha256": "b" * 64,
            "observed_passes": 5,
            "opening_receipt_sha256": "c" * 64,
            "passed_model_seeds": [44, 45, 46, 47, 48],
            "preregistration_sha256": "d" * 64,
            "qualification_id": "v21i-cpu-qual-world-model-v1",
            "required_model_seeds": [44, 45, 46, 47, 48],
            "required_passes": 5,
            "test_split_opened": False,
        },
        "dgx_lineage": runtime,
        "post_dgx_verification": {
            "all_metrics_finite": True,
            "artifact_sha256": "e" * 64,
            "calibration_passed": True,
            "causal_shuffle_passed": True,
            "hazard_quality_passed": True,
            "noncollapse_passed": True,
            "preservation_passed": True,
            "qualification_id": "v21i-post-dgx-verification-v1",
        },
    }
    body_sha = runner._json_sha256(
        body, domain=b"IRV21IPOSTDGXRELEASEBODY\x01"
    )
    payload = {
        "body": body,
        "body_sha256": body_sha,
        "qualification_id": "v21i-post-dgx-cpu-qual-release-v1",
        "schema_version": 1,
        "status": "passed",
    }
    encoded = (runner._canonical_json(payload) + "\n").encode()
    return runner.PostDgxReleaseReceipt(
        path=Path("post-dgx-release.json"),
        payload=payload,
        body=body,
        body_sha256=body_sha,
        file_sha256=sha256(encoded).hexdigest(),
    )


def _loaded_stub() -> runner.LoadedCheckpoint:
    model_config_sha = runner._json_sha256(
        asdict(runner._expected_model_config()), domain=runner._MODEL_CONFIG_DOMAIN
    )
    feature_flags_sha = runner._json_sha256(
        asdict(CONFIG_B_PREDICTIVE), domain=runner._FEATURE_FLAGS_DOMAIN
    )
    source_sha = str(runner.training_source_bundle(PROJECT_ROOT)["sha256"])
    release = _release_receipt(
        checkpoint_sha256="1" * 64,
        state_sha256="2" * 64,
        model_config_sha256=model_config_sha,
        feature_flags_sha256=feature_flags_sha,
        hazard_calibration_sha256="5" * 64,
        source_bundle_sha256=source_sha,
        model_seed=44,
    )
    return runner.LoadedCheckpoint(
        model=object(),  # type: ignore[arg-type]
        checkpoint_sha256="1" * 64,
        state_sha256="2" * 64,
        model_config_sha256=model_config_sha,
        feature_flags_sha256=feature_flags_sha,
        hazard_calibration_sha256="5" * 64,
        source_bundle_sha256=source_sha,
        model_seed=44,
        post_dgx_release=release,
        training_runtime=_training_runtime(),
    )


def _redigest(payload: dict[str, object]) -> None:
    body = dict(payload)
    del body["preregistration_sha256"]
    payload["preregistration_sha256"] = runner._json_sha256(
        body, domain=runner._PREREGISTRATION_BODY_DOMAIN
    )


class FixedPlayQualificationCohortTests(unittest.TestCase):
    def test_exact_64_top_bit_11_seeds_are_fixed_and_ordered(self) -> None:
        seeds = runner.fixed_play_qual_environment_seeds()
        self.assertEqual(len(seeds), 64)
        self.assertEqual(len(set(seeds)), 64)
        self.assertEqual(seeds[0], 0xC000000000000000)
        self.assertEqual(seeds[-1], 0xC00000000000003F)
        self.assertTrue(all(seed >> 62 == 0b11 for seed in seeds))

    def test_random_seed_bank_is_exactly_eight_unique_streams_per_seed(self) -> None:
        bank = runner.fixed_random_stream_seed_bank()
        self.assertEqual(len(bank), 64)
        self.assertTrue(all(len(row) == 8 for row in bank))
        self.assertEqual(len({seed for row in bank for seed in row}), 64 * 8)

    def test_cohort_uses_lossless_hex_and_one_cluster_per_environment(self) -> None:
        cohort = runner._cohort_record()
        seeds = cohort["environment_seeds"]
        self.assertEqual(seeds[0], "c000000000000000")
        self.assertEqual(seeds[-1], "c00000000000003f")
        self.assertTrue(all(len(value) == 16 and value == value.lower() for value in seeds))
        self.assertEqual(len(set(cohort["cluster_ids"])), 64)
        self.assertEqual(cohort["random_stream_ids"], list(range(8)))
        self.assertEqual(cohort["paired_environment_random_stream_count"], 512)

    def test_live_cohort_adapter_matches_the_evaluator_schema_and_digest(self) -> None:
        payload = runner.build_play_qual_preregistration(
            checkpoint=_loaded_stub(), project_root=PROJECT_ROOT
        )
        registration = runner.SealedPlayQualification(
            path=Path("sealed.json"),
            file_sha256="a" * 64,
            preregistration_sha256=str(payload["preregistration_sha256"]),
            payload=payload,
        )
        receipt = runner.live_cohort_receipt(registration)
        config = runner._production_live_config()
        provenance = runner.LivePlayProvenance(
            qualification_id=runner.QUALIFICATION_ID,
            model_identity=runner.OutcomeAwareCoreV2MazePolicy.identity,
            checkpoint_envelope=runner.POST_DGX_ENVELOPE,
            checkpoint_sha256="1" * 64,
            model_state_sha256="6" * 64,
            model_config_sha256="2" * 64,
            feature_flags_sha256="7" * 64,
            source_bundle_sha256="3" * 64,
            evaluator_bundle_sha256="4" * 64,
            environment_source_sha256="5" * 64,
            preregistration_sha256=registration.preregistration_sha256,
            cohort_binding_sha256=runner.live_play_cohort_binding_sha256(receipt),
            post_dgx_release_body_sha256=_loaded_stub().post_dgx_release.body_sha256,
            post_dgx_release_file_sha256=_loaded_stub().post_dgx_release.file_sha256,
        )
        from irene_brain.evaluation.v21_live_play_qualification import (
            verify_live_play_cohort_receipt,
        )

        verify_live_play_cohort_receipt(receipt, config=config, provenance=provenance)
        self.assertEqual(receipt["seed_bank"], payload["seed_bank"])

    def test_live_config_is_not_caller_selectable(self) -> None:
        config = runner._production_live_config()
        self.assertEqual(config.episode_seeds, runner.fixed_play_qual_environment_seeds())
        self.assertEqual(config.max_ticks, 600)
        self.assertEqual(config.random_baseline_streams, 8)
        self.assertEqual(config.deterministic_repeats, 2)
        self.assertEqual(len(set(config.cluster_ids)), 64)


class SealedPlayQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = runner.build_play_qual_preregistration(
            checkpoint=_loaded_stub(), project_root=PROJECT_ROOT
        )

    def test_round_trip_canonical_create_only_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "play-qual-prereg.json"
            file_sha = runner.publish_play_qual_preregistration_create_only(
                path, self.payload, project_root=PROJECT_ROOT
            )
            sealed = runner.load_sealed_play_qual_preregistration(
                path, project_root=PROJECT_ROOT
            )
            self.assertEqual(sealed.file_sha256, file_sha)
            self.assertEqual(
                sealed.preregistration_sha256,
                self.payload["preregistration_sha256"],
            )
            expected = (runner._canonical_json(self.payload) + "\n").encode()
            self.assertEqual(path.read_bytes(), expected)

    def test_create_only_collision_preserves_original_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "play-qual-prereg.json"
            path.write_bytes(b"owner\n")
            with self.assertRaises(runner.PlayQualificationError):
                runner.publish_play_qual_preregistration_create_only(
                    path, self.payload, project_root=PROJECT_ROOT
                )
            self.assertEqual(path.read_bytes(), b"owner\n")

    def test_seed_substitution_fails_even_with_recomputed_body_digest(self) -> None:
        altered = copy.deepcopy(self.payload)
        altered["seed_bank"]["environment_seeds"][0] = "c000000000000040"
        _redigest(altered)
        with self.assertRaisesRegex(runner.PlayQualificationError, "seed bank"):
            runner.validate_play_qual_preregistration(altered, project_root=PROJECT_ROOT)

    def test_stream_count_and_max_ticks_are_frozen(self) -> None:
        for mutate in (
            lambda value: value["seed_bank"].__setitem__(
                "random_streams_per_environment_seed", 7
            ),
            lambda value: value["workload"].__setitem__("max_ticks", 599),
        ):
            altered = copy.deepcopy(self.payload)
            mutate(altered)
            _redigest(altered)
            with self.assertRaises(runner.PlayQualificationError):
                runner.validate_play_qual_preregistration(
                    altered, project_root=PROJECT_ROOT
                )

    def test_live_source_digest_tamper_fails_after_redigest(self) -> None:
        altered = copy.deepcopy(self.payload)
        altered["evaluator_bundle"]["sha256"] = "f" * 64
        _redigest(altered)
        with self.assertRaisesRegex(runner.PlayQualificationError, "source changed"):
            runner.validate_play_qual_preregistration(altered, project_root=PROJECT_ROOT)

    def test_noncanonical_and_duplicate_key_json_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            noncanonical = Path(directory) / "pretty.json"
            noncanonical.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(runner.PlayQualificationError, "canonical"):
                runner.load_sealed_play_qual_preregistration(
                    noncanonical, project_root=PROJECT_ROOT
                )
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"x":1,"x":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(runner.PlayQualificationError, "duplicate"):
                runner.load_sealed_play_qual_preregistration(
                    duplicate, project_root=PROJECT_ROOT
                )


class ExactCheckpointAndProductionSurfaceTests(unittest.TestCase):
    def _checkpoint_payload(self) -> dict[str, object]:
        model = CoreV2Model(
            config=runner._expected_model_config(), flags=CONFIG_B_PREDICTIVE
        )
        state = model.state_dict()
        return {
            "calibration_installation_audit": {},
            "calibration_mode": "bias_only",
            "classification": (
                "bounded_dgx_candidate_requires_post_training_cpu_qual"
            ),
            "config": asdict(runner._expected_model_config()),
            "dataset_partitions": {"TRAIN-FIT": {}, "TRAIN-CAL": {}, "DEV": {}},
            "development_gate": {"passed": True},
            "device": "cuda",
            "expected_calibration_bindings": {},
            "finite_state_audit": {},
            "flags": asdict(CONFIG_B_PREDICTIVE),
            "hazard_calibration": {
                "accepted": True,
                "bias": [0.0] * 5,
                "fit_mode": "bias_only",
                "scale": [1.0] * 5,
            },
            "mode": "v21i_bounded_dgx_candidate_v1",
            "model_seed": 44,
            "model_state_dict": state,
            "schema_version": 2,
            "source_bundle": runner.training_source_bundle(PROJECT_ROOT),
            "state_sha256": runner._state_dict_sha256(state),
            "threads": 1,
            "training": {},
            "training_runtime": _training_runtime(),
            "upstream_uncalibrated_checkpoint": {},
        }

    def _matching_release(
        self,
        path: Path,
        payload: dict[str, object],
    ) -> runner.PostDgxReleaseReceipt:
        return _release_receipt(
            checkpoint_sha256=sha256(path.read_bytes()).hexdigest(),
            state_sha256=str(payload["state_sha256"]),
            model_config_sha256=runner._json_sha256(
                payload["config"], domain=runner._MODEL_CONFIG_DOMAIN
            ),
            feature_flags_sha256=runner._json_sha256(
                payload["flags"], domain=runner._FEATURE_FLAGS_DOMAIN
            ),
            hazard_calibration_sha256=runner._json_sha256(
                payload["hazard_calibration"],
                domain=runner._HAZARD_CALIBRATION_DOMAIN,
            ),
            source_bundle_sha256=str(payload["source_bundle"]["sha256"]),
            model_seed=int(payload["model_seed"]),
        )

    def test_checkpoint_load_is_strict_cpu_and_recomputes_all_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            payload = self._checkpoint_payload()
            torch.save(payload, path)
            checkpoint_sha = sha256(path.read_bytes()).hexdigest()
            release = self._matching_release(path, payload)
            loaded = runner.load_exact_v21i_checkpoint(
                path,
                project_root=PROJECT_ROOT,
                expected_checkpoint_sha256=checkpoint_sha,
                expected_source_bundle_sha256=str(
                    runner.training_source_bundle(PROJECT_ROOT)["sha256"]
                ),
                expected_model_seed=44,
                post_dgx_release=release,
            )
            self.assertEqual(loaded.checkpoint_sha256, checkpoint_sha)
            self.assertTrue(all(value.device.type == "cpu" for value in loaded.model.parameters()))
            self.assertFalse(loaded.model.training)
            self.assertEqual(torch.get_num_threads(), 1)

    def test_checkpoint_extra_field_and_config_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for index, mutate in enumerate(
                (
                    lambda value: value.__setitem__("unexpected", True),
                    lambda value: value["config"].__setitem__("hazard_weight", 0.5),
                )
            ):
                payload = self._checkpoint_payload()
                mutate(payload)
                path = Path(directory) / f"invalid-{index}.pt"
                torch.save(payload, path)
                release = self._matching_release(path, payload)
                with self.assertRaises(runner.PlayQualificationError):
                    runner.load_exact_v21i_checkpoint(
                        path,
                        project_root=PROJECT_ROOT,
                        expected_checkpoint_sha256=sha256(path.read_bytes()).hexdigest(),
                        expected_source_bundle_sha256=str(
                            payload["source_bundle"]["sha256"]
                        ),
                        expected_model_seed=44,
                        post_dgx_release=release,
                    )

    def test_production_surface_has_no_runner_or_environment_injection(self) -> None:
        parameters = inspect.signature(
            runner.run_production_play_qualification
        ).parameters
        self.assertNotIn("episode_runner", parameters)
        self.assertNotIn("environment_factory", parameters)
        self.assertNotIn("controller", parameters)
        live_parameters = inspect.signature(
            runner.qualify_live_play_create_only
        ).parameters
        self.assertNotIn("episode_runner", live_parameters)

    def test_preflight_failure_writes_one_complete_failure_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            with self.assertRaises(runner.PlayQualificationError):
                runner.run_production_play_qualification(
                    checkpoint_path=root / "missing.pt",
                    preregistration_path=root / "missing-prereg.json",
                    post_dgx_release_path=root / "missing-release.json",
                    wallclock_path=root / "missing-wallclock.json",
                    output_path=output,
                    project_root=PROJECT_ROOT,
                )
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["status"], "failed")
            self.assertEqual(
                artifact["classification"],
                "production_play_qualification_preflight_failed",
            )

    def test_synthetic_happy_path_reaches_public_qualifier_with_complete_bindings(
        self,
    ) -> None:
        loaded = _loaded_stub()

        class _Controller:
            identity = runner.OutcomeAwareCoreV2MazePolicy.identity

            def __init__(self, model: object) -> None:
                self.model = model

        calls: list[dict[str, object]] = []

        def fake_qualifier(
            controller: object,
            *,
            config: object,
            provenance: object,
            cohort_receipt: object,
            wall_clock_artifact: object,
            output_path: Path,
        ) -> None:
            calls.append(
                {
                    "controller": controller,
                    "config": config,
                    "provenance": provenance,
                    "cohort_receipt": cohort_receipt,
                    "wall_clock_artifact": wall_clock_artifact,
                }
            )
            runner._publish_json_create_only(
                Path(output_path),
                {"schema_version": 1, "status": "passed"},
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preregistration = root / "preregistration.json"
            payload = runner.build_play_qual_preregistration(
                checkpoint=loaded, project_root=PROJECT_ROOT
            )
            runner.publish_play_qual_preregistration_create_only(
                preregistration, payload, project_root=PROJECT_ROOT
            )
            output = root / "result.json"
            wallclock = {"status": "passed"}
            release_path = root / "post-dgx-release.json"
            release_path.write_bytes(
                (
                    runner._canonical_json(
                        dict(loaded.post_dgx_release.payload)
                    )
                    + "\n"
                ).encode("utf-8")
            )
            with (
                patch.object(runner, "load_exact_v21i_checkpoint", return_value=loaded),
                patch.object(
                    runner,
                    "_load_wallclock_artifact",
                    return_value=wallclock,
                ),
                patch.object(runner, "OutcomeAwareCoreV2MazePolicy", _Controller),
                patch.object(runner, "qualify_live_play_create_only", fake_qualifier),
            ):
                runner.run_production_play_qualification(
                    checkpoint_path=root / "candidate.pt",
                    preregistration_path=preregistration,
                    post_dgx_release_path=release_path,
                    wallclock_path=root / "wallclock.json",
                    output_path=output,
                    project_root=PROJECT_ROOT,
                )
            self.assertEqual(len(calls), 1)
            call = calls[0]
            config = call["config"]
            provenance = call["provenance"]
            cohort = call["cohort_receipt"]
            self.assertEqual(config.max_ticks, 600)
            self.assertEqual(config.random_baseline_streams, 8)
            self.assertEqual(provenance.checkpoint_sha256, loaded.checkpoint_sha256)
            self.assertEqual(
                provenance.cohort_binding_sha256,
                runner.live_play_cohort_binding_sha256(cohort),
            )
            self.assertEqual(cohort["preregistration_sha256"], payload["preregistration_sha256"])
            self.assertIs(call["wall_clock_artifact"], wallclock)
            self.assertEqual(json.loads(output.read_text())["status"], "passed")


if __name__ == "__main__":
    unittest.main()
