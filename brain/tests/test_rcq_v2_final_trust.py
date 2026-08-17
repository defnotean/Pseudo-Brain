"""CPU-only adversarial tests for the RCQ-v2 final trust boundary.

These tests never construct or index either sealed production TEST dataset.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import importlib
import inspect
from itertools import permutations
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from irene_brain.data import MovingShapesSequenceDataset
from irene_brain.evaluation import rcq_v2_final as final
from irene_brain.evaluation import rcq_v2_registration as preregistration
from irene_brain.evaluation import rcq_v2_torch as torch_runner
from irene_brain.evaluation.rcq_v2 import RCQInputError
from irene_brain.runtime.policy import ResourcePolicy
from irene_brain.training.protocol import TrainingStepResult
from irene_brain.training.checkpoint import source_tree_sha256


def _registration_payload() -> dict[str, object]:
    zero = "0" * 64
    slices = {
        "train": {
            "split": "train",
            "local_start": 1_048_576,
            "local_end": 1_056_768,
            "sequences": 8_192,
            "scored_decisions": 49_152,
            "manifest_sha256": zero,
        },
        "development": {
            "split": "validation",
            "local_start": 1_048_576,
            "local_end": 1_048_832,
            "sequences": 256,
            "scored_decisions": 1_536,
            "manifest_sha256": zero,
        },
        "final_recipient": {
            "split": "test",
            "local_start": final.RECIPIENT_OFFSET,
            "local_end": final.RECIPIENT_OFFSET + final.RECIPIENT_SEQUENCES,
            "sequences": final.RECIPIENT_SEQUENCES,
            "scored_decisions": final.FINAL_DECISIONS,
            "manifest_sha256": zero,
        },
        "final_donor_only": {
            "split": "test",
            "local_start": final.DONOR_OFFSET,
            "local_end": final.DONOR_OFFSET + final.DONOR_SEQUENCES,
            "sequences": final.DONOR_SEQUENCES,
            "scored_decisions": 0,
            "manifest_sha256": zero,
        },
    }
    return {
        "schema_version": 2,
        "qualification_id": "rcq_v2_reference_v2",
        "evaluator_id": final.FINAL_EVALUATOR_ID,
        "config_canonical_sha256": zero,
        "config_raw_sha256": zero,
        "source_tree_sha256": zero,
        "evaluator_bundle_sha256": zero,
        "batch_source_manifest_sha256": zero,
        "run_seed": 1702,
        "run_id": "dgx-rcq-v2-reference-seed-1702",
        "model_factory": "irene_brain.training.factory:build_thesis_model",
        "joint_end_step": 1536,
        "final_step": final.FINAL_STEP,
        "sequence_length": 8,
        "burn_in_steps": 2,
        "hazard_count": 3,
        "tick_period_ns": 16_666_667,
        "discount_hex": float(0.99).hex(),
        "slices": slices,
        "bootstrap": {
            "cluster": (
                "unique_recipient_donor_sequence_pair_with_all_six_paired_decisions"
            ),
            "resamples": final.BOOTSTRAP_RESAMPLES,
            "seed": final.BOOTSTRAP_SEED,
            "one_sided_confidence": "0x1.e666666666666p-1",
            "sorted_lower_index_zero_based": final.BOOTSTRAP_LOWER_INDEX,
        },
        "thresholds": final._registered_thresholds(),
        "guard_band": {
            "local_start": final.RETIRED_END,
            "local_end": final.GUARD_END,
            "status": "unused",
        },
        "future_campaign_offset": final.FUTURE_CAMPAIGN_OFFSET,
        "runtime_protocol": {
            "device": "cuda",
            "precision": "bfloat16",
            "allow_tf32": True,
            "deterministic_algorithms": True,
            "compile_model": False,
            "num_workers": 0,
            "world_size": 1,
            "autocast": "torch.autocast(device_type=cuda,dtype=bfloat16)",
            "actuator_exit": "final",
            "button_activation": "logit_strictly_greater_than_zero",
            "target_button_activation": "target_control_strictly_greater_than_0.5",
            "continuous_output": "final_action_control",
            "recurrent_conditions": "independent_normal_and_rgb_deranged",
            "rgb_intervention": (
                "replace_rgb_only_from_one_donor_sequence_all_timesteps"
            ),
            "donor_assignment": (
                "target_blind_one_to_one_sequence_minimum_total_previous_wasd_bit_hamming"
            ),
            "donor_tie_break": (
                "sha256_seeded_donor_order_then_lowest_hungarian_column"
            ),
            "donor_matching_seed": final.DONOR_MATCHING_SEED,
            "donor_rgb_constraint": (
                "all_eight_corresponding_frame_sha256_values_distinct"
            ),
        },
        "development_gates": {
            "entry": {
                "gate": "rcq_v2_development_v1",
                "optimizer_step": 1_536,
            },
            "completion": {
                "gate": "rcq_v2_value_development_v1",
                "optimizer_step": final.FINAL_STEP,
                "entry_gate": "rcq_v2_development_v1",
                "entry_optimizer_step": 1_536,
                "train_timestep_previous_wasd_dev_mse_hex": "0x1.754d5eea85785p-2",
                "absolute_max_mse_hex": "0x1.4ff8d56cab52bp-2",
                "dev_target_variance_hex": "0x1.be77815f41fbfp-2",
                "entry_improvement_ratio_hex": "0x1.ccccccccccccdp-1",
            },
        },
        "workspace_protocol": {
            "contract": "pseudo-brain-workspace-v2",
            "host_account_home_relative_path": "projects/pseudo-brain",
            "host_marker_relative_path": ".pseudo-brain-workspace-v2",
            "host_marker_exact_utf8": "pseudo-brain-workspace-v2\n",
            "dedicated_dispatcher_action": "rcq_v2_final_once_v1",
            "preclaim_dispatcher_action": "rcq_v2_preclaim_v1",
            "receipt_verifier_dispatcher_action": "rcq_v2_verify_receipt_v1",
            "container_release_root": "/workspace/repo",
            "container_run_root": "/workspace/run",
            "container_claim_registry_root": "/workspace/final-claims",
            "container_pin_root": "/workspace/pins",
            "host_pin_directory_relative_path": (
                "qualification-pins/rcq-v2-reference-v2"
            ),
            "pretraining_pin_filename": "pretraining.json",
            "final_authorization_filename": "final-authorization.json",
            "registration_release_relative_path": (
                "registrations/rcq-v2-reference-v2.json"
            ),
            "readiness_receipt_relative_path": (
                "preclaim-readiness/rcq-v2-reference-v2.json"
            ),
        },
        "receipt_directory": f"final-claims/{final._final_range_claim_id()}",
        "config_test_field_status": (
            "disabled_retired_placeholder_generic_trainer_test_forbidden"
        ),
        "scope": (
            "one-seed open-loop teacher-forced reference-policy qualification; "
            "not closed-loop gameplay, architecture superiority, or causal thought use"
        ),
    }


def _write_registration(root: Path, payload: dict[str, object]) -> tuple[Path, str]:
    encoded = (final._canonical(payload) + "\n").encode("utf-8")
    path = root / "registration.json"
    path.write_bytes(encoded)
    return path, sha256(encoded).hexdigest()


def _fake_sequence(
    *,
    episode_seed: int,
    rgb_namespace: str,
    masks: tuple[int, ...],
    target_marker: int,
) -> SimpleNamespace:
    transitions = []
    for time_index, mask in enumerate(masks):
        keys_down = tuple(
            final._MOVEMENT_INDICES[index]
            for index in range(4)
            if mask & (1 << index)
        )
        transitions.append(
            SimpleNamespace(
                observation=SimpleNamespace(
                    rgb=SimpleNamespace(
                        sha256=sha256(
                            f"{rgb_namespace}:{episode_seed}:{time_index}".encode()
                        ).hexdigest()
                    ),
                    previous_control=SimpleNamespace(keys_down=keys_down),
                ),
                action_target=SimpleNamespace(marker=target_marker),
                value_target=float(target_marker),
            )
        )
    return SimpleNamespace(
        episode_seed=episode_seed,
        transitions=tuple(transitions),
    )


def _brute_force_assignment_cost(
    costs: tuple[tuple[int, ...], ...],
    *,
    forbidden: int,
) -> int | None:
    legal = [
        sum(costs[row][column] for row, column in enumerate(candidate))
        for candidate in permutations(range(len(costs)))
        if all(costs[row][column] < forbidden for row, column in enumerate(candidate))
    ]
    return min(legal) if legal else None


def _pretraining_pin_payload(*, registration_sha256: str) -> dict[str, object]:
    zero = "0" * 64
    payload: dict[str, object] = {
        "schema_version": 1,
        "action": "rcq_v2_pin_pretraining_v1",
        "qualification_id": "rcq_v2_reference_v2",
        "workspace": {
            "contract": "pseudo-brain-workspace-v2",
            "host_account_home_relative_path": "projects/pseudo-brain",
            "marker_relative_path": ".pseudo-brain-workspace-v2",
            "marker_file_sha256": sha256(
                b"pseudo-brain-workspace-v2\n"
            ).hexdigest(),
            "claim_registry_relative_path": "final-claims",
            "pin_directory_relative_path": (
                "qualification-pins/rcq-v2-reference-v2"
            ),
        },
        "release": {
            "id": "release-001",
            "relative_path": "releases/release-001",
            "archive_sha256": "1" * 64,
        },
        "registration": {
            "release_relative_path": "registrations/rcq-v2-reference-v2.json",
            "sha256": registration_sha256,
        },
        "config": {
            "release_relative_path": (
                "brain/configs/training/dgx-rcq-v2-reference.toml"
            ),
            "raw_sha256": zero,
            "canonical_sha256": zero,
        },
        "source_tree_sha256": zero,
        "evaluator_bundle_sha256": zero,
        "batch_source_manifest_sha256": zero,
        "runtime": {
            "container_image_reference": "pseudo-brain@sha256:" + "2" * 64,
            "container_image_id": "sha256:" + "2" * 64,
            "container_release_root": "/workspace/repo",
            "container_run_root": "/workspace/run",
            "container_claim_registry_root": "/workspace/final-claims",
            "container_pin_root": "/workspace/pins",
            "network": "none",
        },
        "run": {
            "id": "dgx-rcq-v2-reference-seed-1702",
            "relative_path": "runs/dgx-rcq-v2-reference-seed-1702",
            "seed": 1702,
            "final_step": final.FINAL_STEP,
        },
        "range_claim_id": final._final_range_claim_id(),
        "created_utc": "2026-08-16T12:00:00Z",
    }
    payload["pin_sha256"] = final._digest_payload(
        torch_runner._PRETRAINING_PIN_DOMAIN,
        payload,
    )
    return payload


def _final_authorization_payload(
    *,
    pretraining_file_sha256: str,
    pretraining_pin_sha256: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "action": "rcq_v2_authorize_final_v1",
        "qualification_id": "rcq_v2_reference_v2",
        "pretraining": {
            "relative_path": "pretraining.json",
            "file_sha256": pretraining_file_sha256,
            "pin_sha256": pretraining_pin_sha256,
        },
        "latest": {
            "run_relative_path": "runs/dgx-rcq-v2-reference-seed-1702",
            "relative_path": "checkpoints/latest.json",
            "file_sha256": "3" * 64,
            "checkpoint": "step-00002048.pt",
            "checkpoint_sha256": "4" * 64,
            "optimizer_step": final.FINAL_STEP,
        },
        "entry_checkpoint": {
            "relative_path": "checkpoints/step-00001536.pt",
            "sha256": "9" * 64,
        },
        "checkpoint": {
            "relative_path": "checkpoints/step-00002048.pt",
            "sha256": "4" * 64,
        },
        "readiness": {
            "relative_path": (
                "final-claims/preclaim-readiness/rcq-v2-reference-v2.json"
            ),
            "file_sha256": "5" * 64,
            "readiness_sha256": "6" * 64,
        },
        "range_claim_id": final._final_range_claim_id(),
        "reviewed_utc": "2026-08-16T13:00:00Z",
    }
    payload["authorization_sha256"] = final._digest_payload(
        torch_runner._FINAL_AUTHORIZATION_DOMAIN,
        payload,
    )
    return payload


class RCQV2FinalTrustTests(unittest.TestCase):
    def test_public_surface_has_no_receipt_author_or_evidence_injection(self) -> None:
        self.assertNotIn("FinalAuthorization", final.__all__)
        self.assertNotIn("RCQFinalEvidence", final.__all__)
        self.assertNotIn("RCQFinalReceipt", final.__all__)
        self.assertFalse(hasattr(final, "evaluate_rcq_v2_final_once"))
        self.assertFalse(hasattr(final, "_evaluate_rcq_v2_final_once_trusted"))
        self.assertEqual(torch_runner.__all__, ["evaluator_bundle_sha256"])
        signature = inspect.signature(torch_runner._evaluate_rcq_v2_checkpoint_final_once)
        self.assertNotIn("receipt_root", signature.parameters)
        self.assertNotIn("evidence", signature.parameters)
        self.assertNotIn("model", signature.parameters)
        self.assertEqual(signature.parameters, {})
        parser = torch_runner._parser()
        options = set()
        for action in parser._actions:
            options.update(action.option_strings)
            for child in (getattr(action, "choices", None) or {}).values():
                for child_action in child._actions:
                    options.update(child_action.option_strings)
        self.assertNotIn("--receipt-root", options)
        self.assertNotIn("--readiness-sha256", options)
        self.assertNotIn("--registration", options)
        self.assertNotIn("--checkpoint", options)
        preregistration_options = {
            option
            for action in preregistration._parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--output", preregistration_options)
        self.assertEqual(
            torch_runner._REGISTRATION_RELATIVE_PATH.as_posix(),
            "registrations/rcq-v2-reference-v2.json",
        )

    def test_historical_v1_registration_bytes_are_preserved(self) -> None:
        historical = (
            Path(__file__).resolve().parents[2]
            / "registrations"
            / "rcq-v2-reference-v1.json"
        )
        encoded = historical.read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
        self.assertEqual(payload["qualification_id"], "rcq_v2_reference_v1")
        self.assertEqual(payload["slices"]["final_recipient"]["local_start"], 3_145_728)
        self.assertEqual(
            sha256(encoded).hexdigest(),
            "33f7900c1d71b5e363de5a6b7ca921120f486b315241384d906d209a5e02fce0",
        )
        self.assertNotEqual(
            torch_runner._REGISTRATION_RELATIVE_PATH.as_posix(),
            "registrations/rcq-v2-reference-v1.json",
        )

    def test_live_v2_registration_bytes_are_pinned(self) -> None:
        live = (
            Path(__file__).resolve().parents[2]
            / "registrations"
            / "rcq-v2-reference-v2.json"
        )
        encoded = live.read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
        self.assertEqual(payload["qualification_id"], "rcq_v2_reference_v2")
        self.assertEqual(payload["slices"]["final_recipient"]["local_start"], 3_145_728)
        self.assertEqual(payload["run_id"], "dgx-rcq-v2-reference-seed-1702")
        self.assertEqual(
            sha256(encoded).hexdigest(),
            "6cc98739c78499a990a4b3480524c48dd49243c1e3c63094977a9a917df49690",
        )
        self.assertEqual(
            torch_runner._REGISTRATION_RELATIVE_PATH.as_posix(),
            "registrations/rcq-v2-reference-v2.json",
        )

    def test_registration_parent_may_hold_historical_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw) / "registrations"
            parent.mkdir()
            (parent / "rcq-v2-reference-v1.json").write_bytes(b"{}\n")
            resolved = preregistration._safe_registration_parent(parent)
            self.assertEqual(resolved, parent.resolve(strict=True))
            self.assertTrue((resolved / "rcq-v2-reference-v1.json").is_file())
            self.assertFalse((resolved / "rcq-v2-reference-v2.json").exists())

    def test_cli_help_and_dispatch_derive_all_identities_without_injection(self) -> None:
        help_text = torch_runner._parser().format_help()
        self.assertIn("preclaim", help_text)
        self.assertIn("final-once", help_text)
        self.assertIn("verify-receipt", help_text)
        with mock.patch.object(
            torch_runner,
            "_write_rcq_v2_preclaim_readiness",
            return_value=(Path("/workspace/final-claims/preclaim.json"), "3" * 64),
        ) as preclaim, redirect_stdout(io.StringIO()):
            self.assertEqual(torch_runner.main(["preclaim"]), 0)
        preclaim.assert_called_once_with()
        with mock.patch.object(
            torch_runner,
            "_evaluate_rcq_v2_checkpoint_final_once",
            return_value=SimpleNamespace(canonical_json='{"passed":true}', passed=True),
        ) as final_once, redirect_stdout(io.StringIO()):
            self.assertEqual(torch_runner.main(["final-once"]), 0)
        final_once.assert_called_once_with()
        with mock.patch.object(
            torch_runner,
            "_verify_rcq_v2_final_receipt",
            return_value={
                "verification_status": "authoritative_terminal_receipt",
                "terminal_status": "scientific_failed",
                "passed": False,
            },
        ) as verify, redirect_stdout(io.StringIO()):
            self.assertEqual(torch_runner.main(["verify-receipt"]), 0)
        verify.assert_called_once_with()
        with (
            self.assertRaises(SystemExit),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            torch_runner._parser().parse_args(
                ["final-once", "--registration", "/tmp/alternate.json"]
            )

    def test_terminal_receipt_is_strict_commit_for_all_terminal_statuses(self) -> None:
        class FakeStore:
            def __init__(self, files: dict[str, bytes]) -> None:
                self.files = files

            def read_bytes(self, name: str) -> bytes:
                try:
                    return self.files[name]
                except KeyError as error:
                    raise RCQInputError(f"missing {name}") from error

            def require_entries(self, expected: set[str]) -> None:
                if set(self.files) != expected:
                    raise RCQInputError("entry set changed")

        readiness = {
            "file": "rcq-v2-reference-v2.json",
            "file_sha256": "1" * 64,
            "readiness_sha256": "2" * 64,
        }
        authorization = {"fixed": "authorization"}
        retired = [{"role": "final_recipient"}]
        guard = {"status": "unused"}
        bindings = {
            "qualification_id": "rcq_v2_reference_v2",
            "readiness_binding": readiness,
            "authorization": authorization,
            "retired_ranges": retired,
            "guard_band": guard,
            "future_campaign_offset": final.FUTURE_CAMPAIGN_OFFSET,
        }

        def canonical_bytes(payload: dict[str, object]) -> bytes:
            return (final._canonical(payload) + "\n").encode("utf-8")

        def fixture(status: str) -> tuple[FakeStore, dict[str, object]]:
            claim: dict[str, object] = {
                "schema_version": 1,
                "qualification_id": "rcq_v2_reference_v2",
                "status": "claimed_test_retired",
                "preclaim_readiness": readiness,
                "authorization": authorization,
                "retired_ranges": retired,
                "guard_band": guard,
                "future_campaign_offset": final.FUTURE_CAMPAIGN_OFFSET,
                "reserved_excluded_ranges": final._reserved_excluded_ranges(),
                "retry_permitted": False,
            }
            claim["claim_sha256"] = final._digest_payload(
                b"IRENERCQCLAIM\x01", claim
            )
            files = {torch_runner._CLAIM_NAME: canonical_bytes(claim)}
            range_reference: dict[str, object] = {"status": "not_published"}
            decisions: dict[str, object] = {"status": "not_published"}
            donors: dict[str, object] = {"status": "not_published"}
            if status != "invalid_after_claim":
                range_payload: dict[str, object] = {
                    "schema_version": 1,
                    "qualification_id": "rcq_v2_reference_v2",
                    "claim_sha256": claim["claim_sha256"],
                    "ranges": final._reserved_excluded_ranges(),
                }
                range_payload["ledger_sha256"] = final._digest_payload(
                    b"IRENERCQRANGES\x01", range_payload
                )
                range_bytes = canonical_bytes(range_payload)
                decision_bytes = b'{"decision":1}\n'
                donor_bytes = b'{"donor":1}\n'
                files.update(
                    {
                        torch_runner._RANGE_LEDGER_NAME: range_bytes,
                        torch_runner._DECISION_LEDGER_NAME: decision_bytes,
                        torch_runner._DONOR_LEDGER_NAME: donor_bytes,
                    }
                )
                range_reference = {
                    "status": "published_verified",
                    "file": torch_runner._RANGE_LEDGER_NAME,
                    "records": len(final._reserved_excluded_ranges()),
                    "sha256": sha256(range_bytes).hexdigest(),
                    "ledger_sha256": range_payload["ledger_sha256"],
                }
                decisions = {
                    "status": "published_verified",
                    "file": torch_runner._DECISION_LEDGER_NAME,
                    "records": 1,
                    "sha256": sha256(decision_bytes).hexdigest(),
                }
                donors = {
                    "status": "published_verified",
                    "file": torch_runner._DONOR_LEDGER_NAME,
                    "records": 1,
                    "sha256": sha256(donor_bytes).hexdigest(),
                    "assignment_cost": {"total": 0},
                }
            terminal: dict[str, object] = {
                "schema_version": 1,
                "qualification_id": "rcq_v2_reference_v2",
                "status": status,
                "passed": status == "passed",
                "claim_sha256": claim["claim_sha256"],
                "preclaim_readiness": readiness,
                "donor_assignment_cost": {"status": "test"},
                "authorization": authorization,
                "retired_ranges": retired,
                "guard_band": guard,
                "future_campaign_offset": final.FUTURE_CAMPAIGN_OFFSET,
                "range_ledger": range_reference,
                "retry_permitted": False,
            }
            if status == "invalid_after_claim":
                terminal.update(
                    {
                        "raw_ledgers": {"decisions": decisions, "donors": donors},
                        "error_type": "RuntimeError",
                        "error": "caught failure",
                    }
                )
            else:
                terminal["report"] = {
                    "passed": status == "passed",
                    "raw_ledgers": {"decisions": decisions, "donors": donors},
                }
            terminal["receipt_sha256"] = final._digest_payload(
                b"IRENERCQRECEIPT\x01", terminal
            )
            files[torch_runner._RECEIPT_NAME] = canonical_bytes(terminal)
            return FakeStore(files), terminal

        for status in ("passed", "scientific_failed", "invalid_after_claim"):
            with self.subTest(status=status):
                store, _terminal = fixture(status)
                receipt = torch_runner._validate_terminal_receipt_commit(
                    store,  # type: ignore[arg-type]
                    bindings,
                )
                self.assertEqual(receipt.status, status)

        absent, _terminal = fixture("passed")
        absent.files.pop(torch_runner._RECEIPT_NAME)
        with self.assertRaises(RCQInputError):
            torch_runner._validate_terminal_receipt_commit(
                absent,  # type: ignore[arg-type]
                bindings,
            )
        partial, _terminal = fixture("scientific_failed")
        partial.files.pop(torch_runner._DONOR_LEDGER_NAME)
        with self.assertRaises(RCQInputError):
            torch_runner._validate_terminal_receipt_commit(
                partial,  # type: ignore[arg-type]
                bindings,
            )
        tampered, _terminal = fixture("passed")
        tampered.files[torch_runner._DONOR_LEDGER_NAME] = b'{"donor":2}\n'
        with self.assertRaises(RCQInputError):
            torch_runner._validate_terminal_receipt_commit(
                tampered,  # type: ignore[arg-type]
                bindings,
            )
        malformed, terminal = fixture("scientific_failed")
        terminal["receipt_sha256"] = "0" * 64
        malformed.files[torch_runner._RECEIPT_NAME] = canonical_bytes(terminal)
        with self.assertRaises(RCQInputError):
            torch_runner._validate_terminal_receipt_commit(
                malformed,  # type: ignore[arg-type]
                bindings,
            )

    def test_two_phase_pin_documents_are_strict_canonical_and_semantically_bound(self) -> None:
        registration_payload = _registration_payload()
        registration_bytes = (
            final._canonical(registration_payload) + "\n"
        ).encode("utf-8")
        registration_sha = sha256(registration_bytes).hexdigest()
        pretraining_payload = _pretraining_pin_payload(
            registration_sha256=registration_sha
        )
        with mock.patch.object(
            torch_runner,
            "_strict_owned_pin_file",
            return_value=(pretraining_payload, "7" * 64),
        ):
            pretraining = torch_runner._load_pretraining_pin()
        self.assertEqual(
            pretraining.semantic_sha256,
            pretraining_payload["pin_sha256"],
        )
        registration = final.RCQRegistration(
            payload=registration_payload,
            sha256=registration_sha,
        )
        with mock.patch.object(
            torch_runner,
            "_load_pretraining_pin",
            return_value=pretraining,
        ):
            torch_runner._require_pretraining_pin_matches(
                pretraining,
                registration=registration,
                config=SimpleNamespace(config_sha256="0" * 64),  # type: ignore[arg-type]
                source_digest="0" * 64,
                evaluator_digest="0" * 64,
                batch_source_digest="0" * 64,
            )

        tampered = json.loads(final._canonical(pretraining_payload))
        tampered["runtime"]["network"] = False
        with mock.patch.object(
            torch_runner,
            "_strict_owned_pin_file",
            return_value=(tampered, "7" * 64),
        ), self.assertRaises(RCQInputError):
            torch_runner._load_pretraining_pin()

        final_payload = _final_authorization_payload(
            pretraining_file_sha256=pretraining.file_sha256,
            pretraining_pin_sha256=pretraining.semantic_sha256,
        )
        with mock.patch.object(
            torch_runner,
            "_strict_owned_pin_file",
            return_value=(final_payload, "8" * 64),
        ):
            authorization = torch_runner._load_final_authorization()
        self.assertEqual(
            authorization.semantic_sha256,
            final_payload["authorization_sha256"],
        )
        final_tamper = json.loads(final._canonical(final_payload))
        final_tamper["latest"]["optimizer_step"] = True
        with mock.patch.object(
            torch_runner,
            "_strict_owned_pin_file",
            return_value=(final_tamper, "8" * 64),
        ), self.assertRaises(RCQInputError):
            torch_runner._load_final_authorization()
        entry_tamper = json.loads(final._canonical(final_payload))
        entry_tamper["entry_checkpoint"]["relative_path"] = (
            "checkpoints/step-00002048.pt"
        )
        with mock.patch.object(
            torch_runner,
            "_strict_owned_pin_file",
            return_value=(entry_tamper, "8" * 64),
        ), self.assertRaises(RCQInputError):
            torch_runner._load_final_authorization()

    def test_claim_store_is_locked_dirfd_bound_before_test_materialization(self) -> None:
        lock_source = inspect.getsource(torch_runner._open_locked_claim_registry)
        chain_source = inspect.getsource(torch_runner._open_plain_directory_chain)
        store_source = inspect.getsource(torch_runner._BoundReceiptDirectory)
        final_source = inspect.getsource(
            torch_runner._evaluate_rcq_v2_checkpoint_final_once
        )
        self.assertIn("fcntl.flock", lock_source)
        self.assertIn("LOCK_EX", lock_source)
        self.assertIn("O_NOFOLLOW", chain_source)
        self.assertIn("dir_fd=", store_source)
        self.assertIn("os.fsync(self._root_descriptor)", store_source)
        claim = final_source.index("receipt_store.publish(\n        _CLAIM_NAME")
        materialize = final_source.index("_collect_post_claim_evidence(")
        self.assertLess(claim, materialize)

    @unittest.skipUnless(os.name == "posix", "Linux dirfd/flock trust boundary")
    def test_bound_claim_store_rejects_collision_rebind_and_tampering(self) -> None:
        policy = ResourcePolicy(allow_artifact_write=True)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            registry = base / "registry"
            registry.mkdir()
            registry.chmod(0o700)
            leaf = "a" * 64
            store = torch_runner._BoundReceiptDirectory(registry, leaf)
            content = b'{"claim":true}\n'
            digest = store.publish(
                torch_runner._CLAIM_NAME,
                content,
                existing=set(),
                policy=policy,
            )
            self.assertEqual(digest, sha256(content).hexdigest())
            store.require_entries({torch_runner._CLAIM_NAME})
            with self.assertRaises(RCQInputError):
                torch_runner._BoundReceiptDirectory(registry, leaf)
            with self.assertRaises(RCQInputError):
                store.publish(
                    torch_runner._CLAIM_NAME,
                    content,
                    existing={torch_runner._CLAIM_NAME},
                    policy=policy,
                )

            extra = store.path / "unexpected"
            extra.write_bytes(b"x")
            with self.assertRaisesRegex(RCQInputError, "entries"):
                store.require_entries({torch_runner._CLAIM_NAME})
            extra.unlink()

            claim_path = store.path / torch_runner._CLAIM_NAME
            claim_path.chmod(0o600)
            claim_path.write_bytes(b'{"claim":false}\n')
            with self.assertRaises(RCQInputError):
                store.require_exact(torch_runner._CLAIM_NAME, content)
            store.close()
            with self.assertRaises(RCQInputError):
                torch_runner._BoundReceiptDirectory(registry, leaf)

            symlink_leaf = "b" * 64
            outside = base / "outside"
            outside.mkdir()
            os.symlink(outside, registry / symlink_leaf, target_is_directory=True)
            with self.assertRaises((RCQInputError, OSError)):
                torch_runner._BoundReceiptDirectory(registry, symlink_leaf)

            rebound_registry = base / "rebound-registry"
            rebound_registry.mkdir()
            rebound_registry.chmod(0o700)
            rebound = torch_runner._BoundReceiptDirectory(
                rebound_registry,
                "c" * 64,
            )
            moved = base / "moved-registry"
            rebound_registry.rename(moved)
            rebound_registry.mkdir()
            rebound_registry.chmod(0o700)
            with self.assertRaises(RCQInputError):
                rebound.assert_bound()
            rebound.close()

    def test_registration_is_canonical_strict_and_deeply_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, digest = _write_registration(root, _registration_payload())
            registration = final.load_rcq_v2_registration(
                path,
                expected_sha256=digest,
            )
            with self.assertRaises(TypeError):
                registration.payload["slices"]["train"]["sequences"] = 1  # type: ignore[index]
            with self.assertRaises(TypeError):
                registration.payload["runtime_protocol"]["allow_tf32"] = False  # type: ignore[index]

            pretty = root / "pretty.json"
            encoded = json.dumps(_registration_payload(), indent=2).encode("utf-8")
            pretty.write_bytes(encoded)
            with self.assertRaisesRegex(RCQInputError, "byte-canonical"):
                final.load_rcq_v2_registration(
                    pretty,
                    expected_sha256=sha256(encoded).hexdigest(),
                )

    def test_registration_rejects_bool_for_integer_runtime_field(self) -> None:
        payload = _registration_payload()
        payload["runtime_protocol"]["num_workers"] = False  # type: ignore[index]
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = _write_registration(Path(temporary), payload)
            with self.assertRaisesRegex(RCQInputError, "runtime protocol"):
                final.load_rcq_v2_registration(path, expected_sha256=digest)

    def test_registration_and_import_do_not_index_any_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = _write_registration(
                Path(temporary),
                _registration_payload(),
            )
            original_init = MovingShapesSequenceDataset.__init__

            def guarded_init(instance: object, config: object) -> None:
                split = getattr(config, "split", None)
                start = getattr(config, "seed_offset", -1)
                count = getattr(config, "sequence_count", 0)
                overlaps_sealed = (
                    split is final.DatasetSplit.TEST
                    and start < final.DONOR_OFFSET + final.DONOR_SEQUENCES
                    and start + count > final.RECIPIENT_OFFSET
                )
                if overlaps_sealed:
                    raise AssertionError("sealed TEST construction is forbidden")
                original_init(instance, config)  # type: ignore[arg-type]

            with mock.patch.object(
                MovingShapesSequenceDataset,
                "__init__",
                new=guarded_init,
            ), mock.patch.object(
                MovingShapesSequenceDataset,
                "__getitem__",
                side_effect=AssertionError("dataset indexing is forbidden"),
            ), mock.patch.object(
                MovingShapesSequenceDataset,
                "__iter__",
                side_effect=AssertionError("dataset iteration is forbidden"),
                create=True,
            ):
                final.load_rcq_v2_registration(path, expected_sha256=digest)
                importlib.reload(torch_runner)
                live_release = Path(__file__).resolve().parents[2]
                release = Path(temporary) / "clean-release"
                live_source = live_release / "brain" / "src" / "irene_brain"
                clean_source = release / "brain" / "src" / "irene_brain"
                for source_file in live_source.rglob("*.py"):
                    target = clean_source / source_file.relative_to(live_source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source_file.read_bytes())
                config = (
                    release
                    / "brain"
                    / "configs"
                    / "training"
                    / "dgx-rcq-v2-reference.toml"
                )
                config.parent.mkdir(parents=True, exist_ok=True)
                config.write_bytes(
                    (
                        live_release
                        / "brain"
                        / "configs"
                        / "training"
                        / "dgx-rcq-v2-reference.toml"
                    ).read_bytes()
                )
                with mock.patch.object(
                    preregistration.importlib,
                    "import_module",
                    return_value=SimpleNamespace(
                        __file__=str(clean_source / "__init__.py")
                    ),
                ):
                    encoded, built_digest = (
                        preregistration.build_target_blind_registration(
                            training_release_root=release,
                            config_path=config,
                        )
                    )
                self.assertEqual(sha256(encoded).hexdigest(), built_digest)

    def test_source_tree_requires_one_plain_python_only_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_parent = Path(temporary) / "src"
            package = source_parent / "irene_brain"
            nested = package / "nested"
            nested.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (nested / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
            self.assertEqual(
                torch_runner._source_tree_sha256_exact(package),
                source_tree_sha256(package),
            )

            for relative in (
                "native.pyd",
                "native.so",
                "module.pyc",
                "__pycache__/module.pyc",
            ):
                with self.subTest(relative=relative):
                    target = package / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"not executable")
                    with self.assertRaises(RCQInputError):
                        torch_runner._source_tree_sha256_exact(package)
                    target.unlink()
                    if target.parent.name == "__pycache__":
                        target.parent.rmdir()

            sibling = source_parent / "sitecustomize.py"
            sibling.write_text("raise RuntimeError\n", encoding="utf-8")
            with self.assertRaisesRegex(RCQInputError, "source parent"):
                torch_runner._source_tree_sha256_exact(package)
            sibling.unlink()

            hardlink = Path(temporary) / "external-hardlink.py"
            try:
                os.link(package / "__init__.py", hardlink)
            except OSError:
                hardlink = None  # type: ignore[assignment]
            if hardlink is not None:
                with self.assertRaisesRegex(RCQInputError, "hardlink"):
                    torch_runner._source_tree_sha256_exact(package)
                hardlink.unlink()

            outside = Path(temporary) / "outside"
            outside.mkdir()
            linked = package / "linked"
            try:
                os.symlink(outside, linked, target_is_directory=True)
            except OSError:
                linked = None  # type: ignore[assignment]
            if linked is not None:
                with self.assertRaisesRegex(RCQInputError, "link or reparse"):
                    torch_runner._source_tree_sha256_exact(package)
                linked.unlink()

    def test_claim_publish_precedes_concrete_test_materializer_in_source(self) -> None:
        source = inspect.getsource(torch_runner._evaluate_rcq_v2_checkpoint_final_once)
        claim = source.index("receipt_store.publish(\n        _CLAIM_NAME")
        materialize = source.index("_collect_post_claim_evidence(")
        self.assertLess(claim, materialize)
        helper = inspect.getsource(torch_runner._collect_post_claim_evidence)
        read_claim = helper.index("receipt_store.read_bytes(_CLAIM_NAME)")
        construct_test = helper.index("_registered_test_dataset(")
        self.assertLess(read_claim, construct_test)
        self.assertIn("_claimed_rcq_v2_test_dataset", inspect.getsource(
            torch_runner._registered_test_dataset
        ))
        module_source = inspect.getsource(torch_runner)
        outside_materializer = module_source.replace(helper, "")
        self.assertNotIn("= _registered_test_dataset(", outside_materializer)
        preclaim_source = inspect.getsource(torch_runner._prepare_trusted_preclaim)
        self.assertNotIn("MovingShapesBatchSource", preclaim_source)
        preclaim_batch_init = inspect.getsource(torch_runner._PreclaimBatchSource.__init__)
        self.assertNotIn("DatasetSplit.TEST", preclaim_batch_init)

    def test_donor_mapping_recomputes_assignment_digest(self) -> None:
        assignment = final.DonorFrameAssignment(
            recipient_episode_seed=1,
            time_index=0,
            donor_episode_seed=2,
            recipient_rgb_sha256="1" * 64,
            donor_rgb_sha256="2" * 64,
            recipient_previous_wasd_mask=1,
            donor_previous_wasd_mask=3,
            previous_wasd_bit_hamming=1,
        )
        digest = final._digest_payload(
            b"IRENERCQDONOR\x01",
            [assignment.to_dict()],
        )
        mapping = final.DonorMapping(assignments=(assignment,), sha256=digest)
        self.assertEqual(mapping.sha256, digest)
        tampered = final.DonorFrameAssignment(
            recipient_episode_seed=1,
            time_index=0,
            donor_episode_seed=3,
            recipient_rgb_sha256="1" * 64,
            donor_rgb_sha256="3" * 64,
            recipient_previous_wasd_mask=1,
            donor_previous_wasd_mask=3,
            previous_wasd_bit_hamming=1,
        )
        with self.assertRaisesRegex(RCQInputError, "differs from its assignments"):
            final.DonorMapping(assignments=(tampered,), sha256=digest)
        with self.assertRaisesRegex(RCQInputError, "time_index"):
            final.DonorFrameAssignment(
                recipient_episode_seed=1,
                time_index=True,  # type: ignore[arg-type]
                donor_episode_seed=2,
                recipient_rgb_sha256="1" * 64,
                donor_rgb_sha256="2" * 64,
                recipient_previous_wasd_mask=1,
                donor_previous_wasd_mask=3,
                previous_wasd_bit_hamming=1,
            )
        with self.assertRaisesRegex(RCQInputError, "Hamming"):
            final.DonorFrameAssignment(
                recipient_episode_seed=1,
                time_index=0,
                donor_episode_seed=2,
                recipient_rgb_sha256="1" * 64,
                donor_rgb_sha256="2" * 64,
                recipient_previous_wasd_mask=1,
                donor_previous_wasd_mask=3,
                previous_wasd_bit_hamming=0,
            )

    def test_donor_sequence_assignment_is_minimum_cost_and_one_to_one(self) -> None:
        forbidden = 1_000
        matrices = (
            ((0,),),
            ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
            ((0, 4, 5), (4, 0, 6), (5, 6, 0)),
            ((10, 1, 1), (1, 10, 1), (1, 1, 10)),
            ((9, 2, 7, 8), (6, 4, 3, 7), (5, 8, 1, 8), (7, 6, 9, 4)),
            ((forbidden, 1, 2), (2, forbidden, 1), (1, 2, forbidden)),
        )
        for costs in matrices:
            with self.subTest(costs=costs):
                expected_cost = _brute_force_assignment_cost(
                    costs,
                    forbidden=forbidden,
                )
                self.assertIsNotNone(expected_cost)
                first = final._minimum_cost_permutation(
                    costs=costs,
                    forbidden=forbidden,
                )
                second = final._minimum_cost_permutation(
                    costs=costs,
                    forbidden=forbidden,
                )
                self.assertEqual(first, second)
                self.assertEqual(
                    sum(costs[row][column] for row, column in enumerate(first)),
                    expected_cost,
                )
        self.assertEqual(
            final._minimum_cost_permutation(
                costs=((0, 0, 0), (0, 0, 0), (0, 0, 0)),
                forbidden=forbidden,
            ),
            (0, 1, 2),
        )
        with self.assertRaisesRegex(RCQInputError, "RGB-distinct"):
            final._minimum_cost_permutation(
                costs=((forbidden, 0), (forbidden, forbidden)),
                forbidden=forbidden,
            )

        recipient_masks = (
            (0, 1, 2, 4, 8, 3, 6, 12),
            (1, 1, 1, 1, 2, 2, 2, 2),
            (15, 14, 13, 11, 7, 0, 5, 10),
            (3, 3, 6, 6, 12, 12, 9, 9),
        )
        donor_masks = (
            (0, 1, 3, 4, 8, 2, 6, 12),
            (15, 15, 15, 15, 14, 14, 14, 14),
            (3, 2, 1, 0, 4, 8, 12, 6),
            (5, 5, 10, 10, 1, 2, 4, 8),
        )
        recipients = tuple(
            _fake_sequence(
                episode_seed=10 + index,
                rgb_namespace="recipient",
                masks=masks,
                target_marker=1,
            )
            for index, masks in enumerate(recipient_masks)
        )
        donors = tuple(
            _fake_sequence(
                episode_seed=100 + index,
                rgb_namespace="donor",
                masks=masks,
                target_marker=2,
            )
            for index, masks in enumerate(donor_masks)
        )
        mapping = final.build_target_blind_donor_mapping(
            recipients,  # type: ignore[arg-type]
            donors,  # type: ignore[arg-type]
        )
        self.assertEqual(len(mapping.assignments), 4 * 8)
        pairs: dict[int, set[int]] = {}
        donor_frames = set()
        for assignment in mapping.assignments:
            pairs.setdefault(assignment.recipient_episode_seed, set()).add(
                assignment.donor_episode_seed
            )
            donor_frames.add(
                (assignment.donor_episode_seed, assignment.time_index)
            )
            self.assertNotEqual(
                assignment.recipient_rgb_sha256,
                assignment.donor_rgb_sha256,
            )
        self.assertTrue(all(len(values) == 1 for values in pairs.values()))
        self.assertEqual(len({next(iter(values)) for values in pairs.values()}), 4)
        self.assertEqual(len(donor_frames), 4 * 8)
        changed_targets = tuple(
            _fake_sequence(
                episode_seed=10 + index,
                rgb_namespace="recipient",
                masks=masks,
                target_marker=999,
            )
            for index, masks in enumerate(recipient_masks)
        )
        changed_donors = tuple(
            _fake_sequence(
                episode_seed=100 + index,
                rgb_namespace="donor",
                masks=masks,
                target_marker=-999,
            )
            for index, masks in enumerate(donor_masks)
        )
        target_changed_mapping = final.build_target_blind_donor_mapping(
            changed_targets,  # type: ignore[arg-type]
            changed_donors,  # type: ignore[arg-type]
        )
        self.assertEqual(mapping.sha256, target_changed_mapping.sha256)
        summary = mapping.assignment_cost_summary()
        self.assertEqual(summary["donor_matching_seed"], final.DONOR_MATCHING_SEED)
        self.assertEqual(summary["pair_count"], 4)
        self.assertEqual(
            summary["total_previous_wasd_bit_hamming"],
            sum(item.previous_wasd_bit_hamming for item in mapping.assignments),
        )

    def test_live_value_gate_constants_are_bound_to_registration(self) -> None:
        payload = _registration_payload()
        encoded = (final._canonical(payload) + "\n").encode("utf-8")
        registration = final.RCQRegistration(
            payload=payload,
            sha256=sha256(encoded).hexdigest(),
        )
        torch_runner._require_live_development_gate_protocol(registration)
        with mock.patch.object(torch_runner, "VALUE_ABSOLUTE_MAX_MSE", 0.0):
            with self.assertRaisesRegex(RCQInputError, "absolute value-loss"):
                torch_runner._require_live_development_gate_protocol(registration)

    def test_gate_metric_link_rejects_all_boundary_drift_classes(self) -> None:
        report = {"action": 0.25, "total_loss": 0.5}
        durable = {
            **report,
            "loss": 0.5,
            "samples": 1_536.0,
            "stage_index": 0.0,
        }
        torch_runner._require_gate_metrics_link(
            report_metrics=report,
            validation_metrics=durable,
            name="entry",
        )
        variants = {
            "extra": {**durable, "unregistered": 1.0},
            "missing": {key: value for key, value in durable.items() if key != "action"},
            "type": {**durable, "action": 1},
            "swapped": {**durable, "action": 0.75},
            "negative_zero": {**durable, "action": -0.0},
        }
        zero_report = {"action": 0.0, "total_loss": 0.5}
        for name, variant in variants.items():
            selected_report = zero_report if name == "negative_zero" else report
            with self.subTest(name=name), self.assertRaises(RCQInputError):
                torch_runner._require_gate_metrics_link(
                    report_metrics=selected_report,
                    validation_metrics=variant,
                    name="entry",
                )

    def test_distinct_entry_and_completion_boundary_records_cannot_be_swapped(self) -> None:
        entry_report = {
            "action": 0.25,
            "value_loss": 0.30,
            "total_loss": 0.60,
        }
        completion_report = {
            "action": 0.25,
            "value_loss": 0.20,
            "total_loss": 0.50,
        }
        step_1536 = {
            **entry_report,
            "loss": 0.60,
            "samples": 1_536.0,
            "stage_index": 0.0,
        }
        step_2048 = {
            **completion_report,
            "loss": 0.50,
            "samples": 1_536.0,
            "stage_index": 1.0,
        }
        torch_runner._require_gate_metrics_link(
            report_metrics=entry_report,
            validation_metrics=step_1536,
            name="entry",
        )
        torch_runner._require_gate_metrics_link(
            report_metrics=completion_report,
            validation_metrics=step_2048,
            name="completion",
        )
        with self.assertRaises(RCQInputError):
            torch_runner._require_gate_metrics_link(
                report_metrics=entry_report,
                validation_metrics=step_2048,
                name="entry with swapped step-2048 record",
            )
        with self.assertRaises(RCQInputError):
            torch_runner._require_gate_metrics_link(
                report_metrics=completion_report,
                validation_metrics=step_1536,
                name="completion with swapped step-1536 record",
            )

    def test_development_replay_rejects_model_and_nonvalue_drift(self) -> None:
        replay = TrainingStepResult(
            loss=0.5,
            metrics={"action": 0.25, "value_loss": 0.2, "total_loss": 0.5},
            samples=1_536,
        )
        entry = SimpleNamespace(
            metrics={"action": 0.25, "value_loss": 0.3, "total_loss": 0.6}
        )
        completion = SimpleNamespace(canonical_json="expected")
        durable = {
            "action": 0.25,
            "value_loss": 0.2,
            "total_loss": 0.5,
            "loss": 0.5,
            "samples": 1_536.0,
            "stage_index": 1.0,
        }
        with mock.patch.object(
            torch_runner,
            "evaluate_rcq_v2_value_development",
            return_value=SimpleNamespace(canonical_json="different"),
        ):
            with self.assertRaisesRegex(RCQInputError, "completion evidence"):
                torch_runner._require_development_replay(
                    replay=replay,
                    entry_report=entry,  # type: ignore[arg-type]
                    entry_digest="1" * 64,
                    completion_report=completion,  # type: ignore[arg-type]
                    final_validation_metrics=durable,
                )
        drifted_entry = SimpleNamespace(
            metrics={"action": 0.75, "value_loss": 0.3, "total_loss": 0.6}
        )
        with mock.patch.object(
            torch_runner,
            "evaluate_rcq_v2_value_development",
            return_value=SimpleNamespace(canonical_json="expected"),
        ):
            with self.assertRaisesRegex(RCQInputError, "non-value"):
                torch_runner._require_development_replay(
                    replay=replay,
                    entry_report=drifted_entry,  # type: ignore[arg-type]
                    entry_digest="1" * 64,
                    completion_report=completion,  # type: ignore[arg-type]
                    final_validation_metrics=durable,
                )

    def test_entry_replay_is_bound_to_entry_gate_and_step_1536_metrics(self) -> None:
        replay = TrainingStepResult(
            loss=0.6,
            metrics={"action": 0.25, "value_loss": 0.3, "total_loss": 0.6},
            samples=1_536,
        )
        entry = SimpleNamespace(canonical_json="entry-gate")
        durable = {
            **dict(replay.metrics),
            "loss": replay.loss,
            "samples": float(replay.samples),
            "stage_index": 0.0,
        }
        with mock.patch.object(
            torch_runner,
            "evaluate_rcq_v2_development",
            return_value=SimpleNamespace(canonical_json="entry-gate"),
        ):
            torch_runner._require_entry_development_replay(
                replay=replay,
                entry_report=entry,  # type: ignore[arg-type]
                entry_validation_metrics=durable,
            )
            with self.assertRaisesRegex(RCQInputError, "step-1536"):
                torch_runner._require_entry_development_replay(
                    replay=replay,
                    entry_report=entry,  # type: ignore[arg-type]
                    entry_validation_metrics={**durable, "action": 0.75},
                )
        with mock.patch.object(
            torch_runner,
            "evaluate_rcq_v2_development",
            return_value=SimpleNamespace(canonical_json="forged-entry-gate"),
        ), self.assertRaisesRegex(RCQInputError, "entry-gate evidence"):
            torch_runner._require_entry_development_replay(
                replay=replay,
                entry_report=entry,  # type: ignore[arg-type]
                entry_validation_metrics=durable,
            )

    def test_nonvalue_identity_uses_exact_tensors_and_fresh_probe_outputs(self) -> None:
        import torch

        class FakeObjective:
            def __init__(self, core: float, value: float) -> None:
                self._state = {
                    "model.core.weight": torch.tensor([core], dtype=torch.float32),
                    "model.value_per_thought.weight": torch.tensor(
                        [value], dtype=torch.float32
                    ),
                    "model.value_per_thought.bias": torch.tensor(
                        [value], dtype=torch.float32
                    ),
                }

            def state_dict(self) -> dict[str, object]:
                return dict(self._state)

        snapshot = {
            "schema_version": 1,
            "evaluation_precision": "same_runtime_no_autocast_v1",
            "batch_count": 1,
            "sequence_count": 1,
            "timestep_count": 8,
            "non_value_state_sha256": "1" * 64,
            "action_outputs_sha256": "2" * 64,
            "recurrent_states_sha256": "3" * 64,
        }

        def fake_system(*, core: float, value: float, probe: dict[str, object]) -> object:
            return SimpleNamespace(
                objective=FakeObjective(core, value),
                _capture_invariance_snapshot=lambda _batches: dict(probe),
            )

        stage_state = {
            "invariance_reference": dict(snapshot),
            "invariance_current": dict(snapshot),
        }
        evidence = torch_runner._require_exact_nonvalue_model_identity(
            entry_system=fake_system(core=1.0, value=1.0, probe=snapshot),
            terminal_system=fake_system(core=1.0, value=9.0, probe=snapshot),
            entry_stage_state=stage_state,
            terminal_stage_state=stage_state,
            development_batches=(object(),),
        )
        self.assertEqual(evidence["non_value_tensor_count"], 1)
        with self.assertRaisesRegex(RCQInputError, "non-value tensor"):
            torch_runner._require_exact_nonvalue_model_identity(
                entry_system=fake_system(core=1.0, value=1.0, probe=snapshot),
                terminal_system=fake_system(core=2.0, value=9.0, probe=snapshot),
                entry_stage_state=stage_state,
                terminal_stage_state=stage_state,
                development_batches=(object(),),
            )
        drifted_probe = {**snapshot, "action_outputs_sha256": "4" * 64}
        with self.assertRaisesRegex(RCQInputError, "action/recurrent"):
            torch_runner._require_exact_nonvalue_model_identity(
                entry_system=fake_system(core=1.0, value=1.0, probe=snapshot),
                terminal_system=fake_system(core=1.0, value=9.0, probe=drifted_probe),
                entry_stage_state=stage_state,
                terminal_stage_state={
                    "invariance_reference": drifted_probe,
                    "invariance_current": drifted_probe,
                },
                development_batches=(object(),),
            )
        forged_stage_state = {
            "invariance_reference": drifted_probe,
            "invariance_current": drifted_probe,
        }
        with self.assertRaisesRegex(RCQInputError, "invariance evidence"):
            torch_runner._require_exact_nonvalue_model_identity(
                entry_system=fake_system(core=1.0, value=1.0, probe=snapshot),
                terminal_system=fake_system(core=1.0, value=9.0, probe=snapshot),
                entry_stage_state=forged_stage_state,
                terminal_stage_state=stage_state,
                development_batches=(object(),),
            )

    def test_entry_metrics_checkpoint_binds_exact_prefix_not_future_tail(self) -> None:
        config = SimpleNamespace(
            stages=(SimpleNamespace(end_optimizer_step=1_536), SimpleNamespace()),
            optimization=SimpleNamespace(
                gradient_accumulation_steps=1,
                batch_size=1,
            ),
            dataset=SimpleNamespace(
                validation_sequences=1,
                sequence_length=2,
                burn_in_steps=1,
            ),
            logging=SimpleNamespace(
                log_every_steps=1_536,
                evaluate_every_steps=1_536,
            ),
        )

        def record(split: str, step: int, epoch: int) -> bytes:
            payload = {
                "schema_version": 1,
                "step": step,
                "epoch": epoch,
                "split": split,
                "metrics": {
                    "loss": 0.5,
                    "samples": 1.0,
                    "stage_index": 0.0,
                    "total_loss": 0.5,
                },
            }
            return (final._canonical(payload) + "\n").encode("utf-8")

        prefix = b"".join(
            (
                record("train", 1, 0),
                record("train", 1_536, 1),
                record("validation", 1_536, 1),
            )
        )
        trainer_state = {
            "schema_version": 1,
            "metrics_byte_length": len(prefix),
            "metrics_record_count": 3,
            "metrics_sha256": sha256(prefix).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "metrics.jsonl").write_bytes(prefix + b"future-terminal-tail\n")
            boundary = torch_runner._validate_metrics_prefix(
                run,
                trainer_state,
                config=config,  # type: ignore[arg-type]
                batches_per_epoch=1_536,
                terminal_step=1_536,
            )
        self.assertEqual(set(boundary), {1_536})
        self.assertEqual(boundary[1_536]["total_loss"], 0.5)

    def test_preclaim_restores_full_state_and_rng_before_development_replay(self) -> None:
        source = inspect.getsource(torch_runner._prepare_trusted_preclaim)
        pretraining_pin = source.index("_load_pretraining_pin()")
        entry_load = source.index("entry_loaded = load_checkpoint(")
        entry_restore = source.index("entry_system.restore_checkpoint_state(")
        entry_rng = source.index("entry_system.restore_rng_state(")
        nonvalue = source.index("_require_exact_nonvalue_model_identity(")
        entry_replay = source.index("entry_development_replay =")
        restore = source.index("system.restore_model_for_evaluation(")
        rng = source.index("\n        system.restore_rng_state(")
        replay = source.index("development_replay = _replay_registered_development(")
        claim = inspect.getsource(torch_runner._evaluate_rcq_v2_checkpoint_final_once)
        final_authorization = claim.index("_load_final_authorization()")
        publish = claim.index("receipt_store.publish(\n        _CLAIM_NAME")
        self.assertLess(pretraining_pin, restore)
        self.assertLess(entry_load, entry_restore)
        self.assertLess(entry_restore, entry_rng)
        self.assertLess(entry_rng, nonvalue)
        self.assertLess(nonvalue, entry_replay)
        self.assertLess(restore, rng)
        self.assertLess(rng, replay)
        self.assertLess(final_authorization, publish)
        self.assertNotIn("_collect_post_claim_evidence", source)
        self.assertIn("_require_preclaim_readiness", claim)
        verifier = inspect.getsource(torch_runner._verify_rcq_v2_final_receipt)
        self.assertNotIn("_collect_post_claim_evidence", verifier)
        self.assertNotIn("_registered_test_dataset", verifier)
        self.assertIn("sealed_test_examples_opened_by_this_verifier", verifier)
        self.assertNotIn('"sealed_test_examples_opened":', verifier)

    def test_global_claim_key_depends_only_on_exact_ranges(self) -> None:
        claim_id = final._final_range_claim_id()
        self.assertEqual(len(claim_id), 64)
        payload = _registration_payload()
        self.assertEqual(payload["receipt_directory"], f"final-claims/{claim_id}")
        ranges = final._reserved_excluded_ranges()
        ids = {item["id"] for item in ranges}
        self.assertIn("opened_rcq_guard_v0", ids)
        self.assertTrue(
            {
                "first_matched_hazard_1_suite",
                "first_matched_hazard_3_suite",
                "first_matched_hazard_5_suite",
            }.issubset(ids)
        )


if __name__ == "__main__":
    unittest.main()
