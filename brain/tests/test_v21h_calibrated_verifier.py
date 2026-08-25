"""Fail-closed publication contract for V2.1h calibrated verification."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v21f_hazard_head_probe import (
    _assert_publication_targets_available,
    _require_current_parent_source_bundle,
    _save_checkpoint_create_only,
)
from v21h_verify_calibrated_checkpoint import (
    _assert_output_available,
    _publish_result,
    _source_bundle_matches,
    _validate_calibrated_state_dicts,
    _validate_parent_checkpoint_sha,
)
from v21_cpu_heldout_learning_diagnostic import (
    _assert_diagnostic_targets_available,
    _hash_source_bundle_files,
    _source_bundle,
    _write_json_create_only,
)


class CalibratedVerifierPublicationContract(unittest.TestCase):
    def test_source_bundle_is_deterministic_and_content_bound(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        first = _source_bundle(project_root)
        second = _source_bundle(project_root)
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(len(first["sha256"]), 64)
        self.assertGreaterEqual(len(first["files"]), 100)
        self.assertIn(
            "brain/src/irene_brain/v2/belief_updater.py",
            first["files"],
        )
        self.assertIn(
            "brain/scripts/v21h_verify_calibrated_checkpoint.py",
            first["files"],
        )

    def test_source_bundle_digest_changes_with_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_bytes(b"first")
            first = _hash_source_bundle_files(root, ("source.py",))
            source.write_bytes(b"second")
            second = _hash_source_bundle_files(root, ("source.py",))
            self.assertNotEqual(first["files"], second["files"])
            self.assertNotEqual(first["sha256"], second["sha256"])

    def test_checkpoint_bundle_must_match_parent_current_source_and_calibration(
        self,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        bundle = _source_bundle(project_root)
        calibration = {"source_bundle_sha256": bundle["sha256"]}
        self.assertTrue(
            _source_bundle_matches(bundle, bundle, bundle, calibration)
        )
        changed_current = deepcopy(bundle)
        changed_current["files"]["changed.py"] = "0" * 64
        self.assertFalse(
            _source_bundle_matches(bundle, bundle, changed_current, calibration)
        )
        self.assertFalse(
            _source_bundle_matches(
                bundle,
                bundle,
                bundle,
                {"source_bundle_sha256": "0" * 64},
            )
        )

    def test_calibrator_refuses_stale_parent_source_before_publication(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        bundle = _source_bundle(project_root)
        self.assertEqual(
            _require_current_parent_source_bundle(
                {"source_bundle": bundle},
                project_root,
            ),
            bundle,
        )
        stale = deepcopy(bundle)
        stale["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match current"):
            _require_current_parent_source_bundle(
                {"source_bundle": stale},
                project_root,
            )

    def test_failed_qualification_writes_evidence_then_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed.json"
            payload: dict[str, object] = {
                "schema_version": 1,
                "qualification": {"passed": False, "checks": {"gate": False}},
            }
            with self.assertRaisesRegex(SystemExit, "1"):
                _publish_result(output, payload)
            self.assertTrue(output.is_file())
            self.assertFalse(json.loads(output.read_text())["qualification"]["passed"])

    def test_passed_qualification_returns_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "passed.json"
            payload: dict[str, object] = {
                "schema_version": 1,
                "qualification": {"passed": True, "checks": {"gate": True}},
            }
            _publish_result(output, payload)
            self.assertTrue(output.is_file())

    def test_publication_helpers_refuse_existing_targets_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "evidence.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _assert_output_available(output)
            with self.assertRaises(FileExistsError):
                _assert_publication_targets_available(output, None)
            with self.assertRaises(FileExistsError):
                _assert_diagnostic_targets_available(output)
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")

    def test_publication_helpers_reject_aliased_output_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "same-target"
            with self.assertRaisesRegex(ValueError, "must be distinct"):
                _assert_publication_targets_available(target, target)

    def test_checkpoint_publication_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            first_sha = _save_checkpoint_create_only(
                checkpoint,
                {"value": torch.tensor([1.0])},
            )
            original = checkpoint.read_bytes()
            self.assertEqual(len(first_sha), 64)
            with self.assertRaises(FileExistsError):
                _save_checkpoint_create_only(
                    checkpoint,
                    {"value": torch.tensor([2.0])},
                )
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_diagnostic_json_publication_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "diagnostic.json"
            _write_json_create_only(output, {"value": 1})
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                _write_json_create_only(output, {"value": 2})
            self.assertEqual(output.read_bytes(), original)

    def test_parent_sha_and_hazard_tensor_allowlist_are_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent_path = Path(directory) / "parent.pt"
            _save_checkpoint_create_only(parent_path, {"value": torch.tensor(1)})
            parent_sha = _validate_parent_checkpoint_sha(
                parent_path,
                {"parent_checkpoint_sha256": _sha256(parent_path.read_bytes())},
            )
            self.assertEqual(parent_sha, _sha256(parent_path.read_bytes()))
            with self.assertRaisesRegex(ValueError, "parent checkpoint hash"):
                _validate_parent_checkpoint_sha(
                    parent_path,
                    {"parent_checkpoint_sha256": "0" * 64},
                )

        allowed = "world_model.outcome_model.hazard_head.weight"
        forbidden = "world_model.encoder.weight"
        parent_state = {
            allowed: torch.tensor([1.0]),
            forbidden: torch.tensor([2.0]),
        }
        child_state = {
            allowed: torch.tensor([3.0]),
            forbidden: torch.tensor([2.0]),
        }
        self.assertEqual(
            _validate_calibrated_state_dicts(parent_state, child_state, [allowed]),
            (allowed,),
        )
        child_state[forbidden] = torch.tensor([4.0])
        with self.assertRaisesRegex(ValueError, "non-hazard tensor"):
            _validate_calibrated_state_dicts(
                parent_state,
                child_state,
                [allowed, forbidden],
            )


def _sha256(value: bytes) -> str:
    from hashlib import sha256

    return sha256(value).hexdigest()


if __name__ == "__main__":
    unittest.main()
