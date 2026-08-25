from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from irene_brain.evaluation.v21_cpu_qual_namespace import (
    CpuQualContractError,
    DatasetReservation,
    SequenceContentIdentity,
    authorize_cpu_qual_materialization,
    build_cpu_qual_content_manifest,
    build_preregistration,
    load_sealed_preregistration,
    publish_cpu_qual_content_manifest_create_only,
    publish_cpu_qual_opening_receipt_create_only,
    publish_preregistration_create_only,
)
from irene_brain.data.maze_chase_dataset import (
    MazeChaseDatasetConfig,
    maze_chase_dataset_manifest_sha256,
)
from irene_brain.data.moving_shapes_dataset import DatasetSplit


def _manifest(split: str, tag: int, offset: int, count: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "generator_id": "test.deterministic.maze.v1",
        "split": split,
        "seed_namespace_tag": tag,
        "seed_offset": offset,
        "sequence_count": count,
        "sequence_length": 16,
        "counterfactual_targets": "all_actions_v1",
    }


class V21CpuQualNamespaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source_root = self.directory / "source"
        self.source_root.mkdir()
        (self.source_root / "dataset.py").write_text("GENERATOR = 1\n", encoding="utf-8")
        (self.source_root / "evaluator.py").write_text("METRICS = 1\n", encoding="utf-8")
        self.reservations = (
            DatasetReservation.from_manifest(
                "train", _manifest("train", 0, 16_777_216, 4)
            ),
            DatasetReservation.from_manifest(
                "dev", _manifest("validation", 1, 25_165_824, 3)
            ),
            DatasetReservation.from_manifest(
                "cpu_qual", _manifest("validation", 1, 33_554_432, 2)
            ),
        )
        self.preregistration_path = self.directory / "pins" / "preregistration.json"
        self.content_relative_path = "content/cpu-qual-content.json"

    def _payload(self) -> dict[str, object]:
        return build_preregistration(
            reservations=self.reservations,
            source_root=self.source_root,
            source_paths=("evaluator.py", "dataset.py"),
            evaluation_protocol={
                "candidate_ids": [f"seed-{seed}" for seed in (44, 45, 46, 47, 48)],
                "model_seeds": [44, 45, 46, 47, 48],
                "required_passes": 5,
                "test_split_allowed": False,
            },
            content_manifest_relative_path=self.content_relative_path,
        )

    def _publish_and_authorize(self):
        payload = self._payload()
        publish_preregistration_create_only(self.preregistration_path, payload)
        return authorize_cpu_qual_materialization(
            self.preregistration_path,
            source_root=self.source_root,
        )

    def test_build_is_unmaterialized_and_authorization_requires_publication(self) -> None:
        payload = self._payload()
        self.assertEqual(payload["status"], "sealed_unmaterialized")
        self.assertEqual(payload["test_split_status"], "unopened_forbidden")
        self.assertFalse(self.preregistration_path.exists())
        self.assertFalse((self.directory / "pins/content").exists())
        with self.assertRaisesRegex(CpuQualContractError, "before preregistration"):
            authorize_cpu_qual_materialization(
                self.preregistration_path,
                source_root=self.source_root,
            )

    def test_reservation_uses_the_exact_maze_dataset_manifest_identity(self) -> None:
        config = MazeChaseDatasetConfig(
            split=DatasetSplit.VALIDATION,
            sequence_count=2,
            sequence_length=16,
            seed_offset=33_554_432,
            ghost_count=5,
            ghost_period=1,
            counterfactual_targets="all_actions_v1",
        )
        reservation = DatasetReservation.from_manifest("cpu_qual", config.manifest_dict())
        self.assertEqual(
            reservation.dataset_manifest_sha256,
            maze_chase_dataset_manifest_sha256(config),
        )

    def test_test_split_and_any_local_range_overlap_are_rejected(self) -> None:
        with self.assertRaisesRegex(CpuQualContractError, "TEST must remain unopened"):
            DatasetReservation.from_manifest(
                "cpu_qual", _manifest("test", 2, 40_000_000, 2)
            )
        overlapping = (
            self.reservations[0],
            DatasetReservation.from_manifest(
                "dev", _manifest("validation", 1, 16_777_218, 3)
            ),
            self.reservations[2],
        )
        with self.assertRaisesRegex(CpuQualContractError, "local seed ranges overlap"):
            build_preregistration(
                reservations=overlapping,
                source_root=self.source_root,
                source_paths=("dataset.py",),
                evaluation_protocol={
                    "candidate_ids": ["seed-44"],
                    "required_passes": 1,
                    "test_split_allowed": False,
                    "metric": "bce",
                },
                content_manifest_relative_path=self.content_relative_path,
            )

    def test_protocol_requires_all_preregistered_candidates_and_forbids_test(self) -> None:
        protocol = {
            "candidate_ids": ["seed-44", "seed-45"],
            "required_passes": 1,
            "test_split_allowed": False,
        }
        with self.assertRaisesRegex(CpuQualContractError, "entire candidate cohort"):
            build_preregistration(
                reservations=self.reservations,
                source_root=self.source_root,
                source_paths=("dataset.py",),
                evaluation_protocol=protocol,
                content_manifest_relative_path=self.content_relative_path,
            )
        protocol["required_passes"] = 2
        protocol["test_split_allowed"] = True
        with self.assertRaisesRegex(CpuQualContractError, "explicitly forbid TEST"):
            build_preregistration(
                reservations=self.reservations,
                source_root=self.source_root,
                source_paths=("dataset.py",),
                evaluation_protocol=protocol,
                content_manifest_relative_path=self.content_relative_path,
            )

    def test_preregistration_is_canonical_create_only_and_source_bound(self) -> None:
        payload = self._payload()
        first_sha = publish_preregistration_create_only(self.preregistration_path, payload)
        encoded = self.preregistration_path.read_bytes()
        self.assertEqual(first_sha, sha256(encoded).hexdigest())
        self.assertEqual(
            encoded,
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        )
        with self.assertRaises(FileExistsError):
            publish_preregistration_create_only(self.preregistration_path, payload)
        sealed = load_sealed_preregistration(
            self.preregistration_path,
            source_root=self.source_root,
        )
        self.assertEqual(sealed.artifact_sha256, first_sha)
        (self.source_root / "evaluator.py").write_text("METRICS = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(CpuQualContractError, "live source bundle differs"):
            authorize_cpu_qual_materialization(
                self.preregistration_path,
                source_root=self.source_root,
            )

    def test_noncanonical_or_tampered_preregistration_cannot_authorize(self) -> None:
        payload = self._payload()
        self.preregistration_path.parent.mkdir(parents=True)
        self.preregistration_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(CpuQualContractError, "canonical JSON line"):
            authorize_cpu_qual_materialization(
                self.preregistration_path,
                source_root=self.source_root,
            )

        self.preregistration_path.unlink()
        tampered = copy.deepcopy(payload)
        tampered["evaluation_protocol"]["required_passes"] = 4
        self.preregistration_path.write_bytes(
            (json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
        with self.assertRaisesRegex(CpuQualContractError, "does not match"):
            authorize_cpu_qual_materialization(
                self.preregistration_path,
                source_root=self.source_root,
            )

    def test_content_is_exact_ordered_create_only_and_precedes_evaluation(self) -> None:
        authorization = self._publish_and_authorize()
        identities = (
            SequenceContentIdentity(0, sha256(b"sequence-0").hexdigest()),
            SequenceContentIdentity(1, sha256(b"sequence-1").hexdigest()),
        )
        with self.assertRaisesRegex(CpuQualContractError, "frozen content manifest"):
            publish_cpu_qual_opening_receipt_create_only(
                self.directory / "pins/opened.json",
                preregistration_path=self.preregistration_path,
                source_root=self.source_root,
                candidate_checkpoint_sha256={"seed-44": sha256(b"model").hexdigest()},
            )
        with self.assertRaisesRegex(CpuQualContractError, "ordered contiguous"):
            build_cpu_qual_content_manifest(authorization, reversed(identities))
        content = build_cpu_qual_content_manifest(authorization, identities)
        content_path = self.preregistration_path.parent / self.content_relative_path
        publish_cpu_qual_content_manifest_create_only(
            content_path,
            content,
            authorization=authorization,
        )
        with self.assertRaises(FileExistsError):
            publish_cpu_qual_content_manifest_create_only(
                content_path,
                content,
                authorization=authorization,
            )

        candidates = {
            f"seed-{seed}": sha256(f"model-{seed}".encode()).hexdigest()
            for seed in (44, 45, 46, 47, 48)
        }
        with self.assertRaisesRegex(CpuQualContractError, "cohort differs"):
            publish_cpu_qual_opening_receipt_create_only(
                self.directory / "pins" / "partial-opened.json",
                preregistration_path=self.preregistration_path,
                source_root=self.source_root,
                candidate_checkpoint_sha256={"seed-44": candidates["seed-44"]},
            )
        opened_path = self.directory / "pins" / "opened.json"
        evaluation = publish_cpu_qual_opening_receipt_create_only(
            opened_path,
            preregistration_path=self.preregistration_path,
            source_root=self.source_root,
            candidate_checkpoint_sha256=candidates,
        )
        self.assertEqual(evaluation.opening_receipt_path, opened_path.resolve())
        self.assertTrue(opened_path.is_file())
        receipt = json.loads(opened_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "opened_once_candidates_frozen")
        self.assertEqual(receipt["test_split_status"], "unopened_forbidden")
        self.assertEqual(len(receipt["candidates"]), 5)
        with self.assertRaises(FileExistsError):
            publish_cpu_qual_opening_receipt_create_only(
                opened_path,
                preregistration_path=self.preregistration_path,
                source_root=self.source_root,
                candidate_checkpoint_sha256=candidates,
            )

    def test_content_manifest_wrong_count_and_wrong_publish_path_fail_closed(self) -> None:
        authorization = self._publish_and_authorize()
        with self.assertRaisesRegex(CpuQualContractError, "count differs"):
            build_cpu_qual_content_manifest(
                authorization,
                (SequenceContentIdentity(0, sha256(b"only-one").hexdigest()),),
            )
        identities = (
            SequenceContentIdentity(0, sha256(b"zero").hexdigest()),
            SequenceContentIdentity(1, sha256(b"one").hexdigest()),
        )
        content = build_cpu_qual_content_manifest(authorization, identities)
        with self.assertRaisesRegex(CpuQualContractError, "path differs"):
            publish_cpu_qual_content_manifest_create_only(
                self.directory / "wrong.json",
                content,
                authorization=authorization,
            )


if __name__ == "__main__":
    unittest.main()
