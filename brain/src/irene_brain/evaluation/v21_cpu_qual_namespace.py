"""Sealed TRAIN/DEV/CPU-QUAL namespace and provenance contract for V2.1.

This module deliberately does not import a model, Torch, or a dataset source.
It only reserves deterministic virtual-dataset ranges and controls the order in
which a future qualification runner may materialize and evaluate CPU-QUAL.

The lifecycle is fail closed::

    build preregistration (no rollouts)
      -> publish sealed preregistration, create-only
      -> authorize CPU-QUAL materialization
      -> publish exact ordered sequence identities, create-only
      -> bind the candidate cohort in an opening receipt, create-only
      -> evaluate

TEST is not a role in this contract and any TEST dataset manifest is rejected.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any


SCHEMA_VERSION = 1
QUALIFICATION_ID = "v21i-cpu-qual-world-model-v1"
CONTENT_IDENTITY_ALGORITHM = "ordered-sequence-content-sha256-v1"
SOURCE_BUNDLE_ALGORITHM = "path-and-file-sha256-v1"
DATASET_MANIFEST_ALGORITHM = "irene-maze-dataset-manifest-sha256-v1"
_ROLES = ("train", "dev", "cpu_qual")
_SPLITS = {"train", "validation", "test"}
_SHA256_LENGTH = 64
_AUTHORITY = object()


class CpuQualContractError(ValueError):
    """A CPU-QUAL artifact or requested lifecycle transition is invalid."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CpuQualContractError("value is not canonical-JSON encodable") from exc


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CpuQualContractError(f"{name} must be a lowercase SHA-256")
    return value


def _require_integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int = (1 << 62) - 1,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CpuQualContractError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise CpuQualContractError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _strict_object(encoded: bytes, *, name: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CpuQualContractError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        decoded = encoded.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                CpuQualContractError(f"{name} contains non-finite {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CpuQualContractError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CpuQualContractError(f"{name} root must be an object")
    expected = (_canonical_json(value) + "\n").encode("utf-8")
    if encoded != expected:
        raise CpuQualContractError(f"{name} must be one canonical JSON line")
    return value


def _artifact_body(payload: Mapping[str, object], digest_key: str) -> dict[str, object]:
    body = dict(payload)
    recorded = body.pop(digest_key, None)
    expected = _sha256_bytes(_canonical_json(body).encode("utf-8"))
    if _require_sha256(recorded, name=digest_key) != expected:
        raise CpuQualContractError(f"{digest_key} does not match the artifact body")
    return body


def _with_body_digest(body: Mapping[str, object], digest_key: str) -> dict[str, object]:
    payload = dict(body)
    if digest_key in payload:
        raise CpuQualContractError(f"body already contains {digest_key}")
    payload[digest_key] = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    return payload


def _safe_relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise CpuQualContractError(f"{name} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or any(part in {"", ".", ".."} for part in path.parts):
        raise CpuQualContractError(f"{name} must be a normalized POSIX relative path")
    return path.as_posix()


def _resolved_beneath(root: Path, relative: str, *, name: str) -> Path:
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise CpuQualContractError(f"{name} escapes the registered source root") from exc
    if not candidate.is_file():
        raise CpuQualContractError(f"{name} is not a regular file")
    return candidate


def _publish_create_only(path: Path, payload: Mapping[str, object]) -> str:
    encoded = (_canonical_json(payload) + "\n").encode("utf-8")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard-link publication is atomic and cannot replace an
        # artifact won by another process.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _sha256_bytes(encoded)


@dataclass(frozen=True, slots=True)
class DatasetReservation:
    """One unmaterialized deterministic dataset range."""

    role: str
    dataset_manifest_canonical_json: str
    dataset_manifest_sha256: str
    split: str
    seed_namespace_tag: int
    seed_offset: int
    sequence_count: int

    @classmethod
    def from_manifest(
        cls,
        role: str,
        manifest: Mapping[str, object],
    ) -> DatasetReservation:
        if role not in _ROLES:
            raise CpuQualContractError(f"role must be one of {_ROLES}")
        copied = json.loads(_canonical_json(manifest))
        if not isinstance(copied, dict):
            raise CpuQualContractError("dataset manifest root must be an object")
        split = copied.get("split")
        if split not in _SPLITS:
            raise CpuQualContractError("dataset manifest has an unknown split")
        if split == "test":
            raise CpuQualContractError("TEST must remain unopened; TEST reservations are forbidden")
        expected_split = "train" if role == "train" else "validation"
        if split != expected_split:
            raise CpuQualContractError(
                f"{role} must use the {expected_split} split namespace"
            )
        namespace_tag = _require_integer(
            copied.get("seed_namespace_tag"),
            name=f"{role}.seed_namespace_tag",
            maximum=2,
        )
        expected_tag = 0 if split == "train" else 1
        if namespace_tag != expected_tag:
            raise CpuQualContractError(f"{role} split and seed namespace tag disagree")
        offset = _require_integer(copied.get("seed_offset"), name=f"{role}.seed_offset")
        count = _require_integer(
            copied.get("sequence_count"),
            name=f"{role}.sequence_count",
            minimum=1,
            maximum=1 << 62,
        )
        if offset + count > 1 << 62:
            raise CpuQualContractError(f"{role} range exceeds its split namespace")
        canonical = _canonical_json(copied)
        return cls(
            role=role,
            dataset_manifest_canonical_json=canonical,
            # This is the exact identity used by
            # maze_chase_dataset.maze_chase_dataset_manifest_sha256, not a
            # second lookalike hash owned only by the qualification wrapper.
            dataset_manifest_sha256=_sha256_bytes(
                b"IRMCDATASET\x01" + canonical.encode("ascii")
            ),
            split=split,
            seed_namespace_tag=namespace_tag,
            seed_offset=offset,
            sequence_count=count,
        )

    @property
    def dataset_manifest(self) -> Mapping[str, object]:
        decoded = json.loads(self.dataset_manifest_canonical_json)
        if not isinstance(decoded, dict):  # pragma: no cover - constructor invariant
            raise AssertionError("dataset manifest is no longer an object")
        return MappingProxyType(decoded)

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "split": self.split,
            "seed_namespace_tag": self.seed_namespace_tag,
            "seed_offset": self.seed_offset,
            "sequence_count": self.sequence_count,
            "seed_end_exclusive": self.seed_offset + self.sequence_count,
            "dataset_manifest": dict(self.dataset_manifest),
            "dataset_manifest_identity_algorithm": DATASET_MANIFEST_ALGORITHM,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "materialized_at_preregistration": False,
        }


def _validate_reservations(
    reservations: Sequence[DatasetReservation],
) -> tuple[DatasetReservation, ...]:
    if not isinstance(reservations, Sequence) or isinstance(reservations, (str, bytes)):
        raise CpuQualContractError("reservations must be a sequence")
    frozen = tuple(reservations)
    if len(frozen) != len(_ROLES) or tuple(item.role for item in frozen) != _ROLES:
        raise CpuQualContractError(f"reservations must be ordered exactly as {_ROLES}")
    if any(not isinstance(item, DatasetReservation) for item in frozen):
        raise CpuQualContractError("every reservation must be a DatasetReservation")
    # Require distinct local ranges as well as distinct split-tagged seed IDs.
    # The stricter local rule prevents a future split-tag regression from
    # silently turning TRAIN or DEV into qualification examples.
    for index, left in enumerate(frozen):
        left_range = range(left.seed_offset, left.seed_offset + left.sequence_count)
        for right in frozen[index + 1 :]:
            if max(left_range.start, right.seed_offset) < min(
                left_range.stop, right.seed_offset + right.sequence_count
            ):
                raise CpuQualContractError(
                    f"{left.role} and {right.role} local seed ranges overlap"
                )
    return frozen


def build_source_bundle(
    source_root: Path,
    relative_paths: Sequence[str],
) -> dict[str, object]:
    """Hash an exact, sorted source-file set without importing its code."""

    if not relative_paths:
        raise CpuQualContractError("source bundle must contain at least one file")
    normalized = tuple(
        sorted(_safe_relative_path(value, name="source path") for value in relative_paths)
    )
    if len(set(normalized)) != len(normalized):
        raise CpuQualContractError("source bundle contains duplicate paths")
    digest = sha256(b"PBV21CPUQUALSOURCE\x01")
    files: list[dict[str, object]] = []
    for relative in normalized:
        path = _resolved_beneath(source_root, relative, name=relative)
        content = path.read_bytes()
        file_sha = _sha256_bytes(content)
        encoded_path = relative.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(4, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_sha))
        files.append(
            {"relative_path": relative, "size_bytes": len(content), "sha256": file_sha}
        )
    return {
        "schema_version": 1,
        "algorithm": SOURCE_BUNDLE_ALGORITHM,
        "files": files,
        "sha256": digest.hexdigest(),
    }


def build_preregistration(
    *,
    reservations: Sequence[DatasetReservation],
    source_root: Path,
    source_paths: Sequence[str],
    evaluation_protocol: Mapping[str, object],
    content_manifest_relative_path: str,
) -> dict[str, object]:
    """Build an unmaterialized preregistration payload.

    Merely building this in memory does not authorize CPU-QUAL generation.
    Only a byte-valid, create-only publication can do that.
    """

    frozen = _validate_reservations(reservations)
    protocol = json.loads(_canonical_json(evaluation_protocol))
    if not isinstance(protocol, dict) or not protocol:
        raise CpuQualContractError("evaluation_protocol must be a non-empty object")
    _validate_evaluation_protocol(protocol)
    content_path = _safe_relative_path(
        content_manifest_relative_path,
        name="content_manifest_relative_path",
    )
    source_bundle = build_source_bundle(source_root, source_paths)
    body: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "qualification_id": QUALIFICATION_ID,
        "status": "sealed_unmaterialized",
        "test_split_status": "unopened_forbidden",
        "namespace_policy": "distinct-local-and-split-tagged-ranges-v1",
        "reservations": [reservation.to_dict() for reservation in frozen],
        "source_bundle": source_bundle,
        "evaluation_protocol": protocol,
        "evaluation_protocol_sha256": _sha256_bytes(
            _canonical_json(protocol).encode("utf-8")
        ),
        "content_identity_contract": {
            "algorithm": CONTENT_IDENTITY_ALGORITHM,
            "manifest_relative_path": content_path,
            "publication": "create_only_after_preregistration_before_evaluation",
            "ordered_role": "cpu_qual",
        },
        "lifecycle": [
            "publish_preregistration_create_only",
            "materialize_cpu_qual_once",
            "publish_content_manifest_create_only",
            "publish_candidate_bound_opening_receipt_create_only",
            "evaluate_once",
        ],
    }
    return _with_body_digest(body, "preregistration_sha256")


def publish_preregistration_create_only(path: Path, payload: Mapping[str, object]) -> str:
    validate_preregistration(payload)
    return _publish_create_only(path, payload)


def _reservation_from_payload(value: object) -> DatasetReservation:
    if not isinstance(value, dict):
        raise CpuQualContractError("reservation must be an object")
    expected = {
        "role",
        "split",
        "seed_namespace_tag",
        "seed_offset",
        "sequence_count",
        "seed_end_exclusive",
        "dataset_manifest",
        "dataset_manifest_identity_algorithm",
        "dataset_manifest_sha256",
        "materialized_at_preregistration",
    }
    if set(value) != expected:
        raise CpuQualContractError("reservation keys differ from the fixed schema")
    role = value["role"]
    manifest = value["dataset_manifest"]
    if not isinstance(role, str) or not isinstance(manifest, dict):
        raise CpuQualContractError("reservation role or dataset manifest has the wrong type")
    reservation = DatasetReservation.from_manifest(role, manifest)
    if value["materialized_at_preregistration"] is not False:
        raise CpuQualContractError("preregistration must not claim materialized data")
    if value["dataset_manifest_identity_algorithm"] != DATASET_MANIFEST_ALGORITHM:
        raise CpuQualContractError("dataset manifest identity algorithm differs")
    expected_values = reservation.to_dict()
    if value != expected_values:
        raise CpuQualContractError("reservation redundant identities disagree")
    return reservation


def validate_preregistration(payload: Mapping[str, object]) -> tuple[DatasetReservation, ...]:
    body = _artifact_body(payload, "preregistration_sha256")
    expected_keys = {
        "schema_version",
        "qualification_id",
        "status",
        "test_split_status",
        "namespace_policy",
        "reservations",
        "source_bundle",
        "evaluation_protocol",
        "evaluation_protocol_sha256",
        "content_identity_contract",
        "lifecycle",
    }
    if set(body) != expected_keys:
        raise CpuQualContractError("preregistration keys differ from the fixed schema")
    if body["schema_version"] != SCHEMA_VERSION or isinstance(body["schema_version"], bool):
        raise CpuQualContractError("preregistration schema_version is not supported")
    if body["qualification_id"] != QUALIFICATION_ID:
        raise CpuQualContractError("qualification_id is not fixed")
    if body["status"] != "sealed_unmaterialized":
        raise CpuQualContractError("preregistration status is not sealed_unmaterialized")
    if body["test_split_status"] != "unopened_forbidden":
        raise CpuQualContractError("TEST must remain explicitly unopened")
    if body["namespace_policy"] != "distinct-local-and-split-tagged-ranges-v1":
        raise CpuQualContractError("namespace policy differs")
    raw_reservations = body["reservations"]
    if not isinstance(raw_reservations, list):
        raise CpuQualContractError("reservations must be an array")
    reservations = _validate_reservations(
        tuple(_reservation_from_payload(value) for value in raw_reservations)
    )
    source = body["source_bundle"]
    if not isinstance(source, dict):
        raise CpuQualContractError("source_bundle must be an object")
    _validate_source_bundle_shape(source)
    protocol = body["evaluation_protocol"]
    if not isinstance(protocol, dict) or not protocol:
        raise CpuQualContractError("evaluation_protocol must be a non-empty object")
    _validate_evaluation_protocol(protocol)
    if _sha256_bytes(_canonical_json(protocol).encode("utf-8")) != _require_sha256(
        body["evaluation_protocol_sha256"], name="evaluation_protocol_sha256"
    ):
        raise CpuQualContractError("evaluation protocol digest disagrees")
    identity_contract = body["content_identity_contract"]
    if not isinstance(identity_contract, dict) or set(identity_contract) != {
        "algorithm",
        "manifest_relative_path",
        "publication",
        "ordered_role",
    }:
        raise CpuQualContractError("content identity contract differs from the fixed schema")
    if (
        identity_contract["algorithm"] != CONTENT_IDENTITY_ALGORITHM
        or identity_contract["publication"]
        != "create_only_after_preregistration_before_evaluation"
        or identity_contract["ordered_role"] != "cpu_qual"
    ):
        raise CpuQualContractError("content identity contract differs")
    _safe_relative_path(
        identity_contract["manifest_relative_path"],
        name="content manifest relative path",
    )
    expected_lifecycle = [
        "publish_preregistration_create_only",
        "materialize_cpu_qual_once",
        "publish_content_manifest_create_only",
        "publish_candidate_bound_opening_receipt_create_only",
        "evaluate_once",
    ]
    if body["lifecycle"] != expected_lifecycle:
        raise CpuQualContractError("qualification lifecycle differs")
    return reservations


def _validate_evaluation_protocol(protocol: Mapping[str, object]) -> None:
    """Validate the cohort fields needed before the seal may be opened."""

    candidate_ids = protocol.get("candidate_ids")
    if not isinstance(candidate_ids, list) or not candidate_ids:
        raise CpuQualContractError(
            "evaluation_protocol.candidate_ids must be a non-empty array"
        )
    if any(not isinstance(value, str) or not value for value in candidate_ids):
        raise CpuQualContractError("every candidate id must be a non-empty string")
    if candidate_ids != sorted(set(candidate_ids)):
        raise CpuQualContractError("candidate_ids must be unique and sorted")
    required = _require_integer(
        protocol.get("required_passes"),
        name="evaluation_protocol.required_passes",
        minimum=1,
        maximum=len(candidate_ids),
    )
    if required != len(candidate_ids):
        raise CpuQualContractError("required_passes must require the entire candidate cohort")
    if protocol.get("test_split_allowed") is not False:
        raise CpuQualContractError("evaluation protocol must explicitly forbid TEST")


def _validate_source_bundle_shape(bundle: Mapping[str, object]) -> None:
    if set(bundle) != {"schema_version", "algorithm", "files", "sha256"}:
        raise CpuQualContractError("source bundle keys differ from the fixed schema")
    if bundle["schema_version"] != 1 or isinstance(bundle["schema_version"], bool):
        raise CpuQualContractError("source bundle schema_version differs")
    if bundle["algorithm"] != SOURCE_BUNDLE_ALGORITHM:
        raise CpuQualContractError("source bundle algorithm differs")
    files = bundle["files"]
    if not isinstance(files, list) or not files:
        raise CpuQualContractError("source bundle files must be a non-empty array")
    previous = ""
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise CpuQualContractError("source file record differs from the fixed schema")
        relative = _safe_relative_path(entry["relative_path"], name="source relative path")
        if relative <= previous:
            raise CpuQualContractError("source bundle paths must be unique and sorted")
        previous = relative
        _require_integer(entry["size_bytes"], name="source size", maximum=1 << 62)
        _require_sha256(entry["sha256"], name="source file sha256")
    _require_sha256(bundle["sha256"], name="source bundle sha256")


def _verify_live_source_bundle(payload: Mapping[str, object], source_root: Path) -> None:
    bundle = payload.get("source_bundle")
    if not isinstance(bundle, dict):
        raise CpuQualContractError("preregistration lacks a source bundle")
    _validate_source_bundle_shape(bundle)
    files = bundle["files"]
    if not isinstance(files, list):  # pragma: no cover - validated above
        raise AssertionError("source files are no longer an array")
    paths = [entry["relative_path"] for entry in files if isinstance(entry, dict)]
    if any(not isinstance(path, str) for path in paths):
        raise CpuQualContractError("source path has the wrong type")
    live = build_source_bundle(source_root, paths)
    if live != bundle:
        raise CpuQualContractError("live source bundle differs from preregistration")


@dataclass(frozen=True, slots=True)
class SealedPreregistration:
    """A canonical on-disk preregistration revalidated against live source."""

    path: Path
    canonical_bytes: bytes
    artifact_sha256: str

    @property
    def payload(self) -> Mapping[str, object]:
        return MappingProxyType(_strict_object(self.canonical_bytes, name=str(self.path)))


def load_sealed_preregistration(path: Path, *, source_root: Path) -> SealedPreregistration:
    if not path.is_file():
        raise CpuQualContractError(
            "CPU-QUAL cannot be materialized before preregistration is published"
        )
    encoded = path.read_bytes()
    payload = _strict_object(encoded, name=str(path))
    validate_preregistration(payload)
    _verify_live_source_bundle(payload, source_root)
    return SealedPreregistration(
        path=path.resolve(),
        canonical_bytes=encoded,
        artifact_sha256=_sha256_bytes(encoded),
    )


@dataclass(frozen=True, slots=True)
class CpuQualMaterializationAuthorization:
    preregistration_path: Path
    preregistration_file_sha256: str
    preregistration_body_sha256: str
    dataset_manifest_sha256: str
    sequence_count: int
    content_manifest_relative_path: str
    _authority: object

    def __post_init__(self) -> None:
        if self._authority is not _AUTHORITY:
            raise CpuQualContractError(
                "materialization authorization must come from a sealed preregistration"
            )


def authorize_cpu_qual_materialization(
    preregistration_path: Path,
    *,
    source_root: Path,
) -> CpuQualMaterializationAuthorization:
    sealed = load_sealed_preregistration(preregistration_path, source_root=source_root)
    payload = sealed.payload
    reservations = validate_preregistration(payload)
    cpu_qual = reservations[2]
    identity_contract = payload["content_identity_contract"]
    if not isinstance(identity_contract, dict):  # pragma: no cover - validated above
        raise AssertionError("content identity contract is no longer an object")
    body_sha = _require_sha256(
        payload["preregistration_sha256"], name="preregistration_sha256"
    )
    return CpuQualMaterializationAuthorization(
        preregistration_path=sealed.path,
        preregistration_file_sha256=sealed.artifact_sha256,
        preregistration_body_sha256=body_sha,
        dataset_manifest_sha256=cpu_qual.dataset_manifest_sha256,
        sequence_count=cpu_qual.sequence_count,
        content_manifest_relative_path=str(identity_contract["manifest_relative_path"]),
        _authority=_AUTHORITY,
    )


@dataclass(frozen=True, slots=True)
class SequenceContentIdentity:
    sequence_index: int
    content_sha256: str

    def __post_init__(self) -> None:
        _require_integer(self.sequence_index, name="sequence_index", maximum=(1 << 62) - 1)
        _require_sha256(self.content_sha256, name="sequence content_sha256")

    def to_dict(self) -> dict[str, object]:
        return {"sequence_index": self.sequence_index, "content_sha256": self.content_sha256}


def build_cpu_qual_content_manifest(
    authorization: CpuQualMaterializationAuthorization,
    identities: Iterable[SequenceContentIdentity],
) -> dict[str, object]:
    """Freeze exact ordered content after authorized deterministic generation."""

    if not isinstance(authorization, CpuQualMaterializationAuthorization):
        raise CpuQualContractError("a sealed materialization authorization is required")
    authorization.__post_init__()
    frozen = tuple(identities)
    if len(frozen) != authorization.sequence_count:
        raise CpuQualContractError("content identity count differs from CPU-QUAL reservation")
    if any(not isinstance(item, SequenceContentIdentity) for item in frozen):
        raise CpuQualContractError("every content identity must be a SequenceContentIdentity")
    if tuple(item.sequence_index for item in frozen) != tuple(range(len(frozen))):
        raise CpuQualContractError("content identities must be ordered contiguous sequence indices")
    digest = sha256(b"PBV21CPUQUALCONTENT\x01")
    for item in frozen:
        digest.update(item.sequence_index.to_bytes(8, "big"))
        digest.update(bytes.fromhex(item.content_sha256))
    body: dict[str, object] = {
        "schema_version": 1,
        "qualification_id": QUALIFICATION_ID,
        "status": "content_frozen_unopened",
        "preregistration_file_sha256": authorization.preregistration_file_sha256,
        "preregistration_sha256": authorization.preregistration_body_sha256,
        "dataset_manifest_sha256": authorization.dataset_manifest_sha256,
        "content_identity_algorithm": CONTENT_IDENTITY_ALGORITHM,
        "sequence_count": len(frozen),
        "ordered_content_sha256": digest.hexdigest(),
        "sequences": [item.to_dict() for item in frozen],
    }
    return _with_body_digest(body, "content_manifest_sha256")


def publish_cpu_qual_content_manifest_create_only(
    path: Path,
    payload: Mapping[str, object],
    *,
    authorization: CpuQualMaterializationAuthorization,
) -> str:
    validate_cpu_qual_content_manifest(payload, authorization=authorization)
    expected = (
        authorization.preregistration_path.parent
        / Path(*PurePosixPath(authorization.content_manifest_relative_path).parts)
    ).resolve()
    if path.resolve() != expected:
        raise CpuQualContractError("content manifest path differs from preregistration")
    return _publish_create_only(path, payload)


def validate_cpu_qual_content_manifest(
    payload: Mapping[str, object],
    *,
    authorization: CpuQualMaterializationAuthorization,
) -> None:
    authorization.__post_init__()
    body = _artifact_body(payload, "content_manifest_sha256")
    expected_keys = {
        "schema_version",
        "qualification_id",
        "status",
        "preregistration_file_sha256",
        "preregistration_sha256",
        "dataset_manifest_sha256",
        "content_identity_algorithm",
        "sequence_count",
        "ordered_content_sha256",
        "sequences",
    }
    if set(body) != expected_keys:
        raise CpuQualContractError("content manifest keys differ from the fixed schema")
    fixed = {
        "schema_version": 1,
        "qualification_id": QUALIFICATION_ID,
        "status": "content_frozen_unopened",
        "preregistration_file_sha256": authorization.preregistration_file_sha256,
        "preregistration_sha256": authorization.preregistration_body_sha256,
        "dataset_manifest_sha256": authorization.dataset_manifest_sha256,
        "content_identity_algorithm": CONTENT_IDENTITY_ALGORITHM,
        "sequence_count": authorization.sequence_count,
    }
    if any(body[key] != value for key, value in fixed.items()):
        raise CpuQualContractError("content manifest is not bound to its authorization")
    sequences = body["sequences"]
    if not isinstance(sequences, list):
        raise CpuQualContractError("content manifest sequences must be an array")
    identities: list[SequenceContentIdentity] = []
    for sequence in sequences:
        if not isinstance(sequence, dict) or set(sequence) != {
            "sequence_index",
            "content_sha256",
        }:
            raise CpuQualContractError("sequence content record differs from the fixed schema")
        identities.append(
            SequenceContentIdentity(
                sequence_index=sequence["sequence_index"],
                content_sha256=sequence["content_sha256"],
            )
        )
    rebuilt = build_cpu_qual_content_manifest(authorization, identities)
    if dict(payload) != rebuilt:
        raise CpuQualContractError("content manifest ordered digest disagrees")


@dataclass(frozen=True, slots=True)
class CpuQualEvaluationAuthorization:
    opening_receipt_path: Path
    opening_receipt_sha256: str
    preregistration_sha256: str
    content_manifest_sha256: str
    candidate_cohort_sha256: str
    _authority: object

    def __post_init__(self) -> None:
        if self._authority is not _AUTHORITY:
            raise CpuQualContractError(
                "evaluation authorization must come from a create-only opening receipt"
            )


def publish_cpu_qual_opening_receipt_create_only(
    opening_receipt_path: Path,
    *,
    preregistration_path: Path,
    source_root: Path,
    candidate_checkpoint_sha256: Mapping[str, str],
) -> CpuQualEvaluationAuthorization:
    """Bind the frozen candidates before any CPU-QUAL model evaluation."""

    materialization = authorize_cpu_qual_materialization(
        preregistration_path,
        source_root=source_root,
    )
    content_path = (
        materialization.preregistration_path.parent
        / Path(*PurePosixPath(materialization.content_manifest_relative_path).parts)
    ).resolve()
    if not content_path.is_file():
        raise CpuQualContractError("CPU-QUAL evaluation requires a frozen content manifest")
    content_encoded = content_path.read_bytes()
    content = _strict_object(content_encoded, name=str(content_path))
    validate_cpu_qual_content_manifest(content, authorization=materialization)
    if not candidate_checkpoint_sha256:
        raise CpuQualContractError("candidate cohort must not be empty")
    sealed = load_sealed_preregistration(
        preregistration_path,
        source_root=source_root,
    )
    protocol = sealed.payload["evaluation_protocol"]
    if not isinstance(protocol, dict):  # pragma: no cover - validated by loader
        raise AssertionError("evaluation protocol is no longer an object")
    registered_ids = protocol["candidate_ids"]
    if not isinstance(registered_ids, list):  # pragma: no cover - validated by loader
        raise AssertionError("candidate_ids are no longer an array")
    if sorted(candidate_checkpoint_sha256) != registered_ids:
        raise CpuQualContractError("candidate cohort differs from preregistration")
    candidates: list[dict[str, str]] = []
    for candidate_id in sorted(candidate_checkpoint_sha256):
        if not isinstance(candidate_id, str) or not candidate_id:
            raise CpuQualContractError("candidate id must be a non-empty string")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "checkpoint_sha256": _require_sha256(
                    candidate_checkpoint_sha256[candidate_id],
                    name=f"checkpoint {candidate_id}",
                ),
            }
        )
    cohort_sha = _sha256_bytes(_canonical_json(candidates).encode("utf-8"))
    body: dict[str, object] = {
        "schema_version": 1,
        "qualification_id": QUALIFICATION_ID,
        "status": "opened_once_candidates_frozen",
        "preregistration_file_sha256": materialization.preregistration_file_sha256,
        "preregistration_sha256": materialization.preregistration_body_sha256,
        "content_manifest_file_sha256": _sha256_bytes(content_encoded),
        "content_manifest_sha256": _require_sha256(
            content["content_manifest_sha256"], name="content_manifest_sha256"
        ),
        "candidate_cohort_sha256": cohort_sha,
        "candidates": candidates,
        "test_split_status": "unopened_forbidden",
    }
    receipt = _with_body_digest(body, "opening_receipt_sha256")
    file_sha = _publish_create_only(opening_receipt_path, receipt)
    return CpuQualEvaluationAuthorization(
        opening_receipt_path=opening_receipt_path.resolve(),
        opening_receipt_sha256=file_sha,
        preregistration_sha256=materialization.preregistration_body_sha256,
        content_manifest_sha256=str(content["content_manifest_sha256"]),
        candidate_cohort_sha256=cohort_sha,
        _authority=_AUTHORITY,
    )
