"""Trusted, once-only Torch runner for RCQ-v2 final qualification.

The module CLI is the sole supported production scoring entrypoint.  It accepts
no caller-selected identity roots: fixed container mounts plus an immutable
pretraining pin and a separately reviewed final-authorization pin determine the
release, registration, run, checkpoint, and readiness evidence.  All non-TEST
trust checks complete before it publishes a claim into the canonical host range
registry.  The fresh recipient and donor TEST datasets are constructed only by
the concrete model materializer after that claim exists.

"Once-only" is scoped to the canonical DGX host workspace and its persistent
range-claim registry.  Copying or resetting that host registry is outside this
Python process's authority and must be prevented by the dedicated dispatcher
and the externally archived claim/receipt pins.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import importlib
import json
from math import ceil, fsum, isfinite
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Callable, Iterator, Mapping, Sequence

from ..data import (
    DatasetSplit,
    MovingShapesDatasetConfig,
    MovingShapesSequence,
    MovingShapesSequenceDataset,
    dataset_manifest_sha256,
)
from ..runtime.policy import Capability, ResourcePolicy
from ..training.batches import CONTROL_LAYOUT_ID, TrajectoryBatch
from ..training.config import DatasetConfig, TrainingConfig, load_training_config
from .rcq_v2 import (
    DEVELOPMENT_GATE_ID,
    DEVELOPMENT_SAMPLES,
    RCQDevelopmentReport,
    RCQInputError,
    RCQValueDevelopmentReport,
    VALUE_ABSOLUTE_MAX_MSE,
    VALUE_DEVELOPMENT_GATE_ID,
    VALUE_DEVELOPMENT_TARGET_VARIANCE,
    VALUE_IMPROVEMENT_RATIO,
    VALUE_TRAIN_CONDITIONAL_BASELINE_MSE,
    evaluate_rcq_v2_development,
    evaluate_rcq_v2_value_development,
)
from .rcq_v2_final import (
    DONOR_OFFSET,
    DONOR_SEQUENCES,
    EVALUATOR_BUNDLE_FILES,
    FINAL_DECISIONS,
    FINAL_STAGE_INDEX,
    FINAL_STEP,
    RECIPIENT_OFFSET,
    RECIPIENT_SEQUENCES,
    FUTURE_CAMPAIGN_OFFSET,
    DonorMapping,
    RCQDecisionEvidence,
    RCQFinalEvidence,
    RCQFinalReceipt,
    RCQRegistration,
    TrainLookupReceipt,
    _FinalAuthorization,
    _authorization_payload,
    _canonical,
    _deep_freeze,
    _digest_bytes,
    _digest_payload,
    _evaluate_final_evidence,
    _final_range_claim_id,
    _hash_string,
    _publish_no_replace,
    _raw_jsonl,
    _reserved_excluded_ranges,
    _safe_receipt_root,
    _strict_equal,
    _validate_complete_donor_mapping,
    _validate_evidence,
    build_target_blind_donor_mapping,
    fit_rcq_v2_train_lookup,
    load_rcq_v2_registration,
)


_EXPECTED_ENTRY_CHECKPOINT_NAME = "step-00001536.pt"
_EXPECTED_CHECKPOINT_NAME = "step-00002048.pt"
_EXPECTED_ENTRY_GATE_NAME = "development-gate-step-00001536.json"
_EXPECTED_COMPLETION_GATE_NAME = "final-development-step-00002048.json"
_EXPECTED_INVARIANCE_NAME = "invariance-step-00002048.json"
_CLAIM_NAME = "rcq-v2-final.claim.json"
_RECEIPT_NAME = "rcq-v2-final.receipt.json"
_DECISION_LEDGER_NAME = "rcq-v2-final.decisions.jsonl"
_DONOR_LEDGER_NAME = "rcq-v2-final.donors.jsonl"
_RANGE_LEDGER_NAME = "rcq-v2-final.ranges.json"
_REGISTERED_CONFIG_RELATIVE_PATH = Path(
    "brain/configs/training/dgx-rcq-v2-reference.toml"
)
_REGISTRATION_RELATIVE_PATH = Path(
    "registrations/rcq-v2-reference-v2.json"
)
_CONTAINER_RELEASE_ROOT = Path("/workspace/repo")
_CONTAINER_RUN_ROOT = Path("/workspace/run")
_CONTAINER_CLAIM_REGISTRY_ROOT = Path("/workspace/final-claims")
_CONTAINER_PIN_ROOT = Path("/workspace/pins")
_PRETRAINING_PIN_PATH = _CONTAINER_PIN_ROOT / "pretraining.json"
_FINAL_AUTHORIZATION_PATH = _CONTAINER_PIN_ROOT / "final-authorization.json"
_PRETRAINING_PIN_DOMAIN = b"PSEUDOBRAINRCQPRETRAINPIN\x01"
_FINAL_AUTHORIZATION_DOMAIN = b"PSEUDOBRAINRCQFINALAUTH\x01"


def _plain_resolve(
    value: str | os.PathLike[str],
    *,
    name: str,
    kind: str,
) -> Path:
    absolute = Path(value).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.exists():
            metadata = current.stat(follow_symlinks=False)
            reparse = getattr(metadata, "st_file_attributes", 0) & getattr(
                stat,
                "FILE_ATTRIBUTE_REPARSE_POINT",
                0,
            )
            if current.is_symlink() or reparse:
                raise RCQInputError(f"{name} cannot traverse a link or reparse point")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise RCQInputError(f"cannot resolve {name}: {error}") from error
    if kind == "file" and not resolved.is_file():
        raise RCQInputError(f"{name} must be a regular file")
    if kind == "directory" and not resolved.is_dir():
        raise RCQInputError(f"{name} must be a directory")
    return resolved


def _file_sha256(path: Path) -> str:
    path = _plain_resolve(path, name=str(path), kind="file")
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_published_file(path: Path, content: bytes, *, name: str) -> str:
    resolved = _plain_resolve(path, name=name, kind="file")
    expected = _digest_bytes(content)
    if resolved.read_bytes() != content or _file_sha256(resolved) != expected:
        raise RCQInputError(f"{name} differs from its published bytes")
    if _file_sha256(resolved) != expected:
        raise RCQInputError(f"{name} changed during durable read-back")
    return expected


def _stat_identity(value: os.stat_result) -> tuple[int, int, int]:
    return (value.st_dev, value.st_ino, value.st_mode)


def _open_plain_directory_chain(path: Path) -> int:
    """Open every absolute path component with O_NOFOLLOW and return the leaf fd."""

    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise RCQInputError("trusted directory binding requires Linux O_NOFOLLOW")
    absolute = path.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _plain_directory_identity(path: Path) -> tuple[int, int, int]:
    descriptor = _open_plain_directory_chain(path)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise RCQInputError("trusted directory path is not a directory")
        return _stat_identity(metadata)
    finally:
        os.close(descriptor)


def _require_private_directory(metadata: os.stat_result, *, name: str) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RCQInputError(f"{name} must be an owner-only 0700 directory")


def _open_locked_claim_registry(path: Path) -> tuple[int, tuple[int, int, int]]:
    """Open and exclusively lock the canonical Linux registry without following links."""

    if os.name != "posix":
        raise RCQInputError("trusted final claim locking requires the registered Linux runtime")
    import fcntl

    try:
        descriptor = _open_plain_directory_chain(path)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as error:
        if "descriptor" in locals():
            os.close(descriptor)
        raise RCQInputError(
            f"cannot exclusively lock canonical final-claim registry: {error}"
        ) from error
    metadata = os.fstat(descriptor)
    try:
        _require_private_directory(
            metadata,
            name="canonical final-claim registry",
        )
    except RCQInputError:
        os.close(descriptor)
        raise
    identity = _stat_identity(metadata)
    if _plain_directory_identity(path) != identity:
        os.close(descriptor)
        raise RCQInputError("canonical final-claim registry pathname changed")
    return descriptor, identity


def _require_empty_claim_slot(registry: Path, leaf: str) -> None:
    """Check the fixed claim slot under the same lock used by final publication."""

    descriptor, registry_identity = _open_locked_claim_registry(registry)
    try:
        if _plain_directory_identity(registry) != registry_identity:
            raise RCQInputError("canonical final-claim registry pathname changed")
        try:
            metadata = os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return
        _require_private_directory(metadata, name="canonical final receipt slot")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
            os,
            "O_NOFOLLOW",
            0,
        )
        child = os.open(leaf, flags, dir_fd=descriptor)
        try:
            if _stat_identity(os.fstat(child)) != _stat_identity(metadata):
                raise RCQInputError("canonical final receipt slot changed while opening")
            if os.listdir(child):
                raise RCQInputError(
                    "canonical final receipt directory is not exclusively empty"
                )
        finally:
            os.close(child)
    finally:
        os.close(descriptor)


class _ReceiptSlotAbsent(RCQInputError):
    """The canonical range slot has never been created."""


class _BoundReceiptDirectory:
    """Locked dirfd capability for every authoritative post-claim artifact."""

    __slots__ = (
        "_closed",
        "_leaf",
        "_registry",
        "_registry_descriptor",
        "_registry_identity",
        "_root_descriptor",
        "_root_identity",
        "path",
    )

    def __init__(
        self,
        registry: Path,
        leaf: str,
        *,
        require_empty: bool = True,
    ) -> None:
        if (
            type(leaf) is not str
            or len(leaf) != 64
            or any(character not in "0123456789abcdef" for character in leaf)
        ):
            raise RCQInputError("canonical final receipt leaf is invalid")
        if type(require_empty) is not bool:
            raise RCQInputError("receipt-directory mode must be a boolean")
        descriptor, registry_identity = _open_locked_claim_registry(registry)
        self._closed = False
        self._leaf = leaf
        self._registry = registry
        self._registry_descriptor = descriptor
        self._registry_identity = registry_identity
        self._root_descriptor = -1
        self._root_identity = (0, 0, 0)
        self.path = registry / leaf
        try:
            created = False
            if require_empty:
                try:
                    os.mkdir(leaf, mode=0o700, dir_fd=descriptor)
                    created = True
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
                os,
                "O_NOFOLLOW",
                0,
            )
            try:
                root_descriptor = os.open(leaf, flags, dir_fd=descriptor)
            except FileNotFoundError as error:
                if not require_empty:
                    raise _ReceiptSlotAbsent(
                        "canonical final receipt slot has not been created"
                    ) from error
                raise
            if created:
                os.fchmod(root_descriptor, 0o700)
                os.fsync(root_descriptor)
            root_metadata = os.fstat(root_descriptor)
            try:
                _require_private_directory(
                    root_metadata,
                    name="canonical final receipt root",
                )
            except RCQInputError:
                os.close(root_descriptor)
                raise
            self._root_descriptor = root_descriptor
            self._root_identity = _stat_identity(root_metadata)
            self.assert_bound()
            if require_empty:
                self.require_entries(set())
        except BaseException as error:
            self.close()
            if isinstance(error, RCQInputError):
                raise
            raise RCQInputError(
                f"cannot bind canonical final receipt directory: {error}"
            ) from error

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        if getattr(self, "_root_descriptor", -1) >= 0:
            os.close(self._root_descriptor)
            self._root_descriptor = -1
        if getattr(self, "_registry_descriptor", -1) >= 0:
            os.close(self._registry_descriptor)
            self._registry_descriptor = -1

    def __del__(self) -> None:
        self.close()

    def assert_bound(self) -> None:
        if self._closed:
            raise RCQInputError("canonical final receipt capability is closed")
        registry_metadata = os.fstat(self._registry_descriptor)
        _require_private_directory(
            registry_metadata,
            name="canonical final-claim registry",
        )
        if (
            _stat_identity(registry_metadata) != self._registry_identity
            or _plain_directory_identity(self._registry) != self._registry_identity
        ):
            raise RCQInputError("canonical final-claim registry binding changed")
        leaf_metadata = os.stat(
            self._leaf,
            dir_fd=self._registry_descriptor,
            follow_symlinks=False,
        )
        if (
            _stat_identity(leaf_metadata) != self._root_identity
            or _stat_identity(os.fstat(self._root_descriptor)) != self._root_identity
        ):
            raise RCQInputError("canonical final receipt directory binding changed")
        _require_private_directory(
            leaf_metadata,
            name="canonical final receipt root",
        )
        _require_private_directory(
            os.fstat(self._root_descriptor),
            name="opened canonical final receipt root",
        )

    def require_entries(self, expected: set[str]) -> None:
        self.assert_bound()
        observed = set(os.listdir(self._root_descriptor))
        if observed != expected:
            raise RCQInputError("canonical final receipt directory entries changed")
        self.assert_bound()

    @staticmethod
    def _name(value: str) -> str:
        if (
            type(value) is not str
            or not value
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
        ):
            raise RCQInputError("authoritative receipt filename is invalid")
        return value

    def publish(
        self,
        name: str,
        content: bytes,
        *,
        existing: set[str],
        policy: ResourcePolicy,
    ) -> str:
        name = self._name(name)
        if type(content) is not bytes or not content:
            raise RCQInputError("authoritative receipt content must be nonempty bytes")
        if not isinstance(policy, ResourcePolicy):
            raise RCQInputError("authoritative receipt publication requires a resource policy")
        policy.require(Capability.ARTIFACT_WRITE)
        self.require_entries(existing)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(
            os,
            "O_NOFOLLOW",
            0,
        )
        try:
            descriptor = os.open(
                name,
                flags,
                0o400,
                dir_fd=self._root_descriptor,
            )
        except OSError as error:
            raise RCQInputError(
                f"cannot exclusively publish authoritative receipt file {name!r}: {error}"
            ) from error
        try:
            os.fchmod(descriptor, 0o400)
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        os.fsync(self._root_descriptor)
        self.require_entries(existing | {name})
        return self.require_exact(name, content)

    def read_bytes(self, name: str) -> bytes:
        name = self._name(name)
        self.assert_bound()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, dir_fd=self._root_descriptor)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_uid != os.geteuid()
                or stat.S_IMODE(before.st_mode) != 0o400
            ):
                raise RCQInputError("authoritative receipt artifact is not a private file")
            chunks: list[bytes] = []
            while block := os.read(descriptor, 1024 * 1024):
                chunks.append(block)
            after = os.fstat(descriptor)
            named = os.stat(
                name,
                dir_fd=self._root_descriptor,
                follow_symlinks=False,
            )
            if (
                _stat_identity(before) != _stat_identity(after)
                or _stat_identity(after) != _stat_identity(named)
                or after.st_nlink != 1
                or named.st_nlink != 1
                or after.st_uid != os.geteuid()
                or named.st_uid != os.geteuid()
                or stat.S_IMODE(after.st_mode) != 0o400
                or stat.S_IMODE(named.st_mode) != 0o400
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or before.st_ctime_ns != after.st_ctime_ns
            ):
                raise RCQInputError("authoritative receipt artifact changed while reading")
        finally:
            os.close(descriptor)
        self.assert_bound()
        return b"".join(chunks)

    def require_exact(self, name: str, content: bytes) -> str:
        expected = _digest_bytes(content)
        if self.read_bytes(name) != content or _digest_bytes(self.read_bytes(name)) != expected:
            raise RCQInputError("authoritative receipt artifact differs from published bytes")
        return expected


def _source_tree_sha256_exact(source_root: str | os.PathLike[str]) -> str:
    """Hash only Python source after rejecting every executable side channel."""

    root = _plain_resolve(source_root, name="source_root", kind="directory")
    if root.name != "irene_brain":
        raise RCQInputError("source_root must be the exact irene_brain package directory")
    try:
        parent_entries = tuple(os.scandir(root.parent))
    except OSError as error:
        raise RCQInputError(f"cannot enumerate source parent directory: {error}") from error
    if len(parent_entries) != 1 or parent_entries[0].name != "irene_brain":
        raise RCQInputError(
            "source parent must contain only the registered irene_brain package"
        )
    package_metadata = Path(parent_entries[0].path).stat(follow_symlinks=False)
    package_reparse = getattr(package_metadata, "st_file_attributes", 0) & getattr(
        stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        0,
    )
    if (
        parent_entries[0].is_symlink()
        or package_reparse
        or not stat.S_ISDIR(package_metadata.st_mode)
        or _stat_identity(package_metadata)
        != _stat_identity(root.stat(follow_symlinks=False))
    ):
        raise RCQInputError("registered irene_brain package root changed")
    candidates: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as error:
            raise RCQInputError(f"cannot enumerate source tree: {error}") from error
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(root)
            try:
                metadata = path.stat(follow_symlinks=False)
            except OSError as error:
                raise RCQInputError(
                    f"cannot inspect source tree entry {relative.as_posix()!r}: {error}"
                ) from error
            reparse = getattr(metadata, "st_file_attributes", 0) & getattr(
                stat,
                "FILE_ATTRIBUTE_REPARSE_POINT",
                0,
            )
            if entry.is_symlink() or reparse:
                raise RCQInputError(
                    f"source tree entry {relative.as_posix()!r} is a link or reparse point"
                )
            if stat.S_ISDIR(metadata.st_mode):
                if entry.name == "__pycache__":
                    raise RCQInputError("source tree cannot contain __pycache__")
                pending.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode) or path.suffix != ".py":
                raise RCQInputError(
                    f"source tree entry {relative.as_posix()!r} is not Python source"
                )
            if metadata.st_nlink != 1:
                raise RCQInputError(
                    f"source tree file {relative.as_posix()!r} has an external hardlink"
                )
            candidates.append(path)
    candidates.sort(key=lambda item: item.relative_to(root).as_posix())
    if not candidates:
        raise RCQInputError("source tree contains no Python files")
    digest = sha256(b"IRENESOURCE\x01")
    for candidate in candidates:
        relative = candidate.relative_to(root)
        path = _plain_resolve(
            candidate,
            name=f"source tree file {relative.as_posix()}",
            kind="file",
        )
        if not path.is_relative_to(root):
            raise RCQInputError("source tree file escaped its registered root")
        before = path.stat(follow_symlinks=False)
        if before.st_nlink != 1:
            raise RCQInputError(
                f"source tree file {relative.as_posix()!r} has an external hardlink"
            )
        name = relative.as_posix().encode("utf-8")
        content = path.read_bytes()
        after = path.stat(follow_symlinks=False)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if after.st_nlink != 1 or identity(before) != identity(after):
            raise RCQInputError(
                f"source tree file {relative.as_posix()!r} changed while hashing"
            )
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _dataset_manifest_from_training_config(
    config: TrainingConfig,
    *,
    split: DatasetSplit,
    sequence_count: int,
    seed_offset: int,
) -> str:
    return dataset_manifest_sha256(
        MovingShapesDatasetConfig(
            split=split,
            sequence_count=sequence_count,
            sequence_length=config.dataset.sequence_length,
            seed_offset=seed_offset,
            hazard_count=config.dataset.hazard_count,
            tick_period_ns=config.dataset.tick_period_ns,
            discount=config.dataset.discount,
        )
    )


def _metadata_only_batch_source_manifest_sha256(config: TrainingConfig) -> str:
    """Recompute the trainer batch-source identity without dataset wrappers."""

    counts = {
        DatasetSplit.TRAIN: config.dataset.train_sequences,
        DatasetSplit.VALIDATION: config.dataset.validation_sequences,
        DatasetSplit.TEST: config.dataset.test_sequences,
    }
    manifest = {
        "schema_version": 1,
        "batch_source": "moving_shapes_split_namespaces",
        "control_layout": CONTROL_LAYOUT_ID,
        "input_boundary": "ModelObservation-v1",
        "burn_in_steps": config.dataset.burn_in_steps,
        "splits": {
            split.value: _dataset_manifest_from_training_config(
                config,
                split=split,
                sequence_count=count,
                seed_offset=config.dataset.seed_offset,
            )
            for split, count in sorted(counts.items(), key=lambda item: item[0].value)
        },
    }
    encoded = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return sha256(b"IRTRAINBATCH\x01" + encoded).hexdigest()


class _PreclaimBatchSource:
    """TRAIN/DEVELOPMENT-only source; sealed and placeholder TEST stay unopened."""

    def __init__(self, config: TrainingConfig) -> None:
        if not isinstance(config, TrainingConfig):
            raise RCQInputError("preclaim source requires a TrainingConfig")
        self.config: DatasetConfig = config.dataset
        self.manifest_sha256 = _metadata_only_batch_source_manifest_sha256(config)
        self._datasets = {
            DatasetSplit.TRAIN: MovingShapesSequenceDataset(
                MovingShapesDatasetConfig(
                    split=DatasetSplit.TRAIN,
                    sequence_count=config.dataset.train_sequences,
                    sequence_length=config.dataset.sequence_length,
                    seed_offset=config.dataset.seed_offset,
                    hazard_count=config.dataset.hazard_count,
                    tick_period_ns=config.dataset.tick_period_ns,
                    discount=config.dataset.discount,
                )
            ),
            DatasetSplit.VALIDATION: MovingShapesSequenceDataset(
                MovingShapesDatasetConfig(
                    split=DatasetSplit.VALIDATION,
                    sequence_count=config.dataset.validation_sequences,
                    sequence_length=config.dataset.sequence_length,
                    seed_offset=config.dataset.seed_offset,
                    hazard_count=config.dataset.hazard_count,
                    tick_period_ns=config.dataset.tick_period_ns,
                    discount=config.dataset.discount,
                )
            ),
        }

    @staticmethod
    def _split(value: str) -> DatasetSplit:
        try:
            split = DatasetSplit(value)
        except (TypeError, ValueError) as error:
            raise RCQInputError("preclaim split must be train or validation") from error
        if split is DatasetSplit.TEST:
            raise RCQInputError("preclaim source categorically forbids TEST")
        return split

    def batches_per_epoch(self, *, split: str, batch_size: int) -> int:
        if type(batch_size) is not int or batch_size < 1:
            raise RCQInputError("batch_size must be a positive integer")
        return ceil(len(self._datasets[self._split(split)]) / batch_size)

    def iter_batches(
        self,
        *,
        split: str,
        epoch: int,
        start_batch: int,
        batch_size: int,
        max_batches: int | None = None,
    ) -> Iterator[TrajectoryBatch]:
        partition = self._split(split)
        if type(epoch) is not int or epoch < 0:
            raise RCQInputError("epoch must be a nonnegative integer")
        if type(start_batch) is not int or start_batch < 0:
            raise RCQInputError("start_batch must be a nonnegative integer")
        if type(batch_size) is not int or batch_size < 1:
            raise RCQInputError("batch_size must be a positive integer")
        if max_batches is not None and (
            type(max_batches) is not int or max_batches < 1
        ):
            raise RCQInputError("max_batches must be a positive integer or None")
        dataset = self._datasets[partition]
        total_batches = self.batches_per_epoch(
            split=partition.value,
            batch_size=batch_size,
        )
        if start_batch > total_batches:
            raise RCQInputError("start_batch exceeds the number of batches")
        indices = dataset.epoch_indices(
            epoch=epoch,
            shuffle=partition is DatasetSplit.TRAIN,
        )
        emitted = 0
        for batch_index in range(start_batch, total_batches):
            if max_batches is not None and emitted >= max_batches:
                break
            start = batch_index * batch_size
            selected = indices[start : start + batch_size]
            yield TrajectoryBatch(
                split=partition.value,
                burn_in_steps=self.config.burn_in_steps,
                sequences=tuple(dataset[index] for index in selected),
            )
            emitted += 1


def evaluator_bundle_sha256(source_root: str | os.PathLike[str]) -> str:
    """Hash the exact three-file RCQ evaluator bundle for preregistration."""

    root = _plain_resolve(source_root, name="source_root", kind="directory")
    digest = sha256(b"IRENERCQEVALUATOR\x01")
    for relative_name in EVALUATOR_BUNDLE_FILES:
        relative = Path(relative_name)
        path = _plain_resolve(
            root / relative,
            name=f"evaluator bundle {relative.as_posix()}",
            kind="file",
        )
        if not path.is_file() or not path.is_relative_to(root):
            raise RCQInputError("evaluator bundle path escaped the source tree")
        name = relative.as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _inside_file(path: str | os.PathLike[str], root: Path, *, name: str) -> Path:
    resolved = _plain_resolve(path, name=name, kind="file")
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise RCQInputError(f"{name} must be a file inside the training release")
    return resolved


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RCQInputError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise RCQInputError(f"non-finite JSON number is forbidden: {value}")


def _strict_json_bytes(encoded: bytes, *, name: str, canonical_line: bool) -> object:
    try:
        text = encoded.decode("utf-8", errors="strict")
        raw = json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except RCQInputError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise RCQInputError(f"{name} is not strict JSON: {error}") from error
    if canonical_line and encoded != (_canonical(raw) + "\n").encode("utf-8"):
        raise RCQInputError(f"{name} is not a byte-canonical JSON line")
    return raw


def _strict_json_file(path: Path, *, name: str) -> tuple[object, str]:
    path = _plain_resolve(path, name=name, kind="file")
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise RCQInputError(f"cannot read {name}: {error}") from error
    return (
        _strict_json_bytes(encoded, name=name, canonical_line=True),
        sha256(encoded).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class _PinnedDocument:
    path: Path
    payload: Mapping[str, object]
    file_sha256: str
    semantic_sha256: str

    def __post_init__(self) -> None:
        _hash_string(self.file_sha256, name="pin file sha256")
        _hash_string(self.semantic_sha256, name="pin semantic sha256")
        if not isinstance(self.payload, Mapping):
            raise RCQInputError("pin payload must be a mapping")
        object.__setattr__(self, "payload", _deep_freeze(self.payload))


def _exact_dict(value: object, fields: set[str], *, name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise RCQInputError(f"{name} has incompatible fields")
    return value


def _utc_timestamp(value: object, *, name: str) -> str:
    if type(value) is not str or re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        value,
    ) is None:
        raise RCQInputError(f"{name} must be a second-precision RFC3339 UTC timestamp")
    return value


def _relative_posix(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise RCQInputError(f"{name} must be a nonempty relative POSIX path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or value != parsed.as_posix()
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise RCQInputError(f"{name} must be a canonical safe relative POSIX path")
    return value


def _token(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or not 1 <= len(value) <= 128
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None
    ):
        raise RCQInputError(f"{name} must be a safe identifier")
    return value


def _require_pin_root(*, final_authorization_required: bool) -> Path:
    root = _plain_resolve(
        _CONTAINER_PIN_ROOT,
        name="canonical read-only qualification pin mount",
        kind="directory",
    )
    if root.as_posix() != "/workspace/pins":
        raise RCQInputError("qualification pin mount is not canonical")
    if os.name != "posix":
        raise RCQInputError("qualification pin validation requires the registered Linux runtime")
    descriptor = _open_plain_directory_chain(root)
    try:
        metadata = os.fstat(descriptor)
        if (
            metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise RCQInputError(
                "qualification pin mount must be owner-controlled and not group/other writable"
            )
        observed = set(os.listdir(descriptor))
        expected = {"pretraining.json"}
        if final_authorization_required:
            expected.add("final-authorization.json")
        elif "final-authorization.json" in observed:
            expected.add("final-authorization.json")
        if observed != expected:
            raise RCQInputError("qualification pin mount contains unexpected entries")
    finally:
        os.close(descriptor)
    return root


def _strict_owned_pin_file(path: Path, *, name: str) -> tuple[dict[str, object], str]:
    parent = _require_pin_root(
        final_authorization_required=path.name == "final-authorization.json"
    )
    if path.parent != parent or path.name not in {
        "pretraining.json",
        "final-authorization.json",
    }:
        raise RCQInputError(f"{name} path is not fixed")
    parent_descriptor = _open_plain_directory_chain(parent)
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o400
        ):
            raise RCQInputError(f"{name} must be an owner-only 0400 regular file")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field)
            or getattr(after, field) != getattr(named, field)
            for field in identity_fields
        ):
            raise RCQInputError(f"{name} changed while being read")
        encoded = b"".join(chunks)
    except OSError as error:
        raise RCQInputError(f"cannot securely read {name}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
    raw = _strict_json_bytes(encoded, name=name, canonical_line=True)
    if type(raw) is not dict:
        raise RCQInputError(f"{name} must contain a JSON object")
    return raw, _digest_bytes(encoded)


def _load_pretraining_pin() -> _PinnedDocument:
    raw, file_digest = _strict_owned_pin_file(
        _PRETRAINING_PIN_PATH,
        name="canonical pretraining pin",
    )
    _exact_dict(
        raw,
        {
            "schema_version",
            "action",
            "qualification_id",
            "workspace",
            "release",
            "registration",
            "config",
            "source_tree_sha256",
            "evaluator_bundle_sha256",
            "batch_source_manifest_sha256",
            "runtime",
            "run",
            "range_claim_id",
            "created_utc",
            "pin_sha256",
        },
        name="pretraining pin",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["action"] != "rcq_v2_pin_pretraining_v1"
        or raw["qualification_id"] != "rcq_v2_reference_v2"
        or raw["range_claim_id"] != _final_range_claim_id()
    ):
        raise RCQInputError("pretraining pin identity changed")
    workspace = _exact_dict(
        raw["workspace"],
        {
            "contract",
            "host_account_home_relative_path",
            "marker_relative_path",
            "marker_file_sha256",
            "claim_registry_relative_path",
            "pin_directory_relative_path",
        },
        name="pretraining workspace pin",
    )
    expected_workspace = {
        "contract": "pseudo-brain-workspace-v2",
        "host_account_home_relative_path": "projects/pseudo-brain",
        "marker_relative_path": ".pseudo-brain-workspace-v2",
        "marker_file_sha256": _digest_bytes(b"pseudo-brain-workspace-v2\n"),
        "claim_registry_relative_path": "final-claims",
        "pin_directory_relative_path": "qualification-pins/rcq-v2-reference-v2",
    }
    if not _strict_equal(workspace, expected_workspace):
        raise RCQInputError("pretraining workspace pin changed")
    for field in (
        "host_account_home_relative_path",
        "marker_relative_path",
        "claim_registry_relative_path",
        "pin_directory_relative_path",
    ):
        _relative_posix(workspace[field], name=f"pretraining workspace {field}")
    release = _exact_dict(
        raw["release"],
        {"id", "relative_path", "archive_sha256"},
        name="pretraining release pin",
    )
    release_id = _token(release["id"], name="release id")
    _relative_posix(release["relative_path"], name="release relative path")
    if release["relative_path"] != f"releases/{release_id}":
        raise RCQInputError("pretraining release path differs from its id")
    _hash_string(release["archive_sha256"], name="release archive sha256")
    registration = _exact_dict(
        raw["registration"],
        {"release_relative_path", "sha256"},
        name="pretraining registration pin",
    )
    if registration["release_relative_path"] != _REGISTRATION_RELATIVE_PATH.as_posix():
        raise RCQInputError("pretraining registration path changed")
    _relative_posix(
        registration["release_relative_path"],
        name="registration release relative path",
    )
    _hash_string(registration["sha256"], name="pretraining registration sha256")
    config = _exact_dict(
        raw["config"],
        {"release_relative_path", "raw_sha256", "canonical_sha256"},
        name="pretraining config pin",
    )
    if config["release_relative_path"] != _REGISTERED_CONFIG_RELATIVE_PATH.as_posix():
        raise RCQInputError("pretraining config path changed")
    _relative_posix(
        config["release_relative_path"],
        name="config release relative path",
    )
    for field in ("raw_sha256", "canonical_sha256"):
        _hash_string(config[field], name=f"pretraining config {field}")
    for field in (
        "source_tree_sha256",
        "evaluator_bundle_sha256",
        "batch_source_manifest_sha256",
    ):
        _hash_string(raw[field], name=f"pretraining {field}")
    runtime = _exact_dict(
        raw["runtime"],
        {
            "container_image_reference",
            "container_image_id",
            "container_release_root",
            "container_run_root",
            "container_claim_registry_root",
            "container_pin_root",
            "network",
        },
        name="pretraining runtime pin",
    )
    if (
        type(runtime["container_image_reference"]) is not str
        or not runtime["container_image_reference"]
        or len(runtime["container_image_reference"]) > 512
        or runtime["container_image_reference"].strip()
        != runtime["container_image_reference"]
        or any(character.isspace() for character in runtime["container_image_reference"])
        or type(runtime["container_image_id"]) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", runtime["container_image_id"]) is None
        or runtime["container_release_root"] != "/workspace/repo"
        or runtime["container_run_root"] != "/workspace/run"
        or runtime["container_claim_registry_root"] != "/workspace/final-claims"
        or runtime["container_pin_root"] != "/workspace/pins"
        or runtime["network"] != "none"
    ):
        raise RCQInputError("pretraining runtime pin changed")
    run = _exact_dict(
        raw["run"],
        {"id", "relative_path", "seed", "final_step"},
        name="pretraining run pin",
    )
    _relative_posix(run["relative_path"], name="run relative path")
    if not _strict_equal(
        run,
        {
            "id": "dgx-rcq-v2-reference-seed-1702",
            "relative_path": "runs/dgx-rcq-v2-reference-seed-1702",
            "seed": 1702,
            "final_step": FINAL_STEP,
        },
    ):
        raise RCQInputError("pretraining run pin changed")
    _utc_timestamp(raw["created_utc"], name="pretraining pin created_utc")
    observed_semantic = _hash_string(raw["pin_sha256"], name="pretraining pin sha256")
    body = dict(raw)
    body.pop("pin_sha256")
    expected_semantic = _digest_payload(_PRETRAINING_PIN_DOMAIN, body)
    if observed_semantic != expected_semantic:
        raise RCQInputError("pretraining pin semantic digest changed")
    return _PinnedDocument(
        path=_PRETRAINING_PIN_PATH,
        payload=raw,
        file_sha256=file_digest,
        semantic_sha256=expected_semantic,
    )


def _load_final_authorization() -> _PinnedDocument:
    raw, file_digest = _strict_owned_pin_file(
        _FINAL_AUTHORIZATION_PATH,
        name="canonical final authorization",
    )
    _exact_dict(
        raw,
        {
            "schema_version",
            "action",
            "qualification_id",
            "pretraining",
            "latest",
            "entry_checkpoint",
            "checkpoint",
            "readiness",
            "range_claim_id",
            "reviewed_utc",
            "authorization_sha256",
        },
        name="final authorization",
    )
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or raw["action"] != "rcq_v2_authorize_final_v1"
        or raw["qualification_id"] != "rcq_v2_reference_v2"
        or raw["range_claim_id"] != _final_range_claim_id()
    ):
        raise RCQInputError("final authorization identity changed")
    pretraining = _exact_dict(
        raw["pretraining"],
        {"relative_path", "file_sha256", "pin_sha256"},
        name="final authorization pretraining link",
    )
    if pretraining["relative_path"] != "pretraining.json":
        raise RCQInputError("final authorization pretraining path changed")
    _relative_posix(pretraining["relative_path"], name="authorized pretraining path")
    _hash_string(pretraining["file_sha256"], name="authorized pretraining file sha256")
    _hash_string(pretraining["pin_sha256"], name="authorized pretraining pin sha256")
    latest = _exact_dict(
        raw["latest"],
        {
            "run_relative_path",
            "relative_path",
            "file_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "optimizer_step",
        },
        name="final authorization latest link",
    )
    if not _strict_equal(
        {
            "run_relative_path": latest["run_relative_path"],
            "relative_path": latest["relative_path"],
            "checkpoint": latest["checkpoint"],
            "optimizer_step": latest["optimizer_step"],
        },
        {
            "run_relative_path": "runs/dgx-rcq-v2-reference-seed-1702",
            "relative_path": "checkpoints/latest.json",
            "checkpoint": _EXPECTED_CHECKPOINT_NAME,
            "optimizer_step": FINAL_STEP,
        },
    ):
        raise RCQInputError("final authorization latest identity changed")
    _relative_posix(latest["run_relative_path"], name="authorized run path")
    _relative_posix(latest["relative_path"], name="authorized latest path")
    _hash_string(latest["file_sha256"], name="authorized latest file sha256")
    _hash_string(latest["checkpoint_sha256"], name="authorized latest checkpoint sha256")
    entry_checkpoint = _exact_dict(
        raw["entry_checkpoint"],
        {"relative_path", "sha256"},
        name="final authorization entry checkpoint link",
    )
    if (
        entry_checkpoint["relative_path"]
        != f"checkpoints/{_EXPECTED_ENTRY_CHECKPOINT_NAME}"
    ):
        raise RCQInputError("final authorization entry checkpoint path changed")
    _relative_posix(
        entry_checkpoint["relative_path"],
        name="authorized entry checkpoint path",
    )
    _hash_string(
        entry_checkpoint["sha256"],
        name="authorized entry checkpoint sha256",
    )
    checkpoint = _exact_dict(
        raw["checkpoint"],
        {"relative_path", "sha256"},
        name="final authorization checkpoint link",
    )
    if checkpoint["relative_path"] != f"checkpoints/{_EXPECTED_CHECKPOINT_NAME}":
        raise RCQInputError("final authorization checkpoint path changed")
    _relative_posix(checkpoint["relative_path"], name="authorized checkpoint path")
    _hash_string(checkpoint["sha256"], name="authorized checkpoint sha256")
    readiness = _exact_dict(
        raw["readiness"],
        {"relative_path", "file_sha256", "readiness_sha256"},
        name="final authorization readiness link",
    )
    if (
        readiness["relative_path"]
        != "final-claims/preclaim-readiness/rcq-v2-reference-v2.json"
    ):
        raise RCQInputError("final authorization readiness path changed")
    _relative_posix(readiness["relative_path"], name="authorized readiness path")
    _hash_string(readiness["file_sha256"], name="authorized readiness file sha256")
    _hash_string(readiness["readiness_sha256"], name="authorized readiness sha256")
    _utc_timestamp(raw["reviewed_utc"], name="final authorization reviewed_utc")
    observed_semantic = _hash_string(
        raw["authorization_sha256"],
        name="final authorization sha256",
    )
    body = dict(raw)
    body.pop("authorization_sha256")
    expected_semantic = _digest_payload(_FINAL_AUTHORIZATION_DOMAIN, body)
    if observed_semantic != expected_semantic:
        raise RCQInputError("final authorization semantic digest changed")
    return _PinnedDocument(
        path=_FINAL_AUTHORIZATION_PATH,
        payload=raw,
        file_sha256=file_digest,
        semantic_sha256=expected_semantic,
    )


def _resolve_release() -> tuple[Path, Path, Path, Path]:
    release = _plain_resolve(
        _CONTAINER_RELEASE_ROOT,
        name="training_release_root",
        kind="directory",
    )
    source = _plain_resolve(
        release / "brain" / "src" / "irene_brain",
        name="training source root",
        kind="directory",
    )
    config = _inside_file(
        release / _REGISTERED_CONFIG_RELATIVE_PATH,
        release,
        name="config_path",
    )
    registration = _inside_file(
        release / _REGISTRATION_RELATIVE_PATH,
        release,
        name="registration_path",
    )
    expected_config = _plain_resolve(
        release / _REGISTERED_CONFIG_RELATIVE_PATH,
        name="registered training config",
        kind="file",
    )
    expected_registration = _plain_resolve(
        release / _REGISTRATION_RELATIVE_PATH,
        name="fixed RCQ-v2 registration",
        kind="file",
    )
    if config != expected_config:
        raise RCQInputError("config_path is not the fixed RCQ-v2 reference config")
    if registration != expected_registration:
        raise RCQInputError("registration_path is not the fixed RCQ-v2 registration")
    package = importlib.import_module("irene_brain")
    package_file = getattr(package, "__file__", None)
    if not package_file or _plain_resolve(
        package_file,
        name="imported irene_brain package",
        kind="file",
    ).parent != source:
        raise RCQInputError("imported irene_brain package is not the identified release")
    return release, source, config, registration


def _require_registered_config(
    config: TrainingConfig,
    config_path: Path,
    registration: RCQRegistration,
) -> None:
    raw_digest = _file_sha256(config_path)
    payload = registration.payload
    _require_live_development_gate_protocol(registration)
    if raw_digest != payload["config_raw_sha256"]:
        raise RCQInputError("raw training configuration differs from registration")
    if config.config_sha256 != payload["config_canonical_sha256"]:
        raise RCQInputError("canonical training configuration differs from registration")
    comparisons = (
        ("schema_version", config.schema_version, 3),
        ("run seed", config.run.seed, payload["run_seed"]),
        ("model factory", config.run.model_factory, payload["model_factory"]),
        ("final step", config.run.max_optimizer_steps, payload["final_step"]),
        ("sequence length", config.dataset.sequence_length, payload["sequence_length"]),
        ("burn in", config.dataset.burn_in_steps, payload["burn_in_steps"]),
        ("hazard count", config.dataset.hazard_count, payload["hazard_count"]),
        ("tick period", config.dataset.tick_period_ns, payload["tick_period_ns"]),
        ("discount", config.dataset.discount.hex(), payload["discount_hex"]),
    )
    for name, observed, expected in comparisons:
        if type(observed) is not type(expected) or observed != expected:
            raise RCQInputError(f"registered configuration {name} changed")
    if (
        config.dataset.train_sequences,
        config.dataset.validation_sequences,
        config.dataset.test_sequences,
        config.optimization.batch_size,
        config.optimization.gradient_accumulation_steps,
        config.logging.log_every_steps,
        config.logging.evaluate_every_steps,
        config.logging.validation_batches,
        config.logging.checkpoint_every_steps,
    ) != (8_192, 256, 512, 1, 8, 64, 256, 256, 256):
        raise RCQInputError("registered dataset/optimizer/logging cadence changed")
    if (
        config.dataset.seed_offset != 1_048_576
        or config.dataset.test_sequences != 512
        or registration.payload["config_test_field_status"]
        != "disabled_retired_placeholder_generic_trainer_test_forbidden"
    ):
        raise RCQInputError(
            "schema-3 config TEST must remain the disabled retired placeholder"
        )
    if len(config.stages) != 2:
        raise RCQInputError("RCQ-v2 requires exactly two optimizer stages")
    joint, value = config.stages
    if (
        joint.index,
        joint.start_optimizer_step,
        joint.end_optimizer_step,
        joint.transition_gate,
        value.index,
        value.start_optimizer_step,
        value.end_optimizer_step,
        value.completion_gate,
        value.invariance_audit,
        value.trainable_parameters,
    ) != (
        0,
        0,
        payload["joint_end_step"],
        "rcq_v2_development_v1",
        FINAL_STAGE_INDEX,
        payload["joint_end_step"],
        FINAL_STEP,
        "rcq_v2_value_development_v1",
        "nonvalue_action_state_v1",
        ("model.value_per_thought.bias", "model.value_per_thought.weight"),
    ):
        raise RCQInputError("registered optimizer-stage protocol changed")
    if (
        config.precision.device,
        config.precision.mode,
        config.precision.allow_tf32,
        config.determinism.enabled,
        config.determinism.compile_model,
        config.determinism.num_workers,
    ) != ("cuda", "bfloat16", True, True, False, 0):
        raise RCQInputError("registered BF16 deterministic runtime protocol changed")
    resources = config.resources
    if (
        resources.allow_gpu is not True
        or resources.allow_capture is not False
        or resources.allow_hid_output is not False
        or resources.allow_background_threads is not False
        or resources.allow_network is not False
        or resources.allow_subprocess is not False
        or resources.write_artifacts is not True
    ):
        raise RCQInputError("registered resource-denial policy changed")


def _require_live_development_gate_protocol(
    registration: RCQRegistration,
) -> None:
    """Bind registration strings to the live gate implementation constants."""

    gates = registration.payload["development_gates"]
    if not isinstance(gates, Mapping):
        raise RCQInputError("registered development gates are not a mapping")
    entry = gates.get("entry")
    completion = gates.get("completion")
    if not isinstance(entry, Mapping) or not isinstance(completion, Mapping):
        raise RCQInputError("registered development gate records are incompatible")
    comparisons = (
        ("entry gate id", DEVELOPMENT_GATE_ID, entry.get("gate")),
        ("completion gate id", VALUE_DEVELOPMENT_GATE_ID, completion.get("gate")),
        (
            "completion entry gate id",
            DEVELOPMENT_GATE_ID,
            completion.get("entry_gate"),
        ),
        (
            "TRAIN conditional value baseline",
            VALUE_TRAIN_CONDITIONAL_BASELINE_MSE.hex(),
            completion.get("train_timestep_previous_wasd_dev_mse_hex"),
        ),
        (
            "absolute value-loss maximum",
            VALUE_ABSOLUTE_MAX_MSE.hex(),
            completion.get("absolute_max_mse_hex"),
        ),
        (
            "development target variance",
            VALUE_DEVELOPMENT_TARGET_VARIANCE.hex(),
            completion.get("dev_target_variance_hex"),
        ),
        (
            "entry improvement ratio",
            VALUE_IMPROVEMENT_RATIO.hex(),
            completion.get("entry_improvement_ratio_hex"),
        ),
    )
    for name, observed, expected in comparisons:
        if type(observed) is not type(expected) or observed != expected:
            raise RCQInputError(f"live {name} differs from preregistration")


def _factory(path: str) -> Callable[[TrainingConfig], object]:
    try:
        module_name, attribute = path.split(":", 1)
    except ValueError as error:
        raise RCQInputError("model factory must use module.path:callable syntax") from error
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise RCQInputError("registered model factory is not callable")
    return factory


def _objective(config: TrainingConfig) -> object:
    from ..training.objective import ThoughtFieldObjective

    model = _factory(config.run.model_factory)(config)
    return ThoughtFieldObjective(
        model,
        action_loss_kind=config.objective.action_loss_kind,
        button_support_control_indices=config.objective.button_support_control_indices,
        button_support_weight=config.objective.button_support_weight,
        button_background_weight=config.objective.button_background_weight,
        button_background_tail_mix=config.objective.button_background_tail_mix,
        button_background_tail_temperature=config.objective.button_background_tail_temperature,
        continuous_action_weight=config.objective.continuous_action_weight,
        action_weight=config.objective.action_weight,
        value_weight=config.objective.value_weight,
        world_weight=config.objective.world_weight,
        diversity_weight=config.objective.diversity_weight,
    )


def _require_runtime_fingerprint(
    fingerprint: Mapping[str, str | int | bool],
) -> str:
    exact = {
        "device_type": "cuda",
        "precision": "bfloat16",
        "allow_tf32": True,
        "deterministic_algorithms": True,
        "compile_model": False,
        "num_workers": 0,
        "world_size": 1,
        "control_layout": "generic-hid-307-v1",
    }
    for name, expected in exact.items():
        observed = fingerprint.get(name)
        if type(observed) is not type(expected) or observed != expected:
            raise RCQInputError(f"runtime fingerprint field {name!r} changed")
    if fingerprint.get("cublas_workspace_config") not in {":4096:8", ":16:8"}:
        raise RCQInputError("runtime fingerprint lacks deterministic CUBLAS policy")
    if fingerprint.get("sdpa_policy") != "math_only":
        raise RCQInputError("runtime fingerprint lacks deterministic math-only SDPA")
    return _digest_payload(b"IRENERCQRUNTIME\x01", dict(fingerprint))


def _checkpoint_digest_from_latest(run_dir: Path, checkpoint: Path) -> str:
    pointer_path = run_dir / "checkpoints" / "latest.json"
    raw, _pointer_digest = _strict_json_file(pointer_path, name="latest pointer")
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "checkpoint",
        "checkpoint_sha256",
        "optimizer_step",
    }:
        raise RCQInputError("latest pointer has incompatible fields")
    if (
        type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
        or type(raw["checkpoint"]) is not str
        or raw["checkpoint"] != _EXPECTED_CHECKPOINT_NAME
        or type(raw["optimizer_step"]) is not int
        or raw["optimizer_step"] != FINAL_STEP
    ):
        raise RCQInputError("latest pointer does not identify the exact final checkpoint")
    digest = _hash_string(raw["checkpoint_sha256"], name="latest checkpoint sha256")
    if checkpoint.parent != pointer_path.parent or checkpoint.name != _EXPECTED_CHECKPOINT_NAME:
        raise RCQInputError("checkpoint path is not the registered final run boundary")
    if _file_sha256(checkpoint) != digest:
        raise RCQInputError("latest pointer digest differs from final checkpoint bytes")
    if _file_sha256(pointer_path) != _pointer_digest:
        raise RCQInputError("latest pointer bytes changed during validation")
    return digest


def _validate_metrics_prefix(
    run_dir: Path,
    trainer_state: Mapping[str, object],
    *,
    config: TrainingConfig,
    batches_per_epoch: int,
    terminal_step: int,
) -> dict[int, dict[str, float]]:
    if type(terminal_step) is not int or terminal_step not in {1_536, FINAL_STEP}:
        raise RCQInputError("metrics prefix boundary is not registered")
    required = {
        "schema_version",
        "metrics_byte_length",
        "metrics_record_count",
        "metrics_sha256",
    }
    if not isinstance(trainer_state, Mapping) or set(trainer_state) != required:
        raise RCQInputError("terminal checkpoint trainer state is incompatible")
    if type(trainer_state["schema_version"]) is not int or trainer_state["schema_version"] != 1:
        raise RCQInputError("terminal metrics schema version changed")
    byte_length = trainer_state["metrics_byte_length"]
    record_count = trainer_state["metrics_record_count"]
    digest = trainer_state["metrics_sha256"]
    if type(byte_length) is not int or byte_length < 1:
        raise RCQInputError("terminal metrics byte length is invalid")
    if type(record_count) is not int or record_count < 1:
        raise RCQInputError("terminal metrics record count is invalid")
    _hash_string(digest, name="terminal metrics sha256")
    metrics_path = _plain_resolve(
        run_dir / "metrics.jsonl",
        name="durable metrics",
        kind="file",
    )
    durable_bytes = metrics_path.read_bytes()
    if len(durable_bytes) < byte_length:
        raise RCQInputError("durable metrics file is shorter than checkpoint prefix")
    encoded = durable_bytes[:byte_length]
    if sha256(encoded).hexdigest() != digest:
        raise RCQInputError("durable metrics file differs from checkpoint-bound prefix")
    if terminal_step == FINAL_STEP and len(durable_bytes) != byte_length:
        raise RCQInputError("terminal checkpoint does not bind the complete metrics file")
    lines = encoded.splitlines(keepends=True)
    if len(lines) != record_count or any(not line.endswith(b"\n") for line in lines):
        raise RCQInputError("durable metrics record count is inconsistent")
    observed_order: list[tuple[str, int, int]] = []
    boundary_validation_metrics: dict[int, dict[str, float]] = {}
    for line in lines:
        raw = _strict_json_bytes(line, name="metrics record", canonical_line=True)
        if not isinstance(raw, dict) or set(raw) != {
            "schema_version", "step", "epoch", "split", "metrics"
        }:
            raise RCQInputError("durable metrics record schema changed")
        step = raw["step"]
        epoch = raw["epoch"]
        split = raw["split"]
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
            raise RCQInputError("durable metrics schema version changed")
        metrics = raw["metrics"]
        if (
            type(step) is not int
            or type(epoch) is not int
            or type(split) is not str
            or split not in {"train", "validation"}
            or not isinstance(metrics, dict)
        ):
            raise RCQInputError("durable metrics record values are invalid")
        expected_stage = 0 if step <= config.stages[0].end_optimizer_step else 1
        expected_samples = float(
            (
                config.optimization.gradient_accumulation_steps
                * config.optimization.batch_size
                if split == "train"
                else config.dataset.validation_sequences
            )
            * (config.dataset.sequence_length - config.dataset.burn_in_steps)
        )
        if (
            type(metrics.get("stage_index")) is not float
            or metrics.get("stage_index") != float(expected_stage)
            or type(metrics.get("samples")) is not float
            or metrics.get("samples") != expected_samples
            or type(metrics.get("loss")) is not float
        ):
            raise RCQInputError("durable metrics stage/sample envelope is incompatible")
        if not metrics or any(
            type(name) is not str
            or not name
            or type(value) is not float
            or not isfinite(value)
            or (value == 0.0 and value.hex() != 0.0.hex())
            for name, value in metrics.items()
        ):
            raise RCQInputError("durable metrics values are not canonical finite floats")
        if split == "validation" and step in {1_536, FINAL_STEP}:
            boundary_validation_metrics[step] = dict(metrics)
        observed_order.append((split, step, epoch))
    expected_order: list[tuple[str, int, int]] = []
    accumulation = config.optimization.gradient_accumulation_steps
    for step in range(1, terminal_step + 1):
        epoch = (step * accumulation) // batches_per_epoch
        if step == 1 or step % config.logging.log_every_steps == 0:
            expected_order.append(("train", step, epoch))
        if step % config.logging.evaluate_every_steps == 0:
            expected_order.append(("validation", step, epoch))
    if observed_order != expected_order:
        raise RCQInputError("durable metrics differs from the exact registered cadence")
    expected_boundaries = {1_536}
    if terminal_step == FINAL_STEP:
        expected_boundaries.add(FINAL_STEP)
    if set(boundary_validation_metrics) != expected_boundaries:
        raise RCQInputError("durable metrics lacks exact development gate boundaries")
    final_bytes = metrics_path.read_bytes()
    if (
        len(final_bytes) < byte_length
        or final_bytes[:byte_length] != encoded
        or sha256(final_bytes[:byte_length]).hexdigest() != digest
        or (terminal_step == FINAL_STEP and len(final_bytes) != byte_length)
    ):
        raise RCQInputError("durable metrics changed during prefix validation")
    return boundary_validation_metrics


def _require_gate_metrics_link(
    *,
    report_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    name: str,
) -> None:
    """Require one gate report to be the exact durable validation result."""

    envelope = {"loss", "samples", "stage_index"}
    if set(validation_metrics) != set(report_metrics) | envelope:
        raise RCQInputError(f"{name} metrics differ from the durable validation schema")
    for metric_name, report_value in report_metrics.items():
        validation_value = validation_metrics.get(metric_name)
        if (
            type(report_value) is not float
            or type(validation_value) is not float
            or report_value.hex() != validation_value.hex()
        ):
            raise RCQInputError(
                f"{name} metric {metric_name!r} differs from durable validation"
            )
    total_loss = report_metrics.get("total_loss")
    validation_loss = validation_metrics.get("loss")
    if (
        type(total_loss) is not float
        or type(validation_loss) is not float
        or validation_loss.hex() != total_loss.hex()
    ):
        raise RCQInputError(f"{name} loss envelope differs from total_loss")


def _validate_entry_gate_artifact(
    path: Path,
    *,
    expected_digest: str,
    expected_validation_metrics: Mapping[str, float],
) -> tuple[str, RCQDevelopmentReport]:
    raw, digest = _strict_json_file(path, name=path.name)
    if digest != expected_digest:
        raise RCQInputError(f"{path.name} differs from checkpoint-bound digest")
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version", "gate", "scope", "passed", "checks", "metrics"
    }:
        raise RCQInputError(f"{path.name} has incompatible gate fields")
    if raw["passed"] is not True or not isinstance(raw["metrics"], dict):
        raise RCQInputError(f"{path.name} is not a passed development gate")
    from ..training.protocol import TrainingStepResult

    try:
        reconstructed = evaluate_rcq_v2_development(
            TrainingStepResult(loss=0.0, metrics=raw["metrics"], samples=DEVELOPMENT_SAMPLES)
        )
    except (TypeError, ValueError) as error:
        raise RCQInputError(f"{path.name} cannot be recomputed") from error
    if (reconstructed.canonical_json + "\n").encode("utf-8") != path.read_bytes():
        raise RCQInputError(f"{path.name} differs from recomputed gate evidence")
    _require_gate_metrics_link(
        report_metrics=reconstructed.metrics,
        validation_metrics=expected_validation_metrics,
        name=path.name,
    )
    if _file_sha256(path) != digest:
        raise RCQInputError(f"{path.name} changed during validation")
    return digest, reconstructed


def _validate_completion_gate_artifact(
    path: Path,
    *,
    expected_digest: str,
    entry_report: RCQDevelopmentReport,
    entry_digest: str,
    expected_validation_metrics: Mapping[str, float],
) -> tuple[str, RCQValueDevelopmentReport]:
    raw, digest = _strict_json_file(path, name=path.name)
    if digest != expected_digest:
        raise RCQInputError(f"{path.name} differs from checkpoint-bound digest")
    if not isinstance(raw, dict) or raw.get("passed") is not True:
        raise RCQInputError(f"{path.name} is not a passed value-development gate")
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        raise RCQInputError(f"{path.name} lacks value-development metrics")
    from ..training.protocol import TrainingStepResult

    try:
        reconstructed = evaluate_rcq_v2_value_development(
            TrainingStepResult(
                loss=0.0,
                metrics=metrics,
                samples=DEVELOPMENT_SAMPLES,
            ),
            entry_report=entry_report,
            entry_report_sha256=entry_digest,
        )
    except (TypeError, ValueError) as error:
        raise RCQInputError(f"{path.name} cannot be recomputed") from error
    if (reconstructed.canonical_json + "\n").encode("utf-8") != path.read_bytes():
        raise RCQInputError(f"{path.name} differs from recomputed value gate evidence")
    _require_gate_metrics_link(
        report_metrics=reconstructed.metrics,
        validation_metrics=expected_validation_metrics,
        name=path.name,
    )
    if _file_sha256(path) != digest:
        raise RCQInputError(f"{path.name} changed during validation")
    return digest, reconstructed


def _replay_registered_development(
    *,
    evaluate_batch: Callable[[object], object],
    iter_batches: Callable[..., object],
    batches_per_epoch: Callable[..., int],
    config: TrainingConfig,
) -> object:
    """Reproduce ``Trainer.evaluate(validation)`` without creating a Trainer."""

    from ..training.protocol import TrainingStepResult

    batch_count = min(
        config.logging.validation_batches,
        batches_per_epoch(
            split="validation",
            batch_size=config.optimization.batch_size,
        ),
    )
    results = tuple(
        evaluate_batch(batch)
        for batch in iter_batches(
            split="validation",
            epoch=0,
            start_batch=0,
            batch_size=config.optimization.batch_size,
            max_batches=batch_count,
        )
    )
    if not results or any(not isinstance(item, TrainingStepResult) for item in results):
        raise RCQInputError("registered DEVELOPMENT replay produced invalid batches")
    samples = sum(item.samples for item in results)
    loss = fsum(item.loss * (item.samples / samples) for item in results)
    names = set.intersection(*(set(item.metrics) for item in results))
    metrics = {
        name: fsum(
            item.metrics[name] * (item.samples / samples)
            for item in results
        )
        for name in sorted(names)
    }
    return TrainingStepResult(loss=loss, metrics=metrics, samples=samples)


def _require_development_replay(
    *,
    replay: object,
    entry_report: RCQDevelopmentReport,
    entry_digest: str,
    completion_report: RCQValueDevelopmentReport,
    final_validation_metrics: Mapping[str, float],
) -> None:
    """Bind the restored model to final DEVELOPMENT evidence before TEST claim."""

    from ..training.protocol import TrainingStepResult

    if not isinstance(replay, TrainingStepResult):
        raise RCQInputError("DEVELOPMENT replay returned an incompatible result")
    try:
        reconstructed = evaluate_rcq_v2_value_development(
            replay,
            entry_report=entry_report,
            entry_report_sha256=entry_digest,
        )
    except (TypeError, ValueError) as error:
        raise RCQInputError("restored checkpoint fails exact DEVELOPMENT replay") from error
    if reconstructed.canonical_json != completion_report.canonical_json:
        raise RCQInputError(
            "restored checkpoint DEVELOPMENT replay differs from completion evidence"
        )
    replay_envelope = {
        **dict(replay.metrics),
        "loss": float(replay.loss),
        "samples": float(replay.samples),
        "stage_index": float(FINAL_STAGE_INDEX),
    }
    if set(replay_envelope) != set(final_validation_metrics) or any(
        type(value) is not float
        or type(final_validation_metrics.get(name)) is not float
        or value.hex() != final_validation_metrics[name].hex()
        for name, value in replay_envelope.items()
    ):
        raise RCQInputError(
            "restored checkpoint DEVELOPMENT replay differs from durable step-2048 metrics"
        )
    entry_metrics = entry_report.metrics
    replay_metrics = replay.metrics
    if set(entry_metrics) != set(replay_metrics):
        raise RCQInputError("entry and final DEVELOPMENT metric schemas differ")
    value_dependent = {"value_loss", "total_loss"}
    for name in sorted(set(entry_metrics) - value_dependent):
        if (
            type(entry_metrics[name]) is not float
            or type(replay_metrics[name]) is not float
            or entry_metrics[name].hex() != replay_metrics[name].hex()
        ):
            raise RCQInputError(
                f"frozen non-value DEVELOPMENT metric {name!r} changed after entry gate"
            )


def _set_registered_entry_loss_weights(*, system: object, config: TrainingConfig) -> None:
    """Restore the stage-0 loss semantics used by the step-1536 gate replay."""

    objective = getattr(system, "objective", None)
    setter = getattr(objective, "set_loss_weights", None)
    if not callable(setter) or len(config.stages) != 2:
        raise RCQInputError("entry checkpoint objective cannot restore registered gate weights")
    stage = config.stages[0]
    setter(
        action_weight=stage.action_weight,
        value_weight=stage.value_weight,
        world_weight=stage.world_weight,
        diversity_weight=stage.diversity_weight,
    )


def _require_entry_development_replay(
    *,
    replay: object,
    entry_report: RCQDevelopmentReport,
    entry_validation_metrics: Mapping[str, float],
) -> None:
    """Bind the retained entry-boundary model to its DEVELOPMENT gate."""

    from ..training.protocol import TrainingStepResult

    if not isinstance(replay, TrainingStepResult):
        raise RCQInputError("entry DEVELOPMENT replay returned an incompatible result")
    try:
        reconstructed = evaluate_rcq_v2_development(replay)
    except (TypeError, ValueError) as error:
        raise RCQInputError("entry checkpoint fails exact DEVELOPMENT replay") from error
    if reconstructed.canonical_json != entry_report.canonical_json:
        raise RCQInputError(
            "entry checkpoint DEVELOPMENT replay differs from entry-gate evidence"
        )
    replay_envelope = {
        **dict(replay.metrics),
        "loss": float(replay.loss),
        "samples": float(replay.samples),
        "stage_index": 0.0,
    }
    if set(replay_envelope) != set(entry_validation_metrics) or any(
        type(value) is not float
        or type(entry_validation_metrics.get(name)) is not float
        or value.hex() != entry_validation_metrics[name].hex()
        for name, value in replay_envelope.items()
    ):
        raise RCQInputError(
            "entry checkpoint DEVELOPMENT replay differs from durable step-1536 metrics"
        )


def _registered_development_probe_batches(
    *,
    iter_batches: Callable[..., object],
    batches_per_epoch: Callable[..., int],
    config: TrainingConfig,
) -> tuple[object, ...]:
    """Materialize only the registered DEVELOPMENT batches for invariance probes."""

    batch_count = min(
        config.logging.validation_batches,
        batches_per_epoch(
            split="validation",
            batch_size=config.optimization.batch_size,
        ),
    )
    batches = tuple(
        iter_batches(
            split="validation",
            epoch=0,
            start_batch=0,
            batch_size=config.optimization.batch_size,
            max_batches=batch_count,
        )
    )
    if len(batches) != batch_count or batch_count < 1:
        raise RCQInputError("registered DEVELOPMENT probe slice is incomplete")
    return batches


def _require_exact_nonvalue_model_identity(
    *,
    entry_system: object,
    terminal_system: object,
    entry_stage_state: Mapping[str, object],
    terminal_stage_state: Mapping[str, object],
    development_batches: Sequence[object],
) -> Mapping[str, object]:
    """Compare every non-value tensor and fresh deterministic probe digest."""

    import torch

    entry_objective = getattr(entry_system, "objective", None)
    terminal_objective = getattr(terminal_system, "objective", None)
    if entry_objective is None or terminal_objective is None:
        raise RCQInputError("checkpoint systems lack objectives for non-value comparison")
    entry_state = entry_objective.state_dict()
    terminal_state = terminal_objective.state_dict()
    if not isinstance(entry_state, Mapping) or not isinstance(terminal_state, Mapping):
        raise RCQInputError("checkpoint objective state is incompatible")
    excluded = {
        "model.value_per_thought.weight",
        "model.value_per_thought.bias",
    }
    if (
        set(entry_state) != set(terminal_state)
        or not excluded <= set(entry_state)
    ):
        raise RCQInputError("entry and terminal objective tensor schemas differ")
    nonvalue_names = tuple(sorted(set(entry_state) - excluded))
    for name in nonvalue_names:
        entry_tensor = entry_state[name]
        terminal_tensor = terminal_state[name]
        if (
            not isinstance(entry_tensor, torch.Tensor)
            or not isinstance(terminal_tensor, torch.Tensor)
            or entry_tensor.dtype != terminal_tensor.dtype
            or tuple(entry_tensor.shape) != tuple(terminal_tensor.shape)
            or not torch.equal(
                entry_tensor.detach().cpu(),
                terminal_tensor.detach().cpu(),
            )
        ):
            raise RCQInputError(
                f"terminal checkpoint changed frozen non-value tensor {name!r}"
            )
    entry_capture = getattr(entry_system, "_capture_invariance_snapshot", None)
    terminal_capture = getattr(terminal_system, "_capture_invariance_snapshot", None)
    if not callable(entry_capture) or not callable(terminal_capture):
        raise RCQInputError("checkpoint systems cannot run deterministic invariance probes")
    try:
        entry_snapshot = entry_capture(development_batches)
        terminal_snapshot = terminal_capture(development_batches)
    except (TypeError, ValueError, RuntimeError) as error:
        raise RCQInputError("checkpoint invariance probe failed") from error
    if not isinstance(entry_snapshot, Mapping) or not isinstance(
        terminal_snapshot,
        Mapping,
    ):
        raise RCQInputError("checkpoint invariance probe returned incompatible evidence")
    if not _strict_equal(entry_snapshot, terminal_snapshot):
        raise RCQInputError(
            "entry and terminal deterministic action/recurrent probes differ"
        )
    for label, stage_state in (
        ("entry", entry_stage_state),
        ("terminal", terminal_stage_state),
    ):
        reference = stage_state.get("invariance_reference")
        current = stage_state.get("invariance_current")
        if (
            not isinstance(reference, Mapping)
            or not isinstance(current, Mapping)
            or not _strict_equal(entry_snapshot, reference)
            or not _strict_equal(entry_snapshot, current)
        ):
            raise RCQInputError(
                f"{label} checkpoint invariance evidence differs from fresh probes"
            )
    return _deep_freeze(
        {
            "non_value_tensor_count": len(nonvalue_names),
            "non_value_tensor_names_sha256": _digest_payload(
                b"PSEUDOBRAINRCQNONVALUENAMES\x01",
                list(nonvalue_names),
            ),
            "snapshot": dict(entry_snapshot),
        }
    )


def _validate_invariance_artifact(
    path: Path,
    *,
    config_sha256: str,
    stage_state: Mapping[str, object],
) -> str:
    raw, digest = _strict_json_file(path, name=path.name)
    expected = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "global_optimizer_step": FINAL_STEP,
        "report": {
            "schema_version": 1,
            "stage_index": FINAL_STAGE_INDEX,
            "audit": "nonvalue_action_state_v1",
            "passed": True,
            "reference": stage_state["invariance_reference"],
            "current": stage_state["invariance_current"],
        },
    }
    if _canonical(raw) != _canonical(expected):
        raise RCQInputError("final invariance artifact differs from checkpoint snapshots")
    if _file_sha256(path) != digest:
        raise RCQInputError("final invariance artifact changed during validation")
    return digest


def _registered_test_dataset(
    registration: RCQRegistration,
    *,
    role: str,
) -> object:
    from ..data.moving_shapes_dataset import _claimed_rcq_v2_test_dataset

    raw = registration.payload["slices"][role]  # type: ignore[index]
    if not isinstance(raw, Mapping):
        raise RCQInputError("registered final slice is not a mapping")
    dataset = _claimed_rcq_v2_test_dataset(
        MovingShapesDatasetConfig(
            split=DatasetSplit.TEST,
            sequence_count=raw["sequences"],
            sequence_length=registration.payload["sequence_length"],
            seed_offset=raw["local_start"],
            hazard_count=registration.payload["hazard_count"],
            tick_period_ns=registration.payload["tick_period_ns"],
            discount=float.fromhex(str(registration.payload["discount_hex"])),
        )
    )
    if dataset.manifest_sha256 != raw["manifest_sha256"]:
        raise RCQInputError(f"{role} TEST manifest differs from registration")
    return dataset


def _extract_action(output: object) -> tuple[tuple[int, ...], tuple[float, ...], float]:
    import torch

    from ..training.batches import BUTTON_TARGET_INDICES, CONTINUOUS_TARGET_INDICES

    action = getattr(output, "action", None)
    logits = getattr(action, "button_logits", None)
    control = getattr(action, "control", None)
    value = getattr(output, "value", None)
    if (
        not isinstance(logits, torch.Tensor)
        or tuple(logits.shape) != (1, len(BUTTON_TARGET_INDICES))
        or not isinstance(control, torch.Tensor)
        or tuple(control.shape) != (1, 307)
        or not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (1,)
    ):
        raise RCQInputError("model final exit has incompatible action/value shapes")
    if not (
        bool(torch.isfinite(logits).all())
        and bool(torch.isfinite(control).all())
        and bool(torch.isfinite(value).all())
    ):
        raise RCQInputError("model final exit contains non-finite values")
    packed = (logits[0].float() > 0.0).detach().cpu().tolist()
    active = tuple(
        BUTTON_TARGET_INDICES[index]
        for index, enabled in enumerate(packed)
        if enabled
    )
    continuous = tuple(
        float(control[0, index].float().detach().cpu().item())
        for index in CONTINUOUS_TARGET_INDICES
    )
    return active, continuous, float(value[0].float().detach().cpu().item())


def _run_sequence_condition(
    *,
    model: object,
    sequence: MovingShapesSequence,
    frame_for_time: Callable[[int], object],
    autocast_context: Callable[[], object],
) -> dict[int, tuple[tuple[int, ...], tuple[float, ...], float]]:
    import torch

    from ..training.batches import control_to_vector
    from ..training.objective import _rgb_tensor, deterministic_eval_thought_noise

    parameter = next(model.parameters())
    device = parameter.device
    thought_noise = deterministic_eval_thought_noise(
        thoughtlets=model.config.thoughtlets,
        width=model.config.core_width,
        batch_size=1,
        device=device,
    )
    state = None
    result: dict[int, tuple[tuple[int, ...], tuple[float, ...], float]] = {}
    with torch.no_grad():
        for time_index, transition in enumerate(sequence.transitions):
            pixels = _rgb_tensor(
                (frame_for_time(time_index),),
                device=device,
                resolution=getattr(model, "input_resolution", None),
            )
            previous = torch.tensor(
                [control_to_vector(transition.observation.previous_control)],
                dtype=torch.float32,
                device=device,
            )
            elapsed = torch.tensor(
                [
                    0.0
                    if time_index == 0
                    else (
                        transition.observation.elapsed_ns
                        - sequence.transitions[time_index - 1].observation.elapsed_ns
                    )
                    / 1_000_000_000.0
                ],
                dtype=torch.float32,
                device=device,
            )
            with autocast_context():
                output = model(
                    pixels,
                    previous,
                    elapsed,
                    state,
                    thought_noise=thought_noise,
                )
            state = output.next_state.detach()
            if time_index >= 2:
                result[time_index] = _extract_action(output)
    if set(result) != set(range(2, 8)):
        raise RCQInputError("final trajectory did not yield exactly six decisions")
    return result


def _collect_post_claim_evidence(
    *,
    registration: RCQRegistration,
    authorization: _FinalAuthorization,
    model: object,
    stage_state: Mapping[str, object],
    autocast_context: Callable[[], object],
    receipt_store: _BoundReceiptDirectory,
    expected_claim_entries: set[str],
    expected_claim_sha256: str,
    expected_claim_payload: Mapping[str, object],
) -> tuple[RCQFinalEvidence, DonorMapping]:
    from ..training.batches import (
        BUTTON_TARGET_INDICES,
        CONTINUOUS_TARGET_INDICES,
        control_to_vector,
    )

    claim_bytes = receipt_store.read_bytes(_CLAIM_NAME)
    receipt_store.require_entries(expected_claim_entries)
    claim = _strict_json_bytes(
        claim_bytes,
        name="canonical final claim",
        canonical_line=True,
    )
    if not isinstance(claim, dict) or _canonical(claim) != _canonical(
        expected_claim_payload
    ):
        raise RCQInputError("fresh TEST materialization requires the exact durable claim")
    claim_body = dict(claim)
    observed_claim_sha = claim_body.pop("claim_sha256", None)
    recomputed_claim_sha = _digest_payload(b"IRENERCQCLAIM\x01", claim_body)
    if (
        observed_claim_sha != expected_claim_sha256
        or recomputed_claim_sha != expected_claim_sha256
        or receipt_store.require_exact(_CLAIM_NAME, claim_bytes)
        != _digest_bytes(claim_bytes)
    ):
        raise RCQInputError("fresh TEST materialization requires the exact durable claim")
    # Chronology boundary: these constructors and all iteration occur only
    # after the exact durable claim above has been read back and verified.
    recipients_dataset = _registered_test_dataset(
        registration,
        role="final_recipient",
    )
    donors_dataset = _registered_test_dataset(
        registration,
        role="final_donor_only",
    )
    recipients = tuple(recipients_dataset)
    donors = tuple(donors_dataset)
    mapping = build_target_blind_donor_mapping(recipients, donors)
    assignment_index = mapping.by_recipient_time()
    donors_by_seed = {sequence.episode_seed: sequence for sequence in donors}
    if len(donors_by_seed) != DONOR_SEQUENCES:
        raise RCQInputError("donor TEST slice has duplicate episode seeds")
    decisions: list[RCQDecisionEvidence] = []
    for sequence in recipients:
        normal = _run_sequence_condition(
            model=model,
            sequence=sequence,
            frame_for_time=lambda index, sequence=sequence: (
                sequence.transitions[index].observation.rgb
            ),
            autocast_context=autocast_context,
        )

        def donor_frame(time_index: int) -> object:
            assignment = assignment_index[(sequence.episode_seed, time_index)]
            donor_sequence = donors_by_seed[assignment.donor_episode_seed]
            frame = donor_sequence.transitions[time_index].observation.rgb
            recipient_frame = sequence.transitions[time_index].observation.rgb
            if (
                frame.sha256 != assignment.donor_rgb_sha256
                or recipient_frame.sha256 != assignment.recipient_rgb_sha256
                or frame.sha256 == recipient_frame.sha256
            ):
                raise RCQInputError("donor mapping RGB identity changed during evaluation")
            return frame

        deranged = _run_sequence_condition(
            model=model,
            sequence=sequence,
            frame_for_time=donor_frame,
            autocast_context=autocast_context,
        )
        for time_index in range(2, 8):
            transition = sequence.transitions[time_index]
            assignment = assignment_index[(sequence.episode_seed, time_index)]
            target_vector = control_to_vector(transition.action_target)
            target_buttons = tuple(
                index for index in BUTTON_TARGET_INDICES if target_vector[index] > 0.5
            )
            target_continuous = tuple(
                float(target_vector[index]) for index in CONTINUOUS_TARGET_INDICES
            )
            previous_mask = _movement_mask(
                transition.observation.previous_control.keys_down
            )
            target_mask = _movement_mask(transition.action_target.keys_down)
            normal_buttons, normal_continuous, normal_value = normal[time_index]
            deranged_buttons, deranged_continuous, deranged_value = deranged[time_index]
            decisions.append(
                RCQDecisionEvidence(
                    recipient_episode_seed=sequence.episode_seed,
                    time_index=time_index,
                    donor_episode_seed=assignment.donor_episode_seed,
                    recipient_rgb_sha256=assignment.recipient_rgb_sha256,
                    donor_rgb_sha256=assignment.donor_rgb_sha256,
                    previous_movement_mask=previous_mask,
                    target_movement_mask=target_mask,
                    target_active_buttons=target_buttons,
                    normal_active_buttons=normal_buttons,
                    deranged_active_buttons=deranged_buttons,
                    target_continuous=target_continuous,
                    normal_continuous=normal_continuous,
                    deranged_continuous=deranged_continuous,
                    value_target=float(transition.value_target),
                    normal_value=normal_value,
                    deranged_value=deranged_value,
                    has_event=bool(transition.event_targets),
                    nonzero_return=transition.value_target != 0.0,
                )
            )
    if len(decisions) != FINAL_DECISIONS:
        raise RCQInputError("final evaluator did not collect exactly 3,072 decisions")
    slices = registration.payload["slices"]
    evidence = RCQFinalEvidence(
        registration_sha256=registration.sha256,
        config_sha256=authorization.config_sha256,
        checkpoint_sha256=authorization.checkpoint_sha256,
        checkpoint_step=authorization.checkpoint_step,
        checkpoint_stage_index=authorization.checkpoint_stage_index,
        recipient_manifest_sha256=slices["final_recipient"]["manifest_sha256"],  # type: ignore[index]
        donor_manifest_sha256=slices["final_donor_only"]["manifest_sha256"],  # type: ignore[index]
        donor_mapping_sha256=mapping.sha256,
        invariance_reference=stage_state["invariance_reference"],  # type: ignore[arg-type]
        invariance_current=stage_state["invariance_current"],  # type: ignore[arg-type]
        decisions=tuple(decisions),
    )
    return evidence, mapping


def _movement_mask(keys_down: Sequence[int]) -> int:
    from ..types import HidKey

    keys = set(keys_down)
    return sum(
        1 << index
        for index, key in enumerate((HidKey.W, HidKey.A, HidKey.S, HidKey.D))
        if int(key) in keys
    )


@dataclass(frozen=True, slots=True)
class _PreclaimContext:
    release: Path
    source_root: Path
    config_file: Path
    registration_file: Path
    pretraining_pin: _PinnedDocument
    registration: RCQRegistration
    config: TrainingConfig
    system: object
    entry_system: object
    source: _PreclaimBatchSource
    loaded: object
    entry_loaded: object
    stage_state: Mapping[str, object]
    entry_stage_state: Mapping[str, object]
    run: Path
    latest_pointer: Path
    latest_file_sha256: str
    checkpoint: Path
    entry_checkpoint: Path
    source_digest: str
    runtime_digest: str
    entry_digest: str
    completion_digest: str
    invariance_digest: str
    entry_report: RCQDevelopmentReport
    completion_report: RCQValueDevelopmentReport
    boundary_validation_metrics: Mapping[int, Mapping[str, float]]
    entry_boundary_validation_metrics: Mapping[int, Mapping[str, float]]
    development_replay: object
    entry_development_replay: object
    nonvalue_identity: Mapping[str, object]
    lookup: TrainLookupReceipt
    workspace_protocol: Mapping[str, object]
    claim_registry: Path
    receipt_candidate: Path
    authorization: _FinalAuthorization


def _require_pretraining_pin_matches(
    pin: _PinnedDocument,
    *,
    registration: RCQRegistration,
    config: TrainingConfig,
    source_digest: str,
    evaluator_digest: str,
    batch_source_digest: str,
) -> None:
    payload = pin.payload
    registered = payload["registration"]
    pinned_config = payload["config"]
    runtime = payload["runtime"]
    if not all(
        isinstance(value, Mapping)
        for value in (registered, pinned_config, runtime)
    ):
        raise RCQInputError("pretraining pin lost its validated mapping structure")
    expected = {
        "registration_sha256": registration.sha256,
        "config_raw_sha256": registration.payload["config_raw_sha256"],
        "config_canonical_sha256": config.config_sha256,
        "source_tree_sha256": source_digest,
        "evaluator_bundle_sha256": evaluator_digest,
        "batch_source_manifest_sha256": batch_source_digest,
    }
    observed = {
        "registration_sha256": registered["sha256"],
        "config_raw_sha256": pinned_config["raw_sha256"],
        "config_canonical_sha256": pinned_config["canonical_sha256"],
        "source_tree_sha256": payload["source_tree_sha256"],
        "evaluator_bundle_sha256": payload["evaluator_bundle_sha256"],
        "batch_source_manifest_sha256": payload["batch_source_manifest_sha256"],
    }
    if not _strict_equal(observed, expected):
        raise RCQInputError("pretraining pin differs from the immutable release identities")
    workspace = registration.payload["workspace_protocol"]
    if not isinstance(workspace, Mapping) or (
        runtime["container_release_root"] != workspace["container_release_root"]
        or runtime["container_run_root"] != workspace["container_run_root"]
        or runtime["container_claim_registry_root"]
        != workspace["container_claim_registry_root"]
        or runtime["container_pin_root"] != workspace["container_pin_root"]
    ):
        raise RCQInputError("pretraining pin runtime mounts differ from registration")
    reloaded = _load_pretraining_pin()
    if (
        reloaded.file_sha256 != pin.file_sha256
        or reloaded.semantic_sha256 != pin.semantic_sha256
        or not _strict_equal(reloaded.payload, pin.payload)
    ):
        raise RCQInputError("pretraining pin changed during verification")


def _prepare_trusted_preclaim() -> _PreclaimContext:
    """Run the full trusted pipeline without constructing or claiming TEST."""

    pretraining_pin = _load_pretraining_pin()
    pinned_registration = pretraining_pin.payload["registration"]
    if not isinstance(pinned_registration, Mapping):
        raise RCQInputError("pretraining registration pin is incompatible")
    release, source_root, config_file, registration_file = _resolve_release()
    registration = load_rcq_v2_registration(
        registration_file,
        expected_sha256=str(pinned_registration["sha256"]),
    )
    workspace_protocol = registration.payload["workspace_protocol"]
    if not isinstance(workspace_protocol, Mapping):
        raise RCQInputError("registered workspace protocol is incompatible")
    if (
        release.as_posix() != workspace_protocol["container_release_root"]
        or workspace_protocol["container_pin_root"] != "/workspace/pins"
        or workspace_protocol["pretraining_pin_filename"] != "pretraining.json"
        or workspace_protocol["final_authorization_filename"]
        != "final-authorization.json"
    ):
        raise RCQInputError(
            "final evaluation requires the dedicated canonical DGX release mount"
        )
    config = load_training_config(config_file)
    _require_registered_config(config, config_file, registration)

    from ..training.checkpoint import load_checkpoint
    from ..training.torch_system import TorchTrainingSystem

    source_digest = _source_tree_sha256_exact(source_root)
    if source_digest != registration.payload["source_tree_sha256"]:
        raise RCQInputError("training source tree differs from registration")
    evaluator_digest = evaluator_bundle_sha256(source_root)
    if evaluator_digest != registration.payload["evaluator_bundle_sha256"]:
        raise RCQInputError("RCQ evaluator bundle differs from registration")
    for name, value in config.resource_policy.process_environment().items():
        os.environ[name] = value
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    elif workspace not in {":4096:8", ":16:8"}:
        raise RCQInputError("deterministic CUDA requires a registered CUBLAS workspace")

    config.resource_policy.require(Capability.GPU)
    config.resource_policy.require(Capability.ARTIFACT_WRITE)
    system = TorchTrainingSystem(_objective(config), config)
    entry_system = TorchTrainingSystem(_objective(config), config)
    if not _strict_equal(system.runtime_fingerprint, entry_system.runtime_fingerprint):
        raise RCQInputError("entry and terminal evaluator runtimes differ")
    source = _PreclaimBatchSource(config)
    if source.manifest_sha256 != registration.payload["batch_source_manifest_sha256"]:
        raise RCQInputError("training batch-source manifest differs from registration")
    _require_pretraining_pin_matches(
        pretraining_pin,
        registration=registration,
        config=config,
        source_digest=source_digest,
        evaluator_digest=evaluator_digest,
        batch_source_digest=source.manifest_sha256,
    )
    runtime_digest = _require_runtime_fingerprint(system.runtime_fingerprint)
    run = _plain_resolve(_CONTAINER_RUN_ROOT, name="run_dir", kind="directory")
    if run.as_posix() != workspace_protocol["container_run_root"]:
        raise RCQInputError(
            "final evaluation requires the dedicated canonical DGX run mount"
        )
    checkpoint = _plain_resolve(
        run / "checkpoints" / _EXPECTED_CHECKPOINT_NAME,
        name="checkpoint_path",
        kind="file",
    )
    entry_checkpoint = _plain_resolve(
        run / "checkpoints" / _EXPECTED_ENTRY_CHECKPOINT_NAME,
        name="entry_checkpoint_path",
        kind="file",
    )
    entry_checkpoint_digest = _file_sha256(entry_checkpoint)
    pointer_checkpoint = _checkpoint_digest_from_latest(run, checkpoint)
    latest_pointer = _plain_resolve(
        checkpoint.parent / "latest.json",
        name="canonical latest pointer",
        kind="file",
    )
    latest_file_sha256 = _file_sha256(latest_pointer)
    loaded = load_checkpoint(
        checkpoint,
        expected_config_sha256=config.config_sha256,
        expected_data_sha256=source.manifest_sha256,
        expected_code_sha256=source_digest,
        expected_runtime_fingerprint=system.runtime_fingerprint,
        expected_checkpoint_sha256=pointer_checkpoint,
    )
    entry_loaded = load_checkpoint(
        entry_checkpoint,
        expected_config_sha256=config.config_sha256,
        expected_data_sha256=source.manifest_sha256,
        expected_code_sha256=source_digest,
        expected_runtime_fingerprint=entry_system.runtime_fingerprint,
        expected_checkpoint_sha256=entry_checkpoint_digest,
    )
    if (
        loaded.checkpoint_sha256 != pointer_checkpoint
        or _file_sha256(checkpoint) != pointer_checkpoint
    ):
        raise RCQInputError("checkpoint bytes changed during restricted loading")
    if loaded.stage_state is None or loaded.trainer_state is None:
        raise RCQInputError("final evaluation requires a schema-2 staged checkpoint")
    if (
        entry_loaded.checkpoint_sha256 != entry_checkpoint_digest
        or _file_sha256(entry_checkpoint) != entry_checkpoint_digest
        or entry_loaded.stage_state is None
        or entry_loaded.trainer_state is None
    ):
        raise RCQInputError("entry checkpoint changed or lacks staged provenance")
    batches_per_epoch = source.batches_per_epoch(
        split="train",
        batch_size=config.optimization.batch_size,
    )
    consumed = FINAL_STEP * config.optimization.gradient_accumulation_steps
    expected_cursor = (
        consumed // batches_per_epoch,
        consumed % batches_per_epoch,
        FINAL_STEP,
    )
    observed_cursor = (
        loaded.cursor.epoch,
        loaded.cursor.next_batch,
        loaded.cursor.optimizer_step,
    )
    if observed_cursor != expected_cursor:
        raise RCQInputError("terminal checkpoint sampler/global cursor is incompatible")
    entry_consumed = 1_536 * config.optimization.gradient_accumulation_steps
    expected_entry_cursor = (
        entry_consumed // batches_per_epoch,
        entry_consumed % batches_per_epoch,
        1_536,
    )
    observed_entry_cursor = (
        entry_loaded.cursor.epoch,
        entry_loaded.cursor.next_batch,
        entry_loaded.cursor.optimizer_step,
    )
    if observed_entry_cursor != expected_entry_cursor:
        raise RCQInputError("entry checkpoint sampler/global cursor is incompatible")
    stage_state = loaded.stage_state
    entry_stage_state = entry_loaded.stage_state
    assert entry_stage_state is not None
    if (
        type(stage_state.get("stage_index")) is not int
        or stage_state.get("stage_index") != FINAL_STAGE_INDEX
        or type(stage_state.get("stage_local_optimizer_step")) is not int
        or stage_state.get("stage_local_optimizer_step")
        != config.stages[-1].optimizer_steps
    ):
        raise RCQInputError("terminal checkpoint stage cursor is incompatible")
    if (
        stage_state.get("entry_gate_id") != DEVELOPMENT_GATE_ID
        or stage_state.get("entry_gate_passed") is not True
        or stage_state.get("entry_gate_step") != 1_536
        or stage_state.get("completion_gate_id") != VALUE_DEVELOPMENT_GATE_ID
        or stage_state.get("completion_gate_passed") is not True
        or stage_state.get("completion_gate_step") != FINAL_STEP
    ):
        raise RCQInputError("terminal checkpoint gate provenance is incompatible")
    if (
        type(entry_stage_state.get("stage_index")) is not int
        or entry_stage_state.get("stage_index") != FINAL_STAGE_INDEX
        or type(entry_stage_state.get("stage_local_optimizer_step")) is not int
        or entry_stage_state.get("stage_local_optimizer_step") != 0
        or entry_stage_state.get("entry_gate_id") != DEVELOPMENT_GATE_ID
        or entry_stage_state.get("entry_gate_passed") is not True
        or entry_stage_state.get("entry_gate_step") != 1_536
        or entry_stage_state.get("completion_gate_id")
        != VALUE_DEVELOPMENT_GATE_ID
        or entry_stage_state.get("completion_gate_report_sha256") != "0" * 64
        or entry_stage_state.get("completion_gate_passed") is not None
        or entry_stage_state.get("completion_gate_step") != 0
    ):
        raise RCQInputError("entry checkpoint stage/gate provenance is incompatible")
    boundary_validation_metrics = _validate_metrics_prefix(
        run,
        loaded.trainer_state,
        config=config,
        batches_per_epoch=batches_per_epoch,
        terminal_step=FINAL_STEP,
    )
    entry_boundary_validation_metrics = _validate_metrics_prefix(
        run,
        entry_loaded.trainer_state,
        config=config,
        batches_per_epoch=batches_per_epoch,
        terminal_step=1_536,
    )
    if not _strict_equal(
        entry_boundary_validation_metrics[1_536],
        boundary_validation_metrics[1_536],
    ):
        raise RCQInputError(
            "entry and terminal checkpoints bind different step-1536 metrics"
        )
    if (
        entry_stage_state.get("entry_gate_report_sha256")
        != stage_state.get("entry_gate_report_sha256")
    ):
        raise RCQInputError("entry and terminal checkpoints bind different entry gates")
    entry_digest, entry_report = _validate_entry_gate_artifact(
        run / _EXPECTED_ENTRY_GATE_NAME,
        expected_digest=stage_state["entry_gate_report_sha256"],  # type: ignore[arg-type]
        expected_validation_metrics=boundary_validation_metrics[1_536],
    )
    completion_digest, completion_report = _validate_completion_gate_artifact(
        run / _EXPECTED_COMPLETION_GATE_NAME,
        expected_digest=stage_state["completion_gate_report_sha256"],  # type: ignore[arg-type]
        entry_report=entry_report,
        entry_digest=entry_digest,
        expected_validation_metrics=boundary_validation_metrics[FINAL_STEP],
    )
    invariance_digest = _validate_invariance_artifact(
        run / _EXPECTED_INVARIANCE_NAME,
        config_sha256=config.config_sha256,
        stage_state=stage_state,
    )
    try:
        entry_system.prepare_stage_for_resume(entry_stage_state)
        entry_system.restore_checkpoint_state(entry_loaded.system_state)
        entry_system.restore_rng_state(entry_loaded.rng_state)
        entry_system.objective.train(False)
    except (TypeError, ValueError, RuntimeError) as error:
        raise RCQInputError(
            "entry checkpoint optimizer/scheduler/scaler/RNG state is invalid"
        ) from error
    try:
        system.restore_model_for_evaluation(loaded.system_state, stage_state)
        system.restore_rng_state(loaded.rng_state)
    except (TypeError, ValueError, RuntimeError) as error:
        raise RCQInputError(
            "terminal checkpoint optimizer/scheduler/scaler/RNG state is invalid"
        ) from error
    development_probe_batches = _registered_development_probe_batches(
        iter_batches=source.iter_batches,
        batches_per_epoch=source.batches_per_epoch,
        config=config,
    )
    nonvalue_identity = _require_exact_nonvalue_model_identity(
        entry_system=entry_system,
        terminal_system=system,
        entry_stage_state=entry_stage_state,
        terminal_stage_state=stage_state,
        development_batches=development_probe_batches,
    )
    _set_registered_entry_loss_weights(system=entry_system, config=config)
    entry_development_replay = _replay_registered_development(
        evaluate_batch=entry_system.evaluate_batch,
        iter_batches=source.iter_batches,
        batches_per_epoch=source.batches_per_epoch,
        config=config,
    )
    _require_entry_development_replay(
        replay=entry_development_replay,
        entry_report=entry_report,
        entry_validation_metrics=entry_boundary_validation_metrics[1_536],
    )
    development_replay = _replay_registered_development(
        evaluate_batch=system.evaluate_batch,
        iter_batches=source.iter_batches,
        batches_per_epoch=source.batches_per_epoch,
        config=config,
    )
    _require_development_replay(
        replay=development_replay,
        entry_report=entry_report,
        entry_digest=entry_digest,
        completion_report=completion_report,
        final_validation_metrics=boundary_validation_metrics[FINAL_STEP],
    )
    lookup = fit_rcq_v2_train_lookup(registration)
    receipt_template = registration.payload["receipt_directory"]
    if (
        type(receipt_template) is not str
        or receipt_template != f"final-claims/{_final_range_claim_id()}"
    ):
        raise RCQInputError("registered canonical receipt directory changed")
    claim_registry = _plain_resolve(
        str(workspace_protocol["container_claim_registry_root"]),
        name="canonical final-claim registry mount",
        kind="directory",
    )
    receipt_candidate = claim_registry / _final_range_claim_id()
    _require_empty_claim_slot(claim_registry, _final_range_claim_id())
    receipt_root_digest = _digest_payload(
        b"IRENERCQRECEIPTROOT\x01",
        {
            "workspace_protocol": workspace_protocol,
            "range_claim_id": _final_range_claim_id(),
            "template": receipt_template,
        },
    )
    authorization = _FinalAuthorization(
        config_sha256=config.config_sha256,
        checkpoint_sha256=loaded.checkpoint_sha256,
        checkpoint_step=loaded.cursor.optimizer_step,
        checkpoint_stage_index=stage_state["stage_index"],  # type: ignore[arg-type]
        development_gate_sha256=entry_digest,
        final_development_sha256=completion_digest,
        invariance_report_sha256=invariance_digest,
        source_sha256=source_digest,
        runtime_fingerprint_sha256=runtime_digest,
        receipt_root_sha256=receipt_root_digest,
    )
    if (
        _file_sha256(config_file) != registration.payload["config_raw_sha256"]
        or _file_sha256(registration_file) != registration.sha256
        or _source_tree_sha256_exact(source_root) != source_digest
        or evaluator_bundle_sha256(source_root)
        != registration.payload["evaluator_bundle_sha256"]
        or _file_sha256(checkpoint) != loaded.checkpoint_sha256
        or _file_sha256(entry_checkpoint) != entry_loaded.checkpoint_sha256
        or _file_sha256(run / _EXPECTED_ENTRY_GATE_NAME) != entry_digest
        or _file_sha256(run / _EXPECTED_COMPLETION_GATE_NAME) != completion_digest
        or _file_sha256(run / _EXPECTED_INVARIANCE_NAME) != invariance_digest
        or _file_sha256(latest_pointer) != latest_file_sha256
    ):
        raise RCQInputError("a trusted final input changed during preclaim validation")
    final_pin_check = _load_pretraining_pin()
    if (
        final_pin_check.file_sha256 != pretraining_pin.file_sha256
        or final_pin_check.semantic_sha256 != pretraining_pin.semantic_sha256
        or not _strict_equal(final_pin_check.payload, pretraining_pin.payload)
    ):
        raise RCQInputError("pretraining pin changed during preclaim validation")
    return _PreclaimContext(
        release=release,
        source_root=source_root,
        config_file=config_file,
        registration_file=registration_file,
        pretraining_pin=pretraining_pin,
        registration=registration,
        config=config,
        system=system,
        entry_system=entry_system,
        source=source,
        loaded=loaded,
        entry_loaded=entry_loaded,
        stage_state=stage_state,
        entry_stage_state=entry_stage_state,
        run=run,
        latest_pointer=latest_pointer,
        latest_file_sha256=latest_file_sha256,
        checkpoint=checkpoint,
        entry_checkpoint=entry_checkpoint,
        source_digest=source_digest,
        runtime_digest=runtime_digest,
        entry_digest=entry_digest,
        completion_digest=completion_digest,
        invariance_digest=invariance_digest,
        entry_report=entry_report,
        completion_report=completion_report,
        boundary_validation_metrics=boundary_validation_metrics,
        entry_boundary_validation_metrics=entry_boundary_validation_metrics,
        development_replay=development_replay,
        entry_development_replay=entry_development_replay,
        nonvalue_identity=nonvalue_identity,
        lookup=lookup,
        workspace_protocol=workspace_protocol,
        claim_registry=claim_registry,
        receipt_candidate=receipt_candidate,
        authorization=authorization,
    )


def _readiness_payload(context: _PreclaimContext) -> dict[str, object]:
    from ..training.protocol import TrainingStepResult

    replay = context.development_replay
    entry_replay = context.entry_development_replay
    if not isinstance(replay, TrainingStepResult) or not isinstance(
        entry_replay,
        TrainingStepResult,
    ):
        raise RCQInputError("preclaim DEVELOPMENT replay is incompatible")
    loaded = context.loaded
    entry_loaded = context.entry_loaded
    trainer_state = loaded.trainer_state
    entry_trainer_state = entry_loaded.trainer_state
    if not isinstance(trainer_state, Mapping) or not isinstance(
        entry_trainer_state,
        Mapping,
    ):
        raise RCQInputError("preclaim checkpoint trainer state is incompatible")
    payload: dict[str, object] = {
        "schema_version": 1,
        "action": "rcq_v2_preclaim_v1",
        "qualification_id": context.registration.payload["qualification_id"],
        "evaluator_id": context.registration.payload["evaluator_id"],
        "status": "ready_for_once_only_final",
        "pretraining_pin": {
            "relative_path": "pretraining.json",
            "file_sha256": context.pretraining_pin.file_sha256,
            "pin_sha256": context.pretraining_pin.semantic_sha256,
        },
        "registration_sha256": context.registration.sha256,
        "config_raw_sha256": context.registration.payload["config_raw_sha256"],
        "config_canonical_sha256": context.config.config_sha256,
        "source_tree_sha256": context.source_digest,
        "evaluator_bundle_sha256": context.registration.payload[
            "evaluator_bundle_sha256"
        ],
        "batch_source_manifest_sha256": context.source.manifest_sha256,
        "entry_checkpoint": {
            "relative_path": f"checkpoints/{_EXPECTED_ENTRY_CHECKPOINT_NAME}",
            "sha256": entry_loaded.checkpoint_sha256,
            "optimizer_step": entry_loaded.cursor.optimizer_step,
            "stage_index": context.entry_stage_state["stage_index"],
            "stage_local_optimizer_step": context.entry_stage_state[
                "stage_local_optimizer_step"
            ],
        },
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "latest_file_sha256": context.latest_file_sha256,
        "checkpoint_step": loaded.cursor.optimizer_step,
        "checkpoint_stage_index": context.stage_state["stage_index"],
        "runtime_fingerprint_sha256": context.runtime_digest,
        "entry_gate_report_sha256": context.entry_digest,
        "completion_gate_report_sha256": context.completion_digest,
        "invariance_report_sha256": context.invariance_digest,
        "metrics_prefix": dict(trainer_state),
        "entry_metrics_prefix": dict(entry_trainer_state),
        "train_lookup_sha256": context.lookup.sha256,
        "entry_development_replay": {
            "samples": entry_replay.samples,
            "loss_hex": entry_replay.loss.hex(),
            "metrics_hex": {
                name: value.hex()
                for name, value in sorted(entry_replay.metrics.items())
            },
        },
        "development_replay": {
            "samples": replay.samples,
            "loss_hex": replay.loss.hex(),
            "metrics_hex": {
                name: value.hex() for name, value in sorted(replay.metrics.items())
            },
        },
        "entry_terminal_nonvalue_identity": context.nonvalue_identity,
        "workspace_protocol": context.workspace_protocol,
        "range_claim_id": _final_range_claim_id(),
        "receipt_directory": context.registration.payload["receipt_directory"],
        "range_claim_registry_observed_empty": True,
        "sealed_test_datasets_constructed": 0,
        "sealed_test_examples_opened": 0,
    }
    payload["readiness_sha256"] = _digest_payload(
        b"IRENERCQREADINESS\x01",
        payload,
    )
    return payload


def _readiness_path(context: _PreclaimContext) -> Path:
    relative = context.workspace_protocol["readiness_receipt_relative_path"]
    if type(relative) is not str or relative != (
        "preclaim-readiness/rcq-v2-reference-v2.json"
    ):
        raise RCQInputError("registered preclaim readiness path changed")
    return context.claim_registry / Path(relative)


def _write_rcq_v2_preclaim_readiness() -> tuple[Path, str]:
    """Run safe CUDA readiness and publish one immutable non-TEST receipt."""

    context = _prepare_trusted_preclaim()
    payload = _readiness_payload(context)
    encoded = (_canonical(payload) + "\n").encode("utf-8")
    digest = _digest_bytes(encoded)
    path = _readiness_path(context)
    if path.exists():
        existing = _plain_resolve(path, name="preclaim readiness receipt", kind="file")
        if existing.read_bytes() != encoded or _file_sha256(existing) != digest:
            raise RCQInputError("immutable preclaim readiness receipt already differs")
        return existing, digest
    parent = _safe_receipt_root(path.parent)
    target = parent / path.name
    _publish_no_replace(target, encoded, policy=context.config.resource_policy)
    if _file_sha256(target) != digest:
        raise RCQInputError("published preclaim readiness receipt changed")
    return target, digest


def _require_preclaim_readiness(
    context: _PreclaimContext,
    *,
    expected_readiness_sha256: str,
) -> tuple[Path, str]:
    expected = _hash_string(
        expected_readiness_sha256,
        name="expected_readiness_sha256",
    )
    path = _plain_resolve(
        _readiness_path(context),
        name="preclaim readiness receipt",
        kind="file",
    )
    encoded = (_canonical(_readiness_payload(context)) + "\n").encode("utf-8")
    observed = _file_sha256(path)
    if observed != expected or path.read_bytes() != encoded:
        raise RCQInputError(
            "preclaim readiness receipt differs from the rerun trusted pipeline"
        )
    if _file_sha256(path) != expected:
        raise RCQInputError("preclaim readiness receipt changed during validation")
    return path, expected


def _require_final_authorization_matches(
    context: _PreclaimContext,
    authorization: _PinnedDocument,
    *,
    readiness_path: Path,
    readiness_file_sha256: str,
) -> None:
    payload = authorization.payload
    pretraining = payload["pretraining"]
    latest = payload["latest"]
    entry_checkpoint = payload["entry_checkpoint"]
    checkpoint = payload["checkpoint"]
    readiness = payload["readiness"]
    if not all(
        isinstance(value, Mapping)
        for value in (pretraining, latest, entry_checkpoint, checkpoint, readiness)
    ):
        raise RCQInputError("final authorization lost its validated mapping structure")
    readiness_payload = _readiness_payload(context)
    if not _strict_equal(
        {
            "pretraining_file_sha256": pretraining["file_sha256"],
            "pretraining_pin_sha256": pretraining["pin_sha256"],
            "latest_file_sha256": latest["file_sha256"],
            "latest_checkpoint_sha256": latest["checkpoint_sha256"],
            "entry_checkpoint_sha256": entry_checkpoint["sha256"],
            "checkpoint_sha256": checkpoint["sha256"],
            "readiness_file_sha256": readiness["file_sha256"],
            "readiness_sha256": readiness["readiness_sha256"],
        },
        {
            "pretraining_file_sha256": context.pretraining_pin.file_sha256,
            "pretraining_pin_sha256": context.pretraining_pin.semantic_sha256,
            "latest_file_sha256": context.latest_file_sha256,
            "latest_checkpoint_sha256": context.loaded.checkpoint_sha256,
            "entry_checkpoint_sha256": context.entry_loaded.checkpoint_sha256,
            "checkpoint_sha256": context.loaded.checkpoint_sha256,
            "readiness_file_sha256": readiness_file_sha256,
            "readiness_sha256": readiness_payload["readiness_sha256"],
        },
    ):
        raise RCQInputError("final authorization differs from canonical preclaim evidence")
    if (
        readiness_path != _readiness_path(context)
        or _file_sha256(context.latest_pointer) != context.latest_file_sha256
        or _file_sha256(context.entry_checkpoint)
        != context.entry_loaded.checkpoint_sha256
        or _file_sha256(context.checkpoint) != context.loaded.checkpoint_sha256
        or _file_sha256(readiness_path) != readiness_file_sha256
    ):
        raise RCQInputError("a final-authorized artifact changed")
    reloaded = _load_final_authorization()
    if (
        reloaded.file_sha256 != authorization.file_sha256
        or reloaded.semantic_sha256 != authorization.semantic_sha256
        or not _strict_equal(reloaded.payload, authorization.payload)
    ):
        raise RCQInputError("final authorization changed during verification")


def _load_readiness_receipt_for_verification(
    path: Path,
) -> tuple[dict[str, object], str, str]:
    raw, file_digest = _strict_json_file(path, name="preclaim readiness receipt")
    fields = {
        "schema_version",
        "action",
        "qualification_id",
        "evaluator_id",
        "status",
        "pretraining_pin",
        "registration_sha256",
        "config_raw_sha256",
        "config_canonical_sha256",
        "source_tree_sha256",
        "evaluator_bundle_sha256",
        "batch_source_manifest_sha256",
        "entry_checkpoint",
        "checkpoint_sha256",
        "latest_file_sha256",
        "checkpoint_step",
        "checkpoint_stage_index",
        "runtime_fingerprint_sha256",
        "entry_gate_report_sha256",
        "completion_gate_report_sha256",
        "invariance_report_sha256",
        "metrics_prefix",
        "entry_metrics_prefix",
        "train_lookup_sha256",
        "entry_development_replay",
        "development_replay",
        "entry_terminal_nonvalue_identity",
        "workspace_protocol",
        "range_claim_id",
        "receipt_directory",
        "range_claim_registry_observed_empty",
        "sealed_test_datasets_constructed",
        "sealed_test_examples_opened",
        "readiness_sha256",
    }
    readiness = _exact_dict(raw, fields, name="preclaim readiness receipt")
    if (
        type(readiness["schema_version"]) is not int
        or readiness["schema_version"] != 1
        or readiness["action"] != "rcq_v2_preclaim_v1"
        or readiness["qualification_id"] != "rcq_v2_reference_v2"
        or readiness["evaluator_id"] != "rcq_v2_final_v1"
        or readiness["status"] != "ready_for_once_only_final"
        or readiness["checkpoint_step"] != FINAL_STEP
        or type(readiness["checkpoint_step"]) is not int
        or readiness["checkpoint_stage_index"] != FINAL_STAGE_INDEX
        or type(readiness["checkpoint_stage_index"]) is not int
        or readiness["range_claim_id"] != _final_range_claim_id()
        or readiness["receipt_directory"]
        != f"final-claims/{_final_range_claim_id()}"
        or readiness["range_claim_registry_observed_empty"] is not True
        or readiness["sealed_test_datasets_constructed"] != 0
        or type(readiness["sealed_test_datasets_constructed"]) is not int
        or readiness["sealed_test_examples_opened"] != 0
        or type(readiness["sealed_test_examples_opened"]) is not int
    ):
        raise RCQInputError("preclaim readiness receipt identity changed")
    for field in (
        "registration_sha256",
        "config_raw_sha256",
        "config_canonical_sha256",
        "source_tree_sha256",
        "evaluator_bundle_sha256",
        "batch_source_manifest_sha256",
        "checkpoint_sha256",
        "latest_file_sha256",
        "runtime_fingerprint_sha256",
        "entry_gate_report_sha256",
        "completion_gate_report_sha256",
        "invariance_report_sha256",
        "train_lookup_sha256",
    ):
        _hash_string(readiness[field], name=f"readiness {field}")
    entry = _exact_dict(
        readiness["entry_checkpoint"],
        {
            "relative_path",
            "sha256",
            "optimizer_step",
            "stage_index",
            "stage_local_optimizer_step",
        },
        name="readiness entry checkpoint",
    )
    if not _strict_equal(
        {
            "relative_path": entry["relative_path"],
            "optimizer_step": entry["optimizer_step"],
            "stage_index": entry["stage_index"],
            "stage_local_optimizer_step": entry["stage_local_optimizer_step"],
        },
        {
            "relative_path": f"checkpoints/{_EXPECTED_ENTRY_CHECKPOINT_NAME}",
            "optimizer_step": 1_536,
            "stage_index": FINAL_STAGE_INDEX,
            "stage_local_optimizer_step": 0,
        },
    ):
        raise RCQInputError("readiness entry checkpoint identity changed")
    _hash_string(entry["sha256"], name="readiness entry checkpoint sha256")
    semantic = _hash_string(
        readiness["readiness_sha256"],
        name="preclaim readiness semantic sha256",
    )
    body = dict(readiness)
    body.pop("readiness_sha256")
    if semantic != _digest_payload(b"IRENERCQREADINESS\x01", body):
        raise RCQInputError("preclaim readiness semantic digest changed")
    return readiness, file_digest, semantic


def _prepare_receipt_verification_bindings() -> Mapping[str, object]:
    """Recompute every non-TEST identity referenced by a terminal receipt."""

    pretraining_pin = _load_pretraining_pin()
    final_authorization = _load_final_authorization()
    release, source_root, config_file, registration_file = _resolve_release()
    pinned_registration = pretraining_pin.payload["registration"]
    if not isinstance(pinned_registration, Mapping):
        raise RCQInputError("pretraining registration pin is incompatible")
    registration = load_rcq_v2_registration(
        registration_file,
        expected_sha256=str(pinned_registration["sha256"]),
    )
    config = load_training_config(config_file)
    _require_registered_config(config, config_file, registration)
    source_digest = _source_tree_sha256_exact(source_root)
    evaluator_digest = evaluator_bundle_sha256(source_root)
    source = _PreclaimBatchSource(config)
    if (
        source_digest != registration.payload["source_tree_sha256"]
        or evaluator_digest != registration.payload["evaluator_bundle_sha256"]
        or source.manifest_sha256
        != registration.payload["batch_source_manifest_sha256"]
    ):
        raise RCQInputError("receipt verifier release differs from registration")
    _require_pretraining_pin_matches(
        pretraining_pin,
        registration=registration,
        config=config,
        source_digest=source_digest,
        evaluator_digest=evaluator_digest,
        batch_source_digest=source.manifest_sha256,
    )
    workspace = registration.payload["workspace_protocol"]
    if not isinstance(workspace, Mapping) or release.as_posix() != workspace.get(
        "container_release_root"
    ):
        raise RCQInputError("receipt verifier release mount differs from registration")
    run = _plain_resolve(_CONTAINER_RUN_ROOT, name="run_dir", kind="directory")
    claim_registry = _plain_resolve(
        _CONTAINER_CLAIM_REGISTRY_ROOT,
        name="canonical final-claim registry mount",
        kind="directory",
    )
    terminal_checkpoint = _plain_resolve(
        run / "checkpoints" / _EXPECTED_CHECKPOINT_NAME,
        name="terminal checkpoint",
        kind="file",
    )
    entry_checkpoint = _plain_resolve(
        run / "checkpoints" / _EXPECTED_ENTRY_CHECKPOINT_NAME,
        name="entry checkpoint",
        kind="file",
    )
    latest_path = _plain_resolve(
        run / "checkpoints" / "latest.json",
        name="canonical latest pointer",
        kind="file",
    )
    terminal_sha = _checkpoint_digest_from_latest(run, terminal_checkpoint)
    entry_sha = _file_sha256(entry_checkpoint)
    latest_sha = _file_sha256(latest_path)
    authorization_payload = final_authorization.payload
    auth_pretraining = authorization_payload["pretraining"]
    auth_latest = authorization_payload["latest"]
    auth_entry = authorization_payload["entry_checkpoint"]
    auth_checkpoint = authorization_payload["checkpoint"]
    auth_readiness = authorization_payload["readiness"]
    if not all(
        isinstance(item, Mapping)
        for item in (
            auth_pretraining,
            auth_latest,
            auth_entry,
            auth_checkpoint,
            auth_readiness,
        )
    ):
        raise RCQInputError("final authorization links are incompatible")
    readiness_path = _plain_resolve(
        claim_registry / "preclaim-readiness" / "rcq-v2-reference-v2.json",
        name="preclaim readiness receipt",
        kind="file",
    )
    readiness, readiness_file_sha, readiness_semantic = (
        _load_readiness_receipt_for_verification(readiness_path)
    )
    if not _strict_equal(
        {
            "pretraining_file_sha256": auth_pretraining["file_sha256"],
            "pretraining_pin_sha256": auth_pretraining["pin_sha256"],
            "latest_file_sha256": auth_latest["file_sha256"],
            "latest_checkpoint_sha256": auth_latest["checkpoint_sha256"],
            "entry_checkpoint_sha256": auth_entry["sha256"],
            "checkpoint_sha256": auth_checkpoint["sha256"],
            "readiness_file_sha256": auth_readiness["file_sha256"],
            "readiness_sha256": auth_readiness["readiness_sha256"],
        },
        {
            "pretraining_file_sha256": pretraining_pin.file_sha256,
            "pretraining_pin_sha256": pretraining_pin.semantic_sha256,
            "latest_file_sha256": latest_sha,
            "latest_checkpoint_sha256": terminal_sha,
            "entry_checkpoint_sha256": entry_sha,
            "checkpoint_sha256": terminal_sha,
            "readiness_file_sha256": readiness_file_sha,
            "readiness_sha256": readiness_semantic,
        },
    ):
        raise RCQInputError("final authorization differs from current immutable artifacts")
    entry_readiness = readiness["entry_checkpoint"]
    readiness_pretraining = readiness["pretraining_pin"]
    if not isinstance(entry_readiness, Mapping) or not isinstance(
        readiness_pretraining,
        Mapping,
    ):
        raise RCQInputError("preclaim readiness links are incompatible")
    terminal_trainer_state = readiness["metrics_prefix"]
    entry_trainer_state = readiness["entry_metrics_prefix"]
    if not isinstance(terminal_trainer_state, Mapping) or not isinstance(
        entry_trainer_state,
        Mapping,
    ):
        raise RCQInputError("readiness metrics prefixes are incompatible")
    batches_per_epoch = source.batches_per_epoch(
        split="train",
        batch_size=config.optimization.batch_size,
    )
    terminal_boundaries = _validate_metrics_prefix(
        run,
        terminal_trainer_state,
        config=config,
        batches_per_epoch=batches_per_epoch,
        terminal_step=FINAL_STEP,
    )
    entry_boundaries = _validate_metrics_prefix(
        run,
        entry_trainer_state,
        config=config,
        batches_per_epoch=batches_per_epoch,
        terminal_step=1_536,
    )
    if not _strict_equal(entry_boundaries[1_536], terminal_boundaries[1_536]):
        raise RCQInputError("receipt verifier found divergent entry metrics prefixes")
    entry_gate_path = _plain_resolve(
        run / _EXPECTED_ENTRY_GATE_NAME,
        name="entry development gate",
        kind="file",
    )
    completion_gate_path = _plain_resolve(
        run / _EXPECTED_COMPLETION_GATE_NAME,
        name="completion development gate",
        kind="file",
    )
    invariance_path = _plain_resolve(
        run / _EXPECTED_INVARIANCE_NAME,
        name="final invariance report",
        kind="file",
    )
    entry_gate_sha, entry_report = _validate_entry_gate_artifact(
        entry_gate_path,
        expected_digest=str(readiness["entry_gate_report_sha256"]),
        expected_validation_metrics=entry_boundaries[1_536],
    )
    completion_gate_sha, _completion_report = _validate_completion_gate_artifact(
        completion_gate_path,
        expected_digest=str(readiness["completion_gate_report_sha256"]),
        entry_report=entry_report,
        entry_digest=entry_gate_sha,
        expected_validation_metrics=terminal_boundaries[FINAL_STEP],
    )
    lookup = fit_rcq_v2_train_lookup(registration)
    invariance_sha = _file_sha256(invariance_path)
    if not _strict_equal(
        {
            "pretraining_pin": dict(readiness_pretraining),
            "registration_sha256": readiness["registration_sha256"],
            "config_raw_sha256": readiness["config_raw_sha256"],
            "config_canonical_sha256": readiness["config_canonical_sha256"],
            "source_tree_sha256": readiness["source_tree_sha256"],
            "evaluator_bundle_sha256": readiness["evaluator_bundle_sha256"],
            "batch_source_manifest_sha256": readiness[
                "batch_source_manifest_sha256"
            ],
            "entry_checkpoint_sha256": entry_readiness["sha256"],
            "checkpoint_sha256": readiness["checkpoint_sha256"],
            "latest_file_sha256": readiness["latest_file_sha256"],
            "entry_gate_report_sha256": readiness["entry_gate_report_sha256"],
            "completion_gate_report_sha256": readiness[
                "completion_gate_report_sha256"
            ],
            "invariance_report_sha256": readiness["invariance_report_sha256"],
            "train_lookup_sha256": readiness["train_lookup_sha256"],
        },
        {
            "pretraining_pin": {
                "relative_path": "pretraining.json",
                "file_sha256": pretraining_pin.file_sha256,
                "pin_sha256": pretraining_pin.semantic_sha256,
            },
            "registration_sha256": registration.sha256,
            "config_raw_sha256": registration.payload["config_raw_sha256"],
            "config_canonical_sha256": config.config_sha256,
            "source_tree_sha256": source_digest,
            "evaluator_bundle_sha256": evaluator_digest,
            "batch_source_manifest_sha256": source.manifest_sha256,
            "entry_checkpoint_sha256": entry_sha,
            "checkpoint_sha256": terminal_sha,
            "latest_file_sha256": latest_sha,
            "entry_gate_report_sha256": entry_gate_sha,
            "completion_gate_report_sha256": completion_gate_sha,
            "invariance_report_sha256": invariance_sha,
            "train_lookup_sha256": lookup.sha256,
        },
    ):
        raise RCQInputError("preclaim readiness differs from current immutable artifacts")
    if _canonical(readiness["workspace_protocol"]) != _canonical(workspace):
        raise RCQInputError("preclaim readiness workspace protocol changed")
    receipt_root_sha = _digest_payload(
        b"IRENERCQRECEIPTROOT\x01",
        {
            "workspace_protocol": workspace,
            "range_claim_id": _final_range_claim_id(),
            "template": registration.payload["receipt_directory"],
        },
    )
    expected_authorization = {
        "schema_version": 1,
        "registration_sha256": registration.sha256,
        "config_sha256": config.config_sha256,
        "checkpoint_sha256": terminal_sha,
        "checkpoint_step": FINAL_STEP,
        "checkpoint_stage_index": FINAL_STAGE_INDEX,
        "development_gate_sha256": entry_gate_sha,
        "final_development_sha256": completion_gate_sha,
        "invariance_report_sha256": invariance_sha,
        "train_lookup_sha256": lookup.sha256,
        "source_sha256": source_digest,
        "runtime_fingerprint_sha256": readiness["runtime_fingerprint_sha256"],
        "receipt_root_sha256": receipt_root_sha,
        "pretraining_pin": {
            "relative_path": "pretraining.json",
            "file_sha256": pretraining_pin.file_sha256,
            "pin_sha256": pretraining_pin.semantic_sha256,
        },
        "entry_checkpoint": {
            **dict(entry_readiness),
            "nonvalue_identity": readiness["entry_terminal_nonvalue_identity"],
        },
        "final_authorization": {
            "relative_path": "final-authorization.json",
            "file_sha256": final_authorization.file_sha256,
            "authorization_sha256": final_authorization.semantic_sha256,
        },
    }
    return _deep_freeze(
        {
            "qualification_id": registration.payload["qualification_id"],
            "claim_registry": str(claim_registry),
            "readiness_binding": {
                "file": readiness_path.name,
                "file_sha256": readiness_file_sha,
                "readiness_sha256": readiness_semantic,
            },
            "authorization": expected_authorization,
            "retired_ranges": [
                {
                    "role": "final_recipient",
                    "split": "test",
                    "local_start": RECIPIENT_OFFSET,
                    "local_end": RECIPIENT_OFFSET + RECIPIENT_SEQUENCES,
                },
                {
                    "role": "final_donor_only",
                    "split": "test",
                    "local_start": DONOR_OFFSET,
                    "local_end": DONOR_OFFSET + DONOR_SEQUENCES,
                },
            ],
            "guard_band": registration.payload["guard_band"],
            "future_campaign_offset": FUTURE_CAMPAIGN_OFFSET,
        }
    )


def _canonical_store_json(
    store: _BoundReceiptDirectory,
    name: str,
) -> tuple[dict[str, object], bytes]:
    encoded = store.read_bytes(name)
    raw = _strict_json_bytes(encoded, name=name, canonical_line=True)
    if type(raw) is not dict:
        raise RCQInputError(f"{name} must contain a JSON object")
    return raw, encoded


def _verify_published_reference(
    store: _BoundReceiptDirectory,
    reference: object,
    *,
    expected_name: str,
    jsonl: bool = True,
) -> bool:
    if type(jsonl) is not bool:
        raise RCQInputError("published-reference format selector must be a boolean")
    if type(reference) is not dict or set(reference) == {"status"}:
        if type(reference) is not dict or reference.get("status") != "not_published":
            raise RCQInputError(f"{expected_name} publication reference is malformed")
        return False
    required = {"status", "file", "records", "sha256"}
    if not required <= set(reference) or reference.get("status") != "published_verified":
        raise RCQInputError(f"{expected_name} publication reference is malformed")
    if reference.get("file") != expected_name:
        raise RCQInputError(f"{expected_name} publication path changed")
    records = reference.get("records")
    digest = _hash_string(reference.get("sha256"), name=f"{expected_name} sha256")
    if type(records) is not int or records < 0:
        raise RCQInputError(f"{expected_name} record count is invalid")
    encoded = store.read_bytes(expected_name)
    if _digest_bytes(encoded) != digest:
        raise RCQInputError(f"{expected_name} differs from its terminal receipt hash")
    if jsonl:
        lines = encoded.splitlines(keepends=True)
        if len(lines) != records or any(not line.endswith(b"\n") for line in lines):
            raise RCQInputError(f"{expected_name} record count changed")
        for line in lines:
            _strict_json_bytes(line, name=f"{expected_name} record", canonical_line=True)
    return True


def _validate_terminal_receipt_commit(
    store: _BoundReceiptDirectory,
    bindings: Mapping[str, object],
) -> RCQFinalReceipt:
    """Validate the canonical receipt as the sole terminal commit record."""

    claim, _claim_bytes = _canonical_store_json(store, _CLAIM_NAME)
    receipt, _receipt_bytes = _canonical_store_json(store, _RECEIPT_NAME)
    claim_fields = {
        "schema_version",
        "qualification_id",
        "status",
        "preclaim_readiness",
        "authorization",
        "retired_ranges",
        "guard_band",
        "future_campaign_offset",
        "reserved_excluded_ranges",
        "retry_permitted",
        "claim_sha256",
    }
    _exact_dict(claim, claim_fields, name="canonical final claim")
    claim_semantic = _hash_string(claim["claim_sha256"], name="claim sha256")
    claim_body = dict(claim)
    claim_body.pop("claim_sha256")
    if (
        type(claim["schema_version"]) is not int
        or claim["schema_version"] != 1
        or claim_semantic != _digest_payload(b"IRENERCQCLAIM\x01", claim_body)
        or claim["status"] != "claimed_test_retired"
        or claim["retry_permitted"] is not False
        or _canonical(claim["reserved_excluded_ranges"])
        != _canonical(_reserved_excluded_ranges())
    ):
        raise RCQInputError("canonical final claim is malformed")
    common_receipt_fields = {
        "schema_version",
        "qualification_id",
        "status",
        "passed",
        "claim_sha256",
        "preclaim_readiness",
        "donor_assignment_cost",
        "authorization",
        "retired_ranges",
        "guard_band",
        "future_campaign_offset",
        "range_ledger",
        "retry_permitted",
        "receipt_sha256",
    }
    allowed_receipt_fields = common_receipt_fields | {
        "report",
        "raw_ledgers",
        "error_type",
        "error",
    }
    if not common_receipt_fields <= set(receipt) or not set(receipt) <= allowed_receipt_fields:
        raise RCQInputError("canonical terminal receipt schema changed")
    receipt_semantic = _hash_string(receipt["receipt_sha256"], name="receipt sha256")
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_sha256")
    if receipt_semantic != _digest_payload(b"IRENERCQRECEIPT\x01", receipt_body):
        raise RCQInputError("canonical terminal receipt self-digest changed")
    status = receipt["status"]
    passed = receipt["passed"]
    if (
        type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
        or type(status) is not str
        or status not in {"passed", "scientific_failed", "invalid_after_claim"}
        or type(passed) is not bool
        or passed is not (status == "passed")
        or receipt["retry_permitted"] is not False
    ):
        raise RCQInputError("canonical terminal receipt status is malformed")
    for name in (
        "qualification_id",
        "preclaim_readiness",
        "authorization",
        "retired_ranges",
        "guard_band",
        "future_campaign_offset",
    ):
        if _canonical(receipt[name]) != _canonical(claim[name]):
            raise RCQInputError(f"terminal receipt differs from claim field {name!r}")
    if receipt["claim_sha256"] != claim_semantic:
        raise RCQInputError("terminal receipt differs from canonical claim digest")
    expected = {
        "qualification_id": bindings["qualification_id"],
        "preclaim_readiness": bindings["readiness_binding"],
        "authorization": bindings["authorization"],
        "retired_ranges": bindings["retired_ranges"],
        "guard_band": bindings["guard_band"],
        "future_campaign_offset": bindings["future_campaign_offset"],
    }
    observed = {name: receipt[name] for name in expected}
    if _canonical(observed) != _canonical(expected):
        raise RCQInputError("terminal receipt differs from recomputed immutable bindings")
    range_reference = receipt["range_ledger"]
    range_published = _verify_published_reference(
        store,
        range_reference,
        expected_name=_RANGE_LEDGER_NAME,
        jsonl=False,
    )
    if range_published:
        assert isinstance(range_reference, dict)
        range_payload, _range_bytes = _canonical_store_json(store, _RANGE_LEDGER_NAME)
        if set(range_payload) != {
            "schema_version",
            "qualification_id",
            "claim_sha256",
            "ranges",
            "ledger_sha256",
        }:
            raise RCQInputError("range ledger schema changed")
        ledger_semantic = _hash_string(
            range_payload["ledger_sha256"],
            name="range ledger semantic sha256",
        )
        range_body = dict(range_payload)
        range_body.pop("ledger_sha256")
        if (
            type(range_payload["schema_version"]) is not int
            or range_payload["schema_version"] != 1
            or ledger_semantic != _digest_payload(b"IRENERCQRANGES\x01", range_body)
            or range_reference.get("ledger_sha256") != ledger_semantic
            or range_reference.get("records") != len(_reserved_excluded_ranges())
            or range_payload["claim_sha256"] != claim_semantic
            or _canonical(range_payload["ranges"])
            != _canonical(_reserved_excluded_ranges())
        ):
            raise RCQInputError("range ledger differs from its receipt or claim")
    ledgers = receipt.get("raw_ledgers")
    if status in {"passed", "scientific_failed"}:
        report = receipt.get("report")
        if type(report) is not dict or report.get("passed") is not passed:
            raise RCQInputError("scientific terminal receipt report is malformed")
        ledgers = report.get("raw_ledgers")
    elif (
        type(receipt.get("error_type")) is not str
        or type(receipt.get("error")) is not str
        or (("report" in receipt) == ("raw_ledgers" in receipt))
    ):
        raise RCQInputError("invalid-after-claim receipt evidence is malformed")
    elif "report" in receipt:
        report = receipt["report"]
        if type(report) is not dict:
            raise RCQInputError("invalid-after-claim retained report is malformed")
        ledgers = report.get("raw_ledgers")
    if type(ledgers) is not dict or set(ledgers) != {"decisions", "donors"}:
        raise RCQInputError("terminal receipt raw-ledger references are malformed")
    decisions_published = _verify_published_reference(
        store,
        ledgers["decisions"],
        expected_name=_DECISION_LEDGER_NAME,
    )
    donors_published = _verify_published_reference(
        store,
        ledgers["donors"],
        expected_name=_DONOR_LEDGER_NAME,
    )
    if donors_published and not decisions_published:
        raise RCQInputError("donor ledger cannot precede the decision ledger")
    if decisions_published and not range_published:
        raise RCQInputError("decision ledger cannot precede the range ledger")
    if status in {"passed", "scientific_failed"} and not (
        range_published and decisions_published and donors_published
    ):
        raise RCQInputError("scientific terminal receipt lacks complete raw evidence")
    expected_entries = {_CLAIM_NAME, _RECEIPT_NAME}
    if range_published:
        expected_entries.add(_RANGE_LEDGER_NAME)
    if decisions_published:
        expected_entries.add(_DECISION_LEDGER_NAME)
    if donors_published:
        expected_entries.add(_DONOR_LEDGER_NAME)
    store.require_entries(expected_entries)
    return RCQFinalReceipt(payload=receipt)


def _verify_rcq_v2_final_receipt() -> Mapping[str, object]:
    """Read-only verification; never creates a claim or constructs TEST."""

    claim_registry = _plain_resolve(
        _CONTAINER_CLAIM_REGISTRY_ROOT,
        name="canonical final-claim registry mount",
        kind="directory",
    )
    store: _BoundReceiptDirectory | None = None
    try:
        store = _BoundReceiptDirectory(
            claim_registry,
            _final_range_claim_id(),
            require_empty=False,
        )
        entries = set(os.listdir(store._root_descriptor))
        if not entries:
            return _deep_freeze(
                {
                    "action": "rcq_v2_verify_receipt_v1",
                    "verification_status": "not_claimed",
                    "terminal_status": None,
                    "passed": False,
                    "retry_permitted": True,
                    "sealed_test_examples_opened_by_this_verifier": 0,
                }
            )
        if _CLAIM_NAME not in entries:
            raise RCQInputError("occupied final range slot lacks its canonical claim")
        bindings = _prepare_receipt_verification_bindings()
        receipt = _validate_terminal_receipt_commit(store, bindings)
        return _deep_freeze(
            {
                "action": "rcq_v2_verify_receipt_v1",
                "verification_status": "authoritative_terminal_receipt",
                "terminal_status": receipt.status,
                "passed": receipt.passed,
                "receipt_sha256": receipt.payload["receipt_sha256"],
                "retry_permitted": False,
                "sealed_test_examples_opened_by_this_verifier": 0,
            }
        )
    except _ReceiptSlotAbsent:
        return _deep_freeze(
            {
                "action": "rcq_v2_verify_receipt_v1",
                "verification_status": "not_claimed",
                "terminal_status": None,
                "passed": False,
                "retry_permitted": True,
                "sealed_test_examples_opened_by_this_verifier": 0,
            }
        )
    except (OSError, RCQInputError, TypeError, ValueError) as error:
        return _deep_freeze(
            {
                "action": "rcq_v2_verify_receipt_v1",
                "verification_status": "terminal_invalid_retired",
                "terminal_status": "invalid_or_absent_receipt_after_claim",
                "passed": False,
                "retry_permitted": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "sealed_test_examples_opened_by_this_verifier": 0,
            }
        )
    finally:
        if store is not None:
            store.close()


def _evaluate_rcq_v2_checkpoint_final_once() -> RCQFinalReceipt:
    """Validate one terminal run, claim fresh TEST, evaluate, and receipt once."""

    context = _prepare_trusted_preclaim()
    final_authorization = _load_final_authorization()
    authorized_readiness = final_authorization.payload["readiness"]
    if not isinstance(authorized_readiness, Mapping):
        raise RCQInputError("authorized readiness link is incompatible")
    readiness_path, readiness_digest = _require_preclaim_readiness(
        context,
        expected_readiness_sha256=str(authorized_readiness["file_sha256"]),
    )
    _require_final_authorization_matches(
        context,
        final_authorization,
        readiness_path=readiness_path,
        readiness_file_sha256=readiness_digest,
    )
    registration = context.registration
    config = context.config
    system = context.system
    loaded = context.loaded
    stage_state = context.stage_state
    lookup = context.lookup
    run = context.run
    checkpoint = context.checkpoint
    entry_checkpoint = context.entry_checkpoint
    source_root = context.source_root
    config_file = context.config_file
    registration_file = context.registration_file
    source_digest = context.source_digest
    runtime_digest = context.runtime_digest
    entry_digest = context.entry_digest
    completion_digest = context.completion_digest
    invariance_digest = context.invariance_digest
    workspace_protocol = context.workspace_protocol
    authorization = context.authorization

    import torch

    def autocast_context() -> object:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    config.resource_policy.require(Capability.GPU)
    config.resource_policy.require(Capability.ARTIFACT_WRITE)
    policy = config.resource_policy
    receipt_template = registration.payload["receipt_directory"]
    readiness_payload = _readiness_payload(context)
    readiness_binding = {
        "file": readiness_path.name,
        "file_sha256": readiness_digest,
        "readiness_sha256": readiness_payload["readiness_sha256"],
    }
    authorization_payload = _authorization_payload(
        authorization,
        registration=registration,
        lookup=lookup,
    )
    authorization_payload["pretraining_pin"] = {
        "relative_path": "pretraining.json",
        "file_sha256": context.pretraining_pin.file_sha256,
        "pin_sha256": context.pretraining_pin.semantic_sha256,
    }
    authorization_payload["entry_checkpoint"] = {
        "relative_path": f"checkpoints/{_EXPECTED_ENTRY_CHECKPOINT_NAME}",
        "sha256": context.entry_loaded.checkpoint_sha256,
        "optimizer_step": context.entry_loaded.cursor.optimizer_step,
        "stage_index": context.entry_stage_state["stage_index"],
        "stage_local_optimizer_step": context.entry_stage_state[
            "stage_local_optimizer_step"
        ],
        "nonvalue_identity": context.nonvalue_identity,
    }
    authorization_payload["final_authorization"] = {
        "relative_path": "final-authorization.json",
        "file_sha256": final_authorization.file_sha256,
        "authorization_sha256": final_authorization.semantic_sha256,
    }
    _require_final_authorization_matches(
        context,
        final_authorization,
        readiness_path=readiness_path,
        readiness_file_sha256=readiness_digest,
    )
    pretraining_check = _load_pretraining_pin()
    if (
        _file_sha256(config_file) != registration.payload["config_raw_sha256"]
        or _file_sha256(registration_file) != registration.sha256
        or _source_tree_sha256_exact(source_root) != source_digest
        or evaluator_bundle_sha256(source_root)
        != registration.payload["evaluator_bundle_sha256"]
        or _file_sha256(checkpoint) != loaded.checkpoint_sha256
        or _file_sha256(entry_checkpoint) != context.entry_loaded.checkpoint_sha256
        or _file_sha256(run / _EXPECTED_ENTRY_GATE_NAME) != entry_digest
        or _file_sha256(run / _EXPECTED_COMPLETION_GATE_NAME) != completion_digest
        or _file_sha256(run / _EXPECTED_INVARIANCE_NAME) != invariance_digest
        or _file_sha256(readiness_path) != readiness_digest
        or _file_sha256(context.latest_pointer) != context.latest_file_sha256
        or pretraining_check.file_sha256 != context.pretraining_pin.file_sha256
        or pretraining_check.semantic_sha256
        != context.pretraining_pin.semantic_sha256
        or not _strict_equal(
            pretraining_check.payload,
            context.pretraining_pin.payload,
        )
    ):
        raise RCQInputError("a trusted final input changed before claim publication")
    receipt_store = _BoundReceiptDirectory(
        context.claim_registry,
        _final_range_claim_id(),
    )
    root = receipt_store.path
    if _digest_payload(
        b"IRENERCQRECEIPTROOT\x01",
        {
            "workspace_protocol": workspace_protocol,
            "range_claim_id": root.name,
            "template": receipt_template,
        },
    ) != authorization.receipt_root_sha256:
        receipt_store.close()
        raise RCQInputError("canonical receipt root changed during validation")
    final_pin_check = _load_final_authorization()
    pretraining_pin_check = _load_pretraining_pin()
    if (
        final_pin_check.file_sha256 != final_authorization.file_sha256
        or final_pin_check.semantic_sha256 != final_authorization.semantic_sha256
        or not _strict_equal(final_pin_check.payload, final_authorization.payload)
        or pretraining_pin_check.file_sha256
        != context.pretraining_pin.file_sha256
        or pretraining_pin_check.semantic_sha256
        != context.pretraining_pin.semantic_sha256
        or not _strict_equal(
            pretraining_pin_check.payload,
            context.pretraining_pin.payload,
        )
        or _file_sha256(entry_checkpoint) != context.entry_loaded.checkpoint_sha256
        or _file_sha256(checkpoint) != loaded.checkpoint_sha256
    ):
        receipt_store.close()
        raise RCQInputError("qualification pins changed immediately before claim")
    published_names: set[str] = set()
    retired_ranges = [
        {
            "role": "final_recipient",
            "split": "test",
            "local_start": RECIPIENT_OFFSET,
            "local_end": RECIPIENT_OFFSET + RECIPIENT_SEQUENCES,
        },
        {
            "role": "final_donor_only",
            "split": "test",
            "local_start": DONOR_OFFSET,
            "local_end": DONOR_OFFSET + DONOR_SEQUENCES,
        },
    ]
    claim_payload: dict[str, object] = {
        "schema_version": 1,
        "qualification_id": registration.payload["qualification_id"],
        "status": "claimed_test_retired",
        "preclaim_readiness": readiness_binding,
        "authorization": authorization_payload,
        "retired_ranges": retired_ranges,
        "guard_band": registration.payload["guard_band"],
        "future_campaign_offset": FUTURE_CAMPAIGN_OFFSET,
        "reserved_excluded_ranges": _reserved_excluded_ranges(),
        "retry_permitted": False,
    }
    claim_payload["claim_sha256"] = _digest_payload(
        b"IRENERCQCLAIM\x01",
        claim_payload,
    )
    claim_bytes = (_canonical(claim_payload) + "\n").encode("utf-8")
    receipt_store.publish(
        _CLAIM_NAME,
        claim_bytes,
        existing=published_names,
        policy=policy,
    )
    published_names.add(_CLAIM_NAME)
    range_file: dict[str, object] = {"status": "not_published"}
    raw_ledgers: dict[str, object] = {
        "decisions": {"status": "not_published"},
        "donors": {"status": "not_published"},
    }
    donor_assignment_cost: dict[str, object] = {"status": "not_available"}
    try:
        receipt_store.require_exact(_CLAIM_NAME, claim_bytes)
        receipt_store.require_entries(published_names)
        range_payload: dict[str, object] = {
            "schema_version": 1,
            "qualification_id": registration.payload["qualification_id"],
            "claim_sha256": claim_payload["claim_sha256"],
            "ranges": _reserved_excluded_ranges(),
        }
        range_payload["ledger_sha256"] = _digest_payload(
            b"IRENERCQRANGES\x01",
            range_payload,
        )
        range_bytes = (_canonical(range_payload) + "\n").encode("utf-8")
        range_digest = receipt_store.publish(
            _RANGE_LEDGER_NAME,
            range_bytes,
            existing=published_names,
            policy=policy,
        )
        published_names.add(_RANGE_LEDGER_NAME)
        range_file = {
            "status": "published_verified",
            "file": _RANGE_LEDGER_NAME,
            "records": len(_reserved_excluded_ranges()),
            "sha256": range_digest,
            "ledger_sha256": range_payload["ledger_sha256"],
        }

        # This is the only production call site that can materialize the fresh
        # TEST ranges, and it is textually and dynamically after claim publish.
        evidence, donor_mapping = _collect_post_claim_evidence(
            registration=registration,
            authorization=authorization,
            model=system.objective.model,
            stage_state=stage_state,
            autocast_context=autocast_context,
            receipt_store=receipt_store,
            expected_claim_entries=set(published_names),
            expected_claim_sha256=str(claim_payload["claim_sha256"]),
            expected_claim_payload=claim_payload,
        )
        decision_bytes = _raw_jsonl(
            tuple(item.to_dict() for item in evidence.decisions)
        )
        donor_bytes = _raw_jsonl(
            tuple(item.to_dict() for item in donor_mapping.assignments)
        )
        donor_assignment_cost = donor_mapping.assignment_cost_summary()
        # Preserve raw model and donor evidence before scientific validation so
        # an invalid-after-claim receipt still has forensic ledgers when the
        # concrete materializer completed.
        decision_digest = receipt_store.publish(
            _DECISION_LEDGER_NAME,
            decision_bytes,
            existing=published_names,
            policy=policy,
        )
        published_names.add(_DECISION_LEDGER_NAME)
        raw_ledgers["decisions"] = {
            "status": "published_verified",
            "file": _DECISION_LEDGER_NAME,
            "records": len(evidence.decisions),
            "sha256": decision_digest,
        }
        donor_digest = receipt_store.publish(
            _DONOR_LEDGER_NAME,
            donor_bytes,
            existing=published_names,
            policy=policy,
        )
        published_names.add(_DONOR_LEDGER_NAME)
        raw_ledgers["donors"] = {
            "status": "published_verified",
            "file": _DONOR_LEDGER_NAME,
            "records": len(donor_mapping.assignments),
            "sha256": donor_digest,
            "assignment_cost": donor_assignment_cost,
        }
        _validate_complete_donor_mapping(donor_mapping, evidence)
        _validate_evidence(evidence, registration, authorization)
        report = _evaluate_final_evidence(
            evidence,
            registration=registration,
            authorization=authorization,
            lookup=lookup,
        )
        report["donor_assignment_cost"] = donor_assignment_cost
        receipt_store.require_exact(_CLAIM_NAME, claim_bytes)
        receipt_store.require_exact(_RANGE_LEDGER_NAME, range_bytes)
        receipt_store.require_exact(_DECISION_LEDGER_NAME, decision_bytes)
        receipt_store.require_exact(_DONOR_LEDGER_NAME, donor_bytes)
        receipt_store.require_entries(published_names)
        report["raw_ledgers"] = raw_ledgers
        terminal: dict[str, object] = {
            "schema_version": 1,
            "qualification_id": registration.payload["qualification_id"],
            "status": "passed" if report["passed"] else "scientific_failed",
            "passed": bool(report["passed"]),
            "claim_sha256": claim_payload["claim_sha256"],
            "preclaim_readiness": readiness_binding,
            "donor_assignment_cost": donor_assignment_cost,
            "authorization": authorization_payload,
            "retired_ranges": retired_ranges,
            "guard_band": registration.payload["guard_band"],
            "future_campaign_offset": FUTURE_CAMPAIGN_OFFSET,
            "range_ledger": range_file,
            "retry_permitted": False,
            "report": report,
        }
    except BaseException as error:
        terminal = {
            "schema_version": 1,
            "qualification_id": registration.payload["qualification_id"],
            "status": "invalid_after_claim",
            "passed": False,
            "claim_sha256": claim_payload["claim_sha256"],
            "preclaim_readiness": readiness_binding,
            "donor_assignment_cost": donor_assignment_cost,
            "authorization": authorization_payload,
            "retired_ranges": retired_ranges,
            "guard_band": registration.payload["guard_band"],
            "future_campaign_offset": FUTURE_CAMPAIGN_OFFSET,
            "range_ledger": range_file,
            "raw_ledgers": raw_ledgers,
            "retry_permitted": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    try:
        receipt_store.require_exact(_CLAIM_NAME, claim_bytes)
        if range_file.get("status") == "published_verified":
            receipt_store.require_exact(_RANGE_LEDGER_NAME, range_bytes)
        if raw_ledgers["decisions"].get("status") == "published_verified":  # type: ignore[union-attr]
            receipt_store.require_exact(_DECISION_LEDGER_NAME, decision_bytes)
        if raw_ledgers["donors"].get("status") == "published_verified":  # type: ignore[union-attr]
            receipt_store.require_exact(_DONOR_LEDGER_NAME, donor_bytes)
        receipt_store.require_entries(published_names)
    except BaseException as durability_error:
        terminal = {
            **terminal,
            "status": "invalid_after_claim",
            "passed": False,
            "error_type": type(durability_error).__name__,
            "error": str(durability_error),
        }
    terminal["receipt_sha256"] = _digest_payload(
        b"IRENERCQRECEIPT\x01",
        terminal,
    )
    receipt_bytes = (_canonical(terminal) + "\n").encode("utf-8")
    try:
        receipt_store.publish(
            _RECEIPT_NAME,
            receipt_bytes,
            existing=published_names,
            policy=policy,
        )
        published_names.add(_RECEIPT_NAME)
        receipt_store.require_exact(_RECEIPT_NAME, receipt_bytes)
        receipt_store.require_entries(published_names)
        verified = _validate_terminal_receipt_commit(
            receipt_store,
            _prepare_receipt_verification_bindings(),
        )
        if verified.canonical_json != _canonical(terminal):
            raise RCQInputError("terminal receipt changed during authoritative readback")
        return verified
    finally:
        receipt_store.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run trusted RCQ-v2 preclaim, once-only final evaluation, or "
            "read-only terminal receipt verification"
        )
    )
    actions = parser.add_subparsers(dest="action", required=True)

    actions.add_parser(
        "preclaim",
        help="validate CUDA readiness without claiming or constructing TEST",
    )
    actions.add_parser(
        "final-once",
        help="require readiness, claim TEST once, and publish terminal evidence",
    )
    actions.add_parser(
        "verify-receipt",
        help="verify the canonical terminal commit without opening TEST",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.action == "preclaim":
        path, digest = _write_rcq_v2_preclaim_readiness()
        print(
            _canonical(
                {
                    "action": "rcq_v2_preclaim_v1",
                    "status": "ready_for_once_only_final",
                    "readiness_receipt": str(path),
                    "readiness_sha256": digest,
                    "sealed_test_examples_opened": 0,
                }
            ),
            flush=True,
        )
        return 0
    if arguments.action == "verify-receipt":
        result = _verify_rcq_v2_final_receipt()
        print(_canonical(result), flush=True)
        if result["verification_status"] == "authoritative_terminal_receipt":
            return 0
        return 3 if result["verification_status"] == "not_claimed" else 2
    receipt = _evaluate_rcq_v2_checkpoint_final_once()
    print(receipt.canonical_json, flush=True)
    return 0 if receipt.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "evaluator_bundle_sha256",
]
