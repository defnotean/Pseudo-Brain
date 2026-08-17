"""Leakage audits for lifetime-level dataset splits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .records import LifetimeRecord


@dataclass(frozen=True, slots=True)
class SplitLeakage:
    grouping: str
    identity: str
    splits: tuple[str, ...]
    lifetime_ids: tuple[str, ...]


class SplitIntegrityError(ValueError):
    def __init__(self, leakages: tuple[SplitLeakage, ...]) -> None:
        self.leakages = leakages
        summary = "; ".join(
            f"{item.grouping}:{item.identity} spans {','.join(item.splits)}"
            for item in leakages
        )
        super().__init__(summary)


def audit_split_integrity(
    lifetimes: Iterable[LifetimeRecord],
) -> tuple[SplitLeakage, ...]:
    records = tuple(lifetimes)
    ids: dict[str, LifetimeRecord] = {}
    for record in records:
        if record.lifetime_id in ids:
            raise ValueError(f"duplicate lifetime_id: {record.lifetime_id}")
        ids[record.lifetime_id] = record

    groups: dict[tuple[str, str], list[LifetimeRecord]] = defaultdict(list)
    for record in records:
        groups[("world_lineage", record.world_lineage_id)].append(record)
        groups[("source_session", record.source_session_id)].append(record)
        groups[
            (
                "family_control_mapping",
                f"{record.environment_family}:{record.control_mapping_hash}",
            )
        ].append(record)
        if record.player_pseudonym is not None:
            groups[("player", record.player_pseudonym)].append(record)
        for lifetime_id in (record.lifetime_id, *record.ancestor_lifetime_ids):
            groups[("snapshot_ancestry", lifetime_id)].append(record)

    leakages: list[SplitLeakage] = []
    for (grouping, identity), members in sorted(groups.items()):
        splits = tuple(sorted({member.split for member in members}))
        if len(splits) <= 1:
            continue
        leakages.append(
            SplitLeakage(
                grouping=grouping,
                identity=identity,
                splits=splits,
                lifetime_ids=tuple(sorted({member.lifetime_id for member in members})),
            )
        )
    return tuple(leakages)


def require_split_integrity(
    lifetimes: Iterable[LifetimeRecord],
) -> tuple[LifetimeRecord, ...]:
    records = tuple(lifetimes)
    leakages = audit_split_integrity(records)
    if leakages:
        raise SplitIntegrityError(leakages)
    return records
