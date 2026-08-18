"""Analysis-only checkpoint comparison primitives.

This module is *not* a resume path: payloads are read with restricted
``weights_only`` deserialization for analysis, while
``training.checkpoint.load_checkpoint`` remains the only verified way to
resume a run. Comparison requires both checkpoints to declare the same
configuration, data, and code digests — comparing across runs is refused.
"""

from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor


_PAYLOAD_FIELDS = {
    "schema_version",
    "cursor",
    "config_sha256",
    "data_sha256",
    "code_sha256",
    "runtime_fingerprint",
    "system_state",
    "rng_state",
}


def load_analysis_payload(path: str | Path) -> Mapping[str, object]:
    """Read one checkpoint payload for analysis, failing closed on shape."""

    target = Path(path)
    if not target.is_file():
        raise ValueError(f"checkpoint does not exist: {target}")
    digest = sha256(target.read_bytes()).hexdigest()
    try:
        payload = torch.load(target, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "this PyTorch version lacks restricted weights_only checkpoint loading"
        ) from error
    if not isinstance(payload, Mapping) or not _PAYLOAD_FIELDS <= set(payload):
        raise ValueError("checkpoint payload has incompatible fields")
    if payload["schema_version"] not in (1, 2, 3):
        raise ValueError("unsupported checkpoint schema version")
    result = dict(payload)
    result["analysis_file_sha256"] = digest
    return result


def require_same_run_identity(
    first: Mapping[str, object], second: Mapping[str, object]
) -> None:
    """Refuse comparisons across different configurations, data, or code."""

    for name in ("config_sha256", "data_sha256", "code_sha256"):
        if first[name] != second[name]:
            raise ValueError(f"checkpoints disagree on {name}; refusing to compare")


def objective_states(
    payload: Mapping[str, object],
) -> Mapping[str, Tensor]:
    """Return the objective tensors recorded in an analysis payload."""

    system_state = payload["system_state"]
    if not isinstance(system_state, Mapping) or "objective" not in system_state:
        raise ValueError("checkpoint system state lacks an objective entry")
    objective = system_state["objective"]
    if not isinstance(objective, Mapping) or not objective:
        raise ValueError("checkpoint objective state must be a non-empty mapping")
    tensors: dict[str, Tensor] = {}
    for name, value in objective.items():
        if isinstance(value, Tensor):
            tensors[str(name)] = value.float()
    if not tensors:
        raise ValueError("checkpoint objective state contains no tensors")
    return tensors


def parameter_differences(
    first: Mapping[str, Tensor],
    second: Mapping[str, Tensor],
) -> tuple[dict[str, object], ...]:
    """Per-tensor difference rows, sorted by relative L2 change."""

    if set(first) != set(second):
        raise ValueError("checkpoint objective tensors differ; refusing to compare")
    rows = []
    for name in sorted(first):
        a, b = first[name], second[name]
        if a.shape != b.shape:
            raise ValueError(f"tensor {name} changed shape between checkpoints")
        delta = (b - a).norm().item()
        scale = a.norm().item()
        rows.append(
            {
                "tensor": name,
                "elements": a.numel(),
                "delta_l2": delta,
                "relative_l2": delta / (scale + 1e-12),
                "max_abs_delta": (b - a).abs().max().item() if a.numel() else 0.0,
                "identical": bool(torch.equal(a, b)),
            }
        )
    rows.sort(key=lambda row: (-float(row["relative_l2"]), str(row["tensor"])))
    return tuple(rows)


def metric_delta_table(
    first: Sequence[Mapping[str, float]],
    second: Sequence[Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Mean metric values for two models over the same batches, with deltas."""

    if len(first) != len(second) or not first:
        raise ValueError("metric rows must be non-empty and equally sized")
    keys = set(first[0])
    for rows in (first, second):
        for row in rows:
            if set(row) != keys:
                raise ValueError("metric rows must share one key set")
    table: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        a = sum(float(row[key]) for row in first) / len(first)
        b = sum(float(row[key]) for row in second) / len(second)
        table[key] = {"first": a, "second": b, "delta": b - a}
    return table


__all__ = [
    "load_analysis_payload",
    "metric_delta_table",
    "objective_states",
    "parameter_differences",
    "require_same_run_identity",
]
