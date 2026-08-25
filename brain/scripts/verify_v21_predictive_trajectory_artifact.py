"""Fail-closed verifier for a V2.1 predictive-trajectory result artifact."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os

import torch

from stage_v21_predictive_trajectory_smoke import DATASET_MANIFEST, smoke_gate


ARMS = frozenset(
    {"decision_only_zero_pe", "predictive_zero_pe", "predictive_normal_pe"}
)
PREREGISTRATION = "2026-08-24-core-v2-predictive-trajectory-smoke-prereg"


def require_finite(value: object, path: str = "result") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite value")
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            require_finite(child, f"{path}[{index}]")


def file_sha256(path: str) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(artifact_path: str, checkpoint_dir: str | None, allow_mini: bool) -> dict[str, object]:
    artifact_path = os.path.abspath(artifact_path)
    with open(artifact_path, "r", encoding="utf-8") as handle:
        result = json.load(handle)
    require_finite(result)
    if result.get("schema_version") != 1:
        raise ValueError("artifact schema_version must be 1")
    if result.get("preregistration") != PREREGISTRATION:
        raise ValueError("artifact preregistration mismatch")
    mode = result.get("mode")
    if mode not in ({"smoke", "mini_integration"} if allow_mini else {"smoke"}):
        raise ValueError("artifact mode is not permitted")
    if mode == "smoke" and result.get("dataset_manifest") != DATASET_MANIFEST:
        raise ValueError("smoke dataset manifest mismatch")
    arms = result.get("arms")
    if not isinstance(arms, dict) or set(arms) != ARMS:
        raise ValueError("artifact must contain exactly the three frozen arms")
    if any(arm.get("status") != "completed" for arm in arms.values()):
        raise ValueError("every arm must be completed")

    recomputed_gate = smoke_gate(arms)
    recorded_gate = result.get("smoke_gate")
    if mode == "smoke":
        if not isinstance(recorded_gate, dict):
            raise ValueError("smoke artifact is missing smoke_gate")
        if recorded_gate.get("passed") is not recomputed_gate:
            raise ValueError("recorded smoke gate disagrees with recomputation")
    elif recorded_gate is not None:
        raise ValueError("mini artifact must not contain smoke_gate")

    checkpoint_dir = os.path.abspath(checkpoint_dir or artifact_path + ".checkpoints")
    verified_checkpoints = {}
    for name in sorted(ARMS):
        record = arms[name].get("checkpoint")
        if not isinstance(record, dict):
            raise ValueError(f"{name} is missing checkpoint provenance")
        recorded_path = record.get("path")
        recorded_hash = record.get("sha256")
        if not isinstance(recorded_path, str) or not isinstance(recorded_hash, str):
            raise ValueError(f"{name} checkpoint provenance is malformed")
        path = os.path.join(checkpoint_dir, os.path.basename(recorded_path))
        if not os.path.isfile(path):
            raise ValueError(f"{name} checkpoint file is missing: {path}")
        actual_hash = file_sha256(path)
        if actual_hash != recorded_hash:
            raise ValueError(f"{name} checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1 or payload.get("arm") != name:
            raise ValueError(f"{name} checkpoint metadata mismatch")
        if payload.get("seed") != 42:
            raise ValueError(f"{name} checkpoint training seed mismatch")
        if payload.get("dataset_manifest") != result.get("dataset_manifest"):
            raise ValueError(f"{name} checkpoint dataset manifest mismatch")
        if not isinstance(payload.get("model_state_dict"), dict):
            raise ValueError(f"{name} checkpoint has no model state")
        verified_checkpoints[name] = actual_hash

    return {
        "artifact": artifact_path,
        "artifact_sha256": file_sha256(artifact_path),
        "checkpoint_sha256": verified_checkpoints,
        "dataset_manifest": result["dataset_manifest"],
        "mode": mode,
        "scientific_gate_passed": recomputed_gate if mode == "smoke" else None,
        "verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--allow-mini", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.artifact, args.checkpoint_dir, args.allow_mini),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
