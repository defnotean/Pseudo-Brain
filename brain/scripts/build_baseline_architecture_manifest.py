"""Emit the deterministic architecture manifest for matched baseline runs."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


BRAIN_ROOT = Path(__file__).resolve().parents[1]
SRC = BRAIN_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.model.baselines import (  # noqa: E402
    DENSE_COMMUNICATION_IDENTITY,
    FIXED_MULTI_HORIZON_IDENTITY,
    MATCHED_ENSEMBLE_IDENTITY,
    MONOLITHIC_IDENTITY,
    NO_COMMUNICATION_IDENTITY,
    PARAMETER_MATCHED_MONOLITHIC_IDENTITY,
    REACTIVE_IDENTITY,
    REFERENCE_IDENTITY,
    RECURRENT_TRANSFORMER_IDENTITY,
    RESET_STATE_IDENTITY,
    SERIAL_DEPTH_IDENTITY,
    build_architecture_manifest,
)
from irene_brain.project_paths import resolve_workspace_path  # noqa: E402
from irene_brain.training.config import load_training_config  # noqa: E402
from irene_brain.training.factory import (  # noqa: E402
    build_thesis_dense_routing_model,
    build_thesis_fixed_multi_horizon_model,
    build_thesis_model,
    build_thesis_matched_ensemble_model,
    build_thesis_monolithic_model,
    build_thesis_no_communication_model,
    build_thesis_parameter_matched_monolithic_model,
    build_thesis_reactive_model,
    build_thesis_recurrent_transformer_model,
    build_thesis_reset_slots_model,
    build_thesis_serial_depth_model,
)


REGISTRATIONS = {
    REFERENCE_IDENTITY.variant_id: (
        "configs/training/dgx-stagea-continuation-gate-b.toml",
        "irene_brain.training.factory:build_thesis_model",
        build_thesis_model,
    ),
    NO_COMMUNICATION_IDENTITY.variant_id: (
        "configs/training/baseline-stagea-no-communication.toml",
        "irene_brain.training.factory:build_thesis_no_communication_model",
        build_thesis_no_communication_model,
    ),
    RESET_STATE_IDENTITY.variant_id: (
        "configs/training/baseline-stagea-reset-slots.toml",
        "irene_brain.training.factory:build_thesis_reset_slots_model",
        build_thesis_reset_slots_model,
    ),
    FIXED_MULTI_HORIZON_IDENTITY.variant_id: (
        "configs/training/baseline-stagea-fixed-multi-horizon.toml",
        "irene_brain.training.factory:build_thesis_fixed_multi_horizon_model",
        build_thesis_fixed_multi_horizon_model,
    ),
    DENSE_COMMUNICATION_IDENTITY.variant_id: (
        "configs/training/baseline-stagea-dense-communication.toml",
        "irene_brain.training.factory:build_thesis_dense_routing_model",
        build_thesis_dense_routing_model,
    ),
    REACTIVE_IDENTITY.variant_id: (
        "configs/training/baseline-stagea-reactive.toml",
        "irene_brain.training.factory:build_thesis_reactive_model",
        build_thesis_reactive_model,
    ),
    SERIAL_DEPTH_IDENTITY.variant_id: (
        "configs/training/baseline-stagea-serial-depth.toml",
        "irene_brain.training.factory:build_thesis_serial_depth_model",
        build_thesis_serial_depth_model,
    ),
    MONOLITHIC_IDENTITY.variant_id: (
        "configs/training/baseline-stagea-monolithic-same-width.toml",
        "irene_brain.training.factory:build_thesis_monolithic_model",
        build_thesis_monolithic_model,
    ),
    PARAMETER_MATCHED_MONOLITHIC_IDENTITY.variant_id: (
        "configs/training/baseline-stagea-monolithic-parameter-matched.toml",
        "irene_brain.training.factory:build_thesis_parameter_matched_monolithic_model",
        build_thesis_parameter_matched_monolithic_model,
    ),
    MATCHED_ENSEMBLE_IDENTITY.variant_id: (
        "configs/training/baseline-stagea-matched-ensemble.toml",
        "irene_brain.training.factory:build_thesis_matched_ensemble_model",
        build_thesis_matched_ensemble_model,
    ),
    RECURRENT_TRANSFORMER_IDENTITY.variant_id: (
        "configs/training/baseline-stagea-recurrent-transformer.toml",
        "irene_brain.training.factory:build_thesis_recurrent_transformer_model",
        build_thesis_recurrent_transformer_model,
    ),
}

IMPLEMENTATION_FILES = (
    "src/irene_brain/model/actuator.py",
    "src/irene_brain/model/baselines.py",
    "src/irene_brain/model/brain_cell.py",
    "src/irene_brain/model/sensory.py",
    "src/irene_brain/model/spec.py",
    "src/irene_brain/model/torch_model.py",
    "src/irene_brain/training/factory.py",
    "src/irene_brain/training/objective.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the manifest to this path; stdout is used when omitted",
    )
    return parser


def build_manifest() -> dict[str, object]:
    models = {}
    recipes = {}
    for variant_id, (relative_path, factory_path, factory) in REGISTRATIONS.items():
        path = BRAIN_ROOT / relative_path
        raw = path.read_bytes()
        config = load_training_config(path)
        if config.run.model_factory != factory_path:
            raise ValueError(f"recipe factory mismatch for {variant_id}")
        models[variant_id] = (factory(config), factory_path)
        recipes[variant_id] = {
            "path": relative_path.replace("\\", "/"),
            "raw_sha256": sha256(raw).hexdigest(),
            "canonical_config_sha256": config.config_sha256,
        }

    manifest = build_architecture_manifest(models)
    for entry in manifest["variants"]:
        entry["training_recipe"] = recipes[str(entry["variant_id"])]
    source_files = {
        relative_path: sha256((BRAIN_ROOT / relative_path).read_bytes()).hexdigest()
        for relative_path in IMPLEMENTATION_FILES
    }
    canonical_source_files = json.dumps(
        source_files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest["implementation"] = {
        "source_files": source_files,
        "source_bundle_sha256": sha256(
            canonical_source_files.encode("utf-8")
        ).hexdigest(),
    }
    manifest.pop("manifest_sha256")
    canonical = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest["manifest_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    return manifest


def main() -> int:
    arguments = _parser().parse_args()
    rendered = json.dumps(
        build_manifest(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if arguments.output is None:
        sys.stdout.write(rendered)
    else:
        output = resolve_workspace_path(arguments.output)
        output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
