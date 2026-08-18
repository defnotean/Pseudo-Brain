"""Emit the deterministic manifest for the B2 world-model actor recipe family.

The ``irene.world_model_actor.gru_latent.v1`` control has its own objective
terms (teacher-forced latent rollouts) and therefore its own recipe family:
it must not ride on the slot-suite fairness manifest. This manifest pins the
variant identity, its allocated parameter count against the thesis reference,
its training recipe, its objective class path, and the source files that
define the family.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib
import json
from pathlib import Path
import sys


BRAIN_ROOT = Path(__file__).resolve().parents[1]
SRC = BRAIN_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.model.baselines import (  # noqa: E402
    REFERENCE_IDENTITY,
    architecture_manifest_entry,
    allocated_parameter_counts,
)
from irene_brain.model.world_model_actor import (  # noqa: E402
    WORLD_MODEL_ACTOR_IDENTITY,
)
from irene_brain.project_paths import resolve_workspace_path  # noqa: E402
from irene_brain.training.config import load_training_config  # noqa: E402
from irene_brain.training.factory import (  # noqa: E402
    build_thesis_model,
    build_thesis_world_model_actor_model,
)
from irene_brain.training.objective import ThoughtFieldObjective  # noqa: E402

PARAMETER_TOLERANCE_FRACTION = 0.01

REFERENCE_RECIPE = "configs/training/dgx-stagea-continuation-gate-b.toml"
ACTOR_RECIPE = "configs/training/baseline-stagea-world-model-actor.toml"
ACTOR_FACTORY_PATH = "irene_brain.training.factory:build_thesis_world_model_actor_model"

IMPLEMENTATION_FILES = (
    "src/irene_brain/model/world_model_actor.py",
    "src/irene_brain/training/factory.py",
    "src/irene_brain/training/objective.py",
    "src/irene_brain/training/train.py",
    "src/irene_brain/training/world_model_objective.py",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the manifest to this path; stdout is used when omitted",
    )
    return parser


def _recipe_entry(relative_path: str, expected_factory: str) -> tuple[object, dict[str, object]]:
    path = BRAIN_ROOT / relative_path
    raw = path.read_bytes()
    config = load_training_config(path)
    if config.run.model_factory != expected_factory:
        raise ValueError(f"recipe factory mismatch for {relative_path}")
    return config, {
        "path": relative_path.replace("\\", "/"),
        "raw_sha256": sha256(raw).hexdigest(),
        "canonical_config_sha256": config.config_sha256,
    }


def build_manifest() -> dict[str, object]:
    reference_config, _reference_recipe = _recipe_entry(
        REFERENCE_RECIPE, "irene_brain.training.factory:build_thesis_model"
    )
    reference_model = build_thesis_model(reference_config)
    if reference_model.architecture_variant_id != REFERENCE_IDENTITY.variant_id:
        raise ValueError("reference recipe did not build the routed reference")
    reference_count = allocated_parameter_counts(reference_model)["trainable"]

    actor_config, actor_recipe = _recipe_entry(ACTOR_RECIPE, ACTOR_FACTORY_PATH)
    actor = build_thesis_world_model_actor_model(actor_config)
    if actor.architecture_variant_id != WORLD_MODEL_ACTOR_IDENTITY.variant_id:
        raise ValueError("actor recipe did not build the world-model actor")

    objective_path = actor.training_objective_class_path
    module_name, attribute = objective_path.split(":", 1)
    objective_class = getattr(importlib.import_module(module_name), attribute, None)
    if objective_class is None or not issubclass(objective_class, ThoughtFieldObjective):
        raise ValueError("actor objective class path does not resolve to a ThoughtFieldObjective")

    entry = architecture_manifest_entry(
        actor,
        model_factory=ACTOR_FACTORY_PATH,
        reference_trainable_parameters=reference_count,
    )
    entry["training_recipe"] = actor_recipe
    if entry["absolute_trainable_parameter_delta_fraction"] > PARAMETER_TOLERANCE_FRACTION:
        raise ValueError(
            "world-model actor trainable parameters exceed the preregistered"
            f" {PARAMETER_TOLERANCE_FRACTION:.0%} tolerance band"
        )

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
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_kind": "world_model_actor_recipe_family",
        "parameter_tolerance_fraction": PARAMETER_TOLERANCE_FRACTION,
        "reference_variant_id": REFERENCE_IDENTITY.variant_id,
        "reference_trainable_parameters": reference_count,
        "training_objective_class_path": objective_path,
        "variants": [entry],
        "implementation": {
            "source_files": source_files,
            "source_bundle_sha256": sha256(
                canonical_source_files.encode("utf-8")
            ).hexdigest(),
        },
    }
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
