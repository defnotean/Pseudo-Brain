"""Compare two checkpoints of one run: parameters and shared-batch behavior.

Analysis-only tool, CPU-only. Loads two checkpoints written by
``training.checkpoint.save_checkpoint``, verifies they belong to the same
run identity (config, data, and code digests), rebuilds the model from
``--config``, and reports:

- per-tensor parameter differences, sorted by relative L2 change;
- per-model metrics on the same validation batches (action loss, movement
  exact-match, world losses, collapse diagnostics) with deltas.

This is not a resume path: payloads are read for analysis only, and
``training.checkpoint.load_checkpoint`` remains the only verified resume
interface.

Usage (from the repository root, play-safe Python):

    python brain/scripts/compare_checkpoints.py A.pt B.pt \
        --config brain/configs/training/dgx-smoke.toml --batches 4
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

BRAIN_ROOT = Path(__file__).resolve().parents[1]
SRC = BRAIN_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="override the recipe to CPU/float32 for local analysis of "
        "CUDA-configured checkpoints",
    )
    arguments = parser.parse_args()
    if arguments.batches < 1:
        raise SystemExit("--batches must be positive")

    import torch

    torch.set_num_threads(1)

    from irene_brain.evaluation.checkpoint_compare import (
        load_analysis_payload,
        metric_delta_table,
        objective_states,
        parameter_differences,
        require_same_run_identity,
    )
    from irene_brain.training.batches import MovingShapesBatchSource
    from irene_brain.training.config import load_training_config
    from irene_brain.training.objective import ThoughtFieldObjective
    from irene_brain.training.torch_system import TorchTrainingSystem

    payload_a = load_analysis_payload(arguments.first)
    payload_b = load_analysis_payload(arguments.second)
    require_same_run_identity(payload_a, payload_b)

    config = load_training_config(arguments.config)
    if arguments.cpu:
        from dataclasses import replace

        config = replace(
            config,
            precision=replace(
                config.precision,
                device="cpu",
                mode="float32",
                allow_tf32=False,
            ),
            resources=replace(config.resources, allow_gpu=False, cpu_threads=1),
        )
    module_name, factory_name = config.run.model_factory.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)

    systems = []
    for payload in (payload_a, payload_b):
        model = factory(config)
        system = TorchTrainingSystem(ThoughtFieldObjective(model), config)
        system.restore_checkpoint_state(
            {
                "objective": payload["system_state"]["objective"],
                "optimizer": payload["system_state"]["optimizer"],
                "scheduler": payload["system_state"]["scheduler"],
                "scaler": payload["system_state"]["scaler"],
            }
        )
        systems.append(system)

    source = MovingShapesBatchSource(config.dataset)
    batches = list(
        source.iter_batches(
            split="validation",
            epoch=0,
            start_batch=0,
            batch_size=1,
            max_batches=arguments.batches,
        )
    )
    rows = [
        [dict(system.evaluate_batch(batch).metrics) for batch in batches]
        for system in systems
    ]
    metric_table = {
        name: {key: round(value, 6) for key, value in entry.items()}
        for name, entry in metric_delta_table(
            [{k: float(v) for k, v in row.items()} for row in rows[0]],
            [{k: float(v) for k, v in row.items()} for row in rows[1]],
        ).items()
    }
    parameter_rows = parameter_differences(
        objective_states(payload_a), objective_states(payload_b)
    )
    changed = [row for row in parameter_rows if not row["identical"]]
    report = {
        "first": str(arguments.first),
        "second": str(arguments.second),
        "first_sha256": payload_a["analysis_file_sha256"],
        "second_sha256": payload_b["analysis_file_sha256"],
        "first_optimizer_step": payload_a["cursor"]["optimizer_step"],
        "second_optimizer_step": payload_b["cursor"]["optimizer_step"],
        "tensors_total": len(parameter_rows),
        "tensors_changed": len(changed),
        "top_parameter_changes": parameter_rows[:10],
        "metrics": metric_table,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"tensors changed: {len(changed)}/{len(parameter_rows)}")
    for row in parameter_rows[:5]:
        print(f"  {row['tensor']}: rel_l2={row['relative_l2']:.6f}")
    for key in ("action_loss", "movement_exact_match", "world_loss"):
        if key in metric_table:
            entry = metric_table[key]
            print(
                f"{key}: {entry['first']:.4f} -> {entry['second']:.4f}"
                f" (delta {entry['delta']:+.4f})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
