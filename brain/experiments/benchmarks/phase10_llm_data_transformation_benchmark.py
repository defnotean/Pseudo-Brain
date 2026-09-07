"""Phase 10 Benchmark Runner: Reuse Existing LLM Training Data.

Executes empirical comparison between Models A, B, C, and D across:
- Normal dialogue
- Unseen dialogue
- Interleaved multi-conversation
- Preemption resumption
- Cross-thread dependencies
- Long-term memory delay sweep across L in [10, 32, 64, 128, 256, 512]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from semantic_benchmark.llm_data_transform import (
    execute_phase10_benchmark,
    serialize_phase10_benchmark,
)


def run_benchmark(
    steps: int = 40,
    device: str = "cpu",
    output_dataset: Path = _REPO_ROOT / "docs/runs/artifacts/transformed_cognitive_dataset.jsonl",
    output_json: Path = _REPO_ROOT / "docs/runs/2026-09-07-phase10-llm-data-transformation.json",
    output_md: Path = _REPO_ROOT / "docs/runs/2026-09-07-phase10-llm-data-transformation.md",
) -> None:
    print("Executing Phase 10 Benchmark...")
    results = execute_phase10_benchmark(
        num_train_steps=steps,
        delays=[10, 32, 64, 128, 256, 512],
        device_str=device,
        seed=42,
    )

    print("\nSerializing benchmark artifacts...")
    serialize_phase10_benchmark(
        benchmark_data=results,
        json_path=output_json,
        md_path=output_md,
        dataset_path=output_dataset,
    )
    print("Serialization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 10 LLM Data Transformation Benchmark")
    parser.add_argument("--steps", type=int, default=40, help="Training steps per model")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")
    args = parser.parse_args()

    run_benchmark(steps=args.steps, device=args.device)
