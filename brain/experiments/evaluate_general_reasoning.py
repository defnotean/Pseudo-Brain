"""Frozen, zero-shot GSM8K diagnostic; no training or generated-code execution.

Scoring follows the pinned upstream #### numeric extraction convention. A small
sample of this historical public dataset cannot establish frontier competence
or absence of training contamination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
UPSTREAM_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
UPSTREAM_URL = f"https://raw.githubusercontent.com/openai/grade-school-math/{UPSTREAM_REVISION}/grade_school_math/data/test.jsonl"


def numeric_answer(text: str) -> str | None:
    found = re.search(r"#### (\-?[0-9\.\,]+)", text)
    return None if found is None else found[1].strip().replace(",", "")


def load_questions(path: Path, expected_sha256: str, count: int, seed: int):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Dataset does not match the frozen SHA-256")
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    if not 1 <= count <= len(rows):
        raise ValueError("count must be between one and the dataset size")
    ids = sorted(random.Random(seed).sample(range(len(rows)), count))
    selected = []
    for index in ids:
        row = rows[index]
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Invalid question at row {index}")
        if not isinstance(row.get("answer"), str) or numeric_answer(row["answer"]) is None:
            raise ValueError(f"Invalid evaluator answer at row {index}")
        selected.append((index, row))
    return selected


def evaluate_records(adapter, selected, max_new_tokens: int, emit):
    """The adapter receives only question text and an opaque row ID."""
    correct = 0
    for index, row in selected:
        question = row["question"] + "\nShow your reasoning. End with #### followed by the numeric answer."
        start = time.perf_counter()
        error = None
        try:
            prediction = adapter.answer_record(
                {"question_id": f"gsm8k-test-{index}", "question": question},
                max_new_tokens=max_new_tokens,
            )["answer"]
        except Exception as exc:
            prediction, error = "", f"{type(exc).__name__}: {exc}"
        observed, expected = numeric_answer(prediction), numeric_answer(row["answer"])
        passed = error is None and expected is not None and observed == expected
        correct += int(passed)
        emit({"question_id": f"gsm8k-test-{index}", "question": question,
              "answer": prediction, "extracted_answer": observed,
              "expected_numeric_answer": expected, "correct": passed,
              "error": error, "elapsed_seconds": time.perf_counter() - start})
    return {"total": len(selected), "correct": correct,
            "accuracy": correct / len(selected), "official_full_benchmark": False,
            "frontier_competence_demonstrated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 2048:
        parser.error("max-new-tokens must be between 1 and 2048")
    selected = load_questions(args.dataset, args.dataset_sha256, args.count, args.seed)
    if not args.checkpoint.is_file():
        parser.error("checkpoint must exist; random fallback is forbidden")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["PSEUDO_BRAIN_CPU_ONLY"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent
    from irene_brain.agent.recurrent_text_adapter import RecurrentTextAdapter
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    agent = RecurrentSoftwareAgent(checkpoint_path=args.checkpoint, device="cpu")
    metadata = {"upstream_url": UPSTREAM_URL, "upstream_revision": UPSTREAM_REVISION,
                "dataset_sha256": args.dataset_sha256, "seed": args.seed,
                "row_indices": [index for index, _ in selected],
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "parameters": sum(p.numel() for p in agent.model.parameters()),
                "compensated_state": agent.model.compensated_state,
                "pointer_mode": agent.model.pointer_mode,
                "max_new_tokens": args.max_new_tokens, "temperature": 0,
                "reset_per_question": True, "training_executed": False,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), *sorted((ROOT / "src/irene_brain").rglob("*.py"))]}}
    (args.run_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (args.run_dir / "answers.jsonl").open("x", encoding="utf-8") as output:
        def emit(record):
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(record["question_id"], "PASS" if record["correct"] else "FAIL", flush=True)
        report = evaluate_records(RecurrentTextAdapter(agent), selected, args.max_new_tokens, emit)
    (args.run_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0  # successful measurement; capability is recorded separately


if __name__ == "__main__":
    raise SystemExit(main())
