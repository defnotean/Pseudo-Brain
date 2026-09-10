"""Three-Tier Capability Evaluation Runner for Pseudo-Brain.

Evaluates the champion policy across:
- Level A: Lexical / signature variation of trained algorithms.
- Level B: Intra-domain transfer (unseen algorithms within trained domains).
- Level C: Permanently sealed held-out algorithm families.

Measures all 5 levels of the Action Quality Hierarchy:
Level 1: Verb Grammar
Level 2: Parsable Action
Level 3: Executable Action
Level 4: Task Relevant
Level 5: Task Progressing
And final task completion rate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any

import torch

# Ensure brain/src is in path
repo_root = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(repo_root / "brain" / "src"))

from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent
from irene_brain.agent.procedural_evaluator import (
    ProceduralSoftwareBenchmark,
    evaluate_three_tier_benchmark,
    BenchmarkEvaluationReport,
)


def report_to_dict(r: BenchmarkEvaluationReport) -> Dict[str, Any]:
    return {
        "total_tasks": r.total_tasks,
        "tasks_completed": r.tasks_completed,
        "completion_rate": r.completion_rate,
        "total_actions": r.total_actions,
        "level_1_verb_grammar_count": r.verb_grammar_count,
        "level_1_verb_grammar_rate": r.verb_grammar_rate,
        "level_2_parsable_action_count": r.parsable_action_count,
        "level_2_parsable_action_rate": r.parsable_action_rate,
        "level_3_executable_action_count": r.executable_action_count,
        "level_3_executable_action_rate": r.executable_action_rate,
        "level_4_task_relevant_count": r.task_relevant_count,
        "level_4_task_relevant_rate": r.task_relevant_rate,
        "level_5_task_progressing_count": r.task_progressing_count,
        "level_5_task_progressing_rate": r.task_progressing_rate,
        "useful_first_action_count": r.useful_first_action_count,
        "useful_first_action_rate": r.useful_first_action_rate,
        "error_recoveries": r.error_recoveries,
        "mean_cycles": r.mean_cycles,
        "mode": r.mode,
        "task_results": r.task_results,
    }


def main():
    ckpt_path = repo_root / "brain" / "checkpoints" / "pb_pomdp_champion.pt"
    print(f"Loading champion checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Metadata: tier={ckpt.get('tier')}, vocab={ckpt.get('vocab_size')}, trained_on={ckpt.get('trained_on')}")

    agent = RecurrentSoftwareAgent(checkpoint_path=ckpt_path)
    bench = ProceduralSoftwareBenchmark(seed=42)

    print("\n=======================================================")
    print("  STARTING 3-TIER BENCHMARK EVALUATION (ZERO-SHOT ISOLATION)")
    print("=======================================================\n")

    t0 = time.perf_counter()
    reports = evaluate_three_tier_benchmark(
        agent=agent,
        benchmark=bench,
        count_per_tier=10,
        max_cycles_per_task=4,
        mode="zero_shot",
    )
    elapsed = time.perf_counter() - t0

    rep_a = reports["level_a_lexical"]
    rep_b = reports["level_b_domain_transfer"]
    rep_c = reports["level_c_sealed_ood"]

    print("\n" + "=" * 65)
    print("  THREE-TIER EVALUATION RECEIPT SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<30} | {'Level A':<10} | {'Level B':<10} | {'Level C (Sealed)':<10}")
    print("-" * 65)
    print(f"{'Tasks Completed':<30} | {rep_a.tasks_completed:>2}/{rep_a.total_tasks:<7} | {rep_b.tasks_completed:>2}/{rep_b.total_tasks:<7} | {rep_c.tasks_completed:>2}/{rep_c.total_tasks:<7}")
    print(f"{'Completion Rate':<30} | {rep_a.completion_rate*100:>5.1f}%     | {rep_b.completion_rate*100:>5.1f}%     | {rep_c.completion_rate*100:>5.1f}%")
    print(f"{'Level 1: Verb Grammar':<30} | {rep_a.verb_grammar_rate*100:>5.1f}%     | {rep_b.verb_grammar_rate*100:>5.1f}%     | {rep_c.verb_grammar_rate*100:>5.1f}%")
    print(f"{'Level 2: Parsable Action':<30} | {rep_a.parsable_action_rate*100:>5.1f}%     | {rep_b.parsable_action_rate*100:>5.1f}%     | {rep_c.parsable_action_rate*100:>5.1f}%")
    print(f"{'Level 3: Executable Action':<30} | {rep_a.executable_action_rate*100:>5.1f}%     | {rep_b.executable_action_rate*100:>5.1f}%     | {rep_c.executable_action_rate*100:>5.1f}%")
    print(f"{'Level 4: Task Relevant':<30} | {rep_a.task_relevant_rate*100:>5.1f}%     | {rep_b.task_relevant_rate*100:>5.1f}%     | {rep_c.task_relevant_rate*100:>5.1f}%")
    print(f"{'Level 5: Task Progressing':<30} | {rep_a.task_progressing_rate*100:>5.1f}%     | {rep_b.task_progressing_rate*100:>5.1f}%     | {rep_c.task_progressing_rate*100:>5.1f}%")
    print("=" * 65)

    receipt = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(ckpt_path.name),
        "tier": ckpt.get("tier"),
        "vocab_size": ckpt.get("vocab_size"),
        "trained_on": ckpt.get("trained_on"),
        "evaluation_duration_seconds": elapsed,
        "level_a_lexical": report_to_dict(rep_a),
        "level_b_domain_transfer": report_to_dict(rep_b),
        "level_c_sealed_ood": report_to_dict(rep_c),
    }

    out_file = repo_root / "brain" / "experiments" / "three_tier_benchmark_receipt.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nSaved official 3-tier benchmark receipt to: {out_file}")


if __name__ == "__main__":
    main()
