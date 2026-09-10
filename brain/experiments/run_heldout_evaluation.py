"""Zero-shot held-out evaluation runner on 10 unseen algorithm families.

Evaluates the frozen champion checkpoint (pb_pomdp_champion.pt) with zero additional
training on 10 completely unseen algorithm families.
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import torch

# Ensure brain/src is in path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "brain" / "src"))

from irene_brain.unified.unified_model import make_unified_model
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent
from irene_brain.agent.software_environment import NeuralSoftwareEnvironment
from irene_brain.agent.procedural_evaluator import (
    ProceduralSoftwareBenchmark,
    ProceduralTask,
    make_hidden_task_validator,
    evaluate_action_quality,
)


def classify_failure_mode(task: ProceduralTask, actions: List[str], final_summary: str, workspace_dir: Path) -> str:
    """Classify the failure mode of an evaluation episode."""
    if not actions:
        return "NO_ACTION_EMITTED"

    first_act = actions[0].strip()
    if first_act.startswith("ACTION: UNPARSED"):
        return "UNPARSED_ACTION_SYNTAX"

    target_file = workspace_dir / task.target_module
    if not target_file.exists():
        return "TARGET_FILE_NOT_WRITTEN"

    code = target_file.read_text(encoding="utf-8")
    if not code.strip():
        return "EMPTY_CODE_WRITTEN"

    # Check syntax
    try:
        compile(code, task.target_module, "exec")
    except SyntaxError as se:
        return f"SYNTAX_ERROR: {se.msg} (line {se.lineno})"

    # Check function presence
    if task.target_function not in code:
        return f"MISSING_TARGET_SYMBOL: {task.target_function} not defined"

    # If syntax and symbol are present, check validator summary
    if "Hidden unit tests failed" in final_summary:
        if "AssertionError" in final_summary:
            return "LOGIC_ASSERTION_FAILURE"
        elif "TypeError" in final_summary:
            return "TYPE_SIGNATURE_MISMATCH"
        elif "AttributeError" in final_summary:
            return "ATTRIBUTE_ERROR"
        elif "IndexError" in final_summary or "KeyError" in final_summary:
            return "RUNTIME_INDEX_KEY_ERROR"
        return "LOGIC_RUNTIME_ERROR"

    if "timed out" in final_summary.lower():
        return "TIMEOUT_INFINITE_LOOP"

    return "UNKNOWN_FAILURE"


def run_heldout_evaluation(
    ckpt_path: Path,
    num_tasks: int = 10,
    seed: int = 42,
    max_cycles: int = 4,
    receipt_out: Path = None,
) -> Dict[str, Any]:
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    tier = ckpt.get("tier", "tier2")
    vocab_size = ckpt.get("vocab_size", 32000)
    use_token_skip = ckpt.get("use_token_skip", True)
    trained_on = ckpt.get("trained_on", "unknown")
    
    print(f"Checkpoint Metadata: tier={tier}, vocab_size={vocab_size}, use_token_skip={use_token_skip}, trained_on={trained_on}")

    agent = RecurrentSoftwareAgent(checkpoint_path=ckpt_path)
    benchmark = ProceduralSoftwareBenchmark(seed=seed)
    heldout_tasks = benchmark.generate_heldout_tasks(count=num_tasks)

    print(f"\n=======================================================")
    print(f"  ZERO-SHOT EVALUATION: {len(heldout_tasks)} UNSEEN ALGORITHM FAMILIES")
    print(f"  FROZEN MODEL: {ckpt_path.name} (Zero Fine-Tuning)")
    print(f"=======================================================\n")

    total_actions = 0
    verb_grammar_count = 0
    parsable_action_count = 0
    executable_action_count = 0
    task_relevant_count = 0
    task_progressing_count = 0

    useful_first_actions = 0
    tasks_passed = 0
    task_details: List[Dict[str, Any]] = []

    for idx, task in enumerate(heldout_tasks):
        # Strict zero-shot isolation: clear recurrent and hierarchical memory before each task
        agent.reset()

        print(f"\n--- Task [{idx+1}/{len(heldout_tasks)}]: {task.task_id} ({task.domain}) ---")
        print(f"Goal: {task.goal}")
        
        temp_dir = Path(tempfile.mkdtemp(prefix=f"eval_{task.task_id}_"))
        validator = make_hidden_task_validator(task.hidden_tests_code, task.target_module)
        env = NeuralSoftwareEnvironment(workspace_dir=temp_dir, task_validator=validator)

        # Run POMDP episode
        t_start = time.perf_counter()
        result = agent.execute_pomdp_episode(goal=task.goal, env=env, action_plan=None, max_cycles=max_cycles, reset_state=True)
        duration = time.perf_counter() - t_start

        # Analyze action quality across the 5 tiers
        task_action_tiers = []
        for act, obs in zip(result.actions_taken, result.observations):
            total_actions += 1
            q = evaluate_action_quality(act, obs, task)
            if q.verb_grammar_valid:
                verb_grammar_count += 1
            if q.parsable_action_valid:
                parsable_action_count += 1
            if q.executable_action_valid:
                executable_action_count += 1
            if q.task_relevant_valid:
                task_relevant_count += 1
            if q.task_progressing_valid:
                task_progressing_count += 1
            task_action_tiers.append({
                "action": act,
                "verb_grammar": q.verb_grammar_valid,
                "parsable": q.parsable_action_valid,
                "executable": q.executable_action_valid,
                "task_relevant": q.task_relevant_valid,
                "task_progressing": q.task_progressing_valid,
            })

        if result.actions_taken:
            first_act_parts = result.actions_taken[0].strip().split()
            if len(first_act_parts) >= 2 and first_act_parts[1] in ("READ_FILE", "WRITE_FILE", "RETRIEVE_MEMORY"):
                useful_first_actions += 1

        if result.success:
            tasks_passed += 1
            status_str = "PASSED"
            failure_mode = "NONE"
        else:
            status_str = "FAILED"
            failure_mode = classify_failure_mode(task, result.actions_taken, result.final_summary, temp_dir)

        written_code = ""
        target_file = temp_dir / task.target_module
        if target_file.exists():
            written_code = target_file.read_text(encoding="utf-8")

        print(f"Status: {status_str} (cycles: {result.cycles_completed}, time: {duration:.2f}s)")
        print(f"Failure Mode: {failure_mode}")
        print(f"Actions taken ({len(result.actions_taken)}):")
        for c_i, (a, o) in enumerate(zip(result.actions_taken, result.observations)):
            act_preview = a.replace("\n", "\\n")[:100]
            obs_preview = o.replace("\n", "\\n")[:100]
            print(f"  Cycle {c_i}:")
            print(f"    Action:      {act_preview}")
            print(f"    Observation: {obs_preview}")

        if written_code:
            print(f"Written Code ({len(written_code)} chars):")
            for line in written_code.strip().splitlines()[:10]:
                print(f"    | {line}")
            if len(written_code.strip().splitlines()) > 10:
                print("    | ... [truncated]")

        task_details.append({
            "task_id": task.task_id,
            "domain": task.domain,
            "goal": task.goal,
            "target_module": task.target_module,
            "target_function": task.target_function,
            "success": result.success,
            "failure_mode": failure_mode,
            "cycles_completed": result.cycles_completed,
            "duration_seconds": duration,
            "actions_taken": result.actions_taken,
            "observations": result.observations,
            "action_tiers": task_action_tiers,
            "written_code": written_code,
            "final_summary": result.final_summary,
            "slot_0_delta": result.slot_0_delta,
            "slot_1_delta": result.slot_1_delta,
        })

    completion_rate = (tasks_passed / len(heldout_tasks)) if heldout_tasks else 0.0
    verb_grammar_rate = (verb_grammar_count / total_actions) if total_actions > 0 else 0.0
    parsable_action_rate = (parsable_action_count / total_actions) if total_actions > 0 else 0.0
    executable_action_rate = (executable_action_count / total_actions) if total_actions > 0 else 0.0
    task_relevant_rate = (task_relevant_count / total_actions) if total_actions > 0 else 0.0
    task_progressing_rate = (task_progressing_count / total_actions) if total_actions > 0 else 0.0
    useful_first_action_rate = (useful_first_actions / len(heldout_tasks)) if heldout_tasks else 0.0

    print(f"\n=======================================================")
    print(f"  FINAL HELDOUT BENCHMARK RECEIPT (ACTION HIERARCHY)")
    print(f"=======================================================")
    print(f"Total Held-Out Tasks:        {len(heldout_tasks)}")
    print(f"Tasks Completed (100% pass): {tasks_passed} / {len(heldout_tasks)} ({completion_rate*100:.1f}%)")
    print(f"Level 1: Verb Grammar:       {verb_grammar_count} / {total_actions} ({verb_grammar_rate*100:.1f}%)")
    print(f"Level 2: Parsable Action:    {parsable_action_count} / {total_actions} ({parsable_action_rate*100:.1f}%)")
    print(f"Level 3: Executable Action:  {executable_action_count} / {total_actions} ({executable_action_rate*100:.1f}%)")
    print(f"Level 4: Task Relevant:      {task_relevant_count} / {total_actions} ({task_relevant_rate*100:.1f}%)")
    print(f"Level 5: Task Progressing:   {task_progressing_count} / {total_actions} ({task_progressing_rate*100:.1f}%)")
    print(f"Useful First Action Rate:    {useful_first_actions} / {len(heldout_tasks)} ({useful_first_action_rate*100:.1f}%)")

    # Failure mode distribution
    failure_counts: Dict[str, int] = {}
    for td in task_details:
        fm = td["failure_mode"]
        failure_counts[fm] = failure_counts.get(fm, 0) + 1

    print("\nFailure Mode Breakdown:")
    for fm, cnt in sorted(failure_counts.items(), key=lambda x: -x[1]):
        print(f"  {fm}: {cnt}")

    receipt = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(ckpt_path.name),
        "tier": tier,
        "vocab_size": vocab_size,
        "use_token_skip": use_token_skip,
        "trained_on": trained_on,
        "evaluation_type": "zero_shot_unseen_families",
        "mode": "zero_shot",
        "state_isolation_per_task": True,
        "seed": seed,
        "total_tasks": len(heldout_tasks),
        "tasks_completed": tasks_passed,
        "completion_rate": completion_rate,
        "total_actions": total_actions,
        "action_hierarchy": {
            "level_1_verb_grammar_count": verb_grammar_count,
            "level_1_verb_grammar_rate": verb_grammar_rate,
            "level_2_parsable_action_count": parsable_action_count,
            "level_2_parsable_action_rate": parsable_action_rate,
            "level_3_executable_action_count": executable_action_count,
            "level_3_executable_action_rate": executable_action_rate,
            "level_4_task_relevant_count": task_relevant_count,
            "level_4_task_relevant_rate": task_relevant_rate,
            "level_5_task_progressing_count": task_progressing_count,
            "level_5_task_progressing_rate": task_progressing_rate,
        },
        "valid_action_count": verb_grammar_count,
        "valid_action_rate": verb_grammar_rate,
        "useful_first_action_rate": useful_first_action_rate,
        "failure_mode_breakdown": failure_counts,
        "tasks": task_details,
    }

    if receipt_out:
        receipt_out.parent.mkdir(parents=True, exist_ok=True)
        with open(receipt_out, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
        print(f"\nSaved detailed receipt to: {receipt_out}")

    return receipt


if __name__ == "__main__":
    ckpt = repo_root / "brain" / "checkpoints" / "pb_pomdp_champion.pt"
    out = repo_root / "brain" / "experiments" / "heldout_benchmark_receipt.json"
    run_heldout_evaluation(ckpt_path=ckpt, num_tasks=10, seed=42, max_cycles=4, receipt_out=out)
