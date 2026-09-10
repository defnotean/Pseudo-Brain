"""Test Suite for Autonomous Agent Held-Out Software Engineering Capability.

Verifies:
1. Procedural Task Diversity: Benchmark generates varied programming challenges with hidden unit tests.
2. Ground-Truth Solvability: Reference solutions achieve 100% pass rate on all hidden validators.
3. Strict External Validation: Agent cannot pass simply by declaring `ACTION: FINISH`;
   hidden unit tests must be executed and satisfied in an isolated subprocess.
4. Autonomous Capability Metrics: Measures task completion, action validity rate, and cycle efficiency.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from irene_brain.agent.procedural_evaluator import (
    ProceduralSoftwareBenchmark,
    ProceduralTask,
    evaluate_agent_on_benchmark,
    make_hidden_task_validator,
)
from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent
from irene_brain.agent.software_environment import NeuralSoftwareEnvironment


def test_procedural_benchmark_tasks_generation():
    """Verify that ProceduralSoftwareBenchmark produces distinct solvable programming tasks."""
    benchmark = ProceduralSoftwareBenchmark(seed=101)
    tasks = benchmark.generate_tasks(count=5)

    assert len(tasks) == 5
    domains = {t.domain for t in tasks}
    assert len(domains) >= 3, f"Expected domain diversity, got {domains}"

    for t in tasks:
        assert t.task_id != ""
        assert t.target_module.endswith(".py")
        assert t.target_function != ""
        assert "HIDDEN TESTS PASSED" in t.hidden_tests_code
        assert t.reference_solution != ""


def test_reference_solutions_pass_hidden_validators():
    """Verify ground-truth: each task's reference solution satisfies its hidden unit tests 100%."""
    benchmark = ProceduralSoftwareBenchmark(seed=202)
    tasks = benchmark.generate_tasks(count=5)

    for task in tasks:
        temp_dir = Path(tempfile.mkdtemp(prefix="test_ref_sol_"))
        validator = make_hidden_task_validator(task.hidden_tests_code, task.target_module)
        env = NeuralSoftwareEnvironment(workspace_dir=temp_dir, task_validator=validator)

        # 1. Before writing solution: validator must reject FINISH
        obs_empty = env.execute_action("ACTION: FINISH Premature claim")
        assert obs_empty.success is False
        assert "does not exist in workspace" in obs_empty.observation_text

        # 2. Write reference solution
        write_act = f"ACTION: WRITE_FILE {task.target_module}\n{task.reference_solution}\n"
        obs_write = env.execute_action(write_act)
        assert obs_write.success is True

        # 3. Now validator must approve FINISH
        obs_fin = env.execute_action("ACTION: FINISH Reference solution implemented")
        assert obs_fin.success is True, f"Reference solution failed hidden test for {task.task_id}: {obs_fin.observation_text}"
        assert "Task verified and passed" in obs_fin.observation_text


def test_agent_heldout_capability_evaluation_telemetry():
    """Run honest, leak-free capability evaluation on held-out tasks and record telemetry."""
    pomdp_ckpt = Path("checkpoints/pb_pomdp_champion.pt").resolve() if Path("checkpoints/pb_pomdp_champion.pt").exists() else Path("brain/checkpoints/pb_pomdp_champion.pt").resolve()
    champion_ckpt = Path("checkpoints/pb_35m_champion.pt").resolve() if Path("checkpoints/pb_35m_champion.pt").exists() else Path("brain/checkpoints/pb_35m_champion.pt").resolve()
    ckpt_path = pomdp_ckpt if pomdp_ckpt.exists() else champion_ckpt

    if ckpt_path.exists():
        agent = RecurrentSoftwareAgent(checkpoint_path=ckpt_path)
    else:
        agent = RecurrentSoftwareAgent()

    benchmark = ProceduralSoftwareBenchmark(seed=303)
    tasks = benchmark.generate_tasks(count=3)

    report = evaluate_agent_on_benchmark(agent=agent, tasks=tasks, max_cycles_per_task=4)

    # Basic telemetry assertions
    assert report.total_tasks == 3
    assert report.total_actions >= 3
    assert report.valid_action_rate >= 0.5  # Emits valid actuator commands
    assert 0.0 <= report.completion_rate <= 1.0

    # Ensure memory footprint remains strictly compliant throughout the evaluation
    assert agent.cognitive_state.hierarchical_state.fast_state_bytes() == 4096


def test_heldout_ten_unseen_families_ground_truth_solvability():
    """Verify all 10 held-out algorithm families are uniquely defined and their reference solutions pass 100%."""
    benchmark = ProceduralSoftwareBenchmark(seed=42)
    heldout_tasks = benchmark.generate_heldout_tasks(count=10)

    assert len(heldout_tasks) == 10
    domains = {t.domain for t in heldout_tasks}
    assert len(domains) >= 5, f"Expected wide domain diversity across 10 families, got {domains}"

    for task in heldout_tasks:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"test_gt_{task.task_id}_"))
        env = NeuralSoftwareEnvironment(workspace_dir=temp_dir)
        (temp_dir / task.target_module).write_text(task.reference_solution, encoding="utf-8")
        validator = make_hidden_task_validator(task.hidden_tests_code, task.target_module)
        passed, err = validator(env)
        assert passed is True, f"Reference solution for heldout task {task.task_id} failed: {err}"


def test_benchmark_receipt_provenance_and_generalization_boundary():
    """Verify held-out benchmark receipt: proves tier2 provenance and honest out-of-distribution boundary."""
    import json
    receipt_path = Path("experiments/heldout_benchmark_receipt.json") if Path("experiments/heldout_benchmark_receipt.json").exists() else Path("brain/experiments/heldout_benchmark_receipt.json")
    assert receipt_path.exists(), "Heldout benchmark receipt must exist"

    with open(receipt_path, "r", encoding="utf-8") as f:
        receipt = json.load(f)

    # Checkpoint provenance checks
    assert receipt["tier"] == "tier2", f"Expected tier2 provenance, got {receipt.get('tier')}"
    assert receipt["use_token_skip"] is True, "Expected token-skip enabled"
    assert receipt["evaluation_type"] == "zero_shot_unseen_families"
    assert receipt["total_tasks"] == 10

    # Capability vs Generalization boundary assertions:
    # 1. Action Hierarchy proves protocol mastery alongside out-of-distribution cliff:
    hierarchy = receipt["action_hierarchy"]
    assert hierarchy["level_1_verb_grammar_rate"] >= 0.70, f"Verb grammar too low: {hierarchy['level_1_verb_grammar_rate']}"
    assert hierarchy["level_2_parsable_action_rate"] >= 0.70, f"Parsable rate too low: {hierarchy['level_2_parsable_action_rate']}"
    assert hierarchy["level_3_executable_action_rate"] >= 0.70, f"Executable rate too low: {hierarchy['level_3_executable_action_rate']}"
    # 2. Honest zero-shot reporting on unseen families: zero attractor escape
    assert hierarchy["level_4_task_relevant_rate"] == 0.0, "Zero-shot model must not hallucinate ground truth"
    assert hierarchy["level_5_task_progressing_rate"] == 0.0, "Zero-shot unseen tasks must reflect honest baseline"
    assert receipt["completion_rate"] == 0.0

    # 3. State isolation confirmed
    assert receipt["mode"] == "zero_shot"
    assert receipt["state_isolation_per_task"] is True

    # 4. All failures are tracked and categorized
    assert "failure_mode_breakdown" in receipt
    assert receipt["failure_mode_breakdown"].get("TARGET_FILE_NOT_WRITTEN", 0) > 0


def test_zero_shot_vs_lifelong_mode_dispatch():
    """Verify evaluator cleanly differentiates zero-shot state isolation from lifelong persistence."""
    agent = RecurrentSoftwareAgent()
    benchmark = ProceduralSoftwareBenchmark(seed=505)
    tasks = benchmark.generate_tasks(count=2)

    # 1. Zero-shot mode: runs with agent.reset() per task
    report_zs = evaluate_agent_on_benchmark(agent=agent, tasks=tasks, max_cycles_per_task=2, mode="zero_shot")
    assert report_zs.mode == "zero_shot"
    assert report_zs.total_tasks == 2

    # 2. Lifelong mode: state persists across tasks
    from irene_brain.agent.procedural_evaluator import evaluate_agent_lifelong_benchmark
    report_life = evaluate_agent_lifelong_benchmark(agent=agent, tasks=tasks, max_cycles_per_task=2)
    assert report_life.mode == "lifelong"
    assert report_life.total_tasks == 2


