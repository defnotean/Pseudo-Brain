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
    pomdp_ckpt = Path("brain/checkpoints/pb_pomdp_champion.pt").resolve()
    champion_ckpt = Path("brain/checkpoints/pb_35m_champion.pt").resolve()
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
    assert report.valid_action_rate >= 0.0
    assert 0.0 <= report.completion_rate <= 1.0

    # Ensure memory footprint remains strictly compliant throughout the evaluation
    assert agent.cognitive_state.hierarchical_state.fast_state_bytes() == 4096
