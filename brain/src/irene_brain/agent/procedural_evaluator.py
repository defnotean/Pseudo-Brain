"""Procedural Held-Out Software Engineering Capability Benchmark for Pseudo-Brain.

Evaluates trained Pseudo-Brain models on procedurally generated, variable software tasks
with HIDDEN unit tests that are NOT placed in the agent's visible workspace directory.

When the agent emits `ACTION: FINISH`, an external evaluation oracle executes the
hidden unit tests against the agent's generated workspace. If the tests fail, the environment
rejects the finish claim with `[OBSERVATION: Task Incomplete: ...]` and requires the agent
to inspect the failure and continue unrolling actions.

Tracks genuine autonomous capability metrics:
- Task Completion Rate (% tasks passing 100% hidden unit tests)
- Valid Action Rate (% actions conforming to actuator grammar)
- Useful First Action Rate (% tasks where action 1 is an inspection or write)
- Error Recovery Rate (% instances where an observed test failure was repaired)
- Mean Cycles to Completion
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent
from irene_brain.agent.software_environment import NeuralSoftwareEnvironment


@dataclass
class ProceduralTask:
    """A procedurally generated held-out software engineering problem."""
    task_id: str
    domain: str
    goal: str
    target_module: str
    target_function: str
    initial_files: Dict[str, str]
    hidden_tests_code: str
    reference_solution: str


@dataclass
class BenchmarkEvaluationReport:
    """Rigorous evaluation telemetry across held-out procedural tasks."""
    total_tasks: int
    tasks_completed: int
    completion_rate: float
    total_actions: int
    valid_action_count: int
    valid_action_rate: float
    useful_first_action_count: int
    useful_first_action_rate: float
    error_recoveries: int
    mean_cycles: float
    task_results: List[Dict[str, Any]] = field(default_factory=list)


class ProceduralSoftwareBenchmark:
    """Procedural generator of held-out software engineering challenges."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_tasks(self, count: int = 10) -> List[ProceduralTask]:
        """Generate `count` distinct randomized procedural tasks."""
        generators = [
            self._generate_reverse_words_task,
            self._generate_prime_filter_task,
            self._generate_matrix_transpose_task,
            self._generate_bracket_checker_task,
            self._generate_clamp_numbers_task,
        ]
        tasks: List[ProceduralTask] = []
        for i in range(count):
            gen = generators[i % len(generators)]
            tasks.append(gen(task_index=i))
        return tasks

    def _generate_reverse_words_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"phrase_ops_{task_index}.py"
        fn_name = f"reverse_words_{task_index}"
        goal = (
            f"Implement a function `{fn_name}(s: str) -> str` in `{mod_name}` "
            "that reverses the order of whitespace-separated words in string `s` while stripping extra edge spaces."
        )
        # Hidden tests are kept outside the workspace
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}("hello world") == "world hello"
assert {fn_name}("  the quick brown fox  ") == "fox brown quick the"
assert {fn_name}("") == ""
assert {fn_name}("single") == "single"
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
def {fn_name}(s: str) -> str:
    words = s.strip().split()
    return " ".join(reversed(words))
"""
        return ProceduralTask(
            task_id=f"reverse_words_{task_index}",
            domain="string_algorithms",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden_tests.strip(),
            reference_solution=ref_sol.strip(),
        )

    def _generate_prime_filter_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"math_filter_{task_index}.py"
        fn_name = f"filter_primes_{task_index}"
        goal = (
            f"Create a module `{mod_name}` containing function `{fn_name}(numbers: list[int]) -> list[int]` "
            "that returns a list containing only the prime numbers from the input list, preserving order."
        )
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) == [2, 3, 5, 7, 11]
assert {fn_name}([-5, -1, 0, 1]) == []
assert {fn_name}([13, 17, 19]) == [13, 17, 19]
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
def {fn_name}(numbers):
    primes = []
    for n in numbers:
        if n < 2:
            continue
        is_p = True
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                is_p = False
                break
        if is_p:
            primes.append(n)
    return primes
"""
        return ProceduralTask(
            task_id=f"prime_filter_{task_index}",
            domain="number_theory",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden_tests.strip(),
            reference_solution=ref_sol.strip(),
        )

    def _generate_matrix_transpose_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"matrix_util_{task_index}.py"
        fn_name = f"transpose_matrix_{task_index}"
        goal = (
            f"Write `{mod_name}` with function `{fn_name}(grid: list[list[int]]) -> list[list[int]]` "
            "that returns the transposed matrix of a 2D list."
        )
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
assert {fn_name}([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
assert {fn_name}([]) == []
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
def {fn_name}(grid):
    if not grid or not grid[0]:
        return []
    return [[grid[r][c] for r in range(len(grid))] for c in range(len(grid[0]))]
"""
        return ProceduralTask(
            task_id=f"matrix_transpose_{task_index}",
            domain="linear_algebra",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden_tests.strip(),
            reference_solution=ref_sol.strip(),
        )

    def _generate_bracket_checker_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"syntax_check_{task_index}.py"
        fn_name = f"is_balanced_brackets_{task_index}"
        goal = (
            f"Implement `{fn_name}(s: str) -> bool` in `{mod_name}` returning True if parentheses, brackets, "
            "and braces '()[]{}' in string `s` are correctly nested and balanced, ignoring other characters."
        )
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}("([{{}}])") is True
assert {fn_name}("([)]") is False
assert {fn_name}("((()") is False
assert {fn_name}("def foo(x): return [x]") is True
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
def {fn_name}(s: str) -> bool:
    stack = []
    pairs = {{')': '(', ']': '[', '}}': '{{'}}
    for ch in s:
        if ch in '([{{':
            stack.append(ch)
        elif ch in ')]}}':
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return len(stack) == 0
"""
        return ProceduralTask(
            task_id=f"bracket_check_{task_index}",
            domain="data_structures",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden_tests.strip(),
            reference_solution=ref_sol.strip(),
        )

    def _generate_clamp_numbers_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"range_ops_{task_index}.py"
        fn_name = f"clamp_sequence_{task_index}"
        goal = (
            f"Build `{mod_name}` with `{fn_name}(vals: list[float], low: float, high: float) -> list[float]` "
            "clamping every value in `vals` between `low` and `high` inclusive."
        )
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([1, 5, 10, -2, 8], 0, 7) == [1, 5, 7, 0, 7]
assert {fn_name}([], 0, 10) == []
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
def {fn_name}(vals, low, high):
    return [max(low, min(high, v)) for v in vals]
"""
        return ProceduralTask(
            task_id=f"clamp_seq_{task_index}",
            domain="numerical_algorithms",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden_tests.strip(),
            reference_solution=ref_sol.strip(),
        )


def make_hidden_task_validator(
    hidden_tests_code: str,
    target_module: str,
    python_exe: str = sys.executable,
) -> Callable[[NeuralSoftwareEnvironment], Tuple[bool, str]]:
    """Constructs an external task validation oracle that evaluates hidden tests outside the workspace."""

    def validator(env: NeuralSoftwareEnvironment) -> Tuple[bool, str]:
        # 1. Verify target module exists in workspace
        target_path = env.workspace_dir / target_module
        if not target_path.exists():
            return False, f"Target file '{target_module}' does not exist in workspace"

        # 2. Write hidden tests to a secure temporary directory outside the workspace
        with tempfile.NamedTemporaryFile(suffix="_hidden_test.py", delete=False, mode="w", encoding="utf-8") as tf:
            tf.write(hidden_tests_code)
            test_file_path = tf.name

        try:
            sub_env = os.environ.copy()
            sub_env["PYTHONPATH"] = str(env.workspace_dir) + os.pathsep + sub_env.get("PYTHONPATH", "")

            proc = subprocess.run(
                [python_exe, test_file_path],
                cwd=str(env.workspace_dir),
                capture_output=True,
                text=True,
                timeout=10,
                env=sub_env,
            )

            if proc.returncode == 0 and "HIDDEN TESTS PASSED" in proc.stdout:
                return True, "100% hidden unit tests passed successfully"
            else:
                err_msg = proc.stderr.strip() or proc.stdout.strip() or f"Exit code {proc.returncode}"
                return False, f"Hidden unit tests failed: {err_msg[:200]}"
        except subprocess.TimeoutExpired:
            return False, "Hidden unit tests timed out (>10s)"
        except Exception as e:
            return False, f"Test evaluation error: {e}"
        finally:
            try:
                os.remove(test_file_path)
            except OSError:
                pass

    return validator


def evaluate_agent_on_benchmark(
    agent: RecurrentSoftwareAgent,
    tasks: List[ProceduralTask],
    max_cycles_per_task: int = 8,
) -> BenchmarkEvaluationReport:
    """Run an honest, leak-free capability evaluation across held-out procedural tasks."""
    total_tasks = len(tasks)
    tasks_completed = 0
    total_actions = 0
    valid_action_count = 0
    useful_first_action_count = 0
    error_recoveries = 0
    cycle_counts: List[int] = []
    task_results: List[Dict[str, Any]] = []

    valid_verbs = ("READ_FILE", "WRITE_FILE", "EDIT_FILE", "RUN_TESTS", "RETRIEVE_MEMORY", "FINISH")

    for task in tasks:
        temp_dir = Path(tempfile.mkdtemp(prefix=f"eval_{task.task_id}_"))
        validator = make_hidden_task_validator(task.hidden_tests_code, task.target_module)
        env = NeuralSoftwareEnvironment(workspace_dir=temp_dir, task_validator=validator)

        # Populate any initial files
        for fname, content in task.initial_files.items():
            (temp_dir / fname).parent.mkdir(parents=True, exist_ok=True)
            (temp_dir / fname).write_text(content, encoding="utf-8")

        # Run episode with autonomous policy generation (action_plan=None)
        res = agent.execute_pomdp_episode(goal=task.goal, env=env, action_plan=None, max_cycles=max_cycles_per_task)

        task_passed = res.success
        if task_passed:
            tasks_completed += 1

        total_actions += len(res.actions_taken)
        cycle_counts.append(res.cycles_completed)

        # Track valid actions vs invalid actions
        for act in res.actions_taken:
            parts = act.strip().split()
            if len(parts) >= 2 and parts[1] in valid_verbs:
                valid_action_count += 1

        # Track useful first action
        if res.actions_taken:
            first_act = res.actions_taken[0].strip().split()
            if len(first_act) >= 2 and first_act[1] in ("READ_FILE", "WRITE_FILE", "RETRIEVE_MEMORY"):
                useful_first_action_count += 1

        # Track error recovery
        for i in range(len(res.observations) - 1):
            if "FAILED" in res.observations[i] or "failed" in res.observations[i].lower():
                if "PASSED" in res.observations[i + 1] or "Successfully" in res.observations[i + 1]:
                    error_recoveries += 1

        task_results.append({
            "task_id": task.task_id,
            "success": task_passed,
            "cycles": res.cycles_completed,
            "actions": res.actions_taken,
            "final_summary": res.final_summary,
            "slot_0_delta": res.slot_0_delta,
            "slot_1_delta": res.slot_1_delta,
        })

    completion_rate = (tasks_completed / total_tasks) if total_tasks > 0 else 0.0
    valid_action_rate = (valid_action_count / total_actions) if total_actions > 0 else 0.0
    useful_first_action_rate = (useful_first_action_count / total_tasks) if total_tasks > 0 else 0.0
    mean_cycles = (sum(cycle_counts) / len(cycle_counts)) if cycle_counts else 0.0

    return BenchmarkEvaluationReport(
        total_tasks=total_tasks,
        tasks_completed=tasks_completed,
        completion_rate=completion_rate,
        total_actions=total_actions,
        valid_action_count=valid_action_count,
        valid_action_rate=valid_action_rate,
        useful_first_action_count=useful_first_action_count,
        useful_first_action_rate=useful_first_action_rate,
        error_recoveries=error_recoveries,
        mean_cycles=mean_cycles,
        task_results=task_results,
    )
