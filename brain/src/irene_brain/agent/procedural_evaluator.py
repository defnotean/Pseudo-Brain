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
class ActionQualityMetrics:
    """Multi-tiered evaluation of an agent's actuator decisions."""
    verb_grammar_valid: bool       # Level 1: Matches valid verb (WRITE_FILE, FINISH, etc.)
    parsable_action_valid: bool    # Level 2: Syntactically sound arguments (valid path/query)
    executable_action_valid: bool  # Level 3: Executed by env without OS/syntax crash
    task_relevant_valid: bool      # Level 4: Targets task's module, function, or domain
    task_progressing_valid: bool   # Level 5: Produces working code or passes tests
    target_span_token_accuracy: float = 0.0  # Continuous subword token accuracy against target module
    target_span_char_similarity: float = 0.0 # Normalized character similarity (1 - edit_distance / max_len)


@dataclass
class BenchmarkEvaluationReport:
    """Rigorous evaluation telemetry across held-out procedural tasks."""
    total_tasks: int
    tasks_completed: int
    completion_rate: float
    total_actions: int

    # 5-Tier Action Quality Hierarchy
    verb_grammar_count: int
    verb_grammar_rate: float
    parsable_action_count: int
    parsable_action_rate: float
    executable_action_count: int
    executable_action_rate: float
    task_relevant_count: int
    task_relevant_rate: float
    task_progressing_count: int
    task_progressing_rate: float

    # Continuous Target Span Similarity
    mean_target_span_token_accuracy: float = 0.0
    mean_target_span_char_similarity: float = 0.0

    # Backward compatibility aliases
    valid_action_count: int = 0
    valid_action_rate: float = 0.0
    useful_first_action_count: int = 0
    useful_first_action_rate: float = 0.0

    error_recoveries: int = 0
    mean_cycles: float = 0.0
    mode: str = "zero_shot"  # "zero_shot" or "lifelong"
    task_results: List[Dict[str, Any]] = field(default_factory=list)
    exact_module_binding_rate: float = 0.0


class ProceduralSoftwareBenchmark:
    """Procedural generator of held-out software engineering challenges."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_tasks(self, count: int = 10, split: str = "training") -> List[ProceduralTask]:
        """Generate `count` distinct randomized procedural tasks.

        Args:
            count: Number of tasks to generate.
            split: 'training' for the 5 training families,
                   'heldout' for the 10 completely unseen held-out test families.
        """
        if split == "heldout":
            return self.generate_heldout_tasks(count=count)

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

    def generate_level_a_tasks(self, count: int = 10) -> List[ProceduralTask]:
        """Level A Holdout: Same algorithms as training, but with randomized module names, function signatures, and docstrings."""
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
            # Use offset task index to ensure novel names
            tasks.append(gen(task_index=1000 + i))
        return tasks

    def generate_level_b_tasks(self, count: int = 10) -> List[ProceduralTask]:
        """Level B Holdout: Intra-domain transfer (unseen algorithms within trained domains)."""
        from irene_brain.agent.procedural_training_generator import ProceduralTrainingGenerator
        gen = ProceduralTrainingGenerator(seed=self.rng.randint(10000, 99999))
        return gen.generate_training_tasks(count=count)

    def generate_level_c_tasks(self, count: int = 10) -> List[ProceduralTask]:
        """Level C Holdout: Permanently sealed unseen algorithm families."""
        return self.generate_heldout_tasks(count=count)

    def generate_heldout_tasks(self, count: int = 10) -> List[ProceduralTask]:
        """Generate tasks exclusively from the 10 unseen held-out algorithm families."""
        generators = [
            self._generate_gcd_task,
            self._generate_run_length_encoding_task,
            self._generate_merge_intervals_task,
            self._generate_binary_tree_depth_task,
            self._generate_dedup_order_task,
            self._generate_moving_average_task,
            self._generate_roman_numeral_task,
            self._generate_lru_cache_task,
            self._generate_graph_bfs_task,
            self._generate_csv_parser_task,
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

    def _generate_gcd_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"math_gcd_{task_index}.py"
        fn_name = f"compute_gcd_{task_index}"
        goal = (
            f"Implement `{fn_name}(a: int, b: int) -> int` in `{mod_name}` "
            "computing the greatest common divisor of two non-negative integers using the Euclidean algorithm."
        )
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}(48, 18) == 6
assert {fn_name}(101, 10) == 1
assert {fn_name}(0, 5) == 5
assert {fn_name}(24, 24) == 24
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
def {fn_name}(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)
"""
        return ProceduralTask(
            task_id=f"gcd_{task_index}",
            domain="number_theory",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden_tests.strip(),
            reference_solution=ref_sol.strip(),
        )

    def _generate_run_length_encoding_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"codec_rle_{task_index}.py"
        fn_name = f"run_length_encode_{task_index}"
        goal = (
            f"Implement `{fn_name}(s: str) -> str` in `{mod_name}` "
            "performing run-length compression on string `s` (e.g. 'AAABBC' -> '3A2B1C'). Return empty string for empty input."
        )
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}("AAABBBCC") == "3A3B2C"
assert {fn_name}("A") == "1A"
assert {fn_name}("") == ""
assert {fn_name}("ABCD") == "1A1B1C1D"
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
def {fn_name}(s: str) -> str:
    if not s:
        return ""
    res = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            count += 1
        else:
            res.append(f"{{count}}{{s[i-1]}}")
            count = 1
    res.append(f"{{count}}{{s[-1]}}")
    return "".join(res)
"""
        return ProceduralTask(
            task_id=f"rle_{task_index}",
            domain="string_algorithms",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden_tests.strip(),
            reference_solution=ref_sol.strip(),
        )

    def _generate_merge_intervals_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"interval_ops_{task_index}.py"
        fn_name = f"merge_intervals_{task_index}"
        goal = (
            f"Write `{fn_name}(intervals: list[list[int]]) -> list[list[int]]` in `{mod_name}` "
            "that merges all overlapping intervals and returns non-overlapping intervals sorted by start time."
        )
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert {fn_name}([[1, 4], [4, 5]]) == [[1, 5]]
assert {fn_name}([]) == []
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
def {fn_name}(intervals):
    if not intervals:
        return []
    sorted_int = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_int[0]]
    for cur in sorted_int[1:]:
        prev = merged[-1]
        if cur[0] <= prev[1]:
            prev[1] = max(prev[1], cur[1])
        else:
            merged.append(cur)
    return merged
"""
        return ProceduralTask(
            task_id=f"merge_intervals_{task_index}",
            domain="interval_algorithms",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden_tests.strip(),
            reference_solution=ref_sol.strip(),
        )

    def _generate_binary_tree_depth_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"tree_depth_{task_index}.py"
        fn_name = f"max_tree_depth_{task_index}"
        goal = (
            f"Implement `{fn_name}(node: dict | None) -> int` in `{mod_name}` "
            "computing the maximum depth of a binary tree represented as dicts {'val': v, 'left': ..., 'right': ...}. None has depth 0."
        )
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}(None) == 0
assert {fn_name}({{"val": 1, "left": None, "right": None}}) == 1
assert {fn_name}({{"val": 1, "left": {{"val": 2, "left": None, "right": None}}, "right": None}}) == 2
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
def {fn_name}(node):
    if node is None:
        return 0
    return 1 + max({fn_name}(node.get("left")), {fn_name}(node.get("right")))
"""
        return ProceduralTask(
            task_id=f"tree_depth_{task_index}",
            domain="tree_algorithms",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden_tests.strip(),
            reference_solution=ref_sol.strip(),
        )

    def _generate_dedup_order_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"list_dedup_{task_index}.py"
        fn_name = f"dedup_preserve_order_{task_index}"
        goal = (
            f"Implement `{fn_name}(items: list) -> list` in `{mod_name}` "
            "removing duplicate elements from `items` while preserving the order of their first appearance."
        )
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([3, 1, 2, 3, 1, 4]) == [3, 1, 2, 4]
assert {fn_name}(["a", "b", "a"]) == ["a", "b"]
assert {fn_name}([]) == []
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
def {fn_name}(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
"""
        return ProceduralTask(
            task_id=f"dedup_order_{task_index}",
            domain="data_structures",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden_tests.strip(),
            reference_solution=ref_sol.strip(),
        )

    def _generate_moving_average_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"rolling_stats_{task_index}.py"
        fn_name = f"moving_average_{task_index}"
        goal = (
            f"Write `{fn_name}(vals: list[float], k: int) -> list[float]` in `{mod_name}` "
            "computing sliding window simple moving average with window size `k`. If len(vals) < k, return []."
        )
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([1.0, 2.0, 3.0, 4.0], 2) == [1.5, 2.5, 3.5]
assert {fn_name}([10.0, 20.0, 30.0], 1) == [10.0, 20.0, 30.0]
assert {fn_name}([1.0], 2) == []
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
def {fn_name}(vals, k):
    if len(vals) < k or k <= 0:
        return []
    res = []
    cur_sum = sum(vals[:k])
    res.append(cur_sum / k)
    for i in range(k, len(vals)):
        cur_sum += vals[i] - vals[i - k]
        res.append(cur_sum / k)
    return res
"""
        return ProceduralTask(
            task_id=f"moving_average_{task_index}",
            domain="numerical_algorithms",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden_tests.strip(),
            reference_solution=ref_sol.strip(),
        )

    def _generate_roman_numeral_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"roman_calc_{task_index}.py"
        fn_name = f"parse_roman_numeral_{task_index}"
        goal = (
            f"Implement `{fn_name}(roman: str) -> int` in `{mod_name}` "
            "converting Roman numeral string (I, V, X, L, C, D, M) to its integer value."
        )
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}("XIV") == 14
assert {fn_name}("III") == 3
assert {fn_name}("LVIII") == 58
assert {fn_name}("MCMXCIV") == 1994
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
def {fn_name}(roman: str) -> int:
    vals = {{'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}}
    total = 0
    prev = 0
    for ch in reversed(roman.upper()):
        v = vals.get(ch, 0)
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total
"""
        return ProceduralTask(
            task_id=f"roman_numeral_{task_index}",
            domain="string_algorithms",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden_tests.strip(),
            reference_solution=ref_sol.strip(),
        )

    def _generate_lru_cache_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"cache_store_{task_index}.py"
        cls_name = f"LRUCache_{task_index}"
        goal = (
            f"Implement class `{cls_name}` in `{mod_name}` with `__init__(self, capacity: int)`, "
            "`get(self, key: str) -> int` (returning -1 if not found), and `put(self, key: str, value: int)` evicting least recently used key."
        )
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {cls_name}
c = {cls_name}(2)
c.put("a", 1)
c.put("b", 2)
assert c.get("a") == 1
c.put("c", 3)
assert c.get("b") == -1
assert c.get("c") == 3
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
class {cls_name}:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {{}}

    def get(self, key: str) -> int:
        if key not in self.cache:
            return -1
        val = self.cache.pop(key)
        self.cache[key] = val
        return val

    def put(self, key: str, value: int) -> None:
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = value
"""
        return ProceduralTask(
            task_id=f"lru_cache_{task_index}",
            domain="data_structures",
            goal=goal,
            target_module=mod_name,
            target_function=cls_name,
            initial_files={},
            hidden_tests_code=hidden_tests.strip(),
            reference_solution=ref_sol.strip(),
        )

    def _generate_graph_bfs_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"graph_path_{task_index}.py"
        fn_name = f"bfs_shortest_path_{task_index}"
        goal = (
            f"Write `{fn_name}(adj: dict[str, list[str]], start: str, end: str) -> list[str] | None` in `{mod_name}` "
            "finding shortest path from start to end in unweighted directed graph. Return None if unreachable."
        )
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {fn_name}
g = {{"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}}
assert {fn_name}(g, "A", "D") in [["A", "B", "D"], ["A", "C", "D"]]
assert {fn_name}(g, "A", "A") == ["A"]
assert {fn_name}(g, "D", "A") is None
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
from collections import deque

def {fn_name}(adj, start, end):
    if start == end:
        return [start]
    visited = {{start}}
    queue = deque([[start]])
    while queue:
        path = queue.popleft()
        node = path[-1]
        for neighbor in adj.get(node, []):
            if neighbor == end:
                return path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])
    return None
"""
        return ProceduralTask(
            task_id=f"graph_bfs_{task_index}",
            domain="graph_algorithms",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden_tests.strip(),
            reference_solution=ref_sol.strip(),
        )

    def _generate_csv_parser_task(self, task_index: int) -> ProceduralTask:
        mod_name = f"csv_util_{task_index}.py"
        fn_name = f"parse_csv_row_{task_index}"
        goal = (
            f"Implement `{fn_name}(line: str) -> list[str]` in `{mod_name}` "
            "parsing a comma-separated row into string fields, respecting double-quoted fields that contain commas."
        )
        hidden_tests = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}("a,b,c") == ["a", "b", "c"]
assert {fn_name}('a,"b,c",d') == ["a", "b,c", "d"]
assert {fn_name}("") in [[], [""]]
print("HIDDEN TESTS PASSED")
"""
        ref_sol = f"""
import csv

def {fn_name}(line: str):
    reader = csv.reader([line])
    rows = list(reader)
    return rows[0] if rows else []
"""
        return ProceduralTask(
            task_id=f"csv_parser_{task_index}",
            domain="file_formats",
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
            sub_env["PYTHONDONTWRITEBYTECODE"] = "1"
            # A unique, unwritten cache namespace prevents same-size rapid edits
            # from reusing old .pyc files and falsely passing/failing repairs.
            sub_env["PYTHONPYCACHEPREFIX"] = test_file_path + ".cache"

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
                # Preserve the exception itself rather than only the start of a
                # traceback. Normalize ephemeral paths for repeatable trajectories.
                summary = err_msg.splitlines()[-1]
                summary = summary.replace(str(env.workspace_dir), "<workspace>").replace(test_file_path, "<hidden_test>")
                return False, f"Hidden unit tests failed: {summary[:200]}"
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


def compute_span_similarities(generated_str: str, target_str: str) -> Tuple[float, float]:
    """Compute token-level precision/recall and character-level edit similarity."""
    if not target_str:
        return 0.0, 0.0
    if generated_str == target_str:
        return 1.0, 1.0

    # Levenshtein distance
    m, n = len(generated_str), len(target_str)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if generated_str[i - 1] == target_str[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    edit_dist = dp[m][n]
    char_sim = max(0.0, 1.0 - edit_dist / max(m, n, 1))

    # Token/subword overlap similarity with multiset intersection, strictly bounded in [0.0, 1.0]
    if generated_str == target_str:
        tok_acc = 1.0
    else:
        from collections import Counter
        gen_parts = [p for p in generated_str.replace('.', '_').split('_') if p]
        tgt_parts = [p for p in target_str.replace('.', '_').split('_') if p]
        if tgt_parts and gen_parts:
            gen_counts = Counter(gen_parts)
            tgt_counts = Counter(tgt_parts)
            intersection = sum(min(count, tgt_counts[token]) for token, count in gen_counts.items())
            tok_acc = min(1.0, max(0.0, intersection / max(len(tgt_parts), len(gen_parts))))
        elif not tgt_parts and not gen_parts:
            tok_acc = 1.0
        else:
            tok_acc = 0.0

    return min(1.0, max(0.0, tok_acc)), min(1.0, max(0.0, char_sim))


def evaluate_action_quality(
    action: str,
    observation: str,
    task: ProceduralTask,
) -> ActionQualityMetrics:
    """Classify an action according to the 5-tier Action Quality Hierarchy.

    Hierarchy:
    1. Verb Grammar: Starts with recognized actuator command (WRITE_FILE, FINISH, etc.)
    2. Parsable Action: Contains structurally sound arguments (valid path, query, summary)
    3. Environment-Executable: Executed without OS argument failure or unparseable format
    4. Task-Relevant: Targets the specific module, function, or domain of this task (not an attractor)
    5. Task-Progressing: Successfully compiles, satisfies checks, or passes hidden tests
    """
    valid_verbs = ("READ_FILE", "WRITE_FILE", "EDIT_FILE", "RUN_TESTS", "RETRIEVE_MEMORY", "FINISH")

    # Level 1: Verb Grammar
    parts = action.strip().split()
    verb_valid = len(parts) >= 2 and parts[0] == "ACTION:" and parts[1] in valid_verbs

    # Level 2: Parsable Action (Arguments format)
    parsable_valid = False
    if verb_valid:
        verb = parts[1]
        if verb in ("RUN_TESTS", "RETRIEVE_MEMORY", "FINISH"):
            if verb == "RUN_TESTS":
                parsable_valid = True
            elif len(parts) >= 3:
                parsable_valid = True
        elif verb in ("WRITE_FILE", "READ_FILE", "EDIT_FILE"):
            lines = action.strip().split("\n", 1)
            header_parts = lines[0].strip().split()
            if len(header_parts) >= 3:
                target_path = header_parts[2]
                # Valid path: no spaces in filename, reasonable length, no illegal characters
                if not any(c in target_path for c in ('"', "'", '<', '>', '|', '?', '*')) and len(target_path) < 80:
                    if verb == "WRITE_FILE":
                        parsable_valid = len(lines) > 1 and len(lines[1].strip()) > 0
                    else:
                        parsable_valid = True

    # Level 3: Executable Action
    # Executable if the environment didn't reject it as unrecognized format or OS error (Errno 22)
    exec_valid = False
    if verb_valid:
        unrecognized = (
            "Unrecognized action format" in observation
            or "failed with OS error" in observation
            or "Errno 22" in observation
        )
        exec_valid = not unrecognized

    # Level 4: Task Relevant Action & Continuous Span Similarities
    task_rel_valid = False
    span_tok_acc = 0.0
    span_char_sim = 0.0
    if verb_valid and parsable_valid:
        verb = parts[1]
        if verb in ("WRITE_FILE", "READ_FILE", "EDIT_FILE"):
            lines = action.strip().split("\n", 1)
            target_path = lines[0].strip().split()[2] if len(lines[0].strip().split()) >= 3 else ""
            span_tok_acc, span_char_sim = compute_span_similarities(target_path, task.target_module)
            if target_path == task.target_module:
                task_rel_valid = True
        elif verb == "FINISH":
            if task.target_function in action or task.target_module in action:
                task_rel_valid = True
        elif verb == "RETRIEVE_MEMORY":
            if any(term in action for term in (task.target_function, task.domain, task.target_module)):
                task_rel_valid = True

    # Level 5: Task Progressing Action
    task_prog_valid = False
    if task_rel_valid:
        if "Successfully wrote" in observation or "Task verified and passed" in observation:
            task_prog_valid = True

    return ActionQualityMetrics(
        verb_grammar_valid=verb_valid,
        parsable_action_valid=parsable_valid,
        executable_action_valid=exec_valid,
        task_relevant_valid=task_rel_valid,
        task_progressing_valid=task_prog_valid,
        target_span_token_accuracy=span_tok_acc,
        target_span_char_similarity=span_char_sim,
    )


def recovery_kind(failed, repaired) -> str:
    """Classify proven recovery without upgrading incomplete historical traces."""
    before = failed.get("target_before") or {}
    after = repaired.get("target_after") or {}
    if not after.get("exists") or not after.get("sha256"):
        return "unclassified_recovery"
    if failed["action_type"] in ("WRITE_FILE", "EDIT_FILE"):
        return "rejected_mutation_recovery"
    if before.get("exists") is False:
        return "missing_file_recovery"
    if (before.get("exists") is True and before.get("sha256")
            and before["sha256"] != after["sha256"]):
        return "existing_code_repair"
    return "unclassified_recovery"


def verified_repair_evidence(result, validator_present: bool) -> Dict[str, Any]:
    """Require an observed failure, repair phase, mutation and oracle-approved FINISH."""
    feedback = result.action_feedback
    if (not validator_present or not result.success or result.policy_source != "autonomous"
            or not feedback or feedback[-1]["action_type"] != "FINISH"
            or not feedback[-1]["success"]):
        return {}
    for i, failed in enumerate(feedback[:-1]):
        if failed["success"] or failed["action_type"] not in ("WRITE_FILE", "EDIT_FILE", "RUN_TESTS", "FINISH"):
            continue
        if "[PHASE: REPAIR_" not in failed["transition"]:
            continue
        for repaired in feedback[i + 1:-1]:
            if repaired["success"] and repaired["action_type"] in ("WRITE_FILE", "EDIT_FILE"):
                return {"failure_cycle": failed["cycle"], "repair_cycle": repaired["cycle"],
                        "passed_cycle": feedback[-1]["cycle"],
                        "recovery_kind": recovery_kind(failed, repaired),
                        "reflection": "environment-derived repair phase ingested into recurrent state",
                        "validation": "external hidden-test validator"}
    return {}


def evaluate_agent_on_benchmark(
    agent: RecurrentSoftwareAgent,
    tasks: List[ProceduralTask],
    max_cycles_per_task: int = 8,
    mode: str = "zero_shot",
) -> BenchmarkEvaluationReport:
    """Run an honest, leak-free capability evaluation across held-out procedural tasks.

    Parameters:
    - agent: The RecurrentSoftwareAgent instance.
    - tasks: List of ProceduralTask challenges to evaluate.
    - max_cycles_per_task: Max POMDP cycles per task.
    - mode: "zero_shot" (pure state isolation, resets agent state between tasks) or
            "lifelong" (persists recurrent and hierarchical memory across tasks).
    """
    total_tasks = len(tasks)
    tasks_completed = 0
    total_actions = 0

    # 5-Tier Action Quality Counters
    verb_grammar_count = 0
    parsable_action_count = 0
    executable_action_count = 0
    task_relevant_count = 0
    task_progressing_count = 0
    span_tok_accs: List[float] = []
    span_char_sims: List[float] = []

    useful_first_action_count = 0
    error_recoveries = 0
    cycle_counts: List[int] = []
    task_results: List[Dict[str, Any]] = []

    for task in tasks:
        # Zero-shot contract: explicitly reset agent state before each task
        if mode == "zero_shot":
            agent.reset()

        temp_dir = Path(tempfile.mkdtemp(prefix=f"eval_{task.task_id}_"))
        validator = make_hidden_task_validator(task.hidden_tests_code, task.target_module)
        env = NeuralSoftwareEnvironment(workspace_dir=temp_dir, task_validator=validator)

        # Populate any initial files
        for fname, content in task.initial_files.items():
            (temp_dir / fname).parent.mkdir(parents=True, exist_ok=True)
            (temp_dir / fname).write_text(content, encoding="utf-8")

        # Run episode with autonomous policy generation (action_plan=None)
        res = agent.execute_pomdp_episode(
            goal=task.goal,
            env=env,
            action_plan=None,
            max_cycles=max_cycles_per_task,
            reset_state=(mode == "zero_shot"),
            target_module=task.target_module,
            target_function=task.target_function,
        )

        task_passed = res.success
        if task_passed:
            tasks_completed += 1

        total_actions += len(res.actions_taken)
        cycle_counts.append(res.cycles_completed)

        # Track 5-tier action hierarchy & span similarities
        task_tok_accs: List[float] = []
        task_char_sims: List[float] = []
        for act, obs in zip(res.actions_taken, res.observations):
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
            if q.target_span_char_similarity > 0.0 or "WRITE_FILE" in act:
                span_tok_accs.append(q.target_span_token_accuracy)
                span_char_sims.append(q.target_span_char_similarity)
                task_tok_accs.append(q.target_span_token_accuracy)
                task_char_sims.append(q.target_span_char_similarity)

        # Track useful first action
        if res.actions_taken:
            first_act = res.actions_taken[0].strip().split()
            if len(first_act) >= 2 and first_act[1] in ("READ_FILE", "WRITE_FILE", "RETRIEVE_MEMORY"):
                useful_first_action_count += 1

        # A write after a failure is not evidence of a repaired, passing task.
        repair_evidence = verified_repair_evidence(res, env.task_validator is not None)
        error_recoveries += int(bool(repair_evidence))
        writes = [a.partition("\n")[0].removeprefix("ACTION: WRITE_FILE ").strip()
                  for a in res.actions_taken if a.startswith("ACTION: WRITE_FILE ")]
        exact_module_binding = bool(writes and writes[0] == task.target_module)

        task_results.append({
            "task_id": task.task_id,
            "success": task_passed,
            "cycles": res.cycles_completed,
            "actions": res.actions_taken,
            "observations": res.observations,
            "action_feedback": res.action_feedback,
            "policy_source": res.policy_source,
            "verified_repair": repair_evidence,
            "exact_module_binding": exact_module_binding,
            "final_summary": res.final_summary,
            "slot_0_delta": res.slot_0_delta,
            "slot_1_delta": res.slot_1_delta,
            "max_span_char_similarity": max(task_char_sims, default=0.0),
            "max_span_token_accuracy": max(task_tok_accs, default=0.0),
        })

    completion_rate = (tasks_completed / total_tasks) if total_tasks > 0 else 0.0
    verb_grammar_rate = (verb_grammar_count / total_actions) if total_actions > 0 else 0.0
    parsable_action_rate = (parsable_action_count / total_actions) if total_actions > 0 else 0.0
    executable_action_rate = (executable_action_count / total_actions) if total_actions > 0 else 0.0
    task_relevant_rate = (task_relevant_count / total_actions) if total_actions > 0 else 0.0
    task_progressing_rate = (task_progressing_count / total_actions) if total_actions > 0 else 0.0
    useful_first_action_rate = (useful_first_action_count / total_tasks) if total_tasks > 0 else 0.0
    mean_cycles = (sum(cycle_counts) / len(cycle_counts)) if cycle_counts else 0.0
    mean_target_span_token_accuracy = (sum(span_tok_accs) / len(span_tok_accs)) if span_tok_accs else 0.0
    mean_target_span_char_similarity = (sum(span_char_sims) / len(span_char_sims)) if span_char_sims else 0.0

    return BenchmarkEvaluationReport(
        total_tasks=total_tasks,
        tasks_completed=tasks_completed,
        completion_rate=completion_rate,
        total_actions=total_actions,
        verb_grammar_count=verb_grammar_count,
        verb_grammar_rate=verb_grammar_rate,
        parsable_action_count=parsable_action_count,
        parsable_action_rate=parsable_action_rate,
        executable_action_count=executable_action_count,
        executable_action_rate=executable_action_rate,
        task_relevant_count=task_relevant_count,
        task_relevant_rate=task_relevant_rate,
        task_progressing_count=task_progressing_count,
        task_progressing_rate=task_progressing_rate,
        mean_target_span_token_accuracy=mean_target_span_token_accuracy,
        mean_target_span_char_similarity=mean_target_span_char_similarity,
        valid_action_count=verb_grammar_count,
        valid_action_rate=verb_grammar_rate,
        useful_first_action_count=useful_first_action_count,
        useful_first_action_rate=useful_first_action_rate,
        error_recoveries=error_recoveries,
        mean_cycles=mean_cycles,
        mode=mode,
        task_results=task_results,
        exact_module_binding_rate=sum(t["exact_module_binding"] for t in task_results) / total_tasks if total_tasks else 0.0,
    )


def evaluate_agent_lifelong_benchmark(
    agent: RecurrentSoftwareAgent,
    tasks: List[ProceduralTask],
    max_cycles_per_task: int = 8,
) -> BenchmarkEvaluationReport:
    """Run continual lifelong learning evaluation where recurrent state persists across tasks."""
    return evaluate_agent_on_benchmark(
        agent=agent,
        tasks=tasks,
        max_cycles_per_task=max_cycles_per_task,
        mode="lifelong",
    )


def evaluate_three_tier_benchmark(
    agent: RecurrentSoftwareAgent,
    benchmark: Optional[ProceduralSoftwareBenchmark] = None,
    count_per_tier: int = 10,
    max_cycles_per_task: int = 4,
    mode: str = "zero_shot",
) -> Dict[str, BenchmarkEvaluationReport]:
    """Run comprehensive 3-tier capability evaluation across Lexical, Domain Transfer, and Sealed OOD families."""
    bench = benchmark or ProceduralSoftwareBenchmark(seed=42)

    print(f"\n>>> Running Level A (Lexical Variations, count={count_per_tier})...")
    tasks_a = bench.generate_level_a_tasks(count=count_per_tier)
    report_a = evaluate_agent_on_benchmark(agent, tasks_a, max_cycles_per_task=max_cycles_per_task, mode=mode)

    print(f"\n>>> Running Level B (Intra-Domain Transfer, count={count_per_tier})...")
    tasks_b = bench.generate_level_b_tasks(count=count_per_tier)
    report_b = evaluate_agent_on_benchmark(agent, tasks_b, max_cycles_per_task=max_cycles_per_task, mode=mode)

    print(f"\n>>> Running Level C (Permanently Sealed OOD Families, count={count_per_tier})...")
    tasks_c = bench.generate_level_c_tasks(count=count_per_tier)
    report_c = evaluate_agent_on_benchmark(agent, tasks_c, max_cycles_per_task=max_cycles_per_task, mode=mode)

    return {
        "level_a_lexical": report_a,
        "level_b_domain_transfer": report_b,
        "level_c_sealed_ood": report_c,
    }
