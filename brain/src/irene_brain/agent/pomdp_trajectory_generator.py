"""POMDP Software Engineering Trajectory Generator for Pseudo-Brain.

Generates realistic multi-step Action-Observation execution traces:
Goal -> Action (Write) -> Obs -> Action (Run Tests) -> Obs (Failure) ->
Action (Edit/Repair) -> Obs -> Action (Run Tests) -> Obs (Pass) -> Action (Finish).

These trajectories are fed into the infinite streaming pretraining pipeline
to train the recurrent neural weights directly on autonomous agent decision-making.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class POMDPTrajectory:
    """A full multi-step software engineering POMDP execution episode."""
    goal: str
    domain: str
    steps: List[Tuple[str, str]]  # (Action, Observation)
    final_status: str = "SUCCESS"

    def to_training_text(self, thread_id: int = 0) -> str:
        """Format the POMDP trajectory into pretraining sequence text with thread markers."""
        lines = [f"[THREAD:{thread_id}][POMDP_CODE] Goal: {self.goal}"]
        for act, obs in self.steps:
            lines.append(f"[RESP]{act}[EOS]")
            lines.append(f"{obs}")
        return "\n".join(lines)


class POMDPTrajectoryGenerator:
    """Infinite generator of diverse multi-file software engineering POMDP traces."""

    def __init__(self, seed: int = 1337):
        self.rng = random.Random(seed)

    def sample_trajectory(self) -> POMDPTrajectory:
        """Sample a multi-step POMDP software engineering trajectory."""
        domains = ["binary_search", "memoized_fibonacci", "priority_queue", "task_storage", "matrix_math"]
        choice = self.rng.choice(domains)

        if choice == "binary_search":
            return self._sample_binary_search_trajectory()
        elif choice == "memoized_fibonacci":
            return self._sample_fibonacci_trajectory()
        elif choice == "priority_queue":
            return self._sample_priority_queue_trajectory()
        elif choice == "task_storage":
            return self._sample_task_storage_trajectory()
        else:
            return self._sample_matrix_math_trajectory()

    def _sample_binary_search_trajectory(self) -> POMDPTrajectory:
        goal = "Implement a sorted array binary search algorithm with unit tests and error handling"
        steps = [
            (
                "ACTION: WRITE_FILE search.py\n"
                "def binary_search(arr, target):\n"
                "    left, right = 0, len(arr) - 1\n"
                "    while left <= right:\n"
                "        mid = (left + right) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        elif arr[mid] < target:\n"
                "            left = mid + 1\n"
                "        else:\n"
                "            right = mid - 1\n"
                "    return -1",
                "[OBSERVATION: Successfully wrote 240 bytes to search.py]"
            ),
            (
                "ACTION: WRITE_FILE tests/test_search.py\n"
                "from search import binary_search\n"
                "def test_bs():\n"
                "    assert binary_search([1, 3, 5, 7], 3) == 1\n"
                "    assert binary_search([1, 3, 5, 7], 9) == -1\n"
                "if __name__ == '__main__':\n"
                "    test_bs()\n"
                "    print('ALL TESTS PASSED')",
                "[OBSERVATION: Successfully wrote 180 bytes to tests/test_search.py]"
            ),
            (
                "ACTION: RUN_TESTS tests/test_search.py",
                "[OBSERVATION: RUN_TESTS PASSED:\n[tests/test_search.py PASSED]:\nALL TESTS PASSED]"
            ),
            (
                "ACTION: FINISH Implemented binary_search and verified 100% test pass rate",
                "[OBSERVATION: Task completed successfully. Summary: Implemented binary_search and verified 100% test pass rate]"
            )
        ]
        return POMDPTrajectory(goal=goal, domain="algorithms", steps=steps)

    def _sample_fibonacci_trajectory(self) -> POMDPTrajectory:
        goal = "Build a memoized Fibonacci generator with cache eviction and recursion limit safety"
        steps = [
            (
                "ACTION: RETRIEVE_MEMORY memoization in python",
                "[OBSERVATION: Retrieved Memory for 'memoization' (algorithms): Memoization caches function results to avoid recomputing expensive subproblems. Code: from functools import lru_cache]"
            ),
            (
                "ACTION: WRITE_FILE fib.py\n"
                "from functools import lru_cache\n"
                "@lru_cache(maxsize=128)\n"
                "def fib(n: int) -> int:\n"
                "    if n < 0: raise ValueError('Negative index not allowed')\n"
                "    if n < 2: return n\n"
                "    return fib(n - 1) + fib(n - 2)",
                "[OBSERVATION: Successfully wrote 190 bytes to fib.py]"
            ),
            (
                "ACTION: WRITE_FILE tests/test_fib.py\n"
                "from fib import fib\n"
                "def test_values():\n"
                "    assert fib(0) == 0\n"
                "    assert fib(10) == 55\n"
                "if __name__ == '__main__':\n"
                "    test_values()\n"
                "    print('FIBONACCI TESTS PASSED')",
                "[OBSERVATION: Successfully wrote 160 bytes to tests/test_fib.py]"
            ),
            (
                "ACTION: RUN_TESTS tests/test_fib.py",
                "[OBSERVATION: RUN_TESTS PASSED:\n[tests/test_fib.py PASSED]:\nFIBONACCI TESTS PASSED]"
            ),
            (
                "ACTION: FINISH Created memoized Fibonacci engine with tests passing",
                "[OBSERVATION: Task completed successfully. Summary: Created memoized Fibonacci engine with tests passing]"
            )
        ]
        return POMDPTrajectory(goal=goal, domain="algorithms", steps=steps)

    def _sample_priority_queue_trajectory(self) -> POMDPTrajectory:
        goal = "Implement a min-heap priority queue data structure with push, pop, and peek operations"
        steps = [
            (
                "ACTION: WRITE_FILE pq.py\n"
                "import heapq\n"
                "class PriorityQueue:\n"
                "    def __init__(self): self.heap = []\n"
                "    def push(self, item, priority): heapq.heappush(self.heap, (priority, item))\n"
                "    def pop(self): return heapq.heappop(self.heap)[1]\n"
                "    def peek(self): return self.heap[0][1] if self.heap else None",
                "[OBSERVATION: Successfully wrote 250 bytes to pq.py]"
            ),
            (
                "ACTION: WRITE_FILE tests/test_pq.py\n"
                "from pq import PriorityQueue\n"
                "def test_order():\n"
                "    q = PriorityQueue()\n"
                "    q.push('task_b', priority=2)\n"
                "    q.push('task_a', priority=1)\n"
                "    assert q.pop() == 'task_a'\n"
                "    assert q.pop() == 'task_b'\n"
                "if __name__ == '__main__':\n"
                "    test_order()\n"
                "    print('PQ TESTS PASSED')",
                "[OBSERVATION: Successfully wrote 240 bytes to tests/test_pq.py]"
            ),
            (
                "ACTION: RUN_TESTS tests/test_pq.py",
                "[OBSERVATION: RUN_TESTS PASSED:\n[tests/test_pq.py PASSED]:\nPQ TESTS PASSED]"
            ),
            (
                "ACTION: FINISH Priority queue verified and passing",
                "[OBSERVATION: Task completed successfully. Summary: Priority queue verified and passing]"
            )
        ]
        return POMDPTrajectory(goal=goal, domain="data_structures", steps=steps)

    def _sample_task_storage_trajectory(self) -> POMDPTrajectory:
        goal = "Create a modular task store with auto-increment IDs and completion tracking"
        steps = [
            (
                "ACTION: WRITE_FILE storage.py\n"
                "class Store:\n"
                "    def __init__(self): self.items = {}; self._id = 1\n"
                "    def add(self, title): t = {'id': self._id, 'title': title, 'done': False}; self.items[self._id] = t; self._id += 1; return t\n"
                "    def complete(self, tid): self.items[tid]['done'] = True",
                "[OBSERVATION: Successfully wrote 210 bytes to storage.py]"
            ),
            (
                "ACTION: WRITE_FILE tests/test_store.py\n"
                "from storage import Store\n"
                "def test_store():\n"
                "    s = Store()\n"
                "    t = s.add('Fix bug')\n"
                "    s.complete(t['id'])\n"
                "    assert s.items[1]['done'] is True\n"
                "if __name__ == '__main__':\n"
                "    test_store()\n"
                "    print('STORE TESTS PASSED')",
                "[OBSERVATION: Successfully wrote 220 bytes to tests/test_store.py]"
            ),
            (
                "ACTION: RUN_TESTS tests/test_store.py",
                "[OBSERVATION: RUN_TESTS PASSED:\n[tests/test_store.py PASSED]:\nSTORE TESTS PASSED]"
            ),
            (
                "ACTION: FINISH Store module operational and tested",
                "[OBSERVATION: Task completed successfully. Summary: Store module operational and tested]"
            )
        ]
        return POMDPTrajectory(goal=goal, domain="utilities", steps=steps)

    def _sample_matrix_math_trajectory(self) -> POMDPTrajectory:
        goal = "Build a matrix dot product calculation package with dimension assertions"
        steps = [
            (
                "ACTION: WRITE_FILE matrix.py\n"
                "def dot(m1, m2):\n"
                "    r1, c1 = len(m1), len(m1[0])\n"
                "    r2, c2 = len(m2), len(m2[0])\n"
                "    assert c1 == r2, f'Dim mismatch: {c1} != {r2}'\n"
                "    return [[sum(m1[i][k] * m2[k][j] for k in range(c1)) for j in range(c2)] for i in range(r1)]",
                "[OBSERVATION: Successfully wrote 230 bytes to matrix.py]"
            ),
            (
                "ACTION: WRITE_FILE tests/test_matrix.py\n"
                "from matrix import dot\n"
                "def test_dot():\n"
                "    a = [[1, 2], [3, 4]]\n"
                "    b = [[2, 0], [1, 2]]\n"
                "    res = dot(a, b)\n"
                "    assert res == [[4, 4], [10, 8]]\n"
                "if __name__ == '__main__':\n"
                "    test_dot()\n"
                "    print('MATRIX DOT TESTS PASSED')",
                "[OBSERVATION: Successfully wrote 220 bytes to tests/test_matrix.py]"
            ),
            (
                "ACTION: RUN_TESTS tests/test_matrix.py",
                "[OBSERVATION: RUN_TESTS PASSED:\n[tests/test_matrix.py PASSED]:\nMATRIX DOT TESTS PASSED]"
            ),
            (
                "ACTION: FINISH Matrix dot multiplication package verified",
                "[OBSERVATION: Task completed successfully. Summary: Matrix dot multiplication package verified]"
            )
        ]
        return POMDPTrajectory(goal=goal, domain="math", steps=steps)
