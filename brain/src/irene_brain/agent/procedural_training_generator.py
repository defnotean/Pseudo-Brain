"""Open-World Procedural Task & Trajectory Generator for Pseudo-Brain Training.

Generates hundreds of procedural algorithmic challenges across 8 open domains with
randomized function names, module names, variable names, and signatures.
These tasks form the training distribution, completely disjoint from the 10 sealed
held-out families.
"""

from __future__ import annotations

import random
from typing import List, Dict, Tuple
from dataclasses import dataclass

from irene_brain.agent.procedural_evaluator import ProceduralTask


class ProceduralTrainingGenerator:
    """Generates varied algorithmic challenges across 8 open domains with randomized naming."""

    def __init__(self, seed: int = 1337):
        self.rng = random.Random(seed)

    def generate_training_tasks(self, count: int = 60) -> List[ProceduralTask]:
        """Generate `count` varied procedural tasks across the 8 open domains."""
        generators = [
            self._gen_string_task,
            self._gen_numeric_task,
            self._gen_sorting_searching_task,
            self._gen_lists_sequences_task,
            self._gen_dict_maps_task,
            self._gen_validation_task,
            self._gen_queues_stacks_task,
            self._gen_math_aggregates_task,
        ]
        tasks = []
        for i in range(count):
            gen = generators[i % len(generators)]
            tasks.append(gen(task_idx=i))
        return tasks

    # -------------------------------------------------------------------------
    # Domain 1: String Manipulation
    # -------------------------------------------------------------------------
    def _gen_string_task(self, task_idx: int) -> ProceduralTask:
        subtypes = ["capitalize_words", "count_vowels", "strip_punctuation", "is_palindrome", "reverse_words"]
        stype = self.rng.choice(subtypes)
        mod_name = f"str_ops_{task_idx}.py"

        if stype == "capitalize_words":
            fn_name = f"capitalize_words_{task_idx}"
            goal = f"Implement `{fn_name}(s: str) -> str` in `{mod_name}` capitalizing the first letter of every word."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}("hello world") == "Hello World"
assert {fn_name}("python code") == "Python Code"
assert {fn_name}("") == ""
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(s: str) -> str:
    return " ".join(word.capitalize() for word in s.split(" ")) if s else ""
"""
        elif stype == "count_vowels":
            fn_name = f"count_vowels_{task_idx}"
            goal = f"Implement `{fn_name}(s: str) -> int` in `{mod_name}` counting lowercase and uppercase vowels (a, e, i, o, u)."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}("Hello World") == 3
assert {fn_name}("AEIOU xyz") == 5
assert {fn_name}("bcdfgh") == 0
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(s: str) -> int:
    vowels = set("aeiouAEIOU")
    return sum(1 for ch in s if ch in vowels)
"""
        elif stype == "strip_punctuation":
            fn_name = f"clean_text_{task_idx}"
            goal = f"Implement `{fn_name}(s: str) -> str` in `{mod_name}` removing characters in '!?,.:;' from `s`."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}("Hello, world!") == "Hello world"
assert {fn_name}("What? No: never;") == "What No never"
assert {fn_name}("Clean text") == "Clean text"
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(s: str) -> str:
    punct = set("!?,.:;")
    return "".join(ch for ch in s if ch not in punct)
"""
        elif stype == "is_palindrome":
            fn_name = f"is_palindrome_{task_idx}"
            goal = f"Implement `{fn_name}(s: str) -> bool` in `{mod_name}` checking if string `s` is a case-insensitive palindrome."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}("Radar") is True
assert {fn_name}("hello") is False
assert {fn_name}("") is True
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(s: str) -> bool:
    clean = s.lower()
    return clean == clean[::-1]
"""
        else:  # reverse_words
            fn_name = f"reverse_words_{task_idx}"
            goal = f"Implement `{fn_name}(s: str) -> str` in `{mod_name}` reversing order of words separated by whitespace."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}("quick brown fox") == "fox brown quick"
assert {fn_name}("single") == "single"
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(s: str) -> str:
    words = s.strip().split()
    return " ".join(reversed(words))
"""

        return ProceduralTask(
            task_id=f"str_{task_idx}",
            domain="string_manipulation",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden.strip(),
            reference_solution=ref.strip(),
        )

    # -------------------------------------------------------------------------
    # Domain 2: Numeric Transforms
    # -------------------------------------------------------------------------
    def _gen_numeric_task(self, task_idx: int) -> ProceduralTask:
        subtypes = ["is_prime", "sum_of_digits", "factorial", "collatz_steps", "clamp_number"]
        stype = self.rng.choice(subtypes)
        mod_name = f"num_algo_{task_idx}.py"

        if stype == "is_prime":
            fn_name = f"is_prime_{task_idx}"
            goal = f"Write `{fn_name}(n: int) -> bool` in `{mod_name}` determining if integer `n >= 2` is prime."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}(7) is True
assert {fn_name}(4) is False
assert {fn_name}(1) is False
assert {fn_name}(13) is True
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
"""
        elif stype == "sum_of_digits":
            fn_name = f"sum_digits_{task_idx}"
            goal = f"Write `{fn_name}(n: int) -> int` in `{mod_name}` returning sum of absolute digits of integer `n`."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}(1234) == 10
assert {fn_name}(-45) == 9
assert {fn_name}(0) == 0
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(n: int) -> int:
    return sum(int(ch) for ch in str(abs(n)))
"""
        elif stype == "factorial":
            fn_name = f"compute_fact_{task_idx}"
            goal = f"Write `{fn_name}(n: int) -> int` in `{mod_name}` computing factorial of non-negative integer `n`."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}(0) == 1
assert {fn_name}(4) == 24
assert {fn_name}(5) == 120
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(n: int) -> int:
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res
"""
        elif stype == "collatz_steps":
            fn_name = f"collatz_len_{task_idx}"
            goal = f"Write `{fn_name}(n: int) -> int` in `{mod_name}` computing number of steps for `n >= 1` to reach 1 via Collatz rules."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}(1) == 0
assert {fn_name}(2) == 1
assert {fn_name}(6) == 8
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(n: int) -> int:
    steps = 0
    while n > 1:
        n = (n // 2) if n % 2 == 0 else (3 * n + 1)
        steps += 1
    return steps
"""
        else:  # clamp_number
            fn_name = f"clamp_val_{task_idx}"
            goal = f"Write `{fn_name}(val: float, low: float, high: float) -> float` in `{mod_name}` clamping `val` to `[low, high]`."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}(5.0, 1.0, 10.0) == 5.0
assert {fn_name}(-2.0, 0.0, 5.0) == 0.0
assert {fn_name}(12.0, 0.0, 10.0) == 10.0
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))
"""

        return ProceduralTask(
            task_id=f"num_{task_idx}",
            domain="numeric_transforms",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden.strip(),
            reference_solution=ref.strip(),
        )

    # -------------------------------------------------------------------------
    # Domain 3: Sorting & Searching
    # -------------------------------------------------------------------------
    def _gen_sorting_searching_task(self, task_idx: int) -> ProceduralTask:
        subtypes = ["bubble_sort", "linear_search", "find_min_max", "is_sorted"]
        stype = self.rng.choice(subtypes)
        mod_name = f"search_sort_{task_idx}.py"

        if stype == "bubble_sort":
            fn_name = f"bubble_sort_{task_idx}"
            goal = f"Implement `{fn_name}(items: list[int]) -> list[int]` in `{mod_name}` returning a new list sorted in ascending order."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]
assert {fn_name}([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
assert {fn_name}([]) == []
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(items: list[int]) -> list[int]:
    arr = list(items)
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""
        elif stype == "linear_search":
            fn_name = f"find_first_index_{task_idx}"
            goal = f"Implement `{fn_name}(items: list, target: any) -> int` in `{mod_name}` returning first index of `target`, or -1."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([10, 20, 30, 40], 30) == 2
assert {fn_name}([10, 20], 99) == -1
assert {fn_name}([], 5) == -1
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(items: list, target) -> int:
    for i, x in enumerate(items):
        if x == target:
            return i
    return -1
"""
        elif stype == "find_min_max":
            fn_name = f"get_min_max_{task_idx}"
            goal = f"Implement `{fn_name}(nums: list[float]) -> tuple[float, float] | None` in `{mod_name}` returning (min, max), or None if empty."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([3.0, 1.0, 9.0, 2.0]) == (1.0, 9.0)
assert {fn_name}([42.0]) == (42.0, 42.0)
assert {fn_name}([]) is None
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(nums: list[float]):
    if not nums:
        return None
    return (min(nums), max(nums))
"""
        else:  # is_sorted
            fn_name = f"is_ascending_{task_idx}"
            goal = f"Implement `{fn_name}(items: list[int]) -> bool` in `{mod_name}` returning True if items are sorted non-decreasingly."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([1, 2, 2, 5]) is True
assert {fn_name}([1, 3, 2]) is False
assert {fn_name}([]) is True
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(items: list[int]) -> bool:
    return all(items[i] <= items[i+1] for i in range(len(items) - 1))
"""

        return ProceduralTask(
            task_id=f"sort_{task_idx}",
            domain="sorting_searching",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden.strip(),
            reference_solution=ref.strip(),
        )

    # -------------------------------------------------------------------------
    # Domain 4: Lists & Sequences
    # -------------------------------------------------------------------------
    def _gen_lists_sequences_task(self, task_idx: int) -> ProceduralTask:
        subtypes = ["flatten_list", "chunk_list", "cumulative_sum", "remove_item"]
        stype = self.rng.choice(subtypes)
        mod_name = f"seq_utils_{task_idx}.py"

        if stype == "flatten_list":
            fn_name = f"flatten_nested_{task_idx}"
            goal = f"Implement `{fn_name}(nested: list[list]) -> list` in `{mod_name}` flattening a list of 1-level nested sublists."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]
assert {fn_name}([[], [1], []]) == [1]
assert {fn_name}([]) == []
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(nested: list[list]) -> list:
    out = []
    for sub in nested:
        out.extend(sub)
    return out
"""
        elif stype == "chunk_list":
            fn_name = f"chunk_sequence_{task_idx}"
            goal = f"Implement `{fn_name}(items: list, size: int) -> list[list]` in `{mod_name}` partitioning `items` into chunks of given `size`."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
assert {fn_name}([10, 20], 5) == [[10, 20]]
assert {fn_name}([], 2) == []
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(items: list, size: int) -> list[list]:
    if size <= 0:
        return []
    return [items[i:i + size] for i in range(0, len(items), size)]
"""
        elif stype == "cumulative_sum":
            fn_name = f"prefix_sums_{task_idx}"
            goal = f"Implement `{fn_name}(nums: list[int]) -> list[int]` in `{mod_name}` returning running cumulative sums."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([1, 2, 3, 4]) == [1, 3, 6, 10]
assert {fn_name}([5]) == [5]
assert {fn_name}([]) == []
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(nums: list[int]) -> list[int]:
    out = []
    curr = 0
    for x in nums:
        curr += x
        out.append(curr)
    return out
"""
        else:  # remove_item
            fn_name = f"remove_all_{task_idx}"
            goal = f"Implement `{fn_name}(items: list, target: any) -> list` in `{mod_name}` returning new list without instances of `target`."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([1, 2, 3, 2, 4], 2) == [1, 3, 4]
assert {fn_name}(["a", "b"], "z") == ["a", "b"]
assert {fn_name}([], 1) == []
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(items: list, target) -> list:
    return [x for x in items if x != target]
"""

        return ProceduralTask(
            task_id=f"seq_{task_idx}",
            domain="lists_sequences",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden.strip(),
            reference_solution=ref.strip(),
        )

    # -------------------------------------------------------------------------
    # Domain 5: Dictionaries & Maps
    # -------------------------------------------------------------------------
    def _gen_dict_maps_task(self, task_idx: int) -> ProceduralTask:
        subtypes = ["frequency_map", "invert_dict", "merge_dict_sums"]
        stype = self.rng.choice(subtypes)
        mod_name = f"dict_ops_{task_idx}.py"

        if stype == "frequency_map":
            fn_name = f"count_frequencies_{task_idx}"
            goal = f"Implement `{fn_name}(items: list[str]) -> dict[str, int]` in `{mod_name}` returning frequency count of each string."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}(["apple", "banana", "apple"]) == {{"apple": 2, "banana": 1}}
assert {fn_name}([]) == {{}}
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(items: list[str]) -> dict[str, int]:
    counts = {{}}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts
"""
        elif stype == "invert_dict":
            fn_name = f"invert_mapping_{task_idx}"
            goal = f"Implement `{fn_name}(d: dict[str, int]) -> dict[int, str]` in `{mod_name}` inverting key-value pairs (values are unique)."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}({{"a": 1, "b": 2}}) == {{1: "a", 2: "b"}}
assert {fn_name}({{}}) == {{}}
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(d: dict) -> dict:
    return {{v: k for k, v in d.items()}}
"""
        else:  # merge_dict_sums
            fn_name = f"combine_counts_{task_idx}"
            goal = f"Implement `{fn_name}(d1: dict[str, int], d2: dict[str, int]) -> dict[str, int]` in `{mod_name}` merging dicts summing matching keys."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}({{"a": 2, "b": 3}}, {{"b": 4, "c": 5}}) == {{"a": 2, "b": 7, "c": 5}}
assert {fn_name}({{}}, {{"x": 1}}) == {{"x": 1}}
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(d1: dict[str, int], d2: dict[str, int]) -> dict[str, int]:
    out = dict(d1)
    for k, v in d2.items():
        out[k] = out.get(k, 0) + v
    return out
"""

        return ProceduralTask(
            task_id=f"map_{task_idx}",
            domain="dictionaries_maps",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden.strip(),
            reference_solution=ref.strip(),
        )

    # -------------------------------------------------------------------------
    # Domain 6: Logic & Validation
    # -------------------------------------------------------------------------
    def _gen_validation_task(self, task_idx: int) -> ProceduralTask:
        subtypes = ["is_balanced_parens", "is_valid_identifier", "in_bounds"]
        stype = self.rng.choice(subtypes)
        mod_name = f"validator_{task_idx}.py"

        if stype == "is_balanced_parens":
            fn_name = f"check_parens_{task_idx}"
            goal = f"Write `{fn_name}(s: str) -> bool` in `{mod_name}` checking if parentheses '(' and ')' in `s` are properly balanced."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}("((a + b) * c)") is True
assert {fn_name}(")((") is False
assert {fn_name}("") is True
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0
"""
        elif stype == "is_valid_identifier":
            fn_name = f"validate_name_{task_idx}"
            goal = f"Write `{fn_name}(name: str) -> bool` in `{mod_name}` checking if `name` is a valid non-empty alphanumeric identifier starting with a letter or underscore."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}("valid_var_1") is True
assert {fn_name}("123bad") is False
assert {fn_name}("") is False
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == '_'):
        return False
    return all(ch.isalnum() or ch == '_' for ch in name[1:])
"""
        else:  # in_bounds
            fn_name = f"is_in_bounds_{task_idx}"
            goal = f"Write `{fn_name}(r: int, c: int, num_rows: int, num_cols: int) -> bool` in `{mod_name}` checking if 0 <= r < num_rows and 0 <= c < num_cols."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}(0, 0, 3, 3) is True
assert {fn_name}(3, 2, 3, 3) is False
assert {fn_name}(-1, 0, 5, 5) is False
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(r: int, c: int, num_rows: int, num_cols: int) -> bool:
    return 0 <= r < num_rows and 0 <= c < num_cols
"""

        return ProceduralTask(
            task_id=f"val_{task_idx}",
            domain="logic_validation",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden.strip(),
            reference_solution=ref.strip(),
        )

    # -------------------------------------------------------------------------
    # Domain 7: Stacks & Queues
    # -------------------------------------------------------------------------
    def _gen_queues_stacks_task(self, task_idx: int) -> ProceduralTask:
        mod_name = f"queue_util_{task_idx}.py"
        cls_name = f"SimpleQueue_{task_idx}"
        goal = (
            f"Implement class `{cls_name}` in `{mod_name}` with `__init__(self)`, `push(self, item)`, "
            "`pop(self)` (returns None if empty), and `is_empty(self) -> bool`."
        )
        hidden = f"""
from {mod_name.replace('.py', '')} import {cls_name}
q = {cls_name}()
assert q.is_empty() is True
q.push(10)
q.push(20)
assert q.is_empty() is False
assert q.pop() == 10
assert q.pop() == 20
assert q.pop() is None
print("HIDDEN TESTS PASSED")
"""
        ref = f"""
class {cls_name}:
    def __init__(self):
        self._items = []

    def push(self, item) -> None:
        self._items.append(item)

    def pop(self):
        if not self._items:
            return None
        return self._items.pop(0)

    def is_empty(self) -> bool:
        return len(self._items) == 0
"""
        return ProceduralTask(
            task_id=f"queue_{task_idx}",
            domain="queues_stacks",
            goal=goal,
            target_module=mod_name,
            target_function=cls_name,
            initial_files={},
            hidden_tests_code=hidden.strip(),
            reference_solution=ref.strip(),
        )

    # -------------------------------------------------------------------------
    # Domain 8: Mathematical Aggregates
    # -------------------------------------------------------------------------
    def _gen_math_aggregates_task(self, task_idx: int) -> ProceduralTask:
        subtypes = ["dot_product", "transpose_matrix", "mean_std"]
        stype = self.rng.choice(subtypes)
        mod_name = f"matrix_math_{task_idx}.py"

        if stype == "dot_product":
            fn_name = f"vector_dot_{task_idx}"
            goal = f"Implement `{fn_name}(v1: list[float], v2: list[float]) -> float` in `{mod_name}` returning Euclidean dot product."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0
assert {fn_name}([], []) == 0.0
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(v1: list[float], v2: list[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))
"""
        elif stype == "transpose_matrix":
            fn_name = f"transpose_2d_{task_idx}"
            goal = f"Implement `{fn_name}(grid: list[list[int]]) -> list[list[int]]` in `{mod_name}` transposing a rectangular 2D matrix."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
assert {fn_name}([]) == []
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(grid: list[list[int]]) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    return [[grid[r][c] for r in range(len(grid))] for c in range(len(grid[0]))]
"""
        else:  # mean_std
            fn_name = f"mean_variance_{task_idx}"
            goal = f"Implement `{fn_name}(vals: list[float]) -> tuple[float, float] | None` in `{mod_name}` returning (mean, sample_variance). Return None if len < 2."
            hidden = f"""
from {mod_name.replace('.py', '')} import {fn_name}
assert {fn_name}([1.0]) is None
m, v = {fn_name}([10.0, 20.0, 30.0])
assert abs(m - 20.0) < 1e-6
assert abs(v - 100.0) < 1e-6
print("HIDDEN TESTS PASSED")
"""
            ref = f"""
def {fn_name}(vals: list[float]):
    n = len(vals)
    if n < 2:
        return None
    mean = sum(vals) / n
    variance = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return (mean, variance)
"""

        return ProceduralTask(
            task_id=f"math_{task_idx}",
            domain="mathematical_aggregates",
            goal=goal,
            target_module=mod_name,
            target_function=fn_name,
            initial_files={},
            hidden_tests_code=hidden.strip(),
            reference_solution=ref.strip(),
        )

    # -------------------------------------------------------------------------
    # Multi-Turn Trajectory Synthesizer (Failure Reflection & Self-Repair)
    # -------------------------------------------------------------------------
    def generate_multiturn_pomdp_trajectories(self, tasks: List[ProceduralTask]) -> List[str]:
        """Synthesize multi-turn POMDP episodes covering clean code, syntax repair, test repair, and retrieval."""
        episodes = []
        for i, t in enumerate(tasks):
            p_type = i % 4  # 0: clean, 1: syntax_repair, 2: test_repair, 3: retrieval

            header = (
                f"[GOAL: {t.goal}]\n"
                f"[TARGET_MODULE: {t.target_module}]\n"
                f"[TARGET_FUNCTION: {t.target_function}]\n"
            )

            if p_type == 0:
                # Pattern 1: Direct Clean Implementation (Single-shot success)
                ep = (
                    f"{header}"
                    f"[PHASE: WRITE_CODE]\n"
                    f"[RESP]ACTION: WRITE_FILE {t.target_module}\n{t.reference_solution}[EOS]\n"
                    f"[OBSERVATION: Successfully wrote {len(t.reference_solution)} bytes to {t.target_module}]\n"
                    f"[PHASE: VERIFY_AND_FINISH]\n"
                    f"[RESP]ACTION: FINISH Verified implementation of {t.target_function}[EOS]"
                )
            elif p_type == 1:
                # Pattern 2: Syntax Error Reflection & Self-Repair
                # Deliberately omit colon after def line
                ref_lines = t.reference_solution.splitlines()
                buggy_lines = []
                for line in ref_lines:
                    if line.strip().startswith("def ") and line.endswith(":"):
                        buggy_lines.append(line[:-1])  # Missing colon syntax bug!
                    else:
                        buggy_lines.append(line)
                buggy_code = "\n".join(buggy_lines)

                ep = (
                    f"{header}"
                    f"[PHASE: WRITE_CODE]\n"
                    f"[RESP]ACTION: WRITE_FILE {t.target_module}\n{buggy_code}[EOS]\n"
                    f"[OBSERVATION: WRITE_FILE failed: SyntaxError at line 1, col 24: expected ':']\n"
                    f"[PHASE: REPAIR_SYNTAX]\n"
                    f"[RESP]ACTION: WRITE_FILE {t.target_module}\n{t.reference_solution}[EOS]\n"
                    f"[OBSERVATION: Successfully wrote {len(t.reference_solution)} bytes to {t.target_module}]\n"
                    f"[PHASE: VERIFY_AND_FINISH]\n"
                    f"[RESP]ACTION: FINISH Verified implementation of {t.target_function}[EOS]"
                )
            elif p_type == 2:
                # Pattern 3: Test Failure & Logic Self-Repair
                stub_code = f"def {t.target_function}(*args, **kwargs):\n    return None\n"
                ep = (
                    f"{header}"
                    f"[PHASE: WRITE_CODE]\n"
                    f"[RESP]ACTION: WRITE_FILE {t.target_module}\n{stub_code}[EOS]\n"
                    f"[OBSERVATION: Successfully wrote {len(stub_code)} bytes to {t.target_module}]\n"
                    f"[PHASE: RUN_TESTS]\n"
                    f"[RESP]ACTION: RUN_TESTS\n"
                    f"[OBSERVATION: Tests failed: AssertionError at line 3: expected non-None result]\n"
                    f"[PHASE: REPAIR_LOGIC]\n"
                    f"[RESP]ACTION: WRITE_FILE {t.target_module}\n{t.reference_solution}[EOS]\n"
                    f"[OBSERVATION: Successfully wrote {len(t.reference_solution)} bytes to {t.target_module}]\n"
                    f"[PHASE: VERIFY_AND_FINISH]\n"
                    f"[RESP]ACTION: FINISH Verified implementation of {t.target_function}[EOS]"
                )
            else:
                # Pattern 4: Research / Memory Retrieval Assisted Acquisition
                doc_concept = f"Algorithm specification for {t.target_function}: implement logic for {t.domain}."
                ep = (
                    f"{header}"
                    f"[PHASE: EXPLORE]\n"
                    f"[RESP]ACTION: RETRIEVE_MEMORY {t.target_function} algorithm\n"
                    f"[OBSERVATION: {doc_concept}]\n"
                    f"[PHASE: WRITE_CODE]\n"
                    f"[RESP]ACTION: WRITE_FILE {t.target_module}\n{t.reference_solution}[EOS]\n"
                    f"[OBSERVATION: Successfully wrote {len(t.reference_solution)} bytes to {t.target_module}]\n"
                    f"[PHASE: VERIFY_AND_FINISH]\n"
                    f"[RESP]ACTION: FINISH Verified implementation of {t.target_function}[EOS]"
                )

            episodes.append(ep)

        return episodes

