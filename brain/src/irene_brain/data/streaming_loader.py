"""Infinite-Streaming Multi-Task Pretraining Data Pipeline for Pseudo-Brain.

Features:
1. High-throughput streaming dataset iterator using Hugging Face `datasets` (streaming=True)
   or HTTP stream fallback.
2. Resilient on-the-fly tokenization with fallback to offline generated high-density
   synthetic multi-task corpus if internet/API rate limits occur.
3. Configurable multi-task mixture:
   - Language (default 40%): FineWeb-Edu, SlimPajama, Wikipedia, multi-turn dialogues
   - Code (default 35%): StarCoder, The Stack Smol, algorithmic Python snippets
   - Reasoning/Tools (default 15%): Procedural tool-calling traces, DAGs, step-by-step reasoning
   - Game POMDP (default 10%): Sensory arcade, maze chase, occlusion POMDP control sequences
4. Sequence Packing: Packs variable-length items into uniform sequence blocks [B, T]
   (e.g., T=256, 512, 1024) with cognitive thread ID markers ([THREAD:0..K-1])
   and proper loss masking (boundary isolation, next-token prediction, assistant-only option).
5. Efficient PyTorch DataLoader / IterableDataset generator ready for Google Colab A100 consumption.
6. Zero memory leaks across infinite streaming iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import gc
import json
import logging
import math
import random
import sys
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

logger = logging.getLogger("irene_brain.data.streaming_loader")

try:
    import datasets
    _HF_DATASETS_AVAILABLE = True
except ImportError:
    datasets = None
    _HF_DATASETS_AVAILABLE = False


class TaskType(str, Enum):
    """Supported pretraining data stream modalities."""
    LANGUAGE = "language"
    CODE = "code"
    REASONING = "reasoning"
    GAME_POMDP = "game_pomdp"


@dataclass
class TaskWeights:
    """Configurable sampling weights across pretraining tasks."""
    language: float = 0.40
    code: float = 0.35
    reasoning: float = 0.15
    game_pomdp: float = 0.10

    def to_dict(self) -> Dict[TaskType, float]:
        raw = {
            TaskType.LANGUAGE: float(self.language),
            TaskType.CODE: float(self.code),
            TaskType.REASONING: float(self.reasoning),
            TaskType.GAME_POMDP: float(self.game_pomdp),
        }
        total = sum(raw.values())
        if total <= 0:
            raise ValueError(f"Sum of task weights must be positive, got {total}")
        return {k: v / total for k, v in raw.items()}


@dataclass
class StreamingConfig:
    """Configuration for infinite streaming pretraining data pipeline."""
    seq_len: int = 512
    batch_size: int = 16
    weights: TaskWeights = field(default_factory=TaskWeights)
    max_threads: int = 64
    supervised_only_loss: bool = True
    mask_cross_document_boundary: bool = True
    seed: int = 42
    offline_mode: bool = False
    buffer_size: int = 1000
    hf_dataset_names: Dict[TaskType, str] = field(default_factory=lambda: {
        TaskType.LANGUAGE: "HuggingFaceH4/ultrachat_200k",
        TaskType.CODE: "bigcode/the-stack-smol",
        TaskType.REASONING: "open-r1/OpenR1-Math-220k",
        TaskType.GAME_POMDP: "synthetic_pomdp",
    })
    hf_dataset_splits: Dict[TaskType, str] = field(default_factory=lambda: {
        TaskType.LANGUAGE: "train_sft",
        TaskType.CODE: "train",
        TaskType.REASONING: "train",
        TaskType.GAME_POMDP: "train",
    })


@dataclass
class RawDocument:
    """A single raw textual item from a stream before packing."""
    text: str
    task_type: TaskType
    thread_id: int = 0
    is_dialogue: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PackedSequenceBlock:
    """A single packed sequence block of exact length T."""
    input_ids: torch.Tensor       # [T] long
    labels: torch.Tensor          # [T] long (-100 where masked)
    loss_mask: torch.Tensor       # [T] float32 (1.0 where active, 0.0 where masked)
    thread_ids: torch.Tensor      # [T] long (cognitive thread slot per token)
    task_types: List[str]         # list of tasks contributing to this packed block
    reset_mask: Optional[torch.Tensor] = None  # [T] float32 (1.0 at doc start, 0.0 elsewhere)


@dataclass
class StreamingBatch:
    """Batched packed sequences ready for A100 / GPU consumption."""
    input_ids: torch.Tensor       # [B, T] long
    labels: torch.Tensor          # [B, T] long (-100 for ignored positions)
    loss_mask: torch.Tensor       # [B, T] float32
    thread_ids: torch.Tensor      # [B, T] long
    task_types: List[List[str]]   # [B] list of task strings
    reset_mask: Optional[torch.Tensor] = None  # [B, T] float32

    def to(self, device: Union[str, torch.device], non_blocking: bool = True) -> StreamingBatch:
        """Transfer all tensors to the target device."""
        return StreamingBatch(
            input_ids=self.input_ids.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
            loss_mask=self.loss_mask.to(device, non_blocking=non_blocking),
            thread_ids=self.thread_ids.to(device, non_blocking=non_blocking),
            task_types=self.task_types,
            reset_mask=self.reset_mask.to(device, non_blocking=non_blocking) if self.reset_mask is not None else None,
        )

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access batch['input_ids'] for standard trainer loops."""
        if key == "input_ids":
            return self.input_ids
        elif key == "labels":
            return self.labels
        elif key == "loss_mask":
            return self.loss_mask
        elif key == "thread_ids":
            return self.thread_ids
        elif key == "task_types":
            return self.task_types
        elif key == "reset_mask":
            return self.reset_mask
        raise KeyError(f"Key {key} not found in StreamingBatch")

    def keys(self) -> List[str]:
        return ["input_ids", "labels", "loss_mask", "thread_ids", "task_types", "reset_mask"]


# ==============================================================================
# High-Density Offline Multi-Task Corpus Generators
# ==============================================================================

class SyntheticLanguageStream:
    """Infinite generator of high-density knowledge and multi-turn dialogues across science, tech, and humanities."""

    def __init__(self, seed: int = 42, max_threads: int = 64):
        self.rng = random.Random(seed)
        self.max_threads = max_threads

        self.topics = [
            # Physics & Thermodynamics
            ("Thermodynamic Entropy", "Entropy quantifies the multiplicity of accessible microscopic quantum states for a given macroscopic thermodynamic state (S = k_B * ln(Omega)). In isolated systems, spontaneous processes irreversibly increase total entropy."),
            ("Quantum Decoherence", "Quantum decoherence represents the non-unitary phase degradation of quantum superpositions due to environmental entanglement, causing microscopic wavefunctions to appear as classical probability mixtures."),
            ("General Relativity", "Einstein's field equations relate spacetime metric curvature (G_uv) directly to the stress-energy tensor (T_uv), explaining gravitation as geometric geodesics rather than Newtonian forces."),
            ("Electromagnetic Maxwell Equations", "Maxwell's four partial differential equations unify electric and magnetic phenomena, establishing that oscillating perpendicular vectors propagate self-sustaining transverse electromagnetic waves at the speed of light c."),
            ("Quantum Superposition", "Superposition allows quantum wavefunctions psi to exist as linear combinations of orthogonal eigenstates until projection operator measurement collapses them according to the Born probability rule."),
            ("Heisenberg Uncertainty Principle", "The non-commutativity of conjugate observables (such as position x and momentum p, [x, p] = i*hbar) enforces a fundamental lower bound Delta x * Delta p >= hbar / 2 on simultaneous measurement precision."),
            ("Special Relativity & Time Dilation", "Moving frames of reference observe kinematic time dilation Delta t' = gamma * Delta t where gamma = 1 / sqrt(1 - v^2 / c^2), ensuring the invariant speed of light across all inertial observers."),
            ("Black Hole Event Horizons", "The Schwarzschild radius r_s = 2*G*M / c^2 defines the null boundary beyond which escape velocity exceeds c, terminating timelike geodesics at the central coordinate singularity."),

            # Biology & Biochemistry
            ("Cellular Respiration & ATP Synthase", "Mitochondrial oxidative phosphorylation oxidizes pyruvate via the Krebs cycle and passes electrons along complexes I-IV, pumping protons across the inner membrane to rotate the F_0/F_1 ATP synthase motor."),
            ("CRISPR-Cas9 Gene Editing", "The bacterial adaptive immune endonuclease Cas9 forms a ribonucleoprotein complex with synthetic single guide RNA (sgRNA) to generate targeted double-strand breaks at specific genomic loci containing protospacer adjacent motifs (PAM)."),
            ("DNA Polymerase & Replication", "Replication proceeds bidirectionally from origins where helicase unwinds double-stranded DNA, allowing DNA polymerase delta and epsilon to synthesize leading strands continuously and lagging strands via Okazaki fragments."),
            ("Synaptic Plasticity & LTP", "Long-Term Potentiation (LTP) at glutamatergic synapses involves NMDA receptor calcium influx, triggering CaMKII signaling cascades that recruit additional AMPA receptors to the postsynaptic density."),
            ("Photosynthetic Electron Transport", "Photosystems II and I absorb photons to excite chlorophyll P680 and P700, photolyzing water into oxygen and pumping protons into the thylakoid lumen to generate NADPH and ATP."),

            # Computer Science & Mathematics
            ("Recurrent Working Memory", "Recurrent cognitive architectures maintain localized thought vectors across time intervals, decoupling sequence modeling depth from sequence length without quadratic attention cache explosion."),
            ("Topological Sorting", "Topological sorting constructs a linear ordering of vertices in a directed acyclic graph (DAG) such that for every directed edge (u, v), u precedes v, executable in O(V + E) linear time."),
            ("Shannon Information Theory", "Shannon entropy H(X) = -sum(p(x) * log2(p(x))) defines the minimum expected code length per symbol under optimal lossless prefix-free coding over stochastic channels."),
            ("Dijkstra Shortest Path Algorithm", "Dijkstra's algorithm finds shortest paths from a single source vertex to all other vertices in non-negative weighted graphs in O((V + E) log V) time using a binary or Fibonacci min-heap."),
            ("Compiler Optimization & SSA Form", "Static Single Assignment (SSA) form guarantees that each variable is assigned exactly once, simplifying dominance frontiers, global dead-code elimination, and common subexpression elimination."),
            ("B-Trees and Database Indexing", "B-trees are self-balancing search trees designed for block-storage systems that maintain sorted key-value pairs with logarithmic search, insertion, and deletion bounds (O(log_B N))."),
            ("Asymmetric Public-Key Cryptography", "RSA and Elliptic Curve Cryptography (ECC) exploit computationally intractable one-way trapdoor functions, such as integer factorization and elliptic curve discrete logarithms."),
            ("Markov Decision Processes (MDP)", "An MDP is formally defined by the 5-tuple (S, A, P, R, gamma), providing the mathematical framework for modeling reinforcement learning decision-making under uncertainty."),
            ("Plate Tectonics & Lithospheric Dynamics", "Earth's brittle lithosphere is split into mobile tectonic plates gliding over the ductile asthenosphere, causing subduction zones, mid-ocean ridges, and transform faulting."),
            ("Cosmological Nucleosynthesis", "During the first three minutes following the Big Bang, rapid nuclear fusion forged primary cosmological abundances of light isotopes: 75% hydrogen, 25% helium-4, and trace deuterium and lithium-7."),
        ]

        self.dialogues = [
            ("Hello! Can you help me today?", "Hello! I am Pseudo-Brain, ready to assist you. What problem, algorithm, or concept would you like to work on?"),
            ("How do you remember information across turns?", "I maintain persistent working memory slots (K=16, W=64, strictly 4.0 KB) along with an episodic consolidation bank, updating internal thoughts at 60 Hz without transformer KV-caches."),
            ("What is the primary difference between you and standard transformers?", "Transformers store every historical token in expanding multi-gigabyte KV-caches. In contrast, I use a constant-memory recurrent state update with deep highway projection layers that operate in O(1) time per token."),
            ("Can you help me optimize my Python code?", "Yes! I can help refactor algorithms, eliminate quadratic time bottlenecks, implement caching, vectorize numerical operations, or write unit tests."),
            ("What are your core cognitive capabilities?", "I am trained to execute natural conversation, algorithmic code generation, symbolic math reasoning, multi-step tool execution, and 60 Hz closed-loop POMDP game control."),
            ("How does cognitive preemption work in multi-threaded agents?", "When high-priority sensory interrupts arrive, the router suspends the current active thread slot, addresses the critical event on another thread, and later resumes the suspended context without memory corruption."),
            ("Why is gradient clipping important in deep recurrent networks?", "Deep recurrence across long sequences can compound matrix product singular values, causing exploding gradients. Clipping grad norms to a maximum threshold stabilizes backpropagation through time."),
        ]

        self.query_stems = [
            "Explain the physical principles of",
            "Describe the mathematical foundation of",
            "What are the operational mechanisms behind",
            "Summarize the core concepts of",
            "Provide a comprehensive technical overview of",
            "How does",
        ]

        self.corpus_lines: List[str] = []
        for candidate_path in [
            "/content/data/transformed_hf_conversational_corpus.jsonl",
            "data/transformed_hf_conversational_corpus.jsonl",
            "../data/transformed_hf_conversational_corpus.jsonl",
        ]:
            try:
                from pathlib import Path
                p = Path(candidate_path)
                if p.exists():
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            txt = data.get("text", "")
                            if txt:
                                self.corpus_lines.append(txt)
                    if self.corpus_lines:
                        break
            except Exception:
                pass

    def sample(self) -> RawDocument:
        thread_id = self.rng.randint(0, self.max_threads - 1)
        if self.corpus_lines and self.rng.random() < 0.30:
            raw_text = self.rng.choice(self.corpus_lines)
            if not raw_text.startswith("[THREAD:"):
                raw_text = f"[THREAD:{thread_id}]{raw_text}"
            return RawDocument(text=raw_text, task_type=TaskType.LANGUAGE, thread_id=thread_id, is_dialogue=True)

        if self.rng.random() < 0.75:
            try:
                from .synthetic_distill import SyntheticDistillationEngine
                if not hasattr(self, "_distill"):
                    self._distill = SyntheticDistillationEngine(seed=self.rng.randint(0, 100000), max_threads=self.max_threads)
                if self.rng.random() < 0.5:
                    return RawDocument(text=self._distill.sample_geography(), task_type=TaskType.LANGUAGE, thread_id=thread_id, is_dialogue=True)
                else:
                    return RawDocument(text=self._distill.sample_science(), task_type=TaskType.LANGUAGE, thread_id=thread_id, is_dialogue=True)
            except Exception:
                pass

        mode = self.rng.choice(["knowledge", "dialogue", "query"])
        if mode == "knowledge":
            topic, desc = self.rng.choice(self.topics)
            variant = self.rng.randint(100, 99999)
            text = (
                f"[THREAD:{thread_id}]Topic Query: Describe {topic} (Ref #{variant}). "
                f"[RESP]Description: {desc} "
                f"Further inquiry reveals fundamental mathematical symmetries governing its invariant dynamics.[EOS]"
            )
            return RawDocument(text=text, task_type=TaskType.LANGUAGE, thread_id=thread_id, is_dialogue=False)
        elif mode == "query":
            topic, desc = self.rng.choice(self.topics)
            stem = self.rng.choice(self.query_stems)
            text = (
                f"[THREAD:{thread_id}]User: {stem} {topic}? "
                f"[RESP]{desc} This principle demonstrates invariant physical and mathematical structure.[EOS]"
            )
            return RawDocument(text=text, task_type=TaskType.LANGUAGE, thread_id=thread_id, is_dialogue=True)
        else:
            user_msg, bot_msg = self.rng.choice(self.dialogues)
            variant = self.rng.randint(10, 9999)
            text = (
                f"[THREAD:{thread_id}]User: {user_msg} (ID:{variant}) "
                f"[RESP]{bot_msg}[EOS]"
            )
            return RawDocument(text=text, task_type=TaskType.LANGUAGE, thread_id=thread_id, is_dialogue=True)


class SyntheticCodeStream:
    """Infinite generator of algorithmic Python programming snippets, modules, and tests."""

    def __init__(self, seed: int = 43, max_threads: int = 64):
        self.rng = random.Random(seed)
        self.max_threads = max_threads

        self.code_templates = [
            # Quicksort
            """def quicksort(arr: List[int]) -> List[int]:
    \"\"\"Sort list using in-place partitioning with pivot median.\"\"\"
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)""",

            # Mergesort
            """def mergesort(arr: List[int]) -> List[int]:
    \"\"\"Divide and conquer sorting algorithm with O(N log N) worst-case time.\"\"\"
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = mergesort(arr[:mid])
    right = mergesort(arr[mid:])
    merged: List[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged""",

            # Binary Search
            """def binary_search(arr: Sequence[int], target: int) -> int:
    \"\"\"Logarithmic search over pre-sorted array.\"\"\"
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1""",

            # Fibonacci Memoization
            """def fibonacci(n: int, memo: Optional[Dict[int, int]] = None) -> int:
    \"\"\"Compute n-th Fibonacci number using top-down memoization.\"\"\"
    if memo is None:
        memo = {}
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n in memo:
        return memo[n]
    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]""",

            # Palindrome Checker
            """def is_palindrome(s: str) -> bool:
    \"\"\"Check if string is a palindrome ignoring punctuation and case.\"\"\"
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]""",

            # LRU Cache
            """class LRUCache:
    \"\"\"Least Recently Used Cache using Doubly Linked List and Hash Map.\"\"\"
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache: Dict[int, int] = {}
        self.order: List[int] = []

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.cap:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)""",

            # Dijkstra Shortest Paths
            """def dijkstra(graph: Dict[int, List[Tuple[int, float]]], source: int) -> Dict[int, float]:
    \"\"\"Compute shortest path distances using min-heap priority queue.\"\"\"
    import heapq
    distances = {source: 0.0}
    pq = [(0.0, source)]
    while pq:
        dist, u = heapq.heappop(pq)
        if dist > distances.get(u, float('inf')):
            continue
        for v, weight in graph.get(u, []):
            new_dist = dist + weight
            if new_dist < distances.get(v, float('inf')):
                distances[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    return distances""",

            # Breadth-First Search
            """def bfs_traversal(graph: Dict[int, List[int]], start: int) -> List[int]:
    \"\"\"Traverse graph level-by-level using queue.\"\"\"
    from collections import deque
    visited = {start}
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order""",

            # Prime Sieve
            """def sieve_of_eratosthenes(limit: int) -> List[int]:
    \"\"\"Find all prime numbers up to limit in O(N log log N) time.\"\"\"
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            for multiple in range(p * p, limit + 1, p):
                is_prime[multiple] = False
    return [p for p, prime in enumerate(is_prime) if prime]""",

            # Matrix Multiplication
            """def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    \"\"\"Compute standard matrix product C = A @ B.\"\"\"
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, 'Inner dimensions must match'
    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for k in range(cols_A):
            for j in range(cols_B):
                C[i][j] += A[i][k] * B[k][j]
    return C""",

            # RMSNorm PyTorch Layer
            """class RMSNorm(torch.nn.Module):
    \"\"\"Root Mean Square Layer Normalization for recurrent neural state.\"\"\"
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight""",

            # Stack Data Structure
            """class Stack:
    \"\"\"Last-In First-Out (LIFO) stack data structure.\"\"\"
    def __init__(self):
        self._items: List[Any] = []

    def push(self, item: Any) -> None:
        self._items.append(item)

    def pop(self) -> Any:
        if not self._items:
            raise IndexError('pop from empty stack')
        return self._items.pop()

    def peek(self) -> Any:
        if not self._items:
            raise IndexError('peek from empty stack')
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0""",

            # Two Sum Problem
            """def two_sum(nums: List[int], target: int) -> Optional[Tuple[int, int]]:
    \"\"\"Find indices of two numbers that add up to target in O(N) time.\"\"\"
    seen: Dict[int, int] = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return seen[complement], i
        seen[num] = i
    return None""",

            # Longest Common Subsequence
            """def lcs(s1: str, s2: str) -> int:
    \"\"\"Compute length of longest common subsequence via dynamic programming.\"\"\"
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]""",
        ]

    def sample(self) -> RawDocument:
        thread_id = self.rng.randint(0, self.max_threads - 1)
        if self.rng.random() < 0.80:
            try:
                from .synthetic_distill import SyntheticDistillationEngine
                if not hasattr(self, "_distill"):
                    self._distill = SyntheticDistillationEngine(seed=self.rng.randint(0, 100000), max_threads=self.max_threads)
                return RawDocument(text=self._distill.sample_code(), task_type=TaskType.CODE, thread_id=thread_id, is_dialogue=False)
            except Exception:
                pass

        snippet = self.rng.choice(self.code_templates)
        variant = self.rng.randint(10, 99999)

        code_text = (
            f"[THREAD:{thread_id}]# Specification: Implement algorithmic module pseudo_brain.algo_{variant}\\n"
            f"[RESP]from typing import List, Dict, Optional, Tuple, Sequence, Any\\n"
            f"import torch\\n\\n"
            f"{snippet}\\n[EOS]"
        )
        return RawDocument(text=code_text, task_type=TaskType.CODE, thread_id=thread_id, is_dialogue=False)


class SyntheticReasoningStream:
    """Infinite generator of structured reasoning, symbolic math derivations, tool-calling, and DAGs."""

    def __init__(self, seed: int = 44, max_threads: int = 64):
        self.rng = random.Random(seed)
        self.max_threads = max_threads

        self.tools = [
            ("query_vector_store", {"query": "cognitive latency", "top_k": 3}),
            ("run_memory_audit", {"thread_id": 2, "threshold": 0.05}),
            ("fetch_telemetry", {"sensor_channel": "lidar_radial", "resolution": 64}),
            ("optimize_dag", {"nodes": ["read", "filter", "aggregate"]}),
            ("compute_kalman_gain", {"variance_q": 0.01, "variance_r": 0.1}),
            ("inspect_code_ast", {"filepath": "models/recurrent_core.py"}),
            ("execute_test_suite", {"target": "tests/test_cognitive_stability.py"}),
            ("calculate_eigenvalues", {"matrix_dim": 64, "hermitian": True}),
        ]

    def sample(self) -> RawDocument:
        thread_id = self.rng.randint(0, self.max_threads - 1)
        if self.rng.random() < 0.70:
            try:
                from .synthetic_distill import SyntheticDistillationEngine
                if not hasattr(self, "_distill"):
                    self._distill = SyntheticDistillationEngine(seed=self.rng.randint(0, 100000), max_threads=self.max_threads)
                return RawDocument(text=self._distill.sample_math(), task_type=TaskType.REASONING, thread_id=thread_id, is_dialogue=True)
            except Exception:
                pass

        mode = self.rng.choice(["linear_math", "quadratic_math", "arithmetic_word", "tool_call", "dag", "software_pomdp"])

        if mode == "software_pomdp":
            from irene_brain.agent.pomdp_trajectory_generator import POMDPTrajectoryGenerator
            if not hasattr(self, "_pomdp_gen"):
                self._pomdp_gen = POMDPTrajectoryGenerator(seed=self.rng.randint(0, 100000))
            traj = self._pomdp_gen.sample_trajectory()
            text = traj.to_training_text(thread_id=thread_id)
            return RawDocument(text=text, task_type=TaskType.REASONING, thread_id=thread_id, is_dialogue=True)

        elif mode == "linear_math":
            a = self.rng.randint(2, 15)
            b = self.rng.randint(3, 30)
            c = a * self.rng.randint(1, 12) + b
            x_val = (c - b) // a
            text = (
                f"[THREAD:{thread_id}]Solve for x: {a}*x + {b} = {c} "
                f"[RESP]<thought> Step 1: Subtract {b} from both sides: {a}*x = {c} - {b} = {c - b}. "
                f"Step 2: Divide both sides by {a}: x = {c - b} / {a} = {x_val}. </thought> "
                f"<solution>x = {x_val}</solution>[EOS]"
            )
        elif mode == "quadratic_math":
            r1 = self.rng.randint(1, 7)
            r2 = self.rng.randint(1, 7)
            # (x - r1)(x - r2) = x^2 - (r1+r2)x + r1*r2
            b_coef = -(r1 + r2)
            c_coef = r1 * r2
            sign_b = f"- {abs(b_coef)}" if b_coef < 0 else f"+ {b_coef}"
            text = (
                f"[THREAD:{thread_id}]Solve for x: x^2 {sign_b}*x + {c_coef} = 0 "
                f"[RESP]<thought> Factoring the quadratic: (x - {r1}) * (x - {r2}) = 0. "
                f"Setting each factor to zero gives roots x = {r1} and x = {r2}. </thought> "
                f"<solution>roots = [{min(r1, r2)}, {max(r1, r2)}]</solution>[EOS]"
            )
        elif mode == "arithmetic_word":
            speed = self.rng.randint(30, 80)
            hours = self.rng.randint(2, 8)
            dist = speed * hours
            text = (
                f"[THREAD:{thread_id}]If a vehicle travels at a constant velocity of {speed} km/h for {hours} hours, how far does it go? "
                f"[RESP]<thought> Distance = velocity * time. distance = {speed} km/h * {hours} h = {dist} km. </thought> "
                f"<solution>{dist} kilometers</solution>[EOS]"
            )
        elif mode == "tool_call":
            tool_name, tool_args = self.rng.choice(self.tools)
            call_json = json.dumps({"name": tool_name, "arguments": tool_args})
            resp_json = json.dumps({"status": "success", "result": "computed_payload", "latency_ms": 0.85})
            text = (
                f"[THREAD:{thread_id}]Instruction: Perform automated diagnostic action. "
                f"[RESP]<tool_call>{call_json}</tool_call>[EOS] "
                f"<tool_response>{resp_json}</tool_response> "
                f"[RESP]Diagnostic action concluded successfully without exception.[EOS]"
            )
        else:  # DAG execution plan
            steps = ["ingest_frame", "fused_triton_scan", "latent_lookahead", "dispatch_motor_command"]
            self.rng.shuffle(steps)
            text = (
                f"[THREAD:{thread_id}]Execute Cognitive DAG: "
                f"Node 0: {steps[0]}() [DEP] "
                f"Node 1: {steps[1]}(dep=Node 0) [DEP] "
                f"Node 2: {steps[2]}(dep=Node 1) "
                f"[RESP]Execution schedule topologically ordered and validated.[EOS]"
            )

        return RawDocument(text=text, task_type=TaskType.REASONING, thread_id=thread_id, is_dialogue=True)


class SyntheticPOMDPStream:
    """Infinite generator of sensory arcade and POMDP control sequences."""

    def __init__(self, seed: int = 45, max_threads: int = 64):
        self.rng = random.Random(seed)
        self.max_threads = max_threads
        self.actions = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT", "COLLECT_KEY", "UNLOCK_DOOR", "WAIT"]

    def sample(self) -> RawDocument:
        thread_id = self.rng.randint(0, self.max_threads - 1)
        ep_id = self.rng.randint(100, 99999)
        num_steps = self.rng.randint(2, 5)

        parts = [f"[THREAD:{thread_id}][POMDP]Episode:{ep_id}"]
        for step in range(num_steps):
            x = self.rng.randint(0, 15)
            y = self.rng.randint(0, 15)
            act = self.rng.choice(self.actions)
            reward = 1.0 if act in ("COLLECT_KEY", "UNLOCK_DOOR") else 0.0
            done = (step == num_steps - 1)
            parts.append(
                f" step:{step} obs:(x={x},y={y},chaser_dist={self.rng.randint(1, 10)}) "
                f"[RESP]action:{act} reward:{reward:+.1f} done:{done}[EOS]"
            )

        return RawDocument(text="".join(parts), task_type=TaskType.GAME_POMDP, thread_id=thread_id, is_dialogue=True)


# ==============================================================================
# Streaming Hugging Face / HTTP Ingestion Engine
# ==============================================================================

class HFStreamIterator:
    """Streams rows from Hugging Face datasets with automatic infinite restart & fallback."""

    def __init__(
        self,
        task_type: TaskType,
        dataset_name: str,
        split: str = "train",
        fallback_stream: Optional[Any] = None,
        offline_mode: bool = False,
        seed: int = 42,
    ):
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.split = split
        self.fallback_stream = fallback_stream
        self.offline_mode = offline_mode
        self.seed = seed
        self._iterator: Optional[Iterator[Dict[str, Any]]] = None
        self._use_fallback = offline_mode or not _HF_DATASETS_AVAILABLE

    def _init_hf_stream(self) -> bool:
        if self._use_fallback:
            return False
        try:
            ds = datasets.load_dataset(self.dataset_name, split=self.split, streaming=True)
            self._iterator = iter(ds)
            return True
        except Exception as e:
            logger.warning(
                f"Unable to load HF streaming dataset {self.dataset_name} ({e}). "
                f"Switching {self.task_type.value} stream to high-density synthetic fallback."
            )
            self._use_fallback = True
            return False

    def next_document(self) -> RawDocument:
        """Fetch next document from HF stream, or fallback if unavailable/exhausted."""
        if not self._use_fallback and self._iterator is None:
            if not self._init_hf_stream():
                self._use_fallback = True

        if not self._use_fallback and self._iterator is not None:
            try:
                row = next(self._iterator)
                text = self._extract_text(row)
                if text and len(text.strip()) > 0:
                    return RawDocument(text=text.strip(), task_type=self.task_type)
            except StopIteration:
                # Infinite streaming: restart iterator
                self._init_hf_stream()
            except Exception as e:
                logger.warning(f"Error reading from {self.dataset_name}: {e}. Falling back.")
                self._use_fallback = True

        # Fallback to high-density generator
        if self.fallback_stream is not None:
            return self.fallback_stream.sample()

        raise RuntimeError(f"No stream or fallback available for {self.task_type}")

    def _extract_text(self, row: Dict[str, Any]) -> str:
        """Extract clean text or dialogue from various standard dataset schemas."""
        # 1. Direct text fields
        for field in ("text", "content", "code", "output"):
            if field in row and isinstance(row[field], str) and row[field]:
                return row[field]

        # 2. Multi-turn dialogue messages
        if "messages" in row and isinstance(row["messages"], list):
            parts = []
            for msg in row["messages"]:
                role = msg.get("role", "user")
                c = msg.get("content", "")
                if role == "assistant":
                    parts.append(f"[RESP]{c}[EOS]")
                else:
                    parts.append(f"User: {c}")
            return " ".join(parts)

        # 3. Instruction + Input + Output (Alpaca style)
        if "instruction" in row and "output" in row:
            prompt = row.get("instruction", "")
            inp = row.get("input", "")
            out = row.get("output", "")
            full_prompt = f"{prompt} {inp}".strip() if inp else prompt
            return f"User: {full_prompt} [RESP]{out}[EOS]"

        # 4. Math / Problem + Solution
        if "problem" in row and "solution" in row:
            return f"Problem: {row['problem']} [RESP]Solution: {row['solution']}[EOS]"

        # Fallback string representation of row
        return json.dumps(row)


# ==============================================================================
# Sequence Packer: Uniform Sequence Packing with Thread IDs & Loss Masking
# ==============================================================================

class SequencePacker:
    """Packs variable-length token streams into uniform blocks of length T.

    Ensures:
    1. Uniform block length T (no padding overhead).
    2. Document boundary isolation: cross-document transitions are masked in labels (-100).
    3. Thread ID persistence: tracks [THREAD:0..K-1] cognitive slot transitions.
    4. Optional assistant-only response loss masking ([RESP]..[EOS]).
    """

    def __init__(
        self,
        tokenizer: Any,
        seq_len: int = 512,
        max_threads: int = 64,
        supervised_only_loss: bool = False,
        mask_cross_document_boundary: bool = True,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_threads = max_threads
        self.supervised_only_loss = supervised_only_loss
        self.mask_cross_document_boundary = mask_cross_document_boundary

        self.eos_id = getattr(tokenizer, "eos_id", 2)
        self.resp_id = getattr(tokenizer, "resp_id", 5)
        self.bos_id = getattr(tokenizer, "bos_id", 1)

        # Internal packing accumulators
        self._token_buffer: List[int] = []
        self._target_buffer: List[int] = []
        self._thread_buffer: List[int] = []
        self._task_buffer: List[str] = []
        self._boundary_buffer: List[float] = []

        self._cur_thread_id: int = 0
        self._in_resp: bool = False

    def reset(self) -> None:
        """Clear packing buffers."""
        self._token_buffer.clear()
        self._target_buffer.clear()
        self._thread_buffer.clear()
        self._task_buffer.clear()
        self._boundary_buffer.clear()
        self._cur_thread_id = 0
        self._in_resp = False

    def append_document(self, doc: RawDocument) -> None:
        """Tokenize document and append to packing buffer with proper loss masks."""
        # Ensure thread prefix if not present
        text = doc.text
        if not text.startswith("[THREAD:"):
            text = f"[THREAD:{doc.thread_id % self.max_threads}]{text}"

        # Ensure EOS ending
        if not text.endswith("[EOS]") and not text.endswith("[EOS] "):
            text = f"{text}[EOS]"

        tokens = self.tokenizer.encode(text)
        if not tokens:
            return

        n = len(tokens)
        targets: List[int] = [-100] * n
        threads: List[int] = [0] * n
        boundaries: List[float] = [0.0] * n
        if n > 0:
            boundaries[0] = 1.0  # Document start boundary

        in_resp = False
        cur_thread = doc.thread_id % self.max_threads

        has_resp = (self.resp_id in tokens)
        for i in range(n):
            tok = tokens[i]
            # Check thread marker
            parsed_thread = None
            if hasattr(self.tokenizer, "token_id_to_thread_id"):
                parsed_thread = self.tokenizer.token_id_to_thread_id(tok)
            if parsed_thread is not None:
                cur_thread = parsed_thread % self.max_threads

            threads[i] = cur_thread

            # Compute next-token targets
            if i < n - 1:
                next_tok = tokens[i + 1]
                if tok == self.resp_id:
                    in_resp = True

                if self.supervised_only_loss:
                    if in_resp or not has_resp:
                        targets[i] = next_tok
                else:
                    targets[i] = next_tok

                if next_tok == self.eos_id:
                    in_resp = False

        # If mask_cross_document_boundary is True, mask target at EOS
        if self.mask_cross_document_boundary and n > 0:
            targets[-1] = -100

        self._token_buffer.extend(tokens)
        self._target_buffer.extend(targets)
        self._thread_buffer.extend(threads)
        self._task_buffer.extend([doc.task_type.value] * n)
        self._boundary_buffer.extend(boundaries)

    def can_emit_block(self) -> bool:
        """Check if buffer has at least seq_len tokens."""
        return len(self._token_buffer) >= self.seq_len

    def emit_block(self) -> PackedSequenceBlock:
        """Extract a packed block of exact length seq_len."""
        T = self.seq_len
        input_ids = torch.tensor(self._token_buffer[:T], dtype=torch.long)
        labels = torch.tensor(self._target_buffer[:T], dtype=torch.long)
        thread_ids = torch.tensor(self._thread_buffer[:T], dtype=torch.long)
        loss_mask = (labels != -100).to(torch.float32)
        reset_mask = torch.tensor(self._boundary_buffer[:T], dtype=torch.float32)

        # Unique contributing task types in this block
        tasks_in_block = list(dict.fromkeys(self._task_buffer[:T]))

        # Slice remaining tokens
        self._token_buffer = self._token_buffer[T:]
        self._target_buffer = self._target_buffer[T:]
        self._thread_buffer = self._thread_buffer[T:]
        self._task_buffer = self._task_buffer[T:]
        self._boundary_buffer = self._boundary_buffer[T:]

        return PackedSequenceBlock(
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            thread_ids=thread_ids,
            task_types=tasks_in_block,
            reset_mask=reset_mask,
        )


# ==============================================================================
# Multi-Task Mixture Stream
# ==============================================================================

class MultiTaskMixtureStream:
    """Interleaves diverse task streams according to configurable mixture weights."""

    def __init__(self, config: StreamingConfig, tokenizer: Any, worker_id: int = 0):
        self.config = config
        self.tokenizer = tokenizer
        self.worker_id = worker_id
        seed = config.seed + worker_id * 10007
        self.rng = random.Random(seed)

        # 1. Instantiate synthetic fallbacks
        self.synthetic_streams = {
            TaskType.LANGUAGE: SyntheticLanguageStream(seed=seed + 1, max_threads=config.max_threads),
            TaskType.CODE: SyntheticCodeStream(seed=seed + 2, max_threads=config.max_threads),
            TaskType.REASONING: SyntheticReasoningStream(seed=seed + 3, max_threads=config.max_threads),
            TaskType.GAME_POMDP: SyntheticPOMDPStream(seed=seed + 4, max_threads=config.max_threads),
        }

        # 2. Instantiate stream readers
        self.stream_readers: Dict[TaskType, HFStreamIterator] = {}
        for task_type in TaskType:
            ds_name = config.hf_dataset_names.get(task_type, "")
            ds_split = config.hf_dataset_splits.get(task_type, "train")
            fallback = self.synthetic_streams[task_type]

            # In offline_mode or if dataset name is 'synthetic', force offline
            offline = config.offline_mode or ds_name.startswith("synthetic")

            self.stream_readers[task_type] = HFStreamIterator(
                task_type=task_type,
                dataset_name=ds_name,
                split=ds_split,
                fallback_stream=fallback,
                offline_mode=offline,
                seed=seed,
            )

        # 3. Normalized weights and cumulative distribution for fast multinomial sampling
        self.weights_dict = config.weights.to_dict()
        self.tasks = list(self.weights_dict.keys())
        self.probs = [self.weights_dict[t] for t in self.tasks]

        # 4. Sequence Packer
        self.packer = SequencePacker(
            tokenizer=tokenizer,
            seq_len=config.seq_len,
            max_threads=config.max_threads,
            supervised_only_loss=config.supervised_only_loss,
            mask_cross_document_boundary=config.mask_cross_document_boundary,
        )

    def next_packed_block(self) -> PackedSequenceBlock:
        """Sample documents according to weights until a full block is packed."""
        while not self.packer.can_emit_block():
            task = self.rng.choices(self.tasks, weights=self.probs, k=1)[0]
            reader = self.stream_readers[task]
            doc = reader.next_document()
            self.packer.append_document(doc)

        return self.packer.emit_block()


# ==============================================================================
# PyTorch IterableDataset & Collate Engine
# ==============================================================================

class StreamingMultiTaskDataset(IterableDataset):
    """Infinite PyTorch IterableDataset yielding uniform packed pretraining blocks."""

    def __init__(self, config: StreamingConfig, tokenizer: Any):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

    def __iter__(self) -> Generator[PackedSequenceBlock, None, None]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        mixture = MultiTaskMixtureStream(
            config=self.config,
            tokenizer=self.tokenizer,
            worker_id=worker_id,
        )

        while True:
            yield mixture.next_packed_block()


def collate_streaming_batch(blocks: List[PackedSequenceBlock]) -> StreamingBatch:
    """Collate packed sequence blocks into a high-performance batch [B, T]."""
    input_ids = torch.stack([b.input_ids for b in blocks], dim=0)
    labels = torch.stack([b.labels for b in blocks], dim=0)
    loss_mask = torch.stack([b.loss_mask for b in blocks], dim=0)
    thread_ids = torch.stack([b.thread_ids for b in blocks], dim=0)
    task_types = [b.task_types for b in blocks]
    reset_mask = (
        torch.stack([b.reset_mask for b in blocks], dim=0)
        if (blocks and blocks[0].reset_mask is not None)
        else None
    )

    return StreamingBatch(
        input_ids=input_ids,
        labels=labels,
        loss_mask=loss_mask,
        thread_ids=thread_ids,
        task_types=task_types,
        reset_mask=reset_mask,
    )


def create_streaming_dataloader(
    config: StreamingConfig,
    tokenizer: Any,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Factory creating high-throughput PyTorch DataLoader ready for A100 training."""
    dataset = StreamingMultiTaskDataset(config=config, tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_streaming_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
