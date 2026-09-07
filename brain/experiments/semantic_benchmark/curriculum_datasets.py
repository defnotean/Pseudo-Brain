"""Controlled Language Curriculum Datasets (Stages A, B, and C).

Constructs controlled, reproducible language tasks designed to measure whether
recurrent cognitive state, thread isolation, and fast synaptic latching support
semantic memory, preemption, interleaving, and cross-thread dependencies.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from irene_brain.semantic.tokenizer import SemanticTokenizer


@dataclass
class SemanticEpisode:
    """Represents a single conversational/semantic training or test sequence."""
    text_sequence: str
    tokens: List[int]
    targets: List[int]  # -100 on prompt/unsupervised tokens, target token ID on predicted tokens
    threads: List[int]  # Active cognitive thread index per token
    task_name: str
    metadata: Dict[str, Any]


class LanguageCurriculumGenerator:
    """Generates synthetic language and symbolic tasks across Stages A, B, and C."""

    def __init__(self, tokenizer: Optional[SemanticTokenizer] = None, max_threads: int = 16):
        self.tokenizer = tokenizer or SemanticTokenizer(max_threads=max_threads)
        self.max_threads = max_threads

        # Vocabulary pools for Stage B
        self.names = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry"]
        self.items = ["red car", "blue bike", "green book", "gold watch", "silver key", "black pen", "white cup", "yellow hat"]
        self.cities = ["Toronto", "Tokyo", "Paris", "London", "Berlin", "Sydney", "Rome", "Seoul"]
        self.hobbies = ["chess", "soccer", "guitar", "painting", "coding", "swimming", "reading", "cooking"]
        self.preferences = ["coffee", "tea", "water", "juice", "milk", "soda", "cider", "cocoa"]

    # ==========================================================================
    # Stage A: Symbolic Sequences
    # ==========================================================================

    def generate_stage_a_copy(self, rng: np.random.RandomState, length: int = 4, delay: int = 8) -> SemanticEpisode:
        """Task A1: Sequence Copying after delay corridor."""
        symbols = "ABCDEFGH"
        seq = "".join(rng.choice(list(symbols), size=length))
        prompt = f"[THREAD:0]Store:{seq}." + "." * delay + "Recall:"
        target_str = seq + "[EOS]"

        full_text = prompt + target_str
        tokens = self.tokenizer.encode(full_text)
        prompt_len = len(self.tokenizer.encode(prompt))

        targets = [-100] * len(tokens)
        for i in range(prompt_len, len(tokens)):
            targets[i] = tokens[i]

        threads = [0] * len(tokens)
        return SemanticEpisode(
            text_sequence=full_text,
            tokens=tokens,
            targets=targets,
            threads=threads,
            task_name="stage_a_copy",
            metadata={"seq": seq, "delay": delay},
        )

    def generate_stage_a_associative_binding(
        self,
        rng: np.random.RandomState,
        num_pairs: int = 3,
        delay: int = 8,
    ) -> SemanticEpisode:
        """Task A2: Key-Value Binding (e.g. set X = 4, set Y = 7, query X -> 4)."""
        keys = list("XYZWUV")[:num_pairs]
        vals = [str(v) for v in rng.choice(list(range(10)), size=num_pairs, replace=False)]
        pairs = list(zip(keys, vals))
        rng.shuffle(pairs)

        prompt_parts = ["[THREAD:0]"]
        for k, v in pairs:
            prompt_parts.append(f"set {k}={v};")
        prompt_parts.append("." * delay)

        # Pick one key to query
        q_idx = int(rng.randint(0, num_pairs))
        q_key, q_val = pairs[q_idx]
        prompt_parts.append(f"query {q_key}=")
        prompt = "".join(prompt_parts)
        target_str = q_val + "[EOS]"

        full_text = prompt + target_str
        tokens = self.tokenizer.encode(full_text)
        prompt_len = len(self.tokenizer.encode(prompt))

        targets = [-100] * len(tokens)
        for i in range(prompt_len, len(tokens)):
            targets[i] = tokens[i]

        threads = [0] * len(tokens)
        return SemanticEpisode(
            text_sequence=full_text,
            tokens=tokens,
            targets=targets,
            threads=threads,
            task_name="stage_a_associative_binding",
            metadata={"query_key": q_key, "target_val": q_val, "pairs": pairs},
        )

    # ==========================================================================
    # Stage B: Synthetic Multi-Threaded Language
    # ==========================================================================

    def generate_stage_b_delayed_recall(
        self,
        rng: np.random.RandomState,
        num_facts: int = 2,
        delay: int = 12,
    ) -> SemanticEpisode:
        """Task B1: Multi-entity factual enrollment and delayed recall."""
        selected_indices = rng.choice(len(self.names), size=num_facts, replace=False)
        facts = []
        for idx in selected_indices:
            facts.append((self.names[idx], self.items[idx]))

        prompt_parts = []
        for i, (name, item) in enumerate(facts):
            tid = i % self.max_threads
            prompt_parts.append(f"[THREAD:{tid}]{name} owns a {item}. ")

        # Intervening noise delay corridor
        prompt_parts.append("[THREAD:0]" + " " * delay)

        # Query one fact
        target_fact = facts[int(rng.randint(0, num_facts))]
        q_name, q_item = target_fact
        q_tid = facts.index(target_fact) % self.max_threads
        prompt_parts.append(f"[THREAD:{q_tid}]What does {q_name} own? [RESP]")
        prompt = "".join(prompt_parts)
        target_str = q_item + "[EOS]"

        full_text = prompt + target_str
        tokens = self.tokenizer.encode(full_text)
        prompt_len = len(self.tokenizer.encode(prompt))

        targets = [-100] * len(tokens)
        for i in range(prompt_len, len(tokens)):
            targets[i] = tokens[i]

        # Build thread sequence
        threads = self._extract_thread_ids_from_tokens(tokens)
        return SemanticEpisode(
            text_sequence=full_text,
            tokens=tokens,
            targets=targets,
            threads=threads,
            task_name="stage_b_delayed_recall",
            metadata={"target_name": q_name, "target_item": q_item},
        )

    def generate_stage_b_interleaved_conversations(
        self,
        rng: np.random.RandomState,
        num_conversations: int = 3,
    ) -> SemanticEpisode:
        """Task B2: Interleaved independent conversations across dedicated threads.

        Thread 0: Alice lives in Toronto.
        Thread 1: Bob likes chess.
        Thread 2: Charlie owns a gold watch.
        Query Thread 1: What does Bob like? -> chess
        Query Thread 0: Where does Alice live? -> Toronto
        """
        num_conversations = max(1, min(num_conversations, self.max_threads))
        conv_data = []
        for i in range(num_conversations):
            name = self.names[i]
            city = self.cities[i]
            conv_data.append({
                "thread_id": i,
                "name": name,
                "fact": f"[THREAD:{i}]{name} lives in {city}. ",
                "query": f"[THREAD:{i}]Where does {name} live? [RESP]",
                "answer": city + "[EOS]",
            })

        # Interleave statements
        rng.shuffle(conv_data)
        prompt_parts = [cd["fact"] for cd in conv_data]

        # Interleave queries
        query_order = list(range(num_conversations))
        rng.shuffle(query_order)

        # Pick one query to supervise
        target_conv = conv_data[query_order[0]]
        prompt_parts.append(target_conv["query"])
        prompt = "".join(prompt_parts)
        target_str = target_conv["answer"]

        full_text = prompt + target_str
        tokens = self.tokenizer.encode(full_text)
        prompt_len = len(self.tokenizer.encode(prompt))

        targets = [-100] * len(tokens)
        for i in range(prompt_len, len(tokens)):
            targets[i] = tokens[i]

        threads = self._extract_thread_ids_from_tokens(tokens)
        return SemanticEpisode(
            text_sequence=full_text,
            tokens=tokens,
            targets=targets,
            threads=threads,
            task_name="stage_b_interleaved_conversations",
            metadata={"target_thread": target_conv["thread_id"], "target_answer": target_conv["answer"]},
        )

    def generate_stage_b_preemption(
        self,
        rng: np.random.RandomState,
        interruption_length: int = 16,
    ) -> SemanticEpisode:
        """Task B3: Preemption and resumption.

        Thread 0 begins a reasoning/booking sequence:
          [THREAD:0]Booking hotel in Tokyo for Alice.
        [INTERRUPT] Urgent safety alert on Thread 1:
          [THREAD:1]Alert: server down in data center!
          [THREAD:1]Restarting power unit... ok.
        [RESUME] Thread 0 resumes:
          [THREAD:0]Resume hotel booking for: [RESP] -> Alice
        """
        name = str(rng.choice(self.names))
        city = str(rng.choice(self.cities))

        p1 = f"[THREAD:0]Booking hotel in {city} for {name}. "
        p_interrupt = f"[INTERRUPT][THREAD:1]Alert: server overload! " + "x" * interruption_length + " "
        p_resume = f"[RESUME][THREAD:0]Resume hotel booking for: [RESP]"
        prompt = p1 + p_interrupt + p_resume
        target_str = name + "[EOS]"

        full_text = prompt + target_str
        tokens = self.tokenizer.encode(full_text)
        prompt_len = len(self.tokenizer.encode(prompt))

        targets = [-100] * len(tokens)
        for i in range(prompt_len, len(tokens)):
            targets[i] = tokens[i]

        threads = self._extract_thread_ids_from_tokens(tokens)
        return SemanticEpisode(
            text_sequence=full_text,
            tokens=tokens,
            targets=targets,
            threads=threads,
            task_name="stage_b_preemption",
            metadata={"target_name": name, "city": city, "interruption_len": interruption_length},
        )

    def generate_stage_b_cross_thread_dependency(
        self,
        rng: np.random.RandomState,
    ) -> SemanticEpisode:
        """Task B4: Cross-Thread Dependency.

        Thread 0: The secret passkey is 742.
        Thread 1: [DEP]Compute authorized hash using passkey: [RESP] -> 742
        """
        code = str(rng.randint(100, 999))
        p0 = f"[THREAD:0]The secret passkey is {code}. "
        p1 = f"[THREAD:1][DEP]Authorized access using passkey: [RESP]"
        prompt = p0 + p1
        target_str = code + "[EOS]"

        full_text = prompt + target_str
        tokens = self.tokenizer.encode(full_text)
        prompt_len = len(self.tokenizer.encode(prompt))

        targets = [-100] * len(tokens)
        for i in range(prompt_len, len(tokens)):
            targets[i] = tokens[i]

        threads = self._extract_thread_ids_from_tokens(tokens)
        return SemanticEpisode(
            text_sequence=full_text,
            tokens=tokens,
            targets=targets,
            threads=threads,
            task_name="stage_b_cross_thread_dependency",
            metadata={"target_code": code},
        )

    def generate_stage_b_conversational(
        self,
        rng: np.random.RandomState,
    ) -> SemanticEpisode:
        """Task B5: Conversational dialogue with enrollment, query, interruption, and plan resumption."""
        if rng.rand() < 0.5:
            n0, p0 = "Alice", "coffee"
            n1, p1 = "Bob", "tea"
            city = "Tokyo"
        else:
            n0 = str(rng.choice(self.names))
            n1 = str(rng.choice([n for n in self.names if n != n0]))
            p0 = str(rng.choice(self.preferences))
            p1 = str(rng.choice([p for p in self.preferences if p != p0]))
            city = str(rng.choice(self.cities))

        turns = [
            (f"[THREAD:0]Remember {n0} likes {p0}. [RESP]", f"{p0}[EOS]", 0),
            (f"[THREAD:1]Remember {n1} likes {p1}. [RESP]", f"{p1}[EOS]", 1),
            (f"[THREAD:0]What does {n0} like? [RESP]", f"{p0}[EOS]", 0),
            (f"[THREAD:2]Let's plan a trip to {city}. Step 1: book flights. Step 2: hotel. [RESP]", f"explore {city}[EOS]", 2),
            (f"[THREAD:1]What does {n1} like? [RESP]", f"{p1}[EOS]", 1),
            (f"[THREAD:2]Continue the {city} plan. Step 3: [RESP]" if city != "Tokyo" else "[THREAD:2]Continue the Tokyo plan. Step 3: [RESP]", f"explore {city}[EOS]", 2),
        ]

        full_text = ""
        tokens: List[int] = []
        targets: List[int] = []
        threads: List[int] = []

        for prompt, ans, th in turns:
            p_toks = self.tokenizer.encode(prompt)
            a_toks = self.tokenizer.encode(ans)
            seq = p_toks + a_toks

            tgt = [-100] * len(seq)
            prompt_len = len(p_toks)
            for i in range(prompt_len - 1, len(seq) - 1):
                tgt[i] = seq[i + 1]

            full_text += prompt + ans
            tokens.extend(seq)
            targets.extend(tgt)
            threads.extend([th] * len(seq))

        return SemanticEpisode(
            text_sequence=full_text,
            tokens=tokens,
            targets=targets,
            threads=threads,
            task_name="stage_b_conversational",
            metadata={"n0": n0, "p0": p0, "n1": n1, "p1": p1, "city": city},
        )

    def _extract_thread_ids_from_tokens(self, tokens: List[int]) -> List[int]:
        """Parse token stream to assign active thread ID to every token."""
        cur_tid = 0
        threads = []
        for t in tokens:
            parsed_tid = self.tokenizer.token_id_to_thread_id(t)
            if parsed_tid is not None:
                cur_tid = parsed_tid % self.max_threads
            threads.append(cur_tid)
        return threads


def build_curriculum_batch(
    generator: LanguageCurriculumGenerator,
    batch_size: int = 16,
    task_types: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[SemanticEpisode]]:
    """Build padded batch tensors: (tokens [B, T], targets [B, T], threads [B, T], episodes)."""
    if task_types is None:
        task_types = ["stage_a_copy", "stage_a_binding", "stage_b_recall", "stage_b_interleaved", "stage_b_preemption", "stage_b_dependency"]

    rng = np.random.RandomState(seed)
    episodes: List[SemanticEpisode] = []

    for _ in range(batch_size):
        ttype = rng.choice(task_types)
        if ttype == "stage_a_copy":
            ep = generator.generate_stage_a_copy(rng, length=int(rng.randint(3, 6)), delay=int(rng.randint(4, 12)))
        elif ttype == "stage_a_binding":
            ep = generator.generate_stage_a_associative_binding(rng, num_pairs=int(rng.randint(2, 4)), delay=int(rng.randint(4, 10)))
        elif ttype == "stage_b_recall":
            ep = generator.generate_stage_b_delayed_recall(rng, num_facts=int(rng.randint(2, 4)), delay=int(rng.randint(6, 16)))
        elif ttype == "stage_b_interleaved":
            ep = generator.generate_stage_b_interleaved_conversations(rng, num_conversations=int(rng.randint(2, 4)))
        elif ttype == "stage_b_preemption":
            ep = generator.generate_stage_b_preemption(rng, interruption_length=int(rng.randint(8, 20)))
        elif ttype == "stage_b_dependency":
            ep = generator.generate_stage_b_cross_thread_dependency(rng)
        elif ttype == "stage_b_conversational":
            ep = generator.generate_stage_b_conversational(rng)
        else:
            raise ValueError(f"Unknown task type: {ttype}")
        episodes.append(ep)

    max_len = max(len(ep.tokens) for ep in episodes)
    pad_id = generator.tokenizer.pad_id

    tokens_tensor = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    targets_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long)
    threads_tensor = torch.zeros((batch_size, max_len), dtype=torch.long)

    for b, ep in enumerate(episodes):
        L = len(ep.tokens)
        tokens_tensor[b, :L] = torch.tensor(ep.tokens, dtype=torch.long)
        targets_tensor[b, :L] = torch.tensor(ep.targets, dtype=torch.long)
        threads_tensor[b, :L] = torch.tensor(ep.threads, dtype=torch.long)

    return tokens_tensor, targets_tensor, threads_tensor, episodes
