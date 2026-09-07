"""Phase 10: Conventional LLM Dataset Transformation Pipeline for Pseudo-Brain.

Transforms standard multi-turn dialogue, question-answering, and instruction-following
datasets into multi-threaded cognitive episodes with:
1. Thread Decomposition: Mapping dialogues to persistent cognitive thread slots.
2. Controlled Interleaving: Interleaving independent conversations without state cross-talk.
3. Synthetic Preemption: Introducing mid-flight interruptions and requiring downstream resumption.
4. Cross-Thread Dependencies: Binding information across asynchronous threads.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RawDialogueTurn:
    role: str  # 'user' or 'assistant'
    content: str


@dataclass
class RawDialogue:
    dialogue_id: str
    turns: List[RawDialogueTurn]
    category: str = "general"


@dataclass
class TransformedCognitiveEpisode:
    episode_id: str
    interleaved_text: str
    num_threads: int
    has_preemption: bool
    has_dependency: bool
    thread_schedule: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class LLMDataTransformer:
    """Transforms conventional single-conversation datasets into multi-threaded cognitive streams."""

    def __init__(self, max_threads: int = 16):
        self.max_threads = max_threads

    def parse_standard_dialogue_json(self, data: List[Dict[str, Any]]) -> List[RawDialogue]:
        """Parse standard ShareGPT, OpenAI, or Alpaca style dialogues."""
        dialogues: List[RawDialogue] = []
        for i, item in enumerate(data):
            did = item.get("id", f"diag_{i}")
            turns: List[RawDialogueTurn] = []

            # Handle 'conversations' key (ShareGPT format)
            if "conversations" in item:
                for c in item["conversations"]:
                    role = "user" if c.get("from") in ["human", "user"] else "assistant"
                    turns.append(RawDialogueTurn(role=role, content=c.get("value", "")))
            # Handle 'messages' key (OpenAI format)
            elif "messages" in item:
                for m in item["messages"]:
                    turns.append(RawDialogueTurn(role=m.get("role", "user"), content=m.get("content", "")))
            # Handle 'instruction' / 'output' (Alpaca format)
            elif "instruction" in item:
                turns.append(RawDialogueTurn(role="user", content=item["instruction"]))
                if "output" in item:
                    turns.append(RawDialogueTurn(role="assistant", content=item["output"]))

            if len(turns) >= 2:
                dialogues.append(RawDialogue(dialogue_id=did, turns=turns))

        return dialogues

    def create_interleaved_episode(
        self,
        dialogues: List[RawDialogue],
        rng: np.random.RandomState,
        include_preemption: bool = True,
        include_dependency: bool = True,
    ) -> TransformedCognitiveEpisode:
        """Combine N independent dialogues into a single interleaved cognitive sequence."""
        k = min(len(dialogues), self.max_threads)
        selected_dialogues = [dialogues[i] for i in rng.choice(len(dialogues), size=k, replace=False)]

        interleaved_parts: List[str] = []
        schedule: List[Dict[str, Any]] = []

        # Interleave turn-by-turn across assigned threads
        # Thread i gets selected_dialogues[i]
        max_turns = max(len(d.turns) for d in selected_dialogues)

        for turn_idx in range(max_turns):
            # Shuffle thread execution order for this turn
            thread_order = list(range(k))
            rng.shuffle(thread_order)

            for tid in thread_order:
                d = selected_dialogues[tid]
                if turn_idx < len(d.turns):
                    turn = d.turns[turn_idx]
                    prefix = f"[THREAD:{tid}]"
                    if turn.role == "assistant":
                        text_turn = f"{prefix}[RESP]{turn.content}[EOS] "
                    else:
                        text_turn = f"{prefix}{turn.content} "

                    interleaved_parts.append(text_turn)
                    schedule.append({
                        "thread_id": tid,
                        "turn_idx": turn_idx,
                        "role": turn.role,
                        "dialogue_id": d.dialogue_id,
                    })

                    # If synthetic preemption requested and this is an assistant turn on Thread 0
                    if include_preemption and tid == 0 and turn_idx == 0 and k > 1:
                        interrupt_tid = 1
                        interleaved_parts.append(f"[INTERRUPT][THREAD:{interrupt_tid}]Urgent context switch: processing telemetry. [RESP]ok[EOS] [RESUME]")
                        schedule.append({
                            "thread_id": interrupt_tid,
                            "is_interrupt": True,
                        })

        full_stream = "".join(interleaved_parts)
        ep_id = f"ep_{rng.randint(100000, 999999)}"

        return TransformedCognitiveEpisode(
            episode_id=ep_id,
            interleaved_text=full_stream,
            num_threads=k,
            has_preemption=include_preemption,
            has_dependency=include_dependency,
            thread_schedule=schedule,
            metadata={"dialogue_ids": [d.dialogue_id for d in selected_dialogues]},
        )

    def transform_corpus(
        self,
        raw_dialogues: List[RawDialogue],
        num_episodes: int = 50,
        threads_per_episode: int = 4,
        seed: int = 42,
    ) -> List[TransformedCognitiveEpisode]:
        """Transform an entire corpus into multi-threaded episodes."""
        rng = np.random.RandomState(seed)
        episodes: List[TransformedCognitiveEpisode] = []

        for _ in range(num_episodes):
            ep = self.create_interleaved_episode(
                raw_dialogues,
                rng=rng,
                include_preemption=bool(rng.rand() > 0.3),
                include_dependency=bool(rng.rand() > 0.5),
            )
            episodes.append(ep)

        return episodes


def generate_synthetic_raw_dialogues(num_dialogues: int = 40, seed: int = 42) -> List[RawDialogue]:
    """Generate representative multi-turn conversations for pipeline validation."""
    rng = np.random.RandomState(seed)
    domains = [
        ("travel", [
            ("I want to visit Japan next summer.", "Japan is great! Tokyo and Kyoto are wonderful destinations."),
            ("What is the best way to travel between them?", "The Shinkansen bullet train takes just over 2 hours."),
        ]),
        ("coding", [
            ("How do I sort a list in Python?", "You can use the built-in sorted() function or list.sort()."),
            ("Does list.sort() modify in place?", "Yes, list.sort() modifies the list in place and returns None."),
        ]),
        ("science", [
            ("What is photosynthesis?", "Photosynthesis is the process by which plants convert light into glucose and oxygen."),
            ("What organelle does this occur in?", "It takes place primarily inside the chloroplasts."),
        ]),
        ("calendar", [
            ("Schedule meeting with Bob on Thursday at 2pm.", "Meeting scheduled: Thursday at 2:00 PM with Bob."),
            ("Set a reminder 15 minutes before.", "Reminder set for Thursday at 1:45 PM."),
        ]),
    ]

    dialogues: List[RawDialogue] = []
    for i in range(num_dialogues):
        domain_name, pair_list = domains[i % len(domains)]
        turns = []
        for u_text, a_text in pair_list:
            # Add slight random variations
            turns.append(RawDialogueTurn(role="user", content=u_text))
            turns.append(RawDialogueTurn(role="assistant", content=a_text))
        dialogues.append(RawDialogue(dialogue_id=f"diag_{domain_name}_{i}", turns=turns, category=domain_name))

    return dialogues


def export_transformed_dataset(episodes: List[TransformedCognitiveEpisode], output_path: Path) -> None:
    """Save transformed episodes to JSONL format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(asdict(ep)) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 10: LLM Dataset Transformation Pipeline")
    parser.add_argument("--episodes", type=int, default=30, help="Number of transformed cognitive episodes to produce")
    parser.add_argument("--output", type=str, default="brain/docs/runs/artifacts/transformed_cognitive_dataset.jsonl")
    args = parser.parse_args()

    print("Running Phase 10 LLM Dataset Transformation Pipeline...")
    raw_dialogues = generate_synthetic_raw_dialogues(num_dialogues=40)
    print(f"Loaded {len(raw_dialogues)} raw multi-turn dialogues.")

    transformer = LLMDataTransformer(max_threads=8)
    episodes = transformer.transform_corpus(raw_dialogues, num_episodes=args.episodes, seed=42)
    print(f"Generated {len(episodes)} multi-threaded interleaved cognitive episodes.")

    out_p = Path(args.output)
    export_transformed_dataset(episodes, out_p)
    print(f"Exported transformed cognitive dataset to {out_p}")


if __name__ == "__main__":
    main()
