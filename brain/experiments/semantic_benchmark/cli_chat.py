"""Native Conversational Streaming CLI for Pseudo-Brain.

Demonstrates continuous streaming conversation directly driven by Pseudo-Brain's
persistent cognitive threads, without appending previous conversation turns to an input buffer.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.semantic.native_semantic_model import (
    NativeSemanticPseudoBrain,
    make_semantic_model,
)
from irene_brain.semantic.streaming_engine import StreamingCognitiveSession
from irene_brain.semantic.tokenizer import SemanticTokenizer
from semantic_benchmark.curriculum_datasets import LanguageCurriculumGenerator
from semantic_benchmark.language_cognitive_benchmark import train_semantic_model


def run_scripted_demo(session: StreamingCognitiveSession) -> List[Dict[str, Any]]:
    """Execute scripted multi-threaded conversational dialogue verifying persistence across preemption."""
    print("\n" + "=" * 80)
    print("RUNNING SCRIPTED MULTI-THREADED CONVERSATIONAL VERIFICATION")
    print("=" * 80)

    turns = [
        # Turn 1: Enroll Fact on Thread 0
        {"thread_id": 0, "user_text": "[THREAD:0]Remember Alice likes coffee. [RESP]", "desc": "Enroll Fact: Alice -> coffee on Thread 0"},
        # Turn 2: Enroll Fact on Thread 1
        {"thread_id": 1, "user_text": "[THREAD:1]Remember Bob likes tea. [RESP]", "desc": "Enroll Fact: Bob -> tea on Thread 1"},
        # Turn 3: Query Thread 0
        {"thread_id": 0, "user_text": "[THREAD:0]What does Alice like? [RESP]", "desc": "Query Thread 0: Alice -> coffee"},
        # Turn 4: Preemption - Long task on Thread 2
        {"thread_id": 2, "user_text": "[THREAD:2]Let's plan a trip to Tokyo. Step 1: book flights. Step 2: hotel. [RESP]", "desc": "Preemption: Tokyo trip on Thread 2"},
        # Turn 5: Query Thread 1 after preemption
        {"thread_id": 1, "user_text": "[THREAD:1]What does Bob like? [RESP]", "desc": "Query Thread 1 after preemption: Bob -> tea"},
        # Turn 6: Resume Thread 2
        {"thread_id": 2, "user_text": "[THREAD:2]Continue the Tokyo plan. Step 3: [RESP]", "desc": "Resume Thread 2 after interruption"},
    ]

    history: List[Dict[str, Any]] = []
    for t_idx, turn in enumerate(turns, 1):
        print(f"\n--- Turn {t_idx} [{turn['desc']}] ---")
        print(f"You: {turn['user_text']}")

        res = session.generate_response(
            prompt_text=turn["user_text"],
            thread_id=turn["thread_id"],
            max_new_tokens=16,
            temperature=0.0,
        )

        resp_text = res["response_text"].strip()
        print(f"Pseudo-Brain: {resp_text}")
        print(f"  [Active Thread: {res['active_thread']} | Latency: {res['mean_step_latency_ms']:.3f} ms/step | Total Tokens Ingested: {res['total_tokens_session']}]")

        history.append({
            "turn": t_idx,
            "prompt": turn["user_text"],
            "response": resp_text,
            "thread_id": res["active_thread"],
            "latency_ms": res["mean_step_latency_ms"],
        })

    telemetry = session.get_telemetry()
    print("\n" + "=" * 80)
    print("SESSION COGNITIVE TELEMETRY")
    print(f"  Total Session Tokens: {telemetry['total_tokens']}")
    print(f"  Latency Mean: {telemetry['latency_mean_ms']:.3f} ms (p90: {telemetry['latency_p90_ms']:.3f} ms)")
    print(f"  Sub-16.67ms (60 Hz compliant): {telemetry['latency_under_16_67ms']}")
    print(f"  Active Slot Norm: {telemetry['active_slot_norm']:.4f}")
    print(f"  Active Plastic Latch Norm: {telemetry['active_plastic_norm']:.4f}")
    print("=" * 80)
    return history


def main():
    parser = argparse.ArgumentParser(description="Pseudo-Brain Native Conversational Streaming CLI")
    parser.add_argument("--scripted", action="store_true", help="Run automated scripted multi-turn test")
    parser.add_argument("--train-steps", type=int, default=180, help="Initial curriculum training steps")
    parser.add_argument("--threads", type=int, default=16, help="Number of concurrent cognitive thought slots")
    args = parser.parse_args()

    device = torch.device("cpu")
    tokenizer = SemanticTokenizer(max_threads=args.threads)
    model = make_semantic_model("pseudo_brain", vocab_size=tokenizer.vocab_size, K=args.threads)

    print("Initializing Pseudo-Brain Native Semantic Engine...")
    generator = LanguageCurriculumGenerator(tokenizer=tokenizer, max_threads=args.threads)

    if args.train_steps > 0:
        print(f"Fast curriculum calibration ({args.train_steps} steps)...")
        train_semantic_model(model, generator, num_steps=args.train_steps, device=device)

    session = StreamingCognitiveSession(model=model, tokenizer=tokenizer, device=device)

    if args.scripted:
        run_scripted_demo(session)
        return

    print("\n" + "=" * 70)
    print("PSEUDO-BRAIN NATIVE CONVERSATIONAL CLI")
    print(f"Running on {args.threads} Persistent Cognitive Slots. Single-Threaded CPU.")
    print("Notice: No conversation history replay buffer is used. State persists in recurrent slots.")
    print("Type 'exit' or 'quit' to end session.")
    print("=" * 70)

    cur_thread = 0
    while True:
        try:
            user_input = input(f"\nYou [Thread {cur_thread}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ["exit", "quit"]:
            break

        # Check for thread switch command e.g. /thread 2
        if user_input.startswith("/thread"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].isdigit():
                cur_thread = int(parts[1]) % args.threads
                print(f"[Switched active cognitive thread to {cur_thread}]")
                continue

        res = session.generate_response(
            prompt_text=user_input,
            thread_id=cur_thread,
            max_new_tokens=24,
            temperature=0.0,
        )
        print(f"Pseudo-Brain: {res['response_text'].strip()}")
        print(f"  [Latency: {res['mean_step_latency_ms']:.3f} ms | P_t Norm: {session.get_telemetry()['active_plastic_norm']:.3f}]")


if __name__ == "__main__":
    main()
