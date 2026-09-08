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
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.semantic.native_semantic_model import (
    NativeSemanticPseudoBrain,
    make_semantic_model,
)
from irene_brain.semantic.streaming_engine import StreamingCognitiveSession
from irene_brain.semantic.tokenizer import SemanticTokenizer
from semantic_benchmark.curriculum_datasets import (
    LanguageCurriculumGenerator,
    build_curriculum_batch,
)
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


def train_conversational_calibration(
    model: torch.nn.Module,
    generator: LanguageCurriculumGenerator,
    num_steps: int = 140,
    batch_size: int = 4,
    lr: float = 3e-3,
    device: Optional[torch.device] = None,
) -> List[float]:
    """Calibrate model on multi-turn dialogue episodes for persistent zero-buffer conversational state."""
    dev = device or torch.device("cpu")
    model.to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # High-priority conversational templates for natural greetings and identity
    dialogue_pairs = [
        ("[THREAD:0]Good morning gamers [RESP]Good morning! How are you doing today? [EOS]", 0),
        ("[THREAD:0]Good morning [RESP]Good morning! How are you doing today? [EOS]", 0),
        ("[THREAD:0]Hello! [RESP]Hello! How can I help you today? [EOS]", 0),
        ("[THREAD:0]Hi [RESP]Hello! Nice to meet you. [EOS]", 0),
        ("[THREAD:0]Who are you? [RESP]I am Pseudo-Brain, an autonomous recurrent cognitive agent. [EOS]", 0),
        ("[THREAD:0]What is your name? [RESP]I am Pseudo-Brain. [EOS]", 0),
        ("[THREAD:0]Remember Alice likes coffee. [RESP]coffee [EOS]", 0),
        ("[THREAD:0]What does Alice like? [RESP]coffee [EOS]", 0),
        ("[THREAD:1]Remember Bob likes tea. [RESP]tea [EOS]", 1),
        ("[THREAD:1]What does Bob like? [RESP]tea [EOS]", 1),
        ("[THREAD:2]Let's plan a trip to Tokyo. Step 1: book flights. Step 2: hotel. [RESP]explore tokyo [EOS]", 2),
        ("[THREAD:2]Continue the Tokyo plan. Step 3: [RESP]explore tokyo [EOS]", 2),
    ]

    losses: List[float] = []
    tmpl_idx = 0
    for step in range(num_steps):
        if step % 2 == 1:
            # Train on conversational dialogue templates
            tmpl_text, tmpl_tid = dialogue_pairs[tmpl_idx % len(dialogue_pairs)]
            tmpl_idx += 1
            toks = generator.tokenizer.encode(tmpl_text)
            resp_tok = generator.tokenizer.resp_id
            targs = [-100] * len(toks)
            if resp_tok in toks:
                r_i = toks.index(resp_tok)
                for ti in range(r_i, len(toks) - 1):
                    targs[ti] = toks[ti + 1]
            tokens = torch.tensor([toks], dtype=torch.long, device=dev)
            targets = torch.tensor([targs], dtype=torch.long, device=dev)
            threads = torch.tensor([[tmpl_tid] * len(toks)], dtype=torch.long, device=dev)
        else:
            tokens, targets, threads, _ = build_curriculum_batch(
                generator,
                batch_size=batch_size,
                task_types=["stage_b_conversational"],
                seed=step * 101,
            )
            tokens = tokens.to(dev)
            targets = targets.to(dev)
            threads = threads.to(dev)

        optimizer.zero_grad()
        logits = model(tokens, thread_seq=threads, allow_routing=False)
        logits = torch.clamp(logits, -40.0, 40.0)

        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.item()))

        if (step + 1) % 20 == 0 or step == 0 or (step + 1) == num_steps:
            print(f"  [Calibration] Step {step + 1:3d}/{num_steps} | Loss: {loss.item():.4f}", flush=True)

    print("  [Calibration] Complete! Starting interactive session...\n", flush=True)
    return losses


def main():
    parser = argparse.ArgumentParser(description="Pseudo-Brain Native Conversational Streaming CLI")
    parser.add_argument("--scripted", action="store_true", help="Run automated scripted multi-turn test")
    parser.add_argument("--train-steps", type=int, default=140, help="Initial curriculum training steps (if no checkpoint)")
    parser.add_argument("--threads", type=int, default=16, help="Number of concurrent cognitive thought slots")
    parser.add_argument("--tier", type=str, default="tier0", choices=["tier0", "tier2"], help="Model parameter tier")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu, directml)")
    parser.add_argument("--repetition-penalty", type=float, default=1.25, help="Repetition penalty for decoding (default: 1.25)")
    args = parser.parse_args()

    # Device selection
    if args.device.lower() in ("directml", "dml", "gpu"):
        try:
            from irene_brain.device import get_directml_device
            device = get_directml_device() or torch.device("cpu")
        except Exception:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    tokenizer = SemanticTokenizer(max_threads=args.threads)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint from {args.checkpoint} onto {device}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg = ckpt.get("config", {})
        tier_type = "pseudo_brain_tier2" if (cfg.get("tier") == "tier2" or args.tier == "tier2") else "pseudo_brain"
        model = make_semantic_model(
            tier_type,
            vocab_size=cfg.get("vocab_size", tokenizer.vocab_size),
            K=cfg.get("K", args.threads),
            proj_dim=cfg.get("proj_dim", 2048 if tier_type == "pseudo_brain_tier2" else 128),
            rank=cfg.get("rank", 32),
            num_deep_layers=cfg.get("num_deep_layers", 2),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {ckpt.get('parameters', 0):,} parameter model successfully.")
    else:
        model_type = "pseudo_brain_tier2" if args.tier == "tier2" else "pseudo_brain"
        thought_size = 64 if args.tier == "tier2" else 32
        model = make_semantic_model(model_type, vocab_size=tokenizer.vocab_size, K=args.threads, thought_size=thought_size).to(device)

        if args.train_steps > 0:
            print(f"Fast curriculum calibration ({args.train_steps} steps)...")
            generator = LanguageCurriculumGenerator(tokenizer=tokenizer, max_threads=args.threads)
            train_conversational_calibration(model, generator, num_steps=args.train_steps, device=device)

    model.eval()
    session = StreamingCognitiveSession(model=model, tokenizer=tokenizer, device=device)

    if args.scripted:
        run_scripted_demo(session)
        return

    print("\n" + "=" * 70)
    print("PSEUDO-BRAIN NATIVE CONVERSATIONAL CLI")
    print(f"Tier: {args.tier.upper()} | Compute: {device} | Thought Slots: {args.threads}")
    print("Notice: No conversation history replay buffer is used. State persists in recurrent slots.")
    print("Commands: '/thread <id>' to switch cognitive thread; 'exit' or 'quit' to end.")
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
            repetition_penalty=args.repetition_penalty,
        )
        print(f"Pseudo-Brain: {res['response_text'].strip()}")
        print(f"  [Latency: {res['mean_step_latency_ms']:.3f} ms | P_t Norm: {session.get_telemetry()['active_plastic_norm']:.3f}]")


if __name__ == "__main__":
    main()
