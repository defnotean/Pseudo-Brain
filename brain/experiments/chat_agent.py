"""Interactive Console Chat for Pseudo-Brain Autonomous Lifelong Agent.

Run this script to chat directly with Pseudo-Brain!
Features:
- Epistemic Humility: When asked an unfamiliar technical concept, it says "I don't actually know that offhand — let me research that for you!"
- Autonomous Research: Looks up documentation, algorithms, and code across Python, Bash, Rust, JavaScript, and C++.
- Sandbox Execution: Executes code snippets to verify correctness.
- Continual Learning: Remembers learned concepts across turns without needing to be retrained.
- Law 1 Memory: Strict 16 KB working state memory.

Usage:
    python brain/experiments/chat_agent.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add brain/src to path
repo_src = Path(__file__).resolve().parents[1] / "src"
if str(repo_src) not in sys.path:
    sys.path.insert(0, str(repo_src))

import torch
from irene_brain.agent.continual_learner import AutonomousLifelongAgent
from irene_brain.unified.unified_model import make_unified_model


def main():
    print("=" * 80)
    print("  PSEUDO-BRAIN 1.034B: AUTONOMOUS LIFELONG AGENT (INTERACTIVE CONSOLE)")
    print("  - Epistemic Honesty: Never hallucinates; researches unknown concepts")
    print("  - Sandbox Execution: Validates code in real time")
    print("  - Continual Learning: Remembers lessons across turns without retraining")
    print("  - Law 1 Invariant: Strict 16 KB working state memory")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Engine] Initializing on {device}...")

    state_path = Path.home() / ".pseudo_brain" / "lifelong_agent_state.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    # Use tier2 on-device for responsive sub-50ms local execution
    model = make_unified_model(tier="tier2", vocab_size=32000)
    agent = AutonomousLifelongAgent(
        model=model,
        tier="tier2",
        vocab_size=32000,
        device=device,
        state_save_path=str(state_path),
    )

    print(f"[Engine] Agent ready! Memory state saved at: {state_path}")
    print("[Engine] Type 'exit' or 'quit' to exit, or 'clear' to reset memory.\n")

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            print("Saving lifelong state... Goodbye!")
            agent.save_lifelong_state()
            break
        if prompt.lower() == "clear":
            if state_path.exists():
                state_path.unlink()
            agent = AutonomousLifelongAgent(
                model=model,
                tier="tier2",
                vocab_size=32000,
                device=device,
                state_save_path=str(state_path),
            )
            print("[Engine] Memory cleared! Agent reset to clean state.")
            continue

        res = agent.respond(prompt)
        print(f"\nPseudo-Brain:\n{res.reply}\n")
        print(f"--- Telemetry: {res.elapsed_ms:.1f}ms | Researched: {res.did_research} | Recalled From Memory: {res.recalled_from_episodic} | Working State: {res.state_bytes:,} bytes (Law 1) ---")


if __name__ == "__main__":
    main()
