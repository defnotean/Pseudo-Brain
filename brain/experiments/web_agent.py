"""Interactive Gradio Web UI for Pseudo-Brain Autonomous Lifelong Agent.

Launch via:
    python brain/experiments/web_agent.py
Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_src = Path(__file__).resolve().parents[1] / "src"
if str(repo_src) not in sys.path:
    sys.path.insert(0, str(repo_src))

import torch
import gradio as gr
from irene_brain.agent.continual_learner import AutonomousLifelongAgent
from irene_brain.unified.unified_model import make_unified_model


def create_app():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_path = Path.home() / ".pseudo_brain" / "lifelong_agent_state.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    model = make_unified_model(tier="tier2", vocab_size=32000)
    agent = AutonomousLifelongAgent(
        model=model,
        tier="tier2",
        vocab_size=32000,
        device=device,
        state_save_path=str(state_path),
    )

    def chat_fn(message: str, history: list):
        if not message.strip():
            return "Please enter a question or coding prompt."
        res = agent.respond(message)
        telemetry = (
            f"\n\n---\n"
            f"*Telemetry: Latency={res.elapsed_ms:.1f}ms | "
            f"Researched={res.did_research} | "
            f"Recalled from Memory={res.recalled_from_episodic} | "
            f"Working State={res.state_bytes:,} bytes (Law 1)*"
        )
        return res.reply + telemetry

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="Pseudo-Brain: Autonomous Lifelong Coding & Research Agent",
        description=(
            "An authentic autonomous agent powered by Pseudo-Brain with Law 1 16 KB strict state memory.\n"
            "- **Epistemic Humility**: When unfamiliar with a concept, it says *'I don't know it actually, let me research that'*.\n"
            "- **Autonomous Research**: Searches documentation and minimal examples across Python, Bash, Rust, JS, and C++.\n"
            "- **Continual Learning**: Consolidates lessons into persistent episodic memory without ever needing to be 'born over and over again'."
        ),
        examples=[
            "How does a Python generator with `yield` save memory compared to returning a list?",
            "Can you show me how to stream numbers lazily using yield?",
            "How can I write a robust bash script that fails fast on pipeline errors?",
            "How does error handling with the question mark operator work in Rust?",
            "Can you explain what an algorithm is in simple terms, like you're talking to a friend?",
        ],
    )
    return demo


if __name__ == "__main__":
    app = create_app()
    print("\nLaunching Pseudo-Brain Web UI at http://127.0.0.1:7860 ...")
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
