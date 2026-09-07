"""Interactive CLI Demonstration with Live Cognitive Dashboard."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
from irene_brain.semantic.tokenizer import SemanticTokenizer
from irene_brain.semantic.native_semantic_model import NativeSemanticPseudoBrain
from irene_brain.semantic.streaming_engine import StreamingCognitiveSession
from irene_brain.semantic.dashboard import CognitiveDashboard


def run_demo():
    print("=" * 70)
    print("PSEUDO-BRAIN STREAMING COGNITIVE DEMO WITH LIVE DASHBOARD")
    print("=" * 70)

    tokenizer = SemanticTokenizer(max_threads=8)
    model = NativeSemanticPseudoBrain(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        K=8,
        thought_size=48,
        proj_dim=256,
    )
    session = StreamingCognitiveSession(model=model, tokenizer=tokenizer)
    dashboard = CognitiveDashboard()

    print("\n[1. Ingesting Thread 0 Conversation: 'The red key is in room 4']")
    session.ingest_text("The red key is in room 4", thread_id=0)
    print(dashboard.render_full_dashboard(session))

    print("\n[2. Preempting: Interleaving Thread 1: 'The blue door is locked']")
    session.ingest_text("The blue door is locked", thread_id=1)
    print(dashboard.render_full_dashboard(session))

    print("\n[3. Resuming Thread 0 and Querying: 'Where is the red key?']")
    res = session.generate_response("Where is the red key?", thread_id=0, max_new_tokens=16)
    print(f"Generated tokens: {res['generated_tokens']}")
    print(f"Mean step latency: {res['mean_step_latency_ms']:.2f} ms")
    print(dashboard.render_full_dashboard(session))


if __name__ == "__main__":
    run_demo()
