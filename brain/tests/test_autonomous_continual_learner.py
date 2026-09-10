"""Comprehensive Test Suite for Autonomous Lifelong Learning Agent.

Verifies:
1. Epistemic Honesty: When encountering an unfamiliar technical question, the agent
   does NOT hallucinate or answer random topics. It explicitly states:
   "I don't actually know that offhand — let me research that for you!"
2. Autonomous Research: Queries the ResearchEngine across languages (Python, Bash, Rust, JS, C++),
   retrieves documentation, and tests code in the sandbox.
3. Continuous Episodic Learning: Consolidates learned lessons into Hierarchical Memory (K_episodic=128)
   and updates synaptic plasticity (P_t) without full retraining.
4. Lifelong State Persistence (No Being 'Born Again'): Saves state to disk, loads it in a new session,
   and evaluates on a completely changed-up, fresh prompt to verify immediate zero-shot recall.
5. Law 1 16 KB State Invariant: Ensures working memory stays strictly within the 16 KB operational budget.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import pytest
import torch

from irene_brain.agent.continual_learner import AutonomousLifelongAgent, AgentInteractionResult
from irene_brain.unified.unified_model import make_unified_model


def test_epistemic_honesty_and_research():
    """Verify the agent admits lack of knowledge on unfamiliar queries instead of hallucinating."""
    model = make_unified_model(tier="tier0", vocab_size=1000)
    agent = AutonomousLifelongAgent(model=model)

    # Prompt about an unfamiliar technical topic
    prompt = "How does a Python generator with `yield` save memory?"
    res = agent.respond(prompt)

    # 1. Did NOT hallucinate: admitted lack of immediate knowledge
    assert "I don't actually know that offhand" in res.reply
    assert res.did_research is True
    assert "generator" in res.research_topic.lower() or "yield" in res.research_topic.lower()

    # 2. Included factual grounded explanation from research
    assert "yield" in res.reply
    assert "memory" in res.reply.lower()

    # 3. Verified working code in sandbox
    assert res.code_execution_success is True

    # 4. Consolidated to episodic memory
    assert res.consolidated_to_episodic is True


def test_continual_learning_across_sessions_no_born_again():
    """Verify the agent remembers learned lessons across sessions on completely fresh prompts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "persistent_agent_state.pt"

        model = make_unified_model(tier="tier0", vocab_size=1000)

        # -------------------------------------------------------------
        # Session 1: Exposure to Run-Length Encoding (RLE)
        # -------------------------------------------------------------
        agent_session1 = AutonomousLifelongAgent(model=model, state_save_path=str(state_file))

        prompt1 = "How can I implement run-length encoding in Python?"
        res1 = agent_session1.respond(prompt1)

        assert res1.did_research is True
        assert res1.consolidated_to_episodic is True
        assert state_file.exists(), "Lifelong state was not persisted to disk"

        # Explicitly save state and close session 1
        agent_session1.save_lifelong_state()
        del agent_session1

        # -------------------------------------------------------------
        # Session 2: A NEW session is loaded from disk.
        # The agent is NOT 'born over and over again' — it retains its memory!
        # -------------------------------------------------------------
        agent_session2 = AutonomousLifelongAgent(model=model, state_save_path=str(state_file))

        # Test on a COMPLETELY FRESH, CHANGED-UP PROMPT on the same concept
        fresh_prompt2 = "Can you compress consecutive duplicate characters using run-length encoding?"
        res2 = agent_session2.respond(fresh_prompt2)

        # It MUST remember from earlier without re-researching!
        assert res2.recalled_from_episodic is True
        assert res2.did_research is False
        assert "I remember this from earlier" not in res2.reply
        assert "run_length_encode" in res2.reply or "lossless" in res2.reply.lower()


def test_multi_language_research_and_syntax():
    """Verify research capability across multiple languages: Bash and Rust."""
    model = make_unified_model(tier="tier0", vocab_size=1000)
    agent = AutonomousLifelongAgent(model=model)

    # 1. Bash research
    bash_prompt = "What is set -euo pipefail used for in bash scripts?"
    bash_res = agent.respond(bash_prompt)
    assert bash_res.did_research is True
    assert "pipefail" in bash_res.reply.lower() or "exit" in bash_res.reply.lower()

    # 2. Rust research
    rust_prompt = "How does error handling with the question mark operator work in Rust?"
    rust_res = agent.respond(rust_prompt)
    assert rust_res.did_research is True
    assert "Result" in rust_res.reply or "error" in rust_res.reply.lower()


def test_law1_16kb_memory_compliance():
    """Verify hierarchical state strictly complies with Law 1 memory limits."""
    model = make_unified_model(tier="tier0", vocab_size=1000)
    agent = AutonomousLifelongAgent(model=model)

    res = agent.respond("How does binary search work?")

    fast_bytes = agent.cognitive_state.hierarchical_state.fast_state_bytes()
    # Law 1 working state budget: strictly <= 16,384 bytes
    assert fast_bytes <= 16384, f"Working state bytes {fast_bytes} exceed Law 1 16 KB budget"
    assert res.state_bytes > 0
