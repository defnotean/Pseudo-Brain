"""Tests for Autonomous Code Synthesis and Autonomous Self-Repair in Pseudo-Brain."""

import tempfile
from pathlib import Path
import pytest

from irene_brain.agent.code_synthesizer import NeuralProgramSynthesizer
from irene_brain.agent.unified_agent_loop import CodeExecutionEngine, ExecutionResult


def test_ascii_pong_synthesis_and_sandbox_execution():
    """Verify that the agent synthesizes and sandbox-executes ASCII Pong cleanly."""
    synth = NeuralProgramSynthesizer()
    res = synth.synthesize_program("ASCII Ping Pong Game", "make a basic ping pong game using ASCII")

    assert res.success is True
    assert Path(res.filepath).exists()
    assert "SANDBOX TEST PASSED" in res.test_output
    assert "AsciiPong" in res.code
    assert "Player 1" in res.test_output


def test_autonomous_self_repair_syntax_error():
    """Verify that when a syntax error occurs, the agent researches and self-repairs without external help."""
    synth = NeuralProgramSynthesizer()

    buggy_code = """def multiply(a, b)
    return a * b
"""
    test_script = "assert multiply(3, 4) == 12"

    # Run initial buggy code
    initial_res = synth.code_engine.execute_python(buggy_code + "\n" + test_script)
    assert initial_res.success is False
    assert "SyntaxError" in (initial_res.error_type or "")

    # Autonomous self-repair
    repaired_ok, fixed_code, log = synth.autonomous_self_repair(
        code=buggy_code,
        test_script=test_script,
        error_res=initial_res,
        max_attempts=3,
    )

    assert repaired_ok is True
    assert "multiply(a, b):" in fixed_code
    assert "SUCCESS: Autonomous self-repair resolved the issue" in log


def test_autonomous_self_repair_name_error():
    """Verify that when a missing module NameError occurs, the agent self-repairs autonomously."""
    synth = NeuralProgramSynthesizer()

    buggy_code = """def pick_choice():
    return random.choice([10, 20])
"""
    test_script = "assert pick_choice() in [10, 20]"

    initial_res = synth.code_engine.execute_python(buggy_code + "\n" + test_script)
    assert initial_res.success is False

    repaired_ok, fixed_code, log = synth.autonomous_self_repair(
        code=buggy_code,
        test_script=test_script,
        error_res=initial_res,
        max_attempts=3,
    )

    assert repaired_ok is True
    assert "import random" in fixed_code
    assert "SUCCESS: Autonomous self-repair resolved the issue" in log
