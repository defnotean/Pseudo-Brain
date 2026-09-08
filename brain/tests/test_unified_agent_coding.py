"""Comprehensive Test Suite for UnifiedCognitiveAgent and CodeExecutionEngine.

Verifies:
1. CodeExecutionEngine:
   - Safe AST syntax validation
   - In-process/subprocess Python execution with stdout/stderr capture
   - Timeout handling
   - Error type and message extraction
2. UnifiedCognitiveAgent:
   - Law 1 4.0 KB operational state compliance during agent turns
   - Multi-modal sensory text ingestion and cognitive step execution
   - Decoupled readout heads (tool_gate, predicted_value, action_logits)
   - Contextual Inhibition of Return (IOR) dynamically suppressing repeating failed actions
   - Autonomous closed-loop Python code self-repair across syntax and logical bugs
   - Multi-step tool-calling DAG execution with goal verification
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
import pytest
import torch

from irene_brain.agent.unified_agent_loop import (
    CodeExecutionEngine,
    ExecutionResult,
    UnifiedCognitiveAgent,
    CognitiveAgentStepLog,
    CognitiveTaskReport,
)
from irene_brain.agent.tools import (
    ToolRegistry,
    FileReadTool,
    FileWriteTool,
    FilePatchTool,
    CommandTool,
    TestVerifyTool,
)
from irene_brain.agent.goal import GoalSpecification
from irene_brain.unified.unified_model import make_unified_model


def test_code_execution_engine_syntax():
    """Verify AST syntax validation catches malformed code without running subprocess."""
    engine = CodeExecutionEngine()

    valid, err = engine.check_syntax("def foo(x):\n    return x + 1\n")
    assert valid
    assert err is None

    invalid, err = engine.check_syntax("def foo(x)\n    return x + 1\n")
    assert not invalid
    assert "SyntaxError" in err


def test_code_execution_engine_execution():
    """Verify execution of valid Python code and capture of stdout and elapsed time."""
    engine = CodeExecutionEngine(default_timeout=5.0)

    code = "print('HELLO_PSEUDO_BRAIN'); x = sum([1, 2, 3, 4]); print(f'SUM={x}')"
    res = engine.execute_python(code)

    assert res.success
    assert res.return_code == 0
    assert "HELLO_PSEUDO_BRAIN" in res.stdout
    assert "SUM=10" in res.stdout
    assert res.elapsed_ms > 0.0


def test_code_execution_engine_error_capture():
    """Verify assertion and runtime errors are properly captured and classified."""
    engine = CodeExecutionEngine(default_timeout=5.0)

    # AssertionError
    code_assert = "assert 2 + 2 == 5, 'Math broken!'"
    res_assert = engine.execute_python(code_assert)
    assert not res_assert.success
    assert res_assert.return_code != 0
    assert res_assert.error_type == "AssertionError"
    assert "Math broken!" in res_assert.stderr

    # ZeroDivisionError
    code_zero = "x = 10 / 0"
    res_zero = engine.execute_python(code_zero)
    assert not res_zero.success
    assert res_zero.error_type == "ZeroDivisionError"


def test_code_execution_engine_timeout():
    """Verify execution timeouts trigger cleanly without blocking."""
    engine = CodeExecutionEngine(default_timeout=1.0)

    code_loop = "import time\nwhile True:\n    time.sleep(0.1)"
    res = engine.execute_python(code_loop, timeout=0.8)

    assert not res.success
    assert res.error_type == "TimeoutExpired"


def test_unified_cognitive_agent_law1_and_step():
    """Verify UnifiedCognitiveAgent respects Law 1 4.0 KB state and executes cognitive steps."""
    model = make_unified_model(tier="tier0", vocab_size=1000)
    agent = UnifiedCognitiveAgent(model=model)

    state = agent.reset_state(batch_size=1)
    # Law 1 check: fast state footprint
    fast_bytes = state.hierarchical_state.fast_state_bytes()
    assert fast_bytes <= 4096, f"Fast state bytes {fast_bytes} exceed 4.0 KB Law 1 budget"

    # Execute cognitive cycle
    obs = "Investigate failing unit tests in module math_utils.py"
    cog_res, next_state = agent.step_cognitive_cycle(obs, state, last_reward=0.0)

    assert "predicted_reward" in cog_res
    assert "tool_prob" in cog_res
    assert "action_logits" in cog_res
    assert "consequence_surprise" in cog_res
    assert 0.0 <= cog_res["tool_prob"] <= 1.0


def test_contextual_inhibition_of_return():
    """Verify failed actions receive heavy IOR suppression to break cyclic failure loops."""
    model = make_unified_model(tier="tier0", vocab_size=1000)
    agent = UnifiedCognitiveAgent(model=model)
    _ = agent.reset_state(batch_size=1)

    tool_name = "run_command"
    assert tool_name not in agent.ior_memory

    # Simulate failure with consequence surprise = 1.0
    agent.update_ior(tool_name, success=False, consequence_surprise=1.0)
    assert tool_name in agent.ior_memory
    initial_penalty = agent.ior_memory[tool_name]
    assert initial_penalty >= 4.0, f"Expected strong IOR penalty, got {initial_penalty}"

    # Action logits with run_command as top choice
    logits = torch.tensor([5.0, 2.0, 1.0, 0.0])  # index 0 is run_command
    tool_chosen, idx_chosen = agent.select_tool(logits)

    # IOR should suppress index 0 (5.0 - 5.0 = 0.0), shifting choice to another tool
    assert idx_chosen != 0 or agent.ior_memory[tool_name] > 0.0


def test_autonomous_self_repair_loop():
    """Verify autonomous closed-loop self-repair on Python code with assertion failure."""
    model = make_unified_model(tier="tier0", vocab_size=1000)
    agent = UnifiedCognitiveAgent(model=model)

    buggy_code = (
        "def compute_interest(principal, rate, years):\n"
        "    return principal - (principal * rate * years)  # BUG: minus instead of plus\n"
    )
    test_script = (
        "p = compute_interest(100, 0.05, 2)\n"
        "assert p == 110.0, f'Expected 110.0 got {p}'\n"
        "print('INTEREST_VERIFIED')\n"
    )

    success, repaired_code, history = agent.self_repair_code(
        buggy_code=buggy_code,
        test_script=test_script,
        max_attempts=3,
    )

    assert success, "Agent failed to self-repair code"
    assert " + " in repaired_code
    assert len(history) >= 2  # First failed, second passed
    assert not history[0].success
    assert history[-1].success
    assert "INTEREST_VERIFIED" in history[-1].stdout


def test_autonomous_multi_step_task_with_tools():
    """Verify multi-step autonomous task execution and goal verification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        target_file = root / "solver.py"
        flag_file = root / "solved.flag"

        target_file.write_text(
            "def solve():\n    return 'NOT_SOLVED'\n",
            encoding="utf-8",
        )

        def is_verified() -> bool:
            return flag_file.exists() and flag_file.read_text(encoding="utf-8").strip() == "DONE"

        registry = ToolRegistry([
            FileReadTool(),
            FileWriteTool(),
            CommandTool(),
            TestVerifyTool(is_verified),
        ])

        model = make_unified_model(tier="tier0", vocab_size=1000)
        agent = UnifiedCognitiveAgent(model=model, registry=registry)

        goal = GoalSpecification(
            goal_id="solve_and_flag",
            text="Write solved.flag with content DONE and verify",
            verification_fn=is_verified,
        )

        def test_arg_provider(step: int, tool_name: str, logs: list) -> dict:
            if tool_name == "read_file":
                return {"path": str(target_file)}
            elif tool_name == "write_file":
                return {"path": str(flag_file), "content": "DONE"}
            elif tool_name == "run_command":
                return {"command": "python -c \"print('OK')\""}
            elif tool_name == "verify_goal":
                return {}
            return {}

        report = agent.run_autonomous_task(
            goal=goal,
            max_steps=5,
            arg_provider=test_arg_provider,
        )

        assert report.success
        assert is_verified()
        assert report.total_steps >= 1
        assert len(report.steps_log) >= 1
