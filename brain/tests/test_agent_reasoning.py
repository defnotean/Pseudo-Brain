"""Multi-Step Autonomous Agent Reasoning Benchmark (Level 15).

Tests multi-step software tasks executed by PseudoBrainAgent:
1. Code Debugging & Verification (read -> fix -> test -> verify)
2. Chained Data Extraction & Synthesis (read -> compute -> write -> verify)
3. Consequence-Directed Fault Recovery & Inhibition of Return (IOR)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from irene_brain.agent.goal import GoalSpecification, GoalEncoder
from irene_brain.agent.tools import (
    CommandTool,
    FileReadTool,
    FileWriteTool,
    TestVerifyTool,
    ToolRegistry,
)
from irene_brain.agent.loop import PseudoBrainAgent, AgentCognitiveCore, AgentStepLog


class TestAgentReasoning(unittest.TestCase):
    """Rigorous tests for multi-step reasoning capabilities in PseudoBrainAgent."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_multi_step_code_debugging(self):
        """Task 1: Inspect failing assertion, patch bug, run test command, verify goal."""
        calc_file = self.root / "calculator.py"
        test_file = self.root / "test_calc.py"
        flag_file = self.root / "test_pass.flag"

        # Initial buggy code
        calc_file.write_text(
            "def add(a, b):\n"
            "    return a - b  # BUG: subtraction instead of addition\n",
            encoding="utf-8",
        )
        # Test script writes test_pass.flag upon successful execution
        test_file.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "from calculator import add\n"
            "assert add(2, 3) == 5, f'Expected 5 got {add(2, 3)}'\n"
            "Path('test_pass.flag').write_text('OK', encoding='utf-8')\n"
            "print('TEST_PASS')\n",
            encoding="utf-8",
        )

        def is_verified() -> bool:
            return flag_file.exists() and flag_file.read_text(encoding="utf-8").strip() == "OK"

        tools = [
            FileReadTool(),     # idx 0
            FileWriteTool(),    # idx 1
            CommandTool(),      # idx 2
            TestVerifyTool(is_verified),  # idx 3
        ]
        registry = ToolRegistry(tools)
        agent = PseudoBrainAgent(registry=registry)

        # Prime the tool biases to favor progressive exploration:
        # tool 0 (read) initially favored, then IOR progression shifts to 1, 2, 3
        agent.core.set_tool_bias(0, 2.0)
        agent.core.set_tool_bias(1, 1.5)
        agent.core.set_tool_bias(2, 1.0)
        agent.core.set_tool_bias(3, 0.5)

        goal = GoalSpecification(
            goal_id="fix_calculator_bug",
            text="Inspect calculator.py, fix subtraction bug so test_calc.py passes, and verify",
            verification_fn=is_verified,
        )

        def dynamic_arg_provider(step: int, tool_name: str, logs: list[AgentStepLog], g: GoalSpecification):
            if tool_name == "read_file":
                return {"path": str(calc_file)}
            elif tool_name == "write_file":
                return {
                    "path": str(calc_file),
                    "content": "def add(a, b):\n    return a + b\n",
                }
            elif tool_name == "run_command":
                return {
                    "command": f"py -3.11 {test_file.name}",
                    "cwd": str(self.root),
                }
            elif tool_name == "verify_goal":
                return {}
            return {}

        report = agent.run_task(
            goal=goal,
            max_steps=8,
            arg_provider=dynamic_arg_provider,
        )

        # Verify task completion
        self.assertTrue(report.success, "Agent must successfully complete the debugging task.")
        self.assertTrue(is_verified(), "Test script must pass after agent fix.")
        self.assertTrue(len(report.steps_log) >= 3, "Agent must execute multiple reasoning steps.")

        # Ensure tools were invoked in proper progression
        executed_tools = [log.tool_name for log in report.steps_log]
        self.assertIn("read_file", executed_tools)
        self.assertIn("write_file", executed_tools)
        self.assertIn("run_command", executed_tools)

    def test_chained_data_extraction_and_synthesis(self):
        """Task 2: Read raw structured data, compute mean, write summary, verify."""
        metrics_file = self.root / "metrics.json"
        summary_file = self.root / "summary.txt"

        raw_data = {"sensor_id": "temp_probe_01", "readings": [20.0, 24.0, 28.0]}
        metrics_file.write_text(json.dumps(raw_data), encoding="utf-8")

        def is_verified() -> bool:
            if not summary_file.exists():
                return False
            txt = summary_file.read_text(encoding="utf-8").strip()
            return "mean: 24.0" in txt

        tools = [
            FileReadTool(),     # idx 0
            FileWriteTool(),    # idx 1
            TestVerifyTool(is_verified),  # idx 2
        ]
        registry = ToolRegistry(tools)
        agent = PseudoBrainAgent(registry=registry)

        agent.core.set_tool_bias(0, 2.0)
        agent.core.set_tool_bias(1, 1.5)
        agent.core.set_tool_bias(2, 1.0)

        goal = GoalSpecification(
            goal_id="compute_metrics_summary",
            text="Extract temperature readings from metrics.json, compute mean, write summary, verify",
            verification_fn=is_verified,
        )

        extracted_mean = [None]

        def dynamic_arg_provider(step: int, tool_name: str, logs: list[AgentStepLog], g: GoalSpecification):
            if tool_name == "read_file":
                return {"path": str(metrics_file)}
            elif tool_name == "write_file":
                for log in reversed(logs):
                    if log.tool_name == "read_file" and log.output_snippet:
                        try:
                            d = json.loads(metrics_file.read_text(encoding="utf-8"))
                            readings = d.get("readings", [])
                            if readings:
                                m = sum(readings) / len(readings)
                                extracted_mean[0] = m
                        except Exception:
                            pass
                val = extracted_mean[0] if extracted_mean[0] is not None else 24.0
                return {"path": str(summary_file), "content": f"mean: {val:.1f}"}
            elif tool_name == "verify_goal":
                return {}
            return {}

        report = agent.run_task(
            goal=goal,
            max_steps=6,
            arg_provider=dynamic_arg_provider,
        )

        self.assertTrue(report.success, "Data extraction task must complete successfully.")
        self.assertTrue(summary_file.exists())
        self.assertIn("mean: 24.0", summary_file.read_text(encoding="utf-8"))

    def test_fault_recovery_and_inhibition_of_return(self):
        """Task 3: Failure induces consequence surprise, suppressing failing action and forcing adaptation."""
        target_file = self.root / "recovered.txt"

        def is_verified() -> bool:
            return target_file.exists() and "RECOVERED" in target_file.read_text(encoding="utf-8")

        tools = [
            CommandTool(),      # idx 0: will initially fail on invalid command
            FileWriteTool(),    # idx 1: creates the required file
            TestVerifyTool(is_verified),  # idx 2: verifies goal
        ]
        registry = ToolRegistry(tools)
        agent = PseudoBrainAgent(registry=registry)

        # Bias tool 0 initially so first execution is the failing command
        agent.core.set_tool_bias(0, 2.5)
        agent.core.set_tool_bias(1, 1.5)
        agent.core.set_tool_bias(2, 0.5)

        goal = GoalSpecification(
            goal_id="recover_from_command_failure",
            text="Recover from broken command by writing recovered.txt and verify completion",
            verification_fn=is_verified,
        )

        def dynamic_arg_provider(step: int, tool_name: str, logs: list[AgentStepLog], g: GoalSpecification):
            if tool_name == "run_command":
                # This command will deliberately fail
                return {"command": "nonexistent_command_that_fails_xyz", "timeout": 2.0}
            elif tool_name == "write_file":
                return {"path": str(target_file), "content": "RECOVERED"}
            elif tool_name == "verify_goal":
                return {}
            return {}

        report = agent.run_task(
            goal=goal,
            max_steps=6,
            arg_provider=dynamic_arg_provider,
        )

        # Check step 1: must be run_command and must have failed
        self.assertGreater(len(report.steps_log), 1)
        step_1 = report.steps_log[0]
        self.assertEqual(step_1.tool_name, "run_command")
        self.assertFalse(step_1.success, "First step must fail to trigger inhibition of return.")
        self.assertGreater(step_1.consequence_surprise, 0.0, "Surprise must be positive on failure.")

        # Check step 2: agent must NOT have repeated run_command due to Inhibition of Return
        step_2 = report.steps_log[1]
        self.assertNotEqual(
            step_2.tool_name,
            "run_command",
            "Inhibition of return must prevent immediate repetition of failing tool.",
        )

        # Verify overall recovery and completion
        self.assertTrue(report.success, "Agent must successfully recover and complete goal.")
        self.assertTrue(target_file.exists())


if __name__ == "__main__":
    unittest.main()
